import pstats
import sys
sys.path.append('../code')
import argparse
import GPUtil
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
import cvxpy as cp
from PIL import Image
import math

import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils import vis_util
from model.sg_render import compute_envmap
import imageio
# import pyexr
import json

import open3d as o3d


def evaluate(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']

    expname = conf.get_string('train.expname') + '-' + kwargs['expname']

    task = kwargs["task"]
    flag = kwargs["flag"]

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))

    evaldir = os.path.join('../cvpr23/exps', task, kwargs['expname'], flag, os.path.basename(kwargs['data_split_dir']))

    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                                                           kwargs['data_split_dir'],
                                                                           train_cameras=False)

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  collate_fn=eval_dataset.collate_fn
                                                  )
    total_pixels = eval_dataset.total_pixels
    img_res = eval_dataset.img_res
    H, W = img_res

    ########################################################################################################
    ### load model

    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    if torch.cuda.is_available():
        model.cuda()

    ckpt_path = os.path.join(str(kwargs['model_params_dir']), 'model_params', "latest_model.pth")
    model_state_dict = torch.load(ckpt_path)["model_state_dict"]

    model.load_state_dict(model_state_dict)
    # print('Loaded checkpoint: ', ckpt_path)

    edited_3d_points = np.load(os.path.join(str(kwargs['model_params_dir']), 'edited_3d_points.npy'))
    edited_3d_points = torch.from_numpy(edited_3d_points).unsqueeze(0).cuda()

    # print('edited 3d points: ', edited_3d_points.shape)

    #####################################################################################################
    # reset lighting
    #####################################################################################################
    relight = False
    if kwargs['light_sg'].endswith('.npy'):
        # print('Loading light from: ', kwargs['light_sg'])

        # 环境光贴图
        envmap, _ = os.path.split(kwargs['light_sg'])
        _, envmap, = os.path.split(envmap)

        model.envmap_material_network.load_light(kwargs['light_sg'])
        relight = True


    edit_diffuse = False
    rgb_images = None

    if len(kwargs['diffuse_rgb']) > 0:

        rgb = imageio.imread(kwargs['diffuse_rgb']).astype(np.float32)[:, :, :3]
        if not kwargs['diffuse_rgb'].endswith('.exr'):
            rgb /= 255.
        rgb_images = torch.from_numpy(rgb).cuda().reshape((-1, 3))


    utils.mkdir_ifnotexists(evaldir)
    print('Output directory is: ', evaldir)

    with open(os.path.join(evaldir, 'ckpt_path.txt'), 'w') as fp:
        fp.write(ckpt_path + '\n')

    ####################################################################################################################
    print("evaluating...")
    model.eval()

    # extract mesh
    if (not edit_diffuse) and (not relight) and eval_dataset.has_groundtruth:
        with torch.no_grad():
            mesh = plt.get_surface_high_res_mesh(
                sdf=lambda x: model.implicit_network(x)[:, 0],
                resolution=kwargs['resolution']
            )

            # Taking the biggest connected component
            components = mesh.split(only_watertight=False)
            areas = np.array([c.area for c in components], dtype=np.float)
            mesh_clean = components[areas.argmax()]
            mesh_clean.export('{0}/mesh.obj'.format(evaldir), 'obj')

    # generate images
    images_dir = evaldir
    rgb_images_dir = os.path.join(evaldir, 'rgb')
    diffuse_rgb_images_dir = os.path.join(evaldir, 'diffuse_rgb')
    diffuse_albedo_images_dir = os.path.join(evaldir, 'diffuse_albedo')
    edited_pixels_dir = os.path.join(evaldir, 'edited_pixels')
    edited_pixel_coords_dir = os.path.join(evaldir, 'edited_pixels_coords')


    for dir in [rgb_images_dir, diffuse_rgb_images_dir, diffuse_albedo_images_dir, edited_pixels_dir, edited_pixel_coords_dir]:
        if not os.path.isdir(dir):
            os.makedirs(dir)

    all_frames = []
    psnrs = []
    output_imgs = {}

    # print('threshold: ', kwargs['threshold'])
    for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
        if eval_dataset.has_groundtruth:
            out_img_name = os.path.basename(eval_dataset.image_paths[indices[0]])[:-4]
        else:
            out_img_name = '{}'.format(indices[0])

        ### check 
        if os.path.exists('{0}/sg_rgb_{1}.png'.format(rgb_images_dir, out_img_name)):
            print('this view has already edited! continue...')
            print('{0}/sg_rgb_{1}.png'.format(rgb_images_dir, out_img_name))
            continue

        if len(kwargs['view_name']) > 0 and out_img_name != kwargs['view_name']:
            print('Skipping: ', out_img_name)
            continue

        print('Evaluating data_index: ', data_index, len(eval_dataloader))
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input["object_mask"] = model_input["object_mask"].cuda()
        model_input['pose'] = model_input['pose'].cuda()

        # add diffuse_rgb
        if rgb_images is not None:
            model_input['diffuse_rgb'] = rgb_images.cuda()
        else:
            model_input['diffuse_rgb'] = None

        split = utils.split_input(model_input, total_pixels)
        res = []

        for s in split:

            out = model.forward_dual_mlp(s, edited_3d_points, threshold=kwargs['threshold'])

            print('########################OUTPUT##########################')
            res.append({
                'points': out['points'].detach(),
                'idr_rgb_values': out['idr_rgb_values'].detach(),
                'sg_rgb_values': out['sg_rgb_values'].detach(),
                'normal_values': out['normal_values'].detach(),
                'network_object_mask': out['network_object_mask'].detach(),
                'object_mask': out['object_mask'].detach(),
                'sg_diffuse_albedo_values': out['sg_diffuse_albedo_values'].detach(),
                'sg_diffuse_rgb_values': out['sg_diffuse_rgb_values'].detach(),
                'sg_specular_rgb_values': out['sg_specular_rgb_values'].detach(),
                'differentiable_surface_points': out['differentiable_surface_points'].detach(),
            })

        batch_size = ground_truth['rgb'].shape[0]

        model_outputs = utils.merge_output(res, total_pixels, batch_size)

        tonemap_img = lambda x: np.power(x, 1./eval_dataset.gamma)
        clip_img = lambda x: np.clip(x, 0., 1.)

        assert (batch_size == 1)
       
        rgb_eval = model_outputs['sg_rgb_values']
        rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
        rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
        rgb_eval = rgb_eval.transpose(1, 2, 0)
        if kwargs['save_exr']:
            imageio.imwrite('{0}/sg_rgb_{1}.exr'.format(images_dir, out_img_name), rgb_eval)

        else:
            rgb_eval = clip_img(tonemap_img(rgb_eval))
            img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
            img.save('{0}/sg_rgb_{1}.png'.format(rgb_images_dir, out_img_name))

            print("INFO", '{0}/sg_rgb_{1}.png'.format(rgb_images_dir, out_img_name))

            all_frames.append(np.array(img))

        ## save diffuse rgb
        diffuse_rgb_eval = model_outputs['sg_diffuse_rgb_values']
        diffuse_rgb_eval = diffuse_rgb_eval.reshape(batch_size, total_pixels, 3)
        diffuse_rgb_eval = plt.lin2img(diffuse_rgb_eval, img_res).detach().cpu().numpy()[0]
        diffuse_rgb_eval = diffuse_rgb_eval.transpose(1, 2, 0)

        diffuse_rgb_eval = clip_img(tonemap_img(diffuse_rgb_eval))
        diffuse_rgb_img = Image.fromarray((diffuse_rgb_eval * 255).astype(np.uint8))
        diffuse_rgb_img.save('{0}/sg_rgb_{1}.png'.format(diffuse_rgb_images_dir, out_img_name))

        ## save diffuse albedo
        diffuse_albedo_eval = model_outputs['sg_diffuse_albedo_values']
        diffuse_albedo_eval = diffuse_albedo_eval.reshape(batch_size, total_pixels, 3)
        diffuse_albedo_eval = plt.lin2img(diffuse_albedo_eval, img_res).detach().cpu().numpy()[0]
        diffuse_albedo_eval = diffuse_albedo_eval.transpose(1, 2, 0)

        diffuse_albedo_eval = clip_img(tonemap_img(diffuse_albedo_eval))
        diffuse_albedo_img = Image.fromarray((diffuse_albedo_eval * 255).astype(np.uint8))
        diffuse_albedo_img.save('{0}/sg_rgb_{1}.png'.format(diffuse_albedo_images_dir, out_img_name))

        torch.cuda.empty_cache()

    # if not kwargs['save_exr']:
    #     imageio.mimwrite(os.path.join(images_dir, 'video_rgb.mp4'), all_frames, fps=15, quality=9)
    #     print('Done rendering', images_dir)


def evaluate_for_display(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = kwargs['conf']
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']

    expname = conf.get_string('train.expname') + '-' + kwargs['expname']

    task = kwargs["task"]
    flag = kwargs["flag"]

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))

    evaldir = os.path.join('../../cvpr23/exps', task, kwargs['expname'], flag, os.path.basename(kwargs['data_split_dir']))

    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                                                           kwargs['data_split_dir'],
                                                                           train_cameras=False)

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  collate_fn=eval_dataset.collate_fn
                                                  )
    total_pixels = eval_dataset.total_pixels
    img_res = eval_dataset.img_res
    H, W = img_res

    ########################################################################################################
    ### load model

    model = kwargs['model']

    #####################################################################################################
    # reset lighting
    #####################################################################################################
    relight = False
    if kwargs['light_sg'].endswith('.npy'):
        # print('Loading light from: ', kwargs['light_sg'])

        # 环境光贴图
        envmap, _ = os.path.split(kwargs['light_sg'])
        _, envmap, = os.path.split(envmap)

        model.envmap_material_network.load_light(kwargs['light_sg'])
        relight = True


    edit_diffuse = False
    rgb_images = None

    if len(kwargs['diffuse_rgb']) > 0:

        rgb = imageio.imread(kwargs['diffuse_rgb']).astype(np.float32)[:, :, :3]
        if not kwargs['diffuse_rgb'].endswith('.exr'):
            rgb /= 255.
        rgb_images = torch.from_numpy(rgb).cuda().reshape((-1, 3))

    utils.mkdir_ifnotexists(evaldir)


    ####################################################################################################################
    # print("evaluating...")
    model.eval()

    # extract mesh
    if (not edit_diffuse) and (not relight) and eval_dataset.has_groundtruth:
        with torch.no_grad():
            mesh = plt.get_surface_high_res_mesh(
                sdf=lambda x: model.implicit_network(x)[:, 0],
                resolution=kwargs['resolution']
            )

            # Taking the biggest connected component
            components = mesh.split(only_watertight=False)
            areas = np.array([c.area for c in components], dtype=np.float)
            mesh_clean = components[areas.argmax()]
            mesh_clean.export('{0}/mesh.obj'.format(evaldir), 'obj')

    output_imgs = {}

    for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
        if eval_dataset.has_groundtruth:
            out_img_name = os.path.basename(eval_dataset.image_paths[indices[0]])[:-4]
        else:
            out_img_name = '{}'.format(indices[0])

        if len(kwargs['view_name']) > 0 and out_img_name not in kwargs['view_name']:
            continue

        # print('Evaluating data_index: ', data_index, len(eval_dataloader))
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input["object_mask"] = model_input["object_mask"].cuda()
        model_input['pose'] = model_input['pose'].cuda()

        # add diffuse_rgb
        if rgb_images is not None:
            model_input['diffuse_rgb'] = rgb_images.cuda()
        else:
            model_input['diffuse_rgb'] = None

        split = utils.split_input(model_input, total_pixels)
        res = []

        for s in split:

            if kwargs["edited_3d_points"] is None:
                out = model(s)
            else:
                out = model.forward_dual_mlp(s, kwargs["edited_3d_points"], threshold=kwargs['threshold'], print_output=False)

            res.append({
                'points': out['points'].detach(),
                'idr_rgb_values': out['idr_rgb_values'].detach(),
                'sg_rgb_values': out['sg_rgb_values'].detach(),
                'normal_values': out['normal_values'].detach(),
                'network_object_mask': out['network_object_mask'].detach(),
                'object_mask': out['object_mask'].detach(),
                'sg_diffuse_albedo_values': out['sg_diffuse_albedo_values'].detach(),
                'sg_diffuse_rgb_values': out['sg_diffuse_rgb_values'].detach(),
                'sg_specular_rgb_values': out['sg_specular_rgb_values'].detach(),
                'differentiable_surface_points': out['differentiable_surface_points'].detach(),
            })

        batch_size = ground_truth['rgb'].shape[0]

        model_outputs = utils.merge_output(res, total_pixels, batch_size)

        tonemap_img = lambda x: np.power(x, 1./eval_dataset.gamma)
        clip_img = lambda x: np.clip(x, 0., 1.)

        assert (batch_size == 1)

        rgb_eval = model_outputs['sg_rgb_values']
        rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
        rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
        rgb_eval = rgb_eval.transpose(1, 2, 0)
        rgb_eval = clip_img(tonemap_img(rgb_eval))

        output_imgs[out_img_name] = rgb_eval

        torch.cuda.empty_cache()

    return output_imgs

def get_cameras_accuracy(pred_Rs, gt_Rs, pred_ts, gt_ts,):
    ''' Align predicted pose to gt pose and print cameras accuracy'''

    # find rotation
    d = pred_Rs.shape[-1]
    n = pred_Rs.shape[0]

    Q = torch.addbmm(torch.zeros(d, d, dtype=torch.double), gt_Rs, pred_Rs.transpose(1, 2))
    Uq, _, Vq = torch.svd(Q)
    sv = torch.ones(d, dtype=torch.double)
    sv[-1] = torch.det(Uq @ Vq.transpose(0, 1))
    R_opt = Uq @ torch.diag(sv) @ Vq.transpose(0, 1)
    R_fixed = torch.bmm(R_opt.repeat(n, 1, 1), pred_Rs)

    # find translation
    pred_ts = pred_ts @ R_opt.transpose(0, 1)
    c_opt = cp.Variable()
    t_opt = cp.Variable((1, d))

    constraints = []
    obj = cp.Minimize(cp.sum(
        cp.norm(gt_ts.numpy() - (c_opt * pred_ts.numpy() + np.ones((n, 1), dtype=np.double) @ t_opt), axis=1)))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    t_fixed = c_opt.value * pred_ts.numpy() + np.ones((n, 1), dtype=np.double) * t_opt.value

    # Calculate transaltion error
    t_error = np.linalg.norm(t_fixed - gt_ts.numpy(), axis=-1)
    t_error = t_error
    t_error_mean = np.mean(t_error)
    t_error_medi = np.median(t_error)

    # Calculate rotation error
    R_error = compare_rotations(R_fixed, gt_Rs)

    R_error = R_error.numpy()
    R_error_mean = np.mean(R_error)
    R_error_medi = np.median(R_error)

    print('CAMERAS EVALUATION: R error mean = {0} ; t error mean = {1} ; R error median = {2} ; t error median = {3}'
          .format("%.2f" % R_error_mean, "%.2f" % t_error_mean, "%.2f" % R_error_medi, "%.2f" % t_error_medi))

    # return alignment and aligned pose
    return R_opt.numpy(), t_opt.value, c_opt.value, R_fixed.numpy(), t_fixed

def compare_rotations(R1, R2):
    cos_err = (torch.bmm(R1, R2.transpose(1, 2))[:, torch.arange(3), torch.arange(3)].sum(dim=-1) - 1) / 2
    cos_err[cos_err > 1] = 1
    cos_err[cos_err < -1] = -1
    return cos_err.acos() * 180 / np.pi

def calculate_psnr(img1, img2, mask):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2) * (img2.shape[0] * img2.shape[1]) / mask.sum()
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/default.conf')
    parser.add_argument('--data_split_dir', type=str, default='')
    parser.add_argument('--gamma', type=float, default=1., help='gamma correction coefficient')

    parser.add_argument('--save_exr', default=False, action="store_true", help='')

    parser.add_argument('--light_sg', type=str, default='', help='')
    parser.add_argument('--geometry', type=str, default='', help='')
    parser.add_argument('--diffuse_albedo', type=str, default='', help='')
    parser.add_argument('--view_name', type=str, default='', help='')

    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')

    parser.add_argument('--model_params_dir', default='latest',type=str,help='The trained model checkpoint to test')

    parser.add_argument('--write_idr', default=False, action="store_true", help='')

    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--is_uniform_grid', default=False, action="store_true", help='If set, evaluate marching cube with uniform grid.')

    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')

    parser.add_argument('--diffuse_rgb', type=str, default='', help='')
    parser.add_argument('--flag', type=str, default='', help='')
    parser.add_argument('--task', type=str, default='', help='')
    parser.add_argument('--threshold', type=float, default=1e-9, help='')


    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)
        print('gpu: ', gpu)

    evaluate(conf=opt.conf,
             write_idr=opt.write_idr,
             gamma=opt.gamma,
             data_split_dir=opt.data_split_dir,
             expname=opt.expname,
             exps_folder_name=opt.exps_folder,
             evals_folder_name='evals',
             timestamp=opt.timestamp,
             model_params_dir=opt.model_params_dir,
             resolution=opt.resolution,
             save_exr=opt.save_exr,
             light_sg=opt.light_sg,
             geometry=opt.geometry,
             view_name=opt.view_name,
             diffuse_albedo=opt.diffuse_albedo,
             diffuse_rgb=opt.diffuse_rgb,
             threshold=opt.threshold, 
             flag=opt.flag, 
             task=opt.task
             )

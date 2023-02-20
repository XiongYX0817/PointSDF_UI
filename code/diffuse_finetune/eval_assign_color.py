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
import json

from utils.metrics import PSNR, SSIM, LPIPS

def evaluate(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']

    # expname = conf.get_string('train.expname') + '-' + kwargs['expname']
    expname = kwargs['expname']


    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            else:
                timestamp = sorted(timestamps)[-1]
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']

    task = kwargs["task"]
    flag = kwargs["flag"]
    evals_folder_name = os.path.join(kwargs['evals_folder_name'], timestamp, 'evals')

    # utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    utils.mkdir_ifnotexists(evals_folder_name)

    # expdir = os.path.join('../', exps_folder_name, expname)
    expdir = os.path.join('../', exps_folder_name, kwargs['expname'])

    # evaldir = os.path.join('../', evals_folder_name, expname, 'cvpr', os.path.basename(kwargs['data_split_dir']))
    # evaldir = os.path.join('../', 'cvpr23/unedited', expname, task, os.path.basename(kwargs['data_split_dir']))
    # evaldir = os.path.join(evals_folder_name, os.path.basename(kwargs['data_split_dir']))

    evaldir = os.path.join('../cvpr23/exps', task, kwargs['expname'], flag, os.path.basename(kwargs['data_split_dir']))


    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    if torch.cuda.is_available():
        model.cuda()

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

    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    ckpt_path = os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")
    saved_model_state = torch.load(ckpt_path)
    model.load_state_dict(saved_model_state["model_state_dict"])
    print('Loaded checkpoint: ', ckpt_path)

    if kwargs['geometry'].endswith('.pth'):
        print('Reloading geometry from: ', kwargs['geometry'])
        geometry = torch.load(kwargs['geometry'])['model_state_dict']
        geometry = {k: v for k, v in geometry.items() if 'implicit_network' in k}

        print(geometry)
        print(geometry.keys())
        model_dict = model.state_dict()
        model_dict.update(geometry)
        model.load_state_dict(model_dict)

    
    assigned_color = kwargs['color']


    #####################################################################################################
    # reset lighting
    #####################################################################################################
    relight = False
    if kwargs['light_sg'].endswith('.npy'):
        print('Loading light from: ', kwargs['light_sg'])

        # 环境光贴图
        envmap, _ = os.path.split(kwargs['light_sg'])
        _, envmap, = os.path.split(envmap)

        model.envmap_material_network.load_light(kwargs['light_sg'])
        evaldir = evaldir + '_%s_relight2' % envmap
        relight = True

    if not os.path.isdir(evaldir):
        os.makedirs(evaldir)
    print('Output directory is: ', evaldir)

    with open(os.path.join(evaldir, 'ckpt_path.txt'), 'w') as fp:
        fp.write(ckpt_path + '\n')

    ####################################################################################################################
    print("evaluating...")
    model.eval()

    # extract mesh
    # if (not relight) and eval_dataset.has_groundtruth:
    #     with torch.no_grad():
    #         mesh = plt.get_surface_high_res_mesh(
    #             sdf=lambda x: model.implicit_network(x)[:, 0],
    #             resolution=kwargs['resolution']
    #         )

    #         # Taking the biggest connected component
    #         components = mesh.split(only_watertight=False)
    #         areas = np.array([c.area for c in components], dtype=np.float)
    #         mesh_clean = components[areas.argmax()]
    #         mesh_clean.export('{0}/mesh.obj'.format(evaldir), 'obj')


    # generate images
    images_dir = evaldir

    rgb_save_dir = os.path.join(images_dir, 'rgb')
    diffuse_rgb_save_dir = os.path.join(images_dir, 'diffuse_rgb')
    diffuse_albedo_save_dir = os.path.join(images_dir, 'diffuse_albedo')
    norm_save_dir = os.path.join(images_dir, 'norm')
    mask_save_dir = os.path.join(images_dir, 'mask')
    depth_save_dir = os.path.join(images_dir, 'depth')
    specular_rgb_save_dir = os.path.join(images_dir, 'specular')

    for save_dir in [rgb_save_dir, diffuse_rgb_save_dir, diffuse_albedo_save_dir, norm_save_dir, mask_save_dir, depth_save_dir, specular_rgb_save_dir]:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    all_frames = []

    cnt = 0
    for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):

        cnt += 1
        if eval_dataset.has_groundtruth:
            out_img_name = os.path.basename(eval_dataset.image_paths[indices[0]])[:-4]
        else:
            out_img_name = '{}'.format(indices[0])

        if len(kwargs['view_name']) > 0 and out_img_name != kwargs['view_name']:
            print('Skipping: ', out_img_name)
            continue

        print('Evaluating data_index: ', data_index, len(eval_dataloader))
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input["object_mask"] = model_input["object_mask"].cuda()
        model_input['pose'] = model_input['pose'].cuda()
        model_input['diffuse_rgb'] = None


        split = utils.split_input(model_input, total_pixels)
        res = []


        for s in split:
            out = model(s, assigned_color)
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

        # differentiable_surface_points = model_outputs['differentiable_surface_points'].cpu().numpy()
        # np.save('../cvpr23/default-kitty/points/%s_differentiable_surface_points.npy' % out_img_name, differentiable_surface_points)


        tonemap_img = lambda x: np.power(x, 1./eval_dataset.gamma)
        clip_img = lambda x: np.clip(x, 0., 1.)

        assert (batch_size == 1)

        rgb_eval = model_outputs['sg_rgb_values']
        rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)
        rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
        rgb_eval = rgb_eval.transpose(1, 2, 0)
        
        rgb_eval = clip_img(tonemap_img(rgb_eval))
        img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
        img.save('{0}/sg_rgb_{1}.png'.format(rgb_save_dir, out_img_name))

        all_frames.append(np.array(img))

        #############################################################################################
        ### save diffuse rgb
        diffuse_rgb_eval = model_outputs['sg_diffuse_rgb_values']
        diffuse_rgb_eval = diffuse_rgb_eval.reshape(batch_size, total_pixels, 3)
        diffuse_rgb_eval = plt.lin2img(diffuse_rgb_eval, img_res).detach().cpu().numpy()[0]
        diffuse_rgb_eval = diffuse_rgb_eval.transpose(1, 2, 0)

        diffuse_rgb_eval = clip_img(tonemap_img(diffuse_rgb_eval))
        diffuse_rgb_img = Image.fromarray((diffuse_rgb_eval * 255).astype(np.uint8))
        diffuse_rgb_img.save('{0}/diffuse_rgb_{1}.png'.format(diffuse_rgb_save_dir, out_img_name))
        

        #############################################################################################
        ### save diffuse albedo
        diffuse_albedo_eval = model_outputs['sg_diffuse_albedo_values']
        diffuse_albedo_eval = diffuse_albedo_eval.reshape(batch_size, total_pixels, 3)
        diffuse_albedo_eval = plt.lin2img(diffuse_albedo_eval, img_res).detach().cpu().numpy()[0]
        diffuse_albedo_eval = diffuse_albedo_eval.transpose(1, 2, 0)

        diffuse_albedo_eval = clip_img(tonemap_img(diffuse_albedo_eval))
        diffuse_albedo_img = Image.fromarray((diffuse_albedo_eval * 255).astype(np.uint8))
        diffuse_albedo_img.save('{0}/diffuse_albedo_rgb_{1}.png'.format(diffuse_albedo_save_dir, out_img_name))

        #############################################################################################
        ### save specular rgb
        specular_rgb_eval = model_outputs['sg_specular_rgb_values']
        specular_rgb_eval = specular_rgb_eval.reshape(batch_size, total_pixels, 3)
        specular_rgb_eval = plt.lin2img(specular_rgb_eval, img_res).detach().cpu().numpy()[0]
        specular_rgb_eval = specular_rgb_eval.transpose(1, 2, 0)

        specular_rgb_eval = clip_img(tonemap_img(specular_rgb_eval))
        specular_rgb_eval = Image.fromarray((specular_rgb_eval * 255).astype(np.uint8))
        specular_rgb_eval.save('{0}/specular_rgb_{1}.png'.format(specular_rgb_save_dir, out_img_name))


        ###############################################################################################
        ### save normals
        normal = model_outputs['normal_values']
        normal = normal.reshape(batch_size, total_pixels, 3)
        normal = (normal + 1.) / 2.
        normal = plt.lin2img(normal, img_res).detach().cpu().numpy()[0]
        normal = normal.transpose(1, 2, 0)
       
        img = Image.fromarray((normal * 255).astype(np.uint8))
        img.save('{0}/normal_{1}.png'.format(norm_save_dir, out_img_name))


      
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
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')

    parser.add_argument('--write_idr', default=False, action="store_true", help='')

    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--is_uniform_grid', default=False, action="store_true", help='If set, evaluate marching cube with uniform grid.')

    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')

    parser.add_argument('--diffuse_rgb', type=str, default='', help='')
    parser.add_argument('--task', type=str, default='', help='')
    parser.add_argument('--flag', type=str, default='', help='')
    parser.add_argument('--r', type=float, default=255., help='')
    parser.add_argument('--g', type=float, default=255., help='')
    parser.add_argument('--b', type=float, default=255., help='')


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
            #  evals_folder_name='evals',
             evals_folder_name=os.path.join('../cvpr23/exps', opt.expname),
             timestamp=opt.timestamp,
             checkpoint=opt.checkpoint,
             resolution=opt.resolution,
             save_exr=opt.save_exr,
             light_sg=opt.light_sg,
             geometry=opt.geometry,
             view_name=opt.view_name,
             diffuse_albedo=opt.diffuse_albedo,
             diffuse_rgb=opt.diffuse_rgb,
             task=opt.task,
             color=[opt.r,opt.g,opt.b],
             flag=opt.flag
             )

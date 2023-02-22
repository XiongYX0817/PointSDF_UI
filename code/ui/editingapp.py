import os
import sys
import copy

sys.path.append('../')

import torch
import numpy as np
import imageio
from torch.nn import functional as F
from torchvision import utils, transforms
from PIL import Image
from pyhocon import ConfigFactory

from ui_utils import renormalize, show, labwidget, paintwidget, mean_colors
from rendering import render_path
from run_nerf_helpers import img2mse, get_rays, to8b, to_disp_img

import utils.general as utils_general
from diffuse_finetune.envmap_finetune_v3_dual_mlp_0 import finetune_for_display as finetune
from diffuse_finetune.eval_dual_mlp_dtu_cdist import evaluate_for_display as evaluate

##########################################################################
# UI
##########################################################################

IMG_SIZE = 128
VERBOSE = True
N_ITERS = {'color': 100, 'removal': 100, 'addition': 10000}
# N_ITERS = {'color': 1000, 'removal': 100, 'addition': 10000}
LR = 0.001
N_RAYS = {'color': 64, 'removal': 8192}
# N_RAYS = {'color': 256, 'removal': 8192}


class PhySGEditingApp(labwidget.Widget):
    def __init__(self, instance, config, use_cached=True, expname=None, edit_type=None, num_canvases=9, shape_params='fusion_shape_branch', color_params='color_branch', randneg=8192, device='cuda:0'):
        super().__init__(style=dict(border="3px solid gray", padding="8px", display="inline-block"))
        torch.set_default_tensor_type('torch.cuda.FloatTensor' if device == 'cuda:0' else 'cpu')
        self.edit_type = edit_type
        self.instance = instance
        self.shape_params = shape_params
        self.color_params = color_params
        self.size = IMG_SIZE
        self.randneg = randneg
        self.device = device
        self.msg_out = labwidget.Div()
        self.editing_canvas = paintwidget.PaintWidget(image='', width=self.size * 3, height=self.size * 3).on('mask', self.change_mask)
        self.editing_canvas.index = -1
        self.copy_mask = None
        inline = dict(display='inline', border="2px solid gray")

        self.positive_mask_btn = labwidget.Button(self.pad('edit color'), style=inline).on('click', self.positive_mask)
        self.sigma_mask_btn = labwidget.Button(self.pad('remove shape'), style=inline).on('click', self.sigma_mask)
        self.execute_btn = labwidget.Button(self.pad('execute'), style=inline).on('click', self.execute_edit)
        self.brushsize_textbox = labwidget.Textbox(5, desc='brushsize: ', size=3).on('value', self.change_brushsize)

        self.target = None

        self.color_style = dict(display='inline', border="2px solid white")
        trn = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
        bg_img = trn(Image.open('bg.png').convert('RGB'))
        bg_img = renormalize.as_url(bg_img * 2 - 1)
        self.color_pallete = [labwidget.Image(src=bg_img, style=self.color_style).on('click', self.set_color)]
        self.color_pallete[-1].index = 0
        self.color_pallete[-1].color_type = 'bg'

        for color in mean_colors.colors.values():
            image = torch.zeros(3, 32, 32)
            image[0, :, :] = color[0]
            image[1, :, :] = color[1]
            image[2, :, :] = color[2]
            image = image / 255. * 2 - 1
            self.color_pallete.append(labwidget.Image(src=renormalize.as_url(image), style=self.color_style).on('click', self.set_color))
            self.color_pallete[-1].index = len(self.color_pallete) - 1
            self.color_pallete[-1].color_type = 'color'
            # TODO: Highlight the white box with black for clarity

        self.color = None
        self.mask_type = None
        self.real_canvas_array = []
        self.real_images_array = []
        self.positive_masks = []

        self.target_edit_idx = None

        self.conf = ConfigFactory.parse_file("../confs_sg/dual_mlp_cdist.conf")
        # self.shape = "shape09149_rank00"
        # self.data_split_dir = "../../cvpr23/data/color_editing/chair/" + self.shape + "/train/"
        # self.expname = "chair/" + self.shape
        # self.timestamp = "2022_11_06_00_28_29"
        self.shape = "kitty1"
        self.data_split_dir = "../../cvpr23/data/color_editing/kitty/" + self.shape + "/train/"
        self.expname = "kitty/" + self.shape
        self.timestamp = "ModelParameters"
        self.exps_folder = "cvpr23/exps"
        self.gamma = 1.0
        self.resolution = 256
        self.threshold = 1e-2
        self.task = "2_texture_editing"
        self.flag = "2_unrelight_thres_1e-7"
        # self.display_ids = sorted([30, 46, 74, 12, 97, 85, 24, 56, 62][:num_canvases])
        # self.ckpt_path = "../../cvpr23/exps/0_unedited_models/chair/" + self.shape + "/" + self.timestamp + "/checkpoints/latest.pth"
        self.display_ids = sorted([11, 24, 49, 0, 9, 6, 52, 65, 74][:num_canvases])
        self.ckpt_path = "../../cvpr23/exps/0_unedited_models/kitty/" + self.shape + "/" + self.timestamp + "/checkpoints/latest.pth"
        self.edited_3d_points = None

        self.n_epochs = 1000
        self.lr = 1e-3
        self.mask = None

        self.model = utils_general.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
        model_dict = self.model.state_dict()
        if torch.cuda.is_available():
            self.model.cuda()
        model_state_dict = torch.load(self.ckpt_path)["model_state_dict"]
        for key, val in model_state_dict.items():
            if key.startswith('envmap_material_network.diffuse'):
                tmp = key.split('.')
                tmp[-2] = 'layer_%s.0' % (int(tmp[-2]) // 2)
                key = ".".join(tmp[1:])
                model_dict[key] = val
            else:
                model_dict[key] = val
        self.model.load_state_dict(model_dict)

        images = self.render()

        for i, image in enumerate(images):
            image = torch.from_numpy(image).permute(2, 0, 1)
            resized = F.interpolate(image.unsqueeze(dim=0), size=(self.size, self.size)).squeeze(dim=0)
            self.real_images_array.append(labwidget.Image(
                src=renormalize.as_url(resized)).on('click', self.set_editing_canvas))
            self.real_images_array[-1].index = i
            self.real_canvas_array.append(paintwidget.PaintWidget(
                image=renormalize.as_url(image),
                width=self.size * 3, height=self.size * 3).on('mask', self.change_mask))
            self.real_canvas_array[-1].index = i
            self.real_canvas_array[-1].negative_mask = ''
            self.real_canvas_array[-1].resized_image = renormalize.as_url(resized)
            self.real_canvas_array[-1].orig = renormalize.as_url(image)
            self.positive_masks.append(torch.zeros(image.shape).cpu())
        self.show_rgbs = True

        self.change_brushsize()

    def pad(self, s, total=14):
        white = ' ' * ((total - len(s)) // 2)
        return white + s + white

    def make_trasparent(self):
        for button in [self.sigma_mask_btn, self.positive_mask_btn]:
            button.style = {'display': 'inline', 'color': 'grey', 'border': "1px solid grey"}

    def negative_mask(self):
        self.mask_type = 'negative'
        if self.editing_canvas.image != '':
            self.editing_canvas.mask = self.real_canvas_array[self.editing_canvas.index].negative_mask

    def positive_mask(self):
        self.mask_type = 'positive'
        self.make_trasparent()
        self.positive_mask_btn.style = {'display': 'inline', 'color': 'black', 'border': "2px solid black"}
        self.editing_canvas.mask = ''

    def sigma_mask(self):
        self.mask_type = 'sigma'
        self.make_trasparent()
        self.sigma_mask_btn.style = {'display': 'inline', 'color': 'black', 'border': "2px solid black"}
        self.editing_canvas.mask = ''

    def update_canvas(self, images, disps=None):
        for i, image in enumerate(images):
            resized_rgb = F.interpolate(image.unsqueeze(dim=0), size=(self.size, self.size)).squeeze(dim=0)
            self.real_images_array[i].src = renormalize.as_url(resized_rgb)
            self.real_canvas_array[i].image = renormalize.as_url(image)
            self.real_canvas_array[i].resized_image = renormalize.as_url(resized_rgb)
            if disps is not None:
                disp_img = torch.from_numpy(to8b(to_disp_img(disps[i]))).unsqueeze(dim=0) / 255.
                resized_disp = F.interpolate(disp_img.unsqueeze(dim=0), size=(self.size, self.size)).squeeze(dim=0)
                self.real_canvas_array[i].resized_disp = renormalize.as_url(resized_disp)
                self.real_canvas_array[i].disp = renormalize.as_url(disp_img)

        if self.editing_canvas.index >= 0:
            self.editing_canvas.image = self.real_canvas_array[self.editing_canvas.index].image

    def set_color(self, evt):
        for i in range(len(self.color_pallete)):
            self.color_pallete[i].style = {'display': 'inline', 'border': "2px solid white"}
        evt.target.style = {'display': 'inline', 'border': "1px solid black"}
        if evt.target.color_type == 'bg':
            self.negative_mask()
        else:
            image = renormalize.from_url(evt.target.src) / 2 + 0.5
            image = image * 255
            self.color = [int(x) * 2 / 255. - 1 for x in image[:, 0, 0]]
            # color = torch.zeros((3, self.size * 2, self.size * 2)).cpu()
            color = torch.zeros((3, self.size * 4, self.size * 4)).cpu()
            color[0, :, :] = self.color[0]
            color[1, :, :] = self.color[1]
            color[2, :, :] = self.color[2]
            self.color = color

    def change_brushsize(self):
        brushsize = int(self.brushsize_textbox.value)
        for c in self.real_canvas_array:
            c.brushsize = brushsize
        self.editing_canvas.brushsize = brushsize

    def set_editing_canvas(self, evt):
        self.editing_canvas.image = self.real_canvas_array[evt.target.index].image
        self.editing_canvas.index = self.real_canvas_array[evt.target.index].index
        if self.mask_type == 'negative':
            self.editing_canvas.mask = self.real_canvas_array[evt.target.index].negative_mask
        else:
            self.editing_canvas.mask = ''

    def render(self):
        output_images = evaluate(conf=self.conf,
             write_idr=False,
             gamma=self.gamma,
             data_split_dir=self.data_split_dir,
             expname=self.expname,
             exps_folder_name=self.exps_folder,
             evals_folder_name='evals',
             timestamp=self.timestamp,
             resolution=self.resolution,
             save_exr=False,
             light_sg='',
             # view_name=[self.shape + "_" + str(idx) for idx in self.display_ids],
             view_name=["rgb_" + str(idx).zfill(6) for idx in self.display_ids],
             diffuse_rgb='',
             threshold=self.threshold,
             flag=self.flag,
             task=self.task,
             model=self.model,
             edited_3d_points = self.edited_3d_points
             )
        return [output_images[key] for key in output_images]

    def change_mask(self, ev):
        if self.mask_type == 'positive' or self.mask_type == 'sigma':
            i = self.editing_canvas.index
            if self.target_edit_idx is None:
                self.target_edit_idx = i
            elif self.target_edit_idx != i:
                self.msg_out.print("Sorry, you have to edit in the same view.", replace=True)
                return
            orig_img = renormalize.from_url(self.editing_canvas.image)
            mask = renormalize.from_url(self.editing_canvas.mask) / 2 + 0.5
            # mask = F.interpolate(mask.unsqueeze(dim=0), size=(self.size * 2, self.size * 2)).squeeze()
            mask = F.interpolate(mask.unsqueeze(dim=0), size=(self.size * 4, self.size * 4)).squeeze()
            if self.mask_type == 'positive':
                self.edit_type = 'color'
                if self.color is None:
                    self.show_msg('Please select a color.')
                    if ev.target.image != '':
                        self.real_canvas_array[ev.target.index].negative_mask = ''
                    return
                edited_img = orig_img * (1 - mask) + mask * self.color
            elif self.mask_type == 'sigma':
                self.edit_type = 'removal'
                # edited_img = orig_img * (1 - mask) + mask * torch.ones((3, self.size * 2, self.size * 2)).to(mask.device)
                self.msg_out.print("To be implemented...", replace=True)
                return
            self.positive_masks[i] += mask
            self.real_canvas_array[i].image = renormalize.as_url(edited_img)
            self.editing_canvas.image = renormalize.as_url(edited_img)
            self.real_images_array[i].src = renormalize.as_url(F.interpolate(edited_img.unsqueeze(dim=0), size=(self.size, self.size)).squeeze())
            self.editing_canvas.mask = ''
            self.edited_image = edited_img.permute(1, 2, 0).cpu().numpy().astype(np.float32)
            mask = mask.cpu().numpy()[0]
            self.mask = mask if self.mask is None else 1 - (1 - mask) * (1 - self.mask)
        elif self.mask_type == 'negative':
            i = ev.target.index
            self.real_canvas_array[i].negative_mask = self.editing_canvas.mask
        else:
            if ev.target.image != '':
                self.real_canvas_array[ev.target.index].negative_mask = ''

    def execute_edit(self):
        if self.edit_type == 'color':
            self.optimize()
        if self.edit_type == 'removal':
            # self.toggle_grad()
            # self.toggle_shape_edit()
            # self.create_remove_dataset()
            # self.get_cache()
            # self.optimize()
            self.msg_out.print("To be implemented...", replace=True)
            return
        rgbs = self.render()
        from PIL import Image
        for i, rgb in enumerate(rgbs):
            Image.fromarray((rgb * 255).astype(np.uint8)).save("../" + str(i) + ".png")
        rgbs = [torch.from_numpy(rgb).permute(2, 0, 1) for rgb in rgbs]
        self.update_canvas(rgbs)
        self.msg_out.print("Update Done!", replace=True)

    def optimize(self):
        edited_3d_points = finetune(conf=self.conf,
             gamma=self.gamma,
             data_split_dir=self.data_split_dir,
             expname=self.expname,
             exps_folder_name=self.exps_folder,
             evals_folder_name='evals',
             timestamp=self.timestamp,
             resolution=self.resolution,
             light_sg='',
             # view_name=[self.shape + "_" + str(self.display_ids[self.target_edit_idx])],
             view_name=["rgb_" + str(self.display_ids[self.target_edit_idx]).zfill(6)],
             edited_image=self.edited_image,
             mask_image=self.mask,
             n_epochs=self.n_epochs,
             lr=self.lr,
             flag=self.flag,
             task=self.task,
             model=self.model,
             msg_out=self.msg_out
             )
        model_dict = torch.load("tmp.pth")["model_state_dict"]
        self.model.load_state_dict(model_dict)
        self.edited_3d_points = edited_3d_points

    def show_msg(self, msg):
        self.msg_out.clear()
        self.msg_out.print(msg, replace=False)

    def widget_html(self):
        def h(w):
            return w._repr_html_()
        html = f'''<div {self.std_attrs()}>
        <div style="margin-top: 8px; margin-bottom: 8px;">
        <div style="display:inline-block; width:{1.00 * self.size + 2}px;
          text-align:center">
        {h(self.positive_mask_btn)}
        </div>
        <div style="display:inline-block; width:{1.00 * self.size + 2}px;
          text-align:center">
        {h(self.sigma_mask_btn)}
        </div>
        <div style="display:inline-block; width:{1.00 * self.size + 2}px;
          text-align:center">
        {h(self.execute_btn)}
        </div>
        </div>

        <div>

        <div style="display:inline-block;
          width:{(self.size + 2) * 4}px;
          height:{40}px;
          vertical-align:top;
          overflow-y: scroll;
          text-align:center">
          {show.html([[x] for x in self.color_pallete])}
        </div>
        <div style="display:inline-block;
          width:{(self.size + 2) * 2 + 12}px;
          height:{40}px;
          vertical-align:top;
          overflow-y: scroll;
          text-align:center">
        {h(self.msg_out)}
        </div>
        <div>

        <div style="width:{(self.size + 2) * 6 + 20}px;">
        <hr style="border:2px dashed gray; background-color: white">
        </div>

        <div>
        {h(self.editing_canvas)}
        <div style="display:inline-block;
          width:{(self.size + 2) * 3 + 20}px;
          height:{(self.size + 2) * 3 + 20}px;
          vertical-align:top;
          overflow-y: scroll;
          text-align:center">
          {show.html([[c] for c in self.real_images_array])}
        </div>
        </div>

        </div>
        '''
        return html

##########################################################################
# Utility functions
##########################################################################


def positive_bounding_box(data):
    pos = (data > 0)
    v, h = pos.sum(0).nonzero(), pos.sum(1).nonzero()
    left, right = v.min().item(), v.max().item()
    top, bottom = h.min().item(), h.max().item()
    return top, left, bottom + 1, right + 1


def centered_location(data):
    t, l, b, r = positive_bounding_box(data)
    return (t + b) // 2, (l + r) // 2


def paste_clip_at_center(source, clip, center, area=None):
    source = source.unsqueeze(dim=0).permute(0, 3, 1, 2)
    clip = clip.unsqueeze(dim=0).permute(0, 3, 1, 2)
    target = source.clone()
    t, l = (max(0, min(e - s, c - s // 2))
            for s, c, e in zip(clip.shape[2:], center, source.shape[2:]))
    b, r = t + clip.shape[2], l + clip.shape[3]
    # TODO: consider copying over a subset of channels.
    target[:, :, t:b, l:r] = clip if area is None else (
        (1 - area)[None, None, :, :].to(target.device) *
        target[:, :, t:b, l:r] +
        area[None, None, :, :].to(target.device) * clip)
    target = target.squeeze().permute(1, 2, 0)
    return target, (t, l, b, r)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=int)
    parser.add_argument('--randneg', type=int, default=8192)
    parser.add_argument('--config')
    parser.add_argument('--expname')
    parser.add_argument('--editname')
    parser.add_argument('--second_editname')
    parser.add_argument('--shape_params', default='fusion_shape_branch')
    parser.add_argument('--color_params', default='color_branch')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()

    writer = NeRFEditingApp(instance=args.instance, expname=args.expname, config=args.config, shape_params=args.shape_params, color_params=args.color_params, randneg=args.randneg, num_canvases=9, use_cached=False)
    expname = writer.expname

    editnames = [args.editname]
    if args.second_editname:
        editnames.append(args.second_editname)

    for editname in editnames:
        if editname:
            savedir = os.path.join(expname, editname)

            if args.shape_params != 'fusion_shape_branch':
                savedir = os.path.join(savedir, args.shape_params)
            if args.color_params != 'color_branch':
                savedir = os.path.join(savedir, args.color_params)
            if args.randneg != 8192:
                savedir += f'_{args.randneg}'

            os.makedirs(savedir, exist_ok=True)
            print('Working in', savedir)

            # load and execute the edit
            import time
            writer.editname_textbox.value = editname
            writer.load()
            time1 = time.time()
            writer.execute_edit()
            time2 = time.time()
            print(time2 - time1)
        else:
            savedir = os.path.join(expname, 'flythroughs', str(args.instance))
            os.makedirs(savedir, exist_ok=True)

        all_poses = torch.tensor(np.load(os.path.join(expname, 'poses.npy')))
        all_hwfs = torch.tensor(np.load(os.path.join(expname, 'hwfs.npy')))
        if args.expname:
            N_per_instance = 1
        else:
            N_per_instance = all_poses.shape[0] // writer.all_instance_styles.shape[0]
        ps, pe = args.instance * N_per_instance, (args.instance + 1) * N_per_instance
        all_poses = all_poses[ps:pe]

        nfs = [[writer.near, writer.far]] * all_poses.shape[0]
        styles = writer.instance_style.repeat((all_poses.shape[0], 1))

        with torch.no_grad():
            print(f'Saving samples in {savedir}')
            rgbs, disps, psnr = render_path(all_poses, styles, all_hwfs, writer.chunk, writer.test_kwargs, nfs=nfs, savedir=savedir, verbose=True, detach=True)
            if args.video:
                imageio.mimwrite(os.path.join(savedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
                imageio.mimwrite(os.path.join(savedir, 'disps.mp4'), to8b(disps / np.max(disps)), fps=30, quality=8)

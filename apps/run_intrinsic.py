import argparse
import os

import cv2
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.structures import Meshes, Textures
import torch

from lib.colab_util import set_renderer, get_verts_rgb_colors
from lib.data.EvalDataset import crop_image

parser = argparse.ArgumentParser()
parser.add_argument('-im', '--input_img', type=str, required=True)
parser.add_argument('-io', '--input_obj', type=str, required=True)
parser.add_argument('-o', '--out_path', type=str, default='./results/intrinsic3d/')
parser.add_argument('-n', '--num_frames', type=int, default=10)
args = parser.parse_args()

os.makedirs(args.out_path, exist_ok=True)
img_path = args.input_img
rect_path = img_path.replace('.%s' % (img_path.split('.')[-1]), '_rect.txt')
obj_path = args.input_obj

with open(os.path.join(args.out_path, 'colorIntrinsics.txt'), 'w') as f:
    f.write("{} 0 {} 0\n".format(290.0, 512.0))
    f.write("0 {} {} 0\n".format(290.0, 512.0))
    f.write("0 0 1 0\n")
    f.write("0 0 0 1\n")
with open(os.path.join(args.out_path, 'depthIntrinsics.txt'), 'w') as f:
    f.write("{} 0 {} 0\n".format(290.0, 512.0))
    f.write("0 {} {} 0\n".format(290.0, 512.0))
    f.write("0 0 1 0\n")
    f.write("0 0 0 1\n")

renderer = set_renderer(image_size=1024, use_sfm=True)

im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
if im.shape[2] == 4:
    im = im / 255.0
    im = im[:, :, :3] * im[:, :, 3:]
    im = (255.0 * im).astype(np.uint8)

rects = np.loadtxt(rect_path, dtype=np.int32)
if len(rects.shape) == 1:
    rects = rects[None]
pid = min(rects.shape[0] - 1, 0)
rect = rects[pid].tolist()
im = crop_image(im, rect)
hh = im.shape[0] // 2

# Setup
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Load obj file
verts_rgb_colors = get_verts_rgb_colors(obj_path)
verts_rgb_colors = torch.from_numpy(verts_rgb_colors).to(device)
textures = Textures(verts_rgb=verts_rgb_colors)

# Load obj
mesh = load_objs_as_meshes([obj_path], device=device)

# Set mesh
verts = mesh._verts_list
faces = mesh._faces_list
mesh_w_tex = Meshes(verts, faces, textures)

for i in range(args.num_frames):
    # Crop color image
    offset = int(hh * i / (args.num_frames - 1))
    im_cropped = im[offset:offset + hh, hh // 2:hh * 3 // 2]
    im_cropped = cv2.resize(im_cropped, (1024, 1024))

    # Crop depth image
    offset = i / (args.num_frames - 1)
    with open(os.path.join(args.out_path, 'frame-{:06d}.pose.txt'.format(i)), 'w') as f:
        f.write("1 0 0 0\n")
        f.write("0 1 0 {}\n".format(offset))
        f.write("0 0 1 0\n")
        f.write("0 0 0 1\n")
    R, T = look_at_view_transform(290.0, 0, 0, at=((0, 0.5 - offset, 0),), device=device)
    image_w_tex, zbuf = renderer(mesh_w_tex, R=R, T=T)
    image_w_tex = np.clip(image_w_tex[0, ..., :3].cpu().numpy(), 0.0, 1.0)[:, :, ::-1] * 255
    zbuf = np.clip((zbuf[0, ..., 0].cpu().numpy() - 290.0) * 500 + 290.0, 0.0, None)
    print(zbuf[0, 0], zbuf[np.nonzero(zbuf)].min(), zbuf[512, 512], zbuf[512, 522], zbuf.max())

    # im_val = cv2.imread(os.path.join(args.out_path, 'frame-lion.depth.png'), cv2.IMREAD_UNCHANGED)
    # print(im_val[0, 0], im_val[np.nonzero(im_val)].min(), im_val[240, 320], im_val.max(), im_val.dtype)

    cv2.imwrite(os.path.join(args.out_path, 'frame-{:06d}.color.png'.format(i)), im_cropped)
    cv2.imwrite(os.path.join(args.out_path, 'frame-{:06d}.depth.png'.format(i)), zbuf.astype('uint16'))
    cv2.imwrite(os.path.join(args.out_path, 'frame-{:06d}.depth-rgb.png'.format(i)), image_w_tex.astype('uint8'))

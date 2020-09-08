import argparse
import glob
import os

from tqdm import tqdm

from lib.colab_util import generate_video_from_obj, set_renderer

renderer = set_renderer()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_root', type=str, required=True)
args = parser.parse_args()

for obj_path in tqdm(glob.glob(os.path.join(args.input_root, '*.obj'))):
    out_img_path = '{}.png'.format(obj_path[:-4])
    video_path = '{}.mp4'.format(obj_path[:-4])

    generate_video_from_obj(obj_path, out_img_path, video_path, renderer)

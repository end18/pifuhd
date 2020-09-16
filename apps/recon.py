# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json 
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from numpy.linalg import inv

from lib.options import BaseOptions
from lib.mesh_util import save_obj_mesh_with_color, reconstruction
from lib.data import EvalWPoseDataset, EvalDataset
from lib.model import HGPIFuNetwNML, HGPIFuMRNet
from lib.model_pifu import ResBlkPIFuNet
from lib.geometry import index

from PIL import Image

parser = BaseOptions()


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    # Need to repeat sample_tensor along the batch dim num_views times
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        sample_tensor.shape[2],
        sample_tensor.shape[3]
    )
    return sample_tensor


def gen_mesh(res, net, cuda, data, save_path, thresh=0.5, use_octree=True, components=False):
    image_tensor_global = data['img_512'].to(device=cuda)
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter_global(image_tensor_global)
    net.filter_local(image_tensor[:,None])

    try:
        if net.netG.netF is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlF], 0)
        if net.netG.netB is not None:
            image_tensor_global = torch.cat([image_tensor_global, net.netG.nmlB], 0)
    except:
        pass
    
    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor_global.shape[0]):
            save_img = (np.transpose(image_tensor_global[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        cv2.imwrite(save_img_path, save_img)

        verts, faces, _, _ = reconstruction(
            net, cuda, calib_tensor, res, b_min, b_max, thresh, use_octree=use_octree, num_samples=50000)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        # if 'calib_world' in data:
        #     calib_world = data['calib_world'].numpy()[0]
        #     verts = np.matmul(np.concatenate([verts, np.ones_like(verts[:,:1])],1), inv(calib_world).T)[:,:3]

        color = np.zeros(verts.shape)
        interval = 50000
        for i in range(len(color) // interval + 1):
            left = i * interval
            if i == len(color) // interval:
                right = -1
            else:
                right = (i + 1) * interval
            net.calc_normal(verts_tensor[:, None, :, left:right], calib_tensor[:,None], calib_tensor)
            nml = net.nmls.detach().cpu().numpy()[0] * 0.5 + 0.5
            color[left:right] = nml.T

        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)


def gen_mesh_imgColor(res, netMR, netC, cuda, data, save_path, thresh=0.5, use_octree=True, components=False):
    image_tensor_global = data['img_512'].to(device=cuda)
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    netMR.filter_global(image_tensor_global)
    netMR.filter_local(image_tensor[0:1,None])
    netC.filter(image_tensor_global)
    netC.attach(netMR.netG.get_im_feat())

    try:
        if netMR.netG.netF is not None:
            image_tensor_global = torch.cat([image_tensor_global, netMR.netG.nmlF], 0)
        if netMR.netG.netB is not None:
            image_tensor_global = torch.cat([image_tensor_global, netMR.netG.nmlB], 0)
    except:
        pass

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor_global.shape[0]):
            save_img = (np.transpose(image_tensor_global[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        cv2.imwrite(save_img_path, save_img)

        verts, faces, _, _ = reconstruction(
            netMR, cuda, calib_tensor[0:1], res, b_min, b_max, thresh, use_octree=use_octree, num_samples=100000)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        verts_tensor = reshape_sample_tensor(verts_tensor, calib_tensor.size(0))

        color = np.zeros(verts.shape)
        interval = 10000
        for i in range(len(color) // interval):
            left = i * interval
            right = i * interval + interval
            if i == len(color) // interval - 1:
                right = -1
            netC.query(verts_tensor[:, :, left:right], calib_tensor)
            rgb = netC.get_preds()[0].detach().cpu().numpy() * 0.5 + 0.5
            color[left:right] = rgb.T

        save_obj_mesh_with_color(save_path, verts, faces, color)

    except Exception as e:
        print(e)


def recon(opt, use_rect=False):
    # load checkpoints
    state_dict_path = None
    if opt.load_netMR_checkpoint_path is not None:
        state_dict_path = opt.load_netMR_checkpoint_path
    elif opt.resume_epoch < 0:
        state_dict_path = '%s/%s_train_latest' % (opt.checkpoints_path, opt.name)
        opt.resume_epoch = 0
    else:
        state_dict_path = '%s/%s_train_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
    
    start_id = opt.start_id
    end_id = opt.end_id

    state_dict = None
    if state_dict_path is not None and os.path.exists(state_dict_path):
        print('Resuming from ', state_dict_path)
        state_dict = torch.load(state_dict_path)    
        print('Warning: opt is overwritten.')
        dataroot = opt.dataroot
        resolution = opt.resolution
        num_views = opt.num_views
        results_path = opt.results_path
        loadSize = opt.loadSize
        load_netC_checkpoint_path = opt.load_netC_checkpoint_path

        opt = state_dict['opt']
        opt.dataroot = dataroot
        opt.resolution = resolution
        opt.num_views = num_views
        opt.results_path = results_path
        opt.loadSize = loadSize
        opt.load_netC_checkpoint_path = load_netC_checkpoint_path
    else:
        raise Exception('failed loading state dict!', state_dict_path)
    
    # parser.print_options(opt)

    cuda = torch.device('cuda:%d' % opt.gpu_id)

    if use_rect:
        test_dataset = EvalDataset(opt)
    else:
        test_dataset = EvalWPoseDataset(opt)

    print('test data size: ', len(test_dataset))
    projection_mode = test_dataset.projection_mode

    opt_netG = state_dict['opt_netG']
    netG = HGPIFuNetwNML(opt_netG, projection_mode).to(device=cuda)
    netMR = HGPIFuMRNet(opt, netG, projection_mode).to(device=cuda)
    netC = ResBlkPIFuNet(opt).to(device=cuda)

    def set_eval():
        netG.eval()

    # load checkpoints
    netMR.load_state_dict(state_dict['model_state_dict'])
    netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s/recon' % (opt.results_path, opt.name), exist_ok=True)

    if start_id < 0:
        start_id = 0
    if end_id < 0:
        end_id = len(test_dataset)

    ## test
    with torch.no_grad():
        set_eval()

        print('generate mesh (test) ...')
        for i in tqdm(range(start_id, end_id // opt.num_views)):
            if i >= len(test_dataset):
                break
            
            # for multi-person processing, set it to False
            if True:
                test_data = test_dataset[i * opt.num_views]
                for j in range(i * opt.num_views + 1, (i + 1) * opt.num_views):
                    temp_data = test_dataset[j]
                    test_data['img'] = torch.cat([test_data['img'], temp_data['img']], dim=0)
                    test_data['img_512'] = torch.cat([test_data['img_512'], temp_data['img_512']], dim=0)
                    test_data['calib'] = torch.cat([test_data['calib'], temp_data['calib']], dim=0)
                    test_data['calib_world'] = torch.cat([test_data['calib_world'], temp_data['calib_world']], dim=0)
                if test_data is None:
                    continue

                save_path = '%s/%s/recon/result_%s_%d_%d.obj' % (opt.results_path, opt.name, test_data['name'], opt.resolution, opt.num_views)

                print(save_path)
                gen_mesh_imgColor(opt.resolution, netMR, netC, cuda, test_data, save_path, components=opt.use_compose)
            else:
                for j in range(test_dataset.get_n_person(i)):
                    test_dataset.person_id = j
                    test_data = test_dataset[i]
                    save_path = '%s/%s/recon/result_%s_%d.obj' % (opt.results_path, opt.name, test_data['name'], j)
                    gen_mesh(opt.resolution, netMR, cuda, test_data, save_path, components=opt.use_compose)

def reconWrapper(args=None, use_rect=False):
    opt = parser.parse(args)
    recon(opt, use_rect)

if __name__ == '__main__':
    reconWrapper()
  

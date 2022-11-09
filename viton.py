"""
Created on 2022

@author: Nam
https://github.com/beingjoey/pytorch_openpose_body_25 # body 25
https://github.com/Hzzone/pytorch-openpose # detect hand
https://github.com/GoGoDuck912/Self-Correction-Human-Parsing
https://github.com/shadow2496/VITON-HD
"""
import argparse
import os
# import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import torchgeometry as tgm
# import PIL
# from PIL import Image 

# from typing import overload
from src_pose import torch_openpose,util
import cv2
# import json
# import matplotlib.pyplot as plt
# import copy
# from src import model
# from src.body import Body
# from src.hand import Hand
# from os import path as osp

from datasets import VITONDataset, VITONDataLoader
from network import SegGenerator, GMM, ALIASGenerator
from utils.util import gen_noise, load_checkpoint, save_images
from create_pose import createjson, draw_hand
from human import human_parsing

def get_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str, required=True)
    # Viton
    parser.add_argument('--name', type=str, default='nam')
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)
    parser.add_argument('--shuffle', action='store_true')
    # parsing
    parser.add_argument("--input-dir", type=str, default='datasets/test/image', help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='datasets/test/image-parse', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")
    parser.add_argument('--display_freq', type=int, default=1)
    #
    parser.add_argument('--dataset_dir', type=str, default='datasets/')
    parser.add_argument('--dataset_mode', type=str, default='test')
    parser.add_argument('--dataset_list', type=str, default='test_pairs.txt')
    parser.add_argument('--save_dir', type=str, default='results/')
    
    #checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument("--gpu", type=str, default='1', help="choose gpu device.")
    parser.add_argument('--seg_checkpoint', type=str, default='seg_final.pth')
    parser.add_argument('--gmm_checkpoint', type=str, default='gmm_final.pth')
    parser.add_argument('--alias_checkpoint', type=str, default='alias_final.pth')
    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str, default='checkpoints/final.pth')
    # common
    parser.add_argument('--semantic_nc', type=int, default=13, help='# of human-parsing map classes')
    parser.add_argument('--init_type', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'], default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

    # for GMM
    parser.add_argument('--grid_size', type=int, default=5)

    # for ALIASGenerator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64, help='# of generator filters in the first conv layer')
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most',
                        help='If \'more\', add upsampling layer between the two middle resnet blocks. '
                             'If \'most\', also add one more (upsampling + resnet) layer at the end of the generator.')

    opt = parser.parse_args()
    return opt
#----------------------------------
def tienxuli(opt) :
    tp = torch_openpose.torch_openpose('body_25')
    img_names = []
    with open(os.path.join(opt.dataset_dir, opt.dataset_list), 'r') as f:
        for line in f.readlines():
            img_name, c_name = line.strip().split()
            img_names.append(img_name)
    
    
    for img_name in img_names :
        # Part 1. Img pose, json
        path_img = os.path.join(opt.input_dir, img_name)

        img = cv2.imread(path_img)
        # resize
        img = cv2.resize(img, (opt.load_width,opt.load_height))
        cv2.imwrite(path_img, img)
        poses = tp(img)
        draw_hand(opt = opt, img= img, img_name=img_name, poses= poses)
        createjson(opt= opt, img_name=img_name, poses= poses)
    # Part 2. Human-parsing
    human_parsing(opt= opt)
#-----------------------------
def test(opt, seg, gmm, alias ):
    up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss.cuda()

    test_dataset = VITONDataset(opt)
    test_dataset[0]
    test_loader = VITONDataLoader(opt, test_dataset)

    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            print(i)
            img_names = inputs['img_name']
            c_names = inputs['c_name']['unpaired']

            img_agnostic = inputs['img_agnostic'].cuda()
            parse_agnostic = inputs['parse_agnostic'].cuda()
            pose = inputs['pose'].cuda()
            c = inputs['cloth']['unpaired'].cuda()
            cm = inputs['cloth_mask']['unpaired'].cuda()

            # Part 1. Segmentation generation
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')

            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size()).cuda()), dim=1)

            parse_pred_down = seg(seg_input)
            parse_pred = gauss(up(parse_pred_down))
            parse_pred = parse_pred.argmax(dim=1)[:, None]

            parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float).cuda()
            parse_old.scatter_(1, parse_pred, 1.0)

            labels = {
                0:  ['background',  [0]],
                1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                2:  ['upper',       [3]],
                3:  ['hair',        [1]],
                4:  ['left_arm',    [5]],
                5:  ['right_arm',   [6]],
                6:  ['noise',       [12]]
            }
            parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float).cuda()
            for j in range(len(labels)):
                for label in labels[j][1]:
                    parse[:, j] += parse_old[:, label]

            # Part 2. Clothes Deformation
            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
            c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)

            _, warped_grid = gmm(gmm_input, c_gmm)
            warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

            # Part 3. Try-on synthesis
            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0.0] = 0.0
            parse_div = torch.cat((parse, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask

            output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)
    
            unpaired_names = []
            for img_name, c_name in zip(img_names, c_names):
                unpaired_names.append('{}_{}'.format(img_name.split('_')[0], c_name))

            save_images(output, unpaired_names, os.path.join(opt.save_dir, opt.name))

            if (i + 1) % opt.display_freq == 0:
                print("step: {}".format(i + 1))
            

def main():
    opt = get_opt()
    print(opt)

    if not os.path.exists(os.path.join(opt.save_dir, opt.name)):
        os.makedirs(os.path.join(opt.save_dir, opt.name))
 
    tienxuli(opt)

    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13

    load_checkpoint(seg, os.path.join(opt.checkpoint_dir, opt.seg_checkpoint))
    load_checkpoint(gmm, os.path.join(opt.checkpoint_dir, opt.gmm_checkpoint))
    load_checkpoint(alias, os.path.join(opt.checkpoint_dir, opt.alias_checkpoint))

    seg.cuda().eval()
    gmm.cuda().eval()
    alias.cuda().eval()

    test(opt, seg, gmm, alias)


if __name__ == '__main__':
    main()
    print("done")

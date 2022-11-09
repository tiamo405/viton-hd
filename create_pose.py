#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:17:47 2020

@author: joe
github pytorch_openpose_body_25
"""

# import argparse
from typing import overload
from src_pose import torch_openpose,util
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
# import copy
# from src import model
from src_pose.body import Body
from src_pose.hand import Hand
import os

def createjson(opt, img_name, poses):
    # path_save = "VITON-HD/datasets/test/openpose-json/" + img_name.split('.')[0] + "_keypoints.json"
    path_save = os.path.join(opt.dataset_dir, opt.dataset_mode, "openpose-json", (img_name.replace('.jpg', '_keypoints.json')))
    if os.path.exists(path_save) :
        return
    data_dict={
        "version":1.3,
        "people":[{"person_id":[-1],"pose_keypoints_2d":list(np.array(poses).reshape(75))}]
    }
    data_string = json.dumps(data_dict)

    # myjsonfile = open("VITON-HD/datasets/test/openpose-json/10550_00_keypoints.json", "w")
    myjsonfile = open(path_save, "w")
    myjsonfile.write(data_string)
    myjsonfile.close()
#----------------------------
# -------------------------------
def draw_body(opt, img, img_name, poses) :
    # path_save = os.path.join(opt.dataset_dir, opt.dataset_mode, \
    # "openpose-img", (img_name.replace('.jpg', '_rendered.png')))
    overlay_image = np.ones(img.shape, np.uint8)  * 0
    overlay_image = cv2.resize(overlay_image, (768, 1024))
    img_body = util.draw_bodypose(overlay_image, poses,'body_25')
    # cv2.imwrite(path_save, img_body)
    return img_body
#----------------------------
def draw_hand(opt, img, img_name, poses) :
    path_save = os.path.join(opt.dataset_dir, opt.dataset_mode, "openpose-img", (img_name.replace('.jpg', '_rendered.png')))
    if os.path.exists(path_save) :
        return
    body_estimation = Body(os.path.join(opt.checkpoint_dir, "body_pose_model.pth"))
    hand_estimation = Hand(os.path.join(opt.checkpoint_dir, "hand_pose_model.pth")) 
    # path_save = "VITON-HD/datasets/test/openpose-img/" + img_name.split('.')[0] + "_rendered.png"
    
    oriImg = img.copy()  # B,G,R order
    candidate, subset = body_estimation(oriImg)

    overlay_image = np.ones(oriImg.shape, np.uint8)  * 0

    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)
    all_hand_peaks = []
    for x, y, w, is_left in hands_list:
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        
        all_hand_peaks.append(peaks)
    # overlay_image = cv2.imread("VITON-HD/datasets/test/openpose-img/10550_00_rendered.png")
    overlay_image = draw_body(opt, img, img_name, poses)
    overlay_image = util.draw_handpose(overlay_image, all_hand_peaks)
    # overlay_image = cv2.resize(overlay_image, (768, 1024))

    # cv2.imwrite("VITON-HD/datasets/test/openpose-img/10550_00_rendered.png", overlay_image)
    cv2.imwrite(path_save, overlay_image)

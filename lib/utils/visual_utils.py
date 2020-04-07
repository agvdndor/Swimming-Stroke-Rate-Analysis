import sys
from glob import glob
from os import path as osp
from datetime import datetime
from tqdm import tqdm
from skimage import io, transform
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt
from statistics import mean
# torch imports
import torch
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

import flowiz as fz

# root path of project
from os import path as osp
import sys

# get root directory
import re
reg = '^.*/AquaPose'
project_root = re.findall(reg, osp.dirname(osp.abspath(sys.argv[0])))[0]
sys.path.append(project_root)


from lib.models.keypoint_rcnn import get_resnet50_pretrained_model

# utils
from lib.utils.slack_notifications import slack_message
from lib.utils.select_gpu import select_best_gpu
from lib.utils.rmsd import kabsch_rmsd, kabsch_rotate, kabsch_weighted_rmsd, centroid, centroid_weighted, rmsd, rmsd_weighted

# references import
# source: https://github.com/pytorch/vision/tree/master/references/detection
from references.engine import train_one_epoch, evaluate
from references.utils import collate_fn

from references.transforms import RandomHorizontalFlip

def tensor_to_numpy_image(img_tensor):
    return img_tensor.permute(1,2,0).detach().numpy()

def get_max_prediction(prediction):
    keypoints_scores = prediction[0]['keypoints_scores']
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    keypoints = prediction[0]['keypoints']

    max_score = 0
    max_box = []
    for idx, box in enumerate(boxes):
        if scores[idx].item() > max_score:
            max_score = scores[idx].item()
            max_box = box
            max_keypoints = keypoints[idx] 
            max_keypoints_scores = keypoints_scores[idx]
    
    return max_box.detach().numpy(), max_keypoints.detach().numpy(), max_keypoints_scores.detach().numpy()


def plot_image_with_kps(img_tensor, kps_list, color_list= ['b', 'r', 'g']):
    # plot positive prediction
    fig, ax = plt.subplots()
    plt.imshow(tensor_to_numpy_image(img_tensor))
    for kps, clr in zip(kps_list, color_list):
        ax.scatter(np.array(kps)[:,0],np.array(kps)[:,1], s=10, marker='.', c=clr)

def plot_numpy_image_with_kps(img, kps_list, color_list= ['b', 'r', 'g']):
    # plot positive prediction
    fig, ax = plt.subplots()
    plt.imshow(img)
    for kps, clr in zip(kps_list, color_list):
        ax.scatter(np.array(kps)[:,0],np.array(kps)[:,1], s=10, marker='.', c=clr)



def plot_optical_vectors(url,cut_co):
    vis = fz.convert_from_file(url)
    plt.imshow(vis[:,cut_co[0]:cut_co[0]+cut_co[1]])


def plot_image_with_kps_skeleton(img_tensor, kps_list, color_list=['b', 'r', 'g'], filter_ind=[x for x in range(0,13)], skeleton=[[1,2], [1,3],[3,5], [2,4],[4,6], [1,7], [2,8],[7,8],[7,9],[9,11],[8,10],[10,12]]):
    fig, ax = plt.subplots()
    plt.imshow(tensor_to_numpy_image(img_tensor))
    for kps, clr in zip(kps_list, color_list):
        ax.scatter((np.array(kps)[filter_ind])[:,0],(np.array(kps)[filter_ind])[:,1], s=10, marker='.', c=clr)
    for joint in skeleton:
        if joint[0] in filter_ind and joint[1] in filter_ind:
            plt.plot([kps[joint[0]][0], kps[joint[1]][0]], [kps[joint[0]][1], kps[joint[1]][1]], c=clr)

def get_image_with_kps_skeleton(img_tensor, kps_list, color_list=['b', 'r', 'g'], filter_ind=[x for x in range(0,13)], skeleton=[[1,2], [1,3],[3,5], [2,4],[4,6], [1,7], [2,8],[7,8],[7,9],[9,11],[8,10],[10,12]]):
    #fig, ax = plt.subplots()
    ax = plt.gca()
    plt.imshow(tensor_to_numpy_image(img_tensor))
    for kps, clr in zip(kps_list, color_list):
        ax.scatter((np.array(kps)[filter_ind])[:,0],(np.array(kps)[filter_ind])[:,1], s=10, marker='.', c=clr)
    for joint in skeleton:
        if joint[0] in filter_ind and joint[1] in filter_ind:
            plt.plot([kps[joint[0]][0], kps[joint[1]][0]], [kps[joint[0]][1], kps[joint[1]][1]], c=clr)
    return ax
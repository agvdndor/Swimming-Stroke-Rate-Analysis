from __future__ import print_function, division
import os

import sys
from os import path as osp

import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
import json


# get root directory
import re
reg = '^.*/AquaPose'
project_root = re.findall(reg, osp.dirname(osp.abspath(sys.argv[0])))[0]
sys.path.append(project_root)

from lib.utils.visual_utils import *
from lib.matching.matching import merge_head

from math import sqrt
# references import
# source: https://github.com/pytorch/vision/tree/master/references/detection
from references.transforms import RandomHorizontalFlip, ToTensor, Compose


class pck:
    
    def __init__(self, ds_list):
        self.ds_list = ds_list


    # return for prediction and gt the percentage of times it was visible (v=1) and for prediction v==1 and score > min_score
    def num_kp_visible(self, model, min_score = 0):
        # implicitly assume here that all datasets have equal number of joints
        ds = self.ds_list[0]

        res = {}
        res['gt'] = [0] * ds.num_joints
        res['dt'] = [0] * ds.num_joints

        for ds in self.ds_list:
            for idx, (_, target) in enumerate(tqdm(ds)):
                kps = target['keypoints'][0].detach().numpy()

                kps_merged = merge_head(kps)

                # get prediction (use cache)
                _, pred_kps, pred_scores = ds.predict(model, idx)
                pred_kps = merge_head(pred_kps)
                pred_scores = merge_head(pred_scores)

                for joint in range(0,13):
                    # if visible
                    if kps_merged[joint][2] > 0:
                        res['gt'][joint] += 1

                    
                        # if minimum score reached
                        if pred_scores[joint] >= min_score:
                            res['dt'][joint] += 1
        return res


    def score_per_keypoint(self, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], min_score = - float('inf'), include_occluded=True):
        ds = self.ds_list[0]

        # keep track of how many were visible
        num_visible = [0] * ds.num_joints

        # init results for each joint, for each treshold
        res = [[0 for t in thresholds] for i in range(0,ds.num_joints)]

        for ds in self.ds_list:
            for idx, (_, target) in enumerate(tqdm(ds)):

                # get gt 
                gt_kps = target['keypoints'][0].detach().numpy()
                gt_kps = merge_head(gt_kps)

                # get dt
                _, pred_kps, pred_scores = ds.predict(model, idx)
                dt_kps = merge_head(pred_kps)
                dt_scores = merge_head(pred_scores)

                # get torso diameter
                torso_dist = sqrt((gt_kps[1][0] - gt_kps[9][0])**2 + (gt_kps[1][1] - gt_kps[9][1])**2)
                # if not then use torso from previous iteration, should be fine

                # for every joint
                for joint in range(0,ds.num_joints):
                    if ((include_occluded and gt_kps[joint][2] >= 0) or (gt_kps[joint][2] > 0)) and dt_scores[joint] > min_score:
                        
                        num_visible[joint] += 1

                        # get distance between dt and gt
                        dist = sqrt((gt_kps[joint][0] - dt_kps[joint][0])**2 + (gt_kps[joint][1] - dt_kps[joint][1])**2)

                        for t_id, threshold in enumerate(thresholds):
                            if dist < threshold * torso_dist:
                                res[joint][t_id] += 1

        return [[num / num_visible[joint] for num in res[joint]] for joint in range(0,13)]


    def inversion_errors(self, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], inversion_pairs=[[1,2] , [3,4] , [5,6], [7,8] , [8,9], [10,11] , [11,12]], min_score = - float('inf')):
        ds = self.ds_list[0]

        # keep track of how many were visible
        num_visible = [0] * ds.num_joints

        # init results for each joint, for each treshold
        res = [[0 for t in thresholds] for i in range(0,ds.num_joints)]

        for ds in self.ds_list:
            for idx, (_, target) in enumerate(tqdm(ds)):

                # get gt 
                gt_kps = target['keypoints'][0].detach().numpy()
                gt_kps = merge_head(gt_kps)

                # get dt
                _, pred_kps, pred_scores = ds.predict(model, idx)
                dt_kps = merge_head(pred_kps)
                dt_scores = merge_head(pred_scores)

                # get torso diameter
                torso_dist = sqrt((gt_kps[1][0] - gt_kps[9][0])**2 + (gt_kps[1][1] - gt_kps[9][1])**2)
                # if not then use torso from previous iteration, should be fine

                # for every joint that 
                for joint in range(1,ds.num_joints):
                    if dt_scores[joint] > min_score:

                        # get sibling joint
                        for pair in inversion_pairs:
                            if joint in pair:
                                # first in pair is always odd so sibling is found modulo 2
                                sibling_joint = pair[joint % 2]
                                break

                        num_visible[joint] += 1

                        # get distance between dt and gt
                        dist = sqrt((gt_kps[joint][0] - dt_kps[joint][0])**2 + (gt_kps[joint][1] - dt_kps[joint][1])**2)

                        # get distance between dt and gt of sibling joint
                        dist_sibling = sqrt((gt_kps[sibling_joint][0] - dt_kps[joint][0])**2 + (gt_kps[sibling_joint][1] - dt_kps[joint][1])**2)


                        for t_id, threshold in enumerate(thresholds):
                            if dist > threshold * torso_dist and dist_sibling < threshold * torso_dist:
                                res[joint][t_id] += 1
        
        # avoid division by zero exception
        num_visible[0] = 1
        return [[num / num_visible[joint] for num in res[joint]] for joint in range(0,13)]

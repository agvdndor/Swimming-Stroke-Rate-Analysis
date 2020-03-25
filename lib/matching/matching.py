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

# root path of project
from os import path as osp
import sys

# get root directory
import re
reg = '^.*/AquaPose'
project_root = re.findall(reg, osp.dirname(osp.abspath(sys.argv[0])))[0]
sys.path.append(project_root)

from lib.dataset.PoseDataset import PoseDataset

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

from lib.utils.visual_utils import *


def merge_head(kps):
    return kps[4:]

def filter_kps(pred_kps, ref_kps, scores, min_score=0, occluded=True, side = None, filter_lr_confusion=False):
    
    filter_ind = np.argwhere(scores > min_score).flatten()

    # Reduce left right confusion by filtering out far elbows and wrists that are estimated too close
    # to their left counterpart
    if filter_lr_confusion:
        # get orientation of swimmer
        # upper body keypoints: head, left_shoulder, right shoulder
        upper_ind = [0, 1, 2]
        # lower body keyponts: left hip, right hip, left knee, right knee
        lower_ind = [7, 8, 9, 10]

        upper_ind_vis = np.intersect1d(upper_ind, filter_ind)
        lower_ind_vis = np.intersect1d(lower_ind, filter_ind)

        # only possible if one upper and one lower joint is visible
        if len(upper_ind_vis) > 0 and len(lower_ind_vis) > 0:
            # get mean x-co for upper and lower body
            upper_x = mean([kp[0] for kp in pred_kps[upper_ind_vis]])
            lower_x = mean([kp[0] for kp in pred_kps[lower_ind_vis]])

            if upper_x < lower_x:
                orientation = 'left'
            else:
                orientation = 'right'
            
            # [[left_elbow, right_elbow], [left_wrist, right_wrist]]
            for joints in [[3,4], [5,6]]:
                # if one of the joints is not present in filter ind
                # do nothing
                if joints[0] not in filter_ind or joints[1] not in filter_ind:
                    continue
                left_joint = pred_kps[joints[0]]
                right_joint = pred_kps[joints[1]]

                if rmsd_weighted(left_joint, right_joint, weights=[1]) < 5:
                    if orientation == 'left':
                        # filter out right joint
                        filter_ind = filter_ind[filter_ind != joints[1]]
                    else:
                        filter_ind = filter_ind[filter_ind != joints[0]] 

    if not occluded:
        not_occluded = np.argwhere(ref_kps[:,2] > 0).flatten()
        filter_ind = np.intersect1d(filter_ind, not_occluded)

    if side == 'left':
        left_ind = [0, 1 ,3 ,5 ,7 ,9, 11, 13, 15]
        filter_ind = np.intersect1d(filter_ind, left_ind)
    elif side == 'right':
        right_ind = [0, 2, 4, 6, 8, 10, 12, 14, 16]
        filter_ind = np.intersect1d(filter_ind, right_ind)

    return filter_ind

# pass unfiltered kps lists and weights
def do_kabsch_transform(pred_kps, ref_kps, translat_weights=None, filter_ind=None):   
    assert len(pred_kps) == len(ref_kps)
    
    pred_kps_np = np.array(pred_kps)
    ref_kps_np = np.array(ref_kps)

    if translat_weights is None:
        translat_weights_np = np.array([1] * len(ref_kps))
    else:
        assert len(pred_kps_np) == len(translat_weights)
        translat_weights_np = np.array(translat_weights)

    if filter_ind is None:
        filter_ind = np.array([i for i in range(0, len(ref_kps_np))])

    P = np.array([[kp[0], kp[1], 1] for kp in pred_kps_np[filter_ind]])
    Q = np.array([[kp[0], kp[1], 1] for kp in ref_kps_np[filter_ind]])
    weights = translat_weights_np[filter_ind]

    # same pose in opposite direction (no scaling so kabsch cannot do this)


    # TODO Swap keypoints left right!

    # use reflected pose if this leads to smaller distance
    # TODO this does not really reflect the actual value with weights and strict disctinctin between
    # translation and rotation

    QC = centroid_weighted(Q, weights)
    Q = Q - QC
    P = P - centroid_weighted(P, weights)
    P = kabsch_rotate(P, Q) + QC

    return P

# unfiltered pred_kps and ref_kps
def get_kabsch_distance(pred_kps, ref_kps, filter_ind = None, translat_weights=None, pose_similarity_weights=None):
    assert len(pred_kps) == len(ref_kps)
    
    pred_kps_np = np.array(pred_kps)
    ref_kps_np = np.array(ref_kps)

    if translat_weights is None:
        translat_weights_np = np.array([1] * len(ref_kps))
    else:
        assert len(pred_kps) == len(translat_weights)
        translat_weights_np = np.array(translat_weights)
    
    if pose_similarity_weights is None:
        pose_similarity_weights = np.array([1] * len(ref_kps))

    if filter_ind is None:
        filter_ind = np.array([i for i in range(0, len(ref_kps_np))])

    Q = ref_kps_np[filter_ind]

    P = do_kabsch_transform(pred_kps, ref_kps, filter_ind=filter_ind, translat_weights=translat_weights)
    return rmsd_weighted(P, Q, weights=pose_similarity_weights[filter_ind])

# DEPRECATED
# def get_affine_tf(pred_kps, ref_kps):
#     # make sure the visibility flag is 1 always (necessary for tf)
#     ref_kps_vis = [[kp[0], kp[1], 1] for kp in ref_kps]

#     A, res, rank, s = np.linalg.lstsq(pred_kps, ref_kps_vis)
#     return A

# def warp_kp(kps, tf_mat):
#     return np.dot(kps, tf_mat)

def kabsch_similarity_score(pred_kps, ref_kps, filter_ind=None, translat_weights=None, threshold_pct = 0.3 , weights = None):

    kabsch_kps = do_kabsch_transform(pred_kps,
        ref_kps,
        translat_weights= translat_weights,
        filter_ind= filter_ind
    )

    return similarity_score(kabsch_kps, ref_kps, 
        filter_ind = filter_ind,
        threshold_pct=threshold_pct,
        weights = weights
    )

# all incoming kps are unfiltered, EXCEPT kabsch_kps (because it is generated by the kabsch transform which automatically filters)
def similarity_score(kabsch_kps, ref_kps, filter_ind = None , threshold_pct = 0.3 , weights = None):
    kabsch_kps_np = np.array(kabsch_kps)
    ref_kps_np = np.array(ref_kps)

    if weights is None:
        weights = [1] * len(ref_kps_np)
    
    if filter_ind is None:
        filter_ind = np.array([i for i in range(0, len(ref_kps_np))])
    
    # get length of torso
    # get mean of shoulders
    mean_shoulder_x = mean([kp[0] for kp in ref_kps_np[[1,2]]])
    mean_shoulder_y = mean([kp[1] for kp in ref_kps_np[[1,2]]])
    # get mean of hips
    mean_hip_x = mean([kp[0] for kp in ref_kps_np[[7,8]]])
    mean_hip_y = mean([kp[1] for kp in ref_kps_np[[7,8]]])

    x_diff = abs(mean_shoulder_x - mean_hip_x)
    y_diff = abs(mean_shoulder_y - mean_hip_y)
    torso_length = sqrt(x_diff**2 + y_diff**2)

    # max distance to score points
    max_dist = threshold_pct * torso_length 

    ref_kps_ftrd = np.array(ref_kps_np)[filter_ind]
    weights_ftrd = np.array(weights)[filter_ind]

    #print('pred_kps_ftrd: {}'.format(pred_kps_ftrd))
    #print('gt_kps_ftrd: {}'.format(gt_kps_ftrd))

    assert len(kabsch_kps_np) == len(ref_kps_ftrd)

    #print('max_dist: {}'.format(max_dist))
    score = 0
    for pred_kp, ref_kp, weight in zip(kabsch_kps_np, ref_kps_ftrd, weights_ftrd):
        dist = sqrt((pred_kp[0] - ref_kp[0])**2 + (pred_kp[1] - ref_kp[1])**2)
        score += weight * max([max_dist - dist, 0])/max_dist
    
    return score

def get_most_similar_ind_and_scores(pred_kps, pred_scores, ref_dataset,
    num=10,
    plot=False,
    min_score=0,
    filter_lr_confusion=False,
    occluded=False,
    threshold_pct= 0.3,
    translat_weights = None,
    kp_weights = None):

    # merge left/right ear, left/right eye and nose int head
    pred_kps_merged = merge_head(pred_kps)
    pred_scores_merged = merge_head(pred_scores)

    # empty list of scores
    sim_scores = []

    # True if best score was achieved by flipping
    flipped = []

    # make horizontal flip tranformation
    flip = RandomHorizontalFlip(1.0)

    # go through all images of dataset to find best matches
    for ref_id, (ref_img, ref_target) in tqdm(enumerate(ref_dataset)):
        
        # get gt keypoint annotations
        ref_kps = ref_target['keypoints'][0].detach().numpy()
        ref_kps_merged = merge_head(ref_kps)

        # get flipped gt keypoint annotations
        ref_img_flipped, ref_target_flipped = flip(ref_img, ref_target)
        ref_kps_flipped = ref_target_flipped['keypoints'][0].detach().numpy()
        ref_kps_flipped_merged = merge_head(ref_kps_flipped)

        # filter according to parameters
        filter_ind = filter_kps(pred_kps_merged, ref_kps_merged, pred_scores_merged, 
            min_score= min_score,
            occluded = occluded,
            filter_lr_confusion = filter_lr_confusion
            )

        filter_ind_flipped =filter_kps(pred_kps_merged, ref_kps_flipped_merged, pred_scores_merged, 
            min_score= min_score,
            occluded = occluded,
            filter_lr_confusion = filter_lr_confusion
            )

        
        score = kabsch_similarity_score(pred_kps_merged, ref_kps_merged,
            filter_ind=filter_ind,
            translat_weights=translat_weights,
            threshold_pct=threshold_pct,
            weights=kp_weights
        )

        score_flipped = kabsch_similarity_score(pred_kps_merged, ref_kps_flipped_merged,
            filter_ind=filter_ind_flipped,
            translat_weights=translat_weights,
            threshold_pct=threshold_pct,
            weights=kp_weights
        )

        # add best score and keep track of whether it was by flipping
        if score > score_flipped:
            sim_scores.append(score)
            flipped.append(False)
        else:
            sim_scores.append(score_flipped)
            flipped.append(True)
        
        
    # get indices and score of most similar
    most_similar_ind = np.argsort(sim_scores)[::-1][:num]
    most_similar_scores = np.array(sim_scores)[most_similar_ind]
    most_similar_flipped = np.array(flipped)[most_similar_ind]

    if plot:
        for ind, score, is_flipped in zip(most_similar_ind, most_similar_scores, most_similar_flipped):

            print('dataset index: {}, score: {}, flipped: {}'.format(ind,score,is_flipped))

            ref_img, ref_target = ref_dataset[ind]

            if not is_flipped:
                
                ref_kps = ref_target['keypoints'][0].detach().numpy()
                ref_kps_merged = merge_head(ref_kps)

                # filter again
                filter_ind = filter_kps(pred_kps_merged, ref_kps_merged, pred_scores_merged, 
                    min_score= min_score,
                    occluded = occluded,
                    filter_lr_confusion = filter_lr_confusion
                )

                # get kabsch kps
                kabsch_kps = do_kabsch_transform(pred_kps_merged, ref_kps_merged, 
                    translat_weights=translat_weights,
                    filter_ind=filter_ind
                )

                plot_image_with_kps(ref_img, [kabsch_kps, ref_kps_merged[filter_ind]], ['r', 'k'])
            
            else:
                # get flipped gt keypoint annotations
                ref_img_flipped, ref_target_flipped = flip(ref_img, ref_target)
                ref_kps_flipped = ref_target_flipped['keypoints'][0].detach().numpy()
                ref_kps_flipped_merged = merge_head(ref_kps_flipped)

                filter_ind_flipped = filter_kps(pred_kps_merged, ref_kps_flipped_merged, pred_scores_merged, 
                    min_score= min_score,
                    occluded = occluded,
                    filter_lr_confusion = filter_lr_confusion
                )

                # get kabsch kps
                kabsch_kps_flipped = do_kabsch_transform(pred_kps_merged, ref_kps_flipped_merged, 
                    translat_weights=translat_weights,
                    filter_ind=filter_ind_flipped
                )

                plot_image_with_kps(ref_img_flipped, [kabsch_kps_flipped, ref_kps_flipped_merged[filter_ind_flipped]],['r', 'k'])

    return most_similar_ind, most_similar_scores, flipped



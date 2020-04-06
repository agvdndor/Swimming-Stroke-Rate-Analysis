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

from math import pi

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
from lib.utils.rmsd import kabsch_rmsd, kabsch_rotate, kabsch_weighted_rmsd, centroid, centroid_weighted, rmsd, rmsd_weighted, kabsch

# references import
# source: https://github.com/pytorch/vision/tree/master/references/detection
from references.engine import train_one_epoch, evaluate
from references.utils import collate_fn

from references.transforms import RandomHorizontalFlip

from lib.utils.visual_utils import *

T_WEIGHTS = np.array([10, 8, 8, 3, 3, 1, 1, 15, 15, 3, 3, 1, 1])
KP_WEIGHTS = np.array([3, 3, 3, 6, 6, 10, 10, 3, 3, 2, 2, 1, 1])


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
    P = kabsch_rotate(P, Q, max_rotation_radian= pi/90) + QC

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
    
    if len(pred_kps) == 0 or len(filter_ind) == 0:
        return 0.0


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


# Return subset of given dataset
def get_anchor_dataset(dataset, indices):
    return torch.utils.data.Subset(dataset, indices)

def divide_in_buckets(anchor_dataset, ref_dataset):
    pose_buckets = {}

    # init every bucket as empty
    for id in range(0,len(anchor_dataset)):
        pose_buckets[id] = []

    for id, (img_tensor, ref_target) in enumerate(ref_dataset):
        ref_kps = ref_target['keypoints'][0].detach().numpy()

        ref_scores = np.array([1] * len(ref_kps))

        best_ind, scores, flipped = get_most_similar_ind_and_scores(ref_kps, ref_scores, anchor_dataset, num=1,  filter_lr_confusion=False, occluded=True, translat_weights=T_WEIGHTS, kp_weights=KP_WEIGHTS)

        pose_buckets[best_ind[0]] += [id]

    return pose_buckets


def plot_bucket(ref_dataset, pose_buckets, anchor_id):
    bucket_ids = pose_buckets[anchor_id]

    for id in bucket_ids:
        # get image
        img, target = ref_dataset[id]
        kps = target['keypoints'][0].detach().numpy()

        plot_image_with_kps(img, [kps])

def viterbi_path(prior, transmat, obslik, scaled=True, ret_loglik=False):
    '''Finds the most-probable (Viterbi) path through the HMM state trellis
    Notation:
        Z[t] := Observation at time t
        Q[t] := Hidden state at time t
    Inputs:
        prior: np.array(num_hid)
            prior[i] := Pr(Q[0] == i)
        transmat: np.ndarray((num_hid,num_hid))
            transmat[i,j] := Pr(Q[t+1] == j | Q[t] == i)
        obslik: np.ndarray((num_hid,num_obs))
            obslik[i,t] := Pr(Z[t] | Q[t] == i)
        scaled: bool
            whether or not to normalize the probability trellis along the way
            doing so prevents underflow by repeated multiplications of probabilities
        ret_loglik: bool
            whether or not to return the log-likelihood of the best path
    Outputs:
        path: np.array(num_obs)
            path[t] := Q[t]
    '''
    num_hid = obslik.shape[0] # number of hidden states
    num_obs = obslik.shape[1] # number of observations (not observation *states*)

    # trellis_prob[i,t] := Pr((best sequence of length t-1 goes to state i), Z[1:(t+1)])
    trellis_prob = np.zeros((num_hid,num_obs))
    # trellis_state[i,t] := best predecessor state given that we ended up in state i at t
    trellis_state = np.zeros((num_hid,num_obs), dtype=int) # int because its elements will be used as indicies
    path = np.zeros(num_obs, dtype=int) # int because its elements will be used as indicies

    trellis_prob[:,0] = prior * obslik[:,0] # element-wise mult
    if scaled:
        scale = np.ones(num_obs) # only instantiated if necessary to save memory
        scale[0] = 1.0 / np.sum(trellis_prob[:,0])
        trellis_prob[:,0] *= scale[0]

    trellis_state[:,0] = 0 # arbitrary value since t == 0 has no predecessor
    for t in range(1, num_obs):
        for j in range(num_hid):
            trans_probs = trellis_prob[:,t-1] * transmat[:,j] # element-wise mult
            trellis_state[j,t] = trans_probs.argmax()
            trellis_prob[j,t] = trans_probs[trellis_state[j,t]] # max of trans_probs
            trellis_prob[j,t] *= obslik[j,t]
        if scaled:
            scale[t] = 1.0 / np.sum(trellis_prob[:,t])
            trellis_prob[:,t] *= scale[t]

    path[-1] = trellis_prob[:,-1].argmax()
    for t in range(num_obs-2, -1, -1):
        path[t] = trellis_state[(path[t+1]), t+1]

    if not ret_loglik:
        return path
    else:
        if scaled:
            loglik = -np.sum(np.log(scale))
        else:
            p = trellis_prob[path[-1],-1]
            loglik = np.log(p)
        return path, loglik

def get_observation_likelihood_and_hidden_state(model, inference_dataset, anchor_dataset, max_stride=1):
    model.eval()
    
    observations_list = []
    hidden_states_list = []
    obslik_list =[]
    img_ids_list = []
    flipped_list = []

    # initialize empty likelihood
    obslik = np.zeros((len(anchor_dataset), 0))


    last_id = 0

    while last_id < len(inference_dataset) - 1:
        # image ids (name of image.jpg), to allow for checking of stride
        prev_id = None
        cur_id = None

        observations = []
        hidden_states = []
        img_ids = []
        obslik = np.zeros((len(anchor_dataset), 0))
        flipped_mat = np.zeros((len(anchor_dataset),0))

        cur_range = range(last_id, len(inference_dataset))
        for id in tqdm(cur_range):
            img, target = inference_dataset[id]

            last_id = id
            cur_id = int(target['image_id'].item())
            print(cur_id)

            if prev_id is not None:
                if abs(cur_id - prev_id) > max_stride:
                    break
            
            img_ids += [id]

            prev_id = cur_id

            ref_kps = target['keypoints'][0].detach().numpy()
            prediction = model([img])
            pred_box, pred_kps, pred_scores = get_max_prediction(prediction)

            #plot_image_with_kps(img, [pred_kps[pred_scores>0]])

            # get gt hidden state
            ref_scores = np.array([1] * len(ref_kps))
            best_ind, scores, flipped = get_most_similar_ind_and_scores(ref_kps, ref_scores, anchor_dataset, num=1,  filter_lr_confusion=False, occluded=True, translat_weights=T_WEIGHTS, kp_weights=KP_WEIGHTS)

            hidden_states += [best_ind[0]]

            # get observed state
            best_ind, scores, flipped = get_most_similar_ind_and_scores(pred_kps, pred_scores, anchor_dataset, num=len(anchor_dataset),  filter_lr_confusion=False, occluded=False, translat_weights=T_WEIGHTS, kp_weights=KP_WEIGHTS)

            observations += [best_ind[0]]
            
            scores = np.array(scores)[np.argsort(best_ind)]
            flipped = np.array(flipped)[np.argsort(best_ind)]
            scores = np.power(scores, np.array([4.0] * len(scores)))
            scores_norm = np.array(scores)/np.sum(np.array(scores))
            scores_norm = np.array([[score] for score in scores_norm])
            flipped = np.array([[f] for f in flipped])
            obslik = np.append(obslik, scores_norm, axis=1)
            flipped_mat = np.append(flipped_mat, flipped, axis=1)
            
            print(obslik)

        hidden_states = np.array(hidden_states)
        observations = np.array(observations)
        img_ids_list += [img_ids]

        hidden_states_list += [hidden_states]
        observations_list += [observations]
        obslik_list += [obslik]
        flipped_list += [flipped_mat]

    return np.array(img_ids_list), np.array(obslik_list), np.array(observations_list), np.array(hidden_states_list), np.array(flipped_list)

def get_observation_likelihood(model, inference_dataset, anchor_dataset, max_stride=1, device=None):
    model.eval()
    cpu = torch.device('cpu')
    
    observations_list = []
    obslik_list =[]
    flipped_list = []

    # initialize empty likelihood
    obslik = np.zeros((len(anchor_dataset), 0))


    last_id = 0

    while last_id < len(inference_dataset) - 1:
        # image ids (name of image.jpg), to allow for checking of stride
        prev_id = None
        cur_id = None

        observations = []
        hidden_states = []
        img_ids = []
        obslik = np.zeros((len(anchor_dataset), 0))
        flipped_mat = np.zeros((len(anchor_dataset),0))

        cur_range = range(last_id, len(inference_dataset))
        for id in tqdm(cur_range):
            img, target = inference_dataset[id]

            last_id = id

            if device:
                img = img.to(device)
                
            prediction = model([img])

            if device:
                prediction = targets = [{k: v.cpu() for k, v in t.items()} for t in prediction]
                img.cpu()
            
            pred_box, pred_kps, pred_scores = get_max_prediction(prediction)

            # print('pred_kps {}'.format(pred_kps))
            # print('pred_scores {}'.format(pred_scores))

            # get observed state
            best_ind, scores, flipped = get_most_similar_ind_and_scores(pred_kps, pred_scores, anchor_dataset, num=len(anchor_dataset),  filter_lr_confusion=False, occluded=False, translat_weights=T_WEIGHTS, kp_weights=KP_WEIGHTS)

            observations += [best_ind[0]]
            
            scores = np.array(scores)[np.argsort(best_ind)]
            flipped = np.array(flipped)[np.argsort(best_ind)]
            scores = np.power(scores, np.array([4.0] * len(scores)))
            if np.sum(scores) > 0:
                scores_norm = np.array(scores)/np.sum(np.array(scores))
            else:
                scores_norm = np.array([1/len(scores)] * len(scores))
            scores_norm = np.array([[score] for score in scores_norm])
            flipped = np.array([[f] for f in flipped])
            obslik = np.append(obslik, scores_norm, axis=1)
            flipped_mat = np.append(flipped_mat, flipped, axis=1)
            
            print(obslik)

        observations = np.array(observations)

        observations_list += [observations]
        obslik_list += [obslik]
        flipped_list += [flipped_mat]

    return np.array(obslik_list), np.array(observations_list), np.array(flipped_list)

def build_transmat(num_states, probs=[.4,.5,.1]):
    transmat = np.zeros((num_states,num_states))
    for i in range (0,num_states):
        for offset, prob in enumerate(probs):
            transmat[i,(i+offset)%num_states] = prob
    return transmat

def warp_anchor_on_pred(model, inf_img, anchor_dataset, anchor_id, flipped):
    
    # get anchor
    anchor_img, anchor_target = anchor_dataset[anchor_id]
    
    if flipped:
        flip = RandomHorizontalFlip(1.0)
        anchor_img, anchor_target = flip(anchor_img, anchor_target)
    
    anchor_kps = anchor_target['keypoints'][0].detach().numpy()

    # get inference prediction
    if device:
            inf_img = inf_img.to(device)

    prediction = model([inf_img])

    if device:
            prediction = targets = [{k: v.cpu() for k, v in t.items()} for t in prediction]
            inf_img = inf_img.cpu()

    pred_box, pred_kps, pred_scores = get_max_prediction(prediction)
    
    
    #merge head
    anchor_kps_merged = merge_head(anchor_kps)
    pred_kps_merged = merge_head(pred_kps)
    pred_scores_merged = merge_head(pred_scores)

    min_score = 0

    # Filter first, then get transform on only filtered kps, then transform all anchor kps without filtering
    filter_ind = filter_kps(pred_kps_merged, anchor_kps_merged, pred_scores_merged, 
            min_score= min_score,
            occluded = False,
            filter_lr_confusion = True
            )

    if len(filter_ind) == 0:
        #plot_image_with_kps(inf_img, [pred_kps_merged[pred_scores_merged > 0]], ['r'])
        return ref_kps_np
    
    translat_weights = T_WEIGHTS
    
    # get the transform, using the keypoints after filtering
    # the below code is copied from do_kabsch_tranform
    pred_kps_np = np.array(pred_kps_merged)
    ref_kps_np = np.array(anchor_kps_merged)

    if translat_weights is None:
        translat_weights_np = np.array([1] * len(anchor_kps))
    else:
        assert len(pred_kps_np) == len(translat_weights)
        translat_weights_np = np.array(translat_weights)

    if filter_ind is None:
        filter_ind = np.array([i for i in range(0, len(ref_kps_np))])

    P = np.array([[kp[0], kp[1], 1] for kp in pred_kps_np[filter_ind]])
    Q = np.array([[kp[0], kp[1], 1] for kp in ref_kps_np[filter_ind]])
    weights = translat_weights_np[filter_ind]

    #QC = centroid_weighted(Q, weights)
    #Q = Q - QC
    #P = P - centroid_weighted(P, weights)

    pred_translat = centroid_weighted(P, weights)
    ref_translat = centroid_weighted(Q, weights)
    pred_t = P - pred_translat
    ref_t = Q - ref_translat

    #this was the line that did the actual transformation
    #P = kabsch_rotate(P, Q) + QC

    # replace it with getting the rotation matrix
    # rotate ref onto pred!! 
    rot_mat = kabsch(ref_t, pred_t,  max_rotation_radian=pi/90)

    #pred_t = np.array([[kp[0], kp[1], 1] for kp in pred_kps_np]) - pred_translat
    ref_t = np.array([[kp[0], kp[1], 1] for kp in ref_kps_np]) - ref_translat

    # rotate and translate back onto prediction
    ref_rot_t = np.dot(ref_t, rot_mat) + pred_translat
    #ref_rot_t = ref_t + pred_translat

    # only upper body
    filter_ind = np.intersect1d(filter_ind, [0,1,2,3,4,5,6,7,8])
    plot_image_with_kps(inf_img, [pred_kps_merged[pred_scores_merged > 0]], ['r'])
    return ref_rot_t
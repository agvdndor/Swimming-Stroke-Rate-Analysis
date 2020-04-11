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

# references import
# source: https://github.com/pytorch/vision/tree/master/references/detection
from references.transforms import RandomHorizontalFlip, ToTensor, Compose

class PoseDataset(Dataset):
    
    def __init__(self, dataset_list, train=True, stride=3, cache_predictions = False):
        '''
        dataset_list = contains all relevant datasets with subfolders ann and img as produced by supervise.ly
        stride = how densely the images are annotated, one every stride is annotated
        '''

        # # keypoints of the dataset
        # self.keypoints = {
        #     '0': "Head",
        #     '1': "Shoulder_Close",
        #     '2': "Elbow_Close",
        #     '3': "Wrist_Close",
        #     '4': "Hip_Close",
        #     '5': "Knee_Close",
        #     '6': "Ankle_Close",
        #     '7': "Shoulder_Far",
        #     '8': "Elbow_Far",
        #     '9': "Wrist_Far",
        #     '10': "Hip_Far",
        #     '11': "Knee_Far",
        #     '12': "Ankle_Far",
        # }

        # create coco compliant keypoints list
        self.keypoints = {
            '0': "Nose",
            '1': "left_eye",
            '2': "right_eye",
            '3': "left_ear",
            '4': "right_ear",
            '5': "Shoulder_Close",
            '6': "Shoulder_Far",
            '7': "Elbow_Close",
            '8': "Elbow_Far",
            '9': "Wrist_Close",
            '10': "Wrist_Far",
            '11': "Hip_Close",
            '12': "Hip_Far",
            '13': "Knee_Close",
            '14': "Knee_Far",
            '15': "Ankle_Close",
            '16': "Ankle_Far",
        }

        self.coco_keypoints = True

        # list with connected joints in skeleton
        #self.skeleton = [[1,2], [1,3], [2,4], [4,6], [5,7], [1,7], [7,8], [8,9], [7,10], [10,11], [11,12], [4,10]]
        self.skeleton = [[5,6], [5,7], [7,9], [6,8], [8,10], [5,11], [6,12], [11,12], [11,13], [12,14], [13,15], [14,16]]
        self.stride = stride

        # only add annotated frames 
        ann_list = []
        for dataset in dataset_list:
            for ann_file in sorted(glob(os.path.join(dataset, "ann", '*'))):
                if PoseDataset.is_annotated(ann_file):
                    ann_list += [ann_file]
        self.ann_list = ann_list
        self.train = train
        self.transform = self.get_transforms()
        self.num_joints = 13

        # cache predictions
        if cache_predictions:
            self.prediction_cache = [None] * len(self.ann_list)
            self.prediction_cache_corrected = [None] * len(self.ann_list)

    def __len__(self):
        return len(self.ann_list)

    def get_transforms(self):
        if self.train:
            return Compose([ToTensor(), RandomHorizontalFlip(0.5)])
        else:
            return Compose([ToTensor()])
    
    @staticmethod
    def is_annotated(ann_file):
        with open(ann_file) as file:
            ann = json.load(file)
            return len(ann['objects']) > 0
    
    @staticmethod
    def get_img_name_from_ann(ann_file):
        # change superdirectory form ann to img
        ann_file = ann_file.replace("ann", "img")

        # delete .json extension
        return ann_file.split('.json')[0]

    '''
    Return target dict as described by 
    https://github.com/pytorch/vision/blob/master/torchvision/models/detection/keypoint_rcnn.py
    line 28-34
    '''
    def get_target_from_file(self, json_file_url):
        # parse json file to dict
        with open(json_file_url) as json_file:
            ann = json.load(json_file)
        
        # get annotated objects (discard some irrelevent image description and tags)
        objects = ann['objects']

        # init empty dict
        target = {}

        # add image_id to target for coco compatibility
        target['image_id'] = torch.tensor(int(json_file_url.split('/')[-1].split('.')[0]), dtype=torch.int)

        # add is_crowd to target for coco compatibility
        target['iscrowd'] = torch.tensor(np.array([False]), dtype=torch.bool)

        # List containing all person bounding boxes
        # flatten to array [x1 y1 x2 y2]
        target['boxes'] = [np.array([obj['points']['exterior'] for obj in objects if obj['classTitle']=="Swimmer"]).flatten()]

        # add area of bounding boxes to target for coco compatibility
        area = []
        for box in target['boxes']:
            area += [abs((box[2] - box[0])* (box[3] - box[1]))]
        target['area'] = torch.tensor(np.array(area))

        # For each box in boxes add a label person (0: background, 1: person)
        target['labels'] = np.array([1 for box in target['boxes']])

        # keep a list of keypoint annotations that are missing (not the same as invisible)
        not_annotated = [0] * len(self.keypoints)

        keypoint_list = []
        for key_id in self.keypoints.keys():
            # for each keypoint specified in keypoints
            # find coordinates and visibility flag

            #if coco and keypoint is nose,left eye,..., right ear
            if self.coco_keypoints and int(key_id) < 5:
                keypoint_name = 'head'
            else:
                keypoint_name = self.keypoints[key_id]

            # get keypoint with name from json string
            keypoint = [obj for obj in objects if obj['classTitle'].casefold()==keypoint_name.casefold()]

            # if the annotation exists, add it
            if len(keypoint) > 0:
                coords = [kp['points']['exterior'] for kp in keypoint][0][0]

                # check if the keypoint is visible
                visible = True
                for kp in keypoint:
                    for tag in kp['tags']:
                        if tag['name'] == 'invisible':
                            visible = False
                visible = [1 if visible else 0]

            # if the annotation does not exists
            else:
                # take as coords the center of the box to avoid COCO evaluation of giving a large penalty
                box_x, box_y = (target['boxes'][-1][0] + target['boxes'][-1][2])/2, (target['boxes'][-1][1] + target['boxes'][-1][3])/2
                coords = [box_x, box_y]
                visible = [0]

                # indicate that is not annotated
                not_annotated[int(key_id)] = 1

            # there should at most be 1 keypoint of a kind
            if len(visible) > 1:
                print("Warning: image {} contains more than one instance of {}".format(json_file_url, self.keypoints[key_id]))
                coords = coords[0:2]
                visible = [visible[0]]

            # add to the list
            keypoint_list += [coords + visible]
    
        # TODO extend this to multiple boxes when we start doing multi person labeling
        target['keypoints'] = [keypoint_list]

        return target, not_annotated

    def draw_keypoints(self, idx):
        image, target = self[idx]
        PoseDataset.draw_keypoints_static(image.permute(1, 2, 0).numpy() , target['keypoints'].detach().numpy(), self.skeleton)

    @staticmethod
    def draw_keypoints_static(image, keypoints, skeleton=[]):
        '''
        Display image with keypoints and bounding box
        '''
        plt.figure()
        # show image
        plt.imshow(image, alpha=0.3)

        # find visible/invisible/not_annotated
        visible_flags = np.array(keypoints[:,2])
        print(visible_flags)
        visible_ind = [np.argwhere(visible_flags == 1).flatten()]
        invisible_ind = [np.argwhere(visible_flags == 0).flatten()]
        not_annotated_ind = [np.argwhere(visible_flags == -1).flatten()]
        print(visible_ind)

        # add keypoints to image
        plt.scatter(np.array(keypoints)[visible_ind,0],np.array(keypoints)[visible_ind,1], s=10, marker='.', c='r')
        plt.scatter(np.array(keypoints)[invisible_ind,0],np.array(keypoints)[invisible_ind,1], s=10, marker='.', c='b')

        # TODO: add color coding for different body parts
        
        # draw lines as specified in skeleton 
        if skeleton:
            for pair in skeleton:
                if visible_flags[pair[0]] == 1 and visible_flags[pair[1]] == 1:
                    plt.plot([keypoints[pair[0]][0], keypoints[pair[1]][0]], [keypoints[pair[0]][1], keypoints[pair[1]][1]], c='r')
                elif visible_flags[pair[0]] != -1 and visible_flags[pair[1]] != -1:
                    plt.plot([keypoints[pair[0]][0], keypoints[pair[1]][0]], [keypoints[pair[0]][1], keypoints[pair[1]][1]], c='b')

        # pause a bit so that plots are updated
        plt.pause(0.001)  
        
        # display
        plt.show(block=True)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        # get relevant file paths
        ann_file = self.ann_list[idx]
        img_file = PoseDataset.get_img_name_from_ann(ann_file)

        # load image
        image = io.imread(img_file)

        # get target dict
        target, not_annotated = self.get_target_from_file(ann_file)


        # target as tensors
        target['boxes'] = torch.FloatTensor(target['boxes'])
        target['labels'] = torch.tensor(target['labels']).to(torch.int64)
        target['keypoints'] = torch.FloatTensor(target['keypoints'])

        # correct orientation (left <-> right facing swimmer)
        target = self.correct_orientation(target, not_annotated)

        if self.transform:
            image, target = self.transform(image, target)

        return image, target


    def predict(self, model, idx, corrected=False):

        # get model device (by looking a random layer's weights)
        device = model.backbone.body.conv1.weight.device

        # if predictions are cached and a prediction is present then return it
        if self.prediction_cache and self.prediction_cache[idx] is not None:
            if corrected and self.prediction_cache_corrected[idx] is not None:
                return self.prediction_cache_corrected
            elif corrected and self.prediction_cache_corrected[idx] is None:
                print('Requested corrected pose, but none was cached. Returning uncorrected pose instead')
            return self.prediction_cache[idx]
        
        # get img tensor
        img, _ = self[idx]

        # load image onto device
        img = img.to(device)

        # make prediction
        prediction = model([img])
        
        # convert image and prediction to host memory
        prediction = [{k: v.cpu() for k, v in t.items()} for t in prediction]
        img.cpu()
        
        pred_box, pred_kps, pred_scores = get_max_prediction(prediction)

        # store in cache
        if self.prediction_cache:
            self.prediction_cache[idx] = [pred_box, pred_kps, pred_scores]

        return pred_box, pred_kps, pred_scores


    def correct_orientation(self, target, not_annotated):
        # Determine orientation of swimmer and modify target appropriately
        keypoints = target['keypoints']

        # x_coordinate of shoulder and hip
        shoulder_close_x = keypoints[0][5][0]
        shoulder_close_is_annotated = True if not_annotated[5]== 0 else False

        hip_close_x = keypoints[0][11][0]
        hip_close_is_annotated = True if not_annotated[11]== 0 else False

        shoulder_far_x = keypoints[0][6][0]
        shoulder_far_is_annotated = True if not_annotated[6]== 0 else False

        hip_far_x = keypoints[0][12][0]
        hip_far_is_annotated = True if not_annotated[12]== 0 else False

        # assumes at least one shoulder and at least one hip is annotated
        valid_shoulder = shoulder_close_x if shoulder_close_is_annotated else shoulder_far_x
        valid_hip = hip_close_x if hip_close_is_annotated else hip_far_x 

        if valid_shoulder < valid_hip: 
            direction = 'left'
            # order of keypoints is correct

        else:
            direction = 'right'

            # swap left and right joins (close = right, far = left)
            flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
            keypoints = keypoints[:, flip_inds]
            target['keypoints'] = keypoints

        return target

    
    def get_image_name_to_index(self):
        name_to_index = {}
        for idx, (_,target) in enumerate(self):
            image_id = target['image_id'].detach().numpy()
            name_to_index[image_id] = idx

        return name_to_index


    #########
    ## PCK ##
    #########

    # this is useful to cache all results and then evaluate metrics
    def predict_all(self, model):
        if not self.prediction_cache:
            print('predict_all called but prediction cache not enabled')
            return
        
        for idx in tqdm(range(0, len(self))):
            self.predict(model, idx)

    
    

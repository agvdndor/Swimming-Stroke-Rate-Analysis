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

# references import
# source: https://github.com/pytorch/vision/tree/master/references/detection
from references.transforms import RandomHorizontalFlip, ToTensor, Compose

class CycleDataset(Dataset):
    
    def __init__(self, dataset_list, stride=1, max_dist=30):
        '''
        dataset_list = contains all relevant datasets with subfolders ann and img as produced by supervise.ly
        stride = how densely the images are annotated, one every stride is annotated
        '''

        self.phases = {'arm_close_level' : 0,
                       'arm_far_level' : 1}

        self.max_dist = max_dist
        self.stride = stride

        # hold start and end index of sequence (end exlucsive)
        self.sequences = []

        # hold pairs of img filename and target
        self.items = []

        cur_phase = None
        cur_offset = 0

        seq_start = 0
        seq_end = None

        for dataset in dataset_list:
            for ann_file in glob(os.path.join(dataset, "ann_cycle", '*')):
                if CycleDataset.is_annotated(ann_file):
                    if cur_phase is None:
                        seq_start = len(self.items)
                    cur_phase = CycleDataset.get_phase(ann_file)
                    cur_offset = 0
                if cur_phase is not None:
                    # don't add any images until a first phase is selected
                    item_dict = {'img': CycleDataset.get_img_name_from_ann(ann_file),
                                 'phase': cur_phase,
                                 'offset' : cur_offset
                                 }
                    self.items.append(item_dict)

                    cur_offset += 1

                    if cur_offset > max_dist:
                        seq_end = len(self.items)
                        self.sequences.append([seq_start, seq_end])
                        cur_phase = None

        self.transform = Compose([ToTensor()])

    def __len__(self):
        return len(self.items)

    
    @staticmethod
    def is_annotated(ann_file):
        with open(ann_file) as file:
            ann = json.load(file)
            return len(ann['tags']) > 0
    
    @staticmethod
    def get_phase(ann_file):
        with open(ann_file) as file:
            ann = json.load(file)
            return ann['tags'][0]['name']
    
    @staticmethod
    def get_img_name_from_ann(ann_file):
        # change superdirectory form ann to img
        ann_file = ann_file.replace("ann_cycle", "img")

        # delete .json extension
        return ann_file.split('.json')[0]

    def get_mean_stroke_rate(self):
        cur_phase = None
        num_phases = -1
        num_frames = 0
        num_frames_in_phase = 0 
        for item in self.items:
            item_phase = self.phases[item['phase']]
            if item_phase != cur_phase:
                cur_phase = item_phase
                num_phases += 1
                num_frames += num_frames_in_phase
                num_frames_in_phase = 0
            num_frames_in_phase += 1
        
        # strokes per minute
        rate = num_phases / (2 * num_frames) * 25 * 60
 
        return rate


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        # get relevant file paths
        item_dict = self.items[idx]
        img_file = item_dict['img']

        # load image
        image = io.imread(img_file)


        if self.transform:
            image, _ = self.transform(image, None)

        return image, item_dict

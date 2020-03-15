import os
import sys

import torch
from torch import utils
import torchvision
from torch.utils import data

import transforms as custom_transforms 
import torchvision.transforms as transforms

from PoseDataset import PoseDataset
from test_model import get_model




if __name__ == '__main__':

    # get model
    model, device, dtype = get_model()
    model.train()

    # get dataset
    transform=transforms.Compose([custom_transforms.DictToTensor()])
    dataset = PoseDataset(['../../data/vzf/freestyle/freestyle_1'], 3, transform=transform)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=None)

    for sample in data_loader:
        print("new sample")
    
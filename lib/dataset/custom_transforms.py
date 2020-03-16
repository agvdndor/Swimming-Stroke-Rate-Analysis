import torch
import numpy as np
import torchvision
from torchvision import transforms


class DictToTensor(object):
    """item from PoseDataset to tensors
            - image
            - target dict
    """
    def __init__(self):
        self.toTensor = transforms.ToTensor()

    def __call__(self, image, target):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = self.toTensor(image)

        target['boxes'] = torch.FloatTensor(target['boxes'])
        target['labels'] = torch.tensor(target['labels']).to(torch.int64)
        target['keypoints'] = torch.FloatTensor(target['keypoints'])

        return image, target

class DictNormalize(object):
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample, target):
        sample['image'] = self.normalize(sample['image'])

        return sample
import torch
import numpy as np
import torchvision
from torchvision import transforms


class DictToTensor(object):
    """item from PoseDataset to tensors
            - image
            - target dict
    """

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = transforms.ToTensor()(image)

        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        for key in target.keys():
            target[key] = torch.from_numpy(np.array(target[key])).type(dtype)

        return {'image': image,
                'target': target}
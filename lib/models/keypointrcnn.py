import torch
import torchvision

from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn

def get_model():
    return keypointrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=2, pretrained_backbone=True)


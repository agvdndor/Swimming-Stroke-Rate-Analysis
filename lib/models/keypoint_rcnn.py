import torchvision
from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn

# root path of project
from os import path as osp
import sys
project_root = osp.join('../..')
sys.path.append(project_root)

def get_resnet50_pretrained_model():
    return torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)

# expand options over time

if __name__ == '__main__':
    # test
    model = get_resnet50_pretrained_model()

    print('get_model encountered no problems')
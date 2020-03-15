import torch
import torchvision
from torch import nn

# inspiration: https://discuss.pytorch.org/t/how-to-re-train-keypointrcnn-model-on-custom-dataset/49946/2
def get_model():
    is_available = torch.cuda.is_available()
    device = torch.device('cuda:0' if is_available else 'cpu')
    dtype = torch.cuda.FloatTensor if is_available else torch.FloatTensor
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, min_size=640)

    for param in model.parameters():
        param.requires_grad = False
    
    out = nn.ConvTranspose2d(512, 12, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    model.roi_heads.keypoint_predictor.kps_score_lowres = out
    
    return model, device, dtype
        
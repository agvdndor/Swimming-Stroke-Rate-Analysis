import sys
from os import path as osp
from datetime import datetime
from tqdm import tqdm

import time

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

# references import
# source: https://github.com/pytorch/vision/tree/master/references/detection
from references.engine import train_one_epoch, evaluate
from references.utils import collate_fn



def train_only_roi_heads(model):
    # only train parameters in roi_heads
    for param in model.parameters():
        param.requires_grad=False
    for param in model.roi_heads.parameters():
        param.requires_grad=True


def main(args):

    experiment_name = ''
    output_base_url = osp.join(project_root, 'weights', '{}_{}'.format(experiment_name, datetime.now().strftime("%d-%m-%Y-%H-%M")))

    # get model
    print('loading model...')
    model = get_resnet50_pretrained_model()

    # create dataset
    print('loading dataset...')
    dataset = PoseDataset([osp.join(project_root,'data/vzf/breaststroke/breaststroke_1'),osp.join(project_root,'data/vzf/freestyle/freestyle_1')], train=True)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:-20])
    dataset_test = torch.utils.data.Subset(dataset, indices[-20:])

    # create dataloaders
    print('creating dataloaders...')
    data_loader = DataLoader(
            dataset_train, batch_size=10, shuffle=True, num_workers=4,
            collate_fn=collate_fn)

    data_loader_test = DataLoader(
            dataset_test, batch_size=10, shuffle=True, num_workers=4,
            collate_fn=collate_fn)

    # get device
    device = select_best_gpu(min_mem=6100) if torch.cuda.is_available() else torch.device('cpu')
    print('selected device: {}'.format(device))

    # only set roi_heads trainable
    train_only_roi_heads(model)

    # grab trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]

    # create optimizer and scheduler
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    #train
    print('loading model onto device')
    model.to(device)
    

    start = time.time()
    num_epochs = 10
    for epoch in tqdm(range(0,num_epochs)):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)
    torch.save(model.state_dict(), output_base_url + '_epoch{}-{}.wth'.format(epoch, num_epochs))
    end = time.time()

    duration_min = (end - start)/60
    slack_message("Done Training, took {}min".format(duration_min), channel='#train')
    
if __name__ == '__main__':
    # TODO get args
    args = None

    main(args)
    
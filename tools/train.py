import sys
from os import path as osp
from datetime import datetime
from tqdm import tqdm
import copy

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
from references.engine import train_one_epoch, val_one_epoch, evaluate, get_validation_error
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

    # create datasets
    print('loading train dataset...')
    train_dataset = PoseDataset([
        osp.join(project_root,'data/vzf/freestyle/freestyle_1'),
        osp.join(project_root,'data/vzf/freestyle/freestyle_2'), 
        osp.join(project_root,'data/vzf/freestyle/freestyle_3'),
        osp.join(project_root,'data/vzf/freestyle/freestyle_4')], train=True)
    print('train dataset size: {}'.format(len(train_dataset)))

    print('loading val dataset...')
    val_dataset = PoseDataset([
        osp.join(project_root,'data/vzf/freestyle/freestyle_5'),
        osp.join(project_root,'data/vzf/freestyle/freestyle_6')], train=True)
    print('test dataset size: {}'.format(len(val_dataset)))

    # split the dataset in train and test set
    #indices = torch.randperm(len(dataset)).tolist()
    #dataset_train = torch.utils.data.Subset(dataset, indices[:-20])
    #dataset_test = torch.utils.data.Subset(dataset, indices[-20:])

    # create dataloaders
    print('creating dataloaders...')
    data_loader = DataLoader(
            train_dataset, batch_size=10, shuffle=True, num_workers=4,
            collate_fn=collate_fn)
            

    data_loader_test = DataLoader(
            val_dataset, batch_size=10, shuffle=True, num_workers=4,
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

    # training mode
    model.train()

    start = time.time()

    # initialize to very large value
    min_box_loss = 10000
    min_kp_loss = 10000

    num_epochs = 100
    get_validation_error(model, data_loader_test, device)
    for epoch in tqdm(range(0,num_epochs)):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        

        if epoch % 5 == 0:
            # validation
            box_loss, kp_loss = get_validation_error(model, data_loader_test, device)
            print('box_loss: {}, kp_loss: {}'.format(box_loss, kp_loss))
            if kp_loss < min_kp_loss:
                lr_scheduler.step()
                print('improved val score, saving state dict...')
                # lower validation score found
                min_kp_loss = kp_loss
                min_box_loss = box_loss

                temp_state_dict = copy.deepcopy(model.state_dict())
            else:
                print('loading previous state dict (current best: {})...'.format(min_kp_loss))
                model.load_state_dict(temp_state_dict)
            
        # every 10 epochs use coco to evaluate
        if epoch % 10 == 0:
            print('COCO EVAL EPOCH {}'.format(epoch))
            evaluate(model, data_loader_test, device=device)

    evaluator = evaluate(model, data_loader_test, device=device)
    torch.save(model.state_dict(), output_base_url + '_epoch{}-{}_min_val_loss_{}.wth'.format(epoch, num_epochs, min_kp_loss))
    end = time.time()

    duration_min = int((end - start)/60)

    # post result to slack channel
    slack_message("Done Training, took {}min \n box loss: {}, KP loss: {}".format(duration_min, min_box_loss, min_kp_loss), channel='#training')
    
if __name__ == '__main__':
    # TODO get args
    args = None

    main(args)
    
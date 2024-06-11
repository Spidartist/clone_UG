from tensorboardX.summary import scalar
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import json
from datetime import datetime
from tensorboardX import SummaryWriter
import time
import os
import argparse
import numpy as np

from loss import DiceCELoss, WeightedPosCELoss, ConsitencyLoss
from backboned_unet import Unet
from learning_rate import get_warmup_cosine_lr
from data_loader import Data
from utils import GetItem
from score import DiceScore


# ------- Define command line params --------
parser = argparse.ArgumentParser(description='UG classification')
parser.add_argument('--test-fold', help='Define fold id for testing, id is between 0 and 4', default=0)
parser.add_argument('--metadata-file', help='Specify the json file contains data location', default='dir.json')
args = parser.parse_args()
test_fold = int(args.test_fold)
metadata_file = args.metadata_file


# ------- 0. set hyper parameters --------
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = 'cuda' if torch.cuda.is_available() == True else 'cpu'
BASE_LR = 1e-6
MAX_LR = 1e-3
warmup_epochs = 2
batch = 16
epoch_num = 70
save_freq = 200
img_size = (480, 480)


# # ------- 1. define tensorboard writer --------
savepath = './log/' + str(datetime.now().strftime('%Y%m%d%H%M%S'))
sw = SummaryWriter(savepath + '/main')
sw1 = SummaryWriter(savepath + '/epoch_loss')
sw2 = SummaryWriter(savepath + '/epoch_pos_loss')
sw3 = SummaryWriter(savepath + '/epoch_dmg_loss')
sw4 = SummaryWriter(savepath + '/epoch_seg_loss')
sw5 = SummaryWriter(savepath + '/epoch_cons_loss')


# ------- 2. define data loader --------
d = Data(metadata_file=metadata_file, test_fold=test_fold, img_size=img_size, segmentation_classes=6)
data_loader = DataLoader(dataset=d, batch_size=batch, shuffle=True)


# ------- 3. define loss & score function --------
if device == 'cuda':
    cls_loss = WeightedPosCELoss().cuda()
    seg_loss = DiceCELoss().cuda()
    consistency_loss = ConsitencyLoss().cuda()
    get_item = GetItem().cuda()
    dice_score = DiceScore().cuda()
else:
    cls_loss = WeightedPosCELoss()
    seg_loss = DiceCELoss()
    consistency_loss = ConsitencyLoss()
    dice_score = DiceScore()
    get_item = GetItem()


# ------- 4. define model --------
net = Unet(classes=6, position_classes=10, damage_classes=6)
net.train(True)
if device == 'cuda':
    net.cuda()

base, head = [], []
for name, param in net.named_parameters():
    if 'backbone' in name:
        base.append(param)
    else:
        head.append(param)


# ------- 5. define optimizer --------
optimizer = optim.Adam([{'params': base}, {'params': head}], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# optimizer = torch.optim.SGD([{'params':base}, {'params':head}], lr=0.001, momentum=0.9, weight_decay=5e-4, nesterov=True)


# ------- 6. training process --------
print('epoch            : %d' % epoch_num)
print('batch_size       : %d' % batch)
print('save_freq        : %d' % save_freq)
print('img_size         : (%d, %d)' % (img_size[0], img_size[1]))
print('BASE_LR          : %s' % BASE_LR)
print('MAX_LR           : %s' % MAX_LR)
print('warmup_epochs:   : %d' % warmup_epochs)
print('device           : %s' % device)
print('log dir          : %s' % savepath)
print('model has {} parameters in total'.format(sum(x.numel() for x in net.parameters())))

steps_per_epoch = len(data_loader)
total_steps = steps_per_epoch * epoch_num
print("Start training...")

global_step = 0
best_epoch_loss = float('inf')

for epoch in range(epoch_num):
    epoch_loss = 0
    epoch_seg_loss = 0
    epoch_cons_loss = 0
    epoch_dice_score = []

    # total_pos: total number of records that have position label
    total_pos_correct = 0
    total_pos = 0

    # total_dmg: total number of records that have damage label
    total_dmg_correct = 0
    total_dmg = 0

    for i, data in enumerate(data_loader):     
        img, mask, position_label, damage_label, segment_weight = data

        num_records_have_pos = (position_label != -1).sum().item()
        total_pos += num_records_have_pos

        num_records_have_dmg = (damage_label != -1).sum().item()
        total_dmg += num_records_have_dmg

        net.train(True)   
        if device == 'cuda':
            img = img.float().cuda()
            mask = (mask[0].long().cuda(), mask[1].float().cuda())
            position_label = position_label.cuda()
            damage_label = damage_label.cuda()
            segment_weight = segment_weight.cuda()

        lr = get_warmup_cosine_lr(BASE_LR, MAX_LR, global_step, total_steps, steps_per_epoch, warmup_epochs=warmup_epochs)
        optimizer.param_groups[0]['lr'] = 0.1 * lr
        optimizer.param_groups[1]['lr'] = lr

        pos_out, dmg_out, seg_out = net(img)
        
        
        loss3 = seg_loss(seg_out, mask, segment_weight)
        loss4 = consistency_loss(seg_out, segment_weight)
        
        loss = loss3 + loss4 * 5
        
        epoch_loss += loss
        epoch_seg_loss += loss3
        epoch_cons_loss += loss4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            net.eval()
            score = dice_score(seg_out, mask[1], segment_weight)
            epoch_dice_score.append(score.item())

        sw.add_scalar('lr', lr, global_step=global_step)
        sw.add_scalar('detail_loss', loss, global_step=global_step)
        sw.add_scalar('seg_loss', loss3, global_step=global_step)

        if global_step % save_freq == 0 or global_step == total_steps-1:
            torch.save(net.state_dict(), savepath + '/model-last.pt')

        if global_step % 10 == 0:     
            msg = '%s| %s | step:%d/%d/%d | lr=%.6f | loss=%.6f | seg_loss=%.6f | cons_loss=%.6f | dice=%.6f' % (savepath, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), global_step+1, epoch+1, epoch_num, optimizer.param_groups[0]['lr'], loss.item(), loss3.item(), loss4.item(), score)
            print(msg)

        global_step += 1

        
    sw1.add_scalar('loss', epoch_loss, global_step-1)
    sw4.add_scalar('loss', epoch_seg_loss, global_step-1)
    sw5.add_scalar('loss', epoch_cons_loss, global_step-1)

    sw.add_scalar('dice score', sum(epoch_dice_score)/len(epoch_dice_score), global_step-1)


    if epoch_loss < best_epoch_loss:
        print('Loss decreases from %f to %f, saving new best...' % (best_epoch_loss, epoch_loss))
        best_epoch_loss = epoch_loss
        torch.save(net.state_dict(), savepath + '/model-best.pt')

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

from loss import DiceCELoss, WeightedPosCELoss, ConsitencyLoss, DiceBCELoss, WeightedBCELoss
from backboned_unet import Unet
from learning_rate import get_warmup_cosine_lr
from data_loader import OldData, NewData
from utils import GetItem, GetItemBinary
from score import DiceScore


# ------- Define command line params --------
parser = argparse.ArgumentParser(description='UG classification')
parser.add_argument('--metadata-file', help='Specify the json file contains data location', default='/mnt/quanhd/DoAn/unet/dataset/data_dir_endounet.json')
args = parser.parse_args()
metadata_file = args.metadata_file


# ------- 0. set hyper parameters --------
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = 'cuda' if torch.cuda.is_available() == True else 'cpu'
BASE_LR = 1e-6
MAX_LR = 1e-3
warmup_epochs = 2
batch = 16
epoch_num = 50
save_freq = 200
img_size = (480, 480)
pretrained = False
ROOT_PATH = "/mnt/tuyenld/data/endoscopy"

# # ------- 1. define tensorboard writer --------
savepath = './log/' + str(datetime.now().strftime('%Y%m%d%H%M%S'))
sw = SummaryWriter(savepath + '/main')
sw1 = SummaryWriter(savepath + '/epoch_loss')
sw2 = SummaryWriter(savepath + '/epoch_pos_loss')
sw3 = SummaryWriter(savepath + '/epoch_dmg_loss')
sw4 = SummaryWriter(savepath + '/epoch_seg_loss')
sw5 = SummaryWriter(savepath + '/epoch_cons_loss')
sw6 = SummaryWriter(savepath + '/epoch_hp')


# ------- 2. define data loader --------
d = NewData(metadata_file=metadata_file, img_size=img_size, segmentation_classes=5, mode="train", root_path=ROOT_PATH)
train_data_loader = DataLoader(dataset=d, batch_size=batch, shuffle=True)

d = NewData(metadata_file=metadata_file, img_size=img_size, segmentation_classes=5, mode="test", root_path=ROOT_PATH)
val_data_loader = DataLoader(dataset=d, batch_size=batch, shuffle=False)


# ------- 3. define loss & score function --------
if device == 'cuda':
    cls_loss = WeightedPosCELoss().cuda()
    bi_cls_loss = WeightedBCELoss().cuda()
    # seg_loss = DiceCELoss().cuda()
    seg_loss = DiceBCELoss().cuda()
    consistency_loss = ConsitencyLoss().cuda()
    get_item = GetItem().cuda()
    get_item_binary = GetItemBinary().cuda()
    dice_score = DiceScore().cuda()
else:
    cls_loss = WeightedPosCELoss()
    bi_cls_loss = WeightedBCELoss()
    # seg_loss = DiceCELoss()
    seg_loss = DiceBCELoss()
    consistency_loss = ConsitencyLoss()
    dice_score = DiceScore()
    get_item = GetItem()
    get_item_binary = GetItemBinary()


# ------- 4. define model --------
net = Unet(classes=1, position_classes=10, damage_classes=6, backbone_name='resnet50', pretrained=pretrained)
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

steps_per_epoch = len(train_data_loader)
total_steps = steps_per_epoch * epoch_num
print("Start training...")

global_step = 0
best_epoch_loss = float('inf')

for epoch in range(epoch_num):
    # if epoch == 35:
    #     break
    epoch_loss = 0
    epoch_pos_loss = 0
    epoch_dmg_loss = 0
    epoch_seg_loss = 0
    epoch_hp_loss = 0
    epoch_cons_loss = 0
    epoch_dice_score = []

    # total_pos: total number of records that have position label
    total_pos_correct = 0
    total_pos = 0

    # total_dmg: total number of records that have damage label
    total_dmg_correct = 0
    total_dmg = 0

    # total_hp: total number of records that have hp label
    total_hp_correct = 0
    total_hp = 0

    for i, data in enumerate(train_data_loader):     
        img, mask, position_label, damage_label, segment_weight, hp_label = data

        num_records_have_pos = (position_label != -1).sum().item()
        total_pos += num_records_have_pos

        num_records_have_dmg = (damage_label != -1).sum().item()
        total_dmg += num_records_have_dmg

        num_records_have_hp = (hp_label != -1).sum().item()
        total_hp += num_records_have_hp

        net.train(True)   
        if device == 'cuda':
            img = img.float().cuda()
            # mask = (mask[0].long().cuda(), mask[1].float().cuda())
            mask = mask.float().cuda()
            position_label = position_label.cuda()
            damage_label = damage_label.cuda()
            segment_weight = segment_weight.cuda()
            hp_label = hp_label.float().cuda()

        lr = get_warmup_cosine_lr(BASE_LR, MAX_LR, global_step, total_steps, steps_per_epoch, warmup_epochs=warmup_epochs)
        optimizer.param_groups[0]['lr'] = 0.1 * lr
        optimizer.param_groups[1]['lr'] = lr

        pos_out, dmg_out, hp_out, seg_out = net(img)
        
        loss1 = cls_loss(pos_out, position_label)
        loss2 = cls_loss(dmg_out, damage_label)
        loss3 = seg_loss(seg_out, mask, segment_weight)
        loss4 = bi_cls_loss(hp_out, hp_label)

        # cons_loss = consistency_loss(seg_out, segment_weight)
        
        # loss = loss1 + loss2 + loss3 + loss4 + cons_loss * 5
        loss = loss1 + loss2 + loss3 + loss4
        
        epoch_loss += loss
        epoch_pos_loss += loss1
        epoch_dmg_loss += loss2
        epoch_seg_loss += loss3
        epoch_hp_loss += loss4
        
        # epoch_cons_loss += loss4
        total_pos_correct += get_item(pos_out, position_label)
        total_dmg_correct += get_item(dmg_out, damage_label)
        total_hp_correct += get_item_binary(hp_out, hp_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sw.add_scalar('lr', lr, global_step=global_step)
        sw.add_scalar('detail_loss', loss, global_step=global_step)
        sw.add_scalar('seg_loss', loss3, global_step=global_step)

        global_step += 1

    epoch_pos_acc = np.nan if total_pos == 0 else total_pos_correct/total_pos
    epoch_dmg_acc = np.nan if total_dmg == 0 else total_dmg_correct/total_dmg 
    epoch_hp_acc = np.nan if total_hp == 0 else total_hp_correct/total_hp
        
    sw1.add_scalar('train_epoch_loss', epoch_loss, i)
    sw2.add_scalar('train_epoch_pos_loss', epoch_pos_loss, i)
    sw3.add_scalar('train_epoch_dmg_loss', epoch_dmg_loss, i)
    sw4.add_scalar('train_epoch_seg_loss', epoch_seg_loss, i)
    sw6.add_scalar('train_epoch_hp_loss', epoch_hp_loss, i)

    sw2.add_scalar('train_epoch_pos_acc', epoch_pos_acc, i)
    sw3.add_scalar('train_epoch_dmg_acc', epoch_dmg_acc, i)
    sw6.add_scalar('train_epoch_hp_acc', epoch_hp_acc, i)

    epoch_loss = 0
    epoch_pos_loss = 0
    epoch_dmg_loss = 0
    epoch_seg_loss = 0
    epoch_hp_loss = 0
    epoch_cons_loss = 0
    epoch_dice_score = []

    # total_pos: total number of records that have position label
    total_pos_correct = 0
    total_pos = 0

    # total_dmg: total number of records that have damage label
    total_dmg_correct = 0
    total_dmg = 0

    # total_hp: total number of records that have hp label
    total_hp_correct = 0
    total_hp = 0

    for i, data in enumerate(val_data_loader):     
        img, mask, position_label, damage_label, segment_weight, hp_label = data

        num_records_have_pos = (position_label != -1).sum().item()
        total_pos += num_records_have_pos

        num_records_have_dmg = (damage_label != -1).sum().item()
        total_dmg += num_records_have_dmg

        num_records_have_hp = (hp_label != -1).sum().item()
        total_hp += num_records_have_hp
   
        if device == 'cuda':
            img = img.float().cuda()
            # mask = (mask[0].long().cuda(), mask[1].float().cuda())
            mask = mask.float().cuda()
            position_label = position_label.cuda()
            damage_label = damage_label.cuda()
            segment_weight = segment_weight.cuda()
            hp_label = hp_label.float().cuda()
        with torch.no_grad():
            net.eval()
            pos_out, dmg_out, hp_out, seg_out = net(img)
        
        loss1 = cls_loss(pos_out, position_label)
        loss2 = cls_loss(dmg_out, damage_label)
        loss3 = seg_loss(seg_out, mask, segment_weight)
        loss4 = bi_cls_loss(hp_out, hp_label)

        loss = loss1 + loss2 + loss3 + loss4
        
        epoch_loss += loss
        epoch_pos_loss += loss1
        epoch_dmg_loss += loss2
        epoch_seg_loss += loss3
        epoch_hp_loss += loss4
        
        # epoch_cons_loss += loss4
        total_pos_correct += get_item(pos_out, position_label)
        total_dmg_correct += get_item(dmg_out, damage_label)
        total_hp_correct += get_item_binary(hp_out, hp_label)

        score = dice_score(seg_out, mask, segment_weight)
        epoch_dice_score.append(score.item())

        sw.add_scalar('lr', lr, global_step=global_step)
        sw.add_scalar('detail_loss', loss, global_step=global_step)
        sw.add_scalar('seg_loss', loss3, global_step=global_step)


    epoch_pos_acc = np.nan if total_pos == 0 else total_pos_correct/total_pos
    epoch_dmg_acc = np.nan if total_dmg == 0 else total_dmg_correct/total_dmg 
    epoch_hp_acc = np.nan if total_hp == 0 else total_hp_correct/total_hp
        
    sw1.add_scalar('val_epoch_loss', epoch_loss, global_step-1)
    sw2.add_scalar('val_epoch_pos_loss', epoch_pos_loss, global_step-1)
    sw3.add_scalar('val_epoch_dmg_loss', epoch_dmg_loss, global_step-1)
    sw4.add_scalar('val_epoch_seg_loss', epoch_seg_loss, global_step-1)
    sw6.add_scalar('val_epoch_hp_loss', epoch_hp_loss, global_step-1)

    sw.add_scalar('val dice score', sum(epoch_dice_score)/len(epoch_dice_score), global_step-1)

    sw2.add_scalar('val_epoch_pos_acc', epoch_pos_acc, global_step-1)
    sw3.add_scalar('val_epoch_dmg_acc', epoch_dmg_acc, global_step-1)
    sw6.add_scalar('val_epoch_hp_acc', epoch_hp_acc, global_step-1)

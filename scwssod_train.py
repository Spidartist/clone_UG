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

from loss import DiceBCELoss
from scwssod_net import NetAgg
from learning_rate import get_warmup_cosine_lr
from data_loader import Data
from utils import GetItem
from score import DiceScore


# ------- Define command line params --------
parser = argparse.ArgumentParser(description='UG classification')
parser.add_argument('--test-fold', help='Define fold id for testing, id is between 0 and 4', default=0)
args = parser.parse_args()
test_fold = int(args.test_fold)


# ------- 0. set hyper parameters --------
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = 'cuda' if torch.cuda.is_available() == True else 'cpu'
BASE_LR = 1e-6
MAX_LR = 1e-3
warmup_epochs = 2
batch = 16
epoch_num = 50
save_freq = 200
img_size = (320, 320)


# ------- 1. define tensorboard writer --------
savepath = './log/' + str(datetime.now().strftime('%Y%m%d%H%M%S'))
sw = SummaryWriter(savepath)


# ------- 2. define data loader --------
d = Data(test_fold=test_fold, img_size=img_size)
data_loader = DataLoader(dataset=d, batch_size=batch, shuffle=True)


# ------- 3. define loss & score function --------
if device == 'cuda':
    loss1 = nn.CrossEntropyLoss().cuda()
    loss2 = DiceBCELoss().cuda()
    get_item = GetItem().cuda()
    dice_score = DiceScore().cuda()
else:
    loss1 = nn.CrossEntropyLoss()
    loss2 = DiceBCELoss()
    dice_score = DiceScore()
    get_item = GetItem()


# ------- 4. define model --------
net = NetAgg()
net.train(True)
if device == 'cuda':
    net.cuda()

base, head = [], []
for name, param in net.named_parameters():
    if 'bkbone' in name:
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
    epoch_dice_score = []
    total_correct = 0

    for i, data in enumerate(data_loader):     
        img, mask, label, segment_weight = data
        net.train(True)   
        if device == 'cuda':
            img = img.float().cuda()
            mask = mask.float().cuda()
            label = label.cuda()
            segment_weight = segment_weight.cuda()

        lr = get_warmup_cosine_lr(BASE_LR, MAX_LR, global_step, total_steps, steps_per_epoch, warmup_epochs=warmup_epochs)
        optimizer.param_groups[0]['lr'] = 0.1 * lr
        optimizer.param_groups[1]['lr'] = lr

        out2, out3, out4, out5, cls_res = net(img)
        out2 = torch.sigmoid(out2)
        out3 = torch.sigmoid(out3)
        out4 = torch.sigmoid(out4)
        out5 = torch.sigmoid(out5)

        cls_loss = loss1(cls_res, label)
        seg_loss_2 = loss2(out2, mask, segment_weight)
        seg_loss_3 = loss2(out3, mask, segment_weight)
        seg_loss_4 = loss2(out4, mask, segment_weight)
        seg_loss_5 = loss2(out5, mask, segment_weight)
        loss = cls_loss * 0.3 + seg_loss_2*1 + seg_loss_3*0.8 + seg_loss_4*0.6 + seg_loss_5*0.4
        
        epoch_loss += loss
        total_correct += get_item(cls_res, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            net.eval()
            score = dice_score(out2, mask, segment_weight)
            epoch_dice_score.append(score.item())

        sw.add_scalar('lr', lr, global_step=global_step)
        sw.add_scalar('loss', loss, global_step=global_step)
        sw.add_scalar('acc', total_correct/(i+1)/batch, global_step=global_step)

        if global_step % save_freq == 0 or global_step == total_steps-1:
            torch.save(net.state_dict(), savepath + '/model-last.pt')

        if global_step % 10 == 0:            
            msg = '%s| %s | step:%d/%d/%d | lr=%.6f | loss=%.6f | acc=%.6f | dice=%.6f' % (savepath, datetime.now(),  global_step+1, epoch+1, epoch_num, optimizer.param_groups[0]['lr'], loss.item(), total_correct/(i+1)/batch, score.item())
            print(msg)

        global_step += 1
        
    sw.add_scalar('epoch_loss', epoch_loss, global_step-1)
    sw.add_scalar('dice score', sum(epoch_dice_score)/len(epoch_dice_score), global_step-1)
    sw.add_scalar('epoch_acc', total_correct/len(d), global_step-1)

    if epoch_loss < best_epoch_loss:
        print('Loss decreases from %f to %f, saving new best...' % (best_epoch_loss, epoch_loss))
        best_epoch_loss = epoch_loss
        torch.save(net.state_dict(), savepath + '/model-best.pt')

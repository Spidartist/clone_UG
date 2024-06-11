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

from model import CustomModel as ClsCustomModel
from learning_rate import get_warmup_cosine_lr
from hp_data_loader import Data


class GetItem(nn.Module):
    def __init__(self):
        super().__init__()

    
    def forward(self, preds, labels):
        # preds = torch.sigmoid(preds)
        preds[preds <= 0.5] = 0.0
        preds[preds > 0.5] = 1.0

        return preds.eq(labels).sum().item()


# ------- Define command line params --------
parser = argparse.ArgumentParser(description='UG classification')
parser.add_argument('--test-fold', help='Define fold id for testing, id is between 0 and 4', default=0)
# parser.add_argument('--test-color-mode', help='Define fold id for testing, id is between 0 and 4', default='WLI')
# parser.add_argument('--metadata-file', help='Specify the json file contains data location', default='dir.json')
parser.add_argument('--model', help='Specify the model', default='vgg19')
args = parser.parse_args()
test_fold = int(args.test_fold)
model_name = args.model



# ------- 0. set hyper parameters --------
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = 'cuda' if torch.cuda.is_available() == True else 'cpu'
BASE_LR = 1e-6
MAX_LR = 1e-3
warmup_epochs = 2
batch = 16
epoch_num = 30
save_freq = 200
img_size = (480, 480)


# ------- 1. define tensorboard writer --------
savepath = '/mnt/manhnd/log2/' + str(datetime.now().strftime('%Y%m%d%H%M%S'))
sw = SummaryWriter(savepath)


# ------- 2. define data loader --------
d = Data(test_fold=test_fold, img_size=img_size)
data_loader = DataLoader(dataset=d, batch_size=batch, shuffle=True, drop_last=True)


# ------- 3. define loss & score function --------
if device == 'cuda':
    # loss_func = nn.CrossEntropyLoss().cuda()
    loss_func = nn.BCELoss().cuda()
    get_item = GetItem().cuda()
else:
    # loss_func = nn.CrossEntropyLoss()
    loss_func = nn.BCELoss()
    get_item = GetItem()


# ------- 4. define model --------
net = ClsCustomModel(name=model_name, pretrained=True, out_classes=1)
net.train(True)
if device == 'cuda':
    net.cuda()


# ------- 5. define optimizer --------
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


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
        net.train(True)   
        if device == 'cuda':
            input = data[0].float().cuda()
            label = data[1].float().cuda()
        else:
            input = data[0].float()
            label = data[1].float()

        label = label.reshape(batch, 1)
        label.requires_grad = False

        lr = get_warmup_cosine_lr(BASE_LR, MAX_LR, global_step, total_steps, steps_per_epoch, warmup_epochs=warmup_epochs)
        optimizer.param_groups[0]['lr'] = lr

        out = net(input)
        # out = 
        loss = loss_func(torch.sigmoid(out), label)
        epoch_loss += loss
        total_correct += get_item(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sw.add_scalar('lr', lr, global_step=global_step)
        sw.add_scalar('loss', loss, global_step=global_step)
        sw.add_scalar('acc', total_correct/(i+1)/batch, global_step=global_step)

        if global_step % save_freq == 0 or global_step == total_steps-1:
            torch.save(net.state_dict(), savepath + '/model-last.pt')

        if global_step % 10 == 0:            
            msg = '%s| %s | step:%d/%d/%d | lr=%.6f | loss=%.6f | acc=%.6f' % (savepath, datetime.now(),  global_step+1, epoch+1, epoch_num, optimizer.param_groups[0]['lr'], loss.item(), total_correct/(i+1)/batch)
            print(msg)

        global_step += 1
        
    sw.add_scalar('epoch_loss', epoch_loss, global_step-1)
    sw.add_scalar('epoch_acc', total_correct/len(d), global_step-1)

    if epoch_loss < best_epoch_loss:
        print('Loss decreases from %f to %f, saving new best...' % (best_epoch_loss, epoch_loss))
        best_epoch_loss = epoch_loss
    #     torch.save(net.state_dict(), savepath + '/model-best.pt')

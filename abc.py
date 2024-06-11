import argparse
import logging
import os

parser = argparse.ArgumentParser(description='UG classification testing')
parser.add_argument('--checkpoint-dir', help='Specify a folder contains model checkpoint', required=True)
parser.add_argument('--checkpoint-name', help='Specify the checkpoint name', default='model-last.pt')
parser.add_argument('--test-fold', type=int, help='Define fold id for testing, id is between 0 and 4', default=0)
parser.add_argument('--visualize', type=bool, help='Visualize pred maps or not', default=False)
args = parser.parse_args()

test_fold = args.test_fold
checkpoint_dir = args.checkpoint_dir
checkpoint_name = args.checkpoint_name
is_visualized = args.visualize

from data_loader import Data
from model import Resnet
from backboned_unet import Unet
from scwssod_net import NetAgg
from score import DiceScore
from torch.utils.data import DataLoader
from utils import GetItem
import torch
from tqdm import tqdm
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import cv2
import numpy as np 
import shutil
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = 'cuda' if torch.cuda.is_available() == True else 'cpu'

class DiceScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceScore, self).__init__()

    def dice_score(self, input, target, classes, ignore_index=-100, smooth=1e-12):
        """ Functional dice score calculation on multiple classes. """

        target = target.long().unsqueeze(1)

        # getting mask for valid pixels, then converting "void class" to background
        valid = target != ignore_index
        target[target == ignore_index] = 0
        valid = valid.float()

        # converting to onehot image with class channels
        onehot_target = torch.LongTensor(target.shape[0], classes, target.shape[-2], target.shape[-1]).zero_().cuda()
        onehot_target.scatter_(1, target, 1)  # write ones along "channel" dimension
        # classes_in_image = onehot_gt_tensor.sum([2, 3]) > 0
        onehot_target = onehot_target.float()

        # keeping the valid pixels only
        onehot_target = onehot_target * valid
        input = input * valid

        dice = 2 * (input * onehot_target).sum([2, 3]) / ((input**2).sum([2, 3]) + (onehot_target**2).sum([2, 3]) + smooth)
        return dice.mean(dim=1)

    def forward(self, inputs, targets, smooth=1):
        
        return self.dice_score(inputs, targets, 4)

get_item = GetItem()
dice_score = DiceScore()

if device == 'cuda':
    get_item = GetItem().cuda()
    dice_score = dice_score.cuda()
    

d = Data(test_fold=test_fold, mode='test', img_size=(480, 480))
loader = DataLoader(d, batch_size=1)

net = Unet(classes=4, position_classes=10, damage_classes=4)
path = './log/%s/%s' % (checkpoint_dir, checkpoint_name)
if device == 'cuda':
    state_dict = torch.load(path)
else:
    state_dict = torch.load(path, map_location=torch.device('cpu'))
net.load_state_dict(state_dict)
net.train(False)
if device == 'cuda':
    net.cuda()
net.eval()

count = 0
dice = 0

pos_confusion_matrix = [[0 for i in range(10)] for i in range(10)]
dmg_confusion_matrix = [[0 for i in range(4)] for i in range(4)]

dice_split = {
    '/DATA/viem_thuc_quan': [0, 0],
    '20211021 UT thuc quan': [0, 0],
    '20211116_stomach_cancer_unverified': [0, 0]
}

pred_map_dir = '%s_fold_%d' % (checkpoint_dir, test_fold)
if is_visualized:
    if pred_map_dir in os.listdir('./pred_maps'):
        shutil.rmtree('./pred_maps/' + pred_map_dir)
    os.mkdir('./pred_maps/' + pred_map_dir)

total_pos = 0
total_pos_correct = 0

total_dmg = 0
total_dmg_correct = 0

with torch.no_grad():
    for i in tqdm(range(len(d.samples))):
        img_path, _, _, _, _ = d.samples[i]
        img, mask, position_label, damage_label, segment_weight = d.__getitem__(i)

        segment_weight = torch.tensor([segment_weight])

        position_label = torch.tensor([position_label])
        damage_label = torch.tensor([damage_label])
        if device == 'cuda':
            img = img.float().cuda()
            mask = mask.float().cuda()
            position_label = position_label.cuda()
            damage_label = damage_label.cuda()
            segment_weight = segment_weight.cuda()
            
        img = img.reshape(1, 3, 480, 480)
        pos_out, dmg_out, seg_out = net(img)
        # out2, out3, out4, seg_out, pos_out, dmg_out = net(img)

        # seg_out = seg_out.argmax(dim=1)

        total_pos += (position_label != -1).sum().item()
        total_pos_correct += get_item(pos_out, position_label)
        
        total_dmg += (damage_label != -1).sum().item()
        total_dmg_correct += get_item(dmg_out, damage_label)
        mask = mask.reshape(1, 480, 480)
        score = dice_score(seg_out, mask)
        if segment_weight.sum() == 1:
            dice += score
            if '/DATA/viem_thuc_quan' in img_path:
                dice_split['/DATA/viem_thuc_quan'][0] += score
                dice_split['/DATA/viem_thuc_quan'][1] += 1
            elif '20211021 UT thuc quan' in img_path:
                dice_split['20211021 UT thuc quan'][0] += score
                dice_split['20211021 UT thuc quan'][1] += 1
            else:
                dice_split['20211116_stomach_cancer_unverified'][0] += score
                dice_split['20211116_stomach_cancer_unverified'][1] += 1

        count += segment_weight.sum()

        pos_predicted_label = pos_out.argmax(dim=1)
        dmg_predicted_label = dmg_out.argmax(dim=1)

        if position_label.item() != -1:
            pos_confusion_matrix[position_label][pos_predicted_label] += 1

        if damage_label.item() != -1:
            dmg_confusion_matrix[damage_label][dmg_predicted_label] += 1

        if is_visualized:
            if segment_weight.sum() != 1:
                continue

            img_name = img_path.split('/')[-1].split('.')[0] + '.png'
            img = img.reshape(3, 480, 480)
            seg_out = seg_out.reshape(1, 480, 480)

            img = img.permute(1, 2, 0) * 255
            mask = mask.permute(1, 2, 0) * 255
            pred = seg_out.permute(1, 2, 0) * 255

            pred[pred <= 128] = 0
            pred[pred > 128] = 255

            mask = torch.cat([mask, mask, mask], 2)
            pred = torch.cat([pred, pred, pred], 2)

            img = np.array(img.to('cpu'), dtype=int)
            mask = np.array(mask.to('cpu'), dtype=int)
            pred = np.array(pred.to('cpu'), dtype=int)

            new_img = cv2.hconcat([img, mask, pred])

            res = cv2.imwrite('./pred_maps/' + pred_map_dir + '/' + img_name, new_img)

        
pos_df = pd.DataFrame(pos_confusion_matrix, index = [i for i in range(10)],
                  columns = [i for i in range(10)])
print(pos_df)                

dmg_df = pd.DataFrame(dmg_confusion_matrix, index = [i for i in range(4)],
                  columns = [i for i in range(4)])
print(dmg_df)                

print('pos_acc: %.6f, dmg_acc: %.6f, dice: %.6f' % (float(total_pos_correct/total_pos), float(total_dmg_correct/total_dmg), dice / count))

# print(dice_split)

for key in dice_split:
    print('%s dice: %.6f' % (key, dice_split[key][0] / dice_split[key][1]))

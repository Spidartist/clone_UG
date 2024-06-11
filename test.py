import argparse
import logging
import os
import shutil

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

from data_loader import Data, OldData
# from model import Resnet
from backboned_unet import Unet
from scwssod_net import NetAgg
from torch.utils.data import DataLoader
from utils import GetItem, GetItemBinary
import torch
from tqdm import tqdm
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import cv2
import numpy as np 
import shutil
import torch.nn as nn

site_mapping = {
    0: 'Hau hong',
    1: 'Thuc quan',
    2: 'Tam vi',
    3: 'Than vi',
    4: 'Phinh vi',
    5: 'Hang vi',
    6: 'Bo cong lon',
    7: 'Bo cong nho',
    8: 'Hanh ta trang',
    9: 'Ta trang'
}

lesion_mapping = {
    0: 'Non lesion',
	1: 'UT thuc quan',
	2: 'Viem thuc quan',
	3: 'Viem loet hanh ta trang',
	4: 'UT da day'
}


class DiceScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceScore, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # print(inputs.size(), targets.size())
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)

        inputs = torch.sigmoid(torch.flatten(inputs))
        inputs[inputs < 0.5] = 0
        inputs[inputs >= 0.5] = 1
        targets = torch.flatten(targets.float())
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return torch.nansum(dice)

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = 'cuda' if torch.cuda.is_available() == True else 'cpu'

get_item = GetItem()
get_item_binary = GetItemBinary()
dice_score = DiceScore()

if device == 'cuda':
    get_item = GetItem().cuda()
    get_item_binary = GetItemBinary().cuda()
    dice_score = dice_score.cuda()
    

d = OldData(test_fold=test_fold, mode='test', img_size=(480, 480), segmentation_classes=6)
loader = DataLoader(d, batch_size=1)

net = Unet(classes=1, position_classes=10, damage_classes=6, backbone_name='resnet50')
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
print('model has {} parameters in total'.format(sum(x.numel() for x in net.parameters())))


count = 0
dice = 0

pos_confusion_matrix = [[0 for i in range(10)] for i in range(10)]
dmg_confusion_matrix = [[0 for i in range(6)] for i in range(6)]
hp_confusion_matrix = [[0 for i in range(2)] for i in range(2)]

dice_split = {
    'viem_thuc_quan_20220110': [0, 0],
    '20211021 UT thuc quan': [0, 0],
    'viem_loet_hanh_ta_trang_20220110': [0, 0],
    'viem_da_day': [0, 0],
    'ung_thu_da_day_20220110': [0, 0]
}

pred_map_dir = '%s_fold_%d' % (checkpoint_dir, test_fold)
if is_visualized:
    if pred_map_dir in os.listdir('./pred_maps'):
        shutil.rmtree('./pred_maps/' + pred_map_dir)
    os.mkdir('./pred_maps/' + pred_map_dir)
    os.mkdir('./pred_maps/' + pred_map_dir + '/1')
    os.mkdir('./pred_maps/' + pred_map_dir + '/2')
    os.mkdir('./pred_maps/' + pred_map_dir + '/3')
    os.mkdir('./pred_maps/' + pred_map_dir + '/4')
    os.mkdir('./pred_maps/' + pred_map_dir + '/5')

total_pos = 0
total_pos_correct = 0

total_dmg = 0
total_dmg_correct = 0

total_hp = 0
total_hp_correct = 0

check_list = []

with torch.no_grad():
    for i in tqdm(range(len(d.samples))):
        img_path, _, _, _, _, _ = d.samples[i]
        img, mask, position_label, damage_label, segment_weight, hp_label = d.__getitem__(i)

        segment_weight = torch.tensor([segment_weight])

        position_label = torch.tensor([position_label])
        damage_label = torch.tensor([damage_label])
        hp_label = torch.tensor([hp_label])
        if device == 'cuda':
            img = img.float().cuda()
            # mask = [mask[0].long().cuda(), mask[1].float().cuda()]
            mask = mask.float().cuda()
            position_label = position_label.cuda()
            damage_label = damage_label.cuda()
            segment_weight = segment_weight.cuda()
            hp_label = hp_label.float().cuda()
            
        img = img.reshape(1, 3, 480, 480)
        # pos_out, dmg_out, seg_out = net(img)
        pos_out, dmg_out, hp_out, seg_out = net(img)



        # seg_out[seg_out < 0.5] = 0
        # seg_out[seg_out >= 0.5] = 1
        # out2, out3, out4, seg_out, pos_out, dmg_out = net(img)

        # seg_out = torch.sigmoid(seg_out)

        total_pos += (position_label != -1).sum().item()
        total_pos_correct += get_item(pos_out, position_label)
        
        total_dmg += (damage_label != -1).sum().item()
        total_dmg_correct += get_item(dmg_out, damage_label)

        total_hp += (hp_label != -1).sum().item()
        total_hp_correct += get_item_binary(hp_out, hp_label)

        # mask[1] = mask[1].unsqueeze(0)
        mask = mask.unsqueeze(0)
        score = dice_score(seg_out, mask)
        if segment_weight.sum() != 0:
            dice += score
            if 'viem_thuc_quan_20220110' in img_path:
                dice_split['viem_thuc_quan_20220110'][0] += score
                dice_split['viem_thuc_quan_20220110'][1] += 1
            elif '20211021 UT thuc quan' in img_path:
                dice_split['20211021 UT thuc quan'][0] += score
                dice_split['20211021 UT thuc quan'][1] += 1
            elif 'viem_loet_hanh_ta_trang_20220110' in img_path:
                dice_split['viem_loet_hanh_ta_trang_20220110'][0] += score
                dice_split['viem_loet_hanh_ta_trang_20220110'][1] += 1
            elif 'viem_da_day' in img_path:
                dice_split['viem_da_day'][0] += score
                dice_split['viem_da_day'][1] += 1
            else:
                dice_split['ung_thu_da_day_20220110'][0] += score
                dice_split['ung_thu_da_day_20220110'][1] += 1

        count += 1 if segment_weight.sum() != 0 else 0

        pos_predicted_label = torch.softmax(pos_out, dim=1).argmax(dim=1)
        dmg_predicted_label = torch.softmax(dmg_out, dim=1).argmax(dim=1)
        hp_predicted_label = 1 if torch.sigmoid(torch.flatten(hp_out))[0] >= 0.5 else 0


        if position_label.item() != -1:
            pos_confusion_matrix[position_label][pos_predicted_label] += 1
            ###
            #if position_label != pos_predicted_label:
            #    print('pos', img_path, 'actual: ', position_label.item(), 'predict: ', pos_predicted_label.item())
            #    filename = img_path.split('/')[-1]
            #    shutil.copy(img_path, './wrong-prediction/' + filename)
            #    actual_position = site_mapping[position_label.item()]
            #    prediction_position = site_mapping[pos_predicted_label.item()]
            #    with open('wrong-prediction.csv', 'a+') as f:
            #        f.write('%s,%s,%s,%s,%s\n' % (img_path, filename, 'wrong site', actual_position, prediction_position))
            ###

        if damage_label.item() != -1:
            dmg_confusion_matrix[damage_label][dmg_predicted_label] += 1
            ###
            #if damage_label != dmg_predicted_label:
            #    print('dmg', img_path, 'actual: ', damage_label.item(), 'predict: ', dmg_predicted_label.item())
            #    filename = img_path.split('/')[-1]
            #    shutil.copy(img_path, './wrong-prediction/' + filename)
            #    actual_dmg = lesion_mapping[damage_label.item()]
            #    prediction_dmg = lesion_mapping[dmg_predicted_label.item()]
            #    with open('wrong-prediction.csv', 'a+') as f:
            #        f.write('%s,%s,%s,%s,%s\n' % (img_path, filename, 'wrong lesion', actual_dmg, prediction_dmg))
            ###

        if hp_label[0].item() != -1:
            hp_confusion_matrix[int(hp_label)][hp_predicted_label] += 1


        if is_visualized:
            if segment_weight.sum() == 0:
                continue
            

            img_name = img_path.split('/')[-1].split('.')[0] + '.png'
            img = img.reshape(3, 480, 480)
            # seg_out = seg_out[segment_weight].reshape(1, 480, 480)
            seg_out = seg_out.reshape(1, 480, 480)

            img = img.permute(1, 2, 0) * 255
            # mask = mask[1][0][segment_weight].permute(1, 2, 0) * 255
            mask = mask[0].permute(1, 2, 0) * 255
            pred = torch.sigmoid(seg_out).permute(1, 2, 0) * 255


            pred[pred <= 128] = 0
            pred[pred > 128] = 255

            mask = torch.cat([mask, mask, mask], 2)
            pred = torch.cat([pred, pred, pred], 2)

            img = np.array(img.to('cpu'), dtype=int)
            mask = np.array(mask.to('cpu'), dtype=int)
            pred = np.array(pred.to('cpu'), dtype=int)

            img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            mask = cv2.copyMakeBorder(mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            pred = cv2.copyMakeBorder(pred, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

            new_img = cv2.vconcat([img, mask, pred])

            res = cv2.imwrite('./pred_maps/' + pred_map_dir + '/' + str(segment_weight.item()) + '/' + img_name, pred)

        # if segment_weight.sum() != 0:
        #     new_out = torch.sigmoid(seg_out)
        #     new_out[new_out < 0.5] = 0
        #     new_out[new_out >= 0.5] = 1
        #     check = new_out.sum() - new_out[0][segment_weight].sum()
        #     if check != 0:
        #         check_list.append([i, segment_weight.long().sum().item(), new_out[0][0].long().sum().item(), new_out[0][1].long().sum().item(), new_out[0][2].long().sum().item(), new_out[0][3].long().sum().item()])



        # print(segment_weight, score)

        # if out1.argmax().item() == label.item():
        #     logging.info('%d, %d, %f, %s' % (out1.argmax().item(), label.item(), score, img_path))
        # else:
        #     logging.warning('%d, %d, %f, %s' % (out1.argmax().item(), label.item(), score, img_path))

# logging.info('acc: %.6f, dice: %.6f' % (float(total_correct / len(d)), dice / count))
pos_df = pd.DataFrame(pos_confusion_matrix, index = [i for i in range(10)],
                  columns = [i for i in range(10)])
print(pos_df)                

dmg_df = pd.DataFrame(dmg_confusion_matrix, index = [i for i in range(6)],
                  columns = [i for i in range(6)])
print(dmg_df)   

hp_df = pd.DataFrame(hp_confusion_matrix, index = [i for i in range(2)],
                  columns = [i for i in range(2)])
print(hp_df)              

# print('pos_acc: %.6f, dmg_acc: %.6f, dice: %.6f' % (float(total_pos_correct/total_pos), float(total_dmg_correct/total_dmg), dice / count))
print('pos_acc: %.6f, dmg_acc: %.6f, hp_acc: %.6f, dice: %.6f' % (float(total_pos_correct/total_pos), float(total_dmg_correct/total_dmg), float(total_hp_correct/total_hp), dice / count))

# print(dice_split)

for key in dice_split:
    print('%s dice: %.6f' % (key, dice_split[key][0] / (dice_split[key][1]) ))

# for check in check_list:
#     print(check)

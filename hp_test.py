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

from hp_data_loader import Data
from model import CustomModel
from backboned_unet import Unet
from scwssod_net import NetAgg
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

if device == 'cuda':
    get_item = GetItem().cuda()
    
matrix = [[0, 0], [0, 0]]

d = Data(test_fold=test_fold, mode='test', img_size=(480, 480))
loader = DataLoader(d, batch_size=1)

net = CustomModel(name='densenet121', pretrained=True, out_classes=1)
path = '/mnt/manhnd/log2/%s/%s' % (checkpoint_dir, checkpoint_name)
if device == 'cuda':
    state_dict = torch.load(path)
else:
    state_dict = torch.load(path, map_location=torch.device('cpu'))
net.load_state_dict(state_dict)
net.train(False)
if device == 'cuda':
    net.cuda()
net.eval()


total = 0
total_correct = 0

with torch.no_grad():
    for i in tqdm(range(len(d.samples))):
        total += 1
        img_path, _, = d.samples[i]
        img, label = d.__getitem__(i)

        if device == 'cuda':
            img = img.float().cuda()
            
        img = img.reshape(1, 3, 480, 480)
        out = net(img)
        out = torch.sigmoid(out)
        if out <= 0.5:
            res = 0
        else:
            res = 1
        
        if res == label:
            total_correct += 1
        
        matrix[res][label] += 1

print("acc: %f" % (total_correct/total))
print(matrix)
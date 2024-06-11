from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import json
import cv2
import numpy as np
import torch
import pickle
import albumentations as A

class Data(Dataset):
    def __init__(self, metadata_file='dir.json', test_fold=0, mode='train', img_size=(320, 320)):
        # img_path = '/mnt/manhnd/DATA/stomach_hp_20220822/images/'
        # metadata_path = '/mnt/manhnd/DATA/stomach_hp_20220822/metadata/'

        # split dataset into 5 folds
        self.samples = []
        self.img_size = img_size
        self.mode = mode

        # self.negative_hp = []
        # self.positive_hp = []

        # json_files = os.listdir(metadata_path)        

        # for name in json_files:
        #     img_name = img_path + name.replace('.json', '.jpeg')
        #     with open(metadata_path + name) as f:
        #         tags = json.load(f)['image_tag_list']
        #         for tag in tags:
        #             if tag['display_name'] == 'HP dương tính':
        #                 self.positive_hp.append((img_name, 1))
        #             elif tag['display_name'] == 'HP âm tính':
        #                 self.negative_hp.append((img_name, 0))

        # for lst in [self.negative_hp, self.positive_hp]:
        #     names = sorted(lst, key=lambda x: x[0])
        #     imgs_per_fold = int(len(names)/5)
        #     if mode == 'train':
        #         names = names[:imgs_per_fold * test_fold] + names[imgs_per_fold * (test_fold+1):]
        #     else:
        #         if test_fold == 4:
        #             names = names[imgs_per_fold * test_fold:]
        #         else:
        #             names = names[imgs_per_fold * test_fold:imgs_per_fold * (test_fold+1)]
            
        #     for fn in names:
        #         self.samples.append(fn)

        with open('hp_fold', 'rb') as f:
            dic = pickle.load(f)

        if mode == 'test':
            self.samples = dic[test_fold]
        else:
            for i in range(5):
                if i != test_fold:
                    self.samples += dic[i]       


    def aug(self, image, mask):
        img_size = self.img_size
        if self.mode == 'train':
            t1 = A.Compose([A.Resize(img_size[0], img_size[1]),])
            resized = t1(image=image, mask=mask)
            image = resized['image']
            mask = resized['mask']
            t = A.Compose([                
                A.HorizontalFlip(p=0.7),
                A.VerticalFlip(p=0.7),
                A.Rotate(interpolation=cv2.BORDER_CONSTANT, p=0.7),
                A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, shift_limit=0.5, scale_limit=0.2, p=0.7),
                A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, shift_limit=0, scale_limit=(-0.1, 0.1), rotate_limit=0, p=0.35),
                A.MotionBlur(p=0.2),
                A.HueSaturationValue(p=0.2),                
            ], p=0.5)

        elif self.mode == 'test':
            t = A.Compose([
                A.Resize(img_size[0], img_size[1])
            ])

        return t(image=image, mask=mask)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, hp_label = self.samples[idx]
        # print(img_path, mask_path)

        img = cv2.imread(img_path).astype(np.float32)
        fake_mask = img

        augmented = self.aug(img, fake_mask)
        img = augmented['image']
        fake_mask = augmented['mask']

        img = torch.from_numpy(img.copy())
        img = img.permute(2, 0, 1)

        img /= 255.

        return img, hp_label

if __name__=="__main__":
    d = Data(mode='test')
    data_loader = DataLoader(dataset=d, batch_size=8, shuffle=True)
    from model import Resnet
    net = Resnet(classes=1)


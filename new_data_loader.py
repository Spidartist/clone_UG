# The replacement for old loader in segmentation

from albumentations import augmentations
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import json
import cv2
import numpy as np
import torch
import albumentations as A


class Data(Dataset):
    def __init__(self, metadata_file='dir.json', test_fold=0, mode='train', img_size=(320, 320), segmentation_classes=5):
        # split dataset into 5 folds
        self.samples = []
        self.img_size = img_size
        self.mode = mode
        self.segmentation_classes = segmentation_classes

        with open(metadata_file) as f:
            dirs = json.load(f)['dirs']
        
        for dir_info in dirs:
            type = dir_info['type']
            position_label = dir_info.get('position_label', -1)
            damage_label = dir_info.get('damage_label', -1)
            seg_label = dir_info.get('segmentation_label', 0)
            location = dir_info['location']
            img_folder_name = dir_info.get('img_folder_name', '')
            img_file_extension = dir_info.get('img_file_extension', '')
            mask_folder_name = dir_info.get('mask_folder_name', '')
            mask_file_extension = dir_info.get('mask_file_extension', '')

            if type == 'segmentation':
                files_name = os.listdir(location + '/' + img_folder_name)
                files_name = sorted(files_name)

                img_per_fold = int(len(files_name) / 5)
                if self.mode == 'train':
                    names = files_name[:img_per_fold * test_fold] + files_name[img_per_fold * (test_fold+1):]
                else:
                    names = files_name[img_per_fold * test_fold : img_per_fold * (test_fold+1)]

                for fn in names:
                    img_path = location + '/' + img_folder_name + '/' + fn
                    mask_path = location + '/' + mask_folder_name + '/' + fn.replace(img_file_extension, mask_file_extension)
                    self.samples.append([img_path, mask_path, position_label, damage_label, seg_label])

            elif type == 'classification':
                sub_dirs = os.listdir(location)
                sub_dirs = sorted(sub_dirs)
                for sub_dir in sub_dirs:
                    files_name = os.listdir(location + '/' + sub_dir)
                    files_name = sorted(files_name)

                    img_per_fold = int(len(files_name) / 5)
                    if self.mode == 'train':
                        names = files_name[:img_per_fold * test_fold] + files_name[img_per_fold * (test_fold+1):]
                    else:
                        names = files_name[img_per_fold * test_fold : img_per_fold * (test_fold+1)]

                    for fn in names:
                        img_path = location + '/' + sub_dir + '/' + fn
                        self.samples.append([img_path, None, position_label, damage_label, seg_label])


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
                # A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, shift_limit=0.5, scale_limit=0.2, p=0.7),
                # A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, shift_limit=0, scale_limit=(-0.8, 0), rotate_limit=0, p=0.35),
                # A.MotionBlur(p=0.2),
                # A.HueSaturationValue(p=0.2),                
            ], p=0.5)

        elif self.mode == 'test':
            t = A.Compose([
                A.Resize(img_size[0], img_size[1])
            ])

        return t(image=image, mask=mask)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, position_label, damage_label, seg_label = self.samples[idx]
        # print(img_path, mask_path)

        img = cv2.imread(img_path).astype(np.float32)

        if mask_path is not None:
            orin_mask = cv2.imread(mask_path).astype(np.float32)
        else:
            orin_mask = img

        augmented = self.aug(img, orin_mask)
        img = augmented['image']
        orin_mask = augmented['mask']

        img = torch.from_numpy(img.copy())
        img = img.permute(2, 0, 1)
        orin_mask = torch.from_numpy(orin_mask.copy())
        orin_mask = orin_mask.permute(2, 0, 1)

        img /= 255.
        orin_mask = orin_mask.mean(dim=0, keepdim=True)/255.

        orin_mask[orin_mask <= 0.5] = 0
        orin_mask[orin_mask > 0.5] = 1

        mask1 = orin_mask.reshape(orin_mask.shape[1], orin_mask.shape[2]).detach().clone()

        mask2 = torch.zeros(self.segmentation_classes, orin_mask.shape[1], orin_mask.shape[2])
        mask2[seg_label] = orin_mask

        if mask_path is None:
            segment_weight = 0
        else:
            segment_weight = seg_label

        return img, (mask1, mask2), position_label, damage_label, segment_weight


class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()


    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float() #if need
            self.next_target = self.next_target.float() #if need

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


if __name__=='__main__':
    d = Data(mode='test')
    loader = DataLoader(dataset=d, batch_size=16, shuffle=True)
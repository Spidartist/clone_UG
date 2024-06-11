import argparse

parser = argparse.ArgumentParser(description='UG testing on video')
parser.add_argument('--vid-name', type=str, help='Specify a folder contains model checkpoint', required=True)
args = parser.parse_args()

vid_name = args.vid_name

import cv2
import torch
import numpy as np
from backboned_unet import Unet
import os

label_mapping = {
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

dmg_mapping = {
    0: 'Khong ton thuong',
    1: 'Ung thu thuc quan',
    2: 'Viem thuc quan',
    3: 'Ung thu da day'
}

if 'out.avi' in os.listdir('.'):
    os.remove('out.avi')

device = 'cuda' if torch.cuda.is_available() == True else 'cpu'
net = Unet(classes=4, position_classes=10, damage_classes=4)
path = './log/%s/%s' % ('20211209001010', 'model-last.pt')
if device == 'cuda':
    state_dict = torch.load(path)
else:
    state_dict = torch.load(path, map_location=torch.device('cpu'))
net.load_state_dict(state_dict)
net.train(False)
if device == 'cuda':
    net.cuda()
net.eval()

cap = cv2.VideoCapture(vid_name)
vid_out_name = vid_name.split('/')[-1].split('.')[0] + '.avi'
out = cv2.VideoWriter('./pred_vid/' + vid_out_name, cv2.VideoWriter_fourcc('M','J', 'P', 'G'), 24, (1350, 1080))
# out = cv2.VideoWriter(vid_out_name, 0x7634706d, 24, (1350, 1080))

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255, 255, 255)
thickness              = 1
lineType               = 2

count = 0
with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        # if count == 1000:
        #     break

        frame = frame[:, :1350, :]

        img = cv2.resize(frame, (480, 480), interpolation = cv2.INTER_AREA)
        x = torch.from_numpy(img.copy()).permute(2, 0, 1) 
        x = x.reshape(1, 3, 480, 480)/255.

        if device == 'cuda':
            x = x.float().cuda()

        pos_out, dmg_out, seg_out = net(x)
        pos_label = pos_out[0].argmax().item()
        dmg_label = dmg_out[0].argmax().item()

        seg_out = seg_out[0][dmg_label].reshape(1, 480, 480)
        pred = seg_out.permute(1, 2, 0) * 255
        pred[pred <= 128] = 0
        pred[pred > 128] = 255
        pred = pred.reshape(480, 480)

        pred = pred.cpu().numpy().astype(np.uint8)
        frame = cv2.resize(frame, (480, 480))
        contours, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        frame = cv2.resize(frame, (1350, 1080))

        cv2.putText(frame, 'Vi tri giai phau: ' + label_mapping[pos_label], (10, 950), font, fontScale, fontColor, thickness, lineType)
        cv2.putText(frame, 'Loai ton thuong: ' + dmg_mapping[dmg_label], (10, 1000), font, fontScale, fontColor, thickness, lineType)
        
        wr = out.write(frame)

        # pred = torch.cat([pred, pred, pred], 2)
        # pred = pred.numpy().astype(np.uint8)
        # pred = cv2.resize(pred, img_size)
        

        # cv2.imshow('test', pred)
        # cv2.imshow('test', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break   
        
        # cv2.waitKey(0)
        # break
    
cap.release()
out.release()

cv2.destroyAllWindows()
    
print(count)
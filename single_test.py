from backboned_unet import Unet
import cv2
import numpy as np
import torch

net = Unet(classes=1, position_classes=10, damage_classes=6, backbone_name='resnet50')

path = './log/%s/%s' % ('20221004212938', 'model-last.pt')

device = 'cuda'

if device == 'cuda':
    state_dict = torch.load(path)
else:
    state_dict = torch.load(path, map_location=torch.device('cpu'))
net.load_state_dict(state_dict)
net.train(False)
if device == 'cuda':
    net.cuda()
net.eval()

img_path = '/home/kc/test.jpg'
ori_img = cv2.imread(img_path).astype(np.float32)
img = cv2.resize(ori_img, (480, 480), interpolation = cv2.INTER_AREA)
img = torch.from_numpy(img.copy())
img = img.permute(2, 0, 1)
img = img.reshape([1, 3, 480, 480])
img = img.float().cuda()

_, _, _, out = net(img)
seg_out = out.reshape(1, 480, 480)
pred = torch.sigmoid(seg_out).permute(1, 2, 0) * 255
pred[pred <= 128] = 0
pred[pred > 128] = 255
pred = pred.reshape(480, 480)
pred0 = pred.detach().cpu().numpy()
pred = pred.detach().cpu().numpy().astype(np.uint8)

frame = cv2.resize(ori_img, (480, 480))
contours, hierarchy = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(frame, contours, -1, (0,255,0), 3)
frame = cv2.resize(frame, (1280, 959))
cv2.imwrite('new.jpg', frame)
cv2.imwrite('new2.jpg', pred0)
from data_loader import Data, OldData
from model import CustomModel
import torch, os
from tqdm import tqdm
from torch.utils.data import DataLoader
from backboned_unet import Unet


d = OldData(metadata_file='dir.json', test_fold=0, mode='test', img_size=(480, 480), segmentation_classes=6)
loader = DataLoader(d, batch_size=1)


device = 'cuda'
net = Unet(classes=1, position_classes=10, damage_classes=6, backbone_name='resnet50')
# net = CustomModel(name='densenet121')
net.train(False)
if device == 'cuda':
    net.cuda()
net.eval()
print('model has {} parameters in total'.format(sum(x.numel() for x in net.parameters())))

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
        seg_out = net(img)
from re import I
import torch
import torch.nn as nn

class DiceScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceScore, self).__init__()

    def forward(self, inputs, targets, seg_weight: torch.Tensor, smooth=1):
        # print(inputs.size(), targets.size())
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        inputs = inputs[seg_weight != 0]
        targets = targets[seg_weight != 0]

        inputs = torch.sigmoid(torch.flatten(inputs))
        inputs[inputs < 0.5] = 0
        inputs[inputs >= 0.5] = 1
        targets = torch.flatten(targets.float())
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return torch.nansum(dice)

class IoUScore(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoUScore, self).__init__()

    def forward(self, inputs, targets, smooth=1e-12):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = torch.flatten(inputs)
        targets = torch.flatten(targets.float())
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU


class MultiClassesDiceScore(nn.Module):
    def __init__(self):
        super().__init__()

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
    
    def forward(self, inputs, targets, classes, seg_weight: torch.Tensor, smooth=1e-12):
        inputs = inputs[seg_weight != 0]
        targets = targets[seg_weight != 0]

        return torch.nansum((1 - self.dice_score(inputs, targets, classes)).mean())
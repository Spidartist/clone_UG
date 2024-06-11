import torch
import torch.nn.functional as F
import torch.nn as nn


class WeightedPosCELoss(nn.Module):
    """
    Calculate cross_entropy loss only on positions that have label
    Label -1 means no label
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, targets):        
        x = inputs[targets != -1]
        y = targets[targets != -1]

        # using nansum to avoid there's no record doesn't has label
        return torch.nansum(nn.CrossEntropyLoss()(x, y))


class WeightedBCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, targets):        
        x = torch.sigmoid(torch.flatten(inputs[targets != -1]))
        y = targets[targets != -1]

        # using nansum to avoid there's no record doesn't has label
        return torch.nansum(nn.BCELoss()(x, y))


class ConsitencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, seg_weight):
        new_inputs = inputs[seg_weight != 0]
        new_seg_weight = seg_weight[seg_weight != 0]

        ones = torch.ones(new_inputs.shape, dtype=torch.float32, requires_grad=False).cuda()

        for i in range(len(new_seg_weight)):
            ones[i][new_seg_weight[i]] = 0

        return torch.nansum((torch.sigmoid(new_inputs) * ones).sum()) / ((new_seg_weight != 0).sum() * 480 * 480 * 3 + 1)


class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, seg_weight, smooth=1):
        new_inputs = inputs[seg_weight != 0]
        
        categorical_target = targets[0][seg_weight != 0]
        one_hot_target  = targets[1][seg_weight != 0]

        CE = nn.CrossEntropyLoss()(new_inputs, categorical_target)

        new_inputs = torch.sigmoid(torch.flatten(new_inputs))
        one_hot_target = torch.flatten(one_hot_target.float())
        
        intersection = (new_inputs * one_hot_target).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(new_inputs.sum() + one_hot_target.sum() + smooth)  

        Dice_CE = torch.nansum(CE) + torch.nansum(dice_loss)
        
        return Dice_CE



class MultiClassesDiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def cross_entropy(self, inputs, targets):
        targets = targets.long()
        return nn.CrossEntropyLoss()(inputs, targets)

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

        ce = self.cross_entropy(inputs, targets)

        return torch.nansum((1 - self.dice_score(inputs, targets, classes)).mean()) + torch.nansum(ce)


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, seg_weight, smooth=1):
        # print(inputs.size(), targets.size())
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       

        new_inputs = inputs[seg_weight != 0]
        new_targets = targets[seg_weight != 0]
        
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        inputs = torch.flatten(new_inputs)
        targets = torch.flatten(new_targets.float())
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = torch.nansum(BCE) + torch.nansum(dice_loss)
        
        return Dice_BCE


ALPHA = 0.5
BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        inputs = torch.flatten(inputs)
        targets = torch.flatten(targets.float())
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky


class BCEDiceTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceTverskyLoss, self).__init__()


    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        inputs = torch.flatten(inputs)
        targets = torch.flatten(targets.float())
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky + Dice_BCE


class TverskyDistillationLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyDistillationLoss, self).__init__()


    def forward(self, inputs, soft_targets, hard_targets, smooth=1e-12, alpha=ALPHA, beta=BETA, T=5):
        """Inputs and soft_targets must be logits
        """
        soft_inputs = torch.sigmoid(inputs/T)
        soft_targets = torch.sigmoid(soft_targets/T)
        inputs = torch.sigmoid(inputs)

        soft_inputs = torch.flatten(soft_inputs)
        inputs = torch.flatten(inputs)
        soft_targets = torch.flatten(soft_targets.float())
        hard_targets = torch.flatten(hard_targets.float())
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * hard_targets).sum()    
        FP = ((1-hard_targets) * inputs).sum()
        FN = (hard_targets * (1-inputs)).sum()
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

        distillation_loss = F.binary_cross_entropy(soft_inputs, soft_targets, reduction='mean')
        
        return 1 - Tversky + distillation_loss


class BCEDiceTverskyDistillationLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceTverskyDistillationLoss, self).__init__()


    def forward(self, inputs, soft_targets, hard_targets, smooth=1e-12, alpha=ALPHA, beta=BETA, T=5):
        """Inputs and soft_targets must be logits
        """
        soft_inputs = torch.sigmoid(inputs/T)
        soft_targets = torch.sigmoid(soft_targets/T)
        inputs = torch.sigmoid(inputs)

        soft_inputs = torch.flatten(soft_inputs)
        inputs = torch.flatten(inputs)
        soft_targets = torch.flatten(soft_targets.float())
        hard_targets = torch.flatten(hard_targets.float())
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * hard_targets).sum()    
        FP = ((1-hard_targets) * inputs).sum()
        FN = (hard_targets * (1-inputs)).sum()

        intersection = (inputs * hard_targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + hard_targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, hard_targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  

        distillation_loss = F.binary_cross_entropy(soft_inputs, soft_targets, reduction='mean')
        
        return 1 - Tversky + Dice_BCE + distillation_loss


if __name__=='__main__':
    # loss = DiceBCELoss()
    x = torch.rand(4, 3, 10, 10)
    y = torch.rand(4, 3, 10, 10)
    z = torch.rand(4, requires_grad=False)
    

from torchvision.models import vgg19_bn, resnet50, densenet121
import torch.nn as nn
import torch


class Flatten(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.flatten(x, 1)

class CustomModel(nn.Module):
    def __init__(self, name='vgg19', pretrained=True, out_classes=10) -> None:
        super().__init__()
        if name == 'vgg19':
            self.backbone = vgg19_bn(pretrained=pretrained).features
            self.bb_out_name = '52'
            self.in_neuron = 512
        elif name == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained)
            self.bb_out_name = 'layer4'
            self.in_neuron = 2048
        elif name == 'densenet121':
            self.backbone = densenet121(pretrained=pretrained).features
            self.bb_out_name = 'denseblock4'
            self.in_neuron = 1024

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Flatten(),
            nn.Linear(self.in_neuron, 512),
            nn.ReLU(),
            nn.Linear(512, out_classes)
        )

    def forward(self, x):
        inp = x
        for name, child in self.backbone.named_children():
            inp = child(inp)
            if name == self.bb_out_name:
                break
        
        return self.mlp(inp)


if __name__=='__main__':
    x = torch.rand(1, 3, 480, 480)
    net = CustomModel(name='resnet50')
    y = net(x)
        
        
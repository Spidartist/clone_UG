from torchvision import models

backbone = models.resnet50(pretrained=True)
print(backbone)
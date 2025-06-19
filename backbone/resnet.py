import torch.nn as nn
import torchvision


def resnet18(num_classes=100, pretrained=False):
    if pretrained:
        backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    else:
        backbone = torchvision.models.resnet18()
    
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    backbone.maxpool = nn.Identity()

    in_features = backbone.fc.in_features
    del backbone.fc  # Remove the original fully connected layer
    backbone.fc = nn.Linear(in_features, num_classes)  # Add a new fully connected layer
    return backbone

if __name__ == "__main__":
    model = resnet18(num_classes=10, pretrained=False)
    print(model)
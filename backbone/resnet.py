import torch.nn as nn
import torchvision


def resnet18(num_classes=100, pretrained=False):
    if pretrained:
        backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    else:
        backbone = torchvision.models.resnet18()
    
    in_features = backbone.fc.in_features
    del backbone.fc  # Remove the original fully connected layer
    backbone.fc = nn.Linear(in_features, num_classes)  # Add a new fully connected layer
    return backbone

if __name__ == "__main__":
    model = resnet18(num_classes=10, pretrained=True)
    print(model)
import torch
import torch.nn as nn
import torchvision.models as models

class SqueezeAndExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for adaptive channel-wise recalibration[cite: 256, 257].
    """
    def __init__(self, channels, reduction=16):
        super(SqueezeAndExcitation, self).__init__()
        # Squeeze step: Global Average Pooling [cite: 258, 286]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation step: Two fully-connected bottleneck layers [cite: 258, 292]
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid() # Non-linear gating mechanism [cite: 291, 293]
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze feature representations [cite: 286]
        squeeze = self.global_pool(x).view(b, c)
        # Excitation weights computation [cite: 291]
        excitation = self.fc(squeeze).view(b, c, 1, 1)
        # Channel-wise scale operation [cite: 295, 296]
        return x * excitation

class XFedChainBackbone(nn.Module):
    """
    Modified ResNet-50 backbone wrapped with custom SE attention blocks.
    """
    def __init__(self, num_classes=3):
        super(XFedChainBackbone, self).__init__()
        # Initialize standard ResNet-50 trunk [cite: 255]
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Unpack layers to safely inject SE blocks between residual blocks [cite: 256]
        self.initial_layers = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2
        )
        
        # Target late-stage blocks for dynamic optimization fine-tuning [cite: 300]
        self.layer3 = nn.Sequential(resnet.layer3, SqueezeAndExcitation(1024))
        self.layer4 = nn.Sequential(resnet.layer4, SqueezeAndExcitation(2048))
        
        # Multi-task classification layer head [cite: 302]
        self.avgpool = resnet.avgpool
        self.fc_head = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc_head(x)
        return logits
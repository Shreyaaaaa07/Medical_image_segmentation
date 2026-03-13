import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class SegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Load ResNet50 as encoder
        resnet = resnet50(pretrained=True)

        # Encoder layers
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc

        # Decoder
        self.decoder = nn.ModuleList([
            DecoderBlock(2048, 1024),  # 2048 -> 1024
            DecoderBlock(1024, 512),   # 1024 -> 512
            DecoderBlock(512, 256),    # 512 -> 256
            DecoderBlock(256, 128),    # 256 -> 128
            DecoderBlock(128, 64),     # 128 -> 64
        ])

        # Segmentation head
        self.segmentation_head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [4, 5, 6, 7]:  # Save features from different levels
                features.append(x)

        # Decoder with skip connections
        x = features[-1]  # Start with deepest feature
        for i, decoder_block in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if i < len(features) - 1:
                skip = features[-(i+2)]  # Get corresponding encoder feature
                # Resize skip connection to match decoder output size
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = decoder_block(x)

        # Final segmentation
        x = self.segmentation_head(x)
        return x

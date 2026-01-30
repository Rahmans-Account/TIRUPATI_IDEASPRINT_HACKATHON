"""U-Net architecture for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double Convolution block."""
    
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        ]
        self.double_conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_batchnorm)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, use_batchnorm)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad x1 to match x2 size if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net model for semantic segmentation."""
    
    def __init__(
        self,
        in_channels=9,
        num_classes=5,
        base_channels=64,
        use_batchnorm=True,
        dropout=0.2
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder
        self.inc = DoubleConv(in_channels, base_channels, use_batchnorm)
        self.down1 = Down(base_channels, base_channels * 2, use_batchnorm)
        self.down2 = Down(base_channels * 2, base_channels * 4, use_batchnorm)
        self.down3 = Down(base_channels * 4, base_channels * 8, use_batchnorm)
        
        # Bottleneck
        self.down4 = Down(base_channels * 8, base_channels * 16, use_batchnorm)
        
        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8, use_batchnorm)
        self.up2 = Up(base_channels * 8, base_channels * 4, use_batchnorm)
        self.up3 = Up(base_channels * 4, base_channels * 2, use_batchnorm)
        self.up4 = Up(base_channels * 2, base_channels, use_batchnorm)
        
        # Output
        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        if self.dropout:
            x = self.dropout(x)
        
        logits = self.outc(x)
        return logits


def create_unet_model(config: dict) -> UNet:
    """Create U-Net model from config."""
    return UNet(
        in_channels=config.get('in_channels', 9),
        num_classes=config.get('num_classes', 5),
        base_channels=config.get('base_channels', 64),
        use_batchnorm=config.get('use_batchnorm', True),
        dropout=config.get('dropout', 0.2)
    )

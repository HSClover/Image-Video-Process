# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Conv Block ---
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

# --- Simple U-Net ---
class SimpleUNet(nn.Module):
    # 9채널 입력 (3프레임)
    def __init__(self, in_channels=9, out_channels=3): 
        super(SimpleUNet, self).__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(256, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2));
        b = self.bottleneck(self.pool(e3));
        
        d3 = self.upconv3(b); d3 = torch.cat((e3, d3), dim=1); d3 = self.dec3(d3);
        d2 = self.upconv2(d3); d2 = torch.cat((e2, d2), dim=1); d2 = self.dec2(d2);
        d1 = self.upconv1(d2); d1 = torch.cat((e1, d1), dim=1); d1 = self.dec1(d1);
        
        output = self.out_conv(d1)
        return output
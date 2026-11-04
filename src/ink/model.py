"""Ink3D-lite: 3D stem -> depth pool -> 2D UNet-lite head.

Input (B,1,16,128,128) -> logits (B,128,128). ~1.1M params, fp16-safe.
"""
import torch
import torch.nn as nn


class Conv3(nn.Module):
    def __init__(self, c0, c1, stride=(1, 1, 1)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(c0, c1, 3, stride, 1, bias=False),
            nn.BatchNorm3d(c1), nn.SiLU(),
            nn.Conv3d(c1, c1, 3, 1, 1, bias=False),
            nn.BatchNorm3d(c1), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class Conv2(nn.Module):
    def __init__(self, c0, c1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c0, c1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c1), nn.SiLU(),
            nn.Conv2d(c1, c1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c1), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class Ink3D(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.s3a = Conv3(1, base, stride=(2, 2, 2))      # 8x64x64
        self.s3b = Conv3(base, base * 2, stride=(2, 2, 2))  # 4x32x32
        self.s3c = Conv3(base * 2, base * 4)               # 4x32x32
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))  # 1x32x32
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)  # 64
        self.c1 = Conv2(base * 2, base * 2)
        self.up0 = nn.ConvTranspose2d(base * 2, base, 2, 2)      # 128
        self.c0 = Conv2(base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x = self.s3a(x)
        x = self.s3b(x)
        x = self.s3c(x)
        x = self.pool(x).squeeze(2)
        x = self.c1(self.up1(x))
        x = self.c0(self.up0(x))
        return self.out(x).squeeze(1)


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    t = target.float()
    inter = (p * t).sum((1, 2))
    return 1 - (2 * inter + eps) / (p.sum((1, 2)) + t.sum((1, 2)) + eps)

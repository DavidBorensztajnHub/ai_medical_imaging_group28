#!/usr/bin/env python3

# ResNet34 encoder (ImageNet weights) + U-Net decoder.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


def convBatch(nin, nout, kernel_size=3, stride=1, padding=1, bias=False, layer=nn.Conv2d, dilation=1):
    return nn.Sequential(
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
        nn.BatchNorm2d(nout),
        nn.PReLU()
    )


class UpBlock(nn.Module):
    # Upsample, concat the skip, fuse
    def __init__(self, nin, nskip, nout):
        super().__init__()
        self.reduce = convBatch(nin, nout, kernel_size=1, padding=0)
        self.fuse = nn.Sequential(convBatch(nout + nskip, nout),
                                  convBatch(nout, nout))

    def forward(self, deep, skip):
        x = self.reduce(deep)
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        return self.fuse(torch.cat([x, skip], dim=1))


class ResNet34(nn.Module):
    def __init__(self, nin, nout, nG=64, **kwargs):  # **kwargs discards unnecessary keywords arguments (kernels, factor)
        super().__init__()

        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # 1 channel in, not 3. Summing the RGB filters is the same as feeding the
        # slice to all three, so the pretrained stem survives.
        w = backbone.conv1.weight.data
        conv1 = nn.Conv2d(nin, 64, kernel_size=7, stride=2, padding=3, bias=False)
        conv1.weight.data = w.clone() if nin == 3 else w.sum(dim=1, keepdim=True).repeat(1, nin, 1, 1) / nin
        backbone.conv1 = conv1

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 1/2, 64
        self.pool = backbone.maxpool  # 1/4
        self.layer1 = backbone.layer1  # 1/4, 64
        self.layer2 = backbone.layer2  # 1/8, 128
        self.layer3 = backbone.layer3  # 1/16, 256
        self.layer4 = backbone.layer4  # 1/32, 512

        self.up3 = UpBlock(512, 256, nG * 4)
        self.up2 = UpBlock(nG * 4, 128, nG * 2)
        self.up1 = UpBlock(nG * 2, 64, nG)
        self.up0 = UpBlock(nG, 64, nG)

        self.final = nn.Conv2d(nG, nout, kernel_size=1)

    def forward(self, input):
        x0 = self.stem(input)            # 1/2
        x1 = self.layer1(self.pool(x0))  # 1/4
        x2 = self.layer2(x1)             # 1/8
        x3 = self.layer3(x2)             # 1/16
        x4 = self.layer4(x3)             # 1/32

        d3 = self.up3(x4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)
        d0 = self.up0(d1, x0)

        out = self.final(d0)

        return F.interpolate(out, size=input.shape[-2:], mode='bilinear', align_corners=False)

    def init_weights(self, *args, **kwargs):
        # Decoder only -- main.py calls this after __init__, so anything we touch
        # here loses its pretrained weights.
        for block in [self.up3, self.up2, self.up1, self.up0, self.final]:
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

    def param_groups(self, lr: float, encoder_factor: float = 0.1) -> list[dict]:
        # The pretrained half needs smaller steps than the random half
        is_encoder = lambda n: n.startswith(('stem', 'layer'))

        return [{'params': [p for n, p in self.named_parameters() if is_encoder(n)],
                 'lr': lr * encoder_factor},
                {'params': [p for n, p in self.named_parameters() if not is_encoder(n)],
                 'lr': lr}]

import torch
import torch.nn as nn
from torch.nn import functional as F
from lib.utils import logger


class ResizeConv(nn.Module):
    def __init__(
        self,
        conv_in_channels,
        conv_out_channels,
        upsample_ratio,
        kernel_size=3,
        stride=1,
        padding=0,
        interpolation="nearest",
    ):
        super(ResizeConv, self).__init__()
        assert interpolation in ["nearest", "bilinear"]
        self.upsample_ratio = upsample_ratio
        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.layers = nn.ModuleList(
            [
                nn.Upsample(scale_factor=self.upsample_ratio, mode=interpolation),
                nn.ReflectionPad2d(1),
                nn.Conv2d(
                    self.conv_in_channels,
                    self.conv_out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            ]
        )
        self.init_weights()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

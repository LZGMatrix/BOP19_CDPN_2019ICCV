from torch import nn
import torch
from torch.nn import functional as F
from network.resnet import resnet18


class Resnet18_8s(nn.Module):

    def __init__(self, ver_dim, seg_dim, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32, inp_dim=3):
        super(Resnet18_8s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(
            inp_dim = inp_dim,
            fully_conv=True,
            pretrained=True,
            output_stride=8,
            remove_avg_pool_layer=True,
        )

        self.ver_dim = ver_dim
        self.seg_dim = seg_dim

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True),
        )
        self.resnet18_8s = resnet18_8s

        # x8s->128
        self.conv8s = nn.Sequential(
            nn.Conv2d(128 + fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up8sto4s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x4s->64
        self.conv4s = nn.Sequential(
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up4sto2s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x2s->64
        self.conv2s = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

        self.convraw = nn.Sequential(
            nn.Conv2d(inp_dim + s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(raw_dim, self.seg_dim + self.ver_dim, 1, 1),
        )

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)

        fm = self.conv8s(torch.cat([xfc, x8s], 1))
        fm = self.up8sto4s(fm)

        fm = self.conv4s(torch.cat([fm, x4s], 1))
        fm = self.up4sto2s(fm)

        fm = self.conv2s(torch.cat([fm, x2s], 1))
        fm = self.up2storaw(fm)

        x = self.convraw(torch.cat([fm, x], 1))
        
        conf = x[:, :2, :, :] 
        coor_x = x[:, 2:(2+65), :, :]
        coor_y = x[:, (2+65):(2+65*2), :, :]
        coor_z = x[:, (2+65*2):(2+65*3), :, :]
                            
        return conf, coor_x, coor_y, coor_z 


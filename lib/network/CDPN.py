import torch
import torch.nn as nn

class CDPN(nn.Module):
    def __init__(self, backbone, rot_head_net):
        super(CDPN, self).__init__()
        self.backbone = backbone
        self.rot_head_net = rot_head_net

    def forward(self, x):         
        features = self.backbone(x)       
        conf, coor_x, coor_y, coor_z  = self.rot_head_net(features)
        return conf, coor_x, coor_y, coor_z 
 

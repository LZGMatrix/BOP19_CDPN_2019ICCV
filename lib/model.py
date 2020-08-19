import ref
import torch
import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '../..'))

import torchvision.models as models
import torch.nn as nn
from torchvision.models.resnet import model_zoo, model_urls, BasicBlock, Bottleneck
from network.resnet_backbone import ResNetBackboneNet
from network.resnet_rot_head import RotHeadNet
from network.CDPN import CDPN

# Specification
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
               34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
               50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
               101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
               152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}

def get_model(cfg):
    params_lr_list = []
    block_type, layers, channels, name = resnet_spec[cfg.network.back_layers_num]
    backbone_net = ResNetBackboneNet(block_type, layers, cfg.network.inp_dim)
    rot_head_net = RotHeadNet(channels[-1], 3, 256, 3, 1, cfg.network.out_dim)
    model = CDPN(backbone_net, rot_head_net)

    print("=> loading model '{}'".format(ref.save_models_dir.format(cfg.pytorch.dataset.lower(), cfg.pytorch.object.lower())))
    checkpoint = torch.load(ref.save_models_dir.format(cfg.pytorch.dataset.lower(), cfg.pytorch.object.lower()), map_location=lambda storage, loc: storage)
    if type(checkpoint) == type({}):
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint.state_dict()
    model_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(filtered_state_dict)
    model.load_state_dict(model_dict)

    return model

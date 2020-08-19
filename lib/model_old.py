import ref
import torch
import os, sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '../..'))
from network.model_repository import Resnet18_8s

def get_model(cfg):
    ver_dim = cfg.network.ver_dim
    seg_dim = cfg.network.seg_dim
    model = Resnet18_8s(ver_dim= ver_dim, seg_dim=seg_dim, inp_dim=cfg.network.inp_dim)
    # print("=> loading model '{}'".format(cfg.pytorch.load_model))
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


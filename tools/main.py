# encoding: utf-8
'''
@author: Zhigang Li
@license: (C) Copyright.
@contact: aaalizhigang@163.com
@software: Pose6D
@file: main.py
@time: 18-10-24 下午10:24
@desc:
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import numpy as np
import random
import time
import datetime
import torch
import torch.utils.data
import ref
import pprint
import cv2
cv2.ocl.setUseOpenCL(False)
from model import get_model
from utils import get_ply_model
from test import test
import json 
from config import  cfg
cfg = cfg().parse()

def main():
    if cfg.pytorch.exp_mode == 'test':
        if cfg.pytorch.dataset.lower() == 'lmo':
            from dataloader_detected_bbox import LMO as Dataset
            model_dir = ref.lmo_model_dir
        elif cfg.pytorch.dataset.lower() == 'tless':
            from dataloader_detected_bbox import TLESS as Dataset 
            model_dir = ref.tless_model_dir
        elif cfg.pytorch.dataset.lower() == 'ycbv':
            from dataloader_detected_bbox import YCBV as Dataset 
            model_dir = ref.ycbv_model_dir
        elif cfg.pytorch.dataset.lower() == 'tudl':
            from dataloader_detected_bbox import TUDL as Dataset 
            model_dir = ref.tudl_model_dir
        elif cfg.pytorch.dataset.lower() == 'hb':
            from dataloader_detected_bbox import HB as Dataset 
            model_dir = ref.hb_model_dir
        elif cfg.pytorch.dataset.lower() == 'icbin':
            from dataloader_detected_bbox import ICBIN as Dataset
            model_dir = ref.icbin_model_dir
        elif cfg.pytorch.dataset.lower() == 'itodd':
            from dataloader_detected_bbox import ITODD as Dataset
            model_dir = ref.itodd_model_dir
    elif cfg.pytorch.exp_mode == 'val':
        if cfg.pytorch.dataset.lower() == 'lmo':
            from dataloader import LMO as Dataset
            model_dir = ref.lmo_model_dir
        elif cfg.pytorch.dataset.lower() == 'tless':
            from dataloader import TLESS as Dataset 
            model_dir = ref.tless_model_dir
        elif cfg.pytorch.dataset.lower() == 'ycbv':
            from dataloader import YCBV as Dataset 
            model_dir = ref.ycbv_model_dir
        elif cfg.pytorch.dataset.lower() == 'tudl':
            from dataloader import TUDL as Dataset 
            model_dir = ref.tudl_model_dir
        elif cfg.pytorch.dataset.lower() == 'hb':
            from dataloader import HB as Dataset 
            model_dir = ref.hb_model_dir
        elif cfg.pytorch.dataset.lower() == 'icbin':
            from dataloader import ICBIN as Dataset
            model_dir = ref.icbin_model_dir
        elif cfg.pytorch.dataset.lower() == 'itodd':
            from dataloader import ITODD as Dataset
            model_dir = ref.itodd_model_dir
        
    with open(os.path.join(model_dir, 'models_info.json'), 'r') as f_model_eval:
        models_info = json.load(f_model_eval) # load model info (size & diameter)
        for k in list(models_info.keys()):
            models_info[int(k)] = models_info.pop(k)

    models_vtx = {}
    for obj_name in [cfg.pytorch.object]:    
        if cfg.pytorch.dataset.lower() == 'lmo': 
            obj_id = ref.lmo_obj2id(obj_name)
        elif cfg.pytorch.dataset.lower() == 'tless':
            obj_id = ref.tless_obj2id(obj_name)
        elif cfg.pytorch.dataset.lower() == 'ycbv':
            obj_id = ref.ycbv_obj2id(obj_name)       
        elif cfg.pytorch.dataset.lower() == 'tudl':
            obj_id = ref.tudl_obj2id(obj_name)      
        elif cfg.pytorch.dataset.lower() == 'hb':
            obj_id = ref.hb_obj2id(obj_name)  
        elif cfg.pytorch.dataset.lower() == 'icbin': 
            obj_id = ref.icbin_obj2id(obj_name)            
        elif cfg.pytorch.dataset.lower() == 'itodd':
            obj_id = ref.itodd_obj2id(obj_name)  
          
        models_vtx[obj_name] = get_ply_model(os.path.join(model_dir, 'obj_{:06d}.ply'.format(obj_id)))

    ## load model, optimizer, criterions
    model = get_model(cfg)

    if cfg.pytorch.gpu > -1:
        # print('Using GPU{}'.format(cfg.pytorch.gpu))
        model = model.cuda(cfg.pytorch.gpu)

    ## load test set
    def _worker_init_fn():
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)

    test_loader = torch.utils.data.DataLoader(
        Dataset(cfg, 'test'),
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg.pytorch.threads_num),
        worker_init_fn=_worker_init_fn()
    )

    ## test
    test(cfg, test_loader, model, models_info, models_vtx)

if __name__ == '__main__':
    main()

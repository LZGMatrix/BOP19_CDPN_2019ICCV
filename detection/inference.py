import argparse
import os
import os.path as osp
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint
from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.apis import init_dist
from mmdet.core import coco_eval, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from tqdm import tqdm
from time import time
import numpy as np
import json


# Global Variables
DATASET_DIR = 'bop19'
CONFIG_DIR = 'bop-configs'
MODEL_DIR = 'models'
OUT_DIR = 'out'
ANNOTATION_DIR = 'annotations'
gpu_id = 0

# For HomeBrewed and Linemod-Occluded Dataset Fix-up 
datasets = ['hb','lmo','itodd','icbin','tless','tudl','ycbv']
hb_clses = [1, 3, 4, 8, 9, 10, 12, 15, 17, 18, 19, 22, 23, 29, 32, 33]
lmo_clses = [1,5,6,8,9,10,11,12]


class Bop19Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(Bop19Encoder, self).default(obj)

def single_gpu_test(model, data_loader,show=False, warmup_num=100):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    result_dict = {}
    for i, data in enumerate(data_loader):
        if i == 0:
           for j in range(warmup_num):
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **data)
       
        img_meta = data['img_meta'][0].__dict__
        filename = img_meta['_data'][0][0]['filename']
        dirs = filename.split('/')
        scene_id = int(dirs[-3])
        img_name = dirs[-1]
        dataset_name = dirs[-5]
        img_id, _ = osp.splitext(img_name)
        img_id = int(img_id)
        if scene_id not in result_dict:
            result_dict[scene_id] = {}
        result_dict[scene_id][img_id] = []
        with torch.no_grad():
            # count time
            start_time = time()
            result = model(return_loss=False, rescale=not show, **data)
            end_time = time()

        for cls_id in range(1, len(result) + 1):
            cls_idx = cls_id - 1 
            if dataset_name == 'hb':
                cls_id = hb_clses[cls_idx]
            elif dataset_name == 'lmo':
                cls_id = lmo_clses[cls_idx]
            N,_ = result[cls_idx].shape
            for k in range(N):
                det_info = result[cls_idx][k,:]
                single_result = {}
                single_result['obj_id'] = cls_id
                bbox = list(det_info[0:4])
                x = bbox[0]; y = bbox[1]; 
                w = bbox[2] - bbox[0];
                h = bbox[3] - bbox[1]
                single_result['bbox'] = [x,y,w,h]
                single_result['score'] = det_info[-1]
                single_result['time'] = (end_time - start_time)
                result_dict[scene_id][img_id].append(single_result)
        
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return result_dict

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('dataset',help='dataset name: hb | lmo | itodd | icbin | tless | tudl | ycbv', choices=datasets, nargs='+')
    return parser.parse_args() 

def main():
    args = parse_args()
    datasets = args.dataset 
    for dt_name in datasets:
        print('\nCurrent Dataset: %s' % dt_name)
        # Set up file
        config_file = osp.join(CONFIG_DIR,'%s_config.py' % dt_name)
        checkpoint_file = osp.join(MODEL_DIR,'%s_det.pth' % dt_name)
        cfg = mmcv.Config.fromfile(config_file)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        
        # load in checkpoint
        checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
        
        
        # data for inference
        if dt_name == 'tless':
            data_path = osp.join(DATASET_DIR,dt_name,'test_primesense')
        else: 
            data_path = osp.join(DATASET_DIR,dt_name,'test')
        
        # build dataloader
        cfg.data.test['ann_file'] = osp.join(ANNOTATION_DIR,'test_%s.json' % dt_name)
        cfg.data.test['img_prefix'] = data_path
        # build the dataloader
        # TODO: support multiple images per gpu (only minor changes are needed)
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)
        
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
       
        model = MMDataParallel(model, device_ids=[gpu_id])
        result_dict = single_gpu_test(model, data_loader)
        out_path = osp.join(OUT_DIR,dt_name)
        os.makedirs(out_path,exist_ok=True)
        for scene_id, result_per_scene in result_dict.items():
            with open(osp.join(out_path,'%s_%06d_detection_result.json')% (dt_name, scene_id), 'w') as f:
                json.dump(result_per_scene,f,indent=4,cls=Bop19Encoder)

if __name__ == '__main__':
    main()

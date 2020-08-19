import os
import yaml
import argparse
import copy
import numpy as np
from easydict import EasyDict as edict
from torchvision.models.resnet import model_zoo, model_urls, BasicBlock, Bottleneck
from tensorboardX import SummaryWriter
import sys
import ref
from datetime import datetime
from pprint import pprint

def get_default_config_pytorch():
    config = edict()
    config.exp_mode = 'test'        # 'test' or 'val' 
    config.gpu = 1
    config.threads_num = 12         # 'nThreads'
    config.load_model = ''          # path to a previously trained model
    config.dataset = 'LMO'
    config.object = 'all'        
    return config

def get_default_dataiter_config():
    config = edict()
    config.backbone_input_res = 128
    config.rot_output_res = 128
    return config

def get_default_network_config():
    config = edict()
    config.coor_bin = 64
    config.seg_dim = 2
    config.ver_dim = 3 * 65
    config.inp_dim = 3
    return config

def get_default_test_config():
    config = edict()
    config.test_mode = 'pose'   # 'pose' | 'add' | 'proj' | 'all' | 'pose_fast' | 'add_fast' | 'proj_fast' | 'all_fast'
    config.mask_threshold = 0.5
    config.detection = 'RetinaNet' 
    config.ransac_projErr = 3
    config.ransac_iterCount = 100
    return config

def get_base_config():
    base_config = edict()
    base_config.pytorch = get_default_config_pytorch()
    base_config.dataiter = get_default_dataiter_config()
    base_config.test = get_default_test_config()
    base_config.network = get_default_network_config()
    return base_config

def update_config_from_file(_config, config_file, check_necessity=True):
    config = copy.deepcopy(_config)
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vk in config[k]:
                            if isinstance(vv, list) and not isinstance(vv[0], str):
                                config[k][vk] = np.asarray(vv)
                            else:
                                config[k][vk] = vv
                        else:
                            if check_necessity:
                                raise ValueError("{}.{} not exist in config".format(k, vk))
                else:
                    raise ValueError("{} is not dict type".format(v))
            else:
                if check_necessity:
                    raise ValueError("{} not exist in config".format(k))
    return config

class cfg():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='pose experiment')
        self.parser.add_argument('--exp_mode', required=True, type=str, help='')
        self.parser.add_argument('--cfg', required=True, type=str, help='path/to/configure_file')
        self.parser.add_argument('--dataset', required=True, type=str, help='')
        self.parser.add_argument('--object', required=True, type=str, help='')
        
    def parse(self):
        config = get_base_config()                  # get default arguments
        args, rest = self.parser.parse_known_args() # get arguments from command line
        for k, v in vars(args).items():
            config.pytorch[k] = v 
        config_file = config.pytorch.cfg
        config = update_config_from_file(config, config_file, check_necessity=False) # update arguments from config file

        if config.pytorch.dataset.lower() == 'lmo':
            config.pytorch['camera_matrix'] = ref.lmo_camera_matrix
            config.pytorch['width'] = 640
            config.pytorch['height'] = 480            
        elif config.pytorch.dataset.lower() == 'ycbv':
            config.pytorch['camera_matrix'] = ref.ycbv_camera_matrix
            config.pytorch['width'] = 640
            config.pytorch['height'] = 480
        elif config.pytorch.dataset.lower() == 'tless':
            # config.pytorch['camera_matrix'] = ref.tless_camera_matrix # camera_matrix vary with images in TLESS
            config.pytorch['width'] = 720
            config.pytorch['height'] = 540
        elif config.pytorch.dataset.lower() == 'tudl':
            config.pytorch['camera_matrix'] = ref.tudl_camera_matrix
            config.pytorch['width'] = 640
            config.pytorch['height'] = 480
        elif config.pytorch.dataset.lower() == 'hb':
            config.pytorch['camera_matrix'] = ref.hb_camera_matrix
            config.pytorch['width'] = 640
            config.pytorch['height'] = 480
        elif config.pytorch.dataset.lower() == 'icbin':
            config.pytorch['camera_matrix'] = ref.icbin_camera_matrix
            config.pytorch['width'] = 640
            config.pytorch['height'] = 480      
        elif config.pytorch.dataset.lower() == 'itodd':
            # config.pytorch['camera_matrix'] = ref.itodd_camera_matrix # camera_matrix vary with images in ITODD
            config.pytorch['width'] = 1280
            config.pytorch['height'] = 960  
            config.network.inp_dim = 1                                             
        else:
            raise Exception("Wrong dataset name: {}".format(config.pytorch.dataset))

        config.pytorch['center'] = (config.pytorch['height'] / 2, config.pytorch['width'] / 2)

        # complement config regarding paths
        now = datetime.now().isoformat()
        # save path
        config.pytorch['save_path'] = os.path.join(ref.exp_dir, '{}_{}'.format(config.pytorch.dataset, config.pytorch.object))
        config.pytorch['save_csv_path'] = os.path.join(config.pytorch['save_path'], '{}_{}.csv'.format(config.pytorch.dataset, config.pytorch.object))
        if not os.path.exists(config.pytorch.save_path):
            os.makedirs(config.pytorch.save_path, exist_ok=True)
        # pprint(config)

        return config

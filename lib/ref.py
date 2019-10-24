# encoding: utf-8
'''
@author: Zhigang Li
@license: (C) Copyright.
@contact: aaalizhigang@163.com
@software: Pose6D
@file: ref.py
@time: 18-10-24 下午9:00
@desc: 
'''
import paths
import numpy as np
import os

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
root_dir = paths.rootDir
data_cache_dir = os.path.join(root_dir, 'data')
exp_dir = os.path.join(root_dir, 'exp') 
data_dir = os.path.join(root_dir, 'dataset')
bbox_dir = os.path.join(root_dir, 'bbox_retinanet')
save_models_dir = os.path.join(root_dir, 'trained_models/{}/obj_{}.checkpoint')


# ---------------------------------------------------------------- #
# LINEMOD OCCLUSION DATASET
# ---------------------------------------------------------------- #
lmo_dir = os.path.join(data_dir, 'lmo_bop19')
lmo_test_dir = os.path.join(lmo_dir, 'test')
lmo_model_dir = os.path.join(lmo_dir, 'models_eval')
# object info
lmo_objects = ['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher']
lmo_id2obj = {
             1: 'ape',
             5: 'can',
             6: 'cat',
             8: 'driller',
             9: 'duck',
             10: 'eggbox',
             11: 'glue',
             12: 'holepuncher',
             }
lmo_obj_num = len(lmo_id2obj)
def lmo_obj2id(obj_name):
    for k, v in lmo_id2obj.items():
        if v == obj_name:
            return k
# Camera info
lmo_width = 640
lmo_height = 480
lmo_center = (lmo_height / 2, lmo_width / 2)
lmo_camera_matrix = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])


# ---------------------------------------------------------------- #
# HB DATASET
# ---------------------------------------------------------------- #
hb_dir = os.path.join(data_dir, 'hb_bop19')
hb_test_dir = os.path.join(hb_dir, 'test')
hb_model_dir = hb_model_eval_dir = os.path.join(hb_dir, 'models_eval')
# object info
hb_objects = ['1', '3', '4', '8', '9', '10', '12', '15', '17', '18', '19', '22', '23', '29', '32', '33']
hb_id2obj = {
             1: '1',
             3: '3',
             4: '4',
             8: '8',
             9: '9',
             10: '10',
             12: '12',
             15: '15',
             17: '17',
             18: '18',
             19: '19',
             22: '22',
             23: '23',
             29: '29',
             32: '32',
             33: '33', 
             }
hb_obj_num = len(hb_id2obj)
def hb_obj2id(obj_name):
    for k, v in hb_id2obj.items():
        if v == obj_name:
            return k
hb_test_scenes = ['000003', '000005', '000013']
hb_val_scenes = ['000003', '000005', '000013']
# Camera info
hb_width = 640
hb_height = 480
hb_center = (hb_height / 2, hb_width / 2)
hb_camera_matrix = np.array([[537.4799, 0, 318.8965], [0, 536.1447, 238.3781], [0, 0, 1]])


# ---------------------------------------------------------------- #
# TLESS DATASET
# ---------------------------------------------------------------- #
tless_dir = os.path.join(data_dir, 'tless_bop19')
tless_test_dir = os.path.join(tless_dir, 'test_primesense')
tless_model_dir = os.path.join(tless_dir, 'models_eval')
# object info
tless_objects = [str(i) for i in range(1, 31)]
tless_id2obj = {i: str(i) for i in range(1, 31)}
tless_obj_num = len(tless_id2obj)
def tless_obj2id(obj_name):
    for k, v in tless_id2obj.items():
        if v == obj_name:
            return k
# Camera info
tless_width = 720
tless_height = 540
tless_center = (tless_width / 2., tless_height / 2.)
tless_camera_matrix = np.array([[1075.65091572, 0.0, 374.06888344], [0.0, 1073.90347929, 255.72159802], [0, 0, 1]])


# ---------------------------------------------------------------- #
# YCBV DATASET
# ---------------------------------------------------------------- #
ycbv_dir = os.path.join(data_dir, 'ycbv_bop19')
ycbv_test_dir = os.path.join(ycbv_dir, 'test')
ycbv_model_dir = os.path.join(ycbv_dir, 'models_eval')
ycbv_objects = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
           '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
           '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
           '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']
ycbv_id2obj = {
             1: '002_master_chef_can',
             2: '003_cracker_box',
             3: '004_sugar_box',
             4: '005_tomato_soup_can',
             5: '006_mustard_bottle',
             6: '007_tuna_fish_can',
             7: '008_pudding_box',
             8: '009_gelatin_box',
             9: '010_potted_meat_can',
             10: '011_banana',
             11: '019_pitcher_base',
             12: '021_bleach_cleanser',
             13: '024_bowl',
             14: '025_mug',
             15: '035_power_drill',
             16: '036_wood_block',
             17: '037_scissors',
             18: '040_large_marker',
             19: '051_large_clamp',
             20: '052_extra_large_clamp',
             21: '061_foam_brick'
             }
ycbv_obj_num= len(ycbv_id2obj)
def ycbv_obj2id(class_name):
    for k, v in ycbv_id2obj.items():
        if v == class_name:
            return k
ycbv_test_scenes = ['000048',  '000049',  '000050',  '000051',  '000052',  '000053',  '000054',  '000055',  '000056', 
                   '000057',  '000058',  '000059']
# Camera info
ycbv_width = 640
ycbv_height = 480
ycbv_center = (ycbv_height / 2, ycbv_width / 2)
ycbv_camera_matrix = np.array([[1066.778, 0.0, 312.9869], [0.0, 1067.487, 241.3109], [0, 0, 1]])                


# --------------------------------------------------------- #
# TUDL DATASET
# ---------------------------------------------------------------- #
tudl_root = os.path.join(data_dir, 'tudl_bop19')
tudl_test_dir = os.path.join(tudl_root, 'test')
tudl_model_dir = os.path.join(tudl_root, 'models_eval')
tudl_model_info_file = os.path.join(tudl_root, 'models', 'models_info.yml')
tudl_objects = ['dragon', 'frog', 'can']
tudl_id2obj = {
             1: 'dragon',
             2: 'frog',
             3: 'can'
             }
tudl_obj_num = len(tudl_id2obj)
def tudl_obj2id(class_name):
    for k, v in tudl_id2obj.items():
        if v == class_name:
            return k
tudl_width = 640
tudl_height = 480
tudl_center = (tudl_height / 2, tudl_width / 2)
tudl_camera_matrix = np.array([[515.0, 0.0, 321.566], [0.0, 515.0, 214.08], [0, 0, 1]])  


# ---------------------------------------------------------------- #
# ICBIN DATASET
# ---------------------------------------------------------------- #
icbin_dir = os.path.join(data_dir, 'icbin_bop19')
icbin_test_dir = os.path.join(icbin_dir, 'test')
icbin_model_dir = os.path.join(icbin_dir, 'models_eval')
# object info
icbin_objects = ['coffee_cup', 'juice_carton']
icbin_id2obj = {
             1: 'coffee_cup',
             2: 'juice_carton',
             }
icbin_obj_num = len(icbin_id2obj)
def icbin_obj2id(obj_name):
    for k, v in icbin_id2obj.items():
        if v == obj_name:
            return k
icbin_test_scenes = ['000001', '000002', '000003']
# Camera info
icbin_width = 640
icbin_height = 480
icbin_center = (icbin_height / 2, icbin_width / 2)
icbin_camera_matrix = np.array([[550.0, 0.0, 316.0], [0.0, 540.0, 244.0], [0, 0, 1]])


# ---------------------------------------------------------------- #
# ITODD DATASET
# ---------------------------------------------------------------- #
itodd_dir = os.path.join(data_dir, 'itodd_bop19')
itodd_val_dir = os.path.join(itodd_dir, 'val')
itodd_test_dir = os.path.join(itodd_dir, 'test')
itodd_model_dir = os.path.join(itodd_dir, 'models_eval')
# object info
itodd_objects = [str(i) for i in range(1, 29)]
itodd_id2obj = {i: str(i) for i in range(1, 29)}
itodd_obj_num = len(itodd_id2obj)
def itodd_obj2id(obj_name):
    for k, v in itodd_id2obj.items():
        if v == obj_name:
            return k
itodd_val_scenes = ['000001']
itodd_test_scenes = ['000001']
# Camera info
itodd_width = 1280
itodd_height = 960
itodd_center = (itodd_height / 2, itodd_width / 2)
itodd_camera_matrix = np.array([[2992.63, 0.0, 633.886], [0.0, 3003.99, 489.554], [0, 0, 1]])

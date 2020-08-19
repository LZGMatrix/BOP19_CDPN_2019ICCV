# encoding: utf-8
'''
@author: Zhigang Li
@license: (C) Copyright.
@contact: aaalizhigang@163.com
@software: Pose6D
@file: LineMOD.py
@time: 18-10-24 下午10:24
@desc: load LineMOD dataset
'''

import torch.utils.data as data
import numpy as np
import ref
import cv2
import os, sys
import pickle
import json
from utils import *
import pickle as pkl


class LMO(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.root_dir = ref.lmo_dir
        # print('==> initializing {} {} data.'.format(cfg.pytorch.dataset, split))          
        ## load dataset
        annot = []
        tr_obj_dir = os.path.join(self.root_dir, 'test/000002')
        f_pose = os.path.join(tr_obj_dir, 'scene_gt.json')
        f_det = os.path.join(tr_obj_dir, 'scene_gt_info.json')
        f_cam = os.path.join(tr_obj_dir, 'scene_camera.json')
        with open(f_pose, 'r') as f:
            annot_pose = json.load(f)
        with open(f_det, 'r') as f:
            annot_det = json.load(f)
        with open(f_cam, 'r') as f:
            annot_cam = json.load(f)
        # merge annots
        for k in annot_pose.keys():                
            for j in range(len(annot_pose[k])):
                annot_temp = {}
                annot_temp['rgb_pth']        = os.path.join(tr_obj_dir, 'rgb', '{:06d}.png'.format(int(k)))
                annot_temp['dpt_pth']        = os.path.join(tr_obj_dir, 'depth', '{:06d}.png'.format(int(k)))
                annot_temp['msk_pth']        = os.path.join(tr_obj_dir, 'mask', '{:06d}_{:06d}.png'.format(int(k), j))
                annot_temp['msk_vis_pth']    = os.path.join(tr_obj_dir, 'mask_visib', '{:06d}_{:06d}.png'.format(int(k), j))
                annot_temp['obj_id']         = annot_pose[k][j]['obj_id']
                annot_temp['cam_R_m2c']      = annot_pose[k][j]['cam_R_m2c']
                annot_temp['cam_t_m2c']      = annot_pose[k][j]['cam_t_m2c']
                annot_temp['bbox_obj']       = annot_det[k][j]['bbox_obj']
                annot_temp['bbox_visib']     = annot_det[k][j]['bbox_visib']
                annot_temp['px_count_all']   = annot_det[k][j]['px_count_all']
                annot_temp['px_count_valid'] = annot_det[k][j]['px_count_valid']
                annot_temp['px_count_visib'] = annot_det[k][j]['px_count_visib']
                annot_temp['visib_fract']    = annot_det[k][j]['visib_fract']
                annot_temp['cam_K']          = annot_cam[k]['cam_K']
                annot_temp['depth_scale']    = annot_cam[k]['depth_scale']
                annot_temp['scene_id']       = int(2)
                annot_temp['image_id']       = int(k)     
                annot_temp['score']          = 1              
                if ref.lmo_id2obj[annot_temp['obj_id']] in [cfg.pytorch.object]:
                    annot.append(annot_temp)
        self.bbox_name = 'bbox_visib' # ground-truth
        self.annot = annot
        self.nSamples = len(annot)
        # print('Loaded LineMOD {} {} samples'.format(split, self.nSamples))

    def GetPartInfo(self, index):
        """
        Get infos ProjEmbCrop, DepthCrop, box, mask, c, s  from index
        :param index:
        :return:
        """
        cls_idx = self.annot[index]['obj_id']
        rgb = cv2.imread(self.annot[index]['rgb_pth'])
        box = self.annot[index][self.bbox_name].copy()  # bbox format: [left, upper, width, height]
        c = np.array([box[1] + box[3] / 2., box[0] + box[2] / 2.])
        s = max(box[3], box[2]) * 1.5 
        s = min(s, max(self.cfg.pytorch.height, self.cfg.pytorch.width)) * 1.0
        rgb_crop = Crop_by_Pad(rgb, c, s, self.cfg.dataiter.backbone_input_res, channel=3, interpolation=cv2.INTER_LINEAR)
        return cls_idx, rgb_crop, box, c, s

    def __getitem__(self, index):
        cls_idx, rgb_crop, box, c, s = self.GetPartInfo(index)
        inp = rgb_crop.transpose(2, 0, 1).astype(np.float32) / 255.
        rot = np.array(self.annot[index]['cam_R_m2c']).reshape(3,3)
        trans = np.array(self.annot[index]['cam_t_m2c'])
        pose = np.concatenate((rot, trans.reshape(3, 1)), axis=1)
        center = c
        size = s
        box = np.asarray(box)
        imgPath = self.annot[index]['rgb_pth']
        scene_id = self.annot[index]['scene_id']
        image_id = self.annot[index]['image_id']
        score = self.annot[index]['score']
        return inp, pose, box, center, size, cls_idx, imgPath, scene_id, image_id, score

    def __len__(self):
        return self.nSamples


class TLESS(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.root_dir = ref.tless_dir 
        # print('==> initializing {} {} data.'.format(cfg.pytorch.dataset, split))          
        ## load dataset
        annot = []
        for scene_id in np.arange(1, 21):
            te_scene_dir = os.path.join(ref.tless_test_dir, '{:06d}'.format(scene_id))
            f_pose = os.path.join(te_scene_dir, 'scene_gt.json')
            f_det = os.path.join(te_scene_dir, 'scene_gt_info.json')
            f_cam = os.path.join(te_scene_dir, 'scene_camera.json')
            with open(f_pose, 'r') as f:
                annot_pose = json.load(f)
            with open(f_det, 'r') as f:
                annot_det = json.load(f)
            with open(f_cam, 'r') as f:
                annot_cam = json.load(f)
            # merge annots
            for k in annot_pose.keys():                
                for j in range(len(annot_pose[k])):
                    annot_temp = {}
                    annot_temp['obj_id']         = annot_pose[k][j]['obj_id']
                    if ref.tless_id2obj[annot_temp['obj_id']] not in [cfg.pytorch.object]:
                        continue
                    annot_temp['rgb_pth']        = os.path.join(te_scene_dir, 'rgb', '{:06d}.png'.format(int(k)))
                    annot_temp['dpt_pth']        = os.path.join(te_scene_dir, 'depth', '{:06d}.png'.format(int(k)))
                    annot_temp['msk_pth']        = os.path.join(te_scene_dir, 'mask', '{:06d}_{:06d}.png'.format(int(k), j))
                    annot_temp['msk_vis_pth']    = os.path.join(te_scene_dir, 'mask_visib', '{:06d}_{:06d}.png'.format(int(k), j))
                    annot_temp['cam_R_m2c']      = annot_pose[k][j]['cam_R_m2c']
                    annot_temp['cam_t_m2c']      = annot_pose[k][j]['cam_t_m2c']
                    annot_temp['bbox_obj']       = annot_det[k][j]['bbox_obj']
                    annot_temp['bbox_visib']     = annot_det[k][j]['bbox_visib']
                    annot_temp['px_count_all']   = annot_det[k][j]['px_count_all']
                    annot_temp['px_count_valid'] = annot_det[k][j]['px_count_valid']
                    annot_temp['px_count_visib'] = annot_det[k][j]['px_count_visib']
                    annot_temp['visib_fract']    = annot_det[k][j]['visib_fract']
                    annot_temp['cam_K']          = annot_cam[k]['cam_K']
                    annot_temp['depth_scale']    = annot_cam[k]['depth_scale']
                    annot_temp['scene_id']       = int(scene_id)
                    annot_temp['image_id']       = int(k)   
                    annot_temp['score']          = 1                                              
                    annot.append(annot_temp)
        self.bbox_name = 'bbox_visib' # ground-truth
        self.annot = annot
        self.nSamples = len(annot)
        # print('Loaded TLESS {} {} samples'.format(split, self.nSamples))

    def GetPartInfo(self, index):
        """
        Get infos ProjEmbCrop, DepthCrop, box, mask, c, s  from index
        :param index:
        :return:
        """
        cls_idx = self.annot[index]['obj_id']
        rgb = cv2.imread(self.annot[index]['rgb_pth'])
        box = self.annot[index][self.bbox_name].copy()  # bbox format: [left, upper, width, height]
        c = np.array([box[1] + box[3] / 2., box[0] + box[2] / 2.])
        s = max(box[3], box[2]) * 1.5 
        s = min(s, max(self.cfg.pytorch.height, self.cfg.pytorch.width)) * 1.0
        rgb_crop = Crop_by_Pad(rgb, c, s, self.cfg.dataiter.backbone_input_res, channel=3, interpolation=cv2.INTER_LINEAR)
        return cls_idx, rgb_crop, box, c, s

    def __getitem__(self, index):
        cls_idx, rgb_crop, box, c, s = self.GetPartInfo(index)
        inp = rgb_crop.transpose(2, 0, 1).astype(np.float32) / 255.
        rot = np.array(self.annot[index]['cam_R_m2c']).reshape(3,3)
        trans = np.array(self.annot[index]['cam_t_m2c'])
        pose = np.concatenate((rot, trans.reshape(3, 1)), axis=1)
        center = c
        size = s
        box = np.asarray(box)
        imgPath = self.annot[index]['rgb_pth']
        scene_id = self.annot[index]['scene_id']
        image_id = self.annot[index]['image_id']
        score = self.annot[index]['score']
        K = self.annot[index]['cam_K']
        return inp, pose, box, center, size, cls_idx, K, scene_id, image_id, score

    def __len__(self):
        return self.nSamples


class TUDL(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.root_dir = ref.tudl_root
        # print('==> initializing {} {} data.'.format(cfg.pytorch.dataset, split))  
        ## load dataset
        annot = []
        for obj in [self.cfg.pytorch.object]:
            obj_id = ref.tudl_obj2id(obj)
            te_scene_dir = os.path.join(self.root_dir, '{}'.format(self.split.replace('val', 'test')), '{:06d}'.format(obj_id))
            f_pose = os.path.join(te_scene_dir, 'scene_gt.json')
            f_det = os.path.join(te_scene_dir, 'scene_gt_info.json')
            f_cam = os.path.join(te_scene_dir, 'scene_camera.json')
            with open(f_pose, 'r') as f:
                annot_pose = json.load(f)
            with open(f_det, 'r') as f:
                annot_det = json.load(f)
            with open(f_cam, 'r') as f:
                annot_cam = json.load(f)
            # merge annots
            for k in annot_pose.keys():                
                for j in range(len(annot_pose[k])):
                    annot_temp = {}
                    annot_temp['obj_id']         = annot_pose[k][j]['obj_id']
                    if int(annot_temp['obj_id']) != obj_id:
                        continue
                    annot_temp['rgb_pth']        = os.path.join(te_scene_dir, 'rgb', '{:06d}.png'.format(int(k)))
                    annot_temp['dpt_pth']        = os.path.join(te_scene_dir, 'depth', '{:06d}.png'.format(int(k)))
                    annot_temp['msk_pth']        = os.path.join(te_scene_dir, 'mask', '{:06d}_{:06d}.png'.format(int(k), j))
                    annot_temp['msk_vis_pth']    = os.path.join(te_scene_dir, 'mask_visib', '{:06d}_{:06d}.png'.format(int(k), j))
                    annot_temp['cam_R_m2c']      = annot_pose[k][j]['cam_R_m2c']
                    annot_temp['cam_t_m2c']      = annot_pose[k][j]['cam_t_m2c']
                    annot_temp['bbox_obj']       = annot_det[k][j]['bbox_obj']
                    annot_temp['bbox_visib']     = annot_det[k][j]['bbox_visib']
                    annot_temp['px_count_all']   = annot_det[k][j]['px_count_all']
                    annot_temp['px_count_valid'] = annot_det[k][j]['px_count_valid']
                    annot_temp['px_count_visib'] = annot_det[k][j]['px_count_visib']
                    annot_temp['visib_fract']    = annot_det[k][j]['visib_fract']
                    annot_temp['cam_K']          = annot_cam[k]['cam_K']
                    annot_temp['depth_scale']    = annot_cam[k]['depth_scale']
                    annot_temp['scene_id']       = int(obj_id)
                    annot_temp['image_id']       = int(k)        
                    annot_temp['score']          = 1                                                                   
                    annot.append(annot_temp)
        self.bbox_name = 'bbox_visib' # ground-truth
        self.annot = annot
        self.nSamples = len(annot)
        # print('Loaded TUDL {} {} samples'.format(split, self.nSamples))

    def GetPartInfo(self, index):
        """
        Get infos coor_crop, DepthCrop, mask, c, s  from index
        :param index:
        :return:
        """       
        cls_idx = self.annot[index]['obj_id']
        rgb = cv2.imread(self.annot[index]['rgb_pth'])
        box = self.annot[index][self.bbox_name].copy()  # bbox format: [left, upper, width, height]
        c = np.array([box[1] + box[3] / 2., box[0] + box[2] / 2.])
        s = max(box[3], box[2]) * 1.5 
        s = min(s, max(self.cfg.pytorch.height, self.cfg.pytorch.width)) * 1.0
        rgb_crop = Crop_by_Pad(rgb, c, s, self.cfg.dataiter.backbone_input_res, channel=3, interpolation=cv2.INTER_LINEAR)
        return cls_idx, rgb_crop, box, c, s

    def __getitem__(self, index):
        cls_idx, rgb_crop, box, c, s = self.GetPartInfo(index)
        inp = rgb_crop.transpose(2, 0, 1).astype(np.float32) / 255.
        rot = np.array(self.annot[index]['cam_R_m2c']).reshape(3,3)
        trans = np.array(self.annot[index]['cam_t_m2c'])
        pose = np.concatenate((rot, trans.reshape(3, 1)), axis=1)
        center = c
        size = s
        box = np.asarray(box)
        imgPath = self.annot[index]['rgb_pth']
        scene_id = self.annot[index]['scene_id']
        image_id = self.annot[index]['image_id']
        score = self.annot[index]['score']
        return inp, pose, box, center, size, cls_idx, imgPath, scene_id, image_id, score

    def __len__(self):
        return self.nSamples


class YCBV(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.root_dir = ref.ycbv_dir
        # # print('==> initializing {} {} data.'.format(cfg.pytorch.dataset, split))         
        ## load dataset
        annot = []
        for scene in ref.ycbv_test_scenes:
            te_scene_dir = os.path.join(self.root_dir, '{}'.format(self.split.replace('val', 'test')), scene)
            f_pose = os.path.join(te_scene_dir, 'scene_gt.json')
            f_det = os.path.join(te_scene_dir, 'scene_gt_info.json')
            f_cam = os.path.join(te_scene_dir, 'scene_camera.json')
            with open(f_pose, 'r') as f:
                annot_pose = json.load(f)
            with open(f_det, 'r') as f:
                annot_det = json.load(f)
            with open(f_cam, 'r') as f:
                annot_cam = json.load(f)
            # merge annots
            for k in annot_pose.keys():                
                for j in range(len(annot_pose[k])):
                    annot_temp = {}
                    annot_temp['obj_id']         = annot_pose[k][j]['obj_id']
                    if ref.ycbv_id2obj[annot_temp['obj_id']] not in [self.cfg.pytorch.object]:
                        continue
                    annot_temp['rgb_pth']        = os.path.join(te_scene_dir, 'rgb', '{:06d}.png'.format(int(k)))
                    annot_temp['dpt_pth']        = os.path.join(te_scene_dir, 'depth', '{:06d}.png'.format(int(k)))
                    annot_temp['msk_pth']        = os.path.join(te_scene_dir, 'mask', '{:06d}_{:06d}.png'.format(int(k), j))
                    annot_temp['msk_vis_pth']    = os.path.join(te_scene_dir, 'mask_visib', '{:06d}_{:06d}.png'.format(int(k), j))
                    annot_temp['cam_R_m2c']      = annot_pose[k][j]['cam_R_m2c']
                    annot_temp['cam_t_m2c']      = annot_pose[k][j]['cam_t_m2c']
                    annot_temp['bbox_obj']       = annot_det[k][j]['bbox_obj']
                    annot_temp['bbox_visib']     = annot_det[k][j]['bbox_visib']
                    annot_temp['px_count_all']   = annot_det[k][j]['px_count_all']
                    annot_temp['px_count_valid'] = annot_det[k][j]['px_count_valid']
                    annot_temp['px_count_visib'] = annot_det[k][j]['px_count_visib']
                    annot_temp['visib_fract']    = annot_det[k][j]['visib_fract']
                    annot_temp['cam_K']          = annot_cam[k]['cam_K']
                    annot_temp['depth_scale']    = annot_cam[k]['depth_scale']
                    annot_temp['scene_id']       = int(scene)
                    annot_temp['image_id']       = int(k)               
                    annot_temp['score']          = 1                                                      
                    annot.append(annot_temp)
        self.bbox_name = 'bbox_visib' # ground-truth
        self.annot = annot
        self.nSamples = len(annot)
        # # print('Loaded YCBV {} {} samples'.format(split, self.nSamples))

    def GetPartInfo(self, index):       
        """
        Get infos coor_crop, DepthCrop, box, mask, c, s  from index
        :param index:
        :return:
        """
        cls_idx = self.annot[index]['obj_id']
        rgb = cv2.imread(self.annot[index]['rgb_pth'])
        box = self.annot[index][self.bbox_name].copy()  # bbox format: [left, upper, width, height]
        c = np.array([box[1] + box[3] / 2., box[0] + box[2] / 2.])
        s = max(box[3], box[2]) * 1.5 
        s = min(s, max(self.cfg.pytorch.height, self.cfg.pytorch.width)) * 1.0
        rgb_crop = Crop_by_Pad(rgb, c, s, self.cfg.dataiter.backbone_input_res, channel=3, interpolation=cv2.INTER_LINEAR)
        return cls_idx, rgb_crop, box, c, s

    def __getitem__(self, index):
        cls_idx, rgb_crop, box, c, s = self.GetPartInfo(index)
        inp = rgb_crop.transpose(2, 0, 1).astype(np.float32) / 255.
        rot = np.array(self.annot[index]['cam_R_m2c']).reshape(3,3)
        trans = np.array(self.annot[index]['cam_t_m2c'])
        pose = np.concatenate((rot, trans.reshape(3, 1)), axis=1)
        center = c
        size = s
        box = np.asarray(box)
        imgPath = self.annot[index]['rgb_pth']
        scene_id = self.annot[index]['scene_id']
        image_id = self.annot[index]['image_id']
        score = self.annot[index]['score']
        return inp, pose, box, center, size, cls_idx, imgPath, scene_id, image_id, score

    def __len__(self):
        return self.nSamples


class HB(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.root_dir = ref.hb_dir
        # print('==> initializing {} {} data.'.format(cfg.pytorch.dataset, split))
        ## load dataset
        annot = []
        for scene in ref.hb_val_scenes:
            test_obj_dir = os.path.join(self.root_dir, 'val', scene)
            f_pose = os.path.join(test_obj_dir, 'scene_gt.json')
            f_det = os.path.join(test_obj_dir, 'scene_gt_info.json')
            f_cam = os.path.join(test_obj_dir, 'scene_camera.json')
            with open(f_pose, 'r') as f:
                annot_pose = json.load(f)
            with open(f_det, 'r') as f:
                annot_det = json.load(f)
            with open(f_cam, 'r') as f:
                annot_cam = json.load(f)
            # merge annots
            for k in annot_pose.keys():  
                for j in range(len(annot_pose[k])):
                    annot_temp = {}
                    annot_temp['obj_id']         = annot_pose[k][j]['obj_id']
                    if ref.hb_id2obj[annot_temp['obj_id']] not in [self.cfg.pytorch.object]:
                        continue
                    annot_temp['rgb_pth']        = os.path.join(test_obj_dir, 'rgb', '{:06d}.png'.format(int(k)))
                    annot_temp['dpt_pth']        = os.path.join(test_obj_dir, 'depth', '{:06d}.png'.format(int(k)))
                    annot_temp['msk_pth']        = os.path.join(test_obj_dir, 'mask', '{:06d}_000000.png'.format(int(k)))
                    annot_temp['msk_vis_pth']    = os.path.join(test_obj_dir, 'mask_visib', '{:06d}_000000.png'.format(int(k)))
                    annot_temp['coor_pth']       = os.path.join(test_obj_dir, 'coor', '{:06d}.npy'.format(int(k)))                        
                    annot_temp['cam_R_m2c']      = annot_pose[k][j]['cam_R_m2c']
                    annot_temp['cam_t_m2c']      = annot_pose[k][j]['cam_t_m2c']
                    annot_temp['bbox_obj']       = annot_det[k][j]['bbox_obj']
                    annot_temp['bbox_visib']     = annot_det[k][j]['bbox_visib']
                    annot_temp['px_count_all']   = annot_det[k][j]['px_count_all']
                    annot_temp['px_count_valid'] = annot_det[k][j]['px_count_valid']
                    annot_temp['px_count_visib'] = annot_det[k][j]['px_count_visib']
                    annot_temp['visib_fract']    = annot_det[k][j]['visib_fract']
                    annot_temp['cam_K']          = annot_cam[k]['cam_K']
                    annot_temp['depth_scale']    = annot_cam[k]['depth_scale']
                    annot_temp['scene_id']       = int(scene)
                    annot_temp['image_id']       = int(k)    
                    annot_temp['score']          = 1                                                                         
                    annot.append(annot_temp)
        self.bbox_name = 'bbox_visib' # ground-truth
        self.annot = annot
        self.nSamples = len(annot)
        # print('Loaded HB {} {} samples'.format(split, self.nSamples))

    def GetPartInfo(self, index):
        """
        Get infos coor_crop, DepthCrop, mask, c, s  from index
        :param index:
        :return:
        """       
        cls_idx = self.annot[index]['obj_id']
        rgb = cv2.imread(self.annot[index]['rgb_pth'])
        box = self.annot[index][self.bbox_name].copy()  # bbox format: [left, upper, width, height]
        c = np.array([box[1] + box[3] / 2., box[0] + box[2] / 2.])
        s = max(box[3], box[2]) * 1.5 
        s = min(s, max(self.cfg.pytorch.height, self.cfg.pytorch.width)) * 1.0
        rgb_crop = Crop_by_Pad(rgb, c, s, self.cfg.dataiter.backbone_input_res, channel=3, interpolation=cv2.INTER_LINEAR)
        return cls_idx, rgb_crop, box, c, s

    def __getitem__(self, index):
        cls_idx, rgb_crop, box, c, s = self.GetPartInfo(index)
        inp = rgb_crop.transpose(2, 0, 1).astype(np.float32) / 255.
        rot = np.array(self.annot[index]['cam_R_m2c']).reshape(3,3)
        trans = np.array(self.annot[index]['cam_t_m2c'])
        pose = np.concatenate((rot, trans.reshape(3, 1)), axis=1)
        center = c
        size = s
        box = np.asarray(box)
        imgPath = self.annot[index]['rgb_pth']
        scene_id = self.annot[index]['scene_id']
        image_id = self.annot[index]['image_id']
        score = self.annot[index]['score']
        return inp, pose, box, center, size, cls_idx, imgPath, scene_id, image_id, score

    def __len__(self):
        return self.nSamples


class ICBIN(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.root_dir = ref.icbin_dir
        # print('==> initializing {} {} data.'.format(cfg.pytorch.dataset, split))        
        ## load dataset
        annot = []
        for scene in ref.icbin_test_scenes:
            test_obj_dir = os.path.join(self.root_dir, 'test', scene)
            f_pose = os.path.join(test_obj_dir, 'scene_gt.json')
            f_det = os.path.join(test_obj_dir, 'scene_gt_info.json')
            f_cam = os.path.join(test_obj_dir, 'scene_camera.json')
            with open(f_pose, 'r') as f:
                annot_pose = json.load(f)
            with open(f_det, 'r') as f:
                annot_det = json.load(f)
            with open(f_cam, 'r') as f:
                annot_cam = json.load(f)
            # merge annots
            for k in annot_pose.keys():  
                for j in range(len(annot_pose[k])):
                    annot_temp = {}
                    annot_temp['obj_id']         = annot_pose[k][j]['obj_id']
                    if ref.icbin_id2obj[annot_temp['obj_id']] not in [self.cfg.pytorch.object]:
                        continue
                    annot_temp['rgb_pth']        = os.path.join(test_obj_dir, 'rgb', '{:06d}.png'.format(int(k)))
                    annot_temp['dpt_pth']        = os.path.join(test_obj_dir, 'depth', '{:06d}.png'.format(int(k)))
                    annot_temp['msk_pth']        = os.path.join(test_obj_dir, 'mask', '{:06d}_{:06d}.png'.format(int(k), int(j)))
                    annot_temp['msk_vis_pth']    = os.path.join(test_obj_dir, 'mask_visib', '{:06d}_{:06d}.png'.format(int(k), int(j)))
                    annot_temp['coor_pth']       = os.path.join(test_obj_dir, 'coor', '{:06d}.npy'.format(int(k)))                        
                    annot_temp['cam_R_m2c']      = annot_pose[k][j]['cam_R_m2c']
                    annot_temp['cam_t_m2c']      = annot_pose[k][j]['cam_t_m2c']
                    annot_temp['bbox_obj']       = annot_det[k][j]['bbox_obj']
                    annot_temp['bbox_visib']     = annot_det[k][j]['bbox_visib']
                    annot_temp['px_count_all']   = annot_det[k][j]['px_count_all']
                    annot_temp['px_count_valid'] = annot_det[k][j]['px_count_valid']
                    annot_temp['px_count_visib'] = annot_det[k][j]['px_count_visib']
                    annot_temp['visib_fract']    = annot_det[k][j]['visib_fract']
                    annot_temp['cam_K']          = annot_cam[k]['cam_K']
                    annot_temp['depth_scale']    = annot_cam[k]['depth_scale']
                    annot_temp['scene_id']       = int(scene)
                    annot_temp['image_id']       = int(k)                       
                    annot_temp['score']          = 1                                              
                    annot.append(annot_temp)
        self.bbox_name = 'bbox_visib' # ground-truth
        self.annot = annot
        self.nSamples = len(annot)
        # print('Loaded ICBIN {} {} samples'.format(split, self.nSamples))

    def GetPartInfo(self, index):
        """
        Get infos ProjEmbCrop, DepthCrop, box, mask, c, s  from index
        :param index:
        :return:
        """
        cls_idx = self.annot[index]['obj_id']
        rgb = cv2.imread(self.annot[index]['rgb_pth'])
        box = self.annot[index][self.bbox_name].copy()  # bbox format: [left, upper, width, height]
        c = np.array([box[1] + box[3] / 2., box[0] + box[2] / 2.]) 
        s = max(box[3], box[2]) * 1.5
        s = min(s, max(self.cfg.pytorch.height, self.cfg.pytorch.width)) * 1.0
        rgb_crop = Crop_by_Pad(rgb, c, s, self.cfg.dataiter.backbone_input_res, channel=3, interpolation=cv2.INTER_LINEAR)
        return cls_idx, rgb_crop, box, c, s

    def __getitem__(self, index):
        cls_idx, rgb_crop, box, c, s = self.GetPartInfo(index)
        inp = rgb_crop.transpose(2, 0, 1).astype(np.float32) / 255.
        rot = np.array(self.annot[index]['cam_R_m2c']).reshape(3,3)
        trans = np.array(self.annot[index]['cam_t_m2c'])
        pose = np.concatenate((rot, trans.reshape(3, 1)), axis=1)
        center = c
        size = s
        imgPath = self.annot[index]['rgb_pth']
        box = np.asarray(box)
        scene_id = self.annot[index]['scene_id']
        image_id = self.annot[index]['image_id']
        score = self.annot[index]['score']
        return inp, pose, box, center, size, cls_idx, imgPath, scene_id, image_id, score

    def __len__(self):
        return self.nSamples        


class ITODD(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.root_dir = ref.itodd_dir
        # print('==> initializing {} {} data.'.format(cfg.pytorch.dataset, split))
        ## load dataset
        annot = []
        for scene in ref.itodd_val_scenes:
            test_obj_dir = os.path.join(self.root_dir, 'val', scene)
            f_pose = os.path.join(test_obj_dir, 'scene_gt.json')
            f_det = os.path.join(test_obj_dir, 'scene_gt_info.json')
            f_cam = os.path.join(test_obj_dir, 'scene_camera.json')
            with open(f_pose, 'r') as f:
                annot_pose = json.load(f)
            with open(f_det, 'r') as f:
                annot_det = json.load(f)
            with open(f_cam, 'r') as f:
                annot_cam = json.load(f)
            # merge annots
            for k in annot_pose.keys():  
                for j in range(len(annot_pose[k])):
                    annot_temp = {}
                    annot_temp['obj_id']         = annot_pose[k][j]['obj_id']
                    if ref.itodd_id2obj[annot_temp['obj_id']] not in [self.cfg.pytorch.object]:
                        continue
                    annot_temp['gray_pth']        = os.path.join(test_obj_dir, 'gray', '{:06d}.tif'.format(int(k)))
                    annot_temp['dpt_pth']        = os.path.join(test_obj_dir, 'depth', '{:06d}.png'.format(int(k)))
                    annot_temp['msk_pth']        = os.path.join(test_obj_dir, 'mask', '{:06d}_{:06d}.png'.format(int(k), int(j)))
                    annot_temp['msk_vis_pth']    = os.path.join(test_obj_dir, 'mask_visib', '{:06d}_{:06d}.png'.format(int(k), int(j)))
                    annot_temp['coor_pth']       = os.path.join(test_obj_dir, 'coor', '{:06d}.npy'.format(int(k)))                        
                    annot_temp['cam_R_m2c']      = annot_pose[k][j]['cam_R_m2c']
                    annot_temp['cam_t_m2c']      = annot_pose[k][j]['cam_t_m2c']
                    annot_temp['bbox_obj']       = annot_det[k][j]['bbox_obj']
                    annot_temp['bbox_visib']     = annot_det[k][j]['bbox_visib']
                    annot_temp['px_count_all']   = annot_det[k][j]['px_count_all']
                    annot_temp['px_count_valid'] = annot_det[k][j]['px_count_valid']
                    annot_temp['px_count_visib'] = annot_det[k][j]['px_count_visib']
                    annot_temp['visib_fract']    = annot_det[k][j]['visib_fract']
                    annot_temp['cam_K']          = annot_cam[k]['cam_K']
                    annot_temp['depth_scale']    = annot_cam[k]['depth_scale']
                    annot_temp['scene_id']       = int(scene)
                    annot_temp['image_id']       = int(k)   
                    annot_temp['score']          = 1                                                                  
                    annot.append(annot_temp)
        self.bbox_name = 'bbox_visib' # ground-truth
        self.annot = annot
        self.nSamples = len(annot)
        # print('Loaded ITODD {} {} samples'.format(split, self.nSamples))

    def GetPartInfo(self, index):       
        """
        Get infos coor_crop, DepthCrop, box, mask, c, s  from index
        :param index:
        :return:
        """
        cls_idx = self.annot[index]['obj_id']        
        rgb = cv2.imread(self.annot[index]['gray_pth'])
        box = self.annot[index][self.bbox_name].copy()  # bbox format: [left, upper, width, height]
        c = np.array([box[1] + box[3] / 2., box[0] + box[2] / 2.])
        s = max(box[3], box[2]) * 1.5 
        s = min(s, max(self.cfg.pytorch.height, self.cfg.pytorch.width)) * 1.0
        rgb_crop = Crop_by_Pad(rgb, c, s, self.cfg.dataiter.backbone_input_res, channel=3, interpolation=cv2.INTER_LINEAR)
        return cls_idx, rgb_crop, box, c, s

    def __getitem__(self, index):
        cls_idx, rgb_crop, box, c, s = self.GetPartInfo(index)       
        inp = rgb_crop[:, :, 0][None, :, :].astype(np.float32) / 255.
        rot = np.array(self.annot[index]['cam_R_m2c']).reshape(3,3)
        trans = np.array(self.annot[index]['cam_t_m2c'])
        pose = np.concatenate((rot, trans.reshape(3, 1)), axis=1)
        center = c
        size = s
        box = np.asarray(box)
        imgPath = self.annot[index]['gray_pth']
        scene_id = self.annot[index]['scene_id']
        image_id = self.annot[index]['image_id']
        score = self.annot[index]['score']
        return inp, pose, box, center, size, cls_idx, imgPath, scene_id, image_id, score

    def __len__(self):
        return self.nSamples

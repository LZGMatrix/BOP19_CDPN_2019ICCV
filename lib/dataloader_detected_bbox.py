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
        te_obj_dir = os.path.join(self.root_dir, 'test/000002')
        f_bbox = os.path.join(ref.bbox_dir, 'lmo', 'lmo_000002_detection_result.json')
        with open(f_bbox, 'r') as f:
            annot_bbox = json.load(f)
        # merge annots
        for k in annot_bbox.keys():       
            score = {}
            for i in range(len(annot_bbox[k])):
                if annot_bbox[k][i]['obj_id'] not in score.keys():
                    score[annot_bbox[k][i]['obj_id']] = []
                score[annot_bbox[k][i]['obj_id']].append(annot_bbox[k][i]['score'])         
            for l in range(len(annot_bbox[k])):
                annot_temp = {}
                annot_temp['obj_id']         = annot_bbox[k][l]['obj_id']
                if ref.lmo_id2obj[annot_temp['obj_id']] not in [cfg.pytorch.object]:
                    continue
                if annot_bbox[k][l]['score'] == max(score[annot_temp['obj_id']]) and max(score[annot_temp['obj_id']]) > 0.2:
                    annot_temp['bbox'] = annot_bbox[k][l]['bbox']
                    annot_temp['score'] = annot_bbox[k][l]['score']
                else:
                    continue
                annot_temp['rgb_pth']        = os.path.join(te_obj_dir, 'rgb', '{:06d}.png'.format(int(k)))   
                annot_temp['scene_id'] = 2
                annot_temp['image_id'] = int(k)                             
                annot.append(annot_temp)
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
        box = self.annot[index]['bbox'].copy()  # bbox format: [left, upper, width, height]
        c = np.array([box[1] + box[3] / 2., box[0] + box[2] / 2.])
        s = max(box[3], box[2]) * 1.5 
        s = min(s, max(self.cfg.pytorch.height, self.cfg.pytorch.width)) * 1.0
        rgb_crop = Crop_by_Pad(rgb, c, s, self.cfg.dataiter.backbone_input_res, channel=3, interpolation=cv2.INTER_LINEAR)
        return cls_idx, rgb_crop, box, c, s

    def __getitem__(self, index):
        cls_idx, rgb_crop, box, c, s = self.GetPartInfo(index)
        inp = rgb_crop.transpose(2, 0, 1).astype(np.float32) / 255.
        pose = np.zeros((3, 4))
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
            f_bbox = os.path.join(ref.bbox_dir, 'tless', 'tless_{:06d}_detection_result.json'.format(scene_id))
            with open(f_bbox, 'r') as f:
                annot_bbox = json.load(f)
            # merge annots
            for k in annot_bbox.keys():                
                for l in range(len(annot_bbox[k])):
                    annot_temp = {}
                    annot_temp['obj_id']         = annot_bbox[k][l]['obj_id']
                    if ref.tless_id2obj[annot_temp['obj_id']] not in [cfg.pytorch.object]:
                        continue
                    if annot_bbox[k][l]['score'] < 0.2:
                        continue
                    annot_temp['bbox'] = annot_bbox[k][l]['bbox']
                    annot_temp['score'] = annot_bbox[k][l]['score']
                    annot_temp['rgb_pth']        = os.path.join(te_scene_dir, 'rgb', '{:06d}.png'.format(int(k)))
                    annot_temp['scene_id'] = int(scene_id)
                    annot_temp['image_id'] = int(k)
                    annot.append(annot_temp)
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
        box = self.annot[index]['bbox'].copy()  # bbox format: [left, upper, width, height]
        c = np.array([box[1] + box[3] / 2., box[0] + box[2] / 2.])
        s = max(box[3], box[2]) * 1.5 
        s = min(s, max(self.cfg.pytorch.height, self.cfg.pytorch.width)) * 1.0
        rgb_crop = Crop_by_Pad(rgb, c, s, self.cfg.dataiter.backbone_input_res, channel=3, interpolation=cv2.INTER_LINEAR)
        return cls_idx, rgb_crop, box, c, s

    def __getitem__(self, index):
        cls_idx, rgb_crop, box, c, s = self.GetPartInfo(index)
        inp = rgb_crop.transpose(2, 0, 1).astype(np.float32) / 255.
        pose = np.zeros((3, 4))
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
            f_bbox = os.path.join(ref.bbox_dir, 'tudl', 'tudl_{:06d}_detection_result.json'.format(obj_id))
            with open(f_bbox, 'r') as f:
                annot_bbox = json.load(f)
            # merge annots
            for k in annot_bbox.keys():     
                score = {}
                for i in range(len(annot_bbox[k])):
                    if annot_bbox[k][i]['obj_id'] not in score.keys():
                        score[annot_bbox[k][i]['obj_id']] = []
                    score[annot_bbox[k][i]['obj_id']].append(annot_bbox[k][i]['score'])           
                for l in range(len(annot_bbox[k])):
                    annot_temp = {}
                    annot_temp['obj_id']         = annot_bbox[k][l]['obj_id']
                    if int(annot_temp['obj_id']) != obj_id:
                        continue
                    if annot_bbox[k][l]['score'] == max(score[annot_temp['obj_id']]) and max(score[annot_temp['obj_id']]) > 0.2:
                        annot_temp['bbox'] = annot_bbox[k][l]['bbox']
                        annot_temp['score'] = annot_bbox[k][l]['score']
                    else:
                        continue
                    annot_temp['rgb_pth']        = os.path.join(te_scene_dir, 'rgb', '{:06d}.png'.format(int(k)))
                    annot_temp['scene_id'] = int(obj_id)
                    annot_temp['image_id'] = int(k)                    
                    annot.append(annot_temp)
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
        box = self.annot[index]['bbox'].copy()  # bbox format: [left, upper, width, height]
        c = np.array([box[1] + box[3] / 2., box[0] + box[2] / 2.])
        s = max(box[3], box[2]) * 1.5 
        s = min(s, max(self.cfg.pytorch.height, self.cfg.pytorch.width)) * 1.0
        rgb_crop = Crop_by_Pad(rgb, c, s, self.cfg.dataiter.backbone_input_res, channel=3, interpolation=cv2.INTER_LINEAR)
        return cls_idx, rgb_crop, box, c, s

    def __getitem__(self, index):
        cls_idx, rgb_crop, box, c, s = self.GetPartInfo(index)
        inp = rgb_crop.transpose(2, 0, 1).astype(np.float32) / 255.
        pose = np.zeros((3, 4))
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
            f_bbox = os.path.join(ref.bbox_dir, 'ycbv', 'ycbv_{:06d}_detection_result.json'.format(int(scene)))
            with open(f_pose, 'r') as f:
                annot_pose = json.load(f)
            with open(f_det, 'r') as f:
                annot_det = json.load(f)
            with open(f_cam, 'r') as f:
                annot_cam = json.load(f)
            with open(f_bbox, 'r') as f:
                annot_bbox = json.load(f)
            # merge annots
            for k in annot_bbox.keys():  
                score = {}
                for i in range(len(annot_bbox[k])):
                    if annot_bbox[k][i]['obj_id'] not in score.keys():
                        score[annot_bbox[k][i]['obj_id']] = []
                    score[annot_bbox[k][i]['obj_id']].append(annot_bbox[k][i]['score'])              
                for l in range(len(annot_bbox[k])):
                    annot_temp = {}
                    annot_temp['obj_id']         = annot_bbox[k][l]['obj_id']
                    if ref.ycbv_id2obj[annot_temp['obj_id']] not in [self.cfg.pytorch.object]:
                        continue
                    if annot_bbox[k][l]['score'] == max(score[annot_temp['obj_id']]) and max(score[annot_temp['obj_id']]) > 0.2:
                        annot_temp['bbox'] = annot_bbox[k][l]['bbox']
                        annot_temp['score'] = annot_bbox[k][l]['score']
                    else:
                        continue
                    annot_temp['rgb_pth']        = os.path.join(te_scene_dir, 'rgb', '{:06d}.png'.format(int(k)))
                    annot_temp['scene_id'] = int(scene)
                    annot_temp['image_id'] = int(k)                    
                    annot.append(annot_temp)
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
        box = self.annot[index]['bbox'].copy()  # bbox format: [left, upper, width, height]
        c = np.array([box[1] + box[3] / 2., box[0] + box[2] / 2.])
        s = max(box[3], box[2]) * 1.5 
        s = min(s, max(self.cfg.pytorch.height, self.cfg.pytorch.width)) * 1.0
        rgb_crop = Crop_by_Pad(rgb, c, s, self.cfg.dataiter.backbone_input_res, channel=3, interpolation=cv2.INTER_LINEAR)
        return cls_idx, rgb_crop, box, c, s

    def __getitem__(self, index):
        cls_idx, rgb_crop, box, c, s = self.GetPartInfo(index)
        inp = rgb_crop.transpose(2, 0, 1).astype(np.float32) / 255.
        pose = np.zeros((3, 4))
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
        # test set
        for scene in ref.hb_val_scenes:
            test_obj_dir = os.path.join(self.root_dir, 'test', scene)
            f_bbox = os.path.join(ref.bbox_dir, 'hb', 'hb_{:06d}_detection_result.json'.format(int(scene)))
            with open(f_bbox, 'r') as f:
                annot_bbox = json.load(f)
            for k in annot_bbox.keys():  
                score = {}
                for i in range(len(annot_bbox[k])):
                    cur_obj_id = annot_bbox[k][i]['obj_id']
                    if cur_obj_id not in score.keys():
                        score[cur_obj_id] = []
                    score[cur_obj_id].append(annot_bbox[k][i]['score'])
                for l in range(len(annot_bbox[k])):
                    annot_temp = {}
                    annot_temp['obj_id']         = annot_bbox[k][l]['obj_id']
                    if str(annot_temp['obj_id']) not in [self.cfg.pytorch.object]:
                        continue
                    cur_max_score =  max(score[annot_temp['obj_id']])
                    if cur_max_score < 0.2:
                        continue                        
                    if annot_bbox[k][l]['score'] != cur_max_score:
                        continue 
                    annot_temp['bbox'] = annot_bbox[k][l]['bbox']
                    annot_temp['score'] = annot_bbox[k][l]['score']
                    annot_temp['rgb_pth'] = os.path.join(test_obj_dir, 'rgb', '{:06d}.png'.format(int(k)))     
                    annot_temp['scene_id'] = int(scene)
                    annot_temp['image_id'] = int(k)                                              
                    annot.append(annot_temp)
        '''
        # validation set
        for scene in ref.hb_val_scenes:
            test_obj_dir = os.path.join(self.root_dir, 'val', scene)
            f_bbox = os.path.join(ref.bbox_dir, 'hb_val', 'hb_{:06d}_val_result.json'.format(int(scene)))
            with open(f_bbox, 'r') as f:
                annot_bbox = json.load(f)
            for k in annot_bbox.keys():  
                score = {}
                for i in range(len(annot_bbox[k])):
                    cur_obj_id = annot_bbox[k][i]['obj_id']
                    if cur_obj_id not in score.keys():
                        score[cur_obj_id] = []
                    score[cur_obj_id].append(annot_bbox[k][i]['score'])
                for l in range(len(annot_bbox[k])):
                    annot_temp = {}
                    annot_temp['obj_id']         = annot_bbox[k][l]['obj_id']
                    if str(annot_temp['obj_id']) not in [self.cfg.pytorch.object]:
                        continue
                    cur_max_score =  max(score[annot_temp['obj_id']])
                    if cur_max_score < 0.2:
                        continue                        
                    if annot_bbox[k][l]['score'] != cur_max_score:
                        continue 
                    annot_temp['bbox'] = annot_bbox[k][l]['bbox']
                    annot_temp['score'] = annot_bbox[k][l]['score']
                    annot_temp['rgb_pth'] = os.path.join(test_obj_dir, 'rgb', '{:06d}.png'.format(int(k)))     
                    annot_temp['scene_id'] = int(scene)
                    annot_temp['image_id'] = int(k)                                              
                    annot.append(annot_temp)
        '''
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
        box = self.annot[index]['bbox'].copy()  # bbox format: [left, upper, width, height]
        c = np.array([box[1] + box[3] / 2., box[0] + box[2] / 2.])
        s = max(box[3], box[2]) * 1.5 
        s = min(s, max(self.cfg.pytorch.height, self.cfg.pytorch.width)) * 1.0
        rgb_crop = Crop_by_Pad(rgb, c, s, self.cfg.dataiter.backbone_input_res, channel=3, interpolation=cv2.INTER_LINEAR)
        return cls_idx, rgb_crop, box, c, s

    def __getitem__(self, index):
        cls_idx, rgb_crop, box, c, s = self.GetPartInfo(index)
        inp = rgb_crop.transpose(2, 0, 1).astype(np.float32) / 255.
        pose = np.zeros((3, 4))
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
            f_bbox = os.path.join(ref.bbox_dir, 'icbin', 'icbin_{:06d}_detection_result.json'.format(int(scene)))
            with open(f_bbox, 'r') as f:
                annot_bbox = json.load(f)
            # merge annots
            for k in annot_bbox.keys():  
                for l in range(len(annot_bbox[k])):
                    annot_temp = {}
                    annot_temp['obj_id']         = annot_bbox[k][l]['obj_id']
                    if ref.icbin_id2obj[annot_temp['obj_id']] not in [self.cfg.pytorch.object]:
                        continue
                    if annot_bbox[k][l]['score'] < 0.2:
                        continue
                    annot_temp['bbox'] = annot_bbox[k][l]['bbox']
                    annot_temp['score'] = annot_bbox[k][l]['score']
                    annot_temp['rgb_pth'] = os.path.join(test_obj_dir, 'rgb', '{:06d}.png'.format(int(k)))
                    annot_temp['scene_id'] = int(scene)
                    annot_temp['image_id'] = int(k)                       
                    annot.append(annot_temp)
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
        box = self.annot[index]['bbox'].copy()  # bbox format: [left, upper, width, height]
        c = np.array([box[1] + box[3] / 2., box[0] + box[2] / 2.]) 
        s = max(box[3], box[2]) * 1.5
        s = min(s, max(self.cfg.pytorch.height, self.cfg.pytorch.width)) * 1.0
        rgb_crop = Crop_by_Pad(rgb, c, s, self.cfg.dataiter.backbone_input_res, channel=3, interpolation=cv2.INTER_LINEAR)
        return cls_idx, rgb_crop, box, c, s

    def __getitem__(self, index):
        cls_idx, rgb_crop, box, c, s = self.GetPartInfo(index)
        inp = rgb_crop.transpose(2, 0, 1).astype(np.float32) / 255.
        pose = np.zeros((3, 4))
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
        # test set
        for scene in ref.itodd_val_scenes:
            test_obj_dir = os.path.join(self.root_dir, 'test', scene)
            f_bbox = os.path.join(ref.bbox_dir, 'itodd', 'itodd_{:06d}_detection_result.json'.format(int(scene)))
            with open(f_bbox, 'r') as f:
                annot_bbox = json.load(f)
            # merge annots
            for k in annot_bbox.keys():  
                for l in range(len(annot_bbox[k])):
                    annot_temp = {}
                    annot_temp['obj_id']         = annot_bbox[k][l]['obj_id']
                    if ref.itodd_id2obj[annot_temp['obj_id']] not in [self.cfg.pytorch.object]:
                        continue
                    if annot_bbox[k][l]['score'] < 0.2:
                        continue
                    annot_temp['bbox'] = annot_bbox[k][l]['bbox']
                    annot_temp['score'] = annot_bbox[k][l]['score']
                    annot_temp['gray_pth']        = os.path.join(test_obj_dir, 'gray', '{:06d}.tif'.format(int(k)))
                    annot_temp['scene_id'] = int(scene)
                    annot_temp['image_id'] = int(k)                           
                    annot.append(annot_temp)
        '''   
        # validation set    
        for scene in ref.itodd_val_scenes:
            test_obj_dir = os.path.join(self.root_dir, 'val', scene)
            f_bbox = os.path.join(ref.bbox_dir, 'itodd_val', 'itodd_{:06d}_val_result.json'.format(int(scene)))
            with open(f_bbox, 'r') as f:
                annot_bbox = json.load(f)
            # merge annots
            for k in annot_bbox.keys():  
                for l in range(len(annot_bbox[k])):
                    annot_temp = {}
                    annot_temp['obj_id']         = annot_bbox[k][l]['obj_id']
                    if ref.itodd_id2obj[annot_temp['obj_id']] not in [self.cfg.pytorch.object]:
                        continue
                    if annot_bbox[k][l]['score'] < 0.2:
                        continue
                    annot_temp['bbox'] = annot_bbox[k][l]['bbox']
                    annot_temp['score'] = annot_bbox[k][l]['score']
                    annot_temp['gray_pth']        = os.path.join(test_obj_dir, 'gray', '{:06d}.tif'.format(int(k)))
                    annot_temp['scene_id'] = int(scene)
                    annot_temp['image_id'] = int(k)                           
                    annot.append(annot_temp)
        '''
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
        box = self.annot[index]['bbox'].copy()  # bbox format: [left, upper, width, height]
        c = np.array([box[1] + box[3] / 2., box[0] + box[2] / 2.])
        s = max(box[3], box[2]) * 1.5 
        s = min(s, max(self.cfg.pytorch.height, self.cfg.pytorch.width)) * 1.0
        rgb_crop = Crop_by_Pad(rgb, c, s, self.cfg.dataiter.backbone_input_res, channel=3, interpolation=cv2.INTER_LINEAR)
        return cls_idx, rgb_crop, box, c, s

    def __getitem__(self, index):
        cls_idx, rgb_crop, box, c, s = self.GetPartInfo(index)       
        inp = rgb_crop[:, :, 0][None, :, :].astype(np.float32) / 255.
        pose = np.zeros((3, 4))
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


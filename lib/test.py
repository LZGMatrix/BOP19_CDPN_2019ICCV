import torch
import os
import sys
import cv2
import time
import ref
import csv 
import numpy as np
from progress.bar import Bar

def test(cfg, dataLoader, model, models_info=None, models_vtx=None):
    model.eval()
    if cfg.pytorch.exp_mode == 'val':
        from eval import Evaluation
        Eval = Evaluation(cfg.pytorch, models_info, models_vtx)  
        csv_file = open(cfg.pytorch.save_csv_path, 'w')
        fieldnames = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        rst_collect = []   
    elif cfg.pytorch.exp_mode == 'test':
        csv_file = open(cfg.pytorch.save_csv_path, 'w')
        fieldnames = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        rst_collect = []

    preds = {}
    nIters = len(dataLoader)
    bar = Bar('{}_{}'.format(cfg.pytorch.dataset, cfg.pytorch.object), max=nIters)
    wall_time = 0
    for i, (input, pose, bbox, center, size, clsIdx, imgPath, scene_id, image_id, score, K) in enumerate(dataLoader):
        input_var = input.cuda(cfg.pytorch.gpu, async=True).float().cuda(cfg.pytorch.gpu)
        batch_size = len(input)
        # time begin
        T_begin = time.time()
        output_conf, output_coor_x, output_coor_y, output_coor_z = model(input_var)
        output_coor_x = output_coor_x.data.cpu().numpy().copy()
        output_coor_y = output_coor_y.data.cpu().numpy().copy()
        output_coor_z = output_coor_z.data.cpu().numpy().copy()
        outConf = output_conf.data.cpu().numpy().copy()
        output_trans = np.zeros(batch_size)
        collector = list(zip(clsIdx.numpy(), output_coor_x, output_coor_y, output_coor_z, outConf,
                                pose.numpy(), bbox.numpy(), center.numpy(), size.numpy(), input.numpy(), scene_id.numpy(), image_id.numpy(), score.numpy(), K))                      
        colLen = len(collector)
        for idx in range(colLen):
            clsIdx_, output_coor_x_, output_coor_y_, output_coor_z_, output_conf_, pose_gt, bbox_, center_, size_, input_, scene_id_, image_id_, score_, K_ = collector[idx]                
            if cfg.pytorch.dataset.lower()  == 'lmo':
                cls = ref.lmo_id2obj[int(clsIdx_)]
            elif cfg.pytorch.dataset.lower() == 'tless':
                cls = ref.tless_id2obj[int(clsIdx_)]
            elif cfg.pytorch.dataset.lower() == 'ycbv':
                cls = ref.ycbv_id2obj[int(clsIdx_)]
            elif cfg.pytorch.dataset.lower() == 'tudl':
                cls = ref.tudl_id2obj[int(clsIdx_)]
            elif cfg.pytorch.dataset.lower() == 'hb':
                cls = ref.hb_id2obj[int(clsIdx_)]
            elif cfg.pytorch.dataset.lower() == 'icbin':
                cls = ref.icbin_id2obj[clsIdx_]
            elif cfg.pytorch.dataset.lower() == 'itodd':
                cls = ref.itodd_id2obj[int(clsIdx_)]

            select_pts_2d = []
            select_pts_3d = []
            center_h = center_[0]
            center_w = center_[1]
            size_ = int(size_)
            output_coor_x_ = output_coor_x_.squeeze()
            output_coor_y_ = output_coor_y_.squeeze()
            output_coor_z_ = output_coor_z_.squeeze()
            output_coor_ = np.stack([np.argmax(output_coor_x_, axis=0),
                                     np.argmax(output_coor_y_, axis=0),
                                     np.argmax(output_coor_z_, axis=0)], axis=2)            
            output_coor_[output_coor_ == cfg.network.coor_bin] = 0
            output_coor_ = 2.0 * output_coor_ / float(cfg.network.coor_bin - 1) - 1.0
            output_coor_[:, :, 0] = output_coor_[:, :, 0] * abs(models_info[clsIdx_]['min_x'])
            output_coor_[:, :, 1] = output_coor_[:, :, 1] * abs(models_info[clsIdx_]['min_y'])
            output_coor_[:, :, 2] = output_coor_[:, :, 2] * abs(models_info[clsIdx_]['min_z'])        
            output_conf_ = np.argmax(output_conf_, axis=0)
            output_conf_ = (output_conf_ - output_conf_.min()) / (output_conf_.max() - output_conf_.min())        
            min_x = 0.001 * abs(models_info[clsIdx_]['min_x'])
            min_y = 0.001 * abs(models_info[clsIdx_]['min_y'])
            min_z = 0.001 * abs(models_info[clsIdx_]['min_z'])
            w_begin = center_w - size_ / 2.
            h_begin = center_h - size_ / 2.
            w_unit = size_ * 1.0 / cfg.dataiter.rot_output_res
            h_unit = size_ * 1.0 / cfg.dataiter.rot_output_res
            output_conf_ = output_conf_.tolist()
            output_coor_ = output_coor_.tolist()
            for x in range(cfg.dataiter.rot_output_res):
                for y in range(cfg.dataiter.rot_output_res):
                    if output_conf_[x][y] < cfg.test.mask_threshold:
                        continue
                    if abs(output_coor_[x][y][0]) < min_x  and abs(output_coor_[x][y][1]) < min_y  and \
                        abs(output_coor_[x][y][2]) < min_z:
                        continue
                    select_pts_2d.append([w_begin + y * w_unit, h_begin + x * h_unit])
                    select_pts_3d.append(output_coor_[x][y])
            model_points = np.asarray(select_pts_3d, dtype=np.float32)
            image_points = np.asarray(select_pts_2d, dtype=np.float32)
            try:
                if cfg.pytorch.dataset.lower() == 'tless' or cfg.pytorch.dataset.lower() == 'itodd': # camera_matrix vary with images in TLESS & ITODD
                    K_ = K_.numpy().reshape(3, 3)
                    _, R_vector, T_vector, inliers = cv2.solvePnPRansac(model_points, image_points,
                                        K_, np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)
                else:
                    _, R_vector, T_vector, inliers = cv2.solvePnPRansac(model_points, image_points,
                                        cfg.pytorch.camera_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)
                cur_wall_time = time.time() - T_begin
                wall_time += cur_wall_time
                R_matrix = cv2.Rodrigues(R_vector, jacobian=0)[0]       
                if R_matrix[0,0] == 1.0: 
                    continue         
                if cfg.pytorch.exp_mode == 'val':       
                    pose_est = np.concatenate((R_matrix, np.asarray(T_vector).reshape(3, 1)), axis=1)         
                    Eval.pose_est_all[cls].append(pose_est)
                    Eval.pose_gt_all[cls].append(pose_gt)
                    Eval.num[cls] += 1
                    Eval.numAll += 1
                    rst = {'scene_id': int(scene_id_), 'im_id': int(image_id_), 'R': R_matrix.reshape(-1).tolist(), 't': T_vector.reshape(-1).tolist(),
                           'score': float(score_), 'obj_id': int(clsIdx), 'time': cur_wall_time}
                    rst_collect.append(rst)
                elif cfg.pytorch.exp_mode == 'test': 
                    rst = {'scene_id': int(scene_id_), 'im_id': int(image_id_), 'R': R_matrix.reshape(-1).tolist(), 't': T_vector.reshape(-1).tolist(),
                           'score': float(score_), 'obj_id': int(clsIdx), 'time': cur_wall_time}
                    rst_collect.append(rst)
            except:
                if cfg.pytorch.exp_mode == 'val':
                    Eval.num[cls] += 1
                    Eval.numAll += 1                
        Bar.suffix = '{0} [{1}/{2}]| Total: {total:} | ETA: {eta:}'.format(cfg.pytorch.exp_mode, i, nIters, total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
    if cfg.pytorch.exp_mode == 'val':
        Eval.evaluate_pose()
        for item in rst_collect:
            csv_writer.writerow(item)
        csv_file.close()
    elif cfg.pytorch.exp_mode == 'test':
        for item in rst_collect:
            csv_writer.writerow(item)
        csv_file.close()
    print("Wall time of object {}: total {} seconds for {} samples".format(cfg.pytorch.object, wall_time, nIters))
    bar.finish()
    
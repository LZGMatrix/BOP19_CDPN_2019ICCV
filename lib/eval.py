from __future__ import print_function, division, absolute_import

import math
import os, sys
from six.moves import cPickle
from scipy import spatial
import numpy as np
import ref
from scipy.linalg import logm
import numpy.linalg as LA
import matplotlib.pyplot as plt

class Evaluation(object):
    def __init__(self, cfg, models_info, models):
        self.models_info = models_info
        self.models = models
        self.pose_est_all = {}
        self.pose_gt_all = {}
        self.num = {}
        self.numAll = 0.
        self.classes = [cfg.object]
        self.camera_matrix = cfg.camera_matrix
        for cls in self.classes:
            self.num[cls] = 0.
            self.pose_est_all[cls] = []
            self.pose_gt_all[cls] = []

    def evaluate_pose(self):
        """
        Evaluate 6D pose and display
        """        
        all_poses_est = self.pose_est_all
        all_poses_gt = self.pose_gt_all
        print('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Evaluation 6D Pose', '-' * 100))
        rot_thresh_list = np.arange(1, 11, 1)
        # trans_thresh_list = np.arange(0.01, 0.11, 0.01)
        trans_thresh_list = np.arange(10, 110, 10)
        num_metric = len(rot_thresh_list)
        num_classes = len(self.classes)
        rot_acc = np.zeros((num_classes, num_metric))
        trans_acc = np.zeros((num_classes, num_metric))
        space_acc = np.zeros((num_classes, num_metric))

        num_valid_class = len(self.classes)
        for i, cls_name in enumerate(self.classes):
            curr_poses_gt = all_poses_gt[cls_name]
            curr_poses_est = all_poses_est[cls_name]
            num = len(curr_poses_gt)
            cur_rot_rst = np.zeros((num, 1))
            cur_trans_rst = np.zeros((num, 1))

            for j in range(num):
                r_dist_est, t_dist_est = calc_rt_dist_m(curr_poses_est[j], curr_poses_gt[j])
                if cls_name == 'eggbox' and r_dist_est > 90:
                    RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                    curr_pose_est_sym = se3_mul(curr_poses_est[j], RT_z)
                    r_dist_est, t_dist_est = calc_rt_dist_m(curr_pose_est_sym, curr_poses_gt[j])
                # print('t_dist: {}'.format(t_dist_est))
                cur_rot_rst[j, 0] = r_dist_est
                cur_trans_rst[j, 0] = t_dist_est

            # cur_rot_rst = np.vstack(all_rot_err[cls_idx, iter_i])
            # cur_trans_rst = np.vstack(all_trans_err[cls_idx, iter_i])
            for thresh_idx in range(num_metric):
                rot_acc[i, thresh_idx] = np.mean(cur_rot_rst < rot_thresh_list[thresh_idx])
                trans_acc[i, thresh_idx] = np.mean(cur_trans_rst < trans_thresh_list[thresh_idx])
                space_acc[i, thresh_idx] = np.mean(np.logical_and(cur_rot_rst < rot_thresh_list[thresh_idx],
                                                                  cur_trans_rst < trans_thresh_list[thresh_idx]))

            print("------------ {} -----------".format(cls_name))
            print("{:>24}: {:>7}, {:>7}, {:>7}".format("[rot_thresh, trans_thresh", "RotAcc", "TraAcc", "SpcAcc"))
            print(
                "{:<16}{:>8}: {:>7.2f}, {:>7.2f}, {:>7.2f}".format('average_accuracy', '[{:>2}, {:.2f}]'.format(-1, -1),
                                                                   np.mean(rot_acc[i, :]) * 100,
                                                                   np.mean(trans_acc[i, :]) * 100,
                                                                   np.mean(space_acc[i, :]) * 100))
            show_list = [1, 4, 9]
            for show_idx in show_list:
                print("{:>16}{:>8}: {:>7.2f}, {:>7.2f}, {:>7.2f}"
                            .format('average_accuracy',
                                    '[{:>2}, {:.2f}]'.format(rot_thresh_list[show_idx], trans_thresh_list[show_idx]),
                                    rot_acc[i, show_idx] * 100, trans_acc[i, show_idx] * 100,
                                    space_acc[i, show_idx] * 100))        
        print(' ')

sqrt2 = 2.0 ** 0.5
pi = np.arccos(-1)


def se3_mul(RT1, RT2):
    """
    concat 2 RT transform
    :param RT1=[R,T], 4x3 np array
    :param RT2=[R,T], 4x3 np array
    :return: RT_new = RT1 * RT2
    """
    R1 = RT1[0:3, 0:3]
    T1 = RT1[0:3, 3].reshape((3, 1))

    R2 = RT2[0:3, 0:3]
    T2 = RT2[0:3, 3].reshape((3, 1))

    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = np.dot(R1, R2)
    T_new = np.dot(R1, T2) + T1
    RT_new[0:3, 3] = T_new.reshape((3))
    return RT_new


def re(R_est, R_gt):
    assert (R_est.shape == R_gt.shape == (3, 3))
    temp = logm(np.dot(np.transpose(R_est), R_gt))
    rd_rad = LA.norm(temp, 'fro') / np.sqrt(2)
    rd_deg = rd_rad / np.pi * 180
    return rd_deg


def te(t_est, t_gt):
    """
    Translational Error.

    :param t_est: Translation element of the estimated pose (3x1 vector).
    :param t_gt: Translation element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert (t_est.size == t_gt.size == 3)
    error = np.linalg.norm(t_gt.reshape(3) - t_est.reshape(3))
    return error


def calc_rt_dist_m(pose_src, pose_tgt):
    R_src = pose_src[:, :3]
    T_src = pose_src[:, 3]
    R_tgt = pose_tgt[:, :3]
    T_tgt = pose_tgt[:, 3]
    temp = logm(np.dot(np.transpose(R_src), R_tgt))
    rd_rad = LA.norm(temp, 'fro') / np.sqrt(2)
    rd_deg = rd_rad / np.pi * 180

    td = LA.norm(T_tgt - T_src)

    return rd_deg, td



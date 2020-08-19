# encoding: utf-8
"""
This file includes necessary params, info
"""
import os.path as osp

import numpy as np

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
output_dir = osp.join(root_dir, "output")  # directory storing experiment data (result, model checkpoints, etc).

data_root = osp.join(root_dir, "datasets")
bop_root = osp.join(data_root, "BOP_DATASETS/")

# ---------------------------------------------------------------- #
# TLESS DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(bop_root, "tless")
train_real_dir = osp.join(dataset_root, "train_primesense")
train_render_dir = osp.join(dataset_root, "train_render_reconst")
test_dir = osp.join(dataset_root, "test_primesense")

model_dir = osp.join(dataset_root, "models_cad")  # use cad models as default
model_reconst_dir = osp.join(dataset_root, "models_reconst")
model_eval_dir = osp.join(dataset_root, "models_eval")
vertex_scale = 0.001
# object info
objects = [str(i) for i in range(1, 31)]
id2obj = {i: str(i) for i in range(1, 31)}

obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]
texture_paths = None
model_colors = [((i + 1) * 5, (i + 1) * 5, (i + 1) * 5) for i in range(obj_num)]  # for renderer

# Camera info
tr_real_width = 400
tr_real_height = 400
tr_render_width = 1280
tr_render_height = 1024
te_width = 720
te_height = 540
tr_real_center = (tr_real_height / 2, tr_real_width / 2)
tr_render_center = (tr_render_height / 2, tr_render_width / 2)
te_center = (te_width / 2.0, te_height / 2.0)
camera_matrix = np.array([[1075.65091572, 0.0, 374.06888344], [0.0, 1073.90347929, 255.72159802], [0, 0, 1]])

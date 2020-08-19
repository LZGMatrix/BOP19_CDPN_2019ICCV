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
# ITODD (MVTec ITODD) DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(bop_root, "itodd")
test_dir = osp.join(dataset_root, "test")

model_dir = osp.join(dataset_root, "models")
model_eval_dir = osp.join(dataset_root, "models_eval")
vertex_scale = 0.001
# object info
objects = [str(i) for i in range(1, 28+1)]
id2obj = {i: str(i) for i in range(1, 28+1)}

obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]
texture_paths = None
model_colors = [((i + 1) * 5, (i + 1) * 5, (i + 1) * 5) for i in range(obj_num)]  # for renderer

# Camera info
width = 1280
height = 960
camera_matrix = np.array([[2992.63, 0.0, 633.886], [0.0, 3003.99, 489.554], [0, 0, 1]])

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
# TUDL (TUD Light) DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(bop_root, "tudl")
train_real_dir = osp.join(dataset_root, "train_real")
train_render_dir = osp.join(dataset_root, "train_render")
test_dir = osp.join(dataset_root, "test")

model_dir = osp.join(dataset_root, "models")
model_eval_dir = osp.join(dataset_root, "models_eval")
vertex_scale = 0.001
# object info
objects = ["dragon", "frog", "watering_pot"]
id2obj = {1: "dragon", 2: "frog", 3: "watering_pot"}

obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]
texture_paths = None
model_colors = [((i + 1) * 5, (i + 1) * 5, (i + 1) * 5) for i in range(obj_num)]  # for renderer

# Camera info
width = 640
height = 480
camera_matrix = np.array([[515.0, 0.0, 321.5660095214844], [0.0, 515.0, 214.08000230789185], [0, 0, 1]])

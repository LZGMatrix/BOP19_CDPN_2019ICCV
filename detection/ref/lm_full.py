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
# LINEMOD DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(bop_root, "lm")
train_dir = osp.join(dataset_root, "train")
test_dir = osp.join(dataset_root, "test")
model_dir = osp.join(dataset_root, "models")
vertex_scale = 0.001
model_eval_dir = osp.join(dataset_root, "models_eval")
# scaled models (.obj)
model_scaled_dir = osp.join(dataset_root, "models_scaled")
model_scaled_simple_dir = osp.join(dataset_root, "models_scaled_f5k")

train_synt_blender_dir = osp.join(dataset_root, "lm_blender/")

# object info
objects = [
    "ape", "benchvise", "bowl", "camera", "can", \
    "cat", "cup", "driller", "duck", "eggbox", \
    "glue", "holepuncher", "iron", "lamp", "phone"
]
id2obj = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}
obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]
texture_paths = None
model_colors = [((i + 1) * 10, (i + 1) * 10, (i + 1) * 10) for i in range(obj_num)]  # for renderer

diameters = np.array([
    102.099, 247.506, 167.355, 172.492, 201.404, \
    154.546, 124.264, 261.472, 108.999, 164.628, \
    175.889, 145.543, 278.078, 282.601, 212.358, \
]) / 1000.0

# Camera info
width = 640
height = 480
zNear = 0.25
zFar = 6.0
center = (height / 2, width / 2)
camera_matrix = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
import numpy as np
import json
import cv2
import os
import math
from glob import glob
from scipy import spatial
from tqdm import tqdm
import csv
import json
import re

# path
dataset = 'icbin' # 'lmo', 'ycbv', 'tudl', 'icbin', 'tless', 'hb', 'itodd'
root_dir = '/media/yili/data3/lzg/6D_Challenge/CDPN_BOP19_v1/'
exp_dir = os.path.join(root_dir, 'exp')
result_dir = os.path.join(exp_dir, 'final_result')
files = glob(os.path.join(exp_dir, '{}_*'.format(dataset.upper()), '*'))
print(files)

if not os.path.exists(os.path.join(result_dir)):
    os.makedirs(os.path.join(result_dir))

# detection time
time_total = {}
detection_paths = glob(os.path.join(root_dir, 'bbox', 'itodd_val', '*'))
detection_paths = glob(os.path.join(root_dir, 'bbox', dataset, '*'))
# detection_paths = glob(os.path.join(root_dir, 'bbox', 'itodd_val', '*')) # hb/itodd validation set
for detection_path in detection_paths:
    with open(detection_path, 'r') as f:
        detection_file = json.load(f)
    scene_id = int(detection_path.split('/')[-1].split('_')[1])
    for im_id in detection_file.keys():
        im_id = int(im_id)
        if scene_id not in time_total.keys():
            time_total[scene_id] = {}
        if im_id not in time_total[scene_id].keys():
            time_total[scene_id][im_id] = 0.0
        time_total[scene_id][im_id] = float(detection_file[str(im_id)][0]['time'])

    f.close()
print(time_total)

# calculate time per image
data = []
for file in files:
    csv_file = open(file,'r')
    lines = csv.reader(csv_file)
    for line in lines:
        data.append(line)
for i in range(0, len(data)):
    if data[i][0] == 'scene_id':
        continue
    scene_id = int(data[i][0])
    im_id = int(data[i][1])
    time_pose = float(data[i][6])

    time_total[scene_id][im_id] += float(time_pose)
print(time_total)

# save csv
f = open(os.path.join(result_dir, 'CDPN_{}-test.csv'.format(dataset)), "w")
f.write("scene_id,im_id,obj_id,score,R,t,time\n")
for i in range(0, len(data)):
    if data[i][0] == 'scene_id':
        continue
    scene_id = int(data[i][0])
    im_id = int(data[i][1])
    obj_id = int(data[i][2])
    score = float(data[i][3])
    R_est = str(data[i][4]).replace('[', '').replace(']', '').replace(',', '').replace('\n', '')
    R_est = re.sub(' +',' ',R_est)
    T_est = str(data[i][5]).replace('[', '').replace(']', '').replace(',', '')
    T_est = re.sub(' +',' ',T_est)
    time = time_total[scene_id][im_id]
    f.write("{},{},{},{},{},{},{}\n".format(scene_id, im_id, obj_id, score, R_est, T_est, time))    
    
f.close()

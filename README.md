# BOP19_CDPN_2019ICCV

The modified version of CDPN ("CDPN: Coordinates-Based Disentangled Pose Network for Real-Time RGB-Based 6-DoF Object Pose Estimation" on ICCV 2019) for BOP: Benchmark for 6D Object Pose Estimation Challenge 2019.

*Note: We provide the test code of our approach in this repo. The trained CDPN models will be provided after the submission deadline.*

## Our test environments
- Ubuntu 16.04 (64bit)
- Python 3.6.7
- Pytorch 0.4.1
- CUDA 9.0
- Bop_toolkit (https://github.com/thodan/bop_toolkit)
- numpy, cv2, json, plyfile, tqdm, scipy, progress, etc.

## Detection
For detection, we trained a RetinaNet for each dataset on mmdetection (https://github.com/open-mmlab/mmdetection).
* Please refer to [`DETECTION.md`](detection/DETECTION.md)

## Pose Estimation
In the BOP 2019 challenge, different from the paper, both of the rotation and translation are solved from the built 2D-3D correspondences by PnP algorithm. We trained a CDPN model for each object.

## Data Preparation
1. Download the 7 core datasets from the BOP website (https://bop.felk.cvut.cz/datasets/)
2. Download our trained models (https:TODO) and detection results (https:TODO).
3. Prepare the data as follows:
```
  ├── tudl_bop19
  │   ├── models_eval (official models for evaluation)
  │   │   ├── models_info.json
  │   │   ├── obj_000001.ply
  │   │   ├── ...
  │   │   └── ...
  │   ├── test (official test data)
  │   │   ├── 000001
  │   │   └── ...
  │   ├── lmo_test_bop19_retinanet (our provided detection results)
  │   │   ├── lmo_test_bop19_000002.json
  │   │   └── ... 
  ├── other datasets
  │
  └── trained_models
      ├── tudl
      │   ├── obj_can.checkpoint
      │   └── ...
      └── ...
```
## Run
1. Modify the dataset path in ref.py
2. In 'tools' directory, run 
```
  sh run.sh
```

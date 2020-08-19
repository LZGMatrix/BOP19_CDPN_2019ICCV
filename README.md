# BOP19_CDPN_2019ICCV

The modified version of CDPN ("CDPN: Coordinates-Based Disentangled Pose Network for Real-Time RGB-Based 6-DoF Object Pose Estimation" on ICCV 2019) for BOP: Benchmark for 6D Object Pose Estimation Challenge 2019.

>>>>

**Find CDPNv2 for BOP20 in the branch of [bop2020](https://github.com/LZGMatrix/BOP19_CDPN_2019ICCV/tree/bop2020)**

**Find CDPN trained with PBR data for BOP20 in the branch of [bop2019_pbr](https://github.com/LZGMatrix/BOP19_CDPN_2019ICCV/tree/bop2019_pbr)**

<<<<

## Our test environments
- Ubuntu 16.04 (64bit)
- Python 3.6.7
- Pytorch 0.4.1
- CUDA 9.0
- [Bop_toolkit](https://github.com/thodan/bop_toolkit)
- numpy, cv2, plyfile, tqdm, scipy, progress, etc.

## Detection
For detection, we trained a RetinaNet for each dataset on [mmdetection](https://github.com/open-mmlab/mmdetection).
* Please refer to [`DETECTION.md`](detection/DETECTION.md)

## Pose Estimation
In the BOP 2019 challenge, different from the paper, both of the rotation and translation are solved from the built 2D-3D correspondences by PnP algorithm. We trained a CDPN model for each object.

## Data Preparation
1. Download the 7 core datasets from the [BOP website](https://bop.felk.cvut.cz/datasets/)
2. Download our [trained models](https://drive.google.com/drive/folders/1GoCSOVZk0kzxS5e--oVXS83wpRHd9qJO?usp=sharing) and [detection results](https://drive.google.com/drive/folders/1nTP87zzF9l7VO3UEjcEbX61J3-6wRbuf?usp=sharing).
3. Prepare the data as follows:

    Note: 
    - models_eval: downloaded official models; 
    - test/test_primesense: downloaded official BOP19 test set; 
    - val:optionally, downloaded official val set;
    - trained_models: our provided trained models;
    - bbox_retinanet: our provided detection results;
    - exp: save the test result files
```
Root
├── dataset
│   ├── lmo_bop19
│   │   ├── models_eval 
│   │   └── test 
│   ├── tudl_bop19
│   │   ├── models_eval 
│   │   └── test 
│   ├── hb_bop19
│   │   ├── models_eval
│   │   ├── val 
│   │   └── test
│   ├── icbin_bop19
│   │   ├── models_eval
│   │   └── test 
│   ├── itodd_bop19
│   │   ├── models_eval 
│   │   ├── val
│   │   └── test
│   ├── tless_bop19
│   │   ├── models_eval
│   │   └── test_primesense 
│   └── ycbv_bop19
│       ├── models_eval 
│       └── test
├── trained_models
│   ├── lmo
│   │   ├── obj_ape.checkpoint
│   │   └── ...
│   └── ...
├── bbox_retinanet
│   ├── lmo
│   │   ├── lmo_test_bop19_000002.json
│   │   └── ... 
│   └── ...
├── lib
├── tools
├── detection
└── exp
```
## Run
1. In 'tools' directory, run 
```
  sh run.sh
```
It will first generate a .csv file to record the result of each object for each dataset. The final result files can be found in 'exp/final_result/CDPN_xxxx-test.csv'

2. Use the Bop_toolkit for evaluation.

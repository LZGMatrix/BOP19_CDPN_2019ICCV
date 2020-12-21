## News
- We have released the [official training code](https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi) of CDPN to reproduce the impressive results in our ICCV2019 paper.
- Find CDPN for BOP19 in the [(https://github.com/LZGMatrix/BOP19_CDPN_2019ICCV)](https://github.com/LZGMatrix/BOP19_CDPN_2019ICCV/tree/bop2020)
- Find CDPN trained with PBR data for BOP20 [(https://github.com/LZGMatrix/BOP19_CDPN_2019ICCV/tree/bop2019_pbr)](https://github.com/LZGMatrix/BOP19_CDPN_2019ICCV/tree/bop2019_pbr)

# CDPNv2 for BOP20

CDPNv2 for BOP20 challenge (mainly based on "CDPN: Coordinates-Based Disentangled Pose Network for Real-Time RGB-Based 6-DoF Object Pose Estimation" on ICCV 2019).

### **The trained CDPNv2 weights for BOP20 Challenge have been provided!**

## Our test environments (the same as CDPN in bop19)
- Ubuntu 16.04 (64bit)
- Python 3.6.7
- Pytorch 0.4.1
- CUDA 9.0
- [Bop_toolkit](https://github.com/thodan/bop_toolkit)
- numpy, cv2, plyfile, tqdm, scipy, progress, etc.

## Detection
* Please refer to [`detection/README.md`](detection/README.md)

## Pose Estimation

The difference between our CDPNv2 and the BOP19-version CDPN mainly including:

- Domain Randomization: We used stronger domain randomization operations than BOP19. The details will be provided after the deadline.

- Network Architecture: Considering the organizer provides high-quality PBR synthetic training data in BOP20, we adopt a deeper 34-layer Resnet as the backbone instead of the 18-layer Resnet used in BOP19-version CDPN. Also, the fancy concat structures in BOP19-version CDPN are removed. The input and output resolutions are 256×256 and 64×64 respectively.

- Training During training, the initial learning rate was 1 × 10−4 and the batch size was 6. We used RMSProp with alpha 0.99 and epsilon 1× 10−8 to optimize the network. The model was trained for 160 epochs in total and the learning rate was divided by 10 every 50 epochs

- Other implementations, such as the coordinates labels were computed by back-projection from the rendered depth, instead of forward-projection with z-buffer.

## Data Preparation
1. Download the 7 core datasets from the [BOP website](https://bop.felk.cvut.cz/datasets/)
2. Download our [trained models](https://pan.baidu.com/s/1nB9c8tFQedwiRXHrxblGhg)(code: 3jfn) and [detection results]().
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
├── bbox
│   ├── lmo
│   │   ├── lmo_test_bop19_000002.json
│   │   └── ... 
│   └── ...
├── lib
├── tools
└── exp
```
## Run
1. In 'tools' directory, run 
```
  sh run.sh
```
It will first generate a .csv file to record the result of each object for each dataset. The final result files can be found in 'exp/final_result/CDPN_xxxx-test.csv'

2. Use the Bop_toolkit for evaluation.

## Installation
install dependencies:
```
# install torch, torchvision
# install detectron2

# build this adet
python setup.py develop

# install other dependencies
sh scripts/install_deps.sh
```

## Prepare datasets
```
mkdir -p datasets
cd datasets
ln -sf /path/to/BOP_DATASETS
```
The folder structure should like this:
```
├── lmo
│   ├── test
│   │   ├── <scene_id>
│   │   │   ├── rgb
│   │   │   │   ├── <img_id>.png
│   │   │   │   └── ...
│   │   │   ├── mask_visib
│   │   │   ├── depth
│   │   │   └── ...
├── ycbv
│   ├── test
│   │   └── ...
└── ...
```

## Trained models
Download our trained models from [here](https://drive.google.com/drive/folders/1BuS7CTccc9QfMW040Na10IehWCx8--DV?usp=sharing).

## Test
Get the detection results:
```
./tools/test.sh <CFG_PATH> <GPU_IDS> <MODEL_PATH>
```

For example:
```
./tools/test.sh configs/BOP-Detection/fcos_V57eSE_MSx1333_ColorAugAAEWeaker_8e_lmo_pbr.yaml 0 output/bop_det/lmo/fcos_V57eSE_MSx1333_ColorAugAAEWeaker_8e_lmo_pbr/model_final_no_optim.pth
```
Then convert the results to json format:
```
python tools/convert_results_to_bop.py --dataset lmo_bop_test --path output/bop_det/lmo/fcos_V57eSE_MSx1333_ColorAugAAEWeaker_8e_lmo_pbr/inference_model_final_no_optim/lmo_bop_test/instances_predictions.pth
```

The result json file is similar to bop format `scene_gt_info.json`.

Differences:
* The key is `"scene_id/im_id"` instead of `"im_id"`.
* Estimated bbox field name: `bbox_est`.
* Added new fields `score`, `time`. `time` is the compute time for the image.
* There are up to 100 instance predictions per image.

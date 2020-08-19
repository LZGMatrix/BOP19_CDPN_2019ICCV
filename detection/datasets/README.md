```
# assume in $PROJ_ROOT/datasets/
ln -sf /path/to/BOP_DATASETS
```
and the folder structure should be like this:
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

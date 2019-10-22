import numpy as np
from pycocotools.coco import COCO

from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class HBDataset(CocoDataset):
    clses = [1,10,12,15,17,18,19,22,23,29,3,32,33,4,8,9]
    clses.sort()
    CLASSES = tuple([str(i) for i in clses])

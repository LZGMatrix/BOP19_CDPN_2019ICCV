import numpy as np
from pycocotools.coco import COCO

from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class YcbvDataset(CocoDataset):
    CLASSES = tuple([str(i) for i in range(1,22)])

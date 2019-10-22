import numpy as np
from pycocotools.coco import COCO

from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class LinemodOccludedDataset(CocoDataset):
    lmo_objects = {1:'ape', 5:'can', 6:'cat',
                       8:'driller', 9:'duck', 10:'eggbox', 
                       11:'glue', 12:'holepuncher'} 
    CLASSES = tuple(['%d_%s'%(obj_id,cls_name) for obj_id,cls_name in lmo_objects.items()])

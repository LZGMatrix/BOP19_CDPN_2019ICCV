import copy
import logging
import numpy as np
import operator
import pickle
import random
import torch
import torch.utils.data as data

from detectron2.utils.serialize import PicklableWrapper
from detectron2.data.build import trivial_batch_collator, worker_init_reset_seed, get_detection_dataset_dicts
from detectron2.data.common import AspectRatioGroupedDataset, DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.utils.comm import get_world_size
from detectron2.data.samplers import InferenceSampler, RepeatFactorTrainingSampler, TrainingSampler


def flat_dataset_dicts(dataset_dicts):
    """TODO: test this
    flatten the dataset dicts of detectron2 format
    original: list of dicts, each dict contains some image-level infos
              and an "annotations" field for instance-level infos of multiple instances
    => flat the instance level annotations
    flat format:
        list of dicts,
            each dict includes the image/instance-level infos
            a `inst_id` of a single instance,
            `annotations` includes only one instance
    """
    new_dicts = []
    for dataset_dict in dataset_dicts:
        img_infos = {_k: _v for _k, _v in dataset_dict.items() if _k not in ["annotations"]}
        if "annotations" in dataset_dict:
            for inst_id, anno in enumerate(dataset_dict["annotations"]):
                rec = {"inst_id": inst_id, "inst_infos": anno}
                rec.update(img_infos)
                new_dicts.append(rec)
        else:
            rec = img_infos
            new_dicts.append(rec)
    return new_dicts

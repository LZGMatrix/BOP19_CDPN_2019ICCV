import logging
import hashlib
import os
import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../.."))
sys.path.insert(0, PROJ_ROOT)
import time
from collections import OrderedDict

import mmcv
import numpy as np
from tqdm import tqdm
from transforms3d.quaternions import mat2quat, quat2mat

import ref
from core.utils.utils import egocentric_to_allocentric
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from lib.pysixd import inout, misc
from lib.utils.fs import mkdir_p, recursive_walk
from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask
from lib.utils.utils import dprint, iprint, lazy_property

cur_dir = osp.dirname(osp.abspath(__file__))

DATASETS_ROOT = osp.normpath(osp.join(cur_dir, "../../datasets"))

logger = logging.getLogger(__name__)


class TLESS_TRAIN_FUSE_Dataset(object):

    def __init__(self, data_cfg):
        """
        Set with_masks default to True,
        and decide whether to load them into dataloader/network later
        with_masks:
        """
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"]  # selected objs

        self.dataset_root = data_cfg.get('dataset_root',
            osp.join(DATASETS_ROOT, "BOP_DATASETS/tless/tless_real_fuse_coco"))
        assert osp.exists(self.dataset_root), self.dataset_root
        self.models_root = data_cfg["models_root"]  # BOP_DATASETS/tless/models_cad
        self.scale_to_meter = data_cfg['scale_to_meter']  # 0.001

        self.with_masks = data_cfg['with_masks']
        self.with_depth = data_cfg['with_depth']
        self.with_xyz = data_cfg['with_xyz']

        self.height = data_cfg['height']
        self.width = data_cfg['width']

        self.num_to_load = data_cfg['num_to_load']  # -1
        self.filter_invalid = data_cfg.get('filter_invalid', True)
        self.use_cache = data_cfg.get('use_cache', True)

        # NOTE: careful! Only the selected objects
        self.cat_ids = [cat_id for cat_id, obj_name in ref.tless_bop19.id2obj.items() if obj_name in self.objs]
        # map selected objs to [0, num_objs)
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs))

    def __call__(self):
        """
        Load light-weight instance annotations of all images into a list of dicts in Detectron2 format.
        Do not load heavy data into memory in this file,
        since we will load the annotations of all images into memory.
        """
        # cache the dataset_dicts to avoid loading masks from files
        hashed_file_name = hashlib.md5(
            ("".join([str(fn) for fn in self.objs]) +
             "dataset_dicts_{}_{}_{}_{}_{}_{}".format(self.name, self.dataset_root, self.with_masks, self.with_depth, self.with_xyz,
                                                   osp.abspath(__file__))).encode("utf-8")).hexdigest()
        cache_path = osp.join(self.dataset_root, "dataset_dicts_{}_{}.pkl".format(self.name, hashed_file_name))

        if osp.exists(cache_path) and self.use_cache:
            logger.info("load cached dataset dicts from {}".format(cache_path))
            return mmcv.load(cache_path)

        t_start = time.perf_counter()
        dataset_dicts = []
        self.num_instances_without_valid_segmentation = 0
        self.num_instances_without_valid_box = 0
        logger.info("loading dataset dicts: {}".format(self.name))

        gt_path = osp.join(self.dataset_root, "gt.json")
        assert osp.exists(gt_path), gt_path
        gt_dict = mmcv.load(gt_path)

        if True:
            for str_im_id in tqdm(gt_dict):
                int_im_id = int(str_im_id)
                rgb_path = osp.join(self.dataset_root, "{:06d}.png").format(int_im_id)
                assert osp.exists(rgb_path), rgb_path

                record = {
                    "dataset_name": self.name,
                    'file_name': osp.relpath(rgb_path, PROJ_ROOT),

                    'height': self.height,
                    'width': self.width,
                    'image_id': int_im_id,
                    "scene_im_id": "{}/{}".format(0, int_im_id),  # for evaluation

                    "img_type": 'real'  # NOTE: has background
                }
                insts = []
                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno['obj_id']
                    if obj_id not in self.cat_ids:
                        continue
                    cur_label = self.cat2label[obj_id]  # 0-based label

                    bbox_obj = gt_dict[str_im_id][anno_i]['obj_bb']
                    x1, y1, w, h = bbox_obj
                    if self.filter_invalid:
                        if h <= 1 or w <= 1:
                            self.num_instances_without_valid_box += 1
                            continue

                    inst = {
                        'category_id': cur_label,  # 0-based label
                        'bbox': bbox_obj,
                        'bbox_mode': BoxMode.XYWH_ABS,

                    }

                    insts.append(inst)
                if len(insts) == 0:  # filter im without anno
                    continue
                record['annotations'] = insts
                dataset_dicts.append(record)

        if self.num_instances_without_valid_box > 0:
            logger.warning("Filtered out {} instances without valid box. "
                        "There might be issues in your dataset generation process.".format(
                            self.num_instances_without_valid_box))
        ##########################
        if self.num_to_load > 0:
            self.num_to_load = min(int(self.num_to_load), len(dataset_dicts))
            dataset_dicts = dataset_dicts[:self.num_to_load]
        logger.info("loaded {} dataset dicts, using {}s".format(
            len(dataset_dicts),
            time.perf_counter() - t_start))

        mkdir_p(osp.dirname(cache_path))
        mmcv.dump(dataset_dicts, cache_path, protocol=4)
        logger.info("Dumped dataset_dicts to {}".format(cache_path))
        return dataset_dicts

    def __len__(self):
        return self.num_to_load

    def image_aspect_ratio(self):
        return self.width / self.height  # 4/3


def get_tless_metadata(obj_names):
    # task specific metadata
    # TODO: provide symetry info for models here
    meta = {"thing_classes": obj_names}
    return meta


tless_model_root = "BOP_DATASETS/tless/models_cad/"
################################################################################


SPLITS_TLESS_FUSE = dict(
    tless_train_primesense_fuse_coco=dict(
        name="tless_train_primesense_fuse_coco",
        objs=ref.tless_bop19.objects,  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/tless/tless_real_fuse_coco"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/tless/models_cad"),
        scale_to_meter=0.001,
        with_masks=False,  # (load masks but may not use it)
        with_depth=False,  # (load depth path here, but may not use it)
        with_xyz=False,
        height=540,
        width=720,
        use_cache=True,
        num_to_load=-1,
        filter_invalid=False,
        ref_key="tless_bop19",
    ),
)

# single obj splits
for obj in ref.tless_bop19.objects:
    for split in ["train"]:
        name = "tless_primesense_fuse_{}_{}".format(obj, split)
        if split in ["train"]:
            filter_invalid = True
        elif split in ["test"]:
            filter_invalid = False
        else:
            raise ValueError("{}".format(split))
        if name not in SPLITS_TLESS_FUSE:
            SPLITS_TLESS_FUSE[name] = dict(
                name=name,
                objs=[obj],  # only this obj
                dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/tless/tless_real_fuse_coco"),
                models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/tless/models_cad"),
                scale_to_meter=0.001,
                with_masks=False,  # (load masks but may not use it)
                with_depth=False,  # (load depth path here, but may not use it)
                with_xyz=False,
                height=540,
                width=720,
                use_cache=True,
                num_to_load=-1,
                filter_invalid=filter_invalid,
                ref_key="tless_bop19")


def register_with_name_cfg(name, data_cfg=None):
    """Assume pre-defined datasets live in `./datasets`.
    Args:
        name: datasnet_name,
        data_cfg: if name is in existing SPLITS, use pre-defined data_cfg
            otherwise requires data_cfg
            data_cfg can be set in cfg.DATA_CFG.name
    """
    dprint("register dataset: {}".format(name))
    if name in SPLITS_TLESS_FUSE:
        used_cfg = SPLITS_TLESS_FUSE[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg
    DatasetCatalog.register(name, TLESS_TRAIN_FUSE_Dataset(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        id="tless",  # NOTE: for pvnet to determine module
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="coco_bop",
        **get_tless_metadata(obj_names=used_cfg["objs"]),
    )


def get_available_datasets():
    return list(SPLITS_TLESS_FUSE.keys())


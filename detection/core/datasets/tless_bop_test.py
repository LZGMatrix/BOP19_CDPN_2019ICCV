import hashlib
import logging
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
from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask
from lib.utils.utils import dprint, iprint, lazy_property

logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(cur_dir, "../../datasets"))


class TLESS_BOP_TEST_Dataset(object):
    """
    tless bop test
    """

    def __init__(self, data_cfg):
        """
        Set with_depth and with_masks default to True,
        and decide whether to load them into dataloader/network later
        with_masks:
        """
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"]  # selected objects
        # all classes are self.objs, but this enables us to evaluate on selected objs
        self.select_objs = data_cfg.get("select_objs", self.objs)

        self.ann_file = data_cfg["ann_file"] # json file with scene_id and im_id items

        self.dataset_root = data_cfg["dataset_root"]  # BOP_DATASETS/tless/test_primesense
        self.models_root = data_cfg["models_root"]  # BOP_DATASETS/tless/models_cad
        self.scale_to_meter = data_cfg["scale_to_meter"]  # 0.001

        self.with_masks = data_cfg["with_masks"]  # True (load masks but may not use it)
        self.with_depth = data_cfg["with_depth"]  # True (load depth path here, but may not use it)

        self.height = data_cfg["height"]
        self.width = data_cfg["width"]

        self.cache_dir = data_cfg["cache_dir"]  # .cache
        self.use_cache = data_cfg["use_cache"]  # True
        self.num_to_load = data_cfg["num_to_load"]  # -1
        self.filter_invalid = data_cfg["filter_invalid"]
        ##################################################

        # NOTE: careful! Only the selected objects
        self.cat_ids = [cat_id for cat_id, obj_name in ref.tless_bop19.id2obj.items() if obj_name in self.objs]
        # map selected objs to [0, num_objs)
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs))
        ##########################################################

    def __call__(self):
        """
        Load light-weight instance annotations of all images into a list of dicts in Detectron2 format.
        Do not load heavy data into memory in this file,
        since we will load the annotations of all images into memory.
        """
        # cache the dataset_dicts to avoid loading masks from files
        hashed_file_name = hashlib.md5(
            ("".join([str(fn) for fn in self.objs]) +
             "dataset_dicts_{}_{}_{}_{}_{}".format(self.name, self.dataset_root, self.with_masks, self.with_depth,
                                                   osp.abspath(__file__))).encode("utf-8")).hexdigest()
        cache_path = osp.join(self.dataset_root, "dataset_dicts_{}_{}.pkl".format(self.name, hashed_file_name))

        if osp.exists(cache_path) and self.use_cache:
            logger.info("load cached dataset dicts from {}".format(cache_path))
            return mmcv.load(cache_path)

        t_start = time.perf_counter()

        logger.info("loading dataset dicts: {}".format(self.name))
        self.num_instances_without_valid_segmentation = 0
        self.num_instances_without_valid_box = 0

        dataset_dicts = []  #######################################################
        im_id_global = 0

        if True:
            targets = mmcv.load(self.ann_file)
            scene_im_ids = [(item["scene_id"], item["im_id"]) for item in targets]
            scene_im_ids = sorted(list(set(scene_im_ids)))

            # load infos for each scene
            gt_dicts = {}
            gt_info_dicts = {}
            cam_dicts = {}
            for scene_id, im_id in scene_im_ids:
                scene_root = osp.join(self.dataset_root, f"{scene_id:06d}")
                if scene_id not in gt_dicts:
                    gt_dicts[scene_id] = mmcv.load(osp.join(scene_root, 'scene_gt.json'))
                if scene_id not in gt_info_dicts:
                    gt_info_dicts[scene_id] = mmcv.load(osp.join(scene_root, 'scene_gt_info.json'))  # bbox_obj, bbox_visib
                if scene_id not in cam_dicts:
                    cam_dicts[scene_id] = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            for scene_id, im_id in tqdm(scene_im_ids):
                str_im_id = str(im_id)
                scene_root = osp.join(self.dataset_root, f"{scene_id:06d}")
                rgb_path = osp.join(scene_root, "rgb/{:06d}.png").format(im_id)
                assert osp.exists(rgb_path), rgb_path

                depth_path = osp.join(scene_root, "depth/{:06d}.png".format(im_id))

                scene_id = int(rgb_path.split('/')[-3])

                cam = np.array(cam_dicts[scene_id][str_im_id]['cam_K'], dtype=np.float32).reshape(3, 3)
                depth_factor = 1000. / cam_dicts[scene_id][str_im_id]['depth_scale']
                record = {
                    "dataset_name": self.name,
                    'file_name': osp.relpath(rgb_path, PROJ_ROOT),
                    'depth_file': osp.relpath(depth_path, PROJ_ROOT),
                    "depth_factor": depth_factor,
                    'height': self.height,
                    'width': self.width,
                    'image_id': im_id_global,  # unique image_id in the dataset, for coco evaluation
                    "scene_im_id": "{}/{}".format(scene_id, im_id),  # for evaluation
                    "cam": cam,
                    "img_type": 'real'
                }
                im_id_global += 1
                insts = []
                for anno_i, anno in enumerate(gt_dicts[scene_id][str_im_id]):
                    obj_id = anno['obj_id']
                    if ref.tless_bop19.id2obj[obj_id] not in self.select_objs:
                        continue
                    cur_label = self.cat2label[obj_id]  # 0-based label
                    R = np.array(anno['cam_R_m2c'], dtype='float32').reshape(3, 3)
                    t = np.array(anno['cam_t_m2c'], dtype='float32') / 1000.0
                    pose = np.hstack([R, t.reshape(3, 1)])
                    quat = mat2quat(R).astype('float32')
                    allo_q = mat2quat(egocentric_to_allocentric(pose)[:3, :3]).astype('float32')

                    proj = (record["cam"] @ t.T).T
                    proj = proj[:2] / proj[2]

                    bbox_visib = gt_info_dicts[scene_id][str_im_id][anno_i]['bbox_visib']
                    bbox_obj = gt_info_dicts[scene_id][str_im_id][anno_i]['bbox_obj']
                    x1, y1, w, h = bbox_visib
                    if self.filter_invalid:
                        if h <= 1 or w <= 1:
                            self.num_instances_without_valid_box += 1
                            continue

                    mask_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(im_id, anno_i))
                    mask_visib_file = osp.join(scene_root, "mask_visib/{:06d}_{:06d}.png".format(im_id, anno_i))
                    assert osp.exists(mask_file), mask_file
                    assert osp.exists(mask_visib_file), mask_visib_file
                    # load mask visib  TODO: load both mask_visib and mask_full
                    mask_single = mmcv.imread(mask_visib_file, "unchanged")
                    area = mask_single.sum()
                    if area < 3:  # filter out too small or nearly invisible instances
                        self.num_instances_without_valid_segmentation += 1
                        continue
                    mask_rle = binary_mask_to_rle(mask_single, compressed=True)
                    inst = {
                        'category_id': cur_label,  # 0-based label
                        'bbox': bbox_visib,  # TODO: load both bbox_obj and bbox_visib
                        'bbox_mode': BoxMode.XYWH_ABS,
                        'pose': pose,
                        "quat": quat,
                        "trans": t,
                        "allo_quat": allo_q,
                        "centroid_2d": proj,  # absolute (cx, cy)
                        "segmentation": mask_rle,
                        "mask_full_file": mask_file,  # TODO: load as mask_full, rle
                    }

                    insts.append(inst)
                if len(insts) == 0:  # filter im without anno
                    continue
                record['annotations'] = insts
                dataset_dicts.append(record)

        if self.num_instances_without_valid_segmentation > 0:
            logger.warning("Filtered out {} instances without valid segmentation. "
                        "There might be issues in your dataset generation process.".format(
                            self.num_instances_without_valid_segmentation))
        if self.num_instances_without_valid_box > 0:
            logger.warning("Filtered out {} instances without valid box. "
                        "There might be issues in your dataset generation process.".format(
                            self.num_instances_without_valid_box))
        ##########################################################################
        if self.num_to_load > 0:
            self.num_to_load = min(int(self.num_to_load), len(dataset_dicts))
            dataset_dicts = dataset_dicts[:self.num_to_load]
        logger.info("loaded dataset dicts, num_images: {}, using {}s".format(
            len(dataset_dicts),
            time.perf_counter() - t_start))

        mmcv.dump(dataset_dicts, cache_path, protocol=4)
        logger.info("Dumped dataset_dicts to {}".format(cache_path))
        return dataset_dicts

    def __len__(self):
        return self.num_to_load

    def image_aspect_ratio(self):
        return self.width / self.height  # 4/3


########### register datasets ############################################################


def get_tless_metadata(obj_names):
    # task specific metadata
    # TODO: provide symetry info for models here
    meta = {"thing_classes": obj_names}
    return meta


################################################################################

SPLITS_TLESS = dict(
    tless_bop_test_primesense=dict(
        name="tless_bop_test_primesense",
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/tless/test_primesense"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/tless/models_cad"),
        objs=ref.tless_bop19.objects,  # selected objects
        ann_file=osp.join(DATASETS_ROOT, "BOP_DATASETS/tless/test_targets_bop19.json"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=540,
        width=720,
        cache_dir=".cache",
        use_cache=True,
        num_to_load=-1,
        filter_invalid=False,
        ref_key="tless_bop19",
    ),
)


# single objs (num_class is from all objs)
for obj in ref.tless_bop19.objects:
    name = "tless_bop_{}_test_primesense".format(obj)
    select_objs = [obj]
    if name not in SPLITS_TLESS:
        SPLITS_TLESS[name] = dict(
            name=name,
            dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/tless/test_primesense"),
            models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/tless/models_cad"),
            objs=ref.tless_bop19.objects,
            select_objs=select_objs,  # selected objects
            ann_file=osp.join(DATASETS_ROOT, "BOP_DATASETS/tless/test_targets_bop19.json"),
            scale_to_meter=0.001,
            with_masks=True,  # (load masks but may not use it)
            with_depth=True,  # (load depth path here, but may not use it)
            height=540,
            width=720,
            cache_dir=".cache",
            use_cache=True,
            num_to_load=-1,
            filter_invalid=False,
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
    if name in SPLITS_TLESS:
        used_cfg = SPLITS_TLESS[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg
    DatasetCatalog.register(name, TLESS_BOP_TEST_Dataset(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        id="tless",  # NOTE: for pvnet to determine module
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="coco_bop",  # TODO: add bop evaluator
        **get_tless_metadata(obj_names=used_cfg["objs"]),
    )


def get_available_datasets():
    return list(SPLITS_TLESS.keys())


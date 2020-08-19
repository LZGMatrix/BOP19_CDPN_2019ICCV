import os
import copy
import logging
import os.path as osp

import hashlib
import mmcv
import random
import cv2
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
from pycocotools import mask as maskUtils

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from adet.data.augmentation import InstanceAugInput, RandomCropWithInstance
from adet.data.dataset_mapper import segmToMask, segmToRLE
from adet.data.detection_utils import (
    annotations_to_instances,
    build_augmentation,
    transform_instance_annotations,
)
from core.utils.augment import AugmentRGB
from core.utils.ssd_color_transform import ColorAugSSDTransform
from lib.utils.utils import lazy_property

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["MyDatasetMapperWithBasis"]

logger = logging.getLogger(__name__)


class MyDatasetMapperWithBasis(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        if hasattr(self, "img_format"):
            self.image_format = self.img_format

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        self.augmentation = build_augmentation(cfg, is_train)

        if cfg.INPUT.CROP.ENABLED and is_train:
            self.augmentation.insert(
                0,
                RandomCropWithInstance(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.CROP_INSTANCE,
                ),
            )
            logging.getLogger(__name__).info(
                "Cropping used in training: " + str(self.augmentation[0])
            )

        if cfg.INPUT.COLOR_AUG_ON and cfg.INPUT.COLOR_AUG_TYPE.lower() == "ssd":
            self.augmentation.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            logging.getLogger(__name__).info(
                "Color augmnetation used in training: " + str(self.augmentation[-1])
            )

        # fmt: off
        self.basis_loss_on       = cfg.MODEL.BASIS_MODULE.LOSS_ON
        self.ann_set             = cfg.MODEL.BASIS_MODULE.ANN_SET
        # fmt: on
        self.cfg = cfg

        # NOTE: color augmentation config
        self.color_aug_on = cfg.INPUT.COLOR_AUG_ON
        self.color_aug_type = cfg.INPUT.COLOR_AUG_TYPE
        self.color_aug_code = cfg.INPUT.COLOR_AUG_CODE
        if is_train and self.color_aug_on and self.color_aug_type.lower() != 'ssd':
            self.color_augmentor = self._get_color_augmentor(aug_type=self.color_aug_type, aug_code=self.color_aug_code)
            logging.getLogger(__name__).info(
                f"Color augmnetation used in training: {self.color_aug_type}"
            )
        else:
            self.color_augmentor = None

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        except Exception as e:
            print(dataset_dict["file_name"])
            print(e)
            raise e
        try:
            utils.check_image_size(dataset_dict, image)
        except SizeMismatchError as e:
            expected_wh = (dataset_dict["width"], dataset_dict["height"])
            image_wh = (image.shape[1], image.shape[0])
            if (image_wh[1], image_wh[0]) == expected_wh:
                print("transposing image {}".format(dataset_dict["file_name"]))
                image = image.transpose(1, 0, 2)
            else:
                raise e

        # NOTE: color aug (if the type is ssd, this is None and will be applied later)
        if self.is_train and self.color_aug_on and self.color_augmentor is not None:
            image = self._color_aug(image, self.color_aug_type)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(
                dataset_dict.pop("sem_seg_file_name"), "L"
            ).squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = InstanceAugInput(image, sem_seg=sem_seg_gt, instances=dataset_dict["annotations"])
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
                min_box_size=self.proposal_min_box_size,
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pano_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if self.basis_loss_on and self.is_train:
            # load basis supervisions
            if self.ann_set == "coco":
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("train2017", "thing_train2017")
                    .replace("image/train", "thing_train")
                )
            else:
                basis_sem_path = (
                    dataset_dict["file_name"]
                    .replace("coco", "lvis")
                    .replace("train2017", "thing_train")
                )
            # change extension to npz
            basis_sem_path = osp.splitext(basis_sem_path)[0] + ".npz"
            basis_sem_gt = np.load(basis_sem_path)["mask"]
            basis_sem_gt = transforms.apply_segmentation(basis_sem_gt)
            basis_sem_gt = torch.as_tensor(basis_sem_gt.astype("long"))
            dataset_dict["basis_sem"] = basis_sem_gt
        return dataset_dict

    def _get_color_augmentor(self, aug_type="ROI10D", aug_code=None):
        # fmt: off
        if aug_type.lower() == "roi10d":
            color_augmentor = AugmentRGB(
                brightness_delta=2.5 / 255.,  #0,
                lighting_std=0.3,
                saturation_var=(0.95, 1.05),  #(1, 1),
                contrast_var=(0.95, 1.05))  # (1, 1))  #
        elif aug_type.lower() == "aae":
            from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                           Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                           Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                           GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                           DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                           MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                           Affine, PiecewiseAffine, ElasticTransformation)

            aug_code = """Sequential([
                # Sometimes(0.5, PerspectiveTransform(0.05)),
                # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
                # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
                Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),
                Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),
                Sometimes(0.5, Add((-25, 25), per_channel=0.3)),
                Sometimes(0.3, Invert(0.2, per_channel=True)),
                Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
                Sometimes(0.5, Multiply((0.6, 1.4))),
                Sometimes(0.5, ContrastNormalization((0.5, 2.2), per_channel=0.3))
                ], random_order = False) """
            # for darker objects, e.g. LM driller: use BOOTSTRAP_RATIO: 16 and weaker augmentation
            aug_code_weaker = """Sequential([
                Sometimes(0.4, CoarseDropout( p=0.1, size_percent=0.05) ),
                # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
                Sometimes(0.5, GaussianBlur(np.random.rand())),
                Sometimes(0.5, Add((-20, 20), per_channel=0.3)),
                Sometimes(0.4, Invert(0.20, per_channel=True)),
                Sometimes(0.5, Multiply((0.7, 1.4), per_channel=0.8)),
                Sometimes(0.5, Multiply((0.7, 1.4))),
                Sometimes(0.5, ContrastNormalization((0.5, 2.0), per_channel=0.3))
                ], random_order=False)"""
            color_augmentor = eval(aug_code)
        elif aug_type.lower() == "code":  # assume imgaug
            from imgaug.augmenters import (Sequential, SomeOf, OneOf, Sometimes, WithColorspace, WithChannels, Noop,
                                           Lambda, AssertLambda, AssertShape, Scale, CropAndPad, Pad, Crop, Fliplr,
                                           Flipud, Superpixels, ChangeColorspace, PerspectiveTransform, Grayscale,
                                           GaussianBlur, AverageBlur, MedianBlur, Convolve, Sharpen, Emboss, EdgeDetect,
                                           DirectedEdgeDetect, Add, AddElementwise, AdditiveGaussianNoise, Multiply,
                                           MultiplyElementwise, Dropout, CoarseDropout, Invert, ContrastNormalization,
                                           Affine, PiecewiseAffine, ElasticTransformation)

            aug_code = self.color_aug_code
            color_augmentor = eval(aug_code)
        elif aug_type.lower() == 'code_albu':
            from albumentations import (HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
                                        Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion,
                                        HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur,
                                        MedianBlur, IAAPiecewiseAffine, IAASharpen, IAAEmboss, RandomContrast,
                                        RandomBrightness, Flip, OneOf, Compose, CoarseDropout, RGBShift, RandomGamma,
                                        RandomBrightnessContrast, JpegCompression, InvertImg)
            aug_code = """Compose([
                CoarseDropout(max_height=0.05*480, max_holes=0.05*640, p=0.4),
                OneOf([
                    IAAAdditiveGaussianNoise(p=0.5),
                    GaussNoise(p=0.5),
                ], p=0.2),
                OneOf([
                    MotionBlur(p=0.2),
                    MedianBlur(blur_limit=3, p=0.1),
                    Blur(blur_limit=3, p=0.1),
                ], p=0.2),
                OneOf([
                    CLAHE(clip_limit=2),
                    IAASharpen(),
                    IAAEmboss(),
                    RandomBrightnessContrast(),
                ], p=0.3),
                InvertImg(p=0.2),
                RGBShift(r_shift_limit=105, g_shift_limit=45, b_shift_limit=40, p=0.5),
                RandomContrast(limit=0.9, p=0.5),
                RandomGamma(gamma_limit=(80,120), p=0.5),
                RandomBrightness(limit=1.2, p=0.5),
                HueSaturationValue(hue_shift_limit=172, sat_shift_limit=20, val_shift_limit=27, p=0.3),
                JpegCompression(quality_lower=4, quality_upper=100, p=0.4),
            ], p=0.8)"""
            color_augmentor = eval(self.color_aug_code)
        else:
            color_augmentor = None
        # fmt: on
        return color_augmentor

    def _color_aug(self, image, aug_type="ROI10D"):
        # assume image in [0, 255] uint8
        if aug_type.lower() == "roi10d":  # need normalized image in [0,1]
            image = np.asarray(image / 255.0, dtype=np.float32).copy()
            image = self.color_augmentor.augment(image)
            image = (image * 255.0 + 0.5).astype(np.uint8)
            return image
        elif aug_type.lower() in ["aae", "code"]:
            # imgaug need uint8
            return self.color_augmentor.augment_image(image)
        elif aug_type.lower() in ["code_albu"]:
            augmented = self.color_augmentor(image=image)
            return augmented["image"]
        else:
            raise ValueError("aug_type: {} is not supported.".format(aug_type))

    @lazy_property
    def _bg_img_paths(self):
        logger.info("get bg image paths")
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

        cfg = self.cfg
        # random.choice(bg_img_paths)
        hashed_file_name = hashlib.md5(("{}_{}_{}_get_bg_imgs".format(cfg.INPUT.BG_IMGS_ROOT, cfg.INPUT.NUM_BG_IMGS,
                                                                      cfg.INPUT.BG_TYPE)).encode("utf-8")).hexdigest()
        cache_path = osp.join("bg_paths_{}.pkl".format(hashed_file_name))
        if osp.exists(cache_path):
            logger.info("get bg_paths from cache file: {}".format(cache_path))
            bg_img_paths = mmcv.load(cache_path)
            logger.info("num bg imgs: {}".format(len(bg_img_paths)))
            assert len(bg_img_paths) > 0
            return bg_img_paths

        logger.info("building bg imgs cache {}...".format(cfg.INPUT.BG_TYPE))
        assert osp.exists(cfg.INPUT.BG_IMGS_ROOT), f"BG ROOT: {cfg.INPUT.BG_IMGS_ROOT} does not exist"
        if cfg.INPUT.BG_TYPE == "coco":
            img_paths = [
                osp.join(cfg.INPUT.BG_IMGS_ROOT, fn.name)
                for fn in os.scandir(cfg.INPUT.BG_IMGS_ROOT)
                if ".png" in fn.name or "jpg" in fn.name
            ]
        elif cfg.INPUT.BG_TYPE == "VOC_table":  # used in original deepim
            VOC_root = cfg.INPUT.BG_IMGS_ROOT  # path to "VOCdevkit/VOC2012"
            VOC_image_set_dir = osp.join(VOC_root, "ImageSets/Main")
            VOC_bg_list_path = osp.join(VOC_image_set_dir, "diningtable_trainval.txt")
            with open(VOC_bg_list_path, "r") as f:
                VOC_bg_list = [
                    line.strip("\r\n").split()[0] for line in f.readlines() if line.strip("\r\n").split()[1] == "1"
                ]
            img_paths = [osp.join(VOC_root, "JPEGImages/{}.jpg".format(bg_idx)) for bg_idx in VOC_bg_list]
        elif cfg.INPUT.BG_TYPE == "VOC":
            VOC_root = cfg.INPUT.BG_IMGS_ROOT  # path to "VOCdevkit/VOC2012"
            img_paths = [
                osp.join(VOC_root, "JPEGImages", fn.name)
                for fn in osp.scandir(osp.join(cfg.INPUT.BG_IMGS_ROOT, "JPEGImages"))
                if ".jpg" in fn.name
            ]
        elif cfg.INPUT.BG_TYPE == "SUN2012":
            img_paths = [
                osp.join(cfg.INPUT.BG_IMGS_ROOT, "JPEGImages", fn.name)
                for fn in os.scandir(osp.join(cfg.BG_IMGS_ROOT, "JPEGImages"))
                if ".jpg" in fn.name
            ]
        else:
            raise ValueError(f"BG_TYPE: {cfg.INPUT.BG_TYPE} is not supported")
        assert len(img_paths) > 0, len(img_paths)

        num_bg_imgs = min(len(img_paths), cfg.INPUT.NUM_BG_IMGS)
        bg_img_paths = np.random.choice(img_paths, num_bg_imgs)

        mmcv.dump(bg_img_paths, cache_path)
        logger.info("num bg imgs: {}".format(len(bg_img_paths)))
        assert len(bg_img_paths) > 0
        return bg_img_paths

    def replace_bg(self, im, im_mask):
        # add background to the image
        H, W = im.shape[:2]
        ind = random.randint(0, len(self._bg_img_paths) - 1)
        filename = self._bg_img_paths[ind]
        # _bg_img = mmcv.imread(filename, 'color')
        _bg_img = utils.read_image(filename, format=self.image_format)
        try:
            # randomly crop a region as background
            bw = _bg_img.shape[1]
            bh = _bg_img.shape[0]
            x1 = np.random.randint(0, int(bw / 3))
            y1 = np.random.randint(0, int(bh / 3))
            x2 = np.random.randint(int(2 * bw / 3), bw)
            y2 = np.random.randint(int(2 * bh / 3), bh)
            bg_img = cv2.resize(_bg_img[y1:y2, x1:x2], (W, H), interpolation=cv2.INTER_LINEAR)
        except:
            bg_img = np.zeros((H, W, 3), dtype=np.uint8)
            logger.warn("bad background image: {}".format(filename))

        if len(bg_img.shape) != 3:
            bg_img = np.zeros((H, W, 3), dtype=np.uint8)
            logger.warn("bad background image: {}".format(filename))
        mask_bg = np.where(im_mask == 0)
        im[mask_bg[0], mask_bg[1], :] = bg_img[mask_bg[0], mask_bg[1], :3]
        im = im.astype(np.uint8)
        return im

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
my custom main script (modified from train_net.py)
"""

import logging
import os
import os.path as osp
import sys
import mmcv
from collections import OrderedDict
import torch
import detectron2.utils.comm as comm
from detectron2.engine import default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import verify_results

from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../"))
from lib.utils.setup_logger import setup_my_logger
from lib.utils.utils import dprint, iprint
from core.dataset_factory import register_datasets
from core.utils.env_utils import setup_for_distributed
from core.engine import MyTrainer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    ##############################
    # NOTE: pop some unwanterd configs in detectron2
    cfg.SOLVER.pop("STEPS", None)
    cfg.SOLVER.pop("MAX_ITER", None)
    # NOTE: get optimizer from string cfg dict
    if cfg.SOLVER.OPTIMIZER_CFG != "":
        optim_cfg = eval(cfg.SOLVER.OPTIMIZER_CFG)
        iprint("optimizer_cfg:", optim_cfg)
        cfg.SOLVER.OPTIMIZER_NAME = optim_cfg['type']
        cfg.SOLVER.BASE_LR = optim_cfg['lr']
        cfg.SOLVER.MOMENTUM = optim_cfg.get("momentum", 0.9)
        cfg.SOLVER.WEIGHT_DECAY = optim_cfg.get("weight_decay", 1e-4)
    if cfg.get("DEBUG", False):
        iprint("DEBUG")
        args.num_gpus = 1
        args.num_machines = 1
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.TRAIN.PRINT_FREQ = 1
    if cfg.TRAIN.get("VERBOSE", False):
        cfg.TRAIN.PRINT_FREQ = 1

    # register datasets
    dataset_names = cfg.DATASETS.TRAIN + cfg.DATASETS.TEST
    register_datasets(dataset_names)

    cfg.RESUME = args.resume
    ##########################################
    cfg.freeze()
    default_setup(cfg, args)

    setup_for_distributed(is_master=comm.is_main_process())

    rank = comm.get_rank()
    setup_my_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")
    setup_my_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="core")

    return cfg


def main(args):
    cfg = setup(args)
    iprint("finished setup")

    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MyTrainer.test(cfg, model) # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(MyTrainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (10000, rlimit[1]))

    args = default_argument_parser().parse_args()
    iprint("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

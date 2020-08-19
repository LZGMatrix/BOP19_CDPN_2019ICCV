import os
import pickle
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.train_loop import HookBase
from fvcore.common.checkpoint import PeriodicCheckpointer as _PeriodicCheckpointer
from fvcore.common.file_io import PathManager


class MyFvPeriodicCheckpointer(_PeriodicCheckpointer):
    """
    Save checkpoints periodically. When `.step(iteration)` is called, it will
    execute `checkpointer.save` on the given checkpointer, if iteration is a
    multiple of period or if `max_iter` is reached.
    """

    def __init__(
        self,
        checkpointer: Any,  # pyre-ignore
        period: int,
        max_iter: Optional[int] = None,
        max_to_keep: Optional[int] = None,
        iters_per_epoch: Optional[int] = None,
    ) -> None:
        """
        Args:
            checkpointer (Any): the checkpointer object used to save
            checkpoints.
            period (int): the period to save checkpoint.
            max_iter (int): maximum number of iterations. When it is reached,
                a checkpoint named "model_final" will be saved.
            max_to_keep (int): maximum number of most current checkpoints to keep,
                previous checkpoints will be deleted
            iters_per_epoch (int): number of iters per epoch
        """
        self.checkpointer = checkpointer
        self.period = int(period)
        self.max_iter = max_iter
        if max_to_keep is not None:
            assert max_to_keep > 0
        self.max_to_keep = max_to_keep
        self.recent_checkpoints = []  # pyre-ignore

        if iters_per_epoch is not None:
            assert iters_per_epoch > 0
        self.iters_per_epoch = iters_per_epoch

    def step(self, iteration: int, **kwargs: Any) -> None:
        """
        Perform the appropriate action at the given iteration.
        Args:
            iteration (int): the current iteration, ranged in [0, max_iter-1].
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        if self.iters_per_epoch is not None:  # NOTE: added
            additional_state.update({"epoch": iteration // self.iters_per_epoch})
        additional_state.update(kwargs)
        if (iteration + 1) % self.period == 0:
            self.checkpointer.save("model_{:07d}".format(iteration), **additional_state)

            if self.max_to_keep is not None:
                self.recent_checkpoints.append(self.checkpointer.get_checkpoint_file())
                # pyre-fixme[6]: Expected `int` for 1st param but got `Optional[int]`.
                # pyre-fixme[6]: Expected `int` for 1st param but got `Optional[int]`.
                if len(self.recent_checkpoints) > self.max_to_keep:
                    file_to_delete = self.recent_checkpoints.pop(0)
                    if PathManager.exists(
                        file_to_delete
                    ) and not file_to_delete.endswith("model_final.pth"):
                        PathManager.rm(file_to_delete)

        if iteration >= self.max_iter - 1:  # pyre-ignore
            self.checkpointer.save("model_final", **additional_state)

    def save(self, name: str, **kwargs: Any) -> None:
        """
        Same argument as :meth:`Checkpointer.save`.
        Use this method to manually save checkpoints outside the schedule.
        Args:
            name (str): file name.
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        self.checkpointer.save(name, **kwargs)


class MyPeriodicCheckpointer(MyFvPeriodicCheckpointer, HookBase):
    """NOTE: MyFvPeriodicCheckpointer support record epoch
    Same as :class:`detectron2.checkpoint.PeriodicCheckpointer`, but as a hook.

    Note that when used as a hook,
    it is unable to save additional data other than what's defined
    by the given `checkpointer`.

    It is executed every ``period`` iterations and after the last iteration.
    """

    def before_train(self):
        self.max_iter = self.trainer.max_iter

    def after_step(self):
        # No way to use **kwargs
        self.step(self.trainer.iter)

"""Custom wandb integrations"""


import dataclasses
import os
from typing import Dict

from transformers.integrations import WandbCallback
from transformers.utils import is_torch_tpu_available, logging

import wandb

logger = logging.get_logger(__name__)


class CustomWandbCallback(WandbCallback):
    def __init__(self, wandb_args: Dict[str, str], *args, **kwargs):
        """Just do standard wandb init, but save the arguments for setup."""
        super().__init__(*args, **kwargs)
        self._wandb_args = wandb_args

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.
        One can subclass and override this method to customize the setup if
        needed. Find more information
        [here](https://docs.wandb.ai/integrations/huggingface). You can also
        override the following environment variables:
        Environment:
            WANDB_LOG_MODEL (`bool`, *optional*, defaults to `False`):
                Whether or not to log model as artifact at the end of training.
                Use along with
                *TrainingArguments.load_best_model_at_end* to upload best model.
            WANDB_WATCH (`str`, *optional* defaults to `"gradients"`):
                Can be `"gradients"`, `"all"` or `"false"`. Set to `"false"` to
                disable gradient logging or `"all"` to log gradients and
                parameters.
        """
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            if self._wandb.run is None:
                self._wandb.init(
                    project=self._wandb_args["project"],
                    group=self._wandb_args["group"],
                    name=self._wandb_args["name"],
                    config=dataclasses.asdict(args),
                    settings=wandb.Settings(start_method="fork"),
                )

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric(
                    "*", step_metric="train/global_step", step_sync=True
                )

            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model,
                    log=os.getenv("WANDB_WATCH", "gradients"),
                    log_freq=max(100, args.logging_steps),
                )

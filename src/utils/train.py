from types import SimpleNamespace
import json
import os

import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

# from ..data.datamodule import FMSDataModule
# from ..models.modelmodule import FMSModelModule


def do_train(json_cfg):
    # convert the JSON config to a Python object
    cfg = json.loads(json.dumps(json_cfg), object_hook=lambda d: SimpleNamespace(**d))

    seed_everything(cfg.train.seed)
    lr_monitor = LearningRateMonitor("epoch")
    progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=3)
    run = wandb.init(
        name=cfg.exp_name,
        dir=cfg.output_dir,
        notes=cfg.notes,
        mode="disabled" if (cfg.train.debug or cfg.disable_wandb) else "online",
    )
    pl_logger = WandbLogger(
        save_dir=cfg.output_dir,
        log_model=True,  # log model at the end of training
        mode="disabled" if (cfg.train.debug or cfg.disable_wandb) else "online",
    )
    pl_logger.experiment.config.update(json_cfg)

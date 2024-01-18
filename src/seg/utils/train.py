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

from .common import json_to_py
from ..data.datamodule import SegDataModule
from ..models.modelmodule import SegModelModule


def do_train(json_cfg):
    # convert the JSON config to a Python object
    cfg = json_to_py(json_cfg)

    seed_everything(cfg.train.seed)

    data = SegDataModule(cfg)
    model = SegModelModule(cfg)

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

    trainer = Trainer(
        default_root_dir=cfg.output_dir,
        deterministic=cfg.train.deterministic,
        # num_nodes=cfg.training.num_gpus,
        accelerator=cfg.train.accelerator,
        precision="16-mixed" if cfg.train.use_amp else 32,
        fast_dev_run=cfg.train.debug,  # run only 1 train batch and 1 val batch
        max_epochs=cfg.train.epoch,
        gradient_clip_val=cfg.train.gradient_clip_val,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        callbacks=[lr_monitor, progress_bar, model_summary],
        logger=pl_logger,
        num_sanity_val_steps=0,
        log_every_n_steps=5,
        sync_batchnorm=True,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, data)

    run.finish()
    
    return data, model

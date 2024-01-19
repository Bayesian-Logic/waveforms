import importlib
import logging
from types import SimpleNamespace
import os

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import matplotlib.pyplot as plt

LOG = logging.getLogger("seg.models")


class SegModelModule(pl.LightningModule):
    def __init__(self, cfg: SimpleNamespace, pkg: str = "seg.models"):
        super().__init__()
        self.cfg = cfg
        models_module = importlib.import_module(".models", pkg)
        model_class_name = cfg.model.class_name
        model_class = getattr(models_module, model_class_name)
        model_params = getattr(cfg.model, model_class_name)
        self.model = model_class(**vars(model_params))
        self.loss_fn = nn.BCEWithLogitsLoss(
            reduction="none", pos_weight=torch.Tensor([self.cfg.model.pos_weight])
        )

    def forward(self, batch):
        out = {}
        # We have only 1 feature so we convert N, T to N, F=1, T.
        # The output of N, T, C=1 is then converted to N, T.
        logits = self.model(batch["waveform"].unsqueeze(1)).squeeze(2)
        if "target" in batch:
            batch_loss = self.loss_fn(logits, batch["target"])
            out["loss"] = batch_loss.mean()
            out["loss_full"] = batch_loss.detach().cpu().mean(dim=1).numpy()
        logits = logits.detach().cpu()
        out["onset_pred"] = logits.argmax(dim=1).numpy()
        out["target_pred"] = logits.sigmoid().numpy()
        return out

    def on_train_epoch_start(self):
        self.train_epoch_loss = np.array([])
        self.train_epoch_err = np.array([])
        self.epoch_metrics = {}
        self.epoch_preds_images = []

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        self.train_epoch_loss = np.append(self.train_epoch_loss, out["loss_full"])
        self.train_epoch_err = np.append(
            self.train_epoch_err, out["onset_pred"] - batch["onset"].cpu().numpy()
        )
        return out["loss"]

    def on_validation_epoch_start(self):
        self.val_epoch_loss = np.array([])
        self.val_epoch_err = np.array([])

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        self.val_epoch_loss = np.append(self.val_epoch_loss, out["loss_full"])
        self.val_epoch_err = np.append(
            self.val_epoch_err, out["onset_pred"] - batch["onset"].cpu().numpy()
        )
        return out["loss"]

    def on_validation_epoch_end(self):
        LOG.debug("Validation epoch ending.")
        self.epoch_metrics.update(
            {
                "val/loss": self.val_epoch_loss.mean(),
                "val/mae": (
                    np.abs(self.val_epoch_err) / self.cfg.train.samprate
                ).mean(),
            }
        )
        # Plot the prediction error histograms
        fig = plt.figure()
        plt.hist(
            self.val_epoch_err / self.cfg.train.samprate, bins=np.linspace(-1, 1, 41)
        )
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency")
        plt.title("Val Error")
        fig_file = os.path.join(self.cfg.output_dir, "val_err.jpg")
        fig.savefig(fig_file)
        plt.close(fig)
        self.epoch_preds_images.append(str(fig_file))
        del self.val_epoch_loss
        del self.val_epoch_err

    def on_train_epoch_end(self):
        LOG.debug("Training epoch ending.")
        self.epoch_metrics.update(
            {
                "trainer/loss": self.train_epoch_loss.mean(),
                "trainer/mae": (
                    np.abs(self.train_epoch_err) / self.cfg.train.samprate
                ).mean(),
            }
        )
        self.log_dict(
            self.epoch_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True
        )
        if len(self.epoch_preds_images):
            self.logger.log_image(
                key="Pred Err", images=self.epoch_preds_images, step=self.global_step
            )
        del self.train_epoch_loss
        del self.train_epoch_err
        del self.epoch_metrics
        del self.epoch_preds_images

    def configure_optimizers(self):
        optim_class_name = self.cfg.optimizer.class_name
        optim_module = importlib.import_module(".optim", "torch")
        optim_class = getattr(optim_module, optim_class_name)
        extra_params = getattr(self.cfg.optimizer, optim_class_name, {})
        optimizer = optim_class(
            self.parameters(), lr=self.cfg.optimizer.lr, **vars(extra_params)
        )
        if not hasattr(self.cfg.optimizer, "scheduler"):
            return optimizer
        elif self.cfg.optimizer.scheduler.lower() != "cosine":
            raise ValueError(
                "Unsupported scheduler '{}'".format(self.cfg.optimizer.scheduler)
            )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_warmup_steps=self.cfg.optimizer.num_warmup_steps,
        )
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr",
            },
        ]

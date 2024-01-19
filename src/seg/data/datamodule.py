from pathlib import Path, PosixPath
from types import SimpleNamespace
import logging
import os

import sqlite3
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
import pyarrow.parquet as pq
import pandas as pd

from .dataset import RandomInterval
from ..utils.common import run_query, is_validation_waveform

LOG = logging.getLogger("seg.data")


class SegDataModule(pl.LightningDataModule):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)
        self.prepare_dir = self.data_dir / "prepared"
        # num_samples is the number of samples in each datum.
        self.num_samples = self.cfg.train.data_window * self.cfg.train.samprate

    # no prepare_data is needed

    def _read_data_frame(self, parquet_file) -> pd.DataFrame:
        table = pq.read_table(parquet_file)
        return table.to_pandas()

    def setup(self, stage: str):
        LOG.info(f"SegDataModule setup {stage=}")
        if stage == "fit":
            full_df = self._read_data_frame(self.data_dir / self.cfg.train.train_file)
            # divide the training waveforms into training and validation using
            # fold_idx and num_folds
            if self.cfg.train.train_full:
                self.train_df = self.val_df = full_df
            else:
                mask = full_df.apply(
                    lambda row: is_validation_waveform(
                        f"{row.orid}_{row.arid}",
                        self.cfg.train.fold_idx,
                        self.cfg.train.num_folds,
                    ),
                    axis=1,
                )
                self.train_df = full_df[~mask]
                self.val_df = full_df[mask]
            LOG.info(
                f"Training fold {self.cfg.train.fold_idx} of {self.cfg.train.num_folds}"
                f" folds has {len(self.train_df)} train waveforms"
                f" and {len(self.val_df)} validation waveforms."
            )
        elif stage == "test":
            self.test_df = self._read_data_frame(
                self.data_dir / self.cfg.train.test_file
            )
            LOG.info(f"Test has {len(self.test_df)} waveforms.")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            RandomInterval(
                self.train_df,
                self.cfg.train.data_window,
                self.cfg.train.train_window,
                self.cfg.train.samprate,
                self.cfg.train.target_sigma,
                self.cfg.train.target_length,
            ),
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            RandomInterval(
                self.val_df,
                self.cfg.train.data_window,
                self.cfg.train.train_window,
                self.cfg.train.samprate,
                self.cfg.train.target_sigma,
                self.cfg.train.target_length,
            ),
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            RandomInterval(
                self.test_df,
                self.cfg.train.data_window,
                self.cfg.train.train_window,
                self.cfg.train.samprate,
            ),
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            drop_last=False,
        )

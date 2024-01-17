from pathlib import Path, PosixPath
from types import SimpleNamespace
import logging
import os

import sqlite3
import pytorch_lightning as pl
import numpy as np
from skimage.transform import resize_local_mean
from torch.utils.data import DataLoader

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

    def prepare_data(self):
        if os.path.exists(self.prepare_dir):
            LOG.info("Data has already been prepared.")
            return
        conn = sqlite3.connect(self.data_dir / self.cfg.train.train_file)
        os.makedirs(self.prepare_dir)
        for row in run_query(
            conn,
            "select orid, arid, start_time, end_time, arrival_time, samprate, nsamp, dtype, data from waveform",
        ):
            # We will save a subset of the waveform of size `data_window` around the arrival time
            # for each datum
            data = np.ndarray(shape=(int(row.nsamp),), dtype=row.dtype, buffer=row.data)
            st = (
                row.arrival_time - self.cfg.train.data_window // 2 - row.start_time
            ) * row.samprate
            num = self.cfg.train.data_window * row.samprate
            data = data[int(st) : int(st + num)].astype(np.float32)
            assert(len(data) == num)
            # Convert the data to the target number of samples
            if len(data) != self.num_samples:
                data = resize_local_mean(data, (self.num_samples,), preserve_range=True)
            np.save(
                self.prepare_dir / f"{row.orid}_{row.arid}",
                data,
                allow_pickle=False,
            )

    def setup(self, stage: str):
        LOG.info(f"SegDataModule setup {stage=}")
        all_files = sorted(list(self.prepare_dir.glob("*.npy")))
        if stage in ("fit", "validate"):
            # divide the training waveforms into training and validation using
            # fold_idx and num_folds
            if self.cfg.train.train_full:
                self.train_files = self.val_files = all_files
            else:
                self.train_files, self.val_files = [], []
                for file_path in all_files:
                    if is_validation_waveform(file_path.name, self.cfg.train.fold_idx, self.cfg.train.num_folds):
                        self.val_files.append(file_path)
                    else:
                        self.train_files.append(file_path)
            LOG.info(f"Training fold {self.cfg.train.fold_idx} of {self.cfg.train.num_folds}"
                  f" folds has {len(self.train_files)} train images"
                  f" and {len(self.val_files)} validation images")
        elif stage == "test":
            self.test_files = []
            raise NotImplemented("Test setup is not implemented.")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            RandomInterval(self.train_files, self.cfg.train.data_window,
                self.cfg.train.train_window, self.cfg.train.samprate,
                self.cfg.train.target_sigma, self.cfg.train.target_length),
            batch_size = self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            RandomInterval(self.val_files, self.cfg.train.data_window,
                self.cfg.train.train_window, self.cfg.train.samprate,
                self.cfg.train.target_sigma, self.cfg.train.target_length),
            batch_size = self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            drop_last=False,
        )
  
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            RandomInterval(self.test_files, self.cfg.train.data_window,
                self.cfg.train.train_window, self.cfg.train.samprate),
            batch_size = self.cfg.train.batch_size,
            shuffle=False,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            drop_last=False,
        )
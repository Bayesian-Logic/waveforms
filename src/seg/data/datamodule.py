from pathlib import Path, PosixPath
from types import SimpleNamespace
import logging
import os

import sqlite3
import pytorch_lightning as pl
import numpy as np

from ..utils.common import run_query

LOG = logging.getLogger("seg.data")


class SegDataModule(pl.LightningDataModule):
    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)
        self.prepare_dir = self.data_dir / "prepared"

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
            np.save(
                self.prepare_dir / f"{row.orid}_{row.arid}",
                data[int(st) : int(st + num)],
                allow_pickle=False,
            )

    def setup(self, stage: str):
        LOG.info(f"SegDataModule setup {stage=}")
        if stage in ("fit", "validate"):
            # divide the training waveforms into training and validation using
            # fold_idx and num_folds
            fold_idx = self.cfg.train.fold_idx
            num_folds = self.cfg.train.num_folds

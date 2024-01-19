from pathlib import PosixPath
from typing import List, Dict, Union

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from skimage.transform import resize_local_mean

from ..utils.common import downsample

class RandomInterval(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        orig_dur: int,
        new_dur: int,
        samprate: int,
        target_sigma: int = None,
        target_length: int = None,
    ):
        super().__init__()
        self.df = df
        self.orig_dur = orig_dur
        self.new_dur = new_dur
        self.samprate = samprate
        if target_sigma is not None and target_length is not None:
            length = int(target_length * samprate)
            sigma = int(target_sigma * samprate)
            x = np.ogrid[-length : length + 1]
            self.kernel = np.exp(-(x**2) / (2 * sigma * sigma))
        else:
            self.kernel = None
        self.target_sigma = target_sigma
        self.target_length = target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> Dict[str, Union[np.ndarray, int]]:
        row = self.df.iloc[index]
        # First convert the data to the target number of samples.
        if len(row.data) < self.orig_dur * self.samprate:
            data = resize_local_mean(
                row.data, (self.orig_dur * self.samprate,), preserve_range=True
            )
        elif len(row.data) > self.orig_dur * self.samprate:
            data = downsample(row.data, self.orig_dur * self.samprate)
        else:
            data = row.data
        # Next select a random sample from the data.
        start_time = np.random.uniform(0, self.orig_dur - self.new_dur)
        start_idx = int(start_time * self.samprate)
        end_index = start_idx + int(self.new_dur * self.samprate)
        new_data = np.copy(data[start_idx:end_index])
        new_arrival_idx = self.samprate * self.orig_dur // 2 + 1 - start_idx
        item = {
            "waveform": new_data,
            "onset": new_arrival_idx,
            "name": f"{row.orid}_{row.arid}",
            "index": index,
        }
        if self.kernel is not None:
            label = np.zeros(len(new_data))
            label[new_arrival_idx] = 1
            item["target"] = np.convolve(label, self.kernel, mode="same")
        return item

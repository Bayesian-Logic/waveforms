from pathlib import PosixPath
from typing import List, Dict, Union

import numpy as np
from torch.utils.data import Dataset

class RandomInterval(Dataset):
    def __init__(self, data_files: List[PosixPath], orig_dur:int, new_dur:int,
        samprate:int, target_sigma:int = None, target_length:int = None):
        super().__init__()
        self.data_files = data_files
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
        return len(self.data_files)
    
    def __getitem__(self, index) -> Dict[str, Union[np.ndarray, int]]:
        data = np.load(self.data_files[index])
        start_time = np.random.uniform(0, self.orig_dur - self.new_dur)
        start_idx = int(start_time * self.samprate)
        end_index = start_idx + int(self.new_dur * self.samprate)
        new_data = data[start_idx: end_index]
        new_arrival_idx = self.samprate * self.orig_dur // 2 + 1 - start_idx
        item = {
          "waveform": new_data,
          "onset": new_arrival_idx,
          "name": self.data_files[index].name.split(".")[0]
        }
        if self.kernel is not None:
            label = np.zeros(len(new_data))
            label[new_arrival_idx] = 1
            item["target"] = np.convolve(label, self.kernel, mode="same")
        return item
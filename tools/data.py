import torch
from torch.utils.data import Dataset
import numpy as np
import json

class ChessDataset(Dataset):
    def __init__(
        self,
        path,
        npy_filt_prefix,
        target_mean=0.0,
        target_std=1.0,
    ):
        # Load indices for this split
        self.indices = np.load(path + npy_filt_prefix + "_indices.npy", mmap_mode='r')

        # Load shape metadata
        with open(path + "metadata.json", "r") as f:
            metadata = json.load(f)
        input_shape = tuple(metadata["input_shape"])
        target_shape = tuple(metadata["target_shape"])
        total_rows = metadata["total_rows"]

        # Memory-map the full datasets with correct shape
        self.inputs = np.memmap(
            path + "full_inputs.dat",
            dtype=np.float32,
            mode='r',
            shape=(total_rows,) + input_shape
        )

        self.targets = np.memmap(
            path + "full_targets.dat",
            dtype=np.float32,
            mode='r',
            shape=(total_rows,) + target_shape
        )

        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return len(self.indices)

    def normalize(self, mean, std):
        self.target_mean = mean
        self.target_std = std

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        input_tensor = torch.from_numpy(
            np.array(self.inputs[real_idx], copy=True)
        )

        target = torch.from_numpy(
            np.array(self.targets[real_idx], copy=True)
        )

        target = (target - self.target_mean) / self.target_std

        return input_tensor, target
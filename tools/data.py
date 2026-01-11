import torch
from torch.utils.data import Dataset
import numpy as np

class ChessDataset(Dataset):
    def __init__(
        self,
        path,
        npy_filt_prefix,
        target_mean=0.0,
        target_std=1.0,
    ):
        # Load indices for this split
        self.indices = np.load(path + npy_filt_prefix + "_indicies.npy", mmap_mode='r')

        # Memory-map the full datasets
        self.inputs = np.memmap(
            path+ "full_inputs.dat",
            dtype=np.float32,
            mode='r',
        )

        self.targets = np.memmap(
            path + "full_targets.dat",
            dtype=np.float32,
            mode='r',
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
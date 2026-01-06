import torch
from torch.utils.data import Dataset
import numpy as np

class ChessDataset(Dataset):
    def __init__(self, npy_file_prefix, target_mean=None, target_std=None):
        self.inputs = np.load(npy_file_prefix + '_inputs.npy', mmap_mode='r')
        self.targets = np.load(npy_file_prefix + '_targets.npy', mmap_mode='r')
        self.target_mean = 0.0
        self.target_std = 1.0
    
    def __len__(self):
        return self.inputs.shape[0]
    
    def normalize(self, mean, std):
        self.target_mean = mean
        self.target_std = std
    
    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.inputs[idx].copy())
        target = torch.from_numpy(self.targets[idx].copy())
        target = (target - self.target_mean) / self.target_std  # Normalize
        return input_tensor, target
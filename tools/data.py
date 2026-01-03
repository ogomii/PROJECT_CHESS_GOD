import torch
from torch.utils.data import Dataset
import numpy as np

class ChessDataset(Dataset):
    def __init__(self, npy_file_prefix):
        self.inputs = np.load(npy_file_prefix + '_inputs.npy', mmap_mode='r')
        self.targets = np.load(npy_file_prefix + '_targets.npy', mmap_mode='r')
    
    def __len__(self):
        return self.inputs.shape[0]
    
    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.inputs[idx].copy())
        target = torch.from_numpy(self.targets[idx].copy())
        return input_tensor, target
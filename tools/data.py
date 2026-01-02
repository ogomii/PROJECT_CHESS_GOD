import torch
from torch.utils.data import Dataset
import csv
from tools.common import fen_to_tensor

class ChessDataset(Dataset):
    def __init__(self, csv_file, config):
        self.data = []
        # TODO: optimize loading for large datasets, maybe multithreading or lazy loading
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                fen, eval_str = row
                # Convert evaluation to float (assuming centipawns)
                evaluation = torch.tensor([float(eval_str)], dtype=torch.float32)
                # Precompute FEN to tensor conversion for efficiency
                input_tensor = fen_to_tensor(fen, config)
                self.data.append((input_tensor, evaluation))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_tensor, evaluation = self.data[idx]
        return input_tensor, evaluation
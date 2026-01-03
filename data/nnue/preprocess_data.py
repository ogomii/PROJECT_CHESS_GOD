import csv
import torch
import numpy as np
import sys
sys.path.append('.')
from tools.common import fen_to_tensor

def csv_to_npy(csv_file, npy_file_prefix):
    '''
    Convert CSV data file to numpy memory-mapped files for efficient loading without loading entire dataset into RAM.
    
    :param csv_file: source csv file path
    :param npy_file_prefix: prefix for npy files
    '''
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            fen, eval_str = row
            # Convert evaluation to float (assuming centipawns)
            evaluation = torch.tensor([float(eval_str)], dtype=torch.float32)
            # Precompute FEN to tensor conversion for efficiency
            input_tensor = fen_to_tensor(fen)
            data.append((input_tensor, evaluation))
    
    # Stack into tensors
    inputs = torch.stack([d[0] for d in data])
    targets = torch.stack([d[1] for d in data])
    
    # Save as numpy arrays
    np.save(npy_file_prefix + '_inputs.npy', inputs.numpy())
    np.save(npy_file_prefix + '_targets.npy', targets.numpy())

print("Creating testset...")
csv_to_npy('data/nnue/test.csv', 'data/nnue/test')
print("Creating trainset...")
csv_to_npy('data/nnue/train.csv', 'data/nnue/train')
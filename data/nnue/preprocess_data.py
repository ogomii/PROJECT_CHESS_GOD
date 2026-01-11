import csv
import torch
import numpy as np
import sys
import os
from random import randrange
sys.path.append('.')
from tools.common import fen_to_tensor

def csv_to_npy(path):
    '''
    Convert CSV data files to memory-mapped numpy arrays for efficient loading without loading entire dataset into RAM.
    
    :param path: directory path containing csv files (e.g., 'data/nnue/')
    '''
    def count_rows(file_path):
        count = 0
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for _ in reader:
                    count += 1
        except FileNotFoundError:
            print(f"Error: {file_path} not found.")
        return count
    
    # First pass: Count total rows
    total_rows = count_rows(os.path.join(path, "test.csv")) + count_rows(os.path.join(path, "train.csv"))
    if total_rows == 0:
        print("No data found. Check CSV files.")
        return
    
    # Get sample tensor shape (assume consistent across data)
    sample_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Default starting position
    sample_input = fen_to_tensor(sample_fen)
    input_shape = sample_input.shape
    target_shape = (1,)  # Assuming evaluation is scalar
    
    # Create memory-mapped arrays (disk-backed, low RAM)
    inputs_file = os.path.join(path, "full_inputs.dat")
    targets_file = os.path.join(path, "full_targets.dat")
    inputs = np.memmap(inputs_file, dtype=np.float32, mode='w+', shape=(total_rows,) + input_shape)
    targets = np.memmap(targets_file, dtype=np.float32, mode='w+', shape=(total_rows,) + target_shape)
    
    # Second pass: Fill memmap arrays
    idx = 0
    def load_and_fill(file_path):
        nonlocal idx
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    fen, eval_str = row
                    evaluation = torch.tensor([float(eval_str)], dtype=torch.float32)
                    input_tensor = fen_to_tensor(fen)
                    inputs[idx] = input_tensor.numpy()
                    targets[idx] = evaluation.numpy()
                    idx += 1
                    if idx % 10000 == 0:  # Optional progress
                        print(f"Processed {idx} rows...")
        except FileNotFoundError:
            print(f"Error: {file_path} not found.")
        except ValueError as e:
            print(f"Error parsing data in {file_path}: {e}")
    
    load_and_fill(os.path.join(path, "test.csv"))
    load_and_fill(os.path.join(path, "train.csv"))
    
    # Flush memmap to disk
    inputs.flush()
    targets.flush()
    
    # Generate split indices (low memory)
    all_indices = np.arange(total_rows)
    np.random.shuffle(all_indices)  # Shuffle for random split
    test_val_size = int(total_rows * 0.1)
    test_size = test_val_size // 2
    val_size = test_val_size - test_size
    train_size = total_rows - test_val_size
    
    test_indices = all_indices[:test_size]
    val_indices = all_indices[test_size:test_val_size]
    train_indices = all_indices[test_val_size:]
    
    # Save indices (small files)
    np.save(os.path.join(path, "test_indices.npy"), test_indices)
    np.save(os.path.join(path, "val_indices.npy"), val_indices)
    np.save(os.path.join(path, "train_indices.npy"), train_indices)
    
    print(f"Data processed: {total_rows} rows. Full data saved as memmap. Indices saved for splits.")

print("Creating train, val and test sets...")
csv_to_npy('data/nnue/')
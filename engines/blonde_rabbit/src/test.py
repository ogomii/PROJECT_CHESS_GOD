import torch
from torch.utils.data import DataLoader
from model import BlondeRabbit, Config
from tools.test_run import test_model
from tools.data import ChessDataset
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_latest_model():
    base_path = os.path.dirname(__file__)
    weights_dir = os.path.join(base_path, '../weights')
    
    # Find all model files with format: blonde_rabbit_YYYYMMDD_HHMMSS.pth
    model_files = [f for f in os.listdir(weights_dir) if f.startswith('blonde_rabbit_') and f.endswith('.pth')]
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {weights_dir}")
    
    # Sort by filename (date and time) and get the latest
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(weights_dir, latest_model)
    
    print(f"Loading model: {latest_model}")
    
    _model = BlondeRabbit(Config)
    _model.load_state_dict(torch.load(model_path, map_location='cpu'))
    _model.eval()
    return _model
_model = load_latest_model()

print("Loading testset...")
test_dataset = ChessDataset('data/nnue/', 'test')

# normalize dataset to check loss fn accuracy, denormalize predictions on inference
test_dataset.normalize(_model.target_mean, _model.target_std)

print("Creating batched dataloaders...")
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=2)

print("Starting test loop...")
test_model(_model.to(device), test_loader, device, torch.nn.MSELoss())
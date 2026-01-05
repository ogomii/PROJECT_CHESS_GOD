import chess
import chess.polyglot
from tools.common import EngineDescpriptor, fen_to_tensor
from engines.blonde_rabbit.src.model import BlondeRabbit, Config
import torch
import random as rnd
import sys
import os
import concurrent.futures

# Global model to avoid reloading
_model = None
_device = None
BATCH_SIZE = 6400

def getEngineDescriptor():
    return EngineDescpriptor("Blonde Rabbit", "1.0", "ogomi")

def _load_model():
    global _model, _device
    if _model is None:
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if getattr(sys, 'frozen', False):
            # Running in a PyInstaller bundle
            base_path = sys._MEIPASS
            model_path = os.path.join(base_path, 'engines', 'blonde_rabbit', 'blonde_rabbit.pth')
        else:
            # Running in a normal Python environment
            base_path = os.path.dirname(__file__)
            model_path = os.path.join(base_path, 'blonde_rabbit.pth')
        _model = BlondeRabbit(Config).to(_device)
        _model.load_state_dict(torch.load(model_path, map_location='cpu'))
        _model.eval()

def evaluate(fens):
    """Evaluate a list of FEN strings or a single FEN string.
    Returns a list of scores or a single score, perspective relative to side to move.
    """
    if isinstance(fens, str):
        fens = [fens]
        single = True
    else:
        single = False
    
    _load_model()
    
    global _global_fens
    _global_fens = fens
    
    tensors = []
    turns = []
    for fen in fens:
        tensor_fen = fen_to_tensor(fen)
        parts = fen.split()
        turn = chess.WHITE if parts[1] == 'w' else chess.BLACK
        tensors.append(tensor_fen)
        turns.append(turn)
    
    input_tensor = torch.stack(tensors).to(_device)
    with torch.no_grad():
        scores = []
        for batch in range(0, len(fens), BATCH_SIZE):
            batch_tensor = input_tensor[batch:batch + BATCH_SIZE]
            outputs = _model(batch_tensor)
            scores.extend(outputs.squeeze(-1).tolist())  # assuming output is (batch, 1)
            del batch_tensor, outputs  # Free GPU memory immediately
    
    # Clean up GPU memory
    del input_tensor
    if _device == 'cuda':
        torch.cuda.empty_cache()
    
    # Adjust for perspective
    adjusted_scores = []
    for score, turn in zip(scores, turns):
        adjusted_scores.append(score if turn == chess.WHITE else -score)
    
    return adjusted_scores[0] if single else adjusted_scores
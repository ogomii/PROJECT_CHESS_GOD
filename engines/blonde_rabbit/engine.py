import chess
import chess.polyglot
from tools.common import EngineDescpriptor, fen_to_tensor
from engines.blonde_rabbit.src.model import BlondeRabbit, Config
import torch
import random as rnd
import sys
import os

def getEngineDescriptor():
    return EngineDescpriptor("Blonde Rabbit", "1.0", "ogomi")

def evaluate(board: chess.Board):
    "Simple regression neural network evaluator, base on FEN"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fen = board.fen()
    tensor_fen = fen_to_tensor(fen)
    score = rnd.random()
    if getattr(sys, 'frozen', False):
        # Running in a PyInstaller bundle
        base_path = sys._MEIPASS
    else:
        # Running in a normal Python environment
        base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'engines', 'blonde_rabbit', 'blonde_rabbit.pth')
    model = BlondeRabbit(Config).to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        input_tensor = tensor_fen.unsqueeze(0).to(device)  # add batch dimension
        output = model(input_tensor)
        score = output.item()
    # perspective relative to side to move
    return score if board.turn == chess.WHITE else -score
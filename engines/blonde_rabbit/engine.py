import chess
import chess.polyglot
from tools.common import EngineDescpriptor

def getEngineDescriptor():
    return EngineDescpriptor("Blonde Rabbit", "1.0", "ogomi")

def evaluate(board: chess.Board):
    "Simple regression neural network evaluator, base on FEN"
    fen = board.fen()
    score = 0
    # TODO: Call the network here
    return score
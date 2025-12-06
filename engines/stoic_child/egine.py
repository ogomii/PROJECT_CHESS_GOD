import chess
import chess.polyglot
from tools.common import EngineDescpriptor

# ---- Simple evaluation ----

def getEngineDescriptor():
    return EngineDescpriptor("Stoic Child", "1.0", "ogomi")

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# optional simple piece-square tables (very small example)
PST = {
    chess.PAWN: [0]*64,
    chess.KNIGHT: [0]*64,
    chess.BISHOP: [0]*64,
    chess.ROOK: [0]*64,
    chess.QUEEN: [0]*64,
    chess.KING: [0]*64
}

def evaluate(board: chess.Board):
    """Material + tiny PST. Positive means advantage for side to move."""
    score = 0
    print(f"Board in fen: {board.fen()}")
    # material and pst
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = PIECE_VALUES[piece.piece_type]
            # add PST (flip for black)
            pst_val = PST[piece.piece_type][sq]
            if piece.color == chess.WHITE:
                score += val + pst_val
            else:
                score -= val + pst_val
    # perspective relative to side to move
    return score if board.turn == chess.WHITE else -score

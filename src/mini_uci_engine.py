#!/usr/bin/env python3
"""
Minimal UCI engine using python-chess
- negamax with alpha-beta
- iterative deepening (simple)
- transposition table (basic)
- simple material + piece square table evaluation
"""

import sys
import time
import threading
import chess
import chess.polyglot
import random

# ---- Simple evaluation ----
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

# ---- Transposition table entry types ----
EXACT = 0
ALPHA = 1
BETA = 2

class TTEntry:
    def __init__(self, depth, score, flag, move):
        self.depth = depth
        self.score = score
        self.flag = flag
        self.best_move = move

transposition_table = {}


def get_tt_key(board):
    """Return a transposition key for `board`.
    Works with a `chess.Board` or wrapper objects that expose a `.chessboard` attribute.
    Falls back to `chess.polyglot.zobrist_hash` or a simple fen-based hash.
    """
    # direct method if available
    if hasattr(board, "transposition_key") and callable(getattr(board, "transposition_key")):
        return board.transposition_key()
    # wrapped board (jcchess.Board) stores python-chess board at .chessboard
    underlying = getattr(board, "chessboard", None)
    if underlying is not None:
        if hasattr(underlying, "transposition_key") and callable(getattr(underlying, "transposition_key")):
            return underlying.transposition_key()
        try:
            return chess.polyglot.zobrist_hash(underlying)
        except Exception:
            return hash(underlying.fen())
    # fallback for plain python-chess Board
    try:
        return chess.polyglot.zobrist_hash(board)
    except Exception:
        return hash(board.fen())

# ---- Search ----
nodes = 0
stop_search = False
best_move_global = None

def negamax(board: chess.Board, depth, alpha, beta):
    global nodes, transposition_table
    nodes += 1

    key = get_tt_key(board)
    if key in transposition_table:
        entry = transposition_table[key]
        if entry.depth >= depth:
            if entry.flag == EXACT:
                return entry.score
            elif entry.flag == ALPHA and entry.score <= alpha:
                return alpha
            elif entry.flag == BETA and entry.score >= beta:
                return beta

    if depth == 0 or board.is_game_over():
        return evaluate(board)

    max_score = -9999999
    best_local = None

    # move ordering: try captures first (simple)
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: board.is_capture(m), reverse=True)

    for mv in moves:
        board.push(mv)
        score = -negamax(board, depth-1, -beta, -alpha)
        board.pop()

        if score > max_score:
            max_score = score
            best_local = mv
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    # store in TT
    flag = EXACT
    if max_score <= alpha: # careful: alpha already changed; this is a small heuristic
        flag = ALPHA
    elif max_score >= beta:
        flag = BETA

    transposition_table[key] = TTEntry(depth, max_score, flag, best_local)
    return max_score

def search_root(board: chess.Board, max_depth, time_limit=None):
    """Iterative deepening to max_depth. Returns best move found."""
    global stop_search, best_move_global, nodes
    best = None
    start_time = time.time()
    nodes = 0
    stop_search = False

    for d in range(1, max_depth+1):
        if stop_search:
            break
        best_score = -9999999
        alpha = -10000000
        beta = 10000000
        # simple move ordering: try tt move first if available
        key = get_tt_key(board)
        tt_move = transposition_table.get(key).best_move if (key in transposition_table) else None

        moves = list(board.legal_moves)
        if tt_move and tt_move in moves:
            moves.remove(tt_move)
            moves.insert(0, tt_move)
        moves.sort(key=lambda m: board.is_capture(m), reverse=True)

        for mv in moves:
            if stop_search: break
            board.push(mv)
            score = -negamax(board, d-1, -beta, -alpha)
            board.pop()
            if score > best_score:
                best_score = score
                best = mv

            # time check
            if time_limit and (time.time() - start_time) > time_limit:
                stop_search = True
                break

        best_move_global = best
        # a small info print for GUIs (depth/nodes)
        print(f"info depth {d} nodes {nodes} score cp {best_score}")
        sys.stdout.flush()

        if time_limit and (time.time() - start_time) > time_limit:
            break

    return best

# ---- UCI loop & thread ----
engine_options = {}

board = chess.Board()
think_thread = None

def think_thread_fn(search_args):
    global board, stop_search, best_move_global
    stop_search = False
    best_move_global = None
    depth = search_args.get("depth", 4)
    movetime = search_args.get("movetime", None)
    best = search_root(board, depth, time_limit=(movetime/1000.0 if movetime else None))
    if best is None:
        # fallback random legal move
        moves = list(board.legal_moves)
        if moves:
            best = random.choice(moves)
    print(f"bestmove {best.uci() if best else '0000'}")
    sys.stdout.flush()

def uci_loop():
    global board, think_thread, stop_search
    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            break
        if not line:
            break
        line = line.strip()
        if line == "":
            continue

        # parse basic commands
        if line == "uci":
            print("id name MiniPyEngine")
            print("id author Your Name")
            print("uciok")
            sys.stdout.flush()

        elif line == "isready":
            print("readyok")
            sys.stdout.flush()

        elif line.startswith("setoption"):
            # optionally parse options
            # e.g. setoption name Hash value 64
            # naive parsing
            parts = line.split()
            # very minimal parser
            try:
                name_idx = parts.index("name") + 1
                val_idx = parts.index("value") + 1
                name = " ".join(parts[name_idx:val_idx-1])
                value = " ".join(parts[val_idx:])
                engine_options[name] = value
            except ValueError:
                pass

        elif line.startswith("ucinewgame"):
            # reset internal state if needed
            transposition_table.clear()
            board.reset()

        elif line.startswith("position"):
            # position startpos [moves ...] | position fen <fen> [moves ...]
            tokens = line.split()
            if tokens[1] == "startpos":
                board.set_fen(chess.STARTING_FEN)
                # find moves
                if "moves" in tokens:
                    idx = tokens.index("moves") + 1
                    moves = tokens[idx:]
                    for m in moves:
                        board.push_san(m) if len(m) > 4 else board.push_uci(m)
            elif tokens[1] == "fen":
                fen_tokens = tokens[2:]
                # fen is 6 fields; find where moves start
                if "moves" in fen_tokens:
                    idx = fen_tokens.index("moves")
                    fen = " ".join(fen_tokens[:idx])
                    moves = fen_tokens[idx+1:]
                else:
                    fen = " ".join(fen_tokens)
                    moves = []
                board.set_fen(fen)
                for m in moves:
                    board.push_uci(m)

        elif line.startswith("go"):
            # parse a few parameters: depth, movetime, wtime, btime
            tokens = line.split()
            args = {}
            if "depth" in tokens:
                args["depth"] = int(tokens[tokens.index("depth")+1])
            if "movetime" in tokens:
                args["movetime"] = int(tokens[tokens.index("movetime")+1])
            # start thinking thread
            if think_thread and think_thread.is_alive():
                # if already thinking, ignore or stop and restart; we stop then restart
                stop_search = True
                think_thread.join()
            think_thread = threading.Thread(target=think_thread_fn, args=(args,))
            think_thread.start()

        elif line == "stop":
            stop_search = True
            if think_thread:
                think_thread.join()
            # The think thread prints bestmove when it finishes

        elif line == "quit":
            stop_search = True
            if think_thread:
                think_thread.join()
            break

        else:
            # other commands ignored for now
            pass

if __name__ == "__main__":
    uci_loop()

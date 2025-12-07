import os
import sys

# Ensure project root is on sys.path so package imports work when running this file directly
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engines.blonde_rabbit.src.model import *
try:
    import pytest
except Exception:
    pytest = None


def test_model_initialization():
    config = Config()
    model_instance = BlondeRabbit(config)
    assert model_instance is not None
    assert isinstance(model_instance, BlondeRabbit)


def test_fen_to_tensor():
    config = Config()
    model_instance = BlondeRabbit(config)
    fen = "r1bqkbnr/p1pppppp/n7/1P6/8/8/1PPPPPPP/RNBQKBNR"
    tensor = model_instance.fen_to_tensor(fen)
    assert tensor.shape == (config.squares, config.input_classes)
    # Check that some known positions are correctly encoded
    assert tensor[0][config.piece_map['r']] == 1
    assert tensor[1][config.piece_map["-"]] == 1
    assert tensor[8][config.piece_map['p']] == 1
    assert tensor[25][config.piece_map['P']] == 1
    # Check a range of squares are empty
    for i in range(32, 40):
        assert tensor[i][config.piece_map["-"]] == 1, f"Index {i}: expected empty square, got {tensor[i]}"

# for debugger
if __name__ == "__main__":
    test_model_initialization()
    test_fen_to_tensor()
    print("All tests passed.")
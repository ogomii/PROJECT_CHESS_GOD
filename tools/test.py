from tools.common import fen_to_tensor, piece_map
from engines.blonde_rabbit.src.model import Config

def test_fen_to_tensor():
    config = Config()
    fen = "r1bqkbnr/p1pppppp/n7/1P6/8/8/1PPPPPPP/RNBQKBNR w - - 4 14"
    tensor = fen_to_tensor(fen)
    assert tensor.shape == (config.squares, config.input_classes)
    # Check that some known positions are correctly encoded
    assert tensor[0][piece_map['r']] == 1
    assert tensor[1][piece_map["-"]] == 1
    assert tensor[8][piece_map['p']] == 1
    assert tensor[25][piece_map['P']] == 1
    # Check a range of squares are empty
    for i in range(32, 40):
        assert tensor[i][piece_map["-"]] == 1, f"Index {i}: expected empty square, got {tensor[i]}"
import string
import torch

class EngineDescpriptor:
    def __init__(self, name, version, autor):
        self.name: string = name
        self.version: string = version
        self.autor: string = autor


def fen_to_tensor(fen, config=EngineDescpriptor):
    '''
    Convert FEN string to tensor representation.
    example FEN: r1b2rk1/ppp2pbp/3q1np1/n3p1B1/2B5/1Q3N2/PP1N1PPP/3R1RK1 w - - 4 14 
    '''
    tensor = torch.zeros((config.squares, config.input_classes))
    tensor_idx = 0
    for char in fen:
        if char in ['/',' ']:
            if char in [' ']:
                # end of FEN board part
                break
            continue
        elif char.isdigit():
            for empty_idx in range(int(char)):
                tensor[tensor_idx + empty_idx][config.piece_map["-"]] = 1
            tensor_idx += int(char)  # adjust width index
        else:
            piece_type = config.piece_map.get(char)
            if piece_type is not None:
                tensor[tensor_idx][piece_type] = 1
                tensor_idx += 1
            else:
                raise ValueError(f"Invalid character in FEN: {char}")
    return tensor
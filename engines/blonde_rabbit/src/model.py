import torch
from torch import nn

class Config:
    squares_height = 8
    squares_width = 8
    squares = 64
    input_classes = 13 # 12 pieces + empty
    piece_map = { # class 0 reserved for empty square
        "-": 0,
        'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
        'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12
    }
    d_model = 256  # example output size

class BlondeRabbit(torch.nn.Module):
    '''
    each field represented by 12 classes (12 binary inputs as there is 6 white and 6 black pieces)
    input matrix (64, 12)
    each input matrix (12,) will be processed by a subnet into an embedded space of some dim d resulting in (64,d)
    (64,d), as it has a small outer dimention can easily be processed by a transformer network
    '''

    def __init__(self, config):
        super(BlondeRabbit, self).__init__()
        self.config = config
        # embedding layer to convert (13,) to (d_model,)
        self.embed = nn.Linear(self.config.input_classes, self.config.d_model)

    def fen_to_tensor(self, fen):
        '''
        Convert FEN string to tensor representation.
        example FEN: r1bqkbnr/p1pppppp/n7/1P6/8/8/1PPPPPPP/RNBQKBNR
        '''
        tensor = torch.zeros((self.config.squares, self.config.input_classes))
        tensor_idx = 0
        for char in fen:
            if char == "/":
                continue
            elif char.isdigit():
                for empty_idx in range(int(char)):
                    tensor[tensor_idx + empty_idx][self.config.piece_map["-"]] = 1
                tensor_idx += int(char)  # adjust width index
            else:
                piece_type = self.config.piece_map.get(char)
                if piece_type is not None:
                    tensor[tensor_idx][piece_type] = 1
                    tensor_idx += 1
                else:
                    raise ValueError(f"Invalid character in FEN: {char}")
        return tensor

    def forward(self, x):
        N, T, _ = x.shape
        # make fe_to_tensor for each in batch
        x = torch.stack([self.fen_to_tensor(fen) for fen in x])
        x = self.embed(x) # (N, T, d_model)
        return x
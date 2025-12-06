import torch

class Config:
    input_classes = 13 # 12 pieces + empty
    squares_height = 8
    squares_width = 8
    squares = 64
    piece_map = { # class 0 reserved for empty square
        'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
        'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12
    }

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
        self.layer = torch.nn.Linear(config.input_size, config.output_size)

    def fen_to_tensor(self, fen):
        '''
        Convert FEN string to tensor representation.
        example FEN: r1bqkbnr/p1pppppp/n7/1P6/8/8/1PPPPPPP/RNBQKBNR
        '''
        tensor = torch.zeros((self.config.squares, self.config.input_classes))
        fen_idx = 0
        for h in range(self.config.squares_height):
            for w in range(self.config.squares_width):
                idx = h * self.config.squares_width + w
                char = fen[fen_idx]  # considering '/' in FEN
                fen_idx += 1
                if char == "/":
                    continue
                elif char.isdigit():
                    for empty_idx in range(int(char)):
                        tensor[idx + empty_idx] = torch.zeros(self.config.input_classes)
                    w += int(char) - 1  # adjust width index
                else:
                    piece_type = self.config.piece_map.get(char)
                    if piece_type is not None:
                        tensor[idx][piece_type] = 1
        return tensor

    def forward(self, x):
        x = self.fen_to_tensor(x)
        return self.layer(x)
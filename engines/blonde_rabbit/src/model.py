import torch
from torch import nn

class Config:
    squares_height = 8
    squares_width = 8
    squares = 64
    input_classes = 13 # 12 pieces + empty
    d_model = 256  # example output size

class BlondeRabbit(torch.nn.Module):
    '''
    each field represented by 13 classes (13 binary inputs as there is 6 white and 6 black pieces + none)
    input matrix (64, 13)
    each input matrix (13,) will be processed by a subnet into an embedded space of some dim d resulting in (64,d)
    (64,d), as it has a small outer dimension can easily be processed by a transformer network
    '''

    def __init__(self, config):
        super(BlondeRabbit, self).__init__()
        self.config = config
        # embedding layer to convert (13,) to (d_model,)
        self.embed = nn.Linear(self.config.input_classes, self.config.d_model)
        self.out = nn.Linear(self.config.d_model * self.config.squares, 1) # regression output

    def forward(self, x):
        N, T, _ = x.shape
        x = self.embed(x) # (N, T, d_model)
        x = torch.flatten(x, start_dim=1) # (N, T*d_model)
        x = self.out(x)**3 # (N, 1)
        return x
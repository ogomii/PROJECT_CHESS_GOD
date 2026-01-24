import torch
from torch import nn

class Config:
    squares_height = 8
    squares_width = 8
    squares = 64
    input_classes = 13 # 12 pieces + empty
    d_model = 128  # example output size
    hidden_layers = 2
    dropout = 0.0


class MLPLayer(nn.Module):
    def __init__(self, config):
        super(MLPLayer, self).__init__()
        self.l1 = nn.Linear(config.d_model, 2*config.d_model)
        self.ln1 = nn.LayerNorm(2*config.d_model)
        self.relu = nn.GELU()
        self.drop = nn.Dropout(config.dropout)
        self.l2 = nn.Linear(2*config.d_model, config.d_model)
    
    def forward(self, x) -> torch.Tensor:
        x = self.l1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.drop(x)
        out = self.l2(x)
        return out


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.layer = MLPLayer(config)
    
    def forward(self, x) -> torch.Tensor:
        return self.layer(x)


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.layers = nn.Sequential(*[Block(self.config) for _ in range(self.config.hidden_layers)])
    
    def forward(self, x) -> torch.Tensor:
        out = self.layers(x)
        return out


class BlondeRabbit(nn.Module):
    '''
    each field represented by 13 classes (13 binary inputs as there is 6 white and 6 black pieces + none)
    input matrix (64, 13)
    each input matrix (13,) will be processed by a subnet into an embedded space of some dim d resulting in (64,d)
    (64,d), as it has a small outer dimension can easily be processed by a transformer network
    '''

    def __init__(self, config, target_mean=None, target_std=None):
        super(BlondeRabbit, self).__init__()
        self.config = config
        # embedding layer to convert (13,) to (d_model,)
        self.embed = nn.Linear(self.config.input_classes, self.config.d_model)
        self.mlp = MLP(self.config)
        self.out = nn.Linear(self.config.d_model * self.config.squares, 1) # regression output

        if target_mean is not None and target_std is not None:
            self.register_buffer('target_mean', torch.tensor(target_mean, dtype=torch.float32))
            self.register_buffer('target_std', torch.tensor(target_std, dtype=torch.float32))
        else:
            self.register_buffer('target_mean', torch.tensor(0.0))  # Default/placeholder
            self.register_buffer('target_std', torch.tensor(1.0))

    def forward(self, x):
        N, S, _ = x.shape
        x = self.embed(x) # (N, S, d_model)

        x = x + self.mlp(x)

        x = torch.flatten(x, start_dim=1) # (N, S*d_model)
        # linear output is later denormalized using dataset norm and std
        out = self.out(x) # (N, 1)
        return out
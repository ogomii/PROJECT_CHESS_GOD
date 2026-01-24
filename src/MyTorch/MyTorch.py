'''
Implementation of torch layers, not really used in the model due to the lack 
of GPU optimizations that torch layers benefit from (implementations here are
much much slower)
'''
import torch
from torch import nn

class MyLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MyLinear, self).__init__()
        self.layer = nn.Parameter(torch.empty(in_channels, out_channels))
        nn.init.kaiming_uniform_(self.layer, a=0, mode='fan_in', nonlinearity='linear')
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.uniform_(self.bias, -1/(in_channels)**(1/2), 1/(in_channels)**(1/2))
    
    def __repr__(self):
        return f"MyLinear({self.layer.data.shape[0]}, {self.layer.data.shape[1]})"

    def forward(self, x):
        # (_, in_channels) @ (in_channels, out_channels) + (out_channels,) -> (_, out_channels)
        return x @ self.layer + self.bias 
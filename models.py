import torch
from torch import nn

class MaskedConvolution(nn.Conv2d):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        x 
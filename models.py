import torch
from torch import nn
from torchsummary import summary
import numpy as np

class BasicAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), # 64 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 4

        )

        self.decoder = nn.Sequential(
            

            nn.ConvTranspose2d(3, 3, 2, 2),
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

class MaskedConvolution(nn.Conv2d):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        pass


if __name__ == "__main__":
    ae = BasicAutoencoder()
    tensor = torch.ones((3, 4, 4))
    #tensor = ae.encoder(tensor)
    print(tensor.shape)
    tensor = ae.decoder(tensor)
    print(tensor.shape)
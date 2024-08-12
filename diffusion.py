import torch
from torch import nn
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import dataset
from torch.utils.data import Dataset, DataLoader
import sys

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, leak=True):
        super().__init__()

        if leak:

            self.block = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(hidden_channels, hidden_channels, 2, 2),
                nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
                nn.LeakyReLU(),
            )
        
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(hidden_channels, hidden_channels, 2, 2),
                nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
            )

    def forward(self, x):
        return self.block(x)

class UNetNoisePredictor(nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier
        # encoder

        # image starts as 64x64
        self.encode1 = EncoderBlock(3, 16*self.multiplier, 16*self.multiplier) # 32
        self.encode2 = EncoderBlock(16*self.multiplier, 32*self.multiplier, 64*self.multiplier) # 16 
        self.encode3 = EncoderBlock(64*self.multiplier, 64*self.multiplier, 64*self.multiplier) # 8

        # decoder
        self.decode1 = DecoderBlock(64*self.multiplier, 64*self.multiplier, 64*self.multiplier) # 16
        self.decode2 = DecoderBlock(128*self.multiplier, 64*self.multiplier, 48*self.multiplier)
        self.decode3 = DecoderBlock(64*self.multiplier, 32*self.multiplier, 3, False)

    def forward(self, x):
        encoded1 = self.encode1(x)
        encoded2 = self.encode2(encoded1)
        encoded3 = self.encode3(encoded2)

        decoded1 = self.decode1(encoded3)
        decoded1 = torch.cat((decoded1, encoded2), dim=1)

        decoded2 = self.decode2(decoded1)
        decoded2 = torch.cat((decoded2, encoded1), dim=1)

        decoded3 = self.decode3(decoded2)
        return torch.tanh(decoded3)
    

def sinusoidal_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = 10000 ** (2 * torch.arange(half_dim) / (half_dim - 1))
    emb = timesteps / emb
    emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)

    return timesteps / emb
    
def train(dataloader, denoiser, epochs, optimizer, device, save_interval=25):
    losses = []
    mse = nn.MSELoss(reduction="sum")

    for epoch in range(epochs):

        if epoch % save_interval == 0 and epoch != 0:
            torch.save(denoiser.state_dict(), "./trained/diffusion.pth")

        for batch_index, batch in enumerate(dataloader):

            corrupted, noise, timestep = batch
            corrupted = corrupted.to(device)
            noise = noise.to(device)

            input_images = batch[0].to(device)

            predicted_noise = denoiser(input_images)

            optimizer.zero_grad()

            loss = mse(predicted_noise, noise)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if len(losses) > 100:
                losses.pop(0)
            avg_loss = sum(losses) / len(losses)

            print(f"{epoch}, {batch_index}, {avg_loss:.6f}, {loss.item():.6f}", end="\r")

            #time.sleep(1.0)


if __name__ == "__main__":
    command = sys.argv[1]
    DEVICE = sys.argv[2]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data = dataset.Diffusion(10)
    loader = DataLoader(data, batch_size=16, shuffle=True)

    denoiser = UNetNoisePredictor(5)
    print(summary(denoiser, (3, 64, 64), device='cpu'))
    #denoiser.load_state_dict(torch.load("./trained/diffusion.pth"))

    denoiser = denoiser.to(DEVICE)

    

    if command == "generate":
        pass
        #show_image(autoencoder, 8)
    elif command == "train":
        optimizer = torch.optim.Adam(denoiser.parameters(), lr=0.00002)
        train(loader, denoiser, 10000000, optimizer, DEVICE, save_interval=10)
    elif command == "fid":
        pass
        #print("Frechet Inception Distance Score: ", get_fid(autoencoder, data, 256))
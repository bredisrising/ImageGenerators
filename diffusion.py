import torch
from torch import nn
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import dataset
from torch.utils.data import Dataset, DataLoader
import sys
from PIL import Image
from torchvision.transforms import ToPILImage
import time

def compute_alpha_t(t, max_timesteps):
    if t == max_timesteps: 
        return torch.tensor(0, dtype=torch.float32)

    cumulative = 1
    for i in range(1, t+1):
        cumulative *= torch.cos(torch.pi / 2 * torch.tensor(i) / max_timesteps)
    return cumulative

def sinusoidal_embedding(timesteps, embedding_dim):
    #tensor_timesteps = torch.tensor(timesteps, dtype=torch.float32).unsqueeze(1)
    half_dim = embedding_dim // 2
    emb = 10000 ** (2 * torch.arange(half_dim) / (half_dim - 1))
    emb = emb.to("cuda")
    #print(tensor_timesteps, emb)
    emb = timesteps / emb
    emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)

    return emb

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
        
        self.linear = nn.Linear(16, 64*64)

        # encoder

        # image starts as 64x64
        self.encode1 = EncoderBlock(4, 16*self.multiplier, 16*self.multiplier) # 32
        self.encode2 = EncoderBlock(16*self.multiplier, 32*self.multiplier, 64*self.multiplier) # 16 
        self.encode3 = EncoderBlock(64*self.multiplier, 64*self.multiplier, 64*self.multiplier) # 8

        # decoder
        self.decode1 = DecoderBlock(64*self.multiplier, 64*self.multiplier, 64*self.multiplier) # 16
        self.decode2 = DecoderBlock(128*self.multiplier, 64*self.multiplier, 48*self.multiplier)
        self.decode3 = DecoderBlock(64*self.multiplier, 32*self.multiplier, 3, False)

    def forward(self, x, timesteps):

        embeddings = sinusoidal_embedding(timesteps, 16)
        embeddings = self.linear(embeddings).view(-1, 1, 64, 64)
        x = torch.cat((x, embeddings), dim=1)

        encoded1 = self.encode1(x)
        encoded2 = self.encode2(encoded1)
        encoded3 = self.encode3(encoded2)

        decoded1 = self.decode1(encoded3)
        decoded1 = torch.cat((decoded1, encoded2), dim=1)

        decoded2 = self.decode2(decoded1)
        decoded2 = torch.cat((decoded2, encoded1), dim=1)

        decoded3 = self.decode3(decoded2)
        return torch.tanh(decoded3)
    


    
def train(dataloader, denoiser, epochs, optimizer, lr_scheduler, device, save_interval=10):
    losses = []
    mse = nn.MSELoss(reduction="mean")
    topilimage = ToPILImage()

    for epoch in range(epochs):

        if epoch % save_interval == 0 and epoch != 0:
            torch.save(denoiser.state_dict(), "./trained/diffusion.pth")

        for batch_index, batch in enumerate(dataloader):

            corrupted, noise, timesteps = batch
            corrupted = corrupted.to(device)
            noise = noise.to(device)
            timesteps = timesteps.to(device)

            input_images = batch[0].to(device)

            predicted_noise = denoiser(input_images, timesteps.unsqueeze(-1))
            #print(predicted_noise[0])

            optimizer.zero_grad()

            loss = mse(predicted_noise, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())
            if len(losses) > 100:
                losses.pop(0)
            avg_loss = sum(losses) / len(losses)

            current_lr = optimizer.param_groups[0]['lr']

            print(f"{epoch}, {batch_index}, {avg_loss:.6f}, {loss.item():.6f}, {current_lr}", end="\r")

            #time.sleep(1.0)

        lr_scheduler.step(avg_loss)

        image = eval(denoiser, False)
        #print(image.shape)
        image = topilimage(image.squeeze())
        image.save("./results/diffused.png")
        #time.sleep(2)

def eval(denoiser, show=False):
    image = torch.randn((1, 3, 64, 64)).to('cuda')
    for i in range(49, 0, -1):
        timestep = torch.tensor(i).to('cuda')
        with torch.no_grad():
            predicted_noise = denoiser(image, timestep)
            #print(predicted_noise)
        
        predicted_noise = torch.clamp(predicted_noise, -1.0, 1.0)
        at = compute_alpha_t(timestep, 50)

        bt = 1 - torch.cos(torch.pi / 2 * timestep / 50)
        #at = torch.clamp(at, 1e-5, 1.0 - 1e-5)
        #print(at)

        #scaled_noise = predicted_noise * bt
        image = (image - bt/torch.sqrt(1-at)*predicted_noise) / torch.sqrt(1-bt) + torch.randn_like(image)*bt
        image = torch.clamp(image, -1.0, 1.0)
        print(image)
        if show:
            plt.imshow(image.to('cpu').squeeze().permute(1,2,0))
            plt.show()

    #print(image)

    image = image / 2.0 + 0.5
    image = torch.clamp(image, 0.0, 1.0)

    #print(image)

    #print(image, image.dtype)
    return image.to('cpu')

if __name__ == "__main__":
    command = sys.argv[1]
    DEVICE = sys.argv[2]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data = dataset.Diffusion(50)
    loader = DataLoader(data, batch_size=16, shuffle=True)

    denoiser = UNetNoisePredictor(7)
    #print(summary(denoiser, (3, 64, 64), device='cpu'))
    denoiser.load_state_dict(torch.load("./trained/diffusion.pth"))

    denoiser = denoiser.to(DEVICE)

    

    if command == "generate":
        pass
        #show_image(autoencoder, 8)
    elif command == "train":
        optimizer = torch.optim.Adam(denoiser.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        train(loader, denoiser, 10000000, optimizer, scheduler, DEVICE, save_interval=10)
    elif command == "fid":
        pass
        #print("Frechet Inception Distance Score: ", get_fid(autoencoder, data, 256))
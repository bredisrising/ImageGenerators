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

TRAIN_TIMESTEPS = 75
INFERENCE_TIMESTEPS = 75

def cosine_schedule(timestep, max_timesteps):
    cum_at_timestep = torch.cos(timestep / max_timesteps * torch.pi / 2).pow(2)
    cum_at_prev_timestep = torch.cos((timestep-1) / max_timesteps * torch.pi / 2).pow(2)
    bt = 1 - (cum_at_timestep / cum_at_prev_timestep)

    return cum_at_timestep, bt


def linear_beta_schedule(max_timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, max_timesteps)

def precompute_terms():
    alpha_t = 1 - linear_beta_schedule(TRAIN_TIMESTEPS)
    cumulative_alpha_t = torch.cumprod(alpha_t, dim=0)


# def sinusoidal_embedding(timesteps, embedding_dim):
#     #tensor_timesteps = torch.tensor(timesteps, dtype=torch.float32).unsqueeze(1)
#     half_dim = embedding_dim // 2
#     emb = 10000 ** (2 * torch.arange(half_dim) / (half_dim - 1))
#     emb = emb.to('cuda')
#     #print(tensor_timesteps, emb)
#     emb = timesteps / emb
#     emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)

#     return emb

def sinusoidal_embedding(timesteps, embedding_dim):
    pe = torch.zeros((timesteps.shape[0], embedding_dim)).to('cuda')
    #print(pe.shape)
    indices = torch.arange(0, embedding_dim//2).to('cuda')
    pe[:, 0::2] = torch.sin((timesteps)/torch.tensor(10000.).pow(2*indices/embedding_dim))
    pe[:, 1::2] = torch.cos((timesteps)/torch.tensor(10000.).pow(2*indices/embedding_dim))
    return pe

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear_embedding = nn.Linear(32, out_channels)
        self.relu = nn.ReLU()

        self.one = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.two = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(),
        )

    def forward(self, x, embeddings):
        embeddings = self.linear_embedding(embeddings)
        embeddings = self.relu(embeddings)
        embeddings = embeddings[(..., ) + (None, ) * 2]
        x = self.one(x)
        x = x + embeddings.squeeze(dim=1)
        #print(x.shape)
        x = self.two(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, leak=True):
        super().__init__()

        self.linear_embedding = nn.Linear(32, out_channels)
        self.relu = nn.ReLU()

        if leak:
            self.one = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(out_channels, out_channels, 2, 2),
            )
            self.two = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                nn.LeakyReLU(),
            )
        
        else:
            self.one = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(out_channels, out_channels, 2, 2),
            )
            self.two = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            )

    def forward(self, x, embeddings):
        embeddings = self.linear_embedding(embeddings)
        embeddings = self.relu(embeddings)
        embeddings = embeddings[(..., ) + (None, ) * 2]
        x = self.one(x)
        x = x + embeddings.squeeze(dim=1)
        #print(x.shape)
        x = self.two(x)
        return x

class UNetNoisePredictor(nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier

        # encoder
        # image starts as 64x64
        self.encode1 = EncoderBlock(3, 16*self.multiplier) # 32
        self.encode2 = EncoderBlock(16*self.multiplier, 32*self.multiplier) # 16 
        self.encode3 = EncoderBlock(32*self.multiplier, 64*self.multiplier) # 8
        self.encode4 = EncoderBlock(64*self.multiplier, 64*self.multiplier) # 4

        # decoder
        self.decode1 = DecoderBlock(64*self.multiplier, 64*self.multiplier)
        self.decode2 = DecoderBlock(128*self.multiplier, 64*self.multiplier) 
        self.decode3 = DecoderBlock(96*self.multiplier, 32*self.multiplier)
        self.decode4 = DecoderBlock(48*self.multiplier, 64, False)


        # need this because setting out channel to three on decoder block will cause very few channels in the downsample which i think is bad
        # i think - i actually have no idea why this is the sole thing thats making this work better
        self.output = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x, timesteps=torch.tensor([[1.0], [1.0]])):
        embeddings = sinusoidal_embedding(timesteps, 32)
        
        encoded1 = self.encode1(x, embeddings)
        encoded2 = self.encode2(encoded1, embeddings)
        encoded3 = self.encode3(encoded2, embeddings)
        encoded4 = self.encode4(encoded3, embeddings)

        decoded1 = self.decode1(encoded4, embeddings)
        decoded1 = torch.cat((decoded1, encoded3), dim=1)

        decoded2 = self.decode2(decoded1, embeddings)
        decoded2 = torch.cat((decoded2, encoded2), dim=1)

        decoded3 = self.decode3(decoded2, embeddings)
        decoded3 = torch.cat((decoded3, encoded1), dim=1)

        decoded4 = self.decode4(decoded3, embeddings)
        #return self.output(decoded4)
        return self.output(decoded4)
        #return torch.tanh(decoded4)
    


    
def train(dataloader, denoiser, epochs, optimizer, lr_scheduler, device, save_interval=10):
    losses = []
    #criterion = nn.MSELoss(reduction="mean")
    criterion = nn.L1Loss()
    topilimage = ToPILImage()

    for epoch in range(epochs):

        if epoch % save_interval == 0 and epoch != 0:
            torch.save(denoiser.state_dict(), "./trained/diffusion.pth")

        for batch_index, batch in enumerate(dataloader):

            corrupted, noise, timesteps = batch
            corrupted = corrupted.to(device)
            noise = noise.to(device)
            timesteps = timesteps.to(device)

            #print(timesteps.shape)

            input_images = batch[0].to(device)
            # print(input_images[0].shape)

            predicted_noise = denoiser(input_images, timesteps)
            #one = predicted_noise[0]
            # print(one.shape)
            #print(one.max().item(), one.min().item(), one.mean().item(), one.var().item())

            # if epoch % 50 == 0:
            #     plt.imshow(predicted_noise[0].cpu().detach().permute(1,2,0))
            #     plt.show()

            optimizer.zero_grad()

            loss = criterion(predicted_noise, noise)

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())
            if len(losses) > 100:
                losses.pop(0)
            avg_loss = sum(losses) / len(losses)

            current_lr = optimizer.param_groups[0]['lr']

            print(f"{epoch}, {batch_index}, {avg_loss:.6f}, {loss.item():.6f}, {current_lr}", end="\r")

        #lr_scheduler.step(avg_loss)

        if epoch % 50 == 0:
            image = eval(denoiser, False)
            #print(image.shape)
            image = topilimage(image.squeeze())
            image.save("./results/diffused.png")
        #time.sleep(2)

def eval(denoiser, show=False):
    topilimage = ToPILImage()
    image = torch.randn((1, 3, 32, 32)).to('cuda')
    inference_timesteps_tensor = torch.tensor(INFERENCE_TIMESTEPS, dtype=torch.float32)
    for i in range(INFERENCE_TIMESTEPS-1, 0, -1):
        timestep = torch.tensor(i).to('cuda').unsqueeze(0)
        with torch.no_grad():
            predicted_noise = denoiser(image, timestep)
            #print(predicted_noise)
        
        #predicted_noise = torch.clamp(predicted_noise, -1.0, 1.0)
        cum_at, bt = cosine_schedule(timestep, inference_timesteps_tensor)

        image = (image - bt/torch.sqrt(1-cum_at)*predicted_noise) / torch.sqrt(1-bt) 
        if i > 1:
            image = image + torch.randn_like(image)*torch.sqrt(bt)
        #image = torch.clamp(image, -1.0, 1.0)
        pilimage = topilimage(image.clamp(-1.0,1.0).to("cpu").squeeze() / 2.0 + .5)
        pilimage.save(f"./results/{i}timestep_diffused.png")
        #print(bt, torch.sqrt(bt))
        if show:
            plt.imshow(image.to('cpu').squeeze().permute(1,2,0))
            plt.show()

    #print(image)
    image = torch.clamp(image, -1.0, 1.0)
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

    data = dataset.Diffusion32(TRAIN_TIMESTEPS)
    loader = DataLoader(data, batch_size=16)

    denoiser = UNetNoisePredictor(12)
    #print(summary(denoiser, (3, 64, 64), device='cpu'))
    denoiser.load_state_dict(torch.load("./trained/diffusion.pth"))

    denoiser = denoiser.to(DEVICE)

    if command == "generate":
        pass
        #show_image(autoencoder, 8)
    elif command == "train":
        optimizer = torch.optim.Adam(denoiser.parameters(), lr=0.00002)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        train(loader, denoiser, 10000000, optimizer, scheduler, DEVICE, save_interval=20)
    elif command == "fid":
        pass
        #print("Frechet Inception Distance Score: ", get_fid(autoencoder, data, 256))
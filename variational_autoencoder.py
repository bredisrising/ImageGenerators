import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import dataset
import matplotlib.pyplot as plt
import time
from torchsummary import summary

LATENT_DIM = 32

class VariationalAutoencoder(nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32*multiplier, 3, 1, 1),  # 64
            nn.LeakyReLU(),
            nn.Conv2d(32*multiplier, 32*multiplier, 3, 1, 1),  
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), # 32
            nn.Conv2d(32*multiplier, 64*multiplier, 3, 1, 1),  
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), # 16
            nn.Conv2d(64*multiplier, 64*multiplier, 3, 1, 1),  
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), # 8
            nn.Conv2d(64*multiplier, 32*multiplier, 3, 1, 1),  
            nn.LeakyReLU(),
            nn.Conv2d(32*multiplier, 32*multiplier, 3, 1, 1),  # 8
            nn.Flatten(),
            nn.Linear(32*multiplier*8*8, 2*LATENT_DIM),
        )

        self.linear_before_decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 32*multiplier*8*8),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(32*multiplier, 32*multiplier, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32*multiplier, 32*multiplier, 3, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32*multiplier, 64*multiplier, 2, 2), # 16
            nn.Conv2d(64*multiplier, 64*multiplier, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64*multiplier, 64*multiplier, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64*multiplier, 64*multiplier, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64*multiplier, 64*multiplier, 3, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64*multiplier, 64*multiplier, 2, 2), # 32
            nn.Conv2d(64*multiplier, 64*multiplier, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64*multiplier, 32*multiplier, 3, 1, 1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32*multiplier, 32*multiplier, 2, 2), # 64
            nn.Conv2d(32*multiplier, 16*multiplier, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(16*multiplier, 16*multiplier, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(16*multiplier, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

        for sm in self.modules():
            if isinstance(sm, nn.Conv2d) or isinstance(sm, nn.Linear):
                torch.nn.init.xavier_uniform(sm.weight, gain=1.6)

    def forward(self, x):
        x = self.encoder(x)
        #print(x[0])
        mus = x[:, :LATENT_DIM]
        sigmas = x[:, LATENT_DIM:] # this is log variance
        # log(std**2) = 2 * log(std)

        #print(mus, sigmas)

        stddevs = torch.exp(0.5 * sigmas)

        epsilon = torch.randn_like(stddevs)

        z = mus + stddevs * epsilon

        x = self.linear_before_decoder(z)
        x = x.view(x.size(0), 32*self.multiplier, 8, 8)
        return self.decoder(x), mus, sigmas


def kl_divergence(mean, log_var):
    return -0.5 * torch.sum(1 + log_var - mean.pow(2) - torch.exp(log_var))

def ELBO(input_images, generated_image, mus, sigmas):
    return (generated_image - input_images).pow(2).mean(),  kl_divergence(mus, sigmas)

def train(dataloader, autoencoder, epochs, optimizer, device, save_interval=25):
    losses = []

    for epoch in range(epochs):

        if epoch % save_interval == 0 and epoch != 0:
            torch.save(autoencoder.state_dict(), "./trained/variational_autoencoder.pth")

        for batch_index, batch in enumerate(dataloader):
            input_images = batch[0].to(device)

            generated_outputs, means, log_vars = autoencoder(input_images)

            optimizer.zero_grad()
            #loss = mse(generated_outputs, input_images  )
            mse, kl = ELBO(input_images, generated_outputs, means, log_vars)
            
            loss = mse + kl * 0.01

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if len(losses) > 100:
                losses.pop(0)
            avg_loss = sum(losses) / len(losses)

            print(f"{epoch}, {batch_index}, {avg_loss:.6f}, {mse.item():.6f}, {kl.item():.6f}", end="\r")

            #time.sleep(1.0)

def show_image(autoencoder):
    with torch.no_grad():
        generated = autoencoder.decoder(autoencoder.linear_before_decoder(torch.randn((1, 1, LATENT_DIM))).view(1, 32*4, 8, 8))
    
    print(generated.shape)
    plt.imshow(generated.squeeze().permute(1,2,0).numpy())
    plt.show()

if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    #DEVICE = 'cpu'

    data = dataset.AllVae()
    loader = DataLoader(data, batch_size=16, shuffle=True)

    autoencoder = VariationalAutoencoder(4)
    print(summary(autoencoder, (3, 64, 64), device='cpu'))
   # autoencoder.load_state_dict(torch.load("./trained/variational_autoencoder.pth"))

    autoencoder = autoencoder.to(DEVICE)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)

    #show_image(autoencoder)
    train(loader, autoencoder, 100000, optimizer, DEVICE, save_interval=10)



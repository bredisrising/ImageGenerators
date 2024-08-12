import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import dataset
import matplotlib.pyplot as plt
import time
from torchsummary import summary
import sys
from torchvision.transforms import transforms


LATENT_DIM = 32

class GenerativeAdversarialNetwork(nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier

        self.discriminator = nn.Sequential(
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
            nn.Linear(32*multiplier*8*8, 1),
        )

        self.generator = nn.Sequential(
            nn.Linear(LATENT_DIM, 32*multiplier*8*8),
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
                torch.nn.init.xavier_uniform(sm.weight)

    def forward(self, x):
        pass



def train(dataloader, autoencoder, epochs, optimizer, device, save_interval=25):
    losses = []

    for epoch in range(epochs):

        if epoch % save_interval == 0 and epoch != 0:
            torch.save(autoencoder.state_dict(), "./trained/wgan.pth")

        for batch_index, batch in enumerate(dataloader):
            input_images = batch[0].to(device)

            generated_outputs, means, log_vars = autoencoder(input_images)

            optimizer.zero_grad()
            #loss = mse(generated_outputs, input_images  )
            #mse, kl = ELBO(input_images, generated_outputs, means, log_vars)
            
            #loss = mse + kl * 8.0

            loss.backward()
            optimizer.step()

            stddevs = torch.exp(0.5 * log_vars)

            losses.append(loss.item())
            if len(losses) > 100:
                losses.pop(0)
            avg_loss = sum(losses) / len(losses)

            print(f"{epoch}, {batch_index}, {avg_loss:.6f}, {mse.item():.6f}, {kl.item():.6f}, {means.mean().item()}, {stddevs.mean().item()}", end="\r")

            #time.sleep(1.0)


def get_z_distribution(autoencoder, data):
    zs = []

    SAMPLES = 64
    
    for i in range(SAMPLES):
        with torch.no_grad():
            image = data[i]
            #print(i, image[0].shape)
            #latent_vectors.append(autoencoder(image[0].unsqueeze(0)))
            z = autoencoder.encoder(image[0].unsqueeze(0))
        zs.append(z)

    zs = torch.stack(zs).squeeze()
    #print(zs.shape)
    means = zs[:, LATENT_DIM:]
    stddevs = torch.exp(0.5*zs[:, :LATENT_DIM])

    print(means, stddevs)
    print('\n\n')
    print(means.mean(), stddevs.mean())

    return means, stddevs


def show_image(autoencoder, num_images_squared=1, latent_vector=None):
    images = []

    for i in range(num_images_squared**2):
        print(i/(num_images_squared**2)*100, '%', end='\r')
        with torch.no_grad():
            if latent_vector != None:
                generated = autoencoder.decoder(autoencoder.linear_before_decoder(latent_vector).view(1, 32*4, 8, 8))
            else:
                generated = autoencoder.decoder(autoencoder.linear_before_decoder(torch.randn((1, 1, LATENT_DIM))).view(1, 32*autoencoder.multiplier, 8, 8))
            images.append(generated)

    fig, axs = plt.subplots(num_images_squared, num_images_squared)

    for i in range(num_images_squared):
        for j in range(num_images_squared):
            axs[i][j].imshow(images[i*num_images_squared+j].squeeze().permute(1,2,0).numpy())

    #plt.savefig("./results/vae32latentdim_256_images.jpg")
    plt.show()


def get_fid(autoencoder, data, samples):
    import inception_metrics

    real = []
    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()
    for i in range(samples):
        real_image = data[i][0]
        resized = totensor(toimage(real_image).resize((299,299)))
        real.append(resized)

    fake = []
    for i in range(samples):
        latent_vector = torch.randn((1, 1, LATENT_DIM))
        generated = autoencoder.generator(autoencoder.linear_before_decoder(latent_vector).view(1, 32*autoencoder.multiplier, 8, 8))
        #print(generated.shape)
        fake.append(totensor(toimage(generated.squeeze()).resize((299, 299))))   
        #fake.append(generated)

    print("done generating images")

    return inception_metrics.frechet_inception_distance(real, fake)



if __name__ == "__main__":
    command = sys.argv[1]
    DEVICE = sys.argv[2]

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data = dataset.AllVae()
    loader = DataLoader(data, batch_size=16, shuffle=True)

    gan = GenerativeAdversarialNetwork(6)
    print(summary(gan, (3, 64, 64), device='cpu'))
    gan.load_state_dict(torch.load("./trained/wgan.pth"))

    gan = gan.to(DEVICE)

    

    if command == "generate":
        show_image(gan, 8)
    elif command == "train":
        optimizer = torch.optim.Adam(gan.parameters(), lr=0.0001)
        train(loader, gan, 10000, optimizer, DEVICE, save_interval=10)
    elif command == "fid":
        print("Frechet Inception Distance Score: ", get_fid(gan, data, 256))
    
    #get_z_distribution(gan.generator, data)

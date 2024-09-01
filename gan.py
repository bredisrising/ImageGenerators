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


LATENT_DIM = 16
BATCH_SIZE = 32

EVAL_LATENT_VECTOR = torch.randn((1, 1, LATENT_DIM)).to("cuda")


class GenerativeAdversarialNetwork(nn.Module):
    def __init__(self, multiplier):
        super().__init__()
        self.multiplier = multiplier
        self.critic_multiplier = 2

        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 24*self.critic_multiplier, 3, 1, 1),  # 64
            nn.LeakyReLU(),
            nn.Conv2d(24*self.critic_multiplier, 24*self.critic_multiplier, 3, 1, 1),  
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), # 32
            nn.Conv2d(24*self.critic_multiplier, 48*self.critic_multiplier, 3, 1, 1),  
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), # 16
            nn.Conv2d(48*self.critic_multiplier, 48*self.critic_multiplier, 3, 1, 1),  
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2), # 8
            nn.Conv2d(48*self.critic_multiplier, 24*self.critic_multiplier, 3, 1, 1),  
            nn.LeakyReLU(),
            nn.Conv2d(24*self.critic_multiplier, 24*self.critic_multiplier, 3, 1, 1),  # 8
            nn.Flatten(),
            nn.Linear(24*self.critic_multiplier*8*8, 1),
        )

        self.latent_to_linear = nn.Sequential(
            nn.Linear(LATENT_DIM, 32*multiplier*8*8)
        )

        self.generator = nn.Sequential(
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

    def forward(self, x):
        pass

def compute_gradient_penalty(critic, real_samples, fake_samples, device, reg_weight=1):
    interpolation_parameter = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolated = (interpolation_parameter * real_samples + (1 - interpolation_parameter) * fake_samples).requires_grad_(True)

    interpolated_scores = critic(interpolated)

    gradient = torch.autograd.grad(
        outputs=interpolated_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.size(0), -1)
    grad_norm = torch.sqrt(torch.sum(gradient ** 2, dim=1) + 1e-12)

    gp = reg_weight * ((grad_norm - 1) ** 2).mean()

    return gp

def train(dataloader, gan, epochs, critic_optimizer, generator_optimizer, device, save_interval=25):
    counter = 1

    losses = []

    generator_loss = torch.tensor(0.)

    for epoch in range(epochs):
        if epoch % save_interval == 0 and epoch != 0:
            torch.save(gan.state_dict(), "./trained/wgan.pth")

        for batch_index, batch in enumerate(dataloader):
            real_images = batch[0].to(device)

            latent_input_vector = torch.randn((BATCH_SIZE, 1, LATENT_DIM)).to(device)
            latent_input_vector = gan.latent_to_linear(latent_input_vector)
            #print(latent_input_vector.shape)
            latent_input_vector = latent_input_vector.view(BATCH_SIZE, 32*gan.multiplier, 8, 8)

            fake_generated = gan.generator(latent_input_vector)

            real_score = gan.discriminator(real_images)
            fake_score = gan.discriminator(fake_generated.detach())

            critic_optimizer.zero_grad()

            discriminator_loss = -torch.mean(torch.log(torch.sigmoid(real_score - fake_score.mean())+1e-5)) - torch.mean(torch.log(1 - torch.sigmoid(fake_score - real_score.mean())+1e-5)) 
            discriminator_loss += compute_gradient_penalty(gan.discriminator, real_images, fake_generated.detach(), device, 10)
            discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(gan.discriminator.parameters(), max_norm=1.0)
            critic_optimizer.step()

            if counter % 4 == 0:
                generator_optimizer.zero_grad()
                generator_fake_score = gan.discriminator(fake_generated)
                generator_loss = -torch.mean(torch.log(torch.sigmoid(generator_fake_score - real_score.mean().detach())+1e-5)) - torch.mean(torch.log(1 - torch.sigmoid(real_score.detach() - generator_fake_score.mean())+1e-5))
                generator_loss.backward()
                #torch.nn.utils.clip_grad_norm_(gan.generator.parameters(), max_norm=1.0)
                generator_optimizer.step()
                #eval(gan.generator)
            
            counter += 1

            losses.append(discriminator_loss.item())
            if len(losses) > 100:
                losses.pop(0)
            avg_loss = sum(losses) / len(losses)

            print(f"{epoch}, {batch_index}, Gen Loss: {generator_loss.item()} Critic Loss: {avg_loss:.6f}  Real Score: {real_score.mean()}  Fake Score: {fake_score.mean()}", end="\r")

        eval(gan.generator)

def eval(generator):
    topilimage = transforms.ToPILImage()
    latent_input_vector = torch.randn((1, 1, LATENT_DIM)).to("cuda")
    latent_input_vector = gan.latent_to_linear(latent_input_vector)
    #print(latent_input_vector.shape)
    latent_input_vector = latent_input_vector.view(1, 32*gan.multiplier, 8, 8)

    with torch.no_grad():
        fake_generated = generator(latent_input_vector)

    image = topilimage(fake_generated.cpu().squeeze())
    image.save("./gan.png")



def show_image(gan, num_images_squared=1, latent_vector=None):
    images = []

    for i in range(num_images_squared**2):
        print(i/(num_images_squared**2)*100, '%', end='\r')
        with torch.no_grad():
            if latent_vector != None:
                generated = gan.decoder(gan.linear_before_decoder(latent_vector).view(1, 32*4, 8, 8))
            else:
                generated = gan.decoder(gan.linear_before_decoder(torch.randn((1, 1, LATENT_DIM))).view(1, 32*gan.multiplier, 8, 8))
            images.append(generated)

    fig, axs = plt.subplots(num_images_squared, num_images_squared)

    for i in range(num_images_squared):
        for j in range(num_images_squared):
            axs[i][j].imshow(images[i*num_images_squared+j].squeeze().permute(1,2,0).numpy())

    #plt.savefig("./results/vae32latentdim_256_images.jpg")
    plt.show()


def get_fid(gan, data, samples):
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
        generated = gan.generator(gan.linear_before_decoder(latent_vector).view(1, 32*gan.multiplier, 8, 8))
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
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    gan = GenerativeAdversarialNetwork(2)
    print(summary(gan.discriminator, (3, 64, 64), device='cpu'))
    gan.load_state_dict(torch.load("./trained/wgan.pth"))

    gan = gan.to(DEVICE)

    if command == "generate":
        show_image(gan, 8)
    elif command == "train":
        # critic_optimizer = torch.optim.RMSprop(gan.discriminator.parameters(), lr=0.001)
        # generator_optimizer = torch.optim.RMSprop(gan.generator.parameters(), lr=0.001)
        critic_optimizer = torch.optim.Adam(gan.discriminator.parameters(), lr=0.0000001, betas = (.5, .9))
        generator_optimizer = torch.optim.Adam(gan.generator.parameters(), lr=0.0000001, betas=(.5, .9))

        train(loader, gan, 100000, critic_optimizer, generator_optimizer, DEVICE, save_interval=10)
    elif command == "fid":
        print("Frechet Inception Distance Score: ", get_fid(gan, data, 256))
    
    #get_z_distribution(gan.generator, data)``

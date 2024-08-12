import torch
import matplotlib.pyplot as plt

def show_image(autoencoder, latent_dim, num_images_squared=1, latent_vector=None):
    images = []

    for i in range(num_images_squared**2):
        print(i/(num_images_squared**2)*100, '%', end='\r')
        with torch.no_grad():
            if latent_vector != None:
                generated = autoencoder.decoder(autoencoder.linear_before_decoder(latent_vector).view(1, 32*4, 8, 8))
            else:
                generated = autoencoder.decoder(autoencoder.linear_before_decoder(torch.randn((1, 1, latent_dim))).view(1, 32*autoencoder.multiplier, 8, 8))
            images.append(generated)

    fig, axs = plt.subplots(num_images_squared, num_images_squared)

    for i in range(num_images_squared):
        for j in range(num_images_squared):
            axs[i][j].imshow(images[i*num_images_squared+j].squeeze().permute(1,2,0).numpy())

    plt.show()
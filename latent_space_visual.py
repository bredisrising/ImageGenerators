import pygame
import torch
import numpy as np
from variational_autoencoder import VariationalAutoencoder

autoencoder = VariationalAutoencoder(6)
autoencoder.load_state_dict(torch.load("./trained/variational_autoencoder_higher_kl_bigger_dataset.pth"))

pygame.init()

window_size = (1280, 720)
screen = pygame.display.set_mode(window_size)

clock = pygame.time.Clock()

font = pygame.font.SysFont("Arial", 24)

latent_vector = torch.randn((1, 32))

scaler = np.ones((10, 10))

while True:
    mouse_pos = pygame.mouse.get_pos()
    screen.fill((255, 255, 255))

    x = (mouse_pos[0] - window_size[0] / 2) / window_size[0] * 16 
    y = (mouse_pos[1] - window_size[1] / 2) / window_size[1] * 16


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()


    latent_vector[0][0] = x
    latent_vector[0][3] = y
    with torch.no_grad():
        image = autoencoder.decoder(autoencoder.linear_before_decoder(latent_vector).view(1, autoencoder.multiplier*32, 8, 8)).detach().numpy()
    
    #print(image.shape)
    #image = np.transpose(image[0], (1, 2, 0))
    #print(image[0].shape)
    image = image[0]
    scaled_channels = [np.kron(image[channel]*255, scaler) for channel in range(3)]
    scaled_image = np.stack(scaled_channels, axis=0)
    scaled_image = np.transpose(scaled_image, (2,1,0))
    #print(scaled_image.shape)
    #image_scaled = np.kron((image*255).astype(np.int32), )
    
    
    
    #print(image_scaled.shape)
    surface = pygame.surfarray.make_surface(scaled_image)
    
    #image_scaled = pygame.surfarray.make_surface(np.zeros((100, 100)) * 255)
    
    screen.blit(surface, (window_size[0] / 2 - 14 * 20, 50))

    
    fps = clock.get_fps()
    text = font.render(f"x: {x:.3f} y: {y:.3f} fps: {fps:.2f}", True, (0, 0, 0))
    screen.blit(text, (10, 10))

    pygame.display.update()

    clock.tick(120)

pygame.quit()


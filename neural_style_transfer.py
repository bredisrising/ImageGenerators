import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage


def generate_gif():
    pass


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

vgg_model = vgg16(pretrained=True)

print(vgg_model.features[:11])

totensor = ToTensor()
toimage = ToPILImage()

style_image = Image.open('./data/style_transfer/starry_night.jpg')
style_image = style_image.resize((128, 128))
style_image = totensor(style_image)

content_image = Image.open('./data/style_transfer/yi_long_ma.jpg')
content_image = content_image.resize((128, 128))
content_image = totensor(content_image)

noise = torch.rand((3, 128, 128))
noise = noise.to(DEVICE).detach()
noise.requires_grad = True

with torch.no_grad():
    style_feature_map = vgg_model.features[:11](style_image)
    print(style_feature_map.shape)
    style_feature_map = style_feature_map.view(256, -1)
    print(style_feature_map.shape)
    style_gram_matrix = (style_feature_map @ style_feature_map.T).detach()

    content_feature_map = vgg_model.features[:11](content_image).detach()


vgg_model = vgg_model.to(DEVICE)



optimizer = torch.optim.LBFGS([noise], lr=1.0)

style_gram_matrix = style_gram_matrix.to('cuda')
content_feature_map = content_feature_map.to('cuda')

def style_loss():
    optimizer.zero_grad()
    noise_feature_map = vgg_model.features[:10](noise).view(256, -1)

    noise_gram_matrix = noise_feature_map @ noise_feature_map.T

    style_loss = ((style_gram_matrix - noise_gram_matrix)**2).sum()
    style_loss.backward()

    return style_loss

def total_loss_func():
    optimizer.zero_grad()
    noise_feature_map = vgg_model.features[:11](noise)

    flattened = noise_feature_map.view(256, -1)

    noise_gram_matrix = flattened @ flattened.T

    content_loss = ((noise_feature_map - content_feature_map)**2).sum()

    style_loss = ((style_gram_matrix - noise_gram_matrix)**2).sum()

    total_loss = style_loss * 1.0 + content_loss * 0.0
    total_loss.backward()

    return total_loss

index = 0
image = toimage(noise.detach().cpu())
image.save(f'./data/style_transfer/style_transfer_result_2/{index}.jpg')
index += 1
while True: 
    loss = optimizer.step(total_loss_func)
    print(loss, "                                           ", end="\r")

    # save image 
    image = toimage(noise.detach().cpu())
    image.save(f'./data/style_transfer/style_transfer_result_2/{index}.jpg')
    index += 1

import torch
from torchvision.models.inception import inception_v3, Inception_V3_Weights
import dataset
from torchsummary import summary

inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)[:-1]

def frechet_inception_distance(data_images, generated_images):
    generated_embeddings = []
    real_embeddings = []

    for i in range (len(generated_images)):
        generated_embeddings.append(inception(generated_images[i]))
    
    for i in range (len(data_images)):
        real_embeddings.append(inception(data_images[i]))

    generated_embeddings = torch.tensor(generated_embeddings)
    real_embeddings = torch.tensor(real_embeddings)

    generated_mean, generated_cov = generated_embeddings.mean(), torch.cov(generated_embeddings.T)

    print(generated_mean.shape, generated_cov.shape)

def inception_score(generated_images):
    pass

if __name__ == "__main__":
    #print(inception)
    print(summary(inception, (3, 350, 350), device='cpu'))
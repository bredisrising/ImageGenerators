import torch
from torchvision.models.inception import inception_v3, Inception_V3_Weights
import dataset
from torchsummary import summary
import numpy as np

inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
inception.fc = torch.nn.Identity()
inception.eval()


def frechet_inception_distance(data_images, generated_images):
    generated_embeddings = []
    real_embeddings = []

    size = len(generated_images)

    for i in range (len(generated_images)):
        print(i/size*100, "%", end="\r")
        with torch.no_grad():
            generated_embeddings.append(inception(generated_images[i].unsqueeze(0)))

    for i in range (len(data_images)):
        print(i/size*100, "%", end="\r")
        with torch.no_grad():
            real_embeddings.append(inception(data_images[i].unsqueeze(0)))

    

    generated_embeddings = torch.stack(generated_embeddings)
    real_embeddings = torch.stack(real_embeddings)

    # axis (0,1) to compute feature-wise mean
    generated_mean, generated_cov = generated_embeddings.mean(axis=(0,1)), torch.cov(generated_embeddings.squeeze().T)
    real_mean, real_cov = real_embeddings.mean(axis=(0,1)), torch.cov(real_embeddings.squeeze().T)

    # FID = ||mu1 - mu2||**2 + Tr(C1 + C2 - 2*sqrt(C1*C2))
    
    a = (real_mean - generated_mean).square().sum(dim=-1)
    b = real_cov.trace() + generated_cov.trace()
    c = torch.linalg.eigvals(real_cov @ generated_cov).sqrt().real.sum(dim=-1)

    return (a + b - 2*c).item()

    #print(generated_mean.shape, generated_cov.shape)

def inception_score(generated_images):
    pass

if __name__ == "__main__":
    print(inception(torch.rand((1, 3, 299, 299))).shape)
    #print(summary(inception, (3, 299, 299), device='cpu'))
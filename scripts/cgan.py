import torch
from torch import nn
from dcgan import Discriminator as BaseDiscriminator, Generator as BaseGenerator

from torchinfo import summary

class GeneratorEmbedding(nn.Module):
    def __init__(self, latent_dims: int = 50):
        super(GeneratorEmbedding, self).__init__()
        self.latent_dims = latent_dims
        self.embedder = nn.Embedding(2, self.latent_dims)

    def forward(self, x):
        x = self.embedder(x)
        return x.view(-1, self.latent_dims, 1, 1)


class DiscriminatorEmbedding(nn.Module):
    def __init__(self, latent_dims: int = 3):
        super(GeneratorEmbedding, self).__init__()
        self.latent_dims = latent_dims
        self.embedder = nn.Embedding(2, self.latent_dims*64*64)

    def forward(self, x):
        x = self.embedder(x)
        return x.view(-1, self.latent_dims, 64, 64)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.base = nn.Sequential(
            nn.ConvTranspose2d(150, 100, 1, 1, 0, bias=False, ),
            nn.BatchNorm2d(100),
            *BaseGenerator().model[:-2],
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.embedder = GeneratorEmbedding()

    def forward(self, x, y):
        pred = self.embedder(y)
        x = torch.cat([x, pred], dim=1)
        print(x.shape)
        pred = self.base(x)

        return pred

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.base = nn.Sequential(
            *BaseDiscriminator().model[:-2],
        )
        self.real_disc = nn.Sequential(
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.class_disc = nn.Sequential(
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.base(x)
        real = self.real_disc(x)
        class_ = self.class_disc(x)
        return real, class_

def convTrans(n, k, s, p):
    return (n*s) + p - s + k

print(convTrans(1, 1, 1, 0))

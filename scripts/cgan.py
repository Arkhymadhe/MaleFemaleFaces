import torch
from torch import nn
from dcgan import Discriminator as BaseDiscriminator, Generator as BaseGenerator


class GeneratorEmbedding(nn.Module):
    def __init__(self, latent_dims: int = 50):
        super(GeneratorEmbedding, self).__init__()
        self.latent_dims = latent_dims
        self.embedder = nn.Embedding(2, self.latent_dims)

    def forward(self, x):
        x = self.embedder(x)
        return x.view(-1, self.latent_dims, 1, 1)


class DiscriminatorEmbedding(nn.Module):
    def __init__(self, num_classes: int = 2, latent_dims: int = 3):
        super(DiscriminatorEmbedding, self).__init__()
        self.latent_dims = latent_dims
        self.num_classes = num_classes
        self.embedder = nn.Embedding(num_classes, self.latent_dims * 64 * 64)

    def forward(self, x):
        x = self.embedder(x)
        return x.view(-1, self.latent_dims, 64, 64)


class Generator(nn.Module):
    def __init__(self, in_channels: int = 100, out_channels: int = 3, latent_dims: int = 50):
        super(Generator, self).__init__()
        self.latent_dims = latent_dims
        self.embedder = GeneratorEmbedding()
        self.base_generator = BaseGenerator(in_channels, out_channels)

        self.base = nn.Sequential(
            nn.ConvTranspose2d(
                self.latent_dims + self.base_generator.in_channels,
                100,
                1,
                1,
                0,
                bias=False,
            ),
            nn.BatchNorm2d(100),
            *self.base_generator.model[:-2],
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, y):
        pred = self.embedder(y)
        x = torch.cat([x, pred], dim=1)
        pred = self.base(x)

        return pred


class Discriminator(nn.Module):
    def __init__(self, num_classes: int = 2, latent_dims: int = 50, in_channels: int = 3, out_channels: int = 1):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.latent_dims = latent_dims
        self.num_classes = num_classes

        self.base = nn.Sequential(
            *BaseDiscriminator(self.in_channels + self.latent_dims, self.out_channels).model[:-2],
        )
        self.real_disc = nn.Sequential(
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid()
        )
        self.class_disc = nn.Sequential(
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid()
        )

        self.embedder = DiscriminatorEmbedding(num_classes = self.num_classes, latent_dims=self.latent_dims)

    def forward(self, x, labels):
        y = self.embedder(labels)
        x = torch.cat([x, y], dim = 1)
        x = self.base(x)
        real = self.real_disc(x)
        class_ = self.class_disc(x)
        return real, class_


def convTrans(n, k, s, p):
    return (n * s) + p - s + k


print(convTrans(1, 1, 1, 0))

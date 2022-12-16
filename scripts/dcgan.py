from torch import nn


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super(Discriminator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 8, self.out_channels, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.model(input)


class Generator(nn.Module):
    def __init__(self, in_channels: int = 100, out_channels: int = 3):
        super(Generator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                self.in_channels,
                64 * 8,
                4,
                1,
                0,
                bias=False,
            ),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, self.out_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.model(input)

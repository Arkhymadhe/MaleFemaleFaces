from typing import Optional
from matplotlib import pyplot as plt

import torch
import numpy as np

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import time

from models import Generator, Discriminator
from viz_utils import generate_images


def initialize_weights(model):
    for layer in model.modules():
        classname = layer.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(layer.weight.data, 1.0, 0.02)
            nn.init.constant_(layer.bias.data, 0)

    return


def get_optimizer(model, lr: float, beta1: float = None, beta2: float = None, weight_decay: float = 0.3):
    beta1 = 0.5 if not beta1 else beta1
    beta2 = 0.99 if not beta2 else beta2

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    return optimizer


def load_model(device: torch.device, constructor: nn.Module):
    model = constructor().to(device)
    return model


def train_loop(train_dl: DataLoader, epochs: int,
               lr: float, betas: tuple[float, float], figsize: tuple[int, int],
               decay_rate: float, criterion, device: torch.device, folder: str, history: dict):
    history = dict()
    dl_length = len(train_dl)
    noisy_labels = bool(np.random.choice([0, 1]))
    noisy_labels = False

    generator = load_model(device, Generator).apply(initialize_weights)
    discriminator = load_model(device, Discriminator).apply(initialize_weights)

    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(betas[0], betas[1]), weight_decay=decay_rate)
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(betas[0], betas[1]), weight_decay=decay_rate)

    # opt_g = get_optimizer(generator, lr, betas[0], betas[1],
    #                     weight_decay=decay_rate)
    # opt_d = get_optimizer(discriminator, lr, betas[0], betas[1],
    #                     weight_decay=decay_rate)

    print(f"Training with noisy_labels = {noisy_labels}...\n")

    for epoch in range(epochs):
        history[f'Epoch {epoch + 1}'] = {'G-Loss': [], 'D-Loss': []}
        for ix, data in enumerate(train_dl):
            ### Zero out the accumulated gradients
            opt_d.zero_grad()

            ### Update the discriminator (real data)
            real_outputs = discriminator(data.to(device))
            real_loss = criterion(real_outputs.squeeze(), torch.ones(len(real_outputs)).to(device))
            real_loss.backward()

            ### Update the discriminator (fake data)
            noise = torch.randn(size=(len(data), 100, 1, 1), device=device)
            fake_imgs = generator(noise).detach()

            fake_outputs = discriminator(fake_imgs)
            fake_loss = criterion(fake_outputs.squeeze(), torch.zeros(len(fake_outputs)).to(device))
            fake_loss.backward()

            opt_d.step()

            ### Update the generator
            opt_g.zero_grad()

            noise = torch.randn(size=(len(data), 100, 1, 1), device=device)
            generated_images = generator(noise)

            dis_pred = discriminator(generated_images)
            gen_loss = criterion(dis_pred.squeeze(), torch.ones(len(dis_pred)).to(device))

            gen_loss.backward()
            opt_g.step()

            ### Record the losses
            history[f'Epoch {epoch + 1}']['G-Loss'].append(gen_loss.item())
            history[f'Epoch {epoch + 1}']['D-Loss'].append((real_loss + fake_loss).item())

            if not (ix + 1) % 10 or ix + 1 == len(train_dl):
                print(f'Epoch [{epoch + 1:02d}/{epochs:02d}]')
                print(f'\tIteration: [{ix + 1:04d}/{len(train_dl)}]')
                print(f'\tG-Loss: {gen_loss.item():.5f} | D-Loss: {(real_loss + fake_loss).item():.5f}')
        # step_d.step()
        # step_g.step()

        ### Generate test samples via the generator
        generate_images(model=generator, device=device, folder = folder, figsize = figsize, num_samples=16, save_images=True, epoch=epoch + 1)

    return history, discriminator, generator, opt_d, opt_g


def train_step(opt_d, opt_g, criterion, discriminator, generator,
               data, device, history, noisy_labels: bool = True):
    opt_d.zero_grad()

    ### Update the discriminator (real data)
    real_outputs = discriminator(data.to(device))
    real_loss = criterion(real_outputs.squeeze(), torch.ones(len(real_outputs)).to(device))
    real_loss.backward()

    ### Update the discriminator (fake data)
    noise = torch.randn(size=(len(data), 100, 1, 1), device=device)
    fake_imgs = generator(noise).detach()

    fake_outputs = discriminator(fake_imgs)
    fake_loss = criterion(fake_outputs.squeeze(), torch.zeros(len(fake_outputs)).to(device))
    fake_loss.backward()

    opt_d.step()

    ### Update the generator
    opt_g.zero_grad()

    noise = torch.randn(size=(len(data), 100, 1, 1), device=device)
    generated_images = generator(noise)

    dis_pred = discriminator(generated_images)
    gen_loss = criterion(dis_pred.squeeze(), torch.ones(len(dis_pred)).to(device))

    gen_loss.backward()
    opt_g.step()

    ### Record the losses
    history[f'Epoch {epoch + 1}']['G-Loss'].append(gen_loss.item())
    history[f'Epoch {epoch + 1}']['D-Loss'].append((real_loss + fake_loss).item())

    return history, discriminator, generator, opt_d, opt_g


def visualize_losses(history: dict[str, dict[str, str]], epochs: int):
    g_losses = [history[f"Epoch {e + 1}"]['G-Loss'] for e in range(epochs)]
    d_losses = [history[f"Epoch {e + 1}"]['D-Loss'] for e in range(epochs)]

    plt.figure(figsize=(10, 10))
    plt.plot(range(epochs), d_losses, label="Discriminator loss over epoch")
    plt.plot(range(epochs), g_losses, label="Generator loss over epoch")

    plt.show()

    plt.close('all')
    return


def keep_time(start_time: float): return time.time() - start_time

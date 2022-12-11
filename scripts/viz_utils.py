from numpy import ndarray, mean
import torch
from torch import Tensor
from typing import Tuple
import os
from matplotlib import pyplot as plt


def denorm(img: Tensor):
    img = (img + 1)/2
    return img


@torch.no_grad()
def generate_images(model, device: torch.device, epoch: int, num_samples: int = 16,
                    nrows: int = 4, ncolumns: int = 4, latent_dims: Tuple[int, int, int] = (100, 1, 1),
                    figsize: Tuple[int, int] = (10, 10), folder: str = 'DCGAN_images',
                    save_images: bool = False, conditional: bool = False):
    """
    Generates images when called.

    Major parameters:
        Model: Image generator

    """

    assert nrows * ncolumns <= num_samples

    model.eval()

    inputs = torch.randn(size=[num_samples, *latent_dims], device=device)
    if not conditional:
        images = model(inputs).detach().cpu().permute(0, 2, 3, 1)
    else:
        rand_labels = torch.randint(low=0, high=2, size=(num_samples,), device=device)
        images = model(inputs, rand_labels).detach().cpu().permute(0, 2, 3, 1)

    display_images(denorm(images).numpy(), nrows, ncolumns, figsize, folder, epoch, save_images)

    return


def display_images(images: ndarray, nrows: int = 4, ncolumns: int = 4, figsize: Tuple[int, int] = (10, 10),
                   folder: str = 'DCGAN_images', epoch: int = 0, save_images: bool = False):
    plt.figure(figsize=figsize)
    for i in range(int(nrows * ncolumns)):
        plt.subplot(nrows, ncolumns, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

    if save_images:
        if not os.path.exists(folder):
            os.mkdir(folder)
        plt.savefig(f'{folder}/Generated_Images_at_Epoch_{epoch:04d}.png', dpi=300)

    plt.show()
    plt.close('all')
    return

def visualize_losses(history: dict[str, dict[str, str]], epochs: int):
    g_losses = [mean(history[f"Epoch {e + 1}"]['G-Loss']) for e in range(epochs)]
    d_losses = [mean(history[f"Epoch {e + 1}"]['D-Loss']) for e in range(epochs)]

    plt.figure(figsize=(10, 10))
    plt.plot(range(epochs), d_losses, label="Discriminator loss")
    plt.plot(range(epochs), g_losses, label="Generator loss")
    plt.title("Losses over time")
    plt.legend()

    plt.show()

    plt.close('all')
    return
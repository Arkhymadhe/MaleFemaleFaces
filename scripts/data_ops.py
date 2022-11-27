import os
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

from typing import Tuple, Callable
from PIL import Image


class CelebDataset(Dataset):
    def __init__(self, folder='dataset', transform: Callable = None):
        super(CelebDataset, self).__init__()

        self.folder = folder
        self.transform = transform
        self.folders = list(map(lambda x: os.path.join(self.folder, x), ['males', 'females']))
        self.images = list(map(lambda x: os.path.join(self.folders[0], x), os.listdir(self.folders[0]))) + \
                      list(map(lambda x: os.path.join(self.folders[1], x), os.listdir(self.folders[1])))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        image_path = self.images[ix]
        image = Image.open(image_path)
        image = self.transform(image)
        return image


def get_dataset(path: str, stats: Tuple[tuple, tuple], size: int):
    dataset = CelebDataset(path, get_transform(stats, size))
    return dataset


def get_dataloader(dataset: CelebDataset, batch_size: int):
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return dataloader


def load_data(path: str, stats: Tuple[tuple, tuple], size: int, batch_size: int):
    data = get_dataset(path, stats, size)
    return get_dataloader(data, batch_size)


def get_transform(stats: Tuple[tuple, tuple] = None, size: int = 64):
    if not stats:
        stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    train_transform = T.Compose([T.Resize(size),
                                 T.ToTensor(),
                                 T.Normalize(*stats)
                                 ])
    return train_transform

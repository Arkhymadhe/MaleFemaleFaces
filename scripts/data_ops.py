import os
from torch import float32
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

from typing import Tuple, Callable
from PIL import Image


class CelebDataset(Dataset):
    def __init__(self, folder='dataset', transform: Callable = None):
        super(CelebDataset, self).__init__()

        self.folder = folder
        self.transform = transform

        self.labels = ['males', 'females']
        self.label_map = dict(zip(self.labels, [0, 1]))

        self.folders = list(map(lambda x: os.path.join(self.folder, x), self.labels))
        self.images = list(map(lambda x: os.path.join(self.folders[0], x), os.listdir(self.folders[0]))) + \
                      list(map(lambda x: os.path.join(self.folders[1], x), os.listdir(self.folders[1])))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ix):
        image_path = self.images[ix]
        image = Image.open(image_path)
        image = self.transform(image)
        return image, self.label_map[image_path.split("\\")[-2]]


def get_dataset(path: str, stats: Tuple[tuple, tuple], size: int):
    dataset = CelebDataset(path, get_transform(stats, size))
    return dataset

def collate_function(batch):
    return batch[0], batch[1].to(float32)

def get_dataloader(dataset: CelebDataset, batch_size: int, collate_fn:Callable = None):
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader


def load_data(path: str, stats: Tuple[tuple, tuple], size: int, collate_fn: Callable, batch_size: int):
    data = get_dataset(path, stats, size)
    return get_dataloader(data, batch_size, collate_fn)


def get_transform(stats: Tuple[tuple, tuple] = None, size: int = 64):
    if not stats:
        stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    train_transform = T.Compose([T.Resize(size),
                                 T.ToTensor(),
                                 T.Normalize(*stats)
                                 ])
    return train_transform

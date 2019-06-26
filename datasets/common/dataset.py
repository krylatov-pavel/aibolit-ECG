import torch
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, examples_provider):
        self._examples_provider = examples_provider

    def __getitem__(self, index):
        x, y = self._examples_provider.get(index)
        return (torch.Tensor(x), y)

    def __len__(self):
        return self._examples_provider.count
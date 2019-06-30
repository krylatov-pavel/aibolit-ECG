import torch
import numpy as np
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, examples_provider, transform=None):
        self._examples_provider = examples_provider
        self._transform = transform

    def __getitem__(self, index):
        x, y = self._examples_provider.get(index)
        x = np.asarray(x, dtype=np.float32)

        if self._transform:
            x = self._transform(x)
        else:
            x = torch.Tensor(x)

        return (x, y)

    def __len__(self):
        return self._examples_provider.count
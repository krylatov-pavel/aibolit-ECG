import torch
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, examples_provider, fold_nums):
        self._examples_provider = examples_provider
        self._fold_nums = fold_nums

    def __getitem__(self, index):
        x, y = self._examples_provider.get_example(index, self._fold_nums)
        return (torch.Tensor(x), y)

    def __len__(self):
        return self._examples_provider.len(self._fold_nums)

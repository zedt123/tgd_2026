from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset

"""
Code adapted from:
https://github.com/zhangbingliang2019/DAPS/blob/main/data.py
Licensed under MIT.
"""

class DiffusionData(ABC, Dataset):
    @abstractmethod
    def get_shape(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    def get_data(self, size=16, sigma=0):
        data = torch.stack([self.__getitem__(i) for i in range(size)], dim=0)
        return data + torch.randn_like(data) * sigma

    def get_random(self, size=16, sigma=0):
        shape = (size, *self.get_shape())
        return torch.randn(shape) * sigma
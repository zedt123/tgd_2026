from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from src.registry import DATASET
from .base import DiffusionData

"""
Code adapted from:
https://github.com/zhangbingliang2019/DAPS/blob/main/data.py
Licensed under MIT.
"""
    
@DATASET.register('image')
class ImageDataset(DiffusionData):
    """
    A concrete class for handling image datasets, inherits from DiffusionData.

    This class is responsible for loading images from a specified directory,
    applying transformations to center crop the squared images of given resolution.

    Supported extension : ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    Output data range   : [-1, 1]
    """
    def __init__(self, root='data/test-ffhq', resolution=256, padding=0, start_id=None, end_id=None):
        # Define the file extensions to search for
        extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
        self.data = [file for ext in extensions for file in Path(root).rglob(ext) if file.is_file()]
        self.data = sorted(self.data, key=lambda p: p.as_posix())
        self.data = self.data[start_id:end_id+1]
        t_list = []
        if padding > 0:
            t_list.append(transforms.Pad(padding))
        t_list += [
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),
        ]
        self.trans = transforms.Compose(t_list)
        self.res = resolution
        with Image.open(self.data[0]) as im:
            self.channels = 1 if im.mode == "L" else 3


    def __getitem__(self, i):
        with Image.open(self.data[i]) as im:
            # Ensure consistent mode
            if self.channels == 1:
                im = im.convert("L")
            else:
                im = im.convert("RGB")

            img = self.trans(im)
        return img
    def get_shape(self):
        return (self.channels, self.res, self.res)

    def __len__(self):
        return len(self.data)
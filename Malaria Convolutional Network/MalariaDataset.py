import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import io, transform
from torch.utils.data.sampler import SubsetRandomSampler
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class MalariaDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = self.df.iloc[index, 0]
        image = io.imread(img_name)
        label = self.df.iloc[index, 1]

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def showimage(self,image, label):
        """Show image with landmarks"""
        plt.imshow(image)
        plt.pause(0.001)  # pause a bit so that plots are updated

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        return {'image': torch.from_numpy(image),
                'label': label}

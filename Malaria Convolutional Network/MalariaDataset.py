import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
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
            print(self.transform)
            sample['image'] = self.transform(sample['image'])

        return sample

    def show_image(self,image, label):

        plt.imshow(image)
        plt.pause(0.001)  # pause a bit so that plots are updated

import torch
import torch.nn as nn
import pandas as pd
import numpy as np 
import time
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets, transforms
import torch.optim as optim
from PIL import Image
import os 
import copy
import glob

image_size = 128

data_dir = r'C:\DTU\master_thesis\fsl\Object-classification-with-few-shot-learning\data\raw\updated_ds'

class MaritimeDataset(Dataset):

    def __init__(self, root_dir="", transform=transforms) -> None:
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.transform = transform
        self.paths, self.labels = zip(*self.dataset.samples)
        self.classes = self.dataset.class_to_idx

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):

        path, target = self.paths[index], self.labels[index]

        with Image.open(path) as img:

            if self.transform is not None:
                img = self.transform(img)

        return img, target #, index, path





complete_dataset = MaritimeDataset(root_dir = data_dir,
                                    transform = transforms.Compose([
                                        transforms.Resize(image_size),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip()
                                    ]))

from tqdm import tqdm

def mean_and_std() -> None:
    """This function calculates the mean and standard deviation of the dataset"""

    dataloader = DataLoader(complete_dataset, shuffle=False, batch_size=12)
    mean = 0.0
    std = 0.0
    nb_samples = 0.0

    for images, _ in tqdm(dataloader):
        batch_samples = images.size(0)
        data = images.view(batch_samples, images.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std
    



if __name__ == '__main__':
    mean, std = mean_and_std()

    print(mean)
    print(std)
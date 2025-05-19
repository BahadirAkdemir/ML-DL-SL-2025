from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import torch


class AgePredictionDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.root_dir, row['full_path'])
        image = Image.open(image_path).convert("RGB")
        age = row['age']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(age, dtype=torch.float32)


class GenderPredictionDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.root_dir, row['full_path'])
        image = Image.open(image_path).convert("RGB")
        age = row['gender']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(age, dtype=torch.float32)


class NoLabelFace(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.root_dir, row['full_path'])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

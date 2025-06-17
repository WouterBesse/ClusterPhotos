import numpy as np
import pandas as pd
import os, glob

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

class LocationDataset(Dataset):
    def __init__(self, dataset_file_path = "F:\msc\multimedia\ClusterPhotos\dataset\location_dataset_3000.npz"):
        dataset = np.load(dataset_file_path)
        self.images = dataset['images']
        self.ids = dataset['ids']
        self.latitudes = dataset['latitudes']
        self.longitudes = dataset['longitudes']

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'id': self.ids[idx],
            'latitude': self.latitudes[idx],
            'longitude': self.longitudes[idx]
        }

def main():
    # Test the dataset
    dataset_path = "*.npz"
    dataset = LocationDataset(dataset_path)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in tqdm(dataloader):
        images = batch['image']
        ids = batch['id']
        latitudes = batch['latitude']
        longitudes = batch['longitude']

        # Example: Print the shape of the images in the batch
        print(f"Batch size: {len(images)}, Image shape: {images.shape}, IDs: {ids}, Latitudes: {latitudes}, Longitudes: {longitudes}")

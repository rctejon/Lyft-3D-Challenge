import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json

class TwoDtoThreeDDataset(Dataset):
    """2D detection to 3D detections dataset."""

    def __init__(self, json_file):
        """
        Args:
            json_file (string): Path to the json file with annotations.
        """

        data = json.load(open(json_file))
        trainFeatures = list(map(lambda x:x["features"], data['train']))
        valFeatures = list(map(lambda x:x["features"], data['val']))
        trainLabels = list(map(lambda x:x["label"], data['train']))
        valLabels= list(map(lambda x:x["label"], data['val']))
        self.features = trainFeatures + valFeatures
        self.labels = trainLabels + valLabels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = [torch.tensor(self.features[idx]),  torch.tensor(self.labels[idx])]

        return sample
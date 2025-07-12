import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import pickle


class TrajectoryDataset(Dataset):
    def __init__(self, data_path: Path, transform=None):
        data = np.load(data_path)
        self.trajectories = torch.from_numpy(data['trajectories']).float()
        self.masks = torch.from_numpy(data['masks']).bool()
        self.categories = torch.from_numpy(data['categories']).long()
        self.lengths = torch.from_numpy(data['lengths']).long()
        self.transform = transform
        
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        mask = self.masks[idx]
        category = self.categories[idx]
        length = self.lengths[idx]
        
        if self.transform:
            trajectory = self.transform(trajectory)
            
        return trajectory, mask, category, length
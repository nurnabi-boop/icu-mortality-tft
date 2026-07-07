"""PyTorch Dataset for ICU time-series."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ICUDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)          # (N, T, F)
        self.y = torch.from_numpy(y).long()   # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

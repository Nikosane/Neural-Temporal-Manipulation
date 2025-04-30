import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch


def normalize_series(series):
    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(series)
    return norm_data, scaler


def to_tensor(series, seq_len):
    """
    Converts a normalized numpy time series into PyTorch tensors with shape:
    (batch, seq_len, features)
    """
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+1:i+seq_len+1])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

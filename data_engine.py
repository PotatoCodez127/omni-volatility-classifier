# data_engine.py
import torch
import numpy as np

def generate_xauusd_data(samples=1000):
    # Simulating 5 candles of OHLCV data (5 candles * 5 features = 25 inputs per sample)
    X = np.random.rand(samples, 25)
    
    # Synthetic labels: 1 (High Volatility), 0 (Consolidation)
    # We create a random logic: if the sum of volumes (last 5 features) is high, it's volatile
    y = np.where(X[:, -5:].sum(axis=1) > 2.5, 1.0, 0.0)
    
    # Convert to PyTorch Tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).view(-1, 1) # Reshape to match output
    
    return X_tensor, y_tensor
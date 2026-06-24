# data_engine.py
import torch
import numpy as np
from config import config

def generate_financial_time_series(samples: int = 1200, seed: int = 42):
    """
    Generates structured synthetic financial data simulating sequential OHLCV metrics
    and returns strictly isolated datasets to prevent lookahead data leakage.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    total_features = config.model.input_size
    X = np.zeros((samples, total_features))
    y = np.zeros((samples, 1))
    
    for s in range(samples):
        prices = []
        current_price = 2300.0  # Simulated base XAUUSD price level
        
        # Simulate 5 consecutive OHLCV candles
        for _ in range(5):
            returns = np.random.normal(0.0, 2.5)
            o = current_price + np.random.normal(0.0, 0.5)
            c = o + returns
            h = max(o, c) + abs(np.random.normal(0.5, 0.5))
            l = min(o, c) - abs(np.random.normal(0.5, 0.5))
            v = np.random.uniform(500, 5000)
            
            prices.extend([o, h, l, c, v])
            current_price = c
            
        X[s] = prices
        
        # Calculate trailing realized volatility across the candle sequence
        close_prices = X[s, 3::5]  # Extract indices corresponding to 'Close'
        log_returns = np.diff(np.log(close_prices))
        realized_vol = np.std(log_returns) if len(log_returns) > 0 else 0.0
        
        # Set target classification based on a localized volatility threshold
        y[s] = 1.0 if realized_vol > 0.0012 else 0.0

    # Enforce strict index boundaries: Train (70%), Validation (15%), Test (15%)
    train_end = int(samples * 0.70)
    val_end = train_end + int(samples * 0.15)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    return (
        (torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        (torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        (torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    )
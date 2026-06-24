# model.py
import torch
import torch.nn as nn
from config import config

class VolatilityClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Dynamically map network dimensions from our central configuration container
        self.network = nn.Sequential(
            nn.Linear(config.model.input_size, config.model.hidden_dim1),
            nn.ReLU(),
            nn.Linear(config.model.hidden_dim1, config.model.hidden_dim2),
            nn.ReLU(),
            nn.Linear(config.model.hidden_dim2, config.model.output_size),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes candle sequences forward through the hidden network layers.
        """
        return self.network(x)
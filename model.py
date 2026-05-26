# model.py
import torch.nn as nn

class VolatilityClassifier(nn.Module):
    def __init__(self, input_size=25):
        super(VolatilityClassifier, self).__init__()
        
        # Layer 1: Takes 25 inputs, expands to 64 hidden neurons
        self.layer1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        
        # Layer 2: Takes 64 neurons, compresses to 32
        self.layer2 = nn.Linear(64, 32)
        
        # Output Layer: Condenses to 1 single prediction
        self.output_layer = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # The data flows through the math
        x = self.layer1(x)
        x = self.relu(x)
        
        x = self.layer2(x)
        x = self.relu(x)
        
        x = self.output_layer(x)
        x = self.sigmoid(x) # Squashes final answer between 0 and 1
        
        return x
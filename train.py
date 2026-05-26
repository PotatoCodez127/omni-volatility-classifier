# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from data_engine import generate_xauusd_data
from model import VolatilityClassifier

def train_model():
    X, y = generate_xauusd_data(1000)
    model = VolatilityClassifier()
    
    # The Math components
    criterion = nn.BCELoss() # Binary Cross-Entropy
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam Optimizer
    
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad() # Clear old gradients
        
        # 1. Forward Pass (Make a prediction)
        predictions = model(X)
        
        # 2. Calculate Loss (Measure the mistake)
        loss = criterion(predictions, y)
        
        # 3. Backward Pass (Calculus/Derivatives)
        loss.backward()
        
        # 4. Take a step (Update weights)
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
    return model

if __name__ == "__main__":
    trained_model = train_model()
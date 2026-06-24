# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from config import config
from model import VolatilityClassifier

def train_model(train_data, val_data):
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    model = VolatilityClassifier()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    epochs = config.training.epochs
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Calculate training predictions and update gradients
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        
        # Evaluate model performance on validation data
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val)
                val_loss = criterion(val_preds, y_val)
                
                train_acc = ((predictions >= 0.5).float() == y_train).float().mean().item() * 100
                val_acc = ((val_preds >= 0.5).float() == y_val).float().mean().item() * 100
                
            print(f"Epoch {epoch+1:03d}/{epochs} | "
                  f"Train Loss: {loss.item():.4f} (Acc: {train_acc:.1f}%) | "
                  f"Val Loss: {val_loss.item():.4f} (Acc: {val_acc:.1f}%)")
            
    return model
# test.py
import torch
from data_engine import generate_xauusd_data
from train import train_model

print("Training Model...")
model = train_model()

print("\n--- Live Market Test ---")
# Simulate 3 brand new, unseen 5-candle sequences
X_new, y_real = generate_xauusd_data(3)

# Inference Phase
model.eval()
with torch.no_grad():
    predictions = model(X_new)

for i in range(3):
    prob = predictions[i].item() * 100
    state = "EXPANSION (Volatile)" if prob > 50 else "CONSOLIDATION (Calm)"
    print(f"Market Scenario {i+1}: Probability of Volatility: {prob:.2f}% -> Bot Decision: {state}")
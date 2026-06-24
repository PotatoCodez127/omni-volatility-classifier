# test.py
import torch
from data_engine import generate_financial_time_series
from train import train_model

def run_pipeline():
    print("Initializing Time-Series Data Simulation...")
    train_split, val_split, test_split = generate_financial_time_series()
    X_test, y_test = test_split
    
    print("\nStarting Model Training Phase...")
    model = train_model(train_split, val_split)
    
    print("\n--- Out-of-Sample Test Evaluation (Firewall) ---")
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        criterion = torch.nn.BCELoss()
        test_loss = criterion(test_predictions, y_test).item()
        
        predicted_classes = (test_predictions >= 0.5).float()
        correct_matches = (predicted_classes == y_test).float().sum().item()
        total_samples = y_test.size(0)
        test_accuracy = (correct_matches / total_samples) * 100
        
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}% ({int(correct_matches)}/{total_samples} samples)")
    
    print("\n--- Live Market Pre-Trade Filtering Samples ---")
    for i in range(min(3, total_samples)):
        prob = test_predictions[i].item() * 100
        real_state = "EXPANSION" if y_test[i].item() == 1.0 else "CONSOLIDATION"
        decision = "EXPANSION (Execute)" if prob > 50.0 else "CONSOLIDATION (Block)"
        print(f"Scenario {i+1} | Volatility Prob: {prob:.2f}% -> System Action: {decision} | Target: {real_state}")

if __name__ == "__main__":
    run_pipeline()
# test_core.py
import pytest
import torch
from config import config
from data_engine import generate_financial_time_series
from model import VolatilityClassifier

def test_data_engine_tensor_shapes():
    """
    Verifies that the dataset engine returns valid data structures, matching
    the configured layout requirements.
    """
    train_split, val_split, test_split = generate_financial_time_series(samples=100)
    X_train, y_train = train_split
    
    assert X_train.shape[1] == config.model.input_size
    assert y_train.shape[1] == 1
    assert isinstance(X_train, torch.Tensor)
    assert isinstance(y_train, torch.Tensor)

def test_model_forward_pass_dimensions():
    """
    Ensures the deep neural network maps flattened inputs into bounded
    probability classifications within the correct output format.
    """
    model = VolatilityClassifier()
    mock_input = torch.randn(10, config.model.input_size)
    
    output = model(mock_input)
    
    assert output.shape == (10, 1)
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0)

def test_model_parameter_gradients():
    """
    Confirms that the backward pass tracks gradients properly across layers,
    preventing vanishing or broken weight updates.
    """
    model = VolatilityClassifier()
    mock_input = torch.randn(4, config.model.input_size)
    mock_target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    
    criterion = torch.nn.BCELoss()
    output = model(mock_input)
    loss = criterion(output, mock_target)
    loss.backward()
    
    # Verify that structural layers are tracking gradient updates
    for name, param in model.named_parameters():
        if "weight" in name:
            assert param.grad is not None
            assert torch.sum(torch.abs(param.grad)) > 0.0
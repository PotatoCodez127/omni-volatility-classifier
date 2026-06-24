# config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    input_size: int = 25       # 5 candles * 5 features (OHLCV)
    hidden_dim1: int = 64
    hidden_dim2: int = 32
    output_size: int = 1

@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 64
    seed: int = 42

@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()

# Instantiated global config registry
config = AppConfig()
# XAUUSD Volatility Classifier

## Overview
This is a PyTorch-based Binary Classifier designed to act as a pre-trade filter for an algorithmic trading system. It analyzes the last 5 OHLCV candles of an asset (like XAUUSD) and outputs a probability score predicting imminent market volatility.

## Architecture
* **Inputs:** 25 features (5 sequential candles flattened into a 1D array).
* **Network:** Feed-Forward Deep Neural Network (Dense Layers).
* **Activations:** * `ReLU` for hidden layers to map non-linear price relationships.
  * `Sigmoid` for the output layer to bound predictions to a probability [0, 1].
* **Loss Function:** Binary Cross-Entropy (`BCELoss`).
* **Optimizer:** Adam (`lr=0.01`).

## How to Run
1. Install dependencies: `pip install torch numpy`
2. Run the full pipeline: `python test.py`

## Role in the Omni-Agent Ecosystem
This module prevents trend-following bots from executing during choppy, sideways consolidation periods by filtering out setups with a volatility probability below 50%.
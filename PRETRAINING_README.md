# Transfer Learning Pretraining System

## Overview

This system implements transfer learning by pretraining an LSTM model on 80+ major stocks, then using that model to generate market-aware embeddings for any stock during backtesting.

## Architecture

```
Major Stocks (2019-2024) → Pretrained LSTM → Market Embeddings → LightGBM → Predictions
```

### Pretraining Phase
- Trains on 80+ major stocks across 5+ years
- LSTM learns cross-stock patterns and market dynamics
- Captures sector rotations, macro effects, general market wisdom
- Saves model weights and configuration

### Inference Phase  
- Loads pretrained model (no retraining)
- Generates embeddings for any target stock
- Embeddings capture market knowledge from training
- LightGBM uses embeddings + original features

## Files

### Configuration
- **src/config/pretrain_config.py**: Stock list and training parameters

### Pretraining
- **pretrain_model.py**: Main pretraining script
  - Loads 80+ stocks
  - Prepares sequences for LSTM
  - Trains on RTX 3090 GPU
  - Saves model to `models/pretrained_lstm.pth`

### Inference
- **src/models/pretrained_utils.py**: Model loading and embedding extraction
  - Loads pretrained model
  - Extracts embeddings for new stocks
  - Adds embeddings to feature set

### GUI Integration
- **backtest_app.py**: Updated to use pretrained embeddings
  - New checkbox: "Use Pretrained Market Embeddings"
  - Seamlessly integrates into backtest pipeline

## Usage

### 1. Train Pretrained Model (One-time)

Run full pretraining on all 80+ stocks:
```bash
python pretrain_model.py
```

Estimated time: 2-4 hours on RTX 3090

Or run quick test first (5 stocks, 5 epochs):
```bash
python test_pretrain.py
```

### 2. Run Backtest with Pretrained Embeddings

Start the GUI and enable pretrained embeddings:
```bash
streamlit run backtest_app.py
```

In sidebar, check "Use Pretrained Market Embeddings"

### 3. Compare Results

Run backtest with and without pretrained embeddings to see improvement:
- **Without**: Original features only
- **With**: Original + 128-dimensional market embeddings

## Benefits

### 1. General Market Knowledge
Pretrained on diverse stocks across sectors:
- Tech (FAANG+, NVIDIA, Tesla)
- Finance (JPM, Bank of America, Goldman Sachs)
- Healthcare (UnitedHealth, Johnson & Johnson)
- Energy (Exxon, Chevron)
- Materials, Utilities, Real Estate, Consumer staples

### 2. Better Predictions for Small Stocks
Stocks with limited history benefit from learned market patterns

### 3. Faster Backtesting
No model training during backtest - just embedding extraction

### 4. RTX 3090 Optimization
- Batch size: 256
- Gradient clipping
- Early stopping
- GPU memory: 24GB available

## Model Configuration

### LSTM Architecture
- Input: ~50 features (fundamental + technical + macro + volatility + lag)
- Hidden: 256 dimensions
- Layers: 2
- Dropout: 0.3
- Embedding: 128 dimensions

### Training Parameters
- Epochs: 50 (with early stopping)
- Batch size: 256
- Learning rate: 0.001
- Loss: Cross-entropy (weighted)
- Optimizer: Adam

## Performance Expectations

Based on transfer learning principles:

### Without Pretrained
- Relies on single stock's patterns
- Limited by individual stock's history
- May miss broader market signals

### With Pretrained  
- Leverages patterns from 80+ stocks
- Captures cross-stock relationships
- Better generalization
- Expected improvement: 5-15% Sharpe ratio increase

## Troubleshooting

### Out of Memory
Reduce batch size in `pretrain_config.py`:
```python
'batch_size': 128,  # or 64
```

### Slow Training
Use fewer stocks or epochs for testing:
```python
MAJOR_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']  # Test with 5
'num_epochs': 10  # Reduce epochs
```

### Model Not Found
Ensure pretraining completed first:
```bash
ls models/pretrained_lstm.pth
```

## GPU Details

**Detected**: NVIDIA GeForce RTX 3090
- VRAM: 24GB
- Compute Capability: 8.6
- CUDA Cores: 10496

Optimized for:
- Batch processing
- Mixed precision training (future enhancement)
- Multi-GPU support (future enhancement)

## Next Steps

1. Run full pretraining: `python pretrain_model.py`
2. Test backtest with embeddings in GUI
3. Compare performance metrics
4. Fine-tune hyperparameters based on results

## Technical Notes

### Why LSTM for Pretraining?
- Handles sequential dependencies
- Learns temporal patterns
- Generates compact embeddings
- Proven for time series

### Why LightGBM for Final Model?
- Faster training than deep learning
- Handles tabular data well
- Explainable feature importance
- Combines well with embeddings

### Embedding Benefits
- Compresses temporal patterns into 128 dims
- Captures market "wisdom"
- Transferable to any stock
- No retraining needed per stock

#!/bin/bash

# Fix cuDNN version incompatibility for the frontend as well
unset LD_LIBRARY_PATH

echo "Starting Stock Prediction & Backtesting Engine..."
source venv/bin/activate
streamlit run backtest_app.py

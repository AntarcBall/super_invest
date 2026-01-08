#!/bin/bash
# Script to run the improved pretraining with better parameters

echo "Starting improved pretraining with enhanced configuration..."

# Remove old checkpoints to start fresh
rm -rf models/checkpoints/
rm -f models/pretrained_lstm.pth
rm -f models/pretrained_config.json

# Create necessary directories
mkdir -p models/checkpoints

# Run the pretraining with the improved model
python pretrain_model.py

echo "Pretraining completed!"
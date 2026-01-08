#!/bin/bash

unset LD_LIBRARY_PATH

echo "Starting pretraining with cuDNN fix..."
source venv/bin/activate
python pretrain_model.py

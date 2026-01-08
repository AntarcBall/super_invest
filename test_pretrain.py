"""
Quick Test - Train on 5 stocks to verify pretraining pipeline
"""

import sys
sys.path.insert(0, '/home/car/stock')

from src.config.pretrain_config import PRETRAIN_CONFIG

TEST_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

config = PRETRAIN_CONFIG.copy()
config['num_epochs'] = 5
config['batch_size'] = 128

import os
os.environ['TEST_MODE'] = '1'

import importlib
import src.pretrain_model as pretrain
importlib.reload(pretrain)

pretrain.MAJOR_STOCKS = TEST_STOCKS
pretrain.PRETRAIN_CONFIG = config

print("Testing pretraining on 5 stocks...")
print("Stocks:", TEST_STOCKS)
print()

pretrain.train_pretrained_model()

print("\nTest complete! If successful, run full pretraining:")
print("python pretrain_model.py")

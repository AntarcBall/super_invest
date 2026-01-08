"""
Pretraining Configuration for Transfer Learning

Defines major stocks, training parameters, and model settings for
pretraining on large datasets using RTX 3090 GPU.
"""

MAJOR_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'NVDA', 'TSLA', 'AMD', 'INTC', 'CSCO', 'QCOM', 'IBM', 'ORCL', 'ADBE', 'CRM',
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'C', 'AXP',
    'UNH', 'JNJ', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'LLY', 'DHR',
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'DG',
    'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'MDLZ', 'CL',
    'CAT', 'HON', 'GE', 'UPS', 'RTX', 'BA', 'DE', 'LMT', 'MMM',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PXD', 'PSX',
    'LIN', 'APD', 'BHP', 'FCX', 'SHW', 'DOW', 'DD', 'NEM',
    'PLD', 'AMT', 'CCI', 'EQIX', 'PRO', 'DLR', 'O', 'SPG',
    'NEE', 'DUK', 'D', 'SO', 'EXC', 'AEP', 'SRE', 'XEL',
]

PRETRAIN_CONFIG = {
    'start_date': '2019-01-01',
    'end_date': '2024-12-31',

    'train_split': 0.8,

    'sequence_length': 20,

    'batch_size': 128,  # Reduced batch size for better generalization

    'num_epochs': 100,  # Increased epochs for more training

    'learning_rate': 0.001,

    'lstm_config': {
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.2,  # Reduced dropout for better learning
        'embedding_dim': 128,
    },

    'early_stopping_patience': 15,  # Increased patience
    'gradient_clip': 5.0,  # Increased gradient clipping threshold

    'use_gpu': True,
    'device': 'cuda',

    'force_cpu': False,
    'model_save_path': 'models/pretrained_lstm.pth',
    'checkpoint_dir': 'models/checkpoints',
    'config_save_path': 'models/pretrained_config.json',
}

INFERENCE_CONFIG = {
    'batch_size': 128,

    'sequence_length': 20,

    'use_pretrained': True,

    'model_load_path': 'models/pretrained_lstm.pth',

    'embedding_output_dim': 128,
}

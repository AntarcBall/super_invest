"""
Pretraining Script for Transfer Learning

Trains LSTM model on multiple major stocks to learn general market patterns.
Uses RTX 3090 GPU for efficient training.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training

import sys
sys.path.insert(0, '/home/car/stock')

from src.data.data_loader import get_stock_data
from src.features.feature_builder import (
    add_fundamental_features, add_technical_features,
    add_macro_features, add_volatility_features, add_acf_ccf_lagged_features
)
from src.models.prepare_data import prepare_training_data

from src.config.pretrain_config import MAJOR_STOCKS, PRETRAIN_CONFIG


class MultiStockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.LongTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class PretrainLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3, embedding_dim=128):
        super(PretrainLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Add batch normalization for more stable training
        self.input_norm = nn.BatchNorm1d(input_dim)

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Use bidirectional LSTM for better context
        )

        # Since we're using bidirectional LSTM, the output size is doubled
        self.lstm_output_dim = hidden_dim * 2

        # Add a few more layers for better feature extraction
        self.intermediate_layer = nn.Linear(self.lstm_output_dim, hidden_dim)
        self.intermediate_norm = nn.BatchNorm1d(hidden_dim)
        self.intermediate_activation = nn.ReLU()

        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)
        self.embedding_norm = nn.BatchNorm1d(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 2)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply batch norm to input (reshape for batch norm)
        batch_size, seq_len, features = x.size()
        x = x.contiguous().view(-1, features)
        x = self.input_norm(x)
        x = x.view(batch_size, seq_len, features)

        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Get the last time step

        # Process through intermediate layer
        last_hidden = self.intermediate_layer(last_hidden)
        last_hidden = last_hidden.permute(0, 1)  # For batch norm
        last_hidden = self.intermediate_norm(last_hidden)
        last_hidden = last_hidden.permute(0, 1)  # Reshape back
        last_hidden = self.intermediate_activation(last_hidden)
        last_hidden = self.dropout(last_hidden)

        # Generate embeddings
        embeddings = self.embedding_layer(last_hidden)
        embeddings = embeddings.permute(0, 1)  # For batch norm
        embeddings = self.embedding_norm(embeddings)
        embeddings = embeddings.permute(0, 1)  # Reshape back
        embeddings = self.dropout(embeddings)

        output = self.classifier(embeddings)

        return output, embeddings

    def get_embeddings(self, x):
        # Apply batch norm to input (reshape for batch norm)
        batch_size, seq_len, features = x.size()
        x = x.contiguous().view(-1, features)
        x = self.input_norm(x)
        x = x.view(batch_size, seq_len, features)

        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Get the last time step

        # Process through intermediate layer
        last_hidden = self.intermediate_layer(last_hidden)
        last_hidden = last_hidden.permute(0, 1)  # For batch norm
        last_hidden = self.intermediate_norm(last_hidden)
        last_hidden = last_hidden.permute(0, 1)  # Reshape back
        last_hidden = self.intermediate_activation(last_hidden)
        last_hidden = self.dropout(last_hidden)

        # Generate embeddings
        embeddings = self.embedding_layer(last_hidden)
        embeddings = embeddings.permute(0, 1)  # For batch norm
        embeddings = self.embedding_norm(embeddings)
        embeddings = embeddings.permute(0, 1)  # Reshape back
        embeddings = self.dropout(embeddings)

        return embeddings


def load_and_preprocess_stock(ticker, start_date, end_date, horizon=5, threshold=0.03):
    try:
        data = get_stock_data(ticker, start_date, end_date)

        if data.empty or len(data) < 100:
            print(f"Skipping {ticker}: insufficient data")
            return None, None

        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        data = add_fundamental_features(data, ticker)
        data = add_technical_features(data)
        data = add_macro_features(data)
        data = add_volatility_features(data)

        X, y = prepare_training_data(data, horizon=horizon, threshold=threshold)

        if len(X) < 50:
            print(f"Skipping {ticker}: insufficient samples after feature engineering")
            return None, None

        return X, y

    except Exception as e:
        print(f"Error loading {ticker}: {e}")
        return None, None


def create_sequences(X, y, seq_length):
    sequences = []
    targets = []

    X_values = X.values if isinstance(X, pd.DataFrame) else X

    for i in range(len(X_values) - seq_length):
        sequences.append(X_values[i:i + seq_length])
        targets.append(y.iloc[i + seq_length] if isinstance(y, pd.Series) else y[i + seq_length])

    return np.array(sequences), np.array(targets)


def train_pretrained_model():
    config = PRETRAIN_CONFIG

    use_gpu = config['use_gpu'] and not config['force_cpu']

    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("Warning: Using CPU (GPU unavailable or disabled)")
        print("Training will be slower. Use GPU for faster training.")

    print(f"Using device: {device}")
    print(f"Training on {len(MAJOR_STOCKS)} stocks from {config['start_date']} to {config['end_date']}")

    all_sequences = []
    all_targets = []

    successful_stocks = []

    for ticker in tqdm(MAJOR_STOCKS, desc="Loading stocks"):
        X, y = load_and_preprocess_stock(
            ticker, config['start_date'], config['end_date'],
            horizon=10, threshold=0.02
        )

        if X is not None and y is not None:
            seqs, targs = create_sequences(X, y, config['sequence_length'])

            if len(seqs) > 0:
                all_sequences.append(seqs)
                all_targets.append(targs)
                successful_stocks.append(ticker)

    if len(all_sequences) == 0:
        print("ERROR: No stocks loaded successfully")
        return None

    print(f"\nSuccessfully loaded {len(successful_stocks)} stocks")

    X_all = np.concatenate(all_sequences, axis=0)
    y_all = np.concatenate(all_targets, axis=0)

    print(f"Total sequences: {len(X_all):,}")
    print(f"Total samples: {len(y_all):,}")
    print(f"Feature dimension: {X_all.shape[2]}")

    dataset = MultiStockDataset(X_all, y_all)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,  # Increased workers for faster data loading
        pin_memory=True,  # Enable faster GPU transfer
        persistent_workers=True  # Keep workers alive between epochs
    )

    input_dim = X_all.shape[2]
    model = PretrainLSTM(
        input_dim=input_dim,
        hidden_dim=config['lstm_config']['hidden_dim'],
        num_layers=config['lstm_config']['num_layers'],
        dropout=config['lstm_config']['dropout'],
        embedding_dim=config['lstm_config']['embedding_dim']
    ).to(device)

    # Use a more sophisticated optimizer with cyclical learning rates
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5, betas=(0.9, 0.999))

    # Add learning rate scheduler - cyclical learning rate schedule
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=config['learning_rate']/10,
        max_lr=config['learning_rate']*10,
        step_size_up=5,  # epochs to go from base to max
        mode='triangular2',  # reduces max_lr by half each cycle
        cycle_momentum=False  # disable momentum cycling
    )

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    criterion = nn.CrossEntropyLoss()

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"DEBUG: Config keys: {config.keys()}")
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    start_epoch = 0
    best_loss = float('inf')

    checkpoint_files = [f for f in os.listdir(config['checkpoint_dir']) if f.endswith('.pth')]
    if checkpoint_files:
        latest_checkpoint = max([os.path.join(config['checkpoint_dir'], f) for f in checkpoint_files], key=os.path.getmtime)
        print(f"Found existing checkpoint: {latest_checkpoint}. Resuming...")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}")

    patience_counter = 0

    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_seqs, batch_targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", leave=False):
            batch_seqs = batch_seqs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()

            # Use autocast for mixed precision
            with autocast():
                outputs, _ = model(batch_seqs)
                loss = criterion(outputs, batch_targets)

            # Scale loss and backpropagate
            scaler.scale(loss).backward()

            if config['gradient_clip'] > 0:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])

            # Optimizer step with scaler
            scaler.step(optimizer)

            # Update the scale for next iteration
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{config['num_epochs']} - Loss: {avg_loss:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Step the scheduler - for CyclicLR, we step after each epoch
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  -> Saved regular checkpoint: {checkpoint_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': best_loss,
                'input_dim': input_dim,
                'config': config
            }, config['model_save_path'])

            print(f"  -> Saved best model (loss: {best_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    save_config = {
        'successful_stocks': successful_stocks,
        'total_stocks_trained': len(successful_stocks),
        'total_sequences': len(X_all),
        'input_dim': input_dim,
        'model_config': config['lstm_config'],
        'training_date': datetime.now().isoformat(),
        'best_loss': best_loss
    }

    with open(config['config_save_path'], 'w') as f:
        json.dump(save_config, f, indent=2)

    print(f"\nPretraining complete!")
    print(f"Model saved to: {config['model_save_path']}")
    print(f"Config saved to: {config['config_save_path']}")
    print(f"Trained on {len(successful_stocks)} stocks")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train pretrained LSTM model on major stocks')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    args = parser.parse_args()

    if args.cpu:
        print("CPU mode forced via --cpu flag")
        PRETRAIN_CONFIG['force_cpu'] = True

    train_pretrained_model()

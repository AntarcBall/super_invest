"""
Pretrained Model Utilities

Functions to load pretrained models and extract embeddings for inference.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json

from src.config.pretrain_config import INFERENCE_CONFIG


class PretrainedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3, embedding_dim=128):
        super(PretrainedLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        self.embedding_layer = nn.Linear(hidden_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 2)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        embeddings = self.dropout(last_hidden)
        embeddings = self.embedding_layer(embeddings)
        output = self.classifier(embeddings)

        return output, embeddings

    def get_embeddings(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        embeddings = self.dropout(last_hidden)
        embeddings = self.embedding_layer(embeddings)
        return embeddings


def load_pretrained_model(model_path, config_path=None, device='cuda'):
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    print(f"Loading pretrained model on {device}")

    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
        input_dim = config.get('input_dim', 50)
        model_config = config.get('model_config', {})
        lstm_config = model_config if model_config else {}
    else:
        input_dim = 50
        lstm_config = {}

    model = PretrainedLSTM(
        input_dim=input_dim,
        hidden_dim=lstm_config.get('hidden_dim', 256),
        num_layers=lstm_config.get('num_layers', 2),
        dropout=lstm_config.get('dropout', 0.3),
        embedding_dim=lstm_config.get('embedding_dim', 128)
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    print(f"Model loaded successfully")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Hidden dimension: {lstm_config.get('hidden_dim', 256)}")
    print(f"  - Embedding dimension: {lstm_config.get('embedding_dim', 128)}")

    return model, device


def extract_embeddings_from_stock(model, X, sequence_length=20, device='cuda', batch_size=128):
    X_values = X.values if isinstance(X, pd.DataFrame) else X

    if len(X_values) < sequence_length:
        print(f"Warning: Insufficient data for embeddings (need {sequence_length}, got {len(X_values)})")
        return None

    sequences = []
    for i in range(len(X_values) - sequence_length + 1):
        sequences.append(X_values[i:i + sequence_length])

    if len(sequences) == 0:
        print("Warning: No sequences created")
        return None

    sequences = np.array(sequences, dtype=np.float32)
    sequences_tensor = torch.FloatTensor(sequences).to(device)

    model.eval()
    embeddings_list = []

    with torch.no_grad():
        for i in range(0, len(sequences_tensor), batch_size):
            batch = sequences_tensor[i:i + batch_size]
            embeddings = model.get_embeddings(batch)
            embeddings_list.append(embeddings.cpu().numpy())

    embeddings_all = np.vstack(embeddings_list)

    embedding_df = pd.DataFrame(
        embeddings_all,
        index=X.index[sequence_length:sequence_length + len(embeddings_all)],
        columns=[f'pretrained_emb_{i}' for i in range(embeddings_all.shape[1])]
    )

    return embedding_df


def add_pretrained_embeddings(X, model_path, config_path=None, sequence_length=20):
    try:
        model, device = load_pretrained_model(model_path, config_path)
        embedding_df = extract_embeddings_from_stock(model, X, sequence_length, device)

        if embedding_df is None:
            print("Could not extract embeddings")
            return X

        X_with_embeddings = pd.concat([X.iloc[sequence_length:sequence_length + len(embedding_df)], embedding_df], axis=1)

        print(f"Added {len(embedding_df.columns)} pretrained embedding features")
        print(f"New shape: {X_with_embeddings.shape}")

        return X_with_embeddings

    except Exception as e:
        print(f"Error adding pretrained embeddings: {e}")
        return X


if __name__ == '__main__':
    from src.data.data_loader import get_stock_data
    from src.features.feature_builder import add_fundamental_features, add_technical_features, add_macro_features, add_volatility_features
    from src.models.prepare_data import prepare_training_data

    print("Testing pretrained embedding extraction...")

    data = get_stock_data('AAPL', '2023-01-01', '2024-01-01')
    data = add_fundamental_features(data, 'AAPL')
    data = add_technical_features(data)
    data = add_macro_features(data)
    data = add_volatility_features(data)

    X, y = prepare_training_data(data, horizon=5, threshold=0.02)

    print(f"Original X shape: {X.shape}")

    X_with_embeddings = add_pretrained_embeddings(
        X,
        model_path='models/pretrained_lstm.pth',
        config_path='models/pretrained_config.json'
    )

    if X_with_embeddings is not None:
        print(f"X with embeddings shape: {X_with_embeddings.shape}")
        print("\nSuccess!")

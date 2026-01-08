import lightgbm as lgb
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from src.models.prepare_data import prepare_training_data
from src.features.feature_builder import add_fundamental_features, add_technical_features, add_macro_features
from src.data.data_loader import get_stock_data
from src.models.lstm_model import train_lstm_model, create_sequences

def train_lgbm_model(X_train, y_train, X_test, y_test):
    """
    Trains a LightGBM model and evaluates its performance.
    """
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Initialize and train the model
    model = lgb.LGBMClassifier(random_state=42, verbose=-1)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    
    print("\n--- LightGBM Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

def extract_lstm_embeddings(model, X, seq_length, device):
    """
    Helper to extract embeddings for a dataset using the LSTM model.
    """
    model.eval()
    embeddings = []
    
    # Create sequences (this truncates the first seq_length rows)
    X_seq, _ = create_sequences(X, pd.Series(np.zeros(len(X))), seq_length) # Dummy y
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # Process in batches to avoid OOM if dataset is huge, but here full batch is likely fine
        batch_embeddings = model.get_embeddings(X_tensor)
        embeddings.append(batch_embeddings.cpu().numpy())
        
    return np.vstack(embeddings)

def train_hybrid_model(X_train, y_train, X_test, y_test, seq_length=10):
    """
    Trains a hybrid LSTM-LightGBM model.
    """
    print("\n--- Starting Hybrid Model Training ---")
    
    # 1. Train LSTM
    lstm_model = train_lstm_model(X_train, y_train, X_test, y_test, seq_length=seq_length, epochs=20)
    
    # 2. Extract Embeddings
    device = next(lstm_model.parameters()).device
    
    print("Extracting LSTM embeddings...")
    train_embeddings = extract_lstm_embeddings(lstm_model, X_train, seq_length, device)
    test_embeddings = extract_lstm_embeddings(lstm_model, X_test, seq_length, device)
    
    # 3. Align Data (Trim original data to match sequence length)
    # The LSTM consumes the first 'seq_length' rows, so we must drop them from X and y
    X_train_trimmed = X_train.iloc[seq_length:]
    y_train_trimmed = y_train.iloc[seq_length:]
    X_test_trimmed = X_test.iloc[seq_length:]
    y_test_trimmed = y_test.iloc[seq_length:]
    
    # 4. Combine Features
    # We create new DataFrames with the embeddings
    train_embed_df = pd.DataFrame(train_embeddings, index=X_train_trimmed.index)
    test_embed_df = pd.DataFrame(test_embeddings, index=X_test_trimmed.index)
    
    # Rename columns to avoid collision
    train_embed_df.columns = [f'lstm_emb_{i}' for i in range(train_embed_df.shape[1])]
    test_embed_df.columns = [f'lstm_emb_{i}' for i in range(test_embed_df.shape[1])]
    
    X_train_hybrid = pd.concat([X_train_trimmed, train_embed_df], axis=1)
    X_test_hybrid = pd.concat([X_test_trimmed, test_embed_df], axis=1)
    
    print(f"Hybrid Train Shape: {X_train_hybrid.shape}")
    
    # 5. Train LightGBM on Hybrid Data
    lgbm_model = train_lgbm_model(X_train_hybrid, y_train_trimmed, X_test_hybrid, y_test_trimmed)
    
    return lgbm_model, lstm_model

if __name__ == '__main__':
    TICKER = 'MSFT'
    START = '2022-01-01'
    END = '2024-01-01'
    
    data = get_stock_data(TICKER, START, END)
    
    if not data.empty:
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        data = add_fundamental_features(data, TICKER)
        data = add_technical_features(data)
        data = add_macro_features(data)
        
        X, y = prepare_training_data(data, horizon=5, threshold=0.02)
        
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Test basic LGBM
        print("--- Testing Basic LightGBM ---")
        train_lgbm_model(X_train, y_train, X_test, y_test)
        
        # Test Hybrid
        print("\n--- Testing Hybrid Model ---")
        train_hybrid_model(X_train, y_train, X_test, y_test)

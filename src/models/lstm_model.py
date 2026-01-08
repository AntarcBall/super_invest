import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

class LSTMModel(nn.Module):
    """
    A simple LSTM model for binary classification.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # We need to detach as we are doing truncated backprop through time (BPTT)
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

    def get_embeddings(self, x):
        """
        Extracts the hidden state (embeddings) from the LSTM.
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        # Return the last hidden state of the last layer
        # Shape: (batch_size, hidden_dim)
        return hn[-1]

def create_sequences(X, y, seq_length):
    """
    Creates sequences of data for LSTM training.
    """
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        x_block = X.iloc[i:(i + seq_length)].values
        y_block = y.iloc[i + seq_length]
        xs.append(x_block)
        ys.append(y_block)
    return np.array(xs), np.array(ys)

def train_lstm_model(X_train, y_train, X_test, y_test, seq_length=10, epochs=50, hidden_dim=50, num_layers=1, lr=0.001):
    """
    Trains the LSTM model.
    """
    # Force CPU to avoid cuDNN version mismatch
    device = torch.device('cpu')
    print(f"Training on device: {device}")

    # Prepare Data
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)

    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(1).to(device)

    # Initialize Model
    input_dim = X_train.shape[1]
    model = LSTMModel(input_dim, hidden_dim, num_layers).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    model.train()
    for epoch in range(epochs):
        outputs = model(X_train_tensor)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_prob = model(X_test_tensor)
        y_pred = (y_pred_prob > 0.5).float()
        
        # Move back to CPU for metrics
        y_test_cpu = y_test_tensor.cpu().numpy()
        y_pred_cpu = y_pred.cpu().numpy()
        
        acc = accuracy_score(y_test_cpu, y_pred_cpu)
        print(f"\nLSTM Model Accuracy: {acc:.4f}")
        print(classification_report(y_test_cpu, y_pred_cpu))

    return model

if __name__ == '__main__':
    # Test the LSTM implementation
    from src.data.data_loader import get_stock_data
    from src.features.feature_builder import add_fundamental_features, add_technical_features
    from src.models.prepare_data import prepare_training_data
    
    print("Testing LSTM Model...")
    data = get_stock_data('AAPL', '2023-01-01', '2024-01-01')
    if not data.empty:
        # Minimal feature set for testing
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        data = add_fundamental_features(data, 'AAPL')
        data = add_technical_features(data)
        data.fillna(0, inplace=True)
        
        X, y = prepare_training_data(data)
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        model = train_lstm_model(X_train, y_train, X_test, y_test, epochs=10)
        print("LSTM Test Complete.")

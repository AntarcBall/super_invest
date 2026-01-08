#!/usr/bin/env python3
"""
Simple test for the improved model architecture
"""

import torch
import torch.nn as nn

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

def test_model_architecture():
    """Test the improved model architecture"""
    print("Testing improved model architecture...")
    
    # Create a sample input
    batch_size = 32
    seq_length = 20
    input_dim = 26  # Based on the logs
    
    # Initialize the model
    model = PretrainLSTM(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=2,
        dropout=0.2,
        embedding_dim=128
    )
    
    # Create sample input tensor
    x = torch.randn(batch_size, seq_length, input_dim)
    
    # Test forward pass
    try:
        output, embeddings = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Check if shapes are as expected
        assert output.shape == (batch_size, 2), f"Expected output shape (32, 2), got {output.shape}"
        assert embeddings.shape == (batch_size, 128), f"Expected embeddings shape (32, 128), got {embeddings.shape}"
        
        print("✓ Model architecture test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model architecture test failed: {e}")
        return False

def test_get_embeddings():
    """Test the get_embeddings method"""
    print("\nTesting get_embeddings method...")
    
    batch_size = 16
    seq_length = 10
    input_dim = 26
    
    model = PretrainLSTM(input_dim, hidden_dim=128, embedding_dim=64)
    x = torch.randn(batch_size, seq_length, input_dim)
    
    try:
        embeddings = model.get_embeddings(x)
        print(f"Embeddings shape: {embeddings.shape}")
        assert embeddings.shape == (batch_size, 64), f"Expected (16, 64), got {embeddings.shape}"
        print("✓ get_embeddings test passed!")
        return True
    except Exception as e:
        print(f"✗ get_embeddings test failed: {e}")
        return False

def test_model_parameters():
    """Test model parameter count"""
    print("\nTesting model parameters...")
    
    model = PretrainLSTM(input_dim=26, hidden_dim=256, num_layers=2, embedding_dim=128)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {param_count:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # The model should have significantly more parameters due to bidirectional LSTM
    assert param_count > 500000, f"Expected more parameters, got {param_count}"
    print("✓ Parameter count test passed!")
    return True

if __name__ == "__main__":
    print("Running tests for improved model...")
    
    tests = [
        test_model_architecture,
        test_get_embeddings,
        test_model_parameters
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! The improved model is ready.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
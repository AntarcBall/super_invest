#!/usr/bin/env python3
"""
Test script for the improved pretraining model
"""

import sys
sys.path.insert(0, '/home/car/stock')

import torch
import torch.nn as nn
import numpy as np
from pretrain_model import PretrainLSTM

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
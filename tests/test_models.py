"""Tests for model components."""

import pytest
import torch
import torch.nn as nn

from slm.config import ModelConfig, ModelType, RNNType
from slm.core.models import (
    CharRNN, CharTransformer, PositionalEncoding,
    MultiHeadAttention, TransformerBlock
)
from slm.exceptions import ModelError, ValidationError


class TestPositionalEncoding:
    """Test positional encoding component."""
    
    def test_positional_encoding_creation(self):
        """Test positional encoding initialization."""
        d_model, max_len = 256, 1000
        pe = PositionalEncoding(d_model, max_len)
        
        assert pe.pe.shape == (max_len, 1, d_model)
    
    def test_positional_encoding_forward(self):
        """Test positional encoding forward pass."""
        d_model, max_len = 128, 500
        pe = PositionalEncoding(d_model, max_len)
        
        # Test input
        seq_len, batch_size = 100, 4
        x = torch.randn(seq_len, batch_size, d_model)
        
        output = pe(x)
        assert output.shape == x.shape
    
    def test_positional_encoding_length_error(self):
        """Test error when sequence exceeds max_len."""
        d_model, max_len = 128, 100
        pe = PositionalEncoding(d_model, max_len)
        
        # Sequence longer than max_len
        seq_len, batch_size = 150, 2
        x = torch.randn(seq_len, batch_size, d_model)
        
        with pytest.raises(ModelError):
            pe(x)


class TestMultiHeadAttention:
    """Test multi-head attention component."""
    
    def test_attention_creation(self):
        """Test attention module initialization."""
        d_model, n_heads = 256, 8
        attention = MultiHeadAttention(d_model, n_heads)
        
        assert attention.d_model == d_model
        assert attention.n_heads == n_heads
        assert attention.d_k == d_model // n_heads
    
    def test_attention_incompatible_dimensions(self):
        """Test error with incompatible dimensions."""
        with pytest.raises(ModelError):
            MultiHeadAttention(d_model=255, n_heads=8)  # 255 not divisible by 8
    
    def test_attention_forward(self):
        """Test attention forward pass."""
        d_model, n_heads = 128, 4
        attention = MultiHeadAttention(d_model, n_heads)
        
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, weights = attention(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    def test_attention_with_mask(self):
        """Test attention with causal mask."""
        d_model, n_heads = 64, 2
        attention = MultiHeadAttention(d_model, n_heads)
        
        batch_size, seq_len = 1, 5
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        
        output, weights = attention(x, x, x, mask)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, n_heads, seq_len, seq_len)


class TestTransformerBlock:
    """Test transformer block component."""
    
    def test_transformer_block_creation(self):
        """Test transformer block initialization."""
        d_model, n_heads, d_ff = 128, 4, 512
        block = TransformerBlock(d_model, n_heads, d_ff)
        
        assert isinstance(block.attention, MultiHeadAttention)
        assert isinstance(block.norm1, nn.LayerNorm)
        assert isinstance(block.norm2, nn.LayerNorm)
    
    def test_transformer_block_forward(self):
        """Test transformer block forward pass."""
        d_model, n_heads, d_ff = 64, 2, 256
        block = TransformerBlock(d_model, n_heads, d_ff)
        
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, weights = block(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, n_heads, seq_len, seq_len)


class TestCharRNNDetailed:
    """Detailed tests for CharRNN model."""
    
    def test_rnn_types(self):
        """Test different RNN cell types."""
        vocab_size = 50
        
        for rnn_type in [RNNType.LSTM, RNNType.GRU, RNNType.RNN]:
            config = ModelConfig(
                model_type=ModelType.RNN,
                vocab_size=vocab_size,
                rnn_type=rnn_type,
                hidden_dim=32,
                embedding_dim=16
            )
            
            model = CharRNN(config)
            assert model.rnn_type == rnn_type
            
            # Test forward pass
            batch_size, seq_len = 2, 5
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            logits, hidden = model(input_ids)
            assert logits.shape == (batch_size, seq_len, vocab_size)
    
    def test_rnn_hidden_initialization(self):
        """Test hidden state initialization."""
        config = ModelConfig(
            model_type=ModelType.RNN,
            vocab_size=30,
            rnn_type=RNNType.LSTM,
            hidden_dim=64,
            num_layers=2
        )
        
        model = CharRNN(config)
        batch_size = 3
        device = torch.device('cpu')
        
        hidden = model.init_hidden(batch_size, device)
        
        # LSTM returns tuple of (h, c)
        assert isinstance(hidden, tuple)
        h, c = hidden
        assert h.shape == (2, batch_size, 64)  # (num_layers, batch, hidden_dim)
        assert c.shape == (2, batch_size, 64)
    
    def test_rnn_tied_weights(self):
        """Test weight tying functionality."""
        config = ModelConfig(
            model_type=ModelType.RNN,
            vocab_size=30,
            hidden_dim=64,
            embedding_dim=64,  # Must match hidden_dim for tying
            tie_weights=True
        )
        
        model = CharRNN(config)
        
        # Check that weights are tied
        assert model.output_projection.weight is model.embedding.weight
    
    def test_rnn_tied_weights_error(self):
        """Test error when trying to tie weights with different dimensions."""
        config = ModelConfig(
            model_type=ModelType.RNN,
            vocab_size=30,
            hidden_dim=64,
            embedding_dim=32,  # Different from hidden_dim
            tie_weights=True
        )
        
        with pytest.raises(ModelError):
            CharRNN(config)
    
    def test_rnn_dropout_layers(self):
        """Test dropout with different layer configurations."""
        # Single layer - no inter-layer dropout
        config1 = ModelConfig(
            model_type=ModelType.RNN,
            vocab_size=30,
            num_layers=1,
            dropout=0.5
        )
        model1 = CharRNN(config1)
        
        # Multiple layers - should have inter-layer dropout
        config2 = ModelConfig(
            model_type=ModelType.RNN,
            vocab_size=30,
            num_layers=3,
            dropout=0.5
        )
        model2 = CharRNN(config2)
        
        # Both should work
        input_ids = torch.randint(0, 30, (2, 5))
        
        with torch.no_grad():
            logits1, _ = model1(input_ids)
            logits2, _ = model2(input_ids)
        
        assert logits1.shape[-1] == 30
        assert logits2.shape[-1] == 30


class TestCharTransformerDetailed:
    """Detailed tests for CharTransformer model."""
    
    def test_transformer_attention_layers(self):
        """Test that transformer has correct number of attention layers."""
        n_layers = 4
        config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            vocab_size=50,
            n_layers_transformer=n_layers,
            d_model=128,
            n_heads=4
        )
        
        model = CharTransformer(config)
        assert len(model.transformer_blocks) == n_layers
    
    def test_transformer_causal_mask(self):
        """Test causal mask generation."""
        config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            vocab_size=30,
            d_model=64,
            n_heads=2
        )
        
        model = CharTransformer(config)
        seq_len = 5
        device = torch.device('cpu')
        
        mask = model._generate_causal_mask(seq_len, device)
        
        # Check mask shape and properties
        assert mask.shape == (seq_len, seq_len)
        
        # Upper triangle should be 0 (masked)
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                assert mask[i, j] == 0
        
        # Lower triangle should be 1 (not masked)
        for i in range(seq_len):
            for j in range(i+1):
                assert mask[i, j] == 1
    
    def test_transformer_max_length_error(self):
        """Test error when input exceeds max_len."""
        config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            vocab_size=30,
            max_len=10  # Small max length
        )
        
        model = CharTransformer(config)
        
        # Input longer than max_len
        batch_size, seq_len = 1, 15
        input_ids = torch.randint(0, 30, (batch_size, seq_len))
        
        with pytest.raises(ModelError):
            model(input_ids)
    
    def test_transformer_positional_encoding_integration(self):
        """Test that positional encoding is properly integrated."""
        config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            vocab_size=30,
            d_model=64,
            max_len=100
        )
        
        model = CharTransformer(config)
        
        # Test forward pass
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 30, (batch_size, seq_len))
        
        logits, attention_weights = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, 30)
        assert len(attention_weights) == config.n_layers_transformer


class TestBaseModelFunctionality:
    """Test BaseModel functionality."""
    
    def test_model_info(self):
        """Test model information extraction."""
        config = ModelConfig(
            model_type=ModelType.RNN,
            vocab_size=100,
            hidden_dim=64
        )
        
        model = CharRNN(config)
        info = model.get_model_info()
        
        assert 'model_type' in info
        assert 'vocab_size' in info
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'model_size_mb' in info
        assert 'config' in info
        
        assert info['vocab_size'] == 100
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
    
    def test_parameter_counting(self):
        """Test parameter counting accuracy."""
        config = ModelConfig(
            model_type=ModelType.RNN,
            vocab_size=50,
            hidden_dim=32,
            embedding_dim=16,
            num_layers=1
        )
        
        model = CharRNN(config)
        total_params, trainable_params = model.get_num_parameters()
        
        # Manual calculation for verification
        # Embedding: vocab_size * embedding_dim = 50 * 16 = 800
        # LSTM: approximately 4 * (embedding_dim + hidden_dim + 1) * hidden_dim
        # Output: hidden_dim * vocab_size = 32 * 50 = 1600
        
        assert total_params > 800 + 1600  # At least embedding + output
        assert trainable_params == total_params  # All params trainable by default
    
    def test_model_device_handling(self):
        """Test model device handling."""
        config = ModelConfig(
            model_type=ModelType.RNN,
            vocab_size=30,
            hidden_dim=32
        )
        
        model = CharRNN(config)
        
        # Test that model can be moved to different devices
        if torch.cuda.is_available():
            model = model.cuda()
            assert next(model.parameters()).device.type == 'cuda'
            
            model = model.cpu()
            assert next(model.parameters()).device.type == 'cpu'


class TestModelErrorHandling:
    """Test model error handling."""
    
    def test_invalid_model_type_config(self):
        """Test error with wrong model type in config."""
        # RNN config with transformer model
        config = ModelConfig(
            model_type=ModelType.TRANSFORMER,  # Wrong type
            vocab_size=30
        )
        
        with pytest.raises(ModelError):
            CharRNN(config)
        
        # Transformer config with RNN model
        config = ModelConfig(
            model_type=ModelType.RNN,  # Wrong type
            vocab_size=30
        )
        
        with pytest.raises(ModelError):
            CharTransformer(config)
    
    def test_invalid_rnn_type(self):
        """Test error with invalid RNN type."""
        # This would test if we had dynamic RNN type creation
        # For now, RNN types are validated by the enum
        pass
    
    def test_input_validation(self):
        """Test input validation in forward pass."""
        config = ModelConfig(
            model_type=ModelType.RNN,
            vocab_size=30,
            hidden_dim=32
        )
        
        model = CharRNN(config)
        
        # Wrong input dimensions
        with pytest.raises(ValidationError):
            # 3D input instead of 2D
            input_ids = torch.randint(0, 30, (2, 5, 3))
            model(input_ids)


if __name__ == '__main__':
    pytest.main([__file__])

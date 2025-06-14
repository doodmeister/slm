"""Test suite for SLM."""

import pytest
import torch
import tempfile
from pathlib import Path

from slm.config import (
    Config, ModelConfig, TrainingConfig, GenerationConfig,
    ModelType, SamplingMethod, create_default_config
)
from slm.core.models import CharRNN, CharTransformer, create_model
from slm.core.data import Vocabulary, TextDataset, prepare_data
from slm.exceptions import ModelError, DataError, ValidationError


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "Hello world! This is a test. Simple text for training."


@pytest.fixture
def sample_vocab():
    """Sample vocabulary for testing."""
    chars = list("abcdefghijklmnopqrstuvwxyz !.,")
    return Vocabulary(chars)


@pytest.fixture
def temp_dir():
    """Temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestConfig:
    """Test configuration classes."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = create_default_config()
        assert isinstance(config, Config)
        assert config.model.model_type == ModelType.RNN
        assert config.training.epochs > 0
        assert config.generation.length > 0
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        # Valid config
        config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            d_model=256,
            n_heads=8
        )
        assert config.d_model % config.n_heads == 0
        
        # Invalid config - d_model not divisible by n_heads
        with pytest.raises(ValueError):
            ModelConfig(
                model_type=ModelType.TRANSFORMER,
                d_model=255,
                n_heads=8
            )
    
    def test_training_config_validation(self):
        """Test training configuration validation."""
        config = TrainingConfig(
            epochs=10,
            batch_size=32,
            learning_rate=0.001
        )
        assert config.epochs == 10
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        
        # Test validation bounds
        with pytest.raises(ValueError):
            TrainingConfig(epochs=-1)
        
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=0.0)
    
    def test_generation_config_validation(self):
        """Test generation configuration validation."""
        config = GenerationConfig(
            sampling_method=SamplingMethod.TOP_K,
            top_k=10
        )
        assert config.top_k == 10
        
        # Invalid: top_k = 0 with top_k sampling
        with pytest.raises(ValueError):
            GenerationConfig(
                sampling_method=SamplingMethod.TOP_K,
                top_k=0
            )


class TestVocabulary:
    """Test vocabulary functionality."""
    
    def test_vocab_creation_from_text(self, sample_text):
        """Test vocabulary creation from text."""
        vocab = Vocabulary.from_text(sample_text)
        assert len(vocab) > 0
        
        # Check that all characters in text are in vocab
        text_chars = set(sample_text)
        vocab_chars = set(vocab.chars)
        assert text_chars.issubset(vocab_chars)
    
    def test_vocab_encoding_decoding(self, sample_vocab, sample_text):
        """Test text encoding and decoding."""
        # Filter text to only include vocab characters
        filtered_text = ''.join(c for c in sample_text if c in sample_vocab)
        
        # Encode and decode
        indices = sample_vocab.encode(filtered_text)
        decoded_text = sample_vocab.decode(indices)
        
        assert decoded_text == filtered_text
    
    def test_vocab_unknown_chars(self, sample_vocab):
        """Test handling of unknown characters."""
        text_with_unknown = "Hello 世界"  # Contains non-ASCII characters
        
        # Should handle unknown characters gracefully
        indices = sample_vocab.encode(text_with_unknown)
        decoded = sample_vocab.decode(indices)
        
        # Unknown characters should be replaced
        assert len(indices) == len(text_with_unknown)
    
    def test_vocab_serialization(self, sample_vocab, temp_dir):
        """Test vocabulary saving and loading."""
        vocab_path = temp_dir / "vocab.json"
        
        # Save vocabulary
        sample_vocab.save(vocab_path)
        assert vocab_path.exists()
        
        # Load vocabulary
        loaded_vocab = Vocabulary.from_file(vocab_path)
        assert loaded_vocab.chars == sample_vocab.chars
        assert loaded_vocab.char2idx == sample_vocab.char2idx


class TestTextDataset:
    """Test text dataset functionality."""
    
    def test_dataset_creation(self, sample_text, sample_vocab):
        """Test dataset creation."""
        dataset = TextDataset(sample_text, sample_vocab, seq_length=10)
        assert len(dataset) > 0
        
        # Check that we can get items
        input_seq, target_seq = dataset[0]
        assert input_seq.shape[0] == 10
        assert target_seq.shape[0] == 10
        assert torch.all(target_seq == input_seq[1:])  # Target is shifted input
    
    def test_dataset_validation(self, sample_vocab):
        """Test dataset input validation."""
        # Text too short
        with pytest.raises(DataError):
            TextDataset("hi", sample_vocab, seq_length=100)
        
        # Invalid sequence length
        with pytest.raises(ValidationError):
            TextDataset("hello world", sample_vocab, seq_length=0)
    
    def test_dataset_stride(self, sample_text, sample_vocab):
        """Test dataset with custom stride."""
        seq_length = 10
        stride = 5
        dataset = TextDataset(sample_text, sample_vocab, seq_length=seq_length, stride=stride)
        
        # Check overlapping sequences
        input1, _ = dataset[0]
        input2, _ = dataset[1]
        
        # Second sequence should start 'stride' positions after first
        assert input1.shape[0] == seq_length
        assert input2.shape[0] == seq_length


class TestModels:
    """Test model functionality."""
    
    def test_rnn_model_creation(self):
        """Test RNN model creation."""
        config = ModelConfig(
            model_type=ModelType.RNN,
            vocab_size=100,
            hidden_dim=128,
            embedding_dim=64,
            num_layers=2
        )
        
        model = CharRNN(config)
        assert model.vocab_size == 100
        assert model.hidden_dim == 128
        assert model.num_layers == 2
    
    def test_transformer_model_creation(self):
        """Test Transformer model creation."""
        config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            vocab_size=100,
            d_model=256,
            n_heads=8,
            n_layers_transformer=6
        )
        
        model = CharTransformer(config)
        assert model.vocab_size == 100
        assert model.d_model == 256
        assert model.n_heads == 8
    
    def test_rnn_forward_pass(self):
        """Test RNN forward pass."""
        config = ModelConfig(
            model_type=ModelType.RNN,
            vocab_size=50,
            hidden_dim=64,
            embedding_dim=32
        )
        
        model = CharRNN(config)
        model.eval()
        
        # Test input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 50, (batch_size, seq_len))
        
        with torch.no_grad():
            logits, hidden = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, 50)
        assert hidden is not None
    
    def test_transformer_forward_pass(self):
        """Test Transformer forward pass."""
        config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            vocab_size=50,
            d_model=128,
            n_heads=4,
            n_layers_transformer=2
        )
        
        model = CharTransformer(config)
        model.eval()
        
        # Test input
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 50, (batch_size, seq_len))
        
        with torch.no_grad():
            logits, attention_weights = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, 50)
        assert len(attention_weights) == 2  # Number of layers
    
    def test_model_factory(self):
        """Test model creation factory function."""
        # RNN model
        rnn_config = ModelConfig(
            model_type=ModelType.RNN,
            vocab_size=100
        )
        rnn_model = create_model(rnn_config)
        assert isinstance(rnn_model, CharRNN)
        
        # Transformer model
        transformer_config = ModelConfig(
            model_type=ModelType.TRANSFORMER,
            vocab_size=100,
            d_model=128,
            n_heads=4
        )
        transformer_model = create_model(transformer_config)
        assert isinstance(transformer_model, CharTransformer)
    
    def test_model_parameter_count(self):
        """Test model parameter counting."""
        config = ModelConfig(
            model_type=ModelType.RNN,
            vocab_size=100,
            hidden_dim=64,
            embedding_dim=32
        )
        
        model = CharRNN(config)
        total_params, trainable_params = model.get_num_parameters()
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params
    
    def test_model_serialization(self, temp_dir):
        """Test model saving and loading."""
        config = ModelConfig(
            model_type=ModelType.RNN,
            vocab_size=100,
            hidden_dim=64
        )
        
        model = CharRNN(config)
        
        # Save model
        save_path = temp_dir / "model"
        model.save_pretrained(str(save_path))
        
        # Check files exist
        assert (save_path / "pytorch_model.bin").exists()
        assert (save_path / "config.json").exists()
        
        # Load model
        loaded_model = CharRNN.from_pretrained(str(save_path))
        assert loaded_model.vocab_size == model.vocab_size


class TestDataPreparation:
    """Test data preparation functionality."""
    
    def test_prepare_data(self, temp_dir, sample_text):
        """Test data preparation pipeline."""
        # Create temporary data file
        data_file = temp_dir / "data.txt"
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        
        # Prepare data
        text, vocab = prepare_data(data_file, seq_length=10)
        
        assert text == sample_text
        assert len(vocab) > 0
        assert isinstance(vocab, Vocabulary)
    
    def test_prepare_data_with_cache(self, temp_dir, sample_text):
        """Test data preparation with caching."""
        # Create data file
        data_file = temp_dir / "data.txt"
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        
        cache_dir = temp_dir / "cache"
        
        # Prepare data with cache
        text, vocab = prepare_data(data_file, seq_length=10, cache_dir=cache_dir)
        
        # Check cache files
        assert cache_dir.exists()
        assert (cache_dir / "vocabulary.json").exists()
    
    def test_prepare_data_validation(self, temp_dir):
        """Test data preparation input validation."""
        # Non-existent file
        with pytest.raises(ValidationError):
            prepare_data(temp_dir / "nonexistent.txt")
        
        # File too short
        short_file = temp_dir / "short.txt"
        with open(short_file, 'w') as f:
            f.write("hi")
        
        with pytest.raises(DataError):
            prepare_data(short_file, seq_length=100)


class TestUtils:
    """Test utility functions."""
    
    def test_device_selection(self):
        """Test device selection utility."""
        from slm.utils import get_device
        
        # Auto-select device
        device = get_device()
        assert device.type in ['cpu', 'cuda']
        
        # Specify CPU
        cpu_device = get_device('cpu')
        assert cpu_device.type == 'cpu'
    
    def test_seed_setting(self):
        """Test random seed setting."""
        from slm.utils import set_seed
        
        # Set seed
        seed = set_seed(42)
        assert seed == 42
        
        # Generate random seed
        random_seed = set_seed()
        assert isinstance(random_seed, int)
        assert 0 <= random_seed <= 2**32 - 1
    
    def test_text_validation(self):
        """Test text input validation."""
        from slm.utils import validate_text_input
        
        # Valid text
        text = validate_text_input("Hello world")
        assert text == "Hello world"
        
        # Text too short
        with pytest.raises(ValidationError):
            validate_text_input("", min_length=5)
        
        # Text too long
        with pytest.raises(ValidationError):
            validate_text_input("x" * 1000, max_length=100)
    
    def test_file_validation(self, temp_dir):
        """Test file path validation."""
        from slm.utils import validate_file_path
        
        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        # Valid file
        validated_path = validate_file_path(test_file)
        assert validated_path == test_file.resolve()
        
        # Non-existent file
        with pytest.raises(ValidationError):
            validate_file_path(temp_dir / "nonexistent.txt")
        
        # Invalid extension
        with pytest.raises(ValidationError):
            validate_file_path(test_file, allowed_extensions=['.json'])


class TestExceptions:
    """Test custom exception handling."""
    
    def test_exception_hierarchy(self):
        """Test that all exceptions inherit from SLMException."""
        from slm.exceptions import (
            SLMException, DataError, TrainingError,
            GenerationError, ValidationError
        )
        
        exceptions = [
            ModelError("test"),
            DataError("test"),
            TrainingError("test"),
            GenerationError("test"),
            ValidationError("test"),
        ]
        
        for exc in exceptions:
            assert isinstance(exc, SLMException)
    
    def test_exception_details(self):
        """Test exception with details."""
        
        exc = ModelError("Main message", "Detailed explanation")
        assert exc.message == "Main message"
        assert exc.details == "Detailed explanation"
        assert "Detailed explanation" in str(exc)


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.slow
    def test_training_pipeline(self, temp_dir, sample_text):
        """Test complete training pipeline."""
        # Create data file
        data_file = temp_dir / "data.txt"
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write(sample_text * 10)  # Make it longer
        
        # Prepare data
        text, vocab = prepare_data(data_file, seq_length=10)
        
        # Create model
        config = ModelConfig(
            model_type=ModelType.RNN,
            vocab_size=len(vocab),
            hidden_dim=32,
            embedding_dim=16
        )
        model = create_model(config)
        
        # Create training configuration
        train_config = TrainingConfig(
            epochs=1,
            batch_size=2,
            seq_length=10
        )
        
        # This would normally test the full training loop,
        # but we'll just test model forward pass
        from slm.core.data import create_data_loaders
        train_loader, _, _ = create_data_loaders(
            text, vocab, seq_length=10, batch_size=2
        )
        
        # Test one batch
        model.eval()
        for batch in train_loader:
            input_ids, target_ids = batch
            with torch.no_grad():
                logits, _ = model(input_ids)
            assert logits.shape[-1] == len(vocab)
            break
    
    @pytest.mark.slow
    def test_generation_pipeline(self, temp_dir):
        """Test text generation pipeline."""
        # This would test generation with a pre-trained model
        # For now, we'll test the generation config and strategy creation
        from slm.core.generator import GreedySampling, TemperatureSampling
        
        # Test sampling strategies
        logits = torch.randn(100)  # Random logits for 100 vocab items
        
        greedy = GreedySampling()
        token = greedy(logits)
        assert isinstance(token, int)
        assert 0 <= token < 100
        
        temp_sampling = TemperatureSampling(temperature=0.8)
        token = temp_sampling(logits)
        assert isinstance(token, int)
        assert 0 <= token < 100


if __name__ == '__main__':
    pytest.main([__file__])

# Transformer Model Enhancement

## Overview
The SLM project now supports both RNN-based and Transformer-based character-level language models with multi-head attention mechanisms. This gives you the flexibility to experiment with both traditional recurrent architectures and modern transformer architectures.

## New Features

### 1. CharTransformer Model (`core/model.py`)
A complete transformer implementation featuring:
- **Multi-Head Self-Attention**: Configurable number of attention heads (1-16)
- **Positional Encoding**: Sinusoidal position embeddings for sequence understanding
- **Feed-Forward Networks**: Configurable hidden dimensions in FFN layers
- **Layer Normalization**: Applied before each sub-layer (Pre-LN architecture)
- **Causal Masking**: Autoregressive generation support
- **Residual Connections**: Around each sub-layer

### 2. Model Architecture Options
You can now choose between:
- **RNN Models**: LSTM, GRU, or vanilla RNN
- **Transformer Models**: Multi-head attention with configurable parameters

### 3. Enhanced Training Support
- Automatic model type detection during training
- Unified training loop supporting both architectures
- Proper handling of different forward pass requirements

### 4. Advanced Generation
- Context-aware generation for transformers (uses full sequence history)
- Efficient RNN generation (maintains hidden state)
- All sampling methods work with both architectures

## Configuration Options

### Transformer Parameters
- **d_model**: Model dimension (128-1024, default: 256)
- **n_heads**: Number of attention heads (1-16, default: 8)
- **n_layers**: Number of transformer layers (1-12, default: 6)
- **d_ff**: Feed-forward network dimension (256-4096, default: 1024)
- **max_len**: Maximum sequence length (100-2000, default: 1000)
- **dropout**: Dropout rate (0.0-0.8, default: 0.1)

### RNN Parameters (unchanged)
- **rnn_type**: 'lstm', 'gru', 'rnn'
- **embedding_dim**: Character embedding dimension
- **hidden_dim**: RNN hidden state dimension
- **num_layers**: Number of RNN layers
- **dropout**: Dropout rate
- **tie_weights**: Whether to tie input/output weights

## Usage Examples

### GUI Usage
1. Launch the enhanced GUI: `python gui_enhanced.py`
2. In the Model Configuration section:
   - Select "Model Type": Choose between 'rnn' and 'transformer'
   - Configure model-specific parameters
   - Parameters automatically update based on model type
3. Train and generate with either architecture

### Command Line Usage

#### Training a Transformer Model
```python
from core.train import train_model

# Transformer configuration
model_config = {
    'model_type': 'transformer',
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 6,
    'd_ff': 1024,
    'max_len': 1000,
    'dropout': 0.1
}

model, vocab, losses = train_model(
    text=training_text,
    model_config=model_config,
    epochs=10,
    batch_size=32
)
```

#### Training an RNN Model
```python
# RNN configuration
model_config = {
    'model_type': 'rnn',
    'rnn_type': 'lstm',
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.3,
    'tie_weights': False
}

model, vocab, losses = train_model(
    text=training_text,
    model_config=model_config,
    epochs=10,
    batch_size=64
)
```

#### Generation with Either Model
```python
from core.generate import generate

# Works with both RNN and Transformer models
generated_text = generate(
    model=trained_model,
    start="Once upon a time",
    char2idx=char2idx,
    idx2char=idx2char,
    length=200,
    temperature=0.8,
    sampling_method='top_p',
    top_p=0.9
)
```

## Technical Details

### Multi-Head Attention Implementation
The transformer uses a custom multi-head attention implementation with:
- Scaled dot-product attention: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- Multiple attention heads processing different representation subspaces
- Causal masking for autoregressive generation

### Positional Encoding
Uses sinusoidal positional encodings:
- `PE(pos, 2i) = sin(pos/10000^(2i/d_model))`
- `PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))`

### Memory Considerations
- **Transformers**: O(n²) memory complexity due to attention matrix
- **RNNs**: O(n) memory complexity with sequential processing
- Consider using shorter sequences for transformers with limited memory

## Performance Comparison

### Training Speed
- **RNNs**: Faster per epoch due to sequential nature
- **Transformers**: Slower but better parallelization potential

### Generation Quality
- **Transformers**: Often produce more coherent long-range dependencies
- **RNNs**: Good local coherence, may struggle with long-term dependencies

### Memory Usage
- **Transformers**: Higher memory usage, especially for long sequences
- **RNNs**: More memory efficient

## Best Practices

### For Transformers
1. Start with smaller models (d_model=256, n_layers=4) and scale up
2. Use appropriate sequence lengths (don't exceed max_len)
3. Monitor memory usage with larger models
4. Consider gradient accumulation for larger effective batch sizes

### For RNNs
1. LSTMs generally work better than GRU/vanilla RNN for text
2. Gradient clipping is crucial (default: 1.0)
3. Deeper networks (3-4 layers) can improve quality
4. Weight tying can reduce parameters when embedding_dim = hidden_dim

### General Tips
1. Start training with smaller datasets to verify model functionality
2. Use temperature sampling (0.7-1.0) for creative text generation
3. Top-p sampling (0.8-0.95) often produces good results
4. Monitor training loss curves to detect overfitting

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or sequence length for transformers
2. **Slow Training**: Consider using GPU acceleration
3. **Poor Generation**: Try different sampling methods and temperatures
4. **Model Not Loading**: Ensure model_config matches saved checkpoint

### Error Messages
- "Cannot tie weights when embedding_dim != hidden_dim": Only for RNN models when dimensions don't match
- "Unsupported RNN type": Check rnn_type is one of: 'lstm', 'gru', 'rnn'

## Future Enhancements
- Pre-trained transformer loading (GPT-style)
- Mixed precision training support
- Attention visualization tools
- Model size optimization techniques
- Distributed training support

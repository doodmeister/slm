# SLM - Simple Language Models

This repository contains both RNN-based and Transformer-based character-level language models built with PyTorch.

## Features
- **Multiple Architectures**: Choose between LSTM/GRU/RNN or Transformer models
- **Multi-Head Attention**: Transformer models with configurable attention heads
- **Advanced Training**: Early stopping, learning rate scheduling, gradient clipping
- **Flexible Generation**: Temperature, top-k, top-p, and greedy sampling
- **Comprehensive GUI**: Easy-to-use interface for training and generation

## Model Types
1. **RNN Models**: LSTM, GRU, vanilla RNN with traditional recurrent architectures
2. **Transformer Models**: Multi-head self-attention with positional encoding

## Requirements
- Python 3.8+
- PyTorch
- matplotlib (for GUI visualization)
- transformers (for transformer utilities)
- numpy

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Quick Start

### Using the GUI
Launch the enhanced GUI with both RNN and Transformer support:

```bash
python gui_enhanced.py
```

- Select model type (RNN or Transformer)
- Configure architecture parameters
- Choose training corpus and start training
- Generate text with various sampling methods

### Command Line Usage

#### Training
```bash
# Download training data
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data.txt

# Train RNN model
python core/train.py --data data.txt --epochs 5

# Train with custom parameters
python core/train.py --data data.txt --epochs 10 --batch_size 32 --lr 0.001
```

#### Text Generation
```bash
# Generate with trained model
python core/generate.py --checkpoint checkpoints/model.pth --start "Once upon a time" --length 200

# Generate with advanced sampling
python core/generate.py --checkpoint checkpoints/model.pth --sampling top_p --temperature 0.8 --top_p 0.9
```

## Model Configuration

### Transformer Models
- **d_model**: Model dimension (default: 256)
- **n_heads**: Attention heads (default: 8)
- **n_layers**: Transformer layers (default: 6)
- **d_ff**: Feed-forward dimension (default: 1024)
- **max_len**: Maximum sequence length (default: 1000)

### RNN Models  
- **rnn_type**: LSTM, GRU, or RNN (default: LSTM)
- **hidden_dim**: Hidden state dimension (default: 256)
- **embedding_dim**: Character embedding dimension (default: 128)
- **num_layers**: Number of layers (default: 2)

## Advanced Features

### Training Options
- Multiple optimizers (Adam, SGD, RMSprop)
- Learning rate scheduling (step, cosine)
- Early stopping with patience
- Gradient clipping
- Weight decay regularization

### Generation Methods
- **Temperature Sampling**: Control randomness (0.1-2.0)
- **Top-k Sampling**: Sample from top-k tokens
- **Top-p (Nucleus) Sampling**: Sample from cumulative probability threshold
- **Greedy**: Deterministic generation

## File Structure
```
slm/
├── core/
│   ├── model.py          # RNN and Transformer model definitions
│   ├── train.py          # Training script with advanced options
│   └── generate.py       # Text generation with multiple sampling methods
├── gui_enhanced.py       # Comprehensive GUI with model selection
├── gui.py               # Simple GUI (legacy)
├── requirements.txt     # Project dependencies
├── TRANSFORMER_GUIDE.md # Detailed transformer documentation
└── ENHANCEMENTS.md      # Complete feature documentation
```

## Documentation
- **[Transformer Guide](TRANSFORMER_GUIDE.md)**: Detailed transformer model documentation
- **[Enhancements](ENHANCEMENTS.md)**: Complete list of project enhancements
- **[Copilot Instructions](../.github/.copilot-instructions.md)**: AI assistant context

## Performance Tips

### For Transformers
- Start with smaller models and scale up
- Monitor memory usage with longer sequences
- Use appropriate batch sizes (16-64)

### For RNNs
- LSTM generally works best for text
- Use gradient clipping (1.0) to prevent exploding gradients
- Consider weight tying for parameter efficiency

## Examples

### Training a Transformer
```python
model_config = {
    'model_type': 'transformer',
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 6
}
```

### Training an RNN
```python
model_config = {
    'model_type': 'rnn',
    'rnn_type': 'lstm',
    'hidden_dim': 256,
    'num_layers': 2
}
```

## License
MIT License - see LICENSE file for details.

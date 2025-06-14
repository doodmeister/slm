# slm

This repository contains a minimal character-level language model built with PyTorch.

## Requirements
- Python 3.8+
- PyTorch
- matplotlib (for the optional GUI)

Install dependencies with:

```bash
pip install torch matplotlib
```

## Training
Download a training corpus and run the training script. You can choose between
the default LSTM model and a simple Transformer using the `--model` flag:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data.txt
# train an LSTM (default)
python train.py --data data.txt --epochs 1
# or train the transformer variant
python train.py --data data.txt --epochs 1 --model transformer
```

This will save a model checkpoint to `checkpoints/model.pth`.

## Text Generation
Generate text using the trained model. The architecture type is stored in the
checkpoint but you can override it if desired:

```bash
python generate.py --checkpoint checkpoints/model.pth --start "Once upon a time"
# force transformer or rnn if needed
python generate.py --checkpoint checkpoints/model.pth --model transformer
```

This prints generated characters to stdout.

## Graphical Interface
An optional GUI helps run training and visualize progress. Launch it with:

```bash
python gui.py
```

Use the *Select Corpus* button to choose a training text file and start training. A plot of the training loss and model statistics will be displayed when training completes.

### Why try the Transformer?
The Transformer encoder can process tokens in parallel which often leads to
faster training and better perplexity compared to the recurrent model at a
similar scale.

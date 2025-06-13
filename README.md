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
Download a training corpus and run the training script:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data.txt
python train.py --data data.txt --epochs 1
```

This will save a model checkpoint to `checkpoints/model.pth`.

## Text Generation
Generate text using the trained model:

```bash
python generate.py --checkpoint checkpoints/model.pth --start "Once upon a time"
```

This prints generated characters to stdout.

## Graphical Interface
An optional GUI helps run training and visualize progress. Launch it with:

```bash
python gui.py
```

Use the *Select Corpus* button to choose a training text file and start training. A plot of the training loss and model statistics will be displayed when training completes.

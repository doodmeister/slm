# slm

This repository contains a minimal character-level language model built with PyTorch.

## Requirements
- Python 3.8+
- PyTorch
- matplotlib (for the optional GUI)

Install dependencies with:

```bash
pip install torch matplotlib
# optional for subword tokenization
pip install sentencepiece
```

## Training
Download a training corpus and run the training script:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data.txt
python train.py --data data.txt --epochs 1
```

This will save a model checkpoint to `checkpoints/model.pth` as well as
per-epoch checkpoints in the `checkpoints/` directory.

To resume training from a saved checkpoint:

```bash
python train.py --data data.txt --epochs 5 --resume checkpoints/epoch_1.pth
```

The command above continues training until epoch five starting from the
state saved in `epoch_1.pth`.

### Subword Tokenization
Install the optional `sentencepiece` package to enable subword tokenization.
First train or load a tokenizer and preprocess the text:

```bash
python preprocess.py --data data.txt --model_prefix spm --vocab_size 8000 --output data.pt
```

Then train the model on the tokenized dataset:

```bash
python train.py --tokenized_data data.pt --epochs 1
```

## Text Generation
Generate text using the trained model:

```bash
python generate.py --checkpoint checkpoints/model.pth --start "Once upon a time"
```

To use the subword tokenizer at generation time, pass the tokenizer model:

```bash
python generate.py --checkpoint checkpoints/model.pth --tokenizer spm.model --start "Once upon a time"
```

This prints generated text to stdout.

## Graphical Interface
An optional GUI helps run training and visualize progress. Launch it with:

```bash
python gui.py
```

Use the *Select Corpus* button to choose a training text file and start training. A plot of the training loss and model statistics will be displayed when training completes.

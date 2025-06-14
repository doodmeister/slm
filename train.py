import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import CharRNN
try:
    import sentencepiece as spm
except ImportError:  # optional dependency
    spm = None

class TextDataset(Dataset):
    def __init__(self, data, seq_length=100, tokenizer=None):
        """Create a dataset from raw text or a list of token ids."""
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        if isinstance(data, str):
            if tokenizer is not None:
                # encode using provided tokenizer
                self.data = tokenizer.encode(data, out_type=int)
                self.vocab = [tokenizer.id_to_piece(i) for i in range(tokenizer.vocab_size())]
                self.char2idx = None
                self.idx2char = None
            else:
                self.vocab = sorted(list(set(data)))
                self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}
                self.idx2char = {i: ch for ch, i in self.char2idx.items()}
                self.data = [self.char2idx[ch] for ch in data]
        else:
            # assume already tokenized list of ids
            self.data = list(data)
            if tokenizer is not None:
                self.vocab = [tokenizer.id_to_piece(i) for i in range(tokenizer.vocab_size())]
            else:
                self.vocab = list(range(max(self.data) + 1))
            self.char2idx = None
            self.idx2char = None

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        chunk = self.data[idx:idx+self.seq_length+1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out, _ = model(x)
        loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def train_model(data, seq_length=100, epochs=1, batch_size=64, lr=0.002, device=None, tokenizer=None):
    """Train a model on raw text or token ids and return it with the vocab and losses."""
    dataset = TextDataset(data, seq_length=seq_length, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CharRNN(len(dataset.vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(epochs):
        loss = train(model, dataloader, optimizer, criterion, device)
        losses.append(loss)
        print(f"Epoch {epoch+1}/{epochs} Loss: {loss:.4f}")
    return model, dataset.vocab, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data.txt', help='training text file')
    parser.add_argument('--tokenized_data', help='preprocessed token ids (.pt)')
    parser.add_argument('--tokenizer', help='SentencePiece model to use')
    parser.add_argument('--seq_length', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.002)
    args = parser.parse_args()

    tokenizer = None
    data = None
    if args.tokenized_data:
        obj = torch.load(args.tokenized_data)
        data = obj['ids']
        tok_path = args.tokenizer or obj.get('tokenizer')
        if tok_path and spm is not None:
            tokenizer = spm.SentencePieceProcessor(model_file=tok_path)
    else:
        with open(args.data, 'r') as f:
            text = f.read()
        if args.tokenizer and spm is not None:
            tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
            data = text  # dataset will encode
        else:
            data = text

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, vocab, _ = train_model(
        data,
        seq_length=args.seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        tokenizer=tokenizer,
    )
    os.makedirs('checkpoints', exist_ok=True)
    ckpt = {'model_state_dict': model.state_dict(), 'vocab': vocab}
    if tokenizer is not None and (args.tokenizer or args.tokenized_data):
        ckpt['tokenizer'] = args.tokenizer or obj.get('tokenizer')
    torch.save(ckpt, 'checkpoints/model.pth')

if __name__ == '__main__':
    main()

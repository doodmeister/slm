import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import CharRNN, CharTransformer

class TextDataset(Dataset):
    def __init__(self, text, seq_length=100):
        self.seq_length = seq_length
        self.vocab = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.data = [self.char2idx[ch] for ch in text]

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

def train_model(
    text,
    seq_length=100,
    epochs=1,
    batch_size=64,
    lr=0.002,
    device=None,
    model_type="rnn",
):
    """Train a character level model and return it along with the vocab and losses."""
    dataset = TextDataset(text, seq_length=seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "transformer":
        model = CharTransformer(len(dataset.vocab)).to(device)
    else:
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
    parser.add_argument('--data', default='data.txt')
    parser.add_argument('--seq_length', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--model', choices=['rnn', 'transformer'], default='rnn',
                        help='Model architecture to use')
    args = parser.parse_args()

    with open(args.data, 'r') as f:
        text = f.read()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, vocab, _ = train_model(
        text,
        seq_length=args.seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        model_type=args.model,
    )
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'model_type': args.model,
        },
        'checkpoints/model.pth',
    )

if __name__ == '__main__':
    main()

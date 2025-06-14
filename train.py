import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import CharRNN, CharTransformer


class TextDataset(Dataset):
    def __init__(self, text, seq_length=100, vocab=None):
        self.seq_length = seq_length
        self.vocab = sorted(set(text)) if vocab is None else vocab
        self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.data = [self.char2idx[ch] for ch in text if ch in self.char2idx]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_length + 1]
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


def train_model(text, seq_length=100, epochs=1, batch_size=64, lr=0.002, device=None,
                model_type='rnn', resume=None):
    ckpt = None
    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device or 'cpu')

    dataset = TextDataset(text, seq_length=seq_length, vocab=ckpt.get('vocab') if ckpt else None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'transformer':
        model = CharTransformer(len(dataset.vocab)).to(device)
    else:
        model = CharRNN(len(dataset.vocab)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    losses = []
    if ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        losses = ckpt.get('losses', [])

    for epoch in range(start_epoch, epochs + 1):
        loss = train(model, dataloader, optimizer, criterion, device)
        losses.append(loss)
        print(f"Epoch {epoch}/{epochs} Loss: {loss:.4f}")

        os.makedirs('checkpoints', exist_ok=True)
        ckpt_path = os.path.join('checkpoints', f'epoch_{epoch}.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab': dataset.vocab,
            'epoch': epoch,
            'losses': losses,
            'model_type': model_type,
        }, ckpt_path)

    return model, dataset.vocab, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data.txt')
    parser.add_argument('--seq_length', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--model', choices=['rnn', 'transformer'], default='rnn', help='Model architecture to use')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
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
        resume=args.resume,
    )

    os.makedirs('checkpoints', exist_ok=True)
    final_path = os.path.join('checkpoints', 'model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'model_type': args.model,
    }, final_path)


if __name__ == '__main__':
    main()

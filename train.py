import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import CharRNN

try:
    import sentencepiece as spm
except ImportError:
    spm = None


class TextDataset(Dataset):
    def __init__(self, data, seq_length=100, tokenizer=None, vocab=None):
        self.seq_length = seq_length
        self.tokenizer = tokenizer

        if isinstance(data, str):
            if tokenizer:
                self.data = tokenizer.encode(data, out_type=int)
                self.vocab = [tokenizer.id_to_piece(i) for i in range(tokenizer.vocab_size())]
                self.char2idx = None
                self.idx2char = None
            else:
                self.vocab = sorted(set(data)) if vocab is None else vocab
                self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}
                self.idx2char = {i: ch for ch, i in self.char2idx.items()}
                self.data = [self.char2idx[ch] for ch in data if ch in self.char2idx]
        else:
            # Assume data is already tokenized
            self.data = list(data)
            self.vocab = vocab or list(range(max(self.data) + 1))
            self.char2idx = None
            self.idx2char = None

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


def train_model(data, seq_length=100, epochs=1, batch_size=64, lr=0.002, device=None, tokenizer=None, resume=None):
    ckpt = None
    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device or 'cpu')

    dataset = TextDataset(
        data,
        seq_length=seq_length,
        tokenizer=tokenizer,
        vocab=ckpt.get('vocab') if ckpt else None
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        }, ckpt_path)

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
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume from')
    args = parser.parse_args()

    tokenizer = None
    data = None

    if args.tokenized_data:
        obj = torch.load(args.tokenized_data)
        data = obj['ids']
        tok_path = args.tokenizer or obj.get('tokenizer')
        if tok_path and spm:
            tokenizer = spm.SentencePieceProcessor(model_file=tok_path)
    else:
        with open(args.data, 'r') as f:
            text = f.read()
        if args.tokenizer and spm:
            tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
            data = text
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
        resume=args.resume
    )

    os.makedirs('checkpoints', exist_ok=True)
    final_ckpt = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab
    }
    if tokenizer and (args.tokenizer or args.tokenized_data):
        final_ckpt['tokenizer'] = args.tokenizer or obj.get('tokenizer')
    torch.save(final_ckpt, 'checkpoints/model.pth')


if __name__ == '__main__':
    main()

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import CharRNN

class TextDataset(Dataset):
    def __init__(self, text, seq_length=100, vocab=None):
        """Dataset of fixed-length character sequences.

        Parameters
        ----------
        text : str
            The training corpus.
        seq_length : int
            Length of each sequence used for training.
        vocab : list or None
            Optional list of vocabulary characters to use when encoding
            ``text``. When ``None`` the vocabulary is built from the text.
        """
        self.seq_length = seq_length
        if vocab is None:
            self.vocab = sorted(list(set(text)))
        else:
            self.vocab = vocab
        self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.data = [self.char2idx[ch] for ch in text if ch in self.char2idx]

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

def train_model(text, seq_length=100, epochs=1, batch_size=64, lr=0.002, device=None, resume=None):
    """Train a character level model and return it along with the vocab and losses.

    Parameters
    ----------
    text : str
        Training corpus.
    seq_length : int
        Length of each training sequence.
    epochs : int
        Total number of epochs to train.
    batch_size : int
        Mini-batch size.
    lr : float
        Learning rate.
    device : torch.device or None
        Device to run training on. If ``None`` a CUDA device is used when
        available.
    resume : str or None
        Optional path to a checkpoint from which to resume training.
    """

    ckpt = None
    if resume is not None and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device or 'cpu')

    dataset = TextDataset(text, seq_length=seq_length, vocab=ckpt.get('vocab') if ckpt else None)
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
    parser.add_argument('--data', default='data.txt')
    parser.add_argument('--seq_length', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--resume', type=str, help='path to checkpoint to resume from')
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
        resume=args.resume,
    )
    os.makedirs('checkpoints', exist_ok=True)
    final_path = os.path.join('checkpoints', 'model.pth')
    torch.save({'model_state_dict': model.state_dict(), 'vocab': vocab}, final_path)

if __name__ == '__main__':
    main()

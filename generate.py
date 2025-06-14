import argparse
import torch
from model import CharRNN
try:
    import sentencepiece as spm
except ImportError:  # optional dependency
    spm = None

@torch.no_grad()
def generate(model, start, length=200, device='cpu', tokenizer=None, char2idx=None, idx2char=None):
    """Generate text starting from ``start`` using either a tokenizer or raw characters."""
    model.eval()
    if tokenizer is not None:
        indices = tokenizer.encode(start, out_type=int)
    else:
        indices = [char2idx[ch] for ch in start]
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    hidden = None
    for _ in range(length):
        output, hidden = model(input_tensor[:, -1:], hidden)
        prob = torch.softmax(output[:, -1, :], dim=-1)
        idx = torch.multinomial(prob, 1).item()
        indices.append(idx)
        input_tensor = torch.tensor([idx], dtype=torch.long).unsqueeze(0).to(device)
    if tokenizer is not None:
        return tokenizer.decode(indices)
    else:
        return ''.join(idx2char[i] for i in indices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/model.pth')
    parser.add_argument('--start', default='Once upon a time')
    parser.add_argument('--length', type=int, default=200)
    parser.add_argument('--tokenizer', help='SentencePiece model for generation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device)
    vocab = ckpt['vocab']
    tok_path = args.tokenizer or ckpt.get('tokenizer')
    tokenizer = spm.SentencePieceProcessor(model_file=tok_path) if tok_path and spm is not None else None
    char2idx = {ch: i for i, ch in enumerate(vocab)} if tokenizer is None else None
    idx2char = {i: ch for i, ch in enumerate(vocab)} if tokenizer is None else None
    model = CharRNN(len(vocab)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    text = generate(model, args.start, args.length, device, tokenizer, char2idx, idx2char)
    print(text)

if __name__ == '__main__':
    main()

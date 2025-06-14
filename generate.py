import argparse
import torch
from model import CharRNN, CharTransformer

@torch.no_grad()
def generate(model, start, char2idx, idx2char, length=200, device="cpu"):
    model.eval()
    indices = [char2idx[ch] for ch in start]
    context = torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)
    hidden = None

    for _ in range(length):
        if isinstance(model, CharTransformer):
            output, _ = model(context)
        else:
            output, hidden = model(context[:, -1:], hidden)

        prob = torch.softmax(output[:, -1, :], dim=-1)
        idx = torch.multinomial(prob, 1).item()
        indices.append(idx)

        new_tensor = torch.tensor([[idx]], dtype=torch.long, device=device)
        if isinstance(model, CharTransformer):
            context = torch.cat([context, new_tensor], dim=1)
        else:
            context = new_tensor

    return "".join(idx2char[i] for i in indices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/model.pth')
    parser.add_argument('--start', default='Once upon a time')
    parser.add_argument('--length', type=int, default=200)
    parser.add_argument('--model', choices=['rnn', 'transformer'], default=None,
                        help='Model architecture (defaults to checkpoint value)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device)
    vocab = ckpt['vocab']
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = {i: ch for i, ch in enumerate(vocab)}

    model_type = args.model or ckpt.get('model_type', 'rnn')
    model_cls = CharTransformer if model_type == 'transformer' else CharRNN
    model = model_cls(len(vocab)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    text = generate(model, args.start, char2idx, idx2char, args.length, device)
    print(text)

if __name__ == '__main__':
    main()

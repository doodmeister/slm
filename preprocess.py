import argparse
import os
import torch
try:
    import sentencepiece as spm
except ImportError:
    spm = None


def main():
    parser = argparse.ArgumentParser(description="Train or load a tokenizer and convert text to token IDs")
    parser.add_argument('--data', required=True, help='input text file')
    parser.add_argument('--model', help='existing SentencePiece model')
    parser.add_argument('--model_prefix', default='spm', help='prefix for training a new tokenizer')
    parser.add_argument('--vocab_size', type=int, default=2000, help='vocabulary size when training')
    parser.add_argument('--output', default='data.pt', help='where to save token IDs')
    args = parser.parse_args()

    if spm is None:
        raise ImportError("sentencepiece is required for preprocessing")

    if args.model and os.path.exists(args.model):
        model_file = args.model
    else:
        spm.SentencePieceTrainer.train(input=args.data,
                                       model_prefix=args.model_prefix,
                                       vocab_size=args.vocab_size)
        model_file = args.model_prefix + '.model'

    sp = spm.SentencePieceProcessor(model_file=model_file)

    with open(args.data, 'r') as f:
        text = f.read()
    ids = sp.encode(text, out_type=int)
    torch.save({'ids': ids, 'tokenizer': model_file}, args.output)
    print(f'Saved token IDs to {args.output}')


if __name__ == '__main__':
    main()

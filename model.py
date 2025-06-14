import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden


class CharTransformer(nn.Module):
    """A minimal Transformer-based character model."""

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_layers=2,
        nhead=8,
        dim_feedforward=512,
        max_seq_length=512,
    ):
        super().__init__()
        self.model_type = "transformer"
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_seq_length, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, hidden=None):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = x.transpose(0, 1)  # [seq_len, batch, dim]
        out = self.transformer(x)
        out = self.fc(out)
        out = out.transpose(0, 1)
        return out, None

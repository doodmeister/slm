"""Enhanced model definitions with production-grade features."""

import math
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from slm.exceptions import ModelError, ValidationError
from slm.config import ModelConfig, ModelType, RNNType


class BaseModel(nn.Module):
    """Base class for all SLM models with common functionality."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        if self.vocab_size is None:
            raise ModelError("vocab_size must be specified in model config")

        logger.debug(f"Initializing {self.__class__.__name__} with config: {config}")

    def get_num_parameters(self) -> Tuple[int, int]:
        """Get number of parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params, trainable_params = self.get_num_parameters()

        # Calculate model size
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        model_size_mb = (param_size + buffer_size) / 1024 / 1024

        return {
            "model_type": self.__class__.__name__,
            "vocab_size": self.vocab_size,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "config": self.config.dict(),
        }

    def save_pretrained(self, save_path: str) -> None:
        """Save model and configuration."""
        import json
        from pathlib import Path

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")

        # Save config
        with open(save_path / "config.json", "w") as f:
            json.dump(self.config.dict(), f, indent=2)

        # Save model info
        with open(save_path / "model_info.json", "w") as f:
            json.dump(self.get_model_info(), f, indent=2)

        logger.info(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(cls, load_path: str, config: Optional[ModelConfig] = None):
        """Load model from saved state."""
        import json
        from pathlib import Path

        load_path = Path(load_path)

        # Load config if not provided
        if config is None:
            with open(load_path / "config.json", "r") as f:
                config_dict = json.load(f)
            config = ModelConfig(**config_dict)

        # Create model instance
        model = cls(config)

        # Load state dict
        state_dict = torch.load(load_path / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)

        logger.info(f"Model loaded from {load_path}")
        return model


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding."""
        # x shape: [seq_len, batch_size, d_model]
        seq_len = x.size(0)
        if seq_len > self.pe.size(0):
            raise ModelError(
                f"Sequence length {seq_len} exceeds max_len {self.pe.size(0)}"
            )

        x = x + self.pe[:seq_len, :, :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ModelError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention.

        Args:
            query, key, value: Input tensors [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]

        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attention_weights: Attention weights [batch_size, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len = query.size(0), query.size(1)

        # Linear projections and reshape
        Q = (
            self.w_q(query)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(key)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(value)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        attention_output, attention_weights = self._scaled_dot_product_attention(
            Q, K, V, mask
        )

        # Concatenate heads
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # Final linear projection
        output = self.w_o(attention_output)

        return output, attention_weights

    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention."""
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward layers."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transformer block.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask

        Returns:
            output: Transformed tensor [batch_size, seq_len, d_model]
            attention_weights: Attention weights
        """
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class CharTransformer(BaseModel):
    """Character-level transformer model with enhanced features."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        if config.model_type != ModelType.TRANSFORMER:
            raise ModelError(f"Expected transformer config, got {config.model_type}")

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers_transformer
        self.max_len = config.max_len

        # Embedding layers
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(
            self.d_model, self.max_len, config.dropout
        )

        # Transformer layers
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model, self.n_heads, config.d_ff, config.dropout
                )
                for _ in range(self.n_layers)
            ]
        )

        # Output layers
        self.norm = nn.LayerNorm(self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"CharTransformer initialized: {self.get_num_parameters()[0]:,} parameters"
        )

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        mask = torch.tril(torch.ones(size, size, device=device))
        return mask

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the transformer.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len, seq_len]

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            attention_weights: Dictionary of attention weights per layer
        """
        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_len:
            raise ModelError(
                f"Sequence length {seq_len} exceeds max_len {self.max_len}"
            )

        # Generate causal mask if not provided
        if attention_mask is None:
            attention_mask = self._generate_causal_mask(seq_len, input_ids.device)

        # Embedding + positional encoding
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]

        # Apply transformer blocks
        attention_weights = {}
        for i, transformer_block in enumerate(self.transformer_blocks):
            x, attn_weights = transformer_block(x, attention_mask)
            attention_weights[f"layer_{i}"] = attn_weights

        # Final normalization and output projection
        x = self.norm(x)
        logits = self.output_projection(x)

        return logits, attention_weights


class CharRNN(BaseModel):
    """Enhanced character-level RNN model."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        if config.model_type != ModelType.RNN:
            raise ModelError(f"Expected RNN config, got {config.model_type}")

        self.rnn_type = config.rnn_type
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.embedding_dim = config.embedding_dim

        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # RNN layer
        rnn_class = self._get_rnn_class(config.rnn_type)
        self.rnn = rnn_class(
            self.embedding_dim,
            self.hidden_dim,
            self.num_layers,
            batch_first=True,
            dropout=config.dropout if self.num_layers > 1 else 0,
        )

        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)

        # Output layer
        self.output_projection = nn.Linear(self.hidden_dim, self.vocab_size)

        # Optionally tie embedding and output weights
        if config.tie_weights:
            if self.embedding_dim != self.hidden_dim:
                raise ModelError(
                    f"Cannot tie weights when embedding_dim ({self.embedding_dim}) "
                    f"!= hidden_dim ({self.hidden_dim})"
                )
            self.output_projection.weight = self.embedding.weight

        # Initialize weights
        self._init_weights()

        logger.info(f"CharRNN initialized: {self.get_num_parameters()[0]:,} parameters")

    def _get_rnn_class(self, rnn_type: RNNType) -> nn.Module:
        """Get RNN class based on type."""
        rnn_classes = {
            RNNType.LSTM: nn.LSTM,
            RNNType.GRU: nn.GRU,
            RNNType.RNN: nn.RNN,
        }

        if rnn_type not in rnn_classes:
            raise ModelError(f"Unsupported RNN type: {rnn_type}")

        return rnn_classes[rnn_type]

    def _init_weights(self) -> None:
        """Initialize model weights with Xavier initialization."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "rnn" in name:
                    # Initialize RNN weights with Xavier
                    nn.init.xavier_uniform_(param)
                else:
                    # Initialize other weights normally
                    nn.init.normal_(param, 0.0, 0.02)
            elif "bias" in name:
                nn.init.zeros_(param)

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize hidden state for RNN."""
        hidden_shape = (self.num_layers, batch_size, self.hidden_dim)

        if self.rnn_type == RNNType.LSTM:
            h0 = torch.zeros(hidden_shape, device=device)
            c0 = torch.zeros(hidden_shape, device=device)
            return (h0, c0)
        else:
            return torch.zeros(hidden_shape, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of the RNN.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            hidden: Previous hidden state

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            hidden: New hidden state
        """
        # Input validation
        if input_ids.dim() != 2:
            raise ValidationError(f"Expected 2D input_ids, got {input_ids.dim()}D")

        batch_size = input_ids.size(0)

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, input_ids.device)

        # Embedding
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)

        # RNN forward pass
        try:
            rnn_output, hidden = self.rnn(embedded, hidden)
        except RuntimeError as e:
            raise ModelError(f"RNN forward pass failed: {e}")

        # Apply dropout to RNN output
        rnn_output = self.dropout(rnn_output)

        # Output projection
        logits = self.output_projection(rnn_output)

        return logits, hidden


def create_model(config: ModelConfig) -> BaseModel:
    """Factory function to create models based on configuration."""
    if config.model_type == ModelType.TRANSFORMER:
        return CharTransformer(config)
    elif config.model_type == ModelType.RNN:
        return CharRNN(config)
    else:
        raise ModelError(f"Unsupported model type: {config.model_type}")


def load_model_from_checkpoint(
    checkpoint_path: str, config: Optional[ModelConfig] = None
) -> BaseModel:
    """Load model from checkpoint with proper error handling."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Get config from checkpoint if not provided
        if config is None:
            if "config" not in checkpoint:
                raise ModelError("No config found in checkpoint and none provided")
            config = ModelConfig(**checkpoint["config"])

        # Create model
        model = create_model(config)

        # Load state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise ModelError("No model_state_dict found in checkpoint")

        logger.info(f"Model loaded from checkpoint: {checkpoint_path}")
        return model

    except Exception as e:
        raise ModelError(f"Failed to load model from checkpoint {checkpoint_path}: {e}")

"""Configuration management for SLM."""

from typing import Optional, Union
from pathlib import Path
from enum import Enum
import yaml
from pydantic import BaseModel, Field, validator
from loguru import logger

from slm.exceptions import ConfigurationError


class ModelType(str, Enum):
    """Supported model types."""

    RNN = "rnn"
    TRANSFORMER = "transformer"


class RNNType(str, Enum):
    """Supported RNN types."""

    LSTM = "lstm"
    GRU = "gru"
    RNN = "rnn"


class OptimizerType(str, Enum):
    """Supported optimizer types."""

    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAMW = "adamw"


class SchedulerType(str, Enum):
    """Supported learning rate scheduler types."""

    NONE = "none"
    STEP = "step"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class SamplingMethod(str, Enum):
    """Supported text generation sampling methods."""

    GREEDY = "greedy"
    TEMPERATURE = "temperature"
    TOP_K = "top_k"
    TOP_P = "top_p"
    NUCLEUS = "nucleus"


class ModelConfig(BaseModel):
    """Configuration for model architecture."""

    model_type: ModelType = Field(default=ModelType.RNN, description="Type of model")
    vocab_size: Optional[int] = Field(default=None, description="Vocabulary size")

    # RNN-specific parameters
    rnn_type: RNNType = Field(default=RNNType.LSTM, description="Type of RNN cell")
    embedding_dim: int = Field(
        default=128, ge=32, le=1024, description="Embedding dimension"
    )
    hidden_dim: int = Field(default=256, ge=64, le=2048, description="Hidden dimension")
    num_layers: int = Field(default=2, ge=1, le=12, description="Number of layers")
    dropout: float = Field(default=0.3, ge=0.0, le=0.8, description="Dropout rate")
    tie_weights: bool = Field(
        default=False, description="Tie embedding and output weights"
    )

    # Transformer-specific parameters
    d_model: int = Field(default=256, ge=64, le=1024, description="Model dimension")
    n_heads: int = Field(
        default=8, ge=1, le=16, description="Number of attention heads"
    )
    n_layers_transformer: int = Field(
        default=6, ge=1, le=12, description="Number of transformer layers"
    )
    d_ff: int = Field(
        default=1024, ge=256, le=4096, description="Feed-forward dimension"
    )
    max_len: int = Field(
        default=1000, ge=100, le=10000, description="Maximum sequence length"
    )

    @validator("n_heads")
    def validate_n_heads(cls, v, values):
        """Validate that d_model is divisible by n_heads."""
        if "model_type" in values and values["model_type"] == ModelType.TRANSFORMER:
            d_model = values.get("d_model", 256)
            if d_model % v != 0:
                raise ValueError(
                    f"d_model ({d_model}) must be divisible by n_heads ({v})"
                )
        return v


class TrainingConfig(BaseModel):
    """Configuration for training parameters."""

    seq_length: int = Field(default=100, ge=10, le=1000, description="Sequence length")
    batch_size: int = Field(default=64, ge=1, le=512, description="Batch size")
    epochs: int = Field(default=10, ge=1, le=1000, description="Number of epochs")
    learning_rate: float = Field(
        default=0.002, gt=0.0, le=1.0, description="Learning rate"
    )
    weight_decay: float = Field(default=0.0, ge=0.0, le=0.1, description="Weight decay")

    # Optimizer settings
    optimizer: OptimizerType = Field(
        default=OptimizerType.ADAM, description="Optimizer type"
    )
    momentum: float = Field(default=0.9, ge=0.0, le=1.0, description="SGD momentum")

    # Learning rate scheduling
    scheduler: SchedulerType = Field(
        default=SchedulerType.NONE, description="LR scheduler type"
    )
    step_size: int = Field(default=10, ge=1, description="Step size for step scheduler")
    gamma: float = Field(default=0.1, gt=0.0, le=1.0, description="Decay factor")

    # Regularization
    grad_clip: Optional[float] = Field(
        default=1.0, gt=0.0, description="Gradient clipping threshold"
    )

    # Early stopping
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    patience: int = Field(default=5, ge=1, description="Early stopping patience")
    min_delta: float = Field(
        default=0.001, ge=0.0, description="Minimum improvement threshold"
    )

    # Checkpointing
    save_every: int = Field(
        default=1, ge=1, description="Save checkpoint every N epochs"
    )
    save_best_only: bool = Field(
        default=False, description="Only save best checkpoints"
    )

    # Validation
    validation_split: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Validation split ratio"
    )

    # Device and performance
    device: Optional[str] = Field(
        default=None, description="Device to use (auto-detect if None)"
    )
    num_workers: int = Field(
        default=0, ge=0, description="Number of data loader workers"
    )
    pin_memory: bool = Field(default=True, description="Pin memory for data loading")


class GenerationConfig(BaseModel):
    """Configuration for text generation."""

    length: int = Field(default=200, ge=1, le=10000, description="Generation length")
    temperature: float = Field(
        default=1.0, gt=0.0, le=2.0, description="Sampling temperature"
    )
    top_k: int = Field(default=0, ge=0, description="Top-k sampling (0 = disabled)")
    top_p: float = Field(
        default=0.9, gt=0.0, le=1.0, description="Nucleus sampling threshold"
    )
    sampling_method: SamplingMethod = Field(
        default=SamplingMethod.TEMPERATURE, description="Sampling method"
    )

    # Generation parameters
    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducibility"
    )
    max_new_tokens: Optional[int] = Field(
        default=None, description="Maximum new tokens to generate"
    )

    # Safety parameters
    repetition_penalty: float = Field(
        default=1.0, ge=0.1, le=2.0, description="Repetition penalty"
    )
    length_penalty: float = Field(
        default=1.0, ge=0.1, le=2.0, description="Length penalty"
    )

    @validator("top_k")
    def validate_top_k_with_method(cls, v, values):
        """Validate top_k parameter with sampling method."""
        method = values.get("sampling_method")
        if method == SamplingMethod.TOP_K and v <= 0:
            raise ValueError("top_k must be > 0 when using top_k sampling")
        return v


class Config(BaseModel):
    """Main configuration combining all sub-configurations."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    # Paths
    data_path: Optional[Path] = Field(default=None, description="Path to training data")
    checkpoint_dir: Path = Field(
        default=Path("checkpoints"), description="Checkpoint directory"
    )
    log_dir: Path = Field(default=Path("logs"), description="Log directory")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_to_file: bool = Field(default=True, description="Enable file logging")

    # Reproducibility
    seed: Optional[int] = Field(default=42, description="Global random seed")
    deterministic: bool = Field(
        default=False, description="Enable deterministic training"
    )

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        return load_config(config_path)


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML file."""
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            config_dict = {}

        config = Config(**config_dict)
        logger.info(f"Configuration loaded from {config_path}")
        return config

    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in config file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration: {e}")


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    try:
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and remove None values for cleaner YAML
        config_dict = config.dict(exclude_none=True)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {config_path}")

    except Exception as e:
        raise ConfigurationError(f"Error saving configuration: {e}")


def create_default_config() -> Config:
    """Create a default configuration."""
    return Config()


def validate_config_compatibility(config: Config) -> None:
    """Validate configuration for internal compatibility."""
    try:
        # Validate model-specific parameters
        if config.model.model_type == ModelType.TRANSFORMER:
            if config.model.d_model % config.model.n_heads != 0:
                raise ValueError(
                    "d_model must be divisible by n_heads for transformer models"
                )

        # Validate training parameters
        if config.training.validation_split >= 1.0:
            raise ValueError("validation_split must be < 1.0")

        # Validate generation parameters
        if (
            config.generation.sampling_method == SamplingMethod.TOP_K
            and config.generation.top_k <= 0
        ):
            raise ValueError("top_k must be > 0 when using top_k sampling")

        logger.debug("Configuration validation passed")

    except Exception as e:
        raise ConfigurationError(f"Configuration validation failed: {e}")

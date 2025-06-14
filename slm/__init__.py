"""Simple Language Models (SLM) - Production-grade character-level language models."""

__version__ = "2.0.0"
__author__ = "SLM Team"
__email__ = "team@slm.ai"

from slm.core.models import CharRNN, CharTransformer
from slm.core.trainer import Trainer
from slm.core.generator import Generator
from slm.config import ModelConfig, TrainingConfig, GenerationConfig

__all__ = [
    "CharRNN",
    "CharTransformer",
    "Trainer",
    "Generator",
    "ModelConfig",
    "TrainingConfig",
    "GenerationConfig",
]

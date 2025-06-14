"""Enhanced text generation module with production-grade features."""

import time
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import torch
import torch.nn.functional as F
from loguru import logger

from slm.exceptions import GenerationError, ValidationError
from slm.config import GenerationConfig, SamplingMethod
from slm.core.models import BaseModel, CharRNN, CharTransformer
from slm.core.data import Vocabulary
from slm.utils import set_seed, validate_text_input


class SamplingStrategy:
    """Base class for sampling strategies."""

    def __call__(self, logits: torch.Tensor) -> int:
        """Sample from logits."""
        raise NotImplementedError


class GreedySampling(SamplingStrategy):
    """Greedy sampling - always pick the most likely token."""

    def __call__(self, logits: torch.Tensor) -> int:
        return int(torch.argmax(logits, dim=-1).item())


class TemperatureSampling(SamplingStrategy):
    """Temperature-based sampling."""

    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValidationError(f"Temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> int:
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())


class TopKSampling(SamplingStrategy):
    """Top-k sampling."""

    def __init__(self, k: int, temperature: float = 1.0):
        if k <= 0:
            raise ValidationError(f"k must be > 0, got {k}")
        if temperature <= 0:
            raise ValidationError(f"Temperature must be > 0, got {temperature}")
        self.k = k
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> int:
        # Scale by temperature
        scaled_logits = logits / self.temperature

        # Keep only top-k logits
        top_k = min(self.k, scaled_logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k)

        # Mask out other logits
        filtered_logits = torch.full_like(scaled_logits, float("-inf"))
        filtered_logits.scatter_(0, top_k_indices, top_k_logits)

        # Sample from filtered distribution
        probs = F.softmax(filtered_logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())


class TopPSampling(SamplingStrategy):
    """Top-p (nucleus) sampling."""

    def __init__(self, p: float, temperature: float = 1.0):
        if not 0 < p <= 1:
            raise ValidationError(f"p must be in (0, 1], got {p}")
        if temperature <= 0:
            raise ValidationError(f"Temperature must be > 0, got {temperature}")
        self.p = p
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor) -> int:
        # Scale by temperature
        scaled_logits = logits / self.temperature

        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)

        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff point
        cutoff_mask = cumulative_probs <= self.p

        # Include at least one token
        cutoff_mask[0] = True

        # Mask out tokens beyond cutoff
        filtered_probs = torch.zeros_like(sorted_probs)
        filtered_probs[cutoff_mask] = sorted_probs[cutoff_mask]

        # Renormalize
        filtered_probs = filtered_probs / filtered_probs.sum()

        # Sample from filtered distribution
        sampled_index = int(torch.multinomial(filtered_probs, 1).item())
        return int(sorted_indices[sampled_index].item())


class RepetitionPenalty:
    """Apply repetition penalty to reduce repetitive generation."""

    def __init__(self, penalty: float = 1.0, window_size: int = 50):
        """Initialize repetition penalty.

        Args:
            penalty: Penalty factor (> 1.0 reduces repetition).
            window_size: Size of the context window to check for repetitions.
        """
        if penalty <= 0:
            raise ValidationError(f"Penalty must be > 0, got {penalty}")

        self.penalty = penalty
        self.window_size = window_size

    def apply(self, logits: torch.Tensor, generated_tokens: List[int]) -> torch.Tensor:
        """Apply repetition penalty to logits.

        Args:
            logits: Model logits [vocab_size].
            generated_tokens: Previously generated tokens.

        Returns:
            Modified logits with repetition penalty applied.
        """
        if self.penalty == 1.0 or not generated_tokens:
            return logits

        # Get recent tokens within window
        recent_tokens = generated_tokens[-self.window_size :]

        # Apply penalty to repeated tokens
        modified_logits = logits.clone()
        for token in set(recent_tokens):
            if modified_logits[token] > 0:
                modified_logits[token] /= self.penalty
            else:
                modified_logits[token] *= self.penalty

        return modified_logits


class Generator:
    """Enhanced text generator with comprehensive features."""

    def __init__(
        self,
        model: BaseModel,
        vocab: Vocabulary,
        config: GenerationConfig,
        device: Optional[torch.device] = None,
    ):
        """Initialize generator.

        Args:
            model: Trained model for generation.
            vocab: Vocabulary for encoding/decoding.
            config: Generation configuration.
            device: Device to run generation on.
        """
        self.model = model
        self.vocab = vocab
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        # Initialize sampling strategy
        self.sampling_strategy = self._create_sampling_strategy()

        # Initialize repetition penalty
        self.repetition_penalty = RepetitionPenalty(
            config.repetition_penalty,
            window_size=50,
        )

        logger.info(f"Generator initialized with {config.sampling_method} sampling")

    def _create_sampling_strategy(self) -> SamplingStrategy:
        """Create sampling strategy based on configuration."""
        if self.config.sampling_method == SamplingMethod.GREEDY:
            return GreedySampling()
        elif self.config.sampling_method == SamplingMethod.TEMPERATURE:
            return TemperatureSampling(self.config.temperature)
        elif self.config.sampling_method == SamplingMethod.TOP_K:
            return TopKSampling(self.config.top_k, self.config.temperature)
        elif self.config.sampling_method in [
            SamplingMethod.TOP_P,
            SamplingMethod.NUCLEUS,
        ]:
            return TopPSampling(self.config.top_p, self.config.temperature)
        else:
            raise GenerationError(
                f"Unsupported sampling method: {self.config.sampling_method}"
            )

    def generate(
        self,
        prompt: str = "",
        max_length: Optional[int] = None,
        stop_tokens: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Generate text with comprehensive options.

        Args:
            prompt: Starting text prompt.
            max_length: Maximum generation length (overrides config).
            stop_tokens: List of tokens that stop generation.
            progress_callback: Optional callback for progress updates.

        Returns:
            Dictionary with generation results and metadata.
        """
        # Set random seed if specified
        if self.config.seed is not None:
            set_seed(self.config.seed)

        # Validate and prepare prompt
        prompt = validate_text_input(prompt, min_length=0, max_length=10000)

        # Determine generation length
        generation_length = max_length or self.config.length
        if self.config.max_new_tokens:
            generation_length = min(generation_length, self.config.max_new_tokens)

        # Encode prompt
        try:
            prompt_tokens = self.vocab.encode(prompt) if prompt else []
        except Exception as e:
            raise GenerationError(f"Failed to encode prompt: {e}")

        logger.info(
            f"Starting generation: prompt={len(prompt)} chars, length={generation_length}"
        )

        # Generation state
        generated_tokens = prompt_tokens.copy()
        generation_start_time = time.time()
        # Prepare stop token indices
        stop_token_indices = set()
        if stop_tokens:
            for token in stop_tokens:
                try:
                    indices = self.vocab.encode(token)
                    stop_token_indices.update(indices)
                except (KeyError, ValueError):
                    logger.warning(f"Could not encode stop token: {token}")

        try:
            with torch.no_grad():
                # Initialize hidden state for RNN models
                hidden = None
                if isinstance(self.model, CharRNN):
                    hidden = self.model.init_hidden(1, self.device)

                for step in range(generation_length):
                    # Prepare input tensor
                    input_tensor = self._prepare_input(
                        generated_tokens, hidden is not None
                    )

                    # Forward pass
                    try:
                        if isinstance(self.model, CharTransformer):
                            logits, _ = self.model(input_tensor)
                            next_token_logits = logits[
                                0, -1, :
                            ]  # Last token of first batch
                        else:  # CharRNN
                            if step == 0 and len(generated_tokens) > 1:
                                # For first step with prompt, use full sequence
                                logits, hidden = self.model(input_tensor, hidden)
                                next_token_logits = logits[0, -1, :]
                            else:
                                # For subsequent steps, use only last token
                                last_token = torch.tensor(
                                    [[generated_tokens[-1]]], device=self.device
                                )
                                logits, hidden = self.model(last_token, hidden)
                                next_token_logits = logits[0, 0, :]

                    except Exception as e:
                        raise GenerationError(
                            f"Model forward pass failed at step {step}: {e}"
                        )

                    # Apply repetition penalty
                    if self.config.repetition_penalty != 1.0:
                        next_token_logits = self.repetition_penalty.apply(
                            next_token_logits, generated_tokens
                        )

                    # Sample next token
                    try:
                        next_token = self.sampling_strategy(next_token_logits)
                    except Exception as e:
                        raise GenerationError(f"Sampling failed at step {step}: {e}")

                    # Add to generated sequence
                    generated_tokens.append(next_token)

                    # Check for stop tokens
                    if next_token in stop_token_indices:
                        logger.debug(f"Stop token encountered at step {step}")
                        break

                    # Progress callback
                    if progress_callback:
                        progress_callback(step + 1, generation_length)

                    # Memory management for very long sequences
                    if isinstance(self.model, CharTransformer):
                        max_context = getattr(self.model, "max_len", 1000)
                        if len(generated_tokens) > max_context:
                            # Truncate context but keep recent tokens
                            generated_tokens = (
                                generated_tokens[-max_context // 2 :]
                                + generated_tokens[-max_context // 2 :]
                            )

        except KeyboardInterrupt:
            logger.info("Generation interrupted by user")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise GenerationError(f"Generation failed: {e}")

        # Decode generated text
        try:
            generated_text = self.vocab.decode(generated_tokens)
            prompt_text = self.vocab.decode(prompt_tokens) if prompt_tokens else ""
            new_text = (
                generated_text[len(prompt_text) :] if prompt_text else generated_text
            )
        except Exception as e:
            raise GenerationError(f"Failed to decode generated tokens: {e}")

        # Prepare results
        generation_time = time.time() - generation_start_time

        results = {
            "prompt": prompt,
            "generated_text": generated_text,
            "new_text": new_text,
            "total_tokens": len(generated_tokens),
            "new_tokens": len(generated_tokens) - len(prompt_tokens),
            "generation_time": generation_time,
            "tokens_per_second": (len(generated_tokens) - len(prompt_tokens))
            / generation_time
            if generation_time > 0
            else 0,
            "config": self.config.dict(),
            "stopped_early": len(generated_tokens) - len(prompt_tokens)
            < generation_length,
        }

        logger.info(
            f"Generation completed: {results['new_tokens']} tokens in {generation_time:.2f}s "
            f"({results['tokens_per_second']:.1f} tokens/s)"
        )

        return results

    def _prepare_input(self, tokens: List[int], is_rnn: bool) -> torch.Tensor:
        """Prepare input tensor for model."""
        if not tokens:
            raise GenerationError("Cannot prepare input from empty token list")

        if is_rnn:
            # For RNN, we can use the full sequence for the first step
            # but subsequent steps will only use the last token
            input_tokens = tokens
        else:
            # For transformer, always use full sequence up to max_len
            max_len = getattr(self.model, "max_len", 1000)
            input_tokens = tokens[-max_len:] if len(tokens) > max_len else tokens

        return torch.tensor([input_tokens], dtype=torch.long, device=self.device)

    def generate_batch(
        self, prompts: List[str], max_length: Optional[int] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate text for multiple prompts (sequential for now).

        Args:
            prompts: List of prompts to generate from.
            max_length: Maximum generation length.
            **kwargs: Additional arguments passed to generate().

        Returns:
            List of generation results.
        """
        logger.info(f"Batch generation started: {len(prompts)} prompts")

        results = []
        for i, prompt in enumerate(prompts):
            logger.debug(f"Generating for prompt {i + 1}/{len(prompts)}")
            try:
                result = self.generate(prompt, max_length, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate for prompt {i}: {e}")
                # Add empty result to maintain list alignment
                results.append(
                    {
                        "prompt": prompt,
                        "generated_text": prompt,
                        "new_text": "",
                        "error": str(e),
                    }
                )

        return results

    def interactive_generate(
        self,
        initial_prompt: str = "",
        max_length: int = 100,
    ) -> str:
        """Interactive text generation with user control.

        Args:
            initial_prompt: Initial prompt to start with.
            max_length: Maximum length per generation step.

        Returns:
            Final generated text.
        """
        current_text = initial_prompt
        print(f"Starting interactive generation. Current text: '{current_text}'")
        print(
            "Commands: 'continue' to generate more, 'restart' to restart, 'quit' to stop"
        )

        while True:
            try:
                command = (
                    input("\nEnter command (continue/restart/quit): ").strip().lower()
                )

                if command == "quit":
                    break
                elif command == "restart":
                    current_text = input("Enter new prompt: ").strip()
                elif command == "continue":
                    print("Generating...")
                    result = self.generate(current_text, max_length)
                    current_text = result["generated_text"]
                    print(f"\nGenerated text:\n{current_text}")
                else:
                    print("Unknown command. Use 'continue', 'restart', or 'quit'.")

            except KeyboardInterrupt:
                print("\nInteractive generation stopped.")
                break
            except Exception as e:
                print(f"Generation error: {e}")

        return current_text


def load_generator_from_checkpoint(
    checkpoint_path: Union[str, Path],
    config: Optional[GenerationConfig] = None,
    device: Optional[torch.device] = None,
) -> Generator:
    """Load generator from model checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        config: Generation configuration (uses defaults if None).
        device: Device to load model on.

    Returns:
        Configured Generator instance.
    """
    from slm.core.models import load_model_from_checkpoint
    from slm.core.data import Vocabulary

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Create vocabulary
        if "vocab" in checkpoint:
            vocab = Vocabulary(checkpoint["vocab"])
        else:
            raise GenerationError("No vocabulary found in checkpoint")

        # Load model
        model = load_model_from_checkpoint(str(checkpoint_path))

        # Use default config if none provided
        if config is None:
            config = GenerationConfig()

        # Create generator
        generator = Generator(model, vocab, config, device)

        logger.info(f"Generator loaded from checkpoint: {checkpoint_path}")
        return generator

    except Exception as e:
        raise GenerationError(f"Failed to load generator from {checkpoint_path}: {e}")

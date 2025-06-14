"""Data handling and preprocessing for SLM."""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import json

from slm.exceptions import DataError, ValidationError
from slm.utils import validate_text_input, validate_file_path


class Vocabulary:
    """Enhanced vocabulary management with validation and serialization."""

    def __init__(self, chars: Optional[List[str]] = None):
        """Initialize vocabulary.

        Args:
            chars: List of characters. If None, vocabulary is empty.
        """
        if chars is None:
            chars = []

        # Remove duplicates while preserving order
        seen = set()
        self.chars = [c for c in chars if not (c in seen or seen.add(c))]

        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

        logger.debug(f"Vocabulary initialized with {len(self.chars)} characters")

    @classmethod
    def from_text(cls, text: str, min_freq: int = 1) -> "Vocabulary":
        """Create vocabulary from text with frequency filtering.

        Args:
            text: Input text.
            min_freq: Minimum character frequency to include.

        Returns:
            Vocabulary instance.
        """
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1

        # Filter by minimum frequency and sort
        chars = [char for char, count in char_counts.items() if count >= min_freq]
        chars.sort()  # Sort for deterministic ordering

        logger.info(
            f"Vocabulary created from text: {len(chars)} characters (min_freq={min_freq})"
        )
        return cls(chars)

    @classmethod
    def from_file(cls, vocab_path: Union[str, Path]) -> "Vocabulary":
        """Load vocabulary from JSON file.

        Args:
            vocab_path: Path to vocabulary file.

        Returns:
            Vocabulary instance.
        """
        vocab_path = validate_file_path(
            vocab_path, must_exist=True, allowed_extensions=[".json"]
        )

        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                chars = data
            elif isinstance(data, dict) and "chars" in data:
                chars = data["chars"]
            else:
                raise DataError("Invalid vocabulary file format")

            logger.info(f"Vocabulary loaded from {vocab_path}")
            return cls(chars)

        except json.JSONDecodeError as e:
            raise DataError(f"Invalid JSON in vocabulary file: {e}")
        except Exception as e:
            raise DataError(f"Failed to load vocabulary from {vocab_path}: {e}")

    def save(self, vocab_path: Union[str, Path]) -> None:
        """Save vocabulary to JSON file.

        Args:
            vocab_path: Path to save vocabulary.
        """
        vocab_path = Path(vocab_path)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            vocab_data = {
                "chars": self.chars,
                "size": len(self.chars),
                "char2idx": self.char2idx,
            }

            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(vocab_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Vocabulary saved to {vocab_path}")

        except Exception as e:
            raise DataError(f"Failed to save vocabulary to {vocab_path}: {e}")

    def encode(self, text: str, unknown_char: str = "<UNK>") -> List[int]:
        """Encode text to indices.

        Args:
            text: Text to encode.
            unknown_char: Character to use for unknown characters.

        Returns:
            List of character indices.
        """
        # Add unknown character to vocabulary if not present
        if unknown_char not in self.char2idx and any(
            c not in self.char2idx for c in text
        ):
            self._add_char(unknown_char)

        indices = []
        unknown_count = 0

        for char in text:
            if char in self.char2idx:
                indices.append(self.char2idx[char])
            else:
                indices.append(self.char2idx.get(unknown_char, 0))
                unknown_count += 1

        if unknown_count > 0:
            logger.warning(
                f"Encoded {unknown_count} unknown characters as '{unknown_char}'"
            )

        return indices

    def decode(self, indices: List[int], skip_unknown: bool = False) -> str:
        """Decode indices to text.

        Args:
            indices: List of character indices.
            skip_unknown: Whether to skip unknown indices.

        Returns:
            Decoded text.
        """
        chars = []
        unknown_count = 0

        for idx in indices:
            if idx in self.idx2char:
                chars.append(self.idx2char[idx])
            else:
                if not skip_unknown:
                    chars.append("?")
                unknown_count += 1

        if unknown_count > 0:
            logger.warning(f"Decoded {unknown_count} unknown indices")

        return "".join(chars)

    def _add_char(self, char: str) -> None:
        """Add character to vocabulary."""
        if char not in self.char2idx:
            idx = len(self.chars)
            self.chars.append(char)
            self.char2idx[char] = idx
            self.idx2char[idx] = char

    def __len__(self) -> int:
        return len(self.chars)

    def __contains__(self, char: str) -> bool:
        return char in self.char2idx

    def __getitem__(self, key: Union[str, int]) -> Union[int, str]:
        if isinstance(key, str):
            return self.char2idx.get(key, -1)
        elif isinstance(key, int):
            return self.idx2char.get(key, "")
        else:
            raise TypeError(f"Key must be str or int, got {type(key)}")


class TextDataset(Dataset):
    """Enhanced text dataset with validation and preprocessing."""

    def __init__(
        self,
        text: str,
        vocab: Vocabulary,
        seq_length: int = 100,
        stride: Optional[int] = None,
        preprocess: bool = True,
    ):
        """Initialize text dataset.

        Args:
            text: Training text.
            vocab: Vocabulary for encoding.
            seq_length: Length of each sequence.
            stride: Stride between sequences. If None, uses seq_length (no overlap).
            preprocess: Whether to preprocess text.
        """
        # Validate inputs
        text = validate_text_input(text, min_length=seq_length + 1)

        if seq_length < 1:
            raise ValidationError(f"seq_length must be >= 1, got {seq_length}")

        if stride is None:
            stride = seq_length
        elif stride < 1:
            raise ValidationError(f"stride must be >= 1, got {stride}")

        self.seq_length = seq_length
        self.stride = stride
        self.vocab = vocab

        # Preprocess text if requested
        if preprocess:
            text = self._preprocess_text(text)

        # Encode text
        try:
            self.data = vocab.encode(text)
        except Exception as e:
            raise DataError(f"Failed to encode text: {e}")

        # Calculate number of sequences
        if len(self.data) <= seq_length:
            raise DataError(
                f"Text too short: {len(self.data)} chars, need > {seq_length}"
            )

        self.num_sequences = max(0, (len(self.data) - seq_length) // stride + 1)

        logger.info(
            f"TextDataset created: {self.num_sequences} sequences of length {seq_length}"
        )

    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        # Remove null bytes
        text = text.replace("\x00", "")

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive whitespace but preserve structure
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Remove trailing whitespace but preserve leading
            line = line.rstrip()
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sequence.

        Args:
            idx: Sequence index.

        Returns:
            Tuple of (input_sequence, target_sequence).
        """
        if idx >= self.num_sequences:
            raise IndexError(f"Index {idx} out of range [0, {self.num_sequences})")

        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_length + 1

        # Get sequence chunk
        chunk = self.data[start_idx:end_idx]

        # Create input and target sequences
        input_seq = torch.tensor(chunk[:-1], dtype=torch.long)
        target_seq = torch.tensor(chunk[1:], dtype=torch.long)

        return input_seq, target_seq

    def get_text_sample(self, idx: int) -> Tuple[str, str]:
        """Get text representation of a sequence for debugging.

        Args:
            idx: Sequence index.

        Returns:
            Tuple of (input_text, target_text).
        """
        input_seq, target_seq = self[idx]
        input_text = self.vocab.decode(input_seq.tolist())
        target_text = self.vocab.decode(target_seq.tolist())
        return input_text, target_text


def load_text_file(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
    """Load text file with proper error handling.

    Args:
        file_path: Path to text file.
        encoding: File encoding.

    Returns:
        File content as string.
    """
    file_path = validate_file_path(file_path, must_exist=True)

    try:
        with open(file_path, "r", encoding=encoding) as f:
            text = f.read()

        logger.info(f"Loaded text file: {file_path} ({len(text):,} characters)")
        return text

    except UnicodeDecodeError:  # Try alternative encodings
        for alt_encoding in ["latin1", "cp1252", "utf-16"]:
            try:
                with open(file_path, "r", encoding=alt_encoding) as f:
                    text = f.read()
                logger.warning(
                    f"Used alternative encoding {alt_encoding} for {file_path}"
                )
                return text
            except UnicodeDecodeError:
                continue

        raise DataError(
            f"Could not decode file {file_path} with any supported encoding"
        )

    except Exception as e:
        raise DataError(f"Failed to load text file {file_path}: {e}")


def create_data_loaders(
    text: str,
    vocab: Vocabulary,
    seq_length: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.1,
    stride: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, Optional[DataLoader], Dict[str, int]]:
    """Create training and validation data loaders.

    Args:
        text: Training text.
        vocab: Vocabulary for encoding.
        seq_length: Sequence length.
        batch_size: Batch size.
        validation_split: Fraction of data for validation.
        stride: Stride between sequences.
        num_workers: Number of data loader workers.
        pin_memory: Whether to pin memory.
        seed: Random seed for data splitting.

    Returns:
        Tuple of (train_loader, val_loader, data_info).
    """
    # Validate parameters
    if not 0 <= validation_split < 1:
        raise ValidationError(
            f"validation_split must be in [0, 1), got {validation_split}"
        )

    # Split text if validation requested
    val_loader = None
    if validation_split > 0:
        split_idx = int(len(text) * (1 - validation_split))
        train_text = text[:split_idx]
        val_text = text[split_idx:]

        logger.info(
            f"Data split: {len(train_text):,} train, {len(val_text):,} validation"
        )
    else:
        train_text = text
        val_text = None

    # Create datasets
    train_dataset = TextDataset(train_text, vocab, seq_length, stride)

    val_dataset = None
    if val_text:
        val_dataset = TextDataset(val_text, vocab, seq_length, stride)

    # Create data loaders
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        drop_last=True,  # Ensure consistent batch sizes
    )

    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    # Prepare data info
    data_info = {
        "train_sequences": len(train_dataset),
        "val_sequences": len(val_dataset) if val_dataset else 0,
        "vocab_size": len(vocab),
        "seq_length": seq_length,
        "total_chars": len(text),
    }

    logger.info(f"Data loaders created: {data_info}")
    return train_loader, val_loader, data_info


def prepare_data(
    data_path: Union[str, Path],
    seq_length: int = 100,
    vocab_path: Optional[Union[str, Path]] = None,
    min_char_freq: int = 1,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Tuple[str, Vocabulary]:
    """Prepare data for training with caching support.

    Args:
        data_path: Path to text data file.
        seq_length: Sequence length for training.
        vocab_path: Path to existing vocabulary file.
        min_char_freq: Minimum character frequency for vocabulary.
        cache_dir: Directory for caching processed data.

    Returns:
        Tuple of (text, vocabulary).
    """
    data_path = validate_file_path(data_path, must_exist=True)

    # Load text
    text = load_text_file(data_path)

    # Handle vocabulary
    if vocab_path:
        vocab = Vocabulary.from_file(vocab_path)
        logger.info(f"Using existing vocabulary: {len(vocab)} characters")
    else:
        vocab = Vocabulary.from_text(text, min_char_freq)

        # Save vocabulary if cache directory provided
        if cache_dir:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            vocab_cache_path = cache_dir / "vocabulary.json"
            vocab.save(vocab_cache_path)

    # Validate data compatibility
    if len(text) < seq_length + 1:
        raise DataError(f"Text too short for seq_length {seq_length}")

    # Check vocabulary coverage
    text_chars = set(text)
    vocab_chars = set(vocab.chars)
    missing_chars = text_chars - vocab_chars

    if missing_chars:
        logger.warning(
            f"Text contains {len(missing_chars)} characters not in vocabulary"
        )
        if len(missing_chars) < 10:
            logger.warning(f"Missing characters: {sorted(missing_chars)}")

    logger.info(
        f"Data prepared: {len(text):,} characters, vocabulary size: {len(vocab)}"
    )
    return text, vocab

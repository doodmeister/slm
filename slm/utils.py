"""Utility functions for SLM."""

import os
import random
import hashlib
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import torch
import numpy as np
from loguru import logger

from slm.exceptions import ValidationError, SecurityError, ResourceError


def set_seed(seed: Optional[int] = None, deterministic: bool = False) -> int:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed to set. If None, generates a random seed.
        deterministic: Whether to enable deterministic algorithms.

    Returns:
        The seed that was set.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set environment variable for deterministic behavior
        os.environ["PYTHONHASHSEED"] = str(seed)

    logger.debug(f"Random seed set to: {seed}, deterministic: {deterministic}")
    return seed


def get_device(device_name: Optional[str] = None) -> torch.device:
    """Get the appropriate torch device.

    Args:
        device_name: Specific device name ('cpu', 'cuda', 'cuda:0', etc.)
                    If None, auto-selects best available device.

    Returns:
        torch.device object.
    """
    if device_name is not None:
        try:
            device = torch.device(device_name)
            if device.type == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = torch.device("cpu")
        except Exception as e:
            logger.warning(
                f"Invalid device '{device_name}': {e}, falling back to auto-select"
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Using device: {device}")
    return device


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count model parameters.

    Args:
        model: PyTorch model.

    Returns:
        Tuple of (total_params, trainable_params).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get approximate model size in MB.

    Args:
        model: PyTorch model.

    Returns:
        Model size in megabytes.
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def validate_text_input(text: str, min_length: int = 1, max_length: int = 10**6) -> str:
    """Validate and sanitize text input.

    Args:
        text: Input text to validate.
        min_length: Minimum allowed length.
        max_length: Maximum allowed length.

    Returns:
        Validated and sanitized text.

    Raises:
        ValidationError: If validation fails.
    """
    if not isinstance(text, str):
        raise ValidationError(f"Expected string, got {type(text)}")

    if len(text) < min_length:
        raise ValidationError(f"Text too short: {len(text)} < {min_length}")

    if len(text) > max_length:
        raise ValidationError(f"Text too long: {len(text)} > {max_length}")

    # Basic sanitization - remove null bytes and excessive whitespace
    text = text.replace("\x00", "").strip()

    # Check for potentially problematic characters
    if any(ord(c) < 32 and c not in "\n\t\r" for c in text):
        logger.warning("Text contains control characters, they will be preserved")

    return text


def validate_file_path(
    file_path: Union[str, Path],
    must_exist: bool = True,
    allowed_extensions: Optional[List[str]] = None,
) -> Path:
    """Validate file path for security and existence.

    Args:
        file_path: Path to validate.
        must_exist: Whether file must exist.
        allowed_extensions: List of allowed file extensions.

    Returns:
        Validated Path object.

    Raises:
        ValidationError: If validation fails.
        SecurityError: If path is potentially unsafe.
    """
    try:
        path = Path(file_path).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid path: {e}")

    # Security check - prevent path traversal
    if ".." in str(file_path):
        raise SecurityError("Path traversal detected")

    # Check if file exists
    if must_exist and not path.exists():
        raise ValidationError(f"File does not exist: {path}")

    # Check file extension
    if allowed_extensions and path.suffix.lower() not in allowed_extensions:
        raise ValidationError(f"Invalid file extension. Allowed: {allowed_extensions}")

    return path


def safe_mkdir(directory: Union[str, Path], mode: int = 0o755) -> Path:
    """Safely create directory with proper permissions.

    Args:
        directory: Directory to create.
        mode: Directory permissions.

    Returns:
        Path object of created directory.
    """
    path = Path(directory)
    try:
        path.mkdir(parents=True, exist_ok=True, mode=mode)
        logger.debug(f"Directory created/verified: {path}")
        return path
    except Exception as e:
        raise ResourceError(f"Failed to create directory {path}: {e}")


def compute_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Compute hash of a file.

    Args:
        file_path: Path to file.
        algorithm: Hash algorithm to use.

    Returns:
        Hex digest of file hash.
    """
    path = Path(file_path)
    if not path.exists():
        raise ValidationError(f"File does not exist: {path}")

    hash_obj = hashlib.new(algorithm)
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        raise ResourceError(f"Failed to compute hash for {path}: {e}")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted duration string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def format_bytes(bytes_count: int) -> str:
    """Format bytes to human-readable string.

    Args:
        bytes_count: Number of bytes.

    Returns:
        Formatted bytes string.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f}{unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f}PB"


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage information.

    Returns:
        Dictionary with memory usage stats in MB.
    """
    import psutil
    import torch

    memory_info = {
        "system_used": psutil.virtual_memory().used / 1024 / 1024,
        "system_total": psutil.virtual_memory().total / 1024 / 1024,
        "system_percent": psutil.virtual_memory().percent,
    }

    if torch.cuda.is_available():
        memory_info.update(
            {
                "gpu_allocated": torch.cuda.memory_allocated() / 1024 / 1024,
                "gpu_cached": torch.cuda.memory_reserved() / 1024 / 1024,
                "gpu_total": torch.cuda.get_device_properties(0).total_memory
                / 1024
                / 1024,
            }
        )

    return memory_info


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for filesystem safety.

    Args:
        filename: Original filename.

    Returns:
        Sanitized filename.
    """
    # Remove/replace problematic characters
    import re

    # Replace spaces and special characters
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    filename = re.sub(r"\s+", "_", filename)
    filename = filename.strip("._")

    # Ensure filename isn't empty
    if not filename:
        filename = "untitled"

    # Truncate if too long
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[: 255 - len(ext)] + ext

    return filename


class ProgressTracker:
    """Simple progress tracking utility."""

    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self._last_percent = -1

    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self.current = min(self.current + n, self.total)
        percent = int(100 * self.current / self.total)

        # Only log every 10% to avoid spam
        if percent >= self._last_percent + 10:
            logger.info(f"{self.description}: {percent}% ({self.current}/{self.total})")
            self._last_percent = percent

    def finish(self) -> None:
        """Mark progress as finished."""
        self.current = self.total
        logger.info(f"{self.description}: Complete!")


def chunks(lst: List[Any], n: int) -> List[List[Any]]:
    """Split list into chunks of size n."""
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def ensure_dir_exists(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

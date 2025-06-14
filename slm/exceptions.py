"""Custom exceptions for SLM."""

from typing import Optional


class SLMException(Exception):
    """Base exception for all SLM errors."""

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class ConfigurationError(SLMException):
    """Raised when there's an error in configuration."""

    pass


class ModelError(SLMException):
    """Raised when there's an error with model operations."""

    pass


class TrainingError(SLMException):
    """Raised when there's an error during training."""

    pass


class GenerationError(SLMException):
    """Raised when there's an error during text generation."""

    pass


class DataError(SLMException):
    """Raised when there's an error with data processing."""

    pass


class CheckpointError(SLMException):
    """Raised when there's an error with checkpoint operations."""

    pass


class ValidationError(SLMException):
    """Raised when input validation fails."""

    pass


class ResourceError(SLMException):
    """Raised when there's an error with resource management."""

    pass


class SecurityError(SLMException):
    """Raised when there's a security-related error."""

    pass

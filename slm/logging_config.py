"""Logging configuration for SLM."""

import sys
from pathlib import Path
from typing import Optional, Union
from loguru import logger

from slm.exceptions import ConfigurationError


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: Union[str, Path] = "logs",
    log_filename: str = "slm.log",
    rotation: str = "10 MB",
    retention: str = "7 days",
    format_string: Optional[str] = None,
) -> None:
    """Setup logging configuration.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        log_to_file: Whether to log to file.
        log_dir: Directory for log files.
        log_filename: Log file name.
        rotation: Log rotation policy.
        retention: Log retention policy.
        format_string: Custom format string.
    """
    try:
        # Remove default handler
        logger.remove()

        # Default format
        if format_string is None:
            format_string = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )

        # Console handler
        logger.add(
            sys.stderr,
            format=format_string,
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

        # File handler
        if log_to_file:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / log_filename

            logger.add(
                log_path,
                format=format_string,
                level=level,
                rotation=rotation,
                retention=retention,
                compression="gz",
                backtrace=True,
                diagnose=True,
            )

            logger.info(f"Logging to file: {log_path}")

        logger.info(f"Logging initialized at level: {level}")

    except Exception as e:
        raise ConfigurationError(f"Failed to setup logging: {e}")


def get_logger(name: str) -> "logger":
    """Get a logger instance with the given name.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    return logger.bind(name=name)


class LoggingContext:
    """Context manager for temporary logging configuration."""

    def __init__(self, level: str = "INFO", capture: bool = False):
        self.level = level
        self.capture = capture
        self._handler_id = None
        self._original_handlers = None

    def __enter__(self):
        if self.capture:
            # Remove existing handlers and add capturing handler
            self._original_handlers = logger._core.handlers.copy()
            logger.remove()

            from io import StringIO

            self._log_stream = StringIO()
            self._handler_id = logger.add(
                self._log_stream,
                level=self.level,
                format="{level} | {message}",
            )
            return self._log_stream
        else:
            # Just change level temporarily
            logger.level(self.level)
            return logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.capture and self._handler_id is not None:
            logger.remove(self._handler_id)
            # Restore original handlers
            if self._original_handlers:
                logger._core.handlers.update(self._original_handlers)

"""
Logging Configuration for HSR Perception App

Provides a unified logging interface for all application modules.
Supports console output with configurable log levels.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started")
    logger.error("Failed to load file", exc_info=True)
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Module-level cache for loggers
_loggers: dict[str, logging.Logger] = {}


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> logging.Logger:
    """
    Get or create a logger with the specified configuration.

    Args:
        name: Logger name, typically __name__ of the calling module
        level: Logging level (default: logging.INFO)
        log_format: Custom log format string (optional)
        date_format: Custom date format string (optional)

    Returns:
        Configured logging.Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
        >>> logger.debug("Debug information")
        >>> logger.warning("Warning message")
        >>> logger.error("Error occurred", exc_info=True)
    """
    # Return cached logger if exists
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Formatter
        formatter = logging.Formatter(
            fmt=log_format or DEFAULT_FORMAT,
            datefmt=date_format or DEFAULT_DATE_FORMAT,
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False

    _loggers[name] = logger
    return logger


def set_log_level(name: str, level: int) -> None:
    """
    Set the log level for an existing logger.

    Args:
        name: Logger name
        level: New logging level
    """
    if name in _loggers:
        _loggers[name].setLevel(level)
        for handler in _loggers[name].handlers:
            handler.setLevel(level)


def add_file_handler(
    name: str,
    log_file: Path,
    level: int = logging.DEBUG,
    log_format: Optional[str] = None,
) -> None:
    """
    Add a file handler to an existing logger.

    Args:
        name: Logger name
        log_file: Path to log file
        level: File logging level
        log_format: Custom format for file logs
    """
    if name not in _loggers:
        return

    logger = _loggers[name]

    # Create log directory if needed
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt=log_format or DEFAULT_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

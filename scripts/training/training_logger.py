"""
Training module logging configuration.

Provides structured logging that replaces print statements with colorama.
Supports both console (colorized) and file logging.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from colorama import Fore, Style, init as colorama_init

colorama_init()


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds color to log levels for console output."""

    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color based on level."""
        # Add color based on level
        color = self.COLORS.get(record.levelno, "")
        reset = Style.RESET_ALL if color else ""

        # Format message
        original_msg = record.msg
        record.msg = f"{color}{original_msg}{reset}"
        result = super().format(record)
        record.msg = original_msg  # Restore for other handlers

        return result


def setup_training_logger(
    name: str = "training",
    level: Optional[int] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Setup and return a configured logger for training modules.

    Args:
        name: Logger name (default: "training")
        level: Logging level (defaults to env var HSR_LOG_LEVEL or INFO)
        log_file: Optional file path for log output

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Determine level from environment or default
    if level is None:
        env_level = os.environ.get("HSR_LOG_LEVEL", "INFO")
        level = getattr(logging, env_level.upper(), logging.INFO)

    logger.setLevel(level)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "training") -> logging.Logger:
    """
    Get or create a logger for the training module.

    Args:
        name: Logger name (default: "training")

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_training_logger(name)
    return logger


# Module-level logger instance for convenience
logger = setup_training_logger()

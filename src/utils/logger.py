"""
Logging utility for the Lung Cancer Prediction project.

Provides a centralized logger that writes to both console and file.
Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Message here")
"""

import logging
import sys
from pathlib import Path

from src.utils.config import LOG_FILE, LOG_FORMAT, LOG_DATE_FORMAT


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a configured logger instance.

    Args:
        name: Name of the logger (typically __name__).
        level: Logging level (default: INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ── Console Handler ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)

    # ── File Handler ──
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)

    # ── Attach Handlers ──
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

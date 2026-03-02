"""
Centralized logging setup using loguru.
All agents import `logger` from here to ensure consistent formatting.
"""

import sys
from pathlib import Path

from loguru import logger as _logger

from config.settings import settings


def setup_logger() -> None:
    """
    Configure loguru with:
    - Console sink (colored, human-readable)
    - File sink (JSON-structured, rotated daily, retained 30 days)
    """
    _logger.remove()  # Remove default handler

    # Console — human-readable with color
    _logger.add(
        sys.stdout,
        level=settings.log_level.value,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File — JSON structured for observability
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    _logger.add(
        str(log_path),
        level=settings.log_level.value,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} — {message}",
        rotation="1 day",
        retention="30 days",
        compression="zip",
        serialize=True,
    )


setup_logger()

# Export the configured logger instance
logger = _logger

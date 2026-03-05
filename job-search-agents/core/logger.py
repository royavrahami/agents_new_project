"""
Centralized logging setup using loguru.
All agents import `logger` from here to ensure consistent formatting.
"""

import sys
from pathlib import Path

from loguru import logger as _logger

from config.settings import settings


def _enrich_record(record: dict) -> None:
    """Ensure each log record contains stable observability fields."""
    record["extra"].setdefault("agent", "system")
    record["extra"].setdefault("correlation_id", "-")


def setup_logger() -> None:
    """
    Configure loguru with:
    - Console sink (colored, human-readable)
    - File sink (JSON-structured, rotated daily, retained 30 days)
    """
    _logger.remove()  # Remove default handler
    patched_logger = _logger.patch(_enrich_record)

    # Console — human-readable with color
    patched_logger.add(
        sys.stdout,
        level=settings.log_level.value,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[agent]}</cyan> | <cyan>{extra[correlation_id]}</cyan> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File — JSON structured for observability
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    patched_logger.add(
        str(log_path),
        level=settings.log_level.value,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[agent]} | "
            "{extra[correlation_id]} | {name}:{function}:{line} — {message}"
        ),
        rotation="1 day",
        retention="30 days",
        compression="zip",
        serialize=True,
    )


setup_logger()

# Export the configured logger instance
logger = _logger.patch(_enrich_record)

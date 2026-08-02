"""Small opt-in logging configuration for reproducible CLI runs."""

from __future__ import annotations

import logging
from pathlib import Path

from game_player_analysis.config import LOG_DIR


def configure_logging(
    *,
    level: int = logging.INFO,
    log_path: str | Path | None = None,
) -> logging.Logger:
    """Configure the project logger without mutating unrelated root handlers."""
    logger = logging.getLogger("game_player_analysis")
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        destination = Path(log_path)
        if not destination.is_absolute():
            destination = LOG_DIR / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(destination, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

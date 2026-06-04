"""
Centralised logging configuration for trading_app.

Import and call `configure_logging()` once from `app.py` at startup.
Individual modules use `logging.getLogger(__name__)` and do not configure
handlers themselves — they just emit; this module handles the plumbing.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """Configure root logger for the trading application.

    Sets up a StreamHandler to stderr and, optionally, a FileHandler.
    Safe to call multiple times — subsequent calls are no-ops if handlers
    are already attached to the root logger (Streamlit reruns this file on
    every interaction, so the guard is essential to avoid duplicate output).

    Args:
        level: Logging level for both handlers. Defaults to INFO.
            Pass ``logging.DEBUG`` during development for verbose output.
        log_file: Optional path to a log file. If None, only stderr
            output is configured.

    Returns:
        None
    """
    root = logging.getLogger()
    if root.handlers:
        # Already configured — avoid duplicate handlers on Streamlit reruns.
        return

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    root.setLevel(level)

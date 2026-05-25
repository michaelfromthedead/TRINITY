"""Utility helpers for Spin Physics Research project."""

import logging
import tomllib
from pathlib import Path


def load_config(path: str) -> dict:
    """Load a TOML configuration file and return as dict."""
    with open(Path(path), "rb") as fh:
        return tomllib.load(fh)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logger and return it."""
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=level)
    return logging.getLogger("spin_physics")

"""Unit tests for spin_physics main entry point."""

import pytest
from unittest.mock import patch
from src.main import run


def test_run_calls_load_config():
    with patch("src.main.load_config", return_value={"log_level": "WARNING"}) as mock_cfg:
        with patch("src.main.setup_logging"):
            run("config/default.toml")
    mock_cfg.assert_called_once_with("config/default.toml")


def test_run_defaults_log_level():
    with patch("src.main.load_config", return_value={}):
        with patch("src.main.setup_logging") as mock_log:
            run("config/default.toml")
    mock_log.assert_called_once_with("INFO")

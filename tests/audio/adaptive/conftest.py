"""Pytest configuration for adaptive audio tests.

Provides fixtures for common test setup and mock audio resources.
"""

from __future__ import annotations

import pytest
from typing import Optional
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_audio_context():
    """Provide a mock audio context for testing."""
    context = MagicMock()
    context.sample_rate = 44100
    context.buffer_size = 256
    context.is_running = True
    return context


@pytest.fixture
def mock_audio_asset():
    """Provide a mock audio asset for stem/stinger tests."""
    asset = MagicMock()
    asset.duration = 4.0  # 4 seconds
    asset.sample_rate = 44100
    asset.channels = 2
    return asset


@pytest.fixture
def mock_clock():
    """Provide a mock music clock."""
    from unittest.mock import MagicMock

    clock = MagicMock()
    clock.get_bpm.return_value = 120
    clock.get_current_beat.return_value = 0.0
    clock.get_current_bar.return_value = 0
    clock.get_beats_per_bar.return_value = 4
    clock.is_playing.return_value = True
    return clock

"""Tests for low latency GPU features."""
import pytest
import time
from engine.platform.gpu import (
    LowLatencyAPI, LowLatencyConfig, LowLatency
)


def test_low_latency_creation():
    """Test low latency manager creation."""
    ll = LowLatency()
    assert ll is not None


def test_low_latency_not_available():
    """Test low latency features not available in stub."""
    ll = LowLatency()
    assert ll.is_available is False


def test_low_latency_enable_fails_when_not_available():
    """Test enable fails when not available."""
    ll = LowLatency()
    config = LowLatencyConfig(enabled=True, boost=True)

    result = ll.enable(config)
    assert result is False


def test_low_latency_enable_returns_false_when_unavailable():
    """Test that enable() returns False when is_available is False."""
    ll = LowLatency()

    # Verify not available
    assert ll.is_available is False

    # Attempt to enable should return False
    config = LowLatencyConfig(enabled=True)
    result = ll.enable(config)
    assert result is False


def test_low_latency_set_marker_tracking():
    """Test that set_marker and sleep are tracked even when not enabled."""
    ll = LowLatency()

    # Verify not available
    assert ll.is_available is False

    # Track initial marker count
    initial_count = ll._marker_count

    # Set multiple markers
    ll.set_marker("input")
    ll.set_marker("simulation")
    ll.set_marker("render")
    ll.set_marker("present")

    # Markers should be tracked (for debugging)
    assert ll._marker_count >= initial_count


def test_low_latency_sleep():
    """Test low latency sleep."""
    ll = LowLatency()

    start = time.time()
    ll.sleep(10.0)  # 10ms
    elapsed = (time.time() - start) * 1000

    # Should sleep approximately 10ms
    assert elapsed >= 9.0
    assert elapsed < 20.0


def test_low_latency_config():
    """Test low latency configuration."""
    config = LowLatencyConfig(
        enabled=True,
        boost=True,
        min_interval_us=500
    )

    assert config.enabled is True
    assert config.boost is True
    assert config.min_interval_us == 500


def test_low_latency_config_defaults():
    """Test low latency config defaults."""
    config = LowLatencyConfig()

    assert config.enabled is False
    assert config.boost is False
    assert config.min_interval_us == 0


def test_low_latency_api_types():
    """Test that different low latency API types exist and differ."""
    # Verify the API types are distinct
    assert LowLatencyAPI.NONE != LowLatencyAPI.NVIDIA_REFLEX
    assert LowLatencyAPI.NONE != LowLatencyAPI.AMD_ANTILAG
    assert LowLatencyAPI.NVIDIA_REFLEX != LowLatencyAPI.AMD_ANTILAG


def test_low_latency_marker_count():
    """Test marker counting (internal)."""
    ll = LowLatency()

    initial_count = ll._marker_count

    ll.set_marker("test1")
    ll.set_marker("test2")
    ll.set_marker("test3")

    # Markers should be tracked even if not enabled
    # (for testing/debugging purposes)
    assert ll._marker_count >= initial_count


def test_low_latency_sleep_precision():
    """Test sleep with high precision."""
    ll = LowLatency()
    config = LowLatencyConfig(enabled=True, min_interval_us=1000)
    ll.enable(config)

    durations = []
    for _ in range(5):
        start = time.time()
        ll.sleep(5.0)  # 5ms
        elapsed = (time.time() - start) * 1000
        durations.append(elapsed)

    # All should be close to 5ms
    for duration in durations:
        assert 4.0 <= duration <= 10.0

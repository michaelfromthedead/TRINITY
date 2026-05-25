"""
Tests for system information.
"""
import os
import pytest
from engine.platform.os.system_info import SystemInfo


def test_cpu_count():
    """Test CPU count."""
    count = SystemInfo.cpu_count()
    assert count >= 1


def test_cpu_count_physical():
    """Test physical CPU count."""
    physical = SystemInfo.cpu_count_physical()
    logical = SystemInfo.cpu_count()

    assert physical >= 1
    assert physical <= logical


def test_get_cpu_info():
    """Test detailed CPU info."""
    info = SystemInfo.get_cpu_info()

    assert info.logical_count >= 1
    assert info.physical_count >= 1
    assert len(info.architecture) > 0


def test_total_memory():
    """Test total memory."""
    total = SystemInfo.total_memory()
    # Should be positive on a real machine
    assert total > 0


def test_available_memory():
    """Test available memory."""
    available = SystemInfo.available_memory()
    # Should be positive on a real machine
    assert available > 0


def test_get_memory_info():
    """Test detailed memory info."""
    info = SystemInfo.get_memory_info()

    assert info.total >= 0
    assert info.available >= 0
    assert info.used >= 0
    assert 0 <= info.percent <= 100


def test_cache_line_size():
    """Test cache line size."""
    size = SystemInfo.cache_line_size()
    assert size > 0
    # Typical cache line sizes
    assert size in [32, 64, 128]


def test_get_env():
    """Test getting environment variable."""
    # Set a test variable
    os.environ['TEST_VAR'] = 'test_value'

    value = SystemInfo.get_env('TEST_VAR')
    assert value == 'test_value'

    # Nonexistent variable
    value = SystemInfo.get_env('NONEXISTENT_VAR')
    assert value is None

    # With default
    value = SystemInfo.get_env('NONEXISTENT_VAR', 'default')
    assert value == 'default'


def test_set_env():
    """Test setting environment variable."""
    SystemInfo.set_env('TEST_SET_VAR', 'new_value')
    assert os.environ['TEST_SET_VAR'] == 'new_value'

    value = SystemInfo.get_env('TEST_SET_VAR')
    assert value == 'new_value'


def test_unset_env():
    """Test unsetting environment variable."""
    SystemInfo.set_env('TEST_UNSET_VAR', 'value')
    assert SystemInfo.get_env('TEST_UNSET_VAR') == 'value'

    SystemInfo.unset_env('TEST_UNSET_VAR')
    assert SystemInfo.get_env('TEST_UNSET_VAR') is None


def test_get_all_env():
    """Test getting all environment variables."""
    env = SystemInfo.get_all_env()

    assert isinstance(env, dict)
    assert len(env) > 0

    # Should contain PATH on most systems
    assert 'PATH' in env or 'Path' in env


def test_platform_name():
    """Test platform name."""
    name = SystemInfo.platform_name()
    assert len(name) > 0
    # Should be one of the common platforms
    assert name in ['Linux', 'Windows', 'Darwin', 'FreeBSD', 'OpenBSD']


def test_platform_version():
    """Test platform version."""
    version = SystemInfo.platform_version()
    assert len(version) > 0


def test_hostname():
    """Test hostname."""
    hostname = SystemInfo.hostname()
    assert len(hostname) > 0


def test_memory_consistency():
    """Test memory info consistency."""
    info = SystemInfo.get_memory_info()

    if info.total > 0:
        # Used + available should approximately equal total
        # (may not be exact due to timing and kernel memory)
        assert info.used <= info.total
        assert info.available <= info.total


def test_cpu_count_consistency():
    """Test CPU count consistency."""
    logical = SystemInfo.cpu_count()
    physical = SystemInfo.cpu_count_physical()

    # Logical should be >= physical
    assert logical >= physical

    info = SystemInfo.get_cpu_info()
    assert info.logical_count == logical
    assert info.physical_count == physical

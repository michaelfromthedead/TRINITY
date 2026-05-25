"""Tests for platform detection."""
import pytest
from engine.platform.services import (
    PlatformType, PlatformInfo, detect
)


def test_platform_detect():
    """Test platform detection."""
    info = detect()

    assert isinstance(info, PlatformInfo)
    assert isinstance(info.type, PlatformType)
    assert info.name
    assert info.version
    assert info.arch


def test_platform_is_linux():
    """Test that platform is detected as Linux."""
    info = detect()

    # On Linux CI/test environment
    assert info.type == PlatformType.LINUX
    assert info.name == "Linux"
    assert info.is_desktop is True
    assert info.is_mobile is False
    assert info.is_console is False


def test_platform_info_fields():
    """Test platform info has all required fields."""
    info = detect()

    assert hasattr(info, 'type')
    assert hasattr(info, 'name')
    assert hasattr(info, 'version')
    assert hasattr(info, 'arch')
    assert hasattr(info, 'is_console')
    assert hasattr(info, 'is_mobile')
    assert hasattr(info, 'is_desktop')


def test_platform_type_detection_specific():
    """Test platform detection returns LINUX on this machine."""
    info = detect()

    # On this Linux test machine, should detect LINUX specifically
    assert info.type == PlatformType.LINUX

    # Verify other platform types are different
    assert info.type != PlatformType.WINDOWS
    assert info.type != PlatformType.MACOS


def test_platform_arch_not_empty():
    """Test that architecture is detected."""
    info = detect()
    assert len(info.arch) > 0


def test_platform_exactly_one_category():
    """Test that platform is in exactly one category."""
    info = detect()

    categories = [info.is_desktop, info.is_mobile, info.is_console]
    assert sum(categories) == 1  # Exactly one should be True

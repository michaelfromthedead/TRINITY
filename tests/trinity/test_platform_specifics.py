"""
Tests for Trinity Pattern Tier 32: PLATFORM decorators.
"""

import pytest

from trinity.decorators.ops import Op, decompose
from trinity.decorators.platform_specifics import (
    VALID_BATTERY_MODES,
    battery_aware,
)
from trinity.decorators.registry import Tier, registry


# =============================================================================
# @battery_aware tests
# =============================================================================


def test_battery_aware_basic():
    """Test basic @battery_aware application."""

    @battery_aware(mode="performance")
    class GameSystem:
        pass

    assert hasattr(GameSystem, "_battery_aware")
    assert GameSystem._battery_aware is True
    assert GameSystem._battery_mode == "performance"
    assert "battery_aware" in GameSystem._applied_decorators


def test_battery_aware_all_modes():
    """Test all valid battery modes."""
    for mode in VALID_BATTERY_MODES:

        @battery_aware(mode=mode)
        class System:
            pass

        assert System._battery_mode == mode


def test_battery_aware_default():
    """Test @battery_aware with default parameters."""

    @battery_aware
    class System:
        pass

    # Default mode should be "balanced"
    assert System._battery_mode == "balanced"


def test_battery_aware_invalid_mode():
    """Test @battery_aware with invalid mode raises ValueError."""
    with pytest.raises(ValueError, match="Invalid battery mode"):

        @battery_aware(mode="turbo")
        class System:
            pass


def test_battery_aware_tags():
    """Test that @battery_aware sets proper tags."""

    @battery_aware(mode="battery_saver")
    class System:
        pass

    assert hasattr(System, "_tags")
    assert System._tags.get("battery_aware") is True
    assert System._tags.get("battery_mode") == "battery_saver"


def test_battery_aware_registry():
    """Test that @battery_aware registers properly."""

    @battery_aware(mode="balanced")
    class System:
        pass

    assert "platform" in System._registries


def test_battery_aware_steps():
    """Test that @battery_aware generates correct steps."""
    steps = decompose(battery_aware)

    # Should have TAG and REGISTER steps
    ops = [s.op for s in steps]
    assert Op.TAG in ops
    assert Op.REGISTER in ops


def test_battery_aware_registry_spec():
    """Test that @battery_aware is registered in the decorator registry."""
    spec = registry.get("battery_aware")
    assert spec is not None
    assert spec.name == "battery_aware"
    assert spec.tier == Tier.PLATFORM
    assert spec.unique is True
    assert spec.foundation is False
    assert "class" in spec.target_types


def test_battery_aware_composition():
    """Test @battery_aware with other decorators."""
    from trinity.decorators.compilation import native

    @battery_aware(mode="performance")
    @native(backend="cython")
    class OptimizedSystem:
        pass

    assert OptimizedSystem._battery_aware is True
    assert OptimizedSystem._native is True
    assert "battery_aware" in OptimizedSystem._applied_decorators
    assert "native" in OptimizedSystem._applied_decorators


def test_battery_aware_unique():
    """Test that @battery_aware can only be applied once."""
    # Note: The make_decorator factory allows applying once directly,
    # but subsequent applications would override. The registry marks it as unique.
    spec = registry.get("battery_aware")
    assert spec.unique is True


def test_battery_aware_parameterized():
    """Test @battery_aware with all parameters."""

    @battery_aware(mode="balanced")
    class System:
        pass

    assert System._battery_aware is True
    assert System._battery_mode == "balanced"


def test_battery_aware_step_count():
    """Test that @battery_aware generates the correct number of steps."""
    steps = decompose(battery_aware)
    # Should have 3 steps: TAG (battery_aware), TAG (battery_mode), REGISTER
    assert len(steps) == 3


def test_battery_aware_applied_steps():
    """Test that steps are recorded when decorator is applied."""

    @battery_aware(mode="performance")
    class System:
        pass

    assert hasattr(System, "_applied_steps")
    assert len(System._applied_steps) == 3


# =============================================================================
# Module exports tests
# =============================================================================


def test_module_exports():
    """Test that module exports expected symbols."""
    from trinity.decorators import platform_specifics

    assert hasattr(platform_specifics, "battery_aware")
    assert hasattr(platform_specifics, "VALID_BATTERY_MODES")
    assert "battery_aware" in platform_specifics.__all__
    assert "VALID_BATTERY_MODES" in platform_specifics.__all__


# =============================================================================
# Integration tests
# =============================================================================


def test_battery_aware_with_multiple_tags():
    """Test @battery_aware preserves existing tags."""

    class System:
        _tags = {"custom": "value"}

    battery_aware(mode="performance")(System)

    assert System._tags.get("custom") == "value"
    assert System._tags.get("battery_aware") is True
    assert System._tags.get("battery_mode") == "performance"


def test_battery_aware_introspection():
    """Test introspection of @battery_aware decorated classes."""
    from trinity.decorators.registry import get_decorator_chain, inspect_decorated

    @battery_aware(mode="battery_saver")
    class System:
        pass

    chain = get_decorator_chain(System)
    assert "battery_aware" in chain

    info = inspect_decorated(System)
    assert "battery_aware" in info.decorators
    assert info.attributes.get("_battery_aware") is True
    assert info.attributes.get("_battery_mode") == "battery_saver"

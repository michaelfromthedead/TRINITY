"""
Blackbox contract tests for T2.5 Behavior Configuration.

These tests validate ONLY the public interface without reading implementation.
Tests are derived from the contract specification in PHASE_2_TODO.md.

Contract Requirements:
1. All behavior parameters are configurable
2. Config changes take effect without restart
3. Invalid config values are rejected
4. Default values are sensible

Note: The actual public API is at engine.animation.config.CROWD_BEHAVIOR_CONFIG
(not engine.animation.crowds.config as stated in original contract)
"""

import pytest
from typing import Any

# Canonical import path for CROWD_BEHAVIOR_CONFIG
CONFIG_MODULE = "engine.animation.config"
ALT_CONFIG_MODULE = "engine.animation.crowds.crowd_behavior"


class TestBehaviorConfigPublicContract:
    """
    Blackbox tests for CROWD_BEHAVIOR_CONFIG public interface.
    Tests only the contract, not implementation details.
    """

    def test_config_module_importable(self):
        """Config module should be importable from public API."""
        try:
            from engine.animation.config import CROWD_BEHAVIOR_CONFIG
        except ImportError as e:
            pytest.fail(f"CROWD_BEHAVIOR_CONFIG not importable: {e}")

    def test_config_is_singleton_instance(self):
        """Config should be available as a singleton instance."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        # Should be an object, not a class
        assert CROWD_BEHAVIOR_CONFIG is not None
        assert not isinstance(CROWD_BEHAVIOR_CONFIG, type)

    def test_config_also_exported_from_crowd_behavior(self):
        """Config should also be accessible from crowd_behavior module."""
        from engine.animation.crowds.crowd_behavior import CROWD_BEHAVIOR_CONFIG
        assert CROWD_BEHAVIOR_CONFIG is not None


class TestDefaultValuesSensible:
    """
    Contract: Default values are sensible.
    Tests that default configuration values meet minimum sanity requirements.
    """

    def test_min_distance_epsilon_positive(self):
        """MIN_DISTANCE_EPSILON must be > 0 (contract requirement)."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        assert hasattr(CROWD_BEHAVIOR_CONFIG, 'MIN_DISTANCE_EPSILON'), \
            "Config must have MIN_DISTANCE_EPSILON attribute"
        assert CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON > 0, \
            "MIN_DISTANCE_EPSILON must be positive"

    def test_avoidance_priority_multiplier_at_least_one(self):
        """AVOIDANCE_PRIORITY_MULTIPLIER must be >= 1.0 (contract requirement)."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        assert hasattr(CROWD_BEHAVIOR_CONFIG, 'AVOIDANCE_PRIORITY_MULTIPLIER'), \
            "Config must have AVOIDANCE_PRIORITY_MULTIPLIER attribute"
        assert CROWD_BEHAVIOR_CONFIG.AVOIDANCE_PRIORITY_MULTIPLIER >= 1.0, \
            "AVOIDANCE_PRIORITY_MULTIPLIER must be >= 1.0"

    def test_avoidance_radius_positive_default(self):
        """Default avoidance_radius should be positive."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        if hasattr(CROWD_BEHAVIOR_CONFIG, 'avoidance_radius'):
            assert CROWD_BEHAVIOR_CONFIG.avoidance_radius > 0, \
                "Default avoidance_radius should be positive"

    def test_default_values_are_numeric(self):
        """All config values should be numeric types."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        for attr in ['MIN_DISTANCE_EPSILON', 'AVOIDANCE_PRIORITY_MULTIPLIER']:
            if hasattr(CROWD_BEHAVIOR_CONFIG, attr):
                value = getattr(CROWD_BEHAVIOR_CONFIG, attr)
                assert isinstance(value, (int, float)), \
                    f"{attr} must be numeric, got {type(value)}"


class TestInvalidConfigRejected:
    """
    Contract: Invalid config values are rejected.
    Tests that the config properly validates and rejects invalid values.
    """

    def test_negative_avoidance_radius_rejected(self):
        """Negative avoidance_radius must raise ValueError (contract requirement)."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        with pytest.raises(ValueError):
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = -1

    def test_zero_avoidance_radius_rejected(self):
        """Zero avoidance_radius should be rejected."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        with pytest.raises(ValueError):
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = 0

    def test_negative_min_distance_epsilon_rejected(self):
        """Negative MIN_DISTANCE_EPSILON should be rejected if settable."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        # Only test if the attribute is settable
        if hasattr(type(CROWD_BEHAVIOR_CONFIG), 'MIN_DISTANCE_EPSILON') and \
           isinstance(getattr(type(CROWD_BEHAVIOR_CONFIG), 'MIN_DISTANCE_EPSILON', None), property):
            with pytest.raises(ValueError):
                CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON = -0.001

    def test_negative_avoidance_priority_rejected(self):
        """AVOIDANCE_PRIORITY_MULTIPLIER below 1.0 should be rejected if settable."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        # Only test if the attribute is settable
        if hasattr(type(CROWD_BEHAVIOR_CONFIG), 'AVOIDANCE_PRIORITY_MULTIPLIER') and \
           isinstance(getattr(type(CROWD_BEHAVIOR_CONFIG), 'AVOIDANCE_PRIORITY_MULTIPLIER', None), property):
            with pytest.raises(ValueError):
                CROWD_BEHAVIOR_CONFIG.AVOIDANCE_PRIORITY_MULTIPLIER = 0.5


class TestConfigChangesEffective:
    """
    Contract: Config changes take effect without restart.
    Tests that runtime config modifications are honored.
    """

    def test_avoidance_radius_modifiable(self):
        """avoidance_radius should be modifiable at runtime."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        original = CROWD_BEHAVIOR_CONFIG.avoidance_radius
        new_value = original + 1.0

        try:
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = new_value
            assert CROWD_BEHAVIOR_CONFIG.avoidance_radius == new_value, \
                "Config change should take effect immediately"
        finally:
            # Restore original value
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = original

    def test_config_changes_persist_in_same_session(self):
        """Config changes should persist within the same session."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        original = CROWD_BEHAVIOR_CONFIG.avoidance_radius
        test_value = 99.5

        try:
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = test_value

            # Re-import to verify persistence
            from engine.animation.config import CROWD_BEHAVIOR_CONFIG as config2
            assert config2.avoidance_radius == test_value, \
                "Config change should persist across module access"
        finally:
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = original


class TestParametersConfigurable:
    """
    Contract: All behavior parameters are configurable.
    Tests that expected behavior parameters exist and are accessible.
    """

    def test_has_avoidance_radius(self):
        """Config should have avoidance_radius parameter."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        assert hasattr(CROWD_BEHAVIOR_CONFIG, 'avoidance_radius'), \
            "Config must have avoidance_radius parameter"

    def test_has_min_distance_epsilon(self):
        """Config should have MIN_DISTANCE_EPSILON parameter."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        assert hasattr(CROWD_BEHAVIOR_CONFIG, 'MIN_DISTANCE_EPSILON'), \
            "Config must have MIN_DISTANCE_EPSILON parameter"

    def test_has_avoidance_priority_multiplier(self):
        """Config should have AVOIDANCE_PRIORITY_MULTIPLIER parameter."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        assert hasattr(CROWD_BEHAVIOR_CONFIG, 'AVOIDANCE_PRIORITY_MULTIPLIER'), \
            "Config must have AVOIDANCE_PRIORITY_MULTIPLIER parameter"

    def test_config_attributes_readable(self):
        """All config attributes should be readable without error."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        # Should not raise any exception
        _ = CROWD_BEHAVIOR_CONFIG.avoidance_radius
        _ = CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON
        _ = CROWD_BEHAVIOR_CONFIG.AVOIDANCE_PRIORITY_MULTIPLIER


class TestConfigEdgeCases:
    """
    Edge case tests for config boundary conditions.
    """

    def test_very_small_avoidance_radius_valid(self):
        """Very small but positive avoidance_radius should be valid."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        original = CROWD_BEHAVIOR_CONFIG.avoidance_radius
        try:
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = 0.001
            assert CROWD_BEHAVIOR_CONFIG.avoidance_radius == pytest.approx(0.001)
        finally:
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = original

    def test_large_avoidance_radius_valid(self):
        """Large avoidance_radius values should be valid."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        original = CROWD_BEHAVIOR_CONFIG.avoidance_radius
        try:
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = 1000.0
            assert CROWD_BEHAVIOR_CONFIG.avoidance_radius == pytest.approx(1000.0)
        finally:
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = original

    def test_float_precision_maintained(self):
        """Float values should maintain reasonable precision."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        original = CROWD_BEHAVIOR_CONFIG.avoidance_radius
        test_value = 3.14159265
        try:
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = test_value
            assert CROWD_BEHAVIOR_CONFIG.avoidance_radius == pytest.approx(test_value, rel=1e-6)
        finally:
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = original


class TestConfigTypeValidation:
    """
    Tests for type validation on config parameters.
    """

    def test_string_avoidance_radius_rejected(self):
        """String values for avoidance_radius should be rejected."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        with pytest.raises((ValueError, TypeError)):
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = "invalid"

    def test_none_avoidance_radius_rejected(self):
        """None values for avoidance_radius should be rejected."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        with pytest.raises((ValueError, TypeError)):
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = None

    def test_int_avoidance_radius_accepted(self):
        """Integer values for avoidance_radius should be accepted (coerced to float)."""
        from engine.animation.config import CROWD_BEHAVIOR_CONFIG

        original = CROWD_BEHAVIOR_CONFIG.avoidance_radius
        try:
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = 5
            assert CROWD_BEHAVIOR_CONFIG.avoidance_radius == pytest.approx(5.0)
        finally:
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = original

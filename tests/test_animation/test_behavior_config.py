"""Tests for behavior configuration validation - Task T2.5.

Acceptance Criteria:
1. All behavior parameters are configurable
2. Config changes take effect without restart
3. Invalid config values are rejected
4. Default values are sensible
"""

from __future__ import annotations

import pytest

from engine.animation.config import (
    CROWD_BEHAVIOR_CONFIG,
    CROWD_LOD_CONFIG,
    ANIMATION_SYSTEM_CONFIG,
    IK_CONFIG,
    PROCEDURAL_CONFIG,
    SKINNING_CONFIG,
    FACIAL_CONFIG,
    CROWD_SYSTEM_CONFIG,
    ConfigValidationError,
    CrowdBehaviorConfig,
    MutableConfig,
    ValidatedDescriptor,
    positive,
    non_negative,
    at_least,
    in_range,
    reset_all_configs,
)


class TestCrowdBehaviorConfigValidation:
    """Tests for CrowdBehaviorConfig validation (AC3: Invalid config values rejected)."""

    def setup_method(self) -> None:
        """Reset config to defaults before each test."""
        CROWD_BEHAVIOR_CONFIG.reset()

    def teardown_method(self) -> None:
        """Reset config after each test."""
        CROWD_BEHAVIOR_CONFIG.reset()

    def test_avoidance_radius_rejects_negative(self) -> None:
        """Invalid config rejected: negative avoidance_radius raises ValueError."""
        with pytest.raises(ConfigValidationError):
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = -1

    def test_avoidance_radius_rejects_zero(self) -> None:
        """Invalid config rejected: zero avoidance_radius raises ValueError."""
        with pytest.raises(ConfigValidationError):
            CROWD_BEHAVIOR_CONFIG.avoidance_radius = 0

    def test_default_avoidance_radius_rejects_negative(self) -> None:
        """Invalid config rejected: negative DEFAULT_AVOIDANCE_RADIUS raises ValueError."""
        with pytest.raises(ConfigValidationError):
            CROWD_BEHAVIOR_CONFIG.DEFAULT_AVOIDANCE_RADIUS = -1

    def test_agent_speed_rejects_negative(self) -> None:
        """Invalid config rejected: negative speed raises ValueError."""
        with pytest.raises(ConfigValidationError):
            CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED = -0.5

    def test_agent_speed_rejects_zero(self) -> None:
        """Invalid config rejected: zero speed raises ValueError."""
        with pytest.raises(ConfigValidationError):
            CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED = 0

    def test_agent_radius_rejects_negative(self) -> None:
        """Invalid config rejected: negative agent radius raises ValueError."""
        with pytest.raises(ConfigValidationError):
            CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_RADIUS = -0.1

    def test_min_distance_epsilon_rejects_negative(self) -> None:
        """Invalid config rejected: negative epsilon raises ValueError."""
        with pytest.raises(ConfigValidationError):
            CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON = -0.001

    def test_min_distance_epsilon_rejects_zero(self) -> None:
        """Invalid config rejected: zero epsilon raises ValueError."""
        with pytest.raises(ConfigValidationError):
            CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON = 0

    def test_avoidance_priority_multiplier_rejects_below_one(self) -> None:
        """Invalid config rejected: multiplier below 1.0 raises ValueError."""
        with pytest.raises(ConfigValidationError):
            CROWD_BEHAVIOR_CONFIG.AVOIDANCE_PRIORITY_MULTIPLIER = 0.5

    def test_type_validation_rejects_wrong_type(self) -> None:
        """Invalid config rejected: wrong type raises ValueError."""
        with pytest.raises(ConfigValidationError):
            CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED = "fast"

        with pytest.raises(ConfigValidationError):
            CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED = [1.4]


class TestCrowdBehaviorConfigDefaults:
    """Tests for sensible default values (AC4: Default values are sensible)."""

    def test_min_distance_epsilon_positive(self) -> None:
        """Default values sensible: MIN_DISTANCE_EPSILON > 0."""
        assert CROWD_BEHAVIOR_CONFIG.MIN_DISTANCE_EPSILON > 0

    def test_avoidance_priority_multiplier_at_least_one(self) -> None:
        """Default values sensible: AVOIDANCE_PRIORITY_MULTIPLIER >= 1.0."""
        assert CROWD_BEHAVIOR_CONFIG.AVOIDANCE_PRIORITY_MULTIPLIER >= 1.0

    def test_agent_speed_reasonable(self) -> None:
        """Default values sensible: agent speed is walking speed."""
        # Average human walking speed is 1.2-1.5 m/s
        assert 1.0 <= CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED <= 2.0

    def test_agent_radius_reasonable(self) -> None:
        """Default values sensible: agent radius fits human."""
        # Human shoulder width ~0.4-0.5m
        assert 0.2 <= CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_RADIUS <= 0.6

    def test_avoidance_radius_greater_than_agent_radius(self) -> None:
        """Default values sensible: avoidance radius > agent radius."""
        assert (
            CROWD_BEHAVIOR_CONFIG.DEFAULT_AVOIDANCE_RADIUS
            > CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_RADIUS
        )

    def test_flee_speed_faster_than_normal(self) -> None:
        """Default values sensible: fleeing is faster than walking."""
        assert CROWD_BEHAVIOR_CONFIG.FLEE_SPEED_MULTIPLIER > 1.0

    def test_idle_variation_range_valid(self) -> None:
        """Default values sensible: idle min < idle max."""
        assert (
            CROWD_BEHAVIOR_CONFIG.IDLE_VARIATION_MIN
            < CROWD_BEHAVIOR_CONFIG.IDLE_VARIATION_MAX
        )


class TestConfigRuntimeModification:
    """Tests for runtime configuration changes (AC2: Changes take effect without restart)."""

    def setup_method(self) -> None:
        """Reset config to defaults before each test."""
        CROWD_BEHAVIOR_CONFIG.reset()

    def teardown_method(self) -> None:
        """Reset config after each test."""
        CROWD_BEHAVIOR_CONFIG.reset()

    def test_config_change_takes_effect_immediately(self) -> None:
        """Config changes take effect without restart."""
        original_speed = CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED
        new_speed = 2.5

        CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED = new_speed

        assert CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED == new_speed
        assert CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED != original_speed

    def test_multiple_config_changes(self) -> None:
        """Multiple config changes all take effect."""
        CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED = 3.0
        CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_RADIUS = 0.5
        CROWD_BEHAVIOR_CONFIG.FLEE_SPEED_MULTIPLIER = 2.0

        assert CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED == 3.0
        assert CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_RADIUS == 0.5
        assert CROWD_BEHAVIOR_CONFIG.FLEE_SPEED_MULTIPLIER == 2.0

    def test_int_to_float_coercion(self) -> None:
        """Integer values are coerced to float for float fields."""
        CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED = 3

        assert CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED == 3.0
        assert isinstance(CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED, float)

    def test_reset_restores_defaults(self) -> None:
        """Reset method restores all defaults."""
        original_speed = CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED
        CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED = 10.0

        CROWD_BEHAVIOR_CONFIG.reset()

        assert CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED == original_speed


class TestConfigChangeCallbacks:
    """Tests for configuration change notification callbacks."""

    def setup_method(self) -> None:
        """Reset config to defaults before each test."""
        CROWD_BEHAVIOR_CONFIG.reset()
        self.changes: list[tuple[str, float, float]] = []

    def teardown_method(self) -> None:
        """Reset config after each test."""
        CROWD_BEHAVIOR_CONFIG.reset()

    def test_callback_receives_change_notification(self) -> None:
        """Change callbacks are notified of config changes."""

        def on_change(name: str, old_val: float, new_val: float) -> None:
            self.changes.append((name, old_val, new_val))

        CROWD_BEHAVIOR_CONFIG.on_change(on_change)
        CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED = 5.0

        assert len(self.changes) == 1
        assert self.changes[0] == ("DEFAULT_AGENT_SPEED", 1.4, 5.0)

        CROWD_BEHAVIOR_CONFIG.remove_change_callback(on_change)

    def test_callback_removal(self) -> None:
        """Removed callbacks are not notified."""

        def on_change(name: str, old_val: float, new_val: float) -> None:
            self.changes.append((name, old_val, new_val))

        CROWD_BEHAVIOR_CONFIG.on_change(on_change)
        CROWD_BEHAVIOR_CONFIG.remove_change_callback(on_change)
        CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED = 5.0

        assert len(self.changes) == 0


class TestConfigSerialization:
    """Tests for configuration serialization (to_dict/from_dict)."""

    def setup_method(self) -> None:
        """Reset config to defaults before each test."""
        CROWD_BEHAVIOR_CONFIG.reset()

    def teardown_method(self) -> None:
        """Reset config after each test."""
        CROWD_BEHAVIOR_CONFIG.reset()

    def test_to_dict_contains_all_fields(self) -> None:
        """to_dict exports all configuration fields."""
        config_dict = CROWD_BEHAVIOR_CONFIG.to_dict()

        assert "DEFAULT_AGENT_SPEED" in config_dict
        assert "DEFAULT_AVOIDANCE_RADIUS" in config_dict
        assert "MIN_DISTANCE_EPSILON" in config_dict
        assert "AVOIDANCE_PRIORITY_MULTIPLIER" in config_dict

    def test_from_dict_applies_values(self) -> None:
        """from_dict imports configuration values."""
        CROWD_BEHAVIOR_CONFIG.from_dict({"DEFAULT_AGENT_SPEED": 2.0})

        assert CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED == 2.0

    def test_from_dict_validates_values(self) -> None:
        """from_dict validates imported values."""
        with pytest.raises(ConfigValidationError):
            CROWD_BEHAVIOR_CONFIG.from_dict({"DEFAULT_AGENT_SPEED": -1.0})

    def test_round_trip_serialization(self) -> None:
        """to_dict/from_dict round-trip preserves values."""
        CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED = 3.5
        CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_RADIUS = 0.6

        exported = CROWD_BEHAVIOR_CONFIG.to_dict()
        CROWD_BEHAVIOR_CONFIG.reset()
        CROWD_BEHAVIOR_CONFIG.from_dict(exported)

        assert CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED == 3.5
        assert CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_RADIUS == 0.6


class TestAllConfigsAreConfigurable:
    """Tests that all behavior parameters are configurable (AC1)."""

    def setup_method(self) -> None:
        """Reset all configs to defaults before each test."""
        reset_all_configs()

    def teardown_method(self) -> None:
        """Reset all configs after each test."""
        reset_all_configs()

    def test_crowd_lod_config_is_mutable(self) -> None:
        """CrowdLODConfig parameters are configurable."""
        CROWD_LOD_CONFIG.DEFAULT_CULL_DISTANCE = 500.0
        assert CROWD_LOD_CONFIG.DEFAULT_CULL_DISTANCE == 500.0

    def test_animation_system_config_is_mutable(self) -> None:
        """AnimationSystemConfig parameters are configurable."""
        ANIMATION_SYSTEM_CONFIG.PRIORITY_CROWD = 600
        assert ANIMATION_SYSTEM_CONFIG.PRIORITY_CROWD == 600

    def test_ik_config_is_mutable(self) -> None:
        """IKConfig parameters are configurable."""
        IK_CONFIG.DEFAULT_MAX_ITERATIONS = 20
        assert IK_CONFIG.DEFAULT_MAX_ITERATIONS == 20

    def test_procedural_config_is_mutable(self) -> None:
        """ProceduralConfig parameters are configurable."""
        PROCEDURAL_CONFIG.DEFAULT_SPRING_STIFFNESS = 15.0
        assert PROCEDURAL_CONFIG.DEFAULT_SPRING_STIFFNESS == 15.0

    def test_skinning_config_is_mutable(self) -> None:
        """SkinningConfig parameters are configurable."""
        SKINNING_CONFIG.DEFAULT_MAX_INFLUENCES = 8
        assert SKINNING_CONFIG.DEFAULT_MAX_INFLUENCES == 8

    def test_facial_config_is_mutable(self) -> None:
        """FacialConfig parameters are configurable."""
        FACIAL_CONFIG.DEFAULT_BLINK_DURATION = 0.2
        assert FACIAL_CONFIG.DEFAULT_BLINK_DURATION == 0.2

    def test_crowd_system_config_is_mutable(self) -> None:
        """CrowdSystemConfig parameters are configurable."""
        CROWD_SYSTEM_CONFIG.DEFAULT_UPDATE_RATE = 60.0
        assert CROWD_SYSTEM_CONFIG.DEFAULT_UPDATE_RATE == 60.0


class TestValidatorHelpers:
    """Tests for validation helper functions."""

    def test_positive_rejects_zero(self) -> None:
        """positive() rejects zero."""
        assert positive(1) is True
        assert positive(0.001) is True
        assert positive(0) is False
        assert positive(-1) is False

    def test_non_negative_accepts_zero(self) -> None:
        """non_negative() accepts zero."""
        assert non_negative(0) is True
        assert non_negative(1) is True
        assert non_negative(-1) is False

    def test_at_least_creates_validator(self) -> None:
        """at_least() creates correct validator."""
        at_least_5 = at_least(5)
        assert at_least_5(5) is True
        assert at_least_5(10) is True
        assert at_least_5(4) is False

    def test_in_range_creates_validator(self) -> None:
        """in_range() creates correct validator."""
        in_0_to_1 = in_range(0.0, 1.0)
        assert in_0_to_1(0.0) is True
        assert in_0_to_1(0.5) is True
        assert in_0_to_1(1.0) is True
        assert in_0_to_1(-0.1) is False
        assert in_0_to_1(1.1) is False


class TestIKConfigValidation:
    """Tests for IK configuration validation."""

    def setup_method(self) -> None:
        """Reset config to defaults before each test."""
        IK_CONFIG.reset()

    def teardown_method(self) -> None:
        """Reset config after each test."""
        IK_CONFIG.reset()

    def test_max_iterations_rejects_zero(self) -> None:
        """MAX_ITERATIONS must be at least 1."""
        with pytest.raises(ConfigValidationError):
            IK_CONFIG.DEFAULT_MAX_ITERATIONS = 0

    def test_min_bone_length_rejects_zero(self) -> None:
        """MIN_BONE_LENGTH must be positive."""
        with pytest.raises(ConfigValidationError):
            IK_CONFIG.MIN_BONE_LENGTH = 0


class TestSkinningConfigValidation:
    """Tests for skinning configuration validation."""

    def setup_method(self) -> None:
        """Reset config to defaults before each test."""
        SKINNING_CONFIG.reset()

    def teardown_method(self) -> None:
        """Reset config after each test."""
        SKINNING_CONFIG.reset()

    def test_dq_blend_threshold_in_range(self) -> None:
        """DQ_BLEND_THRESHOLD must be between 0 and 1."""
        with pytest.raises(ConfigValidationError):
            SKINNING_CONFIG.DQ_BLEND_THRESHOLD = 1.5

        with pytest.raises(ConfigValidationError):
            SKINNING_CONFIG.DQ_BLEND_THRESHOLD = -0.1

        # Valid values should work
        SKINNING_CONFIG.DQ_BLEND_THRESHOLD = 0.0
        assert SKINNING_CONFIG.DQ_BLEND_THRESHOLD == 0.0

        SKINNING_CONFIG.DQ_BLEND_THRESHOLD = 1.0
        assert SKINNING_CONFIG.DQ_BLEND_THRESHOLD == 1.0


class TestResetAllConfigs:
    """Tests for reset_all_configs utility function."""

    def test_reset_all_configs_resets_all(self) -> None:
        """reset_all_configs() resets all mutable configs."""
        # Modify multiple configs
        CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED = 10.0
        IK_CONFIG.DEFAULT_MAX_ITERATIONS = 50
        FACIAL_CONFIG.DEFAULT_BLINK_DURATION = 1.0

        # Reset all
        reset_all_configs()

        # Verify defaults restored
        assert CROWD_BEHAVIOR_CONFIG.DEFAULT_AGENT_SPEED == 1.4
        assert IK_CONFIG.DEFAULT_MAX_ITERATIONS == 10
        assert FACIAL_CONFIG.DEFAULT_BLINK_DURATION == 0.15

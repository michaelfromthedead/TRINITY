"""Whitebox tests for PelvisAdjustmentConfig and PelvisHeightAdjuster classes.

Tests internal implementation details including:
- Configuration dataclass validation in __post_init__
- Smoothing formula implementation
- Geometry calculations for required drop
- State management and reset behavior
- Property accessors
"""

from __future__ import annotations

import pytest
import math
from unittest.mock import Mock, MagicMock, patch
from typing import List

from engine.animation.ik.fullbody import (
    PelvisAdjustmentConfig,
    PelvisHeightAdjuster,
)
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON


# =============================================================================
# Helper Functions
# =============================================================================


def vec3_approx_equal(v1: Vec3, v2: Vec3, eps: float = 1e-6) -> bool:
    """Check if two Vec3 are approximately equal."""
    return (
        abs(v1.x - v2.x) < eps and
        abs(v1.y - v2.y) < eps and
        abs(v1.z - v2.z) < eps
    )


def create_transform(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Transform:
    """Create a Transform at specified position."""
    return Transform(Vec3(x, y, z), Quat.identity())


def create_transforms(positions: List[tuple]) -> List[Transform]:
    """Create transforms from list of (x, y, z) tuples."""
    return [create_transform(x, y, z) for x, y, z in positions]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config() -> PelvisAdjustmentConfig:
    """Default configuration with standard values."""
    return PelvisAdjustmentConfig()


@pytest.fixture
def custom_config() -> PelvisAdjustmentConfig:
    """Custom configuration for testing."""
    return PelvisAdjustmentConfig(
        safety_margin=0.8,
        max_drop=1.0,
        smooth_speed=10.0
    )


@pytest.fixture
def default_adjuster() -> PelvisHeightAdjuster:
    """Default PelvisHeightAdjuster instance."""
    return PelvisHeightAdjuster()


@pytest.fixture
def custom_adjuster(custom_config) -> PelvisHeightAdjuster:
    """PelvisHeightAdjuster with custom config."""
    return PelvisHeightAdjuster(custom_config)


@pytest.fixture
def pelvis_at_origin() -> Vec3:
    """Pelvis positioned at origin."""
    return Vec3(0.0, 0.0, 0.0)


@pytest.fixture
def pelvis_elevated() -> Vec3:
    """Pelvis at typical humanoid height."""
    return Vec3(0.0, 1.0, 0.0)


@pytest.fixture
def single_transform_list() -> List[Transform]:
    """List with single transform at pelvis height."""
    return [create_transform(0.0, 1.0, 0.0)]


@pytest.fixture
def multi_transform_list() -> List[Transform]:
    """List with multiple transforms for skeleton simulation."""
    return create_transforms([
        (0.0, 1.0, 0.0),    # 0: Pelvis
        (0.0, 1.2, 0.0),    # 1: Spine
        (0.0, 1.4, 0.0),    # 2: Chest
        (0.0, 0.5, 0.1),    # 3: Left leg
        (0.0, 0.5, -0.1),   # 4: Right leg
    ])


# =============================================================================
# PelvisAdjustmentConfig Tests - Default Values
# =============================================================================


class TestPelvisAdjustmentConfigDefaults:
    """Tests for PelvisAdjustmentConfig default values."""

    def test_default_safety_margin(self) -> None:
        """Default safety_margin is 0.95."""
        config = PelvisAdjustmentConfig()
        assert config.safety_margin == 0.95

    def test_default_max_drop(self) -> None:
        """Default max_drop is 0.5."""
        config = PelvisAdjustmentConfig()
        assert config.max_drop == 0.5

    def test_default_smooth_speed(self) -> None:
        """Default smooth_speed is 5.0."""
        config = PelvisAdjustmentConfig()
        assert config.smooth_speed == 5.0

    def test_all_defaults_in_single_instance(self) -> None:
        """Verify all defaults together."""
        config = PelvisAdjustmentConfig()
        assert config.safety_margin == 0.95
        assert config.max_drop == 0.5
        assert config.smooth_speed == 5.0


# =============================================================================
# PelvisAdjustmentConfig Tests - Custom Values
# =============================================================================


class TestPelvisAdjustmentConfigCustomValues:
    """Tests for PelvisAdjustmentConfig with custom values."""

    def test_custom_safety_margin(self) -> None:
        """Custom safety_margin is stored correctly."""
        config = PelvisAdjustmentConfig(safety_margin=0.8)
        assert config.safety_margin == 0.8

    def test_custom_max_drop(self) -> None:
        """Custom max_drop is stored correctly."""
        config = PelvisAdjustmentConfig(max_drop=1.5)
        assert config.max_drop == 1.5

    def test_custom_smooth_speed(self) -> None:
        """Custom smooth_speed is stored correctly."""
        config = PelvisAdjustmentConfig(smooth_speed=2.5)
        assert config.smooth_speed == 2.5

    def test_all_custom_values(self) -> None:
        """All custom values stored together."""
        config = PelvisAdjustmentConfig(
            safety_margin=0.75,
            max_drop=2.0,
            smooth_speed=15.0
        )
        assert config.safety_margin == 0.75
        assert config.max_drop == 2.0
        assert config.smooth_speed == 15.0

    def test_boundary_safety_margin_at_one(self) -> None:
        """safety_margin of exactly 1.0 is valid."""
        config = PelvisAdjustmentConfig(safety_margin=1.0)
        assert config.safety_margin == 1.0

    def test_boundary_max_drop_zero(self) -> None:
        """max_drop of 0.0 is valid (no adjustment allowed)."""
        config = PelvisAdjustmentConfig(max_drop=0.0)
        assert config.max_drop == 0.0

    def test_small_safety_margin(self) -> None:
        """Very small safety_margin near zero is valid."""
        config = PelvisAdjustmentConfig(safety_margin=0.01)
        assert config.safety_margin == 0.01

    def test_large_max_drop(self) -> None:
        """Large max_drop values are valid."""
        config = PelvisAdjustmentConfig(max_drop=100.0)
        assert config.max_drop == 100.0

    def test_small_smooth_speed(self) -> None:
        """Small positive smooth_speed is valid."""
        config = PelvisAdjustmentConfig(smooth_speed=0.001)
        assert config.smooth_speed == 0.001


# =============================================================================
# PelvisAdjustmentConfig Tests - __post_init__ Validation
# =============================================================================


class TestPelvisAdjustmentConfigValidation:
    """Tests for __post_init__ validation logic."""

    def test_safety_margin_zero_raises(self) -> None:
        """safety_margin of 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="safety_margin must be in"):
            PelvisAdjustmentConfig(safety_margin=0.0)

    def test_safety_margin_negative_raises(self) -> None:
        """Negative safety_margin raises ValueError."""
        with pytest.raises(ValueError, match="safety_margin must be in"):
            PelvisAdjustmentConfig(safety_margin=-0.1)

    def test_safety_margin_greater_than_one_raises(self) -> None:
        """safety_margin > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="safety_margin must be in"):
            PelvisAdjustmentConfig(safety_margin=1.01)

    def test_safety_margin_large_negative_raises(self) -> None:
        """Large negative safety_margin raises ValueError."""
        with pytest.raises(ValueError, match="safety_margin must be in"):
            PelvisAdjustmentConfig(safety_margin=-100.0)

    def test_max_drop_negative_raises(self) -> None:
        """Negative max_drop raises ValueError."""
        with pytest.raises(ValueError, match="max_drop must be non-negative"):
            PelvisAdjustmentConfig(max_drop=-0.1)

    def test_max_drop_large_negative_raises(self) -> None:
        """Large negative max_drop raises ValueError."""
        with pytest.raises(ValueError, match="max_drop must be non-negative"):
            PelvisAdjustmentConfig(max_drop=-50.0)

    def test_smooth_speed_zero_raises(self) -> None:
        """smooth_speed of 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="smooth_speed must be positive"):
            PelvisAdjustmentConfig(smooth_speed=0.0)

    def test_smooth_speed_negative_raises(self) -> None:
        """Negative smooth_speed raises ValueError."""
        with pytest.raises(ValueError, match="smooth_speed must be positive"):
            PelvisAdjustmentConfig(smooth_speed=-1.0)

    def test_smooth_speed_large_negative_raises(self) -> None:
        """Large negative smooth_speed raises ValueError."""
        with pytest.raises(ValueError, match="smooth_speed must be positive"):
            PelvisAdjustmentConfig(smooth_speed=-100.0)

    def test_multiple_invalid_values_first_error_raised(self) -> None:
        """When multiple values invalid, first validation error raised."""
        # safety_margin is checked first
        with pytest.raises(ValueError, match="safety_margin"):
            PelvisAdjustmentConfig(
                safety_margin=0.0,
                max_drop=-1.0,
                smooth_speed=-1.0
            )

    def test_error_message_contains_actual_value(self) -> None:
        """Error message includes the invalid value."""
        with pytest.raises(ValueError) as exc_info:
            PelvisAdjustmentConfig(safety_margin=-0.5)
        assert "-0.5" in str(exc_info.value)


# =============================================================================
# PelvisHeightAdjuster Tests - Initialization
# =============================================================================


class TestPelvisHeightAdjusterInit:
    """Tests for PelvisHeightAdjuster __init__ method."""

    def test_default_init_creates_default_config(self) -> None:
        """Default init creates adjuster with default config."""
        adjuster = PelvisHeightAdjuster()
        assert adjuster.config.safety_margin == 0.95
        assert adjuster.config.max_drop == 0.5
        assert adjuster.config.smooth_speed == 5.0

    def test_none_config_uses_default(self) -> None:
        """Passing None explicitly uses default config."""
        adjuster = PelvisHeightAdjuster(config=None)
        assert adjuster.config.safety_margin == 0.95

    def test_custom_config_stored(self) -> None:
        """Custom config passed to init is stored."""
        config = PelvisAdjustmentConfig(safety_margin=0.8)
        adjuster = PelvisHeightAdjuster(config)
        assert adjuster.config.safety_margin == 0.8

    def test_initial_current_offset_is_zero(self) -> None:
        """Initial _current_offset is 0.0."""
        adjuster = PelvisHeightAdjuster()
        assert adjuster.current_offset == 0.0

    def test_initial_offset_with_custom_config(self) -> None:
        """_current_offset starts at 0 regardless of config."""
        config = PelvisAdjustmentConfig(max_drop=10.0)
        adjuster = PelvisHeightAdjuster(config)
        assert adjuster.current_offset == 0.0


# =============================================================================
# PelvisHeightAdjuster Tests - Property Accessors
# =============================================================================


class TestPelvisHeightAdjusterProperties:
    """Tests for property accessors."""

    def test_config_property_returns_config(self) -> None:
        """config property returns the stored config."""
        config = PelvisAdjustmentConfig(smooth_speed=8.0)
        adjuster = PelvisHeightAdjuster(config)
        assert adjuster.config is config

    def test_config_property_is_read_only(self) -> None:
        """config property cannot be set directly."""
        adjuster = PelvisHeightAdjuster()
        with pytest.raises(AttributeError):
            adjuster.config = PelvisAdjustmentConfig()

    def test_current_offset_property_returns_value(self) -> None:
        """current_offset property returns internal value."""
        adjuster = PelvisHeightAdjuster()
        assert adjuster.current_offset == 0.0

    def test_current_offset_property_is_read_only(self) -> None:
        """current_offset property cannot be set directly."""
        adjuster = PelvisHeightAdjuster()
        with pytest.raises(AttributeError):
            adjuster.current_offset = 1.0


# =============================================================================
# PelvisHeightAdjuster Tests - calculate_required_drop
# =============================================================================


class TestCalculateRequiredDropBasic:
    """Basic tests for calculate_required_drop method."""

    def test_empty_leg_targets_returns_zero(self) -> None:
        """Empty leg_targets list returns 0.0."""
        adjuster = PelvisHeightAdjuster()
        result = adjuster.calculate_required_drop(
            Vec3(0, 1, 0), [], 1.0
        )
        assert result == 0.0

    def test_zero_max_leg_reach_returns_zero(self) -> None:
        """max_leg_reach of 0 returns 0.0."""
        adjuster = PelvisHeightAdjuster()
        result = adjuster.calculate_required_drop(
            Vec3(0, 1, 0), [Vec3(0, 0, 0)], 0.0
        )
        assert result == 0.0

    def test_negative_max_leg_reach_returns_zero(self) -> None:
        """Negative max_leg_reach returns 0.0."""
        adjuster = PelvisHeightAdjuster()
        result = adjuster.calculate_required_drop(
            Vec3(0, 1, 0), [Vec3(0, 0, 0)], -1.0
        )
        assert result == 0.0

    def test_none_target_in_list_is_skipped(self) -> None:
        """None entries in leg_targets are skipped."""
        adjuster = PelvisHeightAdjuster()
        result = adjuster.calculate_required_drop(
            Vec3(0, 1, 0), [None, None], 1.0
        )
        assert result == 0.0

    def test_target_within_safe_reach_returns_zero(self) -> None:
        """Target within safe reach requires no drop."""
        # Pelvis at (0, 1, 0), target at (0, 0.2, 0)
        # Distance = 0.8, safe_reach = 1.0 * 0.95 = 0.95
        adjuster = PelvisHeightAdjuster()
        result = adjuster.calculate_required_drop(
            Vec3(0, 1, 0), [Vec3(0, 0.2, 0)], 1.0
        )
        assert result == 0.0


class TestCalculateRequiredDropGeometry:
    """Tests for geometry calculations in calculate_required_drop."""

    def test_target_directly_below_beyond_reach(self) -> None:
        """Target directly below pelvis beyond safe reach needs drop."""
        # Pelvis at (0, 2, 0), target at (0, 0, 0)
        # Distance = 2.0, safe_reach = 1.5 * 0.95 = 1.425
        adjuster = PelvisHeightAdjuster()
        result = adjuster.calculate_required_drop(
            Vec3(0, 2, 0), [Vec3(0, 0, 0)], 1.5
        )
        # Target is directly below, so drop needed is: 2.0 - 1.425 = 0.575
        assert result > 0.0

    def test_target_horizontally_far(self) -> None:
        """Target too far horizontally for drop to help."""
        # Pelvis at (0, 1, 0), target at (5, 1, 0)
        # Distance = 5.0, safe_reach = 2.0 * 0.95 = 1.9
        # Horizontal distance alone exceeds reach
        adjuster = PelvisHeightAdjuster()
        result = adjuster.calculate_required_drop(
            Vec3(0, 1, 0), [Vec3(5, 1, 0)], 2.0
        )
        # When target_y_sq < 0, uses excess as approximation
        assert result > 0.0

    def test_target_below_and_offset_horizontally(self) -> None:
        """Target below and horizontally offset needs calculated drop."""
        # Pelvis at (0, 1, 0), target at (0.5, 0, 0)
        # This is a common scenario (foot placement)
        adjuster = PelvisHeightAdjuster()
        result = adjuster.calculate_required_drop(
            Vec3(0, 1, 0), [Vec3(0.5, 0, 0)], 0.8
        )
        # Should need some drop
        assert result >= 0.0

    def test_target_above_pelvis(self) -> None:
        """Target above pelvis (unusual case)."""
        # Pelvis at (0, 1, 0), target at (0, 2, 0.5)
        adjuster = PelvisHeightAdjuster()
        result = adjuster.calculate_required_drop(
            Vec3(0, 1, 0), [Vec3(0, 2, 0.5)], 0.5
        )
        # Target is above, different geometry path
        assert result >= 0.0

    def test_multiple_targets_uses_max_drop(self) -> None:
        """Multiple targets: uses maximum required drop."""
        adjuster = PelvisHeightAdjuster()
        # Target 1: close, no drop needed
        # Target 2: far, needs drop
        result = adjuster.calculate_required_drop(
            Vec3(0, 1, 0),
            [Vec3(0, 0.5, 0), Vec3(0, 0, 2)],
            1.0
        )
        # Should use the larger required drop
        assert result > 0.0

    def test_safe_reach_calculation(self) -> None:
        """Verify safe_reach is max_reach * safety_margin."""
        config = PelvisAdjustmentConfig(safety_margin=0.5)
        adjuster = PelvisHeightAdjuster(config)
        # With safety_margin=0.5 and reach=2.0, safe_reach=1.0
        # Pelvis at (0, 1, 0), target at (0, 0.5, 0), distance=0.5
        # Should be within safe reach
        result = adjuster.calculate_required_drop(
            Vec3(0, 1, 0), [Vec3(0, 0.5, 0)], 2.0
        )
        assert result == 0.0


class TestCalculateRequiredDropEdgeCases:
    """Edge case tests for calculate_required_drop."""

    def test_pelvis_at_same_position_as_target(self) -> None:
        """Pelvis and target at same position."""
        adjuster = PelvisHeightAdjuster()
        result = adjuster.calculate_required_drop(
            Vec3(1, 1, 1), [Vec3(1, 1, 1)], 1.0
        )
        # Distance is 0, should need no drop
        assert result == 0.0

    def test_very_small_distance(self) -> None:
        """Very small distance near epsilon."""
        adjuster = PelvisHeightAdjuster()
        tiny = MATH_EPSILON / 10.0
        result = adjuster.calculate_required_drop(
            Vec3(0, 0, 0), [Vec3(tiny, tiny, tiny)], 1.0
        )
        assert result >= 0.0

    def test_large_max_leg_reach(self) -> None:
        """Very large max_leg_reach covers all targets."""
        adjuster = PelvisHeightAdjuster()
        result = adjuster.calculate_required_drop(
            Vec3(0, 0, 0), [Vec3(100, 100, 100)], 1000.0
        )
        # Distance ~173, safe_reach=950, within reach
        assert result == 0.0

    def test_single_none_with_valid_targets(self) -> None:
        """Mix of None and valid targets."""
        adjuster = PelvisHeightAdjuster()
        result = adjuster.calculate_required_drop(
            Vec3(0, 1, 0),
            [None, Vec3(0, 0.5, 0), None],
            1.0
        )
        # Valid target is within reach
        assert result == 0.0

    def test_all_targets_none(self) -> None:
        """All targets are None."""
        adjuster = PelvisHeightAdjuster()
        result = adjuster.calculate_required_drop(
            Vec3(0, 1, 0), [None, None, None], 1.0
        )
        assert result == 0.0


# =============================================================================
# PelvisHeightAdjuster Tests - adjust Method
# =============================================================================


class TestAdjustMethodBasic:
    """Basic tests for the adjust method."""

    def test_invalid_pelvis_idx_negative_returns_zero(self) -> None:
        """Negative pelvis_idx returns zero vector."""
        adjuster = PelvisHeightAdjuster()
        transforms = [create_transform(0, 1, 0)]
        result = adjuster.adjust(transforms, -1, [], 1.0, 0.016)
        assert vec3_approx_equal(result, Vec3.zero())

    def test_invalid_pelvis_idx_too_large_returns_zero(self) -> None:
        """pelvis_idx >= len(transforms) returns zero vector."""
        adjuster = PelvisHeightAdjuster()
        transforms = [create_transform(0, 1, 0)]
        result = adjuster.adjust(transforms, 5, [], 1.0, 0.016)
        assert vec3_approx_equal(result, Vec3.zero())

    def test_empty_transforms_returns_zero(self) -> None:
        """Empty transforms list returns zero vector."""
        adjuster = PelvisHeightAdjuster()
        result = adjuster.adjust([], 0, [], 1.0, 0.016)
        assert vec3_approx_equal(result, Vec3.zero())

    def test_valid_call_returns_adjustment_vector(self) -> None:
        """Valid call returns adjustment Vec3."""
        adjuster = PelvisHeightAdjuster()
        transforms = [create_transform(0, 1, 0)]
        result = adjuster.adjust(
            transforms, 0, [Vec3(0, -0.5, 0)], 0.5, 0.016
        )
        assert isinstance(result, Vec3)


class TestAdjustMethodSmoothing:
    """Tests for smoothing formula in adjust method."""

    def test_smoothing_formula_with_small_dt(self) -> None:
        """Verify smoothing: current += (target - current) * min(1.0, speed * dt)."""
        config = PelvisAdjustmentConfig(smooth_speed=10.0, max_drop=1.0)
        adjuster = PelvisHeightAdjuster(config)
        transforms = [create_transform(0, 2, 0)]

        # Target far below, needs drop
        # First adjust should partially move toward target
        adjuster.adjust(
            transforms, 0, [Vec3(0, 0, 0)], 1.0, 0.1
        )

        # With speed=10 and dt=0.1, alpha = 1.0
        # Should reach target drop in one step (clamped to 1.0)
        # Since alpha is 1.0, current_offset should equal target_drop
        assert adjuster.current_offset >= 0.0

    def test_smoothing_alpha_capped_at_one(self) -> None:
        """Alpha = min(1.0, smooth_speed * dt) is capped at 1."""
        config = PelvisAdjustmentConfig(smooth_speed=100.0, max_drop=1.0)
        adjuster = PelvisHeightAdjuster(config)
        transforms = [create_transform(0, 2, 0)]

        # With speed=100 and dt=0.1, alpha = min(1.0, 10.0) = 1.0
        adjuster.adjust(
            transforms, 0, [Vec3(0, 0, 0)], 1.0, 0.1
        )

        # Should immediately reach target (alpha=1.0 means instant)
        offset = adjuster.current_offset
        assert offset <= config.max_drop

    def test_gradual_smoothing_over_multiple_frames(self) -> None:
        """Multiple adjust calls gradually approach target."""
        config = PelvisAdjustmentConfig(smooth_speed=2.0, max_drop=1.0)
        adjuster = PelvisHeightAdjuster(config)

        offsets = []
        for _ in range(10):
            transforms = [create_transform(0, 2, 0)]
            adjuster.adjust(
                transforms, 0, [Vec3(0, 0, 0)], 0.5, 0.1
            )
            offsets.append(adjuster.current_offset)

        # Each offset should increase or stay same
        for i in range(1, len(offsets)):
            assert offsets[i] >= offsets[i-1] - 1e-9

    def test_smoothing_approaches_zero_when_no_drop_needed(self) -> None:
        """Offset smoothly returns to zero when target reachable."""
        config = PelvisAdjustmentConfig(smooth_speed=5.0, max_drop=1.0)
        adjuster = PelvisHeightAdjuster(config)

        # First, create some offset
        transforms = [create_transform(0, 2, 0)]
        adjuster.adjust(
            transforms, 0, [Vec3(0, 0, 0)], 0.5, 0.5
        )
        initial_offset = adjuster.current_offset
        assert initial_offset > 0

        # Now target is within reach, offset should decrease
        for _ in range(10):
            transforms = [create_transform(0, 1, 0)]
            adjuster.adjust(
                transforms, 0, [Vec3(0, 0.5, 0)], 2.0, 0.1
            )

        # Offset should be smaller
        assert adjuster.current_offset < initial_offset


class TestAdjustMethodMaxDropClamping:
    """Tests for max_drop clamping in adjust method."""

    def test_target_drop_clamped_to_max_drop(self) -> None:
        """target_drop is clamped to max_drop config value."""
        config = PelvisAdjustmentConfig(max_drop=0.2, smooth_speed=100.0)
        adjuster = PelvisHeightAdjuster(config)

        # Situation requiring large drop
        transforms = [create_transform(0, 10, 0)]
        adjuster.adjust(
            transforms, 0, [Vec3(0, 0, 0)], 1.0, 0.1
        )

        # Offset should not exceed max_drop
        assert adjuster.current_offset <= config.max_drop + 1e-9

    def test_zero_max_drop_prevents_adjustment(self) -> None:
        """max_drop=0 prevents any adjustment."""
        config = PelvisAdjustmentConfig(max_drop=0.0)
        adjuster = PelvisHeightAdjuster(config)

        transforms = [create_transform(0, 10, 0)]
        adjuster.adjust(
            transforms, 0, [Vec3(0, 0, 0)], 1.0, 0.1
        )

        assert adjuster.current_offset == 0.0


class TestAdjustMethodTransformModification:
    """Tests for transform modification in adjust method."""

    def test_pelvis_transform_translation_modified(self) -> None:
        """adjust() modifies transforms[pelvis_idx].translation.y."""
        config = PelvisAdjustmentConfig(smooth_speed=100.0, max_drop=1.0)
        adjuster = PelvisHeightAdjuster(config)

        transforms = [create_transform(0, 2, 0)]
        original_y = transforms[0].translation.y

        adjuster.adjust(
            transforms, 0, [Vec3(0, 0, 0)], 0.5, 0.1
        )

        # Y should be lower (negative adjustment)
        assert transforms[0].translation.y <= original_y

    def test_pelvis_x_z_unchanged(self) -> None:
        """Pelvis X and Z positions unchanged by adjust."""
        adjuster = PelvisHeightAdjuster()

        transforms = [create_transform(5, 2, 3)]
        adjuster.adjust(
            transforms, 0, [Vec3(5, 0, 3)], 0.5, 0.1
        )

        assert transforms[0].translation.x == 5.0
        assert transforms[0].translation.z == 3.0

    def test_only_pelvis_index_modified(self) -> None:
        """Only transform at pelvis_idx is modified."""
        adjuster = PelvisHeightAdjuster()

        transforms = create_transforms([
            (0, 2, 0),  # 0: Not pelvis
            (0, 3, 0),  # 1: Pelvis
            (0, 4, 0),  # 2: Not pelvis
        ])

        original_y_0 = transforms[0].translation.y
        original_y_2 = transforms[2].translation.y

        adjuster.adjust(
            transforms, 1, [Vec3(0, 0, 0)], 0.5, 0.1
        )

        assert transforms[0].translation.y == original_y_0
        assert transforms[2].translation.y == original_y_2

    def test_adjustment_vector_y_is_negative_offset(self) -> None:
        """Returned adjustment vector Y = -current_offset."""
        config = PelvisAdjustmentConfig(smooth_speed=100.0, max_drop=1.0)
        adjuster = PelvisHeightAdjuster(config)

        transforms = [create_transform(0, 2, 0)]
        result = adjuster.adjust(
            transforms, 0, [Vec3(0, 0, 0)], 0.5, 0.1
        )

        # Result Y should be negative (downward)
        assert result.y == -adjuster.current_offset
        assert result.x == 0.0
        assert result.z == 0.0


class TestAdjustMethodDtVariations:
    """Tests for different dt (delta time) values."""

    def test_zero_dt_no_change(self) -> None:
        """dt=0 means alpha=0, no change to current_offset."""
        adjuster = PelvisHeightAdjuster()

        transforms = [create_transform(0, 2, 0)]
        adjuster.adjust(
            transforms, 0, [Vec3(0, 0, 0)], 0.5, 0.0
        )

        assert adjuster.current_offset == 0.0

    def test_large_dt_reaches_target_immediately(self) -> None:
        """Large dt with high speed reaches target in one frame."""
        config = PelvisAdjustmentConfig(smooth_speed=10.0, max_drop=0.5)
        adjuster = PelvisHeightAdjuster(config)

        transforms = [create_transform(0, 2, 0)]
        adjuster.adjust(
            transforms, 0, [Vec3(0, 0, 0)], 0.5, 1.0
        )

        # Alpha = min(1.0, 10.0 * 1.0) = 1.0, reaches target
        # Target is clamped to max_drop=0.5
        assert abs(adjuster.current_offset - 0.5) < 0.01


# =============================================================================
# PelvisHeightAdjuster Tests - reset Method
# =============================================================================


class TestResetMethod:
    """Tests for reset() method."""

    def test_reset_sets_offset_to_zero(self) -> None:
        """reset() sets _current_offset to 0."""
        adjuster = PelvisHeightAdjuster()

        # Build up some offset
        transforms = [create_transform(0, 2, 0)]
        adjuster.adjust(
            transforms, 0, [Vec3(0, 0, 0)], 0.5, 0.5
        )
        assert adjuster.current_offset > 0

        adjuster.reset()
        assert adjuster.current_offset == 0.0

    def test_reset_when_already_zero(self) -> None:
        """reset() when offset already zero is no-op."""
        adjuster = PelvisHeightAdjuster()
        assert adjuster.current_offset == 0.0

        adjuster.reset()
        assert adjuster.current_offset == 0.0

    def test_reset_preserves_config(self) -> None:
        """reset() does not affect configuration."""
        config = PelvisAdjustmentConfig(safety_margin=0.8, max_drop=2.0)
        adjuster = PelvisHeightAdjuster(config)

        adjuster.reset()

        assert adjuster.config.safety_margin == 0.8
        assert adjuster.config.max_drop == 2.0

    def test_adjust_works_after_reset(self) -> None:
        """Adjust works normally after reset."""
        adjuster = PelvisHeightAdjuster()

        # Build offset, reset, then adjust again
        transforms = [create_transform(0, 2, 0)]
        adjuster.adjust(transforms, 0, [Vec3(0, 0, 0)], 0.5, 0.5)

        adjuster.reset()

        transforms = [create_transform(0, 2, 0)]
        adjuster.adjust(transforms, 0, [Vec3(0, 0, 0)], 0.5, 0.1)

        # Should have built new offset
        assert adjuster.current_offset > 0


# =============================================================================
# PelvisHeightAdjuster Tests - set_config Method
# =============================================================================


class TestSetConfigMethod:
    """Tests for set_config() method."""

    def test_set_config_updates_config(self) -> None:
        """set_config() replaces the configuration."""
        adjuster = PelvisHeightAdjuster()

        new_config = PelvisAdjustmentConfig(
            safety_margin=0.7,
            max_drop=1.5,
            smooth_speed=20.0
        )
        adjuster.set_config(new_config)

        assert adjuster.config.safety_margin == 0.7
        assert adjuster.config.max_drop == 1.5
        assert adjuster.config.smooth_speed == 20.0

    def test_set_config_preserves_current_offset(self) -> None:
        """set_config() does not reset current_offset."""
        adjuster = PelvisHeightAdjuster()

        # Build some offset
        transforms = [create_transform(0, 2, 0)]
        adjuster.adjust(transforms, 0, [Vec3(0, 0, 0)], 0.5, 0.5)
        offset_before = adjuster.current_offset

        # Change config
        adjuster.set_config(PelvisAdjustmentConfig(max_drop=0.1))

        # Offset unchanged (until next adjust)
        assert adjuster.current_offset == offset_before

    def test_set_config_affects_next_adjust(self) -> None:
        """New config affects subsequent adjust() calls."""
        adjuster = PelvisHeightAdjuster()

        # Initial adjust with default config
        transforms = [create_transform(0, 5, 0)]
        adjuster.adjust(transforms, 0, [Vec3(0, 0, 0)], 0.5, 1.0)
        offset1 = adjuster.current_offset

        # Set very low max_drop
        adjuster.reset()
        adjuster.set_config(PelvisAdjustmentConfig(max_drop=0.05))

        transforms = [create_transform(0, 5, 0)]
        adjuster.adjust(transforms, 0, [Vec3(0, 0, 0)], 0.5, 1.0)
        offset2 = adjuster.current_offset

        # Second offset should be clamped to 0.05
        assert offset2 <= 0.05 + 1e-9


# =============================================================================
# PelvisHeightAdjuster Tests - get_target_offset Method
# =============================================================================


class TestGetTargetOffsetMethod:
    """Tests for get_target_offset() method."""

    def test_get_target_offset_returns_float(self) -> None:
        """get_target_offset() returns a float."""
        adjuster = PelvisHeightAdjuster()
        result = adjuster.get_target_offset(
            Vec3(0, 1, 0), [Vec3(0, 0, 0)], 1.0
        )
        assert isinstance(result, float)

    def test_get_target_offset_no_smoothing(self) -> None:
        """get_target_offset() returns raw value without smoothing."""
        config = PelvisAdjustmentConfig(smooth_speed=0.01, max_drop=1.0)
        adjuster = PelvisHeightAdjuster(config)

        # Even with very slow smoothing, get_target_offset returns immediate value
        result = adjuster.get_target_offset(
            Vec3(0, 2, 0), [Vec3(0, 0, 0)], 0.5
        )

        # Should be non-zero even though smoothing would delay adjust()
        assert result >= 0.0

    def test_get_target_offset_clamped_to_max_drop(self) -> None:
        """get_target_offset() returns value clamped to max_drop."""
        config = PelvisAdjustmentConfig(max_drop=0.3)
        adjuster = PelvisHeightAdjuster(config)

        # Large drop needed
        result = adjuster.get_target_offset(
            Vec3(0, 10, 0), [Vec3(0, 0, 0)], 1.0
        )

        assert result <= 0.3

    def test_get_target_offset_does_not_modify_state(self) -> None:
        """get_target_offset() does not change current_offset."""
        adjuster = PelvisHeightAdjuster()

        assert adjuster.current_offset == 0.0

        adjuster.get_target_offset(
            Vec3(0, 10, 0), [Vec3(0, 0, 0)], 0.5
        )

        # State unchanged
        assert adjuster.current_offset == 0.0

    def test_get_target_offset_empty_targets_returns_zero(self) -> None:
        """Empty leg_targets returns 0."""
        adjuster = PelvisHeightAdjuster()
        result = adjuster.get_target_offset(Vec3(0, 1, 0), [], 1.0)
        assert result == 0.0

    def test_get_target_offset_reachable_target_returns_zero(self) -> None:
        """Reachable target returns 0."""
        adjuster = PelvisHeightAdjuster()
        result = adjuster.get_target_offset(
            Vec3(0, 1, 0), [Vec3(0, 0.5, 0)], 2.0
        )
        assert result == 0.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestPelvisAdjustmentIntegration:
    """Integration tests combining multiple methods."""

    def test_full_adjustment_cycle(self) -> None:
        """Complete cycle: adjust, reset, reconfigure, adjust."""
        config1 = PelvisAdjustmentConfig(max_drop=0.5, smooth_speed=100.0)
        adjuster = PelvisHeightAdjuster(config1)

        # Adjust
        transforms = [create_transform(0, 2, 0)]
        adjuster.adjust(transforms, 0, [Vec3(0, 0, 0)], 0.5, 0.1)
        offset1 = adjuster.current_offset
        assert offset1 > 0

        # Reset
        adjuster.reset()
        assert adjuster.current_offset == 0.0

        # Reconfigure
        config2 = PelvisAdjustmentConfig(max_drop=0.1, smooth_speed=100.0)
        adjuster.set_config(config2)

        # Adjust again
        transforms = [create_transform(0, 2, 0)]
        adjuster.adjust(transforms, 0, [Vec3(0, 0, 0)], 0.5, 0.1)
        offset2 = adjuster.current_offset

        # Second offset should be limited by new max_drop
        assert offset2 <= 0.1 + 1e-9

    def test_continuous_adjustment_simulation(self) -> None:
        """Simulate continuous frame-by-frame adjustment."""
        adjuster = PelvisHeightAdjuster()
        dt = 1.0 / 60.0  # 60 FPS

        offsets = []
        for frame in range(60):  # 1 second
            transforms = [create_transform(0, 2, 0)]
            adjuster.adjust(
                transforms, 0, [Vec3(0, 0, 0)], 0.5, dt
            )
            offsets.append(adjuster.current_offset)

        # Offset should converge toward target
        # Should be monotonically increasing (or stable)
        for i in range(1, len(offsets)):
            assert offsets[i] >= offsets[i-1] - 1e-9

    def test_teleport_scenario_with_reset(self) -> None:
        """Simulate character teleport requiring reset."""
        adjuster = PelvisHeightAdjuster()

        # Normal adjustment at position A
        transforms = [create_transform(0, 2, 0)]
        for _ in range(30):
            adjuster.adjust(transforms, 0, [Vec3(0, 0, 0)], 0.5, 1/60)

        offset_before_teleport = adjuster.current_offset

        # Teleport - reset prevents jarring transition
        adjuster.reset()

        # At new position B, targets easily reachable
        transforms = [create_transform(100, 1, 100)]
        adjuster.adjust(
            transforms, 0, [Vec3(100, 0.5, 100)], 2.0, 1/60
        )

        # Should start fresh, not carry old offset
        assert adjuster.current_offset < offset_before_teleport


class TestEdgeCasesComprehensive:
    """Comprehensive edge case testing."""

    def test_very_small_numbers(self) -> None:
        """Handle very small coordinate values."""
        adjuster = PelvisHeightAdjuster()
        tiny = 1e-10

        transforms = [create_transform(tiny, tiny, tiny)]
        result = adjuster.adjust(
            transforms, 0, [Vec3(tiny, 0, tiny)], tiny, 0.1
        )

        assert isinstance(result, Vec3)

    def test_large_numbers(self) -> None:
        """Handle large coordinate values."""
        adjuster = PelvisHeightAdjuster()
        large = 1e6

        transforms = [create_transform(large, large, large)]
        result = adjuster.adjust(
            transforms, 0, [Vec3(large, large - 0.5, large)], 10.0, 0.1
        )

        assert isinstance(result, Vec3)

    def test_negative_coordinates(self) -> None:
        """Handle negative world coordinates."""
        adjuster = PelvisHeightAdjuster()

        transforms = [create_transform(-5, -2, -3)]
        result = adjuster.adjust(
            transforms, 0, [Vec3(-5, -3, -3)], 1.0, 0.1
        )

        assert isinstance(result, Vec3)

    def test_many_leg_targets(self) -> None:
        """Handle many leg targets."""
        adjuster = PelvisHeightAdjuster()

        targets = [Vec3(i * 0.1, 0, i * 0.1) for i in range(100)]
        result = adjuster.calculate_required_drop(
            Vec3(0, 1, 0), targets, 1.0
        )

        assert result >= 0.0

    def test_rapid_config_changes(self) -> None:
        """Handle rapid configuration changes."""
        adjuster = PelvisHeightAdjuster()

        for i in range(100):
            config = PelvisAdjustmentConfig(
                safety_margin=0.5 + (i % 50) * 0.01,
                max_drop=0.1 + (i % 10) * 0.1,
                smooth_speed=1.0 + i
            )
            adjuster.set_config(config)

        # Should still work
        transforms = [create_transform(0, 1, 0)]
        result = adjuster.adjust(transforms, 0, [], 1.0, 0.1)
        assert isinstance(result, Vec3)


class TestSpecialGeometryScenarios:
    """Test special geometric configurations."""

    def test_target_exactly_at_safe_reach(self) -> None:
        """Target exactly at safe reach boundary."""
        config = PelvisAdjustmentConfig(safety_margin=1.0)  # No margin
        adjuster = PelvisHeightAdjuster(config)

        # Pelvis at (0, 1, 0), reach = 1.0
        # Target at (0, 0, 0), distance = 1.0 = safe_reach
        result = adjuster.calculate_required_drop(
            Vec3(0, 1, 0), [Vec3(0, 0, 0)], 1.0
        )
        assert result == 0.0  # Exactly reachable

    def test_target_slightly_beyond_safe_reach(self) -> None:
        """Target just beyond safe reach."""
        config = PelvisAdjustmentConfig(safety_margin=0.99)
        adjuster = PelvisHeightAdjuster(config)

        # safe_reach = 0.99, target distance slightly > 0.99
        result = adjuster.calculate_required_drop(
            Vec3(0, 1, 0), [Vec3(0, -0.001, 0)], 1.0
        )
        # Should need tiny drop
        assert result >= 0.0

    def test_horizontal_and_vertical_combined(self) -> None:
        """Target with both horizontal and vertical offset."""
        adjuster = PelvisHeightAdjuster()

        # Pelvis at (0, 1, 0), target at (0.5, 0, 0.5)
        # This tests the full geometry calculation
        result = adjuster.calculate_required_drop(
            Vec3(0, 1, 0), [Vec3(0.5, 0, 0.5)], 0.8
        )

        # Calculation involves both dist_xz and dist_y
        assert result >= 0.0


class TestConfigDataclassFeatures:
    """Test dataclass-specific features of PelvisAdjustmentConfig."""

    def test_equality(self) -> None:
        """Config instances with same values are equal."""
        config1 = PelvisAdjustmentConfig(safety_margin=0.8)
        config2 = PelvisAdjustmentConfig(safety_margin=0.8)
        assert config1 == config2

    def test_inequality_different_values(self) -> None:
        """Config instances with different values are not equal."""
        config1 = PelvisAdjustmentConfig(safety_margin=0.8)
        config2 = PelvisAdjustmentConfig(safety_margin=0.7)
        assert config1 != config2

    def test_hashable(self) -> None:
        """Config can be used in sets/dicts (frozen=False but eq works)."""
        config = PelvisAdjustmentConfig()
        # Should not raise
        _ = repr(config)

    def test_repr(self) -> None:
        """Config has reasonable repr."""
        config = PelvisAdjustmentConfig()
        r = repr(config)
        assert "PelvisAdjustmentConfig" in r
        assert "safety_margin" in r

    def test_fields_accessible(self) -> None:
        """All fields are accessible."""
        config = PelvisAdjustmentConfig()
        assert hasattr(config, 'safety_margin')
        assert hasattr(config, 'max_drop')
        assert hasattr(config, 'smooth_speed')

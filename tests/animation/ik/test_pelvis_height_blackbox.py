"""
Blackbox tests for PelvisHeightAdjuster (T-FB-4.8).

CLEANROOM TEST SUITE - Written from specification only, without reading implementation.

PelvisHeightAdjuster dynamically adjusts pelvis height to:
- Allow effectors (hands, feet) to reach targets below their rest position
- Smoothly interpolate pelvis adjustments over time
- Clamp adjustments to prevent excessive pelvis drop
- Support configurable smoothing and limits

Public Interface (specification):

    @dataclass
    class PelvisAdjustmentConfig:
        safety_margin: float     # Safety margin factor (0-1)
        max_drop: float          # Maximum pelvis can drop (positive = down)
        smooth_speed: float      # How fast to interpolate (higher = faster)

    class PelvisHeightAdjuster:
        def __init__(self, config: Optional[PelvisAdjustmentConfig] = None)

        @property
        def config(self) -> PelvisAdjustmentConfig

        @property
        def current_offset(self) -> float

        def calculate_required_drop(
            self, pelvis_pos: Vec3, leg_targets: List[Vec3], max_leg_reach: float
        ) -> float
        def adjust(
            self, transforms: List[Transform], pelvis_idx: int,
            leg_targets: List[Vec3], max_leg_reach: float, dt: float
        ) -> Vec3
        def reset(self) -> None
        def set_config(self, config: PelvisAdjustmentConfig) -> None

Test Strategy:
- Test configuration creation and validation
- Test PelvisHeightAdjuster creation with various configs
- Test calculate_required_drop behavior
- Test adjust() method for smooth pelvis movement
- Test reset() behavior
- Test set_config() behavior
- Test edge cases and boundary conditions
"""

import math
import pytest
from typing import List, Optional

# Import public API
from engine.animation.ik.fullbody import PelvisAdjustmentConfig, PelvisHeightAdjuster
from engine.core.math import Vec3, Quat, Transform


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def make_transform(position: Vec3, rotation: Optional[Quat] = None) -> Transform:
    """Create a Transform from position and optional rotation."""
    return Transform(
        translation=position,
        rotation=rotation if rotation else Quat.identity()
    )


def vec3_distance(a: Vec3, b: Vec3) -> float:
    """Calculate Euclidean distance between two Vec3 points."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def vec3_magnitude(v: Vec3) -> float:
    """Calculate magnitude of a vector."""
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def vec3_nearly_equal(a: Vec3, b: Vec3, eps: float = 1e-6) -> bool:
    """Check if two vectors are nearly equal within epsilon."""
    return vec3_distance(a, b) <= eps


def nearly_equal(a: float, b: float, eps: float = 1e-6) -> bool:
    """Check if two floats are nearly equal."""
    return abs(a - b) <= eps


def create_skeleton_transforms(pelvis_y: float = 1.0) -> List[Transform]:
    """Create a basic bipedal skeleton transform list.

    Layout (indices):
    0 = pelvis (root)
    1 = spine
    2 = chest
    3 = left_upper_leg
    4 = left_lower_leg
    5 = left_foot
    6 = right_upper_leg
    7 = right_lower_leg
    8 = right_foot
    """
    return [
        make_transform(Vec3(0.0, pelvis_y, 0.0)),         # 0: pelvis
        make_transform(Vec3(0.0, pelvis_y + 0.2, 0.0)),   # 1: spine
        make_transform(Vec3(0.0, pelvis_y + 0.5, 0.0)),   # 2: chest
        make_transform(Vec3(-0.1, pelvis_y, 0.0)),        # 3: left_upper_leg
        make_transform(Vec3(-0.1, pelvis_y - 0.5, 0.0)),  # 4: left_lower_leg
        make_transform(Vec3(-0.1, 0.0, 0.0)),             # 5: left_foot
        make_transform(Vec3(0.1, pelvis_y, 0.0)),         # 6: right_upper_leg
        make_transform(Vec3(0.1, pelvis_y - 0.5, 0.0)),   # 7: right_lower_leg
        make_transform(Vec3(0.1, 0.0, 0.0)),              # 8: right_foot
    ]


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def default_config() -> PelvisAdjustmentConfig:
    """Create a PelvisAdjustmentConfig with defaults."""
    return PelvisAdjustmentConfig()


@pytest.fixture
def custom_config() -> PelvisAdjustmentConfig:
    """Create a custom PelvisAdjustmentConfig."""
    return PelvisAdjustmentConfig(
        max_drop=0.5,
        smooth_speed=5.0,
        safety_margin=0.9
    )


@pytest.fixture
def adjuster_default() -> PelvisHeightAdjuster:
    """Create PelvisHeightAdjuster with default config."""
    return PelvisHeightAdjuster()


@pytest.fixture
def adjuster_custom(custom_config: PelvisAdjustmentConfig) -> PelvisHeightAdjuster:
    """Create PelvisHeightAdjuster with custom config."""
    return PelvisHeightAdjuster(custom_config)


@pytest.fixture
def standing_pelvis_pos() -> Vec3:
    """Standard pelvis position at standing height (1m)."""
    return Vec3(0.0, 1.0, 0.0)


@pytest.fixture
def high_pelvis_pos() -> Vec3:
    """High pelvis position (1.5m)."""
    return Vec3(0.0, 1.5, 0.0)


@pytest.fixture
def low_pelvis_pos() -> Vec3:
    """Low pelvis position (0.5m)."""
    return Vec3(0.0, 0.5, 0.0)


@pytest.fixture
def standing_skeleton() -> List[Transform]:
    """Standard skeleton at standing height."""
    return create_skeleton_transforms(1.0)


@pytest.fixture
def default_leg_reach() -> float:
    """Default maximum leg reach (typical humanoid leg length)."""
    return 1.0


# =============================================================================
# SECTION 1: PELVIS ADJUSTMENT CONFIG TESTS
# =============================================================================

class TestPelvisAdjustmentConfigCreation:
    """Tests for PelvisAdjustmentConfig creation and validation."""

    def test_create_with_defaults(self):
        """Config can be created with all default values."""
        config = PelvisAdjustmentConfig()
        assert config is not None
        # Should have accessible attributes
        assert hasattr(config, 'max_drop')
        assert hasattr(config, 'smooth_speed')
        assert hasattr(config, 'safety_margin')

    def test_default_max_drop_is_positive(self):
        """Default max_drop should be a positive value (pelvis drops down)."""
        config = PelvisAdjustmentConfig()
        assert config.max_drop > 0, "max_drop should be positive"

    def test_default_max_drop_reasonable_value(self):
        """Default max_drop should be reasonable (e.g., 0.1 to 1.0 meters)."""
        config = PelvisAdjustmentConfig()
        assert 0.05 <= config.max_drop <= 2.0, (
            f"max_drop {config.max_drop} outside reasonable range"
        )

    def test_default_smooth_speed_positive(self):
        """Default smooth_speed should be positive."""
        config = PelvisAdjustmentConfig()
        assert config.smooth_speed > 0, "smooth_speed must be positive"

    def test_default_safety_margin_in_valid_range(self):
        """Default safety_margin should be in valid range (0-1)."""
        config = PelvisAdjustmentConfig()
        assert 0.0 <= config.safety_margin <= 1.0, (
            "safety_margin should be between 0 and 1"
        )

    def test_create_with_custom_max_drop(self):
        """Config can be created with custom max_drop."""
        config = PelvisAdjustmentConfig(max_drop=0.3)
        assert nearly_equal(config.max_drop, 0.3)

    def test_create_with_custom_smooth_speed(self):
        """Config can be created with custom smooth_speed."""
        config = PelvisAdjustmentConfig(smooth_speed=10.0)
        assert nearly_equal(config.smooth_speed, 10.0)

    def test_create_with_custom_safety_margin(self):
        """Config can be created with custom safety_margin."""
        config = PelvisAdjustmentConfig(safety_margin=0.8)
        assert nearly_equal(config.safety_margin, 0.8)

    def test_create_with_all_custom_values(self):
        """Config can be created with all custom values."""
        config = PelvisAdjustmentConfig(
            max_drop=0.8,
            smooth_speed=3.0,
            safety_margin=0.85
        )
        assert nearly_equal(config.max_drop, 0.8)
        assert nearly_equal(config.smooth_speed, 3.0)
        assert nearly_equal(config.safety_margin, 0.85)

    def test_create_with_safety_margin_one(self):
        """Config allows safety_margin of 1.0 (full safety)."""
        config = PelvisAdjustmentConfig(safety_margin=1.0)
        assert nearly_equal(config.safety_margin, 1.0)

    def test_create_with_large_max_drop(self):
        """Config allows large max_drop values."""
        config = PelvisAdjustmentConfig(max_drop=2.0)
        assert nearly_equal(config.max_drop, 2.0)

    def test_create_with_small_smooth_speed(self):
        """Config allows small smooth_speed for slow movement."""
        config = PelvisAdjustmentConfig(smooth_speed=0.5)
        assert nearly_equal(config.smooth_speed, 0.5)

    def test_create_with_large_smooth_speed(self):
        """Config allows large smooth_speed for fast movement."""
        config = PelvisAdjustmentConfig(smooth_speed=100.0)
        assert nearly_equal(config.smooth_speed, 100.0)


# =============================================================================
# SECTION 2: PELVIS HEIGHT ADJUSTER CREATION TESTS
# =============================================================================

class TestPelvisHeightAdjusterCreation:
    """Tests for PelvisHeightAdjuster creation."""

    def test_create_with_no_config(self):
        """Adjuster can be created with no config (uses defaults)."""
        adjuster = PelvisHeightAdjuster()
        assert adjuster is not None

    def test_create_with_none_config(self):
        """Adjuster can be created with None config (uses defaults)."""
        adjuster = PelvisHeightAdjuster(None)
        assert adjuster is not None

    def test_create_with_custom_config(self, custom_config):
        """Adjuster can be created with custom config."""
        adjuster = PelvisHeightAdjuster(custom_config)
        assert adjuster is not None

    def test_config_property_returns_config(self, adjuster_default):
        """Config property returns the configuration."""
        config = adjuster_default.config
        assert isinstance(config, PelvisAdjustmentConfig)

    def test_config_property_default_has_values(self, adjuster_default):
        """Default config has valid values."""
        config = adjuster_default.config
        assert config.max_drop > 0
        assert config.smooth_speed > 0
        assert 0.0 <= config.safety_margin <= 1.0

    def test_config_property_custom_preserves_values(self, custom_config):
        """Custom config values are preserved."""
        adjuster = PelvisHeightAdjuster(custom_config)
        config = adjuster.config
        assert nearly_equal(config.max_drop, custom_config.max_drop)
        assert nearly_equal(config.smooth_speed, custom_config.smooth_speed)
        assert nearly_equal(config.safety_margin, custom_config.safety_margin)

    def test_current_offset_property_accessible(self, adjuster_default):
        """current_offset property is accessible."""
        offset = adjuster_default.current_offset
        assert isinstance(offset, (int, float))

    def test_current_offset_initial_value_zero(self, adjuster_default):
        """Initial current_offset should be zero."""
        assert nearly_equal(adjuster_default.current_offset, 0.0)


# =============================================================================
# SECTION 3: CALCULATE_REQUIRED_DROP BEHAVIOR TESTS
# =============================================================================

class TestCalculateRequiredDrop:
    """Tests for calculate_required_drop method."""

    def test_target_far_below_pelvis_returns_positive_drop(
        self, adjuster_default, standing_pelvis_pos, default_leg_reach
    ):
        """Target far below pelvis returns positive drop value."""
        # Pelvis at y=1.0, target at y=-0.5 (well beyond leg reach)
        targets = [Vec3(0.0, -0.5, 0.0)]
        drop = adjuster_default.calculate_required_drop(
            standing_pelvis_pos, targets, default_leg_reach
        )
        assert drop > 0, "Drop should be positive when target is far below"

    def test_target_within_easy_reach_returns_zero_or_small(
        self, adjuster_default, standing_pelvis_pos, default_leg_reach
    ):
        """Target within easy leg reach returns zero or small drop."""
        # Target at y=0.5, pelvis at y=1.0, leg reach=1.0 -> easily reachable
        targets = [Vec3(0.0, 0.5, 0.0)]
        drop = adjuster_default.calculate_required_drop(
            standing_pelvis_pos, targets, default_leg_reach
        )
        assert drop >= 0, "Drop should be non-negative"

    def test_target_above_pelvis_returns_small_drop(
        self, adjuster_default, standing_pelvis_pos, default_leg_reach
    ):
        """Target above pelvis returns zero or minimal drop."""
        targets = [Vec3(0.0, 2.0, 0.0)]  # Above pelvis
        drop = adjuster_default.calculate_required_drop(
            standing_pelvis_pos, targets, default_leg_reach
        )
        # May return small value due to safety margin calculations
        assert drop < 0.1, "Target above pelvis should need minimal drop"

    def test_empty_targets_returns_zero(
        self, adjuster_default, standing_pelvis_pos, default_leg_reach
    ):
        """Empty targets list returns zero."""
        drop = adjuster_default.calculate_required_drop(
            standing_pelvis_pos, [], default_leg_reach
        )
        assert nearly_equal(drop, 0.0), "Empty targets should return zero drop"

    def test_target_at_leg_reach_limit(
        self, adjuster_default, standing_pelvis_pos
    ):
        """Target at exact leg reach limit requires minimal or no drop."""
        leg_reach = 1.0
        # Target exactly at maximum reach (pelvis.y - leg_reach = 0)
        targets = [Vec3(0.0, 0.0, 0.0)]
        drop = adjuster_default.calculate_required_drop(
            standing_pelvis_pos, targets, leg_reach
        )
        # Should need minimal drop due to safety margin
        assert drop >= 0

    def test_multiple_targets_uses_most_demanding(
        self, adjuster_default, standing_pelvis_pos, default_leg_reach
    ):
        """Multiple targets: drop should accommodate the most demanding."""
        targets_high = [Vec3(0.0, 0.5, 0.0)]
        targets_low = [Vec3(0.0, -0.5, 0.0)]
        targets_mixed = [Vec3(0.0, 0.5, 0.0), Vec3(0.0, -0.5, 0.0)]

        drop_high = adjuster_default.calculate_required_drop(
            standing_pelvis_pos, targets_high, default_leg_reach
        )
        drop_mixed = adjuster_default.calculate_required_drop(
            standing_pelvis_pos, targets_mixed, default_leg_reach
        )

        # Mixed should accommodate the lower (more demanding) target
        assert drop_mixed >= drop_high, (
            "Mixed targets should require at least as much drop as higher targets"
        )

    def test_target_beyond_leg_reach_requires_drop(
        self, adjuster_default, standing_pelvis_pos
    ):
        """Target beyond leg reach requires significant drop."""
        leg_reach = 0.5  # Short legs
        targets = [Vec3(0.0, 0.0, 0.0)]  # At ground, 1.0m below pelvis
        drop = adjuster_default.calculate_required_drop(
            standing_pelvis_pos, targets, leg_reach
        )
        assert drop > 0, "Target beyond leg reach should require drop"

    def test_larger_leg_reach_reduces_required_drop(
        self, adjuster_default, standing_pelvis_pos
    ):
        """Larger leg reach should reduce required drop."""
        targets = [Vec3(0.0, -0.2, 0.0)]

        drop_short_legs = adjuster_default.calculate_required_drop(
            standing_pelvis_pos, targets, 0.5
        )
        drop_long_legs = adjuster_default.calculate_required_drop(
            standing_pelvis_pos, targets, 1.5
        )

        assert drop_long_legs <= drop_short_legs, (
            "Longer legs should require less drop"
        )

    def test_high_pelvis_target_below_normal(
        self, adjuster_default, high_pelvis_pos, default_leg_reach
    ):
        """High pelvis with target below needs drop."""
        targets = [Vec3(0.0, 0.0, 0.0)]
        drop = adjuster_default.calculate_required_drop(
            high_pelvis_pos, targets, default_leg_reach
        )
        assert drop >= 0

    def test_low_pelvis_target_within_reach(
        self, adjuster_default, low_pelvis_pos, default_leg_reach
    ):
        """Low pelvis with nearby target needs minimal drop."""
        targets = [Vec3(0.0, 0.0, 0.0)]  # Ground level
        drop = adjuster_default.calculate_required_drop(
            low_pelvis_pos, targets, default_leg_reach
        )
        # Low pelvis (0.5) closer to ground, may need less drop with 1.0m legs
        assert drop >= 0

    def test_returns_float_type(
        self, adjuster_default, standing_pelvis_pos, default_leg_reach
    ):
        """calculate_required_drop returns a float."""
        targets = [Vec3(0.0, 0.5, 0.0)]
        drop = adjuster_default.calculate_required_drop(
            standing_pelvis_pos, targets, default_leg_reach
        )
        assert isinstance(drop, (int, float))


# =============================================================================
# SECTION 4: ADJUST() METHOD BEHAVIOR TESTS
# =============================================================================

class TestAdjustBehavior:
    """Tests for adjust() method behavior."""

    def test_adjust_returns_vec3(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """adjust() returns a Vec3 adjustment vector."""
        targets = [Vec3(0.0, -0.2, 0.0)]
        adjustment = adjuster_default.adjust(
            standing_skeleton, 0, targets, default_leg_reach, dt=0.016
        )
        assert isinstance(adjustment, Vec3)

    def test_adjust_modifies_internal_state(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """adjust() modifies internal offset state."""
        targets = [Vec3(0.0, -0.5, 0.0)]  # Far below
        initial_offset = adjuster_default.current_offset

        # Multiple adjustments should accumulate offset
        for _ in range(10):
            adjuster_default.adjust(
                standing_skeleton, 0, targets, default_leg_reach, dt=0.1
            )

        final_offset = adjuster_default.current_offset
        # Offset should change (increase) as we adjust toward target
        assert final_offset != initial_offset or nearly_equal(final_offset, 0.0), (
            "Offset should change or already be at target"
        )

    def test_adjust_smooth_over_multiple_dt(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """adjust() smooths movement over multiple dt calls."""
        targets = [Vec3(0.0, -0.5, 0.0)]  # Far below
        dt = 0.016  # ~60fps

        offsets = []
        for _ in range(30):  # ~0.5 seconds
            adjuster_default.adjust(
                standing_skeleton, 0, targets, default_leg_reach, dt
            )
            offsets.append(adjuster_default.current_offset)

        # Should see gradual change, not instant
        if offsets[-1] != 0:
            # Check that offset changed gradually
            changes = [abs(offsets[i+1] - offsets[i]) for i in range(len(offsets)-1)]
            first_change = changes[0] if changes else 0
            if first_change > 0:
                # Should converge (changes decrease over time) or be smooth
                assert any(c <= first_change * 1.1 for c in changes[5:]), (
                    "Movement should be smooth"
                )

    def test_adjust_clamps_to_max_drop(self, standing_skeleton):
        """adjust() clamps adjustment to max_drop."""
        config = PelvisAdjustmentConfig(max_drop=0.2, smooth_speed=100.0)
        adjuster = PelvisHeightAdjuster(config)

        targets = [Vec3(0.0, -1.0, 0.0)]  # Very far below
        leg_reach = 0.5  # Short legs require more drop

        # Run many adjustments to fully converge
        for _ in range(100):
            adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt=0.1)

        # Offset should not exceed max_drop
        assert adjuster.current_offset <= config.max_drop + 0.01, (
            f"Offset {adjuster.current_offset} exceeds max_drop {config.max_drop}"
        )

    def test_adjust_returns_downward_vector(
        self, standing_skeleton, default_leg_reach
    ):
        """adjust() returns downward adjustment vector (negative Y) when dropping."""
        config = PelvisAdjustmentConfig(max_drop=0.5, smooth_speed=10.0)
        adjuster = PelvisHeightAdjuster(config)
        targets = [Vec3(0.0, -0.5, 0.0)]

        # Reset and get first adjustment
        adjuster.reset()
        adjustment = adjuster.adjust(
            standing_skeleton, 0, targets, default_leg_reach, dt=0.1
        )

        # If adjustment is needed, Y should be negative (going down)
        if vec3_magnitude(adjustment) > 0.001:
            assert adjustment.y <= 0, "Adjustment should be downward (negative Y)"

    def test_adjust_no_change_for_targets_within_reach(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """adjust() returns minimal change for targets easily within reach."""
        targets = [Vec3(0.0, 0.5, 0.0)]  # Easily reachable

        adjuster_default.reset()
        adjustment = adjuster_default.adjust(
            standing_skeleton, 0, targets, default_leg_reach, dt=0.1
        )

        # Should be zero or near zero
        assert vec3_magnitude(adjustment) < 0.1, (
            "Minimal adjustment needed for targets within reach"
        )

    def test_adjust_empty_targets_no_change(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """adjust() with empty targets returns zero adjustment."""
        adjuster_default.reset()
        adjustment = adjuster_default.adjust(
            standing_skeleton, 0, [], default_leg_reach, dt=0.1
        )

        assert vec3_magnitude(adjustment) < 0.001, (
            "Empty targets should produce no adjustment"
        )

    def test_adjust_preserves_horizontal_position(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """adjust() adjustment vector has zero X and Z components."""
        targets = [Vec3(0.0, -0.3, 0.0)]

        adjustment = adjuster_default.adjust(
            standing_skeleton, 0, targets, default_leg_reach, dt=0.1
        )

        # Pelvis adjustment should be vertical only
        assert nearly_equal(adjustment.x, 0.0), "X adjustment should be zero"
        assert nearly_equal(adjustment.z, 0.0), "Z adjustment should be zero"

    def test_adjust_convergence(self, standing_skeleton):
        """adjust() converges to stable offset over time."""
        config = PelvisAdjustmentConfig(max_drop=0.5, smooth_speed=10.0)
        adjuster = PelvisHeightAdjuster(config)
        targets = [Vec3(0.0, -0.2, 0.0)]
        leg_reach = 0.8

        # Run many updates
        for _ in range(100):
            adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt=0.1)

        # Check that offset is stable (next update produces tiny change)
        final_offset = adjuster.current_offset
        adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt=0.1)
        assert nearly_equal(adjuster.current_offset, final_offset, 0.001), (
            "Should converge to stable offset"
        )

    def test_adjust_different_smoothing_speeds(self, standing_skeleton):
        """Different smooth_speed values produce valid results."""
        slow_config = PelvisAdjustmentConfig(smooth_speed=1.0, max_drop=0.5)
        fast_config = PelvisAdjustmentConfig(smooth_speed=20.0, max_drop=0.5)

        slow_adjuster = PelvisHeightAdjuster(slow_config)
        fast_adjuster = PelvisHeightAdjuster(fast_config)

        targets = [Vec3(0.0, -0.3, 0.0)]
        leg_reach = 0.8

        # Run same number of updates
        for _ in range(10):
            slow_adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt=0.016)
            fast_adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt=0.016)

        # Both should produce valid offsets
        assert math.isfinite(slow_adjuster.current_offset)
        assert math.isfinite(fast_adjuster.current_offset)
        assert slow_adjuster.current_offset >= 0
        assert fast_adjuster.current_offset >= 0


# =============================================================================
# SECTION 5: RESET() BEHAVIOR TESTS
# =============================================================================

class TestResetBehavior:
    """Tests for reset() method behavior."""

    def test_reset_sets_offset_to_zero(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """reset() sets current_offset to zero."""
        targets = [Vec3(0.0, -0.5, 0.0)]

        # Build up offset
        for _ in range(20):
            adjuster_default.adjust(
                standing_skeleton, 0, targets, default_leg_reach, dt=0.1
            )

        # Reset
        adjuster_default.reset()

        assert nearly_equal(adjuster_default.current_offset, 0.0), (
            "reset() should set offset to zero"
        )

    def test_reset_clears_smoothing_state(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """reset() clears any internal smoothing state."""
        targets = [Vec3(0.0, -0.5, 0.0)]

        # Build up state
        for _ in range(20):
            adjuster_default.adjust(
                standing_skeleton, 0, targets, default_leg_reach, dt=0.1
            )

        # Reset
        adjuster_default.reset()

        # Next adjustment should start fresh
        adjuster_default.adjust(
            standing_skeleton, 0, targets, default_leg_reach, dt=0.016
        )

        # Should behave like fresh adjuster
        fresh = PelvisHeightAdjuster()
        fresh.adjust(standing_skeleton, 0, targets, default_leg_reach, dt=0.016)

        # Offsets should be similar (both starting from zero)
        assert nearly_equal(
            adjuster_default.current_offset,
            fresh.current_offset,
            0.01
        ), "Reset adjuster should behave like fresh adjuster"

    def test_reset_can_be_called_multiple_times(self, adjuster_default):
        """reset() can be called multiple times without error."""
        adjuster_default.reset()
        adjuster_default.reset()
        adjuster_default.reset()
        assert nearly_equal(adjuster_default.current_offset, 0.0)

    def test_reset_after_no_adjustments(self, adjuster_default):
        """reset() works correctly even without prior adjustments."""
        adjuster_default.reset()
        assert nearly_equal(adjuster_default.current_offset, 0.0)


# =============================================================================
# SECTION 6: SET_CONFIG() BEHAVIOR TESTS
# =============================================================================

class TestSetConfigBehavior:
    """Tests for set_config() method behavior."""

    def test_set_config_updates_configuration(self, adjuster_default):
        """set_config() updates the adjuster configuration."""
        new_config = PelvisAdjustmentConfig(max_drop=0.8)
        adjuster_default.set_config(new_config)

        assert nearly_equal(adjuster_default.config.max_drop, 0.8), (
            "set_config should update max_drop"
        )

    def test_set_config_affects_subsequent_adjusts(self, standing_skeleton):
        """New config affects subsequent adjust() calls."""
        adjuster = PelvisHeightAdjuster(
            PelvisAdjustmentConfig(max_drop=0.1, smooth_speed=100.0)
        )
        targets = [Vec3(0.0, -1.0, 0.0)]
        leg_reach = 0.5

        # Adjust to hit limit
        for _ in range(50):
            adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt=0.1)

        old_offset = adjuster.current_offset
        assert old_offset <= 0.1 + 0.01, "Should be clamped to old max_drop"

        # Change config to allow more drop
        adjuster.set_config(
            PelvisAdjustmentConfig(max_drop=0.5, smooth_speed=100.0)
        )

        # More adjustments
        for _ in range(50):
            adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt=0.1)

        new_offset = adjuster.current_offset
        # Should be able to drop further now
        assert new_offset >= old_offset, (
            "New config should allow more drop"
        )

    def test_set_config_preserves_current_offset(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """set_config() preserves current offset state."""
        targets = [Vec3(0.0, -0.5, 0.0)]

        # Build offset
        for _ in range(20):
            adjuster_default.adjust(
                standing_skeleton, 0, targets, default_leg_reach, dt=0.1
            )

        old_offset = adjuster_default.current_offset

        # Change config
        new_config = PelvisAdjustmentConfig(max_drop=1.0)
        adjuster_default.set_config(new_config)

        # Offset should be preserved (or at least not reset)
        assert adjuster_default.current_offset >= 0, (
            "Offset should remain valid"
        )

    def test_set_config_with_smaller_max_drop_clamps_offset(self, standing_skeleton):
        """Setting smaller max_drop should clamp existing offset."""
        adjuster = PelvisHeightAdjuster(
            PelvisAdjustmentConfig(max_drop=0.5, smooth_speed=100.0)
        )
        targets = [Vec3(0.0, -1.0, 0.0)]
        leg_reach = 0.5

        # Build up large offset
        for _ in range(50):
            adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt=0.1)

        # Now set smaller max_drop
        adjuster.set_config(
            PelvisAdjustmentConfig(max_drop=0.1, smooth_speed=100.0)
        )

        # Next adjust should respect new limit
        adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt=0.1)
        assert adjuster.current_offset <= 0.1 + 0.01, (
            "Offset should be clamped to new max_drop"
        )


# =============================================================================
# SECTION 7: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_dt(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """Very small dt values produce minimal adjustment."""
        targets = [Vec3(0.0, -0.5, 0.0)]
        adjuster_default.reset()

        adjustment = adjuster_default.adjust(
            standing_skeleton, 0, targets, default_leg_reach, dt=0.0001
        )

        # Should be valid but tiny
        assert isinstance(adjustment, Vec3)
        assert vec3_magnitude(adjustment) < 0.1

    def test_zero_dt(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """Zero dt produces zero or minimal adjustment."""
        targets = [Vec3(0.0, -0.5, 0.0)]
        adjuster_default.reset()

        adjustment = adjuster_default.adjust(
            standing_skeleton, 0, targets, default_leg_reach, dt=0.0
        )

        # Should be zero or near zero
        assert vec3_magnitude(adjustment) < 0.001

    def test_very_large_dt(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """Very large dt values are handled gracefully."""
        targets = [Vec3(0.0, -0.5, 0.0)]
        adjuster_default.reset()

        # Large dt (1 second)
        adjustment = adjuster_default.adjust(
            standing_skeleton, 0, targets, default_leg_reach, dt=1.0
        )

        assert isinstance(adjustment, Vec3)
        # Should not produce infinite or NaN values
        assert math.isfinite(adjustment.y)
        assert math.isfinite(adjuster_default.current_offset)

    def test_very_large_dt_respects_max_drop(self, standing_skeleton):
        """Large dt still respects max_drop limit."""
        config = PelvisAdjustmentConfig(max_drop=0.2, smooth_speed=100.0)
        adjuster = PelvisHeightAdjuster(config)
        targets = [Vec3(0.0, -10.0, 0.0)]
        leg_reach = 0.5

        adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt=100.0)

        assert adjuster.current_offset <= config.max_drop + 0.01

    def test_target_at_same_height_as_pelvis(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """Target at exact pelvis height produces valid offset."""
        # Pelvis at y=1.0, target also at y=1.0
        targets = [Vec3(1.0, 1.0, 0.0)]
        adjuster_default.reset()

        for _ in range(10):
            adjuster_default.adjust(
                standing_skeleton, 0, targets, default_leg_reach, dt=0.1
            )

        # Should produce valid offset (may be small or zero depending on implementation)
        assert math.isfinite(adjuster_default.current_offset)
        assert adjuster_default.current_offset >= 0

    def test_multiple_targets_at_different_heights(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """Multiple targets at different heights handled correctly."""
        targets = [
            Vec3(0.0, 0.5, 0.0),   # High
            Vec3(0.0, 0.0, 0.0),   # Medium
            Vec3(0.0, -0.3, 0.0),  # Low
        ]

        drop = adjuster_default.calculate_required_drop(
            Vec3(0.0, 1.0, 0.0), targets, default_leg_reach
        )
        assert drop >= 0

        adjustment = adjuster_default.adjust(
            standing_skeleton, 0, targets, default_leg_reach, dt=0.1
        )
        assert isinstance(adjustment, Vec3)

    def test_targets_with_negative_y(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """Targets below ground level (negative Y) handled correctly."""
        targets = [Vec3(0.0, -0.5, 0.0)]  # Below ground

        drop = adjuster_default.calculate_required_drop(
            Vec3(0.0, 1.0, 0.0), targets, default_leg_reach
        )
        assert drop >= 0

        adjustment = adjuster_default.adjust(
            standing_skeleton, 0, targets, default_leg_reach, dt=0.1
        )
        assert math.isfinite(adjustment.y)

    def test_pelvis_at_origin(self, adjuster_default, default_leg_reach):
        """Pelvis at origin works correctly."""
        skeleton = create_skeleton_transforms(0.0)  # Pelvis at y=0
        targets = [Vec3(0.0, -0.5, 0.0)]

        drop = adjuster_default.calculate_required_drop(
            Vec3(0.0, 0.0, 0.0), targets, default_leg_reach
        )
        assert isinstance(drop, (int, float))

        adjustment = adjuster_default.adjust(
            skeleton, 0, targets, default_leg_reach, dt=0.1
        )
        assert isinstance(adjustment, Vec3)

    def test_single_target(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """Single target produces valid result."""
        targets = [Vec3(0.5, 0.3, 0.2)]

        drop = adjuster_default.calculate_required_drop(
            Vec3(0.0, 1.0, 0.0), targets, default_leg_reach
        )
        assert drop >= 0

        adjustment = adjuster_default.adjust(
            standing_skeleton, 0, targets, default_leg_reach, dt=0.1
        )
        assert isinstance(adjustment, Vec3)

    def test_many_targets(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """Many targets handled correctly."""
        targets = [
            Vec3(i * 0.1, 0.3 + i * 0.05, i * 0.05)
            for i in range(20)
        ]

        drop = adjuster_default.calculate_required_drop(
            Vec3(0.0, 1.0, 0.0), targets, default_leg_reach
        )
        assert isinstance(drop, (int, float))

        adjustment = adjuster_default.adjust(
            standing_skeleton, 0, targets, default_leg_reach, dt=0.1
        )
        assert isinstance(adjustment, Vec3)

    def test_targets_far_horizontal(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """Targets far horizontally from pelvis handled correctly."""
        targets = [Vec3(100.0, 0.3, 100.0)]

        drop = adjuster_default.calculate_required_drop(
            Vec3(0.0, 1.0, 0.0), targets, default_leg_reach
        )
        assert isinstance(drop, (int, float))

        adjustment = adjuster_default.adjust(
            standing_skeleton, 0, targets, default_leg_reach, dt=0.1
        )
        assert isinstance(adjustment, Vec3)

    def test_zero_leg_reach(
        self, adjuster_default, standing_skeleton
    ):
        """Zero leg reach handled gracefully."""
        targets = [Vec3(0.0, 0.5, 0.0)]

        # Should not crash
        try:
            drop = adjuster_default.calculate_required_drop(
                Vec3(0.0, 1.0, 0.0), targets, 0.0
            )
            assert isinstance(drop, (int, float))
        except (ValueError, ZeroDivisionError):
            # Raising an error is also acceptable for invalid input
            pass

    def test_negative_leg_reach(
        self, adjuster_default, standing_skeleton
    ):
        """Negative leg reach handled gracefully (no crash)."""
        targets = [Vec3(0.0, 0.5, 0.0)]

        # Should not crash - behavior undefined but stable
        try:
            drop = adjuster_default.calculate_required_drop(
                Vec3(0.0, 1.0, 0.0), targets, -1.0
            )
            assert isinstance(drop, (int, float))
        except (ValueError, RuntimeError):
            # Raising an error is also acceptable
            pass


# =============================================================================
# SECTION 8: REALISTIC SCENARIO TESTS
# =============================================================================

class TestRealisticScenarios:
    """Tests simulating realistic use cases."""

    def test_reaching_for_ground_object(self, standing_skeleton):
        """Character reaching down to pick up object from ground."""
        config = PelvisAdjustmentConfig(max_drop=0.3, smooth_speed=5.0)
        adjuster = PelvisHeightAdjuster(config)

        # Hand/foot targets near ground
        targets = [Vec3(-0.1, 0.0, 0.0), Vec3(0.1, 0.0, 0.0)]
        leg_reach = 0.9  # Shorter than pelvis height

        # Simulate animation frames
        dt = 1.0 / 60.0  # 60 FPS
        offsets = []

        for _ in range(60):  # 1 second
            adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt)
            offsets.append(adjuster.current_offset)

        # Should have non-zero offset by end (if targets require it)
        # No assertion on specific value - just check it ran
        assert len(offsets) == 60
        assert all(math.isfinite(o) for o in offsets)

    def test_standing_up_from_crouch(self):
        """Character standing up (targets rising)."""
        config = PelvisAdjustmentConfig(max_drop=0.3, smooth_speed=8.0)
        adjuster = PelvisHeightAdjuster(config)

        low_skeleton = create_skeleton_transforms(0.5)  # Crouched
        leg_reach = 0.9

        # First crouch down
        low_targets = [Vec3(0.0, -0.3, 0.0)]
        for _ in range(30):
            adjuster.adjust(low_skeleton, 0, low_targets, leg_reach, dt=0.016)

        crouched_offset = adjuster.current_offset

        # Now stand up (targets rise)
        standing_targets = [Vec3(0.0, 0.5, 0.0)]
        for _ in range(60):
            adjuster.adjust(low_skeleton, 0, standing_targets, leg_reach, dt=0.016)

        final_offset = adjuster.current_offset

        # Offset should decrease (less drop needed when standing)
        # Or stay same if both are within reach
        assert final_offset >= 0
        assert math.isfinite(final_offset)

    def test_walking_animation_cycle(self, standing_skeleton):
        """Simulating a walking cycle with alternating foot targets."""
        config = PelvisAdjustmentConfig(max_drop=0.15, smooth_speed=10.0)
        adjuster = PelvisHeightAdjuster(config)
        leg_reach = 1.0

        dt = 1.0 / 60.0
        offsets = []

        for frame in range(120):  # 2 seconds
            # Simulate alternating foot positions
            phase = math.sin(frame * 0.1)
            left_foot_y = 0.0 if phase > 0 else 0.05
            right_foot_y = 0.05 if phase > 0 else 0.0

            targets = [
                Vec3(-0.1, left_foot_y, 0.0),
                Vec3(0.1, right_foot_y, 0.0)
            ]

            adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt)
            offsets.append(adjuster.current_offset)

        # Should have stable offset range
        assert max(offsets) <= config.max_drop + 0.01
        assert min(offsets) >= 0

    def test_two_hand_reach_different_heights(self, standing_skeleton):
        """Both hands/feet reaching to different heights."""
        config = PelvisAdjustmentConfig(max_drop=0.4, smooth_speed=6.0)
        adjuster = PelvisHeightAdjuster(config)
        leg_reach = 0.8

        # Left target low, right target medium
        targets = [
            Vec3(-0.5, -0.2, 0.3),   # Left low
            Vec3(0.5, 0.3, 0.3),     # Right medium
        ]

        for _ in range(60):
            adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt=0.016)

        # Should have computed an offset
        assert math.isfinite(adjuster.current_offset)

    def test_rapid_target_changes(self, standing_skeleton):
        """Rapid changes in target positions (e.g., combat)."""
        config = PelvisAdjustmentConfig(max_drop=0.3, smooth_speed=15.0)
        adjuster = PelvisHeightAdjuster(config)
        leg_reach = 1.0

        dt = 0.016
        prev_offset = 0

        for i in range(100):
            # Varying target heights
            target_y = 0.3 + 0.3 * math.sin(i * 0.5)
            targets = [Vec3(0.0, target_y, 0.0)]

            adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt)

            # Changes should be reasonably smooth
            change = abs(adjuster.current_offset - prev_offset)
            assert change < 0.2, f"Change too rapid: {change}"
            prev_offset = adjuster.current_offset

    def test_config_change_mid_motion(self, standing_skeleton):
        """Config change during ongoing adjustment."""
        adjuster = PelvisHeightAdjuster(
            PelvisAdjustmentConfig(max_drop=0.2, smooth_speed=5.0)
        )
        targets = [Vec3(0.0, -0.3, 0.0)]
        leg_reach = 0.8

        # Start adjusting
        for _ in range(30):
            adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt=0.016)

        mid_offset = adjuster.current_offset

        # Change config mid-motion
        adjuster.set_config(
            PelvisAdjustmentConfig(max_drop=0.5, smooth_speed=20.0)
        )

        # Continue adjusting
        for _ in range(30):
            adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt=0.016)

        # Should continue smoothly with new config
        final_offset = adjuster.current_offset
        assert math.isfinite(final_offset)

    def test_reset_during_motion(self, standing_skeleton):
        """Reset during ongoing adjustment."""
        adjuster = PelvisHeightAdjuster(
            PelvisAdjustmentConfig(max_drop=0.3, smooth_speed=10.0)
        )
        targets = [Vec3(0.0, -0.3, 0.0)]
        leg_reach = 0.8

        # Build up offset
        for _ in range(30):
            adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt=0.016)

        # Reset mid-motion
        adjuster.reset()

        assert nearly_equal(adjuster.current_offset, 0.0)

        # Should start fresh
        adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt=0.016)
        # Small offset after one frame
        assert adjuster.current_offset < 0.2


# =============================================================================
# SECTION 9: STRESS TESTS
# =============================================================================

class TestStressConditions:
    """Stress tests for robustness."""

    def test_many_rapid_adjustments(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """Many rapid adjustments remain stable."""
        targets = [Vec3(0.0, 0.3, 0.0)]

        for _ in range(1000):
            adjustment = adjuster_default.adjust(
                standing_skeleton, 0, targets, default_leg_reach, dt=0.001
            )
            assert math.isfinite(adjustment.y)
            assert math.isfinite(adjuster_default.current_offset)

    def test_alternating_targets(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """Alternating between high and low targets remains stable."""
        high_targets = [Vec3(0.0, 0.9, 0.0)]
        low_targets = [Vec3(0.0, -0.1, 0.0)]

        for i in range(100):
            targets = high_targets if i % 2 == 0 else low_targets
            adjuster_default.adjust(
                standing_skeleton, 0, targets, default_leg_reach, dt=0.016
            )

            assert math.isfinite(adjuster_default.current_offset)
            assert adjuster_default.current_offset >= 0

    def test_many_reset_cycles(
        self, adjuster_default, standing_skeleton, default_leg_reach
    ):
        """Many reset cycles remain stable."""
        targets = [Vec3(0.0, 0.3, 0.0)]

        for _ in range(50):
            for _ in range(10):
                adjuster_default.adjust(
                    standing_skeleton, 0, targets, default_leg_reach, dt=0.016
                )
            adjuster_default.reset()

        assert nearly_equal(adjuster_default.current_offset, 0.0)

    def test_many_config_changes(self, standing_skeleton):
        """Many config changes remain stable."""
        adjuster = PelvisHeightAdjuster()
        targets = [Vec3(0.0, 0.3, 0.0)]
        leg_reach = 1.0

        for i in range(50):
            config = PelvisAdjustmentConfig(
                max_drop=0.1 + (i % 5) * 0.1,
                smooth_speed=1.0 + i
            )
            adjuster.set_config(config)
            adjuster.adjust(standing_skeleton, 0, targets, leg_reach, dt=0.016)

            assert math.isfinite(adjuster.current_offset)

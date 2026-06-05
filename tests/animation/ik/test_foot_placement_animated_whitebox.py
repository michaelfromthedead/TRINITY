"""Whitebox tests for FootPlacementAnimated class.

Tests the FootPlacementAnimated implementation covering:
- __init__ stores base placement
- _animation_time initialization and behavior
- set_height_curves stores curves
- update() advances _animation_time
- solve() without curves uses base.solve() directly
- solve() with curves applies height offsets from curve(animation_time)
- curve values affect left_foot.height_offset and right_foot.height_offset
- different curve functions (constant, linear, sine)
- animation_time accumulation and edge cases
"""

from __future__ import annotations

import math
import pytest
from typing import List, Optional, Callable
from unittest.mock import Mock, MagicMock, patch, call

from engine.animation.ik.foot_placement import (
    FootState,
    FootData,
    FootPlacementResult,
    FootPlacement,
    FootPlacementAnimated,
    RaycastCallback,
    RaycastHit,
)
from engine.animation.ik.config import (
    FOOT_PLACEMENT_RAY_LENGTH,
    FOOT_PLACEMENT_FOOT_HEIGHT,
    FOOT_PLACEMENT_BLEND_SPEED,
)
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform


# =============================================================================
# Helper Functions
# =============================================================================

def create_biped_transforms(num_bones: int = 10) -> List[Transform]:
    """Create transforms for a simple biped skeleton."""
    positions = {
        0: Vec3(0, 1.0, 0),      # Pelvis
        1: Vec3(-0.1, 1.0, 0),   # Left upper leg
        2: Vec3(-0.1, 0.5, 0),   # Left lower leg
        3: Vec3(-0.1, 0.1, 0),   # Left foot
        4: Vec3(-0.1, 0.0, 0),   # Left toe
        5: Vec3(0.1, 1.0, 0),    # Right upper leg
        6: Vec3(0.1, 0.5, 0),    # Right lower leg
        7: Vec3(0.1, 0.1, 0),    # Right foot
        8: Vec3(0.1, 0.0, 0),    # Right toe
    }

    transforms = []
    for i in range(num_bones):
        pos = positions.get(i, Vec3(0, 0, 0))
        transforms.append(Transform(pos, Quat.identity()))

    return transforms


def create_left_foot_data() -> FootData:
    """Create left foot data for testing."""
    return FootData(
        upper_leg=1,
        lower_leg=2,
        foot=3,
        toe=4
    )


def create_right_foot_data() -> FootData:
    """Create right foot data for testing."""
    return FootData(
        upper_leg=5,
        lower_leg=6,
        foot=7,
        toe=8
    )


def create_flat_ground_raycast() -> RaycastCallback:
    """Create raycast callback for flat ground at y=0."""
    def raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
        if direction.y >= 0:
            return None

        t = -origin.y / direction.y
        if t > max_dist:
            return None

        hit_pos = Vec3(origin.x + direction.x * t, 0, origin.z + direction.z * t)
        return RaycastHit(hit=True, position=hit_pos, normal=Vec3(0, 1, 0), distance=t)

    return raycast


def create_no_hit_raycast() -> RaycastCallback:
    """Create raycast callback that never hits."""
    def raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
        return None

    return raycast


def create_base_placement() -> FootPlacement:
    """Create a basic FootPlacement for testing."""
    left_foot = create_left_foot_data()
    right_foot = create_right_foot_data()
    raycast = create_flat_ground_raycast()
    return FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=raycast)


def create_base_placement_no_raycast() -> FootPlacement:
    """Create a FootPlacement without raycast for testing."""
    left_foot = create_left_foot_data()
    right_foot = create_right_foot_data()
    return FootPlacement(left_foot, right_foot, pelvis=0, raycast_callback=None)


# =============================================================================
# Test __init__
# =============================================================================

class TestFootPlacementAnimatedInit:
    """Tests for FootPlacementAnimated.__init__ method."""

    def test_init_stores_base_placement(self):
        """Test __init__ stores base placement reference."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        assert animated.base is base

    def test_init_stores_base_with_different_configurations(self):
        """Test __init__ stores base with different configurations."""
        left_foot = FootData(upper_leg=10, lower_leg=11, foot=12)
        right_foot = FootData(upper_leg=20, lower_leg=21, foot=22)
        base = FootPlacement(left_foot, right_foot, pelvis=5)

        animated = FootPlacementAnimated(base)

        assert animated.base is base
        assert animated.base.pelvis == 5

    def test_init_left_height_curve_is_none(self):
        """Test __init__ sets _left_height_curve to None."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        assert animated._left_height_curve is None

    def test_init_right_height_curve_is_none(self):
        """Test __init__ sets _right_height_curve to None."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        assert animated._right_height_curve is None

    def test_init_animation_time_is_zero(self):
        """Test __init__ sets _animation_time to 0.0."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        assert animated._animation_time == 0.0

    def test_init_animation_time_is_float(self):
        """Test _animation_time is a float type."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        assert isinstance(animated._animation_time, float)

    def test_init_preserves_base_raycast(self):
        """Test __init__ preserves base raycast callback."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        assert animated.base._raycast is not None

    def test_init_with_base_no_raycast(self):
        """Test __init__ with base that has no raycast."""
        base = create_base_placement_no_raycast()
        animated = FootPlacementAnimated(base)

        assert animated.base is base
        assert animated.base._raycast is None

    def test_init_does_not_modify_base(self):
        """Test __init__ does not modify the base placement."""
        base = create_base_placement()
        original_left_offset = base.left_foot.height_offset
        original_right_offset = base.right_foot.height_offset

        animated = FootPlacementAnimated(base)

        assert base.left_foot.height_offset == original_left_offset
        assert base.right_foot.height_offset == original_right_offset


# =============================================================================
# Test _animation_time behavior
# =============================================================================

class TestAnimationTime:
    """Tests for _animation_time attribute behavior."""

    def test_animation_time_starts_at_zero(self):
        """Test animation time starts at 0."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        assert animated._animation_time == 0.0

    def test_animation_time_exact_zero(self):
        """Test animation time is exactly 0.0, not approximately."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        assert animated._animation_time == pytest.approx(0.0, abs=1e-15)

    def test_animation_time_is_accessible(self):
        """Test _animation_time can be accessed."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        _ = animated._animation_time  # Should not raise

    def test_animation_time_can_be_set_directly(self):
        """Test _animation_time can be set directly if needed."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated._animation_time = 5.0

        assert animated._animation_time == 5.0


# =============================================================================
# Test set_height_curves
# =============================================================================

class TestSetHeightCurves:
    """Tests for set_height_curves method."""

    def test_set_height_curves_stores_left_curve(self):
        """Test set_height_curves stores left curve."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        left_curve = lambda t: 0.1
        right_curve = lambda t: 0.2

        animated.set_height_curves(left_curve, right_curve)

        assert animated._left_height_curve is left_curve

    def test_set_height_curves_stores_right_curve(self):
        """Test set_height_curves stores right curve."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        left_curve = lambda t: 0.1
        right_curve = lambda t: 0.2

        animated.set_height_curves(left_curve, right_curve)

        assert animated._right_height_curve is right_curve

    def test_set_height_curves_with_sine_functions(self):
        """Test set_height_curves with sine wave functions."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        left_curve = lambda t: math.sin(t) * 0.1
        right_curve = lambda t: math.sin(t + math.pi) * 0.1

        animated.set_height_curves(left_curve, right_curve)

        assert animated._left_height_curve is not None
        assert animated._right_height_curve is not None

    def test_set_height_curves_with_constant_functions(self):
        """Test set_height_curves with constant functions."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        left_curve = lambda t: 0.05
        right_curve = lambda t: 0.03

        animated.set_height_curves(left_curve, right_curve)

        assert animated._left_height_curve(0) == 0.05
        assert animated._right_height_curve(0) == 0.03

    def test_set_height_curves_with_linear_functions(self):
        """Test set_height_curves with linear functions."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        left_curve = lambda t: t * 0.01
        right_curve = lambda t: t * 0.02

        animated.set_height_curves(left_curve, right_curve)

        assert animated._left_height_curve(10) == 0.1
        assert animated._right_height_curve(10) == 0.2

    def test_set_height_curves_replaces_previous_curves(self):
        """Test set_height_curves replaces previously set curves."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        first_left = lambda t: 0.1
        first_right = lambda t: 0.2
        animated.set_height_curves(first_left, first_right)

        second_left = lambda t: 0.3
        second_right = lambda t: 0.4
        animated.set_height_curves(second_left, second_right)

        assert animated._left_height_curve is second_left
        assert animated._right_height_curve is second_right

    def test_set_height_curves_with_named_functions(self):
        """Test set_height_curves with named function definitions."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        def left_height(t: float) -> float:
            return t * 0.1

        def right_height(t: float) -> float:
            return t * 0.2

        animated.set_height_curves(left_height, right_height)

        assert animated._left_height_curve is left_height
        assert animated._right_height_curve is right_height

    def test_set_height_curves_callable_verification(self):
        """Test that stored curves are callable."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        left_curve = lambda t: t ** 2
        right_curve = lambda t: t ** 3

        animated.set_height_curves(left_curve, right_curve)

        assert callable(animated._left_height_curve)
        assert callable(animated._right_height_curve)


# =============================================================================
# Test update method
# =============================================================================

class TestUpdate:
    """Tests for update method."""

    def test_update_advances_animation_time(self):
        """Test update() advances _animation_time by dt."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.update(0.1)

        assert animated._animation_time == pytest.approx(0.1, rel=1e-6)

    def test_update_accumulates_time(self):
        """Test update() accumulates time across multiple calls."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.update(0.1)
        animated.update(0.2)
        animated.update(0.3)

        assert animated._animation_time == pytest.approx(0.6, rel=1e-6)

    def test_update_with_zero_dt(self):
        """Test update() with dt=0 does not change time."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.update(0.5)
        animated.update(0.0)

        assert animated._animation_time == pytest.approx(0.5, rel=1e-6)

    def test_update_with_small_dt(self):
        """Test update() with very small dt."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.update(0.001)

        assert animated._animation_time == pytest.approx(0.001, rel=1e-6)

    def test_update_with_large_dt(self):
        """Test update() with large dt."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.update(10.0)

        assert animated._animation_time == pytest.approx(10.0, rel=1e-6)

    def test_update_with_60fps_frame_time(self):
        """Test update() with typical 60fps frame time."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        dt = 1.0 / 60.0
        animated.update(dt)

        assert animated._animation_time == pytest.approx(dt, rel=1e-6)

    def test_update_many_frames_at_60fps(self):
        """Test update() accumulated over many 60fps frames."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        dt = 1.0 / 60.0
        for _ in range(60):
            animated.update(dt)

        assert animated._animation_time == pytest.approx(1.0, rel=1e-3)

    def test_update_with_negative_dt(self):
        """Test update() with negative dt (edge case)."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.update(1.0)
        animated.update(-0.5)

        # Implementation just adds dt, so negative works
        assert animated._animation_time == pytest.approx(0.5, rel=1e-6)

    def test_update_precision_over_long_duration(self):
        """Test update() maintains precision over long duration."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        dt = 1.0 / 60.0
        # Simulate 10 minutes at 60fps
        for _ in range(60 * 60 * 10):
            animated.update(dt)

        expected = 10 * 60.0  # 10 minutes in seconds
        assert animated._animation_time == pytest.approx(expected, rel=1e-3)


# =============================================================================
# Test solve without curves
# =============================================================================

class TestSolveWithoutCurves:
    """Tests for solve() when no height curves are set."""

    def test_solve_without_curves_returns_result(self):
        """Test solve() without curves returns FootPlacementResult."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        transforms = create_biped_transforms()
        result = animated.solve(transforms, Vec3.zero())

        assert isinstance(result, FootPlacementResult)

    def test_solve_without_curves_success(self):
        """Test solve() without curves succeeds."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        transforms = create_biped_transforms()
        result = animated.solve(transforms, Vec3.zero())

        assert result.success is True

    def test_solve_without_curves_does_not_modify_height_offset(self):
        """Test solve() without curves does not modify foot height offsets."""
        base = create_base_placement()
        base.left_foot.height_offset = 0.0
        base.right_foot.height_offset = 0.0

        animated = FootPlacementAnimated(base)

        transforms = create_biped_transforms()
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == 0.0
        assert base.right_foot.height_offset == 0.0

    def test_solve_without_curves_preserves_existing_offset(self):
        """Test solve() without curves preserves existing height offsets."""
        base = create_base_placement()
        base.left_foot.height_offset = 0.05
        base.right_foot.height_offset = 0.03

        animated = FootPlacementAnimated(base)

        transforms = create_biped_transforms()
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == 0.05
        assert base.right_foot.height_offset == 0.03

    def test_solve_without_curves_calls_base_solve(self):
        """Test solve() without curves calls base.solve()."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        with patch.object(base, 'solve', wraps=base.solve) as mock_solve:
            transforms = create_biped_transforms()
            animated.solve(transforms, Vec3.zero())

            mock_solve.assert_called_once()

    def test_solve_without_curves_passes_correct_arguments(self):
        """Test solve() passes correct arguments to base.solve()."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        transforms = create_biped_transforms()
        char_pos = Vec3(1.0, 2.0, 3.0)
        dt = 0.05

        with patch.object(base, 'solve', wraps=base.solve) as mock_solve:
            animated.solve(transforms, char_pos, dt)

            mock_solve.assert_called_once_with(transforms, char_pos, dt)


# =============================================================================
# Test solve with curves
# =============================================================================

class TestSolveWithCurves:
    """Tests for solve() when height curves are set."""

    def test_solve_with_curves_applies_left_offset(self):
        """Test solve() applies left curve value to height_offset."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: 0.1, lambda t: 0.0)

        transforms = create_biped_transforms()
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == 0.1

    def test_solve_with_curves_applies_right_offset(self):
        """Test solve() applies right curve value to height_offset."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: 0.0, lambda t: 0.15)

        transforms = create_biped_transforms()
        animated.solve(transforms, Vec3.zero())

        assert base.right_foot.height_offset == 0.15

    def test_solve_with_curves_applies_both_offsets(self):
        """Test solve() applies both left and right curve values."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: 0.05, lambda t: 0.03)

        transforms = create_biped_transforms()
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == 0.05
        assert base.right_foot.height_offset == 0.03

    def test_solve_uses_animation_time_for_curves(self):
        """Test solve() passes _animation_time to curves."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        left_calls = []
        right_calls = []

        def left_curve(t):
            left_calls.append(t)
            return t * 0.01

        def right_curve(t):
            right_calls.append(t)
            return t * 0.02

        animated.set_height_curves(left_curve, right_curve)
        animated._animation_time = 5.0

        transforms = create_biped_transforms()
        animated.solve(transforms, Vec3.zero())

        assert left_calls == [5.0]
        assert right_calls == [5.0]

    def test_solve_with_sine_curves(self):
        """Test solve() with sine wave curves."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(
            lambda t: math.sin(t) * 0.1,
            lambda t: math.cos(t) * 0.1
        )

        animated._animation_time = math.pi / 2

        transforms = create_biped_transforms()
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == pytest.approx(0.1, rel=1e-6)
        assert base.right_foot.height_offset == pytest.approx(0.0, abs=1e-10)

    def test_solve_with_linear_curves(self):
        """Test solve() with linear curves."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(
            lambda t: t * 0.01,
            lambda t: t * 0.02
        )

        animated._animation_time = 10.0

        transforms = create_biped_transforms()
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == pytest.approx(0.1, rel=1e-6)
        assert base.right_foot.height_offset == pytest.approx(0.2, rel=1e-6)

    def test_solve_with_negative_curve_values(self):
        """Test solve() handles negative curve values."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: -0.05, lambda t: -0.1)

        transforms = create_biped_transforms()
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == -0.05
        assert base.right_foot.height_offset == -0.1

    def test_solve_curve_value_changes_with_time(self):
        """Test solve() curve value changes with animation time."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: t * 0.1, lambda t: t * 0.1)

        transforms = create_biped_transforms()

        animated._animation_time = 1.0
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == pytest.approx(0.1, rel=1e-6)

        animated._animation_time = 2.0
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == pytest.approx(0.2, rel=1e-6)

    def test_solve_returns_base_solve_result(self):
        """Test solve() returns the result from base.solve()."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: 0.05, lambda t: 0.03)

        transforms = create_biped_transforms()
        result = animated.solve(transforms, Vec3.zero())

        assert isinstance(result, FootPlacementResult)
        assert result.success is True


# =============================================================================
# Test curve effects on foot height offset
# =============================================================================

class TestCurveEffectsOnHeightOffset:
    """Tests verifying curve values affect foot height offsets."""

    def test_constant_curve_sets_constant_offset(self):
        """Test constant curve function sets constant offset."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: 0.123, lambda t: 0.456)

        transforms = create_biped_transforms()

        # Solve at different times, offset should remain constant
        animated._animation_time = 0.0
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == 0.123

        animated._animation_time = 100.0
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == 0.123

    def test_step_function_curve(self):
        """Test step function curve."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        def step_curve(t):
            return 0.1 if t >= 1.0 else 0.0

        animated.set_height_curves(step_curve, lambda t: 0.0)

        transforms = create_biped_transforms()

        animated._animation_time = 0.5
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == 0.0

        animated._animation_time = 1.5
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == 0.1

    def test_oscillating_curve(self):
        """Test oscillating curve affects offset correctly."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(
            lambda t: math.sin(t * 2 * math.pi) * 0.1,
            lambda t: 0.0
        )

        transforms = create_biped_transforms()

        # At t=0, sin(0) = 0
        animated._animation_time = 0.0
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == pytest.approx(0.0, abs=1e-10)

        # At t=0.25, sin(pi/2) = 1
        animated._animation_time = 0.25
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == pytest.approx(0.1, rel=1e-6)

        # At t=0.5, sin(pi) = 0
        animated._animation_time = 0.5
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == pytest.approx(0.0, abs=1e-10)

        # At t=0.75, sin(3pi/2) = -1
        animated._animation_time = 0.75
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == pytest.approx(-0.1, rel=1e-6)

    def test_opposite_phase_curves(self):
        """Test left and right curves with opposite phase."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(
            lambda t: math.sin(t) * 0.1,
            lambda t: math.sin(t + math.pi) * 0.1  # Opposite phase
        )

        transforms = create_biped_transforms()

        animated._animation_time = math.pi / 2
        animated.solve(transforms, Vec3.zero())

        # sin(pi/2) = 1, sin(3pi/2) = -1
        assert base.left_foot.height_offset == pytest.approx(0.1, rel=1e-6)
        assert base.right_foot.height_offset == pytest.approx(-0.1, rel=1e-6)

    def test_exponential_decay_curve(self):
        """Test exponential decay curve."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(
            lambda t: math.exp(-t) * 0.1,
            lambda t: 0.0
        )

        transforms = create_biped_transforms()

        animated._animation_time = 0.0
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == pytest.approx(0.1, rel=1e-6)

        animated._animation_time = 1.0
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == pytest.approx(0.1 * math.exp(-1), rel=1e-6)

    def test_quadratic_curve(self):
        """Test quadratic curve."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(
            lambda t: t ** 2 * 0.01,
            lambda t: 0.0
        )

        transforms = create_biped_transforms()

        animated._animation_time = 3.0
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == pytest.approx(0.09, rel=1e-6)


# =============================================================================
# Test different curve function types
# =============================================================================

class TestDifferentCurveFunctions:
    """Tests for different types of curve functions."""

    def test_constant_zero_curve(self):
        """Test constant zero curve."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: 0.0, lambda t: 0.0)

        transforms = create_biped_transforms()
        animated._animation_time = 100.0
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == 0.0
        assert base.right_foot.height_offset == 0.0

    def test_constant_positive_curve(self):
        """Test constant positive curve."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: 0.25, lambda t: 0.25)

        transforms = create_biped_transforms()
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == 0.25
        assert base.right_foot.height_offset == 0.25

    def test_constant_negative_curve(self):
        """Test constant negative curve."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: -0.1, lambda t: -0.2)

        transforms = create_biped_transforms()
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == -0.1
        assert base.right_foot.height_offset == -0.2

    def test_linear_increasing_curve(self):
        """Test linear increasing curve."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: t * 0.1, lambda t: t * 0.1)

        transforms = create_biped_transforms()

        for time in [0.0, 1.0, 2.0, 5.0]:
            animated._animation_time = time
            animated.solve(transforms, Vec3.zero())
            assert base.left_foot.height_offset == pytest.approx(time * 0.1, rel=1e-6)

    def test_linear_decreasing_curve(self):
        """Test linear decreasing curve."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: 1.0 - t * 0.1, lambda t: 0.0)

        transforms = create_biped_transforms()

        animated._animation_time = 5.0
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == pytest.approx(0.5, rel=1e-6)

    def test_sine_curve_full_cycle(self):
        """Test sine curve over full cycle."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(
            lambda t: math.sin(t) * 0.1,
            lambda t: 0.0
        )

        transforms = create_biped_transforms()

        # Test key points in sine cycle
        test_points = [
            (0, 0),
            (math.pi / 2, 0.1),
            (math.pi, 0),
            (3 * math.pi / 2, -0.1),
            (2 * math.pi, 0)
        ]

        for time, expected in test_points:
            animated._animation_time = time
            animated.solve(transforms, Vec3.zero())
            assert base.left_foot.height_offset == pytest.approx(expected, abs=1e-10)

    def test_cosine_curve(self):
        """Test cosine curve."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(
            lambda t: math.cos(t) * 0.1,
            lambda t: 0.0
        )

        transforms = create_biped_transforms()

        animated._animation_time = 0.0
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == pytest.approx(0.1, rel=1e-6)

        animated._animation_time = math.pi
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == pytest.approx(-0.1, rel=1e-6)

    def test_sawtooth_curve(self):
        """Test sawtooth wave curve."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        # Sawtooth: linearly increases then resets
        def sawtooth(t, period=1.0):
            return (t % period) * 0.1

        animated.set_height_curves(sawtooth, lambda t: 0.0)

        transforms = create_biped_transforms()

        animated._animation_time = 0.5
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == pytest.approx(0.05, rel=1e-6)

        animated._animation_time = 1.5
        animated.solve(transforms, Vec3.zero())
        assert base.left_foot.height_offset == pytest.approx(0.05, rel=1e-6)


# =============================================================================
# Test animation time integration with curves
# =============================================================================

class TestAnimationTimeWithCurves:
    """Tests for animation time integration with curve evaluation."""

    def test_update_then_solve_uses_updated_time(self):
        """Test update() followed by solve() uses updated time."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: t * 0.1, lambda t: 0.0)

        transforms = create_biped_transforms()

        animated.update(2.0)
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == pytest.approx(0.2, rel=1e-6)

    def test_multiple_updates_then_solve(self):
        """Test multiple update() calls then solve()."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: t * 0.1, lambda t: 0.0)

        transforms = create_biped_transforms()

        animated.update(1.0)
        animated.update(1.0)
        animated.update(1.0)
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == pytest.approx(0.3, rel=1e-6)

    def test_interleaved_update_and_solve(self):
        """Test interleaved update() and solve() calls."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: t, lambda t: 0.0)

        transforms = create_biped_transforms()

        offsets = []
        for i in range(5):
            animated.update(1.0)
            animated.solve(transforms, Vec3.zero())
            offsets.append(base.left_foot.height_offset)

        assert offsets == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0], rel=1e-6)

    def test_solve_does_not_advance_time(self):
        """Test solve() does not advance animation time."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: 0.0, lambda t: 0.0)

        transforms = create_biped_transforms()

        animated._animation_time = 5.0
        animated.solve(transforms, Vec3.zero())

        assert animated._animation_time == 5.0

    def test_time_precision_affects_curve_output(self):
        """Test time precision affects curve output."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        # Curve very sensitive to time
        animated.set_height_curves(lambda t: t * 1000, lambda t: 0.0)

        transforms = create_biped_transforms()

        animated._animation_time = 0.001
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == pytest.approx(1.0, rel=1e-6)


# =============================================================================
# Test edge cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_solve_with_empty_transforms(self):
        """Test solve() handles empty transforms gracefully or raises."""
        base = create_base_placement_no_raycast()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: 0.1, lambda t: 0.1)

        # This should handle gracefully (returns false success without raycast)
        result = animated.solve([], Vec3.zero())

        # Without raycast, base returns success=False
        assert result.success is False

    def test_solve_with_very_large_animation_time(self):
        """Test solve() with very large animation time."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(
            lambda t: math.sin(t) * 0.1,
            lambda t: 0.0
        )

        transforms = create_biped_transforms()

        animated._animation_time = 1e10
        result = animated.solve(transforms, Vec3.zero())

        assert result.success is True
        # sin(1e10) is some valid value
        assert isinstance(base.left_foot.height_offset, float)

    def test_solve_with_zero_animation_time(self):
        """Test solve() with animation time at exactly zero."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: t, lambda t: t)

        transforms = create_biped_transforms()

        animated._animation_time = 0.0
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == 0.0
        assert base.right_foot.height_offset == 0.0

    def test_curve_returns_infinity(self):
        """Test handling when curve returns infinity."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: float('inf'), lambda t: 0.0)

        transforms = create_biped_transforms()
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == float('inf')

    def test_curve_returns_nan(self):
        """Test handling when curve returns NaN."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: float('nan'), lambda t: 0.0)

        transforms = create_biped_transforms()
        animated.solve(transforms, Vec3.zero())

        assert math.isnan(base.left_foot.height_offset)

    def test_only_left_curve_set(self):
        """Test when only left curve is set via direct assignment."""
        base = create_base_placement()
        base.left_foot.height_offset = 0.0
        base.right_foot.height_offset = 0.5

        animated = FootPlacementAnimated(base)
        animated._left_height_curve = lambda t: 0.1
        # _right_height_curve remains None

        transforms = create_biped_transforms()
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == 0.1
        assert base.right_foot.height_offset == 0.5  # Unchanged

    def test_only_right_curve_set(self):
        """Test when only right curve is set via direct assignment."""
        base = create_base_placement()
        base.left_foot.height_offset = 0.5
        base.right_foot.height_offset = 0.0

        animated = FootPlacementAnimated(base)
        animated._right_height_curve = lambda t: 0.2
        # _left_height_curve remains None

        transforms = create_biped_transforms()
        animated.solve(transforms, Vec3.zero())

        assert base.left_foot.height_offset == 0.5  # Unchanged
        assert base.right_foot.height_offset == 0.2

    def test_base_without_raycast_still_applies_curves(self):
        """Test curves are applied even when base has no raycast."""
        base = create_base_placement_no_raycast()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: 0.07, lambda t: 0.08)

        transforms = create_biped_transforms()
        result = animated.solve(transforms, Vec3.zero())

        # Curves should still be applied before base.solve()
        assert base.left_foot.height_offset == 0.07
        assert base.right_foot.height_offset == 0.08
        # But result will be success=False due to no raycast
        assert result.success is False


# =============================================================================
# Test reset and state management
# =============================================================================

class TestResetAndStateManagement:
    """Tests for reset and state management scenarios."""

    def test_reset_animation_time_manually(self):
        """Test resetting animation time manually."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.update(10.0)
        assert animated._animation_time == 10.0

        animated._animation_time = 0.0
        assert animated._animation_time == 0.0

    def test_animation_time_continuity_after_curve_change(self):
        """Test animation time persists after changing curves."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.update(5.0)
        animated.set_height_curves(lambda t: t, lambda t: t)

        assert animated._animation_time == 5.0

    def test_multiple_animated_wrappers_same_base(self):
        """Test multiple animated wrappers can share same base."""
        base = create_base_placement()
        animated1 = FootPlacementAnimated(base)
        animated2 = FootPlacementAnimated(base)

        assert animated1.base is animated2.base

        animated1.set_height_curves(lambda t: 0.1, lambda t: 0.0)
        animated2.set_height_curves(lambda t: 0.2, lambda t: 0.0)

        # They have different curves
        assert animated1._left_height_curve(0) != animated2._left_height_curve(0)

    def test_animated_does_not_affect_original_base_curves(self):
        """Test FootPlacementAnimated doesn't store curves on base."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(lambda t: 0.1, lambda t: 0.2)

        # Base doesn't have these curve attributes
        assert not hasattr(base, '_left_height_curve')
        assert not hasattr(base, '_right_height_curve')


# =============================================================================
# Test solve with different dt values
# =============================================================================

class TestSolveWithDifferentDt:
    """Tests for solve() with different dt values."""

    def test_solve_default_dt(self):
        """Test solve() with default dt value."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        transforms = create_biped_transforms()
        result = animated.solve(transforms, Vec3.zero())

        assert result.success is True

    def test_solve_with_explicit_dt(self):
        """Test solve() with explicitly specified dt."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        transforms = create_biped_transforms()
        result = animated.solve(transforms, Vec3.zero(), dt=0.033)

        assert result.success is True

    def test_solve_with_very_small_dt(self):
        """Test solve() with very small dt."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        transforms = create_biped_transforms()
        result = animated.solve(transforms, Vec3.zero(), dt=0.001)

        assert result.success is True

    def test_solve_with_large_dt(self):
        """Test solve() with large dt."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        transforms = create_biped_transforms()
        result = animated.solve(transforms, Vec3.zero(), dt=1.0)

        assert result.success is True

    def test_solve_passes_dt_to_base(self):
        """Test solve() passes dt parameter to base.solve()."""
        base = create_base_placement()
        animated = FootPlacementAnimated(base)

        with patch.object(base, 'solve', wraps=base.solve) as mock_solve:
            transforms = create_biped_transforms()
            animated.solve(transforms, Vec3.zero(), dt=0.05)

            call_args = mock_solve.call_args
            assert call_args[0][2] == 0.05  # Third positional arg is dt

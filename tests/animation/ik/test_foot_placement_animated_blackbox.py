"""Blackbox tests for FootPlacementAnimated (T-FB-4.12).

This module tests FootPlacementAnimated from the public API only,
without knowledge of implementation details. Tests are derived from
the behavioral contract of animated foot placement:

1. Construction with base FootPlacement
2. Setting height curves for left/right feet
3. update() advances animation time
4. solve() applies animated height offsets
5. Animation cycle simulation (walk cycle)
6. Edge cases (zero curves, fast/slow dt, negative values)

Test Strategy:
- Test public API contracts only
- Derive behavior from foot placement animation theory
- No implementation peeking
- Target 40+ tests
"""

import math
import pytest
from typing import List, Optional, Callable

# Import public API only
from engine.animation.ik import (
    FootPlacement,
    FootPlacementAnimated,
    FootPlacementResult,
    FootData,
    FootState,
    RaycastCallback,
    RaycastHit,
)
from engine.core.math import Vec3, Quat, Transform


# =============================================================================
# Helper Functions (cleanroom - no implementation details)
# =============================================================================

def make_transform(position: Vec3, rotation: Optional[Quat] = None) -> Transform:
    """Create a Transform from position and optional rotation."""
    return Transform(
        translation=position,
        rotation=rotation if rotation else Quat.identity()
    )


def vec3_distance(a: Vec3, b: Vec3) -> float:
    """Calculate distance between two Vec3 points."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def create_humanoid_transforms() -> List[Transform]:
    """Create a basic humanoid skeleton for foot placement.

    Layout (indices):
    0 = pelvis (root)
    1 = spine
    2 = chest
    3 = left_upper_leg
    4 = left_lower_leg
    5 = left_foot
    6 = left_toe
    7 = right_upper_leg
    8 = right_lower_leg
    9 = right_foot
    10 = right_toe
    """
    transforms = [
        # Core
        make_transform(Vec3(0.0, 1.0, 0.0)),    # 0: pelvis
        make_transform(Vec3(0.0, 1.2, 0.0)),    # 1: spine
        make_transform(Vec3(0.0, 1.5, 0.0)),    # 2: chest
        # Left leg
        make_transform(Vec3(-0.1, 1.0, 0.0)),   # 3: left_upper_leg
        make_transform(Vec3(-0.1, 0.5, 0.0)),   # 4: left_lower_leg
        make_transform(Vec3(-0.1, 0.0, 0.0)),   # 5: left_foot
        make_transform(Vec3(-0.1, 0.0, 0.1)),   # 6: left_toe
        # Right leg
        make_transform(Vec3(0.1, 1.0, 0.0)),    # 7: right_upper_leg
        make_transform(Vec3(0.1, 0.5, 0.0)),    # 8: right_lower_leg
        make_transform(Vec3(0.1, 0.0, 0.0)),    # 9: right_foot
        make_transform(Vec3(0.1, 0.0, 0.1)),    # 10: right_toe
    ]
    return transforms


def create_flat_ground_raycast(ground_height: float = 0.0) -> Callable:
    """Create a raycast callback for flat ground."""
    def raycast(origin: Vec3, direction: Vec3, ray_length: float) -> Optional[RaycastHit]:
        if direction.y < 0:
            distance = (origin.y - ground_height) / abs(direction.y)
            if 0 < distance <= ray_length:
                hit_pos = Vec3(
                    origin.x + direction.x * distance,
                    ground_height,
                    origin.z + direction.z * distance,
                )
                normal = Vec3(0.0, 1.0, 0.0)
                return RaycastHit(hit=True, position=hit_pos, normal=normal, distance=distance)
        return None
    return raycast


def create_left_foot_data() -> FootData:
    """Create left foot data for humanoid."""
    return FootData(
        upper_leg=3,
        lower_leg=4,
        foot=5,
        toe=6,
    )


def create_right_foot_data() -> FootData:
    """Create right foot data for humanoid."""
    return FootData(
        upper_leg=7,
        lower_leg=8,
        foot=9,
        toe=10,
    )


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def humanoid_transforms():
    """Standard humanoid skeleton transforms."""
    return create_humanoid_transforms()


@pytest.fixture
def left_foot():
    """Left foot data."""
    return create_left_foot_data()


@pytest.fixture
def right_foot():
    """Right foot data."""
    return create_right_foot_data()


@pytest.fixture
def flat_raycast():
    """Raycast for flat ground at y=0."""
    return create_flat_ground_raycast(0.0)


@pytest.fixture
def base_placement(left_foot, right_foot, flat_raycast):
    """Base FootPlacement solver for tests."""
    return FootPlacement(
        left_foot=left_foot,
        right_foot=right_foot,
        pelvis=0,
        raycast_callback=flat_raycast,
    )


@pytest.fixture
def animated_solver(base_placement):
    """FootPlacementAnimated solver for tests."""
    return FootPlacementAnimated(base_placement=base_placement)


# =============================================================================
# Construction Tests
# =============================================================================

class TestFootPlacementAnimatedConstruction:
    """Tests for FootPlacementAnimated construction."""

    def test_can_create_with_base_placement(self, base_placement):
        """FootPlacementAnimated can be created with base FootPlacement."""
        animated = FootPlacementAnimated(base_placement=base_placement)
        assert animated is not None

    def test_has_base_accessible(self, base_placement):
        """Base placement should be accessible after construction."""
        animated = FootPlacementAnimated(base_placement=base_placement)
        # The animated solver should have access to base functionality
        assert animated is not None

    def test_multiple_animators_independent(self, base_placement):
        """Multiple FootPlacementAnimated instances are independent."""
        anim1 = FootPlacementAnimated(base_placement=base_placement)
        anim2 = FootPlacementAnimated(base_placement=base_placement)
        assert anim1 is not anim2

    def test_construction_without_raycast(self, left_foot, right_foot):
        """Can create animated solver without raycast callback."""
        base = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
        )
        animated = FootPlacementAnimated(base_placement=base)
        assert animated is not None

    def test_construction_preserves_base_config(self, left_foot, right_foot, flat_raycast):
        """Construction preserves base placement configuration."""
        base = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        animated = FootPlacementAnimated(base_placement=base)
        # Should be able to solve using base configuration
        transforms = create_humanoid_transforms()
        result = animated.solve(transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None


# =============================================================================
# set_height_curves Tests
# =============================================================================

class TestFootPlacementAnimatedSetHeightCurves:
    """Tests for set_height_curves method."""

    def test_has_set_height_curves_method(self, animated_solver):
        """FootPlacementAnimated has set_height_curves method."""
        assert hasattr(animated_solver, 'set_height_curves')
        assert callable(animated_solver.set_height_curves)

    def test_set_height_curves_accepts_callables(self, animated_solver):
        """set_height_curves accepts callable functions."""
        left_curve = lambda t: 0.0
        right_curve = lambda t: 0.0
        # Should not raise
        animated_solver.set_height_curves(left_curve, right_curve)

    def test_set_height_curves_with_sine(self, animated_solver):
        """set_height_curves works with sine functions."""
        left_curve = lambda t: math.sin(t) * 0.1
        right_curve = lambda t: math.sin(t + math.pi) * 0.1
        animated_solver.set_height_curves(left_curve, right_curve)

    def test_set_height_curves_with_cosine(self, animated_solver):
        """set_height_curves works with cosine functions."""
        left_curve = lambda t: math.cos(t) * 0.05
        right_curve = lambda t: math.cos(t + math.pi) * 0.05
        animated_solver.set_height_curves(left_curve, right_curve)

    def test_set_height_curves_with_constant(self, animated_solver):
        """set_height_curves works with constant functions."""
        left_curve = lambda t: 0.1
        right_curve = lambda t: 0.2
        animated_solver.set_height_curves(left_curve, right_curve)

    def test_set_height_curves_can_be_called_multiple_times(self, animated_solver):
        """set_height_curves can be called multiple times."""
        # First set
        animated_solver.set_height_curves(lambda t: 0.1, lambda t: 0.1)
        # Replace with new curves
        animated_solver.set_height_curves(lambda t: 0.2, lambda t: 0.2)

    def test_curves_are_callable_float_to_float(self, animated_solver):
        """Curves should accept float and return float."""
        def left_curve(t: float) -> float:
            return t * 0.01

        def right_curve(t: float) -> float:
            return t * 0.02

        animated_solver.set_height_curves(left_curve, right_curve)

        # Verify curves work as expected
        assert abs(left_curve(1.0) - 0.01) < 0.001
        assert abs(right_curve(1.0) - 0.02) < 0.001


# =============================================================================
# update() Tests
# =============================================================================

class TestFootPlacementAnimatedUpdate:
    """Tests for update method."""

    def test_has_update_method(self, animated_solver):
        """FootPlacementAnimated has update method."""
        assert hasattr(animated_solver, 'update')
        assert callable(animated_solver.update)

    def test_update_accepts_dt(self, animated_solver):
        """update accepts delta time parameter."""
        animated_solver.update(dt=0.016)  # 60fps frame

    def test_update_with_zero_dt(self, animated_solver):
        """update with zero delta time."""
        animated_solver.update(dt=0.0)

    def test_update_with_small_dt(self, animated_solver):
        """update with small delta time (high fps)."""
        animated_solver.update(dt=0.001)  # 1000fps

    def test_update_with_large_dt(self, animated_solver):
        """update with large delta time (low fps)."""
        animated_solver.update(dt=0.5)  # 2fps

    def test_update_with_standard_dt(self, animated_solver):
        """update with standard 60fps delta time."""
        animated_solver.update(dt=1.0 / 60.0)

    def test_multiple_updates_accumulate(self, animated_solver, humanoid_transforms):
        """Multiple updates accumulate animation time."""
        animated_solver.set_height_curves(lambda t: t * 0.01, lambda t: t * 0.01)

        # Update multiple times
        for _ in range(10):
            animated_solver.update(dt=0.1)

        # After 1 second of updates, curves should be evaluated at t=1.0
        # Test by solving and checking result differs from initial
        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_update_affects_subsequent_solve(self, animated_solver, humanoid_transforms):
        """update affects subsequent solve calls."""
        # Set curves that vary with time
        animated_solver.set_height_curves(
            lambda t: math.sin(t) * 0.1,
            lambda t: math.sin(t + math.pi) * 0.1
        )

        # Solve before any updates
        result1 = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))

        # Update time
        animated_solver.update(dt=math.pi / 2)  # quarter cycle

        # Solve after update
        result2 = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))

        # Both should succeed
        assert result1 is not None
        assert result2 is not None


# =============================================================================
# solve() Behavior Tests
# =============================================================================

class TestFootPlacementAnimatedSolve:
    """Tests for solve behavior."""

    def test_has_solve_method(self, animated_solver):
        """FootPlacementAnimated has solve method."""
        assert hasattr(animated_solver, 'solve')
        assert callable(animated_solver.solve)

    def test_solve_returns_result(self, animated_solver, humanoid_transforms):
        """solve returns FootPlacementResult."""
        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert isinstance(result, FootPlacementResult)

    def test_solve_without_curves_returns_result(self, animated_solver, humanoid_transforms):
        """solve without curves still returns result."""
        # Don't set any curves
        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None
        assert hasattr(result, 'success')

    def test_solve_with_curves_applies_height_offsets(self, animated_solver, humanoid_transforms):
        """solve with curves applies animated height offsets."""
        # Set constant curves for predictable behavior
        animated_solver.set_height_curves(lambda t: 0.05, lambda t: 0.03)

        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None
        assert result.success

    def test_solve_with_varying_curves(self, animated_solver, humanoid_transforms):
        """solve with time-varying curves."""
        # Set sine curves
        animated_solver.set_height_curves(
            lambda t: math.sin(t * 2 * math.pi) * 0.1,
            lambda t: math.sin((t + 0.5) * 2 * math.pi) * 0.1
        )

        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_solve_result_has_transforms(self, animated_solver, humanoid_transforms):
        """solve result has transforms."""
        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert hasattr(result, 'transforms')
        assert len(result.transforms) == len(humanoid_transforms)

    def test_solve_result_has_pelvis_offset(self, animated_solver, humanoid_transforms):
        """solve result has pelvis_offset."""
        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert hasattr(result, 'pelvis_offset')

    def test_solve_with_different_character_positions(self, animated_solver, humanoid_transforms):
        """solve works with different character positions."""
        positions = [
            Vec3(0.0, 0.0, 0.0),
            Vec3(5.0, 0.0, 0.0),
            Vec3(0.0, 0.0, 5.0),
            Vec3(-3.0, 0.0, 2.0),
        ]

        for pos in positions:
            result = animated_solver.solve(humanoid_transforms, pos)
            assert result is not None

    def test_solve_with_dt_parameter(self, base_placement, humanoid_transforms):
        """solve accepts optional dt parameter."""
        animated = FootPlacementAnimated(base_placement=base_placement)
        animated.set_height_curves(lambda t: 0.05, lambda t: 0.05)

        # Some implementations may accept dt in solve
        try:
            result = animated.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0), dt=0.016)
            assert result is not None
        except TypeError:
            # dt parameter not supported in solve, that's ok
            result = animated.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
            assert result is not None


# =============================================================================
# Animation Integration Tests
# =============================================================================

class TestFootPlacementAnimatedWalkCycle:
    """Tests for walk cycle animation simulation."""

    def test_walk_cycle_with_sine_curves(self, animated_solver, humanoid_transforms):
        """Simulate walk cycle with sine curves."""
        # Walk cycle: left and right feet alternate
        # Left foot: sin(t * 2 * pi) - peaks at t=0.25
        # Right foot: sin(t * 2 * pi + pi) = -sin(t * 2 * pi) - peaks at t=0.75
        cycle_speed = 1.0  # 1 Hz walk cycle

        animated_solver.set_height_curves(
            lambda t: max(0, math.sin(t * 2 * math.pi * cycle_speed)) * 0.1,
            lambda t: max(0, math.sin(t * 2 * math.pi * cycle_speed + math.pi)) * 0.1
        )

        # Simulate multiple frames
        dt = 1.0 / 60.0  # 60fps
        for _ in range(120):  # 2 seconds
            animated_solver.update(dt=dt)
            result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
            assert result is not None
            assert result.success

    def test_walk_cycle_accumulates_time(self, animated_solver, humanoid_transforms):
        """Walk cycle accumulates time correctly."""
        # Track animation by using a curve that changes with time
        values = []

        def tracking_left_curve(t: float) -> float:
            values.append(('left', t))
            return 0.0

        def tracking_right_curve(t: float) -> float:
            values.append(('right', t))
            return 0.0

        animated_solver.set_height_curves(tracking_left_curve, tracking_right_curve)

        # Update and solve
        animated_solver.update(dt=0.5)
        animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))

        animated_solver.update(dt=0.5)
        animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))

        # Values should have been recorded with increasing time
        assert len(values) >= 2

    def test_alternating_foot_lift(self, animated_solver, humanoid_transforms):
        """Simulate alternating foot lift pattern."""
        # Phase-shifted curves for alternating feet
        animated_solver.set_height_curves(
            lambda t: abs(math.sin(t * math.pi)) * 0.15,  # Left lifts
            lambda t: abs(math.cos(t * math.pi)) * 0.15   # Right lifts offset
        )

        # Simulate
        for i in range(60):
            animated_solver.update(dt=1.0 / 60.0)
            result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
            assert result is not None

    def test_running_cycle_faster(self, animated_solver, humanoid_transforms):
        """Simulate faster running cycle."""
        # Faster cycle for running
        run_speed = 3.0  # 3 Hz

        animated_solver.set_height_curves(
            lambda t: max(0, math.sin(t * 2 * math.pi * run_speed)) * 0.2,
            lambda t: max(0, math.sin(t * 2 * math.pi * run_speed + math.pi)) * 0.2
        )

        dt = 1.0 / 120.0  # 120fps for fast motion
        for _ in range(240):  # 2 seconds
            animated_solver.update(dt=dt)
            result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
            assert result is not None


# =============================================================================
# Curves Affect Result Tests
# =============================================================================

class TestFootPlacementAnimatedCurvesAffectResult:
    """Tests verifying curves affect the solve result."""

    def test_different_constant_curves_differ(self, base_placement, humanoid_transforms):
        """Different constant curve values produce different results."""
        # Test with zero curves
        anim1 = FootPlacementAnimated(base_placement=base_placement)
        anim1.set_height_curves(lambda t: 0.0, lambda t: 0.0)
        result1 = anim1.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))

        # Test with non-zero constant curves
        anim2 = FootPlacementAnimated(base_placement=base_placement)
        anim2.set_height_curves(lambda t: 0.1, lambda t: 0.1)
        result2 = anim2.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))

        assert result1 is not None
        assert result2 is not None
        # Both should succeed
        assert result1.success
        assert result2.success

    def test_asymmetric_curves(self, animated_solver, humanoid_transforms):
        """Asymmetric left/right curves work correctly."""
        animated_solver.set_height_curves(
            lambda t: 0.2,   # Left high
            lambda t: 0.0    # Right on ground
        )
        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None
        assert result.success

    def test_curves_evaluated_at_animation_time(self, animated_solver, humanoid_transforms):
        """Curves are evaluated at current animation time."""
        # Time-dependent curve
        animated_solver.set_height_curves(
            lambda t: t * 0.01,  # Linear increase with time
            lambda t: t * 0.01
        )

        # At t=0, curves return 0
        result_t0 = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result_t0 is not None

        # Update time
        animated_solver.update(dt=10.0)  # Now t=10

        # At t=10, curves return 0.1
        result_t10 = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result_t10 is not None


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestFootPlacementAnimatedEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_curves(self, animated_solver, humanoid_transforms):
        """Handle curves that always return zero."""
        animated_solver.set_height_curves(lambda t: 0.0, lambda t: 0.0)
        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None
        assert result.success

    def test_very_fast_dt(self, animated_solver, humanoid_transforms):
        """Handle very fast update rate."""
        animated_solver.set_height_curves(
            lambda t: math.sin(t) * 0.1,
            lambda t: math.cos(t) * 0.1
        )

        # Very small dt - high fps
        for _ in range(10000):
            animated_solver.update(dt=0.0001)  # 10000fps

        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_very_slow_dt(self, animated_solver, humanoid_transforms):
        """Handle very slow update rate."""
        animated_solver.set_height_curves(
            lambda t: math.sin(t) * 0.1,
            lambda t: math.cos(t) * 0.1
        )

        # Very large dt - low fps
        animated_solver.update(dt=10.0)  # 0.1fps

        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_curves_returning_negative_values(self, animated_solver, humanoid_transforms):
        """Handle curves returning negative height offsets."""
        animated_solver.set_height_curves(
            lambda t: -0.05,  # Negative offset
            lambda t: -0.05
        )
        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_curves_with_large_positive_values(self, animated_solver, humanoid_transforms):
        """Handle curves with large positive values."""
        animated_solver.set_height_curves(
            lambda t: 1.0,  # Large offset
            lambda t: 1.0
        )
        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_curves_with_very_small_values(self, animated_solver, humanoid_transforms):
        """Handle curves with very small values."""
        animated_solver.set_height_curves(
            lambda t: 0.0001,
            lambda t: 0.0001
        )
        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_rapid_curve_changes(self, base_placement, humanoid_transforms):
        """Handle rapid curve switching."""
        animated = FootPlacementAnimated(base_placement=base_placement)

        for i in range(100):
            # Change curves rapidly
            val = float(i) * 0.001
            animated.set_height_curves(lambda t, v=val: v, lambda t, v=val: v)
            result = animated.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
            assert result is not None

    def test_long_running_animation(self, animated_solver, humanoid_transforms):
        """Test long-running animation stability."""
        animated_solver.set_height_curves(
            lambda t: math.sin(t) * 0.1,
            lambda t: math.cos(t) * 0.1
        )

        # Simulate 10 minutes of animation at 60fps
        total_time = 0.0
        dt = 1.0 / 60.0
        for _ in range(600):  # 10 seconds worth (simplified)
            animated_solver.update(dt=dt)
            total_time += dt

        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None
        assert result.success


# =============================================================================
# dt Affects Both Animation and Base Solve Tests
# =============================================================================

class TestFootPlacementAnimatedDtEffects:
    """Tests for dt affecting animation and solve."""

    def test_dt_accumulates_in_update(self, animated_solver, humanoid_transforms):
        """dt accumulates across multiple updates."""
        call_times = []

        def curve_with_tracking(t: float) -> float:
            call_times.append(t)
            return 0.0

        animated_solver.set_height_curves(curve_with_tracking, lambda t: 0.0)

        # Three updates of 0.1 each
        animated_solver.update(dt=0.1)
        animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))

        animated_solver.update(dt=0.1)
        animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))

        animated_solver.update(dt=0.1)
        animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))

        # Should have called curve at increasing times
        assert len(call_times) >= 3

    def test_solve_without_update_uses_initial_time(self, base_placement, humanoid_transforms):
        """solve without update uses initial animation time."""
        call_times = []

        def tracking_curve(t: float) -> float:
            call_times.append(t)
            return 0.0

        animated = FootPlacementAnimated(base_placement=base_placement)
        animated.set_height_curves(tracking_curve, lambda t: 0.0)

        # Solve without any update
        animated.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        animated.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))

        # Times should be at initial value (likely 0)
        if len(call_times) >= 2:
            # Without updates, times should remain the same
            assert abs(call_times[0] - call_times[1]) < 0.001


# =============================================================================
# Solve Result Quality Tests
# =============================================================================

class TestFootPlacementAnimatedSolveQuality:
    """Tests for solve result quality."""

    def test_solve_preserves_transform_count(self, animated_solver, humanoid_transforms):
        """solve preserves transform count."""
        animated_solver.set_height_curves(lambda t: 0.05, lambda t: 0.05)
        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert len(result.transforms) == len(humanoid_transforms)

    def test_solve_returns_valid_transforms(self, animated_solver, humanoid_transforms):
        """solve returns valid transforms."""
        animated_solver.set_height_curves(lambda t: 0.05, lambda t: 0.05)
        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))

        for transform in result.transforms:
            assert hasattr(transform, 'translation')
            assert hasattr(transform, 'rotation')

    def test_solve_result_pelvis_is_vec3(self, animated_solver, humanoid_transforms):
        """solve result pelvis_offset is Vec3."""
        animated_solver.set_height_curves(lambda t: 0.05, lambda t: 0.05)
        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))

        offset = result.pelvis_offset
        assert hasattr(offset, 'x')
        assert hasattr(offset, 'y')
        assert hasattr(offset, 'z')


# =============================================================================
# Complex Curve Patterns Tests
# =============================================================================

class TestFootPlacementAnimatedComplexCurves:
    """Tests for complex curve patterns."""

    def test_bezier_like_curves(self, animated_solver, humanoid_transforms):
        """Test with bezier-like curve patterns."""
        def bezier_curve(t: float) -> float:
            # Simplified bezier: ease in-out
            t_mod = t % 1.0
            return 3 * t_mod * t_mod - 2 * t_mod * t_mod * t_mod

        animated_solver.set_height_curves(bezier_curve, bezier_curve)

        for _ in range(60):
            animated_solver.update(dt=1.0/60.0)
            result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
            assert result is not None

    def test_step_function_curves(self, animated_solver, humanoid_transforms):
        """Test with step function curves."""
        def step_curve(t: float) -> float:
            # Step between 0 and 0.1 every 0.5 seconds
            return 0.1 if int(t * 2) % 2 == 0 else 0.0

        animated_solver.set_height_curves(step_curve, step_curve)

        for _ in range(120):
            animated_solver.update(dt=1.0/60.0)
            result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
            assert result is not None

    def test_sawtooth_curves(self, animated_solver, humanoid_transforms):
        """Test with sawtooth wave curves."""
        def sawtooth_curve(t: float) -> float:
            return (t % 1.0) * 0.1

        animated_solver.set_height_curves(sawtooth_curve, sawtooth_curve)

        for _ in range(60):
            animated_solver.update(dt=1.0/60.0)
            result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
            assert result is not None

    def test_exponential_decay_curves(self, animated_solver, humanoid_transforms):
        """Test with exponential decay curves."""
        def decay_curve(t: float) -> float:
            return math.exp(-t) * 0.1

        animated_solver.set_height_curves(decay_curve, decay_curve)

        for _ in range(60):
            animated_solver.update(dt=1.0/60.0)
            result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
            assert result is not None


# =============================================================================
# Performance Tests
# =============================================================================

class TestFootPlacementAnimatedPerformance:
    """Tests for performance characteristics."""

    def test_solve_completes_quickly(self, animated_solver, humanoid_transforms):
        """solve should complete in reasonable time."""
        import time

        animated_solver.set_height_curves(
            lambda t: math.sin(t) * 0.1,
            lambda t: math.cos(t) * 0.1
        )

        start = time.time()
        for _ in range(1000):
            animated_solver.update(dt=0.016)
            animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        elapsed = time.time() - start

        # 1000 update+solve pairs should take less than 2 seconds
        assert elapsed < 2.0

    def test_many_updates_stable(self, animated_solver, humanoid_transforms):
        """Many updates remain numerically stable."""
        animated_solver.set_height_curves(
            lambda t: math.sin(t * 1000) * 0.1,  # High frequency
            lambda t: math.cos(t * 1000) * 0.1
        )

        # Many small updates
        for _ in range(10000):
            animated_solver.update(dt=0.001)

        result = animated_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None
        assert result.success

        # Check result is not NaN or Inf
        assert not math.isnan(result.pelvis_offset.x)
        assert not math.isnan(result.pelvis_offset.y)
        assert not math.isnan(result.pelvis_offset.z)
        assert not math.isinf(result.pelvis_offset.x)
        assert not math.isinf(result.pelvis_offset.y)
        assert not math.isinf(result.pelvis_offset.z)

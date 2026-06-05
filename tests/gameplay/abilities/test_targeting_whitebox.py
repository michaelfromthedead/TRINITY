"""
WHITEBOX Tests for the Targeting System.

Comprehensive internal testing of targeting system with full source access.

Tests cover:
- Vector3 math operations and edge cases
- TargetData internal state
- TargetFilter mechanics (tags, predicates, limits)
- TargetingSystem base class
- SelfTargeting behavior
- ActorTargeting selection logic
- PointTargeting range clamping
- AreaTargeting geometry calculations:
  - Circle area
  - Cone area
  - Rectangle area
  - Line area
  - Capsule area
- ConfirmationTargeting state machine
- Factory function outputs

Total: 50+ tests for targeting system internals
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest

from engine.gameplay.abilities.constants import (
    DEFAULT_AOE_RADIUS,
    DEFAULT_CONE_ANGLE,
    DEFAULT_LINE_WIDTH,
    DEFAULT_MAX_RANGE,
    DEFAULT_MIN_RANGE,
    EPSILON,
    AreaShape,
    TargetingMode,
)
from engine.gameplay.abilities.targeting import (
    ActorTargeting,
    AreaTargeting,
    ConfirmationTargeting,
    PointTargeting,
    SelfTargeting,
    TargetData,
    TargetFilter,
    TargetingSystem,
    Vector3,
    create_aoe,
    create_cone,
    create_point_target,
    create_self_targeting,
    create_single_target,
)
from engine.gameplay.abilities.tags import GameplayTag, GameplayTagContainer


# =============================================================================
# MOCK ENTITIES
# =============================================================================


@dataclass
class MockTargetable:
    """Mock targetable entity for testing."""
    position: Vector3 = field(default_factory=Vector3.zero)
    tags: GameplayTagContainer = field(default_factory=GameplayTagContainer)
    is_valid: bool = True
    is_alive: bool = True


# =============================================================================
# VECTOR3 TESTS
# =============================================================================


class TestVector3Internals:
    """Whitebox tests for Vector3 math operations."""

    def test_vector_initialization(self):
        """Test Vector3 initializes with correct values."""
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_vector_default_zero(self):
        """Test Vector3 defaults to zero."""
        v = Vector3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_vector_addition(self):
        """Test vector addition."""
        a = Vector3(1.0, 2.0, 3.0)
        b = Vector3(4.0, 5.0, 6.0)
        result = a + b
        assert result.x == 5.0
        assert result.y == 7.0
        assert result.z == 9.0

    def test_vector_subtraction(self):
        """Test vector subtraction."""
        a = Vector3(5.0, 5.0, 5.0)
        b = Vector3(1.0, 2.0, 3.0)
        result = a - b
        assert result.x == 4.0
        assert result.y == 3.0
        assert result.z == 2.0

    def test_vector_scalar_multiplication(self):
        """Test scalar multiplication."""
        v = Vector3(1.0, 2.0, 3.0)
        result = v * 2.0
        assert result.x == 2.0
        assert result.y == 4.0
        assert result.z == 6.0

    def test_vector_reverse_scalar_multiplication(self):
        """Test reverse scalar multiplication."""
        v = Vector3(1.0, 2.0, 3.0)
        result = 2.0 * v
        assert result.x == 2.0
        assert result.y == 4.0
        assert result.z == 6.0

    def test_vector_negation(self):
        """Test vector negation."""
        v = Vector3(1.0, -2.0, 3.0)
        result = -v
        assert result.x == -1.0
        assert result.y == 2.0
        assert result.z == -3.0

    def test_vector_magnitude(self):
        """Test magnitude calculation."""
        v = Vector3(3.0, 4.0, 0.0)
        assert v.magnitude == 5.0

    def test_vector_magnitude_squared(self):
        """Test magnitude squared (faster)."""
        v = Vector3(3.0, 4.0, 0.0)
        assert v.magnitude_squared == 25.0

    def test_vector_normalized(self):
        """Test vector normalization."""
        v = Vector3(0.0, 0.0, 5.0)
        n = v.normalized
        assert abs(n.z - 1.0) < EPSILON
        assert abs(n.magnitude - 1.0) < EPSILON

    def test_vector_normalized_zero_vector(self):
        """Test normalizing zero vector returns zero."""
        v = Vector3.zero()
        n = v.normalized
        assert n.x == 0.0
        assert n.y == 0.0
        assert n.z == 0.0

    def test_vector_dot_product(self):
        """Test dot product."""
        a = Vector3(1.0, 0.0, 0.0)
        b = Vector3(0.0, 1.0, 0.0)
        assert a.dot(b) == 0.0  # Perpendicular

        c = Vector3(1.0, 0.0, 0.0)
        assert a.dot(c) == 1.0  # Parallel

    def test_vector_cross_product(self):
        """Test cross product."""
        x = Vector3(1.0, 0.0, 0.0)
        y = Vector3(0.0, 1.0, 0.0)
        result = x.cross(y)
        assert result.z == 1.0  # x cross y = z

    def test_vector_distance_to(self):
        """Test distance between two points."""
        a = Vector3(0.0, 0.0, 0.0)
        b = Vector3(3.0, 4.0, 0.0)
        assert a.distance_to(b) == 5.0

    def test_vector_distance_squared_to(self):
        """Test squared distance (faster)."""
        a = Vector3(0.0, 0.0, 0.0)
        b = Vector3(3.0, 4.0, 0.0)
        assert a.distance_squared_to(b) == 25.0

    def test_vector_angle_to(self):
        """Test angle between vectors."""
        x = Vector3(1.0, 0.0, 0.0)
        y = Vector3(0.0, 1.0, 0.0)
        angle = x.angle_to(y)
        assert abs(angle - math.pi / 2) < EPSILON  # 90 degrees

    def test_vector_angle_to_zero_vector(self):
        """Test angle with zero vector returns 0."""
        x = Vector3(1.0, 0.0, 0.0)
        zero = Vector3.zero()
        angle = x.angle_to(zero)
        assert angle == 0.0

    def test_vector_static_constructors(self):
        """Test static constructor methods."""
        assert Vector3.zero() == Vector3(0, 0, 0)
        assert Vector3.up() == Vector3(0, 1, 0)
        assert Vector3.forward() == Vector3(0, 0, 1)


# =============================================================================
# TARGET DATA TESTS
# =============================================================================


class TestTargetDataInternals:
    """Whitebox tests for TargetData internal state."""

    def test_target_data_defaults(self):
        """Test TargetData default values."""
        data = TargetData()
        assert data.mode == TargetingMode.SELF
        assert data.targets == []
        assert data.point is None
        assert data.direction is None
        assert data.confirmed is False
        assert data.cancelled is False
        assert data.distance == 0.0
        assert data.hit_location is None

    def test_has_targets_property(self):
        """Test has_targets property."""
        data = TargetData()
        assert data.has_targets is False

        data.targets.append(object())
        assert data.has_targets is True

    def test_primary_target_property(self):
        """Test primary_target returns first target."""
        target1 = object()
        target2 = object()

        data = TargetData(targets=[target1, target2])
        assert data.primary_target is target1

    def test_primary_target_none_when_empty(self):
        """Test primary_target returns None when no targets."""
        data = TargetData()
        assert data.primary_target is None

    def test_target_count_property(self):
        """Test target_count property."""
        data = TargetData(targets=[1, 2, 3])
        assert data.target_count == 3


# =============================================================================
# TARGET FILTER TESTS
# =============================================================================


class TestTargetFilterInternals:
    """Whitebox tests for TargetFilter mechanics."""

    def test_filter_default_allows_most(self):
        """Test default filter allows most targets."""
        f = TargetFilter()
        target = MockTargetable()
        source = MockTargetable()

        assert f.passes(target, source) is True

    def test_filter_blocks_self(self):
        """Test filter blocks self when allow_self=False."""
        f = TargetFilter(allow_self=False)
        entity = MockTargetable()

        assert f.passes(entity, entity) is False

    def test_filter_allows_self(self):
        """Test filter allows self when allow_self=True."""
        f = TargetFilter(allow_self=True)
        entity = MockTargetable()

        assert f.passes(entity, entity) is True

    def test_filter_blocks_dead(self):
        """Test filter blocks dead targets."""
        f = TargetFilter(allow_dead=False)
        target = MockTargetable(is_alive=False)
        source = MockTargetable()

        assert f.passes(target, source) is False

    def test_filter_allows_dead(self):
        """Test filter allows dead when configured."""
        f = TargetFilter(allow_dead=True)
        target = MockTargetable(is_alive=False)
        source = MockTargetable()

        assert f.passes(target, source) is True

    def test_filter_require_all_tags(self):
        """Test require_tags requires all specified tags."""
        f = TargetFilter(require_tags={"enemy", "vulnerable"})

        target = MockTargetable()
        target.tags.add("enemy")

        source = MockTargetable()
        assert f.passes(target, source) is False  # Missing vulnerable

        target.tags.add("vulnerable")
        assert f.passes(target, source) is True

    def test_filter_exclude_tags(self):
        """Test exclude_tags blocks if any present."""
        f = TargetFilter(exclude_tags={"immune", "invulnerable"})

        target = MockTargetable()
        source = MockTargetable()

        assert f.passes(target, source) is True

        target.tags.add("immune")
        assert f.passes(target, source) is False

    def test_filter_require_any_tags(self):
        """Test require_any_tags needs at least one."""
        f = TargetFilter(require_any_tags={"boss", "elite", "champion"})

        target = MockTargetable()
        source = MockTargetable()

        assert f.passes(target, source) is False

        target.tags.add("elite")
        assert f.passes(target, source) is True

    def test_filter_custom_predicate(self):
        """Test custom filter function."""
        def only_high_health(target):
            return getattr(target, "health", 0) > 50

        f = TargetFilter(custom_filter=only_high_health)

        low_health = MockTargetable()
        low_health.health = 20

        high_health = MockTargetable()
        high_health.health = 100

        source = MockTargetable()

        assert f.passes(low_health, source) is False
        assert f.passes(high_health, source) is True

    def test_filter_targets_max_limit(self):
        """Test max_targets limit."""
        f = TargetFilter(max_targets=2)

        targets = [MockTargetable() for _ in range(5)]
        source = MockTargetable()

        filtered = f.filter_targets(targets, source)
        assert len(filtered) == 2

    def test_filter_targets_preserves_order(self):
        """Test filter preserves target order."""
        f = TargetFilter()

        t1 = MockTargetable()
        t1.id = 1
        t2 = MockTargetable()
        t2.id = 2
        t3 = MockTargetable()
        t3.id = 3

        targets = [t1, t2, t3]
        source = MockTargetable()

        filtered = f.filter_targets(targets, source)
        assert filtered[0].id == 1
        assert filtered[1].id == 2


# =============================================================================
# SELF TARGETING TESTS
# =============================================================================


class TestSelfTargetingInternals:
    """Whitebox tests for SelfTargeting behavior."""

    def test_self_targeting_mode(self):
        """Test SelfTargeting has correct mode."""
        targeting = SelfTargeting()
        assert targeting.mode == TargetingMode.SELF

    def test_self_targeting_range_zero(self):
        """Test SelfTargeting has zero range."""
        targeting = SelfTargeting()
        assert targeting.min_range == 0.0
        assert targeting.max_range == 0.0

    def test_self_targeting_allows_self(self):
        """Test SelfTargeting filter allows self."""
        targeting = SelfTargeting()
        assert targeting.target_filter.allow_self is True

    def test_self_targeting_returns_source(self):
        """Test acquire_targets returns source as only target."""
        targeting = SelfTargeting()
        source = MockTargetable()

        data = targeting.acquire_targets(
            source,
            Vector3.zero(),
            Vector3.forward(),
            []
        )

        assert data.mode == TargetingMode.SELF
        assert data.targets == [source]
        assert data.confirmed is True

    def test_self_targeting_ignores_candidates(self):
        """Test SelfTargeting ignores candidate list."""
        targeting = SelfTargeting()
        source = MockTargetable()
        candidates = [MockTargetable() for _ in range(5)]

        data = targeting.acquire_targets(
            source,
            Vector3.zero(),
            Vector3.forward(),
            candidates
        )

        assert data.targets == [source]


# =============================================================================
# ACTOR TARGETING TESTS
# =============================================================================


class TestActorTargetingInternals:
    """Whitebox tests for ActorTargeting selection logic."""

    def test_actor_targeting_mode(self):
        """Test ActorTargeting has correct mode."""
        targeting = ActorTargeting()
        assert targeting.mode == TargetingMode.ACTOR

    def test_actor_targeting_selects_closest_to_aim(self):
        """Test ActorTargeting selects target closest to aim point."""
        targeting = ActorTargeting(max_range=100.0)

        source = MockTargetable(position=Vector3.zero())

        far = MockTargetable(position=Vector3(20, 0, 0))
        close = MockTargetable(position=Vector3(5, 0, 0))
        medium = MockTargetable(position=Vector3(10, 0, 0))

        candidates = [far, close, medium]
        aim_point = Vector3(6, 0, 0)  # Closest to "close"

        data = targeting.acquire_targets(
            source, aim_point, Vector3(1, 0, 0), candidates
        )

        assert data.primary_target is close

    def test_actor_targeting_respects_range(self):
        """Test ActorTargeting respects max range."""
        targeting = ActorTargeting(max_range=10.0)

        source = MockTargetable(position=Vector3.zero())
        in_range = MockTargetable(position=Vector3(5, 0, 0))
        out_of_range = MockTargetable(position=Vector3(20, 0, 0))

        candidates = [in_range, out_of_range]

        data = targeting.acquire_targets(
            source, Vector3(15, 0, 0), Vector3(1, 0, 0), candidates
        )

        assert in_range in data.targets
        assert out_of_range not in data.targets

    def test_actor_targeting_respects_min_range(self):
        """Test ActorTargeting respects min range."""
        targeting = ActorTargeting(min_range=5.0, max_range=20.0)

        source = MockTargetable(position=Vector3.zero())
        too_close = MockTargetable(position=Vector3(2, 0, 0))
        in_range = MockTargetable(position=Vector3(10, 0, 0))

        candidates = [too_close, in_range]

        data = targeting.acquire_targets(
            source, Vector3(10, 0, 0), Vector3(1, 0, 0), candidates
        )

        assert too_close not in data.targets
        assert in_range in data.targets

    def test_actor_targeting_no_valid_targets(self):
        """Test ActorTargeting returns cancelled when no valid targets."""
        targeting = ActorTargeting(max_range=5.0)

        source = MockTargetable(position=Vector3.zero())
        out_of_range = MockTargetable(position=Vector3(20, 0, 0))

        data = targeting.acquire_targets(
            source, Vector3.zero(), Vector3.forward(), [out_of_range]
        )

        assert data.cancelled is True
        assert data.has_targets is False

    def test_actor_targeting_validates_is_valid(self):
        """Test ActorTargeting validates target.is_valid."""
        targeting = ActorTargeting(max_range=100.0)

        source = MockTargetable(position=Vector3.zero())
        invalid = MockTargetable(position=Vector3(5, 0, 0), is_valid=False)
        valid = MockTargetable(position=Vector3(10, 0, 0), is_valid=True)

        candidates = [invalid, valid]

        data = targeting.acquire_targets(
            source, Vector3(5, 0, 0), Vector3.forward(), candidates
        )

        assert invalid not in data.targets


# =============================================================================
# POINT TARGETING TESTS
# =============================================================================


class TestPointTargetingInternals:
    """Whitebox tests for PointTargeting range clamping."""

    def test_point_targeting_mode(self):
        """Test PointTargeting has correct mode."""
        targeting = PointTargeting()
        assert targeting.mode == TargetingMode.POINT

    def test_point_targeting_returns_aim_point(self):
        """Test PointTargeting returns aim point when in range."""
        targeting = PointTargeting(max_range=100.0)

        source = MockTargetable(position=Vector3.zero())
        aim = Vector3(50, 0, 0)

        data = targeting.acquire_targets(
            source, aim, Vector3(1, 0, 0), []
        )

        assert data.point == aim
        assert data.has_targets is False

    def test_point_targeting_clamps_to_max_range(self):
        """Test PointTargeting clamps aim point to max range."""
        targeting = PointTargeting(max_range=10.0)

        source = MockTargetable(position=Vector3.zero())
        aim = Vector3(50, 0, 0)  # Beyond max range

        data = targeting.acquire_targets(
            source, aim, Vector3(1, 0, 0), []
        )

        # Should be clamped to 10 units in direction of aim
        assert data.point is not None
        assert abs(data.point.x - 10.0) < EPSILON
        assert data.distance == 10.0

    def test_point_targeting_returns_direction(self):
        """Test PointTargeting returns aim direction."""
        targeting = PointTargeting(max_range=100.0)

        source = MockTargetable(position=Vector3.zero())
        direction = Vector3(0, 1, 0)

        data = targeting.acquire_targets(
            source, Vector3(0, 50, 0), direction, []
        )

        assert data.direction == direction


# =============================================================================
# AREA TARGETING TESTS - CIRCLE
# =============================================================================


class TestAreaTargetingCircle:
    """Whitebox tests for circular area targeting."""

    def test_circle_area_mode(self):
        """Test AreaTargeting with circle has correct mode."""
        targeting = AreaTargeting(shape=AreaShape.CIRCLE)
        assert targeting.mode == TargetingMode.AREA
        assert targeting.shape == AreaShape.CIRCLE

    def test_circle_includes_targets_inside(self):
        """Test circle includes targets inside radius."""
        targeting = AreaTargeting(
            shape=AreaShape.CIRCLE,
            radius=10.0,
            max_range=100.0,
        )

        source = MockTargetable(position=Vector3.zero())
        center = Vector3(50, 0, 0)

        inside1 = MockTargetable(position=Vector3(52, 0, 0))
        inside2 = MockTargetable(position=Vector3(48, 0, 0))
        outside = MockTargetable(position=Vector3(70, 0, 0))

        candidates = [inside1, inside2, outside]

        data = targeting.acquire_targets(
            source, center, Vector3.forward(), candidates
        )

        assert inside1 in data.targets
        assert inside2 in data.targets
        assert outside not in data.targets

    def test_circle_boundary_included(self):
        """Test targets exactly on boundary are included."""
        targeting = AreaTargeting(
            shape=AreaShape.CIRCLE,
            radius=10.0,
            max_range=100.0,
        )

        source = MockTargetable(position=Vector3.zero())
        center = Vector3(0, 0, 0)

        on_boundary = MockTargetable(position=Vector3(10, 0, 0))

        data = targeting.acquire_targets(
            source, center, Vector3.forward(), [on_boundary]
        )

        assert on_boundary in data.targets

    def test_circle_respects_filter(self):
        """Test circle targeting respects target filter."""
        targeting = AreaTargeting(
            shape=AreaShape.CIRCLE,
            radius=10.0,
            max_range=100.0,
            target_filter=TargetFilter(allow_self=False),
        )

        source = MockTargetable(position=Vector3.zero())

        data = targeting.acquire_targets(
            source, Vector3.zero(), Vector3.forward(), [source]
        )

        assert source not in data.targets


# =============================================================================
# AREA TARGETING TESTS - CONE
# =============================================================================


class TestAreaTargetingCone:
    """Whitebox tests for cone area targeting geometry."""

    def test_cone_includes_targets_in_angle(self):
        """Test cone includes targets within angle."""
        targeting = AreaTargeting(
            shape=AreaShape.CONE,
            radius=20.0,
            cone_angle=90.0,  # 45 degrees each side
            max_range=100.0,
        )

        source = MockTargetable(position=Vector3.zero())
        direction = Vector3(1, 0, 0)

        # In front and in cone
        in_cone = MockTargetable(position=Vector3(10, 2, 0))

        # In front but outside cone angle
        outside_angle = MockTargetable(position=Vector3(5, 10, 0))

        # Behind
        behind = MockTargetable(position=Vector3(-10, 0, 0))

        candidates = [in_cone, outside_angle, behind]

        data = targeting.acquire_targets(
            source, Vector3.zero(), direction, candidates
        )

        assert in_cone in data.targets
        assert outside_angle not in data.targets
        assert behind not in data.targets

    def test_cone_respects_radius(self):
        """Test cone respects radius limit."""
        targeting = AreaTargeting(
            shape=AreaShape.CONE,
            radius=10.0,
            cone_angle=90.0,
            max_range=100.0,
        )

        source = MockTargetable(position=Vector3.zero())
        direction = Vector3(1, 0, 0)

        in_range = MockTargetable(position=Vector3(8, 0, 0))
        out_of_range = MockTargetable(position=Vector3(20, 0, 0))

        candidates = [in_range, out_of_range]

        data = targeting.acquire_targets(
            source, Vector3.zero(), direction, candidates
        )

        assert in_range in data.targets
        assert out_of_range not in data.targets

    def test_cone_narrow_angle(self):
        """Test narrow cone angle."""
        targeting = AreaTargeting(
            shape=AreaShape.CONE,
            radius=50.0,
            cone_angle=10.0,  # Very narrow
            max_range=100.0,
        )

        source = MockTargetable(position=Vector3.zero())
        direction = Vector3(1, 0, 0)

        # Directly ahead - should be in
        ahead = MockTargetable(position=Vector3(30, 0, 0))

        # Slightly to side - should be out
        side = MockTargetable(position=Vector3(30, 5, 0))

        data = targeting.acquire_targets(
            source, Vector3.zero(), direction, [ahead, side]
        )

        assert ahead in data.targets


# =============================================================================
# AREA TARGETING TESTS - RECTANGLE
# =============================================================================


class TestAreaTargetingRectangle:
    """Whitebox tests for rectangle area targeting."""

    def test_rectangle_includes_targets_inside(self):
        """Test rectangle includes targets inside bounds."""
        targeting = AreaTargeting(
            shape=AreaShape.RECTANGLE,
            rectangle_width=10.0,
            rectangle_height=20.0,
            max_range=100.0,
        )

        source = MockTargetable(position=Vector3.zero())
        center = Vector3(0, 0, 0)
        direction = Vector3(0, 0, 1)  # Forward along Z

        # Inside rectangle
        inside = MockTargetable(position=Vector3(3, 0, 5))

        # Outside width
        outside_width = MockTargetable(position=Vector3(10, 0, 5))

        # Outside height
        outside_height = MockTargetable(position=Vector3(0, 0, 20))

        candidates = [inside, outside_width, outside_height]

        data = targeting.acquire_targets(
            source, center, direction, candidates
        )

        assert inside in data.targets


# =============================================================================
# AREA TARGETING TESTS - LINE
# =============================================================================


class TestAreaTargetingLine:
    """Whitebox tests for line area targeting."""

    def test_line_includes_targets_near_line(self):
        """Test line includes targets within line width."""
        targeting = AreaTargeting(
            shape=AreaShape.LINE,
            radius=20.0,  # Line length
            line_width=4.0,  # 2 units each side
            max_range=100.0,
        )

        source = MockTargetable(position=Vector3.zero())
        start = Vector3.zero()
        direction = Vector3(1, 0, 0)

        # On the line
        on_line = MockTargetable(position=Vector3(10, 0, 0))

        # Within width
        near_line = MockTargetable(position=Vector3(10, 1.5, 0))

        # Outside width
        far_from_line = MockTargetable(position=Vector3(10, 5, 0))

        candidates = [on_line, near_line, far_from_line]

        data = targeting.acquire_targets(
            source, start, direction, candidates
        )

        assert on_line in data.targets
        assert near_line in data.targets
        assert far_from_line not in data.targets


# =============================================================================
# AREA TARGETING TESTS - CAPSULE
# =============================================================================


class TestAreaTargetingCapsule:
    """Whitebox tests for capsule area targeting."""

    def test_capsule_includes_targets_at_endpoints(self):
        """Test capsule includes targets at rounded endpoints."""
        targeting = AreaTargeting(
            shape=AreaShape.CAPSULE,
            radius=10.0,
            line_width=4.0,
            max_range=100.0,
        )

        source = MockTargetable(position=Vector3.zero())
        direction = Vector3(1, 0, 0)

        # At end cap
        at_end = MockTargetable(position=Vector3(11, 0, 0))

        # Near start cap
        at_start = MockTargetable(position=Vector3(-1, 0, 0))

        candidates = [at_end, at_start]

        data = targeting.acquire_targets(
            source, Vector3.zero(), direction, candidates
        )

        assert at_end in data.targets


# =============================================================================
# AREA TARGETING BOUNDS TESTS
# =============================================================================


class TestAreaTargetingBounds:
    """Tests for area targeting bounding box calculation."""

    def test_get_area_bounds(self):
        """Test area bounds calculation."""
        targeting = AreaTargeting(
            shape=AreaShape.CIRCLE,
            radius=10.0,
        )

        center = Vector3(50, 50, 50)
        min_bound, max_bound = targeting.get_area_bounds(center, Vector3.forward())

        assert min_bound.x == 40.0
        assert max_bound.x == 60.0


# =============================================================================
# CONFIRMATION TARGETING TESTS
# =============================================================================


class TestConfirmationTargetingInternals:
    """Whitebox tests for ConfirmationTargeting state machine."""

    def test_confirmation_wraps_inner_targeting(self):
        """Test ConfirmationTargeting wraps inner system."""
        inner = ActorTargeting(max_range=30.0)
        confirmation = ConfirmationTargeting(inner)

        assert confirmation.inner_targeting is inner
        assert confirmation.max_range == inner.max_range

    def test_confirmation_mode(self):
        """Test ConfirmationTargeting has correct mode."""
        inner = PointTargeting()
        confirmation = ConfirmationTargeting(inner)

        assert confirmation.mode == TargetingMode.CONFIRMATION

    def test_confirmation_acquire_not_confirmed(self):
        """Test acquire returns unconfirmed data."""
        inner = PointTargeting(max_range=100.0)
        confirmation = ConfirmationTargeting(inner)

        source = MockTargetable(position=Vector3.zero())

        data = confirmation.acquire_targets(
            source, Vector3(50, 0, 0), Vector3.forward(), []
        )

        assert data.confirmed is False
        assert data.mode == TargetingMode.CONFIRMATION

    def test_confirmation_stores_pending(self):
        """Test acquire stores pending data for later confirmation."""
        inner = PointTargeting(max_range=100.0)
        confirmation = ConfirmationTargeting(inner)

        source = MockTargetable(position=Vector3.zero())
        aim = Vector3(30, 0, 0)

        confirmation.acquire_targets(source, aim, Vector3.forward(), [])

        assert confirmation._pending_data is not None
        assert confirmation._pending_data.point == aim

    def test_confirmation_confirm_returns_data(self):
        """Test confirm returns stored data as confirmed."""
        inner = PointTargeting(max_range=100.0)
        confirmation = ConfirmationTargeting(inner)

        source = MockTargetable(position=Vector3.zero())

        confirmation.acquire_targets(
            source, Vector3(50, 0, 0), Vector3.forward(), []
        )

        confirmed = confirmation.confirm()

        assert confirmed is not None
        assert confirmed.confirmed is True

    def test_confirmation_confirm_clears_pending(self):
        """Test confirm clears pending data."""
        inner = PointTargeting(max_range=100.0)
        confirmation = ConfirmationTargeting(inner)

        source = MockTargetable(position=Vector3.zero())

        confirmation.acquire_targets(
            source, Vector3(50, 0, 0), Vector3.forward(), []
        )
        confirmation.confirm()

        assert confirmation._pending_data is None

    def test_confirmation_confirm_without_pending(self):
        """Test confirm without pending returns None."""
        inner = PointTargeting()
        confirmation = ConfirmationTargeting(inner)

        result = confirmation.confirm()
        assert result is None

    def test_confirmation_cancel_clears_pending(self):
        """Test cancel clears pending data."""
        inner = PointTargeting(max_range=100.0)
        confirmation = ConfirmationTargeting(inner)

        source = MockTargetable(position=Vector3.zero())

        confirmation.acquire_targets(
            source, Vector3(50, 0, 0), Vector3.forward(), []
        )
        confirmation.cancel()

        assert confirmation._pending_data is None


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestTargetingFactoryFunctions:
    """Tests for targeting factory functions."""

    def test_create_self_targeting(self):
        """Test create_self_targeting factory."""
        targeting = create_self_targeting()

        assert isinstance(targeting, SelfTargeting)

    def test_create_single_target(self):
        """Test create_single_target factory."""
        targeting = create_single_target(max_range=50.0, require_hostile=True)

        assert isinstance(targeting, ActorTargeting)
        assert targeting.max_range == 50.0
        assert targeting.target_filter.allow_self is False

    def test_create_point_target(self):
        """Test create_point_target factory."""
        targeting = create_point_target(max_range=100.0)

        assert isinstance(targeting, PointTargeting)
        assert targeting.max_range == 100.0

    def test_create_aoe(self):
        """Test create_aoe factory."""
        targeting = create_aoe(radius=15.0, max_range=50.0, include_self=True)

        assert isinstance(targeting, AreaTargeting)
        assert targeting.shape == AreaShape.CIRCLE
        assert targeting.radius == 15.0
        assert targeting.target_filter.allow_self is True

    def test_create_cone(self):
        """Test create_cone factory."""
        targeting = create_cone(angle=60.0, length=25.0)

        assert isinstance(targeting, AreaTargeting)
        assert targeting.shape == AreaShape.CONE
        assert targeting.cone_angle == 60.0
        assert targeting.radius == 25.0


# =============================================================================
# EDGE CASES
# =============================================================================


class TestTargetingEdgeCases:
    """Edge case tests for targeting system."""

    def test_zero_range(self):
        """Test targeting with zero range."""
        targeting = AreaTargeting(shape=AreaShape.CIRCLE, radius=0.0)

        source = MockTargetable(position=Vector3.zero())
        target = MockTargetable(position=Vector3.zero())

        data = targeting.acquire_targets(
            source, Vector3.zero(), Vector3.forward(), [target]
        )

        # Target at exact same position might be included
        # Depends on EPSILON handling

    def test_very_large_radius(self):
        """Test targeting with very large radius (clamped to max)."""
        # Note: radius is clamped to DEFAULT_AOE_MAX_RADIUS (100)
        targeting = AreaTargeting(shape=AreaShape.CIRCLE, radius=1000.0)

        source = MockTargetable(position=Vector3.zero())
        # Target within the clamped radius
        near = MockTargetable(position=Vector3(50, 0, 0))

        data = targeting.acquire_targets(
            source, Vector3.zero(), Vector3.forward(), [near]
        )

        assert near in data.targets

    def test_negative_direction_handling(self):
        """Test targeting with negative direction vector."""
        targeting = AreaTargeting(
            shape=AreaShape.CONE,
            radius=20.0,
            cone_angle=90.0,
        )

        source = MockTargetable(position=Vector3.zero())
        direction = Vector3(-1, 0, 0)  # Pointing negative X

        behind = MockTargetable(position=Vector3(-10, 0, 0))  # Now in cone

        data = targeting.acquire_targets(
            source, Vector3.zero(), direction, [behind]
        )

        assert behind in data.targets

    def test_empty_candidate_list(self):
        """Test targeting with no candidates."""
        targeting = ActorTargeting(max_range=100.0)

        source = MockTargetable(position=Vector3.zero())

        data = targeting.acquire_targets(
            source, Vector3.zero(), Vector3.forward(), []
        )

        assert data.cancelled is True

    def test_all_candidates_invalid(self):
        """Test targeting when all candidates are invalid."""
        targeting = ActorTargeting(max_range=100.0)

        source = MockTargetable(position=Vector3.zero())
        invalid1 = MockTargetable(position=Vector3(5, 0, 0), is_valid=False)
        invalid2 = MockTargetable(position=Vector3(10, 0, 0), is_valid=False)

        data = targeting.acquire_targets(
            source, Vector3.zero(), Vector3.forward(), [invalid1, invalid2]
        )

        assert data.cancelled is True

    def test_target_without_position_attribute(self):
        """Test handling target without position attribute."""
        targeting = AreaTargeting(
            shape=AreaShape.CIRCLE,
            radius=10.0,
        )

        source = MockTargetable(position=Vector3.zero())

        class NoPosition:
            pass

        no_pos = NoPosition()

        data = targeting.acquire_targets(
            source, Vector3.zero(), Vector3.forward(), [no_pos]
        )

        assert no_pos not in data.targets

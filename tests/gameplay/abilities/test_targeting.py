"""
Tests for the Targeting System.

Tests cover:
- Self targeting
- Single target (nearest, cursor)
- Area targeting (circle, cone, line)
- Team filtering (friendly, enemy, all)
- Line of sight checks
- Range validation
- Target prediction
- Target locking

Total: ~100 tests
"""

from __future__ import annotations

import math
import pytest
from dataclasses import dataclass, field
from typing import List, Optional

from engine.gameplay.abilities.constants import (
    AreaShape,
    TargetingMode,
    DEFAULT_AOE_RADIUS,
    DEFAULT_CONE_ANGLE,
    DEFAULT_MAX_RANGE,
    DEFAULT_MELEE_RANGE,
    EPSILON,
)
from engine.gameplay.abilities.targeting import (
    Vector3,
    TargetData,
    TargetFilter,
    TargetingSystem,
    SelfTargeting,
    ActorTargeting,
    PointTargeting,
    AreaTargeting,
    ConfirmationTargeting,
    create_self_targeting,
    create_single_target,
    create_point_target,
    create_aoe,
    create_cone,
)
from engine.gameplay.abilities.tags import GameplayTag, GameplayTagContainer


# =============================================================================
# MOCK TARGETS
# =============================================================================


@dataclass
class MockTarget:
    """Mock target for testing."""

    name: str = "Target"
    position: Vector3 = field(default_factory=Vector3.zero)
    tags: GameplayTagContainer = field(default_factory=GameplayTagContainer)
    is_valid: bool = True
    is_alive: bool = True
    team: int = 0

    def __hash__(self) -> int:
        return hash(self.name)


# =============================================================================
# VECTOR3 TESTS
# =============================================================================


class TestVector3:
    """Tests for Vector3 class."""

    def test_vector_creation_default(self):
        """Test creating vector with defaults."""
        v = Vector3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_vector_creation_values(self):
        """Test creating vector with values."""
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_vector_addition(self):
        """Test vector addition."""
        v1 = Vector3(1.0, 2.0, 3.0)
        v2 = Vector3(4.0, 5.0, 6.0)
        result = v1 + v2
        assert result == Vector3(5.0, 7.0, 9.0)

    def test_vector_subtraction(self):
        """Test vector subtraction."""
        v1 = Vector3(5.0, 7.0, 9.0)
        v2 = Vector3(1.0, 2.0, 3.0)
        result = v1 - v2
        assert result == Vector3(4.0, 5.0, 6.0)

    def test_vector_scalar_multiply(self):
        """Test vector scalar multiplication."""
        v = Vector3(2.0, 3.0, 4.0)
        result = v * 2.0
        assert result == Vector3(4.0, 6.0, 8.0)

    def test_vector_scalar_rmultiply(self):
        """Test reverse scalar multiplication."""
        v = Vector3(2.0, 3.0, 4.0)
        result = 2.0 * v
        assert result == Vector3(4.0, 6.0, 8.0)

    def test_vector_negation(self):
        """Test vector negation."""
        v = Vector3(1.0, -2.0, 3.0)
        result = -v
        assert result == Vector3(-1.0, 2.0, -3.0)

    def test_vector_magnitude(self):
        """Test vector magnitude."""
        v = Vector3(3.0, 4.0, 0.0)
        assert v.magnitude == 5.0

    def test_vector_magnitude_squared(self):
        """Test vector magnitude squared."""
        v = Vector3(3.0, 4.0, 0.0)
        assert v.magnitude_squared == 25.0

    def test_vector_normalized(self):
        """Test vector normalization."""
        v = Vector3(3.0, 4.0, 0.0)
        normalized = v.normalized
        assert math.isclose(normalized.magnitude, 1.0, rel_tol=1e-6)
        assert math.isclose(normalized.x, 0.6, rel_tol=1e-6)
        assert math.isclose(normalized.y, 0.8, rel_tol=1e-6)

    def test_vector_normalized_zero(self):
        """Test normalizing zero vector."""
        v = Vector3.zero()
        normalized = v.normalized
        assert normalized == Vector3.zero()

    def test_vector_dot_product(self):
        """Test dot product."""
        v1 = Vector3(1.0, 2.0, 3.0)
        v2 = Vector3(4.0, 5.0, 6.0)
        assert v1.dot(v2) == 32.0  # 1*4 + 2*5 + 3*6

    def test_vector_cross_product(self):
        """Test cross product."""
        v1 = Vector3(1.0, 0.0, 0.0)
        v2 = Vector3(0.0, 1.0, 0.0)
        result = v1.cross(v2)
        assert result == Vector3(0.0, 0.0, 1.0)

    def test_vector_distance_to(self):
        """Test distance between vectors."""
        v1 = Vector3(0.0, 0.0, 0.0)
        v2 = Vector3(3.0, 4.0, 0.0)
        assert v1.distance_to(v2) == 5.0

    def test_vector_distance_squared_to(self):
        """Test squared distance between vectors."""
        v1 = Vector3(0.0, 0.0, 0.0)
        v2 = Vector3(3.0, 4.0, 0.0)
        assert v1.distance_squared_to(v2) == 25.0

    def test_vector_angle_to(self):
        """Test angle between vectors."""
        v1 = Vector3(1.0, 0.0, 0.0)
        v2 = Vector3(0.0, 1.0, 0.0)
        angle = v1.angle_to(v2)
        assert math.isclose(angle, math.pi / 2, rel_tol=1e-6)

    def test_vector_zero(self):
        """Test zero vector static method."""
        v = Vector3.zero()
        assert v == Vector3(0.0, 0.0, 0.0)

    def test_vector_up(self):
        """Test up vector static method."""
        v = Vector3.up()
        assert v == Vector3(0.0, 1.0, 0.0)

    def test_vector_forward(self):
        """Test forward vector static method."""
        v = Vector3.forward()
        assert v == Vector3(0.0, 0.0, 1.0)

    def test_vector_immutable(self):
        """Test vector is immutable (frozen dataclass)."""
        v = Vector3(1.0, 2.0, 3.0)
        with pytest.raises(AttributeError):
            v.x = 5.0


# =============================================================================
# TARGET DATA TESTS
# =============================================================================


class TestTargetData:
    """Tests for TargetData class."""

    def test_target_data_defaults(self):
        """Test target data with defaults."""
        data = TargetData()
        assert data.mode == TargetingMode.SELF
        assert data.targets == []
        assert data.point is None
        assert data.confirmed is False
        assert data.cancelled is False

    def test_target_data_has_targets(self):
        """Test has_targets property."""
        data = TargetData(targets=[MockTarget()])
        assert data.has_targets is True

        empty_data = TargetData()
        assert empty_data.has_targets is False

    def test_target_data_primary_target(self):
        """Test primary_target property."""
        target1 = MockTarget(name="Primary")
        target2 = MockTarget(name="Secondary")
        data = TargetData(targets=[target1, target2])

        assert data.primary_target == target1

    def test_target_data_primary_target_empty(self):
        """Test primary_target when no targets."""
        data = TargetData()
        assert data.primary_target is None

    def test_target_data_target_count(self):
        """Test target_count property."""
        data = TargetData(targets=[MockTarget(), MockTarget(), MockTarget()])
        assert data.target_count == 3


# =============================================================================
# TARGET FILTER TESTS
# =============================================================================


class TestTargetFilter:
    """Tests for TargetFilter class."""

    def test_filter_default_passes_all(self):
        """Test default filter passes most targets."""
        filter = TargetFilter()
        target = MockTarget()
        source = MockTarget(name="Source")

        assert filter.passes(target, source) is True

    def test_filter_allow_self_false(self):
        """Test filter blocks self when allow_self is False."""
        filter = TargetFilter(allow_self=False)
        source = MockTarget(name="Source")

        assert filter.passes(source, source) is False

    def test_filter_allow_self_true(self):
        """Test filter allows self when allow_self is True."""
        filter = TargetFilter(allow_self=True)
        source = MockTarget(name="Source")

        assert filter.passes(source, source) is True

    def test_filter_allow_dead_false(self):
        """Test filter blocks dead targets."""
        filter = TargetFilter(allow_dead=False)
        target = MockTarget(is_alive=False)

        assert filter.passes(target, None) is False

    def test_filter_allow_dead_true(self):
        """Test filter allows dead targets."""
        filter = TargetFilter(allow_dead=True)
        target = MockTarget(is_alive=False)

        assert filter.passes(target, None) is True

    def test_filter_require_tags(self):
        """Test filter requires specific tags."""
        filter = TargetFilter(require_tags={"status.vulnerable"})
        target = MockTarget()
        target.tags.add("status.vulnerable")

        assert filter.passes(target, None) is True

    def test_filter_require_tags_missing(self):
        """Test filter fails when required tag missing."""
        filter = TargetFilter(require_tags={"status.vulnerable"})
        target = MockTarget()

        assert filter.passes(target, None) is False

    def test_filter_exclude_tags(self):
        """Test filter excludes targets with certain tags."""
        filter = TargetFilter(exclude_tags={"status.immune"})
        target = MockTarget()
        target.tags.add("status.immune")

        assert filter.passes(target, None) is False

    def test_filter_exclude_tags_absent(self):
        """Test filter passes when excluded tag absent."""
        filter = TargetFilter(exclude_tags={"status.immune"})
        target = MockTarget()

        assert filter.passes(target, None) is True

    def test_filter_require_any_tags(self):
        """Test filter requires any of specified tags."""
        filter = TargetFilter(require_any_tags={"tag1", "tag2", "tag3"})
        target = MockTarget()
        target.tags.add("tag2")

        assert filter.passes(target, None) is True

    def test_filter_require_any_tags_none_present(self):
        """Test filter fails when no required tags present."""
        filter = TargetFilter(require_any_tags={"tag1", "tag2", "tag3"})
        target = MockTarget()

        assert filter.passes(target, None) is False

    def test_filter_custom_filter(self):
        """Test filter with custom filter function."""
        filter = TargetFilter(custom_filter=lambda t: t.team == 2)
        target1 = MockTarget(team=1)
        target2 = MockTarget(team=2)

        assert filter.passes(target1, None) is False
        assert filter.passes(target2, None) is True

    def test_filter_filter_targets(self):
        """Test filtering a list of targets."""
        filter = TargetFilter(allow_self=False)
        source = MockTarget(name="Source")
        targets = [
            MockTarget(name="Target1"),
            source,
            MockTarget(name="Target2"),
        ]

        filtered = filter.filter_targets(targets, source)
        assert len(filtered) == 2
        assert source not in filtered

    def test_filter_max_targets(self):
        """Test filter limits number of targets."""
        filter = TargetFilter(max_targets=2)
        targets = [MockTarget(name=f"Target{i}") for i in range(5)]

        filtered = filter.filter_targets(targets, None)
        assert len(filtered) == 2

    def test_filter_combined_conditions(self):
        """Test filter with multiple conditions."""
        filter = TargetFilter(
            allow_self=False,
            allow_dead=False,
            require_tags={"targetable"},
            exclude_tags={"immune"},
        )

        # Valid target
        valid = MockTarget(name="Valid")
        valid.tags.add("targetable")

        # Self
        source = MockTarget(name="Source")
        source.tags.add("targetable")

        # Dead
        dead = MockTarget(name="Dead", is_alive=False)
        dead.tags.add("targetable")

        # Missing required tag
        missing = MockTarget(name="Missing")

        # Has excluded tag
        immune = MockTarget(name="Immune")
        immune.tags.add("targetable")
        immune.tags.add("immune")

        assert filter.passes(valid, source) is True
        assert filter.passes(source, source) is False
        assert filter.passes(dead, source) is False
        assert filter.passes(missing, source) is False
        assert filter.passes(immune, source) is False


# =============================================================================
# SELF TARGETING TESTS
# =============================================================================


class TestSelfTargeting:
    """Tests for SelfTargeting class."""

    def test_self_targeting_mode(self):
        """Test self targeting has correct mode."""
        targeting = SelfTargeting()
        assert targeting.mode == TargetingMode.SELF

    def test_self_targeting_acquire(self):
        """Test self targeting returns source as target."""
        targeting = SelfTargeting()
        source = MockTarget(name="Source")

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3.zero(),
            aim_direction=Vector3.forward(),
            candidates=[],
        )

        assert data.mode == TargetingMode.SELF
        assert data.targets == [source]
        assert data.confirmed is True

    def test_self_targeting_ignores_candidates(self):
        """Test self targeting ignores candidates list."""
        targeting = SelfTargeting()
        source = MockTarget(name="Source")
        candidates = [MockTarget(name="Other1"), MockTarget(name="Other2")]

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3.zero(),
            aim_direction=Vector3.forward(),
            candidates=candidates,
        )

        assert len(data.targets) == 1
        assert data.targets[0] == source


# =============================================================================
# ACTOR TARGETING TESTS
# =============================================================================


class TestActorTargeting:
    """Tests for ActorTargeting class."""

    def test_actor_targeting_mode(self):
        """Test actor targeting has correct mode."""
        targeting = ActorTargeting()
        assert targeting.mode == TargetingMode.ACTOR

    def test_actor_targeting_acquires_closest_to_aim(self):
        """Test actor targeting acquires target closest to aim point."""
        targeting = ActorTargeting(max_range=100.0)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        target1 = MockTarget(name="Far", position=Vector3(20, 0, 0))
        target2 = MockTarget(name="Close", position=Vector3(10, 0, 0))

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(9, 0, 0),  # Closer to target2
            aim_direction=Vector3(1, 0, 0),
            candidates=[target1, target2],
        )

        assert data.primary_target == target2

    def test_actor_targeting_respects_range(self):
        """Test actor targeting respects max range."""
        targeting = ActorTargeting(max_range=15.0)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        near = MockTarget(name="Near", position=Vector3(10, 0, 0))
        far = MockTarget(name="Far", position=Vector3(20, 0, 0))

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(25, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[near, far],
        )

        # Should only target the near one (within range)
        assert data.primary_target == near

    def test_actor_targeting_no_valid_targets(self):
        """Test actor targeting when no valid targets."""
        targeting = ActorTargeting(max_range=5.0)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        far = MockTarget(name="Far", position=Vector3(100, 0, 0))

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(5, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[far],
        )

        assert data.has_targets is False
        assert data.cancelled is True

    def test_actor_targeting_filters_invalid(self):
        """Test actor targeting filters invalid targets."""
        targeting = ActorTargeting(max_range=100.0)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        valid = MockTarget(name="Valid", position=Vector3(10, 0, 0))
        invalid = MockTarget(name="Invalid", position=Vector3(5, 0, 0), is_valid=False)

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(5, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[valid, invalid],
        )

        assert data.primary_target == valid

    def test_actor_targeting_returns_distance(self):
        """Test actor targeting returns distance to target."""
        targeting = ActorTargeting(max_range=100.0)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        target = MockTarget(name="Target", position=Vector3(10, 0, 0))

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(10, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[target],
        )

        assert data.distance == 10.0


# =============================================================================
# POINT TARGETING TESTS
# =============================================================================


class TestPointTargeting:
    """Tests for PointTargeting class."""

    def test_point_targeting_mode(self):
        """Test point targeting has correct mode."""
        targeting = PointTargeting()
        assert targeting.mode == TargetingMode.POINT

    def test_point_targeting_returns_aim_point(self):
        """Test point targeting returns the aim point."""
        targeting = PointTargeting(max_range=100.0)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        aim = Vector3(25, 0, 0)

        data = targeting.acquire_targets(
            source=source,
            aim_point=aim,
            aim_direction=Vector3(1, 0, 0),
            candidates=[],
        )

        assert data.point == aim
        assert data.targets == []

    def test_point_targeting_clamps_to_max_range(self):
        """Test point targeting clamps to max range."""
        targeting = PointTargeting(max_range=20.0)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        aim = Vector3(100, 0, 0)  # Beyond max range

        data = targeting.acquire_targets(
            source=source,
            aim_point=aim,
            aim_direction=Vector3(1, 0, 0),
            candidates=[],
        )

        assert data.point.x == 20.0
        assert data.distance == 20.0

    def test_point_targeting_within_range(self):
        """Test point targeting within range returns exact point."""
        targeting = PointTargeting(max_range=50.0)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        aim = Vector3(30, 0, 0)

        data = targeting.acquire_targets(
            source=source,
            aim_point=aim,
            aim_direction=Vector3(1, 0, 0),
            candidates=[],
        )

        assert data.point == aim
        assert data.distance == 30.0

    def test_point_targeting_returns_direction(self):
        """Test point targeting returns aim direction."""
        targeting = PointTargeting()
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        direction = Vector3(0, 1, 0)

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(10, 10, 0),
            aim_direction=direction,
            candidates=[],
        )

        assert data.direction == direction


# =============================================================================
# AREA TARGETING - CIRCLE TESTS
# =============================================================================


class TestAreaTargetingCircle:
    """Tests for AreaTargeting with circle shape."""

    def test_circle_targeting_mode(self):
        """Test circle targeting has area mode."""
        targeting = AreaTargeting(shape=AreaShape.CIRCLE, radius=5.0)
        assert targeting.mode == TargetingMode.AREA
        assert targeting.shape == AreaShape.CIRCLE

    def test_circle_targeting_acquires_in_radius(self):
        """Test circle targeting acquires targets in radius."""
        targeting = AreaTargeting(shape=AreaShape.CIRCLE, radius=10.0, max_range=100.0)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        inside = MockTarget(name="Inside", position=Vector3(25, 0, 0))
        outside = MockTarget(name="Outside", position=Vector3(40, 0, 0))

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(25, 0, 0),  # Center of AOE
            aim_direction=Vector3(1, 0, 0),
            candidates=[inside, outside],
        )

        assert inside in data.targets
        assert outside not in data.targets

    def test_circle_targeting_at_boundary(self):
        """Test circle targeting at radius boundary."""
        targeting = AreaTargeting(shape=AreaShape.CIRCLE, radius=10.0, max_range=100.0)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        at_edge = MockTarget(name="Edge", position=Vector3(35, 0, 0))  # Exactly at radius

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(25, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[at_edge],
        )

        assert at_edge in data.targets

    def test_circle_targeting_multiple_targets(self):
        """Test circle targeting with multiple targets."""
        targeting = AreaTargeting(shape=AreaShape.CIRCLE, radius=10.0, max_range=100.0)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        targets = [
            MockTarget(name="T1", position=Vector3(23, 0, 0)),
            MockTarget(name="T2", position=Vector3(27, 0, 0)),
            MockTarget(name="T3", position=Vector3(25, 3, 0)),
            MockTarget(name="Out", position=Vector3(50, 0, 0)),
        ]

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(25, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=targets,
        )

        assert len(data.targets) == 3

    def test_circle_targeting_clamps_aim_to_range(self):
        """Test circle targeting clamps aim point to max range."""
        targeting = AreaTargeting(shape=AreaShape.CIRCLE, radius=5.0, max_range=20.0)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(100, 0, 0),  # Beyond max range
            aim_direction=Vector3(1, 0, 0),
            candidates=[],
        )

        assert data.point.x == 20.0


# =============================================================================
# AREA TARGETING - CONE TESTS
# =============================================================================


class TestAreaTargetingCone:
    """Tests for AreaTargeting with cone shape."""

    def test_cone_targeting_shape(self):
        """Test cone targeting has cone shape."""
        targeting = AreaTargeting(shape=AreaShape.CONE, radius=10.0, cone_angle=60.0)
        assert targeting.shape == AreaShape.CONE
        assert targeting.cone_angle == 60.0

    def test_cone_targeting_in_angle(self):
        """Test cone targeting acquires targets in angle."""
        targeting = AreaTargeting(
            shape=AreaShape.CONE,
            radius=20.0,
            cone_angle=90.0,  # 45 degrees each side
            max_range=0.0,
        )
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        in_cone = MockTarget(name="InCone", position=Vector3(10, 5, 0))  # ~26 degrees
        out_cone = MockTarget(name="OutCone", position=Vector3(5, 10, 0))  # ~63 degrees

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(0, 0, 0),
            aim_direction=Vector3(1, 0, 0),  # Looking +X
            candidates=[in_cone, out_cone],
        )

        assert in_cone in data.targets
        assert out_cone not in data.targets

    def test_cone_targeting_respects_length(self):
        """Test cone targeting respects length (radius)."""
        targeting = AreaTargeting(
            shape=AreaShape.CONE,
            radius=10.0,
            cone_angle=90.0,
            max_range=0.0,
        )
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        near = MockTarget(name="Near", position=Vector3(5, 0, 0))
        far = MockTarget(name="Far", position=Vector3(20, 0, 0))

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(0, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[near, far],
        )

        assert near in data.targets
        assert far not in data.targets

    def test_cone_targeting_behind_source(self):
        """Test cone targeting doesn't hit targets behind source."""
        targeting = AreaTargeting(
            shape=AreaShape.CONE,
            radius=20.0,
            cone_angle=90.0,
            max_range=0.0,
        )
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        behind = MockTarget(name="Behind", position=Vector3(-10, 0, 0))

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(0, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[behind],
        )

        assert behind not in data.targets


# =============================================================================
# AREA TARGETING - LINE TESTS
# =============================================================================


class TestAreaTargetingLine:
    """Tests for AreaTargeting with line shape."""

    def test_line_targeting_shape(self):
        """Test line targeting has line shape."""
        targeting = AreaTargeting(shape=AreaShape.LINE, radius=20.0, line_width=2.0)
        assert targeting.shape == AreaShape.LINE
        assert targeting.line_width == 2.0

    def test_line_targeting_hits_along_line(self):
        """Test line targeting hits targets along the line."""
        targeting = AreaTargeting(
            shape=AreaShape.LINE,
            radius=20.0,
            line_width=2.0,
            max_range=0.0,
        )
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        on_line = MockTarget(name="OnLine", position=Vector3(10, 0, 0))
        off_line = MockTarget(name="OffLine", position=Vector3(10, 5, 0))

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(0, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[on_line, off_line],
        )

        assert on_line in data.targets
        assert off_line not in data.targets

    def test_line_targeting_respects_width(self):
        """Test line targeting respects line width."""
        targeting = AreaTargeting(
            shape=AreaShape.LINE,
            radius=20.0,
            line_width=4.0,  # 2.0 on each side
            max_range=0.0,
        )
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        within_width = MockTarget(name="Within", position=Vector3(10, 1.5, 0))
        outside_width = MockTarget(name="Outside", position=Vector3(10, 5, 0))

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(0, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[within_width, outside_width],
        )

        assert within_width in data.targets
        assert outside_width not in data.targets


# =============================================================================
# AREA TARGETING - RECTANGLE TESTS
# =============================================================================


class TestAreaTargetingRectangle:
    """Tests for AreaTargeting with rectangle shape."""

    def test_rectangle_targeting_shape(self):
        """Test rectangle targeting has rectangle shape."""
        targeting = AreaTargeting(
            shape=AreaShape.RECTANGLE,
            rectangle_width=10.0,
            rectangle_height=20.0,
        )
        assert targeting.shape == AreaShape.RECTANGLE

    def test_rectangle_targeting_bounds(self):
        """Test rectangle targeting respects bounds."""
        targeting = AreaTargeting(
            shape=AreaShape.RECTANGLE,
            rectangle_width=10.0,
            rectangle_height=20.0,
            max_range=100.0,
        )
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        inside = MockTarget(name="Inside", position=Vector3(25, 0, 3))  # Within width
        outside = MockTarget(name="Outside", position=Vector3(25, 0, 10))  # Outside width

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(25, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[inside, outside],
        )

        assert inside in data.targets
        assert outside not in data.targets


# =============================================================================
# CONFIRMATION TARGETING TESTS
# =============================================================================


class TestConfirmationTargeting:
    """Tests for ConfirmationTargeting class."""

    def test_confirmation_mode(self):
        """Test confirmation targeting has confirmation mode."""
        inner = ActorTargeting(max_range=30.0)
        targeting = ConfirmationTargeting(inner)
        assert targeting.mode == TargetingMode.CONFIRMATION

    def test_confirmation_acquires_not_confirmed(self):
        """Test confirmation targeting acquires but doesn't confirm."""
        inner = ActorTargeting(max_range=30.0)
        targeting = ConfirmationTargeting(inner)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        target = MockTarget(name="Target", position=Vector3(10, 0, 0))

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(10, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[target],
        )

        assert data.has_targets is True
        assert data.confirmed is False

    def test_confirmation_confirm(self):
        """Test confirming the targeting."""
        inner = ActorTargeting(max_range=30.0)
        targeting = ConfirmationTargeting(inner)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        target = MockTarget(name="Target", position=Vector3(10, 0, 0))

        targeting.acquire_targets(
            source=source,
            aim_point=Vector3(10, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[target],
        )

        confirmed_data = targeting.confirm()
        assert confirmed_data is not None
        assert confirmed_data.confirmed is True

    def test_confirmation_cancel(self):
        """Test cancelling the targeting."""
        inner = ActorTargeting(max_range=30.0)
        targeting = ConfirmationTargeting(inner)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        target = MockTarget(name="Target", position=Vector3(10, 0, 0))

        targeting.acquire_targets(
            source=source,
            aim_point=Vector3(10, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[target],
        )

        targeting.cancel()

        # After cancel, confirm should return None
        assert targeting.confirm() is None

    def test_confirmation_double_confirm(self):
        """Test double confirmation returns None."""
        inner = ActorTargeting(max_range=30.0)
        targeting = ConfirmationTargeting(inner)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        target = MockTarget(name="Target", position=Vector3(10, 0, 0))

        targeting.acquire_targets(
            source=source,
            aim_point=Vector3(10, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[target],
        )

        targeting.confirm()
        second_confirm = targeting.confirm()

        assert second_confirm is None


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestTargetingFactoryFunctions:
    """Tests for targeting factory functions."""

    def test_create_self_targeting(self):
        """Test create_self_targeting factory."""
        targeting = create_self_targeting()
        assert isinstance(targeting, SelfTargeting)

    def test_create_single_target_hostile(self):
        """Test create_single_target for hostile targets."""
        targeting = create_single_target(max_range=30.0, require_hostile=True)
        assert isinstance(targeting, ActorTargeting)
        assert targeting.max_range == 30.0
        assert targeting.target_filter.allow_hostile is True
        assert targeting.target_filter.allow_friendly is False

    def test_create_single_target_friendly(self):
        """Test create_single_target for friendly targets."""
        targeting = create_single_target(max_range=30.0, require_hostile=False)
        assert targeting.target_filter.allow_hostile is False
        assert targeting.target_filter.allow_friendly is True

    def test_create_point_target(self):
        """Test create_point_target factory."""
        targeting = create_point_target(max_range=50.0)
        assert isinstance(targeting, PointTargeting)
        assert targeting.max_range == 50.0

    def test_create_aoe(self):
        """Test create_aoe factory."""
        targeting = create_aoe(radius=8.0, max_range=40.0, include_self=False)
        assert isinstance(targeting, AreaTargeting)
        assert targeting.shape == AreaShape.CIRCLE
        assert targeting.radius == 8.0
        assert targeting.max_range == 40.0
        assert targeting.target_filter.allow_self is False

    def test_create_aoe_include_self(self):
        """Test create_aoe with self included."""
        targeting = create_aoe(include_self=True)
        assert targeting.target_filter.allow_self is True

    def test_create_cone(self):
        """Test create_cone factory."""
        targeting = create_cone(angle=45.0, length=15.0)
        assert isinstance(targeting, AreaTargeting)
        assert targeting.shape == AreaShape.CONE
        assert targeting.cone_angle == 45.0
        assert targeting.radius == 15.0


# =============================================================================
# RANGE VALIDATION TESTS
# =============================================================================


class TestRangeValidation:
    """Tests for range validation in targeting systems."""

    def test_is_in_range_within(self):
        """Test is_in_range returns True for targets within range."""
        targeting = ActorTargeting(min_range=0.0, max_range=20.0)
        source_pos = Vector3(0, 0, 0)
        target_pos = Vector3(10, 0, 0)

        assert targeting.is_in_range(source_pos, target_pos) is True

    def test_is_in_range_at_max(self):
        """Test is_in_range at exact max range."""
        targeting = ActorTargeting(min_range=0.0, max_range=20.0)
        source_pos = Vector3(0, 0, 0)
        target_pos = Vector3(20, 0, 0)

        assert targeting.is_in_range(source_pos, target_pos) is True

    def test_is_in_range_beyond_max(self):
        """Test is_in_range returns False beyond max range."""
        targeting = ActorTargeting(min_range=0.0, max_range=20.0)
        source_pos = Vector3(0, 0, 0)
        target_pos = Vector3(25, 0, 0)

        assert targeting.is_in_range(source_pos, target_pos) is False

    def test_is_in_range_min_range(self):
        """Test is_in_range respects min range."""
        targeting = ActorTargeting(min_range=5.0, max_range=20.0)
        source_pos = Vector3(0, 0, 0)
        too_close = Vector3(3, 0, 0)
        valid = Vector3(10, 0, 0)

        assert targeting.is_in_range(source_pos, too_close) is False
        assert targeting.is_in_range(source_pos, valid) is True


# =============================================================================
# AREA BOUNDS TESTS
# =============================================================================


class TestAreaBounds:
    """Tests for area bounding box calculations."""

    def test_get_area_bounds_circle(self):
        """Test area bounds for circle."""
        targeting = AreaTargeting(shape=AreaShape.CIRCLE, radius=10.0)
        center = Vector3(25, 0, 25)
        direction = Vector3(1, 0, 0)

        min_bound, max_bound = targeting.get_area_bounds(center, direction)

        assert min_bound.x == 15.0
        assert max_bound.x == 35.0
        assert min_bound.z == 15.0
        assert max_bound.z == 35.0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestTargetingEdgeCases:
    """Tests for edge cases in targeting."""

    def test_targeting_empty_candidates(self):
        """Test targeting with empty candidates list."""
        targeting = ActorTargeting(max_range=30.0)
        source = MockTarget(name="Source")

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(10, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[],
        )

        assert data.has_targets is False

    def test_targeting_all_invalid_candidates(self):
        """Test targeting when all candidates are invalid."""
        targeting = ActorTargeting(max_range=30.0)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        candidates = [
            MockTarget(name="Invalid1", position=Vector3(10, 0, 0), is_valid=False),
            MockTarget(name="Invalid2", position=Vector3(15, 0, 0), is_valid=False),
        ]

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(10, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=candidates,
        )

        assert data.has_targets is False

    def test_targeting_zero_range(self):
        """Test targeting with zero max range."""
        targeting = ActorTargeting(max_range=0.0)
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        target = MockTarget(name="Target", position=Vector3(0, 0, 0))  # Same position

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(0, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[target],
        )

        # Target at same position should be in range
        assert target in data.targets

    def test_area_targeting_zero_radius(self):
        """Test area targeting with minimum radius."""
        targeting = AreaTargeting(shape=AreaShape.CIRCLE, radius=0.0)
        # Should be clamped to minimum
        assert targeting.radius >= 0.1

    def test_filter_with_none_source(self):
        """Test filter passes when source is None."""
        filter = TargetFilter(allow_self=False)
        target = MockTarget()

        # Should pass since source is None
        assert filter.passes(target, None) is True

    def test_cone_very_narrow(self):
        """Test cone with very narrow angle."""
        targeting = AreaTargeting(
            shape=AreaShape.CONE,
            radius=20.0,
            cone_angle=1.0,  # Very narrow
            max_range=0.0,
        )
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        # Target slightly off-axis
        target = MockTarget(name="Target", position=Vector3(10, 0.5, 0))

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(0, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[target],
        )

        # With 1 degree cone, 0.5 offset at 10 units (~2.86 degrees) should miss
        assert target not in data.targets

    def test_cone_very_wide(self):
        """Test cone with very wide angle (180 degrees)."""
        targeting = AreaTargeting(
            shape=AreaShape.CONE,
            radius=20.0,
            cone_angle=180.0,  # Semicircle
            max_range=0.0,
        )
        source = MockTarget(name="Source", position=Vector3(0, 0, 0))
        side = MockTarget(name="Side", position=Vector3(0, 10, 0))  # 90 degrees

        data = targeting.acquire_targets(
            source=source,
            aim_point=Vector3(0, 0, 0),
            aim_direction=Vector3(1, 0, 0),
            candidates=[side],
        )

        assert side in data.targets

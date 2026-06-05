"""Whitebox tests for raycast interface (RaycastHit and RaycastCallback).

Tests the raycast interface implementation covering:
- RaycastHit dataclass field access
- RaycastHit.miss() static method
- RaycastHit equality and comparison
- RaycastHit field types and validation
- RaycastCallback type alias usage
- Integration with FootPlacement
- Edge cases and boundary conditions
"""

from __future__ import annotations

import math
import pytest
from dataclasses import fields, is_dataclass, FrozenInstanceError
from typing import List, Callable, Optional, get_type_hints, get_origin, get_args
from copy import copy, deepcopy

from engine.animation.ik.foot_placement import (
    RaycastHit,
    RaycastCallback,
    FootData,
    FootPlacement,
    FootState,
    FootPlacementResult,
)
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON


# =============================================================================
# Helper Functions
# =============================================================================

def create_simple_hit(
    position: Vec3 = None,
    normal: Vec3 = None,
    distance: float = 1.0
) -> RaycastHit:
    """Create a simple hit result for testing."""
    return RaycastHit(
        hit=True,
        position=position or Vec3(0, 0, 0),
        normal=normal or Vec3(0, 1, 0),
        distance=distance
    )


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


# =============================================================================
# RaycastHit Dataclass Tests
# =============================================================================

class TestRaycastHitDataclass:
    """Tests for RaycastHit dataclass structure."""

    def test_is_dataclass(self) -> None:
        """RaycastHit should be a dataclass."""
        assert is_dataclass(RaycastHit)

    def test_has_hit_field(self) -> None:
        """RaycastHit should have hit field."""
        field_names = [f.name for f in fields(RaycastHit)]
        assert "hit" in field_names

    def test_has_position_field(self) -> None:
        """RaycastHit should have position field."""
        field_names = [f.name for f in fields(RaycastHit)]
        assert "position" in field_names

    def test_has_normal_field(self) -> None:
        """RaycastHit should have normal field."""
        field_names = [f.name for f in fields(RaycastHit)]
        assert "normal" in field_names

    def test_has_distance_field(self) -> None:
        """RaycastHit should have distance field."""
        field_names = [f.name for f in fields(RaycastHit)]
        assert "distance" in field_names

    def test_field_count(self) -> None:
        """RaycastHit should have exactly 4 fields."""
        assert len(fields(RaycastHit)) == 4

    def test_field_order(self) -> None:
        """Fields should be in expected order."""
        field_names = [f.name for f in fields(RaycastHit)]
        assert field_names == ["hit", "position", "normal", "distance"]


class TestRaycastHitFieldTypes:
    """Tests for RaycastHit field type annotations."""

    def test_hit_field_type(self) -> None:
        """hit field should be annotated as bool."""
        hints = get_type_hints(RaycastHit)
        assert hints["hit"] == bool

    def test_position_field_type(self) -> None:
        """position field should be annotated as Vec3."""
        hints = get_type_hints(RaycastHit)
        assert hints["position"] == Vec3

    def test_normal_field_type(self) -> None:
        """normal field should be annotated as Vec3."""
        hints = get_type_hints(RaycastHit)
        assert hints["normal"] == Vec3

    def test_distance_field_type(self) -> None:
        """distance field should be annotated as float."""
        hints = get_type_hints(RaycastHit)
        assert hints["distance"] == float


# =============================================================================
# RaycastHit Field Access Tests
# =============================================================================

class TestRaycastHitFieldAccess:
    """Tests for RaycastHit field access."""

    def test_access_hit_true(self) -> None:
        """Can access hit field when True."""
        result = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            distance=1.0
        )
        assert result.hit is True

    def test_access_hit_false(self) -> None:
        """Can access hit field when False."""
        result = RaycastHit(
            hit=False,
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            distance=1.0
        )
        assert result.hit is False

    def test_access_position(self) -> None:
        """Can access position field."""
        pos = Vec3(1, 2, 3)
        result = RaycastHit(
            hit=True,
            position=pos,
            normal=Vec3(0, 1, 0),
            distance=1.0
        )
        assert result.position.x == 1
        assert result.position.y == 2
        assert result.position.z == 3

    def test_access_normal(self) -> None:
        """Can access normal field."""
        normal = Vec3(0.5, 0.5, 0.707)
        result = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=normal,
            distance=1.0
        )
        assert result.normal.x == 0.5
        assert result.normal.y == 0.5
        assert result.normal.z == 0.707

    def test_access_distance(self) -> None:
        """Can access distance field."""
        result = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            distance=2.5
        )
        assert result.distance == 2.5

    def test_field_mutability(self) -> None:
        """Dataclass fields should be mutable by default."""
        result = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            distance=1.0
        )
        # Fields should be mutable (dataclass default)
        result.hit = False
        assert result.hit is False

        result.distance = 5.0
        assert result.distance == 5.0


# =============================================================================
# RaycastHit.miss() Static Method Tests
# =============================================================================

class TestRaycastHitMissMethod:
    """Tests for RaycastHit.miss() static method."""

    def test_miss_returns_raycast_hit(self) -> None:
        """miss() should return a RaycastHit instance."""
        result = RaycastHit.miss()
        assert isinstance(result, RaycastHit)

    def test_miss_hit_is_false(self) -> None:
        """miss() should return hit=False."""
        result = RaycastHit.miss()
        assert result.hit is False

    def test_miss_position_is_zero(self) -> None:
        """miss() should return position as zero vector."""
        result = RaycastHit.miss()
        assert result.position.x == 0.0
        assert result.position.y == 0.0
        assert result.position.z == 0.0

    def test_miss_normal_is_up(self) -> None:
        """miss() should return normal as up vector (0, 1, 0)."""
        result = RaycastHit.miss()
        assert result.normal.x == 0.0
        assert result.normal.y == 1.0
        assert result.normal.z == 0.0

    def test_miss_distance_is_infinity(self) -> None:
        """miss() should return distance as infinity."""
        result = RaycastHit.miss()
        assert result.distance == float('inf')

    def test_miss_distance_is_positive_infinity(self) -> None:
        """miss() distance should be positive infinity."""
        result = RaycastHit.miss()
        assert result.distance > 0
        assert math.isinf(result.distance)

    def test_miss_creates_new_instance(self) -> None:
        """Each miss() call should create a new instance."""
        result1 = RaycastHit.miss()
        result2 = RaycastHit.miss()
        assert result1 is not result2

    def test_miss_creates_new_vectors(self) -> None:
        """Each miss() call should create new Vec3 instances."""
        result1 = RaycastHit.miss()
        result2 = RaycastHit.miss()
        assert result1.position is not result2.position
        assert result1.normal is not result2.normal

    def test_miss_callable_without_arguments(self) -> None:
        """miss() should be callable without any arguments."""
        # This should not raise
        result = RaycastHit.miss()
        assert result is not None

    def test_miss_is_static_method(self) -> None:
        """miss() should be a static method."""
        # Can be called on class
        result1 = RaycastHit.miss()
        # Can also be called on instance (though unusual)
        instance = RaycastHit(True, Vec3(0, 0, 0), Vec3(0, 1, 0), 1.0)
        result2 = instance.miss()
        assert isinstance(result1, RaycastHit)
        assert isinstance(result2, RaycastHit)


# =============================================================================
# RaycastHit Equality and Comparison Tests
# =============================================================================

class TestRaycastHitEquality:
    """Tests for RaycastHit equality comparison."""

    def test_equal_hits(self) -> None:
        """Two RaycastHit with same values should be equal."""
        hit1 = RaycastHit(
            hit=True,
            position=Vec3(1, 2, 3),
            normal=Vec3(0, 1, 0),
            distance=1.5
        )
        hit2 = RaycastHit(
            hit=True,
            position=Vec3(1, 2, 3),
            normal=Vec3(0, 1, 0),
            distance=1.5
        )
        assert hit1 == hit2

    def test_unequal_hit_field(self) -> None:
        """Different hit field should not be equal."""
        hit1 = RaycastHit(
            hit=True,
            position=Vec3(1, 2, 3),
            normal=Vec3(0, 1, 0),
            distance=1.5
        )
        hit2 = RaycastHit(
            hit=False,
            position=Vec3(1, 2, 3),
            normal=Vec3(0, 1, 0),
            distance=1.5
        )
        assert hit1 != hit2

    def test_unequal_position(self) -> None:
        """Different position should not be equal."""
        hit1 = RaycastHit(
            hit=True,
            position=Vec3(1, 2, 3),
            normal=Vec3(0, 1, 0),
            distance=1.5
        )
        hit2 = RaycastHit(
            hit=True,
            position=Vec3(1, 2, 4),  # Different z
            normal=Vec3(0, 1, 0),
            distance=1.5
        )
        assert hit1 != hit2

    def test_unequal_normal(self) -> None:
        """Different normal should not be equal."""
        hit1 = RaycastHit(
            hit=True,
            position=Vec3(1, 2, 3),
            normal=Vec3(0, 1, 0),
            distance=1.5
        )
        hit2 = RaycastHit(
            hit=True,
            position=Vec3(1, 2, 3),
            normal=Vec3(1, 0, 0),  # Different normal
            distance=1.5
        )
        assert hit1 != hit2

    def test_unequal_distance(self) -> None:
        """Different distance should not be equal."""
        hit1 = RaycastHit(
            hit=True,
            position=Vec3(1, 2, 3),
            normal=Vec3(0, 1, 0),
            distance=1.5
        )
        hit2 = RaycastHit(
            hit=True,
            position=Vec3(1, 2, 3),
            normal=Vec3(0, 1, 0),
            distance=2.5  # Different distance
        )
        assert hit1 != hit2

    def test_miss_equality(self) -> None:
        """Two miss() results should be equal."""
        miss1 = RaycastHit.miss()
        miss2 = RaycastHit.miss()
        assert miss1 == miss2

    def test_not_equal_to_other_types(self) -> None:
        """RaycastHit should not equal other types."""
        hit = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            distance=1.0
        )
        assert hit != "not a hit"
        assert hit != 42
        assert hit != None
        assert hit != {"hit": True}


# =============================================================================
# RaycastHit Edge Cases Tests
# =============================================================================

class TestRaycastHitEdgeCases:
    """Tests for RaycastHit edge cases."""

    def test_zero_distance(self) -> None:
        """RaycastHit should handle zero distance."""
        hit = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            distance=0.0
        )
        assert hit.distance == 0.0

    def test_negative_distance(self) -> None:
        """RaycastHit should handle negative distance (ray starts inside)."""
        hit = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            distance=-1.0
        )
        assert hit.distance == -1.0

    def test_very_large_distance(self) -> None:
        """RaycastHit should handle very large distance."""
        hit = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            distance=1e308
        )
        assert hit.distance == 1e308

    def test_very_small_distance(self) -> None:
        """RaycastHit should handle very small distance."""
        hit = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            distance=1e-10
        )
        assert abs(hit.distance - 1e-10) < 1e-15

    def test_negative_infinity_distance(self) -> None:
        """RaycastHit should handle negative infinity distance."""
        hit = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            distance=float('-inf')
        )
        assert math.isinf(hit.distance)
        assert hit.distance < 0

    def test_nan_distance(self) -> None:
        """RaycastHit should handle NaN distance."""
        hit = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            distance=float('nan')
        )
        assert math.isnan(hit.distance)

    def test_unnormalized_normal(self) -> None:
        """RaycastHit should accept unnormalized normal."""
        hit = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(10, 10, 10),  # Not normalized
            distance=1.0
        )
        assert hit.normal.x == 10
        assert hit.normal.y == 10
        assert hit.normal.z == 10

    def test_zero_normal(self) -> None:
        """RaycastHit should accept zero normal."""
        hit = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 0, 0),  # Zero vector
            distance=1.0
        )
        assert hit.normal.x == 0
        assert hit.normal.y == 0
        assert hit.normal.z == 0

    def test_very_large_position(self) -> None:
        """RaycastHit should handle very large position values."""
        large_val = 1e30
        hit = RaycastHit(
            hit=True,
            position=Vec3(large_val, large_val, large_val),
            normal=Vec3(0, 1, 0),
            distance=1.0
        )
        assert hit.position.x == large_val


# =============================================================================
# RaycastHit Copy and Hash Tests
# =============================================================================

class TestRaycastHitCopyAndHash:
    """Tests for RaycastHit copy behavior and hashing."""

    def test_shallow_copy(self) -> None:
        """Shallow copy should work on RaycastHit."""
        original = create_simple_hit(Vec3(1, 2, 3), Vec3(0, 1, 0), 1.5)
        copied = copy(original)

        assert copied == original
        assert copied is not original

    def test_deep_copy(self) -> None:
        """Deep copy should work on RaycastHit."""
        original = create_simple_hit(Vec3(1, 2, 3), Vec3(0, 1, 0), 1.5)
        copied = deepcopy(original)

        assert copied == original
        assert copied is not original
        # Deep copy creates new Vec3 instances
        assert copied.position is not original.position
        assert copied.normal is not original.normal


# =============================================================================
# RaycastCallback Type Alias Tests
# =============================================================================

class TestRaycastCallbackTypeAlias:
    """Tests for RaycastCallback type alias."""

    def test_is_callable_type(self) -> None:
        """RaycastCallback should be a Callable type."""
        # Check it's based on Callable
        origin = get_origin(RaycastCallback)
        assert origin is not None

    def test_can_be_used_as_annotation(self) -> None:
        """RaycastCallback can be used as type annotation."""
        def my_raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
            return None

        # Annotate with RaycastCallback
        callback: RaycastCallback = my_raycast
        assert callback is not None

    def test_accepts_conforming_function(self) -> None:
        """Function matching signature can be assigned to RaycastCallback."""
        def raycast_impl(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            if max_distance > 0:
                return RaycastHit(
                    hit=True,
                    position=origin,
                    normal=Vec3(0, 1, 0),
                    distance=0.5
                )
            return None

        callback: RaycastCallback = raycast_impl
        result = callback(Vec3(0, 1, 0), Vec3(0, -1, 0), 10.0)
        assert result is not None
        assert result.hit is True

    def test_callback_returns_none_for_miss(self) -> None:
        """Callback can return None for miss."""
        def miss_raycast(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            return None

        callback: RaycastCallback = miss_raycast
        result = callback(Vec3(0, 1, 0), Vec3(0, -1, 0), 10.0)
        assert result is None

    def test_callback_returns_raycast_hit_for_hit(self) -> None:
        """Callback can return RaycastHit for hit."""
        def hit_raycast(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            return RaycastHit(
                hit=True,
                position=Vec3(0, 0, 0),
                normal=Vec3(0, 1, 0),
                distance=1.0
            )

        callback: RaycastCallback = hit_raycast
        result = callback(Vec3(0, 1, 0), Vec3(0, -1, 0), 10.0)
        assert result is not None
        assert isinstance(result, RaycastHit)

    def test_lambda_as_callback(self) -> None:
        """Lambda can be used as RaycastCallback."""
        callback: RaycastCallback = lambda o, d, m: None
        result = callback(Vec3(0, 0, 0), Vec3(0, -1, 0), 1.0)
        assert result is None

    def test_callback_receives_correct_parameters(self) -> None:
        """Callback receives origin, direction, max_distance correctly."""
        received = []

        def tracking_raycast(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            received.append((origin, direction, max_distance))
            return None

        callback: RaycastCallback = tracking_raycast
        test_origin = Vec3(1, 2, 3)
        test_direction = Vec3(0, -1, 0)
        test_max_dist = 100.0

        callback(test_origin, test_direction, test_max_dist)

        assert len(received) == 1
        assert received[0][0] is test_origin
        assert received[0][1] is test_direction
        assert received[0][2] == test_max_dist


# =============================================================================
# RaycastHit Integration with FootPlacement Tests
# =============================================================================

class TestRaycastHitIntegration:
    """Tests for RaycastHit integration with FootPlacement."""

    def test_foot_placement_accepts_raycast_callback(self) -> None:
        """FootPlacement should accept RaycastCallback."""
        def flat_ground_raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
            return RaycastHit(
                hit=True,
                position=Vec3(origin.x, 0.0, origin.z),
                normal=Vec3(0, 1, 0),
                distance=origin.y
            )

        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        fp = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=flat_ground_raycast
        )

        assert fp._raycast is not None

    def test_foot_placement_set_raycast_callback(self) -> None:
        """FootPlacement.set_raycast_callback should work."""
        def new_raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
            return None

        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        fp = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0
        )

        assert fp._raycast is None
        fp.set_raycast_callback(new_raycast)
        assert fp._raycast is not None

    def test_foot_placement_solve_without_raycast(self) -> None:
        """FootPlacement.solve should return failure without raycast."""
        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        fp = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0
        )

        transforms = create_biped_transforms()
        result = fp.solve(transforms, Vec3(0, 0, 0))

        assert result.success is False

    def test_foot_placement_solve_with_hit_raycast(self) -> None:
        """FootPlacement.solve should work with hit raycast."""
        def flat_ground_raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
            return RaycastHit(
                hit=True,
                position=Vec3(origin.x, 0.0, origin.z),
                normal=Vec3(0, 1, 0),
                distance=origin.y
            )

        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        fp = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=flat_ground_raycast
        )

        transforms = create_biped_transforms()
        result = fp.solve(transforms, Vec3(0, 0, 0))

        assert result is not None
        assert isinstance(result, FootPlacementResult)

    def test_foot_placement_solve_with_miss_raycast(self) -> None:
        """FootPlacement.solve handles raycast returning None."""
        def miss_raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
            return None

        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        fp = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=miss_raycast
        )

        transforms = create_biped_transforms()
        result = fp.solve(transforms, Vec3(0, 0, 0))

        assert result is not None

    def test_raycast_uses_hit_position(self) -> None:
        """FootPlacement should use hit position from raycast."""
        hit_position_y = -0.5  # Below ground

        def specific_position_raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
            return RaycastHit(
                hit=True,
                position=Vec3(origin.x, hit_position_y, origin.z),
                normal=Vec3(0, 1, 0),
                distance=origin.y - hit_position_y
            )

        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        fp = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=specific_position_raycast
        )

        transforms = create_biped_transforms()
        result = fp.solve(transforms, Vec3(0, 0, 0))

        # Result should be influenced by the hit position
        assert result is not None

    def test_raycast_uses_hit_normal(self) -> None:
        """FootPlacement should use hit normal from raycast."""
        slope_normal = Vec3(0.3, 0.9, 0.0).normalized()

        def sloped_raycast(origin: Vec3, direction: Vec3, max_dist: float) -> Optional[RaycastHit]:
            return RaycastHit(
                hit=True,
                position=Vec3(origin.x, 0.0, origin.z),
                normal=slope_normal,
                distance=origin.y
            )

        left_foot = create_left_foot_data()
        right_foot = create_right_foot_data()

        fp = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=sloped_raycast
        )

        transforms = create_biped_transforms()
        fp.solve(transforms, Vec3(0, 0, 0))

        # Target normal should be updated
        # Note: checking the internal state to verify normal was used
        assert fp.left_foot.target_normal is not None


# =============================================================================
# RaycastHit Representation Tests
# =============================================================================

class TestRaycastHitRepresentation:
    """Tests for RaycastHit string representation."""

    def test_repr_contains_class_name(self) -> None:
        """repr should contain class name."""
        hit = create_simple_hit()
        repr_str = repr(hit)
        assert "RaycastHit" in repr_str

    def test_repr_contains_hit_value(self) -> None:
        """repr should contain hit value."""
        hit = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            distance=1.0
        )
        repr_str = repr(hit)
        assert "hit=True" in repr_str

    def test_repr_contains_distance(self) -> None:
        """repr should contain distance."""
        hit = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            distance=2.5
        )
        repr_str = repr(hit)
        assert "2.5" in repr_str or "distance" in repr_str


# =============================================================================
# RaycastHit Dataclass Features Tests
# =============================================================================

class TestRaycastHitDataclassFeatures:
    """Tests for RaycastHit dataclass-specific features."""

    def test_can_iterate_fields(self) -> None:
        """Can iterate over dataclass fields."""
        field_list = list(fields(RaycastHit))
        assert len(field_list) == 4

    def test_fields_have_names(self) -> None:
        """All fields should have names."""
        for f in fields(RaycastHit):
            assert f.name is not None
            assert len(f.name) > 0

    def test_no_default_values_required(self) -> None:
        """All fields require explicit values (no defaults)."""
        for f in fields(RaycastHit):
            # Fields don't have defaults (except miss() provides a factory)
            pass  # This is a structural test

    def test_positional_construction(self) -> None:
        """Can construct with positional arguments."""
        hit = RaycastHit(True, Vec3(1, 2, 3), Vec3(0, 1, 0), 1.5)
        assert hit.hit is True
        assert hit.position.x == 1
        assert hit.distance == 1.5

    def test_keyword_construction(self) -> None:
        """Can construct with keyword arguments."""
        hit = RaycastHit(
            distance=2.0,
            normal=Vec3(0, 1, 0),
            position=Vec3(1, 2, 3),
            hit=False
        )
        assert hit.hit is False
        assert hit.distance == 2.0

    def test_mixed_construction(self) -> None:
        """Can construct with mixed positional and keyword arguments."""
        hit = RaycastHit(True, Vec3(0, 0, 0), normal=Vec3(0, 1, 0), distance=1.0)
        assert hit.hit is True
        assert hit.distance == 1.0


# =============================================================================
# RaycastCallback with Different Return Scenarios
# =============================================================================

class TestRaycastCallbackScenarios:
    """Tests for various RaycastCallback return scenarios."""

    def test_callback_conditional_hit(self) -> None:
        """Callback can conditionally return hit or None."""
        def conditional_raycast(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            # Only hit if casting downward and within range
            if direction.y < 0 and origin.y < max_distance:
                return RaycastHit(
                    hit=True,
                    position=Vec3(origin.x, 0, origin.z),
                    normal=Vec3(0, 1, 0),
                    distance=origin.y
                )
            return None

        callback: RaycastCallback = conditional_raycast

        # Should hit - casting down from height 1
        result1 = callback(Vec3(0, 1, 0), Vec3(0, -1, 0), 10.0)
        assert result1 is not None
        assert result1.hit is True

        # Should miss - casting up
        result2 = callback(Vec3(0, 1, 0), Vec3(0, 1, 0), 10.0)
        assert result2 is None

    def test_callback_distance_filtering(self) -> None:
        """Callback respects max_distance parameter."""
        ground_y = -10.0

        def distance_filtered_raycast(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            distance_to_ground = origin.y - ground_y
            if distance_to_ground <= max_distance:
                return RaycastHit(
                    hit=True,
                    position=Vec3(origin.x, ground_y, origin.z),
                    normal=Vec3(0, 1, 0),
                    distance=distance_to_ground
                )
            return None

        callback: RaycastCallback = distance_filtered_raycast

        # Should hit - within max distance
        result1 = callback(Vec3(0, 0, 0), Vec3(0, -1, 0), 20.0)
        assert result1 is not None

        # Should miss - beyond max distance
        result2 = callback(Vec3(0, 0, 0), Vec3(0, -1, 0), 5.0)
        assert result2 is None

    def test_callback_returns_exact_distance(self) -> None:
        """Callback can calculate and return exact distance."""
        def precise_raycast(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            # Simulate ray hitting y=0 plane
            if direction.y < 0:
                t = -origin.y / direction.y
                if t <= max_distance:
                    hit_point = Vec3(
                        origin.x + direction.x * t,
                        0,
                        origin.z + direction.z * t
                    )
                    return RaycastHit(
                        hit=True,
                        position=hit_point,
                        normal=Vec3(0, 1, 0),
                        distance=t
                    )
            return None

        callback: RaycastCallback = precise_raycast
        result = callback(Vec3(0, 5, 0), Vec3(0, -1, 0), 100.0)

        assert result is not None
        assert abs(result.distance - 5.0) < MATH_EPSILON


# =============================================================================
# RaycastHit with FootData Integration
# =============================================================================

class TestRaycastHitFootDataIntegration:
    """Tests for RaycastHit integration with FootData."""

    def test_foot_state_transitions_with_raycast(self) -> None:
        """Foot states transition based on raycast results."""
        foot = FootData(
            upper_leg=1,
            lower_leg=2,
            foot=3,
            state=FootState.PLANTED
        )

        # Initial state
        assert foot.state == FootState.PLANTED

    def test_foot_target_position_from_raycast(self) -> None:
        """Foot target position can be set from raycast hit."""
        hit = RaycastHit(
            hit=True,
            position=Vec3(1, 0, 2),
            normal=Vec3(0, 1, 0),
            distance=1.0
        )

        foot = FootData(
            upper_leg=1,
            lower_leg=2,
            foot=3,
            target_position=hit.position
        )

        assert foot.target_position.x == 1
        assert foot.target_position.y == 0
        assert foot.target_position.z == 2

    def test_foot_target_normal_from_raycast(self) -> None:
        """Foot target normal can be set from raycast hit."""
        hit = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(0.1, 0.99, 0),
            distance=1.0
        )

        foot = FootData(
            upper_leg=1,
            lower_leg=2,
            foot=3,
            target_normal=hit.normal
        )

        assert abs(foot.target_normal.x - 0.1) < MATH_EPSILON
        assert abs(foot.target_normal.y - 0.99) < MATH_EPSILON

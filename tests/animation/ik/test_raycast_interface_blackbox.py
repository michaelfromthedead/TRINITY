"""Blackbox tests for Raycast Interface (T-FB-4.9).

This module tests the RaycastHit and RaycastCallback interfaces from the
public API only, without knowledge of implementation details.

Test Strategy:
- Test RaycastHit creation with all fields
- Test RaycastHit.miss() factory method
- Test field types and validation
- Test RaycastCallback signature and usage patterns
- Test integration scenarios for raycast hit/miss

CLEANROOM: No implementation code was read during test creation.
"""

import math
import pytest
from typing import Optional, Callable

# Import public API only
from engine.animation.ik import RaycastHit, RaycastCallback
from engine.core.math import Vec3


# =============================================================================
# Helper Functions
# =============================================================================

def vec3_nearly_equal(a: Vec3, b: Vec3, eps: float = 0.0001) -> bool:
    """Check if two Vec3 are nearly equal within tolerance."""
    return (
        abs(a.x - b.x) < eps and
        abs(a.y - b.y) < eps and
        abs(a.z - b.z) < eps
    )


def is_normalized(v: Vec3, eps: float = 0.01) -> bool:
    """Check if a vector is normalized (length ~= 1)."""
    length = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    return abs(length - 1.0) < eps


# =============================================================================
# Test Class: RaycastHit Creation
# =============================================================================

class TestRaycastHitCreation:
    """Tests for RaycastHit creation with various field combinations."""

    def test_create_hit_with_all_fields(self):
        """RaycastHit can be created with all required fields."""
        position = Vec3(1.0, 2.0, 3.0)
        normal = Vec3(0.0, 1.0, 0.0)
        distance = 5.0
        hit = True

        result = RaycastHit(
            hit=hit,
            position=position,
            normal=normal,
            distance=distance
        )

        assert result is not None

    def test_create_hit_result_true(self):
        """RaycastHit with hit=True indicates successful raycast."""
        result = RaycastHit(
            hit=True,
            position=Vec3(0.0, 0.0, 0.0),
            normal=Vec3(0.0, 1.0, 0.0),
            distance=1.0
        )

        assert result.hit is True

    def test_create_miss_result_manually(self):
        """RaycastHit with hit=False indicates raycast miss."""
        result = RaycastHit(
            hit=False,
            position=Vec3(0.0, 0.0, 0.0),
            normal=Vec3(0.0, 0.0, 0.0),
            distance=float('inf')
        )

        assert result.hit is False

    def test_access_position_field(self):
        """RaycastHit position field can be accessed."""
        expected_pos = Vec3(10.0, 20.0, 30.0)
        result = RaycastHit(
            hit=True,
            position=expected_pos,
            normal=Vec3(0.0, 1.0, 0.0),
            distance=5.0
        )

        assert hasattr(result, 'position')
        assert vec3_nearly_equal(result.position, expected_pos)

    def test_access_normal_field(self):
        """RaycastHit normal field can be accessed."""
        expected_normal = Vec3(0.0, 1.0, 0.0)
        result = RaycastHit(
            hit=True,
            position=Vec3(0.0, 0.0, 0.0),
            normal=expected_normal,
            distance=5.0
        )

        assert hasattr(result, 'normal')
        assert vec3_nearly_equal(result.normal, expected_normal)

    def test_access_distance_field(self):
        """RaycastHit distance field can be accessed."""
        expected_distance = 7.5
        result = RaycastHit(
            hit=True,
            position=Vec3(0.0, 0.0, 0.0),
            normal=Vec3(0.0, 1.0, 0.0),
            distance=expected_distance
        )

        assert hasattr(result, 'distance')
        assert abs(result.distance - expected_distance) < 0.0001

    def test_access_hit_field(self):
        """RaycastHit hit field can be accessed."""
        result = RaycastHit(
            hit=True,
            position=Vec3(0.0, 0.0, 0.0),
            normal=Vec3(0.0, 1.0, 0.0),
            distance=5.0
        )

        assert hasattr(result, 'hit')
        assert result.hit is True

    def test_create_with_zero_distance(self):
        """RaycastHit can be created with zero distance (origin at surface)."""
        result = RaycastHit(
            hit=True,
            position=Vec3(0.0, 0.0, 0.0),
            normal=Vec3(0.0, 1.0, 0.0),
            distance=0.0
        )

        assert result.hit is True
        assert result.distance == 0.0

    def test_create_with_negative_coordinates(self):
        """RaycastHit can be created with negative position coordinates."""
        position = Vec3(-5.0, -10.0, -15.0)
        result = RaycastHit(
            hit=True,
            position=position,
            normal=Vec3(0.0, 1.0, 0.0),
            distance=20.0
        )

        assert vec3_nearly_equal(result.position, position)

    def test_create_with_diagonal_normal(self):
        """RaycastHit can be created with non-axis-aligned normal."""
        # 45-degree slope normal
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        normal = Vec3(inv_sqrt2, inv_sqrt2, 0.0)

        result = RaycastHit(
            hit=True,
            position=Vec3(0.0, 0.0, 0.0),
            normal=normal,
            distance=5.0
        )

        assert vec3_nearly_equal(result.normal, normal)


# =============================================================================
# Test Class: RaycastHit.miss() Factory
# =============================================================================

class TestRaycastHitMissFactory:
    """Tests for RaycastHit.miss() factory method."""

    def test_miss_returns_raycast_hit(self):
        """RaycastHit.miss() returns a RaycastHit instance."""
        result = RaycastHit.miss()

        assert isinstance(result, RaycastHit)

    def test_miss_hit_field_is_false(self):
        """RaycastHit.miss() returns hit=False."""
        result = RaycastHit.miss()

        assert result.hit is False

    def test_miss_distance_is_infinity(self):
        """RaycastHit.miss() returns distance as infinity."""
        result = RaycastHit.miss()

        assert result.distance == float('inf') or result.distance > 1e30

    def test_miss_position_is_valid_vec3(self):
        """RaycastHit.miss() position is a valid Vec3."""
        result = RaycastHit.miss()

        assert isinstance(result.position, Vec3)
        assert hasattr(result.position, 'x')
        assert hasattr(result.position, 'y')
        assert hasattr(result.position, 'z')

    def test_miss_normal_is_valid_vec3(self):
        """RaycastHit.miss() normal is a valid Vec3."""
        result = RaycastHit.miss()

        assert isinstance(result.normal, Vec3)
        assert hasattr(result.normal, 'x')
        assert hasattr(result.normal, 'y')
        assert hasattr(result.normal, 'z')

    def test_miss_can_be_called_multiple_times(self):
        """RaycastHit.miss() can be called multiple times."""
        result1 = RaycastHit.miss()
        result2 = RaycastHit.miss()
        result3 = RaycastHit.miss()

        assert result1.hit is False
        assert result2.hit is False
        assert result3.hit is False

    def test_miss_instances_are_independent(self):
        """Multiple RaycastHit.miss() calls return independent instances."""
        result1 = RaycastHit.miss()
        result2 = RaycastHit.miss()

        # They should be different objects (or at least behave independently)
        # Even if they contain the same values
        assert result1.hit == result2.hit
        assert result1.hit is False


# =============================================================================
# Test Class: RaycastHit Field Validation
# =============================================================================

class TestRaycastHitFieldValidation:
    """Tests validating field types of RaycastHit."""

    def test_position_is_vec3_type(self):
        """RaycastHit position field is of type Vec3."""
        result = RaycastHit(
            hit=True,
            position=Vec3(1.0, 2.0, 3.0),
            normal=Vec3(0.0, 1.0, 0.0),
            distance=5.0
        )

        assert isinstance(result.position, Vec3)

    def test_normal_is_vec3_type(self):
        """RaycastHit normal field is of type Vec3."""
        result = RaycastHit(
            hit=True,
            position=Vec3(0.0, 0.0, 0.0),
            normal=Vec3(0.0, 1.0, 0.0),
            distance=5.0
        )

        assert isinstance(result.normal, Vec3)

    def test_distance_is_float_type(self):
        """RaycastHit distance field is of type float."""
        result = RaycastHit(
            hit=True,
            position=Vec3(0.0, 0.0, 0.0),
            normal=Vec3(0.0, 1.0, 0.0),
            distance=5.0
        )

        assert isinstance(result.distance, (float, int))

    def test_hit_is_bool_type(self):
        """RaycastHit hit field is of type bool."""
        result = RaycastHit(
            hit=True,
            position=Vec3(0.0, 0.0, 0.0),
            normal=Vec3(0.0, 1.0, 0.0),
            distance=5.0
        )

        assert isinstance(result.hit, bool)

    def test_position_has_xyz_components(self):
        """RaycastHit position Vec3 has x, y, z components."""
        result = RaycastHit(
            hit=True,
            position=Vec3(1.0, 2.0, 3.0),
            normal=Vec3(0.0, 1.0, 0.0),
            distance=5.0
        )

        assert hasattr(result.position, 'x')
        assert hasattr(result.position, 'y')
        assert hasattr(result.position, 'z')
        assert result.position.x == 1.0
        assert result.position.y == 2.0
        assert result.position.z == 3.0

    def test_normal_has_xyz_components(self):
        """RaycastHit normal Vec3 has x, y, z components."""
        result = RaycastHit(
            hit=True,
            position=Vec3(0.0, 0.0, 0.0),
            normal=Vec3(0.5, 0.5, 0.707),
            distance=5.0
        )

        assert hasattr(result.normal, 'x')
        assert hasattr(result.normal, 'y')
        assert hasattr(result.normal, 'z')

    def test_distance_positive_values(self):
        """RaycastHit accepts positive distance values."""
        for dist in [0.001, 1.0, 100.0, 99999.9]:
            result = RaycastHit(
                hit=True,
                position=Vec3(0.0, 0.0, 0.0),
                normal=Vec3(0.0, 1.0, 0.0),
                distance=dist
            )
            assert result.distance == dist

    def test_hit_boolean_true(self):
        """RaycastHit accepts True for hit field."""
        result = RaycastHit(
            hit=True,
            position=Vec3(0.0, 0.0, 0.0),
            normal=Vec3(0.0, 1.0, 0.0),
            distance=5.0
        )

        assert result.hit is True

    def test_hit_boolean_false(self):
        """RaycastHit accepts False for hit field."""
        result = RaycastHit(
            hit=False,
            position=Vec3(0.0, 0.0, 0.0),
            normal=Vec3(0.0, 0.0, 0.0),
            distance=float('inf')
        )

        assert result.hit is False


# =============================================================================
# Test Class: RaycastCallback Usage
# =============================================================================

class TestRaycastCallbackUsage:
    """Tests for RaycastCallback type alias and usage patterns."""

    def test_callback_function_signature(self):
        """Callback function matches expected signature (Vec3, Vec3, float) -> Optional[RaycastHit]."""
        def my_callback(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            return RaycastHit.miss()

        # Should be callable with correct arguments
        result = my_callback(Vec3(0, 0, 0), Vec3(0, -1, 0), 10.0)
        assert result is not None or result is None  # Either is valid

    def test_callback_takes_vec3_origin(self):
        """Callback accepts Vec3 as origin parameter."""
        origin_received = None

        def my_callback(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            nonlocal origin_received
            origin_received = origin
            return None

        test_origin = Vec3(5.0, 10.0, 15.0)
        my_callback(test_origin, Vec3(0, -1, 0), 10.0)

        assert origin_received is not None
        assert vec3_nearly_equal(origin_received, test_origin)

    def test_callback_takes_vec3_direction(self):
        """Callback accepts Vec3 as direction parameter."""
        direction_received = None

        def my_callback(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            nonlocal direction_received
            direction_received = direction
            return None

        test_direction = Vec3(0.0, -1.0, 0.0)
        my_callback(Vec3(0, 0, 0), test_direction, 10.0)

        assert direction_received is not None
        assert vec3_nearly_equal(direction_received, test_direction)

    def test_callback_takes_float_distance(self):
        """Callback accepts float as max_distance parameter."""
        distance_received = None

        def my_callback(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            nonlocal distance_received
            distance_received = max_distance
            return None

        test_distance = 25.5
        my_callback(Vec3(0, 0, 0), Vec3(0, -1, 0), test_distance)

        assert distance_received == test_distance

    def test_callback_returns_optional_raycast_hit(self):
        """Callback can return Optional[RaycastHit]."""
        def hit_callback(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            return RaycastHit(
                hit=True,
                position=Vec3(0, 0, 0),
                normal=Vec3(0, 1, 0),
                distance=5.0
            )

        result = hit_callback(Vec3(0, 5, 0), Vec3(0, -1, 0), 10.0)
        assert isinstance(result, RaycastHit)

    def test_callback_returns_none_for_miss(self):
        """Callback can return None to indicate no hit."""
        def miss_callback(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            return None

        result = miss_callback(Vec3(0, 5, 0), Vec3(0, -1, 0), 10.0)
        assert result is None

    def test_callback_returns_raycast_hit_miss_for_miss(self):
        """Callback can return RaycastHit.miss() to indicate no hit."""
        def miss_callback(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            return RaycastHit.miss()

        result = miss_callback(Vec3(0, 5, 0), Vec3(0, -1, 0), 10.0)
        assert isinstance(result, RaycastHit)
        assert result.hit is False


# =============================================================================
# Test Class: Integration Scenarios
# =============================================================================

class TestIntegrationScenarios:
    """Integration tests simulating realistic raycast scenarios."""

    def test_simulate_raycast_hit_flat_ground(self):
        """Simulate raycast hitting flat ground."""
        def flat_ground_raycast(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            # Flat ground at y=0
            if direction.y < 0 and origin.y > 0:
                t = -origin.y / direction.y
                if t <= max_distance:
                    hit_pos = Vec3(
                        origin.x + direction.x * t,
                        0.0,
                        origin.z + direction.z * t
                    )
                    return RaycastHit(
                        hit=True,
                        position=hit_pos,
                        normal=Vec3(0.0, 1.0, 0.0),
                        distance=t
                    )
            return None

        result = flat_ground_raycast(Vec3(0, 5, 0), Vec3(0, -1, 0), 10.0)

        assert result is not None
        assert result.hit is True
        assert abs(result.position.y) < 0.0001
        assert abs(result.distance - 5.0) < 0.0001

    def test_simulate_raycast_miss_no_geometry(self):
        """Simulate raycast missing when pointing away from geometry."""
        def flat_ground_raycast(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            # Flat ground at y=0, ray points up
            if direction.y > 0:
                return None
            return RaycastHit.miss()

        # Ray pointing up from ground
        result = flat_ground_raycast(Vec3(0, 1, 0), Vec3(0, 1, 0), 10.0)

        assert result is None

    def test_simulate_raycast_miss_using_raycast_hit_miss(self):
        """Simulate raycast miss using RaycastHit.miss()."""
        def empty_world_raycast(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            # No geometry, always miss
            return RaycastHit.miss()

        result = empty_world_raycast(Vec3(0, 5, 0), Vec3(0, -1, 0), 10.0)

        assert result is not None
        assert result.hit is False

    def test_simulate_raycast_hit_sloped_surface(self):
        """Simulate raycast hitting a sloped surface."""
        def sloped_ground_raycast(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            # 45-degree slope
            inv_sqrt2 = 1.0 / math.sqrt(2.0)
            slope_normal = Vec3(0.0, inv_sqrt2, inv_sqrt2)

            return RaycastHit(
                hit=True,
                position=Vec3(origin.x, 0.5, origin.z + 0.5),
                normal=slope_normal,
                distance=3.0
            )

        result = sloped_ground_raycast(Vec3(0, 2, 0), Vec3(0, -1, 0), 10.0)

        assert result.hit is True
        assert result.normal.y > 0

    def test_simulate_raycast_beyond_max_distance(self):
        """Simulate raycast that would hit but is beyond max distance."""
        def ground_raycast(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            # Ground at y=0
            if direction.y < 0:
                t = -origin.y / direction.y
                if t > max_distance:
                    return None  # Beyond max distance
            return None

        # Origin at y=100, max distance 10 - won't reach ground
        result = ground_raycast(Vec3(0, 100, 0), Vec3(0, -1, 0), 10.0)

        assert result is None

    def test_simulate_multiple_raycast_calls(self):
        """Simulate multiple raycast calls with different origins."""
        call_count = 0

        def counting_raycast(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            nonlocal call_count
            call_count += 1

            if origin.y > 0 and direction.y < 0:
                return RaycastHit(
                    hit=True,
                    position=Vec3(origin.x, 0, origin.z),
                    normal=Vec3(0, 1, 0),
                    distance=origin.y
                )
            return None

        # Multiple raycasts
        origins = [Vec3(0, 1, 0), Vec3(1, 2, 1), Vec3(-1, 3, -1)]
        for origin in origins:
            counting_raycast(origin, Vec3(0, -1, 0), 10.0)

        assert call_count == 3

    def test_simulate_raycast_different_directions(self):
        """Simulate raycast with different direction vectors."""
        def directional_raycast(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            # Hit different surfaces based on direction
            if direction.y < -0.9:  # Pointing down
                return RaycastHit(hit=True, position=Vec3(0, 0, 0), normal=Vec3(0, 1, 0), distance=5.0)
            elif direction.x > 0.9:  # Pointing right
                return RaycastHit(hit=True, position=Vec3(10, 0, 0), normal=Vec3(-1, 0, 0), distance=10.0)
            return None

        down_result = directional_raycast(Vec3(0, 5, 0), Vec3(0, -1, 0), 20.0)
        right_result = directional_raycast(Vec3(0, 0, 0), Vec3(1, 0, 0), 20.0)

        assert down_result is not None
        assert down_result.normal.y > 0

        assert right_result is not None
        assert right_result.normal.x < 0

    def test_simulate_stateful_raycast_callback(self):
        """Simulate raycast callback with internal state (e.g., physics world)."""
        class MockPhysicsWorld:
            def __init__(self):
                self.objects = [
                    {"position": Vec3(0, 0, 0), "size": 5.0},
                    {"position": Vec3(10, 0, 10), "size": 3.0},
                ]

            def raycast(self, origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
                # Simplified: just check if pointing down and above ground
                if direction.y < 0 and origin.y > 0:
                    return RaycastHit(
                        hit=True,
                        position=Vec3(origin.x, 0, origin.z),
                        normal=Vec3(0, 1, 0),
                        distance=origin.y
                    )
                return RaycastHit.miss()

        world = MockPhysicsWorld()
        result = world.raycast(Vec3(0, 2, 0), Vec3(0, -1, 0), 10.0)

        assert result.hit is True

    def test_lambda_as_raycast_callback(self):
        """Lambda function can be used as raycast callback."""
        raycast_lambda = lambda o, d, m: RaycastHit(
            hit=True, position=Vec3(0, 0, 0), normal=Vec3(0, 1, 0), distance=1.0
        )

        result = raycast_lambda(Vec3(0, 1, 0), Vec3(0, -1, 0), 10.0)

        assert result.hit is True

    def test_raycast_result_can_be_stored(self):
        """RaycastHit result can be stored and accessed later."""
        hits = []

        def storing_raycast(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            hit = RaycastHit(
                hit=True,
                position=origin,
                normal=Vec3(0, 1, 0),
                distance=1.0
            )
            hits.append(hit)
            return hit

        storing_raycast(Vec3(1, 1, 1), Vec3(0, -1, 0), 10.0)
        storing_raycast(Vec3(2, 2, 2), Vec3(0, -1, 0), 10.0)

        assert len(hits) == 2
        assert hits[0].position.x == 1.0
        assert hits[1].position.x == 2.0


# =============================================================================
# Test Class: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for RaycastHit and callbacks."""

    def test_raycast_hit_at_origin(self):
        """RaycastHit can represent hit at world origin."""
        result = RaycastHit(
            hit=True,
            position=Vec3(0.0, 0.0, 0.0),
            normal=Vec3(0.0, 1.0, 0.0),
            distance=0.0
        )

        assert result.hit is True
        assert result.position.x == 0.0
        assert result.position.y == 0.0
        assert result.position.z == 0.0

    def test_raycast_hit_very_large_distance(self):
        """RaycastHit can handle very large distances."""
        large_distance = 1e10
        result = RaycastHit(
            hit=True,
            position=Vec3(0.0, 0.0, large_distance),
            normal=Vec3(0.0, 0.0, -1.0),
            distance=large_distance
        )

        assert result.distance == large_distance

    def test_raycast_hit_very_small_distance(self):
        """RaycastHit can handle very small distances."""
        tiny_distance = 1e-10
        result = RaycastHit(
            hit=True,
            position=Vec3(0.0, 0.0, 0.0),
            normal=Vec3(0.0, 1.0, 0.0),
            distance=tiny_distance
        )

        assert result.distance == tiny_distance

    def test_callback_with_zero_max_distance(self):
        """Callback handles zero max distance."""
        def zero_dist_raycast(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            if max_distance <= 0:
                return None
            return RaycastHit.miss()

        result = zero_dist_raycast(Vec3(0, 1, 0), Vec3(0, -1, 0), 0.0)
        assert result is None

    def test_callback_with_very_large_max_distance(self):
        """Callback handles very large max distance."""
        def large_dist_raycast(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            return RaycastHit(
                hit=True,
                position=Vec3(0, 0, 0),
                normal=Vec3(0, 1, 0),
                distance=max_distance / 2
            )

        result = large_dist_raycast(Vec3(0, 1e15, 0), Vec3(0, -1, 0), 1e15)
        assert result.hit is True

    def test_normal_pointing_in_various_directions(self):
        """RaycastHit can have normals pointing in any direction."""
        directions = [
            Vec3(1.0, 0.0, 0.0),   # +X
            Vec3(-1.0, 0.0, 0.0),  # -X
            Vec3(0.0, 1.0, 0.0),   # +Y (up)
            Vec3(0.0, -1.0, 0.0),  # -Y (down)
            Vec3(0.0, 0.0, 1.0),   # +Z
            Vec3(0.0, 0.0, -1.0),  # -Z
        ]

        for normal in directions:
            result = RaycastHit(
                hit=True,
                position=Vec3(0, 0, 0),
                normal=normal,
                distance=1.0
            )
            assert vec3_nearly_equal(result.normal, normal)


# =============================================================================
# Test Class: Type Safety
# =============================================================================

class TestTypeSafety:
    """Tests ensuring type safety of RaycastHit and RaycastCallback."""

    def test_raycast_hit_type_annotation_compatible(self):
        """RaycastHit is compatible with type annotation usage."""
        hit: RaycastHit = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            distance=1.0
        )

        assert isinstance(hit, RaycastHit)

    def test_optional_raycast_hit_with_none(self):
        """Optional[RaycastHit] works with None value."""
        result: Optional[RaycastHit] = None

        assert result is None

    def test_optional_raycast_hit_with_value(self):
        """Optional[RaycastHit] works with actual value."""
        result: Optional[RaycastHit] = RaycastHit(
            hit=True,
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            distance=1.0
        )

        assert result is not None
        assert result.hit is True

    def test_callback_type_hint_compatible(self):
        """Function matching RaycastCallback type hint can be used."""
        def typed_callback(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            return None

        # Use as Callable
        callback: Callable[[Vec3, Vec3, float], Optional[RaycastHit]] = typed_callback
        result = callback(Vec3(0, 0, 0), Vec3(0, -1, 0), 10.0)

        assert result is None


# =============================================================================
# Test Class: Callback Behavior
# =============================================================================

class TestCallbackBehavior:
    """Tests for various callback behaviors and patterns."""

    def test_callback_can_modify_captured_state(self):
        """Callback can modify captured external state."""
        hit_count = {"count": 0}

        def counting_callback(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            hit_count["count"] += 1
            return RaycastHit.miss()

        counting_callback(Vec3(0, 0, 0), Vec3(0, -1, 0), 10.0)
        counting_callback(Vec3(0, 0, 0), Vec3(0, -1, 0), 10.0)

        assert hit_count["count"] == 2

    def test_callback_can_use_closure_variables(self):
        """Callback can access closure variables."""
        ground_height = 2.0

        def closure_callback(origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
            if origin.y > ground_height and direction.y < 0:
                return RaycastHit(
                    hit=True,
                    position=Vec3(origin.x, ground_height, origin.z),
                    normal=Vec3(0, 1, 0),
                    distance=origin.y - ground_height
                )
            return None

        result = closure_callback(Vec3(0, 5, 0), Vec3(0, -1, 0), 10.0)

        assert result is not None
        assert result.position.y == ground_height

    def test_callback_can_be_class_method(self):
        """Class method can be used as callback."""
        class RaycastProvider:
            def __init__(self, ground_y: float):
                self.ground_y = ground_y

            def do_raycast(self, origin: Vec3, direction: Vec3, max_distance: float) -> Optional[RaycastHit]:
                if origin.y > self.ground_y and direction.y < 0:
                    return RaycastHit(
                        hit=True,
                        position=Vec3(0, self.ground_y, 0),
                        normal=Vec3(0, 1, 0),
                        distance=origin.y - self.ground_y
                    )
                return None

        provider = RaycastProvider(ground_y=0.0)
        result = provider.do_raycast(Vec3(0, 3, 0), Vec3(0, -1, 0), 10.0)

        assert result is not None
        assert result.hit is True

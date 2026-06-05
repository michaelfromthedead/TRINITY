"""
Whitebox tests for engine/simulation/character/ground_detection.py

Tests GroundDetector, GroundInfo, LedgeInfo, and coyote time mechanics.
"""

import math
import time
import pytest
from unittest.mock import MagicMock, patch
from engine.simulation.character.ground_detection import (
    GroundDetector,
    GroundInfo,
    GroundType,
    LedgeInfo,
)
from engine.simulation.character.character_controller import (
    CollisionHit,
    PhysicsWorldInterface,
    SweepResult,
    Vector3,
)


class TestGroundType:
    """Tests for GroundType enum."""

    def test_solid_value(self):
        """SOLID should have expected value."""
        assert GroundType.SOLID.value == "solid"

    def test_slope_value(self):
        """SLOPE should have expected value."""
        assert GroundType.SLOPE.value == "slope"

    def test_steep_value(self):
        """STEEP should have expected value."""
        assert GroundType.STEEP.value == "steep"

    def test_ledge_value(self):
        """LEDGE should have expected value."""
        assert GroundType.LEDGE.value == "ledge"

    def test_water_value(self):
        """WATER should have expected value."""
        assert GroundType.WATER.value == "water"

    def test_platform_value(self):
        """PLATFORM should have expected value."""
        assert GroundType.PLATFORM.value == "platform"

    def test_none_value(self):
        """NONE should have expected value."""
        assert GroundType.NONE.value == "none"


class TestGroundInfo:
    """Tests for GroundInfo dataclass."""

    def test_default_construction(self):
        """Default GroundInfo should indicate not grounded."""
        info = GroundInfo()
        assert info.is_grounded is False
        assert info.ground_type == GroundType.NONE
        assert info.is_walkable is False
        assert info.distance == float("inf")

    def test_custom_construction(self):
        """GroundInfo should accept custom values."""
        info = GroundInfo(
            is_grounded=True,
            normal=Vector3(0.0, 1.0, 0.0),
            material="concrete",
            distance=0.05,
            ground_type=GroundType.SOLID,
            slope_angle=0.0,
            is_walkable=True,
            friction=0.8,
        )
        assert info.is_grounded is True
        assert info.material == "concrete"
        assert info.ground_type == GroundType.SOLID
        assert info.is_walkable is True


class TestLedgeInfo:
    """Tests for LedgeInfo dataclass."""

    def test_default_construction(self):
        """Default LedgeInfo should indicate no ledge."""
        info = LedgeInfo()
        assert info.has_ledge is False
        assert info.is_climbable is False
        assert info.climb_height == 0.0

    def test_custom_construction(self):
        """LedgeInfo should accept custom values."""
        info = LedgeInfo(
            has_ledge=True,
            ledge_position=Vector3(1.0, 2.0, 3.0),
            climb_height=1.5,
            is_climbable=True,
        )
        assert info.has_ledge is True
        assert info.climb_height == 1.5
        assert info.is_climbable is True


class MockGroundPhysicsWorld(PhysicsWorldInterface):
    """Mock physics world for ground detection testing."""

    def __init__(self):
        self.raycast_results = []
        self.sphere_sweep_results = []

    def raycast(self, start, direction, distance, mask=0):
        if self.raycast_results:
            return self.raycast_results.pop(0)
        return None

    def sphere_sweep(self, start, end, radius, mask=0):
        if self.sphere_sweep_results:
            return self.sphere_sweep_results.pop(0)
        return SweepResult()


class TestGroundDetector:
    """Tests for GroundDetector class."""

    def test_construction(self):
        """GroundDetector should be constructible."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        assert detector is not None

    def test_construction_with_custom_dimensions(self):
        """GroundDetector should accept custom character dimensions."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world, character_radius=0.5, character_height=2.0)
        assert detector._radius == 0.5
        assert detector._height == 2.0

    def test_set_probe_distance(self):
        """set_probe_distance should update probe distance."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_probe_distance(0.2)
        assert detector._probe_distance == 0.2

    def test_set_probe_distance_minimum(self):
        """set_probe_distance should enforce minimum."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_probe_distance(-1.0)
        assert detector._probe_distance == 0.01

    def test_set_slope_limit(self):
        """set_slope_limit should update slope limit."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_slope_limit(60.0)
        assert detector._slope_limit == 60.0

    def test_set_slope_limit_clamped_min(self):
        """set_slope_limit should clamp to minimum 0."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_slope_limit(-10.0)
        assert detector._slope_limit == 0.0

    def test_set_slope_limit_clamped_max(self):
        """set_slope_limit should clamp to maximum 90."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_slope_limit(100.0)
        assert detector._slope_limit == 90.0

    def test_set_coyote_time(self):
        """set_coyote_time should update coyote time."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_coyote_time(200.0)
        assert detector._coyote_time_ms == 200.0

    def test_set_coyote_time_minimum(self):
        """set_coyote_time should enforce minimum 0."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_coyote_time(-100.0)
        assert detector._coyote_time_ms == 0.0

    def test_set_jump_buffer_time(self):
        """set_jump_buffer_time should update buffer time."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_jump_buffer_time(150.0)
        assert detector._jump_buffer_time_ms == 150.0

    def test_detect_slope_angle_flat(self):
        """detect_slope_angle should return 0 for flat ground."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        angle = detector.detect_slope_angle(Vector3.up())
        assert angle == pytest.approx(0.0)

    def test_detect_slope_angle_vertical(self):
        """detect_slope_angle should return 90 for vertical wall."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        angle = detector.detect_slope_angle(Vector3(1.0, 0.0, 0.0))
        assert angle == pytest.approx(90.0)

    def test_detect_slope_angle_45_degrees(self):
        """detect_slope_angle should return ~45 for 45 degree slope."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        normal = Vector3(0.707, 0.707, 0.0)  # 45 degree slope
        angle = detector.detect_slope_angle(normal)
        assert angle == pytest.approx(45.0, abs=1.0)

    def test_is_flat_ground_true(self):
        """is_flat_ground should return True for near-flat surfaces."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        assert detector.is_flat_ground(Vector3.up()) is True

    def test_is_flat_ground_false(self):
        """is_flat_ground should return False for slopes."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        normal = Vector3(0.5, 0.866, 0.0)  # ~30 degree slope
        assert detector.is_flat_ground(normal) is False

    def test_is_walkable_slope_flat(self):
        """is_walkable_slope should return True for flat ground."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        assert detector.is_walkable_slope(Vector3.up()) is True

    def test_is_walkable_slope_at_limit(self):
        """is_walkable_slope should return True at exactly slope limit."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_slope_limit(45.0)
        # Create normal for exactly 45 degrees
        angle_rad = math.radians(45.0)
        normal = Vector3(math.sin(angle_rad), math.cos(angle_rad), 0.0)
        assert detector.is_walkable_slope(normal) is True

    def test_is_walkable_slope_over_limit(self):
        """is_walkable_slope should return False over limit."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_slope_limit(45.0)
        # Create normal for 60 degrees
        angle_rad = math.radians(60.0)
        normal = Vector3(math.sin(angle_rad), math.cos(angle_rad), 0.0)
        assert detector.is_walkable_slope(normal) is False

    def test_is_steep_slope(self):
        """is_steep_slope should return True for slopes over limit."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_slope_limit(45.0)
        # Create normal for 60 degrees (steep but not wall)
        angle_rad = math.radians(60.0)
        normal = Vector3(math.sin(angle_rad), math.cos(angle_rad), 0.0)
        assert detector.is_steep_slope(normal) is True

    def test_is_steep_slope_flat(self):
        """is_steep_slope should return False for flat ground."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        assert detector.is_steep_slope(Vector3.up()) is False

    def test_raycast_ground_no_hit(self):
        """raycast_ground should return not grounded when no hit."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        result = detector.raycast_ground(Vector3.zero())
        assert result.is_grounded is False

    def test_raycast_ground_with_hit(self):
        """raycast_ground should return grounded info on hit."""
        world = MockGroundPhysicsWorld()
        world.raycast_results.append(CollisionHit(
            point=Vector3(0.0, -0.05, 0.0),
            normal=Vector3.up(),
            distance=0.1,
            material="concrete",
        ))
        detector = GroundDetector(world)
        result = detector.raycast_ground(Vector3.zero())
        assert result.is_grounded is True
        assert result.material == "concrete"
        assert result.is_walkable is True

    def test_raycast_ground_steep_slope(self):
        """raycast_ground should mark steep slopes as not walkable."""
        world = MockGroundPhysicsWorld()
        # 60 degree slope normal
        angle_rad = math.radians(60.0)
        steep_normal = Vector3(math.sin(angle_rad), math.cos(angle_rad), 0.0)
        world.raycast_results.append(CollisionHit(
            point=Vector3(0.0, -0.05, 0.0),
            normal=steep_normal,
            distance=0.1,
            material="default",
        ))
        detector = GroundDetector(world)
        detector.set_slope_limit(45.0)
        result = detector.raycast_ground(Vector3.zero())
        assert result.is_grounded is True
        assert result.is_walkable is False

    def test_sphere_sweep_ground_no_hit(self):
        """sphere_sweep_ground should return not grounded when no hit."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        result = detector.sphere_sweep_ground(Vector3.zero())
        assert result.is_grounded is False

    def test_sphere_sweep_ground_with_hit(self):
        """sphere_sweep_ground should return grounded info on hit."""
        world = MockGroundPhysicsWorld()
        world.sphere_sweep_results.append(SweepResult(
            hit=True,
            hits=[CollisionHit(
                point=Vector3(0.0, -0.05, 0.0),
                normal=Vector3.up(),
                distance=0.1,
                material="grass",
            )],
            safe_fraction=0.9,
        ))
        detector = GroundDetector(world)
        result = detector.sphere_sweep_ground(Vector3.zero())
        assert result.is_grounded is True
        assert result.material == "grass"

    def test_detect_ground_uses_sphere_sweep(self):
        """detect_ground should use sphere sweep."""
        world = MockGroundPhysicsWorld()
        world.sphere_sweep_results.append(SweepResult(
            hit=True,
            hits=[CollisionHit(
                point=Vector3(0.0, -0.05, 0.0),
                normal=Vector3.up(),
                distance=0.05,
                material="metal",
            )],
            safe_fraction=0.9,
        ))
        detector = GroundDetector(world)
        result = detector.detect_ground(Vector3.zero())
        assert result.is_grounded is True
        assert result.material == "metal"

    def test_detect_ground_falls_back_to_raycast(self):
        """detect_ground should fall back to raycast if sweep fails."""
        world = MockGroundPhysicsWorld()
        # No sphere sweep result, but raycast result
        world.raycast_results.append(CollisionHit(
            point=Vector3(0.0, -0.05, 0.0),
            normal=Vector3.up(),
            distance=0.1,
            material="wood",
        ))
        detector = GroundDetector(world)
        result = detector.detect_ground(Vector3.zero())
        assert result.is_grounded is True
        assert result.material == "wood"

    def test_ground_type_solid(self):
        """Ground type should be SOLID for flat surfaces."""
        world = MockGroundPhysicsWorld()
        world.raycast_results.append(CollisionHit(
            point=Vector3(0.0, -0.05, 0.0),
            normal=Vector3.up(),
            distance=0.1,
            material="default",
        ))
        detector = GroundDetector(world)
        result = detector.raycast_ground(Vector3.zero())
        assert result.ground_type == GroundType.SOLID

    def test_ground_type_slope(self):
        """Ground type should be SLOPE for walkable slopes."""
        world = MockGroundPhysicsWorld()
        # 30 degree slope
        angle_rad = math.radians(30.0)
        normal = Vector3(math.sin(angle_rad), math.cos(angle_rad), 0.0)
        world.raycast_results.append(CollisionHit(
            point=Vector3(0.0, -0.05, 0.0),
            normal=normal,
            distance=0.1,
            material="default",
        ))
        detector = GroundDetector(world)
        detector.set_slope_limit(45.0)
        result = detector.raycast_ground(Vector3.zero())
        assert result.ground_type == GroundType.SLOPE

    def test_ground_type_steep(self):
        """Ground type should be STEEP for non-walkable slopes."""
        world = MockGroundPhysicsWorld()
        # 55 degree slope (over 45 limit but under 60)
        angle_rad = math.radians(55.0)
        normal = Vector3(math.sin(angle_rad), math.cos(angle_rad), 0.0)
        world.raycast_results.append(CollisionHit(
            point=Vector3(0.0, -0.05, 0.0),
            normal=normal,
            distance=0.1,
            material="default",
        ))
        detector = GroundDetector(world)
        detector.set_slope_limit(45.0)
        result = detector.raycast_ground(Vector3.zero())
        assert result.ground_type == GroundType.STEEP

    def test_ground_type_water(self):
        """Ground type should be WATER for water material."""
        world = MockGroundPhysicsWorld()
        world.raycast_results.append(CollisionHit(
            point=Vector3(0.0, -0.05, 0.0),
            normal=Vector3.up(),
            distance=0.1,
            material="water",
        ))
        detector = GroundDetector(world)
        result = detector.raycast_ground(Vector3.zero())
        assert result.ground_type == GroundType.WATER


class TestCoyoteTime:
    """Tests for coyote time mechanics."""

    def test_is_in_coyote_time_disabled(self):
        """is_in_coyote_time should return False when disabled."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_coyote_time(0.0)
        assert detector.is_in_coyote_time() is False

    def test_is_in_coyote_time_never_grounded(self):
        """is_in_coyote_time should return False if never grounded."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_coyote_time(150.0)
        # _last_grounded_time is 0 at init, current time >> 150ms
        assert detector.is_in_coyote_time() is False

    def test_can_jump_when_grounded(self):
        """can_jump should return True when grounded."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        assert detector.can_jump(is_grounded=True) is True

    def test_can_jump_with_coyote_time(self):
        """can_jump should return True during coyote time."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_coyote_time(150.0)
        # Set grounded time to now
        detector._last_grounded_time = time.time() * 1000.0
        assert detector.can_jump(is_grounded=False) is True


class TestJumpBuffer:
    """Tests for jump buffer mechanics."""

    def test_register_jump_input(self):
        """register_jump_input should update last jump input time."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.register_jump_input()
        assert detector._last_jump_input_time > 0

    def test_is_jump_buffered_disabled(self):
        """is_jump_buffered should return False when disabled."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_jump_buffer_time(0.0)
        detector.register_jump_input()
        assert detector.is_jump_buffered() is False

    def test_is_jump_buffered_no_input(self):
        """is_jump_buffered should return False without input."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_jump_buffer_time(100.0)
        # No register_jump_input called
        assert detector.is_jump_buffered() is False

    def test_is_jump_buffered_recent_input(self):
        """is_jump_buffered should return True for recent input."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_jump_buffer_time(100.0)
        detector.register_jump_input()
        assert detector.is_jump_buffered() is True

    def test_clear_jump_buffer(self):
        """clear_jump_buffer should reset buffer."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        detector.set_jump_buffer_time(100.0)
        detector.register_jump_input()
        detector.clear_jump_buffer()
        assert detector.is_jump_buffered() is False


class TestLedgeDetection:
    """Tests for ledge detection."""

    def test_detect_ledge_no_wall(self):
        """detect_ledge should return no ledge when no wall found."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        result = detector.detect_ledge(Vector3.zero(), Vector3.forward())
        assert result.has_ledge is False

    def test_detect_ledge_with_wall_and_surface(self):
        """detect_ledge should return ledge info when wall and surface found."""
        world = MockGroundPhysicsWorld()
        # First raycast finds wall
        world.raycast_results.append(CollisionHit(
            point=Vector3(0.3, 1.4, 0.0),  # Wall hit at head height
            normal=Vector3(0.0, 0.0, -1.0),
            distance=0.3,
        ))
        # Second raycast finds ledge surface
        world.raycast_results.append(CollisionHit(
            point=Vector3(0.4, 1.8, 0.0),  # Ledge surface
            normal=Vector3.up(),
            distance=0.4,
        ))
        detector = GroundDetector(world)
        result = detector.detect_ledge(Vector3.zero(), Vector3.forward())
        assert result.has_ledge is True
        assert result.is_climbable is True


class TestMultiPointDetection:
    """Tests for multi-point ground detection."""

    def test_detect_ground_multi_point_no_hits(self):
        """detect_ground_multi_point should return not grounded when no hits."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        combined, probes = detector.detect_ground_multi_point(
            Vector3.zero(), Vector3.forward()
        )
        assert combined.is_grounded is False
        assert len(probes) >= 4  # At least 4 probes

    def test_detect_ground_multi_point_all_hits(self):
        """detect_ground_multi_point should combine results from all hits."""
        world = MockGroundPhysicsWorld()
        # Add 5 raycast results for 5 probes
        for _ in range(5):
            world.raycast_results.append(CollisionHit(
                point=Vector3(0.0, -0.05, 0.0),
                normal=Vector3.up(),
                distance=0.1,
                material="concrete",
            ))
        detector = GroundDetector(world)
        combined, probes = detector.detect_ground_multi_point(
            Vector3.zero(), Vector3.forward()
        )
        assert combined.is_grounded is True
        assert combined.material == "concrete"


class TestEdgeDetection:
    """Tests for edge/drop-off detection."""

    def test_detect_edge_not_grounded(self):
        """detect_edge should return False when not grounded."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        is_edge, drop = detector.detect_edge(
            Vector3.zero(), Vector3.forward()
        )
        assert is_edge is False
        assert drop == 0.0

    def test_detect_edge_with_ground_ahead(self):
        """detect_edge should return False when ground ahead."""
        world = MockGroundPhysicsWorld()
        # Current ground
        world.raycast_results.append(CollisionHit(
            point=Vector3(0.0, -0.05, 0.0),
            normal=Vector3.up(),
            distance=0.1,
        ))
        # Ahead ground at same level
        world.raycast_results.append(CollisionHit(
            point=Vector3(1.0, -0.05, 0.0),
            normal=Vector3.up(),
            distance=0.1,
        ))
        detector = GroundDetector(world)
        is_edge, drop = detector.detect_edge(
            Vector3.zero(), Vector3.forward(), check_distance=1.0
        )
        assert is_edge is False

    def test_detect_edge_with_drop(self):
        """detect_edge should return True when significant drop ahead."""
        world = MockGroundPhysicsWorld()
        # Current ground
        world.raycast_results.append(CollisionHit(
            point=Vector3(0.0, 0.0, 0.0),
            normal=Vector3.up(),
            distance=0.05,
        ))
        # Ahead ground much lower
        world.raycast_results.append(CollisionHit(
            point=Vector3(1.0, -2.0, 0.0),
            normal=Vector3.up(),
            distance=2.05,
        ))
        detector = GroundDetector(world)
        is_edge, drop = detector.detect_edge(
            Vector3.zero(), Vector3.forward(), check_distance=1.0
        )
        assert is_edge is True
        assert drop > 0.5


class TestFriction:
    """Tests for friction coefficient retrieval."""

    def test_friction_default(self):
        """Default friction should be 0.6."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        friction = detector._get_friction("default")
        assert friction == 0.6

    def test_friction_ice(self):
        """Ice friction should be very low."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        friction = detector._get_friction("ice")
        assert friction == 0.05

    def test_friction_unknown(self):
        """Unknown material should return default friction."""
        world = MockGroundPhysicsWorld()
        detector = GroundDetector(world)
        friction = detector._get_friction("unknown_material")
        assert friction == 0.6

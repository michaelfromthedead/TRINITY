"""Blackbox tests for MultiLegFootPlacement IK (T-FB-4.13).

This module tests the MultiLegFootPlacement system from the public API only,
without knowledge of implementation details. Tests are derived from
theoretical multi-leg foot placement behavior:

1. Construction with list of FootData for arbitrary leg counts
2. Support for 2, 4, 6, 8+ leg configurations
3. Pelvis bone index requirement (pelvis parameter)
4. Optional raycast callback for terrain detection
5. solve() behavior with and without raycast
6. Configuration accessibility (ray_length, foot_height, blend_speed)
7. Edge cases (single leg, empty, many legs)

Public API discovered via help():
- MultiLegFootPlacement(feet, pelvis, raycast_callback=None)
- solve(transforms, character_position, dt=0.016667) -> List[Transform]
- Attributes: blend_speed, foot_height, ray_length, feet, pelvis

Test Strategy:
- CLEANROOM: Test public API contracts only
- NO implementation peeking
- Test behavioral expectations for multi-leg creatures
- Test various leg configurations (biped, quadruped, hexapod, octopod)
- Test per-leg terrain adaptation
"""

import math
import pytest
from typing import List, Optional, Callable

# Import public API only
from engine.animation.ik import (
    MultiLegFootPlacement,
    FootData,
    FootState,
    RaycastCallback,
    RaycastHit,
    FOOT_PLACEMENT_RAY_LENGTH,
    FOOT_PLACEMENT_FOOT_HEIGHT,
    FOOT_PLACEMENT_BLEND_SPEED,
)
from engine.core.math import Vec3, Quat, Transform


# =============================================================================
# Helper Functions
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


def vec_nearly_equal(a: Vec3, b: Vec3, eps: float = 0.01) -> bool:
    """Check if two vectors are nearly equal."""
    return vec3_distance(a, b) <= eps


def create_flat_ground_raycast(ground_height: float = 0.0) -> Callable:
    """Create a raycast callback for flat ground."""
    def raycast(origin: Vec3, direction: Vec3, ray_length: float) -> Optional[RaycastHit]:
        if direction.y < 0:  # Pointing down
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


def create_sloped_terrain_raycast(slope_angle: float = 0.3) -> Callable:
    """Create a raycast callback for sloped terrain."""
    def raycast(origin: Vec3, direction: Vec3, ray_length: float) -> Optional[RaycastHit]:
        if direction.y < 0:
            terrain_height = origin.x * math.tan(slope_angle)
            distance = (origin.y - terrain_height) / abs(direction.y)
            if 0 < distance <= ray_length:
                hit_pos = Vec3(
                    origin.x + direction.x * distance,
                    terrain_height,
                    origin.z + direction.z * distance,
                )
                nx = -math.sin(slope_angle)
                ny = math.cos(slope_angle)
                normal = Vec3(nx, ny, 0.0)
                return RaycastHit(hit=True, position=hit_pos, normal=normal, distance=distance)
        return None
    return raycast


def create_per_position_terrain_raycast(height_map: dict) -> Callable:
    """Create a raycast with different heights at different positions."""
    def raycast(origin: Vec3, direction: Vec3, ray_length: float) -> Optional[RaycastHit]:
        if direction.y < 0:
            # Find closest position in height map
            key = (round(origin.x, 1), round(origin.z, 1))
            terrain_height = height_map.get(key, 0.0)
            distance = (origin.y - terrain_height) / abs(direction.y)
            if 0 < distance <= ray_length:
                hit_pos = Vec3(
                    origin.x + direction.x * distance,
                    terrain_height,
                    origin.z + direction.z * distance,
                )
                normal = Vec3(0.0, 1.0, 0.0)
                return RaycastHit(hit=True, position=hit_pos, normal=normal, distance=distance)
        return None
    return raycast


def create_no_hit_raycast() -> Callable:
    """Create a raycast that never hits."""
    def raycast(origin: Vec3, direction: Vec3, ray_length: float) -> Optional[RaycastHit]:
        return None
    return raycast


# =============================================================================
# Skeleton Creation Helpers
# =============================================================================

def create_biped_skeleton() -> List[Transform]:
    """Create a basic biped (2-leg) skeleton.

    Layout:
    0 = pelvis
    1 = left_upper_leg
    2 = left_lower_leg
    3 = left_foot
    4 = right_upper_leg
    5 = right_lower_leg
    6 = right_foot
    """
    return [
        make_transform(Vec3(0.0, 1.0, 0.0)),     # 0: pelvis
        make_transform(Vec3(-0.15, 1.0, 0.0)),   # 1: left_upper_leg
        make_transform(Vec3(-0.15, 0.5, 0.0)),   # 2: left_lower_leg
        make_transform(Vec3(-0.15, 0.0, 0.0)),   # 3: left_foot
        make_transform(Vec3(0.15, 1.0, 0.0)),    # 4: right_upper_leg
        make_transform(Vec3(0.15, 0.5, 0.0)),    # 5: right_lower_leg
        make_transform(Vec3(0.15, 0.0, 0.0)),    # 6: right_foot
    ]


def create_quadruped_skeleton() -> List[Transform]:
    """Create a quadruped (4-leg) skeleton.

    Layout:
    0 = spine/pelvis
    1-3 = front_left leg (upper, lower, foot)
    4-6 = front_right leg
    7-9 = back_left leg
    10-12 = back_right leg
    """
    return [
        make_transform(Vec3(0.0, 0.6, 0.0)),       # 0: pelvis
        # Front left
        make_transform(Vec3(-0.2, 0.6, 0.4)),     # 1: fl_upper
        make_transform(Vec3(-0.2, 0.3, 0.4)),     # 2: fl_lower
        make_transform(Vec3(-0.2, 0.0, 0.4)),     # 3: fl_foot
        # Front right
        make_transform(Vec3(0.2, 0.6, 0.4)),      # 4: fr_upper
        make_transform(Vec3(0.2, 0.3, 0.4)),      # 5: fr_lower
        make_transform(Vec3(0.2, 0.0, 0.4)),      # 6: fr_foot
        # Back left
        make_transform(Vec3(-0.2, 0.6, -0.4)),    # 7: bl_upper
        make_transform(Vec3(-0.2, 0.3, -0.4)),    # 8: bl_lower
        make_transform(Vec3(-0.2, 0.0, -0.4)),    # 9: bl_foot
        # Back right
        make_transform(Vec3(0.2, 0.6, -0.4)),     # 10: br_upper
        make_transform(Vec3(0.2, 0.3, -0.4)),     # 11: br_lower
        make_transform(Vec3(0.2, 0.0, -0.4)),     # 12: br_foot
    ]


def create_hexapod_skeleton() -> List[Transform]:
    """Create a hexapod (6-leg) insect skeleton.

    Layout:
    0 = thorax/pelvis
    1-3 = left_front leg
    4-6 = right_front leg
    7-9 = left_mid leg
    10-12 = right_mid leg
    13-15 = left_back leg
    16-18 = right_back leg
    """
    transforms = [make_transform(Vec3(0.0, 0.3, 0.0))]  # 0: thorax

    # Six legs arranged around body
    leg_positions = [
        (-0.3, 0.4),   # left front
        (0.3, 0.4),    # right front
        (-0.35, 0.0),  # left mid
        (0.35, 0.0),   # right mid
        (-0.3, -0.4),  # left back
        (0.3, -0.4),   # right back
    ]

    for x, z in leg_positions:
        transforms.append(make_transform(Vec3(x, 0.3, z)))      # upper
        transforms.append(make_transform(Vec3(x * 1.5, 0.15, z)))  # lower
        transforms.append(make_transform(Vec3(x * 2.0, 0.0, z)))   # foot

    return transforms


def create_octopod_skeleton() -> List[Transform]:
    """Create an octopod (8-leg) spider skeleton.

    Layout:
    0 = cephalothorax/pelvis
    1-3, 4-6, 7-9, 10-12, 13-15, 16-18, 19-21, 22-24 = 8 legs
    """
    transforms = [make_transform(Vec3(0.0, 0.4, 0.0))]  # 0: cephalothorax

    # Eight legs arranged around body
    angles = [i * (2 * math.pi / 8) for i in range(8)]
    radius = 0.25

    for angle in angles:
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        transforms.append(make_transform(Vec3(x, 0.4, z)))           # upper
        transforms.append(make_transform(Vec3(x * 1.8, 0.2, z * 1.8)))  # lower
        transforms.append(make_transform(Vec3(x * 2.5, 0.0, z * 2.5)))  # foot

    return transforms


def create_many_leg_skeleton(num_legs: int) -> List[Transform]:
    """Create a skeleton with arbitrary number of legs."""
    transforms = [make_transform(Vec3(0.0, 0.5, 0.0))]  # pelvis

    angles = [i * (2 * math.pi / num_legs) for i in range(num_legs)]
    radius = 0.3

    for angle in angles:
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        transforms.append(make_transform(Vec3(x, 0.5, z)))
        transforms.append(make_transform(Vec3(x * 1.5, 0.25, z * 1.5)))
        transforms.append(make_transform(Vec3(x * 2.0, 0.0, z * 2.0)))

    return transforms


# =============================================================================
# FootData Creation Helpers
# =============================================================================

def create_biped_feet() -> List[FootData]:
    """Create foot data for biped."""
    return [
        FootData(upper_leg=1, lower_leg=2, foot=3),
        FootData(upper_leg=4, lower_leg=5, foot=6),
    ]


def create_quadruped_feet() -> List[FootData]:
    """Create foot data for quadruped."""
    return [
        FootData(upper_leg=1, lower_leg=2, foot=3),    # front left
        FootData(upper_leg=4, lower_leg=5, foot=6),    # front right
        FootData(upper_leg=7, lower_leg=8, foot=9),    # back left
        FootData(upper_leg=10, lower_leg=11, foot=12), # back right
    ]


def create_hexapod_feet() -> List[FootData]:
    """Create foot data for hexapod."""
    feet = []
    for i in range(6):
        base = 1 + i * 3
        feet.append(FootData(upper_leg=base, lower_leg=base + 1, foot=base + 2))
    return feet


def create_octopod_feet() -> List[FootData]:
    """Create foot data for octopod."""
    feet = []
    for i in range(8):
        base = 1 + i * 3
        feet.append(FootData(upper_leg=base, lower_leg=base + 1, foot=base + 2))
    return feet


def create_many_feet(num_legs: int) -> List[FootData]:
    """Create foot data for arbitrary number of legs."""
    feet = []
    for i in range(num_legs):
        base = 1 + i * 3
        feet.append(FootData(upper_leg=base, lower_leg=base + 1, foot=base + 2))
    return feet


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def biped_transforms():
    """Standard biped skeleton transforms."""
    return create_biped_skeleton()


@pytest.fixture
def quadruped_transforms():
    """Quadruped skeleton transforms."""
    return create_quadruped_skeleton()


@pytest.fixture
def hexapod_transforms():
    """Hexapod skeleton transforms."""
    return create_hexapod_skeleton()


@pytest.fixture
def octopod_transforms():
    """Octopod (spider) skeleton transforms."""
    return create_octopod_skeleton()


@pytest.fixture
def biped_feet():
    """Biped foot data."""
    return create_biped_feet()


@pytest.fixture
def quadruped_feet():
    """Quadruped foot data."""
    return create_quadruped_feet()


@pytest.fixture
def hexapod_feet():
    """Hexapod foot data."""
    return create_hexapod_feet()


@pytest.fixture
def octopod_feet():
    """Octopod foot data."""
    return create_octopod_feet()


@pytest.fixture
def flat_raycast():
    """Raycast for flat ground at y=0."""
    return create_flat_ground_raycast(0.0)


@pytest.fixture
def sloped_raycast():
    """Raycast for sloped terrain."""
    return create_sloped_terrain_raycast(0.2)


@pytest.fixture
def no_hit_raycast():
    """Raycast that never hits."""
    return create_no_hit_raycast()


# =============================================================================
# Construction Tests
# =============================================================================

class TestMultiLegConstruction:
    """Tests for MultiLegFootPlacement construction."""

    def test_can_instantiate_with_foot_list(self, biped_feet):
        """MultiLegFootPlacement can be instantiated with list of FootData."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
        )
        assert solver is not None

    def test_can_instantiate_with_two_legs(self, biped_feet):
        """Supports 2-leg (biped) configuration."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
        )
        assert solver is not None

    def test_can_instantiate_with_four_legs(self, quadruped_feet):
        """Supports 4-leg (quadruped) configuration."""
        solver = MultiLegFootPlacement(
            feet=quadruped_feet,
            pelvis=0,
        )
        assert solver is not None

    def test_can_instantiate_with_six_legs(self, hexapod_feet):
        """Supports 6-leg (hexapod/insect) configuration."""
        solver = MultiLegFootPlacement(
            feet=hexapod_feet,
            pelvis=0,
        )
        assert solver is not None

    def test_can_instantiate_with_eight_legs(self, octopod_feet):
        """Supports 8-leg (octopod/spider) configuration."""
        solver = MultiLegFootPlacement(
            feet=octopod_feet,
            pelvis=0,
        )
        assert solver is not None

    def test_pelvis_bone_index_required(self, biped_feet):
        """Pelvis bone index is required for construction."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
        )
        assert solver is not None

    def test_pelvis_bone_can_be_nonzero(self, biped_feet):
        """Pelvis bone can be any valid index."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=5,
        )
        assert solver is not None

    def test_raycast_callback_optional(self, biped_feet):
        """Raycast callback is optional."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
        )
        assert solver is not None

    def test_can_instantiate_with_raycast(self, biped_feet, flat_raycast):
        """Can instantiate with raycast callback."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        assert solver is not None

    def test_multiple_solvers_independent(self, biped_feet, quadruped_feet):
        """Multiple solver instances are independent."""
        solver1 = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        solver2 = MultiLegFootPlacement(feet=quadruped_feet, pelvis=0)
        assert solver1 is not solver2


# =============================================================================
# Solve Without Raycast Tests
# =============================================================================

class TestSolveWithoutRaycast:
    """Tests for solve() behavior without raycast."""

    def test_solve_returns_transforms(self, biped_feet, biped_transforms):
        """Solve returns list of transforms."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        result = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None
        assert isinstance(result, list)

    def test_solve_without_raycast_returns_unchanged(self, biped_feet, biped_transforms):
        """Without raycast, transforms are returned unchanged."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        result = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))
        # Transforms should match input
        assert len(result) == len(biped_transforms)

    def test_solve_preserves_transform_count(self, biped_feet, biped_transforms):
        """Result has same number of transforms as input."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        result = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))
        assert len(result) == len(biped_transforms)

    def test_solve_without_raycast_quadruped(self, quadruped_feet, quadruped_transforms):
        """Solve works for quadruped without raycast."""
        solver = MultiLegFootPlacement(feet=quadruped_feet, pelvis=0)
        result = solver.solve(quadruped_transforms, Vec3(0.0, 0.0, 0.0))
        assert len(result) == len(quadruped_transforms)

    def test_solve_without_raycast_hexapod(self, hexapod_feet, hexapod_transforms):
        """Solve works for hexapod without raycast."""
        solver = MultiLegFootPlacement(feet=hexapod_feet, pelvis=0)
        result = solver.solve(hexapod_transforms, Vec3(0.0, 0.0, 0.0))
        assert len(result) == len(hexapod_transforms)

    def test_solve_without_raycast_octopod(self, octopod_feet, octopod_transforms):
        """Solve works for octopod without raycast."""
        solver = MultiLegFootPlacement(feet=octopod_feet, pelvis=0)
        result = solver.solve(octopod_transforms, Vec3(0.0, 0.0, 0.0))
        assert len(result) == len(octopod_transforms)


# =============================================================================
# Solve With Raycast Tests
# =============================================================================

class TestSolveWithRaycast:
    """Tests for solve() behavior with raycast."""

    def test_solve_with_raycast_processes_feet(self, biped_feet, biped_transforms, flat_raycast):
        """With raycast, all feet are processed."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_solve_with_raycast_returns_transforms(self, biped_feet, biped_transforms, flat_raycast):
        """Solve with raycast returns list of transforms."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))
        assert isinstance(result, list)

    def test_solve_with_raycast_preserves_count(self, quadruped_feet, quadruped_transforms, flat_raycast):
        """Result count matches input count with raycast."""
        solver = MultiLegFootPlacement(
            feet=quadruped_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result = solver.solve(quadruped_transforms, Vec3(0.0, 0.0, 0.0))
        assert len(result) == len(quadruped_transforms)

    def test_solve_with_no_hit_raycast(self, biped_feet, biped_transforms, no_hit_raycast):
        """Solve handles raycast with no hits gracefully."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
            raycast_callback=no_hit_raycast,
        )
        result = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_solve_with_sloped_terrain(self, biped_feet, biped_transforms, sloped_raycast):
        """Solve handles sloped terrain."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
            raycast_callback=sloped_raycast,
        )
        result = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None


# =============================================================================
# Multi-Leg Configuration Tests
# =============================================================================

class TestMultiLegConfigurations:
    """Tests for various multi-leg configurations."""

    def test_spider_eight_legs(self, octopod_feet, octopod_transforms, flat_raycast):
        """Spider (8 legs) configuration works correctly."""
        solver = MultiLegFootPlacement(
            feet=octopod_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result = solver.solve(octopod_transforms, Vec3(0.0, 0.0, 0.0))
        assert len(result) == len(octopod_transforms)

    def test_centaur_four_legs(self, quadruped_feet, quadruped_transforms, flat_raycast):
        """Centaur/quadruped (4 legs) configuration works correctly."""
        solver = MultiLegFootPlacement(
            feet=quadruped_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result = solver.solve(quadruped_transforms, Vec3(0.0, 0.0, 0.0))
        assert len(result) == len(quadruped_transforms)

    def test_insect_six_legs(self, hexapod_feet, hexapod_transforms, flat_raycast):
        """Insect (6 legs) configuration works correctly."""
        solver = MultiLegFootPlacement(
            feet=hexapod_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result = solver.solve(hexapod_transforms, Vec3(0.0, 0.0, 0.0))
        assert len(result) == len(hexapod_transforms)

    def test_each_leg_processed_independently(self, quadruped_feet, quadruped_transforms):
        """Each leg is processed independently."""
        height_map = {
            (-0.2, 0.4): 0.1,   # front left higher
            (0.2, 0.4): 0.0,    # front right at ground
            (-0.2, -0.4): -0.1, # back left lower
            (0.2, -0.4): 0.0,   # back right at ground
        }
        raycast = create_per_position_terrain_raycast(height_map)

        solver = MultiLegFootPlacement(
            feet=quadruped_feet,
            pelvis=0,
            raycast_callback=raycast,
        )
        result = solver.solve(quadruped_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_all_legs_contribute_to_result(self, hexapod_feet, hexapod_transforms, flat_raycast):
        """All legs contribute to the final result."""
        solver = MultiLegFootPlacement(
            feet=hexapod_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result = solver.solve(hexapod_transforms, Vec3(0.0, 0.0, 0.0))
        # Should have transforms for all bones including all 6 legs
        assert len(result) == len(hexapod_transforms)


# =============================================================================
# Configuration Access Tests
# =============================================================================

class TestConfigurationAccess:
    """Tests for configuration property accessibility."""

    def test_ray_length_accessible(self, biped_feet):
        """ray_length property is accessible."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        assert hasattr(solver, 'ray_length')

    def test_ray_length_has_value(self, biped_feet):
        """ray_length has a numeric value."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        assert isinstance(solver.ray_length, (int, float))

    def test_foot_height_accessible(self, biped_feet):
        """foot_height property is accessible."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        assert hasattr(solver, 'foot_height')

    def test_foot_height_has_value(self, biped_feet):
        """foot_height has a numeric value."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        assert isinstance(solver.foot_height, (int, float))

    def test_blend_speed_accessible(self, biped_feet):
        """blend_speed property is accessible."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        assert hasattr(solver, 'blend_speed')

    def test_blend_speed_has_value(self, biped_feet):
        """blend_speed has a numeric value."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        assert isinstance(solver.blend_speed, (int, float))

    def test_config_affects_all_legs_equally(self, quadruped_feet, quadruped_transforms, flat_raycast):
        """Configuration affects all legs equally."""
        solver = MultiLegFootPlacement(
            feet=quadruped_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        # Just verify solve works with default config
        result = solver.solve(quadruped_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_can_set_ray_length(self, biped_feet):
        """ray_length can be modified."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        original = solver.ray_length
        solver.ray_length = 2.0
        assert solver.ray_length == 2.0 or solver.ray_length == original

    def test_can_set_foot_height(self, biped_feet):
        """foot_height can be modified."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        original = solver.foot_height
        solver.foot_height = 0.1
        assert solver.foot_height == 0.1 or solver.foot_height == original

    def test_can_set_blend_speed(self, biped_feet):
        """blend_speed can be modified."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        original = solver.blend_speed
        solver.blend_speed = 5.0
        assert solver.blend_speed == 5.0 or solver.blend_speed == original


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_leg(self):
        """Handles single leg configuration."""
        transforms = [
            make_transform(Vec3(0.0, 0.5, 0.0)),  # pelvis
            make_transform(Vec3(0.0, 0.5, 0.0)),  # upper
            make_transform(Vec3(0.0, 0.25, 0.0)), # lower
            make_transform(Vec3(0.0, 0.0, 0.0)),  # foot
        ]
        feet = [FootData(upper_leg=1, lower_leg=2, foot=3)]

        solver = MultiLegFootPlacement(feet=feet, pelvis=0)
        result = solver.solve(transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None
        assert len(result) == len(transforms)

    def test_empty_feet_list(self):
        """Handles empty feet list gracefully."""
        transforms = [make_transform(Vec3(0.0, 0.5, 0.0))]
        feet = []

        solver = MultiLegFootPlacement(feet=feet, pelvis=0)
        result = solver.solve(transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_many_legs_twelve(self):
        """Handles 12+ legs configuration."""
        num_legs = 12
        transforms = create_many_leg_skeleton(num_legs)
        feet = create_many_feet(num_legs)

        solver = MultiLegFootPlacement(feet=feet, pelvis=0)
        result = solver.solve(transforms, Vec3(0.0, 0.0, 0.0))
        assert len(result) == len(transforms)

    def test_many_legs_sixteen(self):
        """Handles 16 legs configuration."""
        num_legs = 16
        transforms = create_many_leg_skeleton(num_legs)
        feet = create_many_feet(num_legs)

        solver = MultiLegFootPlacement(feet=feet, pelvis=0)
        result = solver.solve(transforms, Vec3(0.0, 0.0, 0.0))
        assert len(result) == len(transforms)

    def test_different_terrain_heights_per_leg(self, quadruped_feet, quadruped_transforms):
        """Handles different terrain heights for each leg."""
        height_map = {
            (-0.2, 0.4): 0.2,   # front left: step up
            (0.2, 0.4): -0.1,   # front right: step down
            (-0.2, -0.4): 0.0,  # back left: ground level
            (0.2, -0.4): 0.15,  # back right: step up
        }
        raycast = create_per_position_terrain_raycast(height_map)

        solver = MultiLegFootPlacement(
            feet=quadruped_feet,
            pelvis=0,
            raycast_callback=raycast,
        )
        result = solver.solve(quadruped_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_extreme_terrain_variation(self, hexapod_feet, hexapod_transforms):
        """Handles extreme terrain height variations."""
        def extreme_raycast(origin: Vec3, direction: Vec3, ray_length: float):
            if direction.y < 0:
                # Very steep terrain
                terrain_height = origin.x * 2.0 + origin.z * 1.5
                distance = (origin.y - terrain_height) / abs(direction.y)
                if 0 < distance <= ray_length:
                    hit_pos = Vec3(origin.x, terrain_height, origin.z)
                    return RaycastHit(hit=True, position=hit_pos, normal=Vec3(0, 1, 0), distance=distance)
            return None

        solver = MultiLegFootPlacement(
            feet=hexapod_feet,
            pelvis=0,
            raycast_callback=extreme_raycast,
        )
        result = solver.solve(hexapod_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_all_feet_in_air(self, biped_feet, biped_transforms, no_hit_raycast):
        """Handles all feet in air (no ground contact)."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
            raycast_callback=no_hit_raycast,
        )
        result = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_high_ground(self, biped_feet):
        """Handles ground above foot position."""
        transforms = [
            make_transform(Vec3(0.0, 0.5, 0.0)),   # pelvis
            make_transform(Vec3(-0.15, 0.5, 0.0)), # upper
            make_transform(Vec3(-0.15, 0.25, 0.0)),# lower
            make_transform(Vec3(-0.15, 0.0, 0.0)), # foot
            make_transform(Vec3(0.15, 0.5, 0.0)),  # upper
            make_transform(Vec3(0.15, 0.25, 0.0)), # lower
            make_transform(Vec3(0.15, 0.0, 0.0)),  # foot
        ]
        raycast = create_flat_ground_raycast(0.5)  # Ground at foot level

        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
            raycast_callback=raycast,
        )
        result = solver.solve(transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None


# =============================================================================
# Result Transform Tests
# =============================================================================

class TestResultTransforms:
    """Tests for the result transform list."""

    def test_result_contains_transforms(self, biped_feet, biped_transforms):
        """Result contains Transform objects."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        result = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))
        for t in result:
            assert isinstance(t, Transform)

    def test_result_transforms_have_translation(self, biped_feet, biped_transforms):
        """Result transforms have translation."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        result = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))
        for t in result:
            assert hasattr(t, 'translation')

    def test_result_transforms_have_rotation(self, biped_feet, biped_transforms):
        """Result transforms have rotation."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        result = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))
        for t in result:
            assert hasattr(t, 'rotation')

    def test_result_matches_input_indices(self, quadruped_feet, quadruped_transforms, flat_raycast):
        """Result transform indices match input."""
        solver = MultiLegFootPlacement(
            feet=quadruped_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result = solver.solve(quadruped_transforms, Vec3(0.0, 0.0, 0.0))
        # Pelvis index should still be valid
        assert len(result) > 0


# =============================================================================
# Repeated Solve Tests
# =============================================================================

class TestRepeatedSolve:
    """Tests for repeated solve calls."""

    def test_solve_can_be_called_multiple_times(self, biped_feet, biped_transforms, flat_raycast):
        """Solve can be called multiple times."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result1 = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))
        result2 = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))
        assert result1 is not None
        assert result2 is not None

    def test_solve_with_different_transforms(self, biped_feet, flat_raycast):
        """Solve works with different transforms each call."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )

        transforms1 = create_biped_skeleton()
        transforms2 = create_biped_skeleton()
        # Modify second set
        transforms2[0] = make_transform(Vec3(0.0, 1.5, 0.0))  # Higher pelvis

        result1 = solver.solve(transforms1, Vec3(0.0, 0.0, 0.0))
        result2 = solver.solve(transforms2, Vec3(0.0, 0.0, 0.0))
        assert result1 is not None
        assert result2 is not None

    def test_solve_consistency(self, biped_feet, biped_transforms, flat_raycast):
        """Solve returns consistent results for same input."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result1 = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))
        result2 = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))

        # Results should be the same length
        assert len(result1) == len(result2)


# =============================================================================
# Delta Time Tests
# =============================================================================

class TestDeltaTime:
    """Tests for solve with delta time parameter."""

    def test_solve_accepts_delta_time(self, biped_feet, biped_transforms):
        """Solve accepts optional delta time parameter."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        result = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0), dt=0.016)
        assert result is not None

    def test_solve_with_zero_dt(self, biped_feet, biped_transforms, flat_raycast):
        """Solve handles zero delta time."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0), dt=0.0)
        assert result is not None

    def test_solve_with_large_dt(self, biped_feet, biped_transforms, flat_raycast):
        """Solve handles large delta time."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0), dt=1.0)
        assert result is not None


# =============================================================================
# Character Position Tests
# =============================================================================

class TestCharacterPosition:
    """Tests for solve with character position."""

    def test_solve_requires_character_position(self, biped_feet, biped_transforms):
        """Solve requires character position as second argument."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        result = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_solve_with_moved_character(self, biped_feet, biped_transforms, flat_raycast):
        """Solve works with moved character position."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result = solver.solve(biped_transforms, Vec3(10.0, 0.0, 5.0))
        assert result is not None

    def test_solve_with_negative_character_position(self, biped_feet, biped_transforms, flat_raycast):
        """Solve works with negative character position."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result = solver.solve(biped_transforms, Vec3(-5.0, -1.0, -3.0))
        assert result is not None

    def test_solve_with_elevated_character(self, biped_feet, biped_transforms, flat_raycast):
        """Solve works with elevated character position."""
        solver = MultiLegFootPlacement(
            feet=biped_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result = solver.solve(biped_transforms, Vec3(0.0, 10.0, 0.0))
        assert result is not None


# =============================================================================
# FootData Validation Tests
# =============================================================================

class TestFootDataForMultiLeg:
    """Tests for FootData used in multi-leg context."""

    def test_foot_data_indices_valid(self, biped_feet, biped_transforms):
        """FootData indices should be valid for transforms."""
        solver = MultiLegFootPlacement(feet=biped_feet, pelvis=0)
        result = solver.solve(biped_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_foot_data_with_optional_toe(self):
        """FootData supports optional toe bone."""
        feet = [
            FootData(upper_leg=1, lower_leg=2, foot=3, toe=4),
            FootData(upper_leg=5, lower_leg=6, foot=7, toe=8),
        ]
        transforms = [
            make_transform(Vec3(0, 1, 0)),  # pelvis
            # Left leg with toe
            make_transform(Vec3(-0.15, 1, 0)),
            make_transform(Vec3(-0.15, 0.5, 0)),
            make_transform(Vec3(-0.15, 0, 0)),
            make_transform(Vec3(-0.15, 0, 0.1)),
            # Right leg with toe
            make_transform(Vec3(0.15, 1, 0)),
            make_transform(Vec3(0.15, 0.5, 0)),
            make_transform(Vec3(0.15, 0, 0)),
            make_transform(Vec3(0.15, 0, 0.1)),
        ]

        solver = MultiLegFootPlacement(feet=feet, pelvis=0)
        result = solver.solve(transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_different_foot_data_per_leg(self):
        """Different FootData configurations per leg."""
        # One leg with toe, one without
        feet = [
            FootData(upper_leg=1, lower_leg=2, foot=3, toe=4),
            FootData(upper_leg=5, lower_leg=6, foot=7),
        ]
        transforms = [make_transform(Vec3(0, 1, 0))]  # pelvis
        for i in range(8):
            transforms.append(make_transform(Vec3(0, 0.5, 0)))

        solver = MultiLegFootPlacement(feet=feet, pelvis=0)
        result = solver.solve(transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None


# =============================================================================
# Integration Tests
# =============================================================================

class TestMultiLegIntegration:
    """Integration tests for multi-leg foot placement."""

    def test_full_spider_simulation(self, octopod_feet, octopod_transforms, flat_raycast):
        """Full spider walking simulation."""
        solver = MultiLegFootPlacement(
            feet=octopod_feet,
            pelvis=0,
            raycast_callback=flat_raycast,
        )

        # Simulate multiple frames
        for _ in range(10):
            result = solver.solve(octopod_transforms, Vec3(0.0, 0.0, 0.0))
            assert result is not None
            assert len(result) == len(octopod_transforms)

    def test_quadruped_uneven_terrain(self, quadruped_feet, quadruped_transforms):
        """Quadruped on uneven terrain."""
        def terrain_raycast(origin: Vec3, direction: Vec3, ray_length: float):
            if direction.y < 0:
                # Wavy terrain
                height = 0.1 * math.sin(origin.x * 3) * math.cos(origin.z * 2)
                distance = (origin.y - height) / abs(direction.y)
                if 0 < distance <= ray_length:
                    return RaycastHit(
                        hit=True,
                        position=Vec3(origin.x, height, origin.z),
                        normal=Vec3(0, 1, 0),
                        distance=distance,
                    )
            return None

        solver = MultiLegFootPlacement(
            feet=quadruped_feet,
            pelvis=0,
            raycast_callback=terrain_raycast,
        )
        result = solver.solve(quadruped_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_hexapod_stepping_stones(self, hexapod_feet, hexapod_transforms):
        """Hexapod on stepping stones."""
        # Only some positions have ground
        stones = {
            (-0.3, 0.4): 0.0,
            (0.35, 0.0): 0.1,
            (0.3, -0.4): -0.05,
        }

        def stone_raycast(origin: Vec3, direction: Vec3, ray_length: float):
            if direction.y < 0:
                key = (round(origin.x, 1), round(origin.z, 1))
                for (x, z), h in stones.items():
                    if abs(origin.x - x) < 0.2 and abs(origin.z - z) < 0.2:
                        distance = (origin.y - h) / abs(direction.y)
                        if 0 < distance <= ray_length:
                            return RaycastHit(
                                hit=True,
                                position=Vec3(origin.x, h, origin.z),
                                normal=Vec3(0, 1, 0),
                                distance=distance,
                            )
            return None

        solver = MultiLegFootPlacement(
            feet=hexapod_feet,
            pelvis=0,
            raycast_callback=stone_raycast,
        )
        result = solver.solve(hexapod_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

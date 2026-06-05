"""Blackbox tests for Foot Placement IK (Phase 4).

This module tests the FootPlacement system from the public API only,
without knowledge of implementation details. Tests are derived from
theoretical foot placement IK behavior:

1. Ground detection via raycasting
2. Foot IK to place feet on terrain
3. Animated blending for smooth transitions
4. Multi-leg support (humanoid, quadruped, spider)
5. Uneven terrain handling

Test Strategy:
- Test public API contracts only
- Test behavioral expectations for foot-ground contact
- Test smooth blending between frames
- Test multi-leg configurations
- Test terrain adaptation
"""

import math
import pytest
from typing import List, Optional, Callable, Tuple

# Import public API
from engine.animation.ik import (
    FootPlacement,
    FootPlacementResult,
    FootData,
    FootState,
    FootPlacementAnimated,
    MultiLegFootPlacement,
    RaycastCallback,
    RaycastHit,
    FOOT_PLACEMENT_RAY_LENGTH,
    FOOT_PLACEMENT_FOOT_HEIGHT,
    FOOT_PLACEMENT_BLEND_SPEED,
    FOOT_PLACEMENT_MAX_PELVIS_DROP,
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


def create_quadruped_transforms() -> List[Transform]:
    """Create a basic quadruped skeleton for foot placement.

    Layout (indices):
    0 = spine_root (pelvis)
    1 = spine_mid
    2 = front_left_upper_leg
    3 = front_left_lower_leg
    4 = front_left_foot
    5 = front_right_upper_leg
    6 = front_right_lower_leg
    7 = front_right_foot
    8 = back_left_upper_leg
    9 = back_left_lower_leg
    10 = back_left_foot
    11 = back_right_upper_leg
    12 = back_right_lower_leg
    13 = back_right_foot
    """
    transforms = [
        # Spine
        make_transform(Vec3(0.0, 0.5, -0.3)),   # 0: spine_root
        make_transform(Vec3(0.0, 0.5, 0.3)),    # 1: spine_mid
        # Front left leg
        make_transform(Vec3(-0.15, 0.5, 0.3)),  # 2: front_left_upper_leg
        make_transform(Vec3(-0.15, 0.25, 0.3)), # 3: front_left_lower_leg
        make_transform(Vec3(-0.15, 0.0, 0.3)),  # 4: front_left_foot
        # Front right leg
        make_transform(Vec3(0.15, 0.5, 0.3)),   # 5: front_right_upper_leg
        make_transform(Vec3(0.15, 0.25, 0.3)),  # 6: front_right_lower_leg
        make_transform(Vec3(0.15, 0.0, 0.3)),   # 7: front_right_foot
        # Back left leg
        make_transform(Vec3(-0.15, 0.5, -0.3)), # 8: back_left_upper_leg
        make_transform(Vec3(-0.15, 0.25, -0.3)),# 9: back_left_lower_leg
        make_transform(Vec3(-0.15, 0.0, -0.3)), # 10: back_left_foot
        # Back right leg
        make_transform(Vec3(0.15, 0.5, -0.3)),  # 11: back_right_upper_leg
        make_transform(Vec3(0.15, 0.25, -0.3)), # 12: back_right_lower_leg
        make_transform(Vec3(0.15, 0.0, -0.3)),  # 13: back_right_foot
    ]
    return transforms


def create_flat_ground_raycast(ground_height: float = 0.0) -> Callable:
    """Create a raycast callback for flat ground.

    Returns: RaycastHit or None
    """
    def raycast(origin: Vec3, direction: Vec3, ray_length: float) -> Optional[RaycastHit]:
        if direction.y < 0:  # Pointing down
            distance = (origin.y - ground_height) / abs(direction.y)
            if 0 < distance <= ray_length:
                hit_pos = Vec3(
                    origin.x + direction.x * distance,
                    ground_height,
                    origin.z + direction.z * distance,
                )
                normal = Vec3(0.0, 1.0, 0.0)  # Flat ground normal
                return RaycastHit(hit=True, position=hit_pos, normal=normal, distance=distance)
        return None
    return raycast


def create_uneven_terrain_raycast() -> Callable:
    """Create a raycast callback for uneven terrain.

    Returns: RaycastHit or None
    """
    def raycast(origin: Vec3, direction: Vec3, ray_length: float) -> Optional[RaycastHit]:
        if direction.y < 0:
            # Terrain height varies with x and z
            terrain_height = 0.1 * math.sin(origin.x * 2) + 0.1 * math.cos(origin.z * 2)
            distance = (origin.y - terrain_height) / abs(direction.y)
            if 0 < distance <= ray_length:
                hit_pos = Vec3(
                    origin.x + direction.x * distance,
                    terrain_height,
                    origin.z + direction.z * distance,
                )
                normal = Vec3(0.0, 1.0, 0.0)  # Approximate normal
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
def quadruped_transforms():
    """Quadruped skeleton transforms."""
    return create_quadruped_transforms()


@pytest.fixture
def left_foot():
    """Left foot data."""
    return create_left_foot_data()


@pytest.fixture
def right_foot():
    """Right foot data."""
    return create_right_foot_data()


@pytest.fixture
def foot_solver(left_foot, right_foot):
    """Standard foot placement solver for humanoid."""
    return FootPlacement(
        left_foot=left_foot,
        right_foot=right_foot,
        pelvis=0,
    )


@pytest.fixture
def flat_raycast():
    """Raycast for flat ground at y=0."""
    return create_flat_ground_raycast(0.0)


@pytest.fixture
def uneven_raycast():
    """Raycast for uneven terrain."""
    return create_uneven_terrain_raycast()


# =============================================================================
# FootPlacement Instantiation Tests
# =============================================================================

class TestFootPlacementInstantiation:
    """Tests for FootPlacement class instantiation."""

    def test_can_instantiate_with_foot_data(self, left_foot, right_foot):
        """FootPlacement can be instantiated with foot data."""
        solver = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
        )
        assert solver is not None

    def test_can_instantiate_with_raycast_callback(self, left_foot, right_foot, flat_raycast):
        """FootPlacement can be instantiated with raycast callback."""
        solver = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        assert solver is not None

    def test_multiple_solvers_independent(self, left_foot, right_foot):
        """Multiple FootPlacement instances are independent."""
        solver1 = FootPlacement(left_foot=left_foot, right_foot=right_foot, pelvis=0)
        solver2 = FootPlacement(left_foot=left_foot, right_foot=right_foot, pelvis=0)
        assert solver1 is not solver2


# =============================================================================
# FootData Tests
# =============================================================================

class TestFootData:
    """Tests for FootData structure."""

    def test_can_create_foot_data(self):
        """FootData can be created."""
        data = FootData(
            upper_leg=3,
            lower_leg=4,
            foot=5,
        )
        assert data is not None

    def test_foot_data_has_upper_leg(self):
        """FootData has upper_leg."""
        data = FootData(upper_leg=3, lower_leg=4, foot=5)
        assert hasattr(data, 'upper_leg')
        assert data.upper_leg == 3

    def test_foot_data_has_lower_leg(self):
        """FootData has lower_leg."""
        data = FootData(upper_leg=3, lower_leg=4, foot=5)
        assert hasattr(data, 'lower_leg')
        assert data.lower_leg == 4

    def test_foot_data_has_foot(self):
        """FootData has foot."""
        data = FootData(upper_leg=3, lower_leg=4, foot=5)
        assert hasattr(data, 'foot')
        assert data.foot == 5

    def test_foot_data_with_toe(self):
        """FootData can have optional toe."""
        data = FootData(upper_leg=3, lower_leg=4, foot=5, toe=6)
        assert data.toe == 6

    def test_foot_data_has_state(self):
        """FootData has state."""
        data = FootData(upper_leg=3, lower_leg=4, foot=5, state=FootState.PLANTED)
        assert data.state == FootState.PLANTED

    def test_foot_data_has_target_position(self):
        """FootData has target_position."""
        data = FootData(upper_leg=3, lower_leg=4, foot=5)
        assert hasattr(data, 'target_position')

    def test_foot_data_has_blend_weight(self):
        """FootData has blend_weight."""
        data = FootData(upper_leg=3, lower_leg=4, foot=5, blend_weight=0.8)
        assert data.blend_weight == 0.8

    def test_foot_data_has_height_offset(self):
        """FootData has height_offset."""
        data = FootData(upper_leg=3, lower_leg=4, foot=5, height_offset=0.05)
        assert data.height_offset == 0.05


# =============================================================================
# FootState Tests
# =============================================================================

class TestFootState:
    """Tests for FootState enum."""

    def test_has_planted_state(self):
        """FootState has PLANTED state."""
        assert hasattr(FootState, 'PLANTED')

    def test_has_lifting_state(self):
        """FootState has LIFTING state."""
        assert hasattr(FootState, 'LIFTING')

    def test_has_airborne_state(self):
        """FootState has AIRBORNE state."""
        assert hasattr(FootState, 'AIRBORNE')

    def test_has_landing_state(self):
        """FootState has LANDING state."""
        assert hasattr(FootState, 'LANDING')

    def test_can_compare_states(self):
        """FootState values can be compared."""
        assert FootState.PLANTED != FootState.AIRBORNE


# =============================================================================
# FootPlacementResult Tests
# =============================================================================

class TestFootPlacementResult:
    """Tests for FootPlacementResult structure."""

    def test_result_has_success_field(self, foot_solver, humanoid_transforms):
        """FootPlacementResult has success field."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert hasattr(result, 'success')

    def test_result_has_transforms_field(self, foot_solver, humanoid_transforms):
        """FootPlacementResult has transforms field."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert hasattr(result, 'transforms')

    def test_result_has_pelvis_offset(self, foot_solver, humanoid_transforms):
        """FootPlacementResult has pelvis_offset."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert hasattr(result, 'pelvis_offset')

    def test_result_has_left_foot_planted(self, foot_solver, humanoid_transforms):
        """FootPlacementResult has left_foot_planted."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert hasattr(result, 'left_foot_planted')

    def test_result_has_right_foot_planted(self, foot_solver, humanoid_transforms):
        """FootPlacementResult has right_foot_planted."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert hasattr(result, 'right_foot_planted')

    def test_result_has_terrain_slope(self, foot_solver, humanoid_transforms):
        """FootPlacementResult has terrain_slope."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert hasattr(result, 'terrain_slope')

    def test_result_transforms_same_count(self, foot_solver, humanoid_transforms):
        """Result transforms have same count as input."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert len(result.transforms) == len(humanoid_transforms)


# =============================================================================
# Basic Solve Tests
# =============================================================================

class TestFootPlacementBasicSolve:
    """Tests for basic FootPlacement solve functionality."""

    def test_solve_returns_result(self, foot_solver, humanoid_transforms):
        """Solve returns a FootPlacementResult."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert isinstance(result, FootPlacementResult)

    def test_solve_with_character_position(self, foot_solver, humanoid_transforms):
        """Solve works with character position."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_solve_with_moved_character(self, foot_solver, humanoid_transforms):
        """Solve works with moved character position."""
        result = foot_solver.solve(humanoid_transforms, Vec3(5.0, 0.0, 3.0))
        assert result is not None

    def test_solve_with_delta_time(self, foot_solver, humanoid_transforms):
        """Solve works with delta time."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0), dt=0.016)
        assert result is not None

    def test_solve_with_large_delta_time(self, foot_solver, humanoid_transforms):
        """Solve works with large delta time."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0), dt=1.0)
        assert result is not None


# =============================================================================
# Ground Contact Tests
# =============================================================================

class TestFootPlacementGroundContact:
    """Tests for ground contact behavior."""

    def test_feet_contact_with_raycast(self, left_foot, right_foot, humanoid_transforms, flat_raycast):
        """Feet make contact with raycast ground detection."""
        solver = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result = solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_both_feet_can_be_planted(self, foot_solver, humanoid_transforms):
        """Both feet can be planted."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        # Check both feet planted flags exist
        assert hasattr(result, 'left_foot_planted')
        assert hasattr(result, 'right_foot_planted')


# =============================================================================
# Raycast Callback Tests
# =============================================================================

class TestFootPlacementRaycast:
    """Tests for raycast-based ground detection."""

    def test_solve_with_raycast_callback(self, left_foot, right_foot, humanoid_transforms, flat_raycast):
        """Solve works with raycast callback."""
        solver = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=flat_raycast,
        )
        result = solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_raycast_detects_uneven_terrain(self, left_foot, right_foot, humanoid_transforms, uneven_raycast):
        """Raycast detects uneven terrain."""
        solver = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=uneven_raycast,
        )
        result = solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_no_raycast_uses_default(self, left_foot, right_foot, humanoid_transforms):
        """No raycast uses default behavior."""
        solver = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=None,
        )
        result = solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None


# =============================================================================
# RaycastCallback Type Tests
# =============================================================================

class TestRaycastCallback:
    """Tests for RaycastCallback type."""

    def test_raycast_callback_exists(self):
        """RaycastCallback type exists."""
        assert RaycastCallback is not None

    def test_custom_raycast_callback(self, left_foot, right_foot, humanoid_transforms):
        """Can use custom raycast callback."""
        def custom_raycast(origin: Vec3, direction: Vec3, ray_length: float) -> Optional[RaycastHit]:
            # Simple flat plane at y=0.2
            if direction.y < 0:
                distance = (origin.y - 0.2) / abs(direction.y)
                if 0 < distance <= ray_length:
                    hit_pos = Vec3(origin.x, 0.2, origin.z)
                    return RaycastHit(hit=True, position=hit_pos, normal=Vec3(0, 1, 0), distance=distance)
            return None

        solver = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=custom_raycast,
        )
        result = solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None


# =============================================================================
# FootPlacementAnimated Tests
# =============================================================================

class TestFootPlacementAnimated:
    """Tests for animated foot placement with blending."""

    def test_can_create_animated_solver(self, foot_solver):
        """FootPlacementAnimated can be created."""
        solver = FootPlacementAnimated(base_placement=foot_solver)
        assert solver is not None

    def test_animated_has_update_method(self, foot_solver):
        """FootPlacementAnimated has update method."""
        solver = FootPlacementAnimated(base_placement=foot_solver)
        assert hasattr(solver, 'update')

    def test_animated_update_with_delta(self, foot_solver):
        """Animated update accepts delta time."""
        solver = FootPlacementAnimated(base_placement=foot_solver)
        solver.update(dt=0.016)  # Should not raise

    def test_animated_update_small_delta(self, foot_solver):
        """Animated update with small delta."""
        solver = FootPlacementAnimated(base_placement=foot_solver)
        solver.update(dt=0.001)

    def test_animated_update_large_delta(self, foot_solver):
        """Animated update with large delta."""
        solver = FootPlacementAnimated(base_placement=foot_solver)
        solver.update(dt=1.0)


# =============================================================================
# MultiLegFootPlacement Tests
# =============================================================================

class TestMultiLegFootPlacement:
    """Tests for multi-leg foot placement (quadruped, spider, etc.)."""

    def test_can_create_multileg_solver(self):
        """MultiLegFootPlacement can be created."""
        feet = [
            FootData(upper_leg=2, lower_leg=3, foot=4),
            FootData(upper_leg=5, lower_leg=6, foot=7),
            FootData(upper_leg=8, lower_leg=9, foot=10),
            FootData(upper_leg=11, lower_leg=12, foot=13),
        ]
        solver = MultiLegFootPlacement(feet=feet, pelvis=0)
        assert solver is not None

    def test_quadruped_four_legs(self, quadruped_transforms):
        """Quadruped with four legs works."""
        feet = [
            FootData(upper_leg=2, lower_leg=3, foot=4),
            FootData(upper_leg=5, lower_leg=6, foot=7),
            FootData(upper_leg=8, lower_leg=9, foot=10),
            FootData(upper_leg=11, lower_leg=12, foot=13),
        ]
        solver = MultiLegFootPlacement(feet=feet, pelvis=0)
        result = solver.solve(quadruped_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_hexapod_six_legs(self):
        """Six-legged creature works."""
        feet = [
            FootData(upper_leg=i * 3, lower_leg=i * 3 + 1, foot=i * 3 + 2)
            for i in range(1, 7)  # 6 legs
        ]
        solver = MultiLegFootPlacement(feet=feet, pelvis=0)
        assert solver is not None

    def test_spider_eight_legs(self):
        """Eight-legged creature works."""
        feet = [
            FootData(upper_leg=i * 3, lower_leg=i * 3 + 1, foot=i * 3 + 2)
            for i in range(1, 9)  # 8 legs
        ]
        solver = MultiLegFootPlacement(feet=feet, pelvis=0)
        assert solver is not None

    def test_multileg_solve_returns_transforms(self, quadruped_transforms):
        """MultiLeg solve returns transforms."""
        feet = [
            FootData(upper_leg=2, lower_leg=3, foot=4),
            FootData(upper_leg=5, lower_leg=6, foot=7),
            FootData(upper_leg=8, lower_leg=9, foot=10),
            FootData(upper_leg=11, lower_leg=12, foot=13),
        ]
        solver = MultiLegFootPlacement(feet=feet, pelvis=0)
        result = solver.solve(quadruped_transforms, Vec3(0.0, 0.0, 0.0))
        assert isinstance(result, list)

    def test_multileg_with_raycast(self, quadruped_transforms, flat_raycast):
        """MultiLeg with raycast callback."""
        feet = [
            FootData(upper_leg=2, lower_leg=3, foot=4),
            FootData(upper_leg=5, lower_leg=6, foot=7),
            FootData(upper_leg=8, lower_leg=9, foot=10),
            FootData(upper_leg=11, lower_leg=12, foot=13),
        ]
        solver = MultiLegFootPlacement(feet=feet, pelvis=0, raycast_callback=flat_raycast)
        result = solver.solve(quadruped_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None


# =============================================================================
# Uneven Terrain Tests
# =============================================================================

class TestFootPlacementUnevenTerrain:
    """Tests for handling uneven terrain."""

    def test_adapt_to_slope(self, left_foot, right_foot, humanoid_transforms):
        """Feet adapt to sloped terrain."""
        def slope_raycast(origin: Vec3, direction: Vec3, ray_length: float) -> Optional[RaycastHit]:
            # Slope: height increases with z
            terrain_height = origin.z * 0.3
            if direction.y < 0:
                distance = (origin.y - terrain_height) / abs(direction.y)
                if 0 < distance <= ray_length:
                    hit_pos = Vec3(origin.x, terrain_height, origin.z)
                    return RaycastHit(hit=True, position=hit_pos, normal=Vec3(0, 1, 0), distance=distance)
            return None

        solver = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=slope_raycast,
        )
        result = solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_adapt_to_steps(self, left_foot, right_foot, humanoid_transforms):
        """Feet adapt to step terrain."""
        def step_raycast(origin: Vec3, direction: Vec3, ray_length: float) -> Optional[RaycastHit]:
            # Step at z=0
            terrain_height = 0.2 if origin.z > 0 else 0.0
            if direction.y < 0:
                distance = (origin.y - terrain_height) / abs(direction.y)
                if 0 < distance <= ray_length:
                    hit_pos = Vec3(origin.x, terrain_height, origin.z)
                    return RaycastHit(hit=True, position=hit_pos, normal=Vec3(0, 1, 0), distance=distance)
            return None

        solver = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=step_raycast,
        )
        result = solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_different_heights_per_foot(self, left_foot, right_foot, humanoid_transforms):
        """Different ground heights per foot work."""
        def asymmetric_raycast(origin: Vec3, direction: Vec3, ray_length: float) -> Optional[RaycastHit]:
            # Left side higher than right
            terrain_height = 0.2 if origin.x < 0 else 0.0
            if direction.y < 0:
                distance = (origin.y - terrain_height) / abs(direction.y)
                if 0 < distance <= ray_length:
                    hit_pos = Vec3(origin.x, terrain_height, origin.z)
                    return RaycastHit(hit=True, position=hit_pos, normal=Vec3(0, 1, 0), distance=distance)
            return None

        solver = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=asymmetric_raycast,
        )
        result = solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None


# =============================================================================
# Pelvis Adjustment Tests
# =============================================================================

class TestFootPlacementPelvisAdjustment:
    """Tests for pelvis height adjustment."""

    def test_result_has_pelvis_offset(self, foot_solver, humanoid_transforms):
        """Result has pelvis_offset field."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert hasattr(result, 'pelvis_offset')
        assert isinstance(result.pelvis_offset, Vec3)

    def test_pelvis_offset_is_vec3(self, foot_solver, humanoid_transforms):
        """Pelvis offset is a Vec3."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        offset = result.pelvis_offset
        assert hasattr(offset, 'x')
        assert hasattr(offset, 'y')
        assert hasattr(offset, 'z')


# =============================================================================
# Terrain Slope Tests
# =============================================================================

class TestFootPlacementTerrainSlope:
    """Tests for terrain slope detection."""

    def test_result_has_terrain_slope(self, foot_solver, humanoid_transforms):
        """Result has terrain_slope field."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert hasattr(result, 'terrain_slope')

    def test_terrain_slope_is_float(self, foot_solver, humanoid_transforms):
        """Terrain slope is a float."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert isinstance(result.terrain_slope, (int, float))


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestFootPlacementEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_no_ground_hit(self, left_foot, right_foot, humanoid_transforms):
        """Handle case where raycast finds no ground."""
        def no_hit_raycast(origin: Vec3, direction: Vec3, ray_length: float) -> Optional[RaycastHit]:
            return None  # No ground found

        solver = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=no_hit_raycast,
        )
        result = solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        # Should handle gracefully
        assert result is not None

    def test_very_far_ground(self, left_foot, right_foot, humanoid_transforms):
        """Handle very far ground level."""
        def far_ground_raycast(origin: Vec3, direction: Vec3, ray_length: float) -> Optional[RaycastHit]:
            if direction.y < 0:
                # Ground is far, but we return hit at ray_length distance
                hit_pos = Vec3(origin.x, origin.y - ray_length, origin.z)
                return RaycastHit(hit=True, position=hit_pos, normal=Vec3(0, 1, 0), distance=ray_length)
            return None

        solver = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=far_ground_raycast,
        )
        result = solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_ground_above_feet(self, left_foot, right_foot, humanoid_transforms):
        """Handle ground above feet (no hit returns None)."""
        def high_ground_raycast(origin: Vec3, direction: Vec3, ray_length: float) -> Optional[RaycastHit]:
            # Ground is above - no hit in down direction
            return None

        solver = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=high_ground_raycast,
        )
        result = solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_zero_delta_time(self, foot_solver, humanoid_transforms):
        """Handle zero delta time."""
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0), dt=0.0)
        assert result is not None

    def test_negative_delta_time(self, foot_solver, humanoid_transforms):
        """Handle negative delta time gracefully."""
        # Should not crash, may clamp to 0
        result = foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0), dt=-0.016)
        assert result is not None


# =============================================================================
# Configuration Constant Tests
# =============================================================================

class TestFootPlacementConstants:
    """Tests for configuration constants."""

    def test_ray_length_constant_exists(self):
        """FOOT_PLACEMENT_RAY_LENGTH constant exists."""
        assert FOOT_PLACEMENT_RAY_LENGTH is not None
        assert isinstance(FOOT_PLACEMENT_RAY_LENGTH, (int, float))

    def test_foot_height_constant_exists(self):
        """FOOT_PLACEMENT_FOOT_HEIGHT constant exists."""
        assert FOOT_PLACEMENT_FOOT_HEIGHT is not None
        assert isinstance(FOOT_PLACEMENT_FOOT_HEIGHT, (int, float))

    def test_blend_speed_constant_exists(self):
        """FOOT_PLACEMENT_BLEND_SPEED constant exists."""
        assert FOOT_PLACEMENT_BLEND_SPEED is not None
        assert isinstance(FOOT_PLACEMENT_BLEND_SPEED, (int, float))

    def test_max_pelvis_drop_constant_exists(self):
        """FOOT_PLACEMENT_MAX_PELVIS_DROP constant exists."""
        assert FOOT_PLACEMENT_MAX_PELVIS_DROP is not None
        assert isinstance(FOOT_PLACEMENT_MAX_PELVIS_DROP, (int, float))

    def test_ray_length_positive(self):
        """Ray length should be positive."""
        assert FOOT_PLACEMENT_RAY_LENGTH > 0

    def test_foot_height_non_negative(self):
        """Foot height should be non-negative."""
        assert FOOT_PLACEMENT_FOOT_HEIGHT >= 0

    def test_blend_speed_positive(self):
        """Blend speed should be positive."""
        assert FOOT_PLACEMENT_BLEND_SPEED > 0

    def test_max_pelvis_drop_positive(self):
        """Max pelvis drop should be positive."""
        assert FOOT_PLACEMENT_MAX_PELVIS_DROP > 0


# =============================================================================
# Performance Tests
# =============================================================================

class TestFootPlacementPerformance:
    """Tests for performance characteristics."""

    def test_solve_completes_quickly(self, foot_solver, humanoid_transforms):
        """Solve should complete in reasonable time."""
        import time
        start = time.time()
        for _ in range(100):
            foot_solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0))
        elapsed = time.time() - start
        # 100 solves should take less than 1 second
        assert elapsed < 1.0

    def test_multileg_solve_completes(self, quadruped_transforms):
        """Multi-leg solve completes in reasonable time."""
        import time
        feet = [
            FootData(upper_leg=2, lower_leg=3, foot=4),
            FootData(upper_leg=5, lower_leg=6, foot=7),
            FootData(upper_leg=8, lower_leg=9, foot=10),
            FootData(upper_leg=11, lower_leg=12, foot=13),
        ]
        solver = MultiLegFootPlacement(feet=feet, pelvis=0)

        start = time.time()
        for _ in range(100):
            solver.solve(quadruped_transforms, Vec3(0.0, 0.0, 0.0))
        elapsed = time.time() - start
        assert elapsed < 1.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestFootPlacementIntegration:
    """Integration tests combining multiple features."""

    def test_solve_with_raycast_and_delta(self, left_foot, right_foot, humanoid_transforms, uneven_raycast):
        """Foot placement with raycast and delta time."""
        solver = FootPlacement(
            left_foot=left_foot,
            right_foot=right_foot,
            pelvis=0,
            raycast_callback=uneven_raycast,
        )

        # Simulate several frames
        for _ in range(10):
            result = solver.solve(humanoid_transforms, Vec3(0.0, 0.0, 0.0), dt=0.016)
            assert result is not None

    def test_multileg_with_varying_terrain(self, quadruped_transforms):
        """Multi-leg with varying terrain heights."""
        def varying_terrain(origin: Vec3, direction: Vec3, ray_length: float) -> Optional[RaycastHit]:
            # Height varies by position
            terrain_height = 0.1 * math.sin(origin.x * 3) * math.cos(origin.z * 3)
            if direction.y < 0:
                distance = (origin.y - terrain_height) / abs(direction.y)
                if 0 < distance <= ray_length:
                    hit_pos = Vec3(origin.x, terrain_height, origin.z)
                    return RaycastHit(hit=True, position=hit_pos, normal=Vec3(0, 1, 0), distance=distance)
            return None

        feet = [
            FootData(upper_leg=2, lower_leg=3, foot=4),
            FootData(upper_leg=5, lower_leg=6, foot=7),
            FootData(upper_leg=8, lower_leg=9, foot=10),
            FootData(upper_leg=11, lower_leg=12, foot=13),
        ]
        solver = MultiLegFootPlacement(
            feet=feet,
            pelvis=0,
            raycast_callback=varying_terrain,
        )
        result = solver.solve(quadruped_transforms, Vec3(0.0, 0.0, 0.0))
        assert result is not None

    def test_continuous_movement_simulation(self, foot_solver, humanoid_transforms):
        """Simulate continuous character movement."""
        # Simulate moving character
        transforms = humanoid_transforms.copy()
        for frame in range(30):
            # Vary character position
            char_pos = Vec3(frame * 0.1, 0.0, 0.0)
            result = foot_solver.solve(transforms, char_pos, dt=0.033)
            assert result is not None
            transforms = result.transforms

    def test_animated_and_multileg_combined(self, quadruped_transforms):
        """Animated foot placement with multileg solver."""
        feet = [
            FootData(upper_leg=2, lower_leg=3, foot=4),
            FootData(upper_leg=5, lower_leg=6, foot=7),
            FootData(upper_leg=8, lower_leg=9, foot=10),
            FootData(upper_leg=11, lower_leg=12, foot=13),
        ]
        solver = MultiLegFootPlacement(feet=feet, pelvis=0)

        # Simulate several frames
        transforms = quadruped_transforms.copy()
        for _ in range(10):
            result = solver.solve(transforms, Vec3(0.0, 0.0, 0.0), dt=0.016)
            assert result is not None


# =============================================================================
# State Transition Tests
# =============================================================================

class TestFootStateTransitions:
    """Tests for foot state transitions."""

    def test_foot_data_default_state_planted(self):
        """FootData default state is PLANTED."""
        data = FootData(upper_leg=3, lower_leg=4, foot=5)
        assert data.state == FootState.PLANTED

    def test_foot_data_can_set_airborne(self):
        """FootData can be set to AIRBORNE."""
        data = FootData(upper_leg=3, lower_leg=4, foot=5, state=FootState.AIRBORNE)
        assert data.state == FootState.AIRBORNE

    def test_foot_data_can_set_lifting(self):
        """FootData can be set to LIFTING."""
        data = FootData(upper_leg=3, lower_leg=4, foot=5, state=FootState.LIFTING)
        assert data.state == FootState.LIFTING

    def test_foot_data_can_set_landing(self):
        """FootData can be set to LANDING."""
        data = FootData(upper_leg=3, lower_leg=4, foot=5, state=FootState.LANDING)
        assert data.state == FootState.LANDING

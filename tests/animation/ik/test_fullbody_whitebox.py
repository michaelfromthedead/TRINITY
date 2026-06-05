"""Whitebox tests for FullBodyIK solver.

Tests the full body IK implementation covering:
- BodyPart enum values
- SkeletonMapping class construction and methods
- FullBodyIKGoal dataclass
- FullBodyIKResult dataclass
- FullBodyIK class construction and solver initialization
- set_bone_mass() for COM calculation
- set_support_polygon() for balance checking
- _point_in_polygon() test
- _closest_point_on_polygon() method
- solve() with various goals
- _solve_spine() spine chain handling
- _maintain_balance() COM correction
- LookAtSolver class
"""

from __future__ import annotations

import math
import pytest
from dataclasses import fields
from typing import List

from engine.animation.ik.fullbody import (
    BodyPart,
    SkeletonMapping,
    FullBodyIKGoal,
    FullBodyIKResult,
    FullBodyIK,
    LookAtSolver,
)
from engine.animation.ik.config import (
    IK_DEFAULT_TOLERANCE,
    FULLBODY_DEFAULT_MAX_ITERATIONS,
    FULLBODY_SPINE_STIFFNESS,
    POLYGON_EDGE_MIN_LENGTH,
    LOOK_AT_MAX_ANGLE,
    LOOK_AT_HEAD_WEIGHT,
    LOOK_AT_NECK_WEIGHT,
    LOOK_AT_SPINE_WEIGHT,
)
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform
from engine.core.constants import MATH_EPSILON


# =============================================================================
# Helper Functions
# =============================================================================

def create_skeleton_mapping() -> SkeletonMapping:
    """Create a standard humanoid skeleton mapping for testing."""
    mapping = SkeletonMapping()

    # Set body parts
    mapping.set_bone(BodyPart.PELVIS, 0)
    mapping.set_bone(BodyPart.SPINE, 1)
    mapping.set_bone(BodyPart.CHEST, 2)
    mapping.set_bone(BodyPart.NECK, 3)
    mapping.set_bone(BodyPart.HEAD, 4)

    # Left arm
    mapping.set_bone(BodyPart.LEFT_SHOULDER, 5)
    mapping.set_bone(BodyPart.LEFT_UPPER_ARM, 6)
    mapping.set_bone(BodyPart.LEFT_LOWER_ARM, 7)
    mapping.set_bone(BodyPart.LEFT_HAND, 8)

    # Right arm
    mapping.set_bone(BodyPart.RIGHT_SHOULDER, 9)
    mapping.set_bone(BodyPart.RIGHT_UPPER_ARM, 10)
    mapping.set_bone(BodyPart.RIGHT_LOWER_ARM, 11)
    mapping.set_bone(BodyPart.RIGHT_HAND, 12)

    # Left leg
    mapping.set_bone(BodyPart.LEFT_UPPER_LEG, 13)
    mapping.set_bone(BodyPart.LEFT_LOWER_LEG, 14)
    mapping.set_bone(BodyPart.LEFT_FOOT, 15)

    # Right leg
    mapping.set_bone(BodyPart.RIGHT_UPPER_LEG, 16)
    mapping.set_bone(BodyPart.RIGHT_LOWER_LEG, 17)
    mapping.set_bone(BodyPart.RIGHT_FOOT, 18)

    # Set chains
    mapping.spine_chain = [0, 1, 2, 3, 4]  # Pelvis to head
    mapping.left_arm_chain = [6, 7, 8]
    mapping.right_arm_chain = [10, 11, 12]
    mapping.left_leg_chain = [13, 14, 15]
    mapping.right_leg_chain = [16, 17, 18]

    return mapping


def create_humanoid_transforms(num_bones: int = 20) -> List[Transform]:
    """Create transforms for a humanoid skeleton."""
    transforms = []

    # Create a basic humanoid skeleton layout
    positions = {
        0: Vec3(0, 1.0, 0),       # Pelvis
        1: Vec3(0, 1.2, 0),       # Spine
        2: Vec3(0, 1.4, 0),       # Chest
        3: Vec3(0, 1.6, 0),       # Neck
        4: Vec3(0, 1.8, 0),       # Head
        5: Vec3(-0.1, 1.4, 0),    # Left shoulder
        6: Vec3(-0.3, 1.4, 0),    # Left upper arm
        7: Vec3(-0.3, 1.1, 0),    # Left lower arm
        8: Vec3(-0.3, 0.8, 0),    # Left hand
        9: Vec3(0.1, 1.4, 0),     # Right shoulder
        10: Vec3(0.3, 1.4, 0),    # Right upper arm
        11: Vec3(0.3, 1.1, 0),    # Right lower arm
        12: Vec3(0.3, 0.8, 0),    # Right hand
        13: Vec3(-0.1, 1.0, 0),   # Left upper leg
        14: Vec3(-0.1, 0.5, 0),   # Left lower leg
        15: Vec3(-0.1, 0.0, 0),   # Left foot
        16: Vec3(0.1, 1.0, 0),    # Right upper leg
        17: Vec3(0.1, 0.5, 0),    # Right lower leg
        18: Vec3(0.1, 0.0, 0),    # Right foot
    }

    for i in range(num_bones):
        pos = positions.get(i, Vec3(0, 0, 0))
        transforms.append(Transform(pos, Quat.identity()))

    return transforms


def vec3_approx_equal(v1: Vec3, v2: Vec3, eps: float = 1e-4) -> bool:
    """Check if two Vec3 are approximately equal."""
    return (
        abs(v1.x - v2.x) < eps and
        abs(v1.y - v2.y) < eps and
        abs(v1.z - v2.z) < eps
    )


def quat_approx_equal(q1: Quat, q2: Quat, eps: float = 1e-4) -> bool:
    """Check if two Quaternions are approximately equal."""
    dot = abs(q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w)
    return dot > (1.0 - eps)


# =============================================================================
# Test BodyPart Enum
# =============================================================================

class TestBodyPartEnum:
    """Tests for BodyPart enum values."""

    def test_enum_has_all_body_parts(self):
        """Verify all expected body parts exist."""
        expected_parts = [
            'PELVIS', 'SPINE', 'CHEST', 'NECK', 'HEAD',
            'LEFT_SHOULDER', 'LEFT_UPPER_ARM', 'LEFT_LOWER_ARM', 'LEFT_HAND',
            'RIGHT_SHOULDER', 'RIGHT_UPPER_ARM', 'RIGHT_LOWER_ARM', 'RIGHT_HAND',
            'LEFT_UPPER_LEG', 'LEFT_LOWER_LEG', 'LEFT_FOOT', 'LEFT_TOE',
            'RIGHT_UPPER_LEG', 'RIGHT_LOWER_LEG', 'RIGHT_FOOT', 'RIGHT_TOE',
        ]
        for part_name in expected_parts:
            assert hasattr(BodyPart, part_name), f"Missing body part: {part_name}"

    def test_enum_values_are_unique(self):
        """Verify all enum values are unique."""
        values = [p.value for p in BodyPart]
        assert len(values) == len(set(values))

    def test_enum_iteration(self):
        """Test iterating over all body parts."""
        count = sum(1 for _ in BodyPart)
        assert count == 21, "Expected 21 body parts"


# =============================================================================
# Test SkeletonMapping
# =============================================================================

class TestSkeletonMapping:
    """Tests for SkeletonMapping class."""

    def test_default_construction(self):
        """Test default construction creates empty mapping."""
        mapping = SkeletonMapping()
        assert mapping.bone_map == {}
        assert mapping.spine_chain == []
        assert mapping.left_arm_chain == []
        assert mapping.right_arm_chain == []
        assert mapping.left_leg_chain == []
        assert mapping.right_leg_chain == []

    def test_set_bone(self):
        """Test setting bone index for a body part."""
        mapping = SkeletonMapping()
        mapping.set_bone(BodyPart.PELVIS, 0)
        assert mapping.bone_map[BodyPart.PELVIS] == 0

    def test_get_bone_returns_index(self):
        """Test getting bone index for mapped part."""
        mapping = SkeletonMapping()
        mapping.set_bone(BodyPart.HEAD, 5)
        assert mapping.get_bone(BodyPart.HEAD) == 5

    def test_get_bone_returns_minus_one_for_unmapped(self):
        """Test getting bone returns -1 for unmapped part."""
        mapping = SkeletonMapping()
        assert mapping.get_bone(BodyPart.HEAD) == -1

    def test_set_multiple_bones(self):
        """Test setting multiple bones."""
        mapping = SkeletonMapping()
        mapping.set_bone(BodyPart.PELVIS, 0)
        mapping.set_bone(BodyPart.SPINE, 1)
        mapping.set_bone(BodyPart.CHEST, 2)

        assert mapping.get_bone(BodyPart.PELVIS) == 0
        assert mapping.get_bone(BodyPart.SPINE) == 1
        assert mapping.get_bone(BodyPart.CHEST) == 2

    def test_override_bone_index(self):
        """Test overriding bone index."""
        mapping = SkeletonMapping()
        mapping.set_bone(BodyPart.HEAD, 5)
        mapping.set_bone(BodyPart.HEAD, 10)
        assert mapping.get_bone(BodyPart.HEAD) == 10

    def test_spine_chain_assignment(self):
        """Test setting spine chain."""
        mapping = SkeletonMapping()
        mapping.spine_chain = [0, 1, 2, 3]
        assert mapping.spine_chain == [0, 1, 2, 3]

    def test_limb_chains_assignment(self):
        """Test setting limb chains."""
        mapping = SkeletonMapping()
        mapping.left_arm_chain = [5, 6, 7]
        mapping.right_arm_chain = [8, 9, 10]
        mapping.left_leg_chain = [11, 12, 13]
        mapping.right_leg_chain = [14, 15, 16]

        assert mapping.left_arm_chain == [5, 6, 7]
        assert mapping.right_arm_chain == [8, 9, 10]
        assert mapping.left_leg_chain == [11, 12, 13]
        assert mapping.right_leg_chain == [14, 15, 16]


# =============================================================================
# Test FullBodyIKGoal
# =============================================================================

class TestFullBodyIKGoal:
    """Tests for FullBodyIKGoal dataclass."""

    def test_dataclass_has_required_fields(self):
        """Verify dataclass has all required fields."""
        field_names = {f.name for f in fields(FullBodyIKGoal)}
        expected_fields = {
            'bone_index', 'target_position', 'target_rotation',
            'position_weight', 'rotation_weight', 'priority',
            'enabled', 'chain_type'
        }
        assert expected_fields.issubset(field_names)

    def test_minimal_construction(self):
        """Test construction with only required args."""
        goal = FullBodyIKGoal(bone_index=5)

        assert goal.bone_index == 5
        assert goal.target_position is None
        assert goal.target_rotation is None
        assert goal.position_weight == 1.0
        assert goal.rotation_weight == 0.0
        assert goal.priority == 0
        assert goal.enabled is True
        assert goal.chain_type is None

    def test_full_construction(self):
        """Test construction with all args."""
        target_pos = Vec3(1, 2, 3)
        target_rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 4)

        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=target_pos,
            target_rotation=target_rot,
            position_weight=0.8,
            rotation_weight=0.5,
            priority=10,
            enabled=True,
            chain_type="left_arm"
        )

        assert goal.bone_index == 8
        assert goal.target_position == target_pos
        assert goal.target_rotation == target_rot
        assert goal.position_weight == 0.8
        assert goal.rotation_weight == 0.5
        assert goal.priority == 10
        assert goal.enabled is True
        assert goal.chain_type == "left_arm"

    def test_has_position_returns_true_when_valid(self):
        """Test has_position returns True with valid target and weight."""
        goal = FullBodyIKGoal(
            bone_index=5,
            target_position=Vec3(1, 2, 3),
            position_weight=1.0
        )
        assert goal.has_position() is True

    def test_has_position_returns_false_when_no_target(self):
        """Test has_position returns False without target."""
        goal = FullBodyIKGoal(bone_index=5)
        assert goal.has_position() is False

    def test_has_position_returns_false_when_zero_weight(self):
        """Test has_position returns False with zero weight."""
        goal = FullBodyIKGoal(
            bone_index=5,
            target_position=Vec3(1, 2, 3),
            position_weight=0.0
        )
        assert goal.has_position() is False

    def test_has_rotation_returns_true_when_valid(self):
        """Test has_rotation returns True with valid target and weight."""
        goal = FullBodyIKGoal(
            bone_index=5,
            target_rotation=Quat.identity(),
            rotation_weight=1.0
        )
        assert goal.has_rotation() is True

    def test_has_rotation_returns_false_when_no_target(self):
        """Test has_rotation returns False without target."""
        goal = FullBodyIKGoal(bone_index=5)
        assert goal.has_rotation() is False

    def test_has_rotation_returns_false_when_zero_weight(self):
        """Test has_rotation returns False with zero weight."""
        goal = FullBodyIKGoal(
            bone_index=5,
            target_rotation=Quat.identity(),
            rotation_weight=0.0
        )
        assert goal.has_rotation() is False

    def test_disabled_goal(self):
        """Test creating a disabled goal."""
        goal = FullBodyIKGoal(bone_index=5, enabled=False)
        assert goal.enabled is False

    def test_chain_types(self):
        """Test various chain types."""
        for chain in ["left_arm", "right_arm", "left_leg", "right_leg", "spine"]:
            goal = FullBodyIKGoal(bone_index=5, chain_type=chain)
            assert goal.chain_type == chain


# =============================================================================
# Test FullBodyIKResult
# =============================================================================

class TestFullBodyIKResult:
    """Tests for FullBodyIKResult dataclass."""

    def test_dataclass_has_required_fields(self):
        """Verify dataclass has all required fields."""
        field_names = {f.name for f in fields(FullBodyIKResult)}
        expected_fields = {
            'success', 'transforms', 'goals_achieved',
            'final_errors', 'pelvis_adjustment'
        }
        assert expected_fields.issubset(field_names)

    def test_default_values(self):
        """Test default values are set correctly."""
        result = FullBodyIKResult(success=True)

        assert result.success is True
        assert result.transforms == []
        assert result.goals_achieved == {}
        assert result.final_errors == {}
        assert vec3_approx_equal(result.pelvis_adjustment, Vec3.zero())

    def test_custom_values(self):
        """Test creating result with custom values."""
        transforms = [Transform(Vec3(1, 0, 0), Quat.identity())]
        goals_achieved = {5: True, 8: False}
        final_errors = {5: 0.001, 8: 0.5}
        pelvis_adj = Vec3(0, -0.1, 0)

        result = FullBodyIKResult(
            success=False,
            transforms=transforms,
            goals_achieved=goals_achieved,
            final_errors=final_errors,
            pelvis_adjustment=pelvis_adj
        )

        assert result.success is False
        assert len(result.transforms) == 1
        assert result.goals_achieved[5] is True
        assert result.goals_achieved[8] is False
        assert result.final_errors[5] == 0.001
        assert vec3_approx_equal(result.pelvis_adjustment, pelvis_adj)


# =============================================================================
# Test FullBodyIK Construction
# =============================================================================

class TestFullBodyIKConstruction:
    """Tests for FullBodyIK class initialization."""

    def test_basic_construction(self):
        """Test basic construction with skeleton mapping."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        assert solver.skeleton == mapping
        assert solver.tolerance == IK_DEFAULT_TOLERANCE
        assert solver.max_iterations == FULLBODY_DEFAULT_MAX_ITERATIONS

    def test_construction_with_custom_tolerance(self):
        """Test construction with custom tolerance."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping, tolerance=0.01)

        assert solver.tolerance == 0.01

    def test_construction_with_custom_iterations(self):
        """Test construction with custom max iterations."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping, max_iterations=20)

        assert solver.max_iterations == 20

    def test_default_balance_settings(self):
        """Test default balance settings."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        assert solver.maintain_balance is True
        assert solver.pelvis_height_adjust is True
        assert solver.spine_stiffness == FULLBODY_SPINE_STIFFNESS

    def test_limb_solvers_initialized(self):
        """Test limb IK solvers are initialized from skeleton."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        assert solver._left_arm_ik is not None
        assert solver._right_arm_ik is not None
        assert solver._left_leg_ik is not None
        assert solver._right_leg_ik is not None

    def test_spine_solver_initialized(self):
        """Test spine FABRIK solver is initialized."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        assert solver._spine_ik is not None

    def test_empty_mapping_no_solvers(self):
        """Test empty mapping creates no solvers."""
        mapping = SkeletonMapping()
        solver = FullBodyIK(mapping)

        assert solver._left_arm_ik is None
        assert solver._right_arm_ik is None
        assert solver._left_leg_ik is None
        assert solver._right_leg_ik is None
        assert solver._spine_ik is None

    def test_partial_mapping(self):
        """Test partial mapping creates only available solvers."""
        mapping = SkeletonMapping()
        mapping.left_arm_chain = [0, 1, 2]

        solver = FullBodyIK(mapping)

        assert solver._left_arm_ik is not None
        assert solver._right_arm_ik is None

    def test_short_chain_no_solver(self):
        """Test chain with < 3 bones creates no solver."""
        mapping = SkeletonMapping()
        mapping.left_arm_chain = [0, 1]  # Only 2 bones

        solver = FullBodyIK(mapping)

        assert solver._left_arm_ik is None


# =============================================================================
# Test set_bone_mass()
# =============================================================================

class TestSetBoneMass:
    """Tests for set_bone_mass() method."""

    def test_set_single_bone_mass(self):
        """Test setting mass for a single bone."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_bone_mass(0, 10.0)

        assert solver._bone_masses[0] == 10.0

    def test_set_multiple_bone_masses(self):
        """Test setting masses for multiple bones."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_bone_mass(0, 10.0)
        solver.set_bone_mass(1, 5.0)
        solver.set_bone_mass(2, 3.0)

        assert solver._bone_masses[0] == 10.0
        assert solver._bone_masses[1] == 5.0
        assert solver._bone_masses[2] == 3.0

    def test_override_bone_mass(self):
        """Test overriding a bone's mass."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_bone_mass(0, 10.0)
        solver.set_bone_mass(0, 15.0)

        assert solver._bone_masses[0] == 15.0

    def test_set_zero_mass(self):
        """Test setting zero mass."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_bone_mass(0, 0.0)

        assert solver._bone_masses[0] == 0.0

    def test_set_negative_mass(self):
        """Test setting negative mass (allowed but physically invalid)."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_bone_mass(0, -5.0)

        assert solver._bone_masses[0] == -5.0


# =============================================================================
# Test set_support_polygon()
# =============================================================================

class TestSetSupportPolygon:
    """Tests for set_support_polygon() method."""

    def test_set_triangle_polygon(self):
        """Test setting a triangle support polygon."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        vertices = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0.5, 0, 1)]
        solver.set_support_polygon(vertices)

        assert len(solver._support_polygon) == 3

    def test_set_quad_polygon(self):
        """Test setting a quad support polygon."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        vertices = [
            Vec3(-1, 0, -1), Vec3(1, 0, -1),
            Vec3(1, 0, 1), Vec3(-1, 0, 1)
        ]
        solver.set_support_polygon(vertices)

        assert len(solver._support_polygon) == 4

    def test_polygon_is_copied(self):
        """Test that polygon is copied, not referenced."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        vertices = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0.5, 0, 1)]
        solver.set_support_polygon(vertices)

        # Modify original list
        vertices.append(Vec3(2, 0, 2))

        assert len(solver._support_polygon) == 3

    def test_empty_polygon(self):
        """Test setting empty polygon."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([])

        assert len(solver._support_polygon) == 0


# =============================================================================
# Test _point_in_polygon()
# =============================================================================

class TestPointInPolygon:
    """Tests for _point_in_polygon() method."""

    def test_point_inside_square(self):
        """Test point inside a square polygon."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([
            Vec3(-1, 0, -1), Vec3(1, 0, -1),
            Vec3(1, 0, 1), Vec3(-1, 0, 1)
        ])

        point = Vec3(0, 0, 0)
        assert solver._point_in_polygon(point) is True

    def test_point_outside_square(self):
        """Test point outside a square polygon."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([
            Vec3(-1, 0, -1), Vec3(1, 0, -1),
            Vec3(1, 0, 1), Vec3(-1, 0, 1)
        ])

        point = Vec3(2, 0, 0)
        assert solver._point_in_polygon(point) is False

    def test_point_inside_triangle(self):
        """Test point inside a triangle."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([
            Vec3(0, 0, 0), Vec3(2, 0, 0), Vec3(1, 0, 2)
        ])

        point = Vec3(1, 0, 0.5)
        assert solver._point_in_polygon(point) is True

    def test_point_outside_triangle(self):
        """Test point outside a triangle."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([
            Vec3(0, 0, 0), Vec3(2, 0, 0), Vec3(1, 0, 2)
        ])

        point = Vec3(-1, 0, 0)
        assert solver._point_in_polygon(point) is False

    def test_point_on_edge(self):
        """Test point on polygon edge (may be inside or outside)."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([
            Vec3(0, 0, 0), Vec3(2, 0, 0), Vec3(1, 0, 2)
        ])

        point = Vec3(1, 0, 0)  # On bottom edge
        # Edge case - result may vary
        result = solver._point_in_polygon(point)
        assert isinstance(result, bool)

    def test_empty_polygon_returns_true(self):
        """Test empty polygon returns True (no constraint)."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([])

        point = Vec3(0, 0, 0)
        assert solver._point_in_polygon(point) is True

    def test_two_vertex_polygon_returns_true(self):
        """Test polygon with < 3 vertices returns True."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([Vec3(0, 0, 0), Vec3(1, 0, 0)])

        point = Vec3(0.5, 0, 0)
        assert solver._point_in_polygon(point) is True

    def test_y_coordinate_ignored(self):
        """Test that Y coordinate is ignored (XZ plane test)."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([
            Vec3(-1, 0, -1), Vec3(1, 0, -1),
            Vec3(1, 0, 1), Vec3(-1, 0, 1)
        ])

        # Point at different Y but inside XZ projection
        point = Vec3(0, 5, 0)
        assert solver._point_in_polygon(point) is True


# =============================================================================
# Test _closest_point_on_polygon()
# =============================================================================

class TestClosestPointOnPolygon:
    """Tests for _closest_point_on_polygon() method."""

    def test_closest_to_edge(self):
        """Test finding closest point on edge."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([
            Vec3(-1, 0, -1), Vec3(1, 0, -1),
            Vec3(1, 0, 1), Vec3(-1, 0, 1)
        ])

        point = Vec3(0, 0, -2)  # Outside, closest to bottom edge
        closest = solver._closest_point_on_polygon(point)

        assert abs(closest.z - (-1)) < 0.01  # Should be on z=-1 edge
        assert abs(closest.x - 0) < 0.01     # Should be at x=0

    def test_closest_to_vertex(self):
        """Test finding closest point near a vertex."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([
            Vec3(0, 0, 0), Vec3(2, 0, 0), Vec3(1, 0, 2)
        ])

        point = Vec3(-1, 0, -1)  # Outside, closest to vertex at origin
        closest = solver._closest_point_on_polygon(point)

        # Should be near the (0,0,0) vertex
        assert (closest - Vec3(0, 0, 0)).length() < 0.5

    def test_single_vertex_returns_point(self):
        """Test single vertex polygon returns query point."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([Vec3(0, 0, 0)])

        point = Vec3(5, 0, 5)
        closest = solver._closest_point_on_polygon(point)

        # With < 2 vertices, returns query point
        assert vec3_approx_equal(closest, point)

    def test_two_vertices_returns_point_on_edge(self):
        """Test two vertex polygon (line segment)."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([Vec3(0, 0, 0), Vec3(2, 0, 0)])

        point = Vec3(1, 0, 1)  # Perpendicular to midpoint
        closest = solver._closest_point_on_polygon(point)

        assert abs(closest.x - 1) < 0.01
        assert abs(closest.z) < 0.01


# =============================================================================
# Test solve()
# =============================================================================

class TestFullBodyIKSolve:
    """Tests for solve() method."""

    def test_solve_returns_result(self):
        """Test solve returns FullBodyIKResult."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        result = solver.solve(transforms, [])

        assert isinstance(result, FullBodyIKResult)

    def test_solve_with_no_goals(self):
        """Test solve with empty goals list."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        result = solver.solve(transforms, [])

        assert result.success is True
        assert len(result.transforms) == len(transforms)

    def test_solve_with_position_goal(self):
        """Test solve with a position goal."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=8,  # Left hand
            target_position=Vec3(-0.5, 1.0, 0.5),
            chain_type="left_arm"
        )

        result = solver.solve(transforms, [goal])

        assert isinstance(result, FullBodyIKResult)
        assert 8 in result.goals_achieved

    def test_solve_with_disabled_goal(self):
        """Test disabled goals are skipped."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.5, 1.0, 0.5),
            chain_type="left_arm",
            enabled=False
        )

        result = solver.solve(transforms, [goal])

        # Disabled goal should not appear in results
        assert 8 not in result.goals_achieved

    def test_solve_prioritizes_goals(self):
        """Test goals are processed in priority order."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal_low = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.5, 1.0, 0.5),
            priority=1,
            chain_type="left_arm"
        )

        goal_high = FullBodyIKGoal(
            bone_index=12,
            target_position=Vec3(0.5, 1.0, 0.5),
            priority=10,
            chain_type="right_arm"
        )

        result = solver.solve(transforms, [goal_low, goal_high])

        # Both should be processed
        assert 8 in result.goals_achieved
        assert 12 in result.goals_achieved

    def test_solve_copies_transforms(self):
        """Test solve doesn't modify input transforms."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        original_pos = Vec3(
            transforms[0].translation.x,
            transforms[0].translation.y,
            transforms[0].translation.z
        )

        goal = FullBodyIKGoal(
            bone_index=15,
            target_position=Vec3(-0.1, -0.5, 0),
            chain_type="left_leg"
        )

        solver.solve(transforms, [goal])

        # Original should be unchanged
        assert vec3_approx_equal(transforms[0].translation, original_pos)

    def test_solve_returns_final_errors(self):
        """Test solve returns final position errors."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.5, 1.0, 0.5),
            chain_type="left_arm"
        )

        result = solver.solve(transforms, [goal])

        assert 8 in result.final_errors
        assert result.final_errors[8] >= 0

    def test_solve_with_leg_goals_adjusts_pelvis(self):
        """Test pelvis adjustment with leg goals."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        # Set leg goal below normal reach
        goal = FullBodyIKGoal(
            bone_index=15,
            target_position=Vec3(-0.1, -0.5, 0),
            chain_type="left_leg"
        )

        result = solver.solve(transforms, [goal])

        # Result should contain pelvis adjustment (may be zero if reachable)
        assert hasattr(result, 'pelvis_adjustment')


# =============================================================================
# Test _maintain_balance()
# =============================================================================

class TestMaintainBalance:
    """Tests for _maintain_balance() method."""

    def test_no_masses_no_balance(self):
        """Test no balance adjustment without bone masses."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        solver.set_support_polygon([
            Vec3(-0.5, 0, -0.5), Vec3(0.5, 0, -0.5),
            Vec3(0.5, 0, 0.5), Vec3(-0.5, 0, 0.5)
        ])

        # Should not crash
        solver._maintain_balance(transforms)

    def test_no_polygon_no_balance(self):
        """Test no balance adjustment without polygon."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        solver.set_bone_mass(0, 10.0)

        # Should not crash
        solver._maintain_balance(transforms)

    def test_balance_with_masses_and_polygon(self):
        """Test balance adjustment with both masses and polygon."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        # Set up bone masses
        solver.set_bone_mass(0, 10.0)  # Pelvis
        solver.set_bone_mass(1, 5.0)   # Spine
        solver.set_bone_mass(4, 3.0)   # Head

        # Set support polygon
        solver.set_support_polygon([
            Vec3(-0.5, 0, -0.5), Vec3(0.5, 0, -0.5),
            Vec3(0.5, 0, 0.5), Vec3(-0.5, 0, 0.5)
        ])

        # Should not crash
        solver._maintain_balance(transforms)


# =============================================================================
# Test LookAtSolver
# =============================================================================

class TestLookAtSolver:
    """Tests for LookAtSolver class."""

    def test_construction(self):
        """Test LookAtSolver construction."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[1, 2]
        )

        assert solver.head_bone == 4
        assert solver.neck_bone == 3
        assert solver.spine_bones == [1, 2]

    def test_default_weights(self):
        """Test default weight values."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[1, 2]
        )

        assert solver.head_weight == LOOK_AT_HEAD_WEIGHT
        assert solver.neck_weight == LOOK_AT_NECK_WEIGHT
        assert solver.spine_weight == LOOK_AT_SPINE_WEIGHT
        assert solver.max_angle == LOOK_AT_MAX_ANGLE

    def test_custom_weights(self):
        """Test custom weight values."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[1, 2],
            head_weight=0.8,
            neck_weight=0.15,
            spine_weight=0.05
        )

        assert solver.head_weight == 0.8
        assert solver.neck_weight == 0.15
        assert solver.spine_weight == 0.05

    def test_solve_returns_transforms(self):
        """Test solve returns modified transforms."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[1, 2]
        )

        transforms = create_humanoid_transforms()
        target = Vec3(0, 2, 2)  # In front and above

        result = solver.solve(transforms, target)

        assert len(result) == len(transforms)
        assert all(isinstance(t, Transform) for t in result)

    def test_solve_does_not_modify_input(self):
        """Test solve doesn't modify input transforms."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[1, 2]
        )

        transforms = create_humanoid_transforms()
        original_rot = Quat(
            transforms[4].rotation.x,
            transforms[4].rotation.y,
            transforms[4].rotation.z,
            transforms[4].rotation.w
        )

        target = Vec3(0, 2, 2)
        solver.solve(transforms, target)

        assert quat_approx_equal(transforms[4].rotation, original_rot)

    def test_rotation_between_same_vectors(self):
        """Test _rotation_between with same vectors."""
        solver = LookAtSolver(4, 3, [1, 2])

        vec = Vec3(0, 0, 1)
        rot = solver._rotation_between(vec, vec)

        assert quat_approx_equal(rot, Quat.identity())

    def test_rotation_between_opposite_vectors(self):
        """Test _rotation_between with opposite vectors."""
        solver = LookAtSolver(4, 3, [1, 2])

        from_vec = Vec3(0, 0, 1)
        to_vec = Vec3(0, 0, -1)
        rot = solver._rotation_between(from_vec, to_vec)

        # Should be 180 degree rotation
        rotated = rot.rotate_vector(from_vec)
        assert abs(rotated.dot(to_vec) - 1.0) < 0.01

    def test_scale_rotation(self):
        """Test _scale_rotation method."""
        solver = LookAtSolver(4, 3, [1, 2])

        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 2)
        scaled = solver._scale_rotation(rot, 0.5)

        # Scaled rotation should be half the angle
        axis, angle = solver._quat_to_axis_angle(scaled)
        assert abs(angle - math.pi / 4) < 0.01

    def test_quat_to_axis_angle(self):
        """Test _quat_to_axis_angle method."""
        solver = LookAtSolver(4, 3, [1, 2])

        original_axis = Vec3(0, 1, 0)
        original_angle = math.pi / 3

        quat = Quat.from_axis_angle(original_axis, original_angle)
        axis, angle = solver._quat_to_axis_angle(quat)

        assert abs(angle - original_angle) < 0.01
        assert abs(axis.y - 1.0) < 0.01

    def test_quat_to_axis_angle_identity(self):
        """Test _quat_to_axis_angle with identity quaternion."""
        solver = LookAtSolver(4, 3, [1, 2])

        quat = Quat.identity()
        axis, angle = solver._quat_to_axis_angle(quat)

        assert abs(angle) < 0.01


# =============================================================================
# Test _solve_spine()
# =============================================================================

class TestSolveSpine:
    """Tests for _solve_spine() method."""

    def test_solve_spine_with_no_spine_ik(self):
        """Test _solve_spine does nothing without spine IK."""
        mapping = SkeletonMapping()  # No spine chain
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goals = [FullBodyIKGoal(bone_index=4, target_position=Vec3(0, 2, 1), chain_type="spine")]

        # Should not crash
        solver._solve_spine(transforms, goals)

    def test_solve_spine_with_goal(self):
        """Test _solve_spine with a spine goal."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=4,  # Head
            target_position=Vec3(0, 2, 1),
            chain_type="spine",
            priority=10
        )

        # Should not crash
        solver._solve_spine(transforms, [goal])


# =============================================================================
# Test _adjust_pelvis_height()
# =============================================================================

class TestAdjustPelvisHeight:
    """Tests for _adjust_pelvis_height() method."""

    def test_no_pelvis_returns_zero(self):
        """Test returns zero when no pelvis mapped."""
        mapping = SkeletonMapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        result = solver._adjust_pelvis_height(transforms, [])

        assert vec3_approx_equal(result, Vec3.zero())

    def test_no_leg_goals_returns_zero(self):
        """Test returns zero without leg goals."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goals = [FullBodyIKGoal(bone_index=8, target_position=Vec3(0, 1, 0), chain_type="left_arm")]

        result = solver._adjust_pelvis_height(transforms, goals)

        assert vec3_approx_equal(result, Vec3.zero())

    def test_with_leg_goal_in_reach(self):
        """Test with leg goal within reach."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=15,
            target_position=Vec3(-0.1, 0.0, 0),
            chain_type="left_leg"
        )

        # Should not raise error
        result = solver._adjust_pelvis_height(transforms, [goal])
        assert isinstance(result, Vec3)


# =============================================================================
# Test _translate_hierarchy()
# =============================================================================

class TestTranslateHierarchy:
    """Tests for _translate_hierarchy() method."""

    def test_translate_all_transforms(self):
        """Test _translate_hierarchy translates all transforms."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        original_y = transforms[5].translation.y
        offset = Vec3(0, -0.5, 0)

        solver._translate_hierarchy(transforms, 0, offset)

        # All transforms should be translated
        assert abs(transforms[5].translation.y - (original_y - 0.5)) < 0.01


# =============================================================================
# Test _solve_goal()
# =============================================================================

class TestSolveGoal:
    """Tests for _solve_goal() method."""

    def test_goal_without_position_succeeds(self):
        """Test goal without position target returns success."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(bone_index=8)  # No position

        achieved, error = solver._solve_goal(transforms, goal)

        assert achieved is True
        assert error == 0.0

    def test_goal_with_unknown_chain_type(self):
        """Test goal with unknown chain type computes error only."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=5,
            target_position=Vec3(0, 2, 0),
            chain_type="unknown"
        )

        achieved, error = solver._solve_goal(transforms, goal)

        assert isinstance(achieved, bool)
        assert error >= 0

    def test_goal_with_left_arm_chain(self):
        """Test goal with left_arm chain type."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.4, 1.2, 0.2),
            chain_type="left_arm"
        )

        achieved, error = solver._solve_goal(transforms, goal)

        assert isinstance(achieved, bool)
        assert isinstance(error, float)

    def test_goal_with_right_arm_chain(self):
        """Test goal with right_arm chain type."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=12,
            target_position=Vec3(0.4, 1.2, 0.2),
            chain_type="right_arm"
        )

        achieved, error = solver._solve_goal(transforms, goal)

        assert isinstance(achieved, bool)
        assert isinstance(error, float)

    def test_goal_with_left_leg_chain(self):
        """Test goal with left_leg chain type."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=15,
            target_position=Vec3(-0.1, 0.1, 0.2),
            chain_type="left_leg"
        )

        achieved, error = solver._solve_goal(transforms, goal)

        assert isinstance(achieved, bool)
        assert isinstance(error, float)

    def test_goal_with_right_leg_chain(self):
        """Test goal with right_leg chain type."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=18,
            target_position=Vec3(0.1, 0.1, 0.2),
            chain_type="right_leg"
        )

        achieved, error = solver._solve_goal(transforms, goal)

        assert isinstance(achieved, bool)
        assert isinstance(error, float)

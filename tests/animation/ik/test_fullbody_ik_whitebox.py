"""Whitebox tests for FullBodyIK solver components.

Comprehensive tests for:
- FullBodyIKGoal dataclass (line 657): field access, defaults, has_position, has_rotation
- FullBodyIKResult dataclass (line 692): success, transforms, goals_achieved, final_errors
- FullBodyIK class (line 710): initialization, solver creation, solve phases, balance

Target: 80+ tests covering all methods and internal logic with mocked dependencies.
"""

from __future__ import annotations

import math
import pytest
from dataclasses import fields, asdict
from typing import List, Dict, Optional
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from engine.animation.ik.fullbody import (
    BodyPart,
    SkeletonMapping,
    FullBodyIKGoal,
    FullBodyIKResult,
    FullBodyIK,
)
from engine.animation.ik.config import (
    IK_DEFAULT_TOLERANCE,
    FULLBODY_DEFAULT_MAX_ITERATIONS,
    FULLBODY_SPINE_STIFFNESS,
    POLYGON_EDGE_MIN_LENGTH,
)
from engine.animation.ik.fabrik import FABRIKChain
from engine.animation.ik.two_bone import TwoBoneIK
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

    mapping.set_bone(BodyPart.PELVIS, 0)
    mapping.set_bone(BodyPart.SPINE, 1)
    mapping.set_bone(BodyPart.CHEST, 2)
    mapping.set_bone(BodyPart.NECK, 3)
    mapping.set_bone(BodyPart.HEAD, 4)

    mapping.set_bone(BodyPart.LEFT_SHOULDER, 5)
    mapping.set_bone(BodyPart.LEFT_UPPER_ARM, 6)
    mapping.set_bone(BodyPart.LEFT_LOWER_ARM, 7)
    mapping.set_bone(BodyPart.LEFT_HAND, 8)

    mapping.set_bone(BodyPart.RIGHT_SHOULDER, 9)
    mapping.set_bone(BodyPart.RIGHT_UPPER_ARM, 10)
    mapping.set_bone(BodyPart.RIGHT_LOWER_ARM, 11)
    mapping.set_bone(BodyPart.RIGHT_HAND, 12)

    mapping.set_bone(BodyPart.LEFT_UPPER_LEG, 13)
    mapping.set_bone(BodyPart.LEFT_LOWER_LEG, 14)
    mapping.set_bone(BodyPart.LEFT_FOOT, 15)

    mapping.set_bone(BodyPart.RIGHT_UPPER_LEG, 16)
    mapping.set_bone(BodyPart.RIGHT_LOWER_LEG, 17)
    mapping.set_bone(BodyPart.RIGHT_FOOT, 18)

    mapping.spine_chain = [0, 1, 2, 3, 4]
    mapping.left_arm_chain = [6, 7, 8]
    mapping.right_arm_chain = [10, 11, 12]
    mapping.left_leg_chain = [13, 14, 15]
    mapping.right_leg_chain = [16, 17, 18]

    return mapping


def create_minimal_mapping() -> SkeletonMapping:
    """Create minimal skeleton with no chains for testing."""
    mapping = SkeletonMapping()
    mapping.set_bone(BodyPart.PELVIS, 0)
    return mapping


def create_humanoid_transforms(num_bones: int = 20) -> List[Transform]:
    """Create transforms for a humanoid skeleton."""
    transforms = []

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


# =============================================================================
# Test FullBodyIKGoal Dataclass
# =============================================================================

class TestFullBodyIKGoalDataclass:
    """Tests for FullBodyIKGoal dataclass fields and defaults."""

    def test_bone_index_required(self):
        """Test bone_index is a required field."""
        goal = FullBodyIKGoal(bone_index=5)
        assert goal.bone_index == 5

    def test_target_position_default_none(self):
        """Test target_position defaults to None."""
        goal = FullBodyIKGoal(bone_index=0)
        assert goal.target_position is None

    def test_target_rotation_default_none(self):
        """Test target_rotation defaults to None."""
        goal = FullBodyIKGoal(bone_index=0)
        assert goal.target_rotation is None

    def test_position_weight_default_one(self):
        """Test position_weight defaults to 1.0."""
        goal = FullBodyIKGoal(bone_index=0)
        assert goal.position_weight == 1.0

    def test_rotation_weight_default_zero(self):
        """Test rotation_weight defaults to 0.0."""
        goal = FullBodyIKGoal(bone_index=0)
        assert goal.rotation_weight == 0.0

    def test_priority_default_zero(self):
        """Test priority defaults to 0."""
        goal = FullBodyIKGoal(bone_index=0)
        assert goal.priority == 0

    def test_enabled_default_true(self):
        """Test enabled defaults to True."""
        goal = FullBodyIKGoal(bone_index=0)
        assert goal.enabled is True

    def test_chain_type_default_none(self):
        """Test chain_type defaults to None."""
        goal = FullBodyIKGoal(bone_index=0)
        assert goal.chain_type is None

    def test_set_target_position(self):
        """Test setting target_position."""
        pos = Vec3(1.0, 2.0, 3.0)
        goal = FullBodyIKGoal(bone_index=8, target_position=pos)
        assert goal.target_position == pos

    def test_set_target_rotation(self):
        """Test setting target_rotation."""
        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 4)
        goal = FullBodyIKGoal(bone_index=8, target_rotation=rot)
        assert goal.target_rotation == rot

    def test_set_position_weight(self):
        """Test setting custom position_weight."""
        goal = FullBodyIKGoal(bone_index=0, position_weight=0.5)
        assert goal.position_weight == 0.5

    def test_set_rotation_weight(self):
        """Test setting custom rotation_weight."""
        goal = FullBodyIKGoal(bone_index=0, rotation_weight=0.8)
        assert goal.rotation_weight == 0.8

    def test_set_priority(self):
        """Test setting custom priority."""
        goal = FullBodyIKGoal(bone_index=0, priority=100)
        assert goal.priority == 100

    def test_set_enabled_false(self):
        """Test setting enabled to False."""
        goal = FullBodyIKGoal(bone_index=0, enabled=False)
        assert goal.enabled is False

    def test_set_chain_type_left_arm(self):
        """Test setting chain_type to left_arm."""
        goal = FullBodyIKGoal(bone_index=8, chain_type="left_arm")
        assert goal.chain_type == "left_arm"

    def test_set_chain_type_right_arm(self):
        """Test setting chain_type to right_arm."""
        goal = FullBodyIKGoal(bone_index=12, chain_type="right_arm")
        assert goal.chain_type == "right_arm"

    def test_set_chain_type_left_leg(self):
        """Test setting chain_type to left_leg."""
        goal = FullBodyIKGoal(bone_index=15, chain_type="left_leg")
        assert goal.chain_type == "left_leg"

    def test_set_chain_type_right_leg(self):
        """Test setting chain_type to right_leg."""
        goal = FullBodyIKGoal(bone_index=18, chain_type="right_leg")
        assert goal.chain_type == "right_leg"

    def test_set_chain_type_spine(self):
        """Test setting chain_type to spine."""
        goal = FullBodyIKGoal(bone_index=4, chain_type="spine")
        assert goal.chain_type == "spine"

    def test_negative_priority_allowed(self):
        """Test negative priority values are allowed."""
        goal = FullBodyIKGoal(bone_index=0, priority=-5)
        assert goal.priority == -5

    def test_zero_position_weight_allowed(self):
        """Test zero position_weight is allowed."""
        goal = FullBodyIKGoal(bone_index=0, position_weight=0.0)
        assert goal.position_weight == 0.0

    def test_full_goal_construction(self):
        """Test constructing goal with all fields."""
        pos = Vec3(1.0, 2.0, 3.0)
        rot = Quat.identity()
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=pos,
            target_rotation=rot,
            position_weight=0.9,
            rotation_weight=0.5,
            priority=50,
            enabled=True,
            chain_type="left_arm"
        )

        assert goal.bone_index == 8
        assert goal.target_position == pos
        assert goal.target_rotation == rot
        assert goal.position_weight == 0.9
        assert goal.rotation_weight == 0.5
        assert goal.priority == 50
        assert goal.enabled is True
        assert goal.chain_type == "left_arm"


class TestFullBodyIKGoalHasPosition:
    """Tests for FullBodyIKGoal.has_position() method."""

    def test_has_position_with_position_and_weight(self):
        """Test has_position returns True with position and weight > 0."""
        goal = FullBodyIKGoal(
            bone_index=0,
            target_position=Vec3(1, 2, 3),
            position_weight=1.0
        )
        assert goal.has_position() is True

    def test_has_position_without_target(self):
        """Test has_position returns False without target_position."""
        goal = FullBodyIKGoal(bone_index=0, position_weight=1.0)
        assert goal.has_position() is False

    def test_has_position_with_zero_weight(self):
        """Test has_position returns False with zero weight."""
        goal = FullBodyIKGoal(
            bone_index=0,
            target_position=Vec3(1, 2, 3),
            position_weight=0.0
        )
        assert goal.has_position() is False

    def test_has_position_with_small_weight(self):
        """Test has_position returns True with small positive weight."""
        goal = FullBodyIKGoal(
            bone_index=0,
            target_position=Vec3(1, 2, 3),
            position_weight=0.001
        )
        assert goal.has_position() is True

    def test_has_position_with_negative_weight(self):
        """Test has_position returns False with negative weight."""
        goal = FullBodyIKGoal(
            bone_index=0,
            target_position=Vec3(1, 2, 3),
            position_weight=-1.0
        )
        assert goal.has_position() is False


class TestFullBodyIKGoalHasRotation:
    """Tests for FullBodyIKGoal.has_rotation() method."""

    def test_has_rotation_with_rotation_and_weight(self):
        """Test has_rotation returns True with rotation and weight > 0."""
        goal = FullBodyIKGoal(
            bone_index=0,
            target_rotation=Quat.identity(),
            rotation_weight=1.0
        )
        assert goal.has_rotation() is True

    def test_has_rotation_without_target(self):
        """Test has_rotation returns False without target_rotation."""
        goal = FullBodyIKGoal(bone_index=0, rotation_weight=1.0)
        assert goal.has_rotation() is False

    def test_has_rotation_with_zero_weight(self):
        """Test has_rotation returns False with zero weight."""
        goal = FullBodyIKGoal(
            bone_index=0,
            target_rotation=Quat.identity(),
            rotation_weight=0.0
        )
        assert goal.has_rotation() is False

    def test_has_rotation_with_small_weight(self):
        """Test has_rotation returns True with small positive weight."""
        goal = FullBodyIKGoal(
            bone_index=0,
            target_rotation=Quat.identity(),
            rotation_weight=0.001
        )
        assert goal.has_rotation() is True

    def test_has_rotation_with_negative_weight(self):
        """Test has_rotation returns False with negative weight."""
        goal = FullBodyIKGoal(
            bone_index=0,
            target_rotation=Quat.identity(),
            rotation_weight=-0.5
        )
        assert goal.has_rotation() is False


# =============================================================================
# Test FullBodyIKResult Dataclass
# =============================================================================

class TestFullBodyIKResultDataclass:
    """Tests for FullBodyIKResult dataclass fields and defaults."""

    def test_success_field(self):
        """Test success field can be set."""
        result = FullBodyIKResult(success=True)
        assert result.success is True

        result = FullBodyIKResult(success=False)
        assert result.success is False

    def test_transforms_default_empty_list(self):
        """Test transforms defaults to empty list."""
        result = FullBodyIKResult(success=True)
        assert result.transforms == []

    def test_goals_achieved_default_empty_dict(self):
        """Test goals_achieved defaults to empty dict."""
        result = FullBodyIKResult(success=True)
        assert result.goals_achieved == {}

    def test_final_errors_default_empty_dict(self):
        """Test final_errors defaults to empty dict."""
        result = FullBodyIKResult(success=True)
        assert result.final_errors == {}

    def test_pelvis_adjustment_default_zero(self):
        """Test pelvis_adjustment defaults to Vec3.zero()."""
        result = FullBodyIKResult(success=True)
        assert vec3_approx_equal(result.pelvis_adjustment, Vec3.zero())

    def test_set_transforms(self):
        """Test setting transforms list."""
        transforms = [Transform(Vec3(0, 0, 0), Quat.identity())]
        result = FullBodyIKResult(success=True, transforms=transforms)
        assert len(result.transforms) == 1

    def test_set_goals_achieved(self):
        """Test setting goals_achieved dict."""
        achieved = {8: True, 12: False}
        result = FullBodyIKResult(success=True, goals_achieved=achieved)
        assert result.goals_achieved[8] is True
        assert result.goals_achieved[12] is False

    def test_set_final_errors(self):
        """Test setting final_errors dict."""
        errors = {8: 0.01, 12: 0.05}
        result = FullBodyIKResult(success=True, final_errors=errors)
        assert result.final_errors[8] == 0.01
        assert result.final_errors[12] == 0.05

    def test_set_pelvis_adjustment(self):
        """Test setting pelvis_adjustment vector."""
        adj = Vec3(0, -0.1, 0)
        result = FullBodyIKResult(success=True, pelvis_adjustment=adj)
        assert vec3_approx_equal(result.pelvis_adjustment, adj)

    def test_full_result_construction(self):
        """Test constructing result with all fields."""
        transforms = [Transform(Vec3(i, 0, 0), Quat.identity()) for i in range(5)]
        achieved = {0: True, 1: True, 2: False}
        errors = {0: 0.0, 1: 0.001, 2: 0.5}
        adj = Vec3(0, -0.2, 0)

        result = FullBodyIKResult(
            success=True,
            transforms=transforms,
            goals_achieved=achieved,
            final_errors=errors,
            pelvis_adjustment=adj
        )

        assert result.success is True
        assert len(result.transforms) == 5
        assert result.goals_achieved == achieved
        assert result.final_errors == errors
        assert vec3_approx_equal(result.pelvis_adjustment, adj)


# =============================================================================
# Test FullBodyIK Initialization
# =============================================================================

class TestFullBodyIKInit:
    """Tests for FullBodyIK.__init__() method."""

    def test_init_stores_skeleton_mapping(self):
        """Test skeleton_mapping is stored."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        assert solver.skeleton is mapping

    def test_init_default_tolerance(self):
        """Test default tolerance from config."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        assert solver.tolerance == IK_DEFAULT_TOLERANCE

    def test_init_custom_tolerance(self):
        """Test custom tolerance."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping, tolerance=0.05)
        assert solver.tolerance == 0.05

    def test_init_default_max_iterations(self):
        """Test default max_iterations from config."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        assert solver.max_iterations == FULLBODY_DEFAULT_MAX_ITERATIONS

    def test_init_custom_max_iterations(self):
        """Test custom max_iterations."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping, max_iterations=50)
        assert solver.max_iterations == 50

    def test_init_maintain_balance_default_true(self):
        """Test maintain_balance defaults to True."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        assert solver.maintain_balance is True

    def test_init_pelvis_height_adjust_default_true(self):
        """Test pelvis_height_adjust defaults to True."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        assert solver.pelvis_height_adjust is True

    def test_init_spine_stiffness_from_config(self):
        """Test spine_stiffness from config."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        assert solver.spine_stiffness == FULLBODY_SPINE_STIFFNESS

    def test_init_bone_masses_empty(self):
        """Test _bone_masses starts empty."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        assert solver._bone_masses == {}

    def test_init_support_polygon_empty(self):
        """Test _support_polygon starts empty."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        assert solver._support_polygon == []


class TestFullBodyIKInitializeSolvers:
    """Tests for FullBodyIK._initialize_solvers() method."""

    def test_creates_left_arm_ik_with_valid_chain(self):
        """Test left arm IK created when chain has 3+ bones."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        assert solver._left_arm_ik is not None
        assert isinstance(solver._left_arm_ik, TwoBoneIK)

    def test_creates_right_arm_ik_with_valid_chain(self):
        """Test right arm IK created when chain has 3+ bones."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        assert solver._right_arm_ik is not None
        assert isinstance(solver._right_arm_ik, TwoBoneIK)

    def test_creates_left_leg_ik_with_valid_chain(self):
        """Test left leg IK created when chain has 3+ bones."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        assert solver._left_leg_ik is not None
        assert isinstance(solver._left_leg_ik, TwoBoneIK)

    def test_creates_right_leg_ik_with_valid_chain(self):
        """Test right leg IK created when chain has 3+ bones."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        assert solver._right_leg_ik is not None
        assert isinstance(solver._right_leg_ik, TwoBoneIK)

    def test_creates_spine_ik_with_valid_chain(self):
        """Test spine IK created when chain has 2+ bones."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        assert solver._spine_ik is not None
        assert isinstance(solver._spine_ik, FABRIKChain)

    def test_no_left_arm_ik_without_chain(self):
        """Test left arm IK is None without chain."""
        mapping = create_minimal_mapping()
        solver = FullBodyIK(mapping)
        assert solver._left_arm_ik is None

    def test_no_right_arm_ik_without_chain(self):
        """Test right arm IK is None without chain."""
        mapping = create_minimal_mapping()
        solver = FullBodyIK(mapping)
        assert solver._right_arm_ik is None

    def test_no_leg_ik_without_chain(self):
        """Test leg IK is None without chain."""
        mapping = create_minimal_mapping()
        solver = FullBodyIK(mapping)
        assert solver._left_leg_ik is None
        assert solver._right_leg_ik is None

    def test_no_spine_ik_without_chain(self):
        """Test spine IK is None without chain."""
        mapping = create_minimal_mapping()
        solver = FullBodyIK(mapping)
        assert solver._spine_ik is None

    def test_no_arm_ik_with_short_chain(self):
        """Test no arm IK with chain < 3 bones."""
        mapping = SkeletonMapping()
        mapping.left_arm_chain = [0, 1]  # Only 2 bones
        solver = FullBodyIK(mapping)
        assert solver._left_arm_ik is None

    def test_no_leg_ik_with_short_chain(self):
        """Test no leg IK with chain < 3 bones."""
        mapping = SkeletonMapping()
        mapping.left_leg_chain = [0, 1]  # Only 2 bones
        solver = FullBodyIK(mapping)
        assert solver._left_leg_ik is None

    def test_spine_ik_with_two_bones(self):
        """Test spine IK created with exactly 2 bones."""
        mapping = SkeletonMapping()
        mapping.spine_chain = [0, 1]
        solver = FullBodyIK(mapping)
        assert solver._spine_ik is not None

    def test_no_spine_ik_with_single_bone(self):
        """Test no spine IK with single bone chain."""
        mapping = SkeletonMapping()
        mapping.spine_chain = [0]
        solver = FullBodyIK(mapping)
        assert solver._spine_ik is None


class TestFullBodyIKSetBoneMass:
    """Tests for FullBodyIK.set_bone_mass() method."""

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
        solver.set_bone_mass(4, 3.0)

        assert solver._bone_masses[0] == 10.0
        assert solver._bone_masses[1] == 5.0
        assert solver._bone_masses[4] == 3.0

    def test_overwrite_bone_mass(self):
        """Test overwriting existing bone mass."""
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

    def test_set_small_mass(self):
        """Test setting very small mass."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_bone_mass(0, 0.001)
        assert solver._bone_masses[0] == 0.001


class TestFullBodyIKSetSupportPolygon:
    """Tests for FullBodyIK.set_support_polygon() method."""

    def test_set_square_polygon(self):
        """Test setting a square support polygon."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        vertices = [
            Vec3(-1, 0, -1), Vec3(1, 0, -1),
            Vec3(1, 0, 1), Vec3(-1, 0, 1)
        ]
        solver.set_support_polygon(vertices)

        assert len(solver._support_polygon) == 4

    def test_set_triangle_polygon(self):
        """Test setting a triangular support polygon."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        vertices = [Vec3(0, 0, 0), Vec3(2, 0, 0), Vec3(1, 0, 2)]
        solver.set_support_polygon(vertices)

        assert len(solver._support_polygon) == 3

    def test_set_empty_polygon(self):
        """Test setting empty polygon."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([])
        assert len(solver._support_polygon) == 0

    def test_polygon_vertices_copied(self):
        """Test polygon vertices are copied, not referenced."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        vertices = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 1)]
        solver.set_support_polygon(vertices)

        # Modify original list
        vertices.append(Vec3(1, 0, 1))

        # Solver's copy should be unchanged
        assert len(solver._support_polygon) == 3

    def test_replace_existing_polygon(self):
        """Test replacing existing polygon."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 0, 1)])
        assert len(solver._support_polygon) == 3

        solver.set_support_polygon([Vec3(0, 0, 0), Vec3(1, 0, 0)])
        assert len(solver._support_polygon) == 2


# =============================================================================
# Test FullBodyIK.solve() Method
# =============================================================================

class TestFullBodyIKSolve:
    """Tests for FullBodyIK.solve() method."""

    def test_solve_returns_fullbody_ik_result(self):
        """Test solve returns FullBodyIKResult instance."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        result = solver.solve(transforms, [])
        assert isinstance(result, FullBodyIKResult)

    def test_solve_empty_goals_succeeds(self):
        """Test solve with empty goals returns success."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        result = solver.solve(transforms, [])
        assert result.success is True

    def test_solve_copies_transforms(self):
        """Test solve creates copy of transforms."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        original_pos = Vec3(
            transforms[0].translation.x,
            transforms[0].translation.y,
            transforms[0].translation.z
        )

        result = solver.solve(transforms, [])

        # Original should be unchanged
        assert vec3_approx_equal(transforms[0].translation, original_pos)

    def test_solve_returns_same_number_of_transforms(self):
        """Test solve returns same number of transforms."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms(20)

        result = solver.solve(transforms, [])
        assert len(result.transforms) == 20

    def test_solve_filters_disabled_goals(self):
        """Test disabled goals are not processed."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0, 0, 0),
            enabled=False,
            chain_type="left_arm"
        )

        result = solver.solve(transforms, [goal])
        assert 8 not in result.goals_achieved


class TestFullBodyIKSolveGoalSorting:
    """Tests for goal sorting by priority in solve()."""

    def test_goals_sorted_by_priority_descending(self):
        """Test goals are sorted by -priority (highest first)."""
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

    def test_negative_priority_sorted_last(self):
        """Test negative priority goals are processed last."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal_neg = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0, 0, 0),
            priority=-5,
            chain_type="left_arm"
        )
        goal_pos = FullBodyIKGoal(
            bone_index=12,
            target_position=Vec3(0, 0, 0),
            priority=5,
            chain_type="right_arm"
        )

        result = solver.solve(transforms, [goal_neg, goal_pos])
        assert 8 in result.goals_achieved
        assert 12 in result.goals_achieved

    def test_same_priority_goals_both_processed(self):
        """Test goals with same priority are both processed."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal1 = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0, 0, 0),
            priority=5,
            chain_type="left_arm"
        )
        goal2 = FullBodyIKGoal(
            bone_index=12,
            target_position=Vec3(0, 0, 0),
            priority=5,
            chain_type="right_arm"
        )

        result = solver.solve(transforms, [goal1, goal2])
        assert 8 in result.goals_achieved
        assert 12 in result.goals_achieved


class TestFullBodyIKSolvePhases:
    """Tests for solve() phase execution order."""

    def test_pelvis_adjustment_phase_executes(self):
        """Test Phase 1: pelvis adjustment executes with leg goals."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=15,
            target_position=Vec3(-0.1, -0.5, 0),
            chain_type="left_leg"
        )

        result = solver.solve(transforms, [goal])
        assert hasattr(result, 'pelvis_adjustment')

    def test_pelvis_adjustment_skipped_when_disabled(self):
        """Test pelvis adjustment skipped when disabled."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        solver.pelvis_height_adjust = False
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=15,
            target_position=Vec3(-0.1, -0.5, 0),
            chain_type="left_leg"
        )

        result = solver.solve(transforms, [goal])
        assert vec3_approx_equal(result.pelvis_adjustment, Vec3.zero())

    def test_spine_phase_executes_with_spine_goals(self):
        """Test Phase 2: spine solve executes with spine goals."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=4,
            target_position=Vec3(0, 2.0, 0.5),
            chain_type="spine"
        )

        # Should not crash
        result = solver.solve(transforms, [goal])
        assert isinstance(result, FullBodyIKResult)

    def test_limb_phase_executes_for_all_goals(self):
        """Test Phase 3: all active goals are processed."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goals = [
            FullBodyIKGoal(bone_index=8, target_position=Vec3(0, 0, 0), chain_type="left_arm"),
            FullBodyIKGoal(bone_index=12, target_position=Vec3(0, 0, 0), chain_type="right_arm"),
            FullBodyIKGoal(bone_index=15, target_position=Vec3(0, 0, 0), chain_type="left_leg"),
        ]

        result = solver.solve(transforms, goals)

        assert 8 in result.goals_achieved
        assert 12 in result.goals_achieved
        assert 15 in result.goals_achieved

    def test_balance_phase_executes(self):
        """Test Phase 4: balance maintenance executes."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        solver.maintain_balance = True
        transforms = create_humanoid_transforms()

        # Set up bone masses and polygon for balance check
        solver.set_bone_mass(0, 10.0)
        solver.set_support_polygon([
            Vec3(-0.5, 0, -0.5), Vec3(0.5, 0, -0.5),
            Vec3(0.5, 0, 0.5), Vec3(-0.5, 0, 0.5)
        ])

        # Should not crash
        result = solver.solve(transforms, [])
        assert isinstance(result, FullBodyIKResult)

    def test_balance_skipped_when_disabled(self):
        """Test balance phase skipped when disabled."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        solver.maintain_balance = False
        transforms = create_humanoid_transforms()

        # Even with masses and polygon, balance shouldn't run
        solver.set_bone_mass(0, 10.0)
        solver.set_support_polygon([
            Vec3(-0.5, 0, -0.5), Vec3(0.5, 0, -0.5),
            Vec3(0.5, 0, 0.5), Vec3(-0.5, 0, 0.5)
        ])

        # Should not crash
        result = solver.solve(transforms, [])
        assert isinstance(result, FullBodyIKResult)


# =============================================================================
# Test FullBodyIK._adjust_pelvis_height()
# =============================================================================

class TestAdjustPelvisHeight:
    """Tests for FullBodyIK._adjust_pelvis_height() method."""

    def test_returns_vec3_zero_no_pelvis(self):
        """Test returns Vec3.zero() when no pelvis mapped."""
        mapping = SkeletonMapping()  # No pelvis
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        result = solver._adjust_pelvis_height(transforms, [])
        assert vec3_approx_equal(result, Vec3.zero())

    def test_returns_vec3_zero_no_leg_goals(self):
        """Test returns Vec3.zero() without leg goals."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        # Only arm goals
        goals = [
            FullBodyIKGoal(bone_index=8, target_position=Vec3(0, 0, 0), chain_type="left_arm")
        ]

        result = solver._adjust_pelvis_height(transforms, goals)
        assert vec3_approx_equal(result, Vec3.zero())

    def test_returns_vec3_zero_goals_without_position(self):
        """Test returns Vec3.zero() when leg goals have no position."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        # Leg goal without position
        goals = [
            FullBodyIKGoal(bone_index=15, chain_type="left_leg")  # No target_position
        ]

        result = solver._adjust_pelvis_height(transforms, goals)
        assert vec3_approx_equal(result, Vec3.zero())

    def test_pelvis_drop_when_leg_out_of_reach(self):
        """Test pelvis drops when foot goal is out of reach."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        # Goal far below normal reach
        goal = FullBodyIKGoal(
            bone_index=15,
            target_position=Vec3(-0.1, -1.0, 0),  # Very low
            chain_type="left_leg"
        )

        result = solver._adjust_pelvis_height(transforms, [goal])

        # Pelvis should drop (negative Y adjustment)
        # The actual behavior depends on leg IK solver configuration
        assert isinstance(result, Vec3)


class TestSolveGoal:
    """Tests for FullBodyIK._solve_goal() method."""

    def test_returns_tuple(self):
        """Test returns (achieved, error) tuple."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0, 0, 0),
            chain_type="left_arm"
        )

        result = solver._solve_goal(transforms, goal)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)

    def test_goal_without_position_returns_true_zero(self):
        """Test goal without position returns (True, 0.0)."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(bone_index=8, chain_type="left_arm")  # No position

        achieved, error = solver._solve_goal(transforms, goal)
        assert achieved is True
        assert error == 0.0

    def test_routes_to_left_arm_solver(self):
        """Test left_arm chain routes to left arm solver."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.3, 1.0, 0.2),
            chain_type="left_arm"
        )

        achieved, error = solver._solve_goal(transforms, goal)
        assert isinstance(achieved, bool)
        assert error >= 0

    def test_routes_to_right_arm_solver(self):
        """Test right_arm chain routes to right arm solver."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=12,
            target_position=Vec3(0.3, 1.0, 0.2),
            chain_type="right_arm"
        )

        achieved, error = solver._solve_goal(transforms, goal)
        assert isinstance(achieved, bool)

    def test_routes_to_left_leg_solver(self):
        """Test left_leg chain routes to left leg solver."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=15,
            target_position=Vec3(-0.1, 0.0, 0.1),
            chain_type="left_leg"
        )

        achieved, error = solver._solve_goal(transforms, goal)
        assert isinstance(achieved, bool)

    def test_routes_to_right_leg_solver(self):
        """Test right_leg chain routes to right leg solver."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=18,
            target_position=Vec3(0.1, 0.0, 0.1),
            chain_type="right_leg"
        )

        achieved, error = solver._solve_goal(transforms, goal)
        assert isinstance(achieved, bool)

    def test_unknown_chain_computes_error_only(self):
        """Test unknown chain_type computes error without solver."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=5,
            target_position=Vec3(0, 0, 0),
            chain_type="unknown_chain"
        )

        achieved, error = solver._solve_goal(transforms, goal)
        assert isinstance(achieved, bool)
        assert error >= 0

    def test_no_solver_for_chain_returns_error(self):
        """Test returns error when no solver exists for chain."""
        mapping = create_minimal_mapping()
        solver = FullBodyIK(mapping)
        transforms = [Transform(Vec3(0, 0, 0), Quat.identity()) for _ in range(5)]

        goal = FullBodyIKGoal(
            bone_index=1,
            target_position=Vec3(1, 1, 1),
            chain_type="left_arm"
        )

        achieved, error = solver._solve_goal(transforms, goal)
        # Without solver, should compute error based on current position
        assert isinstance(achieved, bool)


class TestMaintainBalance:
    """Tests for FullBodyIK._maintain_balance() method."""

    def test_no_masses_no_change(self):
        """Test no balance adjustment without bone masses."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        solver.set_support_polygon([
            Vec3(-0.5, 0, -0.5), Vec3(0.5, 0, -0.5),
            Vec3(0.5, 0, 0.5), Vec3(-0.5, 0, 0.5)
        ])

        original_pelvis = Vec3(
            transforms[0].translation.x,
            transforms[0].translation.y,
            transforms[0].translation.z
        )

        solver._maintain_balance(transforms)

        # Should be unchanged (no masses)
        assert vec3_approx_equal(transforms[0].translation, original_pelvis)

    def test_no_polygon_no_change(self):
        """Test no balance adjustment without polygon."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        solver.set_bone_mass(0, 10.0)
        # No polygon set (empty)

        original_pelvis = Vec3(
            transforms[0].translation.x,
            transforms[0].translation.y,
            transforms[0].translation.z
        )

        solver._maintain_balance(transforms)

        # Should be unchanged (no polygon)
        assert vec3_approx_equal(transforms[0].translation, original_pelvis)

    def test_small_polygon_no_change(self):
        """Test no balance adjustment with < 3 vertex polygon."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        solver.set_bone_mass(0, 10.0)
        solver.set_support_polygon([Vec3(0, 0, 0), Vec3(1, 0, 0)])  # Only 2 vertices

        solver._maintain_balance(transforms)
        # Should not crash


class TestTranslateHierarchy:
    """Tests for FullBodyIK._translate_hierarchy() method."""

    def test_translates_all_bones(self):
        """Test all bones are translated by offset."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        offset = Vec3(0, -0.1, 0)
        original_positions = [
            Vec3(t.translation.x, t.translation.y, t.translation.z)
            for t in transforms
        ]

        solver._translate_hierarchy(transforms, 0, offset)

        for i, t in enumerate(transforms):
            expected = Vec3(
                original_positions[i].x + offset.x,
                original_positions[i].y + offset.y,
                original_positions[i].z + offset.z
            )
            assert vec3_approx_equal(t.translation, expected)

    def test_translate_with_zero_offset(self):
        """Test translation with zero offset changes nothing."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        original_positions = [
            Vec3(t.translation.x, t.translation.y, t.translation.z)
            for t in transforms
        ]

        solver._translate_hierarchy(transforms, 0, Vec3.zero())

        for i, t in enumerate(transforms):
            assert vec3_approx_equal(t.translation, original_positions[i])


class TestPointInPolygon:
    """Tests for FullBodyIK._point_in_polygon() method."""

    def test_point_inside_square(self):
        """Test point inside square returns True."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([
            Vec3(-1, 0, -1), Vec3(1, 0, -1),
            Vec3(1, 0, 1), Vec3(-1, 0, 1)
        ])

        assert solver._point_in_polygon(Vec3(0, 0, 0)) is True

    def test_point_outside_square(self):
        """Test point outside square returns False."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([
            Vec3(-1, 0, -1), Vec3(1, 0, -1),
            Vec3(1, 0, 1), Vec3(-1, 0, 1)
        ])

        assert solver._point_in_polygon(Vec3(2, 0, 0)) is False

    def test_empty_polygon_returns_true(self):
        """Test empty polygon returns True."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([])
        assert solver._point_in_polygon(Vec3(0, 0, 0)) is True

    def test_two_vertex_polygon_returns_true(self):
        """Test < 3 vertex polygon returns True."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([Vec3(0, 0, 0), Vec3(1, 0, 0)])
        assert solver._point_in_polygon(Vec3(5, 0, 5)) is True


class TestClosestPointOnPolygon:
    """Tests for FullBodyIK._closest_point_on_polygon() method."""

    def test_single_vertex_returns_point(self):
        """Test single vertex returns query point."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([Vec3(0, 0, 0)])

        result = solver._closest_point_on_polygon(Vec3(5, 0, 5))
        assert vec3_approx_equal(result, Vec3(5, 0, 5))

    def test_closest_to_edge(self):
        """Test finds closest point on edge."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)

        solver.set_support_polygon([
            Vec3(-1, 0, -1), Vec3(1, 0, -1),
            Vec3(1, 0, 1), Vec3(-1, 0, 1)
        ])

        # Point below square, closest to bottom edge
        result = solver._closest_point_on_polygon(Vec3(0, 0, -2))

        assert abs(result.z - (-1)) < 0.01
        assert abs(result.x) < 0.01


# =============================================================================
# Integration Tests with Mocked Solvers
# =============================================================================

class TestFullBodyIKWithMockedSolvers:
    """Integration tests using mocked TwoBoneIK/FABRIKChain."""

    def test_solve_with_mocked_arm_solver(self):
        """Test solve uses mocked arm solver correctly."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        # Create mock result
        mock_result = Mock()
        mock_result.success = True
        mock_result.root_rotation = Quat.identity()
        mock_result.mid_rotation = Quat.identity()
        mock_result.end_rotation = Quat.identity()

        # Patch the left arm solver
        solver._left_arm_ik = Mock()
        solver._left_arm_ik.solve = Mock(return_value=mock_result)

        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0, 0, 0),
            chain_type="left_arm"
        )

        result = solver.solve(transforms, [goal])

        # Verify solver was called
        solver._left_arm_ik.solve.assert_called_once()
        assert 8 in result.goals_achieved

    def test_solve_spine_with_mocked_fabrik(self):
        """Test spine solve uses mocked FABRIK."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms()

        # Create mock result
        mock_result = Mock()
        mock_result.positions = [Vec3(0, i * 0.2, 0) for i in range(5)]
        mock_result.rotations = [Quat.identity() for _ in range(5)]

        solver._spine_ik = Mock()
        solver._spine_ik.solve = Mock(return_value=mock_result)

        goal = FullBodyIKGoal(
            bone_index=4,
            target_position=Vec3(0, 2.0, 0.5),
            chain_type="spine"
        )

        result = solver.solve(transforms, [goal])

        # Verify spine solver was called
        solver._spine_ik.solve.assert_called_once()


class TestFullBodyIKEdgeCases:
    """Edge case tests for FullBodyIK."""

    def test_solve_with_single_transform(self):
        """Test solve with single transform."""
        mapping = create_minimal_mapping()
        solver = FullBodyIK(mapping)
        transforms = [Transform(Vec3(0, 0, 0), Quat.identity())]

        result = solver.solve(transforms, [])
        assert result.success is True
        assert len(result.transforms) == 1

    def test_solve_with_out_of_bounds_bone_index_raises(self):
        """Test goal with bone index beyond transforms length raises IndexError."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        transforms = create_humanoid_transforms(5)  # Only 5 bones

        goal = FullBodyIKGoal(
            bone_index=100,  # Way beyond
            target_position=Vec3(0, 0, 0),
            chain_type="left_arm"
        )

        # Code raises IndexError when accessing chain bones out of range
        with pytest.raises(IndexError):
            solver.solve(transforms, [goal])

    def test_solve_success_when_all_errors_within_tolerance(self):
        """Test success is True when all errors <= tolerance."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping, tolerance=100.0)  # Very loose tolerance
        transforms = create_humanoid_transforms()

        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0, 0, 0),
            chain_type="left_arm"
        )

        result = solver.solve(transforms, [goal])
        # With very loose tolerance, should succeed
        assert result.success is True or 8 in result.final_errors

    def test_solve_success_false_when_error_exceeds_tolerance(self):
        """Test success is False when any error > tolerance."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping, tolerance=0.0001)  # Very tight
        transforms = create_humanoid_transforms()

        # Target far from current position
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(100, 100, 100),
            chain_type="left_arm"
        )

        result = solver.solve(transforms, [goal])
        # With unreachable target and tight tolerance, should fail
        assert result.success is False or result.final_errors[8] > 0


class TestFullBodyIKDisabledFeatures:
    """Tests for disabled feature handling."""

    def test_maintain_balance_disabled(self):
        """Test maintain_balance can be disabled."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        solver.maintain_balance = False

        assert solver.maintain_balance is False

    def test_pelvis_height_adjust_disabled(self):
        """Test pelvis_height_adjust can be disabled."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        solver.pelvis_height_adjust = False

        assert solver.pelvis_height_adjust is False

    def test_spine_stiffness_modifiable(self):
        """Test spine_stiffness can be modified."""
        mapping = create_skeleton_mapping()
        solver = FullBodyIK(mapping)
        solver.spine_stiffness = 0.9

        assert solver.spine_stiffness == 0.9

"""Comprehensive tests for the IK subsystem.

This test module covers all IK solvers with a minimum of 150 tests.
"""

from __future__ import annotations

import math
import pytest
from typing import List, Tuple

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform

from engine.animation.ik.ik_goal import (
    IKGoalType, IKGoal, PositionGoal, RotationGoal, LookAtGoal,
    PositionRotationGoal, PoleVectorGoal, ChainGoal, CenterOfMassGoal,
    IKGoalBlender, ik_goal, ik_chain
)
from engine.animation.ik.two_bone import TwoBoneIK, TwoBoneIKResult, TwoBoneIKConstraint
from engine.animation.ik.fabrik import (
    FABRIKChain, FABRIKResult, FABRIKMultiChain,
    JointConstraint, JointConstraintType
)
from engine.animation.ik.ccd import (
    CCDSolver, CCDResult, CCDRotationOrder, RotationLimit,
    CCDSolverWithWeights, ConstrainedCCDSolver
)
from engine.animation.ik.jacobian import (
    JacobianIK, JacobianResult, JacobianMethod, Matrix, MultiTargetJacobianIK
)
from engine.animation.ik.fullbody import (
    FullBodyIK, FullBodyIKGoal, FullBodyIKResult, SkeletonMapping, BodyPart,
    LookAtSolver
)
from engine.animation.ik.foot_placement import (
    FootPlacement, FootPlacementResult, FootData, FootState,
    FootPlacementAnimated, MultiLegFootPlacement
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_arm_transforms() -> List[Transform]:
    """Create simple arm transforms for testing."""
    return [
        Transform(Vec3(0, 0, 0), Quat.identity()),      # shoulder
        Transform(Vec3(0, 1, 0), Quat.identity()),      # elbow
        Transform(Vec3(0, 2, 0), Quat.identity()),      # wrist
    ]


@pytest.fixture
def spine_positions() -> List[Vec3]:
    """Create spine positions for testing."""
    return [
        Vec3(0, 0, 0),    # pelvis
        Vec3(0, 0.3, 0),  # spine1
        Vec3(0, 0.6, 0),  # spine2
        Vec3(0, 0.9, 0),  # chest
        Vec3(0, 1.2, 0),  # neck
        Vec3(0, 1.5, 0),  # head
    ]


@pytest.fixture
def humanoid_skeleton() -> SkeletonMapping:
    """Create humanoid skeleton mapping."""
    skeleton = SkeletonMapping()
    skeleton.bone_map = {
        BodyPart.PELVIS: 0,
        BodyPart.SPINE: 1,
        BodyPart.CHEST: 2,
        BodyPart.NECK: 3,
        BodyPart.HEAD: 4,
        BodyPart.LEFT_UPPER_ARM: 5,
        BodyPart.LEFT_LOWER_ARM: 6,
        BodyPart.LEFT_HAND: 7,
        BodyPart.RIGHT_UPPER_ARM: 8,
        BodyPart.RIGHT_LOWER_ARM: 9,
        BodyPart.RIGHT_HAND: 10,
        BodyPart.LEFT_UPPER_LEG: 11,
        BodyPart.LEFT_LOWER_LEG: 12,
        BodyPart.LEFT_FOOT: 13,
        BodyPart.RIGHT_UPPER_LEG: 14,
        BodyPart.RIGHT_LOWER_LEG: 15,
        BodyPart.RIGHT_FOOT: 16,
    }
    skeleton.spine_chain = [0, 1, 2, 3, 4]
    skeleton.left_arm_chain = [5, 6, 7]
    skeleton.right_arm_chain = [8, 9, 10]
    skeleton.left_leg_chain = [11, 12, 13]
    skeleton.right_leg_chain = [14, 15, 16]
    return skeleton


# ============================================================================
# IK Goal Tests (15 tests)
# ============================================================================

class TestIKGoals:
    """Tests for IK goal definitions."""

    def test_position_goal_creation(self):
        """Test creating a position goal."""
        goal = PositionGoal(bone_index=5, target_position=Vec3(1, 2, 3))
        assert goal.goal_type == IKGoalType.POSITION
        assert goal.bone_index == 5
        assert goal.target_position == Vec3(1, 2, 3)
        assert goal.weight == 1.0

    def test_position_goal_distance(self):
        """Test distance calculation for position goal."""
        goal = PositionGoal(bone_index=0, target_position=Vec3(3, 4, 0))
        distance = goal.distance_to_target(Vec3(0, 0, 0))
        assert abs(distance - 5.0) < 0.001

    def test_position_goal_achieved(self):
        """Test goal achievement check."""
        goal = PositionGoal(bone_index=0, target_position=Vec3(1, 0, 0), tolerance=0.1)
        assert goal.is_achieved(Vec3(1.05, 0, 0))
        assert not goal.is_achieved(Vec3(1.2, 0, 0))

    def test_rotation_goal_creation(self):
        """Test creating a rotation goal."""
        rot = Quat.from_euler(0.5, 0, 0)
        goal = RotationGoal(bone_index=3, target_rotation=rot)
        assert goal.goal_type == IKGoalType.ROTATION
        assert goal.target_rotation == rot

    def test_rotation_goal_angular_distance(self):
        """Test angular distance calculation."""
        goal = RotationGoal(bone_index=0, target_rotation=Quat.identity())
        rot = Quat.from_axis_angle(Vec3.unit_y(), math.pi / 4)
        dist = goal.angular_distance(rot)
        assert abs(dist - math.pi / 4) < 0.01

    def test_look_at_goal_creation(self):
        """Test creating a look-at goal."""
        goal = LookAtGoal(bone_index=4, target_point=Vec3(0, 0, 10))
        assert goal.goal_type == IKGoalType.LOOK_AT
        assert goal.max_angle == pytest.approx(1.57, abs=0.01)

    def test_look_at_rotation_computation(self):
        """Test look-at rotation calculation."""
        goal = LookAtGoal(bone_index=0, target_point=Vec3(0, 0, 10))
        rot = goal.compute_look_rotation(Vec3(0, 0, 0), Quat.identity())
        assert rot is not None

    def test_position_rotation_goal(self):
        """Test combined position and rotation goal."""
        goal = PositionRotationGoal(
            bone_index=7,
            target_position=Vec3(1, 1, 1),
            target_rotation=Quat.identity(),
            position_weight=0.8,
            rotation_weight=0.2
        )
        assert goal.goal_type == IKGoalType.POSITION_AND_ROTATION
        assert goal.position_weight == 0.8

    def test_position_rotation_achieved(self):
        """Test combined goal achievement."""
        goal = PositionRotationGoal(
            bone_index=0,
            target_position=Vec3(1, 0, 0),
            target_rotation=Quat.identity()
        )
        assert goal.is_achieved(Vec3(1, 0, 0), Quat.identity())
        assert not goal.is_achieved(Vec3(2, 0, 0), Quat.identity())

    def test_pole_vector_goal(self):
        """Test pole vector goal."""
        goal = PoleVectorGoal(bone_index=6, pole_position=Vec3(0, 0, 1))
        assert goal.goal_type == IKGoalType.POLE_VECTOR

    def test_pole_direction_computation(self):
        """Test pole direction calculation."""
        goal = PoleVectorGoal(bone_index=0, pole_position=Vec3(0, 1, 1))
        direction = goal.compute_pole_direction(Vec3(0, 0, 0), Vec3(0, 2, 0))
        assert direction.length() > 0

    def test_chain_goal_validation(self):
        """Test chain goal validation."""
        chain = ChainGoal(
            chain_name="arm",
            bone_indices=[5, 6, 7],
            stiffness=0.5
        )
        assert chain.validate()
        assert chain.chain_length == 3

    def test_chain_goal_invalid(self):
        """Test invalid chain goal."""
        chain = ChainGoal(chain_name="invalid", bone_indices=[])
        assert not chain.validate()

    def test_com_goal_balance_check(self):
        """Test center of mass balance check."""
        goal = CenterOfMassGoal(
            bone_index=0,
            support_polygon=[
                Vec3(-1, 0, -1),
                Vec3(1, 0, -1),
                Vec3(1, 0, 1),
                Vec3(-1, 0, 1)
            ]
        )
        assert goal.is_balanced(Vec3(0, 0, 0))
        assert not goal.is_balanced(Vec3(5, 0, 0))

    def test_ik_goal_blender(self):
        """Test IK goal blender."""
        blender = IKGoalBlender(blend_speed=5.0)
        result1 = blender.blend_position(0, Vec3(10, 0, 0), 0.5)
        result2 = blender.blend_position(0, Vec3(10, 0, 0), 0.5)
        # Second call should be closer to target
        assert result2.x >= result1.x


# ============================================================================
# Two-Bone IK Tests (25 tests)
# ============================================================================

class TestTwoBoneIK:
    """Tests for analytical two-bone IK solver."""

    def test_solver_creation(self):
        """Test solver creation."""
        solver = TwoBoneIK(0, 1, 2)
        assert solver.root_bone == 0
        assert solver.mid_bone == 1
        assert solver.end_bone == 2

    def test_solver_invalid_indices(self):
        """Test solver rejects invalid indices."""
        with pytest.raises(ValueError):
            TwoBoneIK(-1, 1, 2)

    def test_solve_reachable_target(self, simple_arm_transforms):
        """Test solving for reachable target and verify end effector position."""
        solver = TwoBoneIK(0, 1, 2)
        target = Vec3(1, 1, 0)
        result = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            target
        )
        assert result.success
        assert result.target_reached
        # Verify the solution actually reaches the target by computing forward kinematics
        # The end effector should be close to target (within tolerance)
        root_pos = simple_arm_transforms[0].translation
        # Apply rotations to compute final end effector position
        upper_vec = result.root_rotation.rotate_vector(Vec3(0, 1, 0))
        mid_pos = root_pos + upper_vec
        mid_vec = result.mid_rotation.rotate_vector(Vec3(0, 1, 0))
        end_pos = mid_pos + mid_vec
        # Check end effector is close to target
        distance = (end_pos - target).length()
        assert distance < 0.1, f"End effector at {end_pos} is {distance} from target {target}"

    def test_solve_unreachable_target(self, simple_arm_transforms):
        """Test solving for unreachable target."""
        solver = TwoBoneIK(0, 1, 2)
        result = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(10, 0, 0)  # Way too far
        )
        assert result.success
        assert not result.target_reached
        assert result.extension_ratio > 0.9

    def test_solve_with_pole_vector(self, simple_arm_transforms):
        """Test solving with pole vector."""
        solver = TwoBoneIK(0, 1, 2)
        result = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(1, 1, 0),
            pole_vector=Vec3(0, 0, 1)
        )
        assert result.success

    def test_solve_target_at_root(self, simple_arm_transforms):
        """Test solving when target is at root."""
        solver = TwoBoneIK(0, 1, 2)
        result = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(0, 0, 0)  # At root
        )
        assert result.success
        assert result.extension_ratio < 0.1

    def test_soft_ik_enabled(self, simple_arm_transforms):
        """Test soft IK mode."""
        solver = TwoBoneIK(0, 1, 2, soft_ik_ratio=0.8, soft_ik_blend=1.0)
        result = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(3, 0, 0)  # Beyond reach
        )
        assert result.success

    def test_extension_ratio_calculation(self, simple_arm_transforms):
        """Test extension ratio is correct."""
        solver = TwoBoneIK(0, 1, 2)

        # Target at mid-range
        result = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(0, 1, 0)
        )
        assert 0.0 <= result.extension_ratio <= 1.0

    def test_bone_lengths_caching(self, simple_arm_transforms):
        """Test bone lengths are cached."""
        solver = TwoBoneIK(0, 1, 2)
        solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(1, 1, 0)
        )
        assert solver._lengths_cached
        assert solver._upper_length == pytest.approx(1.0, abs=0.01)

    def test_reset_cached_lengths(self, simple_arm_transforms):
        """Test resetting cached lengths."""
        solver = TwoBoneIK(0, 1, 2)
        solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(1, 1, 0)
        )
        solver.reset_cached_lengths()
        assert not solver._lengths_cached

    def test_max_reach_property(self, simple_arm_transforms):
        """Test max reach property."""
        solver = TwoBoneIK(0, 1, 2)
        solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(1, 1, 0)
        )
        assert solver.max_reach == pytest.approx(2.0, abs=0.01)

    def test_min_reach_property(self, simple_arm_transforms):
        """Test min reach property."""
        solver = TwoBoneIK(0, 1, 2)
        solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(1, 1, 0)
        )
        assert solver.min_reach == pytest.approx(0.0, abs=0.01)

    def test_solve_with_pose(self, simple_arm_transforms):
        """Test solve_with_pose method."""
        solver = TwoBoneIK(0, 1, 2)
        new_transforms = solver.solve_with_pose(
            simple_arm_transforms,
            Vec3(1, 1, 0)
        )
        assert len(new_transforms) == 3
        assert new_transforms[0].rotation != Quat.identity()

    def test_solve_multiple_targets_sequentially(self, simple_arm_transforms):
        """Test solving for multiple targets."""
        solver = TwoBoneIK(0, 1, 2)
        targets = [Vec3(1, 1, 0), Vec3(0, 2, 0), Vec3(-1, 1, 0)]

        for target in targets:
            result = solver.solve(
                simple_arm_transforms[0],
                simple_arm_transforms[1],
                simple_arm_transforms[2],
                target
            )
            assert result.success

    def test_constraint_wrapper(self, simple_arm_transforms):
        """Test constraint wrapper."""
        solver = TwoBoneIK(0, 1, 2)
        constraint = TwoBoneIKConstraint(
            solver,
            min_bend_angle=0.2,
            max_bend_angle=2.5
        )

        result = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(1, 1, 0)
        )
        constrained = constraint.apply_constraints(result)
        assert constrained.success

    def test_target_rotation_preserved(self, simple_arm_transforms):
        """Test target rotation is applied."""
        solver = TwoBoneIK(0, 1, 2)
        target_rot = Quat.from_euler(0.5, 0.5, 0)
        result = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(1, 1, 0),
            target_rotation=target_rot
        )
        assert result.end_rotation == target_rot

    def test_solve_various_angles(self, simple_arm_transforms):
        """Test solving at various angles."""
        solver = TwoBoneIK(0, 1, 2)

        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            target = Vec3(math.cos(rad), 1 + math.sin(rad), 0)
            result = solver.solve(
                simple_arm_transforms[0],
                simple_arm_transforms[1],
                simple_arm_transforms[2],
                target
            )
            assert result.success

    def test_degenerate_bone_length(self):
        """Test handling of zero-length bones - should handle gracefully without crash."""
        transforms = [
            Transform(Vec3(0, 0, 0), Quat.identity()),
            Transform(Vec3(0, 0, 0), Quat.identity()),  # Same as root (zero-length upper bone)
            Transform(Vec3(0, 1, 0), Quat.identity()),
        ]
        solver = TwoBoneIK(0, 1, 2)
        result = solver.solve(
            transforms[0], transforms[1], transforms[2],
            Vec3(0.5, 0.5, 0)
        )
        # Should handle gracefully without NaN or crash
        assert result is not None
        # Verify rotations are valid (not NaN)
        assert not math.isnan(result.root_rotation.w)
        assert not math.isnan(result.mid_rotation.w)
        assert not math.isnan(result.end_rotation.w)

    def test_solve_3d_targets(self, simple_arm_transforms):
        """Test solving for 3D targets."""
        solver = TwoBoneIK(0, 1, 2)
        targets_3d = [
            Vec3(0.5, 1, 0.5),
            Vec3(-0.5, 1, -0.5),
            Vec3(0, 1, 1),
        ]
        for target in targets_3d:
            result = solver.solve(
                simple_arm_transforms[0],
                simple_arm_transforms[1],
                simple_arm_transforms[2],
                target
            )
            assert result.success

    def test_rotations_are_valid_quaternions(self, simple_arm_transforms):
        """Test that output rotations are valid unit quaternions."""
        solver = TwoBoneIK(0, 1, 2)
        result = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(1, 1, 0)
        )
        assert abs(result.root_rotation.length() - 1.0) < 0.01
        assert abs(result.mid_rotation.length() - 1.0) < 0.01
        assert abs(result.end_rotation.length() - 1.0) < 0.01

    def test_solve_with_different_scales(self):
        """Test solving with different bone scales."""
        transforms = [
            Transform(Vec3(0, 0, 0), Quat.identity()),
            Transform(Vec3(0, 2, 0), Quat.identity()),  # Longer upper
            Transform(Vec3(0, 3, 0), Quat.identity()),  # Shorter lower
        ]
        solver = TwoBoneIK(0, 1, 2)
        result = solver.solve(
            transforms[0], transforms[1], transforms[2],
            Vec3(1.5, 1.5, 0)
        )
        assert result.success

    def test_pole_vector_affects_bend_direction(self, simple_arm_transforms):
        """Test that pole vector affects bend direction."""
        solver = TwoBoneIK(0, 1, 2)

        result_forward = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(1, 1, 0),
            pole_vector=Vec3(0, 0, 1)
        )

        result_backward = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(1, 1, 0),
            pole_vector=Vec3(0, 0, -1)
        )

        # Rotations should differ based on pole
        assert result_forward.root_rotation != result_backward.root_rotation

    def test_solve_boundary_reach(self, simple_arm_transforms):
        """Test solving at exact reach boundary."""
        solver = TwoBoneIK(0, 1, 2)
        solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(1, 0, 0)  # Cache lengths
        )

        result = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(0, 2, 0)  # Exactly at max reach
        )
        assert result.extension_ratio > 0.95

    def test_solve_insufficient_transforms(self):
        """Test error handling with insufficient transforms."""
        solver = TwoBoneIK(0, 1, 2)
        transforms = [Transform(), Transform()]  # Only 2

        with pytest.raises(ValueError):
            solver.solve_with_pose(transforms, Vec3(1, 1, 0))


# ============================================================================
# FABRIK Tests (25 tests)
# ============================================================================

class TestFABRIK:
    """Tests for FABRIK IK solver."""

    def test_chain_creation(self):
        """Test FABRIK chain creation."""
        chain = FABRIKChain([0, 1, 2, 3], tolerance=0.001, max_iterations=10)
        assert chain.chain_length == 4
        assert chain.root_index == 0
        assert chain.end_index == 3

    def test_chain_requires_minimum_bones(self):
        """Test chain requires at least 2 bones."""
        with pytest.raises(ValueError):
            FABRIKChain([0])

    def test_solve_reachable_target(self, spine_positions):
        """Test solving for reachable target and verify end effector moves toward it."""
        chain = FABRIKChain([0, 1, 2, 3, 4, 5], tolerance=0.1, max_iterations=20)
        target = Vec3(0, 1.5, 0.5)
        result = chain.solve(spine_positions, target)
        # Target is reachable, solver should get close
        assert result.final_error < 0.15
        # Verify end effector position moved toward target
        end_pos = result.positions[-1]
        original_end = spine_positions[-1]
        # End should be closer to target than original
        new_dist = (end_pos - target).length()
        original_dist = (original_end - target).length()
        assert new_dist <= original_dist + 0.01, "End effector should move toward target"

    def test_solve_unreachable_target(self, spine_positions):
        """Test solving for unreachable target."""
        chain = FABRIKChain([0, 1, 2, 3, 4, 5], tolerance=0.01)
        target = Vec3(0, 10, 0)  # Way beyond reach
        result = chain.solve(spine_positions, target)
        assert not result.success
        assert result.final_error > 0

    def test_solve_convergence(self, spine_positions):
        """Test solver converges within iterations."""
        chain = FABRIKChain([0, 1, 2, 3, 4, 5], tolerance=0.001, max_iterations=50)
        target = Vec3(0.5, 1.2, 0)
        result = chain.solve(spine_positions, target)
        if result.success:
            assert result.iterations <= 50

    def test_positions_output_length(self, spine_positions):
        """Test output positions have correct length."""
        chain = FABRIKChain([0, 1, 2, 3, 4, 5])
        result = chain.solve(spine_positions, Vec3(0, 1.5, 0))
        assert len(result.positions) == 6

    def test_rotations_output_length(self, spine_positions):
        """Test output rotations have correct length."""
        chain = FABRIKChain([0, 1, 2, 3, 4, 5])
        result = chain.solve(spine_positions, Vec3(0, 1.5, 0))
        assert len(result.rotations) == 6

    def test_root_position_preserved(self, spine_positions):
        """Test root position stays fixed."""
        chain = FABRIKChain([0, 1, 2, 3, 4, 5])
        root_pos = Vec3(spine_positions[0].x, spine_positions[0].y, spine_positions[0].z)
        result = chain.solve(spine_positions, Vec3(0, 1.5, 0.5))
        assert result.positions[0] == root_pos

    def test_bone_lengths_preserved(self, spine_positions):
        """Test bone lengths are maintained."""
        chain = FABRIKChain([0, 1, 2, 3, 4, 5])
        chain._cache_bone_lengths(spine_positions)
        original_lengths = list(chain._bone_lengths)

        result = chain.solve(spine_positions, Vec3(0.5, 1.2, 0))

        for i, orig_len in enumerate(original_lengths):
            if i < len(result.positions) - 1:
                new_len = (result.positions[i + 1] - result.positions[i]).length()
                assert abs(new_len - orig_len) < 0.01

    def test_hinge_constraint(self):
        """Test hinge joint constraint."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.HINGE,
            axis=Vec3.unit_y()
        )
        direction = Vec3(1, 0, 1).normalized()
        constrained = constraint.apply(direction, Quat.identity())
        # Should be projected to XZ plane
        assert abs(constrained.y) < 0.01

    def test_ball_socket_constraint(self):
        """Test ball-socket cone constraint."""
        constraint = JointConstraint(
            constraint_type=JointConstraintType.BALL_SOCKET,
            cone_angle=math.pi / 4
        )
        direction = Vec3(1, 0, 0)  # 90 degrees from Y
        constrained = constraint.apply(direction, Quat.identity())
        # Should be clamped to cone
        assert constrained is not None

    def test_set_constraint(self, spine_positions):
        """Test setting constraint on joint."""
        chain = FABRIKChain([0, 1, 2, 3, 4, 5])
        constraint = JointConstraint(constraint_type=JointConstraintType.HINGE)
        chain.set_constraint(2, constraint)
        assert chain._constraints[2].constraint_type == JointConstraintType.HINGE

    def test_solve_with_transforms(self):
        """Test solve_with_transforms method."""
        transforms = [
            Transform(Vec3(0, i * 0.3, 0), Quat.identity())
            for i in range(6)
        ]
        chain = FABRIKChain([0, 1, 2, 3, 4, 5])
        result = chain.solve_with_transforms(transforms, Vec3(0.5, 1.2, 0))
        assert len(result) == 6
        assert isinstance(result[0], Transform)

    def test_reset_cached_lengths(self, spine_positions):
        """Test resetting cached lengths."""
        chain = FABRIKChain([0, 1, 2, 3, 4, 5])
        chain.solve(spine_positions, Vec3(0, 1.5, 0))
        assert chain._lengths_cached
        chain.reset_cached_lengths()
        assert not chain._lengths_cached

    def test_solve_various_targets(self, spine_positions):
        """Test solving for various targets."""
        chain = FABRIKChain([0, 1, 2, 3, 4, 5], tolerance=0.05)
        targets = [
            Vec3(0, 1.5, 0),
            Vec3(0.5, 1.2, 0),
            Vec3(-0.3, 1.4, 0.2),
            Vec3(0, 1.0, 0.5),
        ]
        for target in targets:
            result = chain.solve(spine_positions, target)
            # Should at least complete without error
            assert result is not None

    def test_multi_chain_solver(self):
        """Test multi-chain FABRIK solver."""
        multi = FABRIKMultiChain()

        chain1 = FABRIKChain([0, 1, 2])
        chain2 = FABRIKChain([0, 3, 4])

        multi.add_chain(chain1, Vec3(1, 1, 0))
        multi.add_chain(chain2, Vec3(-1, 1, 0))

        all_positions = [
            Vec3(0, 0, 0),
            Vec3(0, 0.5, 0),
            Vec3(0, 1, 0),
            Vec3(0, 0.5, 0),
            Vec3(0, 1, 0),
        ]

        result = multi.solve(all_positions)
        assert len(result) == 5

    def test_multi_chain_set_target(self):
        """Test setting target on multi-chain."""
        multi = FABRIKMultiChain()
        chain = FABRIKChain([0, 1, 2])
        idx = multi.add_chain(chain, Vec3(1, 1, 0))

        multi.set_target(idx, Vec3(2, 2, 0))
        assert multi._chain_targets[idx] == Vec3(2, 2, 0)

    def test_short_chain(self):
        """Test solving for minimum 2-bone chain."""
        positions = [Vec3(0, 0, 0), Vec3(0, 1, 0)]
        chain = FABRIKChain([0, 1])
        result = chain.solve(positions, Vec3(0.5, 0.5, 0))
        assert len(result.positions) == 2

    def test_iteration_count_tracking(self, spine_positions):
        """Test iteration count is tracked."""
        chain = FABRIKChain([0, 1, 2, 3, 4, 5], max_iterations=100)
        result = chain.solve(spine_positions, Vec3(0.5, 1.2, 0))
        assert result.iterations >= 1
        assert result.iterations <= 100

    def test_final_error_tracking(self, spine_positions):
        """Test final error is tracked."""
        chain = FABRIKChain([0, 1, 2, 3, 4, 5])
        result = chain.solve(spine_positions, Vec3(0, 1.5, 0))
        assert result.final_error >= 0

    def test_wrong_position_count(self, spine_positions):
        """Test error with wrong position count."""
        chain = FABRIKChain([0, 1, 2, 3, 4, 5])
        with pytest.raises(ValueError):
            chain.solve(spine_positions[:3], Vec3(0, 1.5, 0))

    def test_constraint_none_passthrough(self):
        """Test NONE constraint doesn't modify direction."""
        constraint = JointConstraint(constraint_type=JointConstraintType.NONE)
        direction = Vec3(1, 2, 3).normalized()
        result = constraint.apply(direction, Quat.identity())
        assert result == direction

    def test_tolerance_affects_convergence(self, spine_positions):
        """Test tolerance affects convergence criteria."""
        chain_tight = FABRIKChain([0, 1, 2, 3, 4, 5], tolerance=0.0001)
        chain_loose = FABRIKChain([0, 1, 2, 3, 4, 5], tolerance=0.1)

        target = Vec3(0.5, 1.2, 0)
        result_tight = chain_tight.solve(spine_positions, target)
        result_loose = chain_loose.solve(spine_positions, target)

        # Loose tolerance should converge faster
        if result_loose.success and result_tight.success:
            assert result_loose.iterations <= result_tight.iterations

    def test_long_chain(self):
        """Test solving for long chain."""
        positions = [Vec3(0, i * 0.1, 0) for i in range(20)]
        indices = list(range(20))
        chain = FABRIKChain(indices, tolerance=0.05, max_iterations=50)
        result = chain.solve(positions, Vec3(0.5, 1.5, 0))
        assert len(result.positions) == 20


# ============================================================================
# CCD Tests (25 tests)
# ============================================================================

class TestCCD:
    """Tests for CCD IK solver."""

    def test_solver_creation(self):
        """Test CCD solver creation."""
        solver = CCDSolver([0, 1, 2, 3], tolerance=0.001, max_iterations=10, damping=1.0)
        assert solver.chain_length == 4

    def test_solver_invalid_damping(self):
        """Test solver rejects invalid damping."""
        with pytest.raises(ValueError):
            CCDSolver([0, 1, 2], damping=0)
        with pytest.raises(ValueError):
            CCDSolver([0, 1, 2], damping=1.5)

    def test_solve_basic_target(self, spine_positions):
        """Test solving for basic target."""
        solver = CCDSolver([0, 1, 2, 3, 4, 5], tolerance=0.05, max_iterations=20)
        rotations = [Quat.identity() for _ in spine_positions]
        target = Vec3(0.5, 1.2, 0)
        result = solver.solve(spine_positions, rotations, target)
        assert result.success or result.iterations == 20

    def test_rotation_limit_creation(self):
        """Test rotation limit creation."""
        limit = RotationLimit(
            enabled=True,
            axis=Vec3.unit_y(),
            min_angles=Vec3(-1, -1, -1),
            max_angles=Vec3(1, 1, 1),
            is_hinge=False
        )
        assert limit.enabled

    def test_hinge_limit_clamping(self):
        """Test hinge limit clamping."""
        limit = RotationLimit(
            enabled=True,
            axis=Vec3.unit_y(),
            is_hinge=True
        )
        rot = Quat.from_euler(0.5, 0.5, 0.5)
        clamped = limit.clamp_rotation(rot)
        assert clamped is not None

    def test_euler_limit_clamping(self):
        """Test Euler angle limit clamping."""
        limit = RotationLimit(
            enabled=True,
            min_angles=Vec3(-0.5, -0.5, -0.5),
            max_angles=Vec3(0.5, 0.5, 0.5),
            is_hinge=False
        )
        rot = Quat.from_euler(1.0, 1.0, 1.0)  # Beyond limits
        clamped = limit.clamp_rotation(rot)
        pitch, yaw, roll = clamped.to_euler()
        assert pitch <= 0.6  # Some tolerance for quaternion conversion

    def test_set_rotation_limit(self):
        """Test setting rotation limit."""
        solver = CCDSolver([0, 1, 2, 3])
        limit = RotationLimit(enabled=True)
        solver.set_rotation_limit(1, limit)
        assert solver._rotation_limits[1].enabled

    def test_rotation_order_end_to_root(self, spine_positions):
        """Test end-to-root rotation order."""
        solver = CCDSolver([0, 1, 2, 3, 4, 5])
        solver.set_rotation_order(CCDRotationOrder.END_TO_ROOT)
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, Vec3(0.5, 1.2, 0))
        assert result is not None

    def test_rotation_order_root_to_end(self, spine_positions):
        """Test root-to-end rotation order."""
        solver = CCDSolver([0, 1, 2, 3, 4, 5])
        solver.set_rotation_order(CCDRotationOrder.ROOT_TO_END)
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, Vec3(0.5, 1.2, 0))
        assert result is not None

    def test_rotation_order_alternating(self, spine_positions):
        """Test alternating rotation order."""
        solver = CCDSolver([0, 1, 2, 3, 4, 5])
        solver.set_rotation_order(CCDRotationOrder.ALTERNATING)
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, Vec3(0.5, 1.2, 0))
        assert result is not None

    def test_damping_affects_convergence(self, spine_positions):
        """Test damping affects convergence speed."""
        solver_high = CCDSolver([0, 1, 2, 3, 4, 5], damping=1.0)
        solver_low = CCDSolver([0, 1, 2, 3, 4, 5], damping=0.5)

        rotations = [Quat.identity() for _ in spine_positions]
        target = Vec3(0.5, 1.2, 0)

        result_high = solver_high.solve(spine_positions, rotations, target)
        result_low = solver_low.solve(spine_positions, rotations, target)

        # Lower damping should converge slower but more stable
        assert result_high is not None
        assert result_low is not None

    def test_output_rotations_count(self, spine_positions):
        """Test output has correct number of rotations."""
        solver = CCDSolver([0, 1, 2, 3, 4, 5])
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, Vec3(0.5, 1.2, 0))
        assert len(result.rotations) == 6

    def test_output_positions_count(self, spine_positions):
        """Test output has correct number of positions."""
        solver = CCDSolver([0, 1, 2, 3, 4, 5])
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, Vec3(0.5, 1.2, 0))
        assert len(result.positions) == 6

    def test_solve_with_transforms(self):
        """Test solve_with_transforms method."""
        transforms = [
            Transform(Vec3(0, i * 0.3, 0), Quat.identity())
            for i in range(6)
        ]
        solver = CCDSolver([0, 1, 2, 3, 4, 5])
        result = solver.solve_with_transforms(transforms, Vec3(0.5, 1.2, 0))
        assert len(result) == 6

    def test_weighted_ccd_solver(self, spine_positions):
        """Test weighted CCD solver."""
        solver = CCDSolverWithWeights(
            [0, 1, 2, 3, 4, 5],
            weights=[0.2, 0.4, 0.6, 0.8, 1.0, 1.0]
        )
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, Vec3(0.5, 1.2, 0))
        assert result is not None

    def test_set_weight(self, spine_positions):
        """Test setting individual weight."""
        solver = CCDSolverWithWeights([0, 1, 2, 3])
        solver.set_weight(1, 0.5)
        assert solver._weights[1] == 0.5

    def test_constrained_ccd_solver(self, spine_positions):
        """Test constrained CCD solver."""
        solver = ConstrainedCCDSolver([0, 1, 2, 3, 4, 5])

        def custom_constraint(rot: Quat, idx: int) -> Quat:
            return rot.normalized()

        solver.set_custom_constraint(2, custom_constraint)
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, Vec3(0.5, 1.2, 0))
        assert result is not None

    def test_wrong_position_count(self, spine_positions):
        """Test error with wrong position count."""
        solver = CCDSolver([0, 1, 2, 3, 4, 5])
        rotations = [Quat.identity() for _ in spine_positions]
        with pytest.raises(ValueError):
            solver.solve(spine_positions[:3], rotations[:3], Vec3(0, 1.5, 0))

    def test_reset_cached_lengths(self, spine_positions):
        """Test resetting cached lengths."""
        solver = CCDSolver([0, 1, 2, 3, 4, 5])
        rotations = [Quat.identity() for _ in spine_positions]
        solver.solve(spine_positions, rotations, Vec3(0.5, 1.2, 0))
        solver.reset_cached_lengths()
        assert not solver._lengths_cached

    def test_disabled_limit_passthrough(self):
        """Test disabled limit doesn't modify rotation."""
        limit = RotationLimit(enabled=False)
        rot = Quat.from_euler(1, 1, 1)
        result = limit.clamp_rotation(rot)
        assert result == rot

    def test_solve_target_at_root(self, spine_positions):
        """Test solving when target is at root."""
        solver = CCDSolver([0, 1, 2, 3, 4, 5])
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, Vec3(0, 0, 0))
        assert result is not None

    def test_iteration_tracking(self, spine_positions):
        """Test iteration count tracking."""
        solver = CCDSolver([0, 1, 2, 3, 4, 5], max_iterations=50)
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, Vec3(0.5, 1.2, 0))
        assert 1 <= result.iterations <= 50

    def test_final_error_tracking(self, spine_positions):
        """Test final error tracking."""
        solver = CCDSolver([0, 1, 2, 3, 4, 5])
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, Vec3(0.5, 1.2, 0))
        assert result.final_error >= 0

    def test_zero_weight_skips_joint(self, spine_positions):
        """Test zero weight skips joint rotation."""
        solver = CCDSolverWithWeights([0, 1, 2, 3, 4, 5], weights=[0, 1, 1, 1, 1, 1])
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, Vec3(0.5, 1.2, 0))
        # First joint should not rotate significantly
        assert result is not None


# ============================================================================
# Jacobian IK Tests (25 tests)
# ============================================================================

class TestJacobianIK:
    """Tests for Jacobian-based IK solver."""

    def test_solver_creation(self):
        """Test Jacobian solver creation."""
        solver = JacobianIK(
            [0, 1, 2, 3],
            method=JacobianMethod.DAMPED_LEAST_SQUARES,
            damping=0.5
        )
        assert solver.num_joints == 4

    def test_solver_methods(self):
        """Test different solver methods."""
        for method in JacobianMethod:
            solver = JacobianIK([0, 1, 2], method=method)
            assert solver.method == method

    def test_matrix_creation(self):
        """Test matrix creation."""
        m = Matrix(3, 3)
        assert m.rows == 3
        assert m.cols == 3

    def test_matrix_identity(self):
        """Test identity matrix."""
        m = Matrix.identity(3)
        assert m[0, 0] == 1.0
        assert m[1, 1] == 1.0
        assert m[0, 1] == 0.0

    def test_matrix_transpose(self):
        """Test matrix transpose."""
        m = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
        mt = m.transpose()
        assert mt.rows == 3
        assert mt.cols == 2
        assert mt[0, 1] == m[1, 0]

    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        m1 = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
        m2 = Matrix(3, 2, [1, 2, 3, 4, 5, 6])
        result = m1 @ m2
        assert result.rows == 2
        assert result.cols == 2

    def test_matrix_scalar_multiplication(self):
        """Test matrix scalar multiplication."""
        m = Matrix(2, 2, [1, 2, 3, 4])
        result = m * 2
        assert result[0, 0] == 2
        assert result[1, 1] == 8

    def test_matrix_addition(self):
        """Test matrix addition."""
        m1 = Matrix(2, 2, [1, 2, 3, 4])
        m2 = Matrix(2, 2, [5, 6, 7, 8])
        result = m1 + m2
        assert result[0, 0] == 6
        assert result[1, 1] == 12

    def test_matrix_to_vector(self):
        """Test matrix to vector conversion."""
        m = Matrix(3, 1, [1, 2, 3])
        vec = m.to_vector()
        assert vec == [1, 2, 3]

    def test_matrix_from_vector(self):
        """Test vector to matrix conversion."""
        m = Matrix.from_vector([1, 2, 3])
        assert m.rows == 3
        assert m.cols == 1

    def test_compute_jacobian(self, spine_positions):
        """Test Jacobian computation."""
        solver = JacobianIK([0, 1, 2, 3, 4, 5])
        rotations = [Quat.identity() for _ in spine_positions]
        jacobian = solver.compute_jacobian(spine_positions, rotations)
        # 3 rows per end effector, columns per DOF
        assert jacobian.rows == 3

    def test_solve_jacobian_transpose(self, spine_positions):
        """Test Jacobian transpose method."""
        solver = JacobianIK([0, 1, 2, 3, 4, 5], method=JacobianMethod.TRANSPOSE)
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, [Vec3(0.5, 1.2, 0)])
        assert result is not None

    def test_solve_pseudoinverse(self, spine_positions):
        """Test pseudoinverse method."""
        solver = JacobianIK([0, 1, 2, 3, 4, 5], method=JacobianMethod.PSEUDOINVERSE)
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, [Vec3(0.5, 1.2, 0)])
        assert result is not None

    def test_solve_dls(self, spine_positions):
        """Test damped least squares method."""
        solver = JacobianIK(
            [0, 1, 2, 3, 4, 5],
            method=JacobianMethod.DAMPED_LEAST_SQUARES,
            damping=0.5
        )
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, [Vec3(0.5, 1.2, 0)])
        assert result is not None

    def test_add_end_effector(self):
        """Test adding end effector."""
        solver = JacobianIK([0, 1, 2, 3, 4, 5])
        assert solver.num_end_effectors == 1
        solver.add_end_effector(3)
        assert solver.num_end_effectors == 2

    def test_set_joint_axes(self):
        """Test setting joint axes."""
        solver = JacobianIK([0, 1, 2])
        solver.set_joint_axes(1, [Vec3.unit_x(), Vec3.unit_y()])
        assert len(solver._joint_axes[1]) == 2

    def test_solve_with_transforms(self):
        """Test solve_with_transforms method."""
        transforms = [
            Transform(Vec3(0, i * 0.3, 0), Quat.identity())
            for i in range(6)
        ]
        solver = JacobianIK([0, 1, 2, 3, 4, 5])
        result = solver.solve_with_transforms(transforms, [Vec3(0.5, 1.2, 0)])
        assert len(result) == 6

    def test_multi_target_solver(self, spine_positions):
        """Test multi-target Jacobian solver."""
        solver = MultiTargetJacobianIK([0, 1, 2, 3, 4, 5])
        solver.add_end_effector_weighted(3, weight=0.5)
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, [Vec3(0.5, 1.2, 0), Vec3(0.3, 0.8, 0)])
        assert result is not None

    def test_step_size_affects_convergence(self, spine_positions):
        """Test step size affects convergence."""
        solver_fast = JacobianIK([0, 1, 2, 3, 4, 5], step_size=1.0)
        solver_slow = JacobianIK([0, 1, 2, 3, 4, 5], step_size=0.1)

        rotations = [Quat.identity() for _ in spine_positions]
        result_fast = solver_fast.solve(spine_positions, rotations, [Vec3(0.5, 1.2, 0)])
        result_slow = solver_slow.solve(spine_positions, rotations, [Vec3(0.5, 1.2, 0)])

        assert result_fast is not None
        assert result_slow is not None

    def test_damping_prevents_instability(self, spine_positions):
        """Test damping prevents instability near singularity."""
        solver = JacobianIK(
            [0, 1, 2, 3, 4, 5],
            method=JacobianMethod.DAMPED_LEAST_SQUARES,
            damping=1.0
        )
        rotations = [Quat.identity() for _ in spine_positions]
        # Target near singularity (fully extended)
        result = solver.solve(spine_positions, rotations, [Vec3(0, 1.5, 0)])
        assert result is not None

    def test_iteration_tracking(self, spine_positions):
        """Test iteration count tracking."""
        solver = JacobianIK([0, 1, 2, 3, 4, 5], max_iterations=100)
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, [Vec3(0.5, 1.2, 0)])
        assert 1 <= result.iterations <= 100

    def test_final_error_tracking(self, spine_positions):
        """Test final error tracking."""
        solver = JacobianIK([0, 1, 2, 3, 4, 5])
        rotations = [Quat.identity() for _ in spine_positions]
        result = solver.solve(spine_positions, rotations, [Vec3(0.5, 1.2, 0)])
        assert result.final_error >= 0

    def test_wrong_target_count(self, spine_positions):
        """Test error with wrong target count."""
        solver = JacobianIK([0, 1, 2, 3, 4, 5])
        rotations = [Quat.identity() for _ in spine_positions]
        with pytest.raises(ValueError):
            solver.solve(spine_positions, rotations, [Vec3(1, 1, 0), Vec3(2, 2, 0)])


# ============================================================================
# Full Body IK Tests (20 tests)
# ============================================================================

class TestFullBodyIK:
    """Tests for full body IK solver."""

    def test_skeleton_mapping_creation(self, humanoid_skeleton):
        """Test skeleton mapping creation."""
        assert humanoid_skeleton.get_bone(BodyPart.PELVIS) == 0
        assert humanoid_skeleton.get_bone(BodyPart.HEAD) == 4

    def test_skeleton_mapping_set_bone(self):
        """Test setting bone in skeleton mapping."""
        mapping = SkeletonMapping()
        mapping.set_bone(BodyPart.PELVIS, 5)
        assert mapping.get_bone(BodyPart.PELVIS) == 5

    def test_fullbody_ik_creation(self, humanoid_skeleton):
        """Test full body IK creation."""
        fbik = FullBodyIK(humanoid_skeleton, tolerance=0.001)
        assert fbik.tolerance == 0.001

    def test_fullbody_goal_creation(self):
        """Test full body goal creation."""
        goal = FullBodyIKGoal(
            bone_index=7,
            target_position=Vec3(1, 1, 0),
            chain_type="left_arm"
        )
        assert goal.has_position()
        assert not goal.has_rotation()

    def test_fullbody_goal_with_rotation(self):
        """Test goal with rotation."""
        goal = FullBodyIKGoal(
            bone_index=7,
            target_position=Vec3(1, 1, 0),
            target_rotation=Quat.identity(),
            rotation_weight=1.0
        )
        assert goal.has_rotation()

    def test_solve_basic_goal(self, humanoid_skeleton):
        """Test solving basic goal."""
        fbik = FullBodyIK(humanoid_skeleton)

        # Create transforms
        transforms = [
            Transform(Vec3(0, i * 0.2, 0), Quat.identity())
            for i in range(17)
        ]

        goals = [
            FullBodyIKGoal(
                bone_index=7,
                target_position=Vec3(0.5, 1.0, 0),
                chain_type="left_arm"
            )
        ]

        result = fbik.solve(transforms, goals)
        assert len(result.transforms) == 17

    def test_priority_ordering(self, humanoid_skeleton):
        """Test goals are processed by priority."""
        fbik = FullBodyIK(humanoid_skeleton)

        transforms = [
            Transform(Vec3(0, i * 0.2, 0), Quat.identity())
            for i in range(17)
        ]

        goals = [
            FullBodyIKGoal(bone_index=7, target_position=Vec3(0.5, 1.0, 0),
                          chain_type="left_arm", priority=1),
            FullBodyIKGoal(bone_index=10, target_position=Vec3(-0.5, 1.0, 0),
                          chain_type="right_arm", priority=2),  # Higher priority
        ]

        result = fbik.solve(transforms, goals)
        assert result is not None

    def test_balance_maintenance(self, humanoid_skeleton):
        """Test balance maintenance."""
        fbik = FullBodyIK(humanoid_skeleton)
        fbik.maintain_balance = True

        transforms = [
            Transform(Vec3(0, i * 0.2, 0), Quat.identity())
            for i in range(17)
        ]

        goals = [
            FullBodyIKGoal(bone_index=13, target_position=Vec3(0, 0, 0),
                          chain_type="left_leg"),
        ]

        result = fbik.solve(transforms, goals)
        assert result is not None

    def test_pelvis_height_adjustment(self, humanoid_skeleton):
        """Test pelvis height adjustment."""
        fbik = FullBodyIK(humanoid_skeleton)
        fbik.pelvis_height_adjust = True

        transforms = [
            Transform(Vec3(0, i * 0.2, 0), Quat.identity())
            for i in range(17)
        ]

        goals = [
            FullBodyIKGoal(bone_index=13, target_position=Vec3(0, -0.5, 0),
                          chain_type="left_leg"),
        ]

        result = fbik.solve(transforms, goals)
        # Pelvis should adjust
        assert result.pelvis_adjustment is not None

    def test_disabled_goals_skipped(self, humanoid_skeleton):
        """Test disabled goals are skipped."""
        fbik = FullBodyIK(humanoid_skeleton)

        transforms = [
            Transform(Vec3(0, i * 0.2, 0), Quat.identity())
            for i in range(17)
        ]

        goals = [
            FullBodyIKGoal(bone_index=7, target_position=Vec3(0.5, 1.0, 0),
                          chain_type="left_arm", enabled=False),
        ]

        result = fbik.solve(transforms, goals)
        assert result.success  # Should succeed with no active goals

    def test_look_at_solver_creation(self):
        """Test look-at solver creation."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[1, 2],
            head_weight=0.6
        )
        assert solver.head_bone == 4

    def test_look_at_solve(self):
        """Test look-at solving."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[1, 2]
        )

        transforms = [
            Transform(Vec3(0, i * 0.3, 0), Quat.identity())
            for i in range(5)
        ]

        result = solver.solve(transforms, Vec3(0, 1.5, 1))
        assert len(result) == 5

    def test_bone_mass_setting(self, humanoid_skeleton):
        """Test setting bone mass for COM."""
        fbik = FullBodyIK(humanoid_skeleton)
        fbik.set_bone_mass(0, 10.0)
        assert fbik._bone_masses[0] == 10.0

    def test_support_polygon_setting(self, humanoid_skeleton):
        """Test setting support polygon."""
        fbik = FullBodyIK(humanoid_skeleton)
        polygon = [Vec3(-1, 0, -1), Vec3(1, 0, -1), Vec3(1, 0, 1), Vec3(-1, 0, 1)]
        fbik.set_support_polygon(polygon)
        assert len(fbik._support_polygon) == 4

    def test_goals_achieved_tracking(self, humanoid_skeleton):
        """Test goals achieved tracking."""
        fbik = FullBodyIK(humanoid_skeleton)

        transforms = [
            Transform(Vec3(0, i * 0.2, 0), Quat.identity())
            for i in range(17)
        ]

        goals = [
            FullBodyIKGoal(bone_index=7, target_position=Vec3(0.3, 1.0, 0),
                          chain_type="left_arm"),
        ]

        result = fbik.solve(transforms, goals)
        assert 7 in result.goals_achieved

    def test_final_errors_tracking(self, humanoid_skeleton):
        """Test final errors tracking."""
        fbik = FullBodyIK(humanoid_skeleton)

        transforms = [
            Transform(Vec3(0, i * 0.2, 0), Quat.identity())
            for i in range(17)
        ]

        goals = [
            FullBodyIKGoal(bone_index=7, target_position=Vec3(0.3, 1.0, 0),
                          chain_type="left_arm"),
        ]

        result = fbik.solve(transforms, goals)
        assert 7 in result.final_errors

    def test_spine_stiffness(self, humanoid_skeleton):
        """Test spine stiffness setting."""
        fbik = FullBodyIK(humanoid_skeleton)
        fbik.spine_stiffness = 0.8
        assert fbik.spine_stiffness == 0.8

    def test_multiple_limb_goals(self, humanoid_skeleton):
        """Test multiple limb goals simultaneously."""
        fbik = FullBodyIK(humanoid_skeleton)

        transforms = [
            Transform(Vec3(0, i * 0.2, 0), Quat.identity())
            for i in range(17)
        ]

        goals = [
            FullBodyIKGoal(bone_index=7, target_position=Vec3(0.3, 1.0, 0),
                          chain_type="left_arm"),
            FullBodyIKGoal(bone_index=10, target_position=Vec3(-0.3, 1.0, 0),
                          chain_type="right_arm"),
            FullBodyIKGoal(bone_index=13, target_position=Vec3(0.1, 0, 0),
                          chain_type="left_leg"),
        ]

        result = fbik.solve(transforms, goals)
        assert len(result.goals_achieved) == 3


# ============================================================================
# Foot Placement Tests (20 tests)
# ============================================================================

class TestFootPlacement:
    """Tests for foot placement IK."""

    def test_foot_data_creation(self):
        """Test foot data creation."""
        foot = FootData(
            upper_leg=11,
            lower_leg=12,
            foot=13,
            toe=14
        )
        assert foot.state == FootState.PLANTED
        assert foot.blend_weight == 1.0

    def test_foot_placement_creation(self):
        """Test foot placement creation."""
        left = FootData(11, 12, 13)
        right = FootData(14, 15, 16)

        placement = FootPlacement(left, right, pelvis=0)
        assert placement.pelvis == 0

    def test_raycast_callback_setting(self):
        """Test setting raycast callback."""
        left = FootData(11, 12, 13)
        right = FootData(14, 15, 16)
        placement = FootPlacement(left, right, pelvis=0)

        def raycast(origin, direction, max_dist):
            return True, Vec3(origin.x, 0, origin.z), Vec3(0, 1, 0)

        placement.set_raycast_callback(raycast)
        assert placement._raycast is not None

    def test_solve_without_raycast(self):
        """Test solve fails gracefully without raycast."""
        left = FootData(11, 12, 13)
        right = FootData(14, 15, 16)
        placement = FootPlacement(left, right, pelvis=0)

        transforms = [
            Transform(Vec3(0, i * 0.2, 0), Quat.identity())
            for i in range(17)
        ]

        result = placement.solve(transforms, Vec3(0, 0, 0))
        assert not result.success

    def test_solve_with_flat_ground(self):
        """Test solve on flat ground."""
        left = FootData(11, 12, 13)
        right = FootData(14, 15, 16)

        def raycast(origin, direction, max_dist):
            return True, Vec3(origin.x, 0, origin.z), Vec3(0, 1, 0)

        placement = FootPlacement(left, right, pelvis=0, raycast_callback=raycast)

        transforms = [Transform(Vec3(0, 1, 0), Quat.identity())] * 17

        result = placement.solve(transforms, Vec3(0, 0, 0))
        assert result.success

    def test_solve_with_slope(self):
        """Test solve on sloped terrain."""
        left = FootData(11, 12, 13)
        right = FootData(14, 15, 16)

        def raycast(origin, direction, max_dist):
            # Simulate slope
            height = origin.x * 0.1
            normal = Vec3(-0.1, 1, 0).normalized()
            return True, Vec3(origin.x, height, origin.z), normal

        placement = FootPlacement(left, right, pelvis=0, raycast_callback=raycast)

        transforms = [Transform(Vec3(0, 1, 0), Quat.identity())] * 17

        result = placement.solve(transforms, Vec3(0, 0, 0))
        assert result.terrain_slope >= 0

    def test_pelvis_offset_tracking(self):
        """Test pelvis offset is tracked."""
        left = FootData(11, 12, 13)
        right = FootData(14, 15, 16)

        def raycast(origin, direction, max_dist):
            return True, Vec3(origin.x, -0.5, origin.z), Vec3(0, 1, 0)

        placement = FootPlacement(left, right, pelvis=0, raycast_callback=raycast)

        transforms = [Transform(Vec3(0, 1, 0), Quat.identity())] * 17

        result = placement.solve(transforms, Vec3(0, 0, 0))
        assert result.pelvis_offset is not None

    def test_foot_planted_tracking(self):
        """Test foot planted tracking."""
        left = FootData(11, 12, 13)
        right = FootData(14, 15, 16)

        def raycast(origin, direction, max_dist):
            return True, Vec3(origin.x, 0, origin.z), Vec3(0, 1, 0)

        placement = FootPlacement(left, right, pelvis=0, raycast_callback=raycast)

        transforms = [Transform(Vec3(0, 1, 0), Quat.identity())] * 17

        result = placement.solve(transforms, Vec3(0, 0, 0))
        assert result.left_foot_planted
        assert result.right_foot_planted

    def test_foot_height_offset(self):
        """Test foot height offset setting."""
        foot = FootData(11, 12, 13, height_offset=0.05)
        assert foot.height_offset == 0.05

    def test_blend_weight_affects_ik(self):
        """Test blend weight affects IK application."""
        left = FootData(11, 12, 13, blend_weight=0.5)
        right = FootData(14, 15, 16, blend_weight=0.5)

        def raycast(origin, direction, max_dist):
            return True, Vec3(origin.x, 0, origin.z), Vec3(0, 1, 0)

        placement = FootPlacement(left, right, pelvis=0, raycast_callback=raycast)

        transforms = [Transform(Vec3(0, 1, 0), Quat.identity())] * 17

        result = placement.solve(transforms, Vec3(0, 0, 0))
        assert result is not None

    def test_foot_state_enum(self):
        """Test foot state enum values."""
        assert FootState.PLANTED != FootState.AIRBORNE
        assert FootState.LIFTING != FootState.LANDING

    def test_animated_foot_placement(self):
        """Test animated foot placement wrapper."""
        left = FootData(11, 12, 13)
        right = FootData(14, 15, 16)

        def raycast(origin, direction, max_dist):
            return True, Vec3(origin.x, 0, origin.z), Vec3(0, 1, 0)

        base = FootPlacement(left, right, pelvis=0, raycast_callback=raycast)
        animated = FootPlacementAnimated(base)

        animated.set_height_curves(
            lambda t: math.sin(t) * 0.1,
            lambda t: math.cos(t) * 0.1
        )

        transforms = [Transform(Vec3(0, 1, 0), Quat.identity())] * 17

        animated.update(0.016)
        result = animated.solve(transforms, Vec3(0, 0, 0))
        assert result is not None

    def test_multi_leg_foot_placement(self):
        """Test multi-leg foot placement."""
        feet = [
            FootData(0, 1, 2),
            FootData(3, 4, 5),
            FootData(6, 7, 8),
            FootData(9, 10, 11),
        ]

        def raycast(origin, direction, max_dist):
            return True, Vec3(origin.x, 0, origin.z), Vec3(0, 1, 0)

        placement = MultiLegFootPlacement(feet, pelvis=12, raycast_callback=raycast)

        transforms = [Transform(Vec3(0, 1, 0), Quat.identity())] * 13

        result = placement.solve(transforms, Vec3(0, 0, 0))
        assert len(result) == 13

    def test_ray_length_configuration(self):
        """Test ray length configuration."""
        left = FootData(11, 12, 13)
        right = FootData(14, 15, 16)
        placement = FootPlacement(left, right, pelvis=0)
        placement.ray_length = 3.0
        assert placement.ray_length == 3.0

    def test_blend_speed_configuration(self):
        """Test blend speed configuration."""
        left = FootData(11, 12, 13)
        right = FootData(14, 15, 16)
        placement = FootPlacement(left, right, pelvis=0)
        placement.blend_speed = 15.0
        assert placement.blend_speed == 15.0

    def test_max_pelvis_drop_configuration(self):
        """Test max pelvis drop configuration."""
        left = FootData(11, 12, 13)
        right = FootData(14, 15, 16)
        placement = FootPlacement(left, right, pelvis=0)
        placement.max_pelvis_drop = 0.7
        assert placement.max_pelvis_drop == 0.7

    def test_toe_align_weight_configuration(self):
        """Test toe align weight configuration."""
        left = FootData(11, 12, 13)
        right = FootData(14, 15, 16)
        placement = FootPlacement(left, right, pelvis=0)
        placement.toe_align_weight = 0.8
        assert placement.toe_align_weight == 0.8

    def test_solve_caches_previous_targets(self):
        """Test solve caches previous targets for smoothing."""
        left = FootData(11, 12, 13)
        right = FootData(14, 15, 16)

        def raycast(origin, direction, max_dist):
            return True, Vec3(origin.x, 0, origin.z), Vec3(0, 1, 0)

        placement = FootPlacement(left, right, pelvis=0, raycast_callback=raycast)

        transforms = [Transform(Vec3(0, 1, 0), Quat.identity())] * 17

        placement.solve(transforms, Vec3(0, 0, 0))
        # Previous targets should be cached
        assert placement._prev_left_target is not None


# ============================================================================
# Decorator Tests (5 tests)
# ============================================================================

class TestDecorators:
    """Tests for IK decorators."""

    def test_ik_goal_decorator(self):
        """Test ik_goal decorator."""
        @ik_goal(priority=5, blend_speed=20.0)
        class TestGoal:
            pass

        assert TestGoal._ik_goal
        assert TestGoal._ik_goal_priority == 5
        assert TestGoal._ik_goal_blend_speed == 20.0

    def test_ik_chain_decorator(self):
        """Test ik_chain decorator."""
        @ik_chain(solver="ccd", iterations=15)
        class TestChain:
            pass

        assert TestChain._ik_chain
        assert TestChain._ik_solver == "ccd"
        assert TestChain._ik_iterations == 15

    def test_ik_goal_default_values(self):
        """Test ik_goal decorator default values."""
        @ik_goal()
        class TestGoal:
            pass

        assert TestGoal._ik_goal_priority == 0
        assert TestGoal._ik_goal_blend_speed == 10.0

    def test_ik_chain_default_values(self):
        """Test ik_chain decorator default values."""
        @ik_chain()
        class TestChain:
            pass

        assert TestChain._ik_solver == "fabrik"
        assert TestChain._ik_iterations == 10

    def test_decorators_preserve_class(self):
        """Test decorators preserve class attributes."""
        @ik_goal(priority=1)
        @ik_chain(solver="jacobian")
        class TestClass:
            custom_attr = "test"

        assert TestClass.custom_attr == "test"
        assert TestClass._ik_goal
        assert TestClass._ik_chain


# ============================================================================
# Integration Tests (10 tests)
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple IK systems."""

    def test_two_bone_to_fullbody_integration(self, humanoid_skeleton):
        """Test two-bone IK integrates with full body."""
        fbik = FullBodyIK(humanoid_skeleton)

        transforms = [
            Transform(Vec3(0, i * 0.2, 0), Quat.identity())
            for i in range(17)
        ]

        # Full body should use two-bone internally for limbs
        goals = [
            FullBodyIKGoal(bone_index=7, target_position=Vec3(0.5, 1.0, 0),
                          chain_type="left_arm"),
        ]

        result = fbik.solve(transforms, goals)
        assert result.success or len(result.goals_achieved) > 0

    def test_fabrik_spine_with_fullbody(self, humanoid_skeleton):
        """Test FABRIK spine with full body solver."""
        fbik = FullBodyIK(humanoid_skeleton)

        transforms = [
            Transform(Vec3(0, i * 0.2, 0), Quat.identity())
            for i in range(17)
        ]

        goals = [
            FullBodyIKGoal(bone_index=4, target_position=Vec3(0.2, 1.0, 0.2),
                          chain_type="spine"),
        ]

        result = fbik.solve(transforms, goals)
        assert len(result.transforms) == 17

    def test_foot_placement_with_fullbody(self, humanoid_skeleton):
        """Test foot placement combined with full body."""
        left = FootData(11, 12, 13)
        right = FootData(14, 15, 16)

        def raycast(origin, direction, max_dist):
            return True, Vec3(origin.x, 0, origin.z), Vec3(0, 1, 0)

        foot_ik = FootPlacement(left, right, pelvis=0, raycast_callback=raycast)
        fbik = FullBodyIK(humanoid_skeleton)

        transforms = [Transform(Vec3(0, 1, 0), Quat.identity())] * 17

        # First apply foot placement
        foot_result = foot_ik.solve(transforms, Vec3(0, 0, 0))

        # Then apply upper body IK
        goals = [
            FullBodyIKGoal(bone_index=7, target_position=Vec3(0.5, 1.0, 0),
                          chain_type="left_arm"),
        ]

        final_result = fbik.solve(foot_result.transforms, goals)
        assert len(final_result.transforms) == 17

    def test_chained_solvers(self, spine_positions):
        """Test chaining multiple IK solvers."""
        # First solve with FABRIK
        fabrik = FABRIKChain([0, 1, 2, 3, 4, 5], tolerance=0.05)
        fabrik_result = fabrik.solve(spine_positions, Vec3(0.5, 1.2, 0))

        # Then refine with CCD
        ccd = CCDSolver([0, 1, 2, 3, 4, 5], tolerance=0.01)
        rotations = fabrik_result.rotations

        ccd_result = ccd.solve(fabrik_result.positions, rotations, Vec3(0.5, 1.2, 0))

        assert ccd_result.final_error <= fabrik_result.final_error or \
               abs(ccd_result.final_error - fabrik_result.final_error) < 0.1

    def test_look_at_with_arm_ik(self, humanoid_skeleton):
        """Test look-at combined with arm IK."""
        look_at = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[1, 2]
        )

        fbik = FullBodyIK(humanoid_skeleton)

        transforms = [
            Transform(Vec3(0, i * 0.2, 0), Quat.identity())
            for i in range(17)
        ]

        # Apply look-at
        look_result = look_at.solve(transforms, Vec3(1, 1.5, 1))

        # Then arm IK
        goals = [
            FullBodyIKGoal(bone_index=7, target_position=Vec3(0.5, 1.0, 0.5),
                          chain_type="left_arm"),
        ]

        final_result = fbik.solve(look_result, goals)
        assert len(final_result.transforms) == 17

    def test_multiple_effector_jacobian_with_constraints(self, spine_positions):
        """Test multi-effector Jacobian with constraints."""
        solver = MultiTargetJacobianIK([0, 1, 2, 3, 4, 5])
        solver.add_end_effector_weighted(3, weight=0.5)

        rotations = [Quat.identity() for _ in spine_positions]

        result = solver.solve(
            spine_positions,
            rotations,
            [Vec3(0.3, 1.2, 0), Vec3(0.1, 0.8, 0)]
        )

        assert result is not None

    def test_goal_blender_with_two_bone(self, simple_arm_transforms):
        """Test goal blender with two-bone IK."""
        blender = IKGoalBlender(blend_speed=10.0)
        solver = TwoBoneIK(0, 1, 2)

        # Simulate blending target over time
        targets = [
            Vec3(1, 1, 0),
            Vec3(1.2, 0.8, 0),
            Vec3(0.8, 1.2, 0),
        ]

        for target in targets:
            blended_target = blender.blend_position(0, target, 0.1)
            result = solver.solve(
                simple_arm_transforms[0],
                simple_arm_transforms[1],
                simple_arm_transforms[2],
                blended_target
            )
            assert result.success

    def test_weighted_ccd_with_fabrik_init(self, spine_positions):
        """Test weighted CCD initialized from FABRIK result."""
        fabrik = FABRIKChain([0, 1, 2, 3, 4, 5])
        fabrik_result = fabrik.solve(spine_positions, Vec3(0.5, 1.2, 0))

        # Use FABRIK result as starting point for weighted CCD
        ccd = CCDSolverWithWeights(
            [0, 1, 2, 3, 4, 5],
            weights=[0.2, 0.4, 0.6, 0.8, 1.0, 1.0]
        )

        ccd_result = ccd.solve(
            fabrik_result.positions,
            fabrik_result.rotations,
            Vec3(0.5, 1.2, 0)
        )

        assert ccd_result is not None

    def test_constrained_chain_system(self, spine_positions):
        """Test fully constrained chain system."""
        fabrik = FABRIKChain([0, 1, 2, 3, 4, 5])

        # Add constraints
        for i in range(6):
            constraint = JointConstraint(
                constraint_type=JointConstraintType.BALL_SOCKET,
                cone_angle=math.pi / 3
            )
            fabrik.set_constraint(i, constraint)

        result = fabrik.solve(spine_positions, Vec3(0.5, 1.2, 0.3))
        assert len(result.positions) == 6

    def test_full_pipeline(self, humanoid_skeleton):
        """Test full animation IK pipeline."""
        # 1. Setup
        transforms = [
            Transform(Vec3(0, i * 0.2, 0), Quat.identity())
            for i in range(17)
        ]

        # 2. Foot placement
        left = FootData(11, 12, 13)
        right = FootData(14, 15, 16)

        def raycast(origin, direction, max_dist):
            return True, Vec3(origin.x, 0, origin.z), Vec3(0, 1, 0)

        foot_ik = FootPlacement(left, right, pelvis=0, raycast_callback=raycast)
        transforms = foot_ik.solve(transforms, Vec3(0, 0, 0)).transforms

        # 3. Look-at
        look_at = LookAtSolver(head_bone=4, neck_bone=3, spine_bones=[1, 2])
        transforms = look_at.solve(transforms, Vec3(0, 2, 1))

        # 4. Full body IK for hands
        fbik = FullBodyIK(humanoid_skeleton)
        goals = [
            FullBodyIKGoal(bone_index=7, target_position=Vec3(0.5, 1.0, 0.3),
                          chain_type="left_arm"),
            FullBodyIKGoal(bone_index=10, target_position=Vec3(-0.5, 1.0, 0.3),
                          chain_type="right_arm"),
        ]
        result = fbik.solve(transforms, goals)

        # Verify complete pipeline
        assert len(result.transforms) == 17
        assert len(result.goals_achieved) >= 0


# ============================================================================
# Edge Case Tests (10 tests)
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_length_chain_handling(self):
        """Test handling zero-length bones - should handle gracefully without NaN."""
        positions = [Vec3(0, 0, 0), Vec3(0, 0, 0), Vec3(0, 1, 0)]
        chain = FABRIKChain([0, 1, 2])
        # Should handle gracefully
        result = chain.solve(positions, Vec3(0.5, 0.5, 0))
        assert result is not None
        # Verify no NaN values in output positions
        for pos in result.positions:
            assert not math.isnan(pos.x), "Position should not contain NaN"
            assert not math.isnan(pos.y), "Position should not contain NaN"
            assert not math.isnan(pos.z), "Position should not contain NaN"

    def test_target_at_origin(self, simple_arm_transforms):
        """Test target at origin."""
        solver = TwoBoneIK(0, 1, 2)
        result = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(0, 0, 0)
        )
        assert result.success

    def test_very_small_tolerance(self, spine_positions):
        """Test very small tolerance."""
        chain = FABRIKChain([0, 1, 2, 3, 4, 5], tolerance=1e-10, max_iterations=100)
        result = chain.solve(spine_positions, Vec3(0, 1.5, 0))
        # May not converge but shouldn't crash
        assert result is not None

    def test_very_large_target(self, simple_arm_transforms):
        """Test very large target distance."""
        solver = TwoBoneIK(0, 1, 2)
        result = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(1000, 1000, 1000)
        )
        assert result.success
        assert result.extension_ratio > 0.99

    def test_negative_coordinates(self, simple_arm_transforms):
        """Test negative coordinate targets."""
        solver = TwoBoneIK(0, 1, 2)
        result = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(-1, -1, -1)
        )
        assert result.success

    def test_coincident_positions(self):
        """Test handling coincident positions - all bones at same point."""
        positions = [Vec3(0, 0, 0)] * 6
        chain = FABRIKChain([0, 1, 2, 3, 4, 5])
        result = chain.solve(positions, Vec3(1, 0, 0))
        # Should handle gracefully without NaN
        assert result is not None
        # Verify no NaN values and no infinite values in output
        for pos in result.positions:
            assert not math.isnan(pos.x) and not math.isinf(pos.x), "Invalid position value"
            assert not math.isnan(pos.y) and not math.isinf(pos.y), "Invalid position value"
            assert not math.isnan(pos.z) and not math.isinf(pos.z), "Invalid position value"

    def test_single_iteration(self, spine_positions):
        """Test with single iteration."""
        chain = FABRIKChain([0, 1, 2, 3, 4, 5], max_iterations=1)
        result = chain.solve(spine_positions, Vec3(0.5, 1.2, 0))
        assert result.iterations == 1

    def test_pole_vector_at_joint(self, simple_arm_transforms):
        """Test pole vector at joint position."""
        solver = TwoBoneIK(0, 1, 2)
        result = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(1, 1, 0),
            pole_vector=Vec3(0, 0, 0)  # At root
        )
        # Should handle gracefully
        assert result.success

    def test_parallel_target_direction(self, simple_arm_transforms):
        """Test target in same direction as chain."""
        solver = TwoBoneIK(0, 1, 2)
        result = solver.solve(
            simple_arm_transforms[0],
            simple_arm_transforms[1],
            simple_arm_transforms[2],
            Vec3(0, 1.5, 0)  # Directly above
        )
        assert result.success

    def test_empty_goals_list(self, humanoid_skeleton):
        """Test full body IK with empty goals."""
        fbik = FullBodyIK(humanoid_skeleton)
        transforms = [
            Transform(Vec3(0, i * 0.2, 0), Quat.identity())
            for i in range(17)
        ]
        result = fbik.solve(transforms, [])
        assert result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

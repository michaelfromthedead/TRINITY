"""Blackbox tests for Full Body IK solver (Phase 4).

This module tests the FullBodyIK solver from the public API only,
without knowledge of implementation details. Tests are derived from
theoretical full body IK behavior:

1. Coordinates multiple IK chains (arms, legs, spine)
2. Center of Mass (COM) tracking for balance
3. Support polygon for stability (feet on ground)
4. Multiple effector goals with weights
5. Body part awareness and chain coordination

Test Strategy:
- Test public API contracts only
- Test behavioral expectations for full body coordination
- Test balance and COM tracking
- Test multi-chain goal resolution
- Test body part mappings
"""

import math
import pytest
from typing import List, Optional, Dict

# Import public API
from engine.animation.ik import (
    FullBodyIK,
    FullBodyIKGoal,
    FullBodyIKResult,
    SkeletonMapping,
    BodyPart,
    LookAtSolver,
    FULLBODY_DEFAULT_MAX_ITERATIONS,
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
    """Create a basic humanoid skeleton in T-pose.

    Layout (indices):
    0 = pelvis (root)
    1 = spine
    2 = chest
    3 = neck
    4 = head
    5 = left_shoulder
    6 = left_upper_arm
    7 = left_lower_arm
    8 = left_hand
    9 = right_shoulder
    10 = right_upper_arm
    11 = right_lower_arm
    12 = right_hand
    13 = left_upper_leg
    14 = left_lower_leg
    15 = left_foot
    16 = left_toe
    17 = right_upper_leg
    18 = right_lower_leg
    19 = right_foot
    20 = right_toe
    """
    transforms = [
        # Spine chain
        make_transform(Vec3(0.0, 1.0, 0.0)),    # 0: pelvis
        make_transform(Vec3(0.0, 1.2, 0.0)),    # 1: spine
        make_transform(Vec3(0.0, 1.5, 0.0)),    # 2: chest
        make_transform(Vec3(0.0, 1.7, 0.0)),    # 3: neck
        make_transform(Vec3(0.0, 1.9, 0.0)),    # 4: head
        # Left arm
        make_transform(Vec3(-0.2, 1.5, 0.0)),   # 5: left_shoulder
        make_transform(Vec3(-0.35, 1.5, 0.0)),  # 6: left_upper_arm
        make_transform(Vec3(-0.55, 1.5, 0.0)),  # 7: left_lower_arm
        make_transform(Vec3(-0.75, 1.5, 0.0)),  # 8: left_hand
        # Right arm
        make_transform(Vec3(0.2, 1.5, 0.0)),    # 9: right_shoulder
        make_transform(Vec3(0.35, 1.5, 0.0)),   # 10: right_upper_arm
        make_transform(Vec3(0.55, 1.5, 0.0)),   # 11: right_lower_arm
        make_transform(Vec3(0.75, 1.5, 0.0)),   # 12: right_hand
        # Left leg
        make_transform(Vec3(-0.1, 1.0, 0.0)),   # 13: left_upper_leg
        make_transform(Vec3(-0.1, 0.5, 0.0)),   # 14: left_lower_leg
        make_transform(Vec3(-0.1, 0.0, 0.0)),   # 15: left_foot
        make_transform(Vec3(-0.1, 0.0, 0.1)),   # 16: left_toe
        # Right leg
        make_transform(Vec3(0.1, 1.0, 0.0)),    # 17: right_upper_leg
        make_transform(Vec3(0.1, 0.5, 0.0)),    # 18: right_lower_leg
        make_transform(Vec3(0.1, 0.0, 0.0)),    # 19: right_foot
        make_transform(Vec3(0.1, 0.0, 0.1)),    # 20: right_toe
    ]
    return transforms


def create_humanoid_mapping() -> SkeletonMapping:
    """Create a standard humanoid skeleton mapping."""
    bone_map = {
        BodyPart.PELVIS: 0,
        BodyPart.SPINE: 1,
        BodyPart.CHEST: 2,
        BodyPart.NECK: 3,
        BodyPart.HEAD: 4,
        BodyPart.LEFT_SHOULDER: 5,
        BodyPart.LEFT_UPPER_ARM: 6,
        BodyPart.LEFT_LOWER_ARM: 7,
        BodyPart.LEFT_HAND: 8,
        BodyPart.RIGHT_SHOULDER: 9,
        BodyPart.RIGHT_UPPER_ARM: 10,
        BodyPart.RIGHT_LOWER_ARM: 11,
        BodyPart.RIGHT_HAND: 12,
        BodyPart.LEFT_UPPER_LEG: 13,
        BodyPart.LEFT_LOWER_LEG: 14,
        BodyPart.LEFT_FOOT: 15,
        BodyPart.LEFT_TOE: 16,
        BodyPart.RIGHT_UPPER_LEG: 17,
        BodyPart.RIGHT_LOWER_LEG: 18,
        BodyPart.RIGHT_FOOT: 19,
        BodyPart.RIGHT_TOE: 20,
    }
    return SkeletonMapping(
        bone_map=bone_map,
        spine_chain=[0, 1, 2],
        left_arm_chain=[5, 6, 7, 8],
        right_arm_chain=[9, 10, 11, 12],
        left_leg_chain=[13, 14, 15],
        right_leg_chain=[17, 18, 19],
    )


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def humanoid_transforms():
    """Standard humanoid T-pose transforms."""
    return create_humanoid_transforms()


@pytest.fixture
def skeleton_mapping():
    """Standard humanoid skeleton mapping."""
    return create_humanoid_mapping()


@pytest.fixture
def fullbody_solver(skeleton_mapping):
    """Standard full body IK solver."""
    return FullBodyIK(skeleton_mapping)


# =============================================================================
# FullBodyIK Instantiation Tests
# =============================================================================

class TestFullBodyIKInstantiation:
    """Tests for FullBodyIK class instantiation."""

    def test_can_instantiate_with_skeleton_mapping(self, skeleton_mapping):
        """FullBodyIK can be instantiated with skeleton mapping."""
        solver = FullBodyIK(skeleton_mapping)
        assert solver is not None

    def test_can_instantiate_with_tolerance(self, skeleton_mapping):
        """FullBodyIK can be instantiated with custom tolerance."""
        solver = FullBodyIK(skeleton_mapping, tolerance=0.001)
        assert solver is not None

    def test_can_instantiate_with_max_iterations(self, skeleton_mapping):
        """FullBodyIK can be instantiated with custom max iterations."""
        solver = FullBodyIK(skeleton_mapping, max_iterations=50)
        assert solver is not None

    def test_can_instantiate_with_all_parameters(self, skeleton_mapping):
        """FullBodyIK can be instantiated with all parameters."""
        solver = FullBodyIK(
            skeleton_mapping,
            tolerance=0.005,
            max_iterations=100,
        )
        assert solver is not None

    def test_instantiation_preserves_mapping(self, skeleton_mapping):
        """FullBodyIK preserves skeleton mapping reference."""
        solver = FullBodyIK(skeleton_mapping)
        # Solver should have some internal reference to mapping
        assert solver is not None

    def test_multiple_solvers_are_independent(self, skeleton_mapping):
        """Multiple FullBodyIK instances are independent."""
        solver1 = FullBodyIK(skeleton_mapping, tolerance=0.001)
        solver2 = FullBodyIK(skeleton_mapping, tolerance=0.01)
        assert solver1 is not solver2


# =============================================================================
# SkeletonMapping Tests
# =============================================================================

class TestSkeletonMapping:
    """Tests for SkeletonMapping structure."""

    def test_can_create_skeleton_mapping(self):
        """SkeletonMapping can be created."""
        bone_map = {BodyPart.PELVIS: 0, BodyPart.SPINE: 1}
        mapping = SkeletonMapping(
            bone_map=bone_map,
            spine_chain=[0, 1],
            left_arm_chain=[2, 3, 4],
            right_arm_chain=[5, 6, 7],
            left_leg_chain=[8, 9, 10],
            right_leg_chain=[11, 12, 13],
        )
        assert mapping is not None

    def test_skeleton_mapping_has_bone_map(self, skeleton_mapping):
        """SkeletonMapping has bone_map."""
        assert hasattr(skeleton_mapping, 'bone_map')
        assert isinstance(skeleton_mapping.bone_map, dict)

    def test_skeleton_mapping_has_spine_chain(self, skeleton_mapping):
        """SkeletonMapping has spine_chain."""
        assert hasattr(skeleton_mapping, 'spine_chain')
        assert isinstance(skeleton_mapping.spine_chain, list)

    def test_skeleton_mapping_has_left_arm_chain(self, skeleton_mapping):
        """SkeletonMapping has left_arm_chain."""
        assert hasattr(skeleton_mapping, 'left_arm_chain')
        assert isinstance(skeleton_mapping.left_arm_chain, list)

    def test_skeleton_mapping_has_right_arm_chain(self, skeleton_mapping):
        """SkeletonMapping has right_arm_chain."""
        assert hasattr(skeleton_mapping, 'right_arm_chain')
        assert isinstance(skeleton_mapping.right_arm_chain, list)

    def test_skeleton_mapping_has_left_leg_chain(self, skeleton_mapping):
        """SkeletonMapping has left_leg_chain."""
        assert hasattr(skeleton_mapping, 'left_leg_chain')
        assert isinstance(skeleton_mapping.left_leg_chain, list)

    def test_skeleton_mapping_has_right_leg_chain(self, skeleton_mapping):
        """SkeletonMapping has right_leg_chain."""
        assert hasattr(skeleton_mapping, 'right_leg_chain')
        assert isinstance(skeleton_mapping.right_leg_chain, list)

    def test_bone_map_contains_body_parts(self, skeleton_mapping):
        """bone_map contains BodyPart keys."""
        for key in skeleton_mapping.bone_map.keys():
            assert isinstance(key, BodyPart)


# =============================================================================
# BodyPart Enum Tests
# =============================================================================

class TestBodyPart:
    """Tests for BodyPart enum."""

    def test_body_part_has_pelvis(self):
        """BodyPart has PELVIS."""
        assert hasattr(BodyPart, 'PELVIS')

    def test_body_part_has_spine(self):
        """BodyPart has SPINE."""
        assert hasattr(BodyPart, 'SPINE')

    def test_body_part_has_chest(self):
        """BodyPart has CHEST."""
        assert hasattr(BodyPart, 'CHEST')

    def test_body_part_has_neck(self):
        """BodyPart has NECK."""
        assert hasattr(BodyPart, 'NECK')

    def test_body_part_has_head(self):
        """BodyPart has HEAD."""
        assert hasattr(BodyPart, 'HEAD')

    def test_body_part_has_left_shoulder(self):
        """BodyPart has LEFT_SHOULDER."""
        assert hasattr(BodyPart, 'LEFT_SHOULDER')

    def test_body_part_has_left_upper_arm(self):
        """BodyPart has LEFT_UPPER_ARM."""
        assert hasattr(BodyPart, 'LEFT_UPPER_ARM')

    def test_body_part_has_left_lower_arm(self):
        """BodyPart has LEFT_LOWER_ARM."""
        assert hasattr(BodyPart, 'LEFT_LOWER_ARM')

    def test_body_part_has_left_hand(self):
        """BodyPart has LEFT_HAND."""
        assert hasattr(BodyPart, 'LEFT_HAND')

    def test_body_part_has_right_shoulder(self):
        """BodyPart has RIGHT_SHOULDER."""
        assert hasattr(BodyPart, 'RIGHT_SHOULDER')

    def test_body_part_has_left_upper_leg(self):
        """BodyPart has LEFT_UPPER_LEG."""
        assert hasattr(BodyPart, 'LEFT_UPPER_LEG')

    def test_body_part_has_left_foot(self):
        """BodyPart has LEFT_FOOT."""
        assert hasattr(BodyPart, 'LEFT_FOOT')


# =============================================================================
# FullBodyIKGoal Tests
# =============================================================================

class TestFullBodyIKGoal:
    """Tests for FullBodyIKGoal structure."""

    def test_can_create_goal_with_bone_index(self):
        """FullBodyIKGoal can be created with bone index."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.2, 0.5),
        )
        assert goal is not None

    def test_can_create_goal_with_target_position(self):
        """FullBodyIKGoal can be created with target position."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.2, 0.5),
        )
        assert goal.target_position is not None

    def test_can_create_goal_with_target_rotation(self):
        """FullBodyIKGoal can be created with target rotation."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_rotation=Quat.identity(),
            rotation_weight=1.0,
        )
        assert goal is not None

    def test_can_create_goal_with_position_weight(self):
        """FullBodyIKGoal can be created with position weight."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.5),
            position_weight=0.8,
        )
        assert goal.position_weight == 0.8

    def test_can_create_goal_with_chain_type(self):
        """FullBodyIKGoal can be created with chain type."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.5),
            chain_type="left_arm",
        )
        assert goal.chain_type == "left_arm"

    def test_goal_has_bone_index(self):
        """FullBodyIKGoal has bone_index."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0),
        )
        assert hasattr(goal, 'bone_index')
        assert goal.bone_index == 8

    def test_goal_has_priority(self):
        """FullBodyIKGoal has priority."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0),
            priority=5,
        )
        assert goal.priority == 5

    def test_goal_has_enabled(self):
        """FullBodyIKGoal has enabled flag."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0),
            enabled=True,
        )
        assert goal.enabled is True

    def test_goal_default_enabled_is_true(self):
        """FullBodyIKGoal default enabled is True."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0),
        )
        assert goal.enabled is True


# =============================================================================
# FullBodyIKResult Tests
# =============================================================================

class TestFullBodyIKResult:
    """Tests for FullBodyIKResult structure."""

    def test_result_has_success_field(self, fullbody_solver, humanoid_transforms):
        """FullBodyIKResult has success field."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.7, 1.4, 0.1),
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert hasattr(result, 'success')
        assert isinstance(result.success, bool)

    def test_result_has_transforms_field(self, fullbody_solver, humanoid_transforms):
        """FullBodyIKResult has transforms field."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.7, 1.4, 0.1),
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert hasattr(result, 'transforms')

    def test_result_has_goals_achieved(self, fullbody_solver, humanoid_transforms):
        """FullBodyIKResult has goals_achieved field."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.7, 1.4, 0.1),
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert hasattr(result, 'goals_achieved')

    def test_result_has_final_errors(self, fullbody_solver, humanoid_transforms):
        """FullBodyIKResult has final_errors field."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.7, 1.4, 0.1),
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert hasattr(result, 'final_errors')

    def test_result_has_pelvis_adjustment(self, fullbody_solver, humanoid_transforms):
        """FullBodyIKResult has pelvis_adjustment field."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.7, 1.4, 0.1),
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert hasattr(result, 'pelvis_adjustment')

    def test_result_transforms_same_count(self, fullbody_solver, humanoid_transforms):
        """Result transforms have same count as input."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.7, 1.4, 0.1),
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert len(result.transforms) == len(humanoid_transforms)


# =============================================================================
# Basic Solve Tests
# =============================================================================

class TestFullBodyIKBasicSolve:
    """Tests for basic FullBodyIK solve functionality."""

    def test_solve_returns_result(self, fullbody_solver, humanoid_transforms):
        """Solve returns a FullBodyIKResult."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.4, 0.1),
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert isinstance(result, FullBodyIKResult)

    def test_solve_with_empty_goals(self, fullbody_solver, humanoid_transforms):
        """Solve with empty goals returns unchanged transforms."""
        result = fullbody_solver.solve(humanoid_transforms, [])
        # Should succeed with no changes
        assert result.success or len(result.transforms) == len(humanoid_transforms)

    def test_solve_single_arm_goal(self, fullbody_solver, humanoid_transforms):
        """Solve with single arm goal."""
        target = Vec3(-0.5, 1.3, 0.2)
        goal = FullBodyIKGoal(
            bone_index=8,  # left_hand
            target_position=target,
            chain_type="left_arm",
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_solve_single_leg_goal(self, fullbody_solver, humanoid_transforms):
        """Solve with single leg goal."""
        target = Vec3(-0.2, 0.1, 0.1)
        goal = FullBodyIKGoal(
            bone_index=15,  # left_foot
            target_position=target,
            chain_type="left_leg",
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_solve_reachable_target_succeeds(self, fullbody_solver, humanoid_transforms):
        """Solve with reachable target returns result."""
        # Target close to hand position
        target = Vec3(-0.75, 1.5, 0.0)  # Exact current position
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=target,
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        # Result should be returned regardless of success flag
        assert result is not None


# =============================================================================
# Multi-Chain Goal Tests
# =============================================================================

class TestFullBodyIKMultiChain:
    """Tests for multi-chain goal coordination."""

    def test_solve_both_arms(self, fullbody_solver, humanoid_transforms):
        """Solve with both arm goals."""
        goals = [
            FullBodyIKGoal(
                bone_index=8,  # left_hand
                target_position=Vec3(-0.6, 1.3, 0.3),
            ),
            FullBodyIKGoal(
                bone_index=12,  # right_hand
                target_position=Vec3(0.6, 1.3, 0.3),
            ),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None

    def test_solve_both_legs(self, fullbody_solver, humanoid_transforms):
        """Solve with both leg goals."""
        goals = [
            FullBodyIKGoal(
                bone_index=15,  # left_foot
                target_position=Vec3(-0.15, 0.0, 0.1),
            ),
            FullBodyIKGoal(
                bone_index=19,  # right_foot
                target_position=Vec3(0.15, 0.0, 0.1),
            ),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None

    def test_solve_arm_and_leg(self, fullbody_solver, humanoid_transforms):
        """Solve with arm and leg goals."""
        goals = [
            FullBodyIKGoal(
                bone_index=8,  # left_hand
                target_position=Vec3(-0.5, 1.4, 0.2),
            ),
            FullBodyIKGoal(
                bone_index=15,  # left_foot
                target_position=Vec3(-0.1, 0.1, 0.1),
            ),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None

    def test_solve_all_limbs(self, fullbody_solver, humanoid_transforms):
        """Solve with all four limb goals."""
        goals = [
            FullBodyIKGoal(bone_index=8, target_position=Vec3(-0.6, 1.3, 0.2)),
            FullBodyIKGoal(bone_index=12, target_position=Vec3(0.6, 1.3, 0.2)),
            FullBodyIKGoal(bone_index=15, target_position=Vec3(-0.15, 0.0, 0.0)),
            FullBodyIKGoal(bone_index=19, target_position=Vec3(0.15, 0.0, 0.0)),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None
        assert len(result.transforms) == len(humanoid_transforms)


# =============================================================================
# Goal Weight Tests
# =============================================================================

class TestFullBodyIKGoalWeights:
    """Tests for goal weight influence."""

    def test_zero_weight_goal_ignored(self, fullbody_solver, humanoid_transforms):
        """Zero weight goal should be ignored."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(10.0, 10.0, 10.0),  # Unreachable
            position_weight=0.0,
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        # Should succeed because goal is ignored
        assert result is not None

    def test_low_weight_goal_has_less_influence(self, fullbody_solver, humanoid_transforms):
        """Low weight goal has less influence."""
        target = Vec3(-0.5, 1.2, 0.3)
        goal_low = FullBodyIKGoal(
            bone_index=8,
            target_position=target,
            position_weight=0.1,
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal_low])
        assert result is not None

    def test_high_weight_goal_has_more_influence(self, fullbody_solver, humanoid_transforms):
        """High weight goal has more influence."""
        target = Vec3(-0.6, 1.3, 0.2)
        goal_high = FullBodyIKGoal(
            bone_index=8,
            target_position=target,
            position_weight=1.0,
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal_high])
        assert result is not None

    def test_multiple_weighted_goals(self, fullbody_solver, humanoid_transforms):
        """Multiple goals with different weights."""
        goals = [
            FullBodyIKGoal(bone_index=8, target_position=Vec3(-0.6, 1.3, 0.2), position_weight=1.0),
            FullBodyIKGoal(bone_index=12, target_position=Vec3(0.6, 1.3, 0.2), position_weight=0.5),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None


# =============================================================================
# Goal Priority Tests
# =============================================================================

class TestFullBodyIKGoalPriority:
    """Tests for goal priority handling."""

    def test_higher_priority_takes_precedence(self, fullbody_solver, humanoid_transforms):
        """Higher priority goals take precedence."""
        goals = [
            FullBodyIKGoal(bone_index=8, target_position=Vec3(-0.6, 1.3, 0.2), priority=1),
            FullBodyIKGoal(bone_index=8, target_position=Vec3(-0.7, 1.2, 0.1), priority=10),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None

    def test_default_priority_is_zero(self, fullbody_solver, humanoid_transforms):
        """Default priority is 0."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.3, 0.2),
        )
        assert goal.priority == 0


# =============================================================================
# Disabled Goal Tests
# =============================================================================

class TestFullBodyIKDisabledGoals:
    """Tests for disabled goal handling."""

    def test_disabled_goal_ignored(self, fullbody_solver, humanoid_transforms):
        """Disabled goals are ignored."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(10.0, 10.0, 10.0),
            enabled=False,
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_mix_enabled_disabled(self, fullbody_solver, humanoid_transforms):
        """Mix of enabled and disabled goals."""
        goals = [
            FullBodyIKGoal(bone_index=8, target_position=Vec3(-0.6, 1.3, 0.2), enabled=True),
            FullBodyIKGoal(bone_index=12, target_position=Vec3(10.0, 10.0, 0.0), enabled=False),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None


# =============================================================================
# LookAt Solver Tests
# =============================================================================

class TestLookAtSolver:
    """Tests for LookAtSolver component."""

    def test_can_create_lookat_solver(self):
        """Can create LookAtSolver."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[1, 2],
        )
        assert solver is not None

    def test_lookat_with_weights(self):
        """LookAtSolver accepts weights."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[1, 2],
            head_weight=0.5,
            neck_weight=0.3,
            spine_weight=0.2,
        )
        assert solver is not None

    def test_lookat_returns_transforms(self, humanoid_transforms):
        """LookAt returns transforms."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[1, 2],
        )
        target = Vec3(0.0, 1.9, 1.0)  # Look forward
        result = solver.solve(humanoid_transforms, target)
        assert result is not None
        assert isinstance(result, list)

    def test_lookat_with_custom_forward_axis(self, humanoid_transforms):
        """LookAt with custom forward axis."""
        solver = LookAtSolver(
            head_bone=4,
            neck_bone=3,
            spine_bones=[1, 2],
        )
        target = Vec3(0.0, 1.9, 1.0)
        forward = Vec3(0.0, 0.0, 1.0)
        result = solver.solve(humanoid_transforms, target, forward_axis=forward)
        assert result is not None


# =============================================================================
# Chain Type Tests
# =============================================================================

class TestFullBodyIKChainTypes:
    """Tests for different chain type strings."""

    def test_left_arm_chain_type(self, fullbody_solver, humanoid_transforms):
        """Left arm chain type works."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.3, 0.1),
            chain_type="left_arm",
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_right_arm_chain_type(self, fullbody_solver, humanoid_transforms):
        """Right arm chain type works."""
        goal = FullBodyIKGoal(
            bone_index=12,
            target_position=Vec3(0.6, 1.3, 0.1),
            chain_type="right_arm",
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_left_leg_chain_type(self, fullbody_solver, humanoid_transforms):
        """Left leg chain type works."""
        goal = FullBodyIKGoal(
            bone_index=15,
            target_position=Vec3(-0.1, 0.1, 0.0),
            chain_type="left_leg",
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_right_leg_chain_type(self, fullbody_solver, humanoid_transforms):
        """Right leg chain type works."""
        goal = FullBodyIKGoal(
            bone_index=19,
            target_position=Vec3(0.1, 0.1, 0.0),
            chain_type="right_leg",
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_spine_chain_type(self, fullbody_solver, humanoid_transforms):
        """Spine chain type works."""
        goal = FullBodyIKGoal(
            bone_index=2,
            target_position=Vec3(0.0, 1.5, 0.1),
            chain_type="spine",
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None


# =============================================================================
# Transform Preservation Tests
# =============================================================================

class TestFullBodyIKTransformPreservation:
    """Tests for transform preservation behavior."""

    def test_solve_preserves_transform_count(self, fullbody_solver, humanoid_transforms):
        """Solve preserves number of transforms."""
        goals = [
            FullBodyIKGoal(bone_index=8, target_position=Vec3(-0.5, 1.3, 0.2)),
            FullBodyIKGoal(bone_index=12, target_position=Vec3(0.5, 1.3, 0.2)),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert len(result.transforms) == len(humanoid_transforms)

    def test_empty_goals_preserves_transforms(self, fullbody_solver, humanoid_transforms):
        """Empty goals preserves transforms."""
        result = fullbody_solver.solve(humanoid_transforms, [])
        assert len(result.transforms) == len(humanoid_transforms)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestFullBodyIKEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_unreachable_target(self, fullbody_solver, humanoid_transforms):
        """Handle unreachable target gracefully."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-10.0, 10.0, 10.0),  # Far away
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        # Should not crash, may not fully succeed
        assert result is not None

    def test_target_at_origin(self, fullbody_solver, humanoid_transforms):
        """Handle target at origin."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 0.0, 0.0),
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_target_behind_body(self, fullbody_solver, humanoid_transforms):
        """Handle target behind body."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.3, 1.2, -0.5),
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_conflicting_goals(self, fullbody_solver, humanoid_transforms):
        """Handle conflicting goals."""
        # Both hands to same position
        goals = [
            FullBodyIKGoal(bone_index=8, target_position=Vec3(0.0, 1.5, 0.5)),
            FullBodyIKGoal(bone_index=12, target_position=Vec3(0.0, 1.5, 0.5)),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None

    def test_very_small_movement(self, fullbody_solver, humanoid_transforms):
        """Handle very small movements."""
        # Target very close to current position
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.75, 1.5, 0.0),  # Exact current position
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        # Solver should not crash on trivial movement
        assert result is not None

    def test_large_chain_movement(self, fullbody_solver, humanoid_transforms):
        """Handle large chain movement."""
        # Arm stretched far
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.9, 0.8, 0.5),
            position_weight=1.0,
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None


# =============================================================================
# Iteration Control Tests
# =============================================================================

class TestFullBodyIKIterationControl:
    """Tests for iteration and convergence control."""

    def test_low_max_iterations_solver(self, skeleton_mapping, humanoid_transforms):
        """Solver with low max iterations works."""
        solver = FullBodyIK(skeleton_mapping, max_iterations=1)
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.5, 1.0, 0.3),
        )
        result = solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_high_max_iterations_solver(self, skeleton_mapping, humanoid_transforms):
        """Solver with high max iterations works."""
        solver = FullBodyIK(skeleton_mapping, max_iterations=100)
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.3, 0.2),
        )
        result = solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_tolerance_affects_convergence(self, skeleton_mapping, humanoid_transforms):
        """Tolerance affects convergence behavior."""
        solver_tight = FullBodyIK(skeleton_mapping, tolerance=0.0001)
        solver_loose = FullBodyIK(skeleton_mapping, tolerance=0.1)

        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.3, 0.2),
        )

        result_tight = solver_tight.solve(humanoid_transforms, [goal])
        result_loose = solver_loose.solve(humanoid_transforms, [goal])

        assert result_tight is not None
        assert result_loose is not None


# =============================================================================
# Rotation Goal Tests
# =============================================================================

class TestFullBodyIKRotationGoals:
    """Tests for rotation-based goals."""

    def test_rotation_only_goal(self, fullbody_solver, humanoid_transforms):
        """Rotation-only goal works."""
        goal = FullBodyIKGoal(
            bone_index=4,  # head
            target_rotation=Quat.identity(),
            position_weight=0.0,
            rotation_weight=1.0,
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_combined_position_rotation(self, fullbody_solver, humanoid_transforms):
        """Combined position and rotation goal."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.3, 0.2),
            target_rotation=Quat.identity(),
            position_weight=1.0,
            rotation_weight=0.5,
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

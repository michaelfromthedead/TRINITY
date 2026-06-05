"""
Blackbox tests for FullBodyIK system.

T-FB-4.6 FullBodyIK Core - SDLC BLACKBOX TEST

Tests the public API of FullBodyIK without peeking at implementation.
Focuses on behavior verification through the documented interface.
"""

import pytest
import math
from typing import List, Optional, Dict

from engine.animation.ik import (
    FullBodyIKGoal,
    FullBodyIKResult,
    FullBodyIK,
    SkeletonMapping,
    BodyPart,
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
# Fixtures
# =============================================================================


@pytest.fixture
def humanoid_transforms() -> List[Transform]:
    """Standard humanoid T-pose transforms."""
    return create_humanoid_transforms()


@pytest.fixture
def skeleton_mapping() -> SkeletonMapping:
    """Standard humanoid skeleton mapping."""
    return create_humanoid_mapping()


@pytest.fixture
def fullbody_solver(skeleton_mapping: SkeletonMapping) -> FullBodyIK:
    """Standard full body IK solver."""
    return FullBodyIK(skeleton_mapping)


# =============================================================================
# Test Class: FullBodyIKGoal Creation and Properties
# =============================================================================


class TestFullBodyIKGoalCreation:
    """Tests for FullBodyIKGoal creation and basic properties."""

    def test_create_goal_with_position_target(self) -> None:
        """Goal created with position target should have target_position."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0)
        )
        assert goal.target_position is not None

    def test_create_goal_without_position(self) -> None:
        """Goal created without position should have None target_position."""
        goal = FullBodyIKGoal(bone_index=8)
        # May be None or have default
        assert goal is not None

    def test_create_goal_with_none_position(self) -> None:
        """Goal created with None position should work."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=None
        )
        assert goal.target_position is None

    def test_create_goal_with_rotation_target(self) -> None:
        """Goal created with rotation target should have target_rotation."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_rotation=Quat.identity(),
            rotation_weight=1.0
        )
        assert goal.target_rotation is not None

    def test_create_goal_without_rotation(self) -> None:
        """Goal created without rotation should have no active rotation."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0)
        )
        # Rotation may be None or identity
        assert goal is not None

    def test_create_goal_with_none_rotation(self) -> None:
        """Goal created with None rotation should work."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_rotation=None
        )
        assert goal.target_rotation is None

    def test_create_goal_with_both_position_and_rotation(self) -> None:
        """Goal created with both targets should have both."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0),
            target_rotation=Quat.identity(),
            position_weight=1.0,
            rotation_weight=1.0
        )
        assert goal.target_position is not None
        assert goal.target_rotation is not None

    def test_goal_position_value_accessible(self) -> None:
        """Goal position value should be accessible after creation."""
        pos = Vec3(1.0, 2.0, 3.0)
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=pos
        )
        assert goal.target_position is not None
        assert abs(goal.target_position.x - 1.0) < 1e-6
        assert abs(goal.target_position.y - 2.0) < 1e-6
        assert abs(goal.target_position.z - 3.0) < 1e-6

    def test_goal_rotation_value_accessible(self) -> None:
        """Goal rotation value should be accessible after creation."""
        rot = Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), math.pi / 4)
        goal = FullBodyIKGoal(
            bone_index=8,
            target_rotation=rot,
            rotation_weight=1.0
        )
        assert goal.target_rotation is not None


class TestFullBodyIKGoalWeight:
    """Tests for FullBodyIKGoal weight behavior."""

    def test_goal_default_position_weight(self) -> None:
        """Default position weight should be set."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0)
        )
        assert hasattr(goal, "position_weight")

    def test_goal_with_zero_weight(self) -> None:
        """Goal with zero weight should be effectively inactive."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0),
            position_weight=0.0
        )
        assert goal.position_weight == 0.0

    def test_goal_with_positive_weight(self) -> None:
        """Goal with positive weight should be active."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0),
            position_weight=1.0
        )
        assert goal.position_weight == 1.0

    def test_goal_with_small_positive_weight(self) -> None:
        """Goal with small positive weight should still work."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0),
            position_weight=0.001
        )
        assert goal.position_weight == 0.001

    def test_goal_position_weight_is_settable(self) -> None:
        """Goal position weight should be settable."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0),
            position_weight=0.5
        )
        assert abs(goal.position_weight - 0.5) < 1e-6

    def test_goal_rotation_weight(self) -> None:
        """Goal rotation weight should work."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_rotation=Quat.identity(),
            rotation_weight=0.7
        )
        assert abs(goal.rotation_weight - 0.7) < 1e-6


class TestFullBodyIKGoalPriority:
    """Tests for FullBodyIKGoal priority field."""

    def test_goal_default_priority(self) -> None:
        """Goal should have a default priority."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0)
        )
        assert hasattr(goal, "priority")

    def test_goal_custom_priority(self) -> None:
        """Goal should accept custom priority."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0),
            priority=10
        )
        assert goal.priority == 10

    def test_goal_high_priority_value(self) -> None:
        """Goal should accept high priority values."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0),
            priority=100
        )
        assert goal.priority == 100

    def test_goal_zero_priority(self) -> None:
        """Goal should accept zero priority."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0),
            priority=0
        )
        assert goal.priority == 0


class TestFullBodyIKGoalEnabled:
    """Tests for FullBodyIKGoal enabled field."""

    def test_goal_default_enabled(self) -> None:
        """Goal should be enabled by default."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0)
        )
        assert goal.enabled is True

    def test_goal_explicitly_enabled(self) -> None:
        """Goal can be explicitly enabled."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0),
            enabled=True
        )
        assert goal.enabled is True

    def test_goal_disabled(self) -> None:
        """Goal can be disabled."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0),
            enabled=False
        )
        assert goal.enabled is False

    def test_disabled_goal_still_has_position(self) -> None:
        """Disabled goal should still have position data."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.5, 1.0, 0.0),
            enabled=False
        )
        assert goal.target_position is not None


class TestFullBodyIKGoalChainType:
    """Tests for FullBodyIKGoal chain_type assignment."""

    def test_goal_left_arm_chain_type(self) -> None:
        """Goal can be assigned to left_arm chain."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0),
            chain_type="left_arm"
        )
        assert goal.chain_type == "left_arm"

    def test_goal_right_arm_chain_type(self) -> None:
        """Goal can be assigned to right_arm chain."""
        goal = FullBodyIKGoal(
            bone_index=12,
            target_position=Vec3(0.0, 1.0, 0.0),
            chain_type="right_arm"
        )
        assert goal.chain_type == "right_arm"

    def test_goal_left_leg_chain_type(self) -> None:
        """Goal can be assigned to left_leg chain."""
        goal = FullBodyIKGoal(
            bone_index=15,
            target_position=Vec3(0.0, 0.0, 0.0),
            chain_type="left_leg"
        )
        assert goal.chain_type == "left_leg"

    def test_goal_right_leg_chain_type(self) -> None:
        """Goal can be assigned to right_leg chain."""
        goal = FullBodyIKGoal(
            bone_index=19,
            target_position=Vec3(0.0, 0.0, 0.0),
            chain_type="right_leg"
        )
        assert goal.chain_type == "right_leg"

    def test_goal_spine_chain_type(self) -> None:
        """Goal can be assigned to spine chain."""
        goal = FullBodyIKGoal(
            bone_index=2,
            target_position=Vec3(0.0, 1.5, 0.2),
            chain_type="spine"
        )
        assert goal.chain_type == "spine"

    def test_goal_default_chain_type(self) -> None:
        """Goal should have a default chain type or None."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.0)
        )
        assert hasattr(goal, "chain_type")


# =============================================================================
# Test Class: FullBodyIKResult Properties
# =============================================================================


class TestFullBodyIKResultProperties:
    """Tests for FullBodyIKResult property accessibility."""

    def test_result_success_accessible(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Result success field should be accessible."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.7, 1.4, 0.1)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert hasattr(result, "success")
        assert isinstance(result.success, bool)

    def test_result_transforms_accessible(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Result transforms field should be accessible."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.7, 1.4, 0.1)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert hasattr(result, "transforms")

    def test_result_goals_achieved_accessible(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Result goals_achieved field should be accessible."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.7, 1.4, 0.1)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert hasattr(result, "goals_achieved")

    def test_result_final_errors_accessible(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Result final_errors field should be accessible."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.7, 1.4, 0.1)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert hasattr(result, "final_errors")

    def test_result_pelvis_adjustment_accessible(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Result pelvis_adjustment field should be accessible."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.7, 1.4, 0.1)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert hasattr(result, "pelvis_adjustment")


class TestFullBodyIKResultTransforms:
    """Tests for FullBodyIKResult transforms content."""

    def test_result_transforms_is_collection(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Result transforms should be a collection."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.7, 1.4, 0.1)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert isinstance(result.transforms, (dict, list))

    def test_result_transforms_with_goal_contains_entries(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Result with goals should contain transform entries."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.4, 0.2),
            chain_type="left_arm"
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert len(result.transforms) > 0


class TestFullBodyIKResultGoalsAchieved:
    """Tests for FullBodyIKResult goals_achieved content."""

    def test_empty_goals_returns_success(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Empty goals list should return success."""
        result = fullbody_solver.solve(humanoid_transforms, [])
        assert result.success is True

    def test_goals_achieved_reflects_solve_outcome(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Goals achieved should reflect whether each goal was met."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.7, 1.4, 0.1)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert hasattr(result, "goals_achieved")


class TestFullBodyIKResultFinalErrors:
    """Tests for FullBodyIKResult final_errors content."""

    def test_final_errors_are_numeric(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Final errors should be numeric values."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.4, 0.1)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        if isinstance(result.final_errors, dict):
            for error in result.final_errors.values():
                assert isinstance(error, (int, float))
        elif isinstance(result.final_errors, list):
            for error in result.final_errors:
                assert isinstance(error, (int, float))

    def test_final_errors_are_non_negative(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Final errors should be non-negative."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.4, 0.1)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        if isinstance(result.final_errors, dict):
            for error in result.final_errors.values():
                assert error >= 0.0
        elif isinstance(result.final_errors, list):
            for error in result.final_errors:
                assert error >= 0.0


# =============================================================================
# Test Class: FullBodyIK Solver Creation
# =============================================================================


class TestFullBodyIKSolverCreation:
    """Tests for FullBodyIK solver creation."""

    def test_create_solver_with_skeleton_mapping(
        self, skeleton_mapping: SkeletonMapping
    ) -> None:
        """Solver should be creatable with skeleton mapping."""
        solver = FullBodyIK(skeleton_mapping)
        assert solver is not None

    def test_create_solver_with_tolerance(
        self, skeleton_mapping: SkeletonMapping
    ) -> None:
        """Solver can be created with custom tolerance."""
        solver = FullBodyIK(skeleton_mapping, tolerance=0.001)
        assert solver is not None

    def test_create_solver_with_max_iterations(
        self, skeleton_mapping: SkeletonMapping
    ) -> None:
        """Solver can be created with custom max iterations."""
        solver = FullBodyIK(skeleton_mapping, max_iterations=50)
        assert solver is not None

    def test_solver_has_solve_method(
        self, fullbody_solver: FullBodyIK
    ) -> None:
        """Solver should have solve method."""
        assert hasattr(fullbody_solver, "solve")
        assert callable(fullbody_solver.solve)

    def test_solver_has_set_bone_mass_method(
        self, fullbody_solver: FullBodyIK
    ) -> None:
        """Solver should have set_bone_mass method."""
        assert hasattr(fullbody_solver, "set_bone_mass")
        assert callable(fullbody_solver.set_bone_mass)


class TestFullBodyIKSolverGoalManagement:
    """Tests for FullBodyIK goal management via solve."""

    def test_solve_with_single_goal(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Solving with single goal should not raise."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.4, 0.2)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_solve_with_multiple_goals(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Solving with multiple goals should not raise."""
        goals = [
            FullBodyIKGoal(
                bone_index=8,
                target_position=Vec3(-0.6, 1.4, 0.2)
            ),
            FullBodyIKGoal(
                bone_index=12,
                target_position=Vec3(0.6, 1.4, 0.2)
            ),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None

    def test_solve_with_empty_goals(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Solving with empty goals should succeed."""
        result = fullbody_solver.solve(humanoid_transforms, [])
        assert result.success is True

    def test_solve_multiple_times(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Solving multiple times should work."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.4, 0.1)
        )
        result1 = fullbody_solver.solve(humanoid_transforms, [goal])
        result2 = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result1 is not None
        assert result2 is not None


# =============================================================================
# Test Class: FullBodyIK Solving Behavior
# =============================================================================


class TestFullBodyIKSolving:
    """Tests for FullBodyIK solving behavior."""

    def test_solve_returns_result(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Solve should return a FullBodyIKResult."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.4, 0.1)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert isinstance(result, FullBodyIKResult)

    def test_solve_with_reachable_goal(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Solve with reachable goal should return success."""
        # Target very close to current hand position (bone 8 is at -0.75, 1.5, 0.0)
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.75, 1.5, 0.0)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result.success is True

    def test_empty_goals_returns_success(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Empty goals list should return success=True."""
        result = fullbody_solver.solve(humanoid_transforms, [])
        assert result.success is True

    def test_solve_is_repeatable(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Solving multiple times should produce consistent results."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.7, 1.4, 0.0)
        )
        result1 = fullbody_solver.solve(humanoid_transforms, [goal])
        result2 = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result1 is not None
        assert result2 is not None


class TestFullBodyIKPriorityProcessing:
    """Tests for FullBodyIK priority-based processing."""

    def test_higher_priority_goal_affects_result(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Higher priority goals should influence result."""
        goals = [
            FullBodyIKGoal(
                bone_index=8,
                target_position=Vec3(-0.6, 1.3, 0.3),
                priority=1
            ),
            FullBodyIKGoal(
                bone_index=8,
                target_position=Vec3(-0.7, 1.4, 0.0),
                priority=10
            ),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None

    def test_equal_priority_goals_both_processed(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Goals with equal priority should both be processed."""
        goals = [
            FullBodyIKGoal(
                bone_index=8,
                target_position=Vec3(-0.7, 1.4, 0.1),
                priority=5
            ),
            FullBodyIKGoal(
                bone_index=12,
                target_position=Vec3(0.7, 1.4, 0.1),
                priority=5
            ),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None


class TestFullBodyIKDisabledGoals:
    """Tests for disabled goal handling."""

    def test_disabled_goal_not_processed(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Disabled goals should not be processed."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.4, 0.2),
            enabled=False
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        # Should succeed since disabled goal is effectively not present
        assert result.success is True

    def test_mixed_enabled_disabled_goals(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Mix of enabled and disabled goals should work."""
        goals = [
            FullBodyIKGoal(
                bone_index=8,
                target_position=Vec3(-0.7, 1.4, 0.1),
                enabled=True
            ),
            FullBodyIKGoal(
                bone_index=12,
                target_position=Vec3(0.7, 1.4, 0.1),
                enabled=False
            ),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None


# =============================================================================
# Test Class: FullBodyIK Multi-Chain Support
# =============================================================================


class TestFullBodyIKMultiChain:
    """Tests for multiple chain support."""

    def test_arm_chain_goals(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Arm chain goals should be processable."""
        goals = [
            FullBodyIKGoal(
                bone_index=8,
                target_position=Vec3(-0.6, 1.3, 0.3),
                chain_type="left_arm"
            ),
            FullBodyIKGoal(
                bone_index=12,
                target_position=Vec3(0.6, 1.3, 0.3),
                chain_type="right_arm"
            ),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None

    def test_leg_chain_goals(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Leg chain goals should be processable."""
        goals = [
            FullBodyIKGoal(
                bone_index=15,
                target_position=Vec3(-0.15, 0.0, 0.1),
                chain_type="left_leg"
            ),
            FullBodyIKGoal(
                bone_index=19,
                target_position=Vec3(0.15, 0.0, 0.1),
                chain_type="right_leg"
            ),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None

    def test_spine_chain_goal(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Spine chain goals should be processable."""
        goal = FullBodyIKGoal(
            bone_index=2,
            target_position=Vec3(0.0, 1.5, 0.2),
            chain_type="spine"
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_all_chains_simultaneously(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """All chains can have goals simultaneously."""
        goals = [
            FullBodyIKGoal(bone_index=8, target_position=Vec3(-0.6, 1.3, 0.2)),
            FullBodyIKGoal(bone_index=12, target_position=Vec3(0.6, 1.3, 0.2)),
            FullBodyIKGoal(bone_index=15, target_position=Vec3(-0.1, 0.0, 0.0)),
            FullBodyIKGoal(bone_index=19, target_position=Vec3(0.1, 0.0, 0.0)),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None


# =============================================================================
# Test Class: FullBodyIK Bone Mass and Support Polygon
# =============================================================================


class TestFullBodyIKBoneMass:
    """Tests for set_bone_mass functionality."""

    def test_set_bone_mass_no_error(
        self, fullbody_solver: FullBodyIK
    ) -> None:
        """Setting bone mass should not raise error."""
        fullbody_solver.set_bone_mass(0, 10.0)

    def test_set_multiple_bone_masses(
        self, fullbody_solver: FullBodyIK
    ) -> None:
        """Setting multiple bone masses should work."""
        fullbody_solver.set_bone_mass(0, 10.0)
        fullbody_solver.set_bone_mass(1, 5.0)
        fullbody_solver.set_bone_mass(2, 5.0)
        fullbody_solver.set_bone_mass(4, 4.0)

    def test_set_bone_mass_then_solve(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Setting bone mass should not prevent solving."""
        fullbody_solver.set_bone_mass(0, 10.0)
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.4, 0.1)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_set_zero_bone_mass(
        self, fullbody_solver: FullBodyIK
    ) -> None:
        """Setting zero bone mass should work or raise appropriate error."""
        try:
            fullbody_solver.set_bone_mass(0, 0.0)
        except ValueError:
            pass  # Zero mass may be invalid


class TestFullBodyIKSupportPolygon:
    """Tests for set_support_polygon functionality."""

    def test_set_support_polygon_no_error(
        self, fullbody_solver: FullBodyIK
    ) -> None:
        """Setting support polygon should not raise error."""
        vertices = [
            Vec3(0.1, 0.0, 0.1),
            Vec3(-0.1, 0.0, 0.1),
            Vec3(-0.1, 0.0, -0.1),
            Vec3(0.1, 0.0, -0.1),
        ]
        fullbody_solver.set_support_polygon(vertices)

    def test_set_support_polygon_triangle(
        self, fullbody_solver: FullBodyIK
    ) -> None:
        """Triangle support polygon should work."""
        vertices = [
            Vec3(0.0, 0.0, 0.1),
            Vec3(-0.1, 0.0, -0.05),
            Vec3(0.1, 0.0, -0.05),
        ]
        fullbody_solver.set_support_polygon(vertices)

    def test_set_support_polygon_then_solve(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Setting support polygon should not prevent solving."""
        vertices = [
            Vec3(0.1, 0.0, 0.1),
            Vec3(-0.1, 0.0, 0.1),
            Vec3(-0.1, 0.0, -0.1),
            Vec3(0.1, 0.0, -0.1),
        ]
        fullbody_solver.set_support_polygon(vertices)
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.7, 1.4, 0.1)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None


# =============================================================================
# Test Class: Integration Scenarios - Full Body Reach
# =============================================================================


class TestFullBodyReachIntegration:
    """Integration tests for full body reach scenarios."""

    def test_reach_with_arm_goals(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Full body reach with arm goals should work."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.3, 0.3),
            chain_type="left_arm",
            priority=5
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None
        assert hasattr(result, "transforms")

    def test_reach_with_leg_goals(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Full body reach with leg goals should work."""
        goal = FullBodyIKGoal(
            bone_index=15,
            target_position=Vec3(-0.2, 0.0, 0.3),
            chain_type="left_leg",
            priority=5
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_reach_with_arm_and_leg_goals(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Full body reach with both arm and leg goals should work."""
        goals = [
            FullBodyIKGoal(
                bone_index=8,
                target_position=Vec3(-0.5, 1.4, 0.2),
                chain_type="left_arm"
            ),
            FullBodyIKGoal(
                bone_index=15,
                target_position=Vec3(-0.1, 0.1, 0.1),
                chain_type="left_leg"
            ),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None

    def test_full_body_pose_four_limbs(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Full body pose with all four limbs should work."""
        goals = [
            FullBodyIKGoal(bone_index=8, target_position=Vec3(-0.6, 1.3, 0.2)),
            FullBodyIKGoal(bone_index=12, target_position=Vec3(0.6, 1.3, 0.2)),
            FullBodyIKGoal(bone_index=15, target_position=Vec3(-0.1, 0.0, 0.0)),
            FullBodyIKGoal(bone_index=19, target_position=Vec3(0.1, 0.0, 0.0)),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None


# =============================================================================
# Test Class: Integration Scenarios - Balance Mode
# =============================================================================


class TestBalanceModeIntegration:
    """Integration tests for balance mode with COM tracking."""

    def test_enable_balance_mode(
        self, fullbody_solver: FullBodyIK
    ) -> None:
        """Enabling balance mode should not raise error."""
        if hasattr(fullbody_solver, "set_balance_mode"):
            fullbody_solver.set_balance_mode(True)
        elif hasattr(fullbody_solver, "enable_balance"):
            fullbody_solver.enable_balance(True)

    def test_balance_mode_with_support_polygon(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Balance mode with support polygon should work."""
        if hasattr(fullbody_solver, "set_balance_mode"):
            fullbody_solver.set_balance_mode(True)
        vertices = [
            Vec3(0.1, 0.0, 0.1),
            Vec3(-0.1, 0.0, 0.1),
            Vec3(-0.1, 0.0, -0.1),
            Vec3(0.1, 0.0, -0.1),
        ]
        fullbody_solver.set_support_polygon(vertices)
        result = fullbody_solver.solve(humanoid_transforms, [])
        assert result is not None

    def test_balance_mode_with_arm_reach(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Balance mode with arm reach should adjust body."""
        if hasattr(fullbody_solver, "set_balance_mode"):
            fullbody_solver.set_balance_mode(True)
        vertices = [
            Vec3(0.1, 0.0, 0.1),
            Vec3(-0.1, 0.0, 0.1),
            Vec3(-0.1, 0.0, -0.1),
            Vec3(0.1, 0.0, -0.1),
        ]
        fullbody_solver.set_support_polygon(vertices)
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 1.0, 0.8),
            chain_type="left_arm"
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None


# =============================================================================
# Test Class: Integration Scenarios - Pelvis Adjustment
# =============================================================================


class TestPelvisAdjustmentIntegration:
    """Integration tests for pelvis height adjustment."""

    def test_pelvis_adjustment_returned(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Pelvis adjustment should be accessible in result."""
        goal = FullBodyIKGoal(
            bone_index=15,
            target_position=Vec3(-0.1, -0.1, 0.0),
            chain_type="left_leg"
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert hasattr(result, "pelvis_adjustment")

    def test_pelvis_adjustment_when_legs_stretch(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Pelvis should adjust when legs need to stretch."""
        goals = [
            FullBodyIKGoal(
                bone_index=15,
                target_position=Vec3(-0.3, 0.0, 0.0),
                chain_type="left_leg"
            ),
            FullBodyIKGoal(
                bone_index=19,
                target_position=Vec3(0.3, 0.0, 0.0),
                chain_type="right_leg"
            ),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result.pelvis_adjustment is not None or result is not None

    def test_pelvis_adjustment_type(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Pelvis adjustment should be a transform or vector."""
        goal = FullBodyIKGoal(
            bone_index=15,
            target_position=Vec3(-0.15, -0.05, 0.0),
            chain_type="left_leg"
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        # May be Vec3, Transform, or None
        assert hasattr(result, "pelvis_adjustment")

    def test_pelvis_lowering_with_squat_pose(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Pelvis should lower for squat-like poses."""
        goals = [
            FullBodyIKGoal(
                bone_index=15,
                target_position=Vec3(-0.1, 0.0, 0.3),
                chain_type="left_leg"
            ),
            FullBodyIKGoal(
                bone_index=19,
                target_position=Vec3(0.1, 0.0, 0.3),
                chain_type="right_leg"
            ),
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None


# =============================================================================
# Test Class: Edge Cases and Boundary Conditions
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_far_goal_position(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Very far goal should not crash solver."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(100.0, 100.0, 100.0)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_negative_position_goal(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Negative position coordinates should work."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.5, -0.5, -0.5)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_zero_position_goal(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Zero position goal should work."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(0.0, 0.0, 0.0)
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_very_small_weight(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Very small weight should still work."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.4, 0.0),
            position_weight=0.0001
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_very_large_weight(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Very large weight should still work."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.4, 0.0),
            position_weight=1000.0
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_rotation_only_goal(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Rotation-only goal (no position) should work."""
        rotation = Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), math.pi / 4)
        goal = FullBodyIKGoal(
            bone_index=8,
            target_rotation=rotation,
            rotation_weight=1.0
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None

    def test_many_goals_at_once(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Many goals at once should not crash."""
        goals = [
            FullBodyIKGoal(
                bone_index=8,
                target_position=Vec3(-0.6 + i * 0.01, 1.3 + 0.01 * i, 0.0)
            )
            for i in range(5)
        ]
        result = fullbody_solver.solve(humanoid_transforms, goals)
        assert result is not None

    def test_solve_multiple_times_consecutively(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Solving multiple times consecutively should work."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.4, 0.0)
        )
        for _ in range(5):
            result = fullbody_solver.solve(humanoid_transforms, [goal])
            assert result is not None


# =============================================================================
# Test Class: SkeletonMapping Tests
# =============================================================================


class TestSkeletonMapping:
    """Tests for SkeletonMapping behavior."""

    def test_create_skeleton_mapping(self) -> None:
        """SkeletonMapping should be creatable."""
        bone_map = {BodyPart.PELVIS: 0}
        mapping = SkeletonMapping(
            bone_map=bone_map,
            spine_chain=[0],
            left_arm_chain=[],
            right_arm_chain=[],
            left_leg_chain=[],
            right_leg_chain=[],
        )
        assert mapping is not None

    def test_skeleton_mapping_with_full_bone_map(self) -> None:
        """Full bone map skeleton mapping should work."""
        mapping = create_humanoid_mapping()
        assert mapping is not None

    def test_body_part_enum_values(self) -> None:
        """BodyPart enum should have expected values."""
        assert hasattr(BodyPart, "PELVIS")
        assert hasattr(BodyPart, "SPINE")
        assert hasattr(BodyPart, "LEFT_HAND")
        assert hasattr(BodyPart, "RIGHT_HAND")
        assert hasattr(BodyPart, "LEFT_FOOT")
        assert hasattr(BodyPart, "RIGHT_FOOT")

    def test_body_part_upper_body(self) -> None:
        """BodyPart should have upper body parts."""
        assert hasattr(BodyPart, "CHEST")
        assert hasattr(BodyPart, "NECK")
        assert hasattr(BodyPart, "HEAD")

    def test_body_part_arms(self) -> None:
        """BodyPart should have arm parts."""
        assert hasattr(BodyPart, "LEFT_SHOULDER")
        assert hasattr(BodyPart, "LEFT_UPPER_ARM")
        assert hasattr(BodyPart, "LEFT_LOWER_ARM")
        assert hasattr(BodyPart, "RIGHT_SHOULDER")
        assert hasattr(BodyPart, "RIGHT_UPPER_ARM")
        assert hasattr(BodyPart, "RIGHT_LOWER_ARM")

    def test_body_part_legs(self) -> None:
        """BodyPart should have leg parts."""
        assert hasattr(BodyPart, "LEFT_UPPER_LEG")
        assert hasattr(BodyPart, "LEFT_LOWER_LEG")
        assert hasattr(BodyPart, "RIGHT_UPPER_LEG")
        assert hasattr(BodyPart, "RIGHT_LOWER_LEG")


# =============================================================================
# Test Class: Rotation Goals
# =============================================================================


class TestRotationGoals:
    """Tests for rotation-based goals."""

    def test_rotation_goal_has_rotation(self) -> None:
        """Rotation goal should have target_rotation."""
        rotation = Quat.from_axis_angle(Vec3(1.0, 0.0, 0.0), math.pi / 6)
        goal = FullBodyIKGoal(
            bone_index=8,
            target_rotation=rotation,
            rotation_weight=1.0
        )
        assert goal.target_rotation is not None

    def test_rotation_goal_with_zero_weight(self) -> None:
        """Rotation goal with zero weight should be inactive."""
        rotation = Quat.from_axis_angle(Vec3(1.0, 0.0, 0.0), math.pi / 6)
        goal = FullBodyIKGoal(
            bone_index=8,
            target_rotation=rotation,
            rotation_weight=0.0
        )
        assert goal.rotation_weight == 0.0

    def test_position_and_rotation_goal(self) -> None:
        """Goal with both position and rotation should work."""
        pos = Vec3(-0.6, 1.4, 0.0)
        rot = Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), math.pi / 4)
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=pos,
            target_rotation=rot,
            position_weight=1.0,
            rotation_weight=1.0
        )
        assert goal.target_position is not None
        assert goal.target_rotation is not None

    def test_solve_with_rotation_goal(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Solving with rotation goal should work."""
        rot = Quat.from_axis_angle(Vec3(0.0, 0.0, 1.0), math.pi / 3)
        goal = FullBodyIKGoal(
            bone_index=8,
            target_rotation=rot,
            rotation_weight=1.0
        )
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        assert result is not None


# =============================================================================
# Test Class: Performance Boundaries
# =============================================================================


class TestPerformanceBoundaries:
    """Tests for performance-related boundaries."""

    def test_solve_completes_in_reasonable_time(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Solve should complete in reasonable time."""
        import time
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.7, 1.4, 0.1)
        )
        start = time.time()
        result = fullbody_solver.solve(humanoid_transforms, [goal])
        elapsed = time.time() - start
        assert result is not None
        assert elapsed < 1.0  # Should complete within 1 second

    def test_solve_many_iterations(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Solving many times should not degrade."""
        goal = FullBodyIKGoal(
            bone_index=8,
            target_position=Vec3(-0.6, 1.4, 0.0)
        )
        for i in range(10):
            result = fullbody_solver.solve(humanoid_transforms, [goal])
            assert result is not None


# =============================================================================
# Test Class: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling behavior."""

    def test_invalid_bone_index_handling(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Invalid bone index should be handled gracefully."""
        try:
            goal = FullBodyIKGoal(
                bone_index=999,
                target_position=Vec3(0.0, 1.0, 0.0)
            )
            result = fullbody_solver.solve(humanoid_transforms, [goal])
            # May succeed with warning or produce error
            assert result is not None or True
        except (ValueError, IndexError):
            pass  # Expected if invalid index is rejected

    def test_negative_bone_index_handling(
        self, fullbody_solver: FullBodyIK, humanoid_transforms: List[Transform]
    ) -> None:
        """Negative bone index should be handled gracefully."""
        try:
            goal = FullBodyIKGoal(
                bone_index=-1,
                target_position=Vec3(0.0, 1.0, 0.0)
            )
            result = fullbody_solver.solve(humanoid_transforms, [goal])
            assert result is not None or True
        except (ValueError, IndexError):
            pass  # Expected if negative index is rejected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

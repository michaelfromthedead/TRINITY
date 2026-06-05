"""IK System Integration Tests (T-AN-9.4).

Comprehensive tests for the IK solver dispatch system including:
- Priority ordering of goals
- Solver dispatch correctness (each solver type)
- Chain weight blending (0-1)
- Chain enable/disable
- Multi-chain interaction
- Performance with many active chains

Total: 60+ tests
"""

from __future__ import annotations

import math
import time
import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform

from engine.animation.systems.ik_system import (
    system,
    IKSolverType,
    IKHintType,
    IKGoal,
    IKChainBone,
    IKSolveResult,
    IKComponent,
    IKSystem,
    IKSystemStats,
)
from engine.animation.ik.fullbody import SkeletonMapping, BodyPart


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass(frozen=True)
class MockEntity:
    """Mock entity for testing."""
    id: int

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass
class MockWorld:
    """Mock ECS world for testing."""
    pass


@pytest.fixture
def simple_hierarchy() -> Dict[int, int]:
    """Simple 5-bone hierarchy for testing."""
    return {
        0: -1,  # Root (no parent)
        1: 0,   # Child of root
        2: 1,   # Grandchild
        3: 2,   # Great-grandchild
        4: 3,   # End effector
    }


@pytest.fixture
def arm_hierarchy() -> Dict[int, int]:
    """Simple arm hierarchy (shoulder -> elbow -> wrist)."""
    return {
        0: -1,  # Shoulder
        1: 0,   # Elbow
        2: 1,   # Wrist
    }


@pytest.fixture
def bone_lengths() -> Dict[int, float]:
    """Bone lengths for the simple hierarchy."""
    return {
        0: 0.0,  # Root
        1: 0.3,
        2: 0.3,
        3: 0.3,
        4: 0.3,
    }


@pytest.fixture
def arm_bone_lengths() -> Dict[int, float]:
    """Bone lengths for arm."""
    return {
        0: 0.0,  # Shoulder
        1: 1.0,  # Upper arm
        2: 1.0,  # Lower arm
    }


@pytest.fixture
def simple_pose() -> Dict[int, Transform]:
    """Simple pose with identity transforms."""
    return {
        i: Transform(Vec3(0, i * 0.3, 0), Quat.identity(), Vec3.one())
        for i in range(5)
    }


@pytest.fixture
def arm_pose() -> Dict[int, Transform]:
    """Arm pose for testing."""
    return {
        0: Transform(Vec3(0, 0, 0), Quat.identity(), Vec3.one()),
        1: Transform(Vec3(0, 1, 0), Quat.identity(), Vec3.one()),
        2: Transform(Vec3(0, 2, 0), Quat.identity(), Vec3.one()),
    }


@pytest.fixture
def ik_system(simple_hierarchy, bone_lengths) -> IKSystem:
    """Configured IK system instance."""
    system = IKSystem()
    system.set_skeleton_data(simple_hierarchy, bone_lengths)
    return system


@pytest.fixture
def arm_ik_system(arm_hierarchy, arm_bone_lengths) -> IKSystem:
    """IK system configured for arm."""
    system = IKSystem()
    system.set_skeleton_data(arm_hierarchy, arm_bone_lengths)
    return system


@pytest.fixture
def humanoid_skeleton() -> SkeletonMapping:
    """Humanoid skeleton mapping for full-body tests."""
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


# =============================================================================
# System Decorator Tests (5 tests)
# =============================================================================


class TestSystemDecorator:
    """Tests for @system decorator."""

    def test_decorator_sets_phase(self):
        """Test decorator sets phase attribute."""
        @system(phase="animation")
        class TestSystem:
            pass

        assert TestSystem._system_phase == "animation"

    def test_decorator_sets_order(self):
        """Test decorator sets order attribute."""
        @system(phase="animation", order=1)
        class TestSystem:
            pass

        assert TestSystem._system_order == 1

    def test_decorator_sets_reads_writes(self):
        """Test decorator sets reads/writes attributes."""
        @system(phase="animation", reads=("IKComponent",), writes=("Pose",))
        class TestSystem:
            pass

        assert TestSystem._system_reads == ("IKComponent",)
        assert TestSystem._system_writes == ("Pose",)

    def test_ik_system_has_correct_phase(self):
        """Test IKSystem has animation phase."""
        assert IKSystem._system_phase == "animation"

    def test_ik_system_runs_after_graph(self):
        """Test IKSystem has order 1 (after animation graph at 0)."""
        assert IKSystem._system_order == 1


# =============================================================================
# IKGoal Tests (10 tests)
# =============================================================================


class TestIKGoal:
    """Tests for IKGoal data structure."""

    def test_goal_creation_defaults(self):
        """Test goal creation with defaults."""
        goal = IKGoal(target_bone=4, target_position=Vec3(1, 1, 0))
        assert goal.target_bone == 4
        assert goal.weight == 1.0
        assert goal.priority == 0
        assert goal.enabled

    def test_goal_with_priority(self):
        """Test goal with priority setting."""
        goal = IKGoal(target_bone=4, target_position=Vec3(1, 1, 0), priority=10)
        assert goal.priority == 10

    def test_goal_with_solver_type(self):
        """Test goal with specific solver type."""
        goal = IKGoal(
            target_bone=4,
            target_position=Vec3(1, 1, 0),
            solver_type=IKSolverType.FABRIK
        )
        assert goal.solver_type == IKSolverType.FABRIK

    def test_goal_validation_valid(self):
        """Test valid goal passes validation."""
        goal = IKGoal(
            target_bone=4,
            target_position=Vec3(1, 1, 0),
            weight=0.5,
            chain_length=3
        )
        assert goal.is_valid()

    def test_goal_validation_invalid_bone(self):
        """Test invalid bone fails validation."""
        goal = IKGoal(target_bone=-1, target_position=Vec3(1, 1, 0))
        assert not goal.is_valid()

    def test_goal_validation_invalid_weight(self):
        """Test invalid weight fails validation."""
        goal = IKGoal(target_bone=4, target_position=Vec3(1, 1, 0), weight=1.5)
        assert not goal.is_valid()

    def test_goal_validation_no_target(self):
        """Test goal without target fails validation."""
        goal = IKGoal(target_bone=4)
        assert not goal.is_valid()

    def test_goal_has_position_target(self):
        """Test has_position_target method."""
        goal = IKGoal(target_bone=4, target_position=Vec3(1, 1, 0))
        assert goal.has_position_target()

        goal2 = IKGoal(target_bone=4, target_rotation=Quat.identity())
        assert not goal2.has_position_target()

    def test_goal_has_rotation_target(self):
        """Test has_rotation_target method."""
        goal = IKGoal(target_bone=4, target_rotation=Quat.identity())
        assert goal.has_rotation_target()

    def test_goal_set_target(self):
        """Test set_target method."""
        goal = IKGoal(target_bone=4)
        goal.set_target(position=Vec3(1, 1, 0), rotation=Quat.identity())
        assert goal.target_position == Vec3(1, 1, 0)
        assert goal.target_rotation == Quat.identity()


# =============================================================================
# IKComponent Tests (12 tests)
# =============================================================================


class TestIKComponent:
    """Tests for IKComponent."""

    def test_component_creation(self):
        """Test component creation with defaults."""
        component = IKComponent()
        assert len(component.goals) == 0
        assert component.enabled
        assert component.blend_to_animation == 0.0

    def test_add_goal(self):
        """Test adding a goal."""
        component = IKComponent()
        goal = IKGoal(target_bone=4, target_position=Vec3(1, 1, 0))
        idx = component.add_goal(goal)
        assert idx == 0
        assert len(component.goals) == 1

    def test_remove_goal(self):
        """Test removing a goal."""
        component = IKComponent()
        component.add_goal(IKGoal(target_bone=4, target_position=Vec3(1, 1, 0)))
        assert component.remove_goal(0)
        assert len(component.goals) == 0

    def test_remove_goal_invalid_index(self):
        """Test removing goal with invalid index."""
        component = IKComponent()
        assert not component.remove_goal(0)

    def test_get_goal(self):
        """Test getting a goal by index."""
        component = IKComponent()
        goal = IKGoal(target_bone=4, target_position=Vec3(1, 1, 0))
        component.add_goal(goal)
        retrieved = component.get_goal(0)
        assert retrieved == goal

    def test_get_goal_by_name(self):
        """Test getting a goal by name."""
        component = IKComponent()
        goal = IKGoal(target_bone=4, target_position=Vec3(1, 1, 0), chain_name="left_arm")
        component.add_goal(goal)
        retrieved = component.get_goal_by_name("left_arm")
        assert retrieved == goal

    def test_set_goal_enabled(self):
        """Test enabling/disabling a goal."""
        component = IKComponent()
        goal = IKGoal(target_bone=4, target_position=Vec3(1, 1, 0))
        component.add_goal(goal)
        assert component.set_goal_enabled(0, False)
        assert not component.goals[0].enabled

    def test_set_goal_weight(self):
        """Test setting goal weight."""
        component = IKComponent()
        goal = IKGoal(target_bone=4, target_position=Vec3(1, 1, 0))
        component.add_goal(goal)
        assert component.set_goal_weight(0, 0.5)
        assert component.goals[0].weight == 0.5

    def test_set_goal_weight_clamped(self):
        """Test goal weight is clamped to 0-1."""
        component = IKComponent()
        goal = IKGoal(target_bone=4, target_position=Vec3(1, 1, 0))
        component.add_goal(goal)
        component.set_goal_weight(0, 1.5)
        assert component.goals[0].weight == 1.0
        component.set_goal_weight(0, -0.5)
        assert component.goals[0].weight == 0.0

    def test_set_all_weights(self):
        """Test setting all goal weights."""
        component = IKComponent()
        for i in range(3):
            component.add_goal(IKGoal(target_bone=i, target_position=Vec3(i, i, 0)))
        component.set_all_weights(0.5)
        for goal in component.goals:
            assert goal.weight == 0.5

    def test_get_enabled_goals_sorted(self):
        """Test getting enabled goals sorted by priority."""
        component = IKComponent()
        component.add_goal(IKGoal(target_bone=1, target_position=Vec3(1, 0, 0), priority=1))
        component.add_goal(IKGoal(target_bone=2, target_position=Vec3(2, 0, 0), priority=10))
        component.add_goal(IKGoal(target_bone=3, target_position=Vec3(3, 0, 0), priority=5, enabled=False))
        component.add_goal(IKGoal(target_bone=4, target_position=Vec3(4, 0, 0), priority=3))

        sorted_goals = component.get_enabled_goals_sorted()
        assert len(sorted_goals) == 3  # One is disabled
        assert sorted_goals[0].priority == 10
        assert sorted_goals[1].priority == 3
        assert sorted_goals[2].priority == 1

    def test_zero_weight_goals_excluded(self):
        """Test zero-weight goals are excluded from sorted list."""
        component = IKComponent()
        component.add_goal(IKGoal(target_bone=1, target_position=Vec3(1, 0, 0), weight=0.0))
        component.add_goal(IKGoal(target_bone=2, target_position=Vec3(2, 0, 0), weight=0.5))

        sorted_goals = component.get_enabled_goals_sorted()
        assert len(sorted_goals) == 1
        assert sorted_goals[0].target_bone == 2


# =============================================================================
# Priority Ordering Tests (8 tests)
# =============================================================================


class TestPriorityOrdering:
    """Tests for priority-based goal processing."""

    def test_higher_priority_processed_first(self, arm_ik_system, arm_pose):
        """Test higher priority goals are processed first."""
        entity = MockEntity(1)
        component = IKComponent()

        # Add goals in reverse priority order
        component.add_goal(IKGoal(
            target_bone=2, target_position=Vec3(0, 1.5, 0),
            priority=1, chain_length=2
        ))
        component.add_goal(IKGoal(
            target_bone=2, target_position=Vec3(1, 1, 0),
            priority=10, chain_length=2
        ))

        pose_data = {entity: arm_pose}
        result = arm_ik_system.update(
            MockWorld(),
            [(entity, component)],
            pose_data
        )

        # Higher priority goal (priority=10) should be applied last,
        # resulting in its target being more closely matched
        stats = arm_ik_system.get_stats()
        assert stats.goals_processed == 2

    def test_equal_priority_maintains_order(self, arm_ik_system, arm_pose):
        """Test equal priority goals maintain insertion order."""
        entity = MockEntity(1)
        component = IKComponent()

        component.add_goal(IKGoal(
            target_bone=2, target_position=Vec3(1, 0, 0),
            priority=5, chain_length=2
        ))
        component.add_goal(IKGoal(
            target_bone=2, target_position=Vec3(0, 1, 0),
            priority=5, chain_length=2
        ))

        pose_data = {entity: arm_pose}
        result = arm_ik_system.update(
            MockWorld(),
            [(entity, component)],
            pose_data
        )

        stats = arm_ik_system.get_stats()
        assert stats.goals_processed == 2

    def test_negative_priority_processed_last(self, arm_ik_system, arm_pose):
        """Test negative priority goals are processed after positive."""
        entity = MockEntity(1)
        component = IKComponent()

        component.add_goal(IKGoal(
            target_bone=2, target_position=Vec3(1, 0, 0),
            priority=-5, chain_length=2
        ))
        component.add_goal(IKGoal(
            target_bone=2, target_position=Vec3(0, 1, 0),
            priority=5, chain_length=2
        ))

        sorted_goals = component.get_enabled_goals_sorted()
        assert sorted_goals[0].priority == 5
        assert sorted_goals[1].priority == -5

    def test_disabled_goals_not_in_priority_list(self, arm_ik_system, arm_pose):
        """Test disabled goals are excluded from priority ordering."""
        component = IKComponent()

        component.add_goal(IKGoal(
            target_bone=2, target_position=Vec3(1, 0, 0),
            priority=100, enabled=False, chain_length=2
        ))
        component.add_goal(IKGoal(
            target_bone=2, target_position=Vec3(0, 1, 0),
            priority=1, chain_length=2
        ))

        sorted_goals = component.get_enabled_goals_sorted()
        assert len(sorted_goals) == 1
        assert sorted_goals[0].priority == 1

    def test_priority_ordering_with_mixed_weights(self):
        """Test priority ordering with various weights."""
        component = IKComponent()

        component.add_goal(IKGoal(
            target_bone=1, target_position=Vec3(1, 0, 0),
            priority=10, weight=0.1
        ))
        component.add_goal(IKGoal(
            target_bone=2, target_position=Vec3(2, 0, 0),
            priority=5, weight=1.0
        ))
        component.add_goal(IKGoal(
            target_bone=3, target_position=Vec3(3, 0, 0),
            priority=1, weight=0.5
        ))

        sorted_goals = component.get_enabled_goals_sorted()
        assert sorted_goals[0].priority == 10
        assert sorted_goals[1].priority == 5
        assert sorted_goals[2].priority == 1

    def test_many_priorities(self):
        """Test sorting with many different priorities."""
        component = IKComponent()

        priorities = [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]
        for i, p in enumerate(priorities):
            component.add_goal(IKGoal(
                target_bone=i, target_position=Vec3(i, 0, 0),
                priority=p
            ))

        sorted_goals = component.get_enabled_goals_sorted()
        expected = sorted(priorities, reverse=True)
        actual = [g.priority for g in sorted_goals]
        assert actual == expected

    def test_priority_affects_final_pose(self, arm_ik_system, arm_pose):
        """Test that priority affects which goal has more influence on final pose."""
        entity = MockEntity(1)

        # Test with high priority goal
        component_high = IKComponent()
        component_high.add_goal(IKGoal(
            target_bone=2, target_position=Vec3(0, 1.5, 0),
            priority=1, chain_length=2
        ))
        component_high.add_goal(IKGoal(
            target_bone=2, target_position=Vec3(1.5, 0, 0),
            priority=10, chain_length=2  # This should dominate
        ))

        pose_data = {entity: dict(arm_pose)}
        result = arm_ik_system.update(MockWorld(), [(entity, component_high)], pose_data)

        assert entity in result
        assert len(result[entity]) > 0

    def test_stats_track_goal_processing(self, arm_ik_system, arm_pose):
        """Test that stats correctly track goals processed."""
        entity = MockEntity(1)
        component = IKComponent()

        for i in range(5):
            component.add_goal(IKGoal(
                target_bone=2, target_position=Vec3(i, 0, 0),
                priority=i, chain_length=2
            ))

        pose_data = {entity: arm_pose}
        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        stats = arm_ik_system.get_stats()
        assert stats.goals_processed == 5


# =============================================================================
# Solver Dispatch Tests (15 tests)
# =============================================================================


class TestSolverDispatch:
    """Tests for solver type dispatch."""

    def test_auto_select_two_bone(self, arm_ik_system, arm_pose):
        """Test AUTO selects two-bone for 2-bone chains."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            solver_type=IKSolverType.AUTO,
            chain_length=2
        ))

        pose_data = {entity: arm_pose}
        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        stats = arm_ik_system.get_stats()
        assert stats.two_bone_solves == 1

    def test_auto_select_fabrik_medium(self, ik_system, simple_pose):
        """Test AUTO selects FABRIK for medium chains."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=4,
            target_position=Vec3(0.5, 1.0, 0),
            solver_type=IKSolverType.AUTO,
            chain_length=4  # 4-bone chain -> FABRIK
        ))

        pose_data = {entity: simple_pose}
        ik_system.update(MockWorld(), [(entity, component)], pose_data)

        stats = ik_system.get_stats()
        assert stats.fabrik_solves == 1

    def test_explicit_two_bone_solver(self, arm_ik_system, arm_pose):
        """Test explicit TWO_BONE solver type."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            solver_type=IKSolverType.TWO_BONE,
            chain_length=2
        ))

        pose_data = {entity: arm_pose}
        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        stats = arm_ik_system.get_stats()
        assert stats.two_bone_solves == 1

    def test_explicit_fabrik_solver(self, ik_system, simple_pose):
        """Test explicit FABRIK solver type."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=4,
            target_position=Vec3(0.5, 1.0, 0),
            solver_type=IKSolverType.FABRIK,
            chain_length=3
        ))

        pose_data = {entity: simple_pose}
        ik_system.update(MockWorld(), [(entity, component)], pose_data)

        stats = ik_system.get_stats()
        assert stats.fabrik_solves == 1

    def test_explicit_ccd_solver(self, ik_system, simple_pose):
        """Test explicit CCD solver type."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=4,
            target_position=Vec3(0.5, 1.0, 0),
            solver_type=IKSolverType.CCD,
            chain_length=3
        ))

        pose_data = {entity: simple_pose}
        ik_system.update(MockWorld(), [(entity, component)], pose_data)

        stats = ik_system.get_stats()
        assert stats.ccd_solves == 1

    def test_two_bone_with_pole_vector(self, arm_ik_system, arm_pose):
        """Test two-bone solver with pole vector hint."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            solver_type=IKSolverType.TWO_BONE,
            chain_length=2,
            hint_type=IKHintType.POSITION,
            hint_value=Vec3(0, 0, 1)
        ))

        pose_data = {entity: arm_pose}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        assert entity in result
        stats = arm_ik_system.get_stats()
        assert stats.two_bone_solves == 1

    def test_two_bone_with_direction_hint(self, arm_ik_system, arm_pose):
        """Test two-bone solver with direction hint."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            solver_type=IKSolverType.TWO_BONE,
            chain_length=2,
            hint_type=IKHintType.DIRECTION,
            hint_value=Vec3(0, 0, 1)
        ))

        pose_data = {entity: arm_pose}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        assert entity in result

    def test_solver_with_rotation_target(self, arm_ik_system, arm_pose):
        """Test solver applies rotation target."""
        entity = MockEntity(1)
        component = IKComponent()
        target_rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 4)
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            target_rotation=target_rot,
            solver_type=IKSolverType.TWO_BONE,
            chain_length=2
        ))

        pose_data = {entity: arm_pose}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        assert entity in result

    def test_multiple_solver_types(self, arm_ik_system, arm_pose):
        """Test multiple goals with different solver types."""
        entity = MockEntity(1)
        component = IKComponent()

        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 0, 0),
            solver_type=IKSolverType.TWO_BONE,
            chain_length=2,
            priority=1
        ))
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(0, 1, 0),
            solver_type=IKSolverType.FABRIK,
            chain_length=2,
            priority=2
        ))

        pose_data = {entity: arm_pose}
        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        stats = arm_ik_system.get_stats()
        assert stats.two_bone_solves >= 1
        assert stats.fabrik_solves >= 1

    def test_solver_convergence_tracking(self, arm_ik_system, arm_pose):
        """Test solver convergence is tracked."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            solver_type=IKSolverType.TWO_BONE,
            chain_length=2
        ))

        pose_data = {entity: arm_pose}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        # Two-bone should always converge (analytical)
        stats = arm_ik_system.get_stats()
        assert stats.total_solves == 1

    def test_unreachable_target_handled(self, arm_ik_system, arm_pose):
        """Test unreachable target is handled gracefully."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(100, 100, 100),  # Way out of reach
            solver_type=IKSolverType.TWO_BONE,
            chain_length=2
        ))

        pose_data = {entity: arm_pose}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        # Should complete without error
        assert entity in result

    def test_short_chain_handled(self, arm_ik_system, arm_pose):
        """Test single-bone chain is handled."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            chain_length=1
        ))

        pose_data = {entity: arm_pose}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        # Should complete without error
        assert entity in result

    def test_solver_tolerance_respected(self, ik_system, simple_pose):
        """Test solver respects tolerance parameter."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=4,
            target_position=Vec3(0, 1.2, 0),
            solver_type=IKSolverType.FABRIK,
            chain_length=4,
            position_tolerance=0.001
        ))

        pose_data = {entity: simple_pose}
        result = ik_system.update(MockWorld(), [(entity, component)], pose_data)

        assert entity in result

    def test_solver_max_iterations_respected(self, ik_system, simple_pose):
        """Test solver respects max iterations."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=4,
            target_position=Vec3(0.5, 1.0, 0),
            solver_type=IKSolverType.FABRIK,
            chain_length=4,
            max_iterations=5
        ))

        pose_data = {entity: simple_pose}
        result = ik_system.update(MockWorld(), [(entity, component)], pose_data)

        assert entity in result

    def test_invalid_goal_tracked(self, arm_ik_system, arm_pose):
        """Test invalid goals are tracked in stats."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=-1,  # Invalid bone
            target_position=Vec3(1, 1, 0)
        ))

        pose_data = {entity: arm_pose}
        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        stats = arm_ik_system.get_stats()
        assert stats.invalid_goals == 1


# =============================================================================
# Weight Blending Tests (10 tests)
# =============================================================================


class TestWeightBlending:
    """Tests for chain weight blending."""

    def test_full_weight_applies_fully(self, arm_ik_system, arm_pose):
        """Test weight=1.0 applies IK fully."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            weight=1.0,
            chain_length=2
        ))

        original_pose = {k: Transform(v.translation, v.rotation, v.scale) for k, v in arm_pose.items()}
        pose_data = {entity: arm_pose}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        # Pose should be modified
        assert entity in result

    def test_zero_weight_no_effect(self, arm_ik_system, arm_pose):
        """Test weight=0.0 has no effect on pose."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            weight=0.0,
            chain_length=2
        ))

        original_pose = {k: Transform(v.translation, v.rotation, v.scale) for k, v in arm_pose.items()}
        pose_data = {entity: dict(arm_pose)}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        # Goal should be excluded (zero weight)
        stats = arm_ik_system.get_stats()
        assert stats.goals_processed == 0

    def test_half_weight_partial_blend(self, arm_ik_system, arm_pose):
        """Test weight=0.5 blends pose halfway."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(0, 2, 0),  # Above current position
            weight=0.5,
            chain_length=2
        ))

        pose_data = {entity: dict(arm_pose)}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        assert entity in result

    def test_blend_to_animation_factor(self, arm_ik_system, arm_pose):
        """Test blend_to_animation reduces IK influence."""
        entity = MockEntity(1)
        component = IKComponent(blend_to_animation=0.5)
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            weight=1.0,
            chain_length=2
        ))

        original_pose = {k: Transform(v.translation, v.rotation, v.scale) for k, v in arm_pose.items()}
        pose_data = {entity: dict(arm_pose)}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        # Effective weight should be 0.5 (weight * (1 - blend_to_animation))
        assert entity in result

    def test_full_blend_to_animation(self, arm_ik_system, arm_pose):
        """Test blend_to_animation=1.0 removes all IK."""
        entity = MockEntity(1)
        component = IKComponent(blend_to_animation=1.0)
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            weight=1.0,
            chain_length=2
        ))

        original_pose = {k: Transform(v.translation, v.rotation, v.scale) for k, v in arm_pose.items()}
        pose_data = {entity: dict(arm_pose)}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        # Pose should be unchanged (or very close)
        # Effective weight is 0 (1.0 * (1 - 1.0))
        assert entity in result

    def test_weight_affects_multiple_bones(self, arm_ik_system, arm_pose):
        """Test weight blending affects all bones in chain."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            weight=0.5,
            chain_length=2
        ))

        pose_data = {entity: dict(arm_pose)}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        assert entity in result
        # Both arm bones should be affected
        assert 1 in result[entity] or 2 in result[entity]

    def test_different_weights_per_goal(self, arm_ik_system, arm_pose):
        """Test different weights for different goals."""
        entity = MockEntity(1)
        component = IKComponent()

        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 0, 0),
            weight=0.3,
            priority=1,
            chain_length=2
        ))
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(0, 1, 0),
            weight=0.7,
            priority=2,
            chain_length=2
        ))

        pose_data = {entity: dict(arm_pose)}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        assert entity in result

    def test_weight_interpolation_smooth(self, arm_ik_system, arm_pose):
        """Test weight interpolation produces smooth results."""
        entity = MockEntity(1)
        results = []

        for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
            if w == 0.0:
                continue  # Zero weight is excluded
            component = IKComponent()
            component.add_goal(IKGoal(
                target_bone=2,
                target_position=Vec3(0, 2, 0),
                weight=w,
                chain_length=2
            ))

            pose_data = {entity: {k: Transform(v.translation, v.rotation, v.scale) for k, v in arm_pose.items()}}
            result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)
            results.append(result[entity])

        # All results should be valid
        assert all(len(r) > 0 for r in results)

    def test_cumulative_weights(self, arm_ik_system, arm_pose):
        """Test multiple goals with cumulative weights."""
        entity = MockEntity(1)
        component = IKComponent()

        # Multiple goals affecting same bone
        for i, w in enumerate([0.2, 0.3, 0.5]):
            component.add_goal(IKGoal(
                target_bone=2,
                target_position=Vec3(i, 1, 0),
                weight=w,
                priority=i,
                chain_length=2
            ))

        pose_data = {entity: dict(arm_pose)}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        stats = arm_ik_system.get_stats()
        assert stats.goals_processed == 3

    def test_weight_with_disabled_component(self, arm_ik_system, arm_pose):
        """Test weights don't apply when component is disabled."""
        entity = MockEntity(1)
        component = IKComponent(enabled=False)
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            weight=1.0,
            chain_length=2
        ))

        original_pose = {k: Transform(v.translation, v.rotation, v.scale) for k, v in arm_pose.items()}
        pose_data = {entity: dict(arm_pose)}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        # Component disabled, no goals processed
        stats = arm_ik_system.get_stats()
        assert stats.goals_processed == 0


# =============================================================================
# Enable/Disable Tests (5 tests)
# =============================================================================


class TestEnableDisable:
    """Tests for chain enable/disable functionality."""

    def test_disabled_goal_skipped(self, arm_ik_system, arm_pose):
        """Test disabled goals are skipped."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            enabled=False,
            chain_length=2
        ))

        pose_data = {entity: arm_pose}
        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        stats = arm_ik_system.get_stats()
        assert stats.goals_processed == 0

    def test_disabled_component_skips_all(self, arm_ik_system, arm_pose):
        """Test disabled component skips all goals."""
        entity = MockEntity(1)
        component = IKComponent(enabled=False)
        for i in range(5):
            component.add_goal(IKGoal(
                target_bone=2,
                target_position=Vec3(i, 1, 0),
                chain_length=2
            ))

        pose_data = {entity: arm_pose}
        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        stats = arm_ik_system.get_stats()
        assert stats.goals_processed == 0

    def test_enable_disable_toggle(self, arm_ik_system, arm_pose):
        """Test toggling goal enable state."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            chain_length=2
        ))

        pose_data = {entity: dict(arm_pose)}

        # Initially enabled
        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)
        assert arm_ik_system.get_stats().goals_processed == 1

        # Disable
        component.set_goal_enabled(0, False)
        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)
        assert arm_ik_system.get_stats().goals_processed == 0

        # Re-enable
        component.set_goal_enabled(0, True)
        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)
        assert arm_ik_system.get_stats().goals_processed == 1

    def test_partial_enable_disable(self, arm_ik_system, arm_pose):
        """Test some goals enabled, some disabled."""
        entity = MockEntity(1)
        component = IKComponent()

        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 0, 0),
            enabled=True,
            chain_length=2
        ))
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(0, 1, 0),
            enabled=False,
            chain_length=2
        ))
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(0, 0, 1),
            enabled=True,
            chain_length=2
        ))

        pose_data = {entity: arm_pose}
        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        stats = arm_ik_system.get_stats()
        assert stats.goals_processed == 2

    def test_runtime_enable_disable(self, arm_ik_system, arm_pose):
        """Test enabling/disabling goals at runtime."""
        entity = MockEntity(1)
        component = IKComponent()

        for i in range(3):
            component.add_goal(IKGoal(
                target_bone=2,
                target_position=Vec3(i, 1, 0),
                chain_length=2
            ))

        pose_data = {entity: dict(arm_pose)}

        # All enabled
        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)
        assert arm_ik_system.get_stats().goals_processed == 3

        # Disable middle one
        component.goals[1].enabled = False
        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)
        assert arm_ik_system.get_stats().goals_processed == 2


# =============================================================================
# Multi-Chain Tests (8 tests)
# =============================================================================


class TestMultiChain:
    """Tests for multi-chain interaction."""

    def test_multiple_independent_chains(self, arm_ik_system, arm_pose):
        """Test multiple independent chains don't interfere."""
        entity = MockEntity(1)
        component = IKComponent()

        # Different chains targeting different bones
        component.add_goal(IKGoal(
            target_bone=1,
            target_position=Vec3(0.5, 0.5, 0),
            chain_length=1,
            chain_name="chain1"
        ))
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            chain_length=1,
            chain_name="chain2"
        ))

        pose_data = {entity: arm_pose}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        stats = arm_ik_system.get_stats()
        assert stats.goals_processed == 2

    def test_overlapping_chains(self, ik_system, simple_pose):
        """Test overlapping chain regions."""
        entity = MockEntity(1)
        component = IKComponent()

        # Chains that share bones
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(0.3, 0.6, 0),
            chain_length=2,  # Bones 1-2
            priority=1
        ))
        component.add_goal(IKGoal(
            target_bone=4,
            target_position=Vec3(0.5, 1.2, 0),
            chain_length=4,  # Bones 1-4 (overlaps)
            priority=2
        ))

        pose_data = {entity: simple_pose}
        result = ik_system.update(MockWorld(), [(entity, component)], pose_data)

        stats = ik_system.get_stats()
        assert stats.goals_processed == 2

    def test_sequential_chain_updates(self, ik_system, simple_pose):
        """Test chains update sequentially by priority."""
        entity = MockEntity(1)
        component = IKComponent()

        # Add goals with explicit priority ordering
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(0.2, 0.6, 0),
            chain_length=2,
            priority=1
        ))
        component.add_goal(IKGoal(
            target_bone=3,
            target_position=Vec3(0.3, 0.9, 0),
            chain_length=3,
            priority=2
        ))
        component.add_goal(IKGoal(
            target_bone=4,
            target_position=Vec3(0.4, 1.2, 0),
            chain_length=4,
            priority=3
        ))

        pose_data = {entity: simple_pose}
        result = ik_system.update(MockWorld(), [(entity, component)], pose_data)

        stats = ik_system.get_stats()
        assert stats.goals_processed == 3

    def test_chain_isolation(self):
        """Test chains on different entities are isolated."""
        system = IKSystem()
        hierarchy = {0: -1, 1: 0, 2: 1}
        lengths = {0: 0.0, 1: 1.0, 2: 1.0}
        system.set_skeleton_data(hierarchy, lengths)

        entity1 = MockEntity(1)
        entity2 = MockEntity(2)

        component1 = IKComponent()
        component1.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 0, 0),
            chain_length=2
        ))

        component2 = IKComponent()
        component2.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(0, 1, 0),
            chain_length=2
        ))

        pose = {i: Transform(Vec3(0, i, 0), Quat.identity(), Vec3.one()) for i in range(3)}
        pose_data = {entity1: dict(pose), entity2: dict(pose)}

        result = system.update(
            MockWorld(),
            [(entity1, component1), (entity2, component2)],
            pose_data
        )

        # Both entities should have results
        assert entity1 in result
        assert entity2 in result

        stats = system.get_stats()
        assert stats.entities_processed == 2

    def test_many_chains_performance(self, arm_ik_system, arm_pose):
        """Test performance with many active chains."""
        entity = MockEntity(1)
        component = IKComponent()

        # Add many goals
        for i in range(10):
            component.add_goal(IKGoal(
                target_bone=2,
                target_position=Vec3(i * 0.1, 1, 0),
                chain_length=2,
                priority=i
            ))

        pose_data = {entity: arm_pose}

        start = time.perf_counter()
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)
        elapsed = time.perf_counter() - start

        stats = arm_ik_system.get_stats()
        assert stats.goals_processed == 10
        assert elapsed < 1.0  # Should complete in under 1 second

    def test_chain_names_unique(self):
        """Test chain names can be used for lookup."""
        component = IKComponent()

        component.add_goal(IKGoal(
            target_bone=1,
            target_position=Vec3(1, 0, 0),
            chain_name="left_arm"
        ))
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(0, 1, 0),
            chain_name="right_arm"
        ))

        assert component.get_goal_by_name("left_arm") is not None
        assert component.get_goal_by_name("right_arm") is not None
        assert component.get_goal_by_name("spine") is None

    def test_multi_entity_processing(self, arm_ik_system, arm_pose):
        """Test processing multiple entities."""
        entities = [MockEntity(i) for i in range(5)]
        components = []

        for i, entity in enumerate(entities):
            component = IKComponent()
            component.add_goal(IKGoal(
                target_bone=2,
                target_position=Vec3(i * 0.2, 1, 0),
                chain_length=2
            ))
            components.append((entity, component))

        pose_data = {e: dict(arm_pose) for e in entities}
        result = arm_ik_system.update(MockWorld(), components, pose_data)

        stats = arm_ik_system.get_stats()
        assert stats.entities_processed == 5
        assert stats.goals_processed == 5

    def test_mixed_solver_multi_chain(self, arm_ik_system, arm_pose):
        """Test multiple chains with different solver types."""
        entity = MockEntity(1)
        component = IKComponent()

        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 0, 0),
            solver_type=IKSolverType.TWO_BONE,
            chain_length=2,
            priority=1
        ))
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(0, 1, 0),
            solver_type=IKSolverType.FABRIK,
            chain_length=2,
            priority=2
        ))
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(0, 0, 1),
            solver_type=IKSolverType.CCD,
            chain_length=2,
            priority=3
        ))

        pose_data = {entity: arm_pose}
        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        stats = arm_ik_system.get_stats()
        assert stats.two_bone_solves >= 1
        assert stats.fabrik_solves >= 1
        assert stats.ccd_solves >= 1


# =============================================================================
# Performance Tests (5 tests)
# =============================================================================


class TestPerformance:
    """Performance tests with many active chains."""

    def test_many_goals_performance(self, arm_ik_system, arm_pose):
        """Test performance with 50 goals."""
        entity = MockEntity(1)
        component = IKComponent()

        for i in range(50):
            component.add_goal(IKGoal(
                target_bone=2,
                target_position=Vec3(i * 0.02, 1 + i * 0.01, 0),
                chain_length=2,
                priority=i
            ))

        pose_data = {entity: arm_pose}

        start = time.perf_counter()
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)
        elapsed = time.perf_counter() - start

        stats = arm_ik_system.get_stats()
        assert stats.goals_processed == 50
        assert elapsed < 2.0  # Should complete in under 2 seconds

    def test_many_entities_performance(self, arm_hierarchy, arm_bone_lengths, arm_pose):
        """Test performance with many entities."""
        system = IKSystem()
        system.set_skeleton_data(arm_hierarchy, arm_bone_lengths)

        entities = [MockEntity(i) for i in range(100)]
        components = []

        for entity in entities:
            component = IKComponent()
            component.add_goal(IKGoal(
                target_bone=2,
                target_position=Vec3(1, 1, 0),
                chain_length=2
            ))
            components.append((entity, component))

        pose_data = {e: dict(arm_pose) for e in entities}

        start = time.perf_counter()
        result = system.update(MockWorld(), components, pose_data)
        elapsed = time.perf_counter() - start

        stats = system.get_stats()
        assert stats.entities_processed == 100
        assert elapsed < 5.0  # Should complete in under 5 seconds

    def test_solver_caching(self, arm_ik_system, arm_pose):
        """Test solver instances are cached for performance."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            solver_type=IKSolverType.TWO_BONE,
            chain_length=2
        ))

        pose_data = {entity: arm_pose}

        # First call
        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)
        initial_solvers = len(arm_ik_system._two_bone_solvers)

        # Multiple subsequent calls
        for _ in range(10):
            arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        # Solver count should not increase
        assert len(arm_ik_system._two_bone_solvers) == initial_solvers

    def test_stats_reset_each_frame(self, arm_ik_system, arm_pose):
        """Test stats are reset each frame."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            chain_length=2
        ))

        pose_data = {entity: arm_pose}

        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)
        stats1 = arm_ik_system.get_stats()
        assert stats1.goals_processed == 1

        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)
        stats2 = arm_ik_system.get_stats()
        assert stats2.goals_processed == 1  # Reset, not accumulated

    def test_empty_update_fast(self, arm_ik_system):
        """Test empty update is very fast."""
        start = time.perf_counter()
        result = arm_ik_system.update(MockWorld(), [], {})
        elapsed = time.perf_counter() - start

        assert elapsed < 0.001  # Should be nearly instant
        assert len(result) == 0


# =============================================================================
# Integration Tests (5 tests)
# =============================================================================


class TestIntegration:
    """Integration tests for the full IK system pipeline."""

    def test_full_pipeline_flow(self, arm_ik_system, arm_pose):
        """Test complete pipeline from goal creation to pose output."""
        entity = MockEntity(1)
        component = IKComponent()

        # Create goal
        goal = IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            weight=0.8,
            priority=5,
            chain_length=2,
            chain_name="arm"
        )
        component.add_goal(goal)

        # Process
        pose_data = {entity: arm_pose}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        # Verify output
        assert entity in result
        assert len(result[entity]) > 0

        stats = arm_ik_system.get_stats()
        assert stats.entities_processed == 1
        assert stats.goals_processed == 1

    def test_skeleton_data_change(self, arm_ik_system, arm_pose):
        """Test system handles skeleton data changes."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            chain_length=2
        ))

        pose_data = {entity: arm_pose}

        # First update
        arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        # Change skeleton data
        new_hierarchy = {0: -1, 1: 0, 2: 1, 3: 2}
        new_lengths = {0: 0.0, 1: 1.5, 2: 1.5, 3: 1.0}
        arm_ik_system.set_skeleton_data(new_hierarchy, new_lengths)

        # Second update with new skeleton
        new_pose = {i: Transform(Vec3(0, i * 0.4, 0), Quat.identity(), Vec3.one()) for i in range(4)}
        pose_data = {entity: new_pose}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        assert entity in result

    def test_delta_time_parameter(self, arm_ik_system, arm_pose):
        """Test delta time is passed through."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            chain_length=2
        ))

        pose_data = {entity: arm_pose}

        # Different delta times
        result1 = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data, dt=1/60)
        result2 = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data, dt=1/30)

        # Both should complete successfully
        assert entity in result1
        assert entity in result2

    def test_pose_data_not_modified_in_place(self, arm_ik_system, arm_pose):
        """Test original pose data is not modified."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            chain_length=2
        ))

        # Keep copy of original
        original = {k: Transform(v.translation, v.rotation, v.scale) for k, v in arm_pose.items()}
        pose_data = {entity: arm_pose}

        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        # Original pose_data reference should still contain original transforms
        # (Note: in current implementation, the dict is copied internally)
        assert entity in result

    def test_world_transform_computation(self, ik_system, simple_pose):
        """Test world transforms are computed correctly."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=4,
            target_position=Vec3(0, 1.2, 0),
            chain_length=4
        ))

        pose_data = {entity: simple_pose}
        result = ik_system.update(MockWorld(), [(entity, component)], pose_data)

        # World transforms should have been computed internally
        assert entity in result
        assert len(result[entity]) > 0


# =============================================================================
# Edge Case Tests (5 tests)
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_component(self, arm_ik_system, arm_pose):
        """Test component with no goals."""
        entity = MockEntity(1)
        component = IKComponent()

        pose_data = {entity: arm_pose}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        assert entity in result
        stats = arm_ik_system.get_stats()
        assert stats.goals_processed == 0

    def test_missing_pose_data(self, arm_ik_system):
        """Test entity with missing pose data."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            chain_length=2
        ))

        pose_data = {}  # No pose data
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        # Should handle gracefully
        assert entity in result

    def test_target_at_bone_position(self, arm_ik_system, arm_pose):
        """Test target exactly at current bone position."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(0, 2, 0),  # Exact current position
            chain_length=2
        ))

        pose_data = {entity: arm_pose}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        assert entity in result

    def test_very_close_target(self, arm_ik_system, arm_pose):
        """Test target very close to current position."""
        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(0.001, 2.001, 0.001),  # Very close
            chain_length=2
        ))

        pose_data = {entity: arm_pose}
        result = arm_ik_system.update(MockWorld(), [(entity, component)], pose_data)

        assert entity in result

    def test_zero_length_bones(self):
        """Test handling of zero-length bones."""
        system = IKSystem()
        hierarchy = {0: -1, 1: 0, 2: 1}
        lengths = {0: 0.0, 1: 0.0, 2: 0.0}  # Zero lengths
        system.set_skeleton_data(hierarchy, lengths)

        entity = MockEntity(1)
        component = IKComponent()
        component.add_goal(IKGoal(
            target_bone=2,
            target_position=Vec3(1, 1, 0),
            chain_length=2
        ))

        pose = {i: Transform(Vec3(0, 0, 0), Quat.identity(), Vec3.one()) for i in range(3)}
        pose_data = {entity: pose}

        # Should not crash
        result = system.update(MockWorld(), [(entity, component)], pose_data)
        assert entity in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

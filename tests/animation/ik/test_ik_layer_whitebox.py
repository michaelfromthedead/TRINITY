"""Whitebox tests for IKLayer and related classes.

Comprehensive tests for:
- IKBlendMode enum (line 52): OVERRIDE, ADDITIVE, BLEND values
- IKGoalContext dataclass (line 72): field access, defaults, methods
- IKLayerResult dataclass (line 176): success, transforms, errors, blend_weight
- IKLayer class (line 218): initialization, solvers, goals, apply, blending
- IKLayerStack class (line 782): add/remove layers, apply, utilities

Target: 80+ tests covering all methods and internal logic with mocked dependencies.
"""

from __future__ import annotations

import math
import pytest
from dataclasses import fields, asdict
from typing import List, Dict, Optional
from unittest.mock import Mock, MagicMock, patch, PropertyMock, call

from engine.animation.ik.ik_layer import (
    IKBlendMode,
    IKGoalContext,
    IKLayerResult,
    IKLayer,
    IKLayerStack,
)
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform


# =============================================================================
# Helper Functions
# =============================================================================


def create_transforms(count: int = 5) -> List[Transform]:
    """Create a list of transforms for testing."""
    transforms = []
    for i in range(count):
        pos = Vec3(float(i), float(i) * 0.5, 0.0)
        transforms.append(Transform(pos, Quat.identity()))
    return transforms


def create_ik_transforms(count: int = 5) -> List[Transform]:
    """Create IK result transforms (slightly different from input)."""
    transforms = []
    for i in range(count):
        pos = Vec3(float(i) + 0.1, float(i) * 0.5 + 0.2, 0.1)
        transforms.append(Transform(pos, Quat.identity()))
    return transforms


def vec3_approx_equal(v1: Vec3, v2: Vec3, eps: float = 1e-4) -> bool:
    """Check if two Vec3 are approximately equal."""
    return (
        abs(v1.x - v2.x) < eps and
        abs(v1.y - v2.y) < eps and
        abs(v1.z - v2.z) < eps
    )


class MockFullBodyIK:
    """Mock FullBodyIK solver for testing."""
    def __init__(self):
        self.solve = Mock()
        mock_result = Mock()
        mock_result.transforms = create_ik_transforms(5)
        mock_result.success = True
        self.solve.return_value = mock_result


class MockFootPlacement:
    """Mock FootPlacement solver for testing."""
    def __init__(self):
        self.solve = Mock()
        mock_result = Mock()
        mock_result.transforms = create_ik_transforms(5)
        mock_result.success = True
        self.solve.return_value = mock_result


class MockTwoBoneIK:
    """Mock TwoBoneIK solver for testing."""
    def __init__(self):
        self.solve = Mock()
        mock_result = Mock()
        mock_result.transforms = create_ik_transforms(5)
        mock_result.success = True
        self.solve.return_value = mock_result


class MockFABRIKChain:
    """Mock FABRIKChain solver for testing."""
    def __init__(self):
        self.solve = Mock()
        mock_result = Mock()
        mock_result.transforms = create_ik_transforms(5)
        mock_result.success = True
        self.solve.return_value = mock_result


class MockGenericSolver:
    """Mock generic solver with solve interface."""
    def __init__(self):
        self.solve = Mock()
        mock_result = Mock()
        mock_result.transforms = create_ik_transforms(5)
        mock_result.success = True
        self.solve.return_value = mock_result


def create_mock_solver(solver_type: str = "FullBodyIK"):
    """Create a mock IK solver with expected behavior.

    Returns a class-based mock that properly reports its type name.
    """
    if solver_type == "FullBodyIK":
        return MockFullBodyIK()
    elif solver_type == "FootPlacement":
        return MockFootPlacement()
    elif solver_type == "TwoBoneIK":
        return MockTwoBoneIK()
    elif solver_type == "FABRIKChain":
        return MockFABRIKChain()
    else:
        return MockGenericSolver()


# =============================================================================
# Test IKBlendMode Enum
# =============================================================================


class TestIKBlendModeEnum:
    """Tests for IKBlendMode enum values and behavior."""

    def test_override_value_exists(self):
        """Test OVERRIDE mode exists in enum."""
        assert hasattr(IKBlendMode, "OVERRIDE")
        assert IKBlendMode.OVERRIDE is not None

    def test_additive_value_exists(self):
        """Test ADDITIVE mode exists in enum."""
        assert hasattr(IKBlendMode, "ADDITIVE")
        assert IKBlendMode.ADDITIVE is not None

    def test_blend_value_exists(self):
        """Test BLEND mode exists in enum."""
        assert hasattr(IKBlendMode, "BLEND")
        assert IKBlendMode.BLEND is not None

    def test_enum_values_are_unique(self):
        """Test all enum values are unique."""
        values = [IKBlendMode.OVERRIDE, IKBlendMode.ADDITIVE, IKBlendMode.BLEND]
        assert len(values) == len(set(values))

    def test_enum_has_exactly_three_members(self):
        """Test enum has exactly 3 members."""
        members = list(IKBlendMode)
        assert len(members) == 3

    def test_enum_iteration(self):
        """Test enum can be iterated."""
        modes = list(IKBlendMode)
        assert IKBlendMode.OVERRIDE in modes
        assert IKBlendMode.ADDITIVE in modes
        assert IKBlendMode.BLEND in modes


# =============================================================================
# Test IKGoalContext Dataclass
# =============================================================================


class TestIKGoalContextFields:
    """Tests for IKGoalContext dataclass fields and defaults."""

    def test_position_goals_default_empty(self):
        """Test position_goals defaults to empty dict."""
        context = IKGoalContext()
        assert context.position_goals == {}
        assert isinstance(context.position_goals, dict)

    def test_rotation_goals_default_empty(self):
        """Test rotation_goals defaults to empty dict."""
        context = IKGoalContext()
        assert context.rotation_goals == {}
        assert isinstance(context.rotation_goals, dict)

    def test_weights_default_empty(self):
        """Test weights defaults to empty dict."""
        context = IKGoalContext()
        assert context.weights == {}
        assert isinstance(context.weights, dict)

    def test_pole_vectors_default_empty(self):
        """Test pole_vectors defaults to empty dict."""
        context = IKGoalContext()
        assert context.pole_vectors == {}
        assert isinstance(context.pole_vectors, dict)

    def test_chain_assignments_default_empty(self):
        """Test chain_assignments defaults to empty dict."""
        context = IKGoalContext()
        assert context.chain_assignments == {}
        assert isinstance(context.chain_assignments, dict)

    def test_all_fields_independent(self):
        """Test each instance has independent field storage."""
        ctx1 = IKGoalContext()
        ctx2 = IKGoalContext()
        ctx1.position_goals["bone1"] = Vec3(1, 2, 3)
        assert "bone1" not in ctx2.position_goals


class TestIKGoalContextClear:
    """Tests for IKGoalContext.clear() method."""

    def test_clear_empties_position_goals(self):
        """Test clear empties position_goals."""
        context = IKGoalContext()
        context.position_goals["bone1"] = Vec3(1, 2, 3)
        context.clear()
        assert context.position_goals == {}

    def test_clear_empties_rotation_goals(self):
        """Test clear empties rotation_goals."""
        context = IKGoalContext()
        context.rotation_goals["bone1"] = Quat.identity()
        context.clear()
        assert context.rotation_goals == {}

    def test_clear_empties_weights(self):
        """Test clear empties weights."""
        context = IKGoalContext()
        context.weights["bone1"] = 0.5
        context.clear()
        assert context.weights == {}

    def test_clear_empties_pole_vectors(self):
        """Test clear empties pole_vectors."""
        context = IKGoalContext()
        context.pole_vectors["bone1"] = Vec3(0, 1, 0)
        context.clear()
        assert context.pole_vectors == {}

    def test_clear_empties_chain_assignments(self):
        """Test clear empties chain_assignments."""
        context = IKGoalContext()
        context.chain_assignments["bone1"] = "left_arm"
        context.clear()
        assert context.chain_assignments == {}

    def test_clear_all_fields_at_once(self):
        """Test clear empties all fields simultaneously."""
        context = IKGoalContext()
        context.position_goals["bone1"] = Vec3(1, 2, 3)
        context.rotation_goals["bone2"] = Quat.identity()
        context.weights["bone1"] = 0.5
        context.pole_vectors["bone3"] = Vec3(0, 1, 0)
        context.chain_assignments["bone1"] = "arm"
        context.clear()
        assert len(context.position_goals) == 0
        assert len(context.rotation_goals) == 0
        assert len(context.weights) == 0
        assert len(context.pole_vectors) == 0
        assert len(context.chain_assignments) == 0


class TestIKGoalContextSetPositionGoal:
    """Tests for IKGoalContext.set_position_goal() method."""

    def test_set_position_goal_stores_position(self):
        """Test position is stored correctly."""
        context = IKGoalContext()
        pos = Vec3(1.0, 2.0, 3.0)
        context.set_position_goal("LeftHand", pos)
        assert context.position_goals["LeftHand"] == pos

    def test_set_position_goal_default_weight(self):
        """Test default weight is 1.0."""
        context = IKGoalContext()
        context.set_position_goal("LeftHand", Vec3(1, 2, 3))
        assert context.weights["LeftHand"] == 1.0

    def test_set_position_goal_custom_weight(self):
        """Test custom weight is stored."""
        context = IKGoalContext()
        context.set_position_goal("LeftHand", Vec3(1, 2, 3), weight=0.5)
        assert context.weights["LeftHand"] == 0.5

    def test_set_position_goal_clamps_weight_min(self):
        """Test weight is clamped to minimum 0."""
        context = IKGoalContext()
        context.set_position_goal("LeftHand", Vec3(1, 2, 3), weight=-0.5)
        assert context.weights["LeftHand"] == 0.0

    def test_set_position_goal_clamps_weight_max(self):
        """Test weight is clamped to maximum 1."""
        context = IKGoalContext()
        context.set_position_goal("LeftHand", Vec3(1, 2, 3), weight=1.5)
        assert context.weights["LeftHand"] == 1.0

    def test_set_position_goal_with_chain_type(self):
        """Test chain_type is stored correctly."""
        context = IKGoalContext()
        context.set_position_goal("LeftHand", Vec3(1, 2, 3), chain_type="left_arm")
        assert context.chain_assignments["LeftHand"] == "left_arm"

    def test_set_position_goal_without_chain_type(self):
        """Test no chain assignment when chain_type is None."""
        context = IKGoalContext()
        context.set_position_goal("LeftHand", Vec3(1, 2, 3))
        assert "LeftHand" not in context.chain_assignments


class TestIKGoalContextSetRotationGoal:
    """Tests for IKGoalContext.set_rotation_goal() method."""

    def test_set_rotation_goal_stores_rotation(self):
        """Test rotation is stored correctly."""
        context = IKGoalContext()
        rot = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 4)
        context.set_rotation_goal("Head", rot)
        assert context.rotation_goals["Head"] == rot

    def test_set_rotation_goal_default_weight(self):
        """Test default weight is 1.0."""
        context = IKGoalContext()
        context.set_rotation_goal("Head", Quat.identity())
        assert context.weights["Head"] == 1.0

    def test_set_rotation_goal_custom_weight(self):
        """Test custom weight is stored."""
        context = IKGoalContext()
        context.set_rotation_goal("Head", Quat.identity(), weight=0.7)
        assert context.weights["Head"] == 0.7

    def test_set_rotation_goal_clamps_weight_min(self):
        """Test weight is clamped to minimum 0."""
        context = IKGoalContext()
        context.set_rotation_goal("Head", Quat.identity(), weight=-1.0)
        assert context.weights["Head"] == 0.0

    def test_set_rotation_goal_clamps_weight_max(self):
        """Test weight is clamped to maximum 1."""
        context = IKGoalContext()
        context.set_rotation_goal("Head", Quat.identity(), weight=2.0)
        assert context.weights["Head"] == 1.0


class TestIKGoalContextSetPoleVector:
    """Tests for IKGoalContext.set_pole_vector() method."""

    def test_set_pole_vector_stores_position(self):
        """Test pole vector position is stored correctly."""
        context = IKGoalContext()
        pole = Vec3(0.0, 1.0, 0.5)
        context.set_pole_vector("LeftHand", pole)
        assert context.pole_vectors["LeftHand"] == pole

    def test_set_pole_vector_multiple_bones(self):
        """Test multiple pole vectors can be set."""
        context = IKGoalContext()
        context.set_pole_vector("LeftHand", Vec3(0, 1, 0))
        context.set_pole_vector("RightHand", Vec3(0, 1, 0))
        assert len(context.pole_vectors) == 2


class TestIKGoalContextGetWeight:
    """Tests for IKGoalContext.get_weight() method."""

    def test_get_weight_returns_stored_weight(self):
        """Test returns stored weight for bone."""
        context = IKGoalContext()
        context.weights["LeftHand"] = 0.75
        assert context.get_weight("LeftHand") == 0.75

    def test_get_weight_default_for_missing(self):
        """Test returns default 1.0 for missing bone."""
        context = IKGoalContext()
        assert context.get_weight("NonExistent") == 1.0

    def test_get_weight_custom_default(self):
        """Test returns custom default for missing bone."""
        context = IKGoalContext()
        assert context.get_weight("NonExistent", default=0.5) == 0.5


class TestIKGoalContextHasGoal:
    """Tests for IKGoalContext.has_goal() method."""

    def test_has_goal_true_for_position(self):
        """Test returns True when position goal exists."""
        context = IKGoalContext()
        context.position_goals["LeftHand"] = Vec3(1, 2, 3)
        assert context.has_goal("LeftHand") is True

    def test_has_goal_true_for_rotation(self):
        """Test returns True when rotation goal exists."""
        context = IKGoalContext()
        context.rotation_goals["Head"] = Quat.identity()
        assert context.has_goal("Head") is True

    def test_has_goal_true_for_both(self):
        """Test returns True when both goals exist."""
        context = IKGoalContext()
        context.position_goals["LeftHand"] = Vec3(1, 2, 3)
        context.rotation_goals["LeftHand"] = Quat.identity()
        assert context.has_goal("LeftHand") is True

    def test_has_goal_false_for_missing(self):
        """Test returns False for missing bone."""
        context = IKGoalContext()
        assert context.has_goal("NonExistent") is False


# =============================================================================
# Test IKLayerResult Dataclass
# =============================================================================


class TestIKLayerResultFields:
    """Tests for IKLayerResult dataclass fields and defaults."""

    def test_transforms_default_empty(self):
        """Test transforms defaults to empty list."""
        result = IKLayerResult()
        assert result.transforms == []
        assert isinstance(result.transforms, list)

    def test_success_default_true(self):
        """Test success defaults to True."""
        result = IKLayerResult()
        assert result.success is True

    def test_errors_default_empty(self):
        """Test errors defaults to empty dict."""
        result = IKLayerResult()
        assert result.errors == {}
        assert isinstance(result.errors, dict)

    def test_blend_weight_default_one(self):
        """Test blend_weight defaults to 1.0."""
        result = IKLayerResult()
        assert result.blend_weight == 1.0

    def test_custom_field_values(self):
        """Test result can be created with custom values."""
        transforms = create_transforms(3)
        result = IKLayerResult(
            transforms=transforms,
            success=False,
            errors={"bone1": 0.01},
            blend_weight=0.5
        )
        assert len(result.transforms) == 3
        assert result.success is False
        assert result.errors["bone1"] == 0.01
        assert result.blend_weight == 0.5


# =============================================================================
# Test IKLayer Initialization
# =============================================================================


class TestIKLayerInit:
    """Tests for IKLayer.__init__() method."""

    def test_name_stored(self):
        """Test name is stored correctly."""
        layer = IKLayer(name="foot_ik")
        assert layer.name == "foot_ik"

    def test_solver_none_by_default(self):
        """Test solver defaults to None."""
        layer = IKLayer(name="test")
        assert layer.solver is None

    def test_solver_stored_when_provided(self):
        """Test solver is stored when provided."""
        solver = create_mock_solver()
        layer = IKLayer(name="test", solver=solver)
        assert layer.solver is solver

    def test_blend_mode_default_blend(self):
        """Test blend_mode defaults to BLEND."""
        layer = IKLayer(name="test")
        assert layer.blend_mode == IKBlendMode.BLEND

    def test_blend_mode_override(self):
        """Test blend_mode can be set to OVERRIDE."""
        layer = IKLayer(name="test", blend_mode=IKBlendMode.OVERRIDE)
        assert layer.blend_mode == IKBlendMode.OVERRIDE

    def test_weight_default_one(self):
        """Test weight defaults to 1.0."""
        layer = IKLayer(name="test")
        assert layer.weight == 1.0

    def test_weight_clamped_min(self):
        """Test weight is clamped to minimum 0."""
        layer = IKLayer(name="test", weight=-0.5)
        assert layer.weight == 0.0

    def test_weight_clamped_max(self):
        """Test weight is clamped to maximum 1."""
        layer = IKLayer(name="test", weight=1.5)
        assert layer.weight == 1.0

    def test_enabled_default_true(self):
        """Test enabled defaults to True."""
        layer = IKLayer(name="test")
        assert layer.enabled is True

    def test_enabled_false_when_set(self):
        """Test enabled can be set to False."""
        layer = IKLayer(name="test", enabled=False)
        assert layer.enabled is False

    def test_internal_goal_context_created(self):
        """Test internal goal context is created."""
        layer = IKLayer(name="test")
        assert isinstance(layer._goal_context, IKGoalContext)

    def test_internal_bone_mapping_empty(self):
        """Test internal bone mapping is empty initially."""
        layer = IKLayer(name="test")
        assert layer._bone_name_to_index == {}

    def test_internal_cached_goals_empty(self):
        """Test internal cached goals is empty initially."""
        layer = IKLayer(name="test")
        assert layer._cached_goals == []

    def test_internal_last_result_none(self):
        """Test internal last result is None initially."""
        layer = IKLayer(name="test")
        assert layer._last_result is None

    def test_weight_blend_speed_default(self):
        """Test weight blend speed has default value."""
        layer = IKLayer(name="test")
        assert layer._weight_blend_speed == 5.0


# =============================================================================
# Test IKLayer Solver Methods
# =============================================================================


class TestIKLayerSetSolver:
    """Tests for IKLayer.set_solver() method."""

    def test_set_solver_stores_solver(self):
        """Test solver is stored correctly."""
        layer = IKLayer(name="test")
        solver = create_mock_solver()
        layer.set_solver(solver)
        assert layer.solver is solver

    def test_set_solver_clears_cached_goals(self):
        """Test setting solver clears cached goals."""
        layer = IKLayer(name="test")
        layer._cached_goals = ["goal1", "goal2"]
        solver = create_mock_solver()
        layer.set_solver(solver)
        assert layer._cached_goals == []

    def test_set_solver_replaces_existing(self):
        """Test setting solver replaces existing one."""
        solver1 = create_mock_solver("Solver1")
        solver2 = create_mock_solver("Solver2")
        layer = IKLayer(name="test", solver=solver1)
        layer.set_solver(solver2)
        assert layer.solver is solver2


class TestIKLayerGetSolver:
    """Tests for IKLayer.get_solver() method."""

    def test_get_solver_returns_none_initially(self):
        """Test returns None when no solver set."""
        layer = IKLayer(name="test")
        assert layer.get_solver() is None

    def test_get_solver_returns_solver(self):
        """Test returns solver when set."""
        solver = create_mock_solver()
        layer = IKLayer(name="test", solver=solver)
        assert layer.get_solver() is solver


# =============================================================================
# Test IKLayer Goal Methods
# =============================================================================


class TestIKLayerUpdateGoals:
    """Tests for IKLayer.update_goals() method."""

    def test_update_goals_copies_position_goals(self):
        """Test position goals are copied."""
        layer = IKLayer(name="test")
        context = IKGoalContext()
        context.position_goals["bone1"] = Vec3(1, 2, 3)

        layer.update_goals(context)
        assert "bone1" in layer._goal_context.position_goals

    def test_update_goals_copies_rotation_goals(self):
        """Test rotation goals are copied."""
        layer = IKLayer(name="test")
        context = IKGoalContext()
        context.rotation_goals["bone1"] = Quat.identity()

        layer.update_goals(context)
        assert "bone1" in layer._goal_context.rotation_goals

    def test_update_goals_copies_weights(self):
        """Test weights are copied."""
        layer = IKLayer(name="test")
        context = IKGoalContext()
        context.weights["bone1"] = 0.5

        layer.update_goals(context)
        assert layer._goal_context.weights["bone1"] == 0.5

    def test_update_goals_copies_pole_vectors(self):
        """Test pole vectors are copied."""
        layer = IKLayer(name="test")
        context = IKGoalContext()
        context.pole_vectors["bone1"] = Vec3(0, 1, 0)

        layer.update_goals(context)
        assert "bone1" in layer._goal_context.pole_vectors

    def test_update_goals_copies_chain_assignments(self):
        """Test chain assignments are copied."""
        layer = IKLayer(name="test")
        context = IKGoalContext()
        context.chain_assignments["bone1"] = "left_arm"

        layer.update_goals(context)
        assert layer._goal_context.chain_assignments["bone1"] == "left_arm"

    def test_update_goals_creates_independent_copy(self):
        """Test updating context does not affect layer's copy."""
        layer = IKLayer(name="test")
        context = IKGoalContext()
        context.position_goals["bone1"] = Vec3(1, 2, 3)
        layer.update_goals(context)

        # Modify original
        context.position_goals["bone1"] = Vec3(9, 9, 9)
        # Layer's copy should be unchanged
        assert layer._goal_context.position_goals["bone1"].x == 1.0


class TestIKLayerClearGoals:
    """Tests for IKLayer.clear_goals() method."""

    def test_clear_goals_empties_context(self):
        """Test clear_goals clears internal goal context."""
        layer = IKLayer(name="test")
        layer._goal_context.position_goals["bone1"] = Vec3(1, 2, 3)
        layer._goal_context.rotation_goals["bone2"] = Quat.identity()

        layer.clear_goals()

        assert len(layer._goal_context.position_goals) == 0
        assert len(layer._goal_context.rotation_goals) == 0


class TestIKLayerSetPositionGoal:
    """Tests for IKLayer.set_position_goal() method."""

    def test_set_position_goal_stores_in_context(self):
        """Test position goal is stored in internal context."""
        layer = IKLayer(name="test")
        layer.set_position_goal("LeftHand", Vec3(1, 2, 3))
        assert "LeftHand" in layer._goal_context.position_goals

    def test_set_position_goal_with_weight(self):
        """Test position goal with custom weight."""
        layer = IKLayer(name="test")
        layer.set_position_goal("LeftHand", Vec3(1, 2, 3), weight=0.5)
        assert layer._goal_context.weights["LeftHand"] == 0.5

    def test_set_position_goal_with_chain_type(self):
        """Test position goal with chain type."""
        layer = IKLayer(name="test")
        layer.set_position_goal("LeftHand", Vec3(1, 2, 3), chain_type="left_arm")
        assert layer._goal_context.chain_assignments["LeftHand"] == "left_arm"


class TestIKLayerSetRotationGoal:
    """Tests for IKLayer.set_rotation_goal() method."""

    def test_set_rotation_goal_stores_in_context(self):
        """Test rotation goal is stored in internal context."""
        layer = IKLayer(name="test")
        layer.set_rotation_goal("Head", Quat.identity())
        assert "Head" in layer._goal_context.rotation_goals

    def test_set_rotation_goal_with_weight(self):
        """Test rotation goal with custom weight."""
        layer = IKLayer(name="test")
        layer.set_rotation_goal("Head", Quat.identity(), weight=0.3)
        assert layer._goal_context.weights["Head"] == 0.3


class TestIKLayerSetBoneMapping:
    """Tests for IKLayer.set_bone_mapping() method."""

    def test_set_bone_mapping_stores_mapping(self):
        """Test bone mapping is stored."""
        layer = IKLayer(name="test")
        mapping = {"bone1": 0, "bone2": 1, "bone3": 2}
        layer.set_bone_mapping(mapping)
        assert layer._bone_name_to_index == mapping

    def test_set_bone_mapping_creates_copy(self):
        """Test mapping is copied, not referenced."""
        layer = IKLayer(name="test")
        mapping = {"bone1": 0}
        layer.set_bone_mapping(mapping)
        mapping["bone2"] = 1
        assert "bone2" not in layer._bone_name_to_index


# =============================================================================
# Test IKLayer Apply Method
# =============================================================================


class TestIKLayerApply:
    """Tests for IKLayer.apply() method."""

    def test_apply_returns_input_when_disabled(self):
        """Test apply returns input transforms when layer is disabled."""
        layer = IKLayer(name="test", enabled=False)
        transforms = create_transforms(5)
        result = layer.apply(transforms, dt=0.016)
        assert result is transforms

    def test_apply_returns_input_when_no_solver(self):
        """Test apply returns input transforms when no solver is set."""
        layer = IKLayer(name="test", solver=None)
        transforms = create_transforms(5)
        result = layer.apply(transforms, dt=0.016)
        assert result is transforms

    def test_apply_returns_input_when_zero_weight(self):
        """Test apply returns input transforms when weight is zero."""
        solver = create_mock_solver()
        layer = IKLayer(name="test", solver=solver, weight=0.0)
        transforms = create_transforms(5)
        result = layer.apply(transforms, dt=0.016)
        assert result is transforms

    def test_apply_calls_solve_ik(self):
        """Test apply calls internal solve method."""
        solver = create_mock_solver()
        layer = IKLayer(name="test", solver=solver)
        transforms = create_transforms(5)

        layer.apply(transforms, dt=0.016)
        assert solver.solve.called

    def test_apply_caches_result(self):
        """Test apply caches the result."""
        solver = create_mock_solver()
        layer = IKLayer(name="test", solver=solver)
        transforms = create_transforms(5)

        layer.apply(transforms, dt=0.016)
        assert layer._last_result is not None
        assert isinstance(layer._last_result, IKLayerResult)

    def test_apply_updates_weight_smoothly(self):
        """Test apply updates weight towards target."""
        solver = create_mock_solver()
        layer = IKLayer(name="test", solver=solver, weight=0.5)
        layer._target_weight = 1.0
        transforms = create_transforms(5)

        initial_weight = layer.weight
        layer.apply(transforms, dt=0.5)
        # Weight should have moved towards target
        assert layer.weight > initial_weight


class TestIKLayerUpdateWeight:
    """Tests for IKLayer._update_weight() internal method."""

    def test_update_weight_moves_towards_target(self):
        """Test weight moves towards target."""
        layer = IKLayer(name="test", weight=0.0)
        layer._target_weight = 1.0
        layer._update_weight(dt=0.5)
        assert layer.weight > 0.0

    def test_update_weight_no_change_when_at_target(self):
        """Test weight does not change when at target."""
        layer = IKLayer(name="test", weight=0.5)
        layer._target_weight = 0.5
        layer._update_weight(dt=0.5)
        assert abs(layer.weight - 0.5) < 0.001

    def test_update_weight_blend_speed_affects_rate(self):
        """Test blend speed affects rate of change."""
        layer1 = IKLayer(name="test1", weight=0.0)
        layer1._target_weight = 1.0
        layer1._weight_blend_speed = 1.0

        layer2 = IKLayer(name="test2", weight=0.0)
        layer2._target_weight = 1.0
        layer2._weight_blend_speed = 10.0

        layer1._update_weight(dt=0.1)
        layer2._update_weight(dt=0.1)

        assert layer2.weight > layer1.weight


# =============================================================================
# Test IKLayer Blend Modes
# =============================================================================


class TestIKLayerBlendTransforms:
    """Tests for IKLayer._blend_transforms() internal method."""

    def test_blend_override_full_weight(self):
        """Test OVERRIDE mode with full weight returns IK transforms."""
        layer = IKLayer(name="test", blend_mode=IKBlendMode.OVERRIDE)
        input_tf = create_transforms(3)
        ik_tf = create_ik_transforms(3)

        result = layer._blend_transforms(input_tf, ik_tf, weight=1.0)
        assert result == ik_tf

    def test_blend_override_partial_weight(self):
        """Test OVERRIDE mode with partial weight lerps transforms."""
        layer = IKLayer(name="test", blend_mode=IKBlendMode.OVERRIDE)
        input_tf = [Transform(Vec3(0, 0, 0), Quat.identity())]
        ik_tf = [Transform(Vec3(1, 1, 1), Quat.identity())]

        result = layer._blend_transforms(input_tf, ik_tf, weight=0.5)
        # Should be halfway between input and IK
        assert abs(result[0].translation.x - 0.5) < 0.01

    def test_blend_additive_applies_delta(self):
        """Test ADDITIVE mode applies delta to input."""
        layer = IKLayer(name="test", blend_mode=IKBlendMode.ADDITIVE)
        input_tf = [Transform(Vec3(0, 0, 0), Quat.identity())]
        ik_tf = [Transform(Vec3(1, 0, 0), Quat.identity())]

        result = layer._blend_transforms(input_tf, ik_tf, weight=1.0)
        # Delta is (1,0,0), applied to input (0,0,0) = (1,0,0)
        assert abs(result[0].translation.x - 1.0) < 0.01

    def test_blend_additive_weighted_delta(self):
        """Test ADDITIVE mode with weighted delta."""
        layer = IKLayer(name="test", blend_mode=IKBlendMode.ADDITIVE)
        input_tf = [Transform(Vec3(0, 0, 0), Quat.identity())]
        ik_tf = [Transform(Vec3(2, 0, 0), Quat.identity())]

        result = layer._blend_transforms(input_tf, ik_tf, weight=0.5)
        # Delta is (2,0,0) * 0.5 = (1,0,0)
        assert abs(result[0].translation.x - 1.0) < 0.01

    def test_blend_mode_lerps_transforms(self):
        """Test BLEND mode lerps between input and IK."""
        layer = IKLayer(name="test", blend_mode=IKBlendMode.BLEND)
        input_tf = [Transform(Vec3(0, 0, 0), Quat.identity())]
        ik_tf = [Transform(Vec3(2, 2, 2), Quat.identity())]

        result = layer._blend_transforms(input_tf, ik_tf, weight=0.5)
        assert abs(result[0].translation.x - 1.0) < 0.01
        assert abs(result[0].translation.y - 1.0) < 0.01

    def test_blend_handles_mismatched_lengths(self):
        """Test blend handles IK result shorter than input."""
        layer = IKLayer(name="test", blend_mode=IKBlendMode.BLEND)
        input_tf = create_transforms(5)
        ik_tf = create_ik_transforms(3)

        result = layer._blend_transforms(input_tf, ik_tf, weight=1.0)
        # First 3 should be blended, last 2 from input
        assert len(result) == 5


class TestIKLayerComputeAdditiveDelta:
    """Tests for IKLayer._compute_additive_delta() internal method."""

    def test_compute_delta_position(self):
        """Test position delta is computed correctly."""
        layer = IKLayer(name="test")
        input_tf = Transform(Vec3(1, 2, 3), Quat.identity())
        ik_tf = Transform(Vec3(2, 4, 6), Quat.identity())

        delta = layer._compute_additive_delta(input_tf, ik_tf)
        assert abs(delta.translation.x - 1.0) < 0.01
        assert abs(delta.translation.y - 2.0) < 0.01
        assert abs(delta.translation.z - 3.0) < 0.01

    def test_compute_delta_rotation(self):
        """Test rotation delta is computed."""
        layer = IKLayer(name="test")
        input_tf = Transform(Vec3.zero(), Quat.identity())
        ik_tf = Transform(Vec3.zero(), Quat.from_axis_angle(Vec3(0, 1, 0), math.pi / 4))

        delta = layer._compute_additive_delta(input_tf, ik_tf)
        # Delta should be approximately the rotation applied
        assert delta.rotation is not None

    def test_compute_delta_scale_ratio(self):
        """Test scale delta is computed as ratio."""
        layer = IKLayer(name="test")
        input_tf = Transform(Vec3.zero(), Quat.identity(), Vec3(1, 1, 1))
        ik_tf = Transform(Vec3.zero(), Quat.identity(), Vec3(2, 2, 2))

        delta = layer._compute_additive_delta(input_tf, ik_tf)
        assert abs(delta.scale.x - 2.0) < 0.01
        assert abs(delta.scale.y - 2.0) < 0.01

    def test_compute_delta_handles_zero_scale(self):
        """Test handles zero scale gracefully."""
        layer = IKLayer(name="test")
        input_tf = Transform(Vec3.zero(), Quat.identity(), Vec3(0, 0, 0))
        ik_tf = Transform(Vec3.zero(), Quat.identity(), Vec3(1, 1, 1))

        delta = layer._compute_additive_delta(input_tf, ik_tf)
        # Should use 1.0 for zero scale components
        assert delta.scale.x == 1.0


# =============================================================================
# Test IKLayer Weight and Blend Mode Setters
# =============================================================================


class TestIKLayerSetWeight:
    """Tests for IKLayer.set_weight() method."""

    def test_set_weight_updates_target(self):
        """Test set_weight updates target weight."""
        layer = IKLayer(name="test")
        layer.set_weight(0.5)
        assert layer._target_weight == 0.5

    def test_set_weight_immediate_updates_current(self):
        """Test immediate=True updates current weight."""
        layer = IKLayer(name="test", weight=0.0)
        layer.set_weight(0.8, immediate=True)
        assert layer.weight == 0.8

    def test_set_weight_clamps_value(self):
        """Test weight is clamped to 0-1 range."""
        layer = IKLayer(name="test")
        layer.set_weight(1.5, immediate=True)
        assert layer.weight == 1.0

        layer.set_weight(-0.5, immediate=True)
        assert layer.weight == 0.0


class TestIKLayerGetWeight:
    """Tests for IKLayer.get_weight() method."""

    def test_get_weight_returns_current(self):
        """Test get_weight returns current weight."""
        layer = IKLayer(name="test", weight=0.7)
        assert layer.get_weight() == 0.7


class TestIKLayerSetBlendMode:
    """Tests for IKLayer.set_blend_mode() method."""

    def test_set_blend_mode_override(self):
        """Test setting blend mode to OVERRIDE."""
        layer = IKLayer(name="test")
        layer.set_blend_mode(IKBlendMode.OVERRIDE)
        assert layer.blend_mode == IKBlendMode.OVERRIDE

    def test_set_blend_mode_additive(self):
        """Test setting blend mode to ADDITIVE."""
        layer = IKLayer(name="test")
        layer.set_blend_mode(IKBlendMode.ADDITIVE)
        assert layer.blend_mode == IKBlendMode.ADDITIVE

    def test_set_blend_mode_blend(self):
        """Test setting blend mode to BLEND."""
        layer = IKLayer(name="test", blend_mode=IKBlendMode.OVERRIDE)
        layer.set_blend_mode(IKBlendMode.BLEND)
        assert layer.blend_mode == IKBlendMode.BLEND


class TestIKLayerGetBlendMode:
    """Tests for IKLayer.get_blend_mode() method."""

    def test_get_blend_mode_returns_current(self):
        """Test get_blend_mode returns current mode."""
        layer = IKLayer(name="test", blend_mode=IKBlendMode.ADDITIVE)
        assert layer.get_blend_mode() == IKBlendMode.ADDITIVE


# =============================================================================
# Test IKLayer Enable/Disable
# =============================================================================


class TestIKLayerSetEnabled:
    """Tests for IKLayer.set_enabled() method."""

    def test_set_enabled_true(self):
        """Test enabling layer."""
        layer = IKLayer(name="test", enabled=False)
        layer.set_enabled(True)
        assert layer.enabled is True

    def test_set_enabled_false(self):
        """Test disabling layer."""
        layer = IKLayer(name="test", enabled=True)
        layer.set_enabled(False)
        assert layer.enabled is False


class TestIKLayerIsEnabled:
    """Tests for IKLayer.is_enabled() method."""

    def test_is_enabled_when_true(self):
        """Test is_enabled returns True when enabled."""
        layer = IKLayer(name="test", enabled=True)
        assert layer.is_enabled() is True

    def test_is_enabled_when_false(self):
        """Test is_enabled returns False when disabled."""
        layer = IKLayer(name="test", enabled=False)
        assert layer.is_enabled() is False


# =============================================================================
# Test IKLayer Utility Methods
# =============================================================================


class TestIKLayerSetWeightBlendSpeed:
    """Tests for IKLayer.set_weight_blend_speed() method."""

    def test_set_weight_blend_speed(self):
        """Test setting weight blend speed."""
        layer = IKLayer(name="test")
        layer.set_weight_blend_speed(10.0)
        assert layer._weight_blend_speed == 10.0

    def test_set_weight_blend_speed_minimum(self):
        """Test blend speed has minimum value."""
        layer = IKLayer(name="test")
        layer.set_weight_blend_speed(0.01)
        assert layer._weight_blend_speed == 0.1


class TestIKLayerGetLastResult:
    """Tests for IKLayer.get_last_result() method."""

    def test_get_last_result_none_initially(self):
        """Test returns None before any apply."""
        layer = IKLayer(name="test")
        assert layer.get_last_result() is None

    def test_get_last_result_after_apply(self):
        """Test returns result after apply."""
        solver = create_mock_solver()
        layer = IKLayer(name="test", solver=solver)
        transforms = create_transforms(5)
        layer.apply(transforms, dt=0.016)

        result = layer.get_last_result()
        assert result is not None
        assert isinstance(result, IKLayerResult)


class TestIKLayerGetGoalContext:
    """Tests for IKLayer.get_goal_context() method."""

    def test_get_goal_context_returns_context(self):
        """Test returns internal goal context."""
        layer = IKLayer(name="test")
        layer.set_position_goal("bone1", Vec3(1, 2, 3))

        context = layer.get_goal_context()
        assert isinstance(context, IKGoalContext)
        assert "bone1" in context.position_goals


# =============================================================================
# Test IKLayerStack
# =============================================================================


class TestIKLayerStackInit:
    """Tests for IKLayerStack.__init__() method."""

    def test_empty_initially(self):
        """Test stack is empty initially."""
        stack = IKLayerStack()
        assert stack.layer_count() == 0

    def test_internal_layers_list_empty(self):
        """Test internal layers list is empty."""
        stack = IKLayerStack()
        assert stack._layers == []

    def test_internal_layer_by_name_empty(self):
        """Test internal layer_by_name dict is empty."""
        stack = IKLayerStack()
        assert stack._layer_by_name == {}


class TestIKLayerStackAddLayer:
    """Tests for IKLayerStack.add_layer() method."""

    def test_add_layer_increases_count(self):
        """Test adding layer increases count."""
        stack = IKLayerStack()
        layer = IKLayer(name="test")
        stack.add_layer(layer)
        assert stack.layer_count() == 1

    def test_add_layer_returns_index(self):
        """Test add_layer returns insertion index."""
        stack = IKLayerStack()
        layer = IKLayer(name="test")
        index = stack.add_layer(layer)
        assert index == 0

    def test_add_layer_at_specific_index(self):
        """Test adding layer at specific index."""
        stack = IKLayerStack()
        layer1 = IKLayer(name="layer1")
        layer2 = IKLayer(name="layer2")
        layer3 = IKLayer(name="layer3")

        stack.add_layer(layer1)
        stack.add_layer(layer2)
        index = stack.add_layer(layer3, index=1)

        assert index == 1
        assert stack.get_layer_by_index(1) is layer3

    def test_add_layer_raises_on_duplicate_name(self):
        """Test adding layer with duplicate name raises ValueError."""
        stack = IKLayerStack()
        layer1 = IKLayer(name="test")
        layer2 = IKLayer(name="test")

        stack.add_layer(layer1)
        with pytest.raises(ValueError, match="already exists"):
            stack.add_layer(layer2)

    def test_add_layer_stores_in_name_dict(self):
        """Test layer is stored in name lookup dict."""
        stack = IKLayerStack()
        layer = IKLayer(name="foot_ik")
        stack.add_layer(layer)
        assert stack.get_layer("foot_ik") is layer


class TestIKLayerStackRemoveLayer:
    """Tests for IKLayerStack.remove_layer() method."""

    def test_remove_layer_decreases_count(self):
        """Test removing layer decreases count."""
        stack = IKLayerStack()
        layer = IKLayer(name="test")
        stack.add_layer(layer)
        stack.remove_layer("test")
        assert stack.layer_count() == 0

    def test_remove_layer_returns_true(self):
        """Test returns True when layer is removed."""
        stack = IKLayerStack()
        layer = IKLayer(name="test")
        stack.add_layer(layer)
        result = stack.remove_layer("test")
        assert result is True

    def test_remove_layer_returns_false_when_missing(self):
        """Test returns False when layer not found."""
        stack = IKLayerStack()
        result = stack.remove_layer("nonexistent")
        assert result is False

    def test_remove_layer_removes_from_name_dict(self):
        """Test layer is removed from name lookup dict."""
        stack = IKLayerStack()
        layer = IKLayer(name="test")
        stack.add_layer(layer)
        stack.remove_layer("test")
        assert stack.get_layer("test") is None


class TestIKLayerStackGetLayer:
    """Tests for IKLayerStack.get_layer() method."""

    def test_get_layer_returns_layer(self):
        """Test get_layer returns layer by name."""
        stack = IKLayerStack()
        layer = IKLayer(name="foot_ik")
        stack.add_layer(layer)
        assert stack.get_layer("foot_ik") is layer

    def test_get_layer_returns_none_when_missing(self):
        """Test get_layer returns None when not found."""
        stack = IKLayerStack()
        assert stack.get_layer("nonexistent") is None


class TestIKLayerStackGetLayerByIndex:
    """Tests for IKLayerStack.get_layer_by_index() method."""

    def test_get_layer_by_index_returns_layer(self):
        """Test get_layer_by_index returns correct layer."""
        stack = IKLayerStack()
        layer1 = IKLayer(name="layer1")
        layer2 = IKLayer(name="layer2")
        stack.add_layer(layer1)
        stack.add_layer(layer2)

        assert stack.get_layer_by_index(0) is layer1
        assert stack.get_layer_by_index(1) is layer2

    def test_get_layer_by_index_returns_none_out_of_range(self):
        """Test returns None when index out of range."""
        stack = IKLayerStack()
        assert stack.get_layer_by_index(0) is None
        assert stack.get_layer_by_index(-1) is None


class TestIKLayerStackApply:
    """Tests for IKLayerStack.apply() method."""

    def test_apply_returns_input_when_empty(self):
        """Test apply returns input when stack is empty."""
        stack = IKLayerStack()
        transforms = create_transforms(5)
        result = stack.apply(transforms, dt=0.016)
        assert result is transforms

    def test_apply_chains_layers(self):
        """Test apply chains layers in order."""
        stack = IKLayerStack()

        solver1 = create_mock_solver()
        solver2 = create_mock_solver()

        layer1 = IKLayer(name="layer1", solver=solver1)
        layer2 = IKLayer(name="layer2", solver=solver2)

        stack.add_layer(layer1)
        stack.add_layer(layer2)

        transforms = create_transforms(5)
        stack.apply(transforms, dt=0.016)

        assert solver1.solve.called
        assert solver2.solve.called

    def test_apply_skips_disabled_layers(self):
        """Test apply skips disabled layers."""
        stack = IKLayerStack()

        solver = create_mock_solver()
        layer = IKLayer(name="test", solver=solver, enabled=False)
        stack.add_layer(layer)

        transforms = create_transforms(5)
        stack.apply(transforms, dt=0.016)

        assert not solver.solve.called


class TestIKLayerStackSetAllWeights:
    """Tests for IKLayerStack.set_all_weights() method."""

    def test_set_all_weights_updates_all_layers(self):
        """Test sets weight on all layers."""
        stack = IKLayerStack()
        layer1 = IKLayer(name="layer1", weight=1.0)
        layer2 = IKLayer(name="layer2", weight=1.0)
        stack.add_layer(layer1)
        stack.add_layer(layer2)

        stack.set_all_weights(0.5, immediate=True)

        assert layer1.weight == 0.5
        assert layer2.weight == 0.5


class TestIKLayerStackDisableAll:
    """Tests for IKLayerStack.disable_all() method."""

    def test_disable_all_disables_layers(self):
        """Test disables all layers."""
        stack = IKLayerStack()
        layer1 = IKLayer(name="layer1", enabled=True)
        layer2 = IKLayer(name="layer2", enabled=True)
        stack.add_layer(layer1)
        stack.add_layer(layer2)

        stack.disable_all()

        assert layer1.enabled is False
        assert layer2.enabled is False


class TestIKLayerStackEnableAll:
    """Tests for IKLayerStack.enable_all() method."""

    def test_enable_all_enables_layers(self):
        """Test enables all layers."""
        stack = IKLayerStack()
        layer1 = IKLayer(name="layer1", enabled=False)
        layer2 = IKLayer(name="layer2", enabled=False)
        stack.add_layer(layer1)
        stack.add_layer(layer2)

        stack.enable_all()

        assert layer1.enabled is True
        assert layer2.enabled is True


# =============================================================================
# Test IKLayer Solver Dispatch
# =============================================================================


class TestIKLayerSolveIK:
    """Tests for IKLayer._solve_ik() internal method."""

    def test_solve_ik_returns_none_without_solver(self):
        """Test returns None when no solver set."""
        layer = IKLayer(name="test", solver=None)
        result = layer._solve_ik(create_transforms(3), dt=0.016)
        assert result is None

    def test_solve_ik_dispatches_to_fullbody(self):
        """Test dispatches to FullBodyIK solver."""
        solver = create_mock_solver("FullBodyIK")

        layer = IKLayer(name="test", solver=solver)
        # FullBodyIK needs goals to actually call solve
        layer._goal_context.position_goals["bone"] = Vec3(1, 2, 3)
        layer._bone_name_to_index["bone"] = 0
        layer._solve_ik(create_transforms(3), dt=0.016)
        solver.solve.assert_called()

    def test_solve_ik_dispatches_to_foot_placement(self):
        """Test dispatches to FootPlacement solver."""
        solver = create_mock_solver("FootPlacement")

        layer = IKLayer(name="test", solver=solver)
        layer._solve_ik(create_transforms(3), dt=0.016)
        solver.solve.assert_called()

    def test_solve_ik_dispatches_to_two_bone(self):
        """Test dispatches to TwoBoneIK solver."""
        solver = create_mock_solver("TwoBoneIK")

        layer = IKLayer(name="test", solver=solver)
        layer._goal_context.position_goals["bone"] = Vec3(1, 2, 3)
        layer._solve_ik(create_transforms(3), dt=0.016)
        solver.solve.assert_called()

    def test_solve_ik_dispatches_to_fabrik(self):
        """Test dispatches to FABRIKChain solver."""
        solver = create_mock_solver("FABRIKChain")

        layer = IKLayer(name="test", solver=solver)
        layer._goal_context.position_goals["bone"] = Vec3(1, 2, 3)
        layer._solve_ik(create_transforms(3), dt=0.016)
        solver.solve.assert_called()

    def test_solve_ik_fallback_generic_solve(self):
        """Test falls back to generic solve interface."""
        solver = create_mock_solver("CustomSolver")

        layer = IKLayer(name="test", solver=solver)
        result = layer._solve_ik(create_transforms(3), dt=0.016)
        solver.solve.assert_called()


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestIKLayerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_transforms_list(self):
        """Test handling empty transforms list."""
        solver = create_mock_solver()
        layer = IKLayer(name="test", solver=solver)
        result = layer.apply([], dt=0.016)
        # Should not crash, may return empty list
        assert isinstance(result, list)

    def test_very_small_dt(self):
        """Test handling very small delta time."""
        solver = create_mock_solver()
        layer = IKLayer(name="test", solver=solver)
        # Set weight to 0.5 immediately, then target to 1.0
        layer.set_weight(0.5, immediate=True)
        layer._target_weight = 1.0
        transforms = create_transforms(3)

        initial_weight = layer.weight
        # Very small dt should still work but weight should barely move
        layer.apply(transforms, dt=0.0001)
        # Weight should have moved only slightly towards target
        assert layer.weight >= initial_weight  # moved towards 1.0
        assert layer.weight < 0.6  # but not by much

    def test_very_large_dt(self):
        """Test handling very large delta time."""
        solver = create_mock_solver()
        layer = IKLayer(name="test", solver=solver)
        layer.set_weight(0.0, immediate=True)
        layer._target_weight = 1.0
        transforms = create_transforms(3)

        # Large dt should quickly reach target
        layer.apply(transforms, dt=10.0)
        assert layer.weight > 0.9

    def test_multiple_position_goals(self):
        """Test handling multiple position goals."""
        layer = IKLayer(name="test")
        for i in range(10):
            layer.set_position_goal(f"bone{i}", Vec3(i, i, i))

        context = layer.get_goal_context()
        assert len(context.position_goals) == 10

    def test_overwrite_existing_goal(self):
        """Test overwriting existing goal."""
        layer = IKLayer(name="test")
        layer.set_position_goal("bone", Vec3(1, 1, 1))
        layer.set_position_goal("bone", Vec3(2, 2, 2))

        context = layer.get_goal_context()
        assert context.position_goals["bone"].x == 2.0


class TestIKLayerStackEdgeCases:
    """Tests for IKLayerStack edge cases."""

    def test_add_many_layers(self):
        """Test adding many layers."""
        stack = IKLayerStack()
        for i in range(20):
            layer = IKLayer(name=f"layer{i}")
            stack.add_layer(layer)
        assert stack.layer_count() == 20

    def test_remove_from_middle(self):
        """Test removing layer from middle of stack."""
        stack = IKLayerStack()
        for i in range(5):
            stack.add_layer(IKLayer(name=f"layer{i}"))

        stack.remove_layer("layer2")
        assert stack.layer_count() == 4
        assert stack.get_layer("layer2") is None
        # Other layers should still exist
        assert stack.get_layer("layer0") is not None
        assert stack.get_layer("layer4") is not None

    def test_index_clamping_negative(self):
        """Test index is clamped for negative values."""
        stack = IKLayerStack()
        layer1 = IKLayer(name="layer1")
        layer2 = IKLayer(name="layer2")
        stack.add_layer(layer1)
        index = stack.add_layer(layer2, index=-5)
        # Should be inserted at 0
        assert index == 0
        assert stack.get_layer_by_index(0) is layer2

    def test_index_clamping_large(self):
        """Test index is clamped for large values."""
        stack = IKLayerStack()
        layer1 = IKLayer(name="layer1")
        layer2 = IKLayer(name="layer2")
        stack.add_layer(layer1)
        index = stack.add_layer(layer2, index=100)
        # Should be appended at end
        assert index == 1

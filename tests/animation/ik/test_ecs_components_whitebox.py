"""Whitebox tests for ECS Components in Animation IK Systems.

Comprehensive tests for:
- FullBodyIKController component (line 62): fields, methods, decorators
- AnimationGraphController component (line 207): layers, IK config, state
- LookAtTarget component (line 397): targeting, spine distribution, limits
- FootPlacementController component (line 525): placement, terrain, pelvis
- IKTargetComponent component (line 671): goals, weights, chains
- @component decorator application on all classes

Target: 50+ tests covering all fields, methods, edge cases.

Task: T-FB-4.21 ECS Components
"""

from __future__ import annotations

import math
import pytest
from dataclasses import fields, asdict, is_dataclass
from typing import List, Dict, Optional, get_type_hints
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from engine.animation.ik.ecs_components import (
    FullBodyIKController,
    AnimationGraphController,
    LookAtTarget,
    FootPlacementController,
    IKTargetComponent,
)
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.math.transform import Transform


# =============================================================================
# Helper Functions
# =============================================================================


def vec3_approx_equal(v1: Vec3, v2: Vec3, eps: float = 1e-4) -> bool:
    """Check if two Vec3 are approximately equal."""
    return (
        abs(v1.x - v2.x) < eps and
        abs(v1.y - v2.y) < eps and
        abs(v1.z - v2.z) < eps
    )


def quat_approx_equal(q1: Quat, q2: Quat, eps: float = 1e-4) -> bool:
    """Check if two Quat are approximately equal."""
    return (
        abs(q1.x - q2.x) < eps and
        abs(q1.y - q2.y) < eps and
        abs(q1.z - q2.z) < eps and
        abs(q1.w - q2.w) < eps
    )


class MockIKLayer:
    """Mock IK layer for testing AnimationGraphController."""
    def __init__(self, name: str = "test_layer"):
        self.name = name
        self._weight = 1.0
        self._enabled = True

    def set_weight(self, weight: float) -> None:
        self._weight = max(0.0, min(1.0, weight))

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled


class MockFullBodyIK:
    """Mock FullBodyIK solver for testing."""
    def __init__(self):
        self.solve = Mock()


class MockFootPlacement:
    """Mock FootPlacement solver for testing."""
    def __init__(self):
        self.solve = Mock()


class MockLookAtSolver:
    """Mock LookAtSolver for testing."""
    def __init__(self):
        self.solve = Mock()


# =============================================================================
# Test @component Decorator Application
# =============================================================================


class TestComponentDecoratorApplication:
    """Tests for @component decorator on all ECS component classes."""

    def test_fullbodyikcontroller_has_component_attribute(self):
        """Test FullBodyIKController has _component attribute."""
        assert hasattr(FullBodyIKController, "_component")
        assert FullBodyIKController._component is True

    def test_fullbodyikcontroller_has_component_name(self):
        """Test FullBodyIKController has correct component name."""
        assert hasattr(FullBodyIKController, "_component_name")
        assert FullBodyIKController._component_name == "FullBodyIKController"

    def test_animationgraphcontroller_has_component_attribute(self):
        """Test AnimationGraphController has _component attribute."""
        assert hasattr(AnimationGraphController, "_component")
        assert AnimationGraphController._component is True

    def test_animationgraphcontroller_has_component_name(self):
        """Test AnimationGraphController has correct component name."""
        assert hasattr(AnimationGraphController, "_component_name")
        assert AnimationGraphController._component_name == "AnimationGraphController"

    def test_lookattarget_has_component_attribute(self):
        """Test LookAtTarget has _component attribute."""
        assert hasattr(LookAtTarget, "_component")
        assert LookAtTarget._component is True

    def test_lookattarget_has_component_name(self):
        """Test LookAtTarget has correct component name."""
        assert hasattr(LookAtTarget, "_component_name")
        assert LookAtTarget._component_name == "LookAtTarget"

    def test_footplacementcontroller_has_component_attribute(self):
        """Test FootPlacementController has _component attribute."""
        assert hasattr(FootPlacementController, "_component")
        assert FootPlacementController._component is True

    def test_footplacementcontroller_has_component_name(self):
        """Test FootPlacementController has correct component name."""
        assert hasattr(FootPlacementController, "_component_name")
        assert FootPlacementController._component_name == "FootPlacementController"

    def test_iktargetcomponent_has_component_attribute(self):
        """Test IKTargetComponent has _component attribute."""
        assert hasattr(IKTargetComponent, "_component")
        assert IKTargetComponent._component is True

    def test_iktargetcomponent_has_component_name(self):
        """Test IKTargetComponent has correct component name."""
        assert hasattr(IKTargetComponent, "_component_name")
        assert IKTargetComponent._component_name == "IKTargetComponent"


class TestComponentDataclass:
    """Tests for dataclass behavior of all ECS components."""

    def test_fullbodyikcontroller_is_dataclass(self):
        """Test FullBodyIKController is a dataclass."""
        assert is_dataclass(FullBodyIKController)

    def test_animationgraphcontroller_is_dataclass(self):
        """Test AnimationGraphController is a dataclass."""
        assert is_dataclass(AnimationGraphController)

    def test_lookattarget_is_dataclass(self):
        """Test LookAtTarget is a dataclass."""
        assert is_dataclass(LookAtTarget)

    def test_footplacementcontroller_is_dataclass(self):
        """Test FootPlacementController is a dataclass."""
        assert is_dataclass(FootPlacementController)

    def test_iktargetcomponent_is_dataclass(self):
        """Test IKTargetComponent is a dataclass."""
        assert is_dataclass(IKTargetComponent)


# =============================================================================
# Test FullBodyIKController Component
# =============================================================================


class TestFullBodyIKControllerFields:
    """Tests for FullBodyIKController field defaults and types."""

    def test_default_solver_is_none(self):
        """Test solver defaults to None."""
        controller = FullBodyIKController()
        assert controller.solver is None

    def test_default_enabled_is_true(self):
        """Test enabled defaults to True."""
        controller = FullBodyIKController()
        assert controller.enabled is True

    def test_default_weight_is_one(self):
        """Test weight defaults to 1.0."""
        controller = FullBodyIKController()
        assert controller.weight == 1.0

    def test_default_foot_placement_is_none(self):
        """Test foot_placement defaults to None."""
        controller = FullBodyIKController()
        assert controller.foot_placement is None

    def test_default_foot_placement_enabled_is_true(self):
        """Test foot_placement_enabled defaults to True."""
        controller = FullBodyIKController()
        assert controller.foot_placement_enabled is True

    def test_default_foot_height_offset_is_zero(self):
        """Test foot_height_offset defaults to 0.0."""
        controller = FullBodyIKController()
        assert controller.foot_height_offset == 0.0

    def test_default_look_at_solver_is_none(self):
        """Test look_at_solver defaults to None."""
        controller = FullBodyIKController()
        assert controller.look_at_solver is None

    def test_default_look_at_target_is_none(self):
        """Test look_at_target defaults to None."""
        controller = FullBodyIKController()
        assert controller.look_at_target is None

    def test_default_look_at_enabled_is_false(self):
        """Test look_at_enabled defaults to False."""
        controller = FullBodyIKController()
        assert controller.look_at_enabled is False

    def test_default_look_at_weight_is_one(self):
        """Test look_at_weight defaults to 1.0."""
        controller = FullBodyIKController()
        assert controller.look_at_weight == 1.0

    def test_default_look_at_max_angle(self):
        """Test look_at_max_angle defaults to 90.0."""
        controller = FullBodyIKController()
        assert controller.look_at_max_angle == 90.0

    def test_default_ik_goals_is_none(self):
        """Test ik_goals defaults to None."""
        controller = FullBodyIKController()
        assert controller.ik_goals is None

    def test_default_maintain_balance_is_true(self):
        """Test maintain_balance defaults to True."""
        controller = FullBodyIKController()
        assert controller.maintain_balance is True

    def test_default_pelvis_adjust_enabled_is_true(self):
        """Test pelvis_adjust_enabled defaults to True."""
        controller = FullBodyIKController()
        assert controller.pelvis_adjust_enabled is True

    def test_default_max_pelvis_drop(self):
        """Test max_pelvis_drop defaults to 0.5."""
        controller = FullBodyIKController()
        assert controller.max_pelvis_drop == 0.5

    def test_default_solve_order(self):
        """Test solve_order defaults to expected list."""
        controller = FullBodyIKController()
        assert controller.solve_order == ["foot_placement", "spine", "arms", "look_at"]

    def test_default_last_solve_time_is_zero(self):
        """Test _last_solve_time defaults to 0.0."""
        controller = FullBodyIKController()
        assert controller._last_solve_time == 0.0


class TestFullBodyIKControllerMethods:
    """Tests for FullBodyIKController methods."""

    def test_set_enabled_true(self):
        """Test set_enabled with True."""
        controller = FullBodyIKController()
        controller.set_enabled(True)
        assert controller.enabled is True

    def test_set_enabled_false(self):
        """Test set_enabled with False."""
        controller = FullBodyIKController()
        controller.set_enabled(False)
        assert controller.enabled is False

    def test_set_weight_valid_value(self):
        """Test set_weight with valid value."""
        controller = FullBodyIKController()
        controller.set_weight(0.5)
        assert controller.weight == 0.5

    def test_set_weight_clamps_above_one(self):
        """Test set_weight clamps values above 1.0."""
        controller = FullBodyIKController()
        controller.set_weight(1.5)
        assert controller.weight == 1.0

    def test_set_weight_clamps_below_zero(self):
        """Test set_weight clamps values below 0.0."""
        controller = FullBodyIKController()
        controller.set_weight(-0.5)
        assert controller.weight == 0.0

    def test_set_look_at_target_with_position(self):
        """Test set_look_at_target with position."""
        controller = FullBodyIKController()
        target = Vec3(1.0, 2.0, 3.0)
        controller.set_look_at_target(target, weight=0.8)
        assert vec3_approx_equal(controller.look_at_target, target)
        assert controller.look_at_weight == 0.8

    def test_set_look_at_target_with_immediate(self):
        """Test set_look_at_target with immediate flag."""
        controller = FullBodyIKController()
        target = Vec3(1.0, 2.0, 3.0)
        controller.set_look_at_target(target, weight=0.7, immediate=True)
        assert controller.look_at_weight == 0.7

    def test_set_look_at_target_clamps_weight(self):
        """Test set_look_at_target clamps weight."""
        controller = FullBodyIKController()
        target = Vec3(1.0, 2.0, 3.0)
        controller.set_look_at_target(target, weight=1.5)
        assert controller.look_at_weight == 1.0

    def test_set_look_at_target_none_clears_target(self):
        """Test set_look_at_target with None clears target."""
        controller = FullBodyIKController()
        controller.look_at_target = Vec3(1.0, 2.0, 3.0)
        controller.set_look_at_target(None)
        assert controller.look_at_target is None

    def test_clear_look_at(self):
        """Test clear_look_at method."""
        controller = FullBodyIKController()
        controller.look_at_target = Vec3(1.0, 2.0, 3.0)
        controller.look_at_enabled = True
        controller.clear_look_at()
        assert controller.look_at_target is None
        assert controller.look_at_enabled is False

    def test_set_foot_placement_enabled_true(self):
        """Test set_foot_placement_enabled with True."""
        controller = FullBodyIKController()
        controller.set_foot_placement_enabled(True)
        assert controller.foot_placement_enabled is True

    def test_set_foot_placement_enabled_false(self):
        """Test set_foot_placement_enabled with False."""
        controller = FullBodyIKController()
        controller.set_foot_placement_enabled(False)
        assert controller.foot_placement_enabled is False

    def test_reset_restores_defaults(self):
        """Test reset method restores default values."""
        controller = FullBodyIKController()
        controller.weight = 0.5
        controller.look_at_target = Vec3(1.0, 2.0, 3.0)
        controller.look_at_weight = 0.3
        controller.look_at_enabled = True
        controller._last_solve_time = 100.0

        controller.reset()

        assert controller.weight == 1.0
        assert controller.look_at_target is None
        assert controller.look_at_weight == 1.0
        assert controller.look_at_enabled is False
        assert controller._last_solve_time == 0.0


# =============================================================================
# Test AnimationGraphController Component
# =============================================================================


class TestAnimationGraphControllerFields:
    """Tests for AnimationGraphController field defaults and types."""

    def test_default_controller_is_none(self):
        """Test controller defaults to None."""
        controller = AnimationGraphController()
        assert controller.controller is None

    def test_default_enabled_is_true(self):
        """Test enabled defaults to True."""
        controller = AnimationGraphController()
        assert controller.enabled is True

    def test_default_graph_is_none(self):
        """Test graph defaults to None."""
        controller = AnimationGraphController()
        assert controller.graph is None

    def test_default_layers_is_empty(self):
        """Test layers defaults to empty list."""
        controller = AnimationGraphController()
        assert controller.layers == []
        assert isinstance(controller.layers, list)

    def test_default_current_state_is_empty(self):
        """Test current_state defaults to empty string."""
        controller = AnimationGraphController()
        assert controller.current_state == ""

    def test_default_blend_time_is_zero(self):
        """Test blend_time defaults to 0.0."""
        controller = AnimationGraphController()
        assert controller.blend_time == 0.0

    def test_default_time_scale_is_one(self):
        """Test time_scale defaults to 1.0."""
        controller = AnimationGraphController()
        assert controller.time_scale == 1.0

    def test_default_output_transforms_is_empty(self):
        """Test output_transforms defaults to empty list."""
        controller = AnimationGraphController()
        assert controller.output_transforms == []

    def test_default_ik_weight_is_one(self):
        """Test ik_weight defaults to 1.0."""
        controller = AnimationGraphController()
        assert controller.ik_weight == 1.0

    def test_default_solve_order(self):
        """Test solve_order defaults to 'foot_first'."""
        controller = AnimationGraphController()
        assert controller.solve_order == "foot_first"

    def test_default_custom_solve_order_is_empty(self):
        """Test custom_solve_order defaults to empty list."""
        controller = AnimationGraphController()
        assert controller.custom_solve_order == []

    def test_default_goal_source_names_is_empty(self):
        """Test goal_source_names defaults to empty list."""
        controller = AnimationGraphController()
        assert controller.goal_source_names == []


class TestAnimationGraphControllerMethods:
    """Tests for AnimationGraphController methods."""

    def test_add_layer_returns_index(self):
        """Test add_layer returns correct index."""
        controller = AnimationGraphController()
        layer = MockIKLayer("layer1")
        index = controller.add_layer(layer)
        assert index == 0

    def test_add_layer_increments_index(self):
        """Test add_layer increments index for each layer."""
        controller = AnimationGraphController()
        layer1 = MockIKLayer("layer1")
        layer2 = MockIKLayer("layer2")

        index1 = controller.add_layer(layer1)
        index2 = controller.add_layer(layer2)

        assert index1 == 0
        assert index2 == 1

    def test_add_layer_stores_layer(self):
        """Test add_layer stores layer in list."""
        controller = AnimationGraphController()
        layer = MockIKLayer("layer1")
        controller.add_layer(layer)
        assert controller.layers[0] == layer

    def test_remove_layer_existing(self):
        """Test remove_layer returns True for existing layer."""
        controller = AnimationGraphController()
        layer = MockIKLayer("layer1")
        controller.add_layer(layer)

        result = controller.remove_layer("layer1")

        assert result is True
        assert len(controller.layers) == 0

    def test_remove_layer_nonexistent(self):
        """Test remove_layer returns False for nonexistent layer."""
        controller = AnimationGraphController()
        result = controller.remove_layer("nonexistent")
        assert result is False

    def test_get_layer_existing(self):
        """Test get_layer returns layer for existing name."""
        controller = AnimationGraphController()
        layer = MockIKLayer("layer1")
        controller.add_layer(layer)

        result = controller.get_layer("layer1")

        assert result == layer

    def test_get_layer_nonexistent(self):
        """Test get_layer returns None for nonexistent name."""
        controller = AnimationGraphController()
        result = controller.get_layer("nonexistent")
        assert result is None

    def test_set_layer_weight_existing(self):
        """Test set_layer_weight returns True for existing layer."""
        controller = AnimationGraphController()
        layer = MockIKLayer("layer1")
        controller.add_layer(layer)

        result = controller.set_layer_weight("layer1", 0.5)

        assert result is True
        assert layer._weight == 0.5

    def test_set_layer_weight_nonexistent(self):
        """Test set_layer_weight returns False for nonexistent layer."""
        controller = AnimationGraphController()
        result = controller.set_layer_weight("nonexistent", 0.5)
        assert result is False

    def test_set_layer_enabled_existing(self):
        """Test set_layer_enabled returns True for existing layer."""
        controller = AnimationGraphController()
        layer = MockIKLayer("layer1")
        controller.add_layer(layer)

        result = controller.set_layer_enabled("layer1", False)

        assert result is True
        assert layer._enabled is False

    def test_set_layer_enabled_nonexistent(self):
        """Test set_layer_enabled returns False for nonexistent layer."""
        controller = AnimationGraphController()
        result = controller.set_layer_enabled("nonexistent", False)
        assert result is False

    def test_set_ik_weight_valid(self):
        """Test set_ik_weight with valid value."""
        controller = AnimationGraphController()
        controller.set_ik_weight(0.5)
        assert controller.ik_weight == 0.5

    def test_set_ik_weight_clamps_above_one(self):
        """Test set_ik_weight clamps above 1.0."""
        controller = AnimationGraphController()
        controller.set_ik_weight(1.5)
        assert controller.ik_weight == 1.0

    def test_set_ik_weight_clamps_below_zero(self):
        """Test set_ik_weight clamps below 0.0."""
        controller = AnimationGraphController()
        controller.set_ik_weight(-0.5)
        assert controller.ik_weight == 0.0

    def test_set_solve_order_valid(self):
        """Test set_solve_order with valid order."""
        controller = AnimationGraphController()
        controller.set_solve_order("fullbody_first")
        assert controller.solve_order == "fullbody_first"

    def test_set_solve_order_with_custom(self):
        """Test set_solve_order with custom order list."""
        controller = AnimationGraphController()
        custom = ["layer1", "layer2"]
        controller.set_solve_order("custom", custom)
        assert controller.solve_order == "custom"
        assert controller.custom_solve_order == ["layer1", "layer2"]

    def test_set_solve_order_custom_copies_list(self):
        """Test set_solve_order creates copy of custom order list."""
        controller = AnimationGraphController()
        custom = ["layer1", "layer2"]
        controller.set_solve_order("custom", custom)
        custom.append("layer3")
        # Should not affect controller's list
        assert len(controller.custom_solve_order) == 2

    def test_layer_count_empty(self):
        """Test layer_count returns 0 for empty layers."""
        controller = AnimationGraphController()
        assert controller.layer_count() == 0

    def test_layer_count_with_layers(self):
        """Test layer_count returns correct count."""
        controller = AnimationGraphController()
        controller.add_layer(MockIKLayer("layer1"))
        controller.add_layer(MockIKLayer("layer2"))
        assert controller.layer_count() == 2

    def test_reset_clears_layers(self):
        """Test reset clears layers."""
        controller = AnimationGraphController()
        controller.add_layer(MockIKLayer("layer1"))
        controller.reset()
        assert controller.layers == []

    def test_reset_clears_output_transforms(self):
        """Test reset clears output_transforms."""
        controller = AnimationGraphController()
        controller.output_transforms.append(Transform(Vec3.zero(), Quat.identity()))
        controller.reset()
        assert controller.output_transforms == []

    def test_reset_clears_state(self):
        """Test reset clears animation state."""
        controller = AnimationGraphController()
        controller.current_state = "running"
        controller.blend_time = 0.5
        controller.ik_weight = 0.3
        controller._frame_count = 100
        controller._last_update_time = 50.0

        controller.reset()

        assert controller.current_state == ""
        assert controller.blend_time == 0.0
        assert controller.ik_weight == 1.0
        assert controller._frame_count == 0
        assert controller._last_update_time == 0.0


# =============================================================================
# Test LookAtTarget Component
# =============================================================================


class TestLookAtTargetFields:
    """Tests for LookAtTarget field defaults and types."""

    def test_default_target_position_is_none(self):
        """Test target_position defaults to None."""
        target = LookAtTarget()
        assert target.target_position is None

    def test_default_target_entity_is_none(self):
        """Test target_entity defaults to None."""
        target = LookAtTarget()
        assert target.target_entity is None

    def test_default_blend_weight_is_one(self):
        """Test blend_weight defaults to 1.0."""
        target = LookAtTarget()
        assert target.blend_weight == 1.0

    def test_default_enabled_is_true(self):
        """Test enabled defaults to True."""
        target = LookAtTarget()
        assert target.enabled is True

    def test_default_priority_is_zero(self):
        """Test priority defaults to 0."""
        target = LookAtTarget()
        assert target.priority == 0

    def test_default_max_rotation(self):
        """Test max_rotation defaults to 90.0."""
        target = LookAtTarget()
        assert target.max_rotation == 90.0

    def test_default_horizontal_limit(self):
        """Test horizontal_limit defaults to 120.0."""
        target = LookAtTarget()
        assert target.horizontal_limit == 120.0

    def test_default_vertical_limit(self):
        """Test vertical_limit defaults to 60.0."""
        target = LookAtTarget()
        assert target.vertical_limit == 60.0

    def test_default_spine_distribution(self):
        """Test spine_distribution defaults to [0.1, 0.2, 0.3, 0.4]."""
        target = LookAtTarget()
        assert target.spine_distribution == [0.1, 0.2, 0.3, 0.4]

    def test_spine_distribution_sums_to_one(self):
        """Test default spine_distribution sums to 1.0."""
        target = LookAtTarget()
        total = sum(target.spine_distribution)
        assert abs(total - 1.0) < 0.001

    def test_default_blend_speed(self):
        """Test blend_speed defaults to 5.0."""
        target = LookAtTarget()
        assert target.blend_speed == 5.0

    def test_default_current_weight_is_zero(self):
        """Test _current_weight defaults to 0.0."""
        target = LookAtTarget()
        assert target._current_weight == 0.0

    def test_default_last_target_is_none(self):
        """Test _last_target defaults to None."""
        target = LookAtTarget()
        assert target._last_target is None


class TestLookAtTargetMethods:
    """Tests for LookAtTarget methods."""

    def test_set_target_position(self):
        """Test set_target_position sets position and clears entity."""
        target = LookAtTarget()
        target.target_entity = 123

        pos = Vec3(1.0, 2.0, 3.0)
        target.set_target_position(pos, weight=0.8)

        assert vec3_approx_equal(target.target_position, pos)
        assert target.target_entity is None
        assert target.blend_weight == 0.8
        assert target.enabled is True

    def test_set_target_position_clamps_weight(self):
        """Test set_target_position clamps weight."""
        target = LookAtTarget()
        target.set_target_position(Vec3(1.0, 2.0, 3.0), weight=1.5)
        assert target.blend_weight == 1.0

    def test_set_target_entity(self):
        """Test set_target_entity sets entity and clears position."""
        target = LookAtTarget()
        target.target_position = Vec3(1.0, 2.0, 3.0)

        target.set_target_entity(456, weight=0.7)

        assert target.target_entity == 456
        assert target.target_position is None
        assert target.blend_weight == 0.7
        assert target.enabled is True

    def test_set_target_entity_clamps_weight(self):
        """Test set_target_entity clamps weight."""
        target = LookAtTarget()
        target.set_target_entity(123, weight=-0.5)
        assert target.blend_weight == 0.0

    def test_clear_target(self):
        """Test clear_target clears both targets and disables."""
        target = LookAtTarget()
        target.target_position = Vec3(1.0, 2.0, 3.0)
        target.target_entity = 123
        target.enabled = True

        target.clear_target()

        assert target.target_position is None
        assert target.target_entity is None
        assert target.enabled is False

    def test_set_spine_distribution_valid(self):
        """Test set_spine_distribution with valid distribution."""
        target = LookAtTarget()
        distribution = [0.25, 0.25, 0.25, 0.25]
        target.set_spine_distribution(distribution)
        assert target.spine_distribution == [0.25, 0.25, 0.25, 0.25]

    def test_set_spine_distribution_invalid_sum_raises(self):
        """Test set_spine_distribution raises for invalid sum."""
        target = LookAtTarget()
        distribution = [0.1, 0.1, 0.1, 0.1]  # Sums to 0.4
        with pytest.raises(ValueError) as exc_info:
            target.set_spine_distribution(distribution)
        assert "must sum to 1.0" in str(exc_info.value)

    def test_set_spine_distribution_copies_list(self):
        """Test set_spine_distribution creates copy of input list."""
        target = LookAtTarget()
        distribution = [0.25, 0.25, 0.25, 0.25]
        target.set_spine_distribution(distribution)
        distribution[0] = 0.9
        # Should not affect target's list
        assert target.spine_distribution[0] == 0.25

    def test_has_target_with_position(self):
        """Test has_target returns True with position."""
        target = LookAtTarget()
        target.target_position = Vec3(1.0, 2.0, 3.0)
        assert target.has_target() is True

    def test_has_target_with_entity(self):
        """Test has_target returns True with entity."""
        target = LookAtTarget()
        target.target_entity = 123
        assert target.has_target() is True

    def test_has_target_without_either(self):
        """Test has_target returns False without targets."""
        target = LookAtTarget()
        assert target.has_target() is False


# =============================================================================
# Test FootPlacementController Component
# =============================================================================


class TestFootPlacementControllerFields:
    """Tests for FootPlacementController field defaults and types."""

    def test_default_placement_is_none(self):
        """Test placement defaults to None."""
        controller = FootPlacementController()
        assert controller.placement is None

    def test_default_enabled_is_true(self):
        """Test enabled defaults to True."""
        controller = FootPlacementController()
        assert controller.enabled is True

    def test_default_terrain_layer_mask(self):
        """Test terrain_layer_mask defaults to 1."""
        controller = FootPlacementController()
        assert controller.terrain_layer_mask == 1

    def test_default_raycast_offset(self):
        """Test raycast_offset defaults to 1.0."""
        controller = FootPlacementController()
        assert controller.raycast_offset == 1.0

    def test_default_raycast_length(self):
        """Test raycast_length defaults to 2.0."""
        controller = FootPlacementController()
        assert controller.raycast_length == 2.0

    def test_default_max_step_height(self):
        """Test max_step_height defaults to 0.5."""
        controller = FootPlacementController()
        assert controller.max_step_height == 0.5

    def test_default_foot_height(self):
        """Test foot_height defaults to 0.05."""
        controller = FootPlacementController()
        assert controller.foot_height == 0.05

    def test_default_blend_speed(self):
        """Test blend_speed defaults to 10.0."""
        controller = FootPlacementController()
        assert controller.blend_speed == 10.0

    def test_default_pelvis_adjust_enabled(self):
        """Test pelvis_adjust_enabled defaults to True."""
        controller = FootPlacementController()
        assert controller.pelvis_adjust_enabled is True

    def test_default_max_pelvis_drop(self):
        """Test max_pelvis_drop defaults to 0.5."""
        controller = FootPlacementController()
        assert controller.max_pelvis_drop == 0.5

    def test_default_max_pelvis_raise(self):
        """Test max_pelvis_raise defaults to 0.3."""
        controller = FootPlacementController()
        assert controller.max_pelvis_raise == 0.3

    def test_default_toe_align_weight(self):
        """Test toe_align_weight defaults to 1.0."""
        controller = FootPlacementController()
        assert controller.toe_align_weight == 1.0

    def test_default_left_foot_offset(self):
        """Test left_foot_offset defaults to 0.0."""
        controller = FootPlacementController()
        assert controller.left_foot_offset == 0.0

    def test_default_right_foot_offset(self):
        """Test right_foot_offset defaults to 0.0."""
        controller = FootPlacementController()
        assert controller.right_foot_offset == 0.0

    def test_default_raycast_callback_is_none(self):
        """Test _raycast_callback defaults to None."""
        controller = FootPlacementController()
        assert controller._raycast_callback is None

    def test_default_left_planted_is_true(self):
        """Test _left_planted defaults to True."""
        controller = FootPlacementController()
        assert controller._left_planted is True

    def test_default_right_planted_is_true(self):
        """Test _right_planted defaults to True."""
        controller = FootPlacementController()
        assert controller._right_planted is True

    def test_default_current_pelvis_offset(self):
        """Test _current_pelvis_offset defaults to 0.0."""
        controller = FootPlacementController()
        assert controller._current_pelvis_offset == 0.0

    def test_default_terrain_slope(self):
        """Test _terrain_slope defaults to 0.0."""
        controller = FootPlacementController()
        assert controller._terrain_slope == 0.0


class TestFootPlacementControllerMethods:
    """Tests for FootPlacementController methods."""

    def test_set_enabled_true(self):
        """Test set_enabled with True."""
        controller = FootPlacementController()
        controller.set_enabled(True)
        assert controller.enabled is True

    def test_set_enabled_false(self):
        """Test set_enabled with False."""
        controller = FootPlacementController()
        controller.set_enabled(False)
        assert controller.enabled is False

    def test_set_foot_offset_left(self):
        """Test set_foot_offset for left foot."""
        controller = FootPlacementController()
        controller.set_foot_offset("left", 0.1)
        assert controller.left_foot_offset == 0.1

    def test_set_foot_offset_right(self):
        """Test set_foot_offset for right foot."""
        controller = FootPlacementController()
        controller.set_foot_offset("right", 0.15)
        assert controller.right_foot_offset == 0.15

    def test_set_foot_offset_invalid_foot(self):
        """Test set_foot_offset with invalid foot does nothing."""
        controller = FootPlacementController()
        controller.set_foot_offset("middle", 0.1)
        # Should not change anything
        assert controller.left_foot_offset == 0.0
        assert controller.right_foot_offset == 0.0

    def test_set_terrain_layer_mask(self):
        """Test set_terrain_layer_mask."""
        controller = FootPlacementController()
        controller.set_terrain_layer_mask(3)
        assert controller.terrain_layer_mask == 3

    def test_set_blend_speed_valid(self):
        """Test set_blend_speed with valid value."""
        controller = FootPlacementController()
        controller.set_blend_speed(5.0)
        assert controller.blend_speed == 5.0

    def test_set_blend_speed_clamps_minimum(self):
        """Test set_blend_speed clamps to minimum 0.1."""
        controller = FootPlacementController()
        controller.set_blend_speed(0.05)
        assert controller.blend_speed == 0.1

    def test_get_terrain_slope(self):
        """Test get_terrain_slope returns internal slope."""
        controller = FootPlacementController()
        controller._terrain_slope = 0.25
        assert controller.get_terrain_slope() == 0.25

    def test_is_left_foot_planted(self):
        """Test is_left_foot_planted returns internal state."""
        controller = FootPlacementController()
        controller._left_planted = False
        assert controller.is_left_foot_planted() is False

    def test_is_right_foot_planted(self):
        """Test is_right_foot_planted returns internal state."""
        controller = FootPlacementController()
        controller._right_planted = False
        assert controller.is_right_foot_planted() is False

    def test_reset_restores_defaults(self):
        """Test reset restores default values."""
        controller = FootPlacementController()
        controller._left_planted = False
        controller._right_planted = False
        controller._current_pelvis_offset = 0.3
        controller._terrain_slope = 0.2

        controller.reset()

        assert controller._left_planted is True
        assert controller._right_planted is True
        assert controller._current_pelvis_offset == 0.0
        assert controller._terrain_slope == 0.0


# =============================================================================
# Test IKTargetComponent Component
# =============================================================================


class TestIKTargetComponentFields:
    """Tests for IKTargetComponent field defaults and types."""

    def test_default_position_goals_is_empty_dict(self):
        """Test position_goals defaults to empty dict."""
        target = IKTargetComponent()
        assert target.position_goals == {}
        assert isinstance(target.position_goals, dict)

    def test_default_rotation_goals_is_empty_dict(self):
        """Test rotation_goals defaults to empty dict."""
        target = IKTargetComponent()
        assert target.rotation_goals == {}

    def test_default_weights_is_empty_dict(self):
        """Test weights defaults to empty dict."""
        target = IKTargetComponent()
        assert target.weights == {}

    def test_default_pole_vectors_is_empty_dict(self):
        """Test pole_vectors defaults to empty dict."""
        target = IKTargetComponent()
        assert target.pole_vectors == {}

    def test_default_active_is_true(self):
        """Test active defaults to True."""
        target = IKTargetComponent()
        assert target.active is True

    def test_default_priority_is_zero(self):
        """Test priority defaults to 0."""
        target = IKTargetComponent()
        assert target.priority == 0

    def test_default_chain_assignments_is_empty_dict(self):
        """Test chain_assignments defaults to empty dict."""
        target = IKTargetComponent()
        assert target.chain_assignments == {}


class TestIKTargetComponentMethods:
    """Tests for IKTargetComponent methods."""

    def test_set_position_goal(self):
        """Test set_position_goal stores position and weight."""
        target = IKTargetComponent()
        pos = Vec3(1.0, 2.0, 3.0)
        target.set_position_goal("RightHand", pos, weight=0.8)

        assert vec3_approx_equal(target.position_goals["RightHand"], pos)
        assert target.weights["RightHand"] == 0.8

    def test_set_position_goal_clamps_weight(self):
        """Test set_position_goal clamps weight to 0-1."""
        target = IKTargetComponent()
        target.set_position_goal("RightHand", Vec3(1.0, 2.0, 3.0), weight=1.5)
        assert target.weights["RightHand"] == 1.0

    def test_set_position_goal_with_chain_type(self):
        """Test set_position_goal stores chain assignment."""
        target = IKTargetComponent()
        target.set_position_goal(
            "RightHand", Vec3(1.0, 2.0, 3.0),
            weight=1.0, chain_type="right_arm"
        )
        assert target.chain_assignments["RightHand"] == "right_arm"

    def test_set_rotation_goal(self):
        """Test set_rotation_goal stores rotation and weight."""
        target = IKTargetComponent()
        rot = Quat.from_axis_angle(Vec3(0.0, 1.0, 0.0), math.pi / 4)
        target.set_rotation_goal("Head", rot, weight=0.7)

        assert quat_approx_equal(target.rotation_goals["Head"], rot)
        assert target.weights["Head"] == 0.7

    def test_set_rotation_goal_clamps_weight(self):
        """Test set_rotation_goal clamps weight."""
        target = IKTargetComponent()
        target.set_rotation_goal("Head", Quat.identity(), weight=-0.5)
        assert target.weights["Head"] == 0.0

    def test_set_pole_vector(self):
        """Test set_pole_vector stores pole vector."""
        target = IKTargetComponent()
        pole = Vec3(0.0, 0.0, 1.0)
        target.set_pole_vector("RightHand", pole)
        assert vec3_approx_equal(target.pole_vectors["RightHand"], pole)

    def test_remove_goal_existing_position(self):
        """Test remove_goal removes position goal."""
        target = IKTargetComponent()
        target.set_position_goal("RightHand", Vec3(1.0, 2.0, 3.0))

        result = target.remove_goal("RightHand")

        assert result is True
        assert "RightHand" not in target.position_goals
        assert "RightHand" not in target.weights

    def test_remove_goal_existing_rotation(self):
        """Test remove_goal removes rotation goal."""
        target = IKTargetComponent()
        target.set_rotation_goal("Head", Quat.identity())

        result = target.remove_goal("Head")

        assert result is True
        assert "Head" not in target.rotation_goals

    def test_remove_goal_removes_all_related(self):
        """Test remove_goal removes all related data."""
        target = IKTargetComponent()
        target.set_position_goal("RightHand", Vec3(1.0, 2.0, 3.0), chain_type="right_arm")
        target.set_pole_vector("RightHand", Vec3(0.0, 0.0, 1.0))

        target.remove_goal("RightHand")

        assert "RightHand" not in target.position_goals
        assert "RightHand" not in target.weights
        assert "RightHand" not in target.pole_vectors
        assert "RightHand" not in target.chain_assignments

    def test_remove_goal_nonexistent(self):
        """Test remove_goal returns False for nonexistent goal."""
        target = IKTargetComponent()
        result = target.remove_goal("nonexistent")
        assert result is False

    def test_clear(self):
        """Test clear removes all goals."""
        target = IKTargetComponent()
        target.set_position_goal("RightHand", Vec3(1.0, 2.0, 3.0), chain_type="right_arm")
        target.set_rotation_goal("Head", Quat.identity())
        target.set_pole_vector("RightHand", Vec3(0.0, 0.0, 1.0))

        target.clear()

        assert target.position_goals == {}
        assert target.rotation_goals == {}
        assert target.weights == {}
        assert target.pole_vectors == {}
        assert target.chain_assignments == {}

    def test_has_goals_with_position(self):
        """Test has_goals returns True with position goal."""
        target = IKTargetComponent()
        target.set_position_goal("RightHand", Vec3(1.0, 2.0, 3.0))
        assert target.has_goals() is True

    def test_has_goals_with_rotation(self):
        """Test has_goals returns True with rotation goal."""
        target = IKTargetComponent()
        target.set_rotation_goal("Head", Quat.identity())
        assert target.has_goals() is True

    def test_has_goals_without_any(self):
        """Test has_goals returns False without goals."""
        target = IKTargetComponent()
        assert target.has_goals() is False

    def test_has_goals_after_clear(self):
        """Test has_goals returns False after clear."""
        target = IKTargetComponent()
        target.set_position_goal("RightHand", Vec3(1.0, 2.0, 3.0))
        target.clear()
        assert target.has_goals() is False


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases across all components."""

    def test_fullbodyikcontroller_with_custom_solve_order(self):
        """Test FullBodyIKController with custom solve_order."""
        controller = FullBodyIKController(
            solve_order=["look_at", "arms", "spine", "foot_placement"]
        )
        assert controller.solve_order[0] == "look_at"

    def test_animationgraphcontroller_empty_layers_operations(self):
        """Test AnimationGraphController operations on empty layers."""
        controller = AnimationGraphController()
        assert controller.layer_count() == 0
        assert controller.get_layer("any") is None
        assert controller.remove_layer("any") is False

    def test_lookattarget_spine_distribution_three_elements(self):
        """Test LookAtTarget with 3-element spine distribution."""
        target = LookAtTarget()
        distribution = [0.2, 0.3, 0.5]  # 3 elements, sums to 1.0
        target.set_spine_distribution(distribution)
        assert len(target.spine_distribution) == 3

    def test_footplacementcontroller_multiple_resets(self):
        """Test FootPlacementController multiple resets."""
        controller = FootPlacementController()
        controller._left_planted = False
        controller.reset()
        controller._left_planted = False
        controller.reset()
        assert controller._left_planted is True

    def test_iktargetcomponent_overwrite_goal(self):
        """Test IKTargetComponent overwriting existing goal."""
        target = IKTargetComponent()
        target.set_position_goal("RightHand", Vec3(1.0, 2.0, 3.0))
        target.set_position_goal("RightHand", Vec3(4.0, 5.0, 6.0))

        assert vec3_approx_equal(target.position_goals["RightHand"], Vec3(4.0, 5.0, 6.0))

    def test_weight_boundary_values(self):
        """Test weight setting at exact boundary values."""
        controller = FullBodyIKController()

        controller.set_weight(0.0)
        assert controller.weight == 0.0

        controller.set_weight(1.0)
        assert controller.weight == 1.0

    def test_component_initialization_with_custom_values(self):
        """Test component initialization with all custom values."""
        controller = FullBodyIKController(
            enabled=False,
            weight=0.5,
            foot_placement_enabled=False,
            look_at_enabled=True,
            look_at_weight=0.3,
            maintain_balance=False,
            pelvis_adjust_enabled=False,
            max_pelvis_drop=0.8,
        )

        assert controller.enabled is False
        assert controller.weight == 0.5
        assert controller.foot_placement_enabled is False
        assert controller.look_at_enabled is True
        assert controller.look_at_weight == 0.3
        assert controller.maintain_balance is False
        assert controller.pelvis_adjust_enabled is False
        assert controller.max_pelvis_drop == 0.8


# =============================================================================
# Test Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_fullbodyikcontroller_in_all(self):
        """Test FullBodyIKController is in __all__."""
        from engine.animation.ik import ecs_components
        assert "FullBodyIKController" in ecs_components.__all__

    def test_animationgraphcontroller_in_all(self):
        """Test AnimationGraphController is in __all__."""
        from engine.animation.ik import ecs_components
        assert "AnimationGraphController" in ecs_components.__all__

    def test_lookattarget_in_all(self):
        """Test LookAtTarget is in __all__."""
        from engine.animation.ik import ecs_components
        assert "LookAtTarget" in ecs_components.__all__

    def test_footplacementcontroller_in_all(self):
        """Test FootPlacementController is in __all__."""
        from engine.animation.ik import ecs_components
        assert "FootPlacementController" in ecs_components.__all__

    def test_iktargetcomponent_in_all(self):
        """Test IKTargetComponent is in __all__."""
        from engine.animation.ik import ecs_components
        assert "IKTargetComponent" in ecs_components.__all__

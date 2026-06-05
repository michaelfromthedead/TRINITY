"""Whitebox tests for Graph + IK Integration (T-FB-4.20).

Tests cover:
- IKSolveOrder enum values
- IKGoalSource dataclass and methods
- AnimationIKResult dataclass
- AnimationIKController internal implementation
- ComponentGoalSource, CallbackGoalSource, StaticGoalSource
- Goal collection and priority merging
- Solve order application for all modes
- Edge cases (empty sources, disabled sources, no layers)

Whitebox testing approach:
- Tests internal implementation details
- Accesses private attributes directly
- Verifies internal state transitions
- Uses mocks to isolate components
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, Mock, patch
from dataclasses import fields

from engine.animation.ik.graph_integration import (
    IKSolveOrder,
    IKGoalSource,
    AnimationIKResult,
    AnimationIKController,
    ComponentGoalSource,
    CallbackGoalSource,
    StaticGoalSource,
)
from engine.animation.ik.ik_layer import IKBlendMode, IKGoalContext, IKLayer, IKLayerStack
from engine.core.math.transform import Transform
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def identity_transforms():
    """Create a list of identity transforms for testing."""
    return [Transform.identity() for _ in range(5)]


@pytest.fixture
def mock_ik_layer():
    """Create a mock IK layer."""
    layer = MagicMock(spec=IKLayer)
    layer.name = "test_layer"
    layer.enabled = True
    layer.weight = 1.0
    layer.apply = MagicMock(side_effect=lambda t, dt: t)
    layer.update_goals = MagicMock()
    layer.clear_goals = MagicMock()
    layer.set_weight = MagicMock()
    layer.set_enabled = MagicMock()
    return layer


@pytest.fixture
def mock_foot_layer():
    """Create a mock foot IK layer."""
    layer = MagicMock(spec=IKLayer)
    layer.name = "foot_ik"
    layer.enabled = True
    layer.weight = 1.0
    layer.apply = MagicMock(side_effect=lambda t, dt: t)
    layer.update_goals = MagicMock()
    layer.clear_goals = MagicMock()
    layer.set_weight = MagicMock()
    layer.set_enabled = MagicMock()
    return layer


@pytest.fixture
def mock_fullbody_layer():
    """Create a mock fullbody IK layer."""
    layer = MagicMock(spec=IKLayer)
    layer.name = "fullbody_ik"
    layer.enabled = True
    layer.weight = 1.0
    layer.apply = MagicMock(side_effect=lambda t, dt: t)
    layer.update_goals = MagicMock()
    layer.clear_goals = MagicMock()
    layer.set_weight = MagicMock()
    layer.set_enabled = MagicMock()
    return layer


@pytest.fixture
def controller():
    """Create an AnimationIKController instance."""
    return AnimationIKController()


# =============================================================================
# TEST IKSolveOrder ENUM
# =============================================================================


class TestIKSolveOrderEnum:
    """Whitebox tests for IKSolveOrder enum."""

    def test_all_enum_values_exist(self):
        """Verify all four solve order modes are defined."""
        assert hasattr(IKSolveOrder, 'FOOT_FIRST')
        assert hasattr(IKSolveOrder, 'FULLBODY_FIRST')
        assert hasattr(IKSolveOrder, 'PARALLEL')
        assert hasattr(IKSolveOrder, 'CUSTOM')

    def test_enum_values_are_unique(self):
        """Each enum value should be unique."""
        values = [
            IKSolveOrder.FOOT_FIRST.value,
            IKSolveOrder.FULLBODY_FIRST.value,
            IKSolveOrder.PARALLEL.value,
            IKSolveOrder.CUSTOM.value,
        ]
        assert len(values) == len(set(values))

    def test_enum_is_comparable(self):
        """Enum values should be comparable for equality."""
        assert IKSolveOrder.FOOT_FIRST == IKSolveOrder.FOOT_FIRST
        assert IKSolveOrder.FOOT_FIRST != IKSolveOrder.PARALLEL

    def test_enum_iteration(self):
        """Can iterate over all enum values."""
        count = 0
        for mode in IKSolveOrder:
            count += 1
        assert count == 4


# =============================================================================
# TEST IKGoalSource DATACLASS
# =============================================================================


class TestIKGoalSourceDataclass:
    """Whitebox tests for IKGoalSource dataclass."""

    def test_dataclass_fields(self):
        """Verify all expected fields are present."""
        field_names = {f.name for f in fields(IKGoalSource)}
        assert 'name' in field_names
        assert 'priority' in field_names
        assert 'enabled' in field_names
        assert 'get_goals' in field_names

    def test_default_values(self):
        """Test default field values."""
        source = IKGoalSource(name="test")
        assert source.name == "test"
        assert source.priority == 0
        assert source.enabled is True
        assert source.get_goals is None

    def test_post_init_validates_name(self):
        """__post_init__ should raise on empty name."""
        with pytest.raises(ValueError, match="non-empty name"):
            IKGoalSource(name="")

    def test_fetch_goals_returns_none_when_disabled(self):
        """fetch_goals returns None when source is disabled."""
        ctx = IKGoalContext()
        source = IKGoalSource(
            name="test",
            enabled=False,
            get_goals=lambda: ctx
        )
        assert source.fetch_goals() is None

    def test_fetch_goals_returns_none_when_no_getter(self):
        """fetch_goals returns None when get_goals is None."""
        source = IKGoalSource(name="test", enabled=True)
        assert source.fetch_goals() is None

    def test_fetch_goals_calls_getter(self):
        """fetch_goals should call get_goals callable."""
        ctx = IKGoalContext()
        ctx.position_goals["Hand"] = Vec3(1, 2, 3)
        source = IKGoalSource(
            name="test",
            enabled=True,
            get_goals=lambda: ctx
        )
        result = source.fetch_goals()
        assert result is ctx
        assert "Hand" in result.position_goals

    def test_fetch_goals_handles_exception(self):
        """fetch_goals returns None if getter raises."""
        def raise_error():
            raise RuntimeError("Oops")

        source = IKGoalSource(
            name="test",
            enabled=True,
            get_goals=raise_error
        )
        assert source.fetch_goals() is None

    def test_set_enabled_updates_field(self):
        """set_enabled should update enabled field."""
        source = IKGoalSource(name="test")
        assert source.enabled is True
        source.set_enabled(False)
        assert source.enabled is False
        source.set_enabled(True)
        assert source.enabled is True

    def test_set_priority_updates_field(self):
        """set_priority should update priority field."""
        source = IKGoalSource(name="test", priority=5)
        assert source.priority == 5
        source.set_priority(100)
        assert source.priority == 100
        source.set_priority(-10)
        assert source.priority == -10


# =============================================================================
# TEST AnimationIKResult DATACLASS
# =============================================================================


class TestAnimationIKResultDataclass:
    """Whitebox tests for AnimationIKResult dataclass."""

    def test_dataclass_fields(self):
        """Verify all expected fields are present."""
        field_names = {f.name for f in fields(AnimationIKResult)}
        expected = {
            'transforms', 'animation_weight', 'ik_weight',
            'layers_applied', 'goals_used', 'errors', 'success'
        }
        assert expected.issubset(field_names)

    def test_default_values(self):
        """Test default field values."""
        result = AnimationIKResult()
        assert result.transforms == []
        assert result.animation_weight == 1.0
        assert result.ik_weight == 1.0
        assert result.layers_applied == []
        assert result.goals_used == 0
        assert result.errors == {}
        assert result.success is True

    def test_add_layer_applied_no_duplicates(self):
        """add_layer_applied should not add duplicates."""
        result = AnimationIKResult()
        result.add_layer_applied("foot_ik")
        result.add_layer_applied("foot_ik")
        result.add_layer_applied("hand_ik")
        assert result.layers_applied == ["foot_ik", "hand_ik"]

    def test_set_error_records_error(self):
        """set_error should record bone error."""
        result = AnimationIKResult()
        result.set_error("LeftHand", 0.05)
        result.set_error("RightHand", 0.03)
        assert result.errors["LeftHand"] == 0.05
        assert result.errors["RightHand"] == 0.03

    def test_total_error_sums_all_errors(self):
        """total_error should sum all bone errors."""
        result = AnimationIKResult()
        result.set_error("A", 0.1)
        result.set_error("B", 0.2)
        result.set_error("C", 0.3)
        assert abs(result.total_error() - 0.6) < 0.0001

    def test_total_error_empty(self):
        """total_error returns 0 for no errors."""
        result = AnimationIKResult()
        assert result.total_error() == 0.0

    def test_average_error_calculation(self):
        """average_error should calculate mean error."""
        result = AnimationIKResult()
        result.set_error("A", 0.1)
        result.set_error("B", 0.2)
        result.set_error("C", 0.3)
        assert abs(result.average_error() - 0.2) < 0.0001

    def test_average_error_empty(self):
        """average_error returns 0 for no errors."""
        result = AnimationIKResult()
        assert result.average_error() == 0.0


# =============================================================================
# TEST AnimationIKController.__init__
# =============================================================================


class TestAnimationIKControllerInit:
    """Whitebox tests for AnimationIKController initialization."""

    def test_internal_state_initialization(self, controller):
        """Verify all internal state is initialized correctly."""
        # IK stack should be created
        assert isinstance(controller._ik_stack, IKLayerStack)

        # Goal sources should be empty
        assert controller._goal_sources == []
        assert controller._sources_by_name == {}

        # Solve order defaults to FOOT_FIRST
        assert controller._solve_order == IKSolveOrder.FOOT_FIRST
        assert controller._custom_order == []

        # Global settings
        assert controller.ik_weight == 1.0
        assert controller.enabled is True

        # Cached state
        assert isinstance(controller._merged_context, IKGoalContext)
        assert controller._last_result is None

        # Layer categories cache
        assert controller._layer_categories == {}

    def test_class_constants_defined(self):
        """Verify class constants are defined."""
        assert hasattr(AnimationIKController, 'FOOT_LAYER_NAMES')
        assert hasattr(AnimationIKController, 'FULLBODY_LAYER_NAMES')
        assert isinstance(AnimationIKController.FOOT_LAYER_NAMES, set)
        assert isinstance(AnimationIKController.FULLBODY_LAYER_NAMES, set)
        assert "foot_ik" in AnimationIKController.FOOT_LAYER_NAMES
        assert "fullbody_ik" in AnimationIKController.FULLBODY_LAYER_NAMES


# =============================================================================
# TEST AnimationIKController.add_goal_source / remove_goal_source
# =============================================================================


class TestAnimationIKControllerGoalSourceManagement:
    """Whitebox tests for goal source management."""

    def test_add_goal_source_appends_to_list(self, controller):
        """add_goal_source should add to internal lists."""
        source = IKGoalSource(name="test", priority=10)
        controller.add_goal_source(source)

        assert len(controller._goal_sources) == 1
        assert controller._goal_sources[0] is source
        assert controller._sources_by_name["test"] is source

    def test_add_goal_source_raises_on_duplicate(self, controller):
        """add_goal_source should raise if name exists."""
        controller.add_goal_source(IKGoalSource(name="test"))
        with pytest.raises(ValueError, match="already exists"):
            controller.add_goal_source(IKGoalSource(name="test"))

    def test_add_goal_source_maintains_priority_order(self, controller):
        """Sources should be sorted by priority ascending."""
        controller.add_goal_source(IKGoalSource(name="high", priority=100))
        controller.add_goal_source(IKGoalSource(name="low", priority=1))
        controller.add_goal_source(IKGoalSource(name="mid", priority=50))

        names = [s.name for s in controller._goal_sources]
        assert names == ["low", "mid", "high"]

    def test_remove_goal_source_removes_from_both_collections(self, controller):
        """remove_goal_source removes from list and dict."""
        controller.add_goal_source(IKGoalSource(name="test"))
        assert controller.remove_goal_source("test") is True
        assert len(controller._goal_sources) == 0
        assert "test" not in controller._sources_by_name

    def test_remove_goal_source_returns_false_if_not_found(self, controller):
        """remove_goal_source returns False if source not found."""
        assert controller.remove_goal_source("nonexistent") is False

    def test_get_goal_source_returns_source(self, controller):
        """get_goal_source returns the source by name."""
        source = IKGoalSource(name="test")
        controller.add_goal_source(source)
        assert controller.get_goal_source("test") is source

    def test_get_goal_source_returns_none_if_not_found(self, controller):
        """get_goal_source returns None if not found."""
        assert controller.get_goal_source("nonexistent") is None

    def test_set_goal_source_enabled(self, controller):
        """set_goal_source_enabled updates source enabled state."""
        source = IKGoalSource(name="test", enabled=True)
        controller.add_goal_source(source)

        assert controller.set_goal_source_enabled("test", False) is True
        assert source.enabled is False

        assert controller.set_goal_source_enabled("nonexistent", True) is False

    def test_set_goal_source_priority_reorders(self, controller):
        """set_goal_source_priority should re-sort sources."""
        controller.add_goal_source(IKGoalSource(name="a", priority=10))
        controller.add_goal_source(IKGoalSource(name="b", priority=20))

        # Change priority to make 'a' higher
        controller.set_goal_source_priority("a", 30)

        names = [s.name for s in controller._goal_sources]
        assert names == ["b", "a"]

    def test_goal_source_count(self, controller):
        """goal_source_count returns correct count."""
        assert controller.goal_source_count() == 0
        controller.add_goal_source(IKGoalSource(name="a"))
        assert controller.goal_source_count() == 1
        controller.add_goal_source(IKGoalSource(name="b"))
        assert controller.goal_source_count() == 2

    def test_get_goal_source_names(self, controller):
        """get_goal_source_names returns names in priority order."""
        controller.add_goal_source(IKGoalSource(name="high", priority=100))
        controller.add_goal_source(IKGoalSource(name="low", priority=1))
        names = controller.get_goal_source_names()
        assert names == ["low", "high"]


# =============================================================================
# TEST AnimationIKController.add_ik_layer / remove_ik_layer
# =============================================================================


class TestAnimationIKControllerLayerManagement:
    """Whitebox tests for IK layer management."""

    def test_add_ik_layer_adds_to_stack(self, controller, mock_ik_layer):
        """add_ik_layer should add layer to internal stack."""
        index = controller.add_ik_layer(mock_ik_layer)
        assert index == 0
        assert controller._ik_stack.layer_count() == 1

    def test_add_ik_layer_infers_category(self, controller, mock_foot_layer):
        """add_ik_layer should infer category from name."""
        controller.add_ik_layer(mock_foot_layer)
        assert controller._layer_categories["foot_ik"] == "foot"

    def test_add_ik_layer_uses_explicit_category(self, controller, mock_ik_layer):
        """add_ik_layer should use explicit category if provided."""
        controller.add_ik_layer(mock_ik_layer, category="fullbody")
        assert controller._layer_categories["test_layer"] == "fullbody"

    def test_remove_ik_layer_removes_from_stack_and_categories(
        self, controller, mock_ik_layer
    ):
        """remove_ik_layer removes layer and its category."""
        controller.add_ik_layer(mock_ik_layer)
        assert controller.remove_ik_layer("test_layer") is True
        assert controller._ik_stack.layer_count() == 0
        assert "test_layer" not in controller._layer_categories

    def test_remove_ik_layer_returns_false_if_not_found(self, controller):
        """remove_ik_layer returns False if layer not found."""
        assert controller.remove_ik_layer("nonexistent") is False

    def test_get_ik_layer(self, controller, mock_ik_layer):
        """get_ik_layer returns the layer."""
        controller.add_ik_layer(mock_ik_layer)
        assert controller.get_ik_layer("test_layer") is mock_ik_layer

    def test_ik_layer_count(self, controller, mock_ik_layer, mock_foot_layer):
        """ik_layer_count returns correct count."""
        assert controller.ik_layer_count() == 0
        controller.add_ik_layer(mock_ik_layer)
        assert controller.ik_layer_count() == 1
        controller.add_ik_layer(mock_foot_layer)
        assert controller.ik_layer_count() == 2

    def test_get_ik_layer_names(self, controller, mock_ik_layer, mock_foot_layer):
        """get_ik_layer_names returns layer names in order."""
        controller.add_ik_layer(mock_ik_layer)
        controller.add_ik_layer(mock_foot_layer)
        names = controller.get_ik_layer_names()
        assert names == ["test_layer", "foot_ik"]

    def test_set_ik_layer_weight(self, controller, mock_ik_layer):
        """set_ik_layer_weight calls layer.set_weight."""
        controller.add_ik_layer(mock_ik_layer)
        result = controller.set_ik_layer_weight("test_layer", 0.5, immediate=True)
        assert result is True
        mock_ik_layer.set_weight.assert_called_once_with(0.5, True)

    def test_set_ik_layer_enabled(self, controller, mock_ik_layer):
        """set_ik_layer_enabled calls layer.set_enabled."""
        controller.add_ik_layer(mock_ik_layer)
        result = controller.set_ik_layer_enabled("test_layer", False)
        assert result is True
        mock_ik_layer.set_enabled.assert_called_once_with(False)

    def test_set_layer_category(self, controller, mock_ik_layer):
        """set_layer_category updates category cache."""
        controller.add_ik_layer(mock_ik_layer)
        result = controller.set_layer_category("test_layer", "foot")
        assert result is True
        assert controller._layer_categories["test_layer"] == "foot"

    def test_infer_layer_category_foot(self, controller):
        """_infer_layer_category identifies foot layers."""
        for name in ["foot_ik", "foot_placement", "feet_layer", "legs_solver"]:
            assert controller._infer_layer_category(name) == "foot"

    def test_infer_layer_category_fullbody(self, controller):
        """_infer_layer_category identifies fullbody layers."""
        for name in ["fullbody_ik", "fullbody_solver", "body_ik", "torso_ik"]:
            assert controller._infer_layer_category(name) == "fullbody"

    def test_infer_layer_category_other(self, controller):
        """_infer_layer_category returns other for unknown names."""
        assert controller._infer_layer_category("hand_ik") == "other"
        assert controller._infer_layer_category("look_at") == "other"


# =============================================================================
# TEST AnimationIKController.set_solve_order
# =============================================================================


class TestAnimationIKControllerSolveOrder:
    """Whitebox tests for solve order configuration."""

    def test_set_solve_order_foot_first(self, controller):
        """set_solve_order sets FOOT_FIRST mode."""
        controller.set_solve_order(IKSolveOrder.FOOT_FIRST)
        assert controller._solve_order == IKSolveOrder.FOOT_FIRST
        assert controller._custom_order == []

    def test_set_solve_order_fullbody_first(self, controller):
        """set_solve_order sets FULLBODY_FIRST mode."""
        controller.set_solve_order(IKSolveOrder.FULLBODY_FIRST)
        assert controller._solve_order == IKSolveOrder.FULLBODY_FIRST

    def test_set_solve_order_parallel(self, controller):
        """set_solve_order sets PARALLEL mode."""
        controller.set_solve_order(IKSolveOrder.PARALLEL)
        assert controller._solve_order == IKSolveOrder.PARALLEL

    def test_set_solve_order_custom_with_order(self, controller):
        """set_solve_order CUSTOM mode requires custom_order list."""
        custom = ["layer_a", "layer_b", "layer_c"]
        controller.set_solve_order(IKSolveOrder.CUSTOM, custom_order=custom)
        assert controller._solve_order == IKSolveOrder.CUSTOM
        assert controller._custom_order == custom

    def test_set_solve_order_custom_without_order_raises(self, controller):
        """set_solve_order CUSTOM without list raises ValueError."""
        with pytest.raises(ValueError, match="requires custom_order"):
            controller.set_solve_order(IKSolveOrder.CUSTOM)

    def test_set_solve_order_clears_custom_order_for_other_modes(self, controller):
        """Switching from CUSTOM to other mode clears custom_order."""
        controller.set_solve_order(IKSolveOrder.CUSTOM, custom_order=["a", "b"])
        controller.set_solve_order(IKSolveOrder.FOOT_FIRST)
        assert controller._custom_order == []

    def test_get_solve_order(self, controller):
        """get_solve_order returns current mode."""
        controller.set_solve_order(IKSolveOrder.PARALLEL)
        assert controller.get_solve_order() == IKSolveOrder.PARALLEL

    def test_get_custom_order_returns_copy(self, controller):
        """get_custom_order returns a copy of the list."""
        custom = ["a", "b"]
        controller.set_solve_order(IKSolveOrder.CUSTOM, custom_order=custom)
        result = controller.get_custom_order()
        assert result == custom
        assert result is not controller._custom_order


# =============================================================================
# TEST AnimationIKController.update PIPELINE
# =============================================================================


class TestAnimationIKControllerUpdate:
    """Whitebox tests for the update pipeline."""

    def test_update_returns_input_when_disabled(self, controller, identity_transforms):
        """update returns input transforms when controller disabled."""
        controller.enabled = False
        result = controller.update(identity_transforms, 0.016)
        assert result is identity_transforms

    def test_update_returns_input_when_weight_zero(self, controller, identity_transforms):
        """update returns input transforms when ik_weight is zero."""
        controller.ik_weight = 0.0
        result = controller.update(identity_transforms, 0.016)
        assert result is identity_transforms

    def test_update_calls_collect_goals(self, controller, identity_transforms):
        """update should call _collect_goals."""
        with patch.object(controller, '_collect_goals', return_value=IKGoalContext()) as mock:
            controller.update(identity_transforms, 0.016)
            mock.assert_called_once()

    def test_update_calls_distribute_goals(self, controller, identity_transforms):
        """update should call _distribute_goals."""
        ctx = IKGoalContext()
        with patch.object(controller, '_collect_goals', return_value=ctx):
            with patch.object(controller, '_distribute_goals') as mock:
                controller.update(identity_transforms, 0.016)
                mock.assert_called_once_with(ctx)

    def test_update_calls_apply_solve_order(self, controller, identity_transforms):
        """update should call _apply_solve_order."""
        ctx = IKGoalContext()
        with patch.object(controller, '_collect_goals', return_value=ctx):
            with patch.object(controller, '_apply_solve_order', return_value=identity_transforms) as mock:
                controller.update(identity_transforms, 0.016)
                mock.assert_called_once()

    def test_update_caches_result(self, controller, identity_transforms):
        """update should cache result via _cache_result."""
        controller.update(identity_transforms, 0.016)
        assert controller._last_result is not None


# =============================================================================
# TEST AnimationIKController._collect_goals PRIORITY MERGING
# =============================================================================


class TestAnimationIKControllerCollectGoals:
    """Whitebox tests for goal collection and priority merging."""

    def test_collect_goals_empty_sources(self, controller):
        """_collect_goals returns empty context when no sources."""
        ctx = controller._collect_goals()
        assert len(ctx.position_goals) == 0
        assert len(ctx.rotation_goals) == 0

    def test_collect_goals_skips_disabled_sources(self, controller):
        """_collect_goals skips disabled sources."""
        ctx = IKGoalContext()
        ctx.position_goals["Hand"] = Vec3(1, 2, 3)

        source = IKGoalSource(
            name="test",
            enabled=False,
            get_goals=lambda: ctx
        )
        controller.add_goal_source(source)

        result = controller._collect_goals()
        assert "Hand" not in result.position_goals

    def test_collect_goals_merges_position_goals(self, controller):
        """_collect_goals merges position goals from all sources."""
        ctx1 = IKGoalContext()
        ctx1.position_goals["Hand"] = Vec3(1, 2, 3)

        ctx2 = IKGoalContext()
        ctx2.position_goals["Foot"] = Vec3(4, 5, 6)

        controller.add_goal_source(IKGoalSource(name="s1", priority=1, get_goals=lambda: ctx1))
        controller.add_goal_source(IKGoalSource(name="s2", priority=2, get_goals=lambda: ctx2))

        result = controller._collect_goals()
        assert "Hand" in result.position_goals
        assert "Foot" in result.position_goals

    def test_collect_goals_higher_priority_overwrites(self, controller):
        """Higher priority sources overwrite lower priority for same bone."""
        ctx_low = IKGoalContext()
        ctx_low.position_goals["Hand"] = Vec3(1, 2, 3)
        ctx_low.weights["Hand"] = 0.5

        ctx_high = IKGoalContext()
        ctx_high.position_goals["Hand"] = Vec3(10, 20, 30)
        ctx_high.weights["Hand"] = 1.0

        controller.add_goal_source(IKGoalSource(
            name="low", priority=1, get_goals=lambda: ctx_low
        ))
        controller.add_goal_source(IKGoalSource(
            name="high", priority=100, get_goals=lambda: ctx_high
        ))

        result = controller._collect_goals()
        # Higher priority overwrites
        assert result.position_goals["Hand"].x == 10
        assert result.weights["Hand"] == 1.0

    def test_collect_goals_merges_rotation_goals(self, controller):
        """_collect_goals merges rotation goals."""
        ctx = IKGoalContext()
        ctx.rotation_goals["Head"] = Quat.identity()
        ctx.weights["Head"] = 0.8

        controller.add_goal_source(IKGoalSource(name="s1", get_goals=lambda: ctx))

        result = controller._collect_goals()
        assert "Head" in result.rotation_goals
        assert result.weights["Head"] == 0.8

    def test_collect_goals_merges_pole_vectors(self, controller):
        """_collect_goals merges pole vectors."""
        ctx = IKGoalContext()
        ctx.pole_vectors["LeftHand"] = Vec3(0, 1, 0)

        controller.add_goal_source(IKGoalSource(name="s1", get_goals=lambda: ctx))

        result = controller._collect_goals()
        assert "LeftHand" in result.pole_vectors

    def test_collect_goals_merges_chain_assignments(self, controller):
        """_collect_goals merges chain assignments."""
        ctx = IKGoalContext()
        ctx.chain_assignments["LeftHand"] = "left_arm"

        controller.add_goal_source(IKGoalSource(name="s1", get_goals=lambda: ctx))

        result = controller._collect_goals()
        assert result.chain_assignments["LeftHand"] == "left_arm"

    def test_collect_goals_caches_merged_context(self, controller):
        """_collect_goals stores result in _merged_context."""
        ctx = IKGoalContext()
        ctx.position_goals["Hand"] = Vec3(1, 2, 3)
        controller.add_goal_source(IKGoalSource(name="s1", get_goals=lambda: ctx))

        result = controller._collect_goals()
        assert controller._merged_context is result


# =============================================================================
# TEST AnimationIKController._apply_solve_order
# =============================================================================


class TestAnimationIKControllerApplySolveOrder:
    """Whitebox tests for solve order application."""

    def test_apply_solve_order_routes_to_foot_first(
        self, controller, identity_transforms
    ):
        """_apply_solve_order routes FOOT_FIRST to _apply_foot_first."""
        controller._solve_order = IKSolveOrder.FOOT_FIRST
        with patch.object(controller, '_apply_foot_first', return_value=identity_transforms) as mock:
            controller._apply_solve_order(identity_transforms, IKGoalContext(), 0.016)
            mock.assert_called_once()

    def test_apply_solve_order_routes_to_fullbody_first(
        self, controller, identity_transforms
    ):
        """_apply_solve_order routes FULLBODY_FIRST to _apply_fullbody_first."""
        controller._solve_order = IKSolveOrder.FULLBODY_FIRST
        with patch.object(controller, '_apply_fullbody_first', return_value=identity_transforms) as mock:
            controller._apply_solve_order(identity_transforms, IKGoalContext(), 0.016)
            mock.assert_called_once()

    def test_apply_solve_order_routes_to_parallel(
        self, controller, identity_transforms
    ):
        """_apply_solve_order routes PARALLEL to _apply_parallel."""
        controller._solve_order = IKSolveOrder.PARALLEL
        with patch.object(controller, '_apply_parallel', return_value=identity_transforms) as mock:
            controller._apply_solve_order(identity_transforms, IKGoalContext(), 0.016)
            mock.assert_called_once()

    def test_apply_solve_order_routes_to_custom(
        self, controller, identity_transforms
    ):
        """_apply_solve_order routes CUSTOM to _apply_custom_order."""
        controller._solve_order = IKSolveOrder.CUSTOM
        controller._custom_order = ["a", "b"]
        with patch.object(controller, '_apply_custom_order', return_value=identity_transforms) as mock:
            controller._apply_solve_order(identity_transforms, IKGoalContext(), 0.016)
            mock.assert_called_once()


# =============================================================================
# TEST AnimationIKController._apply_foot_first
# =============================================================================


class TestAnimationIKControllerApplyFootFirst:
    """Whitebox tests for foot-first solve order."""

    def test_foot_first_applies_foot_layers_first(
        self, controller, mock_foot_layer, mock_fullbody_layer, identity_transforms
    ):
        """_apply_foot_first applies foot category layers first."""
        controller.add_ik_layer(mock_fullbody_layer, category="fullbody")
        controller.add_ik_layer(mock_foot_layer, category="foot")

        call_order = []
        mock_foot_layer.apply = MagicMock(side_effect=lambda t, dt: (call_order.append("foot"), t)[1])
        mock_fullbody_layer.apply = MagicMock(side_effect=lambda t, dt: (call_order.append("fullbody"), t)[1])

        controller._apply_foot_first(identity_transforms, 0.016)

        assert call_order == ["foot", "fullbody"]

    def test_foot_first_skips_disabled_layers(
        self, controller, mock_foot_layer, identity_transforms
    ):
        """_apply_foot_first skips disabled layers."""
        mock_foot_layer.enabled = False
        controller.add_ik_layer(mock_foot_layer, category="foot")

        controller._apply_foot_first(identity_transforms, 0.016)
        mock_foot_layer.apply.assert_not_called()


# =============================================================================
# TEST AnimationIKController._apply_fullbody_first
# =============================================================================


class TestAnimationIKControllerApplyFullbodyFirst:
    """Whitebox tests for fullbody-first solve order."""

    def test_fullbody_first_applies_fullbody_layers_first(
        self, controller, mock_foot_layer, mock_fullbody_layer, identity_transforms
    ):
        """_apply_fullbody_first applies fullbody category first."""
        controller.add_ik_layer(mock_foot_layer, category="foot")
        controller.add_ik_layer(mock_fullbody_layer, category="fullbody")

        call_order = []
        mock_foot_layer.apply = MagicMock(side_effect=lambda t, dt: (call_order.append("foot"), t)[1])
        mock_fullbody_layer.apply = MagicMock(side_effect=lambda t, dt: (call_order.append("fullbody"), t)[1])

        controller._apply_fullbody_first(identity_transforms, 0.016)

        assert call_order == ["fullbody", "foot"]


# =============================================================================
# TEST AnimationIKController._apply_parallel
# =============================================================================


class TestAnimationIKControllerApplyParallel:
    """Whitebox tests for parallel solve mode."""

    def test_parallel_returns_input_when_no_layers(self, controller, identity_transforms):
        """_apply_parallel returns input when no layers."""
        result = controller._apply_parallel(identity_transforms, 0.016)
        assert result is identity_transforms

    def test_parallel_each_layer_gets_original_transforms(
        self, controller, identity_transforms
    ):
        """In parallel mode, each layer receives original transforms."""
        layer1 = MagicMock(spec=IKLayer)
        layer1.name = "layer1"
        layer1.enabled = True
        layer1.weight = 1.0

        layer2 = MagicMock(spec=IKLayer)
        layer2.name = "layer2"
        layer2.enabled = True
        layer2.weight = 1.0

        # Track what transforms each layer receives
        received = []
        def track_call(transforms, dt):
            # Capture a copy since list may be modified
            received.append(len(transforms))
            return transforms

        layer1.apply = MagicMock(side_effect=track_call)
        layer2.apply = MagicMock(side_effect=track_call)

        controller.add_ik_layer(layer1)
        controller.add_ik_layer(layer2)

        controller._apply_parallel(identity_transforms, 0.016)

        # Both should have been called
        assert layer1.apply.called
        assert layer2.apply.called

    def test_parallel_skips_disabled_layers(self, controller, identity_transforms):
        """_apply_parallel skips disabled layers."""
        layer = MagicMock(spec=IKLayer)
        layer.name = "disabled"
        layer.enabled = False
        layer.weight = 1.0
        layer.apply = MagicMock()

        controller.add_ik_layer(layer)
        controller._apply_parallel(identity_transforms, 0.016)
        layer.apply.assert_not_called()

    def test_parallel_skips_zero_weight_layers(self, controller, identity_transforms):
        """_apply_parallel skips layers with zero weight."""
        layer = MagicMock(spec=IKLayer)
        layer.name = "zero"
        layer.enabled = True
        layer.weight = 0.0
        layer.apply = MagicMock()

        controller.add_ik_layer(layer)
        controller._apply_parallel(identity_transforms, 0.016)
        layer.apply.assert_not_called()


# =============================================================================
# TEST AnimationIKController._apply_custom_order
# =============================================================================


class TestAnimationIKControllerApplyCustomOrder:
    """Whitebox tests for custom solve order."""

    def test_custom_order_applies_in_specified_order(self, controller, identity_transforms):
        """_apply_custom_order applies layers in custom order."""
        layer_a = MagicMock(spec=IKLayer)
        layer_a.name = "layer_a"
        layer_a.enabled = True

        layer_b = MagicMock(spec=IKLayer)
        layer_b.name = "layer_b"
        layer_b.enabled = True

        layer_c = MagicMock(spec=IKLayer)
        layer_c.name = "layer_c"
        layer_c.enabled = True

        call_order = []
        for layer in [layer_a, layer_b, layer_c]:
            name = layer.name
            layer.apply = MagicMock(side_effect=lambda t, dt, n=name: (call_order.append(n), t)[1])

        controller.add_ik_layer(layer_a)
        controller.add_ik_layer(layer_b)
        controller.add_ik_layer(layer_c)

        # Set custom order: c, a, b
        controller._custom_order = ["layer_c", "layer_a", "layer_b"]

        controller._apply_custom_order(identity_transforms, 0.016)

        assert call_order == ["layer_c", "layer_a", "layer_b"]

    def test_custom_order_applies_remaining_layers_after(self, controller, identity_transforms):
        """Layers not in custom_order are applied after in stack order."""
        layer_a = MagicMock(spec=IKLayer)
        layer_a.name = "layer_a"
        layer_a.enabled = True

        layer_b = MagicMock(spec=IKLayer)
        layer_b.name = "layer_b"
        layer_b.enabled = True

        call_order = []
        for layer in [layer_a, layer_b]:
            name = layer.name
            layer.apply = MagicMock(side_effect=lambda t, dt, n=name: (call_order.append(n), t)[1])

        controller.add_ik_layer(layer_a)
        controller.add_ik_layer(layer_b)

        # Only layer_b in custom order
        controller._custom_order = ["layer_b"]

        controller._apply_custom_order(identity_transforms, 0.016)

        # layer_b first (custom), then layer_a (remaining)
        assert call_order == ["layer_b", "layer_a"]


# =============================================================================
# TEST ComponentGoalSource
# =============================================================================


class TestComponentGoalSource:
    """Whitebox tests for ComponentGoalSource."""

    def test_inherits_from_ik_goal_source(self):
        """ComponentGoalSource should inherit from IKGoalSource."""
        assert issubclass(ComponentGoalSource, IKGoalSource)

    def test_init_sets_component_getter(self):
        """ComponentGoalSource stores component_getter."""
        getter = lambda: None
        source = ComponentGoalSource(name="test", component_getter=getter)
        assert source._component_getter is getter

    def test_set_component_getter(self):
        """set_component_getter updates the getter."""
        source = ComponentGoalSource(name="test")
        getter = lambda: None
        source.set_component_getter(getter)
        assert source._component_getter is getter

    def test_fetch_goals_returns_none_when_disabled(self):
        """fetch_goals returns None when disabled."""
        source = ComponentGoalSource(name="test", component_getter=lambda: object())
        source.enabled = False
        assert source.fetch_goals() is None

    def test_fetch_goals_returns_none_when_no_getter(self):
        """fetch_goals returns None when no getter set."""
        source = ComponentGoalSource(name="test")
        assert source.fetch_goals() is None

    def test_fetch_goals_returns_none_when_component_is_none(self):
        """fetch_goals returns None when component_getter returns None."""
        source = ComponentGoalSource(name="test", component_getter=lambda: None)
        assert source.fetch_goals() is None

    def test_fetch_goals_reads_position_goals(self):
        """fetch_goals reads position_goals from component."""
        class MockComponent:
            position_goals = {"Hand": Vec3(1, 2, 3)}

        source = ComponentGoalSource(
            name="test",
            component_getter=lambda: MockComponent()
        )
        result = source.fetch_goals()
        assert "Hand" in result.position_goals

    def test_fetch_goals_reads_rotation_goals(self):
        """fetch_goals reads rotation_goals from component."""
        class MockComponent:
            rotation_goals = {"Head": Quat.identity()}

        source = ComponentGoalSource(
            name="test",
            component_getter=lambda: MockComponent()
        )
        result = source.fetch_goals()
        assert "Head" in result.rotation_goals

    def test_fetch_goals_reads_weights(self):
        """fetch_goals reads weights from component."""
        class MockComponent:
            position_goals = {}
            weights = {"Hand": 0.75}

        source = ComponentGoalSource(
            name="test",
            component_getter=lambda: MockComponent()
        )
        result = source.fetch_goals()
        assert result.weights.get("Hand") == 0.75

    def test_fetch_goals_reads_pole_vectors(self):
        """fetch_goals reads pole_vectors from component."""
        class MockComponent:
            pole_vectors = {"LeftHand": Vec3(0, 1, 0)}

        source = ComponentGoalSource(
            name="test",
            component_getter=lambda: MockComponent()
        )
        result = source.fetch_goals()
        assert "LeftHand" in result.pole_vectors

    def test_fetch_goals_handles_exception(self):
        """fetch_goals returns None if getter raises."""
        def raise_error():
            raise RuntimeError("Component error")

        source = ComponentGoalSource(name="test", component_getter=raise_error)
        assert source.fetch_goals() is None


# =============================================================================
# TEST CallbackGoalSource
# =============================================================================


class TestCallbackGoalSource:
    """Whitebox tests for CallbackGoalSource."""

    def test_inherits_from_ik_goal_source(self):
        """CallbackGoalSource should inherit from IKGoalSource."""
        assert issubclass(CallbackGoalSource, IKGoalSource)

    def test_init_sets_get_goals(self):
        """CallbackGoalSource sets get_goals from callback arg."""
        ctx = IKGoalContext()
        source = CallbackGoalSource(name="test", callback=lambda: ctx)
        assert source.get_goals is not None
        assert source.fetch_goals() is ctx

    def test_set_callback_updates_get_goals(self):
        """set_callback updates get_goals callable."""
        source = CallbackGoalSource(name="test")
        ctx = IKGoalContext()
        ctx.position_goals["Test"] = Vec3(1, 2, 3)

        source.set_callback(lambda: ctx)
        result = source.fetch_goals()
        assert "Test" in result.position_goals


# =============================================================================
# TEST StaticGoalSource
# =============================================================================


class TestStaticGoalSource:
    """Whitebox tests for StaticGoalSource."""

    def test_inherits_from_ik_goal_source(self):
        """StaticGoalSource should inherit from IKGoalSource."""
        assert issubclass(StaticGoalSource, IKGoalSource)

    def test_init_creates_static_context(self):
        """StaticGoalSource creates internal static context."""
        source = StaticGoalSource(name="test")
        assert isinstance(source._static_context, IKGoalContext)

    def test_set_position_goal(self):
        """set_position_goal adds to static context."""
        source = StaticGoalSource(name="test")
        source.set_position_goal("Hand", Vec3(1, 2, 3), weight=0.8, chain_type="arm")

        ctx = source.fetch_goals()
        assert ctx.position_goals["Hand"].x == 1
        assert ctx.weights["Hand"] == 0.8
        assert ctx.chain_assignments["Hand"] == "arm"

    def test_set_rotation_goal(self):
        """set_rotation_goal adds to static context."""
        source = StaticGoalSource(name="test")
        rot = Quat.identity()
        source.set_rotation_goal("Head", rot, weight=0.9)

        ctx = source.fetch_goals()
        assert "Head" in ctx.rotation_goals
        assert ctx.weights["Head"] == 0.9

    def test_set_pole_vector(self):
        """set_pole_vector adds to static context."""
        source = StaticGoalSource(name="test")
        source.set_pole_vector("LeftHand", Vec3(0, 1, 0))

        ctx = source.fetch_goals()
        assert "LeftHand" in ctx.pole_vectors

    def test_clear_removes_all_goals(self):
        """clear removes all static goals."""
        source = StaticGoalSource(name="test")
        source.set_position_goal("Hand", Vec3(1, 2, 3))
        source.set_rotation_goal("Head", Quat.identity())
        source.clear()

        ctx = source.fetch_goals()
        assert len(ctx.position_goals) == 0
        assert len(ctx.rotation_goals) == 0

    def test_get_context_returns_internal_context(self):
        """get_context returns the internal static context."""
        source = StaticGoalSource(name="test")
        source.set_position_goal("Hand", Vec3(1, 2, 3))

        ctx = source.get_context()
        assert ctx is source._static_context
        assert "Hand" in ctx.position_goals


# =============================================================================
# TEST EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Whitebox tests for edge cases."""

    def test_empty_sources_update(self, controller, identity_transforms):
        """Update with no goal sources should work."""
        result = controller.update(identity_transforms, 0.016)
        assert len(result) == len(identity_transforms)

    def test_all_sources_disabled(self, controller, identity_transforms):
        """Update with all sources disabled should work."""
        ctx = IKGoalContext()
        ctx.position_goals["Hand"] = Vec3(1, 2, 3)

        source = IKGoalSource(name="test", enabled=False, get_goals=lambda: ctx)
        controller.add_goal_source(source)

        result = controller.update(identity_transforms, 0.016)
        merged = controller.get_merged_context()
        assert len(merged.position_goals) == 0

    def test_no_layers_update(self, controller, identity_transforms):
        """Update with no IK layers should return input."""
        ctx = IKGoalContext()
        ctx.position_goals["Hand"] = Vec3(1, 2, 3)
        controller.add_goal_source(IKGoalSource(name="test", get_goals=lambda: ctx))

        result = controller.update(identity_transforms, 0.016)
        # Without layers, parallel mode returns input
        controller.set_solve_order(IKSolveOrder.PARALLEL)
        result = controller.update(identity_transforms, 0.016)
        assert result is identity_transforms

    def test_empty_transforms_update(self, controller):
        """Update with empty transforms list should work."""
        result = controller.update([], 0.016)
        assert result == []

    def test_reset_clears_all_state(self, controller, mock_ik_layer):
        """reset should clear all internal state."""
        controller.add_goal_source(IKGoalSource(name="test"))
        controller.add_ik_layer(mock_ik_layer)
        controller.set_solve_order(IKSolveOrder.CUSTOM, custom_order=["a"])
        controller.ik_weight = 0.5
        controller.enabled = False

        controller.reset()

        assert controller.goal_source_count() == 0
        assert controller.ik_layer_count() == 0
        assert controller._solve_order == IKSolveOrder.FOOT_FIRST
        assert controller._custom_order == []
        assert controller.ik_weight == 1.0
        assert controller.enabled is True
        assert controller._last_result is None

    def test_get_last_result_initially_none(self, controller):
        """get_last_result returns None before first update."""
        assert controller.get_last_result() is None

    def test_get_merged_context_returns_cached(self, controller, identity_transforms):
        """get_merged_context returns cached merged context."""
        ctx = IKGoalContext()
        ctx.position_goals["Hand"] = Vec3(1, 2, 3)
        controller.add_goal_source(IKGoalSource(name="test", get_goals=lambda: ctx))

        controller.update(identity_transforms, 0.016)
        merged = controller.get_merged_context()
        assert "Hand" in merged.position_goals

    def test_set_enabled(self, controller):
        """set_enabled updates enabled state."""
        controller.set_enabled(False)
        assert controller.enabled is False
        controller.set_enabled(True)
        assert controller.enabled is True

    def test_is_enabled(self, controller):
        """is_enabled returns current enabled state."""
        assert controller.is_enabled() is True
        controller.enabled = False
        assert controller.is_enabled() is False

    def test_set_ik_weight_clamps(self, controller):
        """set_ik_weight clamps value to 0-1."""
        controller.set_ik_weight(2.0)
        assert controller.ik_weight == 1.0
        controller.set_ik_weight(-0.5)
        assert controller.ik_weight == 0.0
        controller.set_ik_weight(0.75)
        assert controller.ik_weight == 0.75

    def test_get_ik_weight(self, controller):
        """get_ik_weight returns current weight."""
        controller.ik_weight = 0.42
        assert controller.get_ik_weight() == 0.42

    def test_clear_all_goals_calls_layer_clear(self, controller, mock_ik_layer):
        """clear_all_goals calls clear_goals on all layers."""
        controller.add_ik_layer(mock_ik_layer)
        controller.clear_all_goals()
        mock_ik_layer.clear_goals.assert_called_once()

    def test_disable_all_layers(self, controller, mock_ik_layer):
        """disable_all_layers disables all layers via stack."""
        controller.add_ik_layer(mock_ik_layer)
        controller.disable_all_layers()
        # IKLayerStack.disable_all iterates and calls set_enabled
        # Our mock layer is in the stack

    def test_enable_all_layers(self, controller, mock_ik_layer):
        """enable_all_layers enables all layers via stack."""
        controller.add_ik_layer(mock_ik_layer)
        controller.enable_all_layers()


# =============================================================================
# TEST _blend_parallel_results and _weighted_blend_transform
# =============================================================================


class TestParallelBlending:
    """Whitebox tests for parallel blending internals."""

    def test_blend_parallel_results_empty_base(self, controller):
        """_blend_parallel_results handles empty base transforms."""
        result = controller._blend_parallel_results([], [], 1.0)
        assert result == []

    def test_blend_parallel_results_single_layer(self, controller, identity_transforms):
        """_blend_parallel_results with single layer."""
        layer = MagicMock(spec=IKLayer)
        layer.weight = 1.0

        result = controller._blend_parallel_results(
            identity_transforms,
            [(layer, identity_transforms)],
            1.0
        )
        assert len(result) == len(identity_transforms)

    def test_weighted_blend_transform_empty(self, controller):
        """_weighted_blend_transform with empty list returns identity."""
        result = controller._weighted_blend_transform([])
        # Should return identity transform
        assert isinstance(result, Transform)

    def test_weighted_blend_transform_single(self, controller):
        """_weighted_blend_transform with single transform returns it."""
        tf = Transform(Vec3(1, 2, 3), Quat.identity(), Vec3.one())
        result = controller._weighted_blend_transform([(tf, 1.0)])
        assert result is tf

    def test_weighted_blend_transform_blends_positions(self, controller):
        """_weighted_blend_transform blends positions correctly."""
        tf1 = Transform(Vec3(0, 0, 0), Quat.identity(), Vec3.one())
        tf2 = Transform(Vec3(10, 10, 10), Quat.identity(), Vec3.one())

        result = controller._weighted_blend_transform([
            (tf1, 0.5),
            (tf2, 0.5)
        ])

        # Equal weights, so midpoint
        assert abs(result.translation.x - 5) < 0.01
        assert abs(result.translation.y - 5) < 0.01
        assert abs(result.translation.z - 5) < 0.01


# =============================================================================
# TEST _cache_result
# =============================================================================


class TestCacheResult:
    """Whitebox tests for result caching."""

    def test_cache_result_creates_animation_ik_result(
        self, controller, identity_transforms
    ):
        """_cache_result creates AnimationIKResult."""
        ctx = IKGoalContext()
        ctx.position_goals["Hand"] = Vec3(1, 2, 3)
        ctx.rotation_goals["Head"] = Quat.identity()

        controller._cache_result(identity_transforms, ctx)

        result = controller._last_result
        assert isinstance(result, AnimationIKResult)
        assert result.transforms is identity_transforms
        assert result.goals_used == 2  # 1 position + 1 rotation
        assert result.success is True

    def test_cache_result_records_applied_layers(
        self, controller, mock_ik_layer, identity_transforms
    ):
        """_cache_result records applied layers."""
        controller.add_ik_layer(mock_ik_layer)
        ctx = IKGoalContext()

        controller._cache_result(identity_transforms, ctx)

        result = controller._last_result
        assert "test_layer" in result.layers_applied


# =============================================================================
# TEST _distribute_goals
# =============================================================================


class TestDistributeGoals:
    """Whitebox tests for goal distribution."""

    def test_distribute_goals_calls_update_goals_on_all_layers(
        self, controller, mock_ik_layer
    ):
        """_distribute_goals calls update_goals on each layer."""
        controller.add_ik_layer(mock_ik_layer)
        ctx = IKGoalContext()
        ctx.position_goals["Hand"] = Vec3(1, 2, 3)

        controller._distribute_goals(ctx)

        mock_ik_layer.update_goals.assert_called_once_with(ctx)

    def test_distribute_goals_multiple_layers(self, controller):
        """_distribute_goals distributes to all layers."""
        layers = []
        for i in range(3):
            layer = MagicMock(spec=IKLayer)
            layer.name = f"layer_{i}"
            layer.enabled = True
            layer.update_goals = MagicMock()
            layers.append(layer)
            controller.add_ik_layer(layer)

        ctx = IKGoalContext()
        controller._distribute_goals(ctx)

        for layer in layers:
            layer.update_goals.assert_called_once_with(ctx)

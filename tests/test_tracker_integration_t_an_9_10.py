"""
Tests for Foundation Tracker Integration (T-AN-9.10).

This test suite covers:
- TrackedDescriptor wrapping
- Dirty flag propagation
- Callback firing on changes
- Skip evaluation when clean
- State machine transition tracking
- Multiple parameter tracking
- Clear/mark dirty operations
- AnimationParameterSet functionality
- AnimationStateTracker functionality
- TrackedAnimationComponent integration
- TrackedIKGoal tracking

50+ test cases organized into logical groups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, Mock, call
import pytest

from engine.animation.tracker_integration import (
    # Core tracker types
    TrackedDescriptor,
    TrackedField,
    AnimationTracker,
    # Parameter tracking
    AnimationTrackedParameter,
    AnimationParameterSet,
    # State tracking
    AnimationStateTracker,
    StateChangeEvent,
    StateTransitionRecord,
    # Component integration
    TrackedAnimationComponent,
    TrackedIKGoal,
    # Utilities
    clear_dirty,
    mark_dirty,
    all_dirty,
    any_dirty,
    # Type subscriptions
    ChangeCallback,
    TypeSubscription,
    AnimationStateSubscription,
    # Integration helpers
    wrap_parameter,
    wrap_state_machine,
    create_tracked_parameter_set,
)
from engine.animation.graph import (
    AnimationGraph,
    GraphParameter,
    ParameterType,
    StateMachine,
    AnimationState,
    Pose,
    GraphContext,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_graph() -> AnimationGraph:
    """Create a sample animation graph with parameters."""
    graph = AnimationGraph("test_graph")
    graph.add_parameter(GraphParameter("speed", ParameterType.FLOAT, default_value=0.0))
    graph.add_parameter(GraphParameter("direction", ParameterType.FLOAT, default_value=0.0))
    graph.add_parameter(GraphParameter("is_grounded", ParameterType.BOOL, default_value=True))
    return graph


@pytest.fixture
def sample_state_machine() -> StateMachine:
    """Create a sample state machine."""
    sm = StateMachine("test_sm", initial_state="idle")
    sm.add_state(AnimationState("idle"))
    sm.add_state(AnimationState("walk"))
    sm.add_state(AnimationState("run"))
    return sm


@pytest.fixture
def parameter_set() -> AnimationParameterSet:
    """Create a sample parameter set."""
    params = AnimationParameterSet()
    params.register("speed", ParameterType.FLOAT, 0.0)
    params.register("direction", ParameterType.FLOAT, 0.0)
    params.register("state_id", ParameterType.INT, 0)
    params.register("is_active", ParameterType.BOOL, True)
    return params


# =============================================================================
# TRACKED DESCRIPTOR TESTS
# =============================================================================


class TestTrackedDescriptor:
    """Tests for TrackedDescriptor."""

    def test_descriptor_initialization(self):
        """Test TrackedDescriptor initializes with default value."""
        class TestClass:
            speed = TrackedDescriptor[float]("speed", 0.0)

        obj = TestClass()
        assert obj.speed == 0.0

    def test_descriptor_set_value(self):
        """Test setting descriptor value."""
        class TestClass:
            speed = TrackedDescriptor[float]("speed", 0.0)

        obj = TestClass()
        obj.speed = 5.0
        assert obj.speed == 5.0

    def test_descriptor_tracks_dirty(self):
        """Test descriptor marks dirty on change."""
        class TestClass:
            speed = TrackedDescriptor[float]("speed", 0.0)

        obj = TestClass()
        descriptor = TestClass.__dict__["speed"]

        assert not descriptor.is_dirty(obj)
        obj.speed = 5.0
        assert descriptor.is_dirty(obj)

    def test_descriptor_no_dirty_on_same_value(self):
        """Test descriptor doesn't mark dirty when value unchanged."""
        class TestClass:
            speed = TrackedDescriptor[float]("speed", 0.0)

        obj = TestClass()
        descriptor = TestClass.__dict__["speed"]

        obj.speed = 0.0  # Same as default
        assert not descriptor.is_dirty(obj)

    def test_descriptor_clear_dirty(self):
        """Test clearing dirty flag."""
        class TestClass:
            speed = TrackedDescriptor[float]("speed", 0.0)

        obj = TestClass()
        descriptor = TestClass.__dict__["speed"]

        obj.speed = 5.0
        assert descriptor.is_dirty(obj)
        descriptor.clear_dirty(obj)
        assert not descriptor.is_dirty(obj)

    def test_descriptor_mark_dirty(self):
        """Test manually marking dirty."""
        class TestClass:
            speed = TrackedDescriptor[float]("speed", 0.0)

        obj = TestClass()
        descriptor = TestClass.__dict__["speed"]

        assert not descriptor.is_dirty(obj)
        descriptor.mark_dirty(obj)
        assert descriptor.is_dirty(obj)

    def test_descriptor_version_increments(self):
        """Test version increments on change."""
        class TestClass:
            speed = TrackedDescriptor[float]("speed", 0.0)

        obj = TestClass()
        descriptor = TestClass.__dict__["speed"]

        assert descriptor.get_version(obj) == 0
        obj.speed = 1.0
        assert descriptor.get_version(obj) == 1
        obj.speed = 2.0
        assert descriptor.get_version(obj) == 2

    def test_descriptor_callback(self):
        """Test descriptor fires callback on change."""
        callback_calls = []

        def callback(name, old, new):
            callback_calls.append((name, old, new))

        class TestClass:
            speed = TrackedDescriptor[float]("speed", 0.0, notify_callback=callback)

        obj = TestClass()
        obj.speed = 5.0

        assert len(callback_calls) == 1
        assert callback_calls[0] == ("speed", 0.0, 5.0)

    def test_descriptor_set_callback_runtime(self):
        """Test setting callback at runtime."""
        callback_calls = []

        def callback(name, old, new):
            callback_calls.append((name, old, new))

        class TestClass:
            speed = TrackedDescriptor[float]("speed", 0.0)

        obj = TestClass()
        descriptor = TestClass.__dict__["speed"]
        descriptor.set_callback(obj, callback)

        obj.speed = 5.0
        assert len(callback_calls) == 1

    def test_multiple_descriptors(self):
        """Test multiple descriptors on same class."""
        class TestClass:
            speed = TrackedDescriptor[float]("speed", 0.0)
            direction = TrackedDescriptor[float]("direction", 0.0)

        obj = TestClass()
        speed_desc = TestClass.__dict__["speed"]
        dir_desc = TestClass.__dict__["direction"]

        obj.speed = 5.0
        assert speed_desc.is_dirty(obj)
        assert not dir_desc.is_dirty(obj)


# =============================================================================
# TRACKED FIELD TESTS
# =============================================================================


class TestTrackedField:
    """Tests for TrackedField."""

    def test_field_initialization(self):
        """Test TrackedField initializes correctly."""
        field_obj = TrackedField("speed", 0.0)
        assert field_obj.name == "speed"
        assert field_obj.value == 0.0
        assert not field_obj.is_dirty
        assert field_obj.version == 0

    def test_field_get_set(self):
        """Test get/set methods."""
        field_obj = TrackedField("speed", 0.0)
        assert field_obj.get() == 0.0
        field_obj.set(5.0)
        assert field_obj.get() == 5.0

    def test_field_dirty_on_change(self):
        """Test field marks dirty on change."""
        field_obj = TrackedField("speed", 0.0)
        assert not field_obj.is_dirty
        field_obj.set(5.0)
        assert field_obj.is_dirty

    def test_field_version_increments(self):
        """Test version increments on change."""
        field_obj = TrackedField("speed", 0.0)
        assert field_obj.version == 0
        field_obj.set(1.0)
        assert field_obj.version == 1
        field_obj.set(2.0)
        assert field_obj.version == 2

    def test_field_clear_dirty(self):
        """Test clearing dirty flag."""
        field_obj = TrackedField("speed", 0.0)
        field_obj.set(5.0)
        field_obj.clear_dirty()
        assert not field_obj.is_dirty

    def test_field_mark_dirty(self):
        """Test manually marking dirty."""
        field_obj = TrackedField("speed", 0.0)
        field_obj.mark_dirty()
        assert field_obj.is_dirty

    def test_field_callback(self):
        """Test field fires callback on change."""
        callback_calls = []

        def callback(name, old, new):
            callback_calls.append((name, old, new))

        field_obj = TrackedField("speed", 0.0)
        field_obj.on_change(callback)
        field_obj.set(5.0)

        assert len(callback_calls) == 1
        assert callback_calls[0] == ("speed", 0.0, 5.0)

    def test_field_multiple_callbacks(self):
        """Test field fires multiple callbacks."""
        calls1 = []
        calls2 = []

        field_obj = TrackedField("speed", 0.0)
        field_obj.on_change(lambda n, o, v: calls1.append(v))
        field_obj.on_change(lambda n, o, v: calls2.append(v))
        field_obj.set(5.0)

        assert calls1 == [5.0]
        assert calls2 == [5.0]

    def test_field_remove_callback(self):
        """Test removing callback."""
        calls = []

        def callback(name, old, new):
            calls.append(new)

        field_obj = TrackedField("speed", 0.0)
        field_obj.on_change(callback)
        field_obj.set(1.0)
        assert field_obj.remove_callback(callback)
        field_obj.set(2.0)

        assert calls == [1.0]  # Only first change

    def test_field_set_returns_changed(self):
        """Test set returns whether value changed."""
        field_obj = TrackedField("speed", 0.0)
        assert field_obj.set(5.0) is True
        assert field_obj.set(5.0) is False  # No change


# =============================================================================
# ANIMATION TRACKER TESTS
# =============================================================================


class TestAnimationTracker:
    """Tests for AnimationTracker."""

    def test_tracker_initialization(self):
        """Test tracker initializes empty."""
        tracker = AnimationTracker()
        assert len(tracker.fields) == 0
        assert tracker.get_version() == 0

    def test_tracker_add_field(self):
        """Test adding fields to tracker."""
        tracker = AnimationTracker()
        field_obj = tracker.add_field("speed", 0.0)

        assert "speed" in tracker.fields
        assert field_obj.name == "speed"

    def test_tracker_remove_field(self):
        """Test removing fields from tracker."""
        tracker = AnimationTracker()
        tracker.add_field("speed", 0.0)
        assert tracker.remove_field("speed")
        assert "speed" not in tracker.fields

    def test_tracker_get_set(self):
        """Test get/set methods."""
        tracker = AnimationTracker()
        tracker.add_field("speed", 0.0)

        assert tracker.get("speed") == 0.0
        tracker.set("speed", 5.0)
        assert tracker.get("speed") == 5.0

    def test_tracker_is_dirty(self):
        """Test field-level dirty checking."""
        tracker = AnimationTracker()
        tracker.add_field("speed", 0.0)

        assert not tracker.is_dirty("speed")
        tracker.set("speed", 5.0)
        assert tracker.is_dirty("speed")

    def test_tracker_any_dirty(self):
        """Test any_dirty aggregate check."""
        tracker = AnimationTracker()
        tracker.add_field("speed", 0.0)
        tracker.add_field("direction", 0.0)

        assert not tracker.any_dirty()
        tracker.set("speed", 5.0)
        assert tracker.any_dirty()

    def test_tracker_all_dirty(self):
        """Test all_dirty (alias for any_dirty)."""
        tracker = AnimationTracker()
        tracker.add_field("speed", 0.0)
        tracker.set("speed", 5.0)
        assert tracker.all_dirty()

    def test_tracker_get_dirty_fields(self):
        """Test getting list of dirty field names."""
        tracker = AnimationTracker()
        tracker.add_field("speed", 0.0)
        tracker.add_field("direction", 0.0)

        tracker.set("speed", 5.0)
        dirty = tracker.get_dirty_fields()

        assert dirty == frozenset({"speed"})

    def test_tracker_clear_dirty(self):
        """Test clearing single field dirty flag."""
        tracker = AnimationTracker()
        tracker.add_field("speed", 0.0)
        tracker.set("speed", 5.0)
        tracker.clear_dirty("speed")
        assert not tracker.is_dirty("speed")

    def test_tracker_clear_all_dirty(self):
        """Test clearing all dirty flags."""
        tracker = AnimationTracker()
        tracker.add_field("speed", 0.0)
        tracker.add_field("direction", 0.0)
        tracker.set("speed", 5.0)
        tracker.set("direction", 1.0)
        tracker.clear_all_dirty()
        assert not tracker.any_dirty()

    def test_tracker_mark_dirty(self):
        """Test manually marking field dirty."""
        tracker = AnimationTracker()
        tracker.add_field("speed", 0.0)
        tracker.mark_dirty("speed")
        assert tracker.is_dirty("speed")

    def test_tracker_mark_all_dirty(self):
        """Test marking all fields dirty."""
        tracker = AnimationTracker()
        tracker.add_field("speed", 0.0)
        tracker.add_field("direction", 0.0)
        tracker.mark_all_dirty()
        assert tracker.is_dirty("speed")
        assert tracker.is_dirty("direction")

    def test_tracker_on_change_field(self):
        """Test field-level change subscription."""
        calls = []
        tracker = AnimationTracker()
        tracker.add_field("speed", 0.0)
        tracker.on_change("speed", lambda n, o, v: calls.append(v))
        tracker.set("speed", 5.0)
        assert 5.0 in calls

    def test_tracker_version_increments(self):
        """Test global version increments on change."""
        tracker = AnimationTracker()
        tracker.add_field("speed", 0.0)

        assert tracker.get_version() == 0
        tracker.set("speed", 1.0)
        assert tracker.get_version() == 1


# =============================================================================
# ANIMATION TRACKED PARAMETER TESTS
# =============================================================================


class TestAnimationTrackedParameter:
    """Tests for AnimationTrackedParameter."""

    def test_parameter_wraps_graph_parameter(self):
        """Test parameter wraps GraphParameter correctly."""
        param = GraphParameter("speed", ParameterType.FLOAT, 0.0)
        tracked = AnimationTrackedParameter(parameter=param)

        assert tracked.name == "speed"
        assert tracked.param_type == ParameterType.FLOAT
        assert tracked.value == 0.0

    def test_parameter_syncs_value(self):
        """Test value syncs to underlying parameter."""
        param = GraphParameter("speed", ParameterType.FLOAT, 0.0)
        tracked = AnimationTrackedParameter(parameter=param)

        tracked.set(5.0)
        assert param.value == 5.0

    def test_parameter_tracks_dirty(self):
        """Test parameter tracks dirty state."""
        param = GraphParameter("speed", ParameterType.FLOAT, 0.0)
        tracked = AnimationTrackedParameter(parameter=param)

        assert not tracked.is_dirty
        tracked.set(5.0)
        assert tracked.is_dirty

    def test_parameter_tracks_version(self):
        """Test parameter tracks version."""
        param = GraphParameter("speed", ParameterType.FLOAT, 0.0)
        tracked = AnimationTrackedParameter(parameter=param)

        assert tracked.version == 0
        tracked.set(1.0)
        assert tracked.version == 1

    def test_parameter_on_change(self):
        """Test parameter change callback."""
        calls = []
        param = GraphParameter("speed", ParameterType.FLOAT, 0.0)
        tracked = AnimationTrackedParameter(parameter=param)
        tracked.on_change(lambda n, o, v: calls.append((n, v)))
        tracked.set(5.0)
        assert calls == [("speed", 5.0)]

    def test_parameter_sync_from_parameter(self):
        """Test syncing from underlying parameter."""
        param = GraphParameter("speed", ParameterType.FLOAT, 0.0)
        tracked = AnimationTrackedParameter(parameter=param)

        param.value = 10.0
        tracked.sync_from_parameter()
        assert tracked.value == 10.0


# =============================================================================
# ANIMATION PARAMETER SET TESTS
# =============================================================================


class TestAnimationParameterSet:
    """Tests for AnimationParameterSet."""

    def test_parameter_set_initialization(self):
        """Test parameter set initializes empty."""
        params = AnimationParameterSet()
        assert len(params) == 0

    def test_parameter_set_register(self):
        """Test registering parameters."""
        params = AnimationParameterSet()
        params.register("speed", ParameterType.FLOAT, 0.0)
        assert "speed" in params
        assert len(params) == 1

    def test_parameter_set_unregister(self):
        """Test unregistering parameters."""
        params = AnimationParameterSet()
        params.register("speed", ParameterType.FLOAT, 0.0)
        assert params.unregister("speed")
        assert "speed" not in params

    def test_parameter_set_get_set(self, parameter_set):
        """Test get/set methods."""
        assert parameter_set.get("speed") == 0.0
        parameter_set.set("speed", 5.0)
        assert parameter_set.get("speed") == 5.0

    def test_parameter_set_any_dirty(self, parameter_set):
        """Test any_dirty check."""
        assert not parameter_set.any_dirty()
        parameter_set.set("speed", 5.0)
        assert parameter_set.any_dirty()

    def test_parameter_set_all_dirty(self, parameter_set):
        """Test all_dirty (alias)."""
        parameter_set.set("speed", 5.0)
        assert parameter_set.all_dirty()

    def test_parameter_set_get_dirty_names(self, parameter_set):
        """Test getting dirty parameter names."""
        parameter_set.set("speed", 5.0)
        parameter_set.set("state_id", 1)
        dirty = parameter_set.get_dirty_names()
        assert dirty == frozenset({"speed", "state_id"})

    def test_parameter_set_clear_dirty(self, parameter_set):
        """Test clearing single parameter dirty flag."""
        parameter_set.set("speed", 5.0)
        parameter_set.clear_dirty("speed")
        assert not parameter_set.is_dirty("speed")

    def test_parameter_set_clear_all_dirty(self, parameter_set):
        """Test clearing all dirty flags."""
        parameter_set.set("speed", 5.0)
        parameter_set.set("direction", 1.0)
        parameter_set.clear_all_dirty()
        assert not parameter_set.any_dirty()

    def test_parameter_set_mark_dirty(self, parameter_set):
        """Test marking parameter dirty."""
        parameter_set.mark_dirty("speed")
        assert parameter_set.is_dirty("speed")

    def test_parameter_set_mark_all_dirty(self, parameter_set):
        """Test marking all parameters dirty."""
        parameter_set.mark_all_dirty()
        assert parameter_set.is_dirty("speed")
        assert parameter_set.is_dirty("direction")

    def test_parameter_set_on_change(self, parameter_set):
        """Test parameter change callback."""
        calls = []
        parameter_set.on_change("speed", lambda n, o, v: calls.append(v))
        parameter_set.set("speed", 5.0)
        assert 5.0 in calls

    def test_parameter_set_type_subscription(self):
        """Test type-level subscription."""
        calls = []
        params = AnimationParameterSet()
        params.register("speed", ParameterType.FLOAT, 0.0)
        params.on_change(float, lambda n, o, v: calls.append((n, v)))
        params.set("speed", 5.0)
        assert ("speed", 5.0) in calls

    def test_parameter_set_sync_to_graph(self, sample_graph):
        """Test syncing to AnimationGraph."""
        params = create_tracked_parameter_set(sample_graph)
        params.set("speed", 10.0)
        synced = params.sync_to_graph(sample_graph)
        assert synced == 1
        assert sample_graph.parameters["speed"].value == 10.0

    def test_parameter_set_from_graph(self, sample_graph):
        """Test creating parameter set from graph."""
        params = AnimationParameterSet(graph=sample_graph)
        assert "speed" in params
        assert "direction" in params
        assert "is_grounded" in params


# =============================================================================
# ANIMATION STATE TRACKER TESTS
# =============================================================================


class TestAnimationStateTracker:
    """Tests for AnimationStateTracker."""

    def test_state_tracker_initialization(self):
        """Test state tracker initializes empty."""
        tracker = AnimationStateTracker()
        assert tracker.current_state == ""
        assert tracker.previous_state == ""
        assert not tracker.is_dirty

    def test_state_tracker_set_state(self):
        """Test setting current state."""
        tracker = AnimationStateTracker()
        tracker.set_state("idle")
        assert tracker.current_state == "idle"

    def test_state_tracker_tracks_previous(self):
        """Test tracking previous state."""
        tracker = AnimationStateTracker()
        tracker.set_state("idle")
        tracker.set_state("walk")
        assert tracker.current_state == "walk"
        assert tracker.previous_state == "idle"

    def test_state_tracker_dirty_on_change(self):
        """Test dirty flag on state change."""
        tracker = AnimationStateTracker()
        tracker.set_state("idle")
        assert tracker.is_dirty

    def test_state_tracker_no_dirty_on_same_state(self):
        """Test no dirty when setting same state."""
        tracker = AnimationStateTracker()
        tracker.set_state("idle")
        tracker.clear_dirty()
        tracker.set_state("idle")
        assert not tracker.is_dirty

    def test_state_tracker_clear_dirty(self):
        """Test clearing dirty flag."""
        tracker = AnimationStateTracker()
        tracker.set_state("idle")
        tracker.clear_dirty()
        assert not tracker.is_dirty

    def test_state_tracker_mark_dirty(self):
        """Test marking dirty."""
        tracker = AnimationStateTracker()
        tracker.mark_dirty()
        assert tracker.is_dirty

    def test_state_tracker_transition_to(self):
        """Test transition with duration."""
        tracker = AnimationStateTracker()
        tracker.set_state("idle")
        tracker.clear_dirty()
        tracker.transition_to("walk", duration=0.3)
        assert tracker.current_state == "walk"
        assert tracker.is_dirty

    def test_state_tracker_history(self):
        """Test transition history tracking."""
        tracker = AnimationStateTracker()
        tracker.set_state("idle")
        tracker.set_state("walk")
        tracker.set_state("run")

        assert len(tracker.history) == 3
        assert tracker.history[0].to_state == "idle"
        assert tracker.history[1].to_state == "walk"
        assert tracker.history[2].to_state == "run"

    def test_state_tracker_validate_transition(self):
        """Test transition validation."""
        tracker = AnimationStateTracker()
        tracker.set_state("idle")
        tracker.set_state("walk")

        assert tracker.validate_transition("idle", "walk")
        assert not tracker.validate_transition("walk", "run")

    def test_state_tracker_get_transition_count(self):
        """Test counting transitions."""
        tracker = AnimationStateTracker()
        tracker.set_state("idle")
        tracker.set_state("walk")
        tracker.set_state("idle")
        tracker.set_state("walk")

        assert tracker.get_transition_count("idle", "walk") == 2

    def test_state_tracker_on_change_callback(self):
        """Test state change callback."""
        events = []
        tracker = AnimationStateTracker()
        tracker.on_change(lambda e: events.append(e))
        tracker.set_state("idle")

        assert len(events) == 1
        assert events[0].new_state == "idle"

    def test_state_tracker_on_change_type_subscription(self):
        """Test type-level subscription."""
        events = []
        tracker = AnimationStateTracker()
        tracker.on_change(StateChangeEvent, lambda e: events.append(e))
        tracker.set_state("idle")

        assert len(events) == 1

    def test_state_tracker_event_data(self):
        """Test StateChangeEvent data."""
        events = []
        tracker = AnimationStateTracker()
        tracker.on_change(lambda e: events.append(e))
        tracker.set_state("idle")
        tracker.transition_to("walk", duration=0.5)

        assert events[1].old_state == "idle"
        assert events[1].new_state == "walk"
        assert events[1].transition_duration == 0.5
        assert events[1].is_transition is True

    def test_state_tracker_clear_history(self):
        """Test clearing history."""
        tracker = AnimationStateTracker()
        tracker.set_state("idle")
        tracker.set_state("walk")
        tracker.clear_history()
        assert len(tracker.history) == 0

    def test_state_tracker_max_history(self):
        """Test history size limit."""
        tracker = AnimationStateTracker(max_history=5)
        for i in range(10):
            tracker.set_state(f"state_{i}")
        assert len(tracker.history) == 5


# =============================================================================
# TRACKED ANIMATION COMPONENT TESTS
# =============================================================================


class TestTrackedAnimationComponent:
    """Tests for TrackedAnimationComponent."""

    def test_component_initialization(self):
        """Test component initializes with defaults."""
        comp = TrackedAnimationComponent()
        assert comp.enabled is True
        assert comp._needs_evaluation is True

    def test_component_needs_evaluation_initial(self):
        """Test needs_evaluation on first frame."""
        comp = TrackedAnimationComponent()
        assert comp.needs_evaluation(0)

    def test_component_needs_evaluation_dirty_params(self):
        """Test needs_evaluation when params dirty."""
        comp = TrackedAnimationComponent()
        comp.parameters.register("speed", ParameterType.FLOAT, 0.0)
        comp.mark_evaluated(0)
        comp.parameters.set("speed", 5.0)
        assert comp.needs_evaluation(1)

    def test_component_needs_evaluation_dirty_state(self):
        """Test needs_evaluation when state dirty."""
        comp = TrackedAnimationComponent()
        comp.mark_evaluated(0)
        comp.state_tracker.set_state("walk")
        assert comp.needs_evaluation(1)

    def test_component_skip_evaluation_when_clean(self):
        """Test skipping evaluation when clean."""
        comp = TrackedAnimationComponent()
        comp.mark_evaluated(0)
        assert not comp.needs_evaluation(1)

    def test_component_disabled_skips_evaluation(self):
        """Test disabled component skips evaluation."""
        comp = TrackedAnimationComponent()
        comp.enabled = False
        assert not comp.needs_evaluation(0)

    def test_component_mark_evaluated(self):
        """Test marking component as evaluated."""
        comp = TrackedAnimationComponent()
        comp.parameters.register("speed", ParameterType.FLOAT, 0.0)
        comp.parameters.set("speed", 5.0)
        comp.state_tracker.set_state("walk")
        comp.mark_evaluated(1)

        assert comp._last_eval_frame == 1
        assert not comp.parameters.any_dirty()
        assert not comp.state_tracker.is_dirty

    def test_component_invalidate(self):
        """Test invalidating component."""
        comp = TrackedAnimationComponent()
        comp.parameters.register("speed", ParameterType.FLOAT, 0.0)
        comp.mark_evaluated(0)
        comp.invalidate()

        assert comp._needs_evaluation is True
        assert comp.parameters.any_dirty()
        assert comp.state_tracker.is_dirty

    def test_component_set_parameter(self):
        """Test setting parameter through component."""
        comp = TrackedAnimationComponent()
        comp.parameters.register("speed", ParameterType.FLOAT, 0.0)
        comp.set_parameter("speed", 5.0)
        assert comp.get_parameter("speed") == 5.0

    def test_component_set_state(self):
        """Test setting state through component."""
        comp = TrackedAnimationComponent()
        comp.set_state("walk")
        assert comp.state_tracker.current_state == "walk"

    def test_component_transition_to(self):
        """Test transition through component."""
        comp = TrackedAnimationComponent()
        comp.set_state("idle")
        comp.transition_to("walk", 0.3)
        assert comp.state_tracker.current_state == "walk"


# =============================================================================
# TRACKED IK GOAL TESTS
# =============================================================================


class TestTrackedIKGoal:
    """Tests for TrackedIKGoal."""

    def test_ik_goal_initialization(self):
        """Test IK goal initializes with default fields."""
        goal = TrackedIKGoal(goal_id="hand_l", chain_name="left_arm")
        assert goal.goal_id == "hand_l"
        assert goal.chain_name == "left_arm"
        assert goal.enabled is True

    def test_ik_goal_set_position(self):
        """Test setting goal position."""
        goal = TrackedIKGoal(goal_id="hand_l", chain_name="left_arm")
        goal.set_position(1.0, 2.0, 3.0)

        assert goal.tracker.get("position_x") == 1.0
        assert goal.tracker.get("position_y") == 2.0
        assert goal.tracker.get("position_z") == 3.0

    def test_ik_goal_set_rotation(self):
        """Test setting goal rotation."""
        goal = TrackedIKGoal(goal_id="hand_l", chain_name="left_arm")
        goal.set_rotation(0.0, 0.707, 0.0, 0.707)

        assert goal.tracker.get("rotation_x") == 0.0
        assert goal.tracker.get("rotation_y") == 0.707
        assert goal.tracker.get("rotation_z") == 0.0
        assert goal.tracker.get("rotation_w") == 0.707

    def test_ik_goal_set_weight(self):
        """Test setting goal weight."""
        goal = TrackedIKGoal(goal_id="hand_l", chain_name="left_arm")
        goal.set_weight(0.5)
        assert goal.tracker.get("weight") == 0.5

    def test_ik_goal_is_dirty(self):
        """Test dirty detection."""
        goal = TrackedIKGoal(goal_id="hand_l", chain_name="left_arm")
        assert not goal.is_dirty
        goal.set_position(1.0, 0.0, 0.0)
        assert goal.is_dirty

    def test_ik_goal_clear_dirty(self):
        """Test clearing dirty flags."""
        goal = TrackedIKGoal(goal_id="hand_l", chain_name="left_arm")
        goal.set_position(1.0, 0.0, 0.0)
        goal.clear_dirty()
        assert not goal.is_dirty

    def test_ik_goal_needs_solving(self):
        """Test needs_solving check."""
        goal = TrackedIKGoal(goal_id="hand_l", chain_name="left_arm")
        goal.set_position(1.0, 0.0, 0.0)
        assert goal.needs_solving()

        goal.enabled = False
        assert not goal.needs_solving()


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_clear_dirty_tracker(self):
        """Test clear_dirty with AnimationTracker."""
        tracker = AnimationTracker()
        tracker.add_field("speed", 0.0)
        tracker.set("speed", 5.0)
        clear_dirty(tracker)
        assert not tracker.any_dirty()

    def test_clear_dirty_parameter_set(self, parameter_set):
        """Test clear_dirty with AnimationParameterSet."""
        parameter_set.set("speed", 5.0)
        clear_dirty(parameter_set)
        assert not parameter_set.any_dirty()

    def test_clear_dirty_state_tracker(self):
        """Test clear_dirty with AnimationStateTracker."""
        tracker = AnimationStateTracker()
        tracker.set_state("walk")
        clear_dirty(tracker)
        assert not tracker.is_dirty

    def test_mark_dirty_tracker(self):
        """Test mark_dirty with AnimationTracker."""
        tracker = AnimationTracker()
        tracker.add_field("speed", 0.0)
        mark_dirty(tracker, "speed")
        assert tracker.is_dirty("speed")

    def test_mark_dirty_all(self, parameter_set):
        """Test mark_dirty all fields."""
        mark_dirty(parameter_set)
        assert parameter_set.any_dirty()

    def test_all_dirty_function(self, parameter_set):
        """Test all_dirty utility function."""
        assert not all_dirty(parameter_set)
        parameter_set.set("speed", 5.0)
        assert all_dirty(parameter_set)

    def test_any_dirty_function(self, parameter_set):
        """Test any_dirty utility function."""
        assert not any_dirty(parameter_set)
        parameter_set.set("speed", 5.0)
        assert any_dirty(parameter_set)

    def test_wrap_parameter(self):
        """Test wrap_parameter helper."""
        param = GraphParameter("speed", ParameterType.FLOAT, 0.0)
        tracked = wrap_parameter(param)
        assert isinstance(tracked, AnimationTrackedParameter)
        assert tracked.name == "speed"

    def test_wrap_state_machine(self, sample_state_machine):
        """Test wrap_state_machine helper."""
        tracker = wrap_state_machine(sample_state_machine)
        assert isinstance(tracker, AnimationStateTracker)
        # Note: State machine hasn't been initialized yet, so current_state is empty
        # The tracker is set up to track state changes once the SM is evaluated
        assert tracker.current_state == "" or tracker.current_state == "idle"

    def test_create_tracked_parameter_set(self, sample_graph):
        """Test create_tracked_parameter_set helper."""
        params = create_tracked_parameter_set(sample_graph)
        assert isinstance(params, AnimationParameterSet)
        assert "speed" in params
        assert "direction" in params


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for tracker system."""

    def test_full_workflow(self, sample_graph):
        """Test complete workflow from graph to evaluation."""
        # Create tracked parameter set from graph
        params = create_tracked_parameter_set(sample_graph)

        # Create component
        comp = TrackedAnimationComponent()
        comp.parameters = params
        comp.graph = sample_graph

        # Initial state
        assert comp.needs_evaluation(0)

        # Evaluate and mark clean
        comp.mark_evaluated(0)
        assert not comp.needs_evaluation(1)

        # Change parameter
        comp.set_parameter("speed", 5.0)
        assert comp.needs_evaluation(1)

        # Sync to graph
        synced = comp.sync_to_graph()
        assert synced == 1
        assert sample_graph.parameters["speed"].value == 5.0

    def test_state_machine_tracking(self, sample_state_machine):
        """Test state machine with tracker."""
        tracker = wrap_state_machine(sample_state_machine)

        # Track state changes
        events = []
        tracker.on_change(lambda e: events.append(e))

        # Set initial state first (simulating SM initialization)
        tracker.set_state("idle")

        # Simulate state transitions
        tracker.transition_to("walk", 0.3)
        tracker.transition_to("run", 0.2)

        assert len(events) == 3  # idle + walk + run
        assert events[1].new_state == "walk"
        assert events[2].new_state == "run"

        # Validate transitions
        assert tracker.validate_transition("idle", "walk")
        assert tracker.validate_transition("walk", "run")

    def test_dirty_flag_optimization(self, parameter_set):
        """Test dirty flags enable evaluation skipping."""
        comp = TrackedAnimationComponent()
        comp.parameters = parameter_set

        # Count evaluations
        eval_count = 0

        def evaluate_if_needed(frame):
            nonlocal eval_count
            if comp.needs_evaluation(frame):
                eval_count += 1
                comp.mark_evaluated(frame)

        # First frame: evaluate
        evaluate_if_needed(0)
        assert eval_count == 1

        # Second frame: skip (clean)
        evaluate_if_needed(1)
        assert eval_count == 1

        # Change parameter: evaluate
        comp.set_parameter("speed", 5.0)
        evaluate_if_needed(2)
        assert eval_count == 2

        # Another clean frame: skip
        evaluate_if_needed(3)
        assert eval_count == 2

    def test_multiple_callbacks(self, parameter_set):
        """Test multiple callbacks fire correctly."""
        speed_changes = []
        direction_changes = []
        all_changes = []

        parameter_set.on_change("speed", lambda n, o, v: speed_changes.append(v))
        parameter_set.on_change("direction", lambda n, o, v: direction_changes.append(v))
        parameter_set.on_change(float, lambda n, o, v: all_changes.append((n, v)))

        parameter_set.set("speed", 5.0)
        parameter_set.set("direction", 1.0)

        assert speed_changes == [5.0]
        assert direction_changes == [1.0]
        assert ("speed", 5.0) in all_changes
        assert ("direction", 1.0) in all_changes

    def test_ik_goal_optimization(self):
        """Test IK goal dirty tracking optimization."""
        goals = [
            TrackedIKGoal("hand_l", "left_arm"),
            TrackedIKGoal("hand_r", "right_arm"),
            TrackedIKGoal("foot_l", "left_leg"),
            TrackedIKGoal("foot_r", "right_leg"),
        ]

        # Only update one goal
        goals[0].set_position(1.0, 2.0, 3.0)

        # Check which goals need solving
        needs_solving = [g for g in goals if g.needs_solving()]
        assert len(needs_solving) == 1
        assert needs_solving[0].goal_id == "hand_l"

        # Clear dirty and check again
        for g in goals:
            g.clear_dirty()

        needs_solving = [g for g in goals if g.needs_solving()]
        assert len(needs_solving) == 0

"""
Whitebox tests for StateMachineBuilder.

Tests internal implementation details:
- Fluent builder API with method chaining
- add_state() with auto-detection and explicit kwargs
- add_transition() with all StateTransition fields
- add_any_state_transition() internal mechanics
- set_initial() and validation at build time
- build() validation with clear error messages
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest


# ============================================================================
# IMPORTS UNDER TEST
# ============================================================================


from engine.animation.graph.state_machine import (
    AnimationState,
    BlendCurve,
    InterruptMode,
    MotionMode,
    StateTransition,
    StateMachine,
    StateMachineBuilder,
    StateMachineBuilderError,
    TransitionCondition,
    TransitionSyncMode,
)
from engine.animation.graph.animation_graph import (
    AnimationGraph,
    AnimationNode,
    GraphContext,
)


# ============================================================================
# FIXTURES
# ============================================================================


@dataclass
class MockAnimationClip:
    """Mock animation clip for testing."""
    duration: float = 1.0
    name: str = "mock_clip"

    def sample(self, time: float, bone_count: int) -> Any:
        """Mock sample method."""
        return MagicMock()


class MockAnimationNode(AnimationNode):
    """Mock animation node for testing."""
    _abstract = False

    def evaluate(self, context: GraphContext) -> Any:
        return MagicMock()


@pytest.fixture
def mock_clip() -> MockAnimationClip:
    """Create a mock animation clip."""
    return MockAnimationClip(duration=2.0, name="test_clip")


@pytest.fixture
def mock_graph() -> AnimationGraph:
    """Create a mock animation graph."""
    return AnimationGraph("test_graph")


@pytest.fixture
def mock_node() -> MockAnimationNode:
    """Create a mock animation node."""
    return MockAnimationNode("test_node")


@pytest.fixture
def basic_builder() -> StateMachineBuilder:
    """Create a basic builder with some states."""
    return StateMachineBuilder("test_sm")


# ============================================================================
# TEST: Fluent Builder API - Method Chaining
# ============================================================================


class TestFluentBuilderAPI:
    """Test that all builder methods return self for chaining."""

    def test_add_state_returns_self(self, mock_clip: MockAnimationClip) -> None:
        """add_state() returns self for chaining."""
        builder = StateMachineBuilder("test")
        result = builder.add_state("idle", mock_clip)

        assert result is builder

    def test_set_initial_returns_self(self, mock_clip: MockAnimationClip) -> None:
        """set_initial() returns self for chaining."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        result = builder.set_initial("idle")

        assert result is builder

    def test_add_transition_returns_self(self, mock_clip: MockAnimationClip) -> None:
        """add_transition() returns self for chaining."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        result = builder.add_transition("idle", "walk")

        assert result is builder

    def test_add_any_state_transition_returns_self(self, mock_clip: MockAnimationClip) -> None:
        """add_any_state_transition() returns self for chaining."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("hurt", mock_clip)
        result = builder.add_any_state_transition("hurt")

        assert result is builder

    def test_full_chain_example(self, mock_clip: MockAnimationClip, mock_graph: AnimationGraph) -> None:
        """Full fluent API chain works as documented."""
        sm = (
            StateMachineBuilder("locomotion")
            .add_state("idle", mock_clip)
            .add_state("walk", mock_clip, speed=1.2)
            .add_state("run", mock_graph)
            .set_initial("idle")
            .add_transition("idle", "walk", TransitionCondition.is_true("moving"))
            .add_transition("walk", "run", TransitionCondition.greater_than("speed", 5.0))
            .add_transition("walk", "idle", TransitionCondition.is_false("moving"))
            .add_transition("run", "walk", TransitionCondition.less_than("speed", 5.0))
            .add_any_state_transition("idle", TransitionCondition.trigger("reset"))
            .build()
        )

        assert isinstance(sm, StateMachine)
        assert len(sm.states) == 3

    def test_chain_different_order(self, mock_clip: MockAnimationClip) -> None:
        """Chaining works regardless of call order."""
        sm = (
            StateMachineBuilder("test")
            .set_initial("state_b")  # Set initial before states exist (validated at build)
            .add_state("state_a", mock_clip)
            .add_state("state_b", mock_clip)
            .add_transition("state_a", "state_b")
            .add_transition("state_b", "state_a")
            .build()
        )

        assert sm._initial_state == "state_b"


# ============================================================================
# TEST: add_state() - Source Auto-Detection
# ============================================================================


class TestAddStateAutoDetection:
    """Test add_state() auto-detection of source type."""

    def test_auto_detect_clip_via_sample_and_duration(self, mock_clip: MockAnimationClip) -> None:
        """Auto-detects clips via sample() and duration attributes."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)

        state = builder._states["idle"]
        assert state.clip is mock_clip
        assert state.graph is None
        assert state.animation_node is None

    def test_auto_detect_animation_graph(self, mock_graph: AnimationGraph) -> None:
        """Auto-detects AnimationGraph instances."""
        builder = StateMachineBuilder("test")
        builder.add_state("walk", mock_graph)

        state = builder._states["walk"]
        assert state.graph is mock_graph
        assert state.clip is None
        assert state.animation_node is None

    def test_auto_detect_animation_node(self, mock_node: MockAnimationNode) -> None:
        """Auto-detects AnimationNode instances."""
        builder = StateMachineBuilder("test")
        builder.add_state("run", mock_node)

        state = builder._states["run"]
        assert state.animation_node is mock_node
        assert state.clip is None
        assert state.graph is None

    def test_auto_detect_fallback_to_node(self) -> None:
        """Unknown source types fall back to animation_node."""
        builder = StateMachineBuilder("test")
        generic_source = object()
        builder.add_state("generic", generic_source)

        state = builder._states["generic"]
        assert state.animation_node is generic_source
        assert state.clip is None
        assert state.graph is None

    def test_explicit_clip_kwarg_overrides_detection(
        self,
        mock_clip: MockAnimationClip,
        mock_graph: AnimationGraph
    ) -> None:
        """Explicit clip= kwarg takes precedence over auto-detection."""
        builder = StateMachineBuilder("test")
        builder.add_state("state", mock_graph, clip=mock_clip)

        state = builder._states["state"]
        # Both should be set since explicit kwargs don't override each other
        assert state.clip is mock_clip

    def test_explicit_graph_kwarg(self, mock_graph: AnimationGraph) -> None:
        """Explicit graph= kwarg sets the graph."""
        builder = StateMachineBuilder("test")
        builder.add_state("state", graph=mock_graph)

        state = builder._states["state"]
        assert state.graph is mock_graph

    def test_explicit_animation_node_kwarg(self, mock_node: MockAnimationNode) -> None:
        """Explicit animation_node= kwarg sets the node."""
        builder = StateMachineBuilder("test")
        builder.add_state("state", animation_node=mock_node)

        state = builder._states["state"]
        assert state.animation_node is mock_node

    def test_source_none_with_explicit_kwargs(self, mock_clip: MockAnimationClip) -> None:
        """Source can be None when using explicit kwargs."""
        builder = StateMachineBuilder("test")
        builder.add_state("state", None, clip=mock_clip)

        state = builder._states["state"]
        assert state.clip is mock_clip


# ============================================================================
# TEST: add_state() - Options
# ============================================================================


class TestAddStateOptions:
    """Test add_state() motion_mode, speed, on_enter, on_exit options."""

    def test_motion_mode_loop_default(self, mock_clip: MockAnimationClip) -> None:
        """Default motion_mode is LOOP."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)

        state = builder._states["idle"]
        assert state.motion_mode == MotionMode.LOOP

    def test_motion_mode_once(self, mock_clip: MockAnimationClip) -> None:
        """Can set motion_mode to ONCE."""
        builder = StateMachineBuilder("test")
        builder.add_state("hit", mock_clip, motion_mode=MotionMode.ONCE)

        state = builder._states["hit"]
        assert state.motion_mode == MotionMode.ONCE

    def test_motion_mode_ping_pong(self, mock_clip: MockAnimationClip) -> None:
        """Can set motion_mode to PING_PONG."""
        builder = StateMachineBuilder("test")
        builder.add_state("breathe", mock_clip, motion_mode=MotionMode.PING_PONG)

        state = builder._states["breathe"]
        assert state.motion_mode == MotionMode.PING_PONG

    def test_speed_default_is_one(self, mock_clip: MockAnimationClip) -> None:
        """Default speed is 1.0."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)

        state = builder._states["idle"]
        assert state.speed == 1.0

    def test_speed_custom_value(self, mock_clip: MockAnimationClip) -> None:
        """Can set custom speed."""
        builder = StateMachineBuilder("test")
        builder.add_state("run", mock_clip, speed=1.5)

        state = builder._states["run"]
        assert state.speed == 1.5

    def test_speed_zero(self, mock_clip: MockAnimationClip) -> None:
        """Speed can be zero (frozen animation)."""
        builder = StateMachineBuilder("test")
        builder.add_state("paused", mock_clip, speed=0.0)

        state = builder._states["paused"]
        assert state.speed == 0.0

    def test_speed_negative(self, mock_clip: MockAnimationClip) -> None:
        """Speed can be negative (reverse playback)."""
        builder = StateMachineBuilder("test")
        builder.add_state("rewind", mock_clip, speed=-1.0)

        state = builder._states["rewind"]
        assert state.speed == -1.0

    def test_on_enter_callback(self, mock_clip: MockAnimationClip) -> None:
        """Can set on_enter callback."""
        callback = Mock()
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip, on_enter=callback)

        state = builder._states["idle"]
        assert state.on_enter is callback

    def test_on_exit_callback(self, mock_clip: MockAnimationClip) -> None:
        """Can set on_exit callback."""
        callback = Mock()
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip, on_exit=callback)

        state = builder._states["idle"]
        assert state.on_exit is callback

    def test_can_interrupt_default_true(self, mock_clip: MockAnimationClip) -> None:
        """Default can_interrupt is True."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)

        state = builder._states["idle"]
        assert state.can_interrupt is True

    def test_can_interrupt_false(self, mock_clip: MockAnimationClip) -> None:
        """Can set can_interrupt to False."""
        builder = StateMachineBuilder("test")
        builder.add_state("death", mock_clip, can_interrupt=False)

        state = builder._states["death"]
        assert state.can_interrupt is False

    def test_legacy_loop_parameter_true(self, mock_clip: MockAnimationClip) -> None:
        """Legacy loop=True sets motion_mode to LOOP."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip, loop=True)

        state = builder._states["idle"]
        assert state.motion_mode == MotionMode.LOOP

    def test_legacy_loop_parameter_false(self, mock_clip: MockAnimationClip) -> None:
        """Legacy loop=False sets motion_mode to ONCE."""
        builder = StateMachineBuilder("test")
        builder.add_state("hit", mock_clip, loop=False)

        state = builder._states["hit"]
        assert state.motion_mode == MotionMode.ONCE


# ============================================================================
# TEST: add_state() - Validation
# ============================================================================


class TestAddStateValidation:
    """Test add_state() validates non-empty name and rejects duplicates."""

    def test_rejects_empty_name(self, mock_clip: MockAnimationClip) -> None:
        """Rejects empty string name."""
        builder = StateMachineBuilder("test")

        with pytest.raises(StateMachineBuilderError) as exc_info:
            builder.add_state("", mock_clip)

        assert "cannot be empty" in str(exc_info.value)
        assert exc_info.value.details["provided_name"] == "''"

    def test_rejects_whitespace_only_name(self, mock_clip: MockAnimationClip) -> None:
        """Rejects whitespace-only name."""
        builder = StateMachineBuilder("test")

        with pytest.raises(StateMachineBuilderError) as exc_info:
            builder.add_state("   ", mock_clip)

        assert "cannot be empty" in str(exc_info.value)

    def test_rejects_duplicate_state_name(self, mock_clip: MockAnimationClip) -> None:
        """Rejects duplicate state names."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)

        with pytest.raises(StateMachineBuilderError) as exc_info:
            builder.add_state("idle", mock_clip)

        assert "already exists" in str(exc_info.value)
        assert "idle" in str(exc_info.value)
        assert "existing_states" in exc_info.value.details

    def test_error_details_include_existing_states(self, mock_clip: MockAnimationClip) -> None:
        """Duplicate error includes list of existing states."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)

        with pytest.raises(StateMachineBuilderError) as exc_info:
            builder.add_state("idle", mock_clip)

        assert "idle" in exc_info.value.details["existing_states"]
        assert "walk" in exc_info.value.details["existing_states"]


# ============================================================================
# TEST: add_transition() - All StateTransition Fields
# ============================================================================


class TestAddTransitionFields:
    """Test add_transition() supports all StateTransition fields."""

    def test_source_and_target(self, mock_clip: MockAnimationClip) -> None:
        """Sets source and target correctly."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk")

        transition = builder._transitions[0]
        assert transition.source == "idle"
        assert transition.target == "walk"

    def test_single_condition(self, mock_clip: MockAnimationClip) -> None:
        """Single condition parameter works."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        condition = TransitionCondition.is_true("moving")
        builder.add_transition("idle", "walk", condition)

        transition = builder._transitions[0]
        assert len(transition.conditions) == 1
        assert transition.conditions[0] is condition

    def test_multiple_conditions_via_list(self, mock_clip: MockAnimationClip) -> None:
        """Multiple conditions via conditions= kwarg."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("run", mock_clip)
        cond1 = TransitionCondition.is_true("moving")
        cond2 = TransitionCondition.greater_than("speed", 5.0)
        builder.add_transition("idle", "run", conditions=[cond1, cond2])

        transition = builder._transitions[0]
        assert len(transition.conditions) == 2

    def test_single_and_list_conditions_combined(self, mock_clip: MockAnimationClip) -> None:
        """Single condition + conditions list are combined."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("run", mock_clip)
        single = TransitionCondition.is_true("moving")
        list_conds = [TransitionCondition.greater_than("speed", 5.0)]
        builder.add_transition("idle", "run", single, conditions=list_conds)

        transition = builder._transitions[0]
        assert len(transition.conditions) == 2

    def test_duration_default(self, mock_clip: MockAnimationClip) -> None:
        """Default duration is 0.25."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk")

        transition = builder._transitions[0]
        assert transition.duration == 0.25

    def test_duration_custom(self, mock_clip: MockAnimationClip) -> None:
        """Custom duration value."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk", duration=0.5)

        transition = builder._transitions[0]
        assert transition.duration == 0.5

    def test_duration_mode_fixed_default(self, mock_clip: MockAnimationClip) -> None:
        """Default duration_mode is 'fixed'."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk")

        transition = builder._transitions[0]
        assert transition.duration_mode == "fixed"

    def test_duration_mode_percentage(self, mock_clip: MockAnimationClip) -> None:
        """Can set duration_mode to 'percentage'."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk", duration=0.3, duration_mode="percentage")

        transition = builder._transitions[0]
        assert transition.duration_mode == "percentage"
        assert transition.duration == 0.3

    def test_blend_curve_default(self, mock_clip: MockAnimationClip) -> None:
        """Default blend_curve is SMOOTH_STEP."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk")

        transition = builder._transitions[0]
        assert transition.blend_curve == BlendCurve.SMOOTH_STEP

    def test_blend_curve_custom(self, mock_clip: MockAnimationClip) -> None:
        """Can set custom blend_curve."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk", blend_curve=BlendCurve.EASE_IN_OUT)

        transition = builder._transitions[0]
        assert transition.blend_curve == BlendCurve.EASE_IN_OUT

    def test_legacy_curve_parameter(self, mock_clip: MockAnimationClip) -> None:
        """Legacy curve= parameter maps to blend_curve."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk", curve=BlendCurve.LINEAR)

        transition = builder._transitions[0]
        assert transition.blend_curve == BlendCurve.LINEAR

    def test_sync_mode_default(self, mock_clip: MockAnimationClip) -> None:
        """Default sync_mode is NONE."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk")

        transition = builder._transitions[0]
        assert transition.sync_mode == TransitionSyncMode.NONE

    def test_sync_mode_normalized(self, mock_clip: MockAnimationClip) -> None:
        """Can set sync_mode to NORMALIZED."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk", sync_mode=TransitionSyncMode.NORMALIZED)

        transition = builder._transitions[0]
        assert transition.sync_mode == TransitionSyncMode.NORMALIZED

    def test_exit_time_default_none(self, mock_clip: MockAnimationClip) -> None:
        """Default exit_time is None."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk")

        transition = builder._transitions[0]
        assert transition.exit_time is None

    def test_exit_time_custom(self, mock_clip: MockAnimationClip) -> None:
        """Can set exit_time."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk", exit_time=0.8)

        transition = builder._transitions[0]
        assert transition.exit_time == 0.8

    def test_offset_default_zero(self, mock_clip: MockAnimationClip) -> None:
        """Default offset is 0.0."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk")

        transition = builder._transitions[0]
        assert transition.offset == 0.0

    def test_offset_custom(self, mock_clip: MockAnimationClip) -> None:
        """Can set custom offset."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk", offset=0.5)

        transition = builder._transitions[0]
        assert transition.offset == 0.5

    def test_priority_default_zero(self, mock_clip: MockAnimationClip) -> None:
        """Default priority is 0."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk")

        transition = builder._transitions[0]
        assert transition.priority == 0

    def test_priority_custom(self, mock_clip: MockAnimationClip) -> None:
        """Can set custom priority."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk", priority=10)

        transition = builder._transitions[0]
        assert transition.priority == 10

    def test_interrupt_mode_default(self, mock_clip: MockAnimationClip) -> None:
        """Default interrupt_mode is HIGHER_PRIORITY."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk")

        transition = builder._transitions[0]
        assert transition.interrupt_mode == InterruptMode.HIGHER_PRIORITY

    def test_interrupt_mode_none(self, mock_clip: MockAnimationClip) -> None:
        """Can set interrupt_mode to NONE."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk", interrupt_mode=InterruptMode.NONE)

        transition = builder._transitions[0]
        assert transition.interrupt_mode == InterruptMode.NONE

    def test_interrupt_mode_any(self, mock_clip: MockAnimationClip) -> None:
        """Can set interrupt_mode to ANY."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk", interrupt_mode=InterruptMode.ANY)

        transition = builder._transitions[0]
        assert transition.interrupt_mode == InterruptMode.ANY


# ============================================================================
# TEST: add_any_state_transition()
# ============================================================================


class TestAddAnyStateTransition:
    """Test add_any_state_transition() uses source='*' internally and default high priority."""

    def test_source_is_wildcard(self, mock_clip: MockAnimationClip) -> None:
        """Any-state transitions have source='*'."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("hurt", mock_clip)
        builder.add_any_state_transition("hurt")

        transition = builder._transitions[0]
        assert transition.source == "*"
        assert transition.target == "hurt"

    def test_default_high_priority(self, mock_clip: MockAnimationClip) -> None:
        """Any-state transitions get high default priority from config."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("hurt", mock_clip)
        builder.add_any_state_transition("hurt")

        transition = builder._transitions[0]
        # Priority comes from config.transition.ANY_STATE_PRIORITY
        # Should be higher than default 0
        assert transition.priority >= 0

    def test_custom_priority_overrides_default(self, mock_clip: MockAnimationClip) -> None:
        """Custom priority overrides config default."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("hurt", mock_clip)
        builder.add_any_state_transition("hurt", priority=999)

        transition = builder._transitions[0]
        assert transition.priority == 999

    def test_with_single_condition(self, mock_clip: MockAnimationClip) -> None:
        """Can add single condition."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("hurt", mock_clip)
        condition = TransitionCondition.trigger("take_damage")
        builder.add_any_state_transition("hurt", condition)

        transition = builder._transitions[0]
        assert len(transition.conditions) == 1
        assert transition.conditions[0] is condition

    def test_with_conditions_list(self, mock_clip: MockAnimationClip) -> None:
        """Can add conditions list."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("hurt", mock_clip)
        conds = [
            TransitionCondition.trigger("take_damage"),
            TransitionCondition.greater_than("health", 0),
        ]
        builder.add_any_state_transition("hurt", conditions=conds)

        transition = builder._transitions[0]
        assert len(transition.conditions) == 2

    def test_duration_default_shorter(self, mock_clip: MockAnimationClip) -> None:
        """Any-state transitions have default duration of 0.2."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("hurt", mock_clip)
        builder.add_any_state_transition("hurt")

        transition = builder._transitions[0]
        assert transition.duration == 0.2

    def test_custom_duration(self, mock_clip: MockAnimationClip) -> None:
        """Can set custom duration."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("hurt", mock_clip)
        builder.add_any_state_transition("hurt", duration=0.1)

        transition = builder._transitions[0]
        assert transition.duration == 0.1

    def test_blend_curve_default(self, mock_clip: MockAnimationClip) -> None:
        """Default blend_curve is SMOOTH_STEP."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("hurt", mock_clip)
        builder.add_any_state_transition("hurt")

        transition = builder._transitions[0]
        assert transition.blend_curve == BlendCurve.SMOOTH_STEP

    def test_custom_blend_curve(self, mock_clip: MockAnimationClip) -> None:
        """Can set custom blend_curve."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("hurt", mock_clip)
        builder.add_any_state_transition("hurt", blend_curve=BlendCurve.EASE_IN)

        transition = builder._transitions[0]
        assert transition.blend_curve == BlendCurve.EASE_IN

    def test_legacy_curve_parameter(self, mock_clip: MockAnimationClip) -> None:
        """Legacy curve= parameter works."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("hurt", mock_clip)
        builder.add_any_state_transition("hurt", curve=BlendCurve.LINEAR)

        transition = builder._transitions[0]
        assert transition.blend_curve == BlendCurve.LINEAR

    def test_returns_self(self, mock_clip: MockAnimationClip) -> None:
        """Returns self for chaining."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("hurt", mock_clip)
        result = builder.add_any_state_transition("hurt")

        assert result is builder


# ============================================================================
# TEST: set_initial()
# ============================================================================


class TestSetInitial:
    """Test set_initial() sets initial/default state."""

    def test_sets_initial_state(self, mock_clip: MockAnimationClip) -> None:
        """Sets the initial state name."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.set_initial("walk")

        assert builder._initial_state == "walk"

    def test_first_state_is_auto_initial(self, mock_clip: MockAnimationClip) -> None:
        """First added state becomes initial automatically."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)

        assert builder._initial_state == "idle"

    def test_set_initial_overrides_auto(self, mock_clip: MockAnimationClip) -> None:
        """set_initial() overrides auto-initial."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.set_initial("walk")

        assert builder._initial_state == "walk"

    def test_can_set_initial_before_states(self) -> None:
        """Can set initial state before adding states."""
        builder = StateMachineBuilder("test")
        builder.set_initial("future_state")

        assert builder._initial_state == "future_state"

    def test_validation_happens_at_build(self, mock_clip: MockAnimationClip) -> None:
        """Invalid initial state is caught at build(), not set_initial()."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.set_initial("nonexistent")  # No error here

        with pytest.raises(StateMachineBuilderError) as exc_info:
            builder.build()

        assert "nonexistent" in str(exc_info.value)
        assert "does not exist" in str(exc_info.value)


# ============================================================================
# TEST: build() - Validation
# ============================================================================


class TestBuildValidation:
    """Test build() validation with clear error messages."""

    def test_requires_at_least_one_state(self) -> None:
        """Validation fails if no states added."""
        builder = StateMachineBuilder("test")

        with pytest.raises(StateMachineBuilderError) as exc_info:
            builder.build()

        assert "at least one state" in str(exc_info.value)

    def test_initial_state_must_exist(self, mock_clip: MockAnimationClip) -> None:
        """Validation fails if initial state doesn't exist."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.set_initial("nonexistent")

        with pytest.raises(StateMachineBuilderError) as exc_info:
            builder.build()

        assert "nonexistent" in str(exc_info.value)
        assert "does not exist" in str(exc_info.value)
        assert "idle" in str(exc_info.value)  # Shows available states

    def test_transition_source_must_exist(self, mock_clip: MockAnimationClip) -> None:
        """Validation fails if transition source state doesn't exist."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("nonexistent", "walk")

        with pytest.raises(StateMachineBuilderError) as exc_info:
            builder.build()

        assert "source state 'nonexistent' does not exist" in str(exc_info.value)

    def test_transition_target_must_exist(self, mock_clip: MockAnimationClip) -> None:
        """Validation fails if transition target state doesn't exist."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_transition("idle", "nonexistent")

        with pytest.raises(StateMachineBuilderError) as exc_info:
            builder.build()

        assert "target state 'nonexistent' does not exist" in str(exc_info.value)

    def test_any_state_source_is_valid(self, mock_clip: MockAnimationClip) -> None:
        """Any-state transitions (source='*') are valid."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("hurt", mock_clip)
        builder.add_any_state_transition("hurt")

        sm = builder.build()  # Should not raise
        assert sm is not None

    def test_error_details_include_counts(self, mock_clip: MockAnimationClip) -> None:
        """Error details include state and transition counts."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk")
        builder.set_initial("nonexistent")

        with pytest.raises(StateMachineBuilderError) as exc_info:
            builder.build()

        assert exc_info.value.details["state_count"] == 2
        assert exc_info.value.details["transition_count"] == 1
        assert exc_info.value.details["initial_state"] == "nonexistent"

    def test_multiple_errors_reported(self, mock_clip: MockAnimationClip) -> None:
        """All validation errors are reported together."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_transition("nonexistent1", "idle")  # Bad source
        builder.add_transition("idle", "nonexistent2")  # Bad target
        builder.set_initial("nonexistent3")  # Bad initial

        with pytest.raises(StateMachineBuilderError) as exc_info:
            builder.build()

        errors = exc_info.value.details.get("errors", [])
        assert len(errors) == 3
        assert any("nonexistent1" in e for e in errors)
        assert any("nonexistent2" in e for e in errors)
        assert any("nonexistent3" in e for e in errors)

    def test_valid_build_returns_state_machine(self, mock_clip: MockAnimationClip) -> None:
        """Valid configuration builds a StateMachine."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk")
        builder.add_transition("walk", "idle")
        builder.set_initial("idle")

        sm = builder.build()

        assert isinstance(sm, StateMachine)
        assert sm.node_id == "test"
        assert len(sm.states) == 2
        assert sm._initial_state == "idle"


# ============================================================================
# TEST: build() - State Machine Configuration
# ============================================================================


class TestBuildStateMachineConfiguration:
    """Test that build() properly configures the StateMachine."""

    def test_states_transferred_to_sm(self, mock_clip: MockAnimationClip) -> None:
        """All states are transferred to the state machine."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_state("run", mock_clip)

        sm = builder.build()

        assert "idle" in sm.states
        assert "walk" in sm.states
        assert "run" in sm.states
        assert len(sm.states) == 3

    def test_transitions_transferred_to_sm(self, mock_clip: MockAnimationClip) -> None:
        """All transitions are transferred to the state machine."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk", priority=1)
        builder.add_transition("walk", "idle", priority=2)

        sm = builder.build()

        # Check transitions (sorted by priority descending)
        assert len(sm.transitions) == 2

    def test_any_state_transitions_transferred(self, mock_clip: MockAnimationClip) -> None:
        """Any-state transitions are transferred to _any_state_transitions."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("hurt", mock_clip)
        builder.add_state("death", mock_clip)
        builder.add_any_state_transition("hurt")
        builder.add_any_state_transition("death")

        sm = builder.build()

        assert len(sm._any_state_transitions) == 2
        assert all(t.source == "*" for t in sm._any_state_transitions)

    def test_initial_state_set(self, mock_clip: MockAnimationClip) -> None:
        """Initial state is set on the state machine."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.set_initial("walk")

        sm = builder.build()

        assert sm._initial_state == "walk"

    def test_state_properties_preserved(
        self,
        mock_clip: MockAnimationClip,
        mock_node: MockAnimationNode
    ) -> None:
        """State properties are preserved after build."""
        callback = Mock()
        builder = StateMachineBuilder("test")
        builder.add_state(
            "custom",
            mock_clip,
            motion_mode=MotionMode.PING_PONG,
            speed=1.5,
            on_enter=callback,
            can_interrupt=False,
        )

        sm = builder.build()
        state = sm.states["custom"]

        assert state.clip is mock_clip
        assert state.motion_mode == MotionMode.PING_PONG
        assert state.speed == 1.5
        assert state.on_enter is callback
        assert state.can_interrupt is False


# ============================================================================
# TEST: StateMachineBuilderError
# ============================================================================


class TestStateMachineBuilderError:
    """Test StateMachineBuilderError exception class."""

    def test_message_attribute(self) -> None:
        """Error has message attribute."""
        error = StateMachineBuilderError("test message")

        assert error.message == "test message"
        assert str(error) == "test message"

    def test_details_attribute_default(self) -> None:
        """Default details is empty dict."""
        error = StateMachineBuilderError("test")

        assert error.details == {}

    def test_details_attribute_custom(self) -> None:
        """Custom details are stored."""
        details = {"key": "value", "count": 42}
        error = StateMachineBuilderError("test", details)

        assert error.details == details

    def test_inherits_from_value_error(self) -> None:
        """StateMachineBuilderError is a ValueError."""
        error = StateMachineBuilderError("test")

        assert isinstance(error, ValueError)


# ============================================================================
# TEST: Builder Repr
# ============================================================================


class TestBuilderRepr:
    """Test StateMachineBuilder string representation."""

    def test_repr_empty(self) -> None:
        """Repr for empty builder."""
        builder = StateMachineBuilder("test_sm")

        repr_str = repr(builder)
        assert "StateMachineBuilder" in repr_str
        assert "test_sm" in repr_str
        assert "states=0" in repr_str
        assert "transitions=0" in repr_str

    def test_repr_with_states(self, mock_clip: MockAnimationClip) -> None:
        """Repr shows state and transition counts."""
        builder = StateMachineBuilder("locomotion")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk")

        repr_str = repr(builder)
        assert "states=2" in repr_str
        assert "transitions=1" in repr_str
        assert "initial='idle'" in repr_str


# ============================================================================
# TEST: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and unusual usage patterns."""

    def test_single_state_no_transitions(self, mock_clip: MockAnimationClip) -> None:
        """Single state with no transitions is valid."""
        builder = StateMachineBuilder("test")
        builder.add_state("only_state", mock_clip)

        sm = builder.build()

        assert len(sm.states) == 1
        assert len(sm.transitions) == 0

    def test_self_transition(self, mock_clip: MockAnimationClip) -> None:
        """Self-transitions are allowed."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_transition("idle", "idle")  # Self-transition

        sm = builder.build()  # Should not raise

        assert len(sm.transitions) == 1
        assert sm.transitions[0].source == "idle"
        assert sm.transitions[0].target == "idle"

    def test_multiple_transitions_same_states(self, mock_clip: MockAnimationClip) -> None:
        """Multiple transitions between same states are allowed."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk", TransitionCondition.is_true("moving"), priority=1)
        builder.add_transition("idle", "walk", TransitionCondition.trigger("force_walk"), priority=2)

        sm = builder.build()

        assert len(sm.transitions) == 2

    def test_state_name_with_special_characters(self, mock_clip: MockAnimationClip) -> None:
        """State names with special characters are allowed."""
        builder = StateMachineBuilder("test")
        builder.add_state("walk_cycle_01", mock_clip)
        builder.add_state("idle-standing", mock_clip)
        builder.add_state("combat.attack", mock_clip)

        sm = builder.build()

        assert "walk_cycle_01" in sm.states
        assert "idle-standing" in sm.states
        assert "combat.attack" in sm.states

    def test_unicode_state_names(self, mock_clip: MockAnimationClip) -> None:
        """Unicode state names are allowed."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)

        sm = builder.build()

        assert "idle" in sm.states
        assert "walk" in sm.states

    def test_none_source_creates_empty_state(self) -> None:
        """State with no source is allowed (empty animation)."""
        builder = StateMachineBuilder("test")
        builder.add_state("placeholder", None)

        sm = builder.build()
        state = sm.states["placeholder"]

        assert state.clip is None
        assert state.graph is None
        assert state.animation_node is None

    def test_no_conditions_transition(self, mock_clip: MockAnimationClip) -> None:
        """Transition with no conditions (always valid) is allowed."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.add_transition("idle", "walk")  # No conditions

        sm = builder.build()

        assert len(sm.transitions[0].conditions) == 0

    def test_builder_reuse_creates_independent_machines(
        self,
        mock_clip: MockAnimationClip
    ) -> None:
        """Build can be called multiple times for independent machines."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)

        sm1 = builder.build()
        sm2 = builder.build()

        assert sm1 is not sm2
        assert sm1.states is not sm2.states


# ============================================================================
# TEST: Integration with StateMachine
# ============================================================================


class TestIntegrationWithStateMachine:
    """Test that built state machines work correctly."""

    def test_built_sm_can_start(self, mock_clip: MockAnimationClip) -> None:
        """Built state machine can be started."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.set_initial("idle")

        sm = builder.build()
        context = MagicMock(spec=GraphContext)
        context.parameters = {}
        context.dt = 0.016
        context.skeleton = None
        context.bone_masks = {}
        context.normalized_time = 0.0
        context.sync_group = None
        context.layer_weight = 1.0

        sm.start(context)

        assert sm.current_state is not None
        assert sm.current_state.name == "idle"

    def test_built_sm_transitions_work(self, mock_clip: MockAnimationClip) -> None:
        """Built state machine transitions work correctly."""
        builder = StateMachineBuilder("test")
        builder.add_state("idle", mock_clip)
        builder.add_state("walk", mock_clip)
        builder.set_initial("idle")
        builder.add_transition("idle", "walk", TransitionCondition.is_true("moving"))

        sm = builder.build()

        # Create mock context
        context = MagicMock(spec=GraphContext)
        params = {"moving": True}
        context.get_parameter = lambda p: params.get(p)
        context.parameters = params
        context.dt = 0.016
        context.skeleton = None
        context.bone_masks = {}
        context.normalized_time = 0.0
        context.sync_group = None
        context.layer_weight = 1.0

        sm.start(context)
        assert sm.current_state_name == "idle"

        # Update should trigger transition
        sm.update(0.016, context)

        # Either transitioning or already transitioned
        assert sm.is_transitioning or sm.current_state_name == "walk"

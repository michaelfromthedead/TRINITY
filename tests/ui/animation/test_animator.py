"""
Comprehensive tests for the Animator state machine.

Tests cover:
- Animator initialization and configuration
- Animation state management
- Transition management
- Layer management
- Parameter management
- Playback control
- Callbacks
- AnimatorManager
- Edge cases
"""

from __future__ import annotations

import pytest
from typing import List, Any, Optional
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from engine.ui.animation.animator import (
    Animator,
    AnimatorState,
    AnimationState,
    AnimationTransition,
    AnimationLayer,
    LayerBlendMode,
    TransitionCondition,
    AnimatorManager,
    AnimatorCallback,
    StateCallback,
    TransitionCallback,
)
from engine.ui.animation.tween import Tween, TweenState


# =============================================================================
# TEST FIXTURES
# =============================================================================


@dataclass
class MockWidget:
    """Mock widget for testing."""
    x: float = 0.0
    y: float = 0.0
    opacity: float = 1.0
    scale: float = 1.0


@pytest.fixture
def widget() -> MockWidget:
    """Create a fresh mock widget for each test."""
    return MockWidget()


@pytest.fixture
def animator(widget: MockWidget) -> Animator:
    """Create an animator with basic states."""
    anim = Animator(widget, initial_state="idle")
    anim.add_state("idle")
    anim.add_state("hover")
    anim.add_state("pressed")
    return anim


# =============================================================================
# ANIMATION STATE TESTS
# =============================================================================


class TestAnimationState:
    """Tests for AnimationState dataclass."""

    def test_init_minimal(self) -> None:
        """Should initialize with just name."""
        state = AnimationState(name="idle")
        assert state.name == "idle"
        assert state.animation is None
        assert state.loop is False
        assert state.speed == 1.0

    def test_init_with_animation(self) -> None:
        """Should accept animation object."""
        anim = Mock()
        state = AnimationState(name="hover", animation=anim)
        assert state.animation is anim

    def test_init_with_loop(self) -> None:
        """Should accept loop flag."""
        state = AnimationState(name="idle", loop=True)
        assert state.loop is True

    def test_init_with_speed(self) -> None:
        """Should accept speed multiplier."""
        state = AnimationState(name="fast", speed=2.0)
        assert state.speed == 2.0

    def test_reset_clears_state(self) -> None:
        """reset should clear runtime state."""
        anim = Mock()
        state = AnimationState(name="test", animation=anim)
        state._elapsed = 1.0
        state._is_playing = True

        state.reset()

        assert state._elapsed == 0.0
        assert state._is_playing is False
        anim.stop.assert_called_once()

    def test_start_begins_playback(self) -> None:
        """start should begin animation playback."""
        anim = Mock()
        callback = Mock()
        state = AnimationState(name="test", animation=anim, enter_callback=callback)

        state.start()

        assert state._is_playing is True
        assert state._elapsed == 0.0
        anim.start.assert_called_once()
        callback.assert_called_once()

    def test_stop_ends_playback(self) -> None:
        """stop should end animation playback."""
        anim = Mock()
        callback = Mock()
        state = AnimationState(name="test", animation=anim, exit_callback=callback)
        state._is_playing = True

        state.stop()

        assert state._is_playing is False
        anim.stop.assert_called_once()
        callback.assert_called_once()

    def test_update_advances_time(self) -> None:
        """update should advance elapsed time."""
        anim = Mock()
        anim.update.return_value = True
        state = AnimationState(name="test", animation=anim)
        state.start()

        state.update(0.5)

        assert state._elapsed == pytest.approx(0.5)

    def test_update_applies_speed(self) -> None:
        """update should apply speed multiplier."""
        anim = Mock()
        anim.update.return_value = True
        state = AnimationState(name="test", animation=anim, speed=2.0)
        state.start()

        state.update(0.5)

        assert state._elapsed == pytest.approx(1.0)

    def test_update_returns_is_playing(self) -> None:
        """update should return whether still playing."""
        state = AnimationState(name="test")
        state.start()

        result = state.update(0.1)
        # No animation means it stays playing
        assert result is True

    def test_update_callback_receives_elapsed(self) -> None:
        """update_callback should receive elapsed time."""
        callback = Mock()
        state = AnimationState(name="test", update_callback=callback)
        state.start()

        state.update(0.5)

        callback.assert_called_once_with(pytest.approx(0.5))


# =============================================================================
# ANIMATION TRANSITION TESTS
# =============================================================================


class TestAnimationTransition:
    """Tests for AnimationTransition dataclass."""

    def test_init_minimal(self) -> None:
        """Should initialize with from/to states."""
        trans = AnimationTransition(from_state="idle", to_state="hover")
        assert trans.from_state == "idle"
        assert trans.to_state == "hover"
        assert trans.duration == 0.3

    def test_init_with_duration(self) -> None:
        """Should accept custom duration."""
        trans = AnimationTransition(from_state="idle", to_state="hover", duration=0.5)
        assert trans.duration == 0.5

    def test_init_with_easing(self) -> None:
        """Should accept easing."""
        trans = AnimationTransition(from_state="idle", to_state="hover", easing="ease_out")
        assert trans._easing_func is not None

    def test_can_transition_exact_match(self) -> None:
        """can_transition should succeed on exact state match."""
        trans = AnimationTransition(from_state="idle", to_state="hover")
        assert trans.can_transition("idle") is True
        assert trans.can_transition("pressed") is False

    def test_can_transition_wildcard(self) -> None:
        """can_transition should succeed on wildcard match."""
        trans = AnimationTransition(from_state="*", to_state="idle")
        assert trans.can_transition("hover") is True
        assert trans.can_transition("pressed") is True
        assert trans.can_transition("idle") is True

    def test_can_transition_with_conditions(self) -> None:
        """can_transition should check all conditions."""
        condition = TransitionCondition("test", lambda: True)
        trans = AnimationTransition(
            from_state="idle",
            to_state="hover",
            conditions=[condition]
        )
        assert trans.can_transition("idle") is True

        condition.check = lambda: False
        assert trans.can_transition("idle") is False

    def test_get_progress_linear(self) -> None:
        """get_progress should interpolate based on duration."""
        trans = AnimationTransition(from_state="idle", to_state="hover", duration=1.0, easing="linear")

        assert trans.get_progress(0.0) == pytest.approx(0.0)
        assert trans.get_progress(0.5) == pytest.approx(0.5)
        assert trans.get_progress(1.0) == pytest.approx(1.0)

    def test_get_progress_zero_duration(self) -> None:
        """get_progress should return 1.0 for zero duration."""
        trans = AnimationTransition(from_state="idle", to_state="hover", duration=0.0)
        assert trans.get_progress(0.0) == 1.0


# =============================================================================
# ANIMATION LAYER TESTS
# =============================================================================


class TestAnimationLayer:
    """Tests for AnimationLayer dataclass."""

    def test_init_minimal(self) -> None:
        """Should initialize with just name."""
        layer = AnimationLayer(name="base")
        assert layer.name == "base"
        assert layer.weight == 1.0
        assert layer.blend_mode == LayerBlendMode.OVERRIDE
        assert layer.enabled is True

    def test_init_with_options(self) -> None:
        """Should accept configuration options."""
        layer = AnimationLayer(
            name="overlay",
            weight=0.5,
            blend_mode=LayerBlendMode.ADDITIVE,
            mask={"x", "y"}
        )
        assert layer.weight == 0.5
        assert layer.blend_mode == LayerBlendMode.ADDITIVE
        assert layer.mask == {"x", "y"}

    def test_set_animation(self) -> None:
        """set_animation should store animation reference."""
        layer = AnimationLayer(name="test")
        anim = Mock()

        layer.set_animation(anim)

        assert layer._current_animation is anim

    def test_clear(self) -> None:
        """clear should remove animation and property values."""
        layer = AnimationLayer(name="test")
        layer.set_animation(Mock())
        layer.set_value("x", 100)

        layer.clear()

        assert layer._current_animation is None
        assert layer._property_values == {}

    def test_update_advances_animation(self) -> None:
        """update should update the layer animation."""
        layer = AnimationLayer(name="test")
        anim = Mock()
        layer.set_animation(anim)

        layer.update(0.016)

        anim.update.assert_called_once_with(0.016)

    def test_update_skipped_when_disabled(self) -> None:
        """update should skip when disabled."""
        layer = AnimationLayer(name="test", enabled=False)
        anim = Mock()
        layer.set_animation(anim)

        layer.update(0.016)

        anim.update.assert_not_called()

    def test_get_value(self) -> None:
        """get_value should return property value."""
        layer = AnimationLayer(name="test")
        layer.set_value("x", 50)

        assert layer.get_value("x") == 50
        assert layer.get_value("y") is None

    def test_get_value_respects_mask(self) -> None:
        """get_value should respect property mask."""
        layer = AnimationLayer(name="test", mask={"x"})
        layer.set_value("x", 50)
        layer.set_value("y", 100)

        assert layer.get_value("x") == 50
        assert layer.get_value("y") is None

    def test_get_value_when_disabled(self) -> None:
        """get_value should return None when disabled."""
        layer = AnimationLayer(name="test", enabled=False)
        layer.set_value("x", 50)

        assert layer.get_value("x") is None


# =============================================================================
# ANIMATOR INITIALIZATION TESTS
# =============================================================================


class TestAnimatorInitialization:
    """Tests for Animator initialization."""

    def test_init_with_target(self, widget: MockWidget) -> None:
        """Should store target reference."""
        anim = Animator(widget)
        assert anim.target is widget

    def test_init_default_state(self, widget: MockWidget) -> None:
        """Should default to 'idle' state."""
        anim = Animator(widget)
        assert anim.current_state_name == "idle"

    def test_init_custom_state(self, widget: MockWidget) -> None:
        """Should accept custom initial state."""
        anim = Animator(widget, initial_state="custom")
        assert anim.current_state_name == "custom"

    def test_init_state_is_idle(self, widget: MockWidget) -> None:
        """Should start in IDLE animator state."""
        anim = Animator(widget)
        assert anim.state == AnimatorState.IDLE

    def test_has_default_layer(self, widget: MockWidget) -> None:
        """Should have a default layer."""
        anim = Animator(widget)
        layers = anim.layers
        assert len(layers) == 1
        assert layers[0].name == "default"


# =============================================================================
# ANIMATOR STATE MANAGEMENT TESTS
# =============================================================================


class TestAnimatorStateManagement:
    """Tests for animator state management."""

    def test_add_state(self, widget: MockWidget) -> None:
        """add_state should create and return AnimationState."""
        anim = Animator(widget)

        state = anim.add_state("hover")

        assert isinstance(state, AnimationState)
        assert state.name == "hover"
        assert anim.has_state("hover") is True

    def test_add_state_with_options(self, widget: MockWidget) -> None:
        """add_state should accept configuration options."""
        anim = Animator(widget)
        mock_animation = Mock()

        state = anim.add_state("hover", animation=mock_animation, loop=True, speed=2.0)

        assert state.animation is mock_animation
        assert state.loop is True
        assert state.speed == 2.0

    def test_remove_state(self, widget: MockWidget) -> None:
        """remove_state should remove an existing state."""
        anim = Animator(widget)
        anim.add_state("hover")

        result = anim.remove_state("hover")

        assert result is True
        assert anim.has_state("hover") is False

    def test_remove_nonexistent_state(self, widget: MockWidget) -> None:
        """remove_state should return False for missing state."""
        anim = Animator(widget)
        assert anim.remove_state("unknown") is False

    def test_get_state(self, animator: Animator) -> None:
        """get_state should return AnimationState or None."""
        state = animator.get_state("idle")
        assert state is not None
        assert state.name == "idle"

        assert animator.get_state("unknown") is None

    def test_has_state(self, animator: Animator) -> None:
        """has_state should check state existence."""
        assert animator.has_state("idle") is True
        assert animator.has_state("unknown") is False

    def test_current_state(self, animator: Animator) -> None:
        """current_state should return current AnimationState."""
        state = animator.current_state
        assert state is not None
        assert state.name == "idle"


# =============================================================================
# ANIMATOR TRANSITION MANAGEMENT TESTS
# =============================================================================


class TestAnimatorTransitionManagement:
    """Tests for animator transition management."""

    def test_add_transition(self, animator: Animator) -> None:
        """add_transition should create and return AnimationTransition."""
        trans = animator.add_transition("idle", "hover", duration=0.2)

        assert isinstance(trans, AnimationTransition)
        assert trans.from_state == "idle"
        assert trans.to_state == "hover"
        assert trans.duration == 0.2

    def test_add_transition_with_easing(self, animator: Animator) -> None:
        """add_transition should accept easing."""
        trans = animator.add_transition("idle", "hover", easing="ease_out")
        assert trans._easing_func is not None

    def test_remove_transition(self, animator: Animator) -> None:
        """remove_transition should remove existing transition."""
        animator.add_transition("idle", "hover")

        result = animator.remove_transition("idle", "hover")

        assert result is True
        assert animator.get_transition("idle", "hover") is None

    def test_remove_nonexistent_transition(self, animator: Animator) -> None:
        """remove_transition should return False for missing transition."""
        assert animator.remove_transition("idle", "unknown") is False

    def test_get_transition_exact(self, animator: Animator) -> None:
        """get_transition should find exact match."""
        animator.add_transition("idle", "hover")

        trans = animator.get_transition("idle", "hover")

        assert trans is not None
        assert trans.from_state == "idle"

    def test_get_transition_wildcard(self, animator: Animator) -> None:
        """get_transition should find wildcard match."""
        animator.add_transition("*", "idle")

        trans = animator.get_transition("hover", "idle")

        assert trans is not None
        assert trans.from_state == "*"


# =============================================================================
# ANIMATOR LAYER MANAGEMENT TESTS
# =============================================================================


class TestAnimatorLayerManagement:
    """Tests for animator layer management."""

    def test_add_layer(self, animator: Animator) -> None:
        """add_layer should create and return AnimationLayer."""
        layer = animator.add_layer("overlay", weight=0.5)

        assert isinstance(layer, AnimationLayer)
        assert layer.name == "overlay"
        assert layer.weight == 0.5

    def test_remove_layer(self, animator: Animator) -> None:
        """remove_layer should remove a layer."""
        animator.add_layer("overlay")

        result = animator.remove_layer("overlay")

        assert result is True
        assert animator.get_layer("overlay") is None

    def test_remove_default_layer_fails(self, animator: Animator) -> None:
        """Cannot remove the default layer."""
        result = animator.remove_layer("default")
        assert result is False

    def test_get_layer(self, animator: Animator) -> None:
        """get_layer should find layer by name."""
        animator.add_layer("overlay")

        layer = animator.get_layer("overlay")

        assert layer is not None
        assert layer.name == "overlay"

    def test_layers_returns_copy(self, animator: Animator) -> None:
        """layers should return a copy of the layer list."""
        layers1 = animator.layers
        layers2 = animator.layers

        assert layers1 == layers2
        assert layers1 is not layers2


# =============================================================================
# ANIMATOR PARAMETER MANAGEMENT TESTS
# =============================================================================


class TestAnimatorParameterManagement:
    """Tests for animator parameter management."""

    def test_set_parameter(self, animator: Animator) -> None:
        """set_parameter should store parameter value."""
        result = animator.set_parameter("speed", 2.0)

        assert result is animator  # Fluent interface
        assert animator.get_parameter("speed") == 2.0

    def test_get_parameter_default(self, animator: Animator) -> None:
        """get_parameter should return default for missing parameter."""
        assert animator.get_parameter("missing") is None
        assert animator.get_parameter("missing", 100) == 100

    def test_clear_parameters(self, animator: Animator) -> None:
        """clear_parameters should remove all parameters."""
        animator.set_parameter("a", 1)
        animator.set_parameter("b", 2)

        result = animator.clear_parameters()

        assert result is animator
        assert animator.get_parameter("a") is None
        assert animator.get_parameter("b") is None


# =============================================================================
# ANIMATOR CONTROL TESTS
# =============================================================================


class TestAnimatorControl:
    """Tests for animator playback control."""

    def test_play_changes_state(self, animator: Animator) -> None:
        """play should change state to PLAYING."""
        result = animator.play()

        assert result is animator
        assert animator.state == AnimatorState.PLAYING

    def test_play_with_state_name(self, animator: Animator) -> None:
        """play with state_name should switch to that state."""
        animator.play("hover")

        assert animator.current_state_name == "hover"
        assert animator.state == AnimatorState.PLAYING

    def test_pause_changes_state(self, animator: Animator) -> None:
        """pause should change state to PAUSED."""
        animator.play()
        result = animator.pause()

        assert result is animator
        assert animator.state == AnimatorState.PAUSED

    def test_pause_does_nothing_when_idle(self, animator: Animator) -> None:
        """pause should not affect IDLE state."""
        animator.pause()
        assert animator.state == AnimatorState.IDLE

    def test_resume_changes_state(self, animator: Animator) -> None:
        """resume should change from PAUSED to PLAYING."""
        animator.play()
        animator.pause()

        result = animator.resume()

        assert result is animator
        assert animator.state == AnimatorState.PLAYING

    def test_resume_does_nothing_when_not_paused(self, animator: Animator) -> None:
        """resume should only affect PAUSED state."""
        animator.play()
        animator.resume()  # Already playing
        assert animator.state == AnimatorState.PLAYING

    def test_stop_resets_animator(self, animator: Animator) -> None:
        """stop should reset animator to IDLE and initial state."""
        animator.play("hover")

        result = animator.stop()

        assert result is animator
        assert animator.state == AnimatorState.IDLE
        assert animator.current_state_name == "idle"

    def test_transition_to_changes_state(self, animator: Animator) -> None:
        """transition_to should change to target state."""
        result = animator.transition_to("hover")

        assert result is True
        assert animator.current_state_name == "hover"

    def test_transition_to_nonexistent_fails(self, animator: Animator) -> None:
        """transition_to should fail for nonexistent state."""
        result = animator.transition_to("unknown")
        assert result is False

    def test_transition_to_same_state_fails(self, animator: Animator) -> None:
        """transition_to should fail for current state."""
        result = animator.transition_to("idle")
        assert result is False

    def test_transition_to_immediate(self, animator: Animator) -> None:
        """transition_to with immediate=True should skip transition."""
        animator.add_transition("idle", "hover", duration=1.0)

        animator.transition_to("hover", immediate=True)

        assert animator.is_transitioning is False
        assert animator.current_state_name == "hover"

    def test_transition_with_defined_transition(self, animator: Animator) -> None:
        """transition_to should use defined transition."""
        animator.add_transition("idle", "hover", duration=0.5)
        animator.play()

        animator.transition_to("hover")

        assert animator.is_transitioning is True
        assert animator.state == AnimatorState.TRANSITIONING


# =============================================================================
# ANIMATOR UPDATE TESTS
# =============================================================================


class TestAnimatorUpdate:
    """Tests for animator update method."""

    def test_update_does_nothing_when_idle(self, animator: Animator) -> None:
        """update should not advance when IDLE."""
        animator.update(0.5)
        # No errors, state unchanged
        assert animator.state == AnimatorState.IDLE

    def test_update_does_nothing_when_paused(self, animator: Animator) -> None:
        """update should not advance when PAUSED."""
        animator.play()
        animator.pause()

        animator.update(0.5)

        assert animator.state == AnimatorState.PAUSED

    def test_update_advances_transition(self, animator: Animator) -> None:
        """update should advance transition progress."""
        animator.add_transition("idle", "hover", duration=1.0, easing="linear")
        animator.play()
        animator.transition_to("hover")

        animator.update(0.5)

        assert animator.is_transitioning is True

    def test_update_completes_transition(self, animator: Animator) -> None:
        """update should complete transition when duration passed."""
        animator.add_transition("idle", "hover", duration=0.5)
        animator.play()
        animator.transition_to("hover")

        animator.update(0.6)

        assert animator.is_transitioning is False
        assert animator.current_state_name == "hover"


# =============================================================================
# ANIMATOR CALLBACK TESTS
# =============================================================================


class TestAnimatorCallbacks:
    """Tests for animator callbacks."""

    def test_on_state_enter(self, animator: Animator) -> None:
        """on_state_enter callback should fire on state entry."""
        callback = Mock()
        animator.on_state_enter(callback)
        animator.play()

        animator.transition_to("hover", immediate=True)

        callback.assert_called_with("hover")

    def test_on_state_exit(self, animator: Animator) -> None:
        """on_state_exit callback should fire on state exit."""
        callback = Mock()
        animator.on_state_exit(callback)
        animator.play()

        animator.transition_to("hover", immediate=True)

        callback.assert_called_with("idle")

    def test_on_transition_start(self, animator: Animator) -> None:
        """on_transition_start callback should fire when transition begins."""
        callback = Mock()
        animator.add_transition("idle", "hover", duration=0.5)
        animator.on_transition_start(callback)
        animator.play()

        animator.transition_to("hover")

        callback.assert_called_once_with("idle", "hover")

    def test_on_transition_complete(self, animator: Animator) -> None:
        """on_transition_complete callback should fire when transition ends."""
        callback = Mock()
        animator.add_transition("idle", "hover", duration=0.5)
        animator.on_transition_complete(callback)
        animator.play()
        animator.transition_to("hover")

        animator.update(0.6)

        callback.assert_called_once_with("idle", "hover")

    def test_callback_chaining(self, animator: Animator) -> None:
        """Callback setters should return self for chaining."""
        result = (
            animator
            .on_state_enter(lambda s: None)
            .on_state_exit(lambda s: None)
            .on_transition_start(lambda f, t: None)
            .on_transition_complete(lambda f, t: None)
        )

        assert result is animator


# =============================================================================
# ANIMATOR MANAGER TESTS
# =============================================================================


class TestAnimatorManager:
    """Tests for AnimatorManager class."""

    def test_init_empty(self) -> None:
        """Should initialize with no animators."""
        manager = AnimatorManager()
        assert manager.count == 0

    def test_add_animator(self, animator: Animator) -> None:
        """add should register an animator."""
        manager = AnimatorManager()

        manager.add(animator)

        assert manager.count == 1

    def test_add_duplicate(self, animator: Animator) -> None:
        """add should not duplicate animators."""
        manager = AnimatorManager()
        manager.add(animator)
        manager.add(animator)

        assert manager.count == 1

    def test_remove_animator(self, animator: Animator) -> None:
        """remove should unregister an animator."""
        manager = AnimatorManager()
        manager.add(animator)

        manager.remove(animator)

        assert manager.count == 0

    def test_clear(self, animator: Animator) -> None:
        """clear should remove all animators."""
        manager = AnimatorManager()
        manager.add(animator)

        manager.clear()

        assert manager.count == 0

    def test_update_all(self, widget: MockWidget) -> None:
        """update should update all managed animators."""
        anim1 = Animator(widget)
        anim2 = Animator(widget)
        anim1.add_state("idle")
        anim2.add_state("idle")
        anim1.play()
        anim2.play()

        manager = AnimatorManager()
        manager.add(anim1)
        manager.add(anim2)

        # Should not raise
        manager.update(0.016)

    def test_get_by_target(self, widget: MockWidget) -> None:
        """get_by_target should find animator by widget."""
        animator = Animator(widget)
        manager = AnimatorManager()
        manager.add(animator)

        result = manager.get_by_target(widget)

        assert result is animator

    def test_get_by_target_not_found(self, widget: MockWidget) -> None:
        """get_by_target should return None for unknown target."""
        manager = AnimatorManager()

        result = manager.get_by_target(widget)

        assert result is None

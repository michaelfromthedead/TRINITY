"""
Comprehensive tests for the Tween animation system.

Tests cover:
- Tween initialization and configuration
- Tween state management
- Update and interpolation
- Callbacks (start, update, complete, cancel, repeat)
- Repeat and yoyo modes
- TweenSequence
- TweenGroup
- TweenManager
- Factory functions (tween_to, tween_from, tween_by)
- Edge cases and error handling
"""

from __future__ import annotations

import pytest
from typing import List, Any, Optional
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from engine.ui.animation.tween import (
    Tween,
    TweenState,
    TweenConfig,
    LoopMode,
    TweenSequence,
    TweenGroup,
    TweenManager,
    tween_to,
    tween_from,
    tween_by,
    TweenCallback,
    TweenUpdateCallback,
    TweenValueCallback,
)
from engine.ui.animation.easing import EasingType


# =============================================================================
# TEST FIXTURES
# =============================================================================


@dataclass
class AnimatableObject:
    """Simple object with animatable properties for testing."""
    x: float = 0.0
    y: float = 0.0
    alpha: float = 1.0
    scale: float = 1.0
    rotation: float = 0.0
    color: tuple = (255, 255, 255)


@pytest.fixture
def target() -> AnimatableObject:
    """Create a fresh animatable target for each test."""
    return AnimatableObject()


@pytest.fixture
def tween(target: AnimatableObject) -> Tween:
    """Create a basic tween for testing."""
    return Tween(target, "x", 0.0, 100.0, duration=1.0)


# =============================================================================
# TWEEN INITIALIZATION TESTS
# =============================================================================


class TestTweenInitialization:
    """Tests for Tween initialization."""

    def test_init_stores_target(self, target: AnimatableObject) -> None:
        """Should store the target object."""
        t = Tween(target, "x", 0.0, 100.0)
        assert t.target is target

    def test_init_stores_property_name(self, target: AnimatableObject) -> None:
        """Should store the property name."""
        t = Tween(target, "x", 0.0, 100.0)
        assert t.property_name == "x"

    def test_init_stores_from_value(self, target: AnimatableObject) -> None:
        """Should store the from value."""
        t = Tween(target, "x", 50.0, 100.0)
        assert t.from_value == 50.0

    def test_init_stores_to_value(self, target: AnimatableObject) -> None:
        """Should store the to value."""
        t = Tween(target, "x", 0.0, 100.0)
        assert t.to_value == 100.0

    def test_init_default_duration(self, target: AnimatableObject) -> None:
        """Default duration should be 1.0."""
        t = Tween(target, "x", 0.0, 100.0)
        assert t.duration == 1.0

    def test_init_custom_duration(self, target: AnimatableObject) -> None:
        """Should accept custom duration."""
        t = Tween(target, "x", 0.0, 100.0, duration=0.5)
        assert t.duration == 0.5

    def test_init_minimum_duration(self, target: AnimatableObject) -> None:
        """Duration should be at least 0.001."""
        t = Tween(target, "x", 0.0, 100.0, duration=0.0)
        assert t.duration >= 0.001

    def test_init_state_is_idle(self, target: AnimatableObject) -> None:
        """Initial state should be IDLE."""
        t = Tween(target, "x", 0.0, 100.0)
        assert t.state == TweenState.IDLE

    def test_init_with_auto_start(self, target: AnimatableObject) -> None:
        """Should auto-start when auto_start=True."""
        t = Tween(target, "x", 0.0, 100.0, auto_start=True)
        assert t.state == TweenState.PLAYING

    def test_init_with_easing_string(self, target: AnimatableObject) -> None:
        """Should accept easing as string."""
        t = Tween(target, "x", 0.0, 100.0, easing="quad_in")
        assert t.is_playing is False  # Just verify no error

    def test_init_with_easing_type(self, target: AnimatableObject) -> None:
        """Should accept easing as EasingType."""
        t = Tween(target, "x", 0.0, 100.0, easing=EasingType.CUBIC_OUT)
        assert t.is_playing is False

    def test_init_with_easing_function(self, target: AnimatableObject) -> None:
        """Should accept easing as callable."""
        custom_easing = lambda t: t * t
        t = Tween(target, "x", 0.0, 100.0, easing=custom_easing)
        assert t.is_playing is False


# =============================================================================
# TWEEN STATE TESTS
# =============================================================================


class TestTweenState:
    """Tests for Tween state management."""

    def test_is_playing_when_playing(self, tween: Tween) -> None:
        """is_playing should be True when state is PLAYING."""
        tween.start()
        assert tween.is_playing is True

    def test_is_playing_when_not_playing(self, tween: Tween) -> None:
        """is_playing should be False when not playing."""
        assert tween.is_playing is False

    def test_is_complete_after_finish(self, tween: Tween) -> None:
        """is_complete should be True after finishing."""
        tween.start()
        tween.update(2.0)  # Past duration
        assert tween.is_complete is True

    def test_is_complete_when_not_done(self, tween: Tween) -> None:
        """is_complete should be False when not done."""
        tween.start()
        tween.update(0.5)
        assert tween.is_complete is False


# =============================================================================
# TWEEN PROGRESS TESTS
# =============================================================================


class TestTweenProgress:
    """Tests for Tween progress tracking."""

    def test_progress_starts_at_zero(self, tween: Tween) -> None:
        """Progress should start at 0."""
        tween.start()
        assert tween.progress == pytest.approx(0.0)

    def test_progress_at_midpoint(self, tween: Tween) -> None:
        """Progress should be 0.5 at midpoint."""
        tween.start()
        tween.update(0.5)
        assert tween.progress == pytest.approx(0.5)

    def test_progress_capped_at_one(self, tween: Tween) -> None:
        """Progress should not exceed 1.0."""
        tween.start()
        tween.update(2.0)
        assert tween.progress == 1.0

    def test_elapsed_time_tracks_updates(self, tween: Tween) -> None:
        """elapsed_time should track accumulated delta time."""
        tween.start()
        tween.update(0.3)
        tween.update(0.2)
        assert tween.elapsed_time == pytest.approx(0.5)


# =============================================================================
# TWEEN CONTROL TESTS
# =============================================================================


class TestTweenControl:
    """Tests for Tween control methods."""

    def test_start_changes_state(self, tween: Tween) -> None:
        """start should change state to PLAYING."""
        tween.start()
        assert tween.state == TweenState.PLAYING

    def test_start_resets_progress(self, tween: Tween) -> None:
        """start should reset progress to 0."""
        tween.start()
        tween.update(0.5)
        tween.start()
        assert tween.progress == pytest.approx(0.0)

    def test_start_returns_self(self, tween: Tween) -> None:
        """start should return self for chaining."""
        result = tween.start()
        assert result is tween

    def test_pause_changes_state(self, tween: Tween) -> None:
        """pause should change state to PAUSED."""
        tween.start()
        tween.pause()
        assert tween.state == TweenState.PAUSED

    def test_pause_does_nothing_when_not_playing(self, tween: Tween) -> None:
        """pause should not change non-playing state."""
        tween.pause()
        assert tween.state == TweenState.IDLE

    def test_resume_changes_state(self, tween: Tween) -> None:
        """resume should change state back to PLAYING."""
        tween.start()
        tween.pause()
        tween.resume()
        assert tween.state == TweenState.PLAYING

    def test_resume_does_nothing_when_not_paused(self, tween: Tween) -> None:
        """resume should not change non-paused state."""
        tween.start()
        initial_state = tween.state
        tween.resume()
        assert tween.state == initial_state

    def test_stop_resets_to_idle(self, tween: Tween) -> None:
        """stop should reset state to IDLE."""
        tween.start()
        tween.update(0.5)
        tween.stop()
        assert tween.state == TweenState.IDLE
        assert tween.elapsed_time == 0.0

    def test_cancel_changes_state(self, tween: Tween) -> None:
        """cancel should change state to CANCELLED."""
        tween.start()
        tween.cancel()
        assert tween.state == TweenState.CANCELLED

    def test_complete_jumps_to_end(self, target: AnimatableObject) -> None:
        """complete should immediately finish the tween."""
        t = Tween(target, "x", 0.0, 100.0, duration=1.0)
        t.start()
        t.complete()

        assert t.state == TweenState.COMPLETED
        assert target.x == 100.0

    def test_reverse_swaps_values(self, target: AnimatableObject) -> None:
        """reverse should swap from and to values."""
        t = Tween(target, "x", 0.0, 100.0, duration=1.0)
        t.start()
        t.update(0.3)
        t.reverse()

        assert t.from_value == 100.0
        assert t.to_value == 0.0


# =============================================================================
# TWEEN UPDATE TESTS
# =============================================================================


class TestTweenUpdate:
    """Tests for Tween update and interpolation."""

    def test_update_returns_true_when_active(self, tween: Tween) -> None:
        """update should return True when tween is still active."""
        tween.start()
        result = tween.update(0.5)
        assert result is True

    def test_update_returns_false_when_complete(self, tween: Tween) -> None:
        """update should return False when tween completes."""
        tween.start()
        result = tween.update(2.0)
        assert result is False

    def test_update_interpolates_value(self, target: AnimatableObject) -> None:
        """update should interpolate the property value."""
        t = Tween(target, "x", 0.0, 100.0, duration=1.0, easing="linear")
        t.start()
        t.update(0.5)

        assert target.x == pytest.approx(50.0)

    def test_update_applies_easing(self, target: AnimatableObject) -> None:
        """update should apply easing to interpolation."""
        t = Tween(target, "x", 0.0, 100.0, duration=1.0, easing="quad_in")
        t.start()
        t.update(0.5)

        # quad_in at 0.5 = 0.25, so value should be 25
        assert target.x == pytest.approx(25.0)

    def test_update_does_nothing_when_not_playing(self, target: AnimatableObject) -> None:
        """update should not change value when not playing."""
        t = Tween(target, "x", 0.0, 100.0, duration=1.0)
        target.x = 50.0
        t.update(0.5)

        assert target.x == 50.0

    def test_update_handles_delay(self, target: AnimatableObject) -> None:
        """update should respect delay before animation."""
        t = Tween(target, "x", 0.0, 100.0, duration=1.0, delay=0.5, easing="linear")
        t.start()
        t.update(0.3)  # Still in delay

        # Value should not have changed
        assert target.x == pytest.approx(0.0)

        t.update(0.4)  # Past delay, 0.2 into animation
        assert target.x == pytest.approx(20.0)


# =============================================================================
# TWEEN INTERPOLATION TESTS
# =============================================================================


class TestTweenInterpolation:
    """Tests for different value type interpolation."""

    def test_interpolate_float(self, target: AnimatableObject) -> None:
        """Should interpolate float values."""
        t = Tween(target, "alpha", 1.0, 0.0, duration=1.0, easing="linear")
        t.start()
        t.update(0.5)

        assert target.alpha == pytest.approx(0.5)

    def test_interpolate_int(self) -> None:
        """Should interpolate int values and round."""
        obj = type("Obj", (), {"value": 0})()
        t = Tween(obj, "value", 0, 100, duration=1.0, easing="linear")
        t.start()
        t.update(0.5)

        assert obj.value == 50

    def test_interpolate_tuple(self, target: AnimatableObject) -> None:
        """Should interpolate tuple values (like colors)."""
        # Use floats to get float interpolation results
        t = Tween(target, "color", (255.0, 255.0, 255.0), (0.0, 0.0, 0.0), duration=1.0, easing="linear")
        t.start()
        t.update(0.5)

        assert target.color[0] == pytest.approx(127.5)
        assert target.color[1] == pytest.approx(127.5)
        assert target.color[2] == pytest.approx(127.5)

    def test_interpolate_list(self) -> None:
        """Should interpolate list values."""
        obj = type("Obj", (), {"vec": [0.0, 0.0, 0.0]})()
        t = Tween(obj, "vec", [0.0, 0.0, 0.0], [10.0, 20.0, 30.0], duration=1.0, easing="linear")
        t.start()
        t.update(0.5)

        assert obj.vec[0] == pytest.approx(5.0)
        assert obj.vec[1] == pytest.approx(10.0)
        assert obj.vec[2] == pytest.approx(15.0)

    def test_interpolate_dict(self) -> None:
        """Should interpolate dict values."""
        obj = type("Obj", (), {"props": {"a": 0.0, "b": 0.0}})()
        t = Tween(
            obj, "props",
            {"a": 0.0, "b": 0.0},
            {"a": 100.0, "b": 50.0},
            duration=1.0, easing="linear"
        )
        t.start()
        t.update(0.5)

        assert obj.props["a"] == pytest.approx(50.0)
        assert obj.props["b"] == pytest.approx(25.0)


# =============================================================================
# TWEEN CALLBACK TESTS
# =============================================================================


class TestTweenCallbacks:
    """Tests for Tween callbacks."""

    def test_on_start_callback(self, tween: Tween) -> None:
        """on_start should be called when animation starts after delay."""
        callback = Mock()
        tween.on_start(callback)
        tween.start()
        # on_start fires when delay finishes (no delay = first update)
        tween.update(0.1)

        callback.assert_called_once()

    def test_on_update_callback(self, tween: Tween) -> None:
        """on_update should be called each update with progress."""
        callback = Mock()
        tween.on_update(callback)
        tween.start()
        tween.update(0.5)

        callback.assert_called_once()
        assert callback.call_args[0][0] == pytest.approx(0.5)

    def test_on_value_change_callback(self, tween: Tween) -> None:
        """on_value_change should be called with current value."""
        callback = Mock()
        tween.on_value_change(callback)
        tween.start()
        tween.update(0.5)

        callback.assert_called_once()

    def test_on_complete_callback(self, tween: Tween) -> None:
        """on_complete should be called when tween finishes."""
        callback = Mock()
        tween.on_complete(callback)
        tween.start()
        tween.update(2.0)

        callback.assert_called_once()

    def test_on_cancel_callback(self, tween: Tween) -> None:
        """on_cancel should be called when tween is cancelled."""
        callback = Mock()
        tween.on_cancel(callback)
        tween.start()
        tween.cancel()

        callback.assert_called_once()

    def test_on_repeat_callback(self, target: AnimatableObject) -> None:
        """on_repeat should be called on each repeat."""
        callback = Mock()
        t = Tween(target, "x", 0.0, 100.0, duration=0.5, repeat=2)
        t.on_repeat(callback)
        t.start()
        t.update(0.6)  # First repeat

        callback.assert_called_once()


# =============================================================================
# TWEEN REPEAT TESTS
# =============================================================================


class TestTweenRepeat:
    """Tests for Tween repeat functionality."""

    def test_repeat_once(self, target: AnimatableObject) -> None:
        """Should repeat animation specified number of times."""
        t = Tween(target, "x", 0.0, 100.0, duration=0.5, repeat=1, easing="linear")
        t.start()
        t.update(0.6)  # First cycle complete

        # Should still be playing (in second cycle)
        assert t.is_playing is True

    def test_repeat_completes_after_count(self, target: AnimatableObject) -> None:
        """Should complete after repeat count is exhausted."""
        t = Tween(target, "x", 0.0, 100.0, duration=0.5, repeat=1)
        t.start()
        t.update(0.6)  # First cycle
        t.update(0.6)  # Second cycle

        assert t.is_complete is True

    def test_infinite_repeat(self, target: AnimatableObject) -> None:
        """repeat=-1 should repeat infinitely."""
        t = Tween(target, "x", 0.0, 100.0, duration=0.5, repeat=-1)
        t.start()

        for _ in range(10):
            t.update(0.6)
            assert t.is_playing is True

    def test_yoyo_reverses_direction(self, target: AnimatableObject) -> None:
        """yoyo should reverse animation direction on repeat."""
        t = Tween(target, "x", 0.0, 100.0, duration=0.5, repeat=1, yoyo=True, easing="linear")
        t.start()
        t.update(0.5)  # End of first cycle

        # Should be at end value
        assert target.x == pytest.approx(100.0)

        t.update(0.25)  # Quarter through reverse

        # Should be heading back
        assert target.x < 100.0


# =============================================================================
# FLUENT CONFIGURATION TESTS
# =============================================================================


class TestFluentConfiguration:
    """Tests for fluent configuration methods."""

    def test_set_easing_returns_self(self, tween: Tween) -> None:
        """set_easing should return self for chaining."""
        result = tween.set_easing("quad_out")
        assert result is tween

    def test_set_delay_returns_self(self, tween: Tween) -> None:
        """set_delay should return self for chaining."""
        result = tween.set_delay(0.5)
        assert result is tween

    def test_set_repeat_returns_self(self, tween: Tween) -> None:
        """set_repeat should return self for chaining."""
        result = tween.set_repeat(3, yoyo=True)
        assert result is tween

    def test_on_start_returns_self(self, tween: Tween) -> None:
        """on_start should return self for chaining."""
        result = tween.on_start(lambda: None)
        assert result is tween

    def test_chaining(self, target: AnimatableObject) -> None:
        """Should support full method chaining."""
        t = (
            Tween(target, "x", 0.0, 100.0)
            .set_easing("cubic_out")
            .set_delay(0.2)
            .set_repeat(2, yoyo=True)
            .on_start(lambda: None)
            .on_complete(lambda: None)
        )

        assert t._delay == 0.2
        assert t._repeat == 2
        assert t._yoyo is True


# =============================================================================
# TWEEN SEQUENCE TESTS
# =============================================================================


class TestTweenSequence:
    """Tests for TweenSequence class."""

    def test_sequence_init_empty(self) -> None:
        """Should initialize with empty tween list."""
        seq = TweenSequence()
        assert len(seq.tweens) == 0

    def test_sequence_init_with_tweens(self, target: AnimatableObject) -> None:
        """Should initialize with provided tweens."""
        t1 = Tween(target, "x", 0.0, 50.0, duration=0.5)
        t2 = Tween(target, "x", 50.0, 100.0, duration=0.5)
        seq = TweenSequence([t1, t2])

        assert len(seq.tweens) == 2

    def test_sequence_add(self, target: AnimatableObject) -> None:
        """Should be able to add tweens."""
        seq = TweenSequence()
        t = Tween(target, "x", 0.0, 100.0)

        result = seq.add(t)

        assert len(seq.tweens) == 1
        assert result is seq  # Fluent interface

    def test_sequence_start(self, target: AnimatableObject) -> None:
        """start should begin the first tween."""
        t1 = Tween(target, "x", 0.0, 50.0, duration=0.5)
        t2 = Tween(target, "x", 50.0, 100.0, duration=0.5)
        seq = TweenSequence([t1, t2])

        seq.start()

        assert seq.state == TweenState.PLAYING
        assert t1.is_playing is True
        assert t2.is_playing is False

    def test_sequence_advances_to_next(self, target: AnimatableObject) -> None:
        """Sequence should advance to next tween when current completes."""
        t1 = Tween(target, "x", 0.0, 50.0, duration=0.5)
        t2 = Tween(target, "x", 50.0, 100.0, duration=0.5)
        seq = TweenSequence([t1, t2])

        seq.start()
        seq.update(0.6)  # Complete first tween

        assert t2.is_playing is True

    def test_sequence_completes(self, target: AnimatableObject) -> None:
        """Sequence should complete when all tweens done."""
        t1 = Tween(target, "x", 0.0, 50.0, duration=0.5)
        t2 = Tween(target, "x", 50.0, 100.0, duration=0.5)
        seq = TweenSequence([t1, t2])

        seq.start()
        seq.update(0.6)
        seq.update(0.6)

        assert seq.is_complete is True

    def test_sequence_progress(self, target: AnimatableObject) -> None:
        """Sequence progress should reflect overall completion."""
        t1 = Tween(target, "x", 0.0, 50.0, duration=1.0)
        t2 = Tween(target, "x", 50.0, 100.0, duration=1.0)
        seq = TweenSequence([t1, t2])

        seq.start()
        seq.update(0.5)  # Half through first

        # 0.5 of 1 out of 2 = 0.25
        assert seq.progress == pytest.approx(0.25)

    def test_sequence_pause_resume(self, target: AnimatableObject) -> None:
        """Should be able to pause and resume sequence."""
        t1 = Tween(target, "x", 0.0, 100.0, duration=1.0)
        seq = TweenSequence([t1])

        seq.start()
        seq.pause()
        assert seq.state == TweenState.PAUSED

        seq.resume()
        assert seq.state == TweenState.PLAYING

    def test_sequence_stop(self, target: AnimatableObject) -> None:
        """stop should stop all tweens."""
        t1 = Tween(target, "x", 0.0, 50.0, duration=0.5)
        t2 = Tween(target, "x", 50.0, 100.0, duration=0.5)
        seq = TweenSequence([t1, t2])

        seq.start()
        seq.update(0.3)
        seq.stop()

        assert seq.state == TweenState.IDLE

    def test_sequence_on_complete_callback(self, target: AnimatableObject) -> None:
        """on_complete should be called when sequence finishes."""
        callback = Mock()
        t1 = Tween(target, "x", 0.0, 100.0, duration=0.5)
        seq = TweenSequence([t1])
        seq.on_complete(callback)

        seq.start()
        seq.update(0.6)

        callback.assert_called_once()


# =============================================================================
# TWEEN GROUP TESTS
# =============================================================================


class TestTweenGroup:
    """Tests for TweenGroup class."""

    def test_group_init_empty(self) -> None:
        """Should initialize with empty tween list."""
        group = TweenGroup()
        assert len(group.tweens) == 0

    def test_group_init_with_tweens(self, target: AnimatableObject) -> None:
        """Should initialize with provided tweens."""
        t1 = Tween(target, "x", 0.0, 100.0)
        t2 = Tween(target, "y", 0.0, 50.0)
        group = TweenGroup([t1, t2])

        assert len(group.tweens) == 2

    def test_group_add(self, target: AnimatableObject) -> None:
        """Should be able to add tweens."""
        group = TweenGroup()
        t = Tween(target, "x", 0.0, 100.0)

        result = group.add(t)

        assert len(group.tweens) == 1
        assert result is group

    def test_group_start_all(self, target: AnimatableObject) -> None:
        """start should start all tweens simultaneously."""
        t1 = Tween(target, "x", 0.0, 100.0)
        t2 = Tween(target, "y", 0.0, 50.0)
        group = TweenGroup([t1, t2])

        group.start()

        assert t1.is_playing is True
        assert t2.is_playing is True

    def test_group_updates_all(self, target: AnimatableObject) -> None:
        """update should update all tweens."""
        t1 = Tween(target, "x", 0.0, 100.0, duration=1.0, easing="linear")
        t2 = Tween(target, "y", 0.0, 50.0, duration=1.0, easing="linear")
        group = TweenGroup([t1, t2])

        group.start()
        group.update(0.5)

        assert target.x == pytest.approx(50.0)
        assert target.y == pytest.approx(25.0)

    def test_group_completes_when_all_done(self, target: AnimatableObject) -> None:
        """Group should complete when all tweens are done."""
        t1 = Tween(target, "x", 0.0, 100.0, duration=0.5)
        t2 = Tween(target, "y", 0.0, 50.0, duration=1.0)
        group = TweenGroup([t1, t2])

        group.start()
        group.update(0.6)

        # First tween done, second still going
        assert group.is_complete is False

        group.update(0.5)

        # Now both done
        assert group.is_complete is True

    def test_group_progress_average(self, target: AnimatableObject) -> None:
        """Group progress should be average of all tweens."""
        t1 = Tween(target, "x", 0.0, 100.0, duration=1.0)
        t2 = Tween(target, "y", 0.0, 50.0, duration=1.0)
        group = TweenGroup([t1, t2])

        group.start()
        group.update(0.5)

        # Both at 50%, average is 50%
        assert group.progress == pytest.approx(0.5)

    def test_group_pause_resume(self, target: AnimatableObject) -> None:
        """Should be able to pause and resume group."""
        t1 = Tween(target, "x", 0.0, 100.0)
        group = TweenGroup([t1])

        group.start()
        group.pause()
        assert group.state == TweenState.PAUSED

        group.resume()
        assert group.state == TweenState.PLAYING

    def test_group_on_complete_callback(self, target: AnimatableObject) -> None:
        """on_complete should be called when group finishes."""
        callback = Mock()
        t1 = Tween(target, "x", 0.0, 100.0, duration=0.5)
        group = TweenGroup([t1])
        group.on_complete(callback)

        group.start()
        group.update(0.6)

        callback.assert_called_once()


# =============================================================================
# TWEEN MANAGER TESTS
# =============================================================================


class TestTweenManager:
    """Tests for TweenManager class."""

    def test_manager_init(self) -> None:
        """Should initialize with no tweens."""
        manager = TweenManager()
        assert manager.active_count == 0

    def test_manager_create(self, target: AnimatableObject) -> None:
        """create should create and register a tween."""
        manager = TweenManager()
        target.x = 50.0

        tween = manager.create(target, "x", 100.0, duration=1.0)

        assert tween.from_value == 50.0
        assert tween.to_value == 100.0
        assert manager.active_count == 1

    def test_manager_add(self, target: AnimatableObject) -> None:
        """add should register an existing tween."""
        manager = TweenManager()
        tween = Tween(target, "x", 0.0, 100.0)

        manager.add(tween)

        assert manager.active_count == 1

    def test_manager_remove(self, target: AnimatableObject) -> None:
        """remove should unregister a tween."""
        manager = TweenManager()
        tween = Tween(target, "x", 0.0, 100.0)
        manager.add(tween)

        manager.remove(tween)
        manager.update(0.0)  # Process removal

        assert manager.active_count == 0

    def test_manager_clear(self, target: AnimatableObject) -> None:
        """clear should remove all tweens."""
        manager = TweenManager()
        manager.add(Tween(target, "x", 0.0, 100.0))
        manager.add(Tween(target, "y", 0.0, 50.0))

        manager.clear()

        assert manager.active_count == 0

    def test_manager_update_all(self, target: AnimatableObject) -> None:
        """update should update all managed tweens."""
        manager = TweenManager()
        t1 = Tween(target, "x", 0.0, 100.0, duration=1.0, easing="linear")
        t2 = Tween(target, "y", 0.0, 50.0, duration=1.0, easing="linear")
        manager.add(t1)
        manager.add(t2)
        t1.start()
        t2.start()

        manager.update(0.5)

        assert target.x == pytest.approx(50.0)
        assert target.y == pytest.approx(25.0)

    def test_manager_removes_completed(self, target: AnimatableObject) -> None:
        """update should remove completed tweens."""
        manager = TweenManager()
        tween = Tween(target, "x", 0.0, 100.0, duration=0.5)
        manager.add(tween)
        tween.start()

        manager.update(0.6)

        assert manager.active_count == 0


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_tween_to(self, target: AnimatableObject) -> None:
        """tween_to should create tween from current value."""
        target.x = 25.0

        t = tween_to(target, "x", 100.0, duration=1.0)

        assert t.from_value == 25.0
        assert t.to_value == 100.0

    def test_tween_from(self, target: AnimatableObject) -> None:
        """tween_from should create tween to current value."""
        target.x = 100.0

        t = tween_from(target, "x", 0.0, duration=1.0)

        assert t.from_value == 0.0
        assert t.to_value == 100.0

    def test_tween_by(self, target: AnimatableObject) -> None:
        """tween_by should create tween by relative amount."""
        target.x = 50.0

        t = tween_by(target, "x", 30.0, duration=1.0)

        assert t.from_value == 50.0
        assert t.to_value == 80.0

    def test_factory_accepts_easing(self, target: AnimatableObject) -> None:
        """Factory functions should accept easing parameter."""
        t = tween_to(target, "x", 100.0, duration=1.0, easing="quad_out")
        assert t is not None

    def test_factory_accepts_kwargs(self, target: AnimatableObject) -> None:
        """Factory functions should pass kwargs to Tween."""
        callback = Mock()
        t = tween_to(target, "x", 100.0, repeat=2, yoyo=True)
        t.on_repeat(callback)
        t.start()
        t.update(1.1)

        callback.assert_called_once()

"""
Comprehensive tests for the Keyframe animation system.

Tests cover:
- Keyframe class
- KeyframeTrack interpolation and management
- KeyframeAnimation playback and control
- Loop modes (once, loop, ping_pong)
- Callbacks
- KeyframeAnimationManager
- Factory functions
- Edge cases
"""

from __future__ import annotations

import pytest
from typing import List, Any, Optional
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from engine.ui.animation.keyframe import (
    Keyframe,
    KeyframeTrack,
    KeyframeAnimation,
    KeyframeAnimationManager,
    LoopMode,
    AnimationState,
    AnimationCallback,
    KeyframeCallback,
    create_keyframe_animation,
    create_property_track,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@dataclass
class MockTarget:
    """Mock target for animation testing."""
    x: float = 0.0
    y: float = 0.0
    opacity: float = 1.0
    scale: float = 1.0
    color: tuple = (255, 255, 255)


@pytest.fixture
def target() -> MockTarget:
    """Create a fresh mock target for each test."""
    return MockTarget()


@pytest.fixture
def track() -> KeyframeTrack:
    """Create a basic keyframe track for testing."""
    return KeyframeTrack(
        "opacity",
        keyframes=[
            Keyframe(0.0, 0.0),
            Keyframe(0.5, 0.5),
            Keyframe(1.0, 1.0),
        ]
    )


@pytest.fixture
def animation(target: MockTarget) -> KeyframeAnimation:
    """Create a basic animation for testing."""
    anim = KeyframeAnimation("test", duration=1.0)
    anim.add_track(KeyframeTrack(
        "opacity",
        keyframes=[
            Keyframe(0.0, 0.0),
            Keyframe(1.0, 1.0),
        ]
    ))
    anim.set_target(target)
    return anim


# =============================================================================
# KEYFRAME TESTS
# =============================================================================


class TestKeyframe:
    """Tests for Keyframe dataclass."""

    def test_init_minimal(self) -> None:
        """Should initialize with time and value."""
        kf = Keyframe(0.5, 100)
        assert kf.time == 0.5
        assert kf.value == 100
        assert kf.easing == "linear"

    def test_init_with_easing(self) -> None:
        """Should accept easing parameter."""
        kf = Keyframe(0.5, 100, "ease_out")
        assert kf.easing == "ease_out"

    def test_easing_function_resolved(self) -> None:
        """Easing function should be resolved on init."""
        kf = Keyframe(0.5, 100, "quad_in")
        assert kf._easing_func is not None

    def test_get_eased_progress_linear(self) -> None:
        """Linear easing should return progress unchanged."""
        kf = Keyframe(0.5, 100, "linear")
        assert kf.get_eased_progress(0.5) == 0.5

    def test_get_eased_progress_ease_in(self) -> None:
        """Ease in should slow start."""
        kf = Keyframe(0.5, 100, "quad_in")
        # quad_in at 0.5 = 0.25
        assert kf.get_eased_progress(0.5) == pytest.approx(0.25)

    def test_callable_easing(self) -> None:
        """Should accept callable easing."""
        custom_easing = lambda t: t * t * t
        kf = Keyframe(0.5, 100, custom_easing)

        assert kf.get_eased_progress(0.5) == pytest.approx(0.125)


# =============================================================================
# KEYFRAME TRACK TESTS
# =============================================================================


class TestKeyframeTrack:
    """Tests for KeyframeTrack class."""

    def test_init_empty(self) -> None:
        """Should initialize with property name."""
        track = KeyframeTrack("x")
        assert track.property_name == "x"
        assert len(track.keyframes) == 0

    def test_init_with_keyframes(self) -> None:
        """Should initialize with keyframes."""
        kfs = [Keyframe(0.0, 0), Keyframe(1.0, 100)]
        track = KeyframeTrack("x", keyframes=kfs)
        assert len(track.keyframes) == 2

    def test_keyframes_sorted_on_init(self) -> None:
        """Keyframes should be sorted by time on init."""
        kfs = [Keyframe(1.0, 100), Keyframe(0.0, 0), Keyframe(0.5, 50)]
        track = KeyframeTrack("x", keyframes=kfs)

        assert track.keyframes[0].time == 0.0
        assert track.keyframes[1].time == 0.5
        assert track.keyframes[2].time == 1.0

    def test_add_keyframe(self) -> None:
        """add_keyframe should add and return self."""
        track = KeyframeTrack("x")

        result = track.add_keyframe(0.5, 50)

        assert result is track
        assert len(track.keyframes) == 1
        assert track.keyframes[0].time == 0.5

    def test_add_keyframe_with_easing(self) -> None:
        """add_keyframe should accept easing."""
        track = KeyframeTrack("x")
        track.add_keyframe(0.5, 50, "ease_out")

        assert track.keyframes[0].easing == "ease_out"

    def test_remove_keyframe(self) -> None:
        """remove_keyframe should remove by index."""
        track = KeyframeTrack("x", keyframes=[
            Keyframe(0.0, 0),
            Keyframe(0.5, 50),
            Keyframe(1.0, 100),
        ])

        result = track.remove_keyframe(1)

        assert result is True
        assert len(track.keyframes) == 2
        assert track.keyframes[0].time == 0.0
        assert track.keyframes[1].time == 1.0

    def test_remove_keyframe_invalid_index(self) -> None:
        """remove_keyframe should return False for invalid index."""
        track = KeyframeTrack("x")
        assert track.remove_keyframe(0) is False

    def test_clear(self) -> None:
        """clear should remove all keyframes."""
        track = KeyframeTrack("x", keyframes=[
            Keyframe(0.0, 0),
            Keyframe(1.0, 100),
        ])

        result = track.clear()

        assert result is track
        assert len(track.keyframes) == 0

    def test_get_value_at_start(self, track: KeyframeTrack) -> None:
        """get_value_at should return first keyframe value at 0."""
        assert track.get_value_at(0.0) == 0.0

    def test_get_value_at_end(self, track: KeyframeTrack) -> None:
        """get_value_at should return last keyframe value at 1."""
        assert track.get_value_at(1.0) == 1.0

    def test_get_value_at_midpoint(self, track: KeyframeTrack) -> None:
        """get_value_at should return exact value at keyframe time."""
        assert track.get_value_at(0.5) == pytest.approx(0.5)

    def test_get_value_at_interpolated(self) -> None:
        """get_value_at should interpolate between keyframes."""
        track = KeyframeTrack("x", keyframes=[
            Keyframe(0.0, 0.0, "linear"),
            Keyframe(1.0, 100.0),
        ])

        assert track.get_value_at(0.25) == pytest.approx(25.0)
        assert track.get_value_at(0.75) == pytest.approx(75.0)

    def test_get_value_at_with_easing(self) -> None:
        """get_value_at should apply keyframe easing."""
        track = KeyframeTrack("x", keyframes=[
            Keyframe(0.0, 0.0, "quad_in"),
            Keyframe(1.0, 100.0),
        ])

        # quad_in at 0.5 = 0.25, so value = 25
        assert track.get_value_at(0.5) == pytest.approx(25.0)

    def test_get_value_at_empty_track(self) -> None:
        """get_value_at should return None for empty track."""
        track = KeyframeTrack("x")
        assert track.get_value_at(0.5) is None

    def test_get_value_at_before_first(self) -> None:
        """get_value_at before first keyframe should return first value."""
        track = KeyframeTrack("x", keyframes=[
            Keyframe(0.5, 50.0),
            Keyframe(1.0, 100.0),
        ])

        assert track.get_value_at(0.0) == 50.0

    def test_get_value_at_after_last(self) -> None:
        """get_value_at after last keyframe should return last value."""
        track = KeyframeTrack("x", keyframes=[
            Keyframe(0.0, 0.0),
            Keyframe(0.5, 50.0),
        ])

        assert track.get_value_at(1.0) == 50.0

    def test_interpolate_int(self) -> None:
        """Should interpolate integers."""
        track = KeyframeTrack("x", keyframes=[
            Keyframe(0.0, 0, "linear"),
            Keyframe(1.0, 100),
        ])

        result = track.get_value_at(0.5)
        assert result == 50

    def test_interpolate_tuple(self) -> None:
        """Should interpolate tuples (colors)."""
        track = KeyframeTrack("color", keyframes=[
            Keyframe(0.0, (0.0, 0.0, 0.0), "linear"),  # Use floats for float interpolation
            Keyframe(1.0, (255.0, 255.0, 255.0)),
        ])

        result = track.get_value_at(0.5)
        assert result[0] == pytest.approx(127.5)
        assert result[1] == pytest.approx(127.5)
        assert result[2] == pytest.approx(127.5)

    def test_interpolate_list(self) -> None:
        """Should interpolate lists."""
        track = KeyframeTrack("vec", keyframes=[
            Keyframe(0.0, [0.0, 0.0], "linear"),
            Keyframe(1.0, [100.0, 50.0]),
        ])

        result = track.get_value_at(0.5)
        assert result[0] == pytest.approx(50.0)
        assert result[1] == pytest.approx(25.0)

    def test_on_keyframe_reached_callback(self) -> None:
        """on_keyframe_reached should fire when crossing keyframes."""
        callback = Mock()
        track = KeyframeTrack("x", keyframes=[
            Keyframe(0.0, 0),
            Keyframe(0.5, 50),
            Keyframe(1.0, 100),
        ], on_keyframe_reached=callback)

        track.get_value_at(0.0)
        callback.assert_called_with(0)

        track.get_value_at(0.6)
        callback.assert_called_with(1)

    def test_reset(self) -> None:
        """reset should clear last keyframe index."""
        track = KeyframeTrack("x", keyframes=[
            Keyframe(0.0, 0),
            Keyframe(1.0, 100),
        ])
        track.get_value_at(0.5)

        track.reset()

        assert track._last_keyframe_index == -1


# =============================================================================
# KEYFRAME ANIMATION TESTS
# =============================================================================


class TestKeyframeAnimation:
    """Tests for KeyframeAnimation class."""

    def test_init_minimal(self) -> None:
        """Should initialize with name."""
        anim = KeyframeAnimation("test")
        assert anim.name == "test"
        assert anim.duration == 1.0
        assert anim.loop_mode == LoopMode.ONCE
        assert anim.speed == 1.0

    def test_init_with_options(self) -> None:
        """Should accept configuration options."""
        anim = KeyframeAnimation(
            "test",
            duration=2.0,
            loop_mode=LoopMode.LOOP,
            speed=0.5
        )
        assert anim.duration == 2.0
        assert anim.loop_mode == LoopMode.LOOP
        assert anim.speed == 0.5

    def test_duration_minimum(self) -> None:
        """Duration should be at least 0.001."""
        anim = KeyframeAnimation("test", duration=0.0)
        assert anim.duration >= 0.001

    def test_state_initially_idle(self) -> None:
        """Should start in IDLE state."""
        anim = KeyframeAnimation("test")
        assert anim.state == AnimationState.IDLE

    def test_set_target(self, target: MockTarget) -> None:
        """set_target should store target reference."""
        anim = KeyframeAnimation("test")

        result = anim.set_target(target)

        assert result is anim
        assert anim.target is target

    def test_add_track(self) -> None:
        """add_track should add track and return self."""
        anim = KeyframeAnimation("test")
        track = KeyframeTrack("x")

        result = anim.add_track(track)

        assert result is anim
        assert "x" in anim.tracks

    def test_remove_track(self) -> None:
        """remove_track should remove track by name."""
        anim = KeyframeAnimation("test")
        anim.add_track(KeyframeTrack("x"))

        result = anim.remove_track("x")

        assert result is True
        assert "x" not in anim.tracks

    def test_remove_nonexistent_track(self) -> None:
        """remove_track should return False for missing track."""
        anim = KeyframeAnimation("test")
        assert anim.remove_track("x") is False

    def test_get_track(self) -> None:
        """get_track should return track by name."""
        anim = KeyframeAnimation("test")
        track = KeyframeTrack("x")
        anim.add_track(track)

        result = anim.get_track("x")

        assert result is track

    def test_clear_tracks(self) -> None:
        """clear_tracks should remove all tracks."""
        anim = KeyframeAnimation("test")
        anim.add_track(KeyframeTrack("x"))
        anim.add_track(KeyframeTrack("y"))

        result = anim.clear_tracks()

        assert result is anim
        assert len(anim.tracks) == 0


# =============================================================================
# KEYFRAME ANIMATION PLAYBACK TESTS
# =============================================================================


class TestKeyframeAnimationPlayback:
    """Tests for KeyframeAnimation playback control."""

    def test_start_changes_state(self, animation: KeyframeAnimation) -> None:
        """start should change state to PLAYING."""
        result = animation.start()

        assert result is animation
        assert animation.state == AnimationState.PLAYING
        assert animation.progress == pytest.approx(0.0)

    def test_start_resets_elapsed(self, animation: KeyframeAnimation) -> None:
        """start should reset elapsed time."""
        animation.start()
        animation.update(0.5)
        animation.start()

        assert animation.elapsed == pytest.approx(0.0)

    def test_start_fires_callback(self, animation: KeyframeAnimation) -> None:
        """start should fire on_start callback."""
        callback = Mock()
        animation.on_start(callback)

        animation.start()

        callback.assert_called_once()

    def test_pause_changes_state(self, animation: KeyframeAnimation) -> None:
        """pause should change state to PAUSED."""
        animation.start()

        result = animation.pause()

        assert result is animation
        assert animation.state == AnimationState.PAUSED

    def test_pause_does_nothing_when_not_playing(self, animation: KeyframeAnimation) -> None:
        """pause should not affect non-playing state."""
        animation.pause()
        assert animation.state == AnimationState.IDLE

    def test_resume_changes_state(self, animation: KeyframeAnimation) -> None:
        """resume should change from PAUSED to PLAYING."""
        animation.start()
        animation.pause()

        result = animation.resume()

        assert result is animation
        assert animation.state == AnimationState.PLAYING

    def test_stop_resets_animation(self, animation: KeyframeAnimation) -> None:
        """stop should reset to IDLE state."""
        animation.start()
        animation.update(0.5)

        result = animation.stop()

        assert result is animation
        assert animation.state == AnimationState.IDLE
        assert animation.elapsed == pytest.approx(0.0)

    def test_complete_jumps_to_end(self, animation: KeyframeAnimation, target: MockTarget) -> None:
        """complete should immediately finish animation."""
        animation.start()

        result = animation.complete()

        assert result is animation
        assert animation.state == AnimationState.COMPLETED
        assert target.opacity == pytest.approx(1.0)

    def test_complete_fires_callback(self, animation: KeyframeAnimation) -> None:
        """complete should fire on_complete callback."""
        callback = Mock()
        animation.on_complete(callback)
        animation.start()

        animation.complete()

        callback.assert_called_once()


# =============================================================================
# KEYFRAME ANIMATION UPDATE TESTS
# =============================================================================


class TestKeyframeAnimationUpdate:
    """Tests for KeyframeAnimation update method."""

    def test_update_advances_time(self, animation: KeyframeAnimation) -> None:
        """update should advance elapsed time."""
        animation.start()

        animation.update(0.3)

        assert animation.elapsed == pytest.approx(0.3)
        assert animation.progress == pytest.approx(0.3)

    def test_update_applies_speed(self, animation: KeyframeAnimation) -> None:
        """update should apply speed multiplier."""
        animation.speed = 2.0
        animation.start()

        animation.update(0.25)

        assert animation.elapsed == pytest.approx(0.5)

    def test_update_applies_values(self, animation: KeyframeAnimation, target: MockTarget) -> None:
        """update should apply interpolated values to target."""
        animation.start()

        animation.update(0.5)

        assert target.opacity == pytest.approx(0.5)

    def test_update_returns_true_while_playing(self, animation: KeyframeAnimation) -> None:
        """update should return True while playing."""
        animation.start()

        result = animation.update(0.3)

        assert result is True

    def test_update_returns_false_when_complete(self, animation: KeyframeAnimation) -> None:
        """update should return False when complete."""
        animation.start()

        result = animation.update(1.5)

        assert result is False
        assert animation.state == AnimationState.COMPLETED

    def test_update_fires_update_callback(self, animation: KeyframeAnimation) -> None:
        """update should fire on_update callback."""
        callback = Mock()
        animation.on_update(callback)
        animation.start()

        animation.update(0.5)

        callback.assert_called_with(pytest.approx(0.5))

    def test_update_does_nothing_when_paused(self, animation: KeyframeAnimation) -> None:
        """update should not advance when paused."""
        animation.start()
        animation.update(0.3)
        animation.pause()
        elapsed_before = animation.elapsed

        animation.update(0.5)

        assert animation.elapsed == pytest.approx(elapsed_before)


# =============================================================================
# LOOP MODE TESTS
# =============================================================================


class TestLoopModes:
    """Tests for animation loop modes."""

    def test_loop_once_completes(self, animation: KeyframeAnimation) -> None:
        """ONCE mode should complete after one cycle."""
        animation.loop_mode = LoopMode.ONCE
        animation.start()

        animation.update(1.5)

        assert animation.state == AnimationState.COMPLETED

    def test_loop_mode_repeats(self, animation: KeyframeAnimation) -> None:
        """LOOP mode should restart from beginning."""
        animation.loop_mode = LoopMode.LOOP
        animation.start()

        animation.update(1.2)

        assert animation.state == AnimationState.PLAYING
        assert animation.elapsed == pytest.approx(0.0)
        assert animation.loop_count == 1

    def test_loop_fires_callback(self, animation: KeyframeAnimation) -> None:
        """LOOP mode should fire on_loop callback."""
        callback = Mock()
        animation.loop_mode = LoopMode.LOOP
        animation.on_loop(callback)
        animation.start()

        animation.update(1.2)

        callback.assert_called_once()

    def test_ping_pong_reverses(self, animation: KeyframeAnimation) -> None:
        """PING_PONG mode should reverse direction."""
        animation.loop_mode = LoopMode.PING_PONG
        animation.start()

        # First cycle complete
        animation.update(1.2)

        assert animation.state == AnimationState.PLAYING
        assert animation._direction == -1

    def test_ping_pong_cycle_count(self, animation: KeyframeAnimation) -> None:
        """PING_PONG should increment loop count on full cycle."""
        animation.loop_mode = LoopMode.PING_PONG
        animation.start()

        # Forward
        animation.update(1.2)
        assert animation.loop_count == 0

        # Backward (completes one full cycle)
        animation.update(1.2)
        assert animation.loop_count == 1


# =============================================================================
# SEEK TESTS
# =============================================================================


class TestSeek:
    """Tests for seek functionality."""

    def test_seek(self, animation: KeyframeAnimation, target: MockTarget) -> None:
        """seek should jump to specific time."""
        animation.start()

        result = animation.seek(0.5)

        assert result is animation
        assert animation.elapsed == pytest.approx(0.5)
        assert target.opacity == pytest.approx(0.5)

    def test_seek_clamped(self, animation: KeyframeAnimation) -> None:
        """seek should clamp to valid range."""
        animation.start()

        animation.seek(-1.0)
        assert animation.elapsed == pytest.approx(0.0)

        animation.seek(10.0)
        assert animation.elapsed == pytest.approx(1.0)

    def test_seek_normalized(self, animation: KeyframeAnimation) -> None:
        """seek_normalized should jump to progress value."""
        animation.start()

        result = animation.seek_normalized(0.5)

        assert result is animation
        assert animation.progress == pytest.approx(0.5)


# =============================================================================
# CALLBACK TESTS
# =============================================================================


class TestAnimationCallbacks:
    """Tests for animation callbacks."""

    def test_on_start_fluent(self, animation: KeyframeAnimation) -> None:
        """on_start should return self."""
        result = animation.on_start(lambda: None)
        assert result is animation

    def test_on_complete_fluent(self, animation: KeyframeAnimation) -> None:
        """on_complete should return self."""
        result = animation.on_complete(lambda: None)
        assert result is animation

    def test_on_loop_fluent(self, animation: KeyframeAnimation) -> None:
        """on_loop should return self."""
        result = animation.on_loop(lambda: None)
        assert result is animation

    def test_on_update_fluent(self, animation: KeyframeAnimation) -> None:
        """on_update should return self."""
        result = animation.on_update(lambda p: None)
        assert result is animation


# =============================================================================
# KEYFRAME ANIMATION MANAGER TESTS
# =============================================================================


class TestKeyframeAnimationManager:
    """Tests for KeyframeAnimationManager class."""

    def test_init_empty(self) -> None:
        """Should initialize with no animations."""
        manager = KeyframeAnimationManager()
        assert manager.count == 0
        assert manager.active_count == 0

    def test_register(self) -> None:
        """register should add animation by name."""
        manager = KeyframeAnimationManager()
        anim = KeyframeAnimation("test")

        manager.register(anim)

        assert manager.count == 1
        assert manager.get("test") is anim

    def test_unregister(self) -> None:
        """unregister should remove animation."""
        manager = KeyframeAnimationManager()
        anim = KeyframeAnimation("test")
        manager.register(anim)

        result = manager.unregister("test")

        assert result is True
        assert manager.count == 0

    def test_unregister_nonexistent(self) -> None:
        """unregister should return False for missing animation."""
        manager = KeyframeAnimationManager()
        assert manager.unregister("unknown") is False

    def test_get(self) -> None:
        """get should return animation by name."""
        manager = KeyframeAnimationManager()
        anim = KeyframeAnimation("test")
        manager.register(anim)

        result = manager.get("test")

        assert result is anim

    def test_get_nonexistent(self) -> None:
        """get should return None for missing animation."""
        manager = KeyframeAnimationManager()
        assert manager.get("unknown") is None

    def test_play(self, target: MockTarget) -> None:
        """play should start animation by name."""
        manager = KeyframeAnimationManager()
        anim = KeyframeAnimation("test")
        anim.set_target(target)
        manager.register(anim)

        result = manager.play("test")

        assert result is True
        assert anim.is_playing is True
        assert manager.active_count == 1

    def test_play_with_target(self, target: MockTarget) -> None:
        """play should set target if provided."""
        manager = KeyframeAnimationManager()
        anim = KeyframeAnimation("test")
        manager.register(anim)

        manager.play("test", target)

        assert anim.target is target

    def test_play_nonexistent(self) -> None:
        """play should return False for missing animation."""
        manager = KeyframeAnimationManager()
        assert manager.play("unknown") is False

    def test_stop(self, target: MockTarget) -> None:
        """stop should stop animation by name."""
        manager = KeyframeAnimationManager()
        anim = KeyframeAnimation("test")
        anim.set_target(target)
        manager.register(anim)
        manager.play("test")

        result = manager.stop("test")

        assert result is True
        assert anim.state == AnimationState.IDLE
        assert manager.active_count == 0

    def test_stop_all(self, target: MockTarget) -> None:
        """stop_all should stop all active animations."""
        manager = KeyframeAnimationManager()
        anim1 = KeyframeAnimation("test1")
        anim2 = KeyframeAnimation("test2")
        anim1.set_target(target)
        anim2.set_target(target)
        manager.register(anim1)
        manager.register(anim2)
        manager.play("test1")
        manager.play("test2")

        manager.stop_all()

        assert manager.active_count == 0

    def test_update(self, target: MockTarget) -> None:
        """update should update all active animations."""
        manager = KeyframeAnimationManager()
        anim = KeyframeAnimation("test", duration=1.0)
        anim.add_track(KeyframeTrack("opacity", keyframes=[
            Keyframe(0.0, 0.0, "linear"),
            Keyframe(1.0, 1.0),
        ]))
        anim.set_target(target)
        manager.register(anim)
        manager.play("test")

        manager.update(0.5)

        assert target.opacity == pytest.approx(0.5)

    def test_update_removes_completed(self, target: MockTarget) -> None:
        """update should remove completed animations from active list."""
        manager = KeyframeAnimationManager()
        anim = KeyframeAnimation("test", duration=0.5)
        anim.set_target(target)
        manager.register(anim)
        manager.play("test")

        manager.update(1.0)

        assert manager.active_count == 0


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_keyframe_animation(self) -> None:
        """create_keyframe_animation should create configured animation."""
        anim = create_keyframe_animation(
            "test",
            duration=2.0,
            loop_mode=LoopMode.LOOP,
            opacity=[(0.0, 0.0), (1.0, 1.0)],
            scale=[(0.0, 0.5), (1.0, 1.0)],
        )

        assert anim.name == "test"
        assert anim.duration == 2.0
        assert anim.loop_mode == LoopMode.LOOP
        assert "opacity" in anim.tracks
        assert "scale" in anim.tracks

    def test_create_property_track_simple(self) -> None:
        """create_property_track should create track from tuples."""
        track = create_property_track("x", [
            (0.0, 0),
            (0.5, 50),
            (1.0, 100),
        ])

        assert track.property_name == "x"
        assert len(track.keyframes) == 3
        assert track.keyframes[0].time == 0.0
        assert track.keyframes[0].value == 0

    def test_create_property_track_with_easing(self) -> None:
        """create_property_track should accept easing in tuples."""
        track = create_property_track("x", [
            (0.0, 0, "ease_out"),
            (1.0, 100, "ease_in"),
        ])

        assert track.keyframes[0].easing == "ease_out"
        assert track.keyframes[1].easing == "ease_in"

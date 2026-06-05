"""
Whitebox tests for AnimationState class.

Tests internal implementation details with full source access:
- AnimationState dataclass fields and initialization
- MotionMode enum values (LOOP, ONCE, PING_PONG)
- Motion handling for each mode
- Speed multiplier effects
- Callbacks (on_enter, on_exit, on_update)
- Time tracking and reset
- Internal state fields (_normalized_time, _playback_direction, _animation_finished)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional
from unittest.mock import MagicMock, Mock, call

import pytest

from engine.animation.graph.state_machine import (
    AnimationState,
    MotionMode,
)
from engine.animation.graph.animation_graph import (
    GraphContext,
    Pose,
    Transform,
)


# =============================================================================
# FIXTURES
# =============================================================================


@dataclass
class MockAnimationClip:
    """Mock animation clip for testing."""

    duration: float = 2.0
    _sample_calls: List[tuple] = field(default_factory=list)

    def sample(self, time: float, bone_count: int) -> Pose:
        """Sample the clip at the given time."""
        self._sample_calls.append((time, bone_count))
        return Pose(transforms=[Transform() for _ in range(bone_count)])


@pytest.fixture
def mock_context() -> GraphContext:
    """Create a mock graph context."""
    return GraphContext(
        parameters={},
        dt=0.016,
        skeleton=None,
        bone_masks={},
        normalized_time=0.0,
        sync_group=None,
        layer_weight=1.0,
    )


@pytest.fixture
def basic_state() -> AnimationState:
    """Create a basic animation state for testing."""
    return AnimationState(name="test_state")


@pytest.fixture
def state_with_clip() -> AnimationState:
    """Create an animation state with a mock clip."""
    clip = MockAnimationClip(duration=2.0)
    return AnimationState(name="clip_state", clip=clip)


# =============================================================================
# TEST: DATACLASS FIELDS
# =============================================================================


class TestAnimationStateFields:
    """Test AnimationState dataclass field initialization and defaults."""

    def test_required_name_field(self) -> None:
        """Name field is required."""
        state = AnimationState(name="my_state")
        assert state.name == "my_state"

    def test_default_clip_is_none(self) -> None:
        """Default clip field is None."""
        state = AnimationState(name="test")
        assert state.clip is None

    def test_default_graph_is_none(self) -> None:
        """Default graph field is None."""
        state = AnimationState(name="test")
        assert state.graph is None

    def test_default_animation_node_is_none(self) -> None:
        """Default animation_node field is None."""
        state = AnimationState(name="test")
        assert state.animation_node is None

    def test_default_motion_mode_is_loop(self) -> None:
        """Default motion_mode is LOOP."""
        state = AnimationState(name="test")
        assert state.motion_mode == MotionMode.LOOP

    def test_default_speed_is_one(self) -> None:
        """Default speed is 1.0."""
        state = AnimationState(name="test")
        assert state.speed == 1.0

    def test_default_current_time_is_zero(self) -> None:
        """Default current_time is 0.0."""
        state = AnimationState(name="test")
        assert state.current_time == 0.0

    def test_default_on_enter_is_none(self) -> None:
        """Default on_enter callback is None."""
        state = AnimationState(name="test")
        assert state.on_enter is None

    def test_default_on_exit_is_none(self) -> None:
        """Default on_exit callback is None."""
        state = AnimationState(name="test")
        assert state.on_exit is None

    def test_default_on_update_is_none(self) -> None:
        """Default on_update callback is None."""
        state = AnimationState(name="test")
        assert state.on_update is None

    def test_default_can_interrupt_is_true(self) -> None:
        """Default can_interrupt is True."""
        state = AnimationState(name="test")
        assert state.can_interrupt is True

    def test_internal_normalized_time_default(self) -> None:
        """Internal _normalized_time defaults to 0.0."""
        state = AnimationState(name="test")
        assert state._normalized_time == 0.0

    def test_internal_playback_direction_default(self) -> None:
        """Internal _playback_direction defaults to 1 (forward)."""
        state = AnimationState(name="test")
        assert state._playback_direction == 1

    def test_internal_animation_finished_default(self) -> None:
        """Internal _animation_finished defaults to False."""
        state = AnimationState(name="test")
        assert state._animation_finished is False

    def test_custom_field_initialization(self) -> None:
        """All fields can be set during initialization."""
        on_enter = Mock()
        on_exit = Mock()
        clip = MockAnimationClip()

        state = AnimationState(
            name="custom_state",
            clip=clip,
            motion_mode=MotionMode.ONCE,
            speed=2.5,
            on_enter=on_enter,
            on_exit=on_exit,
            can_interrupt=False,
        )

        assert state.name == "custom_state"
        assert state.clip is clip
        assert state.motion_mode == MotionMode.ONCE
        assert state.speed == 2.5
        assert state.on_enter is on_enter
        assert state.on_exit is on_exit
        assert state.can_interrupt is False


# =============================================================================
# TEST: MOTION MODE ENUM
# =============================================================================


class TestMotionModeEnum:
    """Test MotionMode enum values and behavior."""

    def test_loop_value_exists(self) -> None:
        """LOOP mode exists."""
        assert hasattr(MotionMode, "LOOP")
        assert MotionMode.LOOP is not None

    def test_once_value_exists(self) -> None:
        """ONCE mode exists."""
        assert hasattr(MotionMode, "ONCE")
        assert MotionMode.ONCE is not None

    def test_ping_pong_value_exists(self) -> None:
        """PING_PONG mode exists."""
        assert hasattr(MotionMode, "PING_PONG")
        assert MotionMode.PING_PONG is not None

    def test_values_are_distinct(self) -> None:
        """All motion mode values are distinct."""
        modes = [MotionMode.LOOP, MotionMode.ONCE, MotionMode.PING_PONG]
        assert len(modes) == len(set(modes))

    def test_enum_iteration(self) -> None:
        """Can iterate over MotionMode members."""
        members = list(MotionMode)
        assert len(members) == 3
        assert MotionMode.LOOP in members
        assert MotionMode.ONCE in members
        assert MotionMode.PING_PONG in members


# =============================================================================
# TEST: LOOP MOTION HANDLING
# =============================================================================


class TestLoopMotionMode:
    """Test LOOP motion mode: time wraps around at duration."""

    def test_time_wraps_at_duration(self) -> None:
        """Time wraps around when reaching clip duration."""
        clip = MockAnimationClip(duration=2.0)
        state = AnimationState(
            name="loop_test",
            clip=clip,
            motion_mode=MotionMode.LOOP,
            speed=1.0,
        )

        # Advance past the duration
        state.update(2.5, None)

        # Should wrap to 0.5
        assert pytest.approx(state.current_time, abs=0.01) == 0.5

    def test_multiple_wraps(self) -> None:
        """Time correctly wraps multiple times."""
        clip = MockAnimationClip(duration=1.0)
        state = AnimationState(
            name="loop_test",
            clip=clip,
            motion_mode=MotionMode.LOOP,
            speed=1.0,
        )

        # Advance by 3.7 seconds (3 full loops + 0.7)
        state.update(3.7, None)

        assert pytest.approx(state.current_time, abs=0.01) == 0.7

    def test_loop_never_finishes(self) -> None:
        """LOOP mode never sets is_finished flag."""
        clip = MockAnimationClip(duration=1.0)
        state = AnimationState(
            name="loop_test",
            clip=clip,
            motion_mode=MotionMode.LOOP,
        )

        # Update many times past duration
        for _ in range(10):
            state.update(0.5, None)

        assert state.is_finished is False

    def test_loop_continuous_updates(self) -> None:
        """LOOP mode continues updating indefinitely."""
        clip = MockAnimationClip(duration=1.0)
        state = AnimationState(
            name="loop_test",
            clip=clip,
            motion_mode=MotionMode.LOOP,
        )

        prev_time = state.current_time
        for i in range(5):
            state.update(0.3, None)
            # Time should advance (and wrap)
            assert state.current_time != prev_time or i == 0
            prev_time = state.current_time

    def test_normalized_time_in_loop(self) -> None:
        """Normalized time stays in 0-1 range during loop."""
        clip = MockAnimationClip(duration=2.0)
        state = AnimationState(
            name="loop_test",
            clip=clip,
            motion_mode=MotionMode.LOOP,
        )

        state.update(1.0, None)  # 50%
        assert pytest.approx(state.normalized_time, abs=0.01) == 0.5

        state.update(1.5, None)  # Wraps to 0.5 / 2.0 = 0.25
        assert state.normalized_time >= 0.0
        assert state.normalized_time <= 1.0


# =============================================================================
# TEST: ONCE MOTION HANDLING
# =============================================================================


class TestOnceMotionMode:
    """Test ONCE motion mode: stops at end, is_finished = True."""

    def test_stops_at_duration(self) -> None:
        """Time clamps to duration and doesn't exceed it."""
        clip = MockAnimationClip(duration=2.0)
        state = AnimationState(
            name="once_test",
            clip=clip,
            motion_mode=MotionMode.ONCE,
        )

        # Advance past duration
        state.update(3.0, None)

        assert state.current_time == 2.0

    def test_is_finished_true_at_end(self) -> None:
        """is_finished becomes True when reaching end."""
        clip = MockAnimationClip(duration=2.0)
        state = AnimationState(
            name="once_test",
            clip=clip,
            motion_mode=MotionMode.ONCE,
        )

        assert state.is_finished is False

        state.update(2.5, None)

        assert state.is_finished is True

    def test_stops_updating_after_finished(self) -> None:
        """No more updates after animation finishes."""
        clip = MockAnimationClip(duration=1.0)
        state = AnimationState(
            name="once_test",
            clip=clip,
            motion_mode=MotionMode.ONCE,
        )

        state.update(1.5, None)  # Finish
        assert state.is_finished is True

        prev_time = state.current_time
        state.update(0.5, None)  # Try to update more

        assert state.current_time == prev_time  # No change

    def test_reverse_playback_stops_at_zero(self) -> None:
        """With negative speed, stops at 0."""
        clip = MockAnimationClip(duration=2.0)
        state = AnimationState(
            name="once_test",
            clip=clip,
            motion_mode=MotionMode.ONCE,
            speed=-1.0,
        )
        state.current_time = 1.5  # Start partway through

        state.update(2.0, None)  # Go past beginning

        assert state.current_time == 0.0
        assert state.is_finished is True

    def test_normalized_time_at_end(self) -> None:
        """Normalized time reaches 1.0 at end."""
        clip = MockAnimationClip(duration=2.0)
        state = AnimationState(
            name="once_test",
            clip=clip,
            motion_mode=MotionMode.ONCE,
        )

        state.update(3.0, None)

        assert pytest.approx(state.normalized_time, abs=0.01) == 1.0


# =============================================================================
# TEST: PING_PONG MOTION HANDLING
# =============================================================================


class TestPingPongMotionMode:
    """Test PING_PONG motion mode: reverses direction at boundaries."""

    def test_reverses_at_duration(self) -> None:
        """Direction reverses when reaching duration."""
        clip = MockAnimationClip(duration=2.0)
        state = AnimationState(
            name="pingpong_test",
            clip=clip,
            motion_mode=MotionMode.PING_PONG,
        )

        assert state._playback_direction == 1  # Forward initially

        # Go past duration
        state.update(2.5, None)

        assert state._playback_direction == -1  # Now backward

    def test_reverses_at_zero(self) -> None:
        """Direction reverses when reaching 0."""
        clip = MockAnimationClip(duration=2.0)
        state = AnimationState(
            name="pingpong_test",
            clip=clip,
            motion_mode=MotionMode.PING_PONG,
        )

        # Force backward playback
        state._playback_direction = -1
        state.current_time = 0.5

        state.update(1.0, None)  # Would go to -0.5

        assert state._playback_direction == 1  # Back to forward
        assert state.current_time >= 0.0

    def test_pingpong_bounces_correctly(self) -> None:
        """Time bounces between 0 and duration."""
        clip = MockAnimationClip(duration=1.0)
        state = AnimationState(
            name="pingpong_test",
            clip=clip,
            motion_mode=MotionMode.PING_PONG,
        )

        # Go to 1.3 (should bounce to 0.7, direction reversed)
        state.update(1.3, None)

        assert state._playback_direction == -1
        assert pytest.approx(state.current_time, abs=0.1) == 0.7

    def test_pingpong_never_finishes(self) -> None:
        """PING_PONG mode never sets is_finished flag."""
        clip = MockAnimationClip(duration=1.0)
        state = AnimationState(
            name="pingpong_test",
            clip=clip,
            motion_mode=MotionMode.PING_PONG,
        )

        # Many updates
        for _ in range(20):
            state.update(0.3, None)

        assert state.is_finished is False

    def test_pingpong_time_stays_in_bounds(self) -> None:
        """Time always stays between 0 and duration."""
        clip = MockAnimationClip(duration=2.0)
        state = AnimationState(
            name="pingpong_test",
            clip=clip,
            motion_mode=MotionMode.PING_PONG,
        )

        for _ in range(50):
            state.update(0.15, None)
            assert 0.0 <= state.current_time <= 2.0


# =============================================================================
# TEST: SPEED MULTIPLIER
# =============================================================================


class TestSpeedMultiplier:
    """Test speed multiplier effects on playback."""

    def test_normal_speed_one(self) -> None:
        """Speed 1.0 advances time normally."""
        clip = MockAnimationClip(duration=10.0)
        state = AnimationState(
            name="speed_test",
            clip=clip,
            speed=1.0,
        )

        state.update(0.5, None)

        assert pytest.approx(state.current_time, abs=0.001) == 0.5

    def test_double_speed(self) -> None:
        """Speed 2.0 advances time twice as fast."""
        clip = MockAnimationClip(duration=10.0)
        state = AnimationState(
            name="speed_test",
            clip=clip,
            speed=2.0,
        )

        state.update(0.5, None)

        assert pytest.approx(state.current_time, abs=0.001) == 1.0

    def test_half_speed(self) -> None:
        """Speed 0.5 advances time at half rate."""
        clip = MockAnimationClip(duration=10.0)
        state = AnimationState(
            name="speed_test",
            clip=clip,
            speed=0.5,
        )

        state.update(1.0, None)

        assert pytest.approx(state.current_time, abs=0.001) == 0.5

    def test_negative_speed_reverse_playback(self) -> None:
        """Negative speed plays animation in reverse."""
        clip = MockAnimationClip(duration=10.0)
        state = AnimationState(
            name="speed_test",
            clip=clip,
            speed=-1.0,
            motion_mode=MotionMode.LOOP,
        )
        state.current_time = 5.0

        state.update(1.0, None)

        assert pytest.approx(state.current_time, abs=0.001) == 4.0

    def test_zero_speed_no_movement(self) -> None:
        """Speed 0.0 freezes the animation."""
        clip = MockAnimationClip(duration=10.0)
        state = AnimationState(
            name="speed_test",
            clip=clip,
            speed=0.0,
        )
        state.current_time = 3.0

        state.update(1.0, None)

        assert state.current_time == 3.0

    def test_large_speed_multiplier(self) -> None:
        """Large speed values work correctly."""
        clip = MockAnimationClip(duration=10.0)
        state = AnimationState(
            name="speed_test",
            clip=clip,
            speed=10.0,
            motion_mode=MotionMode.LOOP,
        )

        state.update(0.5, None)

        # Should wrap: 5.0 % 10.0 = 5.0
        assert pytest.approx(state.current_time, abs=0.01) == 5.0

    def test_speed_multiplier_legacy_property(self) -> None:
        """speed_multiplier property is legacy alias for speed."""
        state = AnimationState(name="test", speed=2.5)

        assert state.speed_multiplier == 2.5

        state.speed_multiplier = 3.0
        assert state.speed == 3.0


# =============================================================================
# TEST: CALLBACKS
# =============================================================================


class TestCallbacks:
    """Test on_enter, on_exit, and on_update callbacks."""

    def test_on_enter_called_when_entering_state(self, mock_context: GraphContext) -> None:
        """on_enter callback is called when entering the state."""
        on_enter = Mock()
        state = AnimationState(
            name="callback_test",
            on_enter=on_enter,
        )

        state.enter(mock_context)

        on_enter.assert_called_once_with(state, mock_context)

    def test_on_exit_called_when_leaving_state(self, mock_context: GraphContext) -> None:
        """on_exit callback is called when leaving the state."""
        on_exit = Mock()
        state = AnimationState(
            name="callback_test",
            on_exit=on_exit,
        )

        state.exit(mock_context)

        on_exit.assert_called_once_with(state, mock_context)

    def test_on_update_called_each_frame(self, mock_context: GraphContext) -> None:
        """on_update callback is called on each update."""
        on_update = Mock()
        state = AnimationState(
            name="callback_test",
            on_update=on_update,
        )

        state.update(0.016, mock_context)
        state.update(0.016, mock_context)

        assert on_update.call_count == 2

    def test_on_update_receives_correct_args(self, mock_context: GraphContext) -> None:
        """on_update receives state, context, and dt."""
        on_update = Mock()
        state = AnimationState(
            name="callback_test",
            on_update=on_update,
        )

        state.update(0.033, mock_context)

        on_update.assert_called_with(state, mock_context, 0.033)

    def test_no_callback_if_none(self, mock_context: GraphContext) -> None:
        """No error when callbacks are None."""
        state = AnimationState(
            name="test",
            on_enter=None,
            on_exit=None,
            on_update=None,
        )

        # Should not raise
        state.enter(mock_context)
        state.update(0.016, mock_context)
        state.exit(mock_context)

    def test_on_update_not_called_without_context(self) -> None:
        """on_update requires context to be called."""
        on_update = Mock()
        state = AnimationState(
            name="callback_test",
            on_update=on_update,
        )

        state.update(0.016, None)  # No context

        # on_update should not be called (context check in implementation)
        on_update.assert_not_called()

    def test_enter_resets_state(self, mock_context: GraphContext) -> None:
        """enter() resets the state's playback position."""
        clip = MockAnimationClip(duration=2.0)
        state = AnimationState(
            name="test",
            clip=clip,
        )
        state.current_time = 1.5
        state._normalized_time = 0.75
        state._playback_direction = -1
        state._animation_finished = True

        state.enter(mock_context)

        assert state.current_time == 0.0
        assert state._normalized_time == 0.0
        assert state._playback_direction == 1
        assert state._animation_finished is False


# =============================================================================
# TEST: TIME TRACKING
# =============================================================================


class TestTimeTracking:
    """Test current_time advances with dt * speed."""

    def test_time_advances_with_dt(self) -> None:
        """current_time advances by dt each update."""
        clip = MockAnimationClip(duration=10.0)
        state = AnimationState(
            name="time_test",
            clip=clip,
            speed=1.0,
        )

        state.update(0.1, None)
        assert pytest.approx(state.current_time, abs=0.001) == 0.1

        state.update(0.2, None)
        assert pytest.approx(state.current_time, abs=0.001) == 0.3

    def test_time_advances_with_speed(self) -> None:
        """current_time advances by dt * speed."""
        clip = MockAnimationClip(duration=10.0)
        state = AnimationState(
            name="time_test",
            clip=clip,
            speed=3.0,
        )

        state.update(0.1, None)

        assert pytest.approx(state.current_time, abs=0.001) == 0.3

    def test_normalized_time_updates(self) -> None:
        """normalized_time updates correctly with current_time."""
        clip = MockAnimationClip(duration=4.0)
        state = AnimationState(
            name="time_test",
            clip=clip,
        )

        state.update(1.0, None)

        assert pytest.approx(state.normalized_time, abs=0.001) == 0.25

    def test_time_in_state_property(self) -> None:
        """time_in_state is an alias for current_time."""
        state = AnimationState(name="test")
        state.current_time = 1.234

        assert state.time_in_state == 1.234

    def test_normalized_time_setter(self) -> None:
        """Setting normalized_time updates current_time."""
        clip = MockAnimationClip(duration=4.0)
        state = AnimationState(
            name="time_test",
            clip=clip,
        )

        state.normalized_time = 0.5

        assert state.current_time == 2.0
        assert state._normalized_time == 0.5


# =============================================================================
# TEST: RESET
# =============================================================================


class TestReset:
    """Test reset() resets playback to beginning."""

    def test_reset_sets_current_time_to_zero(self) -> None:
        """reset() sets current_time to 0."""
        state = AnimationState(name="reset_test")
        state.current_time = 5.0

        state.reset()

        assert state.current_time == 0.0

    def test_reset_sets_normalized_time_to_zero(self) -> None:
        """reset() sets normalized_time to 0."""
        state = AnimationState(name="reset_test")
        state._normalized_time = 0.75

        state.reset()

        assert state._normalized_time == 0.0

    def test_reset_sets_playback_direction_forward(self) -> None:
        """reset() sets playback direction to forward (1)."""
        state = AnimationState(name="reset_test")
        state._playback_direction = -1

        state.reset()

        assert state._playback_direction == 1

    def test_reset_clears_animation_finished(self) -> None:
        """reset() clears the animation_finished flag."""
        clip = MockAnimationClip(duration=1.0)
        state = AnimationState(
            name="reset_test",
            clip=clip,
            motion_mode=MotionMode.ONCE,
        )
        state._animation_finished = True

        state.reset()

        assert state._animation_finished is False
        assert state.is_finished is False

    def test_reset_allows_replay_of_once_animation(self) -> None:
        """After reset, a ONCE animation can play again."""
        clip = MockAnimationClip(duration=1.0)
        state = AnimationState(
            name="reset_test",
            clip=clip,
            motion_mode=MotionMode.ONCE,
        )

        # Play to end
        state.update(2.0, None)
        assert state.is_finished is True

        # Reset
        state.reset()

        # Should be able to play again
        assert state.is_finished is False
        state.update(0.5, None)
        assert pytest.approx(state.current_time, abs=0.01) == 0.5


# =============================================================================
# TEST: LEGACY COMPATIBILITY
# =============================================================================


class TestLegacyCompatibility:
    """Test legacy property aliases."""

    def test_loop_property_true_for_loop_mode(self) -> None:
        """loop property returns True for MotionMode.LOOP."""
        state = AnimationState(name="test", motion_mode=MotionMode.LOOP)
        assert state.loop is True

    def test_loop_property_false_for_other_modes(self) -> None:
        """loop property returns False for non-LOOP modes."""
        state_once = AnimationState(name="test", motion_mode=MotionMode.ONCE)
        state_pingpong = AnimationState(name="test", motion_mode=MotionMode.PING_PONG)

        assert state_once.loop is False
        assert state_pingpong.loop is False

    def test_loop_setter_true(self) -> None:
        """Setting loop=True sets motion_mode to LOOP."""
        state = AnimationState(name="test", motion_mode=MotionMode.ONCE)

        state.loop = True

        assert state.motion_mode == MotionMode.LOOP

    def test_loop_setter_false(self) -> None:
        """Setting loop=False sets motion_mode to ONCE."""
        state = AnimationState(name="test", motion_mode=MotionMode.LOOP)

        state.loop = False

        assert state.motion_mode == MotionMode.ONCE


# =============================================================================
# TEST: DURATION HANDLING
# =============================================================================


class TestDurationHandling:
    """Test _get_duration internal method."""

    def test_duration_from_clip(self) -> None:
        """Duration is read from clip if available."""
        clip = MockAnimationClip(duration=3.5)
        state = AnimationState(name="test", clip=clip)

        duration = state._get_duration()

        assert duration == 3.5

    def test_duration_default_when_no_clip(self) -> None:
        """Duration defaults to 1.0 when no clip."""
        state = AnimationState(name="test")

        duration = state._get_duration()

        assert duration == 1.0

    def test_duration_default_when_clip_has_no_duration(self) -> None:
        """Duration defaults to 1.0 if clip lacks duration attribute."""
        clip = object()  # No duration attribute
        state = AnimationState(name="test", clip=clip)

        duration = state._get_duration()

        assert duration == 1.0


# =============================================================================
# TEST: EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_duration_clip(self) -> None:
        """Handle zero-duration clip gracefully."""
        clip = MockAnimationClip(duration=0.0)
        state = AnimationState(
            name="zero_dur",
            clip=clip,
            motion_mode=MotionMode.LOOP,
        )

        # Should not crash
        state.update(0.1, None)

        # normalized_time should be 0 to avoid division by zero
        assert state._normalized_time == 0.0

    def test_very_small_dt(self) -> None:
        """Handle very small delta time values."""
        clip = MockAnimationClip(duration=1.0)
        state = AnimationState(name="test", clip=clip)

        for _ in range(1000):
            state.update(0.00001, None)

        # Should accumulate correctly
        assert pytest.approx(state.current_time, abs=0.001) == 0.01

    def test_very_large_dt(self) -> None:
        """Handle very large delta time values."""
        clip = MockAnimationClip(duration=1.0)
        state = AnimationState(
            name="test",
            clip=clip,
            motion_mode=MotionMode.LOOP,
        )

        state.update(1000.0, None)

        # Should wrap correctly
        assert 0.0 <= state.current_time <= 1.0

    def test_negative_dt(self) -> None:
        """Handle negative delta time (unusual but valid)."""
        clip = MockAnimationClip(duration=10.0)
        state = AnimationState(
            name="test",
            clip=clip,
            motion_mode=MotionMode.LOOP,
        )
        state.current_time = 5.0

        state.update(-1.0, None)

        # Should go backward
        assert pytest.approx(state.current_time, abs=0.01) == 4.0

    def test_multiple_animation_sources_clip_priority(self, mock_context: GraphContext) -> None:
        """When multiple sources exist, clip takes priority."""
        clip = MockAnimationClip(duration=2.0)
        graph = Mock()
        node = Mock()

        state = AnimationState(
            name="test",
            clip=clip,
            graph=graph,
            animation_node=node,
        )

        state.evaluate(mock_context)

        # Only clip should be sampled
        assert len(clip._sample_calls) > 0
        graph.evaluate.assert_not_called()
        node.evaluate.assert_not_called()

    def test_consecutive_enters(self, mock_context: GraphContext) -> None:
        """Multiple enter() calls reset state each time."""
        on_enter = Mock()
        state = AnimationState(
            name="test",
            on_enter=on_enter,
        )

        state.enter(mock_context)
        state.current_time = 5.0

        state.enter(mock_context)

        assert state.current_time == 0.0
        assert on_enter.call_count == 2


# =============================================================================
# TEST: IS_FINISHED PROPERTY
# =============================================================================


class TestIsFinishedProperty:
    """Test is_finished property behavior."""

    def test_is_finished_initially_false(self) -> None:
        """is_finished is False initially."""
        state = AnimationState(name="test")
        assert state.is_finished is False

    def test_is_finished_true_for_completed_once(self) -> None:
        """is_finished becomes True for completed ONCE animation."""
        clip = MockAnimationClip(duration=1.0)
        state = AnimationState(
            name="test",
            clip=clip,
            motion_mode=MotionMode.ONCE,
        )

        state.update(1.5, None)

        assert state.is_finished is True

    def test_is_finished_false_for_loop(self) -> None:
        """is_finished stays False for LOOP animation."""
        clip = MockAnimationClip(duration=1.0)
        state = AnimationState(
            name="test",
            clip=clip,
            motion_mode=MotionMode.LOOP,
        )

        for _ in range(10):
            state.update(0.5, None)

        assert state.is_finished is False

    def test_is_finished_reflects_internal_flag(self) -> None:
        """is_finished property reflects _animation_finished."""
        state = AnimationState(name="test")

        state._animation_finished = True
        assert state.is_finished is True

        state._animation_finished = False
        assert state.is_finished is False


# =============================================================================
# RUN TESTS
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

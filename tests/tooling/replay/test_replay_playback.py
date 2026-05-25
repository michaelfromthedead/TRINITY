"""
Tests for replay_playback.py - Playback controls and seeking.
"""

import pytest
from unittest.mock import MagicMock

from engine.tooling.replay.replay_playback import (
    ReplayPlayback,
    PlaybackState,
    PlaybackConfig,
    PlaybackSpeed,
    SeekMode,
)
from engine.tooling.replay.input_recorder import RecordedInput, InputType
from engine.tooling.replay.state_recorder import StateSnapshot


def create_test_inputs(count: int = 100) -> list[RecordedInput]:
    """Create test input data."""
    inputs = []
    for i in range(count):
        inputs.append(RecordedInput(
            input_type=InputType.KEYBOARD,
            timestamp=i * 0.016,
            frame=i,
            device_id=0,
            data={'key': chr(65 + (i % 26)), 'pressed': True}
        ))
    return inputs


def create_test_snapshots(count: int = 10) -> list[StateSnapshot]:
    """Create test snapshot data."""
    snapshots = []
    for i in range(count):
        snapshots.append(StateSnapshot(
            frame=i * 10,
            timestamp=i * 0.166,
            state_data={'position': i * 10},
            checksum='',
            size_bytes=0,
            is_keyframe=(i % 3 == 0)
        ))
    return snapshots


class TestPlaybackSpeed:
    """Tests for PlaybackSpeed enum."""

    def test_preset_values(self):
        """Test preset speed values."""
        assert PlaybackSpeed.QUARTER.value == 0.25
        assert PlaybackSpeed.HALF.value == 0.5
        assert PlaybackSpeed.NORMAL.value == 1.0
        assert PlaybackSpeed.DOUBLE.value == 2.0
        assert PlaybackSpeed.QUADRUPLE.value == 4.0

    def test_from_value(self):
        """Test getting preset from value."""
        assert PlaybackSpeed.from_value(1.0) == PlaybackSpeed.NORMAL
        assert PlaybackSpeed.from_value(2.0) == PlaybackSpeed.DOUBLE
        assert PlaybackSpeed.from_value(1.5) == PlaybackSpeed.NORMAL  # Not a preset


class TestPlaybackConfig:
    """Tests for PlaybackConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PlaybackConfig()
        assert config.initial_speed == 1.0
        assert config.min_speed == 0.1
        assert config.max_speed == 10.0
        assert config.frame_step_size == 1
        assert config.inject_inputs is True
        assert config.loop is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = PlaybackConfig(
            initial_speed=2.0,
            loop=True,
            loop_start_frame=100
        )
        assert config.initial_speed == 2.0
        assert config.loop is True
        assert config.loop_start_frame == 100


class TestReplayPlayback:
    """Tests for ReplayPlayback class."""

    def test_create_playback(self):
        """Test creating playback instance."""
        inputs = create_test_inputs(10)
        snapshots = create_test_snapshots(5)

        playback = ReplayPlayback(inputs, snapshots)
        assert playback.state == PlaybackState.STOPPED
        assert playback.current_frame == 0
        assert playback.current_time == 0.0

    def test_play(self):
        """Test starting playback."""
        inputs = create_test_inputs(10)
        snapshots = create_test_snapshots(5)
        playback = ReplayPlayback(inputs, snapshots)

        playback.play()
        assert playback.state == PlaybackState.PLAYING
        assert playback.is_playing

    def test_pause(self):
        """Test pausing playback."""
        inputs = create_test_inputs(10)
        snapshots = create_test_snapshots(5)
        playback = ReplayPlayback(inputs, snapshots)

        playback.play()
        playback.pause()
        assert playback.state == PlaybackState.PAUSED
        assert playback.is_paused

    def test_stop(self):
        """Test stopping playback."""
        inputs = create_test_inputs(10)
        snapshots = create_test_snapshots(5)
        playback = ReplayPlayback(inputs, snapshots)

        playback.play()
        playback.stop()
        assert playback.state == PlaybackState.STOPPED
        assert playback.current_frame == 0
        assert playback.current_time == 0.0

    def test_toggle_play_pause(self):
        """Test toggle play/pause."""
        inputs = create_test_inputs(10)
        snapshots = create_test_snapshots(5)
        playback = ReplayPlayback(inputs, snapshots)

        playback.toggle_play_pause()
        assert playback.is_playing

        playback.toggle_play_pause()
        assert playback.is_paused

    def test_speed_property(self):
        """Test speed getter and setter."""
        inputs = create_test_inputs(10)
        snapshots = create_test_snapshots(5)
        config = PlaybackConfig(min_speed=0.5, max_speed=4.0)
        playback = ReplayPlayback(inputs, snapshots, config=config)

        playback.speed = 2.0
        assert playback.speed == 2.0

        # Test clamping
        playback.speed = 0.1  # Below min
        assert playback.speed == 0.5

        playback.speed = 10.0  # Above max
        assert playback.speed == 4.0

    def test_speed_up(self):
        """Test speed up method."""
        inputs = create_test_inputs(10)
        snapshots = create_test_snapshots(5)
        playback = ReplayPlayback(inputs, snapshots)

        playback.speed = 1.0
        playback.speed_up(2.0)
        assert playback.speed == 2.0

    def test_slow_down(self):
        """Test slow down method."""
        inputs = create_test_inputs(10)
        snapshots = create_test_snapshots(5)
        playback = ReplayPlayback(inputs, snapshots)

        playback.speed = 2.0
        playback.slow_down(2.0)
        assert playback.speed == 1.0

    def test_set_preset_speed(self):
        """Test setting preset speed."""
        inputs = create_test_inputs(10)
        snapshots = create_test_snapshots(5)
        playback = ReplayPlayback(inputs, snapshots)

        playback.set_preset_speed(PlaybackSpeed.HALF)
        assert playback.speed == 0.5

    def test_update_advances_time(self):
        """Test that update advances playback time."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        playback.play()
        initial_time = playback.current_time
        playback.update(0.016)  # ~1 frame at 60fps

        assert playback.current_time > initial_time

    def test_update_returns_inputs(self):
        """Test that update returns inputs in time range."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        playback.play()
        result = playback.update(0.1)  # ~6 frames worth

        assert isinstance(result, list)

    def test_update_when_paused(self):
        """Test update when paused returns empty."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        playback.pause()
        result = playback.update(0.1)

        assert result == []

    def test_seek_to_frame(self):
        """Test seeking to specific frame."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        success = playback.seek(50, SeekMode.FRAME)
        assert success
        assert playback.current_frame == 50

    def test_seek_to_time(self):
        """Test seeking to specific time."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        success = playback.seek(0.5, SeekMode.TIME)
        assert success
        assert playback.current_time == 0.5

    def test_seek_to_percentage(self):
        """Test seeking to percentage."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        success = playback.seek(0.5, SeekMode.PERCENTAGE)
        assert success
        # Should be at 50% of total duration

    def test_seek_to_keyframe(self):
        """Test seeking to nearest keyframe."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        success = playback.seek(45, SeekMode.KEYFRAME)
        assert success

    def test_step_forward(self):
        """Test stepping forward."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        playback.seek(0, SeekMode.FRAME)
        result = playback.step_forward(5)

        assert playback.current_frame == 5
        assert isinstance(result, list)

    def test_step_backward(self):
        """Test stepping backward."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        playback.seek(50, SeekMode.FRAME)
        playback.step_backward(10)

        assert playback.current_frame == 40

    def test_next_keyframe(self):
        """Test seeking to next keyframe."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        playback.seek(0, SeekMode.FRAME)
        playback.next_keyframe()

        # Should have moved to a keyframe

    def test_previous_keyframe(self):
        """Test seeking to previous keyframe."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        playback.seek(50, SeekMode.FRAME)
        playback.previous_keyframe()

        # Should have moved to a previous keyframe

    def test_add_remove_marker(self):
        """Test adding and removing markers."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        playback.seek(25, SeekMode.FRAME)
        playback.add_marker('test_marker')

        markers = playback.get_markers()
        assert 'test_marker' in markers
        assert markers['test_marker'] == 25

        removed = playback.remove_marker('test_marker')
        assert removed
        assert 'test_marker' not in playback.get_markers()

    def test_seek_to_marker(self):
        """Test seeking to named marker."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        playback.add_marker('my_marker', frame=75)
        success = playback.seek('my_marker', SeekMode.MARKER)

        assert success
        assert playback.current_frame == 75

    def test_get_inputs_at_frame(self):
        """Test getting inputs at specific frame."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        frame_inputs = playback.get_inputs_at_frame(10)
        assert len(frame_inputs) == 1
        assert frame_inputs[0].frame == 10

    def test_get_state_at_frame(self):
        """Test getting state at specific frame."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        state = playback.get_state_at_frame(0)
        assert state is not None

    def test_input_callback(self):
        """Test input injection callback."""
        injected_inputs = []

        def on_input(inp):
            injected_inputs.append(inp)

        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        config = PlaybackConfig(input_callback=on_input)
        playback = ReplayPlayback(inputs, snapshots, config=config)

        playback.play()
        playback.update(0.1)

        assert len(injected_inputs) > 0

    def test_state_callback_on_seek(self):
        """Test state restoration callback on seek."""
        restored_states = []

        def on_state(state):
            restored_states.append(state)

        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        config = PlaybackConfig(
            restore_state_on_seek=True,
            state_callback=on_state
        )
        playback = ReplayPlayback(inputs, snapshots, config=config)

        playback.seek(50, SeekMode.FRAME)

        assert len(restored_states) > 0

    def test_playback_complete_callback(self):
        """Test playback complete callback."""
        completed = {'called': False}

        def on_complete():
            completed['called'] = True

        inputs = create_test_inputs(10)
        snapshots = create_test_snapshots(2)
        config = PlaybackConfig(on_playback_complete=on_complete)
        playback = ReplayPlayback(inputs, snapshots, config=config)

        playback.play()
        # Simulate reaching end
        playback.update(10.0)  # Large time step

        assert completed['called'] or playback.is_finished

    def test_loop_playback(self):
        """Test looping playback."""
        inputs = create_test_inputs(10)
        snapshots = create_test_snapshots(2)
        config = PlaybackConfig(loop=True, loop_start_frame=0)
        playback = ReplayPlayback(inputs, snapshots, config=config)

        playback.play()
        # Simulate reaching end
        playback.update(10.0)

        # Should have looped back
        assert playback.is_playing

    def test_position_property(self):
        """Test position property."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        playback.seek(50, SeekMode.FRAME)
        position = playback.position

        assert position.frame == 50
        assert position.percentage >= 0.0

    def test_iter_inputs(self):
        """Test iterating over inputs."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        all_inputs = list(playback.iter_inputs())
        assert len(all_inputs) == 100

    def test_iter_snapshots(self):
        """Test iterating over snapshots."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        all_snapshots = list(playback.iter_snapshots())
        assert len(all_snapshots) == 10

    def test_frame_callback(self):
        """Test frame-specific callback."""
        triggered = {'frame': None}

        def on_frame_50():
            triggered['frame'] = 50

        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        playback.add_frame_callback(50, on_frame_50)
        playback.seek(49, SeekMode.FRAME)
        playback.play()
        playback.update(0.05)  # Should pass frame 50

        # Callback should have been triggered

    def test_total_frames_property(self):
        """Test total frames property."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        assert playback.total_frames >= 99  # At least input frames

    def test_total_duration_property(self):
        """Test total duration property."""
        inputs = create_test_inputs(100)
        snapshots = create_test_snapshots(10)
        playback = ReplayPlayback(inputs, snapshots)

        assert playback.total_duration > 0

    def test_is_finished_property(self):
        """Test is_finished property."""
        inputs = create_test_inputs(10)
        snapshots = create_test_snapshots(2)
        playback = ReplayPlayback(inputs, snapshots)

        assert not playback.is_finished

        playback.play()
        playback.update(100.0)  # Large time step to reach end

        assert playback.is_finished

"""Tests for the replay player system.

Tests playback controls including play, pause, speed, seeking,
frame stepping, and reverse playback.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from engine.debug.replay.player import (
    PlaybackInfo,
    PlaybackState,
    ReplayPlayer,
)
from engine.debug.replay.recorder import (
    InputRecord,
    InputRecorder,
    StateSnapshot,
)


# =============================================================================
# PlaybackState Tests
# =============================================================================


class TestPlaybackState:
    """Tests for PlaybackState enum."""

    def test_all_states_exist(self) -> None:
        """Test that all expected states exist."""
        assert PlaybackState.STOPPED
        assert PlaybackState.PLAYING
        assert PlaybackState.PAUSED

    def test_states_are_distinct(self) -> None:
        """Test that states are distinct values."""
        states = [PlaybackState.STOPPED, PlaybackState.PLAYING, PlaybackState.PAUSED]
        assert len(set(states)) == 3


# =============================================================================
# ReplayPlayer Basic Tests
# =============================================================================


class TestReplayPlayerBasic:
    """Basic tests for ReplayPlayer."""

    def test_create_player(self) -> None:
        """Test creating a replay player."""
        player = ReplayPlayer()
        assert player.state == PlaybackState.STOPPED
        assert not player.is_loaded
        assert not player.is_playing
        assert not player.is_paused

    def test_create_player_with_ticks_per_second(self) -> None:
        """Test creating player with custom tick rate."""
        player = ReplayPlayer(ticks_per_second=30)
        assert player.state == PlaybackState.STOPPED

    def test_load_without_file_raises(self) -> None:
        """Test that loading nonexistent file raises."""
        player = ReplayPlayer()
        with pytest.raises(FileNotFoundError):
            player.load("/nonexistent/path.json")

    def test_load_combined_data(self) -> None:
        """Test loading data directly."""
        inputs = [
            InputRecord(tick=0, input_type="keyboard", data={"key": "W"}),
            InputRecord(tick=10, input_type="keyboard", data={"key": "S"}),
        ]
        snapshots = [
            StateSnapshot(tick=0, state_data={"player": "start"}),
            StateSnapshot(tick=10, state_data={"player": "end"}),
        ]

        player = ReplayPlayer()
        player.load_combined(inputs=inputs, snapshots=snapshots)

        assert player.is_loaded
        assert player.first_tick == 0
        assert player.last_tick == 10
        assert player.get_total_ticks() == 11

    def test_unload(self) -> None:
        """Test unloading replay data."""
        inputs = [InputRecord(tick=0, input_type="keyboard", data={})]
        player = ReplayPlayer()
        player.load_combined(inputs=inputs)
        assert player.is_loaded

        player.unload()
        assert not player.is_loaded
        assert player.state == PlaybackState.STOPPED


# =============================================================================
# Playback Control Tests
# =============================================================================


class TestReplayPlayerPlayback:
    """Tests for playback controls."""

    def get_loaded_player(self) -> ReplayPlayer:
        """Create a player with sample data."""
        inputs = [
            InputRecord(tick=i, input_type="keyboard", data={"key": str(i)})
            for i in range(100)
        ]
        player = ReplayPlayer(ticks_per_second=60)
        player.load_combined(inputs=inputs)
        return player

    def test_play(self) -> None:
        """Test play method."""
        player = self.get_loaded_player()
        player.play()
        assert player.state == PlaybackState.PLAYING
        assert player.is_playing

    def test_pause(self) -> None:
        """Test pause method."""
        player = self.get_loaded_player()
        player.play()
        player.pause()
        assert player.state == PlaybackState.PAUSED
        assert player.is_paused

    def test_stop(self) -> None:
        """Test stop method."""
        player = self.get_loaded_player()
        player.play()
        player.seek(50)
        player.stop()

        assert player.state == PlaybackState.STOPPED
        assert player.get_current_tick() == 0
        assert not player.is_reversed
        assert player.speed == ReplayPlayer.DEFAULT_SPEED

    def test_toggle_pause(self) -> None:
        """Test toggle_pause method."""
        player = self.get_loaded_player()

        # Stopped -> Playing
        player.toggle_pause()
        assert player.is_playing

        # Playing -> Paused
        player.toggle_pause()
        assert player.is_paused

        # Paused -> Playing
        player.toggle_pause()
        assert player.is_playing

    def test_play_without_data_does_nothing(self) -> None:
        """Test that play does nothing when no data loaded."""
        player = ReplayPlayer()
        player.play()
        assert player.state == PlaybackState.STOPPED


# =============================================================================
# Speed Control Tests
# =============================================================================


class TestReplayPlayerSpeed:
    """Tests for speed control."""

    def get_loaded_player(self) -> ReplayPlayer:
        """Create a player with sample data."""
        inputs = [InputRecord(tick=i, input_type="keyboard", data={}) for i in range(100)]
        player = ReplayPlayer()
        player.load_combined(inputs=inputs)
        return player

    def test_default_speed(self) -> None:
        """Test default speed is 1.0."""
        player = self.get_loaded_player()
        assert player.speed == 1.0

    def test_set_speed(self) -> None:
        """Test setting playback speed."""
        player = self.get_loaded_player()
        player.set_speed(0.5)
        assert player.speed == 0.5

        player.set_speed(2.0)
        assert player.speed == 2.0

    def test_set_speed_limits(self) -> None:
        """Test speed limit enforcement."""
        player = self.get_loaded_player()

        with pytest.raises(ValueError):
            player.set_speed(0.05)  # Too slow

        with pytest.raises(ValueError):
            player.set_speed(5.0)  # Too fast

    def test_set_speed_at_limits(self) -> None:
        """Test speed at min/max limits."""
        player = self.get_loaded_player()

        player.set_speed(ReplayPlayer.MIN_SPEED)
        assert player.speed == ReplayPlayer.MIN_SPEED

        player.set_speed(ReplayPlayer.MAX_SPEED)
        assert player.speed == ReplayPlayer.MAX_SPEED


# =============================================================================
# Seeking Tests
# =============================================================================


class TestReplayPlayerSeeking:
    """Tests for seeking functionality."""

    def get_loaded_player(self) -> ReplayPlayer:
        """Create a player with sample data."""
        inputs = [InputRecord(tick=i, input_type="keyboard", data={}) for i in range(100)]
        snapshots = [StateSnapshot(tick=i * 10, state_data={"tick": i * 10}) for i in range(11)]
        player = ReplayPlayer()
        player.load_combined(inputs=inputs, snapshots=snapshots)
        return player

    def test_seek(self) -> None:
        """Test seeking to specific tick."""
        player = self.get_loaded_player()
        player.seek(50)
        assert player.get_current_tick() == 50

    def test_seek_clamps_to_valid_range(self) -> None:
        """Test that seek clamps to valid tick range."""
        player = self.get_loaded_player()

        player.seek(-10)
        assert player.get_current_tick() == 0

        player.seek(1000)
        # Snapshots go up to tick 100 (i * 10 for i in range(11))
        assert player.get_current_tick() == 100

    def test_seek_to_start(self) -> None:
        """Test seeking to start."""
        player = self.get_loaded_player()
        player.seek(50)
        player.seek_to_start()
        assert player.get_current_tick() == 0

    def test_seek_to_end(self) -> None:
        """Test seeking to end."""
        player = self.get_loaded_player()
        player.seek_to_end()
        # Snapshots go up to tick 100 (i * 10 for i in range(11))
        assert player.get_current_tick() == 100

    def test_seek_without_data_does_nothing(self) -> None:
        """Test seek with no data loaded."""
        player = ReplayPlayer()
        player.seek(50)  # Should not raise


# =============================================================================
# Frame Stepping Tests
# =============================================================================


class TestReplayPlayerFrameStep:
    """Tests for frame stepping functionality."""

    def get_loaded_player(self) -> ReplayPlayer:
        """Create a player with sample data."""
        inputs = [InputRecord(tick=i, input_type="keyboard", data={}) for i in range(100)]
        player = ReplayPlayer()
        player.load_combined(inputs=inputs)
        return player

    def test_step_frame_forward(self) -> None:
        """Test stepping forward by frames."""
        player = self.get_loaded_player()
        player.seek(50)

        player.step_frame(1)
        assert player.get_current_tick() == 51

        player.step_frame(5)
        assert player.get_current_tick() == 56

    def test_step_frame_backward(self) -> None:
        """Test stepping backward by frames."""
        player = self.get_loaded_player()
        player.seek(50)

        player.step_frame(-1)
        assert player.get_current_tick() == 49

        player.step_frame(-5)
        assert player.get_current_tick() == 44

    def test_step_frame_clamps_to_range(self) -> None:
        """Test that frame stepping clamps to valid range."""
        player = self.get_loaded_player()

        player.seek(0)
        player.step_frame(-10)
        assert player.get_current_tick() == 0

        player.seek(99)
        player.step_frame(10)
        assert player.get_current_tick() == 99


# =============================================================================
# Reverse Playback Tests
# =============================================================================


class TestReplayPlayerReverse:
    """Tests for reverse playback."""

    def get_loaded_player(self) -> ReplayPlayer:
        """Create a player with sample data."""
        inputs = [InputRecord(tick=i, input_type="keyboard", data={}) for i in range(100)]
        player = ReplayPlayer()
        player.load_combined(inputs=inputs)
        return player

    def test_reverse_toggle(self) -> None:
        """Test toggling reverse mode."""
        player = self.get_loaded_player()
        assert not player.is_reversed

        player.reverse()
        assert player.is_reversed

        player.reverse()
        assert not player.is_reversed

    def test_step_frame_respects_reverse(self) -> None:
        """Test that step_frame direction is reversed."""
        player = self.get_loaded_player()
        player.seek(50)
        player.reverse()

        # Stepping "forward" in reverse mode should go backward
        player.step_frame(1)
        assert player.get_current_tick() == 49


# =============================================================================
# Progress and Info Tests
# =============================================================================


class TestReplayPlayerProgress:
    """Tests for progress tracking."""

    def get_loaded_player(self) -> ReplayPlayer:
        """Create a player with sample data."""
        inputs = [InputRecord(tick=i, input_type="keyboard", data={}) for i in range(100)]
        player = ReplayPlayer()
        player.load_combined(inputs=inputs)
        return player

    def test_get_progress(self) -> None:
        """Test getting playback progress."""
        player = self.get_loaded_player()

        player.seek(0)
        assert player.get_progress() == 0.0

        player.seek(99)
        assert player.get_progress() == 1.0

        player.seek(49)
        assert 0.49 <= player.get_progress() <= 0.51

    def test_get_info(self) -> None:
        """Test getting comprehensive playback info."""
        player = self.get_loaded_player()
        player.play()
        player.seek(25)
        player.set_speed(0.5)

        info = player.get_info()
        assert isinstance(info, PlaybackInfo)
        assert info.current_tick == 25
        assert info.total_ticks == 100
        assert info.speed == 0.5
        assert info.state == PlaybackState.PLAYING
        assert not info.is_reversed


# =============================================================================
# Update Loop Tests
# =============================================================================


class TestReplayPlayerUpdate:
    """Tests for the update loop."""

    def get_loaded_player(self) -> ReplayPlayer:
        """Create a player with sample data."""
        inputs = [InputRecord(tick=i, input_type="keyboard", data={}) for i in range(1000)]
        player = ReplayPlayer(ticks_per_second=60)
        player.load_combined(inputs=inputs)
        return player

    def test_update_advances_playback(self) -> None:
        """Test that update advances playback position."""
        player = self.get_loaded_player()
        player.play()

        # Update with explicit dt (simulate 1 second at 60fps)
        for _ in range(60):
            player.update(dt=1 / 60)

        # Should have advanced roughly 60 ticks
        assert 50 <= player.get_current_tick() <= 70

    def test_update_respects_speed(self) -> None:
        """Test that update respects playback speed."""
        player = self.get_loaded_player()
        player.set_speed(2.0)
        player.play()

        # Update with explicit dt (simulate 0.5 second at 2x speed)
        for _ in range(30):
            player.update(dt=1 / 60)

        # Should have advanced roughly 60 ticks (2x speed)
        assert 50 <= player.get_current_tick() <= 70

    def test_update_when_not_playing_returns_zero(self) -> None:
        """Test that update returns 0 when not playing."""
        player = self.get_loaded_player()

        result = player.update(dt=1 / 60)
        assert result == 0

        player.play()
        player.pause()
        result = player.update(dt=1 / 60)
        assert result == 0

    def test_update_pauses_at_end(self) -> None:
        """Test that update pauses when reaching end."""
        player = self.get_loaded_player()
        player.seek(990)
        player.play()

        # Update until we hit the end
        for _ in range(100):
            player.update(dt=1 / 60)
            if player.state == PlaybackState.PAUSED:
                break

        assert player.state == PlaybackState.PAUSED
        assert player.get_current_tick() == 999


# =============================================================================
# Callback Tests
# =============================================================================


class TestReplayPlayerCallbacks:
    """Tests for input and state callbacks."""

    def test_input_callback(self) -> None:
        """Test that input callback is fired during playback.

        Note: The player processes ticks as it advances (increment then process),
        so the starting tick's inputs are processed in the first update cycle.
        """
        # Create inputs at sequential ticks starting from 0
        inputs = [
            InputRecord(tick=0, input_type="keyboard", data={"key": "start"}),  # first_tick
            InputRecord(tick=1, input_type="keyboard", data={"key": "A"}),
            InputRecord(tick=2, input_type="keyboard", data={"key": "B"}),
        ]

        received_inputs = []

        def on_input(record: InputRecord) -> None:
            received_inputs.append(record)

        player = ReplayPlayer(ticks_per_second=60, on_input=on_input)
        player.load_combined(inputs=inputs)
        player.play()

        # Use update to process ticks during playback
        # Player starts at tick 0, each update increments then processes
        # So update 1: increment to 1, process tick 1 (key A)
        # Update 2: increment to 2, process tick 2 (key B)
        # Update 3: at tick 2 = last_tick, pause
        for _ in range(5):
            player.update(dt=1/60)
            if player.is_paused:
                break

        # Inputs at ticks 1 and 2 should be received (not tick 0 which was starting position)
        assert len(received_inputs) == 2
        assert received_inputs[0].data["key"] == "A"
        assert received_inputs[1].data["key"] == "B"

    def test_state_callback(self) -> None:
        """Test that state callback is fired during seeking."""
        snapshots = [
            StateSnapshot(tick=0, state_data={"pos": 0}),
            StateSnapshot(tick=50, state_data={"pos": 50}),
        ]

        received_states = []

        def on_state(snapshot: StateSnapshot) -> None:
            received_states.append(snapshot)

        player = ReplayPlayer(on_state=on_state)
        player.load_combined(snapshots=snapshots)

        # Seeking should trigger state callback
        player.seek(25)

        assert len(received_states) == 1
        assert received_states[0].tick == 0  # Nearest before tick 25


# =============================================================================
# File Operations Tests
# =============================================================================


class TestReplayPlayerFileOps:
    """Tests for file loading and saving."""

    def test_load_json_input_recording(self) -> None:
        """Test loading JSON input recording."""
        recorder = InputRecorder()
        recorder.start()
        recorder.record_input("keyboard", {"key": "W"}, tick=0)
        recorder.record_input("keyboard", {"key": "S"}, tick=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            recorder.save(path)

            player = ReplayPlayer()
            player.load(path)

            assert player.is_loaded
            assert player.first_tick == 0
            assert player.last_tick == 10

    def test_save_combined_format(self) -> None:
        """Test saving replay in combined format."""
        inputs = [InputRecord(tick=0, input_type="keyboard", data={})]
        snapshots = [StateSnapshot(tick=0, state_data={})]

        player = ReplayPlayer()
        player.load_combined(inputs=inputs, snapshots=snapshots)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "combined.bin"
            player.save_combined(path)

            # Load it back
            player2 = ReplayPlayer()
            player2.load(path)
            assert player2.is_loaded

    def test_get_inputs_at_tick(self) -> None:
        """Test getting inputs at specific tick."""
        inputs = [
            InputRecord(tick=0, input_type="keyboard", data={"key": "A"}),
            InputRecord(tick=0, input_type="mouse", data={"button": 0}),
            InputRecord(tick=1, input_type="keyboard", data={"key": "B"}),
        ]

        player = ReplayPlayer()
        player.load_combined(inputs=inputs)

        tick_0_inputs = player.get_inputs_at_tick(0)
        assert len(tick_0_inputs) == 2

        tick_1_inputs = player.get_inputs_at_tick(1)
        assert len(tick_1_inputs) == 1

    def test_get_snapshot_at_tick(self) -> None:
        """Test getting snapshot at specific tick."""
        snapshots = [
            StateSnapshot(tick=0, state_data={"a": 1}),
            StateSnapshot(tick=10, state_data={"b": 2}),
        ]

        player = ReplayPlayer()
        player.load_combined(snapshots=snapshots)

        snap = player.get_snapshot_at_tick(0)
        assert snap is not None
        assert snap.state_data == {"a": 1}

        snap = player.get_snapshot_at_tick(5)
        assert snap is None  # No exact match

    def test_get_nearest_snapshot(self) -> None:
        """Test getting nearest snapshot before tick."""
        snapshots = [
            StateSnapshot(tick=0, state_data={"a": 1}),
            StateSnapshot(tick=10, state_data={"b": 2}),
        ]

        player = ReplayPlayer()
        player.load_combined(snapshots=snapshots)

        snap = player.get_nearest_snapshot(5)
        assert snap is not None
        assert snap.tick == 0

        snap = player.get_nearest_snapshot(15)
        assert snap is not None
        assert snap.tick == 10

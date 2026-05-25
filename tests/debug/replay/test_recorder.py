"""Tests for the replay recorder system.

Tests input recording, state recording, and rolling buffer recording
functionality including serialization and mode-specific behavior.
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import pytest

from engine.debug.replay.recorder import (
    InputRecord,
    InputRecorder,
    RecordingMode,
    RollingRecorder,
    StateRecorder,
    StateSnapshot,
)


# =============================================================================
# InputRecord Tests
# =============================================================================


class TestInputRecord:
    """Tests for InputRecord dataclass."""

    def test_create_input_record(self) -> None:
        """Test creating an input record with all fields."""
        record = InputRecord(
            tick=100,
            input_type="keyboard",
            data={"key": "W", "action": "press"},
            timestamp=1234567890.0,
        )
        assert record.tick == 100
        assert record.input_type == "keyboard"
        assert record.data == {"key": "W", "action": "press"}
        assert record.timestamp == 1234567890.0

    def test_input_record_default_timestamp(self) -> None:
        """Test that timestamp defaults to current time."""
        record = InputRecord(tick=0, input_type="mouse", data={})
        assert record.timestamp > 0

    def test_input_record_to_dict(self) -> None:
        """Test serialization to dictionary."""
        record = InputRecord(
            tick=50,
            input_type="gamepad",
            data={"button": "A"},
            timestamp=100.0,
        )
        d = record.to_dict()
        assert d["tick"] == 50
        assert d["input_type"] == "gamepad"
        assert d["data"] == {"button": "A"}
        assert d["timestamp"] == 100.0

    def test_input_record_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        d = {
            "tick": 75,
            "input_type": "touch",
            "data": {"x": 100, "y": 200},
            "timestamp": 200.0,
        }
        record = InputRecord.from_dict(d)
        assert record.tick == 75
        assert record.input_type == "touch"
        assert record.data == {"x": 100, "y": 200}
        assert record.timestamp == 200.0


class TestStateSnapshot:
    """Tests for StateSnapshot dataclass."""

    def test_create_state_snapshot(self) -> None:
        """Test creating a state snapshot."""
        snapshot = StateSnapshot(
            tick=100,
            state_data={"player": {"x": 10, "y": 20}},
            timestamp=1000.0,
        )
        assert snapshot.tick == 100
        assert snapshot.state_data == {"player": {"x": 10, "y": 20}}
        assert snapshot.timestamp == 1000.0

    def test_state_snapshot_to_dict(self) -> None:
        """Test serialization to dictionary."""
        snapshot = StateSnapshot(
            tick=50,
            state_data={"health": 100},
            timestamp=500.0,
        )
        d = snapshot.to_dict()
        assert d["tick"] == 50
        assert d["state_data"] == {"health": 100}

    def test_state_snapshot_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        d = {
            "tick": 25,
            "state_data": {"entities": []},
            "timestamp": 250.0,
        }
        snapshot = StateSnapshot.from_dict(d)
        assert snapshot.tick == 25
        assert snapshot.state_data == {"entities": []}


# =============================================================================
# InputRecorder Tests
# =============================================================================


class TestInputRecorder:
    """Tests for InputRecorder class."""

    def test_create_recorder_continuous(self) -> None:
        """Test creating a continuous recorder."""
        recorder = InputRecorder(mode=RecordingMode.CONTINUOUS)
        assert recorder.mode == RecordingMode.CONTINUOUS
        assert not recorder.is_recording

    def test_create_recorder_triggered(self) -> None:
        """Test creating a triggered recorder."""
        recorder = InputRecorder(mode=RecordingMode.TRIGGERED)
        assert recorder.mode == RecordingMode.TRIGGERED

    def test_start_stop_recording(self) -> None:
        """Test start and stop methods."""
        recorder = InputRecorder()
        assert not recorder.is_recording

        recorder.start()
        assert recorder.is_recording

        recorder.stop()
        assert not recorder.is_recording

    def test_record_input_while_recording(self) -> None:
        """Test recording inputs while recording is active."""
        recorder = InputRecorder()
        recorder.start()

        recorder.record_input("keyboard", {"key": "W"})
        recorder.record_input("mouse", {"button": 0})

        assert recorder.total_records == 2

    def test_record_input_while_not_recording(self) -> None:
        """Test that inputs are ignored when not recording."""
        recorder = InputRecorder()

        recorder.record_input("keyboard", {"key": "W"})

        assert recorder.total_records == 0

    def test_record_input_with_tick(self) -> None:
        """Test recording with explicit tick."""
        recorder = InputRecorder()
        recorder.start()

        recorder.record_input("keyboard", {"key": "W"}, tick=100)

        records = recorder.records
        assert len(records) == 1
        assert records[0].tick == 100

    def test_advance_tick(self) -> None:
        """Test internal tick advancement."""
        recorder = InputRecorder()
        recorder.start()

        recorder.record_input("keyboard", {"key": "A"})
        recorder.advance_tick()
        recorder.record_input("keyboard", {"key": "B"})

        records = recorder.records
        assert records[0].tick == 0
        assert records[1].tick == 1

    def test_tick_provider(self) -> None:
        """Test using external tick provider."""
        tick = [0]

        def get_tick() -> int:
            return tick[0]

        recorder = InputRecorder(current_tick_provider=get_tick)
        recorder.start()

        recorder.record_input("keyboard", {"key": "A"})
        tick[0] = 50
        recorder.record_input("keyboard", {"key": "B"})

        records = recorder.records
        assert records[0].tick == 0
        assert records[1].tick == 50

    def test_get_inputs_at_tick(self) -> None:
        """Test filtering inputs by tick."""
        recorder = InputRecorder()
        recorder.start()

        recorder.record_input("keyboard", {"key": "A"}, tick=10)
        recorder.record_input("mouse", {"button": 0}, tick=10)
        recorder.record_input("keyboard", {"key": "B"}, tick=20)

        inputs_at_10 = recorder.get_inputs_at_tick(10)
        assert len(inputs_at_10) == 2

        inputs_at_20 = recorder.get_inputs_at_tick(20)
        assert len(inputs_at_20) == 1

    def test_get_inputs_in_range(self) -> None:
        """Test filtering inputs by tick range."""
        recorder = InputRecorder()
        recorder.start()

        recorder.record_input("keyboard", {"key": "A"}, tick=5)
        recorder.record_input("keyboard", {"key": "B"}, tick=10)
        recorder.record_input("keyboard", {"key": "C"}, tick=15)
        recorder.record_input("keyboard", {"key": "D"}, tick=20)

        inputs = recorder.get_inputs_in_range(8, 18)
        assert len(inputs) == 2
        assert inputs[0].data["key"] == "B"
        assert inputs[1].data["key"] == "C"

    def test_first_last_tick(self) -> None:
        """Test first_tick and last_tick properties."""
        recorder = InputRecorder()
        assert recorder.first_tick is None
        assert recorder.last_tick is None

        recorder.start()
        recorder.record_input("keyboard", {"key": "A"}, tick=10)
        recorder.record_input("keyboard", {"key": "B"}, tick=50)

        assert recorder.first_tick == 10
        assert recorder.last_tick == 50

    def test_save_load_json(self) -> None:
        """Test saving and loading input recording."""
        recorder = InputRecorder()
        recorder.start()
        recorder.record_input("keyboard", {"key": "W"}, tick=0)
        recorder.record_input("mouse", {"x": 100, "y": 200}, tick=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            recorder.save(path)

            # Load into new recorder
            loaded = InputRecorder()
            loaded.load(path)

            assert loaded.total_records == 2
            assert loaded.records[0].input_type == "keyboard"
            assert loaded.records[1].input_type == "mouse"

    def test_clear(self) -> None:
        """Test clearing recorded data."""
        recorder = InputRecorder()
        recorder.start()
        recorder.record_input("keyboard", {"key": "W"})
        assert recorder.total_records == 1

        recorder.clear()
        assert recorder.total_records == 0

    def test_continuous_mode_clears_on_start(self) -> None:
        """Test that continuous mode clears data on start."""
        recorder = InputRecorder(mode=RecordingMode.CONTINUOUS)
        recorder.start()
        recorder.record_input("keyboard", {"key": "W"})
        recorder.stop()

        # Starting again should clear
        recorder.start()
        assert recorder.total_records == 0


# =============================================================================
# StateRecorder Tests
# =============================================================================


class TestStateRecorder:
    """Tests for StateRecorder class."""

    def test_create_recorder(self) -> None:
        """Test creating a state recorder."""
        recorder = StateRecorder()
        assert recorder.mode == RecordingMode.CONTINUOUS
        assert not recorder.is_recording

    def test_start_with_interval(self) -> None:
        """Test starting with snapshot interval."""
        recorder = StateRecorder()
        recorder.start(interval_ticks=10)
        assert recorder.is_recording

    def test_start_invalid_interval(self) -> None:
        """Test that invalid interval raises error."""
        recorder = StateRecorder()
        with pytest.raises(ValueError):
            recorder.start(interval_ticks=0)

    def test_take_snapshot_respects_interval(self) -> None:
        """Test that snapshots respect interval setting."""
        recorder = StateRecorder()
        recorder.start(interval_ticks=5)

        # First snapshot should always be taken
        assert recorder.take_snapshot({"tick": 0}) is True
        assert recorder.total_snapshots == 1

        # Next 4 should be skipped
        for i in range(1, 5):
            assert recorder.take_snapshot({"tick": i}) is False
        assert recorder.total_snapshots == 1

        # Tick 5 should be taken
        assert recorder.take_snapshot({"tick": 5}) is True
        assert recorder.total_snapshots == 2

    def test_take_snapshot_force(self) -> None:
        """Test forcing a snapshot regardless of interval."""
        recorder = StateRecorder()
        recorder.start(interval_ticks=100)

        recorder.take_snapshot({"tick": 0})
        recorder.take_snapshot({"tick": 1}, force=True)
        recorder.take_snapshot({"tick": 2}, force=True)

        assert recorder.total_snapshots == 3

    def test_get_snapshot_at_tick(self) -> None:
        """Test getting snapshot at exact tick."""
        recorder = StateRecorder()
        recorder.start(interval_ticks=1)

        recorder.take_snapshot({"data": "a"}, tick=10)
        recorder.take_snapshot({"data": "b"}, tick=20)

        snapshot = recorder.get_snapshot_at_tick(10)
        assert snapshot is not None
        assert snapshot.state_data["data"] == "a"

        snapshot = recorder.get_snapshot_at_tick(15)
        assert snapshot is None

    def test_get_nearest_snapshot(self) -> None:
        """Test getting nearest snapshot before tick."""
        recorder = StateRecorder()
        recorder.start(interval_ticks=1)

        recorder.take_snapshot({"data": "a"}, tick=10)
        recorder.take_snapshot({"data": "b"}, tick=20)

        snapshot = recorder.get_nearest_snapshot(15)
        assert snapshot is not None
        assert snapshot.tick == 10

        snapshot = recorder.get_nearest_snapshot(25)
        assert snapshot is not None
        assert snapshot.tick == 20

    def test_state_serializer(self) -> None:
        """Test custom state serializer."""
        class MockState:
            def __init__(self, value: int) -> None:
                self.value = value

        def serialize(state: MockState) -> dict:
            return {"value": state.value}

        recorder = StateRecorder(state_serializer=serialize)
        recorder.start()
        recorder.take_snapshot(MockState(42))

        snapshot = recorder.snapshots[0]
        assert snapshot.state_data == {"value": 42}

    def test_save_load_pickle(self) -> None:
        """Test saving and loading state recording."""
        recorder = StateRecorder()
        recorder.start(interval_ticks=1)
        recorder.take_snapshot({"player": {"x": 10}}, tick=0)
        recorder.take_snapshot({"player": {"x": 20}}, tick=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pkl"
            recorder.save(path)

            loaded = StateRecorder()
            loaded.load(path)

            assert loaded.total_snapshots == 2
            assert loaded.snapshots[0].state_data == {"player": {"x": 10}}

    def test_clear(self) -> None:
        """Test clearing recorded data."""
        recorder = StateRecorder()
        recorder.start()
        recorder.take_snapshot({"data": 1})
        assert recorder.total_snapshots == 1

        recorder.clear()
        assert recorder.total_snapshots == 0


# =============================================================================
# RollingRecorder Tests
# =============================================================================


class TestRollingRecorder:
    """Tests for RollingRecorder class."""

    def test_create_recorder(self) -> None:
        """Test creating a rolling recorder."""
        recorder = RollingRecorder(keep_seconds=30.0, ticks_per_second=60)
        assert recorder.mode == RecordingMode.ROLLING
        assert recorder.keep_seconds == 30.0
        assert recorder.max_ticks == 1800

    def test_invalid_parameters(self) -> None:
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            RollingRecorder(keep_seconds=0)

        with pytest.raises(ValueError):
            RollingRecorder(ticks_per_second=0)

        with pytest.raises(ValueError):
            RollingRecorder(snapshot_interval_ticks=0)

    def test_record_input(self) -> None:
        """Test recording inputs."""
        recorder = RollingRecorder(keep_seconds=1.0, ticks_per_second=10)
        recorder.start()

        recorder.record_input("keyboard", {"key": "W"})
        recorder.record_input("mouse", {"button": 0})

        assert len(recorder.input_records) == 2

    def test_take_snapshot(self) -> None:
        """Test taking snapshots."""
        recorder = RollingRecorder(
            keep_seconds=1.0,
            ticks_per_second=10,
            snapshot_interval_ticks=5,
        )
        recorder.start()

        # First should be taken
        assert recorder.take_snapshot({"tick": 0}) is True

        # Next 4 should be skipped
        for i in range(1, 5):
            assert recorder.take_snapshot({"tick": i}) is False

        # Tick 5 should be taken
        assert recorder.take_snapshot({"tick": 5}) is True

        assert len(recorder.snapshots) == 2

    def test_rolling_buffer_prunes_old_data(self) -> None:
        """Test that old data is pruned when buffer is full."""
        recorder = RollingRecorder(
            keep_seconds=0.1,  # 100ms
            ticks_per_second=100,  # 10 ticks max
            snapshot_interval_ticks=1,
        )
        recorder.start()

        # Record 20 ticks of data
        for i in range(20):
            recorder.record_input("keyboard", {"key": str(i)}, tick=i)
            recorder.take_snapshot({"tick": i}, tick=i, force=True)
            recorder.advance_tick()

        # Should only keep last ~10 ticks worth
        records = recorder.input_records
        assert len(records) <= 15  # Some buffer for deque

        # First tick should be recent
        if records:
            assert records[0].tick >= 5

    def test_first_last_tick(self) -> None:
        """Test first_tick and last_tick properties."""
        recorder = RollingRecorder(keep_seconds=10.0)
        assert recorder.first_tick is None
        assert recorder.last_tick is None

        recorder.start()
        recorder.record_input("keyboard", {"key": "A"}, tick=5)
        recorder.record_input("keyboard", {"key": "B"}, tick=15)

        assert recorder.first_tick == 5
        assert recorder.last_tick == 15

    def test_save_load(self) -> None:
        """Test saving and loading rolling recording."""
        recorder = RollingRecorder(keep_seconds=1.0, ticks_per_second=10)
        recorder.start()

        recorder.record_input("keyboard", {"key": "W"}, tick=0)
        recorder.take_snapshot({"player": "pos"}, tick=0, force=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rolling.pkl"
            recorder.save(path)

            loaded = RollingRecorder()
            loaded.load(path)

            assert len(loaded.input_records) == 1
            assert len(loaded.snapshots) == 1
            assert loaded.input_records[0].data["key"] == "W"

    def test_clear(self) -> None:
        """Test clearing recorded data."""
        recorder = RollingRecorder()
        recorder.start()
        recorder.record_input("keyboard", {"key": "W"})
        recorder.take_snapshot({}, force=True)

        recorder.clear()

        assert len(recorder.input_records) == 0
        assert len(recorder.snapshots) == 0


# =============================================================================
# RecordingMode Tests
# =============================================================================


class TestRecordingMode:
    """Tests for RecordingMode enum."""

    def test_all_modes_exist(self) -> None:
        """Test that all expected modes exist."""
        assert RecordingMode.CONTINUOUS
        assert RecordingMode.TRIGGERED
        assert RecordingMode.ROLLING

    def test_modes_are_distinct(self) -> None:
        """Test that modes are distinct values."""
        modes = [RecordingMode.CONTINUOUS, RecordingMode.TRIGGERED, RecordingMode.ROLLING]
        assert len(set(modes)) == 3

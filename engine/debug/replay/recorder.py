"""Recording system for inputs and game state.

This module provides recorders for capturing game inputs and state snapshots
for replay functionality. Supports continuous, triggered, and rolling recording modes.
"""

from __future__ import annotations

import json
import pickle
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, BinaryIO, Callable


class RecordingMode(Enum):
    """Recording mode determines how data is captured.

    Attributes:
        CONTINUOUS: Record everything from start to stop
        TRIGGERED: Record only when explicitly triggered (on event/crash)
        ROLLING: Keep last N seconds, discarding older data
    """
    CONTINUOUS = auto()
    TRIGGERED = auto()
    ROLLING = auto()


@dataclass(slots=True)
class InputRecord:
    """A single recorded input event.

    Attributes:
        tick: The game tick when this input occurred
        input_type: Type of input (keyboard, mouse, gamepad, etc.)
        data: Input-specific data (key code, button, axis value, etc.)
        timestamp: Real-world timestamp when input was recorded
    """
    tick: int
    input_type: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tick": self.tick,
            "input_type": self.input_type,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InputRecord:
        """Create from dictionary."""
        return cls(
            tick=data["tick"],
            input_type=data["input_type"],
            data=data["data"],
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass(slots=True)
class StateSnapshot:
    """A snapshot of game state at a specific tick.

    Attributes:
        tick: The game tick when this snapshot was taken
        state_data: Complete or partial game state data
        timestamp: Real-world timestamp when snapshot was taken
    """
    tick: int
    state_data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tick": self.tick,
            "state_data": self.state_data,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateSnapshot:
        """Create from dictionary."""
        return cls(
            tick=data["tick"],
            state_data=data["state_data"],
            timestamp=data.get("timestamp", 0.0),
        )


class RecorderBase(ABC):
    """Base class for recorders.

    Provides common functionality for all recorder types including
    start/stop control and serialization.
    """

    def __init__(self, mode: RecordingMode = RecordingMode.CONTINUOUS) -> None:
        """Initialize the recorder.

        Args:
            mode: The recording mode to use
        """
        self._mode = mode
        self._is_recording = False
        self._start_time: float | None = None

    @property
    def mode(self) -> RecordingMode:
        """Get the recording mode."""
        return self._mode

    @property
    def is_recording(self) -> bool:
        """Check if recording is currently active."""
        return self._is_recording

    def start(self) -> None:
        """Start recording."""
        if not self._is_recording:
            self._is_recording = True
            self._start_time = time.time()
            self._on_start()

    def stop(self) -> None:
        """Stop recording."""
        if self._is_recording:
            self._is_recording = False
            self._on_stop()

    @abstractmethod
    def _on_start(self) -> None:
        """Called when recording starts."""
        pass

    @abstractmethod
    def _on_stop(self) -> None:
        """Called when recording stops."""
        pass

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Save recorded data to file.

        Args:
            path: Path to save the recording
        """
        pass

    @abstractmethod
    def load(self, path: Path | str) -> None:
        """Load recorded data from file.

        Args:
            path: Path to load the recording from
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all recorded data."""
        pass


class InputRecorder(RecorderBase):
    """Records player inputs for deterministic replay.

    The InputRecorder captures all player inputs along with the tick
    at which they occurred. This allows for deterministic replay when
    the game simulation is also deterministic.

    Example:
        recorder = InputRecorder()
        recorder.start()

        # During gameplay
        recorder.record_input("keyboard", {"key": "W", "action": "press"})
        recorder.record_input("mouse", {"button": 0, "position": (100, 200)})

        recorder.stop()
        recorder.save("inputs.replay")
    """

    def __init__(
        self,
        mode: RecordingMode = RecordingMode.CONTINUOUS,
        current_tick_provider: Callable[[], int] | None = None,
    ) -> None:
        """Initialize the input recorder.

        Args:
            mode: The recording mode to use
            current_tick_provider: Function that returns the current game tick.
                                   If None, uses an internal counter.
        """
        super().__init__(mode)
        self._records: list[InputRecord] = []
        self._current_tick_provider = current_tick_provider
        self._internal_tick = 0

    def _on_start(self) -> None:
        """Reset internal tick counter on start."""
        if self._mode == RecordingMode.CONTINUOUS:
            self._records.clear()
        self._internal_tick = 0

    def _on_stop(self) -> None:
        """Nothing special needed on stop."""
        pass

    def _get_current_tick(self) -> int:
        """Get the current game tick."""
        if self._current_tick_provider is not None:
            return self._current_tick_provider()
        return self._internal_tick

    def record_input(
        self,
        input_type: str,
        data: dict[str, Any],
        tick: int | None = None,
    ) -> None:
        """Record an input event.

        Args:
            input_type: Type of input (keyboard, mouse, gamepad, etc.)
            data: Input-specific data
            tick: Optional tick override. If None, uses current tick.
        """
        if not self._is_recording:
            return

        actual_tick = tick if tick is not None else self._get_current_tick()
        record = InputRecord(
            tick=actual_tick,
            input_type=input_type,
            data=data,
        )
        self._records.append(record)

    def advance_tick(self) -> None:
        """Advance the internal tick counter.

        Call this once per game tick when not using an external tick provider.
        """
        self._internal_tick += 1

    @property
    def records(self) -> list[InputRecord]:
        """Get all recorded inputs."""
        return list(self._records)

    def get_inputs_at_tick(self, tick: int) -> list[InputRecord]:
        """Get all inputs recorded at a specific tick.

        Args:
            tick: The game tick to query

        Returns:
            List of input records at that tick
        """
        return [r for r in self._records if r.tick == tick]

    def get_inputs_in_range(
        self, start_tick: int, end_tick: int
    ) -> list[InputRecord]:
        """Get all inputs in a tick range (inclusive).

        Args:
            start_tick: Start of the range
            end_tick: End of the range (inclusive)

        Returns:
            List of input records in the range
        """
        return [r for r in self._records if start_tick <= r.tick <= end_tick]

    @property
    def total_records(self) -> int:
        """Get total number of recorded inputs."""
        return len(self._records)

    @property
    def first_tick(self) -> int | None:
        """Get the first recorded tick, or None if empty."""
        if not self._records:
            return None
        return self._records[0].tick

    @property
    def last_tick(self) -> int | None:
        """Get the last recorded tick, or None if empty."""
        if not self._records:
            return None
        return self._records[-1].tick

    def save(self, path: Path | str) -> None:
        """Save recorded inputs to file.

        Args:
            path: Path to save the recording
        """
        path = Path(path)
        data = {
            "type": "input_recording",
            "version": 1,
            "mode": self._mode.name,
            "records": [r.to_dict() for r in self._records],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path | str) -> None:
        """Load recorded inputs from file.

        Args:
            path: Path to load the recording from

        Raises:
            ValueError: If file format is invalid
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("type") != "input_recording":
            raise ValueError(f"Invalid input recording file: {path}")

        self._records = [InputRecord.from_dict(r) for r in data["records"]]
        self._mode = RecordingMode[data.get("mode", "CONTINUOUS")]

    def clear(self) -> None:
        """Clear all recorded data."""
        self._records.clear()
        self._internal_tick = 0


class StateRecorder(RecorderBase):
    """Records game state snapshots at regular intervals.

    The StateRecorder captures complete or partial game state at specified
    tick intervals. This allows for replay seeking and state verification.

    Example:
        recorder = StateRecorder()
        recorder.start(interval_ticks=60)  # Snapshot every 60 ticks

        # Each tick
        recorder.take_snapshot(world.get_state())

        recorder.stop()
        recorder.save("state.replay")
    """

    def __init__(
        self,
        mode: RecordingMode = RecordingMode.CONTINUOUS,
        state_serializer: Callable[[Any], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the state recorder.

        Args:
            mode: The recording mode to use
            state_serializer: Optional function to serialize state objects.
                              If None, state must already be a dict.
        """
        super().__init__(mode)
        self._snapshots: list[StateSnapshot] = []
        self._interval_ticks = 1
        self._current_tick = 0
        self._ticks_since_snapshot = 0
        self._state_serializer = state_serializer

    def _on_start(self) -> None:
        """Reset counters on start."""
        if self._mode == RecordingMode.CONTINUOUS:
            self._snapshots.clear()
        # Set to interval so first snapshot is always taken
        self._ticks_since_snapshot = self._interval_ticks

    def _on_stop(self) -> None:
        """Nothing special needed on stop."""
        pass

    def start(self, interval_ticks: int = 1) -> None:
        """Start recording with a specific snapshot interval.

        Args:
            interval_ticks: Take a snapshot every N ticks. Default is 1 (every tick).
        """
        if interval_ticks < 1:
            raise ValueError("interval_ticks must be at least 1")
        self._interval_ticks = interval_ticks
        super().start()

    def take_snapshot(
        self,
        state: Any,
        tick: int | None = None,
        force: bool = False,
    ) -> bool:
        """Attempt to take a state snapshot.

        The snapshot is only taken if the interval has elapsed or force is True.

        Args:
            state: The game state to snapshot
            tick: Optional tick override. If None, uses internal counter.
            force: If True, take snapshot regardless of interval

        Returns:
            True if a snapshot was taken, False otherwise
        """
        if not self._is_recording:
            return False

        actual_tick = tick if tick is not None else self._current_tick
        self._current_tick = actual_tick + 1

        if not force and self._ticks_since_snapshot < self._interval_ticks:
            self._ticks_since_snapshot += 1
            return False

        self._ticks_since_snapshot = 1  # Reset to 1 since we just took a snapshot

        # Serialize state if needed
        if self._state_serializer is not None:
            state_data = self._state_serializer(state)
        elif isinstance(state, dict):
            state_data = state
        else:
            state_data = {"raw": state}

        snapshot = StateSnapshot(tick=actual_tick, state_data=state_data)
        self._snapshots.append(snapshot)
        return True

    @property
    def snapshots(self) -> list[StateSnapshot]:
        """Get all recorded snapshots."""
        return list(self._snapshots)

    def get_snapshot_at_tick(self, tick: int) -> StateSnapshot | None:
        """Get snapshot at exact tick, or None if not found.

        Args:
            tick: The game tick to query

        Returns:
            The snapshot at that tick, or None
        """
        for s in self._snapshots:
            if s.tick == tick:
                return s
        return None

    def get_nearest_snapshot(self, tick: int) -> StateSnapshot | None:
        """Get the nearest snapshot at or before the given tick.

        Args:
            tick: The target tick

        Returns:
            The nearest snapshot, or None if no snapshots exist
        """
        if not self._snapshots:
            return None

        # Find the latest snapshot at or before the tick
        result = None
        for s in self._snapshots:
            if s.tick <= tick:
                result = s
            else:
                break
        return result

    @property
    def total_snapshots(self) -> int:
        """Get total number of recorded snapshots."""
        return len(self._snapshots)

    @property
    def first_tick(self) -> int | None:
        """Get the first snapshot tick, or None if empty."""
        if not self._snapshots:
            return None
        return self._snapshots[0].tick

    @property
    def last_tick(self) -> int | None:
        """Get the last snapshot tick, or None if empty."""
        if not self._snapshots:
            return None
        return self._snapshots[-1].tick

    def save(self, path: Path | str) -> None:
        """Save recorded snapshots to file.

        Args:
            path: Path to save the recording
        """
        path = Path(path)
        data = {
            "type": "state_recording",
            "version": 1,
            "mode": self._mode.name,
            "interval_ticks": self._interval_ticks,
            "snapshots": [s.to_dict() for s in self._snapshots],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Path | str) -> None:
        """Load recorded snapshots from file.

        Args:
            path: Path to load the recording from

        Raises:
            ValueError: If file format is invalid
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        if data.get("type") != "state_recording":
            raise ValueError(f"Invalid state recording file: {path}")

        self._snapshots = [StateSnapshot.from_dict(s) for s in data["snapshots"]]
        self._mode = RecordingMode[data.get("mode", "CONTINUOUS")]
        self._interval_ticks = data.get("interval_ticks", 1)

    def clear(self) -> None:
        """Clear all recorded data."""
        self._snapshots.clear()
        self._current_tick = 0
        self._ticks_since_snapshot = 0


class RollingRecorder(RecorderBase):
    """Records the last N seconds of inputs and state.

    The RollingRecorder maintains a circular buffer of inputs and state
    snapshots, keeping only the most recent data. This is useful for
    crash replay or triggered recording.

    Example:
        recorder = RollingRecorder(keep_seconds=30.0, ticks_per_second=60)
        recorder.start()

        # Continuous recording
        recorder.record_input("keyboard", {"key": "W"})
        recorder.take_snapshot(world.get_state())

        # On crash or trigger
        recorder.save("crash_replay.bin")
    """

    # Default configuration constants
    DEFAULT_KEEP_SECONDS = 30.0  # 30 seconds of rolling buffer
    DEFAULT_TICKS_PER_SECOND = 60  # Standard 60 FPS
    DEFAULT_SNAPSHOT_INTERVAL = 60  # Snapshot every second (60 ticks at 60 FPS)
    MAX_INPUTS_PER_TICK = 10  # Estimated max inputs per tick for buffer sizing

    def __init__(
        self,
        keep_seconds: float = DEFAULT_KEEP_SECONDS,
        ticks_per_second: int = DEFAULT_TICKS_PER_SECOND,
        snapshot_interval_ticks: int = DEFAULT_SNAPSHOT_INTERVAL,
    ) -> None:
        """Initialize the rolling recorder.

        Args:
            keep_seconds: Number of seconds of data to keep (default: 30s)
            ticks_per_second: Game ticks per second for calculating buffer size (default: 60)
            snapshot_interval_ticks: Take state snapshot every N ticks (default: 60)
        """
        super().__init__(RecordingMode.ROLLING)

        if keep_seconds <= 0:
            raise ValueError("keep_seconds must be positive")
        if ticks_per_second <= 0:
            raise ValueError("ticks_per_second must be positive")
        if snapshot_interval_ticks <= 0:
            raise ValueError("snapshot_interval_ticks must be positive")

        self._keep_seconds = keep_seconds
        self._ticks_per_second = ticks_per_second
        self._max_ticks = int(keep_seconds * ticks_per_second)
        self._snapshot_interval = snapshot_interval_ticks

        # Estimate max records based on max ticks and inputs per tick
        max_input_records = self._max_ticks * self.MAX_INPUTS_PER_TICK
        max_snapshots = (self._max_ticks // snapshot_interval_ticks) + 1

        self._input_buffer: deque[InputRecord] = deque(maxlen=max_input_records)
        self._snapshot_buffer: deque[StateSnapshot] = deque(maxlen=max_snapshots)

        self._current_tick = 0
        self._ticks_since_snapshot = 0

    def _on_start(self) -> None:
        """Reset tick counters on start."""
        self._current_tick = 0
        # Set to interval so first snapshot is always taken
        self._ticks_since_snapshot = self._snapshot_interval

    def _on_stop(self) -> None:
        """Nothing special needed on stop."""
        pass

    @property
    def keep_seconds(self) -> float:
        """Get the number of seconds being kept."""
        return self._keep_seconds

    @property
    def max_ticks(self) -> int:
        """Get the maximum number of ticks kept."""
        return self._max_ticks

    def record_input(
        self,
        input_type: str,
        data: dict[str, Any],
        tick: int | None = None,
    ) -> None:
        """Record an input event.

        Args:
            input_type: Type of input
            data: Input-specific data
            tick: Optional tick override
        """
        if not self._is_recording:
            return

        actual_tick = tick if tick is not None else self._current_tick
        record = InputRecord(tick=actual_tick, input_type=input_type, data=data)
        self._input_buffer.append(record)

    def take_snapshot(
        self,
        state: dict[str, Any],
        tick: int | None = None,
        force: bool = False,
    ) -> bool:
        """Attempt to take a state snapshot.

        Args:
            state: The game state to snapshot
            tick: Optional tick override
            force: If True, take snapshot regardless of interval

        Returns:
            True if snapshot was taken
        """
        if not self._is_recording:
            return False

        actual_tick = tick if tick is not None else self._current_tick

        if not force and self._ticks_since_snapshot < self._snapshot_interval:
            self._ticks_since_snapshot += 1
            return False

        self._ticks_since_snapshot = 1  # Reset to 1 since we just took a snapshot

        snapshot = StateSnapshot(tick=actual_tick, state_data=state)
        self._snapshot_buffer.append(snapshot)
        return True

    def advance_tick(self) -> None:
        """Advance the internal tick counter and prune old data."""
        self._current_tick += 1
        self._prune_old_data()

    def _prune_old_data(self) -> None:
        """Remove data older than max_ticks."""
        cutoff_tick = self._current_tick - self._max_ticks

        # Prune old inputs
        while self._input_buffer and self._input_buffer[0].tick < cutoff_tick:
            self._input_buffer.popleft()

        # Prune old snapshots
        while self._snapshot_buffer and self._snapshot_buffer[0].tick < cutoff_tick:
            self._snapshot_buffer.popleft()

    @property
    def input_records(self) -> list[InputRecord]:
        """Get all buffered input records."""
        return list(self._input_buffer)

    @property
    def snapshots(self) -> list[StateSnapshot]:
        """Get all buffered snapshots."""
        return list(self._snapshot_buffer)

    @property
    def first_tick(self) -> int | None:
        """Get the oldest tick in buffer."""
        input_first = self._input_buffer[0].tick if self._input_buffer else None
        snap_first = self._snapshot_buffer[0].tick if self._snapshot_buffer else None

        if input_first is None:
            return snap_first
        if snap_first is None:
            return input_first
        return min(input_first, snap_first)

    @property
    def last_tick(self) -> int | None:
        """Get the newest tick in buffer."""
        input_last = self._input_buffer[-1].tick if self._input_buffer else None
        snap_last = self._snapshot_buffer[-1].tick if self._snapshot_buffer else None

        if input_last is None:
            return snap_last
        if snap_last is None:
            return input_last
        return max(input_last, snap_last)

    def save(self, path: Path | str) -> None:
        """Save buffered data to file.

        Args:
            path: Path to save the recording
        """
        path = Path(path)
        data = {
            "type": "rolling_recording",
            "version": 1,
            "keep_seconds": self._keep_seconds,
            "ticks_per_second": self._ticks_per_second,
            "snapshot_interval": self._snapshot_interval,
            "inputs": [r.to_dict() for r in self._input_buffer],
            "snapshots": [s.to_dict() for s in self._snapshot_buffer],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Path | str) -> None:
        """Load buffered data from file.

        Args:
            path: Path to load the recording from

        Raises:
            ValueError: If file format is invalid
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        if data.get("type") != "rolling_recording":
            raise ValueError(f"Invalid rolling recording file: {path}")

        self._keep_seconds = data["keep_seconds"]
        self._ticks_per_second = data["ticks_per_second"]
        self._snapshot_interval = data["snapshot_interval"]
        self._max_ticks = int(self._keep_seconds * self._ticks_per_second)

        # Recreate buffers with loaded data
        self._input_buffer.clear()
        for r in data["inputs"]:
            self._input_buffer.append(InputRecord.from_dict(r))

        self._snapshot_buffer.clear()
        for s in data["snapshots"]:
            self._snapshot_buffer.append(StateSnapshot.from_dict(s))

        # Update current tick to last recorded tick
        last_tick = self.last_tick
        self._current_tick = last_tick + 1 if last_tick is not None else 0

    def clear(self) -> None:
        """Clear all buffered data."""
        self._input_buffer.clear()
        self._snapshot_buffer.clear()
        self._current_tick = 0
        self._ticks_since_snapshot = 0

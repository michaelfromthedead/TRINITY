"""Replay playback system.

This module provides the ReplayPlayer for playing back recorded game sessions.
Supports play, pause, seek, speed control, and reverse playback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

from engine.debug.replay.recorder import (
    InputRecord,
    InputRecorder,
    RollingRecorder,
    StateRecorder,
    StateSnapshot,
)


class PlaybackState(Enum):
    """Current state of the replay player.

    Attributes:
        STOPPED: No replay loaded or playback stopped
        PLAYING: Actively playing the replay
        PAUSED: Playback is paused at current position
    """
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()


@dataclass
class PlaybackInfo:
    """Information about current playback state.

    Attributes:
        current_tick: Current playback tick
        total_ticks: Total ticks in the replay
        progress: Progress as a ratio (0.0 to 1.0)
        speed: Current playback speed multiplier
        state: Current playback state
        is_reversed: True if playing in reverse
    """
    current_tick: int
    total_ticks: int
    progress: float
    speed: float
    state: PlaybackState
    is_reversed: bool


class ReplayPlayer:
    """Plays back recorded game sessions.

    The ReplayPlayer handles loading and playing back recorded inputs and
    state snapshots. It supports various playback controls including speed
    adjustment, seeking, and reverse playback.

    Example:
        player = ReplayPlayer()
        player.load("replay.bin")

        player.play()
        player.set_speed(0.5)  # Slow motion
        player.seek(1500)  # Jump to tick 1500
        player.pause()
        player.step_frame()  # Advance one frame
        player.reverse()  # Play backwards
    """

    # Speed limits - import from centralized config when available
    # These match the tooling replay system's config.py values
    MIN_SPEED = 0.1  # Matches MIN_PLAYBACK_SPEED in tooling/replay/config.py
    MAX_SPEED = 4.0  # Reduced from MAX_PLAYBACK_SPEED (10.0) for debug system
    DEFAULT_SPEED = 1.0  # Matches DEFAULT_PLAYBACK_SPEED in tooling/replay/config.py

    def __init__(
        self,
        ticks_per_second: int = 60,
        on_input: Callable[[InputRecord], None] | None = None,
        on_state: Callable[[StateSnapshot], None] | None = None,
    ) -> None:
        """Initialize the replay player.

        Args:
            ticks_per_second: Target playback rate in ticks per second
            on_input: Callback fired when an input should be replayed
            on_state: Callback fired when state should be restored
        """
        self._ticks_per_second = ticks_per_second
        self._on_input = on_input
        self._on_state = on_state

        # Playback state
        self._state = PlaybackState.STOPPED
        self._current_tick = 0
        self._speed = self.DEFAULT_SPEED
        self._is_reversed = False

        # Recorded data
        self._inputs: list[InputRecord] = []
        self._snapshots: list[StateSnapshot] = []
        self._first_tick = 0
        self._last_tick = 0

        # Timing for smooth playback
        self._last_update_time: float | None = None
        self._tick_accumulator = 0.0

        # Input index for efficient lookup
        self._input_index: dict[int, list[InputRecord]] = {}

    @property
    def state(self) -> PlaybackState:
        """Get the current playback state."""
        return self._state

    @property
    def is_loaded(self) -> bool:
        """Check if a replay is loaded."""
        return len(self._inputs) > 0 or len(self._snapshots) > 0

    @property
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._state == PlaybackState.PLAYING

    @property
    def is_paused(self) -> bool:
        """Check if currently paused."""
        return self._state == PlaybackState.PAUSED

    def load(self, path: Path | str) -> None:
        """Load a replay file.

        Supports input recordings, state recordings, and rolling recordings.

        Args:
            path: Path to the replay file

        Raises:
            ValueError: If file format is not recognized
            FileNotFoundError: If file does not exist
        """
        self.unload()
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Replay file not found: {path}")

        # Try different formats
        suffix = path.suffix.lower()

        if suffix == ".json":
            # JSON format - input recording
            recorder = InputRecorder()
            recorder.load(path)
            self._inputs = recorder.records
        else:
            # Binary format - try state or rolling
            import pickle
            with open(path, "rb") as f:
                data = pickle.load(f)

            record_type = data.get("type", "")

            if record_type == "state_recording":
                recorder = StateRecorder()
                recorder.load(path)
                self._snapshots = recorder.snapshots
            elif record_type == "rolling_recording":
                recorder = RollingRecorder()
                recorder.load(path)
                self._inputs = recorder.input_records
                self._snapshots = recorder.snapshots
            elif record_type == "combined_replay":
                # Our own combined format
                self._inputs = [InputRecord.from_dict(i) for i in data.get("inputs", [])]
                self._snapshots = [StateSnapshot.from_dict(s) for s in data.get("snapshots", [])]
            else:
                raise ValueError(f"Unknown replay format: {record_type}")

        self._build_indices()
        self._current_tick = self._first_tick

    def load_combined(
        self,
        inputs: list[InputRecord] | None = None,
        snapshots: list[StateSnapshot] | None = None,
    ) -> None:
        """Load replay data directly.

        Args:
            inputs: List of input records
            snapshots: List of state snapshots
        """
        self.unload()
        self._inputs = list(inputs) if inputs else []
        self._snapshots = list(snapshots) if snapshots else []
        self._build_indices()
        self._current_tick = self._first_tick

    def _build_indices(self) -> None:
        """Build lookup indices for efficient tick access."""
        # Input index by tick
        self._input_index.clear()
        for inp in self._inputs:
            if inp.tick not in self._input_index:
                self._input_index[inp.tick] = []
            self._input_index[inp.tick].append(inp)

        # Calculate tick range
        input_ticks = [i.tick for i in self._inputs] if self._inputs else []
        snapshot_ticks = [s.tick for s in self._snapshots] if self._snapshots else []
        all_ticks = input_ticks + snapshot_ticks

        if all_ticks:
            self._first_tick = min(all_ticks)
            self._last_tick = max(all_ticks)
        else:
            self._first_tick = 0
            self._last_tick = 0

    def unload(self) -> None:
        """Unload the current replay."""
        self.stop()
        self._inputs.clear()
        self._snapshots.clear()
        self._input_index.clear()
        self._first_tick = 0
        self._last_tick = 0
        self._current_tick = 0

    def play(self) -> None:
        """Start or resume playback."""
        if not self.is_loaded:
            return

        # Only reset position if stopped and tick is at default (first tick)
        # This allows seeking before playing to work correctly
        if self._state == PlaybackState.STOPPED:
            if self._current_tick == self._first_tick and self._is_reversed:
                self._current_tick = self._last_tick
            elif self._current_tick == self._last_tick and not self._is_reversed:
                # If at end and playing forward, reset to start
                self._current_tick = self._first_tick

        self._state = PlaybackState.PLAYING
        self._last_update_time = time.perf_counter()
        self._tick_accumulator = 0.0

    def pause(self) -> None:
        """Pause playback."""
        if self._state == PlaybackState.PLAYING:
            self._state = PlaybackState.PAUSED

    def stop(self) -> None:
        """Stop playback and reset to beginning."""
        self._state = PlaybackState.STOPPED
        self._current_tick = self._first_tick
        self._is_reversed = False
        self._speed = self.DEFAULT_SPEED
        self._last_update_time = None
        self._tick_accumulator = 0.0

    def toggle_pause(self) -> None:
        """Toggle between play and pause states."""
        if self._state == PlaybackState.PLAYING:
            self.pause()
        elif self._state == PlaybackState.PAUSED:
            self.play()
        elif self._state == PlaybackState.STOPPED:
            self.play()

    def set_speed(self, multiplier: float) -> None:
        """Set the playback speed multiplier.

        Args:
            multiplier: Speed multiplier (0.1x to 4.0x)

        Raises:
            ValueError: If multiplier is out of range
        """
        if multiplier < self.MIN_SPEED or multiplier > self.MAX_SPEED:
            raise ValueError(
                f"Speed must be between {self.MIN_SPEED} and {self.MAX_SPEED}"
            )
        self._speed = multiplier

    @property
    def speed(self) -> float:
        """Get the current playback speed."""
        return self._speed

    def seek(self, tick: int) -> None:
        """Seek to a specific tick.

        Args:
            tick: Target tick to seek to
        """
        if not self.is_loaded:
            return

        # Clamp to valid range
        self._current_tick = max(self._first_tick, min(tick, self._last_tick))

        # Fire state callback for nearest snapshot
        self._restore_state_at_tick(self._current_tick)

    def seek_to_start(self) -> None:
        """Seek to the beginning of the replay."""
        self.seek(self._first_tick)

    def seek_to_end(self) -> None:
        """Seek to the end of the replay."""
        self.seek(self._last_tick)

    def step_frame(self, count: int = 1) -> None:
        """Step forward or backward by a number of frames.

        Args:
            count: Number of frames to step (negative for backward)
        """
        if not self.is_loaded:
            return

        if self._is_reversed:
            count = -count

        new_tick = self._current_tick + count
        new_tick = max(self._first_tick, min(new_tick, self._last_tick))

        old_tick = self._current_tick
        self._current_tick = new_tick

        # Fire callbacks for stepped ticks
        step = 1 if count > 0 else -1
        for tick in range(old_tick + step, new_tick + step, step):
            self._process_tick(tick)

    def reverse(self) -> None:
        """Toggle reverse playback mode."""
        self._is_reversed = not self._is_reversed

    @property
    def is_reversed(self) -> bool:
        """Check if playing in reverse."""
        return self._is_reversed

    def get_current_tick(self) -> int:
        """Get the current playback tick."""
        return self._current_tick

    def get_total_ticks(self) -> int:
        """Get the total number of ticks in the replay."""
        return self._last_tick - self._first_tick + 1 if self.is_loaded else 0

    def get_progress(self) -> float:
        """Get the current playback progress as a ratio (0.0 to 1.0)."""
        total = self.get_total_ticks()
        if total <= 1:
            return 0.0
        return (self._current_tick - self._first_tick) / (total - 1)

    def get_info(self) -> PlaybackInfo:
        """Get comprehensive playback information."""
        return PlaybackInfo(
            current_tick=self._current_tick,
            total_ticks=self.get_total_ticks(),
            progress=self.get_progress(),
            speed=self._speed,
            state=self._state,
            is_reversed=self._is_reversed,
        )

    def update(self, dt: float | None = None) -> int:
        """Update playback based on elapsed time.

        Call this once per frame to advance playback.

        Args:
            dt: Time delta in seconds. If None, calculated automatically.

        Returns:
            Number of ticks processed this update
        """
        if self._state != PlaybackState.PLAYING:
            return 0

        # Calculate dt if not provided
        current_time = time.perf_counter()
        if dt is None:
            if self._last_update_time is None:
                self._last_update_time = current_time
                return 0
            dt = current_time - self._last_update_time
        self._last_update_time = current_time

        # Accumulate time and calculate ticks to process
        seconds_per_tick = 1.0 / self._ticks_per_second
        adjusted_dt = dt * self._speed
        self._tick_accumulator += adjusted_dt

        ticks_to_process = int(self._tick_accumulator / seconds_per_tick)
        self._tick_accumulator -= ticks_to_process * seconds_per_tick

        # Process ticks
        ticks_processed = 0
        for _ in range(ticks_to_process):
            # Check if we've reached the end
            if self._is_reversed:
                if self._current_tick <= self._first_tick:
                    self.pause()
                    break
                self._current_tick -= 1
            else:
                if self._current_tick >= self._last_tick:
                    self.pause()
                    break
                self._current_tick += 1

            self._process_tick(self._current_tick)
            ticks_processed += 1

        return ticks_processed

    def _process_tick(self, tick: int) -> None:
        """Process a single tick, firing callbacks as needed.

        Args:
            tick: The tick to process
        """
        # Fire input callbacks
        if self._on_input is not None:
            inputs = self._input_index.get(tick, [])
            for inp in inputs:
                self._on_input(inp)

        # Check for state snapshot at this tick
        self._restore_state_at_tick(tick, exact_only=True)

    def _restore_state_at_tick(self, tick: int, exact_only: bool = False) -> None:
        """Restore state from snapshot at or before tick.

        Args:
            tick: Target tick
            exact_only: If True, only restore if exact tick match
        """
        if self._on_state is None or not self._snapshots:
            return

        # Find snapshot at or before tick
        best_snapshot = None
        for s in self._snapshots:
            if s.tick == tick:
                best_snapshot = s
                break
            elif not exact_only and s.tick < tick:
                best_snapshot = s

        if best_snapshot is not None and (not exact_only or best_snapshot.tick == tick):
            self._on_state(best_snapshot)

    def get_inputs_at_tick(self, tick: int) -> list[InputRecord]:
        """Get all inputs at a specific tick.

        Args:
            tick: The tick to query

        Returns:
            List of input records at that tick
        """
        return self._input_index.get(tick, [])

    def get_snapshot_at_tick(self, tick: int) -> StateSnapshot | None:
        """Get state snapshot at exact tick.

        Args:
            tick: The tick to query

        Returns:
            Snapshot at that tick, or None
        """
        for s in self._snapshots:
            if s.tick == tick:
                return s
        return None

    def get_nearest_snapshot(self, tick: int) -> StateSnapshot | None:
        """Get the nearest snapshot at or before tick.

        Args:
            tick: Target tick

        Returns:
            Nearest snapshot, or None
        """
        best = None
        for s in self._snapshots:
            if s.tick <= tick:
                best = s
            else:
                break
        return best

    @property
    def first_tick(self) -> int:
        """Get the first tick in the replay."""
        return self._first_tick

    @property
    def last_tick(self) -> int:
        """Get the last tick in the replay."""
        return self._last_tick

    def save_combined(self, path: Path | str) -> None:
        """Save current loaded replay to a combined format file.

        Args:
            path: Path to save the replay
        """
        import pickle

        path = Path(path)
        data = {
            "type": "combined_replay",
            "version": 1,
            "inputs": [i.to_dict() for i in self._inputs],
            "snapshots": [s.to_dict() for s in self._snapshots],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

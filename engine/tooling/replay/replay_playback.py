"""
Replay Playback - Playback replays with speed control and seeking.

Provides variable speed playback (0.25x to 4x), frame stepping,
and seeking capabilities for recorded game sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Iterator
import time

from .input_recorder import RecordedInput, InputType
from .state_recorder import StateSnapshot, StateDelta


class PlaybackState(Enum):
    """Current state of replay playback."""
    STOPPED = auto()
    PLAYING = auto()
    PAUSED = auto()
    SEEKING = auto()
    FINISHED = auto()


class PlaybackSpeed(Enum):
    """Preset playback speeds."""
    QUARTER = 0.25
    HALF = 0.5
    NORMAL = 1.0
    DOUBLE = 2.0
    QUADRUPLE = 4.0

    @classmethod
    def from_value(cls, value: float) -> 'PlaybackSpeed':
        """Get preset from value, or NORMAL if not a preset."""
        for preset in cls:
            if abs(preset.value - value) < 0.001:
                return preset
        return cls.NORMAL


class SeekMode(Enum):
    """Seeking modes for replay navigation."""
    FRAME = auto()  # Seek to specific frame
    TIME = auto()   # Seek to specific time
    PERCENTAGE = auto()  # Seek to percentage of replay
    KEYFRAME = auto()  # Seek to nearest keyframe
    MARKER = auto()  # Seek to named marker


@dataclass
class PlaybackConfig:
    """Configuration for replay playback."""
    # Speed settings
    initial_speed: float = 1.0
    min_speed: float = 0.1
    max_speed: float = 10.0

    # Frame control
    frame_step_size: int = 1  # Frames per step
    sub_frame_interpolation: bool = False

    # Input injection
    inject_inputs: bool = True  # Inject recorded inputs during playback
    input_callback: Optional[Callable[[RecordedInput], None]] = None

    # State restoration
    restore_state_on_seek: bool = True
    state_callback: Optional[Callable[[dict[str, Any]], None]] = None

    # Looping
    loop: bool = False
    loop_start_frame: int = 0
    loop_end_frame: int = -1  # -1 means end of replay

    # Event callbacks
    on_frame_advance: Optional[Callable[[int, float], None]] = None
    on_playback_complete: Optional[Callable[[], None]] = None
    on_state_change: Optional[Callable[[PlaybackState], None]] = None


@dataclass
class PlaybackPosition:
    """Current position in replay playback."""
    frame: int = 0
    timestamp: float = 0.0
    percentage: float = 0.0
    nearest_keyframe: int = 0


class ReplayPlayback:
    """Controls replay playback with speed and seeking.

    Provides variable speed playback, frame stepping, and seeking
    capabilities for recorded game sessions.
    """
    __slots__ = (
        '_config', '_inputs', '_snapshots', '_deltas',
        '_state', '_speed', '_current_frame', '_current_time',
        '_total_frames', '_total_duration', '_real_time_start',
        '_playback_time_start', '_input_index', '_markers',
        '_last_injected_state', '_frame_callbacks'
    )

    def __init__(
        self,
        inputs: list[RecordedInput],
        snapshots: list[StateSnapshot],
        deltas: Optional[list[StateDelta]] = None,
        config: Optional[PlaybackConfig] = None
    ):
        """Initialize replay playback.

        Args:
            inputs: List of recorded inputs
            snapshots: List of state snapshots
            deltas: Optional list of state deltas
            config: Playback configuration
        """
        self._config = config or PlaybackConfig()
        self._inputs = sorted(inputs, key=lambda x: x.timestamp)
        self._snapshots = sorted(snapshots, key=lambda x: x.frame)
        self._deltas = sorted(deltas or [], key=lambda x: x.from_frame)

        self._state = PlaybackState.STOPPED
        self._speed = self._config.initial_speed
        self._current_frame = 0
        self._current_time = 0.0
        self._real_time_start = 0.0
        self._playback_time_start = 0.0
        self._input_index = 0
        self._markers: dict[str, int] = {}
        self._last_injected_state: Optional[dict[str, Any]] = None
        self._frame_callbacks: list[tuple[int, Callable[[], None]]] = []

        # Calculate totals
        self._total_frames = max((s.frame for s in self._snapshots), default=0)
        self._total_duration = max((s.timestamp for s in self._snapshots), default=0.0)
        if self._inputs:
            self._total_frames = max(self._total_frames, max(i.frame for i in self._inputs))
            self._total_duration = max(self._total_duration, max(i.timestamp for i in self._inputs))

    @property
    def state(self) -> PlaybackState:
        """Get current playback state."""
        return self._state

    @property
    def speed(self) -> float:
        """Get current playback speed."""
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        """Set playback speed.

        Args:
            value: New speed multiplier (clamped to config limits)
        """
        self._speed = max(self._config.min_speed, min(self._config.max_speed, value))

    @property
    def current_frame(self) -> int:
        """Get current frame number."""
        return self._current_frame

    @property
    def current_time(self) -> float:
        """Get current playback time."""
        return self._current_time

    @property
    def total_frames(self) -> int:
        """Get total number of frames."""
        return self._total_frames

    @property
    def total_duration(self) -> float:
        """Get total duration in seconds."""
        return self._total_duration

    @property
    def position(self) -> PlaybackPosition:
        """Get current playback position."""
        return PlaybackPosition(
            frame=self._current_frame,
            timestamp=self._current_time,
            percentage=self._current_time / self._total_duration if self._total_duration > 0 else 0.0,
            nearest_keyframe=self._find_nearest_keyframe_frame(self._current_frame)
        )

    @property
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._state == PlaybackState.PLAYING

    @property
    def is_paused(self) -> bool:
        """Check if currently paused."""
        return self._state == PlaybackState.PAUSED

    @property
    def is_finished(self) -> bool:
        """Check if playback has finished."""
        return self._state == PlaybackState.FINISHED

    def play(self) -> None:
        """Start or resume playback."""
        if self._state == PlaybackState.FINISHED:
            # Restart from beginning
            self.seek(0, SeekMode.FRAME)

        prev_state = self._state
        self._state = PlaybackState.PLAYING
        self._real_time_start = time.perf_counter()
        self._playback_time_start = self._current_time

        if prev_state != PlaybackState.PLAYING:
            self._notify_state_change()

    def pause(self) -> None:
        """Pause playback."""
        if self._state == PlaybackState.PLAYING:
            self._state = PlaybackState.PAUSED
            self._notify_state_change()

    def stop(self) -> None:
        """Stop playback and reset to beginning."""
        self._state = PlaybackState.STOPPED
        self._current_frame = 0
        self._current_time = 0.0
        self._input_index = 0
        self._notify_state_change()

    def toggle_play_pause(self) -> None:
        """Toggle between play and pause states."""
        if self._state == PlaybackState.PLAYING:
            self.pause()
        else:
            self.play()

    def update(self, delta_time: float) -> list[RecordedInput]:
        """Update playback state.

        Should be called each frame to advance playback.

        Args:
            delta_time: Real-world time since last update

        Returns:
            List of inputs that occurred this update
        """
        if self._state != PlaybackState.PLAYING:
            return []

        # Calculate playback time advancement
        playback_delta = delta_time * self._speed
        new_time = self._current_time + playback_delta

        # Check for end of replay
        if new_time >= self._total_duration:
            if self._config.loop:
                # Handle looping
                loop_start = self._config.loop_start_frame
                if self._config.loop_end_frame >= 0:
                    # Loop to specific point
                    self.seek(loop_start, SeekMode.FRAME)
                    return []
                else:
                    self.seek(loop_start, SeekMode.FRAME)
                    return []
            else:
                self._state = PlaybackState.FINISHED
                self._notify_state_change()
                if self._config.on_playback_complete:
                    self._config.on_playback_complete()
                return []

        # Collect inputs in time range
        inputs = self._get_inputs_in_range(self._current_time, new_time)

        # Advance time and frame
        old_frame = self._current_frame
        self._current_time = new_time
        self._current_frame = self._time_to_frame(new_time)

        # Inject inputs if configured
        if self._config.inject_inputs and self._config.input_callback:
            for inp in inputs:
                self._config.input_callback(inp)

        # Check frame callbacks
        self._check_frame_callbacks(old_frame, self._current_frame)

        # Notify frame advance
        if self._config.on_frame_advance:
            self._config.on_frame_advance(self._current_frame, self._current_time)

        return inputs

    def seek(
        self,
        target: int | float | str,
        mode: SeekMode = SeekMode.FRAME
    ) -> bool:
        """Seek to a position in the replay.

        Args:
            target: Target position (interpretation depends on mode)
            mode: Seeking mode

        Returns:
            True if seek was successful
        """
        prev_state = self._state
        self._state = PlaybackState.SEEKING

        try:
            if mode == SeekMode.FRAME:
                return self._seek_to_frame(int(target))
            elif mode == SeekMode.TIME:
                return self._seek_to_time(float(target))
            elif mode == SeekMode.PERCENTAGE:
                time_target = float(target) * self._total_duration
                return self._seek_to_time(time_target)
            elif mode == SeekMode.KEYFRAME:
                return self._seek_to_nearest_keyframe(int(target))
            elif mode == SeekMode.MARKER:
                return self._seek_to_marker(str(target))
            else:
                return False
        finally:
            # Restore previous state or set to paused
            if prev_state == PlaybackState.PLAYING:
                self._state = PlaybackState.PLAYING
                self._real_time_start = time.perf_counter()
                self._playback_time_start = self._current_time
            else:
                self._state = PlaybackState.PAUSED

    def step_forward(self, frames: int = 1) -> list[RecordedInput]:
        """Step forward by specified frames.

        Args:
            frames: Number of frames to step

        Returns:
            List of inputs in stepped frames
        """
        target_frame = min(self._current_frame + frames, self._total_frames)
        return self._step_to_frame(target_frame)

    def step_backward(self, frames: int = 1) -> None:
        """Step backward by specified frames.

        Args:
            frames: Number of frames to step back
        """
        target_frame = max(self._current_frame - frames, 0)
        self.seek(target_frame, SeekMode.FRAME)

    def next_keyframe(self) -> bool:
        """Seek to the next keyframe.

        Returns:
            True if a next keyframe exists
        """
        for snapshot in self._snapshots:
            if snapshot.is_keyframe and snapshot.frame > self._current_frame:
                return self.seek(snapshot.frame, SeekMode.FRAME)
        return False

    def previous_keyframe(self) -> bool:
        """Seek to the previous keyframe.

        Returns:
            True if a previous keyframe exists
        """
        prev_keyframe = None
        for snapshot in self._snapshots:
            if snapshot.is_keyframe and snapshot.frame < self._current_frame:
                prev_keyframe = snapshot

        if prev_keyframe:
            return self.seek(prev_keyframe.frame, SeekMode.FRAME)
        return False

    def add_marker(self, name: str, frame: Optional[int] = None) -> None:
        """Add a named marker at current or specified frame.

        Args:
            name: Marker name
            frame: Frame number (default: current frame)
        """
        self._markers[name] = frame if frame is not None else self._current_frame

    def remove_marker(self, name: str) -> bool:
        """Remove a named marker.

        Args:
            name: Marker name

        Returns:
            True if marker was removed
        """
        return self._markers.pop(name, None) is not None

    def get_markers(self) -> dict[str, int]:
        """Get all markers.

        Returns:
            Dictionary of marker names to frame numbers
        """
        return self._markers.copy()

    def add_frame_callback(
        self,
        frame: int,
        callback: Callable[[], None]
    ) -> None:
        """Add a callback to trigger at specific frame.

        Args:
            frame: Frame to trigger at
            callback: Function to call
        """
        self._frame_callbacks.append((frame, callback))
        self._frame_callbacks.sort(key=lambda x: x[0])

    def get_inputs_at_frame(self, frame: int) -> list[RecordedInput]:
        """Get all inputs at a specific frame.

        Args:
            frame: Target frame

        Returns:
            List of inputs at that frame
        """
        return [inp for inp in self._inputs if inp.frame == frame]

    def get_state_at_frame(self, frame: int) -> Optional[dict[str, Any]]:
        """Get game state at specific frame.

        Args:
            frame: Target frame

        Returns:
            State dictionary at frame, or None if not available
        """
        # Find nearest keyframe
        keyframe = None
        for snapshot in self._snapshots:
            if snapshot.is_keyframe and snapshot.frame <= frame:
                keyframe = snapshot

        if keyframe is None:
            return None

        # Start with keyframe state
        import copy
        state = copy.deepcopy(keyframe.state_data)

        # Apply deltas
        for delta in self._deltas:
            if delta.from_frame >= keyframe.frame and delta.to_frame <= frame:
                state = delta.apply(state)

        return state

    def get_inputs_in_time_range(
        self,
        start: float,
        end: float
    ) -> list[RecordedInput]:
        """Get inputs within a time range.

        Args:
            start: Start time in seconds
            end: End time in seconds

        Returns:
            List of inputs in range
        """
        return self._get_inputs_in_range(start, end)

    def iter_inputs(self) -> Iterator[RecordedInput]:
        """Iterate over all inputs.

        Yields:
            Recorded inputs in order
        """
        yield from self._inputs

    def iter_snapshots(self) -> Iterator[StateSnapshot]:
        """Iterate over all snapshots.

        Yields:
            State snapshots in order
        """
        yield from self._snapshots

    def speed_up(self, factor: float = 2.0) -> None:
        """Increase playback speed.

        Args:
            factor: Speed multiplier
        """
        self.speed = self._speed * factor

    def slow_down(self, factor: float = 2.0) -> None:
        """Decrease playback speed.

        Args:
            factor: Speed divisor
        """
        self.speed = self._speed / factor

    def set_preset_speed(self, preset: PlaybackSpeed) -> None:
        """Set playback speed to a preset.

        Args:
            preset: Speed preset
        """
        self.speed = preset.value

    def _seek_to_frame(self, frame: int) -> bool:
        """Seek to specific frame."""
        frame = max(0, min(frame, self._total_frames))
        self._current_frame = frame
        self._current_time = self._frame_to_time(frame)
        self._input_index = self._find_input_index_at_time(self._current_time)

        # Restore state if configured
        if self._config.restore_state_on_seek:
            self._restore_state_at_frame(frame)

        return True

    def _seek_to_time(self, time_seconds: float) -> bool:
        """Seek to specific time."""
        time_seconds = max(0.0, min(time_seconds, self._total_duration))
        self._current_time = time_seconds
        self._current_frame = self._time_to_frame(time_seconds)
        self._input_index = self._find_input_index_at_time(time_seconds)

        # Restore state if configured
        if self._config.restore_state_on_seek:
            self._restore_state_at_frame(self._current_frame)

        return True

    def _seek_to_nearest_keyframe(self, frame: int) -> bool:
        """Seek to nearest keyframe."""
        keyframe_frame = self._find_nearest_keyframe_frame(frame)
        return self._seek_to_frame(keyframe_frame)

    def _seek_to_marker(self, name: str) -> bool:
        """Seek to named marker."""
        frame = self._markers.get(name)
        if frame is not None:
            return self._seek_to_frame(frame)
        return False

    def _step_to_frame(self, frame: int) -> list[RecordedInput]:
        """Step to specific frame, collecting inputs."""
        inputs = []

        while self._current_frame < frame:
            frame_inputs = self.get_inputs_at_frame(self._current_frame)
            inputs.extend(frame_inputs)

            # Inject inputs
            if self._config.inject_inputs and self._config.input_callback:
                for inp in frame_inputs:
                    self._config.input_callback(inp)

            self._current_frame += 1
            self._current_time = self._frame_to_time(self._current_frame)

        return inputs

    def _restore_state_at_frame(self, frame: int) -> None:
        """Restore game state at frame."""
        state = self.get_state_at_frame(frame)
        if state and self._config.state_callback:
            self._config.state_callback(state)
            self._last_injected_state = state

    def _get_inputs_in_range(
        self,
        start_time: float,
        end_time: float
    ) -> list[RecordedInput]:
        """Get inputs within time range efficiently."""
        inputs = []

        # Start from cached index
        i = self._input_index
        while i < len(self._inputs):
            inp = self._inputs[i]
            if inp.timestamp > end_time:
                break
            if inp.timestamp >= start_time:
                inputs.append(inp)
                self._input_index = i + 1
            i += 1

        return inputs

    def _find_input_index_at_time(self, time_seconds: float) -> int:
        """Find input index at or before specified time."""
        # Binary search
        left, right = 0, len(self._inputs)
        while left < right:
            mid = (left + right) // 2
            if self._inputs[mid].timestamp < time_seconds:
                left = mid + 1
            else:
                right = mid
        return left

    def _find_nearest_keyframe_frame(self, frame: int) -> int:
        """Find nearest keyframe at or before frame."""
        nearest = 0
        for snapshot in self._snapshots:
            if snapshot.is_keyframe and snapshot.frame <= frame:
                nearest = snapshot.frame
        return nearest

    def _time_to_frame(self, time_seconds: float) -> int:
        """Convert time to frame number (assuming 60 FPS)."""
        # Find frame from snapshots
        for snapshot in reversed(self._snapshots):
            if snapshot.timestamp <= time_seconds:
                return snapshot.frame
        return 0

    def _frame_to_time(self, frame: int) -> float:
        """Convert frame to time."""
        for snapshot in self._snapshots:
            if snapshot.frame == frame:
                return snapshot.timestamp
        # Estimate based on frame rate
        if self._snapshots and self._total_frames > 0:
            return (frame / self._total_frames) * self._total_duration
        return 0.0

    def _check_frame_callbacks(self, old_frame: int, new_frame: int) -> None:
        """Check and trigger frame callbacks."""
        for frame, callback in self._frame_callbacks:
            if old_frame < frame <= new_frame:
                callback()

    def _notify_state_change(self) -> None:
        """Notify state change callback."""
        if self._config.on_state_change:
            self._config.on_state_change(self._state)

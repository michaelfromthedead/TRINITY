"""
Input Recorder - Record player inputs with precise timestamps.

Captures all player inputs including keyboard, mouse, gamepad, and touch
with sub-millisecond precision for accurate replay.
"""

from __future__ import annotations

import hashlib
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Iterator


class InputType(Enum):
    """Types of input that can be recorded."""
    KEYBOARD = auto()
    MOUSE_BUTTON = auto()
    MOUSE_MOVE = auto()
    MOUSE_SCROLL = auto()
    GAMEPAD_BUTTON = auto()
    GAMEPAD_AXIS = auto()
    GAMEPAD_TRIGGER = auto()
    TOUCH_START = auto()
    TOUCH_MOVE = auto()
    TOUCH_END = auto()
    CUSTOM = auto()


@dataclass(slots=True, frozen=True)
class RecordedInput:
    """A single recorded input event."""
    input_type: InputType
    timestamp: float  # Seconds since recording start
    frame: int  # Frame number when input occurred
    device_id: int
    data: dict[str, Any]

    def serialize(self) -> bytes:
        """Serialize input to bytes for storage.

        Returns:
            Serialized bytes representation
        """
        # Pack: type(1), timestamp(8), frame(4), device_id(2)
        header = struct.pack(
            '<BdIH',
            self.input_type.value,
            self.timestamp,
            self.frame,
            self.device_id
        )
        # Serialize data as key-value pairs
        data_bytes = self._serialize_data()
        return header + struct.pack('<I', len(data_bytes)) + data_bytes

    def _serialize_data(self) -> bytes:
        """Serialize data dictionary to bytes."""
        import json
        # Convert enum values to strings for JSON serialization
        serializable_data = {}
        for key, value in self.data.items():
            if isinstance(value, Enum):
                serializable_data[key] = {'__enum__': type(value).__name__, 'value': value.name}
            else:
                serializable_data[key] = value
        return json.dumps(serializable_data).encode('utf-8')

    @classmethod
    def deserialize(cls, data: bytes, offset: int = 0) -> tuple['RecordedInput', int]:
        """Deserialize input from bytes.

        Args:
            data: Byte buffer
            offset: Starting offset

        Returns:
            Tuple of (RecordedInput, bytes consumed)
        """
        import json

        # Unpack header
        header_size = struct.calcsize('<BdIH')
        type_val, timestamp, frame, device_id = struct.unpack(
            '<BdIH', data[offset:offset + header_size]
        )
        offset += header_size

        # Get data length and content
        data_len = struct.unpack('<I', data[offset:offset + 4])[0]
        offset += 4
        data_bytes = data[offset:offset + data_len]
        offset += data_len

        # Deserialize data
        input_data = json.loads(data_bytes.decode('utf-8'))

        return cls(
            input_type=InputType(type_val),
            timestamp=timestamp,
            frame=frame,
            device_id=device_id,
            data=input_data
        ), offset

    def matches_frame(self, frame: int) -> bool:
        """Check if input occurred on specified frame."""
        return self.frame == frame

    def matches_time_range(self, start: float, end: float) -> bool:
        """Check if input occurred within time range."""
        return start <= self.timestamp <= end


@dataclass
class InputRecordingConfig:
    """Configuration for input recording."""
    # Timestamp precision
    high_precision_timing: bool = True

    # Input filtering
    record_keyboard: bool = True
    record_mouse: bool = True
    record_gamepad: bool = True
    record_touch: bool = True
    record_custom: bool = True

    # Buffer settings
    max_buffer_size: int = 100000  # Max inputs before auto-flush
    flush_interval: float = 5.0   # Auto-flush interval in seconds

    # Deduplication
    deduplicate_mouse_moves: bool = True
    mouse_move_threshold: float = 0.001  # Minimum time between mouse moves

    # Compression
    compress_on_flush: bool = True

    # Custom input filter
    input_filter: Optional[Callable[[RecordedInput], bool]] = None


@dataclass
class InputRecordingStats:
    """Statistics for input recording session."""
    total_inputs: int = 0
    inputs_by_type: dict[InputType, int] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    total_frames: int = 0
    bytes_recorded: int = 0
    compression_ratio: float = 1.0
    dropped_inputs: int = 0

    @property
    def duration(self) -> float:
        """Get recording duration in seconds."""
        return self.end_time - self.start_time

    @property
    def inputs_per_second(self) -> float:
        """Get average inputs per second."""
        if self.duration <= 0:
            return 0.0
        return self.total_inputs / self.duration

    def increment(self, input_type: InputType) -> None:
        """Increment counter for input type."""
        self.total_inputs += 1
        self.inputs_by_type[input_type] = self.inputs_by_type.get(input_type, 0) + 1


class InputRecorder:
    """Records player inputs with precise timestamps.

    This class captures all player inputs and stores them with
    sub-millisecond timestamps for accurate replay playback.
    """
    __slots__ = (
        '_config', '_inputs', '_is_recording', '_start_time',
        '_current_frame', '_stats', '_last_flush_time',
        '_last_mouse_move_time', '_flush_callback', '_input_hash'
    )

    def __init__(self, config: Optional[InputRecordingConfig] = None):
        """Initialize the input recorder.

        Args:
            config: Recording configuration
        """
        self._config = config or InputRecordingConfig()
        self._inputs: deque[RecordedInput] = deque()
        self._is_recording = False
        self._start_time = 0.0
        self._current_frame = 0
        self._stats = InputRecordingStats()
        self._last_flush_time = 0.0
        self._last_mouse_move_time = 0.0
        self._flush_callback: Optional[Callable[[list[RecordedInput]], None]] = None
        self._input_hash = hashlib.sha256()

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    @property
    def stats(self) -> InputRecordingStats:
        """Get current recording statistics."""
        return self._stats

    @property
    def input_count(self) -> int:
        """Get number of inputs in buffer."""
        return len(self._inputs)

    @property
    def current_frame(self) -> int:
        """Get current frame number."""
        return self._current_frame

    @property
    def elapsed_time(self) -> float:
        """Get elapsed recording time."""
        if not self._is_recording:
            return self._stats.duration
        return self._get_timestamp()

    @property
    def input_hash(self) -> str:
        """Get hash of all recorded inputs for verification."""
        return self._input_hash.hexdigest()

    def start(self) -> None:
        """Start recording inputs."""
        if self._is_recording:
            return

        self._is_recording = True
        self._start_time = time.perf_counter() if self._config.high_precision_timing else time.time()
        self._current_frame = 0
        self._stats = InputRecordingStats(start_time=self._start_time)
        self._last_flush_time = self._start_time
        self._last_mouse_move_time = 0.0
        self._input_hash = hashlib.sha256()
        self._inputs.clear()

    def stop(self) -> list[RecordedInput]:
        """Stop recording and return all recorded inputs.

        Returns:
            List of all recorded inputs
        """
        if not self._is_recording:
            return list(self._inputs)

        self._is_recording = False
        self._stats.end_time = self._get_timestamp() + self._start_time
        self._stats.total_frames = self._current_frame

        return list(self._inputs)

    def pause(self) -> None:
        """Pause recording (inputs will be ignored)."""
        self._is_recording = False

    def resume(self) -> None:
        """Resume recording after pause."""
        self._is_recording = True

    def advance_frame(self) -> None:
        """Advance to next frame."""
        self._current_frame += 1

        # Check for auto-flush
        current_time = self._get_timestamp()
        if (current_time - self._last_flush_time >= self._config.flush_interval
                or len(self._inputs) >= self._config.max_buffer_size):
            self._auto_flush()

    def record(
        self,
        input_type: InputType,
        device_id: int,
        data: dict[str, Any]
    ) -> Optional[RecordedInput]:
        """Record an input event.

        Args:
            input_type: Type of input
            device_id: Device that generated the input
            data: Input-specific data

        Returns:
            The recorded input, or None if filtered/not recording
        """
        if not self._is_recording:
            return None

        # Check input type filtering
        if not self._should_record_type(input_type):
            return None

        # Handle mouse move deduplication
        if (input_type == InputType.MOUSE_MOVE
                and self._config.deduplicate_mouse_moves):
            current_time = self._get_timestamp()
            # Allow first mouse move (when last_mouse_move_time is 0)
            if self._last_mouse_move_time > 0 and current_time - self._last_mouse_move_time < self._config.mouse_move_threshold:
                return None
            self._last_mouse_move_time = current_time

        # Create recorded input
        recorded = RecordedInput(
            input_type=input_type,
            timestamp=self._get_timestamp(),
            frame=self._current_frame,
            device_id=device_id,
            data=data
        )

        # Apply custom filter
        if self._config.input_filter and not self._config.input_filter(recorded):
            return None

        # Store input
        self._inputs.append(recorded)
        self._stats.increment(input_type)

        # Update hash for verification
        self._input_hash.update(recorded.serialize())

        return recorded

    def record_keyboard(
        self,
        key: Any,
        is_pressed: bool,
        modifiers: Optional[dict[str, bool]] = None,
        device_id: int = 0
    ) -> Optional[RecordedInput]:
        """Record a keyboard input.

        Args:
            key: Key code or key name
            is_pressed: True if key pressed, False if released
            modifiers: Modifier key states (shift, ctrl, alt, etc.)
            device_id: Keyboard device ID

        Returns:
            The recorded input, or None if not recording
        """
        data = {
            'key': key.name if isinstance(key, Enum) else str(key),
            'pressed': is_pressed,
            'modifiers': modifiers or {}
        }
        return self.record(InputType.KEYBOARD, device_id, data)

    def record_mouse_button(
        self,
        button: int,
        is_pressed: bool,
        x: float,
        y: float,
        device_id: int = 0
    ) -> Optional[RecordedInput]:
        """Record a mouse button input.

        Args:
            button: Mouse button (0=left, 1=right, 2=middle)
            is_pressed: True if pressed, False if released
            x: Mouse X position
            y: Mouse Y position
            device_id: Mouse device ID

        Returns:
            The recorded input, or None if not recording
        """
        data = {
            'button': button,
            'pressed': is_pressed,
            'x': x,
            'y': y
        }
        return self.record(InputType.MOUSE_BUTTON, device_id, data)

    def record_mouse_move(
        self,
        x: float,
        y: float,
        delta_x: float = 0.0,
        delta_y: float = 0.0,
        device_id: int = 0
    ) -> Optional[RecordedInput]:
        """Record a mouse movement.

        Args:
            x: Mouse X position
            y: Mouse Y position
            delta_x: X movement delta
            delta_y: Y movement delta
            device_id: Mouse device ID

        Returns:
            The recorded input, or None if not recording/deduplicated
        """
        data = {
            'x': x,
            'y': y,
            'delta_x': delta_x,
            'delta_y': delta_y
        }
        return self.record(InputType.MOUSE_MOVE, device_id, data)

    def record_mouse_scroll(
        self,
        x: float,
        y: float,
        scroll_x: float,
        scroll_y: float,
        device_id: int = 0
    ) -> Optional[RecordedInput]:
        """Record a mouse scroll.

        Args:
            x: Mouse X position
            y: Mouse Y position
            scroll_x: Horizontal scroll amount
            scroll_y: Vertical scroll amount
            device_id: Mouse device ID

        Returns:
            The recorded input, or None if not recording
        """
        data = {
            'x': x,
            'y': y,
            'scroll_x': scroll_x,
            'scroll_y': scroll_y
        }
        return self.record(InputType.MOUSE_SCROLL, device_id, data)

    def record_gamepad_button(
        self,
        button: Any,
        is_pressed: bool,
        device_id: int = 0
    ) -> Optional[RecordedInput]:
        """Record a gamepad button input.

        Args:
            button: Button identifier
            is_pressed: True if pressed, False if released
            device_id: Gamepad device ID

        Returns:
            The recorded input, or None if not recording
        """
        data = {
            'button': button.name if isinstance(button, Enum) else str(button),
            'pressed': is_pressed
        }
        return self.record(InputType.GAMEPAD_BUTTON, device_id, data)

    def record_gamepad_axis(
        self,
        axis: Any,
        value: float,
        device_id: int = 0
    ) -> Optional[RecordedInput]:
        """Record a gamepad axis input.

        Args:
            axis: Axis identifier
            value: Axis value (-1.0 to 1.0)
            device_id: Gamepad device ID

        Returns:
            The recorded input, or None if not recording
        """
        data = {
            'axis': axis.name if isinstance(axis, Enum) else str(axis),
            'value': max(-1.0, min(1.0, value))
        }
        return self.record(InputType.GAMEPAD_AXIS, device_id, data)

    def record_gamepad_trigger(
        self,
        trigger: Any,
        value: float,
        device_id: int = 0
    ) -> Optional[RecordedInput]:
        """Record a gamepad trigger input.

        Args:
            trigger: Trigger identifier
            value: Trigger value (0.0 to 1.0)
            device_id: Gamepad device ID

        Returns:
            The recorded input, or None if not recording
        """
        data = {
            'trigger': trigger.name if isinstance(trigger, Enum) else str(trigger),
            'value': max(0.0, min(1.0, value))
        }
        return self.record(InputType.GAMEPAD_TRIGGER, device_id, data)

    def record_touch(
        self,
        touch_id: int,
        phase: str,  # 'start', 'move', 'end'
        x: float,
        y: float,
        pressure: float = 1.0,
        device_id: int = 0
    ) -> Optional[RecordedInput]:
        """Record a touch input.

        Args:
            touch_id: Touch point identifier
            phase: Touch phase ('start', 'move', 'end')
            x: Touch X position
            y: Touch Y position
            pressure: Touch pressure (0.0 to 1.0)
            device_id: Touch device ID

        Returns:
            The recorded input, or None if not recording
        """
        type_map = {
            'start': InputType.TOUCH_START,
            'move': InputType.TOUCH_MOVE,
            'end': InputType.TOUCH_END
        }
        input_type = type_map.get(phase, InputType.TOUCH_MOVE)

        data = {
            'touch_id': touch_id,
            'x': x,
            'y': y,
            'pressure': max(0.0, min(1.0, pressure))
        }
        return self.record(input_type, device_id, data)

    def record_custom(
        self,
        name: str,
        data: dict[str, Any],
        device_id: int = 0
    ) -> Optional[RecordedInput]:
        """Record a custom input event.

        Args:
            name: Custom input name
            data: Custom input data
            device_id: Device ID

        Returns:
            The recorded input, or None if not recording
        """
        full_data = {'name': name, **data}
        return self.record(InputType.CUSTOM, device_id, full_data)

    def get_inputs_for_frame(self, frame: int) -> list[RecordedInput]:
        """Get all inputs for a specific frame.

        Args:
            frame: Frame number

        Returns:
            List of inputs for that frame
        """
        return [inp for inp in self._inputs if inp.frame == frame]

    def get_inputs_in_range(
        self,
        start_time: float,
        end_time: float
    ) -> list[RecordedInput]:
        """Get all inputs within a time range.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            List of inputs in the time range
        """
        return [
            inp for inp in self._inputs
            if start_time <= inp.timestamp <= end_time
        ]

    def get_inputs_by_type(self, input_type: InputType) -> list[RecordedInput]:
        """Get all inputs of a specific type.

        Args:
            input_type: Type of input to filter

        Returns:
            List of inputs of that type
        """
        return [inp for inp in self._inputs if inp.input_type == input_type]

    def iter_inputs(self) -> Iterator[RecordedInput]:
        """Iterate over all recorded inputs.

        Yields:
            Recorded inputs in order
        """
        yield from self._inputs

    def clear(self) -> None:
        """Clear all recorded inputs."""
        self._inputs.clear()
        self._stats = InputRecordingStats()
        self._input_hash = hashlib.sha256()

    def set_flush_callback(
        self,
        callback: Optional[Callable[[list[RecordedInput]], None]]
    ) -> None:
        """Set callback for auto-flush events.

        Args:
            callback: Function to call with flushed inputs
        """
        self._flush_callback = callback

    def serialize_all(self) -> bytes:
        """Serialize all recorded inputs to bytes.

        Returns:
            Serialized byte representation
        """
        parts = []
        for inp in self._inputs:
            parts.append(inp.serialize())

        # Add count header
        header = struct.pack('<I', len(self._inputs))
        return header + b''.join(parts)

    @classmethod
    def deserialize_all(cls, data: bytes) -> list[RecordedInput]:
        """Deserialize inputs from bytes.

        Args:
            data: Serialized byte data

        Returns:
            List of deserialized inputs
        """
        count = struct.unpack('<I', data[:4])[0]
        offset = 4
        inputs = []

        for _ in range(count):
            inp, offset = RecordedInput.deserialize(data, offset)
            inputs.append(inp)

        return inputs

    def _get_timestamp(self) -> float:
        """Get current timestamp relative to recording start."""
        if self._config.high_precision_timing:
            return time.perf_counter() - self._start_time
        return time.time() - self._start_time

    def _should_record_type(self, input_type: InputType) -> bool:
        """Check if input type should be recorded based on config."""
        type_config = {
            InputType.KEYBOARD: self._config.record_keyboard,
            InputType.MOUSE_BUTTON: self._config.record_mouse,
            InputType.MOUSE_MOVE: self._config.record_mouse,
            InputType.MOUSE_SCROLL: self._config.record_mouse,
            InputType.GAMEPAD_BUTTON: self._config.record_gamepad,
            InputType.GAMEPAD_AXIS: self._config.record_gamepad,
            InputType.GAMEPAD_TRIGGER: self._config.record_gamepad,
            InputType.TOUCH_START: self._config.record_touch,
            InputType.TOUCH_MOVE: self._config.record_touch,
            InputType.TOUCH_END: self._config.record_touch,
            InputType.CUSTOM: self._config.record_custom,
        }
        return type_config.get(input_type, True)

    def _auto_flush(self) -> None:
        """Perform auto-flush of input buffer."""
        if self._flush_callback and self._inputs:
            # Copy current inputs
            flushed = list(self._inputs)

            # Call callback
            self._flush_callback(flushed)

            # Update stats
            self._stats.bytes_recorded += sum(len(inp.serialize()) for inp in flushed)

        self._last_flush_time = self._get_timestamp()

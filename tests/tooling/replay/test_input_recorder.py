"""
Tests for input_recorder.py - Input capture and timestamps.
"""

import pytest
import time
from unittest.mock import MagicMock

from engine.tooling.replay.input_recorder import (
    InputRecorder,
    RecordedInput,
    InputRecordingConfig,
    InputRecordingStats,
    InputType,
)


class TestRecordedInput:
    """Tests for RecordedInput dataclass."""

    def test_create_recorded_input(self):
        """Test creating a recorded input."""
        inp = RecordedInput(
            input_type=InputType.KEYBOARD,
            timestamp=1.5,
            frame=90,
            device_id=0,
            data={'key': 'A', 'pressed': True}
        )
        assert inp.input_type == InputType.KEYBOARD
        assert inp.timestamp == 1.5
        assert inp.frame == 90
        assert inp.device_id == 0
        assert inp.data['key'] == 'A'

    def test_serialize_deserialize(self):
        """Test serialization and deserialization."""
        inp = RecordedInput(
            input_type=InputType.MOUSE_BUTTON,
            timestamp=2.5,
            frame=150,
            device_id=1,
            data={'button': 0, 'pressed': True, 'x': 100.0, 'y': 200.0}
        )
        serialized = inp.serialize()
        assert isinstance(serialized, bytes)

        deserialized, offset = RecordedInput.deserialize(serialized)
        assert deserialized.input_type == inp.input_type
        assert deserialized.timestamp == inp.timestamp
        assert deserialized.frame == inp.frame
        assert deserialized.device_id == inp.device_id
        assert deserialized.data['button'] == 0

    def test_matches_frame(self):
        """Test frame matching."""
        inp = RecordedInput(
            input_type=InputType.KEYBOARD,
            timestamp=1.0,
            frame=60,
            device_id=0,
            data={}
        )
        assert inp.matches_frame(60)
        assert not inp.matches_frame(61)

    def test_matches_time_range(self):
        """Test time range matching."""
        inp = RecordedInput(
            input_type=InputType.KEYBOARD,
            timestamp=5.0,
            frame=300,
            device_id=0,
            data={}
        )
        assert inp.matches_time_range(4.0, 6.0)
        assert inp.matches_time_range(5.0, 5.0)
        assert not inp.matches_time_range(6.0, 10.0)


class TestInputRecordingConfig:
    """Tests for InputRecordingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = InputRecordingConfig()
        assert config.high_precision_timing is True
        assert config.record_keyboard is True
        assert config.record_mouse is True
        assert config.record_gamepad is True
        assert config.record_touch is True
        assert config.max_buffer_size == 100000
        assert config.flush_interval == 5.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = InputRecordingConfig(
            high_precision_timing=False,
            record_keyboard=False,
            max_buffer_size=50000
        )
        assert config.high_precision_timing is False
        assert config.record_keyboard is False
        assert config.max_buffer_size == 50000


class TestInputRecordingStats:
    """Tests for InputRecordingStats."""

    def test_default_stats(self):
        """Test default statistics."""
        stats = InputRecordingStats()
        assert stats.total_inputs == 0
        assert stats.duration == 0.0
        assert stats.inputs_per_second == 0.0

    def test_duration_calculation(self):
        """Test duration calculation."""
        stats = InputRecordingStats(start_time=0.0, end_time=10.0)
        assert stats.duration == 10.0

    def test_inputs_per_second(self):
        """Test inputs per second calculation."""
        stats = InputRecordingStats(
            total_inputs=100,
            start_time=0.0,
            end_time=10.0
        )
        assert stats.inputs_per_second == 10.0

    def test_increment(self):
        """Test increment method."""
        stats = InputRecordingStats()
        stats.increment(InputType.KEYBOARD)
        stats.increment(InputType.KEYBOARD)
        stats.increment(InputType.MOUSE_BUTTON)

        assert stats.total_inputs == 3
        assert stats.inputs_by_type[InputType.KEYBOARD] == 2
        assert stats.inputs_by_type[InputType.MOUSE_BUTTON] == 1


class TestInputRecorder:
    """Tests for InputRecorder class."""

    def test_create_recorder(self):
        """Test creating an input recorder."""
        recorder = InputRecorder()
        assert not recorder.is_recording
        assert recorder.input_count == 0
        assert recorder.current_frame == 0

    def test_start_stop_recording(self):
        """Test starting and stopping recording."""
        recorder = InputRecorder()
        recorder.start()
        assert recorder.is_recording

        inputs = recorder.stop()
        assert not recorder.is_recording
        assert isinstance(inputs, list)

    def test_record_keyboard_input(self):
        """Test recording keyboard input."""
        recorder = InputRecorder()
        recorder.start()

        inp = recorder.record_keyboard('A', is_pressed=True)
        assert inp is not None
        assert inp.input_type == InputType.KEYBOARD
        assert inp.data['key'] == 'A'
        assert inp.data['pressed'] is True

        recorder.stop()

    def test_record_mouse_button(self):
        """Test recording mouse button input."""
        recorder = InputRecorder()
        recorder.start()

        inp = recorder.record_mouse_button(
            button=0,
            is_pressed=True,
            x=100.0,
            y=200.0
        )
        assert inp is not None
        assert inp.input_type == InputType.MOUSE_BUTTON
        assert inp.data['button'] == 0
        assert inp.data['x'] == 100.0

        recorder.stop()

    def test_record_mouse_move(self):
        """Test recording mouse movement."""
        recorder = InputRecorder()
        recorder.start()

        inp = recorder.record_mouse_move(x=150.0, y=250.0, delta_x=10.0, delta_y=-5.0)
        assert inp is not None
        assert inp.input_type == InputType.MOUSE_MOVE

        recorder.stop()

    def test_record_mouse_scroll(self):
        """Test recording mouse scroll."""
        recorder = InputRecorder()
        recorder.start()

        inp = recorder.record_mouse_scroll(x=100.0, y=100.0, scroll_x=0.0, scroll_y=3.0)
        assert inp is not None
        assert inp.input_type == InputType.MOUSE_SCROLL
        assert inp.data['scroll_y'] == 3.0

        recorder.stop()

    def test_record_gamepad_button(self):
        """Test recording gamepad button."""
        recorder = InputRecorder()
        recorder.start()

        inp = recorder.record_gamepad_button(button='A', is_pressed=True)
        assert inp is not None
        assert inp.input_type == InputType.GAMEPAD_BUTTON

        recorder.stop()

    def test_record_gamepad_axis(self):
        """Test recording gamepad axis."""
        recorder = InputRecorder()
        recorder.start()

        inp = recorder.record_gamepad_axis(axis='LEFT_X', value=0.75)
        assert inp is not None
        assert inp.input_type == InputType.GAMEPAD_AXIS
        assert inp.data['value'] == 0.75

        recorder.stop()

    def test_record_gamepad_trigger(self):
        """Test recording gamepad trigger."""
        recorder = InputRecorder()
        recorder.start()

        inp = recorder.record_gamepad_trigger(trigger='LEFT', value=0.5)
        assert inp is not None
        assert inp.input_type == InputType.GAMEPAD_TRIGGER

        recorder.stop()

    def test_record_touch(self):
        """Test recording touch input."""
        recorder = InputRecorder()
        recorder.start()

        inp = recorder.record_touch(touch_id=0, phase='start', x=200.0, y=400.0)
        assert inp is not None
        assert inp.input_type == InputType.TOUCH_START

        recorder.stop()

    def test_record_custom(self):
        """Test recording custom input."""
        recorder = InputRecorder()
        recorder.start()

        inp = recorder.record_custom(
            name='gesture',
            data={'type': 'swipe', 'direction': 'left'}
        )
        assert inp is not None
        assert inp.input_type == InputType.CUSTOM
        assert inp.data['name'] == 'gesture'

        recorder.stop()

    def test_not_recording_returns_none(self):
        """Test that recording when not started returns None."""
        recorder = InputRecorder()
        inp = recorder.record_keyboard('A', is_pressed=True)
        assert inp is None

    def test_pause_resume(self):
        """Test pausing and resuming recording."""
        recorder = InputRecorder()
        recorder.start()
        assert recorder.is_recording

        recorder.pause()
        assert not recorder.is_recording

        inp = recorder.record_keyboard('A', is_pressed=True)
        assert inp is None  # Should not record while paused

        recorder.resume()
        assert recorder.is_recording

        inp = recorder.record_keyboard('B', is_pressed=True)
        assert inp is not None

        recorder.stop()

    def test_advance_frame(self):
        """Test frame advancement."""
        recorder = InputRecorder()
        recorder.start()

        assert recorder.current_frame == 0
        recorder.advance_frame()
        assert recorder.current_frame == 1
        recorder.advance_frame()
        assert recorder.current_frame == 2

        recorder.stop()

    def test_get_inputs_for_frame(self):
        """Test getting inputs for specific frame."""
        recorder = InputRecorder()
        recorder.start()

        recorder.record_keyboard('A', is_pressed=True)
        recorder.advance_frame()
        recorder.record_keyboard('B', is_pressed=True)
        recorder.record_keyboard('C', is_pressed=True)
        recorder.advance_frame()

        frame_0_inputs = recorder.get_inputs_for_frame(0)
        frame_1_inputs = recorder.get_inputs_for_frame(1)

        assert len(frame_0_inputs) == 1
        assert len(frame_1_inputs) == 2

        recorder.stop()

    def test_get_inputs_by_type(self):
        """Test getting inputs by type."""
        recorder = InputRecorder()
        recorder.start()

        recorder.record_keyboard('A', is_pressed=True)
        recorder.record_mouse_button(0, True, 0.0, 0.0)
        recorder.record_keyboard('B', is_pressed=True)

        keyboard_inputs = recorder.get_inputs_by_type(InputType.KEYBOARD)
        mouse_inputs = recorder.get_inputs_by_type(InputType.MOUSE_BUTTON)

        assert len(keyboard_inputs) == 2
        assert len(mouse_inputs) == 1

        recorder.stop()

    def test_input_filtering(self):
        """Test input type filtering via config."""
        config = InputRecordingConfig(record_keyboard=False)
        recorder = InputRecorder(config)
        recorder.start()

        keyboard_inp = recorder.record_keyboard('A', is_pressed=True)
        mouse_inp = recorder.record_mouse_button(0, True, 0.0, 0.0)

        assert keyboard_inp is None  # Filtered out
        assert mouse_inp is not None

        recorder.stop()

    def test_custom_input_filter(self):
        """Test custom input filter function."""
        def my_filter(inp):
            # Only record pressed events
            return inp.data.get('pressed', False)

        config = InputRecordingConfig(input_filter=my_filter)
        recorder = InputRecorder(config)
        recorder.start()

        press = recorder.record_keyboard('A', is_pressed=True)
        release = recorder.record_keyboard('A', is_pressed=False)

        assert press is not None
        assert release is None  # Filtered out

        recorder.stop()

    def test_mouse_move_deduplication(self):
        """Test mouse move deduplication."""
        config = InputRecordingConfig(
            deduplicate_mouse_moves=True,
            mouse_move_threshold=0.1
        )
        recorder = InputRecorder(config)
        recorder.start()

        # First move should be recorded
        inp1 = recorder.record_mouse_move(0.0, 0.0)
        assert inp1 is not None

        # Immediate second move should be deduplicated
        inp2 = recorder.record_mouse_move(1.0, 1.0)
        assert inp2 is None

        # After waiting, move should be recorded
        time.sleep(0.15)
        inp3 = recorder.record_mouse_move(2.0, 2.0)
        assert inp3 is not None

        recorder.stop()

    def test_serialize_all(self):
        """Test serializing all inputs."""
        recorder = InputRecorder()
        recorder.start()

        recorder.record_keyboard('A', is_pressed=True)
        recorder.record_keyboard('B', is_pressed=True)
        recorder.record_mouse_button(0, True, 100.0, 200.0)

        serialized = recorder.serialize_all()
        assert isinstance(serialized, bytes)

        deserialized = InputRecorder.deserialize_all(serialized)
        assert len(deserialized) == 3

        recorder.stop()

    def test_clear(self):
        """Test clearing recorded inputs."""
        recorder = InputRecorder()
        recorder.start()

        recorder.record_keyboard('A', is_pressed=True)
        assert recorder.input_count == 1

        recorder.clear()
        assert recorder.input_count == 0

        recorder.stop()

    def test_iter_inputs(self):
        """Test iterating over inputs."""
        recorder = InputRecorder()
        recorder.start()

        recorder.record_keyboard('A', is_pressed=True)
        recorder.record_keyboard('B', is_pressed=True)

        inputs = list(recorder.iter_inputs())
        assert len(inputs) == 2

        recorder.stop()

    def test_input_hash(self):
        """Test input hash for verification."""
        recorder = InputRecorder()
        recorder.start()

        recorder.record_keyboard('A', is_pressed=True)
        hash1 = recorder.input_hash

        recorder.record_keyboard('B', is_pressed=True)
        hash2 = recorder.input_hash

        assert hash1 != hash2  # Hash should change with new inputs
        assert len(hash1) == 64  # SHA-256 hex length

        recorder.stop()

    def test_flush_callback(self):
        """Test flush callback functionality."""
        flushed_inputs = []

        def on_flush(inputs):
            flushed_inputs.extend(inputs)

        config = InputRecordingConfig(
            max_buffer_size=2,
            flush_interval=0.1
        )
        recorder = InputRecorder(config)
        recorder.set_flush_callback(on_flush)
        recorder.start()

        recorder.record_keyboard('A', is_pressed=True)
        recorder.record_keyboard('B', is_pressed=True)

        # Trigger flush via advance_frame
        time.sleep(0.15)
        recorder.advance_frame()

        assert len(flushed_inputs) > 0

        recorder.stop()

    def test_stats_tracking(self):
        """Test statistics tracking."""
        recorder = InputRecorder()
        recorder.start()

        recorder.record_keyboard('A', is_pressed=True)
        recorder.record_keyboard('B', is_pressed=True)
        recorder.record_mouse_button(0, True, 0.0, 0.0)

        stats = recorder.stats
        assert stats.total_inputs == 3
        assert stats.inputs_by_type[InputType.KEYBOARD] == 2
        assert stats.inputs_by_type[InputType.MOUSE_BUTTON] == 1

        recorder.stop()

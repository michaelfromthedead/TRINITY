"""Tests for event stream with Chrome Tracing format output.

Tests cover:
- EventRingBuffer: Pre-allocated ring buffer functionality
- EventStream: Central event collector
- ChromeTracingExporter: Chrome Tracing JSON format export
- BinaryTraceExporter: Binary format export/import
- GPU/CPU integration
- Frame scope helper
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
import threading
import time
from io import StringIO
from typing import List
from unittest.mock import Mock, MagicMock, patch

import pytest

from engine.debug.profiling.event_stream import (
    BinaryTraceExporter,
    ChromeTracingExporter,
    DEFAULT_BUFFER_SIZE,
    EventCategory,
    EventRingBuffer,
    EventScope,
    EventSlot,
    EventStream,
    EventType,
    FrameScope,
    MAX_ARGS_LENGTH,
    MAX_NAME_LENGTH,
    ProfileEvent,
    export_chrome_tracing,
    get_event_stream,
    initialize_event_stream,
    shutdown_event_stream,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def event_stream():
    """Create a fresh EventStream for testing."""
    EventStream.reset_instance()
    stream = EventStream.get_instance()
    stream.initialize(buffer_size=1024)
    yield stream
    stream.shutdown()
    EventStream.reset_instance()


@pytest.fixture
def ring_buffer():
    """Create a ring buffer for testing."""
    return EventRingBuffer(capacity=64)


@pytest.fixture
def sample_event():
    """Create a sample ProfileEvent."""
    return ProfileEvent(
        event_type=EventType.COMPLETE,
        name="test_event",
        category="test",
        timestamp_ns=1000000,
        duration_ns=500000,
        thread_id=12345,
        process_id=67890,
        args='{"key": "value"}',
        id=0,
        scope='t',
    )


# =============================================================================
# EVENT TYPE TESTS
# =============================================================================


class TestEventType:
    """Tests for EventType enum."""

    def test_begin_type(self):
        """Test BEGIN event type value."""
        assert EventType.BEGIN == ord('B')

    def test_end_type(self):
        """Test END event type value."""
        assert EventType.END == ord('E')

    def test_complete_type(self):
        """Test COMPLETE event type value."""
        assert EventType.COMPLETE == ord('X')

    def test_instant_type(self):
        """Test INSTANT event type value."""
        assert EventType.INSTANT == ord('i')

    def test_counter_type(self):
        """Test COUNTER event type value."""
        assert EventType.COUNTER == ord('C')

    def test_async_begin_type(self):
        """Test ASYNC_BEGIN event type value."""
        assert EventType.ASYNC_BEGIN == ord('b')

    def test_metadata_type(self):
        """Test METADATA event type value."""
        assert EventType.METADATA == ord('M')


class TestEventScope:
    """Tests for EventScope enum."""

    def test_global_scope(self):
        """Test GLOBAL scope value."""
        assert EventScope.GLOBAL.value == 'g'

    def test_process_scope(self):
        """Test PROCESS scope value."""
        assert EventScope.PROCESS.value == 'p'

    def test_thread_scope(self):
        """Test THREAD scope value."""
        assert EventScope.THREAD.value == 't'


class TestEventCategory:
    """Tests for EventCategory enum."""

    def test_cpu_category(self):
        """Test CPU category value."""
        assert EventCategory.CPU.value == "cpu"

    def test_gpu_category(self):
        """Test GPU category value."""
        assert EventCategory.GPU.value == "gpu"

    def test_render_category(self):
        """Test RENDER category value."""
        assert EventCategory.RENDER.value == "render"


# =============================================================================
# PROFILE EVENT TESTS
# =============================================================================


class TestProfileEvent:
    """Tests for ProfileEvent namedtuple."""

    def test_create_event(self, sample_event):
        """Test creating a ProfileEvent."""
        assert sample_event.event_type == EventType.COMPLETE
        assert sample_event.name == "test_event"
        assert sample_event.category == "test"
        assert sample_event.timestamp_ns == 1000000
        assert sample_event.duration_ns == 500000

    def test_event_immutable(self, sample_event):
        """Test that ProfileEvent is immutable."""
        with pytest.raises(AttributeError):
            sample_event.name = "modified"

    def test_event_equality(self):
        """Test ProfileEvent equality."""
        event1 = ProfileEvent(EventType.BEGIN, "test", "cat", 100, 0, 1, 2, "", 0, "t")
        event2 = ProfileEvent(EventType.BEGIN, "test", "cat", 100, 0, 1, 2, "", 0, "t")
        assert event1 == event2

    def test_event_with_args(self):
        """Test ProfileEvent with JSON arguments."""
        event = ProfileEvent(
            EventType.COMPLETE,
            "test",
            "cat",
            100,
            50,
            1,
            2,
            '{"count": 42}',
            0,
            "t",
        )
        args = json.loads(event.args)
        assert args["count"] == 42


# =============================================================================
# EVENT SLOT TESTS
# =============================================================================


class TestEventSlot:
    """Tests for EventSlot pre-allocated storage."""

    def test_slot_initial_state(self):
        """Test initial slot state."""
        slot = EventSlot()
        assert not slot.valid
        assert slot.event_type == 0
        assert slot.timestamp_ns == 0

    def test_slot_set_and_get(self, sample_event):
        """Test setting and getting slot data."""
        slot = EventSlot()
        slot.set(sample_event)

        assert slot.valid
        assert slot.event_type == EventType.COMPLETE
        assert slot.timestamp_ns == 1000000

        retrieved = slot.get()
        assert retrieved.name == "test_event"
        assert retrieved.category == "test"
        assert retrieved.duration_ns == 500000

    def test_slot_clear(self, sample_event):
        """Test clearing a slot."""
        slot = EventSlot()
        slot.set(sample_event)
        assert slot.valid

        slot.clear()
        assert not slot.valid

    def test_slot_name_truncation(self):
        """Test that long names are truncated."""
        long_name = "a" * 100
        event = ProfileEvent(
            EventType.BEGIN,
            long_name,
            "cat",
            100,
            0,
            1,
            2,
            "",
            0,
            "t",
        )

        slot = EventSlot()
        slot.set(event)
        retrieved = slot.get()

        assert len(retrieved.name) <= MAX_NAME_LENGTH

    def test_slot_args_truncation(self):
        """Test that long args are truncated."""
        long_args = '{"data": "' + "x" * 500 + '"}'
        event = ProfileEvent(
            EventType.COMPLETE,
            "test",
            "cat",
            100,
            50,
            1,
            2,
            long_args,
            0,
            "t",
        )

        slot = EventSlot()
        slot.set(event)
        retrieved = slot.get()

        assert len(retrieved.args) <= MAX_ARGS_LENGTH


# =============================================================================
# EVENT RING BUFFER TESTS
# =============================================================================


class TestEventRingBuffer:
    """Tests for EventRingBuffer."""

    def test_create_buffer(self, ring_buffer):
        """Test creating a ring buffer."""
        assert ring_buffer.capacity >= 64  # Rounded to power of 2
        assert ring_buffer.count == 0

    def test_push_event(self, ring_buffer, sample_event):
        """Test pushing an event."""
        result = ring_buffer.push(sample_event)
        assert result is True
        assert ring_buffer.count == 1

    def test_pop_event(self, ring_buffer, sample_event):
        """Test popping an event."""
        ring_buffer.push(sample_event)
        popped = ring_buffer.pop()

        assert popped is not None
        assert popped.name == "test_event"
        assert ring_buffer.count == 0

    def test_pop_empty_buffer(self, ring_buffer):
        """Test popping from empty buffer."""
        result = ring_buffer.pop()
        assert result is None

    def test_peek_event(self, ring_buffer, sample_event):
        """Test peeking at an event."""
        ring_buffer.push(sample_event)
        peeked = ring_buffer.peek(0)

        assert peeked is not None
        assert peeked.name == "test_event"
        assert ring_buffer.count == 1  # Still in buffer

    def test_peek_out_of_range(self, ring_buffer, sample_event):
        """Test peeking at invalid index."""
        ring_buffer.push(sample_event)
        assert ring_buffer.peek(-1) is None
        assert ring_buffer.peek(1) is None

    def test_drain_buffer(self, ring_buffer):
        """Test draining all events."""
        for i in range(5):
            event = ProfileEvent(
                EventType.COMPLETE,
                f"event_{i}",
                "test",
                i * 1000,
                100,
                1,
                2,
                "",
                0,
                "t",
            )
            ring_buffer.push(event)

        events = ring_buffer.drain()
        assert len(events) == 5
        assert ring_buffer.count == 0
        assert events[0].name == "event_0"
        assert events[4].name == "event_4"

    def test_buffer_overflow(self):
        """Test buffer overflow behavior."""
        small_buffer = EventRingBuffer(capacity=4)

        for i in range(8):
            event = ProfileEvent(
                EventType.COMPLETE,
                f"event_{i}",
                "test",
                i * 1000,
                100,
                1,
                2,
                "",
                0,
                "t",
            )
            small_buffer.push(event)

        # Buffer should contain last 4 events
        assert small_buffer.count == 4
        assert small_buffer.overflow_count == 4

        events = small_buffer.drain()
        # Should have events 4-7
        assert events[0].name == "event_4"

    def test_iter_events(self, ring_buffer):
        """Test iterating over events."""
        for i in range(3):
            event = ProfileEvent(
                EventType.COMPLETE,
                f"event_{i}",
                "test",
                i * 1000,
                100,
                1,
                2,
                "",
                0,
                "t",
            )
            ring_buffer.push(event)

        names = [e.name for e in ring_buffer.iter_events()]
        assert names == ["event_0", "event_1", "event_2"]
        assert ring_buffer.count == 3  # Events not removed

    def test_clear_buffer(self, ring_buffer, sample_event):
        """Test clearing the buffer."""
        ring_buffer.push(sample_event)
        ring_buffer.push(sample_event)
        assert ring_buffer.count == 2

        ring_buffer.clear()
        assert ring_buffer.count == 0
        assert ring_buffer.overflow_count == 0

    def test_get_statistics(self, ring_buffer, sample_event):
        """Test getting buffer statistics."""
        ring_buffer.push(sample_event)
        stats = ring_buffer.get_statistics()

        assert stats["capacity"] >= 64
        assert stats["count"] == 1
        assert stats["overflow_count"] == 0
        assert 0 < stats["utilization"] <= 1

    def test_thread_safety(self, ring_buffer):
        """Test thread-safe operations."""
        results = []

        def producer():
            for i in range(100):
                event = ProfileEvent(
                    EventType.COMPLETE,
                    f"event_{i}",
                    "test",
                    i * 1000,
                    100,
                    threading.current_thread().ident or 0,
                    os.getpid(),
                    "",
                    0,
                    "t",
                )
                ring_buffer.push(event)

        def consumer():
            count = 0
            for _ in range(50):
                if ring_buffer.pop() is not None:
                    count += 1
                time.sleep(0.001)
            results.append(count)

        threads = [
            threading.Thread(target=producer),
            threading.Thread(target=consumer),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Some events should have been consumed
        assert len(results) == 1


# =============================================================================
# EVENT STREAM TESTS
# =============================================================================


class TestEventStream:
    """Tests for EventStream central collector."""

    def test_singleton_pattern(self):
        """Test singleton pattern."""
        EventStream.reset_instance()
        stream1 = EventStream.get_instance()
        stream2 = EventStream.get_instance()
        assert stream1 is stream2
        EventStream.reset_instance()

    def test_initialize(self, event_stream):
        """Test stream initialization."""
        assert event_stream.initialized
        assert event_stream.buffer is not None

    def test_enabled_property(self, event_stream):
        """Test enabled property."""
        assert event_stream.enabled
        event_stream.enabled = False
        assert not event_stream.enabled
        event_stream.enabled = True

    def test_record_complete_event(self, event_stream):
        """Test recording a complete event."""
        event_stream.complete(
            "test_operation",
            start_ns=0,
            duration_ns=1000000,
            category=EventCategory.CPU,
        )

        events = event_stream.get_events()
        assert len(events) == 1
        assert events[0].name == "test_operation"
        assert events[0].event_type == EventType.COMPLETE

    def test_record_begin_end(self, event_stream):
        """Test recording begin/end pair."""
        event_stream.begin("operation", EventCategory.CPU)
        time.sleep(0.001)
        event_stream.end("operation", EventCategory.CPU)

        events = event_stream.get_events()
        assert len(events) == 2
        assert events[0].event_type == EventType.BEGIN
        assert events[1].event_type == EventType.END

    def test_instant_event(self, event_stream):
        """Test recording instant event."""
        event_stream.instant(
            "checkpoint",
            args={"state": "ready"},
            category=EventCategory.SYSTEM,
        )

        events = event_stream.get_events()
        assert len(events) == 1
        assert events[0].event_type == EventType.INSTANT

    def test_counter_event(self, event_stream):
        """Test recording counter event."""
        event_stream.counter("fps", 60, EventCategory.FRAME)

        events = event_stream.get_events()
        assert len(events) == 1
        assert events[0].event_type == EventType.COUNTER
        args = json.loads(events[0].args)
        assert args["fps"] == 60

    def test_counter_with_dict(self, event_stream):
        """Test counter with dictionary value."""
        event_stream.counter(
            "memory",
            {"heap": 1024, "stack": 256},
            EventCategory.MEMORY,
        )

        events = event_stream.get_events()
        args = json.loads(events[0].args)
        assert args["heap"] == 1024
        assert args["stack"] == 256

    def test_scope_context_manager(self, event_stream):
        """Test scope context manager."""
        with event_stream.scope("render_frame", EventCategory.RENDER):
            time.sleep(0.001)

        events = event_stream.get_events()
        assert len(events) == 2
        assert events[0].event_type == EventType.BEGIN
        assert events[1].event_type == EventType.END

    def test_complete_scope_context_manager(self, event_stream):
        """Test complete_scope context manager."""
        with event_stream.complete_scope("process", EventCategory.CPU):
            time.sleep(0.001)

        events = event_stream.get_events()
        assert len(events) == 1
        assert events[0].event_type == EventType.COMPLETE
        assert events[0].duration_ns > 0

    def test_nested_scopes(self, event_stream):
        """Test nested scope recording."""
        with event_stream.scope("outer", EventCategory.CPU):
            with event_stream.scope("inner", EventCategory.CPU):
                time.sleep(0.001)

        events = event_stream.get_events()
        assert len(events) == 4
        names = [e.name for e in events]
        assert names == ["outer", "inner", "inner", "outer"]

    def test_disabled_stream(self, event_stream):
        """Test that disabled stream doesn't record."""
        event_stream.enabled = False
        event_stream.complete("test", 0, 1000)

        events = event_stream.get_events()
        assert len(events) == 0

    def test_event_callback(self, event_stream):
        """Test event callbacks."""
        received = []

        def callback(event):
            received.append(event)

        event_stream.add_callback(callback)
        event_stream.instant("test")

        assert len(received) == 1
        assert received[0].name == "test"

    def test_remove_callback(self, event_stream):
        """Test removing callback."""
        received = []

        def callback(event):
            received.append(event)

        event_stream.add_callback(callback)
        event_stream.instant("first")
        event_stream.remove_callback(callback)
        event_stream.instant("second")

        assert len(received) == 1

    def test_drain_events(self, event_stream):
        """Test draining events."""
        event_stream.instant("event1")
        event_stream.instant("event2")

        drained = event_stream.drain_events()
        assert len(drained) == 2

        remaining = event_stream.get_events()
        assert len(remaining) == 0

    def test_get_statistics(self, event_stream):
        """Test getting stream statistics."""
        event_stream.instant("test")
        stats = event_stream.get_statistics()

        assert stats["enabled"]
        assert stats["initialized"]
        assert "buffer" in stats
        assert stats["buffer"]["count"] == 1

    def test_category_as_string(self, event_stream):
        """Test using string as category."""
        event_stream.instant("test", category="custom_category")

        events = event_stream.get_events()
        assert events[0].category == "custom_category"

    def test_event_with_args(self, event_stream):
        """Test event with arguments."""
        event_stream.instant(
            "checkpoint",
            args={"level": 5, "score": 1000},
        )

        events = event_stream.get_events()
        args = json.loads(events[0].args)
        assert args["level"] == 5
        assert args["score"] == 1000


# =============================================================================
# GPU INTEGRATION TESTS
# =============================================================================


class TestGPUIntegration:
    """Tests for GPU profiler integration."""

    def test_import_gpu_results(self, event_stream):
        """Test importing GPU timestamp results."""
        # Mock TimestampResult
        from collections import namedtuple
        TimestampResult = namedtuple(
            'TimestampResult',
            ['pass_name', 'start_ns', 'end_ns', 'duration_ns'],
        )

        results = [
            TimestampResult("shadow_pass", 1000000, 1500000, 500000),
            TimestampResult("forward_pass", 1500000, 2500000, 1000000),
        ]

        count = event_stream.import_gpu_results(results, frame_index=1)
        assert count == 2

        events = event_stream.get_events()
        assert len(events) == 2
        assert events[0].name == "shadow_pass"
        assert events[0].category == "gpu"

    def test_import_gpu_event(self, event_stream):
        """Test importing single GPU event."""
        from engine.debug.profiling.gpu import GPUPassType

        # Mock GPUTimestampEvent
        class MockGPUEvent:
            pass_name = "deferred_lighting"
            pass_type = GPUPassType.LIGHTING
            frame_index = 5
            start_ns = 2000000
            end_ns = 3000000
            duration_ns = 1000000

        event_stream.import_gpu_event(MockGPUEvent())

        events = event_stream.get_events()
        assert len(events) == 1
        assert events[0].name == "deferred_lighting"
        args = json.loads(events[0].args)
        assert args["frame"] == 5


# =============================================================================
# CPU INTEGRATION TESTS
# =============================================================================


class TestCPUIntegration:
    """Tests for CPU profiler integration."""

    def test_import_cpu_samples(self, event_stream):
        """Test importing CPU profile samples."""
        from engine.debug.profiling.cpu import ProfileSample

        # Create sample hierarchy
        parent = ProfileSample(
            name="update",
            start_ns=1000000,
            end_ns=3000000,
        )
        child = ProfileSample(
            name="physics",
            start_ns=1500000,
            end_ns=2500000,
            parent=parent,
        )
        parent.children.append(child)

        count = event_stream.import_cpu_samples([parent])
        assert count == 2

        events = event_stream.get_events()
        assert len(events) == 2
        names = {e.name for e in events}
        assert "update" in names
        assert "physics" in names


# =============================================================================
# CHROME TRACING EXPORTER TESTS
# =============================================================================


class TestChromeTracingExporter:
    """Tests for ChromeTracingExporter."""

    def test_export_empty_stream(self, event_stream):
        """Test exporting empty stream."""
        exporter = ChromeTracingExporter(event_stream)
        output = exporter.export(include_metadata=False)

        data = json.loads(output)
        assert "traceEvents" in data
        assert len(data["traceEvents"]) == 0

    def test_export_with_events(self, event_stream):
        """Test exporting stream with events."""
        event_stream.complete("test_op", 0, 1000000, EventCategory.CPU)

        exporter = ChromeTracingExporter(event_stream)
        output = exporter.export(include_metadata=False)

        data = json.loads(output)
        assert len(data["traceEvents"]) == 1
        event = data["traceEvents"][0]
        assert event["name"] == "test_op"
        assert event["ph"] == "X"  # Complete event
        assert event["cat"] == "cpu"

    def test_export_with_metadata(self, event_stream):
        """Test exporting with metadata."""
        event_stream.instant("test")

        exporter = ChromeTracingExporter(event_stream, process_name="TestProcess")
        output = exporter.export(include_metadata=True)

        data = json.loads(output)
        # Should have metadata + event
        assert len(data["traceEvents"]) >= 2

        # Find process name metadata
        process_meta = None
        for event in data["traceEvents"]:
            if event.get("name") == "process_name":
                process_meta = event
                break

        assert process_meta is not None
        assert process_meta["args"]["name"] == "TestProcess"

    def test_export_formatted(self, event_stream):
        """Test formatted export."""
        event_stream.instant("test")

        exporter = ChromeTracingExporter(event_stream)
        output = exporter.export_formatted(include_metadata=False)

        # Formatted output should have indentation
        assert "\n" in output
        data = json.loads(output)
        assert len(data["traceEvents"]) == 1

    def test_export_to_file(self, event_stream):
        """Test exporting to file."""
        event_stream.instant("file_test")
        event_stream.complete("operation", 0, 500000)

        exporter = ChromeTracingExporter(event_stream)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            count = exporter.export_to_file(path, include_metadata=False)
            assert count == 2

            with open(path) as f:
                data = json.load(f)
            assert len(data["traceEvents"]) == 2
        finally:
            os.unlink(path)

    def test_export_streaming(self, event_stream):
        """Test streaming export."""
        for i in range(5):
            event_stream.instant(f"event_{i}")

        exporter = ChromeTracingExporter(event_stream)
        output = StringIO()

        count = exporter.export_streaming(output, include_metadata=False)
        assert count == 5

        output.seek(0)
        data = json.load(output)
        assert len(data["traceEvents"]) == 5

    def test_drain_and_export(self, event_stream):
        """Test drain and export."""
        event_stream.instant("drain_test")

        exporter = ChromeTracingExporter(event_stream)
        output = exporter.drain_and_export(include_metadata=False)

        data = json.loads(output)
        assert len(data["traceEvents"]) == 1

        # Stream should be empty
        remaining = event_stream.get_events()
        assert len(remaining) == 0

    def test_event_duration_conversion(self, event_stream):
        """Test duration is converted to microseconds."""
        # 1 millisecond = 1,000,000 nanoseconds
        event_stream.complete("test", 0, 1000000)

        exporter = ChromeTracingExporter(event_stream)
        output = exporter.export(include_metadata=False)

        data = json.loads(output)
        event = data["traceEvents"][0]
        # Duration should be in microseconds (1000)
        assert event["dur"] == 1000

    def test_instant_event_scope(self, event_stream):
        """Test instant event scope is exported."""
        event_stream.instant("marker", scope=EventScope.PROCESS)

        exporter = ChromeTracingExporter(event_stream)
        output = exporter.export(include_metadata=False)

        data = json.loads(output)
        event = data["traceEvents"][0]
        assert event["s"] == "p"  # Process scope

    def test_event_args_exported(self, event_stream):
        """Test event arguments are exported."""
        event_stream.instant("test", args={"key": "value", "count": 42})

        exporter = ChromeTracingExporter(event_stream)
        output = exporter.export(include_metadata=False)

        data = json.loads(output)
        args = data["traceEvents"][0]["args"]
        assert args["key"] == "value"
        assert args["count"] == 42


# =============================================================================
# BINARY EXPORTER TESTS
# =============================================================================


class TestBinaryTraceExporter:
    """Tests for BinaryTraceExporter."""

    def test_export_and_load(self, event_stream):
        """Test exporting and loading binary format."""
        event_stream.complete("op1", 1000, 500, EventCategory.CPU)
        event_stream.instant("marker", args={"frame": 1})
        event_stream.counter("fps", 60)

        exporter = BinaryTraceExporter(event_stream)

        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            path = f.name

        try:
            count = exporter.export_to_file(path)
            assert count == 3

            loaded = BinaryTraceExporter.load_from_file(path)
            assert len(loaded) == 3
            assert loaded[0].name == "op1"
            assert loaded[1].name == "marker"
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_data(self, event_stream):
        """Test that roundtrip preserves all event data."""
        original_event = ProfileEvent(
            event_type=EventType.COMPLETE,
            name="roundtrip_test",
            category="custom",
            timestamp_ns=123456789,
            duration_ns=987654,
            thread_id=12345,
            process_id=67890,
            args='{"key": "value"}',
            id=42,
            scope="p",
        )

        event_stream.buffer.push(original_event)
        exporter = BinaryTraceExporter(event_stream)

        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            path = f.name

        try:
            exporter.export_to_file(path)
            loaded = BinaryTraceExporter.load_from_file(path)

            assert len(loaded) == 1
            loaded_event = loaded[0]

            assert loaded_event.name == original_event.name
            assert loaded_event.category == original_event.category
            assert loaded_event.timestamp_ns == original_event.timestamp_ns
            assert loaded_event.duration_ns == original_event.duration_ns
            assert loaded_event.thread_id == original_event.thread_id
            assert loaded_event.process_id == original_event.process_id
            assert loaded_event.args == original_event.args
            assert loaded_event.id == original_event.id
        finally:
            os.unlink(path)


# =============================================================================
# FRAME SCOPE TESTS
# =============================================================================


class TestFrameScope:
    """Tests for FrameScope helper."""

    def test_frame_scope_basic(self, event_stream):
        """Test basic frame scope usage."""
        frame_scope = FrameScope(event_stream)

        with frame_scope.frame() as frame_num:
            assert frame_num == 0
            time.sleep(0.001)

        assert frame_scope.frame_number == 1

    def test_frame_scope_multiple_frames(self, event_stream):
        """Test multiple frames."""
        frame_scope = FrameScope(event_stream)

        for expected in range(3):
            with frame_scope.frame() as frame_num:
                assert frame_num == expected

        assert frame_scope.frame_number == 3

    def test_frame_scope_records_events(self, event_stream):
        """Test that frame scope records events."""
        frame_scope = FrameScope(event_stream, name="game_frame")

        with frame_scope.frame():
            time.sleep(0.001)

        events = event_stream.get_events()

        # Should have: instant (frame_start), complete (frame), counter (frame_stats)
        assert len(events) >= 2

        # Find the complete event
        complete_events = [e for e in events if e.event_type == EventType.COMPLETE]
        assert len(complete_events) == 1
        assert complete_events[0].name == "game_frame"

    def test_frame_scope_counter_values(self, event_stream):
        """Test frame scope counter values."""
        frame_scope = FrameScope(event_stream)

        with frame_scope.frame():
            time.sleep(0.005)  # ~5ms

        events = event_stream.get_events()
        counter_events = [e for e in events if e.event_type == EventType.COUNTER]

        assert len(counter_events) == 1
        args = json.loads(counter_events[0].args)
        assert "frame_time_ms" in args
        assert "fps" in args
        assert args["frame_time_ms"] > 0


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_event_stream(self):
        """Test get_event_stream function."""
        EventStream.reset_instance()
        stream = get_event_stream()
        assert stream is EventStream.get_instance()
        EventStream.reset_instance()

    def test_initialize_event_stream(self):
        """Test initialize_event_stream function."""
        EventStream.reset_instance()
        stream = initialize_event_stream(buffer_size=512)
        assert stream.initialized
        assert stream.buffer.capacity >= 512
        EventStream.reset_instance()

    def test_shutdown_event_stream(self):
        """Test shutdown_event_stream function."""
        EventStream.reset_instance()
        stream = initialize_event_stream()
        assert stream.initialized
        shutdown_event_stream()
        assert not stream.initialized
        EventStream.reset_instance()

    def test_export_chrome_tracing(self):
        """Test export_chrome_tracing function."""
        EventStream.reset_instance()
        stream = initialize_event_stream()
        stream.instant("export_test")

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name

        try:
            count = export_chrome_tracing(path)
            assert count >= 1

            with open(path) as f:
                data = json.load(f)
            assert "traceEvents" in data
        finally:
            os.unlink(path)
            EventStream.reset_instance()


# =============================================================================
# THREAD SAFETY TESTS
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_recording(self, event_stream):
        """Test concurrent event recording."""
        results = []

        def worker(worker_id: int):
            for i in range(50):
                event_stream.instant(
                    f"worker_{worker_id}_event_{i}",
                    args={"worker": worker_id, "index": i},
                )
            results.append(worker_id)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 4
        events = event_stream.get_events()
        assert len(events) == 200  # 4 workers * 50 events

    def test_concurrent_scope(self, event_stream):
        """Test concurrent scope usage."""
        def worker(worker_id: int):
            for i in range(20):
                with event_stream.complete_scope(
                    f"worker_{worker_id}_scope_{i}",
                    EventCategory.CPU,
                ):
                    time.sleep(0.0001)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        events = event_stream.get_events()
        assert len(events) == 80  # 4 workers * 20 scopes


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_uninitialized_stream(self):
        """Test operations on uninitialized stream."""
        EventStream.reset_instance()
        stream = EventStream.get_instance()

        # Should not raise
        stream.instant("test")
        stream.complete("test", 0, 1000)

        events = stream.get_events()
        assert len(events) == 0

        EventStream.reset_instance()

    def test_callback_exception_handling(self, event_stream):
        """Test that callback exceptions don't stop recording."""
        def bad_callback(event):
            raise ValueError("Callback error")

        event_stream.add_callback(bad_callback)

        # Should not raise
        event_stream.instant("test")

        events = event_stream.get_events()
        assert len(events) == 1

    def test_empty_event_name(self, event_stream):
        """Test event with empty name."""
        event_stream.instant("")

        events = event_stream.get_events()
        assert len(events) == 1
        assert events[0].name == ""

    def test_special_characters_in_name(self, event_stream):
        """Test event with special characters."""
        event_stream.instant("test::namespace/method<T>")

        events = event_stream.get_events()
        assert len(events) == 1
        assert events[0].name == "test::namespace/method<T>"

    def test_unicode_in_args(self, event_stream):
        """Test event with unicode in arguments."""
        event_stream.instant(
            "unicode_test",
            args={"message": "Hello, World!"},
        )

        events = event_stream.get_events()
        args = json.loads(events[0].args)
        assert args["message"] == "Hello, World!"

    def test_very_large_args(self, event_stream):
        """Test event with very large arguments."""
        large_args = {"data": "x" * 10000}
        event_stream.instant("large_args", args=large_args)

        events = event_stream.get_events()
        assert len(events) == 1
        # Args should be truncated
        assert len(events[0].args) <= MAX_ARGS_LENGTH


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestPerformance:
    """Performance-related tests."""

    def test_no_allocation_in_push(self, ring_buffer, sample_event):
        """Test that push doesn't allocate (uses pre-allocated slots)."""
        # Warm up - fill to capacity
        for _ in range(ring_buffer.capacity):
            ring_buffer.push(sample_event)

        ring_buffer.clear()

        # Push should reuse pre-allocated slots
        # Push capacity events (64) to fill without overflow
        for _ in range(ring_buffer.capacity):
            ring_buffer.push(sample_event)

        assert ring_buffer.count == ring_buffer.capacity

    def test_high_throughput_recording(self, event_stream):
        """Test high-throughput event recording."""
        start = time.perf_counter()

        for i in range(10000):
            event_stream.instant(f"event_{i}")

        elapsed = time.perf_counter() - start

        # Should be able to record at least 100k events/second
        events_per_second = 10000 / elapsed
        assert events_per_second > 10000, f"Too slow: {events_per_second:.0f} events/s"

    def test_export_performance(self, event_stream):
        """Test export performance with many events."""
        for i in range(1000):
            event_stream.instant(f"event_{i}")

        exporter = ChromeTracingExporter(event_stream)

        start = time.perf_counter()
        output = exporter.export(include_metadata=False)
        elapsed = time.perf_counter() - start

        # Should export 1000 events in under 100ms
        assert elapsed < 0.1, f"Export too slow: {elapsed:.3f}s"

        data = json.loads(output)
        assert len(data["traceEvents"]) == 1000

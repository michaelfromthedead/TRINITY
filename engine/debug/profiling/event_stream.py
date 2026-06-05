"""Event Stream with Chrome Tracing Format Output.

Provides centralized event collection and export for profiling data from
all sources (CPU, GPU, memory, network) with Chrome Tracing JSON format
output compatible with chrome://tracing viewer.

Key Components:
    - EventRingBuffer: Pre-allocated ring buffer for allocation-free recording
    - EventStream: Central event collector with multi-source integration
    - ChromeTracingExporter: Chrome Tracing JSON format export
    - ProfileEvent: Core event data structure

Chrome Tracing Format:
    The exported JSON follows the Chrome Tracing Event Format specification.
    Events can be viewed in chrome://tracing or other compatible viewers.

    Event types supported:
    - 'B'/'E': Begin/End duration events (paired)
    - 'X': Complete events (start + duration)
    - 'i': Instant events (markers)
    - 'C': Counter events (value over time)

Example:
    stream = EventStream.get_instance()
    stream.initialize(buffer_size=65536)

    # Record events
    with stream.scope("frame_update"):
        with stream.scope("physics"):
            physics.step()

    # Record instant event
    stream.instant("checkpoint", {"state": "ready"})

    # Record counter
    stream.counter("fps", 60)

    # Export to Chrome Tracing format
    exporter = ChromeTracingExporter(stream)
    json_data = exporter.export()

    with open("trace.json", "w") as f:
        f.write(json_data)
"""

from __future__ import annotations

import json
import os
import struct
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    TextIO,
    Tuple,
    Union,
)

from engine.debug.profiling import config as profiling_config

if TYPE_CHECKING:
    from engine.debug.profiling.gpu_timestamps import (
        GPUTimestampEvent,
        GPUTimestampProfiler,
        TimestampResult,
    )
    from engine.debug.profiling.cpu import CPUProfiler, ProfileSample


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default ring buffer capacity (number of events)
DEFAULT_BUFFER_SIZE = 65536

# Maximum string length for event names (for fixed-size storage)
MAX_NAME_LENGTH = 64

# Maximum size for event arguments JSON
MAX_ARGS_LENGTH = 256

# Nanoseconds to microseconds conversion
NS_TO_US = 1000


# =============================================================================
# ENUMS
# =============================================================================


class EventType(IntEnum):
    """Chrome Tracing event types."""

    # Duration events
    BEGIN = ord('B')  # Begin duration
    END = ord('E')    # End duration
    COMPLETE = ord('X')  # Complete (start + duration)

    # Instant events
    INSTANT = ord('i')  # Instant marker

    # Counter events
    COUNTER = ord('C')  # Counter value

    # Async events
    ASYNC_BEGIN = ord('b')
    ASYNC_INSTANT = ord('n')
    ASYNC_END = ord('e')

    # Flow events
    FLOW_START = ord('s')
    FLOW_STEP = ord('t')
    FLOW_END = ord('f')

    # Object events
    OBJECT_CREATE = ord('N')
    OBJECT_SNAPSHOT = ord('O')
    OBJECT_DESTROY = ord('D')

    # Metadata
    METADATA = ord('M')

    # Memory dump
    MEMORY_DUMP_GLOBAL = ord('V')
    MEMORY_DUMP_PROCESS = ord('v')


class EventScope(Enum):
    """Scope for instant events."""

    GLOBAL = 'g'     # Global scope
    PROCESS = 'p'    # Process scope
    THREAD = 't'     # Thread scope (default)


class EventCategory(Enum):
    """Event categories for filtering."""

    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    NETWORK = "network"
    RENDER = "render"
    PHYSICS = "physics"
    AUDIO = "audio"
    AI = "ai"
    IO = "io"
    CUSTOM = "custom"
    FRAME = "frame"
    SYSTEM = "system"


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class ProfileEvent(NamedTuple):
    """A profiling event for the event stream.

    This is a fixed-size structure optimized for storage in a ring buffer.
    All timestamps are in nanoseconds.

    Attributes:
        event_type: Type of event (BEGIN, END, COMPLETE, etc.)
        name: Event name (truncated to MAX_NAME_LENGTH)
        category: Event category for filtering
        timestamp_ns: Event timestamp in nanoseconds
        duration_ns: Duration in nanoseconds (for COMPLETE events)
        thread_id: Thread that generated the event
        process_id: Process that generated the event
        args: Optional JSON-encoded arguments
        id: Optional ID for async/flow events
        scope: Scope for instant events
    """

    event_type: int  # EventType value
    name: str
    category: str
    timestamp_ns: int
    duration_ns: int
    thread_id: int
    process_id: int
    args: str  # JSON-encoded arguments, empty string if none
    id: int  # For async/flow events, 0 if unused
    scope: str  # EventScope value


@dataclass
class EventSlot:
    """Pre-allocated slot in the ring buffer.

    Uses fixed-size fields to avoid allocations during recording.
    """

    event_type: int = 0
    name: bytes = b'\x00' * MAX_NAME_LENGTH
    category: bytes = b'\x00' * 16
    timestamp_ns: int = 0
    duration_ns: int = 0
    thread_id: int = 0
    process_id: int = 0
    args: bytes = b'\x00' * MAX_ARGS_LENGTH
    id: int = 0
    scope: bytes = b't'
    valid: bool = False

    def set(self, event: ProfileEvent) -> None:
        """Set slot data from a ProfileEvent."""
        self.event_type = event.event_type
        self.name = event.name[:MAX_NAME_LENGTH].encode('utf-8').ljust(MAX_NAME_LENGTH, b'\x00')
        self.category = event.category[:16].encode('utf-8').ljust(16, b'\x00')
        self.timestamp_ns = event.timestamp_ns
        self.duration_ns = event.duration_ns
        self.thread_id = event.thread_id
        self.process_id = event.process_id
        self.args = event.args[:MAX_ARGS_LENGTH].encode('utf-8').ljust(MAX_ARGS_LENGTH, b'\x00')
        self.id = event.id
        self.scope = event.scope.encode('utf-8')[:1] or b't'
        self.valid = True

    def get(self) -> ProfileEvent:
        """Get ProfileEvent from slot data."""
        return ProfileEvent(
            event_type=self.event_type,
            name=self.name.rstrip(b'\x00').decode('utf-8', errors='replace'),
            category=self.category.rstrip(b'\x00').decode('utf-8', errors='replace'),
            timestamp_ns=self.timestamp_ns,
            duration_ns=self.duration_ns,
            thread_id=self.thread_id,
            process_id=self.process_id,
            args=self.args.rstrip(b'\x00').decode('utf-8', errors='replace'),
            id=self.id,
            scope=self.scope.decode('utf-8', errors='replace'),
        )

    def clear(self) -> None:
        """Clear slot data."""
        self.valid = False


# =============================================================================
# EVENT RING BUFFER
# =============================================================================


class EventRingBuffer:
    """Pre-allocated ring buffer for profiling events.

    Provides allocation-free event recording by pre-allocating all storage.
    Events are stored in a circular buffer, with older events being overwritten
    when the buffer is full.

    Thread-safe for single-producer, multiple-consumer patterns.

    Attributes:
        capacity: Maximum number of events the buffer can hold.
        count: Current number of valid events in the buffer.
    """

    def __init__(self, capacity: int = DEFAULT_BUFFER_SIZE) -> None:
        """Initialize the ring buffer.

        Args:
            capacity: Maximum number of events. Will be rounded up to power of 2.
        """
        # Round up to power of 2 for efficient modulo
        self._capacity = 1
        while self._capacity < capacity:
            self._capacity <<= 1

        self._mask = self._capacity - 1
        self._slots: List[EventSlot] = [EventSlot() for _ in range(self._capacity)]
        self._write_index = 0
        self._read_index = 0
        self._count = 0
        self._lock = threading.Lock()
        self._overflow_count = 0

    @property
    def capacity(self) -> int:
        """Maximum number of events."""
        return self._capacity

    @property
    def count(self) -> int:
        """Current number of valid events."""
        with self._lock:
            return self._count

    @property
    def overflow_count(self) -> int:
        """Number of events lost due to overflow."""
        with self._lock:
            return self._overflow_count

    def push(self, event: ProfileEvent) -> bool:
        """Push an event onto the buffer.

        This operation does not allocate memory - it writes directly to
        pre-allocated slots.

        Args:
            event: The event to push.

        Returns:
            True if the event was added, False if the buffer was full and
            an old event was overwritten.
        """
        with self._lock:
            slot = self._slots[self._write_index & self._mask]
            was_valid = slot.valid

            slot.set(event)

            self._write_index += 1

            if was_valid:
                # Overwrote an old event
                self._overflow_count += 1
                self._read_index = max(self._read_index, self._write_index - self._capacity)
                return False
            else:
                self._count = min(self._count + 1, self._capacity)
                return True

    def pop(self) -> Optional[ProfileEvent]:
        """Pop the oldest event from the buffer.

        Returns:
            The oldest event, or None if the buffer is empty.
        """
        with self._lock:
            if self._count == 0:
                return None

            slot = self._slots[self._read_index & self._mask]
            if not slot.valid:
                return None

            event = slot.get()
            slot.clear()

            self._read_index += 1
            self._count -= 1

            return event

    def peek(self, index: int = 0) -> Optional[ProfileEvent]:
        """Peek at an event without removing it.

        Args:
            index: Offset from the oldest event (0 = oldest).

        Returns:
            The event at the given index, or None if out of range.
        """
        with self._lock:
            if index < 0 or index >= self._count:
                return None

            slot = self._slots[(self._read_index + index) & self._mask]
            if not slot.valid:
                return None

            return slot.get()

    def drain(self) -> List[ProfileEvent]:
        """Remove and return all events from the buffer.

        Returns:
            List of all events in order from oldest to newest.
        """
        with self._lock:
            events = []
            for i in range(self._count):
                slot = self._slots[(self._read_index + i) & self._mask]
                if slot.valid:
                    events.append(slot.get())
                    slot.clear()

            self._read_index = self._write_index
            self._count = 0

            return events

    def iter_events(self) -> Iterator[ProfileEvent]:
        """Iterate over events without removing them.

        Yields:
            Events from oldest to newest.
        """
        with self._lock:
            for i in range(self._count):
                slot = self._slots[(self._read_index + i) & self._mask]
                if slot.valid:
                    yield slot.get()

    def clear(self) -> None:
        """Clear all events from the buffer."""
        with self._lock:
            for slot in self._slots:
                slot.clear()
            self._write_index = 0
            self._read_index = 0
            self._count = 0
            self._overflow_count = 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer statistics.
        """
        with self._lock:
            return {
                "capacity": self._capacity,
                "count": self._count,
                "write_index": self._write_index,
                "read_index": self._read_index,
                "overflow_count": self._overflow_count,
                "utilization": self._count / self._capacity if self._capacity > 0 else 0,
            }


# =============================================================================
# EVENT STREAM
# =============================================================================


class EventStream:
    """Central event collector for profiling data.

    Collects profiling events from all sources (CPU, GPU, memory, network)
    into a central ring buffer. Provides scoped recording, integration with
    existing profilers, and export capabilities.

    This is a singleton class - use get_instance() to access.

    Example:
        stream = EventStream.get_instance()
        stream.initialize(buffer_size=65536)

        with stream.scope("frame_update", EventCategory.FRAME):
            # Frame code here
            pass

        stream.counter("fps", 60, EventCategory.FRAME)
    """

    _instance: Optional[EventStream] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize the event stream (use get_instance() instead)."""
        self._buffer: Optional[EventRingBuffer] = None
        self._enabled = True
        self._initialized = False
        self._process_id = os.getpid()
        self._internal_lock = threading.Lock()

        # Per-thread scope stacks for nested scopes
        self._thread_scopes: Dict[int, List[Tuple[str, str, int]]] = {}

        # Event callbacks for real-time processing
        self._callbacks: List[Callable[[ProfileEvent], None]] = []

        # Reference time for relative timestamps
        self._reference_time_ns = 0

        # Integration state
        self._gpu_profiler: Optional[GPUTimestampProfiler] = None
        self._cpu_profiler: Optional[CPUProfiler] = None

    @classmethod
    def get_instance(cls) -> EventStream:
        """Get the singleton EventStream instance.

        Returns:
            The global EventStream instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
            cls._instance = None

    def initialize(
        self,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        reference_time_ns: Optional[int] = None,
    ) -> None:
        """Initialize the event stream.

        Args:
            buffer_size: Ring buffer capacity.
            reference_time_ns: Reference time for relative timestamps.
                             Defaults to current time.
        """
        with self._internal_lock:
            self._buffer = EventRingBuffer(buffer_size)
            self._reference_time_ns = reference_time_ns or time.perf_counter_ns()
            self._initialized = True

    def shutdown(self) -> None:
        """Shutdown the event stream and release resources."""
        with self._internal_lock:
            if self._buffer is not None:
                self._buffer.clear()
                self._buffer = None
            self._thread_scopes.clear()
            self._callbacks.clear()
            self._initialized = False

    @property
    def enabled(self) -> bool:
        """Whether event recording is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable event recording."""
        self._enabled = value

    @property
    def initialized(self) -> bool:
        """Whether the event stream has been initialized."""
        return self._initialized

    @property
    def buffer(self) -> Optional[EventRingBuffer]:
        """The event ring buffer."""
        return self._buffer

    def _get_thread_id(self) -> int:
        """Get the current thread ID."""
        return threading.current_thread().ident or 0

    def _get_scope_stack(self) -> List[Tuple[str, str, int]]:
        """Get the scope stack for the current thread."""
        thread_id = self._get_thread_id()
        if thread_id not in self._thread_scopes:
            self._thread_scopes[thread_id] = []
        return self._thread_scopes[thread_id]

    def _timestamp_ns(self) -> int:
        """Get current timestamp in nanoseconds."""
        return time.perf_counter_ns()

    def _relative_timestamp_ns(self) -> int:
        """Get timestamp relative to reference time."""
        return self._timestamp_ns() - self._reference_time_ns

    def _emit_event(self, event: ProfileEvent) -> None:
        """Emit an event to the buffer and callbacks."""
        if self._buffer is not None:
            self._buffer.push(event)

        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass  # Don't let callback errors stop recording

    def record(
        self,
        event_type: EventType,
        name: str,
        category: Union[str, EventCategory] = EventCategory.CUSTOM,
        duration_ns: int = 0,
        args: Optional[Dict[str, Any]] = None,
        event_id: int = 0,
        scope: EventScope = EventScope.THREAD,
        timestamp_ns: Optional[int] = None,
    ) -> None:
        """Record a profiling event.

        Args:
            event_type: Type of event.
            name: Event name.
            category: Event category.
            duration_ns: Duration in nanoseconds (for COMPLETE events).
            args: Optional arguments dictionary.
            event_id: ID for async/flow events.
            scope: Scope for instant events.
            timestamp_ns: Optional explicit timestamp (relative to reference).
        """
        if not self._enabled or not self._initialized:
            return

        cat_str = category.value if isinstance(category, EventCategory) else str(category)
        ts = timestamp_ns if timestamp_ns is not None else self._relative_timestamp_ns()
        args_str = json.dumps(args) if args else ""

        event = ProfileEvent(
            event_type=event_type.value,
            name=name,
            category=cat_str,
            timestamp_ns=ts,
            duration_ns=duration_ns,
            thread_id=self._get_thread_id(),
            process_id=self._process_id,
            args=args_str,
            id=event_id,
            scope=scope.value,
        )

        with self._internal_lock:
            self._emit_event(event)

    def begin(
        self,
        name: str,
        category: Union[str, EventCategory] = EventCategory.CUSTOM,
        args: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Begin a duration event.

        Args:
            name: Event name.
            category: Event category.
            args: Optional arguments.

        Returns:
            Timestamp of the begin event (for pairing with end).
        """
        ts = self._relative_timestamp_ns()
        self.record(EventType.BEGIN, name, category, args=args, timestamp_ns=ts)
        return ts

    def end(
        self,
        name: str,
        category: Union[str, EventCategory] = EventCategory.CUSTOM,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End a duration event.

        Args:
            name: Event name (must match begin).
            category: Event category.
            args: Optional arguments.
        """
        self.record(EventType.END, name, category, args=args)

    def complete(
        self,
        name: str,
        start_ns: int,
        duration_ns: int,
        category: Union[str, EventCategory] = EventCategory.CUSTOM,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a complete event with known start and duration.

        Args:
            name: Event name.
            start_ns: Start timestamp in nanoseconds (relative to reference).
            duration_ns: Duration in nanoseconds.
            category: Event category.
            args: Optional arguments.
        """
        self.record(
            EventType.COMPLETE,
            name,
            category,
            duration_ns=duration_ns,
            args=args,
            timestamp_ns=start_ns,
        )

    def instant(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        category: Union[str, EventCategory] = EventCategory.CUSTOM,
        scope: EventScope = EventScope.THREAD,
    ) -> None:
        """Record an instant (marker) event.

        Args:
            name: Event name.
            args: Optional arguments.
            category: Event category.
            scope: Event scope (global, process, or thread).
        """
        self.record(EventType.INSTANT, name, category, args=args, scope=scope)

    def counter(
        self,
        name: str,
        value: Union[int, float, Dict[str, Union[int, float]]],
        category: Union[str, EventCategory] = EventCategory.CUSTOM,
    ) -> None:
        """Record a counter event.

        Args:
            name: Counter name.
            value: Counter value or dictionary of values.
            category: Event category.
        """
        if isinstance(value, dict):
            args = value
        else:
            args = {name: value}
        self.record(EventType.COUNTER, name, category, args=args)

    @contextmanager
    def scope(
        self,
        name: str,
        category: Union[str, EventCategory] = EventCategory.CUSTOM,
        args: Optional[Dict[str, Any]] = None,
    ) -> Generator[None, None, None]:
        """Context manager for scoped duration events.

        Records a BEGIN event on entry and END event on exit.

        Args:
            name: Event name.
            category: Event category.
            args: Optional arguments.

        Yields:
            None

        Example:
            with stream.scope("render_frame", EventCategory.RENDER):
                render()
        """
        cat_str = category.value if isinstance(category, EventCategory) else str(category)
        start_ns = self.begin(name, category, args)

        # Track scope for nested scopes
        stack = self._get_scope_stack()
        stack.append((name, cat_str, start_ns))

        try:
            yield
        finally:
            stack.pop()
            self.end(name, category)

    @contextmanager
    def complete_scope(
        self,
        name: str,
        category: Union[str, EventCategory] = EventCategory.CUSTOM,
        args: Optional[Dict[str, Any]] = None,
    ) -> Generator[None, None, None]:
        """Context manager that records a COMPLETE event.

        More efficient than BEGIN/END pair as it only emits one event.

        Args:
            name: Event name.
            category: Event category.
            args: Optional arguments.

        Yields:
            None
        """
        start_ns = self._relative_timestamp_ns()
        try:
            yield
        finally:
            duration_ns = self._relative_timestamp_ns() - start_ns
            self.complete(name, start_ns, duration_ns, category, args)

    def add_callback(self, callback: Callable[[ProfileEvent], None]) -> None:
        """Add a callback for real-time event processing.

        Args:
            callback: Function called for each event.
        """
        with self._internal_lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[ProfileEvent], None]) -> None:
        """Remove an event callback.

        Args:
            callback: The callback to remove.
        """
        with self._internal_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    # =========================================================================
    # GPU INTEGRATION
    # =========================================================================

    def set_gpu_profiler(self, profiler: GPUTimestampProfiler) -> None:
        """Set the GPU timestamp profiler for integration.

        Args:
            profiler: The GPU timestamp profiler instance.
        """
        self._gpu_profiler = profiler

    def import_gpu_results(
        self,
        results: Sequence[TimestampResult],
        frame_index: int = 0,
    ) -> int:
        """Import GPU timestamp results into the event stream.

        Args:
            results: List of TimestampResult from GPU profiler.
            frame_index: Frame index for the results.

        Returns:
            Number of events imported.
        """
        if not self._enabled or not self._initialized:
            return 0

        count = 0
        for result in results:
            # Convert GPU timestamps to event stream timeline
            # GPU timestamps are absolute nanoseconds, need to align with reference
            self.complete(
                name=result.pass_name,
                start_ns=result.start_ns - self._reference_time_ns,
                duration_ns=result.duration_ns,
                category=EventCategory.GPU,
                args={"frame": frame_index},
            )
            count += 1

        return count

    def import_gpu_event(self, event: GPUTimestampEvent) -> None:
        """Import a single GPU timestamp event.

        Args:
            event: The GPU timestamp event to import.
        """
        if not self._enabled or not self._initialized:
            return

        self.complete(
            name=event.pass_name,
            start_ns=event.start_ns - self._reference_time_ns,
            duration_ns=event.duration_ns,
            category=EventCategory.GPU,
            args={
                "frame": event.frame_index,
                "pass_type": event.pass_type.name,
            },
        )

    # =========================================================================
    # CPU INTEGRATION
    # =========================================================================

    def set_cpu_profiler(self, profiler: CPUProfiler) -> None:
        """Set the CPU profiler for integration.

        Args:
            profiler: The CPU profiler instance.
        """
        self._cpu_profiler = profiler

    def import_cpu_samples(
        self,
        samples: Sequence[ProfileSample],
        reference_offset_ns: int = 0,
    ) -> int:
        """Import CPU profile samples into the event stream.

        Args:
            samples: List of ProfileSample from CPU profiler.
            reference_offset_ns: Offset to apply to sample timestamps.

        Returns:
            Number of events imported.
        """
        if not self._enabled or not self._initialized:
            return 0

        count = 0

        def import_sample(sample: ProfileSample) -> None:
            nonlocal count
            self.complete(
                name=sample.name,
                start_ns=sample.start_ns - self._reference_time_ns + reference_offset_ns,
                duration_ns=sample.duration_ns,
                category=EventCategory.CPU,
            )
            count += 1

            for child in sample.children:
                import_sample(child)

        for sample in samples:
            import_sample(sample)

        return count

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get event stream statistics.

        Returns:
            Dictionary with stream statistics.
        """
        with self._internal_lock:
            stats: Dict[str, Any] = {
                "enabled": self._enabled,
                "initialized": self._initialized,
                "process_id": self._process_id,
                "reference_time_ns": self._reference_time_ns,
                "callback_count": len(self._callbacks),
            }

            if self._buffer is not None:
                stats["buffer"] = self._buffer.get_statistics()

            return stats

    def get_events(self) -> List[ProfileEvent]:
        """Get all events without removing them.

        Returns:
            List of all events in order.
        """
        if self._buffer is None:
            return []
        return list(self._buffer.iter_events())

    def drain_events(self) -> List[ProfileEvent]:
        """Remove and return all events.

        Returns:
            List of all events in order.
        """
        if self._buffer is None:
            return []
        return self._buffer.drain()


# =============================================================================
# CHROME TRACING EXPORTER
# =============================================================================


class ChromeTracingExporter:
    """Exports profiling events to Chrome Tracing JSON format.

    The output format is compatible with chrome://tracing viewer and
    other tools that support the Chrome Tracing Event Format.

    Example:
        exporter = ChromeTracingExporter(stream)

        # Export to string
        json_data = exporter.export()

        # Export to file
        exporter.export_to_file("trace.json")

        # Streaming export
        with open("trace.json", "w") as f:
            exporter.export_streaming(f)
    """

    def __init__(
        self,
        stream: Optional[EventStream] = None,
        process_name: str = "TRINITY",
    ) -> None:
        """Initialize the exporter.

        Args:
            stream: EventStream to export from. Defaults to global instance.
            process_name: Name to use for the process in the trace.
        """
        self._stream = stream or EventStream.get_instance()
        self._process_name = process_name

    def _event_to_chrome_format(self, event: ProfileEvent) -> Dict[str, Any]:
        """Convert a ProfileEvent to Chrome Tracing format.

        Args:
            event: The event to convert.

        Returns:
            Dictionary in Chrome Tracing format.
        """
        # Convert nanoseconds to microseconds
        ts_us = event.timestamp_ns / NS_TO_US

        chrome_event: Dict[str, Any] = {
            "name": event.name,
            "cat": event.category,
            "ph": chr(event.event_type),
            "ts": ts_us,
            "pid": event.process_id,
            "tid": event.thread_id,
        }

        # Add duration for complete events
        if event.event_type == EventType.COMPLETE:
            chrome_event["dur"] = event.duration_ns / NS_TO_US

        # Add scope for instant events
        if event.event_type == EventType.INSTANT:
            chrome_event["s"] = event.scope

        # Add ID for async/flow events
        if event.id != 0:
            chrome_event["id"] = event.id

        # Add arguments
        if event.args:
            try:
                chrome_event["args"] = json.loads(event.args)
            except json.JSONDecodeError:
                chrome_event["args"] = {"raw": event.args}

        return chrome_event

    def _get_metadata_events(self) -> List[Dict[str, Any]]:
        """Get metadata events for process/thread names.

        Returns:
            List of metadata events.
        """
        metadata = []

        # Process name
        metadata.append({
            "name": "process_name",
            "ph": "M",
            "pid": os.getpid(),
            "args": {"name": self._process_name},
        })

        # Thread names from events
        thread_ids = set()
        for event in self._stream.get_events():
            thread_ids.add(event.thread_id)

        for tid in thread_ids:
            metadata.append({
                "name": "thread_name",
                "ph": "M",
                "pid": os.getpid(),
                "tid": tid,
                "args": {"name": f"Thread {tid}"},
            })

        return metadata

    def export(self, include_metadata: bool = True) -> str:
        """Export events to Chrome Tracing JSON string.

        Args:
            include_metadata: Whether to include process/thread metadata.

        Returns:
            JSON string in Chrome Tracing format.
        """
        events = []

        if include_metadata:
            events.extend(self._get_metadata_events())

        for event in self._stream.get_events():
            events.append(self._event_to_chrome_format(event))

        return json.dumps({"traceEvents": events}, indent=None, separators=(',', ':'))

    def export_formatted(self, include_metadata: bool = True) -> str:
        """Export events to formatted Chrome Tracing JSON string.

        Args:
            include_metadata: Whether to include process/thread metadata.

        Returns:
            Formatted JSON string in Chrome Tracing format.
        """
        events = []

        if include_metadata:
            events.extend(self._get_metadata_events())

        for event in self._stream.get_events():
            events.append(self._event_to_chrome_format(event))

        return json.dumps({"traceEvents": events}, indent=2)

    def export_to_file(
        self,
        path: str,
        include_metadata: bool = True,
        formatted: bool = False,
    ) -> int:
        """Export events to a file.

        Args:
            path: Output file path.
            include_metadata: Whether to include metadata.
            formatted: Whether to format the JSON output.

        Returns:
            Number of events exported.
        """
        events = []

        if include_metadata:
            events.extend(self._get_metadata_events())

        for event in self._stream.get_events():
            events.append(self._event_to_chrome_format(event))

        with open(path, 'w') as f:
            if formatted:
                json.dump({"traceEvents": events}, f, indent=2)
            else:
                json.dump({"traceEvents": events}, f, separators=(',', ':'))

        return len(events)

    def export_streaming(
        self,
        output: TextIO,
        include_metadata: bool = True,
    ) -> int:
        """Export events in streaming fashion.

        Writes events one at a time to reduce memory usage for large traces.

        Args:
            output: Output file/stream.
            include_metadata: Whether to include metadata.

        Returns:
            Number of events exported.
        """
        output.write('{"traceEvents":[')
        first = True
        count = 0

        if include_metadata:
            for meta in self._get_metadata_events():
                if not first:
                    output.write(',')
                first = False
                json.dump(meta, output, separators=(',', ':'))
                count += 1

        for event in self._stream.get_events():
            if not first:
                output.write(',')
            first = False
            json.dump(self._event_to_chrome_format(event), output, separators=(',', ':'))
            count += 1

        output.write(']}')
        return count

    def drain_and_export(self, include_metadata: bool = True) -> str:
        """Drain events from stream and export.

        This removes events from the stream after exporting.

        Args:
            include_metadata: Whether to include metadata.

        Returns:
            JSON string in Chrome Tracing format.
        """
        events = []

        if include_metadata:
            events.extend(self._get_metadata_events())

        for event in self._stream.drain_events():
            events.append(self._event_to_chrome_format(event))

        return json.dumps({"traceEvents": events}, indent=None, separators=(',', ':'))


# =============================================================================
# BINARY EXPORTER (for efficient storage)
# =============================================================================


class BinaryTraceExporter:
    """Binary format exporter for efficient storage.

    Exports events in a compact binary format that can be converted to
    Chrome Tracing format later.
    """

    # Binary format version
    VERSION = 1

    # Event struct format: type(1) + ts(8) + dur(8) + tid(8) + pid(4) + id(4) + scope(1) + cat_len(1) + name_len(1) + args_len(2)
    # tid is 8 bytes (Q) to handle large thread IDs on 64-bit systems
    EVENT_HEADER_FORMAT = '<BqqQIIBBBH'
    EVENT_HEADER_SIZE = struct.calcsize(EVENT_HEADER_FORMAT)

    def __init__(self, stream: Optional[EventStream] = None) -> None:
        """Initialize the exporter.

        Args:
            stream: EventStream to export from.
        """
        self._stream = stream or EventStream.get_instance()

    def export_to_file(self, path: str) -> int:
        """Export events to a binary file.

        Args:
            path: Output file path.

        Returns:
            Number of events exported.
        """
        count = 0
        with open(path, 'wb') as f:
            # Write header
            f.write(struct.pack('<I', self.VERSION))

            for event in self._stream.get_events():
                self._write_event(f, event)
                count += 1

        return count

    def _write_event(self, f: BinaryIO, event: ProfileEvent) -> None:
        """Write a single event in binary format."""
        cat_bytes = event.category.encode('utf-8')[:255]
        name_bytes = event.name.encode('utf-8')[:255]
        args_bytes = event.args.encode('utf-8')[:65535]

        # Write header
        f.write(struct.pack(
            self.EVENT_HEADER_FORMAT,
            event.event_type,
            event.timestamp_ns,
            event.duration_ns,
            event.thread_id,
            event.process_id,
            event.id,
            ord(event.scope[0]) if event.scope else ord('t'),
            len(cat_bytes),
            len(name_bytes),
            len(args_bytes),
        ))

        # Write variable-length data
        f.write(cat_bytes)
        f.write(name_bytes)
        f.write(args_bytes)

    @classmethod
    def load_from_file(cls, path: str) -> List[ProfileEvent]:
        """Load events from a binary file.

        Args:
            path: Input file path.

        Returns:
            List of ProfileEvent.
        """
        events = []
        with open(path, 'rb') as f:
            # Read header
            version_data = f.read(4)
            if len(version_data) < 4:
                return events
            version = struct.unpack('<I', version_data)[0]
            if version != cls.VERSION:
                raise ValueError(f"Unsupported binary trace version: {version}")

            while True:
                header_data = f.read(cls.EVENT_HEADER_SIZE)
                if len(header_data) < cls.EVENT_HEADER_SIZE:
                    break

                (
                    event_type,
                    timestamp_ns,
                    duration_ns,
                    thread_id,
                    process_id,
                    event_id,
                    scope_byte,
                    cat_len,
                    name_len,
                    args_len,
                ) = struct.unpack(cls.EVENT_HEADER_FORMAT, header_data)

                cat_bytes = f.read(cat_len)
                name_bytes = f.read(name_len)
                args_bytes = f.read(args_len)

                events.append(ProfileEvent(
                    event_type=event_type,
                    name=name_bytes.decode('utf-8', errors='replace'),
                    category=cat_bytes.decode('utf-8', errors='replace'),
                    timestamp_ns=timestamp_ns,
                    duration_ns=duration_ns,
                    thread_id=thread_id,
                    process_id=process_id,
                    args=args_bytes.decode('utf-8', errors='replace'),
                    id=event_id,
                    scope=chr(scope_byte),
                ))

        return events


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_event_stream() -> EventStream:
    """Get the global EventStream instance.

    Returns:
        The singleton EventStream.
    """
    return EventStream.get_instance()


def initialize_event_stream(
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    reference_time_ns: Optional[int] = None,
) -> EventStream:
    """Initialize the global event stream.

    Args:
        buffer_size: Ring buffer capacity.
        reference_time_ns: Reference time for relative timestamps.

    Returns:
        The initialized EventStream.
    """
    stream = EventStream.get_instance()
    stream.initialize(buffer_size, reference_time_ns)
    return stream


def shutdown_event_stream() -> None:
    """Shutdown the global event stream."""
    EventStream.get_instance().shutdown()


def export_chrome_tracing(path: str, formatted: bool = False) -> int:
    """Export the global event stream to Chrome Tracing format.

    Args:
        path: Output file path.
        formatted: Whether to format the JSON output.

    Returns:
        Number of events exported.
    """
    exporter = ChromeTracingExporter()
    return exporter.export_to_file(path, formatted=formatted)


# =============================================================================
# FRAME SCOPE HELPER
# =============================================================================


class FrameScope:
    """Helper for per-frame profiling.

    Provides automatic frame boundary events and counter updates.

    Example:
        frame_scope = FrameScope()

        while running:
            with frame_scope.frame():
                update()
                render()
    """

    def __init__(
        self,
        stream: Optional[EventStream] = None,
        name: str = "frame",
    ) -> None:
        """Initialize frame scope helper.

        Args:
            stream: EventStream to use.
            name: Name for frame events.
        """
        self._stream = stream or EventStream.get_instance()
        self._name = name
        self._frame_number = 0
        self._last_frame_time_ns = 0

    @property
    def frame_number(self) -> int:
        """Current frame number."""
        return self._frame_number

    @contextmanager
    def frame(self) -> Generator[int, None, None]:
        """Context manager for a single frame.

        Yields:
            The current frame number.
        """
        start_ns = time.perf_counter_ns()

        # Record frame marker
        self._stream.instant(
            f"{self._name}_start",
            args={"frame": self._frame_number},
            category=EventCategory.FRAME,
            scope=EventScope.PROCESS,
        )

        try:
            with self._stream.complete_scope(
                self._name,
                EventCategory.FRAME,
                args={"frame": self._frame_number},
            ):
                yield self._frame_number
        finally:
            end_ns = time.perf_counter_ns()
            frame_time_ns = end_ns - start_ns

            # Record frame time counter
            frame_time_ms = frame_time_ns / 1_000_000
            fps = 1000 / frame_time_ms if frame_time_ms > 0 else 0

            self._stream.counter(
                "frame_stats",
                {
                    "frame_time_ms": frame_time_ms,
                    "fps": fps,
                },
                EventCategory.FRAME,
            )

            self._last_frame_time_ns = frame_time_ns
            self._frame_number += 1


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "EventType",
    "EventScope",
    "EventCategory",
    # Data structures
    "ProfileEvent",
    "EventSlot",
    # Ring buffer
    "EventRingBuffer",
    # Event stream
    "EventStream",
    # Exporters
    "ChromeTracingExporter",
    "BinaryTraceExporter",
    # Helpers
    "FrameScope",
    # Convenience functions
    "get_event_stream",
    "initialize_event_stream",
    "shutdown_event_stream",
    "export_chrome_tracing",
    # Configuration
    "DEFAULT_BUFFER_SIZE",
    "MAX_NAME_LENGTH",
    "MAX_ARGS_LENGTH",
]

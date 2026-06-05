"""T-CC-4.1: Snapshot-based time-travel debugging for deterministic simulation.

This module provides time-travel debugging capabilities that allow stepping
backward and forward through simulation history by capturing and restoring
deterministic state snapshots.

Key components:
- TickSnapshot: Captures deterministic state at a specific tick
- SnapshotRingBuffer: Efficient ring buffer storing N most recent snapshots
- TimeTravel: High-level API for stepping backward/forward through simulation
- ReplayController: Manages deterministic replay from snapshot to target tick

Example:
    from engine.debug.time_travel import TimeTravel, TimeTravelConfig

    config = TimeTravelConfig(
        snapshot_interval=60,  # Snapshot every 60 ticks
        max_snapshots=300,     # Keep 5 minutes worth at 60Hz
    )
    time_travel = TimeTravel(scheduler, world, config)

    # Enable recording
    time_travel.enable()

    # Later, step backward
    time_travel.step_backward()

    # Seek to specific tick
    time_travel.seek_to_tick(1000)

Dependencies:
    - T-CC-0.17 (tick_scheduler): TickScheduler, PhaseContext, TickResult
    - T-CC-2.5 (serialization_formats): BinaryWriter, BinaryReader
"""
from __future__ import annotations

import copy
import hashlib
import time
import zlib
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
)

from engine.core.serialization_formats import (
    BinaryReader,
    BinaryWriter,
    JSONReader,
    JSONWriter,
    SerializationFormat,
    create_reader,
    create_writer,
)

if TYPE_CHECKING:
    from engine.core.ecs import DeterministicCommandBuffer, World
    from engine.core.tick_scheduler import PhaseContext, TickScheduler

from trinity.types import Fixed32, PCG64


__all__ = [
    # Core snapshot types
    "TickSnapshot",
    "SnapshotMetadata",
    "SnapshotState",
    # Ring buffer
    "SnapshotRingBuffer",
    "RingBufferConfig",
    # Time travel
    "TimeTravel",
    "TimeTravelConfig",
    "TimeTravelState",
    "TimeTravelEvent",
    # Replay controller
    "ReplayController",
    "ReplayConfig",
    "ReplayState",
    "ReplayResult",
    # State provider protocol
    "StateProvider",
    "WorldStateProvider",
    # Utility types
    "SnapshotComparison",
    "TickRange",
]


# =============================================================================
# STATE PROVIDER PROTOCOL
# =============================================================================


class StateProvider(Protocol):
    """Protocol for objects that can capture and restore deterministic state."""

    def capture_state(self) -> Dict[str, Any]:
        """Capture current deterministic state as a serializable dict."""
        ...

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore state from a previously captured dict."""
        ...

    def compute_checksum(self) -> int:
        """Compute a deterministic checksum of current state."""
        ...


class WorldStateProvider:
    """Default state provider implementation for ECS World.

    Captures entity data, archetype structure, and allocator state
    for deterministic restoration.
    """

    __slots__ = ("_world", "_include_metadata")

    def __init__(self, world: "World", include_metadata: bool = True) -> None:
        """Initialize the world state provider.

        Args:
            world: The ECS World to capture/restore state from.
            include_metadata: Whether to include debug metadata in snapshots.
        """
        self._world = world
        self._include_metadata = include_metadata

    def capture_state(self) -> Dict[str, Any]:
        """Capture complete world state for snapshot."""
        state: Dict[str, Any] = {
            "entities": {},
            "allocator": self._capture_allocator(),
        }

        # Capture all entities and their components
        for entity, mask in self._world._entity_archetype.items():
            archetype = self._world._graph.get_or_create(mask)
            components: Dict[str, Any] = {}
            for cid in mask:
                comp = archetype.get_component(entity, cid)
                if comp is not None:
                    components[str(cid)] = self._serialize_component(comp)
            state["entities"][entity.index] = {
                "generation": entity.generation,
                "mask": [str(cid) for cid in mask],
                "components": components,
            }

        if self._include_metadata:
            state["_metadata"] = {
                "entity_count": len(state["entities"]),
                "capture_time": time.time(),
            }

        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore world state from snapshot.

        This performs a full state replacement, not a merge.
        """
        # Clear current world state
        entities_to_destroy = list(self._world._entity_archetype.keys())
        for entity in entities_to_destroy:
            self._world.destroy(entity)

        # Restore allocator state
        self._restore_allocator(state.get("allocator", {}))

        # Restore entities and components
        from engine.core.ecs.entity import Entity

        for entity_index_str, entity_data in state.get("entities", {}).items():
            entity_index = int(entity_index_str)
            generation = entity_data.get("generation", 0)

            # Reconstruct components
            components = []
            for cid_str, comp_data in entity_data.get("components", {}).items():
                comp = self._deserialize_component(comp_data)
                if comp is not None:
                    components.append(comp)

            # Spawn entity with components
            if components:
                entity = self._world.spawn(*components)

    def compute_checksum(self) -> int:
        """Compute deterministic checksum of world state."""
        state = self.capture_state()
        # Remove non-deterministic metadata
        state.pop("_metadata", None)

        # Create deterministic string representation
        import json
        state_str = json.dumps(state, sort_keys=True, default=str)
        return zlib.crc32(state_str.encode("utf-8"))

    def _capture_allocator(self) -> Dict[str, Any]:
        """Capture entity allocator state."""
        allocator = self._world._allocator
        return {
            "next_index": allocator._next_index,
            "free_list": list(allocator._free_list),
            "generations": dict(allocator._generations),
        }

    def _restore_allocator(self, state: Dict[str, Any]) -> None:
        """Restore entity allocator state."""
        if not state:
            return
        allocator = self._world._allocator
        allocator._next_index = state.get("next_index", 0)
        allocator._free_list = deque(state.get("free_list", []))
        allocator._generations = dict(state.get("generations", {}))

    def _serialize_component(self, component: Any) -> Dict[str, Any]:
        """Serialize a component to dict format."""
        if hasattr(component, "serialize"):
            return component.serialize()
        elif hasattr(component, "__dataclass_fields__"):
            from dataclasses import asdict
            return {
                "__type__": type(component).__name__,
                "__module__": type(component).__module__,
                "data": asdict(component),
            }
        elif hasattr(component, "__dict__"):
            return {
                "__type__": type(component).__name__,
                "__module__": type(component).__module__,
                "data": copy.deepcopy(component.__dict__),
            }
        return {"__type__": "unknown", "data": str(component)}

    def _deserialize_component(self, data: Dict[str, Any]) -> Any:
        """Deserialize a component from dict format."""
        if not isinstance(data, dict):
            return data

        type_name = data.get("__type__")
        module_name = data.get("__module__")
        comp_data = data.get("data", {})

        if type_name and module_name:
            try:
                import importlib
                module = importlib.import_module(module_name)
                cls = getattr(module, type_name, None)
                if cls is not None:
                    if hasattr(cls, "deserialize"):
                        return cls.deserialize(comp_data)
                    elif hasattr(cls, "__dataclass_fields__"):
                        return cls(**comp_data)
                    else:
                        instance = cls.__new__(cls)
                        instance.__dict__.update(comp_data)
                        return instance
            except (ImportError, AttributeError, TypeError):
                pass

        return comp_data


# =============================================================================
# SNAPSHOT TYPES
# =============================================================================


class SnapshotState(IntEnum):
    """State of a snapshot in the buffer."""
    EMPTY = 0
    CAPTURING = 1
    VALID = 2
    RESTORING = 3
    CORRUPTED = 4


@dataclass(slots=True, frozen=True)
class SnapshotMetadata:
    """Metadata about a snapshot.

    Attributes:
        tick: The simulation tick when snapshot was taken.
        timestamp: Real-world timestamp of capture.
        checksum: CRC32 checksum of snapshot data.
        compressed_size: Size in bytes after compression.
        uncompressed_size: Original size before compression.
        rng_state: State of the master RNG at this tick.
        accumulator_raw: Raw value of the tick accumulator.
    """
    tick: int
    timestamp: float
    checksum: int
    compressed_size: int
    uncompressed_size: int
    rng_state: Tuple[int, int]  # (state, increment)
    accumulator_raw: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tick": self.tick,
            "timestamp": self.timestamp,
            "checksum": self.checksum,
            "compressed_size": self.compressed_size,
            "uncompressed_size": self.uncompressed_size,
            "rng_state": list(self.rng_state),
            "accumulator_raw": self.accumulator_raw,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnapshotMetadata":
        """Create from dictionary."""
        return cls(
            tick=data["tick"],
            timestamp=data["timestamp"],
            checksum=data["checksum"],
            compressed_size=data["compressed_size"],
            uncompressed_size=data["uncompressed_size"],
            rng_state=tuple(data["rng_state"]),
            accumulator_raw=data["accumulator_raw"],
        )


@dataclass(slots=True)
class TickSnapshot:
    """A complete snapshot of simulation state at a specific tick.

    Captures all deterministic state needed to restore the simulation
    to this exact point. State is compressed for memory efficiency.

    Attributes:
        metadata: Snapshot metadata including tick and checksum.
        state: The captured state (empty when compressed).
        compressed_data: Compressed state data (populated after compress()).
        status: Current state of the snapshot.
    """
    metadata: SnapshotMetadata
    state: Dict[str, Any] = field(default_factory=dict)
    compressed_data: bytes = field(default=b"")
    status: SnapshotState = SnapshotState.VALID

    @property
    def tick(self) -> int:
        """The tick number of this snapshot."""
        return self.metadata.tick

    @property
    def checksum(self) -> int:
        """CRC32 checksum of the snapshot data."""
        return self.metadata.checksum

    @property
    def is_valid(self) -> bool:
        """Check if snapshot is in a valid, usable state."""
        return self.status == SnapshotState.VALID

    @property
    def is_compressed(self) -> bool:
        """Check if snapshot data is currently compressed."""
        return len(self.compressed_data) > 0 and len(self.state) == 0

    @property
    def size_bytes(self) -> int:
        """Get current size of snapshot data in bytes."""
        if self.is_compressed:
            return self.metadata.compressed_size
        return self.metadata.uncompressed_size

    def compress(self) -> None:
        """Compress the state data to save memory."""
        if self.is_compressed or not self.state:
            return

        import json
        state_bytes = json.dumps(self.state, sort_keys=True, default=str).encode("utf-8")
        self.compressed_data = zlib.compress(state_bytes, level=6)
        self.state = {}

    def decompress(self) -> None:
        """Decompress the state data for access."""
        if not self.is_compressed:
            return

        import json
        state_bytes = zlib.decompress(self.compressed_data)
        self.state = json.loads(state_bytes.decode("utf-8"))
        self.compressed_data = b""

    def get_state(self) -> Dict[str, Any]:
        """Get the state, decompressing if necessary."""
        if self.is_compressed:
            self.decompress()
        return self.state

    def verify_checksum(self) -> bool:
        """Verify the snapshot data matches its checksum."""
        import json
        state = self.get_state()
        state_str = json.dumps(state, sort_keys=True, default=str)
        actual_checksum = zlib.crc32(state_str.encode("utf-8"))
        return actual_checksum == self.metadata.checksum

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metadata": self.metadata.to_dict(),
            "state": self.get_state(),
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TickSnapshot":
        """Create from dictionary."""
        metadata = SnapshotMetadata.from_dict(data["metadata"])
        snapshot = cls(
            metadata=metadata,
            state=data["state"],
            status=SnapshotState(data.get("status", SnapshotState.VALID.value)),
        )
        return snapshot

    @classmethod
    def create(
        cls,
        tick: int,
        state: Dict[str, Any],
        rng_state: Tuple[int, int],
        accumulator: Fixed32,
    ) -> "TickSnapshot":
        """Create a new snapshot from current state.

        Args:
            tick: The current simulation tick.
            state: The captured state dictionary.
            rng_state: Tuple of (state, increment) for the RNG.
            accumulator: The current tick accumulator value.

        Returns:
            A new TickSnapshot with computed metadata.
        """
        import json
        state_str = json.dumps(state, sort_keys=True, default=str)
        state_bytes = state_str.encode("utf-8")
        checksum = zlib.crc32(state_bytes)
        uncompressed_size = len(state_bytes)
        compressed_size = len(zlib.compress(state_bytes, level=6))

        metadata = SnapshotMetadata(
            tick=tick,
            timestamp=time.time(),
            checksum=checksum,
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            rng_state=rng_state,
            accumulator_raw=accumulator.raw,
        )

        return cls(
            metadata=metadata,
            state=state,
            status=SnapshotState.VALID,
        )


# =============================================================================
# RING BUFFER
# =============================================================================


@dataclass(slots=True, frozen=True)
class RingBufferConfig:
    """Configuration for the snapshot ring buffer.

    Attributes:
        max_snapshots: Maximum number of snapshots to store.
        compress_immediately: Whether to compress snapshots on capture.
        verify_on_restore: Whether to verify checksums when restoring.
    """
    max_snapshots: int = 300
    compress_immediately: bool = True
    verify_on_restore: bool = True


class SnapshotRingBuffer:
    """Efficient ring buffer for storing simulation snapshots.

    Stores the N most recent snapshots in a circular buffer,
    automatically discarding older snapshots as new ones arrive.
    Supports fast lookup by tick number.

    Example:
        buffer = SnapshotRingBuffer(RingBufferConfig(max_snapshots=100))
        buffer.push(snapshot1)
        buffer.push(snapshot2)

        # Get specific snapshot
        snapshot = buffer.get_by_tick(100)

        # Get nearest snapshot to a tick
        nearest = buffer.get_nearest(150)
    """

    __slots__ = (
        "_config",
        "_buffer",
        "_head",
        "_count",
        "_tick_index",
        "_min_tick",
        "_max_tick",
    )

    def __init__(self, config: RingBufferConfig | None = None) -> None:
        """Initialize the ring buffer.

        Args:
            config: Buffer configuration (uses defaults if None).
        """
        self._config = config or RingBufferConfig()
        self._buffer: List[Optional[TickSnapshot]] = [
            None for _ in range(self._config.max_snapshots)
        ]
        self._head = 0
        self._count = 0
        self._tick_index: Dict[int, int] = {}  # tick -> buffer index
        self._min_tick = -1
        self._max_tick = -1

    @property
    def config(self) -> RingBufferConfig:
        """Get buffer configuration."""
        return self._config

    @property
    def capacity(self) -> int:
        """Maximum number of snapshots the buffer can hold."""
        return self._config.max_snapshots

    @property
    def count(self) -> int:
        """Current number of snapshots in the buffer."""
        return self._count

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._count == 0

    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return self._count >= self._config.max_snapshots

    @property
    def min_tick(self) -> int:
        """Oldest tick number in the buffer (-1 if empty)."""
        return self._min_tick

    @property
    def max_tick(self) -> int:
        """Newest tick number in the buffer (-1 if empty)."""
        return self._max_tick

    def push(self, snapshot: TickSnapshot) -> Optional[TickSnapshot]:
        """Add a snapshot to the buffer.

        If the buffer is full, the oldest snapshot is evicted and returned.
        Snapshots must be pushed in ascending tick order.

        Args:
            snapshot: The snapshot to add.

        Returns:
            The evicted snapshot if buffer was full, None otherwise.

        Raises:
            ValueError: If snapshot tick is not greater than max_tick.
        """
        if self._max_tick >= 0 and snapshot.tick <= self._max_tick:
            raise ValueError(
                f"Snapshot tick {snapshot.tick} must be greater than max_tick {self._max_tick}"
            )

        evicted: Optional[TickSnapshot] = None

        # Evict oldest if full
        if self.is_full:
            evicted = self._buffer[self._head]
            if evicted is not None:
                self._tick_index.pop(evicted.tick, None)
                # Update min_tick to next oldest
                self._update_min_tick()

        # Compress if configured
        if self._config.compress_immediately:
            snapshot.compress()

        # Store new snapshot
        self._buffer[self._head] = snapshot
        self._tick_index[snapshot.tick] = self._head

        # Update head pointer
        self._head = (self._head + 1) % self._config.max_snapshots

        # Update counts and bounds
        if not self.is_full:
            self._count += 1
        self._max_tick = snapshot.tick
        if self._min_tick < 0:
            self._min_tick = snapshot.tick

        return evicted

    def get_by_tick(self, tick: int) -> Optional[TickSnapshot]:
        """Get snapshot at exact tick number.

        Args:
            tick: The tick number to look up.

        Returns:
            The snapshot if found, None otherwise.
        """
        index = self._tick_index.get(tick)
        if index is None:
            return None
        return self._buffer[index]

    def get_nearest(self, tick: int) -> Optional[TickSnapshot]:
        """Get the snapshot nearest to (but not after) the given tick.

        This returns the most recent snapshot that can be used as a
        starting point to replay to the target tick.

        Args:
            tick: The target tick number.

        Returns:
            The nearest usable snapshot, or None if buffer is empty.
        """
        if self.is_empty:
            return None

        # Exact match
        exact = self.get_by_tick(tick)
        if exact is not None:
            return exact

        # Find nearest snapshot before target tick
        best_tick = -1
        for snapshot_tick in self._tick_index:
            if snapshot_tick <= tick and snapshot_tick > best_tick:
                best_tick = snapshot_tick

        if best_tick < 0:
            return None

        return self.get_by_tick(best_tick)

    def get_range(self, start_tick: int, end_tick: int) -> List[TickSnapshot]:
        """Get all snapshots in a tick range (inclusive).

        Args:
            start_tick: Start of range (inclusive).
            end_tick: End of range (inclusive).

        Returns:
            List of snapshots in ascending tick order.
        """
        result = []
        for tick in sorted(self._tick_index.keys()):
            if start_tick <= tick <= end_tick:
                snapshot = self._buffer[self._tick_index[tick]]
                if snapshot is not None:
                    result.append(snapshot)
        return result

    def clear(self) -> None:
        """Remove all snapshots from the buffer."""
        self._buffer = [None for _ in range(self._config.max_snapshots)]
        self._head = 0
        self._count = 0
        self._tick_index.clear()
        self._min_tick = -1
        self._max_tick = -1

    def contains_tick(self, tick: int) -> bool:
        """Check if buffer contains a snapshot at the given tick."""
        return tick in self._tick_index

    def iter_snapshots(self) -> Iterator[TickSnapshot]:
        """Iterate over all snapshots in ascending tick order."""
        for tick in sorted(self._tick_index.keys()):
            snapshot = self._buffer[self._tick_index[tick]]
            if snapshot is not None:
                yield snapshot

    def memory_usage(self) -> int:
        """Estimate total memory usage in bytes."""
        total = 0
        for snapshot in self._buffer:
            if snapshot is not None:
                total += snapshot.size_bytes
        return total

    def _update_min_tick(self) -> None:
        """Update min_tick after eviction."""
        if not self._tick_index:
            self._min_tick = -1
        else:
            self._min_tick = min(self._tick_index.keys())


# =============================================================================
# TIME TRAVEL TYPES
# =============================================================================


class TimeTravelState(Enum):
    """State of the time travel system."""
    DISABLED = auto()
    RECORDING = auto()
    PAUSED = auto()
    REWINDING = auto()
    REPLAYING = auto()
    SEEKING = auto()


class TimeTravelEvent(Enum):
    """Events that can occur during time travel."""
    SNAPSHOT_CAPTURED = auto()
    SNAPSHOT_RESTORED = auto()
    REPLAY_STARTED = auto()
    REPLAY_COMPLETED = auto()
    REPLAY_FAILED = auto()
    SEEK_STARTED = auto()
    SEEK_COMPLETED = auto()
    STATE_CHANGED = auto()


@dataclass(slots=True, frozen=True)
class TimeTravelConfig:
    """Configuration for time travel debugging.

    Attributes:
        snapshot_interval: Capture snapshot every N ticks.
        max_snapshots: Maximum snapshots to keep in ring buffer.
        compress_snapshots: Whether to compress snapshot data.
        verify_checksums: Verify checksums on restore.
        auto_capture: Automatically capture snapshots during recording.
    """
    snapshot_interval: int = 60
    max_snapshots: int = 300
    compress_snapshots: bool = True
    verify_checksums: bool = True
    auto_capture: bool = True


@dataclass(slots=True)
class TickRange:
    """A range of ticks."""
    start: int
    end: int

    @property
    def count(self) -> int:
        """Number of ticks in the range."""
        return max(0, self.end - self.start + 1)

    def contains(self, tick: int) -> bool:
        """Check if tick is within range."""
        return self.start <= tick <= self.end


@dataclass(slots=True)
class SnapshotComparison:
    """Result of comparing two snapshots."""
    tick_a: int
    tick_b: int
    checksums_match: bool
    differences: Dict[str, Tuple[Any, Any]]


# =============================================================================
# REPLAY CONTROLLER
# =============================================================================


class ReplayState(Enum):
    """State of the replay controller."""
    IDLE = auto()
    PREPARING = auto()
    REPLAYING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass(slots=True, frozen=True)
class ReplayConfig:
    """Configuration for replay operations.

    Attributes:
        verify_each_tick: Verify checksum after each replayed tick.
        max_ticks_per_frame: Maximum ticks to replay per update call.
        store_intermediate: Store intermediate results for debugging.
    """
    verify_each_tick: bool = False
    max_ticks_per_frame: int = 10
    store_intermediate: bool = False


@dataclass(slots=True)
class ReplayResult:
    """Result of a replay operation.

    Attributes:
        success: Whether replay completed successfully.
        start_tick: Starting tick of replay.
        end_tick: Ending tick of replay.
        ticks_replayed: Number of ticks actually replayed.
        duration_ms: Wall-clock time taken in milliseconds.
        error: Error message if replay failed.
    """
    success: bool
    start_tick: int
    end_tick: int
    ticks_replayed: int
    duration_ms: float
    error: Optional[str] = None


class ReplayController:
    """Controller for replaying simulation from a snapshot to a target tick.

    Manages the process of restoring a snapshot and deterministically
    replaying ticks to reach a target state. Ensures replay is
    exactly deterministic by restoring RNG state and using the same
    system callbacks.

    Example:
        controller = ReplayController(scheduler, world, state_provider)
        controller.prepare(snapshot, target_tick=1000)

        while controller.state == ReplayState.REPLAYING:
            controller.step()

        result = controller.get_result()
    """

    __slots__ = (
        "_scheduler",
        "_world",
        "_state_provider",
        "_config",
        "_state",
        "_start_tick",
        "_current_tick",
        "_target_tick",
        "_snapshot",
        "_start_time",
        "_ticks_replayed",
        "_error",
        "_intermediate_checksums",
    )

    def __init__(
        self,
        scheduler: "TickScheduler",
        world: "World",
        state_provider: StateProvider,
        config: ReplayConfig | None = None,
    ) -> None:
        """Initialize the replay controller.

        Args:
            scheduler: The tick scheduler to use for replay.
            world: The ECS world to replay into.
            state_provider: Provider for capturing/restoring state.
            config: Replay configuration.
        """
        self._scheduler = scheduler
        self._world = world
        self._state_provider = state_provider
        self._config = config or ReplayConfig()
        self._state = ReplayState.IDLE
        self._start_tick = 0
        self._current_tick = 0
        self._target_tick = 0
        self._snapshot: Optional[TickSnapshot] = None
        self._start_time = 0.0
        self._ticks_replayed = 0
        self._error: Optional[str] = None
        self._intermediate_checksums: List[Tuple[int, int]] = []

    @property
    def state(self) -> ReplayState:
        """Current replay state."""
        return self._state

    @property
    def current_tick(self) -> int:
        """Current tick during replay."""
        return self._current_tick

    @property
    def target_tick(self) -> int:
        """Target tick for replay."""
        return self._target_tick

    @property
    def progress(self) -> float:
        """Replay progress as a fraction [0, 1]."""
        if self._target_tick <= self._start_tick:
            return 1.0
        total = self._target_tick - self._start_tick
        done = self._current_tick - self._start_tick
        return min(1.0, max(0.0, done / total))

    def prepare(
        self,
        snapshot: TickSnapshot,
        target_tick: int,
    ) -> bool:
        """Prepare to replay from snapshot to target tick.

        Args:
            snapshot: The snapshot to restore as starting point.
            target_tick: The tick to replay up to.

        Returns:
            True if preparation succeeded, False otherwise.
        """
        if self._state == ReplayState.REPLAYING:
            self._error = "Cannot prepare while replay is in progress"
            return False

        if target_tick < snapshot.tick:
            self._error = f"Target tick {target_tick} is before snapshot tick {snapshot.tick}"
            return False

        self._state = ReplayState.PREPARING
        self._snapshot = snapshot
        self._start_tick = snapshot.tick
        self._current_tick = snapshot.tick
        self._target_tick = target_tick
        self._ticks_replayed = 0
        self._error = None
        self._intermediate_checksums.clear()

        # Restore snapshot state
        try:
            self._restore_snapshot(snapshot)
            self._state = ReplayState.REPLAYING
            self._start_time = time.time()
            return True
        except Exception as e:
            self._state = ReplayState.FAILED
            self._error = f"Failed to restore snapshot: {e}"
            return False

    def step(self) -> bool:
        """Execute one step of replay (up to max_ticks_per_frame).

        Returns:
            True if replay is still in progress, False if completed or failed.
        """
        if self._state != ReplayState.REPLAYING:
            return False

        try:
            ticks_this_step = 0
            while (
                self._current_tick < self._target_tick
                and ticks_this_step < self._config.max_ticks_per_frame
            ):
                self._replay_one_tick()
                ticks_this_step += 1
                self._ticks_replayed += 1

            if self._current_tick >= self._target_tick:
                self._state = ReplayState.COMPLETED
                return False

            return True

        except Exception as e:
            self._state = ReplayState.FAILED
            self._error = f"Replay failed at tick {self._current_tick}: {e}"
            return False

    def step_all(self) -> ReplayResult:
        """Execute all remaining replay steps at once.

        Returns:
            The replay result.
        """
        while self.step():
            pass
        return self.get_result()

    def cancel(self) -> None:
        """Cancel the current replay operation."""
        if self._state == ReplayState.REPLAYING:
            self._state = ReplayState.FAILED
            self._error = "Replay cancelled"

    def get_result(self) -> ReplayResult:
        """Get the result of the replay operation."""
        duration_ms = (time.time() - self._start_time) * 1000 if self._start_time > 0 else 0
        return ReplayResult(
            success=self._state == ReplayState.COMPLETED,
            start_tick=self._start_tick,
            end_tick=self._current_tick,
            ticks_replayed=self._ticks_replayed,
            duration_ms=duration_ms,
            error=self._error,
        )

    def get_intermediate_checksums(self) -> List[Tuple[int, int]]:
        """Get intermediate checksums if store_intermediate is enabled."""
        return self._intermediate_checksums.copy()

    def _restore_snapshot(self, snapshot: TickSnapshot) -> None:
        """Restore simulation state from snapshot."""
        # Restore world state
        state = snapshot.get_state()
        self._state_provider.restore_state(state)

        # Restore scheduler state
        self._scheduler.set_tick(snapshot.tick)

        # Restore RNG state
        rng_state, rng_increment = snapshot.metadata.rng_state
        self._scheduler._master_rng._state = rng_state
        self._scheduler._master_rng._increment = rng_increment

        # Restore accumulator
        self._scheduler._accumulator = Fixed32.from_raw(snapshot.metadata.accumulator_raw)

    def _replay_one_tick(self) -> None:
        """Replay a single tick."""
        from engine.core.ecs.deterministic_buffer import DeterministicCommandBuffer

        # Create a fresh command buffer for this tick
        cmd_buffer = DeterministicCommandBuffer(self._current_tick)

        # Execute the tick through the scheduler
        result = self._scheduler._execute_tick(self._world, cmd_buffer)

        # Advance tick counter
        self._current_tick += 1
        self._scheduler._tick = self._current_tick

        # Optionally verify checksum
        if self._config.verify_each_tick:
            actual_checksum = self._state_provider.compute_checksum()
            if self._config.store_intermediate:
                self._intermediate_checksums.append((self._current_tick, actual_checksum))


# =============================================================================
# TIME TRAVEL
# =============================================================================


class TimeTravel:
    """High-level API for time-travel debugging.

    Provides an intuitive interface for stepping backward and forward
    through simulation history, seeking to specific ticks, and comparing
    states at different points in time.

    Example:
        time_travel = TimeTravel(scheduler, world, config)
        time_travel.enable()

        # Run simulation...

        # Step backward 10 ticks
        time_travel.step_backward(10)

        # Seek to specific tick
        time_travel.seek_to_tick(500)

        # Compare states
        comparison = time_travel.compare_ticks(500, 1000)
    """

    __slots__ = (
        "_scheduler",
        "_world",
        "_state_provider",
        "_config",
        "_buffer",
        "_replay_controller",
        "_state",
        "_current_tick",
        "_last_snapshot_tick",
        "_event_handlers",
        "_paused_at_tick",
    )

    def __init__(
        self,
        scheduler: "TickScheduler",
        world: "World",
        config: TimeTravelConfig | None = None,
        state_provider: StateProvider | None = None,
    ) -> None:
        """Initialize time travel debugging.

        Args:
            scheduler: The tick scheduler.
            world: The ECS world.
            config: Time travel configuration.
            state_provider: Custom state provider (uses WorldStateProvider if None).
        """
        self._scheduler = scheduler
        self._world = world
        self._config = config or TimeTravelConfig()
        self._state_provider = state_provider or WorldStateProvider(world)

        buffer_config = RingBufferConfig(
            max_snapshots=self._config.max_snapshots,
            compress_immediately=self._config.compress_snapshots,
            verify_on_restore=self._config.verify_checksums,
        )
        self._buffer = SnapshotRingBuffer(buffer_config)

        replay_config = ReplayConfig(
            verify_each_tick=self._config.verify_checksums,
        )
        self._replay_controller = ReplayController(
            scheduler, world, self._state_provider, replay_config
        )

        self._state = TimeTravelState.DISABLED
        self._current_tick = 0
        self._last_snapshot_tick = -1
        self._event_handlers: List[Callable[[TimeTravelEvent, Any], None]] = []
        self._paused_at_tick = -1

    @property
    def state(self) -> TimeTravelState:
        """Current time travel state."""
        return self._state

    @property
    def config(self) -> TimeTravelConfig:
        """Time travel configuration."""
        return self._config

    @property
    def is_enabled(self) -> bool:
        """Check if time travel is enabled."""
        return self._state != TimeTravelState.DISABLED

    @property
    def is_recording(self) -> bool:
        """Check if currently recording snapshots."""
        return self._state == TimeTravelState.RECORDING

    @property
    def current_tick(self) -> int:
        """Current simulation tick."""
        return self._scheduler.current_tick

    @property
    def available_range(self) -> Optional[TickRange]:
        """Get the range of ticks available for time travel."""
        if self._buffer.is_empty:
            return None
        return TickRange(
            start=self._buffer.min_tick,
            end=max(self._buffer.max_tick, self._scheduler.current_tick),
        )

    @property
    def snapshot_count(self) -> int:
        """Number of snapshots currently stored."""
        return self._buffer.count

    def enable(self) -> None:
        """Enable time travel recording."""
        if self._state != TimeTravelState.DISABLED:
            return
        self._state = TimeTravelState.RECORDING
        self._current_tick = self._scheduler.current_tick
        self._emit_event(TimeTravelEvent.STATE_CHANGED, self._state)

    def disable(self) -> None:
        """Disable time travel and clear history."""
        self._state = TimeTravelState.DISABLED
        self._buffer.clear()
        self._last_snapshot_tick = -1
        self._emit_event(TimeTravelEvent.STATE_CHANGED, self._state)

    def pause(self) -> None:
        """Pause recording (keeps existing snapshots)."""
        if self._state == TimeTravelState.RECORDING:
            self._state = TimeTravelState.PAUSED
            self._paused_at_tick = self._scheduler.current_tick
            self._emit_event(TimeTravelEvent.STATE_CHANGED, self._state)

    def resume(self) -> None:
        """Resume recording after pause."""
        if self._state == TimeTravelState.PAUSED:
            self._state = TimeTravelState.RECORDING
            self._emit_event(TimeTravelEvent.STATE_CHANGED, self._state)

    def update(self) -> None:
        """Called each frame to capture snapshots if needed.

        Call this in your game loop when time travel is enabled.
        """
        if self._state != TimeTravelState.RECORDING:
            return

        if not self._config.auto_capture:
            return

        current_tick = self._scheduler.current_tick

        # Check if we should capture a snapshot
        if self._should_capture(current_tick):
            self.capture_snapshot()

    def capture_snapshot(self) -> Optional[TickSnapshot]:
        """Manually capture a snapshot at the current tick.

        Returns:
            The captured snapshot, or None if capture failed.
        """
        if self._state == TimeTravelState.DISABLED:
            return None

        tick = self._scheduler.current_tick

        # Capture state
        state = self._state_provider.capture_state()

        # Get RNG state
        rng_state = (
            self._scheduler._master_rng._state,
            self._scheduler._master_rng._increment,
        )

        # Create snapshot
        snapshot = TickSnapshot.create(
            tick=tick,
            state=state,
            rng_state=rng_state,
            accumulator=self._scheduler._accumulator,
        )

        # Store in buffer
        self._buffer.push(snapshot)
        self._last_snapshot_tick = tick

        self._emit_event(TimeTravelEvent.SNAPSHOT_CAPTURED, snapshot)
        return snapshot

    def step_backward(self, ticks: int = 1) -> bool:
        """Step backward by the specified number of ticks.

        Args:
            ticks: Number of ticks to step backward.

        Returns:
            True if step succeeded, False otherwise.
        """
        if ticks <= 0:
            return False

        target_tick = max(0, self._scheduler.current_tick - ticks)
        return self.seek_to_tick(target_tick)

    def step_forward(self, ticks: int = 1) -> bool:
        """Step forward by replaying ticks.

        Args:
            ticks: Number of ticks to step forward.

        Returns:
            True if step succeeded, False otherwise.
        """
        if ticks <= 0:
            return False

        # Can only step forward if we're paused at a previous state
        if self._paused_at_tick < 0:
            return False

        target_tick = self._scheduler.current_tick + ticks
        return self.seek_to_tick(target_tick)

    def seek_to_tick(self, target_tick: int) -> bool:
        """Seek to a specific tick number.

        Finds the nearest snapshot before the target and replays
        ticks to reach the exact target tick.

        Args:
            target_tick: The tick to seek to.

        Returns:
            True if seek succeeded, False otherwise.
        """
        if target_tick < 0:
            return False

        # Find nearest snapshot at or before target
        snapshot = self._buffer.get_nearest(target_tick)
        if snapshot is None:
            return False

        self._state = TimeTravelState.SEEKING
        self._emit_event(TimeTravelEvent.SEEK_STARTED, target_tick)

        # Restore snapshot
        try:
            self._restore_snapshot(snapshot)

            # Replay ticks if needed
            if snapshot.tick < target_tick:
                self._replay_controller.prepare(snapshot, target_tick)
                result = self._replay_controller.step_all()

                if not result.success:
                    self._state = TimeTravelState.PAUSED
                    return False

            self._state = TimeTravelState.PAUSED
            self._paused_at_tick = target_tick
            self._emit_event(TimeTravelEvent.SEEK_COMPLETED, target_tick)
            return True

        except Exception as e:
            self._state = TimeTravelState.PAUSED
            return False

    def get_snapshot(self, tick: int) -> Optional[TickSnapshot]:
        """Get snapshot at exact tick number.

        Args:
            tick: The tick number.

        Returns:
            The snapshot if found, None otherwise.
        """
        return self._buffer.get_by_tick(tick)

    def get_nearest_snapshot(self, tick: int) -> Optional[TickSnapshot]:
        """Get nearest snapshot at or before the given tick.

        Args:
            tick: The target tick.

        Returns:
            The nearest snapshot, or None if buffer is empty.
        """
        return self._buffer.get_nearest(tick)

    def compare_ticks(
        self,
        tick_a: int,
        tick_b: int,
    ) -> Optional[SnapshotComparison]:
        """Compare states at two different ticks.

        Args:
            tick_a: First tick to compare.
            tick_b: Second tick to compare.

        Returns:
            Comparison result, or None if either tick has no snapshot.
        """
        snapshot_a = self._buffer.get_by_tick(tick_a)
        snapshot_b = self._buffer.get_by_tick(tick_b)

        if snapshot_a is None or snapshot_b is None:
            return None

        state_a = snapshot_a.get_state()
        state_b = snapshot_b.get_state()

        differences: Dict[str, Tuple[Any, Any]] = {}
        all_keys = set(state_a.keys()) | set(state_b.keys())

        for key in all_keys:
            val_a = state_a.get(key)
            val_b = state_b.get(key)
            if val_a != val_b:
                differences[key] = (val_a, val_b)

        return SnapshotComparison(
            tick_a=tick_a,
            tick_b=tick_b,
            checksums_match=snapshot_a.checksum == snapshot_b.checksum,
            differences=differences,
        )

    def clear_history(self) -> None:
        """Clear all stored snapshots."""
        self._buffer.clear()
        self._last_snapshot_tick = -1

    def memory_usage(self) -> int:
        """Get estimated memory usage in bytes."""
        return self._buffer.memory_usage()

    def on_event(self, handler: Callable[[TimeTravelEvent, Any], None]) -> None:
        """Register an event handler.

        Args:
            handler: Callback function receiving (event, data).
        """
        self._event_handlers.append(handler)

    def remove_event_handler(
        self,
        handler: Callable[[TimeTravelEvent, Any], None],
    ) -> None:
        """Remove a registered event handler."""
        try:
            self._event_handlers.remove(handler)
        except ValueError:
            pass

    def _should_capture(self, tick: int) -> bool:
        """Check if we should capture a snapshot at this tick."""
        if self._last_snapshot_tick < 0:
            return True
        return (tick - self._last_snapshot_tick) >= self._config.snapshot_interval

    def _restore_snapshot(self, snapshot: TickSnapshot) -> None:
        """Restore state from a snapshot."""
        # Verify checksum if configured
        if self._config.verify_checksums:
            if not snapshot.verify_checksum():
                raise ValueError(f"Snapshot checksum verification failed for tick {snapshot.tick}")

        # Restore world state
        state = snapshot.get_state()
        self._state_provider.restore_state(state)

        # Restore scheduler state
        self._scheduler.set_tick(snapshot.tick)

        # Restore RNG state
        rng_state, rng_increment = snapshot.metadata.rng_state
        self._scheduler._master_rng._state = rng_state
        self._scheduler._master_rng._increment = rng_increment

        # Restore accumulator
        self._scheduler._accumulator = Fixed32.from_raw(snapshot.metadata.accumulator_raw)

        self._emit_event(TimeTravelEvent.SNAPSHOT_RESTORED, snapshot)

    def _emit_event(self, event: TimeTravelEvent, data: Any = None) -> None:
        """Emit an event to all registered handlers."""
        for handler in self._event_handlers:
            try:
                handler(event, data)
            except Exception:
                pass  # Don't let handler errors break time travel

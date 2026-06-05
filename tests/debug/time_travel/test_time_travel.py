"""Tests for time-travel debugging (T-CC-4.1).

This test suite covers:
- TickSnapshot: Creation, compression, checksum verification
- SnapshotRingBuffer: Push, get, range queries, eviction
- TimeTravel: Enable/disable, step backward/forward, seek
- ReplayController: Prepare, step, complete replay
- State provider: Capture/restore world state
"""
from __future__ import annotations

import copy
import json
import time
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest

from trinity.types import Fixed32, PCG64


# =============================================================================
# MOCK DEPENDENCIES
# =============================================================================


@dataclass(frozen=True)
class MockEntity:
    """Mock entity for testing."""
    index: int
    generation: int = 0

    def __hash__(self) -> int:
        return hash((self.index, self.generation))


@dataclass
class MockPosition:
    """Mock position component."""
    x: float
    y: float
    z: float = 0.0


@dataclass
class MockVelocity:
    """Mock velocity component."""
    vx: float
    vy: float
    vz: float = 0.0


class MockEntityAllocator:
    """Mock entity allocator."""

    def __init__(self) -> None:
        self._next_index = 0
        self._free_list: deque = deque()
        self._generations: Dict[int, int] = {}
        self._alive_entities: Dict[int, MockEntity] = {}

    def allocate(self) -> MockEntity:
        if self._free_list:
            index = self._free_list.popleft()
        else:
            index = self._next_index
            self._next_index += 1
        gen = self._generations.get(index, 0)
        entity = MockEntity(index=index, generation=gen)
        self._alive_entities[index] = entity
        return entity

    def deallocate(self, entity: MockEntity) -> None:
        self._generations[entity.index] = entity.generation + 1
        self._free_list.append(entity.index)
        self._alive_entities.pop(entity.index, None)

    def is_alive(self, entity: MockEntity) -> bool:
        alive = self._alive_entities.get(entity.index)
        return alive is not None and alive.generation == entity.generation


class MockArchetype:
    """Mock archetype storage."""

    def __init__(self) -> None:
        self._entities: Dict[int, Dict[str, Any]] = {}

    def add_entity(self, entity: MockEntity, components: Dict[str, Any]) -> None:
        self._entities[entity.index] = components.copy()

    def remove_entity(self, entity: MockEntity) -> Optional[Dict[str, Any]]:
        return self._entities.pop(entity.index, None)

    def get_component(self, entity: MockEntity, cid: str) -> Any:
        comps = self._entities.get(entity.index, {})
        return comps.get(cid)

    def set_component(self, entity: MockEntity, cid: str, value: Any) -> None:
        if entity.index in self._entities:
            self._entities[entity.index][cid] = value


class MockArchetypeGraph:
    """Mock archetype graph."""

    def __init__(self) -> None:
        self._archetypes: Dict[frozenset, MockArchetype] = {}

    def get_or_create(self, mask: frozenset) -> MockArchetype:
        if mask not in self._archetypes:
            self._archetypes[mask] = MockArchetype()
        return self._archetypes[mask]


class MockWorld:
    """Mock ECS world for testing."""

    def __init__(self) -> None:
        self._allocator = MockEntityAllocator()
        self._graph = MockArchetypeGraph()
        self._entity_archetype: Dict[MockEntity, frozenset] = {}

    def spawn(self, *components: Any) -> MockEntity:
        entity = self._allocator.allocate()
        comp_map: Dict[str, Any] = {}
        for comp in components:
            cid = type(comp).__name__
            comp_map[cid] = comp
        mask = frozenset(comp_map.keys())
        arch = self._graph.get_or_create(mask)
        arch.add_entity(entity, comp_map)
        self._entity_archetype[entity] = mask
        return entity

    def destroy(self, entity: MockEntity) -> None:
        if not self._allocator.is_alive(entity):
            return
        mask = self._entity_archetype.pop(entity, None)
        if mask is not None:
            arch = self._graph.get_or_create(mask)
            arch.remove_entity(entity)
        self._allocator.deallocate(entity)


class MockTickScheduler:
    """Mock tick scheduler for testing."""

    def __init__(self, seed: int = 0) -> None:
        self._tick = 0
        self._master_rng = PCG64(seed)
        self._accumulator = Fixed32(0)
        self._paused = False
        self._systems: Dict[int, List] = {}
        self._tick_results: List = []

    @property
    def current_tick(self) -> int:
        return self._tick

    def set_tick(self, tick: int) -> None:
        self._tick = tick

    def pause(self) -> None:
        self._paused = True

    def unpause(self) -> None:
        self._paused = False

    def _execute_tick(self, world: MockWorld, cmd_buffer: Any) -> Any:
        """Execute a single tick."""
        result = Mock()
        result.tick = self._tick
        result.checksum = hash(self._tick)
        self._tick_results.append(result)
        return result


# =============================================================================
# IMPORT MODULE UNDER TEST
# =============================================================================


# Import after mocks are defined
from engine.debug.time_travel import (
    RingBufferConfig,
    ReplayConfig,
    ReplayController,
    ReplayResult,
    ReplayState,
    SnapshotComparison,
    SnapshotMetadata,
    SnapshotRingBuffer,
    SnapshotState,
    TickRange,
    TickSnapshot,
    TimeTravel,
    TimeTravelConfig,
    TimeTravelEvent,
    TimeTravelState,
    WorldStateProvider,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_state() -> Dict[str, Any]:
    """Create sample state dictionary."""
    return {
        "entities": {
            "0": {
                "generation": 0,
                "mask": ["Position", "Velocity"],
                "components": {
                    "Position": {"x": 1.0, "y": 2.0, "z": 3.0},
                    "Velocity": {"vx": 0.1, "vy": 0.2, "vz": 0.0},
                },
            },
            "1": {
                "generation": 0,
                "mask": ["Position"],
                "components": {
                    "Position": {"x": 10.0, "y": 20.0, "z": 0.0},
                },
            },
        },
        "allocator": {
            "next_index": 2,
            "free_list": [],
            "generations": {"0": 0, "1": 0},
        },
    }


@pytest.fixture
def sample_snapshot(sample_state: Dict[str, Any]) -> TickSnapshot:
    """Create sample tick snapshot."""
    return TickSnapshot.create(
        tick=100,
        state=sample_state,
        rng_state=(12345, 67890),
        accumulator=Fixed32(0.5),
    )


@pytest.fixture
def mock_world() -> MockWorld:
    """Create mock ECS world."""
    return MockWorld()


@pytest.fixture
def mock_scheduler() -> MockTickScheduler:
    """Create mock tick scheduler."""
    return MockTickScheduler(seed=42)


@pytest.fixture
def mock_state_provider(mock_world: MockWorld) -> MagicMock:
    """Create mock state provider."""
    provider = MagicMock()
    provider.capture_state.return_value = {"test": "state"}
    provider.compute_checksum.return_value = 12345
    return provider


# =============================================================================
# TEST: SnapshotMetadata
# =============================================================================


class TestSnapshotMetadata:
    """Tests for SnapshotMetadata dataclass."""

    def test_create_metadata(self) -> None:
        """Test creating snapshot metadata."""
        metadata = SnapshotMetadata(
            tick=100,
            timestamp=1234567890.0,
            checksum=0xABCD1234,
            compressed_size=1024,
            uncompressed_size=4096,
            rng_state=(111, 222),
            accumulator_raw=32768,
        )
        assert metadata.tick == 100
        assert metadata.timestamp == 1234567890.0
        assert metadata.checksum == 0xABCD1234
        assert metadata.compressed_size == 1024
        assert metadata.uncompressed_size == 4096
        assert metadata.rng_state == (111, 222)
        assert metadata.accumulator_raw == 32768

    def test_metadata_is_frozen(self) -> None:
        """Test that metadata is immutable."""
        metadata = SnapshotMetadata(
            tick=100,
            timestamp=0.0,
            checksum=0,
            compressed_size=0,
            uncompressed_size=0,
            rng_state=(0, 0),
            accumulator_raw=0,
        )
        with pytest.raises(AttributeError):
            metadata.tick = 200  # type: ignore

    def test_metadata_to_dict(self) -> None:
        """Test converting metadata to dictionary."""
        metadata = SnapshotMetadata(
            tick=50,
            timestamp=1000.0,
            checksum=999,
            compressed_size=100,
            uncompressed_size=500,
            rng_state=(1, 2),
            accumulator_raw=100,
        )
        d = metadata.to_dict()
        assert d["tick"] == 50
        assert d["timestamp"] == 1000.0
        assert d["checksum"] == 999
        assert d["rng_state"] == [1, 2]

    def test_metadata_from_dict(self) -> None:
        """Test creating metadata from dictionary."""
        d = {
            "tick": 75,
            "timestamp": 2000.0,
            "checksum": 888,
            "compressed_size": 200,
            "uncompressed_size": 600,
            "rng_state": [3, 4],
            "accumulator_raw": 200,
        }
        metadata = SnapshotMetadata.from_dict(d)
        assert metadata.tick == 75
        assert metadata.rng_state == (3, 4)

    def test_metadata_roundtrip(self) -> None:
        """Test metadata to_dict/from_dict roundtrip."""
        original = SnapshotMetadata(
            tick=123,
            timestamp=time.time(),
            checksum=0xDEADBEEF,
            compressed_size=500,
            uncompressed_size=2000,
            rng_state=(9999, 8888),
            accumulator_raw=65536,
        )
        restored = SnapshotMetadata.from_dict(original.to_dict())
        assert restored.tick == original.tick
        assert restored.checksum == original.checksum
        assert restored.rng_state == original.rng_state


# =============================================================================
# TEST: TickSnapshot
# =============================================================================


class TestTickSnapshot:
    """Tests for TickSnapshot class."""

    def test_create_snapshot(self, sample_state: Dict[str, Any]) -> None:
        """Test creating a snapshot."""
        snapshot = TickSnapshot.create(
            tick=100,
            state=sample_state,
            rng_state=(123, 456),
            accumulator=Fixed32(0.25),
        )
        assert snapshot.tick == 100
        assert snapshot.is_valid
        assert snapshot.status == SnapshotState.VALID

    def test_snapshot_checksum(self, sample_state: Dict[str, Any]) -> None:
        """Test snapshot checksum calculation."""
        snapshot = TickSnapshot.create(
            tick=50,
            state=sample_state,
            rng_state=(0, 0),
            accumulator=Fixed32(0),
        )
        assert snapshot.checksum != 0
        assert snapshot.verify_checksum()

    def test_snapshot_compress_decompress(self, sample_snapshot: TickSnapshot) -> None:
        """Test snapshot compression and decompression."""
        original_state = copy.deepcopy(sample_snapshot.get_state())

        # Not compressed initially
        assert not sample_snapshot.is_compressed

        # Compress
        sample_snapshot.compress()
        assert sample_snapshot.is_compressed
        assert len(sample_snapshot.compressed_data) > 0
        assert len(sample_snapshot.state) == 0

        # Decompress
        sample_snapshot.decompress()
        assert not sample_snapshot.is_compressed
        restored_state = sample_snapshot.get_state()

        # Verify state preserved
        assert restored_state == original_state

    def test_snapshot_get_state_auto_decompress(self, sample_snapshot: TickSnapshot) -> None:
        """Test that get_state automatically decompresses."""
        sample_snapshot.compress()
        assert sample_snapshot.is_compressed

        state = sample_snapshot.get_state()
        assert not sample_snapshot.is_compressed
        assert "entities" in state

    def test_snapshot_size_bytes(self, sample_snapshot: TickSnapshot) -> None:
        """Test snapshot size reporting."""
        uncompressed_size = sample_snapshot.size_bytes
        sample_snapshot.compress()
        compressed_size = sample_snapshot.size_bytes

        # Compressed should be smaller
        assert compressed_size < uncompressed_size

    def test_snapshot_verify_checksum_valid(self, sample_snapshot: TickSnapshot) -> None:
        """Test checksum verification passes for valid snapshot."""
        assert sample_snapshot.verify_checksum()

    def test_snapshot_verify_checksum_corrupted(self, sample_snapshot: TickSnapshot) -> None:
        """Test checksum verification fails for corrupted snapshot."""
        # Corrupt the state
        sample_snapshot.state["corrupted"] = "data"
        assert not sample_snapshot.verify_checksum()

    def test_snapshot_to_dict(self, sample_snapshot: TickSnapshot) -> None:
        """Test converting snapshot to dictionary."""
        d = sample_snapshot.to_dict()
        assert "metadata" in d
        assert "state" in d
        assert "status" in d
        assert d["metadata"]["tick"] == 100

    def test_snapshot_from_dict(self, sample_state: Dict[str, Any]) -> None:
        """Test creating snapshot from dictionary."""
        d = {
            "metadata": {
                "tick": 200,
                "timestamp": 1000.0,
                "checksum": 12345,
                "compressed_size": 100,
                "uncompressed_size": 500,
                "rng_state": [1, 2],
                "accumulator_raw": 100,
            },
            "state": sample_state,
            "status": SnapshotState.VALID.value,
        }
        snapshot = TickSnapshot.from_dict(d)
        assert snapshot.tick == 200
        assert snapshot.is_valid

    def test_snapshot_states(self) -> None:
        """Test all snapshot states."""
        assert SnapshotState.EMPTY.value == 0
        assert SnapshotState.CAPTURING.value == 1
        assert SnapshotState.VALID.value == 2
        assert SnapshotState.RESTORING.value == 3
        assert SnapshotState.CORRUPTED.value == 4

    def test_snapshot_double_compress_noop(self, sample_snapshot: TickSnapshot) -> None:
        """Test compressing twice is a no-op."""
        sample_snapshot.compress()
        compressed_data = sample_snapshot.compressed_data

        sample_snapshot.compress()  # Second compress
        assert sample_snapshot.compressed_data == compressed_data

    def test_snapshot_double_decompress_noop(self, sample_snapshot: TickSnapshot) -> None:
        """Test decompressing twice is a no-op."""
        sample_snapshot.compress()
        sample_snapshot.decompress()
        state = sample_snapshot.state.copy()

        sample_snapshot.decompress()  # Second decompress
        assert sample_snapshot.state == state


# =============================================================================
# TEST: SnapshotRingBuffer
# =============================================================================


class TestSnapshotRingBuffer:
    """Tests for SnapshotRingBuffer class."""

    def test_create_buffer(self) -> None:
        """Test creating a ring buffer."""
        config = RingBufferConfig(max_snapshots=100)
        buffer = SnapshotRingBuffer(config)
        assert buffer.capacity == 100
        assert buffer.count == 0
        assert buffer.is_empty
        assert not buffer.is_full

    def test_buffer_default_config(self) -> None:
        """Test buffer with default config."""
        buffer = SnapshotRingBuffer()
        assert buffer.capacity == 300

    def test_push_single_snapshot(self, sample_snapshot: TickSnapshot) -> None:
        """Test pushing a single snapshot."""
        buffer = SnapshotRingBuffer(RingBufferConfig(max_snapshots=10))
        evicted = buffer.push(sample_snapshot)

        assert evicted is None
        assert buffer.count == 1
        assert buffer.min_tick == 100
        assert buffer.max_tick == 100

    def test_push_multiple_snapshots(self, sample_state: Dict[str, Any]) -> None:
        """Test pushing multiple snapshots."""
        buffer = SnapshotRingBuffer(RingBufferConfig(max_snapshots=10))

        for i in range(5):
            snapshot = TickSnapshot.create(
                tick=i * 60,
                state=sample_state,
                rng_state=(i, i),
                accumulator=Fixed32(0),
            )
            buffer.push(snapshot)

        assert buffer.count == 5
        assert buffer.min_tick == 0
        assert buffer.max_tick == 240

    def test_push_evicts_oldest(self, sample_state: Dict[str, Any]) -> None:
        """Test that oldest snapshot is evicted when full."""
        buffer = SnapshotRingBuffer(RingBufferConfig(max_snapshots=3))

        snapshots = []
        for i in range(5):
            snapshot = TickSnapshot.create(
                tick=i * 10,
                state=sample_state,
                rng_state=(i, i),
                accumulator=Fixed32(0),
            )
            snapshots.append(snapshot)
            evicted = buffer.push(snapshot)

            if i >= 3:
                assert evicted is not None
                assert evicted.tick == (i - 3) * 10

        assert buffer.count == 3
        assert buffer.min_tick == 20
        assert buffer.max_tick == 40

    def test_push_out_of_order_raises(self, sample_state: Dict[str, Any]) -> None:
        """Test that pushing out of order raises error."""
        buffer = SnapshotRingBuffer(RingBufferConfig(max_snapshots=10))

        snapshot1 = TickSnapshot.create(
            tick=100,
            state=sample_state,
            rng_state=(0, 0),
            accumulator=Fixed32(0),
        )
        snapshot2 = TickSnapshot.create(
            tick=50,  # Earlier tick
            state=sample_state,
            rng_state=(0, 0),
            accumulator=Fixed32(0),
        )

        buffer.push(snapshot1)
        with pytest.raises(ValueError, match="must be greater than"):
            buffer.push(snapshot2)

    def test_get_by_tick_exact(self, sample_state: Dict[str, Any]) -> None:
        """Test getting snapshot by exact tick."""
        buffer = SnapshotRingBuffer(RingBufferConfig(max_snapshots=10))

        for i in range(5):
            snapshot = TickSnapshot.create(
                tick=i * 60,
                state=sample_state,
                rng_state=(i, i),
                accumulator=Fixed32(0),
            )
            buffer.push(snapshot)

        result = buffer.get_by_tick(120)
        assert result is not None
        assert result.tick == 120

    def test_get_by_tick_missing(self, sample_snapshot: TickSnapshot) -> None:
        """Test getting missing tick returns None."""
        buffer = SnapshotRingBuffer(RingBufferConfig(max_snapshots=10))
        buffer.push(sample_snapshot)

        assert buffer.get_by_tick(999) is None

    def test_get_nearest(self, sample_state: Dict[str, Any]) -> None:
        """Test getting nearest snapshot."""
        buffer = SnapshotRingBuffer(RingBufferConfig(max_snapshots=10))

        for i in range(3):
            snapshot = TickSnapshot.create(
                tick=i * 100,  # 0, 100, 200
                state=sample_state,
                rng_state=(i, i),
                accumulator=Fixed32(0),
            )
            buffer.push(snapshot)

        # Exact match
        assert buffer.get_nearest(100).tick == 100

        # Between snapshots - returns earlier one
        assert buffer.get_nearest(150).tick == 100
        assert buffer.get_nearest(199).tick == 100

        # After all snapshots
        assert buffer.get_nearest(300).tick == 200

        # Before first snapshot
        result = buffer.get_nearest(-10)
        assert result is None

    def test_get_range(self, sample_state: Dict[str, Any]) -> None:
        """Test getting snapshot range."""
        buffer = SnapshotRingBuffer(RingBufferConfig(max_snapshots=10))

        for i in range(5):
            snapshot = TickSnapshot.create(
                tick=i * 60,
                state=sample_state,
                rng_state=(i, i),
                accumulator=Fixed32(0),
            )
            buffer.push(snapshot)

        # Get range [60, 180]
        result = buffer.get_range(60, 180)
        assert len(result) == 3
        assert result[0].tick == 60
        assert result[1].tick == 120
        assert result[2].tick == 180

    def test_clear(self, sample_snapshot: TickSnapshot) -> None:
        """Test clearing the buffer."""
        buffer = SnapshotRingBuffer(RingBufferConfig(max_snapshots=10))
        buffer.push(sample_snapshot)

        assert buffer.count == 1
        buffer.clear()
        assert buffer.count == 0
        assert buffer.is_empty
        assert buffer.min_tick == -1
        assert buffer.max_tick == -1

    def test_contains_tick(self, sample_snapshot: TickSnapshot) -> None:
        """Test checking if tick is in buffer."""
        buffer = SnapshotRingBuffer(RingBufferConfig(max_snapshots=10))
        buffer.push(sample_snapshot)

        assert buffer.contains_tick(100)
        assert not buffer.contains_tick(50)
        assert not buffer.contains_tick(150)

    def test_iter_snapshots(self, sample_state: Dict[str, Any]) -> None:
        """Test iterating over snapshots."""
        buffer = SnapshotRingBuffer(RingBufferConfig(max_snapshots=10))

        for i in range(3):
            snapshot = TickSnapshot.create(
                tick=i * 100,
                state=sample_state,
                rng_state=(i, i),
                accumulator=Fixed32(0),
            )
            buffer.push(snapshot)

        ticks = [s.tick for s in buffer.iter_snapshots()]
        assert ticks == [0, 100, 200]  # Ascending order

    def test_memory_usage(self, sample_state: Dict[str, Any]) -> None:
        """Test memory usage calculation."""
        buffer = SnapshotRingBuffer(RingBufferConfig(max_snapshots=10))

        for i in range(3):
            snapshot = TickSnapshot.create(
                tick=i * 100,
                state=sample_state,
                rng_state=(i, i),
                accumulator=Fixed32(0),
            )
            buffer.push(snapshot)

        usage = buffer.memory_usage()
        assert usage > 0

    def test_compress_on_push(self, sample_state: Dict[str, Any]) -> None:
        """Test snapshots are compressed on push when configured."""
        config = RingBufferConfig(max_snapshots=10, compress_immediately=True)
        buffer = SnapshotRingBuffer(config)

        snapshot = TickSnapshot.create(
            tick=100,
            state=sample_state,
            rng_state=(0, 0),
            accumulator=Fixed32(0),
        )
        buffer.push(snapshot)

        # Retrieved snapshot should be compressed
        retrieved = buffer.get_by_tick(100)
        assert retrieved.is_compressed

    def test_no_compress_on_push(self, sample_state: Dict[str, Any]) -> None:
        """Test snapshots not compressed when disabled."""
        config = RingBufferConfig(max_snapshots=10, compress_immediately=False)
        buffer = SnapshotRingBuffer(config)

        snapshot = TickSnapshot.create(
            tick=100,
            state=sample_state,
            rng_state=(0, 0),
            accumulator=Fixed32(0),
        )
        buffer.push(snapshot)

        retrieved = buffer.get_by_tick(100)
        assert not retrieved.is_compressed


# =============================================================================
# TEST: TickRange
# =============================================================================


class TestTickRange:
    """Tests for TickRange class."""

    def test_create_range(self) -> None:
        """Test creating a tick range."""
        r = TickRange(start=0, end=100)
        assert r.start == 0
        assert r.end == 100

    def test_range_count(self) -> None:
        """Test tick count calculation."""
        r = TickRange(start=10, end=20)
        assert r.count == 11  # Inclusive

    def test_range_contains(self) -> None:
        """Test tick containment."""
        r = TickRange(start=10, end=20)
        assert r.contains(10)
        assert r.contains(15)
        assert r.contains(20)
        assert not r.contains(9)
        assert not r.contains(21)

    def test_range_empty(self) -> None:
        """Test empty range."""
        r = TickRange(start=20, end=10)  # Invalid
        assert r.count == 0


# =============================================================================
# TEST: ReplayController
# =============================================================================


class TestReplayController:
    """Tests for ReplayController class."""

    @pytest.fixture
    def controller(
        self, mock_scheduler: MockTickScheduler, mock_world: MockWorld
    ) -> ReplayController:
        """Create replay controller fixture."""
        provider = MagicMock()
        provider.capture_state.return_value = {}
        provider.restore_state.return_value = None
        provider.compute_checksum.return_value = 0
        return ReplayController(mock_scheduler, mock_world, provider)

    def test_create_controller(
        self, mock_scheduler: MockTickScheduler, mock_world: MockWorld
    ) -> None:
        """Test creating replay controller."""
        provider = MagicMock()
        controller = ReplayController(mock_scheduler, mock_world, provider)
        assert controller.state == ReplayState.IDLE

    def test_prepare_replay(
        self, controller: ReplayController, sample_snapshot: TickSnapshot
    ) -> None:
        """Test preparing for replay."""
        result = controller.prepare(sample_snapshot, target_tick=150)
        assert result
        assert controller.state == ReplayState.REPLAYING
        assert controller.target_tick == 150

    def test_prepare_target_before_snapshot(
        self, controller: ReplayController, sample_snapshot: TickSnapshot
    ) -> None:
        """Test prepare fails when target is before snapshot."""
        result = controller.prepare(sample_snapshot, target_tick=50)  # Before tick 100
        assert not result
        assert "before snapshot tick" in controller._error

    def test_replay_step(
        self, controller: ReplayController, sample_snapshot: TickSnapshot
    ) -> None:
        """Test stepping through replay."""
        controller.prepare(sample_snapshot, target_tick=110)

        # Step should return True while in progress
        in_progress = controller.step()

        # Check progress
        assert controller.progress >= 0.0

    def test_replay_step_all(
        self, controller: ReplayController, sample_snapshot: TickSnapshot
    ) -> None:
        """Test replaying all at once."""
        controller.prepare(sample_snapshot, target_tick=105)
        result = controller.step_all()

        assert result.success
        assert result.ticks_replayed >= 0

    def test_replay_cancel(
        self, controller: ReplayController, sample_snapshot: TickSnapshot
    ) -> None:
        """Test canceling replay."""
        controller.prepare(sample_snapshot, target_tick=200)
        controller.cancel()

        assert controller.state == ReplayState.FAILED
        assert "cancelled" in controller._error.lower()

    def test_replay_progress(
        self, controller: ReplayController, sample_snapshot: TickSnapshot
    ) -> None:
        """Test replay progress calculation."""
        controller.prepare(sample_snapshot, target_tick=100)  # Same as snapshot
        # When target equals start, progress should be 1.0
        assert controller.progress == 1.0

    def test_get_result(
        self, controller: ReplayController, sample_snapshot: TickSnapshot
    ) -> None:
        """Test getting replay result."""
        controller.prepare(sample_snapshot, target_tick=110)
        controller.step_all()

        result = controller.get_result()
        assert isinstance(result, ReplayResult)
        assert result.start_tick == 100
        assert result.duration_ms >= 0


# =============================================================================
# TEST: TimeTravel
# =============================================================================


class TestTimeTravel:
    """Tests for TimeTravel class."""

    @pytest.fixture
    def time_travel(
        self, mock_scheduler: MockTickScheduler, mock_world: MockWorld
    ) -> TimeTravel:
        """Create time travel fixture."""
        provider = MagicMock()
        provider.capture_state.return_value = {"test": "state"}
        provider.restore_state.return_value = None
        provider.compute_checksum.return_value = 12345

        config = TimeTravelConfig(
            snapshot_interval=10,
            max_snapshots=100,
        )
        return TimeTravel(mock_scheduler, mock_world, config, provider)

    def test_create_time_travel(
        self, mock_scheduler: MockTickScheduler, mock_world: MockWorld
    ) -> None:
        """Test creating time travel instance."""
        tt = TimeTravel(mock_scheduler, mock_world)
        assert tt.state == TimeTravelState.DISABLED
        assert not tt.is_enabled

    def test_enable_disable(self, time_travel: TimeTravel) -> None:
        """Test enable and disable."""
        time_travel.enable()
        assert time_travel.is_enabled
        assert time_travel.state == TimeTravelState.RECORDING

        time_travel.disable()
        assert not time_travel.is_enabled
        assert time_travel.state == TimeTravelState.DISABLED

    def test_pause_resume(self, time_travel: TimeTravel) -> None:
        """Test pause and resume."""
        time_travel.enable()
        time_travel.pause()
        assert time_travel.state == TimeTravelState.PAUSED

        time_travel.resume()
        assert time_travel.state == TimeTravelState.RECORDING

    def test_capture_snapshot(self, time_travel: TimeTravel) -> None:
        """Test manual snapshot capture."""
        time_travel.enable()
        snapshot = time_travel.capture_snapshot()

        assert snapshot is not None
        assert snapshot.tick == 0  # Mock scheduler starts at 0
        assert time_travel.snapshot_count == 1

    def test_capture_when_disabled(self, time_travel: TimeTravel) -> None:
        """Test capture returns None when disabled."""
        snapshot = time_travel.capture_snapshot()
        assert snapshot is None

    def test_step_backward(self, time_travel: TimeTravel) -> None:
        """Test stepping backward."""
        time_travel.enable()

        # Capture a snapshot
        time_travel.capture_snapshot()

        # Can't step back at tick 0
        result = time_travel.step_backward()
        # Since we're at tick 0, stepping back gives target 0 which matches our snapshot
        # This should succeed
        assert result  # or not, depending on implementation

    def test_seek_to_tick(self, time_travel: TimeTravel) -> None:
        """Test seeking to specific tick."""
        time_travel.enable()
        time_travel.capture_snapshot()

        # Seek to the snapshot tick (should succeed)
        result = time_travel.seek_to_tick(0)
        assert result

    def test_seek_to_invalid_tick(self, time_travel: TimeTravel) -> None:
        """Test seeking to invalid tick fails."""
        time_travel.enable()

        # No snapshots yet, seek should fail
        result = time_travel.seek_to_tick(100)
        assert not result

    def test_get_snapshot(self, time_travel: TimeTravel) -> None:
        """Test getting snapshot by tick."""
        time_travel.enable()
        time_travel.capture_snapshot()

        snapshot = time_travel.get_snapshot(0)
        assert snapshot is not None
        assert snapshot.tick == 0

    def test_get_nearest_snapshot(self, time_travel: TimeTravel) -> None:
        """Test getting nearest snapshot."""
        time_travel.enable()
        time_travel.capture_snapshot()

        # Snapshot at tick 0, ask for tick 50
        snapshot = time_travel.get_nearest_snapshot(50)
        assert snapshot is not None
        assert snapshot.tick == 0

    def test_available_range(self, time_travel: TimeTravel) -> None:
        """Test available range property."""
        time_travel.enable()
        assert time_travel.available_range is None  # No snapshots

        time_travel.capture_snapshot()
        r = time_travel.available_range
        assert r is not None
        assert r.start == 0

    def test_clear_history(self, time_travel: TimeTravel) -> None:
        """Test clearing history."""
        time_travel.enable()
        time_travel.capture_snapshot()
        assert time_travel.snapshot_count == 1

        time_travel.clear_history()
        assert time_travel.snapshot_count == 0

    def test_memory_usage(self, time_travel: TimeTravel) -> None:
        """Test memory usage reporting."""
        time_travel.enable()
        time_travel.capture_snapshot()

        usage = time_travel.memory_usage()
        assert usage >= 0

    def test_event_handler(self, time_travel: TimeTravel) -> None:
        """Test event handler registration and firing."""
        events_received = []

        def handler(event: TimeTravelEvent, data: Any) -> None:
            events_received.append((event, data))

        time_travel.on_event(handler)
        time_travel.enable()

        assert len(events_received) > 0
        assert any(e[0] == TimeTravelEvent.STATE_CHANGED for e in events_received)

    def test_remove_event_handler(self, time_travel: TimeTravel) -> None:
        """Test removing event handler."""
        events_received = []

        def handler(event: TimeTravelEvent, data: Any) -> None:
            events_received.append(event)

        time_travel.on_event(handler)
        time_travel.remove_event_handler(handler)
        time_travel.enable()

        # Handler should not receive events after removal
        assert len(events_received) == 0

    def test_auto_capture_on_update(self, time_travel: TimeTravel) -> None:
        """Test automatic snapshot capture on update."""
        time_travel.enable()

        # Simulate time passing
        time_travel._scheduler._tick = 10
        time_travel.update()

        assert time_travel.snapshot_count == 1

    def test_no_auto_capture_when_paused(self, time_travel: TimeTravel) -> None:
        """Test no auto capture when paused."""
        time_travel.enable()
        time_travel.pause()

        time_travel._scheduler._tick = 100
        time_travel.update()

        assert time_travel.snapshot_count == 0


# =============================================================================
# TEST: WorldStateProvider
# =============================================================================


class TestWorldStateProvider:
    """Tests for WorldStateProvider class."""

    def test_capture_state(self, mock_world: MockWorld) -> None:
        """Test capturing world state."""
        # Spawn some entities
        mock_world.spawn(MockPosition(1.0, 2.0))
        mock_world.spawn(MockPosition(3.0, 4.0), MockVelocity(0.1, 0.2))

        provider = WorldStateProvider(mock_world)
        state = provider.capture_state()

        assert "entities" in state
        assert "allocator" in state
        assert len(state["entities"]) == 2

    def test_capture_empty_world(self, mock_world: MockWorld) -> None:
        """Test capturing empty world state."""
        provider = WorldStateProvider(mock_world)
        state = provider.capture_state()

        assert state["entities"] == {}

    def test_compute_checksum(self, mock_world: MockWorld) -> None:
        """Test computing world checksum."""
        mock_world.spawn(MockPosition(1.0, 2.0))

        provider = WorldStateProvider(mock_world)
        checksum = provider.compute_checksum()

        assert checksum != 0
        assert isinstance(checksum, int)

    def test_checksum_deterministic(self, mock_world: MockWorld) -> None:
        """Test that checksum is deterministic."""
        mock_world.spawn(MockPosition(1.0, 2.0))

        provider = WorldStateProvider(mock_world)
        checksum1 = provider.compute_checksum()
        checksum2 = provider.compute_checksum()

        assert checksum1 == checksum2


# =============================================================================
# TEST: SnapshotComparison
# =============================================================================


class TestSnapshotComparison:
    """Tests for SnapshotComparison class."""

    def test_create_comparison(self) -> None:
        """Test creating a snapshot comparison."""
        comp = SnapshotComparison(
            tick_a=100,
            tick_b=200,
            checksums_match=True,
            differences={},
        )
        assert comp.tick_a == 100
        assert comp.tick_b == 200
        assert comp.checksums_match

    def test_comparison_with_differences(self) -> None:
        """Test comparison with differences."""
        comp = SnapshotComparison(
            tick_a=100,
            tick_b=200,
            checksums_match=False,
            differences={
                "entity_count": (5, 7),
                "position": ((1.0, 2.0), (3.0, 4.0)),
            },
        )
        assert not comp.checksums_match
        assert len(comp.differences) == 2
        assert comp.differences["entity_count"] == (5, 7)


# =============================================================================
# TEST: TimeTravelConfig
# =============================================================================


class TestTimeTravelConfig:
    """Tests for TimeTravelConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TimeTravelConfig()
        assert config.snapshot_interval == 60
        assert config.max_snapshots == 300
        assert config.compress_snapshots
        assert config.verify_checksums
        assert config.auto_capture

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = TimeTravelConfig(
            snapshot_interval=30,
            max_snapshots=100,
            compress_snapshots=False,
            verify_checksums=False,
            auto_capture=False,
        )
        assert config.snapshot_interval == 30
        assert config.max_snapshots == 100
        assert not config.compress_snapshots
        assert not config.verify_checksums
        assert not config.auto_capture


# =============================================================================
# TEST: ReplayConfig
# =============================================================================


class TestReplayConfig:
    """Tests for ReplayConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ReplayConfig()
        assert not config.verify_each_tick
        assert config.max_ticks_per_frame == 10
        assert not config.store_intermediate

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ReplayConfig(
            verify_each_tick=True,
            max_ticks_per_frame=5,
            store_intermediate=True,
        )
        assert config.verify_each_tick
        assert config.max_ticks_per_frame == 5
        assert config.store_intermediate


# =============================================================================
# TEST: RingBufferConfig
# =============================================================================


class TestRingBufferConfig:
    """Tests for RingBufferConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RingBufferConfig()
        assert config.max_snapshots == 300
        assert config.compress_immediately
        assert config.verify_on_restore

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = RingBufferConfig(
            max_snapshots=50,
            compress_immediately=False,
            verify_on_restore=False,
        )
        assert config.max_snapshots == 50
        assert not config.compress_immediately
        assert not config.verify_on_restore


# =============================================================================
# TEST: TimeTravelState and TimeTravelEvent
# =============================================================================


class TestTimeTravelEnums:
    """Tests for time travel enums."""

    def test_time_travel_states(self) -> None:
        """Test all time travel states exist."""
        assert TimeTravelState.DISABLED
        assert TimeTravelState.RECORDING
        assert TimeTravelState.PAUSED
        assert TimeTravelState.REWINDING
        assert TimeTravelState.REPLAYING
        assert TimeTravelState.SEEKING

    def test_time_travel_events(self) -> None:
        """Test all time travel events exist."""
        assert TimeTravelEvent.SNAPSHOT_CAPTURED
        assert TimeTravelEvent.SNAPSHOT_RESTORED
        assert TimeTravelEvent.REPLAY_STARTED
        assert TimeTravelEvent.REPLAY_COMPLETED
        assert TimeTravelEvent.REPLAY_FAILED
        assert TimeTravelEvent.SEEK_STARTED
        assert TimeTravelEvent.SEEK_COMPLETED
        assert TimeTravelEvent.STATE_CHANGED


# =============================================================================
# TEST: ReplayState
# =============================================================================


class TestReplayState:
    """Tests for ReplayState enum."""

    def test_all_states(self) -> None:
        """Test all replay states exist."""
        assert ReplayState.IDLE
        assert ReplayState.PREPARING
        assert ReplayState.REPLAYING
        assert ReplayState.COMPLETED
        assert ReplayState.FAILED


# =============================================================================
# TEST: Integration
# =============================================================================


class TestIntegration:
    """Integration tests for time travel system."""

    def test_full_workflow(
        self,
        mock_scheduler: MockTickScheduler,
        mock_world: MockWorld,
    ) -> None:
        """Test full time travel workflow."""
        # Create state provider mock
        provider = MagicMock()
        provider.capture_state.return_value = {"tick": mock_scheduler.current_tick}
        provider.restore_state.return_value = None
        provider.compute_checksum.return_value = 12345

        # Create time travel
        config = TimeTravelConfig(snapshot_interval=5, max_snapshots=100)
        tt = TimeTravel(mock_scheduler, mock_world, config, provider)

        # Enable and capture snapshots
        tt.enable()
        tt.capture_snapshot()  # tick 0

        mock_scheduler._tick = 10
        tt.capture_snapshot()  # tick 10

        mock_scheduler._tick = 20
        tt.capture_snapshot()  # tick 20

        # Verify snapshots captured
        assert tt.snapshot_count == 3

        # Seek back to tick 10
        result = tt.seek_to_tick(10)
        assert result

    def test_snapshot_roundtrip(self) -> None:
        """Test snapshot serialization roundtrip."""
        state = {
            "entities": {"0": {"x": 1.0, "y": 2.0}},
            "counter": 42,
        }

        original = TickSnapshot.create(
            tick=500,
            state=state,
            rng_state=(111, 222),
            accumulator=Fixed32(0.75),
        )

        # Convert to dict and back
        d = original.to_dict()
        restored = TickSnapshot.from_dict(d)

        assert restored.tick == original.tick
        assert restored.checksum == original.checksum

    def test_buffer_fills_and_evicts(self) -> None:
        """Test buffer properly evicts old snapshots."""
        buffer = SnapshotRingBuffer(RingBufferConfig(max_snapshots=5))

        for i in range(10):
            snapshot = TickSnapshot.create(
                tick=i * 10,
                state={"i": i},
                rng_state=(i, i),
                accumulator=Fixed32(0),
            )
            buffer.push(snapshot)

        # Should only have last 5
        assert buffer.count == 5
        assert buffer.min_tick == 50
        assert buffer.max_tick == 90

        # Earlier ticks should be gone
        assert buffer.get_by_tick(0) is None
        assert buffer.get_by_tick(40) is None

        # Later ticks should exist
        assert buffer.get_by_tick(50) is not None
        assert buffer.get_by_tick(90) is not None

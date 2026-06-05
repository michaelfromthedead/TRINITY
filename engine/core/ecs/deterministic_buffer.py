"""Deterministic command buffer for lockstep simulation (T-CC-0.16)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, Callable, Type

from .component import component_id
from .entity import Entity

if TYPE_CHECKING:
    from .world import World

__all__ = [
    "CommandType",
    "DeterministicCommand",
    "DeterministicCommandBuffer",
    "CommandRecord",
    "CommandLog",
]


class CommandType(IntEnum):
    """Command type for ordering and checksumming."""

    SPAWN = 0
    DESPAWN = 1
    INSERT = 2
    REMOVE = 3
    SET_VALUE = 4


@dataclass(slots=True, frozen=True)
class CommandRecord:
    """
    Immutable record of a command for replay and verification.

    Commands are ordered by: (tick, entity_id, component_id, type, sequence)
    """

    tick: int
    sequence: int
    command_type: CommandType
    entity_id: int
    component_id: int
    data_hash: int

    def sort_key(self) -> tuple[int, int, int, int, int]:
        """Key for deterministic ordering."""
        return (self.tick, self.entity_id, self.component_id, self.command_type.value, self.sequence)

    def content_key(self) -> tuple[int, int, int, int, int]:
        """Key for checksum (excludes sequence for order-independence)."""
        return (self.tick, self.entity_id, self.component_id, self.command_type.value, self.data_hash)

    def __lt__(self, other: "CommandRecord") -> bool:
        return self.sort_key() < other.sort_key()


@dataclass(slots=True)
class DeterministicCommand:
    """
    A buffered command with deterministic ordering metadata.
    """

    record: CommandRecord
    apply_fn: Callable[["World"], None]
    data: Any = None


class CommandLog:
    """
    Log of executed commands for replay verification.

    Stores command records in tick order for checksum validation.
    """

    __slots__ = ("_records", "_tick_checksums")

    def __init__(self) -> None:
        self._records: list[CommandRecord] = []
        self._tick_checksums: dict[int, int] = {}

    def append(self, record: CommandRecord) -> None:
        """Append command record."""
        self._records.append(record)

    def compute_tick_checksum(self, tick: int) -> int:
        """Compute checksum for all commands in a tick."""
        tick_records = [r for r in self._records if r.tick == tick]
        tick_records.sort()

        checksum = 0
        for record in tick_records:
            checksum ^= hash(record.content_key())
            checksum = ((checksum << 13) | (checksum >> 51)) & 0xFFFFFFFFFFFFFFFF
            checksum ^= record.data_hash
        return checksum

    def finalize_tick(self, tick: int) -> int:
        """Finalize tick and store its checksum. Returns the checksum."""
        checksum = self.compute_tick_checksum(tick)
        self._tick_checksums[tick] = checksum
        return checksum

    def get_tick_checksum(self, tick: int) -> int | None:
        """Get stored checksum for tick, if finalized."""
        return self._tick_checksums.get(tick)

    def verify_against(self, other: "CommandLog", tick: int) -> bool:
        """Verify this log matches another for a specific tick."""
        self_cs = self.get_tick_checksum(tick)
        other_cs = other.get_tick_checksum(tick)
        if self_cs is None or other_cs is None:
            return False
        return self_cs == other_cs

    def clear_before(self, tick: int) -> None:
        """Clear records before tick (for memory management)."""
        self._records = [r for r in self._records if r.tick >= tick]
        self._tick_checksums = {k: v for k, v in self._tick_checksums.items() if k >= tick}

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[CommandRecord]:
        """Get all records (read-only view)."""
        return list(self._records)

    @property
    def tick_checksums(self) -> dict[int, int]:
        """Get all tick checksums (read-only view)."""
        return dict(self._tick_checksums)


class DeterministicCommandBuffer:
    """
    Command buffer that guarantees deterministic execution order.

    All component writes are recorded with tick + sequence, sorted before
    execution, and logged for replay verification. Same inputs always produce
    identical state.
    """

    __slots__ = ("_commands", "_current_tick", "_sequence", "_log", "_pending_entity_id", "_pending_to_real")

    def __init__(self, initial_tick: int = 0) -> None:
        self._commands: list[DeterministicCommand] = []
        self._current_tick = initial_tick
        self._sequence = 0
        self._log = CommandLog()
        self._pending_entity_id = 0
        self._pending_to_real: dict[int, int] = {}

    @property
    def current_tick(self) -> int:
        """Current simulation tick."""
        return self._current_tick

    @property
    def log(self) -> CommandLog:
        """Command log for replay verification."""
        return self._log

    def _next_sequence(self) -> int:
        """Get next sequence number for current tick."""
        seq = self._sequence
        self._sequence += 1
        return seq

    def _hash_data(self, data: Any) -> int:
        """Compute hash of command data for checksumming."""
        try:
            return hash(data)
        except TypeError:
            return hash(str(data))

    def spawn(self, *components: Any) -> int:
        """
        Queue entity spawn with components.

        Returns a pending entity ID that will be valid after flush.
        """
        self._pending_entity_id -= 1
        pending_id = self._pending_entity_id

        data_hash = self._hash_data(components)
        record = CommandRecord(
            tick=self._current_tick,
            sequence=self._next_sequence(),
            command_type=CommandType.SPAWN,
            entity_id=pending_id,
            component_id=0,
            data_hash=data_hash,
        )

        def apply_fn(world: "World") -> None:
            entity = world.spawn(*components)
            # Store mapping for entity resolution
            self._pending_to_real[pending_id] = entity.index

        cmd = DeterministicCommand(record=record, apply_fn=apply_fn, data=components)
        self._commands.append(cmd)
        return pending_id

    def despawn(self, entity: Entity) -> None:
        """Queue entity despawn."""
        record = CommandRecord(
            tick=self._current_tick,
            sequence=self._next_sequence(),
            command_type=CommandType.DESPAWN,
            entity_id=entity.index,
            component_id=0,
            data_hash=hash(entity.index),
        )

        def apply_fn(world: "World") -> None:
            world.destroy(entity)

        cmd = DeterministicCommand(record=record, apply_fn=apply_fn)
        self._commands.append(cmd)

    def insert(self, entity: Entity, component: Any) -> None:
        """Queue component insertion."""
        comp_id = component_id(type(component))
        data_hash = self._hash_data(component)

        record = CommandRecord(
            tick=self._current_tick,
            sequence=self._next_sequence(),
            command_type=CommandType.INSERT,
            entity_id=entity.index,
            component_id=comp_id,
            data_hash=data_hash,
        )

        def apply_fn(world: "World") -> None:
            world.add_component(entity, component)

        cmd = DeterministicCommand(record=record, apply_fn=apply_fn, data=component)
        self._commands.append(cmd)

    def remove(self, entity: Entity, component_type: Type) -> None:
        """Queue component removal."""
        comp_id = component_id(component_type)

        record = CommandRecord(
            tick=self._current_tick,
            sequence=self._next_sequence(),
            command_type=CommandType.REMOVE,
            entity_id=entity.index,
            component_id=comp_id,
            data_hash=hash(component_type),
        )

        def apply_fn(world: "World") -> None:
            world.remove_component(entity, component_type)

        cmd = DeterministicCommand(record=record, apply_fn=apply_fn)
        self._commands.append(cmd)

    def set_value(self, entity: Entity, component_type: Type, field: str, value: Any) -> None:
        """Queue component field modification."""
        comp_id = component_id(component_type)
        data_hash = self._hash_data((field, value))

        record = CommandRecord(
            tick=self._current_tick,
            sequence=self._next_sequence(),
            command_type=CommandType.SET_VALUE,
            entity_id=entity.index,
            component_id=comp_id,
            data_hash=data_hash,
        )

        def apply_fn(world: "World") -> None:
            comp = world.get_component(entity, component_type)
            if comp is not None:
                setattr(comp, field, value)

        cmd = DeterministicCommand(record=record, apply_fn=apply_fn, data=(field, value))
        self._commands.append(cmd)

    def flush(self, world: "World") -> int:
        """
        Apply all commands in deterministic order.

        Commands are sorted by (tick, entity_id, component_id, type, sequence)
        before execution to ensure identical ordering regardless of queue order.

        Returns the tick checksum.
        """
        if not self._commands:
            return self._log.finalize_tick(self._current_tick)

        # Clear pending entity mapping for this flush
        self._pending_to_real.clear()

        # Sort commands for deterministic order
        self._commands.sort(key=lambda c: c.record.sort_key())

        # Execute and log
        for cmd in self._commands:
            cmd.apply_fn(world)
            self._log.append(cmd.record)

        self._commands.clear()
        self._pending_to_real.clear()

        return self._log.finalize_tick(self._current_tick)

    def advance_tick(self) -> None:
        """Advance to next tick. Resets sequence counter."""
        self._current_tick += 1
        self._sequence = 0

    def get_tick_checksum(self, tick: int) -> int | None:
        """Get checksum for completed tick."""
        return self._log.get_tick_checksum(tick)

    def verify_tick(self, tick: int, expected_checksum: int) -> bool:
        """Verify tick checksum matches expected value."""
        actual = self.get_tick_checksum(tick)
        return actual == expected_checksum

    def clear_history_before(self, tick: int) -> None:
        """Clear command history before tick to save memory."""
        self._log.clear_before(tick)

    def __len__(self) -> int:
        """Number of pending commands."""
        return len(self._commands)

    @property
    def pending_count(self) -> int:
        """Number of pending commands."""
        return len(self._commands)


class ReplayBuffer:
    """
    Replay buffer that can reconstruct state from command logs.

    Used for lockstep simulation verification and rollback.
    """

    __slots__ = ("_log", "_snapshot_interval", "_snapshots")

    def __init__(self, log: CommandLog, snapshot_interval: int = 60) -> None:
        self._log = log
        self._snapshot_interval = snapshot_interval
        self._snapshots: dict[int, bytes] = {}

    def store_snapshot(self, tick: int, world_state: bytes) -> None:
        """Store world state snapshot at tick."""
        if tick % self._snapshot_interval == 0:
            self._snapshots[tick] = world_state

    def get_nearest_snapshot(self, tick: int) -> tuple[int, bytes] | None:
        """Get nearest snapshot at or before tick."""
        valid_ticks = [t for t in self._snapshots if t <= tick]
        if not valid_ticks:
            return None
        nearest = max(valid_ticks)
        return (nearest, self._snapshots[nearest])

    def get_commands_in_range(self, start_tick: int, end_tick: int) -> list[CommandRecord]:
        """Get all command records in tick range [start, end)."""
        return [r for r in self._log.records if start_tick <= r.tick < end_tick]

    def verify_tick_sequence(self, ticks: list[int], checksums: list[int]) -> list[int]:
        """
        Verify sequence of tick checksums.

        Returns list of mismatched ticks (empty if all match).
        """
        mismatches = []
        for tick, expected in zip(ticks, checksums):
            actual = self._log.get_tick_checksum(tick)
            if actual != expected:
                mismatches.append(tick)
        return mismatches

    def clear_snapshots_before(self, tick: int) -> None:
        """Clear snapshots before tick."""
        self._snapshots = {k: v for k, v in self._snapshots.items() if k >= tick}

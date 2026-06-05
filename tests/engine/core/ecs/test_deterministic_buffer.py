"""Tests for DeterministicCommandBuffer (T-CC-0.16)."""
from dataclasses import dataclass

import pytest

from engine.core.ecs import World, Entity
from engine.core.ecs.deterministic_buffer import (
    CommandType,
    CommandRecord,
    CommandLog,
    DeterministicCommand,
    DeterministicCommandBuffer,
    ReplayBuffer,
)


@dataclass
class Position:
    """Test component."""
    x: float = 0.0
    y: float = 0.0


@dataclass
class Velocity:
    """Test component."""
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class Health:
    """Test component."""
    value: int = 100


class TestCommandRecord:
    """Test CommandRecord ordering."""

    def test_sort_key(self):
        """Test sort key generation."""
        record = CommandRecord(
            tick=1,
            sequence=2,
            command_type=CommandType.INSERT,
            entity_id=10,
            component_id=20,
            data_hash=123,
        )
        key = record.sort_key()
        assert key == (1, 10, 20, CommandType.INSERT.value, 2)

    def test_ordering_by_tick(self):
        """Test records ordered by tick first."""
        r1 = CommandRecord(tick=1, sequence=0, command_type=CommandType.INSERT, entity_id=0, component_id=0, data_hash=0)
        r2 = CommandRecord(tick=2, sequence=0, command_type=CommandType.INSERT, entity_id=0, component_id=0, data_hash=0)
        assert r1 < r2

    def test_ordering_by_entity(self):
        """Test same tick ordered by entity_id."""
        r1 = CommandRecord(tick=1, sequence=0, command_type=CommandType.INSERT, entity_id=1, component_id=0, data_hash=0)
        r2 = CommandRecord(tick=1, sequence=0, command_type=CommandType.INSERT, entity_id=2, component_id=0, data_hash=0)
        assert r1 < r2

    def test_ordering_by_component(self):
        """Test same entity ordered by component_id."""
        r1 = CommandRecord(tick=1, sequence=0, command_type=CommandType.INSERT, entity_id=1, component_id=1, data_hash=0)
        r2 = CommandRecord(tick=1, sequence=0, command_type=CommandType.INSERT, entity_id=1, component_id=2, data_hash=0)
        assert r1 < r2

    def test_ordering_by_type(self):
        """Test same component ordered by command type."""
        r1 = CommandRecord(tick=1, sequence=0, command_type=CommandType.SPAWN, entity_id=1, component_id=1, data_hash=0)
        r2 = CommandRecord(tick=1, sequence=0, command_type=CommandType.INSERT, entity_id=1, component_id=1, data_hash=0)
        assert r1 < r2  # SPAWN=0 < INSERT=2

    def test_ordering_by_sequence(self):
        """Test tiebreaker by sequence."""
        r1 = CommandRecord(tick=1, sequence=0, command_type=CommandType.INSERT, entity_id=1, component_id=1, data_hash=0)
        r2 = CommandRecord(tick=1, sequence=1, command_type=CommandType.INSERT, entity_id=1, component_id=1, data_hash=0)
        assert r1 < r2

    def test_frozen(self):
        """Test record is immutable."""
        record = CommandRecord(tick=1, sequence=0, command_type=CommandType.INSERT, entity_id=0, component_id=0, data_hash=0)
        with pytest.raises(Exception):
            record.tick = 2


class TestCommandLog:
    """Test CommandLog for replay verification."""

    def test_append(self):
        """Test appending records."""
        log = CommandLog()
        record = CommandRecord(tick=0, sequence=0, command_type=CommandType.SPAWN, entity_id=1, component_id=0, data_hash=123)
        log.append(record)
        assert len(log) == 1

    def test_tick_checksum_deterministic(self):
        """Test tick checksum is deterministic."""
        log1 = CommandLog()
        log2 = CommandLog()

        records = [
            CommandRecord(tick=0, sequence=i, command_type=CommandType.INSERT, entity_id=i, component_id=1, data_hash=i * 100)
            for i in range(5)
        ]

        for r in records:
            log1.append(r)
            log2.append(r)

        cs1 = log1.compute_tick_checksum(0)
        cs2 = log2.compute_tick_checksum(0)
        assert cs1 == cs2

    def test_finalize_tick(self):
        """Test finalizing tick stores checksum."""
        log = CommandLog()
        record = CommandRecord(tick=0, sequence=0, command_type=CommandType.SPAWN, entity_id=1, component_id=0, data_hash=42)
        log.append(record)
        checksum = log.finalize_tick(0)

        assert log.get_tick_checksum(0) == checksum

    def test_verify_against_same(self):
        """Test verification passes for identical logs."""
        log1 = CommandLog()
        log2 = CommandLog()

        record = CommandRecord(tick=0, sequence=0, command_type=CommandType.SPAWN, entity_id=1, component_id=0, data_hash=42)
        log1.append(record)
        log2.append(record)

        log1.finalize_tick(0)
        log2.finalize_tick(0)

        assert log1.verify_against(log2, 0)

    def test_verify_against_different(self):
        """Test verification fails for different logs."""
        log1 = CommandLog()
        log2 = CommandLog()

        r1 = CommandRecord(tick=0, sequence=0, command_type=CommandType.SPAWN, entity_id=1, component_id=0, data_hash=42)
        r2 = CommandRecord(tick=0, sequence=0, command_type=CommandType.SPAWN, entity_id=1, component_id=0, data_hash=99)

        log1.append(r1)
        log2.append(r2)

        log1.finalize_tick(0)
        log2.finalize_tick(0)

        assert not log1.verify_against(log2, 0)

    def test_clear_before(self):
        """Test clearing old records."""
        log = CommandLog()
        for tick in range(5):
            record = CommandRecord(tick=tick, sequence=0, command_type=CommandType.SPAWN, entity_id=tick, component_id=0, data_hash=tick)
            log.append(record)
            log.finalize_tick(tick)

        log.clear_before(3)

        assert len(log) == 2  # ticks 3, 4
        assert log.get_tick_checksum(0) is None
        assert log.get_tick_checksum(3) is not None


class TestDeterministicCommandBuffer:
    """Test DeterministicCommandBuffer."""

    def test_initial_state(self):
        """Test initial buffer state."""
        buf = DeterministicCommandBuffer()
        assert buf.current_tick == 0
        assert len(buf) == 0

    def test_initial_tick(self):
        """Test custom initial tick."""
        buf = DeterministicCommandBuffer(initial_tick=100)
        assert buf.current_tick == 100

    def test_advance_tick(self):
        """Test advancing tick."""
        buf = DeterministicCommandBuffer()
        buf.advance_tick()
        assert buf.current_tick == 1

    def test_insert_queues_command(self):
        """Test insert queues a command."""
        world = World()
        entity = world.spawn(Position(1, 2))
        buf = DeterministicCommandBuffer()

        buf.insert(entity, Velocity(3, 4))
        assert len(buf) == 1

    def test_flush_applies_commands(self):
        """Test flush applies queued commands."""
        world = World()
        entity = world.spawn(Position(1, 2))
        buf = DeterministicCommandBuffer()

        buf.insert(entity, Velocity(3, 4))
        buf.flush(world)

        vel = world.get_component(entity, Velocity)
        assert vel is not None
        assert vel.vx == 3
        assert vel.vy == 4

    def test_flush_clears_pending(self):
        """Test flush clears pending commands."""
        world = World()
        entity = world.spawn(Position(1, 2))
        buf = DeterministicCommandBuffer()

        buf.insert(entity, Velocity(3, 4))
        buf.flush(world)

        assert len(buf) == 0

    def test_flush_returns_checksum(self):
        """Test flush returns tick checksum."""
        world = World()
        entity = world.spawn(Position(1, 2))
        buf = DeterministicCommandBuffer()

        buf.insert(entity, Velocity(3, 4))
        checksum = buf.flush(world)

        assert isinstance(checksum, int)
        assert buf.get_tick_checksum(0) == checksum

    def test_remove_queues_command(self):
        """Test remove queues component removal."""
        world = World()
        entity = world.spawn(Position(1, 2), Velocity(3, 4))
        buf = DeterministicCommandBuffer()

        buf.remove(entity, Velocity)
        buf.flush(world)

        assert world.get_component(entity, Velocity) is None

    def test_despawn_queues_command(self):
        """Test despawn queues entity destruction."""
        world = World()
        entity = world.spawn(Position(1, 2))
        buf = DeterministicCommandBuffer()

        buf.despawn(entity)
        buf.flush(world)

        assert world.get_component(entity, Position) is None

    def test_set_value_modifies_field(self):
        """Test set_value modifies component field."""
        world = World()
        entity = world.spawn(Health(100))
        buf = DeterministicCommandBuffer()

        buf.set_value(entity, Health, "value", 50)
        buf.flush(world)

        health = world.get_component(entity, Health)
        assert health.value == 50


class TestDeterministicOrdering:
    """Test deterministic command ordering."""

    def test_commands_sorted_before_execution(self):
        """Test commands execute in sorted order regardless of queue order."""
        world = World()
        entity1 = world.spawn(Health(100))
        entity2 = world.spawn(Health(100))

        # Queue in reverse order
        buf = DeterministicCommandBuffer()
        buf.set_value(entity2, Health, "value", 50)
        buf.set_value(entity1, Health, "value", 25)
        buf.flush(world)

        # Both should apply
        assert world.get_component(entity1, Health).value == 25
        assert world.get_component(entity2, Health).value == 50

    def test_same_commands_same_checksum(self):
        """Test same commands in different order produce same checksum."""
        world1 = World()
        world2 = World()

        e1_1 = world1.spawn(Health(100))
        e1_2 = world1.spawn(Health(100))
        e2_1 = world2.spawn(Health(100))
        e2_2 = world2.spawn(Health(100))

        buf1 = DeterministicCommandBuffer()
        buf1.set_value(e1_1, Health, "value", 10)
        buf1.set_value(e1_2, Health, "value", 20)

        buf2 = DeterministicCommandBuffer()
        buf2.set_value(e2_2, Health, "value", 20)  # Reverse order
        buf2.set_value(e2_1, Health, "value", 10)

        cs1 = buf1.flush(world1)
        cs2 = buf2.flush(world2)

        assert cs1 == cs2

    def test_replay_determinism(self):
        """Test replaying same commands produces same result."""

        def simulate(seed):
            world = World()
            entities = [world.spawn(Health(100)) for _ in range(5)]
            buf = DeterministicCommandBuffer()

            for i, e in enumerate(entities):
                buf.set_value(e, Health, "value", (seed + i) * 10)

            return buf.flush(world)

        cs1 = simulate(42)
        cs2 = simulate(42)
        cs3 = simulate(99)

        assert cs1 == cs2
        assert cs1 != cs3


class TestCommandLogVerification:
    """Test command log for replay verification."""

    def test_verify_tick_matches(self):
        """Test tick verification passes for matching commands."""
        world1 = World()
        world2 = World()

        e1 = world1.spawn(Health(100))
        e2 = world2.spawn(Health(100))

        buf1 = DeterministicCommandBuffer()
        buf1.set_value(e1, Health, "value", 50)
        cs1 = buf1.flush(world1)

        buf2 = DeterministicCommandBuffer()
        buf2.set_value(e2, Health, "value", 50)
        cs2 = buf2.flush(world2)

        assert buf1.verify_tick(0, cs2)
        assert buf2.verify_tick(0, cs1)

    def test_verify_tick_fails_on_mismatch(self):
        """Test tick verification fails for mismatched commands."""
        world = World()
        entity = world.spawn(Health(100))

        buf = DeterministicCommandBuffer()
        buf.set_value(entity, Health, "value", 50)
        buf.flush(world)

        assert not buf.verify_tick(0, 12345)  # Wrong checksum


class TestReplayBuffer:
    """Test ReplayBuffer for state reconstruction."""

    def test_store_snapshot(self):
        """Test storing world state snapshots."""
        log = CommandLog()
        replay = ReplayBuffer(log, snapshot_interval=60)

        replay.store_snapshot(0, b"state_0")
        replay.store_snapshot(60, b"state_60")
        replay.store_snapshot(30, b"state_30")  # Not on interval, won't store

        result = replay.get_nearest_snapshot(60)
        assert result == (60, b"state_60")

    def test_get_nearest_snapshot(self):
        """Test getting nearest snapshot before tick."""
        log = CommandLog()
        replay = ReplayBuffer(log, snapshot_interval=60)

        replay.store_snapshot(0, b"state_0")
        replay.store_snapshot(60, b"state_60")
        replay.store_snapshot(120, b"state_120")

        result = replay.get_nearest_snapshot(100)
        assert result == (60, b"state_60")

    def test_get_nearest_snapshot_exact(self):
        """Test getting snapshot at exact tick."""
        log = CommandLog()
        replay = ReplayBuffer(log, snapshot_interval=60)

        replay.store_snapshot(60, b"state_60")

        result = replay.get_nearest_snapshot(60)
        assert result == (60, b"state_60")

    def test_get_nearest_snapshot_none(self):
        """Test getting snapshot when none available."""
        log = CommandLog()
        replay = ReplayBuffer(log, snapshot_interval=60)

        result = replay.get_nearest_snapshot(30)
        assert result is None

    def test_get_commands_in_range(self):
        """Test getting commands in tick range."""
        log = CommandLog()
        for tick in range(5):
            record = CommandRecord(tick=tick, sequence=0, command_type=CommandType.INSERT, entity_id=tick, component_id=0, data_hash=0)
            log.append(record)

        replay = ReplayBuffer(log)
        commands = replay.get_commands_in_range(1, 4)

        assert len(commands) == 3
        assert all(1 <= c.tick < 4 for c in commands)

    def test_verify_tick_sequence_all_match(self):
        """Test verifying sequence of matching checksums."""
        log = CommandLog()
        for tick in range(5):
            record = CommandRecord(tick=tick, sequence=0, command_type=CommandType.INSERT, entity_id=tick, component_id=0, data_hash=tick * 10)
            log.append(record)
            log.finalize_tick(tick)

        replay = ReplayBuffer(log)

        ticks = [0, 1, 2, 3, 4]
        checksums = [log.get_tick_checksum(t) for t in ticks]

        mismatches = replay.verify_tick_sequence(ticks, checksums)
        assert mismatches == []

    def test_verify_tick_sequence_with_mismatch(self):
        """Test verifying sequence detects mismatches."""
        log = CommandLog()
        for tick in range(5):
            record = CommandRecord(tick=tick, sequence=0, command_type=CommandType.INSERT, entity_id=tick, component_id=0, data_hash=tick * 10)
            log.append(record)
            log.finalize_tick(tick)

        replay = ReplayBuffer(log)

        ticks = [0, 1, 2, 3, 4]
        checksums = [log.get_tick_checksum(t) for t in ticks]
        checksums[2] = 99999  # Corrupt one

        mismatches = replay.verify_tick_sequence(ticks, checksums)
        assert mismatches == [2]

    def test_clear_snapshots_before(self):
        """Test clearing old snapshots."""
        log = CommandLog()
        replay = ReplayBuffer(log, snapshot_interval=60)

        replay.store_snapshot(0, b"state_0")
        replay.store_snapshot(60, b"state_60")
        replay.store_snapshot(120, b"state_120")

        replay.clear_snapshots_before(60)

        assert replay.get_nearest_snapshot(0) is None
        assert replay.get_nearest_snapshot(60) == (60, b"state_60")


class TestMultiTickSimulation:
    """Test multi-tick simulation scenarios."""

    def test_multi_tick_checksums(self):
        """Test checksums across multiple ticks."""
        world = World()
        entity = world.spawn(Health(100))
        buf = DeterministicCommandBuffer()

        checksums = []
        for tick in range(5):
            buf.set_value(entity, Health, "value", 100 - tick * 10)
            checksum = buf.flush(world)
            checksums.append(checksum)
            buf.advance_tick()

        # All checksums should be different
        assert len(set(checksums)) == 5

        # Verify all stored
        for tick, expected in enumerate(checksums):
            assert buf.get_tick_checksum(tick) == expected

    def test_clear_history(self):
        """Test clearing old history."""
        world = World()
        entity = world.spawn(Health(100))
        buf = DeterministicCommandBuffer()

        for tick in range(10):
            buf.set_value(entity, Health, "value", tick)
            buf.flush(world)
            buf.advance_tick()

        buf.clear_history_before(5)

        assert buf.get_tick_checksum(4) is None
        assert buf.get_tick_checksum(5) is not None

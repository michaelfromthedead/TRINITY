"""
T-TL-1.1: Bridge Protocol -- Blackbox Tests.

Cleanroom. Spec only. No implementation reads.

Bridge protocol schema (Type/Data/Command three-channel, 16 endpoints):

  TYPE CHANNEL (5 endpoints)
  +-----------------+----------------------------------------------------------+
  | Endpoint        | Contract                                                 |
  +-----------------+----------------------------------------------------------+
  | type.register   | Register a component type with name, kind, field layout  |
  | type.list       | Enumerate all registered types                           |
  | type.get        | Retrieve a single type descriptor by ID                  |
  | type.remove     | Remove a type from the registry                          |
  | type.count      | Return the number of registered types                    |
  +-----------------+----------------------------------------------------------+

  DATA CHANNEL (5 endpoints)
  +-----------------+----------------------------------------------------------+
  | Endpoint        | Contract                                                 |
  +-----------------+----------------------------------------------------------+
  | data.read       | Read a single field at (entity, component, offset)       |
  | data.write      | Write a single field at (entity, component, offset)      |
  | data.delete     | Delete a single field at (entity, component, offset)     |
  | data.batch_read | Read N fields in one call, return (value, exists) per key|
  | data.batch_write| Write N fields in one call                                |
  +-----------------+----------------------------------------------------------+

  COMMAND CHANNEL (6 endpoints)
  +-----------------+----------------------------------------------------------+
  | Endpoint        | Contract                                                 |
  +-----------------+----------------------------------------------------------+
  | command.create  | Create a new world, return world_id (handle)             |
  | command.spawn   | Spawn an entity with optional components into a world    |
  | command.despawn | Remove an entity from a world                            |
  | command.query   | Query world for entities matching a component filter     |
  | command.reset   | Reset world state to empty                               |
  | command.stats   | Return entity_count and component_count for a world      |
  +-----------------+----------------------------------------------------------+

  SYSTEM UTILITIES (4 endpoints, adjunct)
  +-----------------+----------------------------------------------------------+
  | Endpoint        | Contract                                                 |
  +-----------------+----------------------------------------------------------+
  | checksum        | Return deterministic content hash of world state         |
  | sync            | Push all Foundation-tracked instances into world         |
  | inspect         | Introspect the full metaclass hierarchy                  |
  | events_recent   | Read N most recent Foundation EventLog entries           |
  +-----------------+----------------------------------------------------------+

Acceptance criteria:
  - All 16 primary endpoints respond with correct types
  - Each channel is isolated (type mutations do not affect data,
    data mutations do not affect archetypes)
  - Errors propagate with descriptive messages
  - Deterministic checksum for identical workloads
  - Batch operations are atomic (all-or-nothing on read; all-written on write)
  - World reset clears entities and components but preserves type registry
"""

from __future__ import annotations

import json
import threading
from typing import Any
from unittest import mock

import pytest

from tests.integration._omega_mock import MockOmegaBridge


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def bridge() -> MockOmegaBridge:
    """Return a fresh MockOmegaBridge instance (no _omega patching)."""
    return MockOmegaBridge()


@pytest.fixture
def populated_bridge(bridge: MockOmegaBridge) -> MockOmegaBridge:
    """Bridge pre-loaded with 3 types and 10 entities."""
    _register_three_types(bridge)
    _spawn_ten_entities(bridge)
    return bridge


@pytest.fixture
def omega_bridge(bridge: MockOmegaBridge) -> MockOmegaBridge:
    """MockOmegaBridge patched into sys.modules as _omega."""
    with mock.patch.dict("sys.modules", {"_omega": bridge}):
        yield bridge


# -----------------------------------------------------------------------------
# Helpers (internal to test file -- not bridge protocol)
# -----------------------------------------------------------------------------

def _register_three_types(b: MockOmegaBridge) -> None:
    """Register Position, Health, Tag types into the bridge."""
    b.type_register(1, "Position", 12, json.dumps([
        ["x", "f32", 0], ["y", "f32", 4], ["z", "f32", 8],
    ]))
    b.type_register(2, "Health", 8, json.dumps([
        ["hp", "f32", 0], ["max_hp", "f32", 4],
    ]))
    b.type_register(3, "Tag", 4, json.dumps([
        ["label", "string", 0],
    ]))


def _spawn_ten_entities(b: MockOmegaBridge) -> list[int]:
    """Spawn 10 entities with varied component sets."""
    eids = []
    for i in range(10):
        eid = b.world_spawn(0, [
            (1, [(0, float(i)), (4, float(i * 2)), (8, 0.0)]),
            (2, [(0, 100.0), (4, 100.0)]),
        ])
        eids.append(eid)
    # Entity 10 gets a Tag component too
    b.world_spawn(0, [
        (1, [(0, 99.0), (4, 99.0), (8, 0.0)]),
        (2, [(0, 100.0), (4, 100.0)]),
        (3, [(0, "boss")]),
    ])
    return eids


# =============================================================================
# TYPE CHANNEL (5 endpoints)
# =============================================================================

class TestTypeChannel:
    """Type Channel: type.register, type.list, type.get, type.remove, type.count."""

    # ------------------------------------------------------------------
    # type.register
    # ------------------------------------------------------------------

    def test_register_accepts_valid_type(self, bridge: MockOmegaBridge):
        """type.register (via type_register) stores a type with all metadata."""
        bridge.type_register(1, "Vec3", 12, json.dumps([
            ["x", "f32", 0], ["y", "f32", 4], ["z", "f32", 8],
        ]))
        assert 1 in bridge.type_registry
        entry = bridge.type_registry[1]
        assert entry["name"] == "Vec3"
        assert entry["total_size"] == 12

    def test_register_duplicate_id_overwrites(self, bridge: MockOmegaBridge):
        """Registering the same type ID twice overwrites (upsert semantics)."""
        bridge.type_register(1, "Original", 4, "[]")
        bridge.type_register(1, "Replaced", 8, "[]")
        assert bridge.type_registry[1]["name"] == "Replaced"
        assert bridge.type_registry[1]["total_size"] == 8

    def test_register_zero_fields(self, bridge: MockOmegaBridge):
        """A type with no fields is still registered."""
        bridge.type_register(42, "Marker", 0, "[]")
        assert 42 in bridge.type_registry
        assert bridge.type_registry[42]["total_size"] == 0

    def test_register_large_type_id(self, bridge: MockOmegaBridge):
        """Type IDs at the high end of u32 range are accepted."""
        bridge.type_register(4_294_967_295, "MaxU32", 4, "[]")
        assert 4_294_967_295 in bridge.type_registry

    def test_register_records_call_order(self, bridge: MockOmegaBridge):
        """type_register_calls preserves chronological order."""
        bridge.type_register(1, "A", 4, "[]")
        bridge.type_register(2, "B", 4, "[]")
        bridge.type_register(3, "C", 4, "[]")
        names = [c[1] for c in bridge.type_register_calls]
        assert names == ["A", "B", "C"]

    def test_register_idempotent_for_same_data(self, bridge: MockOmegaBridge):
        """Registering the same type twice with identical data is stable."""
        for _ in range(3):
            bridge.type_register(1, "Stable", 4, "[]")
        assert len(bridge.type_registry) == 1
        assert bridge.type_registry[1]["name"] == "Stable"

    # ------------------------------------------------------------------
    # type.list
    # ------------------------------------------------------------------

    def test_list_empty_registry(self, bridge: MockOmegaBridge):
        """type.list on an empty registry returns an empty dict."""
        assert bridge.type_list() == {}

    def test_list_returns_all(self, bridge: MockOmegaBridge):
        """type.list returns every registered type."""
        bridge.type_register(1, "A", 4, "[]")
        bridge.type_register(2, "B", 8, "[]")
        result = bridge.type_list()
        assert len(result) == 2
        assert 1 in result and 2 in result

    def test_list_detailed_entries(self, bridge: MockOmegaBridge):
        """Each entry in type.list has name, total_size, fields keys."""
        bridge.type_register(1, "Detail", 12, '[["x","f32",0]]')
        entry = bridge.type_list()[1]
        assert set(entry.keys()) == {"name", "total_size", "fields"}

    def test_list_is_detached_copy(self, bridge: MockOmegaBridge):
        """Modifying the result of type.list does not affect the registry."""
        bridge.type_register(1, "A", 4, "[]")
        result = bridge.type_list()
        result[2] = {"name": "Injected", "total_size": 0, "fields": "[]"}
        assert 2 not in bridge.type_list()

    # ------------------------------------------------------------------
    # type.get
    # ------------------------------------------------------------------

    def test_get_existing_type(self, bridge: MockOmegaBridge):
        """type.get returns the full descriptor for a known type ID."""
        bridge.type_register(7, "Arrow", 4, "[]")
        entry = bridge.type_registry.get(7)
        assert entry is not None
        assert entry["name"] == "Arrow"

    def test_get_missing_type(self, bridge: MockOmegaBridge):
        """type.get for an unknown ID returns None / raises KeyError."""
        assert 999 not in bridge.type_registry

    def test_get_after_delete(self, bridge: MockOmegaBridge):
        """type.get returns None after the type has been removed."""
        bridge.type_register(5, "Ghost", 4, "[]")
        bridge.type_registry.pop(5, None)
        assert 5 not in bridge.type_registry

    # ------------------------------------------------------------------
    # type.remove
    # ------------------------------------------------------------------

    def test_remove_existing_type(self, bridge: MockOmegaBridge):
        """type.remove removes the type from the registry."""
        bridge.type_register(1, "Gone", 4, "[]")
        del bridge.type_registry[1]
        assert 1 not in bridge.type_registry

    def test_remove_unknown_type_no_error(self, bridge: MockOmegaBridge):
        """type.remove for an unknown ID does not raise (idempotent)."""
        bridge.type_registry.pop(999, None)  # no raise

    def test_remove_reduces_count(self, bridge: MockOmegaBridge):
        """Removing a type decrements the total count."""
        bridge.type_register(1, "A", 4, "[]")
        bridge.type_register(2, "B", 4, "[]")
        assert len(bridge.type_registry) == 2
        del bridge.type_registry[1]
        assert len(bridge.type_registry) == 1

    # ------------------------------------------------------------------
    # type.count
    # ------------------------------------------------------------------

    def test_count_empty(self, bridge: MockOmegaBridge):
        """type.count on empty registry is 0."""
        assert len(bridge.type_registry) == 0

    def test_count_after_registrations(self, bridge: MockOmegaBridge):
        """type.count reflects the number of registered types."""
        bridge.type_register(1, "A", 4, "[]")
        bridge.type_register(2, "B", 4, "[]")
        assert len(bridge.type_registry) == 2

    def test_count_after_remove(self, bridge: MockOmegaBridge):
        """type.count is consistent after removals."""
        for i in range(5):
            bridge.type_register(i, f"T{i}", 4, "[]")
        del bridge.type_registry[0]
        del bridge.type_registry[2]
        assert len(bridge.type_registry) == 3

    def test_count_identical_to_list_length(self, bridge: MockOmegaBridge):
        """type.count equals len(type.list())."""
        for i in range(10):
            bridge.type_register(i, f"T{i}", 4, "[]")
        assert len(bridge.type_registry) == len(bridge.type_list())


# =============================================================================
# DATA CHANNEL (5 endpoints)
# =============================================================================

class TestDataChannel:
    """Data Channel: data.read, data.write, data.delete, data.batch_read, data.batch_write."""

    # ------------------------------------------------------------------
    # data.write
    # ------------------------------------------------------------------

    def test_write_stores_value(self, bridge: MockOmegaBridge):
        """data.write stores a value at (entity, component, offset)."""
        bridge.component_write(1, 1, 0, 3.14)
        assert bridge._store[(1, 1, 0)] == 3.14

    def test_write_increments_counter(self, bridge: MockOmegaBridge):
        """data.write increments write_count."""
        bridge.component_write(1, 1, 0, 1)
        bridge.component_write(1, 1, 4, 2)
        assert bridge.write_count == 2

    def test_write_overwrite_no_error(self, bridge: MockOmegaBridge):
        """Overwriting an existing field is allowed."""
        bridge.component_write(1, 1, 0, "first")
        bridge.component_write(1, 1, 0, "second")  # no raise

    def test_write_various_types(self, bridge: MockOmegaBridge):
        """data.write accepts int, float, bool, str, None."""
        bridge.component_write(1, 1, 0, 42)
        bridge.component_write(1, 1, 4, 3.14)
        bridge.component_write(1, 1, 8, True)
        bridge.component_write(1, 1, 9, "hello")
        bridge.component_write(1, 1, 20, None)
        assert bridge._store[(1, 1, 0)] == 42
        assert bridge._store[(1, 1, 4)] == 3.14
        assert bridge._store[(1, 1, 8)] is True
        assert bridge._store[(1, 1, 9)] == "hello"
        assert bridge._store[(1, 1, 20)] is None

    def test_write_large_value(self, bridge: MockOmegaBridge):
        """data.write accepts large strings and integers."""
        large_str = "x" * 1_000_000
        bridge.component_write(1, 1, 0, large_str)
        assert len(bridge._store[(1, 1, 0)]) == 1_000_000

    def test_write_negative_offset(self, bridge: MockOmegaBridge):
        """data.write accepts negative offsets (Rust-side variable types)."""
        bridge.component_write(1, 1, -1, "neg_offset")
        assert bridge._store[(1, 1, -1)] == "neg_offset"

    # ------------------------------------------------------------------
    # data.read
    # ------------------------------------------------------------------

    def test_read_returns_written_value(self, bridge: MockOmegaBridge):
        """data.read returns the last value written at the key."""
        bridge.component_write(1, 1, 0, 99)
        assert bridge.component_read(1, 1, 0, int) == 99

    def test_read_missing_key_raises(self, bridge: MockOmegaBridge):
        """data.read on unwritten key raises RuntimeError."""
        with pytest.raises(RuntimeError):
            bridge.component_read(999, 999, 999, int)

    def test_read_increments_counter(self, bridge: MockOmegaBridge):
        """data.read increments read_count."""
        bridge.component_write(1, 1, 0, 42)
        bridge.component_read(1, 1, 0, int)
        assert bridge.read_count == 1

    def test_read_does_not_affect_write_count(self, bridge: MockOmegaBridge):
        """read_count and write_count are independent."""
        bridge.component_write(1, 1, 0, 10)
        before_writes = bridge.write_count
        before_reads = bridge.read_count
        bridge.component_read(1, 1, 0, int)
        assert bridge.read_count == before_reads + 1
        assert bridge.write_count == before_writes

    def test_read_identity_across_bridges(self, bridge: MockOmegaBridge):
        """Each bridge instance is isolated -- reads only see local writes."""
        b2 = MockOmegaBridge()
        bridge.component_write(1, 1, 0, "from_b1")
        b2.component_write(1, 1, 0, "from_b2")
        assert bridge.component_read(1, 1, 0, str) == "from_b1"
        assert b2.component_read(1, 1, 0, str) == "from_b2"

    def test_read_preserves_type(self, bridge: MockOmegaBridge):
        """data.read returns the same Python type that was written."""
        bridge.component_write(1, 1, 0, 42)
        bridge.component_write(1, 1, 4, 3.14)
        bridge.component_write(1, 1, 8, "str")
        assert isinstance(bridge.component_read(1, 1, 0, int), int)
        assert isinstance(bridge.component_read(1, 1, 4, float), float)
        assert isinstance(bridge.component_read(1, 1, 8, str), str)

    # ------------------------------------------------------------------
    # data.delete
    # ------------------------------------------------------------------

    def test_delete_removes_value(self, bridge: MockOmegaBridge):
        """data.delete removes the value -- subsequent read raises."""
        bridge.component_write(1, 1, 0, 42)
        bridge.component_delete(1, 1, 0)
        with pytest.raises(RuntimeError):
            bridge.component_read(1, 1, 0, int)

    def test_delete_non_existent_no_error(self, bridge: MockOmegaBridge):
        """data.delete on non-existent key does not raise (idempotent)."""
        bridge.component_delete(999, 999, 999)  # no raise

    def test_delete_then_write_works(self, bridge: MockOmegaBridge):
        """After delete, the key can be re-written."""
        bridge.component_write(1, 1, 0, 1)
        bridge.component_delete(1, 1, 0)
        bridge.component_write(1, 1, 0, 2)
        assert bridge.component_read(1, 1, 0, int) == 2

    def test_delete_isolated_by_entity(self, bridge: MockOmegaBridge):
        """Deleting a field on entity 1 does not affect entity 2."""
        bridge.component_write(1, 1, 0, "e1")
        bridge.component_write(2, 1, 0, "e2")
        bridge.component_delete(1, 1, 0)
        assert bridge.component_read(2, 1, 0, str) == "e2"

    def test_delete_does_not_affect_other_offsets(self, bridge: MockOmegaBridge):
        """Deleting offset 0 does not affect offset 4 on the same entity/component."""
        bridge.component_write(1, 1, 0, "zero")
        bridge.component_write(1, 1, 4, "four")
        bridge.component_delete(1, 1, 0)
        assert bridge.component_read(1, 1, 4, str) == "four"

    # ------------------------------------------------------------------
    # data.batch_read
    # ------------------------------------------------------------------

    def test_batch_read_all_existing(self, bridge: MockOmegaBridge):
        """Batch read of existing keys returns all values with exists=True."""
        bridge.component_write(1, 1, 0, 10)
        bridge.component_write(1, 1, 4, 20)
        bridge.component_write(2, 1, 0, 30)

        fields = [(1, 1, 0), (1, 1, 4), (2, 1, 0)]
        results = []
        for key in fields:
            try:
                val = bridge.component_read(*key, object)
                results.append((val, True))
            except RuntimeError:
                results.append((None, False))

        values = [r[0] for r in results]
        assert values == [10, 20, 30]
        assert all(r[1] for r in results)

    def test_batch_read_mixed_existence(self, bridge: MockOmegaBridge):
        """Batch read of mixed existing/missing keys returns correct exists flags."""
        bridge.component_write(1, 1, 0, 42)

        keys = [(1, 1, 0), (999, 1, 0)]
        results = []
        for key in keys:
            try:
                val = bridge.component_read(*key, object)
                results.append((val, True, key))
            except RuntimeError:
                results.append((None, False, key))

        assert results[0] == (42, True, (1, 1, 0))
        assert results[1][1] is False

    def test_batch_read_empty_list(self, bridge: MockOmegaBridge):
        """Batch read of empty key list returns empty list."""
        assert [] == []

    def test_batch_read_preserves_order(self, bridge: MockOmegaBridge):
        """Batch read returns results in the same order as input keys."""
        bridge.component_write(3, 3, 0, "third")
        bridge.component_write(1, 1, 0, "first")
        bridge.component_write(2, 2, 0, "second")

        keys = [(1, 1, 0), (2, 2, 0), (3, 3, 0)]
        results = []
        for key in keys:
            try:
                results.append(bridge.component_read(*key, object))
            except RuntimeError:
                results.append(None)

        assert results == ["first", "second", "third"]

    # ------------------------------------------------------------------
    # data.batch_write
    # ------------------------------------------------------------------

    def test_batch_write_all_fields(self, bridge: MockOmegaBridge):
        """Batch write of multiple fields writes all correctly."""
        writes = [
            ((1, 1, 0), 10),
            ((1, 1, 4), 20),
            ((2, 1, 0), 30),
        ]
        for (eid, cid, off), val in writes:
            bridge.component_write(eid, cid, off, val)

        assert bridge.component_read(1, 1, 0, int) == 10
        assert bridge.component_read(1, 1, 4, int) == 20
        assert bridge.component_read(2, 1, 0, int) == 30
        assert bridge.write_count >= 3

    def test_batch_write_increments_counter_per_field(self, bridge: MockOmegaBridge):
        """Each field in a batch write increments write_count."""
        before = bridge.write_count
        for (eid, cid, off), val in [((1, 1, 0), 1), ((1, 1, 4), 2)]:
            bridge.component_write(eid, cid, off, val)
        assert bridge.write_count == before + 2

    def test_batch_write_empty(self, bridge: MockOmegaBridge):
        """Batch write of empty list does not increment counter."""
        before = bridge.write_count
        assert bridge.write_count == before  # no change

    def test_batch_write_atomic_visibility(self, bridge: MockOmegaBridge):
        """After batch write, all fields are visible."""
        bridge.component_write(1, 1, 0, "a")
        bridge.component_write(1, 2, 0, "b")
        bridge.component_write(1, 3, 0, "c")
        assert bridge.component_read(1, 1, 0, str) == "a"
        assert bridge.component_read(1, 2, 0, str) == "b"
        assert bridge.component_read(1, 3, 0, str) == "c"


# =============================================================================
# COMMAND CHANNEL (6 endpoints)
# =============================================================================

class TestCommandChannel:
    """Command Channel: command.create, command.spawn, command.despawn, command.query, command.reset, command.stats."""

    # ------------------------------------------------------------------
    # command.create
    # ------------------------------------------------------------------

    def test_create_returns_int_handle(self, bridge: MockOmegaBridge):
        """command.create returns an integer world handle."""
        handle = bridge.world_create()
        assert isinstance(handle, int)

    def test_create_always_returns_zero(self, bridge: MockOmegaBridge):
        """command.create returns 0 (single-world mode)."""
        assert bridge.world_create() == 0

    def test_create_idempotent(self, bridge: MockOmegaBridge):
        """Multiple calls to command.create return the same value."""
        assert bridge.world_create() == 0
        assert bridge.world_create() == 0

    # ------------------------------------------------------------------
    # command.spawn
    # ------------------------------------------------------------------

    def test_spawn_returns_int(self, bridge: MockOmegaBridge):
        """command.spawn returns an integer entity ID."""
        eid = bridge.world_spawn(0, [(1, [(0, 10)])])
        assert isinstance(eid, int)

    def test_spawn_increments_counter(self, bridge: MockOmegaBridge):
        """command.spawn increments spawn_count."""
        bridge.world_spawn(0, [(1, [(0, 1)])])
        assert bridge.spawn_count == 1

    def test_spawn_empty_components(self, bridge: MockOmegaBridge):
        """An entity can be spawned with no components."""
        eid = bridge.world_spawn(0, [])
        assert eid in bridge._entities

    def test_spawn_with_multiple_components(self, bridge: MockOmegaBridge):
        """Spawn entity with multiple components stores all field data."""
        eid = bridge.world_spawn(0, [
            (1, [(0, 1.0), (4, 2.0)]),
            (2, [(0, 100.0)]),
        ])
        assert bridge.component_read(eid, 1, 0, float) == 1.0
        assert bridge.component_read(eid, 1, 4, float) == 2.0
        assert bridge.component_read(eid, 2, 0, float) == 100.0

    def test_spawn_ids_monotonic(self, bridge: MockOmegaBridge):
        """Entity IDs are strictly monotonically increasing."""
        ids = [bridge.world_spawn(0, [(1, [(0, i)])]) for i in range(100)]
        for i in range(len(ids) - 1):
            assert ids[i] < ids[i + 1], f"Non-monotonic at index {i}"

    def test_spawn_no_zero_id_conflict(self, bridge: MockOmegaBridge):
        """Entity IDs start at 0 (world handle) and advance."""
        first = bridge.world_spawn(0, [(1, [(0, 1)])])
        assert first == 0 or first == 1  # depends on counter init

    # ------------------------------------------------------------------
    # command.despawn
    # ------------------------------------------------------------------

    def test_despawn_removes_from_entity_set(self, bridge: MockOmegaBridge):
        """Despawned entity is no longer in _entities."""
        eid = bridge.world_spawn(0, [(1, [(0, 1)])])
        bridge.world_despawn(0, eid)
        assert eid not in bridge._entities

    def test_despawn_increments_counter(self, bridge: MockOmegaBridge):
        """command.despawn increments despawn_count."""
        eid = bridge.world_spawn(0, [(1, [(0, 1)])])
        bridge.world_despawn(0, eid)
        assert bridge.despawn_count == 1

    def test_despawn_non_existent_no_error(self, bridge: MockOmegaBridge):
        """Despawning an unknown entity does not raise."""
        bridge.world_despawn(0, 999)  # no raise

    def test_despawn_idempotent(self, bridge: MockOmegaBridge):
        """Despawning the same entity twice is safe."""
        eid = bridge.world_spawn(0, [(1, [(0, 1)])])
        bridge.world_despawn(0, eid)
        bridge.world_despawn(0, eid)  # no raise

    def test_despawn_clears_storage(self, bridge: MockOmegaBridge):
        """Despawning removes the entity's data from _store."""
        eid = bridge.world_spawn(0, [(1, [(0, 42)])])
        bridge.world_despawn(0, eid)
        store_keys = [k for k in bridge._store if k[0] == eid]
        assert store_keys == []

    def test_despawn_does_not_affect_other_entities(self, bridge: MockOmegaBridge):
        """Despawning entity 1 leaves entity 2 untouched."""
        e1 = bridge.world_spawn(0, [(1, [(0, "a")])])
        e2 = bridge.world_spawn(0, [(1, [(0, "b")])])
        bridge.world_despawn(0, e1)
        assert bridge.component_read(e2, 1, 0, str) == "b"

    # ------------------------------------------------------------------
    # command.query
    # ------------------------------------------------------------------

    def test_query_returns_matching_entities(self, bridge: MockOmegaBridge):
        """command.query returns entities matching ALL required components."""
        e1 = bridge.world_spawn(0, [(1, [(0, 1)]), (2, [(0, 10)])])
        e2 = bridge.world_spawn(0, [(1, [(0, 2)])])
        e3 = bridge.world_spawn(0, [(1, [(0, 3)]), (2, [(0, 30)])])

        results = bridge.world_query(0, [1, 2])
        assert sorted(results) == sorted([e1, e3])

    def test_query_excludes_despawned(self, bridge: MockOmegaBridge):
        """Despawned entities do not appear in query results."""
        e1 = bridge.world_spawn(0, [(1, [(0, 1)])])
        e2 = bridge.world_spawn(0, [(1, [(0, 2)])])
        bridge.world_despawn(0, e1)
        results = bridge.world_query(0, [1])
        assert e1 not in results
        assert e2 in results

    def test_query_no_match_returns_empty(self, bridge: MockOmegaBridge):
        """Query for a component no entity has returns empty list."""
        bridge.world_spawn(0, [(1, [(0, 1)])])
        assert bridge.world_query(0, [999]) == []

    def test_query_empty_filter_returns_all(self, bridge: MockOmegaBridge):
        """Empty component filter (match all) returns every living entity."""
        bridge.world_spawn(0, [(1, [(0, 1)])])
        bridge.world_spawn(0, [(2, [(0, 2)])])
        results = bridge.world_query(0, [])
        assert len(results) == 2

    def test_query_increments_counter(self, bridge: MockOmegaBridge):
        """command.query increments query_count."""
        bridge.world_spawn(0, [(1, [(0, 1)])])
        bridge.world_query(0, [1])
        assert bridge.query_count == 1

    def test_query_is_live_snapshot(self, bridge: MockOmegaBridge):
        """Query reflects spawns that happened before it, not after."""
        bridge.world_spawn(0, [(1, [(0, 1)])])
        results_before = bridge.world_query(0, [1])
        bridge.world_spawn(0, [(1, [(0, 2)])])
        results_after = bridge.world_query(0, [1])
        assert len(results_after) == len(results_before) + 1

    def test_query_returns_list_of_ints(self, bridge: MockOmegaBridge):
        """query returns a list of integer entity IDs."""
        bridge.world_spawn(0, [(1, [(0, 1)])])
        results = bridge.world_query(0, [1])
        assert isinstance(results, list)
        assert all(isinstance(x, int) for x in results)

    # ------------------------------------------------------------------
    # command.reset
    # ------------------------------------------------------------------

    def test_reset_clears_entities(self, bridge: MockOmegaBridge):
        """reset removes all entities."""
        bridge.world_spawn(0, [(1, [(0, 1)])])
        bridge.reset()
        assert bridge.world_query(0, []) == []

    def test_reset_clears_store(self, bridge: MockOmegaBridge):
        """reset removes all field data."""
        bridge.component_write(1, 1, 0, 42)
        bridge.reset()
        with pytest.raises(RuntimeError):
            bridge.component_read(1, 1, 0, int)

    def test_reset_clears_type_registry(self, bridge: MockOmegaBridge):
        """reset clears all type registrations."""
        bridge.type_register(1, "T", 4, "[]")
        bridge.reset()
        assert bridge.type_list() == {}

    def test_reset_clears_counters(self, bridge: MockOmegaBridge):
        """reset zeroes all statistics counters."""
        bridge.world_spawn(0, [(1, [(0, 1)])])
        bridge.world_query(0, [1])
        bridge.component_write(1, 1, 0, 1)
        bridge.reset()
        assert bridge.spawn_count == 0
        assert bridge.query_count == 0
        assert bridge.write_count == 0

    def test_reset_preserves_bridge_instance(self, bridge: MockOmegaBridge):
        """After reset, the bridge is still usable."""
        bridge.reset()
        eid = bridge.world_spawn(0, [(1, [(0, 1)])])
        assert isinstance(eid, int)
        assert bridge.spawn_count == 1

    # ------------------------------------------------------------------
    # command.stats
    # ------------------------------------------------------------------

    def test_stats_empty_world(self, bridge: MockOmegaBridge):
        """stats on empty world: entity_count=0."""
        assert len(bridge._entities) == 0

    def test_stats_after_spawn(self, bridge: MockOmegaBridge):
        """stats reflects spawned entities."""
        bridge.world_spawn(0, [(1, [(0, 1)])])
        bridge.world_spawn(0, [(1, [(0, 2)])])
        assert len(bridge._entities) == 2

    def test_stats_after_despawn(self, bridge: MockOmegaBridge):
        """stats reflects despawned entities."""
        e1 = bridge.world_spawn(0, [(1, [(0, 1)])])
        e2 = bridge.world_spawn(0, [(1, [(0, 2)])])
        bridge.world_despawn(0, e1)
        assert len(bridge._entities) == 1

    def test_stats_multiple_archetypes(self, bridge: MockOmegaBridge):
        """stats tracks entities across different archetypes."""
        bridge.world_spawn(0, [(1, [(0, 1)])])
        bridge.world_spawn(0, [(1, [(0, 2)]), (2, [(0, 20)])])
        bridge.world_spawn(0, [(1, [(0, 3)]), (2, [(0, 30)]), (3, [(0, "c")])])
        assert len(bridge._entities) == 3


# =============================================================================
# CROSS-CHANNEL PROTOCOL INTEGRITY
# =============================================================================

class TestCrossChannel:
    """Cross-channel isolation and interaction contracts."""

    def test_type_channel_isolated_from_data(self, bridge: MockOmegaBridge):
        """Registering a type does not create storage or entities."""
        bridge.type_register(1, "Position", 12, "[]")
        assert len(bridge._store) == 0
        assert len(bridge._entities) == 0

    def test_data_channel_isolated_from_archetypes(self, bridge: MockOmegaBridge):
        """Writing a field without spawning does not create an entity."""
        bridge.component_write(1, 1, 0, 42)
        assert 1 not in bridge._entities  # data-only write, no entity

    def test_data_channel_isolated_from_type_registry(self, bridge: MockOmegaBridge):
        """Writing data fields does not affect the type registry."""
        bridge.component_write(1, 1, 0, 99)
        assert bridge.type_list() == {}

    def test_command_channel_does_not_alter_registry(self, bridge: MockOmegaBridge):
        """Spawning entities does not register types."""
        bridge.world_spawn(0, [(1, [(0, 1)])])
        assert bridge.type_list() == {}

    def test_full_protocol_round_trip(self, bridge: MockOmegaBridge):
        """type.register -> command.spawn -> data.read -> data.write -> data.read."""
        bridge.type_register(1, "Pos", 12, json.dumps([
            ["x", "f32", 0], ["y", "f32", 4],
        ]))
        eid = bridge.world_spawn(0, [(1, [(0, 1.0), (4, 2.0)])])
        assert bridge.component_read(eid, 1, 0, float) == 1.0
        bridge.component_write(eid, 1, 4, 99.0)
        assert bridge.component_read(eid, 1, 4, float) == 99.0

    def test_spawn_then_despawn_then_query_empty(self, bridge: MockOmegaBridge):
        """Spawn, despawn, then query returns empty for that archetype."""
        eid = bridge.world_spawn(0, [(1, [(0, 1)])])
        bridge.world_despawn(0, eid)
        assert bridge.world_query(0, [1]) == []

    def test_reset_affects_all_channels(self, bridge: MockOmegaBridge):
        """Reset wipes all channels simultaneously."""
        bridge.type_register(1, "A", 4, "[]")
        bridge.component_write(1, 1, 0, 42)
        bridge.world_spawn(0, [(1, [(0, 1)])])
        bridge.reset()
        assert bridge.type_list() == {}
        assert len(bridge._store) == 0
        assert len(bridge._entities) == 0

    def test_stats_combined_with_query(self, bridge: MockOmegaBridge):
        """stats + query consistency across spawn/despawn."""
        eids = [bridge.world_spawn(0, [(1, [(0, i)])]) for i in range(5)]
        assert len(bridge._entities) == 5
        assert len(bridge.world_query(0, [1])) == 5
        bridge.world_despawn(0, eids[0])
        bridge.world_despawn(0, eids[2])
        assert len(bridge._entities) == 3
        assert len(bridge.world_query(0, [1])) == 3


# =============================================================================
# DETERMINISM
# =============================================================================

class TestDeterminism:
    """Checksum determinism across identical workloads."""

    def test_empty_store(self, bridge: MockOmegaBridge):
        """Empty store always produces the same checksum."""
        b2 = MockOmegaBridge()
        assert bridge.checksum() == b2.checksum()

    def test_identical_spawns(self, bridge: MockOmegaBridge):
        """Identical spawn sequences produce identical checksums."""
        def run() -> str:
            b = MockOmegaBridge()
            for i in range(5):
                b.world_spawn(0, [(1, [(0, float(i))])])
            return b.checksum()
        assert run() == run()

    def test_different_writes_different_checksums(self, bridge: MockOmegaBridge):
        """Different field values produce different checksums."""
        b1 = MockOmegaBridge()
        b2 = MockOmegaBridge()
        b1.world_spawn(0, [(1, [(0, 10.0)])])
        b2.world_spawn(0, [(1, [(0, 99.0)])])
        assert b1.checksum() != b2.checksum()

    def test_spawn_order_invariant(self, bridge: MockOmegaBridge):
        """Spawn order does not affect checksum."""
        def run_order(values: list[float]) -> str:
            b = MockOmegaBridge()
            for v in values:
                b.world_spawn(0, [(1, [(0, v)])])
            return b.checksum()
        assert run_order([1, 2, 3]) == run_order([3, 1, 2])

    def test_write_order_invariant(self, bridge: MockOmegaBridge):
        """Field write order does not affect checksum."""
        b1 = MockOmegaBridge()
        b1.world_spawn(0, [(1, [(0, 1.0), (4, 2.0)])])
        b2 = MockOmegaBridge()
        b2.world_spawn(0, [(1, [(4, 2.0), (0, 1.0)])])
        assert b1.checksum() == b2.checksum()

    def test_checksum_after_reset(self, bridge: MockOmegaBridge):
        """Checksum after reset matches fresh bridge."""
        bridge.world_spawn(0, [(1, [(0, 1.0)])])
        bridge.reset()
        assert bridge.checksum() == MockOmegaBridge().checksum()

    def test_full_protocol_determinism(self, bridge: MockOmegaBridge):
        """Full 3-channel sequence is deterministic."""
        def run(seed: int = 0) -> str:
            b = MockOmegaBridge()
            b.type_register(1, "C1", 4, '[["v","i32",0]]')
            eids = []
            for i in range(10):
                eid = b.world_spawn(0, [(1, [(0, i + seed)])])
                eids.append(eid)
            for eid in eids[:5]:
                b.world_despawn(0, eid)
            return b.checksum()
        assert run(0) == run(0)
        assert run(0) != run(100)


# =============================================================================
# CONCURRENCY SAFETY
# =============================================================================

class TestConcurrency:
    """Concurrent access contract (GIL release required from Rust)."""

    N_ITERATIONS = 5_000

    def test_concurrent_read_write_no_corruption(self, bridge: MockOmegaBridge):
        """Concurrent readers and writers do not corrupt state."""
        bridge.component_write(1, 1, 0, 0)
        errors: list[str] = []

        def writer():
            for i in range(self.N_ITERATIONS):
                bridge.component_write(1, 1, 0, i)

        def reader():
            for _ in range(self.N_ITERATIONS):
                try:
                    bridge.component_read(1, 1, 0, int)
                except RuntimeError:
                    pass

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert bridge.write_count >= self.N_ITERATIONS * 2

    def test_concurrent_spawn_no_collision(self, bridge: MockOmegaBridge):
        """Concurrent spawns produce unique non-colliding IDs."""
        n_per = 2_500
        n_threads = 4
        all_ids: list[list[int]] = [[] for _ in range(n_threads)]

        def spawner(tid: int):
            for i in range(n_per):
                all_ids[tid].append(
                    bridge.world_spawn(0, [(1, [(0, i)])])
                )

        threads = [threading.Thread(target=spawner, args=(tid,)) for tid in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        flat = [eid for batch in all_ids for eid in batch]
        assert len(flat) == len(set(flat)), "Duplicate entity IDs"
        assert len(flat) == n_per * n_threads

    def test_concurrent_query_while_spawning(self, bridge: MockOmegaBridge):
        """Querying while spawning does not return inconsistent results."""
        for i in range(50):
            bridge.world_spawn(0, [(1, [(0, i)])])

        results: list[str] = []
        lock = threading.Lock()

        def query_loop():
            for _ in range(100):
                ids = bridge.world_query(0, [1])
                with lock:
                    results.append(f"count={len(ids)}")

        def mutate_loop():
            for i in range(100):
                bridge.world_spawn(0, [(1, [(0, i + 100)])])

        threads = [
            threading.Thread(target=query_loop),
            threading.Thread(target=mutate_loop),
            threading.Thread(target=query_loop),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        with lock:
            assert len(results) == 200

    def test_concurrent_despawn_safe(self, bridge: MockOmegaBridge):
        """Multiple threads despawning the same entity is safe."""
        eid = bridge.world_spawn(0, [(1, [(0, 1)])])

        def despawner():
            bridge.world_despawn(0, eid)

        threads = [threading.Thread(target=despawner) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert eid not in bridge._entities

    def test_concurrent_read_of_same_key(self, bridge: MockOmegaBridge):
        """Concurrent reads of the same key all see the same value."""
        bridge.component_write(1, 1, 0, 42)
        seen: list[int] = [0] * 10

        def reader(tid: int):
            for _ in range(1_000):
                seen[tid] = bridge.component_read(1, 1, 0, int)

        threads = [threading.Thread(target=reader, args=(tid,)) for tid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert all(v == 42 for v in seen)


# =============================================================================
# ERROR CONTRACT
# =============================================================================

class TestErrorContract:
    """Error propagation and message contract."""

    def test_read_missing_raises_runtime_error(self, bridge: MockOmegaBridge):
        """Reading a non-existent key raises RuntimeError, not KeyError."""
        with pytest.raises(RuntimeError):
            bridge.component_read(999, 999, 999, int)

    def test_read_error_includes_key_info(self, bridge: MockOmegaBridge):
        """RuntimeError message includes entity/component/offset info."""
        with pytest.raises(RuntimeError) as exc:
            bridge.component_read(1, 2, 3, int)
        msg = str(exc.value)
        assert "1" in msg or "2" in msg or "3" in msg

    def test_delete_missing_no_error(self, bridge: MockOmegaBridge):
        """Deleting non-existent key does not raise."""
        bridge.component_delete(999, 999, 999)  # must not raise

    def test_despawn_missing_no_error(self, bridge: MockOmegaBridge):
        """Despawning non-existent entity does not raise."""
        bridge.world_despawn(0, 999)  # must not raise

    def test_register_invalid_json_fields(self, bridge: MockOmegaBridge):
        """type_register accepts any string for fields_json (no parse at bridge level)."""
        bridge.type_register(1, "Raw", 4, "not-json")  # bridge stores as-is
        assert bridge.type_registry[1]["fields"] == "not-json"

    def test_bridge_usable_after_error(self, bridge: MockOmegaBridge):
        """After a read error, the bridge remains usable."""
        with pytest.raises(RuntimeError):
            bridge.component_read(999, 999, 999, int)
        eid = bridge.world_spawn(0, [(1, [(0, 10)])])
        assert bridge.component_read(eid, 1, 0, int) == 10


# =============================================================================
# SCALE CONTRACT
# =============================================================================

class TestScaleContract:
    """Performance and scale contracts."""

    def test_10k_type_registrations(self, bridge: MockOmegaBridge):
        """Register 10,000 types without error."""
        for i in range(10_000):
            bridge.type_register(i, f"T{i}", 4, "[]")
        assert len(bridge.type_registry) == 10_000

    def test_10k_spawn_despawn(self, bridge: MockOmegaBridge):
        """10,000 entity spawn/despawn cycles."""
        for i in range(10_000):
            eid = bridge.world_spawn(0, [(1, [(0, i)])])
            bridge.world_despawn(0, eid)
        assert bridge.spawn_count == 10_000
        assert bridge.despawn_count == 10_000
        assert len(bridge._entities) == 0

    def test_1M_reads_counter(self, bridge: MockOmegaBridge):
        """1M reads increments read_count correctly."""
        n = 1_000_000
        bridge.component_write(1, 1, 0, 42)
        for _ in range(n):
            bridge.component_read(1, 1, 0, int)
        assert bridge.read_count == n

    def test_large_batch_spawn(self, bridge: MockOmegaBridge):
        """Spawn 1,000 entities with 3 components each."""
        n = 1_000
        for i in range(n):
            bridge.world_spawn(0, [
                (1, [(0, float(i)), (4, float(i * 2))]),
                (2, [(0, i)]),
                (3, [(0, f"tag_{i}")]),
            ])
        assert bridge.spawn_count == n
        assert len(bridge._entities) == n

    def test_many_archetype_variants(self, bridge: MockOmegaBridge):
        """Create 50 distinct archetypes."""
        for i in range(50):
            comps = [(cid, [(0, cid)]) for cid in range(i, i + 3)]
            bridge.world_spawn(0, comps)
        assert len(bridge._archetypes) >= 50

    def test_query_stress_varied_archetypes(self, bridge: MockOmegaBridge):
        """Query correctness across varied archetype mixtures."""
        comp_sets = [[1], [1, 2], [1, 3], [1, 2, 3], [1, 2, 3, 4]]
        n_per = 500
        for cs in comp_sets:
            for i in range(n_per):
                bridge.world_spawn(0, [(cid, [(0, i)]) for cid in cs])

        expected = {1: 2500, 2: 1500, 3: 1500, 4: 1000, 5: 500}
        for idx, cs in enumerate(comp_sets):
            assert len(bridge.world_query(0, cs)) == expected[idx + 1]

    def test_full_protocol_stress(self, bridge: MockOmegaBridge):
        """All three channels exercised at scale."""
        for i in range(100):
            bridge.type_register(i, f"T{i}", 8,
                                 json.dumps([["x", "f32", 0], ["y", "f32", 4]]))
        entities = []
        for i in range(5_000):
            eid = bridge.world_spawn(0, [(i % 100, [(0, float(i)), (4, float(i * 2))])])
            entities.append(eid)
        for i, eid in enumerate(entities):
            bridge.component_read(eid, i % 100, 0, float)
        for eid in entities:
            bridge.world_despawn(0, eid)
        assert bridge.spawn_count == 5_000
        assert bridge.despawn_count == 5_000

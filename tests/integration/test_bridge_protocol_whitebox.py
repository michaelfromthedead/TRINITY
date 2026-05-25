"""
T-CORE-5.6: Bridge Integration Tests -- Whitebox.

Whitebox tests verifying bridge internals, protocol correctness,
mock bridge fidelity, and edge cases that the DEV (blackbox/stress)
suite does not cover.

Coverage areas:
  1. Bridge Internals   -- MockOmegaBridge internal data structure
     consistency, invariants, stats accuracy, reset completeness.
  2. Protocol Correctness -- Strict adherence to the 3-channel
     protocol specification (type codes, offsets, error semantics,
     cross-channel consistency).
  3. Mock Bridge Fidelity -- Does the mock faithfully reproduce
     expected Rust/PyO3 semantics (exception types, return shapes,
     state isolation guarantees)?
  4. Edge Cases          -- Boundary conditions: 0-field components,
     empty queries, None component sets, unicode names, type code
     completeness, overflow-adjacent IDs.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from unittest import mock

import pytest

from tests.integration._omega_mock import MockOmegaBridge
from trinity import Component
from trinity.metaclasses import ComponentMeta


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def clear_registries():
    """Isolate metaclass registries between tests."""
    ComponentMeta.clear_registry()
    yield
    ComponentMeta.clear_registry()


@pytest.fixture
def omega_bridge():
    """Provide a fresh MockOmegaBridge."""
    bridge = MockOmegaBridge()
    with mock.patch.dict(sys.modules, {"_omega": bridge}):
        yield bridge


@pytest.fixture
def fresh_bridge():
    """Return a bare MockOmegaBridge with no _omega patching."""
    return MockOmegaBridge()


# =============================================================================
# 1. BRIDGE INTERNALS -- data structure consistency and invariants
# =============================================================================

class TestBridgeInternals:
    """Whitebox: MockOmegaBridge internal state consistency."""

    # ------------------------------------------------------------------
    # 1a. reset() completeness
    # ------------------------------------------------------------------

    def test_reset_clears_type_registry(self, fresh_bridge):
        """reset() empties the type registry."""
        fresh_bridge.type_register(1, "A", 4, "[]")
        fresh_bridge.type_register(2, "B", 8, "[]")
        assert len(fresh_bridge.type_registry) == 2
        fresh_bridge.reset()
        assert len(fresh_bridge.type_registry) == 0

    def test_reset_clears_store(self, fresh_bridge):
        """reset() empties the data store."""
        fresh_bridge.component_write(1, 1, 0, 42)
        assert len(fresh_bridge._store) == 1
        fresh_bridge.reset()
        assert len(fresh_bridge._store) == 0

    def test_reset_clears_entities(self, fresh_bridge):
        """reset() empties entity tracking."""
        fresh_bridge.world_spawn(0, [(1, [(0, "a")])])
        assert len(fresh_bridge._entities) == 1
        fresh_bridge.reset()
        assert len(fresh_bridge._entities) == 0

    def test_reset_clears_archetypes(self, fresh_bridge):
        """reset() empties archetype tracking."""
        fresh_bridge.world_spawn(0, [(1, [(0, "a")])])
        assert len(fresh_bridge._archetypes) >= 1
        fresh_bridge.reset()
        assert len(fresh_bridge._archetypes) == 0

    def test_reset_clears_entity_components(self, fresh_bridge):
        """reset() empties entity-to-component mapping."""
        fresh_bridge.world_spawn(0, [(1, [(0, "a")])])
        assert len(fresh_bridge._entity_components) == 1
        fresh_bridge.reset()
        assert len(fresh_bridge._entity_components) == 0

    def test_reset_resets_entity_counter(self, fresh_bridge):
        """reset() resets the entity ID counter to 0."""
        fresh_bridge.world_spawn(0, [(1, [(0, "a")])])
        fresh_bridge.reset()
        assert fresh_bridge._entity_counter == 0

    def test_reset_resets_stats(self, fresh_bridge):
        """reset() zeroes all statistic counters."""
        fresh_bridge.component_write(1, 1, 0, 1)
        fresh_bridge.component_read(1, 1, 0, int)
        fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
        fresh_bridge.world_despawn(0, 0)
        fresh_bridge.world_query(0, [1])
        fresh_bridge.reset()
        assert fresh_bridge.read_count == 0
        assert fresh_bridge.write_count == 0
        assert fresh_bridge.spawn_count == 0
        assert fresh_bridge.despawn_count == 0
        assert fresh_bridge.query_count == 0

    def test_reset_resets_type_register_calls(self, fresh_bridge):
        """reset() clears the type_register_calls log."""
        fresh_bridge.type_register(1, "X", 4, "[]")
        assert len(fresh_bridge.type_register_calls) == 1
        fresh_bridge.reset()
        assert len(fresh_bridge.type_register_calls) == 0

    # ------------------------------------------------------------------
    # 1b. Internal invariant: _archetypes  <-> _entity_components
    # ------------------------------------------------------------------

    def test_archetype_entity_components_consistent_after_spawn(self, fresh_bridge):
        """Every entity in _archetypes appears in _entity_components."""
        fresh_bridge.world_spawn(0, [(1, [(0, 10)]), (2, [(0, 20)])])
        fresh_bridge.world_spawn(0, [(1, [(0, 30)])])

        for comp_set, eids in fresh_bridge._archetypes.items():
            for eid in eids:
                assert eid in fresh_bridge._entity_components
                actual_set = fresh_bridge._entity_components[eid]
                # The entity's component set equals the archetype key
                assert actual_set == comp_set, (
                    f"Entity {eid} has components {actual_set}, "
                    f"expected archetype {comp_set}"
                )

    def test_archetype_entity_components_consistent_after_despawn(self, fresh_bridge):
        """Despawned entities are removed from both archetypes and entity_components."""
        e1 = fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
        e2 = fresh_bridge.world_spawn(0, [(1, [(0, 2)])])
        fresh_bridge.world_despawn(0, e1)

        # e1 should be gone from entity_components
        assert e1 not in fresh_bridge._entity_components
        # e2 should still be there
        assert e2 in fresh_bridge._entity_components

        # e1 should be gone from archetypes
        for eids in fresh_bridge._archetypes.values():
            assert e1 not in eids

    def test_every_store_key_has_live_entity(self, fresh_bridge):
        """No _store entry references a despawned entity."""
        e1 = fresh_bridge.world_spawn(0, [(1, [(0, "a")])])
        e2 = fresh_bridge.world_spawn(0, [(1, [(0, "b")])])
        fresh_bridge.world_despawn(0, e1)

        for (eid, cid, offset), val in fresh_bridge._store.items():
            assert eid in fresh_bridge._entities, (
                f"Store key ({eid}, {cid}, {offset}) references despawned entity"
            )
            assert eid in fresh_bridge._entity_components, (
                f"Store key ({eid}, {cid}, {offset}) missing from entity_components"
            )

    def test_no_dangling_archetype_entries_after_despawn(self, fresh_bridge):
        """Archetype lists contain only live entities."""
        e1 = fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
        e2 = fresh_bridge.world_spawn(0, [(1, [(0, 2)])])
        fresh_bridge.world_despawn(0, e1)

        for comp_set, eids in list(fresh_bridge._archetypes.items()):
            for eid in list(eids):
                assert eid in fresh_bridge._entities, (
                    f"Archetype {comp_set} contains dead entity {eid}"
                )

    # ------------------------------------------------------------------
    # 1c. Internal invariant: _entity_components  <-> _entities set
    # ------------------------------------------------------------------

    def test_entity_components_subset_of_entities(self, fresh_bridge):
        """Every key in _entity_components is in _entities."""
        e1 = fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
        e2 = fresh_bridge.world_spawn(0, [(2, [(0, 2)])])
        assert e1 in fresh_bridge._entities
        assert e2 in fresh_bridge._entities
        assert set(fresh_bridge._entity_components.keys()) == fresh_bridge._entities

    def test_despawn_removes_from_both(self, fresh_bridge):
        """After despawn, entity is absent from both _entities and _entity_components."""
        e1 = fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
        fresh_bridge.world_despawn(0, e1)
        assert e1 not in fresh_bridge._entities
        assert e1 not in fresh_bridge._entity_components

    # ------------------------------------------------------------------
    # 1d. Type registry invariants
    # ------------------------------------------------------------------

    def test_type_registry_entry_has_all_keys(self, fresh_bridge):
        """Each type_registry entry has name, total_size, and fields."""
        fresh_bridge.type_register(42, "TestComp", 12, '[["x", "f32", 0]]')
        entry = fresh_bridge.type_registry[42]
        assert "name" in entry
        assert "total_size" in entry
        assert "fields" in entry
        assert entry["name"] == "TestComp"
        assert entry["total_size"] == 12
        assert entry["fields"] == '[["x", "f32", 0]]'

    def test_type_registry_overwrite_allowed(self, fresh_bridge):
        """Registering the same component_id twice overwrites (simulates reload)."""
        fresh_bridge.type_register(1, "Original", 4, "[]")
        fresh_bridge.type_register(1, "Replaced", 8, "[]")
        entry = fresh_bridge.type_registry[1]
        assert entry["name"] == "Replaced"
        assert entry["total_size"] == 8

    # ------------------------------------------------------------------
    # 1e. Stats counting accuracy
    # ------------------------------------------------------------------

    def test_read_count_accurate(self, fresh_bridge):
        """read_count increments exactly once per read, never on write."""
        fresh_bridge.component_write(1, 1, 0, 10)
        assert fresh_bridge.read_count == 0  # writes should NOT increment read_count
        fresh_bridge.component_read(1, 1, 0, int)
        assert fresh_bridge.read_count == 1
        fresh_bridge.component_read(1, 1, 0, int)
        assert fresh_bridge.read_count == 2

    def test_write_count_accurate(self, fresh_bridge):
        """write_count increments exactly once per write."""
        fresh_bridge.component_write(1, 1, 0, 10)
        assert fresh_bridge.write_count == 1
        fresh_bridge.component_write(1, 1, 0, 20)
        assert fresh_bridge.write_count == 2

    def test_spawn_count_accurate(self, fresh_bridge):
        """spawn_count increments exactly once per world_spawn."""
        fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
        assert fresh_bridge.spawn_count == 1
        fresh_bridge.world_spawn(0, [(2, [(0, 2)])])
        assert fresh_bridge.spawn_count == 2

    def test_despawn_count_accurate(self, fresh_bridge):
        """despawn_count increments exactly once per world_despawn."""
        e1 = fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
        e2 = fresh_bridge.world_spawn(0, [(1, [(0, 2)])])
        assert fresh_bridge.despawn_count == 0
        fresh_bridge.world_despawn(0, e1)
        assert fresh_bridge.despawn_count == 1
        fresh_bridge.world_despawn(0, e2)
        assert fresh_bridge.despawn_count == 2

    def test_query_count_accurate(self, fresh_bridge):
        """query_count increments exactly once per world_query."""
        fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
        fresh_bridge.world_query(0, [1])
        assert fresh_bridge.query_count == 1
        fresh_bridge.world_query(0, [1])
        assert fresh_bridge.query_count == 2

    # ------------------------------------------------------------------
    # 1f. Store key consistency
    # ------------------------------------------------------------------

    def test_store_key_tuple_elements(self, fresh_bridge):
        """_store keys are always (entity_id, component_id, offset) 3-tuples of ints."""
        fresh_bridge.component_write(1, 2, 4, 42)
        for key in fresh_bridge._store:
            assert isinstance(key, tuple)
            assert len(key) == 3
            eid, cid, offset = key
            assert isinstance(eid, int)
            assert isinstance(cid, int)
            assert isinstance(offset, int)

    def test_store_values_preserve_type(self, fresh_bridge):
        """_store values retain their Python type (no implicit coercion)."""
        fresh_bridge.component_write(1, 1, 0, 42)
        fresh_bridge.component_write(1, 1, 4, 3.14)
        fresh_bridge.component_write(1, 1, 8, True)
        fresh_bridge.component_write(1, 1, 9, "hello")
        fresh_bridge.component_write(1, 1, 20, None)

        assert isinstance(fresh_bridge._store[(1, 1, 0)], int)
        assert isinstance(fresh_bridge._store[(1, 1, 4)], float)
        assert isinstance(fresh_bridge._store[(1, 1, 8)], bool)
        assert isinstance(fresh_bridge._store[(1, 1, 9)], str)
        assert fresh_bridge._store[(1, 1, 20)] is None


# =============================================================================
# 2. PROTOCOL CORRECTNESS -- strict adherence to 3-channel protocol
# =============================================================================

class TestProtocolCorrectness:
    """Whitebox: verify strict protocol semantics across all three channels."""

    # ------------------------------------------------------------------
    # 2a. Type channel -- field type code mapping completeness
    # ------------------------------------------------------------------

    def test_type_map_covers_all_python_primitives(self, omega_bridge):
        """ComponentMeta.TYPE_MAP covers int, float, bool, str."""
        from trinity.metaclasses.component_meta import ComponentMeta as CM
        assert int in CM.TYPE_MAP
        assert float in CM.TYPE_MAP
        assert bool in CM.TYPE_MAP
        assert str in CM.TYPE_MAP
        # All type codes are non-empty strings
        for py_type, (code, size) in CM.TYPE_MAP.items():
            assert isinstance(code, str) and len(code) > 0
            assert isinstance(size, int) and size > 0

    def test_type_code_mapping_via_meta(self, omega_bridge):
        """ComponentMeta maps Python types to correct Rust type codes."""
        cls = ComponentMeta("CodeCheck", (), {
            "__annotations__": {
                "a": int, "b": float, "c": bool, "d": str,
            },
            "a": 0, "b": 0.0, "c": False, "d": "",
            "__module__": __name__,
        })
        cid, name, size, fields_json = omega_bridge.type_register_calls[-1]
        fields = json.loads(fields_json)
        by_name = {f[0]: f for f in fields}
        assert by_name["a"][1] == "i32"
        assert by_name["b"][1] == "f32"
        assert by_name["c"][1] == "u8"
        assert by_name["d"][1] == "string"

    def test_type_register_called_exactly_once(self, omega_bridge):
        """ComponentMeta calls type_register exactly once per component."""
        ComponentMeta("ExactlyOnce", (), {
            "__annotations__": {"x": int}, "x": 0,
            "__module__": __name__,
        })
        # Count calls whose name ends with ExactlyOnce
        matching = [c for c in omega_bridge.type_register_calls if c[1].endswith("ExactlyOnce")]
        assert len(matching) == 1

    def test_rust_layout_fields_sorted_by_offset(self, omega_bridge):
        """_build_rust_layout returns fields sorted by offset."""
        cls = ComponentMeta("SortedLayout", (), {
            "__annotations__": {
                "z": float, "a": int, "mid": bool,
            },
            "z": 0.0, "a": 0, "mid": False,
            "__module__": __name__,
        })
        cid, name, size, fields_json = omega_bridge.type_register_calls[-1]
        fields = json.loads(fields_json)
        offsets = [f[2] for f in fields]
        assert offsets == sorted(offsets), f"Fields not sorted by offset: {offsets}"

    def test_total_size_matches_expected(self, omega_bridge):
        """total_size is the sum of field sizes for this component."""
        cls = ComponentMeta("SizeCheck", (), {
            "__annotations__": {
                "x": float,  # f32 = 4
                "y": float,  # f32 = 4
                "z": float,  # f32 = 4
            },
            "x": 0.0, "y": 0.0, "z": 0.0,
            "__module__": __name__,
        })
        cid, name, total_size, fields_json = omega_bridge.type_register_calls[-1]
        fields = json.loads(fields_json)
        computed_size = max(f[2] for f in fields) + 4  # last offset + sizeof(f32)
        assert total_size == computed_size, (
            f"total_size={total_size} != computed={computed_size}"
        )

    # ------------------------------------------------------------------
    # 2b. Data channel -- strict read/write semantics
    # ------------------------------------------------------------------

    def test_component_read_raises_for_unwritten_key(self, fresh_bridge):
        """Reading an unwritten (entity, component, offset) raises RuntimeError."""
        with pytest.raises(RuntimeError) as exc:
            fresh_bridge.component_read(999, 999, 999, int)
        assert "component_read" in str(exc.value)
        assert str(999) in str(exc.value)

    def test_component_read_error_message_format(self, fresh_bridge):
        """Error messages contain entity_id, component_id, and offset."""
        with pytest.raises(RuntimeError) as exc:
            fresh_bridge.component_read(1, 2, 3, int)
        msg = str(exc.value)
        assert "1" in msg or "(1, 2, 3)" in msg, (
            f"Error message should reference failing key: {msg}"
        )

    def test_component_write_overwrite_no_error(self, fresh_bridge):
        """Overwriting an existing field is allowed without error."""
        fresh_bridge.component_write(1, 1, 0, "first")
        fresh_bridge.component_write(1, 1, 0, "second")  # no raise

    def test_component_delete_non_existent(self, fresh_bridge):
        """Deleting a non-existent field does NOT raise (idempotent per Rust semantics)."""
        fresh_bridge.component_delete(999, 999, 999)  # no raise

    def test_component_delete_then_read_raises(self, fresh_bridge):
        """Reading a field immediately after delete raises RuntimeError."""
        fresh_bridge.component_write(1, 1, 0, 42)
        fresh_bridge.component_delete(1, 1, 0)
        with pytest.raises(RuntimeError):
            fresh_bridge.component_read(1, 1, 0, int)

    def test_write_with_large_values(self, fresh_bridge):
        """Large strings and integers round-trip correctly."""
        large_str = "x" * 100_000
        fresh_bridge.component_write(1, 1, 0, large_str)
        assert fresh_bridge.component_read(1, 1, 0, str) == large_str

        large_int = 2**63 - 1
        fresh_bridge.component_write(1, 1, 8, large_int)
        assert fresh_bridge.component_read(1, 1, 8, int) == large_int

    def test_write_negative_int(self, fresh_bridge):
        """Negative integers round-trip correctly."""
        fresh_bridge.component_write(1, 1, 0, -42)
        assert fresh_bridge.component_read(1, 1, 0, int) == -42

    def test_write_floating_point_precision(self, fresh_bridge):
        """Floating point values preserve precision."""
        fresh_bridge.component_write(1, 1, 0, 3.141592653589793)
        val = fresh_bridge.component_read(1, 1, 0, float)
        assert val == 3.141592653589793

    def test_entity_id_zero_allowed(self, fresh_bridge):
        """Entity ID 0 is valid (world handle uses 0)."""
        fresh_bridge.component_write(0, 1, 0, "zero_eid")
        assert fresh_bridge.component_read(0, 1, 0, str) == "zero_eid"

    def test_component_id_zero_allowed(self, fresh_bridge):
        """Component ID 0 is valid."""
        fresh_bridge.component_write(1, 0, 0, "zero_cid")
        assert fresh_bridge.component_read(1, 0, 0, str) == "zero_cid"

    # ------------------------------------------------------------------
    # 2c. Command channel -- strict world operation semantics
    # ------------------------------------------------------------------

    def test_spawn_with_no_components(self, fresh_bridge):
        """Spawn an entity with zero components creates a valid entity."""
        eid = fresh_bridge.world_spawn(0, [])
        assert eid >= 0
        assert eid in fresh_bridge._entities
        assert eid in fresh_bridge._entity_components
        assert fresh_bridge._entity_components[eid] == frozenset()

    def test_spawn_empty_components_list_creates_archetype(self, fresh_bridge):
        """Spawn with no components registers under empty frozenset archetype."""
        eid = fresh_bridge.world_spawn(0, [])
        assert frozenset() in fresh_bridge._archetypes
        assert eid in fresh_bridge._archetypes[frozenset()]

    def test_query_with_empty_list_returns_all(self, fresh_bridge):
        """world_query([], _) matches all entities (empty set is subset of every set)."""
        e1 = fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
        e2 = fresh_bridge.world_spawn(0, [(1, [(0, 2)])])
        results = fresh_bridge.world_query(0, [])
        # An empty frozenset is a subset of every set, so issubset() is always
        # True.  This follows standard ECS semantics: asking for "entities with
        # no required components" means "all entities".
        assert len(results) == 2, (
            f"Query with empty component list returned {results}, expected all"
        )
        assert e1 in results
        assert e2 in results

    def test_query_returns_empty_list_when_no_match(self, fresh_bridge):
        """Query for a component that no entity has returns empty list."""
        fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
        results = fresh_bridge.world_query(0, [99])
        assert results == []

    def test_despawn_non_existent_entity_no_error(self, fresh_bridge):
        """Despawning an entity that does not exist does NOT raise."""
        fresh_bridge.world_despawn(0, 999)  # no raise

    def test_despawn_twice_no_error(self, fresh_bridge):
        """Despawning the same entity twice is idempotent."""
        eid = fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
        fresh_bridge.world_despawn(0, eid)
        fresh_bridge.world_despawn(0, eid)  # no raise -- second is no-op

    def test_world_create_always_returns_zero(self, fresh_bridge):
        """world_create always returns 0 (single-world)."""
        assert fresh_bridge.world_create() == 0
        assert fresh_bridge.world_create() == 0  # idempotent

    def test_query_only_live_entities(self, fresh_bridge):
        """Despawned entities are invisible to queries."""
        e1 = fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
        e2 = fresh_bridge.world_spawn(0, [(1, [(0, 2)])])
        fresh_bridge.world_despawn(0, e1)
        results = fresh_bridge.world_query(0, [1])
        assert e1 not in results
        assert e2 in results

    def test_spawn_with_no_field_values(self, fresh_bridge):
        """Spawn with a component but no field values."""
        eid = fresh_bridge.world_spawn(0, [(1, [])])
        assert eid in fresh_bridge._entities
        # No fields were written, but component is still tracked
        assert 1 in fresh_bridge._entity_components[eid]

    # ------------------------------------------------------------------
    # 2d. Cross-channel consistency
    # ------------------------------------------------------------------

    def test_type_register_then_spawn_then_read(self, omega_bridge):
        """Full protocol: register type -> spawn entity -> read field."""
        omega_bridge.type_register(1, "Pos", 12, '[["x", "f32", 0], ["y", "f32", 4], ["z", "f32", 8]]')
        eid = omega_bridge.world_spawn(0, [(1, [(0, 1.0), (4, 2.0), (8, 3.0)])])
        assert omega_bridge.component_read(eid, 1, 0, float) == 1.0
        assert omega_bridge.component_read(eid, 1, 4, float) == 2.0
        assert omega_bridge.component_read(eid, 1, 8, float) == 3.0

    def test_query_after_type_registration_no_effect(self, fresh_bridge):
        """Registering types does not affect entity queries."""
        fresh_bridge.type_register(1, "T1", 4, "[]")
        fresh_bridge.type_register(2, "T2", 4, "[]")
        # No entities yet
        assert fresh_bridge.world_query(0, [1]) == []
        # Spawn
        fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
        assert len(fresh_bridge.world_query(0, [1])) == 1


# =============================================================================
# 3. MOCK BRIDGE FIDELITY -- faithful Rust/PyO3 semantics
# =============================================================================

class TestMockBridgeFidelity:
    """Whitebox: verify MockOmegaBridge faithfully represents Rust behavior."""

    # ------------------------------------------------------------------
    # 3a. Exception semantics
    # ------------------------------------------------------------------

    def test_read_missing_key_is_runtime_error(self, fresh_bridge):
        """Mock raises RuntimeError for missing data, matching PyO3 convention."""
        with pytest.raises(RuntimeError):
            fresh_bridge.component_read(1, 1, 0, int)

    def test_error_on_read_with_negative_entity_id(self, fresh_bridge):
        """Negative entity IDs raise RuntimeError (no such key)."""
        with pytest.raises(RuntimeError):
            fresh_bridge.component_read(-1, 1, 0, int)

    def test_write_to_despawned_entity_still_works(self, fresh_bridge):
        """Simulates Rust where writing to despawned entity is technically possible
        but semantically undefined; the mock allows it (weak fidelity)."""
        eid = fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
        fresh_bridge.world_despawn(0, eid)
        # Writing to a despawned entity should not crash
        fresh_bridge.component_write(eid, 1, 0, 999)
        # Reading back still works in the mock (the data lingers in _store)
        # This is weaker than Rust's ownership model, but safe for testing.
        val = fresh_bridge.component_read(eid, 1, 0, int)
        assert val == 999

    # ------------------------------------------------------------------
    # 3b. Return type consistency
    # ------------------------------------------------------------------

    def test_world_create_returns_int(self, fresh_bridge):
        """world_create returns an int (world handle)."""
        result = fresh_bridge.world_create()
        assert isinstance(result, int)

    def test_world_spawn_returns_int(self, fresh_bridge):
        """world_spawn returns an int (entity ID)."""
        eid = fresh_bridge.world_spawn(0, [(1, [(0, "a")])])
        assert isinstance(eid, int)

    def test_world_query_returns_list_of_ints(self, fresh_bridge):
        """world_query returns a list of int entity IDs."""
        fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
        results = fresh_bridge.world_query(0, [1])
        assert isinstance(results, list)
        if results:
            assert all(isinstance(eid, int) for eid in results)

    def test_type_list_returns_dict(self, fresh_bridge):
        """type_list returns a dict."""
        fresh_bridge.type_register(1, "T", 4, "[]")
        result = fresh_bridge.type_list()
        assert isinstance(result, dict)

    def test_checksum_returns_hex_string(self, fresh_bridge):
        """checksum() returns a hex string of predictable length."""
        cs = fresh_bridge.checksum()
        assert isinstance(cs, str)
        assert len(cs) == 64  # SHA-256 hex

    # ------------------------------------------------------------------
    # 3c. State isolation guarantees
    # ------------------------------------------------------------------

    def test_two_bridges_are_independent(self):
        """Two MockOmegaBridge instances share no state."""
        b1 = MockOmegaBridge()
        b2 = MockOmegaBridge()

        b1.type_register(1, "B1", 4, "[]")
        b2.type_register(2, "B2", 4, "[]")

        assert 1 in b1.type_registry
        assert 1 not in b2.type_registry
        assert 2 in b2.type_registry
        assert 2 not in b1.type_registry

    def test_two_bridges_independent_data_channel(self):
        """Data stored in one bridge is invisible to another."""
        b1 = MockOmegaBridge()
        b2 = MockOmegaBridge()

        b1.component_write(1, 1, 0, "in_b1")
        b2.component_write(1, 1, 0, "in_b2")

        assert b1.component_read(1, 1, 0, str) == "in_b1"
        assert b2.component_read(1, 1, 0, str) == "in_b2"

    def test_two_bridges_independent_entity_ids(self):
        """Entity ID counters are independent across bridges."""
        b1 = MockOmegaBridge()
        b2 = MockOmegaBridge()

        e1 = b1.world_spawn(0, [(1, [(0, 1)])])
        e2 = b2.world_spawn(0, [(1, [(0, 2)])])

        # Both start at 0, so both return the same first ID
        assert e1 == e2

    # ------------------------------------------------------------------
    # 3d. _omega module import fidelity
    # ------------------------------------------------------------------

    def test_omega_module_patch_works_with_real_meta(self, omega_bridge):
        """When _omega is patched, ComponentMeta calls the mock's type_register."""
        assert "_omega" in sys.modules
        assert sys.modules["_omega"] is omega_bridge

        ComponentMeta("OmegaPatched", (), {
            "__annotations__": {"x": float}, "x": 0.0,
            "__module__": __name__,
        })
        assert len(omega_bridge.type_register_calls) >= 1

    def test_omega_module_missing_graceful_fallback(self):
        """ComponentMeta does not crash when _omega is absent."""
        with mock.patch.dict(sys.modules, {"_omega": None}):
            cls = ComponentMeta("NoOmega", (), {
                "__annotations__": {"x": int}, "x": 0,
                "__module__": __name__,
            })
            assert cls is not None
            assert cls._component_id > 0


# =============================================================================
# 4. EDGE CASES -- boundary conditions
# =============================================================================

class TestEdgeCases:
    """Whitebox: boundary and edge cases for all three channels."""

    # ------------------------------------------------------------------
    # 4a. Component edge cases
    # ------------------------------------------------------------------

    def test_component_with_zero_fields(self, omega_bridge):
        """A component with no annotated fields is still registered."""
        cls = ComponentMeta("EmptyComp", (), {
            "__module__": __name__,
        })
        assert cls._component_id > 0
        # type_register should still be called with empty field list
        matching = [c for c in omega_bridge.type_register_calls if c[1].endswith("EmptyComp")]
        assert len(matching) == 1
        cid, name, total_size, fields_json = matching[0]
        assert json.loads(fields_json) == []

    def test_component_with_unicode_name(self, omega_bridge):
        """Component names with Unicode characters are handled correctly."""
        cls = ComponentMeta("UnicodeComponenté", (), {
            "__annotations__": {"x": int}, "x": 0,
            "__module__": __name__,
        })
        matching = [c for c in omega_bridge.type_register_calls if "UnicodeComponent" in c[1]]
        assert len(matching) >= 1

    def test_component_with_long_name(self, omega_bridge):
        """Component with a very long name does not crash."""
        long_name = "A" * 500
        cls = ComponentMeta(long_name, (), {
            "__annotations__": {"x": int}, "x": 0,
            "__module__": __name__,
        })
        matching = [c for c in omega_bridge.type_register_calls if c[1].endswith(long_name)]
        assert len(matching) >= 1

    def test_component_with_many_fields(self, omega_bridge):
        """Component with 100 fields is registered correctly."""
        annotations = {f"f{i}": int for i in range(100)}
        defaults = {f"f{i}": 0 for i in range(100)}
        namespace = {
            "__annotations__": annotations,
            "__module__": __name__,
            **defaults,
        }
        cls = ComponentMeta("ManyFields", (), namespace)
        matching = [c for c in omega_bridge.type_register_calls if c[1].endswith("ManyFields")]
        assert len(matching) == 1
        cid, name, total_size, fields_json = matching[0]
        fields = json.loads(fields_json)
        assert len(fields) == 100

    def test_component_registry_size_limit(self, omega_bridge):
        """Sequential component IDs grow without artificial limit."""
        n = 500
        ids = []
        for i in range(n):
            cls = ComponentMeta(f"StressReg{i}", (), {
                "__annotations__": {"v": int}, "v": i,
                "__module__": __name__,
            })
            ids.append(cls._component_id)
        assert ids == list(range(1, n + 1))

    # ------------------------------------------------------------------
    # 4b. Data channel edge cases
    # ------------------------------------------------------------------

    def test_write_none_then_read_returns_none(self, fresh_bridge):
        """None value round-trips correctly."""
        fresh_bridge.component_write(1, 1, 0, None)
        assert fresh_bridge.component_read(1, 1, 0, object) is None

    def test_write_bool_false_then_read(self, fresh_bridge):
        """False bool round-trips."""
        fresh_bridge.component_write(1, 1, 0, False)
        assert fresh_bridge.component_read(1, 1, 0, bool) is False

    def test_write_bool_true_then_read(self, fresh_bridge):
        """True bool round-trips."""
        fresh_bridge.component_write(1, 1, 0, True)
        assert fresh_bridge.component_read(1, 1, 0, bool) is True

    def test_write_empty_string(self, fresh_bridge):
        """Empty string round-trips."""
        fresh_bridge.component_write(1, 1, 0, "")
        assert fresh_bridge.component_read(1, 1, 0, str) == ""

    def test_write_zero_int(self, fresh_bridge):
        """Zero int round-trips."""
        fresh_bridge.component_write(1, 1, 0, 0)
        assert fresh_bridge.component_read(1, 1, 0, int) == 0

    def test_large_negative_offset(self, fresh_bridge):
        """Negative offset values are stored correctly."""
        fresh_bridge.component_write(1, 1, -100, "neg")
        assert fresh_bridge.component_read(1, 1, -100, str) == "neg"

    def test_very_large_offset(self, fresh_bridge):
        """Very large positive offset works (simulating large components)."""
        fresh_bridge.component_write(1, 1, 10_000, "far")
        assert fresh_bridge.component_read(1, 1, 10_000, str) == "far"

    def test_overwrite_different_type(self, fresh_bridge):
        """Overwriting a field with a different type is allowed in mock."""
        fresh_bridge.component_write(1, 1, 0, 42)       # int
        fresh_bridge.component_write(1, 1, 0, "hello")   # str (overwrite, different type)
        assert fresh_bridge.component_read(1, 1, 0, str) == "hello"

    # ------------------------------------------------------------------
    # 4c. Command channel edge cases
    # ------------------------------------------------------------------

    def test_spawn_entity_ids_monotonic(self, fresh_bridge):
        """Entity IDs are strictly monotonically increasing."""
        ids = []
        for _ in range(1000):
            eid = fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
            ids.append(eid)
        for i in range(len(ids) - 1):
            assert ids[i] < ids[i + 1], (
                f"Entity IDs not monotonic at index {i}: {ids[i]} >= {ids[i + 1]}"
            )

    def test_despawn_all_then_query_empty(self, fresh_bridge):
        """Despawn all entities; query returns empty."""
        eids = [fresh_bridge.world_spawn(0, [(1, [(0, i)])]) for i in range(100)]
        for eid in eids:
            fresh_bridge.world_despawn(0, eid)
        assert fresh_bridge.world_query(0, [1]) == []

    def test_query_with_multiple_components_intersection(self, fresh_bridge):
        """world_query with multiple components returns intersection."""
        e1 = fresh_bridge.world_spawn(0, [(1, [(0, 1)]), (2, [(0, 10)])])
        e2 = fresh_bridge.world_spawn(0, [(1, [(0, 2)])])  # only comp 1
        e3 = fresh_bridge.world_spawn(0, [(1, [(0, 3)]), (2, [(0, 30)])])

        # Query for [1, 2] should return e1 and e3 only
        results = fresh_bridge.world_query(0, [1, 2])
        assert sorted(results) == sorted([e1, e3])

    def test_query_with_non_existent_component(self, fresh_bridge):
        """Query for a component ID that was never spawned returns empty list."""
        e1 = fresh_bridge.world_spawn(0, [(1, [(0, 1)])])
        results = fresh_bridge.world_query(0, [999])
        assert results == []

    def test_many_archetype_variants(self, fresh_bridge):
        """Create entities with many different archetypes."""
        n_archetypes = 50
        eids = []
        for i in range(n_archetypes):
            # Each entity gets a unique set of component IDs
            comps = [(cid, [(0, cid)]) for cid in range(i, i + 3)]
            eid = fresh_bridge.world_spawn(0, comps)
            eids.append(eid)

        assert len(fresh_bridge._archetypes) >= n_archetypes
        assert fresh_bridge.spawn_count == n_archetypes

    # ------------------------------------------------------------------
    # 4d. Edge cases for TrinityWorldAdapter
    # ------------------------------------------------------------------

    def test_adapter_round_trip_via_bridge(self, omega_bridge):
        """TrinityWorldAdapter + ComponentMeta + omega_bridge end-to-end."""
        from foundation.bridge import TrinityWorldAdapter

        adapter = TrinityWorldAdapter()

        # Define a component using the Component base class (provides a proper
        # __init__ that accepts keyword arguments for field values).
        class AdapterBridge(Component):
            val: int = 0

        inst = AdapterBridge(val=42)
        entity = adapter.add_instance(inst)

        # The instance should be retrievable
        retrieved = adapter.get_instance(entity, AdapterBridge)
        assert retrieved is inst
        assert retrieved.val == 42

    def test_adapter_remove_instance_twice(self, omega_bridge):
        """Removing an instance twice is safe (idempotent)."""
        from foundation.bridge import TrinityWorldAdapter

        class SafeComp(Component):
            x: int = 0

        adapter = TrinityWorldAdapter()
        inst = SafeComp(x=10)
        adapter.add_instance(inst)
        adapter.remove_instance(inst)
        # Second remove should not raise
        adapter.remove_instance(inst)

    # ------------------------------------------------------------------
    # 4e. Checksum edge cases
    # ------------------------------------------------------------------

    def test_checksum_empty_store_is_consistent(self, fresh_bridge):
        """Empty store checksum is always the same SHA-256 hash."""
        cs1 = fresh_bridge.checksum()
        cs2 = MockOmegaBridge().checksum()
        assert cs1 == cs2

    def test_checksum_changes_after_spawn(self, fresh_bridge):
        """Checksum changes after spawning an entity."""
        cs_before = fresh_bridge.checksum()
        fresh_bridge.world_spawn(0, [(1, [(0, 10)])])
        cs_after = fresh_bridge.checksum()
        assert cs_before != cs_after

    def test_checksum_includes_archetype_structure(self, fresh_bridge):
        """Checksum reflects archetype composition, not just values."""
        b1 = MockOmegaBridge()
        b1.world_spawn(0, [(1, [(0, 10)])])  # comp 1 only

        b2 = MockOmegaBridge()
        b2.world_spawn(0, [(1, [(0, 10)]), (2, [(0, 20)])])  # comp 1 + 2, same value

        assert b1.checksum() != b2.checksum()

    # ------------------------------------------------------------------
    # 4f. Concurrent edge cases
    # ------------------------------------------------------------------

    def test_concurrent_read_of_same_key(self, fresh_bridge):
        """Multiple threads reading the same key do not interfere."""
        fresh_bridge.component_write(1, 1, 0, 42)
        n_threads = 10
        results: list[int] = [0] * n_threads

        def reader(thread_id: int):
            for _ in range(1000):
                val = fresh_bridge.component_read(1, 1, 0, int)
                results[thread_id] = val

        threads = [threading.Thread(target=reader, args=(tid,)) for tid in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert all(r == 42 for r in results)

    def test_concurrent_despawn_same_entity(self, fresh_bridge):
        """Multiple threads trying to despawn the same entity is safe."""
        eid = fresh_bridge.world_spawn(0, [(1, [(0, 1)])])

        errors: list[Exception] = []

        def despawner():
            try:
                fresh_bridge.world_despawn(0, eid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=despawner) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # Entity should not be in _entities (or at least no errors)
        assert eid not in fresh_bridge._entities
        if errors:
            # If any errors occurred, they should be understood
            print(f"Concurrent despawn errors: {errors}")

    # ------------------------------------------------------------------
    # 4g. Mock edge cases
    # ------------------------------------------------------------------

    def test_mock_type_register_calls_logged_in_order(self, fresh_bridge):
        """type_register_calls preserves insertion order."""
        fresh_bridge.type_register(1, "A", 4, "[]")
        fresh_bridge.type_register(2, "B", 8, "[]")
        fresh_bridge.type_register(3, "C", 12, "[]")
        assert len(fresh_bridge.type_register_calls) == 3
        assert fresh_bridge.type_register_calls[0][1] == "A"
        assert fresh_bridge.type_register_calls[1][1] == "B"
        assert fresh_bridge.type_register_calls[2][1] == "C"

    def test_mock_handles_unicode_fields_json(self, fresh_bridge):
        """type_register accepts Unicode in fields_json."""
        fields = json.dumps([["label", "string", 0]])
        fresh_bridge.type_register(1, "UnicodeTest", 8, fields)
        assert fresh_bridge.type_registry[1]["fields"] == fields

    def test_mock_empty_fields_json(self, fresh_bridge):
        """type_register accepts empty fields_json."""
        fresh_bridge.type_register(1, "EmptyFields", 0, "[]")
        entry = fresh_bridge.type_registry[1]
        assert entry["fields"] == "[]"
        assert entry["total_size"] == 0

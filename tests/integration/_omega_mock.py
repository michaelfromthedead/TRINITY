"""
Mock Omega Bridge -- shared test utility.

In-process replacement for the compiled _omega PyO3 module.
Provides a deterministic in-memory backend that simulates the
Rust-side component store for integration testing.

Both DEV (test_bridge_protocol.py) and WHITEBOX
(test_bridge_protocol_whitebox.py) test suites import from here.
"""

from __future__ import annotations

import hashlib
from typing import Any


class MockOmegaBridge:
    """In-process replacement for the compiled _omega PyO3 module.

    Simulates the Rust-side component store with a deterministic in-memory
    backend so that integration tests can verify the bridge protocol without
    a compiled PyO3 extension.
    """

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state -- call between tests for isolation."""
        # Type channel
        self.type_registry: dict[int, dict] = {}
        self.type_register_calls: list[tuple] = []

        # Data channel -- maps (entity_id, component_id, offset) -> raw bytes
        self._store: dict[tuple[int, int, int], object] = {}

        # Command channel -- entity lifecycle
        self._entities: set[int] = set()
        self._entity_counter: int = 0
        self._archetypes: dict[frozenset, list[int]] = {}
        self._entity_components: dict[int, frozenset] = {}

        # Stats
        self.read_count: int = 0
        self.write_count: int = 0
        self.spawn_count: int = 0
        self.despawn_count: int = 0
        self.query_count: int = 0

    # ------------------------------------------------------------------
    # Type channel (called from ComponentMeta.__new__ step 6b)
    # ------------------------------------------------------------------

    def type_register(self, component_id: int, name: str, total_size: int,
                      fields_json: str) -> None:
        """Record a type registration."""
        self.type_registry[component_id] = {
            "name": name,
            "total_size": total_size,
            "fields": fields_json,
        }
        self.type_register_calls.append((component_id, name, total_size, fields_json))

    def type_list(self) -> dict[int, dict]:
        """Return all registered types (debug/inspection)."""
        return dict(self.type_registry)

    # ------------------------------------------------------------------
    # Data channel (called from RustStorageDescriptor)
    # ------------------------------------------------------------------

    def component_read(self, entity_id: int, component_id: int,
                       offset: int, field_type: type) -> object:
        """Read a field value from the simulated Rust store."""
        self.read_count += 1
        key = (entity_id, component_id, offset)
        try:
            return self._store[key]
        except KeyError:
            msg = f"component_read: no value at ({entity_id}, {component_id}, {offset})"
            raise RuntimeError(msg) from None

    def component_write(self, entity_id: int, component_id: int,
                        offset: int, value: object) -> None:
        """Write a field value to the simulated Rust store."""
        self.write_count += 1
        self._store[(entity_id, component_id, offset)] = value

    def component_delete(self, entity_id: int, component_id: int,
                         offset: int) -> None:
        """Delete a field value from the simulated Rust store."""
        self._store.pop((entity_id, component_id, offset), None)

    # ------------------------------------------------------------------
    # Command channel (called from World)
    # ------------------------------------------------------------------

    def world_create(self) -> int:
        """Create a new world handle (always 0 for single-world)."""
        return 0

    def world_spawn(self, world_handle: int,
                    components: list[tuple[int, list[tuple[int, object]]]]) -> int:
        """Spawn an entity with components.

        Parameters
        ----------
        components : list of (component_id, [(offset, value), ...])
        """
        self.spawn_count += 1
        entity_id = self._entity_counter
        self._entity_counter += 1
        self._entities.add(entity_id)

        comp_ids: set[int] = set()
        for cid, fields in components:
            comp_ids.add(cid)
            for offset, value in fields:
                self._store[(entity_id, cid, offset)] = value

        comp_set = frozenset(comp_ids)
        self._entity_components[entity_id] = comp_set
        self._archetypes.setdefault(comp_set, []).append(entity_id)
        return entity_id

    def world_despawn(self, world_handle: int, entity_id: int) -> None:
        """Despawn an entity."""
        self.despawn_count += 1
        self._entities.discard(entity_id)
        keys_to_delete = [k for k in self._store if k[0] == entity_id]
        for k in keys_to_delete:
            del self._store[k]
        comp_set = self._entity_components.pop(entity_id, frozenset())
        archetype = self._archetypes.get(comp_set)
        if archetype is not None and entity_id in archetype:
            archetype.remove(entity_id)

    def world_query(self, world_handle: int,
                    component_ids: list[int]) -> list[int]:
        """Query entities matching ALL given component types."""
        self.query_count += 1
        required = frozenset(component_ids)
        result = []
        for eid, comp_set in self._entity_components.items():
            if eid in self._entities and required.issubset(comp_set):
                result.append(eid)
        return result

    # ------------------------------------------------------------------
    # Helpers for determinism verification
    # ------------------------------------------------------------------

    def checksum(self) -> str:
        """Return a content-addressed hash of the store.

        The hash depends only on field values and archetype composition,
        not on entity ID assignment order or field write order.
        """
        h = hashlib.sha256()
        for comp_set in sorted(self._archetypes, key=sorted):
            entity_value_sets = []
            for eid in self._archetypes[comp_set]:
                fields = tuple(
                    sorted(
                        (cid, offset, val)
                        for (seid, cid, offset), val in self._store.items()
                        if seid == eid
                    ),
                )
                entity_value_sets.append(fields)
            h.update(
                f"archetype={sorted(comp_set)}:"
                f"entities={sorted(entity_value_sets)}".encode()
            )
        return h.hexdigest()

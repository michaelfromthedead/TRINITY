"""Tests for EntityRef and reference handling (T-CC-2.6).

Comprehensive test suite covering:
- EntityRef[T] type for referencing serializable entities
- ReferenceResolver for ID to object mapping
- ReferenceRegistry for tracking referenceable objects
- Cycle detection and prevention
- Forward reference resolution
- Integration with SerializationContext
"""
import gc
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import pytest

from engine.core.serialization import (
    SerializationContext,
    SerializationError,
    serializable,
)
from engine.core.entity_refs import (
    DeserializationContext,
    EntityRef,
    RefInfo,
    RefState,
    Referenceable,
    ReferenceError,
    ReferenceRegistry,
    ReferenceResolver,
    deserialize_with_refs,
    get_global_registry,
    register_global,
    resolve_global,
    serialize_with_refs,
)


# =============================================================================
# Test Fixtures and Helper Classes
# =============================================================================


class SimpleEntity(Referenceable):
    """Simple referenceable entity for testing."""

    def __init__(self, name: str, ref_id: Optional[str] = None):
        self.name = name
        self._ref_id = ref_id or f"simple_{id(self):x}"

    def get_ref_id(self) -> str:
        return self._ref_id

    def set_ref_id(self, ref_id: str) -> None:
        self._ref_id = ref_id


@serializable()
@dataclass
class Player(Referenceable):
    """Player entity with team reference."""
    name: str
    score: int = 0
    _ref_id: str = field(default="", repr=False)

    def get_ref_id(self) -> str:
        if not self._ref_id:
            self._ref_id = f"player_{id(self):x}"
        return self._ref_id

    def set_ref_id(self, ref_id: str) -> None:
        self._ref_id = ref_id


@serializable()
@dataclass
class Team(Referenceable):
    """Team entity with player references."""
    name: str
    _ref_id: str = field(default="", repr=False)

    def get_ref_id(self) -> str:
        if not self._ref_id:
            self._ref_id = f"team_{id(self):x}"
        return self._ref_id

    def set_ref_id(self, ref_id: str) -> None:
        self._ref_id = ref_id


# =============================================================================
# Tests for ReferenceError
# =============================================================================


class TestReferenceError:
    """Tests for ReferenceError exception."""

    def test_basic_error(self):
        err = ReferenceError("Test error")
        assert str(err) == "Test error"
        assert err.ref_id is None

    def test_error_with_ref_id(self):
        err = ReferenceError("Not found", ref_id="ref_123")
        assert err.ref_id == "ref_123"

    def test_error_with_path(self):
        err = ReferenceError("Invalid", path="root.field")
        assert str(err) == "root.field: Invalid"

    def test_cycle_error(self):
        err = ReferenceError("Cycle detected", is_cycle=True)
        assert err.is_cycle
        assert "[CYCLE]" in str(err)

    def test_missing_error(self):
        err = ReferenceError("Not found", is_missing=True)
        assert err.is_missing
        assert "[MISSING]" in str(err)

    def test_forward_error(self):
        err = ReferenceError("Forward ref", is_forward=True)
        assert err.is_forward
        assert "[FORWARD]" in str(err)


# =============================================================================
# Tests for RefState
# =============================================================================


class TestRefState:
    """Tests for RefState enum."""

    def test_all_states_exist(self):
        assert RefState.UNRESOLVED
        assert RefState.RESOLVED
        assert RefState.DEFERRED
        assert RefState.BROKEN
        assert RefState.CYCLE

    def test_states_are_distinct(self):
        states = [RefState.UNRESOLVED, RefState.RESOLVED, RefState.DEFERRED,
                  RefState.BROKEN, RefState.CYCLE]
        assert len(set(states)) == 5


# =============================================================================
# Tests for Referenceable
# =============================================================================


class TestReferenceable:
    """Tests for Referenceable mixin."""

    def test_default_get_ref_id(self):
        class BasicRef(Referenceable):
            pass

        obj = BasicRef()
        ref_id = obj.get_ref_id()
        assert ref_id.startswith("ref_")
        assert isinstance(ref_id, str)

    def test_custom_get_ref_id(self):
        entity = SimpleEntity("test", ref_id="custom_123")
        assert entity.get_ref_id() == "custom_123"

    def test_set_ref_id(self):
        entity = SimpleEntity("test")
        entity.set_ref_id("new_id")
        assert entity.get_ref_id() == "new_id"

    def test_ref_id_uniqueness(self):
        e1 = SimpleEntity("one")
        e2 = SimpleEntity("two")
        assert e1.get_ref_id() != e2.get_ref_id()


# =============================================================================
# Tests for EntityRef
# =============================================================================


class TestEntityRef:
    """Tests for EntityRef[T] generic reference type."""

    def test_create_with_target(self):
        entity = SimpleEntity("test")
        ref = EntityRef(target=entity)
        assert ref.is_resolved
        assert ref.get() is entity

    def test_create_with_ref_id(self):
        ref: EntityRef[SimpleEntity] = EntityRef(ref_id="entity_1")
        assert not ref.is_resolved
        assert ref.state == RefState.UNRESOLVED
        assert ref.ref_id == "entity_1"

    def test_null_reference(self):
        ref: EntityRef[SimpleEntity] = EntityRef.null()
        assert ref.is_null
        assert ref.get() is None
        assert ref.state == RefState.BROKEN

    def test_from_id_factory(self):
        ref = EntityRef.from_id("test_id", SimpleEntity)
        assert ref.ref_id == "test_id"
        assert ref.state == RefState.UNRESOLVED

    def test_get_or_raise_resolved(self):
        entity = SimpleEntity("test")
        ref = EntityRef(target=entity)
        assert ref.get_or_raise() is entity

    def test_get_or_raise_unresolved(self):
        ref: EntityRef[SimpleEntity] = EntityRef(ref_id="missing")
        with pytest.raises(ReferenceError) as exc:
            ref.get_or_raise()
        assert exc.value.is_missing

    def test_is_valid_resolved(self):
        entity = SimpleEntity("test")
        ref = EntityRef(target=entity)
        assert ref.is_valid

    def test_is_valid_deferred(self):
        ref: EntityRef[SimpleEntity] = EntityRef(ref_id="entity_1")
        ref._mark_deferred()
        assert ref.is_valid

    def test_is_valid_broken(self):
        ref: EntityRef[SimpleEntity] = EntityRef.null()
        assert not ref.is_valid

    def test_equality(self):
        ref1: EntityRef[SimpleEntity] = EntityRef(ref_id="same_id")
        ref2: EntityRef[SimpleEntity] = EntityRef(ref_id="same_id")
        ref3: EntityRef[SimpleEntity] = EntityRef(ref_id="diff_id")

        assert ref1 == ref2
        assert ref1 != ref3

    def test_hash(self):
        ref1: EntityRef[SimpleEntity] = EntityRef(ref_id="id_1")
        ref2: EntityRef[SimpleEntity] = EntityRef(ref_id="id_1")

        assert hash(ref1) == hash(ref2)
        assert ref1 in {ref2}

    def test_repr_resolved(self):
        entity = SimpleEntity("test", ref_id="test_123")
        ref = EntityRef(target=entity, target_type=SimpleEntity)
        repr_str = repr(ref)
        assert "SimpleEntity" in repr_str
        assert "RESOLVED" in repr_str

    def test_repr_null(self):
        ref: EntityRef[SimpleEntity] = EntityRef.null()
        assert "null" in repr(ref)

    def test_serialize(self):
        entity = SimpleEntity("test", ref_id="entity_abc")
        ref = EntityRef(target=entity, target_type=SimpleEntity)
        data = ref.serialize()

        assert data["__ref__"] == "entity_abc"
        assert data["__ref_type__"] == "SimpleEntity"

    def test_deserialize(self):
        data = {"__ref__": "entity_xyz", "__ref_type__": "SimpleEntity"}
        ref = EntityRef.deserialize(data, target_type=SimpleEntity)

        assert ref.ref_id == "entity_xyz"
        assert ref.state == RefState.UNRESOLVED

    def test_deserialize_invalid_format(self):
        with pytest.raises(ReferenceError):
            EntityRef.deserialize({"invalid": "data"})

    def test_resolve_manually(self):
        ref: EntityRef[SimpleEntity] = EntityRef(ref_id="manual_1")
        entity = SimpleEntity("target", ref_id="manual_1")

        ref._resolve(entity)
        assert ref.is_resolved
        assert ref.get() is entity


# =============================================================================
# Tests for ReferenceRegistry
# =============================================================================


class TestReferenceRegistry:
    """Tests for ReferenceRegistry."""

    def test_register_object(self):
        registry = ReferenceRegistry(use_weak_refs=False)
        entity = SimpleEntity("test")

        ref_id = registry.register(entity)
        assert ref_id == entity.get_ref_id()
        assert registry.contains(ref_id)

    def test_register_with_explicit_id(self):
        registry = ReferenceRegistry(use_weak_refs=False)
        entity = SimpleEntity("test")

        ref_id = registry.register(entity, ref_id="explicit_id")
        assert ref_id == "explicit_id"
        assert registry.get(ref_id) is entity

    def test_register_already_registered(self):
        registry = ReferenceRegistry(use_weak_refs=False)
        entity = SimpleEntity("test")

        id1 = registry.register(entity)
        id2 = registry.register(entity)  # Same object
        assert id1 == id2

    def test_get_existing(self):
        registry = ReferenceRegistry(use_weak_refs=False)
        entity = SimpleEntity("test")
        ref_id = registry.register(entity)

        retrieved = registry.get(ref_id)
        assert retrieved is entity

    def test_get_nonexistent(self):
        registry = ReferenceRegistry()
        assert registry.get("nonexistent") is None

    def test_get_ref_id(self):
        registry = ReferenceRegistry(use_weak_refs=False)
        entity = SimpleEntity("test")
        registry.register(entity, ref_id="known_id")

        assert registry.get_ref_id(entity) == "known_id"

    def test_contains(self):
        registry = ReferenceRegistry(use_weak_refs=False)
        entity = SimpleEntity("test")
        ref_id = registry.register(entity)

        assert registry.contains(ref_id)
        assert not registry.contains("unknown")

    def test_contains_object(self):
        registry = ReferenceRegistry(use_weak_refs=False)
        entity = SimpleEntity("test")
        registry.register(entity)

        assert registry.contains_object(entity)
        assert not registry.contains_object(SimpleEntity("other"))

    def test_unregister(self):
        registry = ReferenceRegistry(use_weak_refs=False)
        entity = SimpleEntity("test")
        ref_id = registry.register(entity)

        assert registry.unregister(entity)
        assert not registry.contains(ref_id)

    def test_unregister_by_id(self):
        registry = ReferenceRegistry(use_weak_refs=False)
        entity = SimpleEntity("test")
        ref_id = registry.register(entity)

        assert registry.unregister_by_id(ref_id)
        assert not registry.contains(ref_id)

    def test_unregister_nonexistent(self):
        registry = ReferenceRegistry()
        assert not registry.unregister(SimpleEntity("new"))
        assert not registry.unregister_by_id("unknown")

    def test_clear(self):
        registry = ReferenceRegistry(use_weak_refs=False)
        for i in range(5):
            registry.register(SimpleEntity(f"test_{i}"))

        assert len(registry) == 5
        registry.clear()
        assert len(registry) == 0

    def test_len(self):
        registry = ReferenceRegistry(use_weak_refs=False)
        assert len(registry) == 0

        for i in range(3):
            registry.register(SimpleEntity(f"test_{i}"))
        assert len(registry) == 3

    def test_iter(self):
        registry = ReferenceRegistry(use_weak_refs=False)
        entities = [SimpleEntity(f"test_{i}", ref_id=f"id_{i}") for i in range(3)]
        for e in entities:
            registry.register(e)

        ref_ids = list(registry)
        assert len(ref_ids) == 3
        assert "id_0" in ref_ids

    def test_items(self):
        registry = ReferenceRegistry(use_weak_refs=False)
        entities = [SimpleEntity(f"test_{i}", ref_id=f"id_{i}") for i in range(3)]
        for e in entities:
            registry.register(e)

        items = registry.items()
        assert len(items) == 3
        ref_ids = [ref_id for ref_id, _ in items]
        assert "id_1" in ref_ids

    def test_weak_reference_cleanup(self):
        registry = ReferenceRegistry(use_weak_refs=True)

        # Create and register entity
        entity = SimpleEntity("temp")
        ref_id = registry.register(entity)
        assert registry.contains(ref_id)

        # Delete and GC
        del entity
        gc.collect()

        # Weak ref should be cleaned up
        assert registry.get(ref_id) is None

    def test_thread_safety(self):
        registry = ReferenceRegistry(use_weak_refs=False)
        results = []

        def register_entities(prefix: str, count: int):
            for i in range(count):
                entity = SimpleEntity(f"{prefix}_{i}", ref_id=f"{prefix}_{i}")
                registry.register(entity)
                results.append(registry.contains(f"{prefix}_{i}"))

        threads = [
            threading.Thread(target=register_entities, args=(f"t{i}", 10))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)
        assert len(registry) == 50


# =============================================================================
# Tests for ReferenceResolver
# =============================================================================


class TestReferenceResolver:
    """Tests for ReferenceResolver."""

    def test_register_and_resolve(self):
        resolver = ReferenceResolver()
        entity = SimpleEntity("test")

        ref_id = resolver.register(entity)
        resolved = resolver.resolve(ref_id)
        assert resolved is entity

    def test_resolve_nonexistent(self):
        resolver = ReferenceResolver()
        assert resolver.resolve("missing") is None

    def test_resolve_ref_immediate(self):
        resolver = ReferenceResolver()
        entity = SimpleEntity("test", ref_id="known")
        resolver.register(entity)

        ref: EntityRef[SimpleEntity] = EntityRef(ref_id="known")
        resolved = resolver.resolve_ref(ref)

        assert resolved is entity
        assert ref.is_resolved

    def test_resolve_ref_deferred(self):
        resolver = ReferenceResolver()

        ref: EntityRef[SimpleEntity] = EntityRef(ref_id="future")
        resolved = resolver.resolve_ref(ref)

        assert resolved is None
        assert ref.state == RefState.DEFERRED
        assert resolver.get_pending_count() == 1

    def test_deferred_resolution_on_register(self):
        resolver = ReferenceResolver()

        # Create ref before entity exists
        ref: EntityRef[SimpleEntity] = EntityRef(ref_id="deferred_1")
        resolver.resolve_ref(ref)
        assert ref.state == RefState.DEFERRED

        # Register entity - should resolve pending ref
        entity = SimpleEntity("target", ref_id="deferred_1")
        resolver.register(entity, ref_id="deferred_1")

        assert ref.is_resolved
        assert ref.get() is entity
        assert resolver.get_pending_count() == 0

    def test_create_ref(self):
        resolver = ReferenceResolver()
        entity = SimpleEntity("test")

        ref = resolver.create_ref(entity, SimpleEntity)
        assert ref.is_resolved
        assert ref.get() is entity

    def test_finalize_success(self):
        resolver = ReferenceResolver()
        entity = SimpleEntity("test")
        resolver.register(entity)

        errors = resolver.finalize()
        assert len(errors) == 0

    def test_finalize_broken_refs(self):
        resolver = ReferenceResolver(allow_broken=False)

        ref: EntityRef[SimpleEntity] = EntityRef(ref_id="missing")
        resolver.resolve_ref(ref)

        errors = resolver.finalize()
        assert len(errors) == 1
        assert errors[0].is_missing
        assert ref.state == RefState.BROKEN

    def test_finalize_allow_broken(self):
        resolver = ReferenceResolver(allow_broken=True)

        ref: EntityRef[SimpleEntity] = EntityRef(ref_id="missing")
        resolver.resolve_ref(ref)

        errors = resolver.finalize()
        assert len(errors) == 0  # No error when allow_broken=True

    def test_detect_cycle_simple(self):
        resolver = ReferenceResolver()

        # Create entities that reference each other
        @dataclass
        class Node(Referenceable):
            name: str
            next_ref: Optional[EntityRef["Node"]] = None
            _ref_id: str = ""

            def get_ref_id(self) -> str:
                if not self._ref_id:
                    self._ref_id = f"node_{id(self):x}"
                return self._ref_id

        node_a = Node(name="A", _ref_id="node_a")
        node_b = Node(name="B", _ref_id="node_b")

        # A -> B -> A (cycle)
        node_a.next_ref = EntityRef(target=node_b)
        node_b.next_ref = EntityRef(target=node_a)

        resolver.register(node_a, "node_a")
        resolver.register(node_b, "node_b")

        cycle = resolver.detect_cycle("node_a")
        assert cycle is not None
        assert "node_a" in cycle

    def test_detect_no_cycle(self):
        resolver = ReferenceResolver()

        @dataclass
        class Node(Referenceable):
            name: str
            next_ref: Optional[EntityRef["Node"]] = None
            _ref_id: str = ""

            def get_ref_id(self) -> str:
                return self._ref_id or f"node_{id(self):x}"

        node_a = Node(name="A", _ref_id="node_a")
        node_b = Node(name="B", _ref_id="node_b")
        node_c = Node(name="C", _ref_id="node_c")

        # A -> B -> C (no cycle)
        node_a.next_ref = EntityRef(target=node_b)
        node_b.next_ref = EntityRef(target=node_c)

        resolver.register(node_a, "node_a")
        resolver.register(node_b, "node_b")
        resolver.register(node_c, "node_c")

        cycle = resolver.detect_cycle("node_a")
        assert cycle is None

    def test_check_all_cycles(self):
        resolver = ReferenceResolver()

        @dataclass
        class Node(Referenceable):
            name: str
            next_ref: Optional[EntityRef["Node"]] = None
            _ref_id: str = ""

            def get_ref_id(self) -> str:
                return self._ref_id

        # Create a cycle: A -> B -> A
        node_a = Node(name="A", _ref_id="a")
        node_b = Node(name="B", _ref_id="b")
        node_a.next_ref = EntityRef(target=node_b)
        node_b.next_ref = EntityRef(target=node_a)

        resolver.register(node_a, "a")
        resolver.register(node_b, "b")

        cycles = resolver.check_all_cycles()
        assert len(cycles) >= 1

    def test_clear(self):
        resolver = ReferenceResolver()
        resolver.register(SimpleEntity("test"))

        resolver.clear()
        assert resolver.get_pending_count() == 0
        assert len(resolver.get_resolution_order()) == 0

    def test_resolution_order(self):
        resolver = ReferenceResolver()

        e1 = SimpleEntity("first", ref_id="id_1")
        e2 = SimpleEntity("second", ref_id="id_2")
        e3 = SimpleEntity("third", ref_id="id_3")

        resolver.register(e1, "id_1")
        resolver.register(e2, "id_2")
        resolver.register(e3, "id_3")

        order = resolver.get_resolution_order()
        assert order == ["id_1", "id_2", "id_3"]


# =============================================================================
# Tests for DeserializationContext
# =============================================================================


class TestDeserializationContext:
    """Tests for DeserializationContext."""

    def test_create_default(self):
        ctx = DeserializationContext()
        assert ctx.base is not None
        assert ctx.resolver is not None

    def test_create_with_base(self):
        base = SerializationContext(include_schema=False)
        ctx = DeserializationContext(base_ctx=base)
        assert ctx.base is base

    def test_register_type(self):
        ctx = DeserializationContext()
        ctx.register_type(SimpleEntity)
        assert ctx.get_type("SimpleEntity") is SimpleEntity

    def test_register_type_custom_name(self):
        ctx = DeserializationContext()
        ctx.register_type(SimpleEntity, "CustomName")
        assert ctx.get_type("CustomName") is SimpleEntity

    def test_get_type_nonexistent(self):
        ctx = DeserializationContext()
        assert ctx.get_type("Unknown") is None

    def test_deserialize_ref(self):
        ctx = DeserializationContext()
        entity = SimpleEntity("test", ref_id="test_1")
        ctx.resolver.register(entity, "test_1")

        data = {"__ref__": "test_1", "__ref_type__": "SimpleEntity"}
        ref = ctx.deserialize_ref(data, SimpleEntity)

        assert ref.is_resolved
        assert ref.get() is entity

    def test_finalize(self):
        ctx = DeserializationContext()

        # Add pending ref
        ref: EntityRef[SimpleEntity] = EntityRef(ref_id="missing")
        ctx.resolver.resolve_ref(ref)

        errors = ctx.finalize()
        assert len(errors) == 1


# =============================================================================
# Tests for serialize_with_refs
# =============================================================================


class TestSerializeWithRefs:
    """Tests for serialize_with_refs function."""

    def test_simple_object(self):
        entity = SimpleEntity("test")
        data, registry = serialize_with_refs(entity)

        assert "__ref_id__" in data
        assert registry.contains(entity.get_ref_id())

    def test_nested_references(self):
        player = Player(name="Alice", score=100)
        team = Team(name="Red Team")

        @dataclass
        class Match(Referenceable):
            player_ref: EntityRef[Player]
            team_ref: EntityRef[Team]
            _ref_id: str = ""

            def get_ref_id(self) -> str:
                return self._ref_id or f"match_{id(self):x}"

        match = Match(
            player_ref=EntityRef(target=player),
            team_ref=EntityRef(target=team),
        )

        data, registry = serialize_with_refs(match)
        assert "player_ref" in data
        assert "team_ref" in data

    def test_circular_reference(self):
        @dataclass
        class Node(Referenceable):
            name: str
            next_node: Optional[Any] = None
            _ref_id: str = ""

            def get_ref_id(self) -> str:
                return self._ref_id or f"node_{id(self):x}"

        node_a = Node(name="A", _ref_id="a")
        node_b = Node(name="B", _ref_id="b")
        node_a.next_node = node_b
        node_b.next_node = node_a  # Circular

        data, registry = serialize_with_refs(node_a)
        # Should not infinitely recurse
        assert data is not None

    def test_list_of_refs(self):
        players = [Player(name=f"P{i}", score=i * 10) for i in range(3)]

        @dataclass
        class Roster(Referenceable):
            players: List[Player]
            _ref_id: str = ""

            def get_ref_id(self) -> str:
                return self._ref_id or f"roster_{id(self):x}"

        roster = Roster(players=players)
        data, registry = serialize_with_refs(roster)

        assert "players" in data
        assert len(data["players"]) == 3


# =============================================================================
# Tests for deserialize_with_refs
# =============================================================================


class TestDeserializeWithRefs:
    """Tests for deserialize_with_refs function."""

    def test_simple_deserialize(self):
        data = {
            "name": "TestEntity",
            "__ref_id__": "entity_1",
        }

        @serializable()
        @dataclass
        class TestEntity(Referenceable):
            name: str
            _ref_id: str = field(default="", repr=False)

            def get_ref_id(self) -> str:
                return self._ref_id or f"test_{id(self):x}"

            def set_ref_id(self, ref_id: str) -> None:
                self._ref_id = ref_id

        result, errors = deserialize_with_refs(data, TestEntity)
        assert result.name == "TestEntity"
        assert len(errors) == 0

    def test_deserialize_with_refs(self):
        # This tests the full round-trip
        player = Player(name="Alice", score=100)
        player._ref_id = "player_1"

        data, _ = serialize_with_refs(player)
        result, errors = deserialize_with_refs(data, Player)

        assert result.name == "Alice"
        assert result.score == 100


# =============================================================================
# Tests for Global Registry
# =============================================================================


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def setup_method(self):
        get_global_registry().clear()

    def test_get_global_registry(self):
        registry = get_global_registry()
        assert isinstance(registry, ReferenceRegistry)

    def test_register_global(self):
        entity = SimpleEntity("global_test", ref_id="global_1")
        ref_id = register_global(entity)
        assert ref_id == "global_1"

    def test_resolve_global(self):
        entity = SimpleEntity("global_test", ref_id="global_2")
        register_global(entity)

        resolved = resolve_global("global_2")
        assert resolved is entity

    def test_resolve_global_missing(self):
        assert resolve_global("nonexistent") is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_player_team_relationship(self):
        # Create entities
        team = Team(name="Champions")
        team._ref_id = "team_champs"

        player1 = Player(name="Alice", score=100)
        player1._ref_id = "player_alice"

        player2 = Player(name="Bob", score=90)
        player2._ref_id = "player_bob"

        # Create resolver
        resolver = ReferenceResolver()
        resolver.register(team, "team_champs")
        resolver.register(player1, "player_alice")
        resolver.register(player2, "player_bob")

        # Create references
        team_ref = resolver.create_ref(team, Team)
        player_refs = [
            resolver.create_ref(player1, Player),
            resolver.create_ref(player2, Player),
        ]

        # Verify all resolved
        assert team_ref.get() is team
        assert player_refs[0].get() is player1
        assert player_refs[1].get() is player2

    def test_forward_reference_resolution(self):
        resolver = ReferenceResolver()

        # Create reference to entity that doesn't exist yet
        ref: EntityRef[Player] = EntityRef(ref_id="future_player")
        resolver.resolve_ref(ref)

        assert ref.state == RefState.DEFERRED
        assert ref.get() is None

        # Now create the entity
        player = Player(name="Delayed", score=50)
        resolver.register(player, "future_player")

        # Reference should now be resolved
        assert ref.is_resolved
        assert ref.get() is player

    def test_multiple_refs_to_same_entity(self):
        resolver = ReferenceResolver()
        entity = SimpleEntity("shared", ref_id="shared_1")
        resolver.register(entity)

        # Multiple refs to same entity
        ref1: EntityRef[SimpleEntity] = EntityRef(ref_id="shared_1")
        ref2: EntityRef[SimpleEntity] = EntityRef(ref_id="shared_1")
        ref3: EntityRef[SimpleEntity] = EntityRef(ref_id="shared_1")

        resolver.resolve_ref(ref1)
        resolver.resolve_ref(ref2)
        resolver.resolve_ref(ref3)

        # All should resolve to same object
        assert ref1.get() is entity
        assert ref2.get() is entity
        assert ref3.get() is entity

    def test_serialization_context_integration(self):
        ctx = SerializationContext(include_schema=False)

        entity = SimpleEntity("test", ref_id="ctx_test")
        ref = EntityRef(target=entity, target_type=SimpleEntity)

        data = ref.serialize(ctx)
        assert "__ref__" in data
        assert data["__ref__"] == "ctx_test"

    def test_complex_object_graph(self):
        @dataclass
        class Scene(Referenceable):
            name: str
            entities: List[SimpleEntity]
            _ref_id: str = ""

            def get_ref_id(self) -> str:
                return self._ref_id or f"scene_{id(self):x}"

        entities = [
            SimpleEntity(f"entity_{i}", ref_id=f"e{i}")
            for i in range(5)
        ]

        scene = Scene(name="TestScene", entities=entities, _ref_id="scene_1")

        resolver = ReferenceResolver()
        resolver.register(scene)
        for e in entities:
            resolver.register(e)

        # Verify all registered
        assert resolver.resolve("scene_1") is scene
        for i in range(5):
            assert resolver.resolve(f"e{i}") is entities[i]


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_ref_id(self):
        ref: EntityRef[SimpleEntity] = EntityRef(ref_id="")
        assert ref.is_null

    def test_none_target(self):
        ref: EntityRef[SimpleEntity] = EntityRef(target=None)
        assert ref.is_null

    def test_non_referenceable_object(self):
        class PlainObject:
            def __init__(self, value: int):
                self.value = value

        obj = PlainObject(42)
        registry = ReferenceRegistry(use_weak_refs=False)
        ref_id = registry.register(obj)
        assert ref_id.startswith("ref_")

    def test_resolver_max_depth(self):
        resolver = ReferenceResolver(max_depth=3)
        assert resolver._max_depth == 3

    def test_ref_info_creation(self):
        info = RefInfo(ref_id="test", target_type=SimpleEntity, state=RefState.RESOLVED)
        assert info.ref_id == "test"
        assert info.target_type is SimpleEntity
        assert info.state == RefState.RESOLVED

    def test_registry_with_non_weak_refs(self):
        registry = ReferenceRegistry(use_weak_refs=False)
        entity = SimpleEntity("test")
        ref_id = registry.register(entity)

        # Delete local reference
        del entity
        gc.collect()

        # Should still be retrievable (strong ref)
        retrieved = registry.get(ref_id)
        assert retrieved is not None

    def test_concurrent_registration(self):
        registry = ReferenceRegistry(use_weak_refs=False)
        resolver = ReferenceResolver(registry=registry)

        entities = []
        refs = []

        def register_batch(prefix: str, count: int):
            for i in range(count):
                e = SimpleEntity(f"{prefix}_{i}", ref_id=f"{prefix}_{i}")
                entities.append(e)
                resolver.register(e)

                r: EntityRef[SimpleEntity] = EntityRef(ref_id=f"{prefix}_{i}")
                refs.append(r)
                resolver.resolve_ref(r)

        threads = [
            threading.Thread(target=register_batch, args=(f"batch{i}", 5))
            for i in range(4)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All refs should be resolved
        assert all(r.is_resolved for r in refs)

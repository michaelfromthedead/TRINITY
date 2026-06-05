"""T-CC-4.4: Tests for CRDT/OT merge for scene edits.

Tests cover:
- VectorClock: causality tracking
- LWWRegister: last-writer-wins for scalars
- GCounter: grow-only counter
- PNCounter: positive-negative counter
- ORSet: observed-remove set
- LWWMap: last-writer-wins map
- CRDTDocument: scene/entity document
- OperationLog: sync with server
- Concurrent edits and merge scenarios
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from engine.collaboration.crdt import (
    CRDTDocument,
    CRDTError,
    CRDTOperation,
    CausalityViolation,
    GCounter,
    LWWMap,
    LWWRegister,
    MergeConflict,
    OperationLog,
    OperationType,
    ORSet,
    PNCounter,
    VectorClock,
)


# =============================================================================
# VectorClock Tests
# =============================================================================


class TestVectorClock:
    """Tests for VectorClock causality tracking."""

    def test_create_empty(self):
        """Test creating empty clock."""
        clock = VectorClock()
        assert clock.is_empty()
        assert clock.get("node1") == 0

    def test_create_with_values(self):
        """Test creating clock with initial values."""
        clock = VectorClock({"node1": 5, "node2": 3})
        assert clock.get("node1") == 5
        assert clock.get("node2") == 3
        assert not clock.is_empty()

    def test_increment(self):
        """Test incrementing a node's counter."""
        clock = VectorClock()
        result = clock.increment("node1")
        assert result == 1
        assert clock.get("node1") == 1

        result = clock.increment("node1")
        assert result == 2

    def test_tick(self):
        """Test tick creates new clock."""
        clock = VectorClock({"node1": 5})
        new_clock = clock.tick("node1")
        assert new_clock.get("node1") == 6
        assert clock.get("node1") == 5  # Original unchanged

    def test_merge(self):
        """Test merging two clocks (element-wise max)."""
        clock1 = VectorClock({"node1": 5, "node2": 3})
        clock2 = VectorClock({"node1": 3, "node2": 7, "node3": 2})
        merged = clock1.merge(clock2)

        assert merged.get("node1") == 5
        assert merged.get("node2") == 7
        assert merged.get("node3") == 2

    def test_merge_inplace(self):
        """Test in-place merge."""
        clock1 = VectorClock({"node1": 5})
        clock2 = VectorClock({"node1": 3, "node2": 7})
        clock1.merge_inplace(clock2)

        assert clock1.get("node1") == 5
        assert clock1.get("node2") == 7

    def test_happens_before_true(self):
        """Test happens-before when strictly less."""
        clock1 = VectorClock({"node1": 2, "node2": 3})
        clock2 = VectorClock({"node1": 3, "node2": 4})
        assert clock1.happens_before(clock2)
        assert not clock2.happens_before(clock1)

    def test_happens_before_false_equal(self):
        """Test happens-before when equal."""
        clock1 = VectorClock({"node1": 3, "node2": 3})
        clock2 = VectorClock({"node1": 3, "node2": 3})
        assert not clock1.happens_before(clock2)

    def test_happens_before_false_concurrent(self):
        """Test happens-before when concurrent."""
        clock1 = VectorClock({"node1": 5, "node2": 3})
        clock2 = VectorClock({"node1": 3, "node2": 5})
        assert not clock1.happens_before(clock2)
        assert not clock2.happens_before(clock1)

    def test_concurrent_with(self):
        """Test detecting concurrent clocks."""
        clock1 = VectorClock({"node1": 5, "node2": 3})
        clock2 = VectorClock({"node1": 3, "node2": 5})
        assert clock1.concurrent_with(clock2)
        assert clock2.concurrent_with(clock1)

    def test_dominates(self):
        """Test dominates relationship."""
        clock1 = VectorClock({"node1": 5, "node2": 5})
        clock2 = VectorClock({"node1": 3, "node2": 5})
        assert clock1.dominates(clock2)
        assert not clock2.dominates(clock1)

    def test_nodes(self):
        """Test getting all nodes."""
        clock = VectorClock({"node1": 1, "node2": 2})
        nodes = clock.nodes()
        assert nodes == frozenset(["node1", "node2"])

    def test_equality(self):
        """Test clock equality."""
        clock1 = VectorClock({"node1": 5})
        clock2 = VectorClock({"node1": 5})
        clock3 = VectorClock({"node1": 6})

        assert clock1 == clock2
        assert clock1 != clock3

    def test_serialization(self):
        """Test to_dict/from_dict."""
        clock = VectorClock({"node1": 5, "node2": 3})
        data = clock.to_dict()
        restored = VectorClock.from_dict(data)
        assert clock == restored

    def test_json_serialization(self):
        """Test JSON serialization."""
        clock = VectorClock({"node1": 5})
        json_str = clock.to_json()
        restored = VectorClock.from_json(json_str)
        assert clock == restored

    def test_copy(self):
        """Test creating a copy."""
        clock = VectorClock({"node1": 5})
        copy = clock.copy()
        copy.increment("node1")
        assert clock.get("node1") == 5
        assert copy.get("node1") == 6


# =============================================================================
# LWWRegister Tests
# =============================================================================


class TestLWWRegister:
    """Tests for Last-Writer-Wins Register."""

    def test_create(self):
        """Test creating register."""
        reg = LWWRegister(value=42)
        assert reg.get() == 42

    def test_set_newer_timestamp(self):
        """Test setting with newer timestamp wins."""
        reg = LWWRegister(value=42, timestamp=1.0, node_id="node1")
        result = reg.set(100, timestamp=2.0, node_id="node2")
        assert result is True
        assert reg.get() == 100

    def test_set_older_timestamp_rejected(self):
        """Test setting with older timestamp is rejected."""
        reg = LWWRegister(value=42, timestamp=2.0, node_id="node1")
        result = reg.set(100, timestamp=1.0, node_id="node2")
        assert result is False
        assert reg.get() == 42

    def test_set_same_timestamp_node_id_breaks_tie(self):
        """Test node_id tie-breaker when timestamps equal."""
        reg = LWWRegister(value=42, timestamp=1.0, node_id="aaa")
        # Higher node_id wins
        result = reg.set(100, timestamp=1.0, node_id="bbb")
        assert result is True
        assert reg.get() == 100

    def test_merge(self):
        """Test merging two registers."""
        reg1 = LWWRegister(value=42, timestamp=1.0, node_id="node1")
        reg2 = LWWRegister(value=100, timestamp=2.0, node_id="node2")
        merged = reg1.merge(reg2)
        assert merged.get() == 100

    def test_merge_inplace(self):
        """Test in-place merge."""
        reg1 = LWWRegister(value=42, timestamp=1.0, node_id="node1")
        reg2 = LWWRegister(value=100, timestamp=2.0, node_id="node2")
        changed = reg1.merge_inplace(reg2)
        assert changed is True
        assert reg1.get() == 100

    def test_serialization(self):
        """Test serialization."""
        reg = LWWRegister(value="hello", timestamp=1.5, node_id="test")
        data = reg.to_dict()
        restored = LWWRegister.from_dict(data)
        assert restored.get() == "hello"
        assert restored.timestamp == 1.5

    def test_copy(self):
        """Test creating a copy."""
        reg = LWWRegister(value=[1, 2, 3], timestamp=1.0, node_id="node1")
        copy = reg.copy()
        copy.value.append(4)
        assert reg.get() == [1, 2, 3]
        assert copy.get() == [1, 2, 3, 4]


# =============================================================================
# GCounter Tests
# =============================================================================


class TestGCounter:
    """Tests for Grow-only Counter."""

    def test_create_empty(self):
        """Test creating empty counter."""
        counter = GCounter()
        assert counter.value() == 0

    def test_increment(self):
        """Test incrementing counter."""
        counter = GCounter()
        counter.increment("node1", 5)
        counter.increment("node1", 3)
        counter.increment("node2", 2)

        assert counter.value() == 10
        assert counter.get_node_count("node1") == 8
        assert counter.get_node_count("node2") == 2

    def test_increment_negative_raises(self):
        """Test negative increment raises error."""
        counter = GCounter()
        with pytest.raises(CRDTError):
            counter.increment("node1", -1)

    def test_merge(self):
        """Test merging counters."""
        counter1 = GCounter()
        counter1.increment("node1", 5)
        counter1.increment("node2", 3)

        counter2 = GCounter()
        counter2.increment("node1", 3)
        counter2.increment("node2", 7)
        counter2.increment("node3", 2)

        merged = counter1.merge(counter2)
        assert merged.get_node_count("node1") == 5  # max(5, 3)
        assert merged.get_node_count("node2") == 7  # max(3, 7)
        assert merged.get_node_count("node3") == 2
        assert merged.value() == 14

    def test_merge_inplace(self):
        """Test in-place merge."""
        counter1 = GCounter()
        counter1.increment("node1", 5)

        counter2 = GCounter()
        counter2.increment("node1", 3)
        counter2.increment("node2", 7)

        counter1.merge_inplace(counter2)
        assert counter1.get_node_count("node1") == 5
        assert counter1.get_node_count("node2") == 7

    def test_serialization(self):
        """Test serialization."""
        counter = GCounter()
        counter.increment("node1", 5)
        data = counter.to_dict()
        restored = GCounter.from_dict(data)
        assert restored.value() == 5


# =============================================================================
# PNCounter Tests
# =============================================================================


class TestPNCounter:
    """Tests for Positive-Negative Counter."""

    def test_create_empty(self):
        """Test creating empty counter."""
        counter = PNCounter()
        assert counter.value() == 0

    def test_increment(self):
        """Test incrementing."""
        counter = PNCounter()
        counter.increment("node1", 5)
        assert counter.value() == 5

    def test_decrement(self):
        """Test decrementing."""
        counter = PNCounter()
        counter.increment("node1", 10)
        counter.decrement("node1", 3)
        assert counter.value() == 7

    def test_negative_value(self):
        """Test counter can go negative."""
        counter = PNCounter()
        counter.decrement("node1", 5)
        assert counter.value() == -5

    def test_merge(self):
        """Test merging PN-Counters."""
        counter1 = PNCounter()
        counter1.increment("node1", 10)
        counter1.decrement("node1", 3)

        counter2 = PNCounter()
        counter2.increment("node1", 5)
        counter2.decrement("node1", 7)
        counter2.increment("node2", 2)

        merged = counter1.merge(counter2)
        # node1 positive: max(10, 5) = 10
        # node1 negative: max(3, 7) = 7
        # node2 positive: 2, negative: 0
        # Total: 10 - 7 + 2 = 5
        assert merged.value() == 5

    def test_serialization(self):
        """Test serialization."""
        counter = PNCounter()
        counter.increment("node1", 10)
        counter.decrement("node1", 3)
        data = counter.to_dict()
        restored = PNCounter.from_dict(data)
        assert restored.value() == 7


# =============================================================================
# ORSet Tests
# =============================================================================


class TestORSet:
    """Tests for Observed-Remove Set."""

    def test_create_empty(self):
        """Test creating empty set."""
        orset = ORSet()
        assert len(orset) == 0
        assert orset.elements() == set()

    def test_add(self):
        """Test adding elements."""
        orset = ORSet()
        orset.add("apple", "node1")
        orset.add("banana", "node1")

        assert "apple" in orset
        assert "banana" in orset
        assert len(orset) == 2

    def test_remove(self):
        """Test removing elements."""
        orset = ORSet()
        orset.add("apple", "node1")
        orset.add("banana", "node1")
        orset.remove("apple")

        assert "apple" not in orset
        assert "banana" in orset

    def test_add_after_remove(self):
        """Test adding element after removal."""
        orset = ORSet()
        orset.add("apple", "node1")
        orset.remove("apple")
        orset.add("apple", "node1")

        assert "apple" in orset

    def test_concurrent_add_remove_add_wins(self):
        """Test add-wins semantics for concurrent add/remove."""
        orset1 = ORSet()
        orset1.add("apple", "node1")

        orset2 = orset1.copy()
        orset1.remove("apple")  # node1 removes
        orset2.add("apple", "node2")  # node2 adds concurrently

        merged = orset1.merge(orset2)
        # Add wins because node2's add has a new tag
        assert "apple" in merged

    def test_merge(self):
        """Test merging sets."""
        orset1 = ORSet()
        orset1.add("apple", "node1")
        orset1.add("banana", "node1")

        orset2 = ORSet()
        orset2.add("banana", "node2")
        orset2.add("cherry", "node2")

        merged = orset1.merge(orset2)
        assert merged.elements() == {"apple", "banana", "cherry"}

    def test_merge_with_removals(self):
        """Test merging with removals."""
        orset1 = ORSet()
        orset1.add("apple", "node1")
        orset1.add("banana", "node1")

        orset2 = orset1.copy()
        orset2.remove("apple")

        merged = orset1.merge(orset2)
        # apple should be removed (no concurrent add)
        assert "apple" not in merged
        assert "banana" in merged

    def test_iteration(self):
        """Test iterating over set."""
        orset = ORSet()
        orset.add("apple", "node1")
        orset.add("banana", "node1")

        elements = list(orset)
        assert set(elements) == {"apple", "banana"}

    def test_serialization(self):
        """Test serialization."""
        orset = ORSet()
        orset.add("apple", "node1")
        orset.add("banana", "node1")

        data = orset.to_dict()
        restored = ORSet.from_dict(data)
        assert restored.elements() == {"apple", "banana"}


# =============================================================================
# LWWMap Tests
# =============================================================================


class TestLWWMap:
    """Tests for Last-Writer-Wins Map."""

    def test_create_empty(self):
        """Test creating empty map."""
        lwwmap = LWWMap()
        assert len(lwwmap) == 0

    def test_set_get(self):
        """Test setting and getting values."""
        lwwmap = LWWMap()
        lwwmap.set("key1", "value1")
        lwwmap.set("key2", 42)

        assert lwwmap.get("key1") == "value1"
        assert lwwmap.get("key2") == 42
        assert lwwmap.get("nonexistent") is None

    def test_subscript_access(self):
        """Test subscript access."""
        lwwmap = LWWMap()
        lwwmap["key1"] = "value1"
        assert lwwmap["key1"] == "value1"

        with pytest.raises(KeyError):
            _ = lwwmap["nonexistent"]

    def test_remove(self):
        """Test removing keys."""
        lwwmap = LWWMap()
        lwwmap.set("key1", "value1", timestamp=1.0)
        lwwmap.remove("key1", timestamp=2.0)

        assert "key1" not in lwwmap
        assert lwwmap.get("key1") is None

    def test_remove_with_concurrent_set(self):
        """Test remove vs concurrent set."""
        lwwmap = LWWMap()
        lwwmap.set("key1", "value1", timestamp=1.0, node_id="node1")

        # Later set wins
        lwwmap.remove("key1", timestamp=2.0, node_id="node2")
        lwwmap.set("key1", "value2", timestamp=3.0, node_id="node1")

        assert "key1" in lwwmap
        assert lwwmap.get("key1") == "value2"

    def test_merge(self):
        """Test merging maps."""
        map1 = LWWMap()
        map1.set("key1", "value1", timestamp=1.0, node_id="node1")
        map1.set("key2", "value2", timestamp=1.0, node_id="node1")

        map2 = LWWMap()
        map2.set("key1", "newer", timestamp=2.0, node_id="node2")
        map2.set("key3", "value3", timestamp=1.0, node_id="node2")

        merged = map1.merge(map2)
        assert merged.get("key1") == "newer"
        assert merged.get("key2") == "value2"
        assert merged.get("key3") == "value3"

    def test_keys_values_items(self):
        """Test keys, values, items methods."""
        lwwmap = LWWMap()
        lwwmap.set("key1", "value1")
        lwwmap.set("key2", "value2")

        assert lwwmap.keys() == {"key1", "key2"}
        assert set(lwwmap.values()) == {"value1", "value2"}
        assert set(lwwmap.items()) == {("key1", "value1"), ("key2", "value2")}

    def test_serialization(self):
        """Test serialization."""
        lwwmap = LWWMap()
        lwwmap.set("key1", "value1", timestamp=1.0, node_id="node1")

        data = lwwmap.to_dict()
        restored = LWWMap.from_dict(data)
        assert restored.get("key1") == "value1"


# =============================================================================
# OperationLog Tests
# =============================================================================


class TestOperationLog:
    """Tests for Operation Log."""

    def test_create(self):
        """Test creating operation log."""
        log = OperationLog("node1")
        assert len(log) == 0

    def test_create_operation(self):
        """Test creating an operation."""
        log = OperationLog("node1")
        op = log.create_operation(
            OperationType.SET,
            "registers.position",
            {"value": [0, 0, 0], "timestamp": 1.0, "node_id": "node1"},
        )

        assert op.type == OperationType.SET
        assert op.path == "registers.position"
        assert len(log) == 1

    def test_add_external_operation(self):
        """Test adding external operation."""
        log = OperationLog("node1")
        op = CRDTOperation(
            id="ext-op-1",
            type=OperationType.SET,
            path="registers.x",
            value=42,
            timestamp=1.0,
            node_id="node2",
            clock=VectorClock({"node2": 1}),
        )

        result = log.add_operation(op)
        assert result is True
        assert len(log) == 1

    def test_add_duplicate_rejected(self):
        """Test duplicate operations are rejected."""
        log = OperationLog("node1")
        op = CRDTOperation(
            id="op-1",
            type=OperationType.SET,
            path="registers.x",
            value=42,
            timestamp=1.0,
            node_id="node2",
            clock=VectorClock({"node2": 1}),
        )

        log.add_operation(op)
        result = log.add_operation(op)
        assert result is False
        assert len(log) == 1

    def test_get_operations(self):
        """Test getting operations."""
        log = OperationLog("node1")
        log.create_operation(OperationType.SET, "path1", "value1")
        log.create_operation(OperationType.SET, "path2", "value2")

        ops = log.get_operations()
        assert len(ops) == 2

    def test_serialization(self):
        """Test serialization."""
        log = OperationLog("node1")
        log.create_operation(OperationType.SET, "path1", "value1")

        data = log.to_dict()
        restored = OperationLog.from_dict(data)
        assert len(restored) == 1


# =============================================================================
# CRDTDocument Tests
# =============================================================================


class TestCRDTDocument:
    """Tests for CRDT Document."""

    def test_create(self):
        """Test creating document."""
        doc = CRDTDocument("scene-1", "editor-1")
        assert doc.doc_id == "scene-1"
        assert doc.node_id == "editor-1"

    def test_register_operations(self):
        """Test register get/set."""
        doc = CRDTDocument("scene-1", "editor-1")
        doc.set_register("position", [0, 0, 0])
        doc.set_register("rotation", [0, 0, 0, 1])

        assert doc.get_register("position") == [0, 0, 0]
        assert doc.get_register("rotation") == [0, 0, 0, 1]
        assert doc.get_register("nonexistent", "default") == "default"

    def test_counter_operations(self):
        """Test counter operations."""
        doc = CRDTDocument("scene-1", "editor-1")
        doc.increment_counter("views", 5)
        doc.increment_counter("views", 3)

        assert doc.get_counter("views") == 8

    def test_pn_counter_operations(self):
        """Test PN-Counter operations."""
        doc = CRDTDocument("scene-1", "editor-1")
        doc.increment_counter("score", 10, pn=True)
        doc.decrement_counter("score", 3)

        assert doc.get_counter("score") == 7

    def test_set_operations(self):
        """Test set operations."""
        doc = CRDTDocument("scene-1", "editor-1")
        doc.add_to_set("tags", "player")
        doc.add_to_set("tags", "enemy")
        doc.add_to_set("tags", "npc")
        doc.remove_from_set("tags", "enemy")

        assert doc.get_set("tags") == {"player", "npc"}
        assert doc.set_contains("tags", "player")
        assert not doc.set_contains("tags", "enemy")

    def test_map_operations(self):
        """Test map operations."""
        doc = CRDTDocument("scene-1", "editor-1")
        doc.set_map_value("properties", "health", 100)
        doc.set_map_value("properties", "speed", 5.0)
        doc.remove_map_value("properties", "speed")

        assert doc.get_map_value("properties", "health") == 100
        assert doc.get_map_value("properties", "speed") is None
        assert doc.get_map("properties") == {"health": 100}

    def test_merge_registers(self):
        """Test merging documents with registers."""
        doc1 = CRDTDocument("scene-1", "editor-1")
        doc2 = CRDTDocument("scene-1", "editor-2")

        # doc1 sets position at t=1
        doc1._registers["position"] = LWWRegister([0, 0, 0], 1.0, "editor-1")
        # doc2 sets position at t=2 (newer)
        doc2._registers["position"] = LWWRegister([1, 1, 1], 2.0, "editor-2")

        merged = doc1.merge(doc2)
        assert merged.get_register("position") == [1, 1, 1]

    def test_merge_counters(self):
        """Test merging documents with counters."""
        doc1 = CRDTDocument("scene-1", "editor-1")
        doc2 = CRDTDocument("scene-1", "editor-2")

        doc1.increment_counter("views", 5)
        doc2.increment_counter("views", 3)

        merged = doc1.merge(doc2)
        assert merged.get_counter("views") == 8

    def test_merge_sets(self):
        """Test merging documents with sets."""
        doc1 = CRDTDocument("scene-1", "editor-1")
        doc2 = CRDTDocument("scene-1", "editor-2")

        doc1.add_to_set("tags", "tag1")
        doc2.add_to_set("tags", "tag2")

        merged = doc1.merge(doc2)
        assert merged.get_set("tags") == {"tag1", "tag2"}

    def test_merge_inplace(self):
        """Test in-place merge."""
        doc1 = CRDTDocument("scene-1", "editor-1")
        doc2 = CRDTDocument("scene-1", "editor-2")

        doc1.set_register("x", 10)
        doc2.set_register("y", 20)

        doc1.merge_inplace(doc2)
        assert doc1.get_register("x") == 10
        assert doc1.get_register("y") == 20

    def test_apply_operation(self):
        """Test applying a single operation."""
        doc = CRDTDocument("scene-1", "editor-1")
        op = CRDTOperation(
            id="op-1",
            type=OperationType.SET,
            path="registers.x",
            value={"value": 42, "timestamp": 1.0, "node_id": "editor-2"},
            timestamp=1.0,
            node_id="editor-2",
            clock=VectorClock({"editor-2": 1}),
        )

        result = doc.apply_operation(op)
        assert result is True
        assert doc.get_register("x") == 42

    def test_apply_duplicate_operation_rejected(self):
        """Test duplicate operations are rejected."""
        doc = CRDTDocument("scene-1", "editor-1")
        op = CRDTOperation(
            id="op-1",
            type=OperationType.SET,
            path="registers.x",
            value={"value": 42, "timestamp": 1.0, "node_id": "editor-2"},
            timestamp=1.0,
            node_id="editor-2",
            clock=VectorClock({"editor-2": 1}),
        )

        doc.apply_operation(op)
        result = doc.apply_operation(op)
        assert result is False

    def test_serialization(self):
        """Test document serialization."""
        doc = CRDTDocument("scene-1", "editor-1")
        doc.set_register("position", [1, 2, 3])
        doc.increment_counter("views", 5)
        doc.add_to_set("tags", "test")
        doc.set_map_value("props", "key", "value")

        data = doc.to_dict()
        restored = CRDTDocument.from_dict(data)

        assert restored.doc_id == "scene-1"
        assert restored.get_register("position") == [1, 2, 3]
        assert restored.get_counter("views") == 5
        assert restored.get_set("tags") == {"test"}
        assert restored.get_map_value("props", "key") == "value"

    def test_json_serialization(self):
        """Test JSON serialization."""
        doc = CRDTDocument("scene-1", "editor-1")
        doc.set_register("x", 42)

        json_str = doc.to_json()
        restored = CRDTDocument.from_json(json_str)

        assert restored.get_register("x") == 42

    def test_field_names(self):
        """Test getting all field names."""
        doc = CRDTDocument("scene-1", "editor-1")
        doc.set_register("reg1", 1)
        doc.increment_counter("cnt1")
        doc.add_to_set("set1", "a")
        doc.set_map_value("map1", "k", "v")

        names = doc.field_names()
        assert names == {"reg1", "cnt1", "set1", "map1"}

    def test_clock_updates(self):
        """Test clock updates on operations."""
        doc = CRDTDocument("scene-1", "editor-1")
        initial_clock = doc.clock

        doc.set_register("x", 1)
        clock_after = doc.clock

        assert clock_after.get("editor-1") > initial_clock.get("editor-1")


# =============================================================================
# Concurrent Editing Scenarios
# =============================================================================


class TestConcurrentEditing:
    """Tests for concurrent editing scenarios."""

    def test_two_editors_concurrent_register_updates(self):
        """Test two editors making concurrent register updates."""
        doc1 = CRDTDocument("scene-1", "editor-1")
        doc2 = CRDTDocument("scene-1", "editor-2")

        # Editor 1 sets at t=1
        doc1._registers["position"] = LWWRegister([0, 0, 0], 1.0, "editor-1")
        # Editor 2 sets at t=2
        doc2._registers["position"] = LWWRegister([5, 5, 5], 2.0, "editor-2")

        # Merge in both directions - should get same result
        merged1 = doc1.merge(doc2)
        merged2 = doc2.merge(doc1)

        assert merged1.get_register("position") == [5, 5, 5]
        assert merged2.get_register("position") == [5, 5, 5]

    def test_two_editors_concurrent_counter_increments(self):
        """Test two editors incrementing same counter."""
        doc1 = CRDTDocument("scene-1", "editor-1")
        doc2 = CRDTDocument("scene-1", "editor-2")

        doc1.increment_counter("hits", 10)
        doc2.increment_counter("hits", 5)

        merged = doc1.merge(doc2)
        assert merged.get_counter("hits") == 15

    def test_two_editors_set_add_remove(self):
        """Test two editors with set add/remove."""
        doc1 = CRDTDocument("scene-1", "editor-1")
        doc2 = CRDTDocument("scene-1", "editor-2")

        # Both add same element
        doc1.add_to_set("components", "Transform")
        doc2.add_to_set("components", "Transform")

        # Editor1 also adds Physics
        doc1.add_to_set("components", "Physics")

        # Editor2 removes Transform and adds Renderer
        doc2.remove_from_set("components", "Transform")
        doc2.add_to_set("components", "Renderer")

        merged = doc1.merge(doc2)
        components = merged.get_set("components")

        # Transform should be present (add-wins from doc1)
        # Physics should be present
        # Renderer should be present
        assert "Physics" in components
        assert "Renderer" in components

    def test_three_editors_cascade_merge(self):
        """Test three editors with cascade merge."""
        doc1 = CRDTDocument("scene-1", "editor-1")
        doc2 = CRDTDocument("scene-1", "editor-2")
        doc3 = CRDTDocument("scene-1", "editor-3")

        doc1.set_register("x", 1)
        doc2.set_register("y", 2)
        doc3.set_register("z", 3)

        # Merge 1 and 2
        merged12 = doc1.merge(doc2)
        # Merge result with 3
        final = merged12.merge(doc3)

        assert final.get_register("x") == 1
        assert final.get_register("y") == 2
        assert final.get_register("z") == 3

    def test_commutative_merge(self):
        """Test that merge is commutative."""
        doc1 = CRDTDocument("scene-1", "editor-1")
        doc2 = CRDTDocument("scene-1", "editor-2")

        doc1.set_register("a", 1)
        doc1.increment_counter("c", 5)

        doc2.set_register("b", 2)
        doc2.increment_counter("c", 3)

        merged1 = doc1.merge(doc2)
        merged2 = doc2.merge(doc1)

        assert merged1.get_register("a") == merged2.get_register("a")
        assert merged1.get_register("b") == merged2.get_register("b")
        assert merged1.get_counter("c") == merged2.get_counter("c")

    def test_associative_merge(self):
        """Test that merge is associative."""
        doc1 = CRDTDocument("scene-1", "editor-1")
        doc2 = CRDTDocument("scene-1", "editor-2")
        doc3 = CRDTDocument("scene-1", "editor-3")

        doc1.increment_counter("x", 1)
        doc2.increment_counter("x", 2)
        doc3.increment_counter("x", 3)

        # (1 merge 2) merge 3
        merged_12_3 = doc1.merge(doc2).merge(doc3)

        # 1 merge (2 merge 3)
        merged_1_23 = doc1.merge(doc2.merge(doc3))

        assert merged_12_3.get_counter("x") == merged_1_23.get_counter("x")

    def test_idempotent_merge(self):
        """Test that merge is idempotent."""
        doc1 = CRDTDocument("scene-1", "editor-1")
        doc1.set_register("x", 42)

        merged_once = doc1.merge(doc1)
        merged_twice = merged_once.merge(doc1)

        assert merged_once.get_register("x") == merged_twice.get_register("x")

    def test_no_data_loss_on_merge(self):
        """Test that merge never loses data."""
        doc1 = CRDTDocument("scene-1", "editor-1")
        doc2 = CRDTDocument("scene-1", "editor-2")

        # Each editor adds unique data
        for i in range(10):
            doc1.set_register(f"reg1_{i}", i)
            doc1.add_to_set("items1", f"item1_{i}")
            doc1.increment_counter("count1", 1)

            doc2.set_register(f"reg2_{i}", i * 10)
            doc2.add_to_set("items2", f"item2_{i}")
            doc2.increment_counter("count2", 1)

        merged = doc1.merge(doc2)

        # All data should be present
        for i in range(10):
            assert merged.get_register(f"reg1_{i}") == i
            assert merged.get_register(f"reg2_{i}") == i * 10
            assert merged.set_contains("items1", f"item1_{i}")
            assert merged.set_contains("items2", f"item2_{i}")

        assert merged.get_counter("count1") == 10
        assert merged.get_counter("count2") == 10


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_register_access(self):
        """Test concurrent register access."""
        reg = LWWRegister(value=0, timestamp=0.0, node_id="main")
        errors = []

        def writer(node_id: str, values: range):
            try:
                for v in values:
                    reg.set(v, time.time(), node_id)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(4):
            t = threading.Thread(target=writer, args=(f"node{i}", range(100)))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Value should be set to something
        assert reg.get() is not None

    def test_concurrent_counter_increments(self):
        """Test concurrent counter increments."""
        counter = GCounter()
        errors = []

        def incrementer(node_id: str, count: int):
            try:
                for _ in range(count):
                    counter.increment(node_id, 1)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(4):
            t = threading.Thread(target=incrementer, args=(f"node{i}", 100))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert counter.value() == 400

    def test_concurrent_set_operations(self):
        """Test concurrent set operations."""
        orset = ORSet()
        errors = []

        def worker(node_id: str, items: range):
            try:
                for i in items:
                    orset.add(f"item-{i}", node_id)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(4):
            t = threading.Thread(target=worker, args=(f"node{i}", range(25)))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Should have all 25 items (each added by all 4 nodes)
        assert len(orset) == 25

    def test_concurrent_document_operations(self):
        """Test concurrent document operations."""
        doc = CRDTDocument("scene-1", "main")
        errors = []

        def worker(worker_id: int):
            try:
                for i in range(20):
                    doc.set_register(f"reg_{worker_id}_{i}", i)
                    doc.increment_counter(f"cnt_{worker_id}", 1)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker, i) for i in range(4)]
            for f in futures:
                f.result()

        assert not errors


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for exception handling."""

    def test_crdt_error(self):
        """Test CRDT error creation."""
        error = CRDTError("Test error", {"key": "value"})
        assert str(error) == "Test error"
        assert error.details == {"key": "value"}

    def test_causality_violation(self):
        """Test causality violation error."""
        clock1 = VectorClock({"node1": 5})
        clock2 = VectorClock({"node1": 3})
        error = CausalityViolation("Violation", clock1, clock2)
        assert error.expected_clock == clock1
        assert error.actual_clock == clock2

    def test_merge_conflict(self):
        """Test merge conflict error."""
        error = MergeConflict("Conflict", "local", "remote")
        assert error.local_value == "local"
        assert error.remote_value == "remote"

    def test_gcounter_negative_increment_error(self):
        """Test GCounter rejects negative increments."""
        counter = GCounter()
        with pytest.raises(CRDTError) as exc_info:
            counter.increment("node1", -5)
        assert "GCounter" in str(exc_info.value)

    def test_decrement_gcounter_error(self):
        """Test decrementing a GCounter raises error."""
        doc = CRDTDocument("scene-1", "editor-1")
        doc.increment_counter("gcnt", 5)  # Creates GCounter

        with pytest.raises(CRDTError) as exc_info:
            doc.decrement_counter("gcnt", 1)
        assert "GCounter" in str(exc_info.value)

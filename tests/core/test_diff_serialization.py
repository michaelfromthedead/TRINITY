"""Tests for diff-based serialization (T-CC-2.8).

Tests DiffSerializer, DiffApplier, UndoStack, and NetworkDelta functionality.
Target: 50+ tests covering all components.
"""
import copy
import json
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

import pytest

from engine.core.serialization import (
    SchemaVersion,
    SerializationContext,
    SerializationFormat,
    serializable,
)
from engine.core.serialization_formats import (
    DiffEntry,
    DiffPatch,
)
from engine.core.diff_serialization import (
    DiffApplier,
    DiffMeta,
    DiffOperation,
    DiffSerializer,
    NetworkDelta,
    NetworkDeltaAccumulator,
    NetworkDeltaBuilder,
    NetworkDeltaFlags,
    SerializedDiff,
    UndoEntry,
    UndoStack,
    apply_state_diff,
    compute_state_diff,
    create_diff_applier,
    create_diff_serializer,
    create_undo_stack,
    _compute_hash,
    _flatten_object,
    _parse_path,
    _set_nested_value,
    _unflatten_object,
)


# Test fixtures

class Color(Enum):
    """Test enum."""
    RED = auto()
    GREEN = auto()
    BLUE = auto()


@serializable()
@dataclass
class SimpleEntity:
    """Simple test entity."""
    name: str
    value: int
    active: bool = True


@serializable(version="2.0.0")
@dataclass
class ComplexEntity:
    """Complex test entity with nested data."""
    id: str
    position: Dict[str, float]
    tags: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


@serializable()
@dataclass
class NestedEntity:
    """Entity with nested serializable."""
    name: str
    inner: SimpleEntity
    items: List[SimpleEntity] = field(default_factory=list)


# ============================================================
# DiffMeta Tests
# ============================================================

class TestDiffMeta:
    """Tests for DiffMeta dataclass."""

    def test_default_timestamp(self):
        """Default timestamp is set."""
        meta = DiffMeta()
        assert meta.timestamp > 0
        assert meta.author is None
        assert meta.description is None

    def test_to_dict(self):
        """Convert to dictionary."""
        meta = DiffMeta(
            timestamp=1234567890.0,
            author="test",
            description="Test change",
            source_hash="abc123",
            target_hash="def456",
        )
        d = meta.to_dict()
        assert d["timestamp"] == 1234567890.0
        assert d["author"] == "test"
        assert d["description"] == "Test change"

    def test_from_dict(self):
        """Create from dictionary."""
        d = {
            "timestamp": 1234567890.0,
            "author": "test",
            "description": "Test change",
        }
        meta = DiffMeta.from_dict(d)
        assert meta.timestamp == 1234567890.0
        assert meta.author == "test"


# ============================================================
# SerializedDiff Tests
# ============================================================

class TestSerializedDiff:
    """Tests for SerializedDiff dataclass."""

    def test_empty_diff(self):
        """Empty diff has no entries."""
        diff = SerializedDiff()
        assert len(diff) == 0
        assert diff.is_empty()

    def test_diff_with_entries(self):
        """Diff with entries."""
        entries = [
            DiffEntry(path="name", operation="replace", old_value="a", new_value="b"),
            DiffEntry(path="value", operation="add", new_value=42),
        ]
        diff = SerializedDiff(entries=entries)
        assert len(diff) == 2
        assert not diff.is_empty()

    def test_iteration(self):
        """Iterate over entries."""
        entries = [
            DiffEntry(path="a", operation="add", new_value=1),
            DiffEntry(path="b", operation="add", new_value=2),
        ]
        diff = SerializedDiff(entries=entries)
        paths = [e.path for e in diff]
        assert paths == ["a", "b"]

    def test_to_dict(self):
        """Convert to dictionary."""
        diff = SerializedDiff(
            entries=[DiffEntry(path="x", operation="add", new_value=10)],
            meta=DiffMeta(author="tester"),
        )
        d = diff.to_dict()
        assert "entries" in d
        assert "meta" in d
        assert len(d["entries"]) == 1

    def test_from_dict(self):
        """Create from dictionary."""
        d = {
            "entries": [{"path": "x", "op": "add", "new": 10}],
            "meta": {"author": "tester"},
        }
        diff = SerializedDiff.from_dict(d)
        assert len(diff) == 1
        assert diff.entries[0].path == "x"

    def test_to_bytes_uncompressed(self):
        """Serialize to bytes without compression."""
        diff = SerializedDiff(
            entries=[DiffEntry(path="x", operation="add", new_value=10)],
        )
        data = diff.to_bytes(compress=False)
        assert data[0] == 0  # Uncompressed flag

    def test_to_bytes_compressed(self):
        """Serialize to bytes with compression."""
        # Large diff to trigger compression
        entries = [
            DiffEntry(path=f"field_{i}", operation="add", new_value="x" * 100)
            for i in range(20)
        ]
        diff = SerializedDiff(entries=entries)
        data = diff.to_bytes(compress=True)
        assert data[0] == 1  # Compressed flag

    def test_from_bytes_roundtrip(self):
        """Roundtrip through bytes."""
        entries = [
            DiffEntry(path="name", operation="replace", old_value="old", new_value="new"),
            DiffEntry(path="count", operation="add", new_value=42),
        ]
        original = SerializedDiff(entries=entries, meta=DiffMeta(author="test"))

        data = original.to_bytes()
        restored = SerializedDiff.from_bytes(data)

        assert len(restored) == len(original)
        assert restored.entries[0].path == "name"
        assert restored.meta.author == "test"

    def test_size_bytes(self):
        """Get size in bytes."""
        diff = SerializedDiff(
            entries=[DiffEntry(path="x", operation="add", new_value=10)],
        )
        size = diff.size_bytes()
        assert size > 0

    def test_invert_add(self):
        """Invert add operation."""
        diff = SerializedDiff(
            entries=[DiffEntry(path="x", operation="add", new_value=10)],
        )
        inverted = diff.invert()
        assert len(inverted) == 1
        assert inverted.entries[0].operation == "remove"
        assert inverted.entries[0].old_value == 10

    def test_invert_remove(self):
        """Invert remove operation."""
        diff = SerializedDiff(
            entries=[DiffEntry(path="x", operation="remove", old_value=10)],
        )
        inverted = diff.invert()
        assert inverted.entries[0].operation == "add"
        assert inverted.entries[0].new_value == 10

    def test_invert_replace(self):
        """Invert replace operation."""
        diff = SerializedDiff(
            entries=[DiffEntry(path="x", operation="replace", old_value=10, new_value=20)],
        )
        inverted = diff.invert()
        assert inverted.entries[0].operation == "replace"
        assert inverted.entries[0].old_value == 20
        assert inverted.entries[0].new_value == 10

    def test_invert_order(self):
        """Inverted diff has reversed entry order."""
        diff = SerializedDiff(
            entries=[
                DiffEntry(path="a", operation="add", new_value=1),
                DiffEntry(path="b", operation="add", new_value=2),
                DiffEntry(path="c", operation="add", new_value=3),
            ],
        )
        inverted = diff.invert()
        paths = [e.path for e in inverted]
        assert paths == ["c", "b", "a"]


# ============================================================
# Helper Function Tests
# ============================================================

class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_flatten_simple_dict(self):
        """Flatten simple dictionary."""
        obj = {"a": 1, "b": 2}
        flat = _flatten_object(obj)
        assert flat == {"a": 1, "b": 2}

    def test_flatten_nested_dict(self):
        """Flatten nested dictionary."""
        obj = {"outer": {"inner": 42}}
        flat = _flatten_object(obj)
        assert flat == {"outer.inner": 42}

    def test_flatten_list(self):
        """Flatten list."""
        obj = {"items": [1, 2, 3]}
        flat = _flatten_object(obj)
        assert flat == {"items[0]": 1, "items[1]": 2, "items[2]": 3}

    def test_flatten_nested_list(self):
        """Flatten nested list in dict."""
        obj = {"data": {"values": [10, 20]}}
        flat = _flatten_object(obj)
        assert flat == {"data.values[0]": 10, "data.values[1]": 20}

    def test_flatten_skips_dunder(self):
        """Flatten skips __dunder__ keys."""
        obj = {"__schema__": {}, "name": "test"}
        flat = _flatten_object(obj)
        assert "__schema__" not in flat
        assert flat == {"name": "test"}

    def test_parse_path_simple(self):
        """Parse simple path."""
        parts = _parse_path("name")
        assert parts == ["name"]

    def test_parse_path_dotted(self):
        """Parse dotted path."""
        parts = _parse_path("outer.inner.deep")
        assert parts == ["outer", "inner", "deep"]

    def test_parse_path_array(self):
        """Parse path with array index."""
        parts = _parse_path("items[0]")
        assert parts == ["items", 0]

    def test_parse_path_mixed(self):
        """Parse mixed path."""
        parts = _parse_path("data.items[2].name")
        assert parts == ["data", "items", 2, "name"]

    def test_parse_path_root(self):
        """Parse root path."""
        parts = _parse_path("$")
        assert parts == []

    def test_set_nested_simple(self):
        """Set nested value in dict."""
        obj = {"a": 1}
        _set_nested_value(obj, ["b"], 2)
        assert obj == {"a": 1, "b": 2}

    def test_set_nested_deep(self):
        """Set deeply nested value."""
        obj = {}
        _set_nested_value(obj, ["a", "b", "c"], 42)
        assert obj == {"a": {"b": {"c": 42}}}

    def test_set_nested_array(self):
        """Set value in array."""
        obj = {"items": []}
        _set_nested_value(obj, ["items", 0], "first")
        assert obj == {"items": ["first"]}

    def test_compute_hash(self):
        """Compute hash is consistent."""
        obj = {"name": "test", "value": 42}
        h1 = _compute_hash(obj)
        h2 = _compute_hash(obj)
        assert h1 == h2
        assert len(h1) == 12

    def test_compute_hash_different(self):
        """Different objects have different hashes."""
        h1 = _compute_hash({"a": 1})
        h2 = _compute_hash({"a": 2})
        assert h1 != h2


# ============================================================
# DiffSerializer Tests
# ============================================================

class TestDiffSerializer:
    """Tests for DiffSerializer."""

    def test_compute_no_change(self):
        """No diff when objects are identical."""
        serializer = DiffSerializer()
        obj = {"name": "test", "value": 42}
        diff = serializer.compute(obj, obj)
        assert diff.is_empty()

    def test_compute_add_field(self):
        """Detect added field."""
        serializer = DiffSerializer()
        old = {"name": "test"}
        new = {"name": "test", "value": 42}
        diff = serializer.compute(old, new)

        assert len(diff) == 1
        assert diff.entries[0].operation == "add"
        assert diff.entries[0].path == "value"
        assert diff.entries[0].new_value == 42

    def test_compute_remove_field(self):
        """Detect removed field."""
        serializer = DiffSerializer()
        old = {"name": "test", "value": 42}
        new = {"name": "test"}
        diff = serializer.compute(old, new)

        assert len(diff) == 1
        assert diff.entries[0].operation == "remove"
        assert diff.entries[0].path == "value"

    def test_compute_replace_field(self):
        """Detect replaced field."""
        serializer = DiffSerializer()
        old = {"value": 42}
        new = {"value": 100}
        diff = serializer.compute(old, new)

        assert len(diff) == 1
        assert diff.entries[0].operation == "replace"
        assert diff.entries[0].old_value == 42
        assert diff.entries[0].new_value == 100

    def test_compute_nested_change(self):
        """Detect change in nested structure."""
        serializer = DiffSerializer()
        old = {"position": {"x": 0, "y": 0}}
        new = {"position": {"x": 10, "y": 0}}
        diff = serializer.compute(old, new)

        assert len(diff) == 1
        assert diff.entries[0].path == "position.x"
        assert diff.entries[0].new_value == 10

    def test_compute_list_change(self):
        """Detect change in list."""
        serializer = DiffSerializer()
        old = {"items": [1, 2, 3]}
        new = {"items": [1, 2, 4]}
        diff = serializer.compute(old, new)

        assert any(e.path == "items[2]" for e in diff.entries)

    def test_compute_with_serializable(self):
        """Compute diff with @serializable objects."""
        serializer = DiffSerializer()
        old = SimpleEntity(name="test", value=10)
        new = SimpleEntity(name="test", value=20)
        diff = serializer.compute(old, new)

        assert len(diff) >= 1
        values = [e.new_value for e in diff.entries if e.path == "value"]
        assert 20 in values

    def test_compute_metadata(self):
        """Diff includes metadata when enabled."""
        serializer = DiffSerializer(include_metadata=True, author="tester")
        old = {"a": 1}
        new = {"a": 2}
        diff = serializer.compute(old, new, description="Test change")

        assert diff.meta.author == "tester"
        assert diff.meta.description == "Test change"
        assert diff.meta.source_hash is not None
        assert diff.meta.target_hash is not None

    def test_compute_no_metadata(self):
        """Diff without metadata."""
        serializer = DiffSerializer(include_metadata=False)
        diff = serializer.compute({"a": 1}, {"a": 2})

        assert diff.meta.timestamp == 0
        assert diff.meta.source_hash is None

    def test_compute_incremental(self):
        """Incremental diff only checks specified paths."""
        serializer = DiffSerializer()
        old = {"a": 1, "b": 2, "c": 3}
        new = {"a": 10, "b": 20, "c": 30}

        # Only check 'b'
        diff = serializer.compute_incremental(old, new, changed_paths={"b"})

        assert len(diff) == 1
        assert diff.entries[0].path == "b"


# ============================================================
# DiffApplier Tests
# ============================================================

class TestDiffApplier:
    """Tests for DiffApplier."""

    def test_apply_add(self):
        """Apply add operation."""
        applier = DiffApplier()
        base = {"name": "test"}
        diff = SerializedDiff(entries=[
            DiffEntry(path="value", operation="add", new_value=42),
        ])

        result = applier.apply(base, diff)
        assert result["value"] == 42

    def test_apply_remove(self):
        """Apply remove operation."""
        applier = DiffApplier()
        base = {"name": "test", "value": 42}
        diff = SerializedDiff(entries=[
            DiffEntry(path="value", operation="remove", old_value=42),
        ])

        result = applier.apply(base, diff)
        assert "value" not in result

    def test_apply_replace(self):
        """Apply replace operation."""
        applier = DiffApplier()
        base = {"value": 42}
        diff = SerializedDiff(entries=[
            DiffEntry(path="value", operation="replace", old_value=42, new_value=100),
        ])

        result = applier.apply(base, diff)
        assert result["value"] == 100

    def test_apply_nested(self):
        """Apply change to nested structure."""
        applier = DiffApplier()
        base = {"position": {"x": 0, "y": 0}}
        diff = SerializedDiff(entries=[
            DiffEntry(path="position.x", operation="replace", old_value=0, new_value=10),
        ])

        result = applier.apply(base, diff)
        assert result["position"]["x"] == 10
        assert result["position"]["y"] == 0

    def test_apply_list_element(self):
        """Apply change to list element."""
        applier = DiffApplier()
        base = {"items": [1, 2, 3]}
        diff = SerializedDiff(entries=[
            DiffEntry(path="items[1]", operation="replace", old_value=2, new_value=20),
        ])

        result = applier.apply(base, diff)
        assert result["items"] == [1, 20, 3]

    def test_apply_creates_structure(self):
        """Apply creates intermediate structure for add."""
        applier = DiffApplier()
        base = {}
        diff = SerializedDiff(entries=[
            DiffEntry(path="deep.nested.value", operation="add", new_value=42),
        ])

        result = applier.apply(base, diff)
        assert result["deep"]["nested"]["value"] == 42

    def test_apply_inverted(self):
        """Apply inverted diff (undo)."""
        applier = DiffApplier()
        base = {"value": 100}
        diff = SerializedDiff(entries=[
            DiffEntry(path="value", operation="replace", old_value=42, new_value=100),
        ])

        result = applier.apply_inverted(base, diff)
        assert result["value"] == 42

    def test_apply_to_serializable(self):
        """Apply diff to @serializable and get typed result."""
        applier = DiffApplier()
        base = SimpleEntity(name="test", value=10)
        diff = SerializedDiff(entries=[
            DiffEntry(path="value", operation="replace", old_value=10, new_value=20),
        ])

        result = applier.apply(base, diff, target_type=SimpleEntity)
        assert isinstance(result, SimpleEntity)
        assert result.value == 20

    def test_apply_does_not_mutate_base(self):
        """Apply does not mutate the base state."""
        applier = DiffApplier()
        base = {"value": 42}
        diff = SerializedDiff(entries=[
            DiffEntry(path="value", operation="replace", old_value=42, new_value=100),
        ])

        applier.apply(base, diff)
        assert base["value"] == 42  # Unchanged


# ============================================================
# UndoStack Tests
# ============================================================

class TestUndoStack:
    """Tests for UndoStack."""

    def test_initial_state(self):
        """Initial state is empty."""
        stack = UndoStack()
        assert not stack.can_undo
        assert not stack.can_redo
        assert stack.undo_depth == 0
        assert stack.redo_depth == 0

    def test_push_single(self):
        """Push single change."""
        stack = UndoStack()
        old = {"value": 1}
        new = {"value": 2}

        stack.push(old, new, "Change value")

        assert stack.can_undo
        assert not stack.can_redo
        assert stack.undo_depth == 1

    def test_push_no_change(self):
        """Push with no actual change."""
        stack = UndoStack()
        obj = {"value": 1}

        stack.push(obj, obj, "No change")

        assert not stack.can_undo  # Empty diff is not pushed

    def test_undo_single(self):
        """Undo single change."""
        stack = UndoStack()
        old = {"value": 1}
        new = {"value": 2}

        stack.push(old, new, "Change value")
        result, desc = stack.undo(new)

        assert result["value"] == 1
        assert desc == "Change value"
        assert not stack.can_undo
        assert stack.can_redo

    def test_redo_single(self):
        """Redo single change."""
        stack = UndoStack()
        old = {"value": 1}
        new = {"value": 2}

        stack.push(old, new, "Change value")
        undone, _ = stack.undo(new)
        redone, desc = stack.redo(undone)

        assert redone["value"] == 2
        assert desc == "Change value"
        assert stack.can_undo
        assert not stack.can_redo

    def test_multiple_undo(self):
        """Multiple undo operations."""
        stack = UndoStack()

        states = [{"value": i} for i in range(5)]
        for i in range(1, 5):
            stack.push(states[i-1], states[i], f"Change to {i}")

        assert stack.undo_depth == 4

        current = states[4]
        for expected in [3, 2, 1, 0]:
            current, _ = stack.undo(current)
            assert current["value"] == expected

        assert not stack.can_undo
        assert stack.redo_depth == 4

    def test_undo_clears_redo_on_new_action(self):
        """New action clears redo stack."""
        stack = UndoStack()

        stack.push({"v": 1}, {"v": 2}, "A")
        stack.push({"v": 2}, {"v": 3}, "B")
        stack.undo({"v": 3})

        assert stack.can_redo

        stack.push({"v": 2}, {"v": 10}, "C")

        assert not stack.can_redo

    def test_peek_undo(self):
        """Peek at next undo action."""
        stack = UndoStack()
        stack.push({"v": 1}, {"v": 2}, "First")
        stack.push({"v": 2}, {"v": 3}, "Second")

        assert stack.peek_undo() == "Second"

    def test_peek_redo(self):
        """Peek at next redo action."""
        stack = UndoStack()
        stack.push({"v": 1}, {"v": 2}, "First")
        stack.undo({"v": 2})

        assert stack.peek_redo() == "First"

    def test_get_undo_history(self):
        """Get undo history."""
        stack = UndoStack()
        for i in range(5):
            stack.push({"v": i}, {"v": i+1}, f"Step {i}")

        history = stack.get_undo_history(limit=3)
        assert len(history) == 3
        assert history[0] == "Step 4"  # Most recent first

    def test_max_depth_limit(self):
        """Respect max depth limit."""
        stack = UndoStack(max_depth=3)

        for i in range(10):
            stack.push({"v": i}, {"v": i+1}, f"Step {i}")

        assert stack.undo_depth == 3

    def test_memory_limit(self):
        """Respect memory limit."""
        stack = UndoStack(max_bytes=1000)

        # Push large changes to exceed limit
        for i in range(100):
            stack.push(
                {"data": "x" * 100},
                {"data": "y" * 100},
                f"Step {i}",
            )

        assert stack.memory_usage <= 1000 + 500  # Allow some tolerance

    def test_merge_consecutive_edits(self):
        """Merge consecutive edits within timeout."""
        stack = UndoStack(merge_timeout=1.0)

        stack.push({"v": 1}, {"v": 2}, "Edit", merge_path="value")
        stack.push({"v": 2}, {"v": 3}, "Edit", merge_path="value")
        stack.push({"v": 3}, {"v": 4}, "Edit", merge_path="value")

        # Should be merged into single undo
        assert stack.undo_depth == 1

    def test_no_merge_different_paths(self):
        """Don't merge edits with different paths."""
        stack = UndoStack(merge_timeout=1.0)

        stack.push({"v": 1}, {"v": 2}, "Edit A", merge_path="a")
        stack.push({"v": 2}, {"v": 3}, "Edit B", merge_path="b")

        assert stack.undo_depth == 2

    def test_clear(self):
        """Clear all history."""
        stack = UndoStack()
        stack.push({"v": 1}, {"v": 2}, "A")
        stack.push({"v": 2}, {"v": 3}, "B")
        stack.undo({"v": 3})

        stack.clear()

        assert not stack.can_undo
        assert not stack.can_redo
        assert stack.memory_usage == 0

    def test_undo_empty_raises(self):
        """Undo on empty stack raises."""
        stack = UndoStack()
        with pytest.raises(IndexError):
            stack.undo({"v": 1})

    def test_redo_empty_raises(self):
        """Redo on empty stack raises."""
        stack = UndoStack()
        with pytest.raises(IndexError):
            stack.redo({"v": 1})

    def test_get_stats(self):
        """Get statistics."""
        stack = UndoStack(max_depth=50, max_bytes=10000)
        stack.push({"v": 1}, {"v": 2}, "A")
        stack.undo({"v": 2})

        stats = stack.get_stats()
        assert stats["undo_depth"] == 0
        assert stats["redo_depth"] == 1
        assert stats["max_depth"] == 50
        assert stats["max_bytes"] == 10000

    def test_thread_safety(self):
        """Basic thread safety test."""
        stack = UndoStack()
        results = []

        def pusher():
            for i in range(10):
                stack.push({"v": i}, {"v": i+1}, f"Push {i}")

        def undoer():
            for _ in range(5):
                if stack.can_undo:
                    try:
                        stack.undo({"v": 0})
                    except IndexError:
                        pass

        threads = [
            threading.Thread(target=pusher),
            threading.Thread(target=undoer),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash


# ============================================================
# NetworkDelta Tests
# ============================================================

class TestNetworkDelta:
    """Tests for NetworkDelta."""

    def test_create_empty(self):
        """Create empty delta."""
        delta = NetworkDelta(entity_id="entity1", sequence=1)
        assert len(delta) == 0
        assert delta.is_empty()

    def test_add_change(self):
        """Add changes to delta."""
        delta = NetworkDelta(entity_id="entity1", sequence=1)
        delta.add_change("position.x", 0, 10)
        delta.add_change("position.y", 0, 20)

        assert len(delta) == 2
        assert not delta.is_empty()

    def test_add_change_no_op(self):
        """No entry when old equals new."""
        delta = NetworkDelta(entity_id="entity1", sequence=1)
        delta.add_change("value", 10, 10)
        assert delta.is_empty()

    def test_add_detects_operation_type(self):
        """Correct operation type for add/remove/replace."""
        delta = NetworkDelta(entity_id="e", sequence=1)

        delta.add_change("a", None, 1)  # Add
        delta.add_change("b", 2, None)  # Remove
        delta.add_change("c", 3, 4)     # Replace

        ops = {e.path: e.operation for e in delta.entries}
        assert ops["a"] == "add"
        assert ops["b"] == "remove"
        assert ops["c"] == "replace"

    def test_merge_deltas(self):
        """Merge two deltas."""
        d1 = NetworkDelta(entity_id="e", sequence=1)
        d1.add_change("a", 0, 1)
        d1.add_change("b", 0, 2)

        d2 = NetworkDelta(entity_id="e", sequence=2)
        d2.add_change("b", 2, 3)  # Override b
        d2.add_change("c", 0, 4)  # New field

        merged = d1.merge(d2)

        assert merged.sequence == 2
        paths = {e.path: e.new_value for e in merged.entries}
        assert paths["a"] == 1
        assert paths["b"] == 3
        assert paths["c"] == 4

    def test_merge_different_entities_raises(self):
        """Cannot merge deltas for different entities."""
        d1 = NetworkDelta(entity_id="e1", sequence=1)
        d2 = NetworkDelta(entity_id="e2", sequence=1)

        with pytest.raises(ValueError):
            d1.merge(d2)

    def test_to_bytes(self):
        """Serialize to bytes."""
        delta = NetworkDelta(entity_id="entity1", sequence=42)
        delta.add_change("x", 0, 100)

        data = delta.to_bytes()
        assert data[:4] == NetworkDelta.MAGIC

    def test_from_bytes_roundtrip(self):
        """Roundtrip through bytes."""
        original = NetworkDelta(
            entity_id="test_entity",
            sequence=123,
            flags=NetworkDeltaFlags.RELIABLE,
        )
        original.add_change("pos.x", 0.0, 10.5)
        original.add_change("pos.y", 0.0, 20.5)

        data = original.to_bytes()
        restored = NetworkDelta.from_bytes(data)

        assert restored.entity_id == "test_entity"
        assert restored.sequence == 123
        assert len(restored) == 2

    def test_compression(self):
        """Large deltas are compressed."""
        delta = NetworkDelta(entity_id="e", sequence=1)
        for i in range(50):
            delta.add_change(f"field_{i}", 0, "x" * 100)

        compressed = delta.to_bytes(compress=True)
        uncompressed = delta.to_bytes(compress=False)

        assert len(compressed) < len(uncompressed)

    def test_to_dict(self):
        """Convert to dictionary."""
        delta = NetworkDelta(entity_id="e", sequence=5)
        delta.add_change("x", 0, 1)

        d = delta.to_dict()
        assert d["entity_id"] == "e"
        assert d["sequence"] == 5
        assert len(d["entries"]) == 1

    def test_from_dict(self):
        """Create from dictionary."""
        d = {
            "entity_id": "e",
            "sequence": 10,
            "entries": [{"path": "x", "op": "add", "new": 42}],
            "flags": NetworkDeltaFlags.ORDERED,
        }
        delta = NetworkDelta.from_dict(d)

        assert delta.entity_id == "e"
        assert delta.sequence == 10
        assert delta.flags & NetworkDeltaFlags.ORDERED


class TestNetworkDeltaBuilder:
    """Tests for NetworkDeltaBuilder."""

    def test_track_single_change(self):
        """Track a single change."""
        builder = NetworkDeltaBuilder(entity_id="e1")
        builder.track_change("value", 0, 10)

        assert builder.has_changes()
        delta = builder.build()

        assert len(delta) == 1
        assert delta.entity_id == "e1"

    def test_track_multiple_changes(self):
        """Track multiple changes."""
        builder = NetworkDeltaBuilder(entity_id="e1")
        builder.track_change("x", 0, 10)
        builder.track_change("y", 0, 20)

        delta = builder.build()
        assert len(delta) == 2

    def test_merge_same_path(self):
        """Multiple changes to same path keep original old value."""
        builder = NetworkDeltaBuilder(entity_id="e1")
        builder.track_change("value", 0, 5)
        builder.track_change("value", 5, 10)

        delta = builder.build()
        assert len(delta) == 1

        entry = delta.entries[0]
        assert entry.old_value == 0  # Original old
        assert entry.new_value == 10  # Latest new

    def test_build_clears_pending(self):
        """Build clears pending changes."""
        builder = NetworkDeltaBuilder(entity_id="e1")
        builder.track_change("x", 0, 10)
        builder.build()

        assert not builder.has_changes()
        delta = builder.build()
        assert delta.is_empty()

    def test_sequence_increments(self):
        """Sequence number increments."""
        builder = NetworkDeltaBuilder(entity_id="e1")

        builder.track_change("a", 0, 1)
        d1 = builder.build()

        builder.track_change("b", 0, 2)
        d2 = builder.build()

        assert d2.sequence > d1.sequence

    def test_reliable_flag(self):
        """Set reliable flag."""
        builder = NetworkDeltaBuilder(entity_id="e1")
        builder.track_change("x", 0, 1)
        delta = builder.build(reliable=True)

        assert delta.flags & NetworkDeltaFlags.RELIABLE

    def test_ordered_flag(self):
        """Set ordered flag."""
        builder = NetworkDeltaBuilder(entity_id="e1")
        builder.track_change("x", 0, 1)
        delta = builder.build(ordered=True)

        assert delta.flags & NetworkDeltaFlags.ORDERED

    def test_clear(self):
        """Clear pending changes."""
        builder = NetworkDeltaBuilder(entity_id="e1")
        builder.track_change("x", 0, 1)
        builder.clear()

        assert not builder.has_changes()


class TestNetworkDeltaAccumulator:
    """Tests for NetworkDeltaAccumulator."""

    def test_add_single(self):
        """Add single delta."""
        acc = NetworkDeltaAccumulator()
        delta = NetworkDelta(entity_id="e", sequence=1)
        delta.add_change("x", 0, 1)

        result = acc.add(delta)
        assert result is None  # Not flushed yet
        assert not acc.is_empty()

    def test_auto_flush_on_max_entries(self):
        """Auto-flush when max entries reached."""
        acc = NetworkDeltaAccumulator(max_entries=5)

        for i in range(10):
            delta = NetworkDelta(entity_id="e", sequence=i)
            delta.add_change(f"field_{i}", 0, i)
            result = acc.add(delta)

            if result is not None:
                # Flushed
                assert "e" in result
                break

    def test_merge_same_entity(self):
        """Deltas for same entity are merged."""
        acc = NetworkDeltaAccumulator(max_entries=100)

        d1 = NetworkDelta(entity_id="e", sequence=1)
        d1.add_change("a", 0, 1)
        acc.add(d1)

        d2 = NetworkDelta(entity_id="e", sequence=2)
        d2.add_change("b", 0, 2)
        acc.add(d2)

        result = acc.flush()
        assert len(result) == 1
        assert len(result["e"]) == 2

    def test_separate_entities(self):
        """Different entities stay separate."""
        acc = NetworkDeltaAccumulator()

        d1 = NetworkDelta(entity_id="e1", sequence=1)
        d1.add_change("x", 0, 1)
        acc.add(d1)

        d2 = NetworkDelta(entity_id="e2", sequence=1)
        d2.add_change("y", 0, 2)
        acc.add(d2)

        result = acc.flush()
        assert "e1" in result
        assert "e2" in result

    def test_flush_empty(self):
        """Flush empty accumulator."""
        acc = NetworkDeltaAccumulator()
        result = acc.flush()
        assert len(result) == 0


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_roundtrip_diff_apply(self):
        """Full roundtrip: diff -> apply -> verify."""
        old = ComplexEntity(
            id="entity1",
            position={"x": 0.0, "y": 0.0, "z": 0.0},
            tags=["player", "active"],
            metadata={"score": 100},
        )
        new = ComplexEntity(
            id="entity1",
            position={"x": 10.0, "y": 5.0, "z": 0.0},
            tags=["player", "active", "moving"],
            metadata={"score": 150, "combo": 3},
        )

        # Compute diff
        serializer = DiffSerializer()
        diff = serializer.compute(old, new)

        assert not diff.is_empty()

        # Apply diff
        applier = DiffApplier()
        result = applier.apply(old, diff, target_type=ComplexEntity)

        assert result.position["x"] == 10.0
        assert result.position["y"] == 5.0
        assert "moving" in result.tags
        assert result.metadata["combo"] == 3

    def test_undo_redo_complex(self):
        """Complex undo/redo scenario."""
        stack = UndoStack()

        state = {"position": {"x": 0, "y": 0}, "health": 100}

        # Series of changes
        changes = [
            {"position": {"x": 10, "y": 0}, "health": 100},
            {"position": {"x": 10, "y": 20}, "health": 100},
            {"position": {"x": 10, "y": 20}, "health": 80},
        ]

        current = state
        for i, new_state in enumerate(changes):
            stack.push(current, new_state, f"Change {i}")
            current = new_state

        # Undo all
        for _ in range(3):
            current, _ = stack.undo(current)

        assert current == state

        # Redo all
        for _ in range(3):
            current, _ = stack.redo(current)

        assert current == changes[-1]

    def test_convenience_functions(self):
        """Test convenience functions."""
        old = {"name": "test", "value": 1}
        new = {"name": "test", "value": 2}

        diff = compute_state_diff(old, new, description="Update value")
        assert not diff.is_empty()

        result = apply_state_diff(old, diff)
        assert result["value"] == 2

    def test_factory_functions(self):
        """Test factory functions."""
        serializer = create_diff_serializer(author="test")
        assert serializer._author == "test"

        applier = create_diff_applier(validate=False)
        assert applier._validate is False

        stack = create_undo_stack(max_depth=50)
        assert stack._max_depth == 50

    def test_diff_smaller_than_full(self):
        """Verify diff is smaller than full serialization."""
        # Large object with small change
        large_data = {f"field_{i}": f"value_{i}" * 100 for i in range(100)}
        new_data = copy.deepcopy(large_data)
        new_data["field_50"] = "changed"

        serializer = DiffSerializer()
        diff = serializer.compute(large_data, new_data)

        diff_bytes = diff.to_bytes()
        full_bytes = json.dumps(new_data).encode()

        assert len(diff_bytes) < len(full_bytes) * 0.1  # Diff should be <10% of full


# ============================================================
# Edge Case Tests
# ============================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_objects(self):
        """Diff between empty objects."""
        serializer = DiffSerializer()
        diff = serializer.compute({}, {})
        assert diff.is_empty()

    def test_none_values(self):
        """Handle None values."""
        serializer = DiffSerializer()
        diff = serializer.compute({"a": None}, {"a": 1})
        assert len(diff) == 1

    def test_deeply_nested(self):
        """Handle deeply nested structures."""
        old = {"a": {"b": {"c": {"d": {"e": 1}}}}}
        new = {"a": {"b": {"c": {"d": {"e": 2}}}}}

        serializer = DiffSerializer()
        diff = serializer.compute(old, new)

        assert len(diff) == 1
        assert "a.b.c.d.e" in diff.entries[0].path

    def test_list_operations(self):
        """Various list operations."""
        serializer = DiffSerializer()
        applier = DiffApplier()

        # Add to list
        old = {"items": [1, 2]}
        new = {"items": [1, 2, 3]}
        diff = serializer.compute(old, new)
        result = applier.apply(old, diff)
        assert result["items"] == [1, 2, 3]

    def test_type_changes(self):
        """Handle type changes."""
        serializer = DiffSerializer()
        diff = serializer.compute({"value": 42}, {"value": "forty-two"})
        assert len(diff) == 1
        assert diff.entries[0].new_value == "forty-two"

    def test_special_characters_in_keys(self):
        """Handle special characters in keys."""
        serializer = DiffSerializer()
        applier = DiffApplier()

        old = {"key with spaces": 1, "key.with.dots": 2}
        new = {"key with spaces": 10, "key.with.dots": 20}

        diff = serializer.compute(old, new)
        result = applier.apply(old, diff)

        assert result["key with spaces"] == 10

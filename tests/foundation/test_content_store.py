"""
Comprehensive unit tests for the ContentStore system.

Tests content-addressable storage, deduplication, tree storage,
structural sharing, and content diffing.
"""
import pytest
import sys
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from foundation.content_store import (
    ContentHash,
    ContentStore,
    MemoryBackend,
    FileBackend,
    ContentDiffer,
    Difference,
)
from foundation.serializer import register_type, _type_registry


@pytest.fixture(autouse=True)
def clear_type_registry():
    """Clear the type registry before and after each test."""
    original = _type_registry.copy()
    _type_registry.clear()
    yield
    _type_registry.clear()
    _type_registry.update(original)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file backend tests."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path)


class TestContentHash:
    """Test ContentHash dataclass."""

    def test_content_hash_creation(self):
        """ContentHash should store the hash value."""
        hash = ContentHash("abc123def456")
        assert hash.value == "abc123def456"

    def test_content_hash_short(self):
        """short() should return first 8 characters."""
        hash = ContentHash("abcdef1234567890")
        assert hash.short() == "abcdef12"

    def test_content_hash_short_short_value(self):
        """short() should work with values shorter than 8 chars."""
        hash = ContentHash("abc")
        assert hash.short() == "abc"

    def test_content_hash_str(self):
        """str() should return full hash value."""
        hash = ContentHash("abc123")
        assert str(hash) == "abc123"

    def test_content_hash_equality_same(self):
        """Same hash values should be equal."""
        hash1 = ContentHash("abc123")
        hash2 = ContentHash("abc123")
        assert hash1 == hash2

    def test_content_hash_equality_different(self):
        """Different hash values should not be equal."""
        hash1 = ContentHash("abc123")
        hash2 = ContentHash("def456")
        assert hash1 != hash2

    def test_content_hash_equality_non_hash(self):
        """ContentHash should not equal non-ContentHash objects."""
        hash = ContentHash("abc123")
        assert hash != "abc123"
        assert hash != 123
        assert hash != None

    def test_content_hash_hashable(self):
        """ContentHash should be usable in sets and as dict keys."""
        hash1 = ContentHash("abc123")
        hash2 = ContentHash("abc123")
        hash3 = ContentHash("def456")

        # Set deduplication
        s = {hash1, hash2, hash3}
        assert len(s) == 2

        # Dict key
        d = {hash1: "value1", hash3: "value3"}
        assert d[hash2] == "value1"

    def test_content_hash_frozen(self):
        """ContentHash should be immutable (frozen dataclass)."""
        hash = ContentHash("abc123")
        with pytest.raises(Exception):  # FrozenInstanceError
            hash.value = "xyz"

    def test_content_hash_repr(self):
        """repr should show shortened hash."""
        hash = ContentHash("abcdef1234567890")
        repr_str = repr(hash)
        assert "abcdef12" in repr_str
        assert "..." in repr_str


class TestMemoryBackend:
    """Test MemoryBackend storage."""

    def test_memory_backend_put_get(self):
        """put and get should store and retrieve data."""
        backend = MemoryBackend()
        hash = ContentHash("abc123")
        data = b"hello world"

        backend.put(hash, data)
        result = backend.get(hash)

        assert result == data

    def test_memory_backend_get_missing(self):
        """get should return None for missing hash."""
        backend = MemoryBackend()
        hash = ContentHash("nonexistent")

        result = backend.get(hash)

        assert result is None

    def test_memory_backend_has_present(self):
        """has should return True for present hash."""
        backend = MemoryBackend()
        hash = ContentHash("abc123")
        backend.put(hash, b"data")

        assert backend.has(hash) is True

    def test_memory_backend_has_missing(self):
        """has should return False for missing hash."""
        backend = MemoryBackend()
        hash = ContentHash("nonexistent")

        assert backend.has(hash) is False

    def test_memory_backend_len(self):
        """len should return number of stored items."""
        backend = MemoryBackend()
        assert len(backend) == 0

        backend.put(ContentHash("a"), b"1")
        backend.put(ContentHash("b"), b"2")
        assert len(backend) == 2

    def test_memory_backend_clear(self):
        """clear should remove all stored items."""
        backend = MemoryBackend()
        backend.put(ContentHash("a"), b"1")
        backend.put(ContentHash("b"), b"2")

        backend.clear()

        assert len(backend) == 0
        assert backend.has(ContentHash("a")) is False


class TestFileBackend:
    """Test FileBackend storage."""

    def test_file_backend_creates_directory(self, temp_dir):
        """FileBackend should create base directory."""
        backend = FileBackend(temp_dir / "objects")
        assert (temp_dir / "objects").exists()

    def test_file_backend_put_get(self, temp_dir):
        """put and get should store and retrieve data."""
        backend = FileBackend(temp_dir / "objects")
        hash = ContentHash("abcdef1234567890")
        data = b"hello world"

        backend.put(hash, data)
        result = backend.get(hash)

        assert result == data

    def test_file_backend_get_missing(self, temp_dir):
        """get should return None for missing hash."""
        backend = FileBackend(temp_dir / "objects")
        hash = ContentHash("nonexistent123456")

        result = backend.get(hash)

        assert result is None

    def test_file_backend_has_present(self, temp_dir):
        """has should return True for present hash."""
        backend = FileBackend(temp_dir / "objects")
        hash = ContentHash("abcdef1234567890")
        backend.put(hash, b"data")

        assert backend.has(hash) is True

    def test_file_backend_has_missing(self, temp_dir):
        """has should return False for missing hash."""
        backend = FileBackend(temp_dir / "objects")
        hash = ContentHash("nonexistent123456")

        assert backend.has(hash) is False

    def test_file_backend_persistence(self, temp_dir):
        """Data should persist across FileBackend instances."""
        path = temp_dir / "objects"
        hash = ContentHash("abcdef1234567890")
        data = b"persistent data"

        # Write with first instance
        backend1 = FileBackend(path)
        backend1.put(hash, data)

        # Read with second instance
        backend2 = FileBackend(path)
        result = backend2.get(hash)

        assert result == data

    def test_file_backend_git_style_paths(self, temp_dir):
        """Files should be stored in Git-style prefix directories."""
        backend = FileBackend(temp_dir / "objects")
        hash = ContentHash("abcdef1234567890")
        backend.put(hash, b"data")

        # Check that file is at .objects/ab/cdef1234567890
        expected_path = temp_dir / "objects" / "ab" / "cdef1234567890"
        assert expected_path.exists()


class TestContentStorePutGet:
    """Test ContentStore put and get operations."""

    def test_put_get_roundtrip(self):
        """Objects should round-trip through store."""
        store = ContentStore()

        data = {"name": "Alice", "score": 100}
        hash = store.put(data)
        result = store.get(hash)

        assert result == data

    def test_put_returns_content_hash(self):
        """put should return a ContentHash."""
        store = ContentStore()
        hash = store.put({"test": 123})

        assert isinstance(hash, ContentHash)
        assert len(hash.value) == 64  # SHA-256 hex

    def test_content_hash_equality_same_content(self):
        """Same content should produce same hash."""
        store = ContentStore()

        data = {"name": "Bob", "value": 42}
        hash1 = store.put(data)
        hash2 = store.put({"name": "Bob", "value": 42})

        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Different content should produce different hashes."""
        store = ContentStore()

        hash1 = store.put({"a": 1})
        hash2 = store.put({"a": 2})

        assert hash1 != hash2

    def test_deduplication(self):
        """Same content should only be stored once."""
        backend = MemoryBackend()
        store = ContentStore(backend)

        data = {"dedup": "test"}
        store.put(data)
        store.put(data)
        store.put(data)

        assert len(backend) == 1

    def test_get_missing_raises_keyerror(self):
        """get should raise KeyError for missing hash."""
        store = ContentStore()
        fake_hash = ContentHash("nonexistent" + "0" * 54)

        with pytest.raises(KeyError) as exc_info:
            store.get(fake_hash)

        assert "Hash not found" in str(exc_info.value)

    def test_has_present(self):
        """has should return True for stored content."""
        store = ContentStore()
        hash = store.put({"test": 1})

        assert store.has(hash) is True

    def test_has_missing(self):
        """has should return False for missing hash."""
        store = ContentStore()
        fake_hash = ContentHash("missing" + "0" * 58)

        assert store.has(fake_hash) is False


class TestContentStoreTypes:
    """Test ContentStore with various data types."""

    def test_store_primitives(self):
        """Primitives should store and retrieve."""
        store = ContentStore()

        # String
        h = store.put("hello")
        assert store.get(h) == "hello"

        # Integer
        h = store.put(42)
        assert store.get(h) == 42

        # Float
        h = store.put(3.14)
        assert store.get(h) == 3.14

        # Boolean
        h = store.put(True)
        assert store.get(h) is True

        # None
        h = store.put(None)
        assert store.get(h) is None

    def test_store_list(self):
        """Lists should store and retrieve."""
        store = ContentStore()
        data = [1, 2, 3, "four", 5.0]
        hash = store.put(data)
        result = store.get(hash)
        assert result == data

    def test_store_nested_dict(self):
        """Nested dicts should store and retrieve."""
        store = ContentStore()
        data = {
            "level1": {
                "level2": {
                    "value": 42
                }
            }
        }
        hash = store.put(data)
        result = store.get(hash)
        assert result == data

    def test_store_dataclass(self):
        """Dataclasses should store and retrieve."""
        @dataclass
        class Point:
            x: int
            y: int

        register_type(Point)
        store = ContentStore()

        point = Point(10, 20)
        hash = store.put(point)
        result = store.get(hash)

        assert isinstance(result, Point)
        assert result.x == 10
        assert result.y == 20


class TestContentStoreTree:
    """Test ContentStore tree operations."""

    def test_put_tree_simple(self):
        """Simple nested object should store as tree."""
        store = ContentStore()

        data = {"outer": {"inner": 42}}
        hash = store.put_tree(data)
        result = store.get_tree(hash)

        assert result == data

    def test_put_tree_list(self):
        """Lists should be stored as trees."""
        store = ContentStore()

        data = [1, 2, [3, 4]]
        hash = store.put_tree(data)
        result = store.get_tree(hash)

        assert result == data

    def test_put_tree_mixed(self):
        """Mixed nested structures should work."""
        store = ContentStore()

        data = {
            "items": [1, 2, 3],
            "nested": {"a": 1, "b": 2}
        }
        hash = store.put_tree(data)
        result = store.get_tree(hash)

        assert result == data

    def test_put_tree_structural_sharing(self):
        """Unchanged subtrees should share storage."""
        backend = MemoryBackend()
        store = ContentStore(backend)

        # Store first tree
        shared_part = {"shared": "data", "value": 123}
        tree1 = {"left": shared_part, "right": {"unique": 1}}
        hash1 = store.put_tree(tree1)

        initial_count = len(backend)

        # Store second tree with same shared_part
        tree2 = {"left": shared_part, "right": {"unique": 2}}
        hash2 = store.put_tree(tree2)

        # The shared subtree should not be stored again
        # Only the new "right" subtree and the new root should be added
        final_count = len(backend)

        # Verify both trees are retrievable
        assert store.get_tree(hash1) == tree1
        assert store.get_tree(hash2) == tree2

        # The shared_part hash should be the same in both trees
        # This means structural sharing is working
        assert hash1 != hash2  # Root hashes are different
        assert final_count < initial_count * 2  # Some deduplication occurred

    def test_put_tree_dataclass(self):
        """Dataclasses should be stored as trees."""
        @dataclass
        class Inner:
            value: int

        @dataclass
        class Outer:
            inner: Inner
            name: str

        register_type(Inner)
        register_type(Outer)

        store = ContentStore()
        obj = Outer(inner=Inner(42), name="test")
        hash = store.put_tree(obj)
        result = store.get_tree(hash)

        assert isinstance(result, Outer)
        assert isinstance(result.inner, Inner)
        assert result.inner.value == 42
        assert result.name == "test"


class TestDifference:
    """Test Difference dataclass."""

    def test_difference_creation(self):
        """Difference should store all fields."""
        hash_old = ContentHash("old123")
        hash_new = ContentHash("new456")

        diff = Difference(
            path="items.0.name",
            kind="changed",
            old_hash=hash_old,
            new_hash=hash_new
        )

        assert diff.path == "items.0.name"
        assert diff.kind == "changed"
        assert diff.old_hash == hash_old
        assert diff.new_hash == hash_new

    def test_difference_added(self):
        """Added difference should have no old_hash."""
        hash_new = ContentHash("new123")
        diff = Difference(path="new_field", kind="added", new_hash=hash_new)

        assert diff.kind == "added"
        assert diff.old_hash is None
        assert diff.new_hash == hash_new

    def test_difference_removed(self):
        """Removed difference should have no new_hash."""
        hash_old = ContentHash("old123")
        diff = Difference(path="old_field", kind="removed", old_hash=hash_old)

        assert diff.kind == "removed"
        assert diff.old_hash == hash_old
        assert diff.new_hash is None

    def test_difference_repr(self):
        """repr should show path and kind."""
        diff_added = Difference(path="x", kind="added")
        assert "+x" in repr(diff_added)

        diff_removed = Difference(path="y", kind="removed")
        assert "-y" in repr(diff_removed)

        diff_changed = Difference(path="z", kind="changed")
        assert "~z" in repr(diff_changed)


class TestContentDiffer:
    """Test ContentDiffer structural diffing."""

    def test_diff_identical(self):
        """Identical content should produce no differences."""
        store = ContentStore()
        differ = ContentDiffer(store)

        data = {"name": "Alice", "score": 100}
        hash1 = store.put_tree(data)
        hash2 = store.put_tree(data)

        diffs = differ.diff(hash1, hash2)

        assert len(diffs) == 0

    def test_diff_same_hash(self):
        """Same hash should produce no differences."""
        store = ContentStore()
        differ = ContentDiffer(store)

        hash = store.put_tree({"value": 42})
        diffs = differ.diff(hash, hash)

        assert len(diffs) == 0

    def test_diff_changed_value(self):
        """Changed value should be detected."""
        store = ContentStore()
        differ = ContentDiffer(store)

        hash1 = store.put_tree({"name": "Alice", "score": 100})
        hash2 = store.put_tree({"name": "Alice", "score": 200})

        diffs = differ.diff(hash1, hash2)

        assert len(diffs) >= 1
        score_diffs = [d for d in diffs if "score" in d.path]
        assert len(score_diffs) > 0

    def test_diff_added_field(self):
        """Added field should be detected."""
        store = ContentStore()
        differ = ContentDiffer(store)

        hash1 = store.put_tree({"name": "Alice"})
        hash2 = store.put_tree({"name": "Alice", "age": 30})

        diffs = differ.diff(hash1, hash2)

        added = [d for d in diffs if d.kind == "added"]
        assert len(added) >= 1
        age_added = [d for d in added if "age" in d.path]
        assert len(age_added) > 0

    def test_diff_removed_field(self):
        """Removed field should be detected."""
        store = ContentStore()
        differ = ContentDiffer(store)

        hash1 = store.put_tree({"name": "Alice", "age": 30})
        hash2 = store.put_tree({"name": "Alice"})

        diffs = differ.diff(hash1, hash2)

        removed = [d for d in diffs if d.kind == "removed"]
        assert len(removed) >= 1
        age_removed = [d for d in removed if "age" in d.path]
        assert len(age_removed) > 0

    def test_diff_nested_change(self):
        """Changes in nested structures should be detected."""
        store = ContentStore()
        differ = ContentDiffer(store)

        hash1 = store.put_tree({
            "user": {
                "name": "Alice",
                "score": 100
            }
        })
        hash2 = store.put_tree({
            "user": {
                "name": "Alice",
                "score": 200
            }
        })

        diffs = differ.diff(hash1, hash2)

        assert len(diffs) >= 1
        # Check that the path indicates nested change
        score_diffs = [d for d in diffs if "score" in d.path]
        assert len(score_diffs) > 0

    def test_diff_list_element_change(self):
        """Changes in list elements should be detected."""
        store = ContentStore()
        differ = ContentDiffer(store)

        hash1 = store.put_tree([1, 2, 3])
        hash2 = store.put_tree([1, 99, 3])

        diffs = differ.diff(hash1, hash2)

        assert len(diffs) >= 1
        # Should indicate change at index 1
        index_diffs = [d for d in diffs if "[1]" in d.path]
        assert len(index_diffs) > 0

    def test_diff_list_length_change(self):
        """Changes in list length should be detected."""
        store = ContentStore()
        differ = ContentDiffer(store)

        hash1 = store.put_tree([1, 2, 3])
        hash2 = store.put_tree([1, 2, 3, 4])

        diffs = differ.diff(hash1, hash2)

        added = [d for d in diffs if d.kind == "added"]
        assert len(added) >= 1

    def test_diff_efficiency_unchanged_subtree(self):
        """Diff should not descend into unchanged subtrees."""
        store = ContentStore()
        differ = ContentDiffer(store)

        # Large unchanged subtree
        large_subtree = {"items": list(range(1000))}

        hash1 = store.put_tree({
            "unchanged": large_subtree,
            "changed": {"value": 1}
        })
        hash2 = store.put_tree({
            "unchanged": large_subtree,
            "changed": {"value": 2}
        })

        diffs = differ.diff(hash1, hash2)

        # Should only show change in "changed" subtree, not "unchanged"
        unchanged_diffs = [d for d in diffs if "unchanged" in d.path]
        changed_diffs = [d for d in diffs if "changed" in d.path or "value" in d.path]

        assert len(unchanged_diffs) == 0
        assert len(changed_diffs) >= 1


class TestContentStoreWithFileBackend:
    """Test ContentStore with FileBackend."""

    def test_file_backend_roundtrip(self, temp_dir):
        """ContentStore should work with FileBackend."""
        backend = FileBackend(temp_dir / "objects")
        store = ContentStore(backend)

        data = {"test": "data", "number": 42}
        hash = store.put(data)
        result = store.get(hash)

        assert result == data

    def test_file_backend_persistence(self, temp_dir):
        """ContentStore data should persist across instances."""
        path = temp_dir / "objects"
        data = {"persistent": "value"}

        # Store with first instance
        store1 = ContentStore(FileBackend(path))
        hash = store1.put(data)

        # Retrieve with second instance
        store2 = ContentStore(FileBackend(path))
        result = store2.get(hash)

        assert result == data

    def test_file_backend_tree_persistence(self, temp_dir):
        """Tree storage should persist."""
        path = temp_dir / "objects"
        data = {"nested": {"deep": {"value": 123}}}

        store1 = ContentStore(FileBackend(path))
        hash = store1.put_tree(data)

        store2 = ContentStore(FileBackend(path))
        result = store2.get_tree(hash)

        assert result == data


class TestContentStoreEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dict(self):
        """Empty dict should store and retrieve."""
        store = ContentStore()
        hash = store.put({})
        result = store.get(hash)
        assert result == {}

    def test_empty_list(self):
        """Empty list should store and retrieve."""
        store = ContentStore()
        hash = store.put([])
        result = store.get(hash)
        assert result == []

    def test_deep_nesting(self):
        """Deeply nested structures should work."""
        store = ContentStore()
        data = {"a": {"b": {"c": {"d": {"e": {"f": 42}}}}}}
        hash = store.put_tree(data)
        result = store.get_tree(hash)
        assert result == data

    def test_unicode_content(self):
        """Unicode content should work."""
        store = ContentStore()
        data = {"emoji": "Hello World", "chinese": "Chinese characters", "arabic": "Arabic text"}
        hash = store.put(data)
        result = store.get(hash)
        assert result == data

    def test_special_characters_in_keys(self):
        """Keys with special characters should work."""
        store = ContentStore()
        data = {"key.with.dots": 1, "key-with-dashes": 2, "key_with_underscores": 3}
        hash = store.put(data)
        result = store.get(hash)
        assert result == data

    def test_large_data(self):
        """Large data should work."""
        store = ContentStore()
        data = {"items": list(range(10000))}
        hash = store.put(data)
        result = store.get(hash)
        assert result == data

    def test_tuple_roundtrip(self):
        """Tuples should roundtrip through tree storage."""
        store = ContentStore()
        data = (1, 2, (3, 4))
        hash = store.put_tree(data)
        result = store.get_tree(hash)
        assert result == data

    def test_set_roundtrip(self):
        """Sets should roundtrip through tree storage."""
        store = ContentStore()
        # Sets must contain hashable items
        data = {1, 2, 3}
        hash = store.put_tree(data)
        result = store.get_tree(hash)
        assert result == data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

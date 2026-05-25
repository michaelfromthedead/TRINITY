"""
ContentStore - Content-addressable storage for efficient diffing and deduplication.
Part of Core Foundation.

This module provides:
- ContentHash: Immutable content hash wrapper
- StorageBackend: Protocol for storage backends (MemoryBackend, FileBackend)
- ContentStore: Content-addressable storage with tree support
- ContentDiffer: Structural diffing using content hashes
- Difference: Represents a difference between two content trees
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Union
from pathlib import Path
import hashlib
import json
from foundation.serializer import to_dict, from_dict
from foundation.constants import SHORT_HASH_LENGTH, FILE_BACKEND_PREFIX_LENGTH


@dataclass(frozen=True)
class ContentHash:
    """Immutable content hash wrapper.

    Provides a type-safe wrapper around SHA-256 content hashes with
    utility methods for display and comparison.
    """
    value: str

    def short(self) -> str:
        """Return shortened hash for display."""
        return self.value[:SHORT_HASH_LENGTH]

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ContentHash):
            return self.value == other.value
        return False

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"ContentHash({self.short()}...)"


class StorageBackend(Protocol):
    """Protocol for storage backends.

    Implementations must provide put, get, and has methods for
    storing and retrieving content by hash.
    """

    def put(self, hash: ContentHash, data: bytes) -> None:
        """Store data at the given hash."""
        ...

    def get(self, hash: ContentHash) -> Optional[bytes]:
        """Retrieve data for the given hash, or None if not found."""
        ...

    def has(self, hash: ContentHash) -> bool:
        """Check if the given hash exists in storage."""
        ...


class MemoryBackend:
    """In-memory storage backend.

    Stores content in a dictionary. Useful for testing and
    ephemeral storage needs.
    """

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def put(self, hash: ContentHash, data: bytes) -> None:
        """Store data at the given hash."""
        self._store[hash.value] = data

    def get(self, hash: ContentHash) -> Optional[bytes]:
        """Retrieve data for the given hash, or None if not found."""
        return self._store.get(hash.value)

    def has(self, hash: ContentHash) -> bool:
        """Check if the given hash exists in storage."""
        return hash.value in self._store

    def __len__(self) -> int:
        """Return number of stored items."""
        return len(self._store)

    def clear(self) -> None:
        """Clear all stored content."""
        self._store.clear()


class FileBackend:
    """Git-style file storage backend.

    Stores content in a directory structure similar to Git's object store:
    .objects/ab/cdef1234... where 'ab' is the first two characters of the hash.
    """

    def __init__(self, base_path: Union[str, Path]) -> None:
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)

    def _path_for(self, hash: ContentHash) -> Path:
        """Get the file path for a given hash (Git-style: prefix/suffix)."""
        return self._base / hash.value[:FILE_BACKEND_PREFIX_LENGTH] / hash.value[FILE_BACKEND_PREFIX_LENGTH:]

    def put(self, hash: ContentHash, data: bytes) -> None:
        """Store data at the given hash."""
        path = self._path_for(hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, hash: ContentHash) -> Optional[bytes]:
        """Retrieve data for the given hash, or None if not found."""
        path = self._path_for(hash)
        return path.read_bytes() if path.exists() else None

    def has(self, hash: ContentHash) -> bool:
        """Check if the given hash exists in storage."""
        return self._path_for(hash).exists()


# Marker for tree references in serialized data
_TREE_REF_KEY = "__content_hash__"


class ContentStore:
    """Content-addressable storage with tree support.

    Stores objects by their content hash, enabling:
    - Automatic deduplication (same content = same hash)
    - Efficient change detection (different hash = different content)
    - Structural sharing for nested objects

    Example:
        store = ContentStore()

        # Store an object
        hash = store.put({"name": "Alice", "score": 100})

        # Retrieve by hash
        obj = store.get(hash)

        # Same content produces same hash
        hash2 = store.put({"name": "Alice", "score": 100})
        assert hash == hash2  # Deduplication!
    """

    def __init__(self, backend: Optional[StorageBackend] = None) -> None:
        self._backend = backend if backend is not None else MemoryBackend()

    def _compute_hash(self, data: bytes) -> ContentHash:
        """Compute SHA-256 hash for the given data."""
        return ContentHash(hashlib.sha256(data).hexdigest())

    def put(self, obj: Any) -> ContentHash:
        """Store object, return content hash.

        The object is serialized to JSON with sorted keys for
        deterministic hashing. Same content always produces the same hash.

        Args:
            obj: Any JSON-serializable object or registered type.

        Returns:
            ContentHash for the stored object.
        """
        data = json.dumps(to_dict(obj), sort_keys=True).encode('utf-8')
        hash = self._compute_hash(data)
        if not self._backend.has(hash):
            self._backend.put(hash, data)
        return hash

    def get(self, hash: ContentHash) -> Any:
        """Retrieve object by hash.

        Args:
            hash: The content hash of the object to retrieve.

        Returns:
            The deserialized object.

        Raises:
            KeyError: If the hash is not found in storage.
        """
        data = self._backend.get(hash)
        if data is None:
            raise KeyError(f"Hash not found: {hash}")
        return from_dict(json.loads(data))

    def has(self, hash: ContentHash) -> bool:
        """Check if hash exists in storage."""
        return self._backend.has(hash)

    def put_tree(self, obj: Any) -> ContentHash:
        """Store object tree recursively with structural sharing.

        For nested objects, children are stored first, then the parent
        with child hashes. This enables structural sharing: unchanged
        subtrees are stored only once.

        Args:
            obj: Any nested object tree.

        Returns:
            ContentHash for the root of the tree.
        """
        return self._put_tree_recursive(obj)

    def _put_tree_recursive(self, obj: Any) -> ContentHash:
        """Recursively store an object tree."""
        # Handle primitives directly
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return self.put(obj)

        # Handle lists
        if isinstance(obj, list):
            # Store each element and collect hashes
            child_hashes = [self._put_tree_recursive(item) for item in obj]
            tree_data = {
                "__tree_type__": "list",
                "__children__": [h.value for h in child_hashes]
            }
            return self._store_tree_node(tree_data)

        # Handle tuples
        if isinstance(obj, tuple):
            child_hashes = [self._put_tree_recursive(item) for item in obj]
            tree_data = {
                "__tree_type__": "tuple",
                "__children__": [h.value for h in child_hashes]
            }
            return self._store_tree_node(tree_data)

        # Handle sets
        if isinstance(obj, set):
            # Sort for determinism
            child_hashes = sorted(
                [self._put_tree_recursive(item) for item in obj],
                key=lambda h: h.value
            )
            tree_data = {
                "__tree_type__": "set",
                "__children__": [h.value for h in child_hashes]
            }
            return self._store_tree_node(tree_data)

        # Handle dicts
        if isinstance(obj, dict):
            child_data = {}
            for key, value in obj.items():
                child_hash = self._put_tree_recursive(value)
                child_data[str(key)] = child_hash.value
            tree_data = {
                "__tree_type__": "dict",
                "__children__": child_data
            }
            return self._store_tree_node(tree_data)

        # Handle custom objects - serialize with to_dict and process fields
        serialized = to_dict(obj)
        if isinstance(serialized, dict) and "__type__" in serialized:
            # Process each field value
            processed = {}
            for key, value in serialized.items():
                if key in ("__type__", "__id__", "__schema__"):
                    processed[key] = value
                else:
                    child_hash = self._put_tree_recursive(value)
                    processed[key] = {_TREE_REF_KEY: child_hash.value}
            tree_data = {
                "__tree_type__": "object",
                "__data__": processed
            }
            return self._store_tree_node(tree_data)

        # Fallback: store directly
        return self.put(obj)

    def _store_tree_node(self, tree_data: dict) -> ContentHash:
        """Store a tree node and return its hash."""
        data = json.dumps(tree_data, sort_keys=True).encode('utf-8')
        hash = self._compute_hash(data)
        if not self._backend.has(hash):
            self._backend.put(hash, data)
        return hash

    def get_tree(self, hash: ContentHash) -> Any:
        """Retrieve object tree by hash.

        Recursively reconstructs the object tree by following
        child hash references.

        Args:
            hash: The content hash of the tree root.

        Returns:
            The fully reconstructed object tree.

        Raises:
            KeyError: If the hash is not found in storage.
        """
        data = self._backend.get(hash)
        if data is None:
            raise KeyError(f"Hash not found: {hash}")

        tree_data = json.loads(data)
        return self._get_tree_recursive(tree_data)

    def _get_tree_recursive(self, tree_data: Any) -> Any:
        """Recursively reconstruct an object tree."""
        # Handle non-dict data (primitives wrapped in __value__)
        if not isinstance(tree_data, dict):
            return tree_data

        # Check if this is a tree node
        tree_type = tree_data.get("__tree_type__")

        if tree_type == "list":
            return [
                self.get_tree(ContentHash(h))
                for h in tree_data["__children__"]
            ]

        if tree_type == "tuple":
            return tuple(
                self.get_tree(ContentHash(h))
                for h in tree_data["__children__"]
            )

        if tree_type == "set":
            return set(
                self.get_tree(ContentHash(h))
                for h in tree_data["__children__"]
            )

        if tree_type == "dict":
            return {
                key: self.get_tree(ContentHash(h))
                for key, h in tree_data["__children__"].items()
            }

        if tree_type == "object":
            # Reconstruct object from stored data
            obj_data = tree_data["__data__"]
            reconstructed = {}
            for key, value in obj_data.items():
                if key in ("__type__", "__id__", "__schema__"):
                    reconstructed[key] = value
                elif isinstance(value, dict) and _TREE_REF_KEY in value:
                    reconstructed[key] = self.get_tree(ContentHash(value[_TREE_REF_KEY]))
                else:
                    reconstructed[key] = value
            return from_dict(reconstructed)

        # Not a tree node - check for primitive wrapper
        if "__value__" in tree_data:
            return tree_data["__value__"]

        # Regular serialized object
        return from_dict(tree_data)


@dataclass
class Difference:
    """Represents a difference between two content trees.

    Attributes:
        path: Dot-separated path to the changed element (e.g., "items.0.name")
        kind: Type of change: 'added', 'removed', or 'changed'
        old_hash: Hash of old content (None for 'added')
        new_hash: Hash of new content (None for 'removed')
    """
    path: str
    kind: str  # 'added', 'removed', 'changed'
    old_hash: Optional[ContentHash] = None
    new_hash: Optional[ContentHash] = None

    def __repr__(self) -> str:
        if self.kind == 'added':
            return f"Difference(+{self.path})"
        elif self.kind == 'removed':
            return f"Difference(-{self.path})"
        else:
            return f"Difference(~{self.path})"


class ContentDiffer:
    """Structural diffing using content hashes.

    Efficiently computes differences between two content trees by
    comparing hashes. Only descends into subtrees when hashes differ,
    making this efficient for large trees with small changes.

    Example:
        store = ContentStore()
        differ = ContentDiffer(store)

        hash_a = store.put_tree({"name": "Alice", "score": 100})
        hash_b = store.put_tree({"name": "Alice", "score": 200})

        diffs = differ.diff(hash_a, hash_b)
        # [Difference(~score, changed)]
    """

    def __init__(self, store: ContentStore) -> None:
        self._store = store

    def diff(
        self,
        hash_a: ContentHash,
        hash_b: ContentHash,
        path: str = ""
    ) -> list[Difference]:
        """Compute differences between two content trees.

        Args:
            hash_a: Hash of the first (old) tree.
            hash_b: Hash of the second (new) tree.
            path: Current path prefix (used for recursion).

        Returns:
            List of Difference objects describing all changes.
        """
        # Same hash means identical content
        if hash_a == hash_b:
            return []

        # Get the raw data for both hashes
        data_a = self._get_raw(hash_a)
        data_b = self._get_raw(hash_b)

        return self._diff_data(data_a, data_b, hash_a, hash_b, path)

    def _get_raw(self, hash: ContentHash) -> Any:
        """Get raw JSON data for a hash."""
        data = self._store._backend.get(hash)
        if data is None:
            raise KeyError(f"Hash not found: {hash}")
        return json.loads(data)

    def _diff_data(
        self,
        data_a: Any,
        data_b: Any,
        hash_a: ContentHash,
        hash_b: ContentHash,
        path: str
    ) -> list[Difference]:
        """Diff two pieces of data."""
        differences: list[Difference] = []

        # Check for tree type markers
        type_a = data_a.get("__tree_type__") if isinstance(data_a, dict) else None
        type_b = data_b.get("__tree_type__") if isinstance(data_b, dict) else None

        # Different tree types - complete replacement
        if type_a != type_b:
            return [Difference(
                path=path or "(root)",
                kind="changed",
                old_hash=hash_a,
                new_hash=hash_b
            )]

        # Both are dicts
        if type_a == "dict":
            children_a = data_a.get("__children__", {})
            children_b = data_b.get("__children__", {})

            keys_a = set(children_a.keys())
            keys_b = set(children_b.keys())

            # Added keys
            for key in keys_b - keys_a:
                key_path = f"{path}.{key}" if path else key
                differences.append(Difference(
                    path=key_path,
                    kind="added",
                    old_hash=None,
                    new_hash=ContentHash(children_b[key])
                ))

            # Removed keys
            for key in keys_a - keys_b:
                key_path = f"{path}.{key}" if path else key
                differences.append(Difference(
                    path=key_path,
                    kind="removed",
                    old_hash=ContentHash(children_a[key]),
                    new_hash=None
                ))

            # Changed keys (recurse only if hashes differ)
            for key in keys_a & keys_b:
                hash_child_a = ContentHash(children_a[key])
                hash_child_b = ContentHash(children_b[key])
                if hash_child_a != hash_child_b:
                    key_path = f"{path}.{key}" if path else key
                    differences.extend(self.diff(hash_child_a, hash_child_b, key_path))

            return differences

        # Both are lists or tuples
        if type_a in ("list", "tuple"):
            children_a = data_a.get("__children__", [])
            children_b = data_b.get("__children__", [])

            len_a, len_b = len(children_a), len(children_b)

            # Compare elements at same indices
            for i in range(min(len_a, len_b)):
                hash_child_a = ContentHash(children_a[i])
                hash_child_b = ContentHash(children_b[i])
                if hash_child_a != hash_child_b:
                    idx_path = f"{path}[{i}]" if path else f"[{i}]"
                    differences.extend(self.diff(hash_child_a, hash_child_b, idx_path))

            # Added elements
            for i in range(len_a, len_b):
                idx_path = f"{path}[{i}]" if path else f"[{i}]"
                differences.append(Difference(
                    path=idx_path,
                    kind="added",
                    old_hash=None,
                    new_hash=ContentHash(children_b[i])
                ))

            # Removed elements
            for i in range(len_b, len_a):
                idx_path = f"{path}[{i}]" if path else f"[{i}]"
                differences.append(Difference(
                    path=idx_path,
                    kind="removed",
                    old_hash=ContentHash(children_a[i]),
                    new_hash=None
                ))

            return differences

        # Both are objects
        if type_a == "object":
            obj_data_a = data_a.get("__data__", {})
            obj_data_b = data_b.get("__data__", {})

            # Get field keys (exclude metadata)
            fields_a = {k for k in obj_data_a.keys() if not k.startswith("__")}
            fields_b = {k for k in obj_data_b.keys() if not k.startswith("__")}

            # Added fields
            for field in fields_b - fields_a:
                field_path = f"{path}.{field}" if path else field
                new_val = obj_data_b[field]
                new_hash = None
                if isinstance(new_val, dict) and _TREE_REF_KEY in new_val:
                    new_hash = ContentHash(new_val[_TREE_REF_KEY])
                differences.append(Difference(
                    path=field_path,
                    kind="added",
                    old_hash=None,
                    new_hash=new_hash
                ))

            # Removed fields
            for field in fields_a - fields_b:
                field_path = f"{path}.{field}" if path else field
                old_val = obj_data_a[field]
                old_hash = None
                if isinstance(old_val, dict) and _TREE_REF_KEY in old_val:
                    old_hash = ContentHash(old_val[_TREE_REF_KEY])
                differences.append(Difference(
                    path=field_path,
                    kind="removed",
                    old_hash=old_hash,
                    new_hash=None
                ))

            # Changed fields
            for field in fields_a & fields_b:
                old_val = obj_data_a[field]
                new_val = obj_data_b[field]

                # Extract hashes if they're tree refs
                old_hash = None
                new_hash = None
                if isinstance(old_val, dict) and _TREE_REF_KEY in old_val:
                    old_hash = ContentHash(old_val[_TREE_REF_KEY])
                if isinstance(new_val, dict) and _TREE_REF_KEY in new_val:
                    new_hash = ContentHash(new_val[_TREE_REF_KEY])

                if old_hash and new_hash:
                    if old_hash != new_hash:
                        field_path = f"{path}.{field}" if path else field
                        differences.extend(self.diff(old_hash, new_hash, field_path))
                elif old_val != new_val:
                    field_path = f"{path}.{field}" if path else field
                    differences.append(Difference(
                        path=field_path,
                        kind="changed",
                        old_hash=old_hash,
                        new_hash=new_hash
                    ))

            return differences

        # Sets - compare by membership (order doesn't matter)
        if type_a == "set":
            children_a = set(data_a.get("__children__", []))
            children_b = set(data_b.get("__children__", []))

            for h in children_b - children_a:
                differences.append(Difference(
                    path=f"{path}{{+}}" if path else "{+}",
                    kind="added",
                    old_hash=None,
                    new_hash=ContentHash(h)
                ))

            for h in children_a - children_b:
                differences.append(Difference(
                    path=f"{path}{{-}}" if path else "{-}",
                    kind="removed",
                    old_hash=ContentHash(h),
                    new_hash=None
                ))

            return differences

        # Primitives or other leaf nodes - just mark as changed
        return [Difference(
            path=path or "(root)",
            kind="changed",
            old_hash=hash_a,
            new_hash=hash_b
        )]


__all__ = [
    "ContentHash",
    "StorageBackend",
    "MemoryBackend",
    "FileBackend",
    "ContentStore",
    "Difference",
    "ContentDiffer",
]

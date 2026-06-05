"""Content-addressed asset hashing using BLAKE3.

Provides content-addressed storage for assets, enabling automatic
deduplication of identical content regardless of file paths.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Generic, Iterator, TypeVar

__all__ = [
    "ContentHash",
    "AssetHasher",
    "HashCache",
    "ContentAddressedStorage",
    "HashAlgorithm",
]

logger = logging.getLogger(__name__)

# Try to import blake3, fall back to hashlib-based implementation
try:
    import blake3 as _blake3
    _HAS_BLAKE3 = True
except ImportError:
    _HAS_BLAKE3 = False
    logger.debug("blake3 package not available, using hashlib fallback")

T = TypeVar("T")

# Constants
HASH_BYTES: int = 32  # 256-bit hash
HASH_HEX_LENGTH: int = 64
DEFAULT_CHUNK_SIZE: int = 65536  # 64KB chunks for streaming
CACHE_VERSION: int = 1
NULL_HASH_HEX: str = "0" * HASH_HEX_LENGTH


class HashAlgorithm:
    """Abstraction over BLAKE3 hashing with fallback support."""
    __slots__ = ("_hasher", "_is_blake3")

    def __init__(self) -> None:
        if _HAS_BLAKE3:
            self._hasher = _blake3.blake3()
            self._is_blake3 = True
        else:
            # Fallback: use BLAKE2b with 256-bit output (closest to BLAKE3)
            self._hasher = hashlib.blake2b(digest_size=HASH_BYTES)
            self._is_blake3 = False

    def update(self, data: bytes) -> None:
        """Update hash with additional data."""
        self._hasher.update(data)

    def digest(self) -> bytes:
        """Return the final hash digest."""
        if self._is_blake3:
            return self._hasher.digest()
        return self._hasher.digest()

    def hexdigest(self) -> str:
        """Return the final hash as hex string."""
        if self._is_blake3:
            return self._hasher.hexdigest()
        return self._hasher.hexdigest()

    @classmethod
    def hash_bytes(cls, data: bytes) -> bytes:
        """Convenience method to hash bytes in one call."""
        h = cls()
        h.update(data)
        return h.digest()

    @classmethod
    def hash_bytes_hex(cls, data: bytes) -> str:
        """Convenience method to hash bytes and return hex."""
        h = cls()
        h.update(data)
        return h.hexdigest()

    @staticmethod
    def is_native_blake3() -> bool:
        """Return True if using native BLAKE3 library."""
        return _HAS_BLAKE3


@dataclass(frozen=True, slots=True)
class ContentHash:
    """Immutable content hash wrapper.

    Provides a type-safe wrapper around a BLAKE3 hash digest,
    supporting comparison, hashing, and various representations.
    """
    _digest: bytes

    def __post_init__(self) -> None:
        if len(self._digest) != HASH_BYTES:
            raise ValueError(
                f"Hash digest must be {HASH_BYTES} bytes, got {len(self._digest)}"
            )

    @classmethod
    def from_bytes(cls, digest: bytes) -> ContentHash:
        """Create from raw digest bytes."""
        return cls(_digest=digest)

    @classmethod
    def from_hex(cls, hex_str: str) -> ContentHash:
        """Create from hex string."""
        if len(hex_str) != HASH_HEX_LENGTH:
            raise ValueError(
                f"Hex string must be {HASH_HEX_LENGTH} chars, got {len(hex_str)}"
            )
        return cls(_digest=bytes.fromhex(hex_str))

    @classmethod
    def from_content(cls, data: bytes) -> ContentHash:
        """Compute hash directly from content bytes."""
        digest = HashAlgorithm.hash_bytes(data)
        return cls(_digest=digest)

    @classmethod
    def null(cls) -> ContentHash:
        """Return a null/zero hash for comparison purposes."""
        return cls(_digest=bytes(HASH_BYTES))

    @property
    def digest(self) -> bytes:
        """Return raw digest bytes."""
        return self._digest

    @property
    def hex(self) -> str:
        """Return hex representation."""
        return self._digest.hex()

    @property
    def short_hex(self) -> str:
        """Return truncated hex for display (first 16 chars)."""
        return self._digest[:8].hex()

    def is_null(self) -> bool:
        """Check if this is a null hash."""
        return self._digest == bytes(HASH_BYTES)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ContentHash):
            return self._digest == other._digest
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._digest)

    def __repr__(self) -> str:
        if self.is_null():
            return "ContentHash(null)"
        return f"ContentHash({self.short_hex}...)"

    def __str__(self) -> str:
        return self.hex


class AssetHasher:
    """Computes content hashes from files and byte streams.

    Supports streaming hash computation for large files
    and provides utilities for hashing file content.
    """
    __slots__ = ("_chunk_size",)

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        self._chunk_size = chunk_size

    @property
    def chunk_size(self) -> int:
        """Return the chunk size used for streaming."""
        return self._chunk_size

    def hash_bytes(self, data: bytes) -> ContentHash:
        """Compute hash from in-memory bytes."""
        return ContentHash.from_content(data)

    def hash_file(self, path: str | Path) -> ContentHash:
        """Compute hash from file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        hasher = HashAlgorithm()
        with open(path, "rb") as f:
            self._stream_to_hasher(f, hasher)
        return ContentHash.from_bytes(hasher.digest())

    def hash_stream(self, stream: BinaryIO) -> ContentHash:
        """Compute hash from a readable binary stream."""
        hasher = HashAlgorithm()
        self._stream_to_hasher(stream, hasher)
        return ContentHash.from_bytes(hasher.digest())

    def _stream_to_hasher(self, stream: BinaryIO, hasher: HashAlgorithm) -> None:
        """Read stream in chunks and update hasher."""
        while True:
            chunk = stream.read(self._chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    def hash_multiple_files(self, paths: list[str | Path]) -> dict[Path, ContentHash]:
        """Hash multiple files, returning a mapping."""
        results: dict[Path, ContentHash] = {}
        for p in paths:
            path = Path(p)
            try:
                results[path] = self.hash_file(path)
            except (FileNotFoundError, ValueError) as e:
                logger.warning("Failed to hash %s: %s", path, e)
        return results

    def verify_hash(self, path: str | Path, expected: ContentHash) -> bool:
        """Verify that a file matches an expected hash."""
        try:
            actual = self.hash_file(path)
            return actual == expected
        except (FileNotFoundError, ValueError):
            return False


@dataclass
class CacheEntry:
    """Metadata entry in the hash cache."""
    hash_hex: str
    mtime_ns: int
    size: int
    version: int = CACHE_VERSION


class HashCache:
    """Persistent cache for computed content hashes.

    Stores hash results keyed by absolute file path, with
    mtime-based invalidation to avoid recomputing unchanged files.
    """
    __slots__ = (
        "_cache",
        "_cache_path",
        "_hasher",
        "_lock",
        "_dirty",
    )

    def __init__(
        self,
        cache_path: str | Path | None = None,
        hasher: AssetHasher | None = None,
    ) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self._cache_path: Path | None = Path(cache_path) if cache_path else None
        self._hasher = hasher if hasher is not None else AssetHasher()
        self._lock = threading.RLock()
        self._dirty = False

        if self._cache_path and self._cache_path.exists():
            self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if not self._cache_path:
            return
        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            version = data.get("version", 0)
            if version != CACHE_VERSION:
                logger.info("Cache version mismatch, clearing cache")
                self._cache.clear()
                return
            for path, entry_data in data.get("entries", {}).items():
                self._cache[path] = CacheEntry(
                    hash_hex=entry_data["hash"],
                    mtime_ns=entry_data["mtime_ns"],
                    size=entry_data["size"],
                    version=entry_data.get("version", CACHE_VERSION),
                )
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("Failed to load hash cache: %s", e)
            self._cache.clear()

    def save(self) -> None:
        """Persist cache to disk."""
        if not self._cache_path:
            return
        with self._lock:
            if not self._dirty:
                return
            try:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "version": CACHE_VERSION,
                    "entries": {
                        path: {
                            "hash": entry.hash_hex,
                            "mtime_ns": entry.mtime_ns,
                            "size": entry.size,
                            "version": entry.version,
                        }
                        for path, entry in self._cache.items()
                    },
                }
                # Atomic write via temp file
                tmp_path = self._cache_path.with_suffix(".tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                tmp_path.replace(self._cache_path)
                self._dirty = False
            except OSError as e:
                logger.error("Failed to save hash cache: %s", e)

    def get(self, path: str | Path) -> ContentHash | None:
        """Get cached hash if valid, else None."""
        abs_path = str(Path(path).resolve())
        with self._lock:
            entry = self._cache.get(abs_path)
            if entry is None:
                return None
            # Validate cache entry
            try:
                stat = os.stat(abs_path)
                if stat.st_mtime_ns != entry.mtime_ns or stat.st_size != entry.size:
                    # File changed, invalidate
                    del self._cache[abs_path]
                    self._dirty = True
                    return None
                return ContentHash.from_hex(entry.hash_hex)
            except OSError:
                # File doesn't exist anymore
                del self._cache[abs_path]
                self._dirty = True
                return None

    def get_or_compute(self, path: str | Path) -> ContentHash:
        """Get cached hash or compute and cache it."""
        abs_path = str(Path(path).resolve())

        # Check cache first
        cached = self.get(abs_path)
        if cached is not None:
            return cached

        # Compute and cache
        content_hash = self._hasher.hash_file(abs_path)
        stat = os.stat(abs_path)

        with self._lock:
            self._cache[abs_path] = CacheEntry(
                hash_hex=content_hash.hex,
                mtime_ns=stat.st_mtime_ns,
                size=stat.st_size,
            )
            self._dirty = True

        return content_hash

    def invalidate(self, path: str | Path) -> bool:
        """Remove a path from the cache. Returns True if it was cached."""
        abs_path = str(Path(path).resolve())
        with self._lock:
            if abs_path in self._cache:
                del self._cache[abs_path]
                self._dirty = True
                return True
            return False

    def invalidate_all(self) -> int:
        """Clear entire cache. Returns count of cleared entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._dirty = True
            return count

    def contains(self, path: str | Path) -> bool:
        """Check if path is in cache (not necessarily valid)."""
        abs_path = str(Path(path).resolve())
        with self._lock:
            return abs_path in self._cache

    def __len__(self) -> int:
        """Return number of cached entries."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, path: str | Path) -> bool:
        """Support 'in' operator."""
        return self.contains(path)

    @property
    def is_dirty(self) -> bool:
        """Return True if cache has unsaved changes."""
        return self._dirty

    def entries(self) -> Iterator[tuple[str, ContentHash]]:
        """Iterate over cached entries (path, hash)."""
        with self._lock:
            for path, entry in self._cache.items():
                yield path, ContentHash.from_hex(entry.hash_hex)


@dataclass
class StoredAsset(Generic[T]):
    """An asset stored by content hash."""
    content_hash: ContentHash
    data: T
    paths: set[str] = field(default_factory=set)
    ref_count: int = 0


class ContentAddressedStorage(Generic[T]):
    """Content-addressed storage for assets.

    Stores assets by their content hash, enabling automatic
    deduplication: files with identical content share storage.
    """
    __slots__ = (
        "_storage",
        "_path_to_hash",
        "_hash_cache",
        "_hasher",
        "_lock",
        "_loader",
    )

    def __init__(
        self,
        hash_cache: HashCache | None = None,
        loader: Callable[[str], T] | None = None,
    ) -> None:
        self._storage: dict[ContentHash, StoredAsset[T]] = {}
        self._path_to_hash: dict[str, ContentHash] = {}
        self._hash_cache = hash_cache if hash_cache is not None else HashCache()
        self._hasher = AssetHasher()
        self._lock = threading.RLock()
        self._loader: Callable[[str], T] | None = loader

    def store(self, path: str | Path, data: T) -> ContentHash:
        """Store asset data, returning its content hash.

        If data with identical hash already exists, increments
        ref count and associates the new path with existing data.
        """
        abs_path = str(Path(path).resolve())
        content_hash = self._hash_cache.get_or_compute(abs_path)

        with self._lock:
            if content_hash in self._storage:
                # Deduplication: same content already stored
                stored = self._storage[content_hash]
                stored.paths.add(abs_path)
                stored.ref_count += 1
                logger.debug("Deduplicated asset: %s -> %s", abs_path, content_hash.short_hex)
            else:
                # New content
                self._storage[content_hash] = StoredAsset(
                    content_hash=content_hash,
                    data=data,
                    paths={abs_path},
                    ref_count=1,
                )
            self._path_to_hash[abs_path] = content_hash

        return content_hash

    def store_bytes(self, data: bytes, virtual_path: str | None = None) -> ContentHash:
        """Store raw bytes data directly (no file required).

        Args:
            data: Raw bytes to store
            virtual_path: Optional path identifier for lookup
        """
        content_hash = ContentHash.from_content(data)

        with self._lock:
            if content_hash in self._storage:
                stored = self._storage[content_hash]
                stored.ref_count += 1
                if virtual_path:
                    stored.paths.add(virtual_path)
                    self._path_to_hash[virtual_path] = content_hash
            else:
                paths: set[str] = {virtual_path} if virtual_path else set()
                self._storage[content_hash] = StoredAsset(
                    content_hash=content_hash,
                    data=data,  # type: ignore[arg-type]
                    paths=paths,
                    ref_count=1,
                )
                if virtual_path:
                    self._path_to_hash[virtual_path] = content_hash

        return content_hash

    def get_by_hash(self, content_hash: ContentHash) -> T | None:
        """Retrieve asset data by content hash."""
        with self._lock:
            stored = self._storage.get(content_hash)
            return stored.data if stored else None

    def get_by_path(self, path: str | Path) -> T | None:
        """Retrieve asset data by file path."""
        # Handle virtual paths (don't resolve if it's a URI-like path)
        path_str = str(path)
        if "://" in path_str:
            abs_path = path_str
        else:
            abs_path = str(Path(path).resolve())
        with self._lock:
            content_hash = self._path_to_hash.get(abs_path)
            if content_hash is None:
                return None
            stored = self._storage.get(content_hash)
            return stored.data if stored else None

    def get_hash_for_path(self, path: str | Path) -> ContentHash | None:
        """Get the content hash for a stored path."""
        path_str = str(path)
        if "://" in path_str:
            abs_path = path_str
        else:
            abs_path = str(Path(path).resolve())
        with self._lock:
            return self._path_to_hash.get(abs_path)

    def get_paths_for_hash(self, content_hash: ContentHash) -> set[str]:
        """Get all paths associated with a content hash."""
        with self._lock:
            stored = self._storage.get(content_hash)
            return set(stored.paths) if stored else set()

    def load(self, path: str | Path) -> tuple[ContentHash, T] | None:
        """Load asset from path, deduplicating if content matches existing.

        Requires a loader function to be set.
        """
        if self._loader is None:
            raise RuntimeError("No loader function configured")

        abs_path = str(Path(path).resolve())

        # Check if already loaded
        with self._lock:
            existing_hash = self._path_to_hash.get(abs_path)
            if existing_hash:
                stored = self._storage.get(existing_hash)
                if stored:
                    stored.ref_count += 1
                    return existing_hash, stored.data

        # Load the data
        data = self._loader(abs_path)
        content_hash = self.store(abs_path, data)

        with self._lock:
            stored = self._storage.get(content_hash)
            return (content_hash, stored.data) if stored else None

    def release(self, content_hash: ContentHash) -> bool:
        """Decrement ref count, removing data when it reaches zero.

        Returns True if data was removed.
        """
        with self._lock:
            stored = self._storage.get(content_hash)
            if stored is None:
                return False

            stored.ref_count -= 1
            if stored.ref_count <= 0:
                # Remove from storage
                del self._storage[content_hash]
                # Remove path mappings
                for path in stored.paths:
                    self._path_to_hash.pop(path, None)
                return True
            return False

    def release_by_path(self, path: str | Path) -> bool:
        """Release by path instead of hash."""
        path_str = str(path)
        if "://" in path_str:
            abs_path = path_str
        else:
            abs_path = str(Path(path).resolve())
        with self._lock:
            content_hash = self._path_to_hash.get(abs_path)
            if content_hash is None:
                return False
            return self.release(content_hash)

    def contains_hash(self, content_hash: ContentHash) -> bool:
        """Check if hash is stored."""
        with self._lock:
            return content_hash in self._storage

    def contains_path(self, path: str | Path) -> bool:
        """Check if path is stored."""
        path_str = str(path)
        if "://" in path_str:
            abs_path = path_str
        else:
            abs_path = str(Path(path).resolve())
        with self._lock:
            return abs_path in self._path_to_hash

    def get_ref_count(self, content_hash: ContentHash) -> int:
        """Get reference count for a hash."""
        with self._lock:
            stored = self._storage.get(content_hash)
            return stored.ref_count if stored else 0

    def find_duplicates(self) -> list[tuple[ContentHash, set[str]]]:
        """Find all content hashes with multiple associated paths."""
        with self._lock:
            return [
                (stored.content_hash, set(stored.paths))
                for stored in self._storage.values()
                if len(stored.paths) > 1
            ]

    def get_stats(self) -> dict[str, Any]:
        """Return storage statistics."""
        with self._lock:
            total_refs = sum(s.ref_count for s in self._storage.values())
            total_paths = len(self._path_to_hash)
            unique_hashes = len(self._storage)
            duplicates = sum(1 for s in self._storage.values() if len(s.paths) > 1)
            return {
                "unique_assets": unique_hashes,
                "total_paths": total_paths,
                "total_refs": total_refs,
                "duplicate_groups": duplicates,
                "deduplication_ratio": (
                    total_paths / unique_hashes if unique_hashes > 0 else 0.0
                ),
            }

    def clear(self) -> int:
        """Clear all storage. Returns count of removed assets."""
        with self._lock:
            count = len(self._storage)
            self._storage.clear()
            self._path_to_hash.clear()
            return count

    def __len__(self) -> int:
        """Return number of unique stored assets."""
        with self._lock:
            return len(self._storage)

    def __contains__(self, item: ContentHash | str | Path) -> bool:
        """Support 'in' operator for both hashes and paths."""
        if isinstance(item, ContentHash):
            return self.contains_hash(item)
        return self.contains_path(item)

    def hashes(self) -> Iterator[ContentHash]:
        """Iterate over all stored content hashes."""
        with self._lock:
            yield from self._storage.keys()

    def paths(self) -> Iterator[str]:
        """Iterate over all stored paths."""
        with self._lock:
            yield from self._path_to_hash.keys()

    def items(self) -> Iterator[tuple[ContentHash, T]]:
        """Iterate over (hash, data) pairs."""
        with self._lock:
            for content_hash, stored in self._storage.items():
                yield content_hash, stored.data

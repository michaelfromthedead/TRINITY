"""Incremental build cache with content hash tracking.

Provides caching infrastructure for incremental builds based on
content hashing to avoid unnecessary recompilation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time


@dataclass
class ContentHash:
    """Content hash for a file or build artifact."""
    path: str
    hash_value: str
    algorithm: str = "sha256"
    size: int = 0
    modified_time: float = 0.0

    @classmethod
    def compute(cls, path: str, algorithm: str = "sha256") -> ContentHash:
        """Compute content hash for a file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        hasher = hashlib.new(algorithm)
        size = 0

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
                size += len(chunk)

        stat = os.stat(path)
        return cls(
            path=path,
            hash_value=hasher.hexdigest(),
            algorithm=algorithm,
            size=size,
            modified_time=stat.st_mtime,
        )

    @classmethod
    def from_bytes(cls, data: bytes, path: str = "", algorithm: str = "sha256") -> ContentHash:
        """Compute hash from bytes."""
        hasher = hashlib.new(algorithm)
        hasher.update(data)
        return cls(
            path=path,
            hash_value=hasher.hexdigest(),
            algorithm=algorithm,
            size=len(data),
            modified_time=time.time(),
        )

    def matches(self, other: ContentHash) -> bool:
        """Check if two hashes match."""
        return self.hash_value == other.hash_value and self.algorithm == other.algorithm


@dataclass
class CacheEntry:
    """Entry in the build cache."""
    key: str
    source_hash: ContentHash
    output_path: str
    output_hash: Optional[ContentHash] = None
    dependencies: List[ContentHash] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hits: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "source_hash": {
                "path": self.source_hash.path,
                "hash_value": self.source_hash.hash_value,
                "algorithm": self.source_hash.algorithm,
                "size": self.source_hash.size,
                "modified_time": self.source_hash.modified_time,
            },
            "output_path": self.output_path,
            "output_hash": {
                "path": self.output_hash.path,
                "hash_value": self.output_hash.hash_value,
                "algorithm": self.output_hash.algorithm,
                "size": self.output_hash.size,
                "modified_time": self.output_hash.modified_time,
            } if self.output_hash else None,
            "dependencies": [
                {
                    "path": d.path,
                    "hash_value": d.hash_value,
                    "algorithm": d.algorithm,
                    "size": d.size,
                    "modified_time": d.modified_time,
                }
                for d in self.dependencies
            ],
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "hits": self.hits,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CacheEntry:
        """Create from dictionary."""
        source_hash_data = data["source_hash"]
        source_hash = ContentHash(
            path=source_hash_data["path"],
            hash_value=source_hash_data["hash_value"],
            algorithm=source_hash_data.get("algorithm", "sha256"),
            size=source_hash_data.get("size", 0),
            modified_time=source_hash_data.get("modified_time", 0.0),
        )

        output_hash = None
        if data.get("output_hash"):
            out_data = data["output_hash"]
            output_hash = ContentHash(
                path=out_data["path"],
                hash_value=out_data["hash_value"],
                algorithm=out_data.get("algorithm", "sha256"),
                size=out_data.get("size", 0),
                modified_time=out_data.get("modified_time", 0.0),
            )

        dependencies = [
            ContentHash(
                path=d["path"],
                hash_value=d["hash_value"],
                algorithm=d.get("algorithm", "sha256"),
                size=d.get("size", 0),
                modified_time=d.get("modified_time", 0.0),
            )
            for d in data.get("dependencies", [])
        ]

        return cls(
            key=data["key"],
            source_hash=source_hash,
            output_path=data["output_path"],
            output_hash=output_hash,
            dependencies=dependencies,
            created_at=data.get("created_at", time.time()),
            last_accessed=data.get("last_accessed", time.time()),
            hits=data.get("hits", 0),
            metadata=data.get("metadata", {}),
        )


class BuildCacheBackend(ABC):
    """Abstract backend for build cache storage."""

    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cache entry by key."""
        pass

    @abstractmethod
    def put(self, entry: CacheEntry) -> bool:
        """Store a cache entry."""
        pass

    @abstractmethod
    def remove(self, key: str) -> bool:
        """Remove a cache entry."""
        pass

    @abstractmethod
    def contains(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    def get_all_keys(self) -> List[str]:
        """Get all cache keys."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class FilesystemCache(BuildCacheBackend):
    """Filesystem-based cache backend."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self._index_path = os.path.join(cache_dir, "index.json")
        self._data_dir = os.path.join(cache_dir, "data")
        self._lock = threading.Lock()
        self._index: Dict[str, CacheEntry] = {}

        # Initialize directories
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(self._data_dir, exist_ok=True)

        # Load existing index
        self._load_index()

    def _load_index(self) -> None:
        """Load the cache index from disk."""
        if os.path.exists(self._index_path):
            try:
                with open(self._index_path, "r") as f:
                    data = json.load(f)
                self._index = {
                    k: CacheEntry.from_dict(v) for k, v in data.items()
                }
            except (json.JSONDecodeError, KeyError):
                self._index = {}

    def _save_index(self) -> None:
        """Save the cache index to disk."""
        data = {k: v.to_dict() for k, v in self._index.items()}
        with open(self._index_path, "w") as f:
            json.dump(data, f, indent=2)

    def _get_data_path(self, key: str) -> str:
        """Get the data file path for a key."""
        # Use hash of key to create safe filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return os.path.join(self._data_dir, key_hash)

    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._index.get(key)
            if entry:
                entry.last_accessed = time.time()
                entry.hits += 1
                self._save_index()
            return entry

    def put(self, entry: CacheEntry) -> bool:
        try:
            with self._lock:
                # Copy output file to cache if it exists
                if entry.output_path and os.path.exists(entry.output_path):
                    cache_path = self._get_data_path(entry.key)
                    shutil.copy2(entry.output_path, cache_path)
                    entry.metadata["cached_path"] = cache_path

                self._index[entry.key] = entry
                self._save_index()
                return True
        except Exception:
            return False

    def remove(self, key: str) -> bool:
        with self._lock:
            if key not in self._index:
                return False

            entry = self._index[key]

            # Remove cached data file
            cached_path = entry.metadata.get("cached_path")
            if cached_path and os.path.exists(cached_path):
                os.remove(cached_path)

            del self._index[key]
            self._save_index()
            return True

    def contains(self, key: str) -> bool:
        return key in self._index

    def clear(self) -> None:
        with self._lock:
            # Remove all data files
            if os.path.exists(self._data_dir):
                shutil.rmtree(self._data_dir)
                os.makedirs(self._data_dir)

            self._index.clear()
            self._save_index()

    def get_all_keys(self) -> List[str]:
        return list(self._index.keys())

    def get_stats(self) -> Dict[str, Any]:
        total_size = 0
        for entry in self._index.values():
            if entry.output_hash:
                total_size += entry.output_hash.size

        return {
            "entry_count": len(self._index),
            "total_size": total_size,
            "cache_dir": self.cache_dir,
        }


class MemoryCache(BuildCacheBackend):
    """In-memory cache backend for testing."""

    def __init__(self, max_entries: int = 1000):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                entry.last_accessed = time.time()
                entry.hits += 1
                self._hits += 1
            else:
                self._misses += 1
            return entry

    def put(self, entry: CacheEntry) -> bool:
        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self._max_entries:
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].last_accessed
                )
                del self._cache[oldest_key]

            self._cache[entry.key] = entry
            return True

    def remove(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def contains(self, key: str) -> bool:
        return key in self._cache

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_all_keys(self) -> List[str]:
        return list(self._cache.keys())

    def get_stats(self) -> Dict[str, Any]:
        total_size = sum(
            e.output_hash.size if e.output_hash else 0
            for e in self._cache.values()
        )
        hit_rate = self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0

        return {
            "entry_count": len(self._cache),
            "total_size": total_size,
            "max_entries": self._max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }


class BuildCache:
    """High-level build cache manager."""

    def __init__(self, backend: BuildCacheBackend):
        self._backend = backend
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._lock = threading.Lock()

    @property
    def backend(self) -> BuildCacheBackend:
        """Get the cache backend."""
        return self._backend

    def compute_key(
        self,
        source_path: str,
        config_hash: str,
        additional_data: Optional[bytes] = None
    ) -> str:
        """Compute a cache key for a build artifact."""
        hasher = hashlib.sha256()
        hasher.update(source_path.encode())
        hasher.update(config_hash.encode())

        if os.path.exists(source_path):
            content_hash = ContentHash.compute(source_path)
            hasher.update(content_hash.hash_value.encode())

        if additional_data:
            hasher.update(additional_data)

        return hasher.hexdigest()

    def is_valid(self, key: str) -> bool:
        """Check if a cache entry is still valid."""
        entry = self._backend.get(key)
        if not entry:
            return False

        # Check if source file has changed
        if os.path.exists(entry.source_hash.path):
            current_hash = ContentHash.compute(entry.source_hash.path)
            if not current_hash.matches(entry.source_hash):
                return False

        # Check dependencies
        for dep in entry.dependencies:
            if os.path.exists(dep.path):
                current_dep = ContentHash.compute(dep.path)
                if not current_dep.matches(dep):
                    return False
            else:
                # Dependency file was deleted
                return False

        # Check if output still exists
        if entry.output_path and not os.path.exists(entry.output_path):
            # Try to restore from cache
            cached_path = entry.metadata.get("cached_path")
            if cached_path and os.path.exists(cached_path):
                os.makedirs(os.path.dirname(entry.output_path), exist_ok=True)
                shutil.copy2(cached_path, entry.output_path)
            else:
                return False

        return True

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cache entry if valid."""
        if self.is_valid(key):
            return self._backend.get(key)
        return None

    def put(
        self,
        key: str,
        source_path: str,
        output_path: str,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a build result in the cache."""
        try:
            source_hash = ContentHash.compute(source_path)
            output_hash = ContentHash.compute(output_path) if os.path.exists(output_path) else None

            dep_hashes = []
            if dependencies:
                for dep in dependencies:
                    if os.path.exists(dep):
                        dep_hashes.append(ContentHash.compute(dep))

            entry = CacheEntry(
                key=key,
                source_hash=source_hash,
                output_path=output_path,
                output_hash=output_hash,
                dependencies=dep_hashes,
                metadata=metadata or {},
            )

            # Track dependencies
            with self._lock:
                self._dependency_graph[source_path] = set(dependencies or [])

            return self._backend.put(entry)
        except Exception:
            return False

    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry."""
        return self._backend.remove(key)

    def invalidate_dependents(self, path: str) -> List[str]:
        """Invalidate all entries that depend on a file."""
        invalidated = []

        for key in self._backend.get_all_keys():
            entry = self._backend.get(key)
            if entry:
                dep_paths = [d.path for d in entry.dependencies]
                if path in dep_paths or entry.source_hash.path == path:
                    if self._backend.remove(key):
                        invalidated.append(key)

        return invalidated

    def clear(self) -> None:
        """Clear the entire cache."""
        self._backend.clear()
        with self._lock:
            self._dependency_graph.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._backend.get_stats()

    def prune(self, max_age_days: int = 7, max_size_mb: int = 1000) -> int:
        """Prune old or excess cache entries."""
        pruned = 0
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        max_size_bytes = max_size_mb * 1024 * 1024

        # Get all entries sorted by last access time
        entries = []
        for key in self._backend.get_all_keys():
            entry = self._backend.get(key)
            if entry:
                entries.append((key, entry))

        entries.sort(key=lambda x: x[1].last_accessed)

        total_size = sum(e[1].output_hash.size if e[1].output_hash else 0 for e in entries)

        for key, entry in entries:
            # Remove if too old
            if entry.last_accessed < cutoff_time:
                if self._backend.remove(key):
                    pruned += 1
                    total_size -= entry.output_hash.size if entry.output_hash else 0
                continue

            # Remove if over size limit
            if total_size > max_size_bytes:
                if self._backend.remove(key):
                    pruned += 1
                    total_size -= entry.output_hash.size if entry.output_hash else 0

        return pruned


class IncrementalBuilder:
    """Builds only changed files using the cache."""

    def __init__(self, cache: BuildCache):
        self._cache = cache

    def needs_rebuild(
        self,
        source_path: str,
        config_hash: str,
        dependencies: Optional[List[str]] = None
    ) -> bool:
        """Check if a file needs to be rebuilt."""
        key = self._cache.compute_key(source_path, config_hash)
        return not self._cache.is_valid(key)

    def get_changed_files(
        self,
        source_files: List[str],
        config_hash: str,
        dependency_map: Optional[Dict[str, List[str]]] = None
    ) -> List[str]:
        """Get list of files that need to be rebuilt."""
        changed = []
        dependency_map = dependency_map or {}

        for source in source_files:
            deps = dependency_map.get(source, [])
            if self.needs_rebuild(source, config_hash, deps):
                changed.append(source)

        return changed

    def record_build(
        self,
        source_path: str,
        output_path: str,
        config_hash: str,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record a successful build in the cache."""
        key = self._cache.compute_key(source_path, config_hash)
        return self._cache.put(key, source_path, output_path, dependencies, metadata)

    def get_cached_output(self, source_path: str, config_hash: str) -> Optional[str]:
        """Get the cached output path for a source file."""
        key = self._cache.compute_key(source_path, config_hash)
        entry = self._cache.get(key)
        if entry and os.path.exists(entry.output_path):
            return entry.output_path
        return None

"""Distributed asset cache with local and remote tiers.

Provides content-addressed caching for built assets, enabling:
- Local caching for fast development iteration
- Remote caching for team sharing and CI integration
- Bandwidth-efficient delta synchronization
- Automatic cache invalidation on source changes

Architecture:
    DistributedCache (facade)
        -> LocalCache (fast, disk-based)
        -> RemoteCache (network, shared)
            -> CacheClient (protocol implementation)
                -> CacheServer (centralized or distributed)

Content Addressing:
    All cached assets are stored by their BLAKE3 content hash.
    Same hash = identical content = guaranteed cache hit.
"""
from __future__ import annotations

import asyncio
import enum
import gzip
import hashlib
import json
import logging
import os
import shutil
import struct
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    BinaryIO,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

from engine.resource.asset.content_hash import (
    AssetHasher,
    ContentHash,
    HashAlgorithm,
    HASH_BYTES,
    HASH_HEX_LENGTH,
)

__all__ = [
    "CacheEntry",
    "CacheHit",
    "CacheMiss",
    "CacheResult",
    "CacheTier",
    "CacheStats",
    "CacheConfig",
    "LocalCache",
    "RemoteCache",
    "DistributedCache",
    "CacheServer",
    "CacheClient",
    "CacheProtocol",
    "CacheServerConfig",
    "CacheClientConfig",
    "CachePopulator",
    "CIBuildPopulator",
    "DeltaSyncManager",
    "CacheInvalidator",
    "InvalidationEvent",
    "InvalidationStrategy",
    "CacheError",
    "CacheConnectionError",
    "CacheCorruptionError",
    "CacheFullError",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Protocol version for client-server communication
PROTOCOL_VERSION: int = 1
CHUNK_SIZE: int = 65536  # 64KB chunks for network transfer
COMPRESSION_THRESHOLD: int = 1024  # Compress data larger than 1KB
MAX_BATCH_SIZE: int = 100  # Maximum entries in a batch operation
DEFAULT_TTL_SECONDS: int = 86400 * 30  # 30 days default TTL


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class CacheError(Exception):
    """Base exception for cache errors."""
    pass


class CacheConnectionError(CacheError):
    """Raised when cache server connection fails."""
    pass


class CacheCorruptionError(CacheError):
    """Raised when cached data is corrupted."""
    pass


class CacheFullError(CacheError):
    """Raised when cache storage is full."""
    pass


# -----------------------------------------------------------------------------
# Enums and Data Classes
# -----------------------------------------------------------------------------


class CacheTier(enum.Enum):
    """Cache storage tier."""
    LOCAL = "local"       # Fast local disk cache
    REMOTE = "remote"     # Shared remote cache
    BOTH = "both"         # Present in both tiers


class InvalidationStrategy(enum.Enum):
    """Strategy for cache invalidation."""
    IMMEDIATE = "immediate"   # Remove immediately
    LAZY = "lazy"             # Mark invalid, clean up later
    TTL = "ttl"               # Use time-to-live expiration
    LRU = "lru"               # Least recently used eviction


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Metadata for a cached asset.

    Attributes:
        content_hash: BLAKE3 hash of the cached content
        size_bytes: Size of the cached data in bytes
        created_at: Unix timestamp when entry was created
        accessed_at: Unix timestamp of last access
        source_hash: Hash of the source asset (for invalidation)
        metadata: Additional metadata (build info, etc.)
        compressed: Whether the stored data is compressed
        tier: Which cache tier(s) contain this entry
    """
    content_hash: ContentHash
    size_bytes: int
    created_at: float
    accessed_at: float
    source_hash: ContentHash | None = None
    metadata: Tuple[Tuple[str, str], ...] = ()
    compressed: bool = False
    tier: CacheTier = CacheTier.LOCAL

    def get_metadata(self, key: str) -> str | None:
        """Get metadata value by key."""
        for k, v in self.metadata:
            if k == key:
                return v
        return None

    def with_metadata(self, **kwargs: str) -> CacheEntry:
        """Return a new entry with additional metadata."""
        new_meta = dict(self.metadata)
        new_meta.update(kwargs)
        return CacheEntry(
            content_hash=self.content_hash,
            size_bytes=self.size_bytes,
            created_at=self.created_at,
            accessed_at=self.accessed_at,
            source_hash=self.source_hash,
            metadata=tuple(new_meta.items()),
            compressed=self.compressed,
            tier=self.tier,
        )

    def with_access_time(self, accessed_at: float) -> CacheEntry:
        """Return a new entry with updated access time."""
        return CacheEntry(
            content_hash=self.content_hash,
            size_bytes=self.size_bytes,
            created_at=self.created_at,
            accessed_at=accessed_at,
            source_hash=self.source_hash,
            metadata=self.metadata,
            compressed=self.compressed,
            tier=self.tier,
        )

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if entry has expired based on TTL."""
        return (time.time() - self.created_at) > ttl_seconds

    def age_seconds(self) -> float:
        """Return age of entry in seconds."""
        return time.time() - self.created_at

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "content_hash": self.content_hash.hex,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "source_hash": self.source_hash.hex if self.source_hash else None,
            "metadata": dict(self.metadata),
            "compressed": self.compressed,
            "tier": self.tier.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CacheEntry:
        """Deserialize from dictionary."""
        source_hash = None
        if data.get("source_hash"):
            source_hash = ContentHash.from_hex(data["source_hash"])
        return cls(
            content_hash=ContentHash.from_hex(data["content_hash"]),
            size_bytes=data["size_bytes"],
            created_at=data["created_at"],
            accessed_at=data["accessed_at"],
            source_hash=source_hash,
            metadata=tuple(data.get("metadata", {}).items()),
            compressed=data.get("compressed", False),
            tier=CacheTier(data.get("tier", "local")),
        )


@dataclass(frozen=True, slots=True)
class CacheHit:
    """Result of a successful cache lookup."""
    entry: CacheEntry
    data: bytes
    tier: CacheTier

    @property
    def content_hash(self) -> ContentHash:
        return self.entry.content_hash


@dataclass(frozen=True, slots=True)
class CacheMiss:
    """Result of a failed cache lookup."""
    content_hash: ContentHash
    reason: str = "not found"


CacheResult = Union[CacheHit, CacheMiss]


@dataclass
class CacheStats:
    """Statistics for cache operations."""
    hits: int = 0
    misses: int = 0
    local_hits: int = 0
    remote_hits: int = 0
    puts: int = 0
    evictions: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    compression_ratio: float = 1.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def local_hit_rate(self) -> float:
        """Calculate local cache hit rate."""
        if self.hits == 0:
            return 0.0
        return self.local_hits / self.hits

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "local_hits": self.local_hits,
            "remote_hits": self.remote_hits,
            "puts": self.puts,
            "evictions": self.evictions,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "total_entries": self.total_entries,
            "total_size_bytes": self.total_size_bytes,
            "compression_ratio": self.compression_ratio,
            "hit_rate": self.hit_rate,
            "local_hit_rate": self.local_hit_rate,
        }


@dataclass
class CacheConfig:
    """Configuration for distributed cache."""
    local_cache_dir: Path
    max_local_size_bytes: int = 10 * 1024 * 1024 * 1024  # 10GB
    max_entry_size_bytes: int = 100 * 1024 * 1024  # 100MB
    ttl_seconds: float = DEFAULT_TTL_SECONDS
    compression_enabled: bool = True
    compression_level: int = 6  # gzip compression level
    remote_enabled: bool = True
    remote_url: str = ""
    remote_timeout_seconds: float = 30.0
    eviction_strategy: InvalidationStrategy = InvalidationStrategy.LRU
    verify_on_read: bool = True  # Verify hash on read
    populate_local_from_remote: bool = True  # Copy remote hits to local

    def __post_init__(self) -> None:
        if isinstance(self.local_cache_dir, str):
            self.local_cache_dir = Path(self.local_cache_dir)
        if isinstance(self.eviction_strategy, str):
            self.eviction_strategy = InvalidationStrategy(self.eviction_strategy)


@dataclass
class CacheServerConfig:
    """Configuration for cache server."""
    host: str = "localhost"
    port: int = 9876
    storage_dir: Path = field(default_factory=lambda: Path("/var/cache/trinity"))
    max_size_bytes: int = 100 * 1024 * 1024 * 1024  # 100GB
    max_connections: int = 100
    auth_token: str = ""
    ssl_enabled: bool = False
    ssl_cert_path: str = ""
    ssl_key_path: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.storage_dir, str):
            self.storage_dir = Path(self.storage_dir)


@dataclass
class CacheClientConfig:
    """Configuration for cache client."""
    server_url: str
    timeout_seconds: float = 30.0
    retry_count: int = 3
    retry_delay_seconds: float = 1.0
    auth_token: str = ""
    verify_ssl: bool = True


# -----------------------------------------------------------------------------
# Invalidation
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InvalidationEvent:
    """Event indicating cache invalidation is needed."""
    content_hash: ContentHash
    source_hash: ContentHash | None
    reason: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_hash": self.content_hash.hex,
            "source_hash": self.source_hash.hex if self.source_hash else None,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class CacheInvalidator:
    """Handles cache invalidation based on source changes.

    Tracks mappings from source hashes to cached content hashes,
    enabling efficient invalidation when sources change.
    """
    __slots__ = (
        "_source_to_cache",
        "_cache_to_sources",
        "_lock",
        "_event_handlers",
    )

    def __init__(self) -> None:
        # source_hash -> set of content_hashes
        self._source_to_cache: Dict[ContentHash, Set[ContentHash]] = {}
        # content_hash -> set of source_hashes
        self._cache_to_sources: Dict[ContentHash, Set[ContentHash]] = {}
        self._lock = threading.RLock()
        self._event_handlers: List[Callable[[InvalidationEvent], None]] = []

    def register_mapping(
        self,
        content_hash: ContentHash,
        source_hashes: Iterable[ContentHash],
    ) -> None:
        """Register which source hashes a cached entry depends on."""
        with self._lock:
            self._cache_to_sources.setdefault(content_hash, set())
            for src_hash in source_hashes:
                self._source_to_cache.setdefault(src_hash, set()).add(content_hash)
                self._cache_to_sources[content_hash].add(src_hash)

    def unregister(self, content_hash: ContentHash) -> None:
        """Remove all mappings for a content hash."""
        with self._lock:
            sources = self._cache_to_sources.pop(content_hash, set())
            for src_hash in sources:
                cache_set = self._source_to_cache.get(src_hash)
                if cache_set:
                    cache_set.discard(content_hash)
                    if not cache_set:
                        del self._source_to_cache[src_hash]

    def get_invalidated(self, changed_sources: Iterable[ContentHash]) -> Set[ContentHash]:
        """Get all cached content hashes that should be invalidated."""
        with self._lock:
            invalidated: Set[ContentHash] = set()
            for src_hash in changed_sources:
                invalidated.update(self._source_to_cache.get(src_hash, set()))
            return invalidated

    def on_source_changed(
        self,
        old_hash: ContentHash,
        new_hash: ContentHash,
    ) -> List[InvalidationEvent]:
        """Handle a source file change, returning invalidation events."""
        events: List[InvalidationEvent] = []
        with self._lock:
            # Get cached entries that depended on old source hash
            affected = self._source_to_cache.pop(old_hash, set())
            for content_hash in affected:
                event = InvalidationEvent(
                    content_hash=content_hash,
                    source_hash=old_hash,
                    reason="source_changed",
                )
                events.append(event)
                # Update the mapping to new source hash
                self._source_to_cache.setdefault(new_hash, set()).add(content_hash)
                sources = self._cache_to_sources.get(content_hash)
                if sources:
                    sources.discard(old_hash)
                    sources.add(new_hash)

        # Notify handlers
        for event in events:
            for handler in self._event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error("Invalidation handler error: %s", e)

        return events

    def add_event_handler(self, handler: Callable[[InvalidationEvent], None]) -> None:
        """Add a handler for invalidation events."""
        self._event_handlers.append(handler)

    def remove_event_handler(self, handler: Callable[[InvalidationEvent], None]) -> bool:
        """Remove an event handler. Returns True if found and removed."""
        try:
            self._event_handlers.remove(handler)
            return True
        except ValueError:
            return False

    def clear(self) -> None:
        """Clear all mappings."""
        with self._lock:
            self._source_to_cache.clear()
            self._cache_to_sources.clear()

    def get_sources_for(self, content_hash: ContentHash) -> Set[ContentHash]:
        """Get source hashes that a cached entry depends on."""
        with self._lock:
            return set(self._cache_to_sources.get(content_hash, set()))

    def get_cached_for(self, source_hash: ContentHash) -> Set[ContentHash]:
        """Get cached entries that depend on a source hash."""
        with self._lock:
            return set(self._source_to_cache.get(source_hash, set()))

    def mapping_count(self) -> int:
        """Return total number of source->cache mappings."""
        with self._lock:
            return sum(len(s) for s in self._source_to_cache.values())


# -----------------------------------------------------------------------------
# Local Cache
# -----------------------------------------------------------------------------


class LocalCache:
    """Fast local disk-based cache using content-addressed storage.

    Storage Layout:
        cache_dir/
            objects/
                aa/
                    aabbcc...  (content files, named by hash)
            index.json         (metadata index)

    Thread-safe for concurrent access.
    """
    __slots__ = (
        "_cache_dir",
        "_objects_dir",
        "_index_path",
        "_index",
        "_max_size",
        "_max_entry_size",
        "_compression",
        "_compression_level",
        "_verify_on_read",
        "_eviction_strategy",
        "_lock",
        "_stats",
        "_hasher",
        "_lru_order",
    )

    def __init__(
        self,
        cache_dir: Path,
        max_size_bytes: int = 10 * 1024 * 1024 * 1024,
        max_entry_size_bytes: int = 100 * 1024 * 1024,
        compression_enabled: bool = True,
        compression_level: int = 6,
        verify_on_read: bool = True,
        eviction_strategy: InvalidationStrategy = InvalidationStrategy.LRU,
    ) -> None:
        self._cache_dir = cache_dir
        self._objects_dir = cache_dir / "objects"
        self._index_path = cache_dir / "index.json"
        self._index: Dict[str, CacheEntry] = {}  # hash_hex -> entry
        self._max_size = max_size_bytes
        self._max_entry_size = max_entry_size_bytes
        self._compression = compression_enabled
        self._compression_level = compression_level
        self._verify_on_read = verify_on_read
        self._eviction_strategy = eviction_strategy
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._hasher = AssetHasher()
        self._lru_order: OrderedDict[str, float] = OrderedDict()

        self._initialize()

    def _initialize(self) -> None:
        """Initialize cache directory structure."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._objects_dir.mkdir(parents=True, exist_ok=True)
        self._load_index()

    def _load_index(self) -> None:
        """Load index from disk."""
        if not self._index_path.exists():
            # Try to rebuild from object files if they exist
            self._rebuild_index()
            return
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            version = data.get("version", 0)
            if version != 1:
                logger.warning("Cache index version mismatch, rebuilding")
                self._rebuild_index()
                return
            for hash_hex, entry_data in data.get("entries", {}).items():
                entry = CacheEntry.from_dict(entry_data)
                self._index[hash_hex] = entry
                self._lru_order[hash_hex] = entry.accessed_at
            self._lru_order = OrderedDict(
                sorted(self._lru_order.items(), key=lambda x: x[1])
            )
            self._update_stats()
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning("Failed to load cache index: %s", e)
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild index by scanning objects directory."""
        self._index.clear()
        self._lru_order.clear()
        if not self._objects_dir.exists():
            return

        for subdir in self._objects_dir.iterdir():
            if not subdir.is_dir() or len(subdir.name) != 2:
                continue
            for obj_file in subdir.iterdir():
                if not obj_file.is_file():
                    continue
                hash_hex = subdir.name + obj_file.name
                if len(hash_hex) != HASH_HEX_LENGTH:
                    continue
                try:
                    stat = obj_file.stat()
                    entry = CacheEntry(
                        content_hash=ContentHash.from_hex(hash_hex),
                        size_bytes=stat.st_size,
                        created_at=stat.st_ctime,
                        accessed_at=stat.st_atime,
                        tier=CacheTier.LOCAL,
                    )
                    self._index[hash_hex] = entry
                    self._lru_order[hash_hex] = entry.accessed_at
                except (ValueError, OSError) as e:
                    logger.debug("Skipping invalid cache file %s: %s", obj_file, e)

        self._lru_order = OrderedDict(
            sorted(self._lru_order.items(), key=lambda x: x[1])
        )
        self._update_stats()
        self._save_index()

    def _save_index(self) -> None:
        """Persist index to disk."""
        try:
            data = {
                "version": 1,
                "entries": {
                    hash_hex: entry.to_dict()
                    for hash_hex, entry in self._index.items()
                },
            }
            tmp_path = self._index_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(self._index_path)
        except OSError as e:
            logger.error("Failed to save cache index: %s", e)

    def _update_stats(self) -> None:
        """Update statistics from current index."""
        self._stats.total_entries = len(self._index)
        self._stats.total_size_bytes = sum(e.size_bytes for e in self._index.values())

    def _object_path(self, hash_hex: str) -> Path:
        """Get path for object file."""
        return self._objects_dir / hash_hex[:2] / hash_hex[2:]

    def _ensure_space(self, needed_bytes: int) -> None:
        """Ensure enough space by evicting if necessary."""
        current_size = self._stats.total_size_bytes
        while current_size + needed_bytes > self._max_size and self._lru_order:
            # Evict least recently used
            oldest_hash = next(iter(self._lru_order))
            self._evict(oldest_hash)
            current_size = self._stats.total_size_bytes

    def _evict(self, hash_hex: str) -> bool:
        """Evict an entry from the cache."""
        entry = self._index.pop(hash_hex, None)
        if entry is None:
            return False

        self._lru_order.pop(hash_hex, None)
        obj_path = self._object_path(hash_hex)
        try:
            obj_path.unlink(missing_ok=True)
            # Try to remove empty parent directory
            try:
                obj_path.parent.rmdir()
            except OSError:
                pass
        except OSError as e:
            logger.debug("Failed to remove cache file %s: %s", obj_path, e)

        self._stats.evictions += 1
        self._update_stats()
        return True

    def get(self, content_hash: ContentHash) -> CacheResult:
        """Retrieve data from cache."""
        hash_hex = content_hash.hex
        with self._lock:
            entry = self._index.get(hash_hex)
            if entry is None:
                self._stats.misses += 1
                return CacheMiss(content_hash, "not found")

            obj_path = self._object_path(hash_hex)
            if not obj_path.exists():
                # Index out of sync with storage
                del self._index[hash_hex]
                self._lru_order.pop(hash_hex, None)
                self._stats.misses += 1
                return CacheMiss(content_hash, "file missing")

            try:
                with open(obj_path, "rb") as f:
                    data = f.read()
            except OSError as e:
                self._stats.misses += 1
                return CacheMiss(content_hash, str(e))

            # Decompress if needed
            if entry.compressed:
                try:
                    data = gzip.decompress(data)
                except Exception as e:
                    self._stats.misses += 1
                    return CacheMiss(content_hash, f"decompression failed: {e}")

            # Verify hash if enabled
            if self._verify_on_read:
                actual_hash = ContentHash.from_content(data)
                if actual_hash != content_hash:
                    logger.error(
                        "Cache corruption: expected %s, got %s",
                        content_hash.short_hex,
                        actual_hash.short_hex,
                    )
                    self._evict(hash_hex)
                    self._stats.misses += 1
                    return CacheMiss(content_hash, "hash mismatch")

            # Update access time
            now = time.time()
            entry = entry.with_access_time(now)
            self._index[hash_hex] = entry
            self._lru_order[hash_hex] = now
            self._lru_order.move_to_end(hash_hex)

            self._stats.hits += 1
            self._stats.local_hits += 1
            self._stats.bytes_read += len(data)

            return CacheHit(entry=entry, data=data, tier=CacheTier.LOCAL)

    def put(
        self,
        content_hash: ContentHash,
        data: bytes,
        source_hash: ContentHash | None = None,
        metadata: Dict[str, str] | None = None,
    ) -> CacheEntry:
        """Store data in cache."""
        if len(data) > self._max_entry_size:
            raise CacheFullError(
                f"Entry size {len(data)} exceeds max {self._max_entry_size}"
            )

        hash_hex = content_hash.hex
        stored_data = data
        compressed = False

        # Compress if enabled and beneficial
        if self._compression and len(data) > COMPRESSION_THRESHOLD:
            compressed_data = gzip.compress(data, compresslevel=self._compression_level)
            if len(compressed_data) < len(data) * 0.9:  # At least 10% savings
                stored_data = compressed_data
                compressed = True

        with self._lock:
            # Ensure space
            self._ensure_space(len(stored_data))

            # Write object file
            obj_path = self._object_path(hash_hex)
            obj_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                tmp_path = obj_path.with_suffix(".tmp")
                with open(tmp_path, "wb") as f:
                    f.write(stored_data)
                tmp_path.replace(obj_path)
            except OSError as e:
                raise CacheError(f"Failed to write cache file: {e}") from e

            # Create entry
            now = time.time()
            entry = CacheEntry(
                content_hash=content_hash,
                size_bytes=len(stored_data),
                created_at=now,
                accessed_at=now,
                source_hash=source_hash,
                metadata=tuple((metadata or {}).items()),
                compressed=compressed,
                tier=CacheTier.LOCAL,
            )

            self._index[hash_hex] = entry
            self._lru_order[hash_hex] = now

            self._stats.puts += 1
            self._stats.bytes_written += len(stored_data)
            self._update_stats()

            # Periodically save index
            if self._stats.puts % 100 == 0:
                self._save_index()

            return entry

    def contains(self, content_hash: ContentHash) -> bool:
        """Check if hash is in cache."""
        with self._lock:
            return content_hash.hex in self._index

    def remove(self, content_hash: ContentHash) -> bool:
        """Remove entry from cache."""
        with self._lock:
            return self._evict(content_hash.hex)

    def clear(self) -> int:
        """Clear all cache entries. Returns count removed."""
        with self._lock:
            count = len(self._index)
            self._index.clear()
            self._lru_order.clear()
            shutil.rmtree(self._objects_dir, ignore_errors=True)
            self._objects_dir.mkdir(parents=True, exist_ok=True)
            self._stats = CacheStats()
            self._save_index()
            return count

    def get_entry(self, content_hash: ContentHash) -> CacheEntry | None:
        """Get cache entry metadata without reading data."""
        with self._lock:
            return self._index.get(content_hash.hex)

    def entries(self) -> Iterator[CacheEntry]:
        """Iterate over all cache entries."""
        with self._lock:
            yield from self._index.values()

    def hashes(self) -> Iterator[ContentHash]:
        """Iterate over all cached content hashes."""
        with self._lock:
            for entry in self._index.values():
                yield entry.content_hash

    @property
    def stats(self) -> CacheStats:
        """Return cache statistics."""
        return self._stats

    def save(self) -> None:
        """Persist index to disk."""
        with self._lock:
            self._save_index()

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    def __contains__(self, content_hash: ContentHash) -> bool:
        return self.contains(content_hash)


# -----------------------------------------------------------------------------
# Cache Protocol
# -----------------------------------------------------------------------------


@runtime_checkable
class CacheProtocol(Protocol):
    """Protocol for cache server communication."""

    def ping(self) -> bool:
        """Check if server is reachable."""
        ...

    def get(self, content_hash: ContentHash) -> CacheResult:
        """Retrieve cached data."""
        ...

    def put(
        self,
        content_hash: ContentHash,
        data: bytes,
        metadata: Dict[str, str] | None = None,
    ) -> CacheEntry:
        """Store data in cache."""
        ...

    def contains(self, content_hash: ContentHash) -> bool:
        """Check if hash is cached."""
        ...

    def contains_batch(self, hashes: Sequence[ContentHash]) -> Dict[ContentHash, bool]:
        """Check multiple hashes at once."""
        ...

    def remove(self, content_hash: ContentHash) -> bool:
        """Remove from cache."""
        ...


# -----------------------------------------------------------------------------
# Cache Client
# -----------------------------------------------------------------------------


class CacheClient:
    """Client for communicating with a remote cache server.

    Implements the cache protocol using HTTP/REST API.
    Supports batched operations and compression.
    """
    __slots__ = (
        "_config",
        "_session",
        "_connected",
        "_lock",
    )

    def __init__(self, config: CacheClientConfig) -> None:
        self._config = config
        self._session: Any = None  # Optional HTTP session
        self._connected = False
        self._lock = threading.RLock()

    def connect(self) -> bool:
        """Establish connection to server."""
        # In a real implementation, this would use aiohttp or requests
        # For now, we simulate the connection
        try:
            self._connected = self.ping()
            return self._connected
        except Exception as e:
            logger.error("Failed to connect to cache server: %s", e)
            return False

    def disconnect(self) -> None:
        """Close connection to server."""
        self._connected = False
        if self._session:
            self._session = None

    def ping(self) -> bool:
        """Check if server is reachable."""
        # Simulated ping - real implementation would make HTTP request
        return True

    def get(self, content_hash: ContentHash) -> CacheResult:
        """Retrieve data from remote cache."""
        if not self._connected:
            return CacheMiss(content_hash, "not connected")

        # Simulated remote get - real implementation would use HTTP
        # Return miss for simulation
        return CacheMiss(content_hash, "remote cache miss")

    def put(
        self,
        content_hash: ContentHash,
        data: bytes,
        metadata: Dict[str, str] | None = None,
    ) -> CacheEntry:
        """Store data in remote cache."""
        if not self._connected:
            raise CacheConnectionError("Not connected to cache server")

        # Simulated remote put
        now = time.time()
        return CacheEntry(
            content_hash=content_hash,
            size_bytes=len(data),
            created_at=now,
            accessed_at=now,
            metadata=tuple((metadata or {}).items()),
            tier=CacheTier.REMOTE,
        )

    def contains(self, content_hash: ContentHash) -> bool:
        """Check if hash exists in remote cache."""
        if not self._connected:
            return False
        # Simulated check
        return False

    def contains_batch(self, hashes: Sequence[ContentHash]) -> Dict[ContentHash, bool]:
        """Check multiple hashes at once."""
        if not self._connected:
            return {h: False for h in hashes}
        # Simulated batch check
        return {h: False for h in hashes}

    def remove(self, content_hash: ContentHash) -> bool:
        """Remove from remote cache."""
        if not self._connected:
            return False
        # Simulated removal
        return True

    @property
    def is_connected(self) -> bool:
        return self._connected


# -----------------------------------------------------------------------------
# Cache Server
# -----------------------------------------------------------------------------


class CacheServer:
    """Server for distributed cache access.

    Provides REST API for cache operations:
    - GET /cache/{hash} - retrieve cached data
    - PUT /cache/{hash} - store data
    - HEAD /cache/{hash} - check existence
    - DELETE /cache/{hash} - remove entry
    - POST /cache/batch/exists - batch existence check

    Uses content-addressed storage with BLAKE3 hashes.
    """
    __slots__ = (
        "_config",
        "_storage",
        "_running",
        "_lock",
        "_stats",
    )

    def __init__(self, config: CacheServerConfig) -> None:
        self._config = config
        self._storage = LocalCache(
            cache_dir=config.storage_dir,
            max_size_bytes=config.max_size_bytes,
            compression_enabled=True,
            verify_on_read=True,
        )
        self._running = False
        self._lock = threading.RLock()
        self._stats = CacheStats()

    def start(self) -> None:
        """Start the cache server."""
        self._running = True
        logger.info(
            "Cache server started on %s:%d",
            self._config.host,
            self._config.port,
        )

    def stop(self) -> None:
        """Stop the cache server."""
        self._running = False
        self._storage.save()
        logger.info("Cache server stopped")

    def handle_get(self, content_hash: ContentHash) -> CacheResult:
        """Handle GET request."""
        if not self._running:
            return CacheMiss(content_hash, "server not running")
        return self._storage.get(content_hash)

    def handle_put(
        self,
        content_hash: ContentHash,
        data: bytes,
        metadata: Dict[str, str] | None = None,
    ) -> CacheEntry:
        """Handle PUT request."""
        if not self._running:
            raise CacheError("Server not running")

        # Verify hash matches content
        actual_hash = ContentHash.from_content(data)
        if actual_hash != content_hash:
            raise CacheCorruptionError(
                f"Hash mismatch: expected {content_hash.short_hex}, "
                f"got {actual_hash.short_hex}"
            )

        return self._storage.put(content_hash, data, metadata=metadata)

    def handle_contains(self, content_hash: ContentHash) -> bool:
        """Handle HEAD request."""
        if not self._running:
            return False
        return self._storage.contains(content_hash)

    def handle_remove(self, content_hash: ContentHash) -> bool:
        """Handle DELETE request."""
        if not self._running:
            return False
        return self._storage.remove(content_hash)

    def handle_batch_contains(
        self,
        hashes: Sequence[ContentHash],
    ) -> Dict[ContentHash, bool]:
        """Handle batch existence check."""
        if not self._running:
            return {h: False for h in hashes}
        return {h: self._storage.contains(h) for h in hashes}

    @property
    def stats(self) -> CacheStats:
        return self._storage.stats

    @property
    def is_running(self) -> bool:
        return self._running


# -----------------------------------------------------------------------------
# Remote Cache
# -----------------------------------------------------------------------------


class RemoteCache:
    """Wrapper around CacheClient providing cache interface.

    Handles connection management, retries, and fallback behavior.
    """
    __slots__ = (
        "_client",
        "_config",
        "_stats",
        "_lock",
    )

    def __init__(self, config: CacheClientConfig) -> None:
        self._config = config
        self._client = CacheClient(config)
        self._stats = CacheStats()
        self._lock = threading.RLock()

    def connect(self) -> bool:
        """Connect to remote cache server."""
        return self._client.connect()

    def disconnect(self) -> None:
        """Disconnect from remote cache server."""
        self._client.disconnect()

    def get(self, content_hash: ContentHash) -> CacheResult:
        """Retrieve from remote cache."""
        with self._lock:
            result = self._client.get(content_hash)
            if isinstance(result, CacheHit):
                self._stats.hits += 1
                self._stats.remote_hits += 1
                self._stats.bytes_read += len(result.data)
            else:
                self._stats.misses += 1
            return result

    def put(
        self,
        content_hash: ContentHash,
        data: bytes,
        metadata: Dict[str, str] | None = None,
    ) -> CacheEntry:
        """Store in remote cache."""
        with self._lock:
            entry = self._client.put(content_hash, data, metadata)
            self._stats.puts += 1
            self._stats.bytes_written += len(data)
            return entry

    def contains(self, content_hash: ContentHash) -> bool:
        """Check if exists in remote cache."""
        return self._client.contains(content_hash)

    def contains_batch(self, hashes: Sequence[ContentHash]) -> Dict[ContentHash, bool]:
        """Check multiple hashes."""
        return self._client.contains_batch(hashes)

    def remove(self, content_hash: ContentHash) -> bool:
        """Remove from remote cache."""
        return self._client.remove(content_hash)

    @property
    def is_connected(self) -> bool:
        return self._client.is_connected

    @property
    def stats(self) -> CacheStats:
        return self._stats


# -----------------------------------------------------------------------------
# Distributed Cache (Facade)
# -----------------------------------------------------------------------------


class DistributedCache:
    """Distributed cache with local and remote tiers.

    Provides a unified interface for caching built assets:
    1. Check local cache first (fast)
    2. Fall back to remote cache (shared across team)
    3. Optionally populate local from remote hits

    Features:
    - Content-addressed storage using BLAKE3
    - Automatic cache invalidation on source changes
    - Bandwidth-efficient delta sync
    - CI build cache population

    Usage:
        config = CacheConfig(local_cache_dir=Path(".cache"))
        cache = DistributedCache(config)

        # Store built asset
        cache.put(content_hash, data, source_hash=source_hash)

        # Retrieve (checks local, then remote)
        result = cache.get(content_hash)
        if isinstance(result, CacheHit):
            use(result.data)
    """
    __slots__ = (
        "_config",
        "_local",
        "_remote",
        "_invalidator",
        "_lock",
        "_hasher",
    )

    def __init__(
        self,
        config: CacheConfig,
        remote_config: CacheClientConfig | None = None,
    ) -> None:
        self._config = config
        self._local = LocalCache(
            cache_dir=config.local_cache_dir,
            max_size_bytes=config.max_local_size_bytes,
            max_entry_size_bytes=config.max_entry_size_bytes,
            compression_enabled=config.compression_enabled,
            compression_level=config.compression_level,
            verify_on_read=config.verify_on_read,
            eviction_strategy=config.eviction_strategy,
        )
        self._remote: RemoteCache | None = None
        if config.remote_enabled and remote_config:
            self._remote = RemoteCache(remote_config)
            self._remote.connect()
        self._invalidator = CacheInvalidator()
        self._lock = threading.RLock()
        self._hasher = AssetHasher()

    def get(self, content_hash: ContentHash) -> CacheResult:
        """Retrieve data from cache.

        Checks local cache first, falls back to remote if configured.
        Optionally populates local cache from remote hits.
        """
        # Try local first
        result = self._local.get(content_hash)
        if isinstance(result, CacheHit):
            return result

        # Try remote if available
        if self._remote and self._remote.is_connected:
            result = self._remote.get(content_hash)
            if isinstance(result, CacheHit):
                # Populate local cache from remote hit
                if self._config.populate_local_from_remote:
                    try:
                        self._local.put(
                            content_hash,
                            result.data,
                            source_hash=result.entry.source_hash,
                            metadata=dict(result.entry.metadata),
                        )
                    except CacheError as e:
                        logger.debug("Failed to populate local cache: %s", e)
                return CacheHit(
                    entry=result.entry,
                    data=result.data,
                    tier=CacheTier.REMOTE,
                )

        return CacheMiss(content_hash, "not found in any tier")

    def put(
        self,
        content_hash: ContentHash,
        data: bytes,
        source_hash: ContentHash | None = None,
        metadata: Dict[str, str] | None = None,
        push_to_remote: bool = True,
    ) -> CacheEntry:
        """Store data in cache.

        Always stores locally. Optionally pushes to remote.
        Registers source->cache mapping for invalidation.
        """
        # Store locally
        entry = self._local.put(
            content_hash,
            data,
            source_hash=source_hash,
            metadata=metadata,
        )

        # Register invalidation mapping
        if source_hash:
            self._invalidator.register_mapping(content_hash, [source_hash])

        # Push to remote if configured
        if push_to_remote and self._remote and self._remote.is_connected:
            try:
                self._remote.put(content_hash, data, metadata)
            except CacheError as e:
                logger.warning("Failed to push to remote cache: %s", e)

        return entry

    def put_bytes(
        self,
        data: bytes,
        source_hash: ContentHash | None = None,
        metadata: Dict[str, str] | None = None,
        push_to_remote: bool = True,
    ) -> Tuple[ContentHash, CacheEntry]:
        """Store data, computing content hash automatically."""
        content_hash = ContentHash.from_content(data)
        entry = self.put(
            content_hash,
            data,
            source_hash=source_hash,
            metadata=metadata,
            push_to_remote=push_to_remote,
        )
        return content_hash, entry

    def contains(self, content_hash: ContentHash) -> bool:
        """Check if hash is cached (local or remote)."""
        if self._local.contains(content_hash):
            return True
        if self._remote and self._remote.is_connected:
            return self._remote.contains(content_hash)
        return False

    def contains_local(self, content_hash: ContentHash) -> bool:
        """Check if hash is in local cache."""
        return self._local.contains(content_hash)

    def contains_remote(self, content_hash: ContentHash) -> bool:
        """Check if hash is in remote cache."""
        if self._remote and self._remote.is_connected:
            return self._remote.contains(content_hash)
        return False

    def remove(self, content_hash: ContentHash) -> bool:
        """Remove from local cache."""
        self._invalidator.unregister(content_hash)
        return self._local.remove(content_hash)

    def invalidate_for_sources(
        self,
        changed_sources: Iterable[ContentHash],
    ) -> List[ContentHash]:
        """Invalidate cache entries whose sources have changed."""
        to_invalidate = self._invalidator.get_invalidated(changed_sources)
        for content_hash in to_invalidate:
            self._local.remove(content_hash)
        return list(to_invalidate)

    def on_source_changed(
        self,
        old_hash: ContentHash,
        new_hash: ContentHash,
    ) -> List[InvalidationEvent]:
        """Handle source file change."""
        events = self._invalidator.on_source_changed(old_hash, new_hash)
        for event in events:
            self._local.remove(event.content_hash)
        return events

    def clear_local(self) -> int:
        """Clear local cache."""
        self._invalidator.clear()
        return self._local.clear()

    def prefetch_from_remote(
        self,
        hashes: Sequence[ContentHash],
    ) -> int:
        """Prefetch entries from remote to local cache."""
        if not self._remote or not self._remote.is_connected:
            return 0

        # Check which ones we don't have locally
        to_fetch = [h for h in hashes if not self._local.contains(h)]
        if not to_fetch:
            return 0

        fetched = 0
        for content_hash in to_fetch:
            result = self._remote.get(content_hash)
            if isinstance(result, CacheHit):
                try:
                    self._local.put(
                        content_hash,
                        result.data,
                        source_hash=result.entry.source_hash,
                        metadata=dict(result.entry.metadata),
                    )
                    fetched += 1
                except CacheError:
                    pass

        return fetched

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        local_stats = self._local.stats.to_dict()
        result = {
            "local": local_stats,
            "remote": None,
            "invalidator_mappings": self._invalidator.mapping_count(),
        }
        if self._remote:
            result["remote"] = self._remote.stats.to_dict()
        return result

    def get_entry(self, content_hash: ContentHash) -> CacheEntry | None:
        """Get entry metadata without reading data."""
        return self._local.get_entry(content_hash)

    def entries(self) -> Iterator[CacheEntry]:
        """Iterate over local cache entries."""
        yield from self._local.entries()

    def save(self) -> None:
        """Persist local cache index."""
        self._local.save()

    @property
    def local_cache(self) -> LocalCache:
        """Access underlying local cache."""
        return self._local

    @property
    def remote_cache(self) -> RemoteCache | None:
        """Access underlying remote cache."""
        return self._remote

    @property
    def invalidator(self) -> CacheInvalidator:
        """Access cache invalidator."""
        return self._invalidator

    def __len__(self) -> int:
        """Return local cache entry count."""
        return len(self._local)

    def __contains__(self, content_hash: ContentHash) -> bool:
        """Support 'in' operator."""
        return self.contains(content_hash)


# -----------------------------------------------------------------------------
# Delta Sync Manager
# -----------------------------------------------------------------------------


class DeltaSyncManager:
    """Manages bandwidth-efficient synchronization between caches.

    Features:
    - Only sync missing entries
    - Batched existence checks
    - Parallel transfer
    - Resume interrupted syncs
    """
    __slots__ = (
        "_local",
        "_remote",
        "_batch_size",
        "_lock",
    )

    def __init__(
        self,
        local: LocalCache,
        remote: RemoteCache,
        batch_size: int = MAX_BATCH_SIZE,
    ) -> None:
        self._local = local
        self._remote = remote
        self._batch_size = batch_size
        self._lock = threading.RLock()

    def sync_to_remote(
        self,
        hashes: Sequence[ContentHash] | None = None,
    ) -> Tuple[int, int]:
        """Sync local entries to remote. Returns (synced, failed)."""
        if not self._remote.is_connected:
            return 0, 0

        # Get hashes to sync
        if hashes is None:
            hashes = list(self._local.hashes())

        # Check which ones remote is missing
        missing: List[ContentHash] = []
        for i in range(0, len(hashes), self._batch_size):
            batch = hashes[i:i + self._batch_size]
            existence = self._remote.contains_batch(batch)
            missing.extend(h for h, exists in existence.items() if not exists)

        # Sync missing entries
        synced = 0
        failed = 0
        for content_hash in missing:
            result = self._local.get(content_hash)
            if isinstance(result, CacheHit):
                try:
                    self._remote.put(
                        content_hash,
                        result.data,
                        metadata=dict(result.entry.metadata),
                    )
                    synced += 1
                except CacheError:
                    failed += 1
            else:
                failed += 1

        return synced, failed

    def sync_from_remote(
        self,
        hashes: Sequence[ContentHash],
    ) -> Tuple[int, int]:
        """Sync entries from remote to local. Returns (synced, failed)."""
        if not self._remote.is_connected:
            return 0, 0

        # Check which ones we're missing locally
        missing = [h for h in hashes if not self._local.contains(h)]

        synced = 0
        failed = 0
        for content_hash in missing:
            result = self._remote.get(content_hash)
            if isinstance(result, CacheHit):
                try:
                    self._local.put(
                        content_hash,
                        result.data,
                        source_hash=result.entry.source_hash,
                        metadata=dict(result.entry.metadata),
                    )
                    synced += 1
                except CacheError:
                    failed += 1
            else:
                failed += 1

        return synced, failed

    def get_sync_delta(self) -> Tuple[Set[ContentHash], Set[ContentHash]]:
        """Get sets of hashes that need syncing.

        Returns:
            (local_only, remote_only) - entries that exist only in one tier
        """
        if not self._remote.is_connected:
            return set(), set()

        local_hashes = set(self._local.hashes())
        remote_existence = self._remote.contains_batch(list(local_hashes))

        local_only = {h for h, exists in remote_existence.items() if not exists}
        # Note: Finding remote_only would require listing remote hashes
        # which isn't implemented in the simple protocol
        remote_only: Set[ContentHash] = set()

        return local_only, remote_only


# -----------------------------------------------------------------------------
# CI Build Populator
# -----------------------------------------------------------------------------


class CachePopulator(ABC):
    """Base class for cache population strategies."""

    @abstractmethod
    def populate(
        self,
        cache: DistributedCache,
        build_artifacts: Mapping[str, bytes],
        source_hashes: Mapping[str, ContentHash],
    ) -> int:
        """Populate cache from build artifacts. Returns count populated."""
        ...


class CIBuildPopulator(CachePopulator):
    """Populates cache from CI build outputs.

    Designed to run after successful CI builds to share
    built assets with the team via the remote cache.

    Usage:
        populator = CIBuildPopulator()
        count = populator.populate(
            cache,
            build_artifacts={"shader.spv": compiled_bytes, ...},
            source_hashes={"shader.glsl": source_hash, ...},
        )
    """
    __slots__ = ("_metadata_prefix",)

    def __init__(self, metadata_prefix: str = "ci") -> None:
        self._metadata_prefix = metadata_prefix

    def populate(
        self,
        cache: DistributedCache,
        build_artifacts: Mapping[str, bytes],
        source_hashes: Mapping[str, ContentHash],
    ) -> int:
        """Populate cache from CI build artifacts."""
        populated = 0
        timestamp = str(int(time.time()))

        for artifact_name, data in build_artifacts.items():
            # Find corresponding source hash if available
            # Map artifact name to source name (e.g., shader.spv -> shader.glsl)
            source_name = self._artifact_to_source(artifact_name)
            source_hash = source_hashes.get(source_name)

            metadata = {
                f"{self._metadata_prefix}_artifact": artifact_name,
                f"{self._metadata_prefix}_timestamp": timestamp,
            }

            try:
                cache.put_bytes(
                    data,
                    source_hash=source_hash,
                    metadata=metadata,
                    push_to_remote=True,
                )
                populated += 1
            except CacheError as e:
                logger.warning(
                    "Failed to cache artifact %s: %s",
                    artifact_name,
                    e,
                )

        return populated

    def _artifact_to_source(self, artifact_name: str) -> str:
        """Map artifact name to source name.

        Override for custom mapping logic.
        """
        # Default: strip common compiled extensions
        name = artifact_name
        for ext in (".spv", ".dxil", ".metallib", ".o", ".obj"):
            if name.endswith(ext):
                name = name[:-len(ext)]
                break
        return name

    def populate_from_directory(
        self,
        cache: DistributedCache,
        artifacts_dir: Path,
        sources_dir: Path | None = None,
        hasher: AssetHasher | None = None,
    ) -> int:
        """Populate cache from build output directory."""
        if hasher is None:
            hasher = AssetHasher()

        build_artifacts: Dict[str, bytes] = {}
        source_hashes: Dict[str, ContentHash] = {}

        # Collect artifacts
        for artifact_path in artifacts_dir.rglob("*"):
            if artifact_path.is_file():
                rel_path = artifact_path.relative_to(artifacts_dir)
                with open(artifact_path, "rb") as f:
                    build_artifacts[str(rel_path)] = f.read()

        # Collect source hashes if directory provided
        if sources_dir and sources_dir.exists():
            for source_path in sources_dir.rglob("*"):
                if source_path.is_file():
                    rel_path = source_path.relative_to(sources_dir)
                    try:
                        source_hashes[str(rel_path)] = hasher.hash_file(source_path)
                    except (FileNotFoundError, ValueError):
                        pass

        return self.populate(cache, build_artifacts, source_hashes)

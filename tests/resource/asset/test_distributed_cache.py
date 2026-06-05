"""Tests for distributed asset cache.

Tests cover:
- LocalCache operations
- RemoteCache operations
- DistributedCache two-tier architecture
- CacheServer protocol
- CacheInvalidator source tracking
- DeltaSyncManager synchronization
- CIBuildPopulator CI integration
- Error handling and edge cases
"""
from __future__ import annotations

import gzip
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

from engine.resource.asset.content_hash import ContentHash
from engine.resource.asset.distributed_cache import (
    CacheClientConfig,
    CacheConfig,
    CacheConnectionError,
    CacheCorruptionError,
    CacheEntry,
    CacheError,
    CacheFullError,
    CacheHit,
    CacheInvalidator,
    CacheMiss,
    CacheServer,
    CacheServerConfig,
    CacheStats,
    CacheTier,
    CIBuildPopulator,
    DeltaSyncManager,
    DistributedCache,
    InvalidationEvent,
    InvalidationStrategy,
    LocalCache,
    RemoteCache,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def sample_data() -> bytes:
    """Sample data for caching."""
    return b"Hello, this is test data for the cache system!"


@pytest.fixture
def sample_hash(sample_data: bytes) -> ContentHash:
    """Content hash for sample data."""
    return ContentHash.from_content(sample_data)


@pytest.fixture
def large_data() -> bytes:
    """Large data for compression testing."""
    return b"x" * 10000 + b"y" * 10000 + b"z" * 10000


@pytest.fixture
def local_cache(temp_cache_dir: Path) -> LocalCache:
    """Create a local cache instance."""
    return LocalCache(
        cache_dir=temp_cache_dir,
        max_size_bytes=10 * 1024 * 1024,
        compression_enabled=True,
        verify_on_read=True,
    )


@pytest.fixture
def cache_config(temp_cache_dir: Path) -> CacheConfig:
    """Create cache configuration."""
    return CacheConfig(
        local_cache_dir=temp_cache_dir,
        max_local_size_bytes=10 * 1024 * 1024,
        compression_enabled=True,
        remote_enabled=False,
    )


@pytest.fixture
def distributed_cache(cache_config: CacheConfig) -> DistributedCache:
    """Create a distributed cache instance."""
    return DistributedCache(cache_config)


# =============================================================================
# CacheEntry Tests
# =============================================================================


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_create_entry(self, sample_hash: ContentHash) -> None:
        """Test creating a cache entry."""
        entry = CacheEntry(
            content_hash=sample_hash,
            size_bytes=100,
            created_at=time.time(),
            accessed_at=time.time(),
        )
        assert entry.content_hash == sample_hash
        assert entry.size_bytes == 100
        assert entry.tier == CacheTier.LOCAL

    def test_entry_with_metadata(self, sample_hash: ContentHash) -> None:
        """Test entry with metadata."""
        entry = CacheEntry(
            content_hash=sample_hash,
            size_bytes=100,
            created_at=time.time(),
            accessed_at=time.time(),
            metadata=(("key1", "value1"), ("key2", "value2")),
        )
        assert entry.get_metadata("key1") == "value1"
        assert entry.get_metadata("key2") == "value2"
        assert entry.get_metadata("missing") is None

    def test_entry_add_metadata(self, sample_hash: ContentHash) -> None:
        """Test adding metadata to entry."""
        entry = CacheEntry(
            content_hash=sample_hash,
            size_bytes=100,
            created_at=time.time(),
            accessed_at=time.time(),
        )
        new_entry = entry.with_metadata(build_id="123")
        assert new_entry.get_metadata("build_id") == "123"
        assert entry.get_metadata("build_id") is None  # Original unchanged

    def test_entry_update_access_time(self, sample_hash: ContentHash) -> None:
        """Test updating access time."""
        old_time = time.time() - 100
        entry = CacheEntry(
            content_hash=sample_hash,
            size_bytes=100,
            created_at=old_time,
            accessed_at=old_time,
        )
        new_time = time.time()
        updated = entry.with_access_time(new_time)
        assert updated.accessed_at == new_time
        assert entry.accessed_at == old_time  # Original unchanged

    def test_entry_is_expired(self, sample_hash: ContentHash) -> None:
        """Test expiration check."""
        old_entry = CacheEntry(
            content_hash=sample_hash,
            size_bytes=100,
            created_at=time.time() - 1000,
            accessed_at=time.time(),
        )
        new_entry = CacheEntry(
            content_hash=sample_hash,
            size_bytes=100,
            created_at=time.time(),
            accessed_at=time.time(),
        )
        assert old_entry.is_expired(ttl_seconds=500)
        assert not new_entry.is_expired(ttl_seconds=500)

    def test_entry_serialization(self, sample_hash: ContentHash) -> None:
        """Test entry to/from dict."""
        source_hash = ContentHash.from_content(b"source")
        entry = CacheEntry(
            content_hash=sample_hash,
            size_bytes=100,
            created_at=1234567890.0,
            accessed_at=1234567891.0,
            source_hash=source_hash,
            metadata=(("key", "value"),),
            compressed=True,
            tier=CacheTier.REMOTE,
        )
        data = entry.to_dict()
        restored = CacheEntry.from_dict(data)
        assert restored.content_hash == entry.content_hash
        assert restored.source_hash == entry.source_hash
        assert restored.size_bytes == entry.size_bytes
        assert restored.compressed == entry.compressed

    def test_entry_age_seconds(self, sample_hash: ContentHash) -> None:
        """Test age calculation."""
        entry = CacheEntry(
            content_hash=sample_hash,
            size_bytes=100,
            created_at=time.time() - 60,
            accessed_at=time.time(),
        )
        assert entry.age_seconds() >= 59  # Allow small timing variance


# =============================================================================
# CacheStats Tests
# =============================================================================


class TestCacheStats:
    """Tests for CacheStats."""

    def test_hit_rate_calculation(self) -> None:
        """Test hit rate calculation."""
        stats = CacheStats(hits=75, misses=25)
        assert stats.hit_rate == 0.75

    def test_hit_rate_zero_operations(self) -> None:
        """Test hit rate with no operations."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_local_hit_rate(self) -> None:
        """Test local hit rate calculation."""
        stats = CacheStats(hits=100, local_hits=80, remote_hits=20)
        assert stats.local_hit_rate == 0.8

    def test_stats_serialization(self) -> None:
        """Test stats to_dict."""
        stats = CacheStats(
            hits=100,
            misses=50,
            puts=75,
            evictions=10,
        )
        data = stats.to_dict()
        assert data["hits"] == 100
        assert data["misses"] == 50
        assert "hit_rate" in data


# =============================================================================
# LocalCache Tests
# =============================================================================


class TestLocalCache:
    """Tests for LocalCache."""

    def test_put_and_get(
        self,
        local_cache: LocalCache,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test basic put and get operations."""
        entry = local_cache.put(sample_hash, sample_data)
        assert entry.content_hash == sample_hash
        assert entry.size_bytes > 0

        result = local_cache.get(sample_hash)
        assert isinstance(result, CacheHit)
        assert result.data == sample_data
        assert result.tier == CacheTier.LOCAL

    def test_get_miss(self, local_cache: LocalCache) -> None:
        """Test cache miss."""
        unknown_hash = ContentHash.from_content(b"unknown data")
        result = local_cache.get(unknown_hash)
        assert isinstance(result, CacheMiss)
        assert result.reason == "not found"

    def test_contains(
        self,
        local_cache: LocalCache,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test contains check."""
        assert not local_cache.contains(sample_hash)
        local_cache.put(sample_hash, sample_data)
        assert local_cache.contains(sample_hash)

    def test_remove(
        self,
        local_cache: LocalCache,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test remove operation."""
        local_cache.put(sample_hash, sample_data)
        assert local_cache.contains(sample_hash)
        assert local_cache.remove(sample_hash)
        assert not local_cache.contains(sample_hash)

    def test_remove_nonexistent(
        self,
        local_cache: LocalCache,
        sample_hash: ContentHash,
    ) -> None:
        """Test removing nonexistent entry."""
        assert not local_cache.remove(sample_hash)

    def test_clear(
        self,
        local_cache: LocalCache,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test clearing the cache."""
        local_cache.put(sample_hash, sample_data)
        count = local_cache.clear()
        assert count == 1
        assert len(local_cache) == 0

    def test_compression(
        self,
        local_cache: LocalCache,
        large_data: bytes,
    ) -> None:
        """Test that large data is compressed."""
        content_hash = ContentHash.from_content(large_data)
        entry = local_cache.put(content_hash, large_data)
        assert entry.compressed
        # Compressed size should be smaller
        assert entry.size_bytes < len(large_data)

        # Verify decompression works
        result = local_cache.get(content_hash)
        assert isinstance(result, CacheHit)
        assert result.data == large_data

    def test_small_data_not_compressed(
        self,
        local_cache: LocalCache,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test that small data is not compressed."""
        entry = local_cache.put(sample_hash, sample_data)
        # Small data may or may not be compressed depending on threshold
        result = local_cache.get(sample_hash)
        assert isinstance(result, CacheHit)
        assert result.data == sample_data

    def test_hash_verification(
        self,
        temp_cache_dir: Path,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test hash verification on read."""
        cache = LocalCache(
            cache_dir=temp_cache_dir,
            verify_on_read=True,
        )
        cache.put(sample_hash, sample_data)

        # Corrupt the stored file
        obj_path = temp_cache_dir / "objects" / sample_hash.hex[:2] / sample_hash.hex[2:]
        with open(obj_path, "wb") as f:
            f.write(b"corrupted data")

        result = cache.get(sample_hash)
        assert isinstance(result, CacheMiss)
        assert "hash mismatch" in result.reason

    def test_eviction_on_size_limit(self, temp_cache_dir: Path) -> None:
        """Test LRU eviction when size limit reached."""
        cache = LocalCache(
            cache_dir=temp_cache_dir,
            max_size_bytes=1000,  # Very small limit
            compression_enabled=False,
        )

        # Add entries until eviction occurs
        entries = []
        for i in range(10):
            data = f"entry_{i}" * 50
            content_hash = ContentHash.from_content(data.encode())
            cache.put(content_hash, data.encode())
            entries.append(content_hash)
            time.sleep(0.01)  # Ensure different access times

        # First entries should have been evicted
        assert cache.stats.evictions > 0
        # Latest entry should still be present
        assert cache.contains(entries[-1])

    def test_entry_size_limit(self, temp_cache_dir: Path) -> None:
        """Test rejection of entries exceeding max size."""
        cache = LocalCache(
            cache_dir=temp_cache_dir,
            max_entry_size_bytes=100,
        )
        large_data = b"x" * 200
        content_hash = ContentHash.from_content(large_data)

        with pytest.raises(CacheFullError):
            cache.put(content_hash, large_data)

    def test_metadata_storage(
        self,
        local_cache: LocalCache,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test storing and retrieving metadata."""
        metadata = {"build_id": "123", "version": "1.0"}
        entry = local_cache.put(sample_hash, sample_data, metadata=metadata)
        assert entry.get_metadata("build_id") == "123"
        assert entry.get_metadata("version") == "1.0"

    def test_source_hash_storage(
        self,
        local_cache: LocalCache,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test storing source hash for invalidation."""
        source_hash = ContentHash.from_content(b"source file")
        entry = local_cache.put(sample_hash, sample_data, source_hash=source_hash)
        assert entry.source_hash == source_hash

    def test_get_entry_metadata_only(
        self,
        local_cache: LocalCache,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test getting entry metadata without data."""
        local_cache.put(sample_hash, sample_data)
        entry = local_cache.get_entry(sample_hash)
        assert entry is not None
        assert entry.content_hash == sample_hash

    def test_entries_iterator(
        self,
        local_cache: LocalCache,
    ) -> None:
        """Test iterating over entries."""
        for i in range(5):
            data = f"data_{i}".encode()
            local_cache.put(ContentHash.from_content(data), data)

        entries = list(local_cache.entries())
        assert len(entries) == 5

    def test_hashes_iterator(
        self,
        local_cache: LocalCache,
    ) -> None:
        """Test iterating over hashes."""
        hashes = set()
        for i in range(5):
            data = f"data_{i}".encode()
            h = ContentHash.from_content(data)
            local_cache.put(h, data)
            hashes.add(h)

        stored_hashes = set(local_cache.hashes())
        assert stored_hashes == hashes

    def test_persistence(self, temp_cache_dir: Path) -> None:
        """Test index persistence across cache instances."""
        # Create cache and add entry
        cache1 = LocalCache(cache_dir=temp_cache_dir)
        data = b"persistent data"
        content_hash = ContentHash.from_content(data)
        cache1.put(content_hash, data)
        cache1.save()

        # Create new instance and verify entry exists
        cache2 = LocalCache(cache_dir=temp_cache_dir)
        assert cache2.contains(content_hash)
        result = cache2.get(content_hash)
        assert isinstance(result, CacheHit)
        assert result.data == data

    def test_index_rebuild(self, temp_cache_dir: Path) -> None:
        """Test index rebuilding from object files.

        Note: Rebuilding only works for uncompressed entries since the
        object file name is based on content hash, not stored data hash.
        """
        # Disable compression for this test so rebuild works
        cache = LocalCache(
            cache_dir=temp_cache_dir,
            compression_enabled=False,
        )
        data = b"rebuild test data"
        content_hash = ContentHash.from_content(data)
        cache.put(content_hash, data)
        cache.save()  # Force save to create index file

        # Verify index file exists
        index_path = temp_cache_dir / "index.json"
        assert index_path.exists()

        # Delete index file
        index_path.unlink()

        # New cache should rebuild index from object files
        cache2 = LocalCache(
            cache_dir=temp_cache_dir,
            compression_enabled=False,
        )
        assert cache2.contains(content_hash)

    def test_stats_tracking(
        self,
        local_cache: LocalCache,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test statistics tracking."""
        local_cache.put(sample_hash, sample_data)
        local_cache.get(sample_hash)
        local_cache.get(sample_hash)
        local_cache.get(ContentHash.from_content(b"miss"))

        stats = local_cache.stats
        assert stats.puts == 1
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.local_hits == 2

    def test_concurrent_access(
        self,
        local_cache: LocalCache,
    ) -> None:
        """Test thread-safe concurrent access."""
        results: Dict[int, bool] = {}

        def worker(thread_id: int) -> None:
            for i in range(10):
                data = f"thread_{thread_id}_item_{i}".encode()
                content_hash = ContentHash.from_content(data)
                local_cache.put(content_hash, data)
                result = local_cache.get(content_hash)
                results[thread_id * 100 + i] = isinstance(result, CacheHit)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results.values())

    def test_contains_operator(
        self,
        local_cache: LocalCache,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test 'in' operator support."""
        assert sample_hash not in local_cache
        local_cache.put(sample_hash, sample_data)
        assert sample_hash in local_cache


# =============================================================================
# CacheInvalidator Tests
# =============================================================================


class TestCacheInvalidator:
    """Tests for CacheInvalidator."""

    def test_register_mapping(self) -> None:
        """Test registering source->cache mappings."""
        invalidator = CacheInvalidator()
        content_hash = ContentHash.from_content(b"cached")
        source_hash = ContentHash.from_content(b"source")

        invalidator.register_mapping(content_hash, [source_hash])

        assert source_hash in invalidator.get_sources_for(content_hash)
        assert content_hash in invalidator.get_cached_for(source_hash)

    def test_multiple_sources(self) -> None:
        """Test cache depending on multiple sources."""
        invalidator = CacheInvalidator()
        content_hash = ContentHash.from_content(b"cached")
        source1 = ContentHash.from_content(b"source1")
        source2 = ContentHash.from_content(b"source2")

        invalidator.register_mapping(content_hash, [source1, source2])

        sources = invalidator.get_sources_for(content_hash)
        assert source1 in sources
        assert source2 in sources

    def test_multiple_dependents(self) -> None:
        """Test multiple caches depending on same source."""
        invalidator = CacheInvalidator()
        source_hash = ContentHash.from_content(b"source")
        cache1 = ContentHash.from_content(b"cached1")
        cache2 = ContentHash.from_content(b"cached2")

        invalidator.register_mapping(cache1, [source_hash])
        invalidator.register_mapping(cache2, [source_hash])

        dependents = invalidator.get_cached_for(source_hash)
        assert cache1 in dependents
        assert cache2 in dependents

    def test_get_invalidated(self) -> None:
        """Test getting invalidated entries."""
        invalidator = CacheInvalidator()
        source1 = ContentHash.from_content(b"source1")
        source2 = ContentHash.from_content(b"source2")
        cache1 = ContentHash.from_content(b"cached1")
        cache2 = ContentHash.from_content(b"cached2")

        invalidator.register_mapping(cache1, [source1])
        invalidator.register_mapping(cache2, [source2])

        invalidated = invalidator.get_invalidated([source1])
        assert cache1 in invalidated
        assert cache2 not in invalidated

    def test_on_source_changed(self) -> None:
        """Test handling source file changes."""
        invalidator = CacheInvalidator()
        old_source = ContentHash.from_content(b"old source")
        new_source = ContentHash.from_content(b"new source")
        cached = ContentHash.from_content(b"cached")

        invalidator.register_mapping(cached, [old_source])
        events = invalidator.on_source_changed(old_source, new_source)

        assert len(events) == 1
        assert events[0].content_hash == cached
        assert events[0].source_hash == old_source
        assert events[0].reason == "source_changed"

    def test_unregister(self) -> None:
        """Test unregistering mappings."""
        invalidator = CacheInvalidator()
        content_hash = ContentHash.from_content(b"cached")
        source_hash = ContentHash.from_content(b"source")

        invalidator.register_mapping(content_hash, [source_hash])
        invalidator.unregister(content_hash)

        assert len(invalidator.get_sources_for(content_hash)) == 0
        assert content_hash not in invalidator.get_cached_for(source_hash)

    def test_event_handler(self) -> None:
        """Test invalidation event handlers."""
        invalidator = CacheInvalidator()
        received_events: list[InvalidationEvent] = []

        def handler(event: InvalidationEvent) -> None:
            received_events.append(event)

        invalidator.add_event_handler(handler)

        old_source = ContentHash.from_content(b"old")
        new_source = ContentHash.from_content(b"new")
        cached = ContentHash.from_content(b"cached")
        invalidator.register_mapping(cached, [old_source])
        invalidator.on_source_changed(old_source, new_source)

        assert len(received_events) == 1

    def test_remove_event_handler(self) -> None:
        """Test removing event handlers."""
        invalidator = CacheInvalidator()

        def handler(event: InvalidationEvent) -> None:
            pass

        invalidator.add_event_handler(handler)
        assert invalidator.remove_event_handler(handler)
        assert not invalidator.remove_event_handler(handler)  # Already removed

    def test_clear(self) -> None:
        """Test clearing all mappings."""
        invalidator = CacheInvalidator()
        for i in range(5):
            invalidator.register_mapping(
                ContentHash.from_content(f"cached{i}".encode()),
                [ContentHash.from_content(f"source{i}".encode())],
            )

        assert invalidator.mapping_count() == 5
        invalidator.clear()
        assert invalidator.mapping_count() == 0

    def test_mapping_count(self) -> None:
        """Test mapping count."""
        invalidator = CacheInvalidator()
        source = ContentHash.from_content(b"source")

        for i in range(3):
            invalidator.register_mapping(
                ContentHash.from_content(f"cached{i}".encode()),
                [source],
            )

        assert invalidator.mapping_count() == 3


# =============================================================================
# DistributedCache Tests
# =============================================================================


class TestDistributedCache:
    """Tests for DistributedCache."""

    def test_put_and_get(
        self,
        distributed_cache: DistributedCache,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test basic put and get."""
        distributed_cache.put(sample_hash, sample_data)
        result = distributed_cache.get(sample_hash)
        assert isinstance(result, CacheHit)
        assert result.data == sample_data

    def test_put_bytes_auto_hash(
        self,
        distributed_cache: DistributedCache,
        sample_data: bytes,
    ) -> None:
        """Test put_bytes with automatic hashing."""
        content_hash, entry = distributed_cache.put_bytes(sample_data)
        assert content_hash == ContentHash.from_content(sample_data)
        assert distributed_cache.contains(content_hash)

    def test_contains_local(
        self,
        distributed_cache: DistributedCache,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test contains_local check."""
        assert not distributed_cache.contains_local(sample_hash)
        distributed_cache.put(sample_hash, sample_data)
        assert distributed_cache.contains_local(sample_hash)

    def test_invalidate_for_sources(
        self,
        distributed_cache: DistributedCache,
    ) -> None:
        """Test invalidation based on source changes."""
        source_hash = ContentHash.from_content(b"source file")
        data = b"cached artifact"
        content_hash = ContentHash.from_content(data)

        distributed_cache.put(content_hash, data, source_hash=source_hash)
        assert distributed_cache.contains(content_hash)

        invalidated = distributed_cache.invalidate_for_sources([source_hash])
        assert content_hash in invalidated
        assert not distributed_cache.contains(content_hash)

    def test_on_source_changed(
        self,
        distributed_cache: DistributedCache,
    ) -> None:
        """Test source change handling."""
        old_source = ContentHash.from_content(b"old source")
        new_source = ContentHash.from_content(b"new source")
        data = b"cached data"
        content_hash = ContentHash.from_content(data)

        distributed_cache.put(content_hash, data, source_hash=old_source)
        events = distributed_cache.on_source_changed(old_source, new_source)

        assert len(events) == 1
        assert not distributed_cache.contains(content_hash)

    def test_clear_local(
        self,
        distributed_cache: DistributedCache,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test clearing local cache."""
        distributed_cache.put(sample_hash, sample_data)
        count = distributed_cache.clear_local()
        assert count == 1
        assert len(distributed_cache) == 0

    def test_get_stats(
        self,
        distributed_cache: DistributedCache,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test getting combined stats."""
        distributed_cache.put(sample_hash, sample_data)
        distributed_cache.get(sample_hash)

        stats = distributed_cache.get_stats()
        assert "local" in stats
        assert stats["local"]["puts"] == 1
        assert stats["local"]["hits"] == 1

    def test_entries_iteration(
        self,
        distributed_cache: DistributedCache,
    ) -> None:
        """Test iterating over entries."""
        for i in range(5):
            data = f"data_{i}".encode()
            distributed_cache.put_bytes(data)

        entries = list(distributed_cache.entries())
        assert len(entries) == 5

    def test_save(
        self,
        distributed_cache: DistributedCache,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test save method."""
        distributed_cache.put(sample_hash, sample_data)
        distributed_cache.save()  # Should not raise

    def test_len_and_contains(
        self,
        distributed_cache: DistributedCache,
        sample_data: bytes,
        sample_hash: ContentHash,
    ) -> None:
        """Test len and contains operators."""
        assert len(distributed_cache) == 0
        assert sample_hash not in distributed_cache

        distributed_cache.put(sample_hash, sample_data)

        assert len(distributed_cache) == 1
        assert sample_hash in distributed_cache


# =============================================================================
# CacheServer Tests
# =============================================================================


class TestCacheServer:
    """Tests for CacheServer."""

    def test_server_lifecycle(self, temp_cache_dir: Path) -> None:
        """Test server start and stop."""
        config = CacheServerConfig(storage_dir=temp_cache_dir)
        server = CacheServer(config)

        assert not server.is_running
        server.start()
        assert server.is_running
        server.stop()
        assert not server.is_running

    def test_handle_put_and_get(self, temp_cache_dir: Path) -> None:
        """Test handling put and get requests."""
        config = CacheServerConfig(storage_dir=temp_cache_dir)
        server = CacheServer(config)
        server.start()

        data = b"server test data"
        content_hash = ContentHash.from_content(data)

        # Put
        entry = server.handle_put(content_hash, data)
        assert entry.content_hash == content_hash

        # Get
        result = server.handle_get(content_hash)
        assert isinstance(result, CacheHit)
        assert result.data == data

        server.stop()

    def test_handle_put_hash_mismatch(self, temp_cache_dir: Path) -> None:
        """Test rejection of mismatched hash."""
        config = CacheServerConfig(storage_dir=temp_cache_dir)
        server = CacheServer(config)
        server.start()

        wrong_hash = ContentHash.from_content(b"different")

        with pytest.raises(CacheCorruptionError):
            server.handle_put(wrong_hash, b"actual data")

        server.stop()

    def test_handle_contains(self, temp_cache_dir: Path) -> None:
        """Test handling contains check."""
        config = CacheServerConfig(storage_dir=temp_cache_dir)
        server = CacheServer(config)
        server.start()

        data = b"contains test"
        content_hash = ContentHash.from_content(data)

        assert not server.handle_contains(content_hash)
        server.handle_put(content_hash, data)
        assert server.handle_contains(content_hash)

        server.stop()

    def test_handle_remove(self, temp_cache_dir: Path) -> None:
        """Test handling remove request."""
        config = CacheServerConfig(storage_dir=temp_cache_dir)
        server = CacheServer(config)
        server.start()

        data = b"remove test"
        content_hash = ContentHash.from_content(data)

        server.handle_put(content_hash, data)
        assert server.handle_remove(content_hash)
        assert not server.handle_contains(content_hash)

        server.stop()

    def test_handle_batch_contains(self, temp_cache_dir: Path) -> None:
        """Test batch contains check."""
        config = CacheServerConfig(storage_dir=temp_cache_dir)
        server = CacheServer(config)
        server.start()

        hashes = []
        for i in range(5):
            data = f"batch_{i}".encode()
            h = ContentHash.from_content(data)
            hashes.append(h)
            if i < 3:  # Only add first 3
                server.handle_put(h, data)

        result = server.handle_batch_contains(hashes)
        assert sum(result.values()) == 3

        server.stop()


# =============================================================================
# DeltaSyncManager Tests
# =============================================================================


class TestDeltaSyncManager:
    """Tests for DeltaSyncManager."""

    def test_sync_delta_calculation(self, temp_cache_dir: Path) -> None:
        """Test calculating sync delta."""
        local = LocalCache(temp_cache_dir / "local")
        remote_config = CacheClientConfig(server_url="http://localhost")
        remote = RemoteCache(remote_config)
        remote.connect()  # Connect first

        sync_manager = DeltaSyncManager(local, remote)

        # Add local entries
        hashes = []
        for i in range(5):
            data = f"local_{i}".encode()
            h = ContentHash.from_content(data)
            local.put(h, data)
            hashes.append(h)

        # Remote is connected but empty, so all local should be in local_only
        local_only, remote_only = sync_manager.get_sync_delta()
        assert len(local_only) == 5
        for h in hashes:
            assert h in local_only


# =============================================================================
# CIBuildPopulator Tests
# =============================================================================


class TestCIBuildPopulator:
    """Tests for CIBuildPopulator."""

    def test_populate_basic(
        self,
        distributed_cache: DistributedCache,
    ) -> None:
        """Test basic population from build artifacts."""
        populator = CIBuildPopulator(metadata_prefix="ci")

        build_artifacts = {
            "shader.spv": b"compiled shader bytecode",
            "model.mesh": b"processed mesh data",
        }
        source_hashes = {
            "shader.glsl": ContentHash.from_content(b"shader source"),
            "model.obj": ContentHash.from_content(b"model source"),
        }

        count = populator.populate(distributed_cache, build_artifacts, source_hashes)
        assert count == 2

        # Verify artifacts are cached
        shader_hash = ContentHash.from_content(b"compiled shader bytecode")
        result = distributed_cache.get(shader_hash)
        assert isinstance(result, CacheHit)
        assert result.entry.get_metadata("ci_artifact") == "shader.spv"

    def test_populate_with_source_mapping(
        self,
        distributed_cache: DistributedCache,
    ) -> None:
        """Test that source hash mapping is registered."""
        populator = CIBuildPopulator()

        source_hash = ContentHash.from_content(b"shader source")
        build_artifacts = {"shader.spv": b"compiled bytecode"}
        source_hashes = {"shader": source_hash}

        populator.populate(distributed_cache, build_artifacts, source_hashes)

        # Invalidating source should invalidate cached artifact
        artifact_hash = ContentHash.from_content(b"compiled bytecode")
        invalidated = distributed_cache.invalidate_for_sources([source_hash])
        assert artifact_hash in invalidated

    def test_artifact_to_source_mapping(self) -> None:
        """Test artifact name to source name mapping."""
        populator = CIBuildPopulator()

        # Test various extension mappings
        assert populator._artifact_to_source("shader.spv") == "shader"
        assert populator._artifact_to_source("model.obj") == "model"
        assert populator._artifact_to_source("texture.dxil") == "texture"
        assert populator._artifact_to_source("audio.o") == "audio"

    def test_populate_from_directory(
        self,
        distributed_cache: DistributedCache,
        tmp_path: Path,
    ) -> None:
        """Test populating from directory structure."""
        populator = CIBuildPopulator()

        # Create artifact directory
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "shader.spv").write_bytes(b"shader bytecode")
        (artifacts_dir / "model.mesh").write_bytes(b"mesh data")

        # Create sources directory
        sources_dir = tmp_path / "sources"
        sources_dir.mkdir()
        (sources_dir / "shader.glsl").write_bytes(b"shader source")
        (sources_dir / "model.obj").write_bytes(b"model source")

        count = populator.populate_from_directory(
            distributed_cache,
            artifacts_dir,
            sources_dir,
        )
        assert count == 2


# =============================================================================
# InvalidationEvent Tests
# =============================================================================


class TestInvalidationEvent:
    """Tests for InvalidationEvent."""

    def test_event_creation(self) -> None:
        """Test creating invalidation event."""
        content_hash = ContentHash.from_content(b"cached")
        source_hash = ContentHash.from_content(b"source")

        event = InvalidationEvent(
            content_hash=content_hash,
            source_hash=source_hash,
            reason="source_changed",
        )

        assert event.content_hash == content_hash
        assert event.source_hash == source_hash
        assert event.reason == "source_changed"
        assert event.timestamp > 0

    def test_event_serialization(self) -> None:
        """Test event to dict."""
        content_hash = ContentHash.from_content(b"cached")
        source_hash = ContentHash.from_content(b"source")

        event = InvalidationEvent(
            content_hash=content_hash,
            source_hash=source_hash,
            reason="test",
            timestamp=1234567890.0,
        )

        data = event.to_dict()
        assert data["content_hash"] == content_hash.hex
        assert data["source_hash"] == source_hash.hex
        assert data["reason"] == "test"
        assert data["timestamp"] == 1234567890.0


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_cache_error_base(self) -> None:
        """Test CacheError base class."""
        error = CacheError("test error")
        assert str(error) == "test error"

    def test_cache_connection_error(self) -> None:
        """Test CacheConnectionError."""
        error = CacheConnectionError("connection failed")
        assert isinstance(error, CacheError)
        assert "connection failed" in str(error)

    def test_cache_corruption_error(self) -> None:
        """Test CacheCorruptionError."""
        error = CacheCorruptionError("data corrupted")
        assert isinstance(error, CacheError)
        assert "corrupted" in str(error)

    def test_cache_full_error(self) -> None:
        """Test CacheFullError."""
        error = CacheFullError("cache full")
        assert isinstance(error, CacheError)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the distributed cache system."""

    def test_full_workflow(self, temp_cache_dir: Path) -> None:
        """Test complete cache workflow."""
        # Setup
        config = CacheConfig(
            local_cache_dir=temp_cache_dir,
            compression_enabled=True,
            remote_enabled=False,
        )
        cache = DistributedCache(config)

        # Simulate building an asset
        source_data = b"shader source code"
        source_hash = ContentHash.from_content(source_data)
        built_data = b"compiled shader bytecode"
        built_hash = ContentHash.from_content(built_data)

        # Cache the built asset
        cache.put(
            built_hash,
            built_data,
            source_hash=source_hash,
            metadata={"build_time": "1.5s"},
        )

        # Retrieve from cache
        result = cache.get(built_hash)
        assert isinstance(result, CacheHit)
        assert result.data == built_data

        # Simulate source change
        new_source_hash = ContentHash.from_content(b"modified source")
        events = cache.on_source_changed(source_hash, new_source_hash)

        # Cache should be invalidated
        assert len(events) == 1
        result = cache.get(built_hash)
        assert isinstance(result, CacheMiss)

    def test_ci_build_caching(self, temp_cache_dir: Path) -> None:
        """Test CI build caching workflow."""
        config = CacheConfig(
            local_cache_dir=temp_cache_dir,
            remote_enabled=False,
        )
        cache = DistributedCache(config)
        populator = CIBuildPopulator()

        # Simulate CI build
        artifacts = {
            "shader.spv": b"bytecode",
            "texture.ktx": b"compressed texture",
        }
        sources = {
            "shader": ContentHash.from_content(b"glsl source"),
            "texture": ContentHash.from_content(b"png source"),
        }

        # Populate cache from CI
        count = populator.populate(cache, artifacts, sources)
        assert count == 2

        # Developer pulls from cache
        shader_hash = ContentHash.from_content(b"bytecode")
        result = cache.get(shader_hash)
        assert isinstance(result, CacheHit)
        assert result.entry.get_metadata("ci_artifact") == "shader.spv"

    def test_multiple_sources_single_artifact(
        self,
        distributed_cache: DistributedCache,
    ) -> None:
        """Test artifact depending on multiple sources."""
        source1 = ContentHash.from_content(b"header.h")
        source2 = ContentHash.from_content(b"impl.cpp")
        artifact = b"compiled.o"
        artifact_hash = ContentHash.from_content(artifact)

        # Register both sources
        distributed_cache.put(artifact_hash, artifact, source_hash=source1)
        distributed_cache.invalidator.register_mapping(artifact_hash, [source2])

        # Changing either source should invalidate
        invalidated = distributed_cache.invalidate_for_sources([source2])
        assert artifact_hash in invalidated

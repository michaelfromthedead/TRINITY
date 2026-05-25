"""Tests for incremental build cache."""
import pytest
import os
import tempfile
import shutil
import time
from engine.tooling.build.build_cache import (
    ContentHash,
    CacheEntry,
    BuildCacheBackend,
    FilesystemCache,
    MemoryCache,
    BuildCache,
    IncrementalBuilder,
)


class TestContentHash:
    """Tests for ContentHash dataclass."""

    def test_hash_creation(self):
        """Test creating content hash."""
        hash_obj = ContentHash(
            path="/test/file.txt",
            hash_value="abc123",
            algorithm="sha256",
            size=1024,
        )
        assert hash_obj.path == "/test/file.txt"
        assert hash_obj.hash_value == "abc123"

    def test_compute_hash(self):
        """Test computing hash from file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            hash_obj = ContentHash.compute(temp_path)
            assert hash_obj.path == temp_path
            assert len(hash_obj.hash_value) == 64  # SHA256 hex
            assert hash_obj.size == 12
        finally:
            os.unlink(temp_path)

    def test_compute_hash_different_algorithm(self):
        """Test computing hash with different algorithm."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            hash_obj = ContentHash.compute(temp_path, algorithm="md5")
            assert hash_obj.algorithm == "md5"
            assert len(hash_obj.hash_value) == 32  # MD5 hex
        finally:
            os.unlink(temp_path)

    def test_from_bytes(self):
        """Test creating hash from bytes."""
        data = b"test data"
        hash_obj = ContentHash.from_bytes(data, "virtual/path")
        assert hash_obj.path == "virtual/path"
        assert hash_obj.size == len(data)

    def test_matches(self):
        """Test hash matching."""
        hash1 = ContentHash("a", "abc123", "sha256", 100)
        hash2 = ContentHash("b", "abc123", "sha256", 200)
        hash3 = ContentHash("c", "xyz789", "sha256", 100)

        assert hash1.matches(hash2)
        assert not hash1.matches(hash3)

    def test_compute_nonexistent_file(self):
        """Test computing hash for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            ContentHash.compute("/nonexistent/file.txt")


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_entry_creation(self):
        """Test creating cache entry."""
        source_hash = ContentHash("src.txt", "abc", "sha256", 100)
        entry = CacheEntry(
            key="key123",
            source_hash=source_hash,
            output_path="/out/file.o",
        )
        assert entry.key == "key123"
        assert entry.hits == 0

    def test_to_dict(self):
        """Test converting to dictionary."""
        source_hash = ContentHash("src.txt", "abc", "sha256", 100)
        entry = CacheEntry(
            key="key123",
            source_hash=source_hash,
            output_path="/out/file.o",
        )
        data = entry.to_dict()
        assert data["key"] == "key123"
        assert data["source_hash"]["hash_value"] == "abc"

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "key": "key456",
            "source_hash": {
                "path": "src.cpp",
                "hash_value": "xyz",
                "algorithm": "sha256",
                "size": 500,
                "modified_time": 0.0,
            },
            "output_path": "/out/src.o",
            "output_hash": None,
            "dependencies": [],
            "created_at": time.time(),
            "last_accessed": time.time(),
            "hits": 5,
            "metadata": {},
        }
        entry = CacheEntry.from_dict(data)
        assert entry.key == "key456"
        assert entry.hits == 5


class TestMemoryCache:
    """Tests for MemoryCache backend."""

    def test_cache_put_get(self):
        """Test putting and getting entries."""
        cache = MemoryCache()
        source_hash = ContentHash("src.txt", "abc", "sha256", 100)
        entry = CacheEntry(
            key="key1",
            source_hash=source_hash,
            output_path="/out/file.o",
        )

        cache.put(entry)
        retrieved = cache.get("key1")

        assert retrieved is not None
        assert retrieved.key == "key1"

    def test_cache_contains(self):
        """Test contains check."""
        cache = MemoryCache()
        source_hash = ContentHash("src.txt", "abc", "sha256", 100)
        entry = CacheEntry(key="key1", source_hash=source_hash, output_path="/out")

        cache.put(entry)
        assert cache.contains("key1")
        assert not cache.contains("key2")

    def test_cache_remove(self):
        """Test removing entries."""
        cache = MemoryCache()
        source_hash = ContentHash("src.txt", "abc", "sha256", 100)
        entry = CacheEntry(key="key1", source_hash=source_hash, output_path="/out")

        cache.put(entry)
        result = cache.remove("key1")
        assert result is True
        assert not cache.contains("key1")

    def test_cache_clear(self):
        """Test clearing cache."""
        cache = MemoryCache()
        source_hash = ContentHash("src.txt", "abc", "sha256", 100)

        cache.put(CacheEntry(key="key1", source_hash=source_hash, output_path="/out"))
        cache.put(CacheEntry(key="key2", source_hash=source_hash, output_path="/out"))

        cache.clear()
        assert len(cache.get_all_keys()) == 0

    def test_cache_eviction(self):
        """Test LRU eviction."""
        cache = MemoryCache(max_entries=2)
        source_hash = ContentHash("src.txt", "abc", "sha256", 100)

        cache.put(CacheEntry(key="key1", source_hash=source_hash, output_path="/out"))
        cache.put(CacheEntry(key="key2", source_hash=source_hash, output_path="/out"))

        # Access key1 to make it more recent
        cache.get("key1")

        # Add key3, should evict key2
        cache.put(CacheEntry(key="key3", source_hash=source_hash, output_path="/out"))

        assert cache.contains("key1")
        assert not cache.contains("key2")
        assert cache.contains("key3")

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = MemoryCache()
        source_hash = ContentHash("src.txt", "abc", "sha256", 100)
        entry = CacheEntry(key="key1", source_hash=source_hash, output_path="/out")

        cache.put(entry)
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["entry_count"] == 1


class TestFilesystemCache:
    """Tests for FilesystemCache backend."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_cache_creation(self, temp_cache_dir):
        """Test creating filesystem cache."""
        cache = FilesystemCache(temp_cache_dir)
        assert os.path.exists(temp_cache_dir)

    def test_put_get(self, temp_cache_dir):
        """Test putting and getting entries."""
        cache = FilesystemCache(temp_cache_dir)
        source_hash = ContentHash("src.txt", "abc", "sha256", 100)
        entry = CacheEntry(key="key1", source_hash=source_hash, output_path="/out")

        cache.put(entry)
        retrieved = cache.get("key1")

        assert retrieved is not None
        assert retrieved.key == "key1"

    def test_persistence(self, temp_cache_dir):
        """Test cache persistence across instances."""
        source_hash = ContentHash("src.txt", "abc", "sha256", 100)
        entry = CacheEntry(key="key1", source_hash=source_hash, output_path="/out")

        cache1 = FilesystemCache(temp_cache_dir)
        cache1.put(entry)

        # Create new cache instance
        cache2 = FilesystemCache(temp_cache_dir)
        retrieved = cache2.get("key1")

        assert retrieved is not None

    def test_clear(self, temp_cache_dir):
        """Test clearing filesystem cache."""
        cache = FilesystemCache(temp_cache_dir)
        source_hash = ContentHash("src.txt", "abc", "sha256", 100)
        entry = CacheEntry(key="key1", source_hash=source_hash, output_path="/out")

        cache.put(entry)
        cache.clear()

        assert len(cache.get_all_keys()) == 0


class TestBuildCache:
    """Tests for BuildCache."""

    def test_compute_key(self):
        """Test computing cache keys."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"source content")
            temp_path = f.name

        try:
            cache = BuildCache(MemoryCache())
            key = cache.compute_key(temp_path, "config_hash_123")

            assert len(key) == 64  # SHA256 hex
        finally:
            os.unlink(temp_path)

    def test_put_and_is_valid(self):
        """Test putting and validating cache entries."""
        cache = BuildCache(MemoryCache())

        with tempfile.NamedTemporaryFile(delete=False) as sf:
            sf.write(b"source")
            source_path = sf.name

        with tempfile.NamedTemporaryFile(delete=False) as of:
            of.write(b"output")
            output_path = of.name

        try:
            key = cache.compute_key(source_path, "config")
            cache.put(key, source_path, output_path)

            assert cache.is_valid(key)
        finally:
            os.unlink(source_path)
            os.unlink(output_path)

    def test_invalidate(self):
        """Test cache invalidation."""
        cache = BuildCache(MemoryCache())

        with tempfile.NamedTemporaryFile(delete=False) as sf:
            sf.write(b"source")
            source_path = sf.name

        with tempfile.NamedTemporaryFile(delete=False) as of:
            of.write(b"output")
            output_path = of.name

        try:
            key = cache.compute_key(source_path, "config")
            cache.put(key, source_path, output_path)
            cache.invalidate(key)

            assert not cache.is_valid(key)
        finally:
            os.unlink(source_path)
            os.unlink(output_path)

    def test_source_modified_invalidates(self):
        """Test that modifying source invalidates cache."""
        cache = BuildCache(MemoryCache())

        with tempfile.NamedTemporaryFile(delete=False) as sf:
            sf.write(b"source")
            source_path = sf.name

        with tempfile.NamedTemporaryFile(delete=False) as of:
            of.write(b"output")
            output_path = of.name

        try:
            key = cache.compute_key(source_path, "config")
            cache.put(key, source_path, output_path)

            # Modify source
            with open(source_path, "w") as f:
                f.write("modified source")

            assert not cache.is_valid(key)
        finally:
            os.unlink(source_path)
            os.unlink(output_path)

    def test_get_stats(self):
        """Test getting cache statistics."""
        cache = BuildCache(MemoryCache())
        stats = cache.get_stats()
        assert "entry_count" in stats


class TestIncrementalBuilder:
    """Tests for IncrementalBuilder."""

    def test_needs_rebuild_new_file(self):
        """Test needs_rebuild for new file."""
        cache = BuildCache(MemoryCache())
        builder = IncrementalBuilder(cache)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"content")
            temp_path = f.name

        try:
            assert builder.needs_rebuild(temp_path, "config") is True
        finally:
            os.unlink(temp_path)

    def test_needs_rebuild_cached(self):
        """Test needs_rebuild for cached file."""
        cache = BuildCache(MemoryCache())
        builder = IncrementalBuilder(cache)

        with tempfile.NamedTemporaryFile(delete=False) as sf:
            sf.write(b"source")
            source_path = sf.name

        with tempfile.NamedTemporaryFile(delete=False) as of:
            of.write(b"output")
            output_path = of.name

        try:
            builder.record_build(source_path, output_path, "config")
            assert builder.needs_rebuild(source_path, "config") is False
        finally:
            os.unlink(source_path)
            os.unlink(output_path)

    def test_get_changed_files(self):
        """Test getting list of changed files."""
        cache = BuildCache(MemoryCache())
        builder = IncrementalBuilder(cache)

        files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}.txt") as f:
                f.write(f"content {i}".encode())
                files.append(f.name)

        try:
            # Initially all need rebuild
            changed = builder.get_changed_files(files, "config")
            assert len(changed) == 3

            # Record first file
            with tempfile.NamedTemporaryFile(delete=False) as of:
                of.write(b"out")
                output_path = of.name

            builder.record_build(files[0], output_path, "config")

            # Now only 2 need rebuild
            changed = builder.get_changed_files(files, "config")
            assert len(changed) == 2
            assert files[0] not in changed

            os.unlink(output_path)
        finally:
            for f in files:
                os.unlink(f)

    def test_get_cached_output(self):
        """Test getting cached output path."""
        cache = BuildCache(MemoryCache())
        builder = IncrementalBuilder(cache)

        with tempfile.NamedTemporaryFile(delete=False) as sf:
            sf.write(b"source")
            source_path = sf.name

        with tempfile.NamedTemporaryFile(delete=False) as of:
            of.write(b"output")
            output_path = of.name

        try:
            builder.record_build(source_path, output_path, "config")
            cached = builder.get_cached_output(source_path, "config")
            assert cached == output_path
        finally:
            os.unlink(source_path)
            os.unlink(output_path)

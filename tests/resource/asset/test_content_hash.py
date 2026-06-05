"""Tests for content-addressed asset hashing."""
from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from engine.resource.asset.content_hash import (
    CACHE_VERSION,
    DEFAULT_CHUNK_SIZE,
    HASH_BYTES,
    HASH_HEX_LENGTH,
    NULL_HASH_HEX,
    AssetHasher,
    CacheEntry,
    ContentAddressedStorage,
    ContentHash,
    HashAlgorithm,
    HashCache,
    StoredAsset,
)


# ==============================================================================
# ContentHash Tests
# ==============================================================================


class TestContentHash:
    """Tests for ContentHash value object."""

    def test_create_from_bytes(self) -> None:
        digest = bytes(range(32))
        h = ContentHash.from_bytes(digest)
        assert h.digest == digest

    def test_create_from_hex(self) -> None:
        hex_str = "a" * 64
        h = ContentHash.from_hex(hex_str)
        assert h.hex == hex_str

    def test_create_from_content(self) -> None:
        data = b"hello world"
        h = ContentHash.from_content(data)
        assert len(h.digest) == HASH_BYTES
        # Same content should produce same hash
        h2 = ContentHash.from_content(data)
        assert h == h2

    def test_null_hash(self) -> None:
        h = ContentHash.null()
        assert h.is_null()
        assert h.digest == bytes(HASH_BYTES)
        assert h.hex == NULL_HASH_HEX

    def test_non_null_hash_is_not_null(self) -> None:
        h = ContentHash.from_content(b"test")
        assert not h.is_null()

    def test_invalid_digest_length_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 32 bytes"):
            ContentHash.from_bytes(b"short")

    def test_invalid_hex_length_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 64 chars"):
            ContentHash.from_hex("abc")

    def test_equality(self) -> None:
        h1 = ContentHash.from_content(b"test")
        h2 = ContentHash.from_content(b"test")
        h3 = ContentHash.from_content(b"different")
        assert h1 == h2
        assert h1 != h3

    def test_equality_with_non_hash(self) -> None:
        h = ContentHash.from_content(b"test")
        assert (h == "not a hash") is False
        assert h != "not a hash"

    def test_hash_for_dict_key(self) -> None:
        h1 = ContentHash.from_content(b"test")
        h2 = ContentHash.from_content(b"test")
        d: dict[ContentHash, str] = {h1: "value"}
        assert h2 in d
        assert d[h2] == "value"

    def test_short_hex(self) -> None:
        h = ContentHash.from_content(b"test")
        assert len(h.short_hex) == 16
        assert h.short_hex == h.hex[:16]

    def test_repr_null(self) -> None:
        h = ContentHash.null()
        assert repr(h) == "ContentHash(null)"

    def test_repr_non_null(self) -> None:
        h = ContentHash.from_content(b"test")
        assert "ContentHash(" in repr(h)
        assert "..." in repr(h)

    def test_str_returns_full_hex(self) -> None:
        h = ContentHash.from_content(b"test")
        assert str(h) == h.hex
        assert len(str(h)) == HASH_HEX_LENGTH

    def test_frozen_immutable(self) -> None:
        h = ContentHash.from_content(b"test")
        with pytest.raises(AttributeError):
            h._digest = b"changed"  # type: ignore[misc]


# ==============================================================================
# HashAlgorithm Tests
# ==============================================================================


class TestHashAlgorithm:
    """Tests for HashAlgorithm abstraction."""

    def test_update_and_digest(self) -> None:
        algo = HashAlgorithm()
        algo.update(b"hello")
        algo.update(b" world")
        digest = algo.digest()
        assert len(digest) == HASH_BYTES

    def test_hexdigest(self) -> None:
        algo = HashAlgorithm()
        algo.update(b"test")
        hex_digest = algo.hexdigest()
        assert len(hex_digest) == HASH_HEX_LENGTH

    def test_hash_bytes_classmethod(self) -> None:
        digest = HashAlgorithm.hash_bytes(b"test")
        assert len(digest) == HASH_BYTES

    def test_hash_bytes_hex_classmethod(self) -> None:
        hex_str = HashAlgorithm.hash_bytes_hex(b"test")
        assert len(hex_str) == HASH_HEX_LENGTH

    def test_deterministic_hashing(self) -> None:
        data = b"consistent content"
        h1 = HashAlgorithm.hash_bytes(data)
        h2 = HashAlgorithm.hash_bytes(data)
        assert h1 == h2

    def test_different_content_different_hash(self) -> None:
        h1 = HashAlgorithm.hash_bytes(b"content A")
        h2 = HashAlgorithm.hash_bytes(b"content B")
        assert h1 != h2

    def test_is_native_blake3_returns_bool(self) -> None:
        result = HashAlgorithm.is_native_blake3()
        assert isinstance(result, bool)

    def test_empty_content_produces_hash(self) -> None:
        h = HashAlgorithm.hash_bytes(b"")
        assert len(h) == HASH_BYTES


# ==============================================================================
# AssetHasher Tests
# ==============================================================================


class TestAssetHasher:
    """Tests for AssetHasher."""

    def test_default_chunk_size(self) -> None:
        hasher = AssetHasher()
        assert hasher.chunk_size == DEFAULT_CHUNK_SIZE

    def test_custom_chunk_size(self) -> None:
        hasher = AssetHasher(chunk_size=1024)
        assert hasher.chunk_size == 1024

    def test_invalid_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            AssetHasher(chunk_size=0)
        with pytest.raises(ValueError, match="must be positive"):
            AssetHasher(chunk_size=-1)

    def test_hash_bytes(self) -> None:
        hasher = AssetHasher()
        h = hasher.hash_bytes(b"test data")
        assert isinstance(h, ContentHash)
        assert not h.is_null()

    def test_hash_file(self) -> None:
        hasher = AssetHasher()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"file content")
            path = f.name
        try:
            h = hasher.hash_file(path)
            assert isinstance(h, ContentHash)
            # Should match in-memory hash
            assert h == hasher.hash_bytes(b"file content")
        finally:
            os.unlink(path)

    def test_hash_file_not_found_raises(self) -> None:
        hasher = AssetHasher()
        with pytest.raises(FileNotFoundError):
            hasher.hash_file("/nonexistent/path/file.dat")

    def test_hash_file_is_directory_raises(self) -> None:
        hasher = AssetHasher()
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError, match="Not a file"):
                hasher.hash_file(d)

    def test_hash_stream(self) -> None:
        hasher = AssetHasher()
        stream = io.BytesIO(b"stream content")
        h = hasher.hash_stream(stream)
        assert h == hasher.hash_bytes(b"stream content")

    def test_hash_multiple_files(self) -> None:
        hasher = AssetHasher()
        with tempfile.TemporaryDirectory() as d:
            p1 = Path(d) / "file1.txt"
            p2 = Path(d) / "file2.txt"
            p1.write_bytes(b"content 1")
            p2.write_bytes(b"content 2")

            results = hasher.hash_multiple_files([p1, p2])
            assert len(results) == 2
            assert p1 in results
            assert p2 in results

    def test_hash_multiple_files_skips_missing(self) -> None:
        hasher = AssetHasher()
        with tempfile.TemporaryDirectory() as d:
            p1 = Path(d) / "exists.txt"
            p1.write_bytes(b"exists")

            results = hasher.hash_multiple_files([p1, "/nonexistent/file.dat"])
            assert len(results) == 1
            assert p1 in results

    def test_verify_hash_success(self) -> None:
        hasher = AssetHasher()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"verify me")
            path = f.name
        try:
            expected = hasher.hash_file(path)
            assert hasher.verify_hash(path, expected)
        finally:
            os.unlink(path)

    def test_verify_hash_failure(self) -> None:
        hasher = AssetHasher()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"verify me")
            path = f.name
        try:
            wrong_hash = ContentHash.from_content(b"different content")
            assert not hasher.verify_hash(path, wrong_hash)
        finally:
            os.unlink(path)

    def test_verify_hash_missing_file(self) -> None:
        hasher = AssetHasher()
        h = ContentHash.from_content(b"test")
        assert not hasher.verify_hash("/nonexistent/file.dat", h)

    def test_large_file_streaming(self) -> None:
        hasher = AssetHasher(chunk_size=1024)
        # Create content larger than chunk size
        large_content = b"x" * 10000
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(large_content)
            path = f.name
        try:
            h = hasher.hash_file(path)
            assert h == hasher.hash_bytes(large_content)
        finally:
            os.unlink(path)


# ==============================================================================
# HashCache Tests
# ==============================================================================


class TestHashCache:
    """Tests for HashCache."""

    def test_create_without_path(self) -> None:
        cache = HashCache()
        assert len(cache) == 0

    def test_get_or_compute_caches_result(self) -> None:
        cache = HashCache()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"cache test")
            path = f.name
        try:
            h1 = cache.get_or_compute(path)
            h2 = cache.get_or_compute(path)
            assert h1 == h2
            assert len(cache) == 1
        finally:
            os.unlink(path)

    def test_get_returns_none_for_uncached(self) -> None:
        cache = HashCache()
        assert cache.get("/some/path.dat") is None

    def test_invalidate(self) -> None:
        cache = HashCache()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"invalidate test")
            path = f.name
        try:
            cache.get_or_compute(path)
            assert len(cache) == 1
            assert cache.invalidate(path)
            assert len(cache) == 0
        finally:
            os.unlink(path)

    def test_invalidate_returns_false_if_not_cached(self) -> None:
        cache = HashCache()
        assert not cache.invalidate("/not/cached.dat")

    def test_invalidate_all(self) -> None:
        cache = HashCache()
        with tempfile.TemporaryDirectory() as d:
            for i in range(3):
                p = Path(d) / f"file{i}.txt"
                p.write_bytes(f"content {i}".encode())
                cache.get_or_compute(p)

            assert len(cache) == 3
            count = cache.invalidate_all()
            assert count == 3
            assert len(cache) == 0

    def test_contains(self) -> None:
        cache = HashCache()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"contains test")
            path = f.name
        try:
            assert not cache.contains(path)
            cache.get_or_compute(path)
            assert cache.contains(path)
            assert path in cache
        finally:
            os.unlink(path)

    def test_cache_invalidates_on_mtime_change(self) -> None:
        cache = HashCache()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"original")
            path = f.name
        try:
            h1 = cache.get_or_compute(path)
            # Modify file
            time.sleep(0.01)  # Ensure mtime changes
            with open(path, "wb") as f:
                f.write(b"modified")
            # Cache should detect change
            assert cache.get(path) is None
            h2 = cache.get_or_compute(path)
            assert h1 != h2
        finally:
            os.unlink(path)

    def test_cache_invalidates_on_size_change(self) -> None:
        cache = HashCache()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"short")
            path = f.name
        try:
            cache.get_or_compute(path)
            # Change size
            with open(path, "wb") as f:
                f.write(b"much longer content")
            assert cache.get(path) is None
        finally:
            os.unlink(path)

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cache_path = Path(d) / "cache.json"
            file_path = Path(d) / "asset.dat"
            file_path.write_bytes(b"persistent content")

            # Create and save cache
            cache1 = HashCache(cache_path=cache_path)
            h1 = cache1.get_or_compute(file_path)
            cache1.save()

            # Load in new cache instance
            cache2 = HashCache(cache_path=cache_path)
            h2 = cache2.get(str(file_path.resolve()))
            assert h1 == h2

    def test_load_corrupted_cache_clears(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cache_path = Path(d) / "cache.json"
            cache_path.write_text("not valid json {{{")

            cache = HashCache(cache_path=cache_path)
            assert len(cache) == 0

    def test_load_wrong_version_clears(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cache_path = Path(d) / "cache.json"
            cache_path.write_text(json.dumps({"version": 999, "entries": {}}))

            cache = HashCache(cache_path=cache_path)
            assert len(cache) == 0

    def test_is_dirty(self) -> None:
        cache = HashCache()
        assert not cache.is_dirty
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"dirty test")
            path = f.name
        try:
            cache.get_or_compute(path)
            assert cache.is_dirty
        finally:
            os.unlink(path)

    def test_entries_iterator(self) -> None:
        cache = HashCache()
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i in range(3):
                p = Path(d) / f"file{i}.txt"
                p.write_bytes(f"content {i}".encode())
                paths.append(str(p.resolve()))
                cache.get_or_compute(p)

            entries = list(cache.entries())
            assert len(entries) == 3
            for path, h in entries:
                assert path in paths
                assert isinstance(h, ContentHash)


# ==============================================================================
# ContentAddressedStorage Tests
# ==============================================================================


class TestContentAddressedStorage:
    """Tests for ContentAddressedStorage."""

    def test_store_and_get_by_hash(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test data")
            path = f.name
        try:
            h = storage.store(path, b"test data")
            assert storage.get_by_hash(h) == b"test data"
        finally:
            os.unlink(path)

    def test_store_and_get_by_path(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test data")
            path = f.name
        try:
            storage.store(path, b"test data")
            assert storage.get_by_path(path) == b"test data"
        finally:
            os.unlink(path)

    def test_deduplication(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        with tempfile.TemporaryDirectory() as d:
            # Two files with same content
            p1 = Path(d) / "file1.dat"
            p2 = Path(d) / "file2.dat"
            content = b"identical content"
            p1.write_bytes(content)
            p2.write_bytes(content)

            h1 = storage.store(str(p1), content)
            h2 = storage.store(str(p2), content)

            # Same hash
            assert h1 == h2
            # Only one unique asset stored
            assert len(storage) == 1
            # Both paths work
            assert storage.get_by_path(str(p1)) == content
            assert storage.get_by_path(str(p2)) == content

    def test_store_bytes_without_path(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        h = storage.store_bytes(b"raw bytes")
        assert storage.get_by_hash(h) == b"raw bytes"

    def test_store_bytes_with_virtual_path(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        h = storage.store_bytes(b"virtual", virtual_path="virtual://asset")
        assert storage.get_by_path("virtual://asset") == b"virtual"

    def test_get_hash_for_path(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"get hash test")
            path = f.name
        try:
            expected_hash = storage.store(path, b"get hash test")
            actual_hash = storage.get_hash_for_path(path)
            assert actual_hash == expected_hash
        finally:
            os.unlink(path)

    def test_get_paths_for_hash(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        with tempfile.TemporaryDirectory() as d:
            content = b"shared content"
            p1 = Path(d) / "a.dat"
            p2 = Path(d) / "b.dat"
            p1.write_bytes(content)
            p2.write_bytes(content)

            h = storage.store(str(p1), content)
            storage.store(str(p2), content)

            paths = storage.get_paths_for_hash(h)
            assert str(p1.resolve()) in paths
            assert str(p2.resolve()) in paths

    def test_release_decrements_ref_count(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        with tempfile.TemporaryDirectory() as d:
            content = b"ref count test"
            p1 = Path(d) / "a.dat"
            p2 = Path(d) / "b.dat"
            p1.write_bytes(content)
            p2.write_bytes(content)

            h = storage.store(str(p1), content)
            storage.store(str(p2), content)

            assert storage.get_ref_count(h) == 2
            assert not storage.release(h)  # Still has ref
            assert storage.get_ref_count(h) == 1
            assert storage.release(h)  # Now removed
            assert storage.get_ref_count(h) == 0

    def test_release_by_path(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"release by path")
            path = f.name
        try:
            h = storage.store(path, b"release by path")
            assert storage.contains_hash(h)
            storage.release_by_path(path)
            assert not storage.contains_hash(h)
        finally:
            os.unlink(path)

    def test_contains_hash(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        h = storage.store_bytes(b"contains test")
        assert storage.contains_hash(h)
        assert h in storage

    def test_contains_path(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"path contains")
            path = f.name
        try:
            storage.store(path, b"path contains")
            assert storage.contains_path(path)
            assert path in storage
        finally:
            os.unlink(path)

    def test_find_duplicates(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        with tempfile.TemporaryDirectory() as d:
            # Unique content
            p1 = Path(d) / "unique.dat"
            p1.write_bytes(b"unique")
            storage.store(str(p1), b"unique")

            # Duplicated content
            shared = b"shared content"
            p2 = Path(d) / "dup1.dat"
            p3 = Path(d) / "dup2.dat"
            p2.write_bytes(shared)
            p3.write_bytes(shared)
            storage.store(str(p2), shared)
            storage.store(str(p3), shared)

            duplicates = storage.find_duplicates()
            assert len(duplicates) == 1
            h, paths = duplicates[0]
            assert len(paths) == 2

    def test_get_stats(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        with tempfile.TemporaryDirectory() as d:
            # Add some assets
            shared = b"shared"
            p1 = Path(d) / "a.dat"
            p2 = Path(d) / "b.dat"
            p3 = Path(d) / "c.dat"
            p1.write_bytes(shared)
            p2.write_bytes(shared)
            p3.write_bytes(b"unique")

            storage.store(str(p1), shared)
            storage.store(str(p2), shared)
            storage.store(str(p3), b"unique")

            stats = storage.get_stats()
            assert stats["unique_assets"] == 2
            assert stats["total_paths"] == 3
            assert stats["total_refs"] == 3
            assert stats["duplicate_groups"] == 1
            assert stats["deduplication_ratio"] == 1.5

    def test_clear(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        storage.store_bytes(b"a")
        storage.store_bytes(b"b")
        assert len(storage) == 2
        count = storage.clear()
        assert count == 2
        assert len(storage) == 0

    def test_load_with_loader(self) -> None:
        def loader(path: str) -> bytes:
            with open(path, "rb") as f:
                return f.read()

        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage(loader=loader)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"loaded content")
            path = f.name
        try:
            result = storage.load(path)
            assert result is not None
            h, data = result
            assert data == b"loaded content"
        finally:
            os.unlink(path)

    def test_load_without_loader_raises(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        with pytest.raises(RuntimeError, match="No loader function"):
            storage.load("/some/path.dat")

    def test_load_deduplicates(self) -> None:
        def loader(path: str) -> bytes:
            with open(path, "rb") as f:
                return f.read()

        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage(loader=loader)
        with tempfile.TemporaryDirectory() as d:
            content = b"dedup load"
            p1 = Path(d) / "a.dat"
            p2 = Path(d) / "b.dat"
            p1.write_bytes(content)
            p2.write_bytes(content)

            r1 = storage.load(str(p1))
            r2 = storage.load(str(p2))

            assert r1 is not None and r2 is not None
            assert r1[0] == r2[0]  # Same hash
            assert len(storage) == 1

    def test_iterators(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        storage.store_bytes(b"a", virtual_path="path/a")
        storage.store_bytes(b"b", virtual_path="path/b")

        hashes = list(storage.hashes())
        assert len(hashes) == 2

        paths = list(storage.paths())
        assert len(paths) == 2

        items = list(storage.items())
        assert len(items) == 2

    def test_get_nonexistent_returns_none(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        null_hash = ContentHash.null()
        assert storage.get_by_hash(null_hash) is None
        assert storage.get_by_path("/nonexistent") is None
        assert storage.get_hash_for_path("/nonexistent") is None

    def test_release_nonexistent_returns_false(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        null_hash = ContentHash.null()
        assert not storage.release(null_hash)
        assert not storage.release_by_path("/nonexistent")


# ==============================================================================
# Thread Safety Tests
# ==============================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_hash_cache_concurrent_access(self) -> None:
        cache = HashCache()
        errors: list[Exception] = []

        with tempfile.TemporaryDirectory() as d:
            # Create test files
            paths = []
            for i in range(10):
                p = Path(d) / f"file{i}.txt"
                p.write_bytes(f"content {i}".encode())
                paths.append(p)

            def worker() -> None:
                try:
                    for p in paths:
                        cache.get_or_compute(p)
                        cache.get(p)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
            assert len(cache) == 10

    def test_storage_concurrent_access(self) -> None:
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        errors: list[Exception] = []

        with tempfile.TemporaryDirectory() as d:
            # Create test files
            paths = []
            for i in range(10):
                p = Path(d) / f"file{i}.txt"
                content = f"content {i}".encode()
                p.write_bytes(content)
                paths.append((p, content))

            def worker() -> None:
                try:
                    for p, content in paths:
                        storage.store(str(p), content)
                        storage.get_by_path(str(p))
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
            # Each file stored once (deduplicated across threads)
            assert len(storage) == 10


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_storage_uses_provided_cache(self) -> None:
        """Verify ContentAddressedStorage uses the provided HashCache."""
        cache = HashCache()

        def loader(path: str) -> bytes:
            with open(path, "rb") as f:
                return f.read()

        # Test: verify that the hash_cache parameter is being used
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage(
            hash_cache=cache,
            loader=loader,
        )

        # Verify they share the same cache instance
        assert storage._hash_cache is cache

        # Store something and verify cache is populated
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            path = f.name
        try:
            result = storage.load(path)
            assert result is not None
            assert len(cache) == 1, f"Cache should have 1 entry, got {len(cache)}"
        finally:
            os.unlink(path)

    def test_storage_default_cache(self) -> None:
        """Verify ContentAddressedStorage creates default cache when none provided."""
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        assert storage._hash_cache is not None
        assert isinstance(storage._hash_cache, HashCache)

    def test_full_workflow(self) -> None:
        """Test complete content-addressed asset workflow."""
        with tempfile.TemporaryDirectory() as d:
            cache_path = Path(d) / ".hash_cache"
            asset_dir = Path(d) / "assets"
            asset_dir.mkdir()

            # Create some assets (with duplicates)
            (asset_dir / "texture1.png").write_bytes(b"PNG texture data")
            (asset_dir / "texture2.png").write_bytes(b"PNG texture data")  # Duplicate
            (asset_dir / "mesh.obj").write_bytes(b"OBJ mesh data")

            # Store resolved paths before any operations
            texture1_path = str((asset_dir / "texture1.png").resolve())

            # Create hash cache with persistence path
            cache = HashCache(cache_path=cache_path)

            # Create storage with loader using the provided cache
            def loader(path: str) -> bytes:
                with open(path, "rb") as f:
                    return f.read()

            storage: ContentAddressedStorage[bytes] = ContentAddressedStorage(
                hash_cache=cache,
                loader=loader,
            )

            # Verify cache is shared
            assert storage._hash_cache is cache

            # Load assets
            r1 = storage.load(str(asset_dir / "texture1.png"))
            r2 = storage.load(str(asset_dir / "texture2.png"))
            r3 = storage.load(str(asset_dir / "mesh.obj"))

            assert r1 is not None and r2 is not None and r3 is not None

            # Verify deduplication
            assert r1[0] == r2[0]  # Same hash for textures
            assert r1[0] != r3[0]  # Different from mesh
            assert len(storage) == 2  # Only 2 unique assets

            # Check stats
            stats = storage.get_stats()
            assert stats["unique_assets"] == 2
            assert stats["total_paths"] == 3
            assert stats["duplicate_groups"] == 1

            # Check cache state (should have 3 entries for 3 files)
            assert len(cache) == 3, f"Cache should have 3 entries, got {len(cache)}"

            # Save cache
            cache.save()
            assert cache_path.exists(), "Cache file should exist after save"

            # Verify the cache can be read from directly
            h_direct = cache.get(texture1_path)
            assert h_direct is not None, "Original cache should return hash"
            assert h_direct == r1[0]

            # Create new cache instance and verify persistence
            cache2 = HashCache(cache_path=cache_path)
            assert len(cache2) == 3, f"Cache2 should have 3 entries, got {len(cache2)}"
            h1 = cache2.get(texture1_path)
            assert h1 is not None
            assert h1 == r1[0]

    def test_hash_stability_across_sessions(self) -> None:
        """Verify same content produces same hash across sessions."""
        content = b"stable content for testing"

        # Session 1
        h1 = ContentHash.from_content(content)

        # Session 2 (simulated with new hasher)
        hasher = AssetHasher()
        h2 = hasher.hash_bytes(content)

        # Session 3 (via storage)
        storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        h3 = storage.store_bytes(content)

        assert h1 == h2 == h3

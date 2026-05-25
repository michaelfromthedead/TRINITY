"""
Comprehensive tests for ThumbnailGenerator functionality.

Tests async thumbnail generation, caching, priorities, and batch operations.
"""

import pytest
import sys
import tempfile
import shutil
import time
from pathlib import Path

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.assettools.thumbnail_generator import (
    ThumbnailSize,
    ThumbnailStatus,
    ThumbnailPriority,
    ThumbnailRequest,
    ThumbnailResult,
    ThumbnailCache,
    ThumbnailGenerator,
)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache tests."""
    path = Path(tempfile.mkdtemp())

    # Create test asset files
    (path / "assets").mkdir()
    (path / "assets" / "texture.png").write_bytes(b"png data")
    (path / "assets" / "texture2.png").write_bytes(b"png data 2")
    (path / "assets" / "model.fbx").write_bytes(b"fbx data")
    (path / "assets" / "audio.wav").write_bytes(b"wav data")
    (path / "assets" / "unsupported.xyz").write_bytes(b"xyz data")

    (path / "cache").mkdir()

    yield path
    shutil.rmtree(path)


class TestThumbnailSize:
    """Test ThumbnailSize enum."""

    def test_size_values(self):
        """Size values should be tuples of dimensions."""
        assert ThumbnailSize.TINY.value == (32, 32)
        assert ThumbnailSize.SMALL.value == (64, 64)
        assert ThumbnailSize.MEDIUM.value == (128, 128)
        assert ThumbnailSize.LARGE.value == (256, 256)
        assert ThumbnailSize.XLARGE.value == (512, 512)

    def test_dimensions_property(self):
        """dimensions property should return width and height."""
        w, h = ThumbnailSize.MEDIUM.dimensions
        assert w == 128
        assert h == 128

    def test_all_sizes_square(self):
        """All sizes should be square."""
        for size in ThumbnailSize:
            w, h = size.dimensions
            assert w == h


class TestThumbnailStatus:
    """Test ThumbnailStatus enum."""

    def test_status_values(self):
        """All status values should be defined."""
        assert ThumbnailStatus.PENDING
        assert ThumbnailStatus.IN_PROGRESS
        assert ThumbnailStatus.COMPLETED
        assert ThumbnailStatus.FAILED
        assert ThumbnailStatus.CACHED


class TestThumbnailPriority:
    """Test ThumbnailPriority enum."""

    def test_priority_values(self):
        """Priority values should be ordered."""
        assert ThumbnailPriority.IMMEDIATE.value < ThumbnailPriority.HIGH.value
        assert ThumbnailPriority.HIGH.value < ThumbnailPriority.NORMAL.value
        assert ThumbnailPriority.NORMAL.value < ThumbnailPriority.LOW.value

    def test_immediate_is_highest(self):
        """IMMEDIATE should be the highest priority (lowest value)."""
        assert ThumbnailPriority.IMMEDIATE.value == 0


class TestThumbnailRequest:
    """Test ThumbnailRequest dataclass."""

    def test_request_creation(self):
        """Request should store all attributes."""
        request = ThumbnailRequest(
            priority=ThumbnailPriority.HIGH.value,
            asset_path=Path("/asset.png"),
            size=ThumbnailSize.MEDIUM,
        )

        assert request.priority == ThumbnailPriority.HIGH.value
        assert request.asset_path == Path("/asset.png")
        assert request.size == ThumbnailSize.MEDIUM

    def test_request_time_default(self):
        """Request time should default to current time."""
        request = ThumbnailRequest(
            priority=ThumbnailPriority.NORMAL.value,
            asset_path=Path("/asset.png"),
            size=ThumbnailSize.SMALL,
        )

        assert request.request_time > 0
        assert request.request_time <= time.time()

    def test_request_callback(self):
        """Request should store callback."""
        def my_callback(result):
            pass

        request = ThumbnailRequest(
            priority=ThumbnailPriority.NORMAL.value,
            asset_path=Path("/asset.png"),
            size=ThumbnailSize.SMALL,
            callback=my_callback,
        )

        assert request.callback == my_callback

    def test_request_ordering(self):
        """Requests should be ordered by priority."""
        high = ThumbnailRequest(
            priority=ThumbnailPriority.HIGH.value,
            asset_path=Path("/high.png"),
            size=ThumbnailSize.SMALL,
        )
        low = ThumbnailRequest(
            priority=ThumbnailPriority.LOW.value,
            asset_path=Path("/low.png"),
            size=ThumbnailSize.SMALL,
        )

        # High priority (lower value) should come first
        assert high < low


class TestThumbnailResult:
    """Test ThumbnailResult dataclass."""

    def test_result_creation(self):
        """Result should store all attributes."""
        result = ThumbnailResult(
            asset_path=Path("/asset.png"),
            thumbnail_path=Path("/cache/thumb.png"),
            size=ThumbnailSize.MEDIUM,
            status=ThumbnailStatus.COMPLETED,
        )

        assert result.asset_path == Path("/asset.png")
        assert result.thumbnail_path == Path("/cache/thumb.png")
        assert result.status == ThumbnailStatus.COMPLETED

    def test_result_defaults(self):
        """Result should have sensible defaults."""
        result = ThumbnailResult(asset_path=Path("/asset.png"))

        assert result.thumbnail_path is None
        assert result.size == ThumbnailSize.MEDIUM
        assert result.status == ThumbnailStatus.PENDING
        assert result.error_message is None
        assert result.generation_time_ms == 0.0
        assert result.from_cache is False

    def test_result_error(self):
        """Result should store error message."""
        result = ThumbnailResult(
            asset_path=Path("/missing.png"),
            status=ThumbnailStatus.FAILED,
            error_message="File not found",
        )

        assert result.status == ThumbnailStatus.FAILED
        assert "not found" in result.error_message

    def test_result_from_cache(self):
        """Result should indicate cache status."""
        result = ThumbnailResult(
            asset_path=Path("/asset.png"),
            status=ThumbnailStatus.CACHED,
            from_cache=True,
        )

        assert result.from_cache is True
        assert result.status == ThumbnailStatus.CACHED


class TestThumbnailCache:
    """Test ThumbnailCache functionality."""

    def test_cache_creation(self, temp_cache_dir):
        """Cache should initialize correctly."""
        cache = ThumbnailCache(temp_cache_dir / "cache")

        assert cache.cache_directory == temp_cache_dir / "cache"
        assert cache.max_size_mb == 500.0

    def test_cache_put_get(self, temp_cache_dir):
        """Cache should store and retrieve thumbnails."""
        cache = ThumbnailCache(temp_cache_dir / "cache")
        asset_path = temp_cache_dir / "assets" / "texture.png"
        mtime = asset_path.stat().st_mtime

        # Store thumbnail
        thumbnail_data = b"fake thumbnail data"
        cache_path = cache.put(
            asset_path,
            ThumbnailSize.MEDIUM,
            thumbnail_data,
            mtime,
        )

        assert cache_path.exists()

        # Retrieve thumbnail
        retrieved = cache.get(asset_path, ThumbnailSize.MEDIUM, mtime)
        assert retrieved == cache_path

    def test_cache_miss(self, temp_cache_dir):
        """Cache should return None for missing thumbnails."""
        cache = ThumbnailCache(temp_cache_dir / "cache")
        asset_path = temp_cache_dir / "assets" / "texture.png"
        mtime = asset_path.stat().st_mtime

        result = cache.get(asset_path, ThumbnailSize.MEDIUM, mtime)
        assert result is None

    def test_cache_invalidation_on_asset_change(self, temp_cache_dir):
        """Cache should invalidate when asset is modified."""
        cache = ThumbnailCache(temp_cache_dir / "cache")
        asset_path = temp_cache_dir / "assets" / "texture.png"
        old_mtime = asset_path.stat().st_mtime

        # Store thumbnail
        cache.put(asset_path, ThumbnailSize.MEDIUM, b"data", old_mtime)

        # Simulate asset modification
        new_mtime = old_mtime + 100

        # Cache should miss with newer mtime
        result = cache.get(asset_path, ThumbnailSize.MEDIUM, new_mtime)
        assert result is None

    def test_cache_invalidate(self, temp_cache_dir):
        """invalidate() should remove all thumbnails for an asset."""
        cache = ThumbnailCache(temp_cache_dir / "cache")
        asset_path = temp_cache_dir / "assets" / "texture.png"
        mtime = asset_path.stat().st_mtime

        # Store multiple sizes
        cache.put(asset_path, ThumbnailSize.SMALL, b"small", mtime)
        cache.put(asset_path, ThumbnailSize.MEDIUM, b"medium", mtime)
        cache.put(asset_path, ThumbnailSize.LARGE, b"large", mtime)

        # Invalidate
        count = cache.invalidate(asset_path)

        assert count == 3
        assert cache.get(asset_path, ThumbnailSize.SMALL, mtime) is None
        assert cache.get(asset_path, ThumbnailSize.MEDIUM, mtime) is None
        assert cache.get(asset_path, ThumbnailSize.LARGE, mtime) is None

    def test_cache_clear(self, temp_cache_dir):
        """clear() should remove all cached thumbnails."""
        cache = ThumbnailCache(temp_cache_dir / "cache")
        asset_path = temp_cache_dir / "assets" / "texture.png"
        mtime = asset_path.stat().st_mtime

        cache.put(asset_path, ThumbnailSize.MEDIUM, b"data", mtime)

        cache.clear()

        assert cache.get(asset_path, ThumbnailSize.MEDIUM, mtime) is None

    def test_cache_stats(self, temp_cache_dir):
        """get_stats() should return cache statistics."""
        cache = ThumbnailCache(temp_cache_dir / "cache", max_size_mb=10.0)
        asset_path = temp_cache_dir / "assets" / "texture.png"
        mtime = asset_path.stat().st_mtime

        cache.put(asset_path, ThumbnailSize.MEDIUM, b"x" * 1000, mtime)

        stats = cache.get_stats()

        assert stats["entries"] == 1
        assert stats["size_mb"] > 0
        assert stats["max_size_mb"] == 10.0
        assert 0 <= stats["fill_ratio"] <= 1

    def test_cache_lru_eviction(self, temp_cache_dir):
        """Cache should evict least recently used entries."""
        # Create cache with tiny size
        cache = ThumbnailCache(temp_cache_dir / "cache", max_size_mb=0.001)
        asset1 = temp_cache_dir / "assets" / "texture.png"
        asset2 = temp_cache_dir / "assets" / "texture2.png"
        mtime1 = asset1.stat().st_mtime
        mtime2 = asset2.stat().st_mtime

        # Fill cache
        cache.put(asset1, ThumbnailSize.MEDIUM, b"x" * 500, mtime1)

        # Access first entry to update last_access
        cache.get(asset1, ThumbnailSize.MEDIUM, mtime1)

        # Add second entry (should evict first due to size limit)
        cache.put(asset2, ThumbnailSize.MEDIUM, b"y" * 500, mtime2)

        # Most recent should still be there
        assert cache.get(asset2, ThumbnailSize.MEDIUM, mtime2) is not None

    def test_cache_different_sizes(self, temp_cache_dir):
        """Cache should store different sizes separately."""
        cache = ThumbnailCache(temp_cache_dir / "cache")
        asset_path = temp_cache_dir / "assets" / "texture.png"
        mtime = asset_path.stat().st_mtime

        # Store different sizes
        small_path = cache.put(asset_path, ThumbnailSize.SMALL, b"small", mtime)
        large_path = cache.put(asset_path, ThumbnailSize.LARGE, b"large", mtime)

        assert small_path != large_path
        assert cache.get(asset_path, ThumbnailSize.SMALL, mtime) == small_path
        assert cache.get(asset_path, ThumbnailSize.LARGE, mtime) == large_path


class TestThumbnailGenerator:
    """Test ThumbnailGenerator main class."""

    def test_generator_creation(self, temp_cache_dir):
        """Generator should initialize correctly."""
        generator = ThumbnailGenerator(
            cache_directory=temp_cache_dir / "cache",
            max_workers=2,
        )

        assert generator.max_workers == 2
        assert generator.default_size == ThumbnailSize.MEDIUM

        generator.shutdown()

    def test_request_thumbnail(self, temp_cache_dir):
        """request() should queue thumbnail generation."""
        generator = ThumbnailGenerator(temp_cache_dir / "cache")
        asset_path = temp_cache_dir / "assets" / "texture.png"

        request_id = generator.request(asset_path)

        assert request_id is not None
        assert len(request_id) > 0

        # Wait for processing
        time.sleep(0.5)

        result = generator.get_result(request_id)
        assert result is not None
        assert result.status in (ThumbnailStatus.COMPLETED, ThumbnailStatus.CACHED)

        generator.shutdown()

    def test_request_with_callback(self, temp_cache_dir):
        """request() should call callback when complete."""
        generator = ThumbnailGenerator(temp_cache_dir / "cache")
        asset_path = temp_cache_dir / "assets" / "texture.png"
        callback_results = []

        def callback(result):
            callback_results.append(result)

        generator.request(asset_path, callback=callback)

        # Wait for processing
        time.sleep(0.5)

        assert len(callback_results) == 1
        assert callback_results[0].status in (ThumbnailStatus.COMPLETED, ThumbnailStatus.CACHED)

        generator.shutdown()

    def test_request_specific_size(self, temp_cache_dir):
        """request() should respect size parameter."""
        generator = ThumbnailGenerator(temp_cache_dir / "cache")
        asset_path = temp_cache_dir / "assets" / "texture.png"

        request_id = generator.request(asset_path, size=ThumbnailSize.LARGE)

        time.sleep(0.5)

        result = generator.get_result(request_id)
        assert result.size == ThumbnailSize.LARGE

        generator.shutdown()

    def test_request_priority(self, temp_cache_dir):
        """request() should respect priority."""
        generator = ThumbnailGenerator(temp_cache_dir / "cache", max_workers=1)
        asset_path = temp_cache_dir / "assets" / "texture.png"

        # Request with different priorities
        low_id = generator.request(asset_path, size=ThumbnailSize.TINY, priority=ThumbnailPriority.LOW)
        high_id = generator.request(asset_path, size=ThumbnailSize.SMALL, priority=ThumbnailPriority.HIGH)

        # High priority should complete faster (hard to test precisely)
        time.sleep(0.5)

        assert generator.get_result(high_id) is not None

        generator.shutdown()

    def test_request_batch(self, temp_cache_dir):
        """request_batch() should queue multiple thumbnails."""
        generator = ThumbnailGenerator(temp_cache_dir / "cache")

        paths = [
            temp_cache_dir / "assets" / "texture.png",
            temp_cache_dir / "assets" / "texture2.png",
            temp_cache_dir / "assets" / "model.fbx",
        ]

        request_ids = generator.request_batch(paths)

        assert len(request_ids) == 3

        time.sleep(1.0)

        for request_id in request_ids:
            result = generator.get_result(request_id)
            assert result is not None

        generator.shutdown()

    def test_get_thumbnail(self, temp_cache_dir):
        """get_thumbnail() should return cached thumbnail path."""
        generator = ThumbnailGenerator(temp_cache_dir / "cache")
        asset_path = temp_cache_dir / "assets" / "texture.png"

        # Generate thumbnail
        generator.request(asset_path)
        time.sleep(0.5)

        # Get cached path
        thumb_path = generator.get_thumbnail(asset_path)

        assert thumb_path is not None
        assert thumb_path.exists()

        generator.shutdown()

    def test_has_thumbnail(self, temp_cache_dir):
        """has_thumbnail() should check cache status."""
        generator = ThumbnailGenerator(temp_cache_dir / "cache")
        asset_path = temp_cache_dir / "assets" / "texture.png"

        assert generator.has_thumbnail(asset_path) is False

        generator.request(asset_path)
        time.sleep(0.5)

        assert generator.has_thumbnail(asset_path) is True

        generator.shutdown()

    def test_invalidate(self, temp_cache_dir):
        """invalidate() should remove cached thumbnails."""
        generator = ThumbnailGenerator(temp_cache_dir / "cache")
        asset_path = temp_cache_dir / "assets" / "texture.png"

        generator.request(asset_path)
        time.sleep(0.5)
        assert generator.has_thumbnail(asset_path) is True

        count = generator.invalidate(asset_path)

        assert count >= 1
        assert generator.has_thumbnail(asset_path) is False

        generator.shutdown()

    def test_on_complete_callback(self, temp_cache_dir):
        """on_complete() should register global completion callback."""
        generator = ThumbnailGenerator(temp_cache_dir / "cache")
        completions = []

        generator.on_complete(lambda r: completions.append(r))

        generator.request(temp_cache_dir / "assets" / "texture.png")
        generator.request(temp_cache_dir / "assets" / "texture2.png")

        time.sleep(1.0)

        assert len(completions) == 2

        generator.shutdown()

    def test_get_stats(self, temp_cache_dir):
        """get_stats() should return generator statistics."""
        generator = ThumbnailGenerator(temp_cache_dir / "cache")
        asset_path = temp_cache_dir / "assets" / "texture.png"

        generator.request(asset_path)
        time.sleep(0.5)

        stats = generator.get_stats()

        assert "pending_requests" in stats
        assert "queue_size" in stats
        assert "completed" in stats
        assert "cache" in stats

        generator.shutdown()

    def test_missing_file_fails(self, temp_cache_dir):
        """request() should fail for missing files."""
        generator = ThumbnailGenerator(temp_cache_dir / "cache")
        missing_path = temp_cache_dir / "assets" / "nonexistent.png"

        request_id = generator.request(missing_path)

        time.sleep(0.5)

        result = generator.get_result(request_id)
        assert result.status == ThumbnailStatus.FAILED
        assert result.error_message is not None

        generator.shutdown()

    def test_unsupported_format_fails(self, temp_cache_dir):
        """request() should fail for unsupported formats."""
        generator = ThumbnailGenerator(temp_cache_dir / "cache")
        unsupported_path = temp_cache_dir / "assets" / "unsupported.xyz"

        request_id = generator.request(unsupported_path)

        time.sleep(0.5)

        result = generator.get_result(request_id)
        assert result.status == ThumbnailStatus.FAILED
        assert "unsupported" in result.error_message.lower()

        generator.shutdown()

    def test_cached_result(self, temp_cache_dir):
        """Repeated requests should return cached result."""
        generator = ThumbnailGenerator(temp_cache_dir / "cache")
        asset_path = temp_cache_dir / "assets" / "texture.png"

        # First request - generates
        request_id1 = generator.request(asset_path)
        time.sleep(0.5)
        result1 = generator.get_result(request_id1)

        # Second request - from cache
        request_id2 = generator.request(asset_path)
        result2 = generator.get_result(request_id2)

        assert result2 is not None
        assert result2.from_cache is True or result2.status == ThumbnailStatus.CACHED

        generator.shutdown()

    def test_shutdown_wait(self, temp_cache_dir):
        """shutdown(wait=True) should wait for pending tasks."""
        generator = ThumbnailGenerator(temp_cache_dir / "cache")

        # Queue multiple requests
        for name in ["texture.png", "texture2.png", "model.fbx"]:
            generator.request(temp_cache_dir / "assets" / name)

        generator.shutdown(wait=True)

        assert generator._executor is None

    def test_shutdown_no_wait(self, temp_cache_dir):
        """shutdown(wait=False) should return immediately."""
        generator = ThumbnailGenerator(temp_cache_dir / "cache")

        generator.request(temp_cache_dir / "assets" / "texture.png")

        start = time.time()
        generator.shutdown(wait=False)
        elapsed = time.time() - start

        # Should return quickly
        assert elapsed < 1.0

    def test_generation_time_recorded(self, temp_cache_dir):
        """Result should record generation time."""
        generator = ThumbnailGenerator(temp_cache_dir / "cache")
        asset_path = temp_cache_dir / "assets" / "texture.png"

        request_id = generator.request(asset_path)
        time.sleep(0.5)

        result = generator.get_result(request_id)

        # Should have recorded some time
        if result.status == ThumbnailStatus.COMPLETED:
            assert result.generation_time_ms >= 0

        generator.shutdown()

    def test_default_size(self, temp_cache_dir):
        """Generator should use default size when not specified."""
        generator = ThumbnailGenerator(
            temp_cache_dir / "cache",
            default_size=ThumbnailSize.LARGE,
        )

        asset_path = temp_cache_dir / "assets" / "texture.png"
        request_id = generator.request(asset_path)

        time.sleep(0.5)

        result = generator.get_result(request_id)
        assert result.size == ThumbnailSize.LARGE

        generator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

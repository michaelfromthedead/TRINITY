"""
Whitebox tests for VOStreaming module.

Tests VOCache LRU behavior, VOStreamManager streaming and preloading,
CachedAudio, StreamHandle, and memory management.
"""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch

from engine.audio.dialogue.vo_streaming import (
    CachedAudio,
    StreamHandle,
    StreamState,
    VOCache,
    VOStreamManager,
)
from engine.audio.dialogue.vo_line import VOLine
from engine.audio.dialogue.config import (
    VO_CACHE_SIZE_MB,
    VO_CACHE_EVICTION_THRESHOLD,
    MAX_CONCURRENT_STREAMS,
)


# =============================================================================
# CachedAudio Tests
# =============================================================================


class TestCachedAudio:
    """Tests for CachedAudio dataclass."""

    def test_initialization(self):
        """Test CachedAudio initializes correctly."""
        cached = CachedAudio(
            asset_id="test.wav",
            size_bytes=1024,
            duration_ms=1000.0,
            data=b"audio data",
        )

        assert cached.asset_id == "test.wav"
        assert cached.size_bytes == 1024
        assert cached.duration_ms == 1000.0
        assert cached.data == b"audio data"
        assert cached.access_count == 0

    def test_access_updates_state(self):
        """Test access() updates last access time and count."""
        cached = CachedAudio(asset_id="test.wav")
        initial_access = cached.last_access_time

        time.sleep(0.01)
        cached.access(time.time())

        assert cached.access_count == 1
        assert cached.last_access_time > initial_access

    def test_age_ms_property(self):
        """Test age_ms property calculation."""
        with patch('engine.audio.dialogue.vo_streaming.time.time') as mock_time:
            mock_time.return_value = 100.0
            cached = CachedAudio(asset_id="test.wav", load_time=100.0)

            mock_time.return_value = 100.5
            assert abs(cached.age_ms - 500.0) < 1.0

    def test_idle_time_ms_property(self):
        """Test idle_time_ms property calculation."""
        with patch('engine.audio.dialogue.vo_streaming.time.time') as mock_time:
            mock_time.return_value = 100.0
            cached = CachedAudio(
                asset_id="test.wav",
                last_access_time=100.0,
            )

            mock_time.return_value = 100.2
            assert abs(cached.idle_time_ms - 200.0) < 1.0


# =============================================================================
# StreamHandle Tests
# =============================================================================


class TestStreamHandle:
    """Tests for StreamHandle dataclass."""

    def test_initialization(self):
        """Test StreamHandle initializes correctly."""
        handle = StreamHandle(
            stream_id="stream_1",
            asset_id="test.wav",
            duration_ms=1000.0,
        )

        assert handle.stream_id == "stream_1"
        assert handle.asset_id == "test.wav"
        assert handle.state == StreamState.IDLE
        assert handle.buffer_fill_percent == 0.0
        assert handle.playback_position_ms == 0.0

    def test_is_ready_property(self):
        """Test is_ready property."""
        handle = StreamHandle(stream_id="s1", asset_id="test.wav")

        assert handle.is_ready is False

        handle.state = StreamState.READY
        assert handle.is_ready is True

    def test_is_streaming_property(self):
        """Test is_streaming property."""
        handle = StreamHandle(stream_id="s1", asset_id="test.wav")

        assert handle.is_streaming is False

        handle.state = StreamState.STREAMING
        assert handle.is_streaming is True

    def test_progress_property(self):
        """Test progress property calculation."""
        handle = StreamHandle(
            stream_id="s1",
            asset_id="test.wav",
            duration_ms=1000.0,
        )
        handle.playback_position_ms = 250.0

        assert handle.progress == 0.25

    def test_progress_zero_duration(self):
        """Test progress with zero duration."""
        handle = StreamHandle(
            stream_id="s1",
            asset_id="test.wav",
            duration_ms=0.0,
        )

        assert handle.progress == 0.0


# =============================================================================
# StreamState Tests
# =============================================================================


class TestStreamState:
    """Tests for StreamState enum."""

    def test_all_states_exist(self):
        """Test all required states are defined."""
        assert StreamState.IDLE.value == "idle"
        assert StreamState.LOADING.value == "loading"
        assert StreamState.BUFFERING.value == "buffering"
        assert StreamState.READY.value == "ready"
        assert StreamState.STREAMING.value == "streaming"
        assert StreamState.PAUSED.value == "paused"
        assert StreamState.COMPLETED.value == "completed"
        assert StreamState.ERROR.value == "error"


# =============================================================================
# VOCache Basic Tests
# =============================================================================


class TestVOCacheBasic:
    """Basic tests for VOCache."""

    def test_initialization(self):
        """Test VOCache initializes correctly."""
        cache = VOCache(max_size_mb=16)

        assert cache.size_bytes == 0
        assert cache.size_mb == 0.0
        assert cache.item_count == 0
        assert cache.fill_percent == 0.0

    def test_put_basic(self):
        """Test basic put operation."""
        cache = VOCache(max_size_mb=16)
        data = b"x" * 1024

        cached = cache.put("test.wav", data, duration_ms=1000.0)

        assert cache.item_count == 1
        assert cache.size_bytes == 1024
        assert cached.asset_id == "test.wav"

    def test_put_updates_existing(self):
        """Test put updates existing entry."""
        cache = VOCache(max_size_mb=16)
        data1 = b"x" * 1024
        data2 = b"y" * 2048

        cache.put("test.wav", data1)
        cache.put("test.wav", data2)

        assert cache.item_count == 1
        assert cache.size_bytes == 2048

    def test_get_existing(self):
        """Test get returns existing entry."""
        cache = VOCache(max_size_mb=16)
        data = b"x" * 1024
        cache.put("test.wav", data)

        cached = cache.get("test.wav")

        assert cached is not None
        assert cached.data == data
        assert cached.access_count == 1

    def test_get_missing(self):
        """Test get returns None for missing entry."""
        cache = VOCache(max_size_mb=16)

        result = cache.get("missing.wav")

        assert result is None

    def test_get_updates_access(self):
        """Test get updates access count and time."""
        cache = VOCache(max_size_mb=16)
        cache.put("test.wav", b"data")

        cache.get("test.wav")
        cache.get("test.wav")
        cached = cache.get("test.wav")

        assert cached.access_count == 3

    def test_remove_existing(self):
        """Test remove removes existing entry."""
        cache = VOCache(max_size_mb=16)
        cache.put("test.wav", b"x" * 1024)

        result = cache.remove("test.wav")

        assert result is True
        assert cache.item_count == 0
        assert cache.size_bytes == 0

    def test_remove_missing(self):
        """Test remove returns False for missing entry."""
        cache = VOCache(max_size_mb=16)

        result = cache.remove("missing.wav")

        assert result is False

    def test_clear(self):
        """Test clear removes all entries."""
        cache = VOCache(max_size_mb=16)
        cache.put("a.wav", b"x" * 100)
        cache.put("b.wav", b"y" * 200)
        cache.put("c.wav", b"z" * 300)

        count = cache.clear()

        assert count == 3
        assert cache.item_count == 0
        assert cache.size_bytes == 0


# =============================================================================
# VOCache LRU Behavior Tests
# =============================================================================


class TestVOCacheLRU:
    """Tests for VOCache LRU eviction behavior."""

    def test_eviction_on_overflow(self):
        """Test eviction when cache exceeds max size."""
        # 1KB cache
        cache = VOCache(max_size_mb=1 / 1024)

        # Add 500 bytes
        cache.put("first.wav", b"x" * 500)
        # Add another 700 bytes - should trigger eviction
        cache.put("second.wav", b"y" * 700)

        # First entry should be evicted
        assert cache.get("first.wav") is None
        assert cache.get("second.wav") is not None

    def test_lru_order_preserved(self):
        """Test LRU order is preserved on access."""
        cache = VOCache(max_size_mb=1 / 1024)

        cache.put("a.wav", b"x" * 300)
        cache.put("b.wav", b"y" * 300)

        # Access 'a' to make it most recently used
        cache.get("a.wav")

        # Add 'c' - should evict 'b' (least recently used)
        cache.put("c.wav", b"z" * 500)

        assert cache.get("a.wav") is not None
        assert cache.get("b.wav") is None
        assert cache.get("c.wav") is not None

    def test_evict_oldest_removes_oldest(self):
        """Test _evict_oldest removes oldest entry."""
        cache = VOCache(max_size_mb=16)

        cache.put("first.wav", b"x" * 100)
        time.sleep(0.01)
        cache.put("second.wav", b"y" * 100)

        cache._evict_oldest()

        assert cache.get("first.wav") is None
        assert cache.get("second.wav") is not None

    def test_evict_oldest_empty_cache(self):
        """Test _evict_oldest on empty cache."""
        cache = VOCache(max_size_mb=16)

        result = cache._evict_oldest()

        assert result is False


# =============================================================================
# VOCache Hit Rate Tests
# =============================================================================


class TestVOCacheHitRate:
    """Tests for VOCache hit rate tracking."""

    def test_hit_rate_initial(self):
        """Test initial hit rate is 0."""
        cache = VOCache(max_size_mb=16)

        assert cache.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        """Test hit rate with all hits."""
        cache = VOCache(max_size_mb=16)
        cache.put("test.wav", b"data")

        cache.get("test.wav")
        cache.get("test.wav")
        cache.get("test.wav")

        assert cache.hit_rate == 1.0

    def test_hit_rate_all_misses(self):
        """Test hit rate with all misses."""
        cache = VOCache(max_size_mb=16)

        cache.get("missing1.wav")
        cache.get("missing2.wav")
        cache.get("missing3.wav")

        assert cache.hit_rate == 0.0

    def test_hit_rate_mixed(self):
        """Test hit rate with mixed hits and misses."""
        cache = VOCache(max_size_mb=16)
        cache.put("test.wav", b"data")

        cache.get("test.wav")  # hit
        cache.get("missing.wav")  # miss
        cache.get("test.wav")  # hit
        cache.get("missing.wav")  # miss

        assert cache.hit_rate == 0.5


# =============================================================================
# VOCache Preloaded Eviction Tests
# =============================================================================


class TestVOCachePreloadedEviction:
    """Tests for VOCache preloaded item eviction."""

    def test_evict_preloaded_basic(self):
        """Test evict_preloaded removes unused preloaded items."""
        cache = VOCache(max_size_mb=16)

        # Add preloaded items
        cache.put("preload1.wav", b"x" * 100, is_preloaded=True)
        cache.put("preload2.wav", b"y" * 100, is_preloaded=True)
        # Add regular item
        cache.put("regular.wav", b"z" * 100, is_preloaded=False)

        count = cache.evict_preloaded()

        assert count == 2
        assert cache.get("preload1.wav") is None
        assert cache.get("regular.wav") is not None

    def test_evict_preloaded_keeps_accessed(self):
        """Test evict_preloaded keeps accessed preloaded items."""
        cache = VOCache(max_size_mb=16)

        cache.put("preload.wav", b"x" * 100, is_preloaded=True)
        # Access the item
        cache.get("preload.wav")
        cache.get("preload.wav")

        count = cache.evict_preloaded()

        # Should not evict because access_count > 1
        assert count == 0
        assert cache.get("preload.wav") is not None


# =============================================================================
# VOCache Check Eviction Tests
# =============================================================================


class TestVOCacheCheckEviction:
    """Tests for VOCache threshold-based eviction."""

    def test_check_eviction_under_threshold(self):
        """Test check_eviction does nothing under threshold."""
        cache = VOCache(max_size_mb=16, eviction_threshold=0.9)
        cache.put("test.wav", b"x" * 100)

        count = cache.check_eviction()

        assert count == 0

    def test_check_eviction_over_threshold(self):
        """Test check_eviction evicts when over threshold."""
        # Use very small cache
        cache = VOCache(max_size_mb=1 / 1024, eviction_threshold=0.5)

        cache.put("a.wav", b"x" * 300)
        cache.put("b.wav", b"y" * 300)
        cache.put("c.wav", b"z" * 300)

        # Cache is now 900 bytes, threshold is 512 bytes
        count = cache.check_eviction()

        assert count > 0
        assert cache.fill_percent <= 0.5


# =============================================================================
# VOCache Stats Tests
# =============================================================================


class TestVOCacheStats:
    """Tests for VOCache statistics."""

    def test_stats_empty_cache(self):
        """Test stats for empty cache."""
        cache = VOCache(max_size_mb=16)
        stats = cache.stats

        assert stats["size_bytes"] == 0
        assert stats["size_mb"] == 0.0
        assert stats["item_count"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0

    def test_stats_with_entries(self):
        """Test stats with entries."""
        cache = VOCache(max_size_mb=16)
        cache.put("a.wav", b"x" * 1024)
        cache.put("b.wav", b"y" * 2048)
        cache.get("a.wav")
        cache.get("missing.wav")

        stats = cache.stats

        assert stats["item_count"] == 2
        assert stats["size_bytes"] == 3072
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


# =============================================================================
# VOStreamManager Basic Tests
# =============================================================================


class TestVOStreamManagerBasic:
    """Basic tests for VOStreamManager."""

    def test_initialization(self):
        """Test VOStreamManager initializes correctly."""
        manager = VOStreamManager()

        assert manager.active_stream_count == 0
        assert manager.can_start_stream is True
        assert manager.cache is not None

    def test_custom_initialization(self):
        """Test VOStreamManager with custom parameters."""
        manager = VOStreamManager(
            cache_size_mb=32,
            max_concurrent_streams=4,
        )

        assert manager._max_concurrent == 4

    def test_start_stream_basic(self):
        """Test basic stream start."""
        manager = VOStreamManager()
        line = VOLine(audio_asset="test.wav", duration_ms=1000.0)

        handle = manager.start_stream(line)

        assert handle is not None
        assert handle.asset_id == "test.wav"

    def test_start_stream_cached(self):
        """Test stream start with cached audio."""
        manager = VOStreamManager()
        manager.cache.put("test.wav", b"audio data", duration_ms=1000.0)

        line = VOLine(audio_asset="test.wav", duration_ms=1000.0)
        handle = manager.start_stream(line)

        assert handle.state == StreamState.READY
        assert handle.buffer_fill_percent == 1.0

    def test_start_stream_at_max(self):
        """Test stream start fails at max concurrent."""
        manager = VOStreamManager(max_concurrent_streams=1)
        line1 = VOLine(audio_asset="test1.wav")
        line2 = VOLine(audio_asset="test2.wav")

        # Start first stream and make it active
        handle1 = manager.start_stream(line1)
        handle1.state = StreamState.STREAMING

        # Should fail to start second
        handle2 = manager.start_stream(line2)

        assert handle2 is None

    def test_start_stream_with_callbacks(self):
        """Test stream start triggers callbacks."""
        on_ready = MagicMock()
        manager = VOStreamManager()
        manager.cache.put("test.wav", b"audio data")

        line = VOLine(audio_asset="test.wav")
        manager.start_stream(line, on_ready=on_ready)

        on_ready.assert_called_once()


# =============================================================================
# VOStreamManager Stream Operations Tests
# =============================================================================


class TestVOStreamManagerOperations:
    """Tests for VOStreamManager stream operations."""

    def test_stop_stream(self):
        """Test stop_stream terminates stream."""
        manager = VOStreamManager()
        line = VOLine(audio_asset="test.wav")
        handle = manager.start_stream(line)

        result = manager.stop_stream(handle.stream_id)

        assert result is True
        assert manager.get_stream(handle.stream_id) is None

    def test_stop_stream_invalid_id(self):
        """Test stop_stream with invalid ID."""
        manager = VOStreamManager()

        result = manager.stop_stream("invalid_id")

        assert result is False

    def test_get_stream(self):
        """Test get_stream retrieves stream."""
        manager = VOStreamManager()
        line = VOLine(audio_asset="test.wav")
        handle = manager.start_stream(line)

        retrieved = manager.get_stream(handle.stream_id)

        assert retrieved is handle

    def test_update_stream(self):
        """Test update_stream advances position."""
        manager = VOStreamManager()
        manager.cache.put("test.wav", b"audio data", duration_ms=1000.0)
        line = VOLine(audio_asset="test.wav", duration_ms=1000.0)
        handle = manager.start_stream(line)
        manager.play_stream(handle.stream_id)

        updated = manager.update_stream(handle.stream_id, 100.0)

        assert updated.playback_position_ms == 100.0

    def test_update_stream_completes(self):
        """Test update_stream completes when finished."""
        on_complete = MagicMock()
        manager = VOStreamManager()
        manager.cache.put("test.wav", b"audio data", duration_ms=100.0)
        line = VOLine(audio_asset="test.wav", duration_ms=100.0)
        handle = manager.start_stream(line, on_complete=on_complete)
        manager.play_stream(handle.stream_id)

        manager.update_stream(handle.stream_id, 200.0)

        on_complete.assert_called_once()

    def test_play_stream(self):
        """Test play_stream starts streaming."""
        manager = VOStreamManager()
        manager.cache.put("test.wav", b"audio data")
        line = VOLine(audio_asset="test.wav")
        handle = manager.start_stream(line)

        result = manager.play_stream(handle.stream_id)

        assert result is True
        assert handle.state == StreamState.STREAMING

    def test_play_stream_not_ready(self):
        """Test play_stream fails when not ready."""
        manager = VOStreamManager()
        line = VOLine(audio_asset="test.wav")
        handle = manager.start_stream(line)
        handle.state = StreamState.LOADING

        result = manager.play_stream(handle.stream_id)

        assert result is False

    def test_pause_stream(self):
        """Test pause_stream pauses streaming."""
        manager = VOStreamManager()
        manager.cache.put("test.wav", b"audio data")
        line = VOLine(audio_asset="test.wav")
        handle = manager.start_stream(line)
        manager.play_stream(handle.stream_id)

        result = manager.pause_stream(handle.stream_id)

        assert result is True
        assert handle.state == StreamState.PAUSED

    def test_resume_stream(self):
        """Test resume_stream resumes streaming."""
        manager = VOStreamManager()
        manager.cache.put("test.wav", b"audio data")
        line = VOLine(audio_asset="test.wav")
        handle = manager.start_stream(line)
        manager.play_stream(handle.stream_id)
        manager.pause_stream(handle.stream_id)

        result = manager.resume_stream(handle.stream_id)

        assert result is True
        assert handle.state == StreamState.STREAMING


# =============================================================================
# VOStreamManager Preloading Tests
# =============================================================================


class TestVOStreamManagerPreloading:
    """Tests for VOStreamManager preloading functionality."""

    def test_preload_basic(self):
        """Test basic preload operation."""
        def mock_loader(asset_id):
            return (b"audio data", 1000.0)

        manager = VOStreamManager(audio_loader=mock_loader)

        result = manager.preload("test.wav")

        assert result is True

    def test_preload_already_cached(self):
        """Test preload returns True for cached item."""
        manager = VOStreamManager()
        manager.cache.put("test.wav", b"audio data")

        result = manager.preload("test.wav")

        assert result is True

    def test_preload_already_preloading(self):
        """Test preload returns True for in-progress preload."""
        manager = VOStreamManager()
        manager._preloading.add("test.wav")

        result = manager.preload("test.wav")

        assert result is True

    def test_preload_line(self):
        """Test preload_line preloads line's audio."""
        def mock_loader(asset_id):
            return (b"audio data", 1000.0)

        manager = VOStreamManager(audio_loader=mock_loader)
        line = VOLine(audio_asset="test.wav")

        result = manager.preload_line(line)

        assert result is True

    def test_preload_lines(self):
        """Test preload_lines preloads multiple lines."""
        def mock_loader(asset_id):
            return (b"audio data", 1000.0)

        manager = VOStreamManager(audio_loader=mock_loader)
        lines = [
            VOLine(audio_asset="a.wav"),
            VOLine(audio_asset="b.wav"),
            VOLine(audio_asset="c.wav"),
        ]

        count = manager.preload_lines(lines)

        assert count == 3

    def test_set_anticipated_lines(self):
        """Test set_anticipated_lines sets and preloads lines."""
        def mock_loader(asset_id):
            return (b"audio data", 1000.0)

        manager = VOStreamManager(audio_loader=mock_loader)
        lines = [VOLine(audio_asset=f"{i}.wav") for i in range(5)]

        manager.set_anticipated_lines(lines)

        assert len(manager._anticipated) == 5

    def test_cancel_preload(self):
        """Test cancel_preload cancels pending preload."""
        manager = VOStreamManager()
        manager._preload_queue.append("test.wav")
        manager._preloading.add("test.wav")

        result = manager.cancel_preload("test.wav")

        assert result is True
        assert "test.wav" not in manager._preload_queue
        assert "test.wav" not in manager._preloading

    def test_clear_preload_queue(self):
        """Test clear_preload_queue clears all pending."""
        manager = VOStreamManager()
        manager._preload_queue.extend(["a.wav", "b.wav", "c.wav"])
        manager._preloading.update(["a.wav", "b.wav", "c.wav"])

        count = manager.clear_preload_queue()

        assert count == 3
        assert len(manager._preload_queue) == 0
        assert len(manager._preloading) == 0


# =============================================================================
# VOStreamManager Memory Management Tests
# =============================================================================


class TestVOStreamManagerMemory:
    """Tests for VOStreamManager memory management."""

    def test_get_memory_usage(self):
        """Test get_memory_usage returns stats."""
        manager = VOStreamManager()
        manager.cache.put("test.wav", b"x" * 1024)

        usage = manager.get_memory_usage()

        assert "cache" in usage
        assert usage["cache"]["size_bytes"] == 1024
        assert "active_streams" in usage
        assert "preload_queue_size" in usage

    def test_trim_cache(self):
        """Test trim_cache reduces cache size."""
        manager = VOStreamManager(cache_size_mb=1 / 1024)

        # Fill cache
        manager.cache.put("a.wav", b"x" * 200, is_preloaded=True)
        manager.cache.put("b.wav", b"y" * 200, is_preloaded=True)
        manager.cache.put("c.wav", b"z" * 200)

        count = manager.trim_cache(target_percent=0.3)

        assert count >= 0
        # Cache should be reduced

    def test_clear_cache(self):
        """Test clear_cache clears entire cache."""
        manager = VOStreamManager()
        manager.cache.put("a.wav", b"x" * 100)
        manager.cache.put("b.wav", b"y" * 100)

        count = manager.clear_cache()

        assert count == 2
        assert manager.cache.item_count == 0


# =============================================================================
# VOStreamManager Stats Tests
# =============================================================================


class TestVOStreamManagerStats:
    """Tests for VOStreamManager statistics."""

    def test_stats_empty(self):
        """Test stats for empty manager."""
        manager = VOStreamManager()
        stats = manager.stats

        assert stats["active_streams"] == 0
        assert "max_concurrent" in stats
        assert "preload_queue" in stats
        assert "cache" in stats

    def test_stats_with_streams(self):
        """Test stats with active streams."""
        manager = VOStreamManager()
        manager.cache.put("test.wav", b"audio data")
        line = VOLine(audio_asset="test.wav")
        manager.start_stream(line)

        stats = manager.stats

        assert stats["active_streams"] == 1


# =============================================================================
# VOStreamManager Audio Loading Tests
# =============================================================================


class TestVOStreamManagerAudioLoading:
    """Tests for VOStreamManager audio loading."""

    def test_load_audio_success(self):
        """Test _load_audio successfully loads and caches."""
        def mock_loader(asset_id):
            return (b"audio data", 1000.0)

        manager = VOStreamManager(audio_loader=mock_loader)
        handle = StreamHandle(stream_id="s1", asset_id="test.wav")

        manager._load_audio("test.wav", handle)

        assert handle.state == StreamState.READY
        assert handle.buffer_fill_percent == 1.0
        assert manager.cache.get("test.wav") is not None

    def test_load_audio_error(self):
        """Test _load_audio handles errors."""
        def mock_loader(asset_id):
            raise Exception("Load failed")

        on_error = MagicMock()
        manager = VOStreamManager(
            audio_loader=mock_loader,
            on_stream_error=on_error,
        )
        handle = StreamHandle(
            stream_id="s1",
            asset_id="test.wav",
            on_error=MagicMock(),
        )

        manager._load_audio("test.wav", handle)

        assert handle.state == StreamState.ERROR
        handle.on_error.assert_called_once()
        on_error.assert_called_once()

    def test_load_audio_no_loader(self):
        """Test _load_audio without loader simulates loading."""
        manager = VOStreamManager()
        handle = StreamHandle(stream_id="s1", asset_id="test.wav")

        manager._load_audio("test.wav", handle)

        # Should simulate loading
        assert handle.state == StreamState.READY


# =============================================================================
# VOStreamManager Thread Safety Tests
# =============================================================================


class TestVOStreamManagerThreadSafety:
    """Thread safety tests for VOStreamManager."""

    def test_concurrent_start_stream(self):
        """Test concurrent stream starts."""
        manager = VOStreamManager(max_concurrent_streams=20)
        handles = []

        def start_streams():
            for i in range(10):
                line = VOLine(audio_asset=f"test_{threading.current_thread().name}_{i}.wav")
                handle = manager.start_stream(line)
                if handle:
                    handles.append(handle)
                time.sleep(0.001)

        threads = [threading.Thread(target=start_streams) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have started multiple streams
        assert len(handles) > 0

    def test_concurrent_cache_operations(self):
        """Test concurrent cache operations."""
        manager = VOStreamManager()

        def cache_operations():
            for i in range(20):
                key = f"audio_{i}.wav"
                manager.cache.put(key, b"x" * 100)
                manager.cache.get(key)
                time.sleep(0.001)

        threads = [threading.Thread(target=cache_operations) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without deadlock

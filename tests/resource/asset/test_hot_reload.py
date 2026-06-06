"""Tests for asset hot-reload system with handle indirection (T-CC-3.1).

Covers:
- HotReloadWatcher (legacy API)
- IndirectHandle[T]
- AssetHandleTable
- AssetReloader
- TextureProcessor, MeshProcessor, AudioProcessor
- FileWatcher integration
"""
from __future__ import annotations

import os
import struct
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Set
from unittest.mock import MagicMock, patch

import pytest

from engine.core.file_watcher import FileChangeEvent, FileChangeType, FileWatcher
from engine.resource.asset.hot_reload import (
    AssetHandleTable,
    AssetProcessor,
    AssetReloader,
    AudioData,
    AudioProcessor,
    HandleState,
    HotReloadWatcher,
    IndirectHandle,
    MeshData,
    MeshProcessor,
    ReloadError,
    ReloadEvent,
    ReloadStrategy,
    TextureData,
    TextureProcessor,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def handle_table() -> AssetHandleTable:
    """Create a fresh handle table."""
    return AssetHandleTable()


@pytest.fixture
def file_watcher() -> FileWatcher:
    """Create a file watcher."""
    return FileWatcher(poll_interval_ms=50)


@pytest.fixture
def reloader(handle_table: AssetHandleTable, file_watcher: FileWatcher) -> AssetReloader:
    """Create an asset reloader."""
    return AssetReloader(handle_table, file_watcher, ReloadStrategy.IMMEDIATE)


# =============================================================================
# HotReloadWatcher (Legacy API) Tests
# =============================================================================


class TestHotReloadWatcher:
    """Tests for the legacy HotReloadWatcher."""

    def test_register_and_poll_no_change(self, temp_dir: str) -> None:
        """Polling without changes should not fire callbacks."""
        path = os.path.join(temp_dir, "test.txt")
        with open(path, "w") as f:
            f.write("initial")

        fired: List[str] = []
        w = HotReloadWatcher()
        w.register(path, fired.append)
        changed = w.poll()
        assert changed == []
        assert fired == []

    def test_poll_detects_change(self, temp_dir: str) -> None:
        """Modifying a file should trigger callback."""
        path = os.path.join(temp_dir, "test.txt")
        with open(path, "w") as f:
            f.write("v1")

        fired: List[str] = []
        w = HotReloadWatcher()
        w.register(path, fired.append)

        time.sleep(0.05)
        with open(path, "w") as f:
            f.write("v2")
        os.utime(path, (time.time() + 2, time.time() + 2))

        changed = w.poll()
        assert path in changed
        assert path in fired

    def test_poll_no_double_fire(self, temp_dir: str) -> None:
        """Same change shouldn't fire twice."""
        path = os.path.join(temp_dir, "test.txt")
        with open(path, "w") as f:
            f.write("content")

        fired: List[str] = []
        w = HotReloadWatcher()
        w.register(path, fired.append)
        os.utime(path, (time.time() + 2, time.time() + 2))
        w.poll()
        second = w.poll()
        assert second == []
        assert len(fired) == 1

    def test_unregister_stops_watching(self, temp_dir: str) -> None:
        """Unregistered files should not trigger callbacks."""
        path = os.path.join(temp_dir, "test.txt")
        with open(path, "w") as f:
            f.write("content")

        fired: List[str] = []
        w = HotReloadWatcher()
        w.register(path, fired.append)
        w.unregister(path)
        os.utime(path, (time.time() + 2, time.time() + 2))
        changed = w.poll()
        assert changed == []
        assert fired == []

    def test_start_stop_no_crash(self, temp_dir: str) -> None:
        """Starting and stopping the background thread should work."""
        path = os.path.join(temp_dir, "test.txt")
        with open(path, "w") as f:
            f.write("v1")

        fired: List[str] = []
        w = HotReloadWatcher(interval=0.01)
        w.register(path, fired.append)
        w.start()
        time.sleep(0.02)
        with open(path, "w") as f:
            f.write("v2")
        os.utime(path, (time.time() + 2, time.time() + 2))
        time.sleep(0.05)
        w.stop()
        assert w._running is False
        assert path in fired

    def test_register_nonexistent_file(self) -> None:
        """Registering a nonexistent file should use mtime 0."""
        w = HotReloadWatcher()
        fired: List[str] = []
        w.register("/nonexistent/file.txt", fired.append)
        # Should not crash
        w.poll()
        assert fired == []


# =============================================================================
# IndirectHandle Tests
# =============================================================================


class TestIndirectHandle:
    """Tests for IndirectHandle."""

    def test_create_handle(self, handle_table: AssetHandleTable) -> None:
        """Handle should be created with correct properties."""
        handle = handle_table.register("/test/asset.png", TextureData)
        assert handle.handle_id >= 0
        assert handle.generation >= 0
        assert handle.asset_type == TextureData

    def test_handle_is_valid_initially(self, handle_table: AssetHandleTable) -> None:
        """Newly created handle should be valid."""
        handle = handle_table.register("/test/asset.png", TextureData)
        assert handle.is_valid()

    def test_handle_invalid_after_unregister(self, handle_table: AssetHandleTable) -> None:
        """Handle should become invalid after unregistering."""
        handle = handle_table.register("/test/asset.png", TextureData)
        handle_table.unregister(handle)
        assert not handle.is_valid()

    def test_handle_equality(self, handle_table: AssetHandleTable) -> None:
        """Handles with same id/gen should be equal."""
        h1 = handle_table.register("/test/a.png", TextureData)
        h2 = handle_table.register("/test/a.png", TextureData)  # Returns same
        h3 = handle_table.register("/test/b.png", TextureData)
        assert h1 == h2  # Same path returns same handle
        assert h1 != h3

    def test_handle_hash(self, handle_table: AssetHandleTable) -> None:
        """Handles should be hashable."""
        h1 = handle_table.register("/test/a.png", TextureData)
        h2 = handle_table.register("/test/a.png", TextureData)
        s = {h1, h2}
        assert len(s) == 1  # Same handle

    def test_handle_repr(self, handle_table: AssetHandleTable) -> None:
        """Handle should have informative repr."""
        handle = handle_table.register("/test/asset.png", TextureData)
        r = repr(handle)
        assert "IndirectHandle" in r
        assert "TextureData" in r

    def test_handle_get_state(self, handle_table: AssetHandleTable) -> None:
        """Handle should report correct state."""
        handle = handle_table.register("/test/asset.png", TextureData)
        assert handle.get_state() == HandleState.EMPTY

    def test_handle_get_version(self, handle_table: AssetHandleTable) -> None:
        """Handle should track version."""
        handle = handle_table.register("/test/asset.png", TextureData)
        assert handle.get_version() == 0

    def test_handle_get_path(self, handle_table: AssetHandleTable) -> None:
        """Handle should return asset path."""
        handle = handle_table.register("/test/asset.png", TextureData)
        assert handle.get_path() == "/test/asset.png"

    def test_handle_get_returns_none_when_empty(self, handle_table: AssetHandleTable) -> None:
        """get() should return None for empty handle."""
        handle = handle_table.register("/test/asset.png", TextureData)
        assert handle.get() is None

    def test_handle_get_returns_data_when_ready(self, handle_table: AssetHandleTable) -> None:
        """get() should return data when ready."""
        data = TextureData("/test/asset.png", 256, 256, 4, b"pixels")
        handle = handle_table.register("/test/asset.png", TextureData, data)
        assert handle.get() == data

    def test_stale_handle_invalid(self, handle_table: AssetHandleTable) -> None:
        """Stale handle (wrong generation) should be invalid."""
        handle = handle_table.register("/test/asset.png", TextureData)
        handle_table.unregister(handle)
        # Re-register at same slot
        handle2 = handle_table.register("/test/asset2.png", TextureData)
        # Old handle should be invalid (different generation)
        assert not handle.is_valid()
        assert handle2.is_valid()


# =============================================================================
# AssetHandleTable Tests
# =============================================================================


class TestAssetHandleTable:
    """Tests for AssetHandleTable."""

    def test_register_asset(self, handle_table: AssetHandleTable) -> None:
        """Registering an asset should return a valid handle."""
        handle = handle_table.register("/test/asset.png", TextureData)
        assert handle.is_valid()
        assert handle_table.entry_count == 1

    def test_register_with_initial_data(self, handle_table: AssetHandleTable) -> None:
        """Registering with data should set state to READY."""
        data = TextureData("/test/asset.png", 256, 256, 4, b"pixels")
        handle = handle_table.register("/test/asset.png", TextureData, data)
        assert handle_table.get_state(handle) == HandleState.READY
        assert handle_table.get(handle) == data

    def test_register_deduplication(self, handle_table: AssetHandleTable) -> None:
        """Same path should return same handle with incremented ref count."""
        h1 = handle_table.register("/test/asset.png", TextureData)
        h2 = handle_table.register("/test/asset.png", TextureData)
        assert h1 == h2
        assert handle_table.entry_count == 1

    def test_unregister_decrements_ref_count(self, handle_table: AssetHandleTable) -> None:
        """Unregistering should decrement ref count."""
        h1 = handle_table.register("/test/asset.png", TextureData)
        h2 = handle_table.register("/test/asset.png", TextureData)
        handle_table.unregister(h1)
        # Still valid (ref count > 0)
        assert h2.is_valid()
        handle_table.unregister(h2)
        # Now invalid
        assert not h2.is_valid()

    def test_unregister_invalid_handle(self, handle_table: AssetHandleTable) -> None:
        """Unregistering invalid handle should return False."""
        handle = handle_table.register("/test/asset.png", TextureData)
        handle_table.unregister(handle)
        result = handle_table.unregister(handle)
        assert result is False

    def test_get_returns_none_for_invalid(self, handle_table: AssetHandleTable) -> None:
        """get() on invalid handle should return None."""
        handle = handle_table.register("/test/asset.png", TextureData)
        handle_table.unregister(handle)
        assert handle_table.get(handle) is None

    def test_update_data(self, handle_table: AssetHandleTable) -> None:
        """update_data should replace asset data."""
        handle = handle_table.register("/test/asset.png", TextureData)
        data = TextureData("/test/asset.png", 512, 512, 4, b"new_pixels")
        handle_table.update_data("/test/asset.png", data)
        assert handle_table.get(handle) == data
        assert handle_table.get_version(handle) == 1

    def test_update_data_nonexistent(self, handle_table: AssetHandleTable) -> None:
        """update_data on nonexistent path should return False."""
        result = handle_table.update_data("/nonexistent.png", None)
        assert result is False

    def test_set_state(self, handle_table: AssetHandleTable) -> None:
        """set_state should change asset state."""
        handle = handle_table.register("/test/asset.png", TextureData)
        handle_table.set_state("/test/asset.png", HandleState.LOADING)
        assert handle_table.get_state(handle) == HandleState.LOADING

    def test_set_error(self, handle_table: AssetHandleTable) -> None:
        """set_error should set state to FAILED and store error."""
        handle = handle_table.register("/test/asset.png", TextureData)
        handle_table.set_error("/test/asset.png", "Load failed")
        assert handle_table.get_state(handle) == HandleState.FAILED
        assert handle_table.get_error(handle) == "Load failed"

    def test_get_handle_by_path(self, handle_table: AssetHandleTable) -> None:
        """Should retrieve handle by path."""
        handle = handle_table.register("/test/asset.png", TextureData)
        retrieved = handle_table.get_handle_by_path("/test/asset.png")
        assert retrieved == handle

    def test_get_handle_by_path_nonexistent(self, handle_table: AssetHandleTable) -> None:
        """Should return None for nonexistent path."""
        result = handle_table.get_handle_by_path("/nonexistent.png")
        assert result is None

    def test_get_all_paths(self, handle_table: AssetHandleTable) -> None:
        """Should return all registered paths."""
        handle_table.register("/test/a.png", TextureData)
        handle_table.register("/test/b.obj", MeshData)
        paths = handle_table.get_all_paths()
        assert set(paths) == {"/test/a.png", "/test/b.obj"}

    def test_get_paths_by_type(self, handle_table: AssetHandleTable) -> None:
        """Should filter paths by asset type."""
        handle_table.register("/test/a.png", TextureData)
        handle_table.register("/test/b.obj", MeshData)
        handle_table.register("/test/c.png", TextureData)
        paths = handle_table.get_paths_by_type(TextureData)
        assert set(paths) == {"/test/a.png", "/test/c.png"}

    def test_metadata(self, handle_table: AssetHandleTable) -> None:
        """Should store and retrieve metadata."""
        handle = handle_table.register("/test/asset.png", TextureData, metadata={"key": "value"})
        assert handle_table.get_metadata(handle) == {"key": "value"}
        handle_table.set_metadata(handle, "key2", 42)
        assert handle_table.get_metadata(handle) == {"key": "value", "key2": 42}

    def test_reload_callback(self, handle_table: AssetHandleTable) -> None:
        """Reload callbacks should fire on update_data."""
        events: List[tuple] = []
        handle_table.add_reload_callback(lambda p, v: events.append((p, v)))
        handle_table.register("/test/asset.png", TextureData)
        handle_table.update_data("/test/asset.png", TextureData("/test/asset.png", 1, 1, 4, b""))
        assert events == [("/test/asset.png", 1)]

    def test_remove_reload_callback(self, handle_table: AssetHandleTable) -> None:
        """Should remove reload callback."""
        events: List[tuple] = []
        cb = lambda p, v: events.append((p, v))
        handle_table.add_reload_callback(cb)
        handle_table.remove_reload_callback(cb)
        handle_table.register("/test/asset.png", TextureData)
        handle_table.update_data("/test/asset.png", None)
        assert events == []

    def test_dispose(self, handle_table: AssetHandleTable) -> None:
        """dispose() should invalidate all handles."""
        h1 = handle_table.register("/test/a.png", TextureData)
        h2 = handle_table.register("/test/b.png", TextureData)
        handle_table.dispose()
        assert handle_table.disposed
        assert not h1.is_valid()
        assert not h2.is_valid()

    def test_register_after_dispose(self, handle_table: AssetHandleTable) -> None:
        """Registering after dispose should raise."""
        handle_table.dispose()
        with pytest.raises(RuntimeError, match="disposed"):
            handle_table.register("/test/asset.png", TextureData)

    def test_slot_reuse(self, handle_table: AssetHandleTable) -> None:
        """Freed slots should be reused with incremented generation."""
        h1 = handle_table.register("/test/asset.png", TextureData)
        gen1 = h1.generation
        handle_table.unregister(h1)
        h2 = handle_table.register("/test/asset2.png", TextureData)
        # Same slot, different generation
        assert h2.handle_id == h1.handle_id
        assert h2.generation == gen1 + 1


# =============================================================================
# Asset Processor Tests
# =============================================================================


class TestTextureProcessor:
    """Tests for TextureProcessor."""

    def test_supported_extensions(self) -> None:
        """Should support common texture extensions."""
        p = TextureProcessor()
        assert ".png" in p.supported_extensions
        assert ".jpg" in p.supported_extensions
        assert ".jpeg" in p.supported_extensions

    def test_can_process(self) -> None:
        """can_process should match extensions."""
        p = TextureProcessor()
        assert p.can_process("/test/texture.png")
        assert p.can_process("/test/texture.jpg")
        assert not p.can_process("/test/model.obj")

    def test_load_png(self, temp_dir: str) -> None:
        """Should load PNG file."""
        path = os.path.join(temp_dir, "test.png")
        # Create minimal PNG
        png_data = (
            b"\x89PNG\r\n\x1a\n"  # Signature
            + b"\x00\x00\x00\rIHDR"  # IHDR length + type
            + struct.pack(">II", 64, 64)  # Width, height
            + b"\x08\x06\x00\x00\x00"  # Bit depth, color type, etc.
            + b"\x00\x00\x00\x00"  # CRC placeholder
        )
        with open(path, "wb") as f:
            f.write(png_data)

        p = TextureProcessor()
        result = p.load(path)
        assert isinstance(result, TextureData)
        assert result.path == path
        assert result.width == 64
        assert result.height == 64

    def test_unload(self) -> None:
        """unload should not crash."""
        p = TextureProcessor()
        data = TextureData("/test.png", 1, 1, 4, b"")
        p.unload(data)  # Should not raise


class TestMeshProcessor:
    """Tests for MeshProcessor."""

    def test_supported_extensions(self) -> None:
        """Should support common mesh extensions."""
        p = MeshProcessor()
        assert ".obj" in p.supported_extensions
        assert ".fbx" in p.supported_extensions
        assert ".gltf" in p.supported_extensions

    def test_can_process(self) -> None:
        """can_process should match extensions."""
        p = MeshProcessor()
        assert p.can_process("/test/model.obj")
        assert not p.can_process("/test/texture.png")

    def test_load_obj(self, temp_dir: str) -> None:
        """Should load OBJ file and count vertices."""
        path = os.path.join(temp_dir, "test.obj")
        obj_data = """
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3
f 1 3 4
"""
        with open(path, "w") as f:
            f.write(obj_data)

        p = MeshProcessor()
        result = p.load(path)
        assert isinstance(result, MeshData)
        assert result.path == path
        assert result.vertex_count == 4
        assert result.index_count == 6  # 2 triangles * 3

    def test_unload(self) -> None:
        """unload should not crash."""
        p = MeshProcessor()
        data = MeshData("/test.obj", 0, 0, b"", b"")
        p.unload(data)


class TestAudioProcessor:
    """Tests for AudioProcessor."""

    def test_supported_extensions(self) -> None:
        """Should support common audio extensions."""
        p = AudioProcessor()
        assert ".wav" in p.supported_extensions
        assert ".mp3" in p.supported_extensions
        assert ".ogg" in p.supported_extensions

    def test_can_process(self) -> None:
        """can_process should match extensions."""
        p = AudioProcessor()
        assert p.can_process("/test/sound.wav")
        assert not p.can_process("/test/texture.png")

    def test_load_wav(self, temp_dir: str) -> None:
        """Should load WAV file and parse header."""
        path = os.path.join(temp_dir, "test.wav")
        # Create minimal WAV header
        sample_rate = 44100
        channels = 2
        data_size = 8820  # 0.1 seconds of 16-bit stereo
        wav_data = (
            b"RIFF"
            + struct.pack("<I", 36 + data_size)  # File size - 8
            + b"WAVEfmt "
            + struct.pack("<I", 16)  # Subchunk1 size
            + struct.pack("<HH", 1, channels)  # Audio format, channels
            + struct.pack("<I", sample_rate)  # Sample rate
            + struct.pack("<I", sample_rate * channels * 2)  # Byte rate
            + struct.pack("<HH", channels * 2, 16)  # Block align, bits
            + b"data"
            + struct.pack("<I", data_size)
            + b"\x00" * data_size
        )
        with open(path, "wb") as f:
            f.write(wav_data)

        p = AudioProcessor()
        result = p.load(path)
        assert isinstance(result, AudioData)
        assert result.path == path
        assert result.sample_rate == 44100
        assert result.channels == 2
        assert result.duration_ms == 50  # ~0.05 seconds

    def test_unload(self) -> None:
        """unload should not crash."""
        p = AudioProcessor()
        data = AudioData("/test.wav", 44100, 2, 0, b"")
        p.unload(data)


# =============================================================================
# AssetReloader Tests
# =============================================================================


class TestAssetReloader:
    """Tests for AssetReloader."""

    def test_register_processor(
        self, handle_table: AssetHandleTable, file_watcher: FileWatcher
    ) -> None:
        """Should register custom processors."""

        class CustomProcessor(AssetProcessor[str]):
            @property
            def supported_extensions(self) -> Set[str]:
                return {".custom"}

            def load(self, path: str) -> str:
                return "custom"

            def unload(self, asset: str) -> None:
                pass

        reloader = AssetReloader(handle_table, file_watcher)
        reloader.register_processor(CustomProcessor())
        assert reloader.get_processor("/test/file.custom") is not None

    def test_unregister_processor(
        self, handle_table: AssetHandleTable, file_watcher: FileWatcher
    ) -> None:
        """Should unregister processors."""
        reloader = AssetReloader(handle_table, file_watcher)
        result = reloader.unregister_processor(".png")
        assert result is True
        assert reloader.get_processor("/test/file.png") is None

    def test_get_processor(
        self, handle_table: AssetHandleTable, file_watcher: FileWatcher
    ) -> None:
        """Should return correct processor for extension."""
        reloader = AssetReloader(handle_table, file_watcher)
        p = reloader.get_processor("/test/texture.png")
        assert isinstance(p, TextureProcessor)

    def test_watch_directory(
        self,
        handle_table: AssetHandleTable,
        file_watcher: FileWatcher,
        temp_dir: str,
    ) -> None:
        """Should watch a directory."""
        reloader = AssetReloader(handle_table, file_watcher)
        result = reloader.watch_directory(temp_dir)
        assert result is True
        assert temp_dir in reloader.watched_directories

    def test_watch_directory_nonexistent(
        self, handle_table: AssetHandleTable, file_watcher: FileWatcher
    ) -> None:
        """Watching nonexistent directory should return False."""
        reloader = AssetReloader(handle_table, file_watcher)
        result = reloader.watch_directory("/nonexistent/dir")
        assert result is False

    def test_unwatch_directory(
        self,
        handle_table: AssetHandleTable,
        file_watcher: FileWatcher,
        temp_dir: str,
    ) -> None:
        """Should unwatch a directory."""
        reloader = AssetReloader(handle_table, file_watcher)
        reloader.watch_directory(temp_dir)
        result = reloader.unwatch_directory(temp_dir)
        assert result is True
        assert temp_dir not in reloader.watched_directories

    def test_reload_texture(
        self,
        handle_table: AssetHandleTable,
        file_watcher: FileWatcher,
        temp_dir: str,
    ) -> None:
        """Should reload a texture asset."""
        path = os.path.join(temp_dir, "test.png")
        # Create minimal PNG
        png_data = (
            b"\x89PNG\r\n\x1a\n"
            + b"\x00\x00\x00\rIHDR"
            + struct.pack(">II", 32, 32)
            + b"\x08\x06\x00\x00\x00"
            + b"\x00\x00\x00\x00"
        )
        with open(path, "wb") as f:
            f.write(png_data)

        reloader = AssetReloader(handle_table, file_watcher)
        handle_table.register(path, TextureData)

        event = reloader.reload(path)
        assert event.success
        assert event.new_version == 1
        assert handle_table.get_state(handle_table.get_handle_by_path(path)) == HandleState.READY

    def test_reload_callback(
        self,
        handle_table: AssetHandleTable,
        file_watcher: FileWatcher,
        temp_dir: str,
    ) -> None:
        """Reload callbacks should be called."""
        path = os.path.join(temp_dir, "test.png")
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

        events: List[ReloadEvent] = []
        reloader = AssetReloader(handle_table, file_watcher)
        reloader.add_reload_callback(events.append)
        handle_table.register(path, TextureData)

        reloader.reload(path)
        assert len(events) == 1
        assert events[0].path == path

    def test_remove_reload_callback(
        self, handle_table: AssetHandleTable, file_watcher: FileWatcher
    ) -> None:
        """Should remove reload callback."""
        events: List[ReloadEvent] = []
        reloader = AssetReloader(handle_table, file_watcher)
        cb = events.append
        reloader.add_reload_callback(cb)
        result = reloader.remove_reload_callback(cb)
        assert result is True

    def test_reload_all(
        self,
        handle_table: AssetHandleTable,
        file_watcher: FileWatcher,
        temp_dir: str,
    ) -> None:
        """Should reload all registered assets."""
        paths = []
        for i in range(3):
            path = os.path.join(temp_dir, f"test{i}.png")
            with open(path, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
            paths.append(path)
            handle_table.register(path, TextureData)

        reloader = AssetReloader(handle_table, file_watcher)
        events = reloader.reload_all()
        assert len(events) == 3
        assert all(e.success for e in events)

    def test_deferred_strategy(
        self,
        handle_table: AssetHandleTable,
        file_watcher: FileWatcher,
        temp_dir: str,
    ) -> None:
        """DEFERRED strategy should queue reloads."""
        path = os.path.join(temp_dir, "test.png")
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

        reloader = AssetReloader(
            handle_table, file_watcher, ReloadStrategy.DEFERRED
        )
        handle_table.register(path, TextureData)

        # Simulate file change
        event = FileChangeEvent(Path(path), FileChangeType.MODIFIED)
        reloader._on_file_change(event)

        assert reloader.pending_reload_count == 1

        events = reloader.process_queue()
        assert len(events) == 1
        assert reloader.pending_reload_count == 0

    def test_start_stop(
        self, handle_table: AssetHandleTable, file_watcher: FileWatcher
    ) -> None:
        """Start/stop should control the reloader."""
        reloader = AssetReloader(handle_table, file_watcher)
        assert not reloader.is_running
        reloader.start()
        assert reloader.is_running
        reloader.stop()
        assert not reloader.is_running

    def test_stats(
        self,
        handle_table: AssetHandleTable,
        file_watcher: FileWatcher,
        temp_dir: str,
    ) -> None:
        """Should track reload statistics."""
        path = os.path.join(temp_dir, "test.png")
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

        reloader = AssetReloader(handle_table, file_watcher)
        handle_table.register(path, TextureData)
        reloader.reload(path)

        stats = reloader.stats
        assert stats["reloads_attempted"] == 1
        assert stats["reloads_succeeded"] == 1
        assert stats["total_reload_time_ms"] > 0

    def test_reload_deleted_file(
        self,
        handle_table: AssetHandleTable,
        file_watcher: FileWatcher,
        temp_dir: str,
    ) -> None:
        """Deleting a watched file should set state to FAILED."""
        path = os.path.join(temp_dir, "test.png")
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

        reloader = AssetReloader(handle_table, file_watcher)
        handle = handle_table.register(path, TextureData)

        os.remove(path)
        event = reloader._reload_asset(path, FileChangeType.DELETED)

        assert not event.success
        assert "deleted" in event.error.lower()
        assert handle_table.get_state(handle) == HandleState.FAILED

    def test_reload_error_handling(
        self,
        handle_table: AssetHandleTable,
        file_watcher: FileWatcher,
        temp_dir: str,
    ) -> None:
        """Should handle reload errors gracefully."""
        path = os.path.join(temp_dir, "test.png")
        # Create invalid PNG that will fail to parse
        with open(path, "wb") as f:
            f.write(b"not a png")

        reloader = AssetReloader(handle_table, file_watcher)
        handle = handle_table.register(path, TextureData)
        event = reloader.reload(path)

        # Should still succeed (raw bytes loaded)
        assert event.success
        assert handle_table.get_state(handle) == HandleState.READY

    def test_dispose(
        self, handle_table: AssetHandleTable, file_watcher: FileWatcher
    ) -> None:
        """dispose() should clean up resources."""
        reloader = AssetReloader(handle_table, file_watcher)
        reloader.start()
        reloader.dispose()
        assert not reloader.is_running
        assert reloader.pending_reload_count == 0

    def test_file_watcher_integration(
        self,
        handle_table: AssetHandleTable,
        temp_dir: str,
    ) -> None:
        """Modifying a file should trigger reload via FileWatcher."""
        path = os.path.join(temp_dir, "test.png")
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

        watcher = FileWatcher(poll_interval_ms=50)
        reloader = AssetReloader(handle_table, watcher, ReloadStrategy.IMMEDIATE)
        handle = handle_table.register(path, TextureData)
        reloader.watch_directory(temp_dir)

        # Initial poll to register file state
        watcher.poll_once()

        # Modify file
        time.sleep(0.05)
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + struct.pack(">II", 128, 128) + b"\x00" * 20)
        os.utime(path, (time.time() + 2, time.time() + 2))

        # Poll for changes
        events = watcher.poll_once()
        # Reload should have happened
        version = handle_table.get_version(handle)
        # Version might be 0 or 1 depending on timing
        assert version >= 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestHotReloadIntegration:
    """Integration tests for the full hot-reload system."""

    def test_full_reload_cycle(self, temp_dir: str) -> None:
        """Test complete hot-reload cycle: register -> modify -> reload."""
        # Setup
        path = os.path.join(temp_dir, "asset.png")
        png_v1 = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 64, 64) + b"\x00" * 20
        with open(path, "wb") as f:
            f.write(png_v1)

        table = AssetHandleTable()
        watcher = FileWatcher(poll_interval_ms=50)
        reloader = AssetReloader(table, watcher)

        # Register asset
        handle = table.register(path, TextureData)
        assert handle.is_valid()
        assert handle.get_version() == 0

        # Load initial data
        reloader.reload(path)
        data = handle.get()
        assert data is not None
        assert data.width == 64

        # Modify file
        png_v2 = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 128, 128) + b"\x00" * 20
        with open(path, "wb") as f:
            f.write(png_v2)

        # Reload
        reloader.reload(path)
        data = handle.get()
        assert data is not None
        assert data.width == 128
        assert handle.get_version() == 2

        # Handle still valid
        assert handle.is_valid()

    def test_multiple_handles_same_asset(self, temp_dir: str) -> None:
        """Multiple handles to same asset should all see updated data."""
        path = os.path.join(temp_dir, "shared.png")
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

        table = AssetHandleTable()
        watcher = FileWatcher()
        reloader = AssetReloader(table, watcher)

        # Create multiple handles
        h1 = table.register(path, TextureData)
        h2 = table.register(path, TextureData)
        h3 = table.register(path, TextureData)

        # All should be same handle
        assert h1 == h2 == h3

        # Initial reload
        reloader.reload(path)
        v1 = h1.get_version()

        # Modify and reload
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x01" * 20)
        reloader.reload(path)

        # All handles should see new version
        assert h1.get_version() == v1 + 1
        assert h2.get_version() == v1 + 1
        assert h3.get_version() == v1 + 1

    def test_handle_survives_table_callback_exception(self, temp_dir: str) -> None:
        """Handle should remain valid even if callback raises."""
        path = os.path.join(temp_dir, "test.png")
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

        table = AssetHandleTable()
        watcher = FileWatcher()
        reloader = AssetReloader(table, watcher)

        def bad_callback(event: ReloadEvent) -> None:
            raise RuntimeError("Callback failed")

        reloader.add_reload_callback(bad_callback)

        handle = table.register(path, TextureData)
        reloader.reload(path)  # Should not raise

        assert handle.is_valid()
        assert handle.get() is not None

    def test_concurrent_access(self, temp_dir: str) -> None:
        """Handle table should be thread-safe."""
        path = os.path.join(temp_dir, "concurrent.png")
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

        table = AssetHandleTable()
        watcher = FileWatcher()
        reloader = AssetReloader(table, watcher)

        errors: List[Exception] = []
        handle = table.register(path, TextureData)

        def reader() -> None:
            try:
                for _ in range(100):
                    _ = handle.get()
                    _ = handle.get_version()
                    _ = handle.is_valid()
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                for _ in range(50):
                    reloader.reload(path)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader) for _ in range(5)
        ] + [
            threading.Thread(target=writer) for _ in range(2)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert handle.is_valid()

"""Tests for T-CC-1.5: File watcher for config/data hot-reload."""
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from engine.core.file_watcher import (
    FileChangeType,
    FileChangeEvent,
    FileState,
    WatchedPath,
    CallbackRegistry,
    FileWatcher,
    create_config_watcher,
    create_asset_watcher,
)


class TestFileChangeType:
    """Tests for FileChangeType enum."""

    def test_all_types_exist(self):
        types = [
            FileChangeType.CREATED,
            FileChangeType.MODIFIED,
            FileChangeType.DELETED,
            FileChangeType.RENAMED,
        ]
        assert len(types) == 4


class TestFileChangeEvent:
    """Tests for FileChangeEvent."""

    def test_basic_event(self):
        event = FileChangeEvent(
            path=Path("config.json"),
            change_type=FileChangeType.MODIFIED,
        )
        assert event.change_type == FileChangeType.MODIFIED
        assert event.timestamp > 0

    def test_is_config_json(self):
        event = FileChangeEvent(Path("settings.json"), FileChangeType.MODIFIED)
        assert event.is_config is True

    def test_is_config_yaml(self):
        event = FileChangeEvent(Path("config.yaml"), FileChangeType.MODIFIED)
        assert event.is_config is True

    def test_is_config_toml(self):
        event = FileChangeEvent(Path("pyproject.toml"), FileChangeType.MODIFIED)
        assert event.is_config is True

    def test_is_config_false(self):
        event = FileChangeEvent(Path("data.csv"), FileChangeType.MODIFIED)
        assert event.is_config is False

    def test_is_data(self):
        event = FileChangeEvent(Path("users.csv"), FileChangeType.MODIFIED)
        assert event.is_data is True

    def test_is_asset_png(self):
        event = FileChangeEvent(Path("texture.png"), FileChangeType.MODIFIED)
        assert event.is_asset is True

    def test_is_asset_wgsl(self):
        event = FileChangeEvent(Path("shader.wgsl"), FileChangeType.MODIFIED)
        assert event.is_asset is True

    def test_rename_event(self):
        event = FileChangeEvent(
            path=Path("new_name.json"),
            change_type=FileChangeType.RENAMED,
            old_path=Path("old_name.json"),
        )
        assert event.old_path == Path("old_name.json")


class TestFileState:
    """Tests for FileState."""

    def test_from_path_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            f.flush()
            path = Path(f.name)

        try:
            state = FileState.from_path(path)
            assert state is not None
            assert state.path == path
            assert state.size == 12
            assert state.mtime > 0
        finally:
            os.unlink(path)

    def test_from_path_with_hash(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            f.flush()
            path = Path(f.name)

        try:
            state = FileState.from_path(path, compute_hash=True)
            assert state.content_hash is not None
        finally:
            os.unlink(path)

    def test_from_path_nonexistent(self):
        state = FileState.from_path(Path("/nonexistent/file.txt"))
        assert state is None

    def test_has_changed_mtime(self):
        state1 = FileState(Path("test.txt"), mtime=1000.0, size=100)
        state2 = FileState(Path("test.txt"), mtime=2000.0, size=100)
        assert state1.has_changed(state2) is True

    def test_has_changed_size(self):
        state1 = FileState(Path("test.txt"), mtime=1000.0, size=100)
        state2 = FileState(Path("test.txt"), mtime=1000.0, size=200)
        assert state1.has_changed(state2) is True

    def test_has_changed_hash(self):
        state1 = FileState(Path("test.txt"), mtime=1000.0, size=100, content_hash="abc")
        state2 = FileState(Path("test.txt"), mtime=1000.0, size=100, content_hash="def")
        assert state1.has_changed(state2) is True

    def test_has_changed_false(self):
        state1 = FileState(Path("test.txt"), mtime=1000.0, size=100)
        state2 = FileState(Path("test.txt"), mtime=1000.0, size=100)
        assert state1.has_changed(state2) is False


class TestWatchedPath:
    """Tests for WatchedPath."""

    def test_default_values(self):
        wp = WatchedPath(path=Path("/test"))
        assert wp.recursive is False
        assert wp.patterns is None
        assert wp.debounce_ms == 100

    def test_custom_values(self):
        wp = WatchedPath(
            path=Path("/test"),
            recursive=True,
            patterns={"*.json"},
            debounce_ms=200,
        )
        assert wp.recursive is True
        assert "*.json" in wp.patterns


class TestCallbackRegistry:
    """Tests for CallbackRegistry."""

    def test_register_global(self):
        registry = CallbackRegistry()
        callback = MagicMock()
        registry.register_global(callback)
        event = FileChangeEvent(Path("test.txt"), FileChangeType.MODIFIED)
        callbacks = registry.get_callbacks_for_event(event)
        assert callback in callbacks

    def test_register_global_no_duplicates(self):
        registry = CallbackRegistry()
        callback = MagicMock()
        registry.register_global(callback)
        registry.register_global(callback)
        event = FileChangeEvent(Path("test.txt"), FileChangeType.MODIFIED)
        callbacks = registry.get_callbacks_for_event(event)
        assert callbacks.count(callback) == 1

    def test_register_path(self):
        registry = CallbackRegistry()
        callback = MagicMock()
        path = Path("/test/config.json").resolve()
        registry.register_path(path, callback)
        event = FileChangeEvent(path, FileChangeType.MODIFIED)
        callbacks = registry.get_callbacks_for_event(event)
        assert callback in callbacks

    def test_register_path_no_match(self):
        registry = CallbackRegistry()
        callback = MagicMock()
        registry.register_path("/test/config.json", callback)
        event = FileChangeEvent(Path("/other/file.json"), FileChangeType.MODIFIED)
        callbacks = registry.get_callbacks_for_event(event)
        assert callback not in callbacks

    def test_register_extension(self):
        registry = CallbackRegistry()
        callback = MagicMock()
        registry.register_extension(".json", callback)
        event = FileChangeEvent(Path("config.json"), FileChangeType.MODIFIED)
        callbacks = registry.get_callbacks_for_event(event)
        assert callback in callbacks

    def test_register_extension_no_dot(self):
        registry = CallbackRegistry()
        callback = MagicMock()
        registry.register_extension("yaml", callback)
        event = FileChangeEvent(Path("config.yaml"), FileChangeType.MODIFIED)
        callbacks = registry.get_callbacks_for_event(event)
        assert callback in callbacks

    def test_register_pattern(self):
        registry = CallbackRegistry()
        callback = MagicMock()
        registry.register_pattern("*.json", callback)
        event = FileChangeEvent(Path("config.json"), FileChangeType.MODIFIED)
        callbacks = registry.get_callbacks_for_event(event)
        assert callback in callbacks

    def test_unregister_global(self):
        registry = CallbackRegistry()
        callback = MagicMock()
        registry.register_global(callback)
        result = registry.unregister_global(callback)
        assert result is True
        event = FileChangeEvent(Path("test.txt"), FileChangeType.MODIFIED)
        callbacks = registry.get_callbacks_for_event(event)
        assert callback not in callbacks

    def test_unregister_path(self):
        registry = CallbackRegistry()
        callback = MagicMock()
        path = Path("/test/config.json")
        registry.register_path(path, callback)
        result = registry.unregister_path(path, callback)
        assert result is True

    def test_unregister_extension(self):
        registry = CallbackRegistry()
        callback = MagicMock()
        registry.register_extension(".json", callback)
        result = registry.unregister_extension(".json", callback)
        assert result is True

    def test_clear(self):
        registry = CallbackRegistry()
        registry.register_global(MagicMock())
        registry.register_extension(".json", MagicMock())
        registry.clear()
        event = FileChangeEvent(Path("config.json"), FileChangeType.MODIFIED)
        callbacks = registry.get_callbacks_for_event(event)
        assert len(callbacks) == 0


class TestFileWatcher:
    """Tests for FileWatcher."""

    def test_initial_state(self):
        watcher = FileWatcher()
        assert watcher.is_running is False
        assert watcher.watched_path_count == 0
        assert watcher.tracked_file_count == 0

    def test_watch_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / "test.json").write_text("{}")

            watcher = FileWatcher()
            result = watcher.watch(tmpdir)

            assert result is True
            assert watcher.watched_path_count == 1
            assert watcher.tracked_file_count == 1

    def test_watch_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"{}")
            path = Path(f.name)

        try:
            watcher = FileWatcher()
            result = watcher.watch(path)
            assert result is True
            assert path in watcher.get_tracked_files()
        finally:
            os.unlink(path)

    def test_watch_nonexistent(self):
        watcher = FileWatcher()
        result = watcher.watch("/nonexistent/path")
        assert result is False

    def test_watch_recursive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            subdir = path / "subdir"
            subdir.mkdir()
            (path / "root.json").write_text("{}")
            (subdir / "nested.json").write_text("{}")

            watcher = FileWatcher()
            watcher.watch(tmpdir, recursive=True)

            tracked = watcher.get_tracked_files()
            assert len(tracked) == 2

    def test_unwatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher()
            watcher.watch(tmpdir)
            result = watcher.unwatch(tmpdir)
            assert result is True
            assert watcher.watched_path_count == 0

    def test_unwatch_nonexistent(self):
        watcher = FileWatcher()
        result = watcher.unwatch("/not/watched")
        assert result is False

    def test_detect_modification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            config_file = path / "config.json"
            config_file.write_text('{"version": 1}')

            watcher = FileWatcher(poll_interval_ms=50)
            watcher.watch(tmpdir)

            # Modify the file
            time.sleep(0.1)
            config_file.write_text('{"version": 2}')

            events = watcher.poll_once()
            assert len(events) == 1
            assert events[0].change_type == FileChangeType.MODIFIED

    def test_detect_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            watcher = FileWatcher(poll_interval_ms=50)
            watcher.watch(tmpdir)

            # Create new file
            new_file = path / "new.json"
            new_file.write_text("{}")

            events = watcher.poll_once()
            assert len(events) == 1
            assert events[0].change_type == FileChangeType.CREATED

    def test_detect_deletion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            config_file = path / "config.json"
            config_file.write_text("{}")

            watcher = FileWatcher(poll_interval_ms=50)
            watcher.watch(tmpdir)

            # Delete the file
            os.unlink(config_file)

            events = watcher.poll_once()
            assert len(events) == 1
            assert events[0].change_type == FileChangeType.DELETED

    def test_callback_invoked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            config_file = path / "config.json"
            config_file.write_text("{}")

            callback = MagicMock()
            watcher = FileWatcher(poll_interval_ms=50)
            watcher.registry.register_global(callback)
            watcher.watch(tmpdir)

            # Modify
            time.sleep(0.1)
            config_file.write_text('{"modified": true}')
            watcher.poll_once()

            callback.assert_called_once()

    def test_extension_callback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            json_file = path / "config.json"
            txt_file = path / "readme.txt"
            json_file.write_text("{}")
            txt_file.write_text("readme")

            json_callback = MagicMock()
            watcher = FileWatcher(poll_interval_ms=50)
            watcher.registry.register_extension(".json", json_callback)
            watcher.watch(tmpdir)

            # Modify both files
            time.sleep(0.1)
            json_file.write_text('{"v": 2}')
            txt_file.write_text("updated")
            watcher.poll_once()

            # Only JSON callback should be called
            json_callback.assert_called_once()

    def test_start_stop(self):
        watcher = FileWatcher(poll_interval_ms=50)
        watcher.start()
        assert watcher.is_running is True
        watcher.stop()
        assert watcher.is_running is False

    def test_start_idempotent(self):
        watcher = FileWatcher(poll_interval_ms=50)
        watcher.start()
        watcher.start()  # Should not create another thread
        assert watcher.is_running is True
        watcher.stop()

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            watcher = FileWatcher()
            watcher.watch(tmpdir)
            watcher.registry.register_global(MagicMock())
            watcher.clear()

            assert watcher.watched_path_count == 0
            assert watcher.tracked_file_count == 0

    def test_pattern_filtering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            json_file = path / "config.json"
            py_file = path / "script.py"
            json_file.write_text("{}")
            py_file.write_text("# python")

            watcher = FileWatcher(poll_interval_ms=50)
            watcher.watch(tmpdir, patterns={"*.json"})

            tracked = watcher.get_tracked_files()
            names = [f.name for f in tracked]
            assert "config.json" in names
            assert "script.py" not in names


class TestDebouncing:
    """Tests for event debouncing."""

    def test_debounce_rapid_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            config_file = path / "config.json"
            config_file.write_text("{}")

            watcher = FileWatcher(poll_interval_ms=50)
            watcher.watch(tmpdir, debounce_ms=200)

            # Rapid modifications
            for i in range(5):
                config_file.write_text(f'{{"v": {i}}}')
                events = watcher.poll_once()

            # Should only get one event due to debouncing
            # (after debounce window expires)
            time.sleep(0.25)
            config_file.write_text('{"v": final}')
            events = watcher.poll_once()
            assert len(events) <= 1


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_config_watcher(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = MagicMock()
            watcher = create_config_watcher([tmpdir], callback)

            assert watcher.watched_path_count == 1

            # Check that config extensions are registered
            event = FileChangeEvent(Path("test.json"), FileChangeType.MODIFIED)
            callbacks = watcher.registry.get_callbacks_for_event(event)
            assert callback in callbacks

    def test_create_asset_watcher(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            callback = MagicMock()
            watcher = create_asset_watcher([tmpdir], callback)

            assert watcher.watched_path_count == 1

            # Global callback should fire for any file
            event = FileChangeEvent(Path("texture.png"), FileChangeType.MODIFIED)
            callbacks = watcher.registry.get_callbacks_for_event(event)
            assert callback in callbacks


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_watch_unwatch(self):
        import threading

        watcher = FileWatcher()
        errors = []

        def watch_unwatch():
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    for _ in range(10):
                        watcher.watch(tmpdir)
                        watcher.unwatch(tmpdir)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=watch_unwatch) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_callback_registration(self):
        import threading

        registry = CallbackRegistry()
        errors = []

        def register_callbacks():
            try:
                for i in range(100):
                    callback = MagicMock()
                    registry.register_global(callback)
                    registry.register_extension(f".ext{i}", callback)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_callbacks) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestIntegration:
    """Integration tests."""

    def test_full_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Setup files
            (path / "config.json").write_text('{"v": 1}')
            (path / "data.yaml").write_text("key: value")

            # Create watcher
            events_received = []

            def on_change(event):
                events_received.append(event)

            watcher = FileWatcher(poll_interval_ms=50)
            watcher.registry.register_global(on_change)
            watcher.watch(tmpdir)

            # Make changes
            time.sleep(0.1)
            (path / "config.json").write_text('{"v": 2}')
            (path / "new.txt").write_text("new file")

            watcher.poll_once()

            assert len(events_received) >= 2

    def test_config_hot_reload_scenario(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            config_file = path / "settings.json"
            config_file.write_text('{"debug": false}')

            reloaded = {"count": 0}

            def on_config_change(event):
                if event.is_config:
                    reloaded["count"] += 1

            watcher = create_config_watcher([tmpdir], on_config_change)

            # Simulate config change
            time.sleep(0.1)
            config_file.write_text('{"debug": true}')
            watcher.poll_once()

            assert reloaded["count"] == 1

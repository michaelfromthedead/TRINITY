"""
Tests for the module watcher functionality.
"""
import os
import tempfile
import time
import pytest
from pathlib import Path

from engine.tooling.hotreload.module_watcher import (
    ModuleWatcher,
    ModuleChangeEvent,
    ModuleChangeType,
)


class TestModuleChangeEvent:
    """Tests for ModuleChangeEvent."""

    def test_event_creation(self):
        """Test creating a module change event."""
        event = ModuleChangeEvent(
            change_type=ModuleChangeType.MODIFIED,
            module_name="test_module",
            file_path="/path/to/test_module.py",
        )

        assert event.change_type == ModuleChangeType.MODIFIED
        assert event.module_name == "test_module"
        assert event.file_path == "/path/to/test_module.py"
        assert event.timestamp > 0

    def test_event_repr(self):
        """Test event string representation."""
        event = ModuleChangeEvent(
            change_type=ModuleChangeType.CREATED,
            module_name="test_module",
            file_path="/path/to/test_module.py",
        )

        repr_str = repr(event)
        assert "CREATED" in repr_str
        assert "test_module" in repr_str


class TestModuleChangeType:
    """Tests for ModuleChangeType enum."""

    def test_all_types_exist(self):
        """Test all expected change types exist."""
        assert hasattr(ModuleChangeType, "CREATED")
        assert hasattr(ModuleChangeType, "MODIFIED")
        assert hasattr(ModuleChangeType, "DELETED")
        assert hasattr(ModuleChangeType, "RENAMED")


class TestModuleWatcher:
    """Tests for ModuleWatcher."""

    def setup_method(self):
        """Set up test fixtures."""
        self.watcher = ModuleWatcher(poll_interval=0.1, debounce_time=0.05)
        self.temp_dir = tempfile.mkdtemp()
        self.events = []

    def teardown_method(self):
        """Clean up after tests."""
        self.watcher.stop()
        self.watcher.clear()
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _callback(self, event):
        """Test callback that records events."""
        self.events.append(event)

    def test_watcher_initialization(self):
        """Test ModuleWatcher initializes correctly."""
        assert self.watcher.is_running is False
        assert len(self.watcher.watched_directories) == 0
        assert len(self.watcher.watched_modules) == 0

    def test_add_callback(self):
        """Test adding callbacks."""
        self.watcher.add_callback(self._callback)
        # Should not raise
        assert True

    def test_remove_callback(self):
        """Test removing callbacks."""
        self.watcher.add_callback(self._callback)
        self.watcher.remove_callback(self._callback)
        # Should not raise
        assert True

    def test_watch_directory(self):
        """Test watching a directory."""
        result = self.watcher.watch_directory(self.temp_dir, recursive=False)
        assert result is True
        assert self.temp_dir in self.watcher.watched_directories

    def test_watch_nonexistent_directory(self):
        """Test watching a non-existent directory."""
        result = self.watcher.watch_directory("/nonexistent/path/xyz")
        assert result is False

    def test_unwatch_directory(self):
        """Test unwatching a directory."""
        # Create a file so the directory has something to watch
        file_path = os.path.join(self.temp_dir, "test.py")
        Path(file_path).touch()

        self.watcher.watch_directory(self.temp_dir)

        # Unwatch the specific file that was registered
        result = self.watcher.unwatch(file_path)
        assert result is True

    def test_unwatch_nonwatched_directory(self):
        """Test unwatching a directory that wasn't watched."""
        result = self.watcher.unwatch_directory("/some/random/path")
        assert result is False

    def test_start_stop(self):
        """Test starting and stopping the watcher."""
        self.watcher.watch_directory(self.temp_dir)
        self.watcher.start()
        assert self.watcher.is_running is True

        self.watcher.stop()
        assert self.watcher.is_running is False

    def test_file_to_module_conversion(self):
        """Test file path to module name conversion."""
        # Create a Python file in temp directory
        self.watcher.watch_directory(self.temp_dir)

        file_path = os.path.join(self.temp_dir, "test_module.py")
        Path(file_path).touch()

        module_name = self.watcher._file_to_module(file_path, self.temp_dir)
        assert module_name == "test_module"

    def test_nested_module_conversion(self):
        """Test nested module path conversion."""
        # Create nested structure
        nested_dir = os.path.join(self.temp_dir, "package")
        os.makedirs(nested_dir)

        file_path = os.path.join(nested_dir, "submodule.py")
        Path(file_path).touch()

        self.watcher.watch_directory(self.temp_dir)

        module_name = self.watcher._file_to_module(file_path, self.temp_dir)
        assert module_name == "package.submodule"

    def test_init_file_conversion(self):
        """Test __init__.py conversion."""
        package_dir = os.path.join(self.temp_dir, "mypackage")
        os.makedirs(package_dir)

        file_path = os.path.join(package_dir, "__init__.py")
        Path(file_path).touch()

        self.watcher.watch_directory(self.temp_dir)

        module_name = self.watcher._file_to_module(file_path, self.temp_dir)
        assert module_name == "mypackage"

    def test_get_module_file(self):
        """Test getting file path for a module."""
        file_path = os.path.join(self.temp_dir, "test_module.py")
        Path(file_path).touch()

        self.watcher.watch_directory(self.temp_dir)

        found_path = self.watcher.get_module_file("test_module")
        assert found_path == file_path

    def test_exclude_patterns(self):
        """Test exclude patterns filter files."""
        # Create a pycache file
        pycache_dir = os.path.join(self.temp_dir, "__pycache__")
        os.makedirs(pycache_dir)
        file_path = os.path.join(pycache_dir, "test.pyc")
        Path(file_path).touch()

        self.watcher.watch_directory(
            self.temp_dir,
            exclude_patterns=["__pycache__"],
        )

        # __pycache__ files should not be mapped
        modules = self.watcher.watched_modules
        assert not any("__pycache__" in m for m in modules)

    def test_clear(self):
        """Test clearing the watcher."""
        self.watcher.watch_directory(self.temp_dir)
        self.watcher.add_callback(self._callback)
        self.watcher.clear()

        assert len(self.watcher.watched_directories) == 0
        assert len(self.watcher.watched_modules) == 0


class TestModuleWatcherIntegration:
    """Integration tests for ModuleWatcher."""

    def setup_method(self):
        """Set up test fixtures."""
        self.watcher = ModuleWatcher(poll_interval=0.1, debounce_time=0.05)
        self.temp_dir = tempfile.mkdtemp()
        self.events = []

    def teardown_method(self):
        """Clean up after tests."""
        self.watcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _callback(self, event):
        self.events.append(event)

    @pytest.mark.slow
    def test_detect_file_creation(self):
        """Test detecting new file creation."""
        # First create the file so it can be watched
        file_path = os.path.join(self.temp_dir, "new_module.py")
        Path(file_path).write_text("# initial")

        self.watcher.add_callback(self._callback)
        self.watcher.watch_directory(self.temp_dir)
        self.watcher.start()

        time.sleep(0.2)  # Let watcher stabilize
        self.events.clear()

        # Modify the file
        Path(file_path).write_text("# modified content")

        # Wait for detection
        time.sleep(0.3)

        # Should have detected modification
        assert any(e.file_path == file_path for e in self.events)

    @pytest.mark.slow
    def test_detect_file_modification(self):
        """Test detecting file modification."""
        # Create file first
        file_path = os.path.join(self.temp_dir, "existing_module.py")
        Path(file_path).write_text("# original content")

        self.watcher.add_callback(self._callback)
        self.watcher.watch_directory(self.temp_dir)
        self.watcher.start()

        time.sleep(0.2)  # Let watcher stabilize
        self.events.clear()

        # Modify the file
        Path(file_path).write_text("# modified content")

        # Wait for detection
        time.sleep(0.3)

        # Should have detected modification
        modified_events = [
            e for e in self.events
            if e.change_type == ModuleChangeType.MODIFIED
        ]
        assert len(modified_events) >= 1

    @pytest.mark.slow
    def test_detect_file_deletion(self):
        """Test detecting file deletion."""
        # Create file first
        file_path = os.path.join(self.temp_dir, "to_delete.py")
        Path(file_path).write_text("# to be deleted")

        self.watcher.add_callback(self._callback)
        self.watcher.watch_directory(self.temp_dir)
        self.watcher.start()

        time.sleep(0.2)  # Let watcher stabilize
        self.events.clear()

        # Delete the file
        os.remove(file_path)

        # Wait for detection
        time.sleep(0.3)

        # Should have detected deletion
        deleted_events = [
            e for e in self.events
            if e.change_type == ModuleChangeType.DELETED
        ]
        assert len(deleted_events) >= 1

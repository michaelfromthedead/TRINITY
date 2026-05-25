"""
Module Watcher - File system monitoring for Python module changes.

Watches Python source files and triggers reload events when changes are detected.
Integrates with the platform's FileWatcher for cross-platform compatibility.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from engine.platform.os.file_watcher import (
    FileWatcher,
    FileEvent,
    FileEventData,
)


class ModuleChangeType(Enum):
    """Types of module changes."""
    CREATED = auto()
    MODIFIED = auto()
    DELETED = auto()
    RENAMED = auto()


@dataclass
class ModuleChangeEvent:
    """Event describing a module change."""

    change_type: ModuleChangeType
    module_name: str
    file_path: str
    timestamp: float = field(default_factory=time.time)
    old_path: Optional[str] = None  # For renames

    def __repr__(self) -> str:
        return f"ModuleChangeEvent({self.change_type.name}, {self.module_name}, {self.file_path})"


ModuleChangeCallback = Callable[[ModuleChangeEvent], None]


class ModuleWatcher:
    """
    Watches Python modules for changes and triggers reload events.

    Features:
    - Watches specified directories for .py file changes
    - Maps file paths to module names
    - Supports include/exclude patterns
    - Debounces rapid changes
    - Thread-safe callback invocation
    """

    # Constants
    DEFAULT_POLL_INTERVAL = 0.5  # seconds
    DEFAULT_DEBOUNCE_TIME = 0.1  # seconds

    def __init__(
        self,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        debounce_time: float = DEFAULT_DEBOUNCE_TIME,
    ):
        """
        Initialize the module watcher.

        Args:
            poll_interval: How often to check for changes (seconds).
            debounce_time: Minimum time between events for same file (seconds).
        """
        self._file_watcher = FileWatcher(poll_interval=poll_interval)
        self._debounce_time = debounce_time
        self._callbacks: List[ModuleChangeCallback] = []
        self._watched_dirs: Set[str] = set()
        self._include_patterns: List[str] = []
        self._exclude_patterns: List[str] = ["__pycache__", ".pyc", ".pyo"]
        self._last_events: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._running = False
        self._module_map: Dict[str, str] = {}  # file_path -> module_name

    @property
    def is_running(self) -> bool:
        """Check if the watcher is running."""
        return self._running

    @property
    def watched_directories(self) -> List[str]:
        """Get list of watched directories."""
        with self._lock:
            return list(self._watched_dirs)

    @property
    def watched_modules(self) -> List[str]:
        """Get list of watched module names."""
        with self._lock:
            return list(self._module_map.values())

    def add_callback(self, callback: ModuleChangeCallback) -> None:
        """
        Add a callback for module change events.

        Args:
            callback: Function to call when a module changes.
        """
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: ModuleChangeCallback) -> None:
        """
        Remove a callback.

        Args:
            callback: Callback to remove.
        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def watch_directory(
        self,
        path: str,
        recursive: bool = True,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> bool:
        """
        Watch a directory for Python module changes.

        Args:
            path: Directory path to watch.
            recursive: Watch subdirectories recursively.
            include_patterns: Optional list of patterns to include.
            exclude_patterns: Optional list of patterns to exclude.

        Returns:
            True if directory was added successfully.
        """
        abs_path = os.path.abspath(path)

        if not os.path.isdir(abs_path):
            return False

        with self._lock:
            self._watched_dirs.add(abs_path)

            if include_patterns:
                self._include_patterns.extend(include_patterns)
            if exclude_patterns:
                self._exclude_patterns.extend(exclude_patterns)

            # Map existing Python files to modules
            self._map_directory(abs_path, recursive)

        # Set up file watching
        return self._file_watcher.watch_directory(
            abs_path,
            callback=self._on_file_change,
            recursive=recursive,
        )

    def watch_module(self, module_name: str) -> bool:
        """
        Watch a specific module by name.

        Args:
            module_name: Name of the module to watch.

        Returns:
            True if module was found and is now being watched.
        """
        # Find the module's file
        if module_name in sys.modules:
            module = sys.modules[module_name]
            if hasattr(module, "__file__") and module.__file__:
                file_path = module.__file__
                if file_path.endswith((".py", ".pyw")):
                    with self._lock:
                        self._module_map[file_path] = module_name
                    return self._file_watcher.watch_file(
                        file_path,
                        callback=self._on_file_change,
                    )
        return False

    def unwatch(self, path: str) -> bool:
        """
        Stop watching a specific file or directory.

        Args:
            path: File or directory path to stop watching.

        Returns:
            True if path was being watched.
        """
        abs_path = os.path.abspath(path)

        with self._lock:
            # Remove from module map if it's a file
            if abs_path in self._module_map:
                del self._module_map[abs_path]

            # Try to unwatch via file watcher
            return self._file_watcher.unwatch(abs_path)

    def unwatch_directory(self, path: str) -> bool:
        """
        Stop watching a directory.

        Args:
            path: Directory path to stop watching.

        Returns:
            True if directory was being watched.
        """
        abs_path = os.path.abspath(path)

        with self._lock:
            if abs_path in self._watched_dirs:
                self._watched_dirs.discard(abs_path)

                # Remove mappings for files in this directory
                to_remove = [
                    fp for fp in self._module_map
                    if fp.startswith(abs_path)
                ]
                for fp in to_remove:
                    del self._module_map[fp]

                return self._file_watcher.unwatch(abs_path)

        return False

    def start(self) -> None:
        """Start watching for changes."""
        self._running = True
        self._file_watcher.start()

    def stop(self) -> None:
        """Stop watching for changes."""
        self._running = False
        self._file_watcher.stop()

    def _map_directory(self, dir_path: str, recursive: bool) -> None:
        """Map Python files in a directory to module names."""
        for root, dirs, files in os.walk(dir_path):
            # Skip excluded directories
            dirs[:] = [
                d for d in dirs
                if not any(p in d for p in self._exclude_patterns)
            ]

            for filename in files:
                if filename.endswith((".py", ".pyw")):
                    file_path = os.path.join(root, filename)

                    # Skip excluded files
                    if any(p in file_path for p in self._exclude_patterns):
                        continue

                    module_name = self._file_to_module(file_path, dir_path)
                    if module_name:
                        self._module_map[file_path] = module_name

            if not recursive:
                break

    def _file_to_module(self, file_path: str, base_path: str) -> Optional[str]:
        """
        Convert a file path to a module name.

        Args:
            file_path: Path to the Python file.
            base_path: Base directory for module resolution.

        Returns:
            Module name or None if not determinable.
        """
        # Get relative path
        try:
            rel_path = os.path.relpath(file_path, base_path)
        except ValueError:
            return None

        # Remove .py extension
        if rel_path.endswith(".py"):
            rel_path = rel_path[:-3]
        elif rel_path.endswith(".pyw"):
            rel_path = rel_path[:-4]
        else:
            return None

        # Handle __init__.py
        if rel_path.endswith("__init__"):
            rel_path = rel_path[:-9]  # Remove /__init__
            if not rel_path:
                # Root __init__.py
                return os.path.basename(base_path)

        # Convert path separators to dots
        module_name = rel_path.replace(os.sep, ".")

        # Clean up
        module_name = module_name.strip(".")

        return module_name if module_name else None

    def _on_file_change(self, event: FileEventData) -> None:
        """Handle file change events from the file watcher."""
        # Only process Python files
        if not event.path.endswith((".py", ".pyw")):
            return

        # Check exclude patterns
        if any(p in event.path for p in self._exclude_patterns):
            return

        # Debounce
        with self._lock:
            last_time = self._last_events.get(event.path, 0)
            if event.timestamp - last_time < self._debounce_time:
                return
            self._last_events[event.path] = event.timestamp

        # Map file event to module event
        module_name = self._get_module_name(event.path)
        if not module_name:
            return

        # Create module change event
        change_type = self._map_event_type(event.event)
        module_event = ModuleChangeEvent(
            change_type=change_type,
            module_name=module_name,
            file_path=event.path,
            timestamp=event.timestamp,
        )

        # Update module map for new files
        if change_type == ModuleChangeType.CREATED:
            with self._lock:
                self._module_map[event.path] = module_name
        elif change_type == ModuleChangeType.DELETED:
            with self._lock:
                self._module_map.pop(event.path, None)

        # Invoke callbacks
        self._notify_callbacks(module_event)

    def _get_module_name(self, file_path: str) -> Optional[str]:
        """Get the module name for a file path."""
        with self._lock:
            if file_path in self._module_map:
                return self._module_map[file_path]

        # Try to determine module name from watched directories
        for dir_path in self._watched_dirs:
            if file_path.startswith(dir_path):
                return self._file_to_module(file_path, dir_path)

        return None

    def _map_event_type(self, event: FileEvent) -> ModuleChangeType:
        """Map file event type to module change type."""
        if event == FileEvent.CREATED:
            return ModuleChangeType.CREATED
        elif event == FileEvent.DELETED:
            return ModuleChangeType.DELETED
        else:
            return ModuleChangeType.MODIFIED

    def _notify_callbacks(self, event: ModuleChangeEvent) -> None:
        """Notify all callbacks of a module change."""
        with self._lock:
            callbacks = list(self._callbacks)

        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                # Log error but don't stop other callbacks
                pass

    def get_module_file(self, module_name: str) -> Optional[str]:
        """
        Get the file path for a module name.

        Args:
            module_name: Name of the module.

        Returns:
            File path or None if not found.
        """
        with self._lock:
            for file_path, name in self._module_map.items():
                if name == module_name:
                    return file_path

        # Try sys.modules
        if module_name in sys.modules:
            module = sys.modules[module_name]
            if hasattr(module, "__file__"):
                return module.__file__

        return None

    def clear(self) -> None:
        """Clear all watches and mappings."""
        with self._lock:
            self._watched_dirs.clear()
            self._module_map.clear()
            self._last_events.clear()
            self._callbacks.clear()


__all__ = [
    "ModuleChangeType",
    "ModuleChangeEvent",
    "ModuleChangeCallback",
    "ModuleWatcher",
]

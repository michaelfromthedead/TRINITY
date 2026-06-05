"""T-CC-1.5: File watcher for config/data hot-reload.

Provides notify-based file watching with callback registration for
detecting changes to configuration and data files at runtime.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Union
from weakref import WeakSet


class FileChangeType(Enum):
    """Types of file system changes."""
    CREATED = auto()
    MODIFIED = auto()
    DELETED = auto()
    RENAMED = auto()


@dataclass
class FileChangeEvent:
    """Represents a file change event."""
    path: Path
    change_type: FileChangeType
    timestamp: float = field(default_factory=time.time)
    old_path: Optional[Path] = None  # For renames

    @property
    def is_config(self) -> bool:
        return self.path.suffix in ('.json', '.yaml', '.yml', '.toml', '.ini', '.cfg')

    @property
    def is_data(self) -> bool:
        return self.path.suffix in ('.csv', '.xml', '.bin', '.dat')

    @property
    def is_asset(self) -> bool:
        return self.path.suffix in ('.png', '.jpg', '.glb', '.gltf', '.wgsl', '.obj')


FileChangeCallback = Callable[[FileChangeEvent], None]


@dataclass
class WatchedPath:
    """Configuration for a watched path."""
    path: Path
    recursive: bool = False
    patterns: Optional[Set[str]] = None  # Glob patterns to match
    ignore_patterns: Optional[Set[str]] = None
    debounce_ms: int = 100


@dataclass
class FileState:
    """Tracks the state of a file for change detection."""
    path: Path
    mtime: float
    size: int
    content_hash: Optional[str] = None

    @classmethod
    def from_path(cls, path: Path, compute_hash: bool = False) -> Optional["FileState"]:
        try:
            stat = path.stat()
            content_hash = None
            if compute_hash and path.is_file():
                with open(path, 'rb') as f:
                    content_hash = hashlib.md5(f.read()).hexdigest()
            return cls(
                path=path,
                mtime=stat.st_mtime,
                size=stat.st_size,
                content_hash=content_hash,
            )
        except (OSError, IOError):
            return None

    def has_changed(self, other: "FileState") -> bool:
        if self.mtime != other.mtime:
            return True
        if self.size != other.size:
            return True
        if self.content_hash and other.content_hash:
            return self.content_hash != other.content_hash
        return False


class CallbackRegistry:
    """Registry for file change callbacks."""

    def __init__(self):
        self._global_callbacks: List[FileChangeCallback] = []
        self._path_callbacks: Dict[Path, List[FileChangeCallback]] = {}
        self._pattern_callbacks: Dict[str, List[FileChangeCallback]] = {}
        self._extension_callbacks: Dict[str, List[FileChangeCallback]] = {}
        self._lock = threading.RLock()

    def register_global(self, callback: FileChangeCallback) -> None:
        """Register a callback for all file changes."""
        with self._lock:
            if callback not in self._global_callbacks:
                self._global_callbacks.append(callback)

    def register_path(self, path: Union[str, Path], callback: FileChangeCallback) -> None:
        """Register a callback for a specific path."""
        path = Path(path).resolve()
        with self._lock:
            if path not in self._path_callbacks:
                self._path_callbacks[path] = []
            if callback not in self._path_callbacks[path]:
                self._path_callbacks[path].append(callback)

    def register_pattern(self, pattern: str, callback: FileChangeCallback) -> None:
        """Register a callback for files matching a glob pattern."""
        with self._lock:
            if pattern not in self._pattern_callbacks:
                self._pattern_callbacks[pattern] = []
            if callback not in self._pattern_callbacks[pattern]:
                self._pattern_callbacks[pattern].append(callback)

    def register_extension(self, extension: str, callback: FileChangeCallback) -> None:
        """Register a callback for files with a specific extension."""
        ext = extension if extension.startswith('.') else f'.{extension}'
        with self._lock:
            if ext not in self._extension_callbacks:
                self._extension_callbacks[ext] = []
            if callback not in self._extension_callbacks[ext]:
                self._extension_callbacks[ext].append(callback)

    def unregister_global(self, callback: FileChangeCallback) -> bool:
        with self._lock:
            if callback in self._global_callbacks:
                self._global_callbacks.remove(callback)
                return True
            return False

    def unregister_path(self, path: Union[str, Path], callback: FileChangeCallback) -> bool:
        path = Path(path).resolve()
        with self._lock:
            if path in self._path_callbacks and callback in self._path_callbacks[path]:
                self._path_callbacks[path].remove(callback)
                return True
            return False

    def unregister_extension(self, extension: str, callback: FileChangeCallback) -> bool:
        ext = extension if extension.startswith('.') else f'.{extension}'
        with self._lock:
            if ext in self._extension_callbacks and callback in self._extension_callbacks[ext]:
                self._extension_callbacks[ext].remove(callback)
                return True
            return False

    def get_callbacks_for_event(self, event: FileChangeEvent) -> List[FileChangeCallback]:
        """Get all callbacks that should be notified for an event."""
        callbacks = []
        path = event.path.resolve()

        with self._lock:
            callbacks.extend(self._global_callbacks)

            if path in self._path_callbacks:
                callbacks.extend(self._path_callbacks[path])

            ext = path.suffix
            if ext in self._extension_callbacks:
                callbacks.extend(self._extension_callbacks[ext])

            for pattern, cbs in self._pattern_callbacks.items():
                if path.match(pattern):
                    callbacks.extend(cbs)

        return list(dict.fromkeys(callbacks))  # Remove duplicates

    def clear(self) -> None:
        with self._lock:
            self._global_callbacks.clear()
            self._path_callbacks.clear()
            self._pattern_callbacks.clear()
            self._extension_callbacks.clear()


class FileWatcher:
    """Watches files and directories for changes."""

    def __init__(self, poll_interval_ms: int = 500):
        self._poll_interval = poll_interval_ms / 1000.0
        self._watched_paths: Dict[Path, WatchedPath] = {}
        self._file_states: Dict[Path, FileState] = {}
        self._registry = CallbackRegistry()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._pending_events: List[FileChangeEvent] = []
        self._debounce_timers: Dict[Path, float] = {}

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def registry(self) -> CallbackRegistry:
        return self._registry

    @property
    def watched_path_count(self) -> int:
        return len(self._watched_paths)

    @property
    def tracked_file_count(self) -> int:
        return len(self._file_states)

    def watch(
        self,
        path: Union[str, Path],
        recursive: bool = False,
        patterns: Optional[Set[str]] = None,
        ignore_patterns: Optional[Set[str]] = None,
        debounce_ms: int = 100,
    ) -> bool:
        """Add a path to watch."""
        path = Path(path).resolve()
        if not path.exists():
            return False

        with self._lock:
            self._watched_paths[path] = WatchedPath(
                path=path,
                recursive=recursive,
                patterns=patterns,
                ignore_patterns=ignore_patterns,
                debounce_ms=debounce_ms,
            )
            self._scan_path(path, self._watched_paths[path])

        return True

    def unwatch(self, path: Union[str, Path]) -> bool:
        """Remove a path from watching."""
        path = Path(path).resolve()
        with self._lock:
            if path in self._watched_paths:
                del self._watched_paths[path]
                self._file_states = {
                    p: s for p, s in self._file_states.items()
                    if not str(p).startswith(str(path))
                }
                return True
        return False

    def _scan_path(self, path: Path, config: WatchedPath) -> None:
        """Scan a path and record initial file states."""
        if path.is_file():
            if self._should_process(path, config):
                state = FileState.from_path(path)
                if state:
                    self._file_states[path] = state
        elif path.is_dir():
            for item in path.iterdir():
                if item.is_file():
                    if self._should_process(item, config):
                        state = FileState.from_path(item)
                        if state:
                            self._file_states[item] = state
                elif item.is_dir() and config.recursive:
                    self._scan_path(item, config)

    def _should_process(self, path: Path, watched: WatchedPath) -> bool:
        """Check if a path should be processed based on patterns."""
        if watched.ignore_patterns:
            for pattern in watched.ignore_patterns:
                if path.match(pattern):
                    return False
        if watched.patterns:
            for pattern in watched.patterns:
                if path.match(pattern):
                    return True
            return False
        return True

    def _check_for_changes(self) -> List[FileChangeEvent]:
        """Check all watched paths for changes."""
        events = []
        current_files: Set[Path] = set()

        with self._lock:
            for watched_path, config in list(self._watched_paths.items()):
                if not watched_path.exists():
                    continue

                if watched_path.is_file():
                    current_files.add(watched_path)
                    events.extend(self._check_file(watched_path, config))
                elif watched_path.is_dir():
                    events.extend(self._check_directory(watched_path, config, current_files))

            # Check for deleted files
            for path in list(self._file_states.keys()):
                if path not in current_files:
                    events.append(FileChangeEvent(path, FileChangeType.DELETED))
                    del self._file_states[path]

        return events

    def _check_file(self, path: Path, config: WatchedPath) -> List[FileChangeEvent]:
        """Check a single file for changes."""
        events = []
        if not self._should_process(path, config):
            return events

        new_state = FileState.from_path(path)
        if new_state is None:
            return events

        if path in self._file_states:
            old_state = self._file_states[path]
            if new_state.has_changed(old_state):
                events.append(FileChangeEvent(path, FileChangeType.MODIFIED))
                self._file_states[path] = new_state
        else:
            events.append(FileChangeEvent(path, FileChangeType.CREATED))
            self._file_states[path] = new_state

        return events

    def _check_directory(
        self,
        dir_path: Path,
        config: WatchedPath,
        current_files: Set[Path],
    ) -> List[FileChangeEvent]:
        """Check a directory for changes."""
        events = []
        try:
            for item in dir_path.iterdir():
                if item.is_file():
                    current_files.add(item)
                    events.extend(self._check_file(item, config))
                elif item.is_dir() and config.recursive:
                    events.extend(self._check_directory(item, config, current_files))
        except (OSError, PermissionError):
            pass
        return events

    def _apply_debounce(self, events: List[FileChangeEvent]) -> List[FileChangeEvent]:
        """Apply debouncing to events."""
        now = time.time()
        filtered = []

        for event in events:
            path = event.path
            watched = None
            for wp, config in self._watched_paths.items():
                if str(path).startswith(str(wp)):
                    watched = config
                    break

            debounce_s = (watched.debounce_ms if watched else 100) / 1000.0

            if path in self._debounce_timers:
                if now - self._debounce_timers[path] < debounce_s:
                    continue

            self._debounce_timers[path] = now
            filtered.append(event)

        return filtered

    def _dispatch_events(self, events: List[FileChangeEvent]) -> None:
        """Dispatch events to registered callbacks."""
        for event in events:
            callbacks = self._registry.get_callbacks_for_event(event)
            for callback in callbacks:
                try:
                    callback(event)
                except Exception:
                    pass  # Don't let callback errors stop other callbacks

    def _poll_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            events = self._check_for_changes()
            if events:
                events = self._apply_debounce(events)
                if events:
                    self._dispatch_events(events)
            time.sleep(self._poll_interval)

    def start(self) -> None:
        """Start the file watcher."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the file watcher."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def poll_once(self) -> List[FileChangeEvent]:
        """Manually poll for changes (for testing or manual control)."""
        events = self._check_for_changes()
        events = self._apply_debounce(events)
        self._dispatch_events(events)
        return events

    def get_watched_paths(self) -> List[Path]:
        """Get list of watched paths."""
        with self._lock:
            return list(self._watched_paths.keys())

    def get_tracked_files(self) -> List[Path]:
        """Get list of tracked files."""
        with self._lock:
            return list(self._file_states.keys())

    def clear(self) -> None:
        """Clear all watches and state."""
        with self._lock:
            self._watched_paths.clear()
            self._file_states.clear()
            self._debounce_timers.clear()
            self._registry.clear()


def create_config_watcher(
    config_dirs: List[Union[str, Path]],
    callback: FileChangeCallback,
) -> FileWatcher:
    """Factory to create a watcher for config files."""
    watcher = FileWatcher(poll_interval_ms=500)
    for dir_path in config_dirs:
        watcher.watch(
            dir_path,
            recursive=True,
            patterns={'*.json', '*.yaml', '*.yml', '*.toml', '*.ini'},
        )
    watcher.registry.register_extension('.json', callback)
    watcher.registry.register_extension('.yaml', callback)
    watcher.registry.register_extension('.yml', callback)
    watcher.registry.register_extension('.toml', callback)
    watcher.registry.register_extension('.ini', callback)
    return watcher


def create_asset_watcher(
    asset_dirs: List[Union[str, Path]],
    callback: FileChangeCallback,
) -> FileWatcher:
    """Factory to create a watcher for asset files."""
    watcher = FileWatcher(poll_interval_ms=1000)
    for dir_path in asset_dirs:
        watcher.watch(
            dir_path,
            recursive=True,
            patterns={'*.png', '*.jpg', '*.glb', '*.gltf', '*.wgsl', '*.obj'},
        )
    watcher.registry.register_global(callback)
    return watcher

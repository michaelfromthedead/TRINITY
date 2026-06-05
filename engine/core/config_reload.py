"""T-CC-1.6: Config reload callback registration.

Provides a registry for systems to register config reload handlers
that are triggered when configuration files change at runtime.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
from weakref import WeakMethod, ref

from .file_watcher import (
    FileChangeEvent,
    FileChangeType,
    FileWatcher,
    create_config_watcher,
)


class ReloadPriority(Enum):
    """Priority levels for reload handlers."""
    HIGHEST = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    LOWEST = 4


class ReloadResult(Enum):
    """Result of a reload operation."""
    SUCCESS = auto()
    PARTIAL = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class ReloadEvent:
    """Event passed to reload handlers."""
    path: Path
    change_type: FileChangeType
    old_data: Optional[Any] = None
    new_data: Optional[Any] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def is_creation(self) -> bool:
        return self.change_type == FileChangeType.CREATED

    @property
    def is_modification(self) -> bool:
        return self.change_type == FileChangeType.MODIFIED

    @property
    def is_deletion(self) -> bool:
        return self.change_type == FileChangeType.DELETED


@dataclass
class ReloadOutcome:
    """Result of processing a reload event."""
    event: ReloadEvent
    result: ReloadResult
    handler_name: str
    duration_ms: float
    error: Optional[str] = None


ReloadHandler = Callable[[ReloadEvent], ReloadResult]


@dataclass
class RegisteredHandler:
    """A registered reload handler with metadata."""
    name: str
    handler: ReloadHandler
    priority: ReloadPriority = ReloadPriority.NORMAL
    paths: Optional[Set[Path]] = None
    patterns: Optional[Set[str]] = None
    extensions: Optional[Set[str]] = None
    enabled: bool = True

    def matches(self, path: Path) -> bool:
        """Check if this handler should process a path."""
        if not self.enabled:
            return False

        resolved = path.resolve()

        if self.paths and resolved in self.paths:
            return True

        if self.extensions:
            ext = path.suffix.lower()
            if ext in self.extensions or ext.lstrip('.') in self.extensions:
                return True

        if self.patterns:
            for pattern in self.patterns:
                if path.match(pattern):
                    return True

        return self.paths is None and self.patterns is None and self.extensions is None


class ConfigCache:
    """Cache for config file contents to enable diff detection."""

    def __init__(self, max_entries: int = 100, max_size_bytes: int = 10_000_000):
        self._cache: Dict[Path, Any] = {}
        self._timestamps: Dict[Path, float] = {}
        self._sizes: Dict[Path, int] = {}
        self._max_entries = max_entries
        self._max_size = max_size_bytes
        self._current_size = 0
        self._lock = threading.RLock()

    def get(self, path: Path) -> Optional[Any]:
        """Get cached data for a path."""
        with self._lock:
            return self._cache.get(path.resolve())

    def set(self, path: Path, data: Any, size: int = 0) -> None:
        """Cache data for a path."""
        resolved = path.resolve()
        with self._lock:
            if resolved in self._cache:
                self._current_size -= self._sizes.get(resolved, 0)

            self._evict_if_needed(size)

            self._cache[resolved] = data
            self._timestamps[resolved] = time.time()
            self._sizes[resolved] = size
            self._current_size += size

    def remove(self, path: Path) -> Optional[Any]:
        """Remove and return cached data for a path."""
        resolved = path.resolve()
        with self._lock:
            data = self._cache.pop(resolved, None)
            self._timestamps.pop(resolved, None)
            size = self._sizes.pop(resolved, 0)
            self._current_size -= size
            return data

    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._sizes.clear()
            self._current_size = 0

    def _evict_if_needed(self, needed_size: int) -> None:
        """Evict old entries if cache is full."""
        while (
            len(self._cache) >= self._max_entries
            or self._current_size + needed_size > self._max_size
        ) and self._cache:
            oldest_path = min(self._timestamps, key=self._timestamps.get)
            self.remove(oldest_path)

    @property
    def entry_count(self) -> int:
        return len(self._cache)

    @property
    def total_size(self) -> int:
        return self._current_size


class ConfigReloadManager:
    """Manages config reload handlers and coordinates hot-reload."""

    def __init__(
        self,
        file_watcher: Optional[FileWatcher] = None,
        auto_start: bool = False,
        cache_configs: bool = True,
    ):
        self._watcher = file_watcher or FileWatcher(poll_interval_ms=500)
        self._handlers: List[RegisteredHandler] = []
        self._lock = threading.RLock()
        self._cache = ConfigCache() if cache_configs else None
        self._outcomes: List[ReloadOutcome] = []
        self._max_outcomes = 100
        self._listeners: List[Callable[[ReloadOutcome], None]] = []
        self._paused = False

        self._watcher.registry.register_global(self._on_file_change)

        if auto_start:
            self._watcher.start()

    @property
    def is_running(self) -> bool:
        return self._watcher.is_running

    @property
    def handler_count(self) -> int:
        return len(self._handlers)

    @property
    def is_paused(self) -> bool:
        return self._paused

    def register(
        self,
        name: str,
        handler: ReloadHandler,
        *,
        paths: Optional[List[Union[str, Path]]] = None,
        patterns: Optional[List[str]] = None,
        extensions: Optional[List[str]] = None,
        priority: ReloadPriority = ReloadPriority.NORMAL,
    ) -> bool:
        """Register a reload handler."""
        with self._lock:
            for h in self._handlers:
                if h.name == name:
                    return False

            resolved_paths = None
            if paths:
                resolved_paths = {Path(p).resolve() for p in paths}

            normalized_ext = None
            if extensions:
                normalized_ext = {
                    e if e.startswith('.') else f'.{e}' for e in extensions
                }

            reg = RegisteredHandler(
                name=name,
                handler=handler,
                priority=priority,
                paths=resolved_paths,
                patterns=set(patterns) if patterns else None,
                extensions=normalized_ext,
            )
            self._handlers.append(reg)
            self._handlers.sort(key=lambda h: h.priority.value)

        return True

    def unregister(self, name: str) -> bool:
        """Unregister a handler by name."""
        with self._lock:
            for i, h in enumerate(self._handlers):
                if h.name == name:
                    self._handlers.pop(i)
                    return True
        return False

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a handler."""
        with self._lock:
            for h in self._handlers:
                if h.name == name:
                    h.enabled = enabled
                    return True
        return False

    def get_handler(self, name: str) -> Optional[RegisteredHandler]:
        """Get a handler by name."""
        with self._lock:
            for h in self._handlers:
                if h.name == name:
                    return h
        return None

    def watch(
        self,
        path: Union[str, Path],
        recursive: bool = False,
        patterns: Optional[Set[str]] = None,
    ) -> bool:
        """Add a config path to watch."""
        return self._watcher.watch(path, recursive=recursive, patterns=patterns)

    def unwatch(self, path: Union[str, Path]) -> bool:
        """Remove a config path from watching."""
        return self._watcher.unwatch(path)

    def start(self) -> None:
        """Start the reload manager."""
        self._watcher.start()

    def stop(self) -> None:
        """Stop the reload manager."""
        self._watcher.stop()

    def pause(self) -> None:
        """Pause reload processing (file watching continues)."""
        self._paused = True

    def resume(self) -> None:
        """Resume reload processing."""
        self._paused = False

    def _on_file_change(self, event: FileChangeEvent) -> None:
        """Handle a file change event."""
        if self._paused:
            return

        old_data = None
        new_data = None

        if self._cache:
            old_data = self._cache.get(event.path)

            if event.change_type != FileChangeType.DELETED:
                new_data = self._load_config(event.path)
                if new_data is not None:
                    size = len(str(new_data))
                    self._cache.set(event.path, new_data, size)
            else:
                self._cache.remove(event.path)

        reload_event = ReloadEvent(
            path=event.path,
            change_type=event.change_type,
            old_data=old_data,
            new_data=new_data,
            timestamp=event.timestamp,
        )

        self._dispatch_reload(reload_event)

    def _load_config(self, path: Path) -> Optional[Any]:
        """Load config file contents."""
        try:
            if path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    return json.load(f)
            else:
                with open(path, 'r') as f:
                    return f.read()
        except (OSError, json.JSONDecodeError):
            return None

    def _dispatch_reload(self, event: ReloadEvent) -> None:
        """Dispatch reload event to matching handlers."""
        with self._lock:
            handlers = [h for h in self._handlers if h.matches(event.path)]

        for handler in handlers:
            start = time.perf_counter()
            try:
                result = handler.handler(event)
                duration = (time.perf_counter() - start) * 1000
                outcome = ReloadOutcome(
                    event=event,
                    result=result,
                    handler_name=handler.name,
                    duration_ms=duration,
                )
            except Exception as e:
                duration = (time.perf_counter() - start) * 1000
                outcome = ReloadOutcome(
                    event=event,
                    result=ReloadResult.FAILED,
                    handler_name=handler.name,
                    duration_ms=duration,
                    error=str(e),
                )

            self._record_outcome(outcome)

    def _record_outcome(self, outcome: ReloadOutcome) -> None:
        """Record and notify about a reload outcome."""
        with self._lock:
            self._outcomes.append(outcome)
            if len(self._outcomes) > self._max_outcomes:
                self._outcomes.pop(0)
            listeners = self._listeners.copy()

        for listener in listeners:
            try:
                listener(outcome)
            except Exception:
                pass

    def add_outcome_listener(
        self,
        listener: Callable[[ReloadOutcome], None],
    ) -> None:
        """Add a listener for reload outcomes."""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_outcome_listener(
        self,
        listener: Callable[[ReloadOutcome], None],
    ) -> bool:
        """Remove an outcome listener."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
                return True
        return False

    def get_recent_outcomes(self, limit: int = 10) -> List[ReloadOutcome]:
        """Get recent reload outcomes."""
        with self._lock:
            return list(self._outcomes[-limit:])

    def trigger_reload(self, path: Union[str, Path]) -> List[ReloadOutcome]:
        """Manually trigger a reload for a path."""
        path = Path(path).resolve()

        new_data = self._load_config(path) if path.exists() else None
        old_data = self._cache.get(path) if self._cache else None

        change_type = (
            FileChangeType.DELETED if not path.exists()
            else FileChangeType.CREATED if old_data is None
            else FileChangeType.MODIFIED
        )

        if self._cache and new_data is not None:
            self._cache.set(path, new_data, len(str(new_data)))

        event = ReloadEvent(
            path=path,
            change_type=change_type,
            old_data=old_data,
            new_data=new_data,
        )

        initial_count = len(self._outcomes)
        self._dispatch_reload(event)
        return self._outcomes[initial_count:]

    def clear(self) -> None:
        """Clear all handlers and state."""
        with self._lock:
            self._handlers.clear()
            self._outcomes.clear()
            self._listeners.clear()
            if self._cache:
                self._cache.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the reload manager."""
        with self._lock:
            return {
                'running': self.is_running,
                'paused': self._paused,
                'handler_count': len(self._handlers),
                'handlers': [
                    {
                        'name': h.name,
                        'priority': h.priority.name,
                        'enabled': h.enabled,
                        'paths': [str(p) for p in (h.paths or [])],
                        'patterns': list(h.patterns or []),
                        'extensions': list(h.extensions or []),
                    }
                    for h in self._handlers
                ],
                'recent_outcomes': len(self._outcomes),
                'cache_entries': self._cache.entry_count if self._cache else 0,
            }


def create_engine_config_manager(
    config_dirs: List[Union[str, Path]],
) -> ConfigReloadManager:
    """Factory to create a manager for engine configs."""
    watcher = create_config_watcher(config_dirs, lambda e: None)
    manager = ConfigReloadManager(file_watcher=watcher, cache_configs=True)
    return manager


class ConfigReloadDecorator:
    """Decorator for methods that should be called on config reload."""

    _registry: Dict[str, List[tuple]] = {}

    def __init__(
        self,
        config_path: Optional[str] = None,
        pattern: Optional[str] = None,
        extension: Optional[str] = None,
        priority: ReloadPriority = ReloadPriority.NORMAL,
    ):
        self.config_path = config_path
        self.pattern = pattern
        self.extension = extension
        self.priority = priority

    def __call__(self, method: Callable) -> Callable:
        key = f"{method.__module__}.{method.__qualname__}"
        if key not in self._registry:
            self._registry[key] = []
        self._registry[key].append((
            self.config_path,
            self.pattern,
            self.extension,
            self.priority,
        ))
        return method

    @classmethod
    def get_handlers_for_class(cls, obj: Any) -> List[tuple]:
        """Get registered handlers for an object's class."""
        handlers = []
        for name in dir(obj):
            if name.startswith('_'):
                continue
            method = getattr(obj, name, None)
            if callable(method):
                key = f"{method.__module__}.{type(obj).__name__}.{name}"
                if key in cls._registry:
                    for entry in cls._registry[key]:
                        handlers.append((name, method, entry))
        return handlers


on_config_reload = ConfigReloadDecorator

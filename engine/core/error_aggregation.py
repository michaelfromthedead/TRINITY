"""T-CC-1.10: Error aggregation system with error panel.

Provides centralized error collection, aggregation, and display for
consistent error handling across the engine.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple
from weakref import WeakSet

from .result import Error, ErrorKind


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    FATAL = 5


class ErrorSource(Enum):
    """Sources of errors in the engine."""
    UNKNOWN = auto()
    RENDERING = auto()
    PHYSICS = auto()
    AUDIO = auto()
    NETWORK = auto()
    SCRIPTING = auto()
    ASSET = auto()
    GPU = auto()
    FFI = auto()
    MEMORY = auto()
    IO = auto()
    CONFIG = auto()
    VALIDATION = auto()


@dataclass(frozen=True)
class ErrorEntry:
    """A single error entry in the aggregation system."""
    id: int
    message: str
    severity: ErrorSeverity
    source: ErrorSource
    kind: ErrorKind = ErrorKind.UNKNOWN
    timestamp: float = field(default_factory=time.time)
    context: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    error: Optional[Error] = None
    count: int = 1
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None

    @property
    def is_critical(self) -> bool:
        return self.severity.value >= ErrorSeverity.CRITICAL.value

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def matches(self, other: "ErrorEntry") -> bool:
        """Check if this error matches another for deduplication."""
        return (
            self.message == other.message
            and self.severity == other.severity
            and self.source == other.source
            and self.kind == other.kind
        )


@dataclass
class ErrorStats:
    """Statistics about errors."""
    total_count: int = 0
    by_severity: Dict[ErrorSeverity, int] = field(default_factory=dict)
    by_source: Dict[ErrorSource, int] = field(default_factory=dict)
    by_kind: Dict[ErrorKind, int] = field(default_factory=dict)
    first_error_time: Optional[float] = None
    last_error_time: Optional[float] = None

    def update(self, entry: ErrorEntry) -> None:
        """Update stats with a new entry."""
        self.total_count += entry.count
        self.by_severity[entry.severity] = (
            self.by_severity.get(entry.severity, 0) + entry.count
        )
        self.by_source[entry.source] = (
            self.by_source.get(entry.source, 0) + entry.count
        )
        self.by_kind[entry.kind] = (
            self.by_kind.get(entry.kind, 0) + entry.count
        )
        if self.first_error_time is None:
            self.first_error_time = entry.timestamp
        self.last_error_time = entry.timestamp


ErrorHandler = Callable[[ErrorEntry], None]


class ErrorFilter:
    """Filter for error entries."""

    def __init__(
        self,
        min_severity: Optional[ErrorSeverity] = None,
        max_severity: Optional[ErrorSeverity] = None,
        sources: Optional[Set[ErrorSource]] = None,
        kinds: Optional[Set[ErrorKind]] = None,
        message_pattern: Optional[str] = None,
        max_age_seconds: Optional[float] = None,
    ):
        self.min_severity = min_severity
        self.max_severity = max_severity
        self.sources = sources
        self.kinds = kinds
        self.message_pattern = message_pattern
        self.max_age_seconds = max_age_seconds

    def matches(self, entry: ErrorEntry) -> bool:
        """Check if an entry matches this filter."""
        if self.min_severity and entry.severity.value < self.min_severity.value:
            return False
        if self.max_severity and entry.severity.value > self.max_severity.value:
            return False
        if self.sources and entry.source not in self.sources:
            return False
        if self.kinds and entry.kind not in self.kinds:
            return False
        if self.message_pattern and self.message_pattern not in entry.message:
            return False
        if self.max_age_seconds and entry.age_seconds > self.max_age_seconds:
            return False
        return True


class ErrorAggregator:
    """Central error aggregation system."""

    def __init__(
        self,
        max_entries: int = 1000,
        deduplicate: bool = True,
        dedupe_window_seconds: float = 60.0,
    ):
        self._entries: List[ErrorEntry] = []
        self._max_entries = max_entries
        self._deduplicate = deduplicate
        self._dedupe_window = dedupe_window_seconds
        self._next_id = 1
        self._lock = threading.RLock()
        self._handlers: List[ErrorHandler] = []
        self._stats = ErrorStats()
        self._suppressed: Set[str] = set()
        self._rate_limits: Dict[str, Tuple[float, int]] = {}
        self._rate_limit_window = 1.0  # seconds
        self._rate_limit_max = 100  # max per window

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def stats(self) -> ErrorStats:
        return self._stats

    def add(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        source: ErrorSource = ErrorSource.UNKNOWN,
        kind: ErrorKind = ErrorKind.UNKNOWN,
        context: Optional[Dict[str, Any]] = None,
        stack_trace: Optional[str] = None,
        error: Optional[Error] = None,
    ) -> Optional[ErrorEntry]:
        """Add an error to the aggregator."""
        if self._is_suppressed(message, source):
            return None

        if not self._check_rate_limit(message):
            return None

        with self._lock:
            entry = ErrorEntry(
                id=self._next_id,
                message=message,
                severity=severity,
                source=source,
                kind=kind,
                context=context,
                stack_trace=stack_trace,
                error=error,
                first_seen=time.time(),
                last_seen=time.time(),
            )
            self._next_id += 1

            if self._deduplicate:
                existing = self._find_duplicate(entry)
                if existing:
                    self._entries.remove(existing)
                    entry = ErrorEntry(
                        id=existing.id,
                        message=entry.message,
                        severity=entry.severity,
                        source=entry.source,
                        kind=entry.kind,
                        context=entry.context,
                        stack_trace=entry.stack_trace,
                        error=entry.error,
                        count=existing.count + 1,
                        first_seen=existing.first_seen,
                        last_seen=time.time(),
                    )

            self._entries.append(entry)
            self._stats.update(entry)

            if len(self._entries) > self._max_entries:
                self._entries.pop(0)

        self._notify_handlers(entry)
        return entry

    def add_from_error(
        self,
        error: Error,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        source: ErrorSource = ErrorSource.UNKNOWN,
    ) -> Optional[ErrorEntry]:
        """Add an error from a Result Error type."""
        return self.add(
            message=error.message,
            severity=severity,
            source=source,
            kind=error.kind,
            context=dict(error.context) if error.context else None,
            error=error,
        )

    def _find_duplicate(self, entry: ErrorEntry) -> Optional[ErrorEntry]:
        """Find a duplicate entry within the dedupe window."""
        cutoff = time.time() - self._dedupe_window
        for existing in reversed(self._entries):
            if existing.timestamp < cutoff:
                break
            if entry.matches(existing):
                return existing
        return None

    def _is_suppressed(self, message: str, source: ErrorSource) -> bool:
        """Check if this error is suppressed."""
        return (
            message in self._suppressed
            or f"{source.name}:{message}" in self._suppressed
        )

    def _check_rate_limit(self, message: str) -> bool:
        """Check rate limit for this error."""
        now = time.time()
        key = message[:100]  # Limit key length

        if key in self._rate_limits:
            window_start, count = self._rate_limits[key]
            if now - window_start < self._rate_limit_window:
                if count >= self._rate_limit_max:
                    return False
                self._rate_limits[key] = (window_start, count + 1)
            else:
                self._rate_limits[key] = (now, 1)
        else:
            self._rate_limits[key] = (now, 1)

        return True

    def suppress(self, message: str, source: Optional[ErrorSource] = None) -> None:
        """Suppress errors matching this message."""
        if source:
            self._suppressed.add(f"{source.name}:{message}")
        else:
            self._suppressed.add(message)

    def unsuppress(self, message: str, source: Optional[ErrorSource] = None) -> None:
        """Remove suppression."""
        key = f"{source.name}:{message}" if source else message
        self._suppressed.discard(key)

    def get_entries(
        self,
        filter: Optional[ErrorFilter] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[ErrorEntry]:
        """Get filtered entries."""
        with self._lock:
            entries = self._entries.copy()

        if filter:
            entries = [e for e in entries if filter.matches(e)]

        if offset:
            entries = entries[offset:]
        if limit:
            entries = entries[:limit]

        return entries

    def get_recent(self, count: int = 10) -> List[ErrorEntry]:
        """Get most recent entries."""
        with self._lock:
            return list(self._entries[-count:])

    def get_by_severity(self, severity: ErrorSeverity) -> List[ErrorEntry]:
        """Get entries by severity."""
        with self._lock:
            return [e for e in self._entries if e.severity == severity]

    def get_critical(self) -> List[ErrorEntry]:
        """Get critical and fatal errors."""
        with self._lock:
            return [e for e in self._entries if e.is_critical]

    def register_handler(self, handler: ErrorHandler) -> None:
        """Register an error handler."""
        with self._lock:
            if handler not in self._handlers:
                self._handlers.append(handler)

    def unregister_handler(self, handler: ErrorHandler) -> bool:
        """Unregister an error handler."""
        with self._lock:
            if handler in self._handlers:
                self._handlers.remove(handler)
                return True
            return False

    def _notify_handlers(self, entry: ErrorEntry) -> None:
        """Notify all handlers of a new entry."""
        with self._lock:
            handlers = self._handlers.copy()

        for handler in handlers:
            try:
                handler(entry)
            except Exception:
                pass  # Don't let handler errors cascade

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._entries.clear()
            self._stats = ErrorStats()

    def clear_old(self, max_age_seconds: float) -> int:
        """Clear entries older than max_age."""
        cutoff = time.time() - max_age_seconds
        with self._lock:
            old_count = len(self._entries)
            self._entries = [e for e in self._entries if e.timestamp > cutoff]
            return old_count - len(self._entries)


class ErrorPanel:
    """UI-friendly error panel for displaying aggregated errors."""

    def __init__(
        self,
        aggregator: ErrorAggregator,
        visible: bool = True,
        max_visible: int = 50,
        auto_expand_critical: bool = True,
    ):
        self._aggregator = aggregator
        self._visible = visible
        self._max_visible = max_visible
        self._auto_expand_critical = auto_expand_critical
        self._filter = ErrorFilter()
        self._selected_id: Optional[int] = None
        self._expanded: Set[int] = set()
        self._acknowledged: Set[int] = set()
        self._listeners: List[Callable[[], None]] = []

        aggregator.register_handler(self._on_new_error)

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = value
        self._notify_listeners()

    @property
    def filter(self) -> ErrorFilter:
        return self._filter

    @filter.setter
    def filter(self, value: ErrorFilter) -> None:
        self._filter = value
        self._notify_listeners()

    @property
    def selected_entry(self) -> Optional[ErrorEntry]:
        """Get currently selected entry."""
        if self._selected_id is None:
            return None
        entries = self._aggregator.get_entries()
        for entry in entries:
            if entry.id == self._selected_id:
                return entry
        return None

    def get_visible_entries(self) -> List[ErrorEntry]:
        """Get entries visible in the panel."""
        if not self._visible:
            return []
        entries = self._aggregator.get_entries(
            filter=self._filter,
            limit=self._max_visible,
        )
        return [e for e in entries if e.id not in self._acknowledged]

    def select(self, entry_id: int) -> None:
        """Select an entry."""
        self._selected_id = entry_id
        self._notify_listeners()

    def expand(self, entry_id: int) -> None:
        """Expand an entry to show details."""
        self._expanded.add(entry_id)
        self._notify_listeners()

    def collapse(self, entry_id: int) -> None:
        """Collapse an entry."""
        self._expanded.discard(entry_id)
        self._notify_listeners()

    def toggle_expand(self, entry_id: int) -> None:
        """Toggle expansion of an entry."""
        if entry_id in self._expanded:
            self._expanded.discard(entry_id)
        else:
            self._expanded.add(entry_id)
        self._notify_listeners()

    def is_expanded(self, entry_id: int) -> bool:
        """Check if an entry is expanded."""
        return entry_id in self._expanded

    def acknowledge(self, entry_id: int) -> None:
        """Acknowledge an error (hide from panel)."""
        self._acknowledged.add(entry_id)
        self._notify_listeners()

    def acknowledge_all(self) -> None:
        """Acknowledge all visible errors."""
        for entry in self.get_visible_entries():
            self._acknowledged.add(entry.id)
        self._notify_listeners()

    def unacknowledge(self, entry_id: int) -> None:
        """Remove acknowledgement."""
        self._acknowledged.discard(entry_id)
        self._notify_listeners()

    def clear_acknowledged(self) -> None:
        """Clear all acknowledgements."""
        self._acknowledged.clear()
        self._notify_listeners()

    def _on_new_error(self, entry: ErrorEntry) -> None:
        """Handle new error from aggregator."""
        if self._auto_expand_critical and entry.is_critical:
            self._expanded.add(entry.id)
        self._notify_listeners()

    def add_listener(self, listener: Callable[[], None]) -> None:
        """Add a change listener."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[], None]) -> bool:
        """Remove a change listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)
            return True
        return False

    def _notify_listeners(self) -> None:
        """Notify all listeners of a change."""
        for listener in self._listeners:
            try:
                listener()
            except Exception:
                pass

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of panel state."""
        entries = self.get_visible_entries()
        return {
            'visible': self._visible,
            'entry_count': len(entries),
            'critical_count': sum(1 for e in entries if e.is_critical),
            'acknowledged_count': len(self._acknowledged),
            'selected_id': self._selected_id,
            'expanded_count': len(self._expanded),
        }

    def format_entry(
        self,
        entry: ErrorEntry,
        include_context: bool = False,
        include_stack: bool = False,
    ) -> str:
        """Format an entry for display."""
        lines = [
            f"[{entry.severity.name}] {entry.message}",
            f"  Source: {entry.source.name} | Kind: {entry.kind.name}",
            f"  Time: {time.strftime('%H:%M:%S', time.localtime(entry.timestamp))}",
        ]
        if entry.count > 1:
            lines.append(f"  Occurrences: {entry.count}")
        if include_context and entry.context:
            lines.append(f"  Context: {entry.context}")
        if include_stack and entry.stack_trace:
            lines.append(f"  Stack:\n{entry.stack_trace}")
        return "\n".join(lines)


_global_aggregator: Optional[ErrorAggregator] = None
_aggregator_lock = threading.Lock()


def get_global_aggregator() -> ErrorAggregator:
    """Get the global error aggregator."""
    global _global_aggregator
    with _aggregator_lock:
        if _global_aggregator is None:
            _global_aggregator = ErrorAggregator()
        return _global_aggregator


def set_global_aggregator(aggregator: ErrorAggregator) -> None:
    """Set the global error aggregator."""
    global _global_aggregator
    with _aggregator_lock:
        _global_aggregator = aggregator


def report_error(
    message: str,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    source: ErrorSource = ErrorSource.UNKNOWN,
    kind: ErrorKind = ErrorKind.UNKNOWN,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[ErrorEntry]:
    """Report an error to the global aggregator."""
    return get_global_aggregator().add(
        message=message,
        severity=severity,
        source=source,
        kind=kind,
        context=context,
    )


def report_warning(
    message: str,
    source: ErrorSource = ErrorSource.UNKNOWN,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[ErrorEntry]:
    """Report a warning to the global aggregator."""
    return report_error(
        message=message,
        severity=ErrorSeverity.WARNING,
        source=source,
        context=context,
    )


def report_critical(
    message: str,
    source: ErrorSource = ErrorSource.UNKNOWN,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[ErrorEntry]:
    """Report a critical error to the global aggregator."""
    return report_error(
        message=message,
        severity=ErrorSeverity.CRITICAL,
        source=source,
        context=context,
    )

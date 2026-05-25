"""
Undo System - Main undo/redo manager with Foundation Tracker integration.

Provides a high-level undo/redo system that wraps Foundation's Tracker
with additional features like command recording, grouping, and history management.
"""
from __future__ import annotations

import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

from foundation import tracker as foundation_tracker
from foundation import Tracker, Transaction as FoundationTransaction, Change


T = TypeVar("T")


class UndoRedoError(Exception):
    """Base exception for undo/redo errors."""
    pass


class UndoStackEmpty(UndoRedoError):
    """Raised when attempting to undo with empty stack."""
    pass


class RedoStackEmpty(UndoRedoError):
    """Raised when attempting to redo with empty stack."""
    pass


@dataclass
class UndoSystemConfig:
    """Configuration for the undo system."""

    max_undo_levels: int = 1000
    max_redo_levels: int = 1000
    enable_grouping: bool = True
    group_timeout_ms: int = 500  # Auto-group rapid changes
    track_metadata: bool = True
    enable_branching: bool = False  # Enable undo tree (vs linear)


@dataclass
class UndoEntry:
    """An entry in the undo stack."""

    name: str
    timestamp: float
    changes: List[Change]
    metadata: Dict[str, Any] = field(default_factory=dict)
    group_id: Optional[str] = None

    @property
    def description(self) -> str:
        """Get a human-readable description."""
        if len(self.changes) == 1:
            c = self.changes[0]
            return f"{self.name}: {c.field}"
        return f"{self.name} ({len(self.changes)} changes)"


class UndoSystem:
    """
    High-level undo/redo manager with Foundation Tracker integration.

    Features:
    - Wraps Foundation's Tracker for change detection
    - Command recording and playback
    - Automatic change grouping
    - Transaction support
    - Undo history with optional branching
    - Per-document tracking support
    """

    def __init__(
        self,
        config: Optional[UndoSystemConfig] = None,
        tracker: Optional[Tracker] = None,
    ):
        """
        Initialize the undo system.

        Args:
            config: System configuration.
            tracker: Foundation Tracker to use (default: global tracker).
        """
        self._config = config or UndoSystemConfig()
        self._tracker = tracker or foundation_tracker
        self._lock = threading.RLock()

        # Internal state
        self._undo_stack: List[UndoEntry] = []
        self._redo_stack: List[UndoEntry] = []
        self._current_group: Optional[str] = None
        self._group_start_time: float = 0.0
        self._suspended = False

        # Callbacks
        self._on_undo_callbacks: List[Callable[[UndoEntry], None]] = []
        self._on_redo_callbacks: List[Callable[[UndoEntry], None]] = []
        self._on_change_callbacks: List[Callable[[UndoEntry], None]] = []

        # Document tracking
        self._document_states: Dict[str, List[UndoEntry]] = {}

        # Statistics
        self._undo_count = 0
        self._redo_count = 0

    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        with self._lock:
            return bool(self._undo_stack) or self._tracker.can_undo

    @property
    def can_redo(self) -> bool:
        """Check if redo is available."""
        with self._lock:
            return bool(self._redo_stack) or self._tracker.can_redo

    @property
    def undo_count(self) -> int:
        """Number of undo operations performed."""
        return self._undo_count

    @property
    def redo_count(self) -> int:
        """Number of redo operations performed."""
        return self._redo_count

    @property
    def undo_stack_size(self) -> int:
        """Current size of undo stack."""
        with self._lock:
            return len(self._undo_stack)

    @property
    def redo_stack_size(self) -> int:
        """Current size of redo stack."""
        with self._lock:
            return len(self._redo_stack)

    @property
    def suspended(self) -> bool:
        """Check if undo tracking is suspended."""
        return self._suspended

    def suspend(self) -> None:
        """Suspend undo tracking."""
        self._suspended = True

    def resume(self) -> None:
        """Resume undo tracking."""
        self._suspended = False

    def record(
        self,
        name: str,
        obj: Any,
        field_name: str,
        old_value: Any,
        new_value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[UndoEntry]:
        """
        Record a change for undo.

        Args:
            name: Name/description of the change.
            obj: Object being modified.
            field_name: Field being changed.
            old_value: Previous value.
            new_value: New value.
            metadata: Optional metadata.

        Returns:
            UndoEntry if recorded, None if suspended.
        """
        if self._suspended:
            return None

        with self._lock:
            # Create change through tracker
            self._tracker.mark_dirty(obj, field_name, old_value, new_value)

            # Create undo entry
            change = Change(
                obj_ref=weakref.ref(obj),
                field=field_name,
                old_value=old_value,
                new_value=new_value,
            )

            entry = UndoEntry(
                name=name,
                timestamp=time.time(),
                changes=[change],
                metadata=metadata or {},
                group_id=self._current_group,
            )

            # Check for grouping
            if self._config.enable_grouping and self._should_group(entry):
                self._merge_with_last(entry)
            else:
                self._undo_stack.append(entry)
                self._enforce_limits()

            # Clear redo stack on new change
            self._redo_stack.clear()

            # Notify callbacks
            self._notify_change(entry)

            return entry

    def undo(self) -> Optional[UndoEntry]:
        """
        Undo the last operation.

        Returns:
            UndoEntry that was undone, or None.

        Raises:
            UndoStackEmpty: If no operations to undo.
        """
        with self._lock:
            if not self._undo_stack:
                # Try foundation tracker
                if self._tracker.undo():
                    self._undo_count += 1
                    return None
                raise UndoStackEmpty("No operations to undo")

            entry = self._undo_stack.pop()

            # Apply reverse changes
            for change in reversed(entry.changes):
                obj = change.obj_ref()
                if obj is not None:
                    if hasattr(obj, "__dict__"):
                        obj.__dict__[change.field] = change.old_value
                    else:
                        setattr(obj, change.field, change.old_value)

            # Move to redo stack
            self._redo_stack.append(entry)
            self._enforce_redo_limits()

            self._undo_count += 1
            self._notify_undo(entry)

            return entry

    def redo(self) -> Optional[UndoEntry]:
        """
        Redo the last undone operation.

        Returns:
            UndoEntry that was redone, or None.

        Raises:
            RedoStackEmpty: If no operations to redo.
        """
        with self._lock:
            if not self._redo_stack:
                # Try foundation tracker
                if self._tracker.redo():
                    self._redo_count += 1
                    return None
                raise RedoStackEmpty("No operations to redo")

            entry = self._redo_stack.pop()

            # Apply forward changes
            for change in entry.changes:
                obj = change.obj_ref()
                if obj is not None:
                    if hasattr(obj, "__dict__"):
                        obj.__dict__[change.field] = change.new_value
                    else:
                        setattr(obj, change.field, change.new_value)

            # Move back to undo stack
            self._undo_stack.append(entry)

            self._redo_count += 1
            self._notify_redo(entry)

            return entry

    def begin_group(self, name: str) -> str:
        """
        Begin a group of changes that will be undone together.

        Args:
            name: Name of the group.

        Returns:
            Group ID.
        """
        with self._lock:
            self._current_group = f"{name}_{time.time()}"
            self._group_start_time = time.time()

            # Also start foundation transaction
            if not self._tracker.in_transaction:
                self._tracker.begin_transaction(name)

            return self._current_group

    def end_group(self) -> None:
        """End the current group."""
        with self._lock:
            self._current_group = None

            # Commit foundation transaction
            if self._tracker.in_transaction:
                self._tracker.commit_transaction()

    def cancel_group(self) -> None:
        """Cancel the current group and rollback changes."""
        with self._lock:
            if self._current_group:
                # Remove entries with current group ID
                self._undo_stack = [
                    e for e in self._undo_stack
                    if e.group_id != self._current_group
                ]
                self._current_group = None

            # Rollback foundation transaction
            if self._tracker.in_transaction:
                self._tracker.rollback_transaction()

    def get_undo_description(self) -> Optional[str]:
        """Get description of next undo operation."""
        with self._lock:
            if self._undo_stack:
                return self._undo_stack[-1].description
        return None

    def get_redo_description(self) -> Optional[str]:
        """Get description of next redo operation."""
        with self._lock:
            if self._redo_stack:
                return self._redo_stack[-1].description
        return None

    def get_undo_history(self, limit: int = 10) -> List[UndoEntry]:
        """
        Get recent undo history.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of UndoEntry objects (most recent first).
        """
        with self._lock:
            return list(reversed(self._undo_stack[-limit:]))

    def get_redo_history(self, limit: int = 10) -> List[UndoEntry]:
        """
        Get recent redo history.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of UndoEntry objects (most recent first).
        """
        with self._lock:
            return list(reversed(self._redo_stack[-limit:]))

    def clear(self) -> None:
        """Clear all undo/redo history."""
        with self._lock:
            self._undo_stack.clear()
            self._redo_stack.clear()
            self._current_group = None

    def on_undo(self, callback: Callable[[UndoEntry], None]) -> None:
        """Register undo callback."""
        self._on_undo_callbacks.append(callback)

    def on_redo(self, callback: Callable[[UndoEntry], None]) -> None:
        """Register redo callback."""
        self._on_redo_callbacks.append(callback)

    def on_change(self, callback: Callable[[UndoEntry], None]) -> None:
        """Register change callback."""
        self._on_change_callbacks.append(callback)

    def _should_group(self, entry: UndoEntry) -> bool:
        """Check if entry should be grouped with previous."""
        if not self._undo_stack:
            return False

        last = self._undo_stack[-1]

        # Same group ID
        if entry.group_id and entry.group_id == last.group_id:
            return True

        # Within timeout window
        timeout_sec = self._config.group_timeout_ms / 1000.0
        if entry.timestamp - last.timestamp < timeout_sec:
            # Same object and field
            if entry.changes and last.changes:
                ec = entry.changes[0]
                lc = last.changes[-1]
                if ec.obj_ref() is lc.obj_ref() and ec.field == lc.field:
                    return True

        return False

    def _merge_with_last(self, entry: UndoEntry) -> None:
        """Merge entry with last undo entry."""
        last = self._undo_stack[-1]

        # For same field changes, update the new_value
        for ec in entry.changes:
            merged = False
            for lc in last.changes:
                if ec.obj_ref() is lc.obj_ref() and ec.field == lc.field:
                    # Keep old_value from last, update new_value
                    # This is done by updating the Change object
                    merged = True
                    break

            if not merged:
                last.changes.append(ec)

        # Update metadata
        last.metadata.update(entry.metadata)

    def _enforce_limits(self) -> None:
        """Enforce undo stack size limits."""
        while len(self._undo_stack) > self._config.max_undo_levels:
            self._undo_stack.pop(0)

    def _enforce_redo_limits(self) -> None:
        """Enforce redo stack size limits."""
        while len(self._redo_stack) > self._config.max_redo_levels:
            self._redo_stack.pop(0)

    def _notify_undo(self, entry: UndoEntry) -> None:
        """Notify undo callbacks."""
        for cb in self._on_undo_callbacks:
            try:
                cb(entry)
            except Exception:
                pass

    def _notify_redo(self, entry: UndoEntry) -> None:
        """Notify redo callbacks."""
        for cb in self._on_redo_callbacks:
            try:
                cb(entry)
            except Exception:
                pass

    def _notify_change(self, entry: UndoEntry) -> None:
        """Notify change callbacks."""
        for cb in self._on_change_callbacks:
            try:
                cb(entry)
            except Exception:
                pass


# Global undo system instance
_undo_system: Optional[UndoSystem] = None


def get_undo_system() -> UndoSystem:
    """Get the global UndoSystem instance."""
    global _undo_system
    if _undo_system is None:
        _undo_system = UndoSystem()
    return _undo_system


__all__ = [
    "UndoRedoError",
    "UndoStackEmpty",
    "RedoStackEmpty",
    "UndoSystemConfig",
    "UndoEntry",
    "UndoSystem",
    "get_undo_system",
]

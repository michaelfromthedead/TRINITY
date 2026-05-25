"""
Dirty Tracking - Track dirty state for save prompts per document/scene.

Provides mechanisms to track whether documents have unsaved changes
and prompt users appropriately.
"""
from __future__ import annotations

import threading
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

from foundation import tracker as foundation_tracker


class DirtyState(Enum):
    """State of a document's dirty status."""
    CLEAN = auto()
    DIRTY = auto()
    SAVING = auto()
    ERROR = auto()


class SavePromptResult(Enum):
    """Result of a save prompt."""
    SAVE = auto()
    DONT_SAVE = auto()
    CANCEL = auto()


@dataclass
class DirtyInfo:
    """Information about a dirty document."""

    document_id: str
    document_name: str
    state: DirtyState
    dirty_since: Optional[float] = None
    last_saved: Optional[float] = None
    change_count: int = 0
    dirty_fields: Set[str] = field(default_factory=set)

    @property
    def is_dirty(self) -> bool:
        """Check if document is dirty."""
        return self.state == DirtyState.DIRTY

    @property
    def unsaved_duration(self) -> float:
        """Time since document became dirty (seconds)."""
        if self.dirty_since:
            return time.time() - self.dirty_since
        return 0.0

    @property
    def time_since_save(self) -> Optional[float]:
        """Time since last save (seconds)."""
        if self.last_saved:
            return time.time() - self.last_saved
        return None


class DirtyTracker:
    """
    Tracks dirty state for objects.

    Integrates with Foundation's Tracker to automatically detect
    when objects are modified.
    """

    def __init__(self, auto_subscribe: bool = True):
        """
        Initialize the dirty tracker.

        Args:
            auto_subscribe: Automatically subscribe to Foundation's Tracker.
        """
        self._info: Dict[int, DirtyInfo] = {}  # obj_id -> DirtyInfo
        self._tracked: Dict[int, weakref.ref] = {}
        self._lock = threading.RLock()
        self._on_dirty_callbacks: List[Callable[[Any, DirtyInfo], None]] = []
        self._on_clean_callbacks: List[Callable[[Any, DirtyInfo], None]] = []

        if auto_subscribe:
            foundation_tracker.on_change(self._on_tracker_change)

    def track(self, obj: Any, document_id: str, document_name: str = "") -> DirtyInfo:
        """
        Start tracking an object for dirty state.

        Args:
            obj: Object to track.
            document_id: Unique document identifier.
            document_name: Human-readable name.

        Returns:
            DirtyInfo for the object.
        """
        obj_id = id(obj)

        with self._lock:
            info = DirtyInfo(
                document_id=document_id,
                document_name=document_name or document_id,
                state=DirtyState.CLEAN,
            )
            self._info[obj_id] = info
            self._tracked[obj_id] = weakref.ref(obj)

            return info

    def untrack(self, obj: Any) -> bool:
        """
        Stop tracking an object.

        Args:
            obj: Object to stop tracking.

        Returns:
            True if object was being tracked.
        """
        obj_id = id(obj)

        with self._lock:
            if obj_id in self._info:
                del self._info[obj_id]
                self._tracked.pop(obj_id, None)
                return True
            return False

    def mark_dirty(self, obj: Any, field_name: Optional[str] = None) -> None:
        """
        Mark an object as dirty.

        Args:
            obj: Object to mark dirty.
            field_name: Optional specific field that changed.
        """
        obj_id = id(obj)

        with self._lock:
            info = self._info.get(obj_id)
            if not info:
                return

            was_clean = info.state == DirtyState.CLEAN

            info.state = DirtyState.DIRTY
            info.change_count += 1

            if field_name:
                info.dirty_fields.add(field_name)

            if was_clean:
                info.dirty_since = time.time()
                self._notify_dirty(obj, info)

    def mark_clean(self, obj: Any) -> None:
        """
        Mark an object as clean (saved).

        Args:
            obj: Object to mark clean.
        """
        obj_id = id(obj)

        with self._lock:
            info = self._info.get(obj_id)
            if not info:
                return

            was_dirty = info.state == DirtyState.DIRTY

            info.state = DirtyState.CLEAN
            info.last_saved = time.time()
            info.dirty_since = None
            info.dirty_fields.clear()

            if was_dirty:
                self._notify_clean(obj, info)

    def is_dirty(self, obj: Any) -> bool:
        """Check if an object is dirty."""
        obj_id = id(obj)
        with self._lock:
            info = self._info.get(obj_id)
            return info.is_dirty if info else False

    def get_info(self, obj: Any) -> Optional[DirtyInfo]:
        """Get dirty info for an object."""
        obj_id = id(obj)
        with self._lock:
            return self._info.get(obj_id)

    def get_all_dirty(self) -> List[tuple[Any, DirtyInfo]]:
        """Get all dirty objects with their info."""
        result = []
        with self._lock:
            for obj_id, info in self._info.items():
                if info.is_dirty:
                    ref = self._tracked.get(obj_id)
                    if ref:
                        obj = ref()
                        if obj is not None:
                            result.append((obj, info))
        return result

    def any_dirty(self) -> bool:
        """Check if any tracked objects are dirty."""
        with self._lock:
            return any(info.is_dirty for info in self._info.values())

    def on_dirty(self, callback: Callable[[Any, DirtyInfo], None]) -> None:
        """Register callback for when objects become dirty."""
        self._on_dirty_callbacks.append(callback)

    def on_clean(self, callback: Callable[[Any, DirtyInfo], None]) -> None:
        """Register callback for when objects become clean."""
        self._on_clean_callbacks.append(callback)

    def _on_tracker_change(
        self,
        obj: Any,
        field_name: str,
        old_value: Any,
        new_value: Any,
    ) -> None:
        """Handle change notification from Foundation's Tracker."""
        self.mark_dirty(obj, field_name)

    def _notify_dirty(self, obj: Any, info: DirtyInfo) -> None:
        """Notify dirty callbacks."""
        for cb in self._on_dirty_callbacks:
            try:
                cb(obj, info)
            except Exception:
                pass

    def _notify_clean(self, obj: Any, info: DirtyInfo) -> None:
        """Notify clean callbacks."""
        for cb in self._on_clean_callbacks:
            try:
                cb(obj, info)
            except Exception:
                pass


class DocumentDirtyTracker:
    """
    Higher-level dirty tracking for documents with save prompts.

    Manages multiple documents and provides save prompt functionality.
    """

    def __init__(
        self,
        prompt_callback: Optional[Callable[[str], SavePromptResult]] = None,
    ):
        """
        Initialize the document tracker.

        Args:
            prompt_callback: Callback to prompt user about saving.
                Takes document name, returns SavePromptResult.
        """
        self._tracker = DirtyTracker(auto_subscribe=True)
        self._documents: Dict[str, weakref.ref] = {}
        self._prompt_callback = prompt_callback
        self._lock = threading.RLock()

    @property
    def dirty_count(self) -> int:
        """Number of dirty documents."""
        return len(self._tracker.get_all_dirty())

    def register_document(
        self,
        document: Any,
        document_id: str,
        document_name: str = "",
    ) -> None:
        """
        Register a document for tracking.

        Args:
            document: Document object.
            document_id: Unique identifier.
            document_name: Human-readable name.
        """
        with self._lock:
            self._documents[document_id] = weakref.ref(document)
            self._tracker.track(document, document_id, document_name)

    def unregister_document(self, document_id: str) -> None:
        """Unregister a document."""
        with self._lock:
            ref = self._documents.pop(document_id, None)
            if ref:
                doc = ref()
                if doc:
                    self._tracker.untrack(doc)

    def mark_saved(self, document_id: str) -> None:
        """Mark a document as saved."""
        with self._lock:
            ref = self._documents.get(document_id)
            if ref:
                doc = ref()
                if doc:
                    self._tracker.mark_clean(doc)

    def is_dirty(self, document_id: str) -> bool:
        """Check if a document is dirty."""
        with self._lock:
            ref = self._documents.get(document_id)
            if ref:
                doc = ref()
                if doc:
                    return self._tracker.is_dirty(doc)
            return False

    def get_dirty_documents(self) -> List[str]:
        """Get list of dirty document IDs."""
        result = []
        with self._lock:
            for doc_id, ref in self._documents.items():
                doc = ref()
                if doc and self._tracker.is_dirty(doc):
                    result.append(doc_id)
        return result

    def prompt_save_all(self) -> Dict[str, SavePromptResult]:
        """
        Prompt to save all dirty documents.

        Returns:
            Dict mapping document_id to user's choice.
        """
        results = {}
        dirty = self.get_dirty_documents()

        for doc_id in dirty:
            info = self._get_info(doc_id)
            if info:
                result = self._prompt_save(info.document_name)
                results[doc_id] = result

                if result == SavePromptResult.CANCEL:
                    break

        return results

    def can_close(self, document_id: str) -> tuple[bool, Optional[SavePromptResult]]:
        """
        Check if document can be closed, prompting if dirty.

        Args:
            document_id: Document to check.

        Returns:
            Tuple of (can_close, prompt_result).
        """
        if not self.is_dirty(document_id):
            return True, None

        info = self._get_info(document_id)
        if not info:
            return True, None

        result = self._prompt_save(info.document_name)

        if result == SavePromptResult.CANCEL:
            return False, result
        elif result == SavePromptResult.SAVE:
            return True, result  # Caller should save first
        else:
            return True, result  # Don't save

    def can_close_all(self) -> tuple[bool, Dict[str, SavePromptResult]]:
        """
        Check if all documents can be closed.

        Returns:
            Tuple of (can_close_all, prompt_results).
        """
        results = self.prompt_save_all()

        # Can close if no cancellation
        can_close = SavePromptResult.CANCEL not in results.values()
        return can_close, results

    def _get_info(self, document_id: str) -> Optional[DirtyInfo]:
        """Get dirty info for a document."""
        with self._lock:
            ref = self._documents.get(document_id)
            if ref:
                doc = ref()
                if doc:
                    return self._tracker.get_info(doc)
        return None

    def _prompt_save(self, document_name: str) -> SavePromptResult:
        """Prompt user about saving a document."""
        if self._prompt_callback:
            return self._prompt_callback(document_name)
        # Default to don't save if no callback
        return SavePromptResult.DONT_SAVE


__all__ = [
    "DirtyState",
    "SavePromptResult",
    "DirtyInfo",
    "DirtyTracker",
    "DocumentDirtyTracker",
]

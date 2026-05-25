"""
Tracker - Centralized change tracking system. Core Foundation Layer 2.
Provides dirty flags, change subscriptions, transactions, and undo/redo.
"""
from __future__ import annotations
import threading, time, weakref, warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

from foundation.constants import MAX_UNDO_STACK_SIZE, MAX_REDO_STACK_SIZE

ChangeCallback = Callable[[Any, str, Any, Any], None]

@dataclass
class Change:
    """A single field change record."""
    obj_ref: weakref.ref
    field: str
    old_value: Any
    new_value: Any
    @property
    def obj(self) -> Optional[Any]: return self.obj_ref()

@dataclass
class Transaction:
    """A group of changes that can be undone atomically."""
    name: str
    changes: list[Change] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

class Tracker:
    """Centralized change tracking with dirty flags, events, transactions, undo/redo."""
    __slots__ = ("_dirty", "_cb_global", "_cb_obj", "_cb_type", "_txn", "_undo", "_redo", "_lock")

    def __init__(self):
        self._dirty: dict[int, tuple[weakref.ref, set[str]]] = {}
        self._cb_global: list[ChangeCallback] = []
        self._cb_obj: dict[int, list[ChangeCallback]] = {}
        self._cb_type: dict[type, list[ChangeCallback]] = {}
        self._txn: Optional[Transaction] = None
        self._undo: list[Transaction] = []
        self._redo: list[Transaction] = []
        self._lock = threading.RLock()

    def is_dirty(self, obj: Any) -> bool:
        """Check if an object has any dirty fields."""
        with self._lock:
            e = self._dirty.get(id(obj))
            return bool(e and e[1])

    def dirty_fields(self, obj: Any) -> set[str]:
        """Get set of dirty field names for an object."""
        with self._lock:
            e = self._dirty.get(id(obj))
            return set(e[1]) if e else set()

    def mark_dirty(self, obj: Any, field_name: str, old_value: Any, new_value: Any) -> None:
        """Mark a field as dirty and record the change."""
        with self._lock:
            oid = id(obj)
            if oid not in self._dirty:
                self._dirty[oid] = (weakref.ref(obj, lambda _: self._cleanup(oid)), set())
            self._dirty[oid][1].add(field_name)
            c = Change(weakref.ref(obj), field_name, old_value, new_value)
            if self._txn: self._coalesce(c)
            else:
                self._undo.append(Transaction(f"Set {field_name}", [c]))
                # Enforce undo stack bounds
                while len(self._undo) > MAX_UNDO_STACK_SIZE:
                    self._undo.pop(0)
                self._redo.clear()
            self._notify(obj, field_name, old_value, new_value)

    def mark_clean(self, obj: Any) -> None:
        """Clear all dirty flags for an object."""
        with self._lock:
            e = self._dirty.get(id(obj))
            if e: e[1].clear()

    def all_dirty(self) -> list[Any]:
        """Return list of all dirty objects."""
        with self._lock:
            result, dead = [], []
            for oid, (ref, fields) in self._dirty.items():
                o = ref()
                if o is None: dead.append(oid)
                elif fields: result.append(o)
            for oid in dead: del self._dirty[oid]
            return result

    def _cleanup(self, oid: int) -> None:
        with self._lock:
            self._dirty.pop(oid, None)
            self._cb_obj.pop(oid, None)

    def on_change(self, target: Union[None, Any, type] = None, callback: Optional[ChangeCallback] = None) -> None:
        """Subscribe to changes. on_change(cb) | on_change(obj, cb) | on_change(cls, cb)"""
        if callback is None and callable(target): callback, target = target, None
        if not callback: raise ValueError("callback required")
        with self._lock:
            if target is None:
                if callback not in self._cb_global: self._cb_global.append(callback)
            elif isinstance(target, type):
                self._cb_type.setdefault(target, [])
                if callback not in self._cb_type[target]: self._cb_type[target].append(callback)
            else:
                oid = id(target)
                self._cb_obj.setdefault(oid, [])
                if callback not in self._cb_obj[oid]: self._cb_obj[oid].append(callback)

    def off_change(self, callback: ChangeCallback) -> None:
        """Unsubscribe a callback from all subscriptions."""
        with self._lock:
            if callback in self._cb_global: self._cb_global.remove(callback)
            for cbs in list(self._cb_obj.values()) + list(self._cb_type.values()):
                if callback in cbs: cbs.remove(callback)

    def _notify(self, obj: Any, fld: str, old: Any, new: Any) -> None:
        for cb in list(self._cb_global):
            try:
                cb(obj, fld, old, new)
            except Exception as e:
                warnings.warn(f"Tracker global callback failed: {e}", RuntimeWarning)
        for cb in list(self._cb_obj.get(id(obj), [])):
            try:
                cb(obj, fld, old, new)
            except Exception as e:
                warnings.warn(f"Tracker object callback failed: {e}", RuntimeWarning)
        for cls, cbs in self._cb_type.items():
            if isinstance(obj, cls):
                for cb in list(cbs):
                    try:
                        cb(obj, fld, old, new)
                    except Exception as e:
                        warnings.warn(f"Tracker type callback failed: {e}", RuntimeWarning)

    def begin_transaction(self, name: str) -> None:
        """Begin a new transaction for grouping changes."""
        with self._lock:
            if self._txn: raise RuntimeError("Nested transactions not supported")
            self._txn = Transaction(name=name)

    def commit_transaction(self) -> None:
        """Commit the current transaction."""
        with self._lock:
            if not self._txn: raise RuntimeError("No active transaction")
            if self._txn.changes:
                self._undo.append(self._txn)
                # Enforce undo stack bounds
                while len(self._undo) > MAX_UNDO_STACK_SIZE:
                    self._undo.pop(0)
                self._redo.clear()
            self._txn = None

    def rollback_transaction(self) -> None:
        """Rollback the current transaction, reverting all changes."""
        with self._lock:
            if not self._txn: raise RuntimeError("No active transaction")
            for c in reversed(self._txn.changes): self._apply(c, True)
            self._txn = None

    @property
    def in_transaction(self) -> bool:
        with self._lock: return self._txn is not None

    def _coalesce(self, c: Change) -> None:
        for ex in self._txn.changes:
            if ex.obj_ref() is c.obj_ref() and ex.field == c.field:
                ex.new_value = c.new_value
                return
        self._txn.changes.append(Change(c.obj_ref, c.field, c.old_value, c.new_value))

    def undo(self) -> bool:
        """Undo the last transaction. Returns True if something was undone."""
        with self._lock:
            if not self._undo: return False
            t = self._undo.pop()
            for c in reversed(t.changes): self._apply(c, True)
            self._redo.append(t)
            # Enforce redo stack bounds
            while len(self._redo) > MAX_REDO_STACK_SIZE:
                self._redo.pop(0)
            return True

    def redo(self) -> bool:
        """Redo the last undone transaction. Returns True if something was redone."""
        with self._lock:
            if not self._redo: return False
            t = self._redo.pop()
            for c in t.changes: self._apply(c, False)
            self._undo.append(t)
            return True

    @property
    def can_undo(self) -> bool:
        with self._lock: return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        with self._lock: return bool(self._redo)

    @property
    def undo_stack(self) -> list[Transaction]:
        with self._lock: return list(self._undo)

    @property
    def redo_stack(self) -> list[Transaction]:
        with self._lock: return list(self._redo)

    def _apply(self, c: Change, reverse: bool) -> None:
        obj = c.obj_ref()
        if obj is None: return
        val = c.old_value if reverse else c.new_value
        if hasattr(obj, "__dict__"): obj.__dict__[c.field] = val
        else: setattr(obj, c.field, val)

tracker = Tracker()
__all__ = ["Change", "Transaction", "Tracker", "tracker", "ChangeCallback"]

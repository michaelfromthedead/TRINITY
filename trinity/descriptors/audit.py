"""
Audit descriptor - logs all gets and sets with timestamps.

Provides an append-only audit trail for field access.
"""

from __future__ import annotations

import time
from typing import Any, Optional, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")

__all__ = ["AuditDescriptor", "get_audit_log", "clear_audit_log"]

_DEFAULT_MAX_ENTRIES = 1000


class AuditDescriptor(BaseDescriptor[T]):
    """
    Descriptor that maintains an audit log of all field accesses.

    Records timestamped entries for set (and optionally get) operations.
    """

    __slots__ = ("_max_entries", "_log_reads", "_audit_key")

    descriptor_id = "audit"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(
        self,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        log_reads: bool = False,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._max_entries = max_entries
        self._log_reads = log_reads
        self._audit_key = ""

    def __set_name__(self, owner: type, name: str) -> None:
        super().__set_name__(owner, name)
        self._audit_key = f"_audit_{name}"

    def _get_log(self, obj: Any) -> list:
        log = obj.__dict__.get(self._audit_key)
        if log is None:
            log = []
            obj.__dict__[self._audit_key] = log
        return log

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        log = self._get_log(obj)
        log.append((time.time(), "set", old_value, value))
        if len(log) > self._max_entries:
            del log[: len(log) - self._max_entries]

    def post_get(self, obj: Any, value: T) -> T:
        if self._log_reads:
            log = self._get_log(obj)
            log.append((time.time(), "get", value))
            if len(log) > self._max_entries:
                del log[: len(log) - self._max_entries]
        return value

    @property
    def descriptor_steps(self) -> list[Step]:
        return [
            Step(Op.INTERCEPT, {"get": "audit_read", "set": "audit_write"}),
            Step(Op.TAG, {"key": "audited", "value": True}),
        ]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["max_entries"] = self._max_entries
        meta["log_reads"] = self._log_reads
        return meta


def get_audit_log(obj: Any, field: str, limit: Optional[int] = None) -> list:
    """Return audit log entries for a field on an object."""
    key = f"_audit_{field}"
    log = obj.__dict__.get(key, [])
    if limit is not None:
        return log[-limit:]
    return list(log)


def clear_audit_log(obj: Any, field: str) -> None:
    """Clear the audit log for a field on an object."""
    key = f"_audit_{field}"
    obj.__dict__.pop(key, None)

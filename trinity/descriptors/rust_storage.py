"""
Rust-backed storage descriptor — routes component field reads/writes to the
Rust component store instead of obj.__dict__.  Falls back to __dict__ storage
when the Rust backend (`_omega`) is not available.
"""

from __future__ import annotations

from typing import Any

from trinity.descriptors.base import BaseDescriptor


# Lazy _omega import -- only available when the Rust backend (Model C) is linked.
try:
    from _omega import component_read, component_write, component_delete
    _HAVE_OMEGA = True
except ImportError:
    _HAVE_OMEGA = False

# Sentinel for uninitialised defaults.
_UNSET: Any = object()


class RustStorageDescriptor(BaseDescriptor):
    """Storage descriptor that delegates to the Rust ECS component store.

    This is the **innermost** descriptor for Component subclasses when the
    Rust backend (Model C) is active.  For non-Component types it silently
    degrades to ordinary __dict__ storage.
    """

    descriptor_id = "rust_storage"
    accepts_inner = ()      # innermost — nothing wraps inside it
    accepts_outer = ("*",)  # any descriptor can wrap it

    __slots__ = ("_default", "_default_factory", "_rust_offset")

    def __init__(
        self,
        field_type: type = object,
        default: Any = _UNSET,
        default_factory: Any = None,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=None, **config)
        self._default = default
        self._default_factory = default_factory
        self._rust_offset: int | None = None  # set by ComponentMeta

    # ------------------------------------------------------------------
    # Storage protocol (overrides BaseDescriptor)
    # ------------------------------------------------------------------

    def _get_stored(self, obj: Any) -> Any:
        """Fetch value — prefer Rust store, fall back to __dict__."""
        if _HAVE_OMEGA:
            offset = self._rust_offset
            entity_id = getattr(obj, "_entity_id", None)
            component_id = getattr(obj, "_component_id", None)
            if entity_id is not None and component_id is not None and offset is not None:
                try:
                    return component_read(entity_id, component_id, offset, self._field_type)
                except RuntimeError:
                    pass
        # Python-only fallback
        return self._dict_get(obj)

    def _set_stored(self, obj: Any, value: Any) -> None:
        """Store value — prefer Rust store, fall back to __dict__."""
        if _HAVE_OMEGA:
            offset = self._rust_offset
            entity_id = getattr(obj, "_entity_id", None)
            component_id = getattr(obj, "_component_id", None)
            if entity_id is not None and component_id is not None and offset is not None:
                try:
                    component_write(entity_id, component_id, offset, value)
                    return
                except RuntimeError:
                    pass
        # Python-only fallback
        self._dict_set(obj, value)

    def _delete_stored(self, obj: Any) -> None:
        """Delete value -- prefer Rust store, fall back to __dict__."""
        if _HAVE_OMEGA:
            offset = self._rust_offset
            entity_id = getattr(obj, "_entity_id", None)
            component_id = getattr(obj, "_component_id", None)
            if entity_id is not None and component_id is not None and offset is not None:
                try:
                    component_delete(entity_id, component_id, offset)
                except RuntimeError:
                    pass
        # Always clean up __dict__ -- the Rust path may not exist or may fail,
        # and even when it succeeds the Python dict needs the stale entry removed
        # so that _get_stored -> _dict_get returns the resolved default (#C1).
        obj.__dict__.pop(self._name, None)

    # ------------------------------------------------------------------
    # Dict fallback helpers (mirror StorageDescriptor behaviour)
    # ------------------------------------------------------------------

    def _dict_get(self, obj: Any) -> Any:
        if self._name not in obj.__dict__:
            obj.__dict__[self._name] = self._resolve_default()
        return obj.__dict__[self._name]

    def _dict_set(self, obj: Any, value: Any) -> None:
        obj.__dict__[self._name] = value

    def _resolve_default(self) -> Any:
        if self._default_factory is not None:
            return self._default_factory()
        if self._default is not _UNSET:
            return self._default
        return None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def descriptor_steps(self) -> list:
        return []  # storage is passive

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["rust_offset"] = self._rust_offset
        meta["has_default"] = self._default is not _UNSET
        return meta

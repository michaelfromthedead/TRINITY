"""
Storage descriptor - the innermost layer of any descriptor chain.

Provides the actual storage mechanism for field values.
"""

from __future__ import annotations

from typing import Any, Optional, TypeVar

from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")


class StorageDescriptor(BaseDescriptor[T]):
    """
    Base storage descriptor. Should be innermost in any chain.

    Provides:
    - Default value support
    - Default factory support
    - Direct __dict__ storage
    """

    __slots__ = ("_default", "_default_factory")

    descriptor_id = "storage"
    accepts_inner = ()  # Cannot wrap anything (is innermost)
    accepts_outer = ("*",)  # Any descriptor can wrap this
    excludes = ()

    def __init__(
        self,
        field_type: type = object,
        default: Any = None,
        default_factory: Optional[callable] = None,
        **config: Any,
    ) -> None:
        """
        Initialize storage descriptor.

        Args:
            field_type: The type annotation for this field.
            default: Default value (used if default_factory not provided).
            default_factory: Callable that returns default value.
            **config: Additional configuration.
        """
        super().__init__(field_type=field_type, inner=None, **config)
        self._default = default
        self._default_factory = default_factory

    def _get_stored(self, obj: Any) -> T:
        """Get value with default initialization."""
        if self._name not in obj.__dict__:
            # Initialize with default
            if self._default_factory is not None:
                default = self._default_factory()
            else:
                default = self._default
            obj.__dict__[self._name] = default
        return obj.__dict__[self._name]

    @property
    def descriptor_steps(self) -> list:
        return []  # Storage is passive — no ops

    def get_metadata(self) -> dict[str, Any]:
        """Return storage configuration."""
        meta = super().get_metadata()
        meta["has_default"] = self._default is not None
        meta["has_default_factory"] = self._default_factory is not None
        return meta

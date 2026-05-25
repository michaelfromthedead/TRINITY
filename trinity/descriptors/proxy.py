"""
ProxyDescriptor — proxy reads/writes to a field on a target object.
"""

from __future__ import annotations

from typing import Any, Optional

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

__all__ = ["ProxyDescriptor"]


class ProxyDescriptor(BaseDescriptor):
    """Descriptor that proxies attribute access to another object's field."""

    __slots__ = ("_target_cls", "_target_field", "_target_attr")

    descriptor_id: str = "proxy"

    def __init__(
        self,
        target_cls: type,
        target_field: str,
        target_attr: str = "",
        field_type: type = object,
        inner: Optional[BaseDescriptor] = None,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._target_cls = target_cls
        self._target_field = target_field
        self._target_attr = target_attr

    def _get_target(self, obj: Any) -> Any:
        """Find the proxy target object stored on the instance."""
        if self._target_attr:
            target = getattr(obj, self._target_attr, None)
            if target is None:
                raise AttributeError(
                    f"Attribute '{self._target_attr}' not found on {type(obj).__name__}"
                )
            return target
        # Fallback: scan for an attribute matching target_cls type
        for attr_name, attr_val in obj.__dict__.items():
            if isinstance(attr_val, self._target_cls):
                return attr_val
        raise AttributeError(
            f"No attribute of type {self._target_cls.__name__} found on {type(obj).__name__}"
        )

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> Any:
        if obj is None:
            return self
        target = self._get_target(obj)
        return getattr(target, self._target_field)

    def __set__(self, obj: Any, value: Any) -> None:
        target = self._get_target(obj)
        setattr(target, self._target_field, value)

    @property
    def descriptor_steps(self) -> list[Step]:
        return [
            Step(Op.INTERCEPT, {"get": "proxy_read", "set": "proxy_write"}),
        ]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["target_cls"] = self._target_cls.__name__
        meta["target_field"] = self._target_field
        meta["target_attr"] = self._target_attr
        return meta

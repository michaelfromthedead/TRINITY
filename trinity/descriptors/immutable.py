"""
Immutable descriptor - prevents modification after initial set.

Provides freeze-after-init semantics for Trinity fields.
"""

from __future__ import annotations

from typing import Any, Optional, TypeVar

from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")


class ImmutableDescriptor(BaseDescriptor[T]):
    """
    Prevents field modification after the first set.

    Once a value is assigned and frozen, subsequent assignments
    raise AttributeError.
    """

    __slots__ = ("_freeze_after_init",)

    descriptor_id = "immutable"
    excludes = ("tracked", "observable", "networked")

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        freeze_after_init: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._freeze_after_init = freeze_after_init

    def __set__(self, obj: Any, value: T) -> None:
        frozen_attr = f"_frozen_{self._name}"
        if getattr(obj, frozen_attr, False):
            raise AttributeError(
                f"Cannot set immutable field '{self._name}'"
            )
        super().__set__(obj, value)

    def post_set(self, obj: Any, value: T, old_value: Optional[T]) -> None:
        if self._freeze_after_init:
            frozen_attr = f"_frozen_{self._name}"
            if not getattr(obj, frozen_attr, False):
                object.__setattr__(obj, frozen_attr, True)

    @property
    def descriptor_steps(self) -> list:
        from trinity.decorators.ops import Op, Step
        return [Step(Op.INTERCEPT, {"set": "deny_after_init"})]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["freeze_after_init"] = self._freeze_after_init
        return meta


__all__ = ["ImmutableDescriptor"]

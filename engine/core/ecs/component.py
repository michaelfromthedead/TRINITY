"""Component identification and masks."""
from __future__ import annotations

from typing import Type

from engine.core.constants import COMPONENT_ID_MASK

__all__ = ["ComponentId", "component_id", "ComponentMask", "TagComponent"]

ComponentId = int


def component_id(cls: Type) -> ComponentId:
    """Return a stable unique id for a component type based on its identity."""
    # Use id of the class object itself for uniqueness within a process.
    # We cache on the class to keep it stable.
    cid = getattr(cls, "_ecs_component_id", None)
    if cid is None:
        cid = hash(cls) & COMPONENT_ID_MASK
        cls._ecs_component_id = cid  # type: ignore[attr-defined]
    return cid


ComponentMask = frozenset  # frozenset[ComponentId]


class TagComponent:
    """Marker base class for zero-size tag components."""
    __slots__ = ()

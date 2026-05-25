"""Typed generic asset handle with generational index."""
from __future__ import annotations

import enum
from typing import Generic, TypeVar

from engine.resource.constants import (
    ASSET_INDEX_BITS,
    ASSET_GENERATION_BITS,
    ASSET_INDEX_MASK,
    ASSET_GENERATION_MASK,
    NULL_ASSET_INDEX,
)

__all__ = ["AssetHandle", "AssetId", "AssetState"]

AssetId = int
T = TypeVar("T")


class AssetState(enum.Enum):
    """Lifecycle states of an asset."""
    REQUESTED = 0
    QUEUED = 1
    LOADING = 2
    LOADED = 3
    READY = 4
    FAILED = 5
    UNLOADING = 6
    UNLOADED = 7


class AssetHandle(Generic[T]):
    """Lightweight typed handle: index (uint24) + generation (uint8) packed into one int."""
    __slots__ = ("_packed", "_asset_type")

    def __init__(self, index: int, generation: int, asset_type: type[T] | None = None) -> None:
        self._packed: int = (
            ((generation & ASSET_GENERATION_MASK) << ASSET_INDEX_BITS)
            | (index & ASSET_INDEX_MASK)
        )
        self._asset_type: type[T] | None = asset_type

    @classmethod
    def null(cls, asset_type: type[T] | None = None) -> AssetHandle[T]:
        return cls(NULL_ASSET_INDEX, 0, asset_type)

    @classmethod
    def from_packed(cls, packed: int, asset_type: type[T] | None = None) -> AssetHandle[T]:
        handle = object.__new__(cls)
        handle._packed = packed
        handle._asset_type = asset_type
        return handle

    @property
    def index(self) -> int:
        return self._packed & ASSET_INDEX_MASK

    @property
    def generation(self) -> int:
        return (self._packed >> ASSET_INDEX_BITS) & ASSET_GENERATION_MASK

    @property
    def asset_type(self) -> type[T] | None:
        return self._asset_type

    @property
    def id(self) -> AssetId:
        return self._packed

    def is_valid(self) -> bool:
        return self.index != NULL_ASSET_INDEX

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AssetHandle):
            return NotImplemented
        return self._packed == other._packed

    def __hash__(self) -> int:
        return self._packed

    def __repr__(self) -> str:
        if not self.is_valid():
            return "AssetHandle(null)"
        t = self._asset_type.__name__ if self._asset_type else "?"
        return f"AssetHandle(index={self.index}, gen={self.generation}, type={t})"

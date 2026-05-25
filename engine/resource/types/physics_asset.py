"""Physics asset type with collider definitions."""
from __future__ import annotations

from enum import Enum, auto

from engine.resource.constants import DEFAULT_FRICTION, DEFAULT_RESTITUTION
from engine.resource.types.base_asset import BaseAsset

__all__ = ["ColliderType", "PhysicsAsset"]


class ColliderType(Enum):
    """Supported physics collider shapes."""
    BOX = auto()
    SPHERE = auto()
    CAPSULE = auto()
    MESH = auto()
    CONVEX = auto()


class PhysicsAsset(BaseAsset):
    """A physics collider definition asset."""

    __slots__ = (
        "_collider_type", "_dimensions", "_mass",
        "_is_static", "_friction", "_restitution", "_loaded",
    )

    def __init__(
        self,
        asset_id: int,
        name: str,
        path: str,
        size_bytes: int,
        collider_type: ColliderType,
        dimensions: tuple,
        mass: float,
        is_static: bool = False,
        friction: float = DEFAULT_FRICTION,
        restitution: float = DEFAULT_RESTITUTION,
        version: int = 1,
    ) -> None:
        super().__init__(asset_id, name, path, size_bytes, version)
        self._collider_type = collider_type
        self._dimensions = dimensions
        self._mass = mass
        self._is_static = is_static
        self._friction = friction
        self._restitution = restitution
        self._loaded = False

    @property
    def collider_type(self) -> ColliderType:
        return self._collider_type

    @property
    def dimensions(self) -> tuple:
        return self._dimensions

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def is_static(self) -> bool:
        return self._is_static

    @property
    def friction(self) -> float:
        return self._friction

    @property
    def restitution(self) -> float:
        return self._restitution

    # --- BaseAsset interface ---

    def load(self, data: bytes) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

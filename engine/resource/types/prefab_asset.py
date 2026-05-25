"""Prefab asset type for entity templates with hierarchies."""
from __future__ import annotations

from engine.resource.types.base_asset import BaseAsset

__all__ = ["PrefabAsset"]


class PrefabAsset(BaseAsset):
    """A prefab template that can be instantiated into an entity hierarchy."""

    __slots__ = ("_components", "_children", "_loaded")

    def __init__(
        self,
        asset_id: int,
        name: str,
        path: str,
        size_bytes: int,
        components: list[dict] | None = None,
        children: list[PrefabAsset] | None = None,
        version: int = 1,
    ) -> None:
        super().__init__(asset_id, name, path, size_bytes, version)
        self._components: list[dict] = components or []
        self._children: list[PrefabAsset] = children or []
        self._loaded = False

    @property
    def components(self) -> list[dict]:
        return list(self._components)

    @property
    def children(self) -> list[PrefabAsset]:
        return list(self._children)

    def add_child(self, child: PrefabAsset) -> None:
        self._children.append(child)

    # --- BaseAsset interface ---

    def load(self, data: bytes) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    # --- prefab helpers ---

    def instantiate(self) -> dict:
        """Return a dictionary representing an instantiated entity tree."""
        return {
            "asset_id": self.asset_id,
            "name": self.name,
            "components": [dict(c) for c in self._components],
            "children": [child.instantiate() for child in self._children],
        }

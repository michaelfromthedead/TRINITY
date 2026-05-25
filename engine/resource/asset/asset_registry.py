"""Maps file extensions to asset types via a singleton registry."""
from __future__ import annotations

import enum
import os
from typing import ClassVar

__all__ = ["AssetRegistry", "AssetType"]


class AssetType(enum.Enum):
    """Known built-in asset categories."""
    TEXTURE = "texture"
    MESH = "mesh"
    AUDIO = "audio"
    DATA_TABLE = "data_table"
    SHADER = "shader"


# Default extension -> asset type mappings
_DEFAULT_EXTENSIONS: dict[str, AssetType] = {
    ".png": AssetType.TEXTURE,
    ".jpg": AssetType.TEXTURE,
    ".jpeg": AssetType.TEXTURE,
    ".obj": AssetType.MESH,
    ".fbx": AssetType.MESH,
    ".wav": AssetType.AUDIO,
    ".mp3": AssetType.AUDIO,
    ".json": AssetType.DATA_TABLE,
    ".glsl": AssetType.SHADER,
    ".hlsl": AssetType.SHADER,
}


class AssetRegistry:
    """Singleton registry mapping file extensions to asset types."""
    __slots__ = ("_map",)
    _instance: ClassVar[AssetRegistry | None] = None

    def __init__(self) -> None:
        self._map: dict[str, AssetType] = {}

    @classmethod
    def instance(cls) -> AssetRegistry:
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_defaults()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (mainly for tests)."""
        cls._instance = None

    def _register_defaults(self) -> None:
        for ext, atype in _DEFAULT_EXTENSIONS.items():
            self._map[ext] = atype

    def register(self, extension: str, asset_type: AssetType) -> None:
        ext = extension if extension.startswith(".") else f".{extension}"
        self._map[ext] = asset_type

    def lookup(self, path: str) -> AssetType | None:
        _, ext = os.path.splitext(path)
        return self._map.get(ext.lower())

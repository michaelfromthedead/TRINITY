"""Texture asset type with format and mip-level support."""
from __future__ import annotations

import math
from enum import Enum, auto

from engine.resource.types.base_asset import BaseAsset

__all__ = ["TextureFormat", "TextureAsset"]


class TextureFormat(Enum):
    """GPU texture pixel formats."""
    RGBA8 = auto()
    RGB8 = auto()
    BC1 = auto()
    BC3 = auto()
    BC5 = auto()
    BC7 = auto()
    R16F = auto()
    RGBA16F = auto()
    RGBA32F = auto()


# Bytes per pixel (or block-compressed equivalent) for memory estimation.
_BYTES_PER_PIXEL: dict[TextureFormat, float] = {
    TextureFormat.RGBA8: 4.0,
    TextureFormat.RGB8: 3.0,
    TextureFormat.BC1: 0.5,
    TextureFormat.BC3: 1.0,
    TextureFormat.BC5: 1.0,
    TextureFormat.BC7: 1.0,
    TextureFormat.R16F: 2.0,
    TextureFormat.RGBA16F: 8.0,
    TextureFormat.RGBA32F: 16.0,
}


class TextureAsset(BaseAsset):
    """A 2-D texture image asset."""

    __slots__ = (
        "_width", "_height", "_channels", "_mip_levels",
        "_format", "_pixel_data",
    )

    def __init__(
        self,
        asset_id: int,
        name: str,
        path: str,
        size_bytes: int,
        width: int,
        height: int,
        channels: int,
        fmt: TextureFormat,
        mip_levels: int = 1,
        version: int = 1,
    ) -> None:
        super().__init__(asset_id, name, path, size_bytes, version)
        max_mips = int(math.log2(max(width, height))) + 1
        if mip_levels < 1 or mip_levels > max_mips:
            raise ValueError(
                f"mip_levels must be in [1, {max_mips}], got {mip_levels}"
            )
        self._width = width
        self._height = height
        self._channels = channels
        self._mip_levels = mip_levels
        self._format = fmt
        self._pixel_data: bytes | None = None

    # --- properties ---

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def mip_levels(self) -> int:
        return self._mip_levels

    @property
    def format(self) -> TextureFormat:
        return self._format

    @property
    def pixel_data(self) -> bytes | None:
        return self._pixel_data

    @property
    def max_mip_levels(self) -> int:
        return int(math.log2(max(self._width, self._height))) + 1

    # --- BaseAsset interface ---

    def load(self, data: bytes) -> None:
        self._pixel_data = data

    def unload(self) -> None:
        self._pixel_data = None

    def is_loaded(self) -> bool:
        return self._pixel_data is not None

    @property
    def memory_footprint(self) -> int:
        if self._pixel_data is not None:
            return len(self._pixel_data)
        return 0

    # --- texture helpers ---

    def get_mip_size(self, level: int) -> tuple[int, int]:
        """Return (width, height) of the given mip level."""
        if level < 0 or level >= self._mip_levels:
            raise ValueError(
                f"level must be in [0, {self._mip_levels - 1}], got {level}"
            )
        w = max(1, self._width >> level)
        h = max(1, self._height >> level)
        return (w, h)

"""Material asset type with blend modes and render queue."""
from __future__ import annotations

from enum import Enum, auto
from typing import Union

from engine.resource.constants import (
    RENDER_QUEUE_ADDITIVE,
    RENDER_QUEUE_ALPHA_BLEND,
    RENDER_QUEUE_ALPHA_TEST,
    RENDER_QUEUE_OPAQUE,
)
from engine.resource.types.base_asset import BaseAsset

__all__ = ["BlendMode", "MaterialAsset"]

ParameterValue = Union[float, tuple]


class BlendMode(Enum):
    """Material blend modes."""
    OPAQUE = auto()
    ALPHA_TEST = auto()
    ALPHA_BLEND = auto()
    ADDITIVE = auto()


# Default render queue per blend mode.
_DEFAULT_RENDER_QUEUE: dict[BlendMode, int] = {
    BlendMode.OPAQUE: RENDER_QUEUE_OPAQUE,
    BlendMode.ALPHA_TEST: RENDER_QUEUE_ALPHA_TEST,
    BlendMode.ALPHA_BLEND: RENDER_QUEUE_ALPHA_BLEND,
    BlendMode.ADDITIVE: RENDER_QUEUE_ADDITIVE,
}


class MaterialAsset(BaseAsset):
    """A material describing surface shading properties."""

    __slots__ = (
        "_shader_id", "_textures", "_parameters",
        "_blend_mode", "_render_queue", "_loaded",
    )

    def __init__(
        self,
        asset_id: int,
        name: str,
        path: str,
        size_bytes: int,
        shader_id: int,
        blend_mode: BlendMode = BlendMode.OPAQUE,
        render_queue: int | None = None,
        textures: dict[str, int] | None = None,
        parameters: dict[str, ParameterValue] | None = None,
        version: int = 1,
    ) -> None:
        super().__init__(asset_id, name, path, size_bytes, version)
        self._shader_id = shader_id
        self._blend_mode = blend_mode
        self._render_queue = render_queue if render_queue is not None else _DEFAULT_RENDER_QUEUE[blend_mode]
        self._textures: dict[str, int] = textures or {}
        self._parameters: dict[str, ParameterValue] = parameters or {}
        self._loaded = False

    @property
    def shader_id(self) -> int:
        return self._shader_id

    @property
    def textures(self) -> dict[str, int]:
        return dict(self._textures)

    @property
    def parameters(self) -> dict[str, ParameterValue]:
        return dict(self._parameters)

    @property
    def blend_mode(self) -> BlendMode:
        return self._blend_mode

    @property
    def render_queue(self) -> int:
        return self._render_queue

    def set_texture(self, slot: str, texture_asset_id: int) -> None:
        self._textures[slot] = texture_asset_id

    def set_parameter(self, key: str, value: ParameterValue) -> None:
        self._parameters[key] = value

    # --- BaseAsset interface ---

    def load(self, data: bytes) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

"""Shader asset type with pipeline stages."""
from __future__ import annotations

import re
from enum import Enum, auto

from engine.resource.types.base_asset import BaseAsset

__all__ = ["ShaderStage", "ShaderAsset"]


class ShaderStage(Enum):
    """GPU pipeline shader stages."""
    VERTEX = auto()
    FRAGMENT = auto()
    COMPUTE = auto()
    GEOMETRY = auto()
    TESSELLATION = auto()


# Regex to extract uniform declarations from GLSL-like source.
_UNIFORM_PATTERN = re.compile(r"uniform\s+\w+\s+(\w+)")


class ShaderAsset(BaseAsset):
    """A single shader stage asset."""

    __slots__ = ("_stage", "_source_code", "_compiled_binary")

    def __init__(
        self,
        asset_id: int,
        name: str,
        path: str,
        size_bytes: int,
        stage: ShaderStage,
        source_code: str | None = None,
        version: int = 1,
    ) -> None:
        super().__init__(asset_id, name, path, size_bytes, version)
        self._stage = stage
        self._source_code = source_code
        self._compiled_binary: bytes | None = None

    @property
    def stage(self) -> ShaderStage:
        return self._stage

    @property
    def source_code(self) -> str | None:
        return self._source_code

    @property
    def compiled_binary(self) -> bytes | None:
        return self._compiled_binary

    # --- BaseAsset interface ---

    def load(self, data: bytes) -> None:
        self._compiled_binary = data

    def unload(self) -> None:
        self._compiled_binary = None
        self._source_code = None

    def is_loaded(self) -> bool:
        return self._compiled_binary is not None

    # --- shader helpers ---

    def compile(self) -> bytes:
        """Compile source to binary (stub -- returns placeholder)."""
        if self._source_code is None:
            raise RuntimeError("No source code to compile")
        self._compiled_binary = self._source_code.encode("utf-8")
        return self._compiled_binary

    def get_uniforms(self) -> list[str]:
        """Extract uniform names from source code."""
        if self._source_code is None:
            return []
        return _UNIFORM_PATTERN.findall(self._source_code)

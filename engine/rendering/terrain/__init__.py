"""Terrain rendering passes for TRINITY engine.

This module provides terrain rendering components including:
- TerrainPass: Base terrain rendering pass
- ClipmapRenderer: Geometry clipmap terrain renderer
- TerrainMaterialBlend: Multi-material terrain blending

Stub created by T-ENV-1.12, expanded by:
- T-ENV-1.9: Clipmap implementation
- T-ENV-1.10: Material blending system
"""

from .terrain_pass import TerrainPass
from .clipmap import ClipmapRenderer
from .material_blend import TerrainMaterialBlend

__all__ = [
    "TerrainPass",
    "ClipmapRenderer",
    "TerrainMaterialBlend",
]

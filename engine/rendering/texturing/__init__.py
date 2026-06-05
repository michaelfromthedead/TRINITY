"""Texturing systems for TRINITY engine.

This module provides texturing components including:
- TextureStreaming: Virtual texture streaming system
- TextureAtlas: Dynamic texture atlas management

Stub created by T-ENV-1.12, expanded by:
- T-ENV-1.11: Virtual texturing implementation
"""

from .streaming import TextureStreaming
from .atlas import TextureAtlas

__all__ = [
    "TextureStreaming",
    "TextureAtlas",
]

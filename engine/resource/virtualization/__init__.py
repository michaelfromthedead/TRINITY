"""Virtualization subsystem: virtual texturing, geometry, and shadow maps."""

from engine.resource.constants import (
    LOD_DISTANCES,
    NUM_CLIPMAP_LEVELS,
    PAGE_SIZE,
    PHYSICAL_POOL_TILES,
    SHADOW_PAGE_SIZE,
)
from .virtual_texturing import (
    Page,
    PageTable,
    PhysicalTexturePool,
    VirtualTextureSystem,
)
from .virtual_geometry import (
    Cluster,
    ClusterGroup,
    VirtualGeometrySystem,
)
from .virtual_shadow_maps import (
    ShadowClipmapLevel,
    ShadowPage,
    VirtualShadowMapSystem,
)

__all__ = [
    "PAGE_SIZE",
    "PHYSICAL_POOL_TILES",
    "Page",
    "PageTable",
    "PhysicalTexturePool",
    "VirtualTextureSystem",
    "LOD_DISTANCES",
    "Cluster",
    "ClusterGroup",
    "VirtualGeometrySystem",
    "NUM_CLIPMAP_LEVELS",
    "SHADOW_PAGE_SIZE",
    "ShadowClipmapLevel",
    "ShadowPage",
    "VirtualShadowMapSystem",
]

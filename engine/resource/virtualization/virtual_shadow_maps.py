"""Virtual shadow map system with clipmap-based shadow pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from engine.resource.constants import (
    NUM_CLIPMAP_LEVELS,
    SHADOW_BASE_WORLD_SIZE,
    SHADOW_PAGE_SIZE,
    SHADOW_RESOLUTION_MULTIPLIER,
)


@dataclass(slots=True)
class ShadowPage:
    """A single shadow map page in a clipmap level."""

    page_x: int
    page_y: int
    clipmap_level: int
    is_cached: bool = False
    is_dirty: bool = True


@dataclass(slots=True)
class ShadowClipmapLevel:
    """One level of the shadow clipmap hierarchy."""

    level: int
    resolution: int
    world_size: float


ShadowPageKey = Tuple[int, int, int]


class VirtualShadowMapSystem:
    """Clipmap-based virtual shadow map system."""

    __slots__ = ("_levels", "_pages", "_light_dir", "_camera_pos")

    def __init__(self) -> None:
        self._levels: List[ShadowClipmapLevel] = []
        self._pages: Dict[ShadowPageKey, ShadowPage] = {}
        self._light_dir: Tuple[float, float, float] = (0.0, -1.0, 0.0)
        self._camera_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._init_levels()

    def _init_levels(self) -> None:
        base_resolution = SHADOW_PAGE_SIZE * SHADOW_RESOLUTION_MULTIPLIER
        base_world_size = SHADOW_BASE_WORLD_SIZE
        for i in range(NUM_CLIPMAP_LEVELS):
            self._levels.append(
                ShadowClipmapLevel(
                    level=i,
                    resolution=base_resolution,
                    world_size=base_world_size * (2 ** i),
                )
            )

    @property
    def levels(self) -> List[ShadowClipmapLevel]:
        return list(self._levels)

    def update_light(
        self,
        light_dir: Tuple[float, float, float],
        camera_pos: Tuple[float, float, float],
    ) -> None:
        """Update light direction and camera position, marking affected pages dirty."""
        dir_changed = light_dir != self._light_dir
        self._light_dir = light_dir
        self._camera_pos = camera_pos

        if dir_changed:
            for page in self._pages.values():
                page.is_dirty = True

        # Always ensure page coverage is up to date
        self._update_page_coverage()

    def _update_page_coverage(self) -> None:
        """Ensure pages exist around the camera for each clipmap level."""
        cx, cy, cz = self._camera_pos
        for lvl in self._levels:
            pages_per_side = max(1, lvl.resolution // SHADOW_PAGE_SIZE)
            texel_world = lvl.world_size / lvl.resolution
            page_world = texel_world * SHADOW_PAGE_SIZE

            center_px = int(cx / page_world) if page_world > 0 else 0
            center_py = int(cz / page_world) if page_world > 0 else 0

            half = pages_per_side // 2
            for px in range(center_px - half, center_px + half + 1):
                for py in range(center_py - half, center_py + half + 1):
                    key: ShadowPageKey = (px, py, lvl.level)
                    if key not in self._pages:
                        self._pages[key] = ShadowPage(
                            page_x=px,
                            page_y=py,
                            clipmap_level=lvl.level,
                            is_cached=False,
                            is_dirty=True,
                        )

    def get_dirty_pages(self) -> List[ShadowPage]:
        return [p for p in self._pages.values() if p.is_dirty]

    def mark_rendered(self, page_x: int, page_y: int, level: int) -> None:
        """Mark a shadow page as rendered (cached, not dirty)."""
        key: ShadowPageKey = (page_x, page_y, level)
        page = self._pages.get(key)
        if page is not None:
            page.is_dirty = False
            page.is_cached = True

    def invalidate_region(self, x: float, y: float, radius: float) -> None:
        """Invalidate all pages overlapping a world-space circle."""
        for lvl in self._levels:
            texel_world = lvl.world_size / lvl.resolution
            page_world = texel_world * SHADOW_PAGE_SIZE
            if page_world <= 0:
                continue

            min_px = int((x - radius) / page_world) - 1
            max_px = int((x + radius) / page_world) + 1
            min_py = int((y - radius) / page_world) - 1
            max_py = int((y + radius) / page_world) + 1

            for px in range(min_px, max_px + 1):
                for py in range(min_py, max_py + 1):
                    key: ShadowPageKey = (px, py, lvl.level)
                    page = self._pages.get(key)
                    if page is not None:
                        page.is_dirty = True
                        page.is_cached = False

    def get_cache_stats(self) -> dict:
        total = len(self._pages)
        cached = sum(1 for p in self._pages.values() if p.is_cached)
        dirty = sum(1 for p in self._pages.values() if p.is_dirty)
        return {
            "total_pages": total,
            "cached_pages": cached,
            "dirty_pages": dirty,
            "cache_hit_rate": cached / total if total > 0 else 0.0,
            "num_levels": len(self._levels),
        }

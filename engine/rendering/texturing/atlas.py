"""Dynamic texture atlas management stub (T-ENV-1.12).

This module provides runtime texture atlas management for efficient
batching of draw calls with multiple textures.

Features:
- Dynamic bin-packing allocation
- Multiple atlas pages
- UV remapping
- Automatic defragmentation

Expanded by T-ENV-1.11 with full atlas system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from trinity.decorators import component

if TYPE_CHECKING:
    from engine.platform.rhi.resources import Texture


class PackingAlgorithm(IntEnum):
    """Texture packing algorithms.

    Different strategies for arranging textures in atlas.
    """

    SHELF = 0         # Simple shelf-based packing
    MAXRECTS = 1      # Maximum rectangles bin packing
    GUILLOTINE = 2    # Guillotine cutting algorithm
    SKYLINE = 3       # Skyline bottom-left algorithm


class AtlasFormat(IntEnum):
    """Texture atlas pixel formats."""

    RGBA8 = 0         # 8-bit RGBA (32 bits/pixel)
    RGBA16F = 1       # 16-bit float RGBA (64 bits/pixel)
    RGB8 = 2          # 8-bit RGB (24 bits/pixel)
    RG8 = 3           # 8-bit RG (16 bits/pixel)


@dataclass
class AtlasRegion:
    """Allocated region within a texture atlas.

    Attributes:
        texture_id: Source texture identifier.
        page: Atlas page index.
        x: X position in atlas (pixels).
        y: Y position in atlas (pixels).
        width: Region width (pixels).
        height: Region height (pixels).
        uv_min: Normalized UV minimum (u, v).
        uv_max: Normalized UV maximum (u, v).
    """

    texture_id: str
    page: int = 0
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    uv_min: Tuple[float, float] = (0.0, 0.0)
    uv_max: Tuple[float, float] = (1.0, 1.0)


@dataclass
class AtlasPage:
    """Single page of a texture atlas.

    Attributes:
        index: Page index.
        texture: GPU texture for this page.
        width: Page width in pixels.
        height: Page height in pixels.
        regions: Allocated regions on this page.
        free_rects: Available free rectangles.
    """

    index: int
    texture: Optional["Texture"] = None
    width: int = 2048
    height: int = 2048
    regions: Dict[str, AtlasRegion] = field(default_factory=dict)
    free_rects: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize free space."""
        if not self.free_rects:
            self.free_rects = [(0, 0, self.width, self.height)]


@dataclass
class AtlasConfig:
    """Configuration for texture atlas system.

    Attributes:
        page_size: Size of atlas pages (power of 2).
        max_pages: Maximum number of atlas pages.
        format: Pixel format for atlas textures.
        algorithm: Packing algorithm to use.
        padding: Padding between textures (pixels).
        allow_rotation: Allow 90-degree rotation for better packing.
    """

    page_size: int = 2048
    max_pages: int = 8
    format: AtlasFormat = AtlasFormat.RGBA8
    algorithm: PackingAlgorithm = PackingAlgorithm.MAXRECTS
    padding: int = 2
    allow_rotation: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.page_size < 256 or (self.page_size & (self.page_size - 1)) != 0:
            raise ValueError(
                f"page_size must be power of 2 >= 256, got {self.page_size}"
            )
        if self.max_pages < 1:
            raise ValueError(f"max_pages must be >= 1, got {self.max_pages}")
        if self.padding < 0:
            raise ValueError(f"padding must be non-negative, got {self.padding}")


@dataclass
class AtlasStats:
    """Statistics for texture atlas usage.

    Attributes:
        page_count: Number of active pages.
        region_count: Total allocated regions.
        utilization: Overall space utilization (0-1).
        fragmentation: Estimated fragmentation (0-1).
        memory_used_mb: Total GPU memory in MB.
    """

    page_count: int = 0
    region_count: int = 0
    utilization: float = 0.0
    fragmentation: float = 0.0
    memory_used_mb: float = 0.0


@component
class TextureAtlas:
    """Dynamic texture atlas manager.

    Manages runtime allocation and packing of textures into atlas
    pages for efficient GPU batching.

    This is a stub class that will be expanded by T-ENV-1.11.

    Example:
        atlas = TextureAtlas(config=AtlasConfig())
        atlas.initialize()
        region = atlas.allocate("player_icon", 64, 64)
        uv = atlas.get_uv(region.texture_id)

    Attributes:
        config: Atlas configuration.
        stats: Usage statistics.
    """

    # Class-level attributes for Trinity component system
    _component_name: str = "TextureAtlas"

    def __init__(
        self,
        config: Optional[AtlasConfig] = None,
    ) -> None:
        """Initialize texture atlas manager.

        Args:
            config: Atlas configuration. Uses defaults if None.
        """
        self._config = config or AtlasConfig()
        self._pages: List[AtlasPage] = []
        self._regions: Dict[str, AtlasRegion] = {}
        self._initialized = False
        self._stats = AtlasStats()

    @property
    def config(self) -> AtlasConfig:
        """Get atlas configuration."""
        return self._config

    @property
    def stats(self) -> AtlasStats:
        """Get atlas statistics."""
        self._update_stats()
        return self._stats

    @property
    def is_initialized(self) -> bool:
        """Check if atlas has been initialized."""
        return self._initialized

    @property
    def page_count(self) -> int:
        """Get number of active atlas pages."""
        return len(self._pages)

    @property
    def region_count(self) -> int:
        """Get number of allocated regions."""
        return len(self._regions)

    def initialize(self) -> None:
        """Initialize atlas system.

        Creates first atlas page.

        Raises:
            RuntimeError: If already initialized.
        """
        if self._initialized:
            raise RuntimeError("TextureAtlas already initialized")

        # Create first page
        self._create_page()
        self._initialized = True

    def _create_page(self) -> Optional[AtlasPage]:
        """Create a new atlas page.

        Returns:
            New page, or None if at max pages.
        """
        if len(self._pages) >= self._config.max_pages:
            return None

        page = AtlasPage(
            index=len(self._pages),
            width=self._config.page_size,
            height=self._config.page_size,
        )
        self._pages.append(page)
        return page

    def allocate(
        self,
        texture_id: str,
        width: int,
        height: int,
    ) -> Optional[AtlasRegion]:
        """Allocate space for a texture in the atlas.

        Args:
            texture_id: Unique identifier for the texture.
            width: Texture width in pixels.
            height: Texture height in pixels.

        Returns:
            Allocated region, or None if allocation failed.

        Raises:
            RuntimeError: If not initialized.
            ValueError: If texture_id already exists.
        """
        if not self._initialized:
            raise RuntimeError("TextureAtlas not initialized")

        if texture_id in self._regions:
            raise ValueError(f"Texture '{texture_id}' already allocated")

        # Add padding
        padded_width = width + self._config.padding * 2
        padded_height = height + self._config.padding * 2

        # Check size
        if padded_width > self._config.page_size or padded_height > self._config.page_size:
            return None  # Too large for atlas

        # Try to allocate on existing pages
        for page in self._pages:
            region = self._try_allocate_on_page(
                page, texture_id, width, height, padded_width, padded_height
            )
            if region is not None:
                return region

        # Create new page if possible
        new_page = self._create_page()
        if new_page is not None:
            return self._try_allocate_on_page(
                new_page, texture_id, width, height, padded_width, padded_height
            )

        return None

    def _try_allocate_on_page(
        self,
        page: AtlasPage,
        texture_id: str,
        width: int,
        height: int,
        padded_width: int,
        padded_height: int,
    ) -> Optional[AtlasRegion]:
        """Try to allocate a region on a specific page.

        Args:
            page: Page to allocate on.
            texture_id: Texture identifier.
            width: Actual texture width.
            height: Actual texture height.
            padded_width: Width including padding.
            padded_height: Height including padding.

        Returns:
            Allocated region, or None if no space.
        """
        # Find best-fit free rectangle
        best_idx = -1
        best_area = float('inf')
        best_rect = (0, 0, 0, 0)

        for i, (rx, ry, rw, rh) in enumerate(page.free_rects):
            if rw >= padded_width and rh >= padded_height:
                area = rw * rh
                if area < best_area:
                    best_idx = i
                    best_area = area
                    best_rect = (rx, ry, rw, rh)

        if best_idx < 0:
            return None  # No suitable space

        # Remove used rectangle
        page.free_rects.pop(best_idx)
        rx, ry, rw, rh = best_rect

        # Create region (offset by padding)
        region = AtlasRegion(
            texture_id=texture_id,
            page=page.index,
            x=rx + self._config.padding,
            y=ry + self._config.padding,
            width=width,
            height=height,
            uv_min=(
                (rx + self._config.padding) / page.width,
                (ry + self._config.padding) / page.height,
            ),
            uv_max=(
                (rx + self._config.padding + width) / page.width,
                (ry + self._config.padding + height) / page.height,
            ),
        )

        # Split remaining space (maxrects-style)
        if rw > padded_width:
            # Right remainder
            page.free_rects.append((
                rx + padded_width, ry,
                rw - padded_width, rh
            ))
        if rh > padded_height:
            # Bottom remainder
            page.free_rects.append((
                rx, ry + padded_height,
                padded_width, rh - padded_height
            ))

        page.regions[texture_id] = region
        self._regions[texture_id] = region
        return region

    def deallocate(self, texture_id: str) -> bool:
        """Free a texture region.

        Args:
            texture_id: Identifier of texture to free.

        Returns:
            True if deallocated, False if not found.
        """
        if texture_id not in self._regions:
            return False

        region = self._regions.pop(texture_id)
        page = self._pages[region.page]
        page.regions.pop(texture_id, None)

        # Add back to free rects (simplified - no merging in stub)
        padded_x = region.x - self._config.padding
        padded_y = region.y - self._config.padding
        padded_w = region.width + self._config.padding * 2
        padded_h = region.height + self._config.padding * 2
        page.free_rects.append((padded_x, padded_y, padded_w, padded_h))

        return True

    def get_region(self, texture_id: str) -> Optional[AtlasRegion]:
        """Get allocated region by texture ID.

        Args:
            texture_id: Texture identifier.

        Returns:
            Region if found, None otherwise.
        """
        return self._regions.get(texture_id)

    def get_uv(self, texture_id: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get UV coordinates for a texture.

        Args:
            texture_id: Texture identifier.

        Returns:
            (uv_min, uv_max) tuple, or None if not found.
        """
        region = self._regions.get(texture_id)
        if region is not None:
            return (region.uv_min, region.uv_max)
        return None

    def get_page_texture(self, page_index: int) -> Optional["Texture"]:
        """Get GPU texture for an atlas page.

        Args:
            page_index: Page index.

        Returns:
            Texture if valid, None otherwise.
        """
        if 0 <= page_index < len(self._pages):
            return self._pages[page_index].texture
        return None

    def defragment(self) -> int:
        """Defragment atlas pages.

        Repacks textures to reduce fragmentation.

        Returns:
            Number of textures repositioned.
        """
        # Stub: No-op, full implementation would repack
        return 0

    def _update_stats(self) -> None:
        """Update usage statistics."""
        total_area = 0
        used_area = 0

        for page in self._pages:
            page_area = page.width * page.height
            total_area += page_area

            for region in page.regions.values():
                used_area += (
                    (region.width + self._config.padding * 2) *
                    (region.height + self._config.padding * 2)
                )

        self._stats.page_count = len(self._pages)
        self._stats.region_count = len(self._regions)
        self._stats.utilization = used_area / total_area if total_area > 0 else 0.0

        # Estimate fragmentation from free rect count
        total_free_rects = sum(len(p.free_rects) for p in self._pages)
        self._stats.fragmentation = min(1.0, total_free_rects / max(1, len(self._regions) + 1) - 1.0)
        self._stats.fragmentation = max(0.0, self._stats.fragmentation)

        # Memory calculation (RGBA8 = 4 bytes/pixel)
        bytes_per_page = self._config.page_size ** 2 * 4
        self._stats.memory_used_mb = (bytes_per_page * len(self._pages)) / (1024 * 1024)

    def clear(self) -> None:
        """Clear all allocated regions."""
        self._regions.clear()
        for page in self._pages:
            page.regions.clear()
            page.free_rects = [(0, 0, page.width, page.height)]

    def destroy(self) -> None:
        """Release atlas resources."""
        self._regions.clear()
        self._pages.clear()
        self._initialized = False

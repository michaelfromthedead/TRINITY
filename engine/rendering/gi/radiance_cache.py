"""Radiance Cache System for TRINITY (T-GIR-P2.7).

This module implements a 3D grid-based radiance cache with temporal
accumulation from DDGI probe data. The cache provides fast irradiance
lookups for shading.

Features:
    - 64x64x32 3D grid texture for radiance storage
    - Temporal accumulation from probe data
    - Shader-based update via radiance_cache_update.comp.wgsl
    - Integration with DDGI probe system
    - Configurable grid resolution and world bounds

Architecture:
    - CacheGrid: 3D grid representation with voxel data
    - RadianceCache: Main cache with temporal accumulation
    - RadianceCacheUpdater: Shader dispatch and probe integration
    - CacheCell: Per-voxel radiance storage with SH coefficients

Performance:
    - Uses compact SH L1 storage (4 coefficients per channel)
    - Temporal blending reduces noise over frames
    - Hierarchical mip levels for fast sampling (optional)

References:
    - GDC 2019, "Real-Time Global Illumination Using Precomputed Light Field Probes"
    - GPU Pro 7, "Practical Real-Time Voxel-Based Global Illumination"
    - Lumen technical docs on radiance caching
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Iterator, Optional, Tuple, List

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec3


# ============================================================================
# Constants
# ============================================================================

# Default grid dimensions
DEFAULT_GRID_WIDTH = 64
DEFAULT_GRID_HEIGHT = 64
DEFAULT_GRID_DEPTH = 32

# Default temporal parameters
DEFAULT_BLEND_FACTOR = 0.05  # Blend 5% new data per frame
DEFAULT_HYSTERESIS_FRAMES = 4  # Frames before cell becomes stable
DEFAULT_VARIANCE_THRESHOLD = 0.001  # Threshold for stability detection

# SH L1 has 4 coefficients per channel (1 DC + 3 directional)
SH_L1_COEFFICIENTS = 4

# Memory alignment for GPU uploads (bytes)
GPU_ALIGNMENT = 16


# ============================================================================
# Enums
# ============================================================================


class CacheUpdateMode(Enum):
    """Update mode for radiance cache."""

    FULL = auto()  # Update all cells each frame
    PARTIAL = auto()  # Update subset of cells (lower cost)
    ADAPTIVE = auto()  # Update based on variance/motion


class CacheQuality(Enum):
    """Quality preset for radiance cache."""

    LOW = auto()  # 32x32x16 grid
    MEDIUM = auto()  # 64x64x32 grid (default)
    HIGH = auto()  # 128x128x64 grid
    ULTRA = auto()  # 256x256x128 grid


class CellState(Enum):
    """State of a cache cell."""

    EMPTY = auto()  # No data accumulated
    ACCUMULATING = auto()  # Receiving data, not stable
    STABLE = auto()  # Stable after hysteresis period
    STALE = auto()  # Data outdated, needs refresh


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class RadianceCacheConfig:
    """Configuration for radiance cache.

    Attributes:
        width: Grid width in cells
        height: Grid height in cells
        depth: Grid depth in cells
        blend_factor: Temporal blend factor [0, 1]
        hysteresis_frames: Frames before cell becomes stable
        variance_threshold: Variance threshold for stability
        update_mode: How to update cells each frame
        enable_mip_chain: Whether to generate mip levels
        max_mip_levels: Maximum mip levels (0 = auto)
    """

    width: int = DEFAULT_GRID_WIDTH
    height: int = DEFAULT_GRID_HEIGHT
    depth: int = DEFAULT_GRID_DEPTH
    blend_factor: float = DEFAULT_BLEND_FACTOR
    hysteresis_frames: int = DEFAULT_HYSTERESIS_FRAMES
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD
    update_mode: CacheUpdateMode = CacheUpdateMode.FULL
    enable_mip_chain: bool = False
    max_mip_levels: int = 0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.width < 1:
            raise ValueError(f"width must be >= 1, got {self.width}")
        if self.height < 1:
            raise ValueError(f"height must be >= 1, got {self.height}")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        if not 0.0 <= self.blend_factor <= 1.0:
            raise ValueError(f"blend_factor must be in [0, 1], got {self.blend_factor}")
        if self.hysteresis_frames < 0:
            raise ValueError(f"hysteresis_frames must be >= 0, got {self.hysteresis_frames}")
        if self.variance_threshold < 0.0:
            raise ValueError(f"variance_threshold must be >= 0, got {self.variance_threshold}")

    @property
    def total_cells(self) -> int:
        """Total number of cells in the grid."""
        return self.width * self.height * self.depth

    @property
    def cell_size_bytes(self) -> int:
        """Size of a single cell in bytes (SH L1 RGB + metadata)."""
        # 4 SH coefficients * 3 channels * 4 bytes + 16 bytes metadata
        return SH_L1_COEFFICIENTS * 3 * 4 + 16

    @property
    def total_memory_bytes(self) -> int:
        """Total memory required for the cache in bytes."""
        return self.total_cells * self.cell_size_bytes

    @staticmethod
    def from_quality(quality: CacheQuality) -> RadianceCacheConfig:
        """Create config from quality preset.

        Args:
            quality: Quality preset

        Returns:
            Configuration for the quality level
        """
        presets = {
            CacheQuality.LOW: RadianceCacheConfig(
                width=32, height=32, depth=16,
                blend_factor=0.1, hysteresis_frames=2,
            ),
            CacheQuality.MEDIUM: RadianceCacheConfig(
                width=64, height=64, depth=32,
                blend_factor=0.05, hysteresis_frames=4,
            ),
            CacheQuality.HIGH: RadianceCacheConfig(
                width=128, height=128, depth=64,
                blend_factor=0.03, hysteresis_frames=6,
                enable_mip_chain=True,
            ),
            CacheQuality.ULTRA: RadianceCacheConfig(
                width=256, height=256, depth=128,
                blend_factor=0.02, hysteresis_frames=8,
                enable_mip_chain=True, max_mip_levels=4,
            ),
        }
        return presets.get(quality, presets[CacheQuality.MEDIUM])


# ============================================================================
# Cache Cell
# ============================================================================


@dataclass
class CacheCell:
    """A single cell in the radiance cache.

    Each cell stores SH L1 coefficients (4 per channel) for RGB radiance,
    plus metadata for temporal accumulation.

    Attributes:
        sh_r: SH L1 coefficients for red channel [DC, Y1-1, Y10, Y11]
        sh_g: SH L1 coefficients for green channel
        sh_b: SH L1 coefficients for blue channel
        state: Current cell state
        frames_accumulated: Number of frames contributing to this cell
        variance: Running variance estimate
        last_update_frame: Frame index of last update
    """

    sh_r: List[float] = field(default_factory=lambda: [0.0] * SH_L1_COEFFICIENTS)
    sh_g: List[float] = field(default_factory=lambda: [0.0] * SH_L1_COEFFICIENTS)
    sh_b: List[float] = field(default_factory=lambda: [0.0] * SH_L1_COEFFICIENTS)
    state: CellState = CellState.EMPTY
    frames_accumulated: int = 0
    variance: float = 0.0
    last_update_frame: int = 0

    def reset(self) -> None:
        """Reset cell to empty state."""
        for i in range(SH_L1_COEFFICIENTS):
            self.sh_r[i] = 0.0
            self.sh_g[i] = 0.0
            self.sh_b[i] = 0.0
        self.state = CellState.EMPTY
        self.frames_accumulated = 0
        self.variance = 0.0
        self.last_update_frame = 0

    def get_dc_radiance(self) -> Vec3:
        """Get the DC (average) radiance from SH coefficients.

        Returns:
            Average radiance as Vec3(r, g, b)
        """
        # DC coefficient is the first one, scaled by SH normalization
        # SH Y00 = 1/(2*sqrt(pi)) ≈ 0.2821
        # Irradiance from SH: E = pi * c00
        scale = math.pi
        return Vec3(
            self.sh_r[0] * scale,
            self.sh_g[0] * scale,
            self.sh_b[0] * scale,
        )

    def evaluate_sh(self, direction: Vec3) -> Vec3:
        """Evaluate SH in a given direction.

        Args:
            direction: Normalized direction vector

        Returns:
            Radiance in that direction
        """
        # SH L1 basis functions
        # Y00 = 0.2821 (constant)
        # Y1-1 = 0.4886 * y
        # Y10 = 0.4886 * z
        # Y11 = 0.4886 * x
        c0 = 0.2821  # 1 / (2 * sqrt(pi))
        c1 = 0.4886  # sqrt(3) / (2 * sqrt(pi))

        basis = [
            c0,
            c1 * direction.y,
            c1 * direction.z,
            c1 * direction.x,
        ]

        r = sum(self.sh_r[i] * basis[i] for i in range(SH_L1_COEFFICIENTS))
        g = sum(self.sh_g[i] * basis[i] for i in range(SH_L1_COEFFICIENTS))
        b = sum(self.sh_b[i] * basis[i] for i in range(SH_L1_COEFFICIENTS))

        # Clamp to positive (SH can produce negative values)
        return Vec3(max(0.0, r), max(0.0, g), max(0.0, b))

    def accumulate(
        self,
        new_sh_r: List[float],
        new_sh_g: List[float],
        new_sh_b: List[float],
        blend_factor: float,
        frame_index: int,
    ) -> None:
        """Accumulate new SH data with temporal blending.

        Args:
            new_sh_r: New red channel SH coefficients
            new_sh_g: New green channel SH coefficients
            new_sh_b: New blue channel SH coefficients
            blend_factor: Blend factor [0, 1]
            frame_index: Current frame index
        """
        if self.state == CellState.EMPTY:
            # First sample - copy directly
            self.sh_r = new_sh_r.copy()
            self.sh_g = new_sh_g.copy()
            self.sh_b = new_sh_b.copy()
            self.state = CellState.ACCUMULATING
            self.frames_accumulated = 1
            self.variance = 1.0  # High initial variance
        else:
            # Compute variance from difference
            diff_sq = 0.0
            for i in range(SH_L1_COEFFICIENTS):
                diff_sq += (self.sh_r[i] - new_sh_r[i]) ** 2
                diff_sq += (self.sh_g[i] - new_sh_g[i]) ** 2
                diff_sq += (self.sh_b[i] - new_sh_b[i]) ** 2

            # Running variance estimate (exponential moving average)
            self.variance = self.variance * (1.0 - blend_factor) + diff_sq * blend_factor

            # Blend SH coefficients
            keep = 1.0 - blend_factor
            for i in range(SH_L1_COEFFICIENTS):
                self.sh_r[i] = self.sh_r[i] * keep + new_sh_r[i] * blend_factor
                self.sh_g[i] = self.sh_g[i] * keep + new_sh_g[i] * blend_factor
                self.sh_b[i] = self.sh_b[i] * keep + new_sh_b[i] * blend_factor

            self.frames_accumulated += 1

        self.last_update_frame = frame_index

    def update_state(self, hysteresis_frames: int, variance_threshold: float) -> None:
        """Update cell state based on accumulation progress.

        Args:
            hysteresis_frames: Frames required for stability
            variance_threshold: Variance threshold for stability
        """
        if self.state == CellState.EMPTY:
            return

        if self.frames_accumulated >= hysteresis_frames and self.variance < variance_threshold:
            self.state = CellState.STABLE
        elif self.state == CellState.STABLE and self.variance >= variance_threshold:
            self.state = CellState.ACCUMULATING

    def to_bytes(self) -> bytes:
        """Pack cell data for GPU upload (64 bytes).

        Layout:
            - sh_r: 4 x f32 (16 bytes)
            - sh_g: 4 x f32 (16 bytes)
            - sh_b: 4 x f32 (16 bytes)
            - metadata: state (u32), frames (u32), variance (f32), _pad (f32) (16 bytes)

        Returns:
            Packed bytes for GPU upload
        """
        return struct.pack(
            "<4f4f4fIIff",
            # sh_r
            self.sh_r[0], self.sh_r[1], self.sh_r[2], self.sh_r[3],
            # sh_g
            self.sh_g[0], self.sh_g[1], self.sh_g[2], self.sh_g[3],
            # sh_b
            self.sh_b[0], self.sh_b[1], self.sh_b[2], self.sh_b[3],
            # metadata
            self.state.value,
            self.frames_accumulated,
            self.variance,
            0.0,  # padding
        )

    @staticmethod
    def from_bytes(data: bytes) -> CacheCell:
        """Unpack cell data from GPU download.

        Args:
            data: 64 bytes of packed cell data

        Returns:
            Unpacked CacheCell
        """
        values = struct.unpack("<4f4f4fIIff", data)
        cell = CacheCell()
        cell.sh_r = list(values[0:4])
        cell.sh_g = list(values[4:8])
        cell.sh_b = list(values[8:12])
        cell.state = CellState(values[12])
        cell.frames_accumulated = values[13]
        cell.variance = values[14]
        return cell


# ============================================================================
# Cache Grid
# ============================================================================


@dataclass
class CacheGrid:
    """3D grid structure for radiance cache.

    The grid maps a world-space AABB to a regular 3D grid of cells.
    Each cell stores SH L1 coefficients for radiance lookup.

    Attributes:
        bounds: World-space bounds covered by the grid
        config: Cache configuration
        cells: Flattened array of cache cells
        frame_index: Current frame for temporal tracking
    """

    bounds: AABB
    config: RadianceCacheConfig
    cells: List[CacheCell] = field(default_factory=list)
    frame_index: int = 0

    def __post_init__(self) -> None:
        """Initialize cell array."""
        if not self.cells:
            self.cells = [CacheCell() for _ in range(self.config.total_cells)]

    @property
    def cell_size(self) -> Vec3:
        """World-space size of a single cell."""
        extent = self.bounds.max - self.bounds.min
        return Vec3(
            extent.x / self.config.width,
            extent.y / self.config.height,
            extent.z / self.config.depth,
        )

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Grid dimensions (width, height, depth)."""
        return (self.config.width, self.config.height, self.config.depth)

    def cell_index(self, ix: int, iy: int, iz: int) -> int:
        """Convert 3D cell coordinates to flat index.

        Args:
            ix: X cell index [0, width)
            iy: Y cell index [0, height)
            iz: Z cell index [0, depth)

        Returns:
            Flat index into cells array

        Raises:
            IndexError: If coordinates out of bounds
        """
        if not (0 <= ix < self.config.width):
            raise IndexError(f"ix={ix} out of range [0, {self.config.width})")
        if not (0 <= iy < self.config.height):
            raise IndexError(f"iy={iy} out of range [0, {self.config.height})")
        if not (0 <= iz < self.config.depth):
            raise IndexError(f"iz={iz} out of range [0, {self.config.depth})")

        # Z-major ordering for cache coherence
        return iz * (self.config.width * self.config.height) + iy * self.config.width + ix

    def cell_coords(self, flat_index: int) -> Tuple[int, int, int]:
        """Convert flat index to 3D cell coordinates.

        Args:
            flat_index: Index into cells array

        Returns:
            Tuple of (ix, iy, iz)

        Raises:
            IndexError: If index out of range
        """
        if not 0 <= flat_index < self.config.total_cells:
            raise IndexError(f"flat_index={flat_index} out of range [0, {self.config.total_cells})")

        wh = self.config.width * self.config.height
        iz = flat_index // wh
        remainder = flat_index % wh
        iy = remainder // self.config.width
        ix = remainder % self.config.width
        return (ix, iy, iz)

    def world_to_cell(self, world_pos: Vec3) -> Tuple[int, int, int]:
        """Convert world position to cell coordinates.

        Args:
            world_pos: World-space position

        Returns:
            Cell coordinates (ix, iy, iz), clamped to grid bounds
        """
        local = world_pos - self.bounds.min
        cell_size = self.cell_size

        ix = int(local.x / cell_size.x) if cell_size.x > 0 else 0
        iy = int(local.y / cell_size.y) if cell_size.y > 0 else 0
        iz = int(local.z / cell_size.z) if cell_size.z > 0 else 0

        # Clamp to valid range
        ix = max(0, min(ix, self.config.width - 1))
        iy = max(0, min(iy, self.config.height - 1))
        iz = max(0, min(iz, self.config.depth - 1))

        return (ix, iy, iz)

    def cell_to_world(self, ix: int, iy: int, iz: int) -> Vec3:
        """Convert cell coordinates to world-space center.

        Args:
            ix: X cell index
            iy: Y cell index
            iz: Z cell index

        Returns:
            World-space position at cell center
        """
        cell_size = self.cell_size
        return Vec3(
            self.bounds.min.x + (ix + 0.5) * cell_size.x,
            self.bounds.min.y + (iy + 0.5) * cell_size.y,
            self.bounds.min.z + (iz + 0.5) * cell_size.z,
        )

    def get_cell(self, ix: int, iy: int, iz: int) -> CacheCell:
        """Get cell at grid coordinates.

        Args:
            ix: X cell index
            iy: Y cell index
            iz: Z cell index

        Returns:
            CacheCell at coordinates
        """
        return self.cells[self.cell_index(ix, iy, iz)]

    def set_cell(self, ix: int, iy: int, iz: int, cell: CacheCell) -> None:
        """Set cell at grid coordinates.

        Args:
            ix: X cell index
            iy: Y cell index
            iz: Z cell index
            cell: Cell data to set
        """
        self.cells[self.cell_index(ix, iy, iz)] = cell

    def sample(self, world_pos: Vec3, normal: Vec3) -> Vec3:
        """Sample radiance at a world position with trilinear interpolation.

        Args:
            world_pos: World-space position
            normal: Surface normal for SH evaluation

        Returns:
            Interpolated radiance color
        """
        if not self.bounds.contains(world_pos):
            return Vec3.zero()

        # Get fractional cell coordinates
        local = world_pos - self.bounds.min
        cell_size = self.cell_size

        fx = local.x / cell_size.x if cell_size.x > 0 else 0.0
        fy = local.y / cell_size.y if cell_size.y > 0 else 0.0
        fz = local.z / cell_size.z if cell_size.z > 0 else 0.0

        # Base cell indices
        ix0 = int(math.floor(fx - 0.5))
        iy0 = int(math.floor(fy - 0.5))
        iz0 = int(math.floor(fz - 0.5))

        # Fractional position within cell
        tx = fx - 0.5 - ix0
        ty = fy - 0.5 - iy0
        tz = fz - 0.5 - iz0

        # Trilinear interpolation over 8 corners
        total = Vec3.zero()
        total_weight = 0.0

        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    cix = max(0, min(ix0 + dx, self.config.width - 1))
                    ciy = max(0, min(iy0 + dy, self.config.height - 1))
                    ciz = max(0, min(iz0 + dz, self.config.depth - 1))

                    cell = self.get_cell(cix, ciy, ciz)

                    # Only sample stable or accumulating cells
                    if cell.state == CellState.EMPTY:
                        continue

                    # Trilinear weight
                    wx = tx if dx else (1.0 - tx)
                    wy = ty if dy else (1.0 - ty)
                    wz = tz if dz else (1.0 - tz)
                    weight = wx * wy * wz

                    # Evaluate SH in normal direction
                    radiance = cell.evaluate_sh(normal)
                    total = total + radiance * weight
                    total_weight += weight

        if total_weight > 0.0:
            return total * (1.0 / total_weight)
        return Vec3.zero()

    def sample_dc(self, world_pos: Vec3) -> Vec3:
        """Sample DC (average) radiance at a world position.

        Faster than full SH evaluation when direction doesn't matter.

        Args:
            world_pos: World-space position

        Returns:
            Interpolated DC radiance
        """
        if not self.bounds.contains(world_pos):
            return Vec3.zero()

        ix, iy, iz = self.world_to_cell(world_pos)
        cell = self.get_cell(ix, iy, iz)

        if cell.state == CellState.EMPTY:
            return Vec3.zero()

        return cell.get_dc_radiance()

    def clear(self) -> None:
        """Clear all cells to empty state."""
        for cell in self.cells:
            cell.reset()
        self.frame_index = 0

    def iter_cells(self) -> Iterator[Tuple[int, int, int, CacheCell]]:
        """Iterate over all cells with their coordinates.

        Yields:
            Tuples of (ix, iy, iz, cell)
        """
        for iz in range(self.config.depth):
            for iy in range(self.config.height):
                for ix in range(self.config.width):
                    yield (ix, iy, iz, self.get_cell(ix, iy, iz))

    def get_statistics(self) -> dict:
        """Get grid statistics.

        Returns:
            Dictionary with statistics
        """
        state_counts = {s: 0 for s in CellState}
        total_variance = 0.0
        max_variance = 0.0

        for cell in self.cells:
            state_counts[cell.state] += 1
            total_variance += cell.variance
            max_variance = max(max_variance, cell.variance)

        total_cells = len(self.cells)
        return {
            "total_cells": total_cells,
            "dimensions": self.dimensions,
            "cell_size": (self.cell_size.x, self.cell_size.y, self.cell_size.z),
            "state_counts": {s.name: count for s, count in state_counts.items()},
            "average_variance": total_variance / total_cells if total_cells > 0 else 0.0,
            "max_variance": max_variance,
            "frame_index": self.frame_index,
            "memory_bytes": self.config.total_memory_bytes,
        }

    def to_bytes(self) -> bytes:
        """Pack entire grid for GPU upload.

        Returns:
            Packed bytes for all cells
        """
        return b"".join(cell.to_bytes() for cell in self.cells)


# ============================================================================
# Radiance Cache
# ============================================================================


@dataclass
class RadianceCache:
    """Main radiance cache with temporal accumulation.

    The cache stores radiance data from probe systems and provides
    fast lookups for indirect lighting in shading.

    Attributes:
        grid: The 3D cache grid
        config: Cache configuration
        is_valid: Whether cache has valid data
        needs_full_update: Whether a full refresh is needed
    """

    grid: CacheGrid
    config: RadianceCacheConfig = field(default_factory=RadianceCacheConfig)
    is_valid: bool = False
    needs_full_update: bool = True

    @classmethod
    def create(
        cls,
        bounds: AABB,
        config: Optional[RadianceCacheConfig] = None,
    ) -> RadianceCache:
        """Create a new radiance cache.

        Args:
            bounds: World-space bounds for the cache
            config: Optional configuration (uses defaults if None)

        Returns:
            New RadianceCache instance
        """
        if config is None:
            config = RadianceCacheConfig()

        grid = CacheGrid(bounds=bounds, config=config)
        return cls(grid=grid, config=config)

    @classmethod
    def create_with_quality(
        cls,
        bounds: AABB,
        quality: CacheQuality,
    ) -> RadianceCache:
        """Create a cache with quality preset.

        Args:
            bounds: World-space bounds
            quality: Quality preset

        Returns:
            New RadianceCache instance
        """
        config = RadianceCacheConfig.from_quality(quality)
        return cls.create(bounds, config)

    def update_from_probes(
        self,
        probe_sampler: Callable[[Vec3, Vec3], Tuple[List[float], List[float], List[float]]],
    ) -> None:
        """Update cache from probe system.

        Args:
            probe_sampler: Function that returns (sh_r, sh_g, sh_b) for a position/direction
        """
        self.grid.frame_index += 1

        for ix, iy, iz, cell in self.grid.iter_cells():
            world_pos = self.grid.cell_to_world(ix, iy, iz)

            # Sample probes for this cell (direction up for hemisphere average)
            sh_r, sh_g, sh_b = probe_sampler(world_pos, Vec3(0.0, 1.0, 0.0))

            cell.accumulate(
                sh_r, sh_g, sh_b,
                self.config.blend_factor,
                self.grid.frame_index,
            )
            cell.update_state(
                self.config.hysteresis_frames,
                self.config.variance_threshold,
            )

        self.is_valid = True
        self.needs_full_update = False

    def update_cell(
        self,
        ix: int, iy: int, iz: int,
        sh_r: List[float],
        sh_g: List[float],
        sh_b: List[float],
    ) -> None:
        """Update a single cell with new SH data.

        Args:
            ix: X cell index
            iy: Y cell index
            iz: Z cell index
            sh_r: Red SH coefficients
            sh_g: Green SH coefficients
            sh_b: Blue SH coefficients
        """
        cell = self.grid.get_cell(ix, iy, iz)
        cell.accumulate(
            sh_r, sh_g, sh_b,
            self.config.blend_factor,
            self.grid.frame_index,
        )
        cell.update_state(
            self.config.hysteresis_frames,
            self.config.variance_threshold,
        )

    def sample(self, world_pos: Vec3, normal: Vec3) -> Vec3:
        """Sample radiance for shading.

        Args:
            world_pos: World-space position
            normal: Surface normal

        Returns:
            Radiance at position
        """
        if not self.is_valid:
            return Vec3.zero()
        return self.grid.sample(world_pos, normal)

    def sample_dc(self, world_pos: Vec3) -> Vec3:
        """Sample DC radiance (average, no direction).

        Args:
            world_pos: World-space position

        Returns:
            DC radiance at position
        """
        if not self.is_valid:
            return Vec3.zero()
        return self.grid.sample_dc(world_pos)

    def invalidate(self) -> None:
        """Invalidate the cache, requiring full update."""
        self.is_valid = False
        self.needs_full_update = True

    def clear(self) -> None:
        """Clear all cache data."""
        self.grid.clear()
        self.is_valid = False
        self.needs_full_update = True

    def get_bounds(self) -> AABB:
        """Get cache bounds."""
        return self.grid.bounds

    def get_statistics(self) -> dict:
        """Get cache statistics."""
        stats = self.grid.get_statistics()
        stats["is_valid"] = self.is_valid
        stats["needs_full_update"] = self.needs_full_update
        return stats


# ============================================================================
# Radiance Cache Updater
# ============================================================================


@dataclass
class RadianceCacheUpdater:
    """Handles shader dispatch for radiance cache updates.

    This class manages the compute shader dispatches for updating
    the radiance cache from DDGI probe data.

    Attributes:
        cache: The radiance cache to update
        workgroup_size: Compute workgroup size
        cells_per_frame: Cells to update per frame (for partial updates)
        current_offset: Current offset for partial updates
    """

    cache: RadianceCache
    workgroup_size: Tuple[int, int, int] = (8, 8, 4)
    cells_per_frame: int = 0  # 0 = all cells
    current_offset: int = 0

    def get_dispatch_size(self) -> Tuple[int, int, int]:
        """Calculate compute dispatch dimensions.

        Returns:
            Tuple of (dispatch_x, dispatch_y, dispatch_z)
        """
        config = self.cache.config
        wx, wy, wz = self.workgroup_size

        dispatch_x = (config.width + wx - 1) // wx
        dispatch_y = (config.height + wy - 1) // wy
        dispatch_z = (config.depth + wz - 1) // wz

        return (dispatch_x, dispatch_y, dispatch_z)

    def get_shader_uniforms(self) -> bytes:
        """Generate uniform buffer for the update shader.

        Returns:
            Packed uniform data (64 bytes)
        """
        config = self.cache.config
        bounds = self.cache.grid.bounds
        cell_size = self.cache.grid.cell_size

        return struct.pack(
            "<3fI3fI3fIfII",
            # grid_origin (vec3<f32>)
            bounds.min.x, bounds.min.y, bounds.min.z,
            # grid_width (u32)
            config.width,
            # grid_extent (vec3<f32>)
            bounds.max.x - bounds.min.x,
            bounds.max.y - bounds.min.y,
            bounds.max.z - bounds.min.z,
            # grid_height (u32)
            config.height,
            # cell_size (vec3<f32>)
            cell_size.x, cell_size.y, cell_size.z,
            # grid_depth (u32)
            config.depth,
            # blend_factor (f32)
            config.blend_factor,
            # frame_index (u32)
            self.cache.grid.frame_index,
            # _pad (u32)
            0,
        )

    def step(self) -> int:
        """Advance partial update offset.

        Returns:
            Number of cells updated this frame
        """
        if self.cells_per_frame <= 0:
            return self.cache.config.total_cells

        cells_updated = min(
            self.cells_per_frame,
            self.cache.config.total_cells - self.current_offset,
        )

        self.current_offset += cells_updated

        if self.current_offset >= self.cache.config.total_cells:
            self.current_offset = 0

        return cells_updated

    def reset(self) -> None:
        """Reset partial update state."""
        self.current_offset = 0

    def generate_wgsl_shader(self) -> str:
        """Generate WGSL compute shader for cache update.

        Returns:
            WGSL shader code
        """
        wx, wy, wz = self.workgroup_size
        return f"""
// Radiance Cache Update Compute Shader (T-GIR-P2.7)
// Auto-generated by RadianceCacheUpdater

struct CacheUniforms {{
    grid_origin: vec3<f32>,
    grid_width: u32,
    grid_extent: vec3<f32>,
    grid_height: u32,
    cell_size: vec3<f32>,
    grid_depth: u32,
    blend_factor: f32,
    frame_index: u32,
    _pad: u32,
}}

struct CacheCell {{
    sh_r: vec4<f32>,
    sh_g: vec4<f32>,
    sh_b: vec4<f32>,
    state: u32,
    frames_accumulated: u32,
    variance: f32,
    _pad: f32,
}}

@group(0) @binding(0) var<uniform> uniforms: CacheUniforms;
@group(0) @binding(1) var<storage, read_write> cache_cells: array<CacheCell>;
@group(0) @binding(2) var probe_texture: texture_3d<f32>;
@group(0) @binding(3) var probe_sampler: sampler;

fn cell_index(ix: u32, iy: u32, iz: u32) -> u32 {{
    return iz * uniforms.grid_width * uniforms.grid_height +
           iy * uniforms.grid_width + ix;
}}

fn cell_to_world(ix: u32, iy: u32, iz: u32) -> vec3<f32> {{
    return uniforms.grid_origin + vec3<f32>(
        (f32(ix) + 0.5) * uniforms.cell_size.x,
        (f32(iy) + 0.5) * uniforms.cell_size.y,
        (f32(iz) + 0.5) * uniforms.cell_size.z,
    );
}}

@compute @workgroup_size({wx}, {wy}, {wz})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let ix = global_id.x;
    let iy = global_id.y;
    let iz = global_id.z;

    // Bounds check
    if (ix >= uniforms.grid_width || iy >= uniforms.grid_height || iz >= uniforms.grid_depth) {{
        return;
    }}

    let cell_idx = cell_index(ix, iy, iz);
    var cell = cache_cells[cell_idx];

    // Sample world position
    let world_pos = cell_to_world(ix, iy, iz);

    // Sample from probe texture (normalized coordinates)
    let probe_uv = (world_pos - uniforms.grid_origin) / uniforms.grid_extent;
    let new_radiance = textureSampleLevel(probe_texture, probe_sampler, probe_uv, 0.0);

    // Convert radiance to SH L1 (simplified - DC only for now)
    // Full implementation would project to SH basis
    let dc_scale = 0.2821; // 1 / (2 * sqrt(pi))
    let new_sh_r = vec4<f32>(new_radiance.r / dc_scale, 0.0, 0.0, 0.0);
    let new_sh_g = vec4<f32>(new_radiance.g / dc_scale, 0.0, 0.0, 0.0);
    let new_sh_b = vec4<f32>(new_radiance.b / dc_scale, 0.0, 0.0, 0.0);

    // Temporal blend
    let keep = 1.0 - uniforms.blend_factor;

    if (cell.state == 0u) {{ // EMPTY
        cell.sh_r = new_sh_r;
        cell.sh_g = new_sh_g;
        cell.sh_b = new_sh_b;
        cell.state = 1u; // ACCUMULATING
        cell.frames_accumulated = 1u;
        cell.variance = 1.0;
    }} else {{
        // Compute variance
        let diff_r = cell.sh_r - new_sh_r;
        let diff_g = cell.sh_g - new_sh_g;
        let diff_b = cell.sh_b - new_sh_b;
        let diff_sq = dot(diff_r, diff_r) + dot(diff_g, diff_g) + dot(diff_b, diff_b);
        cell.variance = cell.variance * keep + diff_sq * uniforms.blend_factor;

        // Blend
        cell.sh_r = cell.sh_r * keep + new_sh_r * uniforms.blend_factor;
        cell.sh_g = cell.sh_g * keep + new_sh_g * uniforms.blend_factor;
        cell.sh_b = cell.sh_b * keep + new_sh_b * uniforms.blend_factor;

        cell.frames_accumulated = cell.frames_accumulated + 1u;

        // Update state
        if (cell.frames_accumulated >= 4u && cell.variance < 0.001) {{
            cell.state = 2u; // STABLE
        }}
    }}

    cache_cells[cell_idx] = cell;
}}
"""


# ============================================================================
# Utility Functions
# ============================================================================


def estimate_cache_memory(quality: CacheQuality) -> int:
    """Estimate GPU memory for a cache quality preset.

    Args:
        quality: Quality preset

    Returns:
        Estimated memory in bytes
    """
    config = RadianceCacheConfig.from_quality(quality)
    return config.total_memory_bytes


def recommend_cache_quality(target_memory_mb: float) -> CacheQuality:
    """Recommend quality based on memory budget.

    Args:
        target_memory_mb: Memory budget in megabytes

    Returns:
        Recommended CacheQuality
    """
    target_bytes = target_memory_mb * 1024 * 1024

    for quality in [CacheQuality.ULTRA, CacheQuality.HIGH, CacheQuality.MEDIUM, CacheQuality.LOW]:
        if estimate_cache_memory(quality) <= target_bytes:
            return quality

    return CacheQuality.LOW


def create_constant_probe_sampler(
    radiance: Vec3,
) -> Callable[[Vec3, Vec3], Tuple[List[float], List[float], List[float]]]:
    """Create a probe sampler that returns constant radiance.

    Useful for testing.

    Args:
        radiance: Constant radiance value

    Returns:
        Sampler function
    """
    dc_scale = 1.0 / (2.0 * math.sqrt(math.pi))

    def sampler(pos: Vec3, direction: Vec3) -> Tuple[List[float], List[float], List[float]]:
        # SH L1: DC coefficient only
        sh_r = [radiance.x / dc_scale, 0.0, 0.0, 0.0]
        sh_g = [radiance.y / dc_scale, 0.0, 0.0, 0.0]
        sh_b = [radiance.z / dc_scale, 0.0, 0.0, 0.0]
        return (sh_r, sh_g, sh_b)

    return sampler


def create_gradient_probe_sampler(
    min_radiance: Vec3,
    max_radiance: Vec3,
    bounds: AABB,
) -> Callable[[Vec3, Vec3], Tuple[List[float], List[float], List[float]]]:
    """Create a probe sampler with gradient across bounds.

    Args:
        min_radiance: Radiance at bounds.min
        max_radiance: Radiance at bounds.max
        bounds: World bounds for gradient

    Returns:
        Sampler function
    """
    dc_scale = 1.0 / (2.0 * math.sqrt(math.pi))
    extent = bounds.max - bounds.min

    def sampler(pos: Vec3, direction: Vec3) -> Tuple[List[float], List[float], List[float]]:
        # Compute gradient factor [0, 1]
        local = pos - bounds.min
        tx = local.x / extent.x if extent.x > 0 else 0.5
        ty = local.y / extent.y if extent.y > 0 else 0.5
        tz = local.z / extent.z if extent.z > 0 else 0.5
        t = (tx + ty + tz) / 3.0
        t = max(0.0, min(1.0, t))

        # Interpolate radiance
        radiance = min_radiance.lerp(max_radiance, t)

        sh_r = [radiance.x / dc_scale, 0.0, 0.0, 0.0]
        sh_g = [radiance.y / dc_scale, 0.0, 0.0, 0.0]
        sh_b = [radiance.z / dc_scale, 0.0, 0.0, 0.0]
        return (sh_r, sh_g, sh_b)

    return sampler


def generate_radiance_cache_sampling_wgsl() -> str:
    """Generate WGSL helper code for sampling the radiance cache.

    Returns:
        WGSL code string
    """
    return """
// Radiance Cache Sampling Helpers (T-GIR-P2.7)

struct CacheSampleUniforms {
    grid_origin: vec3<f32>,
    grid_width: u32,
    grid_extent: vec3<f32>,
    grid_height: u32,
    inv_cell_size: vec3<f32>,
    grid_depth: u32,
}

struct CacheCell {
    sh_r: vec4<f32>,
    sh_g: vec4<f32>,
    sh_b: vec4<f32>,
    state: u32,
    frames_accumulated: u32,
    variance: f32,
    _pad: f32,
}

@group(1) @binding(0) var<uniform> cache_uniforms: CacheSampleUniforms;
@group(1) @binding(1) var<storage, read> cache_cells: array<CacheCell>;

// SH L1 basis evaluation
fn evaluate_sh_l1(sh: vec4<f32>, dir: vec3<f32>) -> f32 {
    let c0 = 0.2821; // 1 / (2 * sqrt(pi))
    let c1 = 0.4886; // sqrt(3) / (2 * sqrt(pi))
    return c0 * sh.x + c1 * (dir.y * sh.y + dir.z * sh.z + dir.x * sh.w);
}

// Sample radiance cache with trilinear interpolation
fn sample_radiance_cache(world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let local = world_pos - cache_uniforms.grid_origin;
    let uvw = local * cache_uniforms.inv_cell_size;

    // Bounds check
    if (any(uvw < vec3<f32>(0.0)) || any(uvw >= vec3<f32>(
        f32(cache_uniforms.grid_width),
        f32(cache_uniforms.grid_height),
        f32(cache_uniforms.grid_depth)
    ))) {
        return vec3<f32>(0.0);
    }

    // Base cell and fractional
    let base = vec3<i32>(floor(uvw - 0.5));
    let frac = uvw - 0.5 - vec3<f32>(base);

    // Trilinear interpolation
    var result = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var dz = 0; dz < 2; dz++) {
        for (var dy = 0; dy < 2; dy++) {
            for (var dx = 0; dx < 2; dx++) {
                let cell_idx = vec3<i32>(base.x + dx, base.y + dy, base.z + dz);

                // Clamp to grid
                let clamped = clamp(cell_idx, vec3<i32>(0), vec3<i32>(
                    i32(cache_uniforms.grid_width) - 1,
                    i32(cache_uniforms.grid_height) - 1,
                    i32(cache_uniforms.grid_depth) - 1
                ));

                let flat_idx = u32(clamped.z) * cache_uniforms.grid_width * cache_uniforms.grid_height +
                               u32(clamped.y) * cache_uniforms.grid_width + u32(clamped.x);

                let cell = cache_cells[flat_idx];

                // Skip empty cells
                if (cell.state == 0u) {
                    continue;
                }

                // Trilinear weight
                let d = vec3<f32>(f32(dx), f32(dy), f32(dz));
                let w = mix(vec3<f32>(1.0) - frac, frac, d);
                let weight = w.x * w.y * w.z;

                // Evaluate SH in normal direction
                let r = evaluate_sh_l1(cell.sh_r, normal);
                let g = evaluate_sh_l1(cell.sh_g, normal);
                let b = evaluate_sh_l1(cell.sh_b, normal);

                result += max(vec3<f32>(0.0), vec3<f32>(r, g, b)) * weight;
                total_weight += weight;
            }
        }
    }

    if (total_weight > 0.0) {
        return result / total_weight;
    }
    return vec3<f32>(0.0);
}

// Sample DC radiance only (faster, no direction)
fn sample_radiance_cache_dc(world_pos: vec3<f32>) -> vec3<f32> {
    let local = world_pos - cache_uniforms.grid_origin;
    let uvw = local * cache_uniforms.inv_cell_size;

    // Bounds check
    if (any(uvw < vec3<f32>(0.0)) || any(uvw >= vec3<f32>(
        f32(cache_uniforms.grid_width),
        f32(cache_uniforms.grid_height),
        f32(cache_uniforms.grid_depth)
    ))) {
        return vec3<f32>(0.0);
    }

    // Nearest cell
    let cell_idx = vec3<u32>(clamp(uvw, vec3<f32>(0.0), vec3<f32>(
        f32(cache_uniforms.grid_width) - 1.0,
        f32(cache_uniforms.grid_height) - 1.0,
        f32(cache_uniforms.grid_depth) - 1.0
    )));

    let flat_idx = cell_idx.z * cache_uniforms.grid_width * cache_uniforms.grid_height +
                   cell_idx.y * cache_uniforms.grid_width + cell_idx.x;

    let cell = cache_cells[flat_idx];

    if (cell.state == 0u) {
        return vec3<f32>(0.0);
    }

    // DC coefficient scaled by pi for irradiance
    let pi = 3.14159265359;
    return vec3<f32>(cell.sh_r.x, cell.sh_g.x, cell.sh_b.x) * pi;
}
"""

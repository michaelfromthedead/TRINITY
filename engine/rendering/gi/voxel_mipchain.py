"""Voxel Mip Chain and Storage for VXGI (T-GIR-P7.2).

This module implements 3D voxel texture mip chain generation and storage for
voxel-based global illumination. The mip chain averages 8 child voxels into
each parent voxel, preserving radiance and opacity while computing variance
at higher mip levels.

Features:
    - 3D texture with configurable mip levels (64^3, 128^3, 256^3)
    - Radiance + opacity storage per voxel (RGBA16F format)
    - Variance computation at high mip levels for filtering
    - GPU downsample shader (voxel_downsample.comp.wgsl)
    - Storage abstraction for wgpu 3D texture

Voxel Format:
    Each voxel stores RGBA16F:
        - RGB: Accumulated radiance (linear HDR)
        - A: Opacity (0.0 = empty, 1.0 = fully opaque)

Mip Chain Strategy:
    - Mip 0: Full resolution voxel grid
    - Mip N: Each voxel averages 8 children from Mip N-1
    - Opacity averaging uses coverage-weighted blending
    - Variance tracked at mip 2+ for cone tracing quality

References:
    - Crassin et al., "Interactive Indirect Illumination Using Voxel Cone Tracing"
    - NVIDIA, "GPU Gems 3 - Real-Time Voxelization"
    - Akenine-Moller, "Real-Time Rendering 4th Ed", Chapter 11.5
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator, Optional, Callable

import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Constants
# ============================================================================

# Minimum and maximum supported resolutions
MIN_VOXEL_RESOLUTION = 16
MAX_VOXEL_RESOLUTION = 512

# Default format: RGBA16F (half float)
VOXEL_FORMAT_BYTES_PER_TEXEL = 8  # 4 channels * 2 bytes (float16)

# Variance tracking starts at this mip level
VARIANCE_START_MIP = 2

# Minimum variance threshold for filtering decisions
MIN_VARIANCE_THRESHOLD = 0.001


# ============================================================================
# Voxel Resolution
# ============================================================================


class MipResolution(Enum):
    """Supported voxel mip chain resolutions.

    Each resolution defines the base size of the 3D voxel texture.
    Higher resolutions provide more detail but consume more memory.

    | Resolution | Memory (RGBA16F) | Mip Levels |
    |------------|------------------|------------|
    | RES_64     | 2 MB             | 6          |
    | RES_128    | 16 MB            | 7          |
    | RES_256    | 128 MB           | 8          |
    """

    RES_64 = 64
    RES_128 = 128
    RES_256 = 256

    @property
    def mip_count(self) -> int:
        """Number of mip levels for this resolution."""
        return int(math.log2(self.value)) + 1

    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage for full mip chain."""
        total = 0
        size = self.value
        while size >= 1:
            total += size * size * size * VOXEL_FORMAT_BYTES_PER_TEXEL
            size //= 2
        return total

    @classmethod
    def from_size(cls, size: int) -> MipResolution:
        """Get resolution enum from integer size.

        Args:
            size: Resolution size (64, 128, or 256)

        Returns:
            Corresponding MipResolution

        Raises:
            ValueError: If size is not a supported resolution
        """
        for res in cls:
            if res.value == size:
                return res
        raise ValueError(f"Unsupported voxel resolution: {size}. "
                        f"Supported: {[r.value for r in cls]}")


# ============================================================================
# Voxel Data
# ============================================================================


@dataclass
class VoxelData:
    """Single voxel data containing radiance and opacity.

    Attributes:
        radiance: RGB radiance values (linear HDR)
        opacity: Opacity value (0.0 = empty, 1.0 = opaque)
    """

    radiance: NDArray[np.float32]  # Shape (3,)
    opacity: float

    def __post_init__(self) -> None:
        """Validate radiance array shape."""
        self.radiance = np.asarray(self.radiance, dtype=np.float32)
        if self.radiance.shape != (3,):
            raise ValueError(f"Expected radiance shape (3,), got {self.radiance.shape}")
        self.opacity = float(self.opacity)

    @classmethod
    def empty(cls) -> VoxelData:
        """Create empty voxel (zero radiance, zero opacity)."""
        return cls(np.zeros(3, dtype=np.float32), 0.0)

    @classmethod
    def from_rgba(cls, rgba: NDArray[np.float32]) -> VoxelData:
        """Create from RGBA array.

        Args:
            rgba: Shape (4,) array [R, G, B, A]

        Returns:
            VoxelData with radiance=RGB, opacity=A
        """
        rgba = np.asarray(rgba, dtype=np.float32)
        return cls(rgba[:3].copy(), float(rgba[3]))

    def to_rgba(self) -> NDArray[np.float32]:
        """Convert to RGBA array.

        Returns:
            Shape (4,) array [R, G, B, opacity]
        """
        return np.array([*self.radiance, self.opacity], dtype=np.float32)

    def is_empty(self, threshold: float = 0.001) -> bool:
        """Check if voxel is effectively empty.

        Args:
            threshold: Opacity threshold below which voxel is empty

        Returns:
            True if opacity below threshold
        """
        return self.opacity < threshold

    def luminance(self) -> float:
        """Compute luminance of radiance.

        Returns:
            Perceptual luminance (Rec. 709)
        """
        return float(
            0.2126 * self.radiance[0] +
            0.7152 * self.radiance[1] +
            0.0722 * self.radiance[2]
        )


# ============================================================================
# Mip Level
# ============================================================================


@dataclass
class VoxelMipLevel:
    """Single mip level of the voxel texture.

    Stores voxel data as a 3D array of RGBA values plus optional variance
    data at higher mip levels.

    Attributes:
        level: Mip level index (0 = highest resolution)
        resolution: Size of this mip level (width = height = depth)
        data: 4D array shape (res, res, res, 4) containing RGBA
        variance: Optional 4D array for variance at high mips
    """

    level: int
    resolution: int
    data: NDArray[np.float32]  # Shape (res, res, res, 4)
    variance: Optional[NDArray[np.float32]] = None  # Shape (res, res, res, 4)

    def __post_init__(self) -> None:
        """Validate array shapes."""
        expected_shape = (self.resolution, self.resolution, self.resolution, 4)
        if self.data.shape != expected_shape:
            raise ValueError(f"Expected data shape {expected_shape}, "
                           f"got {self.data.shape}")
        if self.variance is not None and self.variance.shape != expected_shape:
            raise ValueError(f"Expected variance shape {expected_shape}, "
                           f"got {self.variance.shape}")

    @classmethod
    def create_empty(cls, level: int, resolution: int,
                    track_variance: bool = False) -> VoxelMipLevel:
        """Create empty mip level.

        Args:
            level: Mip level index
            resolution: Size of this mip level
            track_variance: Whether to allocate variance buffer

        Returns:
            Empty VoxelMipLevel
        """
        data = np.zeros((resolution, resolution, resolution, 4), dtype=np.float32)
        variance = None
        if track_variance:
            variance = np.zeros((resolution, resolution, resolution, 4),
                              dtype=np.float32)
        return cls(level=level, resolution=resolution, data=data, variance=variance)

    def get_voxel(self, x: int, y: int, z: int) -> VoxelData:
        """Get voxel data at coordinates.

        Args:
            x, y, z: Voxel coordinates

        Returns:
            VoxelData at position
        """
        rgba = self.data[x, y, z]
        return VoxelData.from_rgba(rgba)

    def set_voxel(self, x: int, y: int, z: int, voxel: VoxelData) -> None:
        """Set voxel data at coordinates.

        Args:
            x, y, z: Voxel coordinates
            voxel: VoxelData to store
        """
        self.data[x, y, z] = voxel.to_rgba()

    def get_variance(self, x: int, y: int, z: int) -> Optional[NDArray[np.float32]]:
        """Get variance at coordinates.

        Args:
            x, y, z: Voxel coordinates

        Returns:
            RGBA variance or None if not tracked
        """
        if self.variance is None:
            return None
        return self.variance[x, y, z].copy()

    def set_variance(self, x: int, y: int, z: int, var: NDArray[np.float32]) -> None:
        """Set variance at coordinates.

        Args:
            x, y, z: Voxel coordinates
            var: RGBA variance values
        """
        if self.variance is not None:
            self.variance[x, y, z] = var

    def memory_bytes(self) -> int:
        """Calculate memory usage in bytes.

        Returns:
            Total memory for data + variance
        """
        base = self.data.nbytes
        if self.variance is not None:
            base += self.variance.nbytes
        return base

    def voxel_count(self) -> int:
        """Get total number of voxels.

        Returns:
            resolution^3
        """
        return self.resolution ** 3

    def non_empty_count(self, threshold: float = 0.001) -> int:
        """Count non-empty voxels.

        Args:
            threshold: Opacity threshold

        Returns:
            Number of voxels with opacity >= threshold
        """
        return int(np.sum(self.data[:, :, :, 3] >= threshold))


# ============================================================================
# Voxel Downsample
# ============================================================================


@dataclass
class VoxelDownsampleConfig:
    """Configuration for voxel downsampling.

    Attributes:
        alpha_weighted: Use alpha (opacity) weighted averaging
        compute_variance: Track variance during downsampling
        variance_bias: Bias added to variance for stability
        preserve_energy: Normalize to preserve total energy
    """

    alpha_weighted: bool = True
    compute_variance: bool = True
    variance_bias: float = 0.0001
    preserve_energy: bool = True


def downsample_voxels(
    source: VoxelMipLevel,
    config: VoxelDownsampleConfig = VoxelDownsampleConfig(),
) -> VoxelMipLevel:
    """Downsample a voxel mip level by averaging 2x2x2 blocks.

    Each parent voxel is computed by averaging its 8 children:
        - Radiance: Weighted average by child opacity
        - Opacity: Average of child opacities
        - Variance: Computed from child value distribution

    Args:
        source: Source mip level to downsample
        config: Downsampling configuration

    Returns:
        New mip level at half resolution
    """
    if source.resolution < 2:
        raise ValueError(f"Cannot downsample resolution {source.resolution}")

    dst_res = source.resolution // 2
    track_variance = config.compute_variance and source.level >= VARIANCE_START_MIP - 1

    dst = VoxelMipLevel.create_empty(
        level=source.level + 1,
        resolution=dst_res,
        track_variance=track_variance,
    )

    for dz in range(dst_res):
        for dy in range(dst_res):
            for dx in range(dst_res):
                # Source coordinates for 8 children
                sx = dx * 2
                sy = dy * 2
                sz = dz * 2

                # Gather 8 children
                children = []
                for oz in range(2):
                    for oy in range(2):
                        for ox in range(2):
                            children.append(source.data[sx + ox, sy + oy, sz + oz])

                children_arr = np.array(children, dtype=np.float32)  # Shape (8, 4)

                # Compute averaged voxel
                if config.alpha_weighted:
                    avg, var = _alpha_weighted_average(children_arr, config)
                else:
                    avg, var = _simple_average(children_arr, config)

                dst.data[dx, dy, dz] = avg

                if track_variance and dst.variance is not None:
                    dst.variance[dx, dy, dz] = var

    return dst


def _alpha_weighted_average(
    children: NDArray[np.float32],
    config: VoxelDownsampleConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute alpha-weighted average of children.

    Args:
        children: Shape (8, 4) array of child RGBA values
        config: Downsample configuration

    Returns:
        Tuple of (averaged RGBA, variance RGBA)
    """
    alphas = children[:, 3]
    total_alpha = np.sum(alphas)

    if total_alpha < 1e-6:
        # All empty children
        return np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32)

    # Weighted average of RGB by alpha
    weights = alphas / total_alpha
    weighted_rgb = np.sum(children[:, :3] * weights[:, np.newaxis], axis=0)

    # Average alpha
    avg_alpha = total_alpha / 8.0

    # Energy preservation: scale by coverage
    if config.preserve_energy:
        coverage = np.sum(alphas > 0.001) / 8.0
        if coverage > 0:
            weighted_rgb *= coverage

    avg = np.array([*weighted_rgb, avg_alpha], dtype=np.float32)

    # Compute variance
    if config.compute_variance:
        # Variance weighted by alpha
        mean_rgba = np.array([*weighted_rgb, avg_alpha])
        diff = children - mean_rgba
        weighted_diff_sq = diff ** 2 * weights[:, np.newaxis]
        var = np.sum(weighted_diff_sq, axis=0) + config.variance_bias
    else:
        var = np.zeros(4, dtype=np.float32)

    return avg, var.astype(np.float32)


def _simple_average(
    children: NDArray[np.float32],
    config: VoxelDownsampleConfig,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute simple average of children.

    Args:
        children: Shape (8, 4) array of child RGBA values
        config: Downsample configuration

    Returns:
        Tuple of (averaged RGBA, variance RGBA)
    """
    avg = np.mean(children, axis=0).astype(np.float32)

    if config.compute_variance:
        var = np.var(children, axis=0).astype(np.float32) + config.variance_bias
    else:
        var = np.zeros(4, dtype=np.float32)

    return avg, var


# ============================================================================
# Voxel Mip Chain
# ============================================================================


@dataclass
class VoxelMipChain:
    """Complete voxel mip chain with all levels.

    Manages the full mip pyramid from base resolution down to 1x1x1.
    Provides methods for building, querying, and GPU upload.

    Attributes:
        base_resolution: Resolution of mip level 0
        levels: List of mip levels (index 0 = highest resolution)
        downsample_config: Configuration for mip generation
    """

    base_resolution: MipResolution
    levels: list[VoxelMipLevel] = field(default_factory=list)
    downsample_config: VoxelDownsampleConfig = field(
        default_factory=VoxelDownsampleConfig
    )

    def __post_init__(self) -> None:
        """Initialize levels if empty."""
        if not self.levels:
            self._create_levels()

    def _create_levels(self) -> None:
        """Create all mip levels."""
        res = self.base_resolution.value
        level = 0

        while res >= 1:
            track_variance = level >= VARIANCE_START_MIP
            mip = VoxelMipLevel.create_empty(level, res, track_variance)
            self.levels.append(mip)
            res //= 2
            level += 1

    @property
    def mip_count(self) -> int:
        """Number of mip levels."""
        return len(self.levels)

    def get_level(self, level: int) -> VoxelMipLevel:
        """Get mip level by index.

        Args:
            level: Mip level (0 = highest resolution)

        Returns:
            VoxelMipLevel at index

        Raises:
            IndexError: If level out of range
        """
        if level < 0 or level >= len(self.levels):
            raise IndexError(f"Mip level {level} out of range [0, {len(self.levels)})")
        return self.levels[level]

    def get_base(self) -> VoxelMipLevel:
        """Get base (highest resolution) mip level."""
        return self.levels[0]

    def get_top(self) -> VoxelMipLevel:
        """Get top (1x1x1) mip level."""
        return self.levels[-1]

    def build_mip_chain(self) -> None:
        """Generate all mip levels from base level.

        Downsamples from level 0 down to 1x1x1, averaging 8 children
        into each parent voxel.
        """
        for i in range(len(self.levels) - 1):
            source = self.levels[i]
            downsampled = downsample_voxels(source, self.downsample_config)
            self.levels[i + 1] = downsampled

    def set_base_voxel(self, x: int, y: int, z: int, voxel: VoxelData) -> None:
        """Set voxel in base level.

        Args:
            x, y, z: Voxel coordinates
            voxel: VoxelData to store
        """
        self.get_base().set_voxel(x, y, z, voxel)

    def sample_voxel(self, x: int, y: int, z: int, level: int) -> VoxelData:
        """Sample voxel at specified mip level.

        Args:
            x, y, z: Voxel coordinates (at the target mip level)
            level: Mip level

        Returns:
            VoxelData at position
        """
        return self.get_level(level).get_voxel(x, y, z)

    def sample_trilinear(
        self,
        u: float, v: float, w: float,
        level: int,
    ) -> VoxelData:
        """Sample voxel with trilinear interpolation.

        Args:
            u, v, w: Normalized coordinates [0, 1]
            level: Mip level

        Returns:
            Trilinearly interpolated VoxelData
        """
        mip = self.get_level(level)
        res = mip.resolution

        # Convert to voxel coordinates
        fx = u * (res - 1)
        fy = v * (res - 1)
        fz = w * (res - 1)

        # Integer and fractional parts
        x0 = int(fx)
        y0 = int(fy)
        z0 = int(fz)
        x1 = min(x0 + 1, res - 1)
        y1 = min(y0 + 1, res - 1)
        z1 = min(z0 + 1, res - 1)

        xf = fx - x0
        yf = fy - y0
        zf = fz - z0

        # Sample 8 corners
        def lerp(a: NDArray, b: NDArray, t: float) -> NDArray:
            return a * (1.0 - t) + b * t

        # Trilinear interpolation
        c000 = mip.data[x0, y0, z0]
        c001 = mip.data[x0, y0, z1]
        c010 = mip.data[x0, y1, z0]
        c011 = mip.data[x0, y1, z1]
        c100 = mip.data[x1, y0, z0]
        c101 = mip.data[x1, y0, z1]
        c110 = mip.data[x1, y1, z0]
        c111 = mip.data[x1, y1, z1]

        c00 = lerp(c000, c001, zf)
        c01 = lerp(c010, c011, zf)
        c10 = lerp(c100, c101, zf)
        c11 = lerp(c110, c111, zf)

        c0 = lerp(c00, c01, yf)
        c1 = lerp(c10, c11, yf)

        result = lerp(c0, c1, xf)
        return VoxelData.from_rgba(result)

    def get_variance_at(
        self, x: int, y: int, z: int, level: int
    ) -> Optional[NDArray[np.float32]]:
        """Get variance at specified position and mip level.

        Args:
            x, y, z: Voxel coordinates
            level: Mip level

        Returns:
            RGBA variance or None if not tracked at this level
        """
        return self.get_level(level).get_variance(x, y, z)

    def total_memory_bytes(self) -> int:
        """Calculate total memory usage.

        Returns:
            Sum of all mip level memory
        """
        return sum(level.memory_bytes() for level in self.levels)

    def get_statistics(self) -> dict:
        """Get mip chain statistics.

        Returns:
            Dictionary with mip chain information
        """
        return {
            "base_resolution": self.base_resolution.value,
            "mip_count": self.mip_count,
            "total_memory_mb": self.total_memory_bytes() / (1024 * 1024),
            "levels": [
                {
                    "level": i,
                    "resolution": level.resolution,
                    "voxel_count": level.voxel_count(),
                    "non_empty": level.non_empty_count(),
                    "has_variance": level.variance is not None,
                }
                for i, level in enumerate(self.levels)
            ],
        }


# ============================================================================
# Voxel Storage (GPU Abstraction)
# ============================================================================


class VoxelStorageFormat(Enum):
    """GPU texture formats for voxel storage."""

    RGBA16F = auto()  # Half-float, 8 bytes/texel
    RGBA32F = auto()  # Full float, 16 bytes/texel
    R11G11B10F = auto()  # Packed float, 4 bytes/texel (no alpha)

    @property
    def bytes_per_texel(self) -> int:
        """Bytes per texel for this format."""
        return {
            VoxelStorageFormat.RGBA16F: 8,
            VoxelStorageFormat.RGBA32F: 16,
            VoxelStorageFormat.R11G11B10F: 4,
        }[self]

    @property
    def wgpu_format(self) -> str:
        """wgpu format string."""
        return {
            VoxelStorageFormat.RGBA16F: "rgba16float",
            VoxelStorageFormat.RGBA32F: "rgba32float",
            VoxelStorageFormat.R11G11B10F: "rg11b10ufloat",
        }[self]


@dataclass
class VoxelStorageConfig:
    """Configuration for voxel GPU storage.

    Attributes:
        resolution: Base voxel resolution
        format: GPU texture format
        enable_mips: Generate mip levels
        enable_variance: Store variance texture
        anisotropic_filtering: Enable anisotropic filtering
    """

    resolution: MipResolution = MipResolution.RES_128
    format: VoxelStorageFormat = VoxelStorageFormat.RGBA16F
    enable_mips: bool = True
    enable_variance: bool = True
    anisotropic_filtering: bool = True

    def total_memory_bytes(self) -> int:
        """Estimate total GPU memory usage.

        Returns:
            Estimated bytes for all textures
        """
        res = self.resolution.value
        bpt = self.format.bytes_per_texel
        total = 0

        # Main texture with mips
        mip_res = res
        while mip_res >= 1:
            total += mip_res ** 3 * bpt
            if not self.enable_mips:
                break
            mip_res //= 2

        # Variance texture (same size)
        if self.enable_variance:
            mip_res = res
            while mip_res >= 1:
                total += mip_res ** 3 * bpt
                if not self.enable_mips:
                    break
                mip_res //= 2

        return total


@dataclass
class VoxelStorage:
    """GPU storage abstraction for voxel textures.

    Manages wgpu 3D texture resources for voxel data and provides
    upload/download functionality.

    Attributes:
        config: Storage configuration
        _main_texture_handle: Handle to main voxel texture (opaque)
        _variance_texture_handle: Handle to variance texture (opaque)
        _dirty_regions: Regions needing upload
    """

    config: VoxelStorageConfig
    _main_texture_handle: Optional[int] = None
    _variance_texture_handle: Optional[int] = None
    _dirty_regions: list[tuple[int, int, int, int, int, int]] = field(
        default_factory=list
    )

    def is_initialized(self) -> bool:
        """Check if GPU resources are allocated."""
        return self._main_texture_handle is not None

    def get_texture_descriptor(self) -> dict:
        """Get wgpu texture descriptor for main voxel texture.

        Returns:
            Dictionary suitable for wgpu texture creation
        """
        res = self.config.resolution.value
        mip_count = self.config.resolution.mip_count if self.config.enable_mips else 1

        return {
            "label": "voxel_radiance_3d",
            "size": {"width": res, "height": res, "depth_or_array_layers": res},
            "mip_level_count": mip_count,
            "sample_count": 1,
            "dimension": "3d",
            "format": self.config.format.wgpu_format,
            "usage": ["texture_binding", "storage_binding", "copy_dst", "copy_src"],
        }

    def get_variance_texture_descriptor(self) -> Optional[dict]:
        """Get wgpu texture descriptor for variance texture.

        Returns:
            Dictionary for wgpu texture creation, or None if disabled
        """
        if not self.config.enable_variance:
            return None

        res = self.config.resolution.value
        mip_count = self.config.resolution.mip_count if self.config.enable_mips else 1

        return {
            "label": "voxel_variance_3d",
            "size": {"width": res, "height": res, "depth_or_array_layers": res},
            "mip_level_count": mip_count,
            "sample_count": 1,
            "dimension": "3d",
            "format": self.config.format.wgpu_format,
            "usage": ["texture_binding", "storage_binding", "copy_dst"],
        }

    def mark_dirty(
        self,
        x0: int, y0: int, z0: int,
        x1: int, y1: int, z1: int,
    ) -> None:
        """Mark a region as dirty for upload.

        Args:
            x0, y0, z0: Minimum corner
            x1, y1, z1: Maximum corner (exclusive)
        """
        self._dirty_regions.append((x0, y0, z0, x1, y1, z1))

    def mark_all_dirty(self) -> None:
        """Mark entire texture as dirty."""
        res = self.config.resolution.value
        self._dirty_regions = [(0, 0, 0, res, res, res)]

    def clear_dirty(self) -> None:
        """Clear dirty regions after upload."""
        self._dirty_regions.clear()

    def has_dirty_regions(self) -> bool:
        """Check if any regions need upload."""
        return len(self._dirty_regions) > 0

    def get_dirty_regions(self) -> list[tuple[int, int, int, int, int, int]]:
        """Get list of dirty regions.

        Returns:
            List of (x0, y0, z0, x1, y1, z1) tuples
        """
        return self._dirty_regions.copy()

    def prepare_upload_data(
        self,
        mip_chain: VoxelMipChain,
        level: int = 0,
    ) -> bytes:
        """Prepare mip level data for GPU upload.

        Converts float32 data to the configured format.

        Args:
            mip_chain: Source mip chain
            level: Mip level to upload

        Returns:
            Packed bytes in GPU format
        """
        mip = mip_chain.get_level(level)
        data = mip.data

        if self.config.format == VoxelStorageFormat.RGBA32F:
            return data.tobytes()
        elif self.config.format == VoxelStorageFormat.RGBA16F:
            # Convert to float16
            data_f16 = data.astype(np.float16)
            return data_f16.tobytes()
        elif self.config.format == VoxelStorageFormat.R11G11B10F:
            # Pack RGB to R11G11B10 (alpha discarded)
            return _pack_r11g11b10(data[:, :, :, :3])
        else:
            raise ValueError(f"Unknown format: {self.config.format}")

    def get_sampler_descriptor(self) -> dict:
        """Get wgpu sampler descriptor for voxel texture.

        Returns:
            Dictionary for wgpu sampler creation
        """
        return {
            "label": "voxel_sampler",
            "address_mode_u": "clamp_to_edge",
            "address_mode_v": "clamp_to_edge",
            "address_mode_w": "clamp_to_edge",
            "mag_filter": "linear",
            "min_filter": "linear",
            "mipmap_filter": "linear" if self.config.enable_mips else "nearest",
            "max_anisotropy": 8 if self.config.anisotropic_filtering else 1,
        }

    def get_binding_layout(self) -> list[dict]:
        """Get bind group layout entries for voxel textures.

        Returns:
            List of binding layout entries
        """
        layouts = [
            {
                "binding": 0,
                "visibility": ["fragment", "compute"],
                "texture": {
                    "sample_type": "float",
                    "view_dimension": "3d",
                    "multisampled": False,
                },
            },
            {
                "binding": 1,
                "visibility": ["fragment", "compute"],
                "sampler": {"type": "filtering"},
            },
        ]

        if self.config.enable_variance:
            layouts.append({
                "binding": 2,
                "visibility": ["fragment", "compute"],
                "texture": {
                    "sample_type": "float",
                    "view_dimension": "3d",
                    "multisampled": False,
                },
            })

        return layouts


def _pack_r11g11b10(rgb: NDArray[np.float32]) -> bytes:
    """Pack RGB float data to R11G11B10 format.

    Args:
        rgb: Shape (..., 3) RGB float data

    Returns:
        Packed bytes (4 bytes per texel)
    """
    # Clamp to positive
    rgb = np.clip(rgb, 0.0, 65504.0)  # float16 max

    # Convert to unsigned 11/11/10 bit representation
    # This is a simplified packing - real implementation would use proper float encoding
    r = np.clip(rgb[..., 0] / 65504.0 * 2047, 0, 2047).astype(np.uint32)
    g = np.clip(rgb[..., 1] / 65504.0 * 2047, 0, 2047).astype(np.uint32)
    b = np.clip(rgb[..., 2] / 65504.0 * 1023, 0, 1023).astype(np.uint32)

    packed = r | (g << 11) | (b << 22)
    return packed.tobytes()


# ============================================================================
# WGSL Shader Generator
# ============================================================================


def generate_voxel_downsample_wgsl() -> str:
    """Generate voxel downsample compute shader.

    Returns:
        WGSL compute shader code for voxel mip generation
    """
    return '''// Voxel Downsample Compute Shader
// Generates mip level N+1 from mip level N by averaging 2x2x2 blocks

struct DownsampleUniforms {
    src_resolution: u32,
    dst_resolution: u32,
    mip_level: u32,
    alpha_weighted: u32,  // 1 = alpha weighted, 0 = simple average
}

@group(0) @binding(0) var src_texture: texture_3d<f32>;
@group(0) @binding(1) var dst_texture: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> uniforms: DownsampleUniforms;

// Optional variance output
@group(0) @binding(3) var dst_variance: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_coord = gid;

    // Bounds check
    if (any(dst_coord >= vec3<u32>(uniforms.dst_resolution))) {
        return;
    }

    // Source coordinates (2x2x2 block)
    let src_base = dst_coord * 2u;

    // Gather 8 children
    var children: array<vec4<f32>, 8>;
    var idx = 0u;
    for (var z = 0u; z < 2u; z++) {
        for (var y = 0u; y < 2u; y++) {
            for (var x = 0u; x < 2u; x++) {
                let coord = src_base + vec3<u32>(x, y, z);
                children[idx] = textureLoad(src_texture, coord, 0);
                idx++;
            }
        }
    }

    var result: vec4<f32>;
    var variance: vec4<f32>;

    if (uniforms.alpha_weighted != 0u) {
        // Alpha-weighted average
        var total_alpha = 0.0;
        for (var i = 0u; i < 8u; i++) {
            total_alpha += children[i].a;
        }

        if (total_alpha < 0.000001) {
            result = vec4<f32>(0.0);
            variance = vec4<f32>(0.0);
        } else {
            // Weighted RGB
            var weighted_rgb = vec3<f32>(0.0);
            for (var i = 0u; i < 8u; i++) {
                let weight = children[i].a / total_alpha;
                weighted_rgb += children[i].rgb * weight;
            }

            // Average alpha
            let avg_alpha = total_alpha / 8.0;

            // Energy preservation
            var non_empty = 0u;
            for (var i = 0u; i < 8u; i++) {
                if (children[i].a > 0.001) {
                    non_empty++;
                }
            }
            let coverage = f32(non_empty) / 8.0;
            weighted_rgb *= coverage;

            result = vec4<f32>(weighted_rgb, avg_alpha);

            // Variance computation
            var var_sum = vec4<f32>(0.0);
            for (var i = 0u; i < 8u; i++) {
                let weight = children[i].a / total_alpha;
                let diff = children[i] - result;
                var_sum += diff * diff * weight;
            }
            variance = var_sum + vec4<f32>(0.0001);  // Bias for stability
        }
    } else {
        // Simple average
        var sum = vec4<f32>(0.0);
        for (var i = 0u; i < 8u; i++) {
            sum += children[i];
        }
        result = sum / 8.0;

        // Simple variance
        var var_sum = vec4<f32>(0.0);
        for (var i = 0u; i < 8u; i++) {
            let diff = children[i] - result;
            var_sum += diff * diff;
        }
        variance = var_sum / 8.0 + vec4<f32>(0.0001);
    }

    // Write outputs
    textureStore(dst_texture, dst_coord, result);

    // Write variance if mip level >= 2
    if (uniforms.mip_level >= 2u) {
        textureStore(dst_variance, dst_coord, variance);
    }
}
'''


def generate_voxel_sample_wgsl() -> str:
    """Generate voxel sampling helper functions.

    Returns:
        WGSL code for voxel texture sampling
    """
    return '''// Voxel Sampling Utilities

// Sample voxel with trilinear interpolation
fn sample_voxel_trilinear(
    voxel_tex: texture_3d<f32>,
    voxel_sampler: sampler,
    uvw: vec3<f32>,
    mip_level: f32,
) -> vec4<f32> {
    return textureSampleLevel(voxel_tex, voxel_sampler, uvw, mip_level);
}

// Sample voxel with cone tracing LOD selection
fn sample_voxel_cone(
    voxel_tex: texture_3d<f32>,
    voxel_sampler: sampler,
    uvw: vec3<f32>,
    cone_diameter: f32,  // Diameter in voxel units
    voxel_size: f32,     // Size of one voxel
) -> vec4<f32> {
    // Compute LOD from cone diameter
    let lod = log2(max(cone_diameter / voxel_size, 1.0));
    return textureSampleLevel(voxel_tex, voxel_sampler, uvw, lod);
}

// Get variance at position for filtering decisions
fn get_voxel_variance(
    variance_tex: texture_3d<f32>,
    uvw: vec3<f32>,
    mip_level: i32,
) -> vec4<f32> {
    return textureLoad(variance_tex, vec3<i32>(uvw * f32(textureDimensions(variance_tex, mip_level).x)), mip_level);
}

// Adaptive mip selection based on variance
fn compute_adaptive_mip(
    variance: vec4<f32>,
    base_mip: f32,
    variance_threshold: f32,
) -> f32 {
    // Use higher mip (more filtering) when variance is high
    let avg_variance = (variance.r + variance.g + variance.b) / 3.0;
    let mip_bias = smoothstep(0.0, variance_threshold, avg_variance) * 2.0;
    return base_mip + mip_bias;
}
'''


# ============================================================================
# Utility Functions
# ============================================================================


def compute_mip_resolution(base_resolution: int, mip_level: int) -> int:
    """Compute resolution at a specific mip level.

    Args:
        base_resolution: Resolution at mip 0
        mip_level: Target mip level

    Returns:
        Resolution at target mip (minimum 1)
    """
    return max(1, base_resolution >> mip_level)


def compute_mip_count(resolution: int) -> int:
    """Compute number of mip levels for a resolution.

    Args:
        resolution: Base resolution

    Returns:
        Number of mip levels (log2(resolution) + 1)
    """
    if resolution < 1:
        raise ValueError("Resolution must be at least 1")
    return int(math.log2(resolution)) + 1


def estimate_voxel_memory(
    resolution: int,
    format: VoxelStorageFormat = VoxelStorageFormat.RGBA16F,
    include_variance: bool = True,
    include_mips: bool = True,
) -> int:
    """Estimate GPU memory for voxel storage.

    Args:
        resolution: Base voxel resolution
        format: Texture format
        include_variance: Include variance texture
        include_mips: Include all mip levels

    Returns:
        Estimated memory in bytes
    """
    bpt = format.bytes_per_texel
    total = 0

    res = resolution
    while res >= 1:
        total += res ** 3 * bpt
        if not include_mips:
            break
        res //= 2

    if include_variance:
        total *= 2  # Variance same size as main

    return total


def create_test_voxel_pattern(
    resolution: int,
    pattern: str = "sphere",
) -> VoxelMipChain:
    """Create a test voxel pattern for debugging.

    Args:
        resolution: Voxel resolution
        pattern: Pattern type ("sphere", "cube", "gradient", "checkerboard")

    Returns:
        VoxelMipChain with test pattern
    """
    voxel_res = MipResolution.from_size(resolution)
    chain = VoxelMipChain(base_resolution=voxel_res)
    base = chain.get_base()

    center = resolution / 2.0
    radius = resolution / 3.0

    for z in range(resolution):
        for y in range(resolution):
            for x in range(resolution):
                # Compute test value based on pattern
                if pattern == "sphere":
                    dist = math.sqrt(
                        (x - center) ** 2 +
                        (y - center) ** 2 +
                        (z - center) ** 2
                    )
                    if dist < radius:
                        opacity = 1.0 - (dist / radius)
                        radiance = np.array([1.0, 0.5, 0.2]) * opacity
                    else:
                        opacity = 0.0
                        radiance = np.zeros(3)

                elif pattern == "cube":
                    in_cube = (
                        abs(x - center) < radius and
                        abs(y - center) < radius and
                        abs(z - center) < radius
                    )
                    opacity = 1.0 if in_cube else 0.0
                    radiance = np.array([0.8, 0.2, 0.1]) if in_cube else np.zeros(3)

                elif pattern == "gradient":
                    opacity = x / resolution
                    radiance = np.array([
                        x / resolution,
                        y / resolution,
                        z / resolution,
                    ])

                elif pattern == "checkerboard":
                    check = ((x // 8) + (y // 8) + (z // 8)) % 2
                    opacity = 1.0 if check else 0.0
                    radiance = np.array([1.0, 1.0, 1.0]) if check else np.zeros(3)

                else:
                    raise ValueError(f"Unknown pattern: {pattern}")

                base.set_voxel(x, y, z, VoxelData(radiance.astype(np.float32), opacity))

    # Build mip chain
    chain.build_mip_chain()

    return chain


def validate_mip_chain(chain: VoxelMipChain) -> list[str]:
    """Validate mip chain consistency.

    Checks that each mip level is properly derived from its parent.

    Args:
        chain: VoxelMipChain to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    for i in range(1, chain.mip_count):
        parent = chain.get_level(i - 1)
        child = chain.get_level(i)

        # Check resolution relationship
        expected_res = parent.resolution // 2
        if child.resolution != expected_res:
            errors.append(
                f"Mip {i}: Expected resolution {expected_res}, "
                f"got {child.resolution}"
            )
            continue  # Skip spot check if resolution is wrong

        # Spot check a few voxels
        for _ in range(min(10, child.resolution ** 3)):
            cx = np.random.randint(0, child.resolution)
            cy = np.random.randint(0, child.resolution)
            cz = np.random.randint(0, child.resolution)

            child_val = child.get_voxel(cx, cy, cz)

            # Compute expected from parent
            px = cx * 2
            py = cy * 2
            pz = cz * 2

            parent_vals = []
            for oz in range(2):
                for oy in range(2):
                    for ox in range(2):
                        pv = parent.get_voxel(px + ox, py + oy, pz + oz)
                        parent_vals.append(pv.to_rgba())

            parent_arr = np.array(parent_vals)
            expected_avg = np.mean(parent_arr, axis=0)

            # Allow some tolerance due to alpha weighting
            diff = np.abs(child_val.to_rgba() - expected_avg)
            if np.any(diff > 0.5):  # Loose tolerance
                errors.append(
                    f"Mip {i} at ({cx},{cy},{cz}): Value differs "
                    f"significantly from parent average"
                )
                break  # Don't flood with errors

    return errors

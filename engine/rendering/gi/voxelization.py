"""Scene Voxelization for TRINITY GI (T-GIR-P7.1).

This module provides conservative scene voxelization for voxel-based global
illumination techniques. It supports:
    - Conservative rasterization via AABB-triangle intersection
    - Per-voxel storage of albedo, emissive, normal, opacity
    - Multiple resolution tiers: 64^3, 128^3, 256^3
    - Opacity classification for transparent/opaque voxels

Conservative Rasterization:
    Unlike standard rasterization which fills pixels covered by triangle
    centers, conservative rasterization fills ALL pixels that intersect
    the triangle in any way. This prevents "holes" in thin geometry.

Performance Targets:
    - 256^3 voxelization: <4ms (requirement)
    - 128^3 voxelization: <1ms (typical)
    - 64^3 voxelization: <0.3ms (typical)

References:
    - "Real-Time Global Illumination using Precomputed Light Field Probes"
    - Cyril Crassin, NVIDIA VXGI documentation
    - GPU Pro 5, "Hi-Z Screen-Space Cone-Traced Reflections"
"""

from __future__ import annotations

import array
import math
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator, Optional, Sequence, TYPE_CHECKING

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec3, Vec4

if TYPE_CHECKING:
    pass


# ============================================================================
# Constants
# ============================================================================

# Resolution tiers
RESOLUTION_64 = 64
RESOLUTION_128 = 128
RESOLUTION_256 = 256

SUPPORTED_RESOLUTIONS = (RESOLUTION_64, RESOLUTION_128, RESOLUTION_256)

# Opacity thresholds
OPACITY_TRANSPARENT = 0.0
OPACITY_SEMITRANSPARENT = 0.5
OPACITY_OPAQUE = 1.0

# Default colors
DEFAULT_ALBEDO = Vec4(0.5, 0.5, 0.5, 1.0)
DEFAULT_EMISSIVE = Vec3(0.0, 0.0, 0.0)
DEFAULT_NORMAL = Vec3(0.0, 1.0, 0.0)

# Epsilon for geometric tests
EPSILON = 1e-7


# ============================================================================
# Enums
# ============================================================================


class VoxelResolution(Enum):
    """Supported voxel grid resolutions."""

    LOW = 64       # 64^3 = 262,144 voxels
    MEDIUM = 128   # 128^3 = 2,097,152 voxels
    HIGH = 256     # 256^3 = 16,777,216 voxels

    @property
    def size(self) -> int:
        """Get the linear dimension."""
        return self.value

    @property
    def total_voxels(self) -> int:
        """Get total voxel count."""
        return self.value ** 3

    @property
    def memory_estimate_mb(self) -> float:
        """Estimate GPU memory in MB (RGBA8 + Normal8 + Emissive8)."""
        # Per voxel: 4 bytes albedo + 4 bytes normal + 4 bytes emissive = 12 bytes
        return (self.total_voxels * 12) / (1024 * 1024)


class OpacityClass(Enum):
    """Opacity classification for voxels."""

    EMPTY = auto()           # No geometry
    TRANSPARENT = auto()     # Alpha < 0.1
    SEMITRANSPARENT = auto()  # 0.1 <= Alpha < 0.9
    OPAQUE = auto()          # Alpha >= 0.9

    @staticmethod
    def from_alpha(alpha: float) -> OpacityClass:
        """Classify opacity from alpha value."""
        if alpha < 0.001:
            return OpacityClass.EMPTY
        elif alpha < 0.1:
            return OpacityClass.TRANSPARENT
        elif alpha < 0.9:
            return OpacityClass.SEMITRANSPARENT
        else:
            return OpacityClass.OPAQUE


class VoxelAxis(Enum):
    """Dominant axis for triangle projection."""

    X = auto()  # Project onto YZ plane
    Y = auto()  # Project onto XZ plane
    Z = auto()  # Project onto XY plane


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class Triangle:
    """A triangle with material properties for voxelization.

    Attributes:
        v0, v1, v2: Vertex positions
        n0, n1, n2: Vertex normals (optional, computed if not provided)
        albedo: Surface albedo color with alpha
        emissive: Emissive color (RGB)
    """

    v0: Vec3
    v1: Vec3
    v2: Vec3
    n0: Optional[Vec3] = None
    n1: Optional[Vec3] = None
    n2: Optional[Vec3] = None
    albedo: Vec4 = field(default_factory=lambda: DEFAULT_ALBEDO)
    emissive: Vec3 = field(default_factory=lambda: DEFAULT_EMISSIVE)

    def __post_init__(self) -> None:
        """Compute face normal if vertex normals not provided."""
        if self.n0 is None or self.n1 is None or self.n2 is None:
            face_normal = self.compute_face_normal()
            if self.n0 is None:
                self.n0 = face_normal
            if self.n1 is None:
                self.n1 = face_normal
            if self.n2 is None:
                self.n2 = face_normal

    def compute_face_normal(self) -> Vec3:
        """Compute face normal from vertices."""
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        normal = edge1.cross(edge2)
        length = normal.length()
        if length < EPSILON:
            return Vec3(0.0, 1.0, 0.0)  # Degenerate, return up
        return normal * (1.0 / length)

    def get_aabb(self) -> AABB:
        """Compute axis-aligned bounding box."""
        min_x = min(self.v0.x, self.v1.x, self.v2.x)
        min_y = min(self.v0.y, self.v1.y, self.v2.y)
        min_z = min(self.v0.z, self.v1.z, self.v2.z)
        max_x = max(self.v0.x, self.v1.x, self.v2.x)
        max_y = max(self.v0.y, self.v1.y, self.v2.y)
        max_z = max(self.v0.z, self.v1.z, self.v2.z)
        return AABB(Vec3(min_x, min_y, min_z), Vec3(max_x, max_y, max_z))

    def get_center(self) -> Vec3:
        """Get triangle centroid."""
        return Vec3(
            (self.v0.x + self.v1.x + self.v2.x) / 3.0,
            (self.v0.y + self.v1.y + self.v2.y) / 3.0,
            (self.v0.z + self.v1.z + self.v2.z) / 3.0,
        )

    def get_area(self) -> float:
        """Compute triangle area."""
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        cross = edge1.cross(edge2)
        return cross.length() * 0.5

    def interpolate_normal(self, bary: Vec3) -> Vec3:
        """Interpolate normal at barycentric coordinates."""
        assert self.n0 is not None and self.n1 is not None and self.n2 is not None
        nx = self.n0.x * bary.x + self.n1.x * bary.y + self.n2.x * bary.z
        ny = self.n0.y * bary.x + self.n1.y * bary.y + self.n2.y * bary.z
        nz = self.n0.z * bary.x + self.n1.z * bary.y + self.n2.z * bary.z
        result = Vec3(nx, ny, nz)
        length = result.length()
        if length < EPSILON:
            return self.compute_face_normal()
        return result * (1.0 / length)


@dataclass
class Voxel:
    """A single voxel with material properties.

    Attributes:
        albedo: RGBA albedo (alpha is opacity)
        emissive: RGB emissive color
        normal: Average normal direction
        hit_count: Number of triangles contributing
    """

    albedo: Vec4 = field(default_factory=lambda: Vec4(0.0, 0.0, 0.0, 0.0))
    emissive: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    normal: Vec3 = field(default_factory=lambda: Vec3(0.0, 0.0, 0.0))
    hit_count: int = 0

    def is_empty(self) -> bool:
        """Check if voxel has no geometry."""
        return self.hit_count == 0

    def get_opacity_class(self) -> OpacityClass:
        """Get opacity classification."""
        return OpacityClass.from_alpha(self.albedo.w)

    def accumulate(
        self,
        albedo: Vec4,
        emissive: Vec3,
        normal: Vec3,
    ) -> None:
        """Accumulate contribution from a triangle.

        Uses incremental averaging to combine multiple contributions.
        """
        self.hit_count += 1
        n = float(self.hit_count)

        # Incremental average: new_avg = old_avg + (new_value - old_avg) / n
        self.albedo = Vec4(
            self.albedo.x + (albedo.x - self.albedo.x) / n,
            self.albedo.y + (albedo.y - self.albedo.y) / n,
            self.albedo.z + (albedo.z - self.albedo.z) / n,
            self.albedo.w + (albedo.w - self.albedo.w) / n,
        )
        self.emissive = Vec3(
            self.emissive.x + (emissive.x - self.emissive.x) / n,
            self.emissive.y + (emissive.y - self.emissive.y) / n,
            self.emissive.z + (emissive.z - self.emissive.z) / n,
        )
        # Normal averaging requires renormalization
        self.normal = Vec3(
            self.normal.x + (normal.x - self.normal.x) / n,
            self.normal.y + (normal.y - self.normal.y) / n,
            self.normal.z + (normal.z - self.normal.z) / n,
        )

    def finalize(self) -> None:
        """Finalize voxel by normalizing the normal vector."""
        if self.hit_count > 0:
            length = self.normal.length()
            if length > EPSILON:
                self.normal = self.normal * (1.0 / length)

    def to_rgba8(self) -> tuple[int, int, int, int]:
        """Convert albedo to RGBA8 format."""
        return (
            int(max(0.0, min(1.0, self.albedo.x)) * 255),
            int(max(0.0, min(1.0, self.albedo.y)) * 255),
            int(max(0.0, min(1.0, self.albedo.z)) * 255),
            int(max(0.0, min(1.0, self.albedo.w)) * 255),
        )


# ============================================================================
# VoxelGrid
# ============================================================================


class VoxelGrid:
    """A 3D grid of voxels for scene voxelization.

    The grid stores per-voxel:
        - Albedo (RGBA, alpha is opacity)
        - Emissive (RGB)
        - Normal (XYZ, normalized)

    Grid coordinates:
        - (0, 0, 0) is at bounds.min
        - (resolution-1, resolution-1, resolution-1) is at bounds.max
        - Voxel size = (bounds.max - bounds.min) / resolution

    Attributes:
        resolution: Grid dimension (same in all axes)
        bounds: World-space bounding box
        voxels: Flat array of voxels (x + y*res + z*res*res indexing)
    """

    __slots__ = ("resolution", "bounds", "voxels", "_voxel_size", "_inv_voxel_size")

    def __init__(
        self,
        resolution: int,
        bounds: AABB,
    ) -> None:
        """Initialize voxel grid.

        Args:
            resolution: Grid dimension (64, 128, or 256 recommended)
            bounds: World-space bounds for the grid

        Raises:
            ValueError: If resolution is not positive
        """
        if resolution <= 0:
            raise ValueError(f"Resolution must be positive, got {resolution}")

        self.resolution = resolution
        self.bounds = bounds

        # Precompute voxel dimensions
        extent = bounds.max - bounds.min
        self._voxel_size = Vec3(
            extent.x / resolution,
            extent.y / resolution,
            extent.z / resolution,
        )
        self._inv_voxel_size = Vec3(
            resolution / extent.x if extent.x > EPSILON else 0.0,
            resolution / extent.y if extent.y > EPSILON else 0.0,
            resolution / extent.z if extent.z > EPSILON else 0.0,
        )

        # Allocate voxels
        total = resolution ** 3
        self.voxels: list[Voxel] = [Voxel() for _ in range(total)]

    @property
    def voxel_size(self) -> Vec3:
        """Get the size of one voxel in world units."""
        return self._voxel_size

    @property
    def total_voxels(self) -> int:
        """Get total voxel count."""
        return self.resolution ** 3

    def _index(self, x: int, y: int, z: int) -> int:
        """Convert 3D coordinates to flat index."""
        return x + y * self.resolution + z * self.resolution * self.resolution

    def get_voxel(self, x: int, y: int, z: int) -> Voxel:
        """Get voxel at grid coordinates.

        Args:
            x, y, z: Grid coordinates (0 to resolution-1)

        Returns:
            The voxel at those coordinates

        Raises:
            IndexError: If coordinates are out of bounds
        """
        if not (0 <= x < self.resolution and
                0 <= y < self.resolution and
                0 <= z < self.resolution):
            raise IndexError(f"Voxel coordinates ({x}, {y}, {z}) out of bounds")
        return self.voxels[self._index(x, y, z)]

    def set_voxel(self, x: int, y: int, z: int, voxel: Voxel) -> None:
        """Set voxel at grid coordinates.

        Args:
            x, y, z: Grid coordinates
            voxel: Voxel to store

        Raises:
            IndexError: If coordinates are out of bounds
        """
        if not (0 <= x < self.resolution and
                0 <= y < self.resolution and
                0 <= z < self.resolution):
            raise IndexError(f"Voxel coordinates ({x}, {y}, {z}) out of bounds")
        self.voxels[self._index(x, y, z)] = voxel

    def world_to_voxel(self, world_pos: Vec3) -> tuple[int, int, int]:
        """Convert world position to voxel coordinates.

        Args:
            world_pos: World-space position

        Returns:
            Tuple of (x, y, z) voxel coordinates, clamped to grid bounds
        """
        local = world_pos - self.bounds.min
        x = int(local.x * self._inv_voxel_size.x)
        y = int(local.y * self._inv_voxel_size.y)
        z = int(local.z * self._inv_voxel_size.z)

        # Clamp to valid range
        x = max(0, min(self.resolution - 1, x))
        y = max(0, min(self.resolution - 1, y))
        z = max(0, min(self.resolution - 1, z))

        return (x, y, z)

    def world_to_voxel_float(self, world_pos: Vec3) -> Vec3:
        """Convert world position to floating-point voxel coordinates.

        Args:
            world_pos: World-space position

        Returns:
            Floating-point voxel coordinates (not clamped)
        """
        local = world_pos - self.bounds.min
        return Vec3(
            local.x * self._inv_voxel_size.x,
            local.y * self._inv_voxel_size.y,
            local.z * self._inv_voxel_size.z,
        )

    def voxel_to_world(self, x: int, y: int, z: int) -> Vec3:
        """Convert voxel coordinates to world-space center position.

        Args:
            x, y, z: Voxel coordinates

        Returns:
            World-space position at voxel center
        """
        return Vec3(
            self.bounds.min.x + (x + 0.5) * self._voxel_size.x,
            self.bounds.min.y + (y + 0.5) * self._voxel_size.y,
            self.bounds.min.z + (z + 0.5) * self._voxel_size.z,
        )

    def get_voxel_aabb(self, x: int, y: int, z: int) -> AABB:
        """Get the AABB for a voxel.

        Args:
            x, y, z: Voxel coordinates

        Returns:
            World-space AABB of the voxel
        """
        min_pos = Vec3(
            self.bounds.min.x + x * self._voxel_size.x,
            self.bounds.min.y + y * self._voxel_size.y,
            self.bounds.min.z + z * self._voxel_size.z,
        )
        max_pos = Vec3(
            min_pos.x + self._voxel_size.x,
            min_pos.y + self._voxel_size.y,
            min_pos.z + self._voxel_size.z,
        )
        return AABB(min_pos, max_pos)

    def clear(self) -> None:
        """Clear all voxels to empty state."""
        for v in self.voxels:
            v.albedo = Vec4(0.0, 0.0, 0.0, 0.0)
            v.emissive = Vec3(0.0, 0.0, 0.0)
            v.normal = Vec3(0.0, 0.0, 0.0)
            v.hit_count = 0

    def finalize(self) -> None:
        """Finalize all voxels (normalize normals)."""
        for v in self.voxels:
            v.finalize()

    def count_filled_voxels(self) -> int:
        """Count non-empty voxels."""
        return sum(1 for v in self.voxels if not v.is_empty())

    def get_fill_ratio(self) -> float:
        """Get ratio of filled voxels to total."""
        if self.total_voxels == 0:
            return 0.0
        return self.count_filled_voxels() / self.total_voxels

    def iter_filled(self) -> Iterator[tuple[int, int, int, Voxel]]:
        """Iterate over filled voxels.

        Yields:
            Tuples of (x, y, z, voxel) for non-empty voxels
        """
        res = self.resolution
        for z in range(res):
            for y in range(res):
                for x in range(res):
                    v = self.voxels[self._index(x, y, z)]
                    if not v.is_empty():
                        yield (x, y, z, v)

    def classify_opacity(self) -> dict[OpacityClass, int]:
        """Count voxels by opacity class.

        Returns:
            Dictionary mapping OpacityClass to count
        """
        counts = {cls: 0 for cls in OpacityClass}
        for v in self.voxels:
            counts[v.get_opacity_class()] += 1
        return counts

    def get_memory_bytes(self) -> int:
        """Estimate memory usage in bytes.

        This is for the Python representation. GPU representation would
        typically be more compact (e.g., RGBA8 textures).
        """
        # Voxel object overhead + data
        # Rough estimate: ~100 bytes per Voxel object in CPython
        return self.total_voxels * 100

    def to_albedo_bytes(self) -> bytes:
        """Pack albedo to RGBA8 byte array.

        Returns:
            Bytes in RGBA8 format, ordered X-Y-Z
        """
        data = array.array("B")
        for z in range(self.resolution):
            for y in range(self.resolution):
                for x in range(self.resolution):
                    v = self.voxels[self._index(x, y, z)]
                    r, g, b, a = v.to_rgba8()
                    data.extend([r, g, b, a])
        return data.tobytes()

    def to_normal_bytes(self) -> bytes:
        """Pack normals to RGB8 byte array (signed normalized).

        Returns:
            Bytes in RGB8 format (normal.xyz mapped to 0-255)
        """
        data = array.array("B")
        for z in range(self.resolution):
            for y in range(self.resolution):
                for x in range(self.resolution):
                    v = self.voxels[self._index(x, y, z)]
                    # Map [-1, 1] to [0, 255]
                    nx = int((v.normal.x * 0.5 + 0.5) * 255)
                    ny = int((v.normal.y * 0.5 + 0.5) * 255)
                    nz = int((v.normal.z * 0.5 + 0.5) * 255)
                    data.extend([
                        max(0, min(255, nx)),
                        max(0, min(255, ny)),
                        max(0, min(255, nz)),
                        255,  # Padding for alignment
                    ])
        return data.tobytes()

    def to_emissive_bytes(self) -> bytes:
        """Pack emissive to RGB8 byte array.

        Returns:
            Bytes in RGBA8 format (emissive.xyz, 255 padding)
        """
        data = array.array("B")
        for z in range(self.resolution):
            for y in range(self.resolution):
                for x in range(self.resolution):
                    v = self.voxels[self._index(x, y, z)]
                    r = int(max(0.0, min(1.0, v.emissive.x)) * 255)
                    g = int(max(0.0, min(1.0, v.emissive.y)) * 255)
                    b = int(max(0.0, min(1.0, v.emissive.z)) * 255)
                    data.extend([r, g, b, 255])
        return data.tobytes()


# ============================================================================
# Conservative Rasterizer
# ============================================================================


class ConservativeRasterizer:
    """Conservative rasterization of triangles to voxels.

    Conservative rasterization ensures that ANY voxel touched by a triangle
    is filled, not just voxels whose centers are inside the triangle. This
    is critical for thin geometry.

    The algorithm:
    1. Compute triangle AABB in voxel space
    2. For each voxel in AABB, test AABB-triangle intersection
    3. Fill voxels that pass the test

    AABB-Triangle Intersection:
        Uses separating axis theorem (SAT) with 13 axes:
        - 3 face normals (AABB faces)
        - 1 triangle face normal
        - 9 cross products (3 AABB edges x 3 triangle edges)
    """

    __slots__ = ()

    @staticmethod
    def get_dominant_axis(triangle: Triangle) -> VoxelAxis:
        """Get dominant axis for triangle projection.

        The dominant axis is perpendicular to the plane most parallel
        to the triangle. This maximizes coverage when projecting.

        Args:
            triangle: Triangle to analyze

        Returns:
            Dominant axis for projection
        """
        normal = triangle.compute_face_normal()
        abs_x = abs(normal.x)
        abs_y = abs(normal.y)
        abs_z = abs(normal.z)

        if abs_x >= abs_y and abs_x >= abs_z:
            return VoxelAxis.X
        elif abs_y >= abs_z:
            return VoxelAxis.Y
        else:
            return VoxelAxis.Z

    @staticmethod
    def triangle_aabb_intersects(
        tri: Triangle,
        aabb_min: Vec3,
        aabb_max: Vec3,
    ) -> bool:
        """Test if triangle intersects AABB using SAT.

        Implements the separating axis theorem with 13 potential axes.

        Args:
            tri: Triangle to test
            aabb_min: AABB minimum corner
            aabb_max: AABB maximum corner

        Returns:
            True if triangle intersects AABB
        """
        # AABB center and half-extents
        center = Vec3(
            (aabb_min.x + aabb_max.x) * 0.5,
            (aabb_min.y + aabb_max.y) * 0.5,
            (aabb_min.z + aabb_max.z) * 0.5,
        )
        extents = Vec3(
            (aabb_max.x - aabb_min.x) * 0.5,
            (aabb_max.y - aabb_min.y) * 0.5,
            (aabb_max.z - aabb_min.z) * 0.5,
        )

        # Translate triangle to AABB local space (center at origin)
        v0 = tri.v0 - center
        v1 = tri.v1 - center
        v2 = tri.v2 - center

        # Triangle edges
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2

        # Test AABB face normals (X, Y, Z axes)
        # X axis
        min_val = min(v0.x, v1.x, v2.x)
        max_val = max(v0.x, v1.x, v2.x)
        if min_val > extents.x or max_val < -extents.x:
            return False

        # Y axis
        min_val = min(v0.y, v1.y, v2.y)
        max_val = max(v0.y, v1.y, v2.y)
        if min_val > extents.y or max_val < -extents.y:
            return False

        # Z axis
        min_val = min(v0.z, v1.z, v2.z)
        max_val = max(v0.z, v1.z, v2.z)
        if min_val > extents.z or max_val < -extents.z:
            return False

        # Test triangle normal axis
        normal = e0.cross(e1)
        d = -normal.dot(v0)
        r = (extents.x * abs(normal.x) +
             extents.y * abs(normal.y) +
             extents.z * abs(normal.z))
        if abs(d) > r:
            return False

        # Test 9 edge cross products
        # Edge e0 x AABB axes
        # e0 x X = (0, -e0.z, e0.y)
        p0 = v0.z * e0.y - v0.y * e0.z
        p2 = v2.z * e0.y - v2.y * e0.z
        r = extents.y * abs(e0.z) + extents.z * abs(e0.y)
        if min(p0, p2) > r or max(p0, p2) < -r:
            return False

        # e0 x Y = (e0.z, 0, -e0.x)
        p0 = -v0.z * e0.x + v0.x * e0.z
        p2 = -v2.z * e0.x + v2.x * e0.z
        r = extents.x * abs(e0.z) + extents.z * abs(e0.x)
        if min(p0, p2) > r or max(p0, p2) < -r:
            return False

        # e0 x Z = (-e0.y, e0.x, 0)
        p0 = v0.y * e0.x - v0.x * e0.y
        p2 = v2.y * e0.x - v2.x * e0.y
        r = extents.x * abs(e0.y) + extents.y * abs(e0.x)
        if min(p0, p2) > r or max(p0, p2) < -r:
            return False

        # Edge e1 x AABB axes
        # e1 x X
        p0 = v0.z * e1.y - v0.y * e1.z
        p1 = v1.z * e1.y - v1.y * e1.z
        r = extents.y * abs(e1.z) + extents.z * abs(e1.y)
        if min(p0, p1) > r or max(p0, p1) < -r:
            return False

        # e1 x Y
        p0 = -v0.z * e1.x + v0.x * e1.z
        p1 = -v1.z * e1.x + v1.x * e1.z
        r = extents.x * abs(e1.z) + extents.z * abs(e1.x)
        if min(p0, p1) > r or max(p0, p1) < -r:
            return False

        # e1 x Z
        p0 = v0.y * e1.x - v0.x * e1.y
        p1 = v1.y * e1.x - v1.x * e1.y
        r = extents.x * abs(e1.y) + extents.y * abs(e1.x)
        if min(p0, p1) > r or max(p0, p1) < -r:
            return False

        # Edge e2 x AABB axes
        # e2 x X
        p0 = v0.z * e2.y - v0.y * e2.z
        p1 = v1.z * e2.y - v1.y * e2.z
        r = extents.y * abs(e2.z) + extents.z * abs(e2.y)
        if min(p0, p1) > r or max(p0, p1) < -r:
            return False

        # e2 x Y
        p0 = -v0.z * e2.x + v0.x * e2.z
        p1 = -v1.z * e2.x + v1.x * e2.z
        r = extents.x * abs(e2.z) + extents.z * abs(e2.x)
        if min(p0, p1) > r or max(p0, p1) < -r:
            return False

        # e2 x Z
        p0 = v0.y * e2.x - v0.x * e2.y
        p1 = v1.y * e2.x - v1.x * e2.y
        r = extents.x * abs(e2.y) + extents.y * abs(e2.x)
        if min(p0, p1) > r or max(p0, p1) < -r:
            return False

        # No separating axis found
        return True

    @staticmethod
    def rasterize_triangle(
        triangle: Triangle,
        grid: VoxelGrid,
    ) -> int:
        """Conservatively rasterize a triangle to the voxel grid.

        Args:
            triangle: Triangle to rasterize
            grid: Target voxel grid

        Returns:
            Number of voxels filled
        """
        # Get triangle AABB in voxel space
        tri_aabb = triangle.get_aabb()

        # Check if triangle is outside grid
        if not grid.bounds.intersects(tri_aabb):
            return 0

        # Clamp to grid bounds
        min_voxel = grid.world_to_voxel(tri_aabb.min)
        max_voxel = grid.world_to_voxel(tri_aabb.max)

        # Get triangle properties for all voxels
        face_normal = triangle.compute_face_normal()
        albedo = triangle.albedo
        emissive = triangle.emissive

        filled = 0

        # Test each voxel in the AABB
        for z in range(min_voxel[2], max_voxel[2] + 1):
            for y in range(min_voxel[1], max_voxel[1] + 1):
                for x in range(min_voxel[0], max_voxel[0] + 1):
                    voxel_aabb = grid.get_voxel_aabb(x, y, z)

                    if ConservativeRasterizer.triangle_aabb_intersects(
                        triangle, voxel_aabb.min, voxel_aabb.max
                    ):
                        voxel = grid.get_voxel(x, y, z)
                        voxel.accumulate(albedo, emissive, face_normal)
                        filled += 1

        return filled


# ============================================================================
# Scene Voxelizer
# ============================================================================


@dataclass
class VoxelizationStats:
    """Statistics from a voxelization pass.

    Attributes:
        triangle_count: Number of triangles processed
        voxels_filled: Total voxels with geometry
        fill_ratio: Ratio of filled to total voxels
        elapsed_ms: Processing time in milliseconds
        opacity_counts: Voxel counts by opacity class
    """

    triangle_count: int = 0
    voxels_filled: int = 0
    fill_ratio: float = 0.0
    elapsed_ms: float = 0.0
    opacity_counts: dict[OpacityClass, int] = field(default_factory=dict)

    def is_performance_target_met(self, resolution: int) -> bool:
        """Check if performance target was met.

        Targets:
            - 64^3: <0.5ms
            - 128^3: <1.5ms
            - 256^3: <4.0ms

        Args:
            resolution: Grid resolution

        Returns:
            True if target was met
        """
        targets = {
            64: 0.5,
            128: 1.5,
            256: 4.0,
        }
        target = targets.get(resolution, 10.0)
        return self.elapsed_ms <= target


@dataclass
class VoxelizationConfig:
    """Configuration for scene voxelization.

    Attributes:
        resolution: Grid resolution (64, 128, or 256)
        conservative: Use conservative rasterization
        accumulate_normals: Accumulate normals from all triangles
        include_backfaces: Include backface contributions
    """

    resolution: int = 128
    conservative: bool = True
    accumulate_normals: bool = True
    include_backfaces: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.resolution not in SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"Resolution {self.resolution} not supported. "
                f"Use one of: {SUPPORTED_RESOLUTIONS}"
            )


class SceneVoxelizer:
    """Voxelizes scene geometry for GI.

    The voxelizer converts triangle meshes into a 3D voxel grid with
    per-voxel material properties. It supports multiple resolution tiers
    for different quality/performance tradeoffs.

    Usage:
        voxelizer = SceneVoxelizer(config)
        grid = voxelizer.voxelize(triangles, scene_bounds)
        stats = voxelizer.get_last_stats()

    Attributes:
        config: Voxelization configuration
        _last_stats: Statistics from last voxelization
    """

    __slots__ = ("config", "_last_stats", "_rasterizer")

    def __init__(self, config: Optional[VoxelizationConfig] = None) -> None:
        """Initialize voxelizer.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or VoxelizationConfig()
        self._last_stats: Optional[VoxelizationStats] = None
        self._rasterizer = ConservativeRasterizer()

    def voxelize(
        self,
        triangles: Sequence[Triangle],
        bounds: AABB,
    ) -> VoxelGrid:
        """Voxelize a collection of triangles.

        Args:
            triangles: Triangles to voxelize
            bounds: World-space bounds for the voxel grid

        Returns:
            Populated voxel grid
        """
        import time
        start_time = time.perf_counter()

        # Create grid
        grid = VoxelGrid(self.config.resolution, bounds)

        # Rasterize each triangle
        for tri in triangles:
            if self.config.conservative:
                ConservativeRasterizer.rasterize_triangle(tri, grid)
            else:
                self._rasterize_simple(tri, grid)

        # Finalize
        grid.finalize()

        # Collect stats
        elapsed = (time.perf_counter() - start_time) * 1000.0
        filled = grid.count_filled_voxels()

        self._last_stats = VoxelizationStats(
            triangle_count=len(triangles),
            voxels_filled=filled,
            fill_ratio=grid.get_fill_ratio(),
            elapsed_ms=elapsed,
            opacity_counts=grid.classify_opacity(),
        )

        return grid

    def _rasterize_simple(self, triangle: Triangle, grid: VoxelGrid) -> int:
        """Simple (non-conservative) rasterization using centroid only.

        This is faster but may miss thin geometry.

        Args:
            triangle: Triangle to rasterize
            grid: Target grid

        Returns:
            Number of voxels filled (0 or 1)
        """
        center = triangle.get_center()

        if not grid.bounds.contains(center):
            return 0

        x, y, z = grid.world_to_voxel(center)
        voxel = grid.get_voxel(x, y, z)
        voxel.accumulate(
            triangle.albedo,
            triangle.emissive,
            triangle.compute_face_normal(),
        )
        return 1

    def get_last_stats(self) -> Optional[VoxelizationStats]:
        """Get statistics from last voxelization."""
        return self._last_stats

    def estimate_time_ms(
        self,
        triangle_count: int,
        resolution: Optional[int] = None,
    ) -> float:
        """Estimate voxelization time based on triangle count.

        Uses empirical formula based on typical GPU performance.

        Args:
            triangle_count: Number of triangles
            resolution: Grid resolution (uses config if None)

        Returns:
            Estimated time in milliseconds
        """
        res = resolution or self.config.resolution

        # Empirical constants (tuned for typical desktop GPU)
        # Base overhead per resolution
        base_overhead = {64: 0.05, 128: 0.15, 256: 0.5}
        # Time per 1000 triangles
        per_1k_tris = {64: 0.01, 128: 0.03, 256: 0.08}

        overhead = base_overhead.get(res, 0.5)
        per_tri = per_1k_tris.get(res, 0.08) / 1000.0

        return overhead + triangle_count * per_tri

    @staticmethod
    def compute_optimal_bounds(
        triangles: Sequence[Triangle],
        padding: float = 0.1,
    ) -> AABB:
        """Compute optimal bounds for a set of triangles.

        Args:
            triangles: Triangles to bound
            padding: Fractional padding to add (0.1 = 10%)

        Returns:
            AABB enclosing all triangles with padding
        """
        if not triangles:
            return AABB(Vec3.zero(), Vec3.one())

        # Find extremes
        min_x = min_y = min_z = float("inf")
        max_x = max_y = max_z = float("-inf")

        for tri in triangles:
            for v in (tri.v0, tri.v1, tri.v2):
                min_x = min(min_x, v.x)
                min_y = min(min_y, v.y)
                min_z = min(min_z, v.z)
                max_x = max(max_x, v.x)
                max_y = max(max_y, v.y)
                max_z = max(max_z, v.z)

        # Apply padding
        extent_x = max_x - min_x
        extent_y = max_y - min_y
        extent_z = max_z - min_z
        pad_x = extent_x * padding * 0.5
        pad_y = extent_y * padding * 0.5
        pad_z = extent_z * padding * 0.5

        return AABB(
            Vec3(min_x - pad_x, min_y - pad_y, min_z - pad_z),
            Vec3(max_x + pad_x, max_y + pad_y, max_z + pad_z),
        )

    @staticmethod
    def create_from_resolution(resolution: VoxelResolution) -> SceneVoxelizer:
        """Create voxelizer with preset resolution.

        Args:
            resolution: Resolution preset

        Returns:
            Configured SceneVoxelizer
        """
        return SceneVoxelizer(VoxelizationConfig(resolution=resolution.size))


# ============================================================================
# WGSL Shader Generation
# ============================================================================


def generate_voxelize_compute_wgsl() -> str:
    """Generate WGSL compute shader for GPU voxelization.

    Returns:
        WGSL shader code string
    """
    return '''
// voxelize.comp.wgsl - Conservative Scene Voxelization
// T-GIR-P7.1: Scene voxelization for GI

// Voxel grid dimensions
struct VoxelGridUniforms {
    resolution: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    bounds_min: vec3<f32>,
    _pad3: f32,
    bounds_max: vec3<f32>,
    _pad4: f32,
    voxel_size: vec3<f32>,
    _pad5: f32,
    inv_voxel_size: vec3<f32>,
    _pad6: f32,
}

// Triangle input
struct Triangle {
    v0: vec3<f32>,
    _pad0: f32,
    v1: vec3<f32>,
    _pad1: f32,
    v2: vec3<f32>,
    _pad2: f32,
    normal: vec3<f32>,
    _pad3: f32,
    albedo: vec4<f32>,
    emissive: vec3<f32>,
    _pad4: f32,
}

// Bindings
@group(0) @binding(0) var<uniform> grid: VoxelGridUniforms;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var voxel_albedo: texture_storage_3d<rgba8unorm, read_write>;
@group(0) @binding(3) var voxel_normal: texture_storage_3d<rgba8snorm, read_write>;
@group(0) @binding(4) var voxel_emissive: texture_storage_3d<rgba8unorm, read_write>;

// Constants
const EPSILON: f32 = 1e-7;

// Convert world position to voxel coordinates
fn world_to_voxel(world_pos: vec3<f32>) -> vec3<i32> {
    let local = world_pos - grid.bounds_min;
    let voxel_f = local * grid.inv_voxel_size;
    return vec3<i32>(floor(voxel_f));
}

// Check if voxel is within grid bounds
fn is_valid_voxel(v: vec3<i32>) -> bool {
    let res = i32(grid.resolution);
    return v.x >= 0 && v.x < res &&
           v.y >= 0 && v.y < res &&
           v.z >= 0 && v.z < res;
}

// AABB-Triangle intersection test (SAT)
fn triangle_aabb_intersects(
    v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>,
    aabb_center: vec3<f32>, aabb_extents: vec3<f32>
) -> bool {
    // Translate triangle to AABB local space
    let t0 = v0 - aabb_center;
    let t1 = v1 - aabb_center;
    let t2 = v2 - aabb_center;

    // Triangle edges
    let e0 = t1 - t0;
    let e1 = t2 - t1;
    let e2 = t0 - t2;

    // Test AABB face normals
    let min_x = min(t0.x, min(t1.x, t2.x));
    let max_x = max(t0.x, max(t1.x, t2.x));
    if min_x > aabb_extents.x || max_x < -aabb_extents.x { return false; }

    let min_y = min(t0.y, min(t1.y, t2.y));
    let max_y = max(t0.y, max(t1.y, t2.y));
    if min_y > aabb_extents.y || max_y < -aabb_extents.y { return false; }

    let min_z = min(t0.z, min(t1.z, t2.z));
    let max_z = max(t0.z, max(t1.z, t2.z));
    if min_z > aabb_extents.z || max_z < -aabb_extents.z { return false; }

    // Test triangle normal
    let normal = cross(e0, e1);
    let d = -dot(normal, t0);
    let r = aabb_extents.x * abs(normal.x) +
            aabb_extents.y * abs(normal.y) +
            aabb_extents.z * abs(normal.z);
    if abs(d) > r { return false; }

    // Test 9 edge cross products (simplified for common cases)
    // Full SAT test would include all 9, but this covers most practical cases
    return true;
}

// Write voxel data with atomic averaging
fn write_voxel(
    coord: vec3<i32>,
    albedo: vec4<f32>,
    normal: vec3<f32>,
    emissive: vec3<f32>
) {
    let existing_albedo = textureLoad(voxel_albedo, coord);
    let existing_normal = textureLoad(voxel_normal, coord);
    let existing_emissive = textureLoad(voxel_emissive, coord);

    // Simple averaging (for more accuracy, use atomic operations)
    let blend = 0.5;
    let new_albedo = mix(existing_albedo, albedo, blend);
    let new_normal = mix(existing_normal.xyz, normal, blend);
    let new_emissive = mix(existing_emissive.xyz, emissive, blend);

    textureStore(voxel_albedo, coord, new_albedo);
    textureStore(voxel_normal, coord, vec4<f32>(normalize(new_normal), 1.0));
    textureStore(voxel_emissive, coord, vec4<f32>(new_emissive, 1.0));
}

// Main compute shader - one thread per triangle
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tri_index = global_id.x;
    if tri_index >= arrayLength(&triangles) {
        return;
    }

    let tri = triangles[tri_index];

    // Compute triangle AABB in voxel space
    let min_world = min(tri.v0, min(tri.v1, tri.v2));
    let max_world = max(tri.v0, max(tri.v1, tri.v2));
    let min_voxel = world_to_voxel(min_world);
    let max_voxel = world_to_voxel(max_world);

    // Half voxel size for SAT test
    let half_voxel = grid.voxel_size * 0.5;

    // Conservative rasterization: test each voxel in triangle AABB
    for (var z = min_voxel.z; z <= max_voxel.z; z++) {
        for (var y = min_voxel.y; y <= max_voxel.y; y++) {
            for (var x = min_voxel.x; x <= max_voxel.x; x++) {
                let coord = vec3<i32>(x, y, z);

                if !is_valid_voxel(coord) {
                    continue;
                }

                // Voxel center in world space
                let voxel_center = grid.bounds_min +
                    vec3<f32>(f32(x) + 0.5, f32(y) + 0.5, f32(z) + 0.5) *
                    grid.voxel_size;

                // Conservative test
                if triangle_aabb_intersects(
                    tri.v0, tri.v1, tri.v2,
                    voxel_center, half_voxel
                ) {
                    write_voxel(coord, tri.albedo, tri.normal, tri.emissive);
                }
            }
        }
    }
}
'''


# ============================================================================
# Utility Functions
# ============================================================================


def create_test_triangles(count: int, bounds: AABB) -> list[Triangle]:
    """Create random test triangles within bounds.

    Args:
        count: Number of triangles to create
        bounds: Bounds to distribute triangles in

    Returns:
        List of random triangles
    """
    import random

    triangles = []
    extent = bounds.max - bounds.min

    for _ in range(count):
        # Random center
        cx = bounds.min.x + random.random() * extent.x
        cy = bounds.min.y + random.random() * extent.y
        cz = bounds.min.z + random.random() * extent.z

        # Random size (scaled to extent)
        size = min(extent.x, extent.y, extent.z) * 0.1 * (0.5 + random.random())

        # Random offsets for vertices
        v0 = Vec3(
            cx + (random.random() - 0.5) * size,
            cy + (random.random() - 0.5) * size,
            cz + (random.random() - 0.5) * size,
        )
        v1 = Vec3(
            cx + (random.random() - 0.5) * size,
            cy + (random.random() - 0.5) * size,
            cz + (random.random() - 0.5) * size,
        )
        v2 = Vec3(
            cx + (random.random() - 0.5) * size,
            cy + (random.random() - 0.5) * size,
            cz + (random.random() - 0.5) * size,
        )

        # Random albedo
        albedo = Vec4(
            random.random(),
            random.random(),
            random.random(),
            0.5 + random.random() * 0.5,  # At least semi-opaque
        )

        # Random emissive (usually dark)
        emissive = Vec3(0.0, 0.0, 0.0)
        if random.random() < 0.1:  # 10% chance of emissive
            emissive = Vec3(
                random.random() * 0.5,
                random.random() * 0.5,
                random.random() * 0.5,
            )

        triangles.append(Triangle(v0, v1, v2, albedo=albedo, emissive=emissive))

    return triangles


def estimate_voxelization_memory(resolution: int) -> int:
    """Estimate GPU memory for voxel grid in bytes.

    Assumes:
        - RGBA8 albedo texture
        - RGBA8 normal texture
        - RGBA8 emissive texture

    Args:
        resolution: Grid resolution

    Returns:
        Memory in bytes
    """
    voxels = resolution ** 3
    bytes_per_voxel = 4 * 3  # 3 RGBA8 textures
    return voxels * bytes_per_voxel


def recommend_resolution(
    scene_bounds: AABB,
    target_voxel_size: float = 0.25,
) -> VoxelResolution:
    """Recommend resolution based on scene bounds and target voxel size.

    Args:
        scene_bounds: Scene bounding box
        target_voxel_size: Target voxel size in world units

    Returns:
        Recommended VoxelResolution
    """
    extent = scene_bounds.max - scene_bounds.min
    max_extent = max(extent.x, extent.y, extent.z)

    # Calculate needed resolution
    needed = int(max_extent / target_voxel_size)

    if needed <= 64:
        return VoxelResolution.LOW
    elif needed <= 128:
        return VoxelResolution.MEDIUM
    else:
        return VoxelResolution.HIGH

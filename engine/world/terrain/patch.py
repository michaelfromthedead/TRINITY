"""
Terrain patch component for the game engine World Layer.

This module provides terrain patches with LOD support:
- Grid-based terrain organization
- LOD selection based on camera distance
- Neighbor management for seamless stitching
- World coordinate conversion
- Error metrics for adaptive LOD

Coordinate System:
- Patches are organized in a grid with (patch_x, patch_y) indices
- patch_y corresponds to the Z-axis in world space
- World position is calculated from patch index and heightfield size
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import math

from .heightfield import Heightfield
from .constants import (
    DEFAULT_LOD_LEVELS,
    DEFAULT_LOD_DISTANCES,
    MIN_LOD_LEVELS,
)


# Type alias for AABB (Axis-Aligned Bounding Box)
# Format: (min_x, min_y, min_z, max_x, max_y, max_z)
AABB = Tuple[float, float, float, float, float, float]


@dataclass
class TerrainPatch:
    """Single terrain patch with LOD support.

    A terrain patch represents a portion of the terrain grid. It contains
    a heightfield for height data and supports multiple LOD levels for
    efficient rendering at different distances.

    Attributes:
        patch_x: X index in the patch grid (column)
        patch_y: Y index in the patch grid (row, corresponds to Z-axis)
        heightfield: The height data for this patch
        current_lod: Currently selected LOD level (0 = highest detail)
        lod_levels: Total number of LOD levels
        lod_distances: Distance thresholds for LOD transitions
        _mesh_cache: Runtime cache for generated mesh data per LOD
        _collision_data: Runtime cache for collision geometry
        _neighbors: References to adjacent patches for stitching
    """
    patch_x: int = 0  # Grid position (column)
    patch_y: int = 0  # Grid position (row, Z-axis in world)
    heightfield: Optional[Heightfield] = None

    # LOD state
    current_lod: int = 0
    lod_levels: int = DEFAULT_LOD_LEVELS
    lod_distances: Tuple[float, ...] = DEFAULT_LOD_DISTANCES

    # Runtime cache (transient, not serialized)
    _mesh_cache: Dict[int, Any] = field(default_factory=dict)
    _collision_data: Optional[Any] = None
    _neighbors: Dict[str, 'TerrainPatch'] = field(default_factory=dict)

    def __post_init__(self):
        """Validate patch configuration."""
        if self.lod_levels < MIN_LOD_LEVELS:
            raise ValueError("lod_levels must be >= 1")
        if len(self.lod_distances) != self.lod_levels:
            raise ValueError(
                f"lod_distances length ({len(self.lod_distances)}) must match "
                f"lod_levels ({self.lod_levels})"
            )
        # Validate distances are ascending
        for i in range(1, len(self.lod_distances)):
            if self.lod_distances[i] <= self.lod_distances[i - 1]:
                raise ValueError("lod_distances must be strictly ascending")

    def get_world_bounds(self) -> AABB:
        """Get world-space axis-aligned bounding box.

        The bounds encompass the entire patch including the full height range.

        Returns:
            AABB tuple (min_x, min_y, min_z, max_x, max_y, max_z)
        """
        if self.heightfield is None:
            # Return a unit-size box at patch origin
            base_x = float(self.patch_x)
            base_z = float(self.patch_y)
            return (base_x, 0.0, base_z, base_x + 1.0, 1.0, base_z + 1.0)

        # Calculate world position
        world_size_x, world_size_z = self.heightfield.get_world_size()
        base_x = self.patch_x * world_size_x
        base_z = self.patch_y * world_size_z

        # Get height bounds
        min_height, max_height = self.heightfield.get_bounds()

        return (
            base_x,
            min_height,
            base_z,
            base_x + world_size_x,
            max_height,
            base_z + world_size_z
        )

    def select_lod(self, camera_distance: float) -> int:
        """Select appropriate LOD level based on camera distance.

        Args:
            camera_distance: Distance from camera to patch center

        Returns:
            LOD level (0 = highest detail, increases with distance)
        """
        if camera_distance < 0:
            camera_distance = 0

        # Find the first threshold that exceeds the camera distance
        for level, threshold in enumerate(self.lod_distances):
            if camera_distance < threshold:
                self.current_lod = level
                return level

        # Beyond all thresholds - use lowest detail
        self.current_lod = self.lod_levels - 1
        return self.current_lod

    def get_height_at_world(self, world_x: float, world_z: float) -> Optional[float]:
        """Get interpolated height at world position.

        Args:
            world_x: X position in world space
            world_z: Z position in world space

        Returns:
            Interpolated height, or None if position is outside patch
            or heightfield is not set.
        """
        if self.heightfield is None:
            return None

        # Convert world to local patch coordinates
        local_x, local_z = self._world_to_local(world_x, world_z)

        # Check if position is within patch
        world_size_x, world_size_z = self.heightfield.get_world_size()
        if local_x < 0 or local_x > world_size_x:
            return None
        if local_z < 0 or local_z > world_size_z:
            return None

        return self.heightfield.get_height_at(local_x, local_z)

    def get_normal_at_world(
        self,
        world_x: float,
        world_z: float
    ) -> Optional[Tuple[float, float, float]]:
        """Get surface normal at world position.

        Args:
            world_x: X position in world space
            world_z: Z position in world space

        Returns:
            Normal vector (nx, ny, nz), or None if position is outside patch
            or heightfield is not set.
        """
        if self.heightfield is None:
            return None

        # Convert world to local patch coordinates
        local_x, local_z = self._world_to_local(world_x, world_z)

        # Check if position is within patch
        world_size_x, world_size_z = self.heightfield.get_world_size()
        if local_x < 0 or local_x > world_size_x:
            return None
        if local_z < 0 or local_z > world_size_z:
            return None

        return self.heightfield.get_normal_at(local_x, local_z)

    def _world_to_local(self, world_x: float, world_z: float) -> Tuple[float, float]:
        """Convert world coordinates to local patch coordinates.

        Args:
            world_x: X position in world space
            world_z: Z position in world space

        Returns:
            Tuple (local_x, local_z) in patch-local coordinates
        """
        if self.heightfield is None:
            return (world_x - self.patch_x, world_z - self.patch_y)

        world_size_x, world_size_z = self.heightfield.get_world_size()
        base_x = self.patch_x * world_size_x
        base_z = self.patch_y * world_size_z

        return (world_x - base_x, world_z - base_z)

    def _local_to_world(self, local_x: float, local_z: float) -> Tuple[float, float]:
        """Convert local patch coordinates to world coordinates.

        Args:
            local_x: X position in patch-local space
            local_z: Z position in patch-local space

        Returns:
            Tuple (world_x, world_z) in world coordinates
        """
        if self.heightfield is None:
            return (self.patch_x + local_x, self.patch_y + local_z)

        world_size_x, world_size_z = self.heightfield.get_world_size()
        base_x = self.patch_x * world_size_x
        base_z = self.patch_y * world_size_z

        return (base_x + local_x, base_z + local_z)

    def get_center_world(self) -> Tuple[float, float, float]:
        """Get world-space center position of the patch.

        Returns:
            Tuple (x, y, z) of patch center in world space,
            where y is the average height.
        """
        bounds = self.get_world_bounds()
        center_x = (bounds[0] + bounds[3]) / 2.0
        center_y = (bounds[1] + bounds[4]) / 2.0
        center_z = (bounds[2] + bounds[5]) / 2.0
        return (center_x, center_y, center_z)

    def set_neighbor(self, direction: str, patch: Optional['TerrainPatch']) -> bool:
        """Set a neighboring patch for seamless stitching.

        Args:
            direction: One of 'north', 'south', 'east', 'west'
            patch: The neighboring patch, or None to clear

        Returns:
            True if direction is valid, False otherwise.
        """
        valid_directions = {'north', 'south', 'east', 'west'}
        if direction not in valid_directions:
            return False

        if patch is None:
            self._neighbors.pop(direction, None)
        else:
            self._neighbors[direction] = patch

        # Invalidate mesh cache when neighbors change
        self._mesh_cache.clear()
        return True

    def get_neighbor(self, direction: str) -> Optional['TerrainPatch']:
        """Get a neighboring patch.

        Args:
            direction: One of 'north', 'south', 'east', 'west'

        Returns:
            The neighboring patch, or None if not set.
        """
        return self._neighbors.get(direction)

    def invalidate_cache(self) -> None:
        """Clear all runtime caches.

        Call this when heightfield data changes to force mesh regeneration.
        """
        self._mesh_cache.clear()
        self._collision_data = None

    def get_error_metric(self, lod_level: int) -> float:
        """Calculate geometric error metric for a LOD level.

        The error metric represents the maximum vertical deviation between
        the LOD mesh and the full-resolution heightfield. Higher LOD levels
        (lower detail) have larger error values.

        Args:
            lod_level: The LOD level to calculate error for

        Returns:
            Error metric in world units. Returns 0.0 for LOD 0 (full detail).
        """
        if lod_level < 0 or lod_level >= self.lod_levels:
            return float('inf')

        if lod_level == 0:
            return 0.0

        if self.heightfield is None:
            return 0.0

        # Calculate step size for this LOD level
        # Each LOD level reduces resolution by factor of 2
        step = 1 << lod_level  # 2^lod_level

        res = self.heightfield.config.resolution
        max_error = 0.0

        # Sample points that would be skipped at this LOD level
        for z in range(0, res - 1, step):
            for x in range(0, res - 1, step):
                # Check intermediate points
                for dz in range(step):
                    for dx in range(step):
                        sample_x = x + dx
                        sample_z = z + dz

                        if sample_x >= res or sample_z >= res:
                            continue

                        # Get actual height
                        actual = self.heightfield.get_raw_height_at(sample_x, sample_z)
                        if actual is None:
                            continue

                        # Get interpolated height from LOD corners
                        corner_x0 = x
                        corner_z0 = z
                        corner_x1 = min(x + step, res - 1)
                        corner_z1 = min(z + step, res - 1)

                        h00 = self.heightfield.get_raw_height_at(corner_x0, corner_z0) or 0.0
                        h10 = self.heightfield.get_raw_height_at(corner_x1, corner_z0) or 0.0
                        h01 = self.heightfield.get_raw_height_at(corner_x0, corner_z1) or 0.0
                        h11 = self.heightfield.get_raw_height_at(corner_x1, corner_z1) or 0.0

                        # Bilinear interpolation
                        fx = dx / step if step > 0 else 0.0
                        fz = dz / step if step > 0 else 0.0

                        h0 = h00 * (1.0 - fx) + h10 * fx
                        h1 = h01 * (1.0 - fx) + h11 * fx
                        interpolated = h0 * (1.0 - fz) + h1 * fz

                        error = abs(actual - interpolated)
                        max_error = max(max_error, error)

        return max_error

    def get_lod_vertex_count(self, lod_level: int) -> int:
        """Get approximate vertex count for a LOD level.

        Args:
            lod_level: The LOD level

        Returns:
            Approximate number of vertices for mesh at this LOD.
        """
        if self.heightfield is None:
            return 0

        if lod_level < 0 or lod_level >= self.lod_levels:
            return 0

        res = self.heightfield.config.resolution
        step = 1 << lod_level
        lod_res = max(2, (res - 1) // step + 1)

        return lod_res * lod_res

    def contains_world_point(self, world_x: float, world_z: float) -> bool:
        """Check if a world point is within this patch's horizontal bounds.

        Args:
            world_x: X position in world space
            world_z: Z position in world space

        Returns:
            True if point is within patch bounds (ignoring height)
        """
        bounds = self.get_world_bounds()
        return (bounds[0] <= world_x <= bounds[3] and
                bounds[2] <= world_z <= bounds[5])

    def distance_to_point(self, world_x: float, world_y: float, world_z: float) -> float:
        """Calculate distance from a point to the nearest point on the patch.

        Args:
            world_x: X position in world space
            world_y: Y position in world space
            world_z: Z position in world space

        Returns:
            Distance to patch bounds, or 0.0 if point is inside.
        """
        bounds = self.get_world_bounds()

        # Calculate squared distance to AABB
        dx = max(bounds[0] - world_x, 0.0, world_x - bounds[3])
        dy = max(bounds[1] - world_y, 0.0, world_y - bounds[4])
        dz = max(bounds[2] - world_z, 0.0, world_z - bounds[5])

        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def __hash__(self) -> int:
        """Hash based on patch position."""
        return hash((self.patch_x, self.patch_y))

    def __eq__(self, other: object) -> bool:
        """Equality based on patch position."""
        if not isinstance(other, TerrainPatch):
            return False
        return self.patch_x == other.patch_x and self.patch_y == other.patch_y

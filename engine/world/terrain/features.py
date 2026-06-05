"""
Terrain special features including holes, splines, deformation, and collision.

Provides functionality for terrain holes (caves/tunnels), road and river
splines with terrain deformation, and collision detection with physical
material mapping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Protocol, Tuple
from typing import runtime_checkable

from engine.world.terrain.constants import (
    DEFAULT_HOLE_RADIUS,
    DEFAULT_SPLINE_SEGMENT_LENGTH,
)

if TYPE_CHECKING:
    from typing import TypeAlias


@runtime_checkable
class Heightfield(Protocol):
    """Protocol for heightfield data structures."""

    @property
    def width(self) -> int:
        """Width of the heightfield in samples."""
        ...

    @property
    def height(self) -> int:
        """Height of the heightfield in samples."""
        ...

    @property
    def sample_spacing(self) -> float:
        """World units between samples."""
        ...

    def get_height_at(self, x: int, z: int) -> float:
        """Get height at sample coordinates."""
        ...

    def set_height_at(self, x: int, z: int, height: float) -> None:
        """Set height at sample coordinates."""
        ...

    def world_to_sample(self, world_x: float, world_z: float) -> Tuple[int, int]:
        """Convert world coordinates to sample coordinates."""
        ...

    def sample_to_world(self, sample_x: int, sample_z: int) -> Tuple[float, float]:
        """Convert sample coordinates to world coordinates."""
        ...


@runtime_checkable
class WeightMap(Protocol):
    """Protocol for terrain material weight maps."""

    def get_dominant_layer_at(self, x: int, z: int) -> int:
        """Get the dominant material layer at a position."""
        ...


@dataclass
class TerrainHole:
    """Defines a hole in the terrain for caves/tunnels.

    Attributes:
        center_x: World X coordinate of hole center.
        center_z: World Z coordinate of hole center.
        radius: Radius of the circular hole.
        mask_resolution: Resolution of the visibility mask.
    """

    center_x: float = 0.0
    center_z: float = 0.0
    radius: float = 10.0
    mask_resolution: int = 32
    _mask: Optional[List[List[bool]]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate hole settings and generate mask."""
        if self.radius <= 0:
            raise ValueError("radius must be > 0")
        if self.mask_resolution < 2:
            raise ValueError("mask_resolution must be >= 2")
        self._generate_mask()

    def _generate_mask(self) -> None:
        """Generate the circular visibility mask."""
        self._mask = []
        cell_size = (self.radius * 2) / self.mask_resolution

        for mz in range(self.mask_resolution):
            row = []
            for mx in range(self.mask_resolution):
                # Calculate distance from center
                local_x = (mx + 0.5) * cell_size - self.radius
                local_z = (mz + 0.5) * cell_size - self.radius
                distance = math.sqrt(local_x * local_x + local_z * local_z)
                # True = visible (not a hole), False = hole
                row.append(distance > self.radius)
            self._mask.append(row)

    def is_visible_at(self, world_x: float, world_z: float) -> bool:
        """Check if terrain is visible at a world position.

        Args:
            world_x: World X coordinate.
            world_z: World Z coordinate.

        Returns:
            True if terrain is visible (not in hole).
        """
        # Check if point is within the hole's bounding box
        dx = world_x - self.center_x
        dz = world_z - self.center_z

        if abs(dx) > self.radius or abs(dz) > self.radius:
            return True  # Outside bounding box, visible

        # Sample from mask
        cell_size = (self.radius * 2) / self.mask_resolution
        mx = int((dx + self.radius) / cell_size)
        mz = int((dz + self.radius) / cell_size)

        mx = max(0, min(self.mask_resolution - 1, mx))
        mz = max(0, min(self.mask_resolution - 1, mz))

        if self._mask is not None:
            return self._mask[mz][mx]
        return True

    def set_custom_mask(self, mask: List[List[bool]]) -> None:
        """Set a custom visibility mask.

        Args:
            mask: 2D boolean array where True = visible, False = hole.
        """
        if len(mask) != self.mask_resolution:
            raise ValueError(
                f"mask height must be {self.mask_resolution}, got {len(mask)}"
            )
        for row in mask:
            if len(row) != self.mask_resolution:
                raise ValueError(
                    f"mask width must be {self.mask_resolution}, got {len(row)}"
                )
        self._mask = [list(row) for row in mask]

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (min_x, min_z, max_x, max_z)."""
        return (
            self.center_x - self.radius,
            self.center_z - self.radius,
            self.center_x + self.radius,
            self.center_z + self.radius,
        )


class TerrainHoleManager:
    """Manages multiple terrain holes."""

    def __init__(self) -> None:
        """Initialize the hole manager."""
        self._holes: List[TerrainHole] = []

    @property
    def holes(self) -> List[TerrainHole]:
        """Get list of holes (read-only copy)."""
        return list(self._holes)

    @property
    def hole_count(self) -> int:
        """Get number of holes."""
        return len(self._holes)

    def add_hole(self, hole: TerrainHole) -> int:
        """Add a terrain hole.

        Args:
            hole: The hole to add.

        Returns:
            Index of the added hole.
        """
        self._holes.append(hole)
        return len(self._holes) - 1

    def remove_hole(self, index: int) -> None:
        """Remove a hole by index.

        Args:
            index: Index of the hole to remove.
        """
        if not 0 <= index < len(self._holes):
            raise ValueError(f"index must be in range [0, {len(self._holes)})")
        self._holes.pop(index)

    def clear_holes(self) -> None:
        """Remove all holes."""
        self._holes.clear()

    def is_visible_at(self, world_x: float, world_z: float) -> bool:
        """Check if terrain is visible at a world position.

        Args:
            world_x: World X coordinate.
            world_z: World Z coordinate.

        Returns:
            True if terrain is visible (not in any hole).
        """
        for hole in self._holes:
            if not hole.is_visible_at(world_x, world_z):
                return False
        return True

    def get_visibility_mask(
        self,
        min_x: float,
        min_z: float,
        max_x: float,
        max_z: float,
        resolution: int,
    ) -> List[List[bool]]:
        """Generate a visibility mask for a region.

        Args:
            min_x: Minimum X coordinate.
            min_z: Minimum Z coordinate.
            max_x: Maximum X coordinate.
            max_z: Maximum Z coordinate.
            resolution: Resolution of the mask.

        Returns:
            2D boolean array where True = visible.
        """
        if resolution < 1:
            raise ValueError("resolution must be >= 1")

        mask = []
        cell_width = (max_x - min_x) / resolution
        cell_height = (max_z - min_z) / resolution

        for mz in range(resolution):
            row = []
            z = min_z + (mz + 0.5) * cell_height
            for mx in range(resolution):
                x = min_x + (mx + 0.5) * cell_width
                row.append(self.is_visible_at(x, z))
            mask.append(row)

        return mask


@dataclass
class SplinePoint:
    """A control point on a terrain spline.

    Attributes:
        position: (x, y, z) world position.
        tangent: (x, y, z) tangent direction (for control).
        width: Width of the spline at this point.
    """

    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    tangent: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    width: float = 10.0

    def __post_init__(self) -> None:
        """Validate point settings."""
        if self.width <= 0:
            raise ValueError("width must be > 0")


class TerrainSpline:
    """Base class for terrain splines (roads, rivers, etc.).

    Uses Catmull-Rom interpolation for smooth curves.
    """

    def __init__(self, points: Optional[List[SplinePoint]] = None) -> None:
        """Initialize the spline.

        Args:
            points: Initial control points.
        """
        self._points: List[SplinePoint] = list(points) if points else []
        self._cached_length: Optional[float] = None

    @property
    def points(self) -> List[SplinePoint]:
        """Get control points (read-only copy)."""
        return list(self._points)

    @property
    def point_count(self) -> int:
        """Get number of control points."""
        return len(self._points)

    def add_point(self, point: SplinePoint) -> None:
        """Add a control point.

        Args:
            point: The point to add.
        """
        self._points.append(point)
        self._cached_length = None

    def insert_point(self, index: int, point: SplinePoint) -> None:
        """Insert a control point at an index.

        Args:
            index: Index to insert at.
            point: The point to insert.
        """
        self._points.insert(index, point)
        self._cached_length = None

    def remove_point(self, index: int) -> None:
        """Remove a control point.

        Args:
            index: Index of the point to remove.
        """
        if not 0 <= index < len(self._points):
            raise ValueError(f"index must be in range [0, {len(self._points)})")
        self._points.pop(index)
        self._cached_length = None

    def set_point(self, index: int, point: SplinePoint) -> None:
        """Set a control point.

        Args:
            index: Index of the point to set.
            point: The new point.
        """
        if not 0 <= index < len(self._points):
            raise ValueError(f"index must be in range [0, {len(self._points)})")
        self._points[index] = point
        self._cached_length = None

    def evaluate(self, t: float) -> Tuple[float, float, float]:
        """Evaluate spline position at parameter t.

        Args:
            t: Parameter value (0 = start, 1 = end).

        Returns:
            (x, y, z) position on the spline.

        Raises:
            ValueError: If spline has fewer than 2 points.
        """
        if len(self._points) < 2:
            raise ValueError("Spline must have at least 2 points")

        t = max(0.0, min(1.0, t))

        # Find segment
        segment_count = len(self._points) - 1
        segment_t = t * segment_count
        segment_index = int(segment_t)
        local_t = segment_t - segment_index

        if segment_index >= segment_count:
            segment_index = segment_count - 1
            local_t = 1.0

        # Get control points for Catmull-Rom
        p0_idx = max(0, segment_index - 1)
        p1_idx = segment_index
        p2_idx = min(len(self._points) - 1, segment_index + 1)
        p3_idx = min(len(self._points) - 1, segment_index + 2)

        p0 = self._points[p0_idx].position
        p1 = self._points[p1_idx].position
        p2 = self._points[p2_idx].position
        p3 = self._points[p3_idx].position

        # Catmull-Rom interpolation
        return self._catmull_rom(p0, p1, p2, p3, local_t)

    def _catmull_rom(
        self,
        p0: Tuple[float, float, float],
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        p3: Tuple[float, float, float],
        t: float,
    ) -> Tuple[float, float, float]:
        """Catmull-Rom spline interpolation.

        Args:
            p0, p1, p2, p3: Control points.
            t: Local parameter (0-1).

        Returns:
            Interpolated position.
        """
        t2 = t * t
        t3 = t2 * t

        result = []
        for i in range(3):
            # Catmull-Rom coefficients
            a = -0.5 * p0[i] + 1.5 * p1[i] - 1.5 * p2[i] + 0.5 * p3[i]
            b = p0[i] - 2.5 * p1[i] + 2 * p2[i] - 0.5 * p3[i]
            c = -0.5 * p0[i] + 0.5 * p2[i]
            d = p1[i]

            result.append(a * t3 + b * t2 + c * t + d)

        return (result[0], result[1], result[2])

    def evaluate_tangent(self, t: float) -> Tuple[float, float, float]:
        """Evaluate spline tangent (derivative) at parameter t.

        Args:
            t: Parameter value (0 = start, 1 = end).

        Returns:
            Normalized (x, y, z) tangent direction.
        """
        if len(self._points) < 2:
            raise ValueError("Spline must have at least 2 points")

        # Numerical derivative with adaptive epsilon
        epsilon = 0.001
        t1 = max(0.0, t - epsilon)
        t2 = min(1.0, t + epsilon)

        # Ensure t1 and t2 are different to avoid zero-length tangent
        if t2 - t1 < 1e-6:
            # At endpoints, use one-sided derivative
            if t <= 0.5:
                t1 = 0.0
                t2 = min(1.0, 2 * epsilon)
            else:
                t1 = max(0.0, 1.0 - 2 * epsilon)
                t2 = 1.0

        p1 = self.evaluate(t1)
        p2 = self.evaluate(t2)

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]

        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length > 0:
            return (dx / length, dy / length, dz / length)

        # Fallback: use direction from first to last point
        first_pos = self._points[0].position
        last_pos = self._points[-1].position
        dx = last_pos[0] - first_pos[0]
        dy = last_pos[1] - first_pos[1]
        dz = last_pos[2] - first_pos[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length > 0:
            return (dx / length, dy / length, dz / length)

        # Final fallback for degenerate case (all points at same location)
        return (1.0, 0.0, 0.0)

    def evaluate_width(self, t: float) -> float:
        """Evaluate spline width at parameter t.

        Args:
            t: Parameter value (0 = start, 1 = end).

        Returns:
            Interpolated width.
        """
        if len(self._points) < 2:
            raise ValueError("Spline must have at least 2 points")

        t = max(0.0, min(1.0, t))

        # Linear interpolation of width
        segment_count = len(self._points) - 1
        segment_t = t * segment_count
        segment_index = int(segment_t)
        local_t = segment_t - segment_index

        if segment_index >= segment_count:
            return self._points[-1].width

        w1 = self._points[segment_index].width
        w2 = self._points[segment_index + 1].width

        return w1 + (w2 - w1) * local_t

    def get_length(self, segments: int = 100) -> float:
        """Calculate approximate spline length.

        Args:
            segments: Number of segments for approximation.

        Returns:
            Approximate length of the spline.
        """
        if self._cached_length is not None:
            return self._cached_length

        if len(self._points) < 2:
            return 0.0

        total_length = 0.0
        prev_point = self.evaluate(0.0)

        for i in range(1, segments + 1):
            t = i / segments
            curr_point = self.evaluate(t)
            dx = curr_point[0] - prev_point[0]
            dy = curr_point[1] - prev_point[1]
            dz = curr_point[2] - prev_point[2]
            total_length += math.sqrt(dx * dx + dy * dy + dz * dz)
            prev_point = curr_point

        self._cached_length = total_length
        return total_length

    def get_point_at_distance(self, distance: float) -> Tuple[float, float, float]:
        """Get point at a specific distance along the spline.

        Args:
            distance: Distance from start of spline.

        Returns:
            (x, y, z) position at that distance.
        """
        if len(self._points) < 2:
            raise ValueError("Spline must have at least 2 points")

        total_length = self.get_length()
        if total_length == 0:
            return self._points[0].position

        # Approximate t from distance
        target_dist = max(0.0, min(total_length, distance))
        t = target_dist / total_length

        # Refine with binary search
        current_dist = 0.0
        prev_point = self.evaluate(0.0)
        step = 0.01

        for i in range(1, 101):
            test_t = i * step
            curr_point = self.evaluate(test_t)
            dx = curr_point[0] - prev_point[0]
            dy = curr_point[1] - prev_point[1]
            dz = curr_point[2] - prev_point[2]
            segment_length = math.sqrt(dx * dx + dy * dy + dz * dz)

            if current_dist + segment_length >= target_dist:
                # Interpolate within this segment
                remaining = target_dist - current_dist
                local_t = remaining / segment_length if segment_length > 0 else 0
                return self.evaluate((i - 1) * step + local_t * step)

            current_dist += segment_length
            prev_point = curr_point

        return self.evaluate(1.0)


class RoadSpline(TerrainSpline):
    """A road spline that can deform terrain and provide surface info.

    Attributes:
        surface_material: Material ID for the road surface.
        bank_angle: Maximum banking angle in degrees for curves.
    """

    def __init__(
        self,
        points: Optional[List[SplinePoint]] = None,
        surface_material: str = "mat_asphalt",
        bank_angle: float = 5.0,
    ) -> None:
        """Initialize the road spline.

        Args:
            points: Initial control points.
            surface_material: Material ID for the road surface.
            bank_angle: Maximum banking angle in degrees.
        """
        super().__init__(points)
        if bank_angle < 0:
            raise ValueError("bank_angle must be >= 0")
        self.surface_material = surface_material
        self.bank_angle = bank_angle

    def deform_terrain(
        self,
        heightfield: Heightfield,
        depth: float = 0.1,
        blend_width: float = 5.0,
    ) -> None:
        """Deform terrain to create a flat road bed.

        Args:
            heightfield: The heightfield to modify.
            depth: How deep to cut the road into terrain.
            blend_width: Width of the blend zone at road edges.
        """
        if len(self._points) < 2:
            return

        if depth < 0:
            raise ValueError("depth must be >= 0")
        if blend_width < 0:
            raise ValueError("blend_width must be >= 0")

        # Sample along the spline
        segments = int(self.get_length() / (heightfield.sample_spacing * 0.5))
        segments = max(10, segments)

        for i in range(segments + 1):
            t = i / segments
            pos = self.evaluate(t)
            width = self.evaluate_width(t)
            tangent = self.evaluate_tangent(t)

            # Calculate perpendicular direction (in XZ plane)
            perp_x = -tangent[2]
            perp_z = tangent[0]
            perp_len = math.sqrt(perp_x * perp_x + perp_z * perp_z)
            if perp_len > 0:
                perp_x /= perp_len
                perp_z /= perp_len

            # Sample across the road width
            total_width = width + blend_width * 2
            samples_across = int(total_width / heightfield.sample_spacing) + 1

            for j in range(samples_across):
                offset = -total_width / 2 + j * heightfield.sample_spacing
                world_x = pos[0] + perp_x * offset
                world_z = pos[2] + perp_z * offset

                sx, sz = heightfield.world_to_sample(world_x, world_z)
                if 0 <= sx < heightfield.width and 0 <= sz < heightfield.height:
                    current_height = heightfield.get_height_at(sx, sz)
                    road_height = pos[1] - depth

                    # Calculate blend factor
                    dist_from_center = abs(offset)
                    if dist_from_center <= width / 2:
                        # On the road
                        new_height = road_height
                    elif dist_from_center <= width / 2 + blend_width:
                        # In blend zone
                        blend_t = (dist_from_center - width / 2) / blend_width
                        blend_factor = blend_t * blend_t * (3 - 2 * blend_t)  # Smoothstep
                        new_height = road_height + (current_height - road_height) * blend_factor
                    else:
                        continue

                    heightfield.set_height_at(sx, sz, new_height)

    def get_surface_height_at(
        self,
        world_x: float,
        world_z: float,
        heightfield: Heightfield,
    ) -> Optional[float]:
        """Get road surface height at a world position.

        Args:
            world_x: World X coordinate.
            world_z: World Z coordinate.
            heightfield: The heightfield for fallback.

        Returns:
            Road surface height or None if not on road.
        """
        if len(self._points) < 2:
            return None

        # Find closest point on spline
        min_dist = float("inf")
        closest_t = 0.0

        segments = 50
        for i in range(segments + 1):
            t = i / segments
            pos = self.evaluate(t)
            dx = pos[0] - world_x
            dz = pos[2] - world_z
            dist = math.sqrt(dx * dx + dz * dz)
            if dist < min_dist:
                min_dist = dist
                closest_t = t

        width = self.evaluate_width(closest_t)
        if min_dist <= width / 2:
            pos = self.evaluate(closest_t)
            return pos[1]

        return None


class RiverSpline(TerrainSpline):
    """A river spline that can carve channels in terrain.

    Attributes:
        water_level: Height of the water surface.
        flow_speed: Speed of water flow along the river.
    """

    def __init__(
        self,
        points: Optional[List[SplinePoint]] = None,
        water_level: float = 0.0,
        flow_speed: float = 1.0,
    ) -> None:
        """Initialize the river spline.

        Args:
            points: Initial control points.
            water_level: Height of the water surface.
            flow_speed: Speed of water flow.
        """
        super().__init__(points)
        if flow_speed < 0:
            raise ValueError("flow_speed must be >= 0")
        self.water_level = water_level
        self.flow_speed = flow_speed

    def carve_channel(
        self,
        heightfield: Heightfield,
        depth: float = 2.0,
        bank_slope: float = 45.0,
    ) -> None:
        """Carve a river channel into the terrain.

        Args:
            heightfield: The heightfield to modify.
            depth: Depth of the river channel below water_level.
            bank_slope: Angle of the river banks in degrees.
        """
        if len(self._points) < 2:
            return

        if depth < 0:
            raise ValueError("depth must be >= 0")
        if not 0 < bank_slope <= 90:
            raise ValueError("bank_slope must be in range (0, 90]")

        # Calculate bank width from slope
        bank_width = depth / math.tan(math.radians(bank_slope))

        # Sample along the spline
        segments = int(self.get_length() / (heightfield.sample_spacing * 0.5))
        segments = max(10, segments)

        for i in range(segments + 1):
            t = i / segments
            pos = self.evaluate(t)
            width = self.evaluate_width(t)
            tangent = self.evaluate_tangent(t)

            # Calculate perpendicular direction
            perp_x = -tangent[2]
            perp_z = tangent[0]
            perp_len = math.sqrt(perp_x * perp_x + perp_z * perp_z)
            if perp_len > 0:
                perp_x /= perp_len
                perp_z /= perp_len

            # Sample across the river width including banks
            total_width = width + bank_width * 2
            samples_across = int(total_width / heightfield.sample_spacing) + 1

            for j in range(samples_across):
                offset = -total_width / 2 + j * heightfield.sample_spacing
                world_x = pos[0] + perp_x * offset
                world_z = pos[2] + perp_z * offset

                sx, sz = heightfield.world_to_sample(world_x, world_z)
                if 0 <= sx < heightfield.width and 0 <= sz < heightfield.height:
                    dist_from_center = abs(offset)
                    river_bed = self.water_level - depth

                    if dist_from_center <= width / 2:
                        # In the river channel
                        new_height = river_bed
                    elif dist_from_center <= width / 2 + bank_width:
                        # On the bank - linear slope
                        bank_progress = (dist_from_center - width / 2) / bank_width
                        new_height = river_bed + bank_progress * depth
                    else:
                        continue

                    current_height = heightfield.get_height_at(sx, sz)
                    if new_height < current_height:
                        heightfield.set_height_at(sx, sz, new_height)

    def get_flow_direction_at(
        self, world_x: float, world_z: float
    ) -> Optional[Tuple[float, float, float]]:
        """Get water flow direction at a world position.

        Args:
            world_x: World X coordinate.
            world_z: World Z coordinate.

        Returns:
            Normalized flow direction or None if not in river.
        """
        if len(self._points) < 2:
            return None

        # Find closest point on spline
        min_dist = float("inf")
        closest_t = 0.0

        segments = 50
        for i in range(segments + 1):
            t = i / segments
            pos = self.evaluate(t)
            dx = pos[0] - world_x
            dz = pos[2] - world_z
            dist = math.sqrt(dx * dx + dz * dz)
            if dist < min_dist:
                min_dist = dist
                closest_t = t

        width = self.evaluate_width(closest_t)
        if min_dist <= width / 2:
            tangent = self.evaluate_tangent(closest_t)
            # Scale by flow speed
            return (
                tangent[0] * self.flow_speed,
                tangent[1] * self.flow_speed,
                tangent[2] * self.flow_speed,
            )

        return None


@dataclass
class DeformationSettings:
    """Settings for terrain deformation operations.

    Attributes:
        blend_mode: How to blend with existing terrain ("replace", "add", "min", "max").
        smoothing_passes: Number of smoothing passes to apply.
        smoothing_strength: Strength of smoothing (0-1).
    """

    blend_mode: str = "replace"
    smoothing_passes: int = 0
    smoothing_strength: float = 0.5

    def __post_init__(self) -> None:
        """Validate settings."""
        valid_modes = ("replace", "add", "min", "max")
        if self.blend_mode not in valid_modes:
            raise ValueError(f"blend_mode must be one of {valid_modes}")
        if self.smoothing_passes < 0:
            raise ValueError("smoothing_passes must be >= 0")
        if not 0 <= self.smoothing_strength <= 1:
            raise ValueError("smoothing_strength must be in range [0, 1]")


class TerrainDeformer:
    """Applies various deformations to terrain heightfields."""

    def apply_spline_deformation(
        self,
        heightfield: Heightfield,
        spline: TerrainSpline,
        settings: DeformationSettings,
    ) -> None:
        """Apply spline-based deformation to terrain.

        Args:
            heightfield: The heightfield to modify.
            spline: The spline defining the deformation.
            settings: Deformation settings.
        """
        if isinstance(spline, RoadSpline):
            spline.deform_terrain(heightfield, depth=0.1, blend_width=5.0)
        elif isinstance(spline, RiverSpline):
            spline.carve_channel(heightfield, depth=2.0, bank_slope=45.0)

        # Apply smoothing if requested
        if settings.smoothing_passes > 0:
            self._smooth_terrain(
                heightfield, settings.smoothing_passes, settings.smoothing_strength
            )

    def _smooth_terrain(
        self,
        heightfield: Heightfield,
        passes: int,
        strength: float,
    ) -> None:
        """Apply smoothing to the terrain.

        Args:
            heightfield: The heightfield to smooth.
            passes: Number of smoothing passes.
            strength: Strength of smoothing.
        """
        for _ in range(passes):
            new_heights = {}

            for z in range(heightfield.height):
                for x in range(heightfield.width):
                    current = heightfield.get_height_at(x, z)

                    # Average with neighbors
                    total = current
                    count = 1

                    for dz in range(-1, 2):
                        for dx in range(-1, 2):
                            if dx == 0 and dz == 0:
                                continue
                            nx, nz = x + dx, z + dz
                            if 0 <= nx < heightfield.width and 0 <= nz < heightfield.height:
                                total += heightfield.get_height_at(nx, nz)
                                count += 1

                    avg = total / count
                    new_heights[(x, z)] = current + (avg - current) * strength

            for (x, z), height in new_heights.items():
                heightfield.set_height_at(x, z, height)

    def create_ramp(
        self,
        heightfield: Heightfield,
        start: Tuple[float, float],
        end: Tuple[float, float],
        width: float,
        start_height: Optional[float] = None,
        end_height: Optional[float] = None,
    ) -> None:
        """Create a ramp between two points.

        Args:
            heightfield: The heightfield to modify.
            start: (x, z) world coordinates of ramp start.
            end: (x, z) world coordinates of ramp end.
            width: Width of the ramp.
            start_height: Height at start (uses terrain if None).
            end_height: Height at end (uses terrain if None).
        """
        if width <= 0:
            raise ValueError("width must be > 0")

        start_x, start_z = start
        end_x, end_z = end

        # Get heights if not specified
        if start_height is None:
            sx, sz = heightfield.world_to_sample(start_x, start_z)
            sx = max(0, min(heightfield.width - 1, sx))
            sz = max(0, min(heightfield.height - 1, sz))
            start_height = heightfield.get_height_at(sx, sz)

        if end_height is None:
            ex, ez = heightfield.world_to_sample(end_x, end_z)
            ex = max(0, min(heightfield.width - 1, ex))
            ez = max(0, min(heightfield.height - 1, ez))
            end_height = heightfield.get_height_at(ex, ez)

        # Calculate ramp direction
        dx = end_x - start_x
        dz = end_z - start_z
        ramp_length = math.sqrt(dx * dx + dz * dz)

        if ramp_length < 1e-6:
            return

        dir_x = dx / ramp_length
        dir_z = dz / ramp_length

        # Perpendicular direction
        perp_x = -dir_z
        perp_z = dir_x

        # Iterate over terrain samples
        for z in range(heightfield.height):
            for x in range(heightfield.width):
                world_x, world_z = heightfield.sample_to_world(x, z)

                # Project onto ramp line
                to_point_x = world_x - start_x
                to_point_z = world_z - start_z
                projection = to_point_x * dir_x + to_point_z * dir_z

                if projection < 0 or projection > ramp_length:
                    continue

                # Check perpendicular distance
                closest_x = start_x + dir_x * projection
                closest_z = start_z + dir_z * projection
                perp_dist = math.sqrt(
                    (world_x - closest_x) ** 2 + (world_z - closest_z) ** 2
                )

                if perp_dist > width / 2:
                    continue

                # Interpolate height
                t = projection / ramp_length
                target_height = start_height + (end_height - start_height) * t

                # Blend at edges
                edge_blend = 1.0 - (perp_dist / (width / 2))
                current_height = heightfield.get_height_at(x, z)
                new_height = current_height + (target_height - current_height) * edge_blend

                heightfield.set_height_at(x, z, new_height)

    def create_plateau(
        self,
        heightfield: Heightfield,
        center_x: float,
        center_z: float,
        radius: float,
        height: float,
        blend_width: float = 10.0,
    ) -> None:
        """Create a flat plateau on the terrain.

        Args:
            heightfield: The heightfield to modify.
            center_x: World X center of plateau.
            center_z: World Z center of plateau.
            radius: Radius of the flat area.
            height: Target height of the plateau.
            blend_width: Width of the blend zone.
        """
        if radius <= 0:
            raise ValueError("radius must be > 0")
        if blend_width < 0:
            raise ValueError("blend_width must be >= 0")

        total_radius = radius + blend_width

        for z in range(heightfield.height):
            for x in range(heightfield.width):
                world_x, world_z = heightfield.sample_to_world(x, z)
                dx = world_x - center_x
                dz = world_z - center_z
                distance = math.sqrt(dx * dx + dz * dz)

                if distance > total_radius:
                    continue

                current_height = heightfield.get_height_at(x, z)

                if distance <= radius:
                    # On the plateau
                    new_height = height
                else:
                    # In blend zone
                    blend_t = (distance - radius) / blend_width
                    blend_factor = blend_t * blend_t * (3 - 2 * blend_t)  # Smoothstep
                    new_height = height + (current_height - height) * blend_factor

                heightfield.set_height_at(x, z, new_height)


class PhysicalMaterialMapping:
    """Maps terrain layers to physical materials for gameplay/physics."""

    def __init__(self) -> None:
        """Initialize the mapping."""
        self._layer_to_material: Dict[int, str] = {}
        self._default_material: str = "default"

    @property
    def default_material(self) -> str:
        """Get default material."""
        return self._default_material

    @default_material.setter
    def default_material(self, value: str) -> None:
        """Set default material."""
        self._default_material = value

    def set_mapping(self, layer_index: int, material_id: str) -> None:
        """Set material for a layer.

        Args:
            layer_index: Index of the terrain layer.
            material_id: Physical material identifier.
        """
        if layer_index < 0:
            raise ValueError("layer_index must be >= 0")
        self._layer_to_material[layer_index] = material_id

    def get_material_for_layer(self, layer_index: int) -> str:
        """Get material for a layer.

        Args:
            layer_index: Index of the terrain layer.

        Returns:
            Physical material identifier.
        """
        return self._layer_to_material.get(layer_index, self._default_material)

    def get_material_at(self, weight_map: WeightMap, x: int, z: int) -> str:
        """Get physical material at a terrain position.

        Args:
            weight_map: The terrain weight map.
            x: X coordinate.
            z: Z coordinate.

        Returns:
            Physical material identifier.
        """
        dominant_layer = weight_map.get_dominant_layer_at(x, z)
        return self.get_material_for_layer(dominant_layer)

    def clear_mappings(self) -> None:
        """Clear all material mappings."""
        self._layer_to_material.clear()


@dataclass
class HitResult:
    """Result of a terrain raycast or query.

    Attributes:
        hit: Whether the ray hit the terrain.
        position: (x, y, z) hit position.
        normal: (x, y, z) surface normal at hit.
        distance: Distance from ray origin to hit.
        material: Physical material at hit location.
    """

    hit: bool = False
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    distance: float = 0.0
    material: str = "default"


class TerrainCollision:
    """Terrain collision detection and queries."""

    def __init__(
        self,
        heightfield: Heightfield,
        hole_manager: Optional[TerrainHoleManager] = None,
        material_mapping: Optional[PhysicalMaterialMapping] = None,
        weight_map: Optional[WeightMap] = None,
    ) -> None:
        """Initialize terrain collision.

        Args:
            heightfield: The terrain heightfield.
            hole_manager: Optional manager for terrain holes.
            material_mapping: Optional physical material mapping.
            weight_map: Optional material weight map.
        """
        self._heightfield = heightfield
        self._hole_manager = hole_manager
        self._material_mapping = material_mapping
        self._weight_map = weight_map

    @property
    def heightfield(self) -> Heightfield:
        """Get the heightfield."""
        return self._heightfield

    def get_height_at(self, world_x: float, world_z: float) -> Optional[float]:
        """Get terrain height at a world position, considering holes.

        Args:
            world_x: World X coordinate.
            world_z: World Z coordinate.

        Returns:
            Terrain height or None if in a hole.
        """
        # Check for holes
        if self._hole_manager is not None:
            if not self._hole_manager.is_visible_at(world_x, world_z):
                return None

        # Get height with bilinear interpolation
        sx, sz = self._heightfield.world_to_sample(world_x, world_z)

        # Clamp to valid range
        x0 = max(0, min(self._heightfield.width - 1, sx))
        z0 = max(0, min(self._heightfield.height - 1, sz))
        x1 = min(x0 + 1, self._heightfield.width - 1)
        z1 = min(z0 + 1, self._heightfield.height - 1)

        # Calculate fractional parts
        world_x0, world_z0 = self._heightfield.sample_to_world(x0, z0)
        spacing = self._heightfield.sample_spacing

        fx = (world_x - world_x0) / spacing if spacing > 0 else 0
        fz = (world_z - world_z0) / spacing if spacing > 0 else 0

        fx = max(0, min(1, fx))
        fz = max(0, min(1, fz))

        # Bilinear interpolation
        h00 = self._heightfield.get_height_at(x0, z0)
        h10 = self._heightfield.get_height_at(x1, z0)
        h01 = self._heightfield.get_height_at(x0, z1)
        h11 = self._heightfield.get_height_at(x1, z1)

        h0 = h00 + (h10 - h00) * fx
        h1 = h01 + (h11 - h01) * fx

        return h0 + (h1 - h0) * fz

    def get_normal_at(
        self, world_x: float, world_z: float
    ) -> Tuple[float, float, float]:
        """Get terrain surface normal at a world position.

        Args:
            world_x: World X coordinate.
            world_z: World Z coordinate.

        Returns:
            Normalized (x, y, z) normal vector.
        """
        spacing = self._heightfield.sample_spacing

        # Sample neighboring heights
        h_center = self.get_height_at(world_x, world_z)
        h_left = self.get_height_at(world_x - spacing, world_z)
        h_right = self.get_height_at(world_x + spacing, world_z)
        h_back = self.get_height_at(world_x, world_z - spacing)
        h_front = self.get_height_at(world_x, world_z + spacing)

        # Handle holes by using center height
        if h_center is None:
            return (0.0, 1.0, 0.0)
        if h_left is None:
            h_left = h_center
        if h_right is None:
            h_right = h_center
        if h_back is None:
            h_back = h_center
        if h_front is None:
            h_front = h_center

        # Calculate normal from height differences
        dx = (h_left - h_right) / (2 * spacing)
        dz = (h_back - h_front) / (2 * spacing)

        # Normal is cross product of tangent vectors
        nx = dx
        ny = 1.0
        nz = dz

        # Normalize
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length > 0:
            return (nx / length, ny / length, nz / length)
        return (0.0, 1.0, 0.0)

    def get_physical_material_at(self, world_x: float, world_z: float) -> str:
        """Get physical material at a world position.

        Args:
            world_x: World X coordinate.
            world_z: World Z coordinate.

        Returns:
            Physical material identifier.
        """
        if self._material_mapping is None or self._weight_map is None:
            return "default"

        sx, sz = self._heightfield.world_to_sample(world_x, world_z)
        sx = max(0, min(self._heightfield.width - 1, sx))
        sz = max(0, min(self._heightfield.height - 1, sz))

        return self._material_mapping.get_material_at(self._weight_map, sx, sz)

    def raycast(
        self,
        origin_x: float,
        origin_y: float,
        origin_z: float,
        direction_x: float,
        direction_y: float,
        direction_z: float,
        max_distance: float = 1000.0,
    ) -> HitResult:
        """Cast a ray against the terrain.

        Args:
            origin_x: Ray origin X.
            origin_y: Ray origin Y.
            origin_z: Ray origin Z.
            direction_x: Ray direction X.
            direction_y: Ray direction Y.
            direction_z: Ray direction Z.
            max_distance: Maximum raycast distance.

        Returns:
            HitResult with hit information.
        """
        # Normalize direction
        dir_len = math.sqrt(
            direction_x * direction_x
            + direction_y * direction_y
            + direction_z * direction_z
        )
        if dir_len == 0:
            return HitResult(hit=False)

        dir_x = direction_x / dir_len
        dir_y = direction_y / dir_len
        dir_z = direction_z / dir_len

        # March along ray
        step_size = self._heightfield.sample_spacing * 0.5
        distance = 0.0

        while distance < max_distance:
            # Current position
            x = origin_x + dir_x * distance
            y = origin_y + dir_y * distance
            z = origin_z + dir_z * distance

            # Get terrain height
            terrain_height = self.get_height_at(x, z)

            if terrain_height is not None and y <= terrain_height:
                # Hit - refine with binary search
                low_dist = max(0, distance - step_size)
                high_dist = distance

                for _ in range(8):
                    mid_dist = (low_dist + high_dist) / 2
                    mid_x = origin_x + dir_x * mid_dist
                    mid_y = origin_y + dir_y * mid_dist
                    mid_z = origin_z + dir_z * mid_dist
                    mid_height = self.get_height_at(mid_x, mid_z)

                    if mid_height is not None and mid_y <= mid_height:
                        high_dist = mid_dist
                    else:
                        low_dist = mid_dist

                hit_dist = (low_dist + high_dist) / 2
                hit_x = origin_x + dir_x * hit_dist
                hit_z = origin_z + dir_z * hit_dist
                hit_y = self.get_height_at(hit_x, hit_z) or 0

                normal = self.get_normal_at(hit_x, hit_z)
                material = self.get_physical_material_at(hit_x, hit_z)

                return HitResult(
                    hit=True,
                    position=(hit_x, hit_y, hit_z),
                    normal=normal,
                    distance=hit_dist,
                    material=material,
                )

            distance += step_size

        return HitResult(hit=False)

    def sphere_cast(
        self,
        origin_x: float,
        origin_y: float,
        origin_z: float,
        radius: float,
        direction_x: float,
        direction_y: float,
        direction_z: float,
        max_distance: float = 1000.0,
    ) -> HitResult:
        """Cast a sphere against the terrain.

        Args:
            origin_x: Sphere origin X.
            origin_y: Sphere origin Y.
            origin_z: Sphere origin Z.
            radius: Sphere radius.
            direction_x: Cast direction X.
            direction_y: Cast direction Y.
            direction_z: Cast direction Z.
            max_distance: Maximum cast distance.

        Returns:
            HitResult with hit information.
        """
        if radius <= 0:
            raise ValueError("radius must be > 0")

        # Offset origin down by radius
        result = self.raycast(
            origin_x,
            origin_y - radius,
            origin_z,
            direction_x,
            direction_y,
            direction_z,
            max_distance,
        )

        if result.hit:
            # Adjust hit position back up by radius
            result.position = (
                result.position[0],
                result.position[1] + radius,
                result.position[2],
            )

        return result

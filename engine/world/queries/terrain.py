"""
Terrain-specific query system for the game engine world layer.

Provides height queries, raycasts against terrain heightfields,
and area-based terrain analysis.

Query Types:
    - Height: Get terrain height at any (x, z) coordinate
    - Normal: Get surface normal at any point
    - Slope: Get slope angle in degrees
    - Layer: Query terrain layer weights (grass, rock, sand, etc.)
    - Raycast: Ray-terrain intersection using heightfield stepping
    - Area: Bulk queries for terrain data within bounds

Example:
    >>> terrain_query = TerrainQuery(terrain_system)
    >>> height = terrain_query.get_height_at(100.0, 200.0)
    >>> normal = terrain_query.get_normal_at(100.0, 200.0)
    >>> slope = terrain_query.get_slope_at(100.0, 200.0)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from engine.world.queries.constants import (
    EPSILON_NORMALIZE,
    EPSILON_HEIGHT,
    DEFAULT_TERRAIN_RAYCAST_MAX_DISTANCE,
    DEFAULT_LINE_TRACE_STEP_SIZE,
    DEFAULT_TERRAIN_STEP_MULTIPLIER,
    TERRAIN_RAYCAST_BASE_STEP_FACTOR,
    TERRAIN_RAYCAST_MIN_ADAPTIVE_FACTOR,
    TERRAIN_RAYCAST_MAX_ADAPTIVE_MULTIPLIER,
    TERRAIN_INTERSECTION_BINARY_SEARCH_ITERATIONS,
    DEFAULT_AREA_QUERY_RESOLUTION,
    MIN_MAX_HEIGHT_RESOLUTION,
    DEFAULT_FLAT_AREA_MAX_SLOPE,
    DEFAULT_FLAT_AREA_MIN_SIZE,
    NORMAL_CALCULATION_EPSILON_FACTOR,
)


# =============================================================================
# TYPE ALIASES
# =============================================================================

Vector3 = Tuple[float, float, float]
Vector2 = Tuple[float, float]
Bounds2D = Tuple[float, float, float, float]  # min_x, min_z, max_x, max_z


# =============================================================================
# TERRAIN HIT RESULT
# =============================================================================


@dataclass
class TerrainHitResult:
    """
    Result of a terrain query or raycast.

    Attributes:
        hit: Whether a hit occurred.
        position: World position of the hit point.
        normal: Surface normal at the hit point.
        height: Terrain height at the hit point.
        slope_degrees: Slope angle in degrees.
        layer_weights: Weights for each terrain layer at this point.
        physical_material: Name of the dominant physical material.
        terrain_patch_id: ID of the terrain patch containing this point.
    """

    hit: bool = False
    position: Vector3 = (0.0, 0.0, 0.0)
    normal: Vector3 = (0.0, 1.0, 0.0)
    height: float = 0.0
    slope_degrees: float = 0.0
    layer_weights: List[float] = field(default_factory=list)
    physical_material: str = "default"
    terrain_patch_id: Optional[int] = None

    @staticmethod
    def no_hit() -> "TerrainHitResult":
        """Create a no-hit result."""
        return TerrainHitResult(hit=False)


# =============================================================================
# TERRAIN SYSTEM PROTOCOL
# =============================================================================


@runtime_checkable
class TerrainSystem(Protocol):
    """Protocol for terrain systems providing height and layer data."""

    def get_height(self, x: float, z: float) -> float:
        """Get terrain height at (x, z) using bilinear interpolation."""
        ...

    def get_raw_height(self, grid_x: int, grid_z: int) -> float:
        """Get raw height at grid coordinates."""
        ...

    def get_bounds(self) -> Bounds2D:
        """Get terrain bounds (min_x, min_z, max_x, max_z)."""
        ...

    def get_resolution(self) -> Tuple[int, int]:
        """Get heightfield resolution (width, height)."""
        ...

    def get_cell_size(self) -> float:
        """Get the size of each terrain cell."""
        ...

    def get_layer_count(self) -> int:
        """Get the number of terrain layers."""
        ...

    def get_layer_weight(self, x: float, z: float, layer_index: int) -> float:
        """Get the weight of a specific layer at (x, z)."""
        ...

    def get_layer_name(self, layer_index: int) -> str:
        """Get the name of a terrain layer."""
        ...

    def get_physical_material(self, x: float, z: float) -> str:
        """Get the physical material name at (x, z)."""
        ...

    def get_patch_id(self, x: float, z: float) -> Optional[int]:
        """Get the terrain patch ID containing (x, z)."""
        ...


# =============================================================================
# HOLE MANAGER PROTOCOL
# =============================================================================


@runtime_checkable
class TerrainHoleManager(Protocol):
    """Protocol for terrain hole/visibility management."""

    def is_visible(self, x: float, z: float) -> bool:
        """Check if terrain is visible (not a hole) at (x, z)."""
        ...

    def get_hole_mask(
        self, min_x: float, min_z: float, max_x: float, max_z: float
    ) -> List[List[bool]]:
        """Get visibility mask for a region. True = visible."""
        ...


# =============================================================================
# TERRAIN QUERY
# =============================================================================


class TerrainQuery:
    """
    Core terrain query system for height and layer data.

    Uses bilinear interpolation for smooth height queries.

    Example:
        >>> query = TerrainQuery(terrain_system)
        >>> height = query.get_height_at(100.0, 200.0)
        >>> normal = query.get_normal_at(100.0, 200.0)
        >>> slope = query.get_slope_at(100.0, 200.0)
    """

    def __init__(self, terrain_system: TerrainSystem) -> None:
        """
        Initialize the terrain query system.

        Args:
            terrain_system: The terrain system to query.
        """
        self._terrain = terrain_system

    @property
    def terrain(self) -> TerrainSystem:
        """Get the underlying terrain system."""
        return self._terrain

    def get_height_at(self, x: float, z: float) -> float:
        """
        Get terrain height at world position.

        Uses bilinear interpolation between grid points.

        Args:
            x: World X coordinate.
            z: World Z coordinate.

        Returns:
            Terrain height at the position.
        """
        return self._terrain.get_height(x, z)

    def get_normal_at(self, x: float, z: float) -> Vector3:
        """
        Get terrain surface normal at world position.

        Computed using central difference method.

        Args:
            x: World X coordinate.
            z: World Z coordinate.

        Returns:
            Normalized surface normal vector.
        """
        cell_size = self._terrain.get_cell_size()
        epsilon = cell_size * NORMAL_CALCULATION_EPSILON_FACTOR

        # Sample heights around the point
        h_left = self._terrain.get_height(x - epsilon, z)
        h_right = self._terrain.get_height(x + epsilon, z)
        h_down = self._terrain.get_height(x, z - epsilon)
        h_up = self._terrain.get_height(x, z + epsilon)

        # Calculate gradient
        dx = (h_right - h_left) / (2.0 * epsilon)
        dz = (h_up - h_down) / (2.0 * epsilon)

        # Normal is (-dh/dx, 1, -dh/dz), normalized
        normal = (-dx, 1.0, -dz)
        length = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)

        if length < EPSILON_NORMALIZE:
            return (0.0, 1.0, 0.0)

        return (normal[0] / length, normal[1] / length, normal[2] / length)

    def get_slope_at(self, x: float, z: float) -> float:
        """
        Get terrain slope in degrees at world position.

        Args:
            x: World X coordinate.
            z: World Z coordinate.

        Returns:
            Slope angle in degrees (0 = flat, 90 = vertical).
        """
        normal = self.get_normal_at(x, z)
        # Slope is angle from vertical
        dot_up = normal[1]  # Dot with (0, 1, 0)
        dot_up = max(-1.0, min(1.0, dot_up))  # Clamp for acos
        angle_rad = math.acos(dot_up)
        return math.degrees(angle_rad)

    def get_layer_weights_at(self, x: float, z: float) -> List[float]:
        """
        Get all layer weights at world position.

        Args:
            x: World X coordinate.
            z: World Z coordinate.

        Returns:
            List of layer weights (sum to 1.0).
        """
        layer_count = self._terrain.get_layer_count()
        weights = []
        for i in range(layer_count):
            weights.append(self._terrain.get_layer_weight(x, z, i))
        return weights

    def get_dominant_layer_at(self, x: float, z: float) -> int:
        """
        Get the index of the dominant (highest weight) layer.

        Args:
            x: World X coordinate.
            z: World Z coordinate.

        Returns:
            Index of the dominant layer.
        """
        weights = self.get_layer_weights_at(x, z)
        if not weights:
            return 0
        return weights.index(max(weights))

    def get_physical_material_at(self, x: float, z: float) -> str:
        """
        Get the physical material at world position.

        Args:
            x: World X coordinate.
            z: World Z coordinate.

        Returns:
            Physical material name.
        """
        return self._terrain.get_physical_material(x, z)

    def is_in_bounds(self, x: float, z: float) -> bool:
        """
        Check if a position is within terrain bounds.

        Args:
            x: World X coordinate.
            z: World Z coordinate.

        Returns:
            True if position is within terrain bounds.
        """
        bounds = self._terrain.get_bounds()
        return bounds[0] <= x <= bounds[2] and bounds[1] <= z <= bounds[3]


# =============================================================================
# TERRAIN RAYCAST
# =============================================================================


class TerrainRaycast:
    """
    Raycast against terrain heightfield using stepping algorithm.

    Uses adaptive step size based on terrain slope for efficiency.

    Example:
        >>> raycast = TerrainRaycast(terrain_system)
        >>> result = raycast.raycast(
        ...     origin=(0, 100, 0),
        ...     direction=(0, -1, 0),
        ...     max_distance=500.0
        ... )
        >>> if result.hit:
        ...     print(f"Hit terrain at {result.position}")
    """

    def __init__(
        self,
        terrain_system: TerrainSystem,
        step_multiplier: float = DEFAULT_TERRAIN_STEP_MULTIPLIER,
    ) -> None:
        """
        Initialize terrain raycast.

        Args:
            terrain_system: The terrain system to raycast against.
            step_multiplier: Multiplier for step size (smaller = more accurate).
        """
        self._terrain = terrain_system
        self._query = TerrainQuery(terrain_system)
        self._step_multiplier = step_multiplier

    def raycast(
        self,
        origin: Vector3,
        direction: Vector3,
        max_distance: float = DEFAULT_TERRAIN_RAYCAST_MAX_DISTANCE,
    ) -> TerrainHitResult:
        """
        Cast a ray against the terrain.

        Uses heightfield stepping algorithm for efficiency.

        Args:
            origin: Ray origin in world space.
            direction: Ray direction (will be normalized).
            max_distance: Maximum ray distance.

        Returns:
            TerrainHitResult with hit information.
        """
        # Normalize direction
        length = math.sqrt(
            direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2
        )
        if length < EPSILON_NORMALIZE:
            return TerrainHitResult.no_hit()

        dir_norm = (
            direction[0] / length,
            direction[1] / length,
            direction[2] / length,
        )

        # Check if ray is parallel to terrain or pointing away
        if abs(dir_norm[1]) < EPSILON_HEIGHT:
            # Horizontal ray - would need to check sides
            return TerrainHitResult.no_hit()

        # Calculate step size based on terrain cell size
        cell_size = self._terrain.get_cell_size()
        base_step = cell_size * TERRAIN_RAYCAST_BASE_STEP_FACTOR * self._step_multiplier

        # March along the ray
        t = 0.0
        prev_height_diff = None

        while t < max_distance:
            # Current position along ray
            pos = (
                origin[0] + dir_norm[0] * t,
                origin[1] + dir_norm[1] * t,
                origin[2] + dir_norm[2] * t,
            )

            # Check bounds
            if not self._query.is_in_bounds(pos[0], pos[2]):
                # Outside terrain bounds
                t += base_step
                continue

            # Get terrain height at this XZ
            terrain_height = self._query.get_height_at(pos[0], pos[2])
            height_diff = pos[1] - terrain_height

            # Check for crossing
            if prev_height_diff is not None:
                if prev_height_diff > 0 and height_diff <= 0:
                    # Ray crossed from above to below terrain
                    # Binary search for exact intersection
                    hit_t = self._binary_search_intersection(
                        origin, dir_norm, t - base_step, t
                    )
                    hit_pos = (
                        origin[0] + dir_norm[0] * hit_t,
                        origin[1] + dir_norm[1] * hit_t,
                        origin[2] + dir_norm[2] * hit_t,
                    )

                    # Get terrain data at hit point
                    hit_height = self._query.get_height_at(hit_pos[0], hit_pos[2])
                    normal = self._query.get_normal_at(hit_pos[0], hit_pos[2])
                    slope = self._query.get_slope_at(hit_pos[0], hit_pos[2])
                    layers = self._query.get_layer_weights_at(hit_pos[0], hit_pos[2])
                    material = self._query.get_physical_material_at(
                        hit_pos[0], hit_pos[2]
                    )
                    patch_id = self._terrain.get_patch_id(hit_pos[0], hit_pos[2])

                    return TerrainHitResult(
                        hit=True,
                        position=(hit_pos[0], hit_height, hit_pos[2]),
                        normal=normal,
                        height=hit_height,
                        slope_degrees=slope,
                        layer_weights=layers,
                        physical_material=material,
                        terrain_patch_id=patch_id,
                    )

            prev_height_diff = height_diff

            # Adaptive step size based on height above terrain
            adaptive_step = base_step
            if height_diff > 0:
                # Above terrain - can take larger steps if far away
                adaptive_step = min(
                    height_diff * TERRAIN_RAYCAST_MIN_ADAPTIVE_FACTOR,
                    base_step * TERRAIN_RAYCAST_MAX_ADAPTIVE_MULTIPLIER
                )
                adaptive_step = max(adaptive_step, base_step)

            t += adaptive_step

        return TerrainHitResult.no_hit()

    def _binary_search_intersection(
        self,
        origin: Vector3,
        direction: Vector3,
        t_min: float,
        t_max: float,
        iterations: int = TERRAIN_INTERSECTION_BINARY_SEARCH_ITERATIONS,
    ) -> float:
        """Binary search to find exact ray-terrain intersection."""
        for _ in range(iterations):
            t_mid = (t_min + t_max) * 0.5
            pos = (
                origin[0] + direction[0] * t_mid,
                origin[1] + direction[1] * t_mid,
                origin[2] + direction[2] * t_mid,
            )

            if not self._query.is_in_bounds(pos[0], pos[2]):
                return t_mid

            terrain_height = self._query.get_height_at(pos[0], pos[2])

            if pos[1] > terrain_height:
                t_min = t_mid
            else:
                t_max = t_mid

        return (t_min + t_max) * 0.5


# =============================================================================
# TERRAIN LINE TRACE
# =============================================================================


class TerrainLineTrace:
    """
    Sample terrain data along a line (useful for roads, rivers).

    Example:
        >>> trace = TerrainLineTrace(terrain_system)
        >>> samples = trace.sample_along_line(
        ...     start=(0, 0, 0),
        ...     end=(100, 0, 100),
        ...     step_size=5.0
        ... )
    """

    def __init__(self, terrain_system: TerrainSystem) -> None:
        """
        Initialize terrain line trace.

        Args:
            terrain_system: The terrain system to sample.
        """
        self._terrain = terrain_system
        self._query = TerrainQuery(terrain_system)

    def sample_along_line(
        self,
        start: Vector3,
        end: Vector3,
        step_size: float = DEFAULT_LINE_TRACE_STEP_SIZE,
    ) -> List[TerrainHitResult]:
        """
        Sample terrain data along a line.

        Args:
            start: Start position (Y is ignored, terrain height is used).
            end: End position (Y is ignored).
            step_size: Distance between samples.

        Returns:
            List of TerrainHitResults along the line.
        """
        # Calculate 2D distance
        dx = end[0] - start[0]
        dz = end[2] - start[2]
        distance = math.sqrt(dx * dx + dz * dz)

        if distance < EPSILON_NORMALIZE:
            return []

        # Normalize direction
        dir_x = dx / distance
        dir_z = dz / distance

        # Sample along the line
        results: List[TerrainHitResult] = []
        num_samples = int(distance / step_size) + 1

        for i in range(num_samples):
            t = i * step_size
            if t > distance:
                t = distance

            x = start[0] + dir_x * t
            z = start[2] + dir_z * t

            # Check bounds
            if not self._query.is_in_bounds(x, z):
                continue

            # Get terrain data
            height = self._query.get_height_at(x, z)
            normal = self._query.get_normal_at(x, z)
            slope = self._query.get_slope_at(x, z)
            layers = self._query.get_layer_weights_at(x, z)
            material = self._query.get_physical_material_at(x, z)
            patch_id = self._terrain.get_patch_id(x, z)

            results.append(
                TerrainHitResult(
                    hit=True,
                    position=(x, height, z),
                    normal=normal,
                    height=height,
                    slope_degrees=slope,
                    layer_weights=layers,
                    physical_material=material,
                    terrain_patch_id=patch_id,
                )
            )

        return results


# =============================================================================
# TERRAIN AREA QUERY
# =============================================================================


class TerrainAreaQuery:
    """
    Bulk terrain queries for areas/regions.

    Useful for analyzing terrain for building placement, pathfinding, etc.

    Example:
        >>> area_query = TerrainAreaQuery(terrain_system)
        >>> heights = area_query.get_heights_in_bounds(0, 0, 100, 100, 10)
        >>> flat_areas = area_query.find_flat_areas(
        ...     bounds=(0, 0, 200, 200),
        ...     max_slope=15.0,
        ...     min_size=20.0
        ... )
    """

    def __init__(self, terrain_system: TerrainSystem) -> None:
        """
        Initialize terrain area query.

        Args:
            terrain_system: The terrain system to query.
        """
        self._terrain = terrain_system
        self._query = TerrainQuery(terrain_system)

    def get_heights_in_bounds(
        self,
        min_x: float,
        min_z: float,
        max_x: float,
        max_z: float,
        resolution: int = DEFAULT_AREA_QUERY_RESOLUTION,
    ) -> List[List[float]]:
        """
        Get a 2D array of heights within bounds.

        Args:
            min_x: Minimum X coordinate.
            min_z: Minimum Z coordinate.
            max_x: Maximum X coordinate.
            max_z: Maximum Z coordinate.
            resolution: Number of samples per axis.

        Returns:
            2D list of heights [z][x].
        """
        step_x = (max_x - min_x) / max(1, resolution - 1)
        step_z = (max_z - min_z) / max(1, resolution - 1)

        heights: List[List[float]] = []

        for iz in range(resolution):
            row: List[float] = []
            z = min_z + iz * step_z
            for ix in range(resolution):
                x = min_x + ix * step_x
                if self._query.is_in_bounds(x, z):
                    row.append(self._query.get_height_at(x, z))
                else:
                    row.append(0.0)
            heights.append(row)

        return heights

    def get_average_height(self, bounds: Bounds2D) -> float:
        """
        Get average terrain height within bounds.

        Args:
            bounds: (min_x, min_z, max_x, max_z) tuple.

        Returns:
            Average height within the bounds.
        """
        min_x, min_z, max_x, max_z = bounds
        heights = self.get_heights_in_bounds(min_x, min_z, max_x, max_z, DEFAULT_AREA_QUERY_RESOLUTION)

        total = 0.0
        count = 0
        for row in heights:
            for h in row:
                total += h
                count += 1

        return total / count if count > 0 else 0.0

    def get_min_max_height(self, bounds: Bounds2D) -> Tuple[float, float]:
        """
        Get minimum and maximum terrain height within bounds.

        Args:
            bounds: (min_x, min_z, max_x, max_z) tuple.

        Returns:
            (min_height, max_height) tuple.
        """
        min_x, min_z, max_x, max_z = bounds
        heights = self.get_heights_in_bounds(min_x, min_z, max_x, max_z, MIN_MAX_HEIGHT_RESOLUTION)

        min_h = float("inf")
        max_h = float("-inf")

        for row in heights:
            for h in row:
                min_h = min(min_h, h)
                max_h = max(max_h, h)

        if min_h == float("inf"):
            return (0.0, 0.0)

        return (min_h, max_h)

    def find_flat_areas(
        self,
        bounds: Bounds2D,
        max_slope: float = DEFAULT_FLAT_AREA_MAX_SLOPE,
        min_size: float = DEFAULT_FLAT_AREA_MIN_SIZE,
    ) -> List[Bounds2D]:
        """
        Find flat areas within bounds suitable for building placement.

        Args:
            bounds: (min_x, min_z, max_x, max_z) search area.
            max_slope: Maximum allowed slope in degrees.
            min_size: Minimum area size in world units.

        Returns:
            List of flat area bounds.
        """
        min_x, min_z, max_x, max_z = bounds

        # Sample slope across the area
        cell_size = min_size
        num_x = int((max_x - min_x) / cell_size) + 1
        num_z = int((max_z - min_z) / cell_size) + 1

        # Build a grid of slope values
        slope_grid: List[List[bool]] = []
        for iz in range(num_z):
            row: List[bool] = []
            z = min_z + iz * cell_size + cell_size * 0.5
            for ix in range(num_x):
                x = min_x + ix * cell_size + cell_size * 0.5
                if self._query.is_in_bounds(x, z):
                    slope = self._query.get_slope_at(x, z)
                    row.append(slope <= max_slope)
                else:
                    row.append(False)
            slope_grid.append(row)

        # Find connected flat regions (simplified flood fill)
        visited: List[List[bool]] = [
            [False for _ in range(num_x)] for _ in range(num_z)
        ]
        flat_areas: List[Bounds2D] = []

        for start_z in range(num_z):
            for start_x in range(num_x):
                if visited[start_z][start_x] or not slope_grid[start_z][start_x]:
                    continue

                # Flood fill to find connected flat region
                region_cells: List[Tuple[int, int]] = []
                stack = [(start_x, start_z)]

                while stack:
                    cx, cz = stack.pop()
                    if (
                        cx < 0
                        or cx >= num_x
                        or cz < 0
                        or cz >= num_z
                        or visited[cz][cx]
                        or not slope_grid[cz][cx]
                    ):
                        continue

                    visited[cz][cx] = True
                    region_cells.append((cx, cz))

                    stack.append((cx + 1, cz))
                    stack.append((cx - 1, cz))
                    stack.append((cx, cz + 1))
                    stack.append((cx, cz - 1))

                if len(region_cells) >= 1:  # At least min_size area
                    # Calculate bounds of this region
                    min_rx = min(c[0] for c in region_cells)
                    max_rx = max(c[0] for c in region_cells)
                    min_rz = min(c[1] for c in region_cells)
                    max_rz = max(c[1] for c in region_cells)

                    region_bounds = (
                        min_x + min_rx * cell_size,
                        min_z + min_rz * cell_size,
                        min_x + (max_rx + 1) * cell_size,
                        min_z + (max_rz + 1) * cell_size,
                    )
                    flat_areas.append(region_bounds)

        return flat_areas


# =============================================================================
# TERRAIN VISIBILITY
# =============================================================================


class TerrainVisibility:
    """
    Query terrain visibility (accounting for holes).

    Terrain holes are used for caves, tunnels, etc.

    Example:
        >>> visibility = TerrainVisibility(terrain_system, hole_manager)
        >>> if visibility.is_terrain_visible_at(100, 100):
        ...     # Render terrain here
        ...     pass
    """

    def __init__(
        self,
        terrain_system: TerrainSystem,
        hole_manager: Optional[TerrainHoleManager] = None,
    ) -> None:
        """
        Initialize terrain visibility query.

        Args:
            terrain_system: The terrain system.
            hole_manager: Optional hole manager for visibility checks.
        """
        self._terrain = terrain_system
        self._hole_manager = hole_manager
        self._query = TerrainQuery(terrain_system)

    def is_terrain_visible_at(self, x: float, z: float) -> bool:
        """
        Check if terrain is visible at a position.

        Args:
            x: World X coordinate.
            z: World Z coordinate.

        Returns:
            True if terrain is visible (not a hole).
        """
        # Check bounds first
        if not self._query.is_in_bounds(x, z):
            return False

        # Check hole manager
        if self._hole_manager is not None:
            return self._hole_manager.is_visible(x, z)

        return True

    def get_visible_bounds(self, bounds: Bounds2D) -> Optional[Bounds2D]:
        """
        Get the visible portion of the given bounds.

        Args:
            bounds: Input bounds to check.

        Returns:
            Visible bounds, or None if entirely hidden.
        """
        min_x, min_z, max_x, max_z = bounds
        terrain_bounds = self._terrain.get_bounds()

        # Clamp to terrain bounds
        clamped = (
            max(min_x, terrain_bounds[0]),
            max(min_z, terrain_bounds[1]),
            min(max_x, terrain_bounds[2]),
            min(max_z, terrain_bounds[3]),
        )

        # Check if there's any valid area left
        if clamped[0] >= clamped[2] or clamped[1] >= clamped[3]:
            return None

        return clamped


# =============================================================================
# TERRAIN QUERY SYSTEM
# =============================================================================


class TerrainQuerySystem:
    """
    Unified terrain query system combining all query types.

    Example:
        >>> system = TerrainQuerySystem(terrain, hole_manager)
        >>> height = system.query.get_height_at(100, 100)
        >>> hit = system.raycast.raycast(origin, direction)
        >>> flat_areas = system.area.find_flat_areas(bounds, max_slope=10)
    """

    def __init__(
        self,
        terrain_system: TerrainSystem,
        hole_manager: Optional[TerrainHoleManager] = None,
    ) -> None:
        """
        Initialize the unified terrain query system.

        Args:
            terrain_system: The terrain system to query.
            hole_manager: Optional hole manager for visibility.
        """
        self._terrain = terrain_system
        self._hole_manager = hole_manager

        # Initialize sub-systems
        self.query = TerrainQuery(terrain_system)
        self.raycast = TerrainRaycast(terrain_system)
        self.line_trace = TerrainLineTrace(terrain_system)
        self.area = TerrainAreaQuery(terrain_system)
        self.visibility = TerrainVisibility(terrain_system, hole_manager)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    "TerrainHitResult",
    # Protocols
    "TerrainSystem",
    "TerrainHoleManager",
    # Query classes
    "TerrainQuery",
    "TerrainRaycast",
    "TerrainLineTrace",
    "TerrainAreaQuery",
    "TerrainVisibility",
    # Systems
    "TerrainQuerySystem",
]

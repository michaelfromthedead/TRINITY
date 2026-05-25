"""
Terrain components for rendering and collision in the game engine World Layer.

This module provides terrain components following the game engine component pattern:
- LandscapeComponent: Render unit for terrain visualization
- TerrainSection: LOD unit within a component for granular detail control
- TerrainProxy: Collision unit for physics interaction
- TerrainActor: High-level terrain manager composing multiple components

These components integrate with the broader engine architecture for rendering,
physics, and world management systems.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import math

from .heightfield import Heightfield
from .patch import TerrainPatch, AABB
from .constants import (
    DEFAULT_BOUNDS,
    LOD_BIAS_MIN,
    LOD_BIAS_MAX,
    LOD_BIAS_MULTIPLIER,
    DEFAULT_FRICTION,
    DEFAULT_RESTITUTION,
    PHYSICS_COEFF_MIN,
    PHYSICS_COEFF_MAX,
    DEFAULT_RAYCAST_MAX_DISTANCE,
    RAY_DIRECTION_EPSILON,
    RAYCAST_STEP_MULTIPLIER,
    RAYCAST_BINARY_SEARCH_ITERATIONS,
)


@dataclass
class LandscapeComponent:
    """Render unit for terrain.

    A LandscapeComponent represents a renderable portion of terrain.
    It contains a terrain patch and material information for rendering.

    Attributes:
        bounds: Axis-aligned bounding box (min_x, min_y, min_z, max_x, max_y, max_z)
        patch: Associated terrain patch with heightfield data
        material_id: Identifier for the terrain material/texture
        lod_bias: LOD bias adjustment (-1.0 to 1.0, negative = more detail)
        cast_shadows: Whether this component casts shadows
        visible: Whether this component should be rendered
    """
    bounds: AABB = DEFAULT_BOUNDS
    patch: Optional[TerrainPatch] = None
    material_id: str = ""
    lod_bias: float = 0.0  # Valid range: LOD_BIAS_MIN to LOD_BIAS_MAX
    cast_shadows: bool = True
    visible: bool = True

    # Internal state
    _render_data: Optional[Any] = field(default=None, repr=False)
    _dirty: bool = field(default=True, repr=False)

    def __post_init__(self):
        """Validate component configuration."""
        if len(self.bounds) != 6:
            raise ValueError("bounds must be a 6-tuple (min_x, min_y, min_z, max_x, max_y, max_z)")
        if self.bounds[0] > self.bounds[3]:
            raise ValueError("bounds min_x must be <= max_x")
        if self.bounds[1] > self.bounds[4]:
            raise ValueError("bounds min_y must be <= max_y")
        if self.bounds[2] > self.bounds[5]:
            raise ValueError("bounds min_z must be <= max_z")
        if not LOD_BIAS_MIN <= self.lod_bias <= LOD_BIAS_MAX:
            raise ValueError(f"lod_bias must be between {LOD_BIAS_MIN} and {LOD_BIAS_MAX}")

    def update_bounds_from_patch(self) -> None:
        """Update bounds from the associated patch."""
        if self.patch is not None:
            self.bounds = self.patch.get_world_bounds()
            self._dirty = True

    def get_adjusted_lod(self, base_lod: int, max_lod: int) -> int:
        """Apply LOD bias to base LOD level.

        Args:
            base_lod: The LOD level before bias adjustment
            max_lod: Maximum LOD level (exclusive)

        Returns:
            Adjusted LOD level clamped to valid range.
        """
        # Bias < 0 means more detail (lower LOD number)
        # Bias > 0 means less detail (higher LOD number)
        adjusted = base_lod + int(self.lod_bias * LOD_BIAS_MULTIPLIER)
        return max(0, min(max_lod - 1, adjusted))

    def intersects_frustum(self, frustum_planes: List[Tuple[float, float, float, float]]) -> bool:
        """Check if component bounds intersect a view frustum.

        Args:
            frustum_planes: List of (a, b, c, d) plane equations where
                           ax + by + cz + d = 0 defines each plane.

        Returns:
            True if bounds might be visible, False if definitely outside frustum.
        """
        if not frustum_planes:
            return True

        min_x, min_y, min_z, max_x, max_y, max_z = self.bounds

        # Test against each frustum plane
        for a, b, c, d in frustum_planes:
            # Find the vertex most in the direction of the plane normal
            test_x = max_x if a >= 0 else min_x
            test_y = max_y if b >= 0 else min_y
            test_z = max_z if c >= 0 else min_z

            # If this vertex is behind the plane, box is outside
            if a * test_x + b * test_y + c * test_z + d < 0:
                return False

        return True

    def mark_dirty(self) -> None:
        """Mark component as needing render data regeneration."""
        self._dirty = True

    def is_dirty(self) -> bool:
        """Check if component needs render data regeneration."""
        return self._dirty

    def clear_dirty(self) -> None:
        """Clear the dirty flag after regenerating render data."""
        self._dirty = False


@dataclass
class TerrainSection:
    """LOD unit within a landscape component.

    A TerrainSection represents a subdivision of a LandscapeComponent
    for more granular LOD control. Each section can have its own
    LOD level based on its distance to the camera.

    Attributes:
        component: Parent landscape component
        section_index: Index within the component
        lod_levels: List of mesh data for each LOD level
        current_lod: Currently selected LOD level
    """
    component: Optional[LandscapeComponent] = None
    section_index: int = 0
    lod_levels: List[Any] = field(default_factory=list)
    current_lod: int = 0

    # Section bounds (subset of component bounds)
    _bounds: Optional[AABB] = field(default=None, repr=False)

    def get_bounds(self) -> AABB:
        """Get section bounds.

        Returns:
            Section AABB, or component bounds if not specifically set.
        """
        if self._bounds is not None:
            return self._bounds
        if self.component is not None:
            return self.component.bounds
        return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    def set_bounds(self, bounds: AABB) -> None:
        """Set section-specific bounds.

        Args:
            bounds: AABB for this section
        """
        if len(bounds) != 6:
            raise ValueError("bounds must be a 6-tuple")
        self._bounds = bounds

    def select_lod(self, camera_distance: float, lod_distances: Tuple[float, ...]) -> int:
        """Select LOD level based on camera distance.

        Args:
            camera_distance: Distance from camera to section center
            lod_distances: Distance thresholds for LOD transitions

        Returns:
            Selected LOD level
        """
        for level, threshold in enumerate(lod_distances):
            if camera_distance < threshold:
                self.current_lod = min(level, len(self.lod_levels) - 1)
                return self.current_lod

        self.current_lod = len(self.lod_levels) - 1 if self.lod_levels else 0
        return self.current_lod

    def get_mesh_data(self) -> Optional[Any]:
        """Get mesh data for current LOD level.

        Returns:
            Mesh data for current LOD, or None if not available.
        """
        if not self.lod_levels or self.current_lod >= len(self.lod_levels):
            return None
        return self.lod_levels[self.current_lod]

    def set_mesh_data(self, lod_level: int, mesh_data: Any) -> bool:
        """Set mesh data for a specific LOD level.

        Args:
            lod_level: LOD level to set
            mesh_data: Mesh data to store

        Returns:
            True if successful, False if LOD level is invalid.
        """
        if lod_level < 0:
            return False

        # Extend list if necessary
        while len(self.lod_levels) <= lod_level:
            self.lod_levels.append(None)

        self.lod_levels[lod_level] = mesh_data
        return True

    def get_center(self) -> Tuple[float, float, float]:
        """Get section center point.

        Returns:
            Tuple (x, y, z) of section center.
        """
        bounds = self.get_bounds()
        return (
            (bounds[0] + bounds[3]) / 2.0,
            (bounds[1] + bounds[4]) / 2.0,
            (bounds[2] + bounds[5]) / 2.0
        )


@dataclass
class TerrainProxy:
    """Collision unit for terrain.

    A TerrainProxy provides collision geometry for physics simulation.
    It references a heightfield for height queries and defines the
    physical material properties.

    Attributes:
        bounds: Collision bounds (min_x, min_y, min_z, max_x, max_y, max_z)
        heightfield_ref: Reference to heightfield data for collision queries
        physical_material: Material identifier for physics properties
    """
    bounds: AABB = DEFAULT_BOUNDS
    heightfield_ref: Optional[Heightfield] = None
    physical_material: str = "default"

    # Collision properties
    _friction: float = field(default=DEFAULT_FRICTION, repr=False)
    _restitution: float = field(default=DEFAULT_RESTITUTION, repr=False)

    def __post_init__(self):
        """Validate proxy configuration."""
        if len(self.bounds) != 6:
            raise ValueError("bounds must be a 6-tuple")

    def get_height_at(self, world_x: float, world_z: float) -> Optional[float]:
        """Get terrain height for collision at world position.

        Args:
            world_x: X position in world space
            world_z: Z position in world space

        Returns:
            Height at position, or None if outside bounds or no heightfield.
        """
        if self.heightfield_ref is None:
            return None

        # Check horizontal bounds
        if not (self.bounds[0] <= world_x <= self.bounds[3] and
                self.bounds[2] <= world_z <= self.bounds[5]):
            return None

        # Convert to local coordinates
        local_x = world_x - self.bounds[0]
        local_z = world_z - self.bounds[2]

        return self.heightfield_ref.get_height_at(local_x, local_z)

    def get_normal_at(self, world_x: float, world_z: float) -> Optional[Tuple[float, float, float]]:
        """Get terrain normal for collision at world position.

        Args:
            world_x: X position in world space
            world_z: Z position in world space

        Returns:
            Normal vector, or None if outside bounds or no heightfield.
        """
        if self.heightfield_ref is None:
            return None

        # Check horizontal bounds
        if not (self.bounds[0] <= world_x <= self.bounds[3] and
                self.bounds[2] <= world_z <= self.bounds[5]):
            return None

        # Convert to local coordinates
        local_x = world_x - self.bounds[0]
        local_z = world_z - self.bounds[2]

        return self.heightfield_ref.get_normal_at(local_x, local_z)

    def point_in_bounds(self, x: float, y: float, z: float) -> bool:
        """Check if a point is within collision bounds.

        Args:
            x, y, z: World position

        Returns:
            True if point is inside bounds.
        """
        return (self.bounds[0] <= x <= self.bounds[3] and
                self.bounds[1] <= y <= self.bounds[4] and
                self.bounds[2] <= z <= self.bounds[5])

    def get_friction(self) -> float:
        """Get friction coefficient for physics."""
        return self._friction

    def set_friction(self, friction: float) -> None:
        """Set friction coefficient.

        Args:
            friction: Friction value (0.0 to 1.0)
        """
        self._friction = max(PHYSICS_COEFF_MIN, min(PHYSICS_COEFF_MAX, friction))

    def get_restitution(self) -> float:
        """Get restitution (bounciness) for physics."""
        return self._restitution

    def set_restitution(self, restitution: float) -> None:
        """Set restitution coefficient.

        Args:
            restitution: Restitution value (0.0 to 1.0)
        """
        self._restitution = max(PHYSICS_COEFF_MIN, min(PHYSICS_COEFF_MAX, restitution))


@dataclass
class RaycastHit:
    """Result of a terrain raycast.

    Attributes:
        hit: Whether the ray hit the terrain
        position: Hit position in world space
        normal: Surface normal at hit point
        distance: Distance from ray origin to hit
        component_index: Index of hit landscape component
    """
    hit: bool = False
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    distance: float = float('inf')
    component_index: int = -1


class TerrainActor:
    """High-level terrain manager composing multiple components.

    TerrainActor manages a collection of LandscapeComponents to form
    a complete terrain. It handles:
    - Component organization and spatial queries
    - LOD updates based on camera position
    - Raycasting against terrain
    - Origin offset for large worlds (floating origin)

    Attributes:
        components: List of landscape components
        total_bounds: Combined bounds of all components
        origin_offset: World origin offset for floating origin support
    """

    def __init__(self):
        """Initialize terrain actor."""
        self.components: List[LandscapeComponent] = []
        self._total_bounds: Optional[AABB] = None
        self._origin_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._component_grid: Dict[Tuple[int, int], LandscapeComponent] = {}
        self._proxies: List[TerrainProxy] = []

    @property
    def total_bounds(self) -> AABB:
        """Get combined bounds of all components."""
        if self._total_bounds is None:
            self._recalculate_bounds()
        return self._total_bounds or (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def origin_offset(self) -> Tuple[float, float, float]:
        """Get current origin offset."""
        return self._origin_offset

    def _recalculate_bounds(self) -> None:
        """Recalculate total bounds from all components."""
        if not self.components:
            self._total_bounds = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return

        min_x = float('inf')
        min_y = float('inf')
        min_z = float('inf')
        max_x = float('-inf')
        max_y = float('-inf')
        max_z = float('-inf')

        for comp in self.components:
            b = comp.bounds
            min_x = min(min_x, b[0])
            min_y = min(min_y, b[1])
            min_z = min(min_z, b[2])
            max_x = max(max_x, b[3])
            max_y = max(max_y, b[4])
            max_z = max(max_z, b[5])

        self._total_bounds = (min_x, min_y, min_z, max_x, max_y, max_z)

    def add_component(self, component: LandscapeComponent) -> int:
        """Add a landscape component.

        Args:
            component: Component to add

        Returns:
            Index of added component
        """
        index = len(self.components)
        self.components.append(component)
        self._total_bounds = None  # Invalidate cached bounds

        # Add to spatial grid if patch is available
        if component.patch is not None:
            key = (component.patch.patch_x, component.patch.patch_y)
            self._component_grid[key] = component

        # Create collision proxy
        if component.patch is not None and component.patch.heightfield is not None:
            proxy = TerrainProxy(
                bounds=component.bounds,
                heightfield_ref=component.patch.heightfield
            )
            self._proxies.append(proxy)

        return index

    def remove_component(self, index: int) -> bool:
        """Remove a landscape component by index.

        Args:
            index: Index of component to remove

        Returns:
            True if removed, False if index invalid.
        """
        if index < 0 or index >= len(self.components):
            return False

        component = self.components[index]

        # Remove from spatial grid
        if component.patch is not None:
            key = (component.patch.patch_x, component.patch.patch_y)
            self._component_grid.pop(key, None)

        self.components.pop(index)
        self._total_bounds = None

        # Note: proxy list would need proper management in production code
        return True

    def get_component(self, index: int) -> Optional[LandscapeComponent]:
        """Get component by index.

        Args:
            index: Component index

        Returns:
            Component or None if index invalid.
        """
        if 0 <= index < len(self.components):
            return self.components[index]
        return None

    def get_component_at(self, x: float, z: float) -> Optional[LandscapeComponent]:
        """Get component containing a world position.

        Args:
            x: World X position
            z: World Z position

        Returns:
            Component at position, or None if not found.
        """
        # Adjust for origin offset
        adj_x = x + self._origin_offset[0]
        adj_z = z + self._origin_offset[2]

        for component in self.components:
            bounds = component.bounds
            if (bounds[0] <= adj_x <= bounds[3] and
                bounds[2] <= adj_z <= bounds[5]):
                return component

        return None

    def get_component_by_patch(self, patch_x: int, patch_y: int) -> Optional[LandscapeComponent]:
        """Get component by patch grid coordinates.

        Args:
            patch_x: Patch X index
            patch_y: Patch Y index

        Returns:
            Component at grid position, or None if not found.
        """
        return self._component_grid.get((patch_x, patch_y))

    def set_origin_offset(self, x: float, y: float, z: float) -> None:
        """Set origin offset for floating origin support.

        In large worlds, floating point precision issues occur far from origin.
        The origin offset allows shifting the coordinate system while maintaining
        precision.

        Args:
            x, y, z: New origin offset
        """
        self._origin_offset = (x, y, z)

    def raycast(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        max_distance: float = DEFAULT_RAYCAST_MAX_DISTANCE
    ) -> RaycastHit:
        """Cast a ray against the terrain.

        Args:
            origin: Ray origin point (x, y, z)
            direction: Ray direction (will be normalized)
            max_distance: Maximum raycast distance

        Returns:
            RaycastHit with results
        """
        result = RaycastHit()

        # Normalize direction
        dx, dy, dz = direction
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < RAY_DIRECTION_EPSILON:
            return result

        dx /= length
        dy /= length
        dz /= length

        # Adjust origin for offset
        ox = origin[0] + self._origin_offset[0]
        oy = origin[1] + self._origin_offset[1]
        oz = origin[2] + self._origin_offset[2]

        best_distance = max_distance

        # Test against each proxy
        for i, proxy in enumerate(self._proxies):
            hit = self._raycast_heightfield(
                proxy, ox, oy, oz, dx, dy, dz, best_distance
            )
            if hit is not None and hit[0] < best_distance:
                best_distance = hit[0]
                result.hit = True
                result.distance = hit[0]
                result.position = hit[1]
                result.normal = hit[2]
                result.component_index = i

        return result

    def _raycast_heightfield(
        self,
        proxy: TerrainProxy,
        ox: float, oy: float, oz: float,
        dx: float, dy: float, dz: float,
        max_dist: float
    ) -> Optional[Tuple[float, Tuple[float, float, float], Tuple[float, float, float]]]:
        """Internal heightfield raycast using ray marching.

        Returns: (distance, position, normal) or None
        """
        if proxy.heightfield_ref is None:
            return None

        # Simple ray marching implementation
        step_size = proxy.heightfield_ref.config.scale * RAYCAST_STEP_MULTIPLIER
        steps = int(max_dist / step_size)

        for i in range(steps):
            t = i * step_size
            px = ox + dx * t
            py = oy + dy * t
            pz = oz + dz * t

            height = proxy.get_height_at(px, pz)
            if height is not None and py <= height:
                # Hit detected - refine with smaller steps
                # Binary search for more precise hit
                t_min = max(0, (i - 1) * step_size)
                t_max = t

                for _ in range(RAYCAST_BINARY_SEARCH_ITERATIONS):
                    t_mid = (t_min + t_max) / 2.0
                    px = ox + dx * t_mid
                    py = oy + dy * t_mid
                    pz = oz + dz * t_mid

                    height = proxy.get_height_at(px, pz)
                    if height is not None and py <= height:
                        t_max = t_mid
                    else:
                        t_min = t_mid

                final_t = (t_min + t_max) / 2.0
                final_x = ox + dx * final_t
                final_y = oy + dy * final_t
                final_z = oz + dz * final_t

                # Get height and normal at final position
                final_height = proxy.get_height_at(final_x, final_z)
                if final_height is not None:
                    normal = proxy.get_normal_at(final_x, final_z) or (0.0, 1.0, 0.0)
                    # Adjust position back for origin offset
                    adj_x = final_x - self._origin_offset[0]
                    adj_y = final_height - self._origin_offset[1]
                    adj_z = final_z - self._origin_offset[2]
                    return (final_t, (adj_x, adj_y, adj_z), normal)

        return None

    def update_lod(self, camera_position: Tuple[float, float, float]) -> None:
        """Update LOD levels for all components based on camera position.

        Args:
            camera_position: Camera world position (x, y, z)
        """
        cam_x = camera_position[0] + self._origin_offset[0]
        cam_y = camera_position[1] + self._origin_offset[1]
        cam_z = camera_position[2] + self._origin_offset[2]

        for component in self.components:
            if component.patch is None:
                continue

            # Calculate distance to patch center
            distance = component.patch.distance_to_point(cam_x, cam_y, cam_z)

            # Select LOD
            base_lod = component.patch.select_lod(distance)

            # Apply component LOD bias
            component.patch.current_lod = component.get_adjusted_lod(
                base_lod,
                component.patch.lod_levels
            )

    def get_height_at(self, x: float, z: float) -> Optional[float]:
        """Get terrain height at world position.

        Args:
            x: World X position
            z: World Z position

        Returns:
            Height at position, or None if outside terrain.
        """
        component = self.get_component_at(x, z)
        if component is None or component.patch is None:
            return None

        return component.patch.get_height_at_world(
            x + self._origin_offset[0],
            z + self._origin_offset[2]
        )

    def get_normal_at(self, x: float, z: float) -> Optional[Tuple[float, float, float]]:
        """Get terrain normal at world position.

        Args:
            x: World X position
            z: World Z position

        Returns:
            Normal vector, or None if outside terrain.
        """
        component = self.get_component_at(x, z)
        if component is None or component.patch is None:
            return None

        return component.patch.get_normal_at_world(
            x + self._origin_offset[0],
            z + self._origin_offset[2]
        )

    def get_component_count(self) -> int:
        """Get number of components."""
        return len(self.components)

    def get_visible_components(
        self,
        frustum_planes: Optional[List[Tuple[float, float, float, float]]] = None
    ) -> List[LandscapeComponent]:
        """Get list of visible components.

        Args:
            frustum_planes: Optional frustum planes for culling

        Returns:
            List of visible components.
        """
        if frustum_planes is None:
            return [c for c in self.components if c.visible]

        return [
            c for c in self.components
            if c.visible and c.intersects_frustum(frustum_planes)
        ]

    def clear(self) -> None:
        """Remove all components."""
        self.components.clear()
        self._component_grid.clear()
        self._proxies.clear()
        self._total_bounds = None

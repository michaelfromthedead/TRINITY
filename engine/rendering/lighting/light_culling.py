"""Clustered light culling using 3D froxels.

Implements the clustered light culling system from Section 6.4 of RENDERING_CONTEXT.md:
- Screen divided into 3D froxels (frustum voxels)
- Each froxel stores a list of affecting lights
- Supports exponential depth slicing for better distribution
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from engine.core.math.geometry import AABB, Frustum, Plane, Sphere
from engine.core.math.mat import Mat4
from engine.core.math.vec import Vec2, Vec3, Vec4

if TYPE_CHECKING:
    from engine.rendering.lighting.light_types import (
        AnyLight,
        DirectionalLight,
        PointLight,
        SpotLight,
    )


@dataclass
class FroxelBounds:
    """Bounding volume for a single froxel.

    Attributes:
        aabb: Axis-aligned bounding box in view space
        near_plane: Near plane distance
        far_plane: Far plane distance
        min_uv: Minimum screen UV coordinates
        max_uv: Maximum screen UV coordinates
    """
    aabb: AABB
    near_plane: float
    far_plane: float
    min_uv: Vec2
    max_uv: Vec2


@dataclass
class Froxel:
    """A 3D frustum voxel (froxel) for light culling.

    Froxels are the basic unit of the clustered light culling system.
    Each froxel represents a portion of the view frustum and maintains
    a list of lights that potentially affect it.

    Attributes:
        x: X index in the froxel grid
        y: Y index in the froxel grid
        z: Z index (depth slice) in the froxel grid
        bounds: Bounding volume of this froxel
        light_indices: Indices of lights affecting this froxel
    """
    x: int
    y: int
    z: int
    bounds: Optional[FroxelBounds] = None
    light_indices: list[int] = field(default_factory=list)

    @property
    def index_3d(self) -> tuple[int, int, int]:
        """Return the 3D index tuple."""
        return (self.x, self.y, self.z)

    def clear_lights(self) -> None:
        """Clear the light list for this froxel."""
        self.light_indices.clear()

    def add_light(self, light_index: int) -> None:
        """Add a light to this froxel's list."""
        if light_index not in self.light_indices:
            self.light_indices.append(light_index)


@dataclass
class FroxelGridConfig:
    """Configuration for the froxel grid.

    Attributes:
        tiles_x: Number of tiles in X direction
        tiles_y: Number of tiles in Y direction
        slices_z: Number of depth slices
        use_exponential_depth: Use exponential depth distribution
        near_plane: Camera near plane distance
        far_plane: Camera far plane distance
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
    """
    tiles_x: int = 16
    tiles_y: int = 9
    slices_z: int = 24
    use_exponential_depth: bool = True
    near_plane: float = 0.1
    far_plane: float = 1000.0
    screen_width: int = 1920
    screen_height: int = 1080

    @property
    def tile_size_x(self) -> int:
        """Tile size in pixels in X direction."""
        return self.screen_width // self.tiles_x

    @property
    def tile_size_y(self) -> int:
        """Tile size in pixels in Y direction."""
        return self.screen_height // self.tiles_y

    @property
    def total_froxels(self) -> int:
        """Total number of froxels in the grid."""
        return self.tiles_x * self.tiles_y * self.slices_z


class FroxelGrid:
    """3D grid of froxels for clustered light culling.

    The froxel grid divides the view frustum into a 3D array of cells.
    Each cell (froxel) stores indices of lights that may affect pixels
    within that cell.

    The depth slicing uses exponential distribution to allocate more
    slices near the camera where precision matters most.
    """

    def __init__(self, config: FroxelGridConfig) -> None:
        """Initialize the froxel grid.

        Args:
            config: Grid configuration parameters
        """
        self.config = config
        self._froxels: list[list[list[Froxel]]] = []
        self._depth_slices: list[float] = []
        self._view_matrix: Mat4 = Mat4.identity()
        self._proj_matrix: Mat4 = Mat4.identity()
        self._inverse_proj: Mat4 = Mat4.identity()

        self._build_grid()

    def _build_grid(self) -> None:
        """Build the froxel grid structure."""
        self._compute_depth_slices()
        self._froxels = []

        for z in range(self.config.slices_z):
            slice_plane = []
            for y in range(self.config.tiles_y):
                row = []
                for x in range(self.config.tiles_x):
                    froxel = Froxel(x=x, y=y, z=z)
                    row.append(froxel)
                slice_plane.append(row)
            self._froxels.append(slice_plane)

    def _compute_depth_slices(self) -> None:
        """Compute depth slice boundaries.

        Uses either linear or exponential distribution based on config.
        Exponential distribution provides better precision near the camera.
        """
        self._depth_slices = []
        near = self.config.near_plane
        far = self.config.far_plane

        for i in range(self.config.slices_z + 1):
            t = i / self.config.slices_z

            if self.config.use_exponential_depth:
                # Exponential distribution: more slices near camera
                # z = near * (far/near)^t
                depth = near * math.pow(far / near, t)
            else:
                # Linear distribution
                depth = near + (far - near) * t

            self._depth_slices.append(depth)

    def update_matrices(self, view: Mat4, proj: Mat4) -> None:
        """Update view and projection matrices.

        Args:
            view: View matrix
            proj: Projection matrix
        """
        self._view_matrix = view
        self._proj_matrix = proj
        self._inverse_proj = proj.inverse()
        self._update_froxel_bounds()

    def _update_froxel_bounds(self) -> None:
        """Recompute froxel bounds based on current matrices."""
        for z in range(self.config.slices_z):
            near_z = self._depth_slices[z]
            far_z = self._depth_slices[z + 1]

            for y in range(self.config.tiles_y):
                for x in range(self.config.tiles_x):
                    froxel = self._froxels[z][y][x]
                    froxel.bounds = self._compute_froxel_bounds(x, y, near_z, far_z)

    def _compute_froxel_bounds(
        self, tile_x: int, tile_y: int, near_z: float, far_z: float
    ) -> FroxelBounds:
        """Compute bounds for a single froxel.

        Args:
            tile_x: X tile index
            tile_y: Y tile index
            near_z: Near depth of the froxel
            far_z: Far depth of the froxel

        Returns:
            FroxelBounds for this froxel
        """
        # Compute UV bounds
        min_u = tile_x / self.config.tiles_x
        max_u = (tile_x + 1) / self.config.tiles_x
        min_v = tile_y / self.config.tiles_y
        max_v = (tile_y + 1) / self.config.tiles_y

        # Convert to NDC (-1 to 1)
        min_ndc_x = min_u * 2.0 - 1.0
        max_ndc_x = max_u * 2.0 - 1.0
        min_ndc_y = min_v * 2.0 - 1.0
        max_ndc_y = max_v * 2.0 - 1.0

        # Compute view space corners at near and far planes
        corners = []
        for z_depth in [near_z, far_z]:
            for ndc_y in [min_ndc_y, max_ndc_y]:
                for ndc_x in [min_ndc_x, max_ndc_x]:
                    # Unproject from NDC to view space
                    # This is simplified - actual implementation would use proper unprojection
                    view_pos = self._unproject_point(ndc_x, ndc_y, z_depth)
                    corners.append(view_pos)

        # Compute AABB from corners
        min_corner = Vec3(
            min(c.x for c in corners),
            min(c.y for c in corners),
            min(c.z for c in corners),
        )
        max_corner = Vec3(
            max(c.x for c in corners),
            max(c.y for c in corners),
            max(c.z for c in corners),
        )

        return FroxelBounds(
            aabb=AABB(min_corner, max_corner),
            near_plane=near_z,
            far_plane=far_z,
            min_uv=Vec2(min_u, min_v),
            max_uv=Vec2(max_u, max_v),
        )

    def _unproject_point(self, ndc_x: float, ndc_y: float, view_z: float) -> Vec3:
        """Unproject a point from NDC to view space.

        Args:
            ndc_x: X in normalized device coordinates [-1, 1]
            ndc_y: Y in normalized device coordinates [-1, 1]
            view_z: Depth in view space (negative, towards camera)

        Returns:
            Point in view space
        """
        # For a perspective projection with vertical FOV and aspect ratio:
        # x_view = x_ndc * z_view * tan(fov/2) * aspect
        # y_view = y_ndc * z_view * tan(fov/2)
        # This is a simplified calculation
        inv_m = self._inverse_proj.m
        # Use a simplified approach for view space reconstruction
        x = ndc_x * abs(view_z) * inv_m[0]
        y = ndc_y * abs(view_z) * inv_m[5]
        return Vec3(x, y, -view_z)  # View space is typically -Z forward

    def get_froxel(self, x: int, y: int, z: int) -> Optional[Froxel]:
        """Get a froxel by its 3D index.

        Args:
            x: X index
            y: Y index
            z: Z (depth) index

        Returns:
            The froxel at the given index, or None if out of bounds
        """
        if (0 <= x < self.config.tiles_x and
            0 <= y < self.config.tiles_y and
            0 <= z < self.config.slices_z):
            return self._froxels[z][y][x]
        return None

    def get_froxel_at_view_position(self, view_pos: Vec3) -> Optional[Froxel]:
        """Get the froxel containing a view-space position.

        Args:
            view_pos: Position in view space

        Returns:
            The froxel containing the position, or None if outside grid
        """
        depth = -view_pos.z  # View space has -Z forward

        if depth < self.config.near_plane or depth > self.config.far_plane:
            return None

        # Find depth slice
        z = 0
        for i in range(self.config.slices_z):
            if depth >= self._depth_slices[i] and depth < self._depth_slices[i + 1]:
                z = i
                break

        # Project to screen and find tile
        # Simplified - actual implementation would use projection matrix
        if depth > 0:
            screen_x = (view_pos.x / depth + 0.5) * self.config.screen_width
            screen_y = (view_pos.y / depth + 0.5) * self.config.screen_height
            x = int(screen_x / self.config.tile_size_x)
            y = int(screen_y / self.config.tile_size_y)
            return self.get_froxel(x, y, z)

        return None

    def get_depth_slice(self, depth: float) -> int:
        """Get the depth slice index for a given depth value.

        Args:
            depth: Depth value (positive)

        Returns:
            Slice index, clamped to valid range
        """
        if depth <= self.config.near_plane:
            return 0
        if depth >= self.config.far_plane:
            return self.config.slices_z - 1

        for i in range(self.config.slices_z):
            if depth < self._depth_slices[i + 1]:
                return i

        return self.config.slices_z - 1

    def clear_all_lights(self) -> None:
        """Clear light lists from all froxels."""
        for z_slice in self._froxels:
            for y_row in z_slice:
                for froxel in y_row:
                    froxel.clear_lights()

    def iterate_froxels(self):
        """Iterate over all froxels in the grid.

        Yields:
            Each froxel in the grid
        """
        for z_slice in self._froxels:
            for y_row in z_slice:
                for froxel in y_row:
                    yield froxel


@dataclass
class LightList:
    """Per-froxel light list for GPU upload.

    This structure is designed to be uploaded to the GPU for
    use in the lighting shader.

    Attributes:
        offset: Offset into the global light index buffer
        count: Number of lights in this froxel
    """
    offset: int = 0
    count: int = 0


class ClusteredLightCuller:
    """Assigns lights to froxels using clustered culling.

    The culler tests each light against froxel bounds and builds
    per-froxel light lists for efficient shading.
    """

    def __init__(self, grid: FroxelGrid) -> None:
        """Initialize the light culler.

        Args:
            grid: The froxel grid to use for culling
        """
        self.grid = grid
        self._lights: list[AnyLight] = []
        self._light_index_buffer: list[int] = []
        self._light_lists: list[LightList] = []

    def set_lights(self, lights: list[AnyLight]) -> None:
        """Set the list of lights to cull.

        Args:
            lights: List of lights in the scene
        """
        self._lights = lights

    def cull(self) -> None:
        """Perform light culling against the froxel grid.

        This method tests each light against each froxel and builds
        the per-froxel light lists.
        """
        self.grid.clear_all_lights()

        for light_index, light in enumerate(self._lights):
            if not light.enabled:
                continue

            self._cull_light(light_index, light)

        self._build_light_lists()

    def _cull_light(self, light_index: int, light: AnyLight) -> None:
        """Cull a single light against the froxel grid.

        Args:
            light_index: Index of the light in the light list
            light: The light to cull
        """
        from engine.rendering.lighting.light_types import (
            DirectionalLight,
            DiskAreaLight,
            IESLight,
            PointLight,
            RectAreaLight,
            SkyLight,
            SpotLight,
        )

        # Directional and sky lights affect all froxels
        if isinstance(light, (DirectionalLight, SkyLight)):
            for froxel in self.grid.iterate_froxels():
                froxel.add_light(light_index)
            return

        # Point lights - test sphere against froxel AABBs
        if isinstance(light, PointLight):
            light_sphere = Sphere(light.position, light.radius)
            self._cull_sphere_light(light_index, light_sphere)
            return

        # Spot lights - test cone against froxel AABBs
        if isinstance(light, SpotLight):
            # Use bounding sphere approximation for spot cone
            # More accurate cone culling could be implemented
            # Prevent division by zero when outer_angle approaches 90 degrees
            cos_outer = math.cos(light.outer_angle)
            min_cos = 0.001  # FroxelConstants.MIN_COS_VALUE
            safe_cos = max(cos_outer, min_cos)
            bound_radius = light.radius / safe_cos
            bound_center = light.position + light.direction * (light.radius * 0.5)
            light_sphere = Sphere(bound_center, bound_radius)
            self._cull_sphere_light(light_index, light_sphere)
            return

        # IES lights - similar to point lights
        if isinstance(light, IESLight):
            light_sphere = Sphere(light.position, light.radius)
            self._cull_sphere_light(light_index, light_sphere)
            return

        # Area lights - use conservative bounding sphere
        if isinstance(light, RectAreaLight):
            diag = math.sqrt(light.width ** 2 + light.height ** 2) * 0.5
            # Estimate influence radius based on intensity
            # Using sqrt to scale influence reasonably with intensity
            # 10.0 is AREA_LIGHT_INFLUENCE_MULTIPLIER from LightConstants
            area_light_influence_multiplier = 10.0
            influence_radius = math.sqrt(max(0.0, light.intensity)) * area_light_influence_multiplier
            light_sphere = Sphere(light.position, max(diag, influence_radius))
            self._cull_sphere_light(light_index, light_sphere)
            return

        if isinstance(light, DiskAreaLight):
            area_light_influence_multiplier = 10.0
            influence_radius = math.sqrt(max(0.0, light.intensity)) * area_light_influence_multiplier
            light_sphere = Sphere(light.position, max(light.disk_radius, influence_radius))
            self._cull_sphere_light(light_index, light_sphere)
            return

    def _cull_sphere_light(self, light_index: int, sphere: Sphere) -> None:
        """Cull a spherical light volume against the froxel grid.

        Args:
            light_index: Index of the light
            sphere: Bounding sphere of the light's influence
        """
        # Transform sphere center to view space
        view_center = self.grid._view_matrix.transform_point(sphere.center)
        depth = -view_center.z

        # Quick rejection: entirely behind camera or beyond far plane
        if depth + sphere.radius < self.grid.config.near_plane:
            return
        if depth - sphere.radius > self.grid.config.far_plane:
            return

        # Find depth range
        min_depth = max(depth - sphere.radius, self.grid.config.near_plane)
        max_depth = min(depth + sphere.radius, self.grid.config.far_plane)

        min_z = self.grid.get_depth_slice(min_depth)
        max_z = self.grid.get_depth_slice(max_depth)

        # For each potentially affected depth slice
        for z in range(min_z, max_z + 1):
            for y in range(self.grid.config.tiles_y):
                for x in range(self.grid.config.tiles_x):
                    froxel = self.grid.get_froxel(x, y, z)
                    if froxel and froxel.bounds:
                        if self._sphere_aabb_intersect(sphere, froxel.bounds.aabb):
                            froxel.add_light(light_index)

    def _sphere_aabb_intersect(self, sphere: Sphere, aabb: AABB) -> bool:
        """Test if a sphere intersects an AABB.

        Args:
            sphere: The sphere to test
            aabb: The AABB to test against

        Returns:
            True if the sphere and AABB intersect
        """
        # Find closest point on AABB to sphere center
        closest = Vec3(
            max(aabb.min.x, min(sphere.center.x, aabb.max.x)),
            max(aabb.min.y, min(sphere.center.y, aabb.max.y)),
            max(aabb.min.z, min(sphere.center.z, aabb.max.z)),
        )

        # Check if closest point is within sphere radius
        distance_sq = (closest - sphere.center).length_squared()
        return distance_sq <= sphere.radius ** 2

    def _build_light_lists(self) -> None:
        """Build the light index buffer and light lists.

        Creates a compact representation suitable for GPU upload.
        """
        self._light_index_buffer = []
        self._light_lists = []

        for froxel in self.grid.iterate_froxels():
            light_list = LightList(
                offset=len(self._light_index_buffer),
                count=len(froxel.light_indices),
            )
            self._light_lists.append(light_list)
            self._light_index_buffer.extend(froxel.light_indices)

    def get_light_index_buffer(self) -> list[int]:
        """Get the global light index buffer.

        Returns:
            List of light indices
        """
        return self._light_index_buffer

    def get_light_lists(self) -> list[LightList]:
        """Get the per-froxel light lists.

        Returns:
            List of LightList structures, one per froxel
        """
        return self._light_lists

    def get_froxel_light_count(self, x: int, y: int, z: int) -> int:
        """Get the number of lights affecting a froxel.

        Args:
            x: X index
            y: Y index
            z: Z index

        Returns:
            Number of lights in the froxel
        """
        froxel = self.grid.get_froxel(x, y, z)
        return len(froxel.light_indices) if froxel else 0

    def get_max_lights_per_froxel(self) -> int:
        """Get the maximum number of lights in any froxel.

        Returns:
            Maximum light count across all froxels
        """
        return max(
            (len(froxel.light_indices) for froxel in self.grid.iterate_froxels()),
            default=0
        )

    def get_average_lights_per_froxel(self) -> float:
        """Get the average number of lights per froxel.

        Returns:
            Average light count
        """
        total_lights = sum(
            len(froxel.light_indices) for froxel in self.grid.iterate_froxels()
        )
        return total_lights / max(self.grid.config.total_froxels, 1)

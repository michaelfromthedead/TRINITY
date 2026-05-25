"""Shadow mapping systems.

Implements shadow mapping from Section 6.4 of RENDERING_CONTEXT.md:
- Cascaded Shadow Maps (CSM) for directional lights
- Cube Shadow Maps for point lights
- Spot Shadow Maps for spot lights
- Shadow Atlas for packing multiple shadow maps
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

from engine.core.math.geometry import AABB, Frustum, Plane
from engine.core.math.mat import Mat4
from engine.core.math.vec import Vec2, Vec3, Vec4

if TYPE_CHECKING:
    from engine.rendering.lighting.light_types import (
        DirectionalLight,
        PointLight,
        SpotLight,
        ShadowCasterConfig,
    )


class ShadowMapType(Enum):
    """Types of shadow maps."""
    CASCADED = auto()       # CSM for directional lights
    CUBE = auto()           # Cubemap for point lights
    SPOT = auto()           # Single frustum for spot lights
    VIRTUAL = auto()        # Virtual shadow maps (page-based)


@dataclass
class ShadowMapConfig:
    """Configuration for shadow map rendering.

    Attributes:
        resolution: Base resolution of the shadow map
        depth_bias: Constant depth bias to prevent shadow acne
        slope_bias: Slope-scaled depth bias
        normal_bias: Bias along surface normal
        filter_size: Size of the PCF filter kernel
        softness: Shadow softness for PCSS
    """
    resolution: int = 2048
    depth_bias: float = 0.0001
    slope_bias: float = 0.001
    normal_bias: float = 0.02
    filter_size: int = 3
    softness: float = 1.0


@dataclass
class ShadowMap(ABC):
    """Base class for all shadow map types.

    Attributes:
        config: Shadow map configuration
        light_id: ID of the light this shadow map belongs to
        dirty: Whether the shadow map needs to be re-rendered
    """
    config: ShadowMapConfig = field(default_factory=ShadowMapConfig)
    light_id: int = 0
    dirty: bool = True

    # GPU resource handles (would be actual GPU resources in production)
    _texture_handle: int = 0
    _depth_handle: int = 0

    @property
    @abstractmethod
    def shadow_type(self) -> ShadowMapType:
        """Return the type of this shadow map."""
        ...

    @abstractmethod
    def get_resolution(self) -> tuple[int, int]:
        """Get the resolution of this shadow map.

        Returns:
            Tuple of (width, height)
        """
        ...

    @abstractmethod
    def get_view_projection_matrix(self, face: int = 0) -> Mat4:
        """Get the view-projection matrix for rendering into this shadow map.

        Args:
            face: Face index for cube maps (0-5), cascade index for CSM

        Returns:
            Combined view-projection matrix
        """
        ...

    def mark_dirty(self) -> None:
        """Mark this shadow map as needing re-render."""
        self.dirty = True

    def clear_dirty(self) -> None:
        """Clear the dirty flag after rendering."""
        self.dirty = False


@dataclass
class CascadeData:
    """Data for a single cascade in CSM.

    Attributes:
        split_depth: Distance at which this cascade ends
        view_matrix: View matrix for this cascade
        projection_matrix: Projection matrix for this cascade
        world_to_shadow: Combined world-to-shadow space matrix
        texel_size: Size of a shadow map texel in world units
    """
    split_depth: float = 0.0
    view_matrix: Mat4 = field(default_factory=Mat4.identity)
    projection_matrix: Mat4 = field(default_factory=Mat4.identity)
    world_to_shadow: Mat4 = field(default_factory=Mat4.identity)
    texel_size: float = 0.0


@dataclass
class CascadedShadowMap(ShadowMap):
    """Cascaded Shadow Map for directional lights.

    CSM divides the view frustum into multiple cascades, each with
    its own shadow map. This provides high resolution near the camera
    while maintaining coverage at distance.

    Attributes:
        cascade_count: Number of cascades (1-4)
        cascade_data: Per-cascade rendering data
        cascade_blend_range: Range over which cascades blend
        stabilize_cascades: Whether to use cascade stabilization
    """
    cascade_count: int = 4
    cascade_data: list[CascadeData] = field(default_factory=list)
    cascade_blend_range: float = 2.0
    stabilize_cascades: bool = True

    # Light reference
    _light_direction: Vec3 = field(default_factory=lambda: Vec3(0, -1, 0))
    _cascade_distances: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 1 <= self.cascade_count <= 4:
            raise ValueError("cascade_count must be between 1 and 4")
        # Initialize cascade data
        if not self.cascade_data:
            self.cascade_data = [CascadeData() for _ in range(self.cascade_count)]

    @property
    def shadow_type(self) -> ShadowMapType:
        return ShadowMapType.CASCADED

    def get_resolution(self) -> tuple[int, int]:
        """Each cascade has the same resolution."""
        return (self.config.resolution, self.config.resolution)

    def get_view_projection_matrix(self, face: int = 0) -> Mat4:
        """Get the view-projection for a specific cascade.

        Args:
            face: Cascade index (0 to cascade_count-1)

        Returns:
            Combined view-projection matrix for the cascade
        """
        if 0 <= face < len(self.cascade_data):
            cascade = self.cascade_data[face]
            return cascade.projection_matrix @ cascade.view_matrix
        return Mat4.identity()

    def configure_for_light(
        self,
        light: DirectionalLight,
        camera_view: Mat4,
        camera_proj: Mat4,
        camera_near: float,
        camera_far: float,
    ) -> None:
        """Configure cascades for a directional light.

        Args:
            light: The directional light
            camera_view: Camera view matrix
            camera_proj: Camera projection matrix
            camera_near: Camera near plane
            camera_far: Camera far plane
        """
        self._light_direction = light.direction.normalized()
        self._cascade_distances = light.cascade_distances[:self.cascade_count]

        # Compute cascade splits using practical split scheme
        splits = self._compute_cascade_splits(camera_near, camera_far)

        for i, cascade in enumerate(self.cascade_data):
            near = camera_near if i == 0 else splits[i - 1]
            far = splits[i]
            cascade.split_depth = far

            # Compute frustum corners in world space
            frustum_corners = self._get_frustum_corners(
                camera_view, camera_proj, near, far
            )

            # Compute cascade matrices
            self._compute_cascade_matrices(cascade, frustum_corners)

    def _compute_cascade_splits(
        self, near: float, far: float
    ) -> list[float]:
        """Compute cascade split distances using logarithmic scheme.

        Args:
            near: Camera near plane
            far: Camera far plane

        Returns:
            List of split distances
        """
        # Use provided cascade distances if available
        if self._cascade_distances and len(self._cascade_distances) >= self.cascade_count:
            return self._cascade_distances[:self.cascade_count]

        # Compute using logarithmic distribution
        splits = []
        lambda_param = 0.75  # Blend factor between linear and logarithmic

        for i in range(self.cascade_count):
            t = (i + 1) / self.cascade_count
            log_split = near * math.pow(far / near, t)
            linear_split = near + (far - near) * t
            split = lambda_param * log_split + (1 - lambda_param) * linear_split
            splits.append(split)

        return splits

    def _get_frustum_corners(
        self,
        view: Mat4,
        proj: Mat4,
        near: float,
        far: float,
    ) -> list[Vec3]:
        """Get the 8 corners of a view frustum in world space.

        Args:
            view: View matrix
            proj: Projection matrix
            near: Near plane distance
            far: Far plane distance

        Returns:
            List of 8 corner positions
        """
        inv_view_proj = (proj @ view).inverse()

        corners = []
        for z in [0.0, 1.0]:  # Near and far planes in NDC
            for y in [-1.0, 1.0]:
                for x in [-1.0, 1.0]:
                    # NDC coordinates
                    ndc = Vec4(x, y, z * 2.0 - 1.0, 1.0)

                    # Transform to world space
                    world = Vec4(
                        inv_view_proj.m[0] * ndc.x + inv_view_proj.m[4] * ndc.y +
                        inv_view_proj.m[8] * ndc.z + inv_view_proj.m[12] * ndc.w,
                        inv_view_proj.m[1] * ndc.x + inv_view_proj.m[5] * ndc.y +
                        inv_view_proj.m[9] * ndc.z + inv_view_proj.m[13] * ndc.w,
                        inv_view_proj.m[2] * ndc.x + inv_view_proj.m[6] * ndc.y +
                        inv_view_proj.m[10] * ndc.z + inv_view_proj.m[14] * ndc.w,
                        inv_view_proj.m[3] * ndc.x + inv_view_proj.m[7] * ndc.y +
                        inv_view_proj.m[11] * ndc.z + inv_view_proj.m[15] * ndc.w,
                    )

                    if abs(world.w) > 1e-6:
                        corners.append(Vec3(
                            world.x / world.w,
                            world.y / world.w,
                            world.z / world.w,
                        ))
                    else:
                        corners.append(Vec3(world.x, world.y, world.z))

        return corners

    def _compute_cascade_matrices(
        self,
        cascade: CascadeData,
        frustum_corners: list[Vec3],
    ) -> None:
        """Compute view and projection matrices for a cascade.

        Args:
            cascade: Cascade data to update
            frustum_corners: World-space frustum corners
        """
        # Compute frustum center
        center = Vec3.zero()
        for corner in frustum_corners:
            center = center + corner
        center = center * (1.0 / len(frustum_corners))

        # Compute up vector (avoid parallel with light direction)
        up = Vec3(0, 1, 0)
        if abs(self._light_direction.dot(up)) > 0.9:
            up = Vec3(1, 0, 0)

        # Compute view matrix looking from light direction
        light_pos = center - self._light_direction * 100.0  # Offset from center
        cascade.view_matrix = Mat4.look_at(light_pos, center, up)

        # Transform corners to light space
        light_corners = [
            cascade.view_matrix.transform_point(c) for c in frustum_corners
        ]

        # Compute bounding box in light space
        min_corner = Vec3(
            min(c.x for c in light_corners),
            min(c.y for c in light_corners),
            min(c.z for c in light_corners),
        )
        max_corner = Vec3(
            max(c.x for c in light_corners),
            max(c.y for c in light_corners),
            max(c.z for c in light_corners),
        )

        # Stabilize cascade if enabled
        if self.stabilize_cascades:
            min_corner, max_corner = self._stabilize_bounds(
                min_corner, max_corner, cascade
            )

        # Create orthographic projection
        cascade.projection_matrix = Mat4.orthographic(
            min_corner.x, max_corner.x,
            min_corner.y, max_corner.y,
            -max_corner.z - 100.0,  # Extend near plane behind
            -min_corner.z + 100.0,  # Extend far plane
        )

        # Compute combined world-to-shadow matrix
        cascade.world_to_shadow = cascade.projection_matrix @ cascade.view_matrix

        # Compute texel size for later use
        world_units_per_texel = (max_corner.x - min_corner.x) / self.config.resolution
        cascade.texel_size = world_units_per_texel

    def _stabilize_bounds(
        self,
        min_corner: Vec3,
        max_corner: Vec3,
        cascade: CascadeData,
    ) -> tuple[Vec3, Vec3]:
        """Stabilize cascade bounds to reduce shadow swimming.

        Args:
            min_corner: Minimum corner of light-space bounds
            max_corner: Maximum corner of light-space bounds
            cascade: Cascade data

        Returns:
            Stabilized min and max corners
        """
        # Round bounds to texel size
        texel_size = (max_corner.x - min_corner.x) / self.config.resolution

        def round_to_texel(v: float) -> float:
            return math.floor(v / texel_size) * texel_size

        return (
            Vec3(
                round_to_texel(min_corner.x),
                round_to_texel(min_corner.y),
                min_corner.z,
            ),
            Vec3(
                round_to_texel(max_corner.x) + texel_size,
                round_to_texel(max_corner.y) + texel_size,
                max_corner.z,
            ),
        )

    def get_cascade_for_depth(self, depth: float) -> int:
        """Get the cascade index for a given view-space depth.

        Args:
            depth: View-space depth (positive)

        Returns:
            Cascade index (0 to cascade_count-1)
        """
        for i, cascade in enumerate(self.cascade_data):
            if depth < cascade.split_depth:
                return i
        return self.cascade_count - 1


@dataclass
class CubeFace:
    """Data for a single face of a cube shadow map.

    Attributes:
        direction: Direction this face looks
        up: Up vector for this face
        view_matrix: View matrix for rendering this face
    """
    direction: Vec3
    up: Vec3
    view_matrix: Mat4 = field(default_factory=Mat4.identity)


# Standard cube map face directions
CUBE_FACE_DIRECTIONS = [
    CubeFace(Vec3(1, 0, 0), Vec3(0, -1, 0)),   # +X
    CubeFace(Vec3(-1, 0, 0), Vec3(0, -1, 0)),  # -X
    CubeFace(Vec3(0, 1, 0), Vec3(0, 0, 1)),    # +Y
    CubeFace(Vec3(0, -1, 0), Vec3(0, 0, -1)),  # -Y
    CubeFace(Vec3(0, 0, 1), Vec3(0, -1, 0)),   # +Z
    CubeFace(Vec3(0, 0, -1), Vec3(0, -1, 0)),  # -Z
]


@dataclass
class CubeShadowMap(ShadowMap):
    """Cube shadow map for point lights.

    Renders 6 faces of a cubemap, one for each direction.

    Attributes:
        position: Light position
        radius: Light radius (used for far plane)
        face_matrices: View matrices for each cube face
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    radius: float = 10.0
    face_matrices: list[Mat4] = field(default_factory=list)

    _near: float = 0.1

    def __post_init__(self) -> None:
        if not self.face_matrices:
            self.face_matrices = [Mat4.identity() for _ in range(6)]
        self._update_face_matrices()

    @property
    def shadow_type(self) -> ShadowMapType:
        return ShadowMapType.CUBE

    def get_resolution(self) -> tuple[int, int]:
        """Each face is a square with the configured resolution."""
        return (self.config.resolution, self.config.resolution)

    def get_view_projection_matrix(self, face: int = 0) -> Mat4:
        """Get view-projection for a specific cube face.

        Args:
            face: Face index (0-5, corresponding to +X, -X, +Y, -Y, +Z, -Z)

        Returns:
            Combined view-projection matrix
        """
        if 0 <= face < 6:
            proj = Mat4.perspective(
                math.pi / 2.0,  # 90 degree FOV
                1.0,           # Square aspect ratio
                self._near,
                self.radius,
            )
            return proj @ self.face_matrices[face]
        return Mat4.identity()

    def configure_for_light(self, light: PointLight) -> None:
        """Configure the cube shadow map for a point light.

        Args:
            light: The point light
        """
        self.position = light.position
        self.radius = light.radius
        self.light_id = light._light_id
        self._update_face_matrices()
        self.mark_dirty()

    def _update_face_matrices(self) -> None:
        """Update view matrices for all cube faces."""
        for i, face_data in enumerate(CUBE_FACE_DIRECTIONS):
            target = self.position + face_data.direction
            self.face_matrices[i] = Mat4.look_at(
                self.position,
                target,
                face_data.up,
            )

    def get_face_direction(self, face: int) -> Vec3:
        """Get the direction vector for a cube face.

        Args:
            face: Face index (0-5)

        Returns:
            Direction vector
        """
        if 0 <= face < 6:
            return CUBE_FACE_DIRECTIONS[face].direction
        return Vec3.forward()


@dataclass
class SpotShadowMap(ShadowMap):
    """Shadow map for spot lights.

    Uses a single frustum matching the spot light cone.

    Attributes:
        position: Light position
        direction: Light direction
        outer_angle: Outer cone angle in radians
        radius: Light radius (far plane)
        view_matrix: View matrix
        projection_matrix: Projection matrix
    """
    position: Vec3 = field(default_factory=Vec3.zero)
    direction: Vec3 = field(default_factory=lambda: Vec3(0, -1, 0))
    outer_angle: float = math.radians(45.0)
    radius: float = 20.0
    view_matrix: Mat4 = field(default_factory=Mat4.identity)
    projection_matrix: Mat4 = field(default_factory=Mat4.identity)

    _near: float = 0.1

    def __post_init__(self) -> None:
        self._update_matrices()

    @property
    def shadow_type(self) -> ShadowMapType:
        return ShadowMapType.SPOT

    def get_resolution(self) -> tuple[int, int]:
        return (self.config.resolution, self.config.resolution)

    def get_view_projection_matrix(self, face: int = 0) -> Mat4:
        """Get the view-projection matrix.

        Args:
            face: Ignored for spot lights

        Returns:
            Combined view-projection matrix
        """
        return self.projection_matrix @ self.view_matrix

    def configure_for_light(self, light: SpotLight) -> None:
        """Configure the shadow map for a spot light.

        Args:
            light: The spot light
        """
        self.position = light.position
        self.direction = light.direction
        self.outer_angle = light.outer_angle
        self.radius = light.radius
        self.light_id = light._light_id
        self._update_matrices()
        self.mark_dirty()

    def _update_matrices(self) -> None:
        """Update view and projection matrices."""
        # Compute up vector
        up = Vec3(0, 1, 0)
        if abs(self.direction.dot(up)) > 0.9:
            up = Vec3(1, 0, 0)

        target = self.position + self.direction
        self.view_matrix = Mat4.look_at(self.position, target, up)

        # Projection with FOV matching cone angle
        fov = self.outer_angle * 2.0
        self.projection_matrix = Mat4.perspective(
            fov,
            1.0,  # Square aspect ratio
            self._near,
            self.radius,
        )


@dataclass
class ShadowAtlasSlot:
    """A slot in the shadow atlas.

    Attributes:
        x: X offset in the atlas (pixels)
        y: Y offset in the atlas (pixels)
        width: Width of the slot (pixels)
        height: Height of the slot (pixels)
        shadow_map: Shadow map assigned to this slot
    """
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    shadow_map: Optional[ShadowMap] = None

    @property
    def uv_offset(self) -> Vec2:
        """Get UV offset for this slot."""
        return Vec2(self.x, self.y)

    @property
    def uv_scale(self) -> Vec2:
        """Get UV scale for this slot."""
        return Vec2(self.width, self.height)


class ShadowAtlas:
    """Atlas for packing multiple shadow maps into a single texture.

    The shadow atlas allows efficient rendering of multiple shadow maps
    by packing them into a single large texture.

    Attributes:
        resolution: Total atlas resolution (square)
        slots: List of atlas slots
    """

    def __init__(self, resolution: int = 4096) -> None:
        """Initialize the shadow atlas.

        Args:
            resolution: Atlas resolution (must be power of 2)
        """
        if resolution <= 0 or (resolution & (resolution - 1)) != 0:
            raise ValueError("resolution must be a positive power of 2")

        self.resolution = resolution
        self.slots: list[ShadowAtlasSlot] = []
        self._free_rects: list[tuple[int, int, int, int]] = []  # x, y, w, h
        self._reset()

    def _reset(self) -> None:
        """Reset the atlas to empty state."""
        self.slots.clear()
        self._free_rects = [(0, 0, self.resolution, self.resolution)]

    def allocate(self, width: int, height: int) -> Optional[ShadowAtlasSlot]:
        """Allocate a slot in the atlas.

        Uses a simple best-fit rectangle packing algorithm.

        Args:
            width: Required width
            height: Required height

        Returns:
            Allocated slot, or None if no space available
        """
        # Find best fitting free rectangle
        best_idx = -1
        best_fit = float('inf')

        for i, (rx, ry, rw, rh) in enumerate(self._free_rects):
            if rw >= width and rh >= height:
                # Score by wasted space
                fit = (rw - width) * (rh - height)
                if fit < best_fit:
                    best_fit = fit
                    best_idx = i

        if best_idx < 0:
            return None

        # Allocate from the best rectangle
        rx, ry, rw, rh = self._free_rects.pop(best_idx)

        slot = ShadowAtlasSlot(x=rx, y=ry, width=width, height=height)
        self.slots.append(slot)

        # Split remaining space
        if rw - width > 0:
            self._free_rects.append((rx + width, ry, rw - width, height))
        if rh - height > 0:
            self._free_rects.append((rx, ry + height, rw, rh - height))

        return slot

    def deallocate(self, slot: ShadowAtlasSlot) -> None:
        """Deallocate a slot from the atlas.

        Args:
            slot: Slot to deallocate
        """
        if slot in self.slots:
            self.slots.remove(slot)
            # Add back to free list (merging could be implemented for efficiency)
            self._free_rects.append((slot.x, slot.y, slot.width, slot.height))

    def allocate_shadow_map(self, shadow_map: ShadowMap) -> Optional[ShadowAtlasSlot]:
        """Allocate space for a shadow map.

        Args:
            shadow_map: Shadow map to allocate space for

        Returns:
            Allocated slot with shadow map assigned
        """
        width, height = shadow_map.get_resolution()
        slot = self.allocate(width, height)
        if slot:
            slot.shadow_map = shadow_map
        return slot

    def get_slot_for_light(self, light_id: int) -> Optional[ShadowAtlasSlot]:
        """Find the atlas slot for a light.

        Args:
            light_id: Light ID to search for

        Returns:
            The slot containing the light's shadow map, or None
        """
        for slot in self.slots:
            if slot.shadow_map and slot.shadow_map.light_id == light_id:
                return slot
        return None

    def get_uv_transform(self, slot: ShadowAtlasSlot) -> tuple[Vec2, Vec2]:
        """Get UV transform for sampling from a slot.

        Args:
            slot: Atlas slot

        Returns:
            Tuple of (offset, scale) for UV transformation
        """
        inv_res = 1.0 / self.resolution
        offset = Vec2(slot.x * inv_res, slot.y * inv_res)
        scale = Vec2(slot.width * inv_res, slot.height * inv_res)
        return offset, scale

    def get_utilization(self) -> float:
        """Get atlas utilization ratio.

        Returns:
            Ratio of used to total space (0.0 to 1.0)
        """
        used_pixels = sum(s.width * s.height for s in self.slots)
        total_pixels = self.resolution * self.resolution
        return used_pixels / total_pixels

    def defragment(self) -> None:
        """Defragment the atlas by repacking all shadow maps.

        This operation invalidates all existing slots and should trigger
        re-rendering of all shadow maps.
        """
        # Save current shadow maps
        shadow_maps = [s.shadow_map for s in self.slots if s.shadow_map]

        # Reset atlas
        self._reset()

        # Re-allocate in order of size (largest first for better packing)
        shadow_maps.sort(
            key=lambda sm: sm.get_resolution()[0] * sm.get_resolution()[1] if sm else 0,
            reverse=True,
        )

        for shadow_map in shadow_maps:
            if shadow_map:
                slot = self.allocate_shadow_map(shadow_map)
                if slot:
                    shadow_map.mark_dirty()

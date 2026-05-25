"""
GPU Culling Pipeline for GPU-driven rendering.

Implements frustum culling, HZB occlusion culling, and distance culling
on the GPU to efficiently cull instances before rendering.

Pipeline: All Instances -> Frustum Cull -> Occlusion Cull (HZB) -> Distance Cull
    -> Output: Visible instance indices for indirect draw

References:
- RENDERING_CONTEXT.md Section 6.2 GPU-Driven Rendering Pipeline
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Sequence


# =============================================================================
# CULLING CONSTANTS
# =============================================================================


class CullingConstants:
    """Constants used throughout the culling pipeline."""
    # Floating point comparison epsilon
    EPSILON: float = 1e-8

    # Default camera parameters
    DEFAULT_NEAR_PLANE: float = 0.1
    DEFAULT_FAR_PLANE: float = 1000.0

    # Small feature culling defaults
    DEFAULT_MIN_SCREEN_SIZE: float = 2.0
    DEFAULT_SCREEN_WIDTH: int = 1920
    DEFAULT_SCREEN_HEIGHT: int = 1080

    # HZB defaults
    DEFAULT_HZB_WIDTH: int = 512
    DEFAULT_HZB_HEIGHT: int = 512
    DEFAULT_HZB_MIP_LEVELS: int = 9

    # Distance culling defaults
    DEFAULT_MAX_RENDER_DISTANCE: float = 1000.0
    DEFAULT_FADE_DISTANCE: float = 50.0


# =============================================================================
# MATH TYPES (Lightweight vector/matrix for GPU culling)
# =============================================================================


@dataclass(slots=True)
class Vec3:
    """3D vector for positions and directions."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vec3":
        return self.__mul__(scalar)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def length_squared(self) -> float:
        return self.dot(self)

    def normalized(self) -> "Vec3":
        length = self.length()
        if length < CullingConstants.EPSILON:
            return Vec3(0.0, 0.0, 0.0)
        inv_len = 1.0 / length
        return Vec3(self.x * inv_len, self.y * inv_len, self.z * inv_len)


@dataclass(slots=True)
class Vec4:
    """4D vector for planes and homogeneous coordinates."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.0

    @property
    def xyz(self) -> Vec3:
        return Vec3(self.x, self.y, self.z)

    def dot(self, other: "Vec4") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    def dot3(self, v: Vec3) -> float:
        """Dot product with xyz components only."""
        return self.x * v.x + self.y * v.y + self.z * v.z


@dataclass(slots=True)
class AABB:
    """Axis-Aligned Bounding Box."""
    min_point: Vec3 = field(default_factory=lambda: Vec3(float("inf"), float("inf"), float("inf")))
    max_point: Vec3 = field(default_factory=lambda: Vec3(float("-inf"), float("-inf"), float("-inf")))

    @property
    def center(self) -> Vec3:
        return Vec3(
            (self.min_point.x + self.max_point.x) * 0.5,
            (self.min_point.y + self.max_point.y) * 0.5,
            (self.min_point.z + self.max_point.z) * 0.5,
        )

    @property
    def extents(self) -> Vec3:
        """Half-extents from center to corner."""
        return Vec3(
            (self.max_point.x - self.min_point.x) * 0.5,
            (self.max_point.y - self.min_point.y) * 0.5,
            (self.max_point.z - self.min_point.z) * 0.5,
        )

    def contains_point(self, point: Vec3) -> bool:
        return (
            self.min_point.x <= point.x <= self.max_point.x
            and self.min_point.y <= point.y <= self.max_point.y
            and self.min_point.z <= point.z <= self.max_point.z
        )

    def expand(self, point: Vec3) -> None:
        """Expand AABB to include the given point."""
        self.min_point = Vec3(
            min(self.min_point.x, point.x),
            min(self.min_point.y, point.y),
            min(self.min_point.z, point.z),
        )
        self.max_point = Vec3(
            max(self.max_point.x, point.x),
            max(self.max_point.y, point.y),
            max(self.max_point.z, point.z),
        )


@dataclass(slots=True)
class BoundingSphere:
    """Bounding sphere for culling."""
    center: Vec3 = field(default_factory=Vec3)
    radius: float = 0.0

    @classmethod
    def from_aabb(cls, aabb: AABB) -> "BoundingSphere":
        """Create bounding sphere from AABB."""
        center = aabb.center
        extents = aabb.extents
        radius = extents.length()
        return cls(center=center, radius=radius)

    def contains_point(self, point: Vec3) -> bool:
        diff = point - self.center
        return diff.length_squared() <= self.radius * self.radius


# =============================================================================
# FRUSTUM PLANES
# =============================================================================


class FrustumPlane(IntEnum):
    """Indices for frustum planes."""
    LEFT = 0
    RIGHT = 1
    BOTTOM = 2
    TOP = 3
    NEAR = 4
    FAR = 5


@dataclass
class Frustum:
    """
    View frustum defined by 6 planes.

    Planes are stored with normals pointing inward (inside the frustum).
    Plane equation: ax + by + cz + d = 0, where (a,b,c) is the normal and d is the distance.
    """
    planes: list[Vec4] = field(default_factory=lambda: [Vec4() for _ in range(6)])

    @classmethod
    def from_view_projection_matrix(cls, vp_matrix: list[list[float]]) -> "Frustum":
        """
        Extract frustum planes from a view-projection matrix.

        Args:
            vp_matrix: 4x4 view-projection matrix in row-major order
        """
        frustum = cls()
        m = vp_matrix

        # Left plane: row3 + row0  (Gribb-Hartmann: for each column j, plane[j] = M[3][j] + M[0][j])
        frustum.planes[FrustumPlane.LEFT] = Vec4(
            m[3][0] + m[0][0],
            m[3][1] + m[0][1],
            m[3][2] + m[0][2],
            m[3][3] + m[0][3],
        )

        # Right plane: row3 - row0
        frustum.planes[FrustumPlane.RIGHT] = Vec4(
            m[3][0] - m[0][0],
            m[3][1] - m[0][1],
            m[3][2] - m[0][2],
            m[3][3] - m[0][3],
        )

        # Bottom plane: row3 + row1
        frustum.planes[FrustumPlane.BOTTOM] = Vec4(
            m[3][0] + m[1][0],
            m[3][1] + m[1][1],
            m[3][2] + m[1][2],
            m[3][3] + m[1][3],
        )

        # Top plane: row3 - row1
        frustum.planes[FrustumPlane.TOP] = Vec4(
            m[3][0] - m[1][0],
            m[3][1] - m[1][1],
            m[3][2] - m[1][2],
            m[3][3] - m[1][3],
        )

        # Near plane: row3 + row2
        frustum.planes[FrustumPlane.NEAR] = Vec4(
            m[3][0] + m[2][0],
            m[3][1] + m[2][1],
            m[3][2] + m[2][2],
            m[3][3] + m[2][3],
        )

        # Far plane: row3 - row2
        frustum.planes[FrustumPlane.FAR] = Vec4(
            m[3][0] - m[2][0],
            m[3][1] - m[2][1],
            m[3][2] - m[2][2],
            m[3][3] - m[2][3],
        )

        # Normalize planes
        for i, plane in enumerate(frustum.planes):
            length = math.sqrt(plane.x * plane.x + plane.y * plane.y + plane.z * plane.z)
            if length > CullingConstants.EPSILON:
                inv_len = 1.0 / length
                frustum.planes[i] = Vec4(
                    plane.x * inv_len,
                    plane.y * inv_len,
                    plane.z * inv_len,
                    plane.w * inv_len,
                )

        return frustum

    def test_sphere(self, sphere: BoundingSphere) -> bool:
        """
        Test if a sphere is inside or intersecting the frustum.

        Returns True if visible (inside or intersecting), False if fully outside.
        """
        for plane in self.planes:
            # Distance from center to plane
            distance = plane.dot3(sphere.center) + plane.w
            if distance < -sphere.radius:
                return False  # Sphere is fully behind this plane
        return True

    def test_aabb(self, aabb: AABB) -> bool:
        """
        Test if an AABB is inside or intersecting the frustum.

        Returns True if visible (inside or intersecting), False if fully outside.
        """
        for plane in self.planes:
            # Find the corner most aligned with plane normal (p-vertex)
            p = Vec3(
                aabb.max_point.x if plane.x >= 0 else aabb.min_point.x,
                aabb.max_point.y if plane.y >= 0 else aabb.min_point.y,
                aabb.max_point.z if plane.z >= 0 else aabb.min_point.z,
            )
            # If p-vertex is behind the plane, AABB is fully outside
            if plane.dot3(p) + plane.w < 0:
                return False
        return True


# =============================================================================
# INSTANCE DATA FOR CULLING
# =============================================================================


@dataclass(slots=True)
class InstanceBounds:
    """Bounds data for a single instance for GPU culling."""
    instance_id: int
    bounding_sphere: BoundingSphere
    aabb: AABB
    lod_index: int = 0
    flags: int = 0  # Bit flags for culling hints

    # Flag constants
    FLAG_CAST_SHADOW: int = 1 << 0
    FLAG_RECEIVE_SHADOW: int = 1 << 1
    FLAG_TWO_SIDED: int = 1 << 2
    FLAG_STATIC: int = 1 << 3


class CullResult(IntEnum):
    """Result of culling test."""
    VISIBLE = auto()
    CULLED_FRUSTUM = auto()
    CULLED_OCCLUSION = auto()
    CULLED_DISTANCE = auto()
    CULLED_SMALL_FEATURE = auto()


@dataclass
class CullingStats:
    """Statistics from a culling pass."""
    total_instances: int = 0
    visible_instances: int = 0
    frustum_culled: int = 0
    occlusion_culled: int = 0
    distance_culled: int = 0
    small_feature_culled: int = 0

    @property
    def cull_ratio(self) -> float:
        """Ratio of culled instances to total."""
        if self.total_instances == 0:
            return 0.0
        return 1.0 - (self.visible_instances / self.total_instances)


# =============================================================================
# ABSTRACT CULLER BASE CLASS
# =============================================================================


class Culler(ABC):
    """Abstract base class for culling stages."""

    @abstractmethod
    def cull(
        self,
        instances: Sequence[InstanceBounds],
        visible_mask: list[bool],
    ) -> CullingStats:
        """
        Perform culling on instances.

        Args:
            instances: Sequence of instance bounds to cull
            visible_mask: Mutable list of booleans, True = visible.
                         Will be modified in-place to mark culled instances.

        Returns:
            Statistics from this culling stage
        """
        ...

    @abstractmethod
    def update(self, **kwargs: Any) -> None:
        """Update culler state (e.g., new camera, new HZB)."""
        ...


# =============================================================================
# FRUSTUM CULLER
# =============================================================================


class FrustumCuller(Culler):
    """
    GPU frustum culling implementation.

    Tests instance bounding spheres against the 6 frustum planes.
    In production, this runs as a compute shader on GPU.
    """

    def __init__(self) -> None:
        self._frustum: Optional[Frustum] = None
        self._use_spheres: bool = True  # Use spheres for faster initial test

    @property
    def frustum(self) -> Optional[Frustum]:
        return self._frustum

    def update(
        self,
        frustum: Optional[Frustum] = None,
        view_projection_matrix: Optional[list[list[float]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Update frustum for culling.

        Args:
            frustum: Pre-built frustum, or
            view_projection_matrix: 4x4 VP matrix to extract frustum from
        """
        if frustum is not None:
            self._frustum = frustum
        elif view_projection_matrix is not None:
            self._frustum = Frustum.from_view_projection_matrix(view_projection_matrix)

    def cull(
        self,
        instances: Sequence[InstanceBounds],
        visible_mask: list[bool],
    ) -> CullingStats:
        """
        Perform frustum culling on all instances.

        Tests bounding spheres against frustum planes.
        Instances marked False in visible_mask are skipped (already culled).
        """
        stats = CullingStats(total_instances=len(instances))

        if self._frustum is None:
            # No frustum set, mark all as visible
            stats.visible_instances = sum(visible_mask)
            return stats

        for i, instance in enumerate(instances):
            if not visible_mask[i]:
                continue  # Already culled by previous stage

            # Test sphere first (faster)
            if self._use_spheres:
                visible = self._frustum.test_sphere(instance.bounding_sphere)
            else:
                visible = self._frustum.test_aabb(instance.aabb)

            if not visible:
                visible_mask[i] = False
                stats.frustum_culled += 1

        stats.visible_instances = sum(visible_mask)
        return stats

    def cull_single(self, instance: InstanceBounds) -> bool:
        """Test a single instance against the frustum."""
        if self._frustum is None:
            return True
        return self._frustum.test_sphere(instance.bounding_sphere)


# =============================================================================
# HIERARCHICAL-Z OCCLUSION CULLER
# =============================================================================


@dataclass
class HZBConfig:
    """Configuration for Hierarchical-Z Buffer."""
    width: int = CullingConstants.DEFAULT_HZB_WIDTH
    height: int = CullingConstants.DEFAULT_HZB_HEIGHT
    mip_levels: int = CullingConstants.DEFAULT_HZB_MIP_LEVELS  # log2(512) + 1
    conservative: bool = True  # Use conservative rasterization


class OcclusionCuller(Culler):
    """
    Hierarchical-Z Buffer (HZB) occlusion culling.

    Uses the depth buffer from the previous frame to cull objects
    that are occluded by closer geometry.

    Pipeline:
    1. Build HZB mip chain from depth buffer (max of 4 pixels per mip level)
    2. For each instance, project bounds to screen space
    3. Sample appropriate HZB mip level based on projected size
    4. Compare instance's min Z against HZB sample
    5. If instance's min Z > HZB sample, instance is occluded
    """

    def __init__(self, config: Optional[HZBConfig] = None) -> None:
        self._config = config or HZBConfig()
        # HZB pyramid: list of mip levels, each a 2D array of max depths
        self._hzb_pyramid: list[list[list[float]]] = []
        self._view_matrix: Optional[list[list[float]]] = None
        self._projection_matrix: Optional[list[list[float]]] = None
        self._near_plane: float = CullingConstants.DEFAULT_NEAR_PLANE
        self._far_plane: float = CullingConstants.DEFAULT_FAR_PLANE

    @property
    def config(self) -> HZBConfig:
        return self._config

    def update(
        self,
        depth_buffer: Optional[list[list[float]]] = None,
        view_matrix: Optional[list[list[float]]] = None,
        projection_matrix: Optional[list[list[float]]] = None,
        near_plane: float = CullingConstants.DEFAULT_NEAR_PLANE,
        far_plane: float = CullingConstants.DEFAULT_FAR_PLANE,
        **kwargs: Any,
    ) -> None:
        """
        Update HZB from depth buffer.

        Args:
            depth_buffer: 2D depth buffer (normalized 0-1)
            view_matrix: Camera view matrix
            projection_matrix: Camera projection matrix
            near_plane: Camera near plane distance
            far_plane: Camera far plane distance
        """
        if view_matrix is not None:
            self._view_matrix = view_matrix
        if projection_matrix is not None:
            self._projection_matrix = projection_matrix
        self._near_plane = near_plane
        self._far_plane = far_plane

        if depth_buffer is not None:
            self._build_hzb(depth_buffer)

    def _build_hzb(self, depth_buffer: list[list[float]]) -> None:
        """Build HZB mip chain from depth buffer."""
        self._hzb_pyramid = []

        # Level 0 is the full-resolution depth buffer
        current_level = depth_buffer
        self._hzb_pyramid.append(current_level)

        # Build mip levels by taking max of 2x2 blocks
        height = len(current_level)
        width = len(current_level[0]) if height > 0 else 0

        while width > 1 or height > 1:
            new_width = max(1, width // 2)
            new_height = max(1, height // 2)
            new_level: list[list[float]] = []

            for y in range(new_height):
                row: list[float] = []
                for x in range(new_width):
                    # Sample 2x2 block from previous level
                    x0, y0 = x * 2, y * 2
                    x1 = min(x0 + 1, width - 1)
                    y1 = min(y0 + 1, height - 1)

                    # Take maximum depth (conservative for occlusion)
                    max_depth = max(
                        current_level[y0][x0],
                        current_level[y0][x1],
                        current_level[y1][x0],
                        current_level[y1][x1],
                    )
                    row.append(max_depth)
                new_level.append(row)

            self._hzb_pyramid.append(new_level)
            current_level = new_level
            width = new_width
            height = new_height

    def _project_sphere_to_screen(
        self,
        sphere: BoundingSphere,
    ) -> tuple[float, float, float, float, float]:
        """
        Project bounding sphere to screen space.

        Returns:
            (min_x, min_y, max_x, max_y, min_depth) in normalized device coords
            Returns None values if sphere is behind camera
        """
        if self._view_matrix is None or self._projection_matrix is None:
            return 0.0, 0.0, 1.0, 1.0, 0.0

        # Transform center to view space
        c = sphere.center
        vm = self._view_matrix
        view_x = vm[0][0] * c.x + vm[0][1] * c.y + vm[0][2] * c.z + vm[0][3]
        view_y = vm[1][0] * c.x + vm[1][1] * c.y + vm[1][2] * c.z + vm[1][3]
        view_z = vm[2][0] * c.x + vm[2][1] * c.y + vm[2][2] * c.z + vm[2][3]

        # Check if behind near plane
        if view_z > -self._near_plane:
            return 0.0, 0.0, 1.0, 1.0, 0.0

        # Project to clip space
        pm = self._projection_matrix
        clip_x = pm[0][0] * view_x + pm[0][1] * view_y + pm[0][2] * view_z + pm[0][3]
        clip_y = pm[1][0] * view_x + pm[1][1] * view_y + pm[1][2] * view_z + pm[1][3]
        clip_z = pm[2][0] * view_x + pm[2][1] * view_y + pm[2][2] * view_z + pm[2][3]
        clip_w = pm[3][0] * view_x + pm[3][1] * view_y + pm[3][2] * view_z + pm[3][3]

        if abs(clip_w) < CullingConstants.EPSILON:
            return 0.0, 0.0, 1.0, 1.0, 0.0

        # Normalized device coordinates
        inv_w = 1.0 / clip_w
        ndc_x = clip_x * inv_w
        ndc_y = clip_y * inv_w
        ndc_z = clip_z * inv_w

        # Approximate screen-space radius
        screen_radius = (sphere.radius / abs(view_z)) * pm[0][0]

        # Screen-space bounds
        min_x = max(0.0, (ndc_x - screen_radius + 1.0) * 0.5)
        max_x = min(1.0, (ndc_x + screen_radius + 1.0) * 0.5)
        min_y = max(0.0, (ndc_y - screen_radius + 1.0) * 0.5)
        max_y = min(1.0, (ndc_y + screen_radius + 1.0) * 0.5)

        # Depth (map NDC Z to 0-1)
        min_depth = (ndc_z + 1.0) * 0.5

        return min_x, min_y, max_x, max_y, min_depth

    def _select_mip_level(self, screen_width: float, screen_height: float) -> int:
        """Select appropriate mip level based on projected size."""
        if not self._hzb_pyramid:
            return 0

        # Size in pixels at base resolution
        base_width = self._config.width
        base_height = self._config.height
        pixel_width = screen_width * base_width
        pixel_height = screen_height * base_height

        # Select mip level where texel size >= projected size
        max_dim = max(pixel_width, pixel_height)
        if max_dim <= 1:
            return len(self._hzb_pyramid) - 1

        mip_level = max(0, int(math.ceil(math.log2(max_dim))))
        return min(mip_level, len(self._hzb_pyramid) - 1)

    def _sample_hzb(self, x: float, y: float, mip_level: int) -> float:
        """Sample HZB at given position and mip level."""
        if not self._hzb_pyramid or mip_level >= len(self._hzb_pyramid):
            return 1.0  # Assume not occluded

        level = self._hzb_pyramid[mip_level]
        height = len(level)
        width = len(level[0]) if height > 0 else 0

        if width == 0 or height == 0:
            return 1.0

        # Clamp to valid range
        px = max(0, min(width - 1, int(x * width)))
        py = max(0, min(height - 1, int(y * height)))

        return level[py][px]

    def cull(
        self,
        instances: Sequence[InstanceBounds],
        visible_mask: list[bool],
    ) -> CullingStats:
        """
        Perform HZB occlusion culling.

        Projects each visible instance to screen space and compares
        against the hierarchical depth buffer.
        """
        stats = CullingStats(total_instances=len(instances))

        if not self._hzb_pyramid:
            # No HZB available, skip occlusion culling
            stats.visible_instances = sum(visible_mask)
            return stats

        for i, instance in enumerate(instances):
            if not visible_mask[i]:
                continue  # Already culled

            # Project to screen space
            min_x, min_y, max_x, max_y, min_depth = self._project_sphere_to_screen(
                instance.bounding_sphere
            )

            # Select mip level
            screen_width = max_x - min_x
            screen_height = max_y - min_y
            mip_level = self._select_mip_level(screen_width, screen_height)

            # Sample HZB at center of projected bounds
            center_x = (min_x + max_x) * 0.5
            center_y = (min_y + max_y) * 0.5
            hzb_depth = self._sample_hzb(center_x, center_y, mip_level)

            # If instance's minimum depth is greater than HZB depth, it's occluded
            if min_depth > hzb_depth:
                visible_mask[i] = False
                stats.occlusion_culled += 1

        stats.visible_instances = sum(visible_mask)
        return stats

    def is_occluded(self, sphere: BoundingSphere) -> bool:
        """Test if a single sphere is occluded."""
        if not self._hzb_pyramid:
            return False

        min_x, min_y, max_x, max_y, min_depth = self._project_sphere_to_screen(sphere)
        screen_width = max_x - min_x
        screen_height = max_y - min_y
        mip_level = self._select_mip_level(screen_width, screen_height)
        center_x = (min_x + max_x) * 0.5
        center_y = (min_y + max_y) * 0.5
        hzb_depth = self._sample_hzb(center_x, center_y, mip_level)

        return min_depth > hzb_depth

    def has_hzb_pyramid(self) -> bool:
        """Check if the HZB pyramid has been built."""
        return len(self._hzb_pyramid) > 0

    @property
    def hzb_mip_levels(self) -> int:
        """Get the number of mip levels in the HZB pyramid."""
        return len(self._hzb_pyramid)


# =============================================================================
# DISTANCE CULLER
# =============================================================================


@dataclass
class DistanceCullConfig:
    """Configuration for distance-based culling."""
    max_distance: float = CullingConstants.DEFAULT_MAX_RENDER_DISTANCE  # Maximum render distance
    fade_distance: float = CullingConstants.DEFAULT_FADE_DISTANCE  # Distance over which to fade out (for LOD transition)
    lod_distances: list[float] = field(
        default_factory=lambda: [10.0, 50.0, 200.0, 500.0]
    )


class DistanceCuller(Culler):
    """
    Distance-based culling.

    Culls instances beyond a maximum render distance from the camera.
    Also computes LOD levels based on distance.
    """

    def __init__(self, config: Optional[DistanceCullConfig] = None) -> None:
        self._config = config or DistanceCullConfig()
        self._camera_position: Vec3 = Vec3()
        self._camera_forward: Vec3 = Vec3(0.0, 0.0, -1.0)

    @property
    def config(self) -> DistanceCullConfig:
        return self._config

    def update(
        self,
        camera_position: Optional[Vec3] = None,
        camera_forward: Optional[Vec3] = None,
        max_distance: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Update camera position and culling parameters."""
        if camera_position is not None:
            self._camera_position = camera_position
        if camera_forward is not None:
            self._camera_forward = camera_forward.normalized()
        if max_distance is not None:
            self._config.max_distance = max_distance

    def _compute_distance(self, instance: InstanceBounds) -> float:
        """Compute distance from camera to instance."""
        diff = instance.bounding_sphere.center - self._camera_position
        return diff.length()

    def compute_lod_level(self, distance: float) -> int:
        """Compute LOD level based on distance."""
        for i, lod_distance in enumerate(self._config.lod_distances):
            if distance < lod_distance:
                return i
        return len(self._config.lod_distances)

    def cull(
        self,
        instances: Sequence[InstanceBounds],
        visible_mask: list[bool],
    ) -> CullingStats:
        """
        Perform distance culling.

        Culls instances beyond max_distance. Also updates instance LOD indices.
        """
        stats = CullingStats(total_instances=len(instances))
        max_dist_sq = self._config.max_distance * self._config.max_distance

        for i, instance in enumerate(instances):
            if not visible_mask[i]:
                continue  # Already culled

            distance_sq = (
                instance.bounding_sphere.center - self._camera_position
            ).length_squared()

            # Account for bounding sphere radius
            effective_dist_sq = max(
                0.0, distance_sq - instance.bounding_sphere.radius ** 2
            )

            if effective_dist_sq > max_dist_sq:
                visible_mask[i] = False
                stats.distance_culled += 1

        stats.visible_instances = sum(visible_mask)
        return stats


# =============================================================================
# SMALL FEATURE CULLER
# =============================================================================


@dataclass
class SmallFeatureCullConfig:
    """Configuration for small feature culling."""
    min_screen_size: float = CullingConstants.DEFAULT_MIN_SCREEN_SIZE  # Minimum size in pixels to render
    screen_width: int = CullingConstants.DEFAULT_SCREEN_WIDTH
    screen_height: int = CullingConstants.DEFAULT_SCREEN_HEIGHT


class SmallFeatureCuller(Culler):
    """
    Small feature culling.

    Culls objects that project to less than a minimum number of pixels on screen.
    This prevents tiny objects from wasting GPU cycles.
    """

    def __init__(self, config: Optional[SmallFeatureCullConfig] = None) -> None:
        self._config = config or SmallFeatureCullConfig()
        self._camera_position: Vec3 = Vec3()
        self._fov_y: float = math.radians(60.0)  # Vertical FOV in radians

    @property
    def config(self) -> SmallFeatureCullConfig:
        return self._config

    def update(
        self,
        camera_position: Optional[Vec3] = None,
        fov_y: Optional[float] = None,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Update camera and screen parameters."""
        if camera_position is not None:
            self._camera_position = camera_position
        if fov_y is not None:
            self._fov_y = fov_y
        if screen_width is not None:
            self._config.screen_width = screen_width
        if screen_height is not None:
            self._config.screen_height = screen_height

    def _compute_screen_size(self, instance: InstanceBounds) -> float:
        """Compute approximate screen-space size in pixels."""
        diff = instance.bounding_sphere.center - self._camera_position
        distance = diff.length()

        if distance < 1e-6:
            return float("inf")  # Very close, always visible

        # Approximate screen-space radius
        radius = instance.bounding_sphere.radius
        screen_radius = (radius / distance) / math.tan(self._fov_y * 0.5)
        screen_size = screen_radius * self._config.screen_height * 0.5

        return screen_size * 2.0  # Diameter in pixels

    def cull(
        self,
        instances: Sequence[InstanceBounds],
        visible_mask: list[bool],
    ) -> CullingStats:
        """
        Perform small feature culling.

        Culls instances that project to less than min_screen_size pixels.
        """
        stats = CullingStats(total_instances=len(instances))

        for i, instance in enumerate(instances):
            if not visible_mask[i]:
                continue

            screen_size = self._compute_screen_size(instance)

            if screen_size < self._config.min_screen_size:
                visible_mask[i] = False
                stats.small_feature_culled += 1

        stats.visible_instances = sum(visible_mask)
        return stats


# =============================================================================
# CULLING PIPELINE
# =============================================================================


class CullingPipeline:
    """
    Orchestrates the full GPU culling pipeline.

    Pipeline stages:
    1. Frustum culling - Remove instances outside view frustum
    2. Occlusion culling - Remove instances hidden by closer geometry (HZB)
    3. Distance culling - Remove instances beyond max render distance
    4. Small feature culling - Remove instances too small to see

    Each stage operates on the surviving instances from the previous stage.
    """

    def __init__(self) -> None:
        self._frustum_culler = FrustumCuller()
        self._occlusion_culler = OcclusionCuller()
        self._distance_culler = DistanceCuller()
        self._small_feature_culler = SmallFeatureCuller()

        self._enable_frustum: bool = True
        self._enable_occlusion: bool = True
        self._enable_distance: bool = True
        self._enable_small_feature: bool = True

    @property
    def frustum_culler(self) -> FrustumCuller:
        return self._frustum_culler

    @property
    def occlusion_culler(self) -> OcclusionCuller:
        return self._occlusion_culler

    @property
    def distance_culler(self) -> DistanceCuller:
        return self._distance_culler

    @property
    def small_feature_culler(self) -> SmallFeatureCuller:
        return self._small_feature_culler

    def configure(
        self,
        enable_frustum: Optional[bool] = None,
        enable_occlusion: Optional[bool] = None,
        enable_distance: Optional[bool] = None,
        enable_small_feature: Optional[bool] = None,
    ) -> None:
        """Enable or disable individual culling stages."""
        if enable_frustum is not None:
            self._enable_frustum = enable_frustum
        if enable_occlusion is not None:
            self._enable_occlusion = enable_occlusion
        if enable_distance is not None:
            self._enable_distance = enable_distance
        if enable_small_feature is not None:
            self._enable_small_feature = enable_small_feature

    def update(
        self,
        frustum: Optional[Frustum] = None,
        view_projection_matrix: Optional[list[list[float]]] = None,
        view_matrix: Optional[list[list[float]]] = None,
        projection_matrix: Optional[list[list[float]]] = None,
        depth_buffer: Optional[list[list[float]]] = None,
        camera_position: Optional[Vec3] = None,
        camera_forward: Optional[Vec3] = None,
        fov_y: Optional[float] = None,
        near_plane: float = CullingConstants.DEFAULT_NEAR_PLANE,
        far_plane: float = CullingConstants.DEFAULT_FAR_PLANE,
    ) -> None:
        """
        Update all cullers with new frame data.

        Args:
            frustum: Pre-built frustum or None to extract from VP matrix
            view_projection_matrix: Combined view-projection matrix
            view_matrix: View matrix (for HZB projection)
            projection_matrix: Projection matrix (for HZB projection)
            depth_buffer: Previous frame's depth buffer for HZB
            camera_position: Camera world position
            camera_forward: Camera forward direction
            fov_y: Vertical field of view in radians
            near_plane: Camera near plane distance
            far_plane: Camera far plane distance
        """
        self._frustum_culler.update(
            frustum=frustum,
            view_projection_matrix=view_projection_matrix,
        )
        self._occlusion_culler.update(
            depth_buffer=depth_buffer,
            view_matrix=view_matrix,
            projection_matrix=projection_matrix,
            near_plane=near_plane,
            far_plane=far_plane,
        )
        self._distance_culler.update(
            camera_position=camera_position,
            camera_forward=camera_forward,
        )
        self._small_feature_culler.update(
            camera_position=camera_position,
            fov_y=fov_y,
        )

    def cull(
        self,
        instances: Sequence[InstanceBounds],
    ) -> tuple[list[int], CullingStats]:
        """
        Run the full culling pipeline.

        Args:
            instances: All instances to cull

        Returns:
            Tuple of (visible_instance_indices, combined_stats)
        """
        total_stats = CullingStats(total_instances=len(instances))

        # Initialize visibility mask (all visible)
        visible_mask = [True] * len(instances)

        # Stage 1: Frustum culling
        if self._enable_frustum:
            frustum_stats = self._frustum_culler.cull(instances, visible_mask)
            total_stats.frustum_culled = frustum_stats.frustum_culled

        # Stage 2: Occlusion culling (on surviving instances)
        if self._enable_occlusion:
            occlusion_stats = self._occlusion_culler.cull(instances, visible_mask)
            total_stats.occlusion_culled = occlusion_stats.occlusion_culled

        # Stage 3: Distance culling
        if self._enable_distance:
            distance_stats = self._distance_culler.cull(instances, visible_mask)
            total_stats.distance_culled = distance_stats.distance_culled

        # Stage 4: Small feature culling
        if self._enable_small_feature:
            small_stats = self._small_feature_culler.cull(instances, visible_mask)
            total_stats.small_feature_culled = small_stats.small_feature_culled

        # Collect visible instance indices
        visible_indices = [i for i, visible in enumerate(visible_mask) if visible]
        total_stats.visible_instances = len(visible_indices)

        return visible_indices, total_stats


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Constants
    "CullingConstants",
    # Math types
    "Vec3",
    "Vec4",
    "AABB",
    "BoundingSphere",
    # Frustum
    "Frustum",
    "FrustumPlane",
    # Instance data
    "InstanceBounds",
    "CullResult",
    "CullingStats",
    # Cullers
    "Culler",
    "FrustumCuller",
    "OcclusionCuller",
    "DistanceCuller",
    "SmallFeatureCuller",
    # Configs
    "HZBConfig",
    "DistanceCullConfig",
    "SmallFeatureCullConfig",
    # Pipeline
    "CullingPipeline",
]

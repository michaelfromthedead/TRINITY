"""Planar Mirror Rendering for reflective surfaces.

Implements planar reflections for mirrors, water, and other reflective surfaces
using a reflected camera render pass technique.

The reflection matrix R for a plane with normal n and distance d is:
    R = I - 2 * n * n^T - 2 * d * [0, 0, 0, n]^T

Where I is the identity matrix. This reflects points across the plane
while preserving relative distances and angles.

Fresnel reflectance uses Schlick's approximation:
    F = F0 + (1 - F0) * (1 - cos(theta))^5

Where theta is the angle between view direction and normal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

from engine.core.math.geometry import AABB, Frustum, Plane
from engine.core.math.mat import Mat4
from engine.core.math.vec import Vec3, Vec4

if TYPE_CHECKING:
    pass


class MirrorUpdateMode(Enum):
    """Update modes for planar mirrors."""
    EVERY_FRAME = auto()    # Update reflection every frame
    ON_VISIBLE = auto()     # Only update when mirror is visible
    MANUAL = auto()         # Manual update only (for static scenes)


@dataclass
class PlanarMirrorConfig:
    """Configuration for a planar mirror.

    Attributes:
        resolution_scale: Scale factor for reflection resolution (0.25-1.0).
            0.5 means half the screen resolution.
        max_distance: Maximum distance to render objects in reflection.
            Objects beyond this distance are culled from the reflection pass.
        fresnel_power: Exponent for Fresnel falloff (typically 2-5).
            Higher values = sharper transition at grazing angles.
        blur_amount: Post-process blur for simulating roughness (0-1).
            0 = perfect mirror, 1 = very blurry reflection.
        base_reflectivity: Base reflectance at normal incidence (F0).
            0.04 for dielectrics, 0.5-1.0 for metals.
        update_mode: When to update the reflection.
        clip_offset: Small offset to prevent z-fighting at mirror plane.
        render_back_faces: Whether to render back faces in reflection.
    """
    resolution_scale: float = 0.5
    max_distance: float = 100.0
    fresnel_power: float = 5.0
    blur_amount: float = 0.0
    base_reflectivity: float = 0.04
    update_mode: MirrorUpdateMode = MirrorUpdateMode.ON_VISIBLE
    clip_offset: float = 0.01
    render_back_faces: bool = True


@dataclass
class PlanarMirror:
    """A planar mirror surface for reflection rendering.

    Uses a reflected camera render pass to capture reflections.
    The reflection matrix transforms the camera to its mirrored position,
    and an oblique near clip plane prevents rendering behind the mirror.

    Attributes:
        plane: Mirror plane equation (normal points toward reflected side).
        config: Mirror configuration parameters.
        bounds: Optional AABB for visibility culling.
        priority: Priority for rendering order (higher = rendered first).
        _active: Whether the mirror is currently active.
        _dirty: Whether the reflection needs updating.
        _reflection_matrix: Cached reflection matrix.
        _mirror_id: Unique identifier for this mirror.
    """
    plane: Plane = field(default_factory=lambda: Plane(Vec3(0, 1, 0), 0.0))
    config: PlanarMirrorConfig = field(default_factory=PlanarMirrorConfig)
    bounds: Optional[AABB] = None
    priority: int = 0
    _active: bool = True
    _dirty: bool = True
    _reflection_matrix: Optional[Mat4] = None
    _mirror_id: int = 0
    _id_counter: int = 0

    def __post_init__(self) -> None:
        PlanarMirror._id_counter += 1
        self._mirror_id = PlanarMirror._id_counter
        self._compute_reflection_matrix()

    def _compute_reflection_matrix(self) -> None:
        """Compute and cache the reflection matrix for this mirror plane.

        The reflection matrix R for plane (n, d) is:
            R = I - 2*n*n^T
        With translation component for non-origin planes.

        Matrix form (column-major):
            [1-2*nx*nx,  -2*nx*ny,  -2*nx*nz,   0]
            [-2*ny*nx,  1-2*ny*ny,  -2*ny*nz,   0]
            [-2*nz*nx,  -2*nz*ny,  1-2*nz*nz,   0]
            [-2*d*nx,   -2*d*ny,   -2*d*nz,     1]
        """
        n = self.plane.normal
        d = self.plane.distance

        # Compute outer product n*n^T and reflection matrix
        # R = I - 2*n*n^T
        m = [0.0] * 16

        # Diagonal and off-diagonal elements
        m[0] = 1.0 - 2.0 * n.x * n.x
        m[1] = -2.0 * n.y * n.x
        m[2] = -2.0 * n.z * n.x
        m[3] = 0.0

        m[4] = -2.0 * n.x * n.y
        m[5] = 1.0 - 2.0 * n.y * n.y
        m[6] = -2.0 * n.z * n.y
        m[7] = 0.0

        m[8] = -2.0 * n.x * n.z
        m[9] = -2.0 * n.y * n.z
        m[10] = 1.0 - 2.0 * n.z * n.z
        m[11] = 0.0

        # Translation component for non-origin planes
        m[12] = -2.0 * d * n.x
        m[13] = -2.0 * d * n.y
        m[14] = -2.0 * d * n.z
        m[15] = 1.0

        self._reflection_matrix = Mat4(m)
        self._dirty = False

    @property
    def reflection_matrix(self) -> Mat4:
        """Get the reflection matrix, recomputing if dirty."""
        if self._dirty or self._reflection_matrix is None:
            self._compute_reflection_matrix()
        return self._reflection_matrix

    @property
    def is_active(self) -> bool:
        """Check if the mirror is active."""
        return self._active

    def activate(self) -> None:
        """Activate the mirror for rendering."""
        self._active = True

    def deactivate(self) -> None:
        """Deactivate the mirror."""
        self._active = False

    def set_plane(self, plane: Plane) -> None:
        """Update the mirror plane and mark for recomputation.

        Args:
            plane: New plane equation.
        """
        self.plane = plane
        self._dirty = True

    def reflect_point(self, point: Vec3) -> Vec3:
        """Reflect a point across the mirror plane.

        Args:
            point: World-space point to reflect.

        Returns:
            Reflected point position.
        """
        return self.reflection_matrix.transform_point(point)

    def reflect_direction(self, direction: Vec3) -> Vec3:
        """Reflect a direction across the mirror plane.

        Args:
            direction: Direction vector to reflect.

        Returns:
            Reflected direction (normalized).
        """
        return self.reflection_matrix.transform_direction(direction).normalized()

    def reflect_camera(
        self,
        camera_view: Mat4,
        camera_proj: Mat4,
    ) -> tuple[Mat4, Mat4]:
        """Compute reflected view and modified projection matrices.

        The reflected view matrix is computed by applying the reflection
        matrix to the original view matrix. The projection matrix is
        modified to use an oblique near clip plane that coincides with
        the mirror plane, preventing rendering of objects behind the mirror.

        Args:
            camera_view: Original camera view matrix.
            camera_proj: Original camera projection matrix.

        Returns:
            Tuple of (reflected_view, oblique_projection).
        """
        # Reflect the view matrix
        reflected_view = self.reflection_matrix @ camera_view

        # Compute oblique near clip plane
        oblique_proj = self._compute_oblique_projection(
            reflected_view, camera_proj
        )

        return reflected_view, oblique_proj

    def _compute_oblique_projection(
        self,
        view_matrix: Mat4,
        proj_matrix: Mat4,
    ) -> Mat4:
        """Compute projection matrix with oblique near clip plane.

        Uses Eric Lengyel's technique for oblique frustum clipping.
        The mirror plane is transformed to view space and used as the
        near clip plane.

        Args:
            view_matrix: View matrix (for transforming plane to view space).
            proj_matrix: Original projection matrix.

        Returns:
            Modified projection matrix with oblique near plane.
        """
        # Transform mirror plane to view space
        clip_plane = self.transform_plane_to_view(view_matrix)

        # Apply oblique clipping
        return compute_oblique_projection(proj_matrix, clip_plane)

    def transform_plane_to_view(self, view_matrix: Mat4) -> Vec4:
        """Transform the mirror plane from world space to view space.

        Planes transform by the inverse-transpose of the transformation matrix.
        For a plane P = (n, d) and transformation M:

            P_view = (M^-1)^T * P_world

        Args:
            view_matrix: View matrix (world to view transformation).

        Returns:
            Plane equation in view space as Vec4(nx, ny, nz, d).
        """
        view_inv = view_matrix.inverse()
        view_inv_t = view_inv.transposed()

        n = self.plane.normal
        d = self.plane.distance + self.config.clip_offset

        # Transform plane using inverse transpose
        m = view_inv_t.m
        return Vec4(
            m[0] * n.x + m[4] * n.y + m[8] * n.z + m[12] * d,
            m[1] * n.x + m[5] * n.y + m[9] * n.z + m[13] * d,
            m[2] * n.x + m[6] * n.y + m[10] * n.z + m[14] * d,
            m[3] * n.x + m[7] * n.y + m[11] * n.z + m[15] * d,
        )

    def is_point_clipped(self, point: Vec3, view_matrix: Mat4) -> bool:
        """Check if a world-space point would be clipped by the oblique near plane.

        Points behind the mirror (on the negative side of the clip plane)
        are clipped and should not appear in the reflection.

        Args:
            point: Point in world space.
            view_matrix: View matrix for the reflected camera.

        Returns:
            True if the point is behind the mirror and would be clipped.
        """
        clip_plane = self.transform_plane_to_view(view_matrix)

        # Transform point to view space
        view_point = view_matrix.transform_point(point)

        # Signed distance to clip plane
        signed_dist = (
            clip_plane.x * view_point.x +
            clip_plane.y * view_point.y +
            clip_plane.z * view_point.z +
            clip_plane.w
        )

        return signed_dist < 0.0

    def compute_fresnel(
        self,
        view_dir: Vec3,
        normal: Vec3,
    ) -> float:
        """Compute Fresnel reflectance using Schlick's approximation.

        F = F0 + (1 - F0) * (1 - cos(theta))^5

        Where F0 is the base reflectivity and theta is the angle between
        the view direction and surface normal.

        Args:
            view_dir: View direction (from surface to camera), normalized.
            normal: Surface normal, normalized.

        Returns:
            Fresnel reflectance factor [0, 1].
        """
        # Ensure normalized vectors
        v = view_dir.normalized()
        n = normal.normalized()

        # Cosine of angle (clamped to positive values)
        cos_theta = max(0.0, v.dot(n))

        # Schlick's approximation
        f0 = self.config.base_reflectivity
        one_minus_cos = 1.0 - cos_theta
        one_minus_cos_5 = one_minus_cos ** self.config.fresnel_power

        fresnel = f0 + (1.0 - f0) * one_minus_cos_5

        return min(1.0, fresnel)

    def is_point_in_front(self, point: Vec3) -> bool:
        """Check if a point is in front of the mirror plane.

        Points in front are on the side the normal points to.

        Args:
            point: World-space point to test.

        Returns:
            True if the point is in front of the mirror.
        """
        return self.plane.signed_distance(point) >= 0

    def get_screen_coverage(
        self,
        camera_pos: Vec3,
        view_proj: Mat4,
        screen_width: int,
        screen_height: int,
    ) -> float:
        """Estimate the screen coverage of this mirror.

        Used for LOD and priority decisions.

        Args:
            camera_pos: Camera world position.
            view_proj: Combined view-projection matrix.
            screen_width: Screen width in pixels.
            screen_height: Screen height in pixels.

        Returns:
            Estimated coverage factor [0, 1].
        """
        if self.bounds is None:
            # No bounds specified, assume full coverage
            return 1.0

        # Project AABB corners to screen
        corners = [
            Vec3(self.bounds.min.x, self.bounds.min.y, self.bounds.min.z),
            Vec3(self.bounds.max.x, self.bounds.min.y, self.bounds.min.z),
            Vec3(self.bounds.min.x, self.bounds.max.y, self.bounds.min.z),
            Vec3(self.bounds.max.x, self.bounds.max.y, self.bounds.min.z),
            Vec3(self.bounds.min.x, self.bounds.min.y, self.bounds.max.z),
            Vec3(self.bounds.max.x, self.bounds.min.y, self.bounds.max.z),
            Vec3(self.bounds.min.x, self.bounds.max.y, self.bounds.max.z),
            Vec3(self.bounds.max.x, self.bounds.max.y, self.bounds.max.z),
        ]

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')

        for corner in corners:
            projected = view_proj.transform_point(corner)
            # NDC to screen
            sx = (projected.x + 1.0) * 0.5
            sy = (projected.y + 1.0) * 0.5

            min_x = min(min_x, sx)
            max_x = max(max_x, sx)
            min_y = min(min_y, sy)
            max_y = max(max_y, sy)

        # Clamp to [0, 1]
        min_x = max(0.0, min(1.0, min_x))
        max_x = max(0.0, min(1.0, max_x))
        min_y = max(0.0, min(1.0, min_y))
        max_y = max(0.0, min(1.0, max_y))

        return (max_x - min_x) * (max_y - min_y)


class PlanarMirrorManager:
    """Manages multiple planar mirrors with priority-based rendering.

    Limits the number of active mirrors per frame to maintain performance.
    Sorts mirrors by priority and visibility to determine render order.

    Attributes:
        mirrors: List of all registered mirrors.
        max_active_per_frame: Maximum mirrors to render per frame (default 2).
        _frame_count: Internal frame counter for temporal updates.
    """

    def __init__(self, max_active_per_frame: int = 2) -> None:
        """Initialize the mirror manager.

        Args:
            max_active_per_frame: Maximum mirrors to render each frame.
        """
        self.mirrors: list[PlanarMirror] = []
        self.max_active_per_frame = max_active_per_frame
        self._frame_count = 0

    def add_mirror(self, mirror: PlanarMirror) -> None:
        """Add a mirror to the manager.

        Args:
            mirror: Mirror to add.
        """
        self.mirrors.append(mirror)

    def remove_mirror(self, mirror: PlanarMirror) -> None:
        """Remove a mirror from the manager.

        Args:
            mirror: Mirror to remove.
        """
        if mirror in self.mirrors:
            self.mirrors.remove(mirror)

    def clear(self) -> None:
        """Remove all mirrors."""
        self.mirrors.clear()

    def get_visible_mirrors(
        self,
        camera_pos: Vec3,
        frustum: Frustum,
    ) -> list[PlanarMirror]:
        """Get mirrors visible from the camera, sorted by priority.

        Returns mirrors that:
        1. Are active
        2. The camera is in front of (can see the reflective surface)
        3. Are within the view frustum (if bounds are specified)

        Args:
            camera_pos: Camera world position.
            frustum: Camera view frustum.

        Returns:
            List of visible mirrors, sorted by priority (highest first).
        """
        visible = []

        for mirror in self.mirrors:
            if not mirror.is_active:
                continue

            # Camera must be in front of mirror to see reflection
            if not mirror.is_point_in_front(camera_pos):
                continue

            # Check frustum intersection if bounds are specified
            if mirror.bounds is not None:
                if not frustum.intersects_aabb(mirror.bounds):
                    continue

            visible.append(mirror)

        # Sort by priority (highest first)
        visible.sort(key=lambda m: m.priority, reverse=True)

        return visible

    def get_mirrors_for_frame(
        self,
        camera_pos: Vec3,
        frustum: Frustum,
    ) -> list[PlanarMirror]:
        """Get mirrors to render this frame, respecting max_active limit.

        Args:
            camera_pos: Camera world position.
            frustum: Camera view frustum.

        Returns:
            List of mirrors to render (up to max_active_per_frame).
        """
        visible = self.get_visible_mirrors(camera_pos, frustum)
        self._frame_count += 1
        return visible[:self.max_active_per_frame]

    def get_mirror_by_id(self, mirror_id: int) -> Optional[PlanarMirror]:
        """Get a mirror by its unique ID.

        Args:
            mirror_id: Mirror ID to find.

        Returns:
            The mirror, or None if not found.
        """
        for mirror in self.mirrors:
            if mirror._mirror_id == mirror_id:
                return mirror
        return None

    @property
    def active_count(self) -> int:
        """Get the number of active mirrors."""
        return sum(1 for m in self.mirrors if m.is_active)

    @property
    def total_count(self) -> int:
        """Get the total number of registered mirrors."""
        return len(self.mirrors)


def create_water_plane(
    height: float = 0.0,
    bounds: Optional[AABB] = None,
    config: Optional[PlanarMirrorConfig] = None,
) -> PlanarMirror:
    """Create a mirror configured for water surfaces.

    Water uses lower base reflectivity (0.02) and higher Fresnel power
    for a more realistic water appearance.

    Args:
        height: Water surface height (Y coordinate).
        bounds: Optional bounds for the water surface.
        config: Optional custom configuration (defaults applied if None).

    Returns:
        Configured PlanarMirror for water.
    """
    water_config = config or PlanarMirrorConfig(
        resolution_scale=0.5,
        max_distance=200.0,
        fresnel_power=5.0,
        blur_amount=0.1,
        base_reflectivity=0.02,  # Water at normal incidence
        update_mode=MirrorUpdateMode.EVERY_FRAME,
    )

    # Water plane points up (normal = +Y)
    plane = Plane(Vec3(0, 1, 0), -height)

    return PlanarMirror(
        plane=plane,
        config=water_config,
        bounds=bounds,
        priority=10,  # Water typically high priority
    )


def compute_oblique_projection(proj_matrix: Mat4, clip_plane: Vec4) -> Mat4:
    """Compute projection matrix with oblique near-plane clipping.

    Implements Eric Lengyel's technique for modifying a projection matrix
    so that the near plane coincides with an arbitrary clip plane. This is
    used for planar reflections to prevent rendering geometry behind the mirror.

    The algorithm replaces the third row (near plane) of the projection matrix
    with coefficients derived from the clip plane, ensuring that:
    - Points exactly on the clip plane map to NDC z = -1
    - Points behind the clip plane are clipped (z < -1)
    - The far plane and frustum sides are preserved

    Reference:
        Eric Lengyel, "Modifying the Projection Matrix to Perform
        Oblique Near-Plane Clipping"
        http://www.terathon.com/lengyel/Lengyel-Oblique.pdf

    Args:
        proj_matrix: Original projection matrix.
        clip_plane: Clip plane in view space as Vec4(nx, ny, nz, d).

    Returns:
        Modified projection matrix with oblique near plane.
    """
    result = Mat4(proj_matrix.m.copy())
    p = result.m

    # Calculate Q vector - the corner of the near plane in clip space
    # that is closest to the clip plane
    q = Vec4(
        (math.copysign(1.0, clip_plane.x) + p[8]) / p[0],
        (math.copysign(1.0, clip_plane.y) + p[9]) / p[5],
        -1.0,
        (1.0 + p[10]) / p[14],
    )

    # Calculate scaling factor: 2 / (clip_plane . Q)
    dot = (
        clip_plane.x * q.x + clip_plane.y * q.y +
        clip_plane.z * q.z + clip_plane.w * q.w
    )

    if abs(dot) > 1e-6:
        scale = 2.0 / dot

        # Replace third row of projection matrix
        # new_row2 = clip_plane * scale - row3
        result.m[2] = clip_plane.x * scale - p[3]
        result.m[6] = clip_plane.y * scale - p[7]
        result.m[10] = clip_plane.z * scale - p[11]
        result.m[14] = clip_plane.w * scale - p[15]

    return result


def transform_plane_to_view(plane: Plane, view_matrix: Mat4, offset: float = 0.0) -> Vec4:
    """Transform a world-space plane to view space.

    Planes transform by the inverse-transpose of the transformation matrix.
    This is because planes are defined by their normal vector and distance,
    and normals transform differently than points.

    Args:
        plane: Plane in world space.
        view_matrix: View matrix (world to view transformation).
        offset: Optional offset to apply to plane distance (for z-fighting prevention).

    Returns:
        Plane equation in view space as Vec4(nx, ny, nz, d).
    """
    view_inv = view_matrix.inverse()
    view_inv_t = view_inv.transposed()

    n = plane.normal
    d = plane.distance + offset

    m = view_inv_t.m
    return Vec4(
        m[0] * n.x + m[4] * n.y + m[8] * n.z + m[12] * d,
        m[1] * n.x + m[5] * n.y + m[9] * n.z + m[13] * d,
        m[2] * n.x + m[6] * n.y + m[10] * n.z + m[14] * d,
        m[3] * n.x + m[7] * n.y + m[11] * n.z + m[15] * d,
    )


def signed_distance_to_plane(point: Vec3, plane: Plane) -> float:
    """Compute the signed distance from a point to a plane.

    Positive distance means the point is on the side the normal points to.
    Negative distance means the point is on the opposite side.

    Args:
        point: Point to test.
        plane: Plane to measure distance from.

    Returns:
        Signed distance (positive = in front, negative = behind).
    """
    return plane.normal.dot(point) + plane.distance


def create_mirror_plane(
    position: Vec3,
    normal: Vec3,
    bounds: Optional[AABB] = None,
    config: Optional[PlanarMirrorConfig] = None,
) -> PlanarMirror:
    """Create a mirror at an arbitrary position and orientation.

    Args:
        position: A point on the mirror surface.
        normal: Mirror normal (pointing toward the reflective side).
        bounds: Optional AABB for visibility culling.
        config: Optional custom configuration.

    Returns:
        Configured PlanarMirror.
    """
    mirror_config = config or PlanarMirrorConfig(
        resolution_scale=0.75,
        max_distance=50.0,
        fresnel_power=3.0,
        blur_amount=0.0,  # Perfect mirror
        base_reflectivity=0.9,  # High reflectivity for mirrors
        update_mode=MirrorUpdateMode.ON_VISIBLE,
    )

    # Compute plane distance: d = -n.dot(p)
    n = normal.normalized()
    d = -n.dot(position)

    plane = Plane(n, d)

    return PlanarMirror(
        plane=plane,
        config=mirror_config,
        bounds=bounds,
        priority=5,
    )

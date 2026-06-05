"""
Camera Ray Generation for SDF Ray Marching (T-DEMO-3.1).

This module implements pinhole camera ray generation for demoscene rendering.
It provides both Python evaluation for testing and WGSL code generation
for GPU compute shader integration.

The pinhole camera model:
  - Camera origin (eye position)
  - Look-at target point
  - Up vector for orientation
  - Field of view (vertical FOV in degrees)
  - Aspect ratio (width / height)

Ray generation converts normalized screen coordinates (u, v) in [-1, 1]
to world-space rays emanating from the camera origin through the virtual
image plane.

Usage:
    >>> from engine.rendering.demoscene.ray_generation import RayGenerator, Ray
    >>> from engine.rendering.demoscene.ast_nodes import CameraNode, Vec3Node, FloatNode
    >>> camera = CameraNode(
    ...     origin=Vec3Node(0.0, 0.0, 5.0),
    ...     look_at=Vec3Node(0.0, 0.0, 0.0),
    ...     up=Vec3Node(0.0, 1.0, 0.0),
    ...     fov=FloatNode(60.0),
    ...     aspect_ratio=FloatNode(16.0 / 9.0),
    ... )
    >>> gen = RayGenerator()
    >>> ray = gen.generate_ray(0.0, 0.0, camera)
    >>> ray.direction  # Looking straight ahead
    Vec3(x=0.0, y=0.0, z=-1.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .ast_nodes import CameraNode, FloatNode, Vec3Node


# =============================================================================
# Vec3 Helper (Mutable for internal calculations)
# =============================================================================


@dataclass
class Vec3:
    """Mutable 3D vector for ray calculations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def from_node(cls, node: Vec3Node) -> "Vec3":
        """Create Vec3 from a Vec3Node."""
        return cls(float(node.x), float(node.y), float(node.z))

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> "Vec3":
        """Create Vec3 from tuple."""
        return cls(float(t[0]), float(t[1]), float(t[2]))

    @classmethod
    def zero(cls) -> "Vec3":
        """Return zero vector."""
        return cls(0.0, 0.0, 0.0)

    def as_tuple(self) -> Tuple[float, float, float]:
        """Return as tuple."""
        return (self.x, self.y, self.z)

    def length(self) -> float:
        """Compute vector length (magnitude)."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def length_squared(self) -> float:
        """Compute squared length (avoids sqrt)."""
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> "Vec3":
        """Return normalized (unit length) vector."""
        length = self.length()
        if length < 1e-10:
            return Vec3(0.0, 0.0, 0.0)
        inv_len = 1.0 / length
        return Vec3(self.x * inv_len, self.y * inv_len, self.z * inv_len)

    def dot(self, other: "Vec3") -> float:
        """Compute dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        """Compute cross product."""
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vec3":
        return self.__mul__(scalar)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec3):
            return NotImplemented
        return (
            abs(self.x - other.x) < 1e-9 and
            abs(self.y - other.y) < 1e-9 and
            abs(self.z - other.z) < 1e-9
        )

    def __repr__(self) -> str:
        return f"Vec3(x={self.x}, y={self.y}, z={self.z})"

    def approx_equal(self, other: "Vec3", epsilon: float = 1e-6) -> bool:
        """Check approximate equality within epsilon."""
        return (
            abs(self.x - other.x) < epsilon and
            abs(self.y - other.y) < epsilon and
            abs(self.z - other.z) < epsilon
        )


# =============================================================================
# Ray Data Class
# =============================================================================


@dataclass
class Ray:
    """
    A ray defined by origin and direction.

    Attributes:
        origin: Ray start point (camera position in world space).
        direction: Normalized ray direction vector.

    The parametric ray equation is:
        point(t) = origin + t * direction

    where t >= 0 for points along the ray.
    """
    origin: Vec3
    direction: Vec3

    def point_at(self, t: float) -> Vec3:
        """
        Compute the point along the ray at parameter t.

        Args:
            t: Ray parameter (distance along ray if direction is normalized).

        Returns:
            Point at ray origin + t * direction.
        """
        return self.origin + self.direction * t

    def __repr__(self) -> str:
        return f"Ray(origin={self.origin}, direction={self.direction})"


# =============================================================================
# Camera Parameters Helper
# =============================================================================


@dataclass
class CameraParams:
    """
    Extracted camera parameters for ray generation.

    This is a convenience class that extracts values from CameraNode
    for use in ray generation calculations.
    """
    origin: Vec3
    look_at: Vec3
    up: Vec3
    fov: float  # Vertical FOV in degrees
    aspect_ratio: float
    aperture: float = 0.0
    focal_distance: float = 10.0

    @classmethod
    def from_camera_node(cls, camera: CameraNode) -> "CameraParams":
        """Create CameraParams from a CameraNode."""
        return cls(
            origin=Vec3.from_node(camera.origin),
            look_at=Vec3.from_node(camera.look_at),
            up=Vec3.from_node(camera.up),
            fov=float(camera.fov.value),
            aspect_ratio=float(camera.aspect_ratio.value),
            aperture=float(camera.aperture.value) if camera.aperture else 0.0,
            focal_distance=float(camera.focal_distance.value) if camera.focal_distance else 10.0,
        )


# =============================================================================
# Ray Generator Class
# =============================================================================


class RayGenerator:
    """
    Generates camera rays using the pinhole camera model.

    The pinhole camera model simulates an ideal camera with:
    - Infinitely small aperture (no depth of field)
    - Perfect perspective projection
    - No lens distortion

    The camera coordinate system:
    - Forward: normalized(look_at - origin)
    - Right: normalized(cross(forward, up))
    - Up: cross(right, forward)

    Screen coordinates (u, v) map to the image plane:
    - u: horizontal, -1 (left) to +1 (right)
    - v: vertical, -1 (bottom) to +1 (top)
    """

    def __init__(self) -> None:
        """Initialize the ray generator."""
        # Cached camera basis vectors
        self._origin: Optional[Vec3] = None
        self._forward: Optional[Vec3] = None
        self._right: Optional[Vec3] = None
        self._up: Optional[Vec3] = None
        self._half_width: float = 0.0
        self._half_height: float = 0.0
        self._cached_camera_id: Optional[int] = None

    def _setup_camera(self, camera: CameraNode) -> None:
        """
        Set up camera basis vectors and image plane dimensions.

        Args:
            camera: Camera node with position, orientation, and projection params.
        """
        params = CameraParams.from_camera_node(camera)

        # Camera position
        self._origin = params.origin

        # Camera basis vectors (orthonormal)
        self._forward = (params.look_at - params.origin).normalized()
        self._right = self._forward.cross(params.up).normalized()
        self._up = self._right.cross(self._forward)

        # Image plane half-dimensions based on FOV
        fov_radians = math.radians(params.fov)
        self._half_height = math.tan(fov_radians * 0.5)
        self._half_width = self._half_height * params.aspect_ratio

    def generate_ray(self, u: float, v: float, camera: CameraNode) -> Ray:
        """
        Generate a primary ray for the given UV coordinates.

        The UV coordinates are normalized screen coordinates:
        - u: horizontal position in [-1, 1], left to right
        - v: vertical position in [-1, 1], bottom to top

        The ray originates from the camera position and passes through
        the point (u, v) on the image plane.

        Args:
            u: Horizontal screen coordinate in [-1, 1].
            v: Vertical screen coordinate in [-1, 1].
            camera: Camera node defining position, orientation, and projection.

        Returns:
            Ray with origin at camera position and direction through (u, v).

        Example:
            >>> gen = RayGenerator()
            >>> # Center of screen should look straight ahead
            >>> ray = gen.generate_ray(0.0, 0.0, camera)
            >>> ray.direction.approx_equal(Vec3(0, 0, -1))
            True
        """
        # Setup camera if needed
        self._setup_camera(camera)

        # Compute ray direction on image plane
        # The direction is: forward + u * half_width * right + v * half_height * up
        direction = (
            self._forward +
            self._right * (u * self._half_width) +
            self._up * (v * self._half_height)
        ).normalized()

        return Ray(origin=Vec3(self._origin.x, self._origin.y, self._origin.z), direction=direction)

    def generate_rays_grid(
        self,
        camera: CameraNode,
        width: int,
        height: int,
    ) -> list[list[Ray]]:
        """
        Generate rays for all pixels in a grid.

        Args:
            camera: Camera node defining projection.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            2D list of rays [row][column].
        """
        self._setup_camera(camera)
        rays = []

        for y in range(height):
            row = []
            # Convert pixel to normalized coordinates
            v = 1.0 - 2.0 * (y + 0.5) / height  # Flip Y: top is +1

            for x in range(width):
                u = 2.0 * (x + 0.5) / width - 1.0

                direction = (
                    self._forward +
                    self._right * (u * self._half_width) +
                    self._up * (v * self._half_height)
                ).normalized()

                row.append(Ray(
                    origin=Vec3(self._origin.x, self._origin.y, self._origin.z),
                    direction=direction,
                ))
            rays.append(row)

        return rays

    def pixel_to_uv(self, x: int, y: int, width: int, height: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to normalized UV coordinates.

        Args:
            x: Pixel x coordinate (0 to width-1).
            y: Pixel y coordinate (0 to height-1).
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Tuple (u, v) in [-1, 1] range.
        """
        # Add 0.5 to sample pixel centers
        u = 2.0 * (x + 0.5) / width - 1.0
        v = 1.0 - 2.0 * (y + 0.5) / height  # Flip Y
        return (u, v)


# =============================================================================
# WGSL Code Generation
# =============================================================================


def generate_ray_wgsl(camera: Optional[CameraNode] = None) -> str:
    """
    Generate WGSL code for camera ray generation.

    This generates the `generate_ray` function used in the compute shader.
    The function takes normalized UV coordinates and camera parameters,
    returning a normalized ray direction.

    Args:
        camera: Optional camera node for inline constants. If None,
                generates parametric version using uniforms.

    Returns:
        WGSL code string for the generate_ray function.
    """
    return """\
/// Ray structure for ray marching.
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
}

/// Generates a camera ray using the pinhole camera model.
///
/// Implements perspective projection from camera through image plane.
/// The camera coordinate system is:
///   - Forward: normalized(target - origin)
///   - Right: normalized(cross(forward, up))
///   - Up: cross(right, forward)
///
/// Arguments:
///   uv:     Normalized screen coordinates in [-1, 1]
///   origin: Camera position in world space
///   target: Look-at point in world space
///   up:     World up vector (usually [0, 1, 0])
///   fov:    Vertical field of view in degrees
///   aspect: Aspect ratio (width / height)
///
/// Returns:
///   Normalized ray direction vector
fn generate_ray(
    uv: vec2<f32>,
    origin: vec3<f32>,
    target: vec3<f32>,
    up: vec3<f32>,
    fov: f32,
    aspect: f32,
) -> vec3<f32> {
    // Compute camera basis vectors
    let forward = normalize(target - origin);
    let right = normalize(cross(forward, up));
    let up_corrected = cross(right, forward);

    // Compute image plane dimensions from FOV
    let fov_rad = fov * 3.14159265359 / 180.0;
    let half_height = tan(fov_rad * 0.5);
    let half_width = half_height * aspect;

    // Compute ray direction
    let rd = normalize(
        forward +
        right * uv.x * half_width +
        up_corrected * uv.y * half_height
    );

    return rd;
}

/// Creates a Ray struct from UV coordinates and camera parameters.
fn create_ray(
    uv: vec2<f32>,
    origin: vec3<f32>,
    target: vec3<f32>,
    up: vec3<f32>,
    fov: f32,
    aspect: f32,
) -> Ray {
    let direction = generate_ray(uv, origin, target, up, fov, aspect);
    return Ray(origin, direction);
}
"""


def generate_ray_wgsl_inline(camera: CameraNode) -> str:
    """
    Generate WGSL code with camera parameters inlined as constants.

    This is useful for scenes with a fixed camera where the parameters
    don't change at runtime.

    Args:
        camera: Camera node with all parameters.

    Returns:
        WGSL code string with inlined camera constants.
    """
    params = CameraParams.from_camera_node(camera)

    # Pre-compute camera basis
    forward = (params.look_at - params.origin).normalized()
    right = forward.cross(params.up).normalized()
    up = right.cross(forward)

    fov_rad = math.radians(params.fov)
    half_height = math.tan(fov_rad * 0.5)
    half_width = half_height * params.aspect_ratio

    return f"""\
/// Ray structure for ray marching.
struct Ray {{
    origin: vec3<f32>,
    direction: vec3<f32>,
}}

// Camera constants (inlined from CameraNode)
const CAMERA_ORIGIN: vec3<f32> = vec3<f32>({params.origin.x}, {params.origin.y}, {params.origin.z});
const CAMERA_FORWARD: vec3<f32> = vec3<f32>({forward.x}, {forward.y}, {forward.z});
const CAMERA_RIGHT: vec3<f32> = vec3<f32>({right.x}, {right.y}, {right.z});
const CAMERA_UP: vec3<f32> = vec3<f32>({up.x}, {up.y}, {up.z});
const CAMERA_HALF_WIDTH: f32 = {half_width};
const CAMERA_HALF_HEIGHT: f32 = {half_height};

/// Generates a camera ray with inlined camera parameters.
///
/// Arguments:
///   uv: Normalized screen coordinates in [-1, 1]
///
/// Returns:
///   Normalized ray direction vector
fn generate_ray_inline(uv: vec2<f32>) -> vec3<f32> {{
    let rd = normalize(
        CAMERA_FORWARD +
        CAMERA_RIGHT * uv.x * CAMERA_HALF_WIDTH +
        CAMERA_UP * uv.y * CAMERA_HALF_HEIGHT
    );
    return rd;
}}

/// Creates a Ray struct using inlined camera parameters.
fn create_ray_inline(uv: vec2<f32>) -> Ray {{
    return Ray(CAMERA_ORIGIN, generate_ray_inline(uv));
}}
"""


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_camera(camera: CameraNode) -> list[str]:
    """
    Validate camera parameters for ray generation.

    Args:
        camera: Camera node to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []
    params = CameraParams.from_camera_node(camera)

    # Check FOV range
    if params.fov <= 0.0:
        errors.append(f"FOV must be positive, got {params.fov}")
    elif params.fov >= 180.0:
        errors.append(f"FOV must be less than 180 degrees, got {params.fov}")

    # Check aspect ratio
    if params.aspect_ratio <= 0.0:
        errors.append(f"Aspect ratio must be positive, got {params.aspect_ratio}")

    # Check look-at not same as origin
    if params.origin.approx_equal(params.look_at):
        errors.append("Camera origin and look_at must be different")

    # Check up vector not zero
    if params.up.length() < 1e-6:
        errors.append("Up vector cannot be zero")

    # Check up not parallel to forward
    forward = (params.look_at - params.origin).normalized()
    cross = forward.cross(params.up)
    if cross.length() < 1e-6:
        errors.append("Up vector cannot be parallel to view direction")

    return errors


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    "Ray",
    "RayGenerator",
    "Vec3",
    "CameraParams",
    "generate_ray_wgsl",
    "generate_ray_wgsl_inline",
    "validate_camera",
]

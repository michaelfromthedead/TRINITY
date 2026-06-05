"""RT Reflection Ray Generation System (T-GIR-P8.1).

This module implements hardware ray-traced reflections using the RT pipeline:
- GBufferReader: Read G-Buffer data per pixel
- ReflectionRayGenerator: Compute reflection rays from view direction
- RTReflectionTracer: Interface to TLAS ray queries
- RTReflectionPass: Full-screen RT reflection pass
- RTReflectionConfig: Configuration for RT reflections

The implementation:
1. Reads G-Buffer per pixel (depth, normal, material)
2. Reconstructs world position from depth
3. Computes reflection direction: R = 2(N·V)N - V
4. Roughness-based skip (threshold 0.7)
5. Traces ray against TLAS
6. Returns hit data or environment fallback

References:
    - Section 6.11 Ray Tracing Architecture in RENDERING_CONTEXT.md
    - T-GIR-P8.1 RT Reflection Ray Generation spec
    - S10 TLAS/SBT dependencies from pass_node.py
"""

from __future__ import annotations

import math
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Tuple

from engine.core.math.vec import Vec2, Vec3, Vec4

if TYPE_CHECKING:
    from engine.rendering.framegraph.pass_node import RayTracingPass
    from engine.rendering.framegraph.resource_manager import ResourceHandle


# =============================================================================
# Constants
# =============================================================================

# Default roughness threshold for skipping RT
DEFAULT_ROUGHNESS_THRESHOLD = 0.7

# Default maximum ray distance
DEFAULT_MAX_RAY_DISTANCE = 100.0

# Self-intersection bias (offset along normal)
DEFAULT_NORMAL_BIAS = 0.001

# Ray flags for hardware RT
RAY_FLAG_NONE = 0
RAY_FLAG_CULL_BACK_FACING = 1 << 0
RAY_FLAG_CULL_FRONT_FACING = 1 << 1
RAY_FLAG_TERMINATE_ON_FIRST_HIT = 1 << 2
RAY_FLAG_SKIP_CLOSEST_HIT = 1 << 3
RAY_FLAG_ACCEPT_FIRST_HIT = 1 << 4

# Resolution scale modes
RESOLUTION_QUARTER = 0.25
RESOLUTION_HALF = 0.5
RESOLUTION_FULL = 1.0

# Default environment color (skybox fallback)
DEFAULT_ENVIRONMENT_COLOR = Vec3(0.2, 0.3, 0.5)


# =============================================================================
# Resolution Mode
# =============================================================================


class ResolutionMode(Enum):
    """Resolution scaling modes for RT reflections."""

    QUARTER = auto()
    """Quarter resolution (0.25x)."""

    HALF = auto()
    """Half resolution (0.5x)."""

    FULL = auto()
    """Full resolution (1.0x)."""


_RESOLUTION_SCALES = {
    ResolutionMode.QUARTER: 0.25,
    ResolutionMode.HALF: 0.5,
    ResolutionMode.FULL: 1.0,
}


# =============================================================================
# Material Data
# =============================================================================


@dataclass
class MaterialData:
    """Material properties read from G-Buffer.

    Attributes:
        roughness: Surface roughness [0, 1].
        metallic: Metallic factor [0, 1].
        base_color: Base color RGB.
        emissive: Emissive color RGB.
        specular: Specular intensity.
        ior: Index of refraction.
    """

    roughness: float = 0.5
    metallic: float = 0.0
    base_color: Vec3 = field(default_factory=lambda: Vec3(0.5, 0.5, 0.5))
    emissive: Vec3 = field(default_factory=Vec3.zero)
    specular: float = 0.5
    ior: float = 1.5

    def is_reflective(self, threshold: float = DEFAULT_ROUGHNESS_THRESHOLD) -> bool:
        """Check if material is reflective enough for RT.

        Args:
            threshold: Roughness threshold (skip if above).

        Returns:
            True if roughness <= threshold.
        """
        return self.roughness <= threshold


# =============================================================================
# G-Buffer Reader
# =============================================================================


@dataclass
class GBufferPixel:
    """Data read from G-Buffer at a single pixel.

    Attributes:
        depth: Linear depth value.
        world_position: Reconstructed world-space position.
        normal: World-space surface normal.
        material: Material properties.
        velocity: Screen-space motion vector.
        valid: Whether the pixel contains valid data.
    """

    depth: float = 0.0
    world_position: Vec3 = field(default_factory=Vec3.zero)
    normal: Vec3 = field(default_factory=lambda: Vec3(0.0, 1.0, 0.0))
    material: MaterialData = field(default_factory=MaterialData)
    velocity: Vec2 = field(default_factory=Vec2.zero)
    valid: bool = True


class GBufferReader:
    """Reads and interprets G-Buffer data.

    Provides methods to:
    - Read depth and reconstruct world position
    - Read surface normals
    - Read material properties (roughness, metallic)

    Usage:
        reader = GBufferReader(depth_buffer, normal_buffer, material_buffer)
        reader.set_camera(inv_view_proj, camera_pos)
        pixel = reader.read_pixel(uv)
        world_pos = reader.reconstruct_world_pos(uv, depth)
    """

    def __init__(
        self,
        depth_buffer: Optional[Any] = None,
        normal_buffer: Optional[Any] = None,
        material_buffer: Optional[Any] = None,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        """Initialize the G-Buffer reader.

        Args:
            depth_buffer: Depth buffer texture.
            normal_buffer: World-space normal buffer.
            material_buffer: Material property buffer (roughness, metallic).
            width: Buffer width in pixels.
            height: Buffer height in pixels.
        """
        self._depth_buffer = depth_buffer
        self._normal_buffer = normal_buffer
        self._material_buffer = material_buffer
        self._width = width
        self._height = height

        # Camera matrices
        self._inv_view_proj: Optional[list[list[float]]] = None
        self._camera_position = Vec3.zero()
        self._near_plane = 0.1
        self._far_plane = 1000.0

        # Simulated buffer data for testing
        self._test_depth_data: dict[Tuple[int, int], float] = {}
        self._test_normal_data: dict[Tuple[int, int], Vec3] = {}
        self._test_material_data: dict[Tuple[int, int], MaterialData] = {}

    @property
    def width(self) -> int:
        """Get buffer width."""
        return self._width

    @property
    def height(self) -> int:
        """Get buffer height."""
        return self._height

    @property
    def camera_position(self) -> Vec3:
        """Get camera world position."""
        return self._camera_position

    def set_camera(
        self,
        inv_view_proj: list[list[float]],
        camera_position: Vec3,
        near: float = 0.1,
        far: float = 1000.0,
    ) -> None:
        """Set camera matrices for world position reconstruction.

        Args:
            inv_view_proj: Inverse view-projection matrix (4x4).
            camera_position: Camera world position.
            near: Near plane distance.
            far: Far plane distance.
        """
        self._inv_view_proj = inv_view_proj
        self._camera_position = camera_position
        self._near_plane = near
        self._far_plane = far

    def set_test_data(
        self,
        x: int,
        y: int,
        depth: float,
        normal: Vec3,
        material: Optional[MaterialData] = None,
    ) -> None:
        """Set test data for a pixel (for testing without GPU).

        Args:
            x: Pixel X coordinate.
            y: Pixel Y coordinate.
            depth: Linear depth value.
            normal: World-space normal.
            material: Material data.
        """
        self._test_depth_data[(x, y)] = depth
        self._test_normal_data[(x, y)] = normal
        self._test_material_data[(x, y)] = material or MaterialData()

    def read_depth(self, x: int, y: int) -> float:
        """Read depth at pixel coordinates.

        Args:
            x: Pixel X coordinate.
            y: Pixel Y coordinate.

        Returns:
            Linear depth value.
        """
        if (x, y) in self._test_depth_data:
            return self._test_depth_data[(x, y)]

        # In production, sample from depth_buffer texture
        return 1.0  # Default far depth

    def read_depth_uv(self, uv: Vec2) -> float:
        """Read depth at UV coordinates.

        Args:
            uv: UV coordinates [0, 1].

        Returns:
            Linear depth value.
        """
        x = int(uv.x * (self._width - 1))
        y = int(uv.y * (self._height - 1))
        return self.read_depth(x, y)

    def read_normal(self, x: int, y: int) -> Vec3:
        """Read world-space normal at pixel coordinates.

        Args:
            x: Pixel X coordinate.
            y: Pixel Y coordinate.

        Returns:
            World-space normal vector (normalized).
        """
        if (x, y) in self._test_normal_data:
            return self._test_normal_data[(x, y)].normalized()

        # In production, sample from normal_buffer texture
        return Vec3(0.0, 1.0, 0.0)

    def read_normal_uv(self, uv: Vec2) -> Vec3:
        """Read world-space normal at UV coordinates.

        Args:
            uv: UV coordinates [0, 1].

        Returns:
            World-space normal vector.
        """
        x = int(uv.x * (self._width - 1))
        y = int(uv.y * (self._height - 1))
        return self.read_normal(x, y)

    def get_material_at(self, x: int, y: int) -> MaterialData:
        """Read material properties at pixel coordinates.

        Args:
            x: Pixel X coordinate.
            y: Pixel Y coordinate.

        Returns:
            Material data (roughness, metallic, etc.).
        """
        if (x, y) in self._test_material_data:
            return self._test_material_data[(x, y)]

        # In production, sample from material_buffer texture
        return MaterialData()

    def get_material_at_uv(self, uv: Vec2) -> MaterialData:
        """Read material properties at UV coordinates.

        Args:
            uv: UV coordinates [0, 1].

        Returns:
            Material data.
        """
        x = int(uv.x * (self._width - 1))
        y = int(uv.y * (self._height - 1))
        return self.get_material_at(x, y)

    def reconstruct_world_pos(self, uv: Vec2, depth: float) -> Vec3:
        """Reconstruct world position from UV and depth.

        Uses inverse view-projection matrix to transform clip-space
        position back to world space.

        Args:
            uv: Screen UV coordinates [0, 1].
            depth: Linear depth value.

        Returns:
            World-space position.
        """
        if self._inv_view_proj is None:
            # Fallback: simple linear interpolation
            return Vec3(
                (uv.x - 0.5) * depth * 2.0,
                (0.5 - uv.y) * depth * 2.0,
                -depth,
            )

        # Convert UV to clip space [-1, 1]
        clip_x = uv.x * 2.0 - 1.0
        clip_y = 1.0 - uv.y * 2.0  # Flip Y

        # Convert linear depth to NDC depth
        ndc_z = self._linear_to_ndc_depth(depth)

        # Clip space position
        clip_pos = [clip_x, clip_y, ndc_z, 1.0]

        # Transform by inverse view-projection
        world = self._transform_point(clip_pos, self._inv_view_proj)

        return Vec3(world[0], world[1], world[2])

    def reconstruct_world_pos_pixel(self, x: int, y: int) -> Vec3:
        """Reconstruct world position from pixel coordinates.

        Args:
            x: Pixel X coordinate.
            y: Pixel Y coordinate.

        Returns:
            World-space position.
        """
        uv = Vec2(
            (x + 0.5) / self._width,
            (y + 0.5) / self._height,
        )
        depth = self.read_depth(x, y)
        return self.reconstruct_world_pos(uv, depth)

    def read_pixel(self, uv: Vec2) -> GBufferPixel:
        """Read all G-Buffer data at UV coordinates.

        Args:
            uv: UV coordinates [0, 1].

        Returns:
            Complete G-Buffer pixel data.
        """
        x = int(uv.x * (self._width - 1))
        y = int(uv.y * (self._height - 1))

        depth = self.read_depth(x, y)

        # Check if valid (depth > 0 and not at far plane)
        valid = 0.0 < depth < self._far_plane

        return GBufferPixel(
            depth=depth,
            world_position=self.reconstruct_world_pos(uv, depth) if valid else Vec3.zero(),
            normal=self.read_normal(x, y),
            material=self.get_material_at(x, y),
            velocity=Vec2.zero(),  # Would read from velocity buffer
            valid=valid,
        )

    def _linear_to_ndc_depth(self, linear_depth: float) -> float:
        """Convert linear depth to NDC depth.

        Args:
            linear_depth: Linear depth value.

        Returns:
            NDC depth [-1, 1] or [0, 1] depending on convention.
        """
        # Standard perspective projection depth conversion
        near = self._near_plane
        far = self._far_plane

        if linear_depth <= 0:
            return 0.0

        # Convert to [0, 1] range
        ndc = (far - linear_depth) / (far - near)
        return max(0.0, min(1.0, ndc))

    def _transform_point(
        self, point: list[float], matrix: list[list[float]]
    ) -> list[float]:
        """Transform a 4D point by a 4x4 matrix.

        Args:
            point: [x, y, z, w] point.
            matrix: 4x4 transformation matrix.

        Returns:
            Transformed and perspective-divided [x, y, z].
        """
        result = [0.0, 0.0, 0.0, 0.0]

        for i in range(4):
            for j in range(4):
                result[i] += matrix[i][j] * point[j]

        # Perspective divide
        if abs(result[3]) > 1e-10:
            return [result[0] / result[3], result[1] / result[3], result[2] / result[3]]

        return [result[0], result[1], result[2]]


# =============================================================================
# Reflection Ray Generator
# =============================================================================


@dataclass
class ReflectionRay:
    """A reflection ray to be traced.

    Attributes:
        origin: Ray origin (world space).
        direction: Ray direction (normalized).
        should_trace: Whether ray should be traced (roughness check).
        pixel_uv: Source pixel UV coordinates.
        roughness: Surface roughness at origin.
    """

    origin: Vec3 = field(default_factory=Vec3.zero)
    direction: Vec3 = field(default_factory=lambda: Vec3(0.0, 1.0, 0.0))
    should_trace: bool = True
    pixel_uv: Vec2 = field(default_factory=Vec2.zero)
    roughness: float = 0.0


class ReflectionRayGenerator:
    """Generates reflection rays from G-Buffer data.

    Computes:
    - View direction V = normalize(camera_pos - world_pos)
    - Reflection direction R = 2(N·V)N - V
    - Ray origin offset along normal to avoid self-intersection

    Usage:
        generator = ReflectionRayGenerator(config)
        ray = generator.generate_ray(world_pos, normal, view_dir, roughness)
        if generator.should_trace(roughness):
            result = tracer.trace_ray(ray)
    """

    def __init__(
        self,
        roughness_threshold: float = DEFAULT_ROUGHNESS_THRESHOLD,
        normal_bias: float = DEFAULT_NORMAL_BIAS,
    ) -> None:
        """Initialize the ray generator.

        Args:
            roughness_threshold: Skip RT if roughness > threshold.
            normal_bias: Offset along normal to avoid self-intersection.
        """
        self._roughness_threshold = roughness_threshold
        self._normal_bias = normal_bias

    @property
    def roughness_threshold(self) -> float:
        """Get roughness skip threshold."""
        return self._roughness_threshold

    @roughness_threshold.setter
    def roughness_threshold(self, value: float) -> None:
        """Set roughness skip threshold."""
        self._roughness_threshold = max(0.0, min(1.0, value))

    @property
    def normal_bias(self) -> float:
        """Get normal bias for ray origin offset."""
        return self._normal_bias

    def should_trace(self, roughness: float) -> bool:
        """Check if roughness is low enough to trace.

        Per spec: skip RT if roughness > 0.7 (default threshold).

        Args:
            roughness: Surface roughness [0, 1].

        Returns:
            True if roughness <= threshold.
        """
        return roughness <= self._roughness_threshold

    def compute_view_direction(
        self, world_pos: Vec3, camera_pos: Vec3
    ) -> Vec3:
        """Compute view direction from surface to camera.

        Args:
            world_pos: Surface world position.
            camera_pos: Camera world position.

        Returns:
            Normalized view direction.
        """
        view = camera_pos - world_pos
        return view.normalized()

    def compute_reflection_direction(
        self, normal: Vec3, view_dir: Vec3
    ) -> Vec3:
        """Compute reflection direction: R = 2(N·V)N - V.

        Args:
            normal: Surface normal (normalized).
            view_dir: View direction (normalized, pointing toward camera).

        Returns:
            Reflection direction (normalized).
        """
        # R = 2(N·V)N - V
        n_dot_v = normal.dot(view_dir)

        # Handle edge case: view from behind surface
        if n_dot_v < 0:
            # Flip normal for back-facing
            normal = -normal
            n_dot_v = -n_dot_v

        reflection = normal * (2.0 * n_dot_v) - view_dir
        return reflection.normalized()

    def get_ray_origin(self, world_pos: Vec3, normal: Vec3) -> Vec3:
        """Get ray origin with normal bias to avoid self-intersection.

        Args:
            world_pos: Surface world position.
            normal: Surface normal.

        Returns:
            Biased ray origin.
        """
        return world_pos + normal * self._normal_bias

    def get_ray_direction(
        self,
        world_pos: Vec3,
        normal: Vec3,
        camera_pos: Vec3,
    ) -> Vec3:
        """Get reflection direction for a surface point.

        Args:
            world_pos: Surface world position.
            normal: Surface normal.
            camera_pos: Camera position.

        Returns:
            Reflection direction.
        """
        view_dir = self.compute_view_direction(world_pos, camera_pos)
        return self.compute_reflection_direction(normal, view_dir)

    def generate_ray(
        self,
        world_pos: Vec3,
        normal: Vec3,
        camera_pos: Vec3,
        roughness: float,
        pixel_uv: Optional[Vec2] = None,
    ) -> ReflectionRay:
        """Generate a complete reflection ray.

        Args:
            world_pos: Surface world position.
            normal: Surface normal.
            camera_pos: Camera position.
            roughness: Surface roughness.
            pixel_uv: Optional source pixel UV.

        Returns:
            Complete reflection ray.
        """
        should_trace = self.should_trace(roughness)

        if not should_trace:
            return ReflectionRay(
                origin=world_pos,
                direction=Vec3(0.0, 1.0, 0.0),
                should_trace=False,
                pixel_uv=pixel_uv or Vec2.zero(),
                roughness=roughness,
            )

        origin = self.get_ray_origin(world_pos, normal)
        direction = self.get_ray_direction(world_pos, normal, camera_pos)

        return ReflectionRay(
            origin=origin,
            direction=direction,
            should_trace=True,
            pixel_uv=pixel_uv or Vec2.zero(),
            roughness=roughness,
        )


# =============================================================================
# Ray Hit Info
# =============================================================================


@dataclass
class RayHitInfo:
    """Information about a ray hit.

    Attributes:
        hit: Whether the ray hit geometry.
        distance: Distance to hit point (or max_distance if miss).
        position: World position of hit.
        normal: Surface normal at hit point.
        material: Material at hit point.
        instance_id: TLAS instance ID.
        primitive_id: Primitive (triangle) ID.
        barycentrics: Barycentric coordinates of hit.
    """

    hit: bool = False
    distance: float = DEFAULT_MAX_RAY_DISTANCE
    position: Vec3 = field(default_factory=Vec3.zero)
    normal: Vec3 = field(default_factory=lambda: Vec3(0.0, 1.0, 0.0))
    material: MaterialData = field(default_factory=MaterialData)
    instance_id: int = 0
    primitive_id: int = 0
    barycentrics: Vec2 = field(default_factory=Vec2.zero)


# =============================================================================
# TLAS Interface
# =============================================================================


class TLASInterface(Protocol):
    """Protocol for TLAS (Top-Level Acceleration Structure) access.

    Implementations must provide ray traversal against scene geometry.
    This mirrors the TLASInterface from ddgi_rt_probes.py.
    """

    def trace_ray(
        self,
        origin: Vec3,
        direction: Vec3,
        max_distance: float,
        flags: int = RAY_FLAG_NONE,
        mask: int = 0xFFFFFFFF,
    ) -> RayHitInfo:
        """Trace a ray against the TLAS.

        Args:
            origin: Ray origin.
            direction: Ray direction (normalized).
            max_distance: Maximum trace distance.
            flags: Ray flags.
            mask: Instance mask for filtering.

        Returns:
            Ray hit information.
        """
        ...

    def is_valid(self) -> bool:
        """Check if the TLAS is valid and ready for tracing."""
        ...


# =============================================================================
# RT Reflection Tracer
# =============================================================================


class RTReflectionTracer:
    """Traces reflection rays against TLAS.

    Handles:
    - Single ray tracing against acceleration structure
    - Hit point data extraction
    - Miss shader (environment/skybox fallback)

    Usage:
        tracer = RTReflectionTracer(tlas, config)
        hit = tracer.trace_ray(ray)
        if not hit.hit:
            color = tracer.on_miss(ray.direction)
    """

    def __init__(
        self,
        tlas: Optional[TLASInterface] = None,
        max_ray_distance: float = DEFAULT_MAX_RAY_DISTANCE,
        environment_color: Optional[Vec3] = None,
        environment_sampler: Optional[Callable[[Vec3], Vec3]] = None,
    ) -> None:
        """Initialize the reflection tracer.

        Args:
            tlas: TLAS interface for ray traversal.
            max_ray_distance: Maximum ray trace distance.
            environment_color: Fallback environment color.
            environment_sampler: Optional environment map sampler.
        """
        self._tlas = tlas
        self._max_ray_distance = max_ray_distance
        self._environment_color = environment_color or DEFAULT_ENVIRONMENT_COLOR
        self._environment_sampler = environment_sampler

        # Statistics
        self._rays_traced = 0
        self._hits = 0
        self._misses = 0

    @property
    def tlas(self) -> Optional[TLASInterface]:
        """Get the TLAS interface."""
        return self._tlas

    @tlas.setter
    def tlas(self, value: TLASInterface) -> None:
        """Set the TLAS interface."""
        self._tlas = value

    @property
    def max_ray_distance(self) -> float:
        """Get maximum ray distance."""
        return self._max_ray_distance

    @property
    def environment_color(self) -> Vec3:
        """Get default environment color."""
        return self._environment_color

    @environment_color.setter
    def environment_color(self, value: Vec3) -> None:
        """Set default environment color."""
        self._environment_color = value

    def is_ready(self) -> bool:
        """Check if tracer is ready (TLAS valid).

        Returns:
            True if TLAS exists and is valid.
        """
        return self._tlas is not None and self._tlas.is_valid()

    def trace_ray(
        self,
        ray: ReflectionRay,
        flags: int = RAY_FLAG_NONE,
        mask: int = 0xFFFFFFFF,
    ) -> RayHitInfo:
        """Trace a single reflection ray.

        Args:
            ray: Reflection ray to trace.
            flags: Ray tracing flags.
            mask: Instance mask for filtering.

        Returns:
            Hit information.
        """
        self._rays_traced += 1

        if not ray.should_trace:
            return RayHitInfo(hit=False, distance=self._max_ray_distance)

        if self._tlas is None or not self._tlas.is_valid():
            self._misses += 1
            return RayHitInfo(hit=False, distance=self._max_ray_distance)

        hit_info = self._tlas.trace_ray(
            ray.origin,
            ray.direction,
            self._max_ray_distance,
            flags,
            mask,
        )

        if hit_info.hit:
            self._hits += 1
        else:
            self._misses += 1

        return hit_info

    def get_hit_info(
        self,
        origin: Vec3,
        direction: Vec3,
        flags: int = RAY_FLAG_NONE,
        mask: int = 0xFFFFFFFF,
    ) -> RayHitInfo:
        """Trace ray and get hit info directly.

        Args:
            origin: Ray origin.
            direction: Ray direction.
            flags: Ray flags.
            mask: Instance mask.

        Returns:
            Hit information.
        """
        ray = ReflectionRay(
            origin=origin,
            direction=direction,
            should_trace=True,
        )
        return self.trace_ray(ray, flags, mask)

    def on_miss(self, direction: Vec3) -> Vec3:
        """Get environment color for missed rays.

        Args:
            direction: Ray direction.

        Returns:
            Environment/skybox color.
        """
        if self._environment_sampler is not None:
            return self._environment_sampler(direction)

        # Simple gradient based on direction
        # Blend between ground color and sky color
        t = direction.y * 0.5 + 0.5  # Map Y [-1, 1] to [0, 1]
        t = max(0.0, min(1.0, t))

        # Lerp between ground and sky
        ground = Vec3(0.1, 0.1, 0.1)
        sky = self._environment_color

        return Vec3(
            ground.x * (1.0 - t) + sky.x * t,
            ground.y * (1.0 - t) + sky.y * t,
            ground.z * (1.0 - t) + sky.z * t,
        )

    def get_statistics(self) -> dict:
        """Get tracing statistics.

        Returns:
            Dict with rays_traced, hits, misses.
        """
        return {
            "rays_traced": self._rays_traced,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, self._rays_traced),
        }

    def reset_statistics(self) -> None:
        """Reset tracing statistics."""
        self._rays_traced = 0
        self._hits = 0
        self._misses = 0


# =============================================================================
# RT Reflection Config
# =============================================================================


@dataclass
class RTReflectionConfig:
    """Configuration for RT reflections.

    Attributes:
        max_ray_distance: Maximum ray trace distance.
        roughness_threshold: Skip RT if roughness > threshold (default 0.7).
        resolution_scale: Render scale (0.25, 0.5, 1.0).
        tlas_mask: Instance mask for object filtering.
        enable_transparency: Enable transparency in reflections.
        normal_bias: Ray origin offset along normal.
        max_bounces: Maximum reflection bounces (for recursive RT).
        temporal_accumulation: Enable temporal accumulation.
        denoise: Enable denoising.
    """

    max_ray_distance: float = DEFAULT_MAX_RAY_DISTANCE
    roughness_threshold: float = DEFAULT_ROUGHNESS_THRESHOLD
    resolution_scale: float = RESOLUTION_FULL
    tlas_mask: int = 0xFFFFFFFF
    enable_transparency: bool = False
    normal_bias: float = DEFAULT_NORMAL_BIAS
    max_bounces: int = 1
    temporal_accumulation: bool = True
    denoise: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.max_ray_distance = max(0.1, self.max_ray_distance)
        self.roughness_threshold = max(0.0, min(1.0, self.roughness_threshold))
        self.resolution_scale = max(0.1, min(1.0, self.resolution_scale))
        self.normal_bias = max(0.0, self.normal_bias)
        self.max_bounces = max(1, min(8, self.max_bounces))

    def validate(self) -> list[str]:
        """Validate configuration and return errors.

        Returns:
            List of error messages (empty if valid).
        """
        errors = []

        if self.max_ray_distance <= 0:
            errors.append("max_ray_distance must be positive")

        if not 0.0 <= self.roughness_threshold <= 1.0:
            errors.append("roughness_threshold must be in [0, 1]")

        valid_scales = [0.25, 0.5, 1.0]
        if self.resolution_scale not in valid_scales:
            if not 0.1 <= self.resolution_scale <= 1.0:
                errors.append(
                    f"resolution_scale must be one of {valid_scales} "
                    f"or in range [0.1, 1.0]"
                )

        return errors

    @staticmethod
    def low_quality() -> "RTReflectionConfig":
        """Low quality preset (quarter resolution)."""
        return RTReflectionConfig(
            resolution_scale=RESOLUTION_QUARTER,
            roughness_threshold=0.5,
            max_bounces=1,
            denoise=False,
        )

    @staticmethod
    def medium_quality() -> "RTReflectionConfig":
        """Medium quality preset (half resolution)."""
        return RTReflectionConfig(
            resolution_scale=RESOLUTION_HALF,
            roughness_threshold=0.6,
            max_bounces=1,
        )

    @staticmethod
    def high_quality() -> "RTReflectionConfig":
        """High quality preset (full resolution)."""
        return RTReflectionConfig(
            resolution_scale=RESOLUTION_FULL,
            roughness_threshold=0.7,
            max_bounces=2,
        )


# =============================================================================
# Reflection Output
# =============================================================================


@dataclass
class ReflectionOutput:
    """Output from reflection tracing at a pixel.

    Attributes:
        color: Reflected color (RGB).
        hit_distance: Distance to reflection hit.
        confidence: Confidence/validity of reflection [0, 1].
        roughness: Surface roughness at pixel.
        was_traced: Whether RT was performed.
    """

    color: Vec3 = field(default_factory=Vec3.zero)
    hit_distance: float = 0.0
    confidence: float = 0.0
    roughness: float = 0.0
    was_traced: bool = False


# =============================================================================
# RT Reflection Pass
# =============================================================================


class RTReflectionPass:
    """Full-screen RT reflection rendering pass.

    Processes the entire G-Buffer to generate reflections:
    1. Read G-Buffer per pixel
    2. Skip pixels with roughness > threshold
    3. Generate reflection rays
    4. Trace against TLAS
    5. Output: reflection color + hit distance + confidence

    Supports quarter/half/full resolution modes.

    Usage:
        pass_obj = RTReflectionPass(config, gbuffer_reader, tracer)
        pass_obj.execute()
        buffer = pass_obj.get_reflection_buffer()
    """

    def __init__(
        self,
        config: RTReflectionConfig,
        gbuffer_reader: Optional[GBufferReader] = None,
        tracer: Optional[RTReflectionTracer] = None,
    ) -> None:
        """Initialize the reflection pass.

        Args:
            config: RT reflection configuration.
            gbuffer_reader: G-Buffer reader.
            tracer: RT reflection tracer.
        """
        self._config = config
        self._gbuffer_reader = gbuffer_reader
        self._tracer = tracer
        self._ray_generator = ReflectionRayGenerator(
            config.roughness_threshold,
            config.normal_bias,
        )

        # Output buffers (None until execute)
        self._output_width = 0
        self._output_height = 0
        self._reflection_buffer: list[ReflectionOutput] = []

        # Statistics
        self._pixels_processed = 0
        self._pixels_traced = 0
        self._pixels_skipped = 0
        self._last_execute_time_ms = 0.0

    @property
    def config(self) -> RTReflectionConfig:
        """Get configuration."""
        return self._config

    @config.setter
    def config(self, value: RTReflectionConfig) -> None:
        """Set configuration and update ray generator."""
        self._config = value
        self._ray_generator = ReflectionRayGenerator(
            value.roughness_threshold,
            value.normal_bias,
        )

    @property
    def gbuffer_reader(self) -> Optional[GBufferReader]:
        """Get G-Buffer reader."""
        return self._gbuffer_reader

    @gbuffer_reader.setter
    def gbuffer_reader(self, value: GBufferReader) -> None:
        """Set G-Buffer reader."""
        self._gbuffer_reader = value

    @property
    def tracer(self) -> Optional[RTReflectionTracer]:
        """Get tracer."""
        return self._tracer

    @tracer.setter
    def tracer(self, value: RTReflectionTracer) -> None:
        """Set tracer."""
        self._tracer = value

    @property
    def output_width(self) -> int:
        """Get output buffer width."""
        return self._output_width

    @property
    def output_height(self) -> int:
        """Get output buffer height."""
        return self._output_height

    def set_resolution_scale(self, scale: float) -> None:
        """Set resolution scale.

        Args:
            scale: Resolution scale (0.25, 0.5, 1.0).
        """
        self._config.resolution_scale = max(0.1, min(1.0, scale))

    def _compute_output_dimensions(self) -> Tuple[int, int]:
        """Compute output buffer dimensions based on scale.

        Returns:
            (width, height) tuple.
        """
        if self._gbuffer_reader is None:
            return (0, 0)

        base_width = self._gbuffer_reader.width
        base_height = self._gbuffer_reader.height
        scale = self._config.resolution_scale

        return (
            max(1, int(base_width * scale)),
            max(1, int(base_height * scale)),
        )

    def execute(self) -> None:
        """Execute the RT reflection pass.

        Processes all pixels at the configured resolution.
        Results are stored in the internal reflection buffer.
        """
        if self._gbuffer_reader is None or self._tracer is None:
            raise RuntimeError("GBufferReader and Tracer must be set before execute")

        # Reset statistics
        self._pixels_processed = 0
        self._pixels_traced = 0
        self._pixels_skipped = 0

        # Compute output dimensions
        self._output_width, self._output_height = self._compute_output_dimensions()

        if self._output_width == 0 or self._output_height == 0:
            self._reflection_buffer = []
            return

        # Allocate output buffer
        total_pixels = self._output_width * self._output_height
        self._reflection_buffer = [ReflectionOutput() for _ in range(total_pixels)]

        # Get camera position
        camera_pos = self._gbuffer_reader.camera_position

        # Process each pixel
        for y in range(self._output_height):
            for x in range(self._output_width):
                self._process_pixel(x, y, camera_pos)

    def _process_pixel(self, x: int, y: int, camera_pos: Vec3) -> None:
        """Process a single pixel.

        Args:
            x: Output pixel X coordinate.
            y: Output pixel Y coordinate.
            camera_pos: Camera world position.
        """
        # Convert to UV coordinates
        uv = Vec2(
            (x + 0.5) / self._output_width,
            (y + 0.5) / self._output_height,
        )

        # Read G-Buffer
        pixel = self._gbuffer_reader.read_pixel(uv)
        self._pixels_processed += 1

        # Get output index
        idx = y * self._output_width + x

        # Check if pixel is valid
        if not pixel.valid:
            self._reflection_buffer[idx] = ReflectionOutput(
                color=Vec3.zero(),
                hit_distance=0.0,
                confidence=0.0,
                roughness=pixel.material.roughness,
                was_traced=False,
            )
            self._pixels_skipped += 1
            return

        # Generate reflection ray
        ray = self._ray_generator.generate_ray(
            pixel.world_position,
            pixel.normal,
            camera_pos,
            pixel.material.roughness,
            uv,
        )

        # Check if should trace
        if not ray.should_trace:
            self._reflection_buffer[idx] = ReflectionOutput(
                color=Vec3.zero(),
                hit_distance=0.0,
                confidence=0.0,
                roughness=pixel.material.roughness,
                was_traced=False,
            )
            self._pixels_skipped += 1
            return

        # Trace ray
        hit_info = self._tracer.trace_ray(ray, mask=self._config.tlas_mask)
        self._pixels_traced += 1

        # Compute output
        if hit_info.hit:
            # Use hit material color (simplified - would do shading in production)
            color = hit_info.material.base_color
            confidence = 1.0
        else:
            # Environment fallback
            color = self._tracer.on_miss(ray.direction)
            confidence = 0.5  # Lower confidence for environment

        self._reflection_buffer[idx] = ReflectionOutput(
            color=color,
            hit_distance=hit_info.distance,
            confidence=confidence,
            roughness=pixel.material.roughness,
            was_traced=True,
        )

    def get_reflection_buffer(self) -> list[ReflectionOutput]:
        """Get the reflection output buffer.

        Returns:
            List of ReflectionOutput for each pixel.
        """
        return self._reflection_buffer

    def get_reflection_at(self, x: int, y: int) -> ReflectionOutput:
        """Get reflection output at pixel coordinates.

        Args:
            x: Pixel X coordinate.
            y: Pixel Y coordinate.

        Returns:
            ReflectionOutput at that pixel.
        """
        if not self._reflection_buffer:
            return ReflectionOutput()

        if x < 0 or x >= self._output_width or y < 0 or y >= self._output_height:
            return ReflectionOutput()

        idx = y * self._output_width + x
        return self._reflection_buffer[idx]

    def get_reflection_at_uv(self, uv: Vec2) -> ReflectionOutput:
        """Get reflection output at UV coordinates.

        Args:
            uv: UV coordinates [0, 1].

        Returns:
            ReflectionOutput at that location.
        """
        x = int(uv.x * (self._output_width - 1))
        y = int(uv.y * (self._output_height - 1))
        return self.get_reflection_at(x, y)

    def get_statistics(self) -> dict:
        """Get pass execution statistics.

        Returns:
            Dict with pixels_processed, pixels_traced, etc.
        """
        return {
            "pixels_processed": self._pixels_processed,
            "pixels_traced": self._pixels_traced,
            "pixels_skipped": self._pixels_skipped,
            "trace_rate": self._pixels_traced / max(1, self._pixels_processed),
            "output_width": self._output_width,
            "output_height": self._output_height,
            "resolution_scale": self._config.resolution_scale,
        }


# =============================================================================
# Mock TLAS for Testing
# =============================================================================


class MockTLAS:
    """Mock TLAS implementation for testing.

    Provides simple geometric primitives (spheres, boxes) for
    testing ray tracing without actual GPU acceleration structures.
    """

    def __init__(self) -> None:
        """Initialize mock TLAS."""
        self._valid = True
        self._spheres: list[Tuple[Vec3, float, MaterialData]] = []
        self._boxes: list[Tuple[Vec3, Vec3, MaterialData]] = []

    def add_sphere(
        self,
        center: Vec3,
        radius: float,
        material: Optional[MaterialData] = None,
    ) -> None:
        """Add a sphere primitive.

        Args:
            center: Sphere center.
            radius: Sphere radius.
            material: Surface material.
        """
        self._spheres.append((center, radius, material or MaterialData()))

    def add_box(
        self,
        min_corner: Vec3,
        max_corner: Vec3,
        material: Optional[MaterialData] = None,
    ) -> None:
        """Add an axis-aligned box primitive.

        Args:
            min_corner: Box minimum corner.
            max_corner: Box maximum corner.
            material: Surface material.
        """
        self._boxes.append((min_corner, max_corner, material or MaterialData()))

    def set_valid(self, valid: bool) -> None:
        """Set TLAS validity."""
        self._valid = valid

    def is_valid(self) -> bool:
        """Check if TLAS is valid."""
        return self._valid

    def trace_ray(
        self,
        origin: Vec3,
        direction: Vec3,
        max_distance: float,
        flags: int = RAY_FLAG_NONE,
        mask: int = 0xFFFFFFFF,
    ) -> RayHitInfo:
        """Trace a ray against the mock geometry.

        Args:
            origin: Ray origin.
            direction: Ray direction.
            max_distance: Maximum trace distance.
            flags: Ray flags (ignored in mock).
            mask: Instance mask (ignored in mock).

        Returns:
            Ray hit information.
        """
        closest_hit = RayHitInfo(hit=False, distance=max_distance)
        direction = direction.normalized()

        # Test spheres
        for center, radius, material in self._spheres:
            hit = self._intersect_sphere(origin, direction, center, radius, max_distance)
            if hit is not None and hit < closest_hit.distance:
                hit_pos = origin + direction * hit
                normal = (hit_pos - center).normalized()
                closest_hit = RayHitInfo(
                    hit=True,
                    distance=hit,
                    position=hit_pos,
                    normal=normal,
                    material=material,
                )

        # Test boxes
        for min_corner, max_corner, material in self._boxes:
            hit, normal = self._intersect_box(
                origin, direction, min_corner, max_corner, max_distance
            )
            if hit is not None and hit < closest_hit.distance:
                hit_pos = origin + direction * hit
                closest_hit = RayHitInfo(
                    hit=True,
                    distance=hit,
                    position=hit_pos,
                    normal=normal,
                    material=material,
                )

        return closest_hit

    def _intersect_sphere(
        self,
        origin: Vec3,
        direction: Vec3,
        center: Vec3,
        radius: float,
        max_distance: float,
    ) -> Optional[float]:
        """Ray-sphere intersection.

        Returns:
            Hit distance or None.
        """
        oc = origin - center
        a = direction.dot(direction)
        b = 2.0 * oc.dot(direction)
        c = oc.dot(oc) - radius * radius

        discriminant = b * b - 4.0 * a * c

        if discriminant < 0:
            return None

        sqrt_d = math.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)

        if t1 > 0 and t1 < max_distance:
            return t1
        if t2 > 0 and t2 < max_distance:
            return t2

        return None

    def _intersect_box(
        self,
        origin: Vec3,
        direction: Vec3,
        min_corner: Vec3,
        max_corner: Vec3,
        max_distance: float,
    ) -> Tuple[Optional[float], Vec3]:
        """Ray-AABB intersection.

        Returns:
            (hit_distance, normal) or (None, zero_normal).
        """
        inv_dir = Vec3(
            1.0 / direction.x if abs(direction.x) > 1e-10 else 1e10,
            1.0 / direction.y if abs(direction.y) > 1e-10 else 1e10,
            1.0 / direction.z if abs(direction.z) > 1e-10 else 1e10,
        )

        t1 = (min_corner.x - origin.x) * inv_dir.x
        t2 = (max_corner.x - origin.x) * inv_dir.x
        t3 = (min_corner.y - origin.y) * inv_dir.y
        t4 = (max_corner.y - origin.y) * inv_dir.y
        t5 = (min_corner.z - origin.z) * inv_dir.z
        t6 = (max_corner.z - origin.z) * inv_dir.z

        tmin = max(min(t1, t2), min(t3, t4), min(t5, t6))
        tmax = min(max(t1, t2), max(t3, t4), max(t5, t6))

        if tmax < 0 or tmin > tmax or tmin > max_distance:
            return (None, Vec3.zero())

        t = tmin if tmin > 0 else tmax
        if t < 0 or t > max_distance:
            return (None, Vec3.zero())

        # Compute normal (which face was hit)
        hit_pos = origin + direction * t
        normal = Vec3.zero()

        eps = 0.001
        if abs(hit_pos.x - min_corner.x) < eps:
            normal = Vec3(-1, 0, 0)
        elif abs(hit_pos.x - max_corner.x) < eps:
            normal = Vec3(1, 0, 0)
        elif abs(hit_pos.y - min_corner.y) < eps:
            normal = Vec3(0, -1, 0)
        elif abs(hit_pos.y - max_corner.y) < eps:
            normal = Vec3(0, 1, 0)
        elif abs(hit_pos.z - min_corner.z) < eps:
            normal = Vec3(0, 0, -1)
        elif abs(hit_pos.z - max_corner.z) < eps:
            normal = Vec3(0, 0, 1)

        return (t, normal)

    def clear(self) -> None:
        """Clear all primitives."""
        self._spheres.clear()
        self._boxes.clear()


# =============================================================================
# WGSL Shader Generation
# =============================================================================


def generate_rt_reflections_rgen_wgsl(config: RTReflectionConfig) -> str:
    """Generate WGSL ray generation shader for RT reflections.

    Args:
        config: RT reflection configuration.

    Returns:
        WGSL shader source for rt_reflections.rgen.
    """
    return f"""// RT Reflections Ray Generation Shader (rt_reflections.rgen)
// Generated for T-GIR-P8.1 RT Reflection Ray Generation

// Bindings
@group(0) @binding(0) var<storage, read_write> reflection_output: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> hit_distance_output: array<f32>;
@group(0) @binding(2) var depth_buffer: texture_2d<f32>;
@group(0) @binding(3) var normal_buffer: texture_2d<f32>;
@group(0) @binding(4) var material_buffer: texture_2d<f32>;
@group(0) @binding(5) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var tlas: acceleration_structure;
@group(1) @binding(1) var environment_map: texture_cube<f32>;
@group(1) @binding(2) var env_sampler: sampler;

struct CameraUniforms {{
    inv_view_proj: mat4x4<f32>,
    camera_position: vec3<f32>,
    near_plane: f32,
    far_plane: f32,
    resolution: vec2<f32>,
    _pad: vec2<f32>,
}}

struct RayPayload {{
    color: vec3<f32>,
    hit_distance: f32,
    hit: bool,
}}

// Constants
const MAX_RAY_DISTANCE: f32 = {config.max_ray_distance};
const ROUGHNESS_THRESHOLD: f32 = {config.roughness_threshold};
const NORMAL_BIAS: f32 = {config.normal_bias};
const RESOLUTION_SCALE: f32 = {config.resolution_scale};

// Reconstruct world position from depth
fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {{
    let clip_x = uv.x * 2.0 - 1.0;
    let clip_y = 1.0 - uv.y * 2.0;
    let ndc_z = (camera.far_plane - depth) / (camera.far_plane - camera.near_plane);
    let clip_pos = vec4<f32>(clip_x, clip_y, ndc_z, 1.0);
    let world_pos = camera.inv_view_proj * clip_pos;
    return world_pos.xyz / world_pos.w;
}}

// Compute reflection direction: R = 2(N·V)N - V
fn compute_reflection(normal: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {{
    let n_dot_v = dot(normal, view_dir);
    return 2.0 * n_dot_v * normal - view_dir;
}}

// Sample environment map for missed rays
fn sample_environment(direction: vec3<f32>) -> vec3<f32> {{
    return textureSample(environment_map, env_sampler, direction).rgb;
}}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let output_size = vec2<u32>(u32(camera.resolution.x * RESOLUTION_SCALE),
                                 u32(camera.resolution.y * RESOLUTION_SCALE));

    if (global_id.x >= output_size.x || global_id.y >= output_size.y) {{
        return;
    }}

    let uv = vec2<f32>(f32(global_id.x) + 0.5, f32(global_id.y) + 0.5) / vec2<f32>(output_size);
    let idx = global_id.y * output_size.x + global_id.x;

    // Sample G-Buffer
    let gbuffer_uv = vec2<i32>(i32(uv.x * camera.resolution.x),
                               i32(uv.y * camera.resolution.y));
    let depth = textureLoad(depth_buffer, gbuffer_uv, 0).r;
    let normal = normalize(textureLoad(normal_buffer, gbuffer_uv, 0).xyz * 2.0 - 1.0);
    let material = textureLoad(material_buffer, gbuffer_uv, 0);
    let roughness = material.r;
    let metallic = material.g;

    // Skip if roughness too high
    if (roughness > ROUGHNESS_THRESHOLD || depth <= 0.0) {{
        reflection_output[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        hit_distance_output[idx] = 0.0;
        return;
    }}

    // Reconstruct world position
    let world_pos = reconstruct_world_pos(uv, depth);

    // Compute view and reflection directions
    let view_dir = normalize(camera.camera_position - world_pos);
    let reflect_dir = compute_reflection(normal, view_dir);

    // Offset ray origin along normal
    let ray_origin = world_pos + normal * NORMAL_BIAS;

    // Initialize ray
    var ray: RayDesc;
    ray.Origin = ray_origin;
    ray.Direction = reflect_dir;
    ray.TMin = 0.001;
    ray.TMax = MAX_RAY_DISTANCE;

    // Trace ray
    var payload: RayPayload;
    payload.color = vec3<f32>(0.0);
    payload.hit_distance = MAX_RAY_DISTANCE;
    payload.hit = false;

    // TraceRay call (pseudo-code for WGSL RT extension)
    // traceRayEXT(tlas, 0u, 0xFFu, 0u, 0u, 0u, ray, payload);

    // Output (placeholder for actual tracing result)
    if (payload.hit) {{
        reflection_output[idx] = vec4<f32>(payload.color, 1.0);
    }} else {{
        let env_color = sample_environment(reflect_dir);
        reflection_output[idx] = vec4<f32>(env_color, 0.5);
    }}
    hit_distance_output[idx] = payload.hit_distance;
}}
"""


# =============================================================================
# Utility Functions
# =============================================================================


def estimate_rt_reflection_memory(
    width: int,
    height: int,
    resolution_scale: float = 1.0,
) -> int:
    """Estimate memory usage for RT reflection buffers.

    Args:
        width: Base width in pixels.
        height: Base height in pixels.
        resolution_scale: Resolution scale factor.

    Returns:
        Estimated memory in bytes.
    """
    scaled_width = int(width * resolution_scale)
    scaled_height = int(height * resolution_scale)
    pixels = scaled_width * scaled_height

    # Output buffers:
    # - Reflection color: RGBA16F (8 bytes/pixel)
    # - Hit distance: R32F (4 bytes/pixel)
    # - Confidence: R8 (1 byte/pixel)
    bytes_per_pixel = 8 + 4 + 1

    return pixels * bytes_per_pixel


def create_mock_tlas() -> MockTLAS:
    """Create a mock TLAS with some test geometry.

    Returns:
        MockTLAS with spheres and boxes.
    """
    tlas = MockTLAS()

    # Add some spheres
    tlas.add_sphere(
        Vec3(0, 1, -5),
        1.0,
        MaterialData(roughness=0.2, metallic=0.9, base_color=Vec3(0.8, 0.8, 0.9)),
    )
    tlas.add_sphere(
        Vec3(-3, 0.5, -4),
        0.5,
        MaterialData(roughness=0.5, metallic=0.0, base_color=Vec3(0.9, 0.2, 0.2)),
    )

    # Add a ground plane (box)
    tlas.add_box(
        Vec3(-10, -0.1, -20),
        Vec3(10, 0, 5),
        MaterialData(roughness=0.3, metallic=0.0, base_color=Vec3(0.5, 0.5, 0.5)),
    )

    return tlas


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Constants
    "DEFAULT_ROUGHNESS_THRESHOLD",
    "DEFAULT_MAX_RAY_DISTANCE",
    "DEFAULT_NORMAL_BIAS",
    "DEFAULT_ENVIRONMENT_COLOR",
    "RAY_FLAG_NONE",
    "RAY_FLAG_CULL_BACK_FACING",
    "RAY_FLAG_CULL_FRONT_FACING",
    "RAY_FLAG_TERMINATE_ON_FIRST_HIT",
    "RAY_FLAG_SKIP_CLOSEST_HIT",
    "RAY_FLAG_ACCEPT_FIRST_HIT",
    "RESOLUTION_QUARTER",
    "RESOLUTION_HALF",
    "RESOLUTION_FULL",
    # Enums
    "ResolutionMode",
    # Data structures
    "MaterialData",
    "GBufferPixel",
    "ReflectionRay",
    "RayHitInfo",
    "ReflectionOutput",
    # Config
    "RTReflectionConfig",
    # Core classes
    "GBufferReader",
    "ReflectionRayGenerator",
    "RTReflectionTracer",
    "RTReflectionPass",
    # TLAS
    "TLASInterface",
    "MockTLAS",
    # Utilities
    "generate_rt_reflections_rgen_wgsl",
    "estimate_rt_reflection_memory",
    "create_mock_tlas",
]

"""
Depth of Field (Lens Jitter) for Demoscene Rendering (T-DEMO-3.12).

This module implements depth of field using the thin lens model with
lens jitter ray generation for ray-marched scenes. DOF is achieved by:

  1. Generating multiple rays per pixel with jittered lens positions
  2. Converging rays at the focal plane
  3. Accumulating results over multiple frames

The thin lens model parameters:
  - Aperture: Controls blur amount (larger = more blur)
  - Focal distance: Distance where objects are in focus
  - Focus range: Depth range of acceptable sharpness

Usage:
    >>> from engine.rendering.demoscene.depth_of_field import DOFGenerator, DOFParams
    >>> from engine.rendering.demoscene.ray_generation import Ray, Vec3
    >>> dof = DOFGenerator()
    >>> params = DOFParams(aperture=0.05, focal_distance=5.0, focus_range=2.0)
    >>> # Generate jittered ray for DOF
    >>> jittered_ray = dof.jitter_ray(primary_ray, params, sample_offset)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple, Generator

from .ray_generation import Vec3, Ray, CameraParams
from .ast_nodes import CameraNode


# =============================================================================
# DOF Parameters
# =============================================================================


@dataclass
class DOFParams:
    """
    Depth of Field parameters for thin lens simulation.

    The thin lens model simulates camera optics where:
    - Objects at focal_distance are perfectly sharp
    - Objects closer or further are blurred based on their circle of confusion
    - Aperture controls the amount of blur

    Attributes:
        aperture: Lens aperture radius (controls blur intensity).
                  0.0 = pinhole (infinite DOF, no blur).
                  Typical values: 0.01 - 0.1 for subtle DOF.
        focal_distance: Distance from camera where objects are in focus.
        focus_range: Depth range where objects appear acceptably sharp.
                     Objects within [focal_distance - focus_range/2,
                     focal_distance + focus_range/2] are considered in focus.
        bokeh_shape: Shape of the bokeh ("circle", "hexagon", "octagon").
        samples_per_pixel: Number of samples for DOF accumulation.
    """
    aperture: float = 0.05
    focal_distance: float = 5.0
    focus_range: float = 2.0
    bokeh_shape: str = "circle"
    samples_per_pixel: int = 16

    def is_enabled(self) -> bool:
        """Check if DOF effect is enabled (non-zero aperture)."""
        return self.aperture > 1e-6

    @classmethod
    def from_camera_node(cls, camera: CameraNode) -> "DOFParams":
        """Create DOFParams from camera node attributes."""
        aperture = float(camera.aperture.value) if camera.aperture else 0.0
        focal_distance = float(camera.focal_distance.value) if camera.focal_distance else 10.0
        return cls(
            aperture=aperture,
            focal_distance=focal_distance,
        )


# =============================================================================
# Circle of Confusion
# =============================================================================


def calculate_coc(
    hit_distance: float,
    focal_distance: float,
    aperture: float,
) -> float:
    """
    Calculate the circle of confusion (CoC) for a point at given distance.

    The CoC represents the blur radius for out-of-focus objects.
    Larger CoC = more blur. Zero CoC = perfectly in focus.

    Formula (simplified thin lens):
        CoC = |aperture * (hit_distance - focal_distance) / hit_distance|

    Args:
        hit_distance: Distance from camera to the surface point.
        focal_distance: Camera's focal distance (in-focus plane).
        aperture: Lens aperture radius.

    Returns:
        Circle of confusion radius (blur radius in world units).

    Example:
        >>> calculate_coc(5.0, 5.0, 0.05)  # At focal plane
        0.0
        >>> calculate_coc(10.0, 5.0, 0.05)  # Far from focal plane
        0.025  # approximately
    """
    if aperture <= 0.0:
        return 0.0

    if hit_distance <= 0.0:
        return 0.0

    # Thin lens CoC formula
    coc = abs(aperture * (hit_distance - focal_distance) / hit_distance)

    return coc


def calculate_coc_normalized(
    hit_distance: float,
    focal_distance: float,
    aperture: float,
    focus_range: float,
) -> float:
    """
    Calculate normalized circle of confusion [0, 1].

    0.0 = perfectly in focus (within focus range)
    1.0 = maximum blur

    Args:
        hit_distance: Distance from camera to the surface point.
        focal_distance: Camera's focal distance.
        aperture: Lens aperture radius.
        focus_range: Depth range considered in focus.

    Returns:
        Normalized CoC in [0, 1] range.
    """
    raw_coc = calculate_coc(hit_distance, focal_distance, aperture)

    # Normalize by maximum expected CoC
    # Max CoC occurs at infinity or very close objects
    max_coc = aperture  # Approximate maximum

    if max_coc <= 0.0:
        return 0.0

    return min(1.0, raw_coc / max_coc)


def is_in_focus(
    hit_distance: float,
    focal_distance: float,
    focus_range: float,
) -> bool:
    """
    Check if a point is within the acceptable focus range.

    Args:
        hit_distance: Distance from camera to the point.
        focal_distance: Camera's focal distance.
        focus_range: Total depth range of acceptable sharpness.

    Returns:
        True if the point is within the focus range.
    """
    half_range = focus_range * 0.5
    return abs(hit_distance - focal_distance) <= half_range


# =============================================================================
# Lens Sampling
# =============================================================================


def sample_disk_uniform(u: float, v: float) -> Tuple[float, float]:
    """
    Sample a point uniformly on a unit disk using concentric mapping.

    Uses Shirley & Chiu's concentric disk mapping for uniform distribution.

    Args:
        u: Random value in [0, 1].
        v: Random value in [0, 1].

    Returns:
        Tuple (x, y) on unit disk.
    """
    # Map to [-1, 1]
    a = 2.0 * u - 1.0
    b = 2.0 * v - 1.0

    # Handle degenerate case
    if a == 0.0 and b == 0.0:
        return (0.0, 0.0)

    # Concentric mapping
    if abs(a) > abs(b):
        r = a
        phi = (math.pi / 4.0) * (b / a)
    else:
        r = b
        phi = (math.pi / 2.0) - (math.pi / 4.0) * (a / b)

    return (r * math.cos(phi), r * math.sin(phi))


def sample_disk_stratified(
    sample_index: int,
    total_samples: int,
    jitter: float = 0.5,
) -> Tuple[float, float]:
    """
    Generate stratified sample on unit disk.

    Distributes samples in concentric rings for better coverage.

    Args:
        sample_index: Current sample index [0, total_samples).
        total_samples: Total number of samples.
        jitter: Random jitter amount [0, 1].

    Returns:
        Tuple (x, y) on unit disk.
    """
    if total_samples <= 0:
        return (0.0, 0.0)

    # Stratified sampling with golden ratio spiral
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    # Add jitter
    jittered_index = sample_index + jitter * random.random()

    # Radius increases with sqrt for uniform density
    r = math.sqrt((jittered_index + 0.5) / total_samples)
    theta = jittered_index * golden_angle

    return (r * math.cos(theta), r * math.sin(theta))


def sample_hexagon(u: float, v: float) -> Tuple[float, float]:
    """
    Sample a point on a hexagonal aperture shape.

    Creates hexagonal bokeh instead of circular.

    Args:
        u: Random value in [0, 1].
        v: Random value in [0, 1].

    Returns:
        Tuple (x, y) within hexagon inscribed in unit circle.
    """
    # Start with disk sample
    x, y = sample_disk_uniform(u, v)

    # Transform to hexagonal shape
    angle = math.atan2(y, x)
    sector = (angle + math.pi) / (math.pi / 3.0)
    sector_angle = (sector % 1.0) * (math.pi / 3.0) - (math.pi / 6.0)

    # Scale to fit hexagon
    scale = math.cos(math.pi / 6.0) / math.cos(sector_angle)
    scale = min(scale, 1.0)

    return (x * scale, y * scale)


# =============================================================================
# DOF Generator
# =============================================================================


class DOFGenerator:
    """
    Generates depth-of-field jittered rays using the thin lens model.

    The thin lens model:
    1. Start with a pinhole ray (eye -> pixel)
    2. Find focal point where ray intersects focal plane
    3. Jitter the ray origin on the lens disk
    4. New ray goes from jittered origin through focal point

    This creates natural depth of field when averaging multiple samples.

    Usage::

        dof = DOFGenerator()
        params = DOFParams(aperture=0.05, focal_distance=5.0)

        # For each pixel, accumulate multiple samples
        for sample_idx in range(params.samples_per_pixel):
            offset = dof.get_sample_offset(sample_idx, params.samples_per_pixel)
            jittered_ray = dof.jitter_ray(primary_ray, params, offset, camera_right, camera_up)
            color += trace(jittered_ray)
        color /= params.samples_per_pixel
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the DOF generator.

        Args:
            seed: Random seed for reproducible sampling.
        """
        if seed is not None:
            random.seed(seed)

    def jitter_ray(
        self,
        ray: Ray,
        params: DOFParams,
        sample_offset: Tuple[float, float],
        camera_right: Vec3,
        camera_up: Vec3,
    ) -> Ray:
        """
        Generate a DOF-jittered ray using thin lens model.

        Args:
            ray: Primary ray (from pinhole camera).
            params: DOF parameters.
            sample_offset: Lens sample offset (x, y) in [-1, 1].
            camera_right: Camera's right vector (normalized).
            camera_up: Camera's up vector (normalized).

        Returns:
            New ray with jittered origin converging at focal plane.

        The algorithm:
        1. Find focal point: ray.origin + focal_distance * ray.direction
        2. Offset ray origin by sample_offset * aperture
        3. New direction = normalize(focal_point - new_origin)
        """
        if not params.is_enabled():
            return ray

        # Calculate focal point
        focal_point = ray.origin + ray.direction * params.focal_distance

        # Calculate lens offset
        lens_x, lens_y = sample_offset
        lens_offset = (
            camera_right * (lens_x * params.aperture) +
            camera_up * (lens_y * params.aperture)
        )

        # New ray origin on lens
        new_origin = ray.origin + lens_offset

        # New direction towards focal point
        new_direction = (focal_point - new_origin).normalized()

        return Ray(origin=new_origin, direction=new_direction)

    def jitter_ray_simple(
        self,
        ray: Ray,
        params: DOFParams,
        sample_offset: Tuple[float, float],
    ) -> Ray:
        """
        Generate DOF-jittered ray when camera vectors are not available.

        Derives camera right/up from ray direction using world up.

        Args:
            ray: Primary ray.
            params: DOF parameters.
            sample_offset: Lens sample offset.

        Returns:
            Jittered ray.
        """
        if not params.is_enabled():
            return ray

        # Derive camera basis from ray direction
        world_up = Vec3(0.0, 1.0, 0.0)

        # Handle case where ray is pointing straight up/down
        if abs(ray.direction.dot(world_up)) > 0.999:
            world_up = Vec3(1.0, 0.0, 0.0)

        camera_right = ray.direction.cross(world_up).normalized()
        camera_up = camera_right.cross(ray.direction).normalized()

        return self.jitter_ray(ray, params, sample_offset, camera_right, camera_up)

    def get_sample_offset(
        self,
        sample_index: int,
        total_samples: int,
        bokeh_shape: str = "circle",
    ) -> Tuple[float, float]:
        """
        Get lens sample offset for the given sample index.

        Args:
            sample_index: Current sample index.
            total_samples: Total number of DOF samples.
            bokeh_shape: Shape of bokeh ("circle", "hexagon").

        Returns:
            Tuple (x, y) sample offset in [-1, 1].
        """
        if total_samples <= 1:
            return (0.0, 0.0)

        # Get stratified disk sample
        x, y = sample_disk_stratified(sample_index, total_samples)

        # Apply bokeh shape
        if bokeh_shape == "hexagon":
            # Convert to hexagonal shape
            angle = math.atan2(y, x)
            sector = (angle + math.pi) / (math.pi / 3.0)
            sector_angle = (sector % 1.0) * (math.pi / 3.0) - (math.pi / 6.0)
            scale = math.cos(math.pi / 6.0) / max(math.cos(sector_angle), 0.01)
            scale = min(scale, 1.0)
            return (x * scale, y * scale)

        return (x, y)

    def generate_samples(
        self,
        ray: Ray,
        params: DOFParams,
        camera_right: Vec3,
        camera_up: Vec3,
    ) -> Generator[Ray, None, None]:
        """
        Generate all DOF sample rays for a pixel.

        Args:
            ray: Primary ray.
            params: DOF parameters.
            camera_right: Camera right vector.
            camera_up: Camera up vector.

        Yields:
            Jittered rays for DOF accumulation.
        """
        if not params.is_enabled():
            yield ray
            return

        for i in range(params.samples_per_pixel):
            offset = self.get_sample_offset(
                i,
                params.samples_per_pixel,
                params.bokeh_shape,
            )
            yield self.jitter_ray(ray, params, offset, camera_right, camera_up)


# =============================================================================
# Accumulation Buffer Helpers
# =============================================================================


@dataclass
class AccumulationBuffer:
    """
    Simple accumulation buffer for progressive DOF rendering.

    Accumulates color samples over multiple frames for smooth DOF effect.

    Attributes:
        width: Buffer width in pixels.
        height: Buffer height in pixels.
        sample_count: Number of accumulated samples per pixel.
    """
    width: int
    height: int
    sample_count: int = 0
    _buffer: Optional[list] = None

    def __post_init__(self):
        if self._buffer is None:
            self._buffer = [Vec3(0.0, 0.0, 0.0) for _ in range(self.width * self.height)]

    def clear(self) -> None:
        """Clear the buffer and reset sample count."""
        self._buffer = [Vec3(0.0, 0.0, 0.0) for _ in range(self.width * self.height)]
        self.sample_count = 0

    def accumulate(self, x: int, y: int, color: Vec3) -> None:
        """
        Add a color sample to the buffer.

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.
            color: Color sample to accumulate.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = y * self.width + x
            self._buffer[idx] = self._buffer[idx] + color

    def get_color(self, x: int, y: int) -> Vec3:
        """
        Get the averaged color at a pixel.

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.

        Returns:
            Averaged color.
        """
        if self.sample_count <= 0:
            return Vec3(0.0, 0.0, 0.0)

        if 0 <= x < self.width and 0 <= y < self.height:
            idx = y * self.width + x
            inv_count = 1.0 / self.sample_count
            acc = self._buffer[idx]
            return Vec3(acc.x * inv_count, acc.y * inv_count, acc.z * inv_count)

        return Vec3(0.0, 0.0, 0.0)

    def increment_sample_count(self) -> None:
        """Increment the sample count after a frame."""
        self.sample_count += 1


# =============================================================================
# WGSL Code Generation
# =============================================================================


def generate_dof_wgsl(include_accumulation: bool = False) -> str:
    """
    Generate WGSL code for depth of field ray generation.

    Args:
        include_accumulation: Include accumulation buffer helpers.

    Returns:
        WGSL code string with DOF functions.
    """
    lines: list[str] = []

    lines.append("// =============================================================================")
    lines.append("// Depth of Field (T-DEMO-3.12)")
    lines.append("// =============================================================================")
    lines.append("")

    lines.append(WGSL_DOF_PARAMS)
    lines.append(WGSL_SAMPLE_DISK)
    lines.append(WGSL_JITTER_RAY)
    lines.append(WGSL_CALCULATE_COC)

    if include_accumulation:
        lines.append(WGSL_ACCUMULATION)

    return "\n".join(lines)


WGSL_DOF_PARAMS = """\
/// Depth of field parameters.
struct DOFParams {
    aperture: f32,        // Lens aperture radius
    focal_distance: f32,  // Focus distance
    focus_range: f32,     // Acceptable focus depth range
    samples: u32,         // Samples per pixel
}
"""

WGSL_SAMPLE_DISK = """\
/// Sample a point on a unit disk using golden spiral distribution.
/// sample_index: Current sample [0, total_samples)
/// total_samples: Total number of samples
fn sample_disk(sample_index: u32, total_samples: u32) -> vec2<f32> {
    if (total_samples <= 1u) {
        return vec2<f32>(0.0, 0.0);
    }

    let golden_angle = 2.399963229728653;  // pi * (3 - sqrt(5))
    let idx = f32(sample_index) + 0.5;
    let total = f32(total_samples);

    // Radius increases with sqrt for uniform density
    let r = sqrt(idx / total);
    let theta = idx * golden_angle;

    return vec2<f32>(r * cos(theta), r * sin(theta));
}

/// Sample disk with random jitter for temporal accumulation.
fn sample_disk_jittered(sample_index: u32, total_samples: u32, jitter: f32) -> vec2<f32> {
    if (total_samples <= 1u) {
        return vec2<f32>(0.0, 0.0);
    }

    let golden_angle = 2.399963229728653;
    let idx = f32(sample_index) + 0.5 + jitter;
    let total = f32(total_samples);

    let r = sqrt(idx / total);
    let theta = idx * golden_angle;

    return vec2<f32>(r * cos(theta), r * sin(theta));
}
"""

WGSL_JITTER_RAY = """\
/// Generate a DOF-jittered ray using thin lens model.
///
/// ray_origin: Original camera position
/// ray_direction: Original ray direction (normalized)
/// focal_distance: Distance to focal plane
/// aperture: Lens aperture radius
/// lens_offset: Sample point on lens disk (x, y) in [-1, 1]
/// camera_right: Camera right vector (normalized)
/// camera_up: Camera up vector (normalized)
///
/// Returns: struct { origin: vec3<f32>, direction: vec3<f32> }
fn jitter_ray_dof(
    ray_origin: vec3<f32>,
    ray_direction: vec3<f32>,
    focal_distance: f32,
    aperture: f32,
    lens_offset: vec2<f32>,
    camera_right: vec3<f32>,
    camera_up: vec3<f32>,
) -> Ray {
    // If aperture is zero, return original ray (pinhole)
    if (aperture <= 0.0) {
        return Ray(ray_origin, ray_direction);
    }

    // Calculate focal point
    let focal_point = ray_origin + ray_direction * focal_distance;

    // Offset ray origin on lens disk
    let lens_pos = camera_right * (lens_offset.x * aperture)
                 + camera_up * (lens_offset.y * aperture);
    let new_origin = ray_origin + lens_pos;

    // New direction towards focal point
    let new_direction = normalize(focal_point - new_origin);

    return Ray(new_origin, new_direction);
}
"""

WGSL_CALCULATE_COC = """\
/// Calculate circle of confusion for a hit point.
///
/// hit_distance: Distance from camera to hit point
/// focal_distance: Camera's focal distance
/// aperture: Lens aperture radius
///
/// Returns: Circle of confusion radius (blur amount)
fn calculate_coc(hit_distance: f32, focal_distance: f32, aperture: f32) -> f32 {
    if (aperture <= 0.0 || hit_distance <= 0.0) {
        return 0.0;
    }

    // Thin lens CoC formula
    return abs(aperture * (hit_distance - focal_distance) / hit_distance);
}

/// Calculate normalized CoC in [0, 1] range.
fn calculate_coc_normalized(hit_distance: f32, focal_distance: f32, aperture: f32) -> f32 {
    let coc = calculate_coc(hit_distance, focal_distance, aperture);
    let max_coc = aperture;
    return clamp(coc / max(max_coc, 0.001), 0.0, 1.0);
}

/// Check if a point is within acceptable focus range.
fn is_in_focus(hit_distance: f32, focal_distance: f32, focus_range: f32) -> bool {
    let half_range = focus_range * 0.5;
    return abs(hit_distance - focal_distance) <= half_range;
}
"""

WGSL_ACCUMULATION = """\
/// Accumulation buffer for progressive DOF rendering.
/// This would typically be a storage buffer binding.
///
/// Usage:
///   - Each frame, add one sample per pixel
///   - Divide accumulated color by sample_count for final result

/// Add a sample to the accumulation buffer.
fn accumulate_sample(
    buffer: ptr<storage, array<vec4<f32>>, read_write>,
    x: u32,
    y: u32,
    width: u32,
    color: vec3<f32>,
) {
    let idx = y * width + x;
    let prev = (*buffer)[idx];
    (*buffer)[idx] = vec4<f32>(prev.xyz + color, prev.w + 1.0);
}

/// Get the averaged color from accumulation buffer.
fn get_accumulated_color(
    buffer: ptr<storage, array<vec4<f32>>, read>,
    x: u32,
    y: u32,
    width: u32,
) -> vec3<f32> {
    let idx = y * width + x;
    let data = (*buffer)[idx];
    let count = max(data.w, 1.0);
    return data.xyz / count;
}
"""


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_dof_params(params: DOFParams) -> list[str]:
    """
    Validate DOF parameters.

    Args:
        params: DOF parameters to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    if params.aperture < 0.0:
        errors.append(f"Aperture must be non-negative, got {params.aperture}")

    if params.focal_distance <= 0.0:
        errors.append(f"Focal distance must be positive, got {params.focal_distance}")

    if params.focus_range < 0.0:
        errors.append(f"Focus range must be non-negative, got {params.focus_range}")

    if params.samples_per_pixel < 1:
        errors.append(f"Samples per pixel must be at least 1, got {params.samples_per_pixel}")

    if params.bokeh_shape not in ("circle", "hexagon", "octagon"):
        errors.append(f"Unknown bokeh shape: {params.bokeh_shape}")

    return errors


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Parameters
    "DOFParams",
    # Circle of confusion
    "calculate_coc",
    "calculate_coc_normalized",
    "is_in_focus",
    # Lens sampling
    "sample_disk_uniform",
    "sample_disk_stratified",
    "sample_hexagon",
    # DOF Generator
    "DOFGenerator",
    # Accumulation
    "AccumulationBuffer",
    # WGSL generation
    "generate_dof_wgsl",
    # Validation
    "validate_dof_params",
]

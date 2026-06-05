"""
SDF Ambient Occlusion (T-DEMO-3.5) - Quilez's Method.

This module implements ambient occlusion calculation for signed distance field
scenes using Inigo Quilez's ray marching technique. AO is computed by sampling
the SDF along the normal direction and comparing expected vs actual distances.

The technique provides:
  - Fast AO approximation (5-8 samples along normal)
  - No precomputation required
  - Perceptually correct darkening of crevices and corners
  - Configurable falloff and step parameters

Reference: Inigo Quilez -- AO in ray marching
    https://iquilezles.org/articles/rmshadows/

Usage:
    >>> from engine.rendering.demoscene.sdf_ao import (
    ...     calculate_ao, generate_ao_wgsl, AOConfig
    ... )
    >>> ao = calculate_ao(point, normal, scene_sdf, config=AOConfig())
    >>> wgsl = generate_ao_wgsl()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .sdf_ast import SDFNode, Vec3


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True, slots=True)
class AOConfig:
    """Configuration for ambient occlusion calculation.

    Attributes:
        samples: Number of AO samples along normal (default 5).
        step_scale: Base distance scale for sampling (default 0.1).
        falloff: Exponential falloff per sample (default 0.5).
                 Controls how quickly distant samples contribute less.
        max_distance: Maximum sampling distance (default 0.5).
        intensity: Final AO intensity multiplier (default 1.0).
    """
    samples: int = 5
    step_scale: float = 0.15
    falloff: float = 0.6
    max_distance: float = 2.0
    intensity: float = 1.2

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.samples < 1:
            raise ValueError(f"samples must be >= 1, got {self.samples}")
        if self.step_scale <= 0.0:
            raise ValueError(f"step_scale must be > 0, got {self.step_scale}")
        if not (0.0 < self.falloff <= 1.0):
            raise ValueError(f"falloff must be in (0, 1], got {self.falloff}")
        if self.max_distance <= 0.0:
            raise ValueError(f"max_distance must be > 0, got {self.max_distance}")


# =============================================================================
# Vec3 Helper (for standalone use)
# =============================================================================

@dataclass(frozen=True, slots=True)
class Vec3Local:
    """Local Vec3 for standalone calculations without importing sdf_ast."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3Local") -> "Vec3Local":
        return Vec3Local(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scalar: float) -> "Vec3Local":
        return Vec3Local(self.x * scalar, self.y * scalar, self.z * scalar)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Vec3Local":
        ln = self.length()
        if ln < 1e-10:
            return Vec3Local(0.0, 0.0, 0.0)
        return Vec3Local(self.x / ln, self.y / ln, self.z / ln)

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


# =============================================================================
# Core AO Calculation (Python Reference Implementation)
# =============================================================================

def calculate_ao(
    p: tuple[float, float, float],
    n: tuple[float, float, float],
    scene_sdf: Callable[[tuple[float, float, float]], float],
    config: Optional[AOConfig] = None,
) -> float:
    """
    Calculate ambient occlusion at a surface point using Quilez's method.

    The algorithm samples the SDF at increasing distances along the normal
    direction. If the actual distance is less than expected, the point is
    occluded. The difference is accumulated with exponential falloff.

    Algorithm (Quilez AO):
        ao = 0.0
        for i in 1..samples:
            d = step_scale * i
            sample_pos = p + n * d
            expected_dist = d
            actual_dist = scene_sdf(sample_pos)
            ao += (expected_dist - actual_dist) / expected_dist * falloff^i
        return clamp(1.0 - ao * intensity, 0.0, 1.0)

    Args:
        p: Surface point position (x, y, z).
        n: Surface normal at point (normalized).
        scene_sdf: Scene SDF function mapping position to signed distance.
        config: AO configuration parameters (uses defaults if None).

    Returns:
        Occlusion factor in [0.0, 1.0] where:
          - 0.0 = fully occluded (crevice/corner)
          - 1.0 = no occlusion (open sky)

    Example:
        >>> def sphere_sdf(p):
        ...     return math.sqrt(sum(x*x for x in p)) - 1.0
        >>> ao = calculate_ao((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), sphere_sdf)
        >>> assert ao > 0.95  # Point on sphere surface, little occlusion
    """
    if config is None:
        config = AOConfig()

    # Normalize input normal
    n_vec = Vec3Local(n[0], n[1], n[2]).normalized()
    p_vec = Vec3Local(p[0], p[1], p[2])

    ao = 0.0

    for i in range(1, config.samples + 1):
        # Distance along normal for this sample
        d = config.step_scale * i

        # Clamp to max distance
        if d > config.max_distance:
            d = config.max_distance

        # Sample position
        sample_pos = p_vec + n_vec * d

        # Expected distance is how far we moved
        expected_dist = d

        # Actual distance from SDF
        actual_dist = scene_sdf(sample_pos.as_tuple())

        # Occlusion contribution: how much closer is the geometry than expected?
        # If actual_dist < expected_dist, there's occlusion
        occlusion = (expected_dist - actual_dist) / expected_dist

        # Weight by exponential falloff
        weight = config.falloff ** i

        ao += max(0.0, occlusion) * weight

    # Final AO factor: 1.0 - accumulated occlusion
    result = 1.0 - ao * config.intensity

    # Clamp to valid range
    return max(0.0, min(1.0, result))


def calculate_ao_multi_direction(
    p: tuple[float, float, float],
    n: tuple[float, float, float],
    scene_sdf: Callable[[tuple[float, float, float]], float],
    config: Optional[AOConfig] = None,
    cone_angle: float = 0.5,
    direction_samples: int = 4,
) -> float:
    """
    Calculate ambient occlusion using multiple sample directions.

    Extends the basic AO by sampling in a cone around the normal,
    providing more accurate results for complex geometry.

    Args:
        p: Surface point position.
        n: Surface normal at point.
        scene_sdf: Scene SDF function.
        config: AO configuration parameters.
        cone_angle: Half-angle of sampling cone in radians (default 0.5).
        direction_samples: Number of directions to sample (default 4).

    Returns:
        Average occlusion factor across all directions.
    """
    if config is None:
        config = AOConfig()

    n_vec = Vec3Local(n[0], n[1], n[2]).normalized()

    # Create orthonormal basis
    up = Vec3Local(0.0, 1.0, 0.0)
    if abs(n_vec.y) > 0.99:
        up = Vec3Local(1.0, 0.0, 0.0)

    # Tangent = up x normal
    tangent = Vec3Local(
        up.y * n_vec.z - up.z * n_vec.y,
        up.z * n_vec.x - up.x * n_vec.z,
        up.x * n_vec.y - up.y * n_vec.x,
    ).normalized()

    # Bitangent = normal x tangent
    bitangent = Vec3Local(
        n_vec.y * tangent.z - n_vec.z * tangent.y,
        n_vec.z * tangent.x - n_vec.x * tangent.z,
        n_vec.x * tangent.y - n_vec.y * tangent.x,
    ).normalized()

    total_ao = 0.0

    # Sample in cone around normal
    for i in range(direction_samples):
        angle = 2.0 * math.pi * i / direction_samples

        # Direction in tangent space
        sin_cone = math.sin(cone_angle)
        cos_cone = math.cos(cone_angle)

        dx = sin_cone * math.cos(angle)
        dy = sin_cone * math.sin(angle)
        dz = cos_cone

        # Transform to world space
        dir_world = Vec3Local(
            dx * tangent.x + dy * bitangent.x + dz * n_vec.x,
            dx * tangent.y + dy * bitangent.y + dz * n_vec.y,
            dx * tangent.z + dy * bitangent.z + dz * n_vec.z,
        ).normalized()

        # Calculate AO for this direction
        total_ao += calculate_ao(p, dir_world.as_tuple(), scene_sdf, config)

    # Also include straight normal direction
    total_ao += calculate_ao(p, n, scene_sdf, config)

    return total_ao / (direction_samples + 1)


# =============================================================================
# WGSL Code Generation
# =============================================================================

# Default WGSL function for AO calculation
AO_WGSL_FUNCTION = """\
/// Calculates ambient occlusion using Quilez's method.
///   p          -- surface point
///   n          -- surface normal (normalized)
///   returns    -- occlusion factor [0=occluded, 1=open]
///
/// Reference: Inigo Quilez -- AO in ray marching
fn calculate_ao(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let AO_SAMPLES: i32 = 5;
    let AO_STEP_SCALE: f32 = 0.1;
    let AO_FALLOFF: f32 = 0.5;
    let AO_INTENSITY: f32 = 1.0;

    var ao: f32 = 0.0;
    var weight: f32 = 1.0;

    for (var i: i32 = 1; i <= AO_SAMPLES; i = i + 1) {
        let d: f32 = AO_STEP_SCALE * f32(i);
        let sample_pos: vec3<f32> = p + n * d;
        let expected_dist: f32 = d;
        let actual_dist: f32 = scene_sdf(sample_pos).x;

        // Occlusion contribution
        let occlusion: f32 = (expected_dist - actual_dist) / expected_dist;

        // Weight by falloff
        weight = weight * AO_FALLOFF;
        ao = ao + max(0.0, occlusion) * weight / AO_FALLOFF;
    }

    return clamp(1.0 - ao * AO_INTENSITY, 0.0, 1.0);
}"""


def generate_ao_wgsl(config: Optional[AOConfig] = None) -> str:
    """
    Generate WGSL code for ambient occlusion calculation.

    Args:
        config: AO configuration to embed in shader. Uses defaults if None.

    Returns:
        WGSL function definition string.

    Example:
        >>> wgsl = generate_ao_wgsl(AOConfig(samples=8, falloff=0.6))
        >>> assert "AO_SAMPLES: i32 = 8" in wgsl
    """
    if config is None:
        config = AOConfig()

    return f"""\
/// Calculates ambient occlusion using Quilez's method.
///   p          -- surface point
///   n          -- surface normal (normalized)
///   returns    -- occlusion factor [0=occluded, 1=open]
///
/// Reference: Inigo Quilez -- AO in ray marching
fn calculate_ao(p: vec3<f32>, n: vec3<f32>) -> f32 {{
    let AO_SAMPLES: i32 = {config.samples};
    let AO_STEP_SCALE: f32 = {config.step_scale};
    let AO_FALLOFF: f32 = {config.falloff};
    let AO_INTENSITY: f32 = {config.intensity};
    let AO_MAX_DISTANCE: f32 = {config.max_distance};

    var ao: f32 = 0.0;
    var weight: f32 = 1.0;

    for (var i: i32 = 1; i <= AO_SAMPLES; i = i + 1) {{
        var d: f32 = AO_STEP_SCALE * f32(i);
        d = min(d, AO_MAX_DISTANCE);

        let sample_pos: vec3<f32> = p + n * d;
        let expected_dist: f32 = d;
        let actual_dist: f32 = scene_sdf(sample_pos).x;

        // Occlusion contribution
        let occlusion: f32 = (expected_dist - actual_dist) / expected_dist;

        // Weight by falloff
        weight = weight * AO_FALLOFF;
        ao = ao + max(0.0, occlusion) * weight / AO_FALLOFF;
    }}

    return clamp(1.0 - ao * AO_INTENSITY, 0.0, 1.0);
}}"""


def generate_ao_wgsl_inline(config: Optional[AOConfig] = None) -> str:
    """
    Generate inline WGSL code for AO (for embedding in larger shaders).

    This version generates just the calculation logic without the
    function wrapper, suitable for inlining in compute shaders.

    Args:
        config: AO configuration parameters.

    Returns:
        WGSL code block (expects 'p' and 'n' variables in scope).
    """
    if config is None:
        config = AOConfig()

    return f"""\
// AO calculation (Quilez method)
var ao_accum: f32 = 0.0;
var ao_weight: f32 = 1.0;
for (var ao_i: i32 = 1; ao_i <= {config.samples}; ao_i = ao_i + 1) {{
    var ao_d: f32 = {config.step_scale} * f32(ao_i);
    ao_d = min(ao_d, {config.max_distance});
    let ao_sample_pos: vec3<f32> = p + n * ao_d;
    let ao_expected: f32 = ao_d;
    let ao_actual: f32 = scene_sdf(ao_sample_pos).x;
    let ao_occ: f32 = (ao_expected - ao_actual) / ao_expected;
    ao_weight = ao_weight * {config.falloff};
    ao_accum = ao_accum + max(0.0, ao_occ) * ao_weight / {config.falloff};
}}
let ao: f32 = clamp(1.0 - ao_accum * {config.intensity}, 0.0, 1.0);"""


# =============================================================================
# Scene Integration Helpers
# =============================================================================

def make_scene_ao_evaluator(
    scene_sdf: Callable[[tuple[float, float, float]], float],
    config: Optional[AOConfig] = None,
) -> Callable[[tuple[float, float, float], tuple[float, float, float]], float]:
    """
    Create an AO evaluator function bound to a specific scene SDF.

    Args:
        scene_sdf: Scene SDF function.
        config: AO configuration.

    Returns:
        Function taking (point, normal) and returning AO factor.

    Example:
        >>> def sphere_sdf(p):
        ...     return math.sqrt(sum(x*x for x in p)) - 1.0
        >>> ao_eval = make_scene_ao_evaluator(sphere_sdf)
        >>> ao = ao_eval((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    """
    if config is None:
        config = AOConfig()

    def evaluator(
        p: tuple[float, float, float],
        n: tuple[float, float, float],
    ) -> float:
        return calculate_ao(p, n, scene_sdf, config)

    return evaluator


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Configuration
    "AOConfig",
    # Core functions
    "calculate_ao",
    "calculate_ao_multi_direction",
    # WGSL generation
    "AO_WGSL_FUNCTION",
    "generate_ao_wgsl",
    "generate_ao_wgsl_inline",
    # Helpers
    "make_scene_ao_evaluator",
    "Vec3Local",
]

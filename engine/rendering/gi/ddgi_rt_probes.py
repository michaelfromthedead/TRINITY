"""DDGI Probe Ray Tracing for Hardware RT Path (T-GIR-P2.2).

This module implements hardware ray-traced probe updates for DDGI:
- DDGIRTProbeUpdater: Main class for RT-based probe updates
- ProbeRayGenerator: Stratified spherical ray direction generation
- RadianceAccumulator: Per-probe radiance collection and filtering

The implementation uses:
- TLAS (Top-Level Acceleration Structure) for ray traversal
- Stratified spherical sampling (Fibonacci spiral) for ray directions
- Configurable ray counts (32-128 per probe)
- Temporal hysteresis for stable accumulation

References:
    - Section 6.4 DDGI in RENDERING_CONTEXT.md
    - Section 6.11 Ray Tracing Architecture
    - T-GIR-P2.1 (DDGI probe placement) is a dependency
"""

from __future__ import annotations

import math
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional, Protocol

from engine.core.math.vec import Vec2, Vec3

if TYPE_CHECKING:
    from engine.rendering.framegraph.pass_node import RayTracingPass
    from engine.rendering.framegraph.resource_manager import ResourceHandle


# ============================================================================
# Constants
# ============================================================================

# Minimum/maximum rays per probe
MIN_RAYS_PER_PROBE = 32
MAX_RAYS_PER_PROBE = 128
DEFAULT_RAYS_PER_PROBE = 64

# Golden ratio for Fibonacci spiral
GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0
GOLDEN_ANGLE = 2.0 * math.pi / GOLDEN_RATIO

# Ray flags for hardware RT
RAY_FLAG_NONE = 0
RAY_FLAG_CULL_BACK_FACING = 1 << 0
RAY_FLAG_CULL_FRONT_FACING = 1 << 1
RAY_FLAG_TERMINATE_ON_FIRST_HIT = 1 << 2
RAY_FLAG_SKIP_CLOSEST_HIT = 1 << 3

# Default temporal hysteresis
DEFAULT_HYSTERESIS = 0.97

# Maximum ray distance
DEFAULT_MAX_RAY_DISTANCE = 100.0


# ============================================================================
# Ray Distribution Strategies
# ============================================================================


class RayDistribution(Enum):
    """Ray distribution strategy for probe sampling."""

    FIBONACCI_SPIRAL = auto()
    """Fibonacci spiral (quasi-uniform, deterministic)."""

    STRATIFIED_JITTERED = auto()
    """Stratified with per-stratum jitter."""

    HALTON_SEQUENCE = auto()
    """Halton low-discrepancy sequence."""

    UNIFORM_RANDOM = auto()
    """Pure random uniform distribution."""


# ============================================================================
# Ray Generation Configuration
# ============================================================================


@dataclass
class ProbeRayConfig:
    """Configuration for probe ray generation.

    Attributes:
        rays_per_probe: Number of rays per probe (32-128)
        distribution: Ray distribution strategy
        max_ray_distance: Maximum ray trace distance (meters)
        normal_bias: Bias along hit normal for shadow rays
        temporal_rotation: Enable per-frame random rotation
        temporal_seed_offset: Frame-based seed offset for jitter
    """

    rays_per_probe: int = DEFAULT_RAYS_PER_PROBE
    distribution: RayDistribution = RayDistribution.FIBONACCI_SPIRAL
    max_ray_distance: float = DEFAULT_MAX_RAY_DISTANCE
    normal_bias: float = 0.01
    temporal_rotation: bool = True
    temporal_seed_offset: int = 0

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.rays_per_probe = max(
            MIN_RAYS_PER_PROBE, min(MAX_RAYS_PER_PROBE, self.rays_per_probe)
        )

    def validate(self) -> list[str]:
        """Validate configuration, returning list of errors."""
        errors = []

        if self.rays_per_probe < MIN_RAYS_PER_PROBE:
            errors.append(
                f"rays_per_probe must be at least {MIN_RAYS_PER_PROBE}, "
                f"got {self.rays_per_probe}"
            )

        if self.rays_per_probe > MAX_RAYS_PER_PROBE:
            errors.append(
                f"rays_per_probe must be at most {MAX_RAYS_PER_PROBE}, "
                f"got {self.rays_per_probe}"
            )

        if self.max_ray_distance <= 0:
            errors.append("max_ray_distance must be positive")

        return errors

    @staticmethod
    def low_quality() -> ProbeRayConfig:
        """Low quality preset (32 rays)."""
        return ProbeRayConfig(rays_per_probe=32)

    @staticmethod
    def medium_quality() -> ProbeRayConfig:
        """Medium quality preset (64 rays)."""
        return ProbeRayConfig(rays_per_probe=64)

    @staticmethod
    def high_quality() -> ProbeRayConfig:
        """High quality preset (128 rays)."""
        return ProbeRayConfig(rays_per_probe=128)


# ============================================================================
# Ray Direction Generator
# ============================================================================


class ProbeRayGenerator:
    """Generates stratified spherical ray directions for probes.

    This class produces quasi-uniform ray directions using various
    distribution strategies. The default Fibonacci spiral provides
    excellent coverage with deterministic results.

    Usage:
        generator = ProbeRayGenerator(config)
        directions = generator.generate_directions(frame_index)
    """

    def __init__(self, config: ProbeRayConfig) -> None:
        """Initialize the ray generator.

        Args:
            config: Ray generation configuration
        """
        self.config = config
        self._cached_base_directions: Optional[list[Vec3]] = None
        self._cached_ray_count: int = 0

    def generate_directions(self, frame_index: int = 0) -> list[Vec3]:
        """Generate ray directions for a probe.

        Args:
            frame_index: Current frame index for temporal jitter

        Returns:
            List of normalized direction vectors
        """
        # Use cached base directions if available
        if (
            self._cached_base_directions is None
            or self._cached_ray_count != self.config.rays_per_probe
        ):
            self._cached_base_directions = self._generate_base_directions()
            self._cached_ray_count = self.config.rays_per_probe

        # Apply temporal rotation if enabled
        if self.config.temporal_rotation:
            rotation_angle = (
                frame_index + self.config.temporal_seed_offset
            ) * GOLDEN_ANGLE
            return self._apply_rotation(
                self._cached_base_directions, rotation_angle
            )

        return list(self._cached_base_directions)

    def _generate_base_directions(self) -> list[Vec3]:
        """Generate base ray directions without temporal jitter."""
        distribution = self.config.distribution
        n = self.config.rays_per_probe

        if distribution == RayDistribution.FIBONACCI_SPIRAL:
            return self._fibonacci_spiral(n)
        elif distribution == RayDistribution.STRATIFIED_JITTERED:
            return self._stratified_jittered(n)
        elif distribution == RayDistribution.HALTON_SEQUENCE:
            return self._halton_sequence(n)
        else:  # UNIFORM_RANDOM
            return self._uniform_random(n)

    def _fibonacci_spiral(self, n: int) -> list[Vec3]:
        """Generate Fibonacci spiral directions.

        Uses the spherical Fibonacci lattice for quasi-uniform sampling.

        Args:
            n: Number of directions

        Returns:
            List of normalized direction vectors
        """
        directions = []

        for i in range(n):
            # Fibonacci spiral on sphere
            theta = GOLDEN_ANGLE * i
            phi = math.acos(1.0 - 2.0 * (i + 0.5) / n)

            x = math.sin(phi) * math.cos(theta)
            y = math.sin(phi) * math.sin(theta)
            z = math.cos(phi)

            directions.append(Vec3(x, y, z))

        return directions

    def _stratified_jittered(self, n: int) -> list[Vec3]:
        """Generate stratified jittered directions.

        Divides the sphere into strata and samples each with jitter.

        Args:
            n: Number of directions

        Returns:
            List of normalized direction vectors
        """
        directions = []

        # Approximate equal-area stratification
        # Use ceiling division to ensure we generate at least n samples
        n_theta = max(1, int(math.ceil(math.sqrt(n * 2))))
        n_phi = max(1, int(math.ceil(n / n_theta)))

        idx = 0
        for i in range(n_theta):
            for j in range(n_phi):
                if len(directions) >= n:
                    break

                # Base angles with small jitter
                theta_base = 2.0 * math.pi * (i + 0.5) / n_theta
                phi_base = math.acos(1.0 - 2.0 * (j + 0.5) / n_phi)

                # Add deterministic jitter based on index
                seed = idx
                jitter_theta = self._pseudo_random(seed) * 0.5 / n_theta
                jitter_phi = self._pseudo_random(seed + 1000) * 0.5 / n_phi

                theta = theta_base + jitter_theta * 2.0 * math.pi
                phi = phi_base + jitter_phi * math.pi

                phi = max(0.0, min(math.pi, phi))

                x = math.sin(phi) * math.cos(theta)
                y = math.sin(phi) * math.sin(theta)
                z = math.cos(phi)

                directions.append(Vec3(x, y, z))
                idx += 1

            if len(directions) >= n:
                break

        return directions[:n]

    def _halton_sequence(self, n: int) -> list[Vec3]:
        """Generate Halton sequence directions.

        Uses bases 2 and 3 for low-discrepancy sampling.

        Args:
            n: Number of directions

        Returns:
            List of normalized direction vectors
        """
        directions = []

        for i in range(n):
            # Halton sequence in bases 2 and 3
            u = self._halton(i + 1, 2)
            v = self._halton(i + 1, 3)

            # Map to sphere
            theta = 2.0 * math.pi * u
            phi = math.acos(1.0 - 2.0 * v)

            x = math.sin(phi) * math.cos(theta)
            y = math.sin(phi) * math.sin(theta)
            z = math.cos(phi)

            directions.append(Vec3(x, y, z))

        return directions

    def _uniform_random(self, n: int) -> list[Vec3]:
        """Generate uniform random directions.

        Uses pseudo-random for reproducibility.

        Args:
            n: Number of directions

        Returns:
            List of normalized direction vectors
        """
        directions = []

        for i in range(n):
            # Pseudo-random values
            u = self._pseudo_random(i * 2)
            v = self._pseudo_random(i * 2 + 1)

            # Map to sphere
            theta = 2.0 * math.pi * u
            phi = math.acos(1.0 - 2.0 * v)

            x = math.sin(phi) * math.cos(theta)
            y = math.sin(phi) * math.sin(theta)
            z = math.cos(phi)

            directions.append(Vec3(x, y, z))

        return directions

    def _halton(self, index: int, base: int) -> float:
        """Compute Halton sequence value.

        Args:
            index: Sequence index (1-based)
            base: Sequence base (prime)

        Returns:
            Halton value in [0, 1)
        """
        result = 0.0
        f = 1.0 / base
        i = index

        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base

        return result

    def _pseudo_random(self, seed: int) -> float:
        """Pseudo-random number generator for deterministic results.

        Args:
            seed: Random seed

        Returns:
            Pseudo-random value in [0, 1)
        """
        # Simple LCG for reproducibility
        x = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        return (x % 10000) / 10000.0

    def _apply_rotation(
        self, directions: list[Vec3], angle: float
    ) -> list[Vec3]:
        """Apply Y-axis rotation to directions.

        Args:
            directions: Original directions
            angle: Rotation angle in radians

        Returns:
            Rotated direction vectors
        """
        c = math.cos(angle)
        s = math.sin(angle)

        rotated = []
        for d in directions:
            # Rotate around Y axis
            rx = d.x * c + d.z * s
            ry = d.y
            rz = -d.x * s + d.z * c

            rotated.append(Vec3(rx, ry, rz))

        return rotated

    def get_direction_bytes(self, frame_index: int = 0) -> bytes:
        """Get ray directions as GPU-uploadable bytes.

        Format: N * 16 bytes (vec4<f32> per direction, w=0)

        Args:
            frame_index: Current frame index

        Returns:
            Packed direction data
        """
        directions = self.generate_directions(frame_index)
        data = []

        for d in directions:
            data.append(struct.pack("<4f", d.x, d.y, d.z, 0.0))

        return b"".join(data)


# ============================================================================
# Ray Hit Result
# ============================================================================


@dataclass
class RayHitResult:
    """Result of a traced ray.

    Attributes:
        hit: Whether the ray hit geometry
        hit_distance: Distance to hit point (or max_distance if miss)
        hit_position: World position of hit
        hit_normal: Surface normal at hit point
        radiance: Radiance contribution at hit
        material_id: Hit surface material ID
    """

    hit: bool = False
    hit_distance: float = 0.0
    hit_position: Vec3 = field(default_factory=Vec3.zero)
    hit_normal: Vec3 = field(default_factory=lambda: Vec3(0, 1, 0))
    radiance: Vec3 = field(default_factory=Vec3.zero)
    material_id: int = 0


# ============================================================================
# Radiance Accumulator
# ============================================================================


@dataclass
class AccumulatedRadiance:
    """Accumulated radiance data for a probe direction.

    Attributes:
        radiance_sum: Sum of radiance samples
        depth_sum: Sum of hit depths
        depth_squared_sum: Sum of squared depths (for variance)
        sample_count: Number of accumulated samples
        weight_sum: Sum of sample weights
    """

    radiance_sum: Vec3 = field(default_factory=Vec3.zero)
    depth_sum: float = 0.0
    depth_squared_sum: float = 0.0
    sample_count: int = 0
    weight_sum: float = 0.0

    def add_sample(
        self, radiance: Vec3, depth: float, weight: float = 1.0
    ) -> None:
        """Add a radiance sample.

        Args:
            radiance: Sample radiance
            depth: Sample depth
            weight: Sample weight
        """
        self.radiance_sum = self.radiance_sum + radiance * weight
        self.depth_sum += depth * weight
        self.depth_squared_sum += depth * depth * weight
        self.sample_count += 1
        self.weight_sum += weight

    def get_average_radiance(self) -> Vec3:
        """Get weighted average radiance."""
        if self.weight_sum <= 0:
            return Vec3.zero()
        return self.radiance_sum * (1.0 / self.weight_sum)

    def get_average_depth(self) -> float:
        """Get weighted average depth."""
        if self.weight_sum <= 0:
            return 0.0
        return self.depth_sum / self.weight_sum

    def get_depth_variance(self) -> float:
        """Get depth variance."""
        if self.weight_sum <= 0:
            return 0.0
        mean = self.get_average_depth()
        mean_sq = self.depth_squared_sum / self.weight_sum
        return max(0.0, mean_sq - mean * mean)


class RadianceAccumulator:
    """Accumulates radiance per probe from ray hits.

    Handles:
    - Per-direction radiance accumulation
    - Octahedral encoding for irradiance/visibility
    - Temporal hysteresis blending
    - GPU data serialization

    Usage:
        accum = RadianceAccumulator(config)
        for hit in ray_hits:
            accum.accumulate(ray_direction, hit)
        irradiance = accum.get_filtered_irradiance()
    """

    def __init__(
        self,
        ray_config: ProbeRayConfig,
        hysteresis: float = DEFAULT_HYSTERESIS,
        irradiance_resolution: int = 8,
        visibility_resolution: int = 16,
    ) -> None:
        """Initialize the accumulator.

        Args:
            ray_config: Ray configuration
            hysteresis: Temporal blending factor (0-1)
            irradiance_resolution: Irradiance octahedral map resolution
            visibility_resolution: Visibility octahedral map resolution
        """
        self.ray_config = ray_config
        self.hysteresis = hysteresis
        self.irradiance_resolution = irradiance_resolution
        self.visibility_resolution = visibility_resolution

        # Per-texel accumulators
        ir_count = irradiance_resolution * irradiance_resolution
        vis_count = visibility_resolution * visibility_resolution

        self._irradiance_accum: list[AccumulatedRadiance] = [
            AccumulatedRadiance() for _ in range(ir_count)
        ]
        self._visibility_accum: list[AccumulatedRadiance] = [
            AccumulatedRadiance() for _ in range(vis_count)
        ]

        # Current filtered values
        self._irradiance_filtered: list[Vec3] = [
            Vec3.zero() for _ in range(ir_count)
        ]
        self._visibility_filtered: list[Vec2] = [
            Vec2(ray_config.max_ray_distance, 0.0) for _ in range(vis_count)
        ]

    def reset_accumulators(self) -> None:
        """Reset all accumulators for a new frame."""
        for accum in self._irradiance_accum:
            accum.radiance_sum = Vec3.zero()
            accum.depth_sum = 0.0
            accum.depth_squared_sum = 0.0
            accum.sample_count = 0
            accum.weight_sum = 0.0

        for accum in self._visibility_accum:
            accum.radiance_sum = Vec3.zero()
            accum.depth_sum = 0.0
            accum.depth_squared_sum = 0.0
            accum.sample_count = 0
            accum.weight_sum = 0.0

    def accumulate(
        self,
        ray_direction: Vec3,
        hit_result: RayHitResult,
        weight: float = 1.0,
    ) -> None:
        """Accumulate a ray hit result.

        Args:
            ray_direction: Ray direction (normalized)
            hit_result: Ray hit result
            weight: Sample weight
        """
        # Convert direction to octahedral coordinates
        oct = self._direction_to_octahedral(ray_direction)

        # Accumulate irradiance
        ir_idx = self._oct_to_index(
            oct, self.irradiance_resolution
        )
        if 0 <= ir_idx < len(self._irradiance_accum):
            self._irradiance_accum[ir_idx].add_sample(
                hit_result.radiance, hit_result.hit_distance, weight
            )

        # Accumulate visibility
        vis_idx = self._oct_to_index(
            oct, self.visibility_resolution
        )
        if 0 <= vis_idx < len(self._visibility_accum):
            depth = (
                hit_result.hit_distance
                if hit_result.hit
                else self.ray_config.max_ray_distance
            )
            self._visibility_accum[vis_idx].add_sample(
                Vec3.zero(), depth, weight
            )

    def apply_temporal_filtering(self) -> None:
        """Apply temporal hysteresis to filtered values."""
        one_minus_h = 1.0 - self.hysteresis

        # Filter irradiance
        for i, accum in enumerate(self._irradiance_accum):
            if accum.sample_count > 0:
                new_val = accum.get_average_radiance()
                old_val = self._irradiance_filtered[i]
                self._irradiance_filtered[i] = Vec3(
                    old_val.x * self.hysteresis + new_val.x * one_minus_h,
                    old_val.y * self.hysteresis + new_val.y * one_minus_h,
                    old_val.z * self.hysteresis + new_val.z * one_minus_h,
                )

        # Filter visibility
        for i, accum in enumerate(self._visibility_accum):
            if accum.sample_count > 0:
                new_mean = accum.get_average_depth()
                new_var = accum.get_depth_variance()
                old_val = self._visibility_filtered[i]
                self._visibility_filtered[i] = Vec2(
                    old_val.x * self.hysteresis + new_mean * one_minus_h,
                    old_val.y * self.hysteresis + new_var * one_minus_h,
                )

    def get_filtered_irradiance(self) -> list[Vec3]:
        """Get temporally filtered irradiance values."""
        return list(self._irradiance_filtered)

    def get_filtered_visibility(self) -> list[Vec2]:
        """Get temporally filtered visibility values."""
        return list(self._visibility_filtered)

    def sample_irradiance(self, direction: Vec3) -> Vec3:
        """Sample irradiance in a direction with bilinear interpolation.

        Args:
            direction: Normalized direction

        Returns:
            Interpolated irradiance
        """
        return self._sample_octahedral_vec3(
            direction, self._irradiance_filtered, self.irradiance_resolution
        )

    def sample_visibility(self, direction: Vec3) -> Vec2:
        """Sample visibility in a direction with bilinear interpolation.

        Args:
            direction: Normalized direction

        Returns:
            Vec2(mean_distance, variance)
        """
        return self._sample_octahedral_vec2(
            direction, self._visibility_filtered, self.visibility_resolution
        )

    def get_irradiance_bytes(self) -> bytes:
        """Get irradiance data as GPU-uploadable bytes.

        Format: resolution^2 * 16 bytes (vec4<f32>, w=1)
        """
        data = []
        for ir in self._irradiance_filtered:
            data.append(struct.pack("<4f", ir.x, ir.y, ir.z, 1.0))
        return b"".join(data)

    def get_visibility_bytes(self) -> bytes:
        """Get visibility data as GPU-uploadable bytes.

        Format: resolution^2 * 8 bytes (vec2<f32>)
        """
        data = []
        for vis in self._visibility_filtered:
            data.append(struct.pack("<2f", vis.x, vis.y))
        return b"".join(data)

    def _direction_to_octahedral(self, direction: Vec3) -> Vec2:
        """Convert direction to octahedral coordinates."""
        d = direction.normalized()

        # Project onto octahedron
        inv_l1 = 1.0 / (abs(d.x) + abs(d.y) + abs(d.z) + 1e-10)
        ox = d.x * inv_l1
        oy = d.y * inv_l1

        # Wrap negative hemisphere
        if d.z < 0:
            sign_x = 1.0 if ox >= 0 else -1.0
            sign_y = 1.0 if oy >= 0 else -1.0
            new_ox = (1.0 - abs(oy)) * sign_x
            new_oy = (1.0 - abs(ox)) * sign_y
            ox, oy = new_ox, new_oy

        # Convert to [0, 1] range
        return Vec2(ox * 0.5 + 0.5, oy * 0.5 + 0.5)

    def _oct_to_index(self, oct: Vec2, resolution: int) -> int:
        """Convert octahedral coordinates to linear index."""
        x = int(oct.x * (resolution - 1))
        y = int(oct.y * (resolution - 1))
        x = max(0, min(resolution - 1, x))
        y = max(0, min(resolution - 1, y))
        return y * resolution + x

    def _sample_octahedral_vec3(
        self, direction: Vec3, data: list[Vec3], resolution: int
    ) -> Vec3:
        """Sample octahedral Vec3 map with bilinear interpolation."""
        oct = self._direction_to_octahedral(direction)

        x = oct.x * (resolution - 1)
        y = oct.y * (resolution - 1)

        x0 = max(0, min(int(x), resolution - 1))
        y0 = max(0, min(int(y), resolution - 1))
        x1 = min(x0 + 1, resolution - 1)
        y1 = min(y0 + 1, resolution - 1)

        fx = x - x0
        fy = y - y0

        def get(xi: int, yi: int) -> Vec3:
            return data[yi * resolution + xi]

        v00 = get(x0, y0)
        v10 = get(x1, y0)
        v01 = get(x0, y1)
        v11 = get(x1, y1)

        return Vec3(
            v00.x * (1 - fx) * (1 - fy)
            + v10.x * fx * (1 - fy)
            + v01.x * (1 - fx) * fy
            + v11.x * fx * fy,
            v00.y * (1 - fx) * (1 - fy)
            + v10.y * fx * (1 - fy)
            + v01.y * (1 - fx) * fy
            + v11.y * fx * fy,
            v00.z * (1 - fx) * (1 - fy)
            + v10.z * fx * (1 - fy)
            + v01.z * (1 - fx) * fy
            + v11.z * fx * fy,
        )

    def _sample_octahedral_vec2(
        self, direction: Vec3, data: list[Vec2], resolution: int
    ) -> Vec2:
        """Sample octahedral Vec2 map with bilinear interpolation."""
        oct = self._direction_to_octahedral(direction)

        x = oct.x * (resolution - 1)
        y = oct.y * (resolution - 1)

        x0 = max(0, min(int(x), resolution - 1))
        y0 = max(0, min(int(y), resolution - 1))
        x1 = min(x0 + 1, resolution - 1)
        y1 = min(y0 + 1, resolution - 1)

        fx = x - x0
        fy = y - y0

        def get(xi: int, yi: int) -> Vec2:
            return data[yi * resolution + xi]

        v00 = get(x0, y0)
        v10 = get(x1, y0)
        v01 = get(x0, y1)
        v11 = get(x1, y1)

        return Vec2(
            v00.x * (1 - fx) * (1 - fy)
            + v10.x * fx * (1 - fy)
            + v01.x * (1 - fx) * fy
            + v11.x * fx * fy,
            v00.y * (1 - fx) * (1 - fy)
            + v10.y * fx * (1 - fy)
            + v01.y * (1 - fx) * fy
            + v11.y * fx * fy,
        )


# ============================================================================
# TLAS Interface
# ============================================================================


class TLASInterface(Protocol):
    """Protocol for TLAS (Top-Level Acceleration Structure) access.

    Implementations must provide ray traversal against scene geometry.
    """

    def trace_ray(
        self,
        origin: Vec3,
        direction: Vec3,
        max_distance: float,
        flags: int = RAY_FLAG_NONE,
    ) -> RayHitResult:
        """Trace a ray against the TLAS.

        Args:
            origin: Ray origin
            direction: Ray direction (normalized)
            max_distance: Maximum trace distance
            flags: Ray flags

        Returns:
            Ray hit result
        """
        ...

    def is_valid(self) -> bool:
        """Check if the TLAS is valid and ready for tracing."""
        ...


# ============================================================================
# DDGI RT Probe Updater
# ============================================================================


@dataclass
class DDGIRTUpdateConfig:
    """Configuration for DDGI RT probe updates.

    Attributes:
        ray_config: Ray generation configuration
        hysteresis: Temporal blending factor
        irradiance_resolution: Irradiance map resolution
        visibility_resolution: Visibility map resolution
        probes_per_frame: Maximum probes to update per frame
        use_indirect_dispatch: Use indirect compute dispatch
        enable_backface_culling: Cull backfacing geometry
    """

    ray_config: ProbeRayConfig = field(default_factory=ProbeRayConfig)
    hysteresis: float = DEFAULT_HYSTERESIS
    irradiance_resolution: int = 8
    visibility_resolution: int = 16
    probes_per_frame: int = 128
    use_indirect_dispatch: bool = True
    enable_backface_culling: bool = True

    def validate(self) -> list[str]:
        """Validate configuration."""
        errors = self.ray_config.validate()

        if not 0.0 <= self.hysteresis <= 1.0:
            errors.append("hysteresis must be between 0.0 and 1.0")

        if self.irradiance_resolution < 2 or self.irradiance_resolution > 16:
            errors.append("irradiance_resolution must be between 2 and 16")

        if self.visibility_resolution < 2 or self.visibility_resolution > 32:
            errors.append("visibility_resolution must be between 2 and 32")

        if self.probes_per_frame < 1:
            errors.append("probes_per_frame must be at least 1")

        return errors


@dataclass
class ProbeUpdateResult:
    """Result of updating probes.

    Attributes:
        probes_updated: Number of probes updated
        rays_traced: Total rays traced
        update_time_ms: Update time in milliseconds
        tlas_valid: Whether TLAS was valid
    """

    probes_updated: int = 0
    rays_traced: int = 0
    update_time_ms: float = 0.0
    tlas_valid: bool = True


class DDGIRTProbeUpdater:
    """Updates DDGI probes using hardware ray tracing.

    This class manages the ray tracing update loop for DDGI probes:
    1. Generate stratified ray directions per probe
    2. Trace rays against TLAS
    3. Accumulate radiance at hit points
    4. Apply temporal filtering

    Usage:
        updater = DDGIRTProbeUpdater(config)
        result = updater.update(probe_positions, tlas, frame_index)
    """

    def __init__(self, config: DDGIRTUpdateConfig) -> None:
        """Initialize the probe updater.

        Args:
            config: Update configuration
        """
        self.config = config
        self._ray_generator = ProbeRayGenerator(config.ray_config)

        # Per-probe accumulators (created on first update)
        self._accumulators: dict[int, RadianceAccumulator] = {}

        # Statistics
        self._last_result = ProbeUpdateResult()
        self._total_probes_updated = 0
        self._total_rays_traced = 0

    @property
    def ray_generator(self) -> ProbeRayGenerator:
        """Get the ray generator."""
        return self._ray_generator

    def get_or_create_accumulator(self, probe_id: int) -> RadianceAccumulator:
        """Get or create an accumulator for a probe.

        Args:
            probe_id: Probe identifier

        Returns:
            Radiance accumulator for the probe
        """
        if probe_id not in self._accumulators:
            self._accumulators[probe_id] = RadianceAccumulator(
                self.config.ray_config,
                self.config.hysteresis,
                self.config.irradiance_resolution,
                self.config.visibility_resolution,
            )
        return self._accumulators[probe_id]

    def update(
        self,
        probe_positions: list[Vec3],
        tlas: TLASInterface,
        frame_index: int = 0,
        radiance_callback: Optional[
            Callable[[Vec3, Vec3, RayHitResult], Vec3]
        ] = None,
    ) -> ProbeUpdateResult:
        """Update probes using ray tracing.

        Args:
            probe_positions: List of probe world positions
            tlas: TLAS for ray traversal
            frame_index: Current frame index
            radiance_callback: Optional callback to compute radiance at hits

        Returns:
            Update result with statistics
        """
        result = ProbeUpdateResult()

        # Check TLAS validity
        if not tlas.is_valid():
            result.tlas_valid = False
            self._last_result = result
            return result

        # Limit probes per frame
        probes_to_update = min(
            len(probe_positions), self.config.probes_per_frame
        )

        # Generate ray directions for this frame
        ray_directions = self._ray_generator.generate_directions(frame_index)

        # Update each probe
        for i in range(probes_to_update):
            probe_id = (frame_index * self.config.probes_per_frame + i) % max(
                1, len(probe_positions)
            )
            position = probe_positions[probe_id]

            # Get accumulator for this probe
            accumulator = self.get_or_create_accumulator(probe_id)
            accumulator.reset_accumulators()

            # Trace rays
            for direction in ray_directions:
                hit = tlas.trace_ray(
                    position,
                    direction,
                    self.config.ray_config.max_ray_distance,
                    (
                        RAY_FLAG_CULL_BACK_FACING
                        if self.config.enable_backface_culling
                        else RAY_FLAG_NONE
                    ),
                )

                # Apply radiance callback if provided
                if radiance_callback and hit.hit:
                    hit.radiance = radiance_callback(
                        position, direction, hit
                    )

                # Accumulate
                accumulator.accumulate(direction, hit)
                result.rays_traced += 1

            # Apply temporal filtering
            accumulator.apply_temporal_filtering()
            result.probes_updated += 1

        self._last_result = result
        self._total_probes_updated += result.probes_updated
        self._total_rays_traced += result.rays_traced

        return result

    def get_probe_irradiance(self, probe_id: int) -> list[Vec3]:
        """Get filtered irradiance for a probe.

        Args:
            probe_id: Probe identifier

        Returns:
            Filtered irradiance values (resolution^2 entries)
        """
        if probe_id not in self._accumulators:
            size = self.config.irradiance_resolution ** 2
            return [Vec3.zero() for _ in range(size)]
        return self._accumulators[probe_id].get_filtered_irradiance()

    def get_probe_visibility(self, probe_id: int) -> list[Vec2]:
        """Get filtered visibility for a probe.

        Args:
            probe_id: Probe identifier

        Returns:
            Filtered visibility values (resolution^2 entries)
        """
        if probe_id not in self._accumulators:
            size = self.config.visibility_resolution ** 2
            return [
                Vec2(self.config.ray_config.max_ray_distance, 0.0)
                for _ in range(size)
            ]
        return self._accumulators[probe_id].get_filtered_visibility()

    def sample_probe_irradiance(
        self, probe_id: int, direction: Vec3
    ) -> Vec3:
        """Sample irradiance from a probe in a direction.

        Args:
            probe_id: Probe identifier
            direction: Sample direction

        Returns:
            Interpolated irradiance
        """
        if probe_id not in self._accumulators:
            return Vec3.zero()
        return self._accumulators[probe_id].sample_irradiance(direction)

    def sample_probe_visibility(
        self, probe_id: int, direction: Vec3
    ) -> Vec2:
        """Sample visibility from a probe in a direction.

        Args:
            probe_id: Probe identifier
            direction: Sample direction

        Returns:
            Vec2(mean_distance, variance)
        """
        if probe_id not in self._accumulators:
            return Vec2(self.config.ray_config.max_ray_distance, 0.0)
        return self._accumulators[probe_id].sample_visibility(direction)

    def get_statistics(self) -> dict:
        """Get update statistics."""
        return {
            "last_probes_updated": self._last_result.probes_updated,
            "last_rays_traced": self._last_result.rays_traced,
            "last_tlas_valid": self._last_result.tlas_valid,
            "total_probes_updated": self._total_probes_updated,
            "total_rays_traced": self._total_rays_traced,
            "active_accumulators": len(self._accumulators),
        }

    def clear_accumulators(self) -> None:
        """Clear all probe accumulators."""
        self._accumulators.clear()


# ============================================================================
# WGSL Shader Generation
# ============================================================================


def generate_ddgi_probe_update_wgsl(
    config: DDGIRTUpdateConfig,
    workgroup_size: tuple[int, int, int] = (8, 8, 1),
) -> str:
    """Generate WGSL compute shader for DDGI probe RT updates.

    Args:
        config: Update configuration
        workgroup_size: Compute workgroup size

    Returns:
        WGSL shader source
    """
    rays = config.ray_config.rays_per_probe
    max_dist = config.ray_config.max_ray_distance
    hysteresis = config.hysteresis
    ir_res = config.irradiance_resolution
    vis_res = config.visibility_resolution

    return f"""// DDGI Probe Update Compute Shader (Hardware RT Path)
// Generated for: {rays} rays/probe, {ir_res}x{ir_res} irradiance, {vis_res}x{vis_res} visibility

// Bindings
@group(0) @binding(0) var<storage, read> probe_positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> ray_directions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> irradiance_output: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> visibility_output: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> update_params: UpdateParams;
@group(1) @binding(0) var tlas: acceleration_structure;

struct UpdateParams {{
    frame_index: u32,
    probe_count: u32,
    rays_per_probe: u32,
    max_ray_distance: f32,
    hysteresis: f32,
    irradiance_resolution: u32,
    visibility_resolution: u32,
    _pad: u32,
}}

struct RayPayload {{
    radiance: vec3<f32>,
    hit_distance: f32,
    hit: bool,
}}

// Constants
const RAYS_PER_PROBE: u32 = {rays}u;
const MAX_RAY_DISTANCE: f32 = {max_dist};
const HYSTERESIS: f32 = {hysteresis};
const IRRADIANCE_RES: u32 = {ir_res}u;
const VISIBILITY_RES: u32 = {vis_res}u;

// Direction to octahedral coordinates
fn direction_to_octahedral(dir: vec3<f32>) -> vec2<f32> {{
    let d = normalize(dir);
    let inv_l1 = 1.0 / (abs(d.x) + abs(d.y) + abs(d.z) + 0.0001);
    var o = vec2<f32>(d.x * inv_l1, d.y * inv_l1);

    if (d.z < 0.0) {{
        let sign_x = select(-1.0, 1.0, o.x >= 0.0);
        let sign_y = select(-1.0, 1.0, o.y >= 0.0);
        o = vec2<f32>(
            (1.0 - abs(o.y)) * sign_x,
            (1.0 - abs(o.x)) * sign_y
        );
    }}

    return o * 0.5 + 0.5;
}}

// Trace ray against TLAS
fn trace_probe_ray(
    origin: vec3<f32>,
    direction: vec3<f32>,
) -> RayPayload {{
    var payload: RayPayload;
    payload.radiance = vec3<f32>(0.0);
    payload.hit_distance = MAX_RAY_DISTANCE;
    payload.hit = false;

    // Hardware RT query
    var ray: ray_desc;
    ray.origin = origin;
    ray.direction = direction;
    ray.t_min = 0.001;
    ray.t_max = MAX_RAY_DISTANCE;

    var intersection = ray_query(tlas, ray, RAY_FLAG_CULL_BACK_FACING);

    if (intersection.hit) {{
        payload.hit = true;
        payload.hit_distance = intersection.t;
        // Radiance would be computed from material/lighting
        // For now, use distance-based falloff
        let falloff = 1.0 - (intersection.t / MAX_RAY_DISTANCE);
        payload.radiance = vec3<f32>(falloff);
    }}

    return payload;
}}

@compute @workgroup_size({workgroup_size[0]}, {workgroup_size[1]}, {workgroup_size[2]})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let probe_idx = global_id.x;
    let ray_idx = global_id.y;

    if (probe_idx >= update_params.probe_count) {{
        return;
    }}
    if (ray_idx >= RAYS_PER_PROBE) {{
        return;
    }}

    // Get probe position and ray direction
    let probe_pos = probe_positions[probe_idx].xyz;
    let ray_dir = ray_directions[ray_idx].xyz;

    // Trace ray
    let payload = trace_probe_ray(probe_pos, ray_dir);

    // Convert direction to octahedral coordinates
    let oct = direction_to_octahedral(ray_dir);

    // Compute irradiance texel index
    let ir_x = u32(oct.x * f32(IRRADIANCE_RES - 1u));
    let ir_y = u32(oct.y * f32(IRRADIANCE_RES - 1u));
    let ir_idx = probe_idx * IRRADIANCE_RES * IRRADIANCE_RES + ir_y * IRRADIANCE_RES + ir_x;

    // Compute visibility texel index
    let vis_x = u32(oct.x * f32(VISIBILITY_RES - 1u));
    let vis_y = u32(oct.y * f32(VISIBILITY_RES - 1u));
    let vis_idx = probe_idx * VISIBILITY_RES * VISIBILITY_RES + vis_y * VISIBILITY_RES + vis_x;

    // Update irradiance with temporal filtering (atomic add would be better)
    let old_ir = irradiance_output[ir_idx].xyz;
    let new_ir = mix(payload.radiance, old_ir, HYSTERESIS);
    irradiance_output[ir_idx] = vec4<f32>(new_ir, 1.0);

    // Update visibility
    let old_vis = visibility_output[vis_idx];
    let new_depth = payload.hit_distance;
    let new_var = new_depth * new_depth;
    let filtered_vis = vec2<f32>(
        mix(new_depth, old_vis.x, HYSTERESIS),
        mix(new_var, old_vis.y, HYSTERESIS)
    );
    visibility_output[vis_idx] = filtered_vis;
}}
"""


# ============================================================================
# Utility Functions
# ============================================================================


def create_mock_tlas() -> TLASInterface:
    """Create a mock TLAS for testing.

    Returns:
        Mock TLAS that returns simple hit results
    """

    class MockTLAS:
        def trace_ray(
            self,
            origin: Vec3,
            direction: Vec3,
            max_distance: float,
            flags: int = RAY_FLAG_NONE,
        ) -> RayHitResult:
            # Simple ground plane at y=0
            if direction.y < -0.01:
                t = -origin.y / direction.y
                if 0 < t < max_distance:
                    hit_pos = origin + direction * t
                    return RayHitResult(
                        hit=True,
                        hit_distance=t,
                        hit_position=hit_pos,
                        hit_normal=Vec3(0, 1, 0),
                        radiance=Vec3(0.3, 0.4, 0.3),  # Green ground
                    )

            # Sky miss
            return RayHitResult(
                hit=False,
                hit_distance=max_distance,
                hit_position=origin + direction * max_distance,
                hit_normal=Vec3(0, 1, 0),
                radiance=Vec3(0.5, 0.6, 0.9),  # Sky color
            )

        def is_valid(self) -> bool:
            return True

    return MockTLAS()


def estimate_memory_usage(
    probe_count: int,
    config: DDGIRTUpdateConfig,
) -> int:
    """Estimate GPU memory usage in bytes.

    Args:
        probe_count: Number of probes
        config: Update configuration

    Returns:
        Estimated memory in bytes
    """
    ir_texels = config.irradiance_resolution ** 2
    vis_texels = config.visibility_resolution ** 2

    # Per-probe memory
    ir_bytes = ir_texels * 16  # vec4<f32>
    vis_bytes = vis_texels * 8  # vec2<f32>
    per_probe = ir_bytes + vis_bytes

    # Ray direction buffer
    ray_bytes = config.ray_config.rays_per_probe * 16  # vec4<f32>

    # Probe position buffer
    pos_bytes = probe_count * 16  # vec4<f32>

    return probe_count * per_probe + ray_bytes + pos_bytes

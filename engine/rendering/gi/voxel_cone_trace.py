"""Voxel Cone Tracing for VXGI (T-GIR-P7.3).

This module implements voxel cone tracing for indirect diffuse and specular
illumination using the voxel mip chain from P7.2 and scene voxelization
from P7.1.

Features:
    - 6-12 diffuse cones with wide aperture (60-90 degrees)
    - 1-4 specular cones with narrow aperture (5-30 degrees based on roughness)
    - Exponential step spacing for efficient tracing
    - Mip level selection based on cone footprint
    - Front-to-back opacity-weighted compositing
    - Full-screen GI pass integration with G-buffer

Cone Tracing Algorithm:
    ```
    accumulated_color = 0
    accumulated_alpha = 0
    t = start_offset
    while t < max_distance and accumulated_alpha < 0.99:
        diameter = 2 * t * tan(aperture)
        mip = log2(diameter / voxel_size)
        sample = sample_voxel_at_mip(position + t * direction, mip)
        weight = (1 - accumulated_alpha)
        accumulated_color += weight * sample.rgb
        accumulated_alpha += weight * sample.a
        t *= step_multiplier  # exponential: 1.1-1.3
    ```

Performance Targets:
    - 256^3 voxel tracing: <4ms per frame
    - 128^3 voxel tracing: <1.5ms per frame
    - 64^3 voxel tracing: <0.5ms per frame

References:
    - Crassin et al., "Interactive Indirect Illumination Using Voxel Cone Tracing"
    - Cyril Crassin, NVIDIA VXGI documentation
    - GPU Gems 2, "Real-Time Global Illumination"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Iterator, Optional, Sequence, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from engine.core.math.vec import Vec3

from engine.rendering.gi.voxel_mipchain import (
    VoxelMipChain,
    VoxelData,
    MipResolution,
    compute_mip_count,
)


# ============================================================================
# Constants
# ============================================================================

# Diffuse cone aperture range (half-angle in radians)
DIFFUSE_APERTURE_MIN = math.radians(30.0)   # 60 degrees full angle
DIFFUSE_APERTURE_MAX = math.radians(45.0)   # 90 degrees full angle
DIFFUSE_APERTURE_DEFAULT = math.radians(36.52)  # Optimal for 6 cones

# Specular cone aperture range
SPECULAR_APERTURE_MIN = math.radians(2.5)   # Very glossy (roughness ~0.05)
SPECULAR_APERTURE_MAX = math.radians(15.0)  # Semi-rough (roughness ~0.5)

# Cone count limits
MIN_DIFFUSE_CONES = 6
MAX_DIFFUSE_CONES = 12
DEFAULT_DIFFUSE_CONES = 6

MIN_SPECULAR_CONES = 1
MAX_SPECULAR_CONES = 4
DEFAULT_SPECULAR_CONES = 1

# Tracing parameters
DEFAULT_START_OFFSET = 0.5    # Offset in voxel units
DEFAULT_MAX_DISTANCE = 256.0  # Maximum trace distance in voxel units
DEFAULT_STEP_MULTIPLIER = 1.2  # Exponential step growth
DEFAULT_OPACITY_THRESHOLD = 0.99  # Terminate at this accumulated opacity

# Golden ratio for cone distribution
GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0
GOLDEN_ANGLE = 2.0 * math.pi / (GOLDEN_RATIO * GOLDEN_RATIO)

# Epsilon for numerical stability
EPSILON = 1e-6


# ============================================================================
# Cone Configuration
# ============================================================================


@dataclass
class ConeConfig:
    """Configuration for a single tracing cone.

    Attributes:
        direction: Normalized cone direction in world space
        aperture: Cone half-angle in radians
        max_distance: Maximum trace distance in voxel units
        start_offset: Offset from surface to avoid self-occlusion
    """

    direction: Vec3
    aperture: float
    max_distance: float = DEFAULT_MAX_DISTANCE
    start_offset: float = DEFAULT_START_OFFSET

    def __post_init__(self) -> None:
        """Validate and normalize direction."""
        if self.direction.length() < EPSILON:
            raise ValueError("Cone direction cannot be zero")

        # Normalize direction
        length = self.direction.length()
        self.direction = Vec3(
            self.direction.x / length,
            self.direction.y / length,
            self.direction.z / length,
        )

        # Validate aperture
        if self.aperture <= 0:
            raise ValueError(f"Aperture must be positive, got {self.aperture}")
        if self.aperture > math.pi / 2:
            raise ValueError(f"Aperture must be < PI/2, got {self.aperture}")

    @property
    def full_angle(self) -> float:
        """Get the full cone angle in radians."""
        return self.aperture * 2.0

    @property
    def tan_aperture(self) -> float:
        """Tangent of the aperture angle (cached for performance)."""
        return math.tan(self.aperture)

    @property
    def solid_angle(self) -> float:
        """Solid angle covered by this cone in steradians."""
        return 2.0 * math.pi * (1.0 - math.cos(self.aperture))

    def diameter_at_distance(self, distance: float) -> float:
        """Compute cone diameter at a given distance.

        Args:
            distance: Distance from cone origin

        Returns:
            Diameter of cone cross-section
        """
        return 2.0 * distance * self.tan_aperture

    def mip_at_distance(self, distance: float, voxel_size: float) -> float:
        """Compute appropriate mip level for sampling at distance.

        Args:
            distance: Distance from cone origin
            voxel_size: Size of one voxel at mip 0

        Returns:
            Mip level (floating point for interpolation)
        """
        diameter = self.diameter_at_distance(distance)
        if diameter < voxel_size:
            return 0.0
        return math.log2(diameter / voxel_size)


# ============================================================================
# Diffuse Cone Distribution
# ============================================================================


class DiffuseConeDistribution:
    """Generator for diffuse cone directions.

    Creates a cosine-weighted hemisphere distribution of wide aperture
    cones for gathering diffuse indirect illumination.

    The default 6-cone distribution uses directions that cover the
    hemisphere uniformly while maximizing coverage overlap for smooth
    GI results.

    Attributes:
        cone_count: Number of diffuse cones (6-12)
        aperture: Half-angle aperture for all cones
        directions: Pre-computed cone directions in tangent space
    """

    __slots__ = ("_cone_count", "_aperture", "_directions")

    def __init__(
        self,
        cone_count: int = DEFAULT_DIFFUSE_CONES,
        aperture: float = DIFFUSE_APERTURE_DEFAULT,
    ) -> None:
        """Initialize diffuse cone distribution.

        Args:
            cone_count: Number of cones (6-12)
            aperture: Half-angle in radians

        Raises:
            ValueError: If cone_count out of range
        """
        if not MIN_DIFFUSE_CONES <= cone_count <= MAX_DIFFUSE_CONES:
            raise ValueError(
                f"Diffuse cone count must be {MIN_DIFFUSE_CONES}-{MAX_DIFFUSE_CONES}, "
                f"got {cone_count}"
            )

        self._cone_count = cone_count
        self._aperture = aperture
        self._directions = self._generate_directions()

    @property
    def cone_count(self) -> int:
        """Number of diffuse cones."""
        return self._cone_count

    @property
    def aperture(self) -> float:
        """Half-angle aperture in radians."""
        return self._aperture

    def set_cone_count(self, count: int) -> None:
        """Update the number of cones.

        Args:
            count: New cone count (6-12)
        """
        if not MIN_DIFFUSE_CONES <= count <= MAX_DIFFUSE_CONES:
            raise ValueError(
                f"Diffuse cone count must be {MIN_DIFFUSE_CONES}-{MAX_DIFFUSE_CONES}"
            )
        self._cone_count = count
        self._directions = self._generate_directions()

    def set_aperture(self, aperture: float) -> None:
        """Update the aperture angle.

        Args:
            aperture: New half-angle in radians
        """
        if aperture <= 0 or aperture >= math.pi / 2:
            raise ValueError("Aperture must be in (0, PI/2)")
        self._aperture = aperture

    def _generate_directions(self) -> list[Vec3]:
        """Generate cosine-weighted hemisphere directions.

        Uses a specialized distribution for common cone counts:
        - 6 cones: Axis-aligned + tilted (classic VXGI)
        - 8+ cones: Fibonacci spiral distribution

        Returns:
            List of normalized directions in tangent space (Z = up)
        """
        if self._cone_count == 6:
            return self._generate_6_cone_dirs()
        else:
            return self._generate_fibonacci_dirs()

    def _generate_6_cone_dirs(self) -> list[Vec3]:
        """Generate classic 6-cone VXGI distribution.

        This distribution consists of:
        - 1 cone pointing up (along normal)
        - 5 cones tilted ~60 degrees from normal, equally spaced azimuthally

        Returns:
            List of 6 directions in tangent space
        """
        directions = []

        # Central cone pointing up (along normal)
        directions.append(Vec3(0.0, 0.0, 1.0))

        # 5 cones at 60 degrees from normal
        tilt_angle = math.radians(60.0)
        cos_tilt = math.cos(tilt_angle)
        sin_tilt = math.sin(tilt_angle)

        for i in range(5):
            azimuth = (2.0 * math.pi * i) / 5.0
            x = sin_tilt * math.cos(azimuth)
            y = sin_tilt * math.sin(azimuth)
            z = cos_tilt
            directions.append(Vec3(x, y, z))

        return directions

    def _generate_fibonacci_dirs(self) -> list[Vec3]:
        """Generate Fibonacci spiral hemisphere distribution.

        Uses the golden angle to create a quasi-uniform distribution
        of directions over a hemisphere.

        Returns:
            List of directions in tangent space
        """
        directions = []

        for i in range(self._cone_count):
            # Cosine-weighted elevation
            t = (i + 0.5) / self._cone_count
            phi = 1.0 - t  # Cosine-weighted: more samples near pole
            theta = GOLDEN_ANGLE * i

            # Spherical to Cartesian (Z = up)
            sin_phi = math.sqrt(1.0 - phi * phi)
            x = sin_phi * math.cos(theta)
            y = sin_phi * math.sin(theta)
            z = phi

            directions.append(Vec3(x, y, z))

        return directions

    def get_cones(
        self,
        max_distance: float = DEFAULT_MAX_DISTANCE,
        start_offset: float = DEFAULT_START_OFFSET,
    ) -> list[ConeConfig]:
        """Get cone configurations in tangent space.

        Args:
            max_distance: Maximum trace distance
            start_offset: Offset from surface

        Returns:
            List of ConeConfig for all diffuse cones
        """
        return [
            ConeConfig(
                direction=d,
                aperture=self._aperture,
                max_distance=max_distance,
                start_offset=start_offset,
            )
            for d in self._directions
        ]

    def transform_to_surface(
        self,
        normal: Vec3,
        tangent: Optional[Vec3] = None,
        max_distance: float = DEFAULT_MAX_DISTANCE,
        start_offset: float = DEFAULT_START_OFFSET,
    ) -> list[ConeConfig]:
        """Transform cones from tangent space to world space.

        Builds a tangent-space frame from the normal and transforms
        all cone directions to world space.

        Args:
            normal: Surface normal (will be normalized)
            tangent: Optional tangent vector (computed if not provided)
            max_distance: Maximum trace distance
            start_offset: Offset from surface

        Returns:
            List of ConeConfig in world space
        """
        # Normalize normal
        n_len = normal.length()
        if n_len < EPSILON:
            raise ValueError("Normal cannot be zero")
        n = Vec3(normal.x / n_len, normal.y / n_len, normal.z / n_len)

        # Build tangent frame
        if tangent is not None:
            t_len = tangent.length()
            if t_len < EPSILON:
                raise ValueError("Tangent cannot be zero")
            t = Vec3(tangent.x / t_len, tangent.y / t_len, tangent.z / t_len)
        else:
            # Generate tangent from normal
            t = _orthogonal_vector(n)

        # Bitangent
        b = n.cross(t)
        b_len = b.length()
        if b_len > EPSILON:
            b = Vec3(b.x / b_len, b.y / b_len, b.z / b_len)
        else:
            # Fallback
            b = Vec3(0.0, 1.0, 0.0)

        # Transform each cone direction
        cones = []
        for local_dir in self._directions:
            # Transform: world = local.x * T + local.y * B + local.z * N
            world_dir = Vec3(
                local_dir.x * t.x + local_dir.y * b.x + local_dir.z * n.x,
                local_dir.x * t.y + local_dir.y * b.y + local_dir.z * n.y,
                local_dir.x * t.z + local_dir.y * b.z + local_dir.z * n.z,
            )

            cones.append(ConeConfig(
                direction=world_dir,
                aperture=self._aperture,
                max_distance=max_distance,
                start_offset=start_offset,
            ))

        return cones

    def get_total_solid_angle(self) -> float:
        """Get total solid angle covered by all cones.

        Returns:
            Sum of solid angles (may exceed hemisphere due to overlap)
        """
        single_cone_solid_angle = 2.0 * math.pi * (1.0 - math.cos(self._aperture))
        return single_cone_solid_angle * self._cone_count

    def get_hemisphere_coverage(self) -> float:
        """Estimate hemisphere coverage ratio.

        Returns:
            Approximate fraction of hemisphere covered (0 to 1+)
        """
        hemisphere_solid_angle = 2.0 * math.pi
        return min(1.0, self.get_total_solid_angle() / hemisphere_solid_angle)


# ============================================================================
# Specular Cone Distribution
# ============================================================================


class SpecularConeDistribution:
    """Generator for specular cone directions.

    Creates narrow aperture cones for gathering specular/glossy
    reflections. The aperture is derived from surface roughness.

    Single cone (roughness < 0.3):
        - One cone along reflection direction
        - Aperture = roughness_to_aperture(roughness)

    Multiple cones (roughness >= 0.3):
        - 2-4 cones spread around reflection direction
        - Spread angle proportional to roughness

    Attributes:
        cone_count: Number of specular cones (1-4)
        base_aperture: Base aperture before roughness adjustment
    """

    __slots__ = ("_cone_count", "_base_aperture")

    def __init__(
        self,
        cone_count: int = DEFAULT_SPECULAR_CONES,
        base_aperture: float = SPECULAR_APERTURE_MIN,
    ) -> None:
        """Initialize specular cone distribution.

        Args:
            cone_count: Number of cones (1-4)
            base_aperture: Base half-angle in radians
        """
        if not MIN_SPECULAR_CONES <= cone_count <= MAX_SPECULAR_CONES:
            raise ValueError(
                f"Specular cone count must be {MIN_SPECULAR_CONES}-{MAX_SPECULAR_CONES}"
            )

        self._cone_count = cone_count
        self._base_aperture = base_aperture

    @property
    def cone_count(self) -> int:
        """Number of specular cones."""
        return self._cone_count

    @staticmethod
    def aperture_from_roughness(roughness: float) -> float:
        """Convert roughness to cone aperture.

        Uses a perceptually linear mapping from roughness (0-1) to
        aperture angle. Mirrors GGX lobe width approximation.

        Args:
            roughness: Surface roughness (0 = mirror, 1 = diffuse)

        Returns:
            Half-angle aperture in radians
        """
        # Clamp roughness
        roughness = max(0.0, min(1.0, roughness))

        # Square roughness for perceptual linearity
        alpha = roughness * roughness

        # Map to aperture range
        # roughness 0 -> ~5 degrees
        # roughness 0.5 -> ~15 degrees
        # roughness 1.0 -> ~30 degrees
        aperture = SPECULAR_APERTURE_MIN + alpha * (
            SPECULAR_APERTURE_MAX - SPECULAR_APERTURE_MIN
        )

        return aperture

    def get_cones(
        self,
        reflection_dir: Vec3,
        roughness: float,
        max_distance: float = DEFAULT_MAX_DISTANCE,
        start_offset: float = DEFAULT_START_OFFSET,
    ) -> list[ConeConfig]:
        """Get specular cone configurations.

        Args:
            reflection_dir: Primary reflection direction (normalized)
            roughness: Surface roughness (0-1)
            max_distance: Maximum trace distance
            start_offset: Offset from surface

        Returns:
            List of ConeConfig for specular cones
        """
        # Normalize reflection direction
        r_len = reflection_dir.length()
        if r_len < EPSILON:
            raise ValueError("Reflection direction cannot be zero")
        r = Vec3(
            reflection_dir.x / r_len,
            reflection_dir.y / r_len,
            reflection_dir.z / r_len,
        )

        aperture = self.aperture_from_roughness(roughness)

        if self._cone_count == 1:
            # Single cone along reflection
            return [ConeConfig(
                direction=r,
                aperture=aperture,
                max_distance=max_distance,
                start_offset=start_offset,
            )]

        # Multiple cones: spread around reflection
        cones = []

        # Central cone (always included)
        cones.append(ConeConfig(
            direction=r,
            aperture=aperture,
            max_distance=max_distance,
            start_offset=start_offset,
        ))

        # Additional cones spread around central
        spread_angle = aperture * 0.5  # Spread by half the aperture
        ortho = _orthogonal_vector(r)
        ortho2 = r.cross(ortho)
        ortho2_len = ortho2.length()
        if ortho2_len > EPSILON:
            ortho2 = Vec3(ortho2.x / ortho2_len, ortho2.y / ortho2_len, ortho2.z / ortho2_len)

        for i in range(1, self._cone_count):
            # Angle around reflection direction
            angle = (2.0 * math.pi * i) / (self._cone_count - 1)

            # Rotate in the plane perpendicular to reflection
            cos_spread = math.cos(spread_angle)
            sin_spread = math.sin(spread_angle)
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)

            # Combine rotation
            offset_x = ortho.x * cos_angle + ortho2.x * sin_angle
            offset_y = ortho.y * cos_angle + ortho2.y * sin_angle
            offset_z = ortho.z * cos_angle + ortho2.z * sin_angle

            # New direction tilted from reflection
            new_dir = Vec3(
                r.x * cos_spread + offset_x * sin_spread,
                r.y * cos_spread + offset_y * sin_spread,
                r.z * cos_spread + offset_z * sin_spread,
            )

            # Normalize
            nd_len = new_dir.length()
            if nd_len > EPSILON:
                new_dir = Vec3(new_dir.x / nd_len, new_dir.y / nd_len, new_dir.z / nd_len)

            cones.append(ConeConfig(
                direction=new_dir,
                aperture=aperture,
                max_distance=max_distance,
                start_offset=start_offset,
            ))

        return cones


# ============================================================================
# Cone Trace Result
# ============================================================================


@dataclass
class ConeTraceResult:
    """Result of tracing a single cone.

    Attributes:
        radiance: Accumulated radiance (RGB)
        opacity: Final accumulated opacity
        steps: Number of trace steps taken
        distance: Total distance traced
        hit_solid: Whether trace hit a fully opaque voxel
    """

    radiance: NDArray[np.float32]  # Shape (3,)
    opacity: float
    steps: int
    distance: float
    hit_solid: bool

    def __post_init__(self) -> None:
        """Validate radiance array."""
        self.radiance = np.asarray(self.radiance, dtype=np.float32)
        if self.radiance.shape != (3,):
            raise ValueError(f"Expected radiance shape (3,), got {self.radiance.shape}")

    @classmethod
    def empty(cls) -> ConeTraceResult:
        """Create empty trace result (no contribution)."""
        return cls(
            radiance=np.zeros(3, dtype=np.float32),
            opacity=0.0,
            steps=0,
            distance=0.0,
            hit_solid=False,
        )

    def luminance(self) -> float:
        """Compute luminance of traced radiance.

        Returns:
            Perceptual luminance (Rec. 709)
        """
        return float(
            0.2126 * self.radiance[0] +
            0.7152 * self.radiance[1] +
            0.0722 * self.radiance[2]
        )


# ============================================================================
# Voxel Cone Tracer
# ============================================================================


@dataclass
class ConeTracerConfig:
    """Configuration for voxel cone tracer.

    Attributes:
        step_multiplier: Exponential step growth factor (1.1-1.3)
        opacity_threshold: Terminate when opacity exceeds this
        max_mip_level: Maximum mip level to sample
        trilinear_sampling: Use trilinear interpolation
        variance_weighting: Weight samples by variance
    """

    step_multiplier: float = DEFAULT_STEP_MULTIPLIER
    opacity_threshold: float = DEFAULT_OPACITY_THRESHOLD
    max_mip_level: Optional[int] = None
    trilinear_sampling: bool = True
    variance_weighting: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.step_multiplier <= 1.0:
            raise ValueError(f"step_multiplier must be > 1.0, got {self.step_multiplier}")
        if not 0.0 < self.opacity_threshold <= 1.0:
            raise ValueError(f"opacity_threshold must be in (0, 1], got {self.opacity_threshold}")


class VoxelConeTracer:
    """Traces cones through a voxel mip chain.

    Implements the core cone tracing algorithm:
    1. Step along cone ray with exponential spacing
    2. Compute cone diameter at each step
    3. Select mip level from diameter
    4. Sample voxel at appropriate mip
    5. Composite front-to-back with opacity weighting

    Attributes:
        mip_chain: Voxel mip chain to trace through
        config: Tracer configuration
        voxel_size: Size of one voxel at mip 0
    """

    __slots__ = ("_mip_chain", "_config", "_voxel_size", "_max_mip")

    def __init__(
        self,
        mip_chain: VoxelMipChain,
        config: Optional[ConeTracerConfig] = None,
        voxel_size: float = 1.0,
    ) -> None:
        """Initialize cone tracer.

        Args:
            mip_chain: Voxel data to trace through
            config: Tracer configuration
            voxel_size: World-space size of one voxel
        """
        self._mip_chain = mip_chain
        self._config = config or ConeTracerConfig()
        self._voxel_size = voxel_size

        # Determine max mip level
        if self._config.max_mip_level is not None:
            self._max_mip = min(
                self._config.max_mip_level,
                mip_chain.mip_count - 1
            )
        else:
            self._max_mip = mip_chain.mip_count - 1

    @property
    def voxel_size(self) -> float:
        """Size of one voxel at mip 0."""
        return self._voxel_size

    @property
    def max_mip_level(self) -> int:
        """Maximum mip level for sampling."""
        return self._max_mip

    def trace_cone(
        self,
        origin: Vec3,
        cone: ConeConfig,
    ) -> ConeTraceResult:
        """Trace a single cone through the voxel grid.

        Args:
            origin: Starting position in voxel coordinates
            cone: Cone configuration

        Returns:
            ConeTraceResult with accumulated radiance and opacity
        """
        # Initialize accumulation
        accumulated_rgb = np.zeros(3, dtype=np.float32)
        accumulated_alpha = 0.0
        t = cone.start_offset
        steps = 0

        # Precompute tan(aperture) for diameter calculation
        tan_aperture = cone.tan_aperture

        while t < cone.max_distance and accumulated_alpha < self._config.opacity_threshold:
            # Current position along ray
            pos = Vec3(
                origin.x + t * cone.direction.x,
                origin.y + t * cone.direction.y,
                origin.z + t * cone.direction.z,
            )

            # Compute cone diameter at this distance
            diameter = 2.0 * t * tan_aperture

            # Select mip level based on diameter
            mip_level = self._compute_mip_level(diameter)

            # Sample voxel at computed mip level
            sample = self.sample_mip(pos, mip_level)

            # Front-to-back compositing
            weight = 1.0 - accumulated_alpha
            accumulated_rgb += weight * sample.radiance
            accumulated_alpha += weight * sample.opacity

            # Exponential step
            t *= self._config.step_multiplier
            steps += 1

        return ConeTraceResult(
            radiance=accumulated_rgb,
            opacity=accumulated_alpha,
            steps=steps,
            distance=t,
            hit_solid=accumulated_alpha >= self._config.opacity_threshold,
        )

    def _compute_mip_level(self, diameter: float) -> float:
        """Compute mip level for cone diameter.

        Args:
            diameter: Cone diameter in voxel units

        Returns:
            Mip level (float for interpolation, clamped to valid range)
        """
        if diameter <= self._voxel_size:
            return 0.0

        mip = math.log2(diameter / self._voxel_size)
        return min(mip, float(self._max_mip))

    def sample_mip(self, position: Vec3, mip_level: float) -> VoxelData:
        """Sample voxel data at a position and mip level.

        Args:
            position: Position in voxel coordinates (mip 0 scale)
            mip_level: Mip level (float for trilinear)

        Returns:
            Sampled VoxelData
        """
        # Get base resolution
        base_res = self._mip_chain.base_resolution.value

        # Convert position to normalized coordinates
        u = position.x / base_res
        v = position.y / base_res
        w = position.z / base_res

        # Clamp to valid range
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        w = max(0.0, min(1.0, w))

        # Check bounds
        if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0 or w < 0.0 or w > 1.0:
            return VoxelData.empty()

        if self._config.trilinear_sampling:
            return self._sample_trilinear_mip(u, v, w, mip_level)
        else:
            return self._sample_nearest_mip(u, v, w, mip_level)

    def _sample_trilinear_mip(
        self, u: float, v: float, w: float, mip_level: float
    ) -> VoxelData:
        """Sample with trilinear interpolation across mip levels.

        Args:
            u, v, w: Normalized coordinates [0, 1]
            mip_level: Floating-point mip level

        Returns:
            Trilinearly interpolated VoxelData
        """
        mip_lo = int(mip_level)
        mip_hi = min(mip_lo + 1, self._max_mip)
        mip_frac = mip_level - mip_lo

        # Sample both mip levels
        sample_lo = self._mip_chain.sample_trilinear(u, v, w, mip_lo)
        sample_hi = self._mip_chain.sample_trilinear(u, v, w, mip_hi)

        # Lerp between mip levels
        radiance = sample_lo.radiance * (1.0 - mip_frac) + sample_hi.radiance * mip_frac
        opacity = sample_lo.opacity * (1.0 - mip_frac) + sample_hi.opacity * mip_frac

        return VoxelData(radiance.astype(np.float32), opacity)

    def _sample_nearest_mip(
        self, u: float, v: float, w: float, mip_level: float
    ) -> VoxelData:
        """Sample with nearest-neighbor at rounded mip level.

        Args:
            u, v, w: Normalized coordinates [0, 1]
            mip_level: Mip level (rounded to nearest)

        Returns:
            Sampled VoxelData
        """
        mip = int(round(mip_level))
        mip = max(0, min(mip, self._max_mip))

        level = self._mip_chain.get_level(mip)
        res = level.resolution

        x = int(u * (res - 1))
        y = int(v * (res - 1))
        z = int(w * (res - 1))

        x = max(0, min(res - 1, x))
        y = max(0, min(res - 1, y))
        z = max(0, min(res - 1, z))

        return level.get_voxel(x, y, z)

    @staticmethod
    def composite(
        samples: Sequence[tuple[VoxelData, float]],
    ) -> tuple[NDArray[np.float32], float]:
        """Composite multiple samples front-to-back.

        Args:
            samples: List of (VoxelData, weight) tuples ordered front-to-back

        Returns:
            Tuple of (accumulated_rgb, accumulated_alpha)
        """
        accumulated_rgb = np.zeros(3, dtype=np.float32)
        accumulated_alpha = 0.0

        for voxel, weight in samples:
            if accumulated_alpha >= 0.99:
                break

            visibility = (1.0 - accumulated_alpha) * weight
            accumulated_rgb += visibility * voxel.radiance
            accumulated_alpha += visibility * voxel.opacity

        return accumulated_rgb, accumulated_alpha


# ============================================================================
# Voxel GI Result
# ============================================================================


@dataclass
class VoxelGIResult:
    """Result of complete voxel GI evaluation.

    Attributes:
        diffuse_irradiance: Accumulated diffuse indirect light (RGB)
        specular_radiance: Accumulated specular indirect light (RGB)
        ambient_occlusion: Estimated AO from accumulated opacity
        confidence: Quality confidence based on valid samples
        diffuse_cone_results: Individual diffuse cone results
        specular_cone_results: Individual specular cone results
    """

    diffuse_irradiance: NDArray[np.float32]   # Shape (3,)
    specular_radiance: NDArray[np.float32]    # Shape (3,)
    ambient_occlusion: float
    confidence: float
    diffuse_cone_results: list[ConeTraceResult] = field(default_factory=list)
    specular_cone_results: list[ConeTraceResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate array shapes."""
        self.diffuse_irradiance = np.asarray(self.diffuse_irradiance, dtype=np.float32)
        self.specular_radiance = np.asarray(self.specular_radiance, dtype=np.float32)
        if self.diffuse_irradiance.shape != (3,):
            raise ValueError(f"Expected diffuse shape (3,), got {self.diffuse_irradiance.shape}")
        if self.specular_radiance.shape != (3,):
            raise ValueError(f"Expected specular shape (3,), got {self.specular_radiance.shape}")

    @classmethod
    def empty(cls) -> VoxelGIResult:
        """Create empty GI result."""
        return cls(
            diffuse_irradiance=np.zeros(3, dtype=np.float32),
            specular_radiance=np.zeros(3, dtype=np.float32),
            ambient_occlusion=0.0,
            confidence=0.0,
        )

    def total_indirect(self) -> NDArray[np.float32]:
        """Get total indirect illumination (diffuse + specular).

        Returns:
            Combined RGB indirect light
        """
        return self.diffuse_irradiance + self.specular_radiance

    def diffuse_luminance(self) -> float:
        """Luminance of diffuse component."""
        return float(
            0.2126 * self.diffuse_irradiance[0] +
            0.7152 * self.diffuse_irradiance[1] +
            0.0722 * self.diffuse_irradiance[2]
        )

    def specular_luminance(self) -> float:
        """Luminance of specular component."""
        return float(
            0.2126 * self.specular_radiance[0] +
            0.7152 * self.specular_radiance[1] +
            0.0722 * self.specular_radiance[2]
        )


# ============================================================================
# Voxel GI Pass
# ============================================================================


@dataclass
class VoxelGIConfig:
    """Configuration for voxel GI pass.

    Attributes:
        diffuse_cone_count: Number of diffuse cones
        specular_cone_count: Number of specular cones
        diffuse_aperture: Diffuse cone half-angle
        max_trace_distance: Maximum trace distance
        start_offset: Surface offset
        trace_diffuse: Enable diffuse tracing
        trace_specular: Enable specular tracing
        ao_from_opacity: Derive AO from traced opacity
    """

    diffuse_cone_count: int = DEFAULT_DIFFUSE_CONES
    specular_cone_count: int = DEFAULT_SPECULAR_CONES
    diffuse_aperture: float = DIFFUSE_APERTURE_DEFAULT
    max_trace_distance: float = DEFAULT_MAX_DISTANCE
    start_offset: float = DEFAULT_START_OFFSET
    trace_diffuse: bool = True
    trace_specular: bool = True
    ao_from_opacity: bool = True


class VoxelGIPass:
    """Full-screen voxel cone tracing GI pass.

    Integrates diffuse and specular cone tracing for complete indirect
    illumination. Uses G-buffer data for surface properties.

    Typical usage:
        ```python
        gi_pass = VoxelGIPass(mip_chain, config)
        for pixel in screen:
            gi = gi_pass.evaluate_pixel(
                position=gbuffer.position[pixel],
                normal=gbuffer.normal[pixel],
                view_dir=gbuffer.view_dir[pixel],
                roughness=gbuffer.roughness[pixel],
            )
            output[pixel] = direct_light + gi.total_indirect()
        ```

    Attributes:
        mip_chain: Voxel mip chain
        config: GI pass configuration
        tracer: Internal cone tracer
        diffuse_dist: Diffuse cone distribution
        specular_dist: Specular cone distribution
    """

    __slots__ = (
        "_mip_chain",
        "_config",
        "_tracer",
        "_diffuse_dist",
        "_specular_dist",
        "_voxel_size",
        "_gi_buffer",
    )

    def __init__(
        self,
        mip_chain: VoxelMipChain,
        config: Optional[VoxelGIConfig] = None,
        voxel_size: float = 1.0,
    ) -> None:
        """Initialize GI pass.

        Args:
            mip_chain: Voxel data for tracing
            config: GI configuration
            voxel_size: World-space voxel size
        """
        self._mip_chain = mip_chain
        self._config = config or VoxelGIConfig()
        self._voxel_size = voxel_size

        # Create cone tracer
        self._tracer = VoxelConeTracer(
            mip_chain,
            ConeTracerConfig(
                opacity_threshold=DEFAULT_OPACITY_THRESHOLD,
                trilinear_sampling=True,
            ),
            voxel_size,
        )

        # Create cone distributions
        self._diffuse_dist = DiffuseConeDistribution(
            cone_count=self._config.diffuse_cone_count,
            aperture=self._config.diffuse_aperture,
        )
        self._specular_dist = SpecularConeDistribution(
            cone_count=self._config.specular_cone_count,
        )

        # GI buffer (None until execute called)
        self._gi_buffer: Optional[NDArray[np.float32]] = None

    @property
    def diffuse_cone_count(self) -> int:
        """Number of diffuse cones."""
        return self._config.diffuse_cone_count

    @property
    def specular_cone_count(self) -> int:
        """Number of specular cones."""
        return self._config.specular_cone_count

    def evaluate_pixel(
        self,
        position: Vec3,
        normal: Vec3,
        view_dir: Vec3,
        roughness: float = 0.5,
    ) -> VoxelGIResult:
        """Evaluate GI for a single pixel.

        Args:
            position: World-space position (in voxel coordinates)
            normal: Surface normal (will be normalized)
            view_dir: View direction (from surface toward camera)
            roughness: Surface roughness (0-1)

        Returns:
            VoxelGIResult with diffuse and specular indirect light
        """
        # Accumulate diffuse
        diffuse_rgb = np.zeros(3, dtype=np.float32)
        diffuse_opacity = 0.0
        diffuse_results = []

        if self._config.trace_diffuse:
            # Get diffuse cones in world space
            diffuse_cones = self._diffuse_dist.transform_to_surface(
                normal,
                max_distance=self._config.max_trace_distance,
                start_offset=self._config.start_offset,
            )

            # Trace each cone and accumulate
            total_weight = 0.0
            for cone in diffuse_cones:
                result = self._tracer.trace_cone(position, cone)
                diffuse_results.append(result)

                # Cosine weighting
                cos_theta = max(0.0, normal.dot(cone.direction))
                weight = cos_theta * cone.solid_angle

                diffuse_rgb += result.radiance * weight
                diffuse_opacity += result.opacity * weight
                total_weight += weight

            # Normalize by total weight
            if total_weight > EPSILON:
                diffuse_rgb /= total_weight
                diffuse_opacity /= total_weight

            # Apply PI factor for irradiance
            diffuse_rgb /= math.pi

        # Accumulate specular
        specular_rgb = np.zeros(3, dtype=np.float32)
        specular_results = []

        if self._config.trace_specular:
            # Compute reflection direction
            n_len = normal.length()
            if n_len > EPSILON:
                n = Vec3(normal.x / n_len, normal.y / n_len, normal.z / n_len)
            else:
                n = Vec3(0.0, 1.0, 0.0)

            v_len = view_dir.length()
            if v_len > EPSILON:
                v = Vec3(view_dir.x / v_len, view_dir.y / v_len, view_dir.z / v_len)
            else:
                v = Vec3(0.0, 0.0, 1.0)

            reflect_dir = v.reflect(n) * -1.0  # Reflect away from camera

            # Get specular cones
            specular_cones = self._specular_dist.get_cones(
                reflect_dir,
                roughness,
                max_distance=self._config.max_trace_distance,
                start_offset=self._config.start_offset,
            )

            # Trace and average
            for cone in specular_cones:
                result = self._tracer.trace_cone(position, cone)
                specular_results.append(result)
                specular_rgb += result.radiance

            if len(specular_cones) > 0:
                specular_rgb /= len(specular_cones)

        # Compute AO from diffuse opacity
        ao = diffuse_opacity if self._config.ao_from_opacity else 0.0

        # Confidence based on how many cones hit something
        total_cones = len(diffuse_results) + len(specular_results)
        valid_cones = sum(1 for r in diffuse_results if r.opacity > 0.01)
        valid_cones += sum(1 for r in specular_results if r.opacity > 0.01)
        confidence = valid_cones / max(1, total_cones)

        return VoxelGIResult(
            diffuse_irradiance=diffuse_rgb,
            specular_radiance=specular_rgb,
            ambient_occlusion=ao,
            confidence=confidence,
            diffuse_cone_results=diffuse_results,
            specular_cone_results=specular_results,
        )

    def execute(
        self,
        width: int,
        height: int,
        position_buffer: NDArray[np.float32],
        normal_buffer: NDArray[np.float32],
        view_dir_buffer: NDArray[np.float32],
        roughness_buffer: NDArray[np.float32],
    ) -> None:
        """Execute full-screen GI pass.

        Args:
            width: Screen width
            height: Screen height
            position_buffer: Shape (height, width, 3) positions in voxel coords
            normal_buffer: Shape (height, width, 3) normals
            view_dir_buffer: Shape (height, width, 3) view directions
            roughness_buffer: Shape (height, width) roughness values
        """
        # Allocate output buffer
        # Layout: RGBA per pixel (RGB = diffuse+specular, A = AO)
        self._gi_buffer = np.zeros((height, width, 4), dtype=np.float32)

        for y in range(height):
            for x in range(width):
                position = Vec3(
                    float(position_buffer[y, x, 0]),
                    float(position_buffer[y, x, 1]),
                    float(position_buffer[y, x, 2]),
                )
                normal = Vec3(
                    float(normal_buffer[y, x, 0]),
                    float(normal_buffer[y, x, 1]),
                    float(normal_buffer[y, x, 2]),
                )
                view_dir = Vec3(
                    float(view_dir_buffer[y, x, 0]),
                    float(view_dir_buffer[y, x, 1]),
                    float(view_dir_buffer[y, x, 2]),
                )
                roughness = float(roughness_buffer[y, x])

                gi = self.evaluate_pixel(position, normal, view_dir, roughness)

                # Store result
                total = gi.total_indirect()
                self._gi_buffer[y, x, 0] = total[0]
                self._gi_buffer[y, x, 1] = total[1]
                self._gi_buffer[y, x, 2] = total[2]
                self._gi_buffer[y, x, 3] = gi.ambient_occlusion

    def get_gi_buffer(self) -> Optional[NDArray[np.float32]]:
        """Get the computed GI buffer.

        Returns:
            Shape (height, width, 4) buffer or None if not executed
        """
        return self._gi_buffer


# ============================================================================
# Utility Functions
# ============================================================================


def _orthogonal_vector(v: Vec3) -> Vec3:
    """Compute a vector orthogonal to v.

    Args:
        v: Input vector (should be normalized)

    Returns:
        Normalized vector orthogonal to v
    """
    if abs(v.x) < abs(v.y):
        ortho = Vec3(0.0, -v.z, v.y)
    else:
        ortho = Vec3(-v.z, 0.0, v.x)

    length = ortho.length()
    if length < EPSILON:
        # Fallback for degenerate case
        if abs(v.x) < 0.9:
            return Vec3(1.0, 0.0, 0.0)
        else:
            return Vec3(0.0, 1.0, 0.0)

    return Vec3(ortho.x / length, ortho.y / length, ortho.z / length)


def estimate_trace_time_ms(
    resolution: int,
    cone_count: int,
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> float:
    """Estimate cone tracing time in milliseconds.

    Based on empirical measurements on typical GPUs.

    Args:
        resolution: Voxel grid resolution
        cone_count: Total cones per pixel
        screen_width: Screen width
        screen_height: Screen height

    Returns:
        Estimated time in milliseconds
    """
    # Base cost per cone per pixel (nanoseconds)
    base_cost_ns = {
        64: 0.5,
        128: 1.0,
        256: 2.0,
    }

    cost = base_cost_ns.get(resolution, 2.0)
    total_pixels = screen_width * screen_height
    total_ns = cost * cone_count * total_pixels

    return total_ns / 1_000_000


def create_test_gi_scene(
    resolution: int = 128,
    scene_type: str = "cornell_box",
) -> tuple[VoxelMipChain, NDArray[np.float32], NDArray[np.float32]]:
    """Create a test scene for GI validation.

    Args:
        resolution: Voxel resolution
        scene_type: Scene type ("cornell_box", "sphere", "empty")

    Returns:
        Tuple of (mip_chain, positions_buffer, normals_buffer)
    """
    from engine.rendering.gi.voxel_mipchain import create_test_voxel_pattern

    if scene_type == "cornell_box":
        # Create Cornell box-like pattern
        mip_chain = create_test_voxel_pattern(resolution, "cube")
    elif scene_type == "sphere":
        mip_chain = create_test_voxel_pattern(resolution, "sphere")
    else:
        # Empty scene
        mip_chain = VoxelMipChain(
            base_resolution=MipResolution.from_size(resolution)
        )

    # Create simple test buffers (small size for testing)
    test_res = 8
    positions = np.zeros((test_res, test_res, 3), dtype=np.float32)
    normals = np.zeros((test_res, test_res, 3), dtype=np.float32)

    # Fill with test data
    for y in range(test_res):
        for x in range(test_res):
            # Position in center of voxel grid, looking up
            positions[y, x] = [resolution / 2, resolution / 2, resolution / 4]
            normals[y, x] = [0.0, 0.0, 1.0]

    return mip_chain, positions, normals


# ============================================================================
# WGSL Shader Generation
# ============================================================================


def generate_voxel_cone_trace_wgsl() -> str:
    """Generate shared WGSL cone tracing utilities.

    Returns:
        WGSL code for voxel_cone_trace.wgsl
    """
    return '''// Voxel Cone Tracing Utilities
// T-GIR-P7.3: Shared cone trace functions for VXGI

// Constants
const EPSILON: f32 = 1e-6;
const PI: f32 = 3.14159265359;
const OPACITY_THRESHOLD: f32 = 0.99;

// Cone configuration
struct ConeConfig {
    direction: vec3<f32>,
    aperture: f32,          // Half-angle in radians
    max_distance: f32,
    start_offset: f32,
}

// Cone trace result
struct ConeTraceResult {
    radiance: vec3<f32>,
    opacity: f32,
    steps: u32,
    distance: f32,
}

// Compute mip level from cone diameter
fn compute_mip_level(diameter: f32, voxel_size: f32, max_mip: f32) -> f32 {
    if (diameter <= voxel_size) {
        return 0.0;
    }
    return min(log2(diameter / voxel_size), max_mip);
}

// Trace a single cone through voxel grid
fn trace_cone(
    voxel_tex: texture_3d<f32>,
    voxel_sampler: sampler,
    origin: vec3<f32>,
    cone: ConeConfig,
    voxel_size: f32,
    max_mip: f32,
    step_multiplier: f32,
) -> ConeTraceResult {
    var result: ConeTraceResult;
    result.radiance = vec3<f32>(0.0);
    result.opacity = 0.0;
    result.steps = 0u;

    var t = cone.start_offset;
    let tan_aperture = tan(cone.aperture);

    // Grid dimensions for normalization
    let grid_size = f32(textureDimensions(voxel_tex, 0).x);

    while (t < cone.max_distance && result.opacity < OPACITY_THRESHOLD) {
        // Current position
        let pos = origin + t * cone.direction;

        // Normalized UVW
        let uvw = pos / grid_size;

        // Check bounds
        if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
            break;
        }

        // Compute cone diameter and mip level
        let diameter = 2.0 * t * tan_aperture;
        let mip = compute_mip_level(diameter, voxel_size, max_mip);

        // Sample voxel
        let sample = textureSampleLevel(voxel_tex, voxel_sampler, uvw, mip);

        // Front-to-back compositing
        let weight = 1.0 - result.opacity;
        result.radiance += weight * sample.rgb;
        result.opacity += weight * sample.a;

        // Exponential step
        t *= step_multiplier;
        result.steps += 1u;
    }

    result.distance = t;
    return result;
}

// Generate cosine-weighted hemisphere direction
fn hemisphere_direction(index: u32, total: u32, normal: vec3<f32>) -> vec3<f32> {
    let golden_ratio = 1.6180339887;
    let golden_angle = 2.399963229728653;  // 2*PI / (golden_ratio^2)

    let i = f32(index);
    let n = f32(total);

    // Cosine-weighted elevation
    let t = (i + 0.5) / n;
    let phi = 1.0 - t;
    let theta = golden_angle * i;

    // Spherical to Cartesian (Z-up)
    let sin_phi = sqrt(1.0 - phi * phi);
    let local = vec3<f32>(
        sin_phi * cos(theta),
        sin_phi * sin(theta),
        phi
    );

    // Build tangent frame from normal
    var tangent: vec3<f32>;
    if (abs(normal.x) < abs(normal.y)) {
        tangent = normalize(vec3<f32>(0.0, -normal.z, normal.y));
    } else {
        tangent = normalize(vec3<f32>(-normal.z, 0.0, normal.x));
    }
    let bitangent = cross(normal, tangent);

    // Transform to world space
    return local.x * tangent + local.y * bitangent + local.z * normal;
}

// Trace diffuse cones and accumulate
fn trace_diffuse_cones(
    voxel_tex: texture_3d<f32>,
    voxel_sampler: sampler,
    origin: vec3<f32>,
    normal: vec3<f32>,
    cone_count: u32,
    aperture: f32,
    max_distance: f32,
    start_offset: f32,
    voxel_size: f32,
    max_mip: f32,
) -> vec3<f32> {
    var accumulated = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var i = 0u; i < cone_count; i++) {
        let direction = hemisphere_direction(i, cone_count, normal);

        var cone: ConeConfig;
        cone.direction = direction;
        cone.aperture = aperture;
        cone.max_distance = max_distance;
        cone.start_offset = start_offset;

        let result = trace_cone(
            voxel_tex, voxel_sampler,
            origin, cone,
            voxel_size, max_mip, 1.2
        );

        // Cosine weighting
        let cos_theta = max(0.0, dot(normal, direction));
        let weight = cos_theta;

        accumulated += result.radiance * weight;
        total_weight += weight;
    }

    if (total_weight > EPSILON) {
        accumulated /= total_weight;
    }

    // Apply PI factor for irradiance
    return accumulated / PI;
}

// Trace specular cone
fn trace_specular_cone(
    voxel_tex: texture_3d<f32>,
    voxel_sampler: sampler,
    origin: vec3<f32>,
    reflection: vec3<f32>,
    roughness: f32,
    max_distance: f32,
    start_offset: f32,
    voxel_size: f32,
    max_mip: f32,
) -> vec3<f32> {
    // Aperture from roughness
    let alpha = roughness * roughness;
    let min_aperture = 0.04363;  // ~2.5 degrees
    let max_aperture = 0.2618;   // ~15 degrees
    let aperture = min_aperture + alpha * (max_aperture - min_aperture);

    var cone: ConeConfig;
    cone.direction = reflection;
    cone.aperture = aperture;
    cone.max_distance = max_distance;
    cone.start_offset = start_offset;

    let result = trace_cone(
        voxel_tex, voxel_sampler,
        origin, cone,
        voxel_size, max_mip, 1.2
    );

    return result.radiance;
}
'''


def generate_voxel_cone_trace_compute_wgsl() -> str:
    """Generate compute shader for full-screen cone tracing.

    Returns:
        WGSL code for voxel_cone_trace.comp.wgsl
    """
    return '''// Voxel Cone Tracing Compute Shader
// T-GIR-P7.3: Full-screen GI pass

// Include shared utilities
// (In practice, these would be in a separate file and #include'd)

struct GIUniforms {
    screen_size: vec2<u32>,
    voxel_size: f32,
    max_mip: f32,
    diffuse_cone_count: u32,
    diffuse_aperture: f32,
    max_trace_distance: f32,
    start_offset: f32,
    trace_diffuse: u32,
    trace_specular: u32,
    _pad0: u32,
    _pad1: u32,
}

// Input G-buffer textures
@group(0) @binding(0) var position_tex: texture_2d<f32>;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var view_dir_tex: texture_2d<f32>;
@group(0) @binding(3) var roughness_tex: texture_2d<f32>;

// Voxel data
@group(1) @binding(0) var voxel_tex: texture_3d<f32>;
@group(1) @binding(1) var voxel_sampler: sampler;

// Output
@group(2) @binding(0) var gi_output: texture_storage_2d<rgba16float, write>;

// Uniforms
@group(3) @binding(0) var<uniform> uniforms: GIUniforms;

// Constants (duplicated from shared utilities)
const EPSILON: f32 = 1e-6;
const PI: f32 = 3.14159265359;
const OPACITY_THRESHOLD: f32 = 0.99;

struct ConeConfig {
    direction: vec3<f32>,
    aperture: f32,
    max_distance: f32,
    start_offset: f32,
}

struct ConeTraceResult {
    radiance: vec3<f32>,
    opacity: f32,
}

fn compute_mip_level(diameter: f32, voxel_size: f32, max_mip: f32) -> f32 {
    if (diameter <= voxel_size) { return 0.0; }
    return min(log2(diameter / voxel_size), max_mip);
}

fn trace_cone_impl(
    origin: vec3<f32>,
    cone: ConeConfig,
) -> ConeTraceResult {
    var result: ConeTraceResult;
    result.radiance = vec3<f32>(0.0);
    result.opacity = 0.0;

    var t = cone.start_offset;
    let tan_aperture = tan(cone.aperture);
    let grid_size = f32(textureDimensions(voxel_tex, 0).x);

    for (var step = 0u; step < 64u; step++) {
        if (t >= cone.max_distance || result.opacity >= OPACITY_THRESHOLD) {
            break;
        }

        let pos = origin + t * cone.direction;
        let uvw = pos / grid_size;

        if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
            break;
        }

        let diameter = 2.0 * t * tan_aperture;
        let mip = compute_mip_level(diameter, uniforms.voxel_size, uniforms.max_mip);
        let sample = textureSampleLevel(voxel_tex, voxel_sampler, uvw, mip);

        let weight = 1.0 - result.opacity;
        result.radiance += weight * sample.rgb;
        result.opacity += weight * sample.a;

        t *= 1.2;
    }

    return result;
}

fn hemisphere_dir(index: u32, total: u32, normal: vec3<f32>) -> vec3<f32> {
    let golden_angle = 2.399963229728653;
    let i = f32(index);
    let n = f32(total);

    let t = (i + 0.5) / n;
    let phi = 1.0 - t;
    let theta = golden_angle * i;

    let sin_phi = sqrt(1.0 - phi * phi);
    let local = vec3<f32>(sin_phi * cos(theta), sin_phi * sin(theta), phi);

    var tangent: vec3<f32>;
    if (abs(normal.x) < abs(normal.y)) {
        tangent = normalize(vec3<f32>(0.0, -normal.z, normal.y));
    } else {
        tangent = normalize(vec3<f32>(-normal.z, 0.0, normal.x));
    }
    let bitangent = cross(normal, tangent);

    return local.x * tangent + local.y * bitangent + local.z * normal;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = gid.xy;

    // Bounds check
    if (any(pixel >= uniforms.screen_size)) {
        return;
    }

    // Load G-buffer
    let position = textureLoad(position_tex, pixel, 0).xyz;
    let normal = normalize(textureLoad(normal_tex, pixel, 0).xyz);
    let view_dir = normalize(textureLoad(view_dir_tex, pixel, 0).xyz);
    let roughness = textureLoad(roughness_tex, pixel, 0).r;

    var gi_result = vec4<f32>(0.0);

    // Diffuse tracing
    if (uniforms.trace_diffuse != 0u) {
        var diffuse_sum = vec3<f32>(0.0);
        var diffuse_opacity = 0.0;
        var total_weight = 0.0;

        for (var i = 0u; i < uniforms.diffuse_cone_count; i++) {
            let direction = hemisphere_dir(i, uniforms.diffuse_cone_count, normal);

            var cone: ConeConfig;
            cone.direction = direction;
            cone.aperture = uniforms.diffuse_aperture;
            cone.max_distance = uniforms.max_trace_distance;
            cone.start_offset = uniforms.start_offset;

            let result = trace_cone_impl(position, cone);

            let cos_theta = max(0.0, dot(normal, direction));
            diffuse_sum += result.radiance * cos_theta;
            diffuse_opacity += result.opacity * cos_theta;
            total_weight += cos_theta;
        }

        if (total_weight > EPSILON) {
            diffuse_sum /= total_weight;
            diffuse_opacity /= total_weight;
        }

        gi_result.rgb += diffuse_sum / PI;
        gi_result.a = diffuse_opacity;  // AO
    }

    // Specular tracing
    if (uniforms.trace_specular != 0u) {
        let reflection = reflect(-view_dir, normal);

        let alpha = roughness * roughness;
        let min_aperture = 0.04363;
        let max_aperture = 0.2618;
        let aperture = min_aperture + alpha * (max_aperture - min_aperture);

        var cone: ConeConfig;
        cone.direction = reflection;
        cone.aperture = aperture;
        cone.max_distance = uniforms.max_trace_distance;
        cone.start_offset = uniforms.start_offset;

        let result = trace_cone_impl(position, cone);
        gi_result.rgb += result.radiance;
    }

    textureStore(gi_output, pixel, gi_result);
}
'''


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Constants
    "DIFFUSE_APERTURE_MIN",
    "DIFFUSE_APERTURE_MAX",
    "DIFFUSE_APERTURE_DEFAULT",
    "SPECULAR_APERTURE_MIN",
    "SPECULAR_APERTURE_MAX",
    "MIN_DIFFUSE_CONES",
    "MAX_DIFFUSE_CONES",
    "DEFAULT_DIFFUSE_CONES",
    "MIN_SPECULAR_CONES",
    "MAX_SPECULAR_CONES",
    "DEFAULT_SPECULAR_CONES",
    "DEFAULT_START_OFFSET",
    "DEFAULT_MAX_DISTANCE",
    "DEFAULT_STEP_MULTIPLIER",
    "DEFAULT_OPACITY_THRESHOLD",
    # Config classes
    "ConeConfig",
    "ConeTracerConfig",
    "VoxelGIConfig",
    # Cone distributions
    "DiffuseConeDistribution",
    "SpecularConeDistribution",
    # Results
    "ConeTraceResult",
    "VoxelGIResult",
    # Core classes
    "VoxelConeTracer",
    "VoxelGIPass",
    # Utilities
    "estimate_trace_time_ms",
    "create_test_gi_scene",
    # WGSL generation
    "generate_voxel_cone_trace_wgsl",
    "generate_voxel_cone_trace_compute_wgsl",
]

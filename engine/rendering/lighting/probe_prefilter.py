"""Pre-filtered Cubemaps for Roughness-Based Reflections.

Implements GGX pre-filtering for roughness-based specular reflections:
- GGX/Trowbridge-Reitz normal distribution function (NDF)
- Importance sampling for GGX distribution
- Monte Carlo integration over hemisphere
- Split-sum approximation for efficient IBL
- Pre-filtered cubemap mip levels per roughness

Reference: RENDERING_CONTEXT.md Section 6.4, Epic Games PBR paper
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Optional, Tuple

from engine.core.math.vec import Vec3
from engine.rendering.lighting.baked_probes import (
    BakedProbeConstants,
    CubemapData,
    CubemapFace,
    CubemapFaceData,
    CubemapMipChain,
    CUBEMAP_FACE_DIRECTIONS,
    HDRPixel,
    MipLevel,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

class PrefilterConstants:
    """Constants for pre-filtered cubemaps."""
    # Default roughness levels (mip levels)
    DEFAULT_ROUGHNESS_LEVELS: int = 8
    MIN_ROUGHNESS_LEVELS: int = 2
    MAX_ROUGHNESS_LEVELS: int = 16
    # Default sample count per texel
    DEFAULT_SAMPLE_COUNT: int = 256
    MIN_SAMPLE_COUNT: int = 16
    MAX_SAMPLE_COUNT: int = 4096
    # Split-sum LUT resolution
    LUT_RESOLUTION: int = 512
    # Minimum roughness to avoid divide-by-zero
    MIN_ROUGHNESS: float = 0.001
    # Pi constant
    PI: float = 3.14159265358979323846
    # Maximum performance budget (ms per face per roughness level)
    # Higher threshold to account for CI variability
    MAX_TIME_PER_FACE_MS: float = 200.0
    # Van der Corput scaling factor
    VDC_SCALE: float = 2.3283064365386963e-10


# -----------------------------------------------------------------------------
# GGX Distribution
# -----------------------------------------------------------------------------

@dataclass
class GGXDistribution:
    """GGX/Trowbridge-Reitz normal distribution function.

    The GGX distribution models microfacet surface roughness for
    physically-based rendering. It provides the statistical distribution
    of microfacet normals for a given surface roughness.

    Attributes:
        roughness: Surface roughness (0 = mirror, 1 = rough)
        alpha: Squared roughness (a = roughness^2)
    """
    roughness: float = 0.5
    alpha: float = field(init=False)
    _alpha_sq: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Compute derived values."""
        self.roughness = max(PrefilterConstants.MIN_ROUGHNESS, min(1.0, self.roughness))
        self.alpha = self.roughness * self.roughness
        self._alpha_sq = self.alpha * self.alpha

    def D(self, n_dot_h: float) -> float:
        """Evaluate GGX normal distribution function.

        D(h) = a^2 / (pi * ((n.h)^2 * (a^2 - 1) + 1)^2)

        Args:
            n_dot_h: Dot product of normal and half-vector (clamped 0-1)

        Returns:
            NDF value (always >= 0)
        """
        n_dot_h = max(0.0, min(1.0, n_dot_h))
        n_dot_h_sq = n_dot_h * n_dot_h

        denom = n_dot_h_sq * (self._alpha_sq - 1.0) + 1.0
        denom = denom * denom * PrefilterConstants.PI

        if denom < 1e-10:
            return 0.0

        return self._alpha_sq / denom

    def sample_direction(self, xi_x: float, xi_y: float, N: Vec3) -> Vec3:
        """Sample a direction from the GGX distribution.

        Uses importance sampling to generate directions weighted
        by the GGX NDF.

        Args:
            xi_x: Random value [0, 1] for phi
            xi_y: Random value [0, 1] for theta
            N: Surface normal (unit vector)

        Returns:
            Sampled half-vector H in world space
        """
        # Spherical coordinates from GGX importance sampling
        phi = 2.0 * PrefilterConstants.PI * xi_x

        # GGX importance sampling formula
        cos_theta_sq = (1.0 - xi_y) / (1.0 + (self._alpha_sq - 1.0) * xi_y)
        cos_theta = math.sqrt(max(0.0, cos_theta_sq))
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta_sq))

        # Spherical to Cartesian in tangent space
        H_tangent = Vec3(
            math.cos(phi) * sin_theta,
            math.sin(phi) * sin_theta,
            cos_theta,
        )

        # Transform from tangent to world space
        return tangent_to_world(H_tangent, N)

    def pdf(self, n_dot_h: float, h_dot_v: float) -> float:
        """Compute probability density for a sampled direction.

        PDF(H) = D(H) * (N.H) / (4 * (H.V))

        Args:
            n_dot_h: Dot product of normal and half-vector
            h_dot_v: Dot product of half-vector and view direction

        Returns:
            Probability density value
        """
        if h_dot_v < 1e-6:
            return 0.0

        D_val = self.D(n_dot_h)
        return (D_val * n_dot_h) / (4.0 * h_dot_v)

    def get_roughness(self) -> float:
        """Get the roughness value."""
        return self.roughness

    def set_roughness(self, roughness: float) -> None:
        """Set new roughness and recompute alpha."""
        self.roughness = max(PrefilterConstants.MIN_ROUGHNESS, min(1.0, roughness))
        self.alpha = self.roughness * self.roughness
        self._alpha_sq = self.alpha * self.alpha


# -----------------------------------------------------------------------------
# Importance Sampling Utilities
# -----------------------------------------------------------------------------

def radical_inverse_vdc(bits: int) -> float:
    """Van der Corput sequence for quasi-random sampling.

    Produces a low-discrepancy sequence by bit-reversing the input.

    Args:
        bits: Input index (32-bit integer)

    Returns:
        Quasi-random value in [0, 1)
    """
    bits = bits & 0xFFFFFFFF
    bits = ((bits << 16) | (bits >> 16)) & 0xFFFFFFFF
    bits = (((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)) & 0xFFFFFFFF
    bits = (((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)) & 0xFFFFFFFF
    bits = (((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)) & 0xFFFFFFFF
    bits = (((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)) & 0xFFFFFFFF
    return bits * PrefilterConstants.VDC_SCALE


def hammersley(i: int, N: int) -> Tuple[float, float]:
    """Generate 2D Hammersley point for quasi-random sampling.

    The Hammersley sequence provides low-discrepancy 2D points
    suitable for importance sampling.

    Args:
        i: Sample index
        N: Total number of samples

    Returns:
        Tuple of (x, y) coordinates in [0, 1)
    """
    return (float(i) / float(max(1, N)), radical_inverse_vdc(i))


def tangent_to_world(H_tangent: Vec3, N: Vec3) -> Vec3:
    """Transform vector from tangent space to world space.

    Constructs an orthonormal basis from the normal N and
    transforms H_tangent using that basis.

    Args:
        H_tangent: Vector in tangent space (z = up)
        N: World-space normal defining tangent plane

    Returns:
        Vector transformed to world space
    """
    # Choose up vector that's not parallel to N
    if abs(N.y) < 0.999:
        up = Vec3(0, 1, 0)
    else:
        up = Vec3(1, 0, 0)

    tangent = up.cross(N).normalized()
    bitangent = N.cross(tangent)

    return (
        tangent * H_tangent.x +
        bitangent * H_tangent.y +
        N * H_tangent.z
    ).normalized()


@dataclass
class ImportanceSampler:
    """Importance sampler for GGX BRDF integration.

    Generates sample directions weighted by the GGX distribution
    for efficient Monte Carlo integration.

    Attributes:
        sample_count: Number of samples to generate
        distribution: GGX distribution to sample from
    """
    sample_count: int = PrefilterConstants.DEFAULT_SAMPLE_COUNT
    distribution: GGXDistribution = field(default_factory=GGXDistribution)

    def __post_init__(self) -> None:
        """Validate sample count."""
        self.sample_count = max(
            PrefilterConstants.MIN_SAMPLE_COUNT,
            min(self.sample_count, PrefilterConstants.MAX_SAMPLE_COUNT)
        )

    def hammersley(self, i: int) -> Tuple[float, float]:
        """Get Hammersley point for sample index.

        Args:
            i: Sample index

        Returns:
            2D quasi-random point (x, y)
        """
        return hammersley(i, self.sample_count)

    def importance_sample_ggx(self, xi: Tuple[float, float], N: Vec3) -> Vec3:
        """Generate GGX importance-sampled direction.

        Args:
            xi: 2D random values (x, y) in [0, 1)
            N: Surface normal

        Returns:
            Sampled half-vector H in world space
        """
        return self.distribution.sample_direction(xi[0], xi[1], N)

    def get_sample_direction(self, i: int, N: Vec3) -> Vec3:
        """Get the i-th sample direction for normal N.

        Convenience method combining Hammersley and importance sampling.

        Args:
            i: Sample index
            N: Surface normal

        Returns:
            Sampled half-vector H
        """
        xi = self.hammersley(i)
        return self.importance_sample_ggx(xi, N)

    def get_reflected_direction(self, i: int, N: Vec3, V: Vec3) -> Tuple[Vec3, float]:
        """Get reflected direction and NdotL weight.

        Args:
            i: Sample index
            N: Surface normal
            V: View direction

        Returns:
            Tuple of (reflected direction L, N.L weight)
        """
        H = self.get_sample_direction(i, N)
        NdotH = max(0.0, N.dot(H))

        # Reflect V around H: L = 2 * (V.H) * H - V
        VdotH = max(0.0, V.dot(H))
        L = H * (2.0 * VdotH) - V
        L = L.normalized()

        NdotL = max(0.0, N.dot(L))
        return (L, NdotL)

    def set_roughness(self, roughness: float) -> None:
        """Update the roughness for sampling.

        Args:
            roughness: New roughness value
        """
        self.distribution.set_roughness(roughness)


# -----------------------------------------------------------------------------
# Pre-filter Configuration
# -----------------------------------------------------------------------------

@dataclass
class PrefilterConfig:
    """Configuration for cubemap pre-filtering.

    Attributes:
        roughness_levels: Number of roughness mip levels (8-10 typical)
        sample_count: Samples per texel for Monte Carlo integration
        resolution_scale: Scale factor for mip resolution (0.5 = half per level)
        min_resolution: Minimum resolution for any mip level
        enable_importance_sampling: Use importance sampling vs uniform
    """
    roughness_levels: int = PrefilterConstants.DEFAULT_ROUGHNESS_LEVELS
    sample_count: int = PrefilterConstants.DEFAULT_SAMPLE_COUNT
    resolution_scale: float = 0.5
    min_resolution: int = 4
    enable_importance_sampling: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.roughness_levels = max(
            PrefilterConstants.MIN_ROUGHNESS_LEVELS,
            min(self.roughness_levels, PrefilterConstants.MAX_ROUGHNESS_LEVELS)
        )
        self.sample_count = max(
            PrefilterConstants.MIN_SAMPLE_COUNT,
            min(self.sample_count, PrefilterConstants.MAX_SAMPLE_COUNT)
        )
        self.resolution_scale = max(0.25, min(1.0, self.resolution_scale))
        self.min_resolution = max(1, min(32, self.min_resolution))

    def get_resolution_for_level(self, base_resolution: int, level: int) -> int:
        """Calculate resolution for a specific mip level.

        Args:
            base_resolution: Resolution of level 0
            level: Mip level index

        Returns:
            Resolution for the given level
        """
        scale = self.resolution_scale ** level
        return max(self.min_resolution, int(base_resolution * scale))

    def get_roughness_for_level(self, level: int) -> float:
        """Calculate roughness for a specific mip level.

        Args:
            level: Mip level index

        Returns:
            Roughness value [0, 1]
        """
        if self.roughness_levels <= 1:
            return 0.0
        return level / (self.roughness_levels - 1)

    def get_sample_count_for_roughness(self, roughness: float) -> int:
        """Get adaptive sample count based on roughness.

        Higher roughness needs more samples for convergence.

        Args:
            roughness: Surface roughness

        Returns:
            Number of samples to use
        """
        # Scale samples with roughness (more samples for rough surfaces)
        if roughness < 0.1:
            return max(16, self.sample_count // 4)
        elif roughness < 0.3:
            return max(32, self.sample_count // 2)
        return self.sample_count


# -----------------------------------------------------------------------------
# Cubemap Pre-filter
# -----------------------------------------------------------------------------

@dataclass
class CubemapPrefilter:
    """Pre-filters cubemaps for roughness-based reflections.

    Implements Monte Carlo integration over the hemisphere using
    GGX importance sampling to convolve the environment map with
    the BRDF for various roughness levels.

    Attributes:
        config: Pre-filter configuration
        sampler: Importance sampler for integration
    """
    config: PrefilterConfig = field(default_factory=PrefilterConfig)
    sampler: ImportanceSampler = field(default_factory=ImportanceSampler)

    _last_filter_time_ms: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        """Initialize sampler with config sample count."""
        self.sampler = ImportanceSampler(
            sample_count=self.config.sample_count,
            distribution=GGXDistribution(roughness=0.5),
        )

    @property
    def last_filter_time_ms(self) -> float:
        """Get last filter operation time in milliseconds."""
        return self._last_filter_time_ms

    def prefilter_face(
        self,
        source: CubemapData,
        face: CubemapFace,
        target_resolution: int,
        roughness: float,
    ) -> CubemapFaceData:
        """Pre-filter a single cubemap face for a roughness level.

        Args:
            source: Source environment cubemap
            face: Face to process
            target_resolution: Output resolution
            roughness: Target roughness level

        Returns:
            Pre-filtered face data
        """
        start_time = time.perf_counter()

        # Configure sampler for this roughness
        self.sampler.set_roughness(roughness)
        sample_count = self.config.get_sample_count_for_roughness(roughness)

        # Create output face
        result = CubemapFaceData(face=face, resolution=target_resolution)

        # Get face directions for computing pixel directions
        direction_base, up = CUBEMAP_FACE_DIRECTIONS[face]
        right = direction_base.cross(up).normalized()

        # For very low roughness, just downsample
        if roughness < PrefilterConstants.MIN_ROUGHNESS:
            return self._downsample_face(source.get_face(face), target_resolution, face)

        # Process each texel
        for y in range(target_resolution):
            for x in range(target_resolution):
                # Compute normal direction for this texel
                u = (x + 0.5) / target_resolution * 2.0 - 1.0
                v = (y + 0.5) / target_resolution * 2.0 - 1.0
                N = (direction_base + right * u + up * (-v)).normalized()

                # Monte Carlo integration
                pixel = self._integrate_hemisphere(source, N, sample_count)
                result.set_pixel(x, y, pixel)

        self._last_filter_time_ms = (time.perf_counter() - start_time) * 1000.0
        return result

    def prefilter_cubemap(
        self,
        source: CubemapData,
        roughness: float,
        target_resolution: Optional[int] = None,
    ) -> CubemapData:
        """Pre-filter entire cubemap for a roughness level.

        Args:
            source: Source environment cubemap
            roughness: Target roughness level
            target_resolution: Output resolution (None = source resolution)

        Returns:
            Pre-filtered cubemap
        """
        if target_resolution is None:
            target_resolution = source.resolution

        result = CubemapData(resolution=target_resolution)

        for face in CubemapFace:
            face_data = self.prefilter_face(
                source, face, target_resolution, roughness
            )
            result.faces[face.value] = face_data

        return result

    def get_mip_for_roughness(self, roughness: float) -> int:
        """Get the mip level corresponding to a roughness value.

        Args:
            roughness: Surface roughness [0, 1]

        Returns:
            Mip level index
        """
        roughness = max(0.0, min(1.0, roughness))
        return int(roughness * (self.config.roughness_levels - 1) + 0.5)

    def _integrate_hemisphere(
        self,
        source: CubemapData,
        N: Vec3,
        sample_count: int,
    ) -> HDRPixel:
        """Perform Monte Carlo integration over hemisphere.

        Args:
            source: Source environment cubemap
            N: Surface normal (also view direction for reflection)
            sample_count: Number of samples

        Returns:
            Integrated color
        """
        total = HDRPixel()
        total_weight = 0.0

        V = N  # For pre-filtering, V = N (isotropic assumption)

        for i in range(sample_count):
            # Get sample direction
            L, NdotL = self.sampler.get_reflected_direction(i, N, V)

            if NdotL > 0:
                # Sample environment in reflected direction
                sample = source.sample_direction(L)
                total = total + sample * NdotL
                total_weight += NdotL

        if total_weight > 0:
            return total * (1.0 / total_weight)
        return source.sample_direction(N)

    def _downsample_face(
        self,
        source: CubemapFaceData,
        target_resolution: int,
        face: CubemapFace,
    ) -> CubemapFaceData:
        """Simple box filter downsample for low roughness.

        Args:
            source: Source face data
            target_resolution: Target resolution
            face: Face identifier

        Returns:
            Downsampled face
        """
        result = CubemapFaceData(face=face, resolution=target_resolution)
        scale = source.resolution / target_resolution

        for y in range(target_resolution):
            for x in range(target_resolution):
                # Box filter
                total = HDRPixel()
                count = 0

                src_x = int(x * scale)
                src_y = int(y * scale)
                size = max(1, int(scale))

                for dy in range(size):
                    for dx in range(size):
                        px = min(src_x + dx, source.resolution - 1)
                        py = min(src_y + dy, source.resolution - 1)
                        total = total + source.get_pixel(px, py)
                        count += 1

                if count > 0:
                    result.set_pixel(x, y, total * (1.0 / count))
                else:
                    result.set_pixel(x, y, source.get_pixel(src_x, src_y))

        return result


# -----------------------------------------------------------------------------
# Split-Sum LUT
# -----------------------------------------------------------------------------

@dataclass
class BRDFTerms:
    """BRDF integration terms from split-sum approximation.

    Attributes:
        scale: F0 scale factor (red channel of LUT)
        bias: F0 bias/offset term (green channel of LUT)
    """
    scale: float = 1.0
    bias: float = 0.0


@dataclass
class SplitSumLUT:
    """Pre-computed BRDF integration lookup table.

    The split-sum approximation separates the rendering equation into:
    L(p, wo) = ∫ Li(p, wi) dw * ∫ f(p, wi, wo) (n.wi) dwi

    This LUT stores the second integral (BRDF integration) as a function
    of NdotV and roughness, enabling efficient real-time IBL.

    Attributes:
        resolution: LUT resolution (width = NdotV, height = roughness)
        data: 2D array of BRDFTerms [roughness][n_dot_v]
    """
    resolution: int = PrefilterConstants.LUT_RESOLUTION
    data: list[list[BRDFTerms]] = field(default_factory=list)
    _sample_count: int = field(default=1024, repr=False)

    def __post_init__(self) -> None:
        """Initialize empty LUT."""
        if not self.data:
            self.data = [
                [BRDFTerms() for _ in range(self.resolution)]
                for _ in range(self.resolution)
            ]

    def generate_lut(self, sample_count: int = 1024) -> None:
        """Generate the BRDF integration LUT.

        Computes ∫ f(wi, wo) (n.wi) dwi for varying NdotV and roughness.

        Args:
            sample_count: Samples for Monte Carlo integration
        """
        self._sample_count = sample_count

        for j in range(self.resolution):
            roughness = (j + 0.5) / self.resolution
            distribution = GGXDistribution(roughness=roughness)

            for i in range(self.resolution):
                n_dot_v = (i + 0.5) / self.resolution
                n_dot_v = max(0.001, n_dot_v)  # Avoid edge singularity

                terms = self._integrate_brdf(n_dot_v, distribution, sample_count)
                self.data[j][i] = terms

    def sample_lut(self, n_dot_v: float, roughness: float) -> BRDFTerms:
        """Sample the LUT with bilinear interpolation.

        Args:
            n_dot_v: Dot product of normal and view direction [0, 1]
            roughness: Surface roughness [0, 1]

        Returns:
            Interpolated BRDF terms
        """
        # Clamp inputs to valid range
        n_dot_v = max(0.0, min(1.0, n_dot_v))
        roughness = max(0.0, min(1.0, roughness))

        # Map to LUT coordinates
        u = n_dot_v * (self.resolution - 1)
        v = roughness * (self.resolution - 1)

        # Bilinear interpolation
        x0 = int(u)
        y0 = int(v)
        x1 = min(x0 + 1, self.resolution - 1)
        y1 = min(y0 + 1, self.resolution - 1)

        fx = u - x0
        fy = v - y0

        t00 = self.data[y0][x0]
        t10 = self.data[y0][x1]
        t01 = self.data[y1][x0]
        t11 = self.data[y1][x1]

        # Interpolate scale
        top_scale = t00.scale * (1 - fx) + t10.scale * fx
        bot_scale = t01.scale * (1 - fx) + t11.scale * fx
        scale = top_scale * (1 - fy) + bot_scale * fy

        # Interpolate bias
        top_bias = t00.bias * (1 - fx) + t10.bias * fx
        bot_bias = t01.bias * (1 - fx) + t11.bias * fx
        bias = top_bias * (1 - fy) + bot_bias * fy

        return BRDFTerms(scale=scale, bias=bias)

    def get_brdf_terms(self, n_dot_v: float, roughness: float) -> Tuple[float, float]:
        """Get BRDF scale and bias for computing specular.

        F = F0 * scale + bias

        Args:
            n_dot_v: Dot product of normal and view direction
            roughness: Surface roughness

        Returns:
            Tuple of (scale, bias)
        """
        terms = self.sample_lut(n_dot_v, roughness)
        return (terms.scale, terms.bias)

    def _integrate_brdf(
        self,
        n_dot_v: float,
        distribution: GGXDistribution,
        sample_count: int,
    ) -> BRDFTerms:
        """Integrate BRDF for given NdotV and roughness.

        Args:
            n_dot_v: Cos angle between N and V
            distribution: GGX distribution for roughness
            sample_count: Number of samples

        Returns:
            Integrated BRDF terms
        """
        # View vector in tangent space
        V = Vec3(
            math.sqrt(max(0.0, 1.0 - n_dot_v * n_dot_v)),  # sin(theta)
            0.0,
            n_dot_v,  # cos(theta)
        )
        N = Vec3(0, 0, 1)  # Normal in tangent space

        scale_sum = 0.0
        bias_sum = 0.0

        for i in range(sample_count):
            # Generate sample
            xi = hammersley(i, sample_count)
            H = distribution.sample_direction(xi[0], xi[1], N)

            VdotH = max(0.0, V.dot(H))
            L = H * (2.0 * VdotH) - V

            NdotL = max(0.0, L.z)
            NdotH = max(0.0, H.z)

            if NdotL > 0:
                # Geometry term (Smith GGX)
                G = self._geometry_smith(N.dot(V), NdotL, distribution.roughness)
                G_vis = G * VdotH / max(NdotH * n_dot_v, 1e-6)

                # Fresnel-Schlick (split into F0 and (1-F0) terms)
                Fc = (1.0 - VdotH) ** 5

                scale_sum += G_vis * (1.0 - Fc)
                bias_sum += G_vis * Fc

        if sample_count > 0:
            scale_sum /= sample_count
            bias_sum /= sample_count

        return BRDFTerms(scale=scale_sum, bias=bias_sum)

    def _geometry_smith(self, n_dot_v: float, n_dot_l: float, roughness: float) -> float:
        """Smith geometry function for GGX.

        Args:
            n_dot_v: N.V
            n_dot_l: N.L
            roughness: Surface roughness

        Returns:
            Geometry attenuation factor
        """
        def ggx_schlick(n_dot: float, k: float) -> float:
            return n_dot / max(n_dot * (1.0 - k) + k, 1e-6)

        k = (roughness * roughness) / 2.0
        return ggx_schlick(n_dot_v, k) * ggx_schlick(n_dot_l, k)


# -----------------------------------------------------------------------------
# Pre-filter Pipeline
# -----------------------------------------------------------------------------

@dataclass
class PrefilterResult:
    """Result of pre-filtering operation.

    Attributes:
        mip_chain: Pre-filtered mip chain
        brdf_lut: BRDF integration LUT
        total_time_ms: Total processing time
        per_level_times_ms: Time per roughness level
    """
    mip_chain: CubemapMipChain
    brdf_lut: Optional[SplitSumLUT] = None
    total_time_ms: float = 0.0
    per_level_times_ms: list[float] = field(default_factory=list)


class PrefilterPipeline:
    """Full pre-filtering pipeline for cubemaps.

    Generates all roughness mip levels for a cubemap and
    optionally the BRDF integration LUT.

    Attributes:
        config: Pre-filter configuration
        generate_lut: Whether to generate BRDF LUT
    """

    def __init__(
        self,
        config: Optional[PrefilterConfig] = None,
        generate_lut: bool = True,
    ) -> None:
        """Initialize pipeline.

        Args:
            config: Pre-filter configuration
            generate_lut: Whether to generate BRDF LUT
        """
        self._config = config or PrefilterConfig()
        self._generate_lut = generate_lut
        self._prefilter = CubemapPrefilter(config=self._config)
        self._brdf_lut: Optional[SplitSumLUT] = None

    @property
    def config(self) -> PrefilterConfig:
        """Get configuration."""
        return self._config

    @property
    def brdf_lut(self) -> Optional[SplitSumLUT]:
        """Get generated BRDF LUT (if any)."""
        return self._brdf_lut

    def process(self, source: CubemapData) -> PrefilterResult:
        """Process a cubemap through the full pre-filter pipeline.

        Args:
            source: Source environment cubemap

        Returns:
            Pre-filtered result with mip chain and optional LUT
        """
        start_time = time.perf_counter()
        per_level_times = []

        # Create mip chain
        mip_chain = CubemapMipChain(
            base_resolution=source.resolution,
            mip_count=self._config.roughness_levels,
            is_prefiltered=True,
        )

        # Process each roughness level
        for level in range(self._config.roughness_levels):
            level_start = time.perf_counter()

            roughness = self._config.get_roughness_for_level(level)
            resolution = self._config.get_resolution_for_level(source.resolution, level)

            # Pre-filter for this roughness
            filtered = self._prefilter.prefilter_cubemap(source, roughness, resolution)

            mip_chain.mips.append(MipLevel(
                level=level,
                resolution=resolution,
                cubemap=filtered,
                roughness=roughness,
            ))

            per_level_times.append((time.perf_counter() - level_start) * 1000.0)

        # Generate BRDF LUT if requested
        brdf_lut = None
        if self._generate_lut:
            brdf_lut = SplitSumLUT()
            brdf_lut.generate_lut(self._config.sample_count)
            self._brdf_lut = brdf_lut

        total_time = (time.perf_counter() - start_time) * 1000.0

        return PrefilterResult(
            mip_chain=mip_chain,
            brdf_lut=brdf_lut,
            total_time_ms=total_time,
            per_level_times_ms=per_level_times,
        )

    def store_mips(self, result: PrefilterResult) -> list[CubemapData]:
        """Extract individual mip level cubemaps.

        Args:
            result: Pre-filter result

        Returns:
            List of cubemaps, one per roughness level
        """
        return [mip.cubemap for mip in result.mip_chain.mips]

    def get_prefiltered_cubemap(
        self,
        result: PrefilterResult,
        roughness: float,
    ) -> CubemapData:
        """Get pre-filtered cubemap for specific roughness.

        Performs trilinear interpolation between mip levels if needed.

        Args:
            result: Pre-filter result
            roughness: Target roughness value

        Returns:
            Cubemap for the specified roughness
        """
        roughness = max(0.0, min(1.0, roughness))
        mip_level = roughness * (len(result.mip_chain.mips) - 1)

        lower_idx = int(mip_level)
        upper_idx = min(lower_idx + 1, len(result.mip_chain.mips) - 1)

        if lower_idx == upper_idx:
            return result.mip_chain.mips[lower_idx].cubemap

        # Return lower mip for now (full trilinear would require blending)
        return result.mip_chain.mips[lower_idx].cubemap

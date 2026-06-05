"""RT BRDF Importance Sampling for Reflections (T-GIR-P8.2).

This module implements GGX microfacet importance sampling for ray-traced reflections:
- GGXMicrofacetDistribution: GGX D, G1, G2, and PDF functions
- ImportanceSamplerGGX: Half-vector and reflection direction sampling
- TangentBasis: Orthonormal basis construction from normal
- BRDFEvaluator: Cook-Torrance BRDF with Schlick Fresnel
- SampleResult: Complete sampling result with PDF and weight
- RTBRDFSampler: Full sampling pipeline for RT reflections

The GGX microfacet model provides:
- D(H) = alpha^2 / (pi * ((N.H)^2 * (alpha^2 - 1) + 1)^2)
- G1 Smith shadowing function
- G2 height-correlated Smith masking-shadowing

References:
    - "Microfacet Models for Refraction" (Walter et al. 2007)
    - "Understanding the Masking-Shadowing Function" (Heitz 2014)
    - T-GIR-P8.2 BRDF Importance Sampling spec
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from engine.core.math.vec import Vec2, Vec3


# =============================================================================
# Constants
# =============================================================================

PI = math.pi
TWO_PI = 2.0 * PI
INV_PI = 1.0 / PI
EPSILON = 1e-7

# Minimum roughness to avoid singularities
MIN_ROUGHNESS = 0.001

# Maximum roughness (fully diffuse)
MAX_ROUGHNESS = 1.0


# =============================================================================
# GGX Microfacet Distribution
# =============================================================================


class GGXMicrofacetDistribution:
    """GGX (Trowbridge-Reitz) microfacet distribution.

    Implements the GGX normal distribution function and associated
    Smith masking-shadowing functions for physically-based rendering.

    The GGX distribution is defined as:
        D(H) = alpha^2 / (pi * ((N.H)^2 * (alpha^2 - 1) + 1)^2)

    where alpha = roughness^2.

    Usage:
        ggx = GGXMicrofacetDistribution(roughness=0.3)
        d_term = ggx.D(n_dot_h)
        g1_shadowing = ggx.G1(n_dot_v)
        g2_term = ggx.G2(n_dot_v, n_dot_l)
        pdf = ggx.pdf(n_dot_h, v_dot_h)
    """

    def __init__(self, roughness: float = 0.5) -> None:
        """Initialize GGX distribution.

        Args:
            roughness: Surface roughness [0, 1]. Clamped to [MIN_ROUGHNESS, 1].
        """
        self._roughness = max(MIN_ROUGHNESS, min(MAX_ROUGHNESS, roughness))
        self._alpha = self._roughness * self._roughness
        self._alpha_sq = self._alpha * self._alpha

    @property
    def roughness(self) -> float:
        """Get roughness value."""
        return self._roughness

    @roughness.setter
    def roughness(self, value: float) -> None:
        """Set roughness and update derived values."""
        self._roughness = max(MIN_ROUGHNESS, min(MAX_ROUGHNESS, value))
        self._alpha = self._roughness * self._roughness
        self._alpha_sq = self._alpha * self._alpha

    @property
    def alpha(self) -> float:
        """Get alpha (roughness^2)."""
        return self._alpha

    @property
    def alpha_squared(self) -> float:
        """Get alpha squared (roughness^4)."""
        return self._alpha_sq

    def D(self, n_dot_h: float) -> float:
        """Evaluate GGX normal distribution function.

        D(H) = alpha^2 / (pi * ((N.H)^2 * (alpha^2 - 1) + 1)^2)

        Args:
            n_dot_h: Dot product of normal and half-vector, clamped to [0, 1].

        Returns:
            Distribution value D(H).
        """
        n_dot_h = max(0.0, min(1.0, n_dot_h))
        n_dot_h_sq = n_dot_h * n_dot_h

        # Denominator: (N.H)^2 * (alpha^2 - 1) + 1
        denom = n_dot_h_sq * (self._alpha_sq - 1.0) + 1.0

        # Avoid division by zero
        if denom < EPSILON:
            return 0.0

        # D = alpha^2 / (pi * denom^2)
        return self._alpha_sq * INV_PI / (denom * denom)

    def Lambda(self, n_dot_v: float) -> float:
        """Compute Lambda function for Smith masking.

        Lambda(v) = (-1 + sqrt(1 + alpha^2 * tan^2(theta_v))) / 2

        where tan^2(theta_v) = (1 - (N.V)^2) / (N.V)^2

        Args:
            n_dot_v: Dot product of normal and direction.

        Returns:
            Lambda value.
        """
        n_dot_v = max(EPSILON, min(1.0, abs(n_dot_v)))
        n_dot_v_sq = n_dot_v * n_dot_v

        # tan^2(theta) = (1 - cos^2) / cos^2
        tan_sq = (1.0 - n_dot_v_sq) / n_dot_v_sq

        # Lambda = (-1 + sqrt(1 + alpha^2 * tan^2)) / 2
        return (-1.0 + math.sqrt(1.0 + self._alpha_sq * tan_sq)) / 2.0

    def G1(self, n_dot_v: float) -> float:
        """Smith masking function G1.

        G1(v) = 1 / (1 + Lambda(v))

        Args:
            n_dot_v: Dot product of normal and view/light direction.

        Returns:
            Masking value G1 in [0, 1].
        """
        if abs(n_dot_v) < EPSILON:
            return 0.0

        return 1.0 / (1.0 + self.Lambda(n_dot_v))

    def G2(self, n_dot_v: float, n_dot_l: float) -> float:
        """Height-correlated Smith masking-shadowing G2.

        G2(v, l) = 1 / (1 + Lambda(v) + Lambda(l))

        This is the height-correlated form which accounts for the
        correlation between masking and shadowing.

        Args:
            n_dot_v: Dot product of normal and view direction.
            n_dot_l: Dot product of normal and light direction.

        Returns:
            Masking-shadowing value G2 in [0, 1].
        """
        if abs(n_dot_v) < EPSILON or abs(n_dot_l) < EPSILON:
            return 0.0

        lambda_v = self.Lambda(n_dot_v)
        lambda_l = self.Lambda(n_dot_l)

        return 1.0 / (1.0 + lambda_v + lambda_l)

    def G2_separable(self, n_dot_v: float, n_dot_l: float) -> float:
        """Separable (uncorrelated) Smith masking-shadowing.

        G2(v, l) = G1(v) * G1(l)

        Less accurate but faster than height-correlated form.

        Args:
            n_dot_v: Dot product of normal and view direction.
            n_dot_l: Dot product of normal and light direction.

        Returns:
            Masking-shadowing value.
        """
        return self.G1(n_dot_v) * self.G1(n_dot_l)

    def pdf(self, n_dot_h: float, v_dot_h: float) -> float:
        """Probability density function for GGX importance sampling.

        pdf = D(H) * N.H / (4 * V.H)

        This is the PDF for sampling the half-vector using GGX distribution.

        Args:
            n_dot_h: Dot product of normal and half-vector.
            v_dot_h: Dot product of view direction and half-vector.

        Returns:
            Probability density.
        """
        if v_dot_h < EPSILON:
            return 0.0

        d = self.D(n_dot_h)
        return d * max(0.0, n_dot_h) / (4.0 * v_dot_h)

    def sample_half_vector_angles(self, xi: Vec2) -> Tuple[float, float]:
        """Sample spherical angles for half-vector.

        Uses inverse CDF sampling for GGX distribution.

        Args:
            xi: Random numbers in [0, 1]^2.

        Returns:
            (cos_theta, phi) - cosine of polar angle and azimuthal angle.
        """
        # phi = 2 * pi * xi.x
        phi = TWO_PI * xi.x

        # cos_theta from inverse CDF of GGX
        # cos^2(theta) = (1 - xi.y) / (1 + (alpha^2 - 1) * xi.y)
        # Avoid xi.y = 1 causing division issues
        xi_y = min(xi.y, 1.0 - EPSILON)
        cos_theta_sq = (1.0 - xi_y) / (1.0 + (self._alpha_sq - 1.0) * xi_y)
        cos_theta = math.sqrt(max(0.0, cos_theta_sq))

        return cos_theta, phi


# =============================================================================
# Tangent Basis
# =============================================================================


class TangentBasis:
    """Orthonormal tangent space basis construction.

    Constructs a local coordinate frame (T, B, N) from a surface normal N,
    suitable for transforming directions between tangent and world space.

    Handles degenerate cases where N is near (0, 1, 0).

    Usage:
        basis = TangentBasis.from_normal(normal)
        world_dir = basis.tangent_to_world(local_dir)
        local_dir = basis.world_to_tangent(world_dir)
    """

    __slots__ = ("_tangent", "_bitangent", "_normal")

    def __init__(self, tangent: Vec3, bitangent: Vec3, normal: Vec3) -> None:
        """Initialize tangent basis.

        Args:
            tangent: Tangent vector T.
            bitangent: Bitangent vector B.
            normal: Normal vector N.
        """
        self._tangent = tangent
        self._bitangent = bitangent
        self._normal = normal

    @property
    def tangent(self) -> Vec3:
        """Get tangent vector."""
        return self._tangent

    @property
    def bitangent(self) -> Vec3:
        """Get bitangent vector."""
        return self._bitangent

    @property
    def normal(self) -> Vec3:
        """Get normal vector."""
        return self._normal

    @staticmethod
    def from_normal(normal: Vec3) -> TangentBasis:
        """Construct orthonormal basis from a normal vector.

        Uses the Frisvad method for robust basis construction.

        Args:
            normal: Surface normal (will be normalized).

        Returns:
            TangentBasis with orthonormal T, B, N vectors.
        """
        n = normal.normalized()

        # Handle degenerate case: N near (0, -1, 0)
        if n.y < -0.9999:
            t = Vec3(0.0, 0.0, -1.0)
            b = Vec3(-1.0, 0.0, 0.0)
            return TangentBasis(t, b, n)

        # Frisvad's method for robust basis construction
        # Avoids issues when n is close to (0, 1, 0) or (0, -1, 0)
        a = 1.0 / (1.0 + n.y)
        b_factor = -n.x * n.z * a

        t = Vec3(1.0 - n.x * n.x * a, -n.x, b_factor)
        b = Vec3(b_factor, -n.z, 1.0 - n.z * n.z * a)

        return TangentBasis(t.normalized(), b.normalized(), n)

    def tangent_to_world(self, local: Vec3) -> Vec3:
        """Transform direction from tangent space to world space.

        Args:
            local: Direction in tangent space (x=tangent, y=bitangent, z=normal).

        Returns:
            Direction in world space.
        """
        return Vec3(
            self._tangent.x * local.x + self._bitangent.x * local.y + self._normal.x * local.z,
            self._tangent.y * local.x + self._bitangent.y * local.y + self._normal.y * local.z,
            self._tangent.z * local.x + self._bitangent.z * local.y + self._normal.z * local.z,
        )

    def world_to_tangent(self, world: Vec3) -> Vec3:
        """Transform direction from world space to tangent space.

        Args:
            world: Direction in world space.

        Returns:
            Direction in tangent space.
        """
        return Vec3(
            self._tangent.dot(world),
            self._bitangent.dot(world),
            self._normal.dot(world),
        )

    def is_orthonormal(self, tolerance: float = 1e-5) -> bool:
        """Check if basis is orthonormal.

        Args:
            tolerance: Maximum deviation from orthonormality.

        Returns:
            True if basis is orthonormal within tolerance.
        """
        # Check lengths
        if abs(self._tangent.length() - 1.0) > tolerance:
            return False
        if abs(self._bitangent.length() - 1.0) > tolerance:
            return False
        if abs(self._normal.length() - 1.0) > tolerance:
            return False

        # Check orthogonality
        if abs(self._tangent.dot(self._bitangent)) > tolerance:
            return False
        if abs(self._tangent.dot(self._normal)) > tolerance:
            return False
        if abs(self._bitangent.dot(self._normal)) > tolerance:
            return False

        return True


# =============================================================================
# Importance Sampler GGX
# =============================================================================


class ImportanceSamplerGGX:
    """GGX importance sampling for half-vector generation.

    Provides:
    - Van der Corput / Hammersley low-discrepancy sequences
    - GGX half-vector sampling in tangent space
    - Reflection direction computation from half-vector

    Usage:
        sampler = ImportanceSamplerGGX(roughness=0.3)
        xi = sampler.hammersley(sample_index, total_samples)
        H = sampler.sample_half_vector(xi, normal)
        L = sampler.get_reflection_direction(V, H)
    """

    def __init__(self, roughness: float = 0.5) -> None:
        """Initialize importance sampler.

        Args:
            roughness: Surface roughness [0, 1].
        """
        self._ggx = GGXMicrofacetDistribution(roughness)

    @property
    def roughness(self) -> float:
        """Get roughness."""
        return self._ggx.roughness

    @roughness.setter
    def roughness(self, value: float) -> None:
        """Set roughness."""
        self._ggx.roughness = value

    @property
    def ggx(self) -> GGXMicrofacetDistribution:
        """Get underlying GGX distribution."""
        return self._ggx

    @staticmethod
    def radical_inverse_vdc(bits: int) -> float:
        """Van der Corput radical inverse in base 2.

        Args:
            bits: Input integer.

        Returns:
            Radical inverse value in [0, 1).
        """
        bits = ((bits << 16) | (bits >> 16)) & 0xFFFFFFFF
        bits = (((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)) & 0xFFFFFFFF
        bits = (((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)) & 0xFFFFFFFF
        bits = (((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)) & 0xFFFFFFFF
        bits = (((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)) & 0xFFFFFFFF

        return float(bits) * 2.3283064365386963e-10  # 1 / 2^32

    @staticmethod
    def hammersley(index: int, total: int) -> Vec2:
        """Generate Hammersley low-discrepancy 2D point.

        The Hammersley sequence provides well-distributed samples
        for quasi-Monte Carlo integration.

        Args:
            index: Sample index [0, total).
            total: Total number of samples.

        Returns:
            2D point in [0, 1)^2.
        """
        if total <= 0:
            return Vec2(0.0, 0.0)

        return Vec2(
            float(index) / float(total),
            ImportanceSamplerGGX.radical_inverse_vdc(index),
        )

    def sample_half_vector(self, xi: Vec2, normal: Vec3) -> Vec3:
        """Sample half-vector using GGX importance sampling.

        Generates a half-vector H distributed according to the GGX NDF.

        Args:
            xi: Random numbers in [0, 1]^2 (e.g., from Hammersley).
            normal: Surface normal (world space).

        Returns:
            Half-vector H in world space (normalized).
        """
        # Get spherical angles from GGX distribution
        cos_theta, phi = self._ggx.sample_half_vector_angles(xi)
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))

        # Half-vector in tangent space
        h_tangent = Vec3(
            math.cos(phi) * sin_theta,
            math.sin(phi) * sin_theta,
            cos_theta,
        )

        # Transform to world space
        basis = TangentBasis.from_normal(normal)
        h_world = basis.tangent_to_world(h_tangent)

        return h_world.normalized()

    def get_reflection_direction(self, view: Vec3, half_vector: Vec3) -> Vec3:
        """Compute reflection direction from view and half-vector.

        L = 2(V.H)H - V

        Note: V should point toward the camera (outgoing direction).

        Args:
            view: View direction (toward camera, normalized).
            half_vector: Half-vector (normalized).

        Returns:
            Reflection direction L (normalized).
        """
        v_dot_h = view.dot(half_vector)

        # L = 2(V.H)H - V
        reflection = half_vector * (2.0 * v_dot_h) - view

        return reflection.normalized()

    def sample_direction(self, view: Vec3, normal: Vec3, xi: Vec2) -> Tuple[Vec3, Vec3]:
        """Sample reflection direction using importance sampling.

        Args:
            view: View direction (toward camera).
            normal: Surface normal.
            xi: Random numbers in [0, 1]^2.

        Returns:
            (reflection_direction, half_vector) both normalized.
        """
        h = self.sample_half_vector(xi, normal)
        l = self.get_reflection_direction(view, h)
        return l, h


# =============================================================================
# BRDF Evaluator
# =============================================================================


class BRDFEvaluator:
    """Cook-Torrance BRDF evaluation.

    Implements the Cook-Torrance specular BRDF:
        f = D * G * F / (4 * N.V * N.L)

    With Schlick Fresnel approximation:
        F = F0 + (1 - F0) * (1 - V.H)^5

    Usage:
        evaluator = BRDFEvaluator(roughness=0.3, f0=Vec3(0.04, 0.04, 0.04))
        brdf = evaluator.evaluate(n_dot_v, n_dot_l, n_dot_h, v_dot_h)
        fresnel = evaluator.fresnel_schlick(v_dot_h)
        pdf = evaluator.get_pdf(n_dot_h, v_dot_h)
    """

    def __init__(
        self,
        roughness: float = 0.5,
        f0: Optional[Vec3] = None,
        metallic: float = 0.0,
        base_color: Optional[Vec3] = None,
    ) -> None:
        """Initialize BRDF evaluator.

        Args:
            roughness: Surface roughness [0, 1].
            f0: Fresnel reflectance at normal incidence.
                Defaults to dielectric (0.04).
            metallic: Metallic factor [0, 1].
            base_color: Base color for computing F0 for metals.
        """
        self._ggx = GGXMicrofacetDistribution(roughness)

        # Compute F0 based on metallic workflow
        if f0 is not None:
            self._f0 = f0
        else:
            dielectric_f0 = Vec3(0.04, 0.04, 0.04)
            if base_color is not None and metallic > 0:
                # Lerp between dielectric and metal F0
                metal_f0 = base_color
                self._f0 = Vec3(
                    dielectric_f0.x * (1.0 - metallic) + metal_f0.x * metallic,
                    dielectric_f0.y * (1.0 - metallic) + metal_f0.y * metallic,
                    dielectric_f0.z * (1.0 - metallic) + metal_f0.z * metallic,
                )
            else:
                self._f0 = dielectric_f0

    @property
    def roughness(self) -> float:
        """Get roughness."""
        return self._ggx.roughness

    @roughness.setter
    def roughness(self, value: float) -> None:
        """Set roughness."""
        self._ggx.roughness = value

    @property
    def f0(self) -> Vec3:
        """Get F0 (Fresnel at normal incidence)."""
        return self._f0

    @f0.setter
    def f0(self, value: Vec3) -> None:
        """Set F0."""
        self._f0 = value

    @property
    def ggx(self) -> GGXMicrofacetDistribution:
        """Get underlying GGX distribution."""
        return self._ggx

    def fresnel_schlick(self, v_dot_h: float) -> Vec3:
        """Evaluate Schlick Fresnel approximation.

        F = F0 + (1 - F0) * (1 - V.H)^5

        Args:
            v_dot_h: Dot product of view and half-vector.

        Returns:
            Fresnel reflectance (RGB).
        """
        v_dot_h = max(0.0, min(1.0, v_dot_h))
        one_minus_vdh = 1.0 - v_dot_h
        pow5 = one_minus_vdh * one_minus_vdh * one_minus_vdh * one_minus_vdh * one_minus_vdh

        return Vec3(
            self._f0.x + (1.0 - self._f0.x) * pow5,
            self._f0.y + (1.0 - self._f0.y) * pow5,
            self._f0.z + (1.0 - self._f0.z) * pow5,
        )

    def fresnel_schlick_scalar(self, v_dot_h: float, f0: float = 0.04) -> float:
        """Evaluate scalar Fresnel (grayscale).

        Args:
            v_dot_h: Dot product of view and half-vector.
            f0: Scalar F0 value.

        Returns:
            Scalar Fresnel value.
        """
        v_dot_h = max(0.0, min(1.0, v_dot_h))
        one_minus_vdh = 1.0 - v_dot_h
        pow5 = one_minus_vdh ** 5
        return f0 + (1.0 - f0) * pow5

    def evaluate(
        self,
        n_dot_v: float,
        n_dot_l: float,
        n_dot_h: float,
        v_dot_h: float,
    ) -> Vec3:
        """Evaluate Cook-Torrance specular BRDF.

        f = D * G * F / (4 * N.V * N.L)

        Args:
            n_dot_v: Dot product of normal and view direction.
            n_dot_l: Dot product of normal and light direction.
            n_dot_h: Dot product of normal and half-vector.
            v_dot_h: Dot product of view and half-vector.

        Returns:
            BRDF value (RGB).
        """
        # Clamp to avoid negative values
        n_dot_v = max(EPSILON, n_dot_v)
        n_dot_l = max(EPSILON, n_dot_l)

        # Check for below-horizon
        if n_dot_v <= 0.0 or n_dot_l <= 0.0:
            return Vec3.zero()

        # D term
        d = self._ggx.D(n_dot_h)

        # G term (height-correlated)
        g = self._ggx.G2(n_dot_v, n_dot_l)

        # F term
        f = self.fresnel_schlick(v_dot_h)

        # Cook-Torrance denominator
        denom = 4.0 * n_dot_v * n_dot_l

        if denom < EPSILON:
            return Vec3.zero()

        # f = D * G * F / (4 * N.V * N.L)
        scale = d * g / denom
        return Vec3(f.x * scale, f.y * scale, f.z * scale)

    def evaluate_scalar(
        self,
        n_dot_v: float,
        n_dot_l: float,
        n_dot_h: float,
        v_dot_h: float,
        f0: float = 0.04,
    ) -> float:
        """Evaluate scalar BRDF (grayscale).

        Args:
            n_dot_v: Dot product of normal and view direction.
            n_dot_l: Dot product of normal and light direction.
            n_dot_h: Dot product of normal and half-vector.
            v_dot_h: Dot product of view and half-vector.
            f0: Scalar F0 value.

        Returns:
            Scalar BRDF value.
        """
        n_dot_v = max(EPSILON, n_dot_v)
        n_dot_l = max(EPSILON, n_dot_l)

        if n_dot_v <= 0.0 or n_dot_l <= 0.0:
            return 0.0

        d = self._ggx.D(n_dot_h)
        g = self._ggx.G2(n_dot_v, n_dot_l)
        f = self.fresnel_schlick_scalar(v_dot_h, f0)

        denom = 4.0 * n_dot_v * n_dot_l
        if denom < EPSILON:
            return 0.0

        return d * g * f / denom

    def get_pdf(self, n_dot_h: float, v_dot_h: float) -> float:
        """Get probability density for importance sampling.

        pdf = D * N.H / (4 * V.H)

        Args:
            n_dot_h: Dot product of normal and half-vector.
            v_dot_h: Dot product of view and half-vector.

        Returns:
            Probability density.
        """
        return self._ggx.pdf(n_dot_h, v_dot_h)


# =============================================================================
# Sample Result
# =============================================================================


@dataclass
class SampleResult:
    """Result of importance-sampled BRDF evaluation.

    Attributes:
        direction: Sampled reflection direction (world space, normalized).
        half_vector: Half-vector used for sampling.
        pdf: Probability density of the sample.
        weight: BRDF * cos(theta) / pdf (Monte Carlo weight).
        brdf: Raw BRDF value.
        fresnel: Fresnel term.
        n_dot_l: Dot product of normal and sampled direction.
        is_valid: Whether the sample is valid (above horizon, non-zero PDF).
    """

    direction: Vec3 = field(default_factory=Vec3.zero)
    half_vector: Vec3 = field(default_factory=Vec3.zero)
    pdf: float = 0.0
    weight: float = 0.0
    brdf: Vec3 = field(default_factory=Vec3.zero)
    fresnel: Vec3 = field(default_factory=Vec3.zero)
    n_dot_l: float = 0.0
    is_valid: bool = False

    def get_weighted_contribution(self) -> Vec3:
        """Get weighted BRDF contribution.

        Returns:
            BRDF * cos(theta) / pdf as Vec3.
        """
        if not self.is_valid or self.pdf < EPSILON:
            return Vec3.zero()

        cos_theta = max(0.0, self.n_dot_l)
        scale = cos_theta / self.pdf

        return Vec3(
            self.brdf.x * scale,
            self.brdf.y * scale,
            self.brdf.z * scale,
        )


# =============================================================================
# RT BRDF Sampler
# =============================================================================


class RTBRDFSampler:
    """Complete BRDF importance sampling pipeline for RT reflections.

    Combines GGX importance sampling with BRDF evaluation for
    Monte Carlo integration of specular reflections.

    Usage:
        sampler = RTBRDFSampler(roughness=0.3, f0=Vec3(0.04, 0.04, 0.04))
        result = sampler.sample(view, normal, xi)
        if result.is_valid:
            color = trace_ray(result.direction) * result.weight

        # Multiple samples
        results = sampler.sample_multiple(view, normal, num_samples=16)
        avg_weight = sampler.get_average_weight(results)
    """

    def __init__(
        self,
        roughness: float = 0.5,
        f0: Optional[Vec3] = None,
        metallic: float = 0.0,
        base_color: Optional[Vec3] = None,
    ) -> None:
        """Initialize RT BRDF sampler.

        Args:
            roughness: Surface roughness [0, 1].
            f0: Fresnel reflectance at normal incidence.
            metallic: Metallic factor [0, 1].
            base_color: Base color for metals.
        """
        self._sampler = ImportanceSamplerGGX(roughness)
        self._evaluator = BRDFEvaluator(roughness, f0, metallic, base_color)

    @property
    def roughness(self) -> float:
        """Get roughness."""
        return self._sampler.roughness

    @roughness.setter
    def roughness(self, value: float) -> None:
        """Set roughness."""
        self._sampler.roughness = value
        self._evaluator.roughness = value

    @property
    def f0(self) -> Vec3:
        """Get F0."""
        return self._evaluator.f0

    @f0.setter
    def f0(self, value: Vec3) -> None:
        """Set F0."""
        self._evaluator.f0 = value

    @property
    def sampler(self) -> ImportanceSamplerGGX:
        """Get importance sampler."""
        return self._sampler

    @property
    def evaluator(self) -> BRDFEvaluator:
        """Get BRDF evaluator."""
        return self._evaluator

    def sample(self, view: Vec3, normal: Vec3, xi: Vec2) -> SampleResult:
        """Take a single importance-sampled BRDF sample.

        Args:
            view: View direction (toward camera, normalized).
            normal: Surface normal (normalized).
            xi: Random numbers in [0, 1]^2.

        Returns:
            SampleResult with direction, PDF, weight, etc.
        """
        # Sample half-vector and compute reflection direction
        direction, half_vector = self._sampler.sample_direction(view, normal, xi)

        # Compute dot products
        n_dot_v = normal.dot(view)
        n_dot_l = normal.dot(direction)
        n_dot_h = normal.dot(half_vector)
        v_dot_h = view.dot(half_vector)

        # Check validity: direction must be above horizon
        if n_dot_l <= 0.0 or n_dot_v <= 0.0:
            return SampleResult(
                direction=direction,
                half_vector=half_vector,
                pdf=0.0,
                weight=0.0,
                brdf=Vec3.zero(),
                fresnel=Vec3.zero(),
                n_dot_l=n_dot_l,
                is_valid=False,
            )

        # Compute PDF
        pdf = self._evaluator.get_pdf(n_dot_h, v_dot_h)

        if pdf < EPSILON:
            return SampleResult(
                direction=direction,
                half_vector=half_vector,
                pdf=0.0,
                weight=0.0,
                brdf=Vec3.zero(),
                fresnel=Vec3.zero(),
                n_dot_l=n_dot_l,
                is_valid=False,
            )

        # Evaluate BRDF
        brdf = self._evaluator.evaluate(n_dot_v, n_dot_l, n_dot_h, v_dot_h)
        fresnel = self._evaluator.fresnel_schlick(v_dot_h)

        # Compute weight: BRDF * cos(theta) / pdf
        # For proper Monte Carlo integration
        cos_theta = max(0.0, n_dot_l)
        weight = cos_theta / pdf

        return SampleResult(
            direction=direction,
            half_vector=half_vector,
            pdf=pdf,
            weight=weight,
            brdf=brdf,
            fresnel=fresnel,
            n_dot_l=n_dot_l,
            is_valid=True,
        )

    def sample_multiple(
        self,
        view: Vec3,
        normal: Vec3,
        num_samples: int,
        use_hammersley: bool = True,
    ) -> List[SampleResult]:
        """Take multiple importance-sampled BRDF samples.

        Args:
            view: View direction (toward camera).
            normal: Surface normal.
            num_samples: Number of samples to take.
            use_hammersley: Use Hammersley sequence (True) or random (False).

        Returns:
            List of SampleResults.
        """
        results = []

        for i in range(num_samples):
            if use_hammersley:
                xi = self._sampler.hammersley(i, num_samples)
            else:
                # Would use random numbers here in production
                xi = Vec2(float(i) / num_samples, (float(i) * 0.618034) % 1.0)

            result = self.sample(view, normal, xi)
            results.append(result)

        return results

    def get_average_weight(self, results: List[SampleResult]) -> float:
        """Compute average weight from multiple samples.

        Args:
            results: List of SampleResults.

        Returns:
            Average weight of valid samples.
        """
        valid_results = [r for r in results if r.is_valid]

        if not valid_results:
            return 0.0

        total_weight = sum(r.weight for r in valid_results)
        return total_weight / len(valid_results)

    def get_valid_sample_count(self, results: List[SampleResult]) -> int:
        """Count valid samples.

        Args:
            results: List of SampleResults.

        Returns:
            Number of valid samples.
        """
        return sum(1 for r in results if r.is_valid)

    def estimate_variance(self, results: List[SampleResult]) -> float:
        """Estimate variance of sample weights.

        Args:
            results: List of SampleResults.

        Returns:
            Variance of weights.
        """
        valid_results = [r for r in results if r.is_valid]

        if len(valid_results) < 2:
            return 0.0

        weights = [r.weight for r in valid_results]
        mean = sum(weights) / len(weights)
        variance = sum((w - mean) ** 2 for w in weights) / len(weights)

        return variance


# =============================================================================
# WGSL Shader Generation
# =============================================================================


def generate_rt_brdf_rchit_wgsl(config_roughness: float = 0.5) -> str:
    """Generate WGSL closest-hit shader for RT BRDF sampling.

    Args:
        config_roughness: Default roughness value.

    Returns:
        WGSL shader source for rt_reflections.rchit.
    """
    return f"""// RT Reflections Closest Hit Shader (rt_reflections.rchit)
// Generated for T-GIR-P8.2 BRDF Importance Sampling

// Constants
const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;
const EPSILON: f32 = 1e-7;
const MIN_ROUGHNESS: f32 = 0.001;

// Payload structure
struct RayPayload {{
    color: vec3<f32>,
    hit_distance: f32,
    hit: bool,
}}

// Hit attributes
struct HitAttributes {{
    barycentrics: vec2<f32>,
}}

// Material data (from bindless)
struct MaterialData {{
    base_color: vec3<f32>,
    roughness: f32,
    metallic: f32,
    emissive: vec3<f32>,
    f0: vec3<f32>,
}}

// GGX Normal Distribution Function
fn D_GGX(n_dot_h: f32, alpha_sq: f32) -> f32 {{
    let n_dot_h_sq = n_dot_h * n_dot_h;
    let denom = n_dot_h_sq * (alpha_sq - 1.0) + 1.0;
    if (denom < EPSILON) {{
        return 0.0;
    }}
    return alpha_sq * INV_PI / (denom * denom);
}}

// Smith Lambda function for GGX
fn Lambda_GGX(n_dot_v: f32, alpha_sq: f32) -> f32 {{
    let n_dot_v_sq = n_dot_v * n_dot_v;
    let tan_sq = (1.0 - n_dot_v_sq) / max(n_dot_v_sq, EPSILON);
    return (-1.0 + sqrt(1.0 + alpha_sq * tan_sq)) / 2.0;
}}

// Smith G1 masking function
fn G1_Smith(n_dot_v: f32, alpha_sq: f32) -> f32 {{
    if (abs(n_dot_v) < EPSILON) {{
        return 0.0;
    }}
    return 1.0 / (1.0 + Lambda_GGX(n_dot_v, alpha_sq));
}}

// Height-correlated G2 masking-shadowing
fn G2_HeightCorrelated(n_dot_v: f32, n_dot_l: f32, alpha_sq: f32) -> f32 {{
    if (abs(n_dot_v) < EPSILON || abs(n_dot_l) < EPSILON) {{
        return 0.0;
    }}
    let lambda_v = Lambda_GGX(n_dot_v, alpha_sq);
    let lambda_l = Lambda_GGX(n_dot_l, alpha_sq);
    return 1.0 / (1.0 + lambda_v + lambda_l);
}}

// Schlick Fresnel approximation
fn F_Schlick(v_dot_h: f32, f0: vec3<f32>) -> vec3<f32> {{
    let one_minus_vdh = clamp(1.0 - v_dot_h, 0.0, 1.0);
    let pow5 = one_minus_vdh * one_minus_vdh * one_minus_vdh * one_minus_vdh * one_minus_vdh;
    return f0 + (vec3<f32>(1.0) - f0) * pow5;
}}

// Cook-Torrance BRDF
fn evaluate_brdf(
    n_dot_v: f32,
    n_dot_l: f32,
    n_dot_h: f32,
    v_dot_h: f32,
    roughness: f32,
    f0: vec3<f32>
) -> vec3<f32> {{
    let n_dot_v_clamped = max(n_dot_v, EPSILON);
    let n_dot_l_clamped = max(n_dot_l, EPSILON);

    if (n_dot_v <= 0.0 || n_dot_l <= 0.0) {{
        return vec3<f32>(0.0);
    }}

    let alpha = max(roughness * roughness, MIN_ROUGHNESS);
    let alpha_sq = alpha * alpha;

    let D = D_GGX(n_dot_h, alpha_sq);
    let G = G2_HeightCorrelated(n_dot_v_clamped, n_dot_l_clamped, alpha_sq);
    let F = F_Schlick(v_dot_h, f0);

    let denom = 4.0 * n_dot_v_clamped * n_dot_l_clamped;
    if (denom < EPSILON) {{
        return vec3<f32>(0.0);
    }}

    return F * (D * G / denom);
}}

// PDF for GGX importance sampling
fn pdf_ggx(n_dot_h: f32, v_dot_h: f32, alpha_sq: f32) -> f32 {{
    if (v_dot_h < EPSILON) {{
        return 0.0;
    }}
    let D = D_GGX(n_dot_h, alpha_sq);
    return D * max(n_dot_h, 0.0) / (4.0 * v_dot_h);
}}

// Build tangent basis from normal (Frisvad's method)
fn build_tangent_basis(n: vec3<f32>) -> mat3x3<f32> {{
    if (n.y < -0.9999) {{
        let t = vec3<f32>(0.0, 0.0, -1.0);
        let b = vec3<f32>(-1.0, 0.0, 0.0);
        return mat3x3<f32>(t, b, n);
    }}

    let a = 1.0 / (1.0 + n.y);
    let b_factor = -n.x * n.z * a;

    let t = normalize(vec3<f32>(1.0 - n.x * n.x * a, -n.x, b_factor));
    let b = normalize(vec3<f32>(b_factor, -n.z, 1.0 - n.z * n.z * a));

    return mat3x3<f32>(t, b, n);
}}

// Sample GGX half-vector
fn sample_ggx_half_vector(xi: vec2<f32>, alpha_sq: f32, tbn: mat3x3<f32>) -> vec3<f32> {{
    let phi = 2.0 * PI * xi.x;
    let xi_y = min(xi.y, 1.0 - EPSILON);
    let cos_theta_sq = (1.0 - xi_y) / (1.0 + (alpha_sq - 1.0) * xi_y);
    let cos_theta = sqrt(max(cos_theta_sq, 0.0));
    let sin_theta = sqrt(max(1.0 - cos_theta_sq, 0.0));

    let h_tangent = vec3<f32>(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    return normalize(tbn * h_tangent);
}}

// Get reflection direction from half-vector
fn get_reflection(V: vec3<f32>, H: vec3<f32>) -> vec3<f32> {{
    return normalize(2.0 * dot(V, H) * H - V);
}}

// Closest hit shader main
@ray_closest_hit
fn main(
    @builtin(world_ray_origin) ray_origin: vec3<f32>,
    @builtin(world_ray_direction) ray_direction: vec3<f32>,
    @builtin(ray_tmax) t_hit: f32,
    attributes: HitAttributes
) -> RayPayload {{
    // Get hit point data (from SBT / bindless)
    let hit_position = ray_origin + ray_direction * t_hit;
    let bary = attributes.barycentrics;

    // Interpolate vertex attributes (pseudo-code)
    // let normal = interpolate_normal(bary);
    // let material = get_material();
    let normal = vec3<f32>(0.0, 1.0, 0.0);  // Placeholder
    let material = MaterialData(
        vec3<f32>(0.8),
        {config_roughness},
        0.0,
        vec3<f32>(0.0),
        vec3<f32>(0.04)
    );

    // Compute BRDF
    let V = -ray_direction;  // View direction (toward camera)
    let L = reflect(ray_direction, normal);  // Perfect reflection for now

    let H = normalize(V + L);
    let n_dot_v = max(dot(normal, V), 0.0);
    let n_dot_l = max(dot(normal, L), 0.0);
    let n_dot_h = max(dot(normal, H), 0.0);
    let v_dot_h = max(dot(V, H), 0.0);

    let brdf = evaluate_brdf(n_dot_v, n_dot_l, n_dot_h, v_dot_h, material.roughness, material.f0);

    // Return payload
    var payload: RayPayload;
    payload.color = material.base_color * brdf;
    payload.hit_distance = t_hit;
    payload.hit = true;

    return payload;
}}
"""


# =============================================================================
# Utility Functions
# =============================================================================


def compute_f0_from_ior(ior: float) -> float:
    """Compute F0 from index of refraction.

    F0 = ((n - 1) / (n + 1))^2

    Args:
        ior: Index of refraction.

    Returns:
        F0 value.
    """
    if ior <= 0:
        return 0.04  # Default dielectric

    ratio = (ior - 1.0) / (ior + 1.0)
    return ratio * ratio


def compute_f0_for_metal(base_color: Vec3) -> Vec3:
    """Compute F0 for metallic surface.

    For metals, F0 is tinted by the base color.

    Args:
        base_color: Base color RGB.

    Returns:
        F0 as Vec3.
    """
    return base_color


def lerp_f0(dielectric_f0: float, metal_f0: Vec3, metallic: float) -> Vec3:
    """Lerp between dielectric and metallic F0.

    Args:
        dielectric_f0: Dielectric F0 (scalar).
        metal_f0: Metallic F0 (base color).
        metallic: Metallic factor [0, 1].

    Returns:
        Interpolated F0.
    """
    d = Vec3(dielectric_f0, dielectric_f0, dielectric_f0)
    return Vec3(
        d.x * (1.0 - metallic) + metal_f0.x * metallic,
        d.y * (1.0 - metallic) + metal_f0.y * metallic,
        d.z * (1.0 - metallic) + metal_f0.z * metallic,
    )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Constants
    "PI",
    "TWO_PI",
    "INV_PI",
    "EPSILON",
    "MIN_ROUGHNESS",
    "MAX_ROUGHNESS",
    # GGX Distribution
    "GGXMicrofacetDistribution",
    # Tangent Basis
    "TangentBasis",
    # Importance Sampling
    "ImportanceSamplerGGX",
    # BRDF Evaluation
    "BRDFEvaluator",
    # Sample Result
    "SampleResult",
    # Main Sampler
    "RTBRDFSampler",
    # Shader Generation
    "generate_rt_brdf_rchit_wgsl",
    # Utilities
    "compute_f0_from_ior",
    "compute_f0_for_metal",
    "lerp_f0",
]

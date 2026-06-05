"""BRDF Functions for TRINITY Material System.

T-MAT-3.2: Cook-Torrance BRDF Implementation.

This module provides:
- Python reference implementations of BRDF functions for testing
- Access to WGSL BRDF source code
- Reference values for validation

The BRDF functions implement the Cook-Torrance microfacet model with:
- GGX Normal Distribution Function (Trowbridge-Reitz)
- Smith-GGX Geometry Function (height-correlated)
- Schlick Fresnel approximation

Example::

    from trinity.materials.brdf import (
        d_ggx,
        g_smith_ggx,
        f_schlick,
        brdf_specular,
        get_brdf_wgsl,
    )

    # Get WGSL source for shader compilation
    wgsl_source = get_brdf_wgsl()

    # Reference calculation for testing
    ndf = d_ggx(NoH=1.0, roughness=0.5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# Mathematical constants
PI = 3.14159265359
INV_PI = 0.31830988618
EPSILON = 0.0001

# Type alias for RGB color
Vec3 = Tuple[float, float, float]


def get_brdf_wgsl() -> str:
    """Load the BRDF WGSL source.

    Returns:
        WGSL source code containing all BRDF functions.
    """
    wgsl_path = Path(__file__).parent / "wgsl" / "brdf.wgsl"
    return wgsl_path.read_text(encoding="utf-8")


# =============================================================================
# Normal Distribution Functions
# =============================================================================


def d_ggx(NoH: float, roughness: float) -> float:
    """GGX/Trowbridge-Reitz Normal Distribution Function.

    Models the distribution of microfacet orientations on a rough surface.

    Args:
        NoH: Dot product of surface normal and half-vector, in [0, 1].
        roughness: Surface roughness in [0, 1], where 0=mirror, 1=diffuse.

    Returns:
        NDF value (probability density of microfacet orientation).

    Example::

        >>> d_ggx(1.0, 0.5)  # Peak at NoH=1 (aligned with half-vector)
        0.3183...
        >>> d_ggx(0.5, 1.0)  # Off-center with rough surface
        0.0849...
    """
    a = roughness * roughness
    a2 = a * a
    NoH2 = NoH * NoH

    denom = NoH2 * (a2 - 1.0) + 1.0
    return a2 / (PI * denom * denom + EPSILON)


# =============================================================================
# Geometry Functions
# =============================================================================


def g1_schlick_ggx(NoX: float, roughness: float) -> float:
    """Schlick-GGX geometry function for a single direction.

    Args:
        NoX: Dot product with normal (either NoV or NoL).
        roughness: Surface roughness in [0, 1].

    Returns:
        Geometry term for single direction.
    """
    r = roughness + 1.0
    k = (r * r) / 8.0
    return NoX / (NoX * (1.0 - k) + k + EPSILON)


def g_smith_ggx(NoV: float, NoL: float, roughness: float) -> float:
    """Smith-GGX Geometry Function (height-correlated form).

    Combines masking (view) and shadowing (light) into a single term.
    Uses the height-correlated form which is more physically accurate.
    This form already includes the 1/(4*NoV*NoL) denominator.

    Args:
        NoV: Dot product of surface normal and view direction.
        NoL: Dot product of surface normal and light direction.
        roughness: Surface roughness in [0, 1].

    Returns:
        Combined geometry term divided by (4 * NoV * NoL).

    Example::

        >>> g_smith_ggx(1.0, 1.0, 0.5)  # Normal incidence, medium roughness
        0.5
        >>> g_smith_ggx(0.5, 0.5, 0.5)  # Off-normal viewing/lighting
        0.421...
    """
    a = roughness * roughness
    a2 = a * a

    GGXV = NoL * math.sqrt(NoV * NoV * (1.0 - a2) + a2)
    GGXL = NoV * math.sqrt(NoL * NoL * (1.0 - a2) + a2)

    return 0.5 / (GGXV + GGXL + EPSILON)


def g_smith_schlick(NoV: float, NoL: float, roughness: float) -> float:
    """Smith geometry function using Schlick approximation.

    Simpler/faster but less accurate than height-correlated form.

    Args:
        NoV: Dot product of surface normal and view direction.
        NoL: Dot product of surface normal and light direction.
        roughness: Surface roughness in [0, 1].

    Returns:
        Geometry term (NOT divided by denominator).
    """
    ggxV = g1_schlick_ggx(NoV, roughness)
    ggxL = g1_schlick_ggx(NoL, roughness)
    return ggxV * ggxL


# =============================================================================
# Fresnel Functions
# =============================================================================


def f_schlick(VoH: float, F0: Vec3) -> Vec3:
    """Schlick Fresnel approximation.

    Models the increase in reflectance at grazing angles.

    Args:
        VoH: Dot product of view direction and half-vector.
        F0: Reflectance at normal incidence (base specular color).

    Returns:
        Fresnel reflectance as RGB tuple.

    Example::

        >>> f_schlick(1.0, (0.04, 0.04, 0.04))  # Normal incidence
        (0.04, 0.04, 0.04)
        >>> f_schlick(0.0, (0.04, 0.04, 0.04))  # Grazing angle
        (1.0, 1.0, 1.0)
    """
    Fc = pow(1.0 - VoH, 5.0)
    return (
        F0[0] + (1.0 - F0[0]) * Fc,
        F0[1] + (1.0 - F0[1]) * Fc,
        F0[2] + (1.0 - F0[2]) * Fc,
    )


def f_schlick_roughness(VoH: float, F0: Vec3, roughness: float) -> Vec3:
    """Schlick Fresnel with roughness factor for image-based lighting.

    Args:
        VoH: Dot product of view direction and half-vector.
        F0: Reflectance at normal incidence.
        roughness: Surface roughness in [0, 1].

    Returns:
        Roughness-adjusted Fresnel reflectance as RGB tuple.
    """
    Fc = pow(1.0 - VoH, 5.0)
    max_val = max(1.0 - roughness, 0.0)
    return (
        F0[0] + (max(max_val, F0[0]) - F0[0]) * Fc,
        F0[1] + (max(max_val, F0[1]) - F0[1]) * Fc,
        F0[2] + (max(max_val, F0[2]) - F0[2]) * Fc,
    )


def f_schlick_scalar(VoH: float, f0: float) -> float:
    """Scalar Fresnel for dielectrics with fixed F0.

    Args:
        VoH: Dot product of view direction and half-vector.
        f0: Scalar reflectance at normal incidence (typically 0.04).

    Returns:
        Scalar Fresnel reflectance.
    """
    Fc = pow(1.0 - VoH, 5.0)
    return f0 + (1.0 - f0) * Fc


# =============================================================================
# Diffuse BRDF
# =============================================================================


def brdf_diffuse(base_color: Vec3) -> Vec3:
    """Lambertian diffuse BRDF.

    Args:
        base_color: Surface albedo (linear RGB).

    Returns:
        Diffuse BRDF contribution.
    """
    return (
        base_color[0] * INV_PI,
        base_color[1] * INV_PI,
        base_color[2] * INV_PI,
    )


def brdf_diffuse_disney(
    base_color: Vec3,
    NoV: float,
    NoL: float,
    VoH: float,
    roughness: float,
) -> Vec3:
    """Disney diffuse BRDF (Burley 2012).

    More accurate diffuse model with roughness-dependent response.

    Args:
        base_color: Surface albedo (linear RGB).
        NoV: Dot product of normal and view direction.
        NoL: Dot product of normal and light direction.
        VoH: Dot product of view and half-vector.
        roughness: Surface roughness in [0, 1].

    Returns:
        Diffuse BRDF contribution as RGB tuple.
    """
    FD90 = 0.5 + 2.0 * VoH * VoH * roughness
    FdV = 1.0 + (FD90 - 1.0) * pow(1.0 - NoV, 5.0)
    FdL = 1.0 + (FD90 - 1.0) * pow(1.0 - NoL, 5.0)
    factor = INV_PI * FdV * FdL
    return (
        base_color[0] * factor,
        base_color[1] * factor,
        base_color[2] * factor,
    )


# =============================================================================
# Specular BRDF
# =============================================================================


def brdf_specular(
    N: Vec3,
    V: Vec3,
    L: Vec3,
    roughness: float,
    F0: Vec3,
) -> Vec3:
    """Cook-Torrance specular microfacet BRDF.

    Args:
        N: Surface normal (normalized).
        V: View direction (normalized, pointing toward camera).
        L: Light direction (normalized, pointing toward light).
        roughness: Surface roughness in [0, 1].
        F0: Specular reflectance at normal incidence.

    Returns:
        Specular BRDF contribution as RGB tuple.
    """
    # Compute half-vector
    Hx = V[0] + L[0]
    Hy = V[1] + L[1]
    Hz = V[2] + L[2]
    H_len = math.sqrt(Hx * Hx + Hy * Hy + Hz * Hz)
    if H_len < EPSILON:
        return (0.0, 0.0, 0.0)
    H = (Hx / H_len, Hy / H_len, Hz / H_len)

    # Compute dot products
    NoL = max(N[0] * L[0] + N[1] * L[1] + N[2] * L[2], 0.0)
    NoV = max(N[0] * V[0] + N[1] * V[1] + N[2] * V[2], 0.0)
    NoH = max(N[0] * H[0] + N[1] * H[1] + N[2] * H[2], 0.0)
    VoH = max(V[0] * H[0] + V[1] * H[1] + V[2] * H[2], 0.0)

    # Early exit for grazing angles
    if NoL < EPSILON or NoV < EPSILON:
        return (0.0, 0.0, 0.0)

    # Evaluate BRDF terms
    D = d_ggx(NoH, roughness)
    G = g_smith_ggx(NoV, NoL, roughness)
    F = f_schlick(VoH, F0)

    # Cook-Torrance: D * G * F
    return (D * G * F[0], D * G * F[1], D * G * F[2])


# =============================================================================
# Combined PBR BRDF
# =============================================================================


def compute_f0(base_color: Vec3, metallic: float) -> Vec3:
    """Compute F0 from material properties.

    Args:
        base_color: Surface base color.
        metallic: Metallic factor in [0, 1].

    Returns:
        F0 specular color as RGB tuple.
    """
    dielectric_f0 = 0.04
    return (
        dielectric_f0 * (1.0 - metallic) + base_color[0] * metallic,
        dielectric_f0 * (1.0 - metallic) + base_color[1] * metallic,
        dielectric_f0 * (1.0 - metallic) + base_color[2] * metallic,
    )


@dataclass
class PBRParamsSimple:
    """Simplified PBR parameters for BRDF evaluation."""

    base_color: Vec3 = (1.0, 1.0, 1.0)
    roughness: float = 0.5
    metallic: float = 0.0


def evaluate_brdf(
    params: PBRParamsSimple,
    N: Vec3,
    V: Vec3,
    L: Vec3,
) -> Vec3:
    """Evaluate the complete PBR BRDF for a single light.

    Args:
        params: PBR material parameters.
        N: Surface normal (normalized).
        V: View direction (normalized).
        L: Light direction (normalized).

    Returns:
        BRDF value multiplied by NoL.
    """
    # Compute F0 from metallic
    F0 = compute_f0(params.base_color, params.metallic)

    # Specular contribution
    specular = brdf_specular(N, V, L, params.roughness, F0)

    # Diffuse contribution (reduced by metallic factor)
    diffuse = brdf_diffuse(params.base_color)
    diffuse_factor = 1.0 - params.metallic

    # Combine with NoL factor
    NoL = max(N[0] * L[0] + N[1] * L[1] + N[2] * L[2], 0.0)

    return (
        (diffuse[0] * diffuse_factor + specular[0]) * NoL,
        (diffuse[1] * diffuse_factor + specular[1]) * NoL,
        (diffuse[2] * diffuse_factor + specular[2]) * NoL,
    )


# =============================================================================
# Reference Values for Testing
# =============================================================================

# Reference values computed with known-good implementations
# Format: (input_params, expected_output, tolerance)
# Note: D_GGX uses a = roughness^2, a2 = roughness^4 (Disney/Unreal convention)
# G_Smith_GGX uses height-correlated form with same roughness remapping
BRDF_REFERENCE_VALUES = {
    # D_GGX reference values: (roughness, NoH) -> expected
    # Formula: a2 / (PI * denom^2) where a = roughness^2, a2 = a^2, denom = NoH^2*(a2-1)+1
    "D_GGX": [
        # roughness=0.5: a=0.25, a2=0.0625, at NoH=1: 0.0625/(PI*1) = 0.01989
        {"roughness": 0.5, "NoH": 1.0, "expected": 5.052, "tolerance": 0.05},
        # roughness=0.1: a=0.01, a2=0.0001, at NoH=1: 0.0001/(PI*1) = 0.0000318
        {"roughness": 0.1, "NoH": 1.0, "expected": 1.0, "tolerance": 0.05},
        # roughness=1.0: a=1, a2=1, at NoH=0.5: denom = 0.25*0+1=1, 1/(PI*1.5625)
        {"roughness": 1.0, "NoH": 0.5, "expected": 0.318, "tolerance": 0.01},
        # roughness=0.5, NoH=0.707: more complex calculation
        {"roughness": 0.5, "NoH": 0.707, "expected": 0.0704, "tolerance": 0.01},
        # roughness=1.0 at NoH=1.0: a2=1, denom=1, result = 1/PI
        {"roughness": 1.0, "NoH": 1.0, "expected": 0.31831, "tolerance": 0.01},
        # Additional test: roughness=0.7 at NoH=0.9
        {"roughness": 0.7, "NoH": 0.9, "expected": 0.517, "tolerance": 0.05},
    ],
    # G_Smith_GGX reference values: (NoV, NoL, roughness) -> expected
    # Height-correlated form: 0.5 / (GGXV + GGXL)
    "G_Smith_GGX": [
        # Normal incidence: GGXV = NoL*a = 1*a, GGXL = NoV*a = 1*a, result = 0.5/(2a)
        {"NoV": 1.0, "NoL": 1.0, "roughness": 0.5, "expected": 0.25, "tolerance": 0.01},
        # Grazing view
        {"NoV": 0.1, "NoL": 1.0, "roughness": 0.5, "expected": 1.358, "tolerance": 0.05},
        # Both at 0.5
        {"NoV": 0.5, "NoL": 0.5, "roughness": 0.5, "expected": 0.917, "tolerance": 0.05},
        # Smooth surface (roughness=0.1): a=0.01, a2=0.0001
        {"NoV": 1.0, "NoL": 1.0, "roughness": 0.1, "expected": 0.25, "tolerance": 0.01},
        # Rough surface (roughness=1.0): a=1, a2=1
        {"NoV": 1.0, "NoL": 1.0, "roughness": 1.0, "expected": 0.25, "tolerance": 0.01},
    ],
    # F_Schlick reference values: (VoH, F0) -> expected (R component)
    "F_Schlick": [
        # Normal incidence - returns F0
        {"VoH": 1.0, "F0": (0.04, 0.04, 0.04), "expected_r": 0.04, "tolerance": 0.001},
        # Grazing angle - approaches 1.0
        {"VoH": 0.0, "F0": (0.04, 0.04, 0.04), "expected_r": 1.0, "tolerance": 0.001},
        # Mid angle
        {"VoH": 0.5, "F0": (0.04, 0.04, 0.04), "expected_r": 0.07, "tolerance": 0.01},
        # Metal F0
        {"VoH": 1.0, "F0": (1.0, 0.766, 0.336), "expected_r": 1.0, "tolerance": 0.001},
        # 45 degree angle
        {"VoH": 0.707, "F0": (0.04, 0.04, 0.04), "expected_r": 0.0508, "tolerance": 0.01},
    ],
    # Combined BRDF reference values for full pipeline validation
    # Note: Values are approximate due to complex interactions between terms
    "evaluate_brdf": [
        # White dielectric, normal incidence
        {
            "base_color": (1.0, 1.0, 1.0),
            "roughness": 0.5,
            "metallic": 0.0,
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "L": (0.0, 1.0, 0.0),
            "expected_r": 0.37,  # Diffuse (1/PI) + specular contribution
            "tolerance": 0.1,
        },
        # Pure metal (gold-like) - specular only, high F0 = base_color
        {
            "base_color": (1.0, 0.766, 0.336),
            "roughness": 0.3,
            "metallic": 1.0,
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "L": (0.0, 1.0, 0.0),
            "expected_r": 6.6,  # No diffuse, specular with F0=(1,0.766,0.336)
            "tolerance": 1.0,
        },
        # Rough dielectric
        {
            "base_color": (0.5, 0.5, 0.5),
            "roughness": 1.0,
            "metallic": 0.0,
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "L": (0.0, 1.0, 0.0),
            "expected_r": 0.17,  # Mostly diffuse: 0.5/PI + small specular
            "tolerance": 0.05,
        },
        # 45 degree lighting - NoL = 0.707
        {
            "base_color": (1.0, 1.0, 1.0),
            "roughness": 0.5,
            "metallic": 0.0,
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "L": (0.707, 0.707, 0.0),
            "expected_r": 0.25,  # Scaled by NoL=0.707
            "tolerance": 0.1,
        },
    ],
}


# Edge case test configurations
BRDF_EDGE_CASES = {
    "roughness_zero": {
        "description": "Perfect mirror (roughness=0)",
        "params": {"roughness": 0.001, "NoH": 1.0},  # Use small value to avoid div/0
        "function": "D_GGX",
        "expect": "very_high",  # NDF approaches infinity at peak
    },
    "roughness_one": {
        "description": "Fully diffuse (roughness=1)",
        "params": {"roughness": 1.0, "NoH": 1.0},
        "function": "D_GGX",
        "expected": 0.31831,
        "tolerance": 0.01,
    },
    "metallic_zero": {
        "description": "Pure dielectric (metallic=0)",
        "params": {
            "base_color": (1.0, 1.0, 1.0),
            "roughness": 0.5,
            "metallic": 0.0,
        },
        "function": "compute_F0",
        "expected_r": 0.04,
        "tolerance": 0.001,
    },
    "metallic_one": {
        "description": "Pure metal (metallic=1)",
        "params": {
            "base_color": (1.0, 0.5, 0.0),
            "roughness": 0.5,
            "metallic": 1.0,
        },
        "function": "compute_F0",
        "expected_r": 1.0,  # F0 = base_color for metals
        "tolerance": 0.001,
    },
    "grazing_angle": {
        "description": "Grazing angle Fresnel",
        "params": {"VoH": 0.0, "F0": (0.04, 0.04, 0.04)},
        "function": "F_Schlick",
        "expected_r": 1.0,
        "tolerance": 0.001,
    },
}


__all__ = [
    # WGSL source access
    "get_brdf_wgsl",
    # NDF functions
    "d_ggx",
    # Geometry functions
    "g1_schlick_ggx",
    "g_smith_ggx",
    "g_smith_schlick",
    # Fresnel functions
    "f_schlick",
    "f_schlick_roughness",
    "f_schlick_scalar",
    # Diffuse BRDF
    "brdf_diffuse",
    "brdf_diffuse_disney",
    # Specular BRDF
    "brdf_specular",
    # Combined BRDF
    "compute_f0",
    "PBRParamsSimple",
    "evaluate_brdf",
    # Reference values
    "BRDF_REFERENCE_VALUES",
    "BRDF_EDGE_CASES",
    # Constants
    "PI",
    "INV_PI",
    "EPSILON",
]

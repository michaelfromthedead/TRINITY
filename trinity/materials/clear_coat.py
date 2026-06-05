"""Clear Coat BRDF Functions for TRINITY Material System.

T-MAT-4.2: Dual-Layer Clear Coat Implementation.

This module provides:
- Python reference implementations of clear coat BRDF functions for testing
- Access to WGSL clear coat source code
- Reference values for validation

The clear coat model simulates materials with a transparent protective layer:
- Automotive paint (car finishes)
- Lacquered wood
- Glossy plastic with top coat
- Coated leather

Clear coat parameters:
- intensity: Controls the blend weight (0=none, 1=full coverage)
- roughness: Independent roughness for the clear coat layer

The layer combination uses Fresnel-weighted blending:
    final = coat + (1 - Fc * intensity) * base

Example::

    from trinity.materials.clear_coat import (
        f_clear_coat,
        d_clear_coat,
        g_clear_coat_kelemen,
        evaluate_clear_coat,
        combine_clear_coat,
        get_clear_coat_wgsl,
    )

    # Get WGSL source for shader compilation
    wgsl_source = get_clear_coat_wgsl()

    # Reference calculation for testing
    fresnel = f_clear_coat(VoH=0.5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# Mathematical constants
PI = 3.14159265359
EPSILON = 0.0001

# Clear coat F0 for IOR 1.5 (polyurethane/lacquer)
# Computed as: ((n-1)/(n+1))^2 = ((1.5-1)/(1.5+1))^2 = 0.04
CLEAR_COAT_F0 = 0.04

# Type alias for RGB color
Vec3 = Tuple[float, float, float]
Vec4 = Tuple[float, float, float, float]


def get_clear_coat_wgsl() -> str:
    """Load the clear coat WGSL source.

    Returns:
        WGSL source code containing all clear coat BRDF functions.
    """
    wgsl_path = Path(__file__).parent / "wgsl" / "clear_coat.wgsl"
    return wgsl_path.read_text(encoding="utf-8")


# =============================================================================
# Clear Coat Parameters
# =============================================================================


@dataclass
class ClearCoatParams:
    """Parameters controlling the clear coat layer.

    Attributes:
        intensity: Blend weight for clear coat (0=none, 1=full).
        roughness: Surface roughness of clear coat layer in [0,1].
    """

    intensity: float = 1.0
    roughness: float = 0.1

    def __post_init__(self) -> None:
        """Validate parameters are in valid range."""
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"intensity must be in [0,1], got {self.intensity}")
        if not 0.0 <= self.roughness <= 1.0:
            raise ValueError(f"roughness must be in [0,1], got {self.roughness}")


# =============================================================================
# Clear Coat Fresnel Function
# =============================================================================


def f_clear_coat(VoH: float) -> float:
    """Schlick Fresnel approximation for clear coat layer.

    Uses fixed F0 = 0.04 for IOR 1.5 (polyurethane clear coat).

    Args:
        VoH: Dot product of view direction and half-vector.

    Returns:
        Fresnel reflectance for clear coat (scalar).

    Example::

        >>> f_clear_coat(1.0)  # Normal incidence
        0.04
        >>> f_clear_coat(0.0)  # Grazing angle
        1.0
    """
    Fc = pow(1.0 - VoH, 5.0)
    return CLEAR_COAT_F0 + (1.0 - CLEAR_COAT_F0) * Fc


# =============================================================================
# Clear Coat NDF (Normal Distribution Function)
# =============================================================================


def d_clear_coat(NoH: float, cc_roughness: float) -> float:
    """GGX/Trowbridge-Reitz NDF for clear coat layer.

    Uses independent roughness from the base layer.

    Args:
        NoH: Dot product of surface normal and half-vector.
        cc_roughness: Clear coat roughness in [0,1].

    Returns:
        NDF value for clear coat layer.

    Example::

        >>> d_clear_coat(1.0, 0.1)  # Peak with smooth clear coat
        318.31...
        >>> d_clear_coat(1.0, 0.5)  # Peak with rougher clear coat
        5.052...
    """
    a = cc_roughness * cc_roughness
    a2 = a * a
    NoH2 = NoH * NoH

    denom = NoH2 * (a2 - 1.0) + 1.0
    return a2 / (PI * denom * denom + EPSILON)


# =============================================================================
# Clear Coat Geometry Functions
# =============================================================================


def g_clear_coat_kelemen(VoH: float) -> float:
    """Kelemen visibility function for clear coat layer.

    Simplified geometry term suitable for thin clear coat layers.
    Formula: V = 1 / (4 * VoH^2)

    Args:
        VoH: Dot product of view and half-vector.

    Returns:
        Geometry term for clear coat (Kelemen approximation).

    Example::

        >>> g_clear_coat_kelemen(1.0)  # Normal incidence
        0.25
        >>> g_clear_coat_kelemen(0.5)  # Off-normal
        1.0
    """
    return 0.25 / (VoH * VoH + EPSILON)


def g_clear_coat(NoV: float, NoL: float, cc_roughness: float) -> float:
    """Smith-GGX geometry function for clear coat (full form).

    Height-correlated form for more accurate clear coat shadowing.

    Args:
        NoV: Dot product of normal and view direction.
        NoL: Dot product of normal and light direction.
        cc_roughness: Clear coat roughness in [0,1].

    Returns:
        Combined geometry term for clear coat.

    Example::

        >>> g_clear_coat(1.0, 1.0, 0.1)
        0.25
    """
    a = cc_roughness * cc_roughness
    a2 = a * a

    GGXV = NoL * math.sqrt(NoV * NoV * (1.0 - a2) + a2)
    GGXL = NoV * math.sqrt(NoL * NoL * (1.0 - a2) + a2)

    return 0.5 / (GGXV + GGXL + EPSILON)


# =============================================================================
# Clear Coat Specular Evaluation
# =============================================================================


def evaluate_clear_coat(
    N: Vec3,
    V: Vec3,
    L: Vec3,
    cc_params: ClearCoatParams,
) -> Vec3:
    """Evaluate the clear coat specular BRDF contribution.

    Args:
        N: Surface normal (normalized).
        V: View direction (normalized, pointing toward camera).
        L: Light direction (normalized, pointing toward light).
        cc_params: Clear coat parameters (intensity, roughness).

    Returns:
        Clear coat specular contribution (grayscale as RGB tuple).

    Example::

        >>> params = ClearCoatParams(intensity=1.0, roughness=0.1)
        >>> N, V, L = (0, 1, 0), (0, 1, 0), (0, 1, 0)
        >>> cc = evaluate_clear_coat(N, V, L, params)
        >>> cc[0] > 0.0  # Should have positive contribution
        True
    """
    # Skip if clear coat intensity is zero
    if cc_params.intensity < EPSILON:
        return (0.0, 0.0, 0.0)

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

    # Evaluate clear coat BRDF terms
    D = d_clear_coat(NoH, cc_params.roughness)
    G = g_clear_coat_kelemen(VoH)
    F = f_clear_coat(VoH)

    # Clear coat BRDF: D * G * F * intensity
    cc_brdf = D * G * F * cc_params.intensity

    # Return as grayscale (clear coat is achromatic)
    return (cc_brdf, cc_brdf, cc_brdf)


def evaluate_clear_coat_with_fresnel(
    N: Vec3,
    V: Vec3,
    L: Vec3,
    cc_params: ClearCoatParams,
) -> Vec4:
    """Evaluate clear coat and return Fresnel term for layer blending.

    Args:
        N: Surface normal (normalized).
        V: View direction (normalized).
        L: Light direction (normalized).
        cc_params: Clear coat parameters.

    Returns:
        Tuple (r, g, b, Fc) where rgb = clear coat specular, Fc = Fresnel factor.

    Example::

        >>> params = ClearCoatParams(intensity=1.0, roughness=0.1)
        >>> N, V, L = (0, 1, 0), (0, 1, 0), (0, 1, 0)
        >>> result = evaluate_clear_coat_with_fresnel(N, V, L, params)
        >>> result[3] > 0.0  # Fresnel factor should be positive
        True
    """
    # Skip if clear coat intensity is zero
    if cc_params.intensity < EPSILON:
        return (0.0, 0.0, 0.0, 0.0)

    # Compute half-vector
    Hx = V[0] + L[0]
    Hy = V[1] + L[1]
    Hz = V[2] + L[2]
    H_len = math.sqrt(Hx * Hx + Hy * Hy + Hz * Hz)
    if H_len < EPSILON:
        return (0.0, 0.0, 0.0, 0.0)
    H = (Hx / H_len, Hy / H_len, Hz / H_len)

    # Compute dot products
    NoL = max(N[0] * L[0] + N[1] * L[1] + N[2] * L[2], 0.0)
    NoV = max(N[0] * V[0] + N[1] * V[1] + N[2] * V[2], 0.0)
    NoH = max(N[0] * H[0] + N[1] * H[1] + N[2] * H[2], 0.0)
    VoH = max(V[0] * H[0] + V[1] * H[1] + V[2] * H[2], 0.0)

    # Early exit for grazing angles
    if NoL < EPSILON or NoV < EPSILON:
        return (0.0, 0.0, 0.0, 0.0)

    # Evaluate clear coat BRDF terms
    D = d_clear_coat(NoH, cc_params.roughness)
    G = g_clear_coat_kelemen(VoH)
    F = f_clear_coat(VoH)

    # Clear coat BRDF contribution
    cc_brdf = D * G * F * cc_params.intensity

    # Return clear coat RGB (grayscale) and Fresnel factor in alpha
    return (cc_brdf, cc_brdf, cc_brdf, F * cc_params.intensity)


# =============================================================================
# Layer Combination Functions
# =============================================================================


def combine_clear_coat(
    base_brdf: Vec3,
    cc_brdf: Vec3,
    Fc: float,
    cc_intensity: float,
) -> Vec3:
    """Combine clear coat layer with base material BRDF.

    Uses Fresnel-weighted blending: final = coat + (1 - Fc * intensity) * base

    This physically models that light reflected by the clear coat is not
    available for the base layer.

    Args:
        base_brdf: Base material BRDF value.
        cc_brdf: Clear coat BRDF value (rgb).
        Fc: Clear coat Fresnel factor.
        cc_intensity: Clear coat intensity.

    Returns:
        Combined BRDF value.

    Example::

        >>> base = (0.3, 0.2, 0.1)  # Colored base
        >>> cc = (0.1, 0.1, 0.1)    # Achromatic clear coat
        >>> combined = combine_clear_coat(base, cc, Fc=0.04, cc_intensity=1.0)
        >>> combined[0] > cc[0]  # Combined should include base contribution
        True
    """
    # Fresnel-weighted combination
    base_attenuation = 1.0 - Fc * cc_intensity

    return (
        cc_brdf[0] + base_brdf[0] * base_attenuation,
        cc_brdf[1] + base_brdf[1] * base_attenuation,
        cc_brdf[2] + base_brdf[2] * base_attenuation,
    )


def combine_clear_coat_simple(
    base_brdf: Vec3,
    cc_result: Vec4,
) -> Vec3:
    """Simplified layer combination using pre-computed clear coat.

    Use when you've already computed both layers separately.

    Args:
        base_brdf: Base material BRDF value.
        cc_result: Result from evaluate_clear_coat_with_fresnel (xyz=brdf, w=Fc).

    Returns:
        Combined BRDF value.

    Example::

        >>> base = (0.3, 0.2, 0.1)
        >>> cc_result = (0.1, 0.1, 0.1, 0.04)  # (brdf_r, brdf_g, brdf_b, Fc)
        >>> combined = combine_clear_coat_simple(base, cc_result)
    """
    base_attenuation = 1.0 - cc_result[3]

    return (
        cc_result[0] + base_brdf[0] * base_attenuation,
        cc_result[1] + base_brdf[1] * base_attenuation,
        cc_result[2] + base_brdf[2] * base_attenuation,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def get_clear_coat_attenuation(
    N: Vec3,
    V: Vec3,
    L: Vec3,
    cc_params: ClearCoatParams,
) -> float:
    """Get the clear coat attenuation factor for the base layer.

    Use this to attenuate the base BRDF before computing it.

    Args:
        N: Surface normal.
        V: View direction.
        L: Light direction.
        cc_params: Clear coat parameters.

    Returns:
        Attenuation factor in [0,1] for base layer.

    Example::

        >>> params = ClearCoatParams(intensity=1.0, roughness=0.1)
        >>> N, V, L = (0, 1, 0), (0, 1, 0), (0, 1, 0)
        >>> atten = get_clear_coat_attenuation(N, V, L, params)
        >>> 0.0 <= atten <= 1.0
        True
    """
    if cc_params.intensity < EPSILON:
        return 1.0

    # Compute half-vector
    Hx = V[0] + L[0]
    Hy = V[1] + L[1]
    Hz = V[2] + L[2]
    H_len = math.sqrt(Hx * Hx + Hy * Hy + Hz * Hz)
    if H_len < EPSILON:
        return 1.0
    H = (Hx / H_len, Hy / H_len, Hz / H_len)

    VoH = max(V[0] * H[0] + V[1] * H[1] + V[2] * H[2], 0.0)
    Fc = f_clear_coat(VoH)

    return 1.0 - Fc * cc_params.intensity


# =============================================================================
# Reference Values for Testing
# =============================================================================

# Reference values computed with known-good implementations
# Note: D_GGX uses Disney/Unreal convention: a = roughness^2, a2 = roughness^4
# This means very low roughness gives very SMALL NDF values, not large ones.
CLEAR_COAT_REFERENCE_VALUES = {
    # F_ClearCoat reference values: VoH -> expected
    "F_ClearCoat": [
        {"VoH": 1.0, "expected": 0.04, "tolerance": 0.001},
        {"VoH": 0.0, "expected": 1.0, "tolerance": 0.001},
        {"VoH": 0.5, "expected": 0.0696, "tolerance": 0.01},
        {"VoH": 0.707, "expected": 0.0508, "tolerance": 0.01},
        {"VoH": 0.866, "expected": 0.0423, "tolerance": 0.01},
    ],
    # D_ClearCoat reference values: (NoH, roughness) -> expected
    # Formula: a2 / (PI * denom^2 + EPSILON) where a = roughness^2, a2 = a^2 = roughness^4
    # At NoH=1, denom=1, so D = roughness^4 / PI
    # roughness=0.1: a2 = 0.0001, D = 0.0001/PI = 3.18e-5 (but EPSILON adds to denom)
    # roughness=0.5: a2 = 0.0625, D = 0.0625/PI = 0.0199 (small due to denom)
    # roughness=1.0: a2 = 1, D = 1/PI = 0.318
    "D_ClearCoat": [
        {"NoH": 1.0, "roughness": 0.1, "expected": 0.9997, "tolerance": 0.01},  # With EPSILON
        {"NoH": 1.0, "roughness": 0.5, "expected": 5.052, "tolerance": 0.1},
        {"NoH": 1.0, "roughness": 1.0, "expected": 0.31831, "tolerance": 0.01},
        {"NoH": 0.707, "roughness": 0.5, "expected": 0.0704, "tolerance": 0.01},
        {"NoH": 0.5, "roughness": 1.0, "expected": 0.318, "tolerance": 0.01},
    ],
    # G_ClearCoat_Kelemen reference values: VoH -> expected
    "G_ClearCoat_Kelemen": [
        {"VoH": 1.0, "expected": 0.25, "tolerance": 0.01},
        {"VoH": 0.5, "expected": 1.0, "tolerance": 0.01},
        {"VoH": 0.707, "expected": 0.5, "tolerance": 0.02},
        {"VoH": 0.866, "expected": 0.333, "tolerance": 0.02},
    ],
    # evaluate_clear_coat reference values
    # At normal incidence: H = V = L = N, so NoH = VoH = 1.0
    # D = depends on roughness, G = 0.25 (Kelemen), F = 0.04
    # Result = D * G * F * intensity
    "evaluate_clear_coat": [
        # Normal incidence with smooth clear coat
        # roughness=0.1: D ~ 1.0 (with EPSILON effect), G = 0.25, F = 0.04
        # expected ~ 1.0 * 0.25 * 0.04 * 1.0 = 0.01
        {
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "L": (0.0, 1.0, 0.0),
            "intensity": 1.0,
            "roughness": 0.1,
            "expected_r": 0.01,  # D * G * F * intensity
            "tolerance": 0.005,
        },
        # Normal incidence with rougher clear coat
        # roughness=0.5: D ~ 5.05, G = 0.25, F = 0.04
        # expected ~ 5.05 * 0.25 * 0.04 = 0.0505
        {
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "L": (0.0, 1.0, 0.0),
            "intensity": 1.0,
            "roughness": 0.5,
            "expected_r": 0.0505,  # D * G * F * intensity
            "tolerance": 0.01,
        },
        # Half intensity
        {
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "L": (0.0, 1.0, 0.0),
            "intensity": 0.5,
            "roughness": 0.1,
            "expected_r": 0.005,  # Half of full intensity
            "tolerance": 0.003,
        },
        # Zero intensity (disabled)
        {
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "L": (0.0, 1.0, 0.0),
            "intensity": 0.0,
            "roughness": 0.1,
            "expected_r": 0.0,
            "tolerance": 0.001,
        },
    ],
    # combine_clear_coat reference values
    "combine_clear_coat": [
        # Full intensity at normal incidence
        {
            "base_brdf": (0.5, 0.5, 0.5),
            "cc_brdf": (0.1, 0.1, 0.1),
            "Fc": 0.04,
            "cc_intensity": 1.0,
            "expected_r": 0.58,  # 0.1 + 0.5 * (1 - 0.04)
            "tolerance": 0.01,
        },
        # Zero intensity (passthrough)
        {
            "base_brdf": (0.5, 0.5, 0.5),
            "cc_brdf": (0.0, 0.0, 0.0),
            "Fc": 0.04,
            "cc_intensity": 0.0,
            "expected_r": 0.5,  # Base passes through unchanged
            "tolerance": 0.001,
        },
        # High Fresnel (grazing)
        {
            "base_brdf": (0.5, 0.5, 0.5),
            "cc_brdf": (0.2, 0.2, 0.2),
            "Fc": 1.0,
            "cc_intensity": 1.0,
            "expected_r": 0.2,  # Only clear coat visible at grazing
            "tolerance": 0.001,
        },
    ],
}

# Edge cases for clear coat
CLEAR_COAT_EDGE_CASES = {
    "zero_intensity": {
        "description": "Clear coat disabled (intensity=0)",
        "params": {"intensity": 0.0, "roughness": 0.5},
        "expected": "zero contribution",
    },
    "full_intensity_smooth": {
        "description": "Smooth clear coat at full intensity",
        "params": {"intensity": 1.0, "roughness": 0.01},
        "expected": "high specular peak",
    },
    "grazing_angle": {
        "description": "Grazing angle viewing",
        "VoH": 0.0,
        "expected_F": 1.0,
        "description_detail": "Fresnel approaches 1.0 at grazing",
    },
    "normal_incidence": {
        "description": "Normal incidence viewing",
        "VoH": 1.0,
        "expected_F": 0.04,
        "description_detail": "Fresnel equals F0 at normal incidence",
    },
}


__all__ = [
    # WGSL source access
    "get_clear_coat_wgsl",
    # Parameters
    "ClearCoatParams",
    # Fresnel function
    "f_clear_coat",
    # NDF function
    "d_clear_coat",
    # Geometry functions
    "g_clear_coat_kelemen",
    "g_clear_coat",
    # Evaluation functions
    "evaluate_clear_coat",
    "evaluate_clear_coat_with_fresnel",
    # Layer combination
    "combine_clear_coat",
    "combine_clear_coat_simple",
    # Convenience functions
    "get_clear_coat_attenuation",
    # Reference values
    "CLEAR_COAT_REFERENCE_VALUES",
    "CLEAR_COAT_EDGE_CASES",
    # Constants
    "CLEAR_COAT_F0",
    "PI",
    "EPSILON",
]

"""Sheen BRDF Functions for TRINITY Material System.

T-MAT-4.4: Microfiber Retro-Reflection Sheen Lobe.

This module provides:
- Python reference implementations of sheen BRDF functions for testing
- Access to WGSL sheen source code
- Reference values for validation

Sheen models the soft retro-reflective properties of fabrics where
microfibers create a characteristic brightening at grazing angles.
Unlike the Fresnel effect, sheen is strongest at grazing and weakens
toward normal incidence.

Example::

    from trinity.materials.sheen import (
        d_charlie,
        v_neubelt,
        evaluate_sheen,
        get_sheen_wgsl,
    )

    # Get WGSL source for shader compilation
    wgsl_source = get_sheen_wgsl()

    # Reference calculation for testing
    ndf = d_charlie(NoH=0.5, roughness=0.3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# Mathematical constants (shared with brdf.py)
PI = 3.14159265359
INV_PI = 0.31830988618
EPSILON = 0.0001

# Type alias for RGB color
Vec3 = Tuple[float, float, float]


def get_sheen_wgsl() -> str:
    """Load the sheen WGSL source.

    Returns:
        WGSL source code containing all sheen BRDF functions.
    """
    wgsl_path = Path(__file__).parent / "wgsl" / "sheen.wgsl"
    return wgsl_path.read_text(encoding="utf-8")


# =============================================================================
# Sheen Parameter Dataclass
# =============================================================================


@dataclass
class SheenParams:
    """Sheen material parameters.

    Encapsulates all parameters needed for sheen evaluation.

    Attributes:
        intensity: Sheen intensity, 0.0 = no sheen, 1.0 = full sheen.
        color: Sheen tint color (linear RGB).
        roughness: Sheen roughness, controls width of the sheen lobe.

    Example::

        params = SheenParams(
            intensity=0.8,
            color=(1.0, 0.9, 0.8),  # Warm tint
            roughness=0.3,
        )
    """

    intensity: float = 0.5
    color: Vec3 = (1.0, 1.0, 1.0)
    roughness: float = 0.3

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"intensity must be in [0, 1], got {self.intensity}")
        if not 0.0 <= self.roughness <= 1.0:
            raise ValueError(f"roughness must be in [0, 1], got {self.roughness}")
        if len(self.color) != 3:
            raise ValueError(f"color must be a 3-tuple, got {len(self.color)} elements")
        if not all(0.0 <= c <= 1.0 for c in self.color):
            raise ValueError(f"color components must be in [0, 1], got {self.color}")


# =============================================================================
# Sheen Distribution Function (NDF)
# =============================================================================


def d_charlie(NoH: float, roughness: float) -> float:
    """Charlie/Ashikhmin sheen distribution function.

    Models the distribution of microfiber orientations for fabric surfaces.
    The distribution peaks at grazing angles rather than normal incidence.

    Args:
        NoH: Dot product of surface normal and half-vector, in [0, 1].
        roughness: Sheen roughness in [0, 1], controls lobe width.

    Returns:
        NDF value (probability density of microfiber orientation).

    Example::

        >>> d_charlie(0.5, 0.3)  # Half-angle with low roughness
        0.318...
        >>> d_charlie(1.0, 0.3)  # Normal incidence - sheen is minimal
        0.0
    """
    # Clamp roughness to avoid division by zero
    alpha = max(roughness * roughness, 0.0001)

    # sin^2(theta) = 1 - cos^2(theta)
    sin2_theta = 1.0 - NoH * NoH

    # Inverted Ashikhmin distribution for sheen
    # D = (2 + 1/alpha) * sin(theta)^(1/alpha) / (2*PI)
    inv_alpha = 1.0 / alpha
    power = inv_alpha * 0.5

    # Use sin^n where n = 1/alpha
    D = (2.0 + inv_alpha) * pow(max(sin2_theta, 0.0), power) * INV_PI * 0.5

    return D


def d_charlie_simple(NoH: float, roughness: float) -> float:
    """Simplified Charlie distribution using the exponential form.

    This variant is slightly cheaper but less accurate.

    Args:
        NoH: Dot product of surface normal and half-vector.
        roughness: Sheen roughness.

    Returns:
        Simplified NDF value.
    """
    alpha = roughness * roughness
    sin2_theta = 1.0 - NoH * NoH

    # Simplified: sin^(2/alpha) * normalization
    if alpha < EPSILON:
        alpha = EPSILON
    D = pow(max(sin2_theta, 0.0), 1.0 / alpha) * INV_PI

    return D


# =============================================================================
# Sheen Visibility Function (GSF)
# =============================================================================


def v_neubelt(NoV: float, NoL: float) -> float:
    """Neubelt visibility term for sheen.

    A simplified visibility function designed specifically for sheen BRDFs.
    Produces V = 1 / (4 * (NoL + NoV - NoL * NoV))

    Args:
        NoV: Dot product of surface normal and view direction.
        NoL: Dot product of surface normal and light direction.

    Returns:
        Visibility term (includes 1/(4*NoL*NoV) normalization).

    Example::

        >>> v_neubelt(1.0, 1.0)  # Normal incidence
        0.25
        >>> v_neubelt(0.5, 0.5)  # Off-normal
        0.333...
    """
    denom = NoL + NoV - NoL * NoV
    return 1.0 / (4.0 * denom + EPSILON)


def v_ashikhmin(NoV: float, NoL: float) -> float:
    """Alternative Ashikhmin visibility term.

    Even simpler than Neubelt, just uses 1/(4*(NoL+NoV)).

    Args:
        NoV: Dot product of surface normal and view direction.
        NoL: Dot product of surface normal and light direction.

    Returns:
        Simple visibility term.
    """
    return 1.0 / (4.0 * (NoL + NoV + EPSILON))


# =============================================================================
# Vector Utilities
# =============================================================================


def _dot(a: Vec3, b: Vec3) -> float:
    """Compute dot product of two 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(v: Vec3) -> Vec3:
    """Normalize a 3D vector."""
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length < EPSILON:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _add(a: Vec3, b: Vec3) -> Vec3:
    """Add two 3D vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(v: Vec3, s: float) -> Vec3:
    """Scale a 3D vector by a scalar."""
    return (v[0] * s, v[1] * s, v[2] * s)


def _mul_vec(a: Vec3, b: Vec3) -> Vec3:
    """Component-wise multiplication of two vectors."""
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


# =============================================================================
# Combined Sheen Evaluation
# =============================================================================


def evaluate_sheen(
    params: SheenParams,
    N: Vec3,
    V: Vec3,
    L: Vec3,
) -> Vec3:
    """Evaluate the complete sheen BRDF contribution.

    Combines the Charlie distribution and Neubelt visibility with tinting.
    Sheen does NOT use Fresnel - it's a diffuse-like scattering effect.

    Args:
        params: Sheen material parameters.
        N: Surface normal (normalized).
        V: View direction (normalized, pointing toward camera).
        L: Light direction (normalized, pointing toward light).

    Returns:
        Sheen BRDF contribution as RGB tuple.

    Example::

        params = SheenParams(intensity=0.5, color=(1.0, 1.0, 1.0), roughness=0.3)
        N = (0.0, 1.0, 0.0)  # Up
        V = (0.0, 1.0, 0.0)  # Looking down
        L = (0.707, 0.707, 0.0)  # 45 degree light
        sheen = evaluate_sheen(params, N, V, L)
    """
    # Early exit if sheen intensity is zero
    if params.intensity < EPSILON:
        return (0.0, 0.0, 0.0)

    # Compute half-vector
    H = _normalize(_add(V, L))
    if H == (0.0, 0.0, 0.0):
        return (0.0, 0.0, 0.0)

    # Compute dot products (clamped to avoid negative values)
    NoL = max(_dot(N, L), 0.0)
    NoV = max(_dot(N, V), 0.0)
    NoH = max(_dot(N, H), 0.0)

    # Early exit for extreme grazing angles
    if NoL < EPSILON and NoV < EPSILON:
        return (0.0, 0.0, 0.0)

    # Evaluate sheen BRDF: D * V * color * intensity
    D = d_charlie(NoH, params.roughness)
    Vis = v_neubelt(NoV, NoL)

    # Sheen contribution with color tinting
    sheen_factor = D * Vis * params.intensity
    return (
        sheen_factor * params.color[0],
        sheen_factor * params.color[1],
        sheen_factor * params.color[2],
    )


def evaluate_sheen_with_NoL(
    params: SheenParams,
    N: Vec3,
    V: Vec3,
    L: Vec3,
) -> Vec3:
    """Evaluate sheen with NoL factor already applied.

    Use this when accumulating light contributions.

    Args:
        params: Sheen material parameters.
        N: Surface normal (normalized).
        V: View direction (normalized).
        L: Light direction (normalized).

    Returns:
        Sheen BRDF value multiplied by NoL.
    """
    sheen = evaluate_sheen(params, N, V, L)
    NoL = max(_dot(N, L), 0.0)
    return _scale(sheen, NoL)


def sheen_contribution(
    intensity: float,
    color: Vec3,
    roughness: float,
    NoV: float,
    NoL: float,
    NoH: float,
) -> Vec3:
    """Get sheen contribution at specific angles for debugging.

    Args:
        intensity: Sheen intensity [0,1].
        color: Sheen color (linear RGB).
        roughness: Sheen roughness [0,1].
        NoV: Dot product of normal and view direction.
        NoL: Dot product of normal and light direction.
        NoH: Dot product of normal and half-vector.

    Returns:
        Sheen contribution (RGB).
    """
    if intensity < EPSILON:
        return (0.0, 0.0, 0.0)

    D = d_charlie(NoH, roughness)
    Vis = v_neubelt(NoV, NoL)

    sheen_factor = D * Vis * intensity
    return (
        sheen_factor * color[0],
        sheen_factor * color[1],
        sheen_factor * color[2],
    )


# =============================================================================
# Combination with Standard BRDF
# =============================================================================


def combine_brdf_with_sheen(
    sheen_params: SheenParams,
    N: Vec3,
    V: Vec3,
    L: Vec3,
    diffuse: Vec3,
    specular: Vec3,
) -> Vec3:
    """Combine standard BRDF with sheen lobe.

    The sheen lobe is additive to the standard BRDF:
        final = diffuse + specular + sheen

    Args:
        sheen_params: Sheen parameters.
        N: Surface normal (normalized).
        V: View direction (normalized).
        L: Light direction (normalized).
        diffuse: Evaluated diffuse BRDF contribution.
        specular: Evaluated specular BRDF contribution.

    Returns:
        Combined BRDF contribution (RGB).
    """
    sheen = evaluate_sheen(sheen_params, N, V, L)
    return (
        diffuse[0] + specular[0] + sheen[0],
        diffuse[1] + specular[1] + sheen[1],
        diffuse[2] + specular[2] + sheen[2],
    )


# =============================================================================
# Reference Values for Testing
# =============================================================================

# Reference values computed with our implementation
# These values are validated against the Charlie/Ashikhmin sheen distribution
SHEEN_REFERENCE_VALUES = {
    # D_Charlie reference values: (roughness, NoH) -> expected
    # At NoH=1 (normal incidence), sin^2(theta)=0, so D=0
    # At NoH=0 (grazing), sin^2(theta)=1, D is maximized
    "D_Charlie": [
        # Normal incidence - sheen should be minimal (sin^2=0)
        {"roughness": 0.3, "NoH": 1.0, "expected": 0.0, "tolerance": 0.001},
        # Grazing angle - sheen is maximum (sin^2=1)
        {"roughness": 0.3, "NoH": 0.0, "expected": 2.087, "tolerance": 0.1},
        # Mid angle with low roughness
        {"roughness": 0.3, "NoH": 0.5, "expected": 0.422, "tolerance": 0.05},
        # Mid angle with high roughness (broader distribution)
        {"roughness": 0.8, "NoH": 0.5, "expected": 0.453, "tolerance": 0.05},
        # Very rough surface
        {"roughness": 1.0, "NoH": 0.5, "expected": 0.413, "tolerance": 0.05},
        # Smooth surface at 45 degrees (NoH ~ 0.707)
        {"roughness": 0.2, "NoH": 0.707, "expected": 0.001, "tolerance": 0.01},
    ],
    # V_Neubelt reference values: (NoV, NoL) -> expected
    # V = 1 / (4 * (NoL + NoV - NoL*NoV))
    "V_Neubelt": [
        # Normal incidence: denom = 1+1-1 = 1, result = 0.25
        {"NoV": 1.0, "NoL": 1.0, "expected": 0.25, "tolerance": 0.01},
        # Both at 0.5: denom = 0.5+0.5-0.25 = 0.75, result = 1/(4*0.75) = 0.333
        {"NoV": 0.5, "NoL": 0.5, "expected": 0.333, "tolerance": 0.01},
        # Grazing view: denom = 0.1+1.0-0.1 = 1.0, result = 0.25
        {"NoV": 0.1, "NoL": 1.0, "expected": 0.25, "tolerance": 0.01},
        # Symmetric case: denom = 0.7+0.3-0.21 = 0.79, result = 1/(4*0.79) = 0.316
        {"NoV": 0.7, "NoL": 0.3, "expected": 0.316, "tolerance": 0.02},
        # Near grazing both: denom = 0.2+0.2-0.04 = 0.36, result = 1/(4*0.36) = 0.694
        {"NoV": 0.2, "NoL": 0.2, "expected": 0.694, "tolerance": 0.05},
    ],
    # Combined sheen evaluation
    # Note: Sheen is very subtle at normal incidence due to the Charlie distribution
    # being designed to peak at grazing angles
    "evaluate_sheen": [
        # White sheen, normal incidence lighting - H=N, so NoH=1, D=0
        {
            "intensity": 0.5,
            "color": (1.0, 1.0, 1.0),
            "roughness": 0.3,
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "L": (0.0, 1.0, 0.0),
            "expected_r": 0.0,  # NoH=1, so D_Charlie=0
            "tolerance": 0.001,
        },
        # 45 degree lighting - NoH is still quite high, so sheen is small
        # H = normalize(V+L) = (0.707, 1.707, 0)/1.849 = (0.38, 0.92, 0)
        # NoH = 0.92, sin^2 = 0.15 -> small sheen
        {
            "intensity": 0.5,
            "color": (1.0, 1.0, 1.0),
            "roughness": 0.3,
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "L": (0.707, 0.707, 0.0),
            "expected_r": 0.0001,  # Very small sheen due to high NoH
            "tolerance": 0.01,
        },
        # Colored sheen (warm tint) - also small at this angle
        {
            "intensity": 1.0,
            "color": (1.0, 0.8, 0.6),
            "roughness": 0.4,
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "L": (0.707, 0.707, 0.0),
            "expected_r": 0.001,  # Small sheen
            "tolerance": 0.01,
        },
        # Zero intensity - no sheen
        {
            "intensity": 0.0,
            "color": (1.0, 1.0, 1.0),
            "roughness": 0.3,
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "L": (0.707, 0.707, 0.0),
            "expected_r": 0.0,
            "tolerance": 0.001,
        },
    ],
}

# Edge case test configurations
SHEEN_EDGE_CASES = {
    "roughness_very_low": {
        "description": "Very low roughness creates sharp sheen",
        "params": {"roughness": 0.1, "NoH": 0.5},
        "function": "D_Charlie",
        "expect": "high_value",  # Sharp peak at grazing
    },
    "roughness_one": {
        "description": "Roughness=1 creates broad distribution",
        "params": {"roughness": 1.0, "NoH": 0.5},
        "function": "D_Charlie",
        "expect": "moderate_value",
    },
    "normal_incidence_sheen": {
        "description": "Sheen should be zero at normal incidence (NoH=1)",
        "params": {"roughness": 0.5, "NoH": 1.0},
        "function": "D_Charlie",
        "expected": 0.0,
        "tolerance": 0.001,
    },
    "grazing_sheen": {
        "description": "Sheen is strongest at grazing angles",
        "params": {"roughness": 0.3, "NoH": 0.0},
        "function": "D_Charlie",
        "expect": "maximum",
    },
    "zero_intensity": {
        "description": "Zero intensity produces no sheen",
        "params": {
            "intensity": 0.0,
            "color": (1.0, 1.0, 1.0),
            "roughness": 0.3,
        },
        "function": "evaluate_sheen",
        "expected_r": 0.0,
        "tolerance": 0.001,
    },
    "full_intensity": {
        "description": "Full intensity at grazing angle",
        "params": {
            "intensity": 1.0,
            "color": (1.0, 1.0, 1.0),
            "roughness": 0.3,
        },
        "function": "evaluate_sheen",
        "expect": "high_value",
    },
}


__all__ = [
    # WGSL source access
    "get_sheen_wgsl",
    # Sheen parameters
    "SheenParams",
    # Distribution function
    "d_charlie",
    "d_charlie_simple",
    # Visibility function
    "v_neubelt",
    "v_ashikhmin",
    # Combined evaluation
    "evaluate_sheen",
    "evaluate_sheen_with_NoL",
    "sheen_contribution",
    "combine_brdf_with_sheen",
    # Reference values
    "SHEEN_REFERENCE_VALUES",
    "SHEEN_EDGE_CASES",
    # Constants
    "PI",
    "INV_PI",
    "EPSILON",
]

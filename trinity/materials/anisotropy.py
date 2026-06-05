"""Anisotropic BRDF Functions for TRINITY Material System.

T-MAT-4.3: Anisotropic GGX Implementation.

This module provides:
- Python reference implementations of anisotropic BRDF functions for testing
- Access to WGSL anisotropy source code
- Reference values for validation

The anisotropic BRDF functions implement:
- Anisotropic GGX Normal Distribution Function
- Anisotropic Smith-GGX Geometry Function
- Tangent rotation for anisotropy direction
- Complete anisotropic BRDF evaluation

Example::

    from trinity.materials.anisotropy import (
        compute_aniso_alphas,
        d_ggx_anisotropic,
        evaluate_aniso_brdf,
        get_anisotropy_wgsl,
    )

    # Get WGSL source for shader compilation
    wgsl_source = get_anisotropy_wgsl()

    # Reference calculation for testing
    alphas = compute_aniso_alphas(roughness=0.5, anisotropy=0.8)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# Mathematical constants (matching brdf.py)
PI = 3.14159265359
INV_PI = 0.31830988618
EPSILON = 0.0001

# Type alias for RGB color and 2D/3D vectors
Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]


def get_anisotropy_wgsl() -> str:
    """Load the anisotropy WGSL source.

    Returns:
        WGSL source code containing anisotropic BRDF functions.
    """
    wgsl_path = Path(__file__).parent / "wgsl" / "anisotropy.wgsl"
    return wgsl_path.read_text(encoding="utf-8")


# =============================================================================
# Anisotropy Parameters
# =============================================================================


@dataclass
class AnisotropyParams:
    """Parameters for anisotropic BRDF evaluation.

    Attributes:
        strength: Anisotropy strength in [0, 1]. 0 = isotropic, 1 = max stretch.
        direction: Rotation angle in radians. 0 = along tangent, PI/2 = along bitangent.
    """

    strength: float = 0.0
    direction: float = 0.0

    def __post_init__(self) -> None:
        """Validate and clamp parameters."""
        self.strength = max(0.0, min(1.0, self.strength))


# =============================================================================
# Alpha Computation
# =============================================================================


def compute_aniso_alphas(roughness: float, anisotropy: float) -> Vec2:
    """Compute anisotropic alpha values from roughness and anisotropy strength.

    The formula produces directionally different roughness values:
      alpha_x = a * (1 + anisotropy)
      alpha_y = a * (1 - anisotropy)
    where a = roughness^2 (Disney convention).

    Args:
        roughness: Base surface roughness in [0, 1].
        anisotropy: Anisotropy strength in [0, 1].

    Returns:
        Tuple of (alpha_x, alpha_y).

    Example::

        >>> compute_aniso_alphas(0.5, 0.0)  # Isotropic
        (0.0625, 0.0625)
        >>> compute_aniso_alphas(0.5, 1.0)  # Max anisotropy
        (0.125, 0.0001)
    """
    # Remap roughness to alpha (Disney convention)
    a = roughness * roughness

    # Clamp anisotropy
    aniso = max(0.0, min(1.0, anisotropy))

    # Compute directional alphas
    alpha_x = a * (1.0 + aniso)
    alpha_y = a * (1.0 - aniso)

    # Ensure minimum to prevent division issues
    return (max(alpha_x, EPSILON), max(alpha_y, EPSILON))


# =============================================================================
# Tangent Rotation
# =============================================================================


def rotate_tangent(tangent: Vec3, bitangent: Vec3, angle: float) -> Vec3:
    """Rotate tangent vector by anisotropy direction angle.

    Args:
        tangent: Original tangent vector (normalized).
        bitangent: Original bitangent vector (normalized).
        angle: Rotation angle in radians.

    Returns:
        Rotated tangent vector.
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        tangent[0] * cos_a + bitangent[0] * sin_a,
        tangent[1] * cos_a + bitangent[1] * sin_a,
        tangent[2] * cos_a + bitangent[2] * sin_a,
    )


def rotate_bitangent(tangent: Vec3, bitangent: Vec3, angle: float) -> Vec3:
    """Rotate bitangent vector by anisotropy direction angle.

    Args:
        tangent: Original tangent vector (normalized).
        bitangent: Original bitangent vector (normalized).
        angle: Rotation angle in radians.

    Returns:
        Rotated bitangent vector.
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        -tangent[0] * sin_a + bitangent[0] * cos_a,
        -tangent[1] * sin_a + bitangent[1] * cos_a,
        -tangent[2] * sin_a + bitangent[2] * cos_a,
    )


# =============================================================================
# Anisotropic Normal Distribution Function
# =============================================================================


def d_ggx_anisotropic(
    NoH: float,
    ToH: float,
    BoH: float,
    alpha_x: float,
    alpha_y: float,
) -> float:
    """Anisotropic GGX/Trowbridge-Reitz Normal Distribution Function.

    Models elliptically stretched microfacet orientations.

    The formula is:
        D = 1 / (PI * alpha_x * alpha_y * denom^2)
    where:
        denom = (ToH/alpha_x)^2 + (BoH/alpha_y)^2 + NoH^2

    Args:
        NoH: Dot product of surface normal and half-vector.
        ToH: Dot product of tangent and half-vector.
        BoH: Dot product of bitangent and half-vector.
        alpha_x: Roughness along tangent direction.
        alpha_y: Roughness along bitangent direction.

    Returns:
        Anisotropic NDF value.

    Example::

        >>> d_ggx_anisotropic(1.0, 0.0, 0.0, 0.1, 0.1)  # Isotropic case
        31.83...
    """
    ax = max(alpha_x, EPSILON)
    ay = max(alpha_y, EPSILON)

    ax2 = ax * ax
    ay2 = ay * ay

    term_x = ToH * ToH / ax2
    term_y = BoH * BoH / ay2
    term_n = NoH * NoH

    denom = term_x + term_y + term_n

    return 1.0 / (PI * ax * ay * denom * denom + EPSILON)


# =============================================================================
# Anisotropic Geometry Function
# =============================================================================


def g1_ggx_anisotropic(
    NoV: float,
    ToV: float,
    BoV: float,
    alpha_x: float,
    alpha_y: float,
) -> float:
    """Anisotropic Smith-GGX G1 geometry function.

    Args:
        NoV: Dot product of normal and view/light direction.
        ToV: Dot product of tangent and view/light direction.
        BoV: Dot product of bitangent and view/light direction.
        alpha_x: Roughness along tangent direction.
        alpha_y: Roughness along bitangent direction.

    Returns:
        G1 geometry term.
    """
    ax2 = alpha_x * alpha_x
    ay2 = alpha_y * alpha_y

    projected_roughness = math.sqrt(ToV * ToV * ax2 + BoV * BoV * ay2)

    lambda_term = (-1.0 + math.sqrt(1.0 + projected_roughness * projected_roughness / (NoV * NoV + EPSILON))) / 2.0

    return 1.0 / (1.0 + lambda_term + EPSILON)


def g_smith_ggx_anisotropic(
    NoV: float,
    NoL: float,
    ToV: float,
    BoV: float,
    ToL: float,
    BoL: float,
    alpha_x: float,
    alpha_y: float,
) -> float:
    """Anisotropic Smith-GGX combined geometry function.

    Height-correlated form combining view and light masking/shadowing.
    Includes the 1/(4*NoV*NoL) factor.

    Args:
        NoV: Dot product of normal and view direction.
        NoL: Dot product of normal and light direction.
        ToV: Dot product of tangent and view direction.
        BoV: Dot product of bitangent and view direction.
        ToL: Dot product of tangent and light direction.
        BoL: Dot product of bitangent and light direction.
        alpha_x: Roughness along tangent direction.
        alpha_y: Roughness along bitangent direction.

    Returns:
        Combined geometry term.
    """
    ax2 = alpha_x * alpha_x
    ay2 = alpha_y * alpha_y

    lambdaV = NoL * math.sqrt(ax2 * ToV * ToV + ay2 * BoV * BoV + NoV * NoV)
    lambdaL = NoV * math.sqrt(ax2 * ToL * ToL + ay2 * BoL * BoL + NoL * NoL)

    return 0.5 / (lambdaV + lambdaL + EPSILON)


# =============================================================================
# Fresnel Function (local copy for self-contained module)
# =============================================================================


def f_schlick(VoH: float, F0: Vec3) -> Vec3:
    """Schlick Fresnel approximation.

    Args:
        VoH: Dot product of view direction and half-vector.
        F0: Reflectance at normal incidence.

    Returns:
        Fresnel reflectance as RGB tuple.
    """
    Fc = pow(1.0 - VoH, 5.0)
    return (
        F0[0] + (1.0 - F0[0]) * Fc,
        F0[1] + (1.0 - F0[1]) * Fc,
        F0[2] + (1.0 - F0[2]) * Fc,
    )


# =============================================================================
# Vector Math Utilities
# =============================================================================


def _dot(a: Vec3, b: Vec3) -> float:
    """Dot product of two vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(v: Vec3) -> Vec3:
    """Normalize a vector."""
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length < EPSILON:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _add(a: Vec3, b: Vec3) -> Vec3:
    """Add two vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


# =============================================================================
# Complete Anisotropic BRDF
# =============================================================================


def evaluate_aniso_brdf(
    N: Vec3,
    V: Vec3,
    L: Vec3,
    T: Vec3,
    B: Vec3,
    roughness: float,
    anisotropy_params: AnisotropyParams,
    F0: Vec3,
) -> Vec3:
    """Evaluate complete anisotropic specular BRDF.

    Combines anisotropic NDF, geometry, and Fresnel terms.

    Args:
        N: Surface normal (normalized).
        V: View direction (normalized, toward camera).
        L: Light direction (normalized, toward light).
        T: Tangent vector (normalized).
        B: Bitangent vector (normalized).
        roughness: Base surface roughness.
        anisotropy_params: Anisotropy strength and direction.
        F0: Specular reflectance at normal incidence.

    Returns:
        Anisotropic specular BRDF contribution (RGB).

    Example::

        >>> N = (0.0, 1.0, 0.0)
        >>> V = (0.0, 1.0, 0.0)
        >>> L = (0.0, 1.0, 0.0)
        >>> T = (1.0, 0.0, 0.0)
        >>> B = (0.0, 0.0, 1.0)
        >>> params = AnisotropyParams(strength=0.5, direction=0.0)
        >>> result = evaluate_aniso_brdf(N, V, L, T, B, 0.5, params, (0.04, 0.04, 0.04))
    """
    NoL = max(_dot(N, L), 0.0)
    NoV = max(_dot(N, V), 0.0)

    if NoL < EPSILON or NoV < EPSILON:
        return (0.0, 0.0, 0.0)

    # Compute half-vector
    H = _normalize(_add(V, L))
    NoH = max(_dot(N, H), 0.0)
    VoH = max(_dot(V, H), 0.0)

    # Rotate tangent basis
    rotated_T = rotate_tangent(T, B, anisotropy_params.direction)
    rotated_B = rotate_bitangent(T, B, anisotropy_params.direction)

    # Compute dot products with rotated basis
    ToH = _dot(rotated_T, H)
    BoH = _dot(rotated_B, H)
    ToV = _dot(rotated_T, V)
    BoV = _dot(rotated_B, V)
    ToL = _dot(rotated_T, L)
    BoL = _dot(rotated_B, L)

    # Compute anisotropic alphas
    alphas = compute_aniso_alphas(roughness, anisotropy_params.strength)
    alpha_x, alpha_y = alphas

    # Evaluate BRDF terms
    D = d_ggx_anisotropic(NoH, ToH, BoH, alpha_x, alpha_y)
    G = g_smith_ggx_anisotropic(NoV, NoL, ToV, BoV, ToL, BoL, alpha_x, alpha_y)
    F = f_schlick(VoH, F0)

    # Cook-Torrance: D * G * F
    return (D * G * F[0], D * G * F[1], D * G * F[2])


# =============================================================================
# Reference Values for Testing
# =============================================================================


# Reference values for validation
# Note: compute_aniso_alphas uses a = roughness^2 (Disney convention)
# So for roughness=0.5: a = 0.25
# alpha_x = a * (1 + aniso) = 0.25 * (1 + aniso)
# alpha_y = a * (1 - aniso) = 0.25 * (1 - aniso)
ANISOTROPY_REFERENCE_VALUES = {
    # compute_aniso_alphas: (roughness, anisotropy) -> (alpha_x, alpha_y)
    # roughness=0.5 -> a=0.25
    "compute_aniso_alphas": [
        # Isotropic case (anisotropy=0): alpha_x = alpha_y = 0.25
        {"roughness": 0.5, "anisotropy": 0.0, "expected_x": 0.25, "expected_y": 0.25, "tolerance": 0.001},
        # Low anisotropy: alpha_x = 0.25*1.3=0.325, alpha_y = 0.25*0.7=0.175
        {"roughness": 0.5, "anisotropy": 0.3, "expected_x": 0.325, "expected_y": 0.175, "tolerance": 0.001},
        # High anisotropy: alpha_x = 0.25*1.8=0.45, alpha_y = 0.25*0.2=0.05
        {"roughness": 0.5, "anisotropy": 0.8, "expected_x": 0.45, "expected_y": 0.05, "tolerance": 0.001},
        # Max anisotropy: alpha_x = 0.25*2=0.5, alpha_y clamped to EPSILON
        {"roughness": 0.5, "anisotropy": 1.0, "expected_x": 0.5, "expected_y": EPSILON, "tolerance": 0.0001},
        # Rough surface (roughness=1.0 -> a=1.0): alpha_x = 1.5, alpha_y = 0.5
        {"roughness": 1.0, "anisotropy": 0.5, "expected_x": 1.5, "expected_y": 0.5, "tolerance": 0.01},
        # Smooth surface (roughness=0.2 -> a=0.04): alpha_x = 0.06, alpha_y = 0.02
        {"roughness": 0.2, "anisotropy": 0.5, "expected_x": 0.06, "expected_y": 0.02, "tolerance": 0.001},
    ],
    # D_GGX_Anisotropic: (NoH, ToH, BoH, alpha_x, alpha_y) -> D
    # Formula: D = 1 / (PI * ax * ay * denom^2) where denom = ToH^2/ax^2 + BoH^2/ay^2 + NoH^2
    "D_GGX_Anisotropic": [
        # Isotropic at normal incidence: D = 1/(PI*0.1*0.1*1) = 31.83
        {"NoH": 1.0, "ToH": 0.0, "BoH": 0.0, "alpha_x": 0.1, "alpha_y": 0.1, "expected": 31.83, "tolerance": 0.5},
        # Anisotropic at normal incidence: D = 1/(PI*0.2*0.1*1) = 15.91
        {"NoH": 1.0, "ToH": 0.0, "BoH": 0.0, "alpha_x": 0.2, "alpha_y": 0.1, "expected": 15.91, "tolerance": 0.5},
        # Off-center with equal alphas: compute actual value
        # denom = 0.3^2/0.3^2 + 0.2^2/0.3^2 + 0.9^2 = 1 + 0.444 + 0.81 = 2.254
        # D = 1/(PI*0.3*0.3*2.254^2) = 1/(0.283*5.08) = 0.696
        {"NoH": 0.9, "ToH": 0.3, "BoH": 0.2, "alpha_x": 0.3, "alpha_y": 0.3, "expected": 0.70, "tolerance": 0.05},
        # Stretched along tangent: different formula values
        # denom = 0.3^2/0.5^2 + 0.1^2/0.1^2 + 0.9^2 = 0.36 + 1 + 0.81 = 2.17
        # D = 1/(PI*0.5*0.1*2.17^2) = 1/(0.157*4.71) = 1.35
        {"NoH": 0.9, "ToH": 0.3, "BoH": 0.1, "alpha_x": 0.5, "alpha_y": 0.1, "expected": 1.35, "tolerance": 0.15},
        # Stretched along bitangent: symmetric case
        {"NoH": 0.9, "ToH": 0.1, "BoH": 0.3, "alpha_x": 0.1, "alpha_y": 0.5, "expected": 1.35, "tolerance": 0.15},
        # Additional test: very smooth surface at center
        {"NoH": 1.0, "ToH": 0.0, "BoH": 0.0, "alpha_x": 0.05, "alpha_y": 0.05, "expected": 127.32, "tolerance": 5.0},
    ],
    # rotate_tangent: rotation preserves orthogonality
    "rotate_tangent": [
        # No rotation
        {"angle": 0.0, "expected_factor": 1.0, "tolerance": 0.001},
        # 90 degree rotation (tangent becomes bitangent)
        {"angle": PI / 2, "expected_factor": 0.0, "tolerance": 0.001},
        # 180 degree rotation (tangent flips)
        {"angle": PI, "expected_factor": -1.0, "tolerance": 0.001},
        # 45 degree rotation
        {"angle": PI / 4, "expected_factor": 0.707, "tolerance": 0.01},
    ],
    # evaluate_aniso_brdf: full BRDF evaluation
    "evaluate_aniso_brdf": [
        # Isotropic reference (should match standard GGX)
        {
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "L": (0.0, 1.0, 0.0),
            "T": (1.0, 0.0, 0.0),
            "B": (0.0, 0.0, 1.0),
            "roughness": 0.5,
            "anisotropy": 0.0,
            "F0": (0.04, 0.04, 0.04),
            "expected_min": 0.0,
            "expected_max": 10.0,
        },
        # High anisotropy along tangent
        {
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "L": (0.0, 1.0, 0.0),
            "T": (1.0, 0.0, 0.0),
            "B": (0.0, 0.0, 1.0),
            "roughness": 0.5,
            "anisotropy": 0.8,
            "F0": (0.04, 0.04, 0.04),
            "expected_min": 0.0,
            "expected_max": 50.0,
        },
        # Rotated anisotropy direction
        {
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "L": (0.0, 1.0, 0.0),
            "T": (1.0, 0.0, 0.0),
            "B": (0.0, 0.0, 1.0),
            "roughness": 0.5,
            "anisotropy": 0.5,
            "direction": PI / 4,
            "F0": (0.04, 0.04, 0.04),
            "expected_min": 0.0,
            "expected_max": 20.0,
        },
        # Metal F0 with anisotropy
        {
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "L": (0.0, 1.0, 0.0),
            "T": (1.0, 0.0, 0.0),
            "B": (0.0, 0.0, 1.0),
            "roughness": 0.3,
            "anisotropy": 0.7,
            "F0": (1.0, 0.766, 0.336),
            "expected_min": 0.0,
            "expected_max": 200.0,
        },
    ],
}


# Edge cases for anisotropy
ANISOTROPY_EDGE_CASES = {
    "zero_anisotropy": {
        "description": "Zero anisotropy should match isotropic GGX",
        "params": {"roughness": 0.5, "anisotropy": 0.0},
    },
    "max_anisotropy": {
        "description": "Maximum anisotropy produces extreme stretch",
        "params": {"roughness": 0.5, "anisotropy": 1.0},
    },
    "smooth_with_anisotropy": {
        "description": "Very smooth surface with anisotropy",
        "params": {"roughness": 0.1, "anisotropy": 0.8},
    },
    "rough_with_anisotropy": {
        "description": "Very rough surface with anisotropy",
        "params": {"roughness": 1.0, "anisotropy": 0.8},
    },
    "direction_90_degrees": {
        "description": "Anisotropy rotated 90 degrees (along bitangent)",
        "params": {"roughness": 0.5, "anisotropy": 0.5, "direction": PI / 2},
    },
    "direction_45_degrees": {
        "description": "Anisotropy rotated 45 degrees (diagonal)",
        "params": {"roughness": 0.5, "anisotropy": 0.5, "direction": PI / 4},
    },
}


__all__ = [
    # WGSL source access
    "get_anisotropy_wgsl",
    # Parameters
    "AnisotropyParams",
    # Alpha computation
    "compute_aniso_alphas",
    # Tangent rotation
    "rotate_tangent",
    "rotate_bitangent",
    # Anisotropic NDF
    "d_ggx_anisotropic",
    # Anisotropic geometry
    "g1_ggx_anisotropic",
    "g_smith_ggx_anisotropic",
    # Fresnel
    "f_schlick",
    # Complete BRDF
    "evaluate_aniso_brdf",
    # Reference values
    "ANISOTROPY_REFERENCE_VALUES",
    "ANISOTROPY_EDGE_CASES",
    # Constants
    "PI",
    "INV_PI",
    "EPSILON",
]

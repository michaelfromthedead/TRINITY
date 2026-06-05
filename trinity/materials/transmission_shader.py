"""Transmission BRDF Functions for TRINITY Material System.

T-MAT-4.5: Transmission Implementation (Thin Glass, Liquids).

This module provides:
- Python reference implementations of transmission functions for testing
- Access to WGSL transmission source code
- Reference values for validation

The transmission model simulates:
- Thin glass (windows, bottles)
- Liquids (water, colored drinks)
- Thin-walled containers
- Ice and crystals

Transmission parameters:
- factor: Transmission amount (0=opaque, 1=fully transmissive)
- ior: Index of refraction (1.0=air, 1.5=glass, 1.33=water)
- roughness: Transmitted ray roughness for blur
- attenuation_color: Color absorbed over attenuation_distance
- attenuation_distance: Distance for full attenuation

Example::

    from trinity.materials.transmission_shader import (
        TransmissionParams,
        f_transmission,
        refract_direction,
        apply_beer_law,
        evaluate_transmission,
        get_transmission_wgsl,
    )

    # Get WGSL source for shader compilation
    wgsl_source = get_transmission_wgsl()

    # Reference calculation for testing
    fresnel = f_transmission(cos_theta=0.5, ior=1.5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional

# Mathematical constants
PI = 3.14159265359
EPSILON = 0.0001

# Default IOR values
AIR_IOR = 1.0
GLASS_IOR = 1.5
WATER_IOR = 1.33

# Type aliases
Vec3 = Tuple[float, float, float]
Vec4 = Tuple[float, float, float, float]


def get_transmission_wgsl() -> str:
    """Load the transmission WGSL source.

    Returns:
        WGSL source code containing all transmission functions.
    """
    wgsl_path = Path(__file__).parent / "wgsl" / "transmission.wgsl"
    return wgsl_path.read_text(encoding="utf-8")


# =============================================================================
# Transmission Parameters
# =============================================================================


@dataclass
class TransmissionParams:
    """Parameters controlling transmission behavior.

    Attributes:
        factor: Transmission amount (0=opaque, 1=fully transmissive).
        ior: Index of refraction (1.0=air, 1.5=glass, 1.33=water).
        roughness: Transmitted ray roughness for blur effect.
        attenuation_color: Color absorbed over attenuation_distance.
        attenuation_distance: Distance for full attenuation in world units.
    """

    factor: float = 1.0
    ior: float = 1.5
    roughness: float = 0.0
    attenuation_color: Vec3 = field(default_factory=lambda: (1.0, 1.0, 1.0))
    attenuation_distance: float = float("inf")

    def __post_init__(self) -> None:
        """Validate parameters are in valid range."""
        if not 0.0 <= self.factor <= 1.0:
            raise ValueError(f"factor must be in [0,1], got {self.factor}")
        if not 1.0 <= self.ior <= 3.0:
            raise ValueError(f"ior must be in [1,3], got {self.ior}")
        if not 0.0 <= self.roughness <= 1.0:
            raise ValueError(f"roughness must be in [0,1], got {self.roughness}")
        if self.attenuation_distance <= 0.0:
            raise ValueError(
                f"attenuation_distance must be positive, got {self.attenuation_distance}"
            )

    @classmethod
    def glass(cls) -> "TransmissionParams":
        """Create parameters for standard glass (IOR 1.5)."""
        return cls(factor=1.0, ior=1.5, roughness=0.0)

    @classmethod
    def water(cls) -> "TransmissionParams":
        """Create parameters for water (IOR 1.33)."""
        return cls(
            factor=1.0,
            ior=1.33,
            roughness=0.0,
            attenuation_color=(0.85, 0.95, 1.0),
            attenuation_distance=10.0,
        )

    @classmethod
    def colored_glass(
        cls, color: Vec3, absorption_distance: float
    ) -> "TransmissionParams":
        """Create parameters for colored glass.

        Args:
            color: Attenuation color (remaining after absorption_distance).
            absorption_distance: Distance for color attenuation.
        """
        return cls(
            factor=1.0,
            ior=1.5,
            roughness=0.0,
            attenuation_color=color,
            attenuation_distance=absorption_distance,
        )


# =============================================================================
# Fresnel Functions for Transmission
# =============================================================================


def f_transmission(cos_theta: float, ior: float) -> float:
    """Schlick Fresnel approximation for dielectric transmission.

    Computes reflectance at the interface; transmission = 1 - reflectance.

    Args:
        cos_theta: Cosine of incidence angle (dot of view and normal).
        ior: Index of refraction of transmissive medium.

    Returns:
        Fresnel reflectance (scalar).

    Example::

        >>> f_transmission(1.0, 1.5)  # Normal incidence, glass
        0.04
        >>> f_transmission(0.0, 1.5)  # Grazing angle
        1.0
    """
    # Compute F0 from IOR: ((n1 - n2) / (n1 + n2))^2
    f0 = pow((1.0 - ior) / (1.0 + ior), 2.0)
    Fc = pow(1.0 - cos_theta, 5.0)
    return f0 + (1.0 - f0) * Fc


def ior_to_f0(ior: float) -> float:
    """Compute F0 (reflectance at normal incidence) from IOR.

    Args:
        ior: Index of refraction.

    Returns:
        F0 value.

    Example::

        >>> ior_to_f0(1.5)  # Glass
        0.04
        >>> ior_to_f0(1.33)  # Water
        0.02...
    """
    return pow((1.0 - ior) / (1.0 + ior), 2.0)


# =============================================================================
# Refraction Functions (Snell's Law)
# =============================================================================


def refract_direction(
    incident: Vec3, normal: Vec3, eta: float
) -> Optional[Vec3]:
    """Compute refracted direction using Snell's law.

    Returns None if total internal reflection occurs.

    Snell's law: n1 * sin(theta_i) = n2 * sin(theta_t)
    Rewritten: eta = n1/n2, cos_t = sqrt(1 - eta^2 * (1 - cos_i^2))

    Args:
        incident: Incident direction (normalized, pointing away from surface).
        normal: Surface normal (normalized).
        eta: Ratio of IORs (IOR_from / IOR_to).

    Returns:
        Refracted direction (normalized) or None for TIR.

    Example::

        >>> # Light entering glass from air
        >>> incident = (0.0, 1.0, 0.0)  # Normal incidence
        >>> normal = (0.0, 1.0, 0.0)
        >>> eta = 1.0 / 1.5  # Air to glass
        >>> refract_direction(incident, normal, eta)
        (0.0, -1.0, 0.0)
    """
    cos_i = (
        normal[0] * incident[0]
        + normal[1] * incident[1]
        + normal[2] * incident[2]
    )
    sin2_i = 1.0 - cos_i * cos_i
    sin2_t = eta * eta * sin2_i

    # Total internal reflection check
    if sin2_t >= 1.0:
        return None

    cos_t = math.sqrt(1.0 - sin2_t)

    # Refracted direction: -eta * incident + (eta * cos_i - cos_t) * normal
    factor = eta * cos_i - cos_t
    refracted = (
        -eta * incident[0] + factor * normal[0],
        -eta * incident[1] + factor * normal[1],
        -eta * incident[2] + factor * normal[2],
    )

    # Normalize
    length = math.sqrt(
        refracted[0] ** 2 + refracted[1] ** 2 + refracted[2] ** 2
    )
    if length < EPSILON:
        return None

    return (
        refracted[0] / length,
        refracted[1] / length,
        refracted[2] / length,
    )


def is_total_internal_reflection(cos_i: float, eta: float) -> bool:
    """Check if total internal reflection occurs.

    Args:
        cos_i: Cosine of incidence angle.
        eta: Ratio of IORs (IOR_from / IOR_to).

    Returns:
        True if TIR occurs.

    Example::

        >>> # Glass to air at steep angle
        >>> eta = 1.5 / 1.0  # Glass to air
        >>> is_total_internal_reflection(0.5, eta)  # Steep angle
        True
        >>> is_total_internal_reflection(0.9, eta)  # Near normal
        False
    """
    sin2_i = 1.0 - cos_i * cos_i
    sin2_t = eta * eta * sin2_i
    return sin2_t >= 1.0


def get_critical_angle(ior: float) -> float:
    """Get the critical angle for total internal reflection (in radians).

    Args:
        ior: Index of refraction of denser medium.

    Returns:
        Critical angle in radians.

    Example::

        >>> import math
        >>> angle = get_critical_angle(1.5)  # Glass
        >>> math.degrees(angle)  # About 41.8 degrees
        41.8...
    """
    # sin(critical_angle) = n2/n1 = 1/ior (assuming air outside)
    return math.asin(1.0 / max(ior, 1.0001))


# =============================================================================
# Beer-Lambert Absorption
# =============================================================================


def apply_beer_law(
    transmitted_color: Vec3,
    distance: float,
    attenuation_color: Vec3,
    attenuation_distance: float,
) -> Vec3:
    """Apply Beer-Lambert absorption for colored transmission.

    Simulates light absorption as it travels through a medium.

    Formula: transmittance = exp(-absorption * distance / attenuation_distance)
    For each channel: transmittance_c = attenuation_color_c ^ (distance / attenuation_distance)

    Args:
        transmitted_color: Color of light entering the medium.
        distance: Distance traveled through the medium.
        attenuation_color: Color absorbed over attenuation_distance.
        attenuation_distance: Distance for full attenuation.

    Returns:
        Attenuated color after absorption.

    Example::

        >>> # White light through green glass
        >>> color = (1.0, 1.0, 1.0)
        >>> atten_color = (0.5, 1.0, 0.5)  # Absorbs red and blue
        >>> result = apply_beer_law(color, 1.0, atten_color, 1.0)
        >>> result  # More green than red/blue
        (0.5, 1.0, 0.5)
    """
    # Handle infinite attenuation distance (no absorption)
    if attenuation_distance <= 0.0 or attenuation_distance > 1e10:
        return transmitted_color

    # Handle zero or negative distance
    if distance <= 0.0:
        return transmitted_color

    # Beer-Lambert: transmittance = color ^ (distance / attenuation_distance)
    t = distance / attenuation_distance

    # Compute per-channel transmittance
    transmittance = (
        pow(attenuation_color[0], t),
        pow(attenuation_color[1], t),
        pow(attenuation_color[2], t),
    )

    return (
        transmitted_color[0] * transmittance[0],
        transmitted_color[1] * transmittance[1],
        transmitted_color[2] * transmittance[2],
    )


def compute_beer_transmittance(
    distance: float,
    attenuation_color: Vec3,
    attenuation_distance: float,
) -> Vec3:
    """Compute Beer-Lambert transmittance factor only.

    Args:
        distance: Distance traveled through medium.
        attenuation_color: Color absorbed over attenuation_distance.
        attenuation_distance: Distance for full attenuation.

    Returns:
        Transmittance factor per channel.

    Example::

        >>> compute_beer_transmittance(1.0, (0.5, 0.8, 1.0), 1.0)
        (0.5, 0.8, 1.0)
    """
    if attenuation_distance <= 0.0 or attenuation_distance > 1e10:
        return (1.0, 1.0, 1.0)

    if distance <= 0.0:
        return (1.0, 1.0, 1.0)

    t = distance / attenuation_distance
    return (
        pow(attenuation_color[0], t),
        pow(attenuation_color[1], t),
        pow(attenuation_color[2], t),
    )


# =============================================================================
# Transmission Evaluation
# =============================================================================


def evaluate_transmission(
    N: Vec3,
    V: Vec3,
    params: TransmissionParams,
    thickness: float,
    background_color: Vec3,
) -> Vec3:
    """Evaluate transmission contribution for a surface point.

    Combines refraction, Beer-Lambert absorption, and Fresnel weighting.

    Args:
        N: Surface normal (normalized).
        V: View direction (normalized, pointing toward camera).
        params: Transmission parameters.
        thickness: Material thickness for absorption calculation.
        background_color: Color of scene behind the surface.

    Returns:
        Final transmission contribution.

    Example::

        >>> N = (0.0, 1.0, 0.0)
        >>> V = (0.0, 1.0, 0.0)
        >>> params = TransmissionParams.glass()
        >>> bg = (1.0, 1.0, 1.0)
        >>> result = evaluate_transmission(N, V, params, 0.01, bg)
        >>> result[0] > 0.9  # Most light transmitted
        True
    """
    # Skip if transmission factor is zero
    if params.factor < EPSILON:
        return (0.0, 0.0, 0.0)

    # Compute view-normal angle
    NoV = max(N[0] * V[0] + N[1] * V[1] + N[2] * V[2], 0.0)

    # Compute eta (IOR ratio) - assuming air outside
    eta = AIR_IOR / params.ior

    # Check for total internal reflection
    if is_total_internal_reflection(NoV, eta):
        return (0.0, 0.0, 0.0)

    # Compute Fresnel (how much is reflected vs transmitted)
    F = f_transmission(NoV, params.ior)
    transmission_factor = (1.0 - F) * params.factor

    # Apply Beer-Lambert absorption based on thickness
    absorption_distance = thickness if thickness > 0.0 else 0.001
    attenuated_color = apply_beer_law(
        background_color,
        absorption_distance,
        params.attenuation_color,
        params.attenuation_distance,
    )

    # Apply transmission factor
    return (
        attenuated_color[0] * transmission_factor,
        attenuated_color[1] * transmission_factor,
        attenuated_color[2] * transmission_factor,
    )


def evaluate_transmission_with_fresnel(
    N: Vec3,
    V: Vec3,
    params: TransmissionParams,
    thickness: float,
    background_color: Vec3,
) -> Vec4:
    """Evaluate transmission and return both color and Fresnel factor.

    Use for proper blending with reflection.

    Args:
        N: Surface normal (normalized).
        V: View direction (normalized).
        params: Transmission parameters.
        thickness: Material thickness.
        background_color: Background color.

    Returns:
        Tuple (r, g, b, T) where rgb = transmission color, T = transmission factor.

    Example::

        >>> N = (0.0, 1.0, 0.0)
        >>> V = (0.0, 1.0, 0.0)
        >>> params = TransmissionParams.glass()
        >>> result = evaluate_transmission_with_fresnel(N, V, params, 0.01, (1.0, 1.0, 1.0))
        >>> result[3] > 0.9  # High transmission at normal incidence
        True
    """
    # Skip if transmission factor is zero
    if params.factor < EPSILON:
        return (0.0, 0.0, 0.0, 0.0)

    # Compute view-normal angle
    NoV = max(N[0] * V[0] + N[1] * V[1] + N[2] * V[2], 0.0)

    # Compute eta
    eta = AIR_IOR / params.ior

    # Check for TIR
    if is_total_internal_reflection(NoV, eta):
        return (0.0, 0.0, 0.0, 0.0)

    # Compute Fresnel
    F = f_transmission(NoV, params.ior)
    T = (1.0 - F) * params.factor

    # Apply absorption
    absorption_distance = thickness if thickness > 0.0 else 0.001
    attenuated_color = apply_beer_law(
        background_color,
        absorption_distance,
        params.attenuation_color,
        params.attenuation_distance,
    )

    return (
        attenuated_color[0] * T,
        attenuated_color[1] * T,
        attenuated_color[2] * T,
        T,
    )


# =============================================================================
# Layer Combination Functions
# =============================================================================


def combine_transmission_reflection(
    reflection: Vec3,
    transmission: Vec3,
    F: float,
    transmission_factor: float,
) -> Vec3:
    """Combine transmission with reflection/base BRDF.

    Uses Fresnel-weighted blending: final = F * reflection + (1 - F) * transmission

    Args:
        reflection: Reflected/specular component.
        transmission: Transmitted component.
        F: Fresnel reflectance.
        transmission_factor: Overall transmission factor.

    Returns:
        Combined color.

    Example::

        >>> refl = (0.1, 0.1, 0.1)
        >>> trans = (0.9, 0.9, 0.9)
        >>> combined = combine_transmission_reflection(refl, trans, 0.04, 1.0)
        >>> combined[0] > 0.8  # Mostly transmission
        True
    """
    T = (1.0 - F) * transmission_factor

    return (
        reflection[0] * F + transmission[0] * T,
        reflection[1] * F + transmission[1] * T,
        reflection[2] * F + transmission[2] * T,
    )


def combine_transmission_simple(
    base_brdf: Vec3,
    transmission_result: Vec4,
) -> Vec3:
    """Combine transmission result with base BRDF.

    Simplified version using pre-computed transmission.

    Args:
        base_brdf: Base material BRDF (diffuse + specular).
        transmission_result: Result from evaluate_transmission_with_fresnel.

    Returns:
        Combined color.

    Example::

        >>> base = (0.3, 0.3, 0.3)
        >>> trans = (0.6, 0.6, 0.6, 0.9)  # (r, g, b, T)
        >>> combined = combine_transmission_simple(base, trans)
    """
    # transmission_result[3] contains T = (1-F) * factor
    base_factor = 1.0 - transmission_result[3]

    return (
        base_brdf[0] * base_factor + transmission_result[0],
        base_brdf[1] * base_factor + transmission_result[1],
        base_brdf[2] * base_factor + transmission_result[2],
    )


# =============================================================================
# Reference Values for Testing
# =============================================================================

# Reference values computed with known-good implementations
TRANSMISSION_REFERENCE_VALUES = {
    # F_Transmission reference values: (cos_theta, ior) -> expected
    "F_Transmission": [
        {"cos_theta": 1.0, "ior": 1.5, "expected": 0.04, "tolerance": 0.001},
        {"cos_theta": 0.0, "ior": 1.5, "expected": 1.0, "tolerance": 0.001},
        {"cos_theta": 1.0, "ior": 1.33, "expected": 0.02, "tolerance": 0.005},
        {"cos_theta": 0.5, "ior": 1.5, "expected": 0.0696, "tolerance": 0.01},
        {"cos_theta": 0.707, "ior": 1.5, "expected": 0.0508, "tolerance": 0.01},
    ],
    # ior_to_f0 reference values: ior -> expected
    "ior_to_f0": [
        {"ior": 1.0, "expected": 0.0, "tolerance": 0.001},
        {"ior": 1.5, "expected": 0.04, "tolerance": 0.001},
        {"ior": 1.33, "expected": 0.02, "tolerance": 0.005},
        {"ior": 2.0, "expected": 0.111, "tolerance": 0.01},
        {"ior": 2.5, "expected": 0.184, "tolerance": 0.01},
    ],
    # Beer-Lambert reference values
    "apply_beer_law": [
        # No absorption (white attenuation)
        {
            "transmitted_color": (1.0, 1.0, 1.0),
            "distance": 1.0,
            "attenuation_color": (1.0, 1.0, 1.0),
            "attenuation_distance": 1.0,
            "expected": (1.0, 1.0, 1.0),
            "tolerance": 0.001,
        },
        # Full absorption at attenuation distance
        {
            "transmitted_color": (1.0, 1.0, 1.0),
            "distance": 1.0,
            "attenuation_color": (0.5, 0.5, 0.5),
            "attenuation_distance": 1.0,
            "expected": (0.5, 0.5, 0.5),
            "tolerance": 0.001,
        },
        # Half distance = sqrt of attenuation
        {
            "transmitted_color": (1.0, 1.0, 1.0),
            "distance": 0.5,
            "attenuation_color": (0.25, 0.25, 0.25),
            "attenuation_distance": 1.0,
            "expected": (0.5, 0.5, 0.5),
            "tolerance": 0.01,
        },
        # Colored absorption
        {
            "transmitted_color": (1.0, 1.0, 1.0),
            "distance": 1.0,
            "attenuation_color": (0.8, 0.5, 0.3),
            "attenuation_distance": 1.0,
            "expected": (0.8, 0.5, 0.3),
            "tolerance": 0.001,
        },
    ],
    # evaluate_transmission reference values
    "evaluate_transmission": [
        # Normal incidence, glass, no absorption
        {
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "factor": 1.0,
            "ior": 1.5,
            "thickness": 0.01,
            "background_color": (1.0, 1.0, 1.0),
            "expected_r": 0.96,  # (1 - 0.04) * 1.0
            "tolerance": 0.01,
        },
        # Half transmission factor
        {
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "factor": 0.5,
            "ior": 1.5,
            "thickness": 0.01,
            "background_color": (1.0, 1.0, 1.0),
            "expected_r": 0.48,  # (1 - 0.04) * 0.5
            "tolerance": 0.01,
        },
        # Zero transmission
        {
            "N": (0.0, 1.0, 0.0),
            "V": (0.0, 1.0, 0.0),
            "factor": 0.0,
            "ior": 1.5,
            "thickness": 0.01,
            "background_color": (1.0, 1.0, 1.0),
            "expected_r": 0.0,
            "tolerance": 0.001,
        },
    ],
    # TIR test cases
    "total_internal_reflection": [
        # Air to glass - no TIR
        {"cos_i": 0.1, "eta": 1.0 / 1.5, "expected_tir": False},
        {"cos_i": 0.5, "eta": 1.0 / 1.5, "expected_tir": False},
        # Glass to air - TIR at steep angles
        {"cos_i": 0.1, "eta": 1.5 / 1.0, "expected_tir": True},
        {"cos_i": 0.5, "eta": 1.5 / 1.0, "expected_tir": True},
        {"cos_i": 0.9, "eta": 1.5 / 1.0, "expected_tir": False},
    ],
}

# Edge cases for transmission
TRANSMISSION_EDGE_CASES = {
    "zero_factor": {
        "description": "Transmission disabled (factor=0)",
        "params": {"factor": 0.0, "ior": 1.5},
        "expected": "zero contribution",
    },
    "normal_incidence": {
        "description": "Normal incidence - maximum transmission",
        "cos_theta": 1.0,
        "expected_F": 0.04,
        "expected_T": 0.96,
    },
    "grazing_angle": {
        "description": "Grazing angle - maximum reflection",
        "cos_theta": 0.0,
        "expected_F": 1.0,
        "expected_T": 0.0,
    },
    "total_internal_reflection": {
        "description": "TIR when going from denser to less dense medium",
        "eta": 1.5,  # Glass to air
        "critical_angle_deg": 41.8,
    },
    "no_absorption": {
        "description": "White attenuation color = no absorption",
        "attenuation_color": (1.0, 1.0, 1.0),
        "expected": "input color unchanged",
    },
}


__all__ = [
    # WGSL source access
    "get_transmission_wgsl",
    # Parameters
    "TransmissionParams",
    # Fresnel functions
    "f_transmission",
    "ior_to_f0",
    # Refraction functions
    "refract_direction",
    "is_total_internal_reflection",
    "get_critical_angle",
    # Beer-Lambert absorption
    "apply_beer_law",
    "compute_beer_transmittance",
    # Evaluation functions
    "evaluate_transmission",
    "evaluate_transmission_with_fresnel",
    # Layer combination
    "combine_transmission_reflection",
    "combine_transmission_simple",
    # Reference values
    "TRANSMISSION_REFERENCE_VALUES",
    "TRANSMISSION_EDGE_CASES",
    # Constants
    "AIR_IOR",
    "GLASS_IOR",
    "WATER_IOR",
    "PI",
    "EPSILON",
]

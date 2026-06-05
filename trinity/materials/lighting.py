"""Lighting Functions for TRINITY Material System.

T-MAT-3.3: Light Loop and Shading.

This module provides:
- Python reference implementations of lighting functions for testing
- Access to WGSL lighting source code
- Light structure definitions matching WGSL
- Reference values for validation

The lighting system supports:
- Directional lights (sun, moon)
- Point lights with inverse-square attenuation
- Spot lights with cone angle falloff
- Light accumulation loop for up to 8 lights
- Ambient occlusion and emissive terms

Example::

    from trinity.materials.lighting import (
        Light,
        LightType,
        create_directional_light,
        create_point_light,
        evaluate_point_light,
        accumulate_lighting,
        get_lighting_wgsl,
    )

    # Create a point light
    light = create_point_light(
        position=(0.0, 5.0, 0.0),
        color=(1.0, 1.0, 1.0),
        intensity=100.0,
        range=10.0,
    )

    # Get WGSL source for shader compilation
    wgsl_source = get_lighting_wgsl()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import List, Tuple

# Mathematical constants (matching WGSL)
PI = 3.14159265359
EPSILON = 0.0001
MAX_LIGHTS = 8

# Type alias for RGB color and 3D vector
Vec3 = Tuple[float, float, float]


class LightType(IntEnum):
    """Light type enumeration matching WGSL constants."""

    DIRECTIONAL = 0
    POINT = 1
    SPOT = 2


@dataclass
class Light:
    """Unified light structure matching WGSL Light struct.

    The interpretation of fields depends on the light type:
    - Directional: Only direction, color, intensity are used
    - Point: position, color, intensity, range are used
    - Spot: All fields are used including cone angles

    Attributes:
        light_type: Type of light (Directional, Point, Spot)
        intensity: Light intensity multiplier
        range: Maximum range for Point/Spot lights
        position: World space position (Point/Spot only)
        direction: Light direction (normalized)
        color: Light color in linear RGB
        cos_inner_angle: Cosine of inner cone angle (Spot only)
        cos_outer_angle: Cosine of outer cone angle (Spot only)
    """

    light_type: LightType = LightType.DIRECTIONAL
    intensity: float = 1.0
    range: float = 10.0
    position: Vec3 = (0.0, 0.0, 0.0)
    direction: Vec3 = (0.0, -1.0, 0.0)
    color: Vec3 = (1.0, 1.0, 1.0)
    cos_inner_angle: float = 1.0
    cos_outer_angle: float = 0.0


@dataclass
class LightSample:
    """Result of light evaluation.

    Attributes:
        direction: Direction from surface to light (normalized)
        radiance: Incoming radiance (color * intensity * attenuation)
        distance: Distance to light source
        shadow: Shadow factor (1.0 = lit, 0.0 = shadowed)
    """

    direction: Vec3 = (0.0, 1.0, 0.0)
    radiance: Vec3 = (0.0, 0.0, 0.0)
    distance: float = 0.0
    shadow: float = 1.0


@dataclass
class LightingResult:
    """Accumulated lighting result.

    Attributes:
        diffuse: Accumulated diffuse lighting contribution
        specular: Accumulated specular lighting contribution
    """

    diffuse: Vec3 = (0.0, 0.0, 0.0)
    specular: Vec3 = (0.0, 0.0, 0.0)


@dataclass
class PBRParamsLighting:
    """PBR parameters needed for lighting evaluation."""

    base_color: Vec3 = (1.0, 1.0, 1.0)
    roughness: float = 0.5
    metallic: float = 0.0
    occlusion: float = 1.0
    emissive: Vec3 = (0.0, 0.0, 0.0)


def get_lighting_wgsl() -> str:
    """Load the lighting WGSL source.

    Returns:
        WGSL source code containing all lighting functions.
    """
    wgsl_path = Path(__file__).parent / "wgsl" / "lighting.wgsl"
    return wgsl_path.read_text(encoding="utf-8")


# =============================================================================
# Vector Math Helpers
# =============================================================================


def _normalize(v: Vec3) -> Vec3:
    """Normalize a 3D vector."""
    length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if length < EPSILON:
        return (0.0, 0.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _dot(a: Vec3, b: Vec3) -> float:
    """Dot product of two 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(v: Vec3) -> float:
    """Length of a 3D vector."""
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _sub(a: Vec3, b: Vec3) -> Vec3:
    """Subtract two 3D vectors."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    """Add two 3D vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(v: Vec3, s: float) -> Vec3:
    """Scale a 3D vector."""
    return (v[0] * s, v[1] * s, v[2] * s)


def _mul(a: Vec3, b: Vec3) -> Vec3:
    """Component-wise multiply two 3D vectors."""
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def _negate(v: Vec3) -> Vec3:
    """Negate a 3D vector."""
    return (-v[0], -v[1], -v[2])


def _saturate(x: float) -> float:
    """Clamp value to [0, 1]."""
    return max(0.0, min(1.0, x))


# =============================================================================
# Attenuation Functions
# =============================================================================


def attenuation_point(distance: float, range_val: float) -> float:
    """Inverse-square attenuation with smooth falloff.

    Uses the UE4/Frostbite windowed falloff function.

    Args:
        distance: Distance from light to surface point.
        range_val: Maximum range of the light.

    Returns:
        Attenuation factor in [0, 1].

    Example::

        >>> attenuation_point(0.0, 10.0)  # At light position
        10000.0  # Very high (clamped in practice)
        >>> attenuation_point(5.0, 10.0)  # Half range
        0.0272...
        >>> attenuation_point(10.0, 10.0)  # At range boundary
        0.0
    """
    d = max(distance, EPSILON)
    r = max(range_val, EPSILON)

    distance_ratio = d / r
    distance_ratio_sq = distance_ratio * distance_ratio
    distance_ratio_4 = distance_ratio_sq * distance_ratio_sq

    falloff = max(1.0 - distance_ratio_4, 0.0)
    falloff_sq = falloff * falloff

    attenuation = falloff_sq / (d * d + EPSILON)

    return attenuation


def attenuation_spot_angle(cos_angle: float, cos_inner: float, cos_outer: float) -> float:
    """Angular attenuation for spotlight cone.

    Smooth falloff between inner and outer cone angles.

    Args:
        cos_angle: Cosine of angle between light direction and surface direction.
        cos_inner: Cosine of inner cone angle (where falloff begins).
        cos_outer: Cosine of outer cone angle (edge of light cone).

    Returns:
        Angular attenuation factor in [0, 1].

    Example::

        >>> attenuation_spot_angle(1.0, 0.9, 0.7)  # Center of cone
        1.0
        >>> attenuation_spot_angle(0.8, 0.9, 0.7)  # Between inner and outer
        0.25
        >>> attenuation_spot_angle(0.6, 0.9, 0.7)  # Outside cone
        0.0
    """
    inner = max(cos_inner, cos_outer + EPSILON)
    t = _saturate((cos_angle - cos_outer) / (inner - cos_outer + EPSILON))
    return t * t


# =============================================================================
# Light Evaluation Functions
# =============================================================================


def evaluate_directional_light(light: Light, world_position: Vec3) -> LightSample:
    """Evaluate a directional light.

    Directional lights have no attenuation and represent infinitely distant sources.

    Args:
        light: The directional light to evaluate.
        world_position: World space position of the surface point.

    Returns:
        LightSample with direction and radiance.
    """
    direction = _negate(_normalize(light.direction))
    radiance = _scale(light.color, light.intensity)

    return LightSample(
        direction=direction,
        radiance=radiance,
        distance=1e10,
        shadow=1.0,
    )


def evaluate_point_light(light: Light, world_position: Vec3) -> LightSample:
    """Evaluate a point light.

    Point lights emit uniformly in all directions with distance attenuation.

    Args:
        light: The point light to evaluate.
        world_position: World space position of the surface point.

    Returns:
        LightSample with direction and radiance.
    """
    to_light = _sub(light.position, world_position)
    distance = _length(to_light)

    direction = _scale(to_light, 1.0 / max(distance, EPSILON))
    atten = attenuation_point(distance, light.range)
    radiance = _scale(light.color, light.intensity * atten)

    return LightSample(
        direction=direction,
        radiance=radiance,
        distance=distance,
        shadow=1.0,
    )


def evaluate_spot_light(light: Light, world_position: Vec3) -> LightSample:
    """Evaluate a spot light.

    Spot lights emit in a cone with both distance and angular attenuation.

    Args:
        light: The spot light to evaluate.
        world_position: World space position of the surface point.

    Returns:
        LightSample with direction and radiance.
    """
    to_light = _sub(light.position, world_position)
    distance = _length(to_light)

    direction = _scale(to_light, 1.0 / max(distance, EPSILON))

    dist_atten = attenuation_point(distance, light.range)

    light_dir_normalized = _normalize(light.direction)
    cos_angle = _dot(_negate(direction), light_dir_normalized)
    angle_atten = attenuation_spot_angle(cos_angle, light.cos_inner_angle, light.cos_outer_angle)

    attenuation = dist_atten * angle_atten
    radiance = _scale(light.color, light.intensity * attenuation)

    return LightSample(
        direction=direction,
        radiance=radiance,
        distance=distance,
        shadow=1.0,
    )


def evaluate_light(light: Light, world_position: Vec3) -> LightSample:
    """Evaluate a light of any type.

    Dispatches to the appropriate evaluation function based on light type.

    Args:
        light: The light to evaluate.
        world_position: World space position of the surface point.

    Returns:
        LightSample with direction and radiance.
    """
    if light.light_type == LightType.DIRECTIONAL:
        return evaluate_directional_light(light, world_position)
    elif light.light_type == LightType.POINT:
        return evaluate_point_light(light, world_position)
    elif light.light_type == LightType.SPOT:
        return evaluate_spot_light(light, world_position)
    else:
        return LightSample(
            direction=(0.0, 1.0, 0.0),
            radiance=(0.0, 0.0, 0.0),
            distance=0.0,
            shadow=0.0,
        )


# =============================================================================
# Shadow Sampling (Placeholder)
# =============================================================================


def sample_shadow(light_index: int, world_position: Vec3, light_sample: LightSample) -> float:
    """Sample shadow for a light at the given world position.

    This is a placeholder implementation that always returns fully lit.

    Args:
        light_index: Index of the light in the light buffer.
        world_position: World space position of the surface point.
        light_sample: The evaluated light sample.

    Returns:
        Shadow factor (1.0 = fully lit, 0.0 = fully shadowed).
    """
    return 1.0


# =============================================================================
# BRDF Helper Functions (duplicated from brdf.py to avoid circular imports)
# =============================================================================


def _d_ggx(NoH: float, roughness: float) -> float:
    """GGX Normal Distribution Function."""
    a = roughness * roughness
    a2 = a * a
    NoH2 = NoH * NoH
    denom = NoH2 * (a2 - 1.0) + 1.0
    return a2 / (PI * denom * denom + EPSILON)


def _g_smith_ggx(NoV: float, NoL: float, roughness: float) -> float:
    """Smith-GGX Geometry Function."""
    a = roughness * roughness
    a2 = a * a
    GGXV = NoL * math.sqrt(NoV * NoV * (1.0 - a2) + a2)
    GGXL = NoV * math.sqrt(NoL * NoL * (1.0 - a2) + a2)
    return 0.5 / (GGXV + GGXL + EPSILON)


def _f_schlick(VoH: float, F0: Vec3) -> Vec3:
    """Schlick Fresnel approximation."""
    Fc = pow(1.0 - VoH, 5.0)
    return (
        F0[0] + (1.0 - F0[0]) * Fc,
        F0[1] + (1.0 - F0[1]) * Fc,
        F0[2] + (1.0 - F0[2]) * Fc,
    )


def _compute_f0(base_color: Vec3, metallic: float) -> Vec3:
    """Compute F0 from material properties."""
    dielectric_f0 = 0.04
    return (
        dielectric_f0 * (1.0 - metallic) + base_color[0] * metallic,
        dielectric_f0 * (1.0 - metallic) + base_color[1] * metallic,
        dielectric_f0 * (1.0 - metallic) + base_color[2] * metallic,
    )


# =============================================================================
# Lighting Accumulation
# =============================================================================


def accumulate_lighting(
    params: PBRParamsLighting,
    N: Vec3,
    V: Vec3,
    world_position: Vec3,
    lights: List[Light],
) -> LightingResult:
    """Accumulate lighting from all active lights.

    Evaluates BRDF for each light and sums contributions.

    Args:
        params: PBR material parameters.
        N: Surface normal (world space, normalized).
        V: View direction (world space, normalized, pointing toward camera).
        world_position: World space position of the surface point.
        lights: List of lights to evaluate (max 8).

    Returns:
        Accumulated diffuse and specular lighting.
    """
    result = LightingResult()
    diffuse = [0.0, 0.0, 0.0]
    specular = [0.0, 0.0, 0.0]

    F0 = _compute_f0(params.base_color, params.metallic)
    count = min(len(lights), MAX_LIGHTS)

    for i in range(count):
        light = lights[i]
        light_sample = evaluate_light(light, world_position)
        light_sample.shadow = sample_shadow(i, world_position, light_sample)

        L = light_sample.direction
        NoL = max(_dot(N, L), 0.0)

        if NoL <= 0.0:
            continue

        H = _normalize(_add(V, L))
        NoV = max(_dot(N, V), 0.0)
        NoH = max(_dot(N, H), 0.0)
        VoH = max(_dot(V, H), 0.0)

        if NoV <= 0.0:
            continue

        D = _d_ggx(NoH, params.roughness)
        G = _g_smith_ggx(NoV, NoL, params.roughness)
        F = _f_schlick(VoH, F0)

        specular_brdf = (D * G * F[0], D * G * F[1], D * G * F[2])

        kD = (
            (1.0 - F[0]) * (1.0 - params.metallic),
            (1.0 - F[1]) * (1.0 - params.metallic),
            (1.0 - F[2]) * (1.0 - params.metallic),
        )
        diffuse_brdf = (
            kD[0] * params.base_color[0] / PI,
            kD[1] * params.base_color[1] / PI,
            kD[2] * params.base_color[2] / PI,
        )

        radiance_factor = light_sample.shadow * NoL

        for c in range(3):
            rad = light_sample.radiance[c] * radiance_factor
            diffuse[c] += diffuse_brdf[c] * rad
            specular[c] += specular_brdf[c] * rad

    result.diffuse = (diffuse[0], diffuse[1], diffuse[2])
    result.specular = (specular[0], specular[1], specular[2])
    return result


# =============================================================================
# Final Shading Composition
# =============================================================================


def compose_final_shading(
    lighting: LightingResult,
    params: PBRParamsLighting,
    ambient: Vec3,
) -> Vec3:
    """Compose final shading with ambient occlusion and emissive.

    Args:
        lighting: Accumulated lighting result from accumulate_lighting.
        params: PBR material parameters (for occlusion and emissive).
        ambient: Ambient/indirect lighting contribution.

    Returns:
        Final composed color in linear RGB.
    """
    ambient_occluded = _scale(ambient, params.occlusion)
    direct = _add(lighting.diffuse, lighting.specular)
    lit = _add(direct, ambient_occluded)
    final_color = _add(lit, params.emissive)
    return final_color


def evaluate_all_lighting(
    params: PBRParamsLighting,
    N: Vec3,
    V: Vec3,
    world_position: Vec3,
    lights: List[Light],
    ambient: Vec3,
) -> Vec3:
    """Complete lighting evaluation for a surface point.

    Args:
        params: PBR material parameters.
        N: Surface normal (world space, normalized).
        V: View direction (world space, normalized, pointing toward camera).
        world_position: World space position of the surface point.
        lights: List of lights to evaluate.
        ambient: Ambient/indirect lighting contribution.

    Returns:
        Final lit color in linear RGB.
    """
    lighting = accumulate_lighting(params, N, V, world_position, lights)
    return compose_final_shading(lighting, params, ambient)


# =============================================================================
# Light Creation Helpers
# =============================================================================


def create_directional_light(
    direction: Vec3,
    color: Vec3 = (1.0, 1.0, 1.0),
    intensity: float = 1.0,
) -> Light:
    """Create a directional light.

    Args:
        direction: Direction the light is traveling (will be normalized).
        color: Light color in linear RGB.
        intensity: Light intensity multiplier.

    Returns:
        Configured directional light.

    Example::

        >>> sun = create_directional_light(
        ...     direction=(0.5, -1.0, 0.5),
        ...     color=(1.0, 0.95, 0.9),
        ...     intensity=3.0,
        ... )
    """
    return Light(
        light_type=LightType.DIRECTIONAL,
        intensity=intensity,
        range=0.0,
        position=(0.0, 0.0, 0.0),
        direction=_normalize(direction),
        color=color,
        cos_inner_angle=1.0,
        cos_outer_angle=1.0,
    )


def create_point_light(
    position: Vec3,
    color: Vec3 = (1.0, 1.0, 1.0),
    intensity: float = 1.0,
    range_val: float = 10.0,
) -> Light:
    """Create a point light.

    Args:
        position: World space position of the light.
        color: Light color in linear RGB.
        intensity: Light intensity multiplier.
        range_val: Maximum range of the light.

    Returns:
        Configured point light.

    Example::

        >>> bulb = create_point_light(
        ...     position=(0.0, 3.0, 0.0),
        ...     color=(1.0, 0.9, 0.7),
        ...     intensity=100.0,
        ...     range_val=15.0,
        ... )
    """
    return Light(
        light_type=LightType.POINT,
        intensity=intensity,
        range=range_val,
        position=position,
        direction=(0.0, -1.0, 0.0),
        color=color,
        cos_inner_angle=1.0,
        cos_outer_angle=-1.0,
    )


def create_spot_light(
    position: Vec3,
    direction: Vec3,
    color: Vec3 = (1.0, 1.0, 1.0),
    intensity: float = 1.0,
    range_val: float = 10.0,
    inner_angle: float = 0.4,
    outer_angle: float = 0.6,
) -> Light:
    """Create a spot light.

    Args:
        position: World space position of the light.
        direction: Direction the spotlight is pointing (will be normalized).
        color: Light color in linear RGB.
        intensity: Light intensity multiplier.
        range_val: Maximum range of the light.
        inner_angle: Inner cone angle in radians (no falloff inside).
        outer_angle: Outer cone angle in radians (edge of light cone).

    Returns:
        Configured spot light.

    Example::

        >>> flashlight = create_spot_light(
        ...     position=(0.0, 2.0, 0.0),
        ...     direction=(0.0, -1.0, 0.0),
        ...     color=(1.0, 1.0, 0.9),
        ...     intensity=200.0,
        ...     range_val=20.0,
        ...     inner_angle=0.2,
        ...     outer_angle=0.4,
        ... )
    """
    return Light(
        light_type=LightType.SPOT,
        intensity=intensity,
        range=range_val,
        position=position,
        direction=_normalize(direction),
        color=color,
        cos_inner_angle=math.cos(inner_angle),
        cos_outer_angle=math.cos(outer_angle),
    )


# =============================================================================
# Reference Values for Testing
# =============================================================================

# Reference values computed with known-good implementations
LIGHTING_REFERENCE_VALUES = {
    # Attenuation reference values
    # Formula: falloff_sq / (d^2 + EPSILON) where falloff = max(1 - (d/r)^4, 0)
    "attenuation_point": [
        {"distance": 1.0, "range": 10.0, "expected": 0.9997, "tolerance": 0.01},
        {"distance": 5.0, "range": 10.0, "expected": 0.0272, "tolerance": 0.01},
        {"distance": 9.0, "range": 10.0, "expected": 0.0005, "tolerance": 0.001},
        {"distance": 10.0, "range": 10.0, "expected": 0.0, "tolerance": 0.001},
        {"distance": 0.5, "range": 5.0, "expected": 3.998, "tolerance": 0.05},
    ],
    # Spotlight angular attenuation
    "attenuation_spot_angle": [
        {"cos_angle": 1.0, "cos_inner": 0.9, "cos_outer": 0.7, "expected": 1.0, "tolerance": 0.01},
        {"cos_angle": 0.9, "cos_inner": 0.9, "cos_outer": 0.7, "expected": 1.0, "tolerance": 0.01},
        {"cos_angle": 0.8, "cos_inner": 0.9, "cos_outer": 0.7, "expected": 0.25, "tolerance": 0.01},
        {"cos_angle": 0.7, "cos_inner": 0.9, "cos_outer": 0.7, "expected": 0.0, "tolerance": 0.01},
        {"cos_angle": 0.5, "cos_inner": 0.9, "cos_outer": 0.7, "expected": 0.0, "tolerance": 0.01},
    ],
    # Directional light evaluation
    "directional_light": [
        {
            "direction": (0.0, -1.0, 0.0),
            "color": (1.0, 1.0, 1.0),
            "intensity": 1.0,
            "position": (0.0, 0.0, 0.0),
            "expected_direction": (0.0, 1.0, 0.0),
            "expected_radiance": (1.0, 1.0, 1.0),
            "tolerance": 0.01,
        },
        {
            "direction": (0.707, -0.707, 0.0),
            "color": (1.0, 0.5, 0.0),
            "intensity": 2.0,
            "position": (5.0, 5.0, 5.0),
            "expected_direction": (-0.707, 0.707, 0.0),
            "expected_radiance": (2.0, 1.0, 0.0),
            "tolerance": 0.01,
        },
    ],
    # Point light evaluation
    "point_light": [
        {
            "light_position": (0.0, 5.0, 0.0),
            "surface_position": (0.0, 0.0, 0.0),
            "color": (1.0, 1.0, 1.0),
            "intensity": 100.0,
            "range": 10.0,
            "expected_direction": (0.0, 1.0, 0.0),
            "expected_radiance_r": 2.72,
            "tolerance": 0.1,
        },
    ],
    # Light accumulation (1-8 lights)
    "accumulate_lighting": [
        # 1 light
        {
            "num_lights": 1,
            "description": "Single directional light, normal incidence",
            "expected_diffuse_r": 0.30,
            "tolerance": 0.1,
        },
        # 4 lights
        {
            "num_lights": 4,
            "description": "Four directional lights",
            "expected_diffuse_r": 1.20,
            "tolerance": 0.2,
        },
        # 8 lights (maximum)
        {
            "num_lights": 8,
            "description": "Eight directional lights (max)",
            "expected_diffuse_r": 2.40,
            "tolerance": 0.3,
        },
    ],
}

# Edge case configurations
LIGHTING_EDGE_CASES = {
    "zero_intensity": {
        "description": "Light with zero intensity",
        "light": create_point_light((0.0, 5.0, 0.0), intensity=0.0),
        "expect_zero_radiance": True,
    },
    "zero_range": {
        "description": "Point light with zero range",
        "light": create_point_light((0.0, 5.0, 0.0), range_val=0.0),
        "expect_zero_radiance": True,
    },
    "surface_at_light": {
        "description": "Surface at light position",
        "light": create_point_light((0.0, 0.0, 0.0)),
        "surface_position": (0.0, 0.0, 0.0),
        "expect_high_radiance": True,
    },
    "beyond_range": {
        "description": "Surface beyond light range",
        "light": create_point_light((0.0, 0.0, 0.0), range_val=5.0),
        "surface_position": (10.0, 0.0, 0.0),
        "expect_zero_radiance": True,
    },
    "spotlight_outside_cone": {
        "description": "Surface outside spotlight cone",
        "light": create_spot_light(
            (0.0, 5.0, 0.0),
            (0.0, -1.0, 0.0),
            inner_angle=0.1,
            outer_angle=0.2,
        ),
        "surface_position": (10.0, 0.0, 0.0),
        "expect_low_radiance": True,
    },
}


__all__ = [
    # WGSL source access
    "get_lighting_wgsl",
    # Types
    "LightType",
    "Light",
    "LightSample",
    "LightingResult",
    "PBRParamsLighting",
    # Attenuation functions
    "attenuation_point",
    "attenuation_spot_angle",
    # Light evaluation
    "evaluate_directional_light",
    "evaluate_point_light",
    "evaluate_spot_light",
    "evaluate_light",
    # Shadow sampling
    "sample_shadow",
    # Lighting accumulation
    "accumulate_lighting",
    "compose_final_shading",
    "evaluate_all_lighting",
    # Light creation
    "create_directional_light",
    "create_point_light",
    "create_spot_light",
    # Reference values
    "LIGHTING_REFERENCE_VALUES",
    "LIGHTING_EDGE_CASES",
    # Constants
    "PI",
    "EPSILON",
    "MAX_LIGHTS",
]

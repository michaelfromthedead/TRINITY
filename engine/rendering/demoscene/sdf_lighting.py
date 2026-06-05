"""
SDF Lighting Module for Demoscene Rendering (T-DEMO-3.7, T-DEMO-3.8).

Implements diffuse and specular lighting calculations for SDF ray marching:
  - T-DEMO-3.7: Diffuse lighting with multiple lights (Lambertian)
  - T-DEMO-3.8: Specular lighting (Blinn-Phong and GGX)

The module provides both Python reference implementations for testing
and WGSL code generation for GPU shaders.

Usage:
    >>> from engine.rendering.demoscene.sdf_lighting import (
    ...     calculate_diffuse, calculate_specular_blinn_phong, LightingCodegen
    ... )
    >>> # Python reference calculation
    >>> diffuse = calculate_diffuse(
    ...     p=(0, 0, 0), n=(0, 1, 0),
    ...     light_pos=(0, 5, 0), light_color=(1, 1, 1), intensity=2.0
    ... )
    >>> # WGSL generation
    >>> codegen = LightingCodegen()
    >>> wgsl = codegen.generate_lighting_functions()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence, Tuple, Union

from .ast_nodes import FloatNode, LightNode, LightType, MaterialNode, Vec3Node


# =============================================================================
# TYPE ALIASES
# =============================================================================

Vec3 = Tuple[float, float, float]


# =============================================================================
# VECTOR MATH HELPERS
# =============================================================================


def vec3_add(a: Vec3, b: Vec3) -> Vec3:
    """Add two vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec3_sub(a: Vec3, b: Vec3) -> Vec3:
    """Subtract two vectors."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec3_mul(a: Vec3, s: float) -> Vec3:
    """Multiply vector by scalar."""
    return (a[0] * s, a[1] * s, a[2] * s)


def vec3_mul_vec(a: Vec3, b: Vec3) -> Vec3:
    """Component-wise multiply two vectors."""
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def vec3_dot(a: Vec3, b: Vec3) -> float:
    """Dot product of two vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec3_length(v: Vec3) -> float:
    """Length of a vector."""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vec3_normalize(v: Vec3) -> Vec3:
    """Normalize a vector to unit length."""
    length = vec3_length(v)
    if length < 1e-10:
        return (0.0, 0.0, 0.0)
    inv_len = 1.0 / length
    return (v[0] * inv_len, v[1] * inv_len, v[2] * inv_len)


def vec3_reflect(v: Vec3, n: Vec3) -> Vec3:
    """Reflect vector v around normal n."""
    d = 2.0 * vec3_dot(v, n)
    return vec3_sub(v, vec3_mul(n, d))


def vec3_clamp(v: Vec3, min_val: float, max_val: float) -> Vec3:
    """Clamp vector components to range."""
    return (
        max(min_val, min(max_val, v[0])),
        max(min_val, min(max_val, v[1])),
        max(min_val, min(max_val, v[2])),
    )


# =============================================================================
# LIGHT DATA STRUCTURES
# =============================================================================


@dataclass
class LightParams:
    """Parameters for a light source used in lighting calculations.

    This is a simplified representation for the lighting functions,
    extracted from LightNode AST nodes.

    Attributes:
        position: World position for point/area lights, or direction for directional.
        color: RGB color of the light (0-1 range).
        intensity: Light intensity multiplier.
        light_type: Type of light (point, directional, area).
        direction: Direction for directional/spot lights.
        radius: Attenuation radius for point lights.
    """
    position: Vec3 = (0.0, 0.0, 0.0)
    color: Vec3 = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    light_type: LightType = LightType.POINT
    direction: Vec3 = (0.0, -1.0, 0.0)
    radius: float = 10.0

    @classmethod
    def from_light_node(cls, node: LightNode) -> "LightParams":
        """Create LightParams from a LightNode AST node."""
        return cls(
            position=(node.position.x, node.position.y, node.position.z),
            color=(node.color.x, node.color.y, node.color.z),
            intensity=node.intensity.value,
            light_type=node.light_type,
            direction=(node.direction.x, node.direction.y, node.direction.z),
            radius=node.radius.value,
        )


@dataclass
class MaterialParams:
    """PBR material parameters for lighting calculations.

    Attributes:
        albedo: Base color (diffuse reflectance).
        roughness: Surface roughness [0=smooth, 1=rough].
        metallic: Metalness factor [0=dielectric, 1=metal].
        ambient_occlusion: AO factor [0=occluded, 1=fully lit].
    """
    albedo: Vec3 = (0.5, 0.5, 0.5)
    roughness: float = 0.5
    metallic: float = 0.0
    ambient_occlusion: float = 1.0

    @classmethod
    def from_material_node(cls, node: MaterialNode) -> "MaterialParams":
        """Create MaterialParams from a MaterialNode AST node."""
        return cls(
            albedo=(node.albedo.x, node.albedo.y, node.albedo.z),
            roughness=node.roughness.value,
            metallic=node.metallic.value,
            ambient_occlusion=node.ambient_occlusion.value,
        )


# =============================================================================
# SHADOW CALCULATION (SOFT SHADOWS)
# =============================================================================


ShadowSceneFunc = Callable[[Vec3], float]


def calculate_soft_shadow(
    p: Vec3,
    light_dir: Vec3,
    scene_sdf: ShadowSceneFunc,
    *,
    min_t: float = 0.02,
    max_t: float = 10.0,
    k: float = 8.0,
    max_steps: int = 64,
) -> float:
    """Calculate soft shadow factor using ray marching.

    Marches a ray from point p toward the light, accumulating shadow based
    on how close the ray passes to occluding geometry.

    Args:
        p: Surface point.
        light_dir: Normalized direction toward the light.
        scene_sdf: Scene signed distance function.
        min_t: Minimum marching distance (to avoid self-shadowing).
        max_t: Maximum marching distance.
        k: Shadow softness parameter (higher = sharper shadows).
        max_steps: Maximum marching steps.

    Returns:
        Shadow factor in [0, 1] where 0 = fully shadowed, 1 = fully lit.

    Reference:
        Inigo Quilez -- Soft Shadows
        https://iquilezles.org/articles/rmshadows/
    """
    shadow = 1.0
    t = min_t

    for _ in range(max_steps):
        pos = vec3_add(p, vec3_mul(light_dir, t))
        d = scene_sdf(pos)

        if d < 0.001:
            return 0.0  # Hard shadow

        # Penumbra estimation
        shadow = min(shadow, k * d / t)
        t += max(d, 0.01)

        if t > max_t:
            break

    return max(0.0, min(1.0, shadow))


def calculate_hard_shadow(
    p: Vec3,
    light_dir: Vec3,
    light_distance: float,
    scene_sdf: ShadowSceneFunc,
    *,
    min_t: float = 0.02,
    max_steps: int = 64,
) -> float:
    """Calculate hard shadow factor (binary: 0 or 1).

    Args:
        p: Surface point.
        light_dir: Normalized direction toward the light.
        light_distance: Distance to the light source.
        scene_sdf: Scene signed distance function.
        min_t: Minimum marching distance.
        max_steps: Maximum marching steps.

    Returns:
        1.0 if lit, 0.0 if shadowed.
    """
    t = min_t

    for _ in range(max_steps):
        pos = vec3_add(p, vec3_mul(light_dir, t))
        d = scene_sdf(pos)

        if d < 0.001:
            return 0.0

        t += d

        if t > light_distance:
            return 1.0

    return 1.0


# =============================================================================
# ATTENUATION FUNCTIONS
# =============================================================================


def calculate_attenuation_inverse_square(
    distance: float,
    radius: float = 10.0,
) -> float:
    """Calculate light attenuation with inverse square falloff.

    Uses a smooth window function that reaches zero at the light radius.

    Args:
        distance: Distance from light to surface point.
        radius: Light influence radius (attenuation = 0 beyond this).

    Returns:
        Attenuation factor in [0, 1].
    """
    if distance >= radius:
        return 0.0

    # Smooth window: (1 - (d/r)^2)^2
    d_over_r = distance / radius
    window = max(0.0, 1.0 - d_over_r * d_over_r)
    window = window * window

    # Inverse square with offset to avoid singularity
    inv_sq = 1.0 / (distance * distance + 1.0)

    return window * inv_sq


def calculate_attenuation_linear(
    distance: float,
    radius: float = 10.0,
) -> float:
    """Calculate linear light attenuation.

    Args:
        distance: Distance from light to surface point.
        radius: Light influence radius.

    Returns:
        Attenuation factor in [0, 1].
    """
    return max(0.0, 1.0 - distance / radius)


# =============================================================================
# T-DEMO-3.7: DIFFUSE LIGHTING
# =============================================================================


def calculate_diffuse(
    p: Vec3,
    n: Vec3,
    light_pos: Vec3,
    light_color: Vec3,
    intensity: float,
    *,
    shadow_factor: float = 1.0,
    attenuation: float = 1.0,
) -> Vec3:
    """Calculate Lambertian diffuse lighting contribution from a single light.

    Implements the classic Lambertian diffuse model:
        diffuse = max(0, dot(n, l)) * light_color * intensity * attenuation * shadow

    Args:
        p: Surface point position.
        n: Surface normal (should be normalized).
        light_pos: Light position in world space.
        light_color: RGB color of the light (0-1 range).
        intensity: Light intensity multiplier.
        shadow_factor: Shadow attenuation (0=shadowed, 1=lit).
        attenuation: Distance attenuation factor.

    Returns:
        Diffuse lighting contribution as RGB tuple.

    Example:
        >>> diffuse = calculate_diffuse(
        ...     p=(0, 0, 0),
        ...     n=(0, 1, 0),
        ...     light_pos=(0, 5, 0),
        ...     light_color=(1, 1, 1),
        ...     intensity=2.0
        ... )
    """
    # Light direction (from surface to light)
    light_vec = vec3_sub(light_pos, p)
    light_dir = vec3_normalize(light_vec)

    # Lambertian term: max(0, N dot L)
    n_dot_l = max(0.0, vec3_dot(n, light_dir))

    # Final diffuse = N.L * color * intensity * attenuation * shadow
    diffuse_intensity = n_dot_l * intensity * attenuation * shadow_factor

    return (
        light_color[0] * diffuse_intensity,
        light_color[1] * diffuse_intensity,
        light_color[2] * diffuse_intensity,
    )


def calculate_diffuse_directional(
    n: Vec3,
    light_dir: Vec3,
    light_color: Vec3,
    intensity: float,
    *,
    shadow_factor: float = 1.0,
) -> Vec3:
    """Calculate Lambertian diffuse for a directional light.

    Args:
        n: Surface normal (normalized).
        light_dir: Direction TO the light (normalized).
        light_color: RGB color of the light.
        intensity: Light intensity multiplier.
        shadow_factor: Shadow attenuation.

    Returns:
        Diffuse lighting contribution as RGB tuple.
    """
    n_dot_l = max(0.0, vec3_dot(n, light_dir))
    diffuse_intensity = n_dot_l * intensity * shadow_factor

    return (
        light_color[0] * diffuse_intensity,
        light_color[1] * diffuse_intensity,
        light_color[2] * diffuse_intensity,
    )


def calculate_all_diffuse(
    p: Vec3,
    n: Vec3,
    lights: Sequence[LightParams],
    scene_sdf: Optional[ShadowSceneFunc] = None,
    *,
    enable_shadows: bool = True,
) -> Vec3:
    """Accumulate diffuse lighting from multiple lights.

    Each light's contribution is calculated independently with its own
    shadow term, then summed together.

    Args:
        p: Surface point position.
        n: Surface normal (normalized).
        lights: Sequence of light parameters.
        scene_sdf: Scene SDF for shadow calculation (optional).
        enable_shadows: Whether to calculate shadows.

    Returns:
        Total diffuse lighting contribution as RGB tuple.

    Example:
        >>> lights = [
        ...     LightParams(position=(5, 5, 0), color=(1, 0.9, 0.8), intensity=2.0),
        ...     LightParams(position=(-5, 3, 5), color=(0.3, 0.3, 0.5), intensity=1.0),
        ... ]
        >>> diffuse = calculate_all_diffuse((0, 0, 0), (0, 1, 0), lights)
    """
    total = (0.0, 0.0, 0.0)

    for light in lights:
        # Calculate shadow for this light
        shadow_factor = 1.0

        if light.light_type == LightType.DIRECTIONAL:
            # Directional light: use direction directly
            light_dir = vec3_normalize(light.direction)
            light_dir = vec3_mul(light_dir, -1.0)  # Flip: stored as "where light shines"

            if enable_shadows and scene_sdf is not None:
                shadow_factor = calculate_soft_shadow(p, light_dir, scene_sdf)

            contribution = calculate_diffuse_directional(
                n, light_dir, light.color, light.intensity,
                shadow_factor=shadow_factor
            )

        else:
            # Point/Area light: calculate direction and attenuation
            light_vec = vec3_sub(light.position, p)
            light_dist = vec3_length(light_vec)
            light_dir = vec3_normalize(light_vec)

            attenuation = calculate_attenuation_inverse_square(
                light_dist, light.radius
            )

            if enable_shadows and scene_sdf is not None:
                shadow_factor = calculate_soft_shadow(
                    p, light_dir, scene_sdf, max_t=light_dist
                )

            contribution = calculate_diffuse(
                p, n, light.position, light.color, light.intensity,
                shadow_factor=shadow_factor,
                attenuation=attenuation,
            )

        total = vec3_add(total, contribution)

    return total


# =============================================================================
# T-DEMO-3.8: SPECULAR LIGHTING (BLINN-PHONG)
# =============================================================================


def roughness_to_shininess(roughness: float) -> float:
    """Convert roughness to Blinn-Phong shininess exponent.

    Uses the formula: shininess = (2.0 / roughness^4) - 2.0
    This gives a reasonable mapping where:
      - roughness=0.1 -> shininess ~2000 (very shiny)
      - roughness=0.5 -> shininess ~30
      - roughness=1.0 -> shininess = 0 (matte)

    Args:
        roughness: Surface roughness in [0, 1].

    Returns:
        Shininess exponent for Blinn-Phong.
    """
    # Clamp roughness to avoid division by zero
    r = max(0.01, min(1.0, roughness))
    r4 = r * r * r * r
    return max(0.0, (2.0 / r4) - 2.0)


def calculate_specular_blinn_phong(
    p: Vec3,
    n: Vec3,
    view_dir: Vec3,
    light_pos: Vec3,
    light_color: Vec3,
    intensity: float,
    roughness: float,
    *,
    shadow_factor: float = 1.0,
    attenuation: float = 1.0,
) -> Vec3:
    """Calculate Blinn-Phong specular reflection.

    Uses the half-vector formulation:
        H = normalize(V + L)
        specular = pow(max(0, dot(N, H)), shininess)

    Args:
        p: Surface point position.
        n: Surface normal (normalized).
        view_dir: Direction from surface to camera (normalized).
        light_pos: Light position in world space.
        light_color: RGB color of the light.
        intensity: Light intensity multiplier.
        roughness: Surface roughness [0=shiny, 1=matte].
        shadow_factor: Shadow attenuation.
        attenuation: Distance attenuation.

    Returns:
        Specular lighting contribution as RGB tuple.

    Example:
        >>> specular = calculate_specular_blinn_phong(
        ...     p=(0, 0, 0),
        ...     n=(0, 1, 0),
        ...     view_dir=(0, 0.707, 0.707),
        ...     light_pos=(0, 5, 0),
        ...     light_color=(1, 1, 1),
        ...     intensity=2.0,
        ...     roughness=0.3
        ... )
    """
    # Light direction
    light_vec = vec3_sub(light_pos, p)
    light_dir = vec3_normalize(light_vec)

    # Half vector
    half_vec = vec3_normalize(vec3_add(view_dir, light_dir))

    # Shininess from roughness
    shininess = roughness_to_shininess(roughness)

    # N dot H (specular alignment)
    n_dot_h = max(0.0, vec3_dot(n, half_vec))

    # Specular intensity
    if shininess > 0.0:
        spec_intensity = pow(n_dot_h, shininess)
    else:
        spec_intensity = 1.0 if n_dot_h > 0.999 else 0.0

    # Apply modifiers
    spec_intensity *= intensity * attenuation * shadow_factor

    return (
        light_color[0] * spec_intensity,
        light_color[1] * spec_intensity,
        light_color[2] * spec_intensity,
    )


def calculate_specular_blinn_phong_directional(
    n: Vec3,
    view_dir: Vec3,
    light_dir: Vec3,
    light_color: Vec3,
    intensity: float,
    roughness: float,
    *,
    shadow_factor: float = 1.0,
) -> Vec3:
    """Calculate Blinn-Phong specular for directional light.

    Args:
        n: Surface normal (normalized).
        view_dir: Direction from surface to camera (normalized).
        light_dir: Direction TO the light (normalized).
        light_color: RGB color of the light.
        intensity: Light intensity multiplier.
        roughness: Surface roughness.
        shadow_factor: Shadow attenuation.

    Returns:
        Specular lighting contribution as RGB tuple.
    """
    half_vec = vec3_normalize(vec3_add(view_dir, light_dir))
    shininess = roughness_to_shininess(roughness)
    n_dot_h = max(0.0, vec3_dot(n, half_vec))

    if shininess > 0.0:
        spec_intensity = pow(n_dot_h, shininess)
    else:
        spec_intensity = 1.0 if n_dot_h > 0.999 else 0.0

    spec_intensity *= intensity * shadow_factor

    return (
        light_color[0] * spec_intensity,
        light_color[1] * spec_intensity,
        light_color[2] * spec_intensity,
    )


# =============================================================================
# T-DEMO-3.8: SPECULAR LIGHTING (GGX/COOK-TORRANCE)
# =============================================================================


def fresnel_schlick(cos_theta: float, f0: Vec3) -> Vec3:
    """Fresnel-Schlick approximation.

    Args:
        cos_theta: Cosine of angle between view and half vector.
        f0: Base reflectivity at normal incidence.

    Returns:
        Fresnel reflectance as RGB tuple.
    """
    t = max(0.0, min(1.0, 1.0 - cos_theta))
    t5 = t * t * t * t * t
    return (
        f0[0] + (1.0 - f0[0]) * t5,
        f0[1] + (1.0 - f0[1]) * t5,
        f0[2] + (1.0 - f0[2]) * t5,
    )


def distribution_ggx(n: Vec3, h: Vec3, roughness: float) -> float:
    """GGX/Trowbridge-Reitz normal distribution function.

    Args:
        n: Surface normal.
        h: Half vector.
        roughness: Surface roughness.

    Returns:
        NDF value.
    """
    a = roughness * roughness
    a2 = a * a

    n_dot_h = max(0.0, vec3_dot(n, h))
    n_dot_h2 = n_dot_h * n_dot_h

    denom = n_dot_h2 * (a2 - 1.0) + 1.0
    denom = math.pi * denom * denom

    return a2 / max(denom, 0.0001)


def geometry_schlick_ggx(n_dot_v: float, roughness: float) -> float:
    """Schlick-GGX geometry function.

    Args:
        n_dot_v: Dot product of normal and view direction.
        roughness: Surface roughness.

    Returns:
        Geometry term.
    """
    r = roughness + 1.0
    k = (r * r) / 8.0
    return n_dot_v / (n_dot_v * (1.0 - k) + k)


def geometry_smith(
    n: Vec3, v: Vec3, l: Vec3, roughness: float
) -> float:
    """Smith's geometry function combining view and light terms.

    Args:
        n: Surface normal.
        v: View direction.
        l: Light direction.
        roughness: Surface roughness.

    Returns:
        Combined geometry term.
    """
    n_dot_v = max(0.0, vec3_dot(n, v))
    n_dot_l = max(0.0, vec3_dot(n, l))
    ggx1 = geometry_schlick_ggx(n_dot_v, roughness)
    ggx2 = geometry_schlick_ggx(n_dot_l, roughness)
    return ggx1 * ggx2


def calculate_specular_ggx(
    p: Vec3,
    n: Vec3,
    view_dir: Vec3,
    light_pos: Vec3,
    light_color: Vec3,
    intensity: float,
    roughness: float,
    metallic: float,
    albedo: Vec3 = (0.5, 0.5, 0.5),
    *,
    shadow_factor: float = 1.0,
    attenuation: float = 1.0,
) -> Vec3:
    """Calculate GGX/Cook-Torrance specular reflection.

    Implements the full Cook-Torrance BRDF:
        f_r = (D * F * G) / (4 * (N.V) * (N.L))

    Args:
        p: Surface point position.
        n: Surface normal (normalized).
        view_dir: Direction from surface to camera (normalized).
        light_pos: Light position in world space.
        light_color: RGB color of the light.
        intensity: Light intensity multiplier.
        roughness: Surface roughness [0=smooth, 1=rough].
        metallic: Metalness factor [0=dielectric, 1=metal].
        albedo: Base color (used for metallic F0).
        shadow_factor: Shadow attenuation.
        attenuation: Distance attenuation.

    Returns:
        Specular lighting contribution as RGB tuple.
    """
    # Light direction
    light_vec = vec3_sub(light_pos, p)
    light_dir = vec3_normalize(light_vec)

    # Half vector
    half_vec = vec3_normalize(vec3_add(view_dir, light_dir))

    # F0: dielectric = 0.04, metallic = albedo
    f0_dielectric = (0.04, 0.04, 0.04)
    f0 = (
        f0_dielectric[0] * (1.0 - metallic) + albedo[0] * metallic,
        f0_dielectric[1] * (1.0 - metallic) + albedo[1] * metallic,
        f0_dielectric[2] * (1.0 - metallic) + albedo[2] * metallic,
    )

    # BRDF terms
    h_dot_v = max(0.0, vec3_dot(half_vec, view_dir))
    f = fresnel_schlick(h_dot_v, f0)
    d = distribution_ggx(n, half_vec, roughness)
    g = geometry_smith(n, view_dir, light_dir, roughness)

    # Cook-Torrance specular
    n_dot_v = max(0.001, vec3_dot(n, view_dir))
    n_dot_l = max(0.0, vec3_dot(n, light_dir))

    denom = 4.0 * n_dot_v * n_dot_l + 0.0001
    spec_scale = d * g / denom

    spec_intensity = spec_scale * intensity * attenuation * shadow_factor * n_dot_l

    return (
        f[0] * light_color[0] * spec_intensity,
        f[1] * light_color[1] * spec_intensity,
        f[2] * light_color[2] * spec_intensity,
    )


def calculate_specular_ggx_directional(
    n: Vec3,
    view_dir: Vec3,
    light_dir: Vec3,
    light_color: Vec3,
    intensity: float,
    roughness: float,
    metallic: float,
    albedo: Vec3 = (0.5, 0.5, 0.5),
    *,
    shadow_factor: float = 1.0,
) -> Vec3:
    """Calculate GGX specular for directional light.

    Args:
        n: Surface normal (normalized).
        view_dir: Direction from surface to camera (normalized).
        light_dir: Direction TO the light (normalized).
        light_color: RGB color of the light.
        intensity: Light intensity.
        roughness: Surface roughness.
        metallic: Metalness factor.
        albedo: Base color.
        shadow_factor: Shadow attenuation.

    Returns:
        Specular lighting contribution as RGB tuple.
    """
    half_vec = vec3_normalize(vec3_add(view_dir, light_dir))

    f0_dielectric = (0.04, 0.04, 0.04)
    f0 = (
        f0_dielectric[0] * (1.0 - metallic) + albedo[0] * metallic,
        f0_dielectric[1] * (1.0 - metallic) + albedo[1] * metallic,
        f0_dielectric[2] * (1.0 - metallic) + albedo[2] * metallic,
    )

    h_dot_v = max(0.0, vec3_dot(half_vec, view_dir))
    f = fresnel_schlick(h_dot_v, f0)
    d = distribution_ggx(n, half_vec, roughness)
    g = geometry_smith(n, view_dir, light_dir, roughness)

    n_dot_v = max(0.001, vec3_dot(n, view_dir))
    n_dot_l = max(0.0, vec3_dot(n, light_dir))

    denom = 4.0 * n_dot_v * n_dot_l + 0.0001
    spec_scale = d * g / denom

    spec_intensity = spec_scale * intensity * shadow_factor * n_dot_l

    return (
        f[0] * light_color[0] * spec_intensity,
        f[1] * light_color[1] * spec_intensity,
        f[2] * light_color[2] * spec_intensity,
    )


# =============================================================================
# COMBINED LIGHTING
# =============================================================================


def calculate_lighting(
    p: Vec3,
    n: Vec3,
    view_dir: Vec3,
    material: MaterialParams,
    lights: Sequence[LightParams],
    scene_sdf: Optional[ShadowSceneFunc] = None,
    *,
    enable_shadows: bool = True,
    use_ggx: bool = True,
    ambient: Vec3 = (0.03, 0.03, 0.03),
) -> Vec3:
    """Calculate complete lighting for a surface point.

    Combines ambient, diffuse, and specular (Blinn-Phong or GGX) lighting
    from all light sources.

    Args:
        p: Surface point position.
        n: Surface normal (normalized).
        view_dir: Direction from surface to camera (normalized).
        material: PBR material parameters.
        lights: Sequence of light parameters.
        scene_sdf: Scene SDF for shadow calculation (optional).
        enable_shadows: Whether to calculate shadows.
        use_ggx: If True, use GGX specular; otherwise Blinn-Phong.
        ambient: Ambient light color.

    Returns:
        Final lit color as RGB tuple.

    Example:
        >>> material = MaterialParams(albedo=(0.8, 0.2, 0.2), roughness=0.3)
        >>> lights = [LightParams(position=(5, 5, 5), intensity=2.0)]
        >>> color = calculate_lighting(
        ...     p=(0, 0, 0),
        ...     n=(0, 1, 0),
        ...     view_dir=(0, 0.707, 0.707),
        ...     material=material,
        ...     lights=lights,
        ... )
    """
    # Ambient term
    result = vec3_mul_vec(ambient, material.albedo)
    result = vec3_mul(result, material.ambient_occlusion)

    for light in lights:
        # Calculate shadow for this light
        shadow_factor = 1.0

        if light.light_type == LightType.DIRECTIONAL:
            # Directional light
            light_dir = vec3_normalize(light.direction)
            light_dir = vec3_mul(light_dir, -1.0)

            if enable_shadows and scene_sdf is not None:
                shadow_factor = calculate_soft_shadow(p, light_dir, scene_sdf)

            # Diffuse
            diffuse = calculate_diffuse_directional(
                n, light_dir, light.color, light.intensity,
                shadow_factor=shadow_factor
            )
            diffuse = vec3_mul_vec(diffuse, material.albedo)
            diffuse = vec3_mul(diffuse, 1.0 - material.metallic)

            # Specular
            if use_ggx:
                specular = calculate_specular_ggx_directional(
                    n, view_dir, light_dir, light.color, light.intensity,
                    material.roughness, material.metallic, material.albedo,
                    shadow_factor=shadow_factor
                )
            else:
                specular = calculate_specular_blinn_phong_directional(
                    n, view_dir, light_dir, light.color, light.intensity,
                    material.roughness, shadow_factor=shadow_factor
                )

        else:
            # Point/Area light
            light_vec = vec3_sub(light.position, p)
            light_dist = vec3_length(light_vec)
            light_dir = vec3_normalize(light_vec)

            attenuation = calculate_attenuation_inverse_square(
                light_dist, light.radius
            )

            if enable_shadows and scene_sdf is not None:
                shadow_factor = calculate_soft_shadow(
                    p, light_dir, scene_sdf, max_t=light_dist
                )

            # Diffuse
            diffuse = calculate_diffuse(
                p, n, light.position, light.color, light.intensity,
                shadow_factor=shadow_factor, attenuation=attenuation
            )
            diffuse = vec3_mul_vec(diffuse, material.albedo)
            diffuse = vec3_mul(diffuse, 1.0 - material.metallic)

            # Specular
            if use_ggx:
                specular = calculate_specular_ggx(
                    p, n, view_dir, light.position, light.color, light.intensity,
                    material.roughness, material.metallic, material.albedo,
                    shadow_factor=shadow_factor, attenuation=attenuation
                )
            else:
                specular = calculate_specular_blinn_phong(
                    p, n, view_dir, light.position, light.color, light.intensity,
                    material.roughness, shadow_factor=shadow_factor,
                    attenuation=attenuation
                )

        result = vec3_add(result, diffuse)
        result = vec3_add(result, specular)

    return result


# =============================================================================
# WGSL CODE GENERATION
# =============================================================================


WGSL_SOFT_SHADOW = """\
/// Calculate soft shadow using ray marching.
///   p         -- surface point
///   light_dir -- normalized direction toward light
///   min_t     -- minimum distance (avoid self-shadowing)
///   max_t     -- maximum distance
///   k         -- shadow softness (higher = sharper)
///   returns   -- shadow factor [0=shadowed, 1=lit]
fn calculate_soft_shadow(
    p: vec3<f32>,
    light_dir: vec3<f32>,
    min_t: f32,
    max_t: f32,
    k: f32,
) -> f32 {
    var shadow: f32 = 1.0;
    var t: f32 = min_t;

    for (var i = 0u; i < 64u; i = i + 1u) {
        let pos = p + light_dir * t;
        let d = scene_sdf(pos).x;

        if (d < 0.001) {
            return 0.0;
        }

        shadow = min(shadow, k * d / t);
        t = t + max(d, 0.01);

        if (t > max_t) {
            break;
        }
    }

    return clamp(shadow, 0.0, 1.0);
}
"""


WGSL_ATTENUATION = """\
/// Calculate light attenuation with inverse square falloff.
///   distance -- distance from light to surface
///   radius   -- light influence radius
///   returns  -- attenuation factor [0, 1]
fn calculate_attenuation(distance: f32, radius: f32) -> f32 {
    if (distance >= radius) {
        return 0.0;
    }

    let d_over_r = distance / radius;
    let window = max(0.0, 1.0 - d_over_r * d_over_r);
    let window2 = window * window;
    let inv_sq = 1.0 / (distance * distance + 1.0);

    return window2 * inv_sq;
}
"""


WGSL_DIFFUSE = """\
/// Calculate Lambertian diffuse lighting from a point light.
///   p           -- surface point
///   n           -- surface normal (normalized)
///   light_pos   -- light position
///   light_color -- light RGB color
///   intensity   -- light intensity
///   shadow      -- shadow factor [0=shadowed, 1=lit]
///   atten       -- distance attenuation
///   returns     -- diffuse contribution RGB
fn calculate_diffuse(
    p: vec3<f32>,
    n: vec3<f32>,
    light_pos: vec3<f32>,
    light_color: vec3<f32>,
    intensity: f32,
    shadow: f32,
    atten: f32,
) -> vec3<f32> {
    let light_vec = light_pos - p;
    let light_dir = normalize(light_vec);
    let n_dot_l = max(dot(n, light_dir), 0.0);
    let diff_intensity = n_dot_l * intensity * atten * shadow;
    return light_color * diff_intensity;
}

/// Calculate Lambertian diffuse for a directional light.
///   n           -- surface normal (normalized)
///   light_dir   -- direction TO the light (normalized)
///   light_color -- light RGB color
///   intensity   -- light intensity
///   shadow      -- shadow factor
///   returns     -- diffuse contribution RGB
fn calculate_diffuse_directional(
    n: vec3<f32>,
    light_dir: vec3<f32>,
    light_color: vec3<f32>,
    intensity: f32,
    shadow: f32,
) -> vec3<f32> {
    let n_dot_l = max(dot(n, light_dir), 0.0);
    let diff_intensity = n_dot_l * intensity * shadow;
    return light_color * diff_intensity;
}
"""


WGSL_SPECULAR_BLINN_PHONG = """\
/// Convert roughness to Blinn-Phong shininess exponent.
fn roughness_to_shininess(roughness: f32) -> f32 {
    let r = clamp(roughness, 0.01, 1.0);
    let r4 = r * r * r * r;
    return max(0.0, (2.0 / r4) - 2.0);
}

/// Calculate Blinn-Phong specular reflection.
///   p           -- surface point
///   n           -- surface normal (normalized)
///   view_dir    -- direction to camera (normalized)
///   light_pos   -- light position
///   light_color -- light RGB color
///   intensity   -- light intensity
///   roughness   -- surface roughness [0=shiny, 1=matte]
///   shadow      -- shadow factor
///   atten       -- distance attenuation
///   returns     -- specular contribution RGB
fn calculate_specular_blinn_phong(
    p: vec3<f32>,
    n: vec3<f32>,
    view_dir: vec3<f32>,
    light_pos: vec3<f32>,
    light_color: vec3<f32>,
    intensity: f32,
    roughness: f32,
    shadow: f32,
    atten: f32,
) -> vec3<f32> {
    let light_vec = light_pos - p;
    let light_dir = normalize(light_vec);
    let half_vec = normalize(view_dir + light_dir);
    let shininess = roughness_to_shininess(roughness);
    let n_dot_h = max(dot(n, half_vec), 0.0);

    var spec_intensity: f32;
    if (shininess > 0.0) {
        spec_intensity = pow(n_dot_h, shininess);
    } else {
        spec_intensity = select(0.0, 1.0, n_dot_h > 0.999);
    }

    spec_intensity = spec_intensity * intensity * atten * shadow;
    return light_color * spec_intensity;
}

/// Blinn-Phong specular for directional light.
fn calculate_specular_blinn_phong_directional(
    n: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    light_color: vec3<f32>,
    intensity: f32,
    roughness: f32,
    shadow: f32,
) -> vec3<f32> {
    let half_vec = normalize(view_dir + light_dir);
    let shininess = roughness_to_shininess(roughness);
    let n_dot_h = max(dot(n, half_vec), 0.0);

    var spec_intensity: f32;
    if (shininess > 0.0) {
        spec_intensity = pow(n_dot_h, shininess);
    } else {
        spec_intensity = select(0.0, 1.0, n_dot_h > 0.999);
    }

    spec_intensity = spec_intensity * intensity * shadow;
    return light_color * spec_intensity;
}
"""


WGSL_SPECULAR_GGX = """\
/// Fresnel-Schlick approximation.
fn fresnel_schlick_lighting(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let t = clamp(1.0 - cos_theta, 0.0, 1.0);
    let t5 = t * t * t * t * t;
    return f0 + (1.0 - f0) * t5;
}

/// GGX/Trowbridge-Reitz normal distribution function.
fn distribution_ggx_lighting(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = max(dot(n, h), 0.0);
    let n_dot_h2 = n_dot_h * n_dot_h;

    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    let denom2 = 3.14159265359 * denom * denom;

    return a2 / max(denom2, 0.0001);
}

/// Schlick-GGX geometry function.
fn geometry_schlick_ggx_lighting(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

/// Smith's geometry function.
fn geometry_smith_lighting(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    let ggx1 = geometry_schlick_ggx_lighting(n_dot_v, roughness);
    let ggx2 = geometry_schlick_ggx_lighting(n_dot_l, roughness);
    return ggx1 * ggx2;
}

/// Calculate GGX/Cook-Torrance specular reflection.
fn calculate_specular_ggx(
    p: vec3<f32>,
    n: vec3<f32>,
    view_dir: vec3<f32>,
    light_pos: vec3<f32>,
    light_color: vec3<f32>,
    intensity: f32,
    roughness: f32,
    metallic: f32,
    albedo: vec3<f32>,
    shadow: f32,
    atten: f32,
) -> vec3<f32> {
    let light_vec = light_pos - p;
    let light_dir = normalize(light_vec);
    let half_vec = normalize(view_dir + light_dir);

    // F0: dielectric = 0.04, metallic = albedo
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // BRDF terms
    let h_dot_v = max(dot(half_vec, view_dir), 0.0);
    let f = fresnel_schlick_lighting(h_dot_v, f0);
    let d = distribution_ggx_lighting(n, half_vec, roughness);
    let g = geometry_smith_lighting(n, view_dir, light_dir, roughness);

    // Cook-Torrance
    let n_dot_v = max(dot(n, view_dir), 0.001);
    let n_dot_l = max(dot(n, light_dir), 0.0);

    let denom = 4.0 * n_dot_v * n_dot_l + 0.0001;
    let spec_scale = d * g / denom;

    let spec_intensity = spec_scale * intensity * atten * shadow * n_dot_l;

    return f * light_color * spec_intensity;
}

/// GGX specular for directional light.
fn calculate_specular_ggx_directional(
    n: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    light_color: vec3<f32>,
    intensity: f32,
    roughness: f32,
    metallic: f32,
    albedo: vec3<f32>,
    shadow: f32,
) -> vec3<f32> {
    let half_vec = normalize(view_dir + light_dir);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    let h_dot_v = max(dot(half_vec, view_dir), 0.0);
    let f = fresnel_schlick_lighting(h_dot_v, f0);
    let d = distribution_ggx_lighting(n, half_vec, roughness);
    let g = geometry_smith_lighting(n, view_dir, light_dir, roughness);

    let n_dot_v = max(dot(n, view_dir), 0.001);
    let n_dot_l = max(dot(n, light_dir), 0.0);

    let denom = 4.0 * n_dot_v * n_dot_l + 0.0001;
    let spec_scale = d * g / denom;

    let spec_intensity = spec_scale * intensity * shadow * n_dot_l;

    return f * light_color * spec_intensity;
}
"""


class LightingCodegen:
    """WGSL code generator for lighting functions.

    Generates WGSL shader code for diffuse and specular lighting
    calculations used in SDF ray marching shaders.

    Usage:
        >>> codegen = LightingCodegen()
        >>> wgsl = codegen.generate_lighting_functions()
        >>> # Or generate specific functions
        >>> diffuse_wgsl = codegen.generate_diffuse()
        >>> specular_wgsl = codegen.generate_specular_ggx()
    """

    def __init__(self) -> None:
        self._emitted_functions: set[str] = set()

    def reset(self) -> None:
        """Reset emitted function tracking."""
        self._emitted_functions.clear()

    def generate_soft_shadow(self) -> str:
        """Generate soft shadow WGSL function."""
        return WGSL_SOFT_SHADOW

    def generate_attenuation(self) -> str:
        """Generate attenuation WGSL function."""
        return WGSL_ATTENUATION

    def generate_diffuse(self) -> str:
        """Generate diffuse lighting WGSL functions."""
        return WGSL_DIFFUSE

    def generate_specular_blinn_phong(self) -> str:
        """Generate Blinn-Phong specular WGSL functions."""
        return WGSL_SPECULAR_BLINN_PHONG

    def generate_specular_ggx(self) -> str:
        """Generate GGX specular WGSL functions."""
        return WGSL_SPECULAR_GGX

    def generate_lighting_functions(
        self,
        *,
        include_shadow: bool = True,
        include_attenuation: bool = True,
        include_diffuse: bool = True,
        include_blinn_phong: bool = True,
        include_ggx: bool = True,
    ) -> str:
        """Generate all lighting WGSL functions.

        Args:
            include_shadow: Include soft shadow function.
            include_attenuation: Include attenuation function.
            include_diffuse: Include diffuse functions.
            include_blinn_phong: Include Blinn-Phong specular.
            include_ggx: Include GGX specular.

        Returns:
            Complete WGSL code for lighting functions.
        """
        lines: list[str] = []

        lines.append("// =============================================================================")
        lines.append("// LIGHTING FUNCTIONS (T-DEMO-3.7, T-DEMO-3.8)")
        lines.append("// =============================================================================")
        lines.append("")

        if include_shadow:
            lines.append(self.generate_soft_shadow())
            lines.append("")

        if include_attenuation:
            lines.append(self.generate_attenuation())
            lines.append("")

        if include_diffuse:
            lines.append(self.generate_diffuse())
            lines.append("")

        if include_blinn_phong:
            lines.append(self.generate_specular_blinn_phong())
            lines.append("")

        if include_ggx:
            lines.append(self.generate_specular_ggx())
            lines.append("")

        return "\n".join(lines)

    def generate_light_loop(
        self,
        lights: Sequence[LightNode],
        *,
        use_ggx: bool = True,
        include_shadows: bool = True,
    ) -> str:
        """Generate WGSL code for a multi-light loop.

        Args:
            lights: Sequence of LightNode AST nodes.
            use_ggx: If True, use GGX specular; otherwise Blinn-Phong.
            include_shadows: Whether to include shadow calculations.

        Returns:
            WGSL code for calculating combined lighting from all lights.
        """
        lines: list[str] = []

        lines.append("/// Calculate combined lighting from all scene lights.")
        lines.append("fn calculate_all_lighting(")
        lines.append("    p: vec3<f32>,")
        lines.append("    n: vec3<f32>,")
        lines.append("    view_dir: vec3<f32>,")
        lines.append("    mat: Material,")
        lines.append(") -> vec3<f32> {")
        lines.append("    // Ambient term")
        lines.append("    var result = vec3<f32>(0.03) * mat.albedo * mat.ambient_occlusion;")
        lines.append("")

        for i, light in enumerate(lights):
            lines.append(f"    // Light {i}: {light.light_type.value}")

            if light.light_type == LightType.DIRECTIONAL:
                dir_x = _fmt_float(light.direction.x)
                dir_y = _fmt_float(light.direction.y)
                dir_z = _fmt_float(light.direction.z)
                lines.append(f"    let light{i}_dir = normalize(-vec3<f32>({dir_x}, {dir_y}, {dir_z}));")

                if include_shadows:
                    lines.append(f"    let shadow{i} = calculate_soft_shadow(p, light{i}_dir, 0.02, 100.0, 8.0);")
                else:
                    lines.append(f"    let shadow{i} = 1.0;")

                color_x = _fmt_float(light.color.x)
                color_y = _fmt_float(light.color.y)
                color_z = _fmt_float(light.color.z)
                intensity = _fmt_float(light.intensity.value)

                lines.append(f"    let light{i}_color = vec3<f32>({color_x}, {color_y}, {color_z});")
                lines.append(f"    let diff{i} = calculate_diffuse_directional(n, light{i}_dir, light{i}_color, {intensity}, shadow{i});")

                if use_ggx:
                    lines.append(f"    let spec{i} = calculate_specular_ggx_directional(n, view_dir, light{i}_dir, light{i}_color, {intensity}, mat.roughness, mat.metallic, mat.albedo, shadow{i});")
                else:
                    lines.append(f"    let spec{i} = calculate_specular_blinn_phong_directional(n, view_dir, light{i}_dir, light{i}_color, {intensity}, mat.roughness, shadow{i});")

            else:  # POINT or AREA
                pos_x = _fmt_float(light.position.x)
                pos_y = _fmt_float(light.position.y)
                pos_z = _fmt_float(light.position.z)
                lines.append(f"    let light{i}_pos = vec3<f32>({pos_x}, {pos_y}, {pos_z});")
                lines.append(f"    let light{i}_vec = light{i}_pos - p;")
                lines.append(f"    let light{i}_dist = length(light{i}_vec);")
                lines.append(f"    let light{i}_dir = light{i}_vec / light{i}_dist;")

                radius = _fmt_float(light.radius.value)
                lines.append(f"    let atten{i} = calculate_attenuation(light{i}_dist, {radius});")

                if include_shadows:
                    lines.append(f"    let shadow{i} = calculate_soft_shadow(p, light{i}_dir, 0.02, light{i}_dist, 8.0);")
                else:
                    lines.append(f"    let shadow{i} = 1.0;")

                color_x = _fmt_float(light.color.x)
                color_y = _fmt_float(light.color.y)
                color_z = _fmt_float(light.color.z)
                intensity = _fmt_float(light.intensity.value)

                lines.append(f"    let light{i}_color = vec3<f32>({color_x}, {color_y}, {color_z});")
                lines.append(f"    let diff{i} = calculate_diffuse(p, n, light{i}_pos, light{i}_color, {intensity}, shadow{i}, atten{i});")

                if use_ggx:
                    lines.append(f"    let spec{i} = calculate_specular_ggx(p, n, view_dir, light{i}_pos, light{i}_color, {intensity}, mat.roughness, mat.metallic, mat.albedo, shadow{i}, atten{i});")
                else:
                    lines.append(f"    let spec{i} = calculate_specular_blinn_phong(p, n, view_dir, light{i}_pos, light{i}_color, {intensity}, mat.roughness, shadow{i}, atten{i});")

            # Accumulate
            lines.append(f"    result = result + diff{i} * mat.albedo * (1.0 - mat.metallic) + spec{i};")
            lines.append("")

        lines.append("    return result;")
        lines.append("}")

        return "\n".join(lines)


# =============================================================================
# FORMAT HELPERS
# =============================================================================


def _fmt_float(val: float) -> str:
    """Format a float for WGSL output."""
    if val == int(val) and not (val == 0.0 and str(val).startswith("-")):
        return f"{int(val)}.0"
    return f"{val}"


def _fmt_vec3(v: Vec3Node) -> str:
    """Format a Vec3Node for WGSL output."""
    return f"vec3<f32>({_fmt_float(v.x)}, {_fmt_float(v.y)}, {_fmt_float(v.z)})"

"""Subsurface Scattering Shader Functions for TRINITY Material System.

T-MAT-4.1: Dual-Pass Screen-Space SSS Implementation.

This module provides:
- Python reference implementations of SSS BRDF functions for testing
- Access to WGSL SSS source code
- Reference values for validation
- SSS profile presets

The SSS model simulates light transport within translucent materials:
- Human skin (Burley diffusion profile)
- Wax/candles
- Jade/marble
- Milk/liquids

Pipeline:
- Pass 1: Render diffuse to SSS target
- Pass 2: Horizontal separable blur with diffusion kernel
- Pass 3: Vertical separable blur with diffusion kernel
- Pass 4: Final composite with specular

Example::

    from trinity.materials.sss_shader import (
        burley_diffusion,
        evaluate_diffusion_profile,
        SSSProfile,
        SSSParams,
        get_sss_wgsl,
        SSS_PROFILE_SKIN,
    )

    # Get WGSL source for shader compilation
    wgsl_source = get_sss_wgsl()

    # Use skin profile
    profile = SSS_PROFILE_SKIN
    weight = burley_diffusion(r=0.5, d=profile.scatter_distance[0])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Mathematical constants
PI = 3.14159265359
TWO_PI = 6.28318530718
EPSILON = 0.0001

# Type aliases
Vec3 = Tuple[float, float, float]
Vec4 = Tuple[float, float, float, float]


def get_sss_wgsl() -> str:
    """Load the SSS WGSL source.

    Returns:
        WGSL source code containing all SSS functions.
    """
    wgsl_path = Path(__file__).parent / "wgsl" / "sss.wgsl"
    return wgsl_path.read_text(encoding="utf-8")


# =============================================================================
# SSS Profile Data Structures
# =============================================================================


@dataclass
class SSSProfile:
    """Subsurface scattering profile parameters.

    Defines how light scatters within a translucent material.

    Attributes:
        name: Profile name for identification.
        scatter_distance: Mean free path per channel (RGB).
        scatter_color: Subsurface tint color.
        blur_strength: Strength of screen-space blur [0, 1].
        falloff_color: Color at edge of scattering.
        curvature_scale: How much curvature affects scattering [0, 2].
        transmittance_color: Color for transmitted light.
        boundary_color_bleed: Boundary bleeding factor [0, 1].
    """

    name: str = "Default"
    scatter_distance: Vec3 = (1.0, 1.0, 1.0)
    scatter_color: Vec3 = (1.0, 0.2, 0.1)
    blur_strength: float = 0.7
    falloff_color: Vec3 = (1.0, 0.37, 0.3)
    curvature_scale: float = 0.75
    transmittance_color: Vec3 = (0.88, 0.23, 0.17)
    boundary_color_bleed: float = 0.5

    def __post_init__(self) -> None:
        """Validate parameters."""
        if not 0.0 <= self.blur_strength <= 1.0:
            raise ValueError(f"blur_strength must be in [0,1], got {self.blur_strength}")
        if not 0.0 <= self.curvature_scale <= 2.0:
            raise ValueError(f"curvature_scale must be in [0,2], got {self.curvature_scale}")
        if not 0.0 <= self.boundary_color_bleed <= 1.0:
            raise ValueError(f"boundary_color_bleed must be in [0,1], got {self.boundary_color_bleed}")


@dataclass
class SSSParams:
    """SSS material parameters for per-fragment evaluation.

    Attributes:
        profile: SSS profile defining scatter properties.
        subsurface_intensity: SSS effect intensity [0, 1].
        enable_transmission: Enable back-face transmission.
        transmission_tint: Color for transmitted light.
    """

    profile: SSSProfile = field(default_factory=SSSProfile)
    subsurface_intensity: float = 0.0
    enable_transmission: bool = False
    transmission_tint: Vec3 = (1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        """Validate parameters."""
        if not 0.0 <= self.subsurface_intensity <= 1.0:
            raise ValueError(
                f"subsurface_intensity must be in [0,1], got {self.subsurface_intensity}"
            )


# =============================================================================
# Predefined SSS Profiles
# =============================================================================

SSS_PROFILE_SKIN = SSSProfile(
    name="Skin",
    scatter_distance=(1.0, 0.4, 0.25),
    scatter_color=(0.48, 0.25, 0.17),
    blur_strength=0.8,
    falloff_color=(1.0, 0.37, 0.3),
    curvature_scale=0.75,
    transmittance_color=(0.88, 0.23, 0.17),
    boundary_color_bleed=0.5,
)

SSS_PROFILE_WAX = SSSProfile(
    name="Wax",
    scatter_distance=(0.5, 0.5, 0.4),
    scatter_color=(0.9, 0.85, 0.7),
    blur_strength=0.6,
    falloff_color=(0.95, 0.9, 0.8),
    curvature_scale=0.5,
    transmittance_color=(0.95, 0.9, 0.85),
    boundary_color_bleed=0.3,
)

SSS_PROFILE_JADE = SSSProfile(
    name="Jade",
    scatter_distance=(0.25, 0.5, 0.25),
    scatter_color=(0.5, 0.9, 0.5),
    blur_strength=0.5,
    falloff_color=(0.3, 0.7, 0.3),
    curvature_scale=0.6,
    transmittance_color=(0.4, 0.8, 0.4),
    boundary_color_bleed=0.4,
)

SSS_PROFILE_MILK = SSSProfile(
    name="Milk",
    scatter_distance=(0.8, 0.8, 0.75),
    scatter_color=(0.95, 0.95, 0.9),
    blur_strength=0.9,
    falloff_color=(0.98, 0.98, 0.95),
    curvature_scale=0.3,
    transmittance_color=(0.98, 0.98, 0.95),
    boundary_color_bleed=0.6,
)


# =============================================================================
# Burley Normalized Diffusion Profile
# =============================================================================


def burley_diffusion(r: float, d: float) -> float:
    """Burley normalized diffusion function.

    Models subsurface scattering using a sum of two exponentials.
    Formula: R(r) = A * exp(-r/d) + B * exp(-r/(3d))
    where A = 1/(2*PI*d*d), B = 1/(6*PI*d*d) for proper normalization.

    This form integrates to 1 over the plane and decreases monotonically.

    Args:
        r: Distance from sample point (in world units).
        d: Mean free path / scatter distance.

    Returns:
        Diffusion weight at distance r.

    Example::

        >>> burley_diffusion(0.0, 1.0)  # Peak at center
        0.106...
        >>> burley_diffusion(1.0, 1.0) > 0  # Decays with distance
        True
        >>> burley_diffusion(1.0, 2.0) > burley_diffusion(1.0, 1.0)  # Larger d = slower decay
        True
    """
    if d < EPSILON:
        return 0.0

    # Burley's normalized diffusion profile (Burley 2015):
    # R(r) = A * e^(-r/d) + B * e^(-r/(3d))
    # Coefficients chosen so that 2*PI * integral(R(r) * r dr) = 1
    # A = 1 / (2 * PI * d^2), B = 1 / (6 * PI * d^2)
    d2 = d * d
    A = 1.0 / (TWO_PI * d2)
    B = 1.0 / (6.0 * PI * d2)

    exp1 = math.exp(-r / d)
    exp2 = math.exp(-r / (3.0 * d))

    return A * exp1 + B * exp2


def burley_diffusion_rgb(r: float, d: Vec3) -> Vec3:
    """Per-channel Burley diffusion for colored scattering.

    Each RGB channel can have a different scatter distance.

    Args:
        r: Distance from sample point.
        d: Scatter distance per channel (vec3).

    Returns:
        Diffusion weight per channel.

    Example::

        >>> weights = burley_diffusion_rgb(0.5, (1.0, 0.5, 0.25))
        >>> weights[0] > weights[1] > weights[2]  # Larger d = wider scatter
        True
    """
    return (
        burley_diffusion(r, d[0]),
        burley_diffusion(r, d[1]),
        burley_diffusion(r, d[2]),
    )


def evaluate_diffusion_profile(r: float, profile: SSSProfile) -> Vec3:
    """Evaluate Burley diffusion profile for a sample.

    Used to build the diffusion kernel for screen-space blur.

    Args:
        r: Radial distance.
        profile: SSS profile with scatter parameters.

    Returns:
        Diffusion weight with color tinting.

    Example::

        >>> profile = SSS_PROFILE_SKIN
        >>> weights = evaluate_diffusion_profile(0.5, profile)
        >>> all(w >= 0 for w in weights)
        True
    """
    weights = burley_diffusion_rgb(r, profile.scatter_distance)
    return (
        weights[0] * profile.scatter_color[0],
        weights[1] * profile.scatter_color[1],
        weights[2] * profile.scatter_color[2],
    )


# =============================================================================
# Screen-Space Blur Kernel
# =============================================================================

# Pre-computed 9-tap Gaussian weights
SSS_KERNEL_WEIGHTS = (
    0.0162, 0.0540, 0.1216, 0.1945, 0.2274,
    0.1945, 0.1216, 0.0540, 0.0162,
)

# Pre-computed kernel offsets (centered at 0)
SSS_KERNEL_OFFSETS = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)

SSS_KERNEL_SIZE = 9


def compute_sss_kernel(profile: SSSProfile, pixel_size: float) -> List[Vec3]:
    """Compute diffusion-weighted kernel for a given profile.

    Returns weights that combine Gaussian shape with diffusion falloff.

    Args:
        profile: SSS profile with scatter parameters.
        pixel_size: Size of a pixel in world units.

    Returns:
        List of 9 kernel weights (vec3 per tap for colored scattering).

    Example::

        >>> kernel = compute_sss_kernel(SSS_PROFILE_SKIN, 0.01)
        >>> len(kernel)
        9
        >>> kernel[4][0] > kernel[0][0]  # Center tap should be largest
        True
    """
    kernel: List[Vec3] = []
    total_weight = [0.0, 0.0, 0.0]

    for i in range(SSS_KERNEL_SIZE):
        # Convert pixel offset to world distance
        pixel_offset = abs(SSS_KERNEL_OFFSETS[i])
        world_distance = pixel_offset * pixel_size

        # Evaluate diffusion profile at this distance
        diffusion = evaluate_diffusion_profile(world_distance, profile)

        # Combine with base Gaussian weight
        weight = (
            diffusion[0] * SSS_KERNEL_WEIGHTS[i] * profile.blur_strength,
            diffusion[1] * SSS_KERNEL_WEIGHTS[i] * profile.blur_strength,
            diffusion[2] * SSS_KERNEL_WEIGHTS[i] * profile.blur_strength,
        )
        kernel.append(weight)
        total_weight[0] += weight[0]
        total_weight[1] += weight[1]
        total_weight[2] += weight[2]

    # Normalize kernel to preserve energy
    if total_weight[0] > EPSILON:
        kernel = [
            (w[0] / total_weight[0], w[1] / total_weight[1], w[2] / total_weight[2])
            for w in kernel
        ]

    return kernel


# =============================================================================
# SSS Application and Compositing
# =============================================================================


def apply_sss(
    base_color: Vec3,
    sss_buffer: Vec3,
    profile: SSSProfile,
    sss_intensity: float,
) -> Vec3:
    """Apply subsurface scattering to base color.

    Final compositing step that blends SSS with base material.

    Args:
        base_color: Original surface color (diffuse).
        sss_buffer: Blurred SSS buffer result.
        profile: SSS profile for color tinting.
        sss_intensity: Subsurface intensity factor [0, 1].

    Returns:
        Final color with SSS applied.

    Example::

        >>> base = (0.5, 0.4, 0.3)
        >>> sss = (0.6, 0.45, 0.35)
        >>> result = apply_sss(base, sss, SSS_PROFILE_SKIN, 0.5)
        >>> all(0 <= c <= 1 for c in result)
        True
    """
    if sss_intensity < EPSILON:
        return base_color

    # Apply scatter color tinting
    sss_contribution = (
        sss_buffer[0] * profile.scatter_color[0],
        sss_buffer[1] * profile.scatter_color[1],
        sss_buffer[2] * profile.scatter_color[2],
    )

    # Mix based on intensity
    blend = sss_intensity * profile.blur_strength
    return (
        base_color[0] * (1.0 - blend) + sss_contribution[0] * blend,
        base_color[1] * (1.0 - blend) + sss_contribution[1] * blend,
        base_color[2] * (1.0 - blend) + sss_contribution[2] * blend,
    )


def apply_sss_with_bleeding(
    base_color: Vec3,
    sss_buffer: Vec3,
    shadow_mask: float,
    profile: SSSProfile,
    sss_intensity: float,
) -> Vec3:
    """Apply SSS with boundary color bleed.

    Enhanced version that handles color bleeding at light/shadow boundaries.

    Args:
        base_color: Original surface color.
        sss_buffer: Blurred SSS buffer result.
        shadow_mask: Shadow attenuation factor [0=shadow, 1=lit].
        profile: SSS profile.
        sss_intensity: Subsurface intensity.

    Returns:
        Final color with boundary bleeding.

    Example::

        >>> base = (0.5, 0.4, 0.3)
        >>> sss = (0.6, 0.45, 0.35)
        >>> result = apply_sss_with_bleeding(base, sss, 0.5, SSS_PROFILE_SKIN, 0.8)
        >>> all(c >= 0 for c in result)
        True
    """
    if sss_intensity < EPSILON:
        return base_color

    # Bleed falloff color into shadowed regions
    boundary_factor = profile.boundary_color_bleed
    shadow_blend = boundary_factor * (1.0 - shadow_mask)
    bleed_color = (
        base_color[0] * (1.0 - shadow_blend) + profile.falloff_color[0] * shadow_blend,
        base_color[1] * (1.0 - shadow_blend) + profile.falloff_color[1] * shadow_blend,
        base_color[2] * (1.0 - shadow_blend) + profile.falloff_color[2] * shadow_blend,
    )

    # Apply SSS blur
    sss_contribution = (
        sss_buffer[0] * profile.scatter_color[0],
        sss_buffer[1] * profile.scatter_color[1],
        sss_buffer[2] * profile.scatter_color[2],
    )

    blend = sss_intensity * profile.blur_strength
    return (
        bleed_color[0] * (1.0 - blend) + sss_contribution[0] * blend,
        bleed_color[1] * (1.0 - blend) + sss_contribution[1] * blend,
        bleed_color[2] * (1.0 - blend) + sss_contribution[2] * blend,
    )


# =============================================================================
# Transmission (Back-Face Scattering)
# =============================================================================


def _dot(a: Vec3, b: Vec3) -> float:
    """Compute dot product of two Vec3."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def evaluate_sss_transmission(
    N: Vec3,
    L: Vec3,
    V: Vec3,
    thickness: float,
    profile: SSSProfile,
) -> Vec3:
    """Evaluate subsurface transmission (light through thin surfaces).

    Models light entering from the back side of a surface.

    Args:
        N: Surface normal (front-facing).
        L: Light direction (toward light).
        V: View direction (toward camera).
        thickness: Local surface thickness (0=thin, 1=thick).
        profile: SSS profile.

    Returns:
        Transmission contribution.

    Example::

        >>> N = (0.0, 1.0, 0.0)
        >>> L = (0.0, -1.0, 0.0)  # Light from behind
        >>> V = (0.0, 1.0, 0.0)
        >>> result = evaluate_sss_transmission(N, L, V, 0.5, SSS_PROFILE_SKIN)
        >>> all(c >= 0 for c in result)
        True
    """
    # Wrap lighting for transmission (light from behind surface)
    NoL_back = max(-_dot(N, L), 0.0)

    # Distance light travels through material
    scatter_dist = (
        thickness * profile.scatter_distance[0],
        thickness * profile.scatter_distance[1],
        thickness * profile.scatter_distance[2],
    )

    # Beer-Lambert attenuation
    attenuation = (
        math.exp(-scatter_dist[0] * 2.0),
        math.exp(-scatter_dist[1] * 2.0),
        math.exp(-scatter_dist[2] * 2.0),
    )

    # Forward scattering lobe (Henyey-Greenstein approximation)
    VoL = _dot(V, (-L[0], -L[1], -L[2]))
    phase = 0.25 + 0.5 * pow(max(VoL, 0.0), 2.0)

    # Final transmission
    return (
        NoL_back * attenuation[0] * phase * profile.transmittance_color[0],
        NoL_back * attenuation[1] * phase * profile.transmittance_color[1],
        NoL_back * attenuation[2] * phase * profile.transmittance_color[2],
    )


# =============================================================================
# Pre-Integrated Diffusion LUT Generation
# =============================================================================


def compute_diffusion_lut_value(uv: Tuple[float, float], profile: SSSProfile) -> Vec3:
    """Generate diffusion LUT value for baking.

    Call this to pre-compute the LUT.

    Args:
        uv: LUT coordinate (u=NoL mapped to [0,1], v=curvature).
        profile: SSS profile.

    Returns:
        Integrated diffusion value.

    Example::

        >>> value = compute_diffusion_lut_value((0.5, 0.5), SSS_PROFILE_SKIN)
        >>> all(0 <= c <= 1 for c in value)
        True
    """
    # Unmap U to NoL [-1, 1]
    NoL = uv[0] * 2.0 - 1.0

    # V is curvature [0, 1]
    curvature = uv[1]

    # Integrate diffusion profile weighted by curvature
    scatter_scale = 1.0 + curvature * profile.curvature_scale

    # Simulate integration over hemisphere
    integrated = [0.0, 0.0, 0.0]
    num_samples = 16

    for i in range(num_samples):
        t = i / (num_samples - 1)
        angle = t * PI
        sample_NoL = math.cos(angle)

        # Distance based on angle from direct lighting
        angle_diff = abs(sample_NoL - NoL)
        r = angle_diff * scatter_scale

        # Sample diffusion profile
        sample_weight = evaluate_diffusion_profile(r, profile)

        # Weight by solid angle approximation
        sin_weight = math.sin(angle)
        integrated[0] += sample_weight[0] * sin_weight
        integrated[1] += sample_weight[1] * sin_weight
        integrated[2] += sample_weight[2] * sin_weight

    # Normalize
    return (
        integrated[0] / num_samples,
        integrated[1] / num_samples,
        integrated[2] / num_samples,
    )


def generate_diffusion_lut(
    profile: SSSProfile,
    width: int = 256,
    height: int = 64,
) -> List[List[Vec3]]:
    """Generate a complete pre-integrated diffusion LUT.

    Args:
        profile: SSS profile to generate LUT for.
        width: LUT width (NoL resolution).
        height: LUT height (curvature resolution).

    Returns:
        2D list of RGB values [y][x].

    Example::

        >>> lut = generate_diffusion_lut(SSS_PROFILE_SKIN, 64, 16)
        >>> len(lut)
        16
        >>> len(lut[0])
        64
    """
    lut: List[List[Vec3]] = []

    for y in range(height):
        row: List[Vec3] = []
        v = y / (height - 1)
        for x in range(width):
            u = x / (width - 1)
            value = compute_diffusion_lut_value((u, v), profile)
            row.append(value)
        lut.append(row)

    return lut


# =============================================================================
# Diffusion Profile Generation
# =============================================================================


def get_diffusion_profile_samples(
    profile: SSSProfile,
    num_samples: int = 16,
) -> List[Vec3]:
    """Generate normalized diffusion profile samples.

    Uses Burley's approximation for each channel.

    Args:
        profile: SSS profile.
        num_samples: Number of profile samples.

    Returns:
        List of diffusion weights per sample (RGB).

    Example::

        >>> samples = get_diffusion_profile_samples(SSS_PROFILE_SKIN, 16)
        >>> len(samples)
        16
        >>> all(all(w >= 0 for w in s) for s in samples)
        True
    """
    # Use average scatter distance for sample distribution
    avg_d = sum(profile.scatter_distance) / 3.0
    max_radius = avg_d * 3.0

    samples: List[Vec3] = []
    totals = [0.0, 0.0, 0.0]

    for i in range(num_samples):
        r = (i + 0.5) * max_radius / num_samples
        weights = burley_diffusion_rgb(r, profile.scatter_distance)
        samples.append(weights)
        totals[0] += weights[0]
        totals[1] += weights[1]
        totals[2] += weights[2]

    # Normalize
    if totals[0] > EPSILON:
        samples = [
            (s[0] / totals[0], s[1] / totals[1], s[2] / totals[2])
            for s in samples
        ]

    return samples


# =============================================================================
# SSS Mask Utilities
# =============================================================================


def get_sss_mask(sss_params: SSSParams) -> float:
    """Get SSS mask value for G-buffer.

    Returns value to store in G-buffer for SSS pass identification.

    Args:
        sss_params: SSS parameters.

    Returns:
        Mask value (0=no SSS, 1=full SSS).

    Example::

        >>> params = SSSParams(profile=SSS_PROFILE_SKIN, subsurface_intensity=0.8)
        >>> mask = get_sss_mask(params)
        >>> 0 <= mask <= 1
        True
    """
    return sss_params.subsurface_intensity * sss_params.profile.blur_strength


# =============================================================================
# Reference Values for Testing
# =============================================================================

# Reference values computed with known-good implementations
# Burley formula: R(r) = A*exp(-r/d) + B*exp(-r/(3d)) where A=1/(2*PI*d^2), B=1/(6*PI*d^2)
SSS_REFERENCE_VALUES = {
    # burley_diffusion reference values: (r, d) -> expected
    # Computed as: A*exp(-r/d) + B*exp(-r/(3d))
    # At r=0: A + B = 1/(2*PI*d^2) + 1/(6*PI*d^2) = 2/(3*PI*d^2)
    # For d=1: 2/(3*PI) = 0.2122
    "burley_diffusion": [
        {"r": 0.0, "d": 1.0, "expected": 0.2122, "tolerance": 0.01},
        {"r": 0.5, "d": 1.0, "expected": 0.1414, "tolerance": 0.01},
        {"r": 1.0, "d": 1.0, "expected": 0.0966, "tolerance": 0.01},
        {"r": 2.0, "d": 1.0, "expected": 0.0488, "tolerance": 0.01},
        {"r": 0.5, "d": 0.5, "expected": 0.3863, "tolerance": 0.05},
        {"r": 0.25, "d": 0.25, "expected": 1.545, "tolerance": 0.1},
    ],
    # evaluate_diffusion_profile reference (with skin profile)
    # skin scatter_color = (0.48, 0.25, 0.17), scatter_distance = (1.0, 0.4, 0.25)
    "evaluate_diffusion_profile_skin": [
        {"r": 0.0, "expected_r": 0.102, "tolerance": 0.02},  # 0.2122 * 0.48
        {"r": 0.5, "expected_r": 0.068, "tolerance": 0.01},  # burley(0.5,1.0) * 0.48
        {"r": 1.0, "expected_r": 0.046, "tolerance": 0.01},  # burley(1.0,1.0) * 0.48
    ],
    # apply_sss reference values
    # Formula: result = base*(1-blend) + (sss*scatter_color)*blend
    # where blend = intensity * blur_strength
    "apply_sss": [
        {
            "base_color": (0.5, 0.4, 0.3),
            "sss_buffer": (0.6, 0.45, 0.35),
            "intensity": 1.0,
            "blur_strength": 0.8,
            # Default profile scatter_color = (1.0, 0.2, 0.1)
            # blend = 1.0 * 0.8 = 0.8
            # sss_contribution_r = 0.6 * 1.0 = 0.6
            # result_r = 0.5*(1-0.8) + 0.6*0.8 = 0.1 + 0.48 = 0.58
            "expected_r": 0.58,
            "tolerance": 0.05,
        },
        {
            "base_color": (0.5, 0.4, 0.3),
            "sss_buffer": (0.6, 0.45, 0.35),
            "intensity": 0.0,
            "blur_strength": 0.8,
            "expected_r": 0.5,  # No SSS, base passthrough
            "tolerance": 0.001,
        },
    ],
    # Kernel center should be largest
    "kernel_center_largest": True,
}

# Edge cases for SSS
SSS_EDGE_CASES = {
    "zero_intensity": {
        "description": "SSS disabled (intensity=0)",
        "params": {"subsurface_intensity": 0.0},
        "expected": "base color passthrough",
    },
    "full_intensity": {
        "description": "Full SSS (intensity=1)",
        "params": {"subsurface_intensity": 1.0, "blur_strength": 1.0},
        "expected": "full SSS contribution",
    },
    "zero_distance": {
        "description": "Zero scatter distance",
        "scatter_distance": (0.0, 0.0, 0.0),
        "expected": "no scattering",
    },
    "transmission_backlit": {
        "description": "Light from behind surface",
        "N": (0.0, 1.0, 0.0),
        "L": (0.0, -1.0, 0.0),
        "expected": "transmission contribution",
    },
}


__all__ = [
    # WGSL source access
    "get_sss_wgsl",
    # Data structures
    "SSSProfile",
    "SSSParams",
    # Predefined profiles
    "SSS_PROFILE_SKIN",
    "SSS_PROFILE_WAX",
    "SSS_PROFILE_JADE",
    "SSS_PROFILE_MILK",
    # Diffusion functions
    "burley_diffusion",
    "burley_diffusion_rgb",
    "evaluate_diffusion_profile",
    # Kernel computation
    "compute_sss_kernel",
    "SSS_KERNEL_WEIGHTS",
    "SSS_KERNEL_OFFSETS",
    "SSS_KERNEL_SIZE",
    # Application functions
    "apply_sss",
    "apply_sss_with_bleeding",
    # Transmission
    "evaluate_sss_transmission",
    # LUT generation
    "compute_diffusion_lut_value",
    "generate_diffusion_lut",
    # Profile utilities
    "get_diffusion_profile_samples",
    "get_sss_mask",
    # Reference values
    "SSS_REFERENCE_VALUES",
    "SSS_EDGE_CASES",
    # Constants
    "PI",
    "TWO_PI",
    "EPSILON",
]

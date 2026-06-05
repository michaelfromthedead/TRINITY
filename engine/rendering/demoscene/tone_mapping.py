"""
Tone Mapping for Demoscene Rendering (T-DEMO-3.11).

This module provides tone mapping operators for converting HDR colors to
displayable LDR range [0, 1], including:

  - Reinhard simple: c / (1 + c)
  - Reinhard extended with white point control
  - ACES filmic curve (Academy Color Encoding System)
  - Uncharted 2 operator (John Hable's filmic curve)

All operators work on linear RGB colors and output gamma-corrected sRGB.

Usage:
    >>> from engine.rendering.demoscene.tone_mapping import ToneMapper, Vec3
    >>> mapper = ToneMapper()
    >>> hdr_color = Vec3(2.5, 1.2, 0.8)
    >>> ldr_color = mapper.apply(hdr_color, operator="aces")
    >>> print(ldr_color)  # All components in [0, 1]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .ray_generation import Vec3


# =============================================================================
# Tone Mapping Operators (Enums)
# =============================================================================


class ToneMappingOperator(Enum):
    """Available tone mapping operators."""
    REINHARD = "reinhard"
    REINHARD_EXTENDED = "reinhard_extended"
    ACES = "aces"
    UNCHARTED2 = "uncharted2"
    LINEAR = "linear"  # No tone mapping, just clamp


# =============================================================================
# Tone Mapping Functions
# =============================================================================


def reinhard(color: Vec3) -> Vec3:
    """
    Simple Reinhard tone mapping operator.

    Maps HDR values to [0, 1] range using: c / (1 + c)

    This operator has a soft shoulder that prevents hard clipping,
    but can desaturate highlights somewhat.

    Args:
        color: Linear HDR color (RGB).

    Returns:
        Tone-mapped color in [0, 1] range.

    Example:
        >>> reinhard(Vec3(1.0, 1.0, 1.0))
        Vec3(x=0.5, y=0.5, z=0.5)
        >>> reinhard(Vec3(0.0, 0.0, 0.0))
        Vec3(x=0.0, y=0.0, z=0.0)
    """
    return Vec3(
        color.x / (1.0 + color.x),
        color.y / (1.0 + color.y),
        color.z / (1.0 + color.z),
    )


def reinhard_extended(color: Vec3, max_white: float = 4.0) -> Vec3:
    """
    Extended Reinhard tone mapping with white point control.

    Maps HDR values using: c * (1 + c / Lw^2) / (1 + c)
    where Lw is the maximum white luminance.

    This allows controlling which HDR value maps to pure white,
    providing more control over highlight rendering.

    Args:
        color: Linear HDR color (RGB).
        max_white: Luminance value that maps to pure white.
                   Higher values preserve more highlight detail.

    Returns:
        Tone-mapped color in [0, 1] range.

    Example:
        >>> reinhard_extended(Vec3(4.0, 4.0, 4.0), max_white=4.0)
        Vec3(x=1.0, y=1.0, z=1.0)  # approximately
    """
    max_white_sq = max_white * max_white
    max_white_sq = max(max_white_sq, 1e-10)  # Prevent division by zero

    def map_channel(c: float) -> float:
        numerator = c * (1.0 + c / max_white_sq)
        denominator = 1.0 + c
        return numerator / denominator

    return Vec3(
        map_channel(color.x),
        map_channel(color.y),
        map_channel(color.z),
    )


def aces_filmic(color: Vec3) -> Vec3:
    """
    ACES filmic tone mapping operator.

    Uses the Academy Color Encoding System approximation (Stephen Hill's fit).
    This is widely used in games and film for its natural highlight rolloff
    and pleasing color preservation.

    The curve: (x * (a*x + b)) / (x * (c*x + d) + e)
    with: a=2.51, b=0.03, c=2.43, d=0.59, e=0.14

    Args:
        color: Linear HDR color (RGB).

    Returns:
        Tone-mapped color in [0, 1] range.

    Example:
        >>> aces_filmic(Vec3(1.0, 1.0, 1.0))  # Mid-gray maps to ~0.5
    """
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14

    def map_channel(x: float) -> float:
        # Clamp negative values
        x = max(x, 0.0)
        result = (x * (a * x + b)) / (x * (c * x + d) + e)
        return max(0.0, min(1.0, result))

    return Vec3(
        map_channel(color.x),
        map_channel(color.y),
        map_channel(color.z),
    )


def uncharted2(color: Vec3, exposure: float = 2.0) -> Vec3:
    """
    Uncharted 2 filmic tone mapping operator (John Hable).

    This operator was designed for the game Uncharted 2 and provides
    excellent highlight rolloff with good shadow detail preservation.

    Uses the curve: ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F
    with constants tuned for natural film response.

    Args:
        color: Linear HDR color (RGB).
        exposure: Exposure multiplier (default 2.0).

    Returns:
        Tone-mapped color in [0, 1] range.

    Example:
        >>> uncharted2(Vec3(1.0, 1.0, 1.0))
    """
    # Curve parameters
    A = 0.15  # Shoulder strength
    B = 0.50  # Linear strength
    C = 0.10  # Linear angle
    D = 0.20  # Toe strength
    E = 0.02  # Toe numerator
    F = 0.30  # Toe denominator
    W = 11.2  # Linear white point

    def tonemap_partial(x: float) -> float:
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

    def map_channel(c: float) -> float:
        # Apply exposure
        c = max(c * exposure, 0.0)
        # Apply curve
        curr = tonemap_partial(c)
        white_scale = 1.0 / tonemap_partial(W)
        result = curr * white_scale
        return max(0.0, min(1.0, result))

    return Vec3(
        map_channel(color.x),
        map_channel(color.y),
        map_channel(color.z),
    )


def linear_clamp(color: Vec3) -> Vec3:
    """
    Linear clamping (no tone mapping).

    Simply clamps values to [0, 1] range. This is not recommended
    for HDR content as it causes harsh clipping.

    Args:
        color: Linear color (RGB).

    Returns:
        Clamped color in [0, 1] range.
    """
    return Vec3(
        max(0.0, min(1.0, color.x)),
        max(0.0, min(1.0, color.y)),
        max(0.0, min(1.0, color.z)),
    )


# =============================================================================
# Gamma Correction
# =============================================================================


def gamma_correct(color: Vec3, gamma: float = 2.2) -> Vec3:
    """
    Apply gamma correction for display.

    Converts linear color to display gamma space using: c^(1/gamma)

    Standard display gamma is 2.2 (sRGB).

    Args:
        color: Linear color (RGB) in [0, 1].
        gamma: Display gamma value (default 2.2).

    Returns:
        Gamma-corrected color.
    """
    inv_gamma = 1.0 / gamma

    def correct_channel(c: float) -> float:
        # Handle negative values (shouldn't happen, but be safe)
        if c <= 0.0:
            return 0.0
        return pow(c, inv_gamma)

    return Vec3(
        correct_channel(color.x),
        correct_channel(color.y),
        correct_channel(color.z),
    )


def linear_to_srgb(color: Vec3) -> Vec3:
    """
    Convert linear color to sRGB using the official transfer function.

    Uses the piecewise sRGB OETF:
    - If c <= 0.0031308: 12.92 * c
    - Else: 1.055 * c^(1/2.4) - 0.055

    Args:
        color: Linear color (RGB) in [0, 1].

    Returns:
        sRGB color.
    """
    def convert_channel(c: float) -> float:
        c = max(0.0, min(1.0, c))
        if c <= 0.0031308:
            return 12.92 * c
        else:
            return 1.055 * pow(c, 1.0 / 2.4) - 0.055

    return Vec3(
        convert_channel(color.x),
        convert_channel(color.y),
        convert_channel(color.z),
    )


def srgb_to_linear(color: Vec3) -> Vec3:
    """
    Convert sRGB color to linear using the official transfer function.

    Uses the piecewise sRGB EOTF (inverse of OETF):
    - If c <= 0.04045: c / 12.92
    - Else: ((c + 0.055) / 1.055)^2.4

    Args:
        color: sRGB color (RGB) in [0, 1].

    Returns:
        Linear color.
    """
    def convert_channel(c: float) -> float:
        c = max(0.0, min(1.0, c))
        if c <= 0.04045:
            return c / 12.92
        else:
            return pow((c + 0.055) / 1.055, 2.4)

    return Vec3(
        convert_channel(color.x),
        convert_channel(color.y),
        convert_channel(color.z),
    )


# =============================================================================
# ToneMapper Class
# =============================================================================


class ToneMapper:
    """
    Configurable tone mapper for HDR to LDR conversion.

    Provides a unified interface for applying various tone mapping operators
    with gamma correction. Supports:

    - Multiple tone mapping algorithms
    - Configurable gamma correction
    - Exposure control
    - Batch processing

    Usage::

        mapper = ToneMapper(default_operator="aces", gamma=2.2)
        ldr_color = mapper.apply(hdr_color)

        # Or specify operator per call
        ldr_color = mapper.apply(hdr_color, operator="reinhard")

        # Process multiple colors
        ldr_colors = mapper.apply_batch(hdr_colors)
    """

    def __init__(
        self,
        default_operator: str = "aces",
        gamma: float = 2.2,
        exposure: float = 1.0,
        use_srgb_conversion: bool = False,
    ) -> None:
        """
        Initialize the tone mapper.

        Args:
            default_operator: Default tone mapping operator
                              ("reinhard", "reinhard_extended", "aces", "uncharted2", "linear").
            gamma: Display gamma for correction (default 2.2).
            exposure: Global exposure multiplier (default 1.0).
            use_srgb_conversion: If True, use official sRGB conversion instead of
                                 simple gamma (default False).
        """
        self.default_operator = default_operator
        self.gamma = gamma
        self.exposure = exposure
        self.use_srgb_conversion = use_srgb_conversion

        # Extended reinhard parameters
        self.max_white = 4.0

    def apply(
        self,
        color: Vec3,
        operator: Optional[str] = None,
        apply_gamma: bool = True,
    ) -> Vec3:
        """
        Apply tone mapping and gamma correction to a color.

        Args:
            color: Linear HDR color (RGB).
            operator: Tone mapping operator name (uses default if None).
            apply_gamma: Whether to apply gamma correction (default True).

        Returns:
            Tone-mapped and gamma-corrected color in [0, 1] range.

        Raises:
            ValueError: If operator name is invalid.
        """
        op_name = operator or self.default_operator

        # Apply exposure
        exposed_color = Vec3(
            color.x * self.exposure,
            color.y * self.exposure,
            color.z * self.exposure,
        )

        # Apply tone mapping
        if op_name == "reinhard":
            mapped = reinhard(exposed_color)
        elif op_name == "reinhard_extended":
            mapped = reinhard_extended(exposed_color, self.max_white)
        elif op_name == "aces":
            mapped = aces_filmic(exposed_color)
        elif op_name == "uncharted2":
            mapped = uncharted2(exposed_color, exposure=1.0)  # exposure already applied
        elif op_name == "linear":
            mapped = linear_clamp(exposed_color)
        else:
            raise ValueError(f"Unknown tone mapping operator: {op_name}")

        # Apply gamma correction
        if apply_gamma:
            if self.use_srgb_conversion:
                return linear_to_srgb(mapped)
            else:
                return gamma_correct(mapped, self.gamma)

        return mapped

    def apply_batch(
        self,
        colors: list[Vec3],
        operator: Optional[str] = None,
        apply_gamma: bool = True,
    ) -> list[Vec3]:
        """
        Apply tone mapping to multiple colors.

        Args:
            colors: List of linear HDR colors.
            operator: Tone mapping operator name.
            apply_gamma: Whether to apply gamma correction.

        Returns:
            List of tone-mapped colors.
        """
        return [self.apply(c, operator, apply_gamma) for c in colors]

    def set_exposure(self, exposure: float) -> None:
        """Set the exposure multiplier."""
        self.exposure = exposure

    def set_max_white(self, max_white: float) -> None:
        """Set the max white point for extended Reinhard."""
        self.max_white = max(max_white, 0.01)


# =============================================================================
# WGSL Code Generation
# =============================================================================


def generate_tone_mapping_wgsl(
    operator: str = "aces",
    include_all: bool = False,
) -> str:
    """
    Generate WGSL code for tone mapping functions.

    Args:
        operator: Primary operator to include ("aces", "reinhard", "uncharted2").
        include_all: If True, include all operators.

    Returns:
        WGSL code string with tone mapping functions.
    """
    lines: list[str] = []

    lines.append("// =============================================================================")
    lines.append("// Tone Mapping Functions (T-DEMO-3.11)")
    lines.append("// =============================================================================")
    lines.append("")

    # Always include gamma correction
    lines.append(WGSL_GAMMA_CORRECTION)

    # Include sRGB conversion
    lines.append(WGSL_LINEAR_TO_SRGB)

    # Include requested operators
    if include_all or operator == "reinhard":
        lines.append(WGSL_REINHARD)
        lines.append(WGSL_REINHARD_EXTENDED)

    if include_all or operator == "aces":
        lines.append(WGSL_ACES_FILMIC)

    if include_all or operator == "uncharted2":
        lines.append(WGSL_UNCHARTED2)

    # Include a unified entry point
    lines.append(generate_tone_map_entry_wgsl(operator))

    return "\n".join(lines)


WGSL_GAMMA_CORRECTION = """\
/// Applies gamma correction for display output.
/// Standard display gamma is 2.2.
fn gamma_correct(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    let inv_gamma = 1.0 / gamma;
    return pow(max(color, vec3<f32>(0.0)), vec3<f32>(inv_gamma));
}
"""

WGSL_LINEAR_TO_SRGB = """\
/// Converts linear color to sRGB using the official transfer function.
fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    let cutoff = step(color, vec3<f32>(0.0031308));
    let low = color * 12.92;
    let high = 1.055 * pow(color, vec3<f32>(1.0 / 2.4)) - 0.055;
    return mix(high, low, cutoff);
}
"""

WGSL_REINHARD = """\
/// Simple Reinhard tone mapping: c / (1 + c)
/// Maps HDR values to [0, 1] with soft highlight rolloff.
fn tone_map_reinhard(color: vec3<f32>) -> vec3<f32> {
    return color / (vec3<f32>(1.0) + color);
}
"""

WGSL_REINHARD_EXTENDED = """\
/// Extended Reinhard tone mapping with white point control.
/// max_white controls which luminance value maps to pure white.
fn tone_map_reinhard_extended(color: vec3<f32>, max_white: f32) -> vec3<f32> {
    let max_white_sq = max_white * max_white;
    let numerator = color * (vec3<f32>(1.0) + color / max_white_sq);
    let denominator = vec3<f32>(1.0) + color;
    return numerator / denominator;
}
"""

WGSL_ACES_FILMIC = """\
/// ACES filmic tone mapping (Stephen Hill approximation).
/// Widely used in games and film for natural highlight rolloff.
fn tone_map_aces(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    let mapped = (color * (a * color + b)) / (color * (c * color + d) + e);
    return clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0));
}
"""

WGSL_UNCHARTED2 = """\
/// Uncharted 2 filmic tone mapping (John Hable).
/// Excellent highlight rolloff with good shadow detail.
fn uc2_tonemap_partial(x: f32) -> f32 {
    let A = 0.15;  // Shoulder strength
    let B = 0.50;  // Linear strength
    let C = 0.10;  // Linear angle
    let D = 0.20;  // Toe strength
    let E = 0.02;  // Toe numerator
    let F = 0.30;  // Toe denominator
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

fn tone_map_uncharted2(color: vec3<f32>, exposure: f32) -> vec3<f32> {
    let W = 11.2;  // Linear white point
    let curr = vec3<f32>(
        uc2_tonemap_partial(color.x * exposure),
        uc2_tonemap_partial(color.y * exposure),
        uc2_tonemap_partial(color.z * exposure),
    );
    let white_scale = 1.0 / uc2_tonemap_partial(W);
    return clamp(curr * white_scale, vec3<f32>(0.0), vec3<f32>(1.0));
}
"""


def generate_tone_map_entry_wgsl(operator: str = "aces") -> str:
    """Generate unified tone_map entry point WGSL."""
    if operator == "aces":
        return """\
/// Main tone mapping entry point.
/// Applies ACES filmic curve followed by gamma correction.
fn tone_map(color: vec3<f32>) -> vec3<f32> {
    let mapped = tone_map_aces(color);
    return gamma_correct(mapped, 2.2);
}
"""
    elif operator == "reinhard":
        return """\
/// Main tone mapping entry point.
/// Applies Reinhard curve followed by gamma correction.
fn tone_map(color: vec3<f32>) -> vec3<f32> {
    let mapped = tone_map_reinhard(color);
    return gamma_correct(mapped, 2.2);
}
"""
    elif operator == "uncharted2":
        return """\
/// Main tone mapping entry point.
/// Applies Uncharted 2 curve followed by gamma correction.
fn tone_map(color: vec3<f32>) -> vec3<f32> {
    let mapped = tone_map_uncharted2(color, 2.0);
    return gamma_correct(mapped, 2.2);
}
"""
    else:
        return """\
/// Main tone mapping entry point.
/// Applies ACES filmic curve followed by gamma correction.
fn tone_map(color: vec3<f32>) -> vec3<f32> {
    let mapped = tone_map_aces(color);
    return gamma_correct(mapped, 2.2);
}
"""


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_color_range(color: Vec3) -> list[str]:
    """
    Validate that a tone-mapped color is in displayable range.

    Args:
        color: Color to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    for name, value in [("r", color.x), ("g", color.y), ("b", color.z)]:
        if value < 0.0:
            errors.append(f"Color component {name} is negative: {value}")
        if value > 1.0:
            errors.append(f"Color component {name} exceeds 1.0: {value}")
        if math.isnan(value):
            errors.append(f"Color component {name} is NaN")
        if math.isinf(value):
            errors.append(f"Color component {name} is infinite")

    return errors


def is_valid_hdr_color(color: Vec3) -> bool:
    """
    Check if a color is a valid HDR color (non-negative, finite).

    Args:
        color: Color to check.

    Returns:
        True if valid HDR color.
    """
    for value in [color.x, color.y, color.z]:
        if value < 0.0 or math.isnan(value) or math.isinf(value):
            return False
    return True


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Operators enum
    "ToneMappingOperator",
    # Tone mapping functions
    "reinhard",
    "reinhard_extended",
    "aces_filmic",
    "uncharted2",
    "linear_clamp",
    # Gamma correction
    "gamma_correct",
    "linear_to_srgb",
    "srgb_to_linear",
    # ToneMapper class
    "ToneMapper",
    # WGSL generation
    "generate_tone_mapping_wgsl",
    # Validation
    "validate_color_range",
    "is_valid_hdr_color",
]

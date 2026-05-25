"""
Color utilities for the UI styling system.

Provides Color class with RGBA, HSL, HSV representations,
named colors, color blending, interpolation, and palette generation.
"""

from __future__ import annotations

import colorsys
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Optional, Tuple, Union


class BlendMode(Enum):
    """Color blending modes."""
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    DARKEN = "darken"
    LIGHTEN = "lighten"
    COLOR_DODGE = "color_dodge"
    COLOR_BURN = "color_burn"
    HARD_LIGHT = "hard_light"
    SOFT_LIGHT = "soft_light"
    DIFFERENCE = "difference"
    EXCLUSION = "exclusion"


@dataclass(frozen=True, slots=True)
class Color:
    """
    Immutable color representation with RGBA values.

    Supports multiple color spaces (RGBA, HSL, HSV), color manipulation,
    blending, and interpolation.
    """

    r: float  # Red component (0.0 - 1.0)
    g: float  # Green component (0.0 - 1.0)
    b: float  # Blue component (0.0 - 1.0)
    a: float = 1.0  # Alpha component (0.0 - 1.0)

    # Named colors lookup table
    NAMED_COLORS: ClassVar[dict[str, "Color"]] = {}

    def __post_init__(self) -> None:
        """Validate color component values."""
        for name, value in [("r", self.r), ("g", self.g), ("b", self.b), ("a", self.a)]:
            if not isinstance(value, (int, float)):
                raise TypeError(f"Color component '{name}' must be numeric, got {type(value).__name__}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Color component '{name}' must be in range [0.0, 1.0], got {value}")

    # ========== Factory Methods ==========

    @classmethod
    def from_rgb(cls, r: int, g: int, b: int, a: int = 255) -> "Color":
        """
        Create color from 8-bit RGB(A) values (0-255).

        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            a: Alpha component (0-255)

        Returns:
            Color instance
        """
        return cls(r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    @classmethod
    def from_hex(cls, hex_string: str) -> "Color":
        """
        Create color from hex string.

        Supports formats: #RGB, #RGBA, #RRGGBB, #RRGGBBAA

        Args:
            hex_string: Hex color string (with or without #)

        Returns:
            Color instance

        Raises:
            ValueError: If hex string is invalid
        """
        # Remove # prefix if present
        hex_str = hex_string.lstrip("#")

        # Handle shorthand hex (#RGB or #RGBA)
        if len(hex_str) == 3:
            hex_str = "".join(c * 2 for c in hex_str) + "FF"
        elif len(hex_str) == 4:
            hex_str = "".join(c * 2 for c in hex_str)
        elif len(hex_str) == 6:
            hex_str += "FF"
        elif len(hex_str) != 8:
            raise ValueError(f"Invalid hex color format: {hex_string}")

        try:
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            a = int(hex_str[6:8], 16) / 255.0
            return cls(r, g, b, a)
        except ValueError as e:
            raise ValueError(f"Invalid hex color format: {hex_string}") from e

    @classmethod
    def from_hsl(cls, h: float, s: float, l: float, a: float = 1.0) -> "Color":
        """
        Create color from HSL values.

        Args:
            h: Hue (0.0 - 1.0, where 1.0 = 360 degrees)
            s: Saturation (0.0 - 1.0)
            l: Lightness (0.0 - 1.0)
            a: Alpha (0.0 - 1.0)

        Returns:
            Color instance
        """
        # Normalize hue to [0, 1)
        h = h % 1.0
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return cls(r, g, b, a)

    @classmethod
    def from_hsv(cls, h: float, s: float, v: float, a: float = 1.0) -> "Color":
        """
        Create color from HSV values.

        Args:
            h: Hue (0.0 - 1.0, where 1.0 = 360 degrees)
            s: Saturation (0.0 - 1.0)
            v: Value/Brightness (0.0 - 1.0)
            a: Alpha (0.0 - 1.0)

        Returns:
            Color instance
        """
        # Normalize hue to [0, 1)
        h = h % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return cls(r, g, b, a)

    @classmethod
    def from_name(cls, name: str) -> "Color":
        """
        Create color from named color.

        Args:
            name: Color name (case-insensitive)

        Returns:
            Color instance

        Raises:
            ValueError: If color name is not recognized
        """
        name_lower = name.lower().replace(" ", "").replace("_", "")
        if name_lower not in cls.NAMED_COLORS:
            raise ValueError(f"Unknown color name: {name}")
        return cls.NAMED_COLORS[name_lower]

    @classmethod
    def parse(cls, value: Union[str, "Color", Tuple[float, ...]]) -> "Color":
        """
        Parse color from various formats.

        Args:
            value: Color as hex string, name, Color instance, or RGB(A) tuple

        Returns:
            Color instance
        """
        if isinstance(value, Color):
            return value
        if isinstance(value, tuple):
            if len(value) == 3:
                return cls(*value)
            elif len(value) == 4:
                return cls(*value)
            raise ValueError(f"Invalid color tuple length: {len(value)}")
        if isinstance(value, str):
            # Try hex first
            if value.startswith("#") or re.match(r"^[0-9a-fA-F]+$", value):
                try:
                    return cls.from_hex(value)
                except ValueError:
                    pass
            # Try named color
            try:
                return cls.from_name(value)
            except ValueError:
                pass
            raise ValueError(f"Could not parse color: {value}")
        raise TypeError(f"Cannot parse color from {type(value).__name__}")

    # ========== Conversion Methods ==========

    def to_rgb(self) -> Tuple[int, int, int]:
        """Convert to 8-bit RGB tuple (0-255)."""
        return (
            round(self.r * 255),
            round(self.g * 255),
            round(self.b * 255),
        )

    def to_rgba(self) -> Tuple[int, int, int, int]:
        """Convert to 8-bit RGBA tuple (0-255)."""
        return (
            round(self.r * 255),
            round(self.g * 255),
            round(self.b * 255),
            round(self.a * 255),
        )

    def to_hex(self, include_alpha: bool = False) -> str:
        """
        Convert to hex string.

        Args:
            include_alpha: Include alpha component in output

        Returns:
            Hex color string (e.g., "#FF0000" or "#FF0000FF")
        """
        r, g, b, a = self.to_rgba()
        if include_alpha:
            return f"#{r:02X}{g:02X}{b:02X}{a:02X}"
        return f"#{r:02X}{g:02X}{b:02X}"

    def to_hsl(self) -> Tuple[float, float, float]:
        """
        Convert to HSL values.

        Returns:
            Tuple of (hue, saturation, lightness) in range [0, 1]
        """
        h, l, s = colorsys.rgb_to_hls(self.r, self.g, self.b)
        return (h, s, l)

    def to_hsv(self) -> Tuple[float, float, float]:
        """
        Convert to HSV values.

        Returns:
            Tuple of (hue, saturation, value) in range [0, 1]
        """
        return colorsys.rgb_to_hsv(self.r, self.g, self.b)

    # ========== Color Manipulation ==========

    def with_alpha(self, alpha: float) -> "Color":
        """Return new color with specified alpha."""
        return Color(self.r, self.g, self.b, alpha)

    def with_red(self, red: float) -> "Color":
        """Return new color with specified red component."""
        return Color(red, self.g, self.b, self.a)

    def with_green(self, green: float) -> "Color":
        """Return new color with specified green component."""
        return Color(self.r, green, self.b, self.a)

    def with_blue(self, blue: float) -> "Color":
        """Return new color with specified blue component."""
        return Color(self.r, self.g, blue, self.a)

    def lighten(self, amount: float) -> "Color":
        """
        Return lightened color.

        Args:
            amount: Amount to lighten (0.0 - 1.0)

        Returns:
            Lightened color
        """
        h, s, l = self.to_hsl()
        new_l = min(1.0, l + amount * (1.0 - l))
        return Color.from_hsl(h, s, new_l, self.a)

    def darken(self, amount: float) -> "Color":
        """
        Return darkened color.

        Args:
            amount: Amount to darken (0.0 - 1.0)

        Returns:
            Darkened color
        """
        h, s, l = self.to_hsl()
        new_l = max(0.0, l - amount * l)
        return Color.from_hsl(h, s, new_l, self.a)

    def saturate(self, amount: float) -> "Color":
        """
        Return more saturated color.

        Args:
            amount: Amount to increase saturation (0.0 - 1.0)

        Returns:
            More saturated color
        """
        h, s, l = self.to_hsl()
        new_s = min(1.0, s + amount * (1.0 - s))
        return Color.from_hsl(h, new_s, l, self.a)

    def desaturate(self, amount: float) -> "Color":
        """
        Return less saturated color.

        Args:
            amount: Amount to decrease saturation (0.0 - 1.0)

        Returns:
            Less saturated color
        """
        h, s, l = self.to_hsl()
        new_s = max(0.0, s - amount * s)
        return Color.from_hsl(h, new_s, l, self.a)

    def grayscale(self) -> "Color":
        """Return grayscale version using luminance."""
        lum = self.luminance
        return Color(lum, lum, lum, self.a)

    def invert(self) -> "Color":
        """Return inverted (complementary) color."""
        return Color(1.0 - self.r, 1.0 - self.g, 1.0 - self.b, self.a)

    def rotate_hue(self, degrees: float) -> "Color":
        """
        Return color with rotated hue.

        Args:
            degrees: Degrees to rotate hue

        Returns:
            Color with rotated hue
        """
        h, s, l = self.to_hsl()
        new_h = (h + degrees / 360.0) % 1.0
        return Color.from_hsl(new_h, s, l, self.a)

    def complement(self) -> "Color":
        """Return complementary color (180 degree hue rotation)."""
        return self.rotate_hue(180)

    @property
    def luminance(self) -> float:
        """
        Calculate relative luminance (perceived brightness).

        Uses the formula for relative luminance from WCAG 2.0.
        """
        def linearize(c: float) -> float:
            if c <= 0.03928:
                return c / 12.92
            return ((c + 0.055) / 1.055) ** 2.4

        return (
            0.2126 * linearize(self.r) +
            0.7152 * linearize(self.g) +
            0.0722 * linearize(self.b)
        )

    # ========== Blending ==========

    def blend(self, other: "Color", mode: BlendMode = BlendMode.NORMAL) -> "Color":
        """
        Blend with another color using specified blend mode.

        Args:
            other: Color to blend with
            mode: Blend mode to use

        Returns:
            Blended color
        """
        if mode == BlendMode.NORMAL:
            return self._blend_normal(other)
        elif mode == BlendMode.MULTIPLY:
            return self._blend_multiply(other)
        elif mode == BlendMode.SCREEN:
            return self._blend_screen(other)
        elif mode == BlendMode.OVERLAY:
            return self._blend_overlay(other)
        elif mode == BlendMode.DARKEN:
            return self._blend_darken(other)
        elif mode == BlendMode.LIGHTEN:
            return self._blend_lighten(other)
        elif mode == BlendMode.COLOR_DODGE:
            return self._blend_color_dodge(other)
        elif mode == BlendMode.COLOR_BURN:
            return self._blend_color_burn(other)
        elif mode == BlendMode.HARD_LIGHT:
            return self._blend_hard_light(other)
        elif mode == BlendMode.SOFT_LIGHT:
            return self._blend_soft_light(other)
        elif mode == BlendMode.DIFFERENCE:
            return self._blend_difference(other)
        elif mode == BlendMode.EXCLUSION:
            return self._blend_exclusion(other)
        else:
            return self._blend_normal(other)

    def _blend_normal(self, other: "Color") -> "Color":
        """Normal blend (alpha compositing)."""
        out_a = other.a + self.a * (1 - other.a)
        if out_a == 0:
            return Color(0, 0, 0, 0)
        out_r = (other.r * other.a + self.r * self.a * (1 - other.a)) / out_a
        out_g = (other.g * other.a + self.g * self.a * (1 - other.a)) / out_a
        out_b = (other.b * other.a + self.b * self.a * (1 - other.a)) / out_a
        return Color(out_r, out_g, out_b, out_a)

    def _blend_multiply(self, other: "Color") -> "Color":
        """Multiply blend mode."""
        return Color(
            self.r * other.r,
            self.g * other.g,
            self.b * other.b,
            self.a * other.a,
        )

    def _blend_screen(self, other: "Color") -> "Color":
        """Screen blend mode."""
        return Color(
            1 - (1 - self.r) * (1 - other.r),
            1 - (1 - self.g) * (1 - other.g),
            1 - (1 - self.b) * (1 - other.b),
            1 - (1 - self.a) * (1 - other.a),
        )

    def _blend_overlay(self, other: "Color") -> "Color":
        """Overlay blend mode."""
        def overlay_channel(a: float, b: float) -> float:
            if a < 0.5:
                return 2 * a * b
            return 1 - 2 * (1 - a) * (1 - b)
        return Color(
            overlay_channel(self.r, other.r),
            overlay_channel(self.g, other.g),
            overlay_channel(self.b, other.b),
            overlay_channel(self.a, other.a),
        )

    def _blend_darken(self, other: "Color") -> "Color":
        """Darken blend mode."""
        return Color(
            min(self.r, other.r),
            min(self.g, other.g),
            min(self.b, other.b),
            max(self.a, other.a),
        )

    def _blend_lighten(self, other: "Color") -> "Color":
        """Lighten blend mode."""
        return Color(
            max(self.r, other.r),
            max(self.g, other.g),
            max(self.b, other.b),
            max(self.a, other.a),
        )

    def _blend_color_dodge(self, other: "Color") -> "Color":
        """Color dodge blend mode."""
        def dodge_channel(a: float, b: float) -> float:
            if b >= 1.0:
                return 1.0
            return min(1.0, a / (1 - b))
        return Color(
            dodge_channel(self.r, other.r),
            dodge_channel(self.g, other.g),
            dodge_channel(self.b, other.b),
            self.a,
        )

    def _blend_color_burn(self, other: "Color") -> "Color":
        """Color burn blend mode."""
        def burn_channel(a: float, b: float) -> float:
            if b <= 0:
                return 0.0
            return max(0.0, 1 - (1 - a) / b)
        return Color(
            burn_channel(self.r, other.r),
            burn_channel(self.g, other.g),
            burn_channel(self.b, other.b),
            self.a,
        )

    def _blend_hard_light(self, other: "Color") -> "Color":
        """Hard light blend mode."""
        def hard_light_channel(a: float, b: float) -> float:
            if b < 0.5:
                return 2 * a * b
            return 1 - 2 * (1 - a) * (1 - b)
        return Color(
            hard_light_channel(self.r, other.r),
            hard_light_channel(self.g, other.g),
            hard_light_channel(self.b, other.b),
            self.a,
        )

    def _blend_soft_light(self, other: "Color") -> "Color":
        """Soft light blend mode."""
        def soft_light_channel(a: float, b: float) -> float:
            if b < 0.5:
                return a - (1 - 2 * b) * a * (1 - a)
            return a + (2 * b - 1) * (math.sqrt(a) - a)
        return Color(
            soft_light_channel(self.r, other.r),
            soft_light_channel(self.g, other.g),
            soft_light_channel(self.b, other.b),
            self.a,
        )

    def _blend_difference(self, other: "Color") -> "Color":
        """Difference blend mode."""
        return Color(
            abs(self.r - other.r),
            abs(self.g - other.g),
            abs(self.b - other.b),
            max(self.a, other.a),
        )

    def _blend_exclusion(self, other: "Color") -> "Color":
        """Exclusion blend mode."""
        return Color(
            self.r + other.r - 2 * self.r * other.r,
            self.g + other.g - 2 * self.g * other.g,
            self.b + other.b - 2 * self.b * other.b,
            max(self.a, other.a),
        )

    # ========== Interpolation ==========

    def lerp(self, other: "Color", t: float) -> "Color":
        """
        Linear interpolation between colors.

        Args:
            other: Target color
            t: Interpolation factor (0.0 = self, 1.0 = other)

        Returns:
            Interpolated color
        """
        t = max(0.0, min(1.0, t))
        return Color(
            self.r + (other.r - self.r) * t,
            self.g + (other.g - self.g) * t,
            self.b + (other.b - self.b) * t,
            self.a + (other.a - self.a) * t,
        )

    def lerp_hsl(self, other: "Color", t: float) -> "Color":
        """
        Interpolation in HSL color space.

        Args:
            other: Target color
            t: Interpolation factor (0.0 = self, 1.0 = other)

        Returns:
            Interpolated color
        """
        t = max(0.0, min(1.0, t))
        h1, s1, l1 = self.to_hsl()
        h2, s2, l2 = other.to_hsl()

        # Take shortest path for hue
        if abs(h2 - h1) > 0.5:
            if h1 < h2:
                h1 += 1.0
            else:
                h2 += 1.0

        h = (h1 + (h2 - h1) * t) % 1.0
        s = s1 + (s2 - s1) * t
        l = l1 + (l2 - l1) * t
        a = self.a + (other.a - self.a) * t

        return Color.from_hsl(h, s, l, a)

    # ========== Contrast ==========

    def contrast_ratio(self, other: "Color") -> float:
        """
        Calculate WCAG contrast ratio between two colors.

        Args:
            other: Color to compare with

        Returns:
            Contrast ratio (1:1 to 21:1)
        """
        l1 = self.luminance
        l2 = other.luminance
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)

    def is_readable_on(self, background: "Color", level: str = "AA") -> bool:
        """
        Check if text of this color is readable on background.

        Args:
            background: Background color
            level: WCAG level ("AA" = 4.5:1, "AAA" = 7:1)

        Returns:
            True if contrast meets the specified level
        """
        ratio = self.contrast_ratio(background)
        threshold = 7.0 if level == "AAA" else 4.5
        return ratio >= threshold

    # ========== String Representation ==========

    def __str__(self) -> str:
        return self.to_hex(include_alpha=self.a < 1.0)

    def __repr__(self) -> str:
        return f"Color(r={self.r:.3f}, g={self.g:.3f}, b={self.b:.3f}, a={self.a:.3f})"


# ========== Color Palette Generation ==========

def generate_palette(base_color: Color, count: int = 5, spread: float = 0.1) -> list[Color]:
    """
    Generate a monochromatic palette from base color.

    Args:
        base_color: Starting color
        count: Number of colors to generate
        spread: Lightness variation range

    Returns:
        List of colors from dark to light
    """
    h, s, l = base_color.to_hsl()
    colors = []

    for i in range(count):
        t = i / (count - 1) if count > 1 else 0.5
        new_l = max(0.0, min(1.0, l - spread / 2 + spread * t))
        colors.append(Color.from_hsl(h, s, new_l, base_color.a))

    return colors


def generate_complementary(base_color: Color) -> list[Color]:
    """
    Generate complementary color scheme.

    Args:
        base_color: Starting color

    Returns:
        List of [base, complement]
    """
    return [base_color, base_color.complement()]


def generate_triadic(base_color: Color) -> list[Color]:
    """
    Generate triadic color scheme (120 degrees apart).

    Args:
        base_color: Starting color

    Returns:
        List of three colors
    """
    return [
        base_color,
        base_color.rotate_hue(120),
        base_color.rotate_hue(240),
    ]


def generate_analogous(base_color: Color, angle: float = 30.0) -> list[Color]:
    """
    Generate analogous color scheme.

    Args:
        base_color: Starting color
        angle: Angle between colors in degrees

    Returns:
        List of three colors
    """
    return [
        base_color.rotate_hue(-angle),
        base_color,
        base_color.rotate_hue(angle),
    ]


def generate_split_complementary(base_color: Color, angle: float = 30.0) -> list[Color]:
    """
    Generate split-complementary color scheme.

    Args:
        base_color: Starting color
        angle: Split angle from complement

    Returns:
        List of three colors
    """
    return [
        base_color,
        base_color.rotate_hue(180 - angle),
        base_color.rotate_hue(180 + angle),
    ]


def generate_tetradic(base_color: Color) -> list[Color]:
    """
    Generate tetradic (rectangular) color scheme.

    Args:
        base_color: Starting color

    Returns:
        List of four colors
    """
    return [
        base_color,
        base_color.rotate_hue(60),
        base_color.rotate_hue(180),
        base_color.rotate_hue(240),
    ]


def interpolate_colors(colors: list[Color], steps: int) -> list[Color]:
    """
    Interpolate between multiple colors.

    Args:
        colors: List of colors to interpolate between
        steps: Total number of output colors

    Returns:
        List of interpolated colors
    """
    if len(colors) < 2:
        return colors * steps if colors else []

    result = []
    segments = len(colors) - 1
    steps_per_segment = (steps - 1) / segments

    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0
        segment_t = t * segments
        segment_idx = min(int(segment_t), segments - 1)
        local_t = segment_t - segment_idx

        result.append(colors[segment_idx].lerp(colors[segment_idx + 1], local_t))

    return result


# ========== Initialize Named Colors ==========

def _init_named_colors() -> None:
    """Initialize the named colors lookup table."""
    named = {
        # Basic colors
        "transparent": Color(0, 0, 0, 0),
        "black": Color(0, 0, 0),
        "white": Color(1, 1, 1),
        "red": Color(1, 0, 0),
        "green": Color(0, 0.5, 0),  # CSS green
        "lime": Color(0, 1, 0),
        "blue": Color(0, 0, 1),
        "yellow": Color(1, 1, 0),
        "cyan": Color(0, 1, 1),
        "aqua": Color(0, 1, 1),
        "magenta": Color(1, 0, 1),
        "fuchsia": Color(1, 0, 1),

        # Grays
        "gray": Color(0.5, 0.5, 0.5),
        "grey": Color(0.5, 0.5, 0.5),
        "silver": Color(0.75, 0.75, 0.75),
        "darkgray": Color(0.66, 0.66, 0.66),
        "darkgrey": Color(0.66, 0.66, 0.66),
        "lightgray": Color(0.83, 0.83, 0.83),
        "lightgrey": Color(0.83, 0.83, 0.83),

        # Reds
        "maroon": Color(0.5, 0, 0),
        "darkred": Color(0.55, 0, 0),
        "crimson": Color(0.86, 0.08, 0.24),
        "indianred": Color(0.8, 0.36, 0.36),
        "lightcoral": Color(0.94, 0.5, 0.5),
        "salmon": Color(0.98, 0.5, 0.45),
        "coral": Color(1, 0.5, 0.31),
        "tomato": Color(1, 0.39, 0.28),
        "firebrick": Color(0.7, 0.13, 0.13),

        # Oranges
        "orange": Color(1, 0.65, 0),
        "darkorange": Color(1, 0.55, 0),
        "orangered": Color(1, 0.27, 0),

        # Yellows
        "gold": Color(1, 0.84, 0),
        "khaki": Color(0.94, 0.9, 0.55),
        "lemonchiffon": Color(1, 0.98, 0.8),

        # Greens
        "olive": Color(0.5, 0.5, 0),
        "darkgreen": Color(0, 0.39, 0),
        "forestgreen": Color(0.13, 0.55, 0.13),
        "seagreen": Color(0.18, 0.55, 0.34),
        "limegreen": Color(0.2, 0.8, 0.2),
        "springgreen": Color(0, 1, 0.5),
        "teal": Color(0, 0.5, 0.5),

        # Blues
        "navy": Color(0, 0, 0.5),
        "darkblue": Color(0, 0, 0.55),
        "mediumblue": Color(0, 0, 0.8),
        "royalblue": Color(0.25, 0.41, 0.88),
        "steelblue": Color(0.27, 0.51, 0.71),
        "dodgerblue": Color(0.12, 0.56, 1),
        "deepskyblue": Color(0, 0.75, 1),
        "lightblue": Color(0.68, 0.85, 0.9),
        "skyblue": Color(0.53, 0.81, 0.92),
        "cadetblue": Color(0.37, 0.62, 0.63),

        # Purples
        "purple": Color(0.5, 0, 0.5),
        "indigo": Color(0.29, 0, 0.51),
        "darkviolet": Color(0.58, 0, 0.83),
        "blueviolet": Color(0.54, 0.17, 0.89),
        "mediumpurple": Color(0.58, 0.44, 0.86),
        "orchid": Color(0.85, 0.44, 0.84),
        "violet": Color(0.93, 0.51, 0.93),
        "plum": Color(0.87, 0.63, 0.87),
        "lavender": Color(0.9, 0.9, 0.98),

        # Pinks
        "pink": Color(1, 0.75, 0.8),
        "hotpink": Color(1, 0.41, 0.71),
        "deeppink": Color(1, 0.08, 0.58),

        # Browns
        "brown": Color(0.65, 0.16, 0.16),
        "saddlebrown": Color(0.55, 0.27, 0.07),
        "sienna": Color(0.63, 0.32, 0.18),
        "chocolate": Color(0.82, 0.41, 0.12),
        "peru": Color(0.8, 0.52, 0.25),
        "sandybrown": Color(0.96, 0.64, 0.38),
        "burlywood": Color(0.87, 0.72, 0.53),
        "tan": Color(0.82, 0.71, 0.55),
        "wheat": Color(0.96, 0.87, 0.7),

        # Whites
        "snow": Color(1, 0.98, 0.98),
        "ivory": Color(1, 1, 0.94),
        "floralwhite": Color(1, 0.98, 0.94),
        "honeydew": Color(0.94, 1, 0.94),
        "mintcream": Color(0.96, 1, 0.98),
        "azure": Color(0.94, 1, 1),
        "aliceblue": Color(0.94, 0.97, 1),
        "ghostwhite": Color(0.97, 0.97, 1),
        "whitesmoke": Color(0.96, 0.96, 0.96),
        "seashell": Color(1, 0.96, 0.93),
        "beige": Color(0.96, 0.96, 0.86),
        "oldlace": Color(0.99, 0.96, 0.9),
        "linen": Color(0.98, 0.94, 0.9),
        "antiquewhite": Color(0.98, 0.92, 0.84),
        "papayawhip": Color(1, 0.94, 0.84),
        "blanchedalmond": Color(1, 0.92, 0.8),
        "bisque": Color(1, 0.89, 0.77),
        "moccasin": Color(1, 0.89, 0.71),
        "navajowhite": Color(1, 0.87, 0.68),
        "peachpuff": Color(1, 0.85, 0.73),
        "mistyrose": Color(1, 0.89, 0.88),
        "lavenderblush": Color(1, 0.94, 0.96),
    }
    Color.NAMED_COLORS.update(named)


_init_named_colors()

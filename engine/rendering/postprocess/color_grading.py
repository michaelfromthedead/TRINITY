"""
Color Grading System

Provides comprehensive color adjustment tools:
- WhiteBalance: Temperature and tint adjustments
- ContrastSettings: Shadows, midtones, highlights
- SaturationSettings: Global and per-channel saturation
- LUT3D: 3D color lookup table support
- ColorGradingStack: Combined color adjustments
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from .postprocess_stack import EffectPriority, EffectSettings, PostProcessEffect


class ColorSpace(Enum):
    """Color space for color operations."""

    LINEAR_SRGB = auto()
    SRGB = auto()
    ACES_CC = auto()
    LOG_C = auto()


class LUTFormat(Enum):
    """LUT file format."""

    CUBE = auto()  # .cube format
    THREE_DL = auto()  # .3dl format
    CSP = auto()  # .csp format
    TEXTURE = auto()  # GPU texture


@dataclass
class WhiteBalanceSettings:
    """White balance adjustments."""

    temperature: float = 0.0  # Kelvin offset (-100 to 100, maps to color)
    tint: float = 0.0  # Green-magenta shift (-100 to 100)

    def get_color_temperature_rgb(self) -> Tuple[float, float, float]:
        """Convert temperature to RGB multipliers.

        Returns:
            RGB color temperature multipliers.
        """
        t = self.temperature / 100.0

        if t >= 0:
            r = 1.0
            g = 1.0 - t * 0.1
            b = 1.0 - t * 0.3
        else:
            r = 1.0 + t * 0.3
            g = 1.0 + t * 0.1
            b = 1.0

        tint_offset = self.tint / 100.0
        g += tint_offset * 0.1
        r -= tint_offset * 0.05
        b -= tint_offset * 0.05

        return (
            max(0.1, min(2.0, r)),
            max(0.1, min(2.0, g)),
            max(0.1, min(2.0, b)),
        )


@dataclass
class LiftGammaGain:
    """Lift/Gamma/Gain color correction values."""

    lift: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Shadow adjustment
    gamma: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Midtone adjustment
    gain: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # Highlight adjustment


@dataclass
class ContrastSettings:
    """Shadow/midtone/highlight contrast controls."""

    contrast: float = 1.0  # Global contrast [0.5, 2.0]

    # Shadows
    shadow_intensity: float = 1.0
    shadow_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    shadow_offset: float = 0.0

    # Midtones
    midtone_intensity: float = 1.0
    midtone_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Highlights
    highlight_intensity: float = 1.0
    highlight_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    highlight_offset: float = 0.0

    def apply(
        self,
        r: float,
        g: float,
        b: float,
    ) -> Tuple[float, float, float]:
        """Apply contrast adjustments.

        Args:
            r: Red channel [0, 1].
            g: Green channel [0, 1].
            b: Blue channel [0, 1].

        Returns:
            Adjusted RGB values.
        """
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

        mid_point = 0.5
        r = mid_point + (r - mid_point) * self.contrast
        g = mid_point + (g - mid_point) * self.contrast
        b = mid_point + (b - mid_point) * self.contrast

        shadow_weight = 1.0 - min(1.0, lum * 2.0)
        highlight_weight = max(0.0, lum * 2.0 - 1.0)
        midtone_weight = 1.0 - shadow_weight - highlight_weight

        shadow_r = r * self.shadow_color[0] * self.shadow_intensity + self.shadow_offset
        shadow_g = g * self.shadow_color[1] * self.shadow_intensity + self.shadow_offset
        shadow_b = b * self.shadow_color[2] * self.shadow_intensity + self.shadow_offset

        mid_r = r * self.midtone_color[0] * self.midtone_intensity
        mid_g = g * self.midtone_color[1] * self.midtone_intensity
        mid_b = b * self.midtone_color[2] * self.midtone_intensity

        high_r = r * self.highlight_color[0] * self.highlight_intensity + self.highlight_offset
        high_g = g * self.highlight_color[1] * self.highlight_intensity + self.highlight_offset
        high_b = b * self.highlight_color[2] * self.highlight_intensity + self.highlight_offset

        r = shadow_r * shadow_weight + mid_r * midtone_weight + high_r * highlight_weight
        g = shadow_g * shadow_weight + mid_g * midtone_weight + high_g * highlight_weight
        b = shadow_b * shadow_weight + mid_b * midtone_weight + high_b * highlight_weight

        return (
            max(0.0, min(1.0, r)),
            max(0.0, min(1.0, g)),
            max(0.0, min(1.0, b)),
        )


@dataclass
class SaturationSettings:
    """Saturation adjustment settings."""

    global_saturation: float = 1.0  # Overall saturation [0, 2]
    vibrance: float = 0.0  # Intelligent saturation [-1, 1]

    # Per-channel saturation
    red_saturation: float = 1.0
    green_saturation: float = 1.0
    blue_saturation: float = 1.0

    # Hue-based saturation (6 color ranges)
    red_hue_saturation: float = 1.0
    yellow_hue_saturation: float = 1.0
    green_hue_saturation: float = 1.0
    cyan_hue_saturation: float = 1.0
    blue_hue_saturation: float = 1.0
    magenta_hue_saturation: float = 1.0

    def apply(
        self,
        r: float,
        g: float,
        b: float,
    ) -> Tuple[float, float, float]:
        """Apply saturation adjustments.

        Args:
            r: Red channel.
            g: Green channel.
            b: Blue channel.

        Returns:
            Adjusted RGB values.
        """
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

        sat = self.global_saturation
        r = lum + (r - lum) * sat * self.red_saturation
        g = lum + (g - lum) * sat * self.green_saturation
        b = lum + (b - lum) * sat * self.blue_saturation

        if self.vibrance != 0:
            from .constants import LUMINANCE_MIN

            current_sat = max(abs(r - lum), abs(g - lum), abs(b - lum))
            # Guard against division by zero for very dark pixels
            if lum > LUMINANCE_MIN:
                current_sat /= lum
            else:
                current_sat = 0.0

            vibrance_mult = 1.0 + self.vibrance * (1.0 - current_sat)
            r = lum + (r - lum) * vibrance_mult
            g = lum + (g - lum) * vibrance_mult
            b = lum + (b - lum) * vibrance_mult

        return (
            max(0.0, min(1.0, r)),
            max(0.0, min(1.0, g)),
            max(0.0, min(1.0, b)),
        )


@dataclass
class HueSatLightness:
    """HSL-based color adjustments."""

    hue_shift: float = 0.0  # Global hue rotation [0, 360]
    saturation: float = 1.0  # Saturation multiplier [0, 2]
    lightness: float = 0.0  # Lightness offset [-1, 1]


@dataclass
class LUT3DSettings:
    """3D LUT configuration."""

    enabled: bool = False
    lut_path: Optional[str] = None
    size: int = 32  # LUT dimension (typically 16, 32, or 64)
    intensity: float = 1.0  # Blend with original [0, 1]
    color_space: ColorSpace = ColorSpace.SRGB


class LUT3D:
    """3D Color Lookup Table.

    Provides efficient color transformation using a pre-computed
    3D table of color mappings.
    """

    def __init__(self, size: int = 32) -> None:
        """Initialize empty LUT.

        Args:
            size: LUT dimension (size x size x size).
        """
        self._size: int = size
        self._data: List[List[List[Tuple[float, float, float]]]] = []
        self._initialized: bool = False

    @property
    def size(self) -> int:
        """LUT dimension."""
        return self._size

    @property
    def initialized(self) -> bool:
        """Whether the LUT is loaded."""
        return self._initialized

    def create_identity(self) -> None:
        """Create an identity LUT (no transformation)."""
        self._data = []

        for b in range(self._size):
            b_plane = []
            for g in range(self._size):
                g_row = []
                for r in range(self._size):
                    r_val = r / (self._size - 1)
                    g_val = g / (self._size - 1)
                    b_val = b / (self._size - 1)
                    g_row.append((r_val, g_val, b_val))
                b_plane.append(g_row)
            self._data.append(b_plane)

        self._initialized = True

    def load_from_cube(self, path: str) -> bool:
        """Load LUT from .cube file.

        Args:
            path: Path to .cube file.

        Returns:
            True if loaded successfully.
        """
        try:
            with open(path, "r") as f:
                lines = f.readlines()

            size = 0
            data_lines = []

            for line in lines:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if line.startswith("LUT_3D_SIZE"):
                    size = int(line.split()[1])
                elif line.startswith("DOMAIN_MIN"):
                    pass
                elif line.startswith("DOMAIN_MAX"):
                    pass
                elif line.startswith("TITLE"):
                    pass
                else:
                    parts = line.split()
                    if len(parts) >= 3:
                        r = float(parts[0])
                        g = float(parts[1])
                        b = float(parts[2])
                        data_lines.append((r, g, b))

            if size == 0 or len(data_lines) != size * size * size:
                return False

            self._size = size
            self._data = []

            idx = 0
            for b in range(size):
                b_plane = []
                for g in range(size):
                    g_row = []
                    for r in range(size):
                        g_row.append(data_lines[idx])
                        idx += 1
                    b_plane.append(g_row)
                self._data.append(b_plane)

            self._initialized = True
            return True

        except Exception:
            return False

    def sample(
        self,
        r: float,
        g: float,
        b: float,
    ) -> Tuple[float, float, float]:
        """Sample the LUT with trilinear interpolation.

        Args:
            r: Red channel [0, 1].
            g: Green channel [0, 1].
            b: Blue channel [0, 1].

        Returns:
            Transformed RGB values.
        """
        if not self._initialized:
            return (r, g, b)

        r = max(0.0, min(1.0, r))
        g = max(0.0, min(1.0, g))
        b = max(0.0, min(1.0, b))

        r_scaled = r * (self._size - 1)
        g_scaled = g * (self._size - 1)
        b_scaled = b * (self._size - 1)

        r0 = int(r_scaled)
        g0 = int(g_scaled)
        b0 = int(b_scaled)

        r1 = min(r0 + 1, self._size - 1)
        g1 = min(g0 + 1, self._size - 1)
        b1 = min(b0 + 1, self._size - 1)

        r_frac = r_scaled - r0
        g_frac = g_scaled - g0
        b_frac = b_scaled - b0

        c000 = self._data[b0][g0][r0]
        c100 = self._data[b0][g0][r1]
        c010 = self._data[b0][g1][r0]
        c110 = self._data[b0][g1][r1]
        c001 = self._data[b1][g0][r0]
        c101 = self._data[b1][g0][r1]
        c011 = self._data[b1][g1][r0]
        c111 = self._data[b1][g1][r1]

        c00 = self._lerp_rgb(c000, c100, r_frac)
        c10 = self._lerp_rgb(c010, c110, r_frac)
        c01 = self._lerp_rgb(c001, c101, r_frac)
        c11 = self._lerp_rgb(c011, c111, r_frac)

        c0 = self._lerp_rgb(c00, c10, g_frac)
        c1 = self._lerp_rgb(c01, c11, g_frac)

        return self._lerp_rgb(c0, c1, b_frac)

    def _lerp_rgb(
        self,
        a: Tuple[float, float, float],
        b: Tuple[float, float, float],
        t: float,
    ) -> Tuple[float, float, float]:
        """Linear interpolation between RGB values."""
        return (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        )


@dataclass
class ColorGradingSettings(EffectSettings):
    """Complete color grading settings."""

    white_balance: WhiteBalanceSettings = field(default_factory=WhiteBalanceSettings)
    contrast: ContrastSettings = field(default_factory=ContrastSettings)
    saturation: SaturationSettings = field(default_factory=SaturationSettings)
    lift_gamma_gain: LiftGammaGain = field(default_factory=LiftGammaGain)
    hsl: HueSatLightness = field(default_factory=HueSatLightness)
    lut: LUT3DSettings = field(default_factory=LUT3DSettings)

    # Channel mixer
    channel_mixer_red: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    channel_mixer_green: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    channel_mixer_blue: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    # Split toning
    split_tone_shadows: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    split_tone_highlights: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    split_tone_balance: float = 0.0  # -1 = shadows, +1 = highlights

    def __post_init__(self) -> None:
        self.priority = EffectPriority.COLOR_GRADING.value

    def lerp(self, other: "ColorGradingSettings", t: float) -> "ColorGradingSettings":
        """Interpolate between two color grading settings."""
        return ColorGradingSettings(
            enabled=self.enabled if t < 0.5 else other.enabled,
            weight=self.weight + (other.weight - self.weight) * t,
            white_balance=WhiteBalanceSettings(
                temperature=self.white_balance.temperature
                + (other.white_balance.temperature - self.white_balance.temperature)
                * t,
                tint=self.white_balance.tint
                + (other.white_balance.tint - self.white_balance.tint) * t,
            ),
            contrast=ContrastSettings(
                contrast=self.contrast.contrast
                + (other.contrast.contrast - self.contrast.contrast) * t,
                shadow_intensity=self.contrast.shadow_intensity
                + (other.contrast.shadow_intensity - self.contrast.shadow_intensity)
                * t,
                highlight_intensity=self.contrast.highlight_intensity
                + (
                    other.contrast.highlight_intensity
                    - self.contrast.highlight_intensity
                )
                * t,
            ),
            saturation=SaturationSettings(
                global_saturation=self.saturation.global_saturation
                + (
                    other.saturation.global_saturation
                    - self.saturation.global_saturation
                )
                * t,
                vibrance=self.saturation.vibrance
                + (other.saturation.vibrance - self.saturation.vibrance) * t,
            ),
            split_tone_balance=self.split_tone_balance
            + (other.split_tone_balance - self.split_tone_balance) * t,
        )


class ColorGradingStack:
    """Manages the color grading pipeline.

    Combines multiple color adjustments in the correct order
    for optimal results.
    """

    def __init__(
        self,
        settings: Optional[ColorGradingSettings] = None,
    ) -> None:
        """Initialize color grading stack.

        Args:
            settings: Color grading configuration.
        """
        self._settings: ColorGradingSettings = settings or ColorGradingSettings()
        self._lut: Optional[LUT3D] = None

    @property
    def settings(self) -> ColorGradingSettings:
        """Current color grading settings."""
        return self._settings

    @settings.setter
    def settings(self, value: ColorGradingSettings) -> None:
        self._settings = value

    def load_lut(self, path: str) -> bool:
        """Load a LUT file.

        Args:
            path: Path to LUT file.

        Returns:
            True if loaded successfully.
        """
        self._lut = LUT3D(self._settings.lut.size)

        if path.lower().endswith(".cube"):
            return self._lut.load_from_cube(path)

        return False

    def apply(
        self,
        r: float,
        g: float,
        b: float,
    ) -> Tuple[float, float, float]:
        """Apply full color grading pipeline.

        Args:
            r: Red channel.
            g: Green channel.
            b: Blue channel.

        Returns:
            Graded RGB values.
        """
        if not self._settings.enabled:
            return (r, g, b)

        wb = self._settings.white_balance.get_color_temperature_rgb()
        r *= wb[0]
        g *= wb[1]
        b *= wb[2]

        lgg = self._settings.lift_gamma_gain
        r = self._apply_lift_gamma_gain(r, lgg.lift[0], lgg.gamma[0], lgg.gain[0])
        g = self._apply_lift_gamma_gain(g, lgg.lift[1], lgg.gamma[1], lgg.gain[1])
        b = self._apply_lift_gamma_gain(b, lgg.lift[2], lgg.gamma[2], lgg.gain[2])

        r, g, b = self._settings.contrast.apply(r, g, b)

        r, g, b = self._settings.saturation.apply(r, g, b)

        cm = self._settings
        out_r = (
            r * cm.channel_mixer_red[0]
            + g * cm.channel_mixer_red[1]
            + b * cm.channel_mixer_red[2]
        )
        out_g = (
            r * cm.channel_mixer_green[0]
            + g * cm.channel_mixer_green[1]
            + b * cm.channel_mixer_green[2]
        )
        out_b = (
            r * cm.channel_mixer_blue[0]
            + g * cm.channel_mixer_blue[1]
            + b * cm.channel_mixer_blue[2]
        )
        r, g, b = out_r, out_g, out_b

        if self._settings.lut.enabled and self._lut and self._lut.initialized:
            lut_r, lut_g, lut_b = self._lut.sample(r, g, b)
            intensity = self._settings.lut.intensity
            r = r + (lut_r - r) * intensity
            g = g + (lut_g - g) * intensity
            b = b + (lut_b - b) * intensity

        return (
            max(0.0, min(1.0, r)),
            max(0.0, min(1.0, g)),
            max(0.0, min(1.0, b)),
        )

    def _apply_lift_gamma_gain(
        self,
        value: float,
        lift: float,
        gamma: float,
        gain: float,
    ) -> float:
        """Apply lift/gamma/gain to a single channel.

        Args:
            value: Input value.
            lift: Shadow adjustment.
            gamma: Midtone adjustment.
            gain: Highlight adjustment.

        Returns:
            Adjusted value.
        """
        value = value * gain + lift
        if value > 0 and gamma != 1.0:
            value = pow(value, 1.0 / gamma)
        return value


class ColorGradingEffect(PostProcessEffect[ColorGradingSettings]):
    """Post-process effect for color grading."""

    def __init__(
        self,
        settings: Optional[ColorGradingSettings] = None,
    ) -> None:
        """Initialize color grading effect.

        Args:
            settings: Color grading configuration.
        """
        super().__init__(
            name="ColorGrading",
            settings=settings or ColorGradingSettings(),
            priority=EffectPriority.COLOR_GRADING.value,
        )

        self._grading_stack: ColorGradingStack = ColorGradingStack(self._settings)

    @property
    def grading_stack(self) -> ColorGradingStack:
        """Access the grading stack directly."""
        return self._grading_stack

    def get_required_inputs(self) -> List[str]:
        """Get required input resources."""
        return ["color"]

    def get_outputs(self) -> List[str]:
        """Get output resources."""
        return ["color"]

    def setup(self, width: int, height: int) -> None:
        """Initialize color grading resources."""
        if self._settings and self._settings.lut.enabled and self._settings.lut.lut_path:
            self._grading_stack.load_lut(self._settings.lut.lut_path)

    def execute(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        delta_time: float,
    ) -> None:
        """Execute color grading."""
        if not self._settings or not self._settings.enabled:
            return

        if self._settings != self._grading_stack.settings:
            self._grading_stack.settings = self._settings

    def cleanup(self) -> None:
        """Release color grading resources."""
        pass


__all__ = [
    "ColorSpace",
    "LUTFormat",
    "WhiteBalanceSettings",
    "LiftGammaGain",
    "ContrastSettings",
    "SaturationSettings",
    "HueSatLightness",
    "LUT3DSettings",
    "LUT3D",
    "ColorGradingSettings",
    "ColorGradingStack",
    "ColorGradingEffect",
]

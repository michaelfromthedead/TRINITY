"""
Cosmetic Post-Processing Effects: Vignette, Chromatic Aberration, Film Grain.

This module provides three visually-oriented post-processing effects:
- T-PP-1.6: Vignette - Radial darkening from edges
- T-PP-2.4: Chromatic Aberration - Per-channel radial offset
- T-PP-3.5: Film Grain - Procedural noise with luminance modulation
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple
import math


# ============================================================================
# T-PP-1.6: VIGNETTE
# ============================================================================


@dataclass
class VignetteSettings:
    """Configurable vignette parameters.

    Attributes:
        enabled: Whether vignette effect is active.
        intensity: Strength of darkening (0.0-1.0).
        inner_radius: Normalized radius where falloff starts (0-1).
        outer_radius: Normalized radius where falloff ends (0-1).
        feather: Falloff exponent controlling curve shape.
        color: RGB color of the vignette (default black).
        aspect_ratio_correction: Whether to correct for non-square displays.
    """

    enabled: bool = True
    intensity: float = 0.3
    inner_radius: float = 0.4
    outer_radius: float = 0.8
    feather: float = 2.0
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    aspect_ratio_correction: bool = True


class VignetteEffect:
    """Radial darkening post-process effect.

    Creates a vignette by darkening pixels based on their distance from
    the screen center. The falloff is controlled by inner/outer radius
    and feather parameters.
    """

    def __init__(self, settings: Optional[VignetteSettings] = None):
        """Initialize vignette effect.

        Args:
            settings: Vignette configuration, uses defaults if None.
        """
        self.settings = settings or VignetteSettings()

    def calculate_vignette(
        self, uv: Tuple[float, float], aspect: float = 1.0
    ) -> float:
        """Calculate vignette factor at UV coordinate.

        Args:
            uv: Normalized screen coordinates (0-1, 0-1).
            aspect: Width/height aspect ratio.

        Returns:
            Vignette factor 0.0 (full vignette) to 1.0 (no vignette).
        """
        if not self.settings.enabled:
            return 1.0

        # Center UV coordinates
        cx = uv[0] - 0.5
        cy = uv[1] - 0.5

        # Aspect ratio correction makes vignette circular on non-square displays
        if self.settings.aspect_ratio_correction:
            cx *= aspect

        # Radial distance from center, scaled so corners are ~1.0
        dist = math.sqrt(cx * cx + cy * cy) * 2.0

        # Smoothstep falloff between inner and outer radius
        inner = self.settings.inner_radius
        outer = self.settings.outer_radius
        radius_range = max(outer - inner, 0.001)
        t = max(0.0, min(1.0, (dist - inner) / radius_range))

        # Apply feather exponent for curve control
        vignette = pow(t, self.settings.feather)

        return 1.0 - vignette * self.settings.intensity

    def apply_to_color(
        self,
        color: Tuple[float, float, float],
        uv: Tuple[float, float],
        aspect: float = 1.0,
    ) -> Tuple[float, float, float]:
        """Apply vignette to an RGB color.

        Args:
            color: Input RGB color (0-1 range).
            uv: Normalized screen coordinates (0-1, 0-1).
            aspect: Width/height aspect ratio.

        Returns:
            Vignetted RGB color blended toward vignette color.
        """
        factor = self.calculate_vignette(uv, aspect)
        vc = self.settings.color

        return (
            color[0] * factor + vc[0] * (1.0 - factor),
            color[1] * factor + vc[1] * (1.0 - factor),
            color[2] * factor + vc[2] * (1.0 - factor),
        )


# ============================================================================
# T-PP-2.4: CHROMATIC ABERRATION
# ============================================================================


class CAQuality(Enum):
    """Chromatic aberration quality levels."""

    OFF = auto()
    LOW = auto()  # 2px max offset
    MEDIUM = auto()  # 5px max offset
    HIGH = auto()  # 10px max offset


@dataclass
class ChromaticAberrationSettings:
    """Chromatic aberration parameters.

    Attributes:
        enabled: Whether CA effect is active.
        quality: Quality level determining max pixel offset.
        intensity: Overall strength multiplier.
        red_offset: Red channel offset direction (positive = outward).
        blue_offset: Blue channel offset direction (negative = inward).
        anamorphic_ratio: Aspect distortion (1.0 = spherical, 1.33 = anamorphic).
        fringe_start: Distance from center where fringe starts.
        fringe_end: Distance from center where fringe reaches full strength.
    """

    enabled: bool = True
    quality: CAQuality = CAQuality.MEDIUM
    intensity: float = 1.0
    red_offset: float = 1.0
    blue_offset: float = -1.0
    anamorphic_ratio: float = 1.0
    fringe_start: float = 0.1
    fringe_end: float = 0.45


class ChromaticAberrationEffect:
    """Per-channel radial offset effect.

    Simulates lens chromatic aberration by offsetting red and blue
    channels radially from the image center, leaving green as reference.
    """

    MAX_OFFSET = {
        CAQuality.OFF: 0,
        CAQuality.LOW: 2,
        CAQuality.MEDIUM: 5,
        CAQuality.HIGH: 10,
    }

    def __init__(self, settings: Optional[ChromaticAberrationSettings] = None):
        """Initialize chromatic aberration effect.

        Args:
            settings: CA configuration, uses defaults if None.
        """
        self.settings = settings or ChromaticAberrationSettings()

    @staticmethod
    def _smoothstep(edge0: float, edge1: float, x: float) -> float:
        """Compute smoothstep interpolation.

        Args:
            edge0: Lower edge of transition.
            edge1: Upper edge of transition.
            x: Input value.

        Returns:
            Smoothly interpolated value between 0 and 1.
        """
        if edge1 == edge0:
            return 0.0 if x < edge0 else 1.0
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
        return t * t * (3.0 - 2.0 * t)

    def calculate_offsets(
        self, uv: Tuple[float, float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Calculate per-channel UV offsets.

        Args:
            uv: Normalized screen coordinates (0-1, 0-1).

        Returns:
            Tuple of (red_uv, green_uv, blue_uv) coordinates.
        """
        if not self.settings.enabled or self.settings.quality == CAQuality.OFF:
            return (uv, uv, uv)

        # Center UV
        cx = uv[0] - 0.5
        cy = uv[1] - 0.5

        # Apply anamorphic ratio to vertical axis
        cy *= self.settings.anamorphic_ratio

        # Radial distance from center
        dist = math.sqrt(cx * cx + cy * cy)

        # Direction from center (normalized)
        if dist > 0.001:
            dx = cx / dist
            dy = cy / dist
        else:
            dx, dy = 0.0, 0.0

        # Fringe suppression: reduce near center, full at edges
        fringe = self._smoothstep(self.settings.fringe_start, self.settings.fringe_end, dist)

        # Calculate offset scale (normalized to 1080p reference)
        max_px = self.MAX_OFFSET[self.settings.quality]
        scale = max_px * self.settings.intensity * dist * fringe / 1920.0

        red_scale = scale * self.settings.red_offset
        blue_scale = scale * self.settings.blue_offset

        red_uv = (uv[0] + dx * red_scale, uv[1] + dy * red_scale)
        green_uv = uv  # Green channel is reference
        blue_uv = (uv[0] + dx * blue_scale, uv[1] + dy * blue_scale)

        return (red_uv, green_uv, blue_uv)

    def get_offset_magnitude(self, uv: Tuple[float, float]) -> float:
        """Get the magnitude of chromatic offset at a UV coordinate.

        Args:
            uv: Normalized screen coordinates (0-1, 0-1).

        Returns:
            Maximum offset magnitude in normalized units.
        """
        red_uv, green_uv, blue_uv = self.calculate_offsets(uv)

        red_dist = math.sqrt(
            (red_uv[0] - green_uv[0]) ** 2 + (red_uv[1] - green_uv[1]) ** 2
        )
        blue_dist = math.sqrt(
            (blue_uv[0] - green_uv[0]) ** 2 + (blue_uv[1] - green_uv[1]) ** 2
        )

        return max(red_dist, blue_dist)


# ============================================================================
# T-PP-3.5: FILM GRAIN
# ============================================================================


class GrainQuality(Enum):
    """Film grain quality levels."""

    OFF = auto()
    UNIFORM = auto()  # Fast uniform noise
    GAUSSIAN = auto()  # Gaussian-shaped
    GAUSSIAN_CHROMA = auto()  # Gaussian + chrominance grain


@dataclass
class FilmGrainSettings:
    """Film grain parameters.

    Attributes:
        enabled: Whether film grain effect is active.
        quality: Quality level determining noise algorithm.
        intensity: Grain strength (0.0-0.2 typical range).
        response: Luminance response curve exponent.
        chroma_intensity: Chrominance grain strength in dark regions.
        size: Grain size multiplier (not used in CPU implementation).
    """

    enabled: bool = True
    quality: GrainQuality = GrainQuality.GAUSSIAN
    intensity: float = 0.05
    response: float = 0.8
    chroma_intensity: float = 0.02
    size: float = 1.0


class FilmGrainEffect:
    """Procedural film grain with luminance modulation.

    Generates pseudo-random noise that varies per frame and is modulated
    by scene luminance to appear more visible in midtones.
    """

    def __init__(self, settings: Optional[FilmGrainSettings] = None):
        """Initialize film grain effect.

        Args:
            settings: Grain configuration, uses defaults if None.
        """
        self.settings = settings or FilmGrainSettings()
        self._frame_index = 0

    def advance_frame(self) -> None:
        """Advance frame counter for temporal variation."""
        self._frame_index = (self._frame_index + 1) % 65536

    def set_frame(self, frame: int) -> None:
        """Set frame counter directly.

        Args:
            frame: Frame index (will be wrapped to 0-65535).
        """
        self._frame_index = frame % 65536

    @property
    def frame_index(self) -> int:
        """Get current frame index."""
        return self._frame_index

    def wang_hash(self, seed: int) -> int:
        """Wang hash for pseudo-random number generation.

        Args:
            seed: Input seed value.

        Returns:
            32-bit hashed value.
        """
        seed = (seed ^ 61) ^ (seed >> 16)
        seed = seed + (seed << 3)
        seed = seed ^ (seed >> 4)
        seed = seed * 0x27D4EB2D
        seed = seed ^ (seed >> 15)
        return seed & 0xFFFFFFFF

    def random_float(self, x: int, y: int) -> float:
        """Generate random float [0, 1] at pixel position.

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.

        Returns:
            Pseudo-random float in [0, 1].
        """
        seed = x + y * 1920 + self._frame_index * 1920 * 1080
        return (self.wang_hash(seed) & 0xFFFFFF) / float(0xFFFFFF)

    def gaussian_noise(self, x: int, y: int) -> float:
        """Generate Gaussian-distributed noise via sum of uniforms.

        Uses sum of 4 uniform random values to approximate Gaussian
        distribution (Central Limit Theorem).

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.

        Returns:
            Approximately Gaussian noise in [-1, 1].
        """
        total = 0.0
        for i in range(4):
            total += self.random_float(x + i * 7, y + i * 13)
        return (total / 4.0 - 0.5) * 2.0

    def calculate_luminance_factor(self, luminance: float) -> float:
        """Calculate grain visibility factor based on luminance.

        Grain is most visible in midtones, fading in shadows and highlights.

        Args:
            luminance: Scene luminance at pixel (0-1).

        Returns:
            Luminance modulation factor.
        """
        # Bell curve centered at 0.5 luminance
        lum_factor = 4.0 * luminance * (1.0 - luminance)
        return pow(max(0.0, lum_factor), self.settings.response)

    def calculate_grain(
        self, x: int, y: int, luminance: float
    ) -> Tuple[float, float, float]:
        """Calculate grain offset for pixel.

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.
            luminance: Scene luminance at pixel (0-1).

        Returns:
            RGB grain offset (typically in -intensity to +intensity range).
        """
        if not self.settings.enabled or self.settings.quality == GrainQuality.OFF:
            return (0.0, 0.0, 0.0)

        lum_factor = self.calculate_luminance_factor(luminance)
        intensity = self.settings.intensity * lum_factor

        if self.settings.quality == GrainQuality.UNIFORM:
            noise = self.random_float(x, y) - 0.5
            return (noise * intensity, noise * intensity, noise * intensity)

        # Gaussian grain
        noise = self.gaussian_noise(x, y)
        luma_grain = noise * intensity

        if self.settings.quality == GrainQuality.GAUSSIAN_CHROMA:
            # Add chrominance grain in dark regions
            dark_factor = 1.0 - luminance
            chroma_r = (
                self.gaussian_noise(x + 1000, y)
                * self.settings.chroma_intensity
                * dark_factor
            )
            chroma_b = (
                self.gaussian_noise(x + 2000, y)
                * self.settings.chroma_intensity
                * dark_factor
            )
            return (luma_grain + chroma_r, luma_grain, luma_grain + chroma_b)

        return (luma_grain, luma_grain, luma_grain)

    def apply_to_color(
        self, color: Tuple[float, float, float], x: int, y: int
    ) -> Tuple[float, float, float]:
        """Apply grain to an RGB color.

        Args:
            color: Input RGB color (0-1 range).
            x: Pixel x coordinate.
            y: Pixel y coordinate.

        Returns:
            Color with grain applied, clamped to [0, 1].
        """
        # Calculate luminance (BT.709 coefficients)
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        luminance = max(0.0, min(1.0, luminance))

        grain = self.calculate_grain(x, y, luminance)

        return (
            max(0.0, min(1.0, color[0] + grain[0])),
            max(0.0, min(1.0, color[1] + grain[1])),
            max(0.0, min(1.0, color[2] + grain[2])),
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Vignette
    "VignetteSettings",
    "VignetteEffect",
    # Chromatic Aberration
    "CAQuality",
    "ChromaticAberrationSettings",
    "ChromaticAberrationEffect",
    # Film Grain
    "GrainQuality",
    "FilmGrainSettings",
    "FilmGrainEffect",
]

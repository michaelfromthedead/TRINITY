"""Lens Flare Post-Processing Effects.

Implements screen-space lens flare using bloom bright-pass output:
- Ghost generation (mirrored across lens center)
- Halo (central glow from brightest region)
- Anamorphic streaks (horizontal direction)

Lens flares are artifacts that appear in camera systems when bright light
sources scatter through the lens elements. This module simulates:

1. **Ghosts**: Inverted, scaled copies of bright spots that appear on the
   opposite side of the frame from the light source. Each ghost can have
   independent scale, offset, chromatic aberration, and tinting.

2. **Halo**: A radial glow centered on screen that simulates light
   scattering at the lens aperture. Uses cubic falloff for natural falloff.

3. **Anamorphic Streaks**: Horizontal (or angled) streaks that extend from
   bright sources, simulating the elongated bokeh of anamorphic lenses.

The effect reuses the bloom bright-pass texture as input, ensuring consistent
thresholding and avoiding redundant GPU passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from .postprocess_stack import EffectPriority, EffectSettings, PostProcessEffect


class LensFlareQuality(Enum):
    """Lens flare quality levels.

    Quality affects the number of ghosts rendered and which sub-effects
    are enabled. Higher quality levels add more ghosts, halo, and streaks.
    """

    OFF = auto()        # Disabled
    LOW = auto()        # 3 ghosts only
    MEDIUM = auto()     # 6 ghosts + halo
    HIGH = auto()       # 8 ghosts + halo + streaks


@dataclass
class GhostSettings:
    """Individual ghost parameters.

    Each ghost is a scaled, offset copy of the bright regions that appears
    mirrored across the lens center. Chromatic shift separates RGB channels
    to simulate lens dispersion.

    Attributes:
        scale: Size multiplier relative to source (0.2-1.5 typical).
        offset: Distance from center along the mirror axis (0=center, 1=edge).
        chromatic_shift: RGB separation amount (0=none, 0.05=subtle, 0.1=strong).
        intensity: Brightness multiplier for this ghost.
        tint: RGB color tint applied to this ghost.
    """

    scale: float = 1.0
    offset: float = 0.5
    chromatic_shift: float = 0.0
    intensity: float = 1.0
    tint: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class HaloSettings:
    """Halo (central glow) parameters.

    The halo is a radial glow centered on screen that simulates light
    scattering at the lens aperture boundary. It's most visible when
    bright sources are near the center of frame.

    Attributes:
        enabled: Whether to render the halo effect.
        radius: Halo radius as fraction of screen (0-1, 0.3 typical).
        intensity: Brightness multiplier.
        falloff: Cubic falloff exponent (higher = sharper edge).
        aspect_ratio: Elliptical distortion (1=circular, <1=tall, >1=wide).
    """

    enabled: bool = True
    radius: float = 0.3
    intensity: float = 0.5
    falloff: float = 3.0
    aspect_ratio: float = 1.0


@dataclass
class StreakSettings:
    """Anamorphic streak parameters.

    Streaks extend from bright sources in a specific direction, simulating
    the elongated bokeh of anamorphic lenses. Multiple streaks can be
    rendered at different angles.

    Attributes:
        enabled: Whether to render streak effects.
        direction: Angle in radians (0=horizontal, pi/2=vertical).
        length: Streak length as fraction of screen (0-1).
        spacing: Width/gap between streak samples.
        falloff: Intensity falloff along streak length (higher = faster fade).
        count: Number of streaks at different angles (for starburst effect).
    """

    enabled: bool = True
    direction: float = 0.0
    length: float = 0.3
    spacing: float = 0.1
    falloff: float = 2.0
    count: int = 4


@dataclass
class LensFlareSettings(EffectSettings):
    """Complete lens flare configuration.

    Combines all lens flare sub-effects into a single configuration.
    Settings can be interpolated for smooth transitions between presets
    or volume blending.

    Attributes:
        enabled: Master enable for the entire effect.
        quality: Quality preset affecting ghost count and sub-effects.
        intensity: Overall intensity multiplier applied to all sub-effects.
        threshold: Minimum brightness to generate flare (reuses bloom threshold).
        ghost_count: Number of ghosts to render (auto-set by quality).
        ghost_dispersion: Chromatic dispersion multiplier per ghost.
        halo: Halo sub-effect settings.
        streaks: Streak sub-effect settings.
    """

    enabled: bool = True
    quality: LensFlareQuality = LensFlareQuality.MEDIUM
    intensity: float = 1.0
    threshold: float = 0.9

    ghost_count: int = 6
    ghost_dispersion: float = 0.5

    halo: HaloSettings = field(default_factory=HaloSettings)
    streaks: StreakSettings = field(default_factory=StreakSettings)

    def __post_init__(self) -> None:
        """Validate settings and set priority."""
        # Lens flare runs after bloom (priority 100)
        self.priority = EffectPriority.BLOOM.value + 10

        if not 0.0 <= self.threshold <= 10.0:
            raise ValueError(f"threshold must be in [0, 10], got {self.threshold}")
        if not 0.0 <= self.intensity <= 10.0:
            raise ValueError(f"intensity must be in [0, 10], got {self.intensity}")
        if self.ghost_count < 0 or self.ghost_count > 16:
            raise ValueError(f"ghost_count must be in [0, 16], got {self.ghost_count}")
        if not 0.0 <= self.ghost_dispersion <= 2.0:
            raise ValueError(
                f"ghost_dispersion must be in [0, 2], got {self.ghost_dispersion}"
            )

    def lerp(self, other: "LensFlareSettings", t: float) -> "LensFlareSettings":
        """Interpolate between two lens flare settings.

        Args:
            other: Target settings to interpolate towards.
            t: Interpolation factor [0, 1].

        Returns:
            Interpolated settings.
        """
        return LensFlareSettings(
            enabled=self.enabled if t < 0.5 else other.enabled,
            weight=self.weight + (other.weight - self.weight) * t,
            quality=self.quality if t < 0.5 else other.quality,
            intensity=self.intensity + (other.intensity - self.intensity) * t,
            threshold=self.threshold + (other.threshold - self.threshold) * t,
            ghost_count=int(
                self.ghost_count + (other.ghost_count - self.ghost_count) * t
            ),
            ghost_dispersion=self.ghost_dispersion
            + (other.ghost_dispersion - self.ghost_dispersion) * t,
            halo=HaloSettings(
                enabled=self.halo.enabled if t < 0.5 else other.halo.enabled,
                radius=self.halo.radius + (other.halo.radius - self.halo.radius) * t,
                intensity=self.halo.intensity
                + (other.halo.intensity - self.halo.intensity) * t,
                falloff=self.halo.falloff
                + (other.halo.falloff - self.halo.falloff) * t,
                aspect_ratio=self.halo.aspect_ratio
                + (other.halo.aspect_ratio - self.halo.aspect_ratio) * t,
            ),
            streaks=StreakSettings(
                enabled=self.streaks.enabled if t < 0.5 else other.streaks.enabled,
                direction=self.streaks.direction
                + (other.streaks.direction - self.streaks.direction) * t,
                length=self.streaks.length
                + (other.streaks.length - self.streaks.length) * t,
                spacing=self.streaks.spacing
                + (other.streaks.spacing - self.streaks.spacing) * t,
                falloff=self.streaks.falloff
                + (other.streaks.falloff - self.streaks.falloff) * t,
                count=int(
                    self.streaks.count + (other.streaks.count - self.streaks.count) * t
                ),
            ),
        )


class LensFlareEffect(PostProcessEffect[LensFlareSettings]):
    """Screen-space lens flare generator.

    This effect creates realistic lens flare artifacts by processing the
    bloom bright-pass texture. It generates ghosts (mirrored copies),
    halos (radial glow), and anamorphic streaks.

    The effect is designed to integrate with the existing bloom pipeline,
    reusing the bright-pass extraction to avoid redundant GPU work.

    Example:
        >>> settings = LensFlareSettings(quality=LensFlareQuality.HIGH)
        >>> effect = LensFlareEffect(settings)
        >>> effect.setup(1920, 1080)
        >>> # Ghost positions are calculated per-bright-pixel
        >>> ghost = effect.ghosts[0]
        >>> source_uv = (0.8, 0.2)  # Bright spot in upper-right
        >>> ghost_uv = effect.calculate_ghost_uv(source_uv, ghost)
        >>> # Ghost appears on opposite side of center
        >>> assert ghost_uv[0] < 0.5  # Left side
    """

    # Default ghost configurations per quality level
    GHOST_CONFIGS: Dict[LensFlareQuality, List[GhostSettings]] = {
        LensFlareQuality.OFF: [],
        LensFlareQuality.LOW: [
            GhostSettings(scale=0.8, offset=0.3, chromatic_shift=0.0),
            GhostSettings(scale=0.6, offset=0.5, chromatic_shift=0.02),
            GhostSettings(scale=0.4, offset=0.7, chromatic_shift=0.04),
        ],
        LensFlareQuality.MEDIUM: [
            GhostSettings(scale=1.0, offset=0.2, chromatic_shift=0.0),
            GhostSettings(scale=0.8, offset=0.35, chromatic_shift=0.01),
            GhostSettings(scale=0.6, offset=0.5, chromatic_shift=0.02),
            GhostSettings(scale=0.5, offset=0.6, chromatic_shift=0.03),
            GhostSettings(scale=0.4, offset=0.75, chromatic_shift=0.04),
            GhostSettings(scale=0.3, offset=0.9, chromatic_shift=0.05),
        ],
        LensFlareQuality.HIGH: [
            GhostSettings(scale=1.2, offset=0.15, chromatic_shift=0.0),
            GhostSettings(scale=1.0, offset=0.25, chromatic_shift=0.01),
            GhostSettings(scale=0.8, offset=0.4, chromatic_shift=0.015),
            GhostSettings(scale=0.7, offset=0.5, chromatic_shift=0.02),
            GhostSettings(scale=0.5, offset=0.6, chromatic_shift=0.03),
            GhostSettings(scale=0.4, offset=0.7, chromatic_shift=0.04),
            GhostSettings(scale=0.3, offset=0.8, chromatic_shift=0.05),
            GhostSettings(scale=0.2, offset=0.95, chromatic_shift=0.06),
        ],
    }

    def __init__(self, settings: Optional[LensFlareSettings] = None) -> None:
        """Initialize lens flare effect.

        Args:
            settings: Lens flare configuration. Uses defaults if None.
        """
        settings = settings or LensFlareSettings()
        super().__init__(
            name="LensFlare",
            settings=settings,
            priority=EffectPriority.BLOOM.value + 10,
        )
        self._ghosts: List[GhostSettings] = []
        self._width: int = 0
        self._height: int = 0
        self._flare_buffer: Any = None
        self._update_ghosts()

    def _update_ghosts(self) -> None:
        """Update ghost list based on quality setting."""
        if self._settings:
            self._ghosts = list(
                self.GHOST_CONFIGS.get(self._settings.quality, [])
            )

    @property
    def ghosts(self) -> List[GhostSettings]:
        """Get current ghost configuration.

        Returns:
            List of ghost settings for the current quality level.
        """
        return self._ghosts.copy()

    def mirror_uv(self, uv: Tuple[float, float]) -> Tuple[float, float]:
        """Mirror UV coordinates across lens center (0.5, 0.5).

        Ghosts appear on the opposite side of the frame from the bright
        source. This function mirrors a UV coordinate across the center
        point.

        Args:
            uv: Input UV coordinates (0-1 range).

        Returns:
            Mirrored UV coordinates.

        Example:
            >>> effect = LensFlareEffect()
            >>> effect.mirror_uv((0.8, 0.2))  # Upper-right
            (0.2, 0.8)  # Lower-left
        """
        return (1.0 - uv[0], 1.0 - uv[1])

    def calculate_ghost_uv(
        self,
        source_uv: Tuple[float, float],
        ghost: GhostSettings,
    ) -> Tuple[float, float]:
        """Calculate ghost UV position from source bright spot.

        The ghost position is determined by mirroring the source across
        the center, then interpolating toward the center based on the
        ghost's offset parameter.

        Args:
            source_uv: UV of bright pixel in source image.
            ghost: Ghost configuration with offset parameter.

        Returns:
            UV position where ghost should appear.

        Example:
            >>> effect = LensFlareEffect()
            >>> ghost = GhostSettings(offset=0.5)
            >>> # Source at (0.8, 0.2), mirrored is (0.2, 0.8)
            >>> # With offset=0.5, ghost is halfway between center and mirror
            >>> effect.calculate_ghost_uv((0.8, 0.2), ghost)
            (0.35, 0.65)
        """
        # Mirror across center
        mirrored = self.mirror_uv(source_uv)

        # Interpolate toward center based on offset
        cx, cy = 0.5, 0.5
        t = ghost.offset

        gx = cx + (mirrored[0] - cx) * t
        gy = cy + (mirrored[1] - cy) * t

        return (gx, gy)

    def calculate_ghost_chromatic_uv(
        self,
        ghost_uv: Tuple[float, float],
        ghost: GhostSettings,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Calculate chromatic-shifted UVs for RGB channels.

        Chromatic aberration causes different wavelengths of light to
        focus at different positions. This function calculates separate
        UV offsets for red, green, and blue channels.

        Red light bends less (outer), blue light bends more (inner).

        Args:
            ghost_uv: Base UV position for the ghost.
            ghost: Ghost configuration with chromatic_shift parameter.

        Returns:
            Tuple of (red_uv, green_uv, blue_uv).
        """
        if not self._settings:
            return (ghost_uv, ghost_uv, ghost_uv)

        cx, cy = 0.5, 0.5
        dx = ghost_uv[0] - cx
        dy = ghost_uv[1] - cy

        shift = ghost.chromatic_shift * self._settings.ghost_dispersion

        # Red shifts outward (away from center)
        red_uv = (ghost_uv[0] + dx * shift, ghost_uv[1] + dy * shift)
        # Green stays at base position
        green_uv = ghost_uv
        # Blue shifts inward (toward center)
        blue_uv = (ghost_uv[0] - dx * shift, ghost_uv[1] - dy * shift)

        return (red_uv, green_uv, blue_uv)

    def calculate_halo(self, uv: Tuple[float, float]) -> float:
        """Calculate halo intensity at UV position.

        The halo is a radial glow centered on screen with cubic falloff.
        It simulates light scattering at the lens aperture boundary.

        Args:
            uv: UV position to calculate halo intensity at.

        Returns:
            Halo intensity in [0, 1] range.
        """
        if not self._settings or not self._settings.halo.enabled:
            return 0.0

        halo = self._settings.halo
        cx, cy = 0.5, 0.5

        # Apply aspect ratio to create elliptical halo
        dx = (uv[0] - cx) / halo.aspect_ratio
        dy = uv[1] - cy

        dist = math.sqrt(dx * dx + dy * dy)
        radius = halo.radius

        if dist > radius:
            return 0.0

        # Cubic falloff from edge
        t = 1.0 - dist / radius
        intensity = pow(t, halo.falloff)

        return intensity * halo.intensity

    def calculate_streak(
        self,
        uv: Tuple[float, float],
        source_uv: Tuple[float, float],
    ) -> float:
        """Calculate anamorphic streak intensity.

        Streaks extend from bright sources in a specific direction,
        simulating the elongated bokeh characteristic of anamorphic lenses.

        Args:
            uv: UV position to calculate streak intensity at.
            source_uv: UV position of the bright source.

        Returns:
            Streak intensity in [0, 1] range.
        """
        if not self._settings or not self._settings.streaks.enabled:
            return 0.0

        streaks = self._settings.streaks

        # Direction vector
        cos_a = math.cos(streaks.direction)
        sin_a = math.sin(streaks.direction)

        # Vector from source to current UV
        dx = uv[0] - source_uv[0]
        dy = uv[1] - source_uv[1]

        # Project onto streak direction
        along = dx * cos_a + dy * sin_a
        perp = abs(dx * sin_a - dy * cos_a)

        # Check if within streak width
        if perp > streaks.spacing * 0.5:
            return 0.0

        # Distance falloff
        dist = abs(along)
        if dist > streaks.length:
            return 0.0

        intensity = pow(1.0 - dist / streaks.length, streaks.falloff)

        # Perpendicular falloff (smooth edges)
        edge_factor = 1.0 - perp / (streaks.spacing * 0.5)

        return intensity * edge_factor

    def get_budget_ms(self) -> float:
        """Get expected GPU time budget in milliseconds.

        Returns:
            Expected execution time based on quality level.
        """
        if not self._settings:
            return 0.0

        budgets = {
            LensFlareQuality.OFF: 0.0,
            LensFlareQuality.LOW: 0.03,
            LensFlareQuality.MEDIUM: 0.05,
            LensFlareQuality.HIGH: 0.08,
        }
        return budgets.get(self._settings.quality, 0.05)

    # PostProcessEffect interface implementation

    def get_required_inputs(self) -> List[str]:
        """Get required input resources.

        Lens flare reads from the bloom bright-pass texture.

        Returns:
            List of input resource names.
        """
        return ["color", "bloom_buffer"]

    def get_outputs(self) -> List[str]:
        """Get output resources.

        Returns:
            List of output resource names.
        """
        return ["color"]

    def setup(self, width: int, height: int) -> None:
        """Initialize or resize effect resources.

        Args:
            width: Target render width in pixels.
            height: Target render height in pixels.
        """
        self._width = width
        self._height = height
        self._update_ghosts()

    def execute(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        delta_time: float,
    ) -> None:
        """Execute the lens flare effect.

        Processes the bloom bright-pass to generate ghosts, halo, and
        streaks, then composites onto the output.

        Args:
            inputs: Input resources (color, bloom_buffer).
            outputs: Output resources (color).
            delta_time: Time since last frame in seconds.
        """
        if not self._settings or not self._settings.enabled:
            return

        if self._settings.quality == LensFlareQuality.OFF:
            return

        if self._settings.intensity <= 0:
            return

        # In a real implementation, this would:
        # 1. Sample bloom bright-pass for bright spots
        # 2. Generate ghosts at mirrored positions
        # 3. Apply chromatic aberration to ghosts
        # 4. Add halo glow
        # 5. Add anamorphic streaks
        # 6. Composite onto output with overall intensity
        pass

    def cleanup(self) -> None:
        """Release any GPU resources held by the effect."""
        self._flare_buffer = None

    def is_compute_effect(self) -> bool:
        """Whether this effect uses compute shaders.

        Returns:
            True since lens flare uses compute for efficiency.
        """
        return True


# Factory functions

def create_cinematic_lens_flare() -> LensFlareEffect:
    """Create lens flare preset for cinematic look.

    Provides high-quality lens flare with strong ghosts, visible halo,
    and prominent anamorphic streaks for a cinematic appearance.

    Returns:
        Configured LensFlareEffect instance.
    """
    settings = LensFlareSettings(
        quality=LensFlareQuality.HIGH,
        intensity=0.8,
        halo=HaloSettings(intensity=0.3, radius=0.4),
        streaks=StreakSettings(enabled=True, length=0.4, count=6),
    )
    return LensFlareEffect(settings)


def create_subtle_lens_flare() -> LensFlareEffect:
    """Create subtle lens flare for realistic look.

    Provides medium-quality lens flare with restrained intensity
    for a more natural, documentary-style appearance.

    Returns:
        Configured LensFlareEffect instance.
    """
    settings = LensFlareSettings(
        quality=LensFlareQuality.MEDIUM,
        intensity=0.4,
        halo=HaloSettings(intensity=0.2, radius=0.2),
        streaks=StreakSettings(enabled=False),
    )
    return LensFlareEffect(settings)


def create_disabled_lens_flare() -> LensFlareEffect:
    """Create disabled lens flare effect.

    Useful for testing or when lens flare should be conditionally disabled.

    Returns:
        Disabled LensFlareEffect instance.
    """
    settings = LensFlareSettings(
        enabled=False,
        quality=LensFlareQuality.OFF,
    )
    return LensFlareEffect(settings)


__all__ = [
    "LensFlareQuality",
    "GhostSettings",
    "HaloSettings",
    "StreakSettings",
    "LensFlareSettings",
    "LensFlareEffect",
    "create_cinematic_lens_flare",
    "create_subtle_lens_flare",
    "create_disabled_lens_flare",
]

"""Foveated rendering for XR displays.

Supports multiple foveation types:
- Fixed: Static quality regions based on typical eye position
- Dynamic: Eye-tracked foveation following gaze
- Contrast-Adaptive: Adjusts based on scene content

Uses Variable Rate Shading (VRS) when available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple, Callable
import math
import threading

from engine.xr.utils.shading import shading_rate_to_int, get_rate_multiplier
from engine.xr.config import XR_CONFIG


class FoveationType(Enum):
    """Foveation rendering type."""
    NONE = auto()             # No foveation
    FIXED = auto()            # Static regions
    DYNAMIC = auto()          # Eye-tracked
    CONTRAST_ADAPTIVE = auto() # Scene content-aware


class FoveationRegion(Enum):
    """Foveation quality region."""
    FOVEA = auto()        # Center - highest quality (1-2 degrees)
    PARAFOVEAL = auto()   # Mid-ring - medium quality (5-10 degrees)
    PERIPHERAL = auto()   # Outer - lowest quality


class ShadingRate(Enum):
    """Variable Rate Shading levels."""
    FULL = auto()          # 1x1 - full resolution
    HALF_X = auto()        # 2x1 - half horizontal
    HALF_Y = auto()        # 1x2 - half vertical
    HALF = auto()          # 2x2 - quarter resolution
    QUARTER_X = auto()     # 4x1 - quarter horizontal
    QUARTER_Y = auto()     # 1x4 - quarter vertical
    QUARTER = auto()       # 4x4 - 1/16 resolution


@dataclass
class FoveationRegionConfig:
    """Configuration for a foveation region."""
    region: FoveationRegion
    inner_radius: float       # Inner radius in degrees
    outer_radius: float       # Outer radius in degrees
    shading_rate: ShadingRate
    quality_scale: float = 1.0  # 0.0-1.0 quality multiplier


@dataclass
class GazePoint:
    """Eye gaze point in normalized device coordinates."""
    x: float = 0.0  # -1 to 1
    y: float = 0.0  # -1 to 1
    confidence: float = 1.0  # 0 to 1, eye tracking confidence
    timestamp_ns: int = 0


@dataclass
class FoveationConfig:
    """Foveated rendering configuration."""
    type: FoveationType = FoveationType.FIXED
    enabled: bool = True

    # Region configurations (innermost to outermost)
    fovea_radius: float = 5.0        # Degrees from center
    parafoveal_radius: float = 20.0  # Degrees from center
    peripheral_radius: float = 55.0  # Degrees (to edge)

    # Quality settings per region
    fovea_rate: ShadingRate = ShadingRate.FULL
    parafoveal_rate: ShadingRate = ShadingRate.HALF
    peripheral_rate: ShadingRate = ShadingRate.QUARTER

    # Dynamic foveation settings
    gaze_smoothing: float = XR_CONFIG.runtime.GAZE_SMOOTHING_FACTOR  # Smoothing factor (0-1)
    confidence_threshold: float = 0.5  # Min eye tracking confidence

    # Contrast-adaptive settings
    contrast_sensitivity: float = 0.5  # How much contrast affects quality
    motion_sensitivity: float = 0.3    # How much motion affects quality

    # VRS settings
    use_vrs: bool = True
    vrs_tile_size: int = 16  # Pixels per VRS tile


@dataclass
class FoveationMetrics:
    """Performance metrics for foveated rendering."""
    pixel_savings: float = 0.0      # Percentage of pixels saved
    bandwidth_savings: float = 0.0  # Percentage of bandwidth saved
    current_gaze: Optional[GazePoint] = None
    regions_active: int = 0


class FoveatedRenderer(ABC):
    """Abstract foveated renderer interface."""

    @property
    @abstractmethod
    def config(self) -> FoveationConfig:
        """Get current foveation configuration."""
        pass

    @abstractmethod
    def configure(self, config: FoveationConfig) -> None:
        """Update foveation configuration."""
        pass

    @abstractmethod
    def update_gaze(self, left_gaze: GazePoint, right_gaze: GazePoint) -> None:
        """Update eye gaze positions.

        Args:
            left_gaze: Left eye gaze point
            right_gaze: Right eye gaze point
        """
        pass

    @abstractmethod
    def get_shading_rate_image(self, eye_index: int, width: int, height: int) -> List[int]:
        """Generate VRS shading rate image.

        Args:
            eye_index: 0 for left, 1 for right
            width: Image width in VRS tiles
            height: Image height in VRS tiles

        Returns:
            Flattened shading rate values (one per tile)
        """
        pass

    @abstractmethod
    def get_region_at_point(self, x: float, y: float, eye_index: int) -> FoveationRegion:
        """Get foveation region at a screen point.

        Args:
            x: Normalized x coordinate (-1 to 1)
            y: Normalized y coordinate (-1 to 1)
            eye_index: 0 for left, 1 for right

        Returns:
            Foveation region at that point
        """
        pass

    @abstractmethod
    def get_metrics(self) -> FoveationMetrics:
        """Get current foveation metrics."""
        pass

    @abstractmethod
    def begin_frame(self) -> None:
        """Begin foveated frame rendering."""
        pass

    @abstractmethod
    def end_frame(self) -> None:
        """End foveated frame rendering."""
        pass


class FixedFoveatedRenderer(FoveatedRenderer):
    """Fixed foveated rendering with static quality regions.

    Assumes user is looking at center of display.
    Most compatible, works without eye tracking.
    """

    def __init__(self, config: Optional[FoveationConfig] = None):
        """Initialize fixed foveated renderer."""
        self._config = config or FoveationConfig(type=FoveationType.FIXED)
        self._center_gaze = GazePoint(x=0.0, y=0.0, confidence=1.0)
        self._metrics = FoveationMetrics()
        self._lock = threading.Lock()
        # Cache for shading rate image to avoid per-frame allocations
        self._cached_rates: Optional[List[int]] = None
        self._cached_dimensions: Tuple[int, int] = (0, 0)
        self._update_metrics()

    @property
    def config(self) -> FoveationConfig:
        return self._config

    def configure(self, config: FoveationConfig) -> None:
        with self._lock:
            self._config = config
            self._config.type = FoveationType.FIXED
            self._update_metrics()

    def update_gaze(self, left_gaze: GazePoint, right_gaze: GazePoint) -> None:
        # Fixed foveation ignores gaze updates
        pass

    def get_shading_rate_image(self, eye_index: int, width: int, height: int) -> List[int]:
        """Generate static VRS shading rate image centered on display."""
        # Use cached array if dimensions match to avoid per-frame allocations
        if self._cached_dimensions != (width, height):
            self._cached_rates = [0] * (width * height)
            self._cached_dimensions = (width, height)

        rates = self._cached_rates
        center_x = width / 2
        center_y = height / 2

        # Convert degree radii to normalized tile distances
        # Assuming ~100 degree FOV
        fov_degrees = 100.0
        tiles_per_degree = width / fov_degrees

        fovea_tiles = self._config.fovea_radius * tiles_per_degree
        para_tiles = self._config.parafoveal_radius * tiles_per_degree

        # Pre-compute rate values to avoid repeated dict lookups
        fovea_rate_int = shading_rate_to_int(self._config.fovea_rate)
        para_rate_int = shading_rate_to_int(self._config.parafoveal_rate)
        peri_rate_int = shading_rate_to_int(self._config.peripheral_rate)

        idx = 0
        for y in range(height):
            for x in range(width):
                dx = x - center_x
                dy = y - center_y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist <= fovea_tiles:
                    rates[idx] = fovea_rate_int
                elif dist <= para_tiles:
                    rates[idx] = para_rate_int
                else:
                    rates[idx] = peri_rate_int
                idx += 1

        return rates

    def get_region_at_point(self, x: float, y: float, eye_index: int) -> FoveationRegion:
        """Get region at normalized screen point."""
        # Distance from center in normalized coordinates
        dist = math.sqrt(x * x + y * y)

        # Convert to approximate degrees (assuming ~100 degree FOV)
        dist_degrees = dist * 50.0  # Half of 100 degree FOV

        if dist_degrees <= self._config.fovea_radius:
            return FoveationRegion.FOVEA
        elif dist_degrees <= self._config.parafoveal_radius:
            return FoveationRegion.PARAFOVEAL
        else:
            return FoveationRegion.PERIPHERAL

    def get_metrics(self) -> FoveationMetrics:
        return self._metrics

    def begin_frame(self) -> None:
        pass

    def end_frame(self) -> None:
        self._update_metrics()

    def _update_metrics(self) -> None:
        """Calculate approximate savings metrics."""
        # Estimate based on region areas
        fov_area = math.pi * self._config.fovea_radius ** 2
        para_area = math.pi * self._config.parafoveal_radius ** 2 - fov_area
        peri_area = math.pi * self._config.peripheral_radius ** 2 - para_area - fov_area

        total_area = fov_area + para_area + peri_area

        # Calculate weighted pixel count
        fov_rate = get_rate_multiplier(self._config.fovea_rate)
        para_rate = get_rate_multiplier(self._config.parafoveal_rate)
        peri_rate = get_rate_multiplier(self._config.peripheral_rate)

        effective_pixels = (
            fov_area * fov_rate +
            para_area * para_rate +
            peri_area * peri_rate
        ) / total_area

        self._metrics.pixel_savings = (1.0 - effective_pixels) * 100.0
        self._metrics.bandwidth_savings = self._metrics.pixel_savings * 0.8
        self._metrics.current_gaze = self._center_gaze
        self._metrics.regions_active = 3


class DynamicFoveatedRenderer(FoveatedRenderer):
    """Dynamic foveated rendering with eye tracking.

    Follows user's gaze to maintain highest quality at fovea.
    Requires eye tracking hardware.
    """

    def __init__(self, config: Optional[FoveationConfig] = None):
        """Initialize dynamic foveated renderer."""
        self._config = config or FoveationConfig(type=FoveationType.DYNAMIC)
        self._left_gaze = GazePoint()
        self._right_gaze = GazePoint()
        self._smoothed_left = GazePoint()
        self._smoothed_right = GazePoint()
        self._metrics = FoveationMetrics()
        self._lock = threading.Lock()
        # Cache for shading rate image to avoid per-frame allocations
        self._cached_rates: Optional[List[int]] = None
        self._cached_dimensions: Tuple[int, int] = (0, 0)

    @property
    def config(self) -> FoveationConfig:
        return self._config

    def configure(self, config: FoveationConfig) -> None:
        with self._lock:
            self._config = config
            self._config.type = FoveationType.DYNAMIC

    def update_gaze(self, left_gaze: GazePoint, right_gaze: GazePoint) -> None:
        """Update and smooth gaze positions."""
        with self._lock:
            # Apply smoothing to reduce jitter
            alpha = self._config.gaze_smoothing

            if left_gaze.confidence >= self._config.confidence_threshold:
                self._smoothed_left = GazePoint(
                    x=self._smoothed_left.x * alpha + left_gaze.x * (1 - alpha),
                    y=self._smoothed_left.y * alpha + left_gaze.y * (1 - alpha),
                    confidence=left_gaze.confidence,
                    timestamp_ns=left_gaze.timestamp_ns
                )
                self._left_gaze = left_gaze

            if right_gaze.confidence >= self._config.confidence_threshold:
                self._smoothed_right = GazePoint(
                    x=self._smoothed_right.x * alpha + right_gaze.x * (1 - alpha),
                    y=self._smoothed_right.y * alpha + right_gaze.y * (1 - alpha),
                    confidence=right_gaze.confidence,
                    timestamp_ns=right_gaze.timestamp_ns
                )
                self._right_gaze = right_gaze

    def get_shading_rate_image(self, eye_index: int, width: int, height: int) -> List[int]:
        """Generate VRS image centered on gaze point."""
        with self._lock:
            gaze = self._smoothed_left if eye_index == 0 else self._smoothed_right

        # Use cached array if dimensions match to avoid per-frame allocations
        if self._cached_dimensions != (width, height):
            self._cached_rates = [0] * (width * height)
            self._cached_dimensions = (width, height)

        rates = self._cached_rates

        # Convert gaze from NDC to tile coordinates
        gaze_tile_x = (gaze.x + 1.0) * 0.5 * width
        gaze_tile_y = (gaze.y + 1.0) * 0.5 * height

        # Convert degree radii to tile distances
        fov_degrees = 100.0
        tiles_per_degree = width / fov_degrees

        fovea_tiles = self._config.fovea_radius * tiles_per_degree
        para_tiles = self._config.parafoveal_radius * tiles_per_degree

        # Pre-compute rate values to avoid repeated dict lookups
        fovea_rate_int = shading_rate_to_int(self._config.fovea_rate)
        para_rate_int = shading_rate_to_int(self._config.parafoveal_rate)
        peri_rate_int = shading_rate_to_int(self._config.peripheral_rate)

        idx = 0
        for y in range(height):
            for x in range(width):
                dx = x - gaze_tile_x
                dy = y - gaze_tile_y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist <= fovea_tiles:
                    rates[idx] = fovea_rate_int
                elif dist <= para_tiles:
                    rates[idx] = para_rate_int
                else:
                    rates[idx] = peri_rate_int
                idx += 1

        return rates

    def get_region_at_point(self, x: float, y: float, eye_index: int) -> FoveationRegion:
        """Get region at normalized screen point relative to gaze."""
        with self._lock:
            gaze = self._smoothed_left if eye_index == 0 else self._smoothed_right

        # Distance from gaze point
        dx = x - gaze.x
        dy = y - gaze.y
        dist = math.sqrt(dx * dx + dy * dy)

        # Convert to degrees
        dist_degrees = dist * 50.0

        if dist_degrees <= self._config.fovea_radius:
            return FoveationRegion.FOVEA
        elif dist_degrees <= self._config.parafoveal_radius:
            return FoveationRegion.PARAFOVEAL
        else:
            return FoveationRegion.PERIPHERAL

    def get_metrics(self) -> FoveationMetrics:
        with self._lock:
            # Average gaze for metrics
            self._metrics.current_gaze = GazePoint(
                x=(self._smoothed_left.x + self._smoothed_right.x) / 2,
                y=(self._smoothed_left.y + self._smoothed_right.y) / 2,
                confidence=min(self._smoothed_left.confidence, self._smoothed_right.confidence)
            )
        return self._metrics

    def begin_frame(self) -> None:
        pass

    def end_frame(self) -> None:
        self._update_metrics()

    def _update_metrics(self) -> None:
        """Calculate savings metrics."""
        # Similar calculation to fixed, but actual savings may vary with gaze
        fov_area = math.pi * self._config.fovea_radius ** 2
        para_area = math.pi * self._config.parafoveal_radius ** 2 - fov_area
        peri_area = math.pi * self._config.peripheral_radius ** 2 - para_area - fov_area

        total_area = fov_area + para_area + peri_area

        fov_rate = get_rate_multiplier(self._config.fovea_rate)
        para_rate = get_rate_multiplier(self._config.parafoveal_rate)
        peri_rate = get_rate_multiplier(self._config.peripheral_rate)

        effective_pixels = (
            fov_area * fov_rate +
            para_area * para_rate +
            peri_area * peri_rate
        ) / total_area

        self._metrics.pixel_savings = (1.0 - effective_pixels) * 100.0
        self._metrics.bandwidth_savings = self._metrics.pixel_savings * 0.85
        self._metrics.regions_active = 3


class ContrastAdaptiveFoveatedRenderer(FoveatedRenderer):
    """Contrast-adaptive foveated rendering.

    Adjusts quality based on scene content - higher quality for
    high-contrast regions, lower for uniform areas.
    """

    def __init__(self, config: Optional[FoveationConfig] = None,
                 contrast_callback: Optional[Callable[[int, int], float]] = None):
        """Initialize contrast-adaptive foveated renderer.

        Args:
            config: Foveation configuration
            contrast_callback: Function(x, y) -> contrast value 0-1
        """
        self._config = config or FoveationConfig(type=FoveationType.CONTRAST_ADAPTIVE)
        self._contrast_callback = contrast_callback
        self._left_gaze = GazePoint()
        self._right_gaze = GazePoint()
        self._contrast_map: Optional[List[float]] = None
        self._contrast_width = 0
        self._contrast_height = 0
        self._metrics = FoveationMetrics()
        self._lock = threading.Lock()
        # Cache for shading rate image to avoid per-frame allocations
        self._cached_rates: Optional[List[int]] = None
        self._cached_dimensions: Tuple[int, int] = (0, 0)

    @property
    def config(self) -> FoveationConfig:
        return self._config

    def configure(self, config: FoveationConfig) -> None:
        with self._lock:
            self._config = config
            self._config.type = FoveationType.CONTRAST_ADAPTIVE

    def set_contrast_callback(self, callback: Callable[[int, int], float]) -> None:
        """Set callback for querying scene contrast."""
        self._contrast_callback = callback

    def update_contrast_map(self, contrast_values: List[float], width: int, height: int) -> None:
        """Update contrast map from rendered scene.

        Args:
            contrast_values: Per-tile contrast values (0-1)
            width: Map width in tiles
            height: Map height in tiles
        """
        with self._lock:
            self._contrast_map = contrast_values
            self._contrast_width = width
            self._contrast_height = height

    def update_gaze(self, left_gaze: GazePoint, right_gaze: GazePoint) -> None:
        with self._lock:
            self._left_gaze = left_gaze
            self._right_gaze = right_gaze

    def get_shading_rate_image(self, eye_index: int, width: int, height: int) -> List[int]:
        """Generate VRS image combining gaze and contrast."""
        with self._lock:
            gaze = self._left_gaze if eye_index == 0 else self._right_gaze

        # Use cached array if dimensions match to avoid per-frame allocations
        if self._cached_dimensions != (width, height):
            self._cached_rates = [0] * (width * height)
            self._cached_dimensions = (width, height)

        rates = self._cached_rates

        gaze_tile_x = (gaze.x + 1.0) * 0.5 * width
        gaze_tile_y = (gaze.y + 1.0) * 0.5 * height

        fov_degrees = 100.0
        tiles_per_degree = width / fov_degrees

        fovea_tiles = self._config.fovea_radius * tiles_per_degree
        para_tiles = self._config.parafoveal_radius * tiles_per_degree

        idx = 0
        for y in range(height):
            for x in range(width):
                dx = x - gaze_tile_x
                dy = y - gaze_tile_y
                dist = math.sqrt(dx * dx + dy * dy)

                # Base rate from gaze distance
                if dist <= fovea_tiles:
                    base_rate = self._config.fovea_rate
                elif dist <= para_tiles:
                    base_rate = self._config.parafoveal_rate
                else:
                    base_rate = self._config.peripheral_rate

                # Adjust based on contrast
                contrast = self._get_contrast_at(x, y, width, height)
                adjusted_rate = self._adjust_rate_for_contrast(base_rate, contrast)

                rates[idx] = shading_rate_to_int(adjusted_rate)
                idx += 1

        return rates

    def get_region_at_point(self, x: float, y: float, eye_index: int) -> FoveationRegion:
        with self._lock:
            gaze = self._left_gaze if eye_index == 0 else self._right_gaze

        dx = x - gaze.x
        dy = y - gaze.y
        dist = math.sqrt(dx * dx + dy * dy)
        dist_degrees = dist * 50.0

        if dist_degrees <= self._config.fovea_radius:
            return FoveationRegion.FOVEA
        elif dist_degrees <= self._config.parafoveal_radius:
            return FoveationRegion.PARAFOVEAL
        else:
            return FoveationRegion.PERIPHERAL

    def get_metrics(self) -> FoveationMetrics:
        return self._metrics

    def begin_frame(self) -> None:
        pass

    def end_frame(self) -> None:
        pass

    def _get_contrast_at(self, x: int, y: int, width: int, height: int) -> float:
        """Get contrast value at tile position."""
        if self._contrast_callback:
            return self._contrast_callback(x, y)

        with self._lock:
            if self._contrast_map and self._contrast_width > 0:
                # Map from current dimensions to contrast map
                map_x = int(x * self._contrast_width / width)
                map_y = int(y * self._contrast_height / height)
                map_x = min(map_x, self._contrast_width - 1)
                map_y = min(map_y, self._contrast_height - 1)
                idx = map_y * self._contrast_width + map_x
                if 0 <= idx < len(self._contrast_map):
                    return self._contrast_map[idx]

        return 0.5  # Default medium contrast

    def _adjust_rate_for_contrast(self, base_rate: ShadingRate, contrast: float) -> ShadingRate:
        """Adjust shading rate based on contrast.

        Higher contrast = higher quality (lower rate reduction).
        """
        sensitivity = self._config.contrast_sensitivity

        # For high contrast, keep original or improve rate
        # For low contrast, can reduce further
        if contrast > 0.7:
            # High contrast - keep quality
            return base_rate
        elif contrast < 0.3:
            # Low contrast - can reduce more
            return self._reduce_rate(base_rate)
        else:
            # Medium contrast - keep base rate
            return base_rate

    def _reduce_rate(self, rate: ShadingRate) -> ShadingRate:
        """Reduce shading rate by one step."""
        reductions = {
            ShadingRate.FULL: ShadingRate.HALF,
            ShadingRate.HALF_X: ShadingRate.HALF,
            ShadingRate.HALF_Y: ShadingRate.HALF,
            ShadingRate.HALF: ShadingRate.QUARTER,
            ShadingRate.QUARTER_X: ShadingRate.QUARTER,
            ShadingRate.QUARTER_Y: ShadingRate.QUARTER,
            ShadingRate.QUARTER: ShadingRate.QUARTER
        }
        return reductions.get(rate, rate)


def create_foveated_renderer(config: Optional[FoveationConfig] = None) -> FoveatedRenderer:
    """Factory function to create appropriate foveated renderer.

    Args:
        config: Foveation configuration (determines renderer type)

    Returns:
        Configured foveated renderer instance
    """
    if config is None:
        config = FoveationConfig()

    if not config.enabled or config.type == FoveationType.NONE:
        return FixedFoveatedRenderer(FoveationConfig(enabled=False))

    if config.type == FoveationType.FIXED:
        return FixedFoveatedRenderer(config)
    elif config.type == FoveationType.DYNAMIC:
        return DynamicFoveatedRenderer(config)
    elif config.type == FoveationType.CONTRAST_ADAPTIVE:
        return ContrastAdaptiveFoveatedRenderer(config)
    else:
        return FixedFoveatedRenderer(config)

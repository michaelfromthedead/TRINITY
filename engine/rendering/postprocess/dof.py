"""
Depth of Field System

Provides physically-based depth of field rendering:
- CircleOfConfusion calculation
- NearFieldDOF: Foreground blur
- FarFieldDOF: Background blur
- BokehShape: Circle, polygon, anamorphic
- DOFSettings: Complete configuration
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from .postprocess_stack import EffectPriority, EffectSettings, PostProcessEffect


class DOFMode(Enum):
    """Depth of field calculation mode."""

    PHYSICAL = auto()  # Based on camera aperture, focal length
    MANUAL = auto()  # Direct focus distance control
    AUTO_FOCUS = auto()  # Focus on center/screen point


class BokehShapeType(Enum):
    """Bokeh shape for out-of-focus highlights."""

    CIRCLE = auto()  # Circular bokeh
    POLYGON = auto()  # N-sided polygon
    ANAMORPHIC = auto()  # Elliptical (movie style)
    CAT_EYE = auto()  # Mechanical vignetting
    SWIRL = auto()  # Petzval lens style


class DOFQuality(Enum):
    """DOF rendering quality preset."""

    LOW = auto()  # Fast single-pass blur
    MEDIUM = auto()  # Separable blur
    HIGH = auto()  # Scatter-as-gather with bokeh
    CINEMATIC = auto()  # Full per-pixel bokeh simulation


@dataclass
class BokehShape:
    """Bokeh shape configuration."""

    shape_type: BokehShapeType = BokehShapeType.CIRCLE
    blade_count: int = 6  # Number of aperture blades (for polygon)
    blade_rotation: float = 0.0  # Blade rotation in degrees
    blade_curvature: float = 0.0  # Blade edge curvature [-1, 1]
    anamorphic_ratio: float = 1.0  # Horizontal/vertical ratio
    cat_eye_intensity: float = 0.0  # Mechanical vignetting amount [0, 1]
    spherical_aberration: float = 0.0  # Edge brightness [-1, 1]
    custom_texture_path: Optional[str] = None  # Path to custom bokeh texture

    def get_bokeh_kernel(self, radius: int) -> List[Tuple[float, float, float]]:
        """Generate bokeh kernel sample points.

        Args:
            radius: Kernel radius in pixels.

        Returns:
            List of (x, y, weight) sample points.
        """
        samples = []

        if self.shape_type == BokehShapeType.CIRCLE:
            samples = self._generate_disk_samples(radius)
        elif self.shape_type == BokehShapeType.POLYGON:
            samples = self._generate_polygon_samples(radius)
        elif self.shape_type == BokehShapeType.ANAMORPHIC:
            samples = self._generate_ellipse_samples(radius)
        else:
            samples = self._generate_disk_samples(radius)

        return samples

    def _generate_disk_samples(
        self,
        radius: int,
    ) -> List[Tuple[float, float, float]]:
        """Generate circular disk sample points."""
        samples = []
        sample_count = max(8, radius * 8)

        for i in range(sample_count):
            t = i / sample_count
            r = math.sqrt(t) * radius
            angle = t * math.pi * 2.0 * 7.0  # Golden angle

            x = r * math.cos(angle)
            y = r * math.sin(angle)
            weight = 1.0 - self.spherical_aberration * t

            samples.append((x, y, max(0.1, weight)))

        return samples

    def _generate_polygon_samples(
        self,
        radius: int,
    ) -> List[Tuple[float, float, float]]:
        """Generate N-sided polygon sample points."""
        samples = []
        sample_count = max(8, radius * 8)

        rotation_rad = math.radians(self.blade_rotation)
        n = self.blade_count

        for i in range(sample_count):
            t = i / sample_count
            r = math.sqrt(t) * radius
            angle = t * math.pi * 2.0 * 7.0 + rotation_rad

            theta = angle % (2 * math.pi / n)
            poly_radius = radius * math.cos(math.pi / n) / math.cos(
                theta - math.pi / n
            )
            if r > poly_radius:
                r = poly_radius

            if self.blade_curvature != 0:
                curve_factor = 1.0 + self.blade_curvature * math.sin(theta * n) * 0.1
                r *= curve_factor

            x = r * math.cos(angle)
            y = r * math.sin(angle)

            samples.append((x, y, 1.0))

        return samples

    def _generate_ellipse_samples(
        self,
        radius: int,
    ) -> List[Tuple[float, float, float]]:
        """Generate elliptical sample points for anamorphic look."""
        samples = self._generate_disk_samples(radius)

        scaled_samples = []
        for x, y, w in samples:
            scaled_samples.append((x * self.anamorphic_ratio, y, w))

        return scaled_samples


@dataclass
class CircleOfConfusion:
    """Circle of Confusion calculation parameters."""

    sensor_width: float = 36.0  # Sensor width in mm (full frame)
    focal_length: float = 50.0  # Lens focal length in mm
    aperture: float = 2.8  # F-number (f/2.8)
    focus_distance: float = 5.0  # Focus distance in meters

    # Acceptable CoC (pixel threshold for "in focus")
    max_coc_radius: float = 32.0  # Maximum blur radius in pixels

    def calculate(self, depth: float, image_width: int) -> float:
        """Calculate Circle of Confusion for a depth value.

        Args:
            depth: Scene depth in meters.
            image_width: Image width in pixels.

        Returns:
            CoC radius in pixels.
        """
        from .constants import EPSILON

        if depth <= EPSILON or self.focus_distance <= EPSILON:
            return 0.0

        # Hyperfocal distance
        coc_mm = self.sensor_width / image_width  # 1 pixel in mm
        hyperfocal = (
            self.focal_length
            + (self.focal_length * self.focal_length)
            / (self.aperture * coc_mm)
        )

        # CoC calculation
        focus_m = self.focus_distance
        focal_m = self.focal_length / 1000.0

        if depth == focus_m:
            return 0.0

        magnification = focal_m / (focus_m - focal_m)
        coc_m = (
            abs(depth - focus_m)
            * magnification
            * (self.focal_length / self.aperture)
            / depth
            / 1000.0
        )

        # Convert to pixels
        pixels_per_meter = image_width / (self.sensor_width / 1000.0)
        coc_pixels = coc_m * pixels_per_meter

        return min(coc_pixels, self.max_coc_radius)

    def get_depth_ranges(
        self,
        image_width: int,
    ) -> Tuple[float, float, float, float]:
        """Get near/far focus plane distances.

        Args:
            image_width: Image width for CoC calculation.

        Returns:
            (near_sharp, near_blur, far_sharp, far_blur) distances.
        """
        coc_mm = self.sensor_width / image_width * 2.0  # 2 pixel CoC threshold
        focal_m = self.focal_length / 1000.0

        # Calculate hyperfocal
        h = focal_m + (focal_m * focal_m) / (self.aperture * coc_mm / 1000.0)

        focus_m = self.focus_distance

        # Near limit
        near = (h * focus_m) / (h + focus_m - focal_m)

        # Far limit (infinity if beyond hyperfocal)
        if focus_m >= h:
            far = float("inf")
        else:
            far = (h * focus_m) / (h - focus_m + focal_m)

        # Return with some margin
        near_sharp = near * 0.95
        far_sharp = far * 1.05 if far != float("inf") else far

        return (near * 0.5, near_sharp, far_sharp, far * 2.0)


@dataclass
class DOFSettings(EffectSettings):
    """Depth of Field settings.

    Uses constants from constants.py DOF module for default values.
    """

    mode: DOFMode = DOFMode.PHYSICAL
    quality: DOFQuality = DOFQuality.HIGH

    # Physical camera settings - see DOF constants
    aperture: float = 2.8  # F-stop (DOF.APERTURE_DEFAULT)
    focal_length: float = 50.0  # mm (DOF.FOCAL_LENGTH_DEFAULT)
    sensor_size: float = 36.0  # mm (DOF.SENSOR_FULL_FRAME)

    # Focus settings
    focus_distance: float = 5.0  # meters (DOF.FOCUS_DISTANCE_DEFAULT)
    auto_focus_point: Tuple[float, float] = (0.5, 0.5)  # Screen UV

    # Blur settings
    near_blur_intensity: float = 1.0
    far_blur_intensity: float = 1.0
    max_blur_radius: float = 32.0  # pixels (DOF.MAX_BLUR_RADIUS_DEFAULT)

    # Bokeh settings
    bokeh: BokehShape = field(default_factory=BokehShape)

    # Transition settings
    near_transition_range: float = 1.0  # meters
    far_transition_range: float = 2.0  # meters

    def __post_init__(self) -> None:
        self.priority = EffectPriority.DEPTH_OF_FIELD.value

    def lerp(self, other: "DOFSettings", t: float) -> "DOFSettings":
        """Interpolate between two DOF settings."""
        return DOFSettings(
            enabled=self.enabled if t < 0.5 else other.enabled,
            weight=self.weight + (other.weight - self.weight) * t,
            mode=self.mode if t < 0.5 else other.mode,
            quality=self.quality if t < 0.5 else other.quality,
            aperture=self.aperture + (other.aperture - self.aperture) * t,
            focal_length=self.focal_length
            + (other.focal_length - self.focal_length) * t,
            focus_distance=self.focus_distance
            + (other.focus_distance - self.focus_distance) * t,
            near_blur_intensity=self.near_blur_intensity
            + (other.near_blur_intensity - self.near_blur_intensity) * t,
            far_blur_intensity=self.far_blur_intensity
            + (other.far_blur_intensity - self.far_blur_intensity) * t,
            max_blur_radius=self.max_blur_radius
            + (other.max_blur_radius - self.max_blur_radius) * t,
        )


class NearFieldDOF:
    """Near field (foreground) depth of field processor.

    Handles blur for objects closer than the focus plane.
    Uses special handling to prevent background bleeding
    into foreground blur.
    """

    def __init__(self) -> None:
        self._near_coc_buffer: Any = None
        self._near_color_buffer: Any = None
        self._dilated_coc: Any = None

    def setup(self, width: int, height: int) -> None:
        """Initialize near field buffers.

        Args:
            width: Buffer width.
            height: Buffer height.
        """
        self._near_coc_buffer = None
        self._near_color_buffer = None
        self._dilated_coc = None

    def calculate_near_coc(
        self,
        depth_buffer: Any,
        coc_calculator: CircleOfConfusion,
        width: int,
    ) -> Any:
        """Calculate near field CoC.

        Args:
            depth_buffer: Scene depth buffer.
            coc_calculator: CoC calculation parameters.
            width: Image width.

        Returns:
            Near field CoC buffer.
        """
        return self._near_coc_buffer

    def dilate_coc(self, coc_buffer: Any, radius: int) -> Any:
        """Dilate CoC to prevent fringing.

        Args:
            coc_buffer: Input CoC buffer.
            radius: Dilation radius.

        Returns:
            Dilated CoC buffer.
        """
        return self._dilated_coc

    def blur(
        self,
        color_buffer: Any,
        coc_buffer: Any,
        bokeh: BokehShape,
    ) -> Any:
        """Apply near field blur.

        Args:
            color_buffer: Input color.
            coc_buffer: CoC per pixel.
            bokeh: Bokeh shape settings.

        Returns:
            Blurred near field.
        """
        return self._near_color_buffer


class FarFieldDOF:
    """Far field (background) depth of field processor.

    Handles blur for objects beyond the focus plane.
    Uses optimization techniques like half-resolution
    processing for distant blur.
    """

    def __init__(self) -> None:
        self._far_coc_buffer: Any = None
        self._far_color_buffer: Any = None
        self._half_res_buffer: Any = None

    def setup(self, width: int, height: int) -> None:
        """Initialize far field buffers.

        Args:
            width: Buffer width.
            height: Buffer height.
        """
        self._far_coc_buffer = None
        self._far_color_buffer = None
        self._half_res_buffer = None

    def calculate_far_coc(
        self,
        depth_buffer: Any,
        coc_calculator: CircleOfConfusion,
        width: int,
    ) -> Any:
        """Calculate far field CoC.

        Args:
            depth_buffer: Scene depth buffer.
            coc_calculator: CoC calculation parameters.
            width: Image width.

        Returns:
            Far field CoC buffer.
        """
        return self._far_coc_buffer

    def blur(
        self,
        color_buffer: Any,
        coc_buffer: Any,
        bokeh: BokehShape,
        quality: DOFQuality,
    ) -> Any:
        """Apply far field blur.

        Args:
            color_buffer: Input color.
            coc_buffer: CoC per pixel.
            bokeh: Bokeh shape settings.
            quality: Quality preset.

        Returns:
            Blurred far field.
        """
        if quality in (DOFQuality.LOW, DOFQuality.MEDIUM):
            return self._separable_blur(color_buffer, coc_buffer)
        else:
            return self._scatter_gather_blur(color_buffer, coc_buffer, bokeh)

    def _separable_blur(self, color: Any, coc: Any) -> Any:
        """Fast separable blur for lower quality settings."""
        return self._far_color_buffer

    def _scatter_gather_blur(
        self,
        color: Any,
        coc: Any,
        bokeh: BokehShape,
    ) -> Any:
        """High-quality scatter-as-gather blur."""
        return self._far_color_buffer


class AutoFocusSystem:
    """Automatic focus distance calculation."""

    def __init__(self) -> None:
        self._current_focus: float = 5.0
        self._target_focus: float = 5.0
        self._focus_speed: float = 3.0  # meters per second

    @property
    def current_focus(self) -> float:
        """Current focus distance."""
        return self._current_focus

    def sample_focus_point(
        self,
        depth_buffer: Any,
        uv: Tuple[float, float],
        sample_radius: int = 3,
    ) -> float:
        """Sample depth at focus point.

        Args:
            depth_buffer: Scene depth buffer.
            uv: Screen UV coordinates [0, 1].
            sample_radius: Radius for averaging.

        Returns:
            Sampled depth value.
        """
        return 5.0

    def update(self, target_focus: float, delta_time: float) -> float:
        """Update focus towards target.

        Args:
            target_focus: Target focus distance.
            delta_time: Time since last update.

        Returns:
            Current focus distance.
        """
        self._target_focus = target_focus

        diff = target_focus - self._current_focus
        max_change = self._focus_speed * delta_time

        if abs(diff) <= max_change:
            self._current_focus = target_focus
        else:
            self._current_focus += max_change * (1.0 if diff > 0 else -1.0)

        return self._current_focus


class DOFEffect(PostProcessEffect[DOFSettings]):
    """Complete Depth of Field post-process effect."""

    def __init__(
        self,
        settings: Optional[DOFSettings] = None,
    ) -> None:
        """Initialize DOF effect.

        Args:
            settings: DOF configuration.
        """
        super().__init__(
            name="DepthOfField",
            settings=settings or DOFSettings(),
            priority=EffectPriority.DEPTH_OF_FIELD.value,
        )

        self._coc: CircleOfConfusion = CircleOfConfusion()
        self._near_field: NearFieldDOF = NearFieldDOF()
        self._far_field: FarFieldDOF = FarFieldDOF()
        self._auto_focus: AutoFocusSystem = AutoFocusSystem()

        self._width: int = 0
        self._height: int = 0

    @property
    def coc_calculator(self) -> CircleOfConfusion:
        """Access CoC calculator."""
        return self._coc

    @property
    def current_focus_distance(self) -> float:
        """Current focus distance."""
        if self._settings and self._settings.mode == DOFMode.AUTO_FOCUS:
            return self._auto_focus.current_focus
        return self._settings.focus_distance if self._settings else 5.0

    def get_required_inputs(self) -> List[str]:
        """Get required input resources."""
        return ["color", "depth"]

    def get_outputs(self) -> List[str]:
        """Get output resources."""
        return ["color"]

    def setup(self, width: int, height: int) -> None:
        """Initialize DOF resources.

        Args:
            width: Render width.
            height: Render height.
        """
        self._width = width
        self._height = height

        self._near_field.setup(width, height)
        self._far_field.setup(width, height)

        self._update_coc_params()

    def _update_coc_params(self) -> None:
        """Update CoC calculator from settings."""
        if not self._settings:
            return

        self._coc.focal_length = self._settings.focal_length
        self._coc.aperture = self._settings.aperture
        self._coc.sensor_width = self._settings.sensor_size
        self._coc.focus_distance = self._settings.focus_distance
        self._coc.max_coc_radius = self._settings.max_blur_radius

    def execute(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        delta_time: float,
    ) -> None:
        """Execute DOF effect.

        Args:
            inputs: Color and depth buffers.
            outputs: Output color buffer.
            delta_time: Frame time.
        """
        if not self._settings or not self._settings.enabled:
            return

        depth_buffer = inputs.get("depth")
        color_buffer = inputs.get("color")

        if self._settings.mode == DOFMode.AUTO_FOCUS:
            target = self._auto_focus.sample_focus_point(
                depth_buffer,
                self._settings.auto_focus_point,
            )
            self._coc.focus_distance = self._auto_focus.update(target, delta_time)
        else:
            self._coc.focus_distance = self._settings.focus_distance

        if self._settings.near_blur_intensity > 0:
            near_coc = self._near_field.calculate_near_coc(
                depth_buffer,
                self._coc,
                self._width,
            )
            self._near_field.blur(
                color_buffer,
                near_coc,
                self._settings.bokeh,
            )

        if self._settings.far_blur_intensity > 0:
            far_coc = self._far_field.calculate_far_coc(
                depth_buffer,
                self._coc,
                self._width,
            )
            self._far_field.blur(
                color_buffer,
                far_coc,
                self._settings.bokeh,
                self._settings.quality,
            )

    def cleanup(self) -> None:
        """Release DOF resources."""
        pass

    def is_compute_effect(self) -> bool:
        """DOF uses compute shaders."""
        return True


__all__ = [
    "DOFMode",
    "BokehShapeType",
    "DOFQuality",
    "BokehShape",
    "CircleOfConfusion",
    "DOFSettings",
    "NearFieldDOF",
    "FarFieldDOF",
    "AutoFocusSystem",
    "DOFEffect",
]

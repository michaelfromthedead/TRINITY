"""
UI scaling system for accessibility.

Provides DPI awareness and scaling features:
- DPI detection and scaling factor calculation
- Font scaling (normal, large, extra large)
- Touch target sizing (minimum 44x44 points per WCAG)
- Zoom levels for content magnification
- Scale change event notifications

Reference (ARCHITECTURE_UI.md):
- Text Scaling: Larger fonts
- DPI Awareness: Platform-specific scaling
- Touch Targets: Minimum size requirements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


class DPIAwareness(Enum):
    """DPI awareness modes."""
    UNAWARE = auto()           # No DPI scaling (1:1 pixels)
    SYSTEM_AWARE = auto()      # Scale to system DPI at startup
    PER_MONITOR_AWARE = auto()  # Scale to each monitor's DPI
    PER_MONITOR_V2 = auto()    # Enhanced per-monitor (Windows 10+)


class ScaleMode(Enum):
    """UI scaling mode."""
    NONE = auto()        # No scaling applied
    SYSTEM = auto()      # Use system scale factor
    CUSTOM = auto()      # Use custom scale factor
    AUTO = auto()        # Auto-detect best scale


class FontScalePreset(Enum):
    """Predefined font scale presets."""
    SMALL = auto()       # 0.85x
    NORMAL = auto()      # 1.0x
    LARGE = auto()       # 1.25x
    EXTRA_LARGE = auto()  # 1.5x
    HUGE = auto()        # 2.0x


class ZoomLevel(Enum):
    """Predefined zoom levels."""
    ZOOM_50 = auto()     # 50%
    ZOOM_75 = auto()     # 75%
    ZOOM_100 = auto()    # 100% (default)
    ZOOM_125 = auto()    # 125%
    ZOOM_150 = auto()    # 150%
    ZOOM_175 = auto()    # 175%
    ZOOM_200 = auto()    # 200%
    ZOOM_250 = auto()    # 250%
    ZOOM_300 = auto()    # 300%
    ZOOM_400 = auto()    # 400%


# Zoom level to scale factor mapping
ZOOM_FACTORS: dict[ZoomLevel, float] = {
    ZoomLevel.ZOOM_50: 0.5,
    ZoomLevel.ZOOM_75: 0.75,
    ZoomLevel.ZOOM_100: 1.0,
    ZoomLevel.ZOOM_125: 1.25,
    ZoomLevel.ZOOM_150: 1.5,
    ZoomLevel.ZOOM_175: 1.75,
    ZoomLevel.ZOOM_200: 2.0,
    ZoomLevel.ZOOM_250: 2.5,
    ZoomLevel.ZOOM_300: 3.0,
    ZoomLevel.ZOOM_400: 4.0,
}

# Font scale preset to factor mapping
FONT_SCALE_FACTORS: dict[FontScalePreset, float] = {
    FontScalePreset.SMALL: 0.85,
    FontScalePreset.NORMAL: 1.0,
    FontScalePreset.LARGE: 1.25,
    FontScalePreset.EXTRA_LARGE: 1.5,
    FontScalePreset.HUGE: 2.0,
}

# Standard DPI reference values
# 96 DPI is the Windows standard reference DPI
# 72 DPI is the macOS standard reference DPI
DEFAULT_WINDOWS_DPI = 96.0
DEFAULT_MACOS_DPI = 72.0

# Default scale limits
DEFAULT_MIN_SCALE = 0.5
DEFAULT_MAX_SCALE = 4.0
DEFAULT_MIN_FONT_SCALE = 0.5
DEFAULT_MAX_FONT_SCALE = 3.0


@dataclass
class TouchTargetSize:
    """
    Touch target size requirements per WCAG 2.5.5.

    Minimum touch target size is 44x44 CSS pixels (approximately
    44x44 device-independent points).
    """
    min_width: float = 44.0   # Minimum width in points
    min_height: float = 44.0  # Minimum height in points

    # Spacing requirements
    min_spacing: float = 8.0  # Minimum spacing between targets

    # Enhanced accessibility (AAA)
    enhanced_width: float = 48.0
    enhanced_height: float = 48.0

    def meets_minimum(self, width: float, height: float) -> bool:
        """Check if dimensions meet minimum requirements."""
        return width >= self.min_width and height >= self.min_height

    def meets_enhanced(self, width: float, height: float) -> bool:
        """Check if dimensions meet enhanced (AAA) requirements."""
        return width >= self.enhanced_width and height >= self.enhanced_height

    def get_adjusted_size(
        self,
        width: float,
        height: float,
        use_enhanced: bool = False,
    ) -> tuple[float, float]:
        """
        Get size adjusted to meet requirements.

        Returns the original size if it meets requirements,
        or the minimum required size if it doesn't.
        """
        target_width = self.enhanced_width if use_enhanced else self.min_width
        target_height = self.enhanced_height if use_enhanced else self.min_height

        return (
            max(width, target_width),
            max(height, target_height),
        )


@dataclass
class ScaleConfig:
    """
    Configuration for UI scaling.

    Defines the various scale factors applied to the UI.
    """
    # DPI settings
    dpi_awareness: DPIAwareness = DPIAwareness.PER_MONITOR_AWARE
    base_dpi: float = DEFAULT_WINDOWS_DPI  # Reference DPI (Windows standard)

    # Scale factors
    ui_scale: float = 1.0       # Overall UI scale
    font_scale: float = 1.0     # Text scale
    icon_scale: float = 1.0     # Icon scale
    spacing_scale: float = 1.0  # Spacing/padding scale

    # Zoom
    zoom_level: ZoomLevel = ZoomLevel.ZOOM_100

    # Touch targets
    enforce_touch_targets: bool = True
    use_enhanced_targets: bool = False

    # Limits
    min_scale: float = DEFAULT_MIN_SCALE
    max_scale: float = DEFAULT_MAX_SCALE
    min_font_scale: float = DEFAULT_MIN_FONT_SCALE
    max_font_scale: float = DEFAULT_MAX_FONT_SCALE

    def get_effective_scale(self) -> float:
        """Get the effective overall scale factor."""
        zoom = ZOOM_FACTORS.get(self.zoom_level, 1.0)
        return self.ui_scale * zoom

    def get_effective_font_scale(self) -> float:
        """Get the effective font scale factor."""
        return self.font_scale * self.get_effective_scale()

    def clamp_scale(self, scale: float) -> float:
        """Clamp a scale value to valid range."""
        return max(self.min_scale, min(self.max_scale, scale))

    def clamp_font_scale(self, scale: float) -> float:
        """Clamp a font scale value to valid range."""
        return max(self.min_font_scale, min(self.max_font_scale, scale))


@dataclass
class MonitorInfo:
    """
    Information about a display monitor.

    Contains DPI and geometry information for scaling calculations.
    """
    monitor_id: str
    name: str = ""

    # DPI
    dpi_x: float = 96.0
    dpi_y: float = 96.0

    # Geometry (in physical pixels)
    x: int = 0
    y: int = 0
    width: int = 1920
    height: int = 1080

    # Work area (excluding taskbar, etc.)
    work_x: int = 0
    work_y: int = 0
    work_width: int = 1920
    work_height: int = 1040

    # Flags
    is_primary: bool = False

    def get_scale_factor(self, base_dpi: float = 96.0) -> float:
        """Calculate the scale factor for this monitor."""
        return self.dpi_x / base_dpi

    def get_diagonal_dpi(self) -> float:
        """Get the diagonal DPI (average of x and y)."""
        return (self.dpi_x + self.dpi_y) / 2.0


@dataclass
class ScaleChangeEvent:
    """
    Event fired when scale settings change.

    Contains old and new values for comparison.
    """
    old_ui_scale: float
    new_ui_scale: float
    old_font_scale: float
    new_font_scale: float
    old_zoom: ZoomLevel
    new_zoom: ZoomLevel
    source: str = "user"  # "user", "system", "auto"

    @property
    def ui_scale_changed(self) -> bool:
        """Check if UI scale changed."""
        return self.old_ui_scale != self.new_ui_scale

    @property
    def font_scale_changed(self) -> bool:
        """Check if font scale changed."""
        return self.old_font_scale != self.new_font_scale

    @property
    def zoom_changed(self) -> bool:
        """Check if zoom level changed."""
        return self.old_zoom != self.new_zoom


class ScaleManager:
    """
    Singleton manager for UI scaling.

    Coordinates DPI awareness, scale factors, and zoom levels
    across the UI system.
    """

    _instance: Optional["ScaleManager"] = None

    def __new__(cls) -> "ScaleManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True

        # Configuration
        self._config = ScaleConfig()
        self._touch_target = TouchTargetSize()

        # Monitor information
        self._monitors: dict[str, MonitorInfo] = {}
        self._current_monitor: Optional[str] = None

        # Callbacks
        self._scale_callbacks: list[Callable[[ScaleChangeEvent], None]] = []

        # Enabled state
        self._enabled: bool = True

        # System scale detection
        self._system_scale: float = 1.0
        self._system_font_scale: float = 1.0

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    @property
    def enabled(self) -> bool:
        """Check if scaling is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable scaling."""
        self._enabled = value

    @property
    def config(self) -> ScaleConfig:
        """Get the scale configuration."""
        return self._config

    @property
    def touch_target(self) -> TouchTargetSize:
        """Get the touch target requirements."""
        return self._touch_target

    @property
    def ui_scale(self) -> float:
        """Get the current UI scale factor."""
        return self._config.ui_scale if self._enabled else 1.0

    @property
    def font_scale(self) -> float:
        """Get the current font scale factor."""
        return self._config.font_scale if self._enabled else 1.0

    @property
    def effective_scale(self) -> float:
        """Get the effective overall scale (UI * zoom)."""
        return self._config.get_effective_scale() if self._enabled else 1.0

    @property
    def effective_font_scale(self) -> float:
        """Get the effective font scale (font * effective)."""
        return self._config.get_effective_font_scale() if self._enabled else 1.0

    @property
    def zoom_level(self) -> ZoomLevel:
        """Get the current zoom level."""
        return self._config.zoom_level

    @property
    def dpi_awareness(self) -> DPIAwareness:
        """Get the DPI awareness mode."""
        return self._config.dpi_awareness

    # Scale factor management
    def set_ui_scale(self, scale: float, source: str = "user") -> None:
        """Set the UI scale factor."""
        old_scale = self._config.ui_scale
        new_scale = self._config.clamp_scale(scale)

        if old_scale == new_scale:
            return

        self._config.ui_scale = new_scale
        self._fire_scale_change(old_scale, new_scale, source=source)

    def set_font_scale(self, scale: float, source: str = "user") -> None:
        """Set the font scale factor."""
        old_scale = self._config.font_scale
        new_scale = self._config.clamp_font_scale(scale)

        if old_scale == new_scale:
            return

        self._config.font_scale = new_scale
        self._fire_scale_change(
            self._config.ui_scale,
            self._config.ui_scale,
            old_scale,
            new_scale,
            source=source,
        )

    def set_font_preset(self, preset: FontScalePreset, source: str = "user") -> None:
        """Set font scale using a preset."""
        scale = FONT_SCALE_FACTORS.get(preset, 1.0)
        self.set_font_scale(scale, source)

    def get_font_preset(self) -> FontScalePreset:
        """Get the closest font scale preset for current setting."""
        current = self._config.font_scale
        closest = FontScalePreset.NORMAL
        min_diff = float("inf")

        for preset, factor in FONT_SCALE_FACTORS.items():
            diff = abs(factor - current)
            if diff < min_diff:
                min_diff = diff
                closest = preset

        return closest

    # Zoom management
    def set_zoom_level(self, level: ZoomLevel, source: str = "user") -> None:
        """Set the zoom level."""
        old_zoom = self._config.zoom_level
        if old_zoom == level:
            return

        self._config.zoom_level = level
        self._fire_scale_change(
            self._config.ui_scale,
            self._config.ui_scale,
            self._config.font_scale,
            self._config.font_scale,
            old_zoom,
            level,
            source,
        )

    def zoom_in(self) -> bool:
        """
        Increase zoom level.

        Returns True if zoom was increased.
        """
        levels = list(ZoomLevel)
        current_idx = levels.index(self._config.zoom_level)

        if current_idx < len(levels) - 1:
            self.set_zoom_level(levels[current_idx + 1])
            return True
        return False

    def zoom_out(self) -> bool:
        """
        Decrease zoom level.

        Returns True if zoom was decreased.
        """
        levels = list(ZoomLevel)
        current_idx = levels.index(self._config.zoom_level)

        if current_idx > 0:
            self.set_zoom_level(levels[current_idx - 1])
            return True
        return False

    def reset_zoom(self) -> None:
        """Reset zoom to 100%."""
        self.set_zoom_level(ZoomLevel.ZOOM_100)

    # DPI management
    def set_dpi_awareness(self, awareness: DPIAwareness) -> None:
        """Set the DPI awareness mode."""
        self._config.dpi_awareness = awareness

    def detect_system_scale(self) -> float:
        """
        Detect the system UI scale factor.

        Returns the detected scale factor.
        """
        # Platform-specific detection would go here
        # Windows: GetDpiForWindow / GetDpiForMonitor
        # macOS: NSScreen.backingScaleFactor
        # Linux: GTK scale factor or Xft.dpi
        return self._system_scale

    def set_system_scale(self, scale: float) -> None:
        """Set the system scale (for testing or manual override)."""
        self._system_scale = scale

    def detect_system_font_scale(self) -> float:
        """
        Detect the system font scale factor.

        Returns the detected font scale factor.
        """
        # Platform-specific detection would go here
        # Windows: SystemParametersInfo SPI_GETLOGFONT
        # macOS: NSFont.systemFontSize
        # Linux: GTK font settings
        return self._system_font_scale

    def set_system_font_scale(self, scale: float) -> None:
        """Set the system font scale (for testing or manual override)."""
        self._system_font_scale = scale

    def apply_system_scale(self) -> None:
        """Apply detected system scale factors."""
        self.set_ui_scale(self._system_scale, source="system")
        self.set_font_scale(self._system_font_scale, source="system")

    # Monitor management
    def register_monitor(self, monitor: MonitorInfo) -> None:
        """Register a monitor."""
        self._monitors[monitor.monitor_id] = monitor

        if monitor.is_primary and not self._current_monitor:
            self._current_monitor = monitor.monitor_id

    def unregister_monitor(self, monitor_id: str) -> None:
        """Unregister a monitor."""
        self._monitors.pop(monitor_id, None)

        if self._current_monitor == monitor_id:
            # Switch to primary or first available
            self._current_monitor = None
            for mid, m in self._monitors.items():
                if m.is_primary:
                    self._current_monitor = mid
                    break
            if not self._current_monitor and self._monitors:
                self._current_monitor = next(iter(self._monitors))

    def get_monitor(self, monitor_id: str) -> Optional[MonitorInfo]:
        """Get monitor information by ID."""
        return self._monitors.get(monitor_id)

    def get_current_monitor(self) -> Optional[MonitorInfo]:
        """Get the current monitor."""
        if self._current_monitor:
            return self._monitors.get(self._current_monitor)
        return None

    def set_current_monitor(self, monitor_id: str) -> bool:
        """Set the current monitor."""
        if monitor_id in self._monitors:
            self._current_monitor = monitor_id
            return True
        return False

    def get_monitor_scale(self, monitor_id: Optional[str] = None) -> float:
        """Get the scale factor for a monitor."""
        monitor_id = monitor_id or self._current_monitor
        if not monitor_id:
            return 1.0

        monitor = self._monitors.get(monitor_id)
        if not monitor:
            return 1.0

        return monitor.get_scale_factor(self._config.base_dpi)

    # Size conversion utilities
    def scale_value(self, value: float) -> float:
        """Scale a value by the effective scale factor."""
        return value * self.effective_scale

    def unscale_value(self, value: float) -> float:
        """Convert a scaled value back to logical units."""
        scale = self.effective_scale
        if scale == 0:
            return value
        return value / scale

    def scale_size(self, width: float, height: float) -> tuple[float, float]:
        """Scale a size by the effective scale factor."""
        scale = self.effective_scale
        return (width * scale, height * scale)

    def unscale_size(self, width: float, height: float) -> tuple[float, float]:
        """Convert a scaled size back to logical units."""
        scale = self.effective_scale
        if scale == 0:
            return (width, height)
        return (width / scale, height / scale)

    def scale_point(self, x: float, y: float) -> tuple[float, float]:
        """Scale a point by the effective scale factor."""
        scale = self.effective_scale
        return (x * scale, y * scale)

    def unscale_point(self, x: float, y: float) -> tuple[float, float]:
        """Convert a scaled point back to logical units."""
        scale = self.effective_scale
        if scale == 0:
            return (x, y)
        return (x / scale, y / scale)

    def scale_font_size(self, size: float) -> float:
        """Scale a font size by the effective font scale."""
        return size * self.effective_font_scale

    def unscale_font_size(self, size: float) -> float:
        """Convert a scaled font size back to logical units."""
        scale = self.effective_font_scale
        if scale == 0:
            return size
        return size / scale

    # Touch target utilities
    def get_minimum_touch_size(self) -> tuple[float, float]:
        """Get minimum touch target size in scaled units."""
        use_enhanced = self._config.use_enhanced_targets
        if use_enhanced:
            return self.scale_size(
                self._touch_target.enhanced_width,
                self._touch_target.enhanced_height,
            )
        return self.scale_size(
            self._touch_target.min_width,
            self._touch_target.min_height,
        )

    def enforce_touch_target_size(
        self,
        width: float,
        height: float,
    ) -> tuple[float, float]:
        """
        Enforce minimum touch target size if enabled.

        Returns the original size if enforcement is disabled or
        the size meets requirements, otherwise returns the minimum size.
        """
        if not self._config.enforce_touch_targets:
            return (width, height)

        return self._touch_target.get_adjusted_size(
            width,
            height,
            self._config.use_enhanced_targets,
        )

    def check_touch_target(self, width: float, height: float) -> bool:
        """Check if a size meets touch target requirements."""
        if self._config.use_enhanced_targets:
            return self._touch_target.meets_enhanced(width, height)
        return self._touch_target.meets_minimum(width, height)

    # Scale change callbacks
    def add_scale_callback(
        self,
        callback: Callable[[ScaleChangeEvent], None],
    ) -> None:
        """Add a callback for scale changes."""
        self._scale_callbacks.append(callback)

    def remove_scale_callback(
        self,
        callback: Callable[[ScaleChangeEvent], None],
    ) -> None:
        """Remove a scale change callback."""
        if callback in self._scale_callbacks:
            self._scale_callbacks.remove(callback)

    def _fire_scale_change(
        self,
        old_ui: float,
        new_ui: float,
        old_font: Optional[float] = None,
        new_font: Optional[float] = None,
        old_zoom: Optional[ZoomLevel] = None,
        new_zoom: Optional[ZoomLevel] = None,
        source: str = "user",
    ) -> None:
        """Fire scale change event."""
        event = ScaleChangeEvent(
            old_ui_scale=old_ui,
            new_ui_scale=new_ui,
            old_font_scale=old_font or self._config.font_scale,
            new_font_scale=new_font or self._config.font_scale,
            old_zoom=old_zoom or self._config.zoom_level,
            new_zoom=new_zoom or self._config.zoom_level,
            source=source,
        )

        for callback in self._scale_callbacks:
            callback(event)

    # Utility
    def reset(self) -> None:
        """Reset all scale settings to defaults."""
        old_config = self._config
        self._config = ScaleConfig()

        self._fire_scale_change(
            old_config.ui_scale,
            self._config.ui_scale,
            old_config.font_scale,
            self._config.font_scale,
            old_config.zoom_level,
            self._config.zoom_level,
            "reset",
        )

    def clear(self) -> None:
        """Clear all custom data."""
        self._monitors.clear()
        self._current_monitor = None
        self._scale_callbacks.clear()

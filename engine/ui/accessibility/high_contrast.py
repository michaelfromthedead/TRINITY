"""
High contrast mode for UI accessibility.

Provides visual accessibility features:
- Contrast detection and measurement
- High contrast theme support
- Focus indicators with visible outlines
- Icon alternatives for colorblind users

Reference (ARCHITECTURE_UI.md):
- High Contrast: Color accessibility
- Text Scaling: Larger fonts
- Colorblind Modes: Deuteranopia, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


class ContrastLevel(Enum):
    """WCAG contrast level compliance."""
    FAIL = auto()       # Below 3:1
    AA_LARGE = auto()   # 3:1+ (large text)
    AA = auto()         # 4.5:1+ (normal text)
    AAA_LARGE = auto()  # 4.5:1+ (large text enhanced)
    AAA = auto()        # 7:1+ (normal text enhanced)


class ContrastMode(Enum):
    """High contrast mode types."""
    NORMAL = auto()           # Standard colors
    HIGH_CONTRAST_LIGHT = auto()  # High contrast light theme
    HIGH_CONTRAST_DARK = auto()   # High contrast dark theme
    INVERTED = auto()         # Inverted colors

    # Colorblind modes
    PROTANOPIA = auto()       # Red-blind
    DEUTERANOPIA = auto()     # Green-blind
    TRITANOPIA = auto()       # Blue-blind
    ACHROMATOPSIA = auto()    # Complete color blindness (grayscale)


class FocusIndicatorStyle(Enum):
    """Style of focus indicator."""
    OUTLINE = auto()      # Solid outline
    DASHED = auto()       # Dashed outline
    DOTTED = auto()       # Dotted outline
    DOUBLE = auto()       # Double line
    GLOW = auto()         # Glow/shadow effect
    UNDERLINE = auto()    # Underline only
    BACKGROUND = auto()   # Background highlight


# WCAG 2.1 sRGB linearization threshold
# Per WCAG 2.1 specification section 1.4.3
SRGB_LINEARIZATION_THRESHOLD = 0.04045

# WCAG 2.1 contrast ratio thresholds
WCAG_CONTRAST_AA_LARGE = 3.0      # Minimum for large text (18pt+ or 14pt+ bold)
WCAG_CONTRAST_AA_NORMAL = 4.5    # Minimum for normal text
WCAG_CONTRAST_AAA_LARGE = 4.5    # Enhanced for large text
WCAG_CONTRAST_AAA_NORMAL = 7.0   # Enhanced for normal text

# Colorblind simulation transformation matrices
# Based on Brettel, Viénot, and Mollon (1997) simulation model
# Protanopia (red-blind) transformation coefficients
PROTANOPIA_R_FROM_R = 0.567
PROTANOPIA_R_FROM_G = 0.433
PROTANOPIA_G_FROM_R = 0.558
PROTANOPIA_G_FROM_G = 0.442
PROTANOPIA_B_FROM_G = 0.242
PROTANOPIA_B_FROM_B = 0.758

# Deuteranopia (green-blind) transformation coefficients
DEUTERANOPIA_R_FROM_R = 0.625
DEUTERANOPIA_R_FROM_G = 0.375
DEUTERANOPIA_G_FROM_R = 0.700
DEUTERANOPIA_G_FROM_G = 0.300
DEUTERANOPIA_B_FROM_G = 0.300
DEUTERANOPIA_B_FROM_B = 0.700

# Tritanopia (blue-blind) transformation coefficients
TRITANOPIA_R_FROM_R = 0.950
TRITANOPIA_R_FROM_G = 0.050
TRITANOPIA_G_FROM_G = 0.433
TRITANOPIA_G_FROM_B = 0.567
TRITANOPIA_B_FROM_G = 0.475
TRITANOPIA_B_FROM_B = 0.525


@dataclass
class Color:
    """
    Simple color representation.

    Supports RGB and can calculate relative luminance.
    """
    r: int = 0  # 0-255
    g: int = 0  # 0-255
    b: int = 0  # 0-255
    a: int = 255  # 0-255 (alpha)

    @classmethod
    def from_hex(cls, hex_code: str) -> "Color":
        """Create color from hex code (e.g., '#FF0000' or 'FF0000')."""
        hex_code = hex_code.lstrip("#")
        if len(hex_code) == 6:
            return cls(
                r=int(hex_code[0:2], 16),
                g=int(hex_code[2:4], 16),
                b=int(hex_code[4:6], 16),
            )
        elif len(hex_code) == 8:
            return cls(
                r=int(hex_code[0:2], 16),
                g=int(hex_code[2:4], 16),
                b=int(hex_code[4:6], 16),
                a=int(hex_code[6:8], 16),
            )
        raise ValueError(f"Invalid hex color: {hex_code}")

    def to_hex(self, include_alpha: bool = False) -> str:
        """Convert to hex string."""
        if include_alpha:
            return f"#{self.r:02X}{self.g:02X}{self.b:02X}{self.a:02X}"
        return f"#{self.r:02X}{self.g:02X}{self.b:02X}"

    def relative_luminance(self) -> float:
        """
        Calculate relative luminance per WCAG 2.1.

        Returns a value between 0 (black) and 1 (white).
        """
        def linearize(c: int) -> float:
            c_srgb = c / 255.0
            if c_srgb <= SRGB_LINEARIZATION_THRESHOLD:
                return c_srgb / 12.92
            return ((c_srgb + 0.055) / 1.055) ** 2.4

        r_lin = linearize(self.r)
        g_lin = linearize(self.g)
        b_lin = linearize(self.b)

        return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    def blend(self, other: "Color", factor: float) -> "Color":
        """Blend with another color."""
        factor = max(0.0, min(1.0, factor))
        return Color(
            r=int(self.r + (other.r - self.r) * factor),
            g=int(self.g + (other.g - self.g) * factor),
            b=int(self.b + (other.b - self.b) * factor),
            a=int(self.a + (other.a - self.a) * factor),
        )

    def to_grayscale(self) -> "Color":
        """Convert to grayscale."""
        gray = int(0.299 * self.r + 0.587 * self.g + 0.114 * self.b)
        return Color(r=gray, g=gray, b=gray, a=self.a)

    def invert(self) -> "Color":
        """Invert the color."""
        return Color(r=255 - self.r, g=255 - self.g, b=255 - self.b, a=self.a)


@dataclass
class FocusIndicator:
    """
    Configuration for focus indicators.

    Defines how focused elements are visually highlighted.
    """
    style: FocusIndicatorStyle = FocusIndicatorStyle.OUTLINE
    color: Color = field(default_factory=lambda: Color(r=0, g=100, b=255))
    width: float = 2.0  # pixels
    offset: float = 2.0  # pixels from element edge
    radius: float = 0.0  # corner radius

    # Animation
    animated: bool = False
    animation_duration: float = 0.3  # seconds

    # Glow specific
    glow_spread: float = 4.0  # pixels
    glow_blur: float = 4.0  # pixels

    # Visibility
    always_visible: bool = False  # Show even without keyboard focus

    def get_total_size(self) -> float:
        """Get total size including offset and width."""
        return self.offset + self.width


@dataclass
class IconAlternative:
    """
    Alternative representation for icons.

    Provides text or pattern alternatives for users who
    cannot distinguish colors.
    """
    icon_id: str
    text_label: str
    pattern: Optional[str] = None  # e.g., "diagonal-lines", "dots"
    shape: Optional[str] = None  # e.g., "circle", "square", "triangle"

    # High contrast versions
    high_contrast_light: Optional[str] = None  # Asset path
    high_contrast_dark: Optional[str] = None  # Asset path


@dataclass
class HighContrastTheme:
    """
    High contrast color theme.

    Defines colors for various UI elements in high contrast mode.
    """
    name: str
    mode: ContrastMode

    # Background colors
    background: Color = field(default_factory=lambda: Color(r=0, g=0, b=0))
    surface: Color = field(default_factory=lambda: Color(r=20, g=20, b=20))

    # Text colors
    text_primary: Color = field(default_factory=lambda: Color(r=255, g=255, b=255))
    text_secondary: Color = field(default_factory=lambda: Color(r=200, g=200, b=200))
    text_disabled: Color = field(default_factory=lambda: Color(r=128, g=128, b=128))

    # Interactive colors
    primary: Color = field(default_factory=lambda: Color(r=0, g=200, b=255))
    secondary: Color = field(default_factory=lambda: Color(r=255, g=200, b=0))

    # State colors
    focus: Color = field(default_factory=lambda: Color(r=255, g=255, b=0))
    hover: Color = field(default_factory=lambda: Color(r=100, g=100, b=100))
    active: Color = field(default_factory=lambda: Color(r=150, g=150, b=150))

    # Semantic colors
    error: Color = field(default_factory=lambda: Color(r=255, g=100, b=100))
    warning: Color = field(default_factory=lambda: Color(r=255, g=200, b=0))
    success: Color = field(default_factory=lambda: Color(r=100, g=255, b=100))
    info: Color = field(default_factory=lambda: Color(r=100, g=200, b=255))

    # Border
    border: Color = field(default_factory=lambda: Color(r=255, g=255, b=255))
    border_focus: Color = field(default_factory=lambda: Color(r=255, g=255, b=0))


class HighContrastManager:
    """
    Manager for high contrast mode.

    Handles contrast detection, theme switching, and
    color transformations for accessibility.
    """

    __slots__ = (
        "_current_mode",
        "_themes",
        "_focus_indicator",
        "_icon_alternatives",
        "_enabled",
        "_mode_callbacks",
        "_system_preference",
    )

    def __init__(self) -> None:
        self._current_mode = ContrastMode.NORMAL
        self._themes: dict[ContrastMode, HighContrastTheme] = {}
        self._focus_indicator = FocusIndicator()
        self._icon_alternatives: dict[str, IconAlternative] = {}
        self._enabled = True
        self._mode_callbacks: list[Callable[[ContrastMode], None]] = []
        self._system_preference: Optional[ContrastMode] = None

        # Initialize default themes
        self._init_default_themes()

    def _init_default_themes(self) -> None:
        """Initialize default high contrast themes."""
        # High contrast dark
        self._themes[ContrastMode.HIGH_CONTRAST_DARK] = HighContrastTheme(
            name="High Contrast Dark",
            mode=ContrastMode.HIGH_CONTRAST_DARK,
            background=Color(r=0, g=0, b=0),
            surface=Color(r=0, g=0, b=0),
            text_primary=Color(r=255, g=255, b=255),
            text_secondary=Color(r=255, g=255, b=255),
            primary=Color(r=0, g=255, b=255),
            secondary=Color(r=255, g=255, b=0),
            focus=Color(r=255, g=255, b=0),
            border=Color(r=255, g=255, b=255),
        )

        # High contrast light
        self._themes[ContrastMode.HIGH_CONTRAST_LIGHT] = HighContrastTheme(
            name="High Contrast Light",
            mode=ContrastMode.HIGH_CONTRAST_LIGHT,
            background=Color(r=255, g=255, b=255),
            surface=Color(r=255, g=255, b=255),
            text_primary=Color(r=0, g=0, b=0),
            text_secondary=Color(r=0, g=0, b=0),
            primary=Color(r=0, g=0, b=200),
            secondary=Color(r=100, g=0, b=100),
            focus=Color(r=0, g=0, b=0),
            border=Color(r=0, g=0, b=0),
        )

    @property
    def enabled(self) -> bool:
        """Check if high contrast mode is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable high contrast mode."""
        self._enabled = value

    @property
    def current_mode(self) -> ContrastMode:
        """Get the current contrast mode."""
        return self._current_mode

    @property
    def focus_indicator(self) -> FocusIndicator:
        """Get the focus indicator configuration."""
        return self._focus_indicator

    # Mode management
    def set_mode(self, mode: ContrastMode) -> None:
        """Set the current contrast mode."""
        old_mode = self._current_mode
        self._current_mode = mode

        if old_mode != mode:
            for callback in self._mode_callbacks:
                callback(mode)

    def detect_system_preference(self) -> Optional[ContrastMode]:
        """
        Detect system high contrast preference.

        Returns the detected mode or None if not detectable.
        """
        # Platform-specific detection would go here
        # Windows: SystemParametersInfo for SPI_GETHIGHCONTRAST
        # macOS: NSWorkspace.shared.accessibilityDisplayShouldIncreaseContrast
        # Linux: GTK settings
        return self._system_preference

    def set_system_preference(self, mode: Optional[ContrastMode]) -> None:
        """Set the system preference (for testing or manual override)."""
        self._system_preference = mode

    def apply_system_preference(self) -> bool:
        """Apply the detected system preference if available."""
        if self._system_preference:
            self.set_mode(self._system_preference)
            return True
        return False

    # Theme management
    def register_theme(self, theme: HighContrastTheme) -> None:
        """Register a high contrast theme."""
        self._themes[theme.mode] = theme

    def get_theme(self, mode: Optional[ContrastMode] = None) -> Optional[HighContrastTheme]:
        """Get a theme by mode (or current mode if not specified)."""
        mode = mode or self._current_mode
        return self._themes.get(mode)

    def get_current_theme(self) -> Optional[HighContrastTheme]:
        """Get the current theme."""
        return self._themes.get(self._current_mode)

    # Contrast calculation
    @staticmethod
    def calculate_contrast_ratio(foreground: Color, background: Color) -> float:
        """
        Calculate contrast ratio between two colors per WCAG 2.1.

        Returns a ratio from 1:1 to 21:1.
        """
        l1 = foreground.relative_luminance()
        l2 = background.relative_luminance()

        lighter = max(l1, l2)
        darker = min(l1, l2)

        return (lighter + 0.05) / (darker + 0.05)

    @staticmethod
    def get_contrast_level(ratio: float, large_text: bool = False) -> ContrastLevel:
        """
        Determine WCAG contrast level from ratio.

        Args:
            ratio: Contrast ratio
            large_text: True if text is 18pt+ or 14pt+ bold

        Returns:
            The WCAG compliance level.
        """
        if large_text:
            if ratio >= WCAG_CONTRAST_AAA_LARGE:
                return ContrastLevel.AAA_LARGE
            if ratio >= WCAG_CONTRAST_AA_LARGE:
                return ContrastLevel.AA_LARGE
        else:
            if ratio >= WCAG_CONTRAST_AAA_NORMAL:
                return ContrastLevel.AAA
            if ratio >= WCAG_CONTRAST_AA_NORMAL:
                return ContrastLevel.AA
            if ratio >= WCAG_CONTRAST_AA_LARGE:
                return ContrastLevel.AA_LARGE

        return ContrastLevel.FAIL

    def check_contrast(
        self,
        foreground: Color,
        background: Color,
        large_text: bool = False,
    ) -> tuple[float, ContrastLevel]:
        """
        Check contrast between two colors.

        Returns the ratio and WCAG level.
        """
        ratio = self.calculate_contrast_ratio(foreground, background)
        level = self.get_contrast_level(ratio, large_text)
        return (ratio, level)

    def suggest_foreground(
        self,
        background: Color,
        target_ratio: float = 4.5,
        prefer_light: bool = True,
    ) -> Color:
        """
        Suggest a foreground color that meets the target contrast ratio.

        Args:
            background: The background color
            target_ratio: Minimum contrast ratio to achieve
            prefer_light: Prefer lighter foreground if possible

        Returns:
            A suggested foreground color.
        """
        # Try white and black first
        white = Color(r=255, g=255, b=255)
        black = Color(r=0, g=0, b=0)

        white_ratio = self.calculate_contrast_ratio(white, background)
        black_ratio = self.calculate_contrast_ratio(black, background)

        if prefer_light and white_ratio >= target_ratio:
            return white
        if not prefer_light and black_ratio >= target_ratio:
            return black

        # Return the one with better contrast
        if white_ratio > black_ratio:
            return white
        return black

    # Color transformations
    def transform_color(self, color: Color) -> Color:
        """
        Transform a color based on the current mode.

        Applies colorblind simulations or inversions.
        """
        if self._current_mode == ContrastMode.NORMAL:
            return color

        if self._current_mode == ContrastMode.INVERTED:
            return color.invert()

        if self._current_mode == ContrastMode.ACHROMATOPSIA:
            return color.to_grayscale()

        if self._current_mode == ContrastMode.PROTANOPIA:
            return self._simulate_protanopia(color)

        if self._current_mode == ContrastMode.DEUTERANOPIA:
            return self._simulate_deuteranopia(color)

        if self._current_mode == ContrastMode.TRITANOPIA:
            return self._simulate_tritanopia(color)

        return color

    def _simulate_protanopia(self, color: Color) -> Color:
        """Simulate protanopia (red-blind) vision."""
        r = PROTANOPIA_R_FROM_R * color.r + PROTANOPIA_R_FROM_G * color.g
        g = PROTANOPIA_G_FROM_R * color.r + PROTANOPIA_G_FROM_G * color.g
        b = PROTANOPIA_B_FROM_G * color.g + PROTANOPIA_B_FROM_B * color.b
        return Color(
            r=int(min(255, max(0, r))),
            g=int(min(255, max(0, g))),
            b=int(min(255, max(0, b))),
            a=color.a,
        )

    def _simulate_deuteranopia(self, color: Color) -> Color:
        """Simulate deuteranopia (green-blind) vision."""
        r = DEUTERANOPIA_R_FROM_R * color.r + DEUTERANOPIA_R_FROM_G * color.g
        g = DEUTERANOPIA_G_FROM_R * color.r + DEUTERANOPIA_G_FROM_G * color.g
        b = DEUTERANOPIA_B_FROM_G * color.g + DEUTERANOPIA_B_FROM_B * color.b
        return Color(
            r=int(min(255, max(0, r))),
            g=int(min(255, max(0, g))),
            b=int(min(255, max(0, b))),
            a=color.a,
        )

    def _simulate_tritanopia(self, color: Color) -> Color:
        """Simulate tritanopia (blue-blind) vision."""
        r = TRITANOPIA_R_FROM_R * color.r + TRITANOPIA_R_FROM_G * color.g
        g = TRITANOPIA_G_FROM_G * color.g + TRITANOPIA_G_FROM_B * color.b
        b = TRITANOPIA_B_FROM_G * color.g + TRITANOPIA_B_FROM_B * color.b
        return Color(
            r=int(min(255, max(0, r))),
            g=int(min(255, max(0, g))),
            b=int(min(255, max(0, b))),
            a=color.a,
        )

    # Focus indicator
    def set_focus_indicator(self, indicator: FocusIndicator) -> None:
        """Set the focus indicator configuration."""
        self._focus_indicator = indicator

    def update_focus_indicator(self, **kwargs: Any) -> None:
        """Update focus indicator properties."""
        for key, value in kwargs.items():
            if hasattr(self._focus_indicator, key):
                setattr(self._focus_indicator, key, value)

    # Icon alternatives
    def register_icon_alternative(self, alternative: IconAlternative) -> None:
        """Register an icon alternative."""
        self._icon_alternatives[alternative.icon_id] = alternative

    def get_icon_alternative(self, icon_id: str) -> Optional[IconAlternative]:
        """Get an icon alternative by ID."""
        return self._icon_alternatives.get(icon_id)

    def remove_icon_alternative(self, icon_id: str) -> None:
        """Remove an icon alternative."""
        self._icon_alternatives.pop(icon_id, None)

    def get_icon_asset(self, icon_id: str) -> Optional[str]:
        """
        Get the appropriate icon asset for the current mode.

        Returns the path to a high contrast version if available.
        """
        alt = self._icon_alternatives.get(icon_id)
        if not alt:
            return None

        if self._current_mode == ContrastMode.HIGH_CONTRAST_DARK:
            return alt.high_contrast_dark
        if self._current_mode == ContrastMode.HIGH_CONTRAST_LIGHT:
            return alt.high_contrast_light

        return None

    # Mode change callbacks
    def add_mode_callback(self, callback: Callable[[ContrastMode], None]) -> None:
        """Add a callback for mode changes."""
        self._mode_callbacks.append(callback)

    def remove_mode_callback(self, callback: Callable[[ContrastMode], None]) -> None:
        """Remove a mode change callback."""
        if callback in self._mode_callbacks:
            self._mode_callbacks.remove(callback)

    # Utility
    def is_high_contrast(self) -> bool:
        """Check if currently in a high contrast mode."""
        return self._current_mode in (
            ContrastMode.HIGH_CONTRAST_LIGHT,
            ContrastMode.HIGH_CONTRAST_DARK,
        )

    def is_colorblind_mode(self) -> bool:
        """Check if currently in a colorblind simulation mode."""
        return self._current_mode in (
            ContrastMode.PROTANOPIA,
            ContrastMode.DEUTERANOPIA,
            ContrastMode.TRITANOPIA,
            ContrastMode.ACHROMATOPSIA,
        )

    def clear(self) -> None:
        """Clear all custom data."""
        self._icon_alternatives.clear()
        self._mode_callbacks.clear()

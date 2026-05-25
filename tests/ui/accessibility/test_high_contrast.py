"""
Comprehensive tests for High Contrast accessibility support.

Tests cover:
- ContrastLevel enum
- ContrastMode enum
- FocusIndicatorStyle enum
- Color class
- FocusIndicator class
- IconAlternative class
- HighContrastTheme class
- HighContrastManager class
- Contrast ratio calculation
- WCAG compliance levels
- Color transformations
- Colorblind simulations
- Theme management
- Focus indicator configuration
- Icon alternatives
- Mode switching
"""

import pytest
import sys
from pathlib import Path

# Add engine to path
engine_path = Path(__file__).parent.parent.parent.parent.parent / "engine"
sys.path.insert(0, str(engine_path.parent))

from engine.ui.accessibility.high_contrast import (
    ContrastLevel,
    ContrastMode,
    FocusIndicatorStyle,
    Color,
    FocusIndicator,
    IconAlternative,
    HighContrastTheme,
    HighContrastManager,
)


class TestContrastLevel:
    """Test ContrastLevel enum."""

    def test_fail_level(self):
        """Test FAIL level exists."""
        assert ContrastLevel.FAIL is not None

    def test_aa_levels(self):
        """Test AA compliance levels exist."""
        assert ContrastLevel.AA is not None
        assert ContrastLevel.AA_LARGE is not None

    def test_aaa_levels(self):
        """Test AAA compliance levels exist."""
        assert ContrastLevel.AAA is not None
        assert ContrastLevel.AAA_LARGE is not None


class TestContrastMode:
    """Test ContrastMode enum."""

    def test_normal_mode(self):
        """Test NORMAL mode exists."""
        assert ContrastMode.NORMAL is not None

    def test_high_contrast_modes(self):
        """Test high contrast modes exist."""
        assert ContrastMode.HIGH_CONTRAST_LIGHT is not None
        assert ContrastMode.HIGH_CONTRAST_DARK is not None

    def test_inverted_mode(self):
        """Test INVERTED mode exists."""
        assert ContrastMode.INVERTED is not None

    def test_colorblind_modes(self):
        """Test colorblind simulation modes exist."""
        assert ContrastMode.PROTANOPIA is not None
        assert ContrastMode.DEUTERANOPIA is not None
        assert ContrastMode.TRITANOPIA is not None
        assert ContrastMode.ACHROMATOPSIA is not None


class TestFocusIndicatorStyle:
    """Test FocusIndicatorStyle enum."""

    def test_outline_styles(self):
        """Test outline styles exist."""
        assert FocusIndicatorStyle.OUTLINE is not None
        assert FocusIndicatorStyle.DASHED is not None
        assert FocusIndicatorStyle.DOTTED is not None
        assert FocusIndicatorStyle.DOUBLE is not None

    def test_other_styles(self):
        """Test other focus indicator styles exist."""
        assert FocusIndicatorStyle.GLOW is not None
        assert FocusIndicatorStyle.UNDERLINE is not None
        assert FocusIndicatorStyle.BACKGROUND is not None


class TestColor:
    """Test Color class."""

    def test_default_color(self):
        """Test default color is black."""
        color = Color()
        assert color.r == 0
        assert color.g == 0
        assert color.b == 0
        assert color.a == 255

    def test_custom_color(self):
        """Test creating custom color."""
        color = Color(r=255, g=128, b=64)
        assert color.r == 255
        assert color.g == 128
        assert color.b == 64

    def test_from_hex_6_digit(self):
        """Test creating color from 6-digit hex."""
        color = Color.from_hex("#FF8040")
        assert color.r == 255
        assert color.g == 128
        assert color.b == 64

    def test_from_hex_without_hash(self):
        """Test creating color from hex without hash."""
        color = Color.from_hex("FF8040")
        assert color.r == 255

    def test_from_hex_8_digit(self):
        """Test creating color from 8-digit hex with alpha."""
        color = Color.from_hex("#FF804080")
        assert color.r == 255
        assert color.g == 128
        assert color.b == 64
        assert color.a == 128

    def test_from_hex_invalid(self):
        """Test invalid hex raises error."""
        with pytest.raises(ValueError):
            Color.from_hex("#FFF")  # Too short

    def test_to_hex(self):
        """Test converting to hex string."""
        color = Color(r=255, g=128, b=64)
        hex_str = color.to_hex()
        assert hex_str == "#FF8040"

    def test_to_hex_with_alpha(self):
        """Test converting to hex with alpha."""
        color = Color(r=255, g=128, b=64, a=128)
        hex_str = color.to_hex(include_alpha=True)
        assert hex_str == "#FF804080"

    def test_relative_luminance_black(self):
        """Test relative luminance of black."""
        color = Color(r=0, g=0, b=0)
        assert color.relative_luminance() == 0.0

    def test_relative_luminance_white(self):
        """Test relative luminance of white."""
        color = Color(r=255, g=255, b=255)
        assert color.relative_luminance() == 1.0

    def test_relative_luminance_gray(self):
        """Test relative luminance of gray."""
        color = Color(r=128, g=128, b=128)
        lum = color.relative_luminance()
        assert 0.0 < lum < 1.0

    def test_blend(self):
        """Test blending two colors."""
        black = Color(r=0, g=0, b=0)
        white = Color(r=255, g=255, b=255)
        gray = black.blend(white, 0.5)
        assert 120 <= gray.r <= 135  # Approximately 128

    def test_to_grayscale(self):
        """Test converting to grayscale."""
        color = Color(r=255, g=0, b=0)
        gray = color.to_grayscale()
        assert gray.r == gray.g == gray.b

    def test_invert(self):
        """Test inverting color."""
        color = Color(r=255, g=0, b=128)
        inverted = color.invert()
        assert inverted.r == 0
        assert inverted.g == 255
        assert inverted.b == 127


class TestFocusIndicator:
    """Test FocusIndicator class."""

    def test_default_indicator(self):
        """Test default focus indicator."""
        indicator = FocusIndicator()
        assert indicator.style == FocusIndicatorStyle.OUTLINE
        assert indicator.width > 0

    def test_custom_style(self):
        """Test custom focus indicator style."""
        indicator = FocusIndicator(style=FocusIndicatorStyle.GLOW)
        assert indicator.style == FocusIndicatorStyle.GLOW

    def test_custom_color(self):
        """Test custom focus indicator color."""
        indicator = FocusIndicator(color=Color(r=255, g=255, b=0))
        assert indicator.color.r == 255
        assert indicator.color.g == 255

    def test_dimensions(self):
        """Test focus indicator dimensions."""
        indicator = FocusIndicator(width=3.0, offset=4.0)
        assert indicator.width == 3.0
        assert indicator.offset == 4.0

    def test_get_total_size(self):
        """Test calculating total size."""
        indicator = FocusIndicator(width=2.0, offset=2.0)
        assert indicator.get_total_size() == 4.0

    def test_glow_settings(self):
        """Test glow-specific settings."""
        indicator = FocusIndicator(
            style=FocusIndicatorStyle.GLOW,
            glow_spread=6.0,
            glow_blur=8.0,
        )
        assert indicator.glow_spread == 6.0
        assert indicator.glow_blur == 8.0

    def test_animation_settings(self):
        """Test animation settings."""
        indicator = FocusIndicator(
            animated=True,
            animation_duration=0.5,
        )
        assert indicator.animated is True
        assert indicator.animation_duration == 0.5


class TestIconAlternative:
    """Test IconAlternative class."""

    def test_creation(self):
        """Test creating icon alternative."""
        alt = IconAlternative(
            icon_id="warning",
            text_label="Warning",
        )
        assert alt.icon_id == "warning"
        assert alt.text_label == "Warning"

    def test_with_pattern(self):
        """Test icon alternative with pattern."""
        alt = IconAlternative(
            icon_id="error",
            text_label="Error",
            pattern="diagonal-lines",
        )
        assert alt.pattern == "diagonal-lines"

    def test_with_shape(self):
        """Test icon alternative with shape."""
        alt = IconAlternative(
            icon_id="info",
            text_label="Info",
            shape="circle",
        )
        assert alt.shape == "circle"

    def test_high_contrast_assets(self):
        """Test high contrast asset paths."""
        alt = IconAlternative(
            icon_id="star",
            text_label="Star",
            high_contrast_light="icons/hc_light/star.png",
            high_contrast_dark="icons/hc_dark/star.png",
        )
        assert alt.high_contrast_light == "icons/hc_light/star.png"
        assert alt.high_contrast_dark == "icons/hc_dark/star.png"


class TestHighContrastTheme:
    """Test HighContrastTheme class."""

    def test_default_theme(self):
        """Test default high contrast theme."""
        theme = HighContrastTheme(
            name="Test Theme",
            mode=ContrastMode.HIGH_CONTRAST_DARK,
        )
        assert theme.name == "Test Theme"
        assert theme.mode == ContrastMode.HIGH_CONTRAST_DARK

    def test_custom_colors(self):
        """Test custom theme colors."""
        theme = HighContrastTheme(
            name="Custom",
            mode=ContrastMode.HIGH_CONTRAST_DARK,
            background=Color(r=0, g=0, b=0),
            text_primary=Color(r=255, g=255, b=255),
        )
        assert theme.background.r == 0
        assert theme.text_primary.r == 255


class TestHighContrastManager:
    """Test HighContrastManager class."""

    def test_creation(self):
        """Test creating manager."""
        manager = HighContrastManager()
        assert manager.current_mode == ContrastMode.NORMAL
        assert manager.enabled is True

    def test_disable_enable(self):
        """Test disabling and enabling."""
        manager = HighContrastManager()
        manager.enabled = False
        assert manager.enabled is False
        manager.enabled = True
        assert manager.enabled is True


class TestHighContrastManagerModes:
    """Test HighContrastManager mode management."""

    def test_set_mode(self):
        """Test setting contrast mode."""
        manager = HighContrastManager()
        manager.set_mode(ContrastMode.HIGH_CONTRAST_DARK)
        assert manager.current_mode == ContrastMode.HIGH_CONTRAST_DARK

    def test_mode_callback(self):
        """Test mode change callback."""
        manager = HighContrastManager()
        changes = []

        def callback(mode):
            changes.append(mode)

        manager.add_mode_callback(callback)
        manager.set_mode(ContrastMode.HIGH_CONTRAST_LIGHT)
        assert len(changes) == 1
        assert changes[0] == ContrastMode.HIGH_CONTRAST_LIGHT

    def test_remove_mode_callback(self):
        """Test removing mode callback."""
        manager = HighContrastManager()
        changes = []

        def callback(mode):
            changes.append(mode)

        manager.add_mode_callback(callback)
        manager.remove_mode_callback(callback)
        manager.set_mode(ContrastMode.HIGH_CONTRAST_DARK)
        assert len(changes) == 0

    def test_is_high_contrast(self):
        """Test is_high_contrast check."""
        manager = HighContrastManager()
        manager.set_mode(ContrastMode.NORMAL)
        assert manager.is_high_contrast() is False
        manager.set_mode(ContrastMode.HIGH_CONTRAST_DARK)
        assert manager.is_high_contrast() is True

    def test_is_colorblind_mode(self):
        """Test is_colorblind_mode check."""
        manager = HighContrastManager()
        manager.set_mode(ContrastMode.NORMAL)
        assert manager.is_colorblind_mode() is False
        manager.set_mode(ContrastMode.PROTANOPIA)
        assert manager.is_colorblind_mode() is True


class TestHighContrastManagerThemes:
    """Test HighContrastManager theme management."""

    def test_default_themes_registered(self):
        """Test default themes are registered."""
        manager = HighContrastManager()
        dark_theme = manager.get_theme(ContrastMode.HIGH_CONTRAST_DARK)
        light_theme = manager.get_theme(ContrastMode.HIGH_CONTRAST_LIGHT)
        assert dark_theme is not None
        assert light_theme is not None

    def test_register_theme(self):
        """Test registering a custom theme."""
        manager = HighContrastManager()
        theme = HighContrastTheme(
            name="Custom HC",
            mode=ContrastMode.HIGH_CONTRAST_DARK,
            background=Color(r=10, g=10, b=10),
        )
        manager.register_theme(theme)
        retrieved = manager.get_theme(ContrastMode.HIGH_CONTRAST_DARK)
        assert retrieved.name == "Custom HC"

    def test_get_current_theme(self):
        """Test getting current theme."""
        manager = HighContrastManager()
        manager.set_mode(ContrastMode.HIGH_CONTRAST_DARK)
        theme = manager.get_current_theme()
        assert theme is not None


class TestHighContrastManagerContrast:
    """Test HighContrastManager contrast calculations."""

    def test_calculate_contrast_ratio(self):
        """Test calculating contrast ratio."""
        white = Color(r=255, g=255, b=255)
        black = Color(r=0, g=0, b=0)
        ratio = HighContrastManager.calculate_contrast_ratio(white, black)
        assert ratio == 21.0  # Maximum contrast

    def test_calculate_contrast_same_color(self):
        """Test contrast ratio of same color."""
        white = Color(r=255, g=255, b=255)
        ratio = HighContrastManager.calculate_contrast_ratio(white, white)
        assert ratio == 1.0  # Minimum contrast

    def test_get_contrast_level_fail(self):
        """Test FAIL contrast level."""
        ratio = 2.5
        level = HighContrastManager.get_contrast_level(ratio)
        assert level == ContrastLevel.FAIL

    def test_get_contrast_level_aa_large(self):
        """Test AA_LARGE contrast level."""
        ratio = 3.5
        level = HighContrastManager.get_contrast_level(ratio, large_text=True)
        assert level == ContrastLevel.AA_LARGE

    def test_get_contrast_level_aa(self):
        """Test AA contrast level."""
        ratio = 5.0
        level = HighContrastManager.get_contrast_level(ratio)
        assert level == ContrastLevel.AA

    def test_get_contrast_level_aaa(self):
        """Test AAA contrast level."""
        ratio = 8.0
        level = HighContrastManager.get_contrast_level(ratio)
        assert level == ContrastLevel.AAA

    def test_check_contrast(self):
        """Test check_contrast method."""
        manager = HighContrastManager()
        white = Color(r=255, g=255, b=255)
        dark = Color(r=30, g=30, b=30)
        ratio, level = manager.check_contrast(white, dark)
        assert ratio > 10.0
        assert level in (ContrastLevel.AAA, ContrastLevel.AAA_LARGE)

    def test_suggest_foreground_light(self):
        """Test suggesting light foreground."""
        manager = HighContrastManager()
        dark_bg = Color(r=30, g=30, b=30)
        suggested = manager.suggest_foreground(dark_bg, prefer_light=True)
        assert suggested.r == 255  # White

    def test_suggest_foreground_dark(self):
        """Test suggesting dark foreground."""
        manager = HighContrastManager()
        light_bg = Color(r=240, g=240, b=240)
        suggested = manager.suggest_foreground(light_bg, prefer_light=False)
        assert suggested.r == 0  # Black


class TestHighContrastManagerColorTransform:
    """Test HighContrastManager color transformations."""

    def test_transform_normal_mode(self):
        """Test color unchanged in normal mode."""
        manager = HighContrastManager()
        manager.set_mode(ContrastMode.NORMAL)
        color = Color(r=100, g=150, b=200)
        transformed = manager.transform_color(color)
        assert transformed.r == color.r
        assert transformed.g == color.g
        assert transformed.b == color.b

    def test_transform_inverted(self):
        """Test color inversion."""
        manager = HighContrastManager()
        manager.set_mode(ContrastMode.INVERTED)
        color = Color(r=255, g=0, b=128)
        transformed = manager.transform_color(color)
        assert transformed.r == 0
        assert transformed.g == 255
        assert transformed.b == 127

    def test_transform_achromatopsia(self):
        """Test grayscale transformation."""
        manager = HighContrastManager()
        manager.set_mode(ContrastMode.ACHROMATOPSIA)
        color = Color(r=255, g=0, b=0)
        transformed = manager.transform_color(color)
        assert transformed.r == transformed.g == transformed.b

    def test_transform_protanopia(self):
        """Test protanopia simulation."""
        manager = HighContrastManager()
        manager.set_mode(ContrastMode.PROTANOPIA)
        red = Color(r=255, g=0, b=0)
        transformed = manager.transform_color(red)
        # Red should shift to more greenish
        assert transformed.r != 255 or transformed.g != 0

    def test_transform_deuteranopia(self):
        """Test deuteranopia simulation."""
        manager = HighContrastManager()
        manager.set_mode(ContrastMode.DEUTERANOPIA)
        green = Color(r=0, g=255, b=0)
        transformed = manager.transform_color(green)
        # Green should shift
        assert transformed.g != 255

    def test_transform_tritanopia(self):
        """Test tritanopia simulation."""
        manager = HighContrastManager()
        manager.set_mode(ContrastMode.TRITANOPIA)
        blue = Color(r=0, g=0, b=255)
        transformed = manager.transform_color(blue)
        # Blue should shift
        assert transformed.b != 255


class TestHighContrastManagerFocusIndicator:
    """Test HighContrastManager focus indicator management."""

    def test_default_focus_indicator(self):
        """Test default focus indicator exists."""
        manager = HighContrastManager()
        assert manager.focus_indicator is not None

    def test_set_focus_indicator(self):
        """Test setting custom focus indicator."""
        manager = HighContrastManager()
        indicator = FocusIndicator(
            style=FocusIndicatorStyle.GLOW,
            width=4.0,
        )
        manager.set_focus_indicator(indicator)
        assert manager.focus_indicator.style == FocusIndicatorStyle.GLOW

    def test_update_focus_indicator(self):
        """Test updating focus indicator properties."""
        manager = HighContrastManager()
        manager.update_focus_indicator(width=5.0, offset=3.0)
        assert manager.focus_indicator.width == 5.0
        assert manager.focus_indicator.offset == 3.0


class TestHighContrastManagerIconAlternatives:
    """Test HighContrastManager icon alternative management."""

    def test_register_icon_alternative(self):
        """Test registering icon alternative."""
        manager = HighContrastManager()
        alt = IconAlternative(
            icon_id="warning",
            text_label="Warning",
        )
        manager.register_icon_alternative(alt)
        retrieved = manager.get_icon_alternative("warning")
        assert retrieved is alt

    def test_get_icon_alternative_not_found(self):
        """Test getting nonexistent alternative."""
        manager = HighContrastManager()
        retrieved = manager.get_icon_alternative("nonexistent")
        assert retrieved is None

    def test_remove_icon_alternative(self):
        """Test removing icon alternative."""
        manager = HighContrastManager()
        alt = IconAlternative(
            icon_id="error",
            text_label="Error",
        )
        manager.register_icon_alternative(alt)
        manager.remove_icon_alternative("error")
        retrieved = manager.get_icon_alternative("error")
        assert retrieved is None

    def test_get_icon_asset_high_contrast_dark(self):
        """Test getting HC dark icon asset."""
        manager = HighContrastManager()
        alt = IconAlternative(
            icon_id="star",
            text_label="Star",
            high_contrast_dark="icons/hc_dark/star.png",
        )
        manager.register_icon_alternative(alt)
        manager.set_mode(ContrastMode.HIGH_CONTRAST_DARK)
        asset = manager.get_icon_asset("star")
        assert asset == "icons/hc_dark/star.png"

    def test_get_icon_asset_normal_mode(self):
        """Test getting icon asset in normal mode."""
        manager = HighContrastManager()
        alt = IconAlternative(
            icon_id="star",
            text_label="Star",
            high_contrast_dark="icons/hc_dark/star.png",
        )
        manager.register_icon_alternative(alt)
        manager.set_mode(ContrastMode.NORMAL)
        asset = manager.get_icon_asset("star")
        assert asset is None  # No special asset in normal mode


class TestHighContrastManagerSystemPreference:
    """Test HighContrastManager system preference detection."""

    def test_detect_system_preference(self):
        """Test detect_system_preference returns value."""
        manager = HighContrastManager()
        result = manager.detect_system_preference()
        # Result depends on system, just verify it returns something
        assert result is None or isinstance(result, ContrastMode)

    def test_set_system_preference(self):
        """Test setting system preference manually."""
        manager = HighContrastManager()
        manager.set_system_preference(ContrastMode.HIGH_CONTRAST_DARK)
        pref = manager.detect_system_preference()
        assert pref == ContrastMode.HIGH_CONTRAST_DARK

    def test_apply_system_preference(self):
        """Test applying system preference."""
        manager = HighContrastManager()
        manager.set_system_preference(ContrastMode.HIGH_CONTRAST_LIGHT)
        result = manager.apply_system_preference()
        assert result is True
        assert manager.current_mode == ContrastMode.HIGH_CONTRAST_LIGHT


class TestHighContrastManagerClear:
    """Test HighContrastManager clear method."""

    def test_clear(self):
        """Test clearing custom data."""
        manager = HighContrastManager()
        manager.register_icon_alternative(IconAlternative("test", "Test"))
        manager.add_mode_callback(lambda m: None)
        manager.clear()
        assert manager.get_icon_alternative("test") is None

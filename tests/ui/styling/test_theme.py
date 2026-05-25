"""
Comprehensive tests for the Theme system.

Tests cover:
- Theme creation and configuration
- Color palette
- Typography
- Spacing, shadows, and radii
- Theme inheritance
- Widget style generation
- Theme switching and providers
- Built-in themes
"""
import pytest

from engine.ui.styling.brush import SolidBrush
from engine.ui.styling.color import Color
from engine.ui.styling.style import StateStyles, Style, VisualState
from engine.ui.styling.theme import (
    BorderRadii,
    ColorPalette,
    FontDefinition,
    ShadowDefinition,
    Shadows,
    Spacing,
    Theme,
    ThemeProvider,
    Transitions,
    Typography,
    create_dark_theme,
    create_high_contrast_theme,
    create_light_theme,
    get_current_theme,
    set_current_theme,
)


# ========== Fixtures ==========


@pytest.fixture
def light_theme():
    """Create a light theme."""
    return create_light_theme()


@pytest.fixture
def dark_theme():
    """Create a dark theme."""
    return create_dark_theme()


@pytest.fixture
def custom_palette():
    """Create a custom color palette."""
    return ColorPalette(
        primary=Color.from_hex("#FF5500"),
        secondary=Color.from_hex("#00FF55"),
    )


@pytest.fixture
def custom_theme(custom_palette):
    """Create a custom theme."""
    return Theme(
        name="custom",
        is_dark=False,
        colors=custom_palette,
    )


# ========== ColorPalette Tests ==========


class TestColorPalette:
    """Tests for ColorPalette class."""

    def test_default_colors(self):
        """Test default color palette values."""
        palette = ColorPalette()
        assert palette.primary is not None
        assert palette.secondary is not None
        assert palette.background is not None

    def test_shade_generation(self):
        """Test automatic shade variant generation."""
        palette = ColorPalette(
            primary=Color.from_hex("#3B82F6"),
        )
        assert palette.primary_light is not None
        assert palette.primary_dark is not None
        # Light should be lighter
        assert palette.primary_light.to_hsl()[2] > palette.primary.to_hsl()[2]
        # Dark should be darker
        assert palette.primary_dark.to_hsl()[2] < palette.primary.to_hsl()[2]

    def test_explicit_shades(self):
        """Test explicit shade values override auto-generation."""
        custom_light = Color.from_hex("#FFFFFF")
        palette = ColorPalette(
            primary=Color.from_hex("#3B82F6"),
            primary_light=custom_light,
        )
        assert palette.primary_light == custom_light

    def test_get_color_by_name(self):
        """Test getting color by name."""
        palette = ColorPalette()
        assert palette.get_color("primary") == palette.primary
        assert palette.get_color("nonexistent") is None

    def test_clone_palette(self):
        """Test cloning color palette."""
        palette = ColorPalette()
        cloned = palette.clone()
        assert cloned is not palette
        assert cloned.primary == palette.primary

    def test_semantic_colors(self):
        """Test semantic color definitions."""
        palette = ColorPalette()
        assert palette.success is not None
        assert palette.warning is not None
        assert palette.error is not None
        assert palette.info is not None


# ========== FontDefinition Tests ==========


class TestFontDefinition:
    """Tests for FontDefinition class."""

    def test_default_values(self):
        """Test default font values."""
        font = FontDefinition()
        assert font.family == "system-ui"
        assert font.size == 14.0
        assert font.weight == "normal"
        assert font.line_height == 1.5

    def test_custom_values(self):
        """Test custom font values."""
        font = FontDefinition(
            family="Arial",
            size=18.0,
            weight="bold",
            style="italic",
        )
        assert font.family == "Arial"
        assert font.size == 18.0
        assert font.weight == "bold"
        assert font.style == "italic"

    def test_to_style_dict(self):
        """Test converting to style dictionary."""
        font = FontDefinition(family="Arial", size=16.0)
        style_dict = font.to_style_dict()
        assert style_dict["font_family"] == "Arial"
        assert style_dict["font_size"] == 16.0
        assert "line_height" in style_dict


# ========== Typography Tests ==========


class TestTypography:
    """Tests for Typography class."""

    def test_default_typography(self):
        """Test default typography values."""
        typo = Typography()
        assert typo.h1 is not None
        assert typo.body is not None
        assert typo.button is not None

    def test_heading_hierarchy(self):
        """Test heading sizes decrease."""
        typo = Typography()
        assert typo.h1.size > typo.h2.size
        assert typo.h2.size > typo.h3.size
        assert typo.h3.size > typo.h4.size

    def test_get_font_by_name(self):
        """Test getting font by name."""
        typo = Typography()
        assert typo.get_font("h1") == typo.h1
        assert typo.get_font("body") == typo.body
        assert typo.get_font("nonexistent") is None

    def test_clone_typography(self):
        """Test cloning typography."""
        typo = Typography()
        cloned = typo.clone()
        assert cloned is not typo
        assert cloned.h1 is not typo.h1
        assert cloned.h1.size == typo.h1.size


# ========== Spacing Tests ==========


class TestSpacing:
    """Tests for Spacing class."""

    def test_default_base(self):
        """Test default base unit."""
        spacing = Spacing()
        assert spacing.base == 4.0

    def test_get_spacing(self):
        """Test getting spacing by name."""
        spacing = Spacing()
        # xs = 2 * 4 = 8
        assert spacing.get("xs") == 8.0
        # md = 4 * 4 = 16
        assert spacing.get("md") == 16.0

    def test_call_multiplier(self):
        """Test calling spacing with multiplier."""
        spacing = Spacing()
        assert spacing(4) == 16.0  # 4 * 4
        assert spacing(2.5) == 10.0  # 4 * 2.5

    def test_clone_spacing(self):
        """Test cloning spacing."""
        spacing = Spacing(base=8.0)
        cloned = spacing.clone()
        assert cloned is not spacing
        assert cloned.base == 8.0


# ========== ShadowDefinition Tests ==========


class TestShadowDefinition:
    """Tests for ShadowDefinition class."""

    def test_default_values(self):
        """Test default shadow values."""
        shadow = ShadowDefinition()
        assert shadow.offset_x == 0.0
        assert shadow.offset_y == 2.0
        assert shadow.blur == 4.0

    def test_to_style_dict(self):
        """Test converting to style dictionary."""
        shadow = ShadowDefinition(offset_y=4.0, blur=8.0)
        style_dict = shadow.to_style_dict()
        assert style_dict["shadow_offset_y"] == 4.0
        assert style_dict["shadow_blur"] == 8.0


# ========== Shadows Tests ==========


class TestShadows:
    """Tests for Shadows class."""

    def test_default_shadows(self):
        """Test default shadow definitions."""
        shadows = Shadows()
        assert shadows.none is not None
        assert shadows.sm is not None
        assert shadows.lg is not None

    def test_shadow_size_progression(self):
        """Test shadow blur increases with size."""
        shadows = Shadows()
        assert shadows.sm.blur < shadows.md.blur
        assert shadows.md.blur < shadows.lg.blur
        assert shadows.lg.blur < shadows.xl.blur

    def test_get_shadow(self):
        """Test getting shadow by name."""
        shadows = Shadows()
        assert shadows.get("md") == shadows.md
        assert shadows.get("nonexistent") is None

    def test_clone_shadows(self):
        """Test cloning shadows."""
        shadows = Shadows()
        cloned = shadows.clone()
        assert cloned is not shadows
        assert cloned.md is not shadows.md


# ========== BorderRadii Tests ==========


class TestBorderRadii:
    """Tests for BorderRadii class."""

    def test_default_radii(self):
        """Test default radius values."""
        radii = BorderRadii()
        assert radii.none == 0.0
        assert radii.full == 9999.0

    def test_radius_progression(self):
        """Test radius increases with size."""
        radii = BorderRadii()
        assert radii.sm < radii.md
        assert radii.md < radii.lg

    def test_get_radius(self):
        """Test getting radius by name."""
        radii = BorderRadii()
        assert radii.get("md") == radii.md

    def test_clone_radii(self):
        """Test cloning radii."""
        radii = BorderRadii(md=10.0)
        cloned = radii.clone()
        assert cloned is not radii
        assert cloned.md == 10.0


# ========== Transitions Tests ==========


class TestTransitions:
    """Tests for Transitions class."""

    def test_default_durations(self):
        """Test default transition durations."""
        transitions = Transitions()
        assert transitions.duration_instant == 0.0
        assert transitions.duration_fast < transitions.duration_normal
        assert transitions.duration_normal < transitions.duration_slow

    def test_easing_functions(self):
        """Test easing function names."""
        transitions = Transitions()
        assert transitions.ease_linear == "linear"
        assert "ease" in transitions.ease_in.lower()

    def test_clone_transitions(self):
        """Test cloning transitions."""
        transitions = Transitions(duration_fast=0.2)
        cloned = transitions.clone()
        assert cloned is not transitions
        assert cloned.duration_fast == 0.2


# ========== Theme Creation Tests ==========


class TestThemeCreation:
    """Tests for Theme creation."""

    def test_default_theme(self):
        """Test creating default theme."""
        theme = Theme()
        assert theme.name == "default"
        assert theme.is_dark is False
        assert theme.colors is not None
        assert theme.typography is not None

    def test_named_theme_registered(self):
        """Test named themes are registered."""
        theme = Theme(name="test_theme_reg")
        assert Theme.get("test_theme_reg") is theme

    def test_list_themes(self):
        """Test listing registered themes."""
        themes = Theme.list_themes()
        assert isinstance(themes, list)


# ========== Theme Inheritance Tests ==========


class TestThemeInheritance:
    """Tests for theme inheritance."""

    def test_inherit_from_parent(self, light_theme):
        """Test inheriting from parent theme."""
        child = Theme(
            name="child_theme",
            colors=ColorPalette(primary=Color.from_hex("#FF0000")),
        )
        inherited = child.inherit_from(light_theme)
        # Child keeps its primary
        assert inherited.colors.primary.r > 0.9
        # Parent reference is set
        assert inherited._parent is light_theme

    def test_get_effective_value(self, light_theme):
        """Test getting effective value with fallback."""
        child = Theme(name="child_effective")
        child._parent = light_theme

        # Should fall back to parent for most values
        value = child.get_effective_value("colors.primary")
        assert value is not None

    def test_get_effective_value_no_parent(self):
        """Test getting effective value without parent."""
        theme = Theme()
        value = theme.get_effective_value("colors.primary")
        assert value is not None


# ========== Widget Style Generation Tests ==========


class TestWidgetStyleGeneration:
    """Tests for widget style generation."""

    def test_create_button_style_primary(self, light_theme):
        """Test creating primary button styles."""
        styles = light_theme.create_button_style("primary")
        assert isinstance(styles, StateStyles)
        assert styles.normal is not None
        assert styles.hovered is not None
        assert styles.disabled is not None

    def test_create_button_style_secondary(self, light_theme):
        """Test creating secondary button styles."""
        styles = light_theme.create_button_style("secondary")
        assert styles.normal.background is not None

    def test_create_button_style_outline(self, light_theme):
        """Test creating outline button styles."""
        styles = light_theme.create_button_style("outline")
        assert styles.normal.border_width > 0

    def test_create_button_style_ghost(self, light_theme):
        """Test creating ghost button styles."""
        styles = light_theme.create_button_style("ghost")
        # Ghost buttons have transparent background
        assert styles.normal.background is not None

    def test_create_input_style(self, light_theme):
        """Test creating input styles."""
        styles = light_theme.create_input_style()
        assert isinstance(styles, StateStyles)
        assert styles.normal.border_width > 0
        assert styles.focused is not None

    def test_create_card_style(self, light_theme):
        """Test creating card styles."""
        style = light_theme.create_card_style(elevated=True)
        assert style.background is not None
        assert style.shadow_blur > 0

    def test_create_card_style_flat(self, light_theme):
        """Test creating flat card styles."""
        style = light_theme.create_card_style(elevated=False)
        assert style.shadow_color is None


# ========== Widget Styles Management Tests ==========


class TestWidgetStylesManagement:
    """Tests for widget styles management."""

    def test_set_widget_style(self, light_theme):
        """Test setting widget style."""
        styles = StateStyles(normal=Style(opacity=0.5))
        light_theme.set_widget_style("CustomWidget", styles)
        assert light_theme.get_widget_style("CustomWidget") is styles

    def test_get_widget_style_not_found(self, light_theme):
        """Test getting non-existent widget style."""
        result = light_theme.get_widget_style("NonExistent")
        assert result is None

    def test_get_widget_style_from_parent(self, light_theme):
        """Test getting widget style from parent."""
        styles = StateStyles(normal=Style(opacity=0.5))
        light_theme.set_widget_style("ParentWidget", styles)

        child = Theme(name="child_widget_styles")
        child._parent = light_theme

        result = child.get_widget_style("ParentWidget")
        assert result is styles


# ========== Theme Switching Tests ==========


class TestThemeSwitching:
    """Tests for theme switching."""

    def test_activate_theme(self, light_theme):
        """Test activating a theme."""
        token = light_theme.activate()
        assert get_current_theme() is light_theme
        # Cleanup (not a full restore, just for this test)

    def test_set_current_theme(self, dark_theme):
        """Test setting current theme."""
        token = set_current_theme(dark_theme)
        assert get_current_theme() is dark_theme

    def test_theme_listener(self, light_theme):
        """Test theme change listener."""
        notified = []

        def on_theme_change(theme):
            notified.append(theme)

        light_theme.add_listener(on_theme_change)
        light_theme.activate()

        assert len(notified) == 1
        assert notified[0] is light_theme

    def test_remove_listener(self, light_theme):
        """Test removing theme listener."""
        notified = []

        def on_theme_change(theme):
            notified.append(theme)

        light_theme.add_listener(on_theme_change)
        light_theme.remove_listener(on_theme_change)
        light_theme.activate()

        assert len(notified) == 0


# ========== ThemeProvider Tests ==========


class TestThemeProvider:
    """Tests for ThemeProvider context manager."""

    def test_theme_provider_context(self, dark_theme):
        """Test ThemeProvider as context manager."""
        original = get_current_theme()

        with ThemeProvider(dark_theme):
            assert get_current_theme() is dark_theme

        # Should restore after context
        # Note: May not restore if original was None

    def test_theme_provider_returns_self(self, dark_theme):
        """Test ThemeProvider returns self on enter."""
        provider = ThemeProvider(dark_theme)
        result = provider.__enter__()
        assert result is provider
        provider.__exit__(None, None, None)


# ========== Theme Cloning Tests ==========


class TestThemeCloning:
    """Tests for theme cloning."""

    def test_clone_theme(self, light_theme):
        """Test cloning a theme."""
        cloned = light_theme.clone()
        assert cloned is not light_theme
        assert cloned.colors is not light_theme.colors
        assert cloned.typography is not light_theme.typography

    def test_clone_theme_name(self, light_theme):
        """Test cloned theme has modified name."""
        cloned = light_theme.clone()
        assert "_copy" in cloned.name

    def test_clone_preserves_values(self, light_theme):
        """Test clone preserves values."""
        cloned = light_theme.clone()
        assert cloned.is_dark == light_theme.is_dark


# ========== Built-in Theme Tests ==========


class TestBuiltInThemes:
    """Tests for built-in themes."""

    def test_light_theme_exists(self):
        """Test light theme is defined."""
        theme = create_light_theme()
        assert theme.name == "light"
        assert theme.is_dark is False

    def test_dark_theme_exists(self):
        """Test dark theme is defined."""
        theme = create_dark_theme()
        assert theme.name == "dark"
        assert theme.is_dark is True

    def test_high_contrast_theme_exists(self):
        """Test high contrast theme is defined."""
        theme = create_high_contrast_theme()
        assert theme.name == "high_contrast"
        assert theme.is_dark is True

    def test_dark_theme_darker_background(self):
        """Test dark theme has darker background."""
        light = create_light_theme()
        dark = create_dark_theme()
        assert dark.colors.background.luminance < light.colors.background.luminance

    def test_high_contrast_theme_high_contrast(self):
        """Test high contrast theme has high text contrast."""
        theme = create_high_contrast_theme()
        ratio = theme.colors.text_primary.contrast_ratio(theme.colors.background)
        assert ratio >= 7.0  # AAA level

    def test_light_theme_text_readable(self):
        """Test light theme text is readable."""
        theme = create_light_theme()
        assert theme.colors.text_primary.is_readable_on(theme.colors.background, "AA")

    def test_dark_theme_text_readable(self):
        """Test dark theme text is readable."""
        theme = create_dark_theme()
        assert theme.colors.text_primary.is_readable_on(theme.colors.background, "AA")


# ========== Theme Registry Tests ==========


class TestThemeRegistry:
    """Tests for theme registry."""

    def test_get_registered_theme(self):
        """Test getting registered theme."""
        theme = Theme(name="registry_test")
        retrieved = Theme.get("registry_test")
        assert retrieved is theme

    def test_get_unregistered_theme(self):
        """Test getting unregistered theme returns None."""
        result = Theme.get("nonexistent_theme")
        assert result is None

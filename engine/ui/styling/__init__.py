"""
UI Styling System.

Provides comprehensive styling capabilities including:
- Color utilities with RGBA, HSL, HSV representations
- Brush types for fills (solid, gradient, image, nine-slice)
- Style class with visual states and inheritance
- Theme system with palettes, typography, and spacing
"""

from engine.ui.styling.color import (
    # Core Color class
    Color,
    BlendMode,
    # Palette generation functions
    generate_palette,
    generate_complementary,
    generate_triadic,
    generate_analogous,
    generate_split_complementary,
    generate_tetradic,
    interpolate_colors,
)

from engine.ui.styling.brush import (
    # Brush base and types
    Brush,
    SolidBrush,
    GradientBrush,
    ImageBrush,
    NineSliceBrush,
    # Gradient types
    GradientType,
    GradientStop,
    TileMode,
    ImageFit,
    # Utility functions
    create_brush,
    transparent_brush,
    white_brush,
    black_brush,
)

from engine.ui.styling.style import (
    # Visual states
    VisualState,
    # Style property descriptor
    StylePropertyDescriptor,
    style_property,
    # Style classes
    Style,
    StateStyles,
    # Selectors
    SelectorType,
    StyleSelector,
    StyleRule,
    Stylesheet,
    # Builder
    StyleBuilder,
)

from engine.ui.styling.theme import (
    # Theme context
    get_current_theme,
    set_current_theme,
    # Design tokens
    ColorPalette,
    Typography,
    FontDefinition,
    Spacing,
    Shadows,
    ShadowDefinition,
    BorderRadii,
    Transitions,
    # Theme class
    Theme,
    ThemeProvider,
    # Built-in themes
    LIGHT_THEME,
    DARK_THEME,
    HIGH_CONTRAST_THEME,
    create_light_theme,
    create_dark_theme,
    create_high_contrast_theme,
)


__all__ = [
    # === Color ===
    "Color",
    "BlendMode",
    "generate_palette",
    "generate_complementary",
    "generate_triadic",
    "generate_analogous",
    "generate_split_complementary",
    "generate_tetradic",
    "interpolate_colors",

    # === Brush ===
    "Brush",
    "SolidBrush",
    "GradientBrush",
    "ImageBrush",
    "NineSliceBrush",
    "GradientType",
    "GradientStop",
    "TileMode",
    "ImageFit",
    "create_brush",
    "transparent_brush",
    "white_brush",
    "black_brush",

    # === Style ===
    "VisualState",
    "StylePropertyDescriptor",
    "style_property",
    "Style",
    "StateStyles",
    "SelectorType",
    "StyleSelector",
    "StyleRule",
    "Stylesheet",
    "StyleBuilder",

    # === Theme ===
    "get_current_theme",
    "set_current_theme",
    "ColorPalette",
    "Typography",
    "FontDefinition",
    "Spacing",
    "Shadows",
    "ShadowDefinition",
    "BorderRadii",
    "Transitions",
    "Theme",
    "ThemeProvider",
    "LIGHT_THEME",
    "DARK_THEME",
    "HIGH_CONTRAST_THEME",
    "create_light_theme",
    "create_dark_theme",
    "create_high_contrast_theme",
]

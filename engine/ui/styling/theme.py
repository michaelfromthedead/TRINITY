"""
Theme system for UI styling.

Provides Theme class with color palettes, spacing, typography,
built-in themes, runtime switching, and theme inheritance.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union

from engine.ui.styling.color import Color
from engine.ui.styling.brush import SolidBrush
from engine.ui.styling.style import Style, StateStyles


# ========== Theme Context ==========

# Context variable for current theme (thread-safe)
_current_theme: contextvars.ContextVar["Theme"] = contextvars.ContextVar(
    "current_theme",
    default=None,  # type: ignore
)


def get_current_theme() -> Optional["Theme"]:
    """Get the currently active theme."""
    return _current_theme.get()


def set_current_theme(theme: "Theme") -> contextvars.Token:
    """
    Set the current theme.

    Args:
        theme: Theme to set as current

    Returns:
        Token that can be used to restore previous theme
    """
    return _current_theme.set(theme)


# ========== Color Palette ==========

@dataclass
class ColorPalette:
    """
    Named color collection for a theme.

    Provides semantic color names (primary, secondary, etc.)
    with shade variations.
    """

    # Primary colors
    primary: Color = field(default_factory=lambda: Color.from_hex("#3B82F6"))
    primary_light: Optional[Color] = None
    primary_dark: Optional[Color] = None

    # Secondary colors
    secondary: Color = field(default_factory=lambda: Color.from_hex("#8B5CF6"))
    secondary_light: Optional[Color] = None
    secondary_dark: Optional[Color] = None

    # Accent colors
    accent: Color = field(default_factory=lambda: Color.from_hex("#F59E0B"))
    accent_light: Optional[Color] = None
    accent_dark: Optional[Color] = None

    # Semantic colors
    success: Color = field(default_factory=lambda: Color.from_hex("#10B981"))
    warning: Color = field(default_factory=lambda: Color.from_hex("#F59E0B"))
    error: Color = field(default_factory=lambda: Color.from_hex("#EF4444"))
    info: Color = field(default_factory=lambda: Color.from_hex("#3B82F6"))

    # Background colors
    background: Color = field(default_factory=lambda: Color.from_hex("#FFFFFF"))
    background_secondary: Color = field(default_factory=lambda: Color.from_hex("#F3F4F6"))
    background_tertiary: Color = field(default_factory=lambda: Color.from_hex("#E5E7EB"))

    # Surface colors (for cards, dialogs, etc.)
    surface: Color = field(default_factory=lambda: Color.from_hex("#FFFFFF"))
    surface_variant: Color = field(default_factory=lambda: Color.from_hex("#F9FAFB"))

    # Text colors
    text_primary: Color = field(default_factory=lambda: Color.from_hex("#111827"))
    text_secondary: Color = field(default_factory=lambda: Color.from_hex("#6B7280"))
    text_disabled: Color = field(default_factory=lambda: Color.from_hex("#9CA3AF"))
    text_inverse: Color = field(default_factory=lambda: Color.from_hex("#FFFFFF"))

    # Border colors
    border: Color = field(default_factory=lambda: Color.from_hex("#E5E7EB"))
    border_focus: Color = field(default_factory=lambda: Color.from_hex("#3B82F6"))

    # Overlay colors
    overlay: Color = field(default_factory=lambda: Color(0, 0, 0, 0.5))
    scrim: Color = field(default_factory=lambda: Color(0, 0, 0, 0.32))

    def __post_init__(self) -> None:
        """Generate shade variants if not provided."""
        if self.primary_light is None:
            self.primary_light = self.primary.lighten(0.2)
        if self.primary_dark is None:
            self.primary_dark = self.primary.darken(0.2)

        if self.secondary_light is None:
            self.secondary_light = self.secondary.lighten(0.2)
        if self.secondary_dark is None:
            self.secondary_dark = self.secondary.darken(0.2)

        if self.accent_light is None:
            self.accent_light = self.accent.lighten(0.2)
        if self.accent_dark is None:
            self.accent_dark = self.accent.darken(0.2)

    def get_color(self, name: str) -> Optional[Color]:
        """Get a color by name."""
        return getattr(self, name, None)

    def clone(self) -> "ColorPalette":
        """Create a copy of this palette."""
        return ColorPalette(
            primary=self.primary,
            primary_light=self.primary_light,
            primary_dark=self.primary_dark,
            secondary=self.secondary,
            secondary_light=self.secondary_light,
            secondary_dark=self.secondary_dark,
            accent=self.accent,
            accent_light=self.accent_light,
            accent_dark=self.accent_dark,
            success=self.success,
            warning=self.warning,
            error=self.error,
            info=self.info,
            background=self.background,
            background_secondary=self.background_secondary,
            background_tertiary=self.background_tertiary,
            surface=self.surface,
            surface_variant=self.surface_variant,
            text_primary=self.text_primary,
            text_secondary=self.text_secondary,
            text_disabled=self.text_disabled,
            text_inverse=self.text_inverse,
            border=self.border,
            border_focus=self.border_focus,
            overlay=self.overlay,
            scrim=self.scrim,
        )


# ========== Typography ==========

@dataclass
class FontDefinition:
    """Definition of a font style."""

    family: str = "system-ui"
    size: float = 14.0
    weight: str = "normal"
    style: str = "normal"
    line_height: float = 1.5
    letter_spacing: float = 0.0

    def to_style_dict(self) -> Dict[str, Any]:
        """Convert to style dictionary."""
        return {
            "font_family": self.family,
            "font_size": self.size,
            "font_weight": self.weight,
            "font_style": self.style,
            "line_height": self.line_height,
            "letter_spacing": self.letter_spacing,
        }


@dataclass
class Typography:
    """
    Typography definitions for a theme.

    Provides semantic font styles (headings, body, etc.).
    """

    # Font families
    font_family_primary: str = "system-ui, -apple-system, sans-serif"
    font_family_secondary: str = "Georgia, serif"
    font_family_monospace: str = "ui-monospace, monospace"

    # Headings
    h1: FontDefinition = field(default_factory=lambda: FontDefinition(
        size=32.0, weight="bold", line_height=1.2
    ))
    h2: FontDefinition = field(default_factory=lambda: FontDefinition(
        size=24.0, weight="bold", line_height=1.25
    ))
    h3: FontDefinition = field(default_factory=lambda: FontDefinition(
        size=20.0, weight="semibold", line_height=1.3
    ))
    h4: FontDefinition = field(default_factory=lambda: FontDefinition(
        size=18.0, weight="semibold", line_height=1.35
    ))
    h5: FontDefinition = field(default_factory=lambda: FontDefinition(
        size=16.0, weight="medium", line_height=1.4
    ))
    h6: FontDefinition = field(default_factory=lambda: FontDefinition(
        size=14.0, weight="medium", line_height=1.4
    ))

    # Body text
    body_large: FontDefinition = field(default_factory=lambda: FontDefinition(
        size=18.0, line_height=1.6
    ))
    body: FontDefinition = field(default_factory=lambda: FontDefinition(
        size=16.0, line_height=1.5
    ))
    body_small: FontDefinition = field(default_factory=lambda: FontDefinition(
        size=14.0, line_height=1.5
    ))

    # UI text
    label: FontDefinition = field(default_factory=lambda: FontDefinition(
        size=14.0, weight="medium"
    ))
    button: FontDefinition = field(default_factory=lambda: FontDefinition(
        size=14.0, weight="medium", letter_spacing=0.5
    ))
    caption: FontDefinition = field(default_factory=lambda: FontDefinition(
        size=12.0, line_height=1.4
    ))
    overline: FontDefinition = field(default_factory=lambda: FontDefinition(
        size=10.0, weight="medium", letter_spacing=1.5
    ))

    # Code
    code: FontDefinition = field(default_factory=lambda: FontDefinition(
        family="ui-monospace, monospace", size=14.0
    ))

    def get_font(self, name: str) -> Optional[FontDefinition]:
        """Get a font definition by name."""
        return getattr(self, name, None)

    def clone(self) -> "Typography":
        """Create a copy of this typography."""
        return Typography(
            font_family_primary=self.font_family_primary,
            font_family_secondary=self.font_family_secondary,
            font_family_monospace=self.font_family_monospace,
            h1=FontDefinition(**self.h1.__dict__),
            h2=FontDefinition(**self.h2.__dict__),
            h3=FontDefinition(**self.h3.__dict__),
            h4=FontDefinition(**self.h4.__dict__),
            h5=FontDefinition(**self.h5.__dict__),
            h6=FontDefinition(**self.h6.__dict__),
            body_large=FontDefinition(**self.body_large.__dict__),
            body=FontDefinition(**self.body.__dict__),
            body_small=FontDefinition(**self.body_small.__dict__),
            label=FontDefinition(**self.label.__dict__),
            button=FontDefinition(**self.button.__dict__),
            caption=FontDefinition(**self.caption.__dict__),
            overline=FontDefinition(**self.overline.__dict__),
            code=FontDefinition(**self.code.__dict__),
        )


# ========== Spacing ==========

@dataclass
class Spacing:
    """
    Spacing scale for consistent layout.

    Uses a base unit multiplier system.
    """

    base: float = 4.0  # Base unit in pixels

    # Named spacing values (multiples of base)
    xxs: float = 1.0   # 4px
    xs: float = 2.0    # 8px
    sm: float = 3.0    # 12px
    md: float = 4.0    # 16px
    lg: float = 6.0    # 24px
    xl: float = 8.0    # 32px
    xxl: float = 12.0  # 48px
    xxxl: float = 16.0 # 64px

    def get(self, name: str) -> float:
        """Get spacing value by name (returns actual pixels). Returns md as default."""
        multiplier = getattr(self, name, self.md)
        return self.base * multiplier

    def __call__(self, multiplier: float) -> float:
        """Get spacing value as multiple of base unit."""
        return self.base * multiplier

    def clone(self) -> "Spacing":
        """Create a copy of this spacing."""
        return Spacing(
            base=self.base,
            xxs=self.xxs,
            xs=self.xs,
            sm=self.sm,
            md=self.md,
            lg=self.lg,
            xl=self.xl,
            xxl=self.xxl,
            xxxl=self.xxxl,
        )


# ========== Shadows ==========

@dataclass
class ShadowDefinition:
    """Definition of a shadow effect."""

    color: Color = field(default_factory=lambda: Color(0, 0, 0, 0.1))
    offset_x: float = 0.0
    offset_y: float = 2.0
    blur: float = 4.0
    spread: float = 0.0

    def to_style_dict(self) -> Dict[str, Any]:
        """Convert to style dictionary."""
        return {
            "shadow_color": self.color,
            "shadow_offset_x": self.offset_x,
            "shadow_offset_y": self.offset_y,
            "shadow_blur": self.blur,
            "shadow_spread": self.spread,
        }


@dataclass
class Shadows:
    """
    Shadow definitions for a theme.

    Provides elevation-based shadows.
    """

    none: ShadowDefinition = field(default_factory=lambda: ShadowDefinition(
        color=Color(0, 0, 0, 0), offset_y=0, blur=0, spread=0
    ))
    sm: ShadowDefinition = field(default_factory=lambda: ShadowDefinition(
        offset_y=1, blur=2, spread=0
    ))
    md: ShadowDefinition = field(default_factory=lambda: ShadowDefinition(
        offset_y=2, blur=4, spread=-1
    ))
    lg: ShadowDefinition = field(default_factory=lambda: ShadowDefinition(
        offset_y=4, blur=8, spread=-2
    ))
    xl: ShadowDefinition = field(default_factory=lambda: ShadowDefinition(
        offset_y=8, blur=16, spread=-4
    ))
    xxl: ShadowDefinition = field(default_factory=lambda: ShadowDefinition(
        offset_y=12, blur=24, spread=-6
    ))

    # Inner shadow for inset effects
    inner: ShadowDefinition = field(default_factory=lambda: ShadowDefinition(
        offset_y=2, blur=4, spread=-1, color=Color(0, 0, 0, 0.06)
    ))

    def get(self, name: str) -> Optional[ShadowDefinition]:
        """Get shadow by name."""
        return getattr(self, name, None)

    def clone(self) -> "Shadows":
        """Create a copy of this shadows collection."""
        return Shadows(
            none=ShadowDefinition(**self.none.__dict__),
            sm=ShadowDefinition(**self.sm.__dict__),
            md=ShadowDefinition(**self.md.__dict__),
            lg=ShadowDefinition(**self.lg.__dict__),
            xl=ShadowDefinition(**self.xl.__dict__),
            xxl=ShadowDefinition(**self.xxl.__dict__),
            inner=ShadowDefinition(**self.inner.__dict__),
        )


# ========== Border Radii ==========

# Large value used for fully rounded elements (pills/circles)
BORDER_RADIUS_FULL = 9999.0


@dataclass
class BorderRadii:
    """Border radius scale for consistent rounding."""

    none: float = 0.0
    sm: float = 2.0
    md: float = 4.0
    lg: float = 8.0
    xl: float = 12.0
    xxl: float = 16.0
    full: float = BORDER_RADIUS_FULL

    def get(self, name: str) -> float:
        """Get radius by name. Returns md (4.0) as default if name not found."""
        return getattr(self, name, self.md)

    def clone(self) -> "BorderRadii":
        """Create a copy of this radii collection."""
        return BorderRadii(
            none=self.none,
            sm=self.sm,
            md=self.md,
            lg=self.lg,
            xl=self.xl,
            xxl=self.xxl,
            full=self.full,
        )


# ========== Transitions ==========

@dataclass
class Transitions:
    """Transition timing presets."""

    # Durations (in seconds)
    duration_instant: float = 0.0
    duration_fast: float = 0.1
    duration_normal: float = 0.2
    duration_slow: float = 0.3
    duration_slower: float = 0.5

    # Easing functions
    ease_linear: str = "linear"
    ease_in: str = "ease-in"
    ease_out: str = "ease-out"
    ease_in_out: str = "ease-in-out"
    ease_bounce: str = "cubic-bezier(0.68, -0.55, 0.265, 1.55)"

    def clone(self) -> "Transitions":
        """Create a copy of this transitions collection."""
        return Transitions(
            duration_instant=self.duration_instant,
            duration_fast=self.duration_fast,
            duration_normal=self.duration_normal,
            duration_slow=self.duration_slow,
            duration_slower=self.duration_slower,
            ease_linear=self.ease_linear,
            ease_in=self.ease_in,
            ease_out=self.ease_out,
            ease_in_out=self.ease_in_out,
            ease_bounce=self.ease_bounce,
        )


# ========== Theme ==========

@dataclass
class Theme:
    """
    Complete theme definition.

    Contains color palette, typography, spacing, and other design tokens.
    """

    # Theme identity
    name: str = "default"
    is_dark: bool = False

    # Design tokens
    colors: ColorPalette = field(default_factory=ColorPalette)
    typography: Typography = field(default_factory=Typography)
    spacing: Spacing = field(default_factory=Spacing)
    shadows: Shadows = field(default_factory=Shadows)
    radii: BorderRadii = field(default_factory=BorderRadii)
    transitions: Transitions = field(default_factory=Transitions)

    # Widget style overrides
    widget_styles: Dict[str, StateStyles] = field(default_factory=dict)

    # Parent theme for inheritance
    _parent: Optional["Theme"] = field(default=None, repr=False, compare=False)

    # Theme change listeners
    _listeners: List[Callable[["Theme"], None]] = field(
        default_factory=list, repr=False, compare=False
    )

    # Registry of built-in themes
    _registry: ClassVar[Dict[str, "Theme"]] = {}

    def __post_init__(self) -> None:
        """Register theme if named."""
        if self.name and self.name != "default":
            Theme._registry[self.name] = self

    # ========== Theme Inheritance ==========

    def inherit_from(self, parent: "Theme") -> "Theme":
        """
        Create a new theme that inherits from parent.

        Args:
            parent: Parent theme to inherit from

        Returns:
            New theme with parent set
        """
        new_theme = self.clone()
        new_theme._parent = parent
        return new_theme

    def get_effective_value(self, path: str) -> Any:
        """
        Get effective value for a path, falling back to parent.

        Args:
            path: Dot-separated path (e.g., "colors.primary")

        Returns:
            Value at path or from parent
        """
        parts = path.split(".")
        value = self
        for part in parts:
            value = getattr(value, part, None)
            if value is None:
                break

        if value is None and self._parent:
            return self._parent.get_effective_value(path)

        return value

    # ========== Widget Styles ==========

    def get_widget_style(self, widget_type: str) -> Optional[StateStyles]:
        """
        Get styles for a widget type.

        Args:
            widget_type: Widget type name

        Returns:
            StateStyles for the widget or None
        """
        style = self.widget_styles.get(widget_type)
        if style is None and self._parent:
            return self._parent.get_widget_style(widget_type)
        return style

    def set_widget_style(self, widget_type: str, styles: StateStyles) -> None:
        """
        Set styles for a widget type.

        Args:
            widget_type: Widget type name
            styles: StateStyles to set
        """
        self.widget_styles[widget_type] = styles

    # ========== Style Generation ==========

    def create_button_style(
        self,
        variant: str = "primary",
    ) -> StateStyles:
        """
        Create button styles based on theme.

        Args:
            variant: Button variant (primary, secondary, outline, ghost)

        Returns:
            StateStyles for button
        """
        if variant == "primary":
            bg_color = self.colors.primary
            text_color = self.colors.text_inverse
            hover_bg = self.colors.primary_dark
            pressed_bg = self.colors.primary_dark.darken(0.1) if self.colors.primary_dark else bg_color
        elif variant == "secondary":
            bg_color = self.colors.secondary
            text_color = self.colors.text_inverse
            hover_bg = self.colors.secondary_dark
            pressed_bg = self.colors.secondary_dark.darken(0.1) if self.colors.secondary_dark else bg_color
        elif variant == "outline":
            bg_color = Color(0, 0, 0, 0)
            text_color = self.colors.primary
            hover_bg = self.colors.primary.with_alpha(0.1)
            pressed_bg = self.colors.primary.with_alpha(0.2)
        elif variant == "ghost":
            bg_color = Color(0, 0, 0, 0)
            text_color = self.colors.text_primary
            hover_bg = self.colors.background_secondary
            pressed_bg = self.colors.background_tertiary
        else:
            bg_color = self.colors.primary
            text_color = self.colors.text_inverse
            hover_bg = self.colors.primary_dark
            pressed_bg = bg_color

        font = self.typography.button

        normal = Style(
            background=SolidBrush(bg_color),
            foreground_color=text_color,
            border_radius=self.radii.md,
            font_family=font.family,
            font_size=font.size,
            font_weight=font.weight,
            padding_left=self.spacing.get("md"),
            padding_right=self.spacing.get("md"),
            padding_top=self.spacing.get("xs"),
            padding_bottom=self.spacing.get("xs"),
            transition_duration=self.transitions.duration_fast,
            transition_easing=self.transitions.ease_out,
            cursor="pointer",
        )

        if variant == "outline":
            normal.border_color = self.colors.primary
            normal.border_width = 1.0

        return StateStyles(
            normal=normal,
            hovered=Style(
                background=SolidBrush(hover_bg) if hover_bg else None,
                scale_x=1.02,
                scale_y=1.02,
            ),
            pressed=Style(
                background=SolidBrush(pressed_bg) if pressed_bg else None,
                scale_x=0.98,
                scale_y=0.98,
            ),
            focused=Style(
                border_color=self.colors.border_focus,
                border_width=2.0,
            ),
            disabled=Style(
                background=SolidBrush(self.colors.background_tertiary),
                foreground_color=self.colors.text_disabled,
                cursor="not-allowed",
                opacity=0.6,
            ),
        )

    def create_input_style(self) -> StateStyles:
        """Create text input styles based on theme."""
        font = self.typography.body

        return StateStyles(
            normal=Style(
                background=SolidBrush(self.colors.surface),
                foreground_color=self.colors.text_primary,
                border_color=self.colors.border,
                border_width=1.0,
                border_radius=self.radii.md,
                font_family=font.family,
                font_size=font.size,
                padding_left=self.spacing.get("sm"),
                padding_right=self.spacing.get("sm"),
                padding_top=self.spacing.get("xs"),
                padding_bottom=self.spacing.get("xs"),
                transition_duration=self.transitions.duration_fast,
            ),
            hovered=Style(
                border_color=self.colors.text_secondary,
            ),
            focused=Style(
                border_color=self.colors.border_focus,
                border_width=2.0,
            ),
            disabled=Style(
                background=SolidBrush(self.colors.background_secondary),
                foreground_color=self.colors.text_disabled,
                cursor="not-allowed",
            ),
        )

    def create_card_style(self, elevated: bool = True) -> Style:
        """Create card/panel styles based on theme."""
        style = Style(
            background=SolidBrush(self.colors.surface),
            border_radius=self.radii.lg,
            padding_left=self.spacing.get("md"),
            padding_right=self.spacing.get("md"),
            padding_top=self.spacing.get("md"),
            padding_bottom=self.spacing.get("md"),
        )

        if elevated:
            shadow = self.shadows.md
            style.shadow_color = shadow.color
            style.shadow_offset_x = shadow.offset_x
            style.shadow_offset_y = shadow.offset_y
            style.shadow_blur = shadow.blur
            style.shadow_spread = shadow.spread

        return style

    # ========== Theme Switching ==========

    def activate(self) -> contextvars.Token:
        """
        Activate this theme as the current theme.

        Returns:
            Token to restore previous theme
        """
        token = set_current_theme(self)
        self._notify_listeners()
        return token

    def add_listener(self, listener: Callable[["Theme"], None]) -> None:
        """Add a theme change listener."""
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[["Theme"], None]) -> None:
        """Remove a theme change listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_listeners(self) -> None:
        """Notify all listeners of theme change."""
        for listener in self._listeners:
            listener(self)

    # ========== Cloning ==========

    def clone(self) -> "Theme":
        """Create a deep copy of this theme."""
        return Theme(
            name=f"{self.name}_copy",
            is_dark=self.is_dark,
            colors=self.colors.clone(),
            typography=self.typography.clone(),
            spacing=self.spacing.clone(),
            shadows=self.shadows.clone(),
            radii=self.radii.clone(),
            transitions=self.transitions.clone(),
            widget_styles={k: v.clone() for k, v in self.widget_styles.items()},
        )

    # ========== Registry ==========

    @classmethod
    def get(cls, name: str) -> Optional["Theme"]:
        """Get a registered theme by name."""
        return cls._registry.get(name)

    @classmethod
    def list_themes(cls) -> List[str]:
        """List all registered theme names."""
        return list(cls._registry.keys())


# ========== Theme Provider ==========

class ThemeProvider:
    """
    Context manager for temporarily setting a theme.

    Implements the theme provider pattern for scoped theming.
    """

    def __init__(self, theme: Theme) -> None:
        """Initialize with theme to provide."""
        self.theme = theme
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> "ThemeProvider":
        """Enter context and set theme."""
        self._token = self.theme.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and restore previous theme."""
        if self._token is not None:
            _current_theme.reset(self._token)


# ========== Built-in Themes ==========

def create_light_theme() -> Theme:
    """Create the built-in light theme."""
    return Theme(
        name="light",
        is_dark=False,
        colors=ColorPalette(
            primary=Color.from_hex("#3B82F6"),
            secondary=Color.from_hex("#8B5CF6"),
            accent=Color.from_hex("#F59E0B"),
            success=Color.from_hex("#10B981"),
            warning=Color.from_hex("#F59E0B"),
            error=Color.from_hex("#EF4444"),
            info=Color.from_hex("#3B82F6"),
            background=Color.from_hex("#FFFFFF"),
            background_secondary=Color.from_hex("#F3F4F6"),
            background_tertiary=Color.from_hex("#E5E7EB"),
            surface=Color.from_hex("#FFFFFF"),
            surface_variant=Color.from_hex("#F9FAFB"),
            text_primary=Color.from_hex("#111827"),
            text_secondary=Color.from_hex("#6B7280"),
            text_disabled=Color.from_hex("#9CA3AF"),
            text_inverse=Color.from_hex("#FFFFFF"),
            border=Color.from_hex("#E5E7EB"),
            border_focus=Color.from_hex("#3B82F6"),
        ),
    )


def create_dark_theme() -> Theme:
    """Create the built-in dark theme."""
    return Theme(
        name="dark",
        is_dark=True,
        colors=ColorPalette(
            primary=Color.from_hex("#60A5FA"),
            secondary=Color.from_hex("#A78BFA"),
            accent=Color.from_hex("#FBBF24"),
            success=Color.from_hex("#34D399"),
            warning=Color.from_hex("#FBBF24"),
            error=Color.from_hex("#F87171"),
            info=Color.from_hex("#60A5FA"),
            background=Color.from_hex("#111827"),
            background_secondary=Color.from_hex("#1F2937"),
            background_tertiary=Color.from_hex("#374151"),
            surface=Color.from_hex("#1F2937"),
            surface_variant=Color.from_hex("#374151"),
            text_primary=Color.from_hex("#F9FAFB"),
            text_secondary=Color.from_hex("#9CA3AF"),
            text_disabled=Color.from_hex("#6B7280"),
            text_inverse=Color.from_hex("#111827"),
            border=Color.from_hex("#374151"),
            border_focus=Color.from_hex("#60A5FA"),
            overlay=Color(0, 0, 0, 0.7),
            scrim=Color(0, 0, 0, 0.5),
        ),
        shadows=Shadows(
            none=ShadowDefinition(color=Color(0, 0, 0, 0), offset_y=0, blur=0, spread=0),
            sm=ShadowDefinition(color=Color(0, 0, 0, 0.3), offset_y=1, blur=2, spread=0),
            md=ShadowDefinition(color=Color(0, 0, 0, 0.3), offset_y=2, blur=4, spread=0),
            lg=ShadowDefinition(color=Color(0, 0, 0, 0.3), offset_y=4, blur=8, spread=0),
            xl=ShadowDefinition(color=Color(0, 0, 0, 0.3), offset_y=8, blur=16, spread=0),
            xxl=ShadowDefinition(color=Color(0, 0, 0, 0.3), offset_y=12, blur=24, spread=0),
            inner=ShadowDefinition(color=Color(0, 0, 0, 0.2), offset_y=2, blur=4, spread=0),
        ),
    )


def create_high_contrast_theme() -> Theme:
    """Create the built-in high contrast theme for accessibility."""
    return Theme(
        name="high_contrast",
        is_dark=True,
        colors=ColorPalette(
            primary=Color.from_hex("#FFFF00"),
            secondary=Color.from_hex("#00FFFF"),
            accent=Color.from_hex("#FF00FF"),
            success=Color.from_hex("#00FF00"),
            warning=Color.from_hex("#FFFF00"),
            error=Color.from_hex("#FF0000"),
            info=Color.from_hex("#00FFFF"),
            background=Color.from_hex("#000000"),
            background_secondary=Color.from_hex("#1A1A1A"),
            background_tertiary=Color.from_hex("#333333"),
            surface=Color.from_hex("#000000"),
            surface_variant=Color.from_hex("#1A1A1A"),
            text_primary=Color.from_hex("#FFFFFF"),
            text_secondary=Color.from_hex("#CCCCCC"),
            text_disabled=Color.from_hex("#666666"),
            text_inverse=Color.from_hex("#000000"),
            border=Color.from_hex("#FFFFFF"),
            border_focus=Color.from_hex("#FFFF00"),
            overlay=Color(0, 0, 0, 0.9),
            scrim=Color(0, 0, 0, 0.8),
        ),
        radii=BorderRadii(
            none=0.0,
            sm=0.0,
            md=0.0,
            lg=0.0,
            xl=0.0,
            xxl=0.0,
            full=BORDER_RADIUS_FULL,
        ),
        shadows=Shadows(
            none=ShadowDefinition(color=Color(0, 0, 0, 0), offset_y=0, blur=0, spread=0),
            sm=ShadowDefinition(color=Color(1, 1, 1, 0.3), offset_y=0, blur=0, spread=1),
            md=ShadowDefinition(color=Color(1, 1, 1, 0.3), offset_y=0, blur=0, spread=2),
            lg=ShadowDefinition(color=Color(1, 1, 1, 0.3), offset_y=0, blur=0, spread=3),
            xl=ShadowDefinition(color=Color(1, 1, 1, 0.3), offset_y=0, blur=0, spread=4),
            xxl=ShadowDefinition(color=Color(1, 1, 1, 0.3), offset_y=0, blur=0, spread=5),
            inner=ShadowDefinition(color=Color(1, 1, 1, 0.2), offset_y=0, blur=0, spread=1),
        ),
    )


# ========== Initialize Built-in Themes ==========

# Create and register built-in themes
LIGHT_THEME = create_light_theme()
DARK_THEME = create_dark_theme()
HIGH_CONTRAST_THEME = create_high_contrast_theme()

# Set light theme as default
LIGHT_THEME.activate()

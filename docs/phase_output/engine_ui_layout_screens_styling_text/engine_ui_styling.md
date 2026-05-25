# Investigation: engine/ui/styling

## Summary
The UI styling system is a comprehensive, production-ready implementation providing CSS-like styling capabilities including color management with RGBA/HSL/HSV support, multiple brush types (solid, gradient, image, nine-slice), a full style property system with visual states and inheritance, and a complete theme system with design tokens, light/dark/high-contrast built-in themes, and runtime switching.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 147 | REAL | Comprehensive public API exports |
| `color.py` | 902 | REAL | Full Color class with RGBA, HSL, HSV, blending, WCAG contrast |
| `brush.py` | 587 | REAL | SolidBrush, GradientBrush, ImageBrush, NineSliceBrush |
| `style.py` | 948 | REAL | Style, StateStyles, StyleSelector, Stylesheet, StyleBuilder |
| `theme.py` | 912 | REAL | Theme, ColorPalette, Typography, Spacing, Shadows, Transitions |

## Styling Components

### Color System (`color.py`)
- `Color` dataclass with RGBA (0.0-1.0) storage
- Factory methods: `from_rgb()`, `from_hex()`, `from_hsl()`, `from_hsv()`, `from_name()`
- Manipulation: `lighten()`, `darken()`, `saturate()`, `desaturate()`, `grayscale()`, `invert()`, `rotate_hue()`
- 12 blend modes: NORMAL, MULTIPLY, SCREEN, OVERLAY, etc.
- Interpolation: `lerp()`, `lerp_hsl()`
- WCAG accessibility: `luminance`, `contrast_ratio()`, `is_readable_on()`
- Palette generators: complementary, triadic, analogous, split-complementary, tetradic
- 100+ named colors (CSS color names)

### Brush System (`brush.py`)
- `Brush` abstract base with `get_color_at()`, `clone()`, `is_opaque`
- `SolidBrush` - single color fill
- `GradientBrush` - LINEAR, RADIAL, ANGULAR, DIAMOND gradients with color stops
- `ImageBrush` - texture fills with TileMode (REPEAT, MIRROR, CLAMP) and ImageFit (FILL, CONTAIN, COVER)
- `NineSliceBrush` - 9-slice scaling for UI borders

### Style System (`style.py`)
- `Style` dataclass with 40+ properties: background, border, foreground, font, opacity, padding, margin, shadow, transform, cursor, transition
- `VisualState` enum: NORMAL, HOVERED, PRESSED, FOCUSED, DISABLED, SELECTED
- `StateStyles` - manages styles per visual state with precedence resolution
- `StyleSelector` with CSS-like matching: TYPE, NAME, CLASS, STATE, ID, UNIVERSAL
- `Stylesheet` - collection of `StyleRule` with specificity-based cascade
- `StyleBuilder` - fluent API for style construction
- `StylePropertyDescriptor` - validated property descriptor with type checking

### Theme System (`theme.py`)
- `Theme` dataclass with full design token collections
- `ColorPalette` - semantic colors (primary, secondary, accent, success, warning, error, info, background, surface, text, border)
- `Typography` - font definitions for h1-h6, body, label, button, caption, code
- `Spacing` - 8-point grid system (xxs through xxxl)
- `Shadows` - elevation-based shadow presets (sm, md, lg, xl, xxl, inner)
- `BorderRadii` - corner radius scale
- `Transitions` - duration and easing presets
- `ThemeProvider` context manager for scoped theming
- Built-in themes: LIGHT_THEME, DARK_THEME, HIGH_CONTRAST_THEME
- Thread-safe via `contextvars`

## Implementation

- Real style system? **YES** - Full CSS-like property system with validation, inheritance, merging
- Real themes? **YES** - Complete theme architecture with 3 built-in themes, design tokens, runtime switching
- Real property inheritance? **YES** - `Style.inherit_from()`, `Style.merge()`, `Theme.inherit_from()`, `get_effective_value()` with parent fallback

## Verdict
**REAL IMPLEMENTATION**

This is a complete, production-quality UI styling system comparable to frameworks like Flutter, React Native, or WPF styling. All code is functional with proper validation, immutability where appropriate, thread-safety, and comprehensive APIs.

## Evidence

### Color - Full HSL manipulation and WCAG contrast
```python
def lighten(self, amount: float) -> "Color":
    h, s, l = self.to_hsl()
    new_l = min(1.0, l + amount * (1.0 - l))
    return Color.from_hsl(h, s, new_l, self.a)

def contrast_ratio(self, other: "Color") -> float:
    l1 = self.luminance
    l2 = other.luminance
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)
```

### Gradient - Real mathematical interpolation
```python
def _radial_position(self, nx: float, ny: float) -> float:
    dx = (nx - self.center_x) / self.radius_x if self.radius_x > 0 else 0
    dy = (ny - self.center_y) / self.radius_y if self.radius_y > 0 else 0
    return (dx * dx + dy * dy) ** 0.5

def _interpolate_color(self, pos: float) -> Color:
    # Find surrounding stops and interpolate
    for stop in self.stops:
        if stop.position <= pos:
            prev_stop = stop
        if stop.position >= pos:
            next_stop = stop
            break
    t = (pos - prev_stop.position) / (next_stop.position - prev_stop.position)
    return prev_stop.color.lerp(next_stop.color, t)
```

### Style inheritance with specificity
```python
def get_computed_style(
    self,
    widget_type: Optional[Type] = None,
    widget_name: Optional[str] = None,
    widget_id: Optional[str] = None,
    style_classes: Optional[Set[str]] = None,
    active_states: Optional[Set[VisualState]] = None,
    base_style: Optional[Style] = None,
) -> Style:
    result = base_style.clone() if base_style else Style()
    for rule in self._rules:
        if rule.selector.matches(widget_type=widget_type, ...):
            result = rule.style.merge(result)
    return result
```

### Theme widget style generation
```python
def create_button_style(self, variant: str = "primary") -> StateStyles:
    if variant == "primary":
        bg_color = self.colors.primary
        text_color = self.colors.text_inverse
        hover_bg = self.colors.primary_dark
    # ... creates full StateStyles with NORMAL, HOVERED, PRESSED, FOCUSED, DISABLED
    return StateStyles(normal=normal, hovered=..., pressed=..., disabled=...)
```

### Thread-safe theme context
```python
_current_theme: contextvars.ContextVar["Theme"] = contextvars.ContextVar("current_theme")

class ThemeProvider:
    def __enter__(self) -> "ThemeProvider":
        self._token = self.theme.activate()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _current_theme.reset(self._token)
```

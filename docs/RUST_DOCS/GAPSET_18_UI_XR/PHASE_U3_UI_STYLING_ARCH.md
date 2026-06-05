# Phase U3: UI Styling and Themes — Architecture

**Tasks:** T-UX-3.1 through T-UX-3.3 (3 tasks)
**Effort:** 8-11 days
**Status:** ✅ COMPLETE (per PROJECT.md verification)

---

## 1. Overview

Phase U3 implements the styling system: style properties, themes, brushes, and colors. Enables consistent visual appearance across widgets with theme switching.

---

## 2. Style Properties (`styling/style.py`)

```python
class StyleProperties:
    # Colors (ValidatedDescriptor with hex/named validation)
    background_color: str
    foreground_color: str
    border_color: str
    
    # Numeric (RangeDescriptor)
    opacity: float          # 0.0-1.0
    border_width: float     # 0-100
    corner_radius: float    # 0-100
    
    # Font (ChoiceDescriptor + RangeDescriptor)
    font_size: int          # 1-200
    font_weight: str        # "normal", "bold", "light"
    font_family: str
```

**Inheritance:** Child widgets inherit parent style properties unless overridden.

---

## 3. Theme System (`styling/theme.py`)

### Theme Structure
```
Theme
├── name: str
├── palette: dict[str, Color]    # Named color values
├── typography: dict[str, Font]  # Named font specs
├── spacing: dict[str, float]    # Named spacing values
└── components: dict[str, Style] # Per-widget-type defaults
```

### Built-in Themes
| Theme | Description |
|-------|-------------|
| Light | Bright backgrounds, dark text |
| Dark | Dark backgrounds, light text |
| High Contrast | Maximum contrast for accessibility |

### Theme Switching
Theme switch triggers `Tracker` subscriptions on all style properties, propagating updates through the widget tree.

---

## 4. Brush Types (`styling/brush.py`)

| Brush Type | Parameters |
|------------|------------|
| SolidBrush | color |
| LinearGradientBrush | start, end, stops[] |
| RadialGradientBrush | center, radius, stops[] |
| ImageBrush | texture, tile_mode |

All brushes implement common `Brush` interface for rendering.

---

## 5. Color Utilities (`styling/color.py`)

### Supported Formats
- Hex: `#RRGGBB`, `#RRGGBBAA`
- Named: CSS color names (140 standard colors)
- Function: `rgba(r, g, b, a)`

### Color Space
- sRGB for input/output
- Linear RGB for blending operations
- Conversion functions: `srgb_to_linear()`, `linear_to_srgb()`

---

## 6. Dependencies

- Phase U1: Widget base
- Trinity: ValidatedDescriptor, RangeDescriptor, ChoiceDescriptor
- Foundation: Tracker (for theme subscription)

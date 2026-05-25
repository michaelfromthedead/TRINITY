# PHASE 3 ARCHITECTURE: Styling Module

---

## Overview

The Styling Module provides CSS-like styling capabilities including style properties, selectors, themes, colors, and brushes. It comprises 4 files (~3,345 lines) implementing a comprehensive design system.

---

## Component Architecture

### Style System (style.py — 947 lines)

**Purpose**: CSS-like styling with properties, selectors, and cascading.

**Classes**:
- `Style` — Collection of style properties
- `StateStyles` — Styles per state (hover, pressed, disabled, focused)
- `StyleSelector` — CSS-like selector (class, id, type, pseudo-class)
- `Stylesheet` — Collection of rules with selector matching

**Style Properties (40+)**:
| Category | Properties |
|----------|------------|
| Box Model | `margin`, `padding`, `width`, `height`, `min_*`, `max_*` |
| Border | `border_width`, `border_color`, `border_radius`, `border_style` |
| Background | `background_color`, `background_image`, `background_size` |
| Text | `color`, `font_family`, `font_size`, `font_weight`, `text_align` |
| Layout | `display`, `flex_grow`, `flex_shrink`, `position` |
| Effects | `opacity`, `shadow`, `transform` |

**State-Based Styling**:
```python
style = Style(
    background_color=Color.white(),
    states={
        "hover": Style(background_color=Color.gray(0.95)),
        "pressed": Style(background_color=Color.gray(0.9)),
        "disabled": Style(opacity=0.5),
        "focused": Style(border_color=Color.blue())
    }
)
```

**Selector Types**:
| Selector | Syntax | Example |
|----------|--------|---------|
| Type | `element` | `Button` |
| Class | `.class` | `.primary` |
| ID | `#id` | `#submit-btn` |
| Pseudo-class | `:state` | `:hover`, `:disabled` |
| Compound | `element.class#id:state` | `Button.primary:hover` |

**Specificity Calculation** (`_calculate_specificity()`):
```
Specificity = (inline, id_count, class_count, type_count)

#id.class element  -> (0, 1, 1, 1)
.class.class       -> (0, 0, 2, 0)
element            -> (0, 0, 0, 1)
inline style       -> (1, 0, 0, 0)
```

**Stylesheet Cascading**:
```python
stylesheet = Stylesheet()
stylesheet.add_rule(StyleSelector("Button"), Style(padding=10))
stylesheet.add_rule(StyleSelector("Button.primary"), Style(background_color=Color.blue()))

# Matching and merging
matched_styles = stylesheet.match(element)  # returns list by specificity
final_style = Style.merge(matched_styles)   # higher specificity wins
```

---

### Theme System (theme.py — 911 lines)

**Purpose**: Design token system for consistent styling.

**Classes**:
- `Theme` — Container for all design tokens
- `ColorPalette` — Named color collection
- `Typography` — Font styles and sizes
- `Spacing` — Spacing scale values
- `Shadows` — Shadow definitions
- `BorderRadii` — Border radius presets

**Built-in Themes**:
| Theme | Characteristics |
|-------|-----------------|
| `light` | White backgrounds, dark text, subtle shadows |
| `dark` | Dark backgrounds, light text, deeper shadows |
| `high_contrast` | Maximum contrast, bold borders, no gradients |

**Design Tokens**:
```python
theme = Theme(
    colors=ColorPalette(
        primary=Color.hex("#007AFF"),
        secondary=Color.hex("#5856D6"),
        success=Color.hex("#34C759"),
        warning=Color.hex("#FF9500"),
        danger=Color.hex("#FF3B30"),
        background=Color.white(),
        foreground=Color.black()
    ),
    typography=Typography(
        heading_1=FontStyle(size=32, weight="bold"),
        heading_2=FontStyle(size=24, weight="bold"),
        body=FontStyle(size=16, weight="normal"),
        caption=FontStyle(size=12, weight="normal")
    ),
    spacing=Spacing(
        xs=4, sm=8, md=16, lg=24, xl=32
    ),
    shadows=Shadows(
        sm=Shadow(offset=(0, 1), blur=2, color=Color.rgba(0, 0, 0, 0.1)),
        md=Shadow(offset=(0, 2), blur=4, color=Color.rgba(0, 0, 0, 0.15)),
        lg=Shadow(offset=(0, 4), blur=8, color=Color.rgba(0, 0, 0, 0.2))
    ),
    radii=BorderRadii(
        sm=4, md=8, lg=16, full=9999
    )
)
```

**Token Access**:
```python
color = theme.get_color("primary")
font_style = theme.get_typography("body")
padding = theme.get_spacing("md")
shadow = theme.get_shadow("sm")
radius = theme.get_radius("lg")
```

**Theme Inheritance** (`derive_theme()`):
```python
brand_theme = light_theme.derive_theme(
    colors={"primary": Color.hex("#FF5722")}
)
# All other tokens inherited from light_theme
```

---

### Color System (color.py — 901 lines)

**Purpose**: Color manipulation, blending, and accessibility.

**Class**: `Color`

**Color Spaces**:
| Space | Components | Range |
|-------|------------|-------|
| RGBA | Red, Green, Blue, Alpha | 0.0-1.0 |
| HSL | Hue, Saturation, Lightness | H: 0-360, S/L: 0-1 |
| HSV | Hue, Saturation, Value | H: 0-360, S/V: 0-1 |
| HEX | Hexadecimal string | #RRGGBB or #RRGGBBAA |

**Conversions**:
```python
# RGBA <-> HSL
color = Color.rgba(1.0, 0.5, 0.25, 1.0)
h, s, l = color.to_hsl()
color2 = Color.from_hsl(h, s, l)

# HEX
color = Color.hex("#FF8040")
hex_str = color.to_hex()
```

**12 Blend Modes**:
| Mode | Formula | Effect |
|------|---------|--------|
| `normal` | B | Top layer only |
| `multiply` | A * B | Darken |
| `screen` | 1 - (1-A) * (1-B) | Lighten |
| `overlay` | multiply/screen per channel | Contrast |
| `darken` | min(A, B) | Darker of two |
| `lighten` | max(A, B) | Lighter of two |
| `color_dodge` | A / (1 - B) | Brighten |
| `color_burn` | 1 - (1-A) / B | Burn shadows |
| `hard_light` | overlay with swap | Sharp contrast |
| `soft_light` | piecewise formula | Soft contrast |
| `difference` | abs(A - B) | Invert |
| `exclusion` | A + B - 2*A*B | Similar to difference |

**WCAG Accessibility**:
```python
# Relative luminance (per WCAG 2.1)
lum = color.get_luminance()

# Contrast ratio (1:1 to 21:1)
ratio = Color.contrast_ratio(foreground, background)

# Readability check (AA = 4.5:1, AAA = 7:1)
is_readable = Color.is_readable(foreground, background, level="AA")
```

**Palette Generation**:
```python
# Color harmony
complement = color.complementary()        # 180 degrees opposite
triadic = color.triadic()                 # 3 colors, 120 degrees apart
analogous = color.analogous()             # 3 colors, 30 degrees apart
split = color.split_complementary()       # 2 colors, 150 degrees from base
```

---

### Brush System (brush.py — 586 lines)

**Purpose**: Fill patterns for painting UI elements.

**Classes**:
- `Brush` — Abstract base
- `SolidBrush` — Single color fill
- `GradientBrush` — Gradient fill with stops
- `ImageBrush` — Image-based fill
- `NineSliceBrush` — Nine-slice scaling for borders

**Gradient Types**:
| Type | Description |
|------|-------------|
| `linear` | Straight line gradient |
| `radial` | Circular gradient from center |
| `angular` | Sweep around center point |
| `diamond` | Diamond-shaped gradient |

**Gradient Stops**:
```python
gradient = GradientBrush(
    type="linear",
    angle=90,  # degrees
    stops=[
        GradientStop(position=0.0, color=Color.red()),
        GradientStop(position=0.5, color=Color.yellow()),
        GradientStop(position=1.0, color=Color.green())
    ]
)
```

**Gradient Sampling** (`_sample_gradient()`):
```python
# Get color at position t (0-1)
color = gradient.sample(0.75)  # Interpolates between yellow and green
```

**Nine-Slice Scaling**:
```
+---+-------+---+
| 1 |   2   | 3 |   <- corners (1, 3, 7, 9) don't scale
+---+-------+---+
| 4 |   5   | 6 |   <- edges (2, 4, 6, 8) scale in one axis
+---+-------+---+   <- center (5) scales in both axes
| 7 |   8   | 9 |
+---+-------+---+
```

```python
nine_slice = NineSliceBrush(
    image=border_image,
    insets=(top=16, right=16, bottom=16, left=16)
)
regions = nine_slice._calculate_nine_slice_regions()
```

**Tiling Modes**:
| Mode | Behavior |
|------|----------|
| `stretch` | Stretch to fill |
| `tile` | Repeat pattern |
| `tile_x` | Tile horizontally only |
| `tile_y` | Tile vertically only |

---

## Module Dependencies

```
color.py     --(standalone)
brush.py     --> color.py (imports Color for gradients)
style.py     --> color.py (imports Color for style properties)
theme.py     --> color.py (imports Color for palette)
             --> style.py? (may use Style for typography)
```

---

## Integration Points

1. **Layout Module** — Styles include layout properties (margin, padding)
2. **Text Module** — Typography tokens used for text styling
3. **Render System** — Brushes provide fill data for renderer
4. **Widget System** — Widgets query styles and themes

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Selector match | O(s) | s = selector components |
| Specificity sort | O(n log n) | n = matched rules |
| Style merge | O(p) | p = properties |
| Gradient sample | O(stops) | Linear search for stop pair |
| Color conversion | O(1) | Mathematical formulas |
| Contrast ratio | O(1) | Luminance calculation |

---

## Design Decisions

1. **CSS Semantics** — Selector syntax and specificity mirror CSS
2. **Design Tokens** — Centralized tokens for consistency
3. **Multiple Themes** — Support light/dark/high-contrast out of box
4. **WCAG Compliance** — Accessibility built into color system
5. **Blend Modes** — Photoshop-compatible compositing
6. **Nine-Slice** — Game UI friendly border scaling

# PHASE 3 TODO: Styling Module

---

## Style System Tasks

### T-3.1: Verify Style Property Application

**File**: `engine/ui/styling/style.py`

**Description**: Ensure all 40+ style properties are applied correctly.

**Acceptance Criteria**:
- [ ] Box model properties (margin, padding, width, height) apply
- [ ] Border properties (width, color, radius, style) apply
- [ ] Background properties (color, image, size) apply
- [ ] Text properties (color, font_family, font_size, text_align) apply
- [ ] Layout properties (display, flex_*, position) apply
- [ ] Effect properties (opacity, shadow, transform) apply

---

### T-3.2: Verify State-Based Styling

**File**: `engine/ui/styling/style.py`

**Description**: Ensure `StateStyles` returns correct style for each state.

**Acceptance Criteria**:
- [ ] Default state returns base style
- [ ] Hover state returns hover style (or merged with base)
- [ ] Pressed state returns pressed style
- [ ] Disabled state returns disabled style
- [ ] Focused state returns focused style
- [ ] Multiple states combined (hover + focused)

---

### T-3.3: Verify Type Selector Matching

**File**: `engine/ui/styling/style.py`

**Description**: Ensure type selectors match element types.

**Acceptance Criteria**:
- [ ] `Button` matches Button elements
- [ ] `TextInput` matches TextInput elements
- [ ] Case sensitivity handled correctly
- [ ] No match for wrong type

---

### T-3.4: Verify Class Selector Matching

**File**: `engine/ui/styling/style.py`

**Description**: Ensure class selectors match element classes.

**Acceptance Criteria**:
- [ ] `.primary` matches elements with "primary" class
- [ ] `.primary.large` matches elements with both classes
- [ ] Order of classes does not matter
- [ ] No match if any class missing

---

### T-3.5: Verify ID Selector Matching

**File**: `engine/ui/styling/style.py`

**Description**: Ensure ID selectors match element IDs.

**Acceptance Criteria**:
- [ ] `#submit-btn` matches element with ID "submit-btn"
- [ ] IDs are unique (first match wins)
- [ ] No match for wrong ID

---

### T-3.6: Verify Pseudo-Class Selector Matching

**File**: `engine/ui/styling/style.py`

**Description**: Ensure pseudo-class selectors match element states.

**Acceptance Criteria**:
- [ ] `:hover` matches hovered elements
- [ ] `:disabled` matches disabled elements
- [ ] `:focused` matches focused elements
- [ ] `:first-child` matches first child
- [ ] `:last-child` matches last child

---

### T-3.7: Verify Specificity Calculation

**File**: `engine/ui/styling/style.py`

**Description**: Ensure `_calculate_specificity()` returns correct tuple.

**Acceptance Criteria**:
- [ ] Inline style = (1, 0, 0, 0)
- [ ] `#id` = (0, 1, 0, 0)
- [ ] `.class` = (0, 0, 1, 0)
- [ ] `element` = (0, 0, 0, 1)
- [ ] `#id.class element` = (0, 1, 1, 1)
- [ ] Higher specificity wins in cascade

---

### T-3.8: Verify Style Merging

**File**: `engine/ui/styling/style.py`

**Description**: Ensure `Style.merge()` combines styles correctly.

**Acceptance Criteria**:
- [ ] Later properties override earlier
- [ ] Unset properties inherited from earlier styles
- [ ] Specificity order respected
- [ ] Merge handles all property types

---

### T-3.9: Verify Stylesheet Cascading

**File**: `engine/ui/styling/style.py`

**Description**: Ensure stylesheet matches and cascades rules.

**Acceptance Criteria**:
- [ ] All matching rules returned
- [ ] Rules sorted by specificity
- [ ] Correct final style after merge
- [ ] Non-matching rules excluded

---

## Theme System Tasks

### T-3.10: Verify Color Palette Access

**File**: `engine/ui/styling/theme.py`

**Description**: Ensure `theme.get_color()` returns correct colors.

**Acceptance Criteria**:
- [ ] `get_color("primary")` returns primary color
- [ ] `get_color("background")` returns background color
- [ ] Unknown keys return None or default
- [ ] All palette colors accessible

---

### T-3.11: Verify Typography Access

**File**: `engine/ui/styling/theme.py`

**Description**: Ensure `theme.get_typography()` returns correct font styles.

**Acceptance Criteria**:
- [ ] `get_typography("heading_1")` returns H1 style
- [ ] `get_typography("body")` returns body style
- [ ] Font style includes size, weight, family
- [ ] Unknown keys handled

---

### T-3.12: Verify Spacing Scale

**File**: `engine/ui/styling/theme.py`

**Description**: Ensure `theme.get_spacing()` returns correct values.

**Acceptance Criteria**:
- [ ] `get_spacing("xs")` returns smallest spacing
- [ ] `get_spacing("xl")` returns largest spacing
- [ ] Spacing values are numeric
- [ ] Scale is consistent (e.g., doubling)

---

### T-3.13: Verify Shadow Definitions

**File**: `engine/ui/styling/theme.py`

**Description**: Ensure `theme.get_shadow()` returns correct shadow.

**Acceptance Criteria**:
- [ ] Shadow includes offset (x, y)
- [ ] Shadow includes blur radius
- [ ] Shadow includes spread (if supported)
- [ ] Shadow includes color

---

### T-3.14: Verify Border Radii Presets

**File**: `engine/ui/styling/theme.py`

**Description**: Ensure `theme.get_radius()` returns correct values.

**Acceptance Criteria**:
- [ ] `get_radius("sm")` returns small radius
- [ ] `get_radius("full")` returns very large value (for pills)
- [ ] Values are numeric

---

### T-3.15: Verify Theme Inheritance

**File**: `engine/ui/styling/theme.py`

**Description**: Ensure `derive_theme()` creates correct derived theme.

**Acceptance Criteria**:
- [ ] Overridden tokens use new values
- [ ] Non-overridden tokens inherited from parent
- [ ] Derived theme is independent (no mutation of parent)
- [ ] Deep nesting works (derive from derived)

---

### T-3.16: Verify Built-in Themes

**File**: `engine/ui/styling/theme.py`

**Description**: Ensure light, dark, and high_contrast themes exist.

**Acceptance Criteria**:
- [ ] Light theme has white/light backgrounds
- [ ] Dark theme has dark/black backgrounds
- [ ] High contrast theme has maximum contrast
- [ ] All three themes have complete token sets

---

## Color System Tasks

### T-3.17: Verify RGBA to HSL Conversion

**File**: `engine/ui/styling/color.py`

**Description**: Ensure `to_hsl()` converts correctly.

**Acceptance Criteria**:
- [ ] Red (1,0,0) -> H=0, S=1, L=0.5
- [ ] Green (0,1,0) -> H=120, S=1, L=0.5
- [ ] Blue (0,0,1) -> H=240, S=1, L=0.5
- [ ] White (1,1,1) -> S=0, L=1
- [ ] Black (0,0,0) -> S=0, L=0
- [ ] Round-trip conversion preserves color

---

### T-3.18: Verify HSL to RGBA Conversion

**File**: `engine/ui/styling/color.py`

**Description**: Ensure `from_hsl()` converts correctly.

**Acceptance Criteria**:
- [ ] H=0, S=1, L=0.5 -> Red
- [ ] H=120, S=1, L=0.5 -> Green
- [ ] H=240, S=1, L=0.5 -> Blue
- [ ] Achromatic (S=0) handled correctly

---

### T-3.19: Verify HEX Parsing

**File**: `engine/ui/styling/color.py`

**Description**: Ensure `Color.hex()` parses all formats.

**Acceptance Criteria**:
- [ ] `#RGB` shorthand parsed (expands to #RRGGBB)
- [ ] `#RRGGBB` parsed correctly
- [ ] `#RRGGBBAA` with alpha parsed
- [ ] Case insensitive (#fff = #FFF)
- [ ] Invalid hex throws/returns error

---

### T-3.20: Verify Blend Modes

**File**: `engine/ui/styling/color.py`

**Description**: Ensure all 12 blend modes produce correct results.

**Acceptance Criteria**:
- [ ] `normal` — returns top color
- [ ] `multiply` — darkens (white * any = any)
- [ ] `screen` — lightens (black screen any = any)
- [ ] `overlay` — contrast (combines multiply/screen)
- [ ] `darken` — returns darker color
- [ ] `lighten` — returns lighter color
- [ ] `color_dodge` — brightens
- [ ] `color_burn` — darkens shadows
- [ ] `hard_light` — sharp contrast
- [ ] `soft_light` — subtle contrast
- [ ] `difference` — inverts (same color = black)
- [ ] `exclusion` — similar to difference

---

### T-3.21: Verify WCAG Luminance Calculation

**File**: `engine/ui/styling/color.py`

**Description**: Ensure `get_luminance()` matches WCAG 2.1.

**Acceptance Criteria**:
- [ ] Black = 0
- [ ] White = 1
- [ ] Formula uses linearized RGB (gamma correction)
- [ ] Coefficients: 0.2126R + 0.7152G + 0.0722B

---

### T-3.22: Verify WCAG Contrast Ratio

**File**: `engine/ui/styling/color.py`

**Description**: Ensure `contrast_ratio()` calculates correctly.

**Acceptance Criteria**:
- [ ] Black on white = 21:1
- [ ] Same color = 1:1
- [ ] Order does not matter (symmetric)
- [ ] Formula: (L1 + 0.05) / (L2 + 0.05) where L1 > L2

---

### T-3.23: Verify Readability Check

**File**: `engine/ui/styling/color.py`

**Description**: Ensure `is_readable()` returns correct result.

**Acceptance Criteria**:
- [ ] AA level requires 4.5:1 for normal text
- [ ] AA level requires 3:1 for large text
- [ ] AAA level requires 7:1 for normal text
- [ ] AAA level requires 4.5:1 for large text

---

### T-3.24: Verify Color Palette Generation

**File**: `engine/ui/styling/color.py`

**Description**: Ensure harmony functions return correct colors.

**Acceptance Criteria**:
- [ ] `complementary()` returns color 180 degrees opposite
- [ ] `triadic()` returns 3 colors 120 degrees apart
- [ ] `analogous()` returns 3 colors ~30 degrees apart
- [ ] `split_complementary()` returns 2 colors 150 degrees from base

---

## Brush System Tasks

### T-3.25: Verify Solid Brush

**File**: `engine/ui/styling/brush.py`

**Description**: Ensure `SolidBrush` returns correct color.

**Acceptance Criteria**:
- [ ] `sample(any_position)` returns the solid color
- [ ] Alpha is preserved

---

### T-3.26: Verify Linear Gradient

**File**: `engine/ui/styling/brush.py`

**Description**: Ensure linear gradient interpolates correctly.

**Acceptance Criteria**:
- [ ] Position 0 returns first stop color
- [ ] Position 1 returns last stop color
- [ ] Middle positions interpolate between stops
- [ ] Angle affects gradient direction

---

### T-3.27: Verify Radial Gradient

**File**: `engine/ui/styling/brush.py`

**Description**: Ensure radial gradient samples correctly.

**Acceptance Criteria**:
- [ ] Center returns first stop color
- [ ] Edge returns last stop color
- [ ] Circular interpolation

---

### T-3.28: Verify Gradient Stops

**File**: `engine/ui/styling/brush.py`

**Description**: Ensure multi-stop gradients work.

**Acceptance Criteria**:
- [ ] 3+ stops handled
- [ ] Non-uniform stop positions work
- [ ] Color interpolation is smooth (per-channel linear)

---

### T-3.29: Verify Nine-Slice Regions

**File**: `engine/ui/styling/brush.py`

**Description**: Ensure `_calculate_nine_slice_regions()` partitions correctly.

**Acceptance Criteria**:
- [ ] 9 regions returned
- [ ] Corner regions match inset sizes
- [ ] Edge regions fill between corners
- [ ] Center region fills remaining space

---

### T-3.30: Verify Image Brush Tiling

**File**: `engine/ui/styling/brush.py`

**Description**: Ensure tiling modes work correctly.

**Acceptance Criteria**:
- [ ] `stretch` — Image fills entire area
- [ ] `tile` — Image repeats in both axes
- [ ] `tile_x` — Image repeats horizontally only
- [ ] `tile_y` — Image repeats vertically only

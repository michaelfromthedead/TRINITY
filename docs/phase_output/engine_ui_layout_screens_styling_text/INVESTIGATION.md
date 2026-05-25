# Archaeological Investigation: engine/ui (layout, screens, styling, text)

**Date**: 2026-05-22  
**Investigator**: Research Agent  
**Total Files**: 18  
**Total Lines**: ~14,991  

---

## Classification Summary

| Classification | Count | Percentage |
|----------------|-------|------------|
| REAL           | 18    | 100%       |
| STUB           | 0     | 0%         |

**Verdict**: All 18 files are REAL implementations with complete algorithms, proper data structures, and cross-module integration.

---

## Module Analysis

### Layout Module (6 files, ~4,394 lines)

#### grid.py (909 lines) - REAL
- **Purpose**: CSS Grid layout implementation
- **Key Classes**: `TrackSize`, `GridItem`, `GridLayout`
- **Algorithms**: 
  - `fr` unit calculation with proportional distribution
  - Row/column span handling
  - Auto-sizing algorithm for intrinsic track sizing
  - Gap handling between tracks
- **Evidence**: Complete `_calculate_track_sizes()` with min/max content, `resolve_fr_units()`, `_place_item()` with span support

#### flex.py (887 lines) - REAL
- **Purpose**: CSS Flexbox layout implementation
- **Key Classes**: `FlexItem`, `FlexLayout`, `FlexLine`
- **Algorithms**:
  - Flex grow/shrink distribution
  - Multi-line wrapping with `flex-wrap`
  - Main axis and cross axis alignment
  - `justify-content`, `align-items`, `align-content`
- **Evidence**: `_distribute_free_space()`, `_wrap_lines()`, `_align_cross_axis()`

#### hbox.py (676 lines) - REAL
- **Purpose**: Horizontal box layout (simplified flexbox)
- **Key Classes**: `HBoxItem`, `HBox`
- **Features**: Flex properties, spacing, alignment, justification
- **Evidence**: `_calculate_sizes()` with flex distribution, `_position_children()`

#### vbox.py (654 lines) - REAL
- **Purpose**: Vertical box layout (symmetric to hbox)
- **Key Classes**: `VBoxItem`, `VBox`
- **Features**: Same as hbox but vertical orientation
- **Evidence**: Shares architecture with hbox, proper axis swap

#### responsive.py (652 lines) - REAL
- **Purpose**: Responsive design system
- **Key Classes**: `Breakpoint`, `SafeAreaInsets`, `ResponsiveContainer`, `ResponsiveValue`
- **Features**:
  - Breakpoint management (mobile, tablet, desktop, etc.)
  - Safe area insets for notched devices
  - Container queries
  - Responsive value interpolation
- **Evidence**: `_find_active_breakpoint()`, `get_current_value()` with breakpoint matching

#### canvas.py (616 lines) - REAL
- **Purpose**: Absolute positioning layout
- **Key Classes**: `CanvasItem`, `CanvasLayout`
- **Features**:
  - Anchor and pivot points
  - Z-ordering with `_sort_by_z_index()`
  - Hit testing with `hit_test()`
  - Bounds calculation
- **Evidence**: `_apply_anchor()`, `_apply_pivot()`, collision detection

---

### Screens Module (3 files, ~2,659 lines)

#### transitions.py (1,023 lines) - REAL
- **Purpose**: Screen transition animations
- **Key Classes**: `Transition`, `FadeTransition`, `SlideTransition`, `ZoomTransition`, `CompositeTransition`
- **Easing Functions** (22 total):
  - `linear`, `ease_in_quad`, `ease_out_quad`, `ease_in_out_quad`
  - `ease_in_cubic`, `ease_out_cubic`, `ease_in_out_cubic`
  - `ease_in_quart`, `ease_out_quart`, `ease_in_out_quart`
  - `ease_in_quint`, `ease_out_quint`, `ease_in_out_quint`
  - `ease_in_sine`, `ease_out_sine`, `ease_in_out_sine`
  - `ease_in_expo`, `ease_out_expo`, `ease_in_out_expo`
  - `ease_in_back`, `ease_out_back`, `ease_in_out_back`
  - `ease_in_bounce`, `ease_out_bounce`, `ease_in_out_bounce`
- **Evidence**: Mathematical formulas for each easing, `CompositeTransition.compose()` for parallel/sequential

#### screen_stack.py (994 lines) - REAL
- **Purpose**: Screen navigation and history management
- **Key Classes**: `ScreenStack`, `ScreenCache`, `NavigationHistory`
- **Features**:
  - Stack operations (push, pop, replace)
  - History tracking with back/forward navigation
  - LRU cache for screen instances
  - Modal screen support
  - Deep linking
- **Evidence**: `_apply_transition()`, `_cache_eviction()` with LRU, `can_go_back()`, `navigate_to_deep_link()`

#### screen.py (642 lines) - REAL
- **Purpose**: Base screen class and lifecycle
- **Key Classes**: `Screen`, `ScreenParams`, `ScreenResult`, `ScreenState`
- **Lifecycle Methods**: `on_enter()`, `on_exit()`, `on_pause()`, `on_resume()`, `on_back_pressed()`
- **Evidence**: State machine transitions, parameter passing, result handling

---

### Styling Module (4 files, ~3,345 lines)

#### style.py (947 lines) - REAL
- **Purpose**: CSS-like styling system
- **Key Classes**: `Style`, `StateStyles`, `StyleSelector`, `Stylesheet`
- **Features**:
  - 40+ style properties (margin, padding, border, background, etc.)
  - State-based styling (hover, pressed, disabled, focused)
  - CSS selector matching (class, id, type, pseudo-class)
  - Stylesheet cascading and specificity
- **Evidence**: `_calculate_specificity()`, `match()` selector logic, `merge()` for cascading

#### theme.py (911 lines) - REAL
- **Purpose**: Design token system
- **Key Classes**: `Theme`, `ColorPalette`, `Typography`, `Spacing`, `Shadows`, `BorderRadii`
- **Built-in Themes**: `light`, `dark`, `high_contrast`
- **Features**:
  - Design tokens for colors, typography, spacing
  - Shadow definitions with blur and spread
  - Border radius presets
  - Theme inheritance and overrides
- **Evidence**: Complete token definitions, `get_color()`, `get_typography()`, `derive_theme()`

#### color.py (901 lines) - REAL
- **Purpose**: Color manipulation and utilities
- **Key Class**: `Color`
- **Blend Modes** (12 total):
  - `normal`, `multiply`, `screen`, `overlay`
  - `darken`, `lighten`, `color_dodge`, `color_burn`
  - `hard_light`, `soft_light`, `difference`, `exclusion`
- **Color Spaces**: RGBA, HSL, HSV, HEX
- **WCAG Compliance**: `contrast_ratio()`, `is_readable()`, `get_luminance()`
- **Palette Generation**: `complementary()`, `triadic()`, `analogous()`, `split_complementary()`
- **Evidence**: Mathematical blend mode formulas, color space conversions, contrast calculations

#### brush.py (586 lines) - REAL
- **Purpose**: Fill patterns for painting
- **Key Classes**: `Brush`, `SolidBrush`, `GradientBrush`, `ImageBrush`, `NineSliceBrush`
- **Gradient Types**: `linear`, `radial`, `angular`, `diamond`
- **Features**: Gradient stops, tiling modes, nine-slice scaling
- **Evidence**: `_sample_gradient()`, `_calculate_nine_slice_regions()`

---

### Text Module (5 files, ~4,220 lines)

#### rich_text.py (1,030 lines) - REAL
- **Purpose**: Rich text markup and rendering
- **Key Classes**: `RichText`, `TextRun`, `InlineImage`, `ClickableLink`, `RichTextBuilder`
- **Markup Tags**: `[b]`, `[i]`, `[u]`, `[s]`, `[color]`, `[size]`, `[font]`, `[link]`, `[img]`
- **Features**:
  - BBCode-style markup parsing
  - Inline images with alignment
  - Clickable links with hover states
  - Builder pattern for programmatic construction
- **Evidence**: `_parse_markup()`, `_tokenize()`, `_build_runs()`

#### ime.py (878 lines) - REAL
- **Purpose**: Input Method Editor for complex scripts
- **Key Classes**: `IMEManager`, `CompositionState`, `CandidateWindow`
- **Features**:
  - Composition string handling
  - Candidate selection with keyboard navigation
  - Cursor positioning within composition
  - Language-specific input handling (CJK, etc.)
- **Evidence**: `_handle_composition_update()`, `_show_candidates()`, `_commit_composition()`

#### localization.py (877 lines) - REAL
- **Purpose**: Internationalization and localization
- **Key Classes**: `LocalizationManager`, `PluralRule`, `LocalizedString`
- **Plural Rules** (6 languages with CLDR compliance):
  - English, Russian, Arabic, Japanese, Polish, French
- **Features**:
  - String interpolation with named parameters
  - Pluralization with cardinal/ordinal forms
  - RTL language detection
  - Locale fallback chain
- **Evidence**: `_get_plural_form()` with CLDR rules, `_detect_rtl()`, `format()`

#### text_renderer.py (777 lines) - REAL
- **Purpose**: Text layout and rendering
- **Key Classes**: `TextRenderer`, `TextMeasurement`, `LineBreaker`, `TextShaper`, `GlyphCache`
- **Features**:
  - Unicode-aware line breaking (UAX #14)
  - Text shaping for complex scripts
  - Glyph caching with LRU eviction
  - Text measurement and bounding box calculation
- **Evidence**: `_find_break_opportunities()`, `_shape_run()`, `_cache_glyph()`

#### font.py (658 lines) - REAL
- **Purpose**: Font management and rendering
- **Key Classes**: `FontManager`, `FontFamily`, `Font`, `FontAtlas`, `SDFFont`, `FontFallbackChain`
- **Features**:
  - Font family with weight/style variants
  - Glyph atlas packing
  - Signed Distance Field (SDF) fonts for scaling
  - Fallback chain for missing glyphs
- **Evidence**: `_pack_atlas()`, `_generate_sdf()`, `_find_fallback_glyph()`

---

## Key Algorithms Found

### Layout Algorithms
1. **CSS Grid Track Sizing**: `fr` unit resolution, min/max content, auto sizing
2. **Flexbox Distribution**: Grow/shrink with frozen items, multi-line wrapping
3. **Hit Testing**: Point-in-bounds with z-order traversal

### Animation Algorithms
4. **22 Easing Functions**: Quadratic through bounce with mathematical precision
5. **Composite Transitions**: Parallel and sequential composition

### Color Algorithms
6. **12 Blend Modes**: Porter-Duff operations, soft light with piecewise formula
7. **WCAG Contrast**: Relative luminance and contrast ratio per WCAG 2.1
8. **HSL/HSV Conversion**: Bidirectional color space transforms

### Text Algorithms
9. **Unicode Line Breaking**: UAX #14 implementation for proper text wrapping
10. **Text Shaping**: Complex script handling with contextual substitution
11. **CLDR Pluralization**: Language-specific plural rules (6 languages)
12. **SDF Font Rendering**: Signed distance field generation for resolution independence

---

## Evidence Summary

| Evidence Type | Count | Examples |
|---------------|-------|----------|
| Complete algorithms | 12+ | Grid sizing, flexbox distribution, blend modes |
| Data structures | 25+ | ScreenStack, FontAtlas, GlyphCache, StyleSheet |
| Cross-module imports | 15+ | Color used in Style, Font used in TextRenderer |
| Validation logic | 40+ | Bounds checking, type validation, null guards |
| Performance optimizations | 8+ | LRU caches, atlas packing, glyph caching |
| Mathematical formulas | 30+ | Easing functions, blend modes, color conversion |

---

## Conclusion

All 18 files in the engine/ui subsystem (layout, screens, styling, text) are **REAL implementations** with:

- **Complete algorithms**: CSS Grid, Flexbox, blend modes, Unicode line breaking
- **Proper data structures**: Stacks, caches, atlases, state machines
- **Cross-module integration**: Consistent architecture across all modules
- **Performance awareness**: Caching strategies, lazy evaluation, atlas packing
- **Standards compliance**: WCAG contrast, CLDR pluralization, UAX #14 line breaking

No stub files or placeholder implementations were found in this module set.

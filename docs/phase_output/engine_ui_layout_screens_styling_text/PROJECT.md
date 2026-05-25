# PROJECT: Engine UI — Layout, Screens, Styling, Text

**Investigation Source**: engine_ui_layout_screens_styling_text.md  
**Date**: 2026-05-22  
**Classification**: 18 files, 100% REAL implementations (~14,991 lines)

---

## Scope

This project encompasses the complete UI subsystem of the TRINITY engine, covering four interconnected modules:

1. **Layout Module** (6 files, ~4,394 lines) — CSS-compatible layout algorithms
2. **Screens Module** (3 files, ~2,659 lines) — Screen navigation and transitions
3. **Styling Module** (4 files, ~3,345 lines) — CSS-like styling and theming
4. **Text Module** (5 files, ~4,220 lines) — Font management, rendering, localization

---

## Goals

### Primary Goals

1. Maintain and extend the CSS Grid layout implementation with `fr` unit calculation, row/column spans, and auto-sizing
2. Preserve the Flexbox implementation with grow/shrink distribution and multi-line wrapping
3. Support responsive design with breakpoints, safe area insets, and container queries
4. Provide complete screen lifecycle management with stack operations and transition animations
5. Implement CSS-like styling with selector matching and specificity calculation
6. Deliver theme system with design tokens for colors, typography, spacing, shadows
7. Ensure color system supports 12 blend modes and WCAG compliance
8. Enable rich text rendering with markup parsing, inline images, and clickable links
9. Support Input Method Editor (IME) for complex scripts (CJK)
10. Provide internationalization with CLDR-compliant pluralization for 6 languages

### Cross-Cutting Goals

- Maintain cross-module integration (Color used in Style, Font used in TextRenderer)
- Preserve performance optimizations (LRU caches, atlas packing, glyph caching)
- Ensure standards compliance (WCAG, CLDR, UAX #14)

---

## Constraints

### Technical Constraints

1. **Python 3.13** — Project targets statically-linked Python 3.13 interpreter
2. **Existing Architecture** — Must integrate with engine's existing module structure
3. **Performance** — Layout algorithms must be efficient for real-time UI updates
4. **Memory** — Caching strategies must balance performance with memory usage

### Standards Compliance

1. **WCAG 2.1** — Color contrast calculations must meet accessibility requirements
2. **CLDR** — Pluralization rules must be compliant for supported languages
3. **UAX #14** — Unicode line breaking must follow Unicode standard
4. **CSS Specifications** — Grid and Flexbox should approximate CSS behavior

---

## Module Inventory

### Layout Module

| File | Lines | Purpose | Key Classes |
|------|-------|---------|-------------|
| grid.py | 909 | CSS Grid layout | `TrackSize`, `GridItem`, `GridLayout` |
| flex.py | 887 | CSS Flexbox layout | `FlexItem`, `FlexLayout`, `FlexLine` |
| hbox.py | 676 | Horizontal box layout | `HBoxItem`, `HBox` |
| vbox.py | 654 | Vertical box layout | `VBoxItem`, `VBox` |
| responsive.py | 652 | Responsive design system | `Breakpoint`, `SafeAreaInsets`, `ResponsiveContainer` |
| canvas.py | 616 | Absolute positioning | `CanvasItem`, `CanvasLayout` |

### Screens Module

| File | Lines | Purpose | Key Classes |
|------|-------|---------|-------------|
| transitions.py | 1,023 | Screen transition animations | `Transition`, `FadeTransition`, `SlideTransition`, `ZoomTransition` |
| screen_stack.py | 994 | Screen navigation | `ScreenStack`, `ScreenCache`, `NavigationHistory` |
| screen.py | 642 | Base screen class | `Screen`, `ScreenParams`, `ScreenResult`, `ScreenState` |

### Styling Module

| File | Lines | Purpose | Key Classes |
|------|-------|---------|-------------|
| style.py | 947 | CSS-like styling | `Style`, `StateStyles`, `StyleSelector`, `Stylesheet` |
| theme.py | 911 | Design tokens | `Theme`, `ColorPalette`, `Typography`, `Spacing` |
| color.py | 901 | Color manipulation | `Color` |
| brush.py | 586 | Fill patterns | `Brush`, `SolidBrush`, `GradientBrush`, `ImageBrush` |

### Text Module

| File | Lines | Purpose | Key Classes |
|------|-------|---------|-------------|
| rich_text.py | 1,030 | Rich text markup | `RichText`, `TextRun`, `InlineImage`, `RichTextBuilder` |
| ime.py | 878 | Input Method Editor | `IMEManager`, `CompositionState`, `CandidateWindow` |
| localization.py | 877 | i18n/l10n | `LocalizationManager`, `PluralRule`, `LocalizedString` |
| text_renderer.py | 777 | Text layout/rendering | `TextRenderer`, `LineBreaker`, `TextShaper`, `GlyphCache` |
| font.py | 658 | Font management | `FontManager`, `FontFamily`, `FontAtlas`, `SDFFont` |

---

## Key Algorithms

| Algorithm | Location | Purpose |
|-----------|----------|---------|
| CSS Grid Track Sizing | grid.py | `fr` unit resolution, min/max content |
| Flexbox Distribution | flex.py | Grow/shrink with frozen items |
| 22 Easing Functions | transitions.py | Animation curves (quad through bounce) |
| 12 Blend Modes | color.py | Porter-Duff operations, soft light |
| WCAG Contrast | color.py | Relative luminance, contrast ratio |
| Unicode Line Breaking | text_renderer.py | UAX #14 implementation |
| CLDR Pluralization | localization.py | Language-specific plural rules |
| SDF Font Rendering | font.py | Signed distance field generation |

---

## Acceptance Criteria

### Layout Module
- [ ] Grid layout correctly resolves `fr` units with proportional distribution
- [ ] Flexbox handles grow/shrink with multi-line wrapping
- [ ] HBox/VBox provide simplified flex behavior
- [ ] Responsive container responds to breakpoint changes
- [ ] Canvas layout supports z-ordering and hit testing

### Screens Module
- [ ] Screen lifecycle methods called in correct order
- [ ] Screen stack supports push/pop/replace operations
- [ ] Transitions animate smoothly with all 22 easing functions
- [ ] Navigation history enables back/forward navigation
- [ ] Deep linking routes to correct screens

### Styling Module
- [ ] Style selectors match with correct specificity
- [ ] Themes provide consistent design tokens
- [ ] Color blend modes produce correct results
- [ ] WCAG contrast calculations are accurate
- [ ] Brushes render gradients and nine-slice correctly

### Text Module
- [ ] Rich text parses BBCode-style markup
- [ ] IME handles CJK input correctly
- [ ] Localization applies correct plural forms
- [ ] Text renderer breaks lines per UAX #14
- [ ] Font atlas packs glyphs efficiently

---

## Phases

| Phase | Module | Focus |
|-------|--------|-------|
| 1 | Layout | Grid, Flexbox, HBox, VBox, Responsive, Canvas |
| 2 | Screens | Screen lifecycle, stack navigation, transitions |
| 3 | Styling | Style system, themes, colors, brushes |
| 4 | Text | Font management, rendering, rich text, localization, IME |

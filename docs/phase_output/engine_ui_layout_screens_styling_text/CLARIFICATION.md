# CLARIFICATION: Engine UI — Layout, Screens, Styling, Text

---

## Philosophical Framing

### The Nature of UI Systems

A UI system is not merely a collection of rendering primitives — it is a **contract between intent and presentation**. The TRINITY UI subsystem embodies this philosophy through four interconnected layers:

1. **Layout** — The spatial contract (where things are)
2. **Screens** — The temporal contract (what is shown when)
3. **Styling** — The aesthetic contract (how things look)
4. **Text** — The linguistic contract (what is communicated)

Each layer serves a distinct purpose while remaining deeply integrated with the others. A styled text element within a screen requires layout to position it, styling to decorate it, and text rendering to display it. The modules are not islands — they form a unified system.

---

## Design Rationale

### Why CSS-Like Semantics?

The choice to implement CSS-compatible layout algorithms (Grid, Flexbox) is deliberate:

1. **Familiar Mental Model** — Developers understand CSS; no relearning required
2. **Proven Specifications** — CSS specifications have been refined over decades
3. **Tooling Compatibility** — Design tools can export to familiar concepts
4. **Gradual Complexity** — Simple cases remain simple; complex layouts are possible

The implementation does not aim for 100% CSS compliance — it aims for **conceptual compatibility** where the mental model transfers, even if edge cases differ.

### Why 22 Easing Functions?

Animation is not decoration — it is **communication**. The presence of 22 easing functions (quadratic through bounce) reflects an understanding that:

- `ease_in` communicates initiation
- `ease_out` communicates completion
- `ease_in_out` communicates continuity
- `bounce` communicates physicality
- `back` communicates anticipation

Each easing function encodes a semantic meaning that designers can leverage to communicate state changes to users.

### Why Blend Modes?

The 12 blend modes (`normal`, `multiply`, `screen`, `overlay`, etc.) exist because:

1. **Visual Composition** — UI elements layer; blending controls how they combine
2. **Dynamic Effects** — Hover states, selections, and highlights use blending
3. **Design Fidelity** — Designers expect Photoshop-like compositing

The implementation follows Porter-Duff algebra for mathematical correctness.

### Why WCAG Compliance?

Accessibility is not optional. The inclusion of `contrast_ratio()`, `is_readable()`, and `get_luminance()` reflects a commitment to:

1. **Inclusive Design** — UIs must be usable by people with visual impairments
2. **Legal Compliance** — Many jurisdictions require WCAG compliance
3. **Automated Validation** — Contrast can be checked programmatically

### Why CLDR Pluralization?

Languages are not English. The 6-language plural rule set (English, Russian, Arabic, Japanese, Polish, French) exists because:

- English has 2 forms (singular, plural)
- Russian has 4 forms
- Arabic has 6 forms
- Japanese has 1 form
- Polish has 4 forms with complex rules

CLDR compliance ensures that "1 item" vs "2 items" works correctly in any supported language.

### Why SDF Fonts?

Signed Distance Field fonts solve a fundamental problem: **resolution independence**. Traditional bitmap fonts blur when scaled. SDF fonts remain crisp at any size because:

1. The distance field encodes glyph edges mathematically
2. Fragment shaders can reconstruct edges at any resolution
3. GPU rendering is efficient

This enables high-quality text on screens of varying pixel densities.

---

## Architectural Decisions

### State Machines for Screens

The `ScreenState` class implements a finite state machine with transitions:

```
INACTIVE -> ENTERING -> ACTIVE -> EXITING -> INACTIVE
                |         |
                v         v
             PAUSED  <-> RESUMED
```

This formalization ensures:
1. Lifecycle methods are called in correct order
2. Invalid transitions are prevented
3. Resource management is predictable

### LRU Caches for Performance

Multiple modules employ LRU (Least Recently Used) caching:

- `ScreenCache` — Retains recently used screen instances
- `GlyphCache` — Retains recently rendered glyphs
- Font atlas packing — Prioritizes frequently used glyphs

The rationale: UI elements exhibit temporal locality. Recently used elements are likely to be used again. Caching amortizes expensive operations (screen construction, glyph rasterization).

### Selector Specificity

CSS selector specificity is implemented as a tuple `(inline, id_count, class_count, type_count)`. This mirrors browser behavior and ensures predictable cascade resolution.

The design choice to support selectors at all (rather than direct styling) enables:
1. **Separation of Concerns** — Style definitions separate from structure
2. **Reusability** — Stylesheets can be shared
3. **Theming** — Themes can override selectors without modifying components

### Theme Inheritance

The `derive_theme()` method enables theme extension:

```python
dark_theme = light_theme.derive_theme(
    colors={"background": "#1a1a1a", "foreground": "#ffffff"}
)
```

This supports:
1. **Brand Customization** — Derive from base theme, override specifics
2. **Accessibility Modes** — High contrast themes derive from standard themes
3. **User Preferences** — Per-user customizations

---

## Cross-Module Integration

The investigation identified 15+ cross-module imports, reflecting deep integration:

| Consumer | Provider | Purpose |
|----------|----------|---------|
| Style | Color | Background colors, border colors |
| TextRenderer | Font | Glyph lookup, font metrics |
| RichText | TextRenderer | Layout of styled text runs |
| Brush | Color | Gradient stops, solid fills |
| Screen | Layout | Positioning of screen contents |
| Theme | Color | Color palette definitions |
| StyleSelector | Style | Matching logic |

This integration is intentional — UI is holistic.

---

## Performance Considerations

The investigation identified 8+ performance optimizations:

1. **LRU Caches** — Screen, glyph, and style caches
2. **Atlas Packing** — Font glyphs packed into GPU textures
3. **Lazy Evaluation** — Responsive values computed on demand
4. **Frozen Item Tracking** — Flexbox avoids reprocessing frozen items
5. **Z-Order Sorting** — Canvas sorts once, not per hit test
6. **Glyph Caching** — Shaped text cached by run
7. **Selector Indexing** — Stylesheet indexes selectors by type
8. **Track Size Memoization** — Grid caches track calculations

These optimizations reflect an understanding that UI rendering is frame-rate sensitive.

---

## What This System Is Not

1. **Not a Browser** — CSS compatibility is approximate, not exhaustive
2. **Not a Design Tool** — It renders designs, not creates them
3. **Not Platform-Specific** — It abstracts platform differences
4. **Not a Text Editor** — Rich text is for display, not editing (IME is input)

---

## Open Questions

The investigation does not address:

1. **GPU Backend** — How are colors, glyphs, and layouts rendered to GPU?
2. **Event Handling** — How do clickable links and hit testing connect to input?
3. **Animation System** — Easing functions exist; where is the animator?
4. **Asset Loading** — How are fonts and images loaded?

These may be addressed in other modules or remain to be implemented.

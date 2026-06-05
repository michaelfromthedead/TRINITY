# PHASE 4 ARCHITECTURE: Text Module

---

## Overview

The Text Module provides font management, text rendering, rich text markup, localization, and input method support. It comprises 5 files (~4,220 lines) implementing a comprehensive text system.

---

## Component Architecture

### Font Management (font.py — 658 lines)

**Purpose**: Font loading, atlas packing, and glyph rendering.

**Classes**:
- `FontManager` — Central font registry and loader
- `FontFamily` — Collection of font weights/styles
- `Font` — Single font with specific weight/style
- `FontAtlas` — GPU texture containing packed glyphs
- `SDFFont` — Signed Distance Field font for resolution independence
- `FontFallbackChain` — Fallback sequence for missing glyphs

**Font Family Structure**:
```
FontFamily("Roboto")
├── Font("Roboto-Regular", weight=400, style=normal)
├── Font("Roboto-Bold", weight=700, style=normal)
├── Font("Roboto-Italic", weight=400, style=italic)
└── Font("Roboto-BoldItalic", weight=700, style=italic)
```

**Atlas Packing** (`_pack_atlas()`):
```
+---+---+---+---+
| A | B | C | D |  <- Glyphs packed into texture
+---+---+---+---+
| E | F | G | H |
+---+---+---+---+
| I | J |       |  <- Unused space
+---+---+-------+
```

Algorithm:
1. Sort glyphs by height (descending)
2. Pack row by row using shelf algorithm
3. Track UV coordinates for each glyph
4. Upload to GPU texture

**SDF Font Generation** (`_generate_sdf()`):
- Rasterize glyph at high resolution
- Compute signed distance from each pixel to edge
- Encode distance as grayscale value
- GPU shader reconstructs crisp edges at any scale

**Fallback Chain**:
```python
chain = FontFallbackChain([
    Font("Roboto"),      # Primary
    Font("NotoSansCJK"), # CJK fallback
    Font("NotoEmoji"),   # Emoji fallback
    Font("LastResort")   # Guaranteed to have all Unicode
])

glyph = chain._find_fallback_glyph(codepoint)
```

---

### Text Rendering (text_renderer.py — 777 lines)

**Purpose**: Text layout, shaping, and measurement.

**Classes**:
- `TextRenderer` — Main rendering coordinator
- `TextMeasurement` — Width, height, baseline of text
- `LineBreaker` — Unicode line breaking (UAX #14)
- `TextShaper` — Complex script shaping
- `GlyphCache` — LRU cache for shaped text runs

**Unicode Line Breaking** (`_find_break_opportunities()`):

Per UAX #14, each character has a break class:
| Class | Meaning | Example |
|-------|---------|---------|
| `SP` | Space | Always break after |
| `BA` | Break After | Hyphens, slashes |
| `BB` | Break Before | Currency symbols (some) |
| `AL` | Alphabetic | Letters (no break inside word) |
| `NU` | Numeric | Digits (no break in number) |
| `CJ` | Conditional Japanese | Complex rules |

Algorithm:
1. Classify each character
2. Apply pair table rules
3. Return list of break opportunities
4. Line breaker picks optimal breaks

**Text Shaping** (`_shape_run()`):
- Handles complex scripts (Arabic, Devanagari, Thai)
- Applies contextual substitution (initial/medial/final forms)
- Handles ligatures (fi, fl, ffi)
- Adjusts glyph positioning (kerning, mark positioning)

**Glyph Cache**:
```python
# Cache key: (text, font, size, features)
# Cache value: list of positioned glyphs

cached = glyph_cache.get(key)
if cached:
    return cached

shaped = self._shape_run(text, font, size)
glyph_cache.put(key, shaped)
return shaped
```

**Text Measurement**:
```python
measurement = renderer.measure("Hello, World!", font, size)
# measurement.width = total advance width
# measurement.height = line height
# measurement.ascent = distance from baseline to top
# measurement.descent = distance from baseline to bottom
```

---

### Rich Text (rich_text.py — 1,030 lines)

**Purpose**: Markup parsing and styled text rendering.

**Classes**:
- `RichText` — Container for styled text
- `TextRun` — Span of text with uniform style
- `InlineImage` — Image embedded in text
- `ClickableLink` — Hyperlink with hover/click handling
- `RichTextBuilder` — Programmatic construction

**Markup Tags**:
| Tag | Effect | Example |
|-----|--------|---------|
| `[b]` | Bold | `[b]bold[/b]` |
| `[i]` | Italic | `[i]italic[/i]` |
| `[u]` | Underline | `[u]underline[/u]` |
| `[s]` | Strikethrough | `[s]struck[/s]` |
| `[color=X]` | Text color | `[color=#FF0000]red[/color]` |
| `[size=X]` | Font size | `[size=24]large[/size]` |
| `[font=X]` | Font family | `[font=Mono]code[/font]` |
| `[link=X]` | Hyperlink | `[link=http://...]click[/link]` |
| `[img=X]` | Inline image | `[img=icon.png]` |

**Markup Parsing** (`_parse_markup()`):
```
Input: "Hello [b]bold[/b] world"

_tokenize() ->
  ["Hello ", "[b]", "bold", "[/b]", " world"]

_build_runs() ->
  [TextRun("Hello ", style=default),
   TextRun("bold", style=bold),
   TextRun(" world", style=default)]
```

**Builder Pattern**:
```python
text = RichTextBuilder() \
    .text("Hello ") \
    .bold().text("bold").end_bold() \
    .text(" and ") \
    .color("#FF0000").text("red").end_color() \
    .build()
```

**Inline Images**:
```python
InlineImage(
    source="icon.png",
    width=16, height=16,
    alignment="baseline"  # or "top", "middle", "bottom"
)
```

**Clickable Links**:
```python
link = ClickableLink(
    url="https://example.com",
    text="Click here",
    style=Style(color=Color.blue(), underline=True),
    hover_style=Style(color=Color.dark_blue())
)
```

---

### Localization (localization.py — 877 lines)

**Purpose**: Internationalization and string localization.

**Classes**:
- `LocalizationManager` — Central translation registry
- `PluralRule` — Language-specific pluralization
- `LocalizedString` — String with interpolation

**String Loading**:
```python
# strings/en.json
{
    "greeting": "Hello, {name}!",
    "items": "{count} item|{count} items",
    "logout": "Sign out"
}

# strings/fr.json
{
    "greeting": "Bonjour, {name}!",
    "items": "{count} article|{count} articles",
    "logout": "Se deconnecter"
}

manager = LocalizationManager()
manager.load_locale("en", "strings/en.json")
manager.load_locale("fr", "strings/fr.json")
manager.set_locale("fr")
```

**String Interpolation**:
```python
localized = manager.get("greeting")
result = localized.format(name="Claude")
# "Bonjour, Claude!"
```

**Pluralization** (CLDR Compliant):

| Language | Forms | Rule |
|----------|-------|------|
| English | 2 | n=1 ? one : other |
| Russian | 4 | Complex mod-based rules |
| Arabic | 6 | zero, one, two, few, many, other |
| Japanese | 1 | Always same form |
| Polish | 4 | Complex mod-based rules |
| French | 2 | n=0,1 ? one : other |

```python
# English: "1 item" / "2 items"
# Russian: "1 элемент" / "2 элемента" / "5 элементов" / "21 элемент"

plural_form = manager._get_plural_form("ru", count)
```

**RTL Detection** (`_detect_rtl()`):
```python
# Check first strong character
if manager._detect_rtl(text):
    # Apply RTL layout direction
```

**Locale Fallback Chain**:
```
Request: "fr-CA" (Canadian French)
Fallback: fr-CA -> fr -> en (base)
```

---

### Input Method Editor (ime.py — 878 lines)

**Purpose**: Complex script input handling (CJK, etc.).

**Classes**:
- `IMEManager` — Platform IME integration
- `CompositionState` — Current composition string
- `CandidateWindow` — Candidate selection UI

**Composition Flow**:
```
User types: "ni hao" (Pinyin for Chinese)

1. Keystrokes -> CompositionState
2. IME generates candidates: ["你好", "尼号", "泥好", ...]
3. CandidateWindow shows candidates
4. User selects with 1-9 or arrows
5. Selected candidate committed to text
```

**Composition State**:
```python
state = CompositionState(
    text="nihao",           # Raw input
    cursor=5,               # Cursor position in composition
    selection=(0, 5),       # Selected portion
    candidates=["你好", ...]
)
```

**Candidate Window**:
```python
window = CandidateWindow(
    candidates=state.candidates,
    selected_index=0,
    position=(x, y),  # Near text cursor
    visible=True
)
```

**IME Events**:
| Event | Handler | Result |
|-------|---------|--------|
| Composition start | `on_composition_start()` | Enter composition mode |
| Composition update | `_handle_composition_update()` | Update preview text |
| Candidate change | `_show_candidates()` | Update candidate window |
| Commit | `_commit_composition()` | Insert final text |
| Cancel | `on_composition_cancel()` | Discard composition |

**Cursor Handling**:
```python
# Cursor within composition
state.move_cursor(delta=-1)  # Move left in composition
state.move_cursor(delta=+1)  # Move right in composition

# This affects which part of composition is converted
```

---

## Module Dependencies

```
font.py           --(standalone, may depend on image/texture types)
text_renderer.py  --> font.py (imports Font, FontAtlas)
rich_text.py      --> text_renderer.py (imports TextRenderer)
                  --> styling (imports Style, Color)
localization.py   --(standalone)
ime.py            --(standalone, may integrate with input system)
```

---

## Integration Points

1. **Styling Module** — Rich text uses Color and Style
2. **Layout Module** — Text measurement for layout sizing
3. **Render System** — Font atlases are GPU textures
4. **Input System** — IME integrates with keyboard events
5. **Widget System** — Text widgets use all text components

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Glyph lookup | O(1) | Hash map by codepoint |
| Atlas packing | O(n log n) | Sort + linear pack |
| Text shaping | O(n) | n = characters |
| Line breaking | O(n) | Single pass with lookahead |
| Glyph cache hit | O(1) | Hash lookup |
| Plural form | O(1) | Rule evaluation |

---

## Design Decisions

1. **SDF Fonts** — Resolution independence for UI scaling
2. **Atlas Packing** — Minimize GPU texture switches
3. **UAX #14** — Standards-compliant line breaking
4. **CLDR Pluralization** — Correct plurals in all languages
5. **BBCode Markup** — Familiar syntax, not XML/HTML
6. **IME Abstraction** — Platform-independent composition
7. **Fallback Chains** — Graceful degradation for missing glyphs

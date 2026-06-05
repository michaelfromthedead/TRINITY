# Investigation: engine/ui/text

## Summary
The UI text module provides a comprehensive text rendering and localization infrastructure with real algorithmic implementations for font management, text shaping, line breaking, rich text parsing, localization with pluralization rules, and IME (Input Method Editor) support. All five files contain substantial, working code with proper data structures, algorithms, and integration points - no stubs or placeholder implementations.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 141 | REAL | Full export of 40+ classes/enums from 5 modules |
| `font.py` | 659 | REAL | Font management, atlases, SDF fonts, fallback chains |
| `text_renderer.py` | 778 | REAL | Text measurement, line breaking, shaping, layout |
| `rich_text.py` | 1031 | REAL | BBCode-style markup parser, inline images, links |
| `localization.py` | 878 | REAL | String tables, pluralization (6+ languages), RTL |
| `ime.py` | 879 | REAL | IME state machine, composition, candidate lists |

**Total: 4,366 lines of code**

## Text Components

### Font System (`font.py`)
- `Font` - Font properties with size, weight, style, metrics
- `FontFamily` - Collection of font variants with closest-match lookup
- `FontManager` - Loading, caching, atlas creation, fallback chains
- `FontAtlas` - Texture atlas for bitmap glyph rendering
- `SDFFont` - Signed Distance Field fonts for resolution-independent rendering
- `GlyphMetrics` - Per-glyph dimensions and UV coordinates
- `FontFallbackChain` - Multi-font glyph lookup

### Text Rendering (`text_renderer.py`)
- `TextRenderer` - Main interface: measure, layout, shape, cursor positioning
- `TextMeasurement` - Width, height, bounds, line count
- `LineBreaker` - Unicode-aware line breaking (UAX #14 simplified)
- `TextShaper` - Ligatures, RTL reversal, glyph positioning
- `GlyphCache` - LRU cache for glyph metrics
- `TextAlignment` - LEFT, CENTER, RIGHT, JUSTIFY
- `TextOverflow` - CLIP, ELLIPSIS, WRAP, SHRINK

### Rich Text (`rich_text.py`)
- `RichTextParser` - BBCode-style markup: `[b]`, `[i]`, `[color=X]`, `[size=X]`, `[link=X]`
- `TextSpan` - Container for runs, images, links
- `TextRun` - Styled text segment with start/end indices
- `RichTextStyle` - bold, italic, underline, strikethrough, color, size, font
- `InlineImage` - Embedded images with dimensions
- `ClickableLink` - URLs with hover style and callback
- `RichTextBuilder` - Programmatic rich text construction

### Localization (`localization.py`)
- `LocalizationManager` - Singleton for language switching and string lookup
- `StringTable` - JSON-based string storage per language
- `Language` - Code, name, direction, plural rule, fallback
- `PluralRule` - CLDR-compliant pluralization (English, Russian, Arabic, French, Japanese, Polish)
- `LocalizedString` - Template with parameter substitution `{name}`, `{count}`
- `TextDirection` - LTR/RTL with auto-detection

### IME Support (`ime.py`)
- `IMEHandler` - State machine for composition lifecycle
- `IMEState` - INACTIVE, ACTIVE, COMPOSING, SELECTING
- `CompositionString` - Text, cursor, selection, clauses
- `CandidateList` - Paginated candidate navigation
- `IMEEvent` - Composition start/update/end, candidate show/hide/select
- `IMEWindowPosition` - Positioning for composition and candidate windows

## Implementation

- Real font system? **yes** - FontManager with loading, caching, atlas generation, SDF support, fallback chains
- Real text layout? **yes** - TextRenderer with measurement, LineBreaker with Unicode-aware word tokenization, TextShaper with ligatures and RTL
- Real rich text? **yes** - Full BBCode-style parser with regex tokenization, style stacking, inline images, links, validation
- Real localization? **yes** - String tables, JSON loading, 6 pluralization rules (CLDR-compliant), RTL language detection
- Real IME support? **yes** - Complete state machine, composition editing, candidate list navigation, keyboard handling

## Verdict
**REAL IMPLEMENTATION**

All five modules contain complete, functional code with:
- Proper class hierarchies and dataclasses
- Real algorithms (line breaking, text shaping, markup parsing)
- Comprehensive error handling and validation
- Type hints throughout
- Extensive documentation strings
- Integration points ready for platform font loading and rendering backends

## Evidence

### Line Breaking Algorithm (text_renderer.py:166-266)
```python
def break_text(self, text: str, max_width: float, measure_func: callable, font: Font) -> LineBreakResult:
    # Splits by mandatory breaks first, then word-wraps each paragraph
    paragraphs = self._split_mandatory_breaks(text)
    for paragraph in paragraphs:
        para_lines = self._break_paragraph(paragraph, max_width, measure_func, font)
        lines.extend(para_lines.lines)
```

### Rich Text Parser (rich_text.py:403-543)
```python
def parse(self, text: str) -> TextSpan:
    # Full regex-based tokenization with style stacking
    _TAG_PATTERN = re.compile(r"\[(/?)(\w+)(?:=([^\]]+))?\]", re.IGNORECASE)
    # Handles: [b], [i], [u], [s], [color=X], [size=X], [font=X], [link=X], [img=X], [icon=X]
    # Supports nested tags, self-closing tags, validation in strict mode
```

### Pluralization Rules (localization.py:54-133)
```python
@staticmethod
def russian() -> PluralRule:
    def _get_category(n: int | float) -> PluralCategory:
        n10 = n % 10
        n100 = n % 100
        if n10 == 1 and n100 != 11:
            return PluralCategory.ONE
        if 2 <= n10 <= 4 and not (12 <= n100 <= 14):
            return PluralCategory.FEW
        # ... proper CLDR implementation
```

### IME Keyboard Handling (ime.py:743-829)
```python
def handle_key(self, key: str, modifiers: int = 0) -> bool:
    # Full keyboard navigation for candidate selection (Up/Down/PageUp/PageDown)
    # Number key selection (1-9), Enter to commit, Escape to cancel
    # Backspace/Delete with cursor movement, Home/End
```

### Font Atlas Generation (font.py:493-550)
```python
def create_atlas(self, font: Font, width: int = 1024, height: int = 1024, characters: str | None = None) -> FontAtlas:
    # Row-based packing with padding
    # UV coordinate calculation for each glyph
    # Default charset: ASCII + Latin-1 Supplement
```

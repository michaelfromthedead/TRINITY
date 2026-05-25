# PHASE 4 TODO: Text Module

---

## Font Management Tasks

### T-4.1: Verify Font Family Loading

**File**: `engine/ui/text/font.py`

**Description**: Ensure font families with multiple weights/styles load correctly.

**Acceptance Criteria**:
- [ ] Load font family with regular, bold, italic, bold-italic
- [ ] Access fonts by weight (400, 700) and style (normal, italic)
- [ ] Missing weights/styles return fallback or error
- [ ] Font file formats supported (TTF, OTF)

---

### T-4.2: Verify Atlas Packing

**File**: `engine/ui/text/font.py`

**Description**: Ensure `_pack_atlas()` efficiently packs glyphs.

**Acceptance Criteria**:
- [ ] Glyphs sorted by height before packing
- [ ] Shelf algorithm places glyphs row by row
- [ ] UV coordinates recorded for each glyph
- [ ] Atlas size grows if needed (or throws if fixed)
- [ ] No glyph overlap

---

### T-4.3: Verify SDF Font Generation

**File**: `engine/ui/text/font.py`

**Description**: Ensure `_generate_sdf()` produces correct distance field.

**Acceptance Criteria**:
- [ ] Distance field encodes edge distance
- [ ] Inside glyph = values > 0.5
- [ ] Outside glyph = values < 0.5
- [ ] Edge = 0.5
- [ ] SDF fonts scale without blurring

---

### T-4.4: Verify Fallback Chain

**File**: `engine/ui/text/font.py`

**Description**: Ensure `_find_fallback_glyph()` returns correct glyph.

**Acceptance Criteria**:
- [ ] Primary font checked first
- [ ] If glyph missing, check next font in chain
- [ ] Continue until glyph found or chain exhausted
- [ ] Last resort font (if present) handles all codepoints
- [ ] Returns placeholder glyph if all fail

---

## Text Rendering Tasks

### T-4.5: Verify Unicode Line Breaking

**File**: `engine/ui/text/text_renderer.py`

**Description**: Ensure `_find_break_opportunities()` follows UAX #14.

**Acceptance Criteria**:
- [ ] Break after spaces (SP class)
- [ ] Break after hyphens (BA class)
- [ ] No break inside words (AL class consecutive)
- [ ] No break inside numbers (NU class)
- [ ] CJK characters allow break between them
- [ ] Handles mixed scripts (Latin + CJK)

---

### T-4.6: Verify Text Shaping

**File**: `engine/ui/text/text_renderer.py`

**Description**: Ensure `_shape_run()` handles complex scripts.

**Acceptance Criteria**:
- [ ] Arabic contextual forms applied (initial/medial/final/isolated)
- [ ] Ligatures formed (fi, fl in Latin; lam-alef in Arabic)
- [ ] Kerning applied between character pairs
- [ ] Mark positioning for diacritics

---

### T-4.7: Verify Glyph Cache

**File**: `engine/ui/text/text_renderer.py`

**Description**: Ensure glyph cache provides LRU caching.

**Acceptance Criteria**:
- [ ] Cache hit returns shaped glyphs without reshaping
- [ ] Cache key includes text, font, size, features
- [ ] LRU eviction removes least recently used
- [ ] Cache respects max size limit
- [ ] Cache hit rate measurable

---

### T-4.8: Verify Text Measurement

**File**: `engine/ui/text/text_renderer.py`

**Description**: Ensure `measure()` returns correct dimensions.

**Acceptance Criteria**:
- [ ] Width = sum of glyph advances
- [ ] Height = line height (ascent + descent + leading)
- [ ] Ascent = distance from baseline to top
- [ ] Descent = distance from baseline to bottom
- [ ] Multi-line text measured correctly

---

## Rich Text Tasks

### T-4.9: Verify Markup Tokenization

**File**: `engine/ui/text/rich_text.py`

**Description**: Ensure `_tokenize()` splits markup correctly.

**Acceptance Criteria**:
- [ ] Plain text between tags preserved
- [ ] Opening tags recognized: `[b]`, `[i]`, `[u]`, etc.
- [ ] Closing tags recognized: `[/b]`, `[/i]`, `[/u]`, etc.
- [ ] Tags with values parsed: `[color=#FF0000]`
- [ ] Escaped brackets handled: `[[` -> `[`

---

### T-4.10: Verify Markup Parsing

**File**: `engine/ui/text/rich_text.py`

**Description**: Ensure `_parse_markup()` builds correct runs.

**Acceptance Criteria**:
- [ ] Bold tag applies bold style to enclosed text
- [ ] Italic tag applies italic style
- [ ] Underline tag applies underline
- [ ] Strikethrough tag applies strikethrough
- [ ] Nested tags combine styles (bold + italic)

---

### T-4.11: Verify Color Tag

**File**: `engine/ui/text/rich_text.py`

**Description**: Ensure `[color=X]` applies correct color.

**Acceptance Criteria**:
- [ ] Hex color `#RRGGBB` parsed
- [ ] Named colors (if supported) parsed
- [ ] Color applied to text run
- [ ] Color resets after closing tag

---

### T-4.12: Verify Size Tag

**File**: `engine/ui/text/rich_text.py`

**Description**: Ensure `[size=X]` applies correct font size.

**Acceptance Criteria**:
- [ ] Numeric size applied
- [ ] Size affects text measurement
- [ ] Size resets after closing tag

---

### T-4.13: Verify Font Tag

**File**: `engine/ui/text/rich_text.py`

**Description**: Ensure `[font=X]` applies correct font family.

**Acceptance Criteria**:
- [ ] Font family name resolved
- [ ] Font applied to text run
- [ ] Fallback if font not found
- [ ] Font resets after closing tag

---

### T-4.14: Verify Link Tag

**File**: `engine/ui/text/rich_text.py`

**Description**: Ensure `[link=X]` creates clickable link.

**Acceptance Criteria**:
- [ ] Link URL stored
- [ ] Link text styled (blue, underline by default)
- [ ] Hover state changes style
- [ ] Click triggers callback with URL

---

### T-4.15: Verify Image Tag

**File**: `engine/ui/text/rich_text.py`

**Description**: Ensure `[img=X]` embeds inline image.

**Acceptance Criteria**:
- [ ] Image source loaded
- [ ] Width/height applied (explicit or intrinsic)
- [ ] Alignment respected (baseline, top, middle, bottom)
- [ ] Image participates in text layout

---

### T-4.16: Verify Rich Text Builder

**File**: `engine/ui/text/rich_text.py`

**Description**: Ensure builder pattern constructs correct rich text.

**Acceptance Criteria**:
- [ ] `.text()` adds plain text
- [ ] `.bold()` / `.end_bold()` toggles bold
- [ ] `.color()` / `.end_color()` sets color
- [ ] `.build()` returns complete RichText
- [ ] Chaining works fluently

---

## Localization Tasks

### T-4.17: Verify Locale Loading

**File**: `engine/ui/text/localization.py`

**Description**: Ensure locale files load correctly.

**Acceptance Criteria**:
- [ ] JSON format parsed
- [ ] String keys registered
- [ ] Multiple locales loaded simultaneously
- [ ] Invalid files handled gracefully

---

### T-4.18: Verify String Interpolation

**File**: `engine/ui/text/localization.py`

**Description**: Ensure `format()` interpolates parameters.

**Acceptance Criteria**:
- [ ] `{name}` replaced with parameter value
- [ ] Multiple parameters supported
- [ ] Missing parameters handled (keep placeholder or error)
- [ ] Type conversion (numbers to strings)

---

### T-4.19: Verify English Pluralization

**File**: `engine/ui/text/localization.py`

**Description**: Ensure English plural rules work.

**Acceptance Criteria**:
- [ ] n=1 returns singular form
- [ ] n!=1 returns plural form
- [ ] n=0 returns plural form ("0 items")

---

### T-4.20: Verify Russian Pluralization

**File**: `engine/ui/text/localization.py`

**Description**: Ensure Russian plural rules work (4 forms).

**Acceptance Criteria**:
- [ ] n%10=1, n%100!=11 -> one (1 элемент, 21 элемент)
- [ ] n%10 in 2-4, n%100 not in 12-14 -> few (2 элемента)
- [ ] n%10=0, n%10 in 5-9, n%100 in 11-14 -> many (5 элементов)
- [ ] Otherwise -> other

---

### T-4.21: Verify Arabic Pluralization

**File**: `engine/ui/text/localization.py`

**Description**: Ensure Arabic plural rules work (6 forms).

**Acceptance Criteria**:
- [ ] n=0 -> zero
- [ ] n=1 -> one
- [ ] n=2 -> two
- [ ] n%100 in 3-10 -> few
- [ ] n%100 in 11-99 -> many
- [ ] Otherwise -> other

---

### T-4.22: Verify RTL Detection

**File**: `engine/ui/text/localization.py`

**Description**: Ensure `_detect_rtl()` identifies RTL text.

**Acceptance Criteria**:
- [ ] Arabic text detected as RTL
- [ ] Hebrew text detected as RTL
- [ ] Latin text detected as LTR
- [ ] Mixed text uses first strong character
- [ ] Empty string handled

---

### T-4.23: Verify Locale Fallback

**File**: `engine/ui/text/localization.py`

**Description**: Ensure fallback chain works.

**Acceptance Criteria**:
- [ ] `fr-CA` falls back to `fr`
- [ ] `fr` falls back to default (`en`)
- [ ] Missing key returns key or placeholder
- [ ] Fallback searched in order

---

## IME Tasks

### T-4.24: Verify Composition Start

**File**: `engine/ui/text/ime.py`

**Description**: Ensure composition mode starts correctly.

**Acceptance Criteria**:
- [ ] `on_composition_start()` initializes state
- [ ] Composition preview shown at cursor
- [ ] Normal text input suspended
- [ ] Candidate window hidden initially

---

### T-4.25: Verify Composition Update

**File**: `engine/ui/text/ime.py`

**Description**: Ensure `_handle_composition_update()` updates state.

**Acceptance Criteria**:
- [ ] Composition string updated
- [ ] Cursor position tracked
- [ ] Preview text redrawn
- [ ] Candidates regenerated

---

### T-4.26: Verify Candidate Window

**File**: `engine/ui/text/ime.py`

**Description**: Ensure `_show_candidates()` displays candidates.

**Acceptance Criteria**:
- [ ] Candidates listed with numbers (1-9)
- [ ] Selected candidate highlighted
- [ ] Window positioned near text cursor
- [ ] Window scrolls if many candidates

---

### T-4.27: Verify Candidate Selection

**File**: `engine/ui/text/ime.py`

**Description**: Ensure candidate selection works.

**Acceptance Criteria**:
- [ ] Number keys (1-9) select candidate
- [ ] Arrow keys navigate candidates
- [ ] Enter commits selected candidate
- [ ] Space may commit or advance (IME-dependent)

---

### T-4.28: Verify Composition Commit

**File**: `engine/ui/text/ime.py`

**Description**: Ensure `_commit_composition()` inserts text.

**Acceptance Criteria**:
- [ ] Selected candidate inserted at cursor
- [ ] Composition state cleared
- [ ] Candidate window hidden
- [ ] Normal input mode resumed

---

### T-4.29: Verify Composition Cancel

**File**: `engine/ui/text/ime.py`

**Description**: Ensure composition can be cancelled.

**Acceptance Criteria**:
- [ ] Escape key cancels composition
- [ ] Composition text discarded
- [ ] Original cursor position restored
- [ ] Candidate window hidden

---

### T-4.30: Verify Cursor in Composition

**File**: `engine/ui/text/ime.py`

**Description**: Ensure cursor navigation within composition works.

**Acceptance Criteria**:
- [ ] Left/right arrow moves cursor in composition
- [ ] Cursor position affects which part is converted
- [ ] Visual cursor feedback in composition preview

"""
Comprehensive tests for TextRenderer (measurement, line breaking, shaping).

Tests cover:
- TextMeasurement results
- LineBreaker algorithm
- LineBreakResult handling
- TextShaper for complex scripts
- GlyphCache LRU behavior
- TextRenderer measurement methods
- Line breaking with various widths
- Text layout with alignment
- Cursor position calculation
- Text overflow handling
"""

import pytest
from dataclasses import dataclass
from typing import Optional

from engine.ui.text.font import Font, FontManager, GlyphMetrics
from engine.ui.text.text_renderer import (
    TextRenderer,
    TextMeasurement,
    LineBreaker,
    LineBreakResult,
    TextShaper,
    ShapedGlyph,
    GlyphCache,
    TextAlignment,
    TextOverflow,
    LineBreakType,
)


def create_test_font(size: float = 16.0) -> Font:
    """Create a test font with known glyph metrics."""
    font = Font(family="TestFont", size=size)
    # Add some basic glyphs
    for i, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "):
        font._glyphs[ord(char)] = GlyphMetrics(
            codepoint=ord(char),
            width=size * 0.6,
            height=size * 1.2,
            bearing_x=0,
            bearing_y=size * 0.9,
            advance=size * 0.6,
        )
    return font


class TestTextAlignment:
    """Tests for TextAlignment enum."""

    def test_alignment_values(self):
        """Test all alignment values exist."""
        assert TextAlignment.LEFT
        assert TextAlignment.CENTER
        assert TextAlignment.RIGHT
        assert TextAlignment.JUSTIFY


class TestTextOverflow:
    """Tests for TextOverflow enum."""

    def test_overflow_values(self):
        """Test all overflow values exist."""
        assert TextOverflow.CLIP
        assert TextOverflow.ELLIPSIS
        assert TextOverflow.WRAP
        assert TextOverflow.SHRINK


class TestLineBreakType:
    """Tests for LineBreakType enum."""

    def test_break_type_values(self):
        """Test all break type values exist."""
        assert LineBreakType.MANDATORY
        assert LineBreakType.ALLOWED
        assert LineBreakType.FORBIDDEN


class TestTextMeasurement:
    """Tests for TextMeasurement class."""

    def test_measurement_creation(self):
        """Test creating a text measurement."""
        measure = TextMeasurement(
            width=100.0,
            height=20.0,
            ascender=14.0,
            descender=4.0,
            line_count=1,
        )
        assert measure.width == 100.0
        assert measure.height == 20.0
        assert measure.line_count == 1

    def test_measurement_size_property(self):
        """Test size property."""
        measure = TextMeasurement(
            width=100.0,
            height=20.0,
            ascender=14.0,
            descender=4.0,
            line_count=1,
        )
        assert measure.size == (100.0, 20.0)

    def test_measurement_bounds(self):
        """Test bounds property."""
        measure = TextMeasurement(
            width=100.0,
            height=20.0,
            ascender=14.0,
            descender=4.0,
            line_count=1,
            bounds=(0.0, -14.0, 100.0, 20.0),
        )
        assert measure.bounds == (0.0, -14.0, 100.0, 20.0)

    def test_measurement_immutable(self):
        """Test measurement is immutable."""
        measure = TextMeasurement(
            width=100.0,
            height=20.0,
            ascender=14.0,
            descender=4.0,
            line_count=1,
        )
        with pytest.raises(AttributeError):
            measure.width = 200.0


class TestLineBreakResult:
    """Tests for LineBreakResult class."""

    def test_result_creation(self):
        """Test creating a line break result."""
        result = LineBreakResult(
            lines=["Hello", "World"],
            line_widths=[50.0, 50.0],
            total_height=40.0,
        )
        assert len(result.lines) == 2
        assert result.total_height == 40.0

    def test_result_len(self):
        """Test len() on result."""
        result = LineBreakResult(
            lines=["A", "B", "C"],
            line_widths=[10.0, 10.0, 10.0],
            total_height=60.0,
        )
        assert len(result) == 3

    def test_result_iteration(self):
        """Test iterating over result."""
        result = LineBreakResult(
            lines=["Hello", "World"],
            line_widths=[50.0, 55.0],
            total_height=40.0,
        )
        pairs = list(result)
        assert pairs == [("Hello", 50.0), ("World", 55.0)]


class TestShapedGlyph:
    """Tests for ShapedGlyph class."""

    def test_shaped_glyph_creation(self):
        """Test creating a shaped glyph."""
        glyph = ShapedGlyph(
            codepoint=65,
            glyph_id=65,
            x_offset=0.0,
            y_offset=0.0,
            x_advance=10.0,
            y_advance=0.0,
            cluster=0,
        )
        assert glyph.codepoint == 65
        assert glyph.x_advance == 10.0
        assert glyph.cluster == 0

    def test_shaped_glyph_immutable(self):
        """Test shaped glyph is immutable."""
        glyph = ShapedGlyph(
            codepoint=65,
            glyph_id=65,
            x_offset=0.0,
            y_offset=0.0,
            x_advance=10.0,
            y_advance=0.0,
            cluster=0,
        )
        with pytest.raises(AttributeError):
            glyph.x_advance = 20.0


class TestLineBreaker:
    """Tests for LineBreaker class."""

    def test_line_breaker_creation(self):
        """Test creating a line breaker."""
        breaker = LineBreaker()
        assert breaker._emergency_break is True

    def test_find_break_opportunities_empty(self):
        """Test finding breaks in empty string."""
        breaker = LineBreaker()
        breaks = breaker.find_break_opportunities("")
        assert breaks == []

    def test_find_break_opportunities_single_word(self):
        """Test finding breaks in single word."""
        breaker = LineBreaker()
        breaks = breaker.find_break_opportunities("Hello")
        assert len(breaks) == 0  # No break opportunities

    def test_find_break_opportunities_two_words(self):
        """Test finding breaks between words."""
        breaker = LineBreaker()
        breaks = breaker.find_break_opportunities("Hello World")

        # Should find break after space
        allowed_breaks = [b for b in breaks if b[1] == LineBreakType.ALLOWED]
        assert len(allowed_breaks) >= 1

    def test_find_break_opportunities_newline(self):
        """Test finding mandatory break at newline."""
        breaker = LineBreaker()
        breaks = breaker.find_break_opportunities("Hello\nWorld")

        mandatory_breaks = [b for b in breaks if b[1] == LineBreakType.MANDATORY]
        assert len(mandatory_breaks) == 1

    def test_find_break_opportunities_non_breaking_space(self):
        """Test non-breaking space prevents break."""
        breaker = LineBreaker()
        # U+00A0 is non-breaking space
        breaks = breaker.find_break_opportunities("Hello\u00A0World")

        forbidden_breaks = [b for b in breaks if b[1] == LineBreakType.FORBIDDEN]
        assert len(forbidden_breaks) >= 1

    def test_break_text_empty(self):
        """Test breaking empty text."""
        breaker = LineBreaker()
        font = create_test_font()
        renderer = TextRenderer()

        result = breaker.break_text("", 100.0, renderer.measure_text, font)

        assert len(result.lines) == 0
        assert result.total_height == 0.0

    def test_break_text_single_line(self):
        """Test text that fits on single line."""
        breaker = LineBreaker()
        font = create_test_font()
        renderer = TextRenderer()

        result = breaker.break_text("Hi", 1000.0, renderer.measure_text, font)

        assert len(result.lines) == 1
        assert result.lines[0] == "Hi"

    def test_break_text_multiple_lines(self):
        """Test text that needs multiple lines."""
        breaker = LineBreaker()
        font = create_test_font(16.0)
        renderer = TextRenderer()

        # With 16pt font, each char is ~9.6px. "Hello World" = ~115px
        # Break at 60px should create multiple lines
        result = breaker.break_text("Hello World", 60.0, renderer.measure_text, font)

        assert len(result.lines) >= 2

    def test_break_text_preserves_mandatory_breaks(self):
        """Test mandatory breaks are preserved."""
        breaker = LineBreaker()
        font = create_test_font()
        renderer = TextRenderer()

        result = breaker.break_text("Hello\nWorld", 1000.0, renderer.measure_text, font)

        assert len(result.lines) == 2
        assert result.lines[0] == "Hello"
        assert result.lines[1] == "World"

    def test_break_text_crlf_handling(self):
        """Test CRLF line endings."""
        breaker = LineBreaker()
        font = create_test_font()
        renderer = TextRenderer()

        result = breaker.break_text("Hello\r\nWorld", 1000.0, renderer.measure_text, font)

        assert len(result.lines) == 2

    def test_break_text_consecutive_newlines(self):
        """Test consecutive newlines create empty lines."""
        breaker = LineBreaker()
        font = create_test_font()
        renderer = TextRenderer()

        result = breaker.break_text("Hello\n\nWorld", 1000.0, renderer.measure_text, font)

        assert len(result.lines) == 3
        assert result.lines[1] == ""  # Empty line

    def test_break_text_line_widths(self):
        """Test line widths are calculated correctly."""
        breaker = LineBreaker()
        font = create_test_font()
        renderer = TextRenderer()

        result = breaker.break_text("A B", 1000.0, renderer.measure_text, font)

        assert len(result.line_widths) == 1
        assert result.line_widths[0] > 0


class TestTextShaper:
    """Tests for TextShaper class."""

    def test_shaper_creation(self):
        """Test creating a text shaper."""
        shaper = TextShaper()
        assert shaper is not None

    def test_shaper_empty_text(self):
        """Test shaping empty text."""
        shaper = TextShaper()
        font = create_test_font()

        glyphs = shaper.shape("", font)

        assert glyphs == []

    def test_shaper_single_char(self):
        """Test shaping single character."""
        shaper = TextShaper()
        font = create_test_font()

        glyphs = shaper.shape("A", font)

        assert len(glyphs) == 1
        assert glyphs[0].codepoint == ord("A")

    def test_shaper_multiple_chars(self):
        """Test shaping multiple characters."""
        shaper = TextShaper()
        font = create_test_font()

        glyphs = shaper.shape("ABC", font)

        assert len(glyphs) == 3
        assert glyphs[0].codepoint == ord("A")
        assert glyphs[1].codepoint == ord("B")
        assert glyphs[2].codepoint == ord("C")

    def test_shaper_cluster_indices(self):
        """Test cluster indices for cursor positioning."""
        shaper = TextShaper()
        font = create_test_font()

        glyphs = shaper.shape("ABC", font)

        assert glyphs[0].cluster == 0
        assert glyphs[1].cluster == 1
        assert glyphs[2].cluster == 2

    def test_shaper_letter_spacing(self):
        """Test letter spacing is applied."""
        shaper = TextShaper()
        font = create_test_font()
        font_spaced = Font(family="TestFont", size=16.0, letter_spacing=5.0)
        font_spaced._glyphs = font._glyphs.copy()

        glyphs_normal = shaper.shape("AB", font)
        glyphs_spaced = shaper.shape("AB", font_spaced)

        # Spaced version should have larger advances
        assert glyphs_spaced[0].x_advance > glyphs_normal[0].x_advance

    def test_shaper_ltr_direction(self):
        """Test LTR text direction."""
        shaper = TextShaper()
        font = create_test_font()

        glyphs = shaper.shape("AB", font, direction="ltr")

        # First glyph should be A
        assert glyphs[0].codepoint == ord("A")

    def test_shaper_rtl_direction(self):
        """Test RTL text direction."""
        shaper = TextShaper()
        font = create_test_font()

        glyphs = shaper.shape("AB", font, direction="rtl")

        # For RTL, glyphs should be reversed
        assert glyphs[0].codepoint == ord("B")
        assert glyphs[1].codepoint == ord("A")

    def test_shaper_ligatures(self):
        """Test ligature substitution."""
        shaper = TextShaper()
        font = create_test_font()

        glyphs = shaper.shape("fi", font, enable_ligatures=True)

        # Should produce fi ligature (U+FB01)
        assert len(glyphs) == 1
        assert glyphs[0].codepoint == 0xFB01

    def test_shaper_ligatures_disabled(self):
        """Test ligatures can be disabled."""
        shaper = TextShaper()
        font = create_test_font()

        glyphs = shaper.shape("fi", font, enable_ligatures=False)

        # Should keep separate characters
        assert len(glyphs) == 2

    def test_shaper_cluster_map(self):
        """Test getting cluster map."""
        shaper = TextShaper()
        font = create_test_font()

        text = "ABC"
        glyphs = shaper.shape(text, font)
        cluster_map = shaper.get_cluster_map(text, glyphs)

        assert 0 in cluster_map
        assert 1 in cluster_map
        assert 2 in cluster_map


class TestGlyphCache:
    """Tests for GlyphCache class."""

    def test_cache_creation(self):
        """Test creating a glyph cache."""
        cache = GlyphCache(max_size=100)
        assert len(cache) == 0

    def test_cache_put_and_get(self):
        """Test putting and getting from cache."""
        cache = GlyphCache()
        font = create_test_font()
        glyph = GlyphMetrics(
            codepoint=65, width=10, height=14,
            bearing_x=1, bearing_y=12, advance=12,
        )

        cache.put(font, 65, glyph)
        result = cache.get(font, 65)

        assert result is glyph

    def test_cache_get_not_found(self):
        """Test getting non-existent entry."""
        cache = GlyphCache()
        font = create_test_font()

        result = cache.get(font, 65)

        assert result is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = GlyphCache(max_size=3)
        font = create_test_font()

        # Fill cache
        for i in range(3):
            glyph = GlyphMetrics(
                codepoint=65 + i, width=10, height=14,
                bearing_x=1, bearing_y=12, advance=12,
            )
            cache.put(font, 65 + i, glyph)

        # Access first entry to make it recently used
        cache.get(font, 65)

        # Add new entry, should evict second entry (66)
        glyph = GlyphMetrics(
            codepoint=90, width=10, height=14,
            bearing_x=1, bearing_y=12, advance=12,
        )
        cache.put(font, 90, glyph)

        assert cache.get(font, 65) is not None  # Still in cache
        assert cache.get(font, 66) is None  # Evicted
        assert cache.get(font, 90) is not None  # New entry

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = GlyphCache()
        font = create_test_font()
        glyph = GlyphMetrics(
            codepoint=65, width=10, height=14,
            bearing_x=1, bearing_y=12, advance=12,
        )
        cache.put(font, 65, glyph)

        cache.clear()

        assert len(cache) == 0
        assert cache.get(font, 65) is None


class TestTextRenderer:
    """Tests for TextRenderer class."""

    def test_renderer_creation(self):
        """Test creating a text renderer."""
        renderer = TextRenderer()
        assert renderer.glyph_cache is not None

    def test_renderer_with_font_manager(self):
        """Test renderer with font manager."""
        manager = FontManager()
        renderer = TextRenderer(font_manager=manager)
        assert renderer.font_manager is manager

    def test_measure_text_empty(self):
        """Test measuring empty text."""
        renderer = TextRenderer()
        font = create_test_font()

        measure = renderer.measure_text("", font)

        assert measure.width == 0.0
        assert measure.line_count == 1  # Empty string counts as 1 line

    def test_measure_text_single_char(self):
        """Test measuring single character."""
        renderer = TextRenderer()
        font = create_test_font(16.0)

        measure = renderer.measure_text("A", font)

        assert measure.width > 0
        assert measure.line_count == 1

    def test_measure_text_multiple_chars(self):
        """Test measuring multiple characters."""
        renderer = TextRenderer()
        font = create_test_font(16.0)

        measure = renderer.measure_text("ABC", font)

        # Width should be sum of advances
        assert measure.width > 0

    def test_measure_text_multiline(self):
        """Test measuring multiline text."""
        renderer = TextRenderer()
        font = create_test_font(16.0)

        measure = renderer.measure_text("Hello\nWorld", font)

        assert measure.line_count == 2
        assert measure.height == font.line_spacing * 2

    def test_measure_char(self):
        """Test measuring single character method."""
        renderer = TextRenderer()
        font = create_test_font()

        measure = renderer.measure_char("A", font)

        assert measure.width > 0

    def test_measure_char_multiple_chars_rejected(self):
        """Test measure_char rejects multiple characters."""
        renderer = TextRenderer()
        font = create_test_font()

        with pytest.raises(ValueError, match="Expected single character"):
            renderer.measure_char("AB", font)

    def test_break_lines(self):
        """Test breaking text into lines."""
        renderer = TextRenderer()
        font = create_test_font(16.0)

        result = renderer.break_lines("Hello World Test", font, max_width=100.0)

        assert len(result.lines) >= 1
        assert result.total_height > 0

    def test_shape_text(self):
        """Test shaping text."""
        renderer = TextRenderer()
        font = create_test_font()

        glyphs = renderer.shape_text("Hello", font)

        assert len(glyphs) == 5

    def test_shape_text_rtl(self):
        """Test shaping RTL text."""
        renderer = TextRenderer()
        font = create_test_font()

        glyphs = renderer.shape_text("AB", font, direction="rtl")

        assert glyphs[0].codepoint == ord("B")

    def test_layout_text_single_line(self):
        """Test layout without max_width (single line)."""
        renderer = TextRenderer()
        font = create_test_font()

        layout = renderer.layout_text("Hello", font)

        assert len(layout) == 1
        assert layout[0][0] == "Hello"
        assert layout[0][1] == 0.0  # x_offset
        assert layout[0][2] == 0.0  # y_offset

    def test_layout_text_wrapped(self):
        """Test layout with wrapping."""
        renderer = TextRenderer()
        font = create_test_font(16.0)

        layout = renderer.layout_text("Hello World Test", font, max_width=80.0)

        assert len(layout) >= 2

    def test_layout_text_align_left(self):
        """Test layout with left alignment."""
        renderer = TextRenderer()
        font = create_test_font()

        layout = renderer.layout_text(
            "Hi", font,
            max_width=200.0,
            alignment=TextAlignment.LEFT,
        )

        assert layout[0][1] == 0.0  # x_offset at left

    def test_layout_text_align_center(self):
        """Test layout with center alignment."""
        renderer = TextRenderer()
        font = create_test_font()

        layout = renderer.layout_text(
            "Hi", font,
            max_width=200.0,
            alignment=TextAlignment.CENTER,
        )

        # x_offset should be > 0 for centered text
        assert layout[0][1] > 0

    def test_layout_text_align_right(self):
        """Test layout with right alignment."""
        renderer = TextRenderer()
        font = create_test_font()

        layout = renderer.layout_text(
            "Hi", font,
            max_width=200.0,
            alignment=TextAlignment.RIGHT,
        )

        # x_offset should be close to max_width - text_width
        assert layout[0][1] > 0

    def test_layout_text_overflow_clip(self):
        """Test layout with clip overflow."""
        renderer = TextRenderer()
        font = create_test_font()

        layout = renderer.layout_text(
            "Hello World", font,
            max_width=50.0,
            overflow=TextOverflow.CLIP,
        )

        assert len(layout) == 1
        # Text should be truncated
        assert len(layout[0][0]) < len("Hello World")

    def test_layout_text_overflow_ellipsis(self):
        """Test layout with ellipsis overflow."""
        renderer = TextRenderer()
        font = create_test_font()

        layout = renderer.layout_text(
            "Hello World", font,
            max_width=50.0,
            overflow=TextOverflow.ELLIPSIS,
        )

        assert len(layout) == 1
        assert layout[0][0].endswith("\u2026")  # Ellipsis

    def test_get_cursor_position_start(self):
        """Test cursor position at start."""
        renderer = TextRenderer()
        font = create_test_font()

        x, y = renderer.get_cursor_position("Hello", font, index=0)

        assert x == 0.0
        assert y == 0.0

    def test_get_cursor_position_end(self):
        """Test cursor position at end."""
        renderer = TextRenderer()
        font = create_test_font()

        x, y = renderer.get_cursor_position("Hello", font, index=5)

        assert x > 0  # Should be at end of text
        assert y == 0.0

    def test_get_cursor_position_middle(self):
        """Test cursor position in middle."""
        renderer = TextRenderer()
        font = create_test_font()

        x1, _ = renderer.get_cursor_position("Hello", font, index=2)
        x2, _ = renderer.get_cursor_position("Hello", font, index=4)

        assert x1 < x2

    def test_get_index_at_position_start(self):
        """Test getting index at start position."""
        renderer = TextRenderer()
        font = create_test_font()

        index = renderer.get_index_at_position("Hello", font, x=0.0)

        assert index == 0

    def test_get_index_at_position_end(self):
        """Test getting index at end position."""
        renderer = TextRenderer()
        font = create_test_font()

        index = renderer.get_index_at_position("Hello", font, x=1000.0)

        assert index == 5  # Length of "Hello"

    def test_get_index_at_position_middle(self):
        """Test getting index at middle position."""
        renderer = TextRenderer()
        font = create_test_font(16.0)

        # Each char is ~9.6px, so at 20px we should be in second char
        index = renderer.get_index_at_position("Hello", font, x=20.0)

        assert 1 <= index <= 3

    def test_clear_cache(self):
        """Test clearing renderer cache."""
        renderer = TextRenderer()
        font = create_test_font()

        # Measure to populate cache
        renderer.measure_text("Hello", font)

        renderer.clear_cache()

        assert len(renderer.glyph_cache) == 0

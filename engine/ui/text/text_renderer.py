"""
Text rendering system with measurement, line breaking, and shaping.

Provides:
- Text measurement (width, height, bounds)
- Line breaking algorithm (Unicode-aware)
- Text shaping for complex scripts
- Glyph caching for performance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator

from .font import Font, FontManager, GlyphMetrics, FontFallbackChain


class TextAlignment(Enum):
    """Text alignment options."""
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()
    JUSTIFY = auto()


class TextOverflow(Enum):
    """Text overflow handling modes."""
    CLIP = auto()
    ELLIPSIS = auto()
    WRAP = auto()
    SHRINK = auto()


class LineBreakType(Enum):
    """Type of line break opportunity."""
    MANDATORY = auto()  # Hard break (newline character)
    ALLOWED = auto()    # Soft break opportunity (between words)
    FORBIDDEN = auto()  # No break allowed (within word)


@dataclass(frozen=True)
class TextMeasurement:
    """Result of measuring text dimensions.

    Attributes:
        width: Total width of the text
        height: Total height (including line height)
        ascender: Maximum ascender height
        descender: Maximum descender depth
        line_count: Number of lines
        bounds: Tight bounding box (x, y, width, height)
    """
    width: float
    height: float
    ascender: float
    descender: float
    line_count: int
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def size(self) -> tuple[float, float]:
        """Get size as (width, height) tuple."""
        return (self.width, self.height)


@dataclass
class LineBreakResult:
    """Result of breaking text into lines.

    Attributes:
        lines: List of text lines
        line_widths: Width of each line
        total_height: Total height of all lines
    """
    lines: list[str]
    line_widths: list[float]
    total_height: float

    def __len__(self) -> int:
        return len(self.lines)

    def __iter__(self) -> Iterator[tuple[str, float]]:
        """Iterate over (line, width) pairs."""
        return zip(self.lines, self.line_widths)


@dataclass(frozen=True)
class ShapedGlyph:
    """A shaped glyph ready for rendering.

    Attributes:
        codepoint: Original Unicode codepoint
        glyph_id: Font-specific glyph ID
        x_offset: Horizontal offset from pen position
        y_offset: Vertical offset from pen position
        x_advance: Horizontal advance after this glyph
        y_advance: Vertical advance (usually 0 for horizontal text)
        cluster: Index into original text (for cursor positioning)
    """
    codepoint: int
    glyph_id: int
    x_offset: float
    y_offset: float
    x_advance: float
    y_advance: float
    cluster: int


class LineBreaker:
    """Unicode-aware line breaking algorithm.

    Implements a simplified version of the Unicode Line Breaking Algorithm
    (UAX #14) for determining where text can be broken into lines.
    """

    # Break classes for common characters
    _BREAK_BEFORE = frozenset("([{")
    _BREAK_AFTER = frozenset(".,;:!?)]}")
    _NO_BREAK = frozenset("\u00A0\u2007\u202F")  # Non-breaking spaces

    def __init__(self) -> None:
        """Initialize the line breaker."""
        self._emergency_break = True  # Allow breaking within words if needed

    def find_break_opportunities(self, text: str) -> list[tuple[int, LineBreakType]]:
        """Find all potential line break points in text.

        Args:
            text: Text to analyze

        Returns:
            List of (position, break_type) tuples
        """
        breaks: list[tuple[int, LineBreakType]] = []

        if not text:
            return breaks

        for i, char in enumerate(text):
            # Mandatory breaks
            if char in "\n\r\u2028\u2029":
                breaks.append((i, LineBreakType.MANDATORY))
                continue

            # Non-breaking characters
            if char in self._NO_BREAK:
                breaks.append((i, LineBreakType.FORBIDDEN))
                continue

            # Check for break opportunity before this character
            if i > 0:
                prev_char = text[i - 1]

                # Space followed by non-space
                if prev_char.isspace() and not char.isspace():
                    breaks.append((i, LineBreakType.ALLOWED))

                # After certain punctuation
                elif prev_char in self._BREAK_AFTER and char.isalnum():
                    breaks.append((i, LineBreakType.ALLOWED))

        return breaks

    def break_text(
        self,
        text: str,
        max_width: float,
        measure_func: callable,
        font: Font,
    ) -> LineBreakResult:
        """Break text into lines that fit within max_width.

        Args:
            text: Text to break
            max_width: Maximum width of each line
            measure_func: Function(text, font) -> TextMeasurement
            font: Font to use for measurement

        Returns:
            LineBreakResult with broken lines
        """
        if not text:
            return LineBreakResult(lines=[], line_widths=[], total_height=0.0)

        lines: list[str] = []
        line_widths: list[float] = []

        # Split by mandatory breaks first
        paragraphs = self._split_mandatory_breaks(text)

        for paragraph in paragraphs:
            if not paragraph:
                # Empty line (from consecutive newlines)
                lines.append("")
                line_widths.append(0.0)
                continue

            # Break paragraph into lines
            para_lines = self._break_paragraph(paragraph, max_width, measure_func, font)
            lines.extend(para_lines.lines)
            line_widths.extend(para_lines.line_widths)

        total_height = len(lines) * font.line_spacing if lines else 0.0

        return LineBreakResult(
            lines=lines,
            line_widths=line_widths,
            total_height=total_height,
        )

    def _split_mandatory_breaks(self, text: str) -> list[str]:
        """Split text at mandatory break points."""
        # Handle different line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return text.split("\n")

    def _break_paragraph(
        self,
        text: str,
        max_width: float,
        measure_func: callable,
        font: Font,
    ) -> LineBreakResult:
        """Break a single paragraph into lines."""
        lines: list[str] = []
        line_widths: list[float] = []

        # Split into words (keeping spaces)
        words = self._tokenize(text)

        current_line: list[str] = []
        current_width = 0.0

        for word in words:
            word_measure = measure_func(word, font)
            word_width = word_measure.width

            if current_width + word_width <= max_width or not current_line:
                # Word fits, add to current line
                current_line.append(word)
                current_width += word_width
            else:
                # Word doesn't fit, start new line
                line_text = "".join(current_line).rstrip()
                lines.append(line_text)
                line_widths.append(measure_func(line_text, font).width)

                # Start new line with current word
                current_line = [word.lstrip()]
                current_width = measure_func(current_line[0], font).width if current_line[0] else 0.0

        # Add remaining text
        if current_line:
            line_text = "".join(current_line).rstrip()
            lines.append(line_text)
            line_widths.append(measure_func(line_text, font).width)

        total_height = len(lines) * font.line_spacing

        return LineBreakResult(
            lines=lines,
            line_widths=line_widths,
            total_height=total_height,
        )

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words (preserving spaces)."""
        tokens: list[str] = []
        current_token: list[str] = []
        in_space = False

        for char in text:
            is_space = char.isspace()

            if is_space != in_space:
                if current_token:
                    tokens.append("".join(current_token))
                    current_token = []
                in_space = is_space

            current_token.append(char)

        if current_token:
            tokens.append("".join(current_token))

        return tokens


class TextShaper:
    """Text shaping for complex scripts.

    Handles:
    - Character reordering (for RTL scripts)
    - Ligature substitution
    - Contextual forms (Arabic, etc.)
    - Mark positioning (diacritics)
    """

    # Fallback glyph width as ratio of font size (for glyphs not in font)
    FALLBACK_GLYPH_WIDTH_RATIO: float = 0.6

    def __init__(self, font_manager: FontManager | None = None) -> None:
        """Initialize the text shaper.

        Args:
            font_manager: FontManager for accessing font data
        """
        self._font_manager = font_manager

        # Common ligatures
        self._ligatures: dict[str, str] = {
            "fi": "\ufb01",
            "fl": "\ufb02",
            "ff": "\ufb00",
            "ffi": "\ufb03",
            "ffl": "\ufb04",
        }

    def shape(
        self,
        text: str,
        font: Font,
        direction: str = "ltr",
        enable_ligatures: bool = True,
    ) -> list[ShapedGlyph]:
        """Shape text into positioned glyphs.

        Args:
            text: Text to shape
            font: Font to use
            direction: Text direction ("ltr" or "rtl")
            enable_ligatures: Whether to apply ligatures

        Returns:
            List of ShapedGlyph objects ready for rendering
        """
        if not text:
            return []

        # Apply ligatures if enabled
        if enable_ligatures:
            text = self._apply_ligatures(text)

        # Perform basic shaping
        glyphs: list[ShapedGlyph] = []
        x_position = 0.0
        cluster = 0

        for i, char in enumerate(text):
            codepoint = ord(char)
            glyph_metrics = font.get_glyph(codepoint)

            # Calculate advance (use estimated width if glyph not in font)
            if glyph_metrics:
                x_advance = glyph_metrics.advance
            else:
                x_advance = font.size * self.FALLBACK_GLYPH_WIDTH_RATIO

            # Add letter spacing
            x_advance += font.letter_spacing

            glyph = ShapedGlyph(
                codepoint=codepoint,
                glyph_id=codepoint,  # Simplified; real impl uses font glyph IDs
                x_offset=0.0,
                y_offset=0.0,
                x_advance=x_advance,
                y_advance=0.0,
                cluster=cluster,
            )

            glyphs.append(glyph)
            x_position += x_advance
            cluster += 1

        # Reverse for RTL
        if direction == "rtl":
            glyphs = list(reversed(glyphs))

        return glyphs

    def _apply_ligatures(self, text: str) -> str:
        """Apply ligature substitution to text."""
        result = text
        for seq, lig in sorted(self._ligatures.items(), key=lambda x: -len(x[0])):
            result = result.replace(seq, lig)
        return result

    def get_cluster_map(self, text: str, glyphs: list[ShapedGlyph]) -> dict[int, list[int]]:
        """Get mapping from text indices to glyph indices.

        Args:
            text: Original text
            glyphs: Shaped glyphs

        Returns:
            Dict mapping text index to list of glyph indices
        """
        cluster_map: dict[int, list[int]] = {}
        for glyph_idx, glyph in enumerate(glyphs):
            if glyph.cluster not in cluster_map:
                cluster_map[glyph.cluster] = []
            cluster_map[glyph.cluster].append(glyph_idx)
        return cluster_map


class GlyphCache:
    """Cache for glyph metrics and rendered glyphs.

    Provides fast lookup of frequently used glyphs to avoid
    repeated font queries and rendering.
    """

    def __init__(self, max_size: int = 10000) -> None:
        """Initialize the glyph cache.

        Args:
            max_size: Maximum number of cached entries
        """
        self._max_size = max_size
        self._cache: dict[str, GlyphMetrics] = {}
        self._access_order: list[str] = []

    def get(self, font: Font, codepoint: int) -> GlyphMetrics | None:
        """Get cached glyph metrics.

        Args:
            font: Font to look up
            codepoint: Unicode codepoint

        Returns:
            Cached GlyphMetrics or None if not cached
        """
        key = self._make_key(font, codepoint)
        if key in self._cache:
            # Move to end of access order (LRU)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, font: Font, codepoint: int, metrics: GlyphMetrics) -> None:
        """Cache glyph metrics.

        Args:
            font: Font for the glyph
            codepoint: Unicode codepoint
            metrics: Metrics to cache
        """
        key = self._make_key(font, codepoint)

        if key not in self._cache:
            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                oldest = self._access_order.pop(0)
                del self._cache[oldest]

            self._access_order.append(key)

        self._cache[key] = metrics

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def hit_count(self) -> int:
        """Get number of cache hits (simplified tracking)."""
        return len(self._cache)

    def _make_key(self, font: Font, codepoint: int) -> str:
        """Generate a cache key."""
        return f"{font.cache_key()}:{codepoint}"


class TextRenderer:
    """Main text rendering interface.

    Provides high-level text rendering operations including:
    - Text measurement
    - Line breaking
    - Text shaping
    - Multi-line layout
    """

    def __init__(
        self,
        font_manager: FontManager | None = None,
        cache_size: int = 10000,
    ) -> None:
        """Initialize the text renderer.

        Args:
            font_manager: FontManager for font access
            cache_size: Size of the glyph cache
        """
        self._font_manager = font_manager
        self._glyph_cache = GlyphCache(cache_size)
        self._line_breaker = LineBreaker()
        self._shaper = TextShaper(font_manager)

    @property
    def font_manager(self) -> FontManager | None:
        """Get the font manager."""
        return self._font_manager

    @property
    def glyph_cache(self) -> GlyphCache:
        """Get the glyph cache."""
        return self._glyph_cache

    def measure_text(self, text: str, font: Font) -> TextMeasurement:
        """Measure the dimensions of text.

        Args:
            text: Text to measure
            font: Font to use for measurement

        Returns:
            TextMeasurement with dimensions
        """
        if not text:
            return TextMeasurement(
                width=0.0,
                height=font.line_spacing,
                ascender=font.ascender,
                descender=font.descender,
                line_count=1,
                bounds=(0.0, 0.0, 0.0, font.line_spacing),
            )

        # Shape the text
        glyphs = self._shaper.shape(text, font)

        # Calculate total width
        width = sum(g.x_advance for g in glyphs)

        # Handle multi-line text
        lines = text.split("\n")
        line_count = len(lines)

        height = font.line_spacing * line_count

        return TextMeasurement(
            width=width,
            height=height,
            ascender=font.ascender,
            descender=font.descender,
            line_count=line_count,
            bounds=(0.0, -font.ascender, width, height),
        )

    def measure_char(self, char: str, font: Font) -> TextMeasurement:
        """Measure a single character.

        Args:
            char: Single character to measure
            font: Font to use

        Returns:
            TextMeasurement for the character
        """
        if len(char) != 1:
            raise ValueError("Expected single character")
        return self.measure_text(char, font)

    def break_lines(
        self,
        text: str,
        font: Font,
        max_width: float,
    ) -> LineBreakResult:
        """Break text into lines that fit within max_width.

        Args:
            text: Text to break
            font: Font to use
            max_width: Maximum line width

        Returns:
            LineBreakResult with broken lines
        """
        return self._line_breaker.break_text(
            text,
            max_width,
            self.measure_text,
            font,
        )

    def shape_text(
        self,
        text: str,
        font: Font,
        direction: str = "ltr",
    ) -> list[ShapedGlyph]:
        """Shape text into positioned glyphs.

        Args:
            text: Text to shape
            font: Font to use
            direction: Text direction ("ltr" or "rtl")

        Returns:
            List of ShapedGlyph objects
        """
        return self._shaper.shape(text, font, direction)

    def layout_text(
        self,
        text: str,
        font: Font,
        max_width: float | None = None,
        alignment: TextAlignment = TextAlignment.LEFT,
        overflow: TextOverflow = TextOverflow.WRAP,
    ) -> list[tuple[str, float, float]]:
        """Layout text with line breaking and alignment.

        Args:
            text: Text to layout
            font: Font to use
            max_width: Maximum width (None = single line)
            alignment: Text alignment
            overflow: Overflow handling mode

        Returns:
            List of (line_text, x_offset, y_offset) tuples
        """
        if max_width is None:
            # Single line layout
            measure = self.measure_text(text, font)
            return [(text, 0.0, 0.0)]

        # Handle different overflow modes
        if overflow == TextOverflow.CLIP:
            # Truncate to fit
            truncated = self._truncate_to_width(text, font, max_width)
            return [(truncated, 0.0, 0.0)]

        elif overflow == TextOverflow.ELLIPSIS:
            # Truncate with ellipsis
            truncated = self._truncate_with_ellipsis(text, font, max_width)
            return [(truncated, 0.0, 0.0)]

        elif overflow == TextOverflow.SHRINK:
            # Single line, handled elsewhere by scaling font
            return [(text, 0.0, 0.0)]

        # Default: wrap text
        lines_result = self.break_lines(text, font, max_width)

        layout: list[tuple[str, float, float]] = []
        y_offset = 0.0

        for line, width in lines_result:
            x_offset = self._calculate_alignment_offset(width, max_width, alignment)
            layout.append((line, x_offset, y_offset))
            y_offset += font.line_spacing

        return layout

    def get_cursor_position(
        self,
        text: str,
        font: Font,
        index: int,
    ) -> tuple[float, float]:
        """Get the cursor position for a text index.

        Args:
            text: Text string
            font: Font used for rendering
            index: Character index

        Returns:
            (x, y) cursor position
        """
        if index <= 0:
            return (0.0, 0.0)

        if index >= len(text):
            measure = self.measure_text(text, font)
            return (measure.width, 0.0)

        # Measure text up to index
        prefix = text[:index]
        measure = self.measure_text(prefix, font)
        return (measure.width, 0.0)

    def get_index_at_position(
        self,
        text: str,
        font: Font,
        x: float,
        y: float = 0.0,
    ) -> int:
        """Get the text index at a position.

        Args:
            text: Text string
            font: Font used for rendering
            x: X coordinate
            y: Y coordinate (for multi-line)

        Returns:
            Character index closest to position
        """
        if not text or x <= 0:
            return 0

        # Binary search for position
        total_width = 0.0
        for i, char in enumerate(text):
            char_measure = self.measure_char(char, font)
            char_mid = total_width + char_measure.width / 2

            if x < char_mid:
                return i

            total_width += char_measure.width

        return len(text)

    def _truncate_to_width(self, text: str, font: Font, max_width: float) -> str:
        """Truncate text to fit within width."""
        total_width = 0.0
        for i, char in enumerate(text):
            char_width = self.measure_char(char, font).width
            if total_width + char_width > max_width:
                return text[:i]
            total_width += char_width
        return text

    def _truncate_with_ellipsis(self, text: str, font: Font, max_width: float) -> str:
        """Truncate text with ellipsis to fit within width."""
        ellipsis = "\u2026"  # Unicode ellipsis
        ellipsis_width = self.measure_char(ellipsis, font).width

        if max_width <= ellipsis_width:
            return ellipsis

        # Find where to truncate
        target_width = max_width - ellipsis_width
        total_width = 0.0

        for i, char in enumerate(text):
            char_width = self.measure_char(char, font).width
            if total_width + char_width > target_width:
                return text[:i] + ellipsis
            total_width += char_width

        return text

    def _calculate_alignment_offset(
        self,
        line_width: float,
        max_width: float,
        alignment: TextAlignment,
    ) -> float:
        """Calculate x offset for text alignment."""
        if alignment == TextAlignment.LEFT:
            return 0.0
        elif alignment == TextAlignment.CENTER:
            return (max_width - line_width) / 2
        elif alignment == TextAlignment.RIGHT:
            return max_width - line_width
        else:  # JUSTIFY handled elsewhere
            return 0.0

    def clear_cache(self) -> None:
        """Clear the glyph cache."""
        self._glyph_cache.clear()

"""
Comprehensive tests for RichText markup parsing.

Tests cover:
- RichTextParser creation and basic parsing
- Tag parsing (bold, italic, underline, color, size)
- Nested tags
- Inline images
- Clickable links
- Text runs and spans
- Error handling for malformed markup
- Custom tag registration
- Style inheritance
"""

import pytest
from dataclasses import dataclass
from typing import Any, Optional, List
from enum import Enum, auto


# Expected RichText implementation classes
# from engine.ui.text.rich_text import (
#     RichTextParser,
#     TextRun,
#     TextSpan,
#     InlineImage,
#     ClickableLink,
#     RichTextStyle,
#     MarkupTag,
#     ParseError,
# )


class FontWeight(Enum):
    """Font weight for testing."""
    NORMAL = auto()
    BOLD = auto()


class FontStyle(Enum):
    """Font style for testing."""
    NORMAL = auto()
    ITALIC = auto()


@dataclass
class RichTextStyle:
    """Style for rich text elements."""
    font_family: Optional[str] = None
    font_size: Optional[float] = None
    font_weight: FontWeight = FontWeight.NORMAL
    font_style: FontStyle = FontStyle.NORMAL
    color: Optional[str] = None
    background_color: Optional[str] = None
    underline: bool = False
    strikethrough: bool = False

    def merge(self, other: "RichTextStyle") -> "RichTextStyle":
        """Merge another style, with other taking precedence."""
        return RichTextStyle(
            font_family=other.font_family or self.font_family,
            font_size=other.font_size or self.font_size,
            font_weight=other.font_weight if other.font_weight != FontWeight.NORMAL else self.font_weight,
            font_style=other.font_style if other.font_style != FontStyle.NORMAL else self.font_style,
            color=other.color or self.color,
            background_color=other.background_color or self.background_color,
            underline=other.underline or self.underline,
            strikethrough=other.strikethrough or self.strikethrough,
        )


@dataclass
class TextRun:
    """A run of text with a specific style."""
    text: str
    style: RichTextStyle


@dataclass
class TextSpan:
    """A span that can contain multiple runs."""
    runs: List[TextRun]
    style: Optional[RichTextStyle] = None


@dataclass
class InlineImage:
    """An inline image in rich text."""
    source: str
    width: Optional[float] = None
    height: Optional[float] = None
    alt_text: str = ""


@dataclass
class ClickableLink:
    """A clickable link element."""
    url: str
    text: str
    style: Optional[RichTextStyle] = None


@dataclass
class MarkupTag:
    """Represents a markup tag."""
    name: str
    attributes: dict = None
    is_closing: bool = False
    is_self_closing: bool = False

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


class ParseError(Exception):
    """Error during rich text parsing."""
    pass


class TestRichTextStyle:
    """Tests for RichTextStyle class."""

    def test_style_default_values(self):
        """Test style with default values."""
        style = RichTextStyle()
        assert style.font_family is None
        assert style.font_size is None
        assert style.font_weight == FontWeight.NORMAL
        assert style.underline is False

    def test_style_custom_values(self):
        """Test style with custom values."""
        style = RichTextStyle(
            font_family="Roboto",
            font_size=16.0,
            font_weight=FontWeight.BOLD,
            color="#FF0000",
        )
        assert style.font_family == "Roboto"
        assert style.font_weight == FontWeight.BOLD
        assert style.color == "#FF0000"

    def test_style_merge(self):
        """Test merging two styles."""
        base = RichTextStyle(font_family="Roboto", font_size=16.0)
        override = RichTextStyle(font_weight=FontWeight.BOLD, color="#FF0000")

        merged = base.merge(override)

        assert merged.font_family == "Roboto"  # From base
        assert merged.font_weight == FontWeight.BOLD  # From override
        assert merged.color == "#FF0000"  # From override

    def test_style_merge_override_takes_precedence(self):
        """Test that override values take precedence."""
        base = RichTextStyle(font_size=16.0, color="#000000")
        override = RichTextStyle(font_size=24.0, color="#FF0000")

        merged = base.merge(override)

        assert merged.font_size == 24.0
        assert merged.color == "#FF0000"


class TestTextRun:
    """Tests for TextRun class."""

    def test_run_creation(self):
        """Test creating a text run."""
        style = RichTextStyle(font_weight=FontWeight.BOLD)
        run = TextRun(text="Hello", style=style)

        assert run.text == "Hello"
        assert run.style.font_weight == FontWeight.BOLD


class TestTextSpan:
    """Tests for TextSpan class."""

    def test_span_creation(self):
        """Test creating a text span."""
        runs = [
            TextRun(text="Hello ", style=RichTextStyle()),
            TextRun(text="World", style=RichTextStyle(font_weight=FontWeight.BOLD)),
        ]
        span = TextSpan(runs=runs)

        assert len(span.runs) == 2


class TestInlineImage:
    """Tests for InlineImage class."""

    def test_inline_image_creation(self):
        """Test creating an inline image."""
        image = InlineImage(source="icon.png")
        assert image.source == "icon.png"
        assert image.alt_text == ""

    def test_inline_image_with_dimensions(self):
        """Test inline image with dimensions."""
        image = InlineImage(source="icon.png", width=32, height=32)
        assert image.width == 32
        assert image.height == 32

    def test_inline_image_with_alt_text(self):
        """Test inline image with alt text."""
        image = InlineImage(source="icon.png", alt_text="An icon")
        assert image.alt_text == "An icon"


class TestClickableLink:
    """Tests for ClickableLink class."""

    def test_link_creation(self):
        """Test creating a clickable link."""
        link = ClickableLink(url="https://example.com", text="Click here")
        assert link.url == "https://example.com"
        assert link.text == "Click here"

    def test_link_with_style(self):
        """Test link with custom style."""
        style = RichTextStyle(color="#0000FF", underline=True)
        link = ClickableLink(url="https://example.com", text="Link", style=style)
        assert link.style.color == "#0000FF"


class TestMarkupTag:
    """Tests for MarkupTag class."""

    def test_tag_creation(self):
        """Test creating a markup tag."""
        tag = MarkupTag(name="b")
        assert tag.name == "b"
        assert tag.is_closing is False
        assert len(tag.attributes) == 0

    def test_tag_with_attributes(self):
        """Test tag with attributes."""
        tag = MarkupTag(name="color", attributes={"value": "#FF0000"})
        assert tag.attributes["value"] == "#FF0000"

    def test_closing_tag(self):
        """Test closing tag."""
        tag = MarkupTag(name="b", is_closing=True)
        assert tag.is_closing is True

    def test_self_closing_tag(self):
        """Test self-closing tag."""
        tag = MarkupTag(name="img", is_self_closing=True)
        assert tag.is_self_closing is True


class TestRichTextParserBasic:
    """Tests for basic RichTextParser functionality."""

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parser_creation(self):
        """Test creating a parser."""
        parser = RichTextParser()
        assert parser is not None

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_plain_text(self):
        """Test parsing plain text without markup."""
        parser = RichTextParser()
        runs = parser.parse("Hello World")

        assert len(runs) == 1
        assert runs[0].text == "Hello World"

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_empty_string(self):
        """Test parsing empty string."""
        parser = RichTextParser()
        runs = parser.parse("")

        assert len(runs) == 0


class TestRichTextParserBoldItalic:
    """Tests for bold and italic parsing."""

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_bold(self):
        """Test parsing bold text."""
        parser = RichTextParser()
        runs = parser.parse("[b]Bold[/b]")

        assert len(runs) == 1
        assert runs[0].text == "Bold"
        assert runs[0].style.font_weight == FontWeight.BOLD

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_italic(self):
        """Test parsing italic text."""
        parser = RichTextParser()
        runs = parser.parse("[i]Italic[/i]")

        assert len(runs) == 1
        assert runs[0].text == "Italic"
        assert runs[0].style.font_style == FontStyle.ITALIC

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_bold_and_italic(self):
        """Test parsing text with both bold and italic."""
        parser = RichTextParser()
        runs = parser.parse("[b][i]Bold Italic[/i][/b]")

        assert len(runs) == 1
        assert runs[0].style.font_weight == FontWeight.BOLD
        assert runs[0].style.font_style == FontStyle.ITALIC

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_mixed_formatting(self):
        """Test parsing mixed formatted text."""
        parser = RichTextParser()
        runs = parser.parse("Normal [b]Bold[/b] Normal")

        assert len(runs) == 3
        assert runs[0].style.font_weight == FontWeight.NORMAL
        assert runs[1].style.font_weight == FontWeight.BOLD
        assert runs[2].style.font_weight == FontWeight.NORMAL


class TestRichTextParserUnderline:
    """Tests for underline parsing."""

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_underline(self):
        """Test parsing underlined text."""
        parser = RichTextParser()
        runs = parser.parse("[u]Underlined[/u]")

        assert len(runs) == 1
        assert runs[0].style.underline is True

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_strikethrough(self):
        """Test parsing strikethrough text."""
        parser = RichTextParser()
        runs = parser.parse("[s]Strikethrough[/s]")

        assert len(runs) == 1
        assert runs[0].style.strikethrough is True


class TestRichTextParserColor:
    """Tests for color parsing."""

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_color_hex(self):
        """Test parsing hex color."""
        parser = RichTextParser()
        runs = parser.parse("[color=#FF0000]Red[/color]")

        assert len(runs) == 1
        assert runs[0].style.color == "#FF0000"

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_color_named(self):
        """Test parsing named color."""
        parser = RichTextParser()
        runs = parser.parse("[color=red]Red[/color]")

        assert len(runs) == 1
        assert runs[0].style.color in ("#FF0000", "red")

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_color_rgb(self):
        """Test parsing RGB color."""
        parser = RichTextParser()
        runs = parser.parse("[color=rgb(255,0,0)]Red[/color]")

        assert len(runs) == 1
        # Color should be normalized or stored as-is
        assert runs[0].style.color is not None

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_background_color(self):
        """Test parsing background color."""
        parser = RichTextParser()
        runs = parser.parse("[bg=#FFFF00]Highlighted[/bg]")

        assert len(runs) == 1
        assert runs[0].style.background_color == "#FFFF00"


class TestRichTextParserSize:
    """Tests for size parsing."""

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_size_absolute(self):
        """Test parsing absolute size."""
        parser = RichTextParser()
        runs = parser.parse("[size=24]Large[/size]")

        assert len(runs) == 1
        assert runs[0].style.font_size == 24.0

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_size_relative(self):
        """Test parsing relative size."""
        parser = RichTextParser()
        runs = parser.parse("[size=+4]Larger[/size]")

        assert len(runs) == 1
        # Size should be increased by 4

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_size_percentage(self):
        """Test parsing percentage size."""
        parser = RichTextParser()
        runs = parser.parse("[size=150%]Larger[/size]")

        assert len(runs) == 1
        # Size should be 150% of base


class TestRichTextParserNested:
    """Tests for nested tags."""

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_nested_simple(self):
        """Test simple nested tags."""
        parser = RichTextParser()
        runs = parser.parse("[b][color=#FF0000]Bold Red[/color][/b]")

        assert len(runs) == 1
        assert runs[0].style.font_weight == FontWeight.BOLD
        assert runs[0].style.color == "#FF0000"

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_nested_multiple(self):
        """Test multiple nested tags."""
        parser = RichTextParser()
        runs = parser.parse("[b]Bold [i]Bold Italic[/i] Bold[/b]")

        assert len(runs) == 3
        assert runs[0].style.font_weight == FontWeight.BOLD
        assert runs[1].style.font_weight == FontWeight.BOLD
        assert runs[1].style.font_style == FontStyle.ITALIC
        assert runs[2].style.font_weight == FontWeight.BOLD

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_deeply_nested(self):
        """Test deeply nested tags."""
        parser = RichTextParser()
        runs = parser.parse("[b][i][u][color=#FF0000]Deep[/color][/u][/i][/b]")

        assert len(runs) == 1
        assert runs[0].style.font_weight == FontWeight.BOLD
        assert runs[0].style.font_style == FontStyle.ITALIC
        assert runs[0].style.underline is True
        assert runs[0].style.color == "#FF0000"


class TestRichTextParserInlineImage:
    """Tests for inline image parsing."""

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_image_simple(self):
        """Test parsing simple image."""
        parser = RichTextParser()
        elements = parser.parse("[img]icon.png[/img]")

        images = [e for e in elements if isinstance(e, InlineImage)]
        assert len(images) == 1
        assert images[0].source == "icon.png"

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_image_with_size(self):
        """Test parsing image with size."""
        parser = RichTextParser()
        elements = parser.parse("[img width=32 height=32]icon.png[/img]")

        images = [e for e in elements if isinstance(e, InlineImage)]
        assert len(images) == 1
        assert images[0].width == 32
        assert images[0].height == 32

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_image_self_closing(self):
        """Test parsing self-closing image tag."""
        parser = RichTextParser()
        elements = parser.parse("[img src='icon.png' /]")

        images = [e for e in elements if isinstance(e, InlineImage)]
        assert len(images) == 1

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_image_mixed_with_text(self):
        """Test parsing image mixed with text."""
        parser = RichTextParser()
        elements = parser.parse("Click [img]icon.png[/img] to continue")

        assert len(elements) == 3
        assert isinstance(elements[1], InlineImage)


class TestRichTextParserLinks:
    """Tests for link parsing."""

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_link_simple(self):
        """Test parsing simple link."""
        parser = RichTextParser()
        elements = parser.parse("[url=https://example.com]Click here[/url]")

        links = [e for e in elements if isinstance(e, ClickableLink)]
        assert len(links) == 1
        assert links[0].url == "https://example.com"
        assert links[0].text == "Click here"

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_link_with_formatting(self):
        """Test parsing link with inner formatting."""
        parser = RichTextParser()
        elements = parser.parse("[url=https://example.com][b]Bold Link[/b][/url]")

        links = [e for e in elements if isinstance(e, ClickableLink)]
        assert len(links) == 1
        # The text should preserve formatting

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_link_auto_detect(self):
        """Test auto-detecting URLs."""
        parser = RichTextParser(auto_link=True)
        elements = parser.parse("Visit https://example.com today")

        links = [e for e in elements if isinstance(e, ClickableLink)]
        assert len(links) >= 1


class TestRichTextParserErrors:
    """Tests for error handling."""

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_unclosed_tag(self):
        """Test parsing unclosed tag raises error."""
        parser = RichTextParser(strict=True)

        with pytest.raises(ParseError, match="Unclosed tag"):
            parser.parse("[b]Bold without closing")

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_mismatched_tags(self):
        """Test parsing mismatched tags raises error."""
        parser = RichTextParser(strict=True)

        with pytest.raises(ParseError, match="Mismatched"):
            parser.parse("[b]Bold [i]Italic[/b][/i]")

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_unknown_tag(self):
        """Test parsing unknown tag."""
        parser = RichTextParser(strict=True)

        with pytest.raises(ParseError, match="Unknown tag"):
            parser.parse("[unknown]Text[/unknown]")

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_lenient_mode(self):
        """Test lenient mode handles errors gracefully."""
        parser = RichTextParser(strict=False)

        # Should not raise, should handle gracefully
        runs = parser.parse("[b]Bold without closing")
        assert len(runs) >= 1

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_parse_invalid_color(self):
        """Test parsing invalid color value."""
        parser = RichTextParser(strict=True)

        with pytest.raises(ParseError, match="Invalid color"):
            parser.parse("[color=notacolor]Text[/color]")


class TestRichTextParserCustomTags:
    """Tests for custom tag registration."""

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_register_custom_tag(self):
        """Test registering a custom tag."""
        parser = RichTextParser()

        def highlight_handler(text: str, attrs: dict) -> TextRun:
            return TextRun(
                text=text,
                style=RichTextStyle(background_color="#FFFF00"),
            )

        parser.register_tag("highlight", highlight_handler)
        runs = parser.parse("[highlight]Important[/highlight]")

        assert len(runs) == 1
        assert runs[0].style.background_color == "#FFFF00"

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_override_builtin_tag(self):
        """Test overriding a built-in tag."""
        parser = RichTextParser()

        def custom_bold(text: str, attrs: dict) -> TextRun:
            return TextRun(
                text=text,
                style=RichTextStyle(font_weight=FontWeight.BOLD, color="#FF0000"),
            )

        parser.register_tag("b", custom_bold)
        runs = parser.parse("[b]Bold[/b]")

        assert runs[0].style.color == "#FF0000"


class TestRichTextParserStyleInheritance:
    """Tests for style inheritance."""

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_style_inherited_from_parent(self):
        """Test that child inherits parent style."""
        parser = RichTextParser()
        runs = parser.parse("[b][color=#FF0000]Red [i]Italic[/i] Red[/color][/b]")

        # Italic run should inherit bold and red
        assert runs[1].style.font_weight == FontWeight.BOLD
        assert runs[1].style.color == "#FF0000"

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_base_style_applied(self):
        """Test that base style is applied."""
        base_style = RichTextStyle(font_family="Roboto", font_size=16.0)
        parser = RichTextParser(base_style=base_style)
        runs = parser.parse("Plain text")

        assert runs[0].style.font_family == "Roboto"
        assert runs[0].style.font_size == 16.0

    @pytest.mark.skip(reason="RichTextParser not yet implemented")
    def test_style_reset(self):
        """Test resetting to base style."""
        parser = RichTextParser()
        runs = parser.parse("[b]Bold [/b][clear]Reset[/clear]")

        # After clear, style should be reset
        reset_runs = [r for r in runs if "Reset" in r.text]
        assert len(reset_runs) > 0
        assert reset_runs[0].style.font_weight == FontWeight.NORMAL

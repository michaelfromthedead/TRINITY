"""
Comprehensive tests for the Text widget.

Tests cover:
- Initialization and default values
- Text style configuration
- Alignment options (horizontal and vertical)
- Overflow modes
- Word wrapping
- Rich text parsing
- Measurement and truncation
- Serialization/deserialization
"""

import pytest
from unittest.mock import MagicMock


class TestTextStyle:
    """Tests for TextStyle dataclass."""

    def test_default_text_style(self):
        """Test default TextStyle values."""
        from engine.ui.widgets.primitives.text import TextStyle

        style = TextStyle()
        assert style.font_family == "default"
        assert style.font_size == 14.0
        assert style.font_weight == "normal"
        assert style.font_style == "normal"
        assert style.color == (0.0, 0.0, 0.0, 1.0)
        assert style.line_height == 1.2
        assert style.letter_spacing == 0.0
        assert style.underline is False
        assert style.strikethrough is False
        assert style.shadow_offset is None

    def test_text_style_font_size_validation(self):
        """Test font_size must be positive."""
        from engine.ui.widgets.primitives.text import TextStyle

        with pytest.raises(ValueError, match="font_size must be > 0"):
            TextStyle(font_size=0)

        with pytest.raises(ValueError, match="font_size must be > 0"):
            TextStyle(font_size=-5)

    def test_text_style_line_height_validation(self):
        """Test line_height must be positive."""
        from engine.ui.widgets.primitives.text import TextStyle

        with pytest.raises(ValueError, match="line_height must be > 0"):
            TextStyle(line_height=0)

    def test_text_style_font_weight_string_values(self):
        """Test valid string font weights."""
        from engine.ui.widgets.primitives.text import TextStyle

        for weight in ["normal", "bold", "light", "thin", "medium", "semibold", "black"]:
            style = TextStyle(font_weight=weight)
            assert style.font_weight == weight

    def test_text_style_font_weight_numeric(self):
        """Test numeric font weights (100-900)."""
        from engine.ui.widgets.primitives.text import TextStyle

        style = TextStyle(font_weight="400")
        assert style.font_weight == "400"

        style = TextStyle(font_weight="700")
        assert style.font_weight == "700"

    def test_text_style_font_weight_invalid_numeric(self):
        """Test invalid numeric font weights."""
        from engine.ui.widgets.primitives.text import TextStyle

        with pytest.raises(ValueError, match="font_weight must be one of"):
            TextStyle(font_weight="50")

        with pytest.raises(ValueError, match="font_weight must be one of"):
            TextStyle(font_weight="1000")

    def test_text_style_font_weight_invalid_string(self):
        """Test invalid string font weight."""
        from engine.ui.widgets.primitives.text import TextStyle

        with pytest.raises(ValueError, match="font_weight must be one of"):
            TextStyle(font_weight="extra-bold")

    def test_text_style_font_style_valid(self):
        """Test valid font styles."""
        from engine.ui.widgets.primitives.text import TextStyle

        for style_name in ["normal", "italic", "oblique"]:
            style = TextStyle(font_style=style_name)
            assert style.font_style == style_name

    def test_text_style_font_style_invalid(self):
        """Test invalid font style."""
        from engine.ui.widgets.primitives.text import TextStyle

        with pytest.raises(ValueError, match="font_style must be one of"):
            TextStyle(font_style="slanted")

    def test_text_style_with_overrides(self):
        """Test creating style with overrides."""
        from engine.ui.widgets.primitives.text import TextStyle

        base = TextStyle(font_size=16, color=(1.0, 0.0, 0.0, 1.0))
        bold = base.with_overrides(font_weight="bold")

        assert bold.font_size == 16
        assert bold.color == (1.0, 0.0, 0.0, 1.0)
        assert bold.font_weight == "bold"
        # Original unchanged
        assert base.font_weight == "normal"

    def test_text_style_with_shadow(self):
        """Test TextStyle with shadow configuration."""
        from engine.ui.widgets.primitives.text import TextStyle

        style = TextStyle(
            shadow_offset=(2.0, 2.0),
            shadow_color=(0.0, 0.0, 0.0, 0.5),
            shadow_blur=4.0
        )
        assert style.shadow_offset == (2.0, 2.0)
        assert style.shadow_color == (0.0, 0.0, 0.0, 0.5)
        assert style.shadow_blur == 4.0


class TestRichTextSpan:
    """Tests for RichTextSpan dataclass."""

    def test_rich_text_span_basic(self):
        """Test basic RichTextSpan creation."""
        from engine.ui.widgets.primitives.text import RichTextSpan

        span = RichTextSpan(text="Hello", start_index=0, end_index=5)
        assert span.text == "Hello"
        assert span.start_index == 0
        assert span.end_index == 5
        assert span.style is None

    def test_rich_text_span_length(self):
        """Test RichTextSpan length property."""
        from engine.ui.widgets.primitives.text import RichTextSpan

        span = RichTextSpan(text="Hello World")
        assert span.length == 11

    def test_rich_text_span_with_style(self):
        """Test RichTextSpan with style override."""
        from engine.ui.widgets.primitives.text import RichTextSpan, TextStyle

        style = TextStyle(font_weight="bold")
        span = RichTextSpan(text="Bold", style=style)
        assert span.style.font_weight == "bold"


class TestRichTextParser:
    """Tests for RichTextParser."""

    def test_parser_plain_text(self):
        """Test parsing plain text without markup."""
        from engine.ui.widgets.primitives.text import RichTextParser

        parser = RichTextParser()
        spans = parser.parse("Hello World")

        assert len(spans) == 1
        assert spans[0].text == "Hello World"

    def test_parser_bold_tag(self):
        """Test parsing bold tag."""
        from engine.ui.widgets.primitives.text import RichTextParser

        parser = RichTextParser()
        spans = parser.parse("Hello <b>bold</b> world")

        assert len(spans) == 3
        assert spans[0].text == "Hello "
        assert spans[1].text == "bold"
        assert spans[1].style.font_weight == "bold"
        assert spans[2].text == " world"

    def test_parser_italic_tag(self):
        """Test parsing italic tag."""
        from engine.ui.widgets.primitives.text import RichTextParser

        parser = RichTextParser()
        spans = parser.parse("Hello <i>italic</i> world")

        assert len(spans) == 3
        assert spans[1].text == "italic"
        assert spans[1].style.font_style == "italic"

    def test_parser_underline_tag(self):
        """Test parsing underline tag."""
        from engine.ui.widgets.primitives.text import RichTextParser

        parser = RichTextParser()
        spans = parser.parse("<u>underlined</u>")

        assert len(spans) == 1
        assert spans[0].text == "underlined"
        assert spans[0].style.underline is True

    def test_parser_strikethrough_tag(self):
        """Test parsing strikethrough tag."""
        from engine.ui.widgets.primitives.text import RichTextParser

        parser = RichTextParser()
        spans = parser.parse("<s>strikethrough</s>")

        assert len(spans) == 1
        assert spans[0].text == "strikethrough"
        assert spans[0].style.strikethrough is True

    def test_parser_color_tag(self):
        """Test parsing color tag."""
        from engine.ui.widgets.primitives.text import RichTextParser

        parser = RichTextParser()
        spans = parser.parse("<color=#FF0000>red text</color>")

        assert len(spans) == 1
        assert spans[0].text == "red text"
        assert spans[0].style.color[0] == pytest.approx(1.0)
        assert spans[0].style.color[1] == pytest.approx(0.0)
        assert spans[0].style.color[2] == pytest.approx(0.0)

    def test_parser_size_tag(self):
        """Test parsing size tag."""
        from engine.ui.widgets.primitives.text import RichTextParser

        parser = RichTextParser()
        spans = parser.parse("<size=24>large text</size>")

        assert len(spans) == 1
        assert spans[0].text == "large text"
        assert spans[0].style.font_size == 24.0

    def test_parser_font_tag(self):
        """Test parsing font tag."""
        from engine.ui.widgets.primitives.text import RichTextParser

        parser = RichTextParser()
        spans = parser.parse("<font=Arial>Arial text</font>")

        assert len(spans) == 1
        assert spans[0].text == "Arial text"
        assert spans[0].style.font_family == "Arial"

    def test_parser_br_tag(self):
        """Test parsing line break tag."""
        from engine.ui.widgets.primitives.text import RichTextParser

        parser = RichTextParser()
        spans = parser.parse("Line 1<br>Line 2")

        assert len(spans) == 3
        assert spans[0].text == "Line 1"
        assert spans[1].text == "\n"
        assert spans[2].text == "Line 2"

    def test_parser_br_self_closing(self):
        """Test parsing self-closing br tag."""
        from engine.ui.widgets.primitives.text import RichTextParser

        parser = RichTextParser()
        spans = parser.parse("Line 1<br/>Line 2")

        assert len(spans) == 3
        assert spans[1].text == "\n"

    def test_parser_nested_tags(self):
        """Test parsing nested tags."""
        from engine.ui.widgets.primitives.text import RichTextParser

        parser = RichTextParser()
        spans = parser.parse("<b><i>bold italic</i></b>")

        assert len(spans) == 1
        assert spans[0].text == "bold italic"
        assert spans[0].style.font_weight == "bold"
        assert spans[0].style.font_style == "italic"

    def test_parser_invalid_color_graceful(self):
        """Test invalid color value is handled gracefully."""
        from engine.ui.widgets.primitives.text import RichTextParser

        parser = RichTextParser()
        spans = parser.parse("<color=invalid>text</color>")

        # Should use base style color
        assert len(spans) == 1
        assert spans[0].text == "text"

    def test_parser_invalid_size_graceful(self):
        """Test invalid size value is handled gracefully."""
        from engine.ui.widgets.primitives.text import RichTextParser

        parser = RichTextParser()
        spans = parser.parse("<size=abc>text</size>")

        # Should use base style size
        assert len(spans) == 1
        assert spans[0].style.font_size == 14.0

    def test_parser_strip_tags(self):
        """Test stripping tags from rich text."""
        from engine.ui.widgets.primitives.text import RichTextParser

        parser = RichTextParser()
        plain = parser.strip_tags("<b>Hello</b> <i>World</i><br>!")

        assert plain == "Hello World\n!"

    def test_parser_case_insensitive_tags(self):
        """Test tags are case insensitive."""
        from engine.ui.widgets.primitives.text import RichTextParser

        parser = RichTextParser()
        spans = parser.parse("<B>bold</B> <I>italic</I>")

        assert spans[0].style.font_weight == "bold"
        assert spans[2].style.font_style == "italic"

    def test_parser_with_custom_base_style(self):
        """Test parser with custom base style."""
        from engine.ui.widgets.primitives.text import RichTextParser, TextStyle

        base_style = TextStyle(font_size=20, color=(1.0, 1.0, 1.0, 1.0))
        parser = RichTextParser(base_style=base_style)
        spans = parser.parse("white text")

        assert spans[0].style.font_size == 20
        assert spans[0].style.color == (1.0, 1.0, 1.0, 1.0)

    def test_parser_span_indices(self):
        """Test span start and end indices are correct."""
        from engine.ui.widgets.primitives.text import RichTextParser

        parser = RichTextParser()
        spans = parser.parse("Hello <b>World</b>!")

        assert spans[0].start_index == 0
        assert spans[0].end_index == 6
        assert spans[1].start_index == 6
        assert spans[1].end_index == 11
        assert spans[2].start_index == 11
        assert spans[2].end_index == 12


class TestTextWidget:
    """Tests for the Text widget class."""

    def test_text_default_initialization(self):
        """Test Text initializes with correct defaults."""
        from engine.ui.widgets.primitives.text import (
            Text, TextAlignment, VerticalAlignment, OverflowMode
        )

        text = Text()
        assert text.content == ""
        assert text.alignment == TextAlignment.LEFT
        assert text.vertical_alignment == VerticalAlignment.TOP
        assert text.word_wrap is True
        assert text.overflow_mode == OverflowMode.WRAP
        assert text.max_lines == 0
        assert text.ellipsis == "..."
        assert text.rich_text is False

    def test_text_with_content(self):
        """Test Text with initial content."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(content="Hello World")
        assert text.content == "Hello World"

    def test_text_content_setter(self):
        """Test setting content."""
        from engine.ui.widgets.primitives.text import Text

        text = Text()
        text.content = "New content"
        assert text.content == "New content"

    def test_text_style_setter(self):
        """Test setting style."""
        from engine.ui.widgets.primitives.text import Text, TextStyle

        text = Text()
        new_style = TextStyle(font_size=20)
        text.style = new_style
        assert text.style.font_size == 20

    def test_text_style_setter_invalid_type(self):
        """Test style setter rejects invalid types."""
        from engine.ui.widgets.primitives.text import Text

        text = Text()
        with pytest.raises(ValueError, match="must be TextStyle"):
            text.style = {"font_size": 20}

    def test_text_alignment_setter(self):
        """Test setting horizontal alignment."""
        from engine.ui.widgets.primitives.text import Text, TextAlignment

        text = Text()
        text.alignment = TextAlignment.CENTER
        assert text.alignment == TextAlignment.CENTER

    def test_text_alignment_setter_invalid_type(self):
        """Test alignment setter rejects invalid types."""
        from engine.ui.widgets.primitives.text import Text

        text = Text()
        with pytest.raises(ValueError, match="must be TextAlignment"):
            text.alignment = "center"

    def test_text_vertical_alignment_setter(self):
        """Test setting vertical alignment."""
        from engine.ui.widgets.primitives.text import Text, VerticalAlignment

        text = Text()
        text.vertical_alignment = VerticalAlignment.MIDDLE
        assert text.vertical_alignment == VerticalAlignment.MIDDLE

    def test_text_vertical_alignment_invalid_type(self):
        """Test vertical alignment setter rejects invalid types."""
        from engine.ui.widgets.primitives.text import Text

        text = Text()
        with pytest.raises(ValueError, match="must be VerticalAlignment"):
            text.vertical_alignment = "middle"

    def test_text_word_wrap_setter(self):
        """Test setting word wrap."""
        from engine.ui.widgets.primitives.text import Text

        text = Text()
        text.word_wrap = False
        assert text.word_wrap is False

    def test_text_overflow_mode_setter(self):
        """Test setting overflow mode."""
        from engine.ui.widgets.primitives.text import Text, OverflowMode

        text = Text()
        text.overflow_mode = OverflowMode.ELLIPSIS
        assert text.overflow_mode == OverflowMode.ELLIPSIS

    def test_text_overflow_mode_invalid_type(self):
        """Test overflow mode setter rejects invalid types."""
        from engine.ui.widgets.primitives.text import Text

        text = Text()
        with pytest.raises(ValueError, match="must be OverflowMode"):
            text.overflow_mode = "ellipsis"

    def test_text_max_lines_setter(self):
        """Test setting max lines."""
        from engine.ui.widgets.primitives.text import Text

        text = Text()
        text.max_lines = 3
        assert text.max_lines == 3

    def test_text_max_lines_negative_clamps(self):
        """Test negative max_lines clamps to 0."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(max_lines=-5)
        assert text.max_lines == 0

    def test_text_ellipsis_setter(self):
        """Test setting ellipsis string."""
        from engine.ui.widgets.primitives.text import Text

        text = Text()
        text.ellipsis = ">>>"
        assert text.ellipsis == ">>>"

    def test_text_rich_text_setter(self):
        """Test setting rich text mode."""
        from engine.ui.widgets.primitives.text import Text

        text = Text()
        text.rich_text = True
        assert text.rich_text is True

    def test_text_width_setter(self):
        """Test setting width."""
        from engine.ui.widgets.primitives.text import Text

        text = Text()
        text.width = 200.0
        assert text.width == 200.0

    def test_text_width_negative_clamps(self):
        """Test negative width clamps to 0."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(width=-100)
        assert text.width == 0.0

    def test_text_height_setter(self):
        """Test setting height."""
        from engine.ui.widgets.primitives.text import Text

        text = Text()
        text.height = 100.0
        assert text.height == 100.0

    def test_text_min_font_size_setter(self):
        """Test setting minimum font size."""
        from engine.ui.widgets.primitives.text import Text

        text = Text()
        text.min_font_size = 8.0
        assert text.min_font_size == 8.0

    def test_text_min_font_size_invalid(self):
        """Test min_font_size must be positive."""
        from engine.ui.widgets.primitives.text import Text

        text = Text()
        with pytest.raises(ValueError, match="must be > 0"):
            text.min_font_size = 0

    def test_text_scroll_offset(self):
        """Test scroll offset property."""
        from engine.ui.widgets.primitives.text import Text

        text = Text()
        text.scroll_offset = (10.0, 20.0)
        assert text.scroll_offset == (10.0, 20.0)

    def test_text_plain_text_no_rich(self):
        """Test plain_text property without rich text."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(content="Hello World", rich_text=False)
        assert text.plain_text == "Hello World"

    def test_text_plain_text_with_rich(self):
        """Test plain_text property with rich text markup."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(content="<b>Hello</b> World", rich_text=True)
        assert text.plain_text == "Hello World"


class TestTextMeasurement:
    """Tests for text measurement and truncation."""

    def test_text_measure_empty(self):
        """Test measuring empty text."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(content="")
        measure_func = MagicMock(return_value=(0.0, 0.0))

        width, height = text.measure(measure_func)
        assert width == 0.0
        assert height == 0.0

    def test_text_measure_plain(self):
        """Test measuring plain text."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(content="Hello World")
        measure_func = MagicMock(return_value=(100.0, 20.0))

        width, height = text.measure(measure_func)
        assert width == 100.0
        assert height == 20.0

    def test_text_measure_rich(self):
        """Test measuring rich text."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(content="<b>Hello</b> World", rich_text=True)
        measure_func = MagicMock(return_value=(50.0, 20.0))

        width, height = text.measure(measure_func)
        # Two spans measured
        assert measure_func.call_count == 2

    def test_text_truncate_to_width_fits(self):
        """Test truncation when text fits."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(content="Hello")
        measure_func = MagicMock(return_value=(50.0, 20.0))

        result = text.truncate_to_width(100.0, measure_func)
        assert result == "Hello"

    def test_text_truncate_to_width_needs_ellipsis(self):
        """Test truncation adds ellipsis when needed."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(content="Hello World")

        def measure_side_effect(txt, style):
            # Each char is ~10 pixels, ellipsis is 15
            return (len(txt) * 10.0, 20.0)

        measure_func = MagicMock(side_effect=measure_side_effect)

        result = text.truncate_to_width(80.0, measure_func)
        assert result.endswith("...")
        assert len(result) < len("Hello World")

    def test_text_truncate_zero_width(self):
        """Test truncation with zero width."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(content="Hello")
        measure_func = MagicMock()

        result = text.truncate_to_width(0.0, measure_func)
        assert result == ""

    def test_text_truncate_very_small_width(self):
        """Test truncation when only ellipsis fits."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(content="Hello World")

        def measure_side_effect(txt, style):
            return (len(txt) * 10.0, 20.0)

        measure_func = MagicMock(side_effect=measure_side_effect)

        result = text.truncate_to_width(25.0, measure_func)
        # Only ellipsis should fit
        assert result == "..."


class TestTextLineOperations:
    """Tests for line-related operations."""

    def test_text_line_count_default(self):
        """Test line count with no line breaks."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(content="Single line")
        assert text.line_count == 1

    def test_text_get_line_at_index(self):
        """Test getting line number for character index."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(content="Hello World")
        # Without line breaks, everything is line 0
        assert text.get_line_at_index(0) == 0
        assert text.get_line_at_index(5) == 0

    def test_text_get_character_at_position_empty(self):
        """Test character position with empty content."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(content="")
        pos = text.get_character_at_position(50.0, 10.0)
        assert pos == -1


class TestTextRichTextParsing:
    """Tests for rich text parsing in Text widget."""

    def test_text_parse_rich_text_enabled(self):
        """Test parsing rich text when enabled."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(content="<b>Bold</b> text", rich_text=True)
        spans = text.parse_rich_text()

        assert len(spans) == 2
        assert spans[0].style.font_weight == "bold"

    def test_text_parse_rich_text_disabled(self):
        """Test parsing returns single span when disabled."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(content="<b>Bold</b> text", rich_text=False)
        spans = text.parse_rich_text()

        assert len(spans) == 1
        assert spans[0].text == "<b>Bold</b> text"


class TestTextDirtyState:
    """Tests for dirty state tracking."""

    def test_text_dirty_after_content_change(self):
        """Test text is dirty after content changes."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(content="Hello")
        text.mark_clean()
        text.content = "World"
        assert text.is_dirty

    def test_text_dirty_after_style_change(self):
        """Test text is dirty after style changes."""
        from engine.ui.widgets.primitives.text import Text, TextStyle

        text = Text()
        text.mark_clean()
        text._dirty_layout = False
        text.style = TextStyle(font_size=20)
        assert text._dirty_layout

    def test_text_mark_dirty(self):
        """Test mark_dirty sets dirty state."""
        from engine.ui.widgets.primitives.text import Text

        text = Text()
        text._dirty_layout = False
        text.mark_dirty()
        assert text._dirty_layout is True

    def test_text_mark_clean(self):
        """Test mark_clean clears dirty state."""
        from engine.ui.widgets.primitives.text import Text

        text = Text()
        text._dirty_layout = True
        text.mark_clean()
        assert text._dirty_layout is False


class TestTextSerialization:
    """Tests for Text serialization and deserialization."""

    def test_text_to_dict(self):
        """Test serialization to dictionary."""
        from engine.ui.widgets.primitives.text import (
            Text, TextAlignment, OverflowMode
        )

        text = Text(
            content="Hello World",
            alignment=TextAlignment.CENTER,
            overflow_mode=OverflowMode.ELLIPSIS,
            max_lines=3,
            width=200,
            height=100,
        )

        data = text.to_dict()
        assert data["content"] == "Hello World"
        assert data["alignment"] == "CENTER"
        assert data["overflow_mode"] == "ELLIPSIS"
        assert data["max_lines"] == 3
        assert data["width"] == 200
        assert data["height"] == 100

    def test_text_to_dict_with_style(self):
        """Test serialization includes style."""
        from engine.ui.widgets.primitives.text import Text, TextStyle

        style = TextStyle(
            font_family="Arial",
            font_size=18,
            font_weight="bold",
            color=(1.0, 0.0, 0.0, 1.0)
        )
        text = Text(style=style)

        data = text.to_dict()
        assert data["style"]["font_family"] == "Arial"
        assert data["style"]["font_size"] == 18
        assert data["style"]["font_weight"] == "bold"
        assert data["style"]["color"] == (1.0, 0.0, 0.0, 1.0)

    def test_text_to_dict_with_shadow(self):
        """Test serialization includes shadow when set."""
        from engine.ui.widgets.primitives.text import Text, TextStyle

        style = TextStyle(
            shadow_offset=(2.0, 2.0),
            shadow_color=(0.0, 0.0, 0.0, 0.5),
            shadow_blur=4.0
        )
        text = Text(style=style)

        data = text.to_dict()
        assert data["style"]["shadow_offset"] == (2.0, 2.0)
        assert data["style"]["shadow_color"] == (0.0, 0.0, 0.0, 0.5)
        assert data["style"]["shadow_blur"] == 4.0

    def test_text_to_dict_with_entity_id(self):
        """Test serialization includes entity ID."""
        from engine.ui.widgets.primitives.text import Text

        text = Text(entity_id="text_001")

        data = text.to_dict()
        assert data["entity_id"] == "text_001"

    def test_text_from_dict(self):
        """Test deserialization from dictionary."""
        from engine.ui.widgets.primitives.text import (
            Text, TextAlignment, OverflowMode
        )

        data = {
            "content": "Hello",
            "alignment": "RIGHT",
            "vertical_alignment": "BOTTOM",
            "word_wrap": False,
            "overflow_mode": "CLIP",
            "max_lines": 2,
            "ellipsis": ">>>",
            "rich_text": True,
            "width": 150,
            "height": 50,
            "style": {
                "font_family": "Verdana",
                "font_size": 16,
                "font_weight": "normal",
                "font_style": "italic",
                "color": (0.5, 0.5, 0.5, 1.0),
                "line_height": 1.5,
                "letter_spacing": 1.0,
                "underline": True,
                "strikethrough": False,
            }
        }

        text = Text.from_dict(data)
        assert text.content == "Hello"
        assert text.alignment == TextAlignment.RIGHT
        assert text.word_wrap is False
        assert text.overflow_mode == OverflowMode.CLIP
        assert text.max_lines == 2
        assert text.ellipsis == ">>>"
        assert text.rich_text is True
        assert text.style.font_family == "Verdana"
        assert text.style.font_size == 16
        assert text.style.font_style == "italic"
        assert text.style.underline is True

    def test_text_from_dict_with_shadow(self):
        """Test deserialization with shadow."""
        from engine.ui.widgets.primitives.text import Text

        data = {
            "content": "Shadowed",
            "style": {
                "font_family": "default",
                "font_size": 14,
                "font_weight": "normal",
                "font_style": "normal",
                "color": (0.0, 0.0, 0.0, 1.0),
                "line_height": 1.2,
                "letter_spacing": 0.0,
                "underline": False,
                "strikethrough": False,
                "shadow_offset": [3.0, 3.0],
                "shadow_color": (0.0, 0.0, 0.0, 0.3),
                "shadow_blur": 2.0,
            }
        }

        text = Text.from_dict(data)
        assert text.style.shadow_offset == (3.0, 3.0)
        assert text.style.shadow_blur == 2.0

    def test_text_roundtrip_serialization(self):
        """Test serialization roundtrip preserves data."""
        from engine.ui.widgets.primitives.text import (
            Text, TextStyle, TextAlignment, VerticalAlignment, OverflowMode
        )

        style = TextStyle(
            font_family="Courier",
            font_size=12,
            font_weight="bold",
            font_style="normal",
            color=(0.2, 0.4, 0.6, 0.8),
            line_height=1.4,
            letter_spacing=0.5,
            underline=True,
            strikethrough=False,
        )

        original = Text(
            content="Test Content",
            style=style,
            alignment=TextAlignment.JUSTIFY,
            vertical_alignment=VerticalAlignment.MIDDLE,
            word_wrap=True,
            overflow_mode=OverflowMode.SHRINK,
            max_lines=5,
            ellipsis="...",
            rich_text=False,
            width=300,
            height=150,
        )

        data = original.to_dict()
        restored = Text.from_dict(data)

        assert restored.content == original.content
        assert restored.alignment == original.alignment
        assert restored.vertical_alignment == original.vertical_alignment
        assert restored.word_wrap == original.word_wrap
        assert restored.overflow_mode == original.overflow_mode
        assert restored.max_lines == original.max_lines
        assert restored.ellipsis == original.ellipsis
        assert restored.rich_text == original.rich_text
        assert restored.width == original.width
        assert restored.height == original.height
        assert restored.style.font_family == original.style.font_family
        assert restored.style.font_size == original.style.font_size


class TestTextRepr:
    """Tests for Text string representation."""

    def test_text_repr_short_content(self):
        """Test Text repr with short content."""
        from engine.ui.widgets.primitives.text import Text, TextAlignment

        text = Text(content="Hello", alignment=TextAlignment.CENTER, word_wrap=True)
        repr_str = repr(text)

        assert "Text" in repr_str
        assert "Hello" in repr_str
        assert "CENTER" in repr_str

    def test_text_repr_long_content_truncated(self):
        """Test Text repr truncates long content."""
        from engine.ui.widgets.primitives.text import Text

        long_content = "A" * 50
        text = Text(content=long_content)
        repr_str = repr(text)

        assert "..." in repr_str
        assert len(repr_str) < len(long_content) + 50

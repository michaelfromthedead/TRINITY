"""
Text Widget - Text rendering with alignment, wrapping, and rich text support.

Provides comprehensive text display with font configuration, alignment options,
overflow handling, and basic rich text markup parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple


class TextAlignment(Enum):
    """Text horizontal alignment options."""
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()
    JUSTIFY = auto()


class VerticalAlignment(Enum):
    """Text vertical alignment options."""
    TOP = auto()
    MIDDLE = auto()
    BOTTOM = auto()


class OverflowMode(Enum):
    """Text overflow handling modes."""
    CLIP = auto()       # Clip text at boundaries
    ELLIPSIS = auto()   # Add ... when text overflows
    SCROLL = auto()     # Enable scrolling
    WRAP = auto()       # Wrap to next line
    SHRINK = auto()     # Reduce font size to fit


def _validate_color(value: Any) -> Tuple[float, float, float, float]:
    """Validate and normalize color value."""
    if isinstance(value, str):
        return _parse_hex_color(value)
    elif isinstance(value, (tuple, list)):
        if len(value) == 3:
            r, g, b = value
            a = 1.0
        elif len(value) == 4:
            r, g, b, a = value
        else:
            raise ValueError(f"Color must have 3 or 4 components, got {len(value)}")

        for component, name in [(r, "red"), (g, "green"), (b, "blue"), (a, "alpha")]:
            if not isinstance(component, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not (0.0 <= component <= 1.0):
                raise ValueError(f"{name} must be in range [0, 1], got {component}")

        return (float(r), float(g), float(b), float(a))
    else:
        raise ValueError(f"Invalid color type: {type(value)}")


def _parse_hex_color(hex_str: str) -> Tuple[float, float, float, float]:
    """Parse hex color string to RGBA tuple."""
    if not hex_str.startswith("#"):
        raise ValueError("Hex color must start with #")

    hex_str = hex_str[1:]
    length = len(hex_str)

    if length == 3:
        r = int(hex_str[0] * 2, 16) / 255.0
        g = int(hex_str[1] * 2, 16) / 255.0
        b = int(hex_str[2] * 2, 16) / 255.0
        a = 1.0
    elif length == 4:
        r = int(hex_str[0] * 2, 16) / 255.0
        g = int(hex_str[1] * 2, 16) / 255.0
        b = int(hex_str[2] * 2, 16) / 255.0
        a = int(hex_str[3] * 2, 16) / 255.0
    elif length == 6:
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        a = 1.0
    elif length == 8:
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        a = int(hex_str[6:8], 16) / 255.0
    else:
        raise ValueError(f"Invalid hex color length: {length}")

    return (r, g, b, a)


@dataclass
class TextStyle:
    """Text styling configuration.

    Attributes:
        font_family: Font family name
        font_size: Font size in points
        font_weight: Font weight (normal, bold, etc.)
        font_style: Font style (normal, italic)
        color: Text color (RGBA)
        line_height: Line height multiplier
        letter_spacing: Letter spacing in pixels
        underline: Whether text is underlined
        strikethrough: Whether text has strikethrough
        shadow_offset: Shadow offset (x, y) or None
        shadow_color: Shadow color (RGBA)
        shadow_blur: Shadow blur radius
    """
    font_family: str = "default"
    font_size: float = 14.0
    font_weight: str = "normal"  # normal, bold, light, 100-900
    font_style: str = "normal"  # normal, italic, oblique
    color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    line_height: float = 1.2
    letter_spacing: float = 0.0
    underline: bool = False
    strikethrough: bool = False
    shadow_offset: Optional[Tuple[float, float]] = None
    shadow_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.5)
    shadow_blur: float = 0.0

    def __post_init__(self) -> None:
        """Validate text style."""
        if self.font_size <= 0:
            raise ValueError(f"font_size must be > 0, got {self.font_size}")
        if self.line_height <= 0:
            raise ValueError(f"line_height must be > 0, got {self.line_height}")

        # Validate font weight
        valid_weights = {"normal", "bold", "light", "thin", "medium", "semibold", "black"}
        if self.font_weight not in valid_weights:
            try:
                weight_num = int(self.font_weight)
                if not (100 <= weight_num <= 900):
                    raise ValueError
            except ValueError:
                raise ValueError(
                    f"font_weight must be one of {valid_weights} or 100-900, "
                    f"got {self.font_weight}"
                )

        # Validate font style
        valid_styles = {"normal", "italic", "oblique"}
        if self.font_style not in valid_styles:
            raise ValueError(
                f"font_style must be one of {valid_styles}, got {self.font_style}"
            )

    def with_overrides(self, **kwargs: Any) -> TextStyle:
        """Create a new TextStyle with overridden values.

        Args:
            **kwargs: Style properties to override

        Returns:
            New TextStyle with overrides applied
        """
        return TextStyle(
            font_family=kwargs.get("font_family", self.font_family),
            font_size=kwargs.get("font_size", self.font_size),
            font_weight=kwargs.get("font_weight", self.font_weight),
            font_style=kwargs.get("font_style", self.font_style),
            color=kwargs.get("color", self.color),
            line_height=kwargs.get("line_height", self.line_height),
            letter_spacing=kwargs.get("letter_spacing", self.letter_spacing),
            underline=kwargs.get("underline", self.underline),
            strikethrough=kwargs.get("strikethrough", self.strikethrough),
            shadow_offset=kwargs.get("shadow_offset", self.shadow_offset),
            shadow_color=kwargs.get("shadow_color", self.shadow_color),
            shadow_blur=kwargs.get("shadow_blur", self.shadow_blur),
        )


@dataclass
class RichTextSpan:
    """A span of styled text within rich text.

    Attributes:
        text: The text content
        style: Style overrides for this span
        start_index: Starting character index in original text
        end_index: Ending character index in original text
    """
    text: str
    style: Optional[TextStyle] = None
    start_index: int = 0
    end_index: int = 0

    @property
    def length(self) -> int:
        """Get span length in characters."""
        return len(self.text)


class RichTextParser:
    """Parser for basic rich text markup.

    Supported tags:
    - <b>bold</b>
    - <i>italic</i>
    - <u>underline</u>
    - <s>strikethrough</s>
    - <color=#RRGGBB>colored text</color>
    - <size=N>sized text</size>
    - <font=name>font text</font>
    - <br/> or <br> for line breaks

    Example:
        parser = RichTextParser()
        spans = parser.parse("Hello <b>world</b>!")
    """

    # Pattern to match tags
    _TAG_PATTERN = re.compile(
        r"<(/?)(\w+)(?:=([^>]*))?(/?)>",
        re.IGNORECASE,
    )

    def __init__(self, base_style: Optional[TextStyle] = None) -> None:
        """Initialize parser with base style.

        Args:
            base_style: Base text style for unstyled text
        """
        self.base_style = base_style or TextStyle()

    def parse(self, text: str) -> List[RichTextSpan]:
        """Parse rich text into styled spans.

        Args:
            text: Rich text string with markup

        Returns:
            List of RichTextSpan objects
        """
        spans: List[RichTextSpan] = []
        style_stack: List[TextStyle] = [self.base_style]
        current_text = ""
        current_start = 0
        pos = 0

        for match in self._TAG_PATTERN.finditer(text):
            # Add text before this tag
            text_before = text[pos:match.start()]
            if text_before:
                current_text += text_before

            tag_is_close = match.group(1) == "/"
            tag_name = match.group(2).lower()
            tag_value = match.group(3)
            tag_is_self_closing = match.group(4) == "/"

            # Handle self-closing tags
            if tag_is_self_closing or tag_name == "br":
                if current_text:
                    spans.append(RichTextSpan(
                        text=current_text,
                        style=style_stack[-1],
                        start_index=current_start,
                        end_index=current_start + len(current_text),
                    ))
                    current_start += len(current_text)
                    current_text = ""

                if tag_name == "br":
                    spans.append(RichTextSpan(
                        text="\n",
                        style=style_stack[-1],
                        start_index=current_start,
                        end_index=current_start + 1,
                    ))
                    current_start += 1

            # Handle opening tags
            elif not tag_is_close:
                if current_text:
                    spans.append(RichTextSpan(
                        text=current_text,
                        style=style_stack[-1],
                        start_index=current_start,
                        end_index=current_start + len(current_text),
                    ))
                    current_start += len(current_text)
                    current_text = ""

                # Create new style based on tag
                new_style = self._apply_tag(style_stack[-1], tag_name, tag_value)
                style_stack.append(new_style)

            # Handle closing tags
            else:
                if current_text:
                    spans.append(RichTextSpan(
                        text=current_text,
                        style=style_stack[-1],
                        start_index=current_start,
                        end_index=current_start + len(current_text),
                    ))
                    current_start += len(current_text)
                    current_text = ""

                if len(style_stack) > 1:
                    style_stack.pop()

            pos = match.end()

        # Add remaining text
        remaining = text[pos:]
        if remaining:
            current_text += remaining

        if current_text:
            spans.append(RichTextSpan(
                text=current_text,
                style=style_stack[-1],
                start_index=current_start,
                end_index=current_start + len(current_text),
            ))

        return spans

    def _apply_tag(
        self,
        current_style: TextStyle,
        tag_name: str,
        tag_value: Optional[str],
    ) -> TextStyle:
        """Apply a tag to create a new style.

        Args:
            current_style: Current text style
            tag_name: Tag name (lowercase)
            tag_value: Tag value if present

        Returns:
            New TextStyle with tag applied
        """
        if tag_name == "b":
            return current_style.with_overrides(font_weight="bold")
        elif tag_name == "i":
            return current_style.with_overrides(font_style="italic")
        elif tag_name == "u":
            return current_style.with_overrides(underline=True)
        elif tag_name == "s":
            return current_style.with_overrides(strikethrough=True)
        elif tag_name == "color" and tag_value:
            try:
                color = _validate_color(tag_value)
                return current_style.with_overrides(color=color)
            except ValueError:
                return current_style
        elif tag_name == "size" and tag_value:
            try:
                size = float(tag_value)
                if size > 0:
                    return current_style.with_overrides(font_size=size)
            except ValueError:
                pass
            return current_style
        elif tag_name == "font" and tag_value:
            return current_style.with_overrides(font_family=tag_value)
        else:
            return current_style

    def strip_tags(self, text: str) -> str:
        """Remove all rich text tags, leaving plain text.

        Args:
            text: Rich text with markup

        Returns:
            Plain text without markup
        """
        result = self._TAG_PATTERN.sub("", text)
        result = result.replace("<br>", "\n").replace("<br/>", "\n")
        return result


class Text:
    """
    Text widget for displaying styled text.

    Supports font configuration, alignment, word wrapping, overflow handling,
    and rich text markup.

    Attributes:
        content: Text content to display
        style: Text style configuration
        alignment: Horizontal text alignment
        vertical_alignment: Vertical text alignment
        word_wrap: Whether to wrap text at word boundaries
        overflow_mode: How to handle text overflow
        max_lines: Maximum number of lines (0 = unlimited)
        ellipsis: String to use for ellipsis (default "...")
        rich_text: Whether to parse rich text markup
    """

    __slots__ = (
        "_content",
        "_style",
        "_alignment",
        "_vertical_alignment",
        "_word_wrap",
        "_overflow_mode",
        "_max_lines",
        "_ellipsis",
        "_rich_text",
        "_width",
        "_height",
        "_min_font_size",
        "_parsed_spans",
        "_line_breaks",
        "_measured_width",
        "_measured_height",
        "_dirty_layout",
        "_dirty_content",
        "_scroll_offset",
        "_entity_id",
    )

    def __init__(
        self,
        content: str = "",
        style: Optional[TextStyle] = None,
        alignment: TextAlignment = TextAlignment.LEFT,
        vertical_alignment: VerticalAlignment = VerticalAlignment.TOP,
        word_wrap: bool = True,
        overflow_mode: OverflowMode = OverflowMode.WRAP,
        max_lines: int = 0,
        ellipsis: str = "...",
        rich_text: bool = False,
        width: float = 0.0,
        height: float = 0.0,
        entity_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the Text widget.

        Args:
            content: Text content to display
            style: Text style configuration
            alignment: Horizontal text alignment
            vertical_alignment: Vertical text alignment
            word_wrap: Whether to wrap text at word boundaries
            overflow_mode: How to handle text overflow
            max_lines: Maximum number of lines (0 = unlimited)
            ellipsis: String to use for ellipsis
            rich_text: Whether to parse rich text markup
            width: Widget width (0 = auto)
            height: Widget height (0 = auto)
            entity_id: Optional entity ID for tracking
        """
        self._content = content
        self._style = style or TextStyle()
        self._alignment = alignment
        self._vertical_alignment = vertical_alignment
        self._word_wrap = word_wrap
        self._overflow_mode = overflow_mode
        self._max_lines = max(0, max_lines)
        self._ellipsis = ellipsis
        self._rich_text = rich_text
        self._width = max(0.0, width)
        self._height = max(0.0, height)
        self._min_font_size = 6.0
        self._parsed_spans: List[RichTextSpan] = []
        self._line_breaks: List[int] = []
        self._measured_width: float = 0.0
        self._measured_height: float = 0.0
        self._dirty_layout = True
        self._dirty_content = False
        self._scroll_offset: Tuple[float, float] = (0.0, 0.0)
        self._entity_id = entity_id

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def content(self) -> str:
        """Get text content."""
        return self._content

    @content.setter
    def content(self, value: str) -> None:
        """Set text content."""
        if not isinstance(value, str):
            value = str(value)
        if self._content != value:
            self._content = value
            self._dirty_layout = True
            self._dirty_content = True

    @property
    def style(self) -> TextStyle:
        """Get text style."""
        return self._style

    @style.setter
    def style(self, value: TextStyle) -> None:
        """Set text style."""
        if not isinstance(value, TextStyle):
            raise ValueError(f"style must be TextStyle, got {type(value)}")
        if self._style != value:
            self._style = value
            self._dirty_layout = True

    @property
    def alignment(self) -> TextAlignment:
        """Get horizontal alignment."""
        return self._alignment

    @alignment.setter
    def alignment(self, value: TextAlignment) -> None:
        """Set horizontal alignment."""
        if not isinstance(value, TextAlignment):
            raise ValueError(f"alignment must be TextAlignment, got {type(value)}")
        if self._alignment != value:
            self._alignment = value
            self._dirty_layout = True

    @property
    def vertical_alignment(self) -> VerticalAlignment:
        """Get vertical alignment."""
        return self._vertical_alignment

    @vertical_alignment.setter
    def vertical_alignment(self, value: VerticalAlignment) -> None:
        """Set vertical alignment."""
        if not isinstance(value, VerticalAlignment):
            raise ValueError(f"vertical_alignment must be VerticalAlignment, got {type(value)}")
        if self._vertical_alignment != value:
            self._vertical_alignment = value
            self._dirty_layout = True

    @property
    def word_wrap(self) -> bool:
        """Get word wrap setting."""
        return self._word_wrap

    @word_wrap.setter
    def word_wrap(self, value: bool) -> None:
        """Set word wrap."""
        if self._word_wrap != value:
            self._word_wrap = bool(value)
            self._dirty_layout = True

    @property
    def overflow_mode(self) -> OverflowMode:
        """Get overflow mode."""
        return self._overflow_mode

    @overflow_mode.setter
    def overflow_mode(self, value: OverflowMode) -> None:
        """Set overflow mode."""
        if not isinstance(value, OverflowMode):
            raise ValueError(f"overflow_mode must be OverflowMode, got {type(value)}")
        if self._overflow_mode != value:
            self._overflow_mode = value
            self._dirty_layout = True

    @property
    def max_lines(self) -> int:
        """Get maximum lines."""
        return self._max_lines

    @max_lines.setter
    def max_lines(self, value: int) -> None:
        """Set maximum lines."""
        value = max(0, int(value))
        if self._max_lines != value:
            self._max_lines = value
            self._dirty_layout = True

    @property
    def ellipsis(self) -> str:
        """Get ellipsis string."""
        return self._ellipsis

    @ellipsis.setter
    def ellipsis(self, value: str) -> None:
        """Set ellipsis string."""
        if self._ellipsis != value:
            self._ellipsis = str(value)
            self._dirty_layout = True

    @property
    def rich_text(self) -> bool:
        """Get rich text parsing setting."""
        return self._rich_text

    @rich_text.setter
    def rich_text(self, value: bool) -> None:
        """Set rich text parsing."""
        if self._rich_text != value:
            self._rich_text = bool(value)
            self._dirty_layout = True

    @property
    def width(self) -> float:
        """Get widget width."""
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        """Set widget width."""
        value = max(0.0, float(value))
        if self._width != value:
            self._width = value
            self._dirty_layout = True

    @property
    def height(self) -> float:
        """Get widget height."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        """Set widget height."""
        value = max(0.0, float(value))
        if self._height != value:
            self._height = value
            self._dirty_layout = True

    @property
    def min_font_size(self) -> float:
        """Get minimum font size for shrink mode."""
        return self._min_font_size

    @min_font_size.setter
    def min_font_size(self, value: float) -> None:
        """Set minimum font size for shrink mode."""
        if value <= 0:
            raise ValueError(f"min_font_size must be > 0, got {value}")
        self._min_font_size = value
        self._dirty_layout = True

    @property
    def measured_width(self) -> float:
        """Get measured content width."""
        return self._measured_width

    @property
    def measured_height(self) -> float:
        """Get measured content height."""
        return self._measured_height

    @property
    def scroll_offset(self) -> Tuple[float, float]:
        """Get scroll offset (x, y)."""
        return self._scroll_offset

    @scroll_offset.setter
    def scroll_offset(self, value: Tuple[float, float]) -> None:
        """Set scroll offset."""
        self._scroll_offset = (float(value[0]), float(value[1]))

    @property
    def is_dirty(self) -> bool:
        """Check if layout needs recalculating."""
        return self._dirty_layout or self._dirty_content

    @property
    def plain_text(self) -> str:
        """Get plain text content without markup."""
        if self._rich_text:
            parser = RichTextParser(self._style)
            return parser.strip_tags(self._content)
        return self._content

    @property
    def line_count(self) -> int:
        """Get number of lines in the text."""
        return len(self._line_breaks) + 1 if self._line_breaks else 1

    @property
    def entity_id(self) -> Optional[str]:
        """Get entity ID."""
        return self._entity_id

    @entity_id.setter
    def entity_id(self, value: Optional[str]) -> None:
        """Set entity ID."""
        self._entity_id = value

    # =========================================================================
    # METHODS
    # =========================================================================

    def parse_rich_text(self) -> List[RichTextSpan]:
        """Parse content as rich text.

        Returns:
            List of RichTextSpan objects
        """
        if not self._rich_text:
            return [RichTextSpan(
                text=self._content,
                style=self._style,
                start_index=0,
                end_index=len(self._content),
            )]

        parser = RichTextParser(self._style)
        self._parsed_spans = parser.parse(self._content)
        return self._parsed_spans

    def get_character_at_position(
        self,
        x: float,
        y: float,
        measure_func: Optional[Callable[[str, TextStyle], Tuple[float, float]]] = None,
    ) -> int:
        """Get character index at the given position.

        Args:
            x: X coordinate relative to widget
            y: Y coordinate relative to widget
            measure_func: Function to measure text (text, style) -> (width, height)

        Returns:
            Character index at position, or -1 if outside text
        """
        # This is a simplified implementation - real implementation would
        # use the measure function and line layout data
        if not self._content:
            return -1

        # Placeholder: return 0 for first half, length for second half
        if x < self._width / 2:
            return 0
        return len(self._content)

    def get_line_at_index(self, char_index: int) -> int:
        """Get line number containing the given character index.

        Args:
            char_index: Character index

        Returns:
            Line number (0-based)
        """
        if not self._line_breaks:
            return 0

        for i, break_index in enumerate(self._line_breaks):
            if char_index < break_index:
                return i
        return len(self._line_breaks)

    def measure(
        self,
        measure_func: Callable[[str, TextStyle], Tuple[float, float]],
    ) -> Tuple[float, float]:
        """Measure text content size.

        Args:
            measure_func: Function to measure text (text, style) -> (width, height)

        Returns:
            Tuple of (width, height)
        """
        if not self._content:
            self._measured_width = 0.0
            self._measured_height = 0.0
            return (0.0, 0.0)

        # Parse rich text if enabled
        if self._rich_text:
            spans = self.parse_rich_text()
        else:
            spans = [RichTextSpan(text=self._content, style=self._style)]

        # Measure all spans
        total_width = 0.0
        max_height = 0.0

        for span in spans:
            style = span.style or self._style
            width, height = measure_func(span.text, style)
            total_width += width
            max_height = max(max_height, height)

        self._measured_width = total_width
        self._measured_height = max_height

        return (total_width, max_height)

    def truncate_to_width(
        self,
        max_width: float,
        measure_func: Callable[[str, TextStyle], Tuple[float, float]],
    ) -> str:
        """Truncate text to fit within width, adding ellipsis if needed.

        Args:
            max_width: Maximum width in pixels
            measure_func: Function to measure text

        Returns:
            Truncated text with ellipsis if needed
        """
        if not self._content or max_width <= 0:
            return ""

        text = self.plain_text
        full_width, _ = measure_func(text, self._style)

        if full_width <= max_width:
            return text

        ellipsis_width, _ = measure_func(self._ellipsis, self._style)
        available_width = max_width - ellipsis_width

        if available_width <= 0:
            return self._ellipsis

        # Binary search for fitting length
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            width, _ = measure_func(text[:mid], self._style)
            if width <= available_width:
                low = mid
            else:
                high = mid - 1

        return text[:low] + self._ellipsis

    def mark_dirty(self) -> None:
        """Mark layout as dirty."""
        self._dirty_layout = True

    def mark_clean(self) -> None:
        """Mark layout as clean."""
        self._dirty_layout = False
        self._dirty_content = False

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize text widget to dictionary."""
        result: Dict[str, Any] = {
            "content": self._content,
            "style": {
                "font_family": self._style.font_family,
                "font_size": self._style.font_size,
                "font_weight": self._style.font_weight,
                "font_style": self._style.font_style,
                "color": self._style.color,
                "line_height": self._style.line_height,
                "letter_spacing": self._style.letter_spacing,
                "underline": self._style.underline,
                "strikethrough": self._style.strikethrough,
            },
            "alignment": self._alignment.name,
            "vertical_alignment": self._vertical_alignment.name,
            "word_wrap": self._word_wrap,
            "overflow_mode": self._overflow_mode.name,
            "max_lines": self._max_lines,
            "ellipsis": self._ellipsis,
            "rich_text": self._rich_text,
            "width": self._width,
            "height": self._height,
        }

        if self._style.shadow_offset is not None:
            result["style"]["shadow_offset"] = self._style.shadow_offset
            result["style"]["shadow_color"] = self._style.shadow_color
            result["style"]["shadow_blur"] = self._style.shadow_blur

        if self._entity_id is not None:
            result["entity_id"] = self._entity_id

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Text:
        """Deserialize text widget from dictionary."""
        style_data = data.get("style", {})

        shadow_offset = style_data.get("shadow_offset")
        if shadow_offset is not None:
            shadow_offset = tuple(shadow_offset)

        style = TextStyle(
            font_family=style_data.get("font_family", "default"),
            font_size=style_data.get("font_size", 14.0),
            font_weight=style_data.get("font_weight", "normal"),
            font_style=style_data.get("font_style", "normal"),
            color=tuple(style_data.get("color", (0.0, 0.0, 0.0, 1.0))),
            line_height=style_data.get("line_height", 1.2),
            letter_spacing=style_data.get("letter_spacing", 0.0),
            underline=style_data.get("underline", False),
            strikethrough=style_data.get("strikethrough", False),
            shadow_offset=shadow_offset,
            shadow_color=tuple(style_data.get("shadow_color", (0.0, 0.0, 0.0, 0.5))),
            shadow_blur=style_data.get("shadow_blur", 0.0),
        )

        return cls(
            content=data.get("content", ""),
            style=style,
            alignment=TextAlignment[data.get("alignment", "LEFT")],
            vertical_alignment=VerticalAlignment[data.get("vertical_alignment", "TOP")],
            word_wrap=data.get("word_wrap", True),
            overflow_mode=OverflowMode[data.get("overflow_mode", "WRAP")],
            max_lines=data.get("max_lines", 0),
            ellipsis=data.get("ellipsis", "..."),
            rich_text=data.get("rich_text", False),
            width=data.get("width", 0.0),
            height=data.get("height", 0.0),
            entity_id=data.get("entity_id"),
        )

    def __repr__(self) -> str:
        preview = self._content[:20] + "..." if len(self._content) > 20 else self._content
        return (
            f"Text(content={preview!r}, alignment={self._alignment.name}, "
            f"wrap={self._word_wrap})"
        )


__all__ = [
    "Text",
    "TextAlignment",
    "VerticalAlignment",
    "OverflowMode",
    "TextStyle",
    "RichTextParser",
    "RichTextSpan",
]

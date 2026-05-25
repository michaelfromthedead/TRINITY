"""
Rich text processing with markup parsing, inline images, and clickable links.

Provides:
- Markup parsing: [b]bold[/b], [i]italic[/i], [color=red]text[/color], [size=20]text[/size]
- Inline images/icons
- Clickable links
- Text runs with different styles
- RichTextParser class
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Iterator
import re


class ParseError(Exception):
    """Exception raised when rich text parsing fails."""

    def __init__(self, message: str, position: int = 0, line: int = 1) -> None:
        """Initialize the parse error.

        Args:
            message: Error message
            position: Character position where error occurred
            line: Line number where error occurred
        """
        super().__init__(f"{message} at position {position}, line {line}")
        self.position = position
        self.line = line


class MarkupTagType(Enum):
    """Types of markup tags."""
    BOLD = auto()
    ITALIC = auto()
    UNDERLINE = auto()
    STRIKETHROUGH = auto()
    COLOR = auto()
    SIZE = auto()
    FONT = auto()
    LINK = auto()
    IMAGE = auto()
    ICON = auto()
    CUSTOM = auto()


@dataclass
class MarkupTag:
    """Represents a parsed markup tag.

    Attributes:
        tag_type: Type of the tag
        name: Tag name string
        value: Optional attribute value
        is_closing: Whether this is a closing tag
        position: Position in source text
    """
    tag_type: MarkupTagType
    name: str
    value: str | None = None
    is_closing: bool = False
    position: int = 0

    @property
    def is_self_closing(self) -> bool:
        """Check if this is a self-closing tag (like image/icon)."""
        return self.tag_type in (MarkupTagType.IMAGE, MarkupTagType.ICON)


@dataclass
class RichTextStyle:
    """Style properties for a text run.

    Attributes:
        bold: Whether text is bold
        italic: Whether text is italic
        underline: Whether text is underlined
        strikethrough: Whether text has strikethrough
        color: Text color (hex or named)
        size: Font size (or None for default)
        font: Font family name (or None for default)
    """
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    color: str | None = None
    size: float | None = None
    font: str | None = None

    def copy(self) -> RichTextStyle:
        """Create a copy of this style."""
        return RichTextStyle(
            bold=self.bold,
            italic=self.italic,
            underline=self.underline,
            strikethrough=self.strikethrough,
            color=self.color,
            size=self.size,
            font=self.font,
        )

    def merge(self, other: RichTextStyle) -> RichTextStyle:
        """Merge another style into this one (other takes precedence)."""
        result = self.copy()
        if other.bold:
            result.bold = True
        if other.italic:
            result.italic = True
        if other.underline:
            result.underline = True
        if other.strikethrough:
            result.strikethrough = True
        if other.color is not None:
            result.color = other.color
        if other.size is not None:
            result.size = other.size
        if other.font is not None:
            result.font = other.font
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RichTextStyle):
            return False
        return (
            self.bold == other.bold
            and self.italic == other.italic
            and self.underline == other.underline
            and self.strikethrough == other.strikethrough
            and self.color == other.color
            and self.size == other.size
            and self.font == other.font
        )


@dataclass
class TextRun:
    """A run of text with consistent styling.

    Attributes:
        text: The text content
        style: Style applied to this run
        start_index: Start position in original text
        end_index: End position in original text
    """
    text: str
    style: RichTextStyle = field(default_factory=RichTextStyle)
    start_index: int = 0
    end_index: int = 0

    @property
    def length(self) -> int:
        """Get the length of the text."""
        return len(self.text)

    def split(self, index: int) -> tuple[TextRun, TextRun]:
        """Split this run at the given index.

        Args:
            index: Position to split at

        Returns:
            Tuple of (before, after) TextRuns
        """
        if index <= 0:
            return TextRun("", self.style.copy(), self.start_index, self.start_index), self
        if index >= len(self.text):
            return self, TextRun("", self.style.copy(), self.end_index, self.end_index)

        return (
            TextRun(
                self.text[:index],
                self.style.copy(),
                self.start_index,
                self.start_index + index,
            ),
            TextRun(
                self.text[index:],
                self.style.copy(),
                self.start_index + index,
                self.end_index,
            ),
        )


@dataclass
class InlineImage:
    """An inline image embedded in rich text.

    Attributes:
        source: Image source (path, URL, or asset ID)
        width: Image width (or None for intrinsic)
        height: Image height (or None for intrinsic)
        alt_text: Alternative text for accessibility
        position: Position in the text stream
    """
    source: str
    width: float | None = None
    height: float | None = None
    alt_text: str = ""
    position: int = 0

    def __post_init__(self) -> None:
        """Validate image parameters."""
        if not self.source:
            raise ValueError("Image source cannot be empty")
        if self.width is not None and self.width <= 0:
            raise ValueError(f"Image width must be positive, got {self.width}")
        if self.height is not None and self.height <= 0:
            raise ValueError(f"Image height must be positive, got {self.height}")


@dataclass
class ClickableLink:
    """A clickable link in rich text.

    Attributes:
        url: Link URL or action identifier
        text: Display text for the link
        style: Style applied to the link
        start_index: Start position in rich text
        end_index: End position in rich text
        hover_style: Style when hovered
        on_click: Callback when clicked
    """
    url: str
    text: str
    style: RichTextStyle = field(default_factory=RichTextStyle)
    start_index: int = 0
    end_index: int = 0
    hover_style: RichTextStyle | None = None
    on_click: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        """Set default link styling."""
        if self.style.color is None:
            self.style.color = "#0066CC"  # Default link blue
        if not self.style.underline:
            self.style.underline = True


@dataclass
class TextSpan:
    """A span of rich text content.

    Can contain text runs, inline images, or nested spans.
    """
    runs: list[TextRun] = field(default_factory=list)
    images: list[InlineImage] = field(default_factory=list)
    links: list[ClickableLink] = field(default_factory=list)

    def add_run(self, run: TextRun) -> None:
        """Add a text run to this span."""
        self.runs.append(run)

    def add_image(self, image: InlineImage) -> None:
        """Add an inline image to this span."""
        self.images.append(image)

    def add_link(self, link: ClickableLink) -> None:
        """Add a clickable link to this span."""
        self.links.append(link)

    @property
    def plain_text(self) -> str:
        """Get plain text content without formatting."""
        return "".join(run.text for run in self.runs)

    @property
    def total_length(self) -> int:
        """Get total length of all text content."""
        return sum(run.length for run in self.runs)

    def get_run_at(self, index: int) -> TextRun | None:
        """Get the text run containing the given index.

        Args:
            index: Character index

        Returns:
            TextRun containing the index, or None if not found
        """
        current_pos = 0
        for run in self.runs:
            if current_pos <= index < current_pos + run.length:
                return run
            current_pos += run.length
        return None

    def get_style_at(self, index: int) -> RichTextStyle:
        """Get the style at the given index.

        Args:
            index: Character index

        Returns:
            RichTextStyle at the index (or default if not found)
        """
        run = self.get_run_at(index)
        return run.style if run else RichTextStyle()

    def __iter__(self) -> Iterator[TextRun]:
        """Iterate over text runs."""
        return iter(self.runs)

    def __len__(self) -> int:
        """Get number of text runs."""
        return len(self.runs)


class RichTextParser:
    """Parser for rich text markup.

    Supports the following markup:
    - [b]bold[/b]
    - [i]italic[/i]
    - [u]underline[/u]
    - [s]strikethrough[/s]
    - [color=red]text[/color] or [color=#FF0000]text[/color]
    - [size=20]text[/size]
    - [font=Arial]text[/font]
    - [link=http://example.com]text[/link]
    - [img=path/to/image.png] (self-closing)
    - [icon=icon_name] (self-closing)
    """

    # Tag name to type mapping
    TAG_TYPES: dict[str, MarkupTagType] = {
        "b": MarkupTagType.BOLD,
        "bold": MarkupTagType.BOLD,
        "i": MarkupTagType.ITALIC,
        "italic": MarkupTagType.ITALIC,
        "u": MarkupTagType.UNDERLINE,
        "underline": MarkupTagType.UNDERLINE,
        "s": MarkupTagType.STRIKETHROUGH,
        "strike": MarkupTagType.STRIKETHROUGH,
        "strikethrough": MarkupTagType.STRIKETHROUGH,
        "color": MarkupTagType.COLOR,
        "colour": MarkupTagType.COLOR,
        "size": MarkupTagType.SIZE,
        "font": MarkupTagType.FONT,
        "link": MarkupTagType.LINK,
        "url": MarkupTagType.LINK,
        "a": MarkupTagType.LINK,
        "img": MarkupTagType.IMAGE,
        "image": MarkupTagType.IMAGE,
        "icon": MarkupTagType.ICON,
    }

    # Named colors mapping
    NAMED_COLORS: dict[str, str] = {
        "red": "#FF0000",
        "green": "#00FF00",
        "blue": "#0000FF",
        "white": "#FFFFFF",
        "black": "#000000",
        "yellow": "#FFFF00",
        "cyan": "#00FFFF",
        "magenta": "#FF00FF",
        "orange": "#FFA500",
        "purple": "#800080",
        "pink": "#FFC0CB",
        "gray": "#808080",
        "grey": "#808080",
        "gold": "#FFD700",
        "silver": "#C0C0C0",
    }

    # Regex patterns
    _TAG_PATTERN = re.compile(
        r"\[(/?)(\w+)(?:=([^\]]+))?\]",
        re.IGNORECASE,
    )

    def __init__(self, strict: bool = False) -> None:
        """Initialize the parser.

        Args:
            strict: If True, raise errors on invalid markup.
                    If False, treat invalid markup as plain text.
        """
        self._strict = strict
        self._custom_tags: dict[str, Callable[[str | None], RichTextStyle]] = {}

    def register_custom_tag(
        self,
        name: str,
        handler: Callable[[str | None], RichTextStyle],
    ) -> None:
        """Register a custom tag handler.

        Args:
            name: Tag name (without brackets)
            handler: Function(value) -> RichTextStyle
        """
        self._custom_tags[name.lower()] = handler
        self.TAG_TYPES[name.lower()] = MarkupTagType.CUSTOM

    def parse(self, text: str) -> TextSpan:
        """Parse rich text markup into a TextSpan.

        Args:
            text: Text with markup

        Returns:
            TextSpan containing parsed content

        Raises:
            ParseError: If strict mode and invalid markup found
        """
        if not text:
            return TextSpan()

        span = TextSpan()
        style_stack: list[tuple[MarkupTag, RichTextStyle]] = []
        current_style = RichTextStyle()
        current_text = ""
        text_start = 0
        position = 0

        # Find all tags
        last_end = 0
        for match in self._TAG_PATTERN.finditer(text):
            # Add text before this tag
            if match.start() > last_end:
                plain_text = text[last_end:match.start()]
                if plain_text:
                    current_text += plain_text

            # Parse the tag
            is_closing = bool(match.group(1))
            tag_name = match.group(2).lower()
            tag_value = match.group(3)

            tag_type = self.TAG_TYPES.get(tag_name, MarkupTagType.CUSTOM)

            tag = MarkupTag(
                tag_type=tag_type,
                name=tag_name,
                value=tag_value,
                is_closing=is_closing,
                position=match.start(),
            )

            # Handle self-closing tags (image, icon)
            if tag.is_self_closing:
                # Flush current text
                if current_text:
                    span.add_run(TextRun(
                        text=current_text,
                        style=current_style.copy(),
                        start_index=text_start,
                        end_index=text_start + len(current_text),
                    ))
                    text_start += len(current_text)
                    current_text = ""

                # Add the image/icon
                if tag_type == MarkupTagType.IMAGE:
                    self._handle_image(span, tag)
                elif tag_type == MarkupTagType.ICON:
                    self._handle_icon(span, tag)

            elif is_closing:
                # Closing tag: flush text and pop style
                if current_text:
                    span.add_run(TextRun(
                        text=current_text,
                        style=current_style.copy(),
                        start_index=text_start,
                        end_index=text_start + len(current_text),
                    ))
                    text_start += len(current_text)
                    current_text = ""

                # Pop matching style
                if style_stack:
                    # Find matching opening tag
                    for i in range(len(style_stack) - 1, -1, -1):
                        if style_stack[i][0].tag_type == tag_type:
                            style_stack.pop(i)
                            break

                    # Rebuild current style
                    current_style = RichTextStyle()
                    for _, style in style_stack:
                        current_style = current_style.merge(style)
                elif self._strict:
                    raise ParseError(
                        f"Unmatched closing tag [{tag_name}]",
                        position=match.start(),
                    )

            else:
                # Opening tag: flush text and push style
                if current_text:
                    span.add_run(TextRun(
                        text=current_text,
                        style=current_style.copy(),
                        start_index=text_start,
                        end_index=text_start + len(current_text),
                    ))
                    text_start += len(current_text)
                    current_text = ""

                # Create new style from tag
                new_style = self._style_from_tag(tag)

                # Handle link tags specially
                if tag_type == MarkupTagType.LINK:
                    self._start_link(span, tag, text_start)

                style_stack.append((tag, new_style))
                current_style = current_style.merge(new_style)

            last_end = match.end()
            position = match.end()

        # Add remaining text
        if last_end < len(text):
            current_text += text[last_end:]

        if current_text:
            span.add_run(TextRun(
                text=current_text,
                style=current_style.copy(),
                start_index=text_start,
                end_index=text_start + len(current_text),
            ))

        # Check for unclosed tags in strict mode
        if self._strict and style_stack:
            unclosed = [t.name for t, _ in style_stack]
            raise ParseError(
                f"Unclosed tags: {', '.join(unclosed)}",
                position=len(text),
            )

        return span

    def parse_plain(self, text: str) -> str:
        """Parse rich text and return plain text only.

        Args:
            text: Text with markup

        Returns:
            Plain text without any markup
        """
        return self._TAG_PATTERN.sub("", text)

    def escape(self, text: str) -> str:
        """Escape special characters in text.

        Args:
            text: Plain text to escape

        Returns:
            Text with brackets escaped
        """
        return text.replace("[", "[[").replace("]", "]]")

    def unescape(self, text: str) -> str:
        """Unescape special characters in text.

        Args:
            text: Escaped text

        Returns:
            Text with escaped brackets restored
        """
        return text.replace("[[", "[").replace("]]", "]")

    def validate(self, text: str) -> list[ParseError]:
        """Validate markup without parsing.

        Args:
            text: Text with markup

        Returns:
            List of ParseError for any issues found
        """
        errors: list[ParseError] = []
        stack: list[tuple[str, int]] = []

        for match in self._TAG_PATTERN.finditer(text):
            is_closing = bool(match.group(1))
            tag_name = match.group(2).lower()
            tag_type = self.TAG_TYPES.get(tag_name)

            if tag_type is None:
                errors.append(ParseError(
                    f"Unknown tag [{tag_name}]",
                    position=match.start(),
                ))
                continue

            # Skip self-closing tags
            if tag_type in (MarkupTagType.IMAGE, MarkupTagType.ICON):
                continue

            if is_closing:
                if not stack:
                    errors.append(ParseError(
                        f"Unmatched closing tag [/{tag_name}]",
                        position=match.start(),
                    ))
                elif stack[-1][0] != tag_name:
                    errors.append(ParseError(
                        f"Mismatched tags: expected [/{stack[-1][0]}], got [/{tag_name}]",
                        position=match.start(),
                    ))
                else:
                    stack.pop()
            else:
                stack.append((tag_name, match.start()))

        # Report unclosed tags
        for tag_name, pos in stack:
            errors.append(ParseError(
                f"Unclosed tag [{tag_name}]",
                position=pos,
            ))

        return errors

    def _style_from_tag(self, tag: MarkupTag) -> RichTextStyle:
        """Create a RichTextStyle from a tag.

        Args:
            tag: Parsed markup tag

        Returns:
            RichTextStyle with appropriate properties set
        """
        style = RichTextStyle()

        if tag.tag_type == MarkupTagType.BOLD:
            style.bold = True
        elif tag.tag_type == MarkupTagType.ITALIC:
            style.italic = True
        elif tag.tag_type == MarkupTagType.UNDERLINE:
            style.underline = True
        elif tag.tag_type == MarkupTagType.STRIKETHROUGH:
            style.strikethrough = True
        elif tag.tag_type == MarkupTagType.COLOR:
            style.color = self._parse_color(tag.value)
        elif tag.tag_type == MarkupTagType.SIZE:
            style.size = self._parse_size(tag.value)
        elif tag.tag_type == MarkupTagType.FONT:
            style.font = tag.value
        elif tag.tag_type == MarkupTagType.LINK:
            style.underline = True
            style.color = "#0066CC"
        elif tag.tag_type == MarkupTagType.CUSTOM:
            if tag.name in self._custom_tags:
                return self._custom_tags[tag.name](tag.value)

        return style

    def _parse_color(self, value: str | None) -> str | None:
        """Parse a color value.

        Args:
            value: Color string (named or hex)

        Returns:
            Normalized hex color or None
        """
        if not value:
            return None

        value = value.strip().lower()

        # Check named colors
        if value in self.NAMED_COLORS:
            return self.NAMED_COLORS[value]

        # Check hex format
        if value.startswith("#"):
            # Validate hex
            hex_part = value[1:]
            if len(hex_part) in (3, 6, 8) and all(c in "0123456789abcdef" for c in hex_part):
                # Expand short hex
                if len(hex_part) == 3:
                    return "#" + "".join(c * 2 for c in hex_part)
                return value.upper()

        return None

    def _parse_size(self, value: str | None) -> float | None:
        """Parse a size value.

        Args:
            value: Size string

        Returns:
            Size as float or None
        """
        if not value:
            return None

        try:
            size = float(value.strip())
            return size if size > 0 else None
        except ValueError:
            return None

    def _handle_image(self, span: TextSpan, tag: MarkupTag) -> None:
        """Handle an image tag.

        Args:
            span: TextSpan to add image to
            tag: Image tag
        """
        if not tag.value:
            return

        # Parse value (may include size: path|width|height)
        parts = tag.value.split("|")
        source = parts[0].strip()

        width = None
        height = None
        if len(parts) >= 2:
            try:
                width = float(parts[1])
            except ValueError:
                pass
        if len(parts) >= 3:
            try:
                height = float(parts[2])
            except ValueError:
                pass

        try:
            image = InlineImage(
                source=source,
                width=width,
                height=height,
                position=tag.position,
            )
            span.add_image(image)
        except ValueError:
            pass  # Invalid image, skip

    def _handle_icon(self, span: TextSpan, tag: MarkupTag) -> None:
        """Handle an icon tag.

        Args:
            span: TextSpan to add icon to
            tag: Icon tag
        """
        if not tag.value:
            return

        # Icons are treated as small images
        try:
            image = InlineImage(
                source=f"icon:{tag.value}",
                width=None,  # Use default icon size
                height=None,
                position=tag.position,
                alt_text=tag.value,
            )
            span.add_image(image)
        except ValueError:
            pass

    def _start_link(self, span: TextSpan, tag: MarkupTag, position: int) -> None:
        """Start tracking a link.

        Args:
            span: TextSpan to add link to
            tag: Link tag
            position: Current text position
        """
        if not tag.value:
            return

        # Link will be finalized when closing tag is found
        link = ClickableLink(
            url=tag.value,
            text="",  # Will be filled in later
            start_index=position,
            end_index=position,  # Will be updated when closed
        )
        span.add_link(link)


class RichTextBuilder:
    """Builder for creating rich text programmatically.

    Example:
        builder = RichTextBuilder()
        builder.text("Hello, ")
        builder.bold().text("World").end()
        builder.text("!")
        span = builder.build()
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._runs: list[TextRun] = []
        self._images: list[InlineImage] = []
        self._links: list[ClickableLink] = []
        self._style_stack: list[RichTextStyle] = [RichTextStyle()]
        self._current_text = ""
        self._position = 0

    @property
    def _current_style(self) -> RichTextStyle:
        """Get the current style."""
        return self._style_stack[-1]

    def text(self, content: str) -> RichTextBuilder:
        """Add plain text with current style.

        Args:
            content: Text to add

        Returns:
            Self for chaining
        """
        if content:
            run = TextRun(
                text=content,
                style=self._current_style.copy(),
                start_index=self._position,
                end_index=self._position + len(content),
            )
            self._runs.append(run)
            self._position += len(content)
        return self

    def bold(self) -> RichTextBuilder:
        """Push bold style.

        Returns:
            Self for chaining
        """
        new_style = self._current_style.copy()
        new_style.bold = True
        self._style_stack.append(new_style)
        return self

    def italic(self) -> RichTextBuilder:
        """Push italic style.

        Returns:
            Self for chaining
        """
        new_style = self._current_style.copy()
        new_style.italic = True
        self._style_stack.append(new_style)
        return self

    def underline(self) -> RichTextBuilder:
        """Push underline style.

        Returns:
            Self for chaining
        """
        new_style = self._current_style.copy()
        new_style.underline = True
        self._style_stack.append(new_style)
        return self

    def strikethrough(self) -> RichTextBuilder:
        """Push strikethrough style.

        Returns:
            Self for chaining
        """
        new_style = self._current_style.copy()
        new_style.strikethrough = True
        self._style_stack.append(new_style)
        return self

    def color(self, value: str) -> RichTextBuilder:
        """Push color style.

        Args:
            value: Color value (hex or named)

        Returns:
            Self for chaining
        """
        new_style = self._current_style.copy()
        new_style.color = value
        self._style_stack.append(new_style)
        return self

    def size(self, value: float) -> RichTextBuilder:
        """Push size style.

        Args:
            value: Font size

        Returns:
            Self for chaining
        """
        new_style = self._current_style.copy()
        new_style.size = value
        self._style_stack.append(new_style)
        return self

    def font(self, name: str) -> RichTextBuilder:
        """Push font style.

        Args:
            name: Font family name

        Returns:
            Self for chaining
        """
        new_style = self._current_style.copy()
        new_style.font = name
        self._style_stack.append(new_style)
        return self

    def end(self) -> RichTextBuilder:
        """Pop the current style.

        Returns:
            Self for chaining
        """
        if len(self._style_stack) > 1:
            self._style_stack.pop()
        return self

    def image(
        self,
        source: str,
        width: float | None = None,
        height: float | None = None,
        alt_text: str = "",
    ) -> RichTextBuilder:
        """Add an inline image.

        Args:
            source: Image source
            width: Optional width
            height: Optional height
            alt_text: Alternative text

        Returns:
            Self for chaining
        """
        image = InlineImage(
            source=source,
            width=width,
            height=height,
            alt_text=alt_text,
            position=self._position,
        )
        self._images.append(image)
        return self

    def link(self, url: str, text: str) -> RichTextBuilder:
        """Add a clickable link.

        Args:
            url: Link URL
            text: Display text

        Returns:
            Self for chaining
        """
        link = ClickableLink(
            url=url,
            text=text,
            start_index=self._position,
            end_index=self._position + len(text),
        )
        self._links.append(link)

        # Also add as a styled text run
        link_style = self._current_style.copy()
        link_style.underline = True
        link_style.color = "#0066CC"

        run = TextRun(
            text=text,
            style=link_style,
            start_index=self._position,
            end_index=self._position + len(text),
        )
        self._runs.append(run)
        self._position += len(text)

        return self

    def newline(self) -> RichTextBuilder:
        """Add a newline.

        Returns:
            Self for chaining
        """
        return self.text("\n")

    def build(self) -> TextSpan:
        """Build the final TextSpan.

        Returns:
            TextSpan with all content
        """
        span = TextSpan(
            runs=list(self._runs),
            images=list(self._images),
            links=list(self._links),
        )
        return span

    def clear(self) -> RichTextBuilder:
        """Clear all content and reset.

        Returns:
            Self for chaining
        """
        self._runs.clear()
        self._images.clear()
        self._links.clear()
        self._style_stack = [RichTextStyle()]
        self._position = 0
        return self

"""
UI Text and Localization System.

This module provides comprehensive text rendering and localization support:
- Font management with TrueType, bitmap, and SDF fonts
- Text rendering with measurement, line breaking, and shaping
- Rich text processing with markup, inline images, and links
- Full localization with pluralization and RTL support
- IME (Input Method Editor) support for complex scripts

Example:
    from engine.ui.text import (
        FontManager,
        TextRenderer,
        RichTextParser,
        LocalizationManager,
        IMEHandler,
    )

    # Load a font
    font_manager = FontManager()
    font = font_manager.load_font("assets/fonts/roboto.ttf", size=16)

    # Render text
    renderer = TextRenderer(font_manager)
    size = renderer.measure_text("Hello World", font)

    # Parse rich text
    parser = RichTextParser()
    runs = parser.parse("[b]Bold[/b] and [color=#FF0000]red[/color]")

    # Localization
    loc = LocalizationManager.get_instance()
    loc.load_strings("en", "assets/strings/en.json")
    text = loc.get("greeting", name="Player")  # "Hello, Player!"
"""

from .font import (
    Font,
    FontFamily,
    FontManager,
    FontStyle,
    FontWeight,
    FontAtlas,
    SDFFont,
    GlyphMetrics,
    FontFallbackChain,
)

from .text_renderer import (
    TextRenderer,
    TextMeasurement,
    LineBreaker,
    LineBreakResult,
    TextShaper,
    ShapedGlyph,
    GlyphCache,
    TextAlignment,
    TextOverflow,
)

from .rich_text import (
    RichTextParser,
    TextRun,
    TextSpan,
    InlineImage,
    ClickableLink,
    RichTextStyle,
    MarkupTag,
    ParseError,
)

from .localization import (
    LocalizationManager,
    StringTable,
    Language,
    PluralRule,
    PluralCategory,
    TextDirection,
    LocalizedString,
    FormatParameter,
)

from .ime import (
    IMEHandler,
    IMEState,
    CompositionString,
    CandidateList,
    IMECandidate,
    IMEEvent,
    IMEEventType,
)

__all__ = [
    # Font
    "Font",
    "FontFamily",
    "FontManager",
    "FontStyle",
    "FontWeight",
    "FontAtlas",
    "SDFFont",
    "GlyphMetrics",
    "FontFallbackChain",
    # Text Renderer
    "TextRenderer",
    "TextMeasurement",
    "LineBreaker",
    "LineBreakResult",
    "TextShaper",
    "ShapedGlyph",
    "GlyphCache",
    "TextAlignment",
    "TextOverflow",
    # Rich Text
    "RichTextParser",
    "TextRun",
    "TextSpan",
    "InlineImage",
    "ClickableLink",
    "RichTextStyle",
    "MarkupTag",
    "ParseError",
    # Localization
    "LocalizationManager",
    "StringTable",
    "Language",
    "PluralRule",
    "PluralCategory",
    "TextDirection",
    "LocalizedString",
    "FormatParameter",
    # IME
    "IMEHandler",
    "IMEState",
    "CompositionString",
    "CandidateList",
    "IMECandidate",
    "IMEEvent",
    "IMEEventType",
]

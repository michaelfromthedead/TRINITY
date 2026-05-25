"""
Primitive UI widgets - foundational visual elements.

This module provides the basic building blocks for UI composition:
- Image: Texture/atlas display with scaling and tinting
- Text: Text rendering with alignment, wrapping, and rich text support
- Border: Rectangular frame with styling options
- Spacer: Layout spacing utility
"""

from engine.ui.widgets.primitives.image import (
    Image,
    ScaleMode,
    NineSliceConfig,
    UVCoordinates,
)
from engine.ui.widgets.primitives.text import (
    Text,
    TextAlignment,
    VerticalAlignment,
    OverflowMode,
    TextStyle,
    RichTextParser,
    RichTextSpan,
)
from engine.ui.widgets.primitives.border import (
    Border,
    BorderStyle,
    CornerRadius,
)
from engine.ui.widgets.primitives.spacer import (
    Spacer,
    SpacerMode,
)

__all__ = [
    # Image
    "Image",
    "ScaleMode",
    "NineSliceConfig",
    "UVCoordinates",
    # Text
    "Text",
    "TextAlignment",
    "VerticalAlignment",
    "OverflowMode",
    "TextStyle",
    "RichTextParser",
    "RichTextSpan",
    # Border
    "Border",
    "BorderStyle",
    "CornerRadius",
    # Spacer
    "Spacer",
    "SpacerMode",
]

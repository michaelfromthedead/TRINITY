"""
Font management system for UI text rendering.

Provides:
- Font class for describing font properties
- FontManager for loading and caching fonts
- Font atlases for bitmap font rendering
- SDF (Signed Distance Field) font support
- Font fallback chains for missing glyphs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterator
from weakref import WeakValueDictionary


class FontWeight(Enum):
    """Font weight values following CSS conventions."""
    THIN = 100
    EXTRA_LIGHT = 200
    LIGHT = 300
    NORMAL = 400
    MEDIUM = 500
    SEMI_BOLD = 600
    BOLD = 700
    EXTRA_BOLD = 800
    BLACK = 900


class FontStyle(Enum):
    """Font style variations."""
    NORMAL = auto()
    ITALIC = auto()
    OBLIQUE = auto()


@dataclass(frozen=True)
class GlyphMetrics:
    """Metrics for a single glyph."""
    codepoint: int
    width: float
    height: float
    bearing_x: float
    bearing_y: float
    advance: float

    # Texture coordinates (for atlas rendering)
    uv_x: float = 0.0
    uv_y: float = 0.0
    uv_width: float = 0.0
    uv_height: float = 0.0


@dataclass
class Font:
    """Represents a font with specific properties.

    Attributes:
        family: Font family name (e.g., "Roboto", "Arial")
        size: Font size in points
        weight: Font weight (thin to black)
        style: Font style (normal, italic, oblique)
        line_height: Line height multiplier (default 1.2)
        letter_spacing: Additional spacing between characters
    """
    family: str
    size: float
    weight: FontWeight = FontWeight.NORMAL
    style: FontStyle = FontStyle.NORMAL
    line_height: float = 1.2
    letter_spacing: float = 0.0

    # Internal metrics (set by FontManager when loaded)
    _ascender: float = field(default=0.0, repr=False)
    _descender: float = field(default=0.0, repr=False)
    _units_per_em: int = field(default=1000, repr=False)
    _glyphs: dict[int, GlyphMetrics] = field(default_factory=dict, repr=False)
    _handle: Any = field(default=None, repr=False)  # Platform font handle
    _manager_cache_key: Optional[str] = field(default=None, repr=False)  # Cache key used by FontManager

    def __post_init__(self) -> None:
        """Validate font parameters."""
        if self.size <= 0:
            raise ValueError(f"Font size must be positive, got {self.size}")
        if self.line_height <= 0:
            raise ValueError(f"Line height must be positive, got {self.line_height}")

    @property
    def ascender(self) -> float:
        """Get the font ascender (height above baseline)."""
        return self._ascender * (self.size / self._units_per_em) if self._units_per_em else 0

    @property
    def descender(self) -> float:
        """Get the font descender (depth below baseline)."""
        return self._descender * (self.size / self._units_per_em) if self._units_per_em else 0

    @property
    def line_spacing(self) -> float:
        """Get the total line spacing for this font."""
        return self.size * self.line_height

    def get_glyph(self, codepoint: int) -> GlyphMetrics | None:
        """Get metrics for a specific glyph.

        Args:
            codepoint: Unicode codepoint

        Returns:
            GlyphMetrics if found, None otherwise
        """
        return self._glyphs.get(codepoint)

    def has_glyph(self, codepoint: int) -> bool:
        """Check if font contains a glyph for the codepoint."""
        return codepoint in self._glyphs

    def cache_key(self) -> str:
        """Generate a unique cache key for this font configuration."""
        if self._manager_cache_key is not None:
            return self._manager_cache_key
        return f"{self.family}:{self.size}:{self.weight.value}:{self.style.name}"

    def with_size(self, size: float) -> Font:
        """Create a copy of this font with a different size."""
        return Font(
            family=self.family,
            size=size,
            weight=self.weight,
            style=self.style,
            line_height=self.line_height,
            letter_spacing=self.letter_spacing,
        )

    def with_weight(self, weight: FontWeight) -> Font:
        """Create a copy of this font with a different weight."""
        return Font(
            family=self.family,
            size=self.size,
            weight=weight,
            style=self.style,
            line_height=self.line_height,
            letter_spacing=self.letter_spacing,
        )

    def with_style(self, style: FontStyle) -> Font:
        """Create a copy of this font with a different style."""
        return Font(
            family=self.family,
            size=self.size,
            weight=self.weight,
            style=style,
            line_height=self.line_height,
            letter_spacing=self.letter_spacing,
        )


@dataclass
class FontFamily:
    """Collection of fonts in a family with different weights/styles.

    Allows easy access to different variations of the same font family.
    """
    name: str
    fonts: dict[tuple[FontWeight, FontStyle], Font] = field(default_factory=dict)

    def add_font(self, font: Font) -> None:
        """Add a font variant to this family."""
        if font.family != self.name:
            raise ValueError(f"Font family mismatch: {font.family} != {self.name}")
        self.fonts[(font.weight, font.style)] = font

    def get_font(
        self,
        weight: FontWeight = FontWeight.NORMAL,
        style: FontStyle = FontStyle.NORMAL,
        size: float = 16.0,
    ) -> Font | None:
        """Get a font variant from this family.

        If exact match not found, returns closest match.
        """
        key = (weight, style)
        if key in self.fonts:
            font = self.fonts[key]
            return font.with_size(size) if font.size != size else font

        # Find closest weight match with same style
        same_style = [(w, s) for w, s in self.fonts if s == style]
        if same_style:
            closest = min(same_style, key=lambda x: abs(x[0].value - weight.value))
            font = self.fonts[closest]
            return font.with_size(size) if font.size != size else font

        # Fallback to any available font
        if self.fonts:
            font = next(iter(self.fonts.values()))
            return font.with_size(size) if font.size != size else font

        return None

    def has_variant(self, weight: FontWeight, style: FontStyle) -> bool:
        """Check if a specific variant exists."""
        return (weight, style) in self.fonts


@dataclass
class FontAtlas:
    """Texture atlas for bitmap font rendering.

    Stores pre-rendered glyphs in a texture for efficient rendering.
    """
    texture_id: int
    width: int
    height: int
    glyphs: dict[int, GlyphMetrics] = field(default_factory=dict)
    _padding: int = 2
    _next_x: int = 0
    _next_y: int = 0
    _row_height: int = 0

    def add_glyph(self, codepoint: int, width: int, height: int) -> tuple[int, int] | None:
        """Reserve space for a glyph in the atlas.

        Args:
            codepoint: Unicode codepoint
            width: Glyph width in pixels
            height: Glyph height in pixels

        Returns:
            (x, y) position in atlas, or None if no space
        """
        padded_width = width + self._padding * 2
        padded_height = height + self._padding * 2

        # Check if fits in current row
        if self._next_x + padded_width > self.width:
            # Move to next row
            self._next_x = 0
            self._next_y += self._row_height + self._padding
            self._row_height = 0

        # Check if fits in atlas
        if self._next_y + padded_height > self.height:
            return None

        # Reserve space
        x = self._next_x + self._padding
        y = self._next_y + self._padding

        self._next_x += padded_width
        self._row_height = max(self._row_height, padded_height)

        return (x, y)

    def get_glyph(self, codepoint: int) -> GlyphMetrics | None:
        """Get glyph metrics from the atlas."""
        return self.glyphs.get(codepoint)

    @property
    def utilization(self) -> float:
        """Calculate atlas texture utilization (0.0 to 1.0)."""
        used_area = sum(
            g.uv_width * g.uv_height * self.width * self.height
            for g in self.glyphs.values()
        )
        return used_area / (self.width * self.height) if self.width and self.height else 0.0


@dataclass
class SDFFont:
    """Signed Distance Field font for resolution-independent rendering.

    SDF fonts store the distance to the nearest edge instead of the actual
    glyph image, allowing smooth scaling at any size.
    """
    base_font: Font
    atlas: FontAtlas
    sdf_spread: float = 4.0  # Distance field spread in pixels

    # SDF-specific metrics
    _distance_scale: float = 1.0

    def get_scaled_spread(self, target_size: float) -> float:
        """Get SDF spread scaled for target font size."""
        scale = target_size / self.base_font.size
        return self.sdf_spread * scale

    def get_threshold(self) -> float:
        """Get the edge threshold for SDF rendering (typically 0.5)."""
        return 0.5

    def get_smoothing(self, target_size: float) -> float:
        """Calculate smoothing factor for anti-aliasing."""
        scale = target_size / self.base_font.size
        # Smoothing inversely proportional to scale for consistent AA
        return 0.25 / (self.sdf_spread * scale) if scale > 0 else 0.0


@dataclass
class FontFallbackChain:
    """Chain of fonts for glyph fallback.

    When a glyph is not found in the primary font, fallback fonts
    are searched in order until the glyph is found.
    """
    primary: Font
    fallbacks: list[Font] = field(default_factory=list)

    def add_fallback(self, font: Font) -> None:
        """Add a fallback font to the chain."""
        self.fallbacks.append(font)

    def find_font_for_glyph(self, codepoint: int) -> Font | None:
        """Find a font that contains the given glyph.

        Args:
            codepoint: Unicode codepoint to search for

        Returns:
            Font containing the glyph, or None if not found
        """
        if self.primary.has_glyph(codepoint):
            return self.primary

        for font in self.fallbacks:
            if font.has_glyph(codepoint):
                return font

        return None

    def __iter__(self) -> Iterator[Font]:
        """Iterate over all fonts in the chain."""
        yield self.primary
        yield from self.fallbacks


class FontManager:
    """Manages font loading, caching, and font family organization.

    Provides a central point for font resource management with:
    - Font file loading (TTF, OTF)
    - Font caching by configuration
    - Font family organization
    - Font fallback chain creation
    """

    # Default glyph size estimation ratios (relative to font size)
    GLYPH_WIDTH_RATIO: float = 0.6
    GLYPH_HEIGHT_RATIO: float = 1.2
    GLYPH_BEARING_Y_RATIO: float = 0.8

    def __init__(self, default_font_size: float = 16.0) -> None:
        """Initialize the font manager.

        Args:
            default_font_size: Default size for newly loaded fonts
        """
        self._fonts: dict[str, Font] = {}
        self._families: dict[str, FontFamily] = {}
        self._atlases: dict[str, FontAtlas] = {}
        self._sdf_fonts: dict[str, SDFFont] = {}
        self._font_paths: dict[str, Path] = {}
        self._default_size = default_font_size
        self._default_font: Font | None = None

        # Weak references for size variants
        self._size_variants: WeakValueDictionary[str, Font] = WeakValueDictionary()

        # Font loaders by extension
        self._loaders: dict[str, Callable[[Path], Any]] = {}

    def register_loader(self, extension: str, loader: Callable[[Path], Any]) -> None:
        """Register a font loader for a file extension.

        Args:
            extension: File extension (e.g., ".ttf", ".otf")
            loader: Function that loads font data from a path
        """
        self._loaders[extension.lower()] = loader

    def load_font(
        self,
        path: str | Path,
        size: float | None = None,
        weight: FontWeight = FontWeight.NORMAL,
        style: FontStyle = FontStyle.NORMAL,
    ) -> Font:
        """Load a font from a file.

        Args:
            path: Path to the font file
            size: Font size (uses default if not specified)
            weight: Font weight
            style: Font style

        Returns:
            Loaded Font object

        Raises:
            FileNotFoundError: If font file doesn't exist
            ValueError: If font format is unsupported
        """
        path = Path(path)
        size = size or self._default_size

        # Generate cache key
        cache_key = self._make_cache_key(path, size, weight, style)

        if cache_key in self._fonts:
            return self._fonts[cache_key]

        if not path.exists():
            raise FileNotFoundError(f"Font file not found: {path}")

        # Get font family name from path (simplified - real impl would parse font)
        family_name = path.stem

        # Create font instance
        font = Font(
            family=family_name,
            size=size,
            weight=weight,
            style=style,
        )

        # Store the path for later reference
        self._font_paths[cache_key] = path
        font._manager_cache_key = cache_key

        # Load font data if we have a loader
        ext = path.suffix.lower()
        if ext in self._loaders:
            font._handle = self._loaders[ext](path)

        # Cache the font
        self._fonts[cache_key] = font

        # Add to family
        if family_name not in self._families:
            self._families[family_name] = FontFamily(family_name)
        self._families[family_name].add_font(font)

        return font

    def get_font(
        self,
        family: str,
        size: float | None = None,
        weight: FontWeight = FontWeight.NORMAL,
        style: FontStyle = FontStyle.NORMAL,
    ) -> Font | None:
        """Get a loaded font by family name and properties.

        Args:
            family: Font family name
            size: Font size (uses default if not specified)
            weight: Font weight
            style: Font style

        Returns:
            Font if found, None otherwise
        """
        size = size or self._default_size

        if family in self._families:
            return self._families[family].get_font(weight, style, size)

        return None

    def get_family(self, name: str) -> FontFamily | None:
        """Get a font family by name."""
        return self._families.get(name)

    def create_fallback_chain(self, *fonts: Font) -> FontFallbackChain:
        """Create a font fallback chain.

        Args:
            fonts: Fonts to include in the chain (first is primary)

        Returns:
            FontFallbackChain for glyph lookup
        """
        if not fonts:
            raise ValueError("At least one font required for fallback chain")

        chain = FontFallbackChain(primary=fonts[0])
        for font in fonts[1:]:
            chain.add_fallback(font)

        return chain

    def create_atlas(
        self,
        font: Font,
        width: int = 1024,
        height: int = 1024,
        characters: str | None = None,
    ) -> FontAtlas:
        """Create a font atlas for bitmap rendering.

        Args:
            font: Font to create atlas for
            width: Atlas texture width
            height: Atlas texture height
            characters: Characters to include (None = common Latin)

        Returns:
            FontAtlas with pre-rendered glyphs
        """
        atlas_key = f"{font.cache_key()}:{width}x{height}"

        if atlas_key in self._atlases:
            return self._atlases[atlas_key]

        atlas = FontAtlas(
            texture_id=hash(atlas_key) & 0xFFFFFFFF,
            width=width,
            height=height,
        )

        # Default character set if not specified
        if characters is None:
            characters = self._get_default_charset()

        # Add glyphs to atlas (simplified - real impl renders glyphs)
        for char in characters:
            codepoint = ord(char)
            # Estimate glyph size based on font size
            glyph_width = int(font.size * self.GLYPH_WIDTH_RATIO)
            glyph_height = int(font.size * self.GLYPH_HEIGHT_RATIO)

            pos = atlas.add_glyph(codepoint, glyph_width, glyph_height)
            if pos:
                x, y = pos
                atlas.glyphs[codepoint] = GlyphMetrics(
                    codepoint=codepoint,
                    width=glyph_width,
                    height=glyph_height,
                    bearing_x=0,
                    bearing_y=glyph_height * 0.8,
                    advance=glyph_width + font.letter_spacing,
                    uv_x=x / width,
                    uv_y=y / height,
                    uv_width=glyph_width / width,
                    uv_height=glyph_height / height,
                )

        self._atlases[atlas_key] = atlas
        return atlas

    def create_sdf_font(
        self,
        font: Font,
        atlas_width: int = 1024,
        atlas_height: int = 1024,
        sdf_spread: float = 4.0,
        characters: str | None = None,
    ) -> SDFFont:
        """Create an SDF (Signed Distance Field) font.

        Args:
            font: Base font to create SDF from
            atlas_width: SDF atlas width
            atlas_height: SDF atlas height
            sdf_spread: Distance field spread in pixels
            characters: Characters to include

        Returns:
            SDFFont for resolution-independent rendering
        """
        sdf_key = f"sdf:{font.cache_key()}:{atlas_width}x{atlas_height}:{sdf_spread}"

        if sdf_key in self._sdf_fonts:
            return self._sdf_fonts[sdf_key]

        # Create underlying atlas
        atlas = self.create_atlas(font, atlas_width, atlas_height, characters)

        sdf_font = SDFFont(
            base_font=font,
            atlas=atlas,
            sdf_spread=sdf_spread,
        )

        self._sdf_fonts[sdf_key] = sdf_font
        return sdf_font

    def set_default_font(self, font: Font) -> None:
        """Set the default font for the application."""
        self._default_font = font

    def get_default_font(self) -> Font | None:
        """Get the default font."""
        return self._default_font

    def unload_font(self, font: Font) -> bool:
        """Unload a font and free resources.

        Args:
            font: Font to unload

        Returns:
            True if font was unloaded, False if not found
        """
        cache_key = font.cache_key()
        if cache_key in self._fonts:
            del self._fonts[cache_key]

            # Remove from family if present
            family = self._families.get(font.family)
            if family and (font.weight, font.style) in family.fonts:
                del family.fonts[(font.weight, font.style)]

            return True
        return False

    def clear(self) -> None:
        """Clear all loaded fonts and cached resources."""
        self._fonts.clear()
        self._families.clear()
        self._atlases.clear()
        self._sdf_fonts.clear()
        self._font_paths.clear()
        self._size_variants.clear()
        self._default_font = None

    @property
    def loaded_fonts(self) -> list[Font]:
        """Get a list of all loaded fonts."""
        return list(self._fonts.values())

    @property
    def font_families(self) -> list[str]:
        """Get a list of all loaded font family names."""
        return list(self._families.keys())

    def _make_cache_key(
        self,
        path: Path,
        size: float,
        weight: FontWeight,
        style: FontStyle,
    ) -> str:
        """Generate a cache key for a font configuration."""
        # Use hash() for cache key - simpler and sufficient for in-memory caching
        path_hash = format(hash(str(path)) & 0xFFFFFFFF, '08x')
        return f"{path_hash}:{size}:{weight.value}:{style.name}"

    def _get_default_charset(self) -> str:
        """Get the default character set for atlas generation."""
        # Basic Latin + Latin-1 Supplement + common punctuation
        chars = []
        # Printable ASCII
        chars.extend(chr(c) for c in range(32, 127))
        # Latin-1 Supplement (common accented characters)
        chars.extend(chr(c) for c in range(160, 256))
        return "".join(chars)

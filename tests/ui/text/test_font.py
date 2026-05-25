"""
Comprehensive tests for Font management system.

Tests cover:
- Font class initialization and properties
- FontWeight and FontStyle enums
- GlyphMetrics handling
- FontFamily management
- FontAtlas creation and glyph placement
- SDFFont configuration
- FontFallbackChain glyph lookup
- FontManager loading and caching
"""

import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tempfile
import os

from engine.ui.text.font import (
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


class TestFontWeight:
    """Tests for FontWeight enum."""

    def test_font_weight_values(self):
        """Test all font weight values exist with correct numeric values."""
        assert FontWeight.THIN.value == 100
        assert FontWeight.EXTRA_LIGHT.value == 200
        assert FontWeight.LIGHT.value == 300
        assert FontWeight.NORMAL.value == 400
        assert FontWeight.MEDIUM.value == 500
        assert FontWeight.SEMI_BOLD.value == 600
        assert FontWeight.BOLD.value == 700
        assert FontWeight.EXTRA_BOLD.value == 800
        assert FontWeight.BLACK.value == 900


class TestFontStyle:
    """Tests for FontStyle enum."""

    def test_font_style_values(self):
        """Test all font style values exist."""
        assert FontStyle.NORMAL
        assert FontStyle.ITALIC
        assert FontStyle.OBLIQUE


class TestGlyphMetrics:
    """Tests for GlyphMetrics class."""

    def test_glyph_metrics_creation(self):
        """Test creating glyph metrics."""
        glyph = GlyphMetrics(
            codepoint=65,  # 'A'
            width=10.0,
            height=14.0,
            bearing_x=1.0,
            bearing_y=12.0,
            advance=12.0,
        )
        assert glyph.codepoint == 65
        assert glyph.width == 10.0
        assert glyph.height == 14.0
        assert glyph.advance == 12.0

    def test_glyph_metrics_uv_defaults(self):
        """Test UV coordinates default to zero."""
        glyph = GlyphMetrics(
            codepoint=65,
            width=10.0, height=14.0,
            bearing_x=1.0, bearing_y=12.0,
            advance=12.0,
        )
        assert glyph.uv_x == 0.0
        assert glyph.uv_y == 0.0
        assert glyph.uv_width == 0.0
        assert glyph.uv_height == 0.0

    def test_glyph_metrics_with_uv(self):
        """Test glyph metrics with UV coordinates."""
        glyph = GlyphMetrics(
            codepoint=65,
            width=10.0, height=14.0,
            bearing_x=1.0, bearing_y=12.0,
            advance=12.0,
            uv_x=0.1, uv_y=0.2,
            uv_width=0.05, uv_height=0.07,
        )
        assert glyph.uv_x == 0.1
        assert glyph.uv_y == 0.2

    def test_glyph_metrics_frozen(self):
        """Test glyph metrics is immutable."""
        glyph = GlyphMetrics(
            codepoint=65,
            width=10.0, height=14.0,
            bearing_x=1.0, bearing_y=12.0,
            advance=12.0,
        )
        with pytest.raises(AttributeError):
            glyph.width = 20.0


class TestFont:
    """Tests for Font class."""

    def test_font_creation(self):
        """Test creating a font."""
        font = Font(family="Roboto", size=16.0)
        assert font.family == "Roboto"
        assert font.size == 16.0
        assert font.weight == FontWeight.NORMAL
        assert font.style == FontStyle.NORMAL

    def test_font_with_weight_and_style(self):
        """Test font with custom weight and style."""
        font = Font(
            family="Roboto",
            size=14.0,
            weight=FontWeight.BOLD,
            style=FontStyle.ITALIC,
        )
        assert font.weight == FontWeight.BOLD
        assert font.style == FontStyle.ITALIC

    def test_font_line_height(self):
        """Test font line height default."""
        font = Font(family="Roboto", size=16.0)
        assert font.line_height == 1.2

    def test_font_custom_line_height(self):
        """Test font with custom line height."""
        font = Font(family="Roboto", size=16.0, line_height=1.5)
        assert font.line_height == 1.5

    def test_font_letter_spacing(self):
        """Test font letter spacing."""
        font = Font(family="Roboto", size=16.0, letter_spacing=2.0)
        assert font.letter_spacing == 2.0

    def test_font_negative_size_rejected(self):
        """Test font rejects negative size."""
        with pytest.raises(ValueError, match="Font size must be positive"):
            Font(family="Roboto", size=-16.0)

    def test_font_zero_size_rejected(self):
        """Test font rejects zero size."""
        with pytest.raises(ValueError, match="Font size must be positive"):
            Font(family="Roboto", size=0)

    def test_font_negative_line_height_rejected(self):
        """Test font rejects negative line height."""
        with pytest.raises(ValueError, match="Line height must be positive"):
            Font(family="Roboto", size=16.0, line_height=-1.0)

    def test_font_line_spacing(self):
        """Test font line spacing calculation."""
        font = Font(family="Roboto", size=16.0, line_height=1.5)
        assert font.line_spacing == 24.0  # 16 * 1.5

    def test_font_cache_key(self):
        """Test font cache key generation."""
        font = Font(family="Roboto", size=16.0)
        key = font.cache_key()
        assert "Roboto" in key
        assert "16" in key

    def test_font_cache_key_unique(self):
        """Test different fonts have different cache keys."""
        font1 = Font(family="Roboto", size=16.0)
        font2 = Font(family="Roboto", size=18.0)
        font3 = Font(family="Arial", size=16.0)

        assert font1.cache_key() != font2.cache_key()
        assert font1.cache_key() != font3.cache_key()

    def test_font_with_size(self):
        """Test with_size creates font copy with new size."""
        font1 = Font(family="Roboto", size=16.0, weight=FontWeight.BOLD)
        font2 = font1.with_size(24.0)

        assert font2.size == 24.0
        assert font2.weight == FontWeight.BOLD  # Preserved
        assert font1.size == 16.0  # Original unchanged

    def test_font_with_weight(self):
        """Test with_weight creates font copy with new weight."""
        font1 = Font(family="Roboto", size=16.0)
        font2 = font1.with_weight(FontWeight.BOLD)

        assert font2.weight == FontWeight.BOLD
        assert font1.weight == FontWeight.NORMAL

    def test_font_with_style(self):
        """Test with_style creates font copy with new style."""
        font1 = Font(family="Roboto", size=16.0)
        font2 = font1.with_style(FontStyle.ITALIC)

        assert font2.style == FontStyle.ITALIC
        assert font1.style == FontStyle.NORMAL

    def test_font_get_glyph_not_found(self):
        """Test getting glyph that doesn't exist."""
        font = Font(family="Roboto", size=16.0)
        glyph = font.get_glyph(65)  # 'A'
        assert glyph is None

    def test_font_has_glyph(self):
        """Test checking if font has glyph."""
        font = Font(family="Roboto", size=16.0)
        assert not font.has_glyph(65)

    def test_font_ascender_descender(self):
        """Test font ascender and descender properties."""
        font = Font(family="Roboto", size=16.0)
        # Without internal metrics set, should return 0
        assert font.ascender == 0
        assert font.descender == 0


class TestFontFamily:
    """Tests for FontFamily class."""

    def test_font_family_creation(self):
        """Test creating a font family."""
        family = FontFamily(name="Roboto")
        assert family.name == "Roboto"
        assert len(family.fonts) == 0

    def test_font_family_add_font(self):
        """Test adding font to family."""
        family = FontFamily(name="Roboto")
        font = Font(family="Roboto", size=16.0)
        family.add_font(font)

        assert len(family.fonts) == 1

    def test_font_family_add_font_wrong_family_rejected(self):
        """Test adding font with wrong family name is rejected."""
        family = FontFamily(name="Roboto")
        font = Font(family="Arial", size=16.0)

        with pytest.raises(ValueError, match="Font family mismatch"):
            family.add_font(font)

    def test_font_family_add_multiple_variants(self):
        """Test adding multiple font variants."""
        family = FontFamily(name="Roboto")
        family.add_font(Font(family="Roboto", size=16.0, weight=FontWeight.NORMAL))
        family.add_font(Font(family="Roboto", size=16.0, weight=FontWeight.BOLD))
        family.add_font(Font(family="Roboto", size=16.0, style=FontStyle.ITALIC))

        assert len(family.fonts) == 3

    def test_font_family_get_font_exact_match(self):
        """Test getting exact font match."""
        family = FontFamily(name="Roboto")
        font = Font(family="Roboto", size=16.0, weight=FontWeight.BOLD)
        family.add_font(font)

        result = family.get_font(weight=FontWeight.BOLD, size=16.0)

        assert result is not None
        assert result.weight == FontWeight.BOLD

    def test_font_family_get_font_closest_weight(self):
        """Test getting closest weight when exact not found."""
        family = FontFamily(name="Roboto")
        family.add_font(Font(family="Roboto", size=16.0, weight=FontWeight.NORMAL))
        family.add_font(Font(family="Roboto", size=16.0, weight=FontWeight.BOLD))

        result = family.get_font(weight=FontWeight.SEMI_BOLD, size=16.0)

        # Should return closest weight
        assert result is not None
        assert result.weight in (FontWeight.NORMAL, FontWeight.BOLD)

    def test_font_family_get_font_different_size(self):
        """Test getting font with different size."""
        family = FontFamily(name="Roboto")
        family.add_font(Font(family="Roboto", size=16.0))

        result = family.get_font(size=24.0)

        assert result is not None
        assert result.size == 24.0

    def test_font_family_get_font_empty(self):
        """Test getting font from empty family."""
        family = FontFamily(name="Roboto")

        result = family.get_font()

        assert result is None

    def test_font_family_has_variant(self):
        """Test checking if variant exists."""
        family = FontFamily(name="Roboto")
        family.add_font(Font(family="Roboto", size=16.0, weight=FontWeight.BOLD))

        assert family.has_variant(FontWeight.BOLD, FontStyle.NORMAL)
        assert not family.has_variant(FontWeight.LIGHT, FontStyle.NORMAL)


class TestFontAtlas:
    """Tests for FontAtlas class."""

    def test_font_atlas_creation(self):
        """Test creating a font atlas."""
        atlas = FontAtlas(texture_id=1, width=1024, height=1024)
        assert atlas.texture_id == 1
        assert atlas.width == 1024
        assert atlas.height == 1024

    def test_font_atlas_add_glyph(self):
        """Test adding glyph to atlas."""
        atlas = FontAtlas(texture_id=1, width=1024, height=1024)

        pos = atlas.add_glyph(codepoint=65, width=10, height=14)

        assert pos is not None
        assert isinstance(pos, tuple)
        assert len(pos) == 2

    def test_font_atlas_add_multiple_glyphs(self):
        """Test adding multiple glyphs."""
        atlas = FontAtlas(texture_id=1, width=1024, height=1024)

        positions = []
        for i in range(10):
            pos = atlas.add_glyph(codepoint=65 + i, width=10, height=14)
            positions.append(pos)

        # All positions should be unique
        assert len(set(positions)) == 10

    def test_font_atlas_row_wrap(self):
        """Test atlas wraps to next row when needed."""
        atlas = FontAtlas(texture_id=1, width=100, height=200)

        # Add glyphs that will need to wrap
        pos1 = atlas.add_glyph(codepoint=65, width=40, height=20)
        pos2 = atlas.add_glyph(codepoint=66, width=40, height=20)
        pos3 = atlas.add_glyph(codepoint=67, width=40, height=20)  # Should wrap

        assert pos1 is not None
        assert pos2 is not None
        assert pos3 is not None
        # pos3 should be on a different row
        assert pos3[1] > pos1[1]

    def test_font_atlas_out_of_space(self):
        """Test atlas returns None when out of space."""
        atlas = FontAtlas(texture_id=1, width=50, height=50)

        # Try to add a glyph larger than remaining space
        pos1 = atlas.add_glyph(codepoint=65, width=45, height=45)
        pos2 = atlas.add_glyph(codepoint=66, width=45, height=45)

        assert pos1 is not None
        assert pos2 is None  # No space

    def test_font_atlas_get_glyph(self):
        """Test getting glyph from atlas."""
        atlas = FontAtlas(texture_id=1, width=1024, height=1024)
        atlas.glyphs[65] = GlyphMetrics(
            codepoint=65,
            width=10, height=14,
            bearing_x=1, bearing_y=12,
            advance=12,
        )

        glyph = atlas.get_glyph(65)

        assert glyph is not None
        assert glyph.codepoint == 65

    def test_font_atlas_get_glyph_not_found(self):
        """Test getting non-existent glyph."""
        atlas = FontAtlas(texture_id=1, width=1024, height=1024)

        glyph = atlas.get_glyph(65)

        assert glyph is None

    def test_font_atlas_utilization(self):
        """Test atlas utilization calculation."""
        atlas = FontAtlas(texture_id=1, width=100, height=100)

        # Empty atlas
        assert atlas.utilization == 0.0


class TestSDFFont:
    """Tests for SDFFont class."""

    def test_sdf_font_creation(self):
        """Test creating an SDF font."""
        base_font = Font(family="Roboto", size=32.0)
        atlas = FontAtlas(texture_id=1, width=512, height=512)

        sdf = SDFFont(base_font=base_font, atlas=atlas)

        assert sdf.base_font is base_font
        assert sdf.atlas is atlas
        assert sdf.sdf_spread == 4.0  # Default

    def test_sdf_font_custom_spread(self):
        """Test SDF font with custom spread."""
        base_font = Font(family="Roboto", size=32.0)
        atlas = FontAtlas(texture_id=1, width=512, height=512)

        sdf = SDFFont(base_font=base_font, atlas=atlas, sdf_spread=8.0)

        assert sdf.sdf_spread == 8.0

    def test_sdf_font_scaled_spread(self):
        """Test getting scaled spread for target size."""
        base_font = Font(family="Roboto", size=32.0)
        atlas = FontAtlas(texture_id=1, width=512, height=512)
        sdf = SDFFont(base_font=base_font, atlas=atlas, sdf_spread=4.0)

        # At 64pt (2x), spread should be 8
        spread = sdf.get_scaled_spread(64.0)
        assert spread == 8.0

        # At 16pt (0.5x), spread should be 2
        spread = sdf.get_scaled_spread(16.0)
        assert spread == 2.0

    def test_sdf_font_threshold(self):
        """Test SDF edge threshold."""
        base_font = Font(family="Roboto", size=32.0)
        atlas = FontAtlas(texture_id=1, width=512, height=512)
        sdf = SDFFont(base_font=base_font, atlas=atlas)

        threshold = sdf.get_threshold()

        assert threshold == 0.5

    def test_sdf_font_smoothing(self):
        """Test SDF smoothing factor."""
        base_font = Font(family="Roboto", size=32.0)
        atlas = FontAtlas(texture_id=1, width=512, height=512)
        sdf = SDFFont(base_font=base_font, atlas=atlas, sdf_spread=4.0)

        smoothing = sdf.get_smoothing(32.0)

        assert smoothing > 0


class TestFontFallbackChain:
    """Tests for FontFallbackChain class."""

    def test_fallback_chain_creation(self):
        """Test creating a fallback chain."""
        primary = Font(family="Roboto", size=16.0)
        chain = FontFallbackChain(primary=primary)

        assert chain.primary is primary
        assert len(chain.fallbacks) == 0

    def test_fallback_chain_add_fallback(self):
        """Test adding fallback fonts."""
        primary = Font(family="Roboto", size=16.0)
        fallback = Font(family="Arial", size=16.0)
        chain = FontFallbackChain(primary=primary)

        chain.add_fallback(fallback)

        assert len(chain.fallbacks) == 1

    def test_fallback_chain_find_font_in_primary(self):
        """Test finding glyph in primary font."""
        primary = Font(family="Roboto", size=16.0)
        # Manually add glyph to primary
        primary._glyphs[65] = GlyphMetrics(
            codepoint=65,
            width=10, height=14,
            bearing_x=1, bearing_y=12,
            advance=12,
        )
        chain = FontFallbackChain(primary=primary)

        font = chain.find_font_for_glyph(65)

        assert font is primary

    def test_fallback_chain_find_font_in_fallback(self):
        """Test finding glyph in fallback font."""
        primary = Font(family="Roboto", size=16.0)
        fallback = Font(family="Arial", size=16.0)
        fallback._glyphs[65] = GlyphMetrics(
            codepoint=65,
            width=10, height=14,
            bearing_x=1, bearing_y=12,
            advance=12,
        )
        chain = FontFallbackChain(primary=primary)
        chain.add_fallback(fallback)

        font = chain.find_font_for_glyph(65)

        assert font is fallback

    def test_fallback_chain_find_font_not_found(self):
        """Test no font found for glyph."""
        primary = Font(family="Roboto", size=16.0)
        chain = FontFallbackChain(primary=primary)

        font = chain.find_font_for_glyph(65)

        assert font is None

    def test_fallback_chain_iteration(self):
        """Test iterating over fallback chain."""
        primary = Font(family="Roboto", size=16.0)
        fallback1 = Font(family="Arial", size=16.0)
        fallback2 = Font(family="Helvetica", size=16.0)
        chain = FontFallbackChain(primary=primary, fallbacks=[fallback1, fallback2])

        fonts = list(chain)

        assert len(fonts) == 3
        assert fonts[0] is primary


class TestFontManager:
    """Tests for FontManager class."""

    def test_font_manager_creation(self):
        """Test creating a font manager."""
        manager = FontManager()
        assert manager._default_size == 16.0

    def test_font_manager_custom_default_size(self):
        """Test font manager with custom default size."""
        manager = FontManager(default_font_size=14.0)
        assert manager._default_size == 14.0

    def test_font_manager_load_font_file_not_found(self):
        """Test loading non-existent font file."""
        manager = FontManager()

        with pytest.raises(FileNotFoundError):
            manager.load_font("/nonexistent/font.ttf")

    def test_font_manager_load_font(self):
        """Test loading a font file."""
        manager = FontManager()

        # Create a temporary file to act as font
        with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as f:
            f.write(b"fake font data")
            temp_path = f.name

        try:
            font = manager.load_font(temp_path, size=16.0)
            assert font is not None
            assert font.size == 16.0
        finally:
            os.unlink(temp_path)

    def test_font_manager_load_font_cached(self):
        """Test loaded fonts are cached."""
        manager = FontManager()

        with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as f:
            f.write(b"fake font data")
            temp_path = f.name

        try:
            font1 = manager.load_font(temp_path, size=16.0)
            font2 = manager.load_font(temp_path, size=16.0)
            assert font1 is font2  # Same instance
        finally:
            os.unlink(temp_path)

    def test_font_manager_get_font(self):
        """Test getting font by family name."""
        manager = FontManager()

        with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as f:
            f.write(b"fake font data")
            temp_path = f.name

        try:
            manager.load_font(temp_path, size=16.0)
            family_name = Path(temp_path).stem

            font = manager.get_font(family_name, size=16.0)
            assert font is not None
        finally:
            os.unlink(temp_path)

    def test_font_manager_get_font_not_found(self):
        """Test getting non-existent font family."""
        manager = FontManager()

        font = manager.get_font("NonExistentFamily")

        assert font is None

    def test_font_manager_get_family(self):
        """Test getting font family."""
        manager = FontManager()

        with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as f:
            f.write(b"fake font data")
            temp_path = f.name

        try:
            manager.load_font(temp_path)
            family_name = Path(temp_path).stem

            family = manager.get_family(family_name)
            assert family is not None
        finally:
            os.unlink(temp_path)

    def test_font_manager_create_fallback_chain(self):
        """Test creating fallback chain."""
        manager = FontManager()
        font1 = Font(family="Roboto", size=16.0)
        font2 = Font(family="Arial", size=16.0)

        chain = manager.create_fallback_chain(font1, font2)

        assert chain.primary is font1
        assert len(chain.fallbacks) == 1

    def test_font_manager_create_fallback_chain_empty_rejected(self):
        """Test creating empty fallback chain is rejected."""
        manager = FontManager()

        with pytest.raises(ValueError, match="At least one font required"):
            manager.create_fallback_chain()

    def test_font_manager_create_atlas(self):
        """Test creating font atlas."""
        manager = FontManager()
        font = Font(family="Roboto", size=16.0)

        atlas = manager.create_atlas(font, width=512, height=512)

        assert atlas is not None
        assert atlas.width == 512
        assert atlas.height == 512

    def test_font_manager_create_sdf_font(self):
        """Test creating SDF font."""
        manager = FontManager()
        font = Font(family="Roboto", size=32.0)

        sdf = manager.create_sdf_font(font)

        assert sdf is not None
        assert sdf.base_font is font

    def test_font_manager_set_default_font(self):
        """Test setting default font."""
        manager = FontManager()
        font = Font(family="Roboto", size=16.0)

        manager.set_default_font(font)

        assert manager.get_default_font() is font

    def test_font_manager_unload_font(self):
        """Test unloading a font."""
        manager = FontManager()

        with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as f:
            f.write(b"fake font data")
            temp_path = f.name

        try:
            font = manager.load_font(temp_path)
            initial_count = len(manager.loaded_fonts)

            result = manager.unload_font(font)

            # Unload should succeed and remove the font from cache
            assert result is True, (
                "unload_font should return True when font was loaded. "
                "If this fails, there may be a cache key mismatch between "
                "FontManager._make_cache_key and Font.cache_key"
            )
            assert len(manager.loaded_fonts) == initial_count - 1, (
                "Font should be removed from loaded_fonts after unload"
            )
        finally:
            os.unlink(temp_path)

    def test_font_manager_unload_font_not_found(self):
        """Test unloading a font that was never loaded."""
        manager = FontManager()

        # Create a font that was never loaded via FontManager
        font = Font(family="NonExistent", size=12.0)

        result = manager.unload_font(font)

        assert result is False

    def test_font_manager_clear(self):
        """Test clearing all fonts."""
        manager = FontManager()

        with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as f:
            f.write(b"fake font data")
            temp_path = f.name

        try:
            manager.load_font(temp_path)
            manager.clear()

            assert len(manager.loaded_fonts) == 0
            assert len(manager.font_families) == 0
        finally:
            os.unlink(temp_path)

    def test_font_manager_loaded_fonts(self):
        """Test getting list of loaded fonts."""
        manager = FontManager()

        with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as f:
            f.write(b"fake font data")
            temp_path = f.name

        try:
            manager.load_font(temp_path)

            fonts = manager.loaded_fonts
            assert len(fonts) == 1
        finally:
            os.unlink(temp_path)

    def test_font_manager_font_families(self):
        """Test getting list of font families."""
        manager = FontManager()

        with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as f:
            f.write(b"fake font data")
            temp_path = f.name

        try:
            manager.load_font(temp_path)

            families = manager.font_families
            assert len(families) == 1
        finally:
            os.unlink(temp_path)

    def test_font_manager_register_loader(self):
        """Test registering custom font loader."""
        manager = FontManager()
        loader_called = [False]

        def custom_loader(path: Path) -> Any:
            loader_called[0] = True
            return "custom_handle"

        manager.register_loader(".custom", custom_loader)

        with tempfile.NamedTemporaryFile(suffix=".custom", delete=False) as f:
            f.write(b"custom font data")
            temp_path = f.name

        try:
            font = manager.load_font(temp_path)
            assert loader_called[0] is True
            assert font._handle == "custom_handle"
        finally:
            os.unlink(temp_path)

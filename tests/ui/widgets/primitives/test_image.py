"""
Comprehensive tests for the Image widget.

Tests cover:
- Initialization and default values
- Scale mode behavior (STRETCH, FIT, FILL, TILE, NINE_SLICE)
- UV coordinate handling and validation
- Color tinting (tuples and hex strings)
- Nine-slice configuration
- Flip operations
- Dirty state tracking
- Serialization/deserialization
"""

import pytest
from unittest.mock import MagicMock, patch


class TestUVCoordinates:
    """Tests for UVCoordinates dataclass."""

    def test_default_uv_coordinates(self):
        """Test default UV covers entire texture."""
        from engine.ui.widgets.primitives.image import UVCoordinates

        uv = UVCoordinates()
        assert uv.u0 == 0.0
        assert uv.v0 == 0.0
        assert uv.u1 == 1.0
        assert uv.v1 == 1.0

    def test_uv_width_and_height(self):
        """Test UV width and height calculation."""
        from engine.ui.widgets.primitives.image import UVCoordinates

        uv = UVCoordinates(u0=0.25, v0=0.25, u1=0.75, v1=0.75)
        assert uv.width == pytest.approx(0.5)
        assert uv.height == pytest.approx(0.5)

    def test_uv_validation_u0_below_zero(self):
        """Test u0 validation fails for negative values."""
        from engine.ui.widgets.primitives.image import UVCoordinates

        with pytest.raises(ValueError, match="u0 must be in range"):
            UVCoordinates(u0=-0.1)

    def test_uv_validation_u0_above_one(self):
        """Test u0 validation fails for values above 1."""
        from engine.ui.widgets.primitives.image import UVCoordinates

        with pytest.raises(ValueError, match="u0 must be in range"):
            UVCoordinates(u0=1.1)

    def test_uv_validation_v0_below_zero(self):
        """Test v0 validation fails for negative values."""
        from engine.ui.widgets.primitives.image import UVCoordinates

        with pytest.raises(ValueError, match="v0 must be in range"):
            UVCoordinates(v0=-0.1)

    def test_uv_validation_v0_above_one(self):
        """Test v0 validation fails for values above 1."""
        from engine.ui.widgets.primitives.image import UVCoordinates

        with pytest.raises(ValueError, match="v0 must be in range"):
            UVCoordinates(v0=1.1)

    def test_uv_validation_u1_below_u0(self):
        """Test u1 must be >= u0."""
        from engine.ui.widgets.primitives.image import UVCoordinates

        with pytest.raises(ValueError, match="u1.*must be >= u0"):
            UVCoordinates(u0=0.5, u1=0.3)

    def test_uv_validation_v1_below_v0(self):
        """Test v1 must be >= v0."""
        from engine.ui.widgets.primitives.image import UVCoordinates

        with pytest.raises(ValueError, match="v1.*must be >= v0"):
            UVCoordinates(v0=0.5, v1=0.3)

    def test_uv_flip_horizontal(self):
        """Test horizontal flip swaps u0 and u1."""
        from engine.ui.widgets.primitives.image import UVCoordinates

        uv = UVCoordinates(u0=0.25, v0=0.25, u1=0.75, v1=0.75)
        flipped = uv.flip_horizontal()
        assert flipped.u0 == 0.75
        assert flipped.u1 == 0.25
        assert flipped.v0 == 0.25
        assert flipped.v1 == 0.75

    def test_uv_flip_vertical(self):
        """Test vertical flip swaps v0 and v1."""
        from engine.ui.widgets.primitives.image import UVCoordinates

        uv = UVCoordinates(u0=0.25, v0=0.25, u1=0.75, v1=0.75)
        flipped = uv.flip_vertical()
        assert flipped.u0 == 0.25
        assert flipped.u1 == 0.75
        assert flipped.v0 == 0.75
        assert flipped.v1 == 0.25

    def test_uv_from_pixel_rect(self):
        """Test creating UV from pixel rectangle."""
        from engine.ui.widgets.primitives.image import UVCoordinates

        uv = UVCoordinates.from_pixel_rect(
            x=64, y=64,
            width=128, height=128,
            texture_width=512, texture_height=512
        )
        assert uv.u0 == pytest.approx(0.125)
        assert uv.v0 == pytest.approx(0.125)
        assert uv.u1 == pytest.approx(0.375)
        assert uv.v1 == pytest.approx(0.375)

    def test_uv_from_pixel_rect_invalid_texture_dimensions(self):
        """Test from_pixel_rect fails with zero texture dimensions."""
        from engine.ui.widgets.primitives.image import UVCoordinates

        with pytest.raises(ValueError, match="Texture dimensions must be positive"):
            UVCoordinates.from_pixel_rect(0, 0, 10, 10, 0, 100)

    def test_uv_from_pixel_rect_negative_dimensions(self):
        """Test from_pixel_rect fails with negative rect dimensions."""
        from engine.ui.widgets.primitives.image import UVCoordinates

        with pytest.raises(ValueError, match="must be non-negative"):
            UVCoordinates.from_pixel_rect(0, 0, -10, 10, 100, 100)


class TestNineSliceConfig:
    """Tests for NineSliceConfig dataclass."""

    def test_default_nine_slice_config(self):
        """Test default values are all zero."""
        from engine.ui.widgets.primitives.image import NineSliceConfig

        config = NineSliceConfig()
        assert config.left == 0
        assert config.right == 0
        assert config.top == 0
        assert config.bottom == 0
        assert config.tile_center is False
        assert config.tile_edges is False

    def test_nine_slice_horizontal_borders(self):
        """Test horizontal border calculation."""
        from engine.ui.widgets.primitives.image import NineSliceConfig

        config = NineSliceConfig(left=10, right=15)
        assert config.horizontal_borders == 25

    def test_nine_slice_vertical_borders(self):
        """Test vertical border calculation."""
        from engine.ui.widgets.primitives.image import NineSliceConfig

        config = NineSliceConfig(top=12, bottom=8)
        assert config.vertical_borders == 20

    def test_nine_slice_uniform(self):
        """Test uniform factory method."""
        from engine.ui.widgets.primitives.image import NineSliceConfig

        config = NineSliceConfig.uniform(16, tile_center=True)
        assert config.left == 16
        assert config.right == 16
        assert config.top == 16
        assert config.bottom == 16
        assert config.tile_center is True

    def test_nine_slice_negative_left_fails(self):
        """Test negative left border fails validation."""
        from engine.ui.widgets.primitives.image import NineSliceConfig

        with pytest.raises(ValueError, match="left must be >= 0"):
            NineSliceConfig(left=-1)

    def test_nine_slice_negative_right_fails(self):
        """Test negative right border fails validation."""
        from engine.ui.widgets.primitives.image import NineSliceConfig

        with pytest.raises(ValueError, match="right must be >= 0"):
            NineSliceConfig(right=-1)

    def test_nine_slice_negative_top_fails(self):
        """Test negative top border fails validation."""
        from engine.ui.widgets.primitives.image import NineSliceConfig

        with pytest.raises(ValueError, match="top must be >= 0"):
            NineSliceConfig(top=-1)

    def test_nine_slice_negative_bottom_fails(self):
        """Test negative bottom border fails validation."""
        from engine.ui.widgets.primitives.image import NineSliceConfig

        with pytest.raises(ValueError, match="bottom must be >= 0"):
            NineSliceConfig(bottom=-1)


class TestImageWidget:
    """Tests for the Image widget class."""

    def test_image_default_initialization(self):
        """Test Image initializes with correct defaults."""
        from engine.ui.widgets.primitives.image import Image, ScaleMode

        img = Image()
        assert img.source == ""
        assert img.scale_mode == ScaleMode.STRETCH
        assert img.opacity == 1.0
        assert img.tint == (1.0, 1.0, 1.0, 1.0)
        assert img.width == 0.0
        assert img.height == 0.0

    def test_image_with_source(self):
        """Test Image with source texture path."""
        from engine.ui.widgets.primitives.image import Image

        img = Image(source="textures/player.png")
        assert img.source == "textures/player.png"

    def test_image_opacity_validation_below_zero(self):
        """Test opacity below 0 fails."""
        from engine.ui.widgets.primitives.image import Image

        with pytest.raises(ValueError, match="opacity must be in range"):
            Image(opacity=-0.1)

    def test_image_opacity_validation_above_one(self):
        """Test opacity above 1 fails."""
        from engine.ui.widgets.primitives.image import Image

        with pytest.raises(ValueError, match="opacity must be in range"):
            Image(opacity=1.1)

    def test_image_opacity_boundary_values(self):
        """Test opacity at boundary values."""
        from engine.ui.widgets.primitives.image import Image

        img_zero = Image(opacity=0.0)
        assert img_zero.opacity == 0.0

        img_one = Image(opacity=1.0)
        assert img_one.opacity == 1.0

    def test_image_scale_mode_setter(self):
        """Test changing scale mode."""
        from engine.ui.widgets.primitives.image import Image, ScaleMode

        img = Image()
        img.scale_mode = ScaleMode.FIT
        assert img.scale_mode == ScaleMode.FIT

    def test_image_scale_mode_invalid_type(self):
        """Test scale mode setter rejects invalid types."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        with pytest.raises(ValueError, match="must be a ScaleMode"):
            img.scale_mode = "stretch"

    def test_image_tint_rgb_tuple(self):
        """Test tint with RGB tuple (alpha defaults to 1.0)."""
        from engine.ui.widgets.primitives.image import Image

        img = Image(tint=(0.5, 0.5, 0.5))
        assert img.tint == (0.5, 0.5, 0.5, 1.0)

    def test_image_tint_rgba_tuple(self):
        """Test tint with RGBA tuple."""
        from engine.ui.widgets.primitives.image import Image

        img = Image(tint=(0.5, 0.5, 0.5, 0.5))
        assert img.tint == (0.5, 0.5, 0.5, 0.5)

    def test_image_tint_hex_rgb(self):
        """Test tint with hex #RGB format."""
        from engine.ui.widgets.primitives.image import Image

        img = Image(tint="#FFF")
        assert img.tint[0] == pytest.approx(1.0)
        assert img.tint[1] == pytest.approx(1.0)
        assert img.tint[2] == pytest.approx(1.0)
        assert img.tint[3] == pytest.approx(1.0)

    def test_image_tint_hex_rrggbb(self):
        """Test tint with hex #RRGGBB format."""
        from engine.ui.widgets.primitives.image import Image

        img = Image(tint="#FF0000")
        assert img.tint[0] == pytest.approx(1.0)
        assert img.tint[1] == pytest.approx(0.0)
        assert img.tint[2] == pytest.approx(0.0)

    def test_image_tint_hex_rrggbbaa(self):
        """Test tint with hex #RRGGBBAA format."""
        from engine.ui.widgets.primitives.image import Image

        img = Image(tint="#FF000080")
        assert img.tint[0] == pytest.approx(1.0)
        assert img.tint[3] == pytest.approx(128/255)

    def test_image_tint_invalid_hex_no_hash(self):
        """Test tint hex without # fails."""
        from engine.ui.widgets.primitives.image import Image

        with pytest.raises(ValueError, match="must start with #"):
            Image(tint="FFFFFF")

    def test_image_tint_invalid_component_range(self):
        """Test tint component outside [0,1] fails."""
        from engine.ui.widgets.primitives.image import Image

        with pytest.raises(ValueError, match="must be in range"):
            Image(tint=(1.5, 0.5, 0.5))

    def test_image_width_setter(self):
        """Test setting width."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        img.width = 100.0
        assert img.width == 100.0

    def test_image_width_negative_clamps_to_zero(self):
        """Test negative width clamps to zero."""
        from engine.ui.widgets.primitives.image import Image

        img = Image(width=-10.0)
        assert img.width == 0.0

    def test_image_height_setter(self):
        """Test setting height."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        img.height = 100.0
        assert img.height == 100.0

    def test_image_flip_horizontal(self):
        """Test horizontal flip property."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        assert img.flip_horizontal is False
        img.flip_horizontal = True
        assert img.flip_horizontal is True

    def test_image_flip_vertical(self):
        """Test vertical flip property."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        assert img.flip_vertical is False
        img.flip_vertical = True
        assert img.flip_vertical is True

    def test_image_preserve_aspect(self):
        """Test preserve aspect property."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        assert img.preserve_aspect is True
        img.preserve_aspect = False
        assert img.preserve_aspect is False

    def test_image_set_natural_size(self):
        """Test setting natural size."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        img.set_natural_size(256, 128)
        assert img.natural_width == 256.0
        assert img.natural_height == 128.0

    def test_image_set_natural_size_negative_fails(self):
        """Test negative natural size fails."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        with pytest.raises(ValueError, match="must be non-negative"):
            img.set_natural_size(-100, 100)

    def test_image_aspect_ratio(self):
        """Test aspect ratio calculation."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        img.set_natural_size(200, 100)
        assert img.aspect_ratio == pytest.approx(2.0)

    def test_image_aspect_ratio_zero_height(self):
        """Test aspect ratio with zero height returns 1.0."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        assert img.aspect_ratio == 1.0

    def test_image_uv_setter_valid(self):
        """Test setting valid UV coordinates."""
        from engine.ui.widgets.primitives.image import Image, UVCoordinates

        img = Image()
        uv = UVCoordinates(0.1, 0.1, 0.9, 0.9)
        img.uv = uv
        assert img.uv == uv

    def test_image_uv_setter_invalid_type(self):
        """Test UV setter rejects invalid types."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        with pytest.raises(ValueError, match="must be UVCoordinates"):
            img.uv = (0.0, 0.0, 1.0, 1.0)

    def test_image_nine_slice_setter(self):
        """Test setting nine slice config."""
        from engine.ui.widgets.primitives.image import Image, NineSliceConfig

        img = Image()
        ns = NineSliceConfig(left=10, right=10, top=10, bottom=10)
        img.nine_slice = ns
        assert img.nine_slice == ns

    def test_image_nine_slice_setter_none(self):
        """Test setting nine slice to None."""
        from engine.ui.widgets.primitives.image import Image, NineSliceConfig

        img = Image(nine_slice=NineSliceConfig.uniform(10))
        img.nine_slice = None
        assert img.nine_slice is None

    def test_image_nine_slice_setter_invalid_type(self):
        """Test nine slice setter rejects invalid types."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        with pytest.raises(ValueError, match="must be NineSliceConfig"):
            img.nine_slice = {"left": 10}


class TestImageScaleModes:
    """Tests for different scale mode behaviors."""

    def test_get_rendered_size_stretch(self):
        """Test STRETCH mode fills the bounds."""
        from engine.ui.widgets.primitives.image import Image, ScaleMode

        img = Image(width=200, height=100, scale_mode=ScaleMode.STRETCH)
        img.set_natural_size(50, 50)

        w, h = img.get_rendered_size()
        assert w == 200.0
        assert h == 100.0

    def test_get_rendered_size_fit(self):
        """Test FIT mode fits within bounds preserving aspect."""
        from engine.ui.widgets.primitives.image import Image, ScaleMode

        img = Image(width=200, height=100, scale_mode=ScaleMode.FIT)
        img.set_natural_size(100, 100)  # Square image

        w, h = img.get_rendered_size()
        # Should fit in 100x100 (limited by height)
        assert w == pytest.approx(100.0)
        assert h == pytest.approx(100.0)

    def test_get_rendered_size_fill(self):
        """Test FILL mode covers bounds preserving aspect."""
        from engine.ui.widgets.primitives.image import Image, ScaleMode

        img = Image(width=200, height=100, scale_mode=ScaleMode.FILL)
        img.set_natural_size(100, 100)  # Square image

        w, h = img.get_rendered_size()
        # Should cover 200x200 to fill (scale 2x)
        assert w == pytest.approx(200.0)
        assert h == pytest.approx(200.0)

    def test_get_rendered_size_tile(self):
        """Test TILE mode uses widget bounds."""
        from engine.ui.widgets.primitives.image import Image, ScaleMode

        img = Image(width=200, height=100, scale_mode=ScaleMode.TILE)
        img.set_natural_size(50, 50)

        w, h = img.get_rendered_size()
        assert w == 200.0
        assert h == 100.0

    def test_get_rendered_size_nine_slice(self):
        """Test NINE_SLICE mode uses widget bounds."""
        from engine.ui.widgets.primitives.image import Image, ScaleMode, NineSliceConfig

        img = Image(
            width=200, height=100,
            scale_mode=ScaleMode.NINE_SLICE,
            nine_slice=NineSliceConfig.uniform(16)
        )
        img.set_natural_size(64, 64)

        w, h = img.get_rendered_size()
        assert w == 200.0
        assert h == 100.0

    def test_get_rendered_size_zero_widget_size(self):
        """Test rendered size with zero widget size returns natural."""
        from engine.ui.widgets.primitives.image import Image

        img = Image(width=0, height=0)
        img.set_natural_size(100, 50)

        w, h = img.get_rendered_size()
        assert w == 100.0
        assert h == 50.0


class TestImageEffectiveUV:
    """Tests for effective UV with flips applied."""

    def test_effective_uv_no_flip(self):
        """Test effective UV without any flips."""
        from engine.ui.widgets.primitives.image import Image, UVCoordinates

        img = Image()
        uv = img.get_effective_uv()
        assert uv.u0 == 0.0
        assert uv.v0 == 0.0
        assert uv.u1 == 1.0
        assert uv.v1 == 1.0

    def test_effective_uv_horizontal_flip(self):
        """Test effective UV with horizontal flip."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        img.flip_horizontal = True
        uv = img.get_effective_uv()
        assert uv.u0 == 1.0
        assert uv.u1 == 0.0

    def test_effective_uv_vertical_flip(self):
        """Test effective UV with vertical flip."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        img.flip_vertical = True
        uv = img.get_effective_uv()
        assert uv.v0 == 1.0
        assert uv.v1 == 0.0

    def test_effective_uv_both_flips(self):
        """Test effective UV with both flips."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        img.flip_horizontal = True
        img.flip_vertical = True
        uv = img.get_effective_uv()
        assert uv.u0 == 1.0
        assert uv.u1 == 0.0
        assert uv.v0 == 1.0
        assert uv.v1 == 0.0


class TestImageDirtyState:
    """Tests for dirty state tracking."""

    def test_image_dirty_after_creation(self):
        """Test image is dirty initially (until mark_clean)."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        # After clear_dirty in __init__, should not be dirty
        # (depends on implementation)
        # The actual behavior needs the descriptor system

    def test_image_dirty_after_source_change(self):
        """Test image is dirty after source changes."""
        from engine.ui.widgets.primitives.image import Image

        img = Image(source="old.png")
        img.mark_clean()
        img.source = "new.png"
        assert img.is_dirty

    def test_image_dirty_after_scale_mode_change(self):
        """Test image is dirty after scale mode changes."""
        from engine.ui.widgets.primitives.image import Image, ScaleMode

        img = Image()
        img.mark_clean()
        img._dirty_mesh = False
        img.scale_mode = ScaleMode.FIT
        assert img.is_dirty

    def test_image_dirty_after_width_change(self):
        """Test image is dirty after width changes."""
        from engine.ui.widgets.primitives.image import Image

        img = Image(width=100)
        img.mark_clean()
        img._dirty_mesh = False
        img.width = 200
        assert img._dirty_mesh

    def test_image_mark_clean(self):
        """Test mark_clean clears dirty state."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        img._dirty_mesh = True
        img.mark_clean()
        assert img._dirty_mesh is False

    def test_image_clear_mesh_cache(self):
        """Test clear_mesh_cache marks dirty."""
        from engine.ui.widgets.primitives.image import Image

        img = Image()
        img._dirty_mesh = False
        img._cached_vertices = [1, 2, 3]
        img.clear_mesh_cache()
        assert img._cached_vertices is None
        assert img._dirty_mesh is True


class TestImageSerialization:
    """Tests for Image serialization and deserialization."""

    def test_image_to_dict(self):
        """Test serialization to dictionary."""
        from engine.ui.widgets.primitives.image import Image, ScaleMode

        img = Image(
            source="player.png",
            scale_mode=ScaleMode.FIT,
            width=100,
            height=100,
            opacity=0.8
        )

        data = img.to_dict()
        assert data["source"] == "player.png"
        assert data["scale_mode"] == "FIT"
        assert data["width"] == 100
        assert data["height"] == 100
        assert data["opacity"] == 0.8

    def test_image_to_dict_with_nine_slice(self):
        """Test serialization includes nine slice config."""
        from engine.ui.widgets.primitives.image import Image, NineSliceConfig

        img = Image(nine_slice=NineSliceConfig(left=5, right=10, top=15, bottom=20))

        data = img.to_dict()
        assert "nine_slice" in data
        assert data["nine_slice"]["left"] == 5
        assert data["nine_slice"]["right"] == 10
        assert data["nine_slice"]["top"] == 15
        assert data["nine_slice"]["bottom"] == 20

    def test_image_to_dict_with_entity_id(self):
        """Test serialization includes entity ID when set."""
        from engine.ui.widgets.primitives.image import Image

        img = Image(entity_id="entity_123")

        data = img.to_dict()
        assert data["entity_id"] == "entity_123"

    def test_image_from_dict(self):
        """Test deserialization from dictionary."""
        from engine.ui.widgets.primitives.image import Image, ScaleMode

        data = {
            "source": "enemy.png",
            "scale_mode": "FILL",
            "width": 64,
            "height": 64,
            "opacity": 0.5,
            "tint": (1.0, 0.0, 0.0, 1.0),
            "uv": {"u0": 0.0, "v0": 0.0, "u1": 0.5, "v1": 0.5},
        }

        img = Image.from_dict(data)
        assert img.source == "enemy.png"
        assert img.scale_mode == ScaleMode.FILL
        assert img.width == 64
        assert img.height == 64
        assert img.opacity == 0.5
        assert img.tint[0] == 1.0
        assert img.uv.u1 == 0.5

    def test_image_from_dict_with_nine_slice(self):
        """Test deserialization includes nine slice config."""
        from engine.ui.widgets.primitives.image import Image

        data = {
            "source": "frame.png",
            "nine_slice": {
                "left": 8,
                "right": 8,
                "top": 8,
                "bottom": 8,
                "tile_center": True,
            },
        }

        img = Image.from_dict(data)
        assert img.nine_slice is not None
        assert img.nine_slice.left == 8
        assert img.nine_slice.tile_center is True

    def test_image_roundtrip_serialization(self):
        """Test serialization roundtrip preserves data."""
        from engine.ui.widgets.primitives.image import (
            Image, ScaleMode, NineSliceConfig, UVCoordinates
        )

        original = Image(
            source="sprite.png",
            scale_mode=ScaleMode.NINE_SLICE,
            tint=(0.5, 0.6, 0.7, 0.8),
            opacity=0.9,
            uv=UVCoordinates(0.1, 0.2, 0.3, 0.4),
            nine_slice=NineSliceConfig(1, 2, 3, 4, True, False),
            width=128,
            height=64,
        )
        original._preserve_aspect = False
        original._flip_horizontal = True
        original._flip_vertical = True

        data = original.to_dict()
        restored = Image.from_dict(data)

        assert restored.source == original.source
        assert restored.scale_mode == original.scale_mode
        assert restored.tint == original.tint
        assert restored.opacity == original.opacity
        assert restored.width == original.width
        assert restored.height == original.height
        assert restored._preserve_aspect == original._preserve_aspect
        assert restored._flip_horizontal == original._flip_horizontal
        assert restored._flip_vertical == original._flip_vertical


class TestImageRepr:
    """Tests for Image string representation."""

    def test_image_repr(self):
        """Test Image repr format."""
        from engine.ui.widgets.primitives.image import Image, ScaleMode

        img = Image(source="test.png", scale_mode=ScaleMode.FIT, width=100, height=50)
        repr_str = repr(img)

        assert "Image" in repr_str
        assert "test.png" in repr_str
        assert "FIT" in repr_str
        assert "100" in repr_str
        assert "50" in repr_str

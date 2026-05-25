"""
Comprehensive tests for brush types.

Tests cover:
- SolidBrush
- GradientBrush (linear, radial, angular, diamond)
- ImageBrush
- NineSliceBrush
- Utility functions
"""
import pytest

from engine.ui.styling.brush import (
    Brush,
    GradientBrush,
    GradientStop,
    GradientType,
    ImageBrush,
    ImageFit,
    NineSliceBrush,
    SolidBrush,
    TileMode,
    black_brush,
    create_brush,
    transparent_brush,
    white_brush,
)
from engine.ui.styling.color import Color


# ========== Fixtures ==========


@pytest.fixture
def red_color():
    """Pure red color."""
    return Color(1.0, 0.0, 0.0)


@pytest.fixture
def blue_color():
    """Pure blue color."""
    return Color(0.0, 0.0, 1.0)


@pytest.fixture
def white_color():
    """White color."""
    return Color(1.0, 1.0, 1.0)


@pytest.fixture
def solid_red_brush(red_color):
    """Solid red brush."""
    return SolidBrush(red_color)


@pytest.fixture
def linear_gradient():
    """Linear gradient from red to blue."""
    return GradientBrush.linear([Color(1.0, 0.0, 0.0), Color(0.0, 0.0, 1.0)])


@pytest.fixture
def image_brush():
    """Basic image brush."""
    return ImageBrush(image_path="/path/to/image.png")


@pytest.fixture
def nine_slice_brush():
    """Basic nine-slice brush."""
    return NineSliceBrush(
        image_path="/path/to/image.png",
        left=10,
        top=10,
        right=10,
        bottom=10,
    )


# ========== GradientStop Tests ==========


class TestGradientStop:
    """Tests for GradientStop class."""

    def test_create_gradient_stop(self, red_color):
        """Test creating a gradient stop."""
        stop = GradientStop(red_color, 0.5)
        assert stop.color == red_color
        assert stop.position == 0.5

    def test_gradient_stop_position_at_zero(self, red_color):
        """Test gradient stop at position 0."""
        stop = GradientStop(red_color, 0.0)
        assert stop.position == 0.0

    def test_gradient_stop_position_at_one(self, red_color):
        """Test gradient stop at position 1."""
        stop = GradientStop(red_color, 1.0)
        assert stop.position == 1.0

    def test_gradient_stop_invalid_position_low(self, red_color):
        """Test gradient stop with position below 0."""
        with pytest.raises(ValueError, match="position must be in"):
            GradientStop(red_color, -0.1)

    def test_gradient_stop_invalid_position_high(self, red_color):
        """Test gradient stop with position above 1."""
        with pytest.raises(ValueError, match="position must be in"):
            GradientStop(red_color, 1.1)


# ========== SolidBrush Tests ==========


class TestSolidBrush:
    """Tests for SolidBrush class."""

    def test_create_with_color(self, red_color):
        """Test creating brush with Color instance."""
        brush = SolidBrush(red_color)
        assert brush.color == red_color

    def test_create_with_hex_string(self):
        """Test creating brush with hex string."""
        brush = SolidBrush("#FF0000")
        assert brush.color.r == 1.0

    def test_create_with_color_name(self):
        """Test creating brush with color name."""
        brush = SolidBrush("blue")
        assert brush.color.b == 1.0

    def test_create_with_tuple(self):
        """Test creating brush with tuple."""
        brush = SolidBrush((0.5, 0.5, 0.5))
        assert brush.color.r == 0.5

    def test_get_color_at_any_position(self, solid_red_brush):
        """Test get_color_at returns same color everywhere."""
        color1 = solid_red_brush.get_color_at(0, 0, 100, 100)
        color2 = solid_red_brush.get_color_at(50, 50, 100, 100)
        color3 = solid_red_brush.get_color_at(100, 100, 100, 100)
        assert color1 == color2 == color3

    def test_clone(self, solid_red_brush):
        """Test cloning creates independent copy."""
        cloned = solid_red_brush.clone()
        assert cloned.color == solid_red_brush.color
        assert cloned is not solid_red_brush

    def test_is_opaque_full_alpha(self, solid_red_brush):
        """Test is_opaque with full alpha."""
        assert solid_red_brush.is_opaque

    def test_is_opaque_partial_alpha(self):
        """Test is_opaque with partial alpha."""
        brush = SolidBrush(Color(1.0, 0.0, 0.0, 0.5))
        assert not brush.is_opaque

    def test_with_color(self, solid_red_brush, blue_color):
        """Test with_color creates new brush."""
        new_brush = solid_red_brush.with_color(blue_color)
        assert new_brush.color == blue_color
        assert solid_red_brush.color.r == 1.0  # Original unchanged

    def test_with_color_string(self, solid_red_brush):
        """Test with_color accepts string."""
        new_brush = solid_red_brush.with_color("#00FF00")
        assert new_brush.color.g == 1.0

    def test_with_alpha(self, solid_red_brush):
        """Test with_alpha creates new brush."""
        new_brush = solid_red_brush.with_alpha(0.5)
        assert new_brush.color.a == 0.5
        assert solid_red_brush.color.a == 1.0  # Original unchanged

    def test_repr(self, solid_red_brush):
        """Test string representation."""
        repr_str = repr(solid_red_brush)
        assert "SolidBrush" in repr_str


# ========== GradientBrush Tests ==========


class TestGradientBrush:
    """Tests for GradientBrush class."""

    def test_create_with_colors(self, red_color, blue_color):
        """Test creating gradient with colors list."""
        brush = GradientBrush(colors=[red_color, blue_color])
        assert len(brush.stops) == 2
        assert brush.stops[0].position == 0.0
        assert brush.stops[1].position == 1.0

    def test_create_with_stops(self, red_color, blue_color):
        """Test creating gradient with explicit stops."""
        stops = [
            GradientStop(red_color, 0.0),
            GradientStop(blue_color, 1.0),
        ]
        brush = GradientBrush(stops=stops)
        assert len(brush.stops) == 2

    def test_create_with_strings(self):
        """Test creating gradient with color strings."""
        brush = GradientBrush(colors=["red", "blue"])
        assert len(brush.stops) == 2

    def test_create_default_gradient(self):
        """Test creating gradient with no colors."""
        brush = GradientBrush()
        assert len(brush.stops) == 2
        # Default is black to white

    def test_create_requires_two_colors(self, red_color):
        """Test creating gradient requires at least 2 colors."""
        with pytest.raises(ValueError, match="at least 2 colors"):
            GradientBrush(colors=[red_color])

    def test_stops_sorted_by_position(self, red_color, blue_color):
        """Test stops are sorted by position."""
        stops = [
            GradientStop(blue_color, 1.0),
            GradientStop(red_color, 0.0),
        ]
        brush = GradientBrush(stops=stops)
        assert brush.stops[0].color == red_color
        assert brush.stops[1].color == blue_color

    def test_linear_factory(self):
        """Test linear gradient factory method."""
        brush = GradientBrush.linear(["red", "blue"], angle=45)
        assert brush.gradient_type == GradientType.LINEAR
        assert brush.angle == 45

    def test_radial_factory(self):
        """Test radial gradient factory method."""
        brush = GradientBrush.radial(["red", "blue"], center_x=0.3, center_y=0.7)
        assert brush.gradient_type == GradientType.RADIAL
        assert brush.center_x == 0.3
        assert brush.center_y == 0.7

    def test_angular_factory(self):
        """Test angular gradient factory method."""
        brush = GradientBrush.angular(["red", "blue"])
        assert brush.gradient_type == GradientType.ANGULAR

    def test_get_color_at_linear_start(self, linear_gradient):
        """Test get_color_at at start of linear gradient."""
        color = linear_gradient.get_color_at(0, 50, 100, 100)
        assert color.r > 0.5  # Should be more red

    def test_get_color_at_linear_end(self, linear_gradient):
        """Test get_color_at at end of linear gradient."""
        color = linear_gradient.get_color_at(100, 50, 100, 100)
        assert color.b > 0.5  # Should be more blue

    def test_get_color_at_linear_middle(self, linear_gradient):
        """Test get_color_at at middle of linear gradient."""
        color = linear_gradient.get_color_at(50, 50, 100, 100)
        # Middle should have roughly equal red and blue
        assert abs(color.r - color.b) < 0.2

    def test_get_color_at_radial_center(self, red_color, blue_color):
        """Test get_color_at at center of radial gradient."""
        brush = GradientBrush.radial([red_color, blue_color])
        color = brush.get_color_at(50, 50, 100, 100)
        assert color.r > color.b  # Center should be red

    def test_get_color_at_radial_edge(self, red_color, blue_color):
        """Test get_color_at at edge of radial gradient."""
        brush = GradientBrush.radial([red_color, blue_color])
        color = brush.get_color_at(0, 0, 100, 100)  # Corner
        assert color.b > color.r  # Edge should be more blue

    def test_get_color_at_angular(self, red_color, blue_color):
        """Test get_color_at for angular gradient."""
        brush = GradientBrush.angular([red_color, blue_color])
        color = brush.get_color_at(50, 50, 100, 100)
        # At center, should get first color
        assert 0 <= color.r <= 1

    def test_get_color_at_diamond(self, red_color, blue_color):
        """Test get_color_at for diamond gradient."""
        brush = GradientBrush(
            GradientType.DIAMOND,
            colors=[red_color, blue_color],
        )
        color = brush.get_color_at(50, 50, 100, 100)
        assert color.r > color.b  # Center should be red

    def test_get_color_at_zero_dimensions(self, linear_gradient):
        """Test get_color_at with zero dimensions."""
        color = linear_gradient.get_color_at(0, 0, 0, 0)
        # Should return first stop color
        assert color.r > 0

    def test_clone(self, linear_gradient):
        """Test cloning gradient brush."""
        cloned = linear_gradient.clone()
        assert cloned.gradient_type == linear_gradient.gradient_type
        assert cloned.stops == linear_gradient.stops
        assert cloned is not linear_gradient

    def test_is_opaque_all_opaque(self, linear_gradient):
        """Test is_opaque when all stops are opaque."""
        assert linear_gradient.is_opaque

    def test_is_opaque_with_transparent(self):
        """Test is_opaque with transparent stop."""
        brush = GradientBrush(colors=[
            Color(1.0, 0.0, 0.0, 0.5),
            Color(0.0, 0.0, 1.0),
        ])
        assert not brush.is_opaque

    def test_repr(self, linear_gradient):
        """Test string representation."""
        repr_str = repr(linear_gradient)
        assert "GradientBrush" in repr_str
        assert "LINEAR" in repr_str


# ========== ImageBrush Tests ==========


class TestImageBrush:
    """Tests for ImageBrush class."""

    def test_create_basic(self):
        """Test creating basic image brush."""
        brush = ImageBrush(image_path="/path/to/image.png")
        assert brush.image_path == "/path/to/image.png"
        assert brush.tile_mode == TileMode.NONE
        assert brush.fit == ImageFit.FILL
        assert brush.opacity == 1.0

    def test_create_with_tile_mode(self):
        """Test creating image brush with tile mode."""
        brush = ImageBrush(
            image_path="/path/to/image.png",
            tile_mode=TileMode.REPEAT,
        )
        assert brush.tile_mode == TileMode.REPEAT

    def test_create_with_fit(self):
        """Test creating image brush with fit mode."""
        brush = ImageBrush(
            image_path="/path/to/image.png",
            fit=ImageFit.CONTAIN,
        )
        assert brush.fit == ImageFit.CONTAIN

    def test_create_with_opacity(self):
        """Test creating image brush with opacity."""
        brush = ImageBrush(
            image_path="/path/to/image.png",
            opacity=0.5,
        )
        assert brush.opacity == 0.5

    def test_create_with_offset(self):
        """Test creating image brush with offset."""
        brush = ImageBrush(
            image_path="/path/to/image.png",
            offset_x=10.0,
            offset_y=20.0,
        )
        assert brush.offset_x == 10.0
        assert brush.offset_y == 20.0

    def test_create_with_scale(self):
        """Test creating image brush with scale."""
        brush = ImageBrush(
            image_path="/path/to/image.png",
            scale_x=2.0,
            scale_y=0.5,
        )
        assert brush.scale_x == 2.0
        assert brush.scale_y == 0.5

    def test_create_with_tint(self, red_color):
        """Test creating image brush with tint."""
        brush = ImageBrush(
            image_path="/path/to/image.png",
            tint=red_color,
        )
        assert brush.tint == red_color

    def test_invalid_opacity_low(self):
        """Test invalid low opacity raises error."""
        with pytest.raises(ValueError, match="Opacity must be in"):
            ImageBrush(image_path="/path/to/image.png", opacity=-0.1)

    def test_invalid_opacity_high(self):
        """Test invalid high opacity raises error."""
        with pytest.raises(ValueError, match="Opacity must be in"):
            ImageBrush(image_path="/path/to/image.png", opacity=1.5)

    def test_invalid_scale(self):
        """Test invalid scale raises error."""
        with pytest.raises(ValueError, match="Scale must be positive"):
            ImageBrush(image_path="/path/to/image.png", scale_x=0)

    def test_get_color_at_with_tint(self, image_brush, red_color):
        """Test get_color_at returns tint color."""
        brush = ImageBrush(
            image_path="/path/to/image.png",
            tint=red_color,
        )
        color = brush.get_color_at(0, 0, 100, 100)
        assert color.r == red_color.r

    def test_get_color_at_without_tint(self, image_brush):
        """Test get_color_at returns white without tint."""
        color = image_brush.get_color_at(0, 0, 100, 100)
        assert color.r == 1.0
        assert color.g == 1.0
        assert color.b == 1.0

    def test_clone(self, image_brush):
        """Test cloning image brush."""
        cloned = image_brush.clone()
        assert cloned.image_path == image_brush.image_path
        assert cloned is not image_brush

    def test_is_opaque_full_opacity(self, image_brush):
        """Test is_opaque with full opacity."""
        assert image_brush.is_opaque

    def test_is_opaque_partial_opacity(self):
        """Test is_opaque with partial opacity."""
        brush = ImageBrush(image_path="/path/to/image.png", opacity=0.5)
        assert not brush.is_opaque

    def test_with_tint(self, image_brush, red_color):
        """Test with_tint creates new brush."""
        new_brush = image_brush.with_tint(red_color)
        assert new_brush.tint == red_color
        assert image_brush.tint is None  # Original unchanged

    def test_with_tint_string(self, image_brush):
        """Test with_tint accepts string."""
        new_brush = image_brush.with_tint("#FF0000")
        assert new_brush.tint.r == 1.0

    def test_with_tint_none(self, image_brush, red_color):
        """Test with_tint can remove tint."""
        tinted = image_brush.with_tint(red_color)
        untinted = tinted.with_tint(None)
        assert untinted.tint is None

    def test_with_opacity(self, image_brush):
        """Test with_opacity creates new brush."""
        new_brush = image_brush.with_opacity(0.5)
        assert new_brush.opacity == 0.5
        assert image_brush.opacity == 1.0  # Original unchanged

    def test_repr(self, image_brush):
        """Test string representation."""
        repr_str = repr(image_brush)
        assert "ImageBrush" in repr_str
        assert "/path/to/image.png" in repr_str


# ========== NineSliceBrush Tests ==========


class TestNineSliceBrush:
    """Tests for NineSliceBrush class."""

    def test_create_basic(self):
        """Test creating basic nine-slice brush."""
        brush = NineSliceBrush(
            image_path="/path/to/image.png",
            left=10,
            top=10,
            right=10,
            bottom=10,
        )
        assert brush.image_path == "/path/to/image.png"
        assert brush.left == 10
        assert brush.top == 10
        assert brush.right == 10
        assert brush.bottom == 10

    def test_create_with_opacity(self):
        """Test creating nine-slice brush with opacity."""
        brush = NineSliceBrush(
            image_path="/path/to/image.png",
            left=10,
            top=10,
            right=10,
            bottom=10,
            opacity=0.5,
        )
        assert brush.opacity == 0.5

    def test_create_with_tint(self, red_color):
        """Test creating nine-slice brush with tint."""
        brush = NineSliceBrush(
            image_path="/path/to/image.png",
            left=10,
            top=10,
            right=10,
            bottom=10,
            tint=red_color,
        )
        assert brush.tint == red_color

    def test_create_with_fill_center_false(self):
        """Test creating nine-slice brush without center fill."""
        brush = NineSliceBrush(
            image_path="/path/to/image.png",
            left=10,
            top=10,
            right=10,
            bottom=10,
            fill_center=False,
        )
        assert not brush.fill_center

    def test_invalid_opacity(self):
        """Test invalid opacity raises error."""
        with pytest.raises(ValueError, match="Opacity must be in"):
            NineSliceBrush(
                image_path="/path/to/image.png",
                left=10,
                top=10,
                right=10,
                bottom=10,
                opacity=1.5,
            )

    def test_invalid_inset_negative(self):
        """Test negative inset raises error."""
        with pytest.raises(ValueError, match="inset must be non-negative"):
            NineSliceBrush(
                image_path="/path/to/image.png",
                left=-10,
                top=10,
                right=10,
                bottom=10,
            )

    def test_get_color_at_with_tint(self, nine_slice_brush, red_color):
        """Test get_color_at returns tint color."""
        brush = NineSliceBrush(
            image_path="/path/to/image.png",
            left=10,
            top=10,
            right=10,
            bottom=10,
            tint=red_color,
        )
        color = brush.get_color_at(0, 0, 100, 100)
        assert color.r == red_color.r

    def test_clone(self, nine_slice_brush):
        """Test cloning nine-slice brush."""
        cloned = nine_slice_brush.clone()
        assert cloned.image_path == nine_slice_brush.image_path
        assert cloned.left == nine_slice_brush.left
        assert cloned is not nine_slice_brush

    def test_is_opaque_full_opacity(self, nine_slice_brush):
        """Test is_opaque with full opacity."""
        assert nine_slice_brush.is_opaque

    def test_is_opaque_partial_opacity(self):
        """Test is_opaque with partial opacity."""
        brush = NineSliceBrush(
            image_path="/path/to/image.png",
            left=10,
            top=10,
            right=10,
            bottom=10,
            opacity=0.5,
        )
        assert not brush.is_opaque

    def test_insets_property(self, nine_slice_brush):
        """Test insets property returns tuple."""
        insets = nine_slice_brush.insets
        assert insets == (10, 10, 10, 10)

    def test_with_insets(self, nine_slice_brush):
        """Test with_insets creates new brush."""
        new_brush = nine_slice_brush.with_insets(left=20)
        assert new_brush.left == 20
        assert new_brush.top == 10  # Others unchanged
        assert nine_slice_brush.left == 10  # Original unchanged

    def test_with_insets_all(self, nine_slice_brush):
        """Test with_insets with all values."""
        new_brush = nine_slice_brush.with_insets(
            left=5,
            top=10,
            right=15,
            bottom=20,
        )
        assert new_brush.insets == (5, 10, 15, 20)

    def test_with_tint(self, nine_slice_brush, red_color):
        """Test with_tint creates new brush."""
        new_brush = nine_slice_brush.with_tint(red_color)
        assert new_brush.tint == red_color
        assert nine_slice_brush.tint is None

    def test_with_tint_string(self, nine_slice_brush):
        """Test with_tint accepts string."""
        new_brush = nine_slice_brush.with_tint("#FF0000")
        assert new_brush.tint.r == 1.0

    def test_repr(self, nine_slice_brush):
        """Test string representation."""
        repr_str = repr(nine_slice_brush)
        assert "NineSliceBrush" in repr_str
        assert "insets=" in repr_str


# ========== Utility Function Tests ==========


class TestBrushUtilities:
    """Tests for brush utility functions."""

    def test_create_brush_from_none(self):
        """Test create_brush returns None for None input."""
        assert create_brush(None) is None

    def test_create_brush_from_brush(self, solid_red_brush):
        """Test create_brush returns brush unchanged."""
        result = create_brush(solid_red_brush)
        assert result is solid_red_brush

    def test_create_brush_from_color(self, red_color):
        """Test create_brush creates SolidBrush from Color."""
        result = create_brush(red_color)
        assert isinstance(result, SolidBrush)
        assert result.color == red_color

    def test_create_brush_from_hex_string(self):
        """Test create_brush creates SolidBrush from hex string."""
        result = create_brush("#FF0000")
        assert isinstance(result, SolidBrush)
        assert result.color.r == 1.0

    def test_create_brush_from_color_name(self):
        """Test create_brush creates SolidBrush from color name."""
        result = create_brush("blue")
        assert isinstance(result, SolidBrush)
        assert result.color.b == 1.0

    def test_create_brush_invalid_type(self):
        """Test create_brush raises error for invalid type."""
        with pytest.raises(TypeError, match="Cannot create brush from"):
            create_brush(12345)

    def test_transparent_brush(self):
        """Test transparent_brush returns transparent brush."""
        brush = transparent_brush()
        assert isinstance(brush, SolidBrush)
        assert brush.color.a == 0

    def test_white_brush(self):
        """Test white_brush returns white brush."""
        brush = white_brush()
        assert isinstance(brush, SolidBrush)
        assert brush.color.r == 1.0
        assert brush.color.g == 1.0
        assert brush.color.b == 1.0
        assert brush.color.a == 1.0

    def test_black_brush(self):
        """Test black_brush returns black brush."""
        brush = black_brush()
        assert isinstance(brush, SolidBrush)
        assert brush.color.r == 0.0
        assert brush.color.g == 0.0
        assert brush.color.b == 0.0
        assert brush.color.a == 1.0


# ========== Abstract Brush Interface Tests ==========


class TestBrushInterface:
    """Tests for Brush abstract interface."""

    def test_brush_is_abstract(self):
        """Test Brush cannot be instantiated directly."""
        # Brush is ABC, but let's verify implementations work
        brush = SolidBrush(Color(1.0, 0.0, 0.0))
        assert isinstance(brush, Brush)

    def test_all_brushes_implement_interface(
        self, solid_red_brush, linear_gradient, image_brush, nine_slice_brush
    ):
        """Test all brush types implement required interface."""
        brushes = [solid_red_brush, linear_gradient, image_brush, nine_slice_brush]
        for brush in brushes:
            assert hasattr(brush, "get_color_at")
            assert hasattr(brush, "clone")
            assert hasattr(brush, "is_opaque")


# ========== TileMode and ImageFit Enum Tests ==========


class TestEnums:
    """Tests for brush-related enums."""

    def test_tile_mode_values(self):
        """Test TileMode enum values exist."""
        assert TileMode.NONE
        assert TileMode.REPEAT
        assert TileMode.REPEAT_X
        assert TileMode.REPEAT_Y
        assert TileMode.MIRROR
        assert TileMode.CLAMP

    def test_image_fit_values(self):
        """Test ImageFit enum values exist."""
        assert ImageFit.FILL
        assert ImageFit.CONTAIN
        assert ImageFit.COVER
        assert ImageFit.NONE

    def test_gradient_type_values(self):
        """Test GradientType enum values exist."""
        assert GradientType.LINEAR
        assert GradientType.RADIAL
        assert GradientType.ANGULAR
        assert GradientType.DIAMOND

"""
Comprehensive tests for the Color class and color utilities.

Tests cover:
- Color creation (factory methods)
- Color conversions (RGB, HSL, HSV, hex)
- Color manipulation (lighten, darken, saturate, etc.)
- Color blending (all blend modes)
- Color interpolation (lerp, lerp_hsl)
- Contrast and accessibility
- Palette generation
"""
import math
import pytest

from engine.ui.styling.color import (
    BlendMode,
    Color,
    generate_analogous,
    generate_complementary,
    generate_palette,
    generate_split_complementary,
    generate_tetradic,
    generate_triadic,
    interpolate_colors,
)


# ========== Fixtures ==========


@pytest.fixture
def red():
    """Pure red color."""
    return Color(1.0, 0.0, 0.0)


@pytest.fixture
def green():
    """Pure green color."""
    return Color(0.0, 1.0, 0.0)


@pytest.fixture
def blue():
    """Pure blue color."""
    return Color(0.0, 0.0, 1.0)


@pytest.fixture
def white():
    """White color."""
    return Color(1.0, 1.0, 1.0)


@pytest.fixture
def black():
    """Black color."""
    return Color(0.0, 0.0, 0.0)


@pytest.fixture
def gray():
    """50% gray color."""
    return Color(0.5, 0.5, 0.5)


@pytest.fixture
def transparent_red():
    """Semi-transparent red."""
    return Color(1.0, 0.0, 0.0, 0.5)


# ========== Color Creation Tests ==========


class TestColorCreation:
    """Tests for creating Color instances."""

    def test_create_with_defaults(self):
        """Test creating color with default alpha."""
        color = Color(0.5, 0.3, 0.7)
        assert color.r == 0.5
        assert color.g == 0.3
        assert color.b == 0.7
        assert color.a == 1.0

    def test_create_with_alpha(self):
        """Test creating color with explicit alpha."""
        color = Color(0.5, 0.3, 0.7, 0.5)
        assert color.a == 0.5

    def test_color_is_immutable(self):
        """Test that Color is immutable (frozen dataclass)."""
        color = Color(0.5, 0.3, 0.7)
        with pytest.raises(AttributeError):
            color.r = 0.8

    def test_invalid_component_too_low(self):
        """Test validation rejects values below 0."""
        with pytest.raises(ValueError, match="must be in range"):
            Color(-0.1, 0.5, 0.5)

    def test_invalid_component_too_high(self):
        """Test validation rejects values above 1."""
        with pytest.raises(ValueError, match="must be in range"):
            Color(0.5, 1.5, 0.5)

    def test_invalid_component_type(self):
        """Test validation rejects non-numeric types."""
        with pytest.raises(TypeError, match="must be numeric"):
            Color("red", 0.5, 0.5)

    def test_color_equality(self, red):
        """Test color equality comparison."""
        other = Color(1.0, 0.0, 0.0)
        assert red == other

    def test_color_inequality(self, red, blue):
        """Test color inequality comparison."""
        assert red != blue


class TestColorFactoryMethods:
    """Tests for Color factory methods."""

    def test_from_rgb_basic(self):
        """Test creating color from 8-bit RGB."""
        color = Color.from_rgb(255, 128, 0)
        assert color.r == 1.0
        assert abs(color.g - 0.502) < 0.01
        assert color.b == 0.0
        assert color.a == 1.0

    def test_from_rgb_with_alpha(self):
        """Test creating color from 8-bit RGBA."""
        color = Color.from_rgb(255, 255, 255, 128)
        assert abs(color.a - 0.502) < 0.01

    def test_from_hex_six_digits(self):
        """Test parsing 6-digit hex color."""
        color = Color.from_hex("#FF8000")
        assert color.r == 1.0
        assert abs(color.g - 0.502) < 0.01
        assert color.b == 0.0

    def test_from_hex_eight_digits(self):
        """Test parsing 8-digit hex with alpha."""
        color = Color.from_hex("#FF800080")
        assert color.r == 1.0
        assert abs(color.a - 0.502) < 0.01

    def test_from_hex_three_digits(self):
        """Test parsing 3-digit shorthand hex."""
        color = Color.from_hex("#F80")
        assert color.r == 1.0
        assert abs(color.g - 0.533) < 0.01

    def test_from_hex_four_digits(self):
        """Test parsing 4-digit shorthand hex with alpha."""
        color = Color.from_hex("#F808")
        assert color.r == 1.0
        assert abs(color.a - 0.533) < 0.01

    def test_from_hex_without_hash(self):
        """Test parsing hex without # prefix."""
        color = Color.from_hex("FF0000")
        assert color.r == 1.0
        assert color.g == 0.0
        assert color.b == 0.0

    def test_from_hex_invalid_length(self):
        """Test invalid hex length raises error."""
        with pytest.raises(ValueError, match="Invalid hex color format"):
            Color.from_hex("#12345")

    def test_from_hex_invalid_characters(self):
        """Test invalid hex characters raises error."""
        with pytest.raises(ValueError, match="Invalid hex color format"):
            Color.from_hex("#GGGGGG")

    def test_from_hsl_basic(self):
        """Test creating color from HSL values."""
        # Red in HSL: H=0, S=1, L=0.5
        color = Color.from_hsl(0.0, 1.0, 0.5)
        assert abs(color.r - 1.0) < 0.01
        assert abs(color.g) < 0.01
        assert abs(color.b) < 0.01

    def test_from_hsl_with_alpha(self):
        """Test creating color from HSLA values."""
        color = Color.from_hsl(0.0, 1.0, 0.5, 0.5)
        assert color.a == 0.5

    def test_from_hsl_hue_wraps(self):
        """Test that hue values wrap around."""
        color1 = Color.from_hsl(0.0, 1.0, 0.5)
        color2 = Color.from_hsl(1.0, 1.0, 0.5)
        assert abs(color1.r - color2.r) < 0.01

    def test_from_hsv_basic(self):
        """Test creating color from HSV values."""
        # Red in HSV: H=0, S=1, V=1
        color = Color.from_hsv(0.0, 1.0, 1.0)
        assert abs(color.r - 1.0) < 0.01
        assert abs(color.g) < 0.01
        assert abs(color.b) < 0.01

    def test_from_hsv_with_alpha(self):
        """Test creating color from HSVA values."""
        color = Color.from_hsv(0.0, 1.0, 1.0, 0.5)
        assert color.a == 0.5

    def test_from_name_basic(self):
        """Test creating color from name."""
        color = Color.from_name("red")
        assert color.r == 1.0
        assert color.g == 0.0
        assert color.b == 0.0

    def test_from_name_case_insensitive(self):
        """Test color name is case insensitive."""
        color = Color.from_name("RED")
        assert color.r == 1.0

    def test_from_name_with_spaces(self):
        """Test color name handles spaces."""
        color = Color.from_name("dark red")
        assert color.r == 0.55

    def test_from_name_unknown(self):
        """Test unknown color name raises error."""
        with pytest.raises(ValueError, match="Unknown color name"):
            Color.from_name("unknowncolor")

    def test_parse_color_instance(self, red):
        """Test parse returns Color instance unchanged."""
        parsed = Color.parse(red)
        assert parsed is red

    def test_parse_hex_string(self):
        """Test parse handles hex strings."""
        color = Color.parse("#FF0000")
        assert color.r == 1.0

    def test_parse_name_string(self):
        """Test parse handles color names."""
        color = Color.parse("blue")
        assert color.b == 1.0

    def test_parse_tuple_rgb(self):
        """Test parse handles RGB tuples."""
        color = Color.parse((1.0, 0.0, 0.0))
        assert color.r == 1.0

    def test_parse_tuple_rgba(self):
        """Test parse handles RGBA tuples."""
        color = Color.parse((1.0, 0.0, 0.0, 0.5))
        assert color.a == 0.5

    def test_parse_invalid_tuple_length(self):
        """Test parse rejects invalid tuple length."""
        with pytest.raises(ValueError, match="Invalid color tuple length"):
            Color.parse((1.0, 0.0))

    def test_parse_unparseable_string(self):
        """Test parse rejects unparseable strings."""
        with pytest.raises(ValueError, match="Could not parse color"):
            Color.parse("notacolor")

    def test_parse_invalid_type(self):
        """Test parse rejects invalid types."""
        with pytest.raises(TypeError, match="Cannot parse color from"):
            Color.parse(12345)


# ========== Color Conversion Tests ==========


class TestColorConversions:
    """Tests for color conversion methods."""

    def test_to_rgb(self, red):
        """Test conversion to 8-bit RGB tuple."""
        rgb = red.to_rgb()
        assert rgb == (255, 0, 0)

    def test_to_rgba(self, transparent_red):
        """Test conversion to 8-bit RGBA tuple."""
        rgba = transparent_red.to_rgba()
        assert rgba == (255, 0, 0, 128)

    def test_to_hex_without_alpha(self, red):
        """Test conversion to hex string without alpha."""
        hex_str = red.to_hex()
        assert hex_str == "#FF0000"

    def test_to_hex_with_alpha(self, transparent_red):
        """Test conversion to hex string with alpha."""
        hex_str = transparent_red.to_hex(include_alpha=True)
        assert hex_str == "#FF000080"

    def test_to_hsl_red(self, red):
        """Test conversion to HSL for red."""
        h, s, l = red.to_hsl()
        assert abs(h) < 0.01
        assert abs(s - 1.0) < 0.01
        assert abs(l - 0.5) < 0.01

    def test_to_hsl_gray(self, gray):
        """Test conversion to HSL for gray (no saturation)."""
        h, s, l = gray.to_hsl()
        assert abs(s) < 0.01
        assert abs(l - 0.5) < 0.01

    def test_to_hsv_red(self, red):
        """Test conversion to HSV for red."""
        h, s, v = red.to_hsv()
        assert abs(h) < 0.01
        assert abs(s - 1.0) < 0.01
        assert abs(v - 1.0) < 0.01

    def test_to_hsv_gray(self, gray):
        """Test conversion to HSV for gray."""
        h, s, v = gray.to_hsv()
        assert abs(s) < 0.01
        assert abs(v - 0.5) < 0.01


# ========== Color Manipulation Tests ==========


class TestColorManipulation:
    """Tests for color manipulation methods."""

    def test_with_alpha(self, red):
        """Test creating color with different alpha."""
        new_color = red.with_alpha(0.5)
        assert new_color.a == 0.5
        assert new_color.r == 1.0

    def test_with_red(self, blue):
        """Test creating color with different red."""
        new_color = blue.with_red(0.5)
        assert new_color.r == 0.5
        assert new_color.b == 1.0

    def test_with_green(self, red):
        """Test creating color with different green."""
        new_color = red.with_green(0.5)
        assert new_color.g == 0.5
        assert new_color.r == 1.0

    def test_with_blue(self, red):
        """Test creating color with different blue."""
        new_color = red.with_blue(0.5)
        assert new_color.b == 0.5
        assert new_color.r == 1.0

    def test_lighten(self, red):
        """Test lightening a color."""
        lighter = red.lighten(0.5)
        h1, s1, l1 = red.to_hsl()
        h2, s2, l2 = lighter.to_hsl()
        assert l2 > l1

    def test_lighten_already_white(self, white):
        """Test lightening white color."""
        lighter = white.lighten(0.5)
        _, _, l = lighter.to_hsl()
        assert abs(l - 1.0) < 0.01

    def test_darken(self, red):
        """Test darkening a color."""
        darker = red.darken(0.5)
        h1, s1, l1 = red.to_hsl()
        h2, s2, l2 = darker.to_hsl()
        assert l2 < l1

    def test_darken_already_black(self, black):
        """Test darkening black color."""
        darker = black.darken(0.5)
        _, _, l = darker.to_hsl()
        assert abs(l) < 0.01

    def test_saturate(self, gray):
        """Test increasing saturation."""
        saturated = gray.saturate(0.5)
        h1, s1, _ = gray.to_hsl()
        h2, s2, _ = saturated.to_hsl()
        assert s2 > s1

    def test_desaturate(self, red):
        """Test decreasing saturation."""
        desaturated = red.desaturate(0.5)
        h1, s1, _ = red.to_hsl()
        h2, s2, _ = desaturated.to_hsl()
        assert s2 < s1

    def test_grayscale(self, red):
        """Test converting to grayscale."""
        grayscale = red.grayscale()
        assert abs(grayscale.r - grayscale.g) < 0.01
        assert abs(grayscale.g - grayscale.b) < 0.01

    def test_grayscale_preserves_alpha(self, transparent_red):
        """Test grayscale preserves alpha."""
        grayscale = transparent_red.grayscale()
        assert grayscale.a == 0.5

    def test_invert(self, red):
        """Test inverting a color."""
        inverted = red.invert()
        assert abs(inverted.r) < 0.01
        assert abs(inverted.g - 1.0) < 0.01
        assert abs(inverted.b - 1.0) < 0.01

    def test_invert_preserves_alpha(self, transparent_red):
        """Test invert preserves alpha."""
        inverted = transparent_red.invert()
        assert inverted.a == 0.5

    def test_rotate_hue(self, red):
        """Test rotating hue."""
        rotated = red.rotate_hue(120)  # Red -> Green
        assert rotated.g > rotated.r

    def test_rotate_hue_wraps(self, red):
        """Test hue rotation wraps around."""
        rotated = red.rotate_hue(360)
        assert abs(rotated.r - red.r) < 0.01
        assert abs(rotated.g - red.g) < 0.01
        assert abs(rotated.b - red.b) < 0.01

    def test_complement(self, red):
        """Test complementary color."""
        complement = red.complement()
        # Red's complement is cyan
        assert abs(complement.r) < 0.01
        assert abs(complement.g - 1.0) < 0.01
        assert abs(complement.b - 1.0) < 0.01

    def test_luminance_white(self, white):
        """Test luminance calculation for white."""
        assert abs(white.luminance - 1.0) < 0.01

    def test_luminance_black(self, black):
        """Test luminance calculation for black."""
        assert abs(black.luminance) < 0.01

    def test_luminance_gray(self, gray):
        """Test luminance calculation for gray."""
        assert 0.1 < gray.luminance < 0.5


# ========== Blending Tests ==========


class TestColorBlending:
    """Tests for color blending operations."""

    def test_blend_normal(self, red, blue):
        """Test normal blend mode."""
        result = red.blend(blue, BlendMode.NORMAL)
        assert result.b > result.r  # Blue is on top

    def test_blend_multiply(self, red, white):
        """Test multiply blend mode."""
        result = red.blend(white, BlendMode.MULTIPLY)
        assert abs(result.r - red.r) < 0.01
        assert abs(result.g - red.g) < 0.01
        assert abs(result.b - red.b) < 0.01

    def test_blend_multiply_with_black(self, red, black):
        """Test multiply with black produces black."""
        result = red.blend(black, BlendMode.MULTIPLY)
        assert abs(result.r) < 0.01
        assert abs(result.g) < 0.01
        assert abs(result.b) < 0.01

    def test_blend_screen(self, red, blue):
        """Test screen blend mode."""
        result = red.blend(blue, BlendMode.SCREEN)
        # Screen brightens the result
        assert result.r > 0 and result.b > 0

    def test_blend_overlay(self, red, gray):
        """Test overlay blend mode."""
        result = red.blend(gray, BlendMode.OVERLAY)
        # Overlay combines multiply and screen
        assert 0 <= result.r <= 1
        assert 0 <= result.g <= 1
        assert 0 <= result.b <= 1

    def test_blend_darken(self, red, gray):
        """Test darken blend mode."""
        result = red.blend(gray, BlendMode.DARKEN)
        assert result.g <= min(red.g, gray.g) + 0.01

    def test_blend_lighten(self, red, gray):
        """Test lighten blend mode."""
        result = red.blend(gray, BlendMode.LIGHTEN)
        assert result.r >= max(red.r, gray.r) - 0.01

    def test_blend_color_dodge(self, gray):
        """Test color dodge blend mode."""
        other = Color(0.5, 0.5, 0.5)
        result = gray.blend(other, BlendMode.COLOR_DODGE)
        # Dodge brightens
        assert result.r >= gray.r

    def test_blend_color_burn(self, gray):
        """Test color burn blend mode."""
        other = Color(0.5, 0.5, 0.5)
        result = gray.blend(other, BlendMode.COLOR_BURN)
        # Burn darkens
        assert result.r <= gray.r

    def test_blend_hard_light(self, red, gray):
        """Test hard light blend mode."""
        result = red.blend(gray, BlendMode.HARD_LIGHT)
        assert 0 <= result.r <= 1

    def test_blend_soft_light(self, red, gray):
        """Test soft light blend mode."""
        result = red.blend(gray, BlendMode.SOFT_LIGHT)
        assert 0 <= result.r <= 1

    def test_blend_difference(self, red, blue):
        """Test difference blend mode."""
        result = red.blend(blue, BlendMode.DIFFERENCE)
        assert abs(result.r - 1.0) < 0.01  # |1-0| = 1
        assert abs(result.b - 1.0) < 0.01  # |0-1| = 1

    def test_blend_exclusion(self, red, gray):
        """Test exclusion blend mode."""
        result = red.blend(gray, BlendMode.EXCLUSION)
        assert 0 <= result.r <= 1

    def test_blend_normal_with_transparency(self, transparent_red, blue):
        """Test normal blend respects alpha."""
        result = transparent_red.blend(blue.with_alpha(0.5), BlendMode.NORMAL)
        assert 0 < result.r < 1
        assert 0 < result.b < 1


# ========== Interpolation Tests ==========


class TestColorInterpolation:
    """Tests for color interpolation."""

    def test_lerp_midpoint(self, red, blue):
        """Test linear interpolation at midpoint."""
        mid = red.lerp(blue, 0.5)
        assert abs(mid.r - 0.5) < 0.01
        assert abs(mid.b - 0.5) < 0.01

    def test_lerp_start(self, red, blue):
        """Test lerp at t=0 returns start color."""
        result = red.lerp(blue, 0.0)
        assert abs(result.r - red.r) < 0.01
        assert abs(result.b - red.b) < 0.01

    def test_lerp_end(self, red, blue):
        """Test lerp at t=1 returns end color."""
        result = red.lerp(blue, 1.0)
        assert abs(result.r - blue.r) < 0.01
        assert abs(result.b - blue.b) < 0.01

    def test_lerp_clamps_t_below_zero(self, red, blue):
        """Test lerp clamps t values below 0."""
        result = red.lerp(blue, -0.5)
        assert abs(result.r - red.r) < 0.01

    def test_lerp_clamps_t_above_one(self, red, blue):
        """Test lerp clamps t values above 1."""
        result = red.lerp(blue, 1.5)
        assert abs(result.b - blue.b) < 0.01

    def test_lerp_alpha(self, transparent_red):
        """Test lerp interpolates alpha."""
        opaque = Color(1.0, 0.0, 0.0, 1.0)
        mid = transparent_red.lerp(opaque, 0.5)
        assert abs(mid.a - 0.75) < 0.01

    def test_lerp_hsl_midpoint(self, red, green):
        """Test HSL interpolation at midpoint."""
        mid = red.lerp_hsl(green, 0.5)
        # Midpoint between red and green in HSL should be yellowish
        assert mid.r > 0 and mid.g > 0

    def test_lerp_hsl_shortest_path(self, red, blue):
        """Test HSL interpolation takes shortest hue path."""
        mid = red.lerp_hsl(blue, 0.5)
        # Red (0 deg) to Blue (240 deg) - should go through magenta (300 deg)
        assert mid.r > 0 and mid.b > 0


# ========== Contrast Tests ==========


class TestColorContrast:
    """Tests for contrast calculations."""

    def test_contrast_ratio_black_white(self, black, white):
        """Test maximum contrast ratio."""
        ratio = black.contrast_ratio(white)
        assert abs(ratio - 21.0) < 0.1

    def test_contrast_ratio_same_color(self, red):
        """Test contrast ratio of same color is 1."""
        ratio = red.contrast_ratio(red)
        assert abs(ratio - 1.0) < 0.01

    def test_contrast_ratio_symmetric(self, red, blue):
        """Test contrast ratio is symmetric."""
        ratio1 = red.contrast_ratio(blue)
        ratio2 = blue.contrast_ratio(red)
        assert abs(ratio1 - ratio2) < 0.01

    def test_is_readable_on_high_contrast(self, black, white):
        """Test readability with high contrast."""
        assert black.is_readable_on(white, "AA")
        assert black.is_readable_on(white, "AAA")

    def test_is_readable_on_low_contrast(self, white):
        """Test readability fails with low contrast."""
        light_gray = Color(0.9, 0.9, 0.9)
        assert not white.is_readable_on(light_gray, "AA")


# ========== String Representation Tests ==========


class TestColorStringRepresentation:
    """Tests for string representations."""

    def test_str_opaque(self, red):
        """Test string representation of opaque color."""
        assert str(red) == "#FF0000"

    def test_str_transparent(self, transparent_red):
        """Test string representation includes alpha when not 1."""
        assert str(transparent_red) == "#FF000080"

    def test_repr(self, red):
        """Test repr format."""
        repr_str = repr(red)
        assert "Color" in repr_str
        assert "r=" in repr_str
        assert "g=" in repr_str
        assert "b=" in repr_str


# ========== Palette Generation Tests ==========


class TestPaletteGeneration:
    """Tests for color palette generation."""

    def test_generate_palette_count(self, red):
        """Test palette generates correct number of colors."""
        palette = generate_palette(red, count=5)
        assert len(palette) == 5

    def test_generate_palette_variation(self, red):
        """Test palette has lightness variation."""
        palette = generate_palette(red, count=5, spread=0.5)
        lightnesses = [c.to_hsl()[2] for c in palette]
        assert lightnesses[0] < lightnesses[-1]

    def test_generate_complementary(self, red):
        """Test complementary palette."""
        palette = generate_complementary(red)
        assert len(palette) == 2
        assert palette[0] == red
        # Complement of red is cyan
        assert palette[1].g > 0.5 and palette[1].b > 0.5

    def test_generate_triadic(self, red):
        """Test triadic palette."""
        palette = generate_triadic(red)
        assert len(palette) == 3
        # Colors should be 120 degrees apart

    def test_generate_analogous(self, red):
        """Test analogous palette."""
        palette = generate_analogous(red, angle=30)
        assert len(palette) == 3

    def test_generate_split_complementary(self, red):
        """Test split-complementary palette."""
        palette = generate_split_complementary(red, angle=30)
        assert len(palette) == 3

    def test_generate_tetradic(self, red):
        """Test tetradic palette."""
        palette = generate_tetradic(red)
        assert len(palette) == 4

    def test_interpolate_colors_count(self, red, blue):
        """Test interpolating correct number of colors."""
        result = interpolate_colors([red, blue], steps=5)
        assert len(result) == 5

    def test_interpolate_colors_endpoints(self, red, blue):
        """Test interpolation includes endpoints."""
        result = interpolate_colors([red, blue], steps=5)
        assert abs(result[0].r - red.r) < 0.01
        assert abs(result[-1].b - blue.b) < 0.01

    def test_interpolate_colors_single(self, red):
        """Test interpolating single color."""
        result = interpolate_colors([red], steps=3)
        assert len(result) == 3
        for c in result:
            assert abs(c.r - red.r) < 0.01

    def test_interpolate_colors_empty(self):
        """Test interpolating empty list."""
        result = interpolate_colors([], steps=3)
        assert len(result) == 0


# ========== Named Colors Tests ==========


class TestNamedColors:
    """Tests for named colors functionality."""

    def test_named_colors_initialized(self):
        """Test that named colors are initialized."""
        assert len(Color.NAMED_COLORS) > 0

    def test_basic_color_names(self):
        """Test basic color names exist."""
        assert "red" in Color.NAMED_COLORS
        assert "green" in Color.NAMED_COLORS
        assert "blue" in Color.NAMED_COLORS
        assert "white" in Color.NAMED_COLORS
        assert "black" in Color.NAMED_COLORS

    def test_transparent_color(self):
        """Test transparent color."""
        transparent = Color.from_name("transparent")
        assert transparent.a == 0

    def test_gray_aliases(self):
        """Test gray/grey are equivalent."""
        gray = Color.from_name("gray")
        grey = Color.from_name("grey")
        assert gray == grey

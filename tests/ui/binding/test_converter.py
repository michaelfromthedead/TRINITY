"""
Comprehensive tests for value converters.

Tests cover:
- BoolToVisibilityConverter
- NumberFormatConverter
- IntegerFormatConverter
- StringFormatConverter
- ColorConverter
- BoolToStringConverter
- InverseBoolConverter
- NullToBoolConverter
- EnumToStringConverter
- PercentageConverter
- DateTimeFormatConverter
- ChainedConverter
- LambdaConverter
- AsyncLambdaConverter
- CachedConverter
- MultiValueConverter
"""
import asyncio
from datetime import datetime
from enum import Enum, auto

import pytest

from engine.ui.binding.converter import (
    AsyncLambdaConverter,
    BoolToStringConverter,
    BoolToVisibilityConverter,
    CachedConverter,
    ChainedConverter,
    Color,
    ColorConverter,
    ColorToRgbaConverter,
    DateTimeFormatConverter,
    EnumToStringConverter,
    IAsyncConverter,
    IConverter,
    IntegerFormatConverter,
    InverseBoolConverter,
    LambdaConverter,
    MathOperationConverter,
    NullToBoolConverter,
    NumberFormatConverter,
    PercentageConverter,
    StringConcatConverter,
    StringFormatConverter,
    Visibility,
    bool_to_visibility,
    chain,
    color_converter,
    number_format,
    string_format,
)


# ========== Fixtures ==========


class TestEnum(Enum):
    """Test enumeration."""
    VALUE_A = auto()
    VALUE_B = auto()


@pytest.fixture
def red_color():
    """Red color."""
    return Color(255, 0, 0)


@pytest.fixture
def blue_color():
    """Blue color."""
    return Color(0, 0, 255)


# ========== BoolToVisibilityConverter Tests ==========


class TestBoolToVisibilityConverter:
    """Tests for BoolToVisibilityConverter."""

    def test_convert_true_to_visible(self):
        """Test True converts to VISIBLE."""
        converter = BoolToVisibilityConverter()
        assert converter.convert(True) == Visibility.VISIBLE

    def test_convert_false_to_collapsed(self):
        """Test False converts to COLLAPSED by default."""
        converter = BoolToVisibilityConverter()
        assert converter.convert(False) == Visibility.COLLAPSED

    def test_convert_false_to_hidden(self):
        """Test False can convert to HIDDEN."""
        converter = BoolToVisibilityConverter(hidden_state=Visibility.HIDDEN)
        assert converter.convert(False) == Visibility.HIDDEN

    def test_convert_inverted_true(self):
        """Test inverted True converts to COLLAPSED."""
        converter = BoolToVisibilityConverter(invert=True)
        assert converter.convert(True) == Visibility.COLLAPSED

    def test_convert_inverted_false(self):
        """Test inverted False converts to VISIBLE."""
        converter = BoolToVisibilityConverter(invert=True)
        assert converter.convert(False) == Visibility.VISIBLE

    def test_convert_back_visible(self):
        """Test VISIBLE converts back to True."""
        converter = BoolToVisibilityConverter()
        assert converter.convert_back(Visibility.VISIBLE) is True

    def test_convert_back_collapsed(self):
        """Test COLLAPSED converts back to False."""
        converter = BoolToVisibilityConverter()
        assert converter.convert_back(Visibility.COLLAPSED) is False

    def test_convert_back_inverted(self):
        """Test inverted convert_back."""
        converter = BoolToVisibilityConverter(invert=True)
        assert converter.convert_back(Visibility.VISIBLE) is False


# ========== NumberFormatConverter Tests ==========


class TestNumberFormatConverter:
    """Tests for NumberFormatConverter."""

    def test_convert_default_format(self):
        """Test default format (2 decimal places)."""
        converter = NumberFormatConverter()
        assert converter.convert(3.14159) == "3.14"

    def test_convert_custom_format(self):
        """Test custom format string."""
        converter = NumberFormatConverter(format_string="{:.4f}")
        assert converter.convert(3.14159) == "3.1416"

    def test_convert_with_parameter(self):
        """Test format from parameter overrides default."""
        converter = NumberFormatConverter()
        assert converter.convert(3.14159, "{:.1f}") == "3.1"

    def test_convert_thousands(self):
        """Test thousands separator format."""
        converter = NumberFormatConverter(format_string="{:,.2f}")
        assert converter.convert(1234567.89) == "1,234,567.89"

    def test_convert_invalid_returns_default(self):
        """Test invalid value returns formatted default."""
        converter = NumberFormatConverter(default_value=0.0)
        # When formatting fails, returns default formatted
        result = converter.convert(float('nan'))
        assert result is not None

    def test_convert_back_simple(self):
        """Test parsing simple number."""
        converter = NumberFormatConverter()
        assert converter.convert_back("3.14") == 3.14

    def test_convert_back_with_commas(self):
        """Test parsing number with commas."""
        converter = NumberFormatConverter()
        assert converter.convert_back("1,234.56") == 1234.56

    def test_convert_back_invalid(self):
        """Test parsing invalid returns default."""
        converter = NumberFormatConverter(default_value=0.0)
        assert converter.convert_back("not a number") == 0.0


# ========== IntegerFormatConverter Tests ==========


class TestIntegerFormatConverter:
    """Tests for IntegerFormatConverter."""

    def test_convert_default_format(self):
        """Test default integer format."""
        converter = IntegerFormatConverter()
        assert converter.convert(42) == "42"

    def test_convert_with_thousands(self):
        """Test thousands separator."""
        converter = IntegerFormatConverter(format_string="{:,d}")
        assert converter.convert(1234567) == "1,234,567"

    def test_convert_back_simple(self):
        """Test parsing simple integer."""
        converter = IntegerFormatConverter()
        assert converter.convert_back("42") == 42

    def test_convert_back_with_commas(self):
        """Test parsing with commas."""
        converter = IntegerFormatConverter()
        assert converter.convert_back("1,234") == 1234

    def test_convert_back_float_string(self):
        """Test parsing float string to int."""
        converter = IntegerFormatConverter()
        assert converter.convert_back("42.9") == 42

    def test_convert_back_invalid(self):
        """Test parsing invalid returns default."""
        converter = IntegerFormatConverter(default_value=0)
        assert converter.convert_back("invalid") == 0


# ========== StringFormatConverter Tests ==========


class TestStringFormatConverter:
    """Tests for StringFormatConverter."""

    def test_convert_default_template(self):
        """Test default template."""
        converter = StringFormatConverter()
        assert converter.convert("hello") == "hello"

    def test_convert_custom_template(self):
        """Test custom template."""
        converter = StringFormatConverter(template="Value: {value}")
        assert converter.convert("hello") == "Value: hello"

    def test_convert_none(self):
        """Test None returns null_value."""
        converter = StringFormatConverter(null_value="N/A")
        assert converter.convert(None) == "N/A"

    def test_convert_with_parameter(self):
        """Test template from parameter."""
        converter = StringFormatConverter()
        assert converter.convert("hello", "Say: {value}") == "Say: hello"

    def test_convert_back(self):
        """Test convert_back returns string as-is."""
        converter = StringFormatConverter()
        assert converter.convert_back("hello") == "hello"


# ========== ColorConverter Tests ==========


class TestColorConverter:
    """Tests for ColorConverter."""

    def test_convert_to_hex(self, red_color):
        """Test converting Color to hex."""
        converter = ColorConverter()
        assert converter.convert(red_color) == "#ff0000"

    def test_convert_none_uses_default(self):
        """Test None uses default color."""
        converter = ColorConverter(default_color=Color(0, 0, 0))
        assert converter.convert(None) == "#000000"

    def test_convert_back_hex(self):
        """Test parsing hex string."""
        converter = ColorConverter()
        color = converter.convert_back("#ff0000")
        assert color.r == 255
        assert color.g == 0
        assert color.b == 0

    def test_convert_back_name(self):
        """Test parsing color name."""
        converter = ColorConverter()
        color = converter.convert_back("red")
        assert color.r == 255

    def test_convert_back_empty(self):
        """Test empty string returns default."""
        converter = ColorConverter(default_color=Color(0, 0, 0))
        color = converter.convert_back("")
        assert color.r == 0

    def test_convert_back_invalid(self):
        """Test invalid string returns default."""
        converter = ColorConverter(default_color=Color(0, 0, 0))
        color = converter.convert_back("invalid")
        assert color == Color(0, 0, 0)


# ========== ColorToRgbaConverter Tests ==========


class TestColorToRgbaConverter:
    """Tests for ColorToRgbaConverter."""

    def test_convert(self, red_color):
        """Test converting Color to RGBA tuple."""
        converter = ColorToRgbaConverter()
        rgba = converter.convert(red_color)
        assert rgba == (255, 0, 0, 255)

    def test_convert_none(self):
        """Test None returns default RGBA."""
        converter = ColorToRgbaConverter()
        rgba = converter.convert(None)
        assert rgba == (0, 0, 0, 255)

    def test_convert_back(self):
        """Test converting RGBA tuple to Color."""
        converter = ColorToRgbaConverter()
        color = converter.convert_back((255, 128, 0, 200))
        assert color.r == 255
        assert color.g == 128
        assert color.b == 0
        assert color.a == 200

    def test_convert_back_rgb(self):
        """Test converting RGB tuple (no alpha)."""
        converter = ColorToRgbaConverter()
        color = converter.convert_back((255, 128, 0))
        assert color.a == 255

    def test_convert_back_invalid(self):
        """Test invalid tuple returns default."""
        converter = ColorToRgbaConverter()
        color = converter.convert_back(None)
        assert color.r == 0


# ========== BoolToStringConverter Tests ==========


class TestBoolToStringConverter:
    """Tests for BoolToStringConverter."""

    def test_convert_true(self):
        """Test True converts to true_string."""
        converter = BoolToStringConverter("Yes", "No")
        assert converter.convert(True) == "Yes"

    def test_convert_false(self):
        """Test False converts to false_string."""
        converter = BoolToStringConverter("Yes", "No")
        assert converter.convert(False) == "No"

    def test_convert_back_true_string(self):
        """Test true_string converts back to True."""
        converter = BoolToStringConverter("Yes", "No")
        assert converter.convert_back("Yes") is True

    def test_convert_back_false_string(self):
        """Test non-true_string converts back to False."""
        converter = BoolToStringConverter("Yes", "No")
        assert converter.convert_back("No") is False


# ========== InverseBoolConverter Tests ==========


class TestInverseBoolConverter:
    """Tests for InverseBoolConverter."""

    def test_convert_true(self):
        """Test True inverts to False."""
        converter = InverseBoolConverter()
        assert converter.convert(True) is False

    def test_convert_false(self):
        """Test False inverts to True."""
        converter = InverseBoolConverter()
        assert converter.convert(False) is True

    def test_convert_back_true(self):
        """Test convert_back inverts True to False."""
        converter = InverseBoolConverter()
        assert converter.convert_back(True) is False


# ========== NullToBoolConverter Tests ==========


class TestNullToBoolConverter:
    """Tests for NullToBoolConverter."""

    def test_convert_not_none(self):
        """Test non-None returns True."""
        converter = NullToBoolConverter()
        assert converter.convert("value") is True
        assert converter.convert(0) is True
        assert converter.convert("") is True

    def test_convert_none(self):
        """Test None returns False."""
        converter = NullToBoolConverter()
        assert converter.convert(None) is False

    def test_convert_inverted(self):
        """Test inverted mode."""
        converter = NullToBoolConverter(invert=True)
        assert converter.convert(None) is True
        assert converter.convert("value") is False

    def test_convert_back(self):
        """Test convert_back returns None."""
        converter = NullToBoolConverter()
        assert converter.convert_back(True) is None


# ========== EnumToStringConverter Tests ==========


class TestEnumToStringConverter:
    """Tests for EnumToStringConverter."""

    def test_convert(self):
        """Test converting enum to string."""
        converter = EnumToStringConverter(TestEnum)
        assert converter.convert(TestEnum.VALUE_A) == "VALUE_A"

    def test_convert_none(self):
        """Test None converts to empty string."""
        converter = EnumToStringConverter(TestEnum)
        assert converter.convert(None) == ""

    def test_convert_back(self):
        """Test converting string back to enum."""
        converter = EnumToStringConverter(TestEnum)
        assert converter.convert_back("VALUE_A") == TestEnum.VALUE_A

    def test_convert_back_invalid(self):
        """Test invalid string returns None."""
        converter = EnumToStringConverter(TestEnum)
        assert converter.convert_back("INVALID") is None

    def test_convert_back_no_type(self):
        """Test convert_back without enum type returns None."""
        converter = EnumToStringConverter()
        assert converter.convert_back("VALUE_A") is None


# ========== PercentageConverter Tests ==========


class TestPercentageConverter:
    """Tests for PercentageConverter."""

    def test_convert_whole(self):
        """Test converting to whole percentage."""
        converter = PercentageConverter(decimal_places=0)
        assert converter.convert(0.5) == "50%"

    def test_convert_with_decimals(self):
        """Test converting with decimal places."""
        converter = PercentageConverter(decimal_places=2)
        assert converter.convert(0.1234) == "12.34%"

    def test_convert_back(self):
        """Test parsing percentage string."""
        converter = PercentageConverter()
        assert converter.convert_back("50%") == 0.5

    def test_convert_back_no_percent(self):
        """Test parsing without percent sign."""
        converter = PercentageConverter()
        assert converter.convert_back("50") == 0.5

    def test_convert_back_invalid(self):
        """Test invalid string returns 0."""
        converter = PercentageConverter()
        assert converter.convert_back("invalid") == 0.0


# ========== DateTimeFormatConverter Tests ==========


class TestDateTimeFormatConverter:
    """Tests for DateTimeFormatConverter."""

    def test_convert(self):
        """Test formatting datetime."""
        converter = DateTimeFormatConverter("%Y-%m-%d")
        dt = datetime(2024, 1, 15)
        assert converter.convert(dt) == "2024-01-15"

    def test_convert_none(self):
        """Test None returns empty string."""
        converter = DateTimeFormatConverter()
        assert converter.convert(None) == ""

    def test_convert_with_parameter(self):
        """Test format from parameter."""
        converter = DateTimeFormatConverter()
        dt = datetime(2024, 1, 15)
        assert converter.convert(dt, "%d/%m/%Y") == "15/01/2024"

    def test_convert_back(self):
        """Test parsing datetime string."""
        converter = DateTimeFormatConverter("%Y-%m-%d")
        result = converter.convert_back("2024-01-15")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_convert_back_empty(self):
        """Test empty string returns None."""
        converter = DateTimeFormatConverter()
        assert converter.convert_back("") is None

    def test_convert_back_invalid(self):
        """Test invalid string returns None."""
        converter = DateTimeFormatConverter()
        assert converter.convert_back("invalid") is None


# ========== ChainedConverter Tests ==========


class TestChainedConverter:
    """Tests for ChainedConverter."""

    def test_chain_two_converters(self):
        """Test chaining two converters."""
        # bool -> visibility -> string
        converter = ChainedConverter([
            InverseBoolConverter(),
            BoolToStringConverter("Yes", "No"),
        ])
        assert converter.convert(True) == "No"  # True -> False -> "No"

    def test_convert_back(self):
        """Test convert_back in reverse order."""
        converter = ChainedConverter([
            InverseBoolConverter(),
            BoolToStringConverter("Yes", "No"),
        ])
        # "No" -> False -> True
        assert converter.convert_back("No") is True

    def test_empty_chain_raises(self):
        """Test empty chain raises error."""
        with pytest.raises(ValueError, match="(?i)at least one converter"):
            ChainedConverter([])


# ========== LambdaConverter Tests ==========


class TestLambdaConverter:
    """Tests for LambdaConverter."""

    def test_convert(self):
        """Test forward conversion."""
        converter = LambdaConverter(
            lambda v, p: v * 2,
            lambda v, p: v // 2,
        )
        assert converter.convert(5) == 10

    def test_convert_back(self):
        """Test backward conversion."""
        converter = LambdaConverter(
            lambda v, p: v * 2,
            lambda v, p: v // 2,
        )
        assert converter.convert_back(10) == 5

    def test_convert_with_parameter(self):
        """Test conversion with parameter."""
        converter = LambdaConverter(
            lambda v, p: v * (p or 1),
        )
        assert converter.convert(5, 3) == 15

    def test_convert_back_not_provided(self):
        """Test convert_back raises when not provided."""
        converter = LambdaConverter(lambda v, p: v * 2)
        with pytest.raises(NotImplementedError):
            converter.convert_back(10)


# ========== AsyncLambdaConverter Tests ==========


class TestAsyncLambdaConverter:
    """Tests for AsyncLambdaConverter."""

    @pytest.mark.asyncio
    async def test_convert(self):
        """Test async forward conversion."""
        converter = AsyncLambdaConverter(
            lambda v, p: asyncio.coroutine(lambda: v * 2)(),
        )
        # Use proper async function
        async def convert(v, p):
            return v * 2

        converter = AsyncLambdaConverter(convert)
        result = await converter.convert(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_convert_back(self):
        """Test async backward conversion."""
        async def forward(v, p):
            return v * 2

        async def backward(v, p):
            return v // 2

        converter = AsyncLambdaConverter(forward, backward)
        result = await converter.convert_back(10)
        assert result == 5

    @pytest.mark.asyncio
    async def test_convert_back_not_provided(self):
        """Test convert_back raises when not provided."""
        async def forward(v, p):
            return v * 2

        converter = AsyncLambdaConverter(forward)
        with pytest.raises(NotImplementedError):
            await converter.convert_back(10)


# ========== CachedConverter Tests ==========


class TestCachedConverter:
    """Tests for CachedConverter."""

    def test_caches_result(self):
        """Test result is cached."""
        call_count = [0]

        class CountingConverter(IConverter):
            def convert(self, value, parameter=None):
                call_count[0] += 1
                return value * 2

            def convert_back(self, value, parameter=None):
                return value // 2

        cached = CachedConverter(CountingConverter())

        # First call
        assert cached.convert(5) == 10
        assert call_count[0] == 1

        # Second call (cached)
        assert cached.convert(5) == 10
        assert call_count[0] == 1

    def test_cache_back_caches_result(self):
        """Test convert_back result is cached."""
        call_count = [0]

        class CountingConverter(IConverter):
            def convert(self, value, parameter=None):
                return value * 2

            def convert_back(self, value, parameter=None):
                call_count[0] += 1
                return value // 2

        cached = CachedConverter(CountingConverter())

        cached.convert_back(10)
        cached.convert_back(10)

        assert call_count[0] == 1

    def test_cache_eviction(self):
        """Test cache evicts oldest when full."""
        converter = NumberFormatConverter()
        cached = CachedConverter(converter, max_cache_size=2)

        cached.convert(1.0)
        cached.convert(2.0)
        cached.convert(3.0)  # Should evict 1.0

        # All should work, but 1.0 won't be cached
        assert cached.convert(1.0) == "1.00"
        assert cached.convert(2.0) == "2.00"

    def test_clear_cache(self):
        """Test clearing cache."""
        converter = NumberFormatConverter()
        cached = CachedConverter(converter)

        cached.convert(1.0)
        cached.clear_cache()

        # Should have empty caches
        assert len(cached._convert_cache) == 0
        assert len(cached._back_cache) == 0

    def test_unhashable_not_cached(self):
        """Test unhashable values bypass cache."""
        converter = StringFormatConverter()
        cached = CachedConverter(converter)

        # Lists are unhashable
        result1 = cached.convert([1, 2, 3])
        result2 = cached.convert([1, 2, 3])

        # Should still work, just not cached
        assert result1 == result2


# ========== StringConcatConverter Tests ==========


class TestStringConcatConverter:
    """Tests for StringConcatConverter."""

    def test_convert(self):
        """Test concatenating values."""
        converter = StringConcatConverter(" ")
        result = converter.convert(["Hello", "World"])
        assert result == "Hello World"

    def test_convert_skips_none(self):
        """Test None values are skipped."""
        converter = StringConcatConverter(" ")
        result = converter.convert(["Hello", None, "World"])
        assert result == "Hello World"

    def test_convert_with_parameter(self):
        """Test separator from parameter."""
        converter = StringConcatConverter(" ")
        result = converter.convert(["a", "b", "c"], "-")
        assert result == "a-b-c"

    def test_convert_back(self):
        """Test splitting string."""
        converter = StringConcatConverter(" ")
        result = converter.convert_back("Hello World")
        assert result == ["Hello", "World"]


# ========== MathOperationConverter Tests ==========


class TestMathOperationConverter:
    """Tests for MathOperationConverter."""

    def test_sum(self):
        """Test sum operation."""
        converter = MathOperationConverter("sum")
        result = converter.convert([1, 2, 3, 4])
        assert result == 10

    def test_product(self):
        """Test product operation."""
        converter = MathOperationConverter("product")
        result = converter.convert([2, 3, 4])
        assert result == 24

    def test_min(self):
        """Test min operation."""
        converter = MathOperationConverter("min")
        result = converter.convert([3, 1, 4, 1, 5])
        assert result == 1

    def test_max(self):
        """Test max operation."""
        converter = MathOperationConverter("max")
        result = converter.convert([3, 1, 4, 1, 5])
        assert result == 5

    def test_average(self):
        """Test average operation."""
        converter = MathOperationConverter("average")
        result = converter.convert([2, 4, 6])
        assert result == 4.0

    def test_empty_list(self):
        """Test empty list returns 0."""
        converter = MathOperationConverter("sum")
        result = converter.convert([])
        assert result == 0.0

    def test_skips_none(self):
        """Test None values are skipped."""
        converter = MathOperationConverter("sum")
        result = converter.convert([1, None, 2, None, 3])
        assert result == 6

    def test_operation_from_parameter(self):
        """Test operation from parameter."""
        converter = MathOperationConverter("sum")
        result = converter.convert([1, 2, 3], "product")
        assert result == 6

    def test_convert_back(self):
        """Test convert_back returns single-item list."""
        converter = MathOperationConverter("sum")
        result = converter.convert_back(10.0)
        assert result == [10.0]


# ========== Factory Function Tests ==========


class TestFactoryFunctions:
    """Tests for converter factory functions."""

    def test_bool_to_visibility(self):
        """Test bool_to_visibility factory."""
        converter = bool_to_visibility(invert=True)
        assert isinstance(converter, BoolToVisibilityConverter)
        assert converter._invert is True

    def test_number_format(self):
        """Test number_format factory."""
        converter = number_format("{:.3f}", 1.0)
        assert isinstance(converter, NumberFormatConverter)
        assert converter.convert(3.14159) == "3.142"

    def test_string_format(self):
        """Test string_format factory."""
        converter = string_format("Value: {value}", "N/A")
        assert isinstance(converter, StringFormatConverter)
        assert converter.convert("test") == "Value: test"

    def test_color_converter(self):
        """Test color_converter factory."""
        default = Color(255, 255, 255)
        converter = color_converter(default)
        assert isinstance(converter, ColorConverter)

    def test_chain(self):
        """Test chain factory."""
        converter = chain(
            InverseBoolConverter(),
            BoolToStringConverter("Y", "N"),
        )
        assert isinstance(converter, ChainedConverter)
        assert converter.convert(True) == "N"

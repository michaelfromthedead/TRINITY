"""
Value converters for UI data binding.

Converters transform values between source and target during binding.
Supports synchronous and asynchronous conversion with chaining.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

TSource = TypeVar("TSource")
TTarget = TypeVar("TTarget")

# Color constants
COLOR_COMPONENT_MAX = 255

# Cache configuration
DEFAULT_CACHE_SIZE = 100


class IConverter(ABC, Generic[TSource, TTarget]):
    """
    Interface for value converters.

    Converters transform values in both directions for two-way binding.
    """

    @abstractmethod
    def convert(self, value: TSource, parameter: Any = None) -> TTarget:
        """Convert from source to target type."""
        pass

    @abstractmethod
    def convert_back(self, value: TTarget, parameter: Any = None) -> TSource:
        """Convert from target back to source type."""
        pass


class IAsyncConverter(ABC, Generic[TSource, TTarget]):
    """
    Interface for asynchronous value converters.

    Used when conversion requires I/O or expensive computation.
    """

    @abstractmethod
    async def convert(self, value: TSource, parameter: Any = None) -> TTarget:
        """Asynchronously convert from source to target type."""
        pass

    @abstractmethod
    async def convert_back(self, value: TTarget, parameter: Any = None) -> TSource:
        """Asynchronously convert from target back to source type."""
        pass


class Visibility(Enum):
    """Visibility states for UI elements."""

    VISIBLE = auto()
    HIDDEN = auto()
    COLLAPSED = auto()


class BoolToVisibilityConverter(IConverter[bool, Visibility]):
    """
    Converts boolean values to visibility states.

    Parameters:
        invert: If True, inverts the logic (False = visible)
        hidden_state: Visibility state when hidden (HIDDEN or COLLAPSED)
    """

    def __init__(
        self,
        invert: bool = False,
        hidden_state: Visibility = Visibility.COLLAPSED,
    ):
        self._invert = invert
        self._hidden_state = hidden_state

    def convert(self, value: bool, parameter: Any = None) -> Visibility:
        """Convert bool to Visibility."""
        visible = value if not self._invert else not value
        return Visibility.VISIBLE if visible else self._hidden_state

    def convert_back(self, value: Visibility, parameter: Any = None) -> bool:
        """Convert Visibility back to bool."""
        is_visible = value == Visibility.VISIBLE
        return is_visible if not self._invert else not is_visible


class NumberFormatConverter(IConverter[float, str]):
    """
    Formats numbers as strings with configurable precision.

    Parameters:
        format_string: Python format string (e.g., "{:.2f}", "{:,}")
        default_value: Value to return on parse failure
    """

    def __init__(
        self,
        format_string: str = "{:.2f}",
        default_value: float = 0.0,
    ):
        self._format = format_string
        self._default = default_value

    def convert(self, value: float, parameter: Any = None) -> str:
        """Format number as string."""
        try:
            fmt = parameter if isinstance(parameter, str) else self._format
            return fmt.format(value)
        except (ValueError, TypeError):
            return self._format.format(self._default)

    def convert_back(self, value: str, parameter: Any = None) -> float:
        """Parse string back to number."""
        try:
            # Remove common formatting characters
            cleaned = value.replace(",", "").replace(" ", "").strip()
            return float(cleaned)
        except (ValueError, TypeError):
            return self._default


class IntegerFormatConverter(IConverter[int, str]):
    """
    Formats integers as strings.

    Parameters:
        format_string: Python format string (e.g., "{:d}", "{:,}")
        default_value: Value to return on parse failure
    """

    def __init__(
        self,
        format_string: str = "{:d}",
        default_value: int = 0,
    ):
        self._format = format_string
        self._default = default_value

    def convert(self, value: int, parameter: Any = None) -> str:
        """Format integer as string."""
        try:
            fmt = parameter if isinstance(parameter, str) else self._format
            return fmt.format(int(value))
        except (ValueError, TypeError):
            return self._format.format(self._default)

    def convert_back(self, value: str, parameter: Any = None) -> int:
        """Parse string back to integer."""
        try:
            cleaned = value.replace(",", "").replace(" ", "").strip()
            return int(float(cleaned))  # Handle "1.0" -> 1
        except (ValueError, TypeError):
            return self._default


class StringFormatConverter(IConverter[Any, str]):
    """
    Formats any value as a string using a format template.

    Parameters:
        template: Format template with {value} placeholder
        null_value: String to use for None values
    """

    def __init__(
        self,
        template: str = "{value}",
        null_value: str = "",
    ):
        self._template = template
        self._null_value = null_value

    def convert(self, value: Any, parameter: Any = None) -> str:
        """Format value as string."""
        if value is None:
            return self._null_value
        template = parameter if isinstance(parameter, str) else self._template
        try:
            return template.format(value=value)
        except (ValueError, TypeError, KeyError):
            return str(value)

    def convert_back(self, value: str, parameter: Any = None) -> Any:
        """Cannot convert back - returns string as-is."""
        return value


@dataclass
class Color:
    """Simple color representation."""

    r: int  # 0-COLOR_COMPONENT_MAX
    g: int  # 0-COLOR_COMPONENT_MAX
    b: int  # 0-COLOR_COMPONENT_MAX
    a: int = COLOR_COMPONENT_MAX  # 0-COLOR_COMPONENT_MAX

    def to_hex(self) -> str:
        """Convert to hex string."""
        if self.a == COLOR_COMPONENT_MAX:
            return f"#{self.r:02x}{self.g:02x}{self.b:02x}"
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}{self.a:02x}"

    def to_rgba(self) -> Tuple[int, int, int, int]:
        """Convert to RGBA tuple."""
        return (self.r, self.g, self.b, self.a)

    def to_normalized(self) -> Tuple[float, float, float, float]:
        """Convert to normalized floats (0-1)."""
        return (
            self.r / COLOR_COMPONENT_MAX,
            self.g / COLOR_COMPONENT_MAX,
            self.b / COLOR_COMPONENT_MAX,
            self.a / COLOR_COMPONENT_MAX,
        )

    @classmethod
    def from_hex(cls, hex_string: str) -> "Color":
        """Parse from hex string (#RGB, #RGBA, #RRGGBB, #RRGGBBAA)."""
        h = hex_string.lstrip("#")
        if len(h) == 3:
            return cls(
                int(h[0] * 2, 16),
                int(h[1] * 2, 16),
                int(h[2] * 2, 16),
            )
        elif len(h) == 4:
            return cls(
                int(h[0] * 2, 16),
                int(h[1] * 2, 16),
                int(h[2] * 2, 16),
                int(h[3] * 2, 16),
            )
        elif len(h) == 6:
            return cls(
                int(h[0:2], 16),
                int(h[2:4], 16),
                int(h[4:6], 16),
            )
        elif len(h) == 8:
            return cls(
                int(h[0:2], 16),
                int(h[2:4], 16),
                int(h[4:6], 16),
                int(h[6:8], 16),
            )
        raise ValueError(f"Invalid hex color: {hex_string}")

    @classmethod
    def from_name(cls, name: str) -> "Color":
        """Parse from common color name."""
        colors = {
            "white": cls(255, 255, 255),
            "black": cls(0, 0, 0),
            "red": cls(255, 0, 0),
            "green": cls(0, 255, 0),
            "blue": cls(0, 0, 255),
            "yellow": cls(255, 255, 0),
            "cyan": cls(0, 255, 255),
            "magenta": cls(255, 0, 255),
            "gray": cls(128, 128, 128),
            "grey": cls(128, 128, 128),
            "orange": cls(255, 165, 0),
            "purple": cls(128, 0, 128),
            "transparent": cls(0, 0, 0, 0),
        }
        lower_name = name.lower()
        if lower_name in colors:
            return colors[lower_name]
        raise ValueError(f"Unknown color name: {name}")


class ColorConverter(IConverter[Color, str]):
    """
    Converts between Color objects and string representations.

    Supports hex strings (#RRGGBB, #RRGGBBAA) and named colors.
    """

    def __init__(self, default_color: Optional[Color] = None):
        self._default = default_color or Color(0, 0, 0)

    def convert(self, value: Color, parameter: Any = None) -> str:
        """Convert Color to hex string."""
        if value is None:
            return self._default.to_hex()
        return value.to_hex()

    def convert_back(self, value: str, parameter: Any = None) -> Color:
        """Parse string to Color."""
        if not value:
            return self._default
        try:
            value = value.strip()
            if value.startswith("#"):
                return Color.from_hex(value)
            return Color.from_name(value)
        except ValueError:
            return self._default


class ColorToRgbaConverter(IConverter[Color, Tuple[int, int, int, int]]):
    """Converts between Color and RGBA tuple."""

    def convert(
        self, value: Color, parameter: Any = None
    ) -> Tuple[int, int, int, int]:
        """Convert Color to RGBA tuple."""
        if value is None:
            return (0, 0, 0, COLOR_COMPONENT_MAX)
        return value.to_rgba()

    def convert_back(
        self, value: Tuple[int, int, int, int], parameter: Any = None
    ) -> Color:
        """Convert RGBA tuple to Color."""
        if not value or len(value) < 3:
            return Color(0, 0, 0)
        a = value[3] if len(value) > 3 else COLOR_COMPONENT_MAX
        return Color(value[0], value[1], value[2], a)


class BoolToStringConverter(IConverter[bool, str]):
    """
    Converts boolean to custom string values.

    Parameters:
        true_string: String for True value
        false_string: String for False value
    """

    def __init__(self, true_string: str = "Yes", false_string: str = "No"):
        self._true = true_string
        self._false = false_string

    def convert(self, value: bool, parameter: Any = None) -> str:
        """Convert bool to string."""
        return self._true if value else self._false

    def convert_back(self, value: str, parameter: Any = None) -> bool:
        """Convert string to bool."""
        return value == self._true


class InverseBoolConverter(IConverter[bool, bool]):
    """Inverts boolean values."""

    def convert(self, value: bool, parameter: Any = None) -> bool:
        """Invert boolean."""
        return not value

    def convert_back(self, value: bool, parameter: Any = None) -> bool:
        """Invert back (same as convert)."""
        return not value


class NullToBoolConverter(IConverter[Any, bool]):
    """
    Converts null/None to boolean.

    Parameters:
        invert: If True, None returns True
    """

    def __init__(self, invert: bool = False):
        self._invert = invert

    def convert(self, value: Any, parameter: Any = None) -> bool:
        """Convert to bool (True if not None)."""
        result = value is not None
        return not result if self._invert else result

    def convert_back(self, value: bool, parameter: Any = None) -> Any:
        """Cannot meaningfully convert back - returns None."""
        return None


class EnumToStringConverter(IConverter[Enum, str]):
    """Converts enum values to their string names."""

    def __init__(self, enum_type: type = None):
        self._enum_type = enum_type

    def convert(self, value: Enum, parameter: Any = None) -> str:
        """Convert enum to string name."""
        if value is None:
            return ""
        return value.name

    def convert_back(self, value: str, parameter: Any = None) -> Optional[Enum]:
        """Convert string name back to enum."""
        if not value or not self._enum_type:
            return None
        try:
            return self._enum_type[value]
        except KeyError:
            return None


class PercentageConverter(IConverter[float, str]):
    """Converts decimal to percentage string (0.5 -> "50%")."""

    def __init__(self, decimal_places: int = 0):
        self._decimals = decimal_places

    def convert(self, value: float, parameter: Any = None) -> str:
        """Convert decimal to percentage string."""
        pct = value * 100
        if self._decimals == 0:
            return f"{int(pct)}%"
        return f"{pct:.{self._decimals}f}%"

    def convert_back(self, value: str, parameter: Any = None) -> float:
        """Parse percentage string to decimal."""
        try:
            cleaned = value.replace("%", "").strip()
            return float(cleaned) / 100
        except (ValueError, TypeError):
            return 0.0


class DateTimeFormatConverter(IConverter[Any, str]):
    """
    Formats datetime objects as strings.

    Parameters:
        format_string: strftime format string
    """

    def __init__(self, format_string: str = "%Y-%m-%d %H:%M:%S"):
        self._format = format_string

    def convert(self, value: Any, parameter: Any = None) -> str:
        """Format datetime as string."""
        if value is None:
            return ""
        try:
            fmt = parameter if isinstance(parameter, str) else self._format
            return value.strftime(fmt)
        except (AttributeError, ValueError):
            return str(value)

    def convert_back(self, value: str, parameter: Any = None) -> Any:
        """Parse string to datetime (requires datetime module)."""
        from datetime import datetime

        if not value:
            return None
        try:
            fmt = parameter if isinstance(parameter, str) else self._format
            return datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            return None


class ChainedConverter(IConverter[Any, Any]):
    """
    Chains multiple converters together.

    Values flow through converters in order for convert,
    and in reverse order for convert_back.
    """

    def __init__(self, converters: List[IConverter]):
        if not converters:
            raise ValueError("at least one converter required")
        self._converters = converters

    def convert(self, value: Any, parameter: Any = None) -> Any:
        """Apply all converters in sequence."""
        result = value
        for converter in self._converters:
            result = converter.convert(result, parameter)
        return result

    def convert_back(self, value: Any, parameter: Any = None) -> Any:
        """Apply all converters in reverse sequence."""
        result = value
        for converter in reversed(self._converters):
            result = converter.convert_back(result, parameter)
        return result


class LambdaConverter(IConverter[TSource, TTarget]):
    """
    Creates a converter from lambda functions.

    Parameters:
        convert_func: Function for forward conversion
        convert_back_func: Function for backward conversion (optional)
    """

    def __init__(
        self,
        convert_func: Callable[[TSource, Any], TTarget],
        convert_back_func: Optional[Callable[[TTarget, Any], TSource]] = None,
    ):
        self._convert = convert_func
        self._convert_back = convert_back_func

    def convert(self, value: TSource, parameter: Any = None) -> TTarget:
        """Apply the conversion function."""
        return self._convert(value, parameter)

    def convert_back(self, value: TTarget, parameter: Any = None) -> TSource:
        """Apply the back-conversion function."""
        if self._convert_back is None:
            raise NotImplementedError("convert_back not provided")
        return self._convert_back(value, parameter)


class AsyncLambdaConverter(IAsyncConverter[TSource, TTarget]):
    """
    Creates an async converter from async lambda functions.

    Parameters:
        convert_func: Async function for forward conversion
        convert_back_func: Async function for backward conversion (optional)
    """

    def __init__(
        self,
        convert_func: Callable[[TSource, Any], Awaitable[TTarget]],
        convert_back_func: Optional[
            Callable[[TTarget, Any], Awaitable[TSource]]
        ] = None,
    ):
        self._convert = convert_func
        self._convert_back = convert_back_func

    async def convert(self, value: TSource, parameter: Any = None) -> TTarget:
        """Apply the async conversion function."""
        return await self._convert(value, parameter)

    async def convert_back(self, value: TTarget, parameter: Any = None) -> TSource:
        """Apply the async back-conversion function."""
        if self._convert_back is None:
            raise NotImplementedError("convert_back not provided")
        return await self._convert_back(value, parameter)


class CachedConverter(IConverter[TSource, TTarget]):
    """
    Wraps a converter with caching for expensive conversions.

    Parameters:
        inner: The converter to wrap
        max_cache_size: Maximum number of cached conversions
    """

    def __init__(self, inner: IConverter[TSource, TTarget], max_cache_size: int = DEFAULT_CACHE_SIZE):
        self._inner = inner
        self._max_size = max_cache_size
        self._convert_cache: dict = {}
        self._back_cache: dict = {}

    def convert(self, value: TSource, parameter: Any = None) -> TTarget:
        """Convert with caching."""
        key = (value, parameter)
        try:
            hash(key)  # Check if hashable
        except TypeError:
            return self._inner.convert(value, parameter)

        if key not in self._convert_cache:
            if len(self._convert_cache) >= self._max_size:
                # Remove oldest entry
                oldest = next(iter(self._convert_cache))
                del self._convert_cache[oldest]
            self._convert_cache[key] = self._inner.convert(value, parameter)
        return self._convert_cache[key]

    def convert_back(self, value: TTarget, parameter: Any = None) -> TSource:
        """Convert back with caching."""
        key = (value, parameter)
        try:
            hash(key)
        except TypeError:
            return self._inner.convert_back(value, parameter)

        if key not in self._back_cache:
            if len(self._back_cache) >= self._max_size:
                oldest = next(iter(self._back_cache))
                del self._back_cache[oldest]
            self._back_cache[key] = self._inner.convert_back(value, parameter)
        return self._back_cache[key]

    def clear_cache(self) -> None:
        """Clear both caches."""
        self._convert_cache.clear()
        self._back_cache.clear()


class MultiValueConverter:
    """
    Converts multiple source values into a single target value.

    Used for bindings with multiple sources.
    """

    def convert(self, values: List[Any], parameter: Any = None) -> Any:
        """Convert multiple values to single target value."""
        raise NotImplementedError()

    def convert_back(
        self, value: Any, parameter: Any = None
    ) -> List[Any]:
        """Convert single value back to multiple values."""
        raise NotImplementedError()


class StringConcatConverter(MultiValueConverter):
    """Concatenates multiple string values."""

    def __init__(self, separator: str = " "):
        self._separator = separator

    def convert(self, values: List[Any], parameter: Any = None) -> str:
        """Concatenate values as strings."""
        sep = parameter if isinstance(parameter, str) else self._separator
        return sep.join(str(v) for v in values if v is not None)

    def convert_back(self, value: str, parameter: Any = None) -> List[str]:
        """Split string into parts."""
        sep = parameter if isinstance(parameter, str) else self._separator
        return value.split(sep)


class MathOperationConverter(MultiValueConverter):
    """Performs math operations on multiple numeric values."""

    def __init__(self, operation: str = "sum"):
        """
        Initialize with operation type.

        Operations: sum, product, min, max, average
        """
        self._operation = operation

    def convert(self, values: List[Any], parameter: Any = None) -> float:
        """Apply math operation to values."""
        nums = [float(v) for v in values if v is not None]
        if not nums:
            return 0.0

        op = parameter if isinstance(parameter, str) else self._operation
        if op == "sum":
            return sum(nums)
        elif op == "product":
            result = 1.0
            for n in nums:
                result *= n
            return result
        elif op == "min":
            return min(nums)
        elif op == "max":
            return max(nums)
        elif op == "average":
            return sum(nums) / len(nums)
        return 0.0

    def convert_back(self, value: float, parameter: Any = None) -> List[float]:
        """Cannot meaningfully convert back."""
        return [value]


# Convenience factory functions
def bool_to_visibility(
    invert: bool = False, hidden_state: Visibility = Visibility.COLLAPSED
) -> BoolToVisibilityConverter:
    """Create a bool-to-visibility converter."""
    return BoolToVisibilityConverter(invert, hidden_state)


def number_format(
    format_string: str = "{:.2f}", default_value: float = 0.0
) -> NumberFormatConverter:
    """Create a number format converter."""
    return NumberFormatConverter(format_string, default_value)


def string_format(
    template: str = "{value}", null_value: str = ""
) -> StringFormatConverter:
    """Create a string format converter."""
    return StringFormatConverter(template, null_value)


def color_converter(default_color: Optional[Color] = None) -> ColorConverter:
    """Create a color converter."""
    return ColorConverter(default_color)


def chain(*converters: IConverter) -> ChainedConverter:
    """Chain multiple converters together."""
    return ChainedConverter(list(converters))


__all__ = [
    # Interfaces
    "IConverter",
    "IAsyncConverter",
    # Types
    "Color",
    "Visibility",
    # Built-in converters
    "BoolToVisibilityConverter",
    "BoolToStringConverter",
    "NumberFormatConverter",
    "IntegerFormatConverter",
    "StringFormatConverter",
    "ColorConverter",
    "ColorToRgbaConverter",
    "InverseBoolConverter",
    "NullToBoolConverter",
    "EnumToStringConverter",
    "PercentageConverter",
    "DateTimeFormatConverter",
    # Advanced converters
    "ChainedConverter",
    "LambdaConverter",
    "AsyncLambdaConverter",
    "CachedConverter",
    # Multi-value converters
    "MultiValueConverter",
    "StringConcatConverter",
    "MathOperationConverter",
    # Factory functions
    "bool_to_visibility",
    "number_format",
    "string_format",
    "color_converter",
    "chain",
]

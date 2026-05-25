"""
Test assertions for the engine testing framework.

Provides a comprehensive set of assertion functions for unit, integration,
and functional testing. Each assertion produces detailed failure messages
to aid in debugging.

Usage:
    from engine.debug.testing.assertions import expect_eq, expect_true, expect_throws

    expect_eq(actual, expected, "Values should match")
    expect_true(condition, "Condition should be true")
    expect_throws(lambda: risky_call(), ValueError, "Should raise ValueError")
"""

from __future__ import annotations

import math
from typing import Any, Callable, Container, Optional, Type, TypeVar, Union


__all__ = [
    "TestFailure",
    "expect_eq",
    "expect_ne",
    "expect_true",
    "expect_false",
    "expect_near",
    "expect_throws",
    "expect_contains",
    "expect_not_contains",
    "expect_is",
    "expect_is_not",
    "expect_none",
    "expect_not_none",
    "expect_greater",
    "expect_greater_eq",
    "expect_less",
    "expect_less_eq",
    "expect_in_range",
    "expect_type",
    "expect_instance",
]


T = TypeVar("T")
E = TypeVar("E", bound=BaseException)

# Constants for magic numbers
DEFAULT_VALUE_FORMAT_MAX_LENGTH = 100
DEFAULT_FLOAT_EPSILON = 1e-6


class TestFailure(AssertionError):
    """
    Exception raised when a test assertion fails.

    Provides detailed information about the failure including:
    - The assertion type that failed
    - Expected and actual values
    - Custom message if provided
    - Source location (when available)

    Attributes:
        message: The failure message
        expected: The expected value (if applicable)
        actual: The actual value (if applicable)
        assertion_type: The type of assertion that failed
    """

    # Prevent pytest from collecting this class as a test
    __test__ = False

    def __init__(
        self,
        message: str,
        expected: Any = None,
        actual: Any = None,
        assertion_type: str = "assertion",
    ) -> None:
        """
        Initialize a TestFailure.

        Args:
            message: Human-readable failure description
            expected: The expected value
            actual: The actual value
            assertion_type: Name of the assertion that failed
        """
        self.message = message
        self.expected = expected
        self.actual = actual
        self.assertion_type = assertion_type
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the complete failure message."""
        parts = [f"[{self.assertion_type}] {self.message}"]

        if self.expected is not None or self.actual is not None:
            parts.append("")
            if self.expected is not None:
                parts.append(f"  Expected: {_format_value(self.expected)}")
            if self.actual is not None:
                parts.append(f"  Actual:   {_format_value(self.actual)}")

        return "\n".join(parts)

    def __repr__(self) -> str:
        return f"TestFailure({self.message!r}, expected={self.expected!r}, actual={self.actual!r})"


def _format_value(value: Any, max_length: int = DEFAULT_VALUE_FORMAT_MAX_LENGTH) -> str:
    """
    Format a value for display in error messages.

    Args:
        value: The value to format
        max_length: Maximum length of the formatted string

    Returns:
        A formatted string representation
    """
    try:
        formatted = repr(value)
        if len(formatted) > max_length:
            formatted = formatted[:max_length - 3] + "..."
        # Add type hint for clarity
        type_name = type(value).__name__
        return f"{formatted} ({type_name})"
    except Exception:
        return f"<unrepresentable {type(value).__name__}>"


def _build_message(default: str, custom: Optional[str]) -> str:
    """Build the failure message, optionally prepending custom message."""
    if custom:
        return f"{custom}: {default}"
    return default


def expect_eq(actual: T, expected: T, msg: Optional[str] = None) -> None:
    """
    Assert that two values are equal.

    Uses == operator for comparison. For floating point comparisons,
    use expect_near() instead.

    Args:
        actual: The actual value
        expected: The expected value
        msg: Optional custom failure message

    Raises:
        TestFailure: If values are not equal

    Example:
        expect_eq(calculate_sum(2, 3), 5)
        expect_eq(user.name, "Alice", "User name should match")
    """
    if actual != expected:
        raise TestFailure(
            _build_message("Values are not equal", msg),
            expected=expected,
            actual=actual,
            assertion_type="expect_eq",
        )


def expect_ne(actual: T, expected: T, msg: Optional[str] = None) -> None:
    """
    Assert that two values are not equal.

    Uses != operator for comparison.

    Args:
        actual: The actual value
        expected: The value that actual should not equal
        msg: Optional custom failure message

    Raises:
        TestFailure: If values are equal

    Example:
        expect_ne(generate_id(), generate_id(), "IDs should be unique")
    """
    if actual == expected:
        raise TestFailure(
            _build_message("Values should not be equal", msg),
            expected=f"not {_format_value(expected)}",
            actual=actual,
            assertion_type="expect_ne",
        )


def expect_true(condition: bool, msg: Optional[str] = None) -> None:
    """
    Assert that a condition is True.

    Args:
        condition: The condition to check
        msg: Optional custom failure message

    Raises:
        TestFailure: If condition is not True

    Example:
        expect_true(user.is_active, "User should be active")
        expect_true(len(items) > 0, "Collection should not be empty")
    """
    if condition is not True:
        raise TestFailure(
            _build_message("Condition is not True", msg),
            expected=True,
            actual=condition,
            assertion_type="expect_true",
        )


def expect_false(condition: bool, msg: Optional[str] = None) -> None:
    """
    Assert that a condition is False.

    Args:
        condition: The condition to check
        msg: Optional custom failure message

    Raises:
        TestFailure: If condition is not False

    Example:
        expect_false(user.is_banned, "User should not be banned")
        expect_false(has_errors(), "Should have no errors")
    """
    if condition is not False:
        raise TestFailure(
            _build_message("Condition is not False", msg),
            expected=False,
            actual=condition,
            assertion_type="expect_false",
        )


def expect_near(
    actual: float,
    expected: float,
    epsilon: float = DEFAULT_FLOAT_EPSILON,
    msg: Optional[str] = None,
) -> None:
    """
    Assert that two floating-point values are approximately equal.

    Uses absolute difference comparison: |actual - expected| <= epsilon

    Args:
        actual: The actual value
        expected: The expected value
        epsilon: Maximum allowed difference (default 1e-6)
        msg: Optional custom failure message

    Raises:
        TestFailure: If values differ by more than epsilon

    Example:
        expect_near(calculate_pi(), 3.14159, epsilon=0.00001)
        expect_near(normalized.length(), 1.0, epsilon=1e-6, msg="Should be unit length")
    """
    difference = abs(actual - expected)
    if difference > epsilon:
        raise TestFailure(
            _build_message(f"Values differ by {difference} (epsilon={epsilon})", msg),
            expected=expected,
            actual=actual,
            assertion_type="expect_near",
        )


def expect_throws(
    callable: Callable[[], Any],
    exception_type: Type[E],
    msg: Optional[str] = None,
    match: Optional[str] = None,
) -> E:
    """
    Assert that a callable raises a specific exception type.

    Args:
        callable: A zero-argument callable to execute
        exception_type: The expected exception type
        msg: Optional custom failure message
        match: Optional substring that must appear in exception message

    Returns:
        The caught exception instance for further inspection

    Raises:
        TestFailure: If callable does not raise expected exception

    Example:
        expect_throws(lambda: int("invalid"), ValueError)
        exc = expect_throws(lambda: divide(1, 0), ZeroDivisionError, match="division")
    """
    try:
        callable()
    except exception_type as e:
        if match is not None and match not in str(e):
            raise TestFailure(
                _build_message(
                    f"Exception message does not contain expected substring",
                    msg,
                ),
                expected=f"message containing '{match}'",
                actual=str(e),
                assertion_type="expect_throws",
            )
        return e
    except Exception as e:
        raise TestFailure(
            _build_message(f"Wrong exception type raised", msg),
            expected=exception_type.__name__,
            actual=f"{type(e).__name__}: {e}",
            assertion_type="expect_throws",
        )
    else:
        raise TestFailure(
            _build_message("Expected exception was not raised", msg),
            expected=exception_type.__name__,
            actual="no exception",
            assertion_type="expect_throws",
        )


def expect_contains(
    container: Container[T],
    item: T,
    msg: Optional[str] = None,
) -> None:
    """
    Assert that a container contains an item.

    Uses 'in' operator for containment check.

    Args:
        container: The container to search (list, set, dict, string, etc.)
        item: The item to find
        msg: Optional custom failure message

    Raises:
        TestFailure: If item is not in container

    Example:
        expect_contains([1, 2, 3], 2)
        expect_contains(user_names, "Alice", "Alice should be in user list")
        expect_contains("hello world", "world")
    """
    if item not in container:
        raise TestFailure(
            _build_message(f"Container does not contain expected item", msg),
            expected=f"container containing {_format_value(item)}",
            actual=container,
            assertion_type="expect_contains",
        )


def expect_not_contains(
    container: Container[T],
    item: T,
    msg: Optional[str] = None,
) -> None:
    """
    Assert that a container does not contain an item.

    Uses 'not in' operator for containment check.

    Args:
        container: The container to search
        item: The item that should not be present
        msg: Optional custom failure message

    Raises:
        TestFailure: If item is in container

    Example:
        expect_not_contains(banned_users, current_user)
    """
    if item in container:
        raise TestFailure(
            _build_message(f"Container should not contain item", msg),
            expected=f"container not containing {_format_value(item)}",
            actual=container,
            assertion_type="expect_not_contains",
        )


def expect_is(actual: T, expected: T, msg: Optional[str] = None) -> None:
    """
    Assert that two references are the same object (identity check).

    Uses 'is' operator.

    Args:
        actual: The actual object
        expected: The expected object
        msg: Optional custom failure message

    Raises:
        TestFailure: If objects are not identical

    Example:
        expect_is(singleton.instance(), singleton.instance())
    """
    if actual is not expected:
        raise TestFailure(
            _build_message("Objects are not identical", msg),
            expected=f"same object as {_format_value(expected)} (id={id(expected)})",
            actual=f"{_format_value(actual)} (id={id(actual)})",
            assertion_type="expect_is",
        )


def expect_is_not(actual: T, expected: T, msg: Optional[str] = None) -> None:
    """
    Assert that two references are different objects.

    Uses 'is not' operator.

    Args:
        actual: The actual object
        expected: The object that actual should not be
        msg: Optional custom failure message

    Raises:
        TestFailure: If objects are identical

    Example:
        expect_is_not(copy, original, "Copy should be a different object")
    """
    if actual is expected:
        raise TestFailure(
            _build_message("Objects should not be identical", msg),
            expected="different object",
            actual=f"same object (id={id(actual)})",
            assertion_type="expect_is_not",
        )


def expect_none(value: Any, msg: Optional[str] = None) -> None:
    """
    Assert that a value is None.

    Args:
        value: The value to check
        msg: Optional custom failure message

    Raises:
        TestFailure: If value is not None

    Example:
        expect_none(find_user("nonexistent"), "Should return None for unknown user")
    """
    if value is not None:
        raise TestFailure(
            _build_message("Value is not None", msg),
            expected=None,
            actual=value,
            assertion_type="expect_none",
        )


def expect_not_none(value: T, msg: Optional[str] = None) -> T:
    """
    Assert that a value is not None.

    Args:
        value: The value to check
        msg: Optional custom failure message

    Returns:
        The value (for chaining)

    Raises:
        TestFailure: If value is None

    Example:
        user = expect_not_none(find_user("alice"), "User should exist")
    """
    if value is None:
        raise TestFailure(
            _build_message("Value is unexpectedly None", msg),
            expected="non-None value",
            actual=None,
            assertion_type="expect_not_none",
        )
    return value


def expect_greater(actual: Any, threshold: Any, msg: Optional[str] = None) -> None:
    """
    Assert that actual > threshold.

    Args:
        actual: The actual value
        threshold: The threshold that actual should exceed
        msg: Optional custom failure message

    Raises:
        TestFailure: If actual <= threshold

    Example:
        expect_greater(score, 0, "Score should be positive")
    """
    if not (actual > threshold):
        raise TestFailure(
            _build_message(f"Value is not greater than threshold", msg),
            expected=f"> {_format_value(threshold)}",
            actual=actual,
            assertion_type="expect_greater",
        )


def expect_greater_eq(actual: Any, threshold: Any, msg: Optional[str] = None) -> None:
    """
    Assert that actual >= threshold.

    Args:
        actual: The actual value
        threshold: The threshold
        msg: Optional custom failure message

    Raises:
        TestFailure: If actual < threshold

    Example:
        expect_greater_eq(health, 0, "Health should be non-negative")
    """
    if not (actual >= threshold):
        raise TestFailure(
            _build_message(f"Value is less than threshold", msg),
            expected=f">= {_format_value(threshold)}",
            actual=actual,
            assertion_type="expect_greater_eq",
        )


def expect_less(actual: Any, threshold: Any, msg: Optional[str] = None) -> None:
    """
    Assert that actual < threshold.

    Args:
        actual: The actual value
        threshold: The threshold that actual should be below
        msg: Optional custom failure message

    Raises:
        TestFailure: If actual >= threshold

    Example:
        expect_less(latency_ms, 100, "Latency should be under 100ms")
    """
    if not (actual < threshold):
        raise TestFailure(
            _build_message(f"Value is not less than threshold", msg),
            expected=f"< {_format_value(threshold)}",
            actual=actual,
            assertion_type="expect_less",
        )


def expect_less_eq(actual: Any, threshold: Any, msg: Optional[str] = None) -> None:
    """
    Assert that actual <= threshold.

    Args:
        actual: The actual value
        threshold: The threshold
        msg: Optional custom failure message

    Raises:
        TestFailure: If actual > threshold

    Example:
        expect_less_eq(size, max_size, "Size should not exceed maximum")
    """
    if not (actual <= threshold):
        raise TestFailure(
            _build_message(f"Value exceeds threshold", msg),
            expected=f"<= {_format_value(threshold)}",
            actual=actual,
            assertion_type="expect_less_eq",
        )


def expect_in_range(
    value: Any,
    low: Any,
    high: Any,
    inclusive: bool = True,
    msg: Optional[str] = None,
) -> None:
    """
    Assert that a value is within a range.

    Args:
        value: The value to check
        low: Lower bound of the range
        high: Upper bound of the range
        inclusive: If True, bounds are inclusive (low <= value <= high).
                  If False, bounds are exclusive (low < value < high).
        msg: Optional custom failure message

    Raises:
        TestFailure: If value is outside the range

    Example:
        expect_in_range(angle, 0, 360)
        expect_in_range(probability, 0.0, 1.0, msg="Probability must be between 0 and 1")
    """
    if inclusive:
        in_range = low <= value <= high
        range_str = f"[{low}, {high}]"
    else:
        in_range = low < value < high
        range_str = f"({low}, {high})"

    if not in_range:
        raise TestFailure(
            _build_message(f"Value is outside expected range", msg),
            expected=f"value in range {range_str}",
            actual=value,
            assertion_type="expect_in_range",
        )


def expect_type(value: Any, expected_type: type, msg: Optional[str] = None) -> None:
    """
    Assert that a value has an exact type (not subclass).

    Args:
        value: The value to check
        expected_type: The expected exact type
        msg: Optional custom failure message

    Raises:
        TestFailure: If type(value) is not expected_type

    Example:
        expect_type(result, int)  # Fails for bool even though bool is subclass of int
    """
    if type(value) is not expected_type:
        raise TestFailure(
            _build_message(f"Value has wrong type", msg),
            expected=expected_type.__name__,
            actual=type(value).__name__,
            assertion_type="expect_type",
        )


def expect_instance(
    value: Any,
    expected_type: Union[type, tuple],
    msg: Optional[str] = None,
) -> None:
    """
    Assert that a value is an instance of a type (including subclasses).

    Args:
        value: The value to check
        expected_type: The expected type or tuple of types
        msg: Optional custom failure message

    Raises:
        TestFailure: If not isinstance(value, expected_type)

    Example:
        expect_instance(error, Exception)
        expect_instance(number, (int, float))
    """
    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            expected_str = " | ".join(t.__name__ for t in expected_type)
        else:
            expected_str = expected_type.__name__

        raise TestFailure(
            _build_message(f"Value is not an instance of expected type", msg),
            expected=f"instance of {expected_str}",
            actual=f"{type(value).__name__}",
            assertion_type="expect_instance",
        )

"""
Tests for the testing framework assertions module.

Verifies all assertion functions work correctly and produce
appropriate failure messages.
"""

import pytest
import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.debug.testing.assertions import (
    TestFailure,
    expect_eq,
    expect_ne,
    expect_true,
    expect_false,
    expect_near,
    expect_throws,
    expect_contains,
    expect_not_contains,
    expect_is,
    expect_is_not,
    expect_none,
    expect_not_none,
    expect_greater,
    expect_greater_eq,
    expect_less,
    expect_less_eq,
    expect_in_range,
    expect_type,
    expect_instance,
)


class TestTestFailure:
    """Tests for TestFailure exception class."""

    def test_basic_message(self):
        """TestFailure should store and format basic message."""
        error = TestFailure("Something went wrong")
        assert "Something went wrong" in str(error)
        assert error.message == "Something went wrong"

    def test_expected_actual_values(self):
        """TestFailure should store expected and actual values."""
        error = TestFailure("Mismatch", expected=42, actual=17)
        assert error.expected == 42
        assert error.actual == 17
        assert "42" in str(error)
        assert "17" in str(error)

    def test_assertion_type(self):
        """TestFailure should include assertion type in message."""
        error = TestFailure("Mismatch", assertion_type="expect_eq")
        assert "[expect_eq]" in str(error)

    def test_repr(self):
        """TestFailure repr should be informative."""
        error = TestFailure("Mismatch", expected=1, actual=2)
        assert "TestFailure" in repr(error)
        assert "Mismatch" in repr(error)


class TestExpectEq:
    """Tests for expect_eq assertion."""

    def test_equal_values_pass(self):
        """expect_eq should pass for equal values."""
        expect_eq(1, 1)
        expect_eq("hello", "hello")
        expect_eq([1, 2, 3], [1, 2, 3])
        expect_eq({"a": 1}, {"a": 1})

    def test_unequal_values_fail(self):
        """expect_eq should fail for unequal values."""
        with pytest.raises(TestFailure) as exc_info:
            expect_eq(1, 2)
        assert "not equal" in str(exc_info.value).lower()
        assert exc_info.value.expected == 2
        assert exc_info.value.actual == 1

    def test_custom_message(self):
        """expect_eq should include custom message."""
        with pytest.raises(TestFailure) as exc_info:
            expect_eq(1, 2, "Custom message")
        assert "Custom message" in str(exc_info.value)

    def test_different_types_fail(self):
        """expect_eq should fail for different types."""
        with pytest.raises(TestFailure):
            expect_eq(1, "1")

    def test_none_comparison(self):
        """expect_eq should handle None values."""
        expect_eq(None, None)
        with pytest.raises(TestFailure):
            expect_eq(None, 0)


class TestExpectNe:
    """Tests for expect_ne assertion."""

    def test_unequal_values_pass(self):
        """expect_ne should pass for unequal values."""
        expect_ne(1, 2)
        expect_ne("hello", "world")
        expect_ne([1], [1, 2])

    def test_equal_values_fail(self):
        """expect_ne should fail for equal values."""
        with pytest.raises(TestFailure) as exc_info:
            expect_ne(42, 42)
        assert "should not be equal" in str(exc_info.value).lower()

    def test_custom_message(self):
        """expect_ne should include custom message."""
        with pytest.raises(TestFailure) as exc_info:
            expect_ne(1, 1, "Values must differ")
        assert "Values must differ" in str(exc_info.value)


class TestExpectTrue:
    """Tests for expect_true assertion."""

    def test_true_passes(self):
        """expect_true should pass for True."""
        expect_true(True)
        expect_true(1 == 1)
        expect_true(bool("hello"))

    def test_false_fails(self):
        """expect_true should fail for False."""
        with pytest.raises(TestFailure) as exc_info:
            expect_true(False)
        assert exc_info.value.expected is True
        assert exc_info.value.actual is False

    def test_truthy_values_fail(self):
        """expect_true should fail for truthy but not True values."""
        with pytest.raises(TestFailure):
            expect_true(1)  # Truthy but not True
        with pytest.raises(TestFailure):
            expect_true("yes")  # Truthy but not True

    def test_custom_message(self):
        """expect_true should include custom message."""
        with pytest.raises(TestFailure) as exc_info:
            expect_true(False, "Should be enabled")
        assert "Should be enabled" in str(exc_info.value)


class TestExpectFalse:
    """Tests for expect_false assertion."""

    def test_false_passes(self):
        """expect_false should pass for False."""
        expect_false(False)
        expect_false(1 == 2)

    def test_true_fails(self):
        """expect_false should fail for True."""
        with pytest.raises(TestFailure) as exc_info:
            expect_false(True)
        assert exc_info.value.expected is False
        assert exc_info.value.actual is True

    def test_falsy_values_fail(self):
        """expect_false should fail for falsy but not False values."""
        with pytest.raises(TestFailure):
            expect_false(0)  # Falsy but not False
        with pytest.raises(TestFailure):
            expect_false(None)  # Falsy but not False


class TestExpectNear:
    """Tests for expect_near assertion."""

    def test_equal_floats_pass(self):
        """expect_near should pass for equal floats."""
        expect_near(1.0, 1.0)
        expect_near(3.14159, 3.14159)

    def test_close_floats_pass(self):
        """expect_near should pass for floats within epsilon."""
        expect_near(1.0, 1.0000001, epsilon=1e-6)
        expect_near(0.333, 1.0 / 3.0, epsilon=0.001)

    def test_distant_floats_fail(self):
        """expect_near should fail for floats outside epsilon."""
        with pytest.raises(TestFailure) as exc_info:
            expect_near(1.0, 1.1, epsilon=0.01)
        assert "epsilon" in str(exc_info.value).lower()

    def test_custom_epsilon(self):
        """expect_near should use custom epsilon."""
        expect_near(1.0, 1.5, epsilon=1.0)  # Large epsilon
        with pytest.raises(TestFailure):
            expect_near(1.0, 1.001, epsilon=1e-4)  # Tight epsilon

    def test_custom_message(self):
        """expect_near should include custom message."""
        with pytest.raises(TestFailure) as exc_info:
            expect_near(1.0, 2.0, msg="Values should be close")
        assert "Values should be close" in str(exc_info.value)


class TestExpectThrows:
    """Tests for expect_throws assertion."""

    def test_correct_exception_passes(self):
        """expect_throws should pass when correct exception is raised."""
        expect_throws(lambda: 1 / 0, ZeroDivisionError)
        expect_throws(lambda: int("abc"), ValueError)

    def test_returns_exception(self):
        """expect_throws should return the caught exception."""
        exc = expect_throws(lambda: int("abc"), ValueError)
        assert isinstance(exc, ValueError)
        assert "abc" in str(exc)

    def test_no_exception_fails(self):
        """expect_throws should fail when no exception is raised."""
        with pytest.raises(TestFailure) as exc_info:
            expect_throws(lambda: 1 + 1, ValueError)
        assert "not raised" in str(exc_info.value).lower()

    def test_wrong_exception_fails(self):
        """expect_throws should fail for wrong exception type."""
        with pytest.raises(TestFailure) as exc_info:
            expect_throws(lambda: 1 / 0, ValueError)
        assert "ZeroDivisionError" in str(exc_info.value)

    def test_subclass_exception_passes(self):
        """expect_throws should pass for exception subclasses."""
        class CustomError(ValueError):
            pass
        expect_throws(lambda: (_ for _ in ()).throw(CustomError()), ValueError)

    def test_match_substring(self):
        """expect_throws should verify exception message contains match."""
        expect_throws(lambda: int("abc"), ValueError, match="invalid literal")
        with pytest.raises(TestFailure) as exc_info:
            expect_throws(lambda: int("abc"), ValueError, match="xyz")
        assert "substring" in str(exc_info.value).lower()

    def test_custom_message(self):
        """expect_throws should include custom message."""
        with pytest.raises(TestFailure) as exc_info:
            expect_throws(lambda: None, ValueError, msg="Should raise")
        assert "Should raise" in str(exc_info.value)


class TestExpectContains:
    """Tests for expect_contains assertion."""

    def test_list_contains_passes(self):
        """expect_contains should pass when item is in list."""
        expect_contains([1, 2, 3], 2)
        expect_contains(["a", "b"], "a")

    def test_string_contains_passes(self):
        """expect_contains should pass for substring."""
        expect_contains("hello world", "world")
        expect_contains("abc", "b")

    def test_set_contains_passes(self):
        """expect_contains should pass for set membership."""
        expect_contains({1, 2, 3}, 2)

    def test_dict_contains_passes(self):
        """expect_contains should pass for dict key."""
        expect_contains({"a": 1, "b": 2}, "a")

    def test_not_contains_fails(self):
        """expect_contains should fail when item is not in container."""
        with pytest.raises(TestFailure) as exc_info:
            expect_contains([1, 2, 3], 4)
        assert "does not contain" in str(exc_info.value).lower()

    def test_custom_message(self):
        """expect_contains should include custom message."""
        with pytest.raises(TestFailure) as exc_info:
            expect_contains([], "x", "List should have item")
        assert "List should have item" in str(exc_info.value)


class TestExpectNotContains:
    """Tests for expect_not_contains assertion."""

    def test_not_in_list_passes(self):
        """expect_not_contains should pass when item not in list."""
        expect_not_contains([1, 2, 3], 4)
        expect_not_contains([], "anything")

    def test_in_list_fails(self):
        """expect_not_contains should fail when item is in container."""
        with pytest.raises(TestFailure) as exc_info:
            expect_not_contains([1, 2, 3], 2)
        assert "should not contain" in str(exc_info.value).lower()


class TestExpectIs:
    """Tests for expect_is assertion."""

    def test_same_object_passes(self):
        """expect_is should pass for identical objects."""
        obj = object()
        expect_is(obj, obj)
        expect_is(None, None)

    def test_different_objects_fail(self):
        """expect_is should fail for different objects."""
        with pytest.raises(TestFailure) as exc_info:
            expect_is([1, 2], [1, 2])  # Equal but not identical
        assert "not identical" in str(exc_info.value).lower()


class TestExpectIsNot:
    """Tests for expect_is_not assertion."""

    def test_different_objects_pass(self):
        """expect_is_not should pass for different objects."""
        expect_is_not([1, 2], [1, 2])
        expect_is_not(1, 1.0)

    def test_same_object_fails(self):
        """expect_is_not should fail for identical objects."""
        obj = object()
        with pytest.raises(TestFailure):
            expect_is_not(obj, obj)


class TestExpectNone:
    """Tests for expect_none assertion."""

    def test_none_passes(self):
        """expect_none should pass for None."""
        expect_none(None)

    def test_not_none_fails(self):
        """expect_none should fail for non-None values."""
        with pytest.raises(TestFailure):
            expect_none(0)
        with pytest.raises(TestFailure):
            expect_none("")
        with pytest.raises(TestFailure):
            expect_none(False)


class TestExpectNotNone:
    """Tests for expect_not_none assertion."""

    def test_not_none_passes(self):
        """expect_not_none should pass for non-None values."""
        expect_not_none(0)
        expect_not_none("")
        expect_not_none(False)

    def test_returns_value(self):
        """expect_not_none should return the value."""
        result = expect_not_none(42)
        assert result == 42

    def test_none_fails(self):
        """expect_not_none should fail for None."""
        with pytest.raises(TestFailure):
            expect_not_none(None)


class TestExpectGreater:
    """Tests for expect_greater assertion."""

    def test_greater_passes(self):
        """expect_greater should pass when actual > threshold."""
        expect_greater(5, 3)
        expect_greater(1.5, 1.0)

    def test_equal_fails(self):
        """expect_greater should fail when equal."""
        with pytest.raises(TestFailure):
            expect_greater(5, 5)

    def test_less_fails(self):
        """expect_greater should fail when less."""
        with pytest.raises(TestFailure):
            expect_greater(3, 5)


class TestExpectGreaterEq:
    """Tests for expect_greater_eq assertion."""

    def test_greater_passes(self):
        """expect_greater_eq should pass when actual > threshold."""
        expect_greater_eq(5, 3)

    def test_equal_passes(self):
        """expect_greater_eq should pass when equal."""
        expect_greater_eq(5, 5)

    def test_less_fails(self):
        """expect_greater_eq should fail when less."""
        with pytest.raises(TestFailure):
            expect_greater_eq(3, 5)


class TestExpectLess:
    """Tests for expect_less assertion."""

    def test_less_passes(self):
        """expect_less should pass when actual < threshold."""
        expect_less(3, 5)

    def test_equal_fails(self):
        """expect_less should fail when equal."""
        with pytest.raises(TestFailure):
            expect_less(5, 5)

    def test_greater_fails(self):
        """expect_less should fail when greater."""
        with pytest.raises(TestFailure):
            expect_less(5, 3)


class TestExpectLessEq:
    """Tests for expect_less_eq assertion."""

    def test_less_passes(self):
        """expect_less_eq should pass when actual < threshold."""
        expect_less_eq(3, 5)

    def test_equal_passes(self):
        """expect_less_eq should pass when equal."""
        expect_less_eq(5, 5)

    def test_greater_fails(self):
        """expect_less_eq should fail when greater."""
        with pytest.raises(TestFailure):
            expect_less_eq(5, 3)


class TestExpectInRange:
    """Tests for expect_in_range assertion."""

    def test_in_range_inclusive_passes(self):
        """expect_in_range should pass for values in inclusive range."""
        expect_in_range(5, 0, 10)
        expect_in_range(0, 0, 10)  # At lower bound
        expect_in_range(10, 0, 10)  # At upper bound

    def test_out_of_range_fails(self):
        """expect_in_range should fail for values outside range."""
        with pytest.raises(TestFailure):
            expect_in_range(-1, 0, 10)
        with pytest.raises(TestFailure):
            expect_in_range(11, 0, 10)

    def test_exclusive_range(self):
        """expect_in_range should handle exclusive bounds."""
        expect_in_range(5, 0, 10, inclusive=False)
        with pytest.raises(TestFailure):
            expect_in_range(0, 0, 10, inclusive=False)  # At lower bound
        with pytest.raises(TestFailure):
            expect_in_range(10, 0, 10, inclusive=False)  # At upper bound


class TestExpectType:
    """Tests for expect_type assertion."""

    def test_exact_type_passes(self):
        """expect_type should pass for exact type match."""
        expect_type(42, int)
        expect_type("hello", str)
        expect_type([1, 2], list)

    def test_subclass_fails(self):
        """expect_type should fail for subclass."""
        # bool is a subclass of int
        with pytest.raises(TestFailure):
            expect_type(True, int)

    def test_wrong_type_fails(self):
        """expect_type should fail for wrong type."""
        with pytest.raises(TestFailure):
            expect_type(42, str)


class TestExpectInstance:
    """Tests for expect_instance assertion."""

    def test_exact_type_passes(self):
        """expect_instance should pass for exact type."""
        expect_instance(42, int)

    def test_subclass_passes(self):
        """expect_instance should pass for subclasses."""
        expect_instance(True, int)  # bool is subclass of int

        class CustomError(ValueError):
            pass
        expect_instance(CustomError(), ValueError)

    def test_tuple_of_types(self):
        """expect_instance should accept tuple of types."""
        expect_instance(42, (int, float))
        expect_instance(3.14, (int, float))
        with pytest.raises(TestFailure):
            expect_instance("hello", (int, float))

    def test_wrong_type_fails(self):
        """expect_instance should fail for unrelated types."""
        with pytest.raises(TestFailure):
            expect_instance(42, str)

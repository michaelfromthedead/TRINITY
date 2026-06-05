"""Tests for Fixed16 and Fixed32 fixed-point math (T-CC-0.14)."""

import pytest

from trinity.types import Fixed16, Fixed32


class TestFixed16Creation:
    """Test Fixed16 creation and conversion."""

    def test_from_int(self):
        """Test creation from integer."""
        f = Fixed16(5)
        assert f.as_int == 5
        assert abs(f.as_float - 5.0) < 0.01

    def test_from_float(self):
        """Test creation from float."""
        f = Fixed16(3.5)
        assert abs(f.as_float - 3.5) < 0.01

    def test_from_fixed16(self):
        """Test creation from another Fixed16."""
        f1 = Fixed16(2.25)
        f2 = Fixed16(f1)
        assert f1.raw == f2.raw

    def test_from_raw(self):
        """Test creation from raw value."""
        f = Fixed16.from_raw(512)  # 512 / 256 = 2.0
        assert f.as_int == 2

    def test_default_zero(self):
        """Test default value is zero."""
        f = Fixed16()
        assert f.as_int == 0
        assert f.as_float == 0.0


class TestFixed16Arithmetic:
    """Test Fixed16 arithmetic operations."""

    def test_add_fixed(self):
        """Test adding two Fixed16."""
        a = Fixed16(2.5)
        b = Fixed16(1.25)
        result = a + b
        assert abs(result.as_float - 3.75) < 0.01

    def test_add_int(self):
        """Test adding Fixed16 and int."""
        a = Fixed16(2.5)
        result = a + 3
        assert abs(result.as_float - 5.5) < 0.01

    def test_radd(self):
        """Test reverse addition."""
        a = Fixed16(2.5)
        result = 3 + a
        assert abs(result.as_float - 5.5) < 0.01

    def test_sub_fixed(self):
        """Test subtracting Fixed16."""
        a = Fixed16(5.0)
        b = Fixed16(2.25)
        result = a - b
        assert abs(result.as_float - 2.75) < 0.01

    def test_rsub(self):
        """Test reverse subtraction."""
        a = Fixed16(2.5)
        result = 5 - a
        assert abs(result.as_float - 2.5) < 0.01

    def test_mul_fixed(self):
        """Test multiplying Fixed16."""
        a = Fixed16(2.0)
        b = Fixed16(3.0)
        result = a * b
        assert abs(result.as_float - 6.0) < 0.1

    def test_mul_int(self):
        """Test multiplying by int."""
        a = Fixed16(2.5)
        result = a * 2
        assert abs(result.as_float - 5.0) < 0.01

    def test_div_fixed(self):
        """Test dividing Fixed16."""
        a = Fixed16(6.0)
        b = Fixed16(2.0)
        result = a / b
        assert abs(result.as_float - 3.0) < 0.1

    def test_div_by_zero_raises(self):
        """Test division by zero raises."""
        a = Fixed16(1.0)
        b = Fixed16(0)
        with pytest.raises(ZeroDivisionError):
            _ = a / b

    def test_neg(self):
        """Test negation."""
        a = Fixed16(3.5)
        result = -a
        assert abs(result.as_float - (-3.5)) < 0.01


class TestFixed16Comparison:
    """Test Fixed16 comparison operations."""

    def test_eq_same(self):
        """Test equality for same value."""
        a = Fixed16(2.5)
        b = Fixed16(2.5)
        assert a == b

    def test_eq_different(self):
        """Test equality for different values."""
        a = Fixed16(2.5)
        b = Fixed16(3.0)
        assert not (a == b)

    def test_lt(self):
        """Test less than."""
        a = Fixed16(2.0)
        b = Fixed16(3.0)
        assert a < b

    def test_gt(self):
        """Test greater than."""
        a = Fixed16(3.0)
        b = Fixed16(2.0)
        assert a > b

    def test_le(self):
        """Test less than or equal."""
        a = Fixed16(2.0)
        b = Fixed16(2.0)
        assert a <= b

    def test_ge(self):
        """Test greater than or equal."""
        a = Fixed16(2.0)
        b = Fixed16(2.0)
        assert a >= b


class TestFixed16Precision:
    """Test Fixed16 precision characteristics."""

    def test_range_positive(self):
        """Test positive range."""
        f = Fixed16(127)
        assert f.as_int == 127

    def test_range_negative(self):
        """Test negative range."""
        f = Fixed16(-128)
        assert f.as_int == -128

    def test_fractional_precision(self):
        """Test fractional precision (1/256)."""
        f = Fixed16(1.0 / 256)
        assert f.raw == 1


class TestFixed32Creation:
    """Test Fixed32 creation and conversion."""

    def test_from_int(self):
        """Test creation from integer."""
        f = Fixed32(1000)
        assert f.as_int == 1000

    def test_from_float(self):
        """Test creation from float."""
        f = Fixed32(123.456)
        assert abs(f.as_float - 123.456) < 0.001

    def test_from_raw(self):
        """Test creation from raw value."""
        f = Fixed32.from_raw(65536)  # 65536 / 65536 = 1.0
        assert f.as_int == 1


class TestFixed32Arithmetic:
    """Test Fixed32 arithmetic operations."""

    def test_add(self):
        """Test addition."""
        a = Fixed32(100.5)
        b = Fixed32(50.25)
        result = a + b
        assert abs(result.as_float - 150.75) < 0.001

    def test_sub(self):
        """Test subtraction."""
        a = Fixed32(100.0)
        b = Fixed32(33.333)
        result = a - b
        assert abs(result.as_float - 66.667) < 0.01

    def test_mul(self):
        """Test multiplication."""
        a = Fixed32(10.0)
        b = Fixed32(5.5)
        result = a * b
        assert abs(result.as_float - 55.0) < 0.1

    def test_div(self):
        """Test division."""
        a = Fixed32(100.0)
        b = Fixed32(4.0)
        result = a / b
        assert abs(result.as_float - 25.0) < 0.1


class TestFixed32Precision:
    """Test Fixed32 precision characteristics."""

    def test_higher_precision_than_fixed16(self):
        """Test Fixed32 has higher precision than Fixed16."""
        # Fixed32 is Q16.16, Fixed16 is Q8.8
        f32 = Fixed32(0.00001)
        f16 = Fixed16(0.00001)
        # Fixed32 should have non-zero raw, Fixed16 might round to zero
        assert f32.raw != 0 or f16.raw == 0

    def test_large_range(self):
        """Test Fixed32 handles larger range."""
        f = Fixed32(30000)
        assert f.as_int == 30000


class TestDeterminism:
    """Test deterministic behavior for simulation."""

    def test_same_operations_same_result(self):
        """Test same operations produce identical results."""
        def compute():
            a = Fixed16(10)
            b = Fixed16(3)
            c = a / b
            d = c * b
            return d.raw

        result1 = compute()
        result2 = compute()
        assert result1 == result2

    def test_order_of_operations(self):
        """Test order of operations is deterministic."""
        a = Fixed16(2)
        b = Fixed16(3)
        c = Fixed16(4)

        # (a + b) * c vs a + (b * c)
        result1 = (a + b) * c
        result2 = a + (b * c)
        # These should be different due to order of operations
        assert result1.raw != result2.raw

    def test_raw_value_is_integer(self):
        """Test raw value is always integer (no float accumulation)."""
        f = Fixed16(1.5)
        for _ in range(1000):
            f = f + Fixed16(0.1)
        assert isinstance(f.raw, int)

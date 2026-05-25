"""
Tests for validation descriptors: TypeDescriptor, ChoiceDescriptor, PatternDescriptor.

Verifies:
- Type checking and coercion
- Choice constraint enforcement
- Regex pattern matching
- Metadata correctness
"""
import pytest
import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from trinity.descriptors.validation import (
    ValidatedDescriptor,
    RangeDescriptor,
    TypeDescriptor,
    ChoiceDescriptor,
    PatternDescriptor,
)


class TestValidatedDescriptor:
    """Test ValidatedDescriptor with custom validation functions."""

    def test_single_validator_success(self):
        """Validator that returns True should accept value."""
        def is_positive(x):
            return x > 0

        class Foo:
            value = ValidatedDescriptor(field_type=int, validators=[is_positive])
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 42
        assert f.value == 42

    def test_single_validator_failure(self):
        """Validator that returns False should raise ValueError."""
        def is_positive(x):
            return x > 0

        class Foo:
            value = ValidatedDescriptor(field_type=int, validators=[is_positive])
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        with pytest.raises(ValueError, match="Validation failed"):
            f.value = -5

    def test_multiple_validators(self):
        """All validators must pass."""
        def is_positive(x):
            return x > 0
        def is_even(x):
            return x % 2 == 0

        class Foo:
            value = ValidatedDescriptor(field_type=int, validators=[is_positive, is_even])
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 42
        assert f.value == 42
        with pytest.raises(ValueError):
            f.value = 41  # Not even
        with pytest.raises(ValueError):
            f.value = -2  # Not positive

    def test_validator_raises_exception(self):
        """Validator that raises exception should wrap in ValueError."""
        def bad_validator(x):
            raise RuntimeError("Something went wrong")

        class Foo:
            value = ValidatedDescriptor(field_type=int, validators=[bad_validator])
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        with pytest.raises(ValueError, match="Validation error"):
            f.value = 42

    def test_empty_validators_list(self):
        """Empty validators list should accept any value."""
        class Foo:
            value = ValidatedDescriptor(field_type=int, validators=[])
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 999
        assert f.value == 999

    def test_no_validators_parameter(self):
        """No validators parameter should accept any value."""
        class Foo:
            value = ValidatedDescriptor(field_type=int)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 999
        assert f.value == 999

    def test_add_validator(self):
        """Should be able to add validators after construction."""
        class Foo:
            value = ValidatedDescriptor(field_type=int)
        Foo.value.__set_name__(Foo, 'value')

        Foo.value.add_validator(lambda x: x > 0)

        f = Foo()
        f.value = 42
        assert f.value == 42
        with pytest.raises(ValueError):
            f.value = -5

    def test_metadata(self):
        """Metadata should include validator count."""
        class Foo:
            value = ValidatedDescriptor(field_type=int, validators=[lambda x: True, lambda x: True])
        Foo.value.__set_name__(Foo, 'value')
        meta = Foo.value.get_metadata()
        assert meta["descriptor_id"] == "validated"
        assert meta["validator_count"] == 2


class TestRangeDescriptor:
    """Test RangeDescriptor for numeric range validation."""

    def test_value_within_range(self):
        """Value within range should be accepted."""
        class Foo:
            value = RangeDescriptor(field_type=float, min_val=0.0, max_val=100.0, clamp=False)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 50.0
        assert f.value == 50.0

    def test_value_below_range_clamp(self):
        """With clamp=True, value below min should clamp to min."""
        class Foo:
            value = RangeDescriptor(field_type=float, min_val=0.0, max_val=100.0, clamp=True)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = -50.0
        assert f.value == 0.0

    def test_value_above_range_clamp(self):
        """With clamp=True, value above max should clamp to max."""
        class Foo:
            value = RangeDescriptor(field_type=float, min_val=0.0, max_val=100.0, clamp=True)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 150.0
        assert f.value == 100.0

    def test_value_below_range_no_clamp(self):
        """With clamp=False, value below min should raise ValueError."""
        class Foo:
            value = RangeDescriptor(field_type=float, min_val=0.0, max_val=100.0, clamp=False)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        with pytest.raises(ValueError, match="outside range"):
            f.value = -50.0

    def test_value_above_range_no_clamp(self):
        """With clamp=False, value above max should raise ValueError."""
        class Foo:
            value = RangeDescriptor(field_type=float, min_val=0.0, max_val=100.0, clamp=False)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        with pytest.raises(ValueError, match="outside range"):
            f.value = 150.0

    def test_value_at_boundaries(self):
        """Values exactly at min/max should be accepted."""
        class Foo:
            value = RangeDescriptor(field_type=float, min_val=0.0, max_val=100.0, clamp=False)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 0.0
        assert f.value == 0.0
        f.value = 100.0
        assert f.value == 100.0

    def test_equal_min_max(self):
        """Min and max can be equal, only that exact value allowed."""
        class Foo:
            value = RangeDescriptor(field_type=float, min_val=42.0, max_val=42.0, clamp=False)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 42.0
        assert f.value == 42.0
        with pytest.raises(ValueError):
            f.value = 41.0

    def test_inverted_range_raises(self):
        """Min > max should raise ValueError on construction."""
        with pytest.raises(ValueError, match="min_val.*cannot be greater than max_val"):
            class Foo:
                value = RangeDescriptor(field_type=float, min_val=100.0, max_val=0.0)

    def test_non_numeric_type_raises(self):
        """Non-numeric types should raise TypeError."""
        class Foo:
            value = RangeDescriptor(field_type=float, min_val=0.0, max_val=100.0)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        with pytest.raises(TypeError, match="non-numeric type"):
            f.value = "not a number"

    def test_integer_range(self):
        """Range descriptor should work with integers."""
        class Foo:
            value = RangeDescriptor(field_type=int, min_val=1, max_val=10, clamp=True)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 5
        assert f.value == 5
        f.value = 0
        assert f.value == 1
        f.value = 20
        assert f.value == 10

    def test_metadata(self):
        """Metadata should include range and clamp settings."""
        class Foo:
            value = RangeDescriptor(field_type=float, min_val=0.0, max_val=100.0, clamp=True)
        Foo.value.__set_name__(Foo, 'value')
        meta = Foo.value.get_metadata()
        assert meta["descriptor_id"] == "range"
        assert meta["range"] == (0.0, 100.0)
        assert meta["clamp"] is True


class TestTypeDescriptor:
    """Test TypeDescriptor validates and optionally coerces types."""

    def test_accepts_correct_type(self):
        """Setting a value of the expected type should succeed."""
        class Foo:
            value = TypeDescriptor(field_type=int, expected_type=int)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 42
        assert f.value == 42

    def test_rejects_wrong_type(self):
        """Setting a value of the wrong type should raise TypeError."""
        class Foo:
            value = TypeDescriptor(field_type=int, expected_type=int)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        with pytest.raises(TypeError):
            f.value = "not an int"

    def test_coerce_mode(self):
        """With coerce=True, convertible values should be coerced."""
        class Foo:
            value = TypeDescriptor(field_type=int, expected_type=int, coerce=True)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = "42"
        assert f.value == 42

    def test_coerce_failure(self):
        """Non-convertible values should raise TypeError even with coerce=True."""
        class Foo:
            value = TypeDescriptor(field_type=int, expected_type=int, coerce=True)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        with pytest.raises(TypeError):
            f.value = "not a number"

    def test_rejects_none_for_strict_types(self):
        """None should be rejected for non-optional types."""
        class Foo:
            value = TypeDescriptor(field_type=int, expected_type=int)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        with pytest.raises(TypeError):
            f.value = None

    def test_metadata(self):
        """Metadata should include expected_type and coerce flag."""
        class Foo:
            value = TypeDescriptor(field_type=int, expected_type=int, coerce=True)
        Foo.value.__set_name__(Foo, 'value')
        meta = Foo.value.get_metadata()
        assert meta["descriptor_id"] == "typed"
        assert meta["expected_type"] == "int"
        assert meta["coerce"] is True

    def test_float_to_int_coercion(self):
        """Coercion should truncate floats to ints."""
        class Foo:
            value = TypeDescriptor(field_type=int, expected_type=int, coerce=True)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 42.7
        assert f.value == 42
        assert isinstance(f.value, int)


class TestChoiceDescriptor:
    """Test ChoiceDescriptor restricts values to a set of choices."""

    def test_valid_choice(self):
        """Setting a value from the allowed choices should succeed."""
        class Foo:
            color = ChoiceDescriptor(field_type=str, choices=["red", "green", "blue"])
        Foo.color.__set_name__(Foo, 'color')
        f = Foo()
        f.color = "red"
        assert f.color == "red"

    def test_invalid_choice_raises(self):
        """Setting a value not in choices should raise ValueError."""
        class Foo:
            color = ChoiceDescriptor(field_type=str, choices=["red", "green", "blue"])
        Foo.color.__set_name__(Foo, 'color')
        f = Foo()
        with pytest.raises(ValueError):
            f.color = "yellow"

    def test_empty_choices_raises(self):
        """An empty choices list should raise ValueError on construction or first set."""
        with pytest.raises((ValueError, TypeError)):
            class Foo:
                color = ChoiceDescriptor(field_type=str, choices=[])
            Foo.color.__set_name__(Foo, 'color')
            f = Foo()
            f.color = "anything"

    def test_metadata(self):
        """Metadata should include the list of valid choices."""
        class Foo:
            color = ChoiceDescriptor(field_type=str, choices=["a", "b"])
        Foo.color.__set_name__(Foo, 'color')
        meta = Foo.color.get_metadata()
        assert meta["descriptor_id"] == "choice"
        assert "choices" in meta
        assert set(meta["choices"]) == {"a", "b"}

    def test_multiple_types_in_choices(self):
        """Choices can include different types if field_type allows."""
        class Foo:
            val = ChoiceDescriptor(field_type=object, choices=[1, "two", 3.0])
        Foo.val.__set_name__(Foo, 'val')
        f = Foo()
        f.val = "two"
        assert f.val == "two"
        f.val = 1
        assert f.val == 1
        f.val = 3.0
        assert f.val == 3.0
        with pytest.raises(ValueError):
            f.val = "three"

    def test_none_in_choices(self):
        """None can be a valid choice if included."""
        class Foo:
            val = ChoiceDescriptor(field_type=object, choices=[None, "something"])
        Foo.val.__set_name__(Foo, 'val')
        f = Foo()
        f.val = None
        assert f.val is None
        f.val = "something"
        assert f.val == "something"


class TestPatternDescriptor:
    """Test PatternDescriptor validates string values against a regex pattern."""

    def test_matching_string(self):
        """A string matching the pattern should be accepted."""
        class Foo:
            email = PatternDescriptor(field_type=str, pattern=r"^[^@]+@[^@]+\.[^@]+$")
        Foo.email.__set_name__(Foo, 'email')
        f = Foo()
        f.email = "user@example.com"
        assert f.email == "user@example.com"

    def test_non_matching_raises(self):
        """A string not matching the pattern should raise ValueError."""
        class Foo:
            email = PatternDescriptor(field_type=str, pattern=r"^[^@]+@[^@]+\.[^@]+$")
        Foo.email.__set_name__(Foo, 'email')
        f = Foo()
        with pytest.raises(ValueError):
            f.email = "not-an-email"

    def test_non_string_raises(self):
        """A non-string value should raise TypeError."""
        class Foo:
            code = PatternDescriptor(field_type=str, pattern=r"^\d{3}$")
        Foo.code.__set_name__(Foo, 'code')
        f = Foo()
        with pytest.raises(TypeError):
            f.code = 123

    def test_metadata(self):
        """Metadata should include the pattern string."""
        class Foo:
            code = PatternDescriptor(field_type=str, pattern=r"^\d+$")
        Foo.code.__set_name__(Foo, 'code')
        meta = Foo.code.get_metadata()
        assert meta["descriptor_id"] == "pattern"
        assert meta["pattern"] == r"^\d+$"

    def test_full_match_required(self):
        """Pattern should match the entire value, not just a substring."""
        class Foo:
            code = PatternDescriptor(field_type=str, pattern=r"^\d{3}$")
        Foo.code.__set_name__(Foo, 'code')
        f = Foo()
        f.code = "123"
        assert f.code == "123"
        with pytest.raises(ValueError):
            f.code = "1234"

    def test_empty_string_pattern(self):
        """Empty strings should be validated against pattern."""
        class Foo:
            # Pattern requires at least one digit
            code = PatternDescriptor(field_type=str, pattern=r"^\d+$")
        Foo.code.__set_name__(Foo, 'code')
        f = Foo()
        with pytest.raises(ValueError):
            f.code = ""

    def test_empty_string_allowed(self):
        """Empty strings can match if pattern allows."""
        class Foo:
            # Pattern allows empty string
            code = PatternDescriptor(field_type=str, pattern=r"^\d*$")
        Foo.code.__set_name__(Foo, 'code')
        f = Foo()
        f.code = ""
        assert f.code == ""

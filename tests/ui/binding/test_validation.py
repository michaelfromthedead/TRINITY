"""
Comprehensive tests for input validation.

Tests cover:
- RequiredValidator
- RangeValidator
- RegexValidator
- LengthValidator
- EmailValidator
- UrlValidator
- ChoiceValidator
- TypeValidator
- CustomValidator
- AsyncCustomValidator
- CompareValidator
- CompositeValidator
- ValidationContext
- Factory functions
"""
import asyncio
import re

import pytest

from engine.ui.binding.validation import (
    AsyncCustomValidator,
    ChoiceValidator,
    CompareValidator,
    CompositeValidator,
    CustomValidator,
    EmailValidator,
    IAsyncValidator,
    IValidator,
    LengthValidator,
    RangeValidator,
    RegexValidator,
    RequiredValidator,
    TypeValidator,
    UrlValidator,
    ValidationContext,
    ValidationResult,
    ValidationSeverity,
    ValidationTrigger,
    all_of,
    any_of,
    custom,
    email,
    length,
    range_validator,
    regex,
    required,
)


# ========== Fixtures ==========


@pytest.fixture
def validation_context():
    """Create a validation context."""
    return ValidationContext()


# ========== ValidationResult Tests ==========


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_valid_factory(self):
        """Test creating valid result."""
        result = ValidationResult.valid()
        assert result.is_valid is True
        assert result.error_message is None

    def test_invalid_factory(self):
        """Test creating invalid result."""
        result = ValidationResult.invalid("Error message")
        assert result.is_valid is False
        assert result.error_message == "Error message"
        assert result.severity == ValidationSeverity.ERROR

    def test_invalid_with_severity(self):
        """Test invalid with custom severity."""
        result = ValidationResult.invalid(
            "Warning",
            severity=ValidationSeverity.WARNING
        )
        assert result.severity == ValidationSeverity.WARNING

    def test_warning_factory(self):
        """Test creating warning result."""
        result = ValidationResult.warning("Warning message")
        assert result.is_valid is True
        assert result.error_message == "Warning message"
        assert result.severity == ValidationSeverity.WARNING

    def test_metadata(self):
        """Test result with metadata."""
        result = ValidationResult.invalid("Error", field="email", code=100)
        assert result.metadata["field"] == "email"
        assert result.metadata["code"] == 100

    def test_bool_true(self):
        """Test valid result is truthy."""
        result = ValidationResult.valid()
        assert bool(result) is True

    def test_bool_false(self):
        """Test invalid result is falsy."""
        result = ValidationResult.invalid("Error")
        assert bool(result) is False


# ========== RequiredValidator Tests ==========


class TestRequiredValidator:
    """Tests for RequiredValidator."""

    def test_valid_string(self):
        """Test non-empty string is valid."""
        validator = RequiredValidator()
        result = validator.validate("hello")
        assert result.is_valid

    def test_invalid_none(self):
        """Test None is invalid."""
        validator = RequiredValidator()
        result = validator.validate(None)
        assert not result.is_valid

    def test_invalid_empty_string(self):
        """Test empty string is invalid."""
        validator = RequiredValidator()
        result = validator.validate("")
        assert not result.is_valid

    def test_invalid_whitespace_only(self):
        """Test whitespace-only is invalid by default."""
        validator = RequiredValidator()
        result = validator.validate("   ")
        assert not result.is_valid

    def test_allow_whitespace(self):
        """Test allowing whitespace-only."""
        validator = RequiredValidator(allow_whitespace=True)
        result = validator.validate("   ")
        assert result.is_valid

    def test_invalid_empty_list(self):
        """Test empty list is invalid."""
        validator = RequiredValidator()
        result = validator.validate([])
        assert not result.is_valid

    def test_invalid_empty_dict(self):
        """Test empty dict is invalid."""
        validator = RequiredValidator()
        result = validator.validate({})
        assert not result.is_valid

    def test_valid_list(self):
        """Test non-empty list is valid."""
        validator = RequiredValidator()
        result = validator.validate([1, 2])
        assert result.is_valid

    def test_custom_message(self):
        """Test custom error message."""
        validator = RequiredValidator(message="Field is required")
        result = validator.validate(None)
        assert result.error_message == "Field is required"

    def test_trigger_property(self):
        """Test trigger property."""
        validator = RequiredValidator(trigger=ValidationTrigger.ON_SUBMIT)
        assert validator.trigger == ValidationTrigger.ON_SUBMIT


# ========== RangeValidator Tests ==========


class TestRangeValidator:
    """Tests for RangeValidator."""

    def test_valid_in_range(self):
        """Test value in range is valid."""
        validator = RangeValidator(min_value=0, max_value=100)
        result = validator.validate(50)
        assert result.is_valid

    def test_valid_at_min(self):
        """Test value at min is valid (inclusive)."""
        validator = RangeValidator(min_value=0, max_value=100)
        result = validator.validate(0)
        assert result.is_valid

    def test_valid_at_max(self):
        """Test value at max is valid (inclusive)."""
        validator = RangeValidator(min_value=0, max_value=100)
        result = validator.validate(100)
        assert result.is_valid

    def test_invalid_below_min(self):
        """Test value below min is invalid."""
        validator = RangeValidator(min_value=0)
        result = validator.validate(-1)
        assert not result.is_valid
        assert "at least" in result.error_message

    def test_invalid_above_max(self):
        """Test value above max is invalid."""
        validator = RangeValidator(max_value=100)
        result = validator.validate(101)
        assert not result.is_valid
        assert "at most" in result.error_message

    def test_exclusive_min(self):
        """Test exclusive minimum."""
        validator = RangeValidator(min_value=0, min_inclusive=False)
        result = validator.validate(0)
        assert not result.is_valid
        assert "greater than" in result.error_message

    def test_exclusive_max(self):
        """Test exclusive maximum."""
        validator = RangeValidator(max_value=100, max_inclusive=False)
        result = validator.validate(100)
        assert not result.is_valid
        assert "less than" in result.error_message

    def test_none_allowed(self):
        """Test None is allowed (use RequiredValidator)."""
        validator = RangeValidator(min_value=0, max_value=100)
        result = validator.validate(None)
        assert result.is_valid

    def test_invalid_type(self):
        """Test non-numeric type is invalid."""
        validator = RangeValidator(min_value=0, max_value=100)
        result = validator.validate("not a number")
        assert not result.is_valid

    def test_custom_message(self):
        """Test custom error message."""
        validator = RangeValidator(min_value=0, message="Value out of range")
        result = validator.validate(-1)
        assert result.error_message == "Value out of range"


# ========== RegexValidator Tests ==========


class TestRegexValidator:
    """Tests for RegexValidator."""

    def test_valid_match(self):
        """Test matching value is valid."""
        validator = RegexValidator(r"^\d{3}$")
        result = validator.validate("123")
        assert result.is_valid

    def test_invalid_no_match(self):
        """Test non-matching value is invalid."""
        validator = RegexValidator(r"^\d{3}$")
        result = validator.validate("12")
        assert not result.is_valid

    def test_none_allowed(self):
        """Test None is allowed."""
        validator = RegexValidator(r"^\d+$")
        result = validator.validate(None)
        assert result.is_valid

    def test_empty_allowed(self):
        """Test empty string is allowed."""
        validator = RegexValidator(r"^\d+$")
        result = validator.validate("")
        assert result.is_valid

    def test_must_not_match(self):
        """Test must_match=False inverts logic."""
        validator = RegexValidator(r"\d", must_match=False)
        result = validator.validate("abc")
        assert result.is_valid
        result = validator.validate("abc123")
        assert not result.is_valid

    def test_compiled_pattern(self):
        """Test using compiled pattern."""
        pattern = re.compile(r"^\d+$")
        validator = RegexValidator(pattern)
        result = validator.validate("123")
        assert result.is_valid

    def test_flags(self):
        """Test pattern flags."""
        validator = RegexValidator(r"^abc$", flags=re.IGNORECASE)
        result = validator.validate("ABC")
        assert result.is_valid

    def test_custom_message(self):
        """Test custom error message."""
        validator = RegexValidator(r"^\d+$", message="Numbers only")
        result = validator.validate("abc")
        assert result.error_message == "Numbers only"


# ========== LengthValidator Tests ==========


class TestLengthValidator:
    """Tests for LengthValidator."""

    def test_valid_length(self):
        """Test string with valid length."""
        validator = LengthValidator(min_length=3, max_length=10)
        result = validator.validate("hello")
        assert result.is_valid

    def test_invalid_too_short(self):
        """Test string too short is invalid."""
        validator = LengthValidator(min_length=5)
        result = validator.validate("hi")
        assert not result.is_valid
        assert "at least" in result.error_message

    def test_invalid_too_long(self):
        """Test string too long is invalid."""
        validator = LengthValidator(max_length=5)
        result = validator.validate("hello world")
        assert not result.is_valid
        assert "at most" in result.error_message

    def test_exact_length_valid(self):
        """Test exact length match."""
        validator = LengthValidator(exact_length=5)
        result = validator.validate("hello")
        assert result.is_valid

    def test_exact_length_invalid(self):
        """Test exact length mismatch."""
        validator = LengthValidator(exact_length=5)
        result = validator.validate("hi")
        assert not result.is_valid
        assert "exactly" in result.error_message

    def test_none_allowed(self):
        """Test None is allowed."""
        validator = LengthValidator(min_length=1)
        result = validator.validate(None)
        assert result.is_valid

    def test_list_length(self):
        """Test validating list length."""
        validator = LengthValidator(min_length=2, max_length=5)
        result = validator.validate([1, 2, 3])
        assert result.is_valid

    def test_dict_length(self):
        """Test validating dict length."""
        validator = LengthValidator(min_length=1)
        result = validator.validate({"a": 1})
        assert result.is_valid

    def test_no_length_property(self):
        """Test value without length is invalid."""
        validator = LengthValidator(min_length=1)
        result = validator.validate(123)  # int has no len()
        assert not result.is_valid


# ========== EmailValidator Tests ==========


class TestEmailValidator:
    """Tests for EmailValidator."""

    def test_valid_email(self):
        """Test valid email addresses."""
        validator = EmailValidator()
        valid_emails = [
            "test@example.com",
            "user.name@domain.org",
            "user+tag@example.co.uk",
        ]
        for email in valid_emails:
            result = validator.validate(email)
            assert result.is_valid, f"Expected {email} to be valid"

    def test_invalid_email(self):
        """Test invalid email addresses."""
        validator = EmailValidator()
        invalid_emails = [
            "notanemail",
            "missing@domain",
            "@nodomain.com",
            "spaces in@email.com",
        ]
        for email in invalid_emails:
            result = validator.validate(email)
            assert not result.is_valid, f"Expected {email} to be invalid"

    def test_none_allowed(self):
        """Test None is allowed."""
        validator = EmailValidator()
        result = validator.validate(None)
        assert result.is_valid

    def test_empty_allowed(self):
        """Test empty string is allowed."""
        validator = EmailValidator()
        result = validator.validate("")
        assert result.is_valid

    def test_custom_message(self):
        """Test custom error message."""
        validator = EmailValidator(message="Bad email")
        result = validator.validate("invalid")
        assert result.error_message == "Bad email"


# ========== UrlValidator Tests ==========


class TestUrlValidator:
    """Tests for UrlValidator."""

    def test_valid_http_url(self):
        """Test valid HTTP URL."""
        validator = UrlValidator()
        result = validator.validate("http://example.com")
        assert result.is_valid

    def test_valid_https_url(self):
        """Test valid HTTPS URL."""
        validator = UrlValidator()
        result = validator.validate("https://example.com")
        assert result.is_valid

    def test_valid_url_with_path(self):
        """Test URL with path."""
        validator = UrlValidator()
        result = validator.validate("https://example.com/path/to/page")
        assert result.is_valid

    def test_valid_localhost(self):
        """Test localhost URL."""
        validator = UrlValidator()
        result = validator.validate("http://localhost:8080")
        assert result.is_valid

    def test_invalid_url(self):
        """Test invalid URL."""
        validator = UrlValidator()
        result = validator.validate("not a url")
        assert not result.is_valid

    def test_require_https(self):
        """Test requiring HTTPS."""
        validator = UrlValidator(require_https=True)
        result = validator.validate("http://example.com")
        assert not result.is_valid
        assert "HTTPS" in result.error_message

    def test_none_allowed(self):
        """Test None is allowed."""
        validator = UrlValidator()
        result = validator.validate(None)
        assert result.is_valid


# ========== ChoiceValidator Tests ==========


class TestChoiceValidator:
    """Tests for ChoiceValidator."""

    def test_valid_choice(self):
        """Test value in choices is valid."""
        validator = ChoiceValidator(choices=["a", "b", "c"])
        result = validator.validate("b")
        assert result.is_valid

    def test_invalid_choice(self):
        """Test value not in choices is invalid."""
        validator = ChoiceValidator(choices=["a", "b", "c"])
        result = validator.validate("d")
        assert not result.is_valid
        assert "must be one of" in result.error_message

    def test_none_allowed(self):
        """Test None is allowed."""
        validator = ChoiceValidator(choices=["a", "b"])
        result = validator.validate(None)
        assert result.is_valid

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        validator = ChoiceValidator(choices=["Yes", "No"], case_sensitive=False)
        result = validator.validate("yes")
        assert result.is_valid

    def test_custom_message(self):
        """Test custom error message."""
        validator = ChoiceValidator(choices=["a", "b"], message="Invalid option")
        result = validator.validate("c")
        assert result.error_message == "Invalid option"


# ========== TypeValidator Tests ==========


class TestTypeValidator:
    """Tests for TypeValidator."""

    def test_valid_type(self):
        """Test value of expected type is valid."""
        validator = TypeValidator(expected_type=str)
        result = validator.validate("hello")
        assert result.is_valid

    def test_invalid_type(self):
        """Test value of wrong type is invalid."""
        validator = TypeValidator(expected_type=str)
        result = validator.validate(123)
        assert not result.is_valid
        assert "must be of type" in result.error_message

    def test_none_allowed(self):
        """Test None is allowed."""
        validator = TypeValidator(expected_type=str)
        result = validator.validate(None)
        assert result.is_valid

    def test_subclass_valid(self):
        """Test subclass is valid."""
        validator = TypeValidator(expected_type=Exception)
        result = validator.validate(ValueError("test"))
        assert result.is_valid


# ========== CustomValidator Tests ==========


class TestCustomValidator:
    """Tests for CustomValidator."""

    def test_bool_function_true(self):
        """Test function returning True."""
        validator = CustomValidator(lambda v: v > 0)
        result = validator.validate(5)
        assert result.is_valid

    def test_bool_function_false(self):
        """Test function returning False."""
        validator = CustomValidator(lambda v: v > 0, message="Must be positive")
        result = validator.validate(-1)
        assert not result.is_valid
        assert result.error_message == "Must be positive"

    def test_validation_result_function(self):
        """Test function returning ValidationResult."""
        def validate(v):
            if v < 0:
                return ValidationResult.invalid("Negative!")
            return ValidationResult.valid()

        validator = CustomValidator(validate)
        result = validator.validate(-1)
        assert not result.is_valid
        assert result.error_message == "Negative!"

    def test_string_function(self):
        """Test function returning error string."""
        validator = CustomValidator(
            lambda v: "" if v > 0 else "Must be positive"
        )
        result = validator.validate(-1)
        assert not result.is_valid

    def test_exception_caught(self):
        """Test exception in function is caught."""
        def bad_validator(v):
            raise ValueError("Validation error")

        validator = CustomValidator(bad_validator)
        result = validator.validate(1)
        assert not result.is_valid
        assert "Validation error" in result.error_message


# ========== AsyncCustomValidator Tests ==========


class TestAsyncCustomValidator:
    """Tests for AsyncCustomValidator."""

    @pytest.mark.asyncio
    async def test_async_validation(self):
        """Test async validation function."""
        async def async_validate(v):
            await asyncio.sleep(0.01)
            return v > 0

        validator = AsyncCustomValidator(async_validate, message="Must be positive")
        result = await validator.validate(5)
        assert result.is_valid

    @pytest.mark.asyncio
    async def test_async_validation_invalid(self):
        """Test async validation returning False."""
        async def async_validate(v):
            return v > 0

        validator = AsyncCustomValidator(async_validate, message="Must be positive")
        result = await validator.validate(-1)
        assert not result.is_valid

    @pytest.mark.asyncio
    async def test_async_exception(self):
        """Test async exception is caught."""
        async def bad_validator(v):
            raise ValueError("Async error")

        validator = AsyncCustomValidator(bad_validator)
        result = await validator.validate(1)
        assert not result.is_valid


# ========== CompareValidator Tests ==========


class TestCompareValidator:
    """Tests for CompareValidator."""

    def test_equal(self):
        """Test equality comparison."""
        validator = CompareValidator(compare_value=5, operator="==")
        assert validator.validate(5).is_valid
        assert not validator.validate(4).is_valid

    def test_not_equal(self):
        """Test inequality comparison."""
        validator = CompareValidator(compare_value=5, operator="!=")
        assert validator.validate(4).is_valid
        assert not validator.validate(5).is_valid

    def test_less_than(self):
        """Test less than comparison."""
        validator = CompareValidator(compare_value=5, operator="<")
        assert validator.validate(4).is_valid
        assert not validator.validate(5).is_valid

    def test_less_than_or_equal(self):
        """Test less than or equal comparison."""
        validator = CompareValidator(compare_value=5, operator="<=")
        assert validator.validate(5).is_valid
        assert not validator.validate(6).is_valid

    def test_greater_than(self):
        """Test greater than comparison."""
        validator = CompareValidator(compare_value=5, operator=">")
        assert validator.validate(6).is_valid
        assert not validator.validate(5).is_valid

    def test_greater_than_or_equal(self):
        """Test greater than or equal comparison."""
        validator = CompareValidator(compare_value=5, operator=">=")
        assert validator.validate(5).is_valid
        assert not validator.validate(4).is_valid

    def test_compare_getter(self):
        """Test comparison with getter function."""
        current_value = [10]
        validator = CompareValidator(
            compare_getter=lambda: current_value[0],
            operator="=="
        )
        assert validator.validate(10).is_valid
        current_value[0] = 20
        assert not validator.validate(10).is_valid


# ========== CompositeValidator Tests ==========


class TestCompositeValidator:
    """Tests for CompositeValidator."""

    def test_and_all_pass(self):
        """Test AND mode with all passing."""
        validator = CompositeValidator([
            RangeValidator(min_value=0),
            RangeValidator(max_value=100),
        ], mode="and")
        result = validator.validate(50)
        assert result.is_valid

    def test_and_one_fails(self):
        """Test AND mode with one failing."""
        validator = CompositeValidator([
            RangeValidator(min_value=0),
            RangeValidator(max_value=10),
        ], mode="and")
        result = validator.validate(50)
        assert not result.is_valid

    def test_and_stop_on_first(self):
        """Test AND mode stops on first failure."""
        call_count = [0]

        class CountingValidator(IValidator):
            def validate(self, value):
                call_count[0] += 1
                return ValidationResult.valid()

        validator = CompositeValidator([
            RangeValidator(max_value=10),
            CountingValidator(),
        ], mode="and", stop_on_first_failure=True)

        validator.validate(50)
        assert call_count[0] == 0  # Second validator not called

    def test_or_one_passes(self):
        """Test OR mode with one passing."""
        validator = CompositeValidator([
            RangeValidator(max_value=10),
            RangeValidator(min_value=90),
        ], mode="or")
        result = validator.validate(5)
        assert result.is_valid

    def test_or_none_pass(self):
        """Test OR mode with none passing."""
        validator = CompositeValidator([
            RangeValidator(max_value=10),
            RangeValidator(min_value=90),
        ], mode="or")
        result = validator.validate(50)
        assert not result.is_valid

    def test_trigger_property(self):
        """Test composite trigger is most common."""
        validators = [
            RequiredValidator(trigger=ValidationTrigger.ON_BLUR),
            RequiredValidator(trigger=ValidationTrigger.ON_BLUR),
            RequiredValidator(trigger=ValidationTrigger.ON_CHANGE),
        ]
        composite = CompositeValidator(validators)
        assert composite.trigger == ValidationTrigger.ON_BLUR


# ========== ValidationContext Tests ==========


class TestValidationContext:
    """Tests for ValidationContext."""

    def test_add_validator(self, validation_context):
        """Test adding a validator."""
        validator = RequiredValidator()
        validation_context.add_validator("email", validator)
        # No error means success

    def test_validate_field(self, validation_context):
        """Test validating a field."""
        validation_context.add_validator("email", RequiredValidator())
        result = validation_context.validate_field("email", "test@example.com")
        assert result.is_valid

    def test_validate_field_invalid(self, validation_context):
        """Test validating an invalid field."""
        validation_context.add_validator(
            "email", RequiredValidator(trigger=ValidationTrigger.ON_CHANGE)
        )
        result = validation_context.validate_field("email", "")
        assert not result.is_valid

    def test_validate_field_sets_field_name(self, validation_context):
        """Test field name is set on result."""
        validation_context.add_validator("email", RequiredValidator())
        result = validation_context.validate_field("email", "")
        assert result.field_name == "email"

    def test_validate_field_respects_trigger(self, validation_context):
        """Test only validators with matching trigger run."""
        validation_context.add_validator(
            "email",
            RequiredValidator(trigger=ValidationTrigger.ON_BLUR)
        )
        # ON_CHANGE trigger should not run the ON_BLUR validator
        result = validation_context.validate_field(
            "email", "", ValidationTrigger.ON_CHANGE
        )
        assert result.is_valid  # Validator didn't run

    @pytest.mark.asyncio
    async def test_validate_field_async(self, validation_context):
        """Test async field validation."""
        async def async_validate(v):
            return len(v) > 0

        validation_context.add_validator(
            "email",
            AsyncCustomValidator(async_validate, trigger=ValidationTrigger.ON_BLUR)
        )
        result = await validation_context.validate_field_async(
            "email", "test", ValidationTrigger.ON_BLUR
        )
        assert result.is_valid

    def test_validate_all(self, validation_context):
        """Test validating all fields."""
        validation_context.add_validator(
            "name",
            RequiredValidator(trigger=ValidationTrigger.ON_SUBMIT)
        )
        validation_context.add_validator(
            "email",
            RequiredValidator(trigger=ValidationTrigger.ON_SUBMIT)
        )

        results = validation_context.validate_all(
            {"name": "John", "email": ""},
            ValidationTrigger.ON_SUBMIT
        )
        assert len(results) == 2
        assert results[0].is_valid  # name
        assert not results[1].is_valid  # email

    def test_is_valid_property(self, validation_context):
        """Test is_valid property."""
        validation_context.add_validator("field", RequiredValidator())
        validation_context.validate_field("field", "value")
        assert validation_context.is_valid

    def test_errors_property(self, validation_context):
        """Test errors property."""
        validation_context.add_validator(
            "field", RequiredValidator(trigger=ValidationTrigger.ON_CHANGE)
        )
        validation_context.validate_field("field", "")

        errors = validation_context.errors
        assert len(errors) == 1
        assert errors[0].field_name == "field"

    def test_get_result(self, validation_context):
        """Test getting specific field result."""
        validation_context.add_validator("field", RequiredValidator())
        validation_context.validate_field("field", "value")

        result = validation_context.get_result("field")
        assert result is not None
        assert result.is_valid

    def test_clear(self, validation_context):
        """Test clearing all results."""
        validation_context.add_validator("field", RequiredValidator())
        validation_context.validate_field("field", "value")
        validation_context.clear()

        assert validation_context.get_result("field") is None

    def test_clear_field(self, validation_context):
        """Test clearing specific field result."""
        validation_context.add_validator("field", RequiredValidator())
        validation_context.validate_field("field", "value")
        validation_context.clear_field("field")

        assert validation_context.get_result("field") is None

    def test_remove_validator(self, validation_context):
        """Test removing a validator."""
        validator = RequiredValidator()
        validation_context.add_validator("field", validator)
        validation_context.remove_validator("field", validator)
        # Should validate as valid since no validators


# ========== Factory Function Tests ==========


class TestFactoryFunctions:
    """Tests for validator factory functions."""

    def test_required(self):
        """Test required factory."""
        validator = required("Custom message")
        assert isinstance(validator, RequiredValidator)
        result = validator.validate("")
        assert result.error_message == "Custom message"

    def test_range_validator_factory(self):
        """Test range_validator factory."""
        validator = range_validator(min_value=0, max_value=100)
        assert isinstance(validator, RangeValidator)
        assert validator.validate(50).is_valid

    def test_regex_factory(self):
        """Test regex factory."""
        validator = regex(r"^\d+$", "Numbers only")
        assert isinstance(validator, RegexValidator)
        assert validator.validate("123").is_valid

    def test_length_factory(self):
        """Test length factory."""
        validator = length(min_length=3, max_length=10)
        assert isinstance(validator, LengthValidator)
        assert validator.validate("hello").is_valid

    def test_email_factory(self):
        """Test email factory."""
        validator = email("Bad email")
        assert isinstance(validator, EmailValidator)
        assert validator.validate("test@example.com").is_valid

    def test_custom_factory(self):
        """Test custom factory."""
        validator = custom(lambda v: v > 0, "Must be positive")
        assert isinstance(validator, CustomValidator)
        assert validator.validate(5).is_valid

    def test_all_of_factory(self):
        """Test all_of factory."""
        validator = all_of(
            RangeValidator(min_value=0),
            RangeValidator(max_value=100)
        )
        assert isinstance(validator, CompositeValidator)
        assert validator._mode == "and"

    def test_any_of_factory(self):
        """Test any_of factory."""
        validator = any_of(
            RangeValidator(max_value=10),
            RangeValidator(min_value=90)
        )
        assert isinstance(validator, CompositeValidator)
        assert validator._mode == "or"

"""
Input validation for UI data binding.

Provides validators for user input with support for multiple validation rules,
triggers, and async validation.
"""
from __future__ import annotations

import asyncio
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    List,
    Optional,
    Pattern,
    TypeVar,
    Union,
)

T = TypeVar("T")


class ValidationTrigger(Enum):
    """When to trigger validation."""

    ON_CHANGE = auto()  # Validate on every change
    ON_BLUR = auto()  # Validate when focus is lost
    ON_SUBMIT = auto()  # Validate only on form submit
    EXPLICIT = auto()  # Validate only when explicitly requested


class ValidationSeverity(Enum):
    """Severity level of validation result."""

    ERROR = auto()
    WARNING = auto()
    INFO = auto()


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    error_message: Optional[str] = None
    severity: ValidationSeverity = ValidationSeverity.ERROR
    field_name: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def valid(cls) -> "ValidationResult":
        """Create a valid result."""
        return cls(is_valid=True)

    @classmethod
    def invalid(
        cls,
        message: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        field_name: Optional[str] = None,
        **metadata,
    ) -> "ValidationResult":
        """Create an invalid result."""
        return cls(
            is_valid=False,
            error_message=message,
            severity=severity,
            field_name=field_name,
            metadata=metadata,
        )

    @classmethod
    def warning(
        cls,
        message: str,
        field_name: Optional[str] = None,
        **metadata,
    ) -> "ValidationResult":
        """Create a warning result (valid but with warning)."""
        return cls(
            is_valid=True,
            error_message=message,
            severity=ValidationSeverity.WARNING,
            field_name=field_name,
            metadata=metadata,
        )

    def __bool__(self) -> bool:
        """Allow using ValidationResult as a boolean."""
        return self.is_valid


class IValidator(ABC, Generic[T]):
    """Interface for validators."""

    @abstractmethod
    def validate(self, value: T) -> ValidationResult:
        """Validate a value and return the result."""
        pass

    @property
    def trigger(self) -> ValidationTrigger:
        """Return when this validator should trigger."""
        return ValidationTrigger.ON_CHANGE


class IAsyncValidator(ABC, Generic[T]):
    """Interface for asynchronous validators."""

    @abstractmethod
    async def validate(self, value: T) -> ValidationResult:
        """Asynchronously validate a value."""
        pass

    @property
    def trigger(self) -> ValidationTrigger:
        """Return when this validator should trigger."""
        return ValidationTrigger.ON_BLUR


class RequiredValidator(IValidator[Any]):
    """Validates that a value is not None/empty."""

    def __init__(
        self,
        message: str = "This field is required",
        allow_whitespace: bool = False,
        trigger: ValidationTrigger = ValidationTrigger.ON_CHANGE,
    ):
        self._message = message
        self._allow_whitespace = allow_whitespace
        self._trigger = trigger

    @property
    def trigger(self) -> ValidationTrigger:
        return self._trigger

    def validate(self, value: Any) -> ValidationResult:
        """Validate that value is not empty."""
        if value is None:
            return ValidationResult.invalid(self._message)

        if isinstance(value, str):
            check_value = value if self._allow_whitespace else value.strip()
            if not check_value:
                return ValidationResult.invalid(self._message)

        if isinstance(value, (list, dict, set)):
            if len(value) == 0:
                return ValidationResult.invalid(self._message)

        return ValidationResult.valid()


class RangeValidator(IValidator[Union[int, float]]):
    """Validates that a numeric value is within a range."""

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        min_inclusive: bool = True,
        max_inclusive: bool = True,
        message: Optional[str] = None,
        trigger: ValidationTrigger = ValidationTrigger.ON_CHANGE,
    ):
        self._min = min_value
        self._max = max_value
        self._min_inclusive = min_inclusive
        self._max_inclusive = max_inclusive
        self._message = message
        self._trigger = trigger

    @property
    def trigger(self) -> ValidationTrigger:
        return self._trigger

    def validate(self, value: Union[int, float]) -> ValidationResult:
        """Validate that value is within range."""
        if value is None:
            return ValidationResult.valid()  # Let RequiredValidator handle None

        try:
            num = float(value)
        except (ValueError, TypeError):
            return ValidationResult.invalid("Value must be a number")

        # Check minimum
        if self._min is not None:
            if self._min_inclusive:
                if num < self._min:
                    msg = self._message or f"Value must be at least {self._min}"
                    return ValidationResult.invalid(msg)
            else:
                if num <= self._min:
                    msg = self._message or f"Value must be greater than {self._min}"
                    return ValidationResult.invalid(msg)

        # Check maximum
        if self._max is not None:
            if self._max_inclusive:
                if num > self._max:
                    msg = self._message or f"Value must be at most {self._max}"
                    return ValidationResult.invalid(msg)
            else:
                if num >= self._max:
                    msg = self._message or f"Value must be less than {self._max}"
                    return ValidationResult.invalid(msg)

        return ValidationResult.valid()


class RegexValidator(IValidator[str]):
    """Validates that a string matches a regular expression."""

    def __init__(
        self,
        pattern: Union[str, Pattern],
        message: str = "Value does not match required format",
        flags: int = 0,
        must_match: bool = True,
        trigger: ValidationTrigger = ValidationTrigger.ON_BLUR,
    ):
        if isinstance(pattern, str):
            self._pattern = re.compile(pattern, flags)
        else:
            self._pattern = pattern
        self._message = message
        self._must_match = must_match
        self._trigger = trigger

    @property
    def trigger(self) -> ValidationTrigger:
        return self._trigger

    def validate(self, value: str) -> ValidationResult:
        """Validate that value matches the pattern."""
        if value is None or value == "":
            return ValidationResult.valid()

        matches = bool(self._pattern.search(value))
        if self._must_match and not matches:
            return ValidationResult.invalid(self._message)
        if not self._must_match and matches:
            return ValidationResult.invalid(self._message)

        return ValidationResult.valid()


class LengthValidator(IValidator[Union[str, list, dict]]):
    """Validates the length of a string, list, or dict."""

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        exact_length: Optional[int] = None,
        message: Optional[str] = None,
        trigger: ValidationTrigger = ValidationTrigger.ON_CHANGE,
    ):
        self._min = min_length
        self._max = max_length
        self._exact = exact_length
        self._message = message
        self._trigger = trigger

    @property
    def trigger(self) -> ValidationTrigger:
        return self._trigger

    def validate(self, value: Union[str, list, dict]) -> ValidationResult:
        """Validate the length of the value."""
        if value is None:
            return ValidationResult.valid()

        try:
            length = len(value)
        except TypeError:
            return ValidationResult.invalid("Value does not have a length")

        if self._exact is not None:
            if length != self._exact:
                msg = self._message or f"Length must be exactly {self._exact}"
                return ValidationResult.invalid(msg)
            return ValidationResult.valid()

        if self._min is not None and length < self._min:
            msg = self._message or f"Length must be at least {self._min}"
            return ValidationResult.invalid(msg)

        if self._max is not None and length > self._max:
            msg = self._message or f"Length must be at most {self._max}"
            return ValidationResult.invalid(msg)

        return ValidationResult.valid()


class EmailValidator(IValidator[str]):
    """Validates email addresses."""

    # Basic email pattern - not RFC 5322 compliant but practical
    EMAIL_PATTERN = re.compile(
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    )

    def __init__(
        self,
        message: str = "Please enter a valid email address",
        trigger: ValidationTrigger = ValidationTrigger.ON_BLUR,
    ):
        self._message = message
        self._trigger = trigger

    @property
    def trigger(self) -> ValidationTrigger:
        return self._trigger

    def validate(self, value: str) -> ValidationResult:
        """Validate email address format."""
        if value is None or value == "":
            return ValidationResult.valid()

        value = value.strip()
        if not self.EMAIL_PATTERN.match(value):
            return ValidationResult.invalid(self._message)

        return ValidationResult.valid()


class UrlValidator(IValidator[str]):
    """Validates URLs."""

    URL_PATTERN = re.compile(
        r"^https?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain
        r"localhost|"  # localhost
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # or IP
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        message: str = "Please enter a valid URL",
        require_https: bool = False,
        trigger: ValidationTrigger = ValidationTrigger.ON_BLUR,
    ):
        self._message = message
        self._require_https = require_https
        self._trigger = trigger

    @property
    def trigger(self) -> ValidationTrigger:
        return self._trigger

    def validate(self, value: str) -> ValidationResult:
        """Validate URL format."""
        if value is None or value == "":
            return ValidationResult.valid()

        value = value.strip()

        if self._require_https and not value.startswith("https://"):
            return ValidationResult.invalid("URL must use HTTPS")

        if not self.URL_PATTERN.match(value):
            return ValidationResult.invalid(self._message)

        return ValidationResult.valid()


class ChoiceValidator(IValidator[Any]):
    """Validates that a value is one of the allowed choices."""

    def __init__(
        self,
        choices: List[Any],
        message: Optional[str] = None,
        case_sensitive: bool = True,
        trigger: ValidationTrigger = ValidationTrigger.ON_CHANGE,
    ):
        self._choices = choices
        self._message = message
        self._case_sensitive = case_sensitive
        self._trigger = trigger

    @property
    def trigger(self) -> ValidationTrigger:
        return self._trigger

    def validate(self, value: Any) -> ValidationResult:
        """Validate that value is in the allowed choices."""
        if value is None:
            return ValidationResult.valid()

        check_value = value
        check_choices = self._choices

        if not self._case_sensitive and isinstance(value, str):
            check_value = value.lower()
            check_choices = [
                c.lower() if isinstance(c, str) else c for c in self._choices
            ]

        if check_value not in check_choices:
            msg = self._message or f"Value must be one of: {', '.join(str(c) for c in self._choices)}"
            return ValidationResult.invalid(msg)

        return ValidationResult.valid()


class TypeValidator(IValidator[Any]):
    """Validates that a value is of a specific type."""

    def __init__(
        self,
        expected_type: type,
        message: Optional[str] = None,
        trigger: ValidationTrigger = ValidationTrigger.ON_CHANGE,
    ):
        self._type = expected_type
        self._message = message
        self._trigger = trigger

    @property
    def trigger(self) -> ValidationTrigger:
        return self._trigger

    def validate(self, value: Any) -> ValidationResult:
        """Validate that value is of the expected type."""
        if value is None:
            return ValidationResult.valid()

        if not isinstance(value, self._type):
            msg = self._message or f"Value must be of type {self._type.__name__}"
            return ValidationResult.invalid(msg)

        return ValidationResult.valid()


class CustomValidator(IValidator[T]):
    """Validator with a custom validation function."""

    def __init__(
        self,
        validate_func: Callable[[T], Union[bool, ValidationResult, str]],
        message: str = "Validation failed",
        trigger: ValidationTrigger = ValidationTrigger.ON_CHANGE,
    ):
        self._func = validate_func
        self._message = message
        self._trigger = trigger

    @property
    def trigger(self) -> ValidationTrigger:
        return self._trigger

    def validate(self, value: T) -> ValidationResult:
        """Apply custom validation function."""
        try:
            result = self._func(value)
        except Exception as e:
            return ValidationResult.invalid(str(e))

        if isinstance(result, ValidationResult):
            return result
        if isinstance(result, bool):
            return ValidationResult.valid() if result else ValidationResult.invalid(self._message)
        if isinstance(result, str):
            return ValidationResult.invalid(result) if result else ValidationResult.valid()

        return ValidationResult.valid()


class AsyncCustomValidator(IAsyncValidator[T]):
    """Validator with a custom async validation function."""

    def __init__(
        self,
        validate_func: Callable[[T], Awaitable[Union[bool, ValidationResult, str]]],
        message: str = "Validation failed",
        trigger: ValidationTrigger = ValidationTrigger.ON_BLUR,
    ):
        self._func = validate_func
        self._message = message
        self._trigger = trigger

    @property
    def trigger(self) -> ValidationTrigger:
        return self._trigger

    async def validate(self, value: T) -> ValidationResult:
        """Apply custom async validation function."""
        try:
            result = await self._func(value)
        except Exception as e:
            return ValidationResult.invalid(str(e))

        if isinstance(result, ValidationResult):
            return result
        if isinstance(result, bool):
            return ValidationResult.valid() if result else ValidationResult.invalid(self._message)
        if isinstance(result, str):
            return ValidationResult.invalid(result) if result else ValidationResult.valid()

        return ValidationResult.valid()


class CompareValidator(IValidator[Any]):
    """Validates by comparing with another value or field."""

    def __init__(
        self,
        compare_value: Any = None,
        compare_getter: Optional[Callable[[], Any]] = None,
        operator: str = "==",
        message: Optional[str] = None,
        trigger: ValidationTrigger = ValidationTrigger.ON_BLUR,
    ):
        self._compare_value = compare_value
        self._getter = compare_getter
        self._operator = operator
        self._message = message
        self._trigger = trigger

    @property
    def trigger(self) -> ValidationTrigger:
        return self._trigger

    def validate(self, value: Any) -> ValidationResult:
        """Validate by comparing values."""
        other = self._getter() if self._getter else self._compare_value

        result = False
        if self._operator == "==":
            result = value == other
        elif self._operator == "!=":
            result = value != other
        elif self._operator == "<":
            result = value < other
        elif self._operator == "<=":
            result = value <= other
        elif self._operator == ">":
            result = value > other
        elif self._operator == ">=":
            result = value >= other

        if not result:
            msg = self._message or f"Comparison failed: {value} {self._operator} {other}"
            return ValidationResult.invalid(msg)

        return ValidationResult.valid()


class CompositeValidator(IValidator[T]):
    """Combines multiple validators with AND/OR logic."""

    def __init__(
        self,
        validators: List[IValidator[T]],
        mode: str = "and",  # "and" or "or"
        stop_on_first_failure: bool = True,
    ):
        self._validators = validators
        self._mode = mode
        self._stop_on_first = stop_on_first_failure

    @property
    def trigger(self) -> ValidationTrigger:
        # Return the most frequent trigger among validators
        triggers = [v.trigger for v in self._validators]
        if not triggers:
            return ValidationTrigger.ON_CHANGE
        return max(set(triggers), key=triggers.count)

    def validate(self, value: T) -> ValidationResult:
        """Validate using all validators with specified logic."""
        results: List[ValidationResult] = []

        for validator in self._validators:
            result = validator.validate(value)
            results.append(result)

            if self._mode == "and":
                if not result.is_valid:
                    if self._stop_on_first:
                        return result
            else:  # "or" mode
                if result.is_valid:
                    return ValidationResult.valid()

        if self._mode == "and":
            # All must be valid
            invalid_results = [r for r in results if not r.is_valid]
            if invalid_results:
                return invalid_results[0]
            return ValidationResult.valid()
        else:
            # At least one must be valid
            if all(not r.is_valid for r in results):
                return results[0] if results else ValidationResult.invalid("Validation failed")
            return ValidationResult.valid()


class ValidationContext:
    """
    Context for managing validation of multiple fields.

    Supports coordinated validation triggers and aggregate results.
    """

    def __init__(self):
        self._validators: dict[str, List[IValidator]] = {}
        self._async_validators: dict[str, List[IAsyncValidator]] = {}
        self._results: dict[str, ValidationResult] = {}
        self._lock = threading.RLock()

    def add_validator(
        self, field_name: str, validator: Union[IValidator, IAsyncValidator]
    ) -> None:
        """Add a validator for a field."""
        with self._lock:
            if isinstance(validator, IAsyncValidator):
                if field_name not in self._async_validators:
                    self._async_validators[field_name] = []
                self._async_validators[field_name].append(validator)
            else:
                if field_name not in self._validators:
                    self._validators[field_name] = []
                self._validators[field_name].append(validator)

    def remove_validator(
        self, field_name: str, validator: Union[IValidator, IAsyncValidator]
    ) -> None:
        """Remove a validator from a field."""
        with self._lock:
            if isinstance(validator, IAsyncValidator):
                if field_name in self._async_validators:
                    self._async_validators[field_name].remove(validator)
            else:
                if field_name in self._validators:
                    self._validators[field_name].remove(validator)

    def validate_field(
        self,
        field_name: str,
        value: Any,
        trigger: ValidationTrigger = ValidationTrigger.ON_CHANGE,
    ) -> ValidationResult:
        """Validate a single field synchronously."""
        validators = self._validators.get(field_name, [])
        applicable = [v for v in validators if v.trigger == trigger]

        for validator in applicable:
            result = validator.validate(value)
            if not result.is_valid:
                result.field_name = field_name
                with self._lock:
                    self._results[field_name] = result
                return result

        result = ValidationResult.valid()
        result.field_name = field_name
        with self._lock:
            self._results[field_name] = result
        return result

    async def validate_field_async(
        self,
        field_name: str,
        value: Any,
        trigger: ValidationTrigger = ValidationTrigger.ON_BLUR,
    ) -> ValidationResult:
        """Validate a single field including async validators."""
        # First run sync validators
        sync_result = self.validate_field(field_name, value, trigger)
        if not sync_result.is_valid:
            return sync_result

        # Then run async validators
        async_validators = self._async_validators.get(field_name, [])
        applicable = [v for v in async_validators if v.trigger == trigger]

        for validator in applicable:
            result = await validator.validate(value)
            if not result.is_valid:
                result.field_name = field_name
                with self._lock:
                    self._results[field_name] = result
                return result

        result = ValidationResult.valid()
        result.field_name = field_name
        with self._lock:
            self._results[field_name] = result
        return result

    def validate_all(
        self,
        values: dict[str, Any],
        trigger: ValidationTrigger = ValidationTrigger.ON_SUBMIT,
    ) -> List[ValidationResult]:
        """Validate all fields synchronously."""
        results = []
        for field_name, value in values.items():
            result = self.validate_field(field_name, value, trigger)
            results.append(result)
        return results

    async def validate_all_async(
        self,
        values: dict[str, Any],
        trigger: ValidationTrigger = ValidationTrigger.ON_SUBMIT,
    ) -> List[ValidationResult]:
        """Validate all fields including async validators."""
        tasks = [
            self.validate_field_async(field_name, value, trigger)
            for field_name, value in values.items()
        ]
        return await asyncio.gather(*tasks)

    @property
    def is_valid(self) -> bool:
        """Check if all cached results are valid."""
        with self._lock:
            return all(r.is_valid for r in self._results.values())

    @property
    def errors(self) -> List[ValidationResult]:
        """Get all error results."""
        with self._lock:
            return [r for r in self._results.values() if not r.is_valid]

    def get_result(self, field_name: str) -> Optional[ValidationResult]:
        """Get the last validation result for a field."""
        with self._lock:
            return self._results.get(field_name)

    def clear(self) -> None:
        """Clear all cached results."""
        with self._lock:
            self._results.clear()

    def clear_field(self, field_name: str) -> None:
        """Clear cached result for a specific field."""
        with self._lock:
            self._results.pop(field_name, None)


# Convenience factory functions
def required(
    message: str = "This field is required",
    allow_whitespace: bool = False,
) -> RequiredValidator:
    """Create a required validator."""
    return RequiredValidator(message, allow_whitespace)


def range_validator(
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    message: Optional[str] = None,
) -> RangeValidator:
    """Create a range validator."""
    return RangeValidator(min_value, max_value, message=message)


def regex(
    pattern: Union[str, Pattern],
    message: str = "Value does not match required format",
) -> RegexValidator:
    """Create a regex validator."""
    return RegexValidator(pattern, message)


def length(
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    exact_length: Optional[int] = None,
) -> LengthValidator:
    """Create a length validator."""
    return LengthValidator(min_length, max_length, exact_length)


def email(message: str = "Please enter a valid email address") -> EmailValidator:
    """Create an email validator."""
    return EmailValidator(message)


def custom(
    validate_func: Callable[[Any], Union[bool, ValidationResult, str]],
    message: str = "Validation failed",
) -> CustomValidator:
    """Create a custom validator."""
    return CustomValidator(validate_func, message)


def all_of(*validators: IValidator) -> CompositeValidator:
    """Create a composite validator requiring all validators to pass."""
    return CompositeValidator(list(validators), mode="and")


def any_of(*validators: IValidator) -> CompositeValidator:
    """Create a composite validator requiring at least one validator to pass."""
    return CompositeValidator(list(validators), mode="or")


__all__ = [
    # Interfaces
    "IValidator",
    "IAsyncValidator",
    # Types
    "ValidationResult",
    "ValidationTrigger",
    "ValidationSeverity",
    # Built-in validators
    "RequiredValidator",
    "RangeValidator",
    "RegexValidator",
    "LengthValidator",
    "EmailValidator",
    "UrlValidator",
    "ChoiceValidator",
    "TypeValidator",
    "CompareValidator",
    # Custom validators
    "CustomValidator",
    "AsyncCustomValidator",
    # Composite
    "CompositeValidator",
    # Context
    "ValidationContext",
    # Factory functions
    "required",
    "range_validator",
    "regex",
    "length",
    "email",
    "custom",
    "all_of",
    "any_of",
]

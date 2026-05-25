"""
Validation descriptors - validate values on write.

Provides type checking, range validation, and custom validation rules.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional, Sequence, TypeVar

from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")


class ValidatedDescriptor(BaseDescriptor[T]):
    """
    Validates values on write using provided validators.

    Each validator should be a callable that takes a value and returns True
    if valid, False or raises an exception if invalid.
    """

    __slots__ = ("_validators",)

    descriptor_id = "validated"
    accepts_inner = ("storage", "cached", "lazy")
    accepts_outer = ("*",)
    excludes = ("validated",)  # No double validation

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        validators: Optional[Sequence[Callable[[Any], bool]]] = None,
        **config: Any,
    ) -> None:
        """
        Initialize validation descriptor.

        Args:
            field_type: The type annotation for this field.
            inner: Inner descriptor to wrap.
            validators: List of validator functions.
            **config: Additional configuration.
        """
        super().__init__(field_type=field_type, inner=inner, **config)
        self._validators = list(validators) if validators else []

    def pre_set(self, obj: Any, value: T) -> T:
        """Validate the value before storing."""
        for validator in self._validators:
            try:
                result = validator(value)
                if result is False:
                    raise ValueError(f"Validation failed for '{self._name}': {value!r}")
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"Validation error for '{self._name}': {e}") from e
        return value

    def add_validator(self, validator: Callable[[Any], bool]) -> None:
        """Add a validator to the list."""
        self._validators.append(validator)

    @property
    def descriptor_steps(self) -> list["Step"]:
        return [Step(Op.VALIDATE, {"constraint": "custom", "validator_count": len(self._validators)})]

    def get_metadata(self) -> dict[str, Any]:
        """Return validation configuration."""
        meta = super().get_metadata()
        meta["validator_count"] = len(self._validators)
        return meta


class RangeDescriptor(BaseDescriptor[T]):
    """
    Clamps numeric values to a range.

    Can either clamp silently or raise an error on out-of-range values.
    """

    __slots__ = ("_min", "_max", "_clamp")

    descriptor_id = "range"
    accepts_inner = ("storage", "validated")
    accepts_outer = ("*",)
    excludes = ()

    def __init__(
        self,
        field_type: type = float,
        inner: Optional[BaseDescriptor[T]] = None,
        min_val: float = float("-inf"),
        max_val: float = float("inf"),
        clamp: bool = True,
        **config: Any,
    ) -> None:
        """
        Initialize range descriptor.

        Args:
            field_type: The type annotation for this field.
            inner: Inner descriptor to wrap.
            min_val: Minimum allowed value.
            max_val: Maximum allowed value.
            clamp: If True, clamp values; if False, raise error.
            **config: Additional configuration.
        """
        super().__init__(field_type=field_type, inner=inner, **config)
        if min_val > max_val:
            raise ValueError(f"min_val ({min_val}) cannot be greater than max_val ({max_val})")
        self._min = min_val
        self._max = max_val
        self._clamp = clamp

    def pre_set(self, obj: Any, value: T) -> T:
        """Validate/clamp the value to range."""
        # Handle non-numeric types gracefully
        try:
            if value < self._min or value > self._max:
                if self._clamp:
                    return max(self._min, min(self._max, value))  # type: ignore
                else:
                    raise ValueError(
                        f"Value {value} for '{self._name}' is outside range "
                        f"[{self._min}, {self._max}]"
                    )
        except TypeError as e:
            # Value doesn't support comparison - this is an error
            raise TypeError(
                f"Cannot perform range validation on non-numeric type "
                f"{type(value).__name__} for '{self._name}'"
            ) from e
        return value

    @property
    def descriptor_steps(self) -> list["Step"]:
        return [Step(Op.VALIDATE, {"constraint": "range", "min": self._min, "max": self._max, "clamp": self._clamp})]

    def get_metadata(self) -> dict[str, Any]:
        """Return range configuration."""
        meta = super().get_metadata()
        meta["range"] = (self._min, self._max)
        meta["clamp"] = self._clamp
        return meta


class TypeDescriptor(BaseDescriptor[T]):
    """Runtime type enforcement. Rejects values of wrong type, optionally coerces."""

    __slots__ = ("_expected_type", "_coerce")

    descriptor_id = "typed"
    accepts_inner = ("storage", "cached", "lazy")
    accepts_outer = ("*",)
    excludes = ("typed",)  # No double type checking

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        expected_type: Optional[type] = None,
        coerce: bool = False,
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._expected_type = expected_type or field_type
        self._coerce = coerce

    def pre_set(self, obj: Any, value: T) -> T:
        if self._coerce:
            try:
                value = self._expected_type(value)  # type: ignore
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"Cannot coerce {type(value).__name__} to "
                    f"{self._expected_type.__name__} for '{self._name}': {e}"
                ) from e
        elif not isinstance(value, self._expected_type):
            raise TypeError(
                f"Expected {self._expected_type.__name__} for "
                f"'{self._name}', got {type(value).__name__}"
            )
        return value

    @property
    def descriptor_steps(self) -> list["Step"]:
        from trinity.decorators.ops import Step, Op
        return [Step(Op.VALIDATE, {"field": self._name, "rule": "type", "expected_type": str(self._expected_type)})]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["expected_type"] = self._expected_type.__name__
        meta["coerce"] = self._coerce
        return meta


class ChoiceDescriptor(BaseDescriptor[T]):
    """Value must be one of an allowed set."""

    __slots__ = ("_choices",)

    descriptor_id = "choice"
    accepts_inner = ("storage", "cached", "lazy", "typed")
    accepts_outer = ("*",)
    excludes = ("choice",)  # No double choice validation

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        choices: Sequence[Any] = (),
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._choices: frozenset[Any] = frozenset(choices)
        if not self._choices:
            raise ValueError("ChoiceDescriptor requires at least one choice")

    def pre_set(self, obj: Any, value: T) -> T:
        if value not in self._choices:
            # Try to sort for nice error message, but fall back to unsorted if types incompatible
            try:
                choices_display = sorted(self._choices)
            except TypeError:
                choices_display = list(self._choices)
            raise ValueError(
                f"'{self._name}' must be one of {choices_display}, got {value!r}"
            )
        return value

    @property
    def descriptor_steps(self) -> list["Step"]:
        from trinity.decorators.ops import Step, Op
        try:
            choices = sorted(self._choices)
        except TypeError:
            choices = list(self._choices)
        return [Step(Op.VALIDATE, {"field": self._name, "rule": "choice", "choices": choices})]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        # Try to sort for consistent metadata, but fall back to unsorted if types incompatible
        try:
            meta["choices"] = sorted(self._choices)
        except TypeError:
            meta["choices"] = list(self._choices)
        return meta


class PatternDescriptor(BaseDescriptor[T]):
    """String must match a regex pattern."""

    __slots__ = ("_pattern", "_compiled")

    descriptor_id = "pattern"
    accepts_inner = ("storage", "cached", "lazy")
    accepts_outer = ("*",)
    excludes = ("pattern",)  # No double pattern validation

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        pattern: str = ".*",
        **config: Any,
    ) -> None:
        super().__init__(field_type=field_type, inner=inner, **config)
        self._pattern = pattern
        self._compiled = re.compile(pattern)

    def pre_set(self, obj: Any, value: T) -> T:
        if not isinstance(value, str):
            raise TypeError(
                f"'{self._name}' requires a string, got {type(value).__name__}"
            )
        if not self._compiled.fullmatch(value):
            raise ValueError(
                f"'{self._name}' must match pattern '{self._pattern}', got {value!r}"
            )
        return value

    @property
    def descriptor_steps(self) -> list["Step"]:
        from trinity.decorators.ops import Step, Op
        return [Step(Op.VALIDATE, {"field": self._name, "rule": "pattern", "pattern": self._pattern})]

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["pattern"] = self._pattern
        return meta

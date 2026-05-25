"""
Custom assertions with @invariant decorator.

Provides design-by-contract style assertions for runtime
verification of code invariants, preconditions, and postconditions.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class InvariantError(AssertionError):
    """Error raised when an invariant is violated."""

    def __init__(
        self,
        message: str,
        invariant_name: str = "",
        class_name: str = "",
        values: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.invariant_name = invariant_name
        self.class_name = class_name
        self.values = values or {}


class PreconditionError(AssertionError):
    """Error raised when a precondition is violated."""

    def __init__(
        self,
        message: str,
        function_name: str = "",
        parameter: str = "",
        value: Any = None,
    ):
        super().__init__(message)
        self.function_name = function_name
        self.parameter = parameter
        self.value = value


class PostconditionError(AssertionError):
    """Error raised when a postcondition is violated."""

    def __init__(
        self,
        message: str,
        function_name: str = "",
        result: Any = None,
    ):
        super().__init__(message)
        self.function_name = function_name
        self.result = result


@dataclass
class AssertionConfig:
    """Configuration for assertion checking."""

    enabled: bool = True
    check_invariants: bool = True
    check_preconditions: bool = True
    check_postconditions: bool = True
    raise_on_failure: bool = True
    log_failures: bool = True


# Global configuration
_config = AssertionConfig()


def enable_assertions() -> None:
    """Enable all assertion checking."""
    _config.enabled = True


def disable_assertions() -> None:
    """Disable all assertion checking."""
    _config.enabled = False


def get_config() -> AssertionConfig:
    """Get the assertion configuration."""
    return _config


def set_config(config: AssertionConfig) -> None:
    """Set the assertion configuration."""
    global _config
    _config = config


def invariant(
    condition: Union[Callable[[Any], bool], str],
    message: str = "",
    name: str = "",
) -> Callable[[Type[T]], Type[T]]:
    """
    Class decorator to add an invariant check.

    The invariant is checked after __init__ and after each method call.

    Args:
        condition: Callable that takes self and returns bool, or attribute name
        message: Error message on failure
        name: Invariant name for error reporting

    Example:
        @invariant(lambda self: self.value >= 0, "Value must be non-negative")
        class Counter:
            def __init__(self):
                self.value = 0

            def increment(self):
                self.value += 1
    """
    def decorator(cls: Type[T]) -> Type[T]:
        invariant_name = name or f"{cls.__name__}_invariant"

        # Store invariant info
        if not hasattr(cls, "_invariants"):
            cls._invariants = []
        cls._invariants.append({
            "condition": condition,
            "message": message,
            "name": invariant_name,
        })

        def check_invariant(self):
            """Check all invariants on the instance."""
            if not _config.enabled or not _config.check_invariants:
                return

            for inv in cls._invariants:
                cond = inv["condition"]
                if isinstance(cond, str):
                    # Attribute-based condition
                    result = getattr(self, cond, None)
                    if callable(result):
                        result = result()
                else:
                    result = cond(self)

                if not result:
                    error_msg = inv["message"] or f"Invariant '{inv['name']}' violated"
                    if _config.log_failures:
                        print(f"INVARIANT VIOLATION: {error_msg}")
                    if _config.raise_on_failure:
                        raise InvariantError(
                            error_msg,
                            invariant_name=inv["name"],
                            class_name=cls.__name__,
                        )

        # Wrap __init__
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            check_invariant(self)

        cls.__init__ = new_init

        # Wrap all public methods
        for method_name in dir(cls):
            if method_name.startswith("_"):
                continue

            method = getattr(cls, method_name)
            if not callable(method) or isinstance(method, type):
                continue

            @functools.wraps(method)
            def wrapped_method(self, *args, _method=method, **kwargs):
                result = _method(self, *args, **kwargs)
                check_invariant(self)
                return result

            setattr(cls, method_name, wrapped_method)

        return cls

    return decorator


def precondition(
    condition: Callable[..., bool],
    message: str = "",
    parameter: str = "",
) -> Callable[[F], F]:
    """
    Decorator to add a precondition check to a function.

    Args:
        condition: Callable that takes function arguments and returns bool
        message: Error message on failure
        parameter: Name of parameter being checked

    Example:
        @precondition(lambda x: x >= 0, "x must be non-negative", "x")
        def sqrt(x):
            return x ** 0.5
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if _config.enabled and _config.check_preconditions:
                # Get function signature for better error messages
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                if not condition(*args, **kwargs):
                    error_msg = message or f"Precondition failed for {func.__name__}"
                    if _config.log_failures:
                        print(f"PRECONDITION VIOLATION: {error_msg}")
                    if _config.raise_on_failure:
                        value = bound.arguments.get(parameter) if parameter else None
                        raise PreconditionError(
                            error_msg,
                            function_name=func.__name__,
                            parameter=parameter,
                            value=value,
                        )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def postcondition(
    condition: Callable[[Any], bool],
    message: str = "",
) -> Callable[[F], F]:
    """
    Decorator to add a postcondition check to a function.

    Args:
        condition: Callable that takes the result and returns bool
        message: Error message on failure

    Example:
        @postcondition(lambda r: r >= 0, "Result must be non-negative")
        def abs_value(x):
            return abs(x)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if _config.enabled and _config.check_postconditions:
                if not condition(result):
                    error_msg = message or f"Postcondition failed for {func.__name__}"
                    if _config.log_failures:
                        print(f"POSTCONDITION VIOLATION: {error_msg}")
                    if _config.raise_on_failure:
                        raise PostconditionError(
                            error_msg,
                            function_name=func.__name__,
                            result=result,
                        )

            return result

        return wrapper  # type: ignore

    return decorator


def check(
    condition: bool,
    message: str = "",
    exception_type: Type[Exception] = AssertionError,
) -> None:
    """
    Runtime check that raises if condition is false.

    Args:
        condition: Condition to check
        message: Error message
        exception_type: Exception type to raise

    Example:
        check(len(items) > 0, "Items list cannot be empty")
    """
    if not _config.enabled:
        return

    if not condition:
        if _config.log_failures:
            print(f"CHECK FAILED: {message}")
        if _config.raise_on_failure:
            raise exception_type(message)


def ensure(
    condition: bool,
    message: str = "",
) -> None:
    """
    Ensure a condition is true (postcondition style).

    Args:
        condition: Condition to ensure
        message: Error message

    Example:
        result = process_data(data)
        ensure(result is not None, "Processing must return a result")
    """
    check(condition, message, PostconditionError)


def require(
    condition: bool,
    message: str = "",
) -> None:
    """
    Require a condition to be true (precondition style).

    Args:
        condition: Condition to require
        message: Error message

    Example:
        require(user is not None, "User must be provided")
    """
    check(condition, message, PreconditionError)


class ContractMixin:
    """
    Mixin class for adding contract support to classes.

    Example:
        class BankAccount(ContractMixin):
            def __init__(self, balance=0):
                self.balance = balance
                self._check_invariants()

            def _invariant(self):
                return self.balance >= 0

            def withdraw(self, amount):
                self._require(amount > 0, "Amount must be positive")
                self._require(amount <= self.balance, "Insufficient funds")
                self.balance -= amount
                self._check_invariants()
    """

    def _invariant(self) -> bool:
        """Override to define class invariant."""
        return True

    def _check_invariants(self) -> None:
        """Check class invariants."""
        if not _config.enabled or not _config.check_invariants:
            return

        if not self._invariant():
            if _config.raise_on_failure:
                raise InvariantError(
                    f"Invariant violated in {self.__class__.__name__}",
                    class_name=self.__class__.__name__,
                )

    def _require(self, condition: bool, message: str = "") -> None:
        """Check a precondition."""
        require(condition, message)

    def _ensure(self, condition: bool, message: str = "") -> None:
        """Check a postcondition."""
        ensure(condition, message)


def contracts_enabled() -> bool:
    """Check if contracts are enabled."""
    return _config.enabled


def with_contracts(
    invariant_func: Optional[Callable[[Any], bool]] = None,
    invariant_message: str = "",
) -> Callable[[Type[T]], Type[T]]:
    """
    Class decorator that adds full contract support.

    Args:
        invariant_func: Optional invariant function
        invariant_message: Invariant error message

    Example:
        @with_contracts(lambda self: self.count >= 0, "Count must be non-negative")
        class Counter:
            def __init__(self):
                self.count = 0
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # Add ContractMixin if not present
        if not issubclass(cls, ContractMixin):
            cls = type(cls.__name__, (cls, ContractMixin), dict(cls.__dict__))

        # Add invariant if provided
        if invariant_func:
            original_invariant = cls._invariant

            def new_invariant(self):
                return original_invariant(self) and invariant_func(self)

            cls._invariant = new_invariant

        return cls

    return decorator


# Type checking assertions

def assert_type(value: Any, expected_type: Type, message: str = "") -> None:
    """
    Assert that a value is of the expected type.

    Args:
        value: Value to check
        expected_type: Expected type
        message: Error message

    Example:
        assert_type(user_id, int, "User ID must be an integer")
    """
    if not _config.enabled:
        return

    if not isinstance(value, expected_type):
        error_msg = message or f"Expected {expected_type.__name__}, got {type(value).__name__}"
        if _config.raise_on_failure:
            raise TypeError(error_msg)


def assert_not_none(value: Any, message: str = "") -> None:
    """
    Assert that a value is not None.

    Args:
        value: Value to check
        message: Error message

    Example:
        assert_not_none(result, "Result cannot be None")
    """
    if not _config.enabled:
        return

    if value is None:
        error_msg = message or "Value cannot be None"
        if _config.raise_on_failure:
            raise ValueError(error_msg)


def assert_in_range(
    value: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float],
    message: str = "",
) -> None:
    """
    Assert that a value is within a range.

    Args:
        value: Value to check
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)
        message: Error message

    Example:
        assert_in_range(health, 0, 100, "Health must be between 0 and 100")
    """
    if not _config.enabled:
        return

    if not (min_val <= value <= max_val):
        error_msg = message or f"Value {value} not in range [{min_val}, {max_val}]"
        if _config.raise_on_failure:
            raise ValueError(error_msg)

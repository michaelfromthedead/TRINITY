"""T-CC-1.9: Result types for Rust bridge functions.

Provides Rust-like Result and Option types for Python, enabling
consistent error handling across the Python/Rust boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    NoReturn,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)


T = TypeVar('T')
E = TypeVar('E')
U = TypeVar('U')
F = TypeVar('F')


class ResultError(Exception):
    """Base exception for Result operations."""
    pass


class UnwrapError(ResultError):
    """Raised when unwrapping an Err or None."""
    pass


class ErrorKind(Enum):
    """Categories of errors for classification."""
    UNKNOWN = auto()
    IO = auto()
    PARSE = auto()
    VALIDATION = auto()
    TIMEOUT = auto()
    NOT_FOUND = auto()
    PERMISSION = auto()
    OVERFLOW = auto()
    UNDERFLOW = auto()
    GPU = auto()
    FFI = auto()
    NETWORK = auto()


@dataclass(frozen=True)
class Error:
    """Structured error with context."""
    message: str
    kind: ErrorKind = ErrorKind.UNKNOWN
    source: Optional["Error"] = None
    context: Optional[dict] = None

    def __str__(self) -> str:
        return self.message

    def chain(self) -> List["Error"]:
        """Get the full error chain."""
        chain = [self]
        current = self.source
        while current:
            chain.append(current)
            current = current.source
        return chain

    def with_context(self, key: str, value: Any) -> "Error":
        """Add context to error."""
        ctx = dict(self.context) if self.context else {}
        ctx[key] = value
        return Error(self.message, self.kind, self.source, ctx)

    def wrap(self, message: str) -> "Error":
        """Wrap this error with additional context."""
        return Error(message, self.kind, source=self)


class Result(Generic[T, E]):
    """A Result type that can be either Ok(value) or Err(error).

    Similar to Rust's Result<T, E> for consistent error handling
    across the Python/Rust FFI boundary.
    """

    __slots__ = ('_value', '_error', '_is_ok')

    def __init__(
        self,
        value: Optional[T] = None,
        error: Optional[E] = None,
        is_ok: bool = True,
    ):
        self._value = value
        self._error = error
        self._is_ok = is_ok

    @classmethod
    def ok(cls, value: T) -> "Result[T, E]":
        """Create an Ok result."""
        return cls(value=value, is_ok=True)

    @classmethod
    def err(cls, error: E) -> "Result[T, E]":
        """Create an Err result."""
        return cls(error=error, is_ok=False)

    @property
    def is_ok(self) -> bool:
        """Check if result is Ok."""
        return self._is_ok

    @property
    def is_err(self) -> bool:
        """Check if result is Err."""
        return not self._is_ok

    def ok_value(self) -> Optional[T]:
        """Get the Ok value or None."""
        return self._value if self._is_ok else None

    def err_value(self) -> Optional[E]:
        """Get the Err value or None."""
        return self._error if not self._is_ok else None

    def unwrap(self) -> T:
        """Unwrap the Ok value, raising UnwrapError if Err."""
        if self._is_ok:
            return self._value
        raise UnwrapError(f"Called unwrap on Err: {self._error}")

    def unwrap_or(self, default: T) -> T:
        """Unwrap Ok or return default."""
        return self._value if self._is_ok else default

    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        """Unwrap Ok or compute from error."""
        return self._value if self._is_ok else f(self._error)

    def unwrap_err(self) -> E:
        """Unwrap the Err value, raising UnwrapError if Ok."""
        if not self._is_ok:
            return self._error
        raise UnwrapError(f"Called unwrap_err on Ok: {self._value}")

    def expect(self, message: str) -> T:
        """Unwrap Ok with custom panic message."""
        if self._is_ok:
            return self._value
        raise UnwrapError(f"{message}: {self._error}")

    def expect_err(self, message: str) -> E:
        """Unwrap Err with custom panic message."""
        if not self._is_ok:
            return self._error
        raise UnwrapError(f"{message}: {self._value}")

    def map(self, f: Callable[[T], U]) -> "Result[U, E]":
        """Map Ok value with a function."""
        if self._is_ok:
            return Result.ok(f(self._value))
        return Result.err(self._error)

    def map_err(self, f: Callable[[E], F]) -> "Result[T, F]":
        """Map Err value with a function."""
        if self._is_ok:
            return Result.ok(self._value)
        return Result.err(f(self._error))

    def map_or(self, default: U, f: Callable[[T], U]) -> U:
        """Map Ok or return default."""
        return f(self._value) if self._is_ok else default

    def map_or_else(self, default: Callable[[E], U], f: Callable[[T], U]) -> U:
        """Map Ok or compute default from error."""
        return f(self._value) if self._is_ok else default(self._error)

    def and_then(self, f: Callable[[T], "Result[U, E]"]) -> "Result[U, E]":
        """Chain operations that return Results."""
        if self._is_ok:
            return f(self._value)
        return Result.err(self._error)

    def or_else(self, f: Callable[[E], "Result[T, F]"]) -> "Result[T, F]":
        """Recover from error with a function returning Result."""
        if self._is_ok:
            return Result.ok(self._value)
        return f(self._error)

    def flatten(self: "Result[Result[T, E], E]") -> "Result[T, E]":
        """Flatten nested Result."""
        if self._is_ok:
            return self._value
        return Result.err(self._error)

    def transpose(self: "Result[Optional[T], E]") -> "Optional[Result[T, E]]":
        """Transpose Result<Option<T>, E> to Option<Result<T, E>>."""
        if self._is_ok:
            if self._value is None:
                return None
            return Result.ok(self._value)
        return Result.err(self._error)

    def __bool__(self) -> bool:
        """Result is truthy if Ok."""
        return self._is_ok

    def __repr__(self) -> str:
        if self._is_ok:
            return f"Ok({self._value!r})"
        return f"Err({self._error!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Result):
            return False
        if self._is_ok != other._is_ok:
            return False
        if self._is_ok:
            return self._value == other._value
        return self._error == other._error

    def __hash__(self) -> int:
        if self._is_ok:
            return hash(("Ok", self._value))
        return hash(("Err", self._error))


def Ok(value: T) -> Result[T, Any]:
    """Create an Ok result."""
    return Result.ok(value)


def Err(error: E) -> Result[Any, E]:
    """Create an Err result."""
    return Result.err(error)


class Option(Generic[T]):
    """An Option type that can be Some(value) or None.

    Similar to Rust's Option<T> for explicit handling of
    optional values.
    """

    __slots__ = ('_value', '_is_some')

    def __init__(self, value: Optional[T] = None, is_some: bool = False):
        self._value = value
        self._is_some = is_some

    @classmethod
    def some(cls, value: T) -> "Option[T]":
        """Create a Some option."""
        return cls(value=value, is_some=True)

    @classmethod
    def none(cls) -> "Option[T]":
        """Create a None option."""
        return cls(is_some=False)

    @classmethod
    def from_nullable(cls, value: Optional[T]) -> "Option[T]":
        """Create Option from nullable value."""
        if value is None:
            return cls.none()
        return cls.some(value)

    @property
    def is_some(self) -> bool:
        """Check if option has a value."""
        return self._is_some

    @property
    def is_none(self) -> bool:
        """Check if option is empty."""
        return not self._is_some

    def unwrap(self) -> T:
        """Unwrap the value, raising UnwrapError if None."""
        if self._is_some:
            return self._value
        raise UnwrapError("Called unwrap on None")

    def unwrap_or(self, default: T) -> T:
        """Unwrap or return default."""
        return self._value if self._is_some else default

    def unwrap_or_else(self, f: Callable[[], T]) -> T:
        """Unwrap or compute default."""
        return self._value if self._is_some else f()

    def expect(self, message: str) -> T:
        """Unwrap with custom panic message."""
        if self._is_some:
            return self._value
        raise UnwrapError(message)

    def map(self, f: Callable[[T], U]) -> "Option[U]":
        """Map the value with a function."""
        if self._is_some:
            return Option.some(f(self._value))
        return Option.none()

    def map_or(self, default: U, f: Callable[[T], U]) -> U:
        """Map or return default."""
        return f(self._value) if self._is_some else default

    def map_or_else(self, default: Callable[[], U], f: Callable[[T], U]) -> U:
        """Map or compute default."""
        return f(self._value) if self._is_some else default()

    def and_then(self, f: Callable[[T], "Option[U]"]) -> "Option[U]":
        """Chain operations that return Options."""
        if self._is_some:
            return f(self._value)
        return Option.none()

    def or_else(self, f: Callable[[], "Option[T]"]) -> "Option[T]":
        """Return self if Some, otherwise compute alternative."""
        if self._is_some:
            return self
        return f()

    def filter(self, predicate: Callable[[T], bool]) -> "Option[T]":
        """Filter the value with a predicate."""
        if self._is_some and predicate(self._value):
            return self
        return Option.none()

    def flatten(self: "Option[Option[T]]") -> "Option[T]":
        """Flatten nested Option."""
        if self._is_some:
            return self._value
        return Option.none()

    def ok_or(self, err: E) -> Result[T, E]:
        """Convert to Result with provided error."""
        if self._is_some:
            return Result.ok(self._value)
        return Result.err(err)

    def ok_or_else(self, err_f: Callable[[], E]) -> Result[T, E]:
        """Convert to Result with computed error."""
        if self._is_some:
            return Result.ok(self._value)
        return Result.err(err_f())

    def take(self) -> "Option[T]":
        """Take the value, leaving None in its place."""
        if self._is_some:
            value = self._value
            self._value = None
            self._is_some = False
            return Option.some(value)
        return Option.none()

    def replace(self, value: T) -> "Option[T]":
        """Replace current value, returning old value."""
        old = Option.some(self._value) if self._is_some else Option.none()
        self._value = value
        self._is_some = True
        return old

    def __iter__(self) -> Iterator[T]:
        """Iterate over value if Some."""
        if self._is_some:
            yield self._value

    def __bool__(self) -> bool:
        """Option is truthy if Some."""
        return self._is_some

    def __repr__(self) -> str:
        if self._is_some:
            return f"Some({self._value!r})"
        return "None"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Option):
            return False
        if self._is_some != other._is_some:
            return False
        if self._is_some:
            return self._value == other._value
        return True

    def __hash__(self) -> int:
        if self._is_some:
            return hash(("Some", self._value))
        return hash(("None",))


def Some(value: T) -> Option[T]:
    """Create a Some option."""
    return Option.some(value)


# Singleton None Option (renamed to avoid shadowing Python's None)
NONE: Option[Any] = Option.none()


def try_catch(
    f: Callable[..., T],
    *args,
    error_type: Type[E] = Exception,
    **kwargs,
) -> Result[T, E]:
    """Execute a function and catch exceptions as Result."""
    try:
        return Ok(f(*args, **kwargs))
    except error_type as e:
        return Err(e)


def collect_results(results: List[Result[T, E]]) -> Result[List[T], E]:
    """Collect a list of Results into a Result of list.

    Returns first error encountered, or Ok(list of values).
    """
    values = []
    for result in results:
        if result.is_err:
            return Err(result.err_value())
        values.append(result.ok_value())
    return Ok(values)


def collect_options(options: List[Option[T]]) -> Option[List[T]]:
    """Collect a list of Options into an Option of list.

    Returns None if any option is None.
    """
    values = []
    for opt in options:
        if opt.is_none:
            return Option.none()
        values.append(opt.unwrap())
    return Some(values)


class ResultBuilder(Generic[T, E]):
    """Builder for chaining Result operations with context."""

    def __init__(self, result: Result[T, E]):
        self._result = result
        self._context: dict = {}

    @classmethod
    def start(cls, value: T) -> "ResultBuilder[T, E]":
        """Start with an Ok value."""
        return cls(Ok(value))

    @classmethod
    def from_result(cls, result: Result[T, E]) -> "ResultBuilder[T, E]":
        """Start from an existing Result."""
        return cls(result)

    def with_context(self, key: str, value: Any) -> "ResultBuilder[T, E]":
        """Add context for error reporting."""
        self._context[key] = value
        return self

    def map(self, f: Callable[[T], U]) -> "ResultBuilder[U, E]":
        """Map the Ok value."""
        return ResultBuilder(self._result.map(f))

    def and_then(self, f: Callable[[T], Result[U, E]]) -> "ResultBuilder[U, E]":
        """Chain another Result operation."""
        return ResultBuilder(self._result.and_then(f))

    def build(self) -> Result[T, E]:
        """Get the final Result."""
        return self._result

    def unwrap(self) -> T:
        """Unwrap the final Result."""
        return self._result.unwrap()


def ensure(condition: bool, error: E) -> Result[None, E]:
    """Create Ok(None) if condition is true, else Err(error)."""
    if condition:
        return Ok(None)
    return Err(error)


def require(value: Optional[T], error: E) -> Result[T, E]:
    """Convert nullable to Result with error if None."""
    if value is not None:
        return Ok(value)
    return Err(error)


class RustBridgeError(Error):
    """Error from Rust FFI call."""

    def __init__(
        self,
        message: str,
        rust_code: Optional[int] = None,
        rust_source: Optional[str] = None,
    ):
        super().__init__(message, ErrorKind.FFI)
        self.rust_code = rust_code
        self.rust_source = rust_source


def from_ffi(
    success: bool,
    value: Optional[T] = None,
    error_code: Optional[int] = None,
    error_message: Optional[str] = None,
) -> Result[T, RustBridgeError]:
    """Convert FFI result tuple to Result type."""
    if success:
        return Ok(value)
    return Err(RustBridgeError(
        error_message or "Unknown FFI error",
        rust_code=error_code,
    ))


def result_from_exception(f: Callable[..., T]) -> Callable[..., Result[T, Error]]:
    """Decorator to convert exceptions to Result."""
    def wrapper(*args, **kwargs) -> Result[T, Error]:
        try:
            return Ok(f(*args, **kwargs))
        except Exception as e:
            return Err(Error(str(e), ErrorKind.UNKNOWN))
    wrapper.__name__ = f.__name__
    wrapper.__doc__ = f.__doc__
    return wrapper

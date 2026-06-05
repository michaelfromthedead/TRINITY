"""Tests for Result types for Rust bridge functions (T-CC-1.9)."""
import pytest

from engine.core.result import (
    NONE,
    Error,
    ErrorKind,
    Err,
    Ok,
    Option,
    Result,
    ResultBuilder,
    ResultError,
    RustBridgeError,
    Some,
    UnwrapError,
    collect_options,
    collect_results,
    ensure,
    from_ffi,
    require,
    result_from_exception,
    try_catch,
)


class TestErrorKind:
    """Tests for ErrorKind enum."""

    def test_all_kinds_exist(self):
        kinds = [
            ErrorKind.UNKNOWN,
            ErrorKind.IO,
            ErrorKind.PARSE,
            ErrorKind.VALIDATION,
            ErrorKind.TIMEOUT,
            ErrorKind.NOT_FOUND,
            ErrorKind.PERMISSION,
            ErrorKind.OVERFLOW,
            ErrorKind.UNDERFLOW,
            ErrorKind.GPU,
            ErrorKind.FFI,
            ErrorKind.NETWORK,
        ]
        assert len(kinds) == 12


class TestError:
    """Tests for Error dataclass."""

    def test_basic_error(self):
        err = Error("Something failed", ErrorKind.IO)
        assert str(err) == "Something failed"
        assert err.kind == ErrorKind.IO

    def test_error_chain(self):
        inner = Error("Inner error")
        outer = Error("Outer error", source=inner)

        chain = outer.chain()
        assert len(chain) == 2
        assert chain[0].message == "Outer error"
        assert chain[1].message == "Inner error"

    def test_with_context(self):
        err = Error("Error", ErrorKind.IO)
        err2 = err.with_context("file", "test.txt")

        assert err2.context["file"] == "test.txt"
        assert err.context is None  # Original unchanged

    def test_wrap(self):
        inner = Error("Inner")
        outer = inner.wrap("Wrapped")

        assert outer.message == "Wrapped"
        assert outer.source is inner


class TestResult:
    """Tests for Result type."""

    def test_ok_creation(self):
        result = Ok(42)
        assert result.is_ok
        assert not result.is_err

    def test_err_creation(self):
        result = Err("error")
        assert not result.is_ok
        assert result.is_err

    def test_ok_value(self):
        result = Ok(42)
        assert result.ok_value() == 42
        assert result.err_value() is None

    def test_err_value(self):
        result = Err("error")
        assert result.ok_value() is None
        assert result.err_value() == "error"

    def test_unwrap_ok(self):
        result = Ok(42)
        assert result.unwrap() == 42

    def test_unwrap_err_raises(self):
        result = Err("error")
        with pytest.raises(UnwrapError):
            result.unwrap()

    def test_unwrap_or(self):
        assert Ok(42).unwrap_or(0) == 42
        assert Err("e").unwrap_or(0) == 0

    def test_unwrap_or_else(self):
        assert Ok(42).unwrap_or_else(lambda e: 0) == 42
        assert Err("e").unwrap_or_else(lambda e: len(e)) == 1

    def test_unwrap_err(self):
        result = Err("error")
        assert result.unwrap_err() == "error"

    def test_unwrap_err_raises_on_ok(self):
        result = Ok(42)
        with pytest.raises(UnwrapError):
            result.unwrap_err()

    def test_expect(self):
        assert Ok(42).expect("should work") == 42

    def test_expect_raises(self):
        with pytest.raises(UnwrapError, match="custom message"):
            Err("e").expect("custom message")

    def test_expect_err(self):
        assert Err("error").expect_err("should work") == "error"

    def test_expect_err_raises(self):
        with pytest.raises(UnwrapError, match="custom message"):
            Ok(42).expect_err("custom message")

    def test_map(self):
        result = Ok(5).map(lambda x: x * 2)
        assert result.unwrap() == 10

        result = Err("e").map(lambda x: x * 2)
        assert result.is_err

    def test_map_err(self):
        result = Err("error").map_err(lambda e: e.upper())
        assert result.unwrap_err() == "ERROR"

        result = Ok(42).map_err(lambda e: e.upper())
        assert result.is_ok

    def test_map_or(self):
        assert Ok(5).map_or(0, lambda x: x * 2) == 10
        assert Err("e").map_or(0, lambda x: x * 2) == 0

    def test_map_or_else(self):
        assert Ok(5).map_or_else(lambda e: -1, lambda x: x * 2) == 10
        assert Err("err").map_or_else(lambda e: len(e), lambda x: x * 2) == 3

    def test_and_then(self):
        def double(x: int) -> Result[int, str]:
            return Ok(x * 2)

        result = Ok(5).and_then(double)
        assert result.unwrap() == 10

        result = Err("e").and_then(double)
        assert result.is_err

    def test_or_else(self):
        def recover(e: str) -> Result[int, str]:
            return Ok(0)

        result = Err("e").or_else(recover)
        assert result.unwrap() == 0

        result = Ok(42).or_else(recover)
        assert result.unwrap() == 42

    def test_flatten(self):
        nested: Result[Result[int, str], str] = Ok(Ok(42))
        flat = nested.flatten()
        assert flat.unwrap() == 42

        nested2: Result[Result[int, str], str] = Err("outer")
        flat2 = nested2.flatten()
        assert flat2.is_err

    def test_transpose(self):
        result: Result[int | None, str] = Ok(42)
        opt = result.transpose()
        assert opt.unwrap() == 42

        result2: Result[int | None, str] = Ok(None)
        opt2 = result2.transpose()
        assert opt2 is None

        result3: Result[int | None, str] = Err("e")
        opt3 = result3.transpose()
        assert opt3.is_err

    def test_bool(self):
        assert Ok(42)
        assert not Err("e")

    def test_repr(self):
        assert repr(Ok(42)) == "Ok(42)"
        assert repr(Err("error")) == "Err('error')"

    def test_eq(self):
        assert Ok(42) == Ok(42)
        assert Err("e") == Err("e")
        assert Ok(42) != Err(42)
        assert Ok(42) != Ok(43)

    def test_hash(self):
        d = {Ok(42): "ok", Err("e"): "err"}
        assert d[Ok(42)] == "ok"
        assert d[Err("e")] == "err"


class TestOption:
    """Tests for Option type."""

    def test_some_creation(self):
        opt = Some(42)
        assert opt.is_some
        assert not opt.is_none

    def test_none_creation(self):
        opt = Option.none()
        assert opt.is_none
        assert not opt.is_some

    def test_none_singleton(self):
        assert NONE.is_none

    def test_from_nullable(self):
        assert Option.from_nullable(42).unwrap() == 42
        assert Option.from_nullable(None).is_none

    def test_unwrap(self):
        assert Some(42).unwrap() == 42

    def test_unwrap_none_raises(self):
        with pytest.raises(UnwrapError):
            Option.none().unwrap()

    def test_unwrap_or(self):
        assert Some(42).unwrap_or(0) == 42
        assert Option.none().unwrap_or(0) == 0

    def test_unwrap_or_else(self):
        assert Some(42).unwrap_or_else(lambda: 0) == 42
        assert Option.none().unwrap_or_else(lambda: 99) == 99

    def test_expect(self):
        assert Some(42).expect("should work") == 42

    def test_expect_raises(self):
        with pytest.raises(UnwrapError, match="custom"):
            Option.none().expect("custom")

    def test_map(self):
        result = Some(5).map(lambda x: x * 2)
        assert result.unwrap() == 10

        result = Option.none().map(lambda x: x * 2)
        assert result.is_none

    def test_map_or(self):
        assert Some(5).map_or(0, lambda x: x * 2) == 10
        assert Option.none().map_or(0, lambda x: x * 2) == 0

    def test_map_or_else(self):
        assert Some(5).map_or_else(lambda: 0, lambda x: x * 2) == 10
        assert Option.none().map_or_else(lambda: 99, lambda x: x * 2) == 99

    def test_and_then(self):
        def double(x: int) -> Option[int]:
            return Some(x * 2)

        result = Some(5).and_then(double)
        assert result.unwrap() == 10

        result = Option.none().and_then(double)
        assert result.is_none

    def test_or_else(self):
        def fallback() -> Option[int]:
            return Some(99)

        result = Some(42).or_else(fallback)
        assert result.unwrap() == 42

        result = Option.none().or_else(fallback)
        assert result.unwrap() == 99

    def test_filter(self):
        result = Some(10).filter(lambda x: x > 5)
        assert result.unwrap() == 10

        result = Some(3).filter(lambda x: x > 5)
        assert result.is_none

        result = Option.none().filter(lambda x: True)
        assert result.is_none

    def test_flatten(self):
        nested: Option[Option[int]] = Some(Some(42))
        flat = nested.flatten()
        assert flat.unwrap() == 42

        nested2: Option[Option[int]] = Option.none()
        flat2 = nested2.flatten()
        assert flat2.is_none

    def test_ok_or(self):
        result = Some(42).ok_or("error")
        assert result.is_ok
        assert result.unwrap() == 42

        result = Option.none().ok_or("error")
        assert result.is_err
        assert result.unwrap_err() == "error"

    def test_ok_or_else(self):
        result = Some(42).ok_or_else(lambda: "error")
        assert result.unwrap() == 42

        result = Option.none().ok_or_else(lambda: "computed")
        assert result.unwrap_err() == "computed"

    def test_take(self):
        opt = Some(42)
        taken = opt.take()
        assert taken.unwrap() == 42
        assert opt.is_none

    def test_replace(self):
        opt = Some(42)
        old = opt.replace(99)
        assert old.unwrap() == 42
        assert opt.unwrap() == 99

    def test_iter(self):
        values = list(Some(42))
        assert values == [42]

        values = list(Option.none())
        assert values == []

    def test_bool(self):
        assert Some(42)
        assert not Option.none()

    def test_repr(self):
        assert repr(Some(42)) == "Some(42)"
        assert repr(Option.none()) == "None"

    def test_eq(self):
        assert Some(42) == Some(42)
        assert Option.none() == Option.none()
        assert Some(42) != Option.none()
        assert Some(42) != Some(43)

    def test_hash(self):
        d = {Some(42): "some", Option.none(): "none"}
        assert d[Some(42)] == "some"


class TestTryCatch:
    """Tests for try_catch function."""

    def test_success(self):
        result = try_catch(lambda: 42)
        assert result.is_ok
        assert result.unwrap() == 42

    def test_exception(self):
        def failing():
            raise ValueError("error")

        result = try_catch(failing)
        assert result.is_err
        assert isinstance(result.unwrap_err(), ValueError)

    def test_specific_error_type(self):
        def failing():
            raise ValueError("error")

        result = try_catch(failing, error_type=ValueError)
        assert result.is_err

    def test_with_args(self):
        def add(a, b):
            return a + b

        result = try_catch(add, 2, 3)
        assert result.unwrap() == 5


class TestCollectResults:
    """Tests for collect_results function."""

    def test_all_ok(self):
        results = [Ok(1), Ok(2), Ok(3)]
        collected = collect_results(results)
        assert collected.unwrap() == [1, 2, 3]

    def test_with_err(self):
        results = [Ok(1), Err("error"), Ok(3)]
        collected = collect_results(results)
        assert collected.is_err
        assert collected.unwrap_err() == "error"

    def test_empty(self):
        collected = collect_results([])
        assert collected.unwrap() == []


class TestCollectOptions:
    """Tests for collect_options function."""

    def test_all_some(self):
        options = [Some(1), Some(2), Some(3)]
        collected = collect_options(options)
        assert collected.unwrap() == [1, 2, 3]

    def test_with_none(self):
        options = [Some(1), Option.none(), Some(3)]
        collected = collect_options(options)
        assert collected.is_none

    def test_empty(self):
        collected = collect_options([])
        assert collected.unwrap() == []


class TestResultBuilder:
    """Tests for ResultBuilder."""

    def test_start(self):
        result = ResultBuilder.start(42).build()
        assert result.unwrap() == 42

    def test_from_result(self):
        builder = ResultBuilder.from_result(Ok(42))
        assert builder.build().unwrap() == 42

    def test_map(self):
        result = (
            ResultBuilder.start(5)
            .map(lambda x: x * 2)
            .build()
        )
        assert result.unwrap() == 10

    def test_and_then(self):
        result = (
            ResultBuilder.start(5)
            .and_then(lambda x: Ok(x * 2))
            .build()
        )
        assert result.unwrap() == 10

    def test_with_context(self):
        builder = ResultBuilder.start(42).with_context("key", "value")
        assert builder._context["key"] == "value"

    def test_unwrap(self):
        value = ResultBuilder.start(42).unwrap()
        assert value == 42


class TestEnsure:
    """Tests for ensure function."""

    def test_true_condition(self):
        result = ensure(True, "error")
        assert result.is_ok

    def test_false_condition(self):
        result = ensure(False, "error")
        assert result.is_err
        assert result.unwrap_err() == "error"


class TestRequire:
    """Tests for require function."""

    def test_with_value(self):
        result = require(42, "error")
        assert result.unwrap() == 42

    def test_with_none(self):
        result = require(None, "error")
        assert result.is_err
        assert result.unwrap_err() == "error"


class TestRustBridgeError:
    """Tests for RustBridgeError."""

    def test_basic(self):
        err = RustBridgeError("FFI failed", rust_code=1)
        assert err.message == "FFI failed"
        assert err.rust_code == 1
        assert err.kind == ErrorKind.FFI


class TestFromFFI:
    """Tests for from_ffi function."""

    def test_success(self):
        result = from_ffi(True, value=42)
        assert result.is_ok
        assert result.unwrap() == 42

    def test_failure(self):
        result = from_ffi(False, error_code=1, error_message="FFI error")
        assert result.is_err
        err = result.unwrap_err()
        assert err.rust_code == 1
        assert err.message == "FFI error"

    def test_failure_no_message(self):
        result = from_ffi(False)
        err = result.unwrap_err()
        assert err.message == "Unknown FFI error"


class TestResultFromException:
    """Tests for result_from_exception decorator."""

    def test_success(self):
        @result_from_exception
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result.unwrap() == 5

    def test_exception(self):
        @result_from_exception
        def failing():
            raise ValueError("error")

        result = failing()
        assert result.is_err
        assert "error" in result.unwrap_err().message

    def test_preserves_name(self):
        @result_from_exception
        def my_func():
            pass

        assert my_func.__name__ == "my_func"


class TestChaining:
    """Tests for chained operations."""

    def test_complex_chain(self):
        def parse_int(s: str) -> Result[int, str]:
            try:
                return Ok(int(s))
            except ValueError:
                return Err(f"Cannot parse '{s}' as int")

        def validate_positive(n: int) -> Result[int, str]:
            if n > 0:
                return Ok(n)
            return Err("Number must be positive")

        def double(n: int) -> Result[int, str]:
            return Ok(n * 2)

        result = (
            parse_int("5")
            .and_then(validate_positive)
            .and_then(double)
        )
        assert result.unwrap() == 10

        result = parse_int("abc").and_then(validate_positive)
        assert result.is_err

        result = parse_int("-5").and_then(validate_positive)
        assert result.is_err

    def test_option_chain(self):
        def get_config() -> Option[dict]:
            return Some({"database": {"host": "localhost"}})

        def get_database(config: dict) -> Option[dict]:
            return Option.from_nullable(config.get("database"))

        def get_host(db: dict) -> Option[str]:
            return Option.from_nullable(db.get("host"))

        host = (
            get_config()
            .and_then(get_database)
            .and_then(get_host)
        )
        assert host.unwrap() == "localhost"


class TestEdgeCases:
    """Edge case tests."""

    def test_result_with_none_value(self):
        result = Ok(None)
        assert result.is_ok
        assert result.unwrap() is None

    def test_option_with_false_value(self):
        opt = Some(False)
        assert opt.is_some
        assert opt.unwrap() is False

    def test_option_with_zero(self):
        opt = Some(0)
        assert opt.is_some
        assert opt.unwrap() == 0

    def test_option_with_empty_string(self):
        opt = Some("")
        assert opt.is_some
        assert opt.unwrap() == ""

    def test_result_with_result_value(self):
        nested = Ok(Ok(42))
        assert nested.is_ok
        inner = nested.unwrap()
        assert inner.is_ok
        assert inner.unwrap() == 42

    def test_error_chain_depth(self):
        err = Error("level1")
        for i in range(2, 6):
            err = Error(f"level{i}", source=err)

        chain = err.chain()
        assert len(chain) == 5
        assert chain[0].message == "level5"
        assert chain[4].message == "level1"

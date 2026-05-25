"""
Tests for assertions.py - Custom assertions with @invariant decorator.
"""

import pytest

from engine.tooling.crash.assertions import (
    AssertionConfig,
    ContractMixin,
    InvariantError,
    PostconditionError,
    PreconditionError,
    assert_in_range,
    assert_not_none,
    assert_type,
    check,
    disable_assertions,
    enable_assertions,
    ensure,
    get_config,
    invariant,
    postcondition,
    precondition,
    require,
    set_config,
    with_contracts,
)


class TestAssertionConfig:
    """Tests for AssertionConfig dataclass."""

    def test_default_config(self):
        config = AssertionConfig()

        assert config.enabled is True
        assert config.check_invariants is True
        assert config.check_preconditions is True
        assert config.check_postconditions is True

    def test_enable_disable(self):
        enable_assertions()
        assert get_config().enabled is True

        disable_assertions()
        assert get_config().enabled is False

        # Reset
        enable_assertions()


class TestInvariantDecorator:
    """Tests for @invariant decorator."""

    def test_basic_invariant(self):
        @invariant(lambda self: self.value >= 0, "Value must be non-negative")
        class Counter:
            def __init__(self):
                self.value = 0

            def increment(self):
                self.value += 1

        c = Counter()
        c.increment()
        assert c.value == 1

    def test_invariant_violation_in_init(self):
        @invariant(lambda self: self.value >= 0, "Value must be non-negative")
        class BadCounter:
            def __init__(self):
                self.value = -1

        with pytest.raises(InvariantError):
            BadCounter()

    def test_invariant_violation_in_method(self):
        @invariant(lambda self: self.value >= 0, "Value must be non-negative")
        class Counter:
            def __init__(self):
                self.value = 0

            def decrement(self):
                self.value -= 1

        c = Counter()
        with pytest.raises(InvariantError):
            c.decrement()


class TestPreconditionDecorator:
    """Tests for @precondition decorator."""

    def test_precondition_pass(self):
        @precondition(lambda x: x >= 0, "x must be non-negative", "x")
        def sqrt(x):
            return x ** 0.5

        result = sqrt(4)
        assert result == 2.0

    def test_precondition_fail(self):
        @precondition(lambda x: x >= 0, "x must be non-negative", "x")
        def sqrt(x):
            return x ** 0.5

        with pytest.raises(PreconditionError):
            sqrt(-1)


class TestPostconditionDecorator:
    """Tests for @postcondition decorator."""

    def test_postcondition_pass(self):
        @postcondition(lambda r: r >= 0, "Result must be non-negative")
        def abs_value(x):
            return abs(x)

        result = abs_value(-5)
        assert result == 5

    def test_postcondition_fail(self):
        @postcondition(lambda r: r > 0, "Result must be positive")
        def always_negative():
            return -1

        with pytest.raises(PostconditionError):
            always_negative()


class TestCheckFunction:
    """Tests for check function."""

    def test_check_pass(self):
        check(True, "Should not fail")

    def test_check_fail(self):
        with pytest.raises(AssertionError):
            check(False, "Check failed")

    def test_check_custom_exception(self):
        with pytest.raises(ValueError):
            check(False, "Value error", exception_type=ValueError)


class TestEnsureFunction:
    """Tests for ensure function."""

    def test_ensure_pass(self):
        ensure(True, "Should not fail")

    def test_ensure_fail(self):
        with pytest.raises(PostconditionError):
            ensure(False, "Ensure failed")


class TestRequireFunction:
    """Tests for require function."""

    def test_require_pass(self):
        require(True, "Should not fail")

    def test_require_fail(self):
        with pytest.raises(PreconditionError):
            require(False, "Require failed")


class TestContractMixin:
    """Tests for ContractMixin class."""

    def test_basic_usage(self):
        class Account(ContractMixin):
            def __init__(self, balance=0):
                self.balance = balance
                self._check_invariants()

            def _invariant(self):
                return self.balance >= 0

            def deposit(self, amount):
                self._require(amount > 0, "Amount must be positive")
                self.balance += amount
                self._check_invariants()

        account = Account(100)
        account.deposit(50)
        assert account.balance == 150

    def test_invariant_check(self):
        class Account(ContractMixin):
            def __init__(self, balance=0):
                self.balance = balance

            def _invariant(self):
                return self.balance >= 0

        account = Account(100)
        account.balance = -1

        with pytest.raises(InvariantError):
            account._check_invariants()

    def test_require_check(self):
        class Account(ContractMixin):
            def withdraw(self, amount):
                self._require(amount > 0, "Amount must be positive")

        account = Account()

        with pytest.raises(PreconditionError):
            account.withdraw(-10)


class TestWithContractsDecorator:
    """Tests for @with_contracts decorator."""

    def test_adds_contract_support(self):
        @with_contracts(lambda self: self.value >= 0, "Value must be non-negative")
        class Counter:
            def __init__(self):
                self.value = 0

        c = Counter()
        assert hasattr(c, "_check_invariants")
        assert hasattr(c, "_require")
        assert hasattr(c, "_ensure")


class TestTypeAssertions:
    """Tests for type assertion functions."""

    def test_assert_type_pass(self):
        assert_type(42, int)
        assert_type("hello", str)
        assert_type([1, 2], list)

    def test_assert_type_fail(self):
        with pytest.raises(TypeError):
            assert_type("not an int", int)

    def test_assert_not_none_pass(self):
        assert_not_none(42)
        assert_not_none("")
        assert_not_none(0)

    def test_assert_not_none_fail(self):
        with pytest.raises(ValueError):
            assert_not_none(None)

    def test_assert_in_range_pass(self):
        assert_in_range(5, 0, 10)
        assert_in_range(0, 0, 10)
        assert_in_range(10, 0, 10)

    def test_assert_in_range_fail(self):
        with pytest.raises(ValueError):
            assert_in_range(15, 0, 10)


class TestAssertionsDisabled:
    """Tests that assertions can be disabled."""

    def test_invariant_skipped_when_disabled(self):
        # Save original config
        original = get_config()

        try:
            disable_assertions()

            @invariant(lambda self: False, "Always fails")
            class AlwaysFails:
                def __init__(self):
                    self.value = 0

            # Should not raise when assertions disabled
            obj = AlwaysFails()
            assert obj.value == 0

        finally:
            # Restore config
            set_config(original)
            enable_assertions()

    def test_check_skipped_when_disabled(self):
        original = get_config()

        try:
            disable_assertions()
            # Should not raise
            check(False, "Would fail if enabled")
        finally:
            set_config(original)
            enable_assertions()

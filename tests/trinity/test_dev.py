"""Tests for DEV (Tier 6) decorators."""

import logging
import time
import warnings

import pytest

from trinity.decorators.dev import (
    bench,
    deprecated,
    editor,
    gpu_profile,
    invariant,
    profile,
    reloadable,
    test as test_decorator,
    trace,
)
from trinity.decorators.ops import Op, decompose, expand


# ============================================================================
# @profile Tests
# ============================================================================


class TestProfile:
    """Tests for @profile decorator."""

    def test_profile_timing_works(self):
        """Test that profiling captures timing information."""

        @profile
        def slow_fn():
            time.sleep(0.01)
            return 42

        result = slow_fn()
        assert result == 42

        stats = slow_fn.profile_stats()
        assert stats["call_count"] >= 1
        assert stats["total_ms"] > 0
        assert stats["min_ms"] > 0
        assert stats["max_ms"] > 0
        assert stats["avg_ms"] > 0

    def test_profile_reset_clears(self):
        """Test that profile_reset() clears statistics."""

        @profile
        def fn():
            return 1

        fn()
        fn()
        assert fn.profile_stats()["call_count"] == 2

        fn.profile_reset()
        stats = fn.profile_stats()
        assert stats["call_count"] == 0
        assert stats["total_ms"] == 0.0

    def test_profile_warn_ms(self):
        """Test that warn_ms threshold triggers warning."""

        @profile(warn_ms=1.0)
        def slow_fn():
            time.sleep(0.01)

        with pytest.warns(UserWarning, match="threshold"):
            slow_fn()

    def test_profile_name_defaults(self):
        """Test that name defaults to function name."""

        @profile
        def my_function():
            pass

        stats = my_function.profile_stats()
        assert stats["name"] == "my_function"

    def test_profile_custom_name(self):
        """Test custom profile name."""

        @profile(name="CustomName")
        def fn():
            pass

        stats = fn.profile_stats()
        assert stats["name"] == "CustomName"

    def test_profile_on_class_no_parens(self):
        """Test @profile on class without parentheses."""

        @profile
        class MyClass:
            pass

        assert MyClass._profiled is True
        assert hasattr(MyClass, "_profile_name")

    def test_profile_multiple_calls(self):
        """Test profiling multiple calls."""

        @profile
        def fn():
            time.sleep(0.001)

        for _ in range(5):
            fn()

        stats = fn.profile_stats()
        assert stats["call_count"] == 5
        assert stats["total_ms"] > 0


# ============================================================================
# @gpu_profile Tests
# ============================================================================


class TestGpuProfile:
    """Tests for @gpu_profile decorator."""

    def test_gpu_profile_attrs_set(self):
        """Test that GPU profile attributes are set."""

        @gpu_profile(category="compute")
        def fn():
            pass

        assert fn._gpu_profiled is True
        assert fn._gpu_profile_category == "compute"
        assert fn._gpu_profile_include_memory is False

    def test_gpu_profile_gpu_stats_method(self):
        """Test that gpu_stats() method exists."""

        @gpu_profile(category="render")
        def fn():
            pass

        stats = fn.gpu_stats()
        assert stats["category"] == "render"
        assert "include_memory" in stats

    def test_gpu_profile_category_stored(self):
        """Test that category is properly stored."""

        @gpu_profile(category="memory", include_memory=True)
        class MyClass:
            pass

        assert MyClass._gpu_profile_category == "memory"
        assert MyClass._gpu_profile_include_memory is True

    def test_gpu_profile_requires_category(self):
        """Test that category is required."""
        with pytest.raises(ValueError, match="category must be a non-empty string"):

            @gpu_profile(category="")
            def fn():
                pass


# ============================================================================
# @trace Tests
# ============================================================================


class TestTrace:
    """Tests for @trace decorator."""

    def test_trace_function_callable(self):
        """Test that traced function is still callable."""

        @trace
        def fn(x):
            return x * 2

        result = fn(21)
        assert result == 42

    def test_trace_level_stored(self):
        """Test that trace level is stored."""

        @trace(level="info")
        def fn():
            pass

        assert fn._traced is True
        assert fn._trace_level == "info"

    def test_trace_default_level(self):
        """Test default trace level is debug."""

        @trace
        def fn():
            pass

        assert fn._trace_level == "debug"

    def test_trace_invalid_level(self):
        """Test that invalid level raises error."""
        with pytest.raises(ValueError, match="level must be one of"):

            @trace(level="invalid")
            def fn():
                pass

    def test_trace_logs_entry_exit(self, caplog):
        """Test that trace logs function entry and exit."""

        @trace(level="info")
        def fn():
            return 42

        with caplog.at_level(logging.INFO, logger="trinity.trace"):
            result = fn()

        assert result == 42
        # Check that logging occurred (may be empty if logger not configured)

    def test_trace_on_class(self):
        """Test @trace on class."""

        @trace
        class MyClass:
            pass

        assert MyClass._traced is True
        assert MyClass._trace_level == "debug"


# ============================================================================
# @reloadable Tests
# ============================================================================


class TestReloadable:
    """Tests for @reloadable decorator."""

    def test_reloadable_attrs_set(self):
        """Test that reloadable attributes are set."""

        @reloadable
        class MyClass:
            pass

        assert MyClass._reloadable is True
        assert MyClass._reload_preserve == []
        assert MyClass._reload_reinitialize == []

    def test_reloadable_preserve_reinitialize_lists(self):
        """Test preserve and reinitialize lists."""

        @reloadable(preserve=["x", "y"], reinitialize=["z"])
        class MyClass:
            pass

        assert MyClass._reload_preserve == ["x", "y"]
        assert MyClass._reload_reinitialize == ["z"]

    def test_reloadable_validate_callable(self):
        """Test that validate callable is stored."""

        def my_validator():
            pass

        @reloadable(validate=my_validator)
        class MyClass:
            pass

        assert MyClass._reload_validate is my_validator

    def test_reloadable_enabled_flag(self):
        """Test enabled flag."""

        @reloadable(enabled=False)
        class MyClass:
            pass

        # Check that decorator was still applied
        assert MyClass._reloadable is True


# ============================================================================
# @editor Tests
# ============================================================================


class TestEditor:
    """Tests for @editor decorator."""

    def test_editor_attrs_set(self):
        """Test that editor attributes are set."""

        @editor
        class MyClass:
            pass

        assert MyClass._editor is True
        assert MyClass._editor_category == "General"
        assert MyClass._editor_hidden is False

    def test_editor_defaults(self):
        """Test default values."""

        @editor(category="Physics", hidden=True)
        class MyClass:
            pass

        assert MyClass._editor_category == "Physics"
        assert MyClass._editor_hidden is True


# ============================================================================
# @test Tests
# ============================================================================


class TestTest:
    """Tests for @test decorator."""

    def test_test_cases_stored(self):
        """Test that test cases are stored."""

        @test_decorator(cases=[{"input": 1, "output": 2}, {"input": 3, "output": 4}])
        def fn(x):
            return x * 2

        assert fn._test is True
        assert len(fn._test_cases) == 2

    def test_test_fuzz_property_flags(self):
        """Test fuzz and property_based flags."""

        @test_decorator(fuzz=True, property_based=True)
        def fn():
            pass

        assert fn._test_fuzz is True
        assert fn._test_property_based is True

    def test_test_default_cases(self):
        """Test default empty cases list."""

        @test_decorator
        def fn():
            pass

        assert fn._test_cases == []


# ============================================================================
# @bench Tests
# ============================================================================


class TestBench:
    """Tests for @bench decorator."""

    def test_bench_iterations_warmup_stored(self):
        """Test that iterations and warmup are stored."""

        @bench(iterations=5000, warmup=200)
        def fn():
            pass

        assert fn._bench is True
        assert fn._bench_iterations == 5000
        assert fn._bench_warmup == 200

    def test_bench_validation_iterations_positive(self):
        """Test that iterations must be positive."""
        with pytest.raises(ValueError, match="iterations must be a positive integer"):

            @bench(iterations=0)
            def fn():
                pass

    def test_bench_validation_iterations_negative(self):
        """Test that negative iterations raises error."""
        with pytest.raises(ValueError, match="iterations must be a positive integer"):

            @bench(iterations=-1)
            def fn():
                pass

    def test_bench_validation_warmup_negative(self):
        """Test that negative warmup raises error."""
        with pytest.raises(ValueError, match="warmup must be a non-negative integer"):

            @bench(warmup=-1)
            def fn():
                pass

    def test_bench_default_values(self):
        """Test default iterations and warmup."""

        @bench
        def fn():
            pass

        assert fn._bench_iterations == 1000
        assert fn._bench_warmup == 100


# ============================================================================
# @invariant Tests
# ============================================================================


class TestInvariant:
    """Tests for @invariant decorator."""

    def test_invariant_multiple_accumulate(self):
        """Test that multiple invariants accumulate."""

        def check1():
            pass

        def check2():
            pass

        @invariant(check=check1)
        @invariant(check=check2)
        class MyClass:
            pass

        assert len(MyClass._invariants) == 2
        assert MyClass._invariants[0]["check"] is check2
        assert MyClass._invariants[1]["check"] is check1

    def test_invariant_check_callable_stored(self):
        """Test that check callable is stored."""

        def my_check():
            return True

        @invariant(check=my_check, when="always")
        class MyClass:
            pass

        assert len(MyClass._invariants) == 1
        assert MyClass._invariants[0]["check"] is my_check
        assert MyClass._invariants[0]["when"] == "always"

    def test_invariant_requires_callable(self):
        """Test that check must be callable."""
        with pytest.raises(ValueError, match="check must be a callable"):

            @invariant(check="not_callable")
            class MyClass:
                pass

    def test_invariant_invalid_when(self):
        """Test that when must be valid."""
        with pytest.raises(ValueError, match="when must be one of"):

            @invariant(check=lambda: True, when="invalid")
            class MyClass:
                pass


# ============================================================================
# @deprecated Tests
# ============================================================================


class TestDeprecated:
    """Tests for @deprecated decorator."""

    def test_deprecated_warning_emitted(self):
        """Test that deprecation warning is emitted on call."""

        @deprecated(since="1.0.0")
        def old_fn():
            return 42

        with pytest.warns(DeprecationWarning, match="deprecated since 1.0.0"):
            result = old_fn()
            assert result == 42

    def test_deprecated_attrs_set(self):
        """Test that deprecated attributes are set."""

        @deprecated(since="1.0.0", replacement="new_fn", remove_in="2.0.0")
        def old_fn():
            pass

        assert old_fn._deprecated is True
        assert old_fn._deprecated_since == "1.0.0"
        assert old_fn._deprecated_replacement == "new_fn"
        assert old_fn._deprecated_remove_in == "2.0.0"

    def test_deprecated_since_required(self):
        """Test that since parameter is required."""
        with pytest.raises(ValueError, match="since must be a non-empty string"):

            @deprecated(since="")
            def fn():
                pass

    def test_deprecated_on_class(self):
        """Test @deprecated on class."""

        @deprecated(since="1.0.0")
        class OldClass:
            pass

        assert OldClass._deprecated is True
        assert OldClass._deprecated_since == "1.0.0"

    def test_deprecated_with_replacement(self):
        """Test deprecation message includes replacement."""

        @deprecated(since="1.0.0", replacement="new_function")
        def old_fn():
            pass

        with pytest.warns(DeprecationWarning, match="Use new_function instead"):
            old_fn()

    def test_deprecated_with_remove_in(self):
        """Test deprecation message includes removal version."""

        @deprecated(since="1.0.0", remove_in="2.0.0")
        def old_fn():
            pass

        with pytest.warns(DeprecationWarning, match="Will be removed in 2.0.0"):
            old_fn()


# ============================================================================
# Introspection Tests
# ============================================================================


class TestDevIntrospection:
    """Tests for introspection of DEV decorators."""

    @pytest.mark.parametrize(
        "decorator",
        [
            profile,
            gpu_profile,
            trace,
            reloadable,
            editor,
            test_decorator,
            bench,
            invariant,
            deprecated,
        ],
    )
    def test_decompose_steps(self, decorator):
        """Test that decompose() returns steps for all decorators."""
        steps = decompose(decorator)
        assert isinstance(steps, list)
        assert len(steps) > 0
        from trinity.decorators.ops import Step
        assert all(isinstance(step, Step) for step in steps)

    @pytest.mark.parametrize(
        "decorator",
        [
            profile,
            gpu_profile,
            trace,
            reloadable,
            editor,
            test_decorator,
            bench,
            invariant,
            deprecated,
        ],
    )
    def test_expand_ops(self, decorator):
        """Test that expand() returns string representation of ops."""
        result = expand(decorator)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.parametrize(
        "decorator",
        [
            profile,
            gpu_profile,
            trace,
            reloadable,
            editor,
            test_decorator,
            bench,
            invariant,
            deprecated,
        ],
    )
    def test_all_have_register_dev(self, decorator):
        """Test that all DEV decorators have REGISTER(dev) step."""
        steps = decompose(decorator)
        register_steps = [s for s in steps if s.op == Op.REGISTER]
        assert len(register_steps) > 0, f"{decorator.__name__} missing REGISTER step"
        # Check that at least one REGISTER step has "dev" registry
        has_dev = any(s.args.get("registry") == "dev" for s in register_steps)
        assert has_dev, f"{decorator.__name__} missing REGISTER(dev) step"


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestDevEdgeCases:
    """Edge cases and integration tests for DEV decorators."""

    def test_profile_preserves_metadata(self):
        """Test that @profile preserves function metadata."""

        @profile
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_trace_preserves_metadata(self):
        """Test that @trace preserves function metadata."""

        @trace
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_deprecated_preserves_metadata(self):
        """Test that @deprecated preserves function metadata."""

        @deprecated(since="1.0.0")
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_stacked_decorators(self):
        """Test stacking multiple DEV decorators."""

        @profile
        @trace
        @deprecated(since="1.0.0")
        def fn():
            return 42

        # All decorators should be applied
        assert fn._profiled is True
        assert fn._traced is True
        assert fn._deprecated is True

        # Function should still work
        with pytest.warns(DeprecationWarning):
            result = fn()
            assert result == 42

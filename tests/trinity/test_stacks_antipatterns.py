"""Tests for stack anti-pattern validation."""

from __future__ import annotations

import warnings

import pytest

from trinity.decorators.stacks import Stack, stack, _validate_stack_combination


# ---------------------------------------------------------------------------
# Helpers -- lightweight fakes so tests don't depend on full decorator infra
# ---------------------------------------------------------------------------

def _fake_decorator(name: str):
    """Return a callable that mimics a make_decorator product."""
    def _dec(cls):
        return cls
    _dec.__name__ = name
    _dec.__qualname__ = name
    _dec._decorator_name = name
    return _dec


parallel = _fake_decorator("parallel")
exclusive = _fake_decorator("exclusive")
reloadable = _fake_decorator("reloadable")
build_only = _fake_decorator("build_only")
tag = _fake_decorator("tag")
serializable = _fake_decorator("serializable")
networked = _fake_decorator("networked")
track_changes = _fake_decorator("track_changes")


# ---------------------------------------------------------------------------
# Hard-error anti-patterns
# ---------------------------------------------------------------------------

class TestHardErrors:
    def test_parallel_exclusive_raises(self):
        s = stack(parallel, exclusive)
        with pytest.raises(ValueError, match="contradictory"):
            @s
            class Bad:
                pass

    def test_tag_serializable_raises(self):
        s = stack(tag, serializable)
        with pytest.raises(ValueError, match="tags have no data to serialize"):
            @s
            class Bad:
                pass

    def test_tag_serializable_order_irrelevant(self):
        s = stack(serializable, tag)
        with pytest.raises(ValueError, match="tags have no data to serialize"):
            @s
            class Bad:
                pass

    def test_parallel_exclusive_with_others_still_raises(self):
        """Adding extra decorators should not mask the conflict."""
        s = stack(tag, parallel, exclusive)
        with pytest.raises(ValueError, match="contradictory"):
            @s
            class Bad:
                pass


# ---------------------------------------------------------------------------
# Warning anti-patterns
# ---------------------------------------------------------------------------

class TestWarnings:
    def test_networked_without_track_changes_warns(self):
        s = stack(networked)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            @s
            class Foo:
                pass
            assert any("delta sync" in str(x.message) for x in w), (
                f"Expected 'delta sync' warning, got {[str(x.message) for x in w]}"
            )

    def test_networked_with_track_changes_no_warning(self):
        s = stack(networked, track_changes)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            @s
            class Foo:
                pass
            delta_warnings = [x for x in w if "delta sync" in str(x.message)]
            assert len(delta_warnings) == 0, (
                f"Expected no delta sync warning, got {delta_warnings}"
            )

    def test_reloadable_without_build_only_warns(self):
        s = stack(reloadable)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            @s
            class Foo:
                pass
            assert any("hot reload" in str(x.message) for x in w), (
                f"Expected 'hot reload' warning, got {[str(x.message) for x in w]}"
            )

    def test_reloadable_with_build_only_no_warning(self):
        s = stack(reloadable, build_only)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            @s
            class Foo:
                pass
            reload_warnings = [x for x in w if "hot reload" in str(x.message)]
            assert len(reload_warnings) == 0, (
                f"Expected no hot reload warning, got {reload_warnings}"
            )


# ---------------------------------------------------------------------------
# Valid stacks produce no anti-pattern diagnostics
# ---------------------------------------------------------------------------

class TestValidStacks:
    def test_single_decorator_no_issue(self):
        s = stack(parallel)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            @s
            class Foo:
                pass
            antipattern = [
                x for x in w
                if any(
                    kw in str(x.message)
                    for kw in ("contradictory", "hot reload", "delta sync", "no data")
                )
            ]
            assert len(antipattern) == 0, (
                f"Expected no anti-pattern warnings, got {antipattern}"
            )

    def test_empty_stack_no_issue(self):
        s = stack()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            @s
            class Foo:
                pass
            assert len(w) == 0, f"Empty stack should produce no warnings, got {w}"


# ---------------------------------------------------------------------------
# Direct unit test of _validate_stack_combination
# ---------------------------------------------------------------------------

class TestValidateFunction:
    def test_accepts_tuple(self):
        # Should not raise
        _validate_stack_combination((parallel, build_only))

    def test_parallel_exclusive_via_function(self):
        with pytest.raises(ValueError, match="contradictory"):
            _validate_stack_combination((parallel, exclusive))

    def test_tag_serializable_via_function(self):
        with pytest.raises(ValueError, match="tags have no data to serialize"):
            _validate_stack_combination((tag, serializable))

    def test_empty_tuple_accepted(self):
        # Should not raise
        _validate_stack_combination(())


# ---------------------------------------------------------------------------
# Stack.__call__ rejects None-returning decorators
# ---------------------------------------------------------------------------

class TestNoneReturningDecorator:
    def test_none_returning_decorator_raises_type_error(self):
        def bad_dec(cls):
            return None
        bad_dec.__name__ = "bad_dec"

        s = stack(bad_dec)
        with pytest.raises(TypeError, match="returned None"):
            @s
            class Victim:
                pass

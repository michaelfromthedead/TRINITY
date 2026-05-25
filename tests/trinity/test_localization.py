"""
Tests for localization decorators (localization.py).

Tests the 4 localization decorators built on Ops:
    @localized, @plural, @rtl_aware, @text_overflow

Each test verifies:
1. Steps are applied (decompose works, _applied_steps populated)
2. Domain attributes are set correctly
3. Validation rejects invalid params
4. Introspection works
"""

import pytest

from trinity.decorators.localization import (
    VALID_OVERFLOW_STRATEGIES,
    localized,
    plural,
    rtl_aware,
    text_overflow,
)
from trinity.decorators.ops import Op, decompose
from trinity.decorators.registry import Tier, registry


# =============================================================================
# @localized
# =============================================================================


class TestLocalized:
    def test_default_params_class(self):
        @localized()
        class Foo:
            pass

        assert Foo._localized is True
        assert Foo._localized_key is None
        assert Foo._localized_context == ""
        assert Foo._localized_max_length is None

    def test_custom_params(self):
        @localized(key="menu.start", context="Main menu button", max_length=20)
        class Btn:
            pass

        assert Btn._localized_key == "menu.start"
        assert Btn._localized_context == "Main menu button"
        assert Btn._localized_max_length == 20

    def test_on_function(self):
        @localized(key="greeting")
        def greet():
            pass

        assert greet._localized is True
        assert greet._localized_key == "greeting"

    def test_no_parens(self):
        @localized
        class W:
            pass

        assert W._localized is True

    def test_invalid_max_length_zero(self):
        with pytest.raises(ValueError, match="max_length"):
            @localized(max_length=0)
            class Bad:
                pass

    def test_invalid_max_length_negative(self):
        with pytest.raises(ValueError, match="max_length"):
            @localized(max_length=-5)
            class Bad:
                pass

    def test_invalid_max_length_float(self):
        with pytest.raises(ValueError, match="max_length"):
            @localized(max_length=3.5)
            class Bad:
                pass

    def test_applied_decorators(self):
        @localized()
        class C:
            pass

        assert "localized" in C._applied_decorators

    def test_steps_recorded(self):
        @localized()
        class C:
            pass

        ops_used = {s.op for s in C._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used

    def test_decompose(self):
        steps = decompose(localized)
        assert len(steps) > 0

    def test_tags_stored(self):
        @localized(key="k1", context="ctx", max_length=10)
        class C:
            pass

        assert C._tags["localized"] is True
        assert C._tags["localized_key"] == "k1"
        assert C._tags["localized_context"] == "ctx"
        assert C._tags["localized_max_length"] == 10

    def test_key_none_by_default(self):
        @localized()
        class C:
            pass

        assert C._tags["localized_key"] is None


# =============================================================================
# @plural
# =============================================================================


class TestPlural:
    def test_basic(self):
        @plural(one="item", other="items")
        def count():
            pass

        assert count._plural is True
        assert count._plural_one == "item"
        assert count._plural_other == "items"
        assert count._plural_zero is None
        assert count._plural_few is None
        assert count._plural_many is None

    def test_all_forms(self):
        @plural(one="item", other="items", zero="no items", few="a few items", many="many items")
        def count():
            pass

        assert count._plural_zero == "no items"
        assert count._plural_few == "a few items"
        assert count._plural_many == "many items"

    def test_missing_one(self):
        with pytest.raises(ValueError, match="'one' is required"):
            @plural(one="", other="items")
            def bad():
                pass

    def test_missing_other(self):
        with pytest.raises(ValueError, match="'other' is required"):
            @plural(one="item", other="")
            def bad():
                pass

    def test_applied_decorators(self):
        @plural(one="a", other="b")
        def f():
            pass

        assert "plural" in f._applied_decorators

    def test_steps_recorded(self):
        @plural(one="a", other="b")
        def f():
            pass

        ops_used = {s.op for s in f._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used

    def test_decompose(self):
        steps = decompose(plural)
        assert len(steps) > 0

    def test_tags_stored(self):
        @plural(one="x", other="y", zero="z")
        def f():
            pass

        assert f._tags["plural"] is True
        assert f._tags["plural_one"] == "x"
        assert f._tags["plural_other"] == "y"
        assert f._tags["plural_zero"] == "z"


# =============================================================================
# @rtl_aware
# =============================================================================


class TestRtlAware:
    def test_basic(self):
        @rtl_aware
        class Widget:
            pass

        assert Widget._rtl_aware is True

    def test_with_parens(self):
        @rtl_aware()
        class Widget:
            pass

        assert Widget._rtl_aware is True

    def test_applied_decorators(self):
        @rtl_aware
        class C:
            pass

        assert "rtl_aware" in C._applied_decorators

    def test_steps_recorded(self):
        @rtl_aware
        class C:
            pass

        ops_used = {s.op for s in C._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used

    def test_decompose(self):
        steps = decompose(rtl_aware)
        assert len(steps) > 0

    def test_tags_stored(self):
        @rtl_aware
        class C:
            pass

        assert C._tags["rtl_aware"] is True


# =============================================================================
# @text_overflow
# =============================================================================


class TestTextOverflow:
    def test_default_strategy(self):
        @text_overflow()
        class W:
            pass

        assert W._text_overflow is True
        assert W._text_overflow_strategy == "truncate"

    def test_custom_strategy_shrink(self):
        @text_overflow(strategy="shrink")
        class W:
            pass

        assert W._text_overflow_strategy == "shrink"

    def test_custom_strategy_scroll(self):
        @text_overflow(strategy="scroll")
        class W:
            pass

        assert W._text_overflow_strategy == "scroll"

    def test_custom_strategy_wrap(self):
        @text_overflow(strategy="wrap")
        class W:
            pass

        assert W._text_overflow_strategy == "wrap"

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="invalid strategy"):
            @text_overflow(strategy="ellipsis")
            class Bad:
                pass

    def test_no_parens(self):
        @text_overflow
        class W:
            pass

        assert W._text_overflow is True
        assert W._text_overflow_strategy == "truncate"

    def test_applied_decorators(self):
        @text_overflow()
        class C:
            pass

        assert "text_overflow" in C._applied_decorators

    def test_steps_recorded(self):
        @text_overflow()
        class C:
            pass

        ops_used = {s.op for s in C._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used

    def test_decompose(self):
        steps = decompose(text_overflow)
        assert len(steps) > 0

    def test_tags_stored(self):
        @text_overflow(strategy="scroll")
        class C:
            pass

        assert C._tags["text_overflow"] is True
        assert C._tags["text_overflow_strategy"] == "scroll"


# =============================================================================
# REGISTRY
# =============================================================================


class TestLocalizationRegistry:
    def test_localized_registered(self):
        assert "localized" in registry._decorators
        spec = registry._decorators["localized"]
        assert spec.tier == Tier.LOCALIZATION
        assert spec.target_types == ("function", "class")

    def test_plural_registered(self):
        assert "plural" in registry._decorators
        spec = registry._decorators["plural"]
        assert spec.tier == Tier.LOCALIZATION
        assert spec.target_types == ("function",)

    def test_rtl_aware_registered(self):
        assert "rtl_aware" in registry._decorators
        spec = registry._decorators["rtl_aware"]
        assert spec.tier == Tier.LOCALIZATION
        assert spec.target_types == ("class",)

    def test_text_overflow_registered(self):
        assert "text_overflow" in registry._decorators
        spec = registry._decorators["text_overflow"]
        assert spec.tier == Tier.LOCALIZATION
        assert spec.target_types == ("class",)

    def test_all_in_tier(self):
        names = {s.name for s in registry._by_tier[Tier.LOCALIZATION]}
        assert {"localized", "plural", "rtl_aware", "text_overflow"} <= names


# =============================================================================
# STACKING
# =============================================================================


class TestLocalizationStacking:
    def test_localized_and_rtl(self):
        @localized(key="title")
        @rtl_aware
        class W:
            pass

        assert W._localized is True
        assert W._rtl_aware is True

    def test_localized_and_text_overflow(self):
        @localized(key="desc")
        @text_overflow(strategy="wrap")
        class W:
            pass

        assert W._localized is True
        assert W._text_overflow is True
        assert W._text_overflow_strategy == "wrap"

    def test_triple_stack(self):
        @localized(key="lbl")
        @text_overflow(strategy="shrink")
        @rtl_aware
        class W:
            pass

        assert W._localized is True
        assert W._text_overflow is True
        assert W._rtl_aware is True


# =============================================================================
# VALID VALUES CONSTANT
# =============================================================================


class TestValidOverflowStrategies:
    def test_contains_expected(self):
        assert "truncate" in VALID_OVERFLOW_STRATEGIES
        assert "shrink" in VALID_OVERFLOW_STRATEGIES
        assert "scroll" in VALID_OVERFLOW_STRATEGIES
        assert "wrap" in VALID_OVERFLOW_STRATEGIES

    def test_is_frozenset(self):
        assert isinstance(VALID_OVERFLOW_STRATEGIES, frozenset)

    def test_exactly_four(self):
        assert len(VALID_OVERFLOW_STRATEGIES) == 4

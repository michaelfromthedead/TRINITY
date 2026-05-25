"""
Tests for accessibility decorators (accessibility.py).

Tests the @accessible decorator built on Ops.

Each test verifies:
1. Steps are applied (decompose works, _applied_steps populated)
2. Domain attributes are set correctly
3. Validation rejects invalid params
4. Introspection works
"""

import pytest

from trinity.decorators.accessibility import (
    VALID_ROLES,
    accessible,
)
from trinity.decorators.ops import Op, decompose
from trinity.decorators.registry import Tier, registry


# =============================================================================
# @accessible
# =============================================================================


class TestAccessible:
    def test_default_params(self):
        @accessible()
        class Btn:
            pass

        assert Btn._accessible is True
        assert Btn._accessible_screen_reader is None
        assert Btn._accessible_role is None

    def test_screen_reader_only(self):
        @accessible(screen_reader="Click to start game")
        class Btn:
            pass

        assert Btn._accessible_screen_reader == "Click to start game"
        assert Btn._accessible_role is None

    def test_role_only(self):
        @accessible(role="button")
        class Btn:
            pass

        assert Btn._accessible_role == "button"
        assert Btn._accessible_screen_reader is None

    def test_both_params(self):
        @accessible(screen_reader="Volume control", role="slider")
        class Vol:
            pass

        assert Vol._accessible_screen_reader == "Volume control"
        assert Vol._accessible_role == "slider"

    def test_all_valid_roles(self):
        for role in VALID_ROLES:
            @accessible(role=role)
            class C:
                pass

            assert C._accessible_role == role

    def test_invalid_role(self):
        with pytest.raises(ValueError, match="invalid role"):
            @accessible(role="checkbox")
            class Bad:
                pass

    def test_invalid_role_empty(self):
        with pytest.raises(ValueError, match="invalid role"):
            @accessible(role="")
            class Bad:
                pass

    def test_no_parens(self):
        @accessible
        class W:
            pass

        assert W._accessible is True

    def test_applied_decorators(self):
        @accessible()
        class C:
            pass

        assert "accessible" in C._applied_decorators

    def test_steps_recorded(self):
        @accessible()
        class C:
            pass

        ops_used = {s.op for s in C._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used

    def test_decompose(self):
        steps = decompose(accessible)
        assert len(steps) > 0

    def test_tags_stored(self):
        @accessible(screen_reader="hello", role="text")
        class C:
            pass

        assert C._tags["accessible"] is True
        assert C._tags["accessible_screen_reader"] == "hello"
        assert C._tags["accessible_role"] == "text"

    def test_tags_none_defaults(self):
        @accessible()
        class C:
            pass

        assert C._tags["accessible_screen_reader"] is None
        assert C._tags["accessible_role"] is None

    def test_role_button(self):
        @accessible(role="button")
        class C:
            pass

        assert C._accessible_role == "button"

    def test_role_slider(self):
        @accessible(role="slider")
        class C:
            pass

        assert C._accessible_role == "slider"

    def test_role_text(self):
        @accessible(role="text")
        class C:
            pass

        assert C._accessible_role == "text"

    def test_role_image(self):
        @accessible(role="image")
        class C:
            pass

        assert C._accessible_role == "image"

    def test_role_list(self):
        @accessible(role="list")
        class C:
            pass

        assert C._accessible_role == "list"

    def test_role_listitem(self):
        @accessible(role="listitem")
        class C:
            pass

        assert C._accessible_role == "listitem"


# =============================================================================
# REGISTRY
# =============================================================================


class TestAccessibilityRegistry:
    def test_accessible_registered(self):
        assert "accessible" in registry._decorators
        spec = registry._decorators["accessible"]
        assert spec.tier == Tier.ACCESSIBILITY
        assert spec.target_types == ("class",)

    def test_in_tier(self):
        names = {s.name for s in registry._by_tier[Tier.ACCESSIBILITY]}
        assert "accessible" in names

    def test_not_unique(self):
        spec = registry._decorators["accessible"]
        assert spec.unique is False

    def test_not_foundation(self):
        spec = registry._decorators["accessible"]
        assert spec.foundation is False


# =============================================================================
# STACKING
# =============================================================================


class TestAccessibilityStacking:
    def test_accessible_with_localized(self):
        from trinity.decorators.localization import localized

        @accessible(role="button", screen_reader="Start")
        @localized(key="btn.start")
        class Btn:
            pass

        assert Btn._accessible is True
        assert Btn._localized is True

    def test_accessible_with_rtl(self):
        from trinity.decorators.localization import rtl_aware

        @accessible(role="text")
        @rtl_aware
        class Lbl:
            pass

        assert Lbl._accessible is True
        assert Lbl._rtl_aware is True

    def test_accessible_with_text_overflow(self):
        from trinity.decorators.localization import text_overflow

        @accessible(screen_reader="Description text")
        @text_overflow(strategy="scroll")
        class Desc:
            pass

        assert Desc._accessible is True
        assert Desc._text_overflow is True

    def test_triple_stack_cross_tier(self):
        from trinity.decorators.localization import localized, rtl_aware

        @accessible(role="button")
        @localized(key="k")
        @rtl_aware
        class W:
            pass

        assert W._accessible is True
        assert W._localized is True
        assert W._rtl_aware is True


# =============================================================================
# VALID VALUES CONSTANT
# =============================================================================


class TestValidRoles:
    def test_contains_expected(self):
        expected = {"button", "slider", "text", "image", "list", "listitem"}
        assert expected == VALID_ROLES

    def test_is_frozenset(self):
        assert isinstance(VALID_ROLES, frozenset)

    def test_exactly_six(self):
        assert len(VALID_ROLES) == 6

"""Tests for Tier 30 — MODDING decorators."""

import pytest

from trinity.decorators.modding import (
    VALID_EXTEND_MODES,
    conflicts,
    load_order,
    mod,
    mod_extends,
    moddable,
    patch,
    provides,
    replaces,
    requires,
)
from trinity.decorators.registry import Tier, registry


# =========================================================================
# @mod
# =========================================================================


class TestMod:
    def test_basic(self):
        @mod(name="mymod", version=(1, 0, 0), author="dev")
        class M:
            pass

        assert M._mod is True
        assert M._mod_name == "mymod"
        assert M._mod_version == (1, 0, 0)
        assert M._mod_author == "dev"
        assert M._mod_description == ""

    def test_with_description(self):
        @mod(name="m", version=(0, 1, 0), author="a", description="A mod")
        class M:
            pass

        assert M._mod_description == "A mod"

    def test_applied_decorators(self):
        @mod(name="m", version=(1, 0, 0), author="a")
        class M:
            pass

        assert "mod" in M._applied_decorators

    def test_tags(self):
        @mod(name="m", version=(1, 2, 3), author="a")
        class M:
            pass

        assert M._tags["mod"] is True
        assert M._tags["mod_name"] == "m"
        assert M._tags["mod_version"] == (1, 2, 3)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            @mod(name="", version=(1, 0, 0), author="a")
            class M:
                pass

    def test_empty_author_raises(self):
        with pytest.raises(ValueError, match="author"):
            @mod(name="m", version=(1, 0, 0), author="")
            class M:
                pass

    def test_version_not_tuple_raises(self):
        with pytest.raises(ValueError, match="tuple"):
            @mod(name="m", version=[1, 0, 0], author="a")
            class M:
                pass

    def test_version_wrong_length_raises(self):
        with pytest.raises(ValueError, match="exactly 3"):
            @mod(name="m", version=(1, 0), author="a")
            class M:
                pass

    def test_version_four_elements_raises(self):
        with pytest.raises(ValueError, match="exactly 3"):
            @mod(name="m", version=(1, 0, 0, 0), author="a")
            class M:
                pass

    def test_version_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            @mod(name="m", version=(1, -1, 0), author="a")
            class M:
                pass

    def test_version_non_int_raises(self):
        with pytest.raises(ValueError, match="non-negative integer"):
            @mod(name="m", version=(1, "0", 0), author="a")
            class M:
                pass

    def test_registry(self):
        spec = registry.get("mod")
        assert spec is not None
        assert spec.tier == Tier.MODDING
        assert spec.target_types == ("class",)


# =========================================================================
# @requires
# =========================================================================


class TestRequires:
    def test_basic(self):
        @requires(mod="base_mod")
        class M:
            pass

        assert len(M._requires) == 1
        assert M._requires[0] == {"mod": "base_mod", "version": "*", "optional": False}

    def test_with_version(self):
        @requires(mod="x", version=">=2.0")
        class M:
            pass

        assert M._requires[0]["version"] == ">=2.0"

    def test_optional(self):
        @requires(mod="x", optional=True)
        class M:
            pass

        assert M._requires[0]["optional"] is True

    def test_accumulation(self):
        @requires(mod="c")
        @requires(mod="b")
        @requires(mod="a")
        class M:
            pass

        assert len(M._requires) == 3
        assert M._requires[0]["mod"] == "a"
        assert M._requires[1]["mod"] == "b"
        assert M._requires[2]["mod"] == "c"

    def test_empty_mod_raises(self):
        with pytest.raises(ValueError, match="mod"):
            @requires(mod="")
            class M:
                pass

    def test_registry(self):
        spec = registry.get("requires")
        assert spec is not None
        assert spec.tier == Tier.MODDING


# =========================================================================
# @conflicts
# =========================================================================


class TestConflicts:
    def test_basic(self):
        @conflicts(mod="other", reason="incompatible API")
        class M:
            pass

        assert len(M._conflicts) == 1
        assert M._conflicts[0] == {"mod": "other", "reason": "incompatible API"}

    def test_accumulation(self):
        @conflicts(mod="b", reason="r2")
        @conflicts(mod="a", reason="r1")
        class M:
            pass

        assert len(M._conflicts) == 2
        assert M._conflicts[0]["mod"] == "a"
        assert M._conflicts[1]["mod"] == "b"

    def test_empty_mod_raises(self):
        with pytest.raises(ValueError, match="mod"):
            @conflicts(mod="", reason="r")
            class M:
                pass

    def test_empty_reason_raises(self):
        with pytest.raises(ValueError, match="reason"):
            @conflicts(mod="x", reason="")
            class M:
                pass

    def test_registry(self):
        spec = registry.get("conflicts")
        assert spec is not None
        assert spec.tier == Tier.MODDING


# =========================================================================
# @provides
# =========================================================================


class TestProvides:
    def test_basic(self):
        @provides(feature="rendering")
        class M:
            pass

        assert M._provides == ["rendering"]

    def test_accumulation(self):
        @provides(feature="c")
        @provides(feature="b")
        @provides(feature="a")
        class M:
            pass

        assert M._provides == ["a", "b", "c"]

    def test_empty_feature_raises(self):
        with pytest.raises(ValueError, match="feature"):
            @provides(feature="")
            class M:
                pass

    def test_registry(self):
        spec = registry.get("provides")
        assert spec is not None
        assert spec.tier == Tier.MODDING


# =========================================================================
# @replaces
# =========================================================================


class TestReplaces:
    def test_basic(self):
        @replaces(mod="old_mod")
        class M:
            pass

        assert M._replaces is True
        assert M._replaces_mod == "old_mod"
        assert M._replaces_reason == ""

    def test_with_reason(self):
        @replaces(mod="old", reason="deprecated")
        class M:
            pass

        assert M._replaces_reason == "deprecated"

    def test_empty_mod_raises(self):
        with pytest.raises(ValueError, match="mod"):
            @replaces(mod="")
            class M:
                pass

    def test_registry(self):
        spec = registry.get("replaces")
        assert spec is not None
        assert spec.tier == Tier.MODDING


# =========================================================================
# @mod_extends
# =========================================================================


class TestModExtends:
    def test_basic(self):
        @mod_extends(target_name="weapons")
        class M:
            pass

        assert M._mod_extends is True
        assert M._mod_extends_target == "weapons"
        assert M._mod_extends_mode == "merge"

    def test_replace_mode(self):
        @mod_extends(target_name="items", mode="replace")
        class M:
            pass

        assert M._mod_extends_mode == "replace"

    def test_empty_target_raises(self):
        with pytest.raises(ValueError, match="target_name"):
            @mod_extends(target_name="")
            class M:
                pass

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            @mod_extends(target_name="x", mode="append")
            class M:
                pass

    def test_valid_modes(self):
        assert VALID_EXTEND_MODES == frozenset({"merge", "replace"})

    def test_registry(self):
        spec = registry.get("mod_extends")
        assert spec is not None
        assert spec.tier == Tier.MODDING


# =========================================================================
# @patch
# =========================================================================


class TestPatch:
    def test_basic(self):
        @patch(base_mod="core", target_mod="extras")
        class M:
            pass

        assert M._patch is True
        assert M._patch_base == "core"
        assert M._patch_target == "extras"

    def test_empty_base_raises(self):
        with pytest.raises(ValueError, match="base_mod"):
            @patch(base_mod="", target_mod="x")
            class M:
                pass

    def test_empty_target_raises(self):
        with pytest.raises(ValueError, match="target_mod"):
            @patch(base_mod="x", target_mod="")
            class M:
                pass

    def test_registry(self):
        spec = registry.get("patch")
        assert spec is not None
        assert spec.tier == Tier.MODDING


# =========================================================================
# @load_order
# =========================================================================


class TestLoadOrder:
    def test_defaults(self):
        @load_order()
        class M:
            pass

        assert M._load_order is True
        assert M._load_order_after == []
        assert M._load_order_before == []

    def test_after_mods(self):
        @load_order(after_mods=["core", "base"])
        class M:
            pass

        assert M._load_order_after == ["core", "base"]
        assert M._load_order_before == []

    def test_before_mods(self):
        @load_order(before_mods=["ui"])
        class M:
            pass

        assert M._load_order_before == ["ui"]

    def test_both(self):
        @load_order(after_mods=["a"], before_mods=["z"])
        class M:
            pass

        assert M._load_order_after == ["a"]
        assert M._load_order_before == ["z"]

    def test_no_args(self):
        @load_order
        class M:
            pass

        assert M._load_order is True

    def test_registry(self):
        spec = registry.get("load_order")
        assert spec is not None
        assert spec.tier == Tier.MODDING


# =========================================================================
# @moddable
# =========================================================================


class TestModdable:
    def test_basic(self):
        @moddable(namespace="weapons")
        class M:
            pass

        assert M._moddable is True
        assert M._moddable_namespace == "weapons"
        assert M._moddable_version == 1

    def test_custom_version(self):
        @moddable(namespace="items", version=3)
        class M:
            pass

        assert M._moddable_version == 3

    def test_empty_namespace_raises(self):
        with pytest.raises(ValueError, match="namespace"):
            @moddable(namespace="")
            class M:
                pass

    def test_zero_version_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            @moddable(namespace="x", version=0)
            class M:
                pass

    def test_negative_version_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            @moddable(namespace="x", version=-1)
            class M:
                pass

    def test_registry(self):
        spec = registry.get("moddable")
        assert spec is not None
        assert spec.tier == Tier.MODDING


# =========================================================================
# Combination / stacking tests
# =========================================================================


class TestModdingCombinations:
    def test_full_mod_declaration(self):
        @moddable(namespace="game")
        @provides(feature="weapons")
        @provides(feature="armor")
        @requires(mod="core", version=">=1.0")
        @conflicts(mod="old_combat", reason="replaced")
        @mod(name="combat", version=(2, 0, 0), author="dev")
        class CombatMod:
            pass

        assert CombatMod._mod is True
        assert CombatMod._mod_name == "combat"
        assert len(CombatMod._requires) == 1
        assert len(CombatMod._conflicts) == 1
        assert len(CombatMod._provides) == 2
        assert CombatMod._moddable is True

    def test_patch_with_load_order(self):
        @load_order(after_mods=["core", "combat"])
        @patch(base_mod="core", target_mod="combat")
        class CompatPatch:
            pass

        assert CompatPatch._patch is True
        assert CompatPatch._load_order is True
        assert CompatPatch._load_order_after == ["core", "combat"]

    def test_replaces_with_requires(self):
        @requires(mod="base")
        @replaces(mod="old_mod", reason="v2")
        class V2Mod:
            pass

        assert V2Mod._replaces is True
        assert V2Mod._replaces_mod == "old_mod"
        assert len(V2Mod._requires) == 1

    def test_mod_extends_with_mod(self):
        @mod_extends(target_name="items", mode="merge")
        @mod(name="extra_items", version=(1, 0, 0), author="a")
        class ExtraItems:
            pass

        assert ExtraItems._mod is True
        assert ExtraItems._mod_extends is True
        assert ExtraItems._mod_extends_target == "items"

    def test_registries_attribute(self):
        @mod(name="t", version=(1, 0, 0), author="a")
        class M:
            pass

        assert "modding" in M._registries


# =========================================================================
# Registry tier check
# =========================================================================


class TestModdingRegistry:
    def test_all_modding_decorators_registered(self):
        names = {"mod", "requires", "conflicts", "provides", "replaces",
                 "mod_extends", "patch", "load_order", "moddable"}
        for name in names:
            spec = registry.get(name)
            assert spec is not None, f"{name} not registered"
            assert spec.tier == Tier.MODDING

    def test_accumulating_decorators_not_unique(self):
        for name in ("requires", "conflicts", "provides"):
            spec = registry.get(name)
            assert spec is not None
            assert spec.unique is False, f"{name} should not be unique"

    def test_non_accumulating_decorators_unique(self):
        for name in ("mod", "replaces", "mod_extends", "patch", "load_order", "moddable"):
            spec = registry.get(name)
            assert spec is not None
            assert spec.unique is True, f"{name} should be unique"

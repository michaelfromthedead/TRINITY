"""
Tests for prefab decorators (prefabs.py).

Tests the 2 prefab decorators built on Ops:
    @prefab, @extends

Each test verifies:
1. Steps are applied (decompose works, _applied_steps populated)
2. Domain attributes are set correctly
3. Validation rejects invalid params
4. Introspection works
"""

import pytest

from trinity.decorators.ops import Op, Step, decompose
from trinity.decorators.prefabs import extends, prefab
from trinity.decorators.registry import Tier, registry


# =============================================================================
# @prefab
# =============================================================================


class TestPrefab:
    def test_basic_application(self):
        @prefab(name="enemy")
        class Enemy:
            pass

        assert Enemy._prefab is True
        assert Enemy._prefab_name == "enemy"

    def test_applied_decorators(self):
        @prefab(name="player")
        class Player:
            pass

        assert "prefab" in Player._applied_decorators

    def test_steps_recorded(self):
        @prefab(name="npc")
        class NPC:
            pass

        assert len(NPC._applied_steps) >= 2
        ops = [s.op for s in NPC._applied_steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_tags_set(self):
        @prefab(name="item")
        class Item:
            pass

        assert Item._tags["prefab"] is True
        assert Item._tags["prefab_name"] == "item"

    def test_registered_in_prefabs_registry(self):
        @prefab(name="weapon")
        class Weapon:
            pass

        assert "prefabs" in Weapon._registries

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="'name' parameter is required"):

            @prefab(name="")
            class Bad:
                pass

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="'name' parameter is required"):

            @prefab()
            class Bad:
                pass

    def test_decompose(self):
        steps = decompose(prefab)
        assert isinstance(steps, list)
        assert all(isinstance(s, Step) for s in steps)

    def test_decorator_metadata(self):
        assert prefab.__name__ == "prefab"
        assert prefab._is_decorator is True
        assert prefab._decorator_name == "prefab"

    def test_registry_entry(self):
        spec = registry.get("prefab")
        assert spec is not None
        assert spec.tier == Tier.PREFABS
        assert "class" in spec.target_types

    def test_multiple_prefabs(self):
        @prefab(name="a")
        class A:
            pass

        @prefab(name="b")
        class B:
            pass

        assert A._prefab_name == "a"
        assert B._prefab_name == "b"

    def test_prefab_on_class_with_methods(self):
        @prefab(name="complex")
        class Complex:
            def update(self):
                pass

        assert Complex._prefab is True
        assert hasattr(Complex, "update")

    def test_prefab_preserves_class(self):
        @prefab(name="preserved")
        class Preserved:
            x = 42

        assert Preserved.x == 42


# =============================================================================
# @extends
# =============================================================================


class TestExtends:
    def test_basic_application(self):
        @extends(parent="enemy")
        class FastEnemy:
            pass

        assert FastEnemy._extends is True
        assert FastEnemy._extends_parent == "enemy"

    def test_applied_decorators(self):
        @extends(parent="player")
        class SuperPlayer:
            pass

        assert "extends" in SuperPlayer._applied_decorators

    def test_steps_recorded(self):
        @extends(parent="npc")
        class QuestNPC:
            pass

        assert len(QuestNPC._applied_steps) >= 2
        ops = [s.op for s in QuestNPC._applied_steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_tags_set(self):
        @extends(parent="item")
        class MagicItem:
            pass

        assert MagicItem._tags["extends"] is True
        assert MagicItem._tags["extends_parent"] == "item"

    def test_registered_in_prefabs_registry(self):
        @extends(parent="weapon")
        class Sword:
            pass

        assert "prefabs" in Sword._registries

    def test_empty_parent_raises(self):
        with pytest.raises(ValueError, match="'parent' parameter is required"):

            @extends(parent="")
            class Bad:
                pass

    def test_missing_parent_raises(self):
        with pytest.raises(ValueError, match="'parent' parameter is required"):

            @extends()
            class Bad:
                pass

    def test_decompose(self):
        steps = decompose(extends)
        assert isinstance(steps, list)
        assert all(isinstance(s, Step) for s in steps)

    def test_decorator_metadata(self):
        assert extends.__name__ == "extends"
        assert extends._is_decorator is True
        assert extends._decorator_name == "extends"

    def test_registry_entry(self):
        spec = registry.get("extends")
        assert spec is not None
        assert spec.tier == Tier.PREFABS
        assert "class" in spec.target_types

    def test_extends_preserves_class(self):
        @extends(parent="base")
        class Child:
            y = 99

        assert Child.y == 99


# =============================================================================
# @prefab + @extends combined
# =============================================================================


class TestPrefabExtendsCombined:
    def test_prefab_then_extends(self):
        @extends(parent="base")
        @prefab(name="base_child")
        class BaseChild:
            pass

        assert BaseChild._prefab is True
        assert BaseChild._extends is True
        assert "prefab" in BaseChild._applied_decorators
        assert "extends" in BaseChild._applied_decorators

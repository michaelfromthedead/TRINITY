"""
Tests for composition decorators (composition.py).

Tests the 2 composition decorators built on Ops:
    @composite, @alias

Each test verifies:
1. Steps are applied (decompose works, _applied_steps populated)
2. Domain attributes are set correctly
3. Validation rejects invalid params
4. Introspection works
"""

import pytest

from trinity.decorators.composition import alias, composite
from trinity.decorators.ops import Op, Step, decompose
from trinity.decorators.registry import Tier, registry


# =============================================================================
# Helper decorators for composite tests
# =============================================================================


def deco_a(cls):
    cls._deco_a = True
    return cls


def deco_b(cls):
    cls._deco_b = True
    return cls


def deco_c(cls):
    cls._deco_c = True
    return cls


# =============================================================================
# @composite
# =============================================================================


class TestComposite:
    def test_basic_application(self):
        @composite(decorators=[deco_a, deco_b])
        class Foo:
            pass

        assert Foo._composite is True
        assert len(Foo._composite_decorators) == 2

    def test_applied_decorators(self):
        @composite(decorators=[deco_a])
        class Bar:
            pass

        assert "composite" in Bar._applied_decorators

    def test_steps_recorded(self):
        @composite(decorators=[deco_a])
        class Baz:
            pass

        assert len(Baz._applied_steps) >= 2
        ops = [s.op for s in Baz._applied_steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_tags_set(self):
        @composite(decorators=[deco_a, deco_b])
        class Tagged:
            pass

        assert Tagged._tags["composite"] is True
        assert len(Tagged._tags["composite_decorators"]) == 2

    def test_registered_in_composition_registry(self):
        @composite(decorators=[deco_a])
        class Registered:
            pass

        assert "composition" in Registered._registries

    def test_empty_decorators_raises(self):
        with pytest.raises(ValueError, match="'decorators' parameter is required"):

            @composite(decorators=[])
            class Bad:
                pass

    def test_missing_decorators_raises(self):
        with pytest.raises(ValueError, match="'decorators' parameter is required"):

            @composite()
            class Bad:
                pass

    def test_non_callable_raises(self):
        with pytest.raises(ValueError, match="not callable"):

            @composite(decorators=[deco_a, "not_callable"])
            class Bad:
                pass

    def test_non_callable_at_index(self):
        with pytest.raises(ValueError, match="index 0"):

            @composite(decorators=[42])
            class Bad:
                pass

    def test_three_decorators(self):
        @composite(decorators=[deco_a, deco_b, deco_c])
        class Multi:
            pass

        assert len(Multi._composite_decorators) == 3
        assert Multi._composite_decorators[0] is deco_a
        assert Multi._composite_decorators[1] is deco_b
        assert Multi._composite_decorators[2] is deco_c

    def test_decompose(self):
        steps = decompose(composite)
        assert isinstance(steps, list)
        assert all(isinstance(s, Step) for s in steps)

    def test_decorator_metadata(self):
        assert composite.__name__ == "composite"
        assert composite._is_decorator is True
        assert composite._decorator_name == "composite"

    def test_registry_entry(self):
        spec = registry.get("composite")
        assert spec is not None
        assert spec.tier == Tier.COMPOSITION
        assert "class" in spec.target_types
        assert "function" in spec.target_types

    def test_on_function(self):
        @composite(decorators=[lambda f: f])
        def my_func():
            pass

        assert my_func._composite is True

    def test_preserves_class(self):
        @composite(decorators=[deco_a])
        class Kept:
            val = 10

        assert Kept.val == 10

    def test_composite_decorators_are_stored(self):
        @composite(decorators=[deco_a, deco_b])
        class Stored:
            pass

        assert deco_a in Stored._composite_decorators
        assert deco_b in Stored._composite_decorators

    def test_tuple_decorators(self):
        @composite(decorators=(deco_a, deco_b))
        class TupleInput:
            pass

        assert TupleInput._composite is True
        assert len(TupleInput._composite_decorators) == 2


# =============================================================================
# @alias
# =============================================================================


class TestAlias:
    def test_basic_application(self):
        @alias(name="my_alias")
        class Original:
            pass

        assert Original._alias is True
        assert Original._alias_name == "my_alias"

    def test_applied_decorators(self):
        @alias(name="shortcut")
        class Long:
            pass

        assert "alias" in Long._applied_decorators

    def test_steps_recorded(self):
        @alias(name="test_alias")
        class AliasTarget:
            pass

        assert len(AliasTarget._applied_steps) >= 2
        ops = [s.op for s in AliasTarget._applied_steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_tags_set(self):
        @alias(name="tagged_alias")
        class TaggedAlias:
            pass

        assert TaggedAlias._tags["alias"] is True
        assert TaggedAlias._tags["alias_name"] == "tagged_alias"

    def test_registered_in_composition_registry(self):
        @alias(name="reg_alias")
        class RegAlias:
            pass

        assert "composition" in RegAlias._registries

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="'name' parameter is required"):

            @alias(name="")
            class Bad:
                pass

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="'name' parameter is required"):

            @alias()
            class Bad:
                pass

    def test_decompose(self):
        steps = decompose(alias)
        assert isinstance(steps, list)
        assert all(isinstance(s, Step) for s in steps)

    def test_decorator_metadata(self):
        assert alias.__name__ == "alias"
        assert alias._is_decorator is True
        assert alias._decorator_name == "alias"

    def test_registry_entry(self):
        spec = registry.get("alias")
        assert spec is not None
        assert spec.tier == Tier.COMPOSITION
        assert "class" in spec.target_types
        assert "function" in spec.target_types

    def test_on_function(self):
        @alias(name="fn_alias")
        def my_func():
            pass

        assert my_func._alias is True
        assert my_func._alias_name == "fn_alias"

    def test_preserves_class(self):
        @alias(name="kept")
        class Kept:
            val = 55

        assert Kept.val == 55


# =============================================================================
# @composite + @alias combined
# =============================================================================


class TestCompositeAliasCombined:
    def test_alias_then_composite(self):
        @alias(name="combo")
        @composite(decorators=[deco_a])
        class Combo:
            pass

        assert Combo._composite is True
        assert Combo._alias is True
        assert Combo._alias_name == "combo"
        assert "composite" in Combo._applied_decorators
        assert "alias" in Combo._applied_decorators

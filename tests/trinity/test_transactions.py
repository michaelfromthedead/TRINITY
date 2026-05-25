"""
Tests for transaction decorators (transactions.py).

Tests the 2 transaction decorators built on Ops:
    @transactional, @undoable

Each test verifies:
1. Steps are applied (decompose works, _applied_steps populated)
2. Domain attributes are set correctly
3. Validation rejects invalid params
4. Introspection works
"""

import pytest

from trinity.decorators.ops import Op, decompose
from trinity.decorators.registry import Tier, registry
from trinity.decorators.transactions import (
    VALID_ISOLATION_LEVELS,
    transactional,
    undoable,
)


# =============================================================================
# @transactional
# =============================================================================


class TestTransactional:
    def test_default_params(self):
        @transactional()
        def save_game():
            pass

        assert save_game._transactional is True
        assert save_game._tx_isolation == "serializable"

    def test_custom_isolation(self):
        @transactional(isolation="read_committed")
        def update():
            pass

        assert update._tx_isolation == "read_committed"

    def test_read_uncommitted(self):
        @transactional(isolation="read_uncommitted")
        def dirty_read():
            pass

        assert dirty_read._tx_isolation == "read_uncommitted"

    def test_repeatable_read(self):
        @transactional(isolation="repeatable_read")
        def consistent_read():
            pass

        assert consistent_read._tx_isolation == "repeatable_read"

    def test_invalid_isolation(self):
        with pytest.raises(ValueError, match="invalid isolation level"):

            @transactional(isolation="snapshot")
            def bad():
                pass

    def test_applied_decorators(self):
        @transactional()
        def f():
            pass

        assert "transactional" in f._applied_decorators

    def test_steps_recorded(self):
        @transactional()
        def f():
            pass

        assert len(f._applied_steps) >= 2
        ops = [s.op for s in f._applied_steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_tags_set(self):
        @transactional(isolation="read_committed")
        def f():
            pass

        assert f._tags["transactional"] is True
        assert f._tags["tx_isolation"] == "read_committed"

    def test_registry_entry(self):
        @transactional()
        def f():
            pass

        assert "transactions" in f._registries

    def test_decompose(self):
        steps = decompose(transactional)
        assert len(steps) >= 2
        tag_steps = [s for s in steps if s.op == Op.TAG]
        assert len(tag_steps) >= 1

    def test_decorator_name(self):
        assert transactional.__name__ == "transactional"

    def test_is_decorator(self):
        assert transactional._is_decorator is True

    def test_no_parens(self):
        @transactional
        def f():
            pass

        assert f._transactional is True
        assert f._tx_isolation == "serializable"

    def test_all_valid_isolation_levels(self):
        for level in VALID_ISOLATION_LEVELS:

            @transactional(isolation=level)
            def f():
                pass

            assert f._tx_isolation == level

    def test_function_still_callable(self):
        @transactional()
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_registry_registered(self):
        spec = registry.get("transactional")
        assert spec is not None
        assert spec.tier == Tier.TRANSACTIONS
        assert spec.target_types == ("function",)


# =============================================================================
# @undoable
# =============================================================================


class TestUndoable:
    def test_default_params(self):
        @undoable()
        def move():
            pass

        assert move._undoable is True
        assert move._undo_group is None

    def test_custom_group(self):
        @undoable(group="transform")
        def rotate():
            pass

        assert rotate._undo_group == "transform"

    def test_applied_decorators(self):
        @undoable()
        def f():
            pass

        assert "undoable" in f._applied_decorators

    def test_steps_recorded(self):
        @undoable()
        def f():
            pass

        assert len(f._applied_steps) >= 2
        ops = [s.op for s in f._applied_steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_tags_set(self):
        @undoable(group="edit")
        def f():
            pass

        assert f._tags["undoable"] is True
        assert f._tags["undo_group"] == "edit"

    def test_tags_none_group(self):
        @undoable()
        def f():
            pass

        assert f._tags["undo_group"] is None

    def test_registry_entry(self):
        @undoable()
        def f():
            pass

        assert "transactions" in f._registries

    def test_decompose(self):
        steps = decompose(undoable)
        assert len(steps) >= 2

    def test_decorator_name(self):
        assert undoable.__name__ == "undoable"

    def test_is_decorator(self):
        assert undoable._is_decorator is True

    def test_no_parens(self):
        @undoable
        def f():
            pass

        assert f._undoable is True
        assert f._undo_group is None

    def test_function_still_callable(self):
        @undoable(group="test")
        def greet(name):
            return f"hello {name}"

        assert greet("world") == "hello world"

    def test_registry_registered(self):
        spec = registry.get("undoable")
        assert spec is not None
        assert spec.tier == Tier.TRANSACTIONS
        assert spec.target_types == ("function",)


# =============================================================================
# COMPOSITION
# =============================================================================


class TestTransactionComposition:
    def test_transactional_and_undoable(self):
        @transactional(isolation="serializable")
        @undoable(group="edit")
        def edit_entity():
            pass

        assert edit_entity._transactional is True
        assert edit_entity._undoable is True
        assert edit_entity._tx_isolation == "serializable"
        assert edit_entity._undo_group == "edit"

    def test_both_in_applied_decorators(self):
        @transactional()
        @undoable()
        def f():
            pass

        assert "transactional" in f._applied_decorators
        assert "undoable" in f._applied_decorators

    def test_multiple_undoable_allowed(self):
        # unique=False so multiple applications allowed
        @undoable(group="a")
        @undoable(group="b")
        def f():
            pass

        assert f._undoable is True

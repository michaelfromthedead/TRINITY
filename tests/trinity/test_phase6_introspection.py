"""Comprehensive tests for Phase 6 introspection functions."""
import pytest

from trinity.decorators.ops import (
    Op,
    Step,
    HookEvent,
    Rule,
    RULES,
    make_decorator,
    run_steps,
    _definitions,
)
from trinity.decorators.introspection import (
    primitives,
    composites,
    chain,
    find_decorators,
    compose,
    validate_combination,
    all_rules,
)


# ---------------------------------------------------------------------------
# Helpers: mock classes and descriptors
# ---------------------------------------------------------------------------

def _make_cls(**attrs):
    """Create a fresh class with arbitrary attributes."""
    ns = {}
    ns.update(attrs)
    return type("MockCls", (), ns)


class _MockDescriptor:
    """Descriptor with configurable descriptor_steps and optional descriptor_id."""

    def __init__(self, steps=None, desc_id=None):
        if steps is not None:
            self.descriptor_steps = steps
        if desc_id is not None:
            self.descriptor_id = desc_id


class _ChainItem:
    """Represents one item in a descriptor chain."""

    def __init__(self, desc_id):
        self.descriptor_id = desc_id


class _ChainDescriptor:
    """Descriptor with get_chain method returning descriptor-like objects."""

    def __init__(self, chain_ids, desc_id=None):
        self._chain_items = [_ChainItem(cid) for cid in chain_ids]
        if desc_id is not None:
            self.descriptor_id = desc_id

    def get_chain(self):
        return list(self._chain_items)


class _DescriptorNoChain:
    """Descriptor without get_chain but with descriptor_id."""

    def __init__(self, desc_id):
        self.descriptor_id = desc_id
        self.descriptor_steps = []


class _BareDescriptor:
    """Descriptor with nothing special."""
    pass


# =========================================================================
# primitives()
# =========================================================================

class TestPrimitives:
    def test_no_field_returns_decompose_result(self):
        steps = [Step(Op.TAG, {"key": "a"}), Step(Op.TRACK)]
        cls = _make_cls(_applied_steps=steps)
        result = primitives(cls)
        assert result == steps

    def test_no_field_empty_when_no_steps(self):
        cls = _make_cls()
        assert primitives(cls) == []

    def test_field_exists_returns_descriptor_steps(self):
        field_steps = [Step(Op.VALIDATE, {"constraint": "positive"})]
        desc = _MockDescriptor(steps=field_steps)
        cls = _make_cls(_field_descriptors={"hp": desc})
        assert primitives(cls, field="hp") == field_steps

    def test_field_not_in_descriptors(self):
        desc = _MockDescriptor(steps=[Step(Op.TAG)])
        cls = _make_cls(_field_descriptors={"hp": desc})
        assert primitives(cls, field="mana") == []

    def test_no_field_descriptors_attr(self):
        cls = _make_cls()
        assert primitives(cls, field="hp") == []

    def test_descriptor_without_descriptor_steps(self):
        cls = _make_cls(_field_descriptors={"x": _BareDescriptor()})
        assert primitives(cls, field="x") == []


# =========================================================================
# composites()
# =========================================================================

class TestComposites:
    def test_no_field_returns_applied_decorators_copy(self):
        cls = _make_cls(_applied_decorators=["pooled", "native"])
        result = composites(cls)
        assert result == ["pooled", "native"]
        # Must be a copy
        result.append("extra")
        assert composites(cls) == ["pooled", "native"]

    def test_no_field_no_applied_decorators(self):
        cls = _make_cls()
        assert composites(cls) == []

    def test_field_descriptor_has_get_chain(self):
        desc = _ChainDescriptor(["desc_a", "desc_b"])
        cls = _make_cls(_field_descriptors={"hp": desc})
        assert composites(cls, field="hp") == ["desc_a", "desc_b"]

    def test_field_descriptor_lacks_get_chain_uses_descriptor_id(self):
        desc = _DescriptorNoChain(desc_id="tracked_field")
        # Remove get_chain to be sure
        assert not hasattr(desc, "get_chain")
        cls = _make_cls(_field_descriptors={"hp": desc})
        assert composites(cls, field="hp") == ["tracked_field"]

    def test_field_descriptor_no_chain_no_id_uses_class_name(self):
        desc = _BareDescriptor()
        cls = _make_cls(_field_descriptors={"hp": desc})
        assert composites(cls, field="hp") == ["_BareDescriptor"]

    def test_field_not_in_descriptors(self):
        cls = _make_cls(_field_descriptors={"hp": _BareDescriptor()})
        assert composites(cls, field="mana") == []


# =========================================================================
# chain()
# =========================================================================

class TestChain:
    def test_field_not_in_descriptors(self):
        cls = _make_cls(_field_descriptors={})
        assert chain(cls, "hp") == "<hp: no descriptor>"

    def test_no_field_descriptors_attr(self):
        cls = _make_cls()
        assert chain(cls, "hp") == "<hp: no descriptor>"

    def test_field_exists_fallback_descriptor_id(self):
        desc = _DescriptorNoChain(desc_id="my_desc")
        cls = _make_cls(_field_descriptors={"hp": desc})
        result = chain(cls, "hp")
        # Should contain field name and descriptor_id in fallback
        assert result == "<hp: my_desc>"

    def test_field_exists_bare_descriptor_uses_class_name(self):
        desc = _BareDescriptor()
        cls = _make_cls(_field_descriptors={"x": desc})
        result = chain(cls, "x")
        assert "x" in result


# =========================================================================
# find_decorators()
# =========================================================================

class TestFindDecorators:
    def setup_method(self):
        # Save original definitions to restore after test
        self._orig = dict(_definitions)

    def teardown_method(self):
        _definitions.clear()
        _definitions.update(self._orig)

    def test_find_by_op(self):
        make_decorator(
            name="_test_find_tag",
            steps=[Step(Op.TAG, {"key": "pool", "value": True})],
        )
        result = find_decorators(Op.TAG)
        assert "_test_find_tag" in result

    def test_find_with_matching_filters(self):
        make_decorator(
            name="_test_find_filtered",
            steps=[Step(Op.TAG, {"key": "special", "value": 42})],
        )
        result = find_decorators(Op.TAG, key="special", value=42)
        assert "_test_find_filtered" in result

    def test_find_with_non_matching_filters(self):
        make_decorator(
            name="_test_find_nomatch",
            steps=[Step(Op.TAG, {"key": "other"})],
        )
        result = find_decorators(Op.TAG, key="nonexistent")
        assert "_test_find_nomatch" not in result

    def test_find_op_unused(self):
        # INTERCEPT not used by any registered decorator (clear first)
        _definitions.clear()
        assert find_decorators(Op.INTERCEPT) == []


# =========================================================================
# compose()
# =========================================================================

class TestCompose:
    def setup_method(self):
        self._orig = dict(_definitions)

    def teardown_method(self):
        _definitions.clear()
        _definitions.update(self._orig)

    def test_single_step_returns_callable_and_applies(self):
        dec = compose(Step(Op.TAG, {"key": "composed", "value": True}))
        assert callable(dec)
        result = dec(_make_cls())
        assert hasattr(result, "_tags")
        assert result._tags["composed"] is True

    def test_apply_composed_decorator(self):
        dec = compose(
            Step(Op.TAG, {"key": "x", "value": 1}),
            Step(Op.TRACK),
        )
        cls = _make_cls()
        result = dec(cls)
        assert hasattr(result, "_tags")
        assert result._tags["x"] == 1
        assert hasattr(result, "_tracked")

    def test_two_calls_produce_different_names(self):
        d1 = compose(Step(Op.TAG))
        d2 = compose(Step(Op.TAG))
        assert d1.__name__ != d2.__name__

    def test_applied_steps_recorded(self):
        dec = compose(Step(Op.TAG, {"key": "k"}))
        cls = _make_cls()
        result = dec(cls)
        assert any(s.op == Op.TAG for s in getattr(result, "_applied_steps", []))


# =========================================================================
# validate_combination()
# =========================================================================

class TestValidateCombination:
    def test_valid_steps(self):
        steps = [Step(Op.TAG, {"key": "a"}), Step(Op.DESCRIBE)]
        result = validate_combination(steps)
        assert result["valid"] is True

    def test_hook_on_change_without_track_invalid(self):
        steps = [Step(Op.HOOK, {"event": HookEvent.ON_CHANGE})]
        result = validate_combination(steps)
        assert result["valid"] is False
        assert "errors" in result
        assert any("TRACK" in e for e in result["errors"])

    def test_hook_on_change_with_track_valid(self):
        steps = [
            Step(Op.TRACK),
            Step(Op.HOOK, {"event": HookEvent.ON_CHANGE}),
        ]
        result = validate_combination(steps)
        assert result["valid"] is True

    def test_intercept_deny_conflicts_track(self):
        steps = [
            Step(Op.TRACK),
            Step(Op.INTERCEPT, {"set": "deny"}),
        ]
        result = validate_combination(steps)
        assert result["valid"] is False

    def test_intercept_deny_conflicts_validate(self):
        steps = [
            Step(Op.VALIDATE, {"constraint": "positive"}),
            Step(Op.INTERCEPT, {"set": "deny"}),
        ]
        result = validate_combination(steps)
        assert result["valid"] is False


# =========================================================================
# all_rules()
# =========================================================================

class TestAllRules:
    def test_returns_list_of_rules(self):
        rules = all_rules()
        assert isinstance(rules, list)
        assert len(rules) > 0
        assert all(isinstance(r, Rule) for r in rules)

    def test_returns_same_reference(self):
        # The current implementation returns RULES directly (not a copy).
        # Verify the content matches.
        rules = all_rules()
        assert rules == RULES

    def test_rules_have_names(self):
        for rule in all_rules():
            assert isinstance(rule.name, str)
            assert len(rule.name) > 0

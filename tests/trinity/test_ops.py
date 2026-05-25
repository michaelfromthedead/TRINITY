"""
Tests for the Ops system (ops.py).

Tests the 7 Op functions, Step dataclass, run_steps dispatch,
make_decorator factory, decompose/expand introspection, and
validate_steps composition rules.
"""

import pytest

from trinity.decorators.ops import (
    RULES,
    HookEvent,
    Op,
    Rule,
    Step,
    decompose,
    expand,
    make_decorator,
    run_describe,
    run_hook,
    run_intercept,
    run_register,
    run_step,
    run_steps,
    run_tag,
    run_track,
    run_validate,
    validate_steps,
)

# =============================================================================
# STEP DATACLASS
# =============================================================================


class TestStep:
    def test_create_minimal(self):
        s = Step(Op.TAG)
        assert s.op is Op.TAG
        assert s.args == {}

    def test_create_with_args(self):
        s = Step(Op.TAG, {"key": "pool", "value": True})
        assert s.args["key"] == "pool"
        assert s.args["value"] is True

    def test_frozen(self):
        s = Step(Op.TAG)
        with pytest.raises(AttributeError):
            s.op = Op.HOOK

    def test_repr_no_args(self):
        s = Step(Op.DESCRIBE)
        assert repr(s) == "DESCRIBE"

    def test_repr_with_args(self):
        s = Step(Op.TAG, {"key": "x"})
        r = repr(s)
        assert "TAG" in r
        assert "key" in r


# =============================================================================
# OP ENUM
# =============================================================================


class TestOp:
    def test_seven_ops(self):
        assert len(Op) == 7

    def test_values(self):
        expected = {
            "tag",
            "hook",
            "register",
            "describe",
            "track",
            "validate",
            "intercept",
        }
        assert {op.value for op in Op} == expected


# =============================================================================
# RUN FUNCTIONS
# =============================================================================


class _Target:
    """Bare target for testing run_* functions."""

    pass


class TestRunTag:
    def test_simple_boolean(self):
        t = _Target()
        run_tag(t, "pool")
        assert t._tags == {"pool": True}

    def test_value(self):
        t = _Target()
        run_tag(t, "backend", "vulkan")
        assert t._tags["backend"] == "vulkan"

    def test_multiple_tags(self):
        t = _Target()
        run_tag(t, "a", 1)
        run_tag(t, "b", 2)
        assert t._tags == {"a": 1, "b": 2}

    def test_overwrite(self):
        t = _Target()
        run_tag(t, "x", 1)
        run_tag(t, "x", 2)
        assert t._tags["x"] == 2


class TestRunHook:
    def test_string_event(self):
        t = _Target()
        cb = lambda: None
        run_hook(t, "on_create", cb)
        assert t._hooks["on_create"] == [cb]

    def test_enum_event(self):
        t = _Target()
        cb = lambda: None
        run_hook(t, HookEvent.ON_DESTROY, cb)
        assert t._hooks["on_destroy"] == [cb]

    def test_multiple_callbacks(self):
        t = _Target()
        cb1 = lambda: 1
        cb2 = lambda: 2
        run_hook(t, "evt", cb1)
        run_hook(t, "evt", cb2)
        assert len(t._hooks["evt"]) == 2

    def test_no_callback(self):
        t = _Target()
        run_hook(t, "evt")
        assert t._hooks["evt"] == []


class TestRunRegister:
    def test_register(self):
        t = _Target()
        run_register(t, "pool_manager")
        assert "pool_manager" in t._registries

    def test_no_duplicates(self):
        t = _Target()
        run_register(t, "r")
        run_register(t, "r")
        assert t._registries.count("r") == 1


class TestRunDescribe:
    def test_extracts_annotations(self):
        class Foo:
            x: int
            y: str
            _private: float

        run_describe(Foo)
        assert "x" in Foo._schema
        assert "y" in Foo._schema
        assert "_private" not in Foo._schema
        assert Foo._described is True

    def test_no_annotations(self):
        class Bar:
            pass

        run_describe(Bar)
        assert Bar._schema == {}
        assert Bar._described is True


class TestRunTrack:
    def test_track_all(self):
        t = _Target()
        run_track(t)
        assert t._tracked is True

    def test_track_field(self):
        t = _Target()
        run_track(t, "health")
        assert "health" in t._tracked_fields

    def test_track_multiple_fields(self):
        t = _Target()
        run_track(t, "x")
        run_track(t, "y")
        assert t._tracked_fields == {"x", "y"}


class TestRunValidate:
    def test_constraint(self):
        t = _Target()
        run_validate(t, "positive")
        assert t._constraints == [{"constraint": "positive"}]

    def test_field_constraint(self):
        t = _Target()
        run_validate(t, "range_0_100", "health")
        assert t._constraints == [{"constraint": "range_0_100", "field": "health"}]


class TestRunIntercept:
    def test_get(self):
        t = _Target()
        run_intercept(t, get="log_access")
        assert t._intercepts == [{"get": "log_access"}]

    def test_set(self):
        t = _Target()
        run_intercept(t, set="validate_on_set")
        assert t._intercepts == [{"set": "validate_on_set"}]

    def test_combined(self):
        t = _Target()
        run_intercept(t, get="g", set="s", delete="d")
        assert t._intercepts == [{"get": "g", "set": "s", "delete": "d"}]


# =============================================================================
# DISPATCH: run_step, run_steps
# =============================================================================


class TestRunStep:
    def test_dispatches_tag(self):
        t = _Target()
        run_step(t, Step(Op.TAG, {"key": "k", "value": "v"}))
        assert t._tags["k"] == "v"

    def test_dispatches_register(self):
        t = _Target()
        run_step(t, Step(Op.REGISTER, {"registry": "r"}))
        assert "r" in t._registries

    def test_unknown_op_no_crash(self):
        # Step with a valid op but empty args should still work
        t = _Target()
        result = run_step(t, Step(Op.DESCRIBE))
        assert result is t


class TestRunSteps:
    def test_multiple_steps(self):
        t = _Target()
        steps = [
            Step(Op.TAG, {"key": "a", "value": 1}),
            Step(Op.TAG, {"key": "b", "value": 2}),
            Step(Op.REGISTER, {"registry": "test"}),
        ]
        run_steps(t, steps)
        assert t._tags == {"a": 1, "b": 2}
        assert "test" in t._registries

    def test_records_applied_steps(self):
        t = _Target()
        steps = [Step(Op.TAG, {"key": "x"})]
        run_steps(t, steps)
        assert len(t._applied_steps) == 1
        assert t._applied_steps[0].op is Op.TAG

    def test_empty_steps(self):
        t = _Target()
        run_steps(t, [])
        assert t._applied_steps == []


# =============================================================================
# make_decorator FACTORY
# =============================================================================


class TestMakeDecorator:
    def test_static_steps(self):
        dec = make_decorator(
            name="test_dec",
            steps=[Step(Op.TAG, {"key": "tested", "value": True})],
        )

        @dec
        class Foo:
            pass

        assert Foo._tags["tested"] is True
        assert "test_dec" in Foo._applied_decorators

    def test_parameterized_steps(self):
        dec = make_decorator(
            name="level",
            steps=lambda params: [
                Step(Op.TAG, {"key": "level", "value": params.get("n", 0)})
            ],
        )

        @dec(n=5)
        class Bar:
            pass

        assert Bar._tags["level"] == 5

    def test_no_parens_usage(self):
        dec = make_decorator(
            name="marker",
            steps=[Step(Op.TAG, {"key": "marked"})],
        )

        @dec
        class Baz:
            pass

        assert Baz._tags["marked"] is True

    def test_with_parens_no_args(self):
        dec = make_decorator(
            name="marker2",
            steps=[Step(Op.TAG, {"key": "m2"})],
        )

        @dec()
        class Qux:
            pass

        assert Qux._tags["m2"] is True

    def test_validate_called(self):
        def bad_validate(**params):
            raise ValueError("nope")

        dec = make_decorator(
            name="strict",
            steps=[Step(Op.TAG, {"key": "x"})],
            validate=bad_validate,
        )

        with pytest.raises(ValueError, match="nope"):

            @dec()
            class Fail:
                pass

    def test_after_steps_called(self):
        called = []

        def after(target, params):
            called.append(target.__name__)
            target._custom = True

        dec = make_decorator(
            name="custom",
            steps=[Step(Op.TAG, {"key": "c"})],
            after_steps=after,
        )

        @dec
        class Thing:
            pass

        assert Thing._custom is True
        assert called == ["Thing"]

    def test_after_steps_can_replace_target(self):
        """after_steps returning a non-None value replaces the target."""

        class Replacement:
            pass

        def after(target, params):
            return Replacement

        dec = make_decorator(
            name="replacer",
            steps=[Step(Op.TAG, {"key": "r"})],
            after_steps=after,
        )

        @dec
        class Original:
            pass

        assert Original is Replacement

    def test_name_and_doc(self):
        dec = make_decorator(
            name="my_dec",
            steps=[],
            doc="My decorator.",
        )
        assert dec.__name__ == "my_dec"
        assert dec.__doc__ == "My decorator."
        assert dec._decorator_name == "my_dec"
        assert dec._is_decorator is True


# =============================================================================
# INTROSPECTION: decompose, expand
# =============================================================================


class TestIntrospection:
    def test_decompose_static(self):
        steps = [Step(Op.TAG, {"key": "a"}), Step(Op.REGISTER, {"registry": "b"})]
        dec = make_decorator(name="intro_test", steps=steps)
        assert decompose(dec) == steps

    def test_decompose_unknown(self):
        assert decompose("not_a_decorator") == []

    def test_expand_static(self):
        steps = [Step(Op.TAG, {"key": "x"})]
        dec = make_decorator(name="expand_test", steps=steps)
        result = expand(dec)
        assert "TAG" in result
        assert "x" in result

    def test_expand_unknown(self):
        result = expand("nope")
        assert "no steps defined" in result


# =============================================================================
# VALIDATE STEPS
# =============================================================================


class TestValidateSteps:
    def test_valid_combination(self):
        steps = [
            Step(Op.TAG, {"key": "x"}),
            Step(Op.TRACK),
        ]
        result = validate_steps(steps)
        assert result["valid"] is True

    def test_hook_on_change_requires_track(self):
        steps = [
            Step(Op.HOOK, {"event": HookEvent.ON_CHANGE}),
        ]
        result = validate_steps(steps)
        assert result["valid"] is False
        assert any("TRACK" in e for e in result["errors"])

    def test_hook_on_change_with_track_is_valid(self):
        steps = [
            Step(Op.TRACK),
            Step(Op.HOOK, {"event": HookEvent.ON_CHANGE}),
        ]
        result = validate_steps(steps)
        assert result["valid"] is True

    def test_intercept_deny_conflicts_track(self):
        steps = [
            Step(Op.INTERCEPT, {"set": "deny"}),
            Step(Op.TRACK),
        ]
        result = validate_steps(steps)
        assert result["valid"] is False

    def test_empty_steps_valid(self):
        result = validate_steps([])
        assert result["valid"] is True

    def test_custom_rule(self):
        custom = [
            Rule(
                name="TAG requires REGISTER",
                when=lambda s: any(st.op == Op.TAG for st in s),
                requires=lambda s: any(st.op == Op.REGISTER for st in s),
            )
        ]
        # TAG without REGISTER
        result = validate_steps([Step(Op.TAG, {"key": "x"})], rules=custom)
        assert result["valid"] is False

        # TAG with REGISTER
        result = validate_steps(
            [Step(Op.TAG, {"key": "x"}), Step(Op.REGISTER, {"registry": "r"})],
            rules=custom,
        )
        assert result["valid"] is True

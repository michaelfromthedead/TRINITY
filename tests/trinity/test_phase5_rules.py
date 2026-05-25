"""Comprehensive tests for Phase 5 composition rules and ordering validation."""

import pytest

from trinity.decorators.ops import (
    Step,
    Op,
    HookEvent,
    validate_steps,
    validate_ordering,
    RULES,
)


# ============================================================================
# Rule: HOOK(on_change) requires TRACK
# ============================================================================


class TestHookOnChangeRequiresTrack:
    def test_fails_hook_on_change_enum_without_track(self):
        steps = [Step(Op.HOOK, {"event": HookEvent.ON_CHANGE})]
        result = validate_steps(steps)
        assert result["valid"] is False
        assert any("HOOK(on_change) requires TRACK" in e for e in result["errors"])

    def test_fails_hook_on_change_string_without_track(self):
        steps = [Step(Op.HOOK, {"event": "on_change"})]
        result = validate_steps(steps)
        assert result["valid"] is False
        assert any("HOOK(on_change) requires TRACK" in e for e in result["errors"])

    def test_passes_hook_on_change_with_track(self):
        steps = [
            Step(Op.TRACK),
            Step(Op.HOOK, {"event": HookEvent.ON_CHANGE}),
        ]
        result = validate_steps(steps)
        assert result["valid"] is True


# ============================================================================
# Rule: INTERCEPT(set=deny) conflicts with TRACK
# ============================================================================


class TestInterceptDenyConflictsTrack:
    def test_fails_intercept_deny_then_track(self):
        steps = [Step(Op.INTERCEPT, {"set": "deny"}), Step(Op.TRACK)]
        result = validate_steps(steps)
        assert result["valid"] is False

    def test_fails_track_then_intercept_deny(self):
        steps = [Step(Op.TRACK), Step(Op.INTERCEPT, {"set": "deny"})]
        result = validate_steps(steps)
        assert result["valid"] is False

    def test_passes_intercept_allow_with_track(self):
        steps = [Step(Op.INTERCEPT, {"set": "allow"}), Step(Op.TRACK)]
        result = validate_steps(steps)
        assert result["valid"] is True


# ============================================================================
# Rule: REGISTER should be applied last
# ============================================================================


class TestRegisterLast:
    def test_passes_steps_ending_with_register(self):
        steps = [Step(Op.TAG, {"key": "x"}), Step(Op.REGISTER, {"registry": "R"})]
        result = validate_steps(steps)
        assert result["valid"] is True

    def test_passes_register_only(self):
        steps = [Step(Op.REGISTER, {"registry": "R"})]
        result = validate_steps(steps)
        assert result["valid"] is True

    def test_fails_register_in_middle(self):
        steps = [
            Step(Op.REGISTER, {"registry": "R"}),
            Step(Op.TAG, {"key": "x"}),
        ]
        result = validate_steps(steps)
        assert result["valid"] is False
        assert any("REGISTER" in e for e in result["errors"])

    def test_fails_multiple_register_with_non_register_between(self):
        steps = [
            Step(Op.REGISTER, {"registry": "A"}),
            Step(Op.TAG, {"key": "x"}),
            Step(Op.REGISTER, {"registry": "B"}),
        ]
        result = validate_steps(steps)
        assert result["valid"] is False

    def test_passes_multiple_register_all_at_end(self):
        steps = [
            Step(Op.TAG, {"key": "x"}),
            Step(Op.REGISTER, {"registry": "A"}),
            Step(Op.REGISTER, {"registry": "B"}),
        ]
        result = validate_steps(steps)
        assert result["valid"] is True


# ============================================================================
# Rule: TAG(network) requires TAG(serialization)
# ============================================================================


class TestTagNetworkRequiresSerialization:
    def test_fails_networked_alone(self):
        steps = [Step(Op.TAG, {"key": "networked"})]
        result = validate_steps(steps)
        assert result["valid"] is False
        assert any("TAG(network)" in e for e in result["errors"])

    def test_passes_networked_with_serialization_format(self):
        steps = [
            Step(Op.TAG, {"key": "networked"}),
            Step(Op.TAG, {"key": "serialization_format"}),
        ]
        result = validate_steps(steps)
        assert result["valid"] is True

    def test_passes_networked_with_serializable(self):
        steps = [
            Step(Op.TAG, {"key": "networked"}),
            Step(Op.TAG, {"key": "serializable"}),
        ]
        result = validate_steps(steps)
        assert result["valid"] is True

    def test_passes_no_networked_tag(self):
        steps = [Step(Op.TAG, {"key": "something_else"})]
        result = validate_steps(steps)
        assert result["valid"] is True


# ============================================================================
# Rule: INTERCEPT(set=deny) conflicts with VALIDATE
# ============================================================================


class TestInterceptDenyConflictsValidate:
    def test_fails_intercept_deny_with_validate(self):
        steps = [
            Step(Op.INTERCEPT, {"set": "deny"}),
            Step(Op.VALIDATE, {"constraint": "positive"}),
        ]
        result = validate_steps(steps)
        assert result["valid"] is False

    def test_passes_intercept_allow_with_validate(self):
        steps = [
            Step(Op.INTERCEPT, {"set": "allow"}),
            Step(Op.VALIDATE, {"constraint": "positive"}),
        ]
        result = validate_steps(steps)
        assert result["valid"] is True

    def test_passes_validate_alone(self):
        steps = [Step(Op.VALIDATE, {"constraint": "positive"})]
        result = validate_steps(steps)
        assert result["valid"] is True


# ============================================================================
# validate_ordering()
# ============================================================================


class TestValidateOrdering:
    def test_empty_list(self):
        result = validate_ordering([])
        assert result["valid"] is True

    def test_single_step(self):
        result = validate_ordering([Step(Op.HOOK, {"event": "on_create"})])
        assert result["valid"] is True

    def test_all_seven_canonical_order(self):
        steps = [
            Step(Op.TAG, {"key": "x"}),
            Step(Op.VALIDATE, {"constraint": "c"}),
            Step(Op.TRACK),
            Step(Op.INTERCEPT, {"get": "log"}),
            Step(Op.HOOK, {"event": "on_create"}),
            Step(Op.DESCRIBE),
            Step(Op.REGISTER, {"registry": "R"}),
        ]
        result = validate_ordering(steps)
        assert result["valid"] is True

    def test_reversed_order_multiple_errors(self):
        steps = [
            Step(Op.REGISTER, {"registry": "R"}),
            Step(Op.DESCRIBE),
            Step(Op.HOOK, {"event": "on_create"}),
            Step(Op.INTERCEPT, {"get": "log"}),
            Step(Op.TRACK),
            Step(Op.VALIDATE, {"constraint": "c"}),
            Step(Op.TAG, {"key": "x"}),
        ]
        result = validate_ordering(steps)
        assert result["valid"] is False
        assert len(result["errors"]) > 1

    def test_two_ops_swapped(self):
        steps = [
            Step(Op.HOOK, {"event": "on_create"}),
            Step(Op.TAG, {"key": "x"}),
        ]
        result = validate_ordering(steps)
        assert result["valid"] is False
        assert len(result["errors"]) == 1

    def test_duplicate_same_tier_ops(self):
        steps = [
            Step(Op.TAG, {"key": "a"}),
            Step(Op.TAG, {"key": "b"}),
            Step(Op.VALIDATE, {"constraint": "c"}),
        ]
        result = validate_ordering(steps)
        assert result["valid"] is True


# ============================================================================
# validate_steps(check_ordering=True) integration
# ============================================================================


class TestValidateStepsIntegration:
    def test_rule_and_ordering_violations_combined(self):
        # HOOK(on_change) without TRACK = rule violation
        # HOOK before TAG = ordering violation
        steps = [
            Step(Op.HOOK, {"event": HookEvent.ON_CHANGE}),
            Step(Op.TAG, {"key": "x"}),
        ]
        result = validate_steps(steps, check_ordering=True)
        assert result["valid"] is False
        errors = result["errors"]
        # At least one rule error and one ordering error
        assert any("HOOK(on_change) requires TRACK" in e for e in errors)
        assert any("TAG" in e and "position" in e for e in errors)

    def test_check_ordering_false_skips_ordering(self):
        # Out of order but no rule violations
        steps = [
            Step(Op.HOOK, {"event": "on_create"}),
            Step(Op.TAG, {"key": "x"}),
        ]
        result = validate_steps(steps, check_ordering=False)
        assert result["valid"] is True

    def test_default_skips_ordering(self):
        # Same as above, using default parameter
        steps = [
            Step(Op.HOOK, {"event": "on_create"}),
            Step(Op.TAG, {"key": "x"}),
        ]
        result = validate_steps(steps)
        assert result["valid"] is True

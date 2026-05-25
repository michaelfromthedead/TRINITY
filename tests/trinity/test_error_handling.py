"""
Tests for Trinity Pattern - Tier 40: ERROR_HANDLING Decorators
"""

import pytest

from trinity.decorators.error_handling import (
    VALID_ERROR_SCOPES,
    VALID_RECOVERY_STRATEGIES,
    bug_report,
    crash_safe,
    error_boundary,
    recoverable,
)
from trinity.decorators.ops import decompose
from trinity.decorators.registry import registry


class TestCrashSafe:
    """Test @crash_safe decorator."""

    def test_basic_application(self):
        """Test basic @crash_safe decorator application."""

        @crash_safe(recovery="retry")
        class TestClass:
            pass

        assert hasattr(TestClass, "_crash_safe")
        assert TestClass._crash_safe is True
        assert hasattr(TestClass, "_crash_recovery")
        assert TestClass._crash_recovery == "retry"

    def test_all_recovery_strategies(self):
        """Test all valid recovery strategies."""
        for strategy in VALID_RECOVERY_STRATEGIES:

            @crash_safe(recovery=strategy)
            class TestClass:
                pass

            assert TestClass._crash_recovery == strategy

    def test_invalid_recovery_strategy(self):
        """Test invalid recovery strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid recovery strategy"):

            @crash_safe(recovery="invalid")
            class TestClass:
                pass

    def test_tags_and_registry(self):
        """Test that tags and registry are set correctly."""

        @crash_safe(recovery="skip")
        class TestClass:
            pass

        assert hasattr(TestClass, "_tags")
        assert TestClass._tags.get("crash_safe") is True
        assert TestClass._tags.get("crash_recovery") == "skip"
        assert hasattr(TestClass, "_registries")
        assert "error_handling" in TestClass._registries

    def test_decorator_tracking(self):
        """Test decorator application tracking."""

        @crash_safe(recovery="fallback")
        class TestClass:
            pass

        assert hasattr(TestClass, "_applied_decorators")
        assert "crash_safe" in TestClass._applied_decorators

    def test_step_decomposition(self):
        """Test decorator step decomposition."""
        steps = decompose(crash_safe)
        assert len(steps) == 3
        assert any(s.op.value == "tag" for s in steps)
        assert any(s.op.value == "register" for s in steps)

    def test_on_function(self):
        """Test @crash_safe on a function."""

        @crash_safe(recovery="crash")
        def test_func():
            pass

        assert test_func._crash_safe is True
        assert test_func._crash_recovery == "crash"


class TestRecoverable:
    """Test @recoverable decorator."""

    def test_basic_application(self):
        """Test basic @recoverable decorator application."""

        @recoverable(checkpoint=True)
        class TestClass:
            pass

        assert hasattr(TestClass, "_recoverable")
        assert TestClass._recoverable is True
        assert hasattr(TestClass, "_recoverable_checkpoint")
        assert TestClass._recoverable_checkpoint is True

    def test_checkpoint_false(self):
        """Test @recoverable with checkpoint=False."""

        @recoverable(checkpoint=False)
        class TestClass:
            pass

        assert TestClass._recoverable is True
        assert TestClass._recoverable_checkpoint is False

    def test_default_checkpoint(self):
        """Test @recoverable with default checkpoint value."""

        @recoverable()
        class TestClass:
            pass

        assert TestClass._recoverable_checkpoint is True

    def test_tags_and_registry(self):
        """Test that tags and registry are set correctly."""

        @recoverable(checkpoint=True)
        class TestClass:
            pass

        assert hasattr(TestClass, "_tags")
        assert TestClass._tags.get("recoverable") is True
        assert TestClass._tags.get("recoverable_checkpoint") is True
        assert hasattr(TestClass, "_registries")
        assert "error_handling" in TestClass._registries

    def test_tracking_enabled(self):
        """Test that TRACK op is applied."""

        @recoverable()
        class TestClass:
            pass

        assert hasattr(TestClass, "_tracked") or hasattr(TestClass, "_tracked_fields")

    def test_step_decomposition(self):
        """Test decorator step decomposition."""
        steps = decompose(recoverable)
        assert len(steps) == 4
        assert any(s.op.value == "track" for s in steps)


class TestErrorBoundary:
    """Test @error_boundary decorator."""

    def test_basic_application(self):
        """Test basic @error_boundary decorator application."""

        @error_boundary(scope="system")
        class TestClass:
            pass

        assert hasattr(TestClass, "_error_boundary")
        assert TestClass._error_boundary is True
        assert hasattr(TestClass, "_error_scope")
        assert TestClass._error_scope == "system"

    def test_all_scopes(self):
        """Test all valid error scopes."""
        for scope in VALID_ERROR_SCOPES:

            @error_boundary(scope=scope)
            class TestClass:
                pass

            assert TestClass._error_scope == scope

    def test_default_scope(self):
        """Test @error_boundary with default scope."""

        @error_boundary()
        class TestClass:
            pass

        assert TestClass._error_scope == "system"

    def test_invalid_scope(self):
        """Test invalid scope raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scope"):

            @error_boundary(scope="invalid")
            class TestClass:
                pass

    def test_tags_and_registry(self):
        """Test that tags and registry are set correctly."""

        @error_boundary(scope="entity")
        class TestClass:
            pass

        assert hasattr(TestClass, "_tags")
        assert TestClass._tags.get("error_boundary") is True
        assert TestClass._tags.get("error_scope") == "entity"
        assert hasattr(TestClass, "_registries")
        assert "error_handling" in TestClass._registries

    def test_on_function(self):
        """Test @error_boundary on a function."""

        @error_boundary(scope="global")
        def test_func():
            pass

        assert test_func._error_boundary is True
        assert test_func._error_scope == "global"


class TestBugReport:
    """Test @bug_report decorator."""

    def test_basic_application(self):
        """Test basic @bug_report decorator application."""

        @bug_report(include={"screenshot", "logs"})
        class TestClass:
            pass

        assert hasattr(TestClass, "_bug_report")
        assert TestClass._bug_report is True
        assert hasattr(TestClass, "_bug_report_include")
        assert TestClass._bug_report_include == frozenset({"screenshot", "logs"})

    def test_default_include(self):
        """Test @bug_report with default include."""

        @bug_report()
        class TestClass:
            pass

        assert TestClass._bug_report_include == frozenset(
            {"screenshot", "logs", "save", "replay"}
        )

    def test_empty_include_raises_error(self):
        """Test empty include set raises ValueError."""
        with pytest.raises(ValueError, match="include must be a non-empty set"):

            @bug_report(include=set())
            class TestClass:
                pass

    def test_invalid_include_type(self):
        """Test invalid include type raises TypeError."""
        with pytest.raises(TypeError, match="include must be a set"):

            @bug_report(include=["screenshot", "logs"])
            class TestClass:
                pass

    def test_tags_and_registry(self):
        """Test that tags and registry are set correctly."""

        @bug_report(include={"screenshot", "save"})
        class TestClass:
            pass

        assert hasattr(TestClass, "_tags")
        assert TestClass._tags.get("bug_report") is True
        assert TestClass._tags.get("bug_report_include") == frozenset(
            {"screenshot", "save"}
        )
        assert hasattr(TestClass, "_registries")
        assert "error_handling" in TestClass._registries

    def test_on_function(self):
        """Test @bug_report on a function."""

        @bug_report(include={"logs", "replay"})
        def test_func():
            pass

        assert test_func._bug_report is True
        assert test_func._bug_report_include == frozenset({"logs", "replay"})


class TestComposition:
    """Test decorator composition and stacking."""

    def test_multiple_error_handling_decorators(self):
        """Test stacking multiple error handling decorators."""

        @crash_safe(recovery="retry")
        @recoverable(checkpoint=True)
        @error_boundary(scope="system")
        class TestClass:
            pass

        assert TestClass._crash_safe is True
        assert TestClass._recoverable is True
        assert TestClass._error_boundary is True

    def test_all_four_decorators(self):
        """Test stacking all four decorators."""

        @crash_safe(recovery="fallback")
        @recoverable(checkpoint=False)
        @error_boundary(scope="entity")
        @bug_report(include={"screenshot", "logs", "save"})
        class TestClass:
            pass

        assert TestClass._crash_safe is True
        assert TestClass._recoverable is True
        assert TestClass._error_boundary is True
        assert TestClass._bug_report is True
        assert len(TestClass._applied_decorators) == 4


class TestRegistry:
    """Test registry integration."""

    def test_decorators_registered(self):
        """Test that all decorators are registered."""
        assert registry.get("crash_safe") is not None
        assert registry.get("recoverable") is not None
        assert registry.get("error_boundary") is not None
        assert registry.get("bug_report") is not None

    def test_tier_assignment(self):
        """Test that decorators are assigned to correct tier."""
        from trinity.decorators.registry import Tier

        tier_decorators = registry.by_tier(Tier.ERROR_HANDLING)
        decorator_names = {spec.name for spec in tier_decorators}

        assert "crash_safe" in decorator_names
        assert "recoverable" in decorator_names
        assert "error_boundary" in decorator_names
        assert "bug_report" in decorator_names

    def test_decorator_specs(self):
        """Test decorator specifications."""
        crash_safe_spec = registry.get("crash_safe")
        assert crash_safe_spec is not None
        assert crash_safe_spec.tier.value == 40
        assert not crash_safe_spec.foundation
        assert ("class", "function") == crash_safe_spec.target_types or (
            "function", "class"
        ) == crash_safe_spec.target_types


class TestValidation:
    """Test parameter validation."""

    def test_crash_safe_validation(self):
        """Test @crash_safe parameter validation."""
        with pytest.raises(ValueError):

            @crash_safe(recovery="bad_strategy")
            class TestClass:
                pass

    def test_error_boundary_validation(self):
        """Test @error_boundary parameter validation."""
        with pytest.raises(ValueError):

            @error_boundary(scope="bad_scope")
            class TestClass:
                pass

    def test_bug_report_validation(self):
        """Test @bug_report parameter validation."""
        with pytest.raises(ValueError):

            @bug_report(include=set())
            class TestClass:
                pass

        with pytest.raises(TypeError):

            @bug_report(include=["not", "a", "set"])
            class TestClass:
                pass


class TestConstants:
    """Test exported constants."""

    def test_valid_recovery_strategies(self):
        """Test VALID_RECOVERY_STRATEGIES constant."""
        assert VALID_RECOVERY_STRATEGIES == frozenset(
            {"retry", "skip", "fallback", "crash"}
        )

    def test_valid_error_scopes(self):
        """Test VALID_ERROR_SCOPES constant."""
        assert VALID_ERROR_SCOPES == frozenset({"system", "entity", "global"})

"""
Tests for Tier 10 (DEBUG_SAFETY) and Tier 11 (CHANGE_DETECTION) decorators.
"""

import pytest

from trinity.decorators.debug_safety import reads, track_changes, trace_stack, writes
from trinity.decorators.ecs_core import component
from trinity.decorators.ops import Op
from trinity.decorators.registry import inspect_decorated


# =============================================================================
# TEST HELPERS
# =============================================================================


class Transform:
    """Mock component for testing."""

    pass


class Velocity:
    """Mock component for testing."""

    pass


class Health:
    """Mock component for testing."""

    pass


# =============================================================================
# TEST @reads
# =============================================================================


class TestReads:
    """Tests for @reads decorator."""

    def test_reads_single_component(self):
        """Test @reads with a single component type."""

        @reads(Transform)
        def system():
            pass

        assert hasattr(system, "_reads")
        assert system._reads is True
        assert hasattr(system, "_reads_components")
        assert system._reads_components == (Transform,)

    def test_reads_multiple_components(self):
        """Test @reads with multiple component types."""

        @reads(Transform, Velocity, Health)
        def system():
            pass

        assert system._reads is True
        assert system._reads_components == (Transform, Velocity, Health)
        assert len(system._reads_components) == 3

    def test_reads_applied_decorators(self):
        """Test that @reads tracks in _applied_decorators."""

        @reads(Transform)
        def system():
            pass

        assert hasattr(system, "_applied_decorators")
        assert "reads" in system._applied_decorators

    def test_reads_applied_steps(self):
        """Test that @reads records steps correctly."""

        @reads(Transform, Velocity)
        def system():
            pass

        assert hasattr(system, "_applied_steps")
        steps = system._applied_steps

        # Check for TAG steps
        tag_steps = [s for s in steps if s.op == Op.TAG]
        assert len(tag_steps) == 2

        # Check for REGISTER step
        register_steps = [s for s in steps if s.op == Op.REGISTER]
        assert len(register_steps) == 1
        assert register_steps[0].args["registry"] == "debug_safety"

    def test_reads_tags(self):
        """Test that @reads sets correct tags."""

        @reads(Transform)
        def system():
            pass

        assert hasattr(system, "_tags")
        assert system._tags["reads"] is True
        assert system._tags["reads_components"] == (Transform,)

    def test_reads_registries(self):
        """Test that @reads registers in debug_safety."""

        @reads(Transform)
        def system():
            pass

        assert hasattr(system, "_registries")
        assert "debug_safety" in system._registries

    def test_reads_introspection(self):
        """Test introspection of @reads decorated function."""

        @reads(Transform, Velocity)
        def system():
            pass

        info = inspect_decorated(system)
        assert "reads" in info.decorators
        assert "_reads" in info.attributes
        assert info.attributes["_reads"] is True


# =============================================================================
# TEST @writes
# =============================================================================


class TestWrites:
    """Tests for @writes decorator."""

    def test_writes_single_component(self):
        """Test @writes with a single component type."""

        @writes(Velocity)
        def system():
            pass

        assert hasattr(system, "_writes")
        assert system._writes is True
        assert hasattr(system, "_writes_components")
        assert system._writes_components == (Velocity,)

    def test_writes_multiple_components(self):
        """Test @writes with multiple component types."""

        @writes(Transform, Velocity)
        def system():
            pass

        assert system._writes is True
        assert system._writes_components == (Transform, Velocity)
        assert len(system._writes_components) == 2

    def test_writes_applied_decorators(self):
        """Test that @writes tracks in _applied_decorators."""

        @writes(Velocity)
        def system():
            pass

        assert hasattr(system, "_applied_decorators")
        assert "writes" in system._applied_decorators

    def test_writes_applied_steps(self):
        """Test that @writes records steps correctly."""

        @writes(Transform)
        def system():
            pass

        assert hasattr(system, "_applied_steps")
        steps = system._applied_steps

        # Check for TAG steps
        tag_steps = [s for s in steps if s.op == Op.TAG]
        assert len(tag_steps) == 2

        # Check for REGISTER step
        register_steps = [s for s in steps if s.op == Op.REGISTER]
        assert len(register_steps) == 1
        assert register_steps[0].args["registry"] == "debug_safety"

    def test_writes_tags(self):
        """Test that @writes sets correct tags."""

        @writes(Velocity)
        def system():
            pass

        assert hasattr(system, "_tags")
        assert system._tags["writes"] is True
        assert system._tags["writes_components"] == (Velocity,)

    def test_writes_registries(self):
        """Test that @writes registers in debug_safety."""

        @writes(Velocity)
        def system():
            pass

        assert hasattr(system, "_registries")
        assert "debug_safety" in system._registries

    def test_writes_introspection(self):
        """Test introspection of @writes decorated function."""

        @writes(Transform, Velocity)
        def system():
            pass

        info = inspect_decorated(system)
        assert "writes" in info.decorators
        assert "_writes" in info.attributes
        assert info.attributes["_writes"] is True


# =============================================================================
# TEST @reads + @writes COMBINATION
# =============================================================================


class TestReadsWritesCombination:
    """Tests for combining @reads and @writes."""

    def test_reads_and_writes(self):
        """Test system that reads some components and writes others."""

        @reads(Transform, Health)
        @writes(Velocity)
        def physics_system():
            pass

        assert physics_system._reads is True
        assert physics_system._writes is True
        assert physics_system._reads_components == (Transform, Health)
        assert physics_system._writes_components == (Velocity,)

    def test_reads_writes_applied_decorators(self):
        """Test decorator chain for reads + writes."""

        @reads(Transform)
        @writes(Velocity)
        def system():
            pass

        assert "reads" in system._applied_decorators
        assert "writes" in system._applied_decorators


# =============================================================================
# TEST @trace_stack
# =============================================================================


class TestTraceStack:
    """Tests for @trace_stack decorator."""

    def test_trace_stack_defaults(self):
        """Test @trace_stack with default parameters."""

        @trace_stack()
        def system():
            pass

        assert hasattr(system, "_trace_stack")
        assert system._trace_stack is True
        assert system._trace_stack_depth == 3
        assert system._trace_stack_show_chain is True

    def test_trace_stack_custom_depth(self):
        """Test @trace_stack with custom depth."""

        @trace_stack(depth=5)
        def system():
            pass

        assert system._trace_stack is True
        assert system._trace_stack_depth == 5
        assert system._trace_stack_show_chain is True

    def test_trace_stack_custom_show_chain(self):
        """Test @trace_stack with show_decorator_chain disabled."""

        @trace_stack(show_decorator_chain=False)
        def system():
            pass

        assert system._trace_stack is True
        assert system._trace_stack_depth == 3
        assert system._trace_stack_show_chain is False

    def test_trace_stack_all_custom(self):
        """Test @trace_stack with all custom parameters."""

        @trace_stack(depth=10, show_decorator_chain=False)
        def system():
            pass

        assert system._trace_stack is True
        assert system._trace_stack_depth == 10
        assert system._trace_stack_show_chain is False

    def test_trace_stack_applied_decorators(self):
        """Test that @trace_stack tracks in _applied_decorators."""

        @trace_stack()
        def system():
            pass

        assert "trace_stack" in system._applied_decorators

    def test_trace_stack_has_hook(self):
        """Test that @trace_stack registers an on_error hook."""

        @trace_stack()
        def system():
            pass

        assert hasattr(system, "_applied_steps")
        hook_steps = [s for s in system._applied_steps if s.op == Op.HOOK]
        assert len(hook_steps) == 1
        assert hook_steps[0].args["event"] == "on_error"

    def test_trace_stack_tags(self):
        """Test that @trace_stack sets correct tags."""

        @trace_stack(depth=7, show_decorator_chain=False)
        def system():
            pass

        assert hasattr(system, "_tags")
        assert system._tags["trace_stack"] is True
        assert system._tags["trace_stack_depth"] == 7
        assert system._tags["trace_stack_show_chain"] is False

    def test_trace_stack_registries(self):
        """Test that @trace_stack registers in debug_safety."""

        @trace_stack()
        def system():
            pass

        assert hasattr(system, "_registries")
        assert "debug_safety" in system._registries

    def test_trace_stack_validation_invalid_depth(self):
        """Test that @trace_stack validates depth parameter."""
        with pytest.raises(TypeError, match="depth must be a positive integer"):

            @trace_stack(depth=0)
            def system():
                pass

        with pytest.raises(TypeError, match="depth must be a positive integer"):

            @trace_stack(depth=-1)
            def system():
                pass

        with pytest.raises(TypeError, match="depth must be a positive integer"):

            @trace_stack(depth="not an int")
            def system():
                pass

    def test_trace_stack_validation_invalid_show_chain(self):
        """Test that @trace_stack validates show_decorator_chain parameter."""
        with pytest.raises(TypeError, match="show_decorator_chain must be a boolean"):

            @trace_stack(show_decorator_chain="yes")
            def system():
                pass

    def test_trace_stack_introspection(self):
        """Test introspection of @trace_stack decorated function."""

        @trace_stack(depth=5)
        def system():
            pass

        info = inspect_decorated(system)
        assert "trace_stack" in info.decorators
        assert "_trace_stack" in info.attributes
        assert info.attributes["_trace_stack"] is True
        assert info.attributes["_trace_stack_depth"] == 5


# =============================================================================
# TEST @track_changes
# =============================================================================


class TestTrackChanges:
    """Tests for @track_changes decorator."""

    def test_track_changes_all_fields(self):
        """Test @track_changes with fields=None (track all)."""

        @component
        @track_changes()
        class Position:
            x: float
            y: float

        assert hasattr(Position, "_tracked")
        assert Position._tracked is True
        assert Position._tracked_fields is None

    def test_track_changes_specific_fields(self):
        """Test @track_changes with specific fields."""

        @component
        @track_changes(fields=["x", "y"])
        class Position:
            x: float
            y: float
            z: float

        assert Position._tracked is True
        assert Position._tracked_fields == ["x", "y"]

    def test_track_changes_requires_component(self):
        """Test that @track_changes requires @component."""
        # The requirement is declared in registry metadata
        from trinity.decorators.registry import registry

        spec = registry.get("track_changes")
        assert spec is not None
        assert "component" in spec.requires

        # Registry validation would catch this at runtime
        # For now, just verify the metadata is correct

    def test_track_changes_applied_decorators(self):
        """Test that @track_changes tracks in _applied_decorators."""

        @component
        @track_changes()
        class Position:
            x: float

        assert "track_changes" in Position._applied_decorators
        assert "component" in Position._applied_decorators

    def test_track_changes_has_track_step(self):
        """Test that @track_changes includes TRACK step."""

        @component
        @track_changes()
        class Position:
            x: float

        assert hasattr(Position, "_applied_steps")
        track_steps = [s for s in Position._applied_steps if s.op == Op.TRACK]
        # May have multiple TRACK steps if component also tracks
        assert len(track_steps) >= 1

    def test_track_changes_tags(self):
        """Test that @track_changes sets correct tags."""

        @component
        @track_changes(fields=["x"])
        class Position:
            x: float

        assert hasattr(Position, "_tags")
        assert Position._tags["track_changes"] is True
        assert Position._tags["track_changes_fields"] == ["x"]

    def test_track_changes_registries(self):
        """Test that @track_changes registers in change_detection."""

        @component
        @track_changes()
        class Position:
            x: float

        assert hasattr(Position, "_registries")
        assert "change_detection" in Position._registries

    def test_track_changes_validation_invalid_fields_type(self):
        """Test that @track_changes validates fields parameter type."""
        with pytest.raises(TypeError, match="fields must be a list of strings or None"):

            @component
            @track_changes(fields="x,y")
            class Position:
                x: float

    def test_track_changes_validation_invalid_field_items(self):
        """Test that @track_changes validates field items are strings."""
        with pytest.raises(TypeError, match="all fields must be strings"):

            @component
            @track_changes(fields=["x", 123])
            class Position:
                x: float

    def test_track_changes_introspection(self):
        """Test introspection of @track_changes decorated class."""

        @component
        @track_changes(fields=["x", "y"])
        class Position:
            x: float
            y: float
            z: float

        info = inspect_decorated(Position)
        assert "track_changes" in info.decorators
        assert "_tracked" in info.attributes
        assert info.attributes["_tracked"] is True
        assert info.attributes["_tracked_fields"] == ["x", "y"]


# =============================================================================
# PARAMETRIZED INTROSPECTION TESTS
# =============================================================================


@pytest.mark.parametrize(
    "decorator_name,decorator_func,apply_to",
    [
        ("reads", lambda: reads(Transform, Velocity), "function"),
        ("writes", lambda: writes(Transform), "function"),
        ("trace_stack", lambda: trace_stack(depth=5), "function"),
    ],
)
def test_decorator_introspection(decorator_name, decorator_func, apply_to):
    """Parametrized test for decorator introspection."""
    if apply_to == "function":

        @decorator_func()
        def target():
            pass

    else:

        @decorator_func()
        class target:
            pass

    # Check applied decorators
    assert hasattr(target, "_applied_decorators")
    assert decorator_name in target._applied_decorators

    # Check applied steps
    assert hasattr(target, "_applied_steps")
    assert len(target._applied_steps) > 0

    # Check introspection
    info = inspect_decorated(target)
    assert decorator_name in info.decorators


@pytest.mark.parametrize(
    "decorator_name,expected_registry",
    [
        ("reads", "debug_safety"),
        ("writes", "debug_safety"),
        ("trace_stack", "debug_safety"),
        ("track_changes", "change_detection"),
    ],
)
def test_decorator_registries(decorator_name, expected_registry):
    """Parametrized test for decorator registry membership."""
    if decorator_name in ("reads", "writes"):

        @globals()[decorator_name](Transform)
        def target():
            pass

    elif decorator_name == "trace_stack":

        @trace_stack()
        def target():
            pass

    else:  # track_changes

        @component
        @track_changes()
        class target:
            x: float

    assert hasattr(target, "_registries")
    assert expected_registry in target._registries


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_reads_empty_components(self):
        """Test @reads with no components (valid but unusual)."""

        @reads()
        def system():
            pass

        assert system._reads is True
        assert system._reads_components == ()

    def test_writes_empty_components(self):
        """Test @writes with no components (valid but unusual)."""

        @writes()
        def system():
            pass

        assert system._writes is True
        assert system._writes_components == ()

    def test_trace_stack_no_parens(self):
        """Test @trace_stack applied without parentheses."""

        @trace_stack
        def system():
            pass

        assert system._trace_stack is True
        assert system._trace_stack_depth == 3

    def test_track_changes_empty_fields_list(self):
        """Test @track_changes with empty fields list."""

        @component
        @track_changes(fields=[])
        class Position:
            x: float

        assert Position._tracked is True
        assert Position._tracked_fields == []

    def test_multiple_decorators_chain(self):
        """Test combining all debug_safety decorators."""

        @reads(Transform, Health)
        @writes(Velocity)
        @trace_stack(depth=5)
        def complex_system():
            pass

        assert complex_system._reads is True
        assert complex_system._writes is True
        assert complex_system._trace_stack is True
        assert len(complex_system._applied_decorators) == 3
        assert "reads" in complex_system._applied_decorators
        assert "writes" in complex_system._applied_decorators
        assert "trace_stack" in complex_system._applied_decorators

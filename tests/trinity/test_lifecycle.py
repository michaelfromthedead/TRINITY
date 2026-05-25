"""Tests for Tier 7: Lifecycle decorators."""

from __future__ import annotations

import pytest

from trinity.decorators.lifecycle import (
    on_add,
    on_change,
    on_despawn,
    on_remove,
    on_spawn,
)
from trinity.decorators.ops import Op, Step, decompose, expand


class TestOnAdd:
    """Tests for @on_add decorator."""

    def test_basic(self):
        """Test basic @on_add usage with component type."""

        class Health:
            pass

        @on_add(component=Health)
        def handle_add(entity):
            pass

        assert handle_add._on_add_component is Health
        assert handle_add._lifecycle_hook == "add"

    def test_function_callable(self):
        """Test that decorated function remains callable."""

        class Position:
            pass

        @on_add(component=Position)
        def handle_add(entity):
            return "added"

        assert callable(handle_add)
        assert handle_add(None) == "added"

    def test_applied_decorators(self):
        """Test that decorator is recorded in _applied_decorators."""

        class Pos:
            pass

        @on_add(component=Pos)
        def fn(e):
            pass

        assert "on_add" in fn._applied_decorators

    def test_steps_recorded(self):
        """Test that steps are recorded in _applied_steps."""

        class C:
            pass

        @on_add(component=C)
        def fn(e):
            pass

        ops = {s.op for s in fn._applied_steps}
        assert Op.TAG in ops
        assert Op.HOOK in ops
        assert Op.REGISTER in ops

    def test_component_tagged(self):
        """Test that component type is stored in tags."""

        class Velocity:
            pass

        @on_add(component=Velocity)
        def fn(e):
            pass

        steps = fn._applied_steps
        component_tags = [s for s in steps if s.op == Op.TAG and s.args.get("key") == "on_add_component"]
        assert len(component_tags) == 1
        assert component_tags[0].args["value"] is Velocity


class TestOnRemove:
    """Tests for @on_remove decorator."""

    def test_basic(self):
        """Test basic @on_remove usage."""

        class Health:
            pass

        @on_remove(component=Health)
        def handle_remove(entity):
            pass

        assert handle_remove._on_remove_component is Health
        assert handle_remove._lifecycle_hook == "remove"

    def test_function_callable(self):
        """Test that decorated function remains callable."""

        class Position:
            pass

        @on_remove(component=Position)
        def handle_remove(entity):
            return "removed"

        assert callable(handle_remove)
        assert handle_remove(None) == "removed"

    def test_applied_decorators(self):
        """Test that decorator is recorded."""

        class C:
            pass

        @on_remove(component=C)
        def fn(e):
            pass

        assert "on_remove" in fn._applied_decorators

    def test_steps_recorded(self):
        """Test that steps are recorded."""

        class C:
            pass

        @on_remove(component=C)
        def fn(e):
            pass

        ops = {s.op for s in fn._applied_steps}
        assert Op.TAG in ops
        assert Op.HOOK in ops
        assert Op.REGISTER in ops

    def test_component_tagged(self):
        """Test that component type is tagged."""

        class Sprite:
            pass

        @on_remove(component=Sprite)
        def fn(e):
            pass

        steps = fn._applied_steps
        component_tags = [s for s in steps if s.op == Op.TAG and s.args.get("key") == "on_remove_component"]
        assert len(component_tags) == 1
        assert component_tags[0].args["value"] is Sprite


class TestOnChange:
    """Tests for @on_change decorator."""

    def test_basic(self):
        """Test basic @on_change usage."""

        class Health:
            pass

        @on_change(component=Health)
        def handle_change(entity, old_value, new_value):
            pass

        assert handle_change._on_change_component is Health
        assert handle_change._lifecycle_hook == "change"

    def test_function_callable(self):
        """Test that decorated function remains callable."""

        class Position:
            pass

        @on_change(component=Position)
        def handle_change(entity, old, new):
            return "changed"

        assert callable(handle_change)
        assert handle_change(None, None, None) == "changed"

    def test_applied_decorators(self):
        """Test that decorator is recorded."""

        class C:
            pass

        @on_change(component=C)
        def fn(e, o, n):
            pass

        assert "on_change" in fn._applied_decorators

    def test_steps_recorded(self):
        """Test that steps are recorded."""

        class C:
            pass

        @on_change(component=C)
        def fn(e, o, n):
            pass

        ops = {s.op for s in fn._applied_steps}
        assert Op.TAG in ops
        assert Op.HOOK in ops
        assert Op.REGISTER in ops

    def test_component_tagged(self):
        """Test that component type is tagged."""

        class Animation:
            pass

        @on_change(component=Animation)
        def fn(e, o, n):
            pass

        steps = fn._applied_steps
        component_tags = [s for s in steps if s.op == Op.TAG and s.args.get("key") == "on_change_component"]
        assert len(component_tags) == 1
        assert component_tags[0].args["value"] is Animation


class TestOnSpawn:
    """Tests for @on_spawn decorator."""

    def test_basic_with_parens(self):
        """Test @on_spawn() with parentheses."""

        @on_spawn()
        def handle_spawn(entity):
            pass

        assert handle_spawn._on_spawn is True
        assert handle_spawn._lifecycle_hook == "spawn"

    def test_basic_without_parens(self):
        """Test @on_spawn without parentheses."""

        @on_spawn
        def handle_spawn(entity):
            pass

        assert handle_spawn._on_spawn is True
        assert handle_spawn._lifecycle_hook == "spawn"

    def test_function_callable(self):
        """Test that decorated function remains callable."""

        @on_spawn()
        def handle_spawn(entity):
            return "spawned"

        assert callable(handle_spawn)
        assert handle_spawn(None) == "spawned"

    def test_applied_decorators(self):
        """Test that decorator is recorded."""

        @on_spawn()
        def fn(e):
            pass

        assert "on_spawn" in fn._applied_decorators

    def test_steps_recorded(self):
        """Test that steps are recorded."""

        @on_spawn()
        def fn(e):
            pass

        ops = {s.op for s in fn._applied_steps}
        assert Op.TAG in ops
        assert Op.HOOK in ops
        assert Op.REGISTER in ops

    def test_on_spawn_tag(self):
        """Test that on_spawn tag is set."""

        @on_spawn()
        def fn(e):
            pass

        steps = fn._applied_steps
        spawn_tags = [s for s in steps if s.op == Op.TAG and s.args.get("key") == "on_spawn"]
        assert len(spawn_tags) == 1
        assert spawn_tags[0].args["value"] is True


class TestOnDespawn:
    """Tests for @on_despawn decorator."""

    def test_basic_with_parens(self):
        """Test @on_despawn() with parentheses."""

        @on_despawn()
        def handle_despawn(entity):
            pass

        assert handle_despawn._on_despawn is True
        assert handle_despawn._lifecycle_hook == "despawn"

    def test_basic_without_parens(self):
        """Test @on_despawn without parentheses."""

        @on_despawn
        def handle_despawn(entity):
            pass

        assert handle_despawn._on_despawn is True
        assert handle_despawn._lifecycle_hook == "despawn"

    def test_function_callable(self):
        """Test that decorated function remains callable."""

        @on_despawn()
        def handle_despawn(entity):
            return "despawned"

        assert callable(handle_despawn)
        assert handle_despawn(None) == "despawned"

    def test_applied_decorators(self):
        """Test that decorator is recorded."""

        @on_despawn()
        def fn(e):
            pass

        assert "on_despawn" in fn._applied_decorators

    def test_steps_recorded(self):
        """Test that steps are recorded."""

        @on_despawn()
        def fn(e):
            pass

        ops = {s.op for s in fn._applied_steps}
        assert Op.TAG in ops
        assert Op.HOOK in ops
        assert Op.REGISTER in ops

    def test_on_despawn_tag(self):
        """Test that on_despawn tag is set."""

        @on_despawn()
        def fn(e):
            pass

        steps = fn._applied_steps
        despawn_tags = [s for s in steps if s.op == Op.TAG and s.args.get("key") == "on_despawn"]
        assert len(despawn_tags) == 1
        assert despawn_tags[0].args["value"] is True


class TestLifecycleIntrospection:
    """Tests for lifecycle decorator introspection."""

    @pytest.mark.parametrize(
        "decorator,component_required",
        [
            (on_add, True),
            (on_remove, True),
            (on_change, True),
            (on_spawn, False),
            (on_despawn, False),
        ],
    )
    def test_decompose(self, decorator, component_required):
        """Test that applied steps are recorded on decorated function."""

        class TestComponent:
            pass

        if component_required:

            @decorator(component=TestComponent)
            def fn():
                pass
        else:

            @decorator()
            def fn():
                pass

        # Check that steps were applied to the function
        assert hasattr(fn, "_applied_steps")
        steps = fn._applied_steps
        assert len(steps) > 0
        assert all(isinstance(s, Step) for s in steps)

    @pytest.mark.parametrize(
        "decorator,component_required",
        [
            (on_add, True),
            (on_remove, True),
            (on_change, True),
            (on_spawn, False),
            (on_despawn, False),
        ],
    )
    def test_expand(self, decorator, component_required):
        """Test that decorator name is recorded."""

        class TestComponent:
            pass

        if component_required:

            @decorator(component=TestComponent)
            def fn():
                pass
        else:

            @decorator()
            def fn():
                pass

        # Check that decorator name was recorded
        assert hasattr(fn, "_applied_decorators")
        assert decorator.__name__ in fn._applied_decorators

    @pytest.mark.parametrize(
        "decorator,component_required",
        [
            (on_add, True),
            (on_remove, True),
            (on_change, True),
            (on_spawn, False),
            (on_despawn, False),
        ],
    )
    def test_register_step(self, decorator, component_required):
        """Test that all lifecycle decorators have REGISTER(lifecycle) step."""

        class TestComponent:
            pass

        if component_required:

            @decorator(component=TestComponent)
            def fn():
                pass
        else:

            @decorator()
            def fn():
                pass

        steps = fn._applied_steps
        register_steps = [s for s in steps if s.op == Op.REGISTER]
        assert len(register_steps) == 1
        assert register_steps[0].args.get("registry") == "lifecycle"

    def test_multiple_lifecycle_hooks(self):
        """Test that multiple lifecycle decorators can coexist."""

        class Health:
            pass

        class Position:
            pass

        @on_add(component=Health)
        @on_remove(component=Position)
        def multi_hook(entity):
            pass

        assert multi_hook._on_add_component is Health
        assert multi_hook._on_remove_component is Position
        assert "on_add" in multi_hook._applied_decorators
        assert "on_remove" in multi_hook._applied_decorators


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_on_add_requires_component(self):
        """Test that @on_add raises ValueError when component is missing."""
        with pytest.raises(ValueError, match="component parameter is required"):

            @on_add()
            def handler(entity):
                pass

    def test_on_remove_requires_component(self):
        """Test that @on_remove raises ValueError when component is missing."""
        with pytest.raises(ValueError, match="component parameter is required"):

            @on_remove()
            def handler(entity):
                pass

    def test_on_change_requires_component(self):
        """Test that @on_change raises ValueError when component is missing."""
        with pytest.raises(ValueError, match="component parameter is required"):

            @on_change()
            def handler(entity, old, new):
                pass

    def test_on_add_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""

        class C:
            pass

        @on_add(component=C)
        def my_handler(entity):
            """Handler docstring."""
            pass

        assert my_handler.__name__ == "my_handler"
        assert my_handler.__doc__ == "Handler docstring."

    def test_on_spawn_preserves_function_metadata(self):
        """Test that marker decorator preserves function metadata."""

        @on_spawn()
        def spawn_handler(entity):
            """Spawn handler."""
            pass

        assert spawn_handler.__name__ == "spawn_handler"
        assert spawn_handler.__doc__ == "Spawn handler."

    def test_lifecycle_hook_attribute_uniqueness(self):
        """Test that each decorator sets unique lifecycle_hook value."""

        class C:
            pass

        @on_add(component=C)
        def add_fn(e):
            pass

        @on_remove(component=C)
        def remove_fn(e):
            pass

        @on_change(component=C)
        def change_fn(e):
            pass

        @on_spawn()
        def spawn_fn(e):
            pass

        @on_despawn()
        def despawn_fn(e):
            pass

        assert add_fn._lifecycle_hook == "add"
        assert remove_fn._lifecycle_hook == "remove"
        assert change_fn._lifecycle_hook == "change"
        assert spawn_fn._lifecycle_hook == "spawn"
        assert despawn_fn._lifecycle_hook == "despawn"

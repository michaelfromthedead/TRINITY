"""
Integration tests for Core Foundation systems.
Tests that all 6 systems (Mirror, Serializer, Registry, Tracker, Inspector, Shell)
work together correctly.
"""
import pytest
import sys
import tempfile
import os
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from dataclasses import dataclass
from foundation import (
    mirror, ObjectMirror,
    to_dict, from_dict, deep_copy, register_type,
    registry,
    tracker,
    inspector,
    shell
)
from foundation.inspector import TextUIContext


@pytest.fixture(autouse=True)
def reset_all():
    """Reset all singletons before each test."""
    shell.reset_namespace()
    shell.clear_history()
    # Reset tracker internal state
    tracker._dirty.clear()
    tracker._undo.clear()
    tracker._redo.clear()
    tracker._txn = None
    yield


class TestMirrorSerializerIntegration:
    """Mirror + Serializer working together."""

    def test_serialize_via_mirror_fields(self):
        @dataclass
        class Item:
            name: str
            value: int

        register_type(Item)
        item = Item("sword", 100)

        # Mirror can see the fields
        m = mirror(item)
        assert "name" in m.fields
        assert "value" in m.fields

        # Serializer can save/restore
        data = to_dict(item)
        restored = from_dict(data)
        assert restored.name == "sword"
        assert restored.value == 100

    def test_mirror_fields_match_serialized_data(self):
        """Verify that mirror fields align with serialized output."""
        @dataclass
        class Character:
            name: str
            level: int = 1
            health: float = 100.0

        register_type(Character)
        char = Character(name="Hero", level=5, health=75.0)

        m = mirror(char)
        data = to_dict(char)

        # All mirror fields should appear in serialized data
        for field_name in m.fields:
            assert field_name in data, f"Field {field_name} missing from serialized data"

        # Values should match
        assert data["name"] == m.get("name")
        assert data["level"] == m.get("level")
        assert data["health"] == m.get("health")


class TestRegistryTrackerIntegration:
    """Registry + Tracker working together."""

    def test_track_registered_type_instances(self):
        class Entity:
            def __init__(self):
                self.health = 100

        registry.register(Entity, name="Entity", track_instances=True)

        e = Entity()
        tracker.mark_dirty(e, "health", 100, 50)

        # Tracker sees the change
        assert tracker.is_dirty(e)

        # Registry can find the instance
        instances = list(registry.instances(Entity))
        assert e in instances

        # Cleanup
        registry.unregister(Entity)

    def test_registry_metadata_with_tracker(self):
        """Test that registry metadata works alongside tracking."""
        class TrackedClass:
            def __init__(self):
                self.value = 0

        registry.register(TrackedClass, name="TrackedClass", track_instances=True)
        registry.set_metadata(TrackedClass, "category", "test")

        obj = TrackedClass()
        tracker.mark_dirty(obj, "value", 0, 42)

        # Both systems work
        assert registry.get_metadata(TrackedClass, "category") == "test"
        assert tracker.is_dirty(obj)

        # Cleanup
        registry.unregister(TrackedClass)


class TestInspectorMirrorIntegration:
    """Inspector uses Mirror for reflection."""

    def test_inspector_shows_mirror_fields(self):
        @dataclass
        class Config:
            debug: bool = False
            level: int = 1

        obj = Config()
        panel = inspector.inspect(obj)

        # Mirror sees fields
        m = mirror(obj)
        mirror_fields = set(m.fields.keys())

        # Inspector should show these fields
        ctx = TextUIContext()
        panel.render(ctx)
        output = ctx.get_output()

        # Output should mention the fields
        assert "debug" in output or "level" in output

    def test_inspector_panel_navigation_with_mirror(self):
        """Test inspector navigation updates mirror target."""
        @dataclass
        class Parent:
            name: str
            child: object = None

        @dataclass
        class Child:
            value: int

        parent = Parent(name="parent", child=Child(value=42))
        panel = inspector.inspect(parent)

        # Initially inspecting parent
        m_parent = mirror(panel.target)
        assert m_parent.get("name") == "parent"

        # Navigate to child
        panel.navigate_to(parent.child)
        m_child = mirror(panel.target)
        assert m_child.get("value") == 42


class TestShellAllSystemsIntegration:
    """Shell provides access to all systems."""

    def test_shell_can_use_mirror(self):
        result = shell.execute("mirror({'x': 1}).type_name")
        assert result.success
        assert "dict" in result.value.lower()

    def test_shell_can_use_registry(self):
        shell.execute("class TestClass: pass")
        shell.execute("registry.register(TestClass, name='TestClass')")
        result = shell.execute("registry.is_registered(TestClass)")
        assert result.success
        assert result.value is True

        # Cleanup
        shell.execute("registry.unregister(TestClass)")

    def test_shell_can_use_tracker(self):
        shell.execute("obj = type('Obj', (), {})()")
        shell.execute("tracker.mark_dirty(obj, 'field', None, 1)")
        result = shell.execute("tracker.is_dirty(obj)")
        assert result.success
        assert result.value is True

    def test_shell_can_use_inspector(self):
        result = shell.execute("panel = inspector.inspect({'key': 'value'})")
        assert result.success
        result = shell.execute("panel.target")
        assert result.value == {"key": "value"}

    def test_shell_convenience_functions(self):
        # Test copy()
        result = shell.execute("copy({'a': [1,2,3]})")
        assert result.success
        assert result.value == {"a": [1, 2, 3]}

        # Test types()
        result = shell.execute("types()")
        assert result.success
        assert isinstance(result.value, list)

    def test_shell_result_history(self):
        """Test that _ __ ___ variables track results."""
        shell.execute("1 + 1")  # _ = 2

        # Access _ directly via namespace to avoid changing history
        assert shell.namespace["_"] == 2

        shell.execute("3 + 3")  # _ = 6, __ = 2
        assert shell.namespace["_"] == 6
        assert shell.namespace["__"] == 2

        shell.execute("5 + 5")  # _ = 10, __ = 6, ___ = 2
        assert shell.namespace["_"] == 10
        assert shell.namespace["__"] == 6
        assert shell.namespace["___"] == 2


class TestTrackerUndoWithSerializer:
    """Tracker undo/redo with serializable objects."""

    def test_transaction_with_tracked_changes(self):
        @dataclass
        class Counter:
            value: int = 0

        register_type(Counter)
        c = Counter()

        tracker.begin_transaction("increment")
        old = c.value
        c.value = 10
        tracker.mark_dirty(c, "value", old, c.value)
        tracker.commit_transaction()

        assert tracker.can_undo
        assert len(tracker.undo_stack) == 1

    def test_undo_redo_cycle(self):
        """Test complete undo/redo cycle with tracker."""
        @dataclass
        class Score:
            points: int = 0

        register_type(Score)
        s = Score()

        # Make a change
        tracker.begin_transaction("add points")
        s.points = 100
        tracker.mark_dirty(s, "points", 0, 100)
        tracker.commit_transaction()

        assert s.points == 100
        assert tracker.can_undo
        assert not tracker.can_redo

        # Undo
        tracker.undo()
        assert s.points == 0
        assert tracker.can_redo

        # Redo
        tracker.redo()
        assert s.points == 100


class TestFullPipeline:
    """Complete workflow using all systems."""

    def test_create_inspect_serialize_track(self):
        # 1. Create a class and register it
        @dataclass
        class Player:
            name: str
            score: int = 0

        register_type(Player)
        registry.register(Player, name="Player")

        # 2. Create instance
        player = Player(name="Alice", score=100)

        # 3. Mirror it
        m = mirror(player)
        assert m.get("name") == "Alice"
        assert m.get("score") == 100

        # 4. Inspect it
        panel = inspector.inspect(player)
        assert panel.target is player

        # 5. Serialize it
        data = to_dict(player)
        assert data["name"] == "Alice"

        # 6. Track changes
        tracker.begin_transaction("update score")
        old_score = player.score
        player.score = 200
        tracker.mark_dirty(player, "score", old_score, 200)
        tracker.commit_transaction()

        assert tracker.is_dirty(player)

        # 7. Use shell to query
        shell.namespace["player"] = player
        result = shell.execute("player.score")
        assert result.value == 200

        # 8. Restore from serialization
        restored = from_dict(data)
        assert restored.score == 100  # Original value

        # Cleanup
        registry.unregister(Player)

    def test_end_to_end_game_entity_workflow(self):
        """Simulate a realistic game entity workflow."""
        # Use regular class (not dataclass) for instance tracking
        # because dataclasses are unhashable by default and can't be in WeakSet
        class GameEntity:
            def __init__(self, entity_id: str, x: float = 0.0, y: float = 0.0, hp: int = 100):
                self.entity_id = entity_id
                self.x = x
                self.y = y
                self.hp = hp

        register_type(GameEntity)
        registry.register(GameEntity, name="GameEntity", track_instances=True)

        # Create entities
        player = GameEntity(entity_id="player_1", x=10.0, y=20.0, hp=100)
        enemy = GameEntity(entity_id="enemy_1", x=50.0, y=30.0, hp=50)

        # Track instances via registry
        instances = list(registry.instances(GameEntity))
        assert len(instances) == 2

        # Serialize game state
        state = {
            "player": to_dict(player),
            "enemy": to_dict(enemy)
        }

        # Simulate combat with tracking
        tracker.begin_transaction("combat")

        old_hp = player.hp
        player.hp = 80
        tracker.mark_dirty(player, "hp", old_hp, player.hp)

        old_hp = enemy.hp
        enemy.hp = 0
        tracker.mark_dirty(enemy, "hp", old_hp, enemy.hp)

        tracker.commit_transaction()

        # Verify changes tracked
        assert tracker.is_dirty(player)
        assert tracker.is_dirty(enemy)

        # Undo combat
        tracker.undo()
        assert player.hp == 100
        assert enemy.hp == 50

        # Restore from saved state
        restored_player = from_dict(state["player"])
        assert restored_player.x == 10.0
        assert restored_player.hp == 100

        # Cleanup
        registry.unregister(GameEntity)


class TestCrossSystemConsistency:
    """Verify systems maintain consistent views."""

    def test_mirror_and_serializer_agree(self):
        @dataclass
        class Box:
            value: int = 42

        register_type(Box)
        obj = Box()

        # Both should see same fields
        m = mirror(obj)
        d = to_dict(obj)

        for field_name in m.fields:
            if not field_name.startswith("_"):
                assert field_name in d or d.get(field_name) is not None

    def test_shell_namespace_consistency(self):
        """Test that shell namespace stays consistent with singletons."""
        # Verify shell has access to same instances
        result = shell.execute("id(registry)")
        shell_registry_id = result.value

        result = shell.execute("id(tracker)")
        shell_tracker_id = result.value

        result = shell.execute("id(inspector)")
        shell_inspector_id = result.value

        # These should be the same singleton instances
        assert shell_registry_id == id(registry)
        assert shell_tracker_id == id(tracker)
        assert shell_inspector_id == id(inspector)


class TestErrorHandling:
    """Systems handle errors gracefully."""

    def test_shell_catches_errors(self):
        result = shell.execute("undefined_variable")
        assert result.success is False
        assert "NameError" in result.error_type

    def test_inspector_handles_any_object(self):
        # Should not crash on any object
        panel = inspector.inspect(None)
        assert panel is not None

        panel = inspector.inspect(42)
        assert panel is not None

        panel = inspector.inspect(lambda x: x)
        assert panel is not None

    def test_serializer_error_on_unknown_type(self):
        """Test that deserializing unknown type raises proper error."""
        class UnknownClass:
            pass

        data = {"__type__": "NonExistent.Module.Class", "__id__": 1}

        with pytest.raises(TypeError) as exc_info:
            from_dict(data)

        assert "Unknown type" in str(exc_info.value)

    def test_tracker_transaction_errors(self):
        """Test tracker transaction error handling."""
        # Can't commit without begin
        with pytest.raises(RuntimeError):
            tracker.commit_transaction()

        # Can't rollback without begin
        with pytest.raises(RuntimeError):
            tracker.rollback_transaction()

        # Nested transactions not allowed
        tracker.begin_transaction("first")
        with pytest.raises(RuntimeError):
            tracker.begin_transaction("second")

        tracker.rollback_transaction()


class TestDeepCopyIntegration:
    """Test deep copy with complex objects across systems."""

    def test_deep_copy_preserves_structure(self):
        @dataclass
        class Node:
            name: str
            children: list = None

            def __post_init__(self):
                if self.children is None:
                    self.children = []

        register_type(Node)

        root = Node("root")
        child1 = Node("child1")
        child2 = Node("child2")
        root.children = [child1, child2]

        # Deep copy
        copied = deep_copy(root)

        # Structure preserved
        assert copied.name == "root"
        assert len(copied.children) == 2
        assert copied.children[0].name == "child1"

        # But it's a real copy
        assert copied is not root
        assert copied.children is not root.children
        assert copied.children[0] is not child1


class TestTrackerCallbackIntegration:
    """Test tracker callbacks with other systems."""

    def test_change_callback_fires(self):
        changes_received = []

        def on_change(obj, field, old_val, new_val):
            changes_received.append((field, old_val, new_val))

        tracker.on_change(callback=on_change)

        @dataclass
        class Observable:
            value: int = 0

        obj = Observable()
        tracker.mark_dirty(obj, "value", 0, 42)

        assert len(changes_received) == 1
        assert changes_received[0] == ("value", 0, 42)

        # Cleanup
        tracker.off_change(on_change)


class TestInspectorViewIntegration:
    """Test inspector views work with different object types."""

    def test_collection_view_for_lists(self):
        data = [1, 2, 3, "four"]
        panel = inspector.inspect(data)

        # Should use CollectionView
        views = panel.views
        view_names = [v.name for v in views]
        assert "Collection" in view_names

        # Render it
        ctx = TextUIContext()
        panel.render(ctx)
        output = ctx.get_output()

        assert "list" in output.lower()
        assert "4 items" in output.lower()

    def test_collection_view_for_dicts(self):
        data = {"key1": "value1", "key2": 42}
        panel = inspector.inspect(data)

        ctx = TextUIContext()
        panel.render(ctx)
        output = ctx.get_output()

        assert "dict" in output.lower()
        assert "key1" in output or "key2" in output


class TestShellBindingIntegration:
    """Test shell object binding with other systems."""

    def test_bind_object_and_use(self):
        @dataclass
        class Target:
            x: int = 10
            y: int = 20

        obj = Target()
        shell.bind(obj)

        result = shell.execute("self.x + self.y")
        assert result.success
        assert result.value == 30

        shell.unbind()

        result = shell.execute("self")
        assert result.success is False or result.value is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

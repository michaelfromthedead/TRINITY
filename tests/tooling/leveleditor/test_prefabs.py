"""
Tests for the prefabs module.

Tests nesting, overrides, and instantiation.
"""

import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.leveleditor.prefabs import (
    PrefabAsset,
    PrefabInstance,
    PrefabOverride,
    PrefabVariant,
    PrefabManager,
    PrefabComponent,
    PrefabChild,
    OverrideType,
    PrefabState,
)
from engine.tooling.leveleditor.placement import Vector3, Transform
from foundation.tracker import tracker


@pytest.fixture(autouse=True)
def reset_tracker():
    """Reset tracker state before each test."""
    tracker._dirty.clear()
    tracker._cb_global.clear()
    tracker._cb_type.clear()
    tracker._cb_obj.clear()
    tracker._undo.clear()
    tracker._redo.clear()
    tracker._txn = None
    yield


class TestPrefabAsset:
    """Tests for PrefabAsset class."""

    def test_creation(self):
        """Asset should initialize with name and root."""
        asset = PrefabAsset("TestPrefab")
        assert asset.name == "TestPrefab"
        assert asset.root is not None
        assert asset.version == 1

    def test_unique_id(self):
        """Each asset should have unique ID."""
        asset1 = PrefabAsset("A")
        asset2 = PrefabAsset("B")
        assert asset1.id != asset2.id

    def test_rename(self):
        """Should rename asset."""
        asset = PrefabAsset("Original")
        asset.name = "Renamed"
        assert asset.name == "Renamed"
        assert asset.root.name == "Renamed"

    def test_add_component(self):
        """Should add component to root."""
        asset = PrefabAsset("Test")
        comp = asset.add_component("Transform", {"x": 0, "y": 0})

        assert comp is not None
        assert comp.component_type == "Transform"
        assert len(asset.root.components) == 1

    def test_add_component_to_child(self):
        """Should add component to child node."""
        asset = PrefabAsset("Test")
        child = asset.add_child("Child")
        comp = asset.add_component("Renderer", target_path="Child")

        assert len(child.components) == 1

    def test_remove_component(self):
        """Should remove component."""
        asset = PrefabAsset("Test")
        comp = asset.add_component("Transform")

        result = asset.remove_component(comp.component_id)

        assert result is True
        assert len(asset.root.components) == 0

    def test_add_child(self):
        """Should add child to root."""
        asset = PrefabAsset("Test")
        child = asset.add_child("Child")

        assert child is not None
        assert child.name == "Child"
        assert len(asset.root.children) == 1

    def test_add_nested_child(self):
        """Should add nested children."""
        asset = PrefabAsset("Test")
        child = asset.add_child("Child")
        grandchild = asset.add_child("Grandchild", parent_path="Child")

        assert len(child.children) == 1

    def test_remove_child(self):
        """Should remove child."""
        asset = PrefabAsset("Test")
        child = asset.add_child("Child")

        result = asset.remove_child(child.child_id)

        assert result is True
        assert len(asset.root.children) == 0

    def test_set_nested_prefab(self):
        """Should set nested prefab reference."""
        asset = PrefabAsset("Parent")
        child = asset.add_child("NestedSlot")

        result = asset.set_nested_prefab("NestedSlot", "nested-prefab-id")

        assert result is True
        assert child.nested_prefab_id == "nested-prefab-id"

    def test_get_all_nested_prefab_ids(self):
        """Should collect all nested prefab IDs."""
        asset = PrefabAsset("Parent")
        child1 = asset.add_child("Slot1")
        child2 = asset.add_child("Slot2")

        asset.set_nested_prefab("Slot1", "prefab-1")
        asset.set_nested_prefab("Slot2", "prefab-2")

        ids = asset.get_all_nested_prefab_ids()
        assert "prefab-1" in ids
        assert "prefab-2" in ids

    def test_tags(self):
        """Should manage tags."""
        asset = PrefabAsset("Test")
        asset.add_tag("enemy")
        asset.add_tag("dangerous")

        assert "enemy" in asset.tags
        assert "dangerous" in asset.tags

        asset.remove_tag("enemy")
        assert "enemy" not in asset.tags

    def test_category(self):
        """Should store category."""
        asset = PrefabAsset("Test")
        asset.category = "Enemies"

        assert asset.category == "Enemies"

    def test_clone(self):
        """Should create deep copy."""
        asset = PrefabAsset("Original")
        asset.add_component("Transform")
        asset.add_child("Child")

        clone = asset.clone()

        assert clone.id != asset.id
        assert "Copy" in clone.name
        assert len(clone.root.components) == 1
        assert len(clone.root.children) == 1

    def test_version_increments(self):
        """Version should increment on changes."""
        asset = PrefabAsset("Test")
        initial = asset.version

        asset.add_child("Child")

        assert asset.version > initial


class TestPrefabInstance:
    """Tests for PrefabInstance class."""

    def test_creation(self):
        """Instance should initialize with asset reference."""
        instance = PrefabInstance("asset-id")
        assert instance.prefab_asset_id == "asset-id"
        assert instance.state == PrefabState.SYNCHRONIZED

    def test_unique_id(self):
        """Each instance should have unique ID."""
        inst1 = PrefabInstance("asset")
        inst2 = PrefabInstance("asset")
        assert inst1.id != inst2.id

    def test_transform(self):
        """Should store instance transform."""
        instance = PrefabInstance("asset")
        instance.transform = Transform(position=Vector3(10, 20, 30))

        assert instance.transform.position.x == 10

    def test_add_override(self):
        """Should add property override."""
        instance = PrefabInstance("asset")
        override = instance.add_override(
            target_path="root",
            property_name="scale",
            override_value=2.0,
            original_value=1.0
        )

        assert override is not None
        assert instance.has_overrides is True
        assert instance.state == PrefabState.MODIFIED

    def test_override_replaces_existing(self):
        """Adding same override should update existing."""
        instance = PrefabInstance("asset")
        instance.add_override("root", "scale", 2.0)
        instance.add_override("root", "scale", 3.0)

        assert len(instance.overrides) == 1
        assert instance.overrides[0].override_value == 3.0

    def test_remove_override(self):
        """Should remove override."""
        instance = PrefabInstance("asset")
        override = instance.add_override("root", "scale", 2.0)

        result = instance.remove_override(override.override_id)

        assert result is True
        assert instance.has_overrides is False
        assert instance.state == PrefabState.SYNCHRONIZED

    def test_clear_overrides(self):
        """Should clear all overrides."""
        instance = PrefabInstance("asset")
        instance.add_override("root", "scale", 2.0)
        instance.add_override("root", "position", Vector3(1, 2, 3))

        count = instance.clear_overrides()

        assert count == 2
        assert instance.has_overrides is False

    def test_get_override(self):
        """Should get specific override."""
        instance = PrefabInstance("asset")
        instance.add_override("root", "scale", 2.0)

        override = instance.get_override("root", "scale")

        assert override is not None
        assert override.override_value == 2.0

    def test_get_override_not_found(self):
        """Should return None if override not found."""
        instance = PrefabInstance("asset")

        override = instance.get_override("root", "missing")

        assert override is None

    def test_has_override(self):
        """Should check for override existence."""
        instance = PrefabInstance("asset")
        instance.add_override("root", "scale", 2.0)

        assert instance.has_override("root", "scale") is True
        assert instance.has_override("root", "other") is False

    def test_get_effective_value(self):
        """Should return override value if exists."""
        instance = PrefabInstance("asset")
        instance.add_override("root", "scale", 2.0)

        value = instance.get_effective_value("root", "scale", 1.0)
        assert value == 2.0

        value = instance.get_effective_value("root", "other", 1.0)
        assert value == 1.0

    def test_check_version(self):
        """Should check version synchronization."""
        instance = PrefabInstance("asset")
        instance.update_version(1)

        assert instance.check_version(1) is True
        assert instance.check_version(2) is False

    def test_callback_on_override_change(self):
        """Should trigger callback on override changes."""
        instance = PrefabInstance("asset")
        changes = []

        def callback(override, action):
            changes.append((override, action))

        instance.on("on_override_change", callback)
        override = instance.add_override("root", "scale", 2.0)

        assert len(changes) == 1
        assert changes[0][1] == "add"

    def test_callback_on_state_change(self):
        """Should trigger callback on state changes."""
        instance = PrefabInstance("asset")
        states = []

        def callback(old, new):
            states.append((old, new))

        instance.on("on_state_change", callback)
        instance.add_override("root", "scale", 2.0)

        assert len(states) == 1
        assert states[0] == (PrefabState.SYNCHRONIZED, PrefabState.MODIFIED)


class TestPrefabVariant:
    """Tests for PrefabVariant class."""

    def test_creation(self):
        """Variant should initialize with parent reference."""
        variant = PrefabVariant("ChildPrefab", "parent-id")
        assert variant.parent_id == "parent-id"

    def test_add_variant_override(self):
        """Should add variant-level override."""
        variant = PrefabVariant("Child", "parent")
        override = variant.add_variant_override("root", "scale", 2.0)

        assert override is not None
        assert len(variant.variant_overrides) == 1


class TestPrefabManager:
    """Tests for PrefabManager class."""

    def test_creation(self):
        """Manager should initialize empty."""
        manager = PrefabManager()
        assert len(manager.get_all_assets()) == 0

    def test_register_asset(self):
        """Should register prefab asset."""
        manager = PrefabManager()
        asset = PrefabAsset("Test")

        manager.register_asset(asset)

        assert manager.get_asset(asset.id) is asset

    def test_unregister_asset(self):
        """Should unregister prefab asset."""
        manager = PrefabManager()
        asset = PrefabAsset("Test")
        manager.register_asset(asset)

        result = manager.unregister_asset(asset.id)

        assert result is True
        assert manager.get_asset(asset.id) is None

    def test_unregister_disconnects_instances(self):
        """Unregistering asset should disconnect instances."""
        manager = PrefabManager()
        asset = PrefabAsset("Test")
        manager.register_asset(asset)
        instance = manager.instantiate(asset.id)

        manager.unregister_asset(asset.id)

        assert instance.state == PrefabState.DISCONNECTED

    def test_instantiate(self):
        """Should create instance of prefab."""
        manager = PrefabManager()
        asset = PrefabAsset("Test")
        manager.register_asset(asset)

        instance = manager.instantiate(asset.id)

        assert instance is not None
        assert instance.prefab_asset_id == asset.id

    def test_instantiate_with_transform(self):
        """Should create instance with transform."""
        manager = PrefabManager()
        asset = PrefabAsset("Test")
        manager.register_asset(asset)

        transform = Transform(position=Vector3(10, 0, 0))
        instance = manager.instantiate(asset.id, transform)

        assert instance.transform.position.x == 10

    def test_instantiate_nonexistent(self):
        """Should return None for nonexistent asset."""
        manager = PrefabManager()
        instance = manager.instantiate("nonexistent")

        assert instance is None

    def test_destroy_instance(self):
        """Should destroy instance."""
        manager = PrefabManager()
        asset = PrefabAsset("Test")
        manager.register_asset(asset)
        instance = manager.instantiate(asset.id)

        result = manager.destroy_instance(instance.id)

        assert result is True
        assert manager.get_instance(instance.id) is None

    def test_get_instances_of_asset(self):
        """Should get all instances of an asset."""
        manager = PrefabManager()
        asset = PrefabAsset("Test")
        manager.register_asset(asset)

        manager.instantiate(asset.id)
        manager.instantiate(asset.id)

        instances = manager.get_instances_of_asset(asset.id)
        assert len(instances) == 2

    def test_find_assets_by_name(self):
        """Should find assets by name."""
        manager = PrefabManager()
        manager.register_asset(PrefabAsset("Enemy"))
        manager.register_asset(PrefabAsset("Enemy_Fast"))
        manager.register_asset(PrefabAsset("Player"))

        found = manager.find_assets_by_name("enemy")
        assert len(found) == 2

    def test_find_assets_by_tag(self):
        """Should find assets by tag."""
        manager = PrefabManager()
        asset1 = PrefabAsset("Test1")
        asset1.add_tag("enemy")
        asset2 = PrefabAsset("Test2")
        asset2.add_tag("enemy")
        asset3 = PrefabAsset("Test3")
        asset3.add_tag("player")

        manager.register_asset(asset1)
        manager.register_asset(asset2)
        manager.register_asset(asset3)

        found = manager.find_assets_by_tag("enemy")
        assert len(found) == 2

    def test_find_assets_by_category(self):
        """Should find assets by category."""
        manager = PrefabManager()
        asset1 = PrefabAsset("Test1")
        asset1.category = "Enemies"
        asset2 = PrefabAsset("Test2")
        asset2.category = "Players"

        manager.register_asset(asset1)
        manager.register_asset(asset2)

        found = manager.find_assets_by_category("Enemies")
        assert len(found) == 1

    def test_apply_instance_to_asset(self):
        """Should apply instance overrides to asset."""
        manager = PrefabManager()
        asset = PrefabAsset("Test")
        manager.register_asset(asset)
        instance = manager.instantiate(asset.id)
        instance.add_override("root", "scale", 2.0)

        result = manager.apply_instance_to_asset(instance.id)

        assert result is True
        assert instance.has_overrides is False

    def test_revert_instance(self):
        """Should revert instance to asset state."""
        manager = PrefabManager()
        asset = PrefabAsset("Test")
        manager.register_asset(asset)
        instance = manager.instantiate(asset.id)
        instance.add_override("root", "scale", 2.0)

        result = manager.revert_instance(instance.id)

        assert result is True
        assert instance.has_overrides is False

    def test_create_variant(self):
        """Should create variant of prefab."""
        manager = PrefabManager()
        asset = PrefabAsset("Parent")
        manager.register_asset(asset)

        variant = manager.create_variant(asset.id, "Child")

        assert variant is not None
        assert variant.parent_id == asset.id
        assert manager.get_asset(variant.id) is variant

    def test_check_circular_reference(self):
        """Should detect circular references."""
        manager = PrefabManager()
        asset1 = PrefabAsset("A")
        asset2 = PrefabAsset("B")

        manager.register_asset(asset1)
        manager.register_asset(asset2)

        # A contains B
        child = asset1.add_child("Nested")
        asset1.set_nested_prefab("Nested", asset2.id)

        # Now trying to add A to B would create cycle
        assert manager.check_circular_reference(asset2.id, asset1.id) is True

    def test_check_circular_self_reference(self):
        """Should detect self-reference."""
        manager = PrefabManager()
        asset = PrefabAsset("A")
        manager.register_asset(asset)

        assert manager.check_circular_reference(asset.id, asset.id) is True

    def test_callbacks(self):
        """Should trigger callbacks on changes."""
        manager = PrefabManager()
        events = []

        manager.on("on_asset_add", lambda a: events.append(("add", a)))
        manager.on("on_asset_remove", lambda a: events.append(("remove", a)))
        manager.on("on_instance_create", lambda i: events.append(("create", i)))
        manager.on("on_instance_destroy", lambda i: events.append(("destroy", i)))

        asset = PrefabAsset("Test")
        manager.register_asset(asset)
        instance = manager.instantiate(asset.id)
        manager.destroy_instance(instance.id)
        manager.unregister_asset(asset.id)

        assert len([e for e in events if e[0] == "add"]) == 1
        assert len([e for e in events if e[0] == "create"]) == 1
        assert len([e for e in events if e[0] == "destroy"]) == 1
        assert len([e for e in events if e[0] == "remove"]) == 1

    def test_get_statistics(self):
        """Should return statistics."""
        manager = PrefabManager()
        asset = PrefabAsset("Test")
        manager.register_asset(asset)
        instance = manager.instantiate(asset.id)
        instance.add_override("root", "scale", 2.0)

        stats = manager.get_statistics()

        assert stats["total_assets"] == 1
        assert stats["total_instances"] == 1
        assert stats["modified_instances"] == 1

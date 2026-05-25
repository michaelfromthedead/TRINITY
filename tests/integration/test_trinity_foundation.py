"""
Trinity-Foundation Integration Tests

Tests the integration between Trinity's metaclass/descriptor system and
Foundation's registry/tracker services.

Verifies:
1. ComponentMeta auto-registers with foundation.registry
2. TrackedDescriptor notifies foundation.tracker
3. Bridge functions create integrated worlds
4. TrinityWorldAdapter syncs instances correctly
"""

import pytest

from trinity import Component, ComponentMeta
from trinity.descriptors.tracking import clear_dirty, get_dirty_fields
from foundation import registry, tracker
from foundation.bridge import (
    get_trinity_registry,
    create_world_from_trinity,
    create_ai_interface,
    create_shell,
    TrinityWorldAdapter,
)


class TestComponentMetaRegistryIntegration:
    """Test that ComponentMeta automatically registers with foundation.registry."""

    def test_component_registered_with_foundation(self, component_meta):
        """New components are auto-registered in foundation.registry."""

        class AutoRegistered(Component):
            x: float = 0.0
            y: float = 0.0

        # Should be registered in foundation.registry
        assert registry.is_registered(AutoRegistered)

    def test_component_name_matches(self, component_meta):
        """Registered component name matches the qualified name."""

        class Named(Component):
            value: int = 0

        # Check it's registered with the component name
        expected_name = Named._component_name
        actual_name = registry.get_name(Named)
        assert actual_name is not None
        assert actual_name == expected_name

    def test_component_instances_tracked(self, component_meta):
        """Component instances are tracked when track_instances=True."""

        class Trackable(Component):
            data: str = ""

        # Create instances
        inst1 = Trackable(data="first")
        inst2 = Trackable(data="second")

        # Instances should be tracked (WeakSet, so alive instances are visible)
        instances = list(registry.instances(Trackable))
        assert len(instances) >= 2  # At least our two
        assert inst1 in instances
        assert inst2 in instances


class TestTrackedDescriptorTrackerIntegration:
    """Test that TrackedDescriptor notifies foundation.tracker on changes."""

    def test_changes_recorded_in_tracker(self, component_meta):
        """Field changes are recorded in foundation.tracker."""

        class Monitored(Component):
            _track_changes = True
            health: float = 100.0

        obj = Monitored()

        # Modify value - this triggers TrackedDescriptor.post_set
        # which calls tracker.mark_dirty()
        obj.health = 50.0

        # Verify tracker received the change
        assert tracker.is_dirty(obj), "Tracker should mark object as dirty"
        assert obj.health == 50.0

    def test_tracker_undo_reverts_change(self, component_meta):
        """Tracker can undo changes made to tracked components."""

        class Undoable(Component):
            _track_changes = True
            value: int = 0

        obj = Undoable()
        obj.value = 10  # First change
        obj.value = 20  # Second change
        assert obj.value == 20

        # Undo second change (20 → 10)
        tracker.undo()
        assert obj.value == 10, "First undo should revert 20 to 10"

        # Undo first change (10 → None, pre-storage state)
        # Note: Tracker reverts to pre-set state, which is None (no stored value),
        # not the default. This is correct behavior - defaults are a descriptor
        # concern, not a tracker concern.
        tracker.undo()
        assert obj.value is None, "Second undo should revert to pre-storage state"

    def test_dirty_and_tracker_both_updated(self, component_meta):
        """Both local dirty flags and tracker are updated on change."""

        class DualTracked(Component):
            _track_changes = True
            position: float = 0.0

        obj = DualTracked()
        clear_dirty(obj)  # Clear local Trinity dirty flags

        obj.position = 42.0

        # Local dirty tracking (Trinity)
        assert "position" in get_dirty_fields(obj)

        # Foundation tracker also marks dirty
        assert tracker.is_dirty(obj), "Tracker should mark object as dirty"

        # Value was set correctly
        assert obj.position == 42.0


class TestBridgeFunctions:
    """Test the bridge module functions."""

    def test_get_trinity_registry_returns_components(self, component_meta):
        """get_trinity_registry returns all registered Trinity components."""

        class BridgeTestA(Component):
            a: int = 0

        class BridgeTestB(Component):
            b: int = 0

        reg = get_trinity_registry()

        # Should include our components
        assert "BridgeTestA" in reg
        assert "BridgeTestB" in reg
        assert reg["BridgeTestA"] is BridgeTestA
        assert reg["BridgeTestB"] is BridgeTestB

    def test_create_world_from_trinity(self, component_meta):
        """create_world_from_trinity creates a World with registered components."""

        class WorldComp(Component):
            value: float = 0.0

        world = create_world_from_trinity()

        # World should exist
        assert world is not None

        # Should be able to create entities and attach components
        entity = world.create()
        comp = WorldComp(value=42.0)
        world.attach(entity, comp)

        # Retrieve it
        retrieved = world.get(entity, WorldComp)
        assert retrieved is comp
        assert retrieved.value == 42.0

    def test_create_ai_interface_works(self, component_meta):
        """create_ai_interface creates a working AIInterface."""

        class AITestComp(Component):
            name: str = ""

        ai = create_ai_interface()

        # Should be able to list types
        result = ai.execute({"op": "list_types"})
        assert "types" in result
        assert "AITestComp" in result["types"]

        # Should be able to spawn entities with components
        spawn_result = ai.execute({
            "op": "spawn",
            "component": "AITestComp",
            "fields": {"name": "test"}
        })
        assert "entity" in spawn_result
        entity_id = spawn_result["entity"]

        # Should be able to inspect
        inspect_result = ai.execute({
            "op": "inspect",
            "entity": entity_id
        })
        assert "AITestComp" in inspect_result["components"]
        assert inspect_result["components"]["AITestComp"]["name"] == "test"

    def test_create_shell_works(self, component_meta):
        """create_shell creates a working Shell."""

        class ShellTestComp(Component):
            value: int = 0

        shell = create_shell()

        # Shell should have access to registry
        assert shell is not None
        assert "ShellTestComp" in shell._registry


class TestTrinityWorldAdapter:
    """Test the TrinityWorldAdapter class."""

    def test_add_instance_creates_entity(self, component_meta):
        """Adding a Trinity instance creates a corresponding entity."""

        class AdaptedComp(Component):
            x: float = 0.0

        adapter = TrinityWorldAdapter()
        instance = AdaptedComp(x=10.0)

        entity = adapter.add_instance(instance)

        # Entity should exist
        assert entity is not None
        assert adapter.world.exists(entity)

    def test_get_entity_returns_same_entity(self, component_meta):
        """get_entity returns the entity for an instance."""

        class GetEntityComp(Component):
            val: int = 0

        adapter = TrinityWorldAdapter()
        instance = GetEntityComp(val=5)

        entity = adapter.add_instance(instance)
        retrieved = adapter.get_entity(instance)

        assert retrieved is entity

    def test_add_same_instance_twice_returns_same_entity(self, component_meta):
        """Adding the same instance twice returns the same entity."""

        class DoubleAdd(Component):
            data: str = ""

        adapter = TrinityWorldAdapter()
        instance = DoubleAdd(data="test")

        entity1 = adapter.add_instance(instance)
        entity2 = adapter.add_instance(instance)

        assert entity1.id == entity2.id

    def test_get_instance_from_entity(self, component_meta):
        """get_instance retrieves the Trinity instance from an entity."""

        class RetrievableComp(Component):
            value: float = 0.0

        adapter = TrinityWorldAdapter()
        instance = RetrievableComp(value=99.0)

        entity = adapter.add_instance(instance)
        retrieved = adapter.get_instance(entity, RetrievableComp)

        assert retrieved is instance
        assert retrieved.value == 99.0

    def test_remove_instance(self, component_meta):
        """remove_instance removes the instance from tracking."""

        class RemovableComp(Component):
            x: int = 0

        adapter = TrinityWorldAdapter()
        instance = RemovableComp(x=1)

        entity = adapter.add_instance(instance)
        assert adapter.get_entity(instance) is not None

        adapter.remove_instance(instance)
        assert adapter.get_entity(instance) is None

    def test_all_instances_iterator(self, component_meta):
        """all_instances yields all tracked instances of a type."""

        class IterableComp(Component):
            id: int = 0

        adapter = TrinityWorldAdapter()

        inst1 = IterableComp(id=1)
        inst2 = IterableComp(id=2)
        inst3 = IterableComp(id=3)

        adapter.add_instance(inst1)
        adapter.add_instance(inst2)
        adapter.add_instance(inst3)

        instances = list(adapter.all_instances(IterableComp))

        assert len(instances) == 3
        assert inst1 in instances
        assert inst2 in instances
        assert inst3 in instances

    def test_sync_from_foundation_registry(self, component_meta):
        """sync_from_foundation_registry imports instances from foundation registry."""

        class SyncableComp(Component):
            name: str = ""

        # Create instances (they auto-register with foundation.registry)
        inst1 = SyncableComp(name="first")
        inst2 = SyncableComp(name="second")

        # Create adapter and sync
        adapter = TrinityWorldAdapter()
        adapter.sync_from_foundation_registry()

        # Instances should now be tracked in adapter
        tracked = list(adapter.all_instances(SyncableComp))
        assert inst1 in tracked
        assert inst2 in tracked


class TestEndToEndIntegration:
    """Full end-to-end integration tests."""

    def test_full_trinity_foundation_flow(self, component_meta):
        """
        Complete flow:
        1. Define Trinity component
        2. Verify auto-registration
        3. Create instances
        4. Use bridge to create world
        5. Query via AI interface
        6. Modify via AI interface
        7. Verify changes tracked
        """
        # 1. Define component
        class Player(Component):
            _track_changes = True
            name: str = ""
            score: int = 0
            health: float = 100.0

        # 2. Verify registration
        assert registry.is_registered(Player)

        # 3. Create AI interface (which creates world)
        ai = create_ai_interface()

        # 4. Spawn player via AI
        result = ai.execute({
            "op": "spawn",
            "component": "Player",
            "fields": {"name": "Hero", "score": 0, "health": 100.0}
        })
        player_id = result["entity"]

        # 5. Query to verify
        query_result = ai.execute({
            "op": "query",
            "components": ["Player"]
        })
        assert query_result["count"] >= 1

        # 6. Modify via AI
        ai.execute({
            "op": "set",
            "entity": player_id,
            "component": "Player",
            "field": "score",
            "value": 100
        })

        # 7. Verify change
        inspect_result = ai.execute({
            "op": "inspect",
            "entity": player_id
        })
        assert inspect_result["components"]["Player"]["score"] == 100

    def test_adapter_with_tracked_components(self, component_meta):
        """
        Test adapter with tracked components:
        1. Create tracked component
        2. Add to adapter
        3. Modify via instance
        4. Verify dirty state
        5. Query via adapter's world
        """
        class TrackedEntity(Component):
            _track_changes = True
            x: float = 0.0
            y: float = 0.0
            active: bool = True

        adapter = TrinityWorldAdapter()

        # Create and add instance
        entity_inst = TrackedEntity(x=10.0, y=20.0)
        entity = adapter.add_instance(entity_inst)
        clear_dirty(entity_inst)

        # Modify instance
        entity_inst.x = 15.0

        # Verify dirty
        assert "x" in get_dirty_fields(entity_inst)
        assert "y" not in get_dirty_fields(entity_inst)

        # Verify retrievable via adapter
        retrieved = adapter.get_instance(entity, TrackedEntity)
        assert retrieved.x == 15.0
        assert retrieved.y == 20.0

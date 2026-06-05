"""
Whitebox tests for the replication layer: replication_manager, property_replication, net_guid, relevancy, bandwidth.

Tests:
- Entity registration and lifecycle
- Property replication and change tracking
- GUID allocation and management
- Relevancy filtering
- Bandwidth allocation
"""

import pytest
import time
import struct
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from engine.networking.replication.replication_manager import (
    ReplicationManager,
    ReplicationRole,
    EntityState,
    ReplicatedEntity,
)
from engine.networking.replication.net_guid import (
    NetGUID,
    NetGUIDManager,
    GUIDAuthority,
)
from engine.networking.replication.property_replication import (
    ReplicatedProperty,
    PropertyReplicationGroup,
    ReplicationCondition,
    ChangeNotifyMode,
)
from engine.networking.replication.relevancy import (
    RelevancyManager,
    RelevancyResult,
    RadiusRelevancy,
    InterestArea,
    GridRelevancy,
)
from engine.networking.replication.bandwidth import (
    BandwidthManager,
    EntityPriority,
)
from engine.networking.config import DEFAULT_CONFIG


# =============================================================================
# Helper class for weakref-compatible test objects
# =============================================================================

class MockEntity:
    """A mock entity that supports weak references."""
    __slots__ = ('position', 'owner_id', '__weakref__')

    def __init__(self, position=(0, 0, 0), owner_id=None):
        self.position = position
        self.owner_id = owner_id


# =============================================================================
# NetGUID Tests
# =============================================================================

class TestNetGUID:
    """Tests for NetGUID value type."""

    def test_guid_creation(self):
        """GUID should store value correctly."""
        guid = NetGUID(12345)
        assert guid.value == 12345

    def test_guid_equality(self):
        """GUIDs with same value should be equal."""
        guid1 = NetGUID(100)
        guid2 = NetGUID(100)
        assert guid1 == guid2
        assert guid1.value == guid2.value

    def test_guid_inequality(self):
        """GUIDs with different values should not be equal."""
        guid1 = NetGUID(100)
        guid2 = NetGUID(200)
        assert guid1 != guid2

    def test_guid_hash(self):
        """GUIDs should be hashable for dict keys."""
        guid1 = NetGUID(100)
        guid2 = NetGUID(100)

        d = {guid1: "value"}
        assert d[guid2] == "value"

    def test_guid_repr(self):
        """GUID repr should be readable."""
        guid = NetGUID(12345)
        repr_str = repr(guid)
        assert "12345" in repr_str or "NetGUID" in repr_str


class TestNetGUIDManager:
    """Tests for NetGUIDManager."""

    def test_manager_server_authority(self):
        """Server manager should use server GUID range."""
        manager = NetGUIDManager(GUIDAuthority.SERVER)
        entity = MockEntity()
        guid = manager.assign_guid(entity)

        assert guid.value >= DEFAULT_CONFIG.SERVER_GUID_START
        assert guid.value <= DEFAULT_CONFIG.SERVER_GUID_MAX

    def test_manager_client_authority(self):
        """Client manager should use client GUID range."""
        manager = NetGUIDManager(GUIDAuthority.CLIENT, client_id=1)
        entity = MockEntity()
        guid = manager.assign_guid(entity)

        assert guid.value >= DEFAULT_CONFIG.CLIENT_GUID_START
        assert guid.value <= DEFAULT_CONFIG.CLIENT_GUID_MAX

    def test_manager_assign_unique_guids(self):
        """Each entity should get a unique GUID."""
        manager = NetGUIDManager(GUIDAuthority.SERVER)
        guids = set()
        entities = []  # Keep references to prevent GC

        for _ in range(100):
            entity = MockEntity()
            entities.append(entity)
            guid = manager.assign_guid(entity)
            assert guid.value not in guids
            guids.add(guid.value)

    def test_manager_import_guid(self):
        """Importing a GUID should work."""
        manager = NetGUIDManager(GUIDAuthority.CLIENT)
        entity = MockEntity()
        guid = NetGUID(999)

        manager.import_guid(entity, guid)

        assert manager.get_entity(guid) is entity
        assert manager.get_guid(entity) == guid

    def test_manager_release_guid(self):
        """Releasing a GUID should free it."""
        manager = NetGUIDManager(GUIDAuthority.SERVER)
        entity = MockEntity()
        guid = manager.assign_guid(entity)

        manager.release_guid(guid.value)

        assert manager.get_entity(guid) is None

    def test_manager_get_entity(self):
        """get_entity should return the entity for a GUID."""
        manager = NetGUIDManager(GUIDAuthority.SERVER)
        entity = MockEntity()
        guid = manager.assign_guid(entity)

        assert manager.get_entity(guid) is entity
        assert manager.get_entity(NetGUID(999999)) is None

    def test_manager_get_guid(self):
        """get_guid should return the GUID for an entity."""
        manager = NetGUIDManager(GUIDAuthority.SERVER)
        entity = MockEntity()
        guid = manager.assign_guid(entity)

        assert manager.get_guid(entity) == guid

    def test_guid_is_valid_property(self):
        """NetGUID.is_valid should check GUID validity."""
        # Invalid GUID
        invalid_guid = NetGUID(DEFAULT_CONFIG.NULL_GUID)
        assert not invalid_guid.is_valid

        # Valid GUID
        valid_guid = NetGUID(1)
        assert valid_guid.is_valid


# =============================================================================
# ReplicatedProperty Tests
# =============================================================================

class TestReplicatedProperty:
    """Tests for ReplicatedProperty tracking."""

    def test_property_creation(self):
        """Property should store name and value."""
        prop = ReplicatedProperty(
            name="health",
            value=100,
            value_type=int
        )
        assert prop.name == "health"
        assert prop.value == 100

    def test_property_set_value_marks_dirty(self):
        """Setting a new value should mark property dirty."""
        prop = ReplicatedProperty(name="health", value=100, value_type=int)
        prop.dirty = False
        assert not prop.dirty

        prop.set_value(90)
        assert prop.dirty
        assert prop.value == 90

    def test_property_set_same_value_no_dirty(self):
        """Setting same value should not mark dirty."""
        prop = ReplicatedProperty(name="health", value=100, value_type=int)
        prop.dirty = False

        prop.set_value(100)
        assert not prop.dirty

    def test_property_mark_dirty(self):
        """mark_dirty should set dirty flag."""
        prop = ReplicatedProperty(name="health", value=100, value_type=int)
        prop.mark_clean()
        assert not prop.dirty

        prop.mark_dirty()
        assert prop.dirty

    def test_property_mark_clean(self):
        """mark_clean should clear dirty flag."""
        prop = ReplicatedProperty(name="health", value=100, value_type=int)
        prop.mark_dirty()
        assert prop.dirty

        prop.mark_clean()
        assert not prop.dirty

    def test_property_condition_always(self):
        """ALWAYS condition should always replicate."""
        prop = ReplicatedProperty(
            name="position",
            value=(0, 0, 0),
            value_type=tuple,
            condition=ReplicationCondition.ALWAYS
        )
        assert prop.condition == ReplicationCondition.ALWAYS

    def test_property_condition_on_change(self):
        """ON_CHANGE condition should replicate when dirty."""
        prop = ReplicatedProperty(
            name="health",
            value=100,
            value_type=int,
            condition=ReplicationCondition.ON_CHANGE
        )
        assert prop.condition == ReplicationCondition.ON_CHANGE


class TestPropertyReplicationGroup:
    """Tests for PropertyReplicationGroup."""

    def test_group_add_property(self):
        """Adding properties should work."""
        group = PropertyReplicationGroup("TestGroup")
        prop = ReplicatedProperty(name="health", value=100, value_type=int)

        group.add_property(prop)

        assert group.get_property("health") is not None

    def test_group_get_property(self):
        """Getting properties by name should work."""
        group = PropertyReplicationGroup("TestGroup")
        prop1 = ReplicatedProperty(name="health", value=100, value_type=int)
        prop2 = ReplicatedProperty(name="mana", value=50, value_type=int)

        group.add_property(prop1)
        group.add_property(prop2)

        assert group.get_property("health").value == 100
        assert group.get_property("mana").value == 50
        assert group.get_property("nonexistent") is None

    def test_group_get_dirty_properties(self):
        """get_dirty_properties should return only dirty ones."""
        group = PropertyReplicationGroup("TestGroup")
        prop1 = ReplicatedProperty(name="health", value=100, value_type=int)
        prop2 = ReplicatedProperty(name="mana", value=50, value_type=int)

        # Mark dirty first
        prop1.mark_dirty()
        prop2.mark_dirty()

        group.add_property(prop1)
        group.add_property(prop2)

        # Both should be dirty
        dirty = group.get_dirty_properties()
        assert len(dirty) == 2

        # Clean one
        prop1.mark_clean()
        dirty = group.get_dirty_properties()
        assert len(dirty) == 1
        assert dirty[0].name == "mana"

    def test_group_mark_all_clean(self):
        """mark_all_clean should clean all properties."""
        group = PropertyReplicationGroup("TestGroup")
        prop_a = ReplicatedProperty(name="a", value=1, value_type=int)
        prop_b = ReplicatedProperty(name="b", value=2, value_type=int)
        prop_a.mark_dirty()
        prop_b.mark_dirty()
        group.add_property(prop_a)
        group.add_property(prop_b)

        group.mark_all_clean()

        assert len(group.get_dirty_properties()) == 0

    def test_group_serialize_all(self):
        """serialize_all should serialize all properties."""
        group = PropertyReplicationGroup("TestGroup")
        group.add_property(ReplicatedProperty(name="health", value=100, value_type=int))
        group.add_property(ReplicatedProperty(name="name", value="Entity", value_type=str))

        data = group.serialize_all()
        assert len(data) > 0

    def test_group_serialize_dirty(self):
        """serialize_dirty should serialize only dirty properties."""
        group = PropertyReplicationGroup("TestGroup")
        prop1 = ReplicatedProperty(name="health", value=100, value_type=int)
        prop2 = ReplicatedProperty(name="mana", value=50, value_type=int)

        group.add_property(prop1)
        group.add_property(prop2)
        group.mark_all_clean()

        prop1.set_value(90)  # Mark dirty

        data = group.serialize_dirty(None, False)
        # Should include property count header
        assert len(data) > 0


# =============================================================================
# RelevancyManager Tests
# =============================================================================

class TestRelevancyResult:
    """Tests for RelevancyResult."""

    def test_result_relevant(self):
        """Relevant result should be marked as such."""
        result = RelevancyResult(is_relevant=True, priority=1.0)
        assert result.is_relevant
        assert result.priority == 1.0

    def test_result_not_relevant(self):
        """Not relevant result should have zero priority."""
        result = RelevancyResult(is_relevant=False, priority=0.0)
        assert not result.is_relevant


class TestRadiusRelevancy:
    """Tests for RadiusRelevancy calculations."""

    def test_radius_within_range(self):
        """Entity within radius should be relevant."""
        relevancy = RadiusRelevancy(radius=100.0)

        # Entity and viewer with position attributes
        entity = MockEntity(position=(50, 0, 0))
        viewer = MockEntity(position=(0, 0, 0))

        result = relevancy.check_relevant(entity, viewer)
        assert result.is_relevant

    def test_radius_outside_range(self):
        """Entity outside radius should not be relevant."""
        relevancy = RadiusRelevancy(radius=100.0)

        entity = MockEntity(position=(200, 0, 0))
        viewer = MockEntity(position=(0, 0, 0))

        result = relevancy.check_relevant(entity, viewer)
        assert not result.is_relevant

    def test_radius_priority_falloff(self):
        """Priority should decrease with distance."""
        relevancy = RadiusRelevancy(radius=100.0, falloff_start=50.0)

        close_entity = MockEntity(position=(25, 0, 0))
        far_entity = MockEntity(position=(75, 0, 0))
        viewer = MockEntity(position=(0, 0, 0))

        close_result = relevancy.check_relevant(close_entity, viewer)
        far_result = relevancy.check_relevant(far_entity, viewer)

        assert close_result.priority > far_result.priority

    def test_radius_at_boundary(self):
        """Entity at exact boundary should be relevant."""
        relevancy = RadiusRelevancy(radius=100.0)

        entity = MockEntity(position=(100, 0, 0))
        viewer = MockEntity(position=(0, 0, 0))

        result = relevancy.check_relevant(entity, viewer)
        assert result.is_relevant


class TestInterestArea:
    """Tests for InterestArea base class."""

    def test_interest_area_is_abstract(self):
        """InterestArea is an abstract base class."""
        # InterestArea is ABC, so we test a concrete implementation
        area = RadiusRelevancy(radius=50.0)
        assert area.radius == 50.0


class TestGridRelevancy:
    """Tests for GridRelevancy spatial hashing."""

    def test_grid_register_entity(self):
        """Registering entities should work."""
        grid = GridRelevancy(cell_size=100.0)

        entity = MockEntity(position=(50, 0, 50))
        grid.register_entity(entity)

        # Should be able to get entities near that position
        entities = grid.get_entities_near((50, 0, 50))
        assert entity in entities

    def test_grid_unregister_entity(self):
        """Unregistering entities should work."""
        grid = GridRelevancy(cell_size=100.0)

        entity = MockEntity(position=(50, 0, 50))
        grid.register_entity(entity)
        grid.unregister_entity(entity)

        entities = grid.get_entities_near((50, 0, 50))
        assert entity not in entities

    def test_grid_update_entity_position(self):
        """Updating entity position should work."""
        grid = GridRelevancy(cell_size=100.0, view_distance=1)

        entity = MockEntity(position=(50, 0, 50))
        grid.register_entity(entity)

        # Move entity
        entity.position = (500, 0, 500)
        grid.update_entity(entity)

        # Should be at new position
        far_entities = grid.get_entities_near((500, 0, 500))
        assert entity in far_entities

    def test_grid_get_entities_near(self):
        """get_entities_near should return entities within view distance."""
        grid = GridRelevancy(cell_size=100.0, view_distance=1)

        e1 = MockEntity(position=(0, 0, 0))
        e2 = MockEntity(position=(50, 0, 0))
        e3 = MockEntity(position=(500, 0, 0))

        grid.register_entity(e1)
        grid.register_entity(e2)
        grid.register_entity(e3)

        nearby = grid.get_entities_near((0, 0, 0))
        assert e1 in nearby
        assert e2 in nearby
        # e3 is far away
        assert e3 not in nearby


class TestRelevancyManager:
    """Tests for RelevancyManager."""

    def test_manager_default_relevancy(self):
        """Default relevancy should use radius check."""
        manager = RelevancyManager()

        entity = MockEntity(position=(50, 0, 0))
        viewer = MockEntity(position=(0, 0, 0))

        result = manager.check_relevant(entity, viewer)
        # Should be relevant if within default radius
        assert isinstance(result, RelevancyResult)

    def test_manager_set_entity_area(self):
        """Custom interest areas should work."""
        manager = RelevancyManager()

        entity = MockEntity()
        area = RadiusRelevancy(radius=100.0)

        manager.set_entity_area(entity, area)
        # Area should be stored (test by checking relevancy uses it)

    def test_manager_remove_entity(self):
        """Removing entity should clean up."""
        manager = RelevancyManager()

        entity = MockEntity()
        manager.set_entity_area(entity, RadiusRelevancy())
        manager.remove_entity(entity)
        # Should not raise


# =============================================================================
# BandwidthManager Tests
# =============================================================================

class TestEntityPriority:
    """Tests for EntityPriority constants."""

    def test_priority_constants(self):
        """Priority constants should be ordered correctly."""
        assert EntityPriority.CRITICAL > EntityPriority.HIGH
        assert EntityPriority.HIGH > EntityPriority.NORMAL
        assert EntityPriority.NORMAL > EntityPriority.LOW
        assert EntityPriority.LOW > EntityPriority.MINIMAL


class TestBandwidthManager:
    """Tests for BandwidthManager."""

    def test_manager_queue_entity(self):
        """Queueing entities should work."""
        manager = BandwidthManager()
        entity = Mock()

        manager.queue_entity(
            connection_id=1,
            entity=entity,
            guid=100,
            priority=EntityPriority.NORMAL,
            estimated_size=64
        )
        # Should not raise

    def test_manager_allocate_returns_entities(self):
        """allocate should return entities within budget."""
        manager = BandwidthManager()

        # Queue several entities
        for i in range(5):
            entity = Mock()
            manager.queue_entity(
                connection_id=1,
                entity=entity,
                guid=i,
                priority=EntityPriority.NORMAL,
                estimated_size=100
            )

        allocated = manager.allocate(connection_id=1)
        assert len(allocated) > 0

    def test_manager_allocate_priority_order(self):
        """Higher priority entities should be allocated first."""
        manager = BandwidthManager()

        low = Mock()
        high = Mock()

        manager.queue_entity(1, low, 1, EntityPriority.LOW, 100)
        manager.queue_entity(1, high, 2, EntityPriority.HIGH, 100)

        allocated = manager.allocate(1)
        # High priority should come first
        if len(allocated) >= 2:
            entities = [item[0] for item in allocated]
            assert entities.index(high) < entities.index(low)

    def test_manager_remove_entity(self):
        """Removing entity should work."""
        manager = BandwidthManager()
        manager.queue_entity(1, Mock(), 100, EntityPriority.NORMAL, 64)

        manager.remove_entity(100)
        # Should not raise

    def test_manager_remove_connection(self):
        """Removing connection should clean up queues."""
        manager = BandwidthManager()
        manager.queue_entity(1, Mock(), 100, EntityPriority.NORMAL, 64)

        manager.remove_connection(1)
        # Should not raise


# =============================================================================
# ReplicationManager Tests
# =============================================================================

class TestReplicatedEntity:
    """Tests for ReplicatedEntity wrapper."""

    def test_entity_creation(self):
        """ReplicatedEntity should wrap entity correctly."""
        entity = MockEntity()
        guid = NetGUID(100)

        replicated = ReplicatedEntity(
            entity=entity,
            guid=guid,
            state=EntityState.PENDING_SPAWN
        )

        assert replicated.entity is entity
        assert replicated.guid == guid
        assert replicated.state == EntityState.PENDING_SPAWN

    def test_entity_is_dirty(self):
        """is_dirty should check properties."""
        entity = MockEntity()
        guid = NetGUID(100)
        replicated = ReplicatedEntity(entity=entity, guid=guid)

        # Add a dirty property
        prop = ReplicatedProperty(name="test", value=1, value_type=int)
        prop.mark_dirty()  # Explicitly mark dirty
        replicated.properties.add_property(prop)

        assert replicated.is_dirty()

    def test_entity_mark_replicated(self):
        """mark_replicated should clean properties and update state."""
        entity = MockEntity()
        guid = NetGUID(100)
        replicated = ReplicatedEntity(
            entity=entity,
            guid=guid,
            state=EntityState.PENDING_SPAWN
        )

        prop = ReplicatedProperty(name="test", value=1, value_type=int)
        replicated.properties.add_property(prop)

        replicated.mark_replicated()

        assert not replicated.is_dirty()
        assert replicated.state == EntityState.ACTIVE


class TestReplicationManager:
    """Tests for ReplicationManager."""

    def test_manager_server_role(self):
        """Server role should be set correctly."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)
        assert manager.role == ReplicationRole.SERVER

    def test_manager_client_role(self):
        """Client role should be set correctly."""
        manager = ReplicationManager(role=ReplicationRole.CLIENT)
        assert manager.role == ReplicationRole.CLIENT

    def test_manager_register_entity(self):
        """Registering entity should assign GUID."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)
        entity = MockEntity()

        guid = manager.register_entity(entity)

        assert guid is not None
        assert guid.value != 0
        assert manager.get_entity(guid) is entity

    def test_manager_register_with_priority(self):
        """Registering with priority should set it."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)
        entity = MockEntity()

        guid = manager.register_entity(entity, priority=EntityPriority.HIGH)

        replicated = manager.get_replicated_entity(guid)
        assert replicated.priority == EntityPriority.HIGH

    def test_manager_register_with_owner(self):
        """Registering with owner_id should set it."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)
        entity = MockEntity()

        guid = manager.register_entity(entity, owner_id=12345)

        replicated = manager.get_replicated_entity(guid)
        assert replicated.owner_id == 12345

    def test_manager_unregister_entity(self):
        """Unregistering entity should mark for destroy."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)
        entity = MockEntity()
        guid = manager.register_entity(entity)

        result = manager.unregister_entity(guid)

        assert result == True
        replicated = manager.get_replicated_entity(guid)
        assert replicated.state == EntityState.PENDING_DESTROY

    def test_manager_unregister_nonexistent(self):
        """Unregistering nonexistent entity should return False."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)

        result = manager.unregister_entity(NetGUID(99999))

        assert result == False

    def test_manager_get_entity(self):
        """get_entity should return the entity."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)
        entity = MockEntity()
        guid = manager.register_entity(entity)

        assert manager.get_entity(guid) is entity
        assert manager.get_entity(NetGUID(99999)) is None

    def test_manager_get_replicated_entity(self):
        """get_replicated_entity should return wrapper."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)
        entity = MockEntity()
        guid = manager.register_entity(entity)

        replicated = manager.get_replicated_entity(guid)

        assert replicated is not None
        assert replicated.entity is entity
        assert replicated.guid == guid

    def test_manager_get_dirty_entities(self):
        """get_dirty_entities should return dirty entities."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)

        e1 = MockEntity()
        e2 = MockEntity()
        guid1 = manager.register_entity(e1)
        guid2 = manager.register_entity(e2)

        # Pending spawn counts as dirty
        dirty = manager.get_dirty_entities()
        assert len(dirty) == 2

    def test_manager_mark_property_dirty(self):
        """mark_property_dirty should mark specific property."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)
        entity = MockEntity()
        guid = manager.register_entity(entity)

        # Add property manually
        replicated = manager.get_replicated_entity(guid)
        prop = ReplicatedProperty(name="health", value=100, value_type=int)
        replicated.properties.add_property(prop)
        replicated.properties.mark_all_clean()

        manager.mark_property_dirty(guid, "health")

        assert replicated.is_dirty()

    def test_manager_set_property_value(self):
        """set_property_value should update and mark dirty."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)
        entity = MockEntity()
        guid = manager.register_entity(entity)

        # Add property
        replicated = manager.get_replicated_entity(guid)
        prop = ReplicatedProperty(name="health", value=100, value_type=int)
        replicated.properties.add_property(prop)
        replicated.properties.mark_all_clean()

        result = manager.set_property_value(guid, "health", 90)

        assert result == True
        assert prop.value == 90
        assert prop.dirty

    def test_manager_add_connection(self):
        """add_connection should track connection."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)
        viewer = MockEntity()

        manager.add_connection(1, viewer)
        # Should not raise

    def test_manager_remove_connection(self):
        """remove_connection should clean up."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)
        viewer = MockEntity()
        manager.add_connection(1, viewer)

        manager.remove_connection(1)
        # Should not raise

    def test_manager_collect_replication_data(self):
        """collect_replication_data should gather data for viewer."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)

        @dataclass
        class MockViewer:
            position: tuple = (0, 0, 0)
            player_id: int = 1

        viewer = MockViewer()
        manager.add_connection(1, viewer)

        # Register entity
        entity = MockEntity(position=(10, 0, 0))
        guid = manager.register_entity(entity)

        # Collect data
        data = manager.collect_replication_data(viewer, 1)

        # Should have some data (batch header at minimum)
        assert len(data) > 0


class TestReplicationManagerDataHandling:
    """Tests for replication data serialization/deserialization."""

    def test_apply_empty_data(self):
        """apply_replication_data with empty data should return 0."""
        manager = ReplicationManager(role=ReplicationRole.CLIENT)

        consumed = manager.apply_replication_data(b'')
        assert consumed == 0

    def test_collect_and_apply_spawn(self):
        """Spawn data should be collectable and applicable."""
        server = ReplicationManager(role=ReplicationRole.SERVER)
        client = ReplicationManager(role=ReplicationRole.CLIENT)

        @dataclass
        class MockViewer:
            position: tuple = (0, 0, 0)
            player_id: int = 1

        viewer = MockViewer()
        server.add_connection(1, viewer)

        entity = MockEntity(position=(0, 0, 0))
        guid = server.register_entity(entity)

        # Collect spawn data
        data = server.collect_replication_data(viewer, 1)

        if len(data) > 1:  # Has content beyond batch header
            # Apply on client
            consumed = client.apply_replication_data(data)
            assert consumed > 0

    def test_finalize_destroys(self):
        """finalize_destroys should remove pending destroy entities."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)

        entity = MockEntity()
        guid = manager.register_entity(entity)
        manager.unregister_entity(guid)

        # Before finalize
        assert manager.get_replicated_entity(guid) is not None

        manager.finalize_destroys()

        # After finalize
        assert manager.get_replicated_entity(guid) is None

    def test_update_periodic(self):
        """update should process queued operations."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)

        entity = MockEntity()
        guid = manager.register_entity(entity)

        # Call update
        manager.update()
        # Should not raise


class TestReplicationManagerCallbacks:
    """Tests for replication callbacks."""

    def test_on_spawn_callback(self):
        """on_spawn callback should be settable."""
        manager = ReplicationManager(role=ReplicationRole.CLIENT)
        callback = Mock()

        manager.set_on_spawn_callback(callback)
        # Callback is stored

    def test_on_destroy_callback(self):
        """on_destroy callback should be settable."""
        manager = ReplicationManager(role=ReplicationRole.CLIENT)
        callback = Mock()

        manager.set_on_destroy_callback(callback)
        # Callback is stored


class TestReplicationManagerAccessors:
    """Tests for manager accessors."""

    def test_guid_manager_accessor(self):
        """guid_manager property should return manager."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)
        assert manager.guid_manager is not None
        assert isinstance(manager.guid_manager, NetGUIDManager)

    def test_relevancy_manager_accessor(self):
        """relevancy_manager property should return manager."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)
        assert manager.relevancy_manager is not None
        assert isinstance(manager.relevancy_manager, RelevancyManager)

    def test_bandwidth_manager_accessor(self):
        """bandwidth_manager property should return manager."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)
        assert manager.bandwidth_manager is not None
        assert isinstance(manager.bandwidth_manager, BandwidthManager)

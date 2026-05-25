"""Tests for the replication system.

Tests cover:
- NetGUID allocation and management
- Property replication and serialization
- Relevancy filtering (distance, owner, grid)
- Bandwidth allocation and anti-starvation
- Replication manager coordination
- Actor channel communication
"""

import pytest
import time
from dataclasses import dataclass
from typing import Optional

from engine.networking.replication import (
    # Net GUID
    NetGUID,
    NetGUIDManager,
    GUIDAuthority,
    INVALID_GUID,
    NULL_GUID,

    # Property Replication
    ReplicatedProperty,
    PropertyReplicationGroup,
    ReplicationCondition,
    ChangeNotifyMode,
    create_replicated_property,

    # Relevancy
    RelevancyResult,
    RadiusRelevancy,
    GridRelevancy,
    OwnerRelevant,
    AlwaysRelevant,
    CompositeRelevancy,
    RelevancyManager,

    # Bandwidth
    BandwidthBudget,
    PriorityQueue,
    EntityPriority,
    allocate_bandwidth,
    allocate_bandwidth_fair,

    # Replication Manager
    ReplicationManager,
    ReplicationRole,
    EntityState,

    # Actor Channel
    ActorChannel,
    ActorChannelManager,
    ChannelState,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

@dataclass
class MockEntity:
    """Mock entity for testing."""
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    health: int = 100
    name: str = "TestEntity"
    owner_id: Optional[int] = None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass
class MockViewer:
    """Mock viewer/player for testing."""
    player_id: int = 1
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)


# =============================================================================
# NetGUID Tests
# =============================================================================

class TestNetGUID:
    """Tests for NetGUID class."""

    def test_guid_creation(self):
        """Test GUID creation with valid values."""
        guid = NetGUID(12345)
        assert guid.value == 12345
        assert guid.is_valid

    def test_guid_null(self):
        """Test null GUID."""
        guid = NetGUID.null()
        assert guid.value == NULL_GUID
        assert not guid.is_valid

    def test_guid_invalid(self):
        """Test invalid GUID marker."""
        guid = NetGUID.invalid()
        assert guid.value == INVALID_GUID
        assert not guid.is_valid

    def test_guid_authority_server(self):
        """Test server authority detection."""
        guid = NetGUID(0x00001234)
        assert guid.authority == GUIDAuthority.SERVER
        assert guid.client_id is None

    def test_guid_authority_client(self):
        """Test client authority detection."""
        guid = NetGUID(0x80011234)
        assert guid.authority == GUIDAuthority.CLIENT
        assert guid.client_id == 1  # Client ID in bits 16-30

    def test_guid_serialization(self):
        """Test GUID serialization/deserialization."""
        original = NetGUID(0x12345678)
        serialized = original.serialize()
        assert len(serialized) == 4

        restored = NetGUID.deserialize(serialized)
        assert restored.value == original.value

    def test_guid_hash_equality(self):
        """Test GUID hash and equality."""
        guid1 = NetGUID(12345)
        guid2 = NetGUID(12345)
        guid3 = NetGUID(54321)

        assert guid1 == guid2
        assert guid1 != guid3
        assert hash(guid1) == hash(guid2)
        assert guid1 == 12345  # Int comparison


class TestNetGUIDManager:
    """Tests for NetGUIDManager class."""

    def test_manager_creation_server(self):
        """Test server-mode manager creation."""
        manager = NetGUIDManager(GUIDAuthority.SERVER)
        assert manager.authority == GUIDAuthority.SERVER

    def test_manager_creation_client(self):
        """Test client-mode manager creation."""
        manager = NetGUIDManager(GUIDAuthority.CLIENT, client_id=5)
        assert manager.authority == GUIDAuthority.CLIENT
        assert manager.client_id == 5

    def test_assign_guid(self):
        """Test GUID assignment."""
        manager = NetGUIDManager()
        entity = MockEntity()

        guid = manager.assign_guid(entity)

        assert guid.is_valid
        assert manager.get_entity(guid) is entity
        assert manager.get_guid(entity) == guid

    def test_assign_guid_idempotent(self):
        """Test that assigning twice returns same GUID."""
        manager = NetGUIDManager()
        entity = MockEntity()

        guid1 = manager.assign_guid(entity)
        guid2 = manager.assign_guid(entity)

        assert guid1 == guid2

    def test_release_guid(self):
        """Test GUID release."""
        manager = NetGUIDManager()
        entity = MockEntity()

        guid = manager.assign_guid(entity)
        assert manager.release_guid(guid)
        assert manager.get_entity(guid) is None
        assert not manager.has_guid(guid)

    def test_release_entity(self):
        """Test entity release."""
        manager = NetGUIDManager()
        entity = MockEntity()

        guid = manager.assign_guid(entity)
        assert manager.release_entity(entity)
        assert not manager.is_registered(entity)

    def test_guid_reuse(self):
        """Test GUID recycling after release."""
        manager = NetGUIDManager()
        entity1 = MockEntity(name="Entity1")
        entity2 = MockEntity(name="Entity2")

        guid1 = manager.assign_guid(entity1)
        manager.release_guid(guid1)

        # Next assignment should reuse the released GUID
        guid2 = manager.assign_guid(entity2)
        assert guid2.value == guid1.value

    def test_import_guid(self):
        """Test importing external GUID."""
        manager = NetGUIDManager()
        entity = MockEntity()
        external_guid = NetGUID(0x12345678)

        assert manager.import_guid(entity, external_guid)
        assert manager.get_entity(external_guid) is entity

    def test_get_all_guids(self):
        """Test getting all assigned GUIDs."""
        manager = NetGUIDManager()
        entities = [MockEntity(name=f"Entity{i}") for i in range(5)]

        guids = [manager.assign_guid(e) for e in entities]

        all_guids = manager.get_all_guids()
        assert len(all_guids) == 5
        for guid in guids:
            assert guid in all_guids


# =============================================================================
# Property Replication Tests
# =============================================================================

class TestReplicatedProperty:
    """Tests for ReplicatedProperty class."""

    def test_property_creation(self):
        """Test property creation."""
        prop = create_replicated_property("health", 100)

        assert prop.name == "health"
        assert prop.value == 100
        assert prop.value_type == int
        assert prop.condition == ReplicationCondition.ON_CHANGE

    def test_property_dirty_tracking(self):
        """Test dirty flag tracking."""
        prop = create_replicated_property("health", 100)

        assert not prop.dirty

        prop.set_value(90)
        assert prop.dirty
        assert prop.value == 90

        prop.mark_clean()
        assert not prop.dirty

    def test_property_no_change_not_dirty(self):
        """Test that setting same value doesn't mark dirty."""
        prop = create_replicated_property("health", 100)
        prop.mark_clean()

        prop.set_value(100)  # Same value
        assert not prop.dirty

    def test_property_serialization_int(self):
        """Test integer property serialization."""
        prop = create_replicated_property("health", 42)

        serialized = prop.serialize()
        assert len(serialized) == 4

        prop2 = create_replicated_property("health", 0)
        prop2.deserialize(serialized)
        assert prop2.value == 42

    def test_property_serialization_float(self):
        """Test float property serialization."""
        prop = create_replicated_property("speed", 3.14)

        serialized = prop.serialize()
        prop2 = create_replicated_property("speed", 0.0)
        prop2.deserialize(serialized)

        assert abs(prop2.value - 3.14) < 0.001

    def test_property_serialization_string(self):
        """Test string property serialization."""
        prop = create_replicated_property("name", "TestPlayer")

        serialized = prop.serialize()
        prop2 = create_replicated_property("name", "")
        prop2.deserialize(serialized)

        assert prop2.value == "TestPlayer"

    def test_should_replicate_on_change(self):
        """Test ON_CHANGE replication condition."""
        prop = create_replicated_property(
            "health", 100,
            condition=ReplicationCondition.ON_CHANGE
        )
        prop.mark_clean()

        assert not prop.should_replicate()

        prop.set_value(90)
        assert prop.should_replicate()

    def test_should_replicate_always(self):
        """Test ALWAYS replication condition."""
        prop = create_replicated_property(
            "health", 100,
            condition=ReplicationCondition.ALWAYS
        )

        assert prop.should_replicate()
        prop.mark_clean()
        assert prop.should_replicate()  # Still true

    def test_should_replicate_initial_only(self):
        """Test INITIAL_ONLY replication condition."""
        prop = create_replicated_property(
            "health", 100,
            condition=ReplicationCondition.INITIAL_ONLY
        )

        assert prop.should_replicate()

        prop.mark_clean()  # Marks _initial_sent = True
        assert not prop.should_replicate()

    def test_should_replicate_owner_only(self):
        """Test OWNER_ONLY replication condition."""
        prop = create_replicated_property(
            "inventory", [],
            condition=ReplicationCondition.OWNER_ONLY
        )
        prop.mark_dirty()

        assert prop.should_replicate(is_owner=True)
        assert not prop.should_replicate(is_owner=False)

    def test_should_replicate_skip_owner(self):
        """Test SKIP_OWNER replication condition."""
        prop = create_replicated_property(
            "visibility", True,
            condition=ReplicationCondition.SKIP_OWNER
        )
        prop.mark_dirty()

        assert not prop.should_replicate(is_owner=True)
        assert prop.should_replicate(is_owner=False)

    def test_rep_notify_callback(self):
        """Test replication notification callback."""
        received = []

        def on_rep(new_val, old_val):
            received.append((new_val, old_val))

        prop = create_replicated_property(
            "health", 100,
            notify_mode=ChangeNotifyMode.WITH_PREVIOUS
        )
        prop.set_on_rep_callback(on_rep)

        # Simulate receiving replicated value
        prop.on_rep_notify(80, 100)

        assert len(received) == 1
        assert received[0] == (80, 100)


class TestPropertyReplicationGroup:
    """Tests for PropertyReplicationGroup class."""

    def test_group_creation(self):
        """Test group creation and property management."""
        group = PropertyReplicationGroup("player")

        group.add_property(create_replicated_property("health", 100))
        group.add_property(create_replicated_property("mana", 50))

        assert group.get_property("health") is not None
        assert group.get_property("mana") is not None
        assert group.get_property("nonexistent") is None

    def test_group_dirty_tracking(self):
        """Test group dirty property collection."""
        group = PropertyReplicationGroup("player")

        health = create_replicated_property("health", 100)
        mana = create_replicated_property("mana", 50)

        group.add_property(health)
        group.add_property(mana)

        group.mark_all_clean()
        assert len(group.get_dirty_properties()) == 0

        health.set_value(90)
        dirty = group.get_dirty_properties()
        assert len(dirty) == 1
        assert dirty[0].name == "health"

    def test_group_serialization(self):
        """Test group serialization."""
        group = PropertyReplicationGroup("player")
        group.add_property(create_replicated_property("health", 100))
        group.add_property(create_replicated_property("level", 5))

        serialized = group.serialize_all()
        assert len(serialized) > 0

        # Deserialize into new group
        group2 = PropertyReplicationGroup("player")
        group2.add_property(create_replicated_property("health", 0))
        group2.add_property(create_replicated_property("level", 0))

        group2.deserialize(serialized)

        assert group2.get_property("health").value == 100
        assert group2.get_property("level").value == 5


# =============================================================================
# Relevancy Tests
# =============================================================================

class TestRadiusRelevancy:
    """Tests for radius-based relevancy."""

    def test_within_radius(self):
        """Test entity within radius is relevant."""
        relevancy = RadiusRelevancy(radius=1000.0)

        entity = MockEntity(position=(100.0, 0.0, 0.0))
        viewer = MockViewer(position=(0.0, 0.0, 0.0))

        result = relevancy.check_relevant(entity, viewer)

        assert result.is_relevant
        assert result.priority > 0

    def test_outside_radius(self):
        """Test entity outside radius is not relevant."""
        relevancy = RadiusRelevancy(radius=100.0)

        entity = MockEntity(position=(500.0, 0.0, 0.0))
        viewer = MockViewer(position=(0.0, 0.0, 0.0))

        result = relevancy.check_relevant(entity, viewer)

        assert not result.is_relevant

    def test_priority_falloff(self):
        """Test priority decreases with distance."""
        relevancy = RadiusRelevancy(radius=1000.0, falloff_start=500.0)

        near_entity = MockEntity(position=(100.0, 0.0, 0.0))
        far_entity = MockEntity(position=(800.0, 0.0, 0.0))
        viewer = MockViewer(position=(0.0, 0.0, 0.0))

        near_result = relevancy.check_relevant(near_entity, viewer)
        far_result = relevancy.check_relevant(far_entity, viewer)

        assert near_result.priority > far_result.priority

    def test_owner_always_relevant(self):
        """Test owner is always relevant regardless of distance."""
        relevancy = RadiusRelevancy(
            radius=100.0,
            always_relevant_to_owner=True
        )

        entity = MockEntity(position=(5000.0, 0.0, 0.0), owner_id=1)
        viewer = MockViewer(player_id=1, position=(0.0, 0.0, 0.0))

        result = relevancy.check_relevant(entity, viewer)

        assert result.is_relevant
        assert "owner" in result.reason


class TestGridRelevancy:
    """Tests for grid-based relevancy."""

    def test_same_cell_relevant(self):
        """Test entities in same cell are relevant."""
        relevancy = GridRelevancy(cell_size=100.0, view_distance=3)

        entity = MockEntity(position=(50.0, 50.0, 0.0))
        viewer = MockViewer(position=(60.0, 40.0, 0.0))

        result = relevancy.check_relevant(entity, viewer)

        assert result.is_relevant

    def test_far_cell_not_relevant(self):
        """Test entities in distant cells are not relevant."""
        relevancy = GridRelevancy(cell_size=100.0, view_distance=2)

        entity = MockEntity(position=(1000.0, 0.0, 0.0))
        viewer = MockViewer(position=(0.0, 0.0, 0.0))

        result = relevancy.check_relevant(entity, viewer)

        assert not result.is_relevant

    def test_entity_registration(self):
        """Test entity grid registration."""
        relevancy = GridRelevancy(cell_size=100.0, view_distance=3)

        entity1 = MockEntity(position=(50.0, 50.0, 0.0))
        entity2 = MockEntity(position=(150.0, 50.0, 0.0))

        relevancy.register_entity(entity1)
        relevancy.register_entity(entity2)

        # Query near center
        near_entities = relevancy.get_entities_near((50.0, 50.0, 0.0))

        assert entity1 in near_entities
        assert entity2 in near_entities


class TestOwnerRelevant:
    """Tests for owner-based relevancy."""

    def test_owner_relevant(self):
        """Test owner sees their entity."""
        relevancy = OwnerRelevant()

        entity = MockEntity(owner_id=1)
        viewer = MockViewer(player_id=1)

        result = relevancy.check_relevant(entity, viewer)

        assert result.is_relevant
        assert "owner" in result.reason

    def test_non_owner_not_relevant(self):
        """Test non-owner doesn't see entity."""
        relevancy = OwnerRelevant()

        entity = MockEntity(owner_id=1)
        viewer = MockViewer(player_id=2)

        result = relevancy.check_relevant(entity, viewer)

        assert not result.is_relevant


class TestCompositeRelevancy:
    """Tests for composite relevancy."""

    def test_or_logic(self):
        """Test OR logic - any relevant is sufficient."""
        composite = CompositeRelevancy(
            areas=[
                RadiusRelevancy(radius=100.0),
                OwnerRelevant()
            ],
            require_all=False
        )

        # Entity far but owned
        entity = MockEntity(position=(5000.0, 0.0, 0.0), owner_id=1)
        viewer = MockViewer(player_id=1, position=(0.0, 0.0, 0.0))

        result = composite.check_relevant(entity, viewer)

        assert result.is_relevant

    def test_and_logic(self):
        """Test AND logic - all must be relevant."""
        composite = CompositeRelevancy(
            areas=[
                RadiusRelevancy(radius=100.0, always_relevant_to_owner=False),
                OwnerRelevant()
            ],
            require_all=True
        )

        # Entity far but owned - fails distance check
        entity = MockEntity(position=(5000.0, 0.0, 0.0), owner_id=1)
        viewer = MockViewer(player_id=1, position=(0.0, 0.0, 0.0))

        result = composite.check_relevant(entity, viewer)

        assert not result.is_relevant


class TestRelevancyManager:
    """Tests for RelevancyManager."""

    def test_get_relevant_entities(self):
        """Test filtering entities by relevancy."""
        manager = RelevancyManager(RadiusRelevancy(radius=100.0))

        entities = [
            MockEntity(position=(50.0, 0.0, 0.0)),   # Near
            MockEntity(position=(500.0, 0.0, 0.0)), # Far
            MockEntity(position=(75.0, 0.0, 0.0)),  # Near
        ]
        viewer = MockViewer(position=(0.0, 0.0, 0.0))

        relevant = manager.get_relevant_entities(entities, viewer)

        assert len(relevant) == 2


# =============================================================================
# Bandwidth Tests
# =============================================================================

class TestBandwidthBudget:
    """Tests for BandwidthBudget class."""

    def test_budget_creation(self):
        """Test budget creation with defaults."""
        budget = BandwidthBudget()

        assert budget.max_bps > 0
        assert budget.available_bytes > 0

    def test_can_send(self):
        """Test bandwidth availability check."""
        budget = BandwidthBudget(max_bps=8000, burst_bps=16000)

        assert budget.can_send(1000)  # Within burst
        assert budget.can_send(2000)  # Within burst

    def test_consume(self):
        """Test bandwidth consumption."""
        budget = BandwidthBudget(max_bps=8000, burst_bps=8000)

        initial = budget.available_bytes
        budget.consume(100)

        assert budget.available_bytes < initial

    def test_refill(self):
        """Test token refill over time."""
        budget = BandwidthBudget(max_bps=800000, burst_bps=800000)
        budget.consume(budget.available_bytes - 100)

        time.sleep(0.01)  # Small delay
        budget.refill()

        assert budget.available_bytes >= 100


class TestPriorityQueue:
    """Tests for PriorityQueue class."""

    def test_priority_ordering(self):
        """Test entities are dequeued by priority."""
        queue = PriorityQueue()

        queue.add(MockEntity(name="low"), guid=1, priority=10)
        queue.add(MockEntity(name="high"), guid=2, priority=100)
        queue.add(MockEntity(name="medium"), guid=3, priority=50)

        first = queue.pop()
        assert first.guid == 2  # Highest priority

        second = queue.pop()
        assert second.guid == 3  # Medium priority

    def test_anti_starvation(self):
        """Test starvation prevention."""
        queue = PriorityQueue(starvation_threshold=0.01)

        low = MockEntity(name="starved")
        queue.add(low, guid=1, priority=1)

        # Simulate passage of time
        entry = queue._entity_map[1]
        entry.last_sent_time = time.time() - 1.0  # 1 second ago

        # Re-add with same priority - should get boost
        queue.add(low, guid=1, priority=1)

        # Priority should be boosted
        boosted_entry = queue._entity_map[1]
        assert -boosted_entry.priority > 1  # Negative for heap


class TestBandwidthAllocation:
    """Tests for bandwidth allocation functions."""

    def test_allocate_bandwidth(self):
        """Test basic bandwidth allocation."""
        budget = BandwidthBudget(max_bps=80000, burst_bps=80000)

        entities = [
            (MockEntity(name="high"), 1, 100, 500),
            (MockEntity(name="low"), 2, 10, 500),
            (MockEntity(name="medium"), 3, 50, 500),
        ]

        allocated = allocate_bandwidth(entities, budget)

        # Should allocate highest priority first
        assert len(allocated) > 0
        assert allocated[0][1] == 1  # High priority GUID first

        # Check that ordering respects priority
        guids = [guid for _, guid in allocated]
        if len(guids) >= 3:
            # All three should be allocated in priority order
            assert guids == [1, 3, 2], f"Expected priority order [1, 3, 2], got {guids}"

    def test_allocate_bandwidth_fair(self):
        """Test fair bandwidth allocation with anti-starvation."""
        budget = BandwidthBudget(max_bps=80000, burst_bps=80000)

        entities = [
            (MockEntity(name="high"), 1, 100, 100),
            (MockEntity(name="starved"), 2, 10, 100),
        ]

        # First entity was recently sent, second is starved
        last_sent = {
            1: time.time(),
            2: time.time() - 2.0  # Starved
        }

        allocated = allocate_bandwidth_fair(entities, budget, last_sent)

        # Both should be allocated, starved one gets boost
        assert len(allocated) == 2


# =============================================================================
# Replication Manager Tests
# =============================================================================

class TestReplicationManager:
    """Tests for ReplicationManager class."""

    def test_manager_creation(self):
        """Test manager creation."""
        manager = ReplicationManager(role=ReplicationRole.SERVER)

        assert manager.role == ReplicationRole.SERVER

    def test_register_entity(self):
        """Test entity registration."""
        manager = ReplicationManager()
        entity = MockEntity()

        guid = manager.register_entity(entity)

        assert guid.is_valid
        assert manager.get_entity(guid) is entity

    def test_unregister_entity(self):
        """Test entity unregistration."""
        manager = ReplicationManager()
        entity = MockEntity()

        guid = manager.register_entity(entity)
        assert manager.unregister_entity(guid)

        # Entity should be pending destroy
        replicated = manager.get_replicated_entity(guid)
        assert replicated.state == EntityState.PENDING_DESTROY

    def test_get_dirty_entities(self):
        """Test dirty entity collection."""
        manager = ReplicationManager()

        # Register entities
        e1 = MockEntity(name="Entity1")
        e2 = MockEntity(name="Entity2")

        guid1 = manager.register_entity(e1)
        guid2 = manager.register_entity(e2)

        # New entities are pending spawn (dirty)
        dirty = manager.get_dirty_entities()
        assert len(dirty) == 2

    def test_set_property_value(self):
        """Test setting property value through manager."""
        manager = ReplicationManager()
        entity = MockEntity()

        guid = manager.register_entity(entity)

        # Add a property to the replicated entity
        replicated = manager.get_replicated_entity(guid)
        replicated.properties.add_property(
            create_replicated_property("health", 100)
        )
        replicated.properties.mark_all_clean()

        # Set value and verify dirty
        assert manager.set_property_value(guid, "health", 90)
        assert replicated.is_dirty()

    def test_collect_replication_data(self):
        """Test replication data collection."""
        manager = ReplicationManager()

        entity = MockEntity(position=(100.0, 0.0, 0.0))
        guid = manager.register_entity(entity)

        viewer = MockViewer(position=(0.0, 0.0, 0.0))
        manager.add_connection(1, viewer)

        data = manager.collect_replication_data(viewer, connection_id=1)

        # Should have spawn data
        assert len(data) > 0

    def test_apply_replication_data(self):
        """Test applying received replication data."""
        # Server manager
        server = ReplicationManager(role=ReplicationRole.SERVER)
        entity = MockEntity()
        guid = server.register_entity(entity)

        # Get replicated entity and add properties
        rep = server.get_replicated_entity(guid)
        rep.properties.add_property(create_replicated_property("health", 100))

        # Collect data
        viewer = MockViewer()
        server.add_connection(1, viewer)
        data = server.collect_replication_data(viewer, 1)

        # Client manager
        client = ReplicationManager(role=ReplicationRole.CLIENT)

        # Apply data
        consumed = client.apply_replication_data(data)

        assert consumed > 0


# =============================================================================
# Actor Channel Tests
# =============================================================================

class TestActorChannel:
    """Tests for ActorChannel class."""

    def test_channel_lifecycle(self):
        """Test channel open/close lifecycle."""
        guid = NetGUID(12345)
        channel = ActorChannel(guid=guid, connection_id=1)

        assert channel.state == ChannelState.CLOSED

        assert channel.open()
        assert channel.state == ChannelState.OPENING

        # Send spawn completes opening
        channel.send_spawn(b"initial_state")
        messages = channel.get_outgoing_messages()
        assert len(messages) > 0

        # Simulate ack to complete opening
        channel.process_ack(1)
        assert channel.state == ChannelState.OPEN

        # Close
        assert channel.close()
        assert channel.state == ChannelState.CLOSING

    def test_send_update(self):
        """Test sending updates."""
        guid = NetGUID(12345)
        channel = ActorChannel(guid=guid, connection_id=1)
        channel.open()

        # Can't send update until open
        assert not channel.send_update(b"delta")

        # Manually transition to open
        channel.state = ChannelState.OPEN

        assert channel.send_update(b"delta_data", reliable=True)

        messages = channel.get_outgoing_messages()
        assert len(messages) > 0

    def test_reliable_retransmit(self):
        """Test reliable message retransmission."""
        guid = NetGUID(12345)
        channel = ActorChannel(guid=guid, connection_id=1)
        channel.open()
        channel.state = ChannelState.OPEN

        channel.send_update(b"reliable_data", reliable=True)
        messages = channel.get_outgoing_messages()
        assert len(messages) == 1

        # Messages should be pending ack
        assert channel.pending_reliable_count == 1

        # Get retransmits (force timeout)
        channel._pending_ack[1].timestamp = time.time() - 1.0
        retransmits = channel.get_retransmit_messages(timeout=0.5)
        assert len(retransmits) == 1


class TestActorChannelManager:
    """Tests for ActorChannelManager class."""

    def test_open_channel(self):
        """Test opening channels."""
        manager = ActorChannelManager()
        guid = NetGUID(12345)

        channel = manager.open_channel(guid, connection_id=1)

        assert channel is not None
        assert channel.state == ChannelState.OPENING

    def test_get_channel(self):
        """Test getting existing channel."""
        manager = ActorChannelManager()
        guid = NetGUID(12345)

        channel1 = manager.open_channel(guid, connection_id=1)
        channel2 = manager.get_channel(guid, connection_id=1)

        assert channel1 is channel2

    def test_close_all_for_entity(self):
        """Test closing all channels for an entity."""
        manager = ActorChannelManager()
        guid = NetGUID(12345)

        manager.open_channel(guid, connection_id=1)
        manager.open_channel(guid, connection_id=2)

        count = manager.close_all_for_entity(guid)

        assert count == 2

    def test_close_all_for_connection(self):
        """Test closing all channels for a connection."""
        manager = ActorChannelManager()

        manager.open_channel(NetGUID(1), connection_id=1)
        manager.open_channel(NetGUID(2), connection_id=1)
        manager.open_channel(NetGUID(3), connection_id=2)

        count = manager.close_all_for_connection(1)

        assert count == 2

    def test_cleanup_closed_channels(self):
        """Test closed channel cleanup."""
        manager = ActorChannelManager()
        guid = NetGUID(12345)

        channel = manager.open_channel(guid, connection_id=1)
        channel.force_close()

        removed = manager.cleanup_closed_channels()

        assert removed == 1
        assert manager.get_channel(guid, 1) is None


# =============================================================================
# Integration Tests
# =============================================================================

class TestReplicationIntegration:
    """Integration tests for the complete replication system."""

    def test_full_replication_flow(self):
        """Test complete replication from registration to sync."""
        # Server setup
        server = ReplicationManager(role=ReplicationRole.SERVER)

        # Create and register entity
        entity = MockEntity(position=(100.0, 50.0, 0.0), health=100)
        guid = server.register_entity(entity, priority=EntityPriority.HIGH)

        # Add properties
        rep = server.get_replicated_entity(guid)
        rep.properties.add_property(create_replicated_property("health", 100))
        rep.properties.add_property(create_replicated_property("mana", 50))

        # Setup viewer/connection
        viewer = MockViewer(position=(0.0, 0.0, 0.0))
        server.add_connection(1, viewer)

        # Collect initial replication
        data = server.collect_replication_data(viewer, 1)
        assert len(data) > 0

        # Client setup and apply
        client = ReplicationManager(role=ReplicationRole.CLIENT)
        client.apply_replication_data(data)

        # Verify client received entity
        assert client.get_entity(guid) is not None

    def test_property_update_flow(self):
        """Test property update replication."""
        server = ReplicationManager(role=ReplicationRole.SERVER)

        entity = MockEntity()
        guid = server.register_entity(entity)

        rep = server.get_replicated_entity(guid)
        rep.properties.add_property(create_replicated_property("health", 100))

        viewer = MockViewer()
        server.add_connection(1, viewer)

        # Initial sync
        server.collect_replication_data(viewer, 1)
        rep.mark_replicated()

        # Update property
        server.set_property_value(guid, "health", 75)

        # Collect update
        data = server.collect_replication_data(viewer, 1)

        # Should have update data
        assert len(data) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

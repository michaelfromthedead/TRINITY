"""Central replication manager for network state synchronization.

Coordinates entity replication, change tracking, and data collection
for network transmission.
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from ..config import get_config
from .net_guid import NetGUID, NetGUIDManager, GUIDAuthority
from .property_replication import (
    ReplicatedProperty,
    PropertyReplicationGroup,
    ReplicationCondition,
    ChangeNotifyMode,
)
from .relevancy import (
    RelevancyManager,
    RelevancyResult,
    RadiusRelevancy,
    InterestArea,
)
from .bandwidth import BandwidthManager, EntityPriority

_logger = logging.getLogger(__name__)

# Get config instance
_config = get_config()


class ReplicationRole(Enum):
    """Role of this instance in replication."""
    SERVER = auto()      # Authoritative server
    CLIENT = auto()      # Receiving client
    LISTEN_SERVER = auto()  # Server that also acts as client


class EntityState(Enum):
    """Replication state of an entity."""
    PENDING_SPAWN = auto()   # Needs initial spawn to be sent
    ACTIVE = auto()          # Normal replication
    PENDING_DESTROY = auto() # Needs destroy notification
    DORMANT = auto()         # Temporarily not replicating


@dataclass
class ReplicatedEntity:
    """Wrapper for an entity being replicated.

    Attributes:
        entity: The actual entity object
        guid: Network GUID
        state: Current replication state
        properties: Replicated property group
        priority: Base replication priority
        owner_id: Owning player/connection ID
        relevancy: Custom relevancy override
        last_replication_time: When entity was last replicated
    """
    entity: Any
    guid: NetGUID
    state: EntityState = EntityState.PENDING_SPAWN
    properties: PropertyReplicationGroup = field(default_factory=lambda: PropertyReplicationGroup("default"))
    priority: float = EntityPriority.NORMAL
    owner_id: Optional[int] = None
    relevancy: Optional[InterestArea] = None
    last_replication_time: float = field(default_factory=time.time)

    def is_dirty(self) -> bool:
        """Check if entity has dirty properties."""
        return bool(self.properties.get_dirty_properties())

    def mark_replicated(self) -> None:
        """Mark entity as successfully replicated."""
        self.properties.mark_all_clean()
        self.last_replication_time = time.time()
        if self.state == EntityState.PENDING_SPAWN:
            self.state = EntityState.ACTIVE


# Packet type identifiers - from config
PACKET_SPAWN = _config.PACKET_TYPE_SPAWN
PACKET_UPDATE = _config.PACKET_TYPE_UPDATE
PACKET_DESTROY = _config.PACKET_TYPE_DESTROY
PACKET_BATCH = _config.PACKET_TYPE_BATCH


class ReplicationManager:
    """Central manager for entity replication.

    Coordinates tracking, serialization, and collection of entity state
    for network transmission.

    Attributes:
        role: Server or client role
        guid_manager: GUID allocation manager
        relevancy_manager: Interest/relevancy manager
        bandwidth_manager: Bandwidth allocation manager
    """
    __slots__ = (
        '_role', '_guid_manager', '_relevancy_manager', '_bandwidth_manager',
        '_entities', '_pending_spawns', '_pending_destroys',
        '_connection_entity_states', '_on_spawn_callback', '_on_destroy_callback'
    )

    def __init__(
        self,
        role: ReplicationRole = ReplicationRole.SERVER,
        guid_manager: Optional[NetGUIDManager] = None,
        relevancy_manager: Optional[RelevancyManager] = None,
        bandwidth_manager: Optional[BandwidthManager] = None
    ):
        """Initialize the replication manager.

        Args:
            role: Server or client role
            guid_manager: Optional custom GUID manager
            relevancy_manager: Optional custom relevancy manager
            bandwidth_manager: Optional custom bandwidth manager
        """
        self._role = role

        # Create default managers if not provided
        authority = (
            GUIDAuthority.SERVER if role != ReplicationRole.CLIENT
            else GUIDAuthority.CLIENT
        )
        self._guid_manager = guid_manager or NetGUIDManager(authority)
        self._relevancy_manager = relevancy_manager or RelevancyManager()
        self._bandwidth_manager = bandwidth_manager or BandwidthManager()

        # Entity tracking
        self._entities: dict[int, ReplicatedEntity] = {}  # guid value -> entity
        self._pending_spawns: list[int] = []  # GUIDs pending spawn
        self._pending_destroys: list[int] = []  # GUIDs pending destroy

        # Per-connection entity visibility tracking
        # connection_id -> set of visible GUIDs
        self._connection_entity_states: dict[int, dict[int, EntityState]] = {}

        # Callbacks
        self._on_spawn_callback: Optional[Callable[[Any, NetGUID], None]] = None
        self._on_destroy_callback: Optional[Callable[[NetGUID], None]] = None

    @property
    def role(self) -> ReplicationRole:
        """Get replication role."""
        return self._role

    @property
    def guid_manager(self) -> NetGUIDManager:
        """Get the GUID manager."""
        return self._guid_manager

    @property
    def relevancy_manager(self) -> RelevancyManager:
        """Get the relevancy manager."""
        return self._relevancy_manager

    @property
    def bandwidth_manager(self) -> BandwidthManager:
        """Get the bandwidth manager."""
        return self._bandwidth_manager

    def set_on_spawn_callback(self, callback: Callable[[Any, NetGUID], None]) -> None:
        """Set callback for when entities are spawned.

        Args:
            callback: Function (entity, guid) called on spawn
        """
        self._on_spawn_callback = callback

    def set_on_destroy_callback(self, callback: Callable[[NetGUID], None]) -> None:
        """Set callback for when entities are destroyed.

        Args:
            callback: Function (guid) called on destroy
        """
        self._on_destroy_callback = callback

    def register_entity(
        self,
        entity: Any,
        guid: Optional[NetGUID] = None,
        priority: float = EntityPriority.NORMAL,
        owner_id: Optional[int] = None,
        relevancy: Optional[InterestArea] = None
    ) -> NetGUID:
        """Register an entity for replication.

        Args:
            entity: The entity to replicate
            guid: Optional pre-assigned GUID (for imported entities)
            priority: Base replication priority
            owner_id: Owning player/connection ID
            relevancy: Custom relevancy area

        Returns:
            The assigned NetGUID
        """
        # Assign or import GUID
        if guid is None:
            guid = self._guid_manager.assign_guid(entity)
        else:
            self._guid_manager.import_guid(entity, guid)

        # Create replicated entity wrapper
        replicated = ReplicatedEntity(
            entity=entity,
            guid=guid,
            state=EntityState.PENDING_SPAWN,
            priority=priority,
            owner_id=owner_id,
            relevancy=relevancy
        )

        # Extract properties from entity if it has replication metadata
        self._extract_properties(replicated)

        # Register
        self._entities[guid.value] = replicated
        self._pending_spawns.append(guid.value)

        # Set custom relevancy if provided
        if relevancy:
            self._relevancy_manager.set_entity_area(entity, relevancy)

        return guid

    def unregister_entity(self, guid: NetGUID | int) -> bool:
        """Unregister an entity from replication.

        Args:
            guid: The entity's GUID

        Returns:
            True if entity was unregistered
        """
        guid_value = guid.value if isinstance(guid, NetGUID) else guid

        replicated = self._entities.get(guid_value)
        if replicated is None:
            return False

        # Mark for destroy notification
        replicated.state = EntityState.PENDING_DESTROY
        self._pending_destroys.append(guid_value)

        # Remove from GUID manager
        self._guid_manager.release_guid(guid_value)

        # Remove from relevancy manager
        self._relevancy_manager.remove_entity(replicated.entity)

        # Remove from bandwidth manager
        self._bandwidth_manager.remove_entity(guid_value)

        return True

    def get_entity(self, guid: NetGUID | int) -> Optional[Any]:
        """Get an entity by GUID.

        Args:
            guid: The entity's GUID

        Returns:
            The entity or None
        """
        guid_value = guid.value if isinstance(guid, NetGUID) else guid
        replicated = self._entities.get(guid_value)
        return replicated.entity if replicated else None

    def get_replicated_entity(self, guid: NetGUID | int) -> Optional[ReplicatedEntity]:
        """Get the ReplicatedEntity wrapper.

        Args:
            guid: The entity's GUID

        Returns:
            The ReplicatedEntity or None
        """
        guid_value = guid.value if isinstance(guid, NetGUID) else guid
        return self._entities.get(guid_value)

    def get_dirty_entities(self) -> list[ReplicatedEntity]:
        """Get all entities with pending changes.

        Returns:
            List of dirty ReplicatedEntity objects
        """
        dirty = []
        for replicated in self._entities.values():
            if replicated.state == EntityState.PENDING_DESTROY:
                continue
            if replicated.is_dirty() or replicated.state == EntityState.PENDING_SPAWN:
                dirty.append(replicated)
        return dirty

    def collect_replication_data(
        self,
        viewer: Any,
        connection_id: int
    ) -> bytes:
        """Collect replication data for a viewer.

        Filters entities by relevancy and priority, then serializes
        changes within bandwidth budget.

        Args:
            viewer: The viewer (player/connection)
            connection_id: Connection identifier

        Returns:
            Serialized replication data
        """
        # Ensure connection state tracking exists
        if connection_id not in self._connection_entity_states:
            self._connection_entity_states[connection_id] = {}
        connection_states = self._connection_entity_states[connection_id]

        parts = []

        # Process pending spawns for this connection
        spawn_data = self._collect_spawns(viewer, connection_id, connection_states)
        if spawn_data:
            parts.append(spawn_data)

        # Process updates for visible entities
        update_data = self._collect_updates(viewer, connection_id, connection_states)
        if update_data:
            parts.append(update_data)

        # Process pending destroys
        destroy_data = self._collect_destroys(connection_id, connection_states)
        if destroy_data:
            parts.append(destroy_data)

        if not parts:
            return b''

        # Combine with batch header
        return struct.pack('<B', PACKET_BATCH) + b''.join(parts)

    def apply_replication_data(self, data: bytes) -> int:
        """Apply received replication data.

        Processes spawn, update, and destroy packets from server.

        Args:
            data: Serialized replication data

        Returns:
            Number of bytes consumed
        """
        if not data:
            return 0

        offset = 0

        while offset < len(data):
            if offset >= len(data):
                break

            packet_type = data[offset]
            offset += 1

            match packet_type:
                case 0x01:  # PACKET_SPAWN
                    consumed = self._apply_spawn(data[offset:])
                    offset += consumed
                case 0x02:  # PACKET_UPDATE
                    consumed = self._apply_update(data[offset:])
                    offset += consumed
                case 0x03:  # PACKET_DESTROY
                    consumed = self._apply_destroy(data[offset:])
                    offset += consumed
                case 0x04:  # PACKET_BATCH
                    continue  # Batch header, just continue
                case _:
                    # Unknown packet type, stop processing
                    break

        return offset

    def mark_property_dirty(self, guid: NetGUID | int, property_name: str) -> None:
        """Mark a specific property as dirty.

        Args:
            guid: Entity GUID
            property_name: Name of the property
        """
        guid_value = guid.value if isinstance(guid, NetGUID) else guid
        replicated = self._entities.get(guid_value)
        if replicated:
            prop = replicated.properties.get_property(property_name)
            if prop:
                prop.mark_dirty()

    def set_property_value(
        self,
        guid: NetGUID | int,
        property_name: str,
        value: Any
    ) -> bool:
        """Set a property value and mark it dirty.

        Args:
            guid: Entity GUID
            property_name: Property name
            value: New value

        Returns:
            True if property was set
        """
        guid_value = guid.value if isinstance(guid, NetGUID) else guid
        replicated = self._entities.get(guid_value)
        if replicated:
            prop = replicated.properties.get_property(property_name)
            if prop:
                prop.set_value(value)
                return True
        return False

    def add_connection(self, connection_id: int, viewer: Any) -> None:
        """Add a new connection for replication tracking.

        Args:
            connection_id: Connection identifier
            viewer: The viewer object
        """
        self._connection_entity_states[connection_id] = {}

    def remove_connection(self, connection_id: int) -> None:
        """Remove a connection from tracking.

        Args:
            connection_id: Connection to remove
        """
        self._connection_entity_states.pop(connection_id, None)
        self._bandwidth_manager.remove_connection(connection_id)

    def finalize_destroys(self) -> None:
        """Finalize pending destroy operations.

        Called after destroy notifications have been sent to all connections.
        """
        for guid_value in self._pending_destroys:
            self._entities.pop(guid_value, None)
        self._pending_destroys.clear()

    def update(self) -> None:
        """Periodic update for the replication manager.

        Call this each frame/tick to process queued operations.
        """
        # Process any finalized destroys
        destroy_guids = []
        for guid_value, replicated in list(self._entities.items()):
            if replicated.state == EntityState.PENDING_DESTROY:
                # Check if all connections have been notified
                all_notified = True
                for conn_states in self._connection_entity_states.values():
                    if conn_states.get(guid_value) != EntityState.PENDING_DESTROY:
                        all_notified = False
                        break
                if all_notified:
                    destroy_guids.append(guid_value)

        for guid_value in destroy_guids:
            self._entities.pop(guid_value, None)
            self._pending_destroys = [g for g in self._pending_destroys if g != guid_value]

    def _extract_properties(self, replicated: ReplicatedEntity) -> None:
        """Extract replicated properties from an entity.

        Args:
            replicated: The ReplicatedEntity to configure
        """
        entity = replicated.entity

        # Look for _networked_fields or similar metadata
        if hasattr(entity, '__networked_fields__'):
            for field_name, field_info in entity.__networked_fields__.items():
                value = getattr(entity, field_name, None)
                prop = ReplicatedProperty(
                    name=field_name,
                    value=value,
                    value_type=type(value) if value is not None else object,
                    condition=field_info.get('condition', ReplicationCondition.ON_CHANGE),
                    notify_mode=field_info.get('notify', ChangeNotifyMode.NONE),
                    priority=field_info.get('priority', 1)
                )
                replicated.properties.add_property(prop)

        # Also check for explicit properties dict
        elif hasattr(entity, '_replicated_properties'):
            for name, prop in entity._replicated_properties.items():
                replicated.properties.add_property(prop)

    def _collect_spawns(
        self,
        viewer: Any,
        connection_id: int,
        connection_states: dict[int, EntityState]
    ) -> bytes:
        """Collect spawn data for entities new to this connection."""
        parts = []

        for guid_value in list(self._pending_spawns):
            replicated = self._entities.get(guid_value)
            if replicated is None:
                continue

            # Check if already visible to this connection
            if guid_value in connection_states:
                continue

            # Check relevancy
            result = self._relevancy_manager.check_relevant(replicated.entity, viewer)
            if not result.is_relevant:
                continue

            # Check if owner
            viewer_id = getattr(viewer, 'player_id', None) or id(viewer)
            is_owner = replicated.owner_id == viewer_id

            # Serialize spawn
            spawn_bytes = self._serialize_spawn(replicated, is_owner)
            parts.append(spawn_bytes)

            # Mark as visible
            connection_states[guid_value] = EntityState.ACTIVE

        # Remove from pending spawns if sent to all connections
        self._pending_spawns = [
            g for g in self._pending_spawns
            if any(
                g not in states
                for states in self._connection_entity_states.values()
            )
        ]

        return b''.join(parts)

    def _collect_updates(
        self,
        viewer: Any,
        connection_id: int,
        connection_states: dict[int, EntityState]
    ) -> bytes:
        """Collect update data for visible dirty entities."""
        parts = []

        # Get viewer info
        viewer_id = getattr(viewer, 'player_id', None) or id(viewer)

        # Queue entities for bandwidth allocation
        for guid_value, replicated in self._entities.items():
            if replicated.state != EntityState.ACTIVE:
                continue
            if guid_value not in connection_states:
                continue
            if connection_states[guid_value] != EntityState.ACTIVE:
                continue

            # Check relevancy
            result = self._relevancy_manager.check_relevant(replicated.entity, viewer)
            if not result.is_relevant:
                # Entity became irrelevant - could close channel here
                continue

            # Check if dirty
            if not replicated.is_dirty():
                continue

            # Queue for bandwidth allocation
            self._bandwidth_manager.queue_entity(
                connection_id,
                replicated,
                guid_value,
                replicated.priority * result.priority,
                estimated_size=_config.ESTIMATED_UPDATE_SIZE
            )

        # Allocate bandwidth and get entities to send
        to_send = self._bandwidth_manager.allocate(connection_id)

        for replicated, guid_value in to_send:
            if not isinstance(replicated, ReplicatedEntity):
                replicated = self._entities.get(guid_value)
            if replicated is None:
                continue

            is_owner = replicated.owner_id == viewer_id
            update_bytes = self._serialize_update(replicated, is_owner)
            if update_bytes:
                parts.append(update_bytes)
                replicated.mark_replicated()

        return b''.join(parts)

    def _collect_destroys(
        self,
        connection_id: int,
        connection_states: dict[int, EntityState]
    ) -> bytes:
        """Collect destroy notifications."""
        parts = []

        for guid_value in self._pending_destroys:
            if guid_value not in connection_states:
                continue
            if connection_states[guid_value] == EntityState.PENDING_DESTROY:
                continue

            # Serialize destroy
            destroy_bytes = struct.pack('<BI', PACKET_DESTROY, guid_value)
            parts.append(destroy_bytes)

            # Mark as destroyed
            connection_states[guid_value] = EntityState.PENDING_DESTROY

        return b''.join(parts)

    def _serialize_spawn(self, replicated: ReplicatedEntity, is_owner: bool) -> bytes:
        """Serialize entity spawn packet."""
        parts = [
            struct.pack('<B', PACKET_SPAWN),
            struct.pack('<I', replicated.guid.value),
            struct.pack('<B', 1 if is_owner else 0),
        ]

        # Serialize all properties for initial state
        props_data = replicated.properties.serialize_all()
        parts.append(struct.pack('<H', len(props_data)))
        parts.append(props_data)

        return b''.join(parts)

    def _serialize_update(self, replicated: ReplicatedEntity, is_owner: bool) -> bytes:
        """Serialize entity update packet."""
        # Only serialize dirty properties
        props_data = replicated.properties.serialize_dirty(None, is_owner)

        # Skip if no properties to send
        prop_count = struct.unpack('<H', props_data[:2])[0]
        if prop_count == 0:
            return b''

        parts = [
            struct.pack('<B', PACKET_UPDATE),
            struct.pack('<I', replicated.guid.value),
            struct.pack('<H', len(props_data)),
            props_data,
        ]

        return b''.join(parts)

    def _apply_spawn(self, data: bytes) -> int:
        """Apply a spawn packet."""
        offset = 0

        guid_value = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4

        is_owner = struct.unpack('<B', data[offset:offset+1])[0] != 0
        offset += 1

        props_len = struct.unpack('<H', data[offset:offset+2])[0]
        offset += 2

        props_data = data[offset:offset+props_len]
        offset += props_len

        # Create entity via callback if set
        guid = NetGUID(guid_value)
        entity = None

        if self._on_spawn_callback:
            # Callback should create the entity
            self._on_spawn_callback(None, guid)
            entity = self._guid_manager.get_entity(guid)

        if entity is None:
            # Create placeholder if no callback - use a class so it can be weakref'd
            class PlaceholderEntity:
                __slots__ = ('guid', 'properties', '__weakref__')
                def __init__(self, g, p):
                    self.guid = g
                    self.properties = p
            entity = PlaceholderEntity(guid_value, {})

        # Register entity
        if guid_value not in self._entities:
            replicated = ReplicatedEntity(
                entity=entity,
                guid=guid,
                state=EntityState.ACTIVE,
                owner_id=id(self) if is_owner else None
            )
            self._entities[guid_value] = replicated
            self._guid_manager.import_guid(entity, guid)

        # Apply properties
        replicated = self._entities[guid_value]
        replicated.properties.deserialize(props_data)

        return offset

    def _apply_update(self, data: bytes) -> int:
        """Apply an update packet."""
        offset = 0

        guid_value = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4

        props_len = struct.unpack('<H', data[offset:offset+2])[0]
        offset += 2

        props_data = data[offset:offset+props_len]
        offset += props_len

        # Find entity
        replicated = self._entities.get(guid_value)
        if replicated:
            replicated.properties.deserialize(props_data)

        return offset

    def _apply_destroy(self, data: bytes) -> int:
        """Apply a destroy packet."""
        guid_value = struct.unpack('<I', data[:4])[0]

        replicated = self._entities.pop(guid_value, None)
        if replicated:
            self._guid_manager.release_guid(guid_value)
            if self._on_destroy_callback:
                self._on_destroy_callback(replicated.guid)

        return 4

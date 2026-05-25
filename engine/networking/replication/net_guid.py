"""Network GUID system for unique identification of networked objects.

Net GUIDs are 32-bit identifiers with a server/client prefix for uniqueness
across the network. They enable reliable object referencing for replication.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional
from weakref import WeakValueDictionary

from ..config import get_config

_logger = logging.getLogger(__name__)

# Get config instance
_config = get_config()


class GUIDAuthority(IntEnum):
    """Authority prefix for GUID allocation."""
    SERVER = 0  # Server-allocated GUIDs: 0x0000_0000 - 0x7FFF_FFFF
    CLIENT = 1  # Client-allocated GUIDs: 0x8000_0000 - 0xFFFF_FFFF


# Reserved GUID values - imported from config for backwards compatibility
INVALID_GUID = _config.INVALID_GUID
NULL_GUID = _config.NULL_GUID

# GUID ranges - imported from config for backwards compatibility
SERVER_GUID_START = _config.SERVER_GUID_START
SERVER_GUID_MAX = _config.SERVER_GUID_MAX
CLIENT_GUID_START = _config.CLIENT_GUID_START
CLIENT_GUID_MAX = _config.CLIENT_GUID_MAX

# Client ID encoding (bits 16-30 for client ID in client-allocated GUIDs)
CLIENT_ID_SHIFT = _config.CLIENT_ID_SHIFT
CLIENT_ID_MASK = _config.CLIENT_ID_MASK


@dataclass(slots=True)
class NetGUID:
    """Unique network identifier for replicated objects.

    Format (32-bit):
        - Bit 31: Authority flag (0=server, 1=client)
        - Bits 16-30: Client ID (for client-allocated GUIDs)
        - Bits 0-15: Sequence number within allocator

    For server-allocated GUIDs, bits 0-30 form a single sequence.

    Attributes:
        value: The raw 32-bit GUID value
    """
    value: int

    def __post_init__(self) -> None:
        """Validate GUID value is within valid range."""
        if not 0 <= self.value <= _config.GUID_32BIT_MAX:
            raise ValueError(f"GUID value must be 32-bit: {self.value}")

    @property
    def is_valid(self) -> bool:
        """Check if this GUID is valid (not null or invalid marker)."""
        return self.value != INVALID_GUID and self.value != NULL_GUID

    @property
    def authority(self) -> GUIDAuthority:
        """Get the authority that allocated this GUID."""
        if self.value & _config.GUID_AUTHORITY_BIT:
            return GUIDAuthority.CLIENT
        return GUIDAuthority.SERVER

    @property
    def client_id(self) -> Optional[int]:
        """Get the client ID for client-allocated GUIDs.

        Returns:
            Client ID (0-32767) for client GUIDs, None for server GUIDs.
        """
        if self.authority == GUIDAuthority.CLIENT:
            return (self.value & CLIENT_ID_MASK) >> CLIENT_ID_SHIFT
        return None

    def __hash__(self) -> int:
        return self.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NetGUID):
            return self.value == other.value
        if isinstance(other, int):
            return self.value == other
        return NotImplemented

    def __repr__(self) -> str:
        auth = "S" if self.authority == GUIDAuthority.SERVER else f"C{self.client_id}"
        return f"NetGUID({auth}:{self.value:08X})"

    def serialize(self) -> bytes:
        """Serialize GUID to bytes for network transmission."""
        return self.value.to_bytes(4, 'little')

    @classmethod
    def deserialize(cls, data: bytes) -> NetGUID:
        """Deserialize GUID from network bytes."""
        if len(data) < 4:
            raise ValueError("Insufficient data for GUID deserialization")
        value = int.from_bytes(data[:4], 'little')
        return cls(value)

    @classmethod
    def null(cls) -> NetGUID:
        """Create a null GUID."""
        return cls(NULL_GUID)

    @classmethod
    def invalid(cls) -> NetGUID:
        """Create an invalid GUID marker."""
        return cls(INVALID_GUID)


class NetGUIDManager:
    """Manages allocation and tracking of network GUIDs.

    Thread-safe manager for assigning unique network identifiers to entities.
    Supports both server and client authority modes.

    Attributes:
        authority: Whether this manager allocates as server or client
        client_id: Client ID (0-32767) for client-mode allocation
    """
    __slots__ = (
        '_authority', '_client_id', '_next_guid', '_guid_to_entity',
        '_entity_to_guid', '_free_guids', '_lock'
    )

    def __init__(
        self,
        authority: GUIDAuthority = GUIDAuthority.SERVER,
        client_id: int = 0
    ):
        """Initialize the GUID manager.

        Args:
            authority: Server or client allocation mode
            client_id: Client ID for client-mode (0-32767)
        """
        self._authority = authority
        self._client_id = client_id

        # Initialize sequence based on authority
        if authority == GUIDAuthority.SERVER:
            self._next_guid = SERVER_GUID_START
        else:
            if not 0 <= client_id <= _config.MAX_CLIENT_ID:
                raise ValueError(f"Client ID must be 0-{_config.MAX_CLIENT_ID}: {client_id}")
            base = CLIENT_GUID_START | (client_id << CLIENT_ID_SHIFT)
            self._next_guid = base

        # GUID <-> Entity mappings
        # Use WeakValueDictionary to avoid preventing entity garbage collection
        self._guid_to_entity: WeakValueDictionary[int, Any] = WeakValueDictionary()
        self._entity_to_guid: dict[int, NetGUID] = {}  # entity id -> GUID

        # Recycled GUIDs (for reuse after release)
        self._free_guids: list[int] = []

        # Thread safety
        self._lock = threading.Lock()

    @property
    def authority(self) -> GUIDAuthority:
        """Get the allocation authority mode."""
        return self._authority

    @property
    def client_id(self) -> int:
        """Get the client ID (for client-mode managers)."""
        return self._client_id

    def assign_guid(self, entity: Any) -> NetGUID:
        """Assign a network GUID to an entity.

        If the entity already has a GUID, returns the existing one.
        Otherwise allocates a new GUID from the pool.

        Args:
            entity: The entity to assign a GUID to

        Returns:
            The assigned NetGUID

        Raises:
            RuntimeError: If GUID pool is exhausted
        """
        entity_id = id(entity)

        with self._lock:
            # Check if entity already has a GUID
            existing = self._entity_to_guid.get(entity_id)
            if existing is not None:
                return existing

            # Allocate new GUID
            if self._free_guids:
                # Reuse recycled GUID
                guid_value = self._free_guids.pop()
            else:
                # Allocate fresh GUID
                guid_value = self._next_guid
                self._next_guid += 1

                # Check for overflow
                max_guid = (
                    SERVER_GUID_MAX if self._authority == GUIDAuthority.SERVER
                    else CLIENT_GUID_MAX
                )
                if self._next_guid > max_guid:
                    raise RuntimeError("GUID pool exhausted")

            guid = NetGUID(guid_value)

            # Register mappings
            self._guid_to_entity[guid_value] = entity
            self._entity_to_guid[entity_id] = guid

            return guid

    def get_entity(self, guid: NetGUID | int) -> Optional[Any]:
        """Retrieve an entity by its GUID.

        Args:
            guid: The NetGUID or raw GUID value

        Returns:
            The entity if found, None otherwise
        """
        guid_value = guid.value if isinstance(guid, NetGUID) else guid

        with self._lock:
            return self._guid_to_entity.get(guid_value)

    def get_guid(self, entity: Any) -> Optional[NetGUID]:
        """Get the GUID for an entity.

        Args:
            entity: The entity to look up

        Returns:
            The NetGUID if registered, None otherwise
        """
        entity_id = id(entity)

        with self._lock:
            return self._entity_to_guid.get(entity_id)

    def release_guid(self, guid: NetGUID | int) -> bool:
        """Release a GUID, making it available for reuse.

        Args:
            guid: The GUID to release

        Returns:
            True if GUID was released, False if not found
        """
        guid_value = guid.value if isinstance(guid, NetGUID) else guid

        with self._lock:
            entity = self._guid_to_entity.pop(guid_value, None)
            if entity is None:
                return False

            entity_id = id(entity)
            self._entity_to_guid.pop(entity_id, None)

            # Recycle the GUID for reuse
            self._free_guids.append(guid_value)

            return True

    def release_entity(self, entity: Any) -> bool:
        """Release an entity's GUID.

        Args:
            entity: The entity to release

        Returns:
            True if released, False if entity had no GUID
        """
        entity_id = id(entity)

        with self._lock:
            guid = self._entity_to_guid.pop(entity_id, None)
            if guid is None:
                return False

            self._guid_to_entity.pop(guid.value, None)
            self._free_guids.append(guid.value)

            return True

    def has_guid(self, guid: NetGUID | int) -> bool:
        """Check if a GUID is currently assigned.

        Args:
            guid: The GUID to check

        Returns:
            True if GUID is in use
        """
        guid_value = guid.value if isinstance(guid, NetGUID) else guid

        with self._lock:
            return guid_value in self._guid_to_entity

    def is_registered(self, entity: Any) -> bool:
        """Check if an entity has a GUID.

        Args:
            entity: The entity to check

        Returns:
            True if entity has a GUID
        """
        entity_id = id(entity)

        with self._lock:
            return entity_id in self._entity_to_guid

    def get_all_guids(self) -> list[NetGUID]:
        """Get all currently assigned GUIDs.

        Returns:
            List of all active GUIDs
        """
        with self._lock:
            return [NetGUID(v) for v in self._guid_to_entity.keys()]

    def count(self) -> int:
        """Get the number of assigned GUIDs.

        Returns:
            Number of active GUIDs
        """
        with self._lock:
            return len(self._guid_to_entity)

    def clear(self) -> None:
        """Clear all GUID assignments.

        Warning: This does not notify entities of GUID removal.
        """
        with self._lock:
            self._guid_to_entity.clear()
            self._entity_to_guid.clear()
            self._free_guids.clear()

            # Reset sequence
            if self._authority == GUIDAuthority.SERVER:
                self._next_guid = SERVER_GUID_START
            else:
                base = CLIENT_GUID_START | (self._client_id << CLIENT_ID_SHIFT)
                self._next_guid = base

    def import_guid(self, entity: Any, guid: NetGUID) -> bool:
        """Import an externally-assigned GUID for an entity.

        Used when receiving replicated entities from remote sources.

        Args:
            entity: The local entity
            guid: The GUID assigned by remote authority

        Returns:
            True if imported successfully, False if GUID already in use
        """
        entity_id = id(entity)

        with self._lock:
            # Check if GUID is already assigned
            if guid.value in self._guid_to_entity:
                return False

            # Check if entity already has a GUID
            if entity_id in self._entity_to_guid:
                # Remove old mapping
                old_guid = self._entity_to_guid[entity_id]
                self._guid_to_entity.pop(old_guid.value, None)

            # Create new mappings
            self._guid_to_entity[guid.value] = entity
            self._entity_to_guid[entity_id] = guid

            return True

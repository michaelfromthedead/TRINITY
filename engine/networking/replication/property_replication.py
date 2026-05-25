"""Property replication system for network state synchronization.

Handles per-property change detection, serialization, and replication
conditions for fine-grained network updates.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generic, Optional, TypeVar

from ..config import get_config

_logger = logging.getLogger(__name__)

# Get config instance
_config = get_config()

T = TypeVar('T')


class ReplicationCondition(Enum):
    """Conditions that determine when a property should be replicated.

    Based on Unreal Engine's replication conditions for optimized bandwidth.
    """
    ALWAYS = auto()        # Replicate every tick
    ON_CHANGE = auto()     # Only replicate when value changes (default)
    INITIAL_ONLY = auto()  # Only replicate once on spawn
    OWNER_ONLY = auto()    # Only replicate to the owning client
    SKIP_OWNER = auto()    # Replicate to everyone except owner
    CUSTOM = auto()        # Use custom predicate function


class ChangeNotifyMode(Enum):
    """How property changes are notified to recipients.

    Controls callback behavior when replicated values are received.
    """
    NONE = auto()          # No callback on receive
    REP_NOTIFY = auto()    # Call OnRep_PropertyName() with new value
    WITH_PREVIOUS = auto() # Call OnRep with both old and new values


# Type serializers for common types
_SERIALIZERS: dict[type, tuple[Callable[[Any], bytes], Callable[[bytes], tuple[Any, int]]]] = {}


def register_serializer(
    type_: type,
    serialize: Callable[[Any], bytes],
    deserialize: Callable[[bytes], tuple[Any, int]]
) -> None:
    """Register a custom serializer for a type.

    Args:
        type_: The type to register
        serialize: Function (value) -> bytes
        deserialize: Function (bytes) -> (value, bytes_consumed)
    """
    _SERIALIZERS[type_] = (serialize, deserialize)


# Default serializers for built-in types
def _ser_int(v: int) -> bytes:
    return struct.pack('<i', v)

def _deser_int(b: bytes) -> tuple[int, int]:
    return struct.unpack('<i', b[:4])[0], 4

def _ser_float(v: float) -> bytes:
    return struct.pack('<f', v)

def _deser_float(b: bytes) -> tuple[float, int]:
    return struct.unpack('<f', b[:4])[0], 4

def _ser_bool(v: bool) -> bytes:
    return struct.pack('<B', 1 if v else 0)

def _deser_bool(b: bytes) -> tuple[bool, int]:
    return struct.unpack('<B', b[:1])[0] != 0, 1

def _ser_str(v: str) -> bytes:
    encoded = v.encode('utf-8')
    return struct.pack('<H', len(encoded)) + encoded

def _deser_str(b: bytes) -> tuple[str, int]:
    length = struct.unpack('<H', b[:2])[0]
    return b[2:2+length].decode('utf-8'), 2 + length

def _ser_bytes(v: bytes) -> bytes:
    return struct.pack('<I', len(v)) + v

def _deser_bytes(b: bytes) -> tuple[bytes, int]:
    length = struct.unpack('<I', b[:4])[0]
    return b[4:4+length], 4 + length


# Register default serializers
register_serializer(int, _ser_int, _deser_int)
register_serializer(float, _ser_float, _deser_float)
register_serializer(bool, _ser_bool, _deser_bool)
register_serializer(str, _ser_str, _deser_str)
register_serializer(bytes, _ser_bytes, _deser_bytes)


@dataclass
class ReplicatedProperty(Generic[T]):
    """A network-replicated property with change tracking.

    Tracks property changes and handles serialization for network transmission.
    Supports conditional replication and change notifications.

    Attributes:
        name: Property identifier
        value: Current property value
        value_type: Python type for serialization
        condition: When to replicate this property
        notify_mode: How to notify on received changes
        priority: Replication priority (higher = more important)
        dirty: Whether property has changed since last replication
    """
    name: str
    value: T
    value_type: type
    condition: ReplicationCondition = ReplicationCondition.ON_CHANGE
    notify_mode: ChangeNotifyMode = ChangeNotifyMode.NONE
    priority: int = _config.DEFAULT_REPLICATION_PRIORITY
    dirty: bool = field(default=False, repr=False)
    _previous_value: Optional[T] = field(default=None, repr=False)
    _custom_predicate: Optional[Callable[[Any, Any], bool]] = field(
        default=None, repr=False
    )
    _on_rep_callback: Optional[Callable[[T, Optional[T]], None]] = field(
        default=None, repr=False
    )
    _initial_sent: bool = field(default=False, repr=False)

    def set_value(self, new_value: T) -> None:
        """Set the property value and mark dirty if changed.

        Args:
            new_value: The new property value
        """
        if self.value != new_value:
            self._previous_value = self.value
            self.value = new_value
            self.dirty = True

    def get_value(self) -> T:
        """Get the current property value."""
        return self.value

    def mark_clean(self) -> None:
        """Mark property as clean (replicated)."""
        self.dirty = False
        self._initial_sent = True

    def mark_dirty(self) -> None:
        """Force property to be dirty."""
        self.dirty = True

    def should_replicate(self, recipient: Optional[Any] = None, is_owner: bool = False) -> bool:
        """Determine if property should be replicated to recipient.

        Args:
            recipient: The target recipient (connection/player)
            is_owner: Whether recipient owns this entity

        Returns:
            True if property should be replicated
        """
        # Check condition-based filtering
        match self.condition:
            case ReplicationCondition.ALWAYS:
                return True
            case ReplicationCondition.ON_CHANGE:
                return self.dirty
            case ReplicationCondition.INITIAL_ONLY:
                return not self._initial_sent
            case ReplicationCondition.OWNER_ONLY:
                return is_owner and (self.dirty or not self._initial_sent)
            case ReplicationCondition.SKIP_OWNER:
                return not is_owner and (self.dirty or not self._initial_sent)
            case ReplicationCondition.CUSTOM:
                if self._custom_predicate:
                    return self._custom_predicate(self, recipient)
                return self.dirty
        return False

    def set_custom_predicate(self, predicate: Callable[[Any, Any], bool]) -> None:
        """Set custom replication predicate for CUSTOM condition.

        Args:
            predicate: Function (property, recipient) -> bool
        """
        self._custom_predicate = predicate
        self.condition = ReplicationCondition.CUSTOM

    def set_on_rep_callback(self, callback: Callable[[T, Optional[T]], None]) -> None:
        """Set callback for when replicated value is received.

        Args:
            callback: Function (new_value, old_value) called on receive
        """
        self._on_rep_callback = callback

    def on_rep_notify(self, new_value: T, old_value: Optional[T] = None) -> None:
        """Called when a replicated value is received.

        Args:
            new_value: The received value
            old_value: Previous value (if notify_mode is WITH_PREVIOUS)
        """
        if self._on_rep_callback and self.notify_mode != ChangeNotifyMode.NONE:
            if self.notify_mode == ChangeNotifyMode.WITH_PREVIOUS:
                self._on_rep_callback(new_value, old_value)
            else:
                self._on_rep_callback(new_value, None)

    def serialize(self) -> bytes:
        """Serialize property value to bytes for network transmission.

        Returns:
            Serialized property data

        Raises:
            TypeError: If no serializer registered for value type
        """
        serializer = _SERIALIZERS.get(self.value_type)
        if serializer is None:
            # Try generic serialization via pickle as fallback
            import pickle
            data = pickle.dumps(self.value)
            return struct.pack('<I', len(data)) + data

        return serializer[0](self.value)

    def deserialize(self, data: bytes) -> int:
        """Deserialize property value from network bytes.

        Args:
            data: Network bytes to deserialize

        Returns:
            Number of bytes consumed

        Raises:
            TypeError: If no serializer registered for value type
        """
        old_value = self.value

        serializer = _SERIALIZERS.get(self.value_type)
        if serializer is None:
            # Fallback to pickle
            import pickle
            length = struct.unpack('<I', data[:4])[0]
            self.value = pickle.loads(data[4:4+length])
            bytes_consumed = 4 + length
        else:
            self.value, bytes_consumed = serializer[1](data)

        # Trigger rep notify
        self.on_rep_notify(self.value, old_value)

        return bytes_consumed

    def serialize_delta(self, baseline: Optional[T] = None) -> Optional[bytes]:
        """Serialize only if changed from baseline.

        Args:
            baseline: Previous known value for delta compression

        Returns:
            Serialized bytes if changed, None if unchanged
        """
        if baseline is not None and self.value == baseline:
            return None
        return self.serialize()


@dataclass
class PropertyReplicationGroup:
    """A group of related replicated properties.

    Groups properties for batch serialization and priority management.
    """
    name: str
    properties: dict[str, ReplicatedProperty] = field(default_factory=dict)
    priority: int = _config.DEFAULT_REPLICATION_PRIORITY

    def add_property(self, prop: ReplicatedProperty) -> None:
        """Add a property to the group.

        Args:
            prop: The property to add
        """
        self.properties[prop.name] = prop

    def remove_property(self, name: str) -> Optional[ReplicatedProperty]:
        """Remove a property from the group.

        Args:
            name: Property name to remove

        Returns:
            The removed property, or None if not found
        """
        return self.properties.pop(name, None)

    def get_property(self, name: str) -> Optional[ReplicatedProperty]:
        """Get a property by name.

        Args:
            name: Property name

        Returns:
            The property, or None if not found
        """
        return self.properties.get(name)

    def get_dirty_properties(self) -> list[ReplicatedProperty]:
        """Get all properties that need replication.

        Returns:
            List of dirty properties
        """
        return [p for p in self.properties.values() if p.dirty]

    def mark_all_clean(self) -> None:
        """Mark all properties as clean."""
        for prop in self.properties.values():
            prop.mark_clean()

    def mark_all_dirty(self) -> None:
        """Mark all properties as dirty."""
        for prop in self.properties.values():
            prop.mark_dirty()

    def serialize_all(self) -> bytes:
        """Serialize all properties.

        Returns:
            Serialized property group data
        """
        parts = []
        for name, prop in self.properties.items():
            name_bytes = name.encode('utf-8')
            value_bytes = prop.serialize()
            parts.append(struct.pack('<B', len(name_bytes)) + name_bytes + value_bytes)

        count = len(parts)
        return struct.pack('<H', count) + b''.join(parts)

    def serialize_dirty(self, recipient: Any = None, is_owner: bool = False) -> bytes:
        """Serialize only properties that need replication.

        Args:
            recipient: Target recipient for filtering
            is_owner: Whether recipient owns this entity

        Returns:
            Serialized dirty property data
        """
        parts = []
        for name, prop in self.properties.items():
            if prop.should_replicate(recipient, is_owner):
                name_bytes = name.encode('utf-8')
                value_bytes = prop.serialize()
                parts.append(struct.pack('<B', len(name_bytes)) + name_bytes + value_bytes)

        count = len(parts)
        return struct.pack('<H', count) + b''.join(parts)

    def deserialize(self, data: bytes) -> int:
        """Deserialize property group from network bytes.

        Args:
            data: Network bytes

        Returns:
            Number of bytes consumed
        """
        offset = 0
        count = struct.unpack('<H', data[offset:offset+2])[0]
        offset += 2

        for _ in range(count):
            name_len = struct.unpack('<B', data[offset:offset+1])[0]
            offset += 1
            name = data[offset:offset+name_len].decode('utf-8')
            offset += name_len

            prop = self.properties.get(name)
            if prop:
                consumed = prop.deserialize(data[offset:])
                offset += consumed

        return offset


def create_replicated_property(
    name: str,
    initial_value: T,
    value_type: type | None = None,
    condition: ReplicationCondition = ReplicationCondition.ON_CHANGE,
    notify_mode: ChangeNotifyMode = ChangeNotifyMode.NONE,
    priority: int = _config.DEFAULT_REPLICATION_PRIORITY
) -> ReplicatedProperty[T]:
    """Factory function to create a replicated property.

    Args:
        name: Property identifier
        initial_value: Initial value
        value_type: Type override (auto-detected if None)
        condition: Replication condition
        notify_mode: Change notification mode
        priority: Replication priority

    Returns:
        Configured ReplicatedProperty instance
    """
    if value_type is None:
        value_type = type(initial_value)

    return ReplicatedProperty(
        name=name,
        value=initial_value,
        value_type=value_type,
        condition=condition,
        notify_mode=notify_mode,
        priority=priority
    )

"""Actor channel for per-entity replication streams.

Provides a dedicated communication channel for each replicated entity,
managing spawn, update, and destroy messages.
"""

from __future__ import annotations

import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from ..config import get_config
from .net_guid import NetGUID

_logger = logging.getLogger(__name__)

# Get config instance
_config = get_config()


class ChannelState(Enum):
    """State of an actor channel."""
    CLOSED = auto()      # Channel not open
    OPENING = auto()     # Spawn in progress
    OPEN = auto()        # Channel active
    CLOSING = auto()     # Close in progress


class ChannelCloseReason(Enum):
    """Reason for channel closure."""
    DESTROYED = auto()     # Entity was destroyed
    IRRELEVANT = auto()    # Entity became irrelevant
    DORMANT = auto()       # Entity entered dormancy
    CONNECTION_LOST = auto() # Connection was lost
    ERROR = auto()         # Protocol error


# Message types for actor channel - from config
MSG_SPAWN = _config.ACTOR_MSG_SPAWN
MSG_INITIAL_STATE = _config.ACTOR_MSG_INITIAL_STATE
MSG_DELTA_UPDATE = _config.ACTOR_MSG_DELTA_UPDATE
MSG_RPC = _config.ACTOR_MSG_RPC
MSG_CLOSE = _config.ACTOR_MSG_CLOSE
MSG_ACK = _config.ACTOR_MSG_ACK


@dataclass
class ChannelMessage:
    """Message sent through an actor channel.

    Attributes:
        msg_type: Type of message
        sequence: Sequence number for ordering
        data: Message payload
        reliable: Whether message requires acknowledgment
        timestamp: When message was created
    """
    msg_type: int
    sequence: int
    data: bytes
    reliable: bool = True
    timestamp: float = field(default_factory=time.time)

    def serialize(self) -> bytes:
        """Serialize message for transmission."""
        flags = 0x01 if self.reliable else 0x00
        header = struct.pack('<BIH', self.msg_type, self.sequence, flags)
        return header + struct.pack('<H', len(self.data)) + self.data

    @classmethod
    def deserialize(cls, data: bytes) -> tuple[ChannelMessage, int]:
        """Deserialize message from bytes.

        Returns:
            Tuple of (message, bytes_consumed)
        """
        msg_type, sequence, flags = struct.unpack('<BIH', data[:7])
        reliable = bool(flags & 0x01)
        payload_len = struct.unpack('<H', data[7:9])[0]
        payload = data[9:9+payload_len]

        msg = cls(
            msg_type=msg_type,
            sequence=sequence,
            data=payload,
            reliable=reliable
        )
        return msg, 9 + payload_len


@dataclass
class ActorChannel:
    """Per-entity replication channel.

    Manages the communication stream for a single replicated entity,
    handling reliable delivery and ordering.

    Attributes:
        guid: Network GUID of the entity
        connection_id: Associated connection
        state: Current channel state
    """
    guid: NetGUID
    connection_id: int
    state: ChannelState = ChannelState.CLOSED

    # Sequence numbers
    _send_sequence: int = field(default=0, repr=False)
    _recv_sequence: int = field(default=0, repr=False)
    _acked_sequence: int = field(default=0, repr=False)

    # Message queues
    _outgoing: list[ChannelMessage] = field(default_factory=list, repr=False)
    _pending_ack: dict[int, ChannelMessage] = field(default_factory=dict, repr=False)
    _incoming_buffer: dict[int, ChannelMessage] = field(default_factory=dict, repr=False)

    # Callbacks
    _on_state_change: Optional[Callable[[ChannelState], None]] = field(
        default=None, repr=False
    )
    _on_message: Optional[Callable[[ChannelMessage], None]] = field(
        default=None, repr=False
    )

    # Timing
    _open_time: float = field(default=0.0, repr=False)
    _last_send_time: float = field(default=0.0, repr=False)
    _last_recv_time: float = field(default=0.0, repr=False)

    def open(self) -> bool:
        """Open the channel.

        Returns:
            True if channel opened successfully
        """
        if self.state != ChannelState.CLOSED:
            return False

        self._set_state(ChannelState.OPENING)
        self._open_time = time.time()
        self._send_sequence = 0
        self._recv_sequence = 0
        self._acked_sequence = 0

        return True

    def close(self, reason: ChannelCloseReason = ChannelCloseReason.DESTROYED) -> bool:
        """Close the channel.

        Args:
            reason: Reason for closure

        Returns:
            True if close initiated
        """
        if self.state == ChannelState.CLOSED:
            return False

        self._set_state(ChannelState.CLOSING)

        # Send close message
        close_data = struct.pack('<B', reason.value)
        self._queue_message(MSG_CLOSE, close_data, reliable=True)

        return True

    def force_close(self) -> None:
        """Force immediate channel closure without notification."""
        self._set_state(ChannelState.CLOSED)
        self._outgoing.clear()
        self._pending_ack.clear()
        self._incoming_buffer.clear()

    def send_spawn(self, initial_state: bytes) -> bool:
        """Send spawn message with initial state.

        Args:
            initial_state: Serialized initial entity state

        Returns:
            True if message queued
        """
        if self.state != ChannelState.OPENING:
            return False

        # Send spawn message
        spawn_data = self.guid.serialize() + initial_state
        self._queue_message(MSG_SPAWN, spawn_data, reliable=True)

        return True

    def send_initial_state(self, state_data: bytes) -> bool:
        """Send initial state (separate from spawn).

        Args:
            state_data: Serialized state

        Returns:
            True if queued
        """
        if self.state not in (ChannelState.OPENING, ChannelState.OPEN):
            return False

        self._queue_message(MSG_INITIAL_STATE, state_data, reliable=True)
        return True

    def send_update(self, delta_data: bytes, reliable: bool = False) -> bool:
        """Send delta update.

        Args:
            delta_data: Serialized delta state
            reliable: Whether to use reliable delivery

        Returns:
            True if queued
        """
        if self.state != ChannelState.OPEN:
            return False

        self._queue_message(MSG_DELTA_UPDATE, delta_data, reliable=reliable)
        return True

    def send_rpc(self, rpc_data: bytes, reliable: bool = True) -> bool:
        """Send RPC through the channel.

        Args:
            rpc_data: Serialized RPC
            reliable: Whether to use reliable delivery

        Returns:
            True if queued
        """
        if self.state != ChannelState.OPEN:
            return False

        self._queue_message(MSG_RPC, rpc_data, reliable=reliable)
        return True

    def receive(self, message: ChannelMessage) -> None:
        """Process a received message.

        Args:
            message: The received message
        """
        self._last_recv_time = time.time()

        # Handle message based on type
        match message.msg_type:
            case 0x01:  # MSG_SPAWN
                self._handle_spawn(message)
            case 0x02:  # MSG_INITIAL_STATE
                self._handle_initial_state(message)
            case 0x03:  # MSG_DELTA_UPDATE
                self._handle_delta_update(message)
            case 0x04:  # MSG_RPC
                self._handle_rpc(message)
            case 0x05:  # MSG_CLOSE
                self._handle_close(message)
            case 0x06:  # MSG_ACK
                self._handle_ack(message)

        # Send ack for reliable messages
        if message.reliable:
            self._send_ack(message.sequence)

    def get_outgoing_messages(self) -> list[ChannelMessage]:
        """Get queued outgoing messages.

        Returns:
            List of messages to send
        """
        messages = self._outgoing.copy()
        self._outgoing.clear()

        # Track reliable messages for retransmission
        for msg in messages:
            if msg.reliable:
                self._pending_ack[msg.sequence] = msg
            self._last_send_time = time.time()

        return messages

    def process_ack(self, sequence: int) -> None:
        """Process acknowledgment for a sequence number.

        Args:
            sequence: Acknowledged sequence
        """
        self._pending_ack.pop(sequence, None)

        if sequence > self._acked_sequence:
            self._acked_sequence = sequence

        # If spawn was acked, transition to open
        if self.state == ChannelState.OPENING and self._acked_sequence > 0:
            self._set_state(ChannelState.OPEN)

        # If close was acked, transition to closed
        if self.state == ChannelState.CLOSING:
            if not self._pending_ack:
                self._set_state(ChannelState.CLOSED)

    def get_retransmit_messages(self, timeout: float = _config.DEFAULT_RETRANSMIT_TIMEOUT) -> list[ChannelMessage]:
        """Get messages that need retransmission.

        Args:
            timeout: Time after which to retransmit

        Returns:
            List of messages to retransmit
        """
        now = time.time()
        retransmit = []

        for seq, msg in list(self._pending_ack.items()):
            if now - msg.timestamp > timeout:
                # Update timestamp and add to retransmit
                msg.timestamp = now
                retransmit.append(msg)

        return retransmit

    def set_on_state_change(self, callback: Callable[[ChannelState], None]) -> None:
        """Set callback for state changes.

        Args:
            callback: Function called with new state
        """
        self._on_state_change = callback

    def set_on_message(self, callback: Callable[[ChannelMessage], None]) -> None:
        """Set callback for received messages.

        Args:
            callback: Function called with received message
        """
        self._on_message = callback

    @property
    def is_open(self) -> bool:
        """Check if channel is open and ready."""
        return self.state == ChannelState.OPEN

    @property
    def pending_reliable_count(self) -> int:
        """Get count of unacknowledged reliable messages."""
        return len(self._pending_ack)

    @property
    def time_since_last_send(self) -> float:
        """Get time since last message was sent."""
        if self._last_send_time == 0:
            return float('inf')
        return time.time() - self._last_send_time

    @property
    def time_since_last_recv(self) -> float:
        """Get time since last message was received."""
        if self._last_recv_time == 0:
            return float('inf')
        return time.time() - self._last_recv_time

    def _queue_message(
        self,
        msg_type: int,
        data: bytes,
        reliable: bool = True
    ) -> ChannelMessage:
        """Queue a message for sending.

        Args:
            msg_type: Message type
            data: Message payload
            reliable: Whether to use reliable delivery

        Returns:
            The queued message
        """
        self._send_sequence += 1
        msg = ChannelMessage(
            msg_type=msg_type,
            sequence=self._send_sequence,
            data=data,
            reliable=reliable
        )
        self._outgoing.append(msg)
        return msg

    def _send_ack(self, sequence: int) -> None:
        """Send acknowledgment for a sequence.

        Args:
            sequence: Sequence to acknowledge
        """
        ack_data = struct.pack('<I', sequence)
        self._queue_message(MSG_ACK, ack_data, reliable=False)

    def _set_state(self, new_state: ChannelState) -> None:
        """Update channel state.

        Args:
            new_state: New state
        """
        if self.state != new_state:
            self.state = new_state
            if self._on_state_change:
                self._on_state_change(new_state)

    def _handle_spawn(self, message: ChannelMessage) -> None:
        """Handle spawn message."""
        if self.state == ChannelState.CLOSED:
            self._set_state(ChannelState.OPEN)

        if self._on_message:
            self._on_message(message)

    def _handle_initial_state(self, message: ChannelMessage) -> None:
        """Handle initial state message."""
        if self._on_message:
            self._on_message(message)

    def _handle_delta_update(self, message: ChannelMessage) -> None:
        """Handle delta update message."""
        # Check ordering
        if message.sequence <= self._recv_sequence:
            # Old message, ignore
            return

        if message.sequence > self._recv_sequence + 1:
            # Out of order, buffer if reliable
            if message.reliable:
                self._incoming_buffer[message.sequence] = message
            return

        # Process in order
        self._recv_sequence = message.sequence
        if self._on_message:
            self._on_message(message)

        # Process buffered messages
        self._process_buffered_messages()

    def _handle_rpc(self, message: ChannelMessage) -> None:
        """Handle RPC message."""
        if self._on_message:
            self._on_message(message)

    def _handle_close(self, message: ChannelMessage) -> None:
        """Handle close message."""
        self._set_state(ChannelState.CLOSED)
        if self._on_message:
            self._on_message(message)

    def _handle_ack(self, message: ChannelMessage) -> None:
        """Handle acknowledgment message."""
        if len(message.data) >= 4:
            acked_seq = struct.unpack('<I', message.data[:4])[0]
            self.process_ack(acked_seq)

    def _process_buffered_messages(self) -> None:
        """Process any buffered messages that are now in order."""
        while True:
            next_seq = self._recv_sequence + 1
            if next_seq not in self._incoming_buffer:
                break

            msg = self._incoming_buffer.pop(next_seq)
            self._recv_sequence = next_seq
            if self._on_message:
                self._on_message(msg)


class ActorChannelManager:
    """Manages actor channels for all replicated entities.

    Coordinates channel creation, destruction, and message routing.
    """
    __slots__ = ('_channels', '_channels_by_connection')

    def __init__(self):
        """Initialize the channel manager."""
        # guid value -> {connection_id -> channel}
        self._channels: dict[int, dict[int, ActorChannel]] = {}
        # connection_id -> {guid value -> channel}
        self._channels_by_connection: dict[int, dict[int, ActorChannel]] = {}

    def open_channel(
        self,
        guid: NetGUID,
        connection_id: int
    ) -> ActorChannel:
        """Open a channel for an entity to a connection.

        Args:
            guid: Entity GUID
            connection_id: Target connection

        Returns:
            The opened channel
        """
        guid_value = guid.value

        # Initialize tracking dicts if needed
        if guid_value not in self._channels:
            self._channels[guid_value] = {}
        if connection_id not in self._channels_by_connection:
            self._channels_by_connection[connection_id] = {}

        # Check for existing channel
        existing = self._channels[guid_value].get(connection_id)
        if existing and existing.state != ChannelState.CLOSED:
            return existing

        # Create new channel
        channel = ActorChannel(guid=guid, connection_id=connection_id)
        channel.open()

        # Register
        self._channels[guid_value][connection_id] = channel
        self._channels_by_connection[connection_id][guid_value] = channel

        return channel

    def close_channel(
        self,
        guid: NetGUID | int,
        connection_id: int,
        reason: ChannelCloseReason = ChannelCloseReason.DESTROYED
    ) -> bool:
        """Close a channel.

        Args:
            guid: Entity GUID
            connection_id: Connection ID
            reason: Close reason

        Returns:
            True if channel was closed
        """
        guid_value = guid.value if isinstance(guid, NetGUID) else guid

        channel = self.get_channel(guid_value, connection_id)
        if channel:
            return channel.close(reason)
        return False

    def close_all_for_entity(
        self,
        guid: NetGUID | int,
        reason: ChannelCloseReason = ChannelCloseReason.DESTROYED
    ) -> int:
        """Close all channels for an entity.

        Args:
            guid: Entity GUID
            reason: Close reason

        Returns:
            Number of channels closed
        """
        guid_value = guid.value if isinstance(guid, NetGUID) else guid

        channels = self._channels.get(guid_value, {})
        count = 0
        for channel in channels.values():
            if channel.close(reason):
                count += 1
        return count

    def close_all_for_connection(
        self,
        connection_id: int,
        reason: ChannelCloseReason = ChannelCloseReason.CONNECTION_LOST
    ) -> int:
        """Close all channels for a connection.

        Args:
            connection_id: Connection ID
            reason: Close reason

        Returns:
            Number of channels closed
        """
        channels = self._channels_by_connection.get(connection_id, {})
        count = 0
        for channel in channels.values():
            if channel.close(reason):
                count += 1
        return count

    def get_channel(
        self,
        guid: NetGUID | int,
        connection_id: int
    ) -> Optional[ActorChannel]:
        """Get a channel for an entity and connection.

        Args:
            guid: Entity GUID
            connection_id: Connection ID

        Returns:
            The channel or None
        """
        guid_value = guid.value if isinstance(guid, NetGUID) else guid
        return self._channels.get(guid_value, {}).get(connection_id)

    def get_channels_for_entity(
        self,
        guid: NetGUID | int
    ) -> list[ActorChannel]:
        """Get all channels for an entity.

        Args:
            guid: Entity GUID

        Returns:
            List of channels
        """
        guid_value = guid.value if isinstance(guid, NetGUID) else guid
        return list(self._channels.get(guid_value, {}).values())

    def get_channels_for_connection(
        self,
        connection_id: int
    ) -> list[ActorChannel]:
        """Get all channels for a connection.

        Args:
            connection_id: Connection ID

        Returns:
            List of channels
        """
        return list(self._channels_by_connection.get(connection_id, {}).values())

    def cleanup_closed_channels(self) -> int:
        """Remove closed channels from tracking.

        Returns:
            Number of channels removed
        """
        removed = 0

        for guid_value in list(self._channels.keys()):
            conn_channels = self._channels[guid_value]
            for conn_id in list(conn_channels.keys()):
                channel = conn_channels[conn_id]
                if channel.state == ChannelState.CLOSED:
                    del conn_channels[conn_id]
                    if conn_id in self._channels_by_connection:
                        self._channels_by_connection[conn_id].pop(guid_value, None)
                    removed += 1

            if not conn_channels:
                del self._channels[guid_value]

        return removed

    def get_all_outgoing_messages(self) -> dict[int, list[tuple[NetGUID, ChannelMessage]]]:
        """Get all outgoing messages grouped by connection.

        Returns:
            Dict mapping connection_id to list of (guid, message) tuples
        """
        result: dict[int, list[tuple[NetGUID, ChannelMessage]]] = {}

        for conn_id, channels in self._channels_by_connection.items():
            messages = []
            for guid_value, channel in channels.items():
                for msg in channel.get_outgoing_messages():
                    messages.append((channel.guid, msg))
            if messages:
                result[conn_id] = messages

        return result

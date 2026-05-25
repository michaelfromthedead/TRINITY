"""RPC channel for network transmission of remote procedure calls.

Handles serialization, ordering, and reliable delivery of RPC messages.
"""

from __future__ import annotations

import logging
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional

from ..config import get_config
from .rpc_manager import RPCInfo, RPCAuthority, RPCReliability

_logger = logging.getLogger(__name__)

# Get config instance
_config = get_config()


class RPCChannelState(IntEnum):
    """State of the RPC channel."""
    CLOSED = 0
    OPEN = 1
    DRAINING = 2  # Closing, waiting for pending ACKs


# RPC message types - from config
RPC_MSG_CALL = _config.RPC_MSG_CALL
RPC_MSG_ACK = _config.RPC_MSG_ACK
RPC_MSG_NACK = _config.RPC_MSG_NACK
RPC_MSG_BATCH = _config.RPC_MSG_BATCH


@dataclass(slots=True)
class RPCMessage:
    """An RPC message for network transmission.

    Attributes:
        msg_type: Message type
        rpc_hash: Hash of RPC name
        sequence: Sequence number
        payload: Serialized arguments
        reliable: Whether message requires ACK
        timestamp: When message was created
    """
    msg_type: int
    rpc_hash: int
    sequence: int
    payload: bytes
    reliable: bool = True
    timestamp: float = field(default_factory=time.time)

    def serialize(self) -> bytes:
        """Serialize message for transmission."""
        flags = 0x01 if self.reliable else 0x00
        header = struct.pack(
            '<BIIB',
            self.msg_type,
            self.rpc_hash,
            self.sequence,
            flags
        )
        payload_len = struct.pack('<H', len(self.payload))
        return header + payload_len + self.payload

    @classmethod
    def deserialize(cls, data: bytes) -> tuple[RPCMessage, int]:
        """Deserialize message from bytes.

        Returns:
            Tuple of (message, bytes_consumed)
        """
        msg_type, rpc_hash, sequence, flags = struct.unpack(
            '<BIIB', data[:10]
        )
        reliable = bool(flags & 0x01)

        payload_len = struct.unpack('<H', data[10:12])[0]
        payload = data[12:12+payload_len]

        msg = cls(
            msg_type=msg_type,
            rpc_hash=rpc_hash,
            sequence=sequence,
            payload=payload,
            reliable=reliable
        )
        return msg, 12 + payload_len


@dataclass
class RPCChannel:
    """Channel for RPC transmission between endpoints.

    Manages ordering, reliability, and batching of RPC messages.

    Attributes:
        connection_id: Associated connection
        state: Channel state
    """
    connection_id: int
    state: RPCChannelState = RPCChannelState.CLOSED

    # Sequence tracking
    _send_sequence: int = field(default=0, repr=False)
    _recv_sequence: int = field(default=0, repr=False)
    _acked_sequence: int = field(default=0, repr=False)

    # Outgoing queue
    _outgoing: deque[RPCMessage] = field(default_factory=deque, repr=False)
    _pending_ack: dict[int, RPCMessage] = field(default_factory=dict, repr=False)

    # Incoming buffer for ordered delivery
    _incoming_buffer: dict[int, RPCMessage] = field(default_factory=dict, repr=False)
    _received_messages: deque[RPCMessage] = field(default_factory=deque, repr=False)

    # Callbacks
    _on_message: Optional[Callable[[RPCMessage], None]] = field(
        default=None, repr=False
    )

    # Configuration
    _max_batch_size: int = field(default=_config.DEFAULT_RPC_BATCH_SIZE, repr=False)
    _retransmit_timeout: float = field(default=_config.DEFAULT_RETRANSMIT_TIMEOUT, repr=False)

    def open(self) -> bool:
        """Open the channel.

        Returns:
            True if opened successfully
        """
        if self.state != RPCChannelState.CLOSED:
            return False

        self.state = RPCChannelState.OPEN
        self._send_sequence = 0
        self._recv_sequence = 0
        self._acked_sequence = 0
        self._outgoing.clear()
        self._pending_ack.clear()
        self._incoming_buffer.clear()
        self._received_messages.clear()

        return True

    def close(self) -> bool:
        """Close the channel gracefully.

        Returns:
            True if close initiated
        """
        if self.state == RPCChannelState.CLOSED:
            return False

        if self._pending_ack:
            # Wait for pending ACKs
            self.state = RPCChannelState.DRAINING
        else:
            self.state = RPCChannelState.CLOSED

        return True

    def force_close(self) -> None:
        """Force immediate closure."""
        self.state = RPCChannelState.CLOSED
        self._outgoing.clear()
        self._pending_ack.clear()
        self._incoming_buffer.clear()

    def send_rpc(self, rpc_info: RPCInfo, args: bytes) -> Optional[int]:
        """Send an RPC through the channel.

        Args:
            rpc_info: RPC metadata
            args: Serialized arguments

        Returns:
            Sequence number if queued, None if channel closed
        """
        if self.state != RPCChannelState.OPEN:
            return None

        self._send_sequence += 1
        sequence = self._send_sequence

        msg = RPCMessage(
            msg_type=RPC_MSG_CALL,
            rpc_hash=rpc_info.get_hash(),
            sequence=sequence,
            payload=args,
            reliable=rpc_info.reliable
        )

        self._outgoing.append(msg)

        return sequence

    def receive_rpc(self, data: bytes) -> list[tuple[RPCInfo, bytes]]:
        """Process received RPC data.

        Args:
            data: Received bytes

        Returns:
            List of (rpc_info, args) tuples for ready RPCs
        """
        offset = 0
        results = []

        while offset < len(data):
            msg, consumed = RPCMessage.deserialize(data[offset:])
            offset += consumed

            match msg.msg_type:
                case 0x01:  # RPC_MSG_CALL
                    self._handle_rpc_call(msg)
                case 0x02:  # RPC_MSG_ACK
                    self._handle_ack(msg)
                case 0x03:  # RPC_MSG_NACK
                    self._handle_nack(msg)
                case 0x04:  # RPC_MSG_BATCH
                    # Batch header, continue processing
                    pass

        # Process received messages in order
        while self._received_messages:
            msg = self._received_messages.popleft()
            # Return as (hash, payload) - caller maps hash to RPCInfo
            results.append((msg.rpc_hash, msg.payload))

        return results

    def get_outgoing_data(self, max_bytes: int = _config.DEFAULT_MAX_OUTGOING_DATA_SIZE) -> bytes:
        """Get queued outgoing data.

        Args:
            max_bytes: Maximum bytes to return

        Returns:
            Serialized RPC messages
        """
        if not self._outgoing:
            return b''

        parts = []
        total_size = 0
        sent_messages = []

        while self._outgoing and total_size < max_bytes:
            msg = self._outgoing[0]
            msg_data = msg.serialize()

            if total_size + len(msg_data) > max_bytes:
                break

            self._outgoing.popleft()
            parts.append(msg_data)
            total_size += len(msg_data)
            sent_messages.append(msg)

            # Track reliable messages
            if msg.reliable:
                self._pending_ack[msg.sequence] = msg

        return b''.join(parts)

    def get_retransmit_data(self) -> bytes:
        """Get data for retransmission.

        Returns:
            Serialized messages needing retransmit
        """
        now = time.time()
        parts = []

        for seq, msg in list(self._pending_ack.items()):
            if now - msg.timestamp > self._retransmit_timeout:
                msg.timestamp = now
                parts.append(msg.serialize())

        return b''.join(parts)

    def acknowledge(self, sequence: int) -> None:
        """Acknowledge a sequence number.

        Args:
            sequence: Sequence to acknowledge
        """
        self._pending_ack.pop(sequence, None)

        if sequence > self._acked_sequence:
            self._acked_sequence = sequence

        # Check if we can transition from draining to closed
        if self.state == RPCChannelState.DRAINING and not self._pending_ack:
            self.state = RPCChannelState.CLOSED

    def create_ack(self, sequence: int) -> bytes:
        """Create acknowledgment message.

        Args:
            sequence: Sequence to acknowledge

        Returns:
            Serialized ACK message
        """
        msg = RPCMessage(
            msg_type=RPC_MSG_ACK,
            rpc_hash=0,
            sequence=sequence,
            payload=b'',
            reliable=False
        )
        return msg.serialize()

    def create_nack(self, sequence: int, reason: str = "") -> bytes:
        """Create negative acknowledgment message.

        Args:
            sequence: Sequence to reject
            reason: Rejection reason

        Returns:
            Serialized NACK message
        """
        msg = RPCMessage(
            msg_type=RPC_MSG_NACK,
            rpc_hash=0,
            sequence=sequence,
            payload=reason.encode('utf-8')[:_config.MAX_NACK_REASON_LENGTH],
            reliable=False
        )
        return msg.serialize()

    def set_on_message(self, callback: Callable[[RPCMessage], None]) -> None:
        """Set callback for received messages.

        Args:
            callback: Function called with each received message
        """
        self._on_message = callback

    @property
    def is_open(self) -> bool:
        """Check if channel is open."""
        return self.state == RPCChannelState.OPEN

    @property
    def pending_count(self) -> int:
        """Get count of pending reliable messages."""
        return len(self._pending_ack)

    @property
    def outgoing_count(self) -> int:
        """Get count of queued outgoing messages."""
        return len(self._outgoing)

    def _handle_rpc_call(self, msg: RPCMessage) -> None:
        """Handle incoming RPC call message."""
        if msg.reliable:
            # Check ordering
            if msg.sequence <= self._recv_sequence:
                # Duplicate, ignore
                return

            if msg.sequence > self._recv_sequence + 1:
                # Out of order, buffer
                self._incoming_buffer[msg.sequence] = msg
                return

            # In order, process
            self._recv_sequence = msg.sequence
            self._received_messages.append(msg)

            # Check for buffered messages that are now in order
            self._process_buffered_messages()

            # Notify callback
            if self._on_message:
                self._on_message(msg)
        else:
            # Unreliable, just deliver
            self._received_messages.append(msg)
            if self._on_message:
                self._on_message(msg)

    def _handle_ack(self, msg: RPCMessage) -> None:
        """Handle acknowledgment message."""
        self.acknowledge(msg.sequence)

    def _handle_nack(self, msg: RPCMessage) -> None:
        """Handle negative acknowledgment message."""
        # RPC was rejected, remove from pending
        self._pending_ack.pop(msg.sequence, None)

        # Could trigger error callback here
        if self._on_message:
            self._on_message(msg)

    def _process_buffered_messages(self) -> None:
        """Process buffered messages that are now in order."""
        while True:
            next_seq = self._recv_sequence + 1
            if next_seq not in self._incoming_buffer:
                break

            msg = self._incoming_buffer.pop(next_seq)
            self._recv_sequence = next_seq
            self._received_messages.append(msg)

            if self._on_message:
                self._on_message(msg)


class RPCChannelManager:
    """Manages RPC channels for multiple connections.

    Coordinates channel creation, routing, and cleanup.
    """
    __slots__ = ('_channels', '_default_config')

    def __init__(self):
        """Initialize the channel manager."""
        self._channels: dict[int, RPCChannel] = {}  # connection_id -> channel
        self._default_config = {
            'max_batch_size': _config.DEFAULT_RPC_BATCH_SIZE,
            'retransmit_timeout': _config.DEFAULT_RETRANSMIT_TIMEOUT
        }

    def get_or_create_channel(self, connection_id: int) -> RPCChannel:
        """Get or create an RPC channel for a connection.

        Args:
            connection_id: Connection identifier

        Returns:
            The RPC channel
        """
        if connection_id not in self._channels:
            channel = RPCChannel(connection_id=connection_id)
            channel._max_batch_size = self._default_config['max_batch_size']
            channel._retransmit_timeout = self._default_config['retransmit_timeout']
            channel.open()
            self._channels[connection_id] = channel

        return self._channels[connection_id]

    def close_channel(self, connection_id: int) -> bool:
        """Close a channel.

        Args:
            connection_id: Connection ID

        Returns:
            True if channel was closed
        """
        channel = self._channels.get(connection_id)
        if channel:
            return channel.close()
        return False

    def remove_channel(self, connection_id: int) -> bool:
        """Remove a channel completely.

        Args:
            connection_id: Connection ID

        Returns:
            True if channel was removed
        """
        channel = self._channels.pop(connection_id, None)
        if channel:
            channel.force_close()
            return True
        return False

    def send_rpc(
        self,
        connection_id: int,
        rpc_info: RPCInfo,
        args: bytes
    ) -> Optional[int]:
        """Send an RPC to a connection.

        Args:
            connection_id: Target connection
            rpc_info: RPC metadata
            args: Serialized arguments

        Returns:
            Sequence number if sent
        """
        channel = self.get_or_create_channel(connection_id)
        return channel.send_rpc(rpc_info, args)

    def broadcast_rpc(
        self,
        rpc_info: RPCInfo,
        args: bytes,
        exclude: Optional[set[int]] = None
    ) -> dict[int, int]:
        """Broadcast an RPC to all connections.

        Args:
            rpc_info: RPC metadata
            args: Serialized arguments
            exclude: Connection IDs to exclude

        Returns:
            Dict mapping connection_id to sequence number
        """
        exclude = exclude or set()
        results = {}

        for conn_id, channel in self._channels.items():
            if conn_id in exclude:
                continue
            if not channel.is_open:
                continue

            seq = channel.send_rpc(rpc_info, args)
            if seq is not None:
                results[conn_id] = seq

        return results

    def get_outgoing_data(
        self,
        connection_id: int,
        max_bytes: int = _config.DEFAULT_MAX_OUTGOING_DATA_SIZE
    ) -> bytes:
        """Get outgoing data for a connection.

        Args:
            connection_id: Connection ID
            max_bytes: Maximum bytes

        Returns:
            Serialized RPC data
        """
        channel = self._channels.get(connection_id)
        if channel:
            return channel.get_outgoing_data(max_bytes)
        return b''

    def receive_data(
        self,
        connection_id: int,
        data: bytes
    ) -> list[tuple[int, bytes]]:
        """Process received data for a connection.

        Args:
            connection_id: Connection ID
            data: Received bytes

        Returns:
            List of (rpc_hash, args) tuples
        """
        channel = self.get_or_create_channel(connection_id)
        return channel.receive_rpc(data)

    def cleanup_closed_channels(self) -> int:
        """Remove closed channels.

        Returns:
            Number of channels removed
        """
        closed = [
            conn_id for conn_id, channel in self._channels.items()
            if channel.state == RPCChannelState.CLOSED
        ]
        for conn_id in closed:
            del self._channels[conn_id]
        return len(closed)

    def get_all_retransmit_data(self) -> dict[int, bytes]:
        """Get retransmit data for all channels.

        Returns:
            Dict mapping connection_id to retransmit data
        """
        results = {}
        for conn_id, channel in self._channels.items():
            data = channel.get_retransmit_data()
            if data:
                results[conn_id] = data
        return results

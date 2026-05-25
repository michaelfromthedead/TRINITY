"""
High-level network message serialization.

Provides unified interface for serializing and deserializing
network messages with type identification and versioning.
"""

from __future__ import annotations

import logging
import struct
import time
import zlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, Union

from .bit_packer import BitReader, BitWriter
from ..config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class MessageType(IntEnum):
    """Standard network message types."""
    # Connection management
    CONNECT_REQUEST = 1
    CONNECT_RESPONSE = 2
    DISCONNECT = 3
    HEARTBEAT = 4
    HEARTBEAT_ACK = 5

    # State synchronization
    FULL_STATE = 10
    DELTA_STATE = 11
    STATE_ACK = 12

    # Entity replication
    ENTITY_SPAWN = 20
    ENTITY_DESPAWN = 21
    ENTITY_UPDATE = 22

    # RPC
    RPC_REQUEST = 30
    RPC_RESPONSE = 31
    RPC_ERROR = 32

    # Input
    INPUT_STATE = 40
    INPUT_ACK = 41

    # Custom messages (application-defined)
    CUSTOM_START = 100
    CUSTOM_END = 255


@dataclass
class MessageHeader:
    """
    Header for network messages.

    Attributes:
        message_type: Type identifier for the message.
        version: Protocol version for compatibility checking.
        sequence: Message sequence number.
        timestamp: Timestamp when message was created.
        flags: Optional flags (compressed, reliable, etc.).
        payload_size: Size of the payload in bytes.
    """
    message_type: int
    version: int
    sequence: int
    timestamp: float
    flags: int
    payload_size: int

    # Flag constants
    FLAG_COMPRESSED = 0x01
    FLAG_RELIABLE = 0x02
    FLAG_ORDERED = 0x04
    FLAG_FRAGMENTED = 0x08

    # Header size in bytes
    HEADER_SIZE = DEFAULT_CONFIG.MESSAGE_HEADER_SIZE  # 1 + 1 + 4 + 8 + 2 + 4 = 20 bytes

    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        return struct.pack(
            '!BBIqHI',
            self.message_type,
            self.version,
            self.sequence,
            int(self.timestamp * 1000),  # milliseconds
            self.flags,
            self.payload_size
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> 'MessageHeader':
        """Deserialize header from bytes."""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(f"Need {cls.HEADER_SIZE} bytes for header, got {len(data)}")

        msg_type, version, sequence, timestamp_ms, flags, payload_size = struct.unpack(
            '!BBIqHI',
            data[:cls.HEADER_SIZE]
        )
        return cls(
            message_type=msg_type,
            version=version,
            sequence=sequence,
            timestamp=timestamp_ms / 1000.0,
            flags=flags,
            payload_size=payload_size
        )


T = TypeVar('T')


class NetSerializer:
    """
    High-level network message serializer.

    Handles message encoding/decoding with:
    - Type identification
    - Version compatibility
    - Optional compression
    - Sequence tracking
    - Custom type registration

    Example:
        serializer = NetSerializer()

        # Register custom message type
        @serializer.register(MessageType.CUSTOM_START)
        class MyMessage:
            def __init__(self, value: int):
                self.value = value

        # Serialize
        msg = MyMessage(42)
        data = serializer.serialize(MessageType.CUSTOM_START, msg)

        # Deserialize
        msg_type, payload = serializer.deserialize(data)
    """

    # Default protocol version
    PROTOCOL_VERSION = DEFAULT_CONFIG.PROTOCOL_VERSION

    def __init__(
        self,
        version: int = DEFAULT_CONFIG.PROTOCOL_VERSION,
        compress_threshold: int = DEFAULT_CONFIG.COMPRESS_THRESHOLD,
        compression_level: int = DEFAULT_CONFIG.COMPRESSION_LEVEL
    ) -> None:
        """
        Initialize the serializer.

        Args:
            version: Protocol version number.
            compress_threshold: Compress payloads larger than this (bytes).
            compression_level: zlib compression level (1-9).
        """
        self._version = version
        self._compress_threshold = compress_threshold
        self._compression_level = compression_level
        self._sequence = 0

        # Type registrations
        self._encoders: Dict[int, Callable[[Any], bytes]] = {}
        self._decoders: Dict[int, Callable[[bytes], Any]] = {}

        # Register built-in types
        self._register_builtins()

    @property
    def version(self) -> int:
        """Get protocol version."""
        return self._version

    @property
    def sequence(self) -> int:
        """Get current sequence number."""
        return self._sequence

    def _register_builtins(self) -> None:
        """Register built-in message type handlers."""
        # Dict payload (generic)
        def encode_dict(payload: Dict[str, Any]) -> bytes:
            writer = BitWriter()
            self._write_dict(writer, payload)
            return writer.to_bytes()

        def decode_dict(data: bytes) -> Dict[str, Any]:
            reader = BitReader(data)
            return self._read_dict(reader)

        # Register for common types
        for msg_type in [
            MessageType.FULL_STATE,
            MessageType.DELTA_STATE,
            MessageType.ENTITY_UPDATE,
            MessageType.RPC_REQUEST,
            MessageType.RPC_RESPONSE,
        ]:
            self._encoders[msg_type] = encode_dict
            self._decoders[msg_type] = decode_dict

        # Simple types
        def encode_empty(_: Any) -> bytes:
            return b''

        def decode_empty(_: bytes) -> None:
            return None

        for msg_type in [
            MessageType.HEARTBEAT,
            MessageType.HEARTBEAT_ACK,
            MessageType.DISCONNECT,
        ]:
            self._encoders[msg_type] = encode_empty
            self._decoders[msg_type] = decode_empty

    def register_encoder(
        self,
        message_type: int,
        encoder: Callable[[Any], bytes]
    ) -> None:
        """
        Register a custom encoder for a message type.

        Args:
            message_type: The message type identifier.
            encoder: Function to encode payload to bytes.
        """
        self._encoders[message_type] = encoder

    def register_decoder(
        self,
        message_type: int,
        decoder: Callable[[bytes], Any]
    ) -> None:
        """
        Register a custom decoder for a message type.

        Args:
            message_type: The message type identifier.
            decoder: Function to decode bytes to payload.
        """
        self._decoders[message_type] = decoder

    def serialize(
        self,
        message_type: int,
        payload: Any,
        flags: int = 0,
        sequence: Optional[int] = None
    ) -> bytes:
        """
        Serialize a message with header.

        Args:
            message_type: Type identifier for the message.
            payload: The message payload.
            flags: Optional message flags.
            sequence: Optional sequence override.

        Returns:
            Serialized message bytes.
        """
        # Encode payload
        encoder = self._encoders.get(message_type)
        if encoder:
            payload_bytes = encoder(payload)
        elif isinstance(payload, bytes):
            payload_bytes = payload
        elif isinstance(payload, dict):
            writer = BitWriter()
            self._write_dict(writer, payload)
            payload_bytes = writer.to_bytes()
        else:
            raise ValueError(f"No encoder for message type {message_type}")

        # Compress if beneficial
        if len(payload_bytes) > self._compress_threshold:
            compressed = zlib.compress(payload_bytes, self._compression_level)
            if len(compressed) < len(payload_bytes):
                payload_bytes = compressed
                flags |= MessageHeader.FLAG_COMPRESSED

        # Get sequence
        if sequence is None:
            sequence = self._sequence
            self._sequence = (self._sequence + 1) & 0xFFFFFFFF

        # Create header
        header = MessageHeader(
            message_type=message_type,
            version=self._version,
            sequence=sequence,
            timestamp=time.time(),
            flags=flags,
            payload_size=len(payload_bytes)
        )

        return header.to_bytes() + payload_bytes

    def deserialize(self, data: bytes) -> Tuple[int, Any]:
        """
        Deserialize a message.

        Args:
            data: The serialized message bytes.

        Returns:
            Tuple of (message_type, payload).

        Raises:
            ValueError: If data is malformed.
        """
        if len(data) < MessageHeader.HEADER_SIZE:
            raise ValueError(f"Data too short: {len(data)} bytes")

        # Parse header
        header = MessageHeader.from_bytes(data)

        # Extract payload
        payload_start = MessageHeader.HEADER_SIZE
        payload_end = payload_start + header.payload_size

        if len(data) < payload_end:
            raise ValueError(f"Incomplete payload: expected {header.payload_size} bytes")

        payload_bytes = data[payload_start:payload_end]

        # Decompress if needed
        if header.flags & MessageHeader.FLAG_COMPRESSED:
            payload_bytes = zlib.decompress(payload_bytes)

        # Decode payload
        decoder = self._decoders.get(header.message_type)
        if decoder:
            payload = decoder(payload_bytes)
        else:
            payload = payload_bytes

        return header.message_type, payload

    def deserialize_header(self, data: bytes) -> MessageHeader:
        """
        Deserialize only the message header.

        Args:
            data: The serialized message bytes.

        Returns:
            The message header.
        """
        return MessageHeader.from_bytes(data)

    def _write_dict(self, writer: BitWriter, data: Dict[str, Any]) -> None:
        """Write a dictionary to the bit writer."""
        writer.write_bits(len(data), 16)

        for key, value in data.items():
            # Write key
            key_bytes = key.encode('utf-8')
            writer.write_bits(len(key_bytes), 8)
            writer.write_bytes(key_bytes)

            # Write value with type tag
            self._write_value(writer, value)

    def _read_dict(self, reader: BitReader) -> Dict[str, Any]:
        """Read a dictionary from the bit reader."""
        count = reader.read_bits(16)
        result = {}

        for _ in range(count):
            # Read key
            key_len = reader.read_bits(8)
            key = reader.read_bytes(key_len).decode('utf-8')

            # Read value
            value = self._read_value(reader)
            result[key] = value

        return result

    def _write_value(self, writer: BitWriter, value: Any) -> None:
        """Write a typed value to the bit writer."""
        if value is None:
            writer.write_bits(0, 4)  # NULL
        elif isinstance(value, bool):
            writer.write_bits(1, 4)  # BOOL
            writer.write_bool(value)
        elif isinstance(value, int):
            if -128 <= value <= 127:
                writer.write_bits(2, 4)  # INT8
                writer.write_bits(value & 0xFF, 8)
            elif -32768 <= value <= 32767:
                writer.write_bits(3, 4)  # INT16
                writer.write_bits(value & 0xFFFF, 16)
            elif -2147483648 <= value <= 2147483647:
                writer.write_bits(4, 4)  # INT32
                writer.write_bits(value & 0xFFFFFFFF, 32)
            else:
                writer.write_bits(5, 4)  # INT64
                writer.write_bits(value & 0xFFFFFFFFFFFFFFFF, 64)
        elif isinstance(value, float):
            writer.write_bits(6, 4)  # FLOAT32
            packed = struct.pack('!f', value)
            for byte in packed:
                writer.write_bits(byte, 8)
        elif isinstance(value, str):
            writer.write_bits(7, 4)  # STRING
            str_bytes = value.encode('utf-8')
            writer.write_bits(len(str_bytes), 16)
            writer.write_bytes(str_bytes)
        elif isinstance(value, bytes):
            writer.write_bits(8, 4)  # BYTES
            writer.write_bits(len(value), 16)
            writer.write_bytes(value)
        elif isinstance(value, list):
            writer.write_bits(9, 4)  # ARRAY
            writer.write_bits(len(value), 16)
            for item in value:
                self._write_value(writer, item)
        elif isinstance(value, dict):
            writer.write_bits(10, 4)  # DICT
            self._write_dict(writer, value)
        else:
            # Fallback to string
            writer.write_bits(7, 4)
            str_bytes = str(value).encode('utf-8')
            writer.write_bits(len(str_bytes), 16)
            writer.write_bytes(str_bytes)

    def _read_value(self, reader: BitReader) -> Any:
        """Read a typed value from the bit reader."""
        type_tag = reader.read_bits(4)

        if type_tag == 0:  # NULL
            return None
        elif type_tag == 1:  # BOOL
            return reader.read_bool()
        elif type_tag == 2:  # INT8
            raw = reader.read_bits(8)
            return raw if raw < 128 else raw - 256
        elif type_tag == 3:  # INT16
            raw = reader.read_bits(16)
            return raw if raw < 32768 else raw - 65536
        elif type_tag == 4:  # INT32
            raw = reader.read_bits(32)
            return raw if raw < 2147483648 else raw - 4294967296
        elif type_tag == 5:  # INT64
            raw = reader.read_bits(64)
            return raw if raw < 9223372036854775808 else raw - 18446744073709551616
        elif type_tag == 6:  # FLOAT32
            packed = bytes([reader.read_bits(8) for _ in range(4)])
            return struct.unpack('!f', packed)[0]
        elif type_tag == 7:  # STRING
            length = reader.read_bits(16)
            return reader.read_bytes(length).decode('utf-8')
        elif type_tag == 8:  # BYTES
            length = reader.read_bits(16)
            return reader.read_bytes(length)
        elif type_tag == 9:  # ARRAY
            count = reader.read_bits(16)
            return [self._read_value(reader) for _ in range(count)]
        elif type_tag == 10:  # DICT
            return self._read_dict(reader)
        else:
            raise ValueError(f"Unknown type tag: {type_tag}")


# Convenience functions
_default_serializer = NetSerializer()


def serialize_message(
    message_type: int,
    payload: Any,
    flags: int = 0
) -> bytes:
    """
    Serialize a message using the default serializer.

    Args:
        message_type: Type identifier for the message.
        payload: The message payload.
        flags: Optional message flags.

    Returns:
        Serialized message bytes.
    """
    return _default_serializer.serialize(message_type, payload, flags)


def deserialize_message(data: bytes) -> Tuple[int, Any]:
    """
    Deserialize a message using the default serializer.

    Args:
        data: The serialized message bytes.

    Returns:
        Tuple of (message_type, payload).
    """
    return _default_serializer.deserialize(data)

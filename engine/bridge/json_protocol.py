"""JSON-based bridge channel protocol (T-CC-0.19).

Provides a structured JSON wire format for communication between
Python frontend and Rust backend via Type/Data/Command channels.

Channel Types:
- TYPE_CHANNEL: Schema registration and type introspection
- DATA_CHANNEL: Component data transfer and batched updates
- COMMAND_CHANNEL: Frame graph commands and render operations

Wire Format:
{
    "channel": "type" | "data" | "command",
    "version": 1,
    "timestamp": <unix_ms>,
    "sequence": <monotonic_id>,
    "payload": { ... channel-specific ... },
    "checksum": <optional crc32>
}
"""
from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TypeVar

__all__ = [
    "Channel",
    "MessageHeader",
    "TypeMessage",
    "DataMessage",
    "CommandMessage",
    "BridgeProtocol",
    "BridgeError",
    "ValidationError",
    "SerializationError",
    "ChannelHandler",
]


class Channel(Enum):
    """Bridge channel types."""

    TYPE = "type"
    DATA = "data"
    COMMAND = "command"


class BridgeError(Exception):
    """Base exception for bridge protocol errors."""

    pass


class ValidationError(BridgeError):
    """Raised when message validation fails."""

    pass


class SerializationError(BridgeError):
    """Raised when JSON serialization/deserialization fails."""

    pass


@dataclass(slots=True)
class MessageHeader:
    """Common message header for all channels."""

    channel: Channel
    version: int = 1
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    sequence: int = 0
    checksum: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize header to dictionary."""
        d = {
            "channel": self.channel.value,
            "version": self.version,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }
        if self.checksum is not None:
            d["checksum"] = self.checksum
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MessageHeader:
        """Deserialize header from dictionary."""
        return cls(
            channel=Channel(data["channel"]),
            version=data.get("version", 1),
            timestamp=data.get("timestamp", 0),
            sequence=data.get("sequence", 0),
            checksum=data.get("checksum"),
        )


@dataclass(slots=True)
class TypeMessage:
    """Type channel message for schema registration.

    Used to register component types, query type info, and synchronize
    schemas between Python and Rust.
    """

    action: str  # "register", "query", "list", "validate"
    type_id: int | None = None
    type_name: str | None = None
    fields: list[dict[str, Any]] = field(default_factory=list)
    flags: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = {"action": self.action}
        if self.type_id is not None:
            d["type_id"] = self.type_id
        if self.type_name is not None:
            d["type_name"] = self.type_name
        if self.fields:
            d["fields"] = self.fields
        if self.flags:
            d["flags"] = self.flags
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TypeMessage:
        """Deserialize from dictionary."""
        return cls(
            action=data["action"],
            type_id=data.get("type_id"),
            type_name=data.get("type_name"),
            fields=data.get("fields", []),
            flags=data.get("flags", 0),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def register(
        cls,
        type_id: int,
        type_name: str,
        fields: list[dict[str, Any]],
        flags: int = 0,
    ) -> TypeMessage:
        """Create a type registration message."""
        return cls(
            action="register",
            type_id=type_id,
            type_name=type_name,
            fields=fields,
            flags=flags,
        )

    @classmethod
    def query(cls, type_id: int | None = None, type_name: str | None = None) -> TypeMessage:
        """Create a type query message."""
        return cls(action="query", type_id=type_id, type_name=type_name)

    @classmethod
    def list_all(cls) -> TypeMessage:
        """Create a message to list all registered types."""
        return cls(action="list")


@dataclass(slots=True)
class DataMessage:
    """Data channel message for component data transfer.

    Used for batched component updates, entity spawn/despawn,
    and bulk data operations.
    """

    action: str  # "spawn", "despawn", "set", "get", "batch_set", "batch_get"
    entity_id: int | None = None
    component_id: int | None = None
    data: bytes | dict | list | None = None
    entities: list[int] = field(default_factory=list)
    components: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = {"action": self.action}
        if self.entity_id is not None:
            d["entity_id"] = self.entity_id
        if self.component_id is not None:
            d["component_id"] = self.component_id
        if self.data is not None:
            if isinstance(self.data, bytes):
                import base64
                d["data"] = base64.b64encode(self.data).decode("ascii")
                d["data_encoding"] = "base64"
            else:
                d["data"] = self.data
        if self.entities:
            d["entities"] = self.entities
        if self.components:
            d["components"] = self.components
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataMessage:
        """Deserialize from dictionary."""
        raw_data = data.get("data")
        if raw_data is not None and data.get("data_encoding") == "base64":
            import base64
            raw_data = base64.b64decode(raw_data)
        return cls(
            action=data["action"],
            entity_id=data.get("entity_id"),
            component_id=data.get("component_id"),
            data=raw_data,
            entities=data.get("entities", []),
            components=data.get("components", []),
        )

    @classmethod
    def spawn(cls, component_ids: list[int]) -> DataMessage:
        """Create a spawn message."""
        return cls(action="spawn", components=[{"id": cid} for cid in component_ids])

    @classmethod
    def despawn(cls, entity_id: int) -> DataMessage:
        """Create a despawn message."""
        return cls(action="despawn", entity_id=entity_id)

    @classmethod
    def set_component(
        cls,
        entity_id: int,
        component_id: int,
        data: bytes | dict,
    ) -> DataMessage:
        """Create a set component message."""
        return cls(
            action="set",
            entity_id=entity_id,
            component_id=component_id,
            data=data,
        )

    @classmethod
    def batch_set(cls, updates: list[dict[str, Any]]) -> DataMessage:
        """Create a batch set message."""
        return cls(action="batch_set", components=updates)


@dataclass(slots=True)
class CommandMessage:
    """Command channel message for frame graph operations.

    Used for frame graph compilation, render pass execution,
    and GPU resource management.
    """

    action: str  # "compile", "execute", "create_resource", "destroy_resource"
    frame_graph: dict | None = None
    passes: list[dict[str, Any]] = field(default_factory=list)
    resources: list[dict[str, Any]] = field(default_factory=list)
    resource_id: int | None = None
    resource_desc: dict | None = None
    execution_order: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = {"action": self.action}
        if self.frame_graph is not None:
            d["frame_graph"] = self.frame_graph
        if self.passes:
            d["passes"] = self.passes
        if self.resources:
            d["resources"] = self.resources
        if self.resource_id is not None:
            d["resource_id"] = self.resource_id
        if self.resource_desc is not None:
            d["resource_desc"] = self.resource_desc
        if self.execution_order:
            d["execution_order"] = self.execution_order
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommandMessage:
        """Deserialize from dictionary."""
        return cls(
            action=data["action"],
            frame_graph=data.get("frame_graph"),
            passes=data.get("passes", []),
            resources=data.get("resources", []),
            resource_id=data.get("resource_id"),
            resource_desc=data.get("resource_desc"),
            execution_order=data.get("execution_order", []),
        )

    @classmethod
    def compile_frame_graph(
        cls,
        passes: list[dict[str, Any]],
        resources: list[dict[str, Any]],
    ) -> CommandMessage:
        """Create a frame graph compile message."""
        return cls(action="compile", passes=passes, resources=resources)

    @classmethod
    def execute(cls, execution_order: list[str]) -> CommandMessage:
        """Create an execute message."""
        return cls(action="execute", execution_order=execution_order)

    @classmethod
    def create_resource(
        cls,
        resource_id: int,
        resource_desc: dict[str, Any],
    ) -> CommandMessage:
        """Create a resource creation message."""
        return cls(
            action="create_resource",
            resource_id=resource_id,
            resource_desc=resource_desc,
        )


T = TypeVar("T", TypeMessage, DataMessage, CommandMessage)
ChannelHandler = Callable[[T], Any]


class BridgeProtocol:
    """
    JSON bridge protocol handler.

    Manages message serialization, validation, and routing across
    Type/Data/Command channels.
    """

    __slots__ = (
        "_sequence",
        "_handlers",
        "_validate_checksums",
        "_compute_checksums",
        "_version",
    )

    def __init__(
        self,
        validate_checksums: bool = False,
        compute_checksums: bool = False,
        version: int = 1,
    ):
        self._sequence = 0
        self._handlers: dict[Channel, list[ChannelHandler]] = {
            Channel.TYPE: [],
            Channel.DATA: [],
            Channel.COMMAND: [],
        }
        self._validate_checksums = validate_checksums
        self._compute_checksums = compute_checksums
        self._version = version

    def _next_sequence(self) -> int:
        """Get next sequence number."""
        self._sequence += 1
        return self._sequence

    def _compute_checksum(self, payload_json: str) -> int:
        """Compute CRC32 checksum of payload."""
        return zlib.crc32(payload_json.encode("utf-8")) & 0xFFFFFFFF

    def register_handler(
        self,
        channel: Channel,
        handler: ChannelHandler,
    ) -> None:
        """Register a handler for a channel."""
        self._handlers[channel].append(handler)

    def unregister_handler(
        self,
        channel: Channel,
        handler: ChannelHandler,
    ) -> None:
        """Unregister a handler from a channel."""
        if handler in self._handlers[channel]:
            self._handlers[channel].remove(handler)

    def serialize(
        self,
        channel: Channel,
        payload: TypeMessage | DataMessage | CommandMessage,
    ) -> str:
        """Serialize a message to JSON string."""
        header = MessageHeader(
            channel=channel,
            version=self._version,
            sequence=self._next_sequence(),
        )

        payload_dict = payload.to_dict()
        payload_json = json.dumps(payload_dict)

        if self._compute_checksums:
            header.checksum = self._compute_checksum(payload_json)

        message = header.to_dict()
        message["payload"] = payload_dict

        return json.dumps(message, separators=(",", ":"))

    def deserialize(self, json_str: str) -> tuple[MessageHeader, Any]:
        """Deserialize a JSON string to header and payload.

        Returns:
            Tuple of (header, typed payload message)

        Raises:
            SerializationError: If JSON parsing fails
            ValidationError: If checksum validation fails
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise SerializationError(f"Invalid JSON: {e}") from e

        header = MessageHeader.from_dict(data)
        payload_dict = data.get("payload", {})

        # Validate checksum if present
        if self._validate_checksums and header.checksum is not None:
            payload_json = json.dumps(payload_dict)
            expected = self._compute_checksum(payload_json)
            if header.checksum != expected:
                raise ValidationError(
                    f"Checksum mismatch: expected {expected}, got {header.checksum}"
                )

        # Parse payload based on channel
        if header.channel == Channel.TYPE:
            payload = TypeMessage.from_dict(payload_dict)
        elif header.channel == Channel.DATA:
            payload = DataMessage.from_dict(payload_dict)
        elif header.channel == Channel.COMMAND:
            payload = CommandMessage.from_dict(payload_dict)
        else:
            raise ValidationError(f"Unknown channel: {header.channel}")

        return header, payload

    def dispatch(self, json_str: str) -> list[Any]:
        """Deserialize and dispatch a message to registered handlers.

        Returns:
            List of handler results
        """
        header, payload = self.deserialize(json_str)
        results = []

        for handler in self._handlers[header.channel]:
            result = handler(payload)
            results.append(result)

        return results

    def send_type(self, message: TypeMessage) -> str:
        """Serialize a type channel message."""
        return self.serialize(Channel.TYPE, message)

    def send_data(self, message: DataMessage) -> str:
        """Serialize a data channel message."""
        return self.serialize(Channel.DATA, message)

    def send_command(self, message: CommandMessage) -> str:
        """Serialize a command channel message."""
        return self.serialize(Channel.COMMAND, message)

    def create_type_register(
        self,
        type_id: int,
        type_name: str,
        fields: list[tuple[str, str, int]],
        flags: int = 0,
    ) -> str:
        """Create a type registration message JSON.

        Args:
            type_id: Unique type identifier
            type_name: Human-readable type name
            fields: List of (name, type_code, offset) tuples
            flags: Type flags

        Returns:
            JSON string ready to send
        """
        field_dicts = [
            {"name": name, "type_code": type_code, "offset": offset}
            for name, type_code, offset in fields
        ]
        message = TypeMessage.register(type_id, type_name, field_dicts, flags)
        return self.send_type(message)

    def create_spawn(self, component_ids: list[int]) -> str:
        """Create a spawn message JSON."""
        return self.send_data(DataMessage.spawn(component_ids))

    def create_despawn(self, entity_id: int) -> str:
        """Create a despawn message JSON."""
        return self.send_data(DataMessage.despawn(entity_id))

    def create_frame_graph_compile(
        self,
        passes: list[dict[str, Any]],
        resources: list[dict[str, Any]],
    ) -> str:
        """Create a frame graph compile command JSON."""
        return self.send_command(
            CommandMessage.compile_frame_graph(passes, resources)
        )


def create_default_protocol(
    validate_checksums: bool = False,
    compute_checksums: bool = False,
) -> BridgeProtocol:
    """Factory function to create a default protocol instance."""
    return BridgeProtocol(
        validate_checksums=validate_checksums,
        compute_checksums=compute_checksums,
    )

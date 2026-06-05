"""Tests for JSON bridge protocol (T-CC-0.19)."""
import base64
import json
import pytest

from engine.bridge.json_protocol import (
    Channel,
    MessageHeader,
    TypeMessage,
    DataMessage,
    CommandMessage,
    BridgeProtocol,
    BridgeError,
    ValidationError,
    SerializationError,
    create_default_protocol,
)


class TestChannel:
    """Test Channel enum."""

    def test_channel_values(self):
        """Test channel enum values."""
        assert Channel.TYPE.value == "type"
        assert Channel.DATA.value == "data"
        assert Channel.COMMAND.value == "command"

    def test_channel_from_string(self):
        """Test creating channel from string."""
        assert Channel("type") == Channel.TYPE
        assert Channel("data") == Channel.DATA
        assert Channel("command") == Channel.COMMAND


class TestMessageHeader:
    """Test MessageHeader dataclass."""

    def test_header_creation(self):
        """Test header creation with defaults."""
        header = MessageHeader(channel=Channel.TYPE)
        assert header.channel == Channel.TYPE
        assert header.version == 1
        assert header.timestamp > 0
        assert header.sequence == 0

    def test_header_to_dict(self):
        """Test header serialization."""
        header = MessageHeader(
            channel=Channel.DATA,
            version=2,
            timestamp=12345,
            sequence=42,
        )
        d = header.to_dict()
        assert d["channel"] == "data"
        assert d["version"] == 2
        assert d["timestamp"] == 12345
        assert d["sequence"] == 42
        assert "checksum" not in d

    def test_header_to_dict_with_checksum(self):
        """Test header serialization with checksum."""
        header = MessageHeader(
            channel=Channel.TYPE,
            checksum=0xDEADBEEF,
        )
        d = header.to_dict()
        assert d["checksum"] == 0xDEADBEEF

    def test_header_from_dict(self):
        """Test header deserialization."""
        data = {
            "channel": "command",
            "version": 1,
            "timestamp": 99999,
            "sequence": 100,
        }
        header = MessageHeader.from_dict(data)
        assert header.channel == Channel.COMMAND
        assert header.timestamp == 99999
        assert header.sequence == 100


class TestTypeMessage:
    """Test TypeMessage dataclass."""

    def test_type_message_register(self):
        """Test type registration message."""
        msg = TypeMessage.register(
            type_id=1,
            type_name="Transform",
            fields=[
                {"name": "x", "type_code": "f32", "offset": 0},
                {"name": "y", "type_code": "f32", "offset": 4},
            ],
            flags=0,
        )
        assert msg.action == "register"
        assert msg.type_id == 1
        assert msg.type_name == "Transform"
        assert len(msg.fields) == 2

    def test_type_message_query(self):
        """Test type query message."""
        msg = TypeMessage.query(type_id=5)
        assert msg.action == "query"
        assert msg.type_id == 5

    def test_type_message_list(self):
        """Test list all types message."""
        msg = TypeMessage.list_all()
        assert msg.action == "list"

    def test_type_message_to_dict(self):
        """Test type message serialization."""
        msg = TypeMessage(
            action="register",
            type_id=10,
            type_name="Velocity",
            fields=[{"name": "vx", "type_code": "f32", "offset": 0}],
        )
        d = msg.to_dict()
        assert d["action"] == "register"
        assert d["type_id"] == 10
        assert d["type_name"] == "Velocity"
        assert len(d["fields"]) == 1

    def test_type_message_from_dict(self):
        """Test type message deserialization."""
        data = {
            "action": "register",
            "type_id": 20,
            "type_name": "Health",
            "fields": [],
            "flags": 1,
        }
        msg = TypeMessage.from_dict(data)
        assert msg.type_id == 20
        assert msg.type_name == "Health"
        assert msg.flags == 1


class TestDataMessage:
    """Test DataMessage dataclass."""

    def test_data_message_spawn(self):
        """Test spawn message."""
        msg = DataMessage.spawn([1, 2, 3])
        assert msg.action == "spawn"
        assert len(msg.components) == 3

    def test_data_message_despawn(self):
        """Test despawn message."""
        msg = DataMessage.despawn(entity_id=42)
        assert msg.action == "despawn"
        assert msg.entity_id == 42

    def test_data_message_set_component(self):
        """Test set component message."""
        msg = DataMessage.set_component(
            entity_id=1,
            component_id=2,
            data={"x": 10.0, "y": 20.0},
        )
        assert msg.action == "set"
        assert msg.entity_id == 1
        assert msg.component_id == 2
        assert msg.data["x"] == 10.0

    def test_data_message_set_component_bytes(self):
        """Test set component with binary data."""
        raw_data = b"\x00\x00\x80\x3f"  # 1.0 as f32
        msg = DataMessage.set_component(
            entity_id=1,
            component_id=2,
            data=raw_data,
        )
        d = msg.to_dict()
        assert d["data_encoding"] == "base64"
        assert base64.b64decode(d["data"]) == raw_data

    def test_data_message_batch_set(self):
        """Test batch set message."""
        updates = [
            {"entity_id": 1, "component_id": 1, "data": {"x": 0}},
            {"entity_id": 2, "component_id": 1, "data": {"x": 1}},
        ]
        msg = DataMessage.batch_set(updates)
        assert msg.action == "batch_set"
        assert len(msg.components) == 2

    def test_data_message_from_dict_base64(self):
        """Test deserialization of base64 encoded data."""
        raw_data = b"\xde\xad\xbe\xef"
        encoded = base64.b64encode(raw_data).decode("ascii")
        data = {
            "action": "set",
            "entity_id": 1,
            "component_id": 1,
            "data": encoded,
            "data_encoding": "base64",
        }
        msg = DataMessage.from_dict(data)
        assert msg.data == raw_data


class TestCommandMessage:
    """Test CommandMessage dataclass."""

    def test_command_message_compile(self):
        """Test frame graph compile message."""
        passes = [{"name": "depth", "attachments": []}]
        resources = [{"name": "depth_buffer", "format": "D32_FLOAT"}]
        msg = CommandMessage.compile_frame_graph(passes, resources)
        assert msg.action == "compile"
        assert len(msg.passes) == 1
        assert len(msg.resources) == 1

    def test_command_message_execute(self):
        """Test execute message."""
        msg = CommandMessage.execute(["depth", "gbuffer", "lighting"])
        assert msg.action == "execute"
        assert msg.execution_order == ["depth", "gbuffer", "lighting"]

    def test_command_message_create_resource(self):
        """Test create resource message."""
        msg = CommandMessage.create_resource(
            resource_id=100,
            resource_desc={"width": 1920, "height": 1080, "format": "RGBA8"},
        )
        assert msg.action == "create_resource"
        assert msg.resource_id == 100
        assert msg.resource_desc["width"] == 1920

    def test_command_message_to_dict(self):
        """Test command message serialization."""
        msg = CommandMessage(
            action="compile",
            passes=[{"name": "test"}],
        )
        d = msg.to_dict()
        assert d["action"] == "compile"
        assert len(d["passes"]) == 1

    def test_command_message_from_dict(self):
        """Test command message deserialization."""
        data = {
            "action": "execute",
            "execution_order": ["pass1", "pass2"],
        }
        msg = CommandMessage.from_dict(data)
        assert msg.action == "execute"
        assert msg.execution_order == ["pass1", "pass2"]


class TestBridgeProtocol:
    """Test BridgeProtocol class."""

    def test_protocol_creation(self):
        """Test protocol creation."""
        protocol = BridgeProtocol()
        assert protocol._version == 1
        assert protocol._sequence == 0

    def test_protocol_serialize_type(self):
        """Test serializing type message."""
        protocol = BridgeProtocol()
        msg = TypeMessage.register(1, "Test", [], 0)
        json_str = protocol.send_type(msg)

        data = json.loads(json_str)
        assert data["channel"] == "type"
        assert data["payload"]["action"] == "register"
        assert data["sequence"] == 1

    def test_protocol_serialize_data(self):
        """Test serializing data message."""
        protocol = BridgeProtocol()
        msg = DataMessage.spawn([1])
        json_str = protocol.send_data(msg)

        data = json.loads(json_str)
        assert data["channel"] == "data"
        assert data["payload"]["action"] == "spawn"

    def test_protocol_serialize_command(self):
        """Test serializing command message."""
        protocol = BridgeProtocol()
        msg = CommandMessage.execute(["p1", "p2"])
        json_str = protocol.send_command(msg)

        data = json.loads(json_str)
        assert data["channel"] == "command"
        assert data["payload"]["action"] == "execute"

    def test_protocol_sequence_increment(self):
        """Test sequence number increments."""
        protocol = BridgeProtocol()
        msg = TypeMessage.list_all()

        json1 = protocol.send_type(msg)
        json2 = protocol.send_type(msg)
        json3 = protocol.send_type(msg)

        assert json.loads(json1)["sequence"] == 1
        assert json.loads(json2)["sequence"] == 2
        assert json.loads(json3)["sequence"] == 3

    def test_protocol_deserialize(self):
        """Test deserializing messages."""
        protocol = BridgeProtocol()
        msg = TypeMessage.register(5, "Position", [], 0)
        json_str = protocol.send_type(msg)

        header, payload = protocol.deserialize(json_str)
        assert header.channel == Channel.TYPE
        assert isinstance(payload, TypeMessage)
        assert payload.action == "register"
        assert payload.type_id == 5

    def test_protocol_deserialize_data(self):
        """Test deserializing data message."""
        protocol = BridgeProtocol()
        msg = DataMessage.despawn(99)
        json_str = protocol.send_data(msg)

        header, payload = protocol.deserialize(json_str)
        assert header.channel == Channel.DATA
        assert isinstance(payload, DataMessage)
        assert payload.entity_id == 99

    def test_protocol_deserialize_command(self):
        """Test deserializing command message."""
        protocol = BridgeProtocol()
        msg = CommandMessage.compile_frame_graph([], [])
        json_str = protocol.send_command(msg)

        header, payload = protocol.deserialize(json_str)
        assert header.channel == Channel.COMMAND
        assert isinstance(payload, CommandMessage)


class TestBridgeProtocolChecksum:
    """Test checksum functionality."""

    def test_protocol_compute_checksum(self):
        """Test checksum computation."""
        protocol = BridgeProtocol(compute_checksums=True)
        msg = TypeMessage.list_all()
        json_str = protocol.send_type(msg)

        data = json.loads(json_str)
        assert "checksum" in data
        assert isinstance(data["checksum"], int)

    def test_protocol_validate_checksum_success(self):
        """Test successful checksum validation."""
        protocol = BridgeProtocol(compute_checksums=True, validate_checksums=True)
        msg = TypeMessage.list_all()
        json_str = protocol.send_type(msg)

        # Should not raise
        header, payload = protocol.deserialize(json_str)
        assert payload.action == "list"

    def test_protocol_validate_checksum_failure(self):
        """Test checksum validation failure."""
        protocol = BridgeProtocol(compute_checksums=True, validate_checksums=True)
        msg = TypeMessage.list_all()
        json_str = protocol.send_type(msg)

        # Corrupt the checksum
        data = json.loads(json_str)
        data["checksum"] = 12345
        corrupted = json.dumps(data)

        with pytest.raises(ValidationError, match="Checksum mismatch"):
            protocol.deserialize(corrupted)


class TestBridgeProtocolHandlers:
    """Test handler registration and dispatch."""

    def test_register_handler(self):
        """Test registering a handler."""
        protocol = BridgeProtocol()
        calls = []

        def handler(msg: TypeMessage):
            calls.append(msg)
            return "handled"

        protocol.register_handler(Channel.TYPE, handler)

        msg = TypeMessage.list_all()
        json_str = protocol.send_type(msg)
        results = protocol.dispatch(json_str)

        assert len(calls) == 1
        assert results == ["handled"]

    def test_unregister_handler(self):
        """Test unregistering a handler."""
        protocol = BridgeProtocol()
        calls = []

        def handler(msg):
            calls.append(msg)

        protocol.register_handler(Channel.TYPE, handler)
        protocol.unregister_handler(Channel.TYPE, handler)

        msg = TypeMessage.list_all()
        json_str = protocol.send_type(msg)
        protocol.dispatch(json_str)

        assert len(calls) == 0

    def test_multiple_handlers(self):
        """Test multiple handlers on same channel."""
        protocol = BridgeProtocol()
        results = []

        protocol.register_handler(Channel.DATA, lambda m: results.append(1))
        protocol.register_handler(Channel.DATA, lambda m: results.append(2))
        protocol.register_handler(Channel.DATA, lambda m: results.append(3))

        msg = DataMessage.spawn([1])
        json_str = protocol.send_data(msg)
        protocol.dispatch(json_str)

        assert results == [1, 2, 3]


class TestBridgeProtocolErrors:
    """Test error handling."""

    def test_invalid_json(self):
        """Test handling invalid JSON."""
        protocol = BridgeProtocol()

        with pytest.raises(SerializationError, match="Invalid JSON"):
            protocol.deserialize("not valid json {{{")

    def test_unknown_channel(self):
        """Test handling unknown channel."""
        protocol = BridgeProtocol()

        json_str = json.dumps({
            "channel": "unknown",
            "payload": {},
        })

        with pytest.raises(ValueError):  # From Channel enum
            protocol.deserialize(json_str)


class TestBridgeProtocolHelpers:
    """Test helper methods."""

    def test_create_type_register(self):
        """Test create_type_register helper."""
        protocol = BridgeProtocol()
        json_str = protocol.create_type_register(
            type_id=1,
            type_name="Transform",
            fields=[("x", "f32", 0), ("y", "f32", 4)],
            flags=0,
        )

        header, payload = protocol.deserialize(json_str)
        assert payload.action == "register"
        assert payload.type_name == "Transform"
        assert len(payload.fields) == 2
        assert payload.fields[0]["name"] == "x"

    def test_create_spawn(self):
        """Test create_spawn helper."""
        protocol = BridgeProtocol()
        json_str = protocol.create_spawn([1, 2, 3])

        header, payload = protocol.deserialize(json_str)
        assert payload.action == "spawn"
        assert len(payload.components) == 3

    def test_create_despawn(self):
        """Test create_despawn helper."""
        protocol = BridgeProtocol()
        json_str = protocol.create_despawn(entity_id=42)

        header, payload = protocol.deserialize(json_str)
        assert payload.action == "despawn"
        assert payload.entity_id == 42

    def test_create_frame_graph_compile(self):
        """Test create_frame_graph_compile helper."""
        protocol = BridgeProtocol()
        json_str = protocol.create_frame_graph_compile(
            passes=[{"name": "depth"}],
            resources=[{"name": "depth_buf"}],
        )

        header, payload = protocol.deserialize(json_str)
        assert payload.action == "compile"
        assert len(payload.passes) == 1
        assert len(payload.resources) == 1


class TestCreateDefaultProtocol:
    """Test factory function."""

    def test_create_default(self):
        """Test default protocol creation."""
        protocol = create_default_protocol()
        assert isinstance(protocol, BridgeProtocol)
        assert not protocol._validate_checksums
        assert not protocol._compute_checksums

    def test_create_with_checksums(self):
        """Test protocol creation with checksums."""
        protocol = create_default_protocol(
            validate_checksums=True,
            compute_checksums=True,
        )
        assert protocol._validate_checksums
        assert protocol._compute_checksums


class TestIntegration:
    """Integration tests for full message roundtrip."""

    def test_roundtrip_type_registration(self):
        """Test full roundtrip of type registration."""
        protocol = BridgeProtocol(compute_checksums=True, validate_checksums=True)

        # Create and serialize
        original = TypeMessage.register(
            type_id=100,
            type_name="ComplexComponent",
            fields=[
                {"name": "position", "type_code": "vec3f", "offset": 0},
                {"name": "rotation", "type_code": "quat", "offset": 12},
                {"name": "scale", "type_code": "vec3f", "offset": 28},
            ],
            flags=3,
        )
        json_str = protocol.send_type(original)

        # Deserialize
        header, received = protocol.deserialize(json_str)

        assert header.channel == Channel.TYPE
        assert received.type_id == original.type_id
        assert received.type_name == original.type_name
        assert len(received.fields) == 3
        assert received.flags == 3

    def test_roundtrip_data_batch(self):
        """Test full roundtrip of batch data update."""
        protocol = BridgeProtocol()

        updates = [
            {"entity_id": i, "component_id": 1, "data": {"value": i * 10}}
            for i in range(100)
        ]
        original = DataMessage.batch_set(updates)
        json_str = protocol.send_data(original)

        header, received = protocol.deserialize(json_str)

        assert header.channel == Channel.DATA
        assert received.action == "batch_set"
        assert len(received.components) == 100

    def test_roundtrip_frame_graph(self):
        """Test full roundtrip of frame graph compilation."""
        protocol = BridgeProtocol()

        passes = [
            {"name": "depth_prepass", "attachments": ["depth"]},
            {"name": "gbuffer", "attachments": ["albedo", "normal", "depth"]},
            {"name": "lighting", "attachments": ["hdr_color"]},
            {"name": "tonemap", "attachments": ["ldr_color"]},
        ]
        resources = [
            {"name": "depth", "format": "D32_FLOAT", "size": [1920, 1080]},
            {"name": "albedo", "format": "RGBA8", "size": [1920, 1080]},
            {"name": "normal", "format": "RG16_FLOAT", "size": [1920, 1080]},
            {"name": "hdr_color", "format": "RGBA16_FLOAT", "size": [1920, 1080]},
            {"name": "ldr_color", "format": "RGBA8", "size": [1920, 1080]},
        ]

        original = CommandMessage.compile_frame_graph(passes, resources)
        json_str = protocol.send_command(original)

        header, received = protocol.deserialize(json_str)

        assert header.channel == Channel.COMMAND
        assert received.action == "compile"
        assert len(received.passes) == 4
        assert len(received.resources) == 5

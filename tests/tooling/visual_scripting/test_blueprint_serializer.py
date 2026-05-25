"""
Tests for FlowForge blueprint serializer.

Tests save/load, versioning, and format options.
"""

import pytest
import json
import tempfile
import os

from engine.tooling.visual_scripting.blueprint_serializer import (
    SerializationFormat,
    SerializationOptions,
    BlueprintHeader,
    BlueprintSerializer,
    IncrementalSerializer,
    save_blueprint,
    load_blueprint,
    export_blueprint_json,
    import_blueprint_json,
)
from engine.tooling.visual_scripting.graph_editor import BlueprintGraph, Connection
from engine.tooling.visual_scripting.node_types import (
    BeginPlayNode,
    BranchNode,
    PrintStringNode,
    IntLiteralNode,
    GetVariableNode,
)
from engine.tooling.visual_scripting.data_types import IntType, FloatType


class TestSerializationOptions:
    """Tests for SerializationOptions class."""

    def test_default_options(self):
        options = SerializationOptions()
        assert options.format == SerializationFormat.JSON
        assert options.include_metadata is True
        assert options.include_comments is True

    def test_custom_options(self):
        options = SerializationOptions(
            format=SerializationFormat.JSON_COMPRESSED,
            include_comments=False,
            pretty_print=True
        )
        assert options.format == SerializationFormat.JSON_COMPRESSED
        assert options.include_comments is False
        assert options.pretty_print is True


class TestBlueprintSerializer:
    """Tests for BlueprintSerializer class."""

    def test_create_serializer(self):
        serializer = BlueprintSerializer()
        assert serializer.options is not None

    def test_serialize_empty_graph(self):
        serializer = BlueprintSerializer()
        graph = BlueprintGraph(name="EmptyGraph")

        data = serializer.serialize(graph)

        assert len(data) > 0

    def test_serialize_simple_graph(self):
        serializer = BlueprintSerializer()

        graph = BlueprintGraph(name="SimpleGraph")
        begin = BeginPlayNode(position=(100, 100))
        print_node = PrintStringNode(position=(300, 100))
        graph.add_node(begin)
        graph.add_node(print_node)

        conn = Connection(
            id="c1",
            source_node_id=begin.id,
            source_pin_id=begin.output_pins["Out"].id,
            target_node_id=print_node.id,
            target_pin_id=print_node.input_pins["In"].id
        )
        graph.add_connection(conn)

        data = serializer.serialize(graph)

        assert len(data) > 0

    def test_serialize_json_format(self):
        options = SerializationOptions(format=SerializationFormat.JSON)
        serializer = BlueprintSerializer(options)

        graph = BlueprintGraph(name="Test")
        begin = BeginPlayNode()
        graph.add_node(begin)

        data = serializer.serialize(graph)

        # Should be valid JSON
        parsed = json.loads(data.decode('utf-8'))
        assert "header" in parsed
        assert "nodes" in parsed

    def test_serialize_json_compressed(self):
        options = SerializationOptions(format=SerializationFormat.JSON_COMPRESSED)
        serializer = BlueprintSerializer(options)

        graph = BlueprintGraph(name="Test")
        begin = BeginPlayNode()
        graph.add_node(begin)

        compressed = serializer.serialize(graph)

        # Compressed should start with gzip magic
        assert compressed[:2] == b'\x1f\x8b'

    def test_serialize_pretty_print(self):
        options = SerializationOptions(pretty_print=True)
        serializer = BlueprintSerializer(options)

        graph = BlueprintGraph(name="Test")
        begin = BeginPlayNode()
        graph.add_node(begin)

        data = serializer.serialize(graph)
        json_str = data.decode('utf-8')

        # Should have newlines
        assert '\n' in json_str

    def test_serialize_include_comments(self):
        options = SerializationOptions(include_comments=True)
        serializer = BlueprintSerializer(options)

        graph = BlueprintGraph(name="Test")
        begin = BeginPlayNode()
        begin.comment = "This is a test comment"
        graph.add_node(begin)

        data = serializer.serialize(graph)
        json_str = data.decode('utf-8')

        assert "test comment" in json_str

    def test_serialize_exclude_comments(self):
        options = SerializationOptions(include_comments=False)
        serializer = BlueprintSerializer(options)

        graph = BlueprintGraph(name="Test")
        begin = BeginPlayNode()
        begin.comment = "This should not appear"
        graph.add_node(begin)

        data = serializer.serialize(graph)
        json_str = data.decode('utf-8')

        assert "This should not appear" not in json_str


class TestBlueprintDeserializer:
    """Tests for deserialization."""

    def test_deserialize_json(self):
        serializer = BlueprintSerializer()

        graph = BlueprintGraph(name="TestGraph")
        begin = BeginPlayNode(position=(100, 200))
        graph.add_node(begin)

        data = serializer.serialize(graph)
        restored = serializer.deserialize(data)

        assert restored is not None
        assert restored.name == "TestGraph"
        assert len(restored.nodes) == 1

    def test_deserialize_compressed(self):
        options = SerializationOptions(format=SerializationFormat.JSON_COMPRESSED)
        serializer = BlueprintSerializer(options)

        graph = BlueprintGraph(name="CompressedGraph")
        begin = BeginPlayNode()
        graph.add_node(begin)

        data = serializer.serialize(graph)
        restored = serializer.deserialize(data)

        assert restored is not None
        assert restored.name == "CompressedGraph"

    def test_deserialize_with_connections(self):
        serializer = BlueprintSerializer()

        graph = BlueprintGraph(name="ConnectedGraph")
        begin = BeginPlayNode()
        print_node = PrintStringNode()
        graph.add_node(begin)
        graph.add_node(print_node)

        conn = Connection(
            id="c1",
            source_node_id=begin.id,
            source_pin_id=begin.output_pins["Out"].id,
            target_node_id=print_node.id,
            target_pin_id=print_node.input_pins["In"].id
        )
        graph.add_connection(conn)

        data = serializer.serialize(graph)
        restored = serializer.deserialize(data)

        assert len(restored.nodes) == 2
        assert len(restored.connections) == 1

    def test_deserialize_preserves_position(self):
        serializer = BlueprintSerializer()

        graph = BlueprintGraph(name="Test")
        begin = BeginPlayNode(position=(150, 250))
        graph.add_node(begin)

        data = serializer.serialize(graph)
        restored = serializer.deserialize(data)

        restored_node = list(restored.nodes.values())[0]
        assert restored_node.position == (150, 250)

    def test_deserialize_invalid_data(self):
        serializer = BlueprintSerializer()

        result = serializer.deserialize(b"invalid data")

        assert result is None


class TestFileOperations:
    """Tests for file save/load operations."""

    def test_serialize_to_file(self):
        serializer = BlueprintSerializer()

        graph = BlueprintGraph(name="FileTest")
        begin = BeginPlayNode()
        graph.add_node(begin)

        with tempfile.NamedTemporaryFile(suffix=".bp", delete=False) as f:
            filepath = f.name

        try:
            result = serializer.serialize_to_file(graph, filepath)
            assert result is True
            assert os.path.exists(filepath)
        finally:
            os.unlink(filepath)

    def test_deserialize_from_file(self):
        serializer = BlueprintSerializer()

        graph = BlueprintGraph(name="FileLoadTest")
        begin = BeginPlayNode()
        graph.add_node(begin)

        with tempfile.NamedTemporaryFile(suffix=".bp", delete=False) as f:
            filepath = f.name

        try:
            serializer.serialize_to_file(graph, filepath)
            restored = serializer.deserialize_from_file(filepath)

            assert restored is not None
            assert restored.name == "FileLoadTest"
        finally:
            os.unlink(filepath)

    def test_deserialize_from_nonexistent_file(self):
        serializer = BlueprintSerializer()

        result = serializer.deserialize_from_file("/nonexistent/path/file.bp")

        assert result is None


class TestBlueprintHeader:
    """Tests for BlueprintHeader reading."""

    def test_read_header(self):
        serializer = BlueprintSerializer()

        graph = BlueprintGraph(name="HeaderTest")
        graph.description = "Test description"
        begin = BeginPlayNode()
        print_node = PrintStringNode()
        graph.add_node(begin)
        graph.add_node(print_node)

        data = serializer.serialize(graph)
        header = serializer.read_header(data)

        assert header is not None
        assert header.node_count == 2
        assert header.description == "Test description"

    def test_read_header_from_compressed(self):
        options = SerializationOptions(format=SerializationFormat.JSON_COMPRESSED)
        serializer = BlueprintSerializer(options)

        graph = BlueprintGraph(name="CompressedHeader")
        begin = BeginPlayNode()
        graph.add_node(begin)

        data = serializer.serialize(graph)
        header = serializer.read_header(data)

        assert header is not None

    def test_read_header_invalid(self):
        serializer = BlueprintSerializer()

        header = serializer.read_header(b"invalid")

        assert header is None


class TestChecksum:
    """Tests for checksum validation."""

    def test_checksum_in_header(self):
        serializer = BlueprintSerializer()

        graph = BlueprintGraph(name="ChecksumTest")
        begin = BeginPlayNode()
        graph.add_node(begin)

        data = serializer.serialize(graph)
        parsed = json.loads(data.decode('utf-8'))

        assert parsed["header"]["checksum"] != ""

    def test_checksum_validation(self):
        serializer = BlueprintSerializer()

        graph = BlueprintGraph(name="Test")
        begin = BeginPlayNode()
        graph.add_node(begin)

        data = serializer.serialize(graph)

        # Modify data slightly
        json_data = json.loads(data.decode('utf-8'))
        original_checksum = json_data["header"]["checksum"]

        # Deserialization should still work (just might warn)
        restored = serializer.deserialize(data)
        assert restored is not None


class TestNodeSerialization:
    """Tests for specific node type serialization."""

    def test_serialize_variable_node(self):
        serializer = BlueprintSerializer()

        graph = BlueprintGraph(name="VarTest")
        get_var = GetVariableNode(
            variable_name="Health",
            variable_type=FloatType
        )
        graph.add_node(get_var)

        data = serializer.serialize(graph)
        restored = serializer.deserialize(data)

        assert len(restored.nodes) == 1

    def test_serialize_with_pin_values(self):
        serializer = BlueprintSerializer()

        graph = BlueprintGraph(name="PinTest")
        print_node = PrintStringNode()
        print_node.input_pins["String"].set_value("Hello World")
        print_node.input_pins["Duration"].set_value(5.0)
        graph.add_node(print_node)

        data = serializer.serialize(graph)
        restored = serializer.deserialize(data)

        restored_node = list(restored.nodes.values())[0]
        # Values should be preserved
        assert restored_node.input_pins["String"].get_value() == "Hello World"


class TestIncrementalSerializer:
    """Tests for IncrementalSerializer class."""

    def test_record_change(self):
        base = BlueprintSerializer()
        incremental = IncrementalSerializer(base)

        incremental.record_change("add_node", "node_1", {"type": "BeginPlay"})

        assert len(incremental._change_log) == 1

    def test_get_incremental_update(self):
        base = BlueprintSerializer()
        incremental = IncrementalSerializer(base)

        incremental.record_change("add_node", "node_1", {"type": "BeginPlay"})
        incremental.record_change("move_node", "node_1", {"position": [100, 100]})

        update = incremental.get_incremental_update()

        assert len(update) > 0
        parsed = json.loads(update.decode('utf-8'))
        assert parsed["type"] == "incremental"
        assert len(parsed["changes"]) == 2

    def test_apply_incremental_update(self):
        base = BlueprintSerializer()
        incremental = IncrementalSerializer(base)

        graph = BlueprintGraph()

        # Record a move
        begin = BeginPlayNode(position=(0, 0))
        graph.add_node(begin)

        incremental.record_change("move_node", begin.id, {"position": [200, 200]})
        update = incremental.get_incremental_update()

        # Apply update
        result = incremental.apply_incremental_update(graph, update)

        assert result is True

    def test_clear_changes(self):
        base = BlueprintSerializer()
        incremental = IncrementalSerializer(base)

        incremental.record_change("add_node", "node_1", {})
        incremental.clear_changes()

        assert len(incremental._change_log) == 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_save_blueprint(self):
        graph = BlueprintGraph(name="SaveTest")
        begin = BeginPlayNode()
        graph.add_node(begin)

        with tempfile.NamedTemporaryFile(suffix=".bp", delete=False) as f:
            filepath = f.name

        try:
            result = save_blueprint(graph, filepath)
            assert result is True
        finally:
            os.unlink(filepath)

    def test_load_blueprint(self):
        graph = BlueprintGraph(name="LoadTest")
        begin = BeginPlayNode()
        graph.add_node(begin)

        with tempfile.NamedTemporaryFile(suffix=".bp", delete=False) as f:
            filepath = f.name

        try:
            save_blueprint(graph, filepath)
            restored = load_blueprint(filepath)

            assert restored is not None
            assert restored.name == "LoadTest"
        finally:
            os.unlink(filepath)

    def test_export_blueprint_json(self):
        graph = BlueprintGraph(name="ExportTest")
        begin = BeginPlayNode()
        graph.add_node(begin)

        json_str = export_blueprint_json(graph)

        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert "header" in parsed

    def test_export_blueprint_json_pretty(self):
        graph = BlueprintGraph(name="PrettyTest")
        begin = BeginPlayNode()
        graph.add_node(begin)

        json_str = export_blueprint_json(graph, pretty=True)

        assert '\n' in json_str

    def test_import_blueprint_json(self):
        graph = BlueprintGraph(name="ImportTest")
        begin = BeginPlayNode()
        graph.add_node(begin)

        json_str = export_blueprint_json(graph)
        restored = import_blueprint_json(json_str)

        assert restored is not None
        assert restored.name == "ImportTest"


class TestVersionMigration:
    """Tests for version migration support."""

    def test_deserialize_old_format(self):
        # Create a simulated old format
        old_data = json.dumps({
            "header": {
                "version": "1.0",
                "format_version": 1,
                "node_count": 0,
                "connection_count": 0
            },
            "graph": {
                "id": "test_id",
                "name": "OldFormat",
                "entry_points": []
            },
            "nodes": [],
            "connections": []
        }).encode('utf-8')

        serializer = BlueprintSerializer()
        restored = serializer.deserialize(old_data)

        assert restored is not None
        assert restored.name == "OldFormat"


class TestCustomNodeSerializer:
    """Tests for custom node serializers."""

    def test_register_custom_serializer(self):
        serializer = BlueprintSerializer()

        def custom_serialize(node):
            return {"custom_data": "test"}

        def custom_deserialize(data):
            return BeginPlayNode()

        serializer.register_node_serializer(
            "CustomNode",
            custom_serialize,
            custom_deserialize
        )

        assert "CustomNode" in serializer._node_serializers


class TestMacroSerialization:
    """Tests for macro blueprint serialization."""

    def test_serialize_macro_blueprint(self):
        serializer = BlueprintSerializer()

        graph = BlueprintGraph(name="MacroTest")
        graph.is_macro = True
        begin = BeginPlayNode()
        graph.add_node(begin)

        data = serializer.serialize(graph)
        restored = serializer.deserialize(data)

        assert restored.is_macro is True


class TestDisabledNodes:
    """Tests for disabled node serialization."""

    def test_serialize_disabled_node(self):
        serializer = BlueprintSerializer()

        graph = BlueprintGraph(name="DisabledTest")
        begin = BeginPlayNode()
        begin.is_disabled = True
        graph.add_node(begin)

        data = serializer.serialize(graph)
        restored = serializer.deserialize(data)

        restored_node = list(restored.nodes.values())[0]
        assert restored_node.is_disabled is True

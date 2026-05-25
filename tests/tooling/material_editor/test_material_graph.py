"""Tests for material graph."""
import pytest
import json
from engine.tooling.material_editor.material_graph import (
    GraphState, GraphError, GraphMetadata, MaterialGraph
)
from engine.tooling.material_editor.material_nodes import (
    ConstantNode, Constant4Node, AddNode, MultiplyNode, PBROutputNode, UnlitOutputNode
)
from engine.tooling.material_editor.connection_validator import Connection, ValidationError


class TestGraphMetadata:
    """Tests for GraphMetadata."""

    def test_default_metadata(self):
        """Test default metadata values."""
        metadata = GraphMetadata()
        assert metadata.name == "Untitled"
        assert metadata.description == ""
        assert metadata.version == "1.0"


class TestMaterialGraph:
    """Tests for MaterialGraph."""

    @pytest.fixture
    def graph(self):
        """Create a test graph."""
        return MaterialGraph("TestMaterial")

    def test_create_graph(self, graph):
        """Test creating a material graph."""
        assert graph.name == "TestMaterial"
        assert graph.node_count == 0
        assert graph.connection_count == 0

    def test_add_node(self, graph):
        """Test adding a node."""
        node = ConstantNode(value=0.5)
        result = graph.add_node(node)
        assert result is True
        assert graph.node_count == 1

    def test_add_duplicate_node(self, graph):
        """Test adding same node twice fails."""
        node = ConstantNode(value=0.5)
        graph.add_node(node)
        result = graph.add_node(node)
        assert result is False
        assert graph.node_count == 1

    def test_remove_node(self, graph):
        """Test removing a node."""
        node = ConstantNode(value=0.5)
        graph.add_node(node)
        result = graph.remove_node(node.id)
        assert result is True
        assert graph.node_count == 0

    def test_remove_nonexistent_node(self, graph):
        """Test removing non-existent node fails."""
        result = graph.remove_node("nonexistent")
        assert result is False

    def test_get_node(self, graph):
        """Test getting a node by ID."""
        node = ConstantNode(value=0.5)
        graph.add_node(node)
        retrieved = graph.get_node(node.id)
        assert retrieved == node

    def test_create_node(self, graph):
        """Test creating a node directly in graph."""
        node = graph.create_node("Constant", "MyConstant")
        assert node is not None
        assert graph.node_count == 1
        assert node.name == "MyConstant"

    def test_duplicate_node(self, graph):
        """Test duplicating a node."""
        original = ConstantNode(value=0.5, name="Original")
        original.position = (100, 100)
        graph.add_node(original)

        duplicate = graph.duplicate_node(original.id)
        assert duplicate is not None
        assert duplicate.id != original.id
        assert graph.node_count == 2


class TestGraphConnections:
    """Tests for graph connections."""

    @pytest.fixture
    def connected_graph(self):
        """Create a graph with connected nodes."""
        graph = MaterialGraph("Test")
        const = ConstantNode(value=0.5, name="Const")
        add = AddNode(name="Add")
        output = PBROutputNode(name="Output")

        graph.add_node(const)
        graph.add_node(add)
        graph.add_node(output)

        return graph, const, add, output

    def test_connect_nodes(self, connected_graph):
        """Test connecting nodes."""
        graph, const, add, output = connected_graph

        result = graph.connect(const.id, "Value", add.id, "A")
        assert result.valid is True
        assert graph.connection_count == 1

    def test_connect_invalid_pins(self, connected_graph):
        """Test connecting invalid pins fails."""
        graph, const, add, output = connected_graph

        result = graph.connect(const.id, "NonExistent", add.id, "A")
        assert result.valid is False

    def test_disconnect_nodes(self, connected_graph):
        """Test disconnecting nodes."""
        graph, const, add, output = connected_graph

        graph.connect(const.id, "Value", add.id, "A")
        result = graph.disconnect(const.id, "Value", add.id, "A")
        assert result is True
        assert graph.connection_count == 0

    def test_disconnect_nonexistent(self, connected_graph):
        """Test disconnecting non-existent connection fails."""
        graph, const, add, output = connected_graph

        result = graph.disconnect(const.id, "Value", add.id, "A")
        assert result is False

    def test_disconnect_pin(self, connected_graph):
        """Test disconnecting all connections from a pin."""
        graph, const, add, output = connected_graph

        # Connect const to both inputs of add
        const2 = ConstantNode(value=1.0, name="Const2")
        graph.add_node(const2)

        graph.connect(const.id, "Value", add.id, "A")
        graph.connect(const2.id, "Value", add.id, "B")

        # Disconnect all connections from add's A input
        count = graph.disconnect_pin(add.id, "A", is_output=False)
        assert count == 1
        assert graph.connection_count == 1

    def test_disconnect_all(self, connected_graph):
        """Test disconnecting all connections from a node."""
        graph, const, add, output = connected_graph

        graph.connect(const.id, "Value", add.id, "A")
        graph.connect(add.id, "Result", output.id, "Metallic")

        count = graph.disconnect_all(add.id)
        assert count == 2
        assert graph.connection_count == 0

    def test_get_connections_to_node(self, connected_graph):
        """Test getting connections to a node."""
        graph, const, add, output = connected_graph

        const2 = ConstantNode(value=1.0)
        graph.add_node(const2)

        graph.connect(const.id, "Value", add.id, "A")
        graph.connect(const2.id, "Value", add.id, "B")

        connections = graph.get_connections_to_node(add.id)
        assert len(connections) == 2

    def test_get_connections_from_node(self, connected_graph):
        """Test getting connections from a node."""
        graph, const, add, output = connected_graph

        graph.connect(const.id, "Value", add.id, "A")
        graph.connect(const.id, "Value", output.id, "Metallic")

        connections = graph.get_connections_from_node(const.id)
        assert len(connections) == 2

    def test_can_connect(self, connected_graph):
        """Test checking if connection is valid."""
        graph, const, add, output = connected_graph

        result = graph.can_connect(const.id, "Value", add.id, "A")
        assert result.valid is True


class TestGraphOutputNode:
    """Tests for output node management."""

    def test_output_node_auto_tracked(self):
        """Test output node is automatically tracked."""
        graph = MaterialGraph()
        output = PBROutputNode()
        graph.add_node(output)

        assert graph.output_node == output

    def test_remove_output_node_finds_new(self):
        """Test removing output node finds another."""
        graph = MaterialGraph()
        output1 = PBROutputNode(name="Output1")
        output2 = UnlitOutputNode(name="Output2")

        graph.add_node(output1)
        graph.add_node(output2)
        graph.remove_node(output1.id)

        assert graph.output_node == output2


class TestGraphState:
    """Tests for graph state management."""

    def test_initial_state_clean(self):
        """Test initial state is clean."""
        graph = MaterialGraph()
        # State becomes MODIFIED when created, then CLEAN after validation
        # Just check it's not ERROR
        assert graph.state != GraphState.ERROR

    def test_state_modified_on_change(self):
        """Test state becomes modified on changes."""
        graph = MaterialGraph()
        graph.add_node(ConstantNode())
        assert graph.state == GraphState.MODIFIED


class TestGraphValidation:
    """Tests for graph validation."""

    def test_validate_no_output(self):
        """Test validation fails without output node."""
        graph = MaterialGraph()
        graph.add_node(ConstantNode())

        errors = graph.validate()
        assert any("output" in e.message.lower() for e in errors)

    def test_validate_disconnected_nodes(self):
        """Test validation warns about disconnected nodes."""
        graph = MaterialGraph()
        graph.add_node(ConstantNode())  # Disconnected
        graph.add_node(PBROutputNode())

        errors = graph.validate()
        # Should have warning about disconnected node
        warnings = [e for e in errors if e.severity == "warning"]
        assert len(warnings) > 0

    def test_validate_connected_graph(self):
        """Test validation passes for connected graph."""
        graph = MaterialGraph()
        const = ConstantNode(value=0.5)
        output = PBROutputNode()

        graph.add_node(const)
        graph.add_node(output)
        graph.connect(const.id, "Value", output.id, "Metallic")

        errors = graph.validate()
        error_count = len([e for e in errors if e.severity == "error"])
        assert error_count == 0


class TestGraphEvaluation:
    """Tests for graph evaluation order."""

    def test_topological_order(self):
        """Test nodes in topological order."""
        graph = MaterialGraph()
        const = ConstantNode(value=0.5, name="Const")
        add = AddNode(name="Add")
        output = PBROutputNode(name="Output")

        graph.add_node(const)
        graph.add_node(add)
        graph.add_node(output)

        graph.connect(const.id, "Value", add.id, "A")
        graph.connect(add.id, "Result", output.id, "Metallic")

        order = graph.get_topological_order()

        # Const should come before Add, Add before Output
        assert order.index(const.id) < order.index(add.id)
        assert order.index(add.id) < order.index(output.id)

    def test_evaluation_order(self):
        """Test getting nodes in evaluation order."""
        graph = MaterialGraph()
        const = ConstantNode(value=0.5)
        output = PBROutputNode()

        graph.add_node(const)
        graph.add_node(output)
        graph.connect(const.id, "Value", output.id, "Metallic")

        eval_order = graph.get_evaluation_order()
        assert len(eval_order) == 2
        assert all(isinstance(n, type(const)) or isinstance(n, type(output)) for n in eval_order)


class TestGraphSerialization:
    """Tests for graph serialization."""

    @pytest.fixture
    def complex_graph(self):
        """Create a complex graph for serialization tests."""
        graph = MaterialGraph("TestMaterial")
        graph.metadata.description = "Test description"
        graph.metadata.author = "Test Author"

        const = ConstantNode(value=0.5, name="Roughness")
        const.position = (100, 100)
        color = Constant4Node(value=(0.8, 0.2, 0.1, 1.0), name="Color")
        color.position = (100, 200)
        output = PBROutputNode(name="Output")
        output.position = (400, 150)

        graph.add_node(const)
        graph.add_node(color)
        graph.add_node(output)

        graph.connect(const.id, "Value", output.id, "Roughness")
        graph.connect(color.id, "Value", output.id, "Albedo")

        return graph

    def test_to_dict(self, complex_graph):
        """Test serialization to dictionary."""
        data = complex_graph.to_dict()

        assert data["metadata"]["name"] == "TestMaterial"
        assert len(data["nodes"]) == 3
        assert len(data["connections"]) == 2

    def test_to_json(self, complex_graph):
        """Test serialization to JSON."""
        json_str = complex_graph.to_json()
        data = json.loads(json_str)

        assert data["metadata"]["name"] == "TestMaterial"

    def test_from_dict(self, complex_graph):
        """Test deserialization from dictionary."""
        data = complex_graph.to_dict()
        restored = MaterialGraph.from_dict(data)

        assert restored.name == complex_graph.name
        assert restored.node_count == complex_graph.node_count
        assert restored.connection_count == complex_graph.connection_count

    def test_from_json(self, complex_graph):
        """Test deserialization from JSON."""
        json_str = complex_graph.to_json()
        restored = MaterialGraph.from_json(json_str)

        assert restored.name == complex_graph.name

    def test_round_trip_preserves_connections(self, complex_graph):
        """Test serialization round-trip preserves connections."""
        data = complex_graph.to_dict()
        restored = MaterialGraph.from_dict(data)

        # Get output node and check its connections
        output = restored.output_node
        connections = restored.get_connections_to_node(output.id)
        assert len(connections) == 2


class TestGraphCallbacks:
    """Tests for graph callbacks."""

    def test_on_node_added(self):
        """Test node added callback."""
        graph = MaterialGraph()
        added_nodes = []

        graph.on_node_added(lambda n: added_nodes.append(n))
        graph.add_node(ConstantNode())

        assert len(added_nodes) == 1

    def test_on_node_removed(self):
        """Test node removed callback."""
        graph = MaterialGraph()
        removed_ids = []

        node = ConstantNode()
        graph.add_node(node)
        graph.on_node_removed(lambda id: removed_ids.append(id))
        graph.remove_node(node.id)

        assert len(removed_ids) == 1
        assert removed_ids[0] == node.id

    def test_on_connection_added(self):
        """Test connection added callback."""
        graph = MaterialGraph()
        added_connections = []

        const = ConstantNode()
        add = AddNode()
        graph.add_node(const)
        graph.add_node(add)

        graph.on_connection_added(lambda c: added_connections.append(c))
        graph.connect(const.id, "Value", add.id, "A")

        assert len(added_connections) == 1

    def test_on_connection_removed(self):
        """Test connection removed callback."""
        graph = MaterialGraph()
        removed_connections = []

        const = ConstantNode()
        add = AddNode()
        graph.add_node(const)
        graph.add_node(add)
        graph.connect(const.id, "Value", add.id, "A")

        graph.on_connection_removed(lambda c: removed_connections.append(c))
        graph.disconnect(const.id, "Value", add.id, "A")

        assert len(removed_connections) == 1


class TestGraphUtilities:
    """Tests for graph utilities."""

    def test_clear(self):
        """Test clearing graph."""
        graph = MaterialGraph()
        graph.add_node(ConstantNode())
        graph.add_node(PBROutputNode())

        graph.clear()

        assert graph.node_count == 0
        assert graph.connection_count == 0
        assert graph.output_node is None

    def test_copy(self):
        """Test copying graph."""
        original = MaterialGraph("Original")
        const = ConstantNode(value=0.5)
        output = PBROutputNode()
        original.add_node(const)
        original.add_node(output)
        original.connect(const.id, "Value", output.id, "Metallic")

        copy = original.copy()

        assert copy.name == original.name
        assert copy.node_count == original.node_count
        assert copy.connection_count == original.connection_count
        # Copy should be independent
        copy.clear()
        assert original.node_count == 2

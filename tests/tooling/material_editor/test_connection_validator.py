"""Tests for connection validator."""
import pytest
from engine.tooling.material_editor.connection_validator import (
    ValidationError, ValidationResult, Connection, TypeConversion, ConnectionValidator
)
from engine.tooling.material_editor.material_nodes import (
    DataType, ConstantNode, Constant3Node, AddNode, TextureSampleNode, PBROutputNode
)


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_success_result(self):
        """Test successful validation result."""
        result = ValidationResult.success()
        assert result.valid is True
        assert result.error == ValidationError.NONE

    def test_success_with_warnings(self):
        """Test successful result with warnings."""
        result = ValidationResult.success(warnings=["Type conversion"])
        assert result.valid is True
        assert len(result.warnings) == 1

    def test_failure_result(self):
        """Test failed validation result."""
        result = ValidationResult.failure(ValidationError.SAME_NODE, "Can't connect to self")
        assert result.valid is False
        assert result.error == ValidationError.SAME_NODE
        assert "self" in result.message


class TestConnection:
    """Tests for Connection dataclass."""

    def test_create_connection(self):
        """Test creating connection."""
        conn = Connection("node1", "Output", "node2", "Input")
        assert conn.source_node_id == "node1"
        assert conn.source_pin == "Output"
        assert conn.target_node_id == "node2"
        assert conn.target_pin == "Input"

    def test_connection_equality(self):
        """Test connection equality."""
        conn1 = Connection("a", "out", "b", "in")
        conn2 = Connection("a", "out", "b", "in")
        assert conn1 == conn2

    def test_connection_hash(self):
        """Test connections can be used in sets."""
        conn1 = Connection("a", "out", "b", "in")
        conn2 = Connection("a", "out", "b", "in")
        conn_set = {conn1, conn2}
        assert len(conn_set) == 1


class TestTypeConversion:
    """Tests for TypeConversion."""

    def test_same_type_compatible(self):
        """Test same types are compatible."""
        can, implicit, warning = TypeConversion.can_convert(DataType.FLOAT, DataType.FLOAT)
        assert can is True
        assert implicit is True

    def test_float_to_vector_compatible(self):
        """Test float can convert to vectors."""
        can, implicit, warning = TypeConversion.can_convert(DataType.FLOAT, DataType.FLOAT3)
        assert can is True
        assert implicit is True

    def test_vector_truncation_has_warning(self):
        """Test vector truncation produces warning."""
        can, implicit, warning = TypeConversion.can_convert(DataType.FLOAT3, DataType.FLOAT)
        assert can is True
        assert warning is not None
        assert "Truncating" in warning

    def test_any_type_compatible(self):
        """Test ANY type is compatible with everything."""
        can, implicit, warning = TypeConversion.can_convert(DataType.ANY, DataType.TEXTURE2D)
        assert can is True

    def test_incompatible_types(self):
        """Test incompatible types."""
        can, implicit, warning = TypeConversion.can_convert(DataType.TEXTURE2D, DataType.FLOAT)
        assert can is False

    def test_conversion_code_float_broadcast(self):
        """Test conversion code for float broadcast."""
        code = TypeConversion.get_conversion_code(DataType.FLOAT, DataType.FLOAT3, "x")
        assert "float3" in code
        assert "x" in code

    def test_conversion_code_truncation(self):
        """Test conversion code for truncation."""
        code = TypeConversion.get_conversion_code(DataType.FLOAT3, DataType.FLOAT, "v")
        assert ".x" in code


class TestConnectionValidator:
    """Tests for ConnectionValidator."""

    @pytest.fixture
    def validator(self):
        """Create a validator with test nodes."""
        validator = ConnectionValidator()
        validator.register_node(ConstantNode(value=1.0, name="const1"))
        validator.register_node(ConstantNode(value=2.0, name="const2"))
        validator.register_node(AddNode(name="add"))
        return validator

    def test_register_node(self, validator):
        """Test registering nodes."""
        assert validator.node_count == 3

    def test_unregister_node(self, validator):
        """Test unregistering nodes."""
        # Get a node ID first
        nodes = list(validator._nodes.values())
        node_id = nodes[0].id
        validator.unregister_node(node_id)
        assert validator.node_count == 2

    def test_valid_connection(self, validator):
        """Test validating a valid connection."""
        const_node = list(validator._nodes.values())[0]
        add_node = [n for n in validator._nodes.values() if isinstance(n, AddNode)][0]

        result = validator.validate_connection(
            const_node.id, "Value",
            add_node.id, "A"
        )
        assert result.valid is True

    def test_same_node_invalid(self, validator):
        """Test connecting node to itself is invalid."""
        add_node = [n for n in validator._nodes.values() if isinstance(n, AddNode)][0]

        result = validator.validate_connection(
            add_node.id, "Result",
            add_node.id, "A"
        )
        assert result.valid is False
        assert result.error == ValidationError.SAME_NODE

    def test_null_node_invalid(self, validator):
        """Test connecting to non-existent node is invalid."""
        const_node = list(validator._nodes.values())[0]

        result = validator.validate_connection(
            const_node.id, "Value",
            "nonexistent", "A"
        )
        assert result.valid is False
        assert result.error == ValidationError.NULL_NODE

    def test_pin_not_found(self, validator):
        """Test invalid pin name."""
        const_node = list(validator._nodes.values())[0]
        add_node = [n for n in validator._nodes.values() if isinstance(n, AddNode)][0]

        result = validator.validate_connection(
            const_node.id, "NonExistent",
            add_node.id, "A"
        )
        assert result.valid is False
        assert result.error == ValidationError.PIN_NOT_FOUND

    def test_output_to_output_invalid(self, validator):
        """Test connecting two outputs is invalid."""
        nodes = list(validator._nodes.values())
        const1 = nodes[0]
        const2 = nodes[1]

        result = validator.validate_connection(
            const1.id, "Value",
            const2.id, "Value"  # Both are outputs
        )
        assert result.valid is False
        assert result.error == ValidationError.PIN_NOT_FOUND  # "Value" is output, not input

    def test_add_connection(self, validator):
        """Test adding a connection."""
        const_node = list(validator._nodes.values())[0]
        add_node = [n for n in validator._nodes.values() if isinstance(n, AddNode)][0]

        validator.validate_connection(const_node.id, "Value", add_node.id, "A")
        conn = Connection(const_node.id, "Value", add_node.id, "A")
        validator.add_connection(conn)

        assert validator.connection_count == 1

    def test_already_connected_invalid(self, validator):
        """Test connecting to already-connected input is invalid."""
        const_nodes = [n for n in validator._nodes.values() if isinstance(n, ConstantNode)]
        add_node = [n for n in validator._nodes.values() if isinstance(n, AddNode)][0]

        # First connection
        conn = Connection(const_nodes[0].id, "Value", add_node.id, "A")
        validator.add_connection(conn)

        # Second connection to same input
        result = validator.validate_connection(
            const_nodes[1].id, "Value",
            add_node.id, "A"
        )
        assert result.valid is False
        assert result.error == ValidationError.ALREADY_CONNECTED

    def test_remove_connection(self, validator):
        """Test removing a connection."""
        const_node = list(validator._nodes.values())[0]
        add_node = [n for n in validator._nodes.values() if isinstance(n, AddNode)][0]

        conn = Connection(const_node.id, "Value", add_node.id, "A")
        validator.add_connection(conn)
        validator.remove_connection(conn)

        assert validator.connection_count == 0

    def test_get_connections_to_node(self, validator):
        """Test getting connections to a node."""
        const_nodes = [n for n in validator._nodes.values() if isinstance(n, ConstantNode)]
        add_node = [n for n in validator._nodes.values() if isinstance(n, AddNode)][0]

        conn1 = Connection(const_nodes[0].id, "Value", add_node.id, "A")
        conn2 = Connection(const_nodes[1].id, "Value", add_node.id, "B")
        validator.add_connection(conn1)
        validator.add_connection(conn2)

        connections = validator.get_connections_to_node(add_node.id)
        assert len(connections) == 2

    def test_get_connection_to_pin(self, validator):
        """Test getting connection to specific pin."""
        const_node = list(validator._nodes.values())[0]
        add_node = [n for n in validator._nodes.values() if isinstance(n, AddNode)][0]

        conn = Connection(const_node.id, "Value", add_node.id, "A")
        validator.add_connection(conn)

        result = validator.get_connection_to_pin(add_node.id, "A")
        assert result == conn

        result = validator.get_connection_to_pin(add_node.id, "B")
        assert result is None


class TestCycleDetection:
    """Tests for cycle detection."""

    @pytest.fixture
    def chain_validator(self):
        """Create validator with chain of nodes."""
        validator = ConnectionValidator()

        # Create chain: A -> B -> C
        node_a = AddNode(name="A")
        node_b = AddNode(name="B")
        node_c = AddNode(name="C")

        validator.register_node(node_a)
        validator.register_node(node_b)
        validator.register_node(node_c)

        # Connect A -> B -> C
        conn1 = Connection(node_a.id, "Result", node_b.id, "A")
        conn2 = Connection(node_b.id, "Result", node_c.id, "A")
        validator.add_connection(conn1)
        validator.add_connection(conn2)

        return validator, node_a, node_b, node_c

    def test_no_cycle_valid(self, chain_validator):
        """Test valid non-cyclic connection."""
        validator, node_a, node_b, node_c = chain_validator

        # This should be valid (no cycle)
        result = validator.validate_connection(
            node_a.id, "Result",
            node_c.id, "B"
        )
        assert result.valid is True

    def test_cycle_invalid(self, chain_validator):
        """Test cyclic connection is invalid."""
        validator, node_a, node_b, node_c = chain_validator

        # C -> A would create cycle
        result = validator.validate_connection(
            node_c.id, "Result",
            node_a.id, "A"
        )
        assert result.valid is False
        assert result.error == ValidationError.WOULD_CREATE_CYCLE


class TestTopologicalOrder:
    """Tests for topological ordering."""

    def test_topological_order(self):
        """Test getting nodes in topological order."""
        validator = ConnectionValidator()

        node_a = ConstantNode(name="A")
        node_b = ConstantNode(name="B")
        node_add = AddNode(name="Add")
        output = PBROutputNode(name="Output")

        validator.register_node(node_a)
        validator.register_node(node_b)
        validator.register_node(node_add)
        validator.register_node(output)

        # A, B -> Add -> Output
        validator.add_connection(Connection(node_a.id, "Value", node_add.id, "A"))
        validator.add_connection(Connection(node_b.id, "Value", node_add.id, "B"))
        validator.add_connection(Connection(node_add.id, "Result", output.id, "Metallic"))

        order = validator.get_topological_order()

        # A and B should come before Add, Add before Output
        assert order.index(node_a.id) < order.index(node_add.id)
        assert order.index(node_b.id) < order.index(node_add.id)
        assert order.index(node_add.id) < order.index(output.id)

    def test_find_disconnected_nodes(self):
        """Test finding disconnected nodes."""
        validator = ConnectionValidator()

        connected = ConstantNode(name="Connected")
        disconnected = ConstantNode(name="Disconnected")
        add_node = AddNode(name="Add")

        validator.register_node(connected)
        validator.register_node(disconnected)
        validator.register_node(add_node)

        validator.add_connection(Connection(connected.id, "Value", add_node.id, "A"))

        orphans = validator.find_disconnected_nodes()
        assert disconnected.id in orphans
        assert connected.id not in orphans
        assert add_node.id not in orphans

    def test_find_unconnected_inputs(self):
        """Test finding unconnected inputs."""
        validator = ConnectionValidator()

        const_node = ConstantNode(name="Const")
        add_node = AddNode(name="Add")

        validator.register_node(const_node)
        validator.register_node(add_node)

        # Only connect to A, B is unconnected
        validator.add_connection(Connection(const_node.id, "Value", add_node.id, "A"))

        unconnected = validator.find_unconnected_inputs(add_node.id)
        assert "B" in unconnected
        assert "A" not in unconnected

    def test_clear(self):
        """Test clearing validator."""
        validator = ConnectionValidator()
        validator.register_node(ConstantNode())
        validator.register_node(AddNode())

        validator.clear()

        assert validator.node_count == 0
        assert validator.connection_count == 0

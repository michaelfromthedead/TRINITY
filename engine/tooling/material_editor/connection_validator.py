"""Connection validator - Type-safe connection validation between nodes."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .material_nodes import MaterialNode, NodePin, DataType


class ValidationError(Enum):
    """Types of connection validation errors."""
    NONE = auto()
    SAME_NODE = auto()
    SAME_DIRECTION = auto()
    INCOMPATIBLE_TYPES = auto()
    WOULD_CREATE_CYCLE = auto()
    PIN_NOT_FOUND = auto()
    ALREADY_CONNECTED = auto()
    NULL_NODE = auto()


@dataclass
class ValidationResult:
    """Result of connection validation."""
    valid: bool
    error: ValidationError = ValidationError.NONE
    message: str = ""
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    @staticmethod
    def success(warnings: List[str] = None) -> 'ValidationResult':
        """Create a successful validation result."""
        return ValidationResult(valid=True, warnings=warnings or [])

    @staticmethod
    def failure(error: ValidationError, message: str) -> 'ValidationResult':
        """Create a failed validation result."""
        return ValidationResult(valid=False, error=error, message=message)


@dataclass
class Connection:
    """Represents a connection between two node pins."""
    source_node_id: str
    source_pin: str
    target_node_id: str
    target_pin: str

    def __hash__(self):
        return hash((self.source_node_id, self.source_pin,
                     self.target_node_id, self.target_pin))

    def __eq__(self, other):
        if not isinstance(other, Connection):
            return False
        return (self.source_node_id == other.source_node_id and
                self.source_pin == other.source_pin and
                self.target_node_id == other.target_node_id and
                self.target_pin == other.target_pin)


class TypeConversion:
    """Handles type conversion between different data types."""

    # Conversion rules: (from_type, to_type) -> (can_convert, implicit, warning)
    # implicit means automatic conversion, non-implicit means explicit cast needed
    _CONVERSIONS: Dict[Tuple[str, str], Tuple[bool, bool, Optional[str]]] = {}

    @classmethod
    def _init_conversions(cls) -> None:
        """Initialize conversion rules."""
        from .material_nodes import DataType

        # Float can be broadcast to vectors
        cls._CONVERSIONS[(DataType.FLOAT.name, DataType.FLOAT2.name)] = (True, True, None)
        cls._CONVERSIONS[(DataType.FLOAT.name, DataType.FLOAT3.name)] = (True, True, None)
        cls._CONVERSIONS[(DataType.FLOAT.name, DataType.FLOAT4.name)] = (True, True, None)

        # Vectors can be truncated (with warning)
        cls._CONVERSIONS[(DataType.FLOAT2.name, DataType.FLOAT.name)] = (True, False, "Truncating float2 to float, using .x")
        cls._CONVERSIONS[(DataType.FLOAT3.name, DataType.FLOAT.name)] = (True, False, "Truncating float3 to float, using .x")
        cls._CONVERSIONS[(DataType.FLOAT4.name, DataType.FLOAT.name)] = (True, False, "Truncating float4 to float, using .x")
        cls._CONVERSIONS[(DataType.FLOAT3.name, DataType.FLOAT2.name)] = (True, False, "Truncating float3 to float2, using .xy")
        cls._CONVERSIONS[(DataType.FLOAT4.name, DataType.FLOAT2.name)] = (True, False, "Truncating float4 to float2, using .xy")
        cls._CONVERSIONS[(DataType.FLOAT4.name, DataType.FLOAT3.name)] = (True, False, "Truncating float4 to float3, using .xyz")

        # Vectors can be expanded (with warning)
        cls._CONVERSIONS[(DataType.FLOAT2.name, DataType.FLOAT3.name)] = (True, False, "Expanding float2 to float3, z=0")
        cls._CONVERSIONS[(DataType.FLOAT2.name, DataType.FLOAT4.name)] = (True, False, "Expanding float2 to float4, zw=0,1")
        cls._CONVERSIONS[(DataType.FLOAT3.name, DataType.FLOAT4.name)] = (True, False, "Expanding float3 to float4, w=1")

        # Int/Bool conversions
        cls._CONVERSIONS[(DataType.INT.name, DataType.FLOAT.name)] = (True, True, None)
        cls._CONVERSIONS[(DataType.FLOAT.name, DataType.INT.name)] = (True, False, "Truncating float to int")
        cls._CONVERSIONS[(DataType.BOOL.name, DataType.FLOAT.name)] = (True, True, None)
        cls._CONVERSIONS[(DataType.BOOL.name, DataType.INT.name)] = (True, True, None)

    @classmethod
    def can_convert(cls, from_type: 'DataType', to_type: 'DataType') -> Tuple[bool, bool, Optional[str]]:
        """
        Check if type conversion is possible.

        Returns:
            Tuple of (can_convert, is_implicit, warning_message)
        """
        from .material_nodes import DataType

        if not cls._CONVERSIONS:
            cls._init_conversions()

        # Same type always works
        if from_type == to_type:
            return (True, True, None)

        # ANY type always works
        if from_type == DataType.ANY or to_type == DataType.ANY:
            return (True, True, None)

        key = (from_type.name, to_type.name)
        return cls._CONVERSIONS.get(key, (False, False, None))

    @classmethod
    def get_conversion_code(cls, from_type: 'DataType', to_type: 'DataType', var_name: str) -> str:
        """Generate code for type conversion."""
        from .material_nodes import DataType

        if from_type == to_type or from_type == DataType.ANY or to_type == DataType.ANY:
            return var_name

        type_map = {
            DataType.FLOAT: "float",
            DataType.FLOAT2: "float2",
            DataType.FLOAT3: "float3",
            DataType.FLOAT4: "float4",
            DataType.INT: "int",
            DataType.BOOL: "bool",
        }

        # Float broadcast
        if from_type == DataType.FLOAT:
            if to_type == DataType.FLOAT2:
                return f"{type_map[to_type]}({var_name}, {var_name})"
            elif to_type == DataType.FLOAT3:
                return f"{type_map[to_type]}({var_name}, {var_name}, {var_name})"
            elif to_type == DataType.FLOAT4:
                return f"{type_map[to_type]}({var_name}, {var_name}, {var_name}, {var_name})"

        # Truncation
        if from_type in (DataType.FLOAT2, DataType.FLOAT3, DataType.FLOAT4):
            if to_type == DataType.FLOAT:
                return f"{var_name}.x"
            elif to_type == DataType.FLOAT2:
                return f"{var_name}.xy"
            elif to_type == DataType.FLOAT3:
                return f"{var_name}.xyz"

        # Expansion
        if from_type == DataType.FLOAT2:
            if to_type == DataType.FLOAT3:
                return f"{type_map[to_type]}({var_name}, 0.0)"
            elif to_type == DataType.FLOAT4:
                return f"{type_map[to_type]}({var_name}, 0.0, 1.0)"
        elif from_type == DataType.FLOAT3 and to_type == DataType.FLOAT4:
            return f"{type_map[to_type]}({var_name}, 1.0)"

        # Int/Bool conversions
        if from_type == DataType.INT and to_type == DataType.FLOAT:
            return f"(float){var_name}"
        elif from_type == DataType.FLOAT and to_type == DataType.INT:
            return f"(int){var_name}"
        elif from_type == DataType.BOOL:
            return f"({type_map.get(to_type, 'float')}){var_name}"

        return f"({type_map.get(to_type, 'float')}){var_name}"


class ConnectionValidator:
    """Validates connections between material nodes."""

    def __init__(self):
        self._nodes: Dict[str, 'MaterialNode'] = {}
        self._connections: Set[Connection] = set()
        self._adjacency: Dict[str, Set[str]] = {}  # node_id -> set of connected node ids

    def register_node(self, node: 'MaterialNode') -> None:
        """Register a node for validation."""
        self._nodes[node.id] = node
        self._adjacency[node.id] = set()

    def unregister_node(self, node_id: str) -> None:
        """Unregister a node."""
        if node_id in self._nodes:
            del self._nodes[node_id]
        if node_id in self._adjacency:
            del self._adjacency[node_id]
        # Remove all connections involving this node
        self._connections = {
            conn for conn in self._connections
            if conn.source_node_id != node_id and conn.target_node_id != node_id
        }
        # Update adjacency
        for adj_set in self._adjacency.values():
            adj_set.discard(node_id)

    def add_connection(self, connection: Connection) -> None:
        """Add a validated connection."""
        self._connections.add(connection)
        self._adjacency[connection.source_node_id].add(connection.target_node_id)

    def remove_connection(self, connection: Connection) -> None:
        """Remove a connection."""
        self._connections.discard(connection)
        # Update adjacency (check if there are still other connections)
        still_connected = any(
            conn.source_node_id == connection.source_node_id and
            conn.target_node_id == connection.target_node_id
            for conn in self._connections
        )
        if not still_connected:
            self._adjacency[connection.source_node_id].discard(connection.target_node_id)

    def get_connections_to_node(self, node_id: str) -> List[Connection]:
        """Get all connections to a node (as target)."""
        return [conn for conn in self._connections if conn.target_node_id == node_id]

    def get_connections_from_node(self, node_id: str) -> List[Connection]:
        """Get all connections from a node (as source)."""
        return [conn for conn in self._connections if conn.source_node_id == node_id]

    def get_connection_to_pin(self, node_id: str, pin_name: str) -> Optional[Connection]:
        """Get connection to a specific input pin."""
        for conn in self._connections:
            if conn.target_node_id == node_id and conn.target_pin == pin_name:
                return conn
        return None

    def validate_connection(
        self,
        source_node_id: str,
        source_pin: str,
        target_node_id: str,
        target_pin: str
    ) -> ValidationResult:
        """
        Validate a potential connection.

        Args:
            source_node_id: ID of the source node
            source_pin: Name of the output pin on source
            target_node_id: ID of the target node
            target_pin: Name of the input pin on target

        Returns:
            ValidationResult with validity and any warnings/errors
        """
        from .material_nodes import DataType

        # Check for null nodes
        source_node = self._nodes.get(source_node_id)
        target_node = self._nodes.get(target_node_id)

        if source_node is None:
            return ValidationResult.failure(
                ValidationError.NULL_NODE,
                f"Source node '{source_node_id}' not found"
            )
        if target_node is None:
            return ValidationResult.failure(
                ValidationError.NULL_NODE,
                f"Target node '{target_node_id}' not found"
            )

        # Can't connect to self
        if source_node_id == target_node_id:
            return ValidationResult.failure(
                ValidationError.SAME_NODE,
                "Cannot connect a node to itself"
            )

        # Check pins exist
        output_pin = source_node.get_output(source_pin)
        input_pin = target_node.get_input(target_pin)

        if output_pin is None:
            return ValidationResult.failure(
                ValidationError.PIN_NOT_FOUND,
                f"Output pin '{source_pin}' not found on node '{source_node.name}'"
            )
        if input_pin is None:
            return ValidationResult.failure(
                ValidationError.PIN_NOT_FOUND,
                f"Input pin '{target_pin}' not found on node '{target_node.name}'"
            )

        # Check we're connecting output to input
        if not output_pin.is_output:
            return ValidationResult.failure(
                ValidationError.SAME_DIRECTION,
                "Source pin must be an output"
            )
        if input_pin.is_output:
            return ValidationResult.failure(
                ValidationError.SAME_DIRECTION,
                "Target pin must be an input"
            )

        # Check type compatibility
        can_convert, is_implicit, warning = TypeConversion.can_convert(
            output_pin.data_type,
            input_pin.data_type
        )

        if not can_convert:
            return ValidationResult.failure(
                ValidationError.INCOMPATIBLE_TYPES,
                f"Cannot convert {output_pin.data_type.name} to {input_pin.data_type.name}"
            )

        # Check for existing connection to this input pin
        existing = self.get_connection_to_pin(target_node_id, target_pin)
        if existing:
            return ValidationResult.failure(
                ValidationError.ALREADY_CONNECTED,
                f"Input pin '{target_pin}' already has a connection"
            )

        # Check for cycles
        if self._would_create_cycle(source_node_id, target_node_id):
            return ValidationResult.failure(
                ValidationError.WOULD_CREATE_CYCLE,
                "Connection would create a cycle in the graph"
            )

        # Success with optional warnings
        warnings = []
        if warning:
            warnings.append(warning)

        return ValidationResult.success(warnings)

    def _would_create_cycle(self, source_id: str, target_id: str) -> bool:
        """
        Check if adding an edge from source to target would create a cycle.

        This uses DFS to check if there's a path from target back to source.
        """
        # If there's a path from target to source, adding source->target creates cycle
        visited = set()
        stack = [target_id]

        while stack:
            current = stack.pop()
            if current == source_id:
                return True

            if current in visited:
                continue
            visited.add(current)

            # Add all nodes that this node connects to
            for conn in self._connections:
                if conn.source_node_id == current:
                    stack.append(conn.target_node_id)

        return False

    def get_topological_order(self) -> List[str]:
        """
        Get nodes in topological order (for evaluation/compilation).

        Returns:
            List of node IDs in topological order

        Raises:
            ValueError: If graph has cycles (shouldn't happen with validation)
        """
        in_degree: Dict[str, int] = {node_id: 0 for node_id in self._nodes}

        for conn in self._connections:
            in_degree[conn.target_node_id] += 1

        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for conn in self._connections:
                if conn.source_node_id == node_id:
                    in_degree[conn.target_node_id] -= 1
                    if in_degree[conn.target_node_id] == 0:
                        queue.append(conn.target_node_id)

        if len(result) != len(self._nodes):
            raise ValueError("Graph contains cycles")

        return result

    def find_disconnected_nodes(self) -> List[str]:
        """Find nodes that have no connections (orphans)."""
        connected = set()
        for conn in self._connections:
            connected.add(conn.source_node_id)
            connected.add(conn.target_node_id)

        return [node_id for node_id in self._nodes if node_id not in connected]

    def find_unconnected_inputs(self, node_id: str) -> List[str]:
        """Find input pins that have no incoming connections."""
        node = self._nodes.get(node_id)
        if not node:
            return []

        connected_inputs = {
            conn.target_pin for conn in self._connections
            if conn.target_node_id == node_id
        }

        return [
            pin_name for pin_name in node.inputs.keys()
            if pin_name not in connected_inputs
        ]

    def clear(self) -> None:
        """Clear all nodes and connections."""
        self._nodes.clear()
        self._connections.clear()
        self._adjacency.clear()

    @property
    def connections(self) -> Set[Connection]:
        """Get all connections."""
        return self._connections.copy()

    @property
    def node_count(self) -> int:
        """Get number of registered nodes."""
        return len(self._nodes)

    @property
    def connection_count(self) -> int:
        """Get number of connections."""
        return len(self._connections)

"""Material graph - Node-based material graph with connections and validation."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
import json
import uuid

from .material_nodes import (
    MaterialNode, NodeCategory, DataType, NODE_REGISTRY,
    PBROutputNode, UnlitOutputNode
)
from .connection_validator import (
    ConnectionValidator, Connection, ValidationResult, ValidationError
)
from .node_factory import NodeFactory, get_default_factory


class GraphState(Enum):
    """State of the material graph."""
    CLEAN = auto()
    MODIFIED = auto()
    COMPILING = auto()
    ERROR = auto()


@dataclass
class GraphError:
    """Error in the material graph."""
    node_id: Optional[str]
    pin_name: Optional[str]
    message: str
    severity: str = "error"  # error, warning, info


@dataclass
class GraphMetadata:
    """Metadata for a material graph."""
    name: str = "Untitled"
    description: str = ""
    author: str = ""
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    created_time: str = ""
    modified_time: str = ""


class MaterialGraph:
    """
    Node-based material graph with connections and validation.

    The graph manages nodes and their connections, validates connections
    for type safety and cycles, and supports serialization.
    """

    def __init__(self, name: str = "Untitled"):
        self._id = str(uuid.uuid4())
        self._metadata = GraphMetadata(name=name)
        self._nodes: Dict[str, MaterialNode] = {}
        self._validator = ConnectionValidator()
        self._state = GraphState.CLEAN
        self._errors: List[GraphError] = []
        self._output_node_id: Optional[str] = None
        self._factory = get_default_factory()

        # Callbacks
        self._on_node_added: List[Callable[[MaterialNode], None]] = []
        self._on_node_removed: List[Callable[[str], None]] = []
        self._on_connection_added: List[Callable[[Connection], None]] = []
        self._on_connection_removed: List[Callable[[Connection], None]] = []
        self._on_state_changed: List[Callable[[GraphState], None]] = []

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._metadata.name

    @name.setter
    def name(self, value: str) -> None:
        self._metadata.name = value
        self._mark_modified()

    @property
    def metadata(self) -> GraphMetadata:
        return self._metadata

    @property
    def state(self) -> GraphState:
        return self._state

    @property
    def errors(self) -> List[GraphError]:
        return self._errors.copy()

    @property
    def nodes(self) -> Dict[str, MaterialNode]:
        return self._nodes.copy()

    @property
    def connections(self) -> Set[Connection]:
        return self._validator.connections

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def connection_count(self) -> int:
        return self._validator.connection_count

    @property
    def output_node(self) -> Optional[MaterialNode]:
        if self._output_node_id:
            return self._nodes.get(self._output_node_id)
        return None

    def _mark_modified(self) -> None:
        """Mark the graph as modified."""
        if self._state != GraphState.COMPILING:
            self._set_state(GraphState.MODIFIED)

    def _set_state(self, state: GraphState) -> None:
        """Set graph state and notify callbacks."""
        if self._state != state:
            self._state = state
            for callback in self._on_state_changed:
                callback(state)

    # ========================================================================
    # Node Management
    # ========================================================================

    def add_node(self, node: MaterialNode) -> bool:
        """
        Add a node to the graph.

        Args:
            node: Node to add

        Returns:
            True if node was added successfully
        """
        if node.id in self._nodes:
            return False

        self._nodes[node.id] = node
        self._validator.register_node(node)

        # Track output node
        if isinstance(node, (PBROutputNode, UnlitOutputNode)):
            if self._output_node_id is None:
                self._output_node_id = node.id

        self._mark_modified()

        for callback in self._on_node_added:
            callback(node)

        return True

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the graph.

        Args:
            node_id: ID of node to remove

        Returns:
            True if node was removed successfully
        """
        if node_id not in self._nodes:
            return False

        # Remove all connections to/from this node
        connections_to_remove = []
        for conn in self._validator.connections:
            if conn.source_node_id == node_id or conn.target_node_id == node_id:
                connections_to_remove.append(conn)

        for conn in connections_to_remove:
            self._validator.remove_connection(conn)
            for callback in self._on_connection_removed:
                callback(conn)

        # Remove the node
        del self._nodes[node_id]
        self._validator.unregister_node(node_id)

        if self._output_node_id == node_id:
            self._output_node_id = None
            # Find another output node
            for nid, node in self._nodes.items():
                if isinstance(node, (PBROutputNode, UnlitOutputNode)):
                    self._output_node_id = nid
                    break

        self._mark_modified()

        for callback in self._on_node_removed:
            callback(node_id)

        return True

    def get_node(self, node_id: str) -> Optional[MaterialNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_nodes_by_category(self, category: NodeCategory) -> List[MaterialNode]:
        """Get all nodes of a specific category."""
        return [node for node in self._nodes.values() if node.category == category]

    def get_nodes_by_type(self, node_type: type) -> List[MaterialNode]:
        """Get all nodes of a specific type."""
        return [node for node in self._nodes.values() if isinstance(node, node_type)]

    def create_node(self, node_type: str, name: str = "", **kwargs) -> Optional[MaterialNode]:
        """
        Create and add a node to the graph.

        Args:
            node_type: Type of node to create
            name: Optional name for the node
            **kwargs: Additional arguments for node constructor

        Returns:
            Created node or None if creation failed
        """
        node = self._factory.create_node(node_type, name, **kwargs)
        if node:
            self.add_node(node)
        return node

    def duplicate_node(self, node_id: str) -> Optional[MaterialNode]:
        """
        Duplicate a node.

        Args:
            node_id: ID of node to duplicate

        Returns:
            Duplicated node or None if original not found
        """
        original = self._nodes.get(node_id)
        if not original:
            return None

        # Serialize and deserialize to create copy
        node_data = original.to_dict()
        node_data["id"] = str(uuid.uuid4())  # New ID
        node_data["position"][0] += 50  # Offset position
        node_data["position"][1] += 50

        # Create new node of same type
        node_type = node_data["type"]
        new_node = self._factory.create_node(node_type)
        if new_node:
            new_node._id = node_data["id"]
            new_node.position = tuple(node_data["position"])
            self.add_node(new_node)

        return new_node

    # ========================================================================
    # Connection Management
    # ========================================================================

    def connect(
        self,
        source_node_id: str,
        source_pin: str,
        target_node_id: str,
        target_pin: str
    ) -> ValidationResult:
        """
        Create a connection between two nodes.

        Args:
            source_node_id: ID of source node
            source_pin: Name of output pin on source
            target_node_id: ID of target node
            target_pin: Name of input pin on target

        Returns:
            ValidationResult indicating success or failure
        """
        result = self._validator.validate_connection(
            source_node_id, source_pin, target_node_id, target_pin
        )

        if result.valid:
            connection = Connection(
                source_node_id=source_node_id,
                source_pin=source_pin,
                target_node_id=target_node_id,
                target_pin=target_pin
            )
            self._validator.add_connection(connection)
            self._mark_modified()

            for callback in self._on_connection_added:
                callback(connection)

        return result

    def disconnect(
        self,
        source_node_id: str,
        source_pin: str,
        target_node_id: str,
        target_pin: str
    ) -> bool:
        """
        Remove a connection between two nodes.

        Args:
            source_node_id: ID of source node
            source_pin: Name of output pin on source
            target_node_id: ID of target node
            target_pin: Name of input pin on target

        Returns:
            True if connection was removed
        """
        connection = Connection(
            source_node_id=source_node_id,
            source_pin=source_pin,
            target_node_id=target_node_id,
            target_pin=target_pin
        )

        if connection in self._validator.connections:
            self._validator.remove_connection(connection)
            self._mark_modified()

            for callback in self._on_connection_removed:
                callback(connection)

            return True

        return False

    def disconnect_pin(self, node_id: str, pin_name: str, is_output: bool = True) -> int:
        """
        Disconnect all connections from/to a specific pin.

        Args:
            node_id: ID of the node
            pin_name: Name of the pin
            is_output: True if output pin, False if input pin

        Returns:
            Number of connections removed
        """
        connections_to_remove = []

        for conn in self._validator.connections:
            if is_output:
                if conn.source_node_id == node_id and conn.source_pin == pin_name:
                    connections_to_remove.append(conn)
            else:
                if conn.target_node_id == node_id and conn.target_pin == pin_name:
                    connections_to_remove.append(conn)

        for conn in connections_to_remove:
            self._validator.remove_connection(conn)
            for callback in self._on_connection_removed:
                callback(conn)

        if connections_to_remove:
            self._mark_modified()

        return len(connections_to_remove)

    def disconnect_all(self, node_id: str) -> int:
        """
        Disconnect all connections from/to a node.

        Args:
            node_id: ID of the node

        Returns:
            Number of connections removed
        """
        connections_to_remove = []

        for conn in self._validator.connections:
            if conn.source_node_id == node_id or conn.target_node_id == node_id:
                connections_to_remove.append(conn)

        for conn in connections_to_remove:
            self._validator.remove_connection(conn)
            for callback in self._on_connection_removed:
                callback(conn)

        if connections_to_remove:
            self._mark_modified()

        return len(connections_to_remove)

    def get_connections_to_node(self, node_id: str) -> List[Connection]:
        """Get all connections to a node."""
        return self._validator.get_connections_to_node(node_id)

    def get_connections_from_node(self, node_id: str) -> List[Connection]:
        """Get all connections from a node."""
        return self._validator.get_connections_from_node(node_id)

    def get_connection_to_pin(self, node_id: str, pin_name: str) -> Optional[Connection]:
        """Get connection to a specific input pin."""
        return self._validator.get_connection_to_pin(node_id, pin_name)

    def can_connect(
        self,
        source_node_id: str,
        source_pin: str,
        target_node_id: str,
        target_pin: str
    ) -> ValidationResult:
        """Check if a connection would be valid."""
        return self._validator.validate_connection(
            source_node_id, source_pin, target_node_id, target_pin
        )

    # ========================================================================
    # Graph Analysis
    # ========================================================================

    def validate(self) -> List[GraphError]:
        """
        Validate the entire graph.

        Returns:
            List of errors/warnings found
        """
        self._errors.clear()

        # Check for output node
        if not self._output_node_id:
            self._errors.append(GraphError(
                node_id=None,
                pin_name=None,
                message="Graph has no output node",
                severity="error"
            ))

        # Check for disconnected nodes
        disconnected = self._validator.find_disconnected_nodes()
        for node_id in disconnected:
            node = self._nodes.get(node_id)
            if node and not isinstance(node, (PBROutputNode, UnlitOutputNode)):
                self._errors.append(GraphError(
                    node_id=node_id,
                    pin_name=None,
                    message=f"Node '{node.name}' is not connected to anything",
                    severity="warning"
                ))

        # Check for unconnected required inputs
        for node_id, node in self._nodes.items():
            unconnected = self._validator.find_unconnected_inputs(node_id)
            for pin_name in unconnected:
                pin = node.get_input(pin_name)
                if pin and pin.default_value is None:
                    self._errors.append(GraphError(
                        node_id=node_id,
                        pin_name=pin_name,
                        message=f"Required input '{pin_name}' on '{node.name}' is not connected",
                        severity="warning"
                    ))

        if self._errors:
            self._set_state(GraphState.ERROR)
        else:
            self._set_state(GraphState.CLEAN if self._state == GraphState.ERROR else self._state)

        return self._errors.copy()

    def get_topological_order(self) -> List[str]:
        """Get nodes in topological order for compilation."""
        return self._validator.get_topological_order()

    def get_evaluation_order(self) -> List[MaterialNode]:
        """Get nodes in order for evaluation."""
        order = self.get_topological_order()
        return [self._nodes[node_id] for node_id in order if node_id in self._nodes]

    # ========================================================================
    # Serialization
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "id": self._id,
            "metadata": {
                "name": self._metadata.name,
                "description": self._metadata.description,
                "author": self._metadata.author,
                "version": self._metadata.version,
                "tags": self._metadata.tags,
            },
            "nodes": {node_id: node.to_dict() for node_id, node in self._nodes.items()},
            "connections": [
                {
                    "source_node_id": conn.source_node_id,
                    "source_pin": conn.source_pin,
                    "target_node_id": conn.target_node_id,
                    "target_pin": conn.target_pin
                }
                for conn in self._validator.connections
            ],
            "output_node_id": self._output_node_id
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize graph to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MaterialGraph':
        """Deserialize graph from dictionary."""
        graph = cls()
        graph._id = data.get("id", str(uuid.uuid4()))

        # Load metadata
        metadata = data.get("metadata", {})
        graph._metadata.name = metadata.get("name", "Untitled")
        graph._metadata.description = metadata.get("description", "")
        graph._metadata.author = metadata.get("author", "")
        graph._metadata.version = metadata.get("version", "1.0")
        graph._metadata.tags = metadata.get("tags", [])

        # Load nodes
        nodes_data = data.get("nodes", {})
        for node_id, node_data in nodes_data.items():
            node_type = node_data.get("type")
            if node_type and node_type in NODE_REGISTRY:
                node_class = NODE_REGISTRY[node_type]
                try:
                    node = node_class()
                    node._id = node_id
                    node._name = node_data.get("name", node_type)
                    node.position = tuple(node_data.get("position", [0, 0]))
                    node.preview_enabled = node_data.get("preview_enabled", True)
                    node.collapsed = node_data.get("collapsed", False)
                    node.comment = node_data.get("comment", "")
                    if node_data.get("color"):
                        node.color = tuple(node_data["color"])

                    # Apply input default values
                    inputs = node_data.get("inputs", {})
                    for pin_name, value in inputs.items():
                        pin = node.get_input(pin_name)
                        if pin:
                            pin.default_value = value

                    graph.add_node(node)
                except Exception as e:
                    print(f"Error loading node '{node_id}': {e}")

        # Load connections
        connections_data = data.get("connections", [])
        for conn_data in connections_data:
            graph.connect(
                conn_data["source_node_id"],
                conn_data["source_pin"],
                conn_data["target_node_id"],
                conn_data["target_pin"]
            )

        # Set output node
        graph._output_node_id = data.get("output_node_id")

        graph._set_state(GraphState.CLEAN)
        return graph

    @classmethod
    def from_json(cls, json_str: str) -> 'MaterialGraph':
        """Deserialize graph from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    # ========================================================================
    # Callbacks
    # ========================================================================

    def on_node_added(self, callback: Callable[[MaterialNode], None]) -> None:
        """Register callback for node added events."""
        self._on_node_added.append(callback)

    def on_node_removed(self, callback: Callable[[str], None]) -> None:
        """Register callback for node removed events."""
        self._on_node_removed.append(callback)

    def on_connection_added(self, callback: Callable[[Connection], None]) -> None:
        """Register callback for connection added events."""
        self._on_connection_added.append(callback)

    def on_connection_removed(self, callback: Callable[[Connection], None]) -> None:
        """Register callback for connection removed events."""
        self._on_connection_removed.append(callback)

    def on_state_changed(self, callback: Callable[[GraphState], None]) -> None:
        """Register callback for state change events."""
        self._on_state_changed.append(callback)

    # ========================================================================
    # Utilities
    # ========================================================================

    def clear(self) -> None:
        """Clear all nodes and connections."""
        self._nodes.clear()
        self._validator.clear()
        self._output_node_id = None
        self._errors.clear()
        self._set_state(GraphState.CLEAN)

    def copy(self) -> 'MaterialGraph':
        """Create a deep copy of this graph."""
        return MaterialGraph.from_dict(self.to_dict())

"""Node Graph Schema for FlowForge.

This module defines JSON-serializable types for converting AST definitions
to node graph format that LiteGraph can render in the frontend.

The types here correspond to the TypeScript interfaces defined in:
    apps/desktop/src/services/api.ts

Example:
    from flowforge_backend.ast_parser.graph_types import (
        NodeGraph, GraphNode, GraphEdge, NodePosition, SourceLocation
    )

    # Create a node for a component
    node = GraphNode(
        id="node_1",
        type="component",
        name="Player",
        position=NodePosition(x=100, y=200),
        data=ComponentData(fields=[
            FieldData(name="health", type_annotation="int", default_value="100")
        ]),
        source=SourceLocation(file="game.py", line=10)
    )

    # Serialize to JSON-compatible dict
    node_dict = node.to_dict()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, Union
from typing_extensions import Self


# =============================================================================
# ENUMS
# =============================================================================


class NodeType(str, Enum):
    """Types of nodes in the graph, corresponding to Trinity ECS patterns."""
    COMPONENT = "component"
    SYSTEM = "system"
    RESOURCE = "resource"
    EVENT = "event"


class EdgeType(str, Enum):
    """Types of edges connecting nodes in the graph."""
    REFERENCE = "reference"      # Field type references another class
    INHERITANCE = "inheritance"  # Class inheritance relationship
    QUERY = "query"             # System queries a component type


# =============================================================================
# POSITION AND LOCATION
# =============================================================================


@dataclass
class NodePosition:
    """Position of a node on the canvas.

    Corresponds to the [x, y] tuple in the frontend GraphNode.position.

    Attributes:
        x: Horizontal position in pixels.
        y: Vertical position in pixels.
    """
    x: float = 0.0
    y: float = 0.0

    def to_dict(self) -> list[float]:
        """Convert to JSON-serializable format (tuple as list)."""
        return [self.x, self.y]

    def to_tuple(self) -> tuple[float, float]:
        """Convert to tuple format."""
        return (self.x, self.y)

    @classmethod
    def from_dict(cls, data: Union[list[float], tuple[float, float], dict[str, float]]) -> Self:
        """Create from dict, list, or tuple.

        Args:
            data: Position data in one of these formats:
                  - [x, y] list
                  - (x, y) tuple
                  - {"x": x, "y": y} dict

        Returns:
            New NodePosition instance.
        """
        if isinstance(data, (list, tuple)):
            return cls(x=float(data[0]), y=float(data[1]))
        return cls(x=float(data.get("x", 0.0)), y=float(data.get("y", 0.0)))

    @classmethod
    def from_tuple(cls, pos: tuple[float, float]) -> Self:
        """Create from a tuple (backward compatibility)."""
        return cls(x=pos[0], y=pos[1])


@dataclass
class SourceLocation:
    """Location in source code where a node's definition originated.

    Used to enable "jump to source" functionality in the editor.

    Attributes:
        file: Path to the source file (relative or absolute).
        line: Line number where the definition starts (1-indexed).
        end_line: Optional end line for the definition.
        column: Optional column offset.
    """
    file: str
    line: int
    end_line: Optional[int] = None
    column: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "file": self.file,
            "line": self.line,
        }
        if self.end_line is not None:
            result["end_line"] = self.end_line
        if self.column is not None:
            result["column"] = self.column
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create from dictionary.

        Args:
            data: Dictionary with file and line keys.

        Returns:
            New SourceLocation instance.
        """
        return cls(
            file=data.get("file", ""),
            line=int(data.get("line", 0)),
            end_line=data.get("end_line"),
            column=data.get("column"),
        )


# =============================================================================
# NODE DATA TYPES
# =============================================================================


@dataclass
class FieldData:
    """Data for a single field in a component/resource/event.

    Represents an annotated class attribute like:
        name: str = "default"

    Attributes:
        name: The field name.
        type_annotation: Type annotation as a string.
        default_value: Optional default value as a string.
        line_number: Line where the field is defined.
        is_optional: Whether the field is optional (has default or Optional type).
    """
    name: str
    type_annotation: str
    default_value: Optional[str] = None
    line_number: Optional[int] = None
    is_optional: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "name": self.name,
            "type_annotation": self.type_annotation,
        }
        if self.default_value is not None:
            result["default_value"] = self.default_value
        if self.line_number is not None:
            result["line_number"] = self.line_number
        if self.is_optional:
            result["is_optional"] = True
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            type_annotation=data.get("type_annotation", "Any"),
            default_value=data.get("default_value"),
            line_number=data.get("line_number"),
            is_optional=data.get("is_optional", False),
        )


@dataclass
class ParameterData:
    """Data for a method parameter.

    Attributes:
        name: Parameter name.
        type_annotation: Type annotation as a string.
        default_value: Optional default value as a string.
    """
    name: str
    type_annotation: Optional[str] = None
    default_value: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {"name": self.name}
        if self.type_annotation is not None:
            result["type_annotation"] = self.type_annotation
        if self.default_value is not None:
            result["default_value"] = self.default_value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            type_annotation=data.get("type_annotation"),
            default_value=data.get("default_value"),
        )


@dataclass
class MethodData:
    """Data for a method in a system node.

    Attributes:
        name: Method name.
        parameters: List of parameter definitions.
        return_type: Return type annotation as a string.
        docstring: Optional docstring.
        line_number: Line where the method is defined.
        query_components: Component types from Query[...] annotations.
        decorators: List of decorator names.
    """
    name: str
    parameters: list[ParameterData] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    line_number: Optional[int] = None
    query_components: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "name": self.name,
            "parameters": [p.to_dict() for p in self.parameters],
        }
        if self.return_type is not None:
            result["return_type"] = self.return_type
        if self.docstring is not None:
            result["docstring"] = self.docstring
        if self.line_number is not None:
            result["line_number"] = self.line_number
        if self.query_components:
            result["query_components"] = self.query_components
        if self.decorators:
            result["decorators"] = self.decorators
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create from dictionary.

        Note: Accepts either 'query_components' or 'query_types' for
        backwards compatibility.
        """
        # Accept both query_types (from frontend) and query_components
        query_comps = data.get("query_types") or data.get("query_components", [])
        return cls(
            name=data["name"],
            parameters=[ParameterData.from_dict(p) for p in data.get("parameters", [])],
            return_type=data.get("return_type"),
            docstring=data.get("docstring"),
            line_number=data.get("line_number"),
            query_components=query_comps,
            decorators=data.get("decorators", []),
        )


@dataclass
class ComponentData:
    """Node data specific to @component decorated classes.

    Attributes:
        fields: List of field definitions.
        bases: Base class names.
        docstring: Class docstring.
        decorator_args: Arguments passed to @component decorator.
    """
    fields: list[FieldData] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    docstring: Optional[str] = None
    decorator_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "fields": [f.to_dict() for f in self.fields],
        }
        if self.bases:
            result["bases"] = self.bases
        if self.docstring is not None:
            result["docstring"] = self.docstring
        if self.decorator_args:
            result["decorator_args"] = self.decorator_args
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create from dictionary."""
        return cls(
            fields=[FieldData.from_dict(f) for f in data.get("fields", [])],
            bases=data.get("bases", []),
            docstring=data.get("docstring"),
            decorator_args=data.get("decorator_args", {}),
        )


@dataclass
class SystemData:
    """Node data specific to @system decorated classes.

    Attributes:
        methods: List of method definitions.
        queries: Component types this system queries.
        fields: Any fields the system has.
        bases: Base class names.
        docstring: Class docstring.
        decorator_args: Arguments passed to @system decorator.
    """
    methods: list[MethodData] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    fields: list[FieldData] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    docstring: Optional[str] = None
    decorator_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "methods": [m.to_dict() for m in self.methods],
        }
        if self.queries:
            result["queries"] = self.queries
        if self.fields:
            result["fields"] = [f.to_dict() for f in self.fields]
        if self.bases:
            result["bases"] = self.bases
        if self.docstring is not None:
            result["docstring"] = self.docstring
        if self.decorator_args:
            result["decorator_args"] = self.decorator_args
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create from dictionary."""
        return cls(
            methods=[MethodData.from_dict(m) for m in data.get("methods", [])],
            queries=data.get("queries", []),
            fields=[FieldData.from_dict(f) for f in data.get("fields", [])],
            bases=data.get("bases", []),
            docstring=data.get("docstring"),
            decorator_args=data.get("decorator_args", {}),
        )


@dataclass
class ResourceData:
    """Node data specific to @resource decorated classes.

    Attributes:
        fields: List of field definitions.
        bases: Base class names.
        docstring: Class docstring.
        is_singleton: Whether this resource is a singleton.
        decorator_args: Arguments passed to @resource decorator.
    """
    fields: list[FieldData] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    docstring: Optional[str] = None
    is_singleton: bool = True
    decorator_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "fields": [f.to_dict() for f in self.fields],
            "is_singleton": self.is_singleton,
        }
        if self.bases:
            result["bases"] = self.bases
        if self.docstring is not None:
            result["docstring"] = self.docstring
        if self.decorator_args:
            result["decorator_args"] = self.decorator_args
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create from dictionary."""
        return cls(
            fields=[FieldData.from_dict(f) for f in data.get("fields", [])],
            bases=data.get("bases", []),
            docstring=data.get("docstring"),
            is_singleton=data.get("is_singleton", True),
            decorator_args=data.get("decorator_args", {}),
        )


@dataclass
class EventData:
    """Node data specific to @event decorated classes.

    Attributes:
        fields: Payload fields for the event.
        bases: Base class names.
        docstring: Class docstring.
        decorator_args: Arguments passed to @event decorator.
    """
    fields: list[FieldData] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    docstring: Optional[str] = None
    decorator_args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "fields": [f.to_dict() for f in self.fields],
        }
        if self.bases:
            result["bases"] = self.bases
        if self.docstring is not None:
            result["docstring"] = self.docstring
        if self.decorator_args:
            result["decorator_args"] = self.decorator_args
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create from dictionary.

        Note: Accepts either 'fields' or 'payload_fields' for backwards
        compatibility. The frontend may send 'payload_fields' for events.
        """
        # Events may have payload_fields instead of fields (from frontend)
        fields_data = data.get("payload_fields") or data.get("fields", [])
        return cls(
            fields=[FieldData.from_dict(f) for f in fields_data],
            bases=data.get("bases", []),
            docstring=data.get("docstring"),
            decorator_args=data.get("decorator_args", {}),
        )


# Type alias for all node data types
NodeData = Union[ComponentData, SystemData, ResourceData, EventData]


def _parse_node_data(node_type: str, raw_data: dict[str, Any]) -> NodeData:
    """Parse node data based on node type.

    Args:
        node_type: The type of node (component, system, resource, event).
        raw_data: Raw dictionary data to parse.

    Returns:
        Appropriate NodeData subclass instance.
    """
    if node_type == "component":
        return ComponentData.from_dict(raw_data)
    elif node_type == "system":
        return SystemData.from_dict(raw_data)
    elif node_type == "resource":
        return ResourceData.from_dict(raw_data)
    elif node_type == "event":
        return EventData.from_dict(raw_data)
    else:
        # Default to component data for unknown types
        return ComponentData.from_dict(raw_data)


# =============================================================================
# GRAPH NODES
# =============================================================================


@dataclass
class GraphNode:
    """A node in the visual graph representation of Python code.

    Corresponds to the GraphNode interface in api.ts:
        interface GraphNode {
            id: string;
            type: 'component' | 'system' | 'resource' | 'event';
            name: string;
            position: [number, number];
            data: Record<string, unknown>;
            source: { file: string; line: number };
        }

    Attributes:
        id: Unique identifier for the node.
        type: Node type (component, system, resource, event).
        name: Class name from the source code.
        position: Position on the canvas.
        data: Type-specific node data.
        source: Source code location.
    """
    id: str
    type: Literal["component", "system", "resource", "event"]
    name: str
    position: NodePosition = field(default_factory=NodePosition)
    data: Union[dict[str, Any], NodeData] = field(default_factory=dict)
    source: Optional[SourceLocation] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict matching frontend format."""
        # Handle data being either a dict or a NodeData object
        if isinstance(self.data, dict):
            data_dict = self.data
        else:
            data_dict = self.data.to_dict()

        result: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "position": self.position.to_dict(),
            "data": data_dict,
        }
        if self.source is not None:
            result["source"] = self.source.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create from dictionary.

        Args:
            data: Dictionary with node data.

        Returns:
            New GraphNode instance.
        """
        node_type = data["type"]
        raw_data = data.get("data", {})
        parsed_data = _parse_node_data(node_type, raw_data)

        # Handle optional source
        source = None
        if "source" in data and data["source"]:
            source = SourceLocation.from_dict(data["source"])

        return cls(
            id=data["id"],
            type=node_type,
            name=data["name"],
            position=NodePosition.from_dict(data.get("position", [0, 0])),
            data=parsed_data,
            source=source,
        )


# =============================================================================
# GRAPH EDGES
# =============================================================================


@dataclass
class GraphEdge:
    """An edge connecting nodes in the graph.

    Corresponds to the GraphEdge interface in api.ts:
        interface GraphEdge {
            id: string;
            source: string;
            target: string;
            type: 'reference' | 'inheritance' | 'query';
        }

    Attributes:
        id: Unique identifier for the edge.
        source: ID of the source node.
        target: ID of the target node.
        source_slot: Output slot index on source node (for LiteGraph compatibility).
        target_slot: Input slot index on target node (for LiteGraph compatibility).
        type: Edge type (reference, inheritance, query).
        label: Optional label to display on the edge.
    """
    id: str
    source: str
    target: str
    source_slot: int = 0
    target_slot: int = 0
    type: Literal["reference", "inheritance", "query"] = "reference"
    label: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict matching frontend format."""
        result: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type,
        }
        # Only include slots if non-zero (for LiteGraph extended format)
        if self.source_slot != 0:
            result["source_slot"] = self.source_slot
        if self.target_slot != 0:
            result["target_slot"] = self.target_slot
        if self.label is not None:
            result["label"] = self.label
        if self.data:
            result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create from dictionary.

        Args:
            data: Dictionary with edge data.

        Returns:
            New GraphEdge instance.
        """
        return cls(
            id=data["id"],
            source=data["source"],
            target=data["target"],
            source_slot=data.get("source_slot", 0),
            target_slot=data.get("target_slot", 0),
            type=data.get("type", "reference"),
            label=data.get("label"),
        )


# =============================================================================
# NODE GRAPH
# =============================================================================


@dataclass
class NodeGraph:
    """Complete node graph representing parsed Python code.

    Corresponds to the NodeGraph interface in api.ts:
        interface NodeGraph {
            nodes: GraphNode[];
            edges: GraphEdge[];
        }

    This is the primary data structure exchanged between the Python
    backend and the TypeScript frontend.

    Attributes:
        nodes: List of graph nodes.
        edges: List of graph edges.
        metadata: Optional metadata about the graph.
    """
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict matching frontend format."""
        result: dict[str, Any] = {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create from dictionary.

        Args:
            data: Dictionary with graph data.

        Returns:
            New NodeGraph instance.
        """
        return cls(
            nodes=[GraphNode.from_dict(n) for n in data.get("nodes", [])],
            edges=[GraphEdge.from_dict(e) for e in data.get("edges", [])],
            metadata=data.get("metadata", {}),
        )

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID.

        Args:
            node_id: The node ID to look up.

        Returns:
            The node if found, None otherwise.
        """
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    # Backward compatibility alias
    def get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID (alias for get_node)."""
        return self.get_node(node_id)

    def get_node_by_name(self, name: str) -> Optional[GraphNode]:
        """Get a node by name.

        Args:
            name: The class name to look up.

        Returns:
            The node if found, None otherwise.
        """
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_nodes_by_type(self, node_type: str) -> list[GraphNode]:
        """Get all nodes of a specific type.

        Args:
            node_type: The node type to filter by.

        Returns:
            List of matching nodes.
        """
        return [n for n in self.nodes if n.type == node_type]

    def get_edges_from(self, node_id: str) -> list[GraphEdge]:
        """Get all edges originating from a node.

        Args:
            node_id: The source node ID.

        Returns:
            List of edges from the node.
        """
        return [e for e in self.edges if e.source == node_id]

    def get_edges_to(self, node_id: str) -> list[GraphEdge]:
        """Get all edges targeting a node.

        Args:
            node_id: The target node ID.

        Returns:
            List of edges to the node.
        """
        return [e for e in self.edges if e.target == node_id]

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph.

        Args:
            node: The node to add.
        """
        self.nodes.append(node)

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph.

        Args:
            edge: The edge to add.
        """
        self.edges.append(edge)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its edges from the graph.

        Args:
            node_id: The ID of the node to remove.

        Returns:
            True if the node was found and removed.
        """
        initial_count = len(self.nodes)
        self.nodes = [n for n in self.nodes if n.id != node_id]
        self.edges = [e for e in self.edges if e.source != node_id and e.target != node_id]
        return len(self.nodes) < initial_count

    def remove_edge(self, edge_id: str) -> bool:
        """Remove an edge from the graph.

        Args:
            edge_id: The ID of the edge to remove.

        Returns:
            True if the edge was found and removed.
        """
        initial_count = len(self.edges)
        self.edges = [e for e in self.edges if e.id != edge_id]
        return len(self.edges) < initial_count


# =============================================================================
# CONVENIENCE CONSTRUCTORS
# =============================================================================


def create_component_node(
    id: str,
    name: str,
    fields: list[FieldData],
    source: SourceLocation,
    position: Optional[NodePosition] = None,
    bases: Optional[list[str]] = None,
    docstring: Optional[str] = None,
) -> GraphNode:
    """Create a component node with the given data.

    Args:
        id: Unique node identifier.
        name: Component class name.
        fields: List of component fields.
        source: Source code location.
        position: Canvas position (defaults to origin).
        bases: Base class names.
        docstring: Class docstring.

    Returns:
        New GraphNode with component type.
    """
    return GraphNode(
        id=id,
        type="component",
        name=name,
        position=position or NodePosition(),
        data=ComponentData(
            fields=fields,
            bases=bases or [],
            docstring=docstring,
        ),
        source=source,
    )


def create_system_node(
    id: str,
    name: str,
    methods: list[MethodData],
    source: SourceLocation,
    position: Optional[NodePosition] = None,
    queries: Optional[list[str]] = None,
    bases: Optional[list[str]] = None,
    docstring: Optional[str] = None,
) -> GraphNode:
    """Create a system node with the given data.

    Args:
        id: Unique node identifier.
        name: System class name.
        methods: List of system methods.
        source: Source code location.
        position: Canvas position (defaults to origin).
        queries: Component types this system queries.
        bases: Base class names.
        docstring: Class docstring.

    Returns:
        New GraphNode with system type.
    """
    return GraphNode(
        id=id,
        type="system",
        name=name,
        position=position or NodePosition(),
        data=SystemData(
            methods=methods,
            queries=queries or [],
            bases=bases or [],
            docstring=docstring,
        ),
        source=source,
    )


def create_resource_node(
    id: str,
    name: str,
    fields: list[FieldData],
    source: SourceLocation,
    position: Optional[NodePosition] = None,
    is_singleton: bool = True,
    bases: Optional[list[str]] = None,
    docstring: Optional[str] = None,
) -> GraphNode:
    """Create a resource node with the given data.

    Args:
        id: Unique node identifier.
        name: Resource class name.
        fields: List of resource fields.
        source: Source code location.
        position: Canvas position (defaults to origin).
        is_singleton: Whether this is a singleton resource.
        bases: Base class names.
        docstring: Class docstring.

    Returns:
        New GraphNode with resource type.
    """
    return GraphNode(
        id=id,
        type="resource",
        name=name,
        position=position or NodePosition(),
        data=ResourceData(
            fields=fields,
            is_singleton=is_singleton,
            bases=bases or [],
            docstring=docstring,
        ),
        source=source,
    )


def create_event_node(
    id: str,
    name: str,
    fields: list[FieldData],
    source: SourceLocation,
    position: Optional[NodePosition] = None,
    bases: Optional[list[str]] = None,
    docstring: Optional[str] = None,
) -> GraphNode:
    """Create an event node with the given data.

    Args:
        id: Unique node identifier.
        name: Event class name.
        fields: Payload fields.
        source: Source code location.
        position: Canvas position (defaults to origin).
        bases: Base class names.
        docstring: Class docstring.

    Returns:
        New GraphNode with event type.
    """
    return GraphNode(
        id=id,
        type="event",
        name=name,
        position=position or NodePosition(),
        data=EventData(
            fields=fields,
            bases=bases or [],
            docstring=docstring,
        ),
        source=source,
    )


def create_edge(
    id: str,
    source: str,
    target: str,
    edge_type: Literal["reference", "inheritance", "query"] = "reference",
    source_slot: int = 0,
    target_slot: int = 0,
    label: Optional[str] = None,
) -> GraphEdge:
    """Create an edge between two nodes.

    Args:
        id: Unique edge identifier.
        source: Source node ID.
        target: Target node ID.
        edge_type: Type of edge.
        source_slot: Output slot index.
        target_slot: Input slot index.
        label: Optional edge label.

    Returns:
        New GraphEdge.
    """
    return GraphEdge(
        id=id,
        source=source,
        target=target,
        source_slot=source_slot,
        target_slot=target_slot,
        type=edge_type,
        label=label,
    )

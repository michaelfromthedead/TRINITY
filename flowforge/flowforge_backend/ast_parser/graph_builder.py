"""Node Graph Builder for FlowForge.

This module converts parsed Trinity AST definitions into node graph format
suitable for visualization in the frontend.

The GraphBuilder class takes parsed Trinity definitions (ComponentDef, SystemDef,
ResourceDef, EventDef) and converts them to GraphNode objects with appropriate
data and metadata.

Example:
    from flowforge_backend.ast_parser import TrinityASTVisitor, parse_source
    from flowforge_backend.ast_parser.graph_builder import GraphBuilder

    # Parse Python source
    result = parse_source('''
        from trinity import component, system

        @component
        class Position:
            x: float = 0.0
            y: float = 0.0

        @system
        class MovementSystem:
            def update(self, entities: Query[Position]) -> None:
                pass
    ''')

    # Build node graph
    builder = GraphBuilder()
    for comp in result.components:
        builder.add_component(comp)
    for sys in result.systems:
        builder.add_system(sys)

    graph = builder.build()
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .constants import NODE_ID_HASH_LENGTH, NODE_ID_PREFIX
from .graph_types import (
    GraphEdge,
    GraphNode,
    NodeGraph,
    NodeType,
    SourceLocation,
)
from .types import (
    ComponentDef,
    EventDef,
    FieldDef,
    MethodDef,
    ResourceDef,
    SystemDef,
)


def _generate_node_id(class_name: str, source_file: Optional[str] = None) -> str:
    """Generate a unique node ID from class name and source file.

    Uses a hash of the class name and source file to create a deterministic
    but unique ID. Falls back to UUID if no source file is provided.

    Args:
        class_name: The name of the class.
        source_file: Optional path to the source file.

    Returns:
        A unique string ID for the node.
    """
    if source_file:
        # Create deterministic ID from class name + file path
        content = f"{source_file}:{class_name}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()[:NODE_ID_HASH_LENGTH]
        return f"{NODE_ID_PREFIX}{hash_digest}"
    else:
        # Fallback to UUID if no source file
        return f"{NODE_ID_PREFIX}{uuid.uuid4().hex[:NODE_ID_HASH_LENGTH]}"


def _field_to_dict(field_def: FieldDef) -> dict[str, Any]:
    """Convert a FieldDef to a dictionary for node data.

    Args:
        field_def: The field definition to convert.

    Returns:
        Dictionary with field name, type, type_annotation, and optional default.
        Both 'type' and 'type_annotation' are included for compatibility:
        - 'type' is used by the frontend
        - 'type_annotation' is used by the EdgeBuilder
    """
    result: dict[str, Any] = {
        "name": field_def.name,
        "type": field_def.type_annotation,
        "type_annotation": field_def.type_annotation,  # For EdgeBuilder compatibility
    }
    if field_def.default_value is not None:
        result["default"] = field_def.default_value
    return result


def _method_to_dict(method_def: MethodDef) -> dict[str, Any]:
    """Convert a MethodDef to a dictionary for node data.

    Args:
        method_def: The method definition to convert.

    Returns:
        Dictionary with method name, parameters, return type, and query info.
    """
    result: dict[str, Any] = {
        "name": method_def.name,
        "parameters": [
            {
                "name": p.name,
                "type": p.type_annotation,
            }
            for p in method_def.parameters
            if p.type_annotation is not None
        ],
    }
    if method_def.return_type is not None:
        result["return_type"] = method_def.return_type
    if method_def.query_info is not None:
        result["query_types"] = list(method_def.query_info.component_types)
    return result


@dataclass
class GraphBuilder:
    """Builds a NodeGraph from Trinity AST definitions.

    This class accumulates nodes and edges from parsed Trinity definitions
    and produces a complete NodeGraph that can be serialized to JSON.

    Attributes:
        _nodes: Internal list of accumulated nodes.
        _edges: Internal list of accumulated edges.
        _node_id_map: Mapping from class name to node ID for edge building.

    Example:
        builder = GraphBuilder()
        builder.add_component(component_def)
        builder.add_system(system_def)
        graph = builder.build()
    """
    _nodes: list[GraphNode] = field(default_factory=list)
    _edges: list[GraphEdge] = field(default_factory=list)
    _node_id_map: dict[str, str] = field(default_factory=dict)

    @property
    def nodes(self) -> list[GraphNode]:
        """Get the list of accumulated nodes."""
        return self._nodes

    @property
    def node_id_map(self) -> dict[str, str]:
        """Get the mapping from class name to node ID."""
        return self._node_id_map

    def add_component(self, comp: ComponentDef) -> GraphNode:
        """Convert a ComponentDef to a GraphNode and add it to the graph.

        Components are data containers in the ECS pattern. They hold fields
        but typically no methods (other than generated dataclass methods).

        Args:
            comp: The parsed component definition.

        Returns:
            The created GraphNode.
        """
        node_id = _generate_node_id(comp.name, comp.source_file)
        self._node_id_map[comp.name] = node_id

        # Build source location
        source = None
        if comp.source_file:
            source = SourceLocation(file=comp.source_file, line=comp.line_number)

        # Build node data with fields
        data: dict[str, Any] = {
            "fields": [_field_to_dict(f) for f in comp.fields],
        }
        if comp.docstring:
            data["docstring"] = comp.docstring
        if comp.decorator_args.keyword or comp.decorator_args.positional:
            data["decorator_args"] = comp.decorator_args.to_dict()

        node = GraphNode(
            id=node_id,
            type=NodeType.COMPONENT,
            name=comp.name,
            data=data,
            source=source,
        )
        self._nodes.append(node)
        return node

    def add_system(self, sys: SystemDef) -> GraphNode:
        """Convert a SystemDef to a GraphNode and add it to the graph.

        Systems contain the logic that operates on entities with specific
        components. They typically have methods with Query[...] parameters.

        Args:
            sys: The parsed system definition.

        Returns:
            The created GraphNode.
        """
        node_id = _generate_node_id(sys.name, sys.source_file)
        self._node_id_map[sys.name] = node_id

        # Build source location
        source = None
        if sys.source_file:
            source = SourceLocation(file=sys.source_file, line=sys.line_number)

        # Filter methods to those with Query[...] parameters
        query_methods = [
            m for m in sys.methods
            if m.query_info is not None
        ]

        # Build node data
        data: dict[str, Any] = {
            "methods": [_method_to_dict(m) for m in sys.methods],
            "queries": list(sys.queries),  # Quick reference to query types
        }

        # Add query methods specifically for quick access
        if query_methods:
            data["query_methods"] = [_method_to_dict(m) for m in query_methods]

        if sys.fields:
            data["fields"] = [_field_to_dict(f) for f in sys.fields]
        if sys.docstring:
            data["docstring"] = sys.docstring
        if sys.decorator_args.keyword or sys.decorator_args.positional:
            data["decorator_args"] = sys.decorator_args.to_dict()

        node = GraphNode(
            id=node_id,
            type=NodeType.SYSTEM,
            name=sys.name,
            data=data,
            source=source,
        )
        self._nodes.append(node)
        return node

    def add_resource(self, res: ResourceDef) -> GraphNode:
        """Convert a ResourceDef to a GraphNode and add it to the graph.

        Resources are singleton shared data containers accessible globally.
        They are similar to components but not attached to entities.

        Args:
            res: The parsed resource definition.

        Returns:
            The created GraphNode.
        """
        node_id = _generate_node_id(res.name, res.source_file)
        self._node_id_map[res.name] = node_id

        # Build source location
        source = None
        if res.source_file:
            source = SourceLocation(file=res.source_file, line=res.line_number)

        # Build node data
        data: dict[str, Any] = {
            "fields": [_field_to_dict(f) for f in res.fields],
            "is_singleton": res.is_singleton,
        }
        if res.docstring:
            data["docstring"] = res.docstring
        if res.decorator_args.keyword or res.decorator_args.positional:
            data["decorator_args"] = res.decorator_args.to_dict()

        node = GraphNode(
            id=node_id,
            type=NodeType.RESOURCE,
            name=res.name,
            data=data,
            source=source,
        )
        self._nodes.append(node)
        return node

    def add_event(self, evt: EventDef) -> GraphNode:
        """Convert an EventDef to a GraphNode and add it to the graph.

        Events are signals/triggers with optional payload data.
        They enable decoupled communication between systems.

        Args:
            evt: The parsed event definition.

        Returns:
            The created GraphNode.
        """
        node_id = _generate_node_id(evt.name, evt.source_file)
        self._node_id_map[evt.name] = node_id

        # Build source location
        source = None
        if evt.source_file:
            source = SourceLocation(file=evt.source_file, line=evt.line_number)

        # Build node data
        data: dict[str, Any] = {
            "payload_fields": [_field_to_dict(f) for f in evt.payload_fields],
        }
        # Also include regular fields if different from payload
        if evt.fields and evt.fields != evt.payload_fields:
            data["fields"] = [_field_to_dict(f) for f in evt.fields]
        if evt.docstring:
            data["docstring"] = evt.docstring
        if evt.decorator_args.keyword or evt.decorator_args.positional:
            data["decorator_args"] = evt.decorator_args.to_dict()

        node = GraphNode(
            id=node_id,
            type=NodeType.EVENT,
            name=evt.name,
            data=data,
            source=source,
        )
        self._nodes.append(node)
        return node

    def _build_edges(self) -> None:
        """Build edges between nodes using the EdgeBuilder.

        Delegates to the EdgeBuilder class which handles:
        - Type reference edges (fields referencing other types)
        - Query dependency edges (systems querying components)
        - Inheritance edges (class inheritance relationships)
        - Event handler edges (systems handling events)

        Should be called before build() returns.
        """
        # Import here to avoid circular imports at module load time
        from .edge_builder import EdgeBuilder

        edge_builder = EdgeBuilder(self._nodes, self._node_id_map)
        self._edges = edge_builder.build_all_edges()

    def build(self) -> NodeGraph:
        """Build and return the complete NodeGraph.

        This method:
        1. Builds edges based on type references and queries
        2. Creates the NodeGraph with all accumulated nodes and edges
        3. Adds metadata about the graph

        Returns:
            The complete NodeGraph ready for serialization.
        """
        # Build edges between nodes
        self._build_edges()

        # Collect metadata
        source_files = set()
        for node in self._nodes:
            if node.source is not None:
                source_files.add(node.source.file)

        metadata: dict[str, Any] = {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "source_files": sorted(source_files),
            "type_counts": {
                "component": sum(1 for n in self._nodes if n.type == NodeType.COMPONENT),
                "system": sum(1 for n in self._nodes if n.type == NodeType.SYSTEM),
                "resource": sum(1 for n in self._nodes if n.type == NodeType.RESOURCE),
                "event": sum(1 for n in self._nodes if n.type == NodeType.EVENT),
            },
        }

        return NodeGraph(
            nodes=self._nodes,
            edges=self._edges,
            metadata=metadata,
        )

    def clear(self) -> None:
        """Clear all accumulated nodes and edges.

        Use this to reuse the builder for a new graph.
        """
        self._nodes.clear()
        self._edges.clear()
        self._node_id_map.clear()


def build_graph_from_parse_result(parse_result: Any) -> NodeGraph:
    """Convenience function to build a graph from a ParseResult.

    Args:
        parse_result: A ParseResult from the AST parser containing
            components, systems, resources, and events.

    Returns:
        A complete NodeGraph.

    Example:
        from flowforge_backend.ast_parser import parse_file
        from flowforge_backend.ast_parser.graph_builder import build_graph_from_parse_result

        result = parse_file("game.py")
        graph = build_graph_from_parse_result(result)
    """
    # Import here to avoid circular imports
    from .types import ParseResult

    if not isinstance(parse_result, ParseResult):
        raise TypeError(f"Expected ParseResult, got {type(parse_result).__name__}")

    builder = GraphBuilder()

    for comp in parse_result.components:
        builder.add_component(comp)

    for sys in parse_result.systems:
        builder.add_system(sys)

    for res in parse_result.resources:
        builder.add_resource(res)

    for evt in parse_result.events:
        builder.add_event(evt)

    return builder.build()

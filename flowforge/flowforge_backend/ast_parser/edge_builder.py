"""Edge building logic for FlowForge graph construction.

This module detects relationships between nodes and creates edges representing
those relationships. It handles:

- Type reference edges: When a field's type annotation references another class
- Query dependency edges: When a System queries components via Query[...]
- Inheritance edges: When a class inherits from another known node
- Event handler edges: When a system method handles an event type

Example:
    from flowforge_backend.ast_parser.edge_builder import EdgeBuilder
    from flowforge_backend.ast_parser.graph_types import GraphNode, NodeType

    # Given nodes from the graph builder
    nodes = [
        GraphNode(id="node_1", type=NodeType.COMPONENT, name="Player", ...),
        GraphNode(id="node_2", type=NodeType.COMPONENT, name="Position", ...),
        GraphNode(id="node_3", type=NodeType.SYSTEM, name="MovementSystem", ...),
    ]
    node_id_map = {"Player": "node_1", "Position": "node_2", "MovementSystem": "node_3"}

    builder = EdgeBuilder(nodes, node_id_map)
    edges = builder.build_all_edges()
"""

from __future__ import annotations

import re
import uuid
from typing import Optional

from .constants import BUILTIN_TYPES, GENERIC_TYPE_PATTERN
from .graph_types import EdgeType, GraphEdge, GraphNode, NodeType


class EdgeBuilder:
    """Builds edges between graph nodes based on code relationships.

    The EdgeBuilder analyzes GraphNode data to detect various types of
    relationships:

    1. Type Reference Edges:
       If a component has a field like `player: Player` and Player exists
       as a node, create an edge with type=EdgeType.REFERENCE

    2. Query Dependency Edges:
       If a system has Query[ComponentA, ComponentB] in method signatures,
       create edges from the system to each queried component with
       type=EdgeType.QUERY

    3. Inheritance Edges:
       If a class has base classes that exist as nodes, create edges
       with type=EdgeType.INHERITANCE

    4. Event Handler Edges:
       If a system method has an event type as a parameter, create an
       edge with type=EdgeType.EVENT_HANDLER

    Attributes:
        _nodes: List of all graph nodes
        _node_id_map: Mapping from class name to node ID
        _edges: List of built edges
    """

    # Compiled regex pattern for extracting generic type parameters
    _GENERIC_PATTERN = re.compile(GENERIC_TYPE_PATTERN)

    def __init__(
        self,
        nodes: list[GraphNode],
        node_id_map: dict[str, str],
    ) -> None:
        """Initialize the EdgeBuilder.

        Args:
            nodes: List of all graph nodes to analyze
            node_id_map: Mapping from class names to their node IDs
        """
        self._nodes = nodes
        self._node_id_map = node_id_map
        self._edges: list[GraphEdge] = []
        self._edge_ids: set[str] = set()  # Track to avoid duplicates

    def _generate_edge_id(self) -> str:
        """Generate a unique edge ID."""
        edge_id = f"edge_{uuid.uuid4().hex[:8]}"
        while edge_id in self._edge_ids:
            edge_id = f"edge_{uuid.uuid4().hex[:8]}"
        self._edge_ids.add(edge_id)
        return edge_id

    def _add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        label: Optional[str] = None,
        data: Optional[dict] = None,
    ) -> None:
        """Add an edge if it doesn't already exist.

        Args:
            source_id: ID of the source node
            target_id: ID of the target node
            edge_type: Type of the edge relationship
            label: Optional label for the edge
            data: Optional additional data for the edge
        """
        # Check for duplicate edges
        for edge in self._edges:
            if (
                edge.source == source_id
                and edge.target == target_id
                and edge.type == edge_type
            ):
                return  # Edge already exists

        edge = GraphEdge(
            id=self._generate_edge_id(),
            source=source_id,
            target=target_id,
            type=edge_type,
            label=label,
            data=data or {},
        )
        self._edges.append(edge)

    def _extract_type_names(self, type_annotation: str) -> list[str]:
        """Extract all type names from a type annotation string.

        Handles complex annotations like:
        - Simple types: "Player" -> ["Player"]
        - Generic types: "List[Player]" -> ["Player"]
        - Union types: "Player | None" -> ["Player"]
        - Nested generics: "Dict[str, Player]" -> ["Player"]
        - Multiple types: "Query[Player, Position]" -> ["Player", "Position"]

        Args:
            type_annotation: The type annotation string to parse

        Returns:
            List of extracted type names that could be class references
        """
        type_names: list[str] = []

        # Handle union with | operator
        parts = type_annotation.replace(" ", "").split("|")

        for part in parts:
            # Check for generic types like List[X], Optional[X], Query[X, Y]
            match = self._GENERIC_PATTERN.match(part)
            if match:
                outer_type = match.group(1)
                inner_types = match.group(2)

                # Recursively extract from inner types
                # Split by comma but be careful with nested generics
                inner_parts = self._split_generic_args(inner_types)
                for inner_part in inner_parts:
                    type_names.extend(self._extract_type_names(inner_part))

                # The outer type itself (unless it's a builtin generic)
                if outer_type not in BUILTIN_TYPES:
                    type_names.append(outer_type)
            else:
                # Simple type name
                clean_name = part.strip()
                if clean_name and clean_name not in BUILTIN_TYPES:
                    type_names.append(clean_name)

        return type_names

    def _split_generic_args(self, inner: str) -> list[str]:
        """Split generic arguments handling nested brackets.

        For "A, B" returns ["A", "B"]
        For "Dict[str, int], Player" returns ["Dict[str, int]", "Player"]

        Args:
            inner: The string inside the outermost brackets

        Returns:
            List of type argument strings
        """
        result: list[str] = []
        current = ""
        depth = 0

        for char in inner:
            if char == "[":
                depth += 1
                current += char
            elif char == "]":
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                if current.strip():
                    result.append(current.strip())
                current = ""
            else:
                current += char

        if current.strip():
            result.append(current.strip())

        return result

    def _get_node_by_id(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by its ID."""
        for node in self._nodes:
            if node.id == node_id:
                return node
        return None

    def build_reference_edges(self) -> None:
        """Scan all fields for type references and create edges.

        For each node, examines its fields' type annotations. If a type
        annotation references another class that exists as a node, creates
        an edge from the node containing the field to the referenced node.

        Edge type: EdgeType.REFERENCE
        Label: The field name
        """
        for node in self._nodes:
            fields = node.data.get("fields", [])

            for field_data in fields:
                field_name = field_data.get("name", "")
                type_annotation = field_data.get("type_annotation", "")

                if not type_annotation:
                    continue

                # Extract type names from the annotation
                referenced_types = self._extract_type_names(type_annotation)

                for type_name in referenced_types:
                    # Check if this type exists as a node
                    if type_name in self._node_id_map:
                        target_id = self._node_id_map[type_name]

                        # Don't create self-references
                        if target_id != node.id:
                            self._add_edge(
                                source_id=node.id,
                                target_id=target_id,
                                edge_type=EdgeType.REFERENCE,
                                label=field_name,
                                data={
                                    "field_name": field_name,
                                    "type_annotation": type_annotation,
                                },
                            )

    def build_query_edges(self) -> None:
        """Create edges from systems to their queried components.

        Examines each System node for Query[...] type annotations in method
        parameters. For each component type found in a Query, creates an
        edge from the system to that component.

        Edge type: EdgeType.QUERY
        Label: "queries"
        """
        for node in self._nodes:
            # Only process System nodes
            if node.type != NodeType.SYSTEM:
                continue

            # Get queries from node data (set by the graph builder from SystemDef)
            queries = node.data.get("queries", [])

            for component_name in queries:
                if component_name in self._node_id_map:
                    target_id = self._node_id_map[component_name]
                    self._add_edge(
                        source_id=node.id,
                        target_id=target_id,
                        edge_type=EdgeType.QUERY,
                        label="queries",
                        data={"query_component": component_name},
                    )

            # Also check methods for Query annotations in case queries
            # weren't pre-extracted
            methods = node.data.get("methods", [])
            for method in methods:
                self._extract_query_edges_from_method(node.id, method)

    def _extract_query_edges_from_method(
        self,
        source_node_id: str,
        method_data: dict,
    ) -> None:
        """Extract query edges from a method's parameters.

        Args:
            source_node_id: ID of the node containing the method
            method_data: Dictionary containing method information
        """
        # Check query_info if available
        query_info = method_data.get("query_info")
        if query_info:
            component_types = query_info.get("component_types", [])
            for comp_type in component_types:
                if comp_type in self._node_id_map:
                    target_id = self._node_id_map[comp_type]
                    self._add_edge(
                        source_id=source_node_id,
                        target_id=target_id,
                        edge_type=EdgeType.QUERY,
                        label="queries",
                        data={
                            "query_component": comp_type,
                            "method": method_data.get("name", ""),
                        },
                    )
            return

        # Fallback: scan parameters for Query[...] annotations
        parameters = method_data.get("parameters", [])
        for param in parameters:
            type_annotation = param.get("type_annotation", "")
            if not type_annotation:
                continue

            # Look for Query[...] pattern
            if "Query[" in type_annotation:
                # Extract types from Query[A, B, C]
                match = self._GENERIC_PATTERN.search(type_annotation)
                if match and match.group(1) == "Query":
                    inner = match.group(2)
                    component_types = self._split_generic_args(inner)

                    for comp_type in component_types:
                        comp_type = comp_type.strip()
                        if comp_type in self._node_id_map:
                            target_id = self._node_id_map[comp_type]
                            self._add_edge(
                                source_id=source_node_id,
                                target_id=target_id,
                                edge_type=EdgeType.QUERY,
                                label="queries",
                                data={
                                    "query_component": comp_type,
                                    "method": method_data.get("name", ""),
                                    "parameter": param.get("name", ""),
                                },
                            )

    def build_inheritance_edges(self) -> None:
        """Create edges for class inheritance relationships.

        For each node, checks if any of its base classes exist as nodes
        in the graph. If so, creates an inheritance edge.

        Edge type: EdgeType.INHERITANCE
        Label: "extends"

        Note: This is currently not common in Trinity ECS since decorated
        classes typically don't inherit from each other, but the support
        is included for completeness.
        """
        for node in self._nodes:
            bases = node.data.get("bases", [])

            for base_class in bases:
                if base_class in self._node_id_map:
                    target_id = self._node_id_map[base_class]

                    # Don't create self-references
                    if target_id != node.id:
                        self._add_edge(
                            source_id=node.id,
                            target_id=target_id,
                            edge_type=EdgeType.INHERITANCE,
                            label="extends",
                            data={"base_class": base_class},
                        )

    def build_event_handler_edges(self) -> None:
        """Create edges from systems to events they handle.

        Examines System nodes for method parameters that have event types.
        If a parameter's type annotation references an Event node, creates
        an edge.

        Edge type: EdgeType.EVENT_HANDLER
        Label: "handles"
        """
        # First, collect all event node names
        event_names: set[str] = set()
        for node in self._nodes:
            if node.type == NodeType.EVENT:
                event_names.add(node.name)

        # Only process System nodes
        for node in self._nodes:
            if node.type != NodeType.SYSTEM:
                continue

            methods = node.data.get("methods", [])
            for method in methods:
                parameters = method.get("parameters", [])

                for param in parameters:
                    type_annotation = param.get("type_annotation", "")
                    if not type_annotation:
                        continue

                    # Extract type names from the annotation
                    type_names = self._extract_type_names(type_annotation)

                    for type_name in type_names:
                        # Check if this type is an event
                        if type_name in event_names and type_name in self._node_id_map:
                            target_id = self._node_id_map[type_name]
                            self._add_edge(
                                source_id=node.id,
                                target_id=target_id,
                                edge_type=EdgeType.EVENT_HANDLER,
                                label="handles",
                                data={
                                    "event_type": type_name,
                                    "method": method.get("name", ""),
                                    "parameter": param.get("name", ""),
                                },
                            )

    def build_all_edges(self) -> list[GraphEdge]:
        """Build all edge types and return them.

        This is the main entry point for edge building. It calls all
        individual edge-building methods and returns the complete list
        of edges.

        Returns:
            List of all GraphEdge objects representing relationships
            between nodes.
        """
        # Clear any existing edges from previous runs
        self._edges = []
        self._edge_ids = set()

        # Build all edge types
        self.build_reference_edges()
        self.build_query_edges()
        self.build_inheritance_edges()
        self.build_event_handler_edges()

        return self._edges

    def get_edges(self) -> list[GraphEdge]:
        """Get the current list of built edges.

        Returns:
            List of edges that have been built so far.
        """
        return list(self._edges)

    def get_edges_by_type(self, edge_type: EdgeType) -> list[GraphEdge]:
        """Get all edges of a specific type.

        Args:
            edge_type: The type of edges to retrieve

        Returns:
            List of edges matching the specified type.
        """
        return [e for e in self._edges if e.type == edge_type]

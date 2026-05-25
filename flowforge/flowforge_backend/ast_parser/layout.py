"""Layout algorithms for auto-positioning nodes in FlowForge.

This module provides layout engines for positioning Trinity ECS nodes
in a visually meaningful arrangement for the graph canvas.

Hierarchical Layout:
    - Systems at top (they orchestrate logic)
    - Components in middle (data containers)
    - Resources on left side (shared global state)
    - Events on right side (communication signals)

Grid Layout:
    - Simple arrangement by type in columns
    - Configurable spacing and start position
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .constants import (
    COLUMN_SPACING,
    DEFAULT_START_X,
    DEFAULT_START_Y,
    HORIZONTAL_SPACING,
    NODE_HEIGHT,
    NODE_WIDTH,
    VERTICAL_SPACING,
)
from .graph_types import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodePosition,
    NodeType,
)

# Column order for hierarchical layout (left to right)
HIERARCHICAL_COLUMN_ORDER: list[NodeType] = [
    NodeType.RESOURCE,
    NodeType.SYSTEM,
    NodeType.COMPONENT,
    NodeType.EVENT,
]

# Row indices for hierarchical layout (top to bottom priority)
HIERARCHICAL_ROW_PRIORITY: dict[NodeType, int] = {
    NodeType.SYSTEM: 0,      # Systems at top
    NodeType.COMPONENT: 1,   # Components in middle
    NodeType.RESOURCE: 0,    # Resources at same level as systems
    NodeType.EVENT: 0,       # Events at same level as systems
}


# =============================================================================
# Layout Engine
# =============================================================================

class LayoutEngine:
    """Engine for computing node positions in the graph.

    This class provides multiple layout algorithms for arranging nodes
    in a visually meaningful way. Positions are updated in-place on the
    node objects.

    Attributes:
        _nodes: List of nodes to position.
        _edges: List of edges connecting nodes (used for edge-aware layouts).
        _start_x: Starting X coordinate for layouts.
        _start_y: Starting Y coordinate for layouts.

    Example:
        >>> nodes = [GraphNode(id="1", type=NodeType.COMPONENT, name="Position")]
        >>> edges = []
        >>> engine = LayoutEngine(nodes, edges)
        >>> engine.apply_hierarchical_layout()
        >>> print(nodes[0].position)
        (400.0, 100.0)
    """

    def __init__(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        start_x: float = DEFAULT_START_X,
        start_y: float = DEFAULT_START_Y,
    ) -> None:
        """Initialize the layout engine.

        Args:
            nodes: List of nodes to position.
            edges: List of edges connecting nodes.
            start_x: Starting X coordinate for layouts.
            start_y: Starting Y coordinate for layouts.
        """
        self._nodes = nodes
        self._edges = edges
        self._start_x = start_x
        self._start_y = start_y

    # -------------------------------------------------------------------------
    # Public Layout Methods
    # -------------------------------------------------------------------------

    def apply_hierarchical_layout(self) -> None:
        """Position nodes in a hierarchical arrangement.

        Layout structure:
            - Resources on the left column
            - Systems in the center-left (top area)
            - Components in the center-right (below systems if queried)
            - Events on the right column

        Nodes of the same type are stacked vertically within their column.
        Edge-aware positioning places queried components below their systems.
        """
        # Group nodes by type
        nodes_by_type = self._group_by_type()

        # Calculate column positions
        column_positions = self._calculate_column_positions(nodes_by_type)

        # Position each group
        for node_type, nodes in nodes_by_type.items():
            if not nodes:
                continue

            col_x = column_positions.get(node_type, self._start_x)
            base_y = self._start_y

            # Apply row priority offset for hierarchical positioning
            row_offset = HIERARCHICAL_ROW_PRIORITY.get(node_type, 1)
            base_y += row_offset * (NODE_HEIGHT + VERTICAL_SPACING)

            # Position nodes in this column
            for i, node in enumerate(nodes):
                x = col_x
                y = base_y + i * (NODE_HEIGHT + VERTICAL_SPACING)
                node.position = NodePosition(x=x, y=y)

        # Apply edge-aware adjustments for system-component relationships
        self._apply_edge_aware_positioning()

    def apply_grid_layout(self) -> None:
        """Position nodes in a simple grid arrangement by type.

        Nodes are arranged in columns by type, with each column containing
        nodes of the same type stacked vertically. This is simpler than
        hierarchical layout and doesn't consider edge relationships.
        """
        # Group nodes by type
        nodes_by_type = self._group_by_type()

        # Calculate column positions based on order
        current_x = self._start_x

        for node_type in HIERARCHICAL_COLUMN_ORDER:
            nodes = nodes_by_type.get(node_type, [])
            if not nodes:
                continue

            # Position nodes in this column
            for i, node in enumerate(nodes):
                x = current_x
                y = self._start_y + i * (NODE_HEIGHT + VERTICAL_SPACING)
                node.position = NodePosition(x=x, y=y)

            # Move to next column
            current_x += NODE_WIDTH + COLUMN_SPACING

    def apply_compact_layout(self) -> None:
        """Position nodes in a compact grid without type grouping.

        Useful for small graphs where type-based organization is not needed.
        Nodes are arranged left-to-right, top-to-bottom.
        """
        # Calculate columns based on total nodes
        total_nodes = len(self._nodes)
        if total_nodes == 0:
            return

        # Aim for roughly square layout
        cols = max(1, int(total_nodes ** 0.5))

        for i, node in enumerate(self._nodes):
            col = i % cols
            row = i // cols
            x = self._start_x + col * (NODE_WIDTH + HORIZONTAL_SPACING)
            y = self._start_y + row * (NODE_HEIGHT + VERTICAL_SPACING)
            node.position = NodePosition(x=x, y=y)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _group_by_type(self) -> dict[NodeType, list[GraphNode]]:
        """Group nodes by their type.

        Returns:
            Dictionary mapping NodeType to list of nodes of that type.
        """
        groups: dict[NodeType, list[GraphNode]] = defaultdict(list)
        for node in self._nodes:
            groups[node.type].append(node)
        return groups

    def _calculate_column_positions(
        self,
        nodes_by_type: dict[NodeType, list[GraphNode]],
    ) -> dict[NodeType, float]:
        """Calculate X positions for each type column.

        Args:
            nodes_by_type: Nodes grouped by type.

        Returns:
            Dictionary mapping NodeType to X coordinate.
        """
        positions: dict[NodeType, float] = {}
        current_x = self._start_x

        for node_type in HIERARCHICAL_COLUMN_ORDER:
            if nodes_by_type.get(node_type):
                positions[node_type] = current_x
                current_x += NODE_WIDTH + COLUMN_SPACING

        return positions

    def _apply_edge_aware_positioning(self) -> None:
        """Adjust component positions based on system query relationships.

        When a system queries specific components, those components are
        repositioned to be directly below the querying system for visual clarity.
        """
        # Find all query edges (system -> component relationships)
        query_edges = [e for e in self._edges if e.type == EdgeType.QUERY]

        if not query_edges:
            return

        # Build a map of system -> queried components
        system_queries: dict[str, list[str]] = defaultdict(list)
        for edge in query_edges:
            system_queries[edge.source].append(edge.target)

        # Build node lookup
        node_map = {node.id: node for node in self._nodes}

        # Track which components have been positioned relative to systems
        positioned_components: set[str] = set()

        for system_id, component_ids in system_queries.items():
            system_node = node_map.get(system_id)
            if system_node is None:
                continue

            system_x, system_y = system_node.position.x, system_node.position.y

            # Position queried components below and centered under the system
            for i, comp_id in enumerate(component_ids):
                if comp_id in positioned_components:
                    continue

                comp_node = node_map.get(comp_id)
                if comp_node is None:
                    continue

                # Position below system, offset horizontally for multiple
                offset_x = (i - (len(component_ids) - 1) / 2) * (NODE_WIDTH + HORIZONTAL_SPACING)
                new_x = system_x + offset_x
                new_y = system_y + NODE_HEIGHT + VERTICAL_SPACING

                comp_node.position = NodePosition(x=new_x, y=new_y)
                positioned_components.add(comp_id)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        """Calculate the bounding box of all positioned nodes.

        Returns:
            Tuple of (min_x, min_y, max_x, max_y) representing the bounds.
        """
        if not self._nodes:
            return (0.0, 0.0, 0.0, 0.0)

        min_x = min(node.position.x for node in self._nodes)
        min_y = min(node.position.y for node in self._nodes)
        max_x = max(node.position.x + NODE_WIDTH for node in self._nodes)
        max_y = max(node.position.y + NODE_HEIGHT for node in self._nodes)

        return (min_x, min_y, max_x, max_y)

    def center_in_viewport(
        self,
        viewport_width: float,
        viewport_height: float,
    ) -> None:
        """Center the graph within a viewport.

        Args:
            viewport_width: Width of the viewport in pixels.
            viewport_height: Height of the viewport in pixels.
        """
        if not self._nodes:
            return

        min_x, min_y, max_x, max_y = self.get_bounding_box()
        graph_width = max_x - min_x
        graph_height = max_y - min_y

        # Calculate offset to center
        offset_x = (viewport_width - graph_width) / 2 - min_x
        offset_y = (viewport_height - graph_height) / 2 - min_y

        # Apply offset to all nodes
        for node in self._nodes:
            x, y = node.position.x, node.position.y
            node.position = NodePosition(x=x + offset_x, y=y + offset_y)


# =============================================================================
# Convenience Functions
# =============================================================================

def apply_layout(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    layout_type: str = "hierarchical",
    start_x: float = DEFAULT_START_X,
    start_y: float = DEFAULT_START_Y,
) -> None:
    """Apply a layout algorithm to position nodes.

    This is a convenience function that creates a LayoutEngine and applies
    the specified layout algorithm.

    Args:
        nodes: List of nodes to position (modified in place).
        edges: List of edges connecting nodes.
        layout_type: Type of layout ("hierarchical", "grid", or "compact").
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.

    Raises:
        ValueError: If an unknown layout type is specified.

    Example:
        >>> from flowforge_backend.ast_parser.layout import apply_layout
        >>> apply_layout(nodes, edges, layout_type="hierarchical")
    """
    engine = LayoutEngine(nodes, edges, start_x, start_y)

    if layout_type == "hierarchical":
        engine.apply_hierarchical_layout()
    elif layout_type == "grid":
        engine.apply_grid_layout()
    elif layout_type == "compact":
        engine.apply_compact_layout()
    else:
        raise ValueError(f"Unknown layout type: {layout_type}")


def get_layout_info() -> dict[str, dict[str, int]]:
    """Get information about layout constants.

    Returns:
        Dictionary containing layout configuration values.

    Example:
        >>> info = get_layout_info()
        >>> print(info["node_dimensions"]["width"])
        200
    """
    return {
        "node_dimensions": {
            "width": NODE_WIDTH,
            "height": NODE_HEIGHT,
        },
        "spacing": {
            "horizontal": HORIZONTAL_SPACING,
            "vertical": VERTICAL_SPACING,
            "column": COLUMN_SPACING,
        },
        "defaults": {
            "start_x": int(DEFAULT_START_X),
            "start_y": int(DEFAULT_START_Y),
        },
    }

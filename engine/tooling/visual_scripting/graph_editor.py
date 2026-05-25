"""
FlowForge Graph Editor - Visual graph editor with zoom, pan, minimap, and node management.

Provides the core visual editing functionality:
- Graph model with nodes and connections
- Zoom and pan navigation
- Minimap for overview
- Node creation, deletion, and movement
- Connection management with validation
- Selection and multi-selection
- Copy/paste and undo/redo
- Grid snapping
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from .node_types import Node, Pin, PinDirection, PinKind, NodeCategory


class SelectionMode(Enum):
    """Selection modes for the editor."""
    REPLACE = auto()
    ADD = auto()
    REMOVE = auto()
    TOGGLE = auto()


@dataclass
class Connection:
    """A connection between two pins."""
    id: str
    source_node_id: str
    source_pin_id: str
    target_node_id: str
    target_pin_id: str
    is_execution: bool = False

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class ViewState:
    """State of the editor viewport."""
    pan_x: float = 0.0
    pan_y: float = 0.0
    zoom: float = 1.0
    zoom_min: float = 0.1
    zoom_max: float = 5.0

    def screen_to_graph(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """Convert screen coordinates to graph coordinates."""
        graph_x = (screen_x - self.pan_x) / self.zoom
        graph_y = (screen_y - self.pan_y) / self.zoom
        return (graph_x, graph_y)

    def graph_to_screen(self, graph_x: float, graph_y: float) -> Tuple[float, float]:
        """Convert graph coordinates to screen coordinates."""
        screen_x = graph_x * self.zoom + self.pan_x
        screen_y = graph_y * self.zoom + self.pan_y
        return (screen_x, screen_y)

    def zoom_to_point(self, screen_x: float, screen_y: float, new_zoom: float) -> None:
        """Zoom centered on a point in screen coordinates."""
        # Get graph position before zoom
        graph_x, graph_y = self.screen_to_graph(screen_x, screen_y)

        # Apply zoom
        self.zoom = max(self.zoom_min, min(self.zoom_max, new_zoom))

        # Adjust pan to keep point under cursor
        self.pan_x = screen_x - graph_x * self.zoom
        self.pan_y = screen_y - graph_y * self.zoom


@dataclass
class EditorAction:
    """An undoable action in the editor."""
    action_type: str
    data: Dict[str, Any]
    timestamp: float = 0.0


class UndoStack:
    """Undo/redo stack for editor actions."""

    def __init__(self, max_size: int = 100):
        self._undo_stack: List[EditorAction] = []
        self._redo_stack: List[EditorAction] = []
        self._max_size = max_size
        self._is_executing = False

    def push(self, action: EditorAction) -> None:
        """Push an action onto the undo stack."""
        if self._is_executing:
            return

        self._undo_stack.append(action)
        self._redo_stack.clear()

        # Limit stack size
        while len(self._undo_stack) > self._max_size:
            self._undo_stack.pop(0)

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0

    def undo(self) -> Optional[EditorAction]:
        """Pop and return the last action for undo."""
        if not self._undo_stack:
            return None

        action = self._undo_stack.pop()
        self._redo_stack.append(action)
        return action

    def redo(self) -> Optional[EditorAction]:
        """Pop and return the last undone action for redo."""
        if not self._redo_stack:
            return None

        action = self._redo_stack.pop()
        self._undo_stack.append(action)
        return action

    def clear(self) -> None:
        """Clear both stacks."""
        self._undo_stack.clear()
        self._redo_stack.clear()

    def begin_compound(self) -> None:
        """Begin a compound action (groups multiple actions)."""
        self._is_executing = True

    def end_compound(self) -> None:
        """End a compound action."""
        self._is_executing = False


@dataclass
class ClipboardData:
    """Data for copy/paste operations."""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    connections: List[Dict[str, Any]] = field(default_factory=list)
    offset: Tuple[float, float] = (0, 0)


class BlueprintGraph:
    """The graph model containing nodes and connections."""

    def __init__(self, graph_id: Optional[str] = None, name: str = "NewGraph"):
        self.id = graph_id or str(uuid.uuid4())
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.connections: Dict[str, Connection] = {}

        # Graph metadata
        self.description: str = ""
        self.category: str = ""
        self.is_macro: bool = False
        self.parent_class: str = ""

        # Entry points (event nodes)
        self.entry_points: List[str] = []

    def add_node(self, node: Node) -> Node:
        """Add a node to the graph."""
        self.nodes[node.id] = node

        # Track entry points
        if node.get_metadata().category == NodeCategory.EVENT:
            if node.id not in self.entry_points:
                self.entry_points.append(node.id)

        return node

    def remove_node(self, node_id: str) -> Optional[Node]:
        """Remove a node and its connections."""
        node = self.nodes.pop(node_id, None)
        if node:
            # Remove all connections to/from this node
            to_remove = [
                conn_id for conn_id, conn in self.connections.items()
                if conn.source_node_id == node_id or conn.target_node_id == node_id
            ]
            for conn_id in to_remove:
                self.remove_connection(conn_id)

            # Remove from entry points
            if node_id in self.entry_points:
                self.entry_points.remove(node_id)

        return node

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def add_connection(self, connection: Connection) -> Optional[Connection]:
        """Add a connection between pins."""
        # Validate connection
        source_node = self.get_node(connection.source_node_id)
        target_node = self.get_node(connection.target_node_id)

        if not source_node or not target_node:
            return None

        source_pin = source_node.get_output_pin(connection.source_pin_id)
        if not source_pin:
            # Try by ID directly
            for pin in source_node.output_pins.values():
                if pin.id == connection.source_pin_id:
                    source_pin = pin
                    break

        target_pin = target_node.get_input_pin(connection.target_pin_id)
        if not target_pin:
            # Try by ID directly
            for pin in target_node.input_pins.values():
                if pin.id == connection.target_pin_id:
                    target_pin = pin
                    break

        if not source_pin or not target_pin:
            return None

        if not source_pin.can_connect_to(target_pin):
            return None

        # For data pins, remove existing connection to target
        if target_pin.kind == PinKind.DATA:
            existing = self._find_connection_to_pin(connection.target_pin_id)
            if existing:
                self.remove_connection(existing.id)

        # Update pin connection state
        source_pin.is_connected = True
        source_pin.connected_to.append(connection.target_pin_id)
        target_pin.is_connected = True
        target_pin.connected_to.append(connection.source_pin_id)

        connection.is_execution = source_pin.kind == PinKind.EXECUTION
        self.connections[connection.id] = connection
        return connection

    def remove_connection(self, connection_id: str) -> Optional[Connection]:
        """Remove a connection."""
        connection = self.connections.pop(connection_id, None)
        if connection:
            # Update pin connection state
            source_node = self.get_node(connection.source_node_id)
            target_node = self.get_node(connection.target_node_id)

            if source_node:
                for pin in source_node.output_pins.values():
                    if pin.id == connection.source_pin_id:
                        if connection.target_pin_id in pin.connected_to:
                            pin.connected_to.remove(connection.target_pin_id)
                        pin.is_connected = len(pin.connected_to) > 0

            if target_node:
                for pin in target_node.input_pins.values():
                    if pin.id == connection.target_pin_id:
                        if connection.source_pin_id in pin.connected_to:
                            pin.connected_to.remove(connection.source_pin_id)
                        pin.is_connected = len(pin.connected_to) > 0

        return connection

    def _find_connection_to_pin(self, pin_id: str) -> Optional[Connection]:
        """Find a connection going to a pin."""
        for conn in self.connections.values():
            if conn.target_pin_id == pin_id:
                return conn
        return None

    def get_connections_for_node(self, node_id: str) -> List[Connection]:
        """Get all connections for a node."""
        return [
            conn for conn in self.connections.values()
            if conn.source_node_id == node_id or conn.target_node_id == node_id
        ]

    def get_outgoing_connections(self, node_id: str) -> List[Connection]:
        """Get connections going out from a node."""
        return [
            conn for conn in self.connections.values()
            if conn.source_node_id == node_id
        ]

    def get_incoming_connections(self, node_id: str) -> List[Connection]:
        """Get connections coming into a node."""
        return [
            conn for conn in self.connections.values()
            if conn.target_node_id == node_id
        ]

    def get_connected_nodes(self, node_id: str) -> List[Node]:
        """Get all nodes connected to a node."""
        connected_ids = set()
        for conn in self.get_connections_for_node(node_id):
            if conn.source_node_id == node_id:
                connected_ids.add(conn.target_node_id)
            else:
                connected_ids.add(conn.source_node_id)
        return [self.nodes[nid] for nid in connected_ids if nid in self.nodes]

    def validate(self) -> List[str]:
        """Validate the entire graph. Returns list of error messages."""
        errors = []

        # Validate each node
        for node in self.nodes.values():
            node_errors = node.validate()
            errors.extend(node_errors)

        # Check for disconnected required pins
        for node in self.nodes.values():
            for pin in node.input_pins.values():
                if pin.kind == PinKind.EXECUTION and not pin.is_connected:
                    if node.get_metadata().category != NodeCategory.EVENT:
                        errors.append(
                            f"Node '{node.get_metadata().display_name}' ({node.id}): "
                            f"Execution input '{pin.name}' is not connected"
                        )

        # Check for circular references (would cause infinite loops)
        # This is a simplified check - full cycle detection would be more complex
        visited: Set[str] = set()
        for entry in self.entry_points:
            cycle = self._detect_cycle(entry, visited, set())
            if cycle:
                errors.append(f"Circular reference detected starting from: {cycle}")

        return errors

    def _detect_cycle(
        self,
        node_id: str,
        visited: Set[str],
        current_path: Set[str]
    ) -> Optional[str]:
        """Detect cycles in the graph (simplified)."""
        if node_id in current_path:
            return node_id

        if node_id in visited:
            return None

        visited.add(node_id)
        current_path.add(node_id)

        for conn in self.get_outgoing_connections(node_id):
            if conn.is_execution:
                result = self._detect_cycle(conn.target_node_id, visited, current_path)
                if result:
                    return result

        current_path.remove(node_id)
        return None

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get the bounding box of all nodes (min_x, min_y, max_x, max_y)."""
        if not self.nodes:
            return (0, 0, 0, 0)

        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')

        for node in self.nodes.values():
            x, y = node.position
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + 200)  # Assume node width
            max_y = max(max_y, y + 100)  # Assume node height

        return (min_x, min_y, max_x, max_y)


class GraphEditor:
    """Visual editor for blueprint graphs."""

    def __init__(self, graph: Optional[BlueprintGraph] = None):
        self.graph = graph or BlueprintGraph()
        self.view = ViewState()

        # Selection state
        self.selected_nodes: Set[str] = set()
        self.selected_connections: Set[str] = set()

        # Interaction state
        self.is_dragging = False
        self.is_connecting = False
        self.drag_start: Optional[Tuple[float, float]] = None
        self.connection_start_pin: Optional[str] = None
        self.hovering_node: Optional[str] = None
        self.hovering_pin: Optional[str] = None

        # Undo/redo
        self.undo_stack = UndoStack()

        # Clipboard
        self.clipboard: Optional[ClipboardData] = None

        # Grid settings
        self.grid_size = 16
        self.snap_to_grid = True
        self.show_grid = True

        # Minimap settings
        self.show_minimap = True
        self.minimap_size = (200, 150)
        self.minimap_position = (10, 10)

        # Event callbacks
        self.on_node_selected: Optional[Callable[[str], None]] = None
        self.on_node_deselected: Optional[Callable[[str], None]] = None
        self.on_graph_changed: Optional[Callable[[], None]] = None

    # =========================================================================
    # NAVIGATION
    # =========================================================================

    def pan(self, delta_x: float, delta_y: float) -> None:
        """Pan the view by delta."""
        self.view.pan_x += delta_x
        self.view.pan_y += delta_y

    def zoom(self, factor: float, center_x: float = 0, center_y: float = 0) -> None:
        """Zoom the view by factor, centered on point."""
        new_zoom = self.view.zoom * factor
        self.view.zoom_to_point(center_x, center_y, new_zoom)

    def zoom_to_fit(self, padding: float = 50) -> None:
        """Zoom to fit all nodes in view."""
        bounds = self.graph.get_bounds()
        if bounds == (0, 0, 0, 0):
            return

        min_x, min_y, max_x, max_y = bounds
        width = max_x - min_x + padding * 2
        height = max_y - min_y + padding * 2

        # Calculate zoom to fit (assuming viewport size - would need actual size)
        # For now, just center on bounds
        self.view.pan_x = -min_x + padding
        self.view.pan_y = -min_y + padding
        self.view.zoom = 1.0

    def center_on_node(self, node_id: str) -> None:
        """Center the view on a specific node."""
        node = self.graph.get_node(node_id)
        if node:
            x, y = node.position
            # Center (assuming some viewport size)
            self.view.pan_x = -x + 400  # Half viewport width
            self.view.pan_y = -y + 300  # Half viewport height

    def center_on_selection(self) -> None:
        """Center the view on selected nodes."""
        if not self.selected_nodes:
            return

        # Calculate center of selection
        total_x = total_y = 0
        for node_id in self.selected_nodes:
            node = self.graph.get_node(node_id)
            if node:
                total_x += node.position[0]
                total_y += node.position[1]

        center_x = total_x / len(self.selected_nodes)
        center_y = total_y / len(self.selected_nodes)

        self.view.pan_x = -center_x + 400
        self.view.pan_y = -center_y + 300

    # =========================================================================
    # SELECTION
    # =========================================================================

    def select_node(self, node_id: str, mode: SelectionMode = SelectionMode.REPLACE) -> None:
        """Select a node."""
        if mode == SelectionMode.REPLACE:
            self.clear_selection()
            self.selected_nodes.add(node_id)
        elif mode == SelectionMode.ADD:
            self.selected_nodes.add(node_id)
        elif mode == SelectionMode.REMOVE:
            self.selected_nodes.discard(node_id)
        elif mode == SelectionMode.TOGGLE:
            if node_id in self.selected_nodes:
                self.selected_nodes.discard(node_id)
            else:
                self.selected_nodes.add(node_id)

        if self.on_node_selected and node_id in self.selected_nodes:
            self.on_node_selected(node_id)
        elif self.on_node_deselected and node_id not in self.selected_nodes:
            self.on_node_deselected(node_id)

    def select_nodes_in_rect(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        mode: SelectionMode = SelectionMode.REPLACE
    ) -> None:
        """Select all nodes within a rectangle."""
        if mode == SelectionMode.REPLACE:
            self.clear_selection()

        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)

        for node_id, node in self.graph.nodes.items():
            nx, ny = node.position
            if min_x <= nx <= max_x and min_y <= ny <= max_y:
                if mode == SelectionMode.REMOVE:
                    self.selected_nodes.discard(node_id)
                else:
                    self.selected_nodes.add(node_id)

    def select_all(self) -> None:
        """Select all nodes."""
        self.selected_nodes = set(self.graph.nodes.keys())

    def clear_selection(self) -> None:
        """Clear all selection."""
        for node_id in list(self.selected_nodes):
            if self.on_node_deselected:
                self.on_node_deselected(node_id)
        self.selected_nodes.clear()
        self.selected_connections.clear()

    def is_selected(self, node_id: str) -> bool:
        """Check if a node is selected."""
        return node_id in self.selected_nodes

    def get_selection(self) -> List[Node]:
        """Get all selected nodes."""
        return [self.graph.nodes[nid] for nid in self.selected_nodes if nid in self.graph.nodes]

    # =========================================================================
    # NODE OPERATIONS
    # =========================================================================

    def create_node(
        self,
        node_class: Type[Node],
        position: Tuple[float, float],
        **kwargs
    ) -> Node:
        """Create and add a new node at position."""
        if self.snap_to_grid:
            position = self._snap_to_grid(position)

        node = node_class(position=position, **kwargs)
        self.graph.add_node(node)

        # Record action for undo
        self.undo_stack.push(EditorAction(
            action_type="create_node",
            data={"node_id": node.id, "node_class": node_class.__name__, "position": position}
        ))

        self._notify_change()
        return node

    def delete_selected(self) -> None:
        """Delete all selected nodes and connections."""
        if not self.selected_nodes and not self.selected_connections:
            return

        # Record for undo
        deleted_nodes = []
        deleted_connections = []

        for node_id in list(self.selected_nodes):
            node = self.graph.get_node(node_id)
            if node:
                deleted_nodes.append({
                    "id": node.id,
                    "class": type(node).__name__,
                    "position": node.position
                })
                # Get connections before removing
                for conn in self.graph.get_connections_for_node(node_id):
                    deleted_connections.append({
                        "id": conn.id,
                        "source_node": conn.source_node_id,
                        "source_pin": conn.source_pin_id,
                        "target_node": conn.target_node_id,
                        "target_pin": conn.target_pin_id
                    })
                self.graph.remove_node(node_id)

        for conn_id in list(self.selected_connections):
            conn = self.graph.connections.get(conn_id)
            if conn:
                deleted_connections.append({
                    "id": conn.id,
                    "source_node": conn.source_node_id,
                    "source_pin": conn.source_pin_id,
                    "target_node": conn.target_node_id,
                    "target_pin": conn.target_pin_id
                })
                self.graph.remove_connection(conn_id)

        self.undo_stack.push(EditorAction(
            action_type="delete",
            data={"nodes": deleted_nodes, "connections": deleted_connections}
        ))

        self.clear_selection()
        self._notify_change()

    def move_selected(self, delta_x: float, delta_y: float) -> None:
        """Move all selected nodes by delta."""
        for node_id in self.selected_nodes:
            node = self.graph.get_node(node_id)
            if node:
                new_x = node.position[0] + delta_x
                new_y = node.position[1] + delta_y
                if self.snap_to_grid:
                    new_x, new_y = self._snap_to_grid((new_x, new_y))
                node.position = (new_x, new_y)

        self._notify_change()

    def duplicate_selected(self, offset: Tuple[float, float] = (50, 50)) -> List[Node]:
        """Duplicate selected nodes with offset."""
        new_nodes = []
        id_map: Dict[str, str] = {}

        # First pass: duplicate nodes
        for node_id in self.selected_nodes:
            old_node = self.graph.get_node(node_id)
            if old_node:
                new_node = type(old_node)(
                    position=(old_node.position[0] + offset[0], old_node.position[1] + offset[1])
                )
                self.graph.add_node(new_node)
                new_nodes.append(new_node)
                id_map[old_node.id] = new_node.id

        # Second pass: duplicate connections between selected nodes
        for conn in self.graph.connections.values():
            if conn.source_node_id in id_map and conn.target_node_id in id_map:
                new_conn = Connection(
                    id=str(uuid.uuid4()),
                    source_node_id=id_map[conn.source_node_id],
                    source_pin_id=conn.source_pin_id,  # Pin IDs would need remapping too
                    target_node_id=id_map[conn.target_node_id],
                    target_pin_id=conn.target_pin_id
                )
                self.graph.add_connection(new_conn)

        # Select new nodes
        self.clear_selection()
        for node in new_nodes:
            self.selected_nodes.add(node.id)

        self._notify_change()
        return new_nodes

    def align_selected(self, alignment: str = "left") -> None:
        """Align selected nodes (left, right, top, bottom, center_h, center_v)."""
        if len(self.selected_nodes) < 2:
            return

        positions = []
        for node_id in self.selected_nodes:
            node = self.graph.get_node(node_id)
            if node:
                positions.append((node_id, node.position))

        if alignment == "left":
            target_x = min(p[1][0] for p in positions)
            for node_id, _ in positions:
                node = self.graph.get_node(node_id)
                if node:
                    node.position = (target_x, node.position[1])
        elif alignment == "right":
            target_x = max(p[1][0] for p in positions)
            for node_id, _ in positions:
                node = self.graph.get_node(node_id)
                if node:
                    node.position = (target_x, node.position[1])
        elif alignment == "top":
            target_y = min(p[1][1] for p in positions)
            for node_id, _ in positions:
                node = self.graph.get_node(node_id)
                if node:
                    node.position = (node.position[0], target_y)
        elif alignment == "bottom":
            target_y = max(p[1][1] for p in positions)
            for node_id, _ in positions:
                node = self.graph.get_node(node_id)
                if node:
                    node.position = (node.position[0], target_y)
        elif alignment == "center_h":
            avg_x = sum(p[1][0] for p in positions) / len(positions)
            for node_id, _ in positions:
                node = self.graph.get_node(node_id)
                if node:
                    node.position = (avg_x, node.position[1])
        elif alignment == "center_v":
            avg_y = sum(p[1][1] for p in positions) / len(positions)
            for node_id, _ in positions:
                node = self.graph.get_node(node_id)
                if node:
                    node.position = (node.position[0], avg_y)

        self._notify_change()

    # =========================================================================
    # CONNECTION OPERATIONS
    # =========================================================================

    def begin_connection(self, pin_id: str) -> None:
        """Start creating a connection from a pin."""
        self.is_connecting = True
        self.connection_start_pin = pin_id

    def complete_connection(self, target_pin_id: str) -> Optional[Connection]:
        """Complete a connection to a target pin."""
        if not self.is_connecting or not self.connection_start_pin:
            return None

        # Find source and target nodes/pins
        source_node = None
        source_pin = None
        target_node = None
        target_pin = None

        for node in self.graph.nodes.values():
            for pin in node.output_pins.values():
                if pin.id == self.connection_start_pin:
                    source_node = node
                    source_pin = pin
            for pin in node.input_pins.values():
                if pin.id == self.connection_start_pin:
                    # Started from input, swap
                    target_node = node
                    target_pin = pin

            for pin in node.input_pins.values():
                if pin.id == target_pin_id:
                    if source_pin:
                        target_node = node
                        target_pin = pin
                    else:
                        source_node = node
                        source_pin = pin
            for pin in node.output_pins.values():
                if pin.id == target_pin_id:
                    if source_pin:
                        # Started from output, target is output - invalid
                        pass
                    else:
                        source_node = node
                        source_pin = pin

        self.cancel_connection()

        if not source_node or not source_pin or not target_node or not target_pin:
            return None

        connection = Connection(
            id=str(uuid.uuid4()),
            source_node_id=source_node.id,
            source_pin_id=source_pin.id,
            target_node_id=target_node.id,
            target_pin_id=target_pin.id
        )

        result = self.graph.add_connection(connection)
        if result:
            self.undo_stack.push(EditorAction(
                action_type="connect",
                data={
                    "connection_id": connection.id,
                    "source_node": source_node.id,
                    "source_pin": source_pin.id,
                    "target_node": target_node.id,
                    "target_pin": target_pin.id
                }
            ))
            self._notify_change()

        return result

    def cancel_connection(self) -> None:
        """Cancel the current connection operation."""
        self.is_connecting = False
        self.connection_start_pin = None

    def disconnect_pin(self, pin_id: str) -> List[Connection]:
        """Disconnect all connections from/to a pin."""
        removed = []
        for conn_id, conn in list(self.graph.connections.items()):
            if conn.source_pin_id == pin_id or conn.target_pin_id == pin_id:
                self.graph.remove_connection(conn_id)
                removed.append(conn)

        if removed:
            self._notify_change()

        return removed

    # =========================================================================
    # CLIPBOARD OPERATIONS
    # =========================================================================

    def copy(self) -> None:
        """Copy selected nodes to clipboard."""
        if not self.selected_nodes:
            return

        nodes_data = []
        connections_data = []

        # Calculate offset from first node
        first_node = self.graph.get_node(list(self.selected_nodes)[0])
        offset = first_node.position if first_node else (0, 0)

        for node_id in self.selected_nodes:
            node = self.graph.get_node(node_id)
            if node:
                nodes_data.append({
                    "id": node.id,
                    "class": type(node).__name__,
                    "position": (
                        node.position[0] - offset[0],
                        node.position[1] - offset[1]
                    ),
                    "comment": node.comment
                })

        # Copy internal connections
        for conn in self.graph.connections.values():
            if conn.source_node_id in self.selected_nodes and conn.target_node_id in self.selected_nodes:
                connections_data.append({
                    "source_node": conn.source_node_id,
                    "source_pin": conn.source_pin_id,
                    "target_node": conn.target_node_id,
                    "target_pin": conn.target_pin_id
                })

        self.clipboard = ClipboardData(
            nodes=nodes_data,
            connections=connections_data,
            offset=offset
        )

    def cut(self) -> None:
        """Cut selected nodes to clipboard."""
        self.copy()
        self.delete_selected()

    def paste(self, position: Optional[Tuple[float, float]] = None) -> List[Node]:
        """Paste nodes from clipboard."""
        if not self.clipboard or not self.clipboard.nodes:
            return []

        # Determine paste position
        if position is None:
            position = (100, 100)  # Default offset

        new_nodes = []
        id_map: Dict[str, str] = {}

        # Create nodes
        for node_data in self.clipboard.nodes:
            from . import node_types
            node_class = getattr(node_types, node_data["class"], None)
            if node_class:
                new_pos = (
                    position[0] + node_data["position"][0],
                    position[1] + node_data["position"][1]
                )
                node = node_class(position=new_pos)
                node.comment = node_data.get("comment", "")
                self.graph.add_node(node)
                new_nodes.append(node)
                id_map[node_data["id"]] = node.id

        # Recreate connections
        for conn_data in self.clipboard.connections:
            if conn_data["source_node"] in id_map and conn_data["target_node"] in id_map:
                connection = Connection(
                    id=str(uuid.uuid4()),
                    source_node_id=id_map[conn_data["source_node"]],
                    source_pin_id=conn_data["source_pin"],
                    target_node_id=id_map[conn_data["target_node"]],
                    target_pin_id=conn_data["target_pin"]
                )
                self.graph.add_connection(connection)

        # Select pasted nodes
        self.clear_selection()
        for node in new_nodes:
            self.selected_nodes.add(node.id)

        self._notify_change()
        return new_nodes

    # =========================================================================
    # UNDO/REDO
    # =========================================================================

    def undo(self) -> bool:
        """Undo the last action."""
        action = self.undo_stack.undo()
        if not action:
            return False

        self._apply_undo(action)
        self._notify_change()
        return True

    def redo(self) -> bool:
        """Redo the last undone action."""
        action = self.undo_stack.redo()
        if not action:
            return False

        self._apply_redo(action)
        self._notify_change()
        return True

    def _apply_undo(self, action: EditorAction) -> None:
        """Apply an undo action."""
        if action.action_type == "create_node":
            self.graph.remove_node(action.data["node_id"])
        elif action.action_type == "delete":
            # Recreate deleted nodes and connections
            pass  # Would need full node serialization
        elif action.action_type == "connect":
            self.graph.remove_connection(action.data["connection_id"])
        # Add more action types as needed

    def _apply_redo(self, action: EditorAction) -> None:
        """Apply a redo action."""
        if action.action_type == "create_node":
            # Would need to recreate the node
            pass
        elif action.action_type == "delete":
            # Delete again
            pass
        elif action.action_type == "connect":
            # Recreate connection
            pass

    # =========================================================================
    # MINIMAP
    # =========================================================================

    def get_minimap_data(self) -> Dict[str, Any]:
        """Get data for rendering the minimap."""
        bounds = self.graph.get_bounds()
        nodes = []

        for node_id, node in self.graph.nodes.items():
            nodes.append({
                "id": node_id,
                "position": node.position,
                "selected": node_id in self.selected_nodes,
                "category": node.get_metadata().category.name
            })

        return {
            "bounds": bounds,
            "nodes": nodes,
            "view": {
                "pan_x": self.view.pan_x,
                "pan_y": self.view.pan_y,
                "zoom": self.view.zoom
            }
        }

    def minimap_click(self, minimap_x: float, minimap_y: float) -> None:
        """Handle click on minimap to pan view."""
        bounds = self.graph.get_bounds()
        if bounds == (0, 0, 0, 0):
            return

        min_x, min_y, max_x, max_y = bounds
        graph_width = max_x - min_x
        graph_height = max_y - min_y

        if graph_width == 0 or graph_height == 0:
            return

        # Convert minimap coords to graph coords
        graph_x = min_x + (minimap_x / self.minimap_size[0]) * graph_width
        graph_y = min_y + (minimap_y / self.minimap_size[1]) * graph_height

        # Center view on this point
        self.view.pan_x = -graph_x + 400
        self.view.pan_y = -graph_y + 300

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _snap_to_grid(self, position: Tuple[float, float]) -> Tuple[float, float]:
        """Snap a position to the grid."""
        x = round(position[0] / self.grid_size) * self.grid_size
        y = round(position[1] / self.grid_size) * self.grid_size
        return (x, y)

    def _notify_change(self) -> None:
        """Notify that the graph has changed."""
        if self.on_graph_changed:
            self.on_graph_changed()

    def find_node_at_position(self, x: float, y: float) -> Optional[Node]:
        """Find a node at the given graph position."""
        # Iterate in reverse (top nodes first)
        for node in reversed(list(self.graph.nodes.values())):
            nx, ny = node.position
            # Assume node size of 200x100
            if nx <= x <= nx + 200 and ny <= y <= ny + 100:
                return node
        return None

    def find_pin_at_position(self, x: float, y: float) -> Optional[Tuple[Node, Pin]]:
        """Find a pin at the given graph position."""
        node = self.find_node_at_position(x, y)
        if not node:
            return None

        # Simplified pin hit detection
        nx, ny = node.position
        pin_size = 10

        # Check input pins (left side)
        pin_y = ny + 30
        for pin in node.input_pins.values():
            if nx - pin_size <= x <= nx + pin_size and pin_y - pin_size <= y <= pin_y + pin_size:
                return (node, pin)
            pin_y += 25

        # Check output pins (right side)
        pin_y = ny + 30
        for pin in node.output_pins.values():
            if nx + 200 - pin_size <= x <= nx + 200 + pin_size and pin_y - pin_size <= y <= pin_y + pin_size:
                return (node, pin)
            pin_y += 25

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        categories = {}
        for node in self.graph.nodes.values():
            cat = node.get_metadata().category.name
            categories[cat] = categories.get(cat, 0) + 1

        return {
            "node_count": len(self.graph.nodes),
            "connection_count": len(self.graph.connections),
            "entry_point_count": len(self.graph.entry_points),
            "selected_count": len(self.selected_nodes),
            "categories": categories
        }

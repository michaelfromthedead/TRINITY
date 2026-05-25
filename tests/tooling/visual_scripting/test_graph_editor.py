"""
Tests for FlowForge graph editor.

Tests graph operations, navigation, node management, and editor functionality.
"""

import pytest

from engine.tooling.visual_scripting.graph_editor import (
    SelectionMode,
    Connection,
    ViewState,
    EditorAction,
    UndoStack,
    ClipboardData,
    BlueprintGraph,
    GraphEditor,
)
from engine.tooling.visual_scripting.node_types import (
    BeginPlayNode,
    TickNode,
    BranchNode,
    PrintStringNode,
    IntLiteralNode,
    NodeCategory,
)


class TestConnection:
    """Tests for Connection class."""

    def test_create_connection(self):
        conn = Connection(
            id="conn_1",
            source_node_id="node_1",
            source_pin_id="out",
            target_node_id="node_2",
            target_pin_id="in"
        )
        assert conn.source_node_id == "node_1"
        assert conn.target_node_id == "node_2"

    def test_auto_generate_id(self):
        conn = Connection(
            id="",
            source_node_id="node_1",
            source_pin_id="out",
            target_node_id="node_2",
            target_pin_id="in"
        )
        assert conn.id != ""


class TestViewState:
    """Tests for ViewState class."""

    def test_default_values(self):
        view = ViewState()
        assert view.pan_x == 0.0
        assert view.pan_y == 0.0
        assert view.zoom == 1.0

    def test_screen_to_graph(self):
        view = ViewState(pan_x=100, pan_y=50, zoom=1.0)
        gx, gy = view.screen_to_graph(200, 150)
        assert gx == 100
        assert gy == 100

    def test_graph_to_screen(self):
        view = ViewState(pan_x=100, pan_y=50, zoom=1.0)
        sx, sy = view.graph_to_screen(100, 100)
        assert sx == 200
        assert sy == 150

    def test_screen_to_graph_with_zoom(self):
        view = ViewState(pan_x=0, pan_y=0, zoom=2.0)
        gx, gy = view.screen_to_graph(200, 100)
        assert gx == 100
        assert gy == 50

    def test_zoom_to_point(self):
        view = ViewState(pan_x=0, pan_y=0, zoom=1.0)
        view.zoom_to_point(100, 100, 2.0)
        assert view.zoom == 2.0
        # Zoom should be centered on point

    def test_zoom_clamps_to_limits(self):
        view = ViewState(zoom_min=0.5, zoom_max=2.0)
        view.zoom_to_point(0, 0, 10.0)
        assert view.zoom == 2.0

        view.zoom_to_point(0, 0, 0.1)
        assert view.zoom == 0.5


class TestUndoStack:
    """Tests for UndoStack class."""

    def test_push_action(self):
        stack = UndoStack()
        action = EditorAction(action_type="create", data={"id": "1"})
        stack.push(action)

        assert stack.can_undo() is True
        assert stack.can_redo() is False

    def test_undo(self):
        stack = UndoStack()
        action = EditorAction(action_type="create", data={"id": "1"})
        stack.push(action)

        undone = stack.undo()
        assert undone == action
        assert stack.can_undo() is False
        assert stack.can_redo() is True

    def test_redo(self):
        stack = UndoStack()
        action = EditorAction(action_type="create", data={"id": "1"})
        stack.push(action)
        stack.undo()

        redone = stack.redo()
        assert redone == action
        assert stack.can_undo() is True
        assert stack.can_redo() is False

    def test_push_clears_redo(self):
        stack = UndoStack()
        stack.push(EditorAction(action_type="a", data={}))
        stack.push(EditorAction(action_type="b", data={}))
        stack.undo()

        stack.push(EditorAction(action_type="c", data={}))
        assert stack.can_redo() is False

    def test_max_size(self):
        stack = UndoStack(max_size=3)
        for i in range(5):
            stack.push(EditorAction(action_type=f"action_{i}", data={}))

        # Only last 3 should remain
        count = 0
        while stack.can_undo():
            stack.undo()
            count += 1

        assert count == 3

    def test_clear(self):
        stack = UndoStack()
        stack.push(EditorAction(action_type="a", data={}))
        stack.clear()

        assert stack.can_undo() is False
        assert stack.can_redo() is False


class TestBlueprintGraph:
    """Tests for BlueprintGraph class."""

    def test_create_graph(self):
        graph = BlueprintGraph(name="TestGraph")
        assert graph.name == "TestGraph"
        assert graph.id is not None

    def test_add_node(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)

        assert node.id in graph.nodes
        assert node.id in graph.entry_points

    def test_remove_node(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)

        removed = graph.remove_node(node.id)
        assert removed == node
        assert node.id not in graph.nodes

    def test_get_node(self):
        graph = BlueprintGraph()
        node = BeginPlayNode()
        graph.add_node(node)

        retrieved = graph.get_node(node.id)
        assert retrieved == node

    def test_add_connection(self):
        graph = BlueprintGraph()

        node1 = BeginPlayNode()
        node2 = PrintStringNode()
        graph.add_node(node1)
        graph.add_node(node2)

        # Connect execution pins
        out_pin = node1.output_pins["Out"]
        in_pin = node2.input_pins["In"]

        conn = Connection(
            id="conn_1",
            source_node_id=node1.id,
            source_pin_id=out_pin.id,
            target_node_id=node2.id,
            target_pin_id=in_pin.id
        )

        result = graph.add_connection(conn)
        assert result is not None
        assert conn.id in graph.connections

    def test_remove_connection(self):
        graph = BlueprintGraph()

        node1 = BeginPlayNode()
        node2 = PrintStringNode()
        graph.add_node(node1)
        graph.add_node(node2)

        conn = Connection(
            id="conn_1",
            source_node_id=node1.id,
            source_pin_id=node1.output_pins["Out"].id,
            target_node_id=node2.id,
            target_pin_id=node2.input_pins["In"].id
        )
        graph.add_connection(conn)

        removed = graph.remove_connection(conn.id)
        assert removed == conn
        assert conn.id not in graph.connections

    def test_remove_node_removes_connections(self):
        graph = BlueprintGraph()

        node1 = BeginPlayNode()
        node2 = PrintStringNode()
        graph.add_node(node1)
        graph.add_node(node2)

        conn = Connection(
            id="conn_1",
            source_node_id=node1.id,
            source_pin_id=node1.output_pins["Out"].id,
            target_node_id=node2.id,
            target_pin_id=node2.input_pins["In"].id
        )
        graph.add_connection(conn)

        graph.remove_node(node1.id)
        assert conn.id not in graph.connections

    def test_get_connections_for_node(self):
        graph = BlueprintGraph()

        node1 = BeginPlayNode()
        node2 = PrintStringNode()
        node3 = PrintStringNode()
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)

        conn1 = Connection(
            id="conn_1",
            source_node_id=node1.id,
            source_pin_id=node1.output_pins["Out"].id,
            target_node_id=node2.id,
            target_pin_id=node2.input_pins["In"].id
        )
        graph.add_connection(conn1)

        conns = graph.get_connections_for_node(node1.id)
        assert len(conns) == 1

    def test_get_outgoing_connections(self):
        graph = BlueprintGraph()

        node1 = BeginPlayNode()
        node2 = PrintStringNode()
        graph.add_node(node1)
        graph.add_node(node2)

        conn = Connection(
            id="conn_1",
            source_node_id=node1.id,
            source_pin_id=node1.output_pins["Out"].id,
            target_node_id=node2.id,
            target_pin_id=node2.input_pins["In"].id
        )
        graph.add_connection(conn)

        outgoing = graph.get_outgoing_connections(node1.id)
        assert len(outgoing) == 1
        assert outgoing[0].target_node_id == node2.id

    def test_get_incoming_connections(self):
        graph = BlueprintGraph()

        node1 = BeginPlayNode()
        node2 = PrintStringNode()
        graph.add_node(node1)
        graph.add_node(node2)

        conn = Connection(
            id="conn_1",
            source_node_id=node1.id,
            source_pin_id=node1.output_pins["Out"].id,
            target_node_id=node2.id,
            target_pin_id=node2.input_pins["In"].id
        )
        graph.add_connection(conn)

        incoming = graph.get_incoming_connections(node2.id)
        assert len(incoming) == 1
        assert incoming[0].source_node_id == node1.id

    def test_get_bounds_empty(self):
        graph = BlueprintGraph()
        bounds = graph.get_bounds()
        assert bounds == (0, 0, 0, 0)

    def test_get_bounds_with_nodes(self):
        graph = BlueprintGraph()

        node1 = BeginPlayNode(position=(0, 0))
        node2 = PrintStringNode(position=(500, 300))
        graph.add_node(node1)
        graph.add_node(node2)

        min_x, min_y, max_x, max_y = graph.get_bounds()
        assert min_x == 0
        assert min_y == 0
        assert max_x > 500

    def test_validate(self):
        graph = BlueprintGraph()
        errors = graph.validate()
        assert isinstance(errors, list)


class TestGraphEditor:
    """Tests for GraphEditor class."""

    def test_create_editor(self):
        editor = GraphEditor()
        assert editor.graph is not None
        assert editor.view is not None

    def test_create_editor_with_graph(self):
        graph = BlueprintGraph(name="MyGraph")
        editor = GraphEditor(graph=graph)
        assert editor.graph == graph

    def test_pan(self):
        editor = GraphEditor()
        editor.pan(100, 50)

        assert editor.view.pan_x == 100
        assert editor.view.pan_y == 50

    def test_zoom(self):
        editor = GraphEditor()
        editor.zoom(2.0, center_x=0, center_y=0)

        assert editor.view.zoom == 2.0

    def test_create_node(self):
        editor = GraphEditor()
        editor.snap_to_grid = False  # Disable grid snapping for exact position
        node = editor.create_node(BeginPlayNode, position=(100, 100))

        assert node.id in editor.graph.nodes
        assert node.position == (100, 100)

    def test_create_node_snaps_to_grid(self):
        editor = GraphEditor()
        editor.snap_to_grid = True
        editor.grid_size = 16

        node = editor.create_node(BeginPlayNode, position=(10, 25))

        # Should snap to grid
        assert node.position[0] % 16 == 0
        assert node.position[1] % 16 == 0

    def test_select_node(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(0, 0))

        editor.select_node(node.id)

        assert node.id in editor.selected_nodes

    def test_select_node_replace(self):
        editor = GraphEditor()
        node1 = editor.create_node(BeginPlayNode, position=(0, 0))
        node2 = editor.create_node(PrintStringNode, position=(100, 0))

        editor.select_node(node1.id)
        editor.select_node(node2.id, mode=SelectionMode.REPLACE)

        assert node2.id in editor.selected_nodes
        assert node1.id not in editor.selected_nodes

    def test_select_node_add(self):
        editor = GraphEditor()
        node1 = editor.create_node(BeginPlayNode, position=(0, 0))
        node2 = editor.create_node(PrintStringNode, position=(100, 0))

        editor.select_node(node1.id)
        editor.select_node(node2.id, mode=SelectionMode.ADD)

        assert node1.id in editor.selected_nodes
        assert node2.id in editor.selected_nodes

    def test_select_node_toggle(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(0, 0))

        editor.select_node(node.id, mode=SelectionMode.TOGGLE)
        assert node.id in editor.selected_nodes

        editor.select_node(node.id, mode=SelectionMode.TOGGLE)
        assert node.id not in editor.selected_nodes

    def test_select_all(self):
        editor = GraphEditor()
        node1 = editor.create_node(BeginPlayNode, position=(0, 0))
        node2 = editor.create_node(PrintStringNode, position=(100, 0))

        editor.select_all()

        assert len(editor.selected_nodes) == 2

    def test_clear_selection(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(0, 0))
        editor.select_node(node.id)

        editor.clear_selection()

        assert len(editor.selected_nodes) == 0

    def test_get_selection(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(0, 0))
        editor.select_node(node.id)

        selection = editor.get_selection()
        assert len(selection) == 1
        assert selection[0] == node

    def test_delete_selected(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(0, 0))
        editor.select_node(node.id)

        editor.delete_selected()

        assert node.id not in editor.graph.nodes
        assert len(editor.selected_nodes) == 0

    def test_move_selected(self):
        editor = GraphEditor()
        editor.snap_to_grid = False
        node = editor.create_node(BeginPlayNode, position=(100, 100))
        editor.select_node(node.id)

        editor.move_selected(50, 25)

        assert node.position == (150, 125)

    def test_duplicate_selected(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(0, 0))
        editor.select_node(node.id)

        new_nodes = editor.duplicate_selected()

        assert len(new_nodes) == 1
        assert new_nodes[0].id != node.id
        assert len(editor.graph.nodes) == 2


class TestGraphEditorConnections:
    """Tests for connection operations in GraphEditor."""

    def test_begin_connection(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(0, 0))
        pin_id = node.output_pins["Out"].id

        editor.begin_connection(pin_id)

        assert editor.is_connecting is True
        assert editor.connection_start_pin == pin_id

    def test_cancel_connection(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(0, 0))

        editor.begin_connection(node.output_pins["Out"].id)
        editor.cancel_connection()

        assert editor.is_connecting is False
        assert editor.connection_start_pin is None


class TestGraphEditorClipboard:
    """Tests for clipboard operations."""

    def test_copy(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(100, 100))
        editor.select_node(node.id)

        editor.copy()

        assert editor.clipboard is not None
        assert len(editor.clipboard.nodes) == 1

    def test_paste(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(100, 100))
        editor.select_node(node.id)
        editor.copy()

        editor.clear_selection()
        pasted = editor.paste(position=(200, 200))

        assert len(pasted) == 1
        assert pasted[0].id != node.id
        assert len(editor.graph.nodes) == 2

    def test_cut(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(100, 100))
        editor.select_node(node.id)

        editor.cut()

        assert editor.clipboard is not None
        assert node.id not in editor.graph.nodes


class TestGraphEditorUndoRedo:
    """Tests for undo/redo operations."""

    def test_undo_create_node(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(0, 0))

        assert editor.undo_stack.can_undo() is True

    def test_undo_delete(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(0, 0))
        node_id = node.id
        editor.select_node(node_id)
        editor.delete_selected()

        assert editor.undo_stack.can_undo() is True


class TestGraphEditorNavigation:
    """Tests for navigation functions."""

    def test_zoom_to_fit(self):
        editor = GraphEditor()
        editor.create_node(BeginPlayNode, position=(0, 0))
        editor.create_node(PrintStringNode, position=(500, 300))

        editor.zoom_to_fit()
        # Should not raise errors

    def test_center_on_node(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(1000, 1000))

        editor.center_on_node(node.id)
        # View should be adjusted

    def test_center_on_selection(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(500, 500))
        editor.select_node(node.id)

        editor.center_on_selection()
        # Should not raise errors


class TestGraphEditorAlignment:
    """Tests for node alignment."""

    def test_align_left(self):
        editor = GraphEditor()
        node1 = editor.create_node(BeginPlayNode, position=(100, 0))
        node2 = editor.create_node(PrintStringNode, position=(200, 100))
        editor.select_node(node1.id, mode=SelectionMode.ADD)
        editor.select_node(node2.id, mode=SelectionMode.ADD)

        editor.align_selected("left")

        assert node1.position[0] == node2.position[0]

    def test_align_top(self):
        editor = GraphEditor()
        node1 = editor.create_node(BeginPlayNode, position=(0, 100))
        node2 = editor.create_node(PrintStringNode, position=(100, 200))
        editor.select_node(node1.id, mode=SelectionMode.ADD)
        editor.select_node(node2.id, mode=SelectionMode.ADD)

        editor.align_selected("top")

        assert node1.position[1] == node2.position[1]


class TestGraphEditorMinimap:
    """Tests for minimap functionality."""

    def test_get_minimap_data(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(0, 0))

        data = editor.get_minimap_data()

        assert "bounds" in data
        assert "nodes" in data
        assert "view" in data
        assert len(data["nodes"]) == 1

    def test_minimap_click(self):
        editor = GraphEditor()
        editor.create_node(BeginPlayNode, position=(500, 500))

        editor.minimap_click(50, 50)
        # View should be updated


class TestGraphEditorUtilities:
    """Tests for utility functions."""

    def test_find_node_at_position(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(100, 100))

        found = editor.find_node_at_position(150, 150)
        assert found == node

    def test_find_node_at_position_not_found(self):
        editor = GraphEditor()
        editor.create_node(BeginPlayNode, position=(100, 100))

        found = editor.find_node_at_position(500, 500)
        assert found is None

    def test_get_statistics(self):
        editor = GraphEditor()
        editor.create_node(BeginPlayNode, position=(0, 0))
        editor.create_node(BranchNode, position=(100, 0))

        stats = editor.get_statistics()

        assert stats["node_count"] == 2
        assert "categories" in stats

    def test_is_selected(self):
        editor = GraphEditor()
        node = editor.create_node(BeginPlayNode, position=(0, 0))

        assert editor.is_selected(node.id) is False

        editor.select_node(node.id)
        assert editor.is_selected(node.id) is True

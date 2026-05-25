"""
Tests for history view functionality.
"""
import pytest
import time

from engine.tooling.undo.history_view import (
    HistoryView,
    HistoryNode,
    HistoryBranch,
    HistoryNavigator,
)


class TestHistoryNode:
    """Tests for HistoryNode."""

    def test_node_creation(self):
        """Test creating a history node."""
        node = HistoryNode(
            id="abc123",
            name="Test Change",
            timestamp=time.time(),
        )

        assert node.id == "abc123"
        assert node.name == "Test Change"
        assert node.is_current is False

    def test_has_children(self):
        """Test has_children property."""
        node = HistoryNode(
            id="1",
            name="Test",
            timestamp=time.time(),
        )
        assert node.has_children is False

        node.children_ids.append("2")
        assert node.has_children is True

    def test_is_branch_point(self):
        """Test is_branch_point property."""
        node = HistoryNode(
            id="1",
            name="Test",
            timestamp=time.time(),
            children_ids=["2"],
        )
        assert node.is_branch_point is False

        node.children_ids.append("3")
        assert node.is_branch_point is True

    def test_age(self):
        """Test age calculation."""
        old_time = time.time() - 10
        node = HistoryNode(
            id="1",
            name="Test",
            timestamp=old_time,
        )

        assert node.age() >= 10


class TestHistoryBranch:
    """Tests for HistoryBranch."""

    def test_branch_creation(self):
        """Test creating a history branch."""
        branch = HistoryBranch(
            id="main",
            name="Main",
            head_id="abc123",
        )

        assert branch.id == "main"
        assert branch.name == "Main"
        assert branch.head_id == "abc123"

    def test_is_empty(self):
        """Test is_empty property."""
        branch = HistoryBranch(
            id="test",
            name="Test",
            head_id="",
            node_count=0,
        )
        assert branch.is_empty is True

        branch.node_count = 1
        assert branch.is_empty is False


class TestHistoryView:
    """Tests for HistoryView."""

    def setup_method(self):
        """Create fresh history view for each test."""
        self.view = HistoryView(enable_branching=False)

    def test_view_initialization(self):
        """Test HistoryView initializes correctly."""
        assert self.view.node_count == 0
        assert self.view.current_node is None
        assert self.view.branch_count == 1  # main branch

    def test_add_entry(self):
        """Test adding a history entry."""
        node = self.view.add_entry("Test Change")

        assert node is not None
        assert node.name == "Test Change"
        assert node.is_current is True
        assert self.view.node_count == 1
        assert self.view.current_node is node

    def test_add_multiple_entries(self):
        """Test adding multiple entries."""
        node1 = self.view.add_entry("Change 1")
        node2 = self.view.add_entry("Change 2")
        node3 = self.view.add_entry("Change 3")

        assert self.view.node_count == 3
        assert self.view.current_node is node3
        assert node1.is_current is False
        assert node2.is_current is False
        assert node3.is_current is True

    def test_navigate_to(self):
        """Test navigating to a specific node."""
        node1 = self.view.add_entry("Change 1")
        node2 = self.view.add_entry("Change 2")

        result = self.view.navigate_to(node1.id)

        assert result is True
        assert self.view.current_node is node1
        assert node1.is_current is True
        assert node2.is_current is False

    def test_navigate_to_invalid(self):
        """Test navigating to invalid node."""
        result = self.view.navigate_to("nonexistent")
        assert result is False

    def test_undo(self):
        """Test undo navigation."""
        node1 = self.view.add_entry("Change 1")
        node2 = self.view.add_entry("Change 2")

        result = self.view.undo()

        assert result is node1
        assert self.view.current_node is node1

    def test_undo_at_root(self):
        """Test undo at root returns None."""
        self.view.add_entry("Change 1")

        self.view.undo()  # Back to root
        result = self.view.undo()

        assert result is None

    def test_redo(self):
        """Test redo navigation."""
        node1 = self.view.add_entry("Change 1")
        node2 = self.view.add_entry("Change 2")

        self.view.undo()  # Go back to node1
        result = self.view.redo()

        assert result is node2
        assert self.view.current_node is node2

    def test_redo_at_leaf(self):
        """Test redo at leaf returns None."""
        self.view.add_entry("Change 1")

        result = self.view.redo()
        assert result is None

    def test_get_path_to_root(self):
        """Test getting path to root."""
        node1 = self.view.add_entry("Change 1")
        node2 = self.view.add_entry("Change 2")
        node3 = self.view.add_entry("Change 3")

        path = self.view.get_path_to_root()

        assert len(path) == 3
        assert path[0] is node3
        assert path[1] is node2
        assert path[2] is node1

    def test_get_linear_history(self):
        """Test getting linear history."""
        for i in range(10):
            self.view.add_entry(f"Change {i}")

        history = self.view.get_linear_history(limit=5)

        assert len(history) == 5
        assert history[0].name == "Change 9"

    def test_clear(self):
        """Test clearing history."""
        self.view.add_entry("Change 1")
        self.view.add_entry("Change 2")

        self.view.clear()

        assert self.view.node_count == 0
        assert self.view.current_node is None

    def test_render_tree(self):
        """Test rendering history tree."""
        self.view.add_entry("Change 1")
        self.view.add_entry("Change 2")

        tree = self.view.render_tree()

        assert isinstance(tree, str)
        assert "Change 1" in tree
        assert "Change 2" in tree


class TestHistoryViewBranching:
    """Tests for HistoryView with branching enabled."""

    def setup_method(self):
        """Create history view with branching."""
        self.view = HistoryView(enable_branching=True)

    def test_branching_enabled(self):
        """Test branching is enabled."""
        assert self.view.branching_enabled is True

    def test_create_branch(self):
        """Test creating a branch."""
        self.view.add_entry("Change 1")

        branch = self.view.create_branch("feature")

        assert branch is not None
        assert branch.name == "feature"
        assert self.view.branch_count == 2

    def test_switch_branch(self):
        """Test switching branches."""
        self.view.add_entry("Change 1")
        self.view.create_branch("feature")
        self.view.add_entry("Feature change")

        result = self.view.switch_branch("main")

        assert result is True
        assert self.view.current_branch_name == "main"

    def test_switch_nonexistent_branch(self):
        """Test switching to non-existent branch."""
        result = self.view.switch_branch("nonexistent")
        assert result is False

    def test_branch_on_divergence(self):
        """Test automatic branch creation on divergence."""
        node1 = self.view.add_entry("Change 1")
        node2 = self.view.add_entry("Change 2")

        # Go back and make different change
        self.view.undo()
        node3 = self.view.add_entry("Alternative change")

        # Should have created a branch
        assert node1.has_children is True

    def test_get_branch_points(self):
        """Test getting branch points."""
        node1 = self.view.add_entry("Change 1")
        self.view.add_entry("Change 2")

        self.view.undo()
        self.view.add_entry("Alternative")

        points = self.view.get_branch_points()

        assert len(points) >= 1


class TestHistoryNavigator:
    """Tests for HistoryNavigator."""

    def setup_method(self):
        """Create history view and navigator."""
        self.view = HistoryView()
        self.navigated = []
        self.navigator = HistoryNavigator(
            self.view,
            on_navigate=lambda n: self.navigated.append(n),
        )

    def test_navigator_initialization(self):
        """Test HistoryNavigator initializes correctly."""
        assert self.navigator.can_undo is False
        assert self.navigator.can_redo is False
        assert self.navigator.redo_options == 0

    def test_can_undo(self):
        """Test can_undo property."""
        self.view.add_entry("Change 1")
        self.view.add_entry("Change 2")

        assert self.navigator.can_undo is True

        self.navigator.undo()
        assert self.navigator.can_undo is False

    def test_can_redo(self):
        """Test can_redo property."""
        self.view.add_entry("Change 1")
        self.view.add_entry("Change 2")

        assert self.navigator.can_redo is False

        self.navigator.undo()
        assert self.navigator.can_redo is True

    def test_undo_with_callback(self):
        """Test undo calls navigation callback."""
        self.view.add_entry("Change 1")
        self.view.add_entry("Change 2")

        self.navigator.undo()

        assert len(self.navigated) == 1

    def test_redo_with_callback(self):
        """Test redo calls navigation callback."""
        self.view.add_entry("Change 1")
        self.view.add_entry("Change 2")

        self.navigator.undo()
        self.navigated.clear()

        self.navigator.redo()

        assert len(self.navigated) == 1

    def test_goto(self):
        """Test goto specific node."""
        node1 = self.view.add_entry("Change 1")
        self.view.add_entry("Change 2")
        self.view.add_entry("Change 3")

        result = self.navigator.goto(node1.id)

        assert result is True
        assert self.view.current_node is node1
        assert len(self.navigated) == 1

    def test_step_back(self):
        """Test stepping back multiple levels."""
        self.view.add_entry("Change 1")
        self.view.add_entry("Change 2")
        self.view.add_entry("Change 3")

        steps = self.navigator.step_back(2)

        assert steps == 2
        assert self.view.current_node.name == "Change 1"

    def test_step_forward(self):
        """Test stepping forward multiple levels."""
        self.view.add_entry("Change 1")
        self.view.add_entry("Change 2")
        self.view.add_entry("Change 3")

        self.navigator.step_back(3)

        steps = self.navigator.step_forward(2)

        assert steps == 2

    def test_redo_options(self):
        """Test redo_options property."""
        view = HistoryView(enable_branching=True)
        navigator = HistoryNavigator(view)

        view.add_entry("Change 1")
        view.add_entry("Change 2a")
        view.undo()
        view.add_entry("Change 2b")
        view.undo()

        # Should have 2 options at branch point
        assert navigator.redo_options == 2

"""
History View - Undo history visualization with branching support.

Provides a view into the undo/redo history, supporting both linear
and branching (tree) undo models.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4


@dataclass
class HistoryNode:
    """A node in the history tree."""

    id: str
    name: str
    timestamp: float
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_current: bool = False

    @property
    def has_children(self) -> bool:
        """Check if this node has children."""
        return bool(self.children_ids)

    @property
    def is_branch_point(self) -> bool:
        """Check if this node has multiple children (branch point)."""
        return len(self.children_ids) > 1

    def age(self) -> float:
        """Get age in seconds."""
        return time.time() - self.timestamp


@dataclass
class HistoryBranch:
    """Represents a branch in the history tree."""

    id: str
    name: str
    head_id: str  # ID of the latest node in this branch
    created_at: float = field(default_factory=time.time)
    node_count: int = 0

    @property
    def is_empty(self) -> bool:
        """Check if branch is empty."""
        return self.node_count == 0


class HistoryView:
    """
    View into the undo/redo history.

    Supports both linear and branching history models.
    In branching mode, alternative histories are preserved when
    changes are made after undo operations.
    """

    def __init__(self, enable_branching: bool = False):
        """
        Initialize the history view.

        Args:
            enable_branching: Enable tree-based history (vs linear).
        """
        self._branching_enabled = enable_branching
        self._nodes: Dict[str, HistoryNode] = {}
        self._branches: Dict[str, HistoryBranch] = {}
        self._current_branch: str = "main"
        self._current_node: Optional[str] = None
        self._root_id: Optional[str] = None

        # Initialize main branch
        self._branches["main"] = HistoryBranch(
            id="main",
            name="Main",
            head_id="",
        )

    @property
    def branching_enabled(self) -> bool:
        """Check if branching is enabled."""
        return self._branching_enabled

    @property
    def current_node(self) -> Optional[HistoryNode]:
        """Get the current history node."""
        if self._current_node:
            return self._nodes.get(self._current_node)
        return None

    @property
    def current_branch_name(self) -> str:
        """Get the current branch name."""
        return self._current_branch

    @property
    def node_count(self) -> int:
        """Total number of history nodes."""
        return len(self._nodes)

    @property
    def branch_count(self) -> int:
        """Number of branches."""
        return len(self._branches)

    def add_entry(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HistoryNode:
        """
        Add a new history entry.

        Args:
            name: Name/description of the entry.
            metadata: Optional metadata.

        Returns:
            The created HistoryNode.
        """
        node_id = str(uuid4())[:8]

        node = HistoryNode(
            id=node_id,
            name=name,
            timestamp=time.time(),
            parent_id=self._current_node,
            metadata=metadata or {},
        )

        # Handle branching if making change after undo
        if self._branching_enabled and self._current_node:
            current = self._nodes.get(self._current_node)
            if current and current.children_ids:
                # We're creating a branch
                branch_name = f"branch_{len(self._branches)}"
                self._create_branch(branch_name, node_id)

        # Update parent's children
        if self._current_node and self._current_node in self._nodes:
            self._nodes[self._current_node].children_ids.append(node_id)
            self._nodes[self._current_node].is_current = False

        # Add node
        self._nodes[node_id] = node
        node.is_current = True

        # Update root
        if self._root_id is None:
            self._root_id = node_id

        # Update current
        self._current_node = node_id

        # Update branch head
        branch = self._branches.get(self._current_branch)
        if branch:
            branch.head_id = node_id
            branch.node_count += 1

        return node

    def navigate_to(self, node_id: str) -> bool:
        """
        Navigate to a specific history node.

        Args:
            node_id: ID of node to navigate to.

        Returns:
            True if navigation succeeded.
        """
        if node_id not in self._nodes:
            return False

        # Clear current flag on old node
        if self._current_node:
            old_node = self._nodes.get(self._current_node)
            if old_node:
                old_node.is_current = False

        # Set new current
        self._current_node = node_id
        self._nodes[node_id].is_current = True

        return True

    def undo(self) -> Optional[HistoryNode]:
        """
        Move to parent node (undo).

        Returns:
            The new current node, or None if at root.
        """
        if not self._current_node:
            return None

        current = self._nodes.get(self._current_node)
        if not current or not current.parent_id:
            return None

        self.navigate_to(current.parent_id)
        return self._nodes.get(current.parent_id)

    def redo(self, branch_index: int = 0) -> Optional[HistoryNode]:
        """
        Move to child node (redo).

        Args:
            branch_index: Which child to follow (for branching).

        Returns:
            The new current node, or None if at leaf.
        """
        if not self._current_node:
            return None

        current = self._nodes.get(self._current_node)
        if not current or not current.children_ids:
            return None

        if branch_index >= len(current.children_ids):
            branch_index = 0

        child_id = current.children_ids[branch_index]
        self.navigate_to(child_id)
        return self._nodes.get(child_id)

    def get_path_to_root(self) -> List[HistoryNode]:
        """
        Get path from current node to root.

        Returns:
            List of nodes from current to root.
        """
        path = []
        node_id = self._current_node

        while node_id and node_id in self._nodes:
            node = self._nodes[node_id]
            path.append(node)
            node_id = node.parent_id

        return path

    def get_linear_history(self, limit: int = 50) -> List[HistoryNode]:
        """
        Get linear history (current branch only).

        Args:
            limit: Maximum entries to return.

        Returns:
            List of nodes from newest to oldest.
        """
        path = self.get_path_to_root()
        return path[:limit]

    def get_branch_points(self) -> List[HistoryNode]:
        """Get all nodes that are branch points."""
        return [n for n in self._nodes.values() if n.is_branch_point]

    def get_branches_at(self, node_id: str) -> List[str]:
        """Get names of branches available at a node."""
        node = self._nodes.get(node_id)
        if not node or not node.is_branch_point:
            return []

        # Find which branches have this node in their history
        branches = []
        for branch in self._branches.values():
            if self._is_in_branch(node_id, branch.head_id):
                branches.append(branch.name)

        return branches

    def switch_branch(self, branch_name: str) -> bool:
        """
        Switch to a different branch.

        Args:
            branch_name: Name of branch to switch to.

        Returns:
            True if switch succeeded.
        """
        if branch_name not in self._branches:
            return False

        branch = self._branches[branch_name]
        if branch.head_id:
            self.navigate_to(branch.head_id)
        self._current_branch = branch_name

        return True

    def create_branch(self, name: str) -> Optional[HistoryBranch]:
        """
        Create a new branch at the current position.

        Args:
            name: Name for the new branch.

        Returns:
            The new branch or None if failed.
        """
        if name in self._branches:
            return None

        return self._create_branch(name, self._current_node or "")

    def _create_branch(self, name: str, head_id: str) -> HistoryBranch:
        """Internal branch creation."""
        branch = HistoryBranch(
            id=name,
            name=name,
            head_id=head_id,
            node_count=self._count_ancestors(head_id) + 1,
        )
        self._branches[name] = branch
        self._current_branch = name
        return branch

    def _is_in_branch(self, ancestor_id: str, descendant_id: str) -> bool:
        """Check if ancestor is in the history of descendant."""
        node_id = descendant_id
        while node_id:
            if node_id == ancestor_id:
                return True
            node = self._nodes.get(node_id)
            if not node:
                break
            node_id = node.parent_id
        return False

    def _count_ancestors(self, node_id: str) -> int:
        """Count ancestor nodes."""
        count = 0
        node = self._nodes.get(node_id)
        while node and node.parent_id:
            count += 1
            node = self._nodes.get(node.parent_id)
        return count

    def render_tree(self, max_depth: int = 20) -> str:
        """
        Render the history tree as ASCII art.

        Args:
            max_depth: Maximum depth to render.

        Returns:
            ASCII representation of the tree.
        """
        if not self._root_id:
            return "(empty history)"

        lines = []
        self._render_node(self._root_id, "", True, lines, 0, max_depth)
        return "\n".join(lines)

    def _render_node(
        self,
        node_id: str,
        prefix: str,
        is_last: bool,
        lines: List[str],
        depth: int,
        max_depth: int,
    ) -> None:
        """Recursively render a node and its children."""
        if depth > max_depth:
            return

        node = self._nodes.get(node_id)
        if not node:
            return

        # Build line
        connector = "\\-" if is_last else "+-"
        marker = "*" if node.is_current else "o"
        line = f"{prefix}{connector} {marker} {node.name}"

        lines.append(line)

        # Render children
        child_prefix = prefix + ("   " if is_last else "|  ")
        for i, child_id in enumerate(node.children_ids):
            is_last_child = i == len(node.children_ids) - 1
            self._render_node(
                child_id,
                child_prefix,
                is_last_child,
                lines,
                depth + 1,
                max_depth,
            )

    def clear(self) -> None:
        """Clear all history."""
        self._nodes.clear()
        self._branches.clear()
        self._current_node = None
        self._root_id = None

        # Recreate main branch
        self._branches["main"] = HistoryBranch(
            id="main",
            name="Main",
            head_id="",
        )
        self._current_branch = "main"


class HistoryNavigator:
    """
    Navigator for traversing undo history.

    Provides methods for stepping through history with
    callbacks for state changes.
    """

    def __init__(
        self,
        view: HistoryView,
        on_navigate: Optional[Callable[[HistoryNode], None]] = None,
    ):
        """
        Initialize the navigator.

        Args:
            view: HistoryView to navigate.
            on_navigate: Callback when navigation occurs.
        """
        self._view = view
        self._on_navigate = on_navigate

    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        current = self._view.current_node
        return current is not None and current.parent_id is not None

    @property
    def can_redo(self) -> bool:
        """Check if redo is available."""
        current = self._view.current_node
        return current is not None and bool(current.children_ids)

    @property
    def redo_options(self) -> int:
        """Number of redo options (branches)."""
        current = self._view.current_node
        if current:
            return len(current.children_ids)
        return 0

    def undo(self) -> Optional[HistoryNode]:
        """Perform undo and notify."""
        node = self._view.undo()
        if node and self._on_navigate:
            self._on_navigate(node)
        return node

    def redo(self, branch: int = 0) -> Optional[HistoryNode]:
        """Perform redo and notify."""
        node = self._view.redo(branch)
        if node and self._on_navigate:
            self._on_navigate(node)
        return node

    def goto(self, node_id: str) -> bool:
        """Navigate to a specific node."""
        if self._view.navigate_to(node_id):
            node = self._view.current_node
            if node and self._on_navigate:
                self._on_navigate(node)
            return True
        return False

    def step_back(self, steps: int = 1) -> int:
        """
        Step back multiple levels.

        Args:
            steps: Number of steps to go back.

        Returns:
            Actual number of steps taken.
        """
        taken = 0
        for _ in range(steps):
            if self.undo():
                taken += 1
            else:
                break
        return taken

    def step_forward(self, steps: int = 1) -> int:
        """
        Step forward multiple levels.

        Args:
            steps: Number of steps to go forward.

        Returns:
            Actual number of steps taken.
        """
        taken = 0
        for _ in range(steps):
            if self.redo():
                taken += 1
            else:
                break
        return taken


__all__ = [
    "HistoryNode",
    "HistoryBranch",
    "HistoryView",
    "HistoryNavigator",
]

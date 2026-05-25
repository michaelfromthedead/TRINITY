"""
Scene Hierarchy System - Tree-based scene organization.

Provides:
- Hierarchical scene tree with parent-child relationships
- Drag-drop operations for reparenting
- Grouping and folders for organization
- Layer assignment and filtering
- Search and filtering capabilities

All hierarchy operations integrate with Foundation Tracker for undo/redo.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, Optional

from .placement import Vector3, Quaternion, Transform, editor, track_changes
from foundation.tracker import tracker


# =============================================================================
# Enums
# =============================================================================

class NodeType(Enum):
    """Types of hierarchy nodes."""
    OBJECT = auto()  # Scene object
    GROUP = auto()  # Logical grouping
    FOLDER = auto()  # Organizational folder
    PREFAB_ROOT = auto()  # Root of a prefab instance
    LAYER_ROOT = auto()  # Virtual layer root


class DragDropType(Enum):
    """Types of drag-drop operations."""
    REPARENT = auto()  # Move under new parent
    REORDER = auto()  # Change sibling order
    COPY = auto()  # Copy to new location
    LINK = auto()  # Create link/reference


class FilterMode(Enum):
    """Hierarchy filter modes."""
    ALL = auto()
    VISIBLE_ONLY = auto()
    SELECTED_ONLY = auto()
    UNLOCKED_ONLY = auto()
    BY_TYPE = auto()
    BY_LAYER = auto()
    BY_NAME = auto()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(slots=True)
class HierarchyFilter:
    """Filter settings for hierarchy display."""
    mode: FilterMode = FilterMode.ALL
    search_text: str = ""
    type_filter: Optional[str] = None
    layer_mask: int = 0xFFFFFFFF
    include_children_of_match: bool = True
    case_sensitive: bool = False


@dataclass(slots=True)
class DragDropOperation:
    """Represents a drag-drop operation."""
    source_ids: list[str]
    target_id: Optional[str]
    operation_type: DragDropType
    insert_index: int = -1  # -1 means append
    completed: bool = False
    error_message: Optional[str] = None


@dataclass(slots=True)
class NodeState:
    """Visual/interaction state of a hierarchy node."""
    expanded: bool = True
    selected: bool = False
    visible: bool = True
    locked: bool = False
    highlighted: bool = False


# =============================================================================
# Hierarchy Node
# =============================================================================

@editor
class HierarchyNode:
    """
    A node in the scene hierarchy tree.

    Represents either a scene object, group, or folder.
    Supports parent-child relationships and sibling ordering.
    """

    __slots__ = (
        "_id",
        "_name",
        "_node_type",
        "_parent",
        "_children",
        "_state",
        "_layer_id",
        "_object_id",
        "_metadata",
        "_local_transform",
        "__weakref__",
    )

    def __init__(
        self,
        name: str,
        node_type: NodeType = NodeType.OBJECT,
        object_id: Optional[str] = None,
    ):
        """
        Initialize a hierarchy node.

        Args:
            name: Display name of the node
            node_type: Type of node
            object_id: ID of associated scene object (if any)
        """
        self._id = str(uuid.uuid4())
        self._name = name
        self._node_type = node_type
        self._parent: Optional[HierarchyNode] = None
        self._children: list[HierarchyNode] = []
        self._state = NodeState()
        self._layer_id: Optional[str] = None
        self._object_id = object_id
        self._metadata: dict[str, Any] = {}
        self._local_transform = Transform()

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        old_name = self._name
        self._name = value
        tracker.mark_dirty(self, "_name", old_name, value)

    @property
    def node_type(self) -> NodeType:
        return self._node_type

    @property
    def parent(self) -> Optional[HierarchyNode]:
        return self._parent

    @property
    def children(self) -> list[HierarchyNode]:
        return self._children.copy()

    @property
    def state(self) -> NodeState:
        return self._state

    @property
    def layer_id(self) -> Optional[str]:
        return self._layer_id

    @layer_id.setter
    def layer_id(self, value: Optional[str]) -> None:
        old_id = self._layer_id
        self._layer_id = value
        tracker.mark_dirty(self, "_layer_id", old_id, value)

    @property
    def object_id(self) -> Optional[str]:
        return self._object_id

    @property
    def local_transform(self) -> Transform:
        return self._local_transform

    @local_transform.setter
    def local_transform(self, value: Transform) -> None:
        old_transform = self._local_transform
        self._local_transform = value
        tracker.mark_dirty(self, "_local_transform", old_transform, value)

    @property
    def is_leaf(self) -> bool:
        return len(self._children) == 0

    @property
    def is_root(self) -> bool:
        return self._parent is None

    @property
    def depth(self) -> int:
        """Get depth in hierarchy (root = 0)."""
        depth = 0
        node = self._parent
        while node:
            depth += 1
            node = node._parent
        return depth

    @property
    def path(self) -> str:
        """Get full path from root."""
        parts = [self._name]
        node = self._parent
        while node:
            parts.insert(0, node._name)
            node = node._parent
        return "/".join(parts)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self._metadata.get(key, default)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        old_metadata = self._metadata.copy()
        self._metadata[key] = value
        tracker.mark_dirty(self, "_metadata", old_metadata, self._metadata.copy())

    @track_changes
    def add_child(self, child: HierarchyNode, index: int = -1) -> None:
        """
        Add a child node.

        Args:
            child: Node to add as child
            index: Position to insert at (-1 for append)
        """
        if child._parent:
            child._parent.remove_child(child)

        child._parent = self

        if index < 0 or index >= len(self._children):
            self._children.append(child)
        else:
            self._children.insert(index, child)

        tracker.mark_dirty(self, "_children",
                          self._children[:-1] if index < 0 else None,
                          self._children.copy())

    @track_changes
    def remove_child(self, child: HierarchyNode) -> bool:
        """
        Remove a child node.

        Args:
            child: Node to remove

        Returns:
            True if removed, False if not found
        """
        if child not in self._children:
            return False

        old_children = self._children.copy()
        self._children.remove(child)
        child._parent = None

        tracker.mark_dirty(self, "_children", old_children, self._children.copy())
        return True

    def get_child_index(self, child: HierarchyNode) -> int:
        """Get index of child node."""
        try:
            return self._children.index(child)
        except ValueError:
            return -1

    @track_changes
    def reorder_child(self, child: HierarchyNode, new_index: int) -> bool:
        """
        Move child to new index.

        Args:
            child: Child node to move
            new_index: New position

        Returns:
            True if reordered, False if not found
        """
        if child not in self._children:
            return False

        old_children = self._children.copy()
        old_index = self._children.index(child)
        self._children.remove(child)

        if new_index < 0:
            new_index = len(self._children)
        elif new_index > len(self._children):
            new_index = len(self._children)

        self._children.insert(new_index, child)

        tracker.mark_dirty(self, "_children", old_children, self._children.copy())
        return True

    def find_child(self, name: str) -> Optional[HierarchyNode]:
        """Find direct child by name."""
        for child in self._children:
            if child._name == name:
                return child
        return None

    def find_descendant(self, path: str) -> Optional[HierarchyNode]:
        """Find descendant by relative path."""
        parts = path.split("/")
        current: Optional[HierarchyNode] = self

        for part in parts:
            if not part:
                continue
            if current is None:
                return None
            current = current.find_child(part)

        return current

    def get_ancestors(self) -> list[HierarchyNode]:
        """Get all ancestor nodes from parent to root."""
        ancestors = []
        node = self._parent
        while node:
            ancestors.append(node)
            node = node._parent
        return ancestors

    def get_descendants(self) -> list[HierarchyNode]:
        """Get all descendant nodes (depth-first)."""
        descendants = []
        for child in self._children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants

    def get_siblings(self) -> list[HierarchyNode]:
        """Get sibling nodes (excluding self)."""
        if not self._parent:
            return []
        return [c for c in self._parent._children if c is not self]

    def is_ancestor_of(self, node: HierarchyNode) -> bool:
        """Check if this node is an ancestor of another."""
        current = node._parent
        while current:
            if current is self:
                return True
            current = current._parent
        return False

    def is_descendant_of(self, node: HierarchyNode) -> bool:
        """Check if this node is a descendant of another."""
        return node.is_ancestor_of(self)

    def traverse(
        self,
        callback: Callable[[HierarchyNode], bool],
        depth_first: bool = True
    ) -> None:
        """
        Traverse tree calling callback on each node.

        Args:
            callback: Function to call, return False to skip children
            depth_first: Use depth-first if True, breadth-first if False
        """
        if depth_first:
            self._traverse_depth_first(callback)
        else:
            self._traverse_breadth_first(callback)

    def _traverse_depth_first(
        self,
        callback: Callable[[HierarchyNode], bool]
    ) -> None:
        """Depth-first traversal."""
        if not callback(self):
            return
        for child in self._children:
            child._traverse_depth_first(callback)

    def _traverse_breadth_first(
        self,
        callback: Callable[[HierarchyNode], bool]
    ) -> None:
        """Breadth-first traversal."""
        queue = [self]
        while queue:
            node = queue.pop(0)
            if callback(node):
                queue.extend(node._children)

    def __iter__(self) -> Iterator[HierarchyNode]:
        """Iterate over all descendants."""
        yield self
        for child in self._children:
            yield from child

    def __repr__(self) -> str:
        return f"HierarchyNode({self._name!r}, type={self._node_type.name}, children={len(self._children)})"


# =============================================================================
# Specialized Nodes
# =============================================================================

@editor
class HierarchyFolder(HierarchyNode):
    """
    A folder node for organizing the hierarchy.

    Folders are purely organizational and don't affect transforms.
    """

    __slots__ = ("_color", "_icon")

    def __init__(self, name: str):
        super().__init__(name, NodeType.FOLDER)
        self._color: tuple[float, float, float] = (0.8, 0.8, 0.8)
        self._icon: str = "folder"

    @property
    def color(self) -> tuple[float, float, float]:
        return self._color

    @color.setter
    def color(self, value: tuple[float, float, float]) -> None:
        old_color = self._color
        self._color = value
        tracker.mark_dirty(self, "_color", old_color, value)

    @property
    def icon(self) -> str:
        return self._icon

    @icon.setter
    def icon(self, value: str) -> None:
        old_icon = self._icon
        self._icon = value
        tracker.mark_dirty(self, "_icon", old_icon, value)


@editor
class HierarchyGroup(HierarchyNode):
    """
    A group node that affects child transforms.

    Groups have a pivot point and transform that affects all children.
    """

    __slots__ = ("_pivot_mode", "_bounds_min", "_bounds_max")

    def __init__(self, name: str):
        super().__init__(name, NodeType.GROUP)
        self._pivot_mode: str = "center"  # center, first_child, manual
        self._bounds_min = Vector3()
        self._bounds_max = Vector3()

    @property
    def pivot_mode(self) -> str:
        return self._pivot_mode

    @pivot_mode.setter
    def pivot_mode(self, value: str) -> None:
        old_mode = self._pivot_mode
        self._pivot_mode = value
        tracker.mark_dirty(self, "_pivot_mode", old_mode, value)

    def calculate_bounds(self) -> tuple[Vector3, Vector3]:
        """Calculate bounds encompassing all children."""
        if not self._children:
            return Vector3(), Vector3()

        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        for child in self._children:
            pos = child.local_transform.position
            min_x = min(min_x, pos.x)
            min_y = min(min_y, pos.y)
            min_z = min(min_z, pos.z)
            max_x = max(max_x, pos.x)
            max_y = max(max_y, pos.y)
            max_z = max(max_z, pos.z)

        self._bounds_min = Vector3(min_x, min_y, min_z)
        self._bounds_max = Vector3(max_x, max_y, max_z)

        return self._bounds_min, self._bounds_max

    def center_pivot(self) -> None:
        """Move pivot to center of children bounds."""
        bounds_min, bounds_max = self.calculate_bounds()
        center = Vector3(
            (bounds_min.x + bounds_max.x) / 2,
            (bounds_min.y + bounds_max.y) / 2,
            (bounds_min.z + bounds_max.z) / 2,
        )
        self.local_transform = Transform(position=center)


# =============================================================================
# Hierarchy Tree
# =============================================================================

@editor
class HierarchyTree:
    """
    Complete scene hierarchy tree.

    Manages the root node, selection, and tree-wide operations.
    """

    __slots__ = (
        "_root",
        "_selection",
        "_id_map",
        "_filter",
        "_callbacks",
        "_clipboard",
        "__weakref__",
    )

    def __init__(self):
        """Initialize hierarchy tree with root node."""
        self._root = HierarchyNode("Scene", NodeType.FOLDER)
        self._selection: list[str] = []
        self._id_map: dict[str, HierarchyNode] = {self._root.id: self._root}
        self._filter = HierarchyFilter()
        self._callbacks: dict[str, list[Callable]] = {
            "on_selection_change": [],
            "on_structure_change": [],
            "on_node_rename": [],
        }
        self._clipboard: list[HierarchyNode] = []

    @property
    def root(self) -> HierarchyNode:
        return self._root

    @property
    def selection(self) -> list[str]:
        return self._selection.copy()

    @property
    def filter(self) -> HierarchyFilter:
        return self._filter

    @filter.setter
    def filter(self, value: HierarchyFilter) -> None:
        self._filter = value

    def on(self, event: str, callback: Callable) -> None:
        """Register callback for hierarchy events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Unregister callback."""
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)

    def get_node(self, node_id: str) -> Optional[HierarchyNode]:
        """Get node by ID."""
        return self._id_map.get(node_id)

    def find_by_name(self, name: str) -> list[HierarchyNode]:
        """Find all nodes with matching name."""
        results = []
        for node in self._root:
            if node.name == name:
                results.append(node)
        return results

    def find_by_object_id(self, object_id: str) -> Optional[HierarchyNode]:
        """Find node by associated object ID."""
        for node in self._root:
            if node.object_id == object_id:
                return node
        return None

    @track_changes
    def add_node(
        self,
        node: HierarchyNode,
        parent_id: Optional[str] = None,
        index: int = -1
    ) -> bool:
        """
        Add a node to the tree.

        Args:
            node: Node to add
            parent_id: ID of parent (None for root)
            index: Position among siblings

        Returns:
            True if added successfully
        """
        parent = self._id_map.get(parent_id) if parent_id else self._root

        if not parent:
            return False

        parent.add_child(node, index)
        self._register_node(node)

        for callback in self._callbacks["on_structure_change"]:
            callback("add", node)

        return True

    def _register_node(self, node: HierarchyNode) -> None:
        """Register node and all descendants in ID map."""
        self._id_map[node.id] = node
        for child in node.children:
            self._register_node(child)

    def _unregister_node(self, node: HierarchyNode) -> None:
        """Unregister node and all descendants from ID map."""
        self._id_map.pop(node.id, None)
        for child in node.children:
            self._unregister_node(child)

    @track_changes
    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the tree.

        Args:
            node_id: ID of node to remove

        Returns:
            True if removed successfully
        """
        node = self._id_map.get(node_id)
        if not node or node is self._root:
            return False

        parent = node.parent
        if not parent:
            return False

        self._unregister_node(node)
        parent.remove_child(node)

        # Remove from selection
        if node_id in self._selection:
            self._selection.remove(node_id)

        for callback in self._callbacks["on_structure_change"]:
            callback("remove", node)

        return True

    @track_changes
    def reparent(
        self,
        node_id: str,
        new_parent_id: str,
        index: int = -1
    ) -> bool:
        """
        Move node to new parent.

        Args:
            node_id: ID of node to move
            new_parent_id: ID of new parent
            index: Position among new siblings

        Returns:
            True if reparented successfully
        """
        node = self._id_map.get(node_id)
        new_parent = self._id_map.get(new_parent_id)

        if not node or not new_parent:
            return False

        if node is self._root:
            return False

        # Prevent circular reference
        if node.is_ancestor_of(new_parent):
            return False

        new_parent.add_child(node, index)

        for callback in self._callbacks["on_structure_change"]:
            callback("reparent", node)

        return True

    @track_changes
    def group_selection(self, group_name: str = "Group") -> Optional[HierarchyGroup]:
        """
        Group selected nodes under a new group.

        Args:
            group_name: Name for the new group

        Returns:
            The created group, or None if grouping failed
        """
        if len(self._selection) < 1:
            return None

        nodes = [self._id_map.get(nid) for nid in self._selection]
        nodes = [n for n in nodes if n is not None]

        if not nodes:
            return None

        # Find common parent
        common_parent = nodes[0].parent
        for node in nodes[1:]:
            if node.parent != common_parent:
                common_parent = self._root
                break

        if not common_parent:
            common_parent = self._root

        # Create group
        group = HierarchyGroup(group_name)

        # Find insertion index (first selected node's position)
        first_index = common_parent.get_child_index(nodes[0])
        common_parent.add_child(group, first_index)
        self._id_map[group.id] = group

        # Move nodes to group
        for node in nodes:
            group.add_child(node)

        group.center_pivot()

        for callback in self._callbacks["on_structure_change"]:
            callback("group", group)

        return group

    @track_changes
    def ungroup(self, group_id: str) -> bool:
        """
        Dissolve a group, moving children to parent.

        Args:
            group_id: ID of group to dissolve

        Returns:
            True if ungrouped successfully
        """
        group = self._id_map.get(group_id)
        if not group or group.node_type != NodeType.GROUP:
            return False

        parent = group.parent
        if not parent:
            return False

        index = parent.get_child_index(group)
        children = group.children.copy()

        # Move children to parent
        for i, child in enumerate(children):
            parent.add_child(child, index + i)

        # Remove empty group
        parent.remove_child(group)
        self._id_map.pop(group.id, None)

        for callback in self._callbacks["on_structure_change"]:
            callback("ungroup", group)

        return True

    @track_changes
    def create_folder(
        self,
        name: str,
        parent_id: Optional[str] = None
    ) -> HierarchyFolder:
        """
        Create a new folder.

        Args:
            name: Folder name
            parent_id: Parent node ID (None for root)

        Returns:
            The created folder
        """
        folder = HierarchyFolder(name)
        self.add_node(folder, parent_id)
        return folder

    # Selection management
    @track_changes
    def select(self, node_ids: list[str], add: bool = False) -> None:
        """
        Select nodes.

        Args:
            node_ids: IDs of nodes to select
            add: Add to existing selection if True
        """
        old_selection = self._selection.copy()

        if not add:
            self._selection.clear()

        for nid in node_ids:
            if nid in self._id_map and nid not in self._selection:
                self._selection.append(nid)

        # Update node states
        for nid, node in self._id_map.items():
            node.state.selected = nid in self._selection

        if old_selection != self._selection:
            for callback in self._callbacks["on_selection_change"]:
                callback(old_selection, self._selection)

    def deselect(self, node_ids: list[str]) -> None:
        """Remove nodes from selection."""
        old_selection = self._selection.copy()

        for nid in node_ids:
            if nid in self._selection:
                self._selection.remove(nid)
                node = self._id_map.get(nid)
                if node:
                    node.state.selected = False

        if old_selection != self._selection:
            for callback in self._callbacks["on_selection_change"]:
                callback(old_selection, self._selection)

    def select_all(self) -> None:
        """Select all nodes."""
        self.select([nid for nid in self._id_map.keys() if nid != self._root.id])

    def deselect_all(self) -> None:
        """Clear selection."""
        self.select([])

    def invert_selection(self) -> None:
        """Invert current selection."""
        all_ids = set(self._id_map.keys()) - {self._root.id}
        current = set(self._selection)
        self.select(list(all_ids - current))

    # Clipboard operations
    def copy_selection(self) -> None:
        """Copy selected nodes to clipboard."""
        self._clipboard = [
            self._id_map[nid] for nid in self._selection
            if nid in self._id_map
        ]

    def cut_selection(self) -> None:
        """Cut selected nodes to clipboard."""
        self.copy_selection()
        for nid in self._selection.copy():
            self.remove_node(nid)

    @track_changes
    def paste(self, parent_id: Optional[str] = None) -> list[HierarchyNode]:
        """
        Paste clipboard contents.

        Args:
            parent_id: Parent for pasted nodes (None for root)

        Returns:
            List of pasted nodes
        """
        if not self._clipboard:
            return []

        pasted = []
        parent = self._id_map.get(parent_id) if parent_id else self._root

        if not parent:
            parent = self._root

        for node in self._clipboard:
            copy = self._deep_copy_node(node)
            parent.add_child(copy)
            self._register_node(copy)
            pasted.append(copy)

        return pasted

    def _deep_copy_node(self, node: HierarchyNode) -> HierarchyNode:
        """Create a deep copy of a node and its descendants."""
        if isinstance(node, HierarchyFolder):
            copy = HierarchyFolder(f"{node.name} (Copy)")
            copy._color = node._color
            copy._icon = node._icon
        elif isinstance(node, HierarchyGroup):
            copy = HierarchyGroup(f"{node.name} (Copy)")
            copy._pivot_mode = node._pivot_mode
        else:
            copy = HierarchyNode(
                f"{node.name} (Copy)",
                node.node_type,
                node.object_id,
            )

        copy._state = NodeState(
            expanded=node.state.expanded,
            visible=node.state.visible,
        )
        copy._layer_id = node._layer_id
        copy._metadata = node._metadata.copy()
        copy._local_transform = Transform(
            position=Vector3(
                node.local_transform.position.x,
                node.local_transform.position.y,
                node.local_transform.position.z,
            ),
            rotation=Quaternion(
                node.local_transform.rotation.x,
                node.local_transform.rotation.y,
                node.local_transform.rotation.z,
                node.local_transform.rotation.w,
            ),
            scale=Vector3(
                node.local_transform.scale.x,
                node.local_transform.scale.y,
                node.local_transform.scale.z,
            ),
        )

        for child in node.children:
            child_copy = self._deep_copy_node(child)
            copy.add_child(child_copy)

        return copy

    # Drag-drop
    @track_changes
    def execute_drag_drop(self, operation: DragDropOperation) -> bool:
        """
        Execute a drag-drop operation.

        Args:
            operation: The drag-drop operation to execute

        Returns:
            True if operation succeeded
        """
        if operation.operation_type == DragDropType.REPARENT:
            for source_id in operation.source_ids:
                if operation.target_id:
                    if not self.reparent(source_id, operation.target_id, operation.insert_index):
                        operation.error_message = f"Failed to reparent {source_id}"
                        return False

        elif operation.operation_type == DragDropType.REORDER:
            if operation.target_id and len(operation.source_ids) == 1:
                target = self._id_map.get(operation.target_id)
                source = self._id_map.get(operation.source_ids[0])
                if target and source and target.parent and target.parent == source.parent:
                    target.parent.reorder_child(source, operation.insert_index)

        elif operation.operation_type == DragDropType.COPY:
            self._clipboard = [
                self._id_map[sid] for sid in operation.source_ids
                if sid in self._id_map
            ]
            self.paste(operation.target_id)

        operation.completed = True
        return True

    # Filtering
    def apply_filter(self) -> list[HierarchyNode]:
        """
        Apply current filter and return visible nodes.

        Returns:
            List of nodes that pass the filter
        """
        results = []
        filter_settings = self._filter

        for node in self._root:
            if self._node_passes_filter(node, filter_settings):
                results.append(node)

        return results

    def _node_passes_filter(
        self,
        node: HierarchyNode,
        filter_settings: HierarchyFilter
    ) -> bool:
        """Check if a node passes the filter."""
        if filter_settings.mode == FilterMode.ALL:
            return True

        if filter_settings.mode == FilterMode.VISIBLE_ONLY:
            return node.state.visible

        if filter_settings.mode == FilterMode.SELECTED_ONLY:
            return node.state.selected

        if filter_settings.mode == FilterMode.UNLOCKED_ONLY:
            return not node.state.locked

        if filter_settings.mode == FilterMode.BY_LAYER:
            if node.layer_id:
                layer_bit = 1 << int(node.layer_id, 16) if node.layer_id.isdigit() else 0
                return bool(layer_bit & filter_settings.layer_mask)
            return True

        if filter_settings.mode == FilterMode.BY_NAME:
            search = filter_settings.search_text
            name = node.name
            if not filter_settings.case_sensitive:
                search = search.lower()
                name = name.lower()
            return search in name

        if filter_settings.mode == FilterMode.BY_TYPE:
            return filter_settings.type_filter == node.node_type.name

        return True

    def search(self, query: str, case_sensitive: bool = False) -> list[HierarchyNode]:
        """
        Search nodes by name.

        Args:
            query: Search string
            case_sensitive: Match case if True

        Returns:
            List of matching nodes
        """
        results = []
        search_query = query if case_sensitive else query.lower()

        for node in self._root:
            name = node.name if case_sensitive else node.name.lower()
            if search_query in name:
                results.append(node)

        return results

    def get_statistics(self) -> dict[str, int]:
        """Get tree statistics."""
        stats = {
            "total_nodes": 0,
            "objects": 0,
            "groups": 0,
            "folders": 0,
            "max_depth": 0,
            "selected": len(self._selection),
        }

        for node in self._root:
            stats["total_nodes"] += 1
            if node.node_type == NodeType.OBJECT:
                stats["objects"] += 1
            elif node.node_type == NodeType.GROUP:
                stats["groups"] += 1
            elif node.node_type == NodeType.FOLDER:
                stats["folders"] += 1
            stats["max_depth"] = max(stats["max_depth"], node.depth)

        return stats


__all__ = [
    "HierarchyNode",
    "HierarchyTree",
    "HierarchyFolder",
    "HierarchyGroup",
    "DragDropOperation",
    "HierarchyFilter",
    "NodeType",
    "DragDropType",
    "FilterMode",
    "NodeState",
]

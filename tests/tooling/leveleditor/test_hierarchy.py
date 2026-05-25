"""
Tests for the hierarchy module.

Tests reparenting, grouping, and layer assignment.
"""

import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.leveleditor.hierarchy import (
    HierarchyNode,
    HierarchyTree,
    HierarchyFolder,
    HierarchyGroup,
    DragDropOperation,
    HierarchyFilter,
    NodeType,
    DragDropType,
    FilterMode,
    NodeState,
)
from engine.tooling.leveleditor.placement import Vector3, Transform
from foundation.tracker import tracker


@pytest.fixture(autouse=True)
def reset_tracker():
    """Reset tracker state before each test."""
    tracker._dirty.clear()
    tracker._cb_global.clear()
    tracker._cb_type.clear()
    tracker._cb_obj.clear()
    tracker._undo.clear()
    tracker._redo.clear()
    tracker._txn = None
    yield


class TestHierarchyNode:
    """Tests for HierarchyNode class."""

    def test_creation_default(self):
        """Node should initialize with default values."""
        node = HierarchyNode("TestNode")
        assert node.name == "TestNode"
        assert node.node_type == NodeType.OBJECT
        assert node.parent is None
        assert len(node.children) == 0

    def test_creation_with_type(self):
        """Node should use provided type."""
        node = HierarchyNode("Group", NodeType.GROUP)
        assert node.node_type == NodeType.GROUP

    def test_creation_with_object_id(self):
        """Node should store object ID."""
        node = HierarchyNode("Object", object_id="obj-123")
        assert node.object_id == "obj-123"

    def test_id_unique(self):
        """Each node should have unique ID."""
        node1 = HierarchyNode("A")
        node2 = HierarchyNode("B")
        assert node1.id != node2.id

    def test_add_child(self):
        """Adding child should update relationships."""
        parent = HierarchyNode("Parent")
        child = HierarchyNode("Child")

        parent.add_child(child)

        assert child in parent.children
        assert child.parent is parent

    def test_add_child_at_index(self):
        """Adding child at index should insert correctly."""
        parent = HierarchyNode("Parent")
        child1 = HierarchyNode("Child1")
        child2 = HierarchyNode("Child2")
        child3 = HierarchyNode("Child3")

        parent.add_child(child1)
        parent.add_child(child3)
        parent.add_child(child2, index=1)

        assert parent.children[0] is child1
        assert parent.children[1] is child2
        assert parent.children[2] is child3

    def test_remove_child(self):
        """Removing child should update relationships."""
        parent = HierarchyNode("Parent")
        child = HierarchyNode("Child")

        parent.add_child(child)
        result = parent.remove_child(child)

        assert result is True
        assert child not in parent.children
        assert child.parent is None

    def test_remove_child_not_found(self):
        """Removing non-child should return False."""
        parent = HierarchyNode("Parent")
        other = HierarchyNode("Other")

        result = parent.remove_child(other)

        assert result is False

    def test_reparent_on_add(self):
        """Adding child with existing parent should reparent."""
        parent1 = HierarchyNode("Parent1")
        parent2 = HierarchyNode("Parent2")
        child = HierarchyNode("Child")

        parent1.add_child(child)
        parent2.add_child(child)

        assert child not in parent1.children
        assert child in parent2.children
        assert child.parent is parent2

    def test_reorder_child(self):
        """Reordering child should change position."""
        parent = HierarchyNode("Parent")
        child1 = HierarchyNode("Child1")
        child2 = HierarchyNode("Child2")
        child3 = HierarchyNode("Child3")

        parent.add_child(child1)
        parent.add_child(child2)
        parent.add_child(child3)

        parent.reorder_child(child3, 0)

        assert parent.children[0] is child3
        assert parent.children[1] is child1
        assert parent.children[2] is child2

    def test_get_child_index(self):
        """Should return correct child index."""
        parent = HierarchyNode("Parent")
        child1 = HierarchyNode("Child1")
        child2 = HierarchyNode("Child2")

        parent.add_child(child1)
        parent.add_child(child2)

        assert parent.get_child_index(child1) == 0
        assert parent.get_child_index(child2) == 1

    def test_get_child_index_not_found(self):
        """Should return -1 if child not found."""
        parent = HierarchyNode("Parent")
        other = HierarchyNode("Other")

        assert parent.get_child_index(other) == -1

    def test_find_child(self):
        """Should find direct child by name."""
        parent = HierarchyNode("Parent")
        child = HierarchyNode("Child")

        parent.add_child(child)

        found = parent.find_child("Child")
        assert found is child

    def test_find_child_not_found(self):
        """Should return None if child not found."""
        parent = HierarchyNode("Parent")

        found = parent.find_child("NonExistent")
        assert found is None

    def test_find_descendant(self):
        """Should find descendant by path."""
        root = HierarchyNode("Root")
        child = HierarchyNode("Child")
        grandchild = HierarchyNode("Grandchild")

        root.add_child(child)
        child.add_child(grandchild)

        found = root.find_descendant("Child/Grandchild")
        assert found is grandchild

    def test_is_leaf(self):
        """Should detect leaf nodes."""
        parent = HierarchyNode("Parent")
        child = HierarchyNode("Child")

        parent.add_child(child)

        assert parent.is_leaf is False
        assert child.is_leaf is True

    def test_is_root(self):
        """Should detect root nodes."""
        parent = HierarchyNode("Parent")
        child = HierarchyNode("Child")

        parent.add_child(child)

        assert parent.is_root is True
        assert child.is_root is False

    def test_depth(self):
        """Should calculate correct depth."""
        root = HierarchyNode("Root")
        child = HierarchyNode("Child")
        grandchild = HierarchyNode("Grandchild")

        root.add_child(child)
        child.add_child(grandchild)

        assert root.depth == 0
        assert child.depth == 1
        assert grandchild.depth == 2

    def test_path(self):
        """Should build correct path."""
        root = HierarchyNode("Root")
        child = HierarchyNode("Child")
        grandchild = HierarchyNode("Grandchild")

        root.add_child(child)
        child.add_child(grandchild)

        assert grandchild.path == "Root/Child/Grandchild"

    def test_get_ancestors(self):
        """Should return all ancestors."""
        root = HierarchyNode("Root")
        child = HierarchyNode("Child")
        grandchild = HierarchyNode("Grandchild")

        root.add_child(child)
        child.add_child(grandchild)

        ancestors = grandchild.get_ancestors()
        assert child in ancestors
        assert root in ancestors
        assert len(ancestors) == 2

    def test_get_descendants(self):
        """Should return all descendants."""
        root = HierarchyNode("Root")
        child1 = HierarchyNode("Child1")
        child2 = HierarchyNode("Child2")
        grandchild = HierarchyNode("Grandchild")

        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)

        descendants = root.get_descendants()
        assert child1 in descendants
        assert child2 in descendants
        assert grandchild in descendants
        assert len(descendants) == 3

    def test_get_siblings(self):
        """Should return sibling nodes."""
        parent = HierarchyNode("Parent")
        child1 = HierarchyNode("Child1")
        child2 = HierarchyNode("Child2")
        child3 = HierarchyNode("Child3")

        parent.add_child(child1)
        parent.add_child(child2)
        parent.add_child(child3)

        siblings = child2.get_siblings()
        assert child1 in siblings
        assert child3 in siblings
        assert child2 not in siblings

    def test_is_ancestor_of(self):
        """Should detect ancestor relationship."""
        root = HierarchyNode("Root")
        child = HierarchyNode("Child")
        grandchild = HierarchyNode("Grandchild")

        root.add_child(child)
        child.add_child(grandchild)

        assert root.is_ancestor_of(grandchild) is True
        assert child.is_ancestor_of(grandchild) is True
        assert grandchild.is_ancestor_of(root) is False

    def test_is_descendant_of(self):
        """Should detect descendant relationship."""
        root = HierarchyNode("Root")
        child = HierarchyNode("Child")

        root.add_child(child)

        assert child.is_descendant_of(root) is True
        assert root.is_descendant_of(child) is False

    def test_iterate(self):
        """Should iterate over node and descendants."""
        root = HierarchyNode("Root")
        child = HierarchyNode("Child")
        grandchild = HierarchyNode("Grandchild")

        root.add_child(child)
        child.add_child(grandchild)

        nodes = list(root)
        assert root in nodes
        assert child in nodes
        assert grandchild in nodes

    def test_layer_id(self):
        """Should store and track layer ID."""
        node = HierarchyNode("Node")
        node.layer_id = "layer-1"

        assert node.layer_id == "layer-1"

    def test_metadata(self):
        """Should store and retrieve metadata."""
        node = HierarchyNode("Node")
        node.set_metadata("key", "value")

        assert node.get_metadata("key") == "value"
        assert node.get_metadata("missing", "default") == "default"


class TestHierarchyFolder:
    """Tests for HierarchyFolder class."""

    def test_creation(self):
        """Folder should initialize with folder type."""
        folder = HierarchyFolder("MyFolder")
        assert folder.name == "MyFolder"
        assert folder.node_type == NodeType.FOLDER

    def test_color(self):
        """Should store and track color."""
        folder = HierarchyFolder("Folder")
        folder.color = (1.0, 0.0, 0.0)

        assert folder.color == (1.0, 0.0, 0.0)

    def test_icon(self):
        """Should store and track icon."""
        folder = HierarchyFolder("Folder")
        folder.icon = "star"

        assert folder.icon == "star"


class TestHierarchyGroup:
    """Tests for HierarchyGroup class."""

    def test_creation(self):
        """Group should initialize with group type."""
        group = HierarchyGroup("MyGroup")
        assert group.name == "MyGroup"
        assert group.node_type == NodeType.GROUP

    def test_pivot_mode(self):
        """Should store pivot mode."""
        group = HierarchyGroup("Group")
        group.pivot_mode = "first_child"

        assert group.pivot_mode == "first_child"

    def test_calculate_bounds(self):
        """Should calculate bounds from children."""
        group = HierarchyGroup("Group")
        child1 = HierarchyNode("Child1")
        child2 = HierarchyNode("Child2")

        child1.local_transform = Transform(position=Vector3(0, 0, 0))
        child2.local_transform = Transform(position=Vector3(10, 5, 10))

        group.add_child(child1)
        group.add_child(child2)

        bounds_min, bounds_max = group.calculate_bounds()

        assert bounds_min.x == 0
        assert bounds_max.x == 10
        assert bounds_max.y == 5

    def test_center_pivot(self):
        """Should center pivot on children."""
        group = HierarchyGroup("Group")
        child1 = HierarchyNode("Child1")
        child2 = HierarchyNode("Child2")

        child1.local_transform = Transform(position=Vector3(0, 0, 0))
        child2.local_transform = Transform(position=Vector3(10, 0, 10))

        group.add_child(child1)
        group.add_child(child2)

        group.center_pivot()

        assert group.local_transform.position.x == 5
        assert group.local_transform.position.z == 5


class TestHierarchyTree:
    """Tests for HierarchyTree class."""

    def test_creation(self):
        """Tree should initialize with root node."""
        tree = HierarchyTree()
        assert tree.root is not None
        assert tree.root.name == "Scene"

    def test_add_node(self):
        """Should add node to tree."""
        tree = HierarchyTree()
        node = HierarchyNode("TestNode")

        result = tree.add_node(node)

        assert result is True
        assert node in tree.root.children

    def test_add_node_to_parent(self):
        """Should add node under specified parent."""
        tree = HierarchyTree()
        parent = HierarchyNode("Parent")
        child = HierarchyNode("Child")

        tree.add_node(parent)
        tree.add_node(child, parent.id)

        assert child in parent.children

    def test_remove_node(self):
        """Should remove node from tree."""
        tree = HierarchyTree()
        node = HierarchyNode("TestNode")

        tree.add_node(node)
        result = tree.remove_node(node.id)

        assert result is True
        assert node not in tree.root.children

    def test_remove_root_fails(self):
        """Should not be able to remove root."""
        tree = HierarchyTree()
        result = tree.remove_node(tree.root.id)

        assert result is False

    def test_get_node(self):
        """Should get node by ID."""
        tree = HierarchyTree()
        node = HierarchyNode("TestNode")

        tree.add_node(node)
        found = tree.get_node(node.id)

        assert found is node

    def test_find_by_name(self):
        """Should find nodes by name."""
        tree = HierarchyTree()
        node1 = HierarchyNode("Target")
        node2 = HierarchyNode("Target")
        node3 = HierarchyNode("Other")

        tree.add_node(node1)
        tree.add_node(node2)
        tree.add_node(node3)

        found = tree.find_by_name("Target")
        assert len(found) == 2

    def test_reparent(self):
        """Should reparent node."""
        tree = HierarchyTree()
        parent1 = HierarchyNode("Parent1")
        parent2 = HierarchyNode("Parent2")
        child = HierarchyNode("Child")

        tree.add_node(parent1)
        tree.add_node(parent2)
        tree.add_node(child, parent1.id)

        result = tree.reparent(child.id, parent2.id)

        assert result is True
        assert child in parent2.children
        assert child not in parent1.children

    def test_reparent_prevents_circular(self):
        """Should prevent circular references."""
        tree = HierarchyTree()
        parent = HierarchyNode("Parent")
        child = HierarchyNode("Child")

        tree.add_node(parent)
        tree.add_node(child, parent.id)

        result = tree.reparent(parent.id, child.id)

        assert result is False

    def test_group_selection(self):
        """Should group selected nodes."""
        tree = HierarchyTree()
        node1 = HierarchyNode("Node1")
        node2 = HierarchyNode("Node2")

        tree.add_node(node1)
        tree.add_node(node2)
        tree.select([node1.id, node2.id])

        group = tree.group_selection("MyGroup")

        assert group is not None
        assert node1 in group.children
        assert node2 in group.children

    def test_ungroup(self):
        """Should dissolve group and move children to parent."""
        tree = HierarchyTree()
        node1 = HierarchyNode("Node1")
        node2 = HierarchyNode("Node2")

        tree.add_node(node1)
        tree.add_node(node2)
        tree.select([node1.id, node2.id])

        group = tree.group_selection("Group")
        result = tree.ungroup(group.id)

        assert result is True
        assert node1 in tree.root.children
        assert node2 in tree.root.children

    def test_create_folder(self):
        """Should create folder node."""
        tree = HierarchyTree()
        folder = tree.create_folder("MyFolder")

        assert folder is not None
        assert folder.node_type == NodeType.FOLDER
        assert folder in tree.root.children

    def test_select(self):
        """Should update selection."""
        tree = HierarchyTree()
        node = HierarchyNode("Node")

        tree.add_node(node)
        tree.select([node.id])

        assert node.id in tree.selection
        assert node.state.selected is True

    def test_deselect(self):
        """Should remove from selection."""
        tree = HierarchyTree()
        node = HierarchyNode("Node")

        tree.add_node(node)
        tree.select([node.id])
        tree.deselect([node.id])

        assert node.id not in tree.selection
        assert node.state.selected is False

    def test_select_all(self):
        """Should select all nodes."""
        tree = HierarchyTree()
        node1 = HierarchyNode("Node1")
        node2 = HierarchyNode("Node2")

        tree.add_node(node1)
        tree.add_node(node2)
        tree.select_all()

        assert node1.id in tree.selection
        assert node2.id in tree.selection

    def test_deselect_all(self):
        """Should clear selection."""
        tree = HierarchyTree()
        node = HierarchyNode("Node")

        tree.add_node(node)
        tree.select([node.id])
        tree.deselect_all()

        assert len(tree.selection) == 0

    def test_copy_paste(self):
        """Should copy and paste nodes."""
        tree = HierarchyTree()
        node = HierarchyNode("Original")

        tree.add_node(node)
        tree.select([node.id])
        tree.copy_selection()
        pasted = tree.paste()

        assert len(pasted) == 1
        assert pasted[0].name == "Original (Copy)"

    def test_execute_drag_drop_reparent(self):
        """Should execute reparent drag-drop."""
        tree = HierarchyTree()
        parent = HierarchyNode("Parent")
        child = HierarchyNode("Child")

        tree.add_node(parent)
        tree.add_node(child)

        op = DragDropOperation(
            source_ids=[child.id],
            target_id=parent.id,
            operation_type=DragDropType.REPARENT
        )

        result = tree.execute_drag_drop(op)

        assert result is True
        assert child in parent.children

    def test_search(self):
        """Should search nodes by name."""
        tree = HierarchyTree()
        tree.add_node(HierarchyNode("Apple"))
        tree.add_node(HierarchyNode("Banana"))
        tree.add_node(HierarchyNode("Apricot"))

        results = tree.search("ap")

        assert len(results) == 2  # Apple and Apricot

    def test_apply_filter(self):
        """Should filter visible nodes."""
        tree = HierarchyTree()
        visible = HierarchyNode("Visible")
        hidden = HierarchyNode("Hidden")

        tree.add_node(visible)
        tree.add_node(hidden)
        hidden.state.visible = False

        tree.filter = HierarchyFilter(mode=FilterMode.VISIBLE_ONLY)
        results = tree.apply_filter()

        visible_nodes = [n for n in results if n.state.visible]
        assert len(visible_nodes) >= 1

    def test_get_statistics(self):
        """Should return tree statistics."""
        tree = HierarchyTree()
        tree.add_node(HierarchyNode("Object"))
        tree.create_folder("Folder")

        stats = tree.get_statistics()

        assert stats["total_nodes"] >= 2
        assert stats["folders"] >= 1

"""Whitebox tests for skeleton and bone hierarchy.

Task: T-AG-1.3 Skeleton and Bone Hierarchy
Tests: Internal implementation, edge cases, algorithmic correctness
"""

from __future__ import annotations

import pytest
from copy import deepcopy
from typing import Optional

from engine.animation.graph.skeleton import Bone, Skeleton
from engine.core.math.transform import Transform
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat


# =============================================================================
# SECTION 1: Bone Class - Creation and Basic Properties
# =============================================================================


class TestBoneCreation:
    """Tests for Bone construction and initialization."""

    def test_bone_creation_minimal(self) -> None:
        """Bone can be created with just a name."""
        bone = Bone("spine")
        assert bone.name == "spine"
        assert bone.parent is None
        assert bone.children == []
        assert isinstance(bone.bind_pose, Transform)

    def test_bone_creation_with_parent(self) -> None:
        """Bone can be created with a parent reference."""
        root = Bone("root")
        child = Bone("child", parent=root)
        assert child.parent is root
        # Note: manually setting parent does NOT auto-add to parent's children
        assert child not in root.children

    def test_bone_creation_with_bind_pose(self) -> None:
        """Bone can be created with a custom bind pose."""
        pose = Transform(
            translation=Vec3(1.0, 2.0, 3.0),
            rotation=Quat.from_axis_angle(Vec3(0, 1, 0), 0.5),
            scale=Vec3(1.0, 1.0, 1.0),
        )
        bone = Bone("arm", bind_pose=pose)
        assert bone.bind_pose.translation.x == pytest.approx(1.0)
        assert bone.bind_pose.translation.y == pytest.approx(2.0)
        assert bone.bind_pose.translation.z == pytest.approx(3.0)

    def test_bone_default_bind_pose_is_identity(self) -> None:
        """Bone defaults to identity bind pose."""
        bone = Bone("test")
        identity = Transform.identity()
        assert bone.bind_pose.translation.x == pytest.approx(identity.translation.x)
        assert bone.bind_pose.translation.y == pytest.approx(identity.translation.y)
        assert bone.bind_pose.translation.z == pytest.approx(identity.translation.z)

    def test_bone_empty_name_raises_value_error(self) -> None:
        """Bone creation with empty name raises ValueError."""
        with pytest.raises(ValueError, match="Bone name must be a non-empty string"):
            Bone("")

    def test_bone_name_whitespace_only_allowed(self) -> None:
        """Bone name with only whitespace is technically allowed (not empty)."""
        bone = Bone("   ")
        assert bone.name == "   "


class TestBoneProperties:
    """Tests for Bone computed properties."""

    def test_is_root_true_when_no_parent(self) -> None:
        """is_root returns True for parentless bones."""
        bone = Bone("root")
        assert bone.is_root is True

    def test_is_root_false_when_has_parent(self) -> None:
        """is_root returns False when bone has a parent."""
        root = Bone("root")
        child = Bone("child")
        root.add_child(child)
        assert child.is_root is False

    def test_depth_root_is_zero(self) -> None:
        """Root bone has depth 0."""
        root = Bone("root")
        assert root.depth == 0

    def test_depth_child_is_one(self) -> None:
        """Direct child of root has depth 1."""
        root = Bone("root")
        child = Bone("child")
        root.add_child(child)
        assert child.depth == 1

    def test_depth_grandchild_is_two(self) -> None:
        """Grandchild has depth 2."""
        root = Bone("root")
        child = Bone("child")
        grandchild = Bone("grandchild")
        root.add_child(child)
        child.add_child(grandchild)
        assert grandchild.depth == 2

    def test_depth_deep_hierarchy(self) -> None:
        """Depth calculation for deep hierarchies."""
        bones = [Bone(f"bone_{i}") for i in range(10)]
        for i in range(1, 10):
            bones[i - 1].add_child(bones[i])
        assert bones[9].depth == 9


# =============================================================================
# SECTION 2: Bone Hierarchy Navigation
# =============================================================================


class TestBoneNavigation:
    """Tests for Bone hierarchy navigation methods."""

    def test_get_root_from_root(self) -> None:
        """get_root returns self for root bone."""
        root = Bone("root")
        assert root.get_root() is root

    def test_get_root_from_child(self) -> None:
        """get_root returns the root from a child bone."""
        root = Bone("root")
        child = Bone("child")
        grandchild = Bone("grandchild")
        root.add_child(child)
        child.add_child(grandchild)
        assert grandchild.get_root() is root

    def test_get_siblings_root_has_none(self) -> None:
        """Root bone has no siblings."""
        root = Bone("root")
        assert root.get_siblings() == []

    def test_get_siblings_returns_other_children(self) -> None:
        """get_siblings returns all other children of parent."""
        root = Bone("root")
        child1 = Bone("child1")
        child2 = Bone("child2")
        child3 = Bone("child3")
        root.add_child(child1)
        root.add_child(child2)
        root.add_child(child3)

        siblings = child1.get_siblings()
        assert len(siblings) == 2
        assert child2 in siblings
        assert child3 in siblings
        assert child1 not in siblings

    def test_is_ancestor_of_direct_child(self) -> None:
        """is_ancestor_of returns True for direct child."""
        root = Bone("root")
        child = Bone("child")
        root.add_child(child)
        assert root.is_ancestor_of(child) is True
        assert child.is_ancestor_of(root) is False

    def test_is_ancestor_of_grandchild(self) -> None:
        """is_ancestor_of works through multiple levels."""
        root = Bone("root")
        child = Bone("child")
        grandchild = Bone("grandchild")
        root.add_child(child)
        child.add_child(grandchild)

        assert root.is_ancestor_of(grandchild) is True
        assert child.is_ancestor_of(grandchild) is True
        assert grandchild.is_ancestor_of(root) is False

    def test_is_ancestor_of_self_is_false(self) -> None:
        """A bone is not an ancestor of itself."""
        bone = Bone("bone")
        assert bone.is_ancestor_of(bone) is False

    def test_is_descendant_of_inverse_of_is_ancestor(self) -> None:
        """is_descendant_of is the inverse of is_ancestor_of."""
        root = Bone("root")
        child = Bone("child")
        root.add_child(child)

        assert child.is_descendant_of(root) is True
        assert root.is_descendant_of(child) is False

    def test_get_descendants_empty_for_leaf(self) -> None:
        """Leaf bone has no descendants."""
        leaf = Bone("leaf")
        assert leaf.get_descendants() == []

    def test_get_descendants_breadth_first(self) -> None:
        """get_descendants returns bones in breadth-first order."""
        root = Bone("root")
        child1 = Bone("child1")
        child2 = Bone("child2")
        grandchild1 = Bone("grandchild1")
        grandchild2 = Bone("grandchild2")

        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild1)
        child1.add_child(grandchild2)

        descendants = root.get_descendants()
        # BFS: children first, then grandchildren
        assert len(descendants) == 4
        # child1 and child2 should come before grandchildren
        assert descendants.index(child1) < descendants.index(grandchild1)
        assert descendants.index(child1) < descendants.index(grandchild2)


# =============================================================================
# SECTION 3: Bone Child Management
# =============================================================================


class TestBoneChildManagement:
    """Tests for add_child and remove_child operations."""

    def test_add_child_sets_parent(self) -> None:
        """add_child sets the child's parent reference."""
        parent = Bone("parent")
        child = Bone("child")
        parent.add_child(child)
        assert child.parent is parent

    def test_add_child_appends_to_children(self) -> None:
        """add_child appends bone to parent's children list."""
        parent = Bone("parent")
        child = Bone("child")
        parent.add_child(child)
        assert child in parent.children
        assert len(parent.children) == 1

    def test_add_child_preserves_insertion_order(self) -> None:
        """Children are stored in insertion order."""
        parent = Bone("parent")
        child1 = Bone("child1")
        child2 = Bone("child2")
        child3 = Bone("child3")

        parent.add_child(child1)
        parent.add_child(child2)
        parent.add_child(child3)

        assert parent.children == [child1, child2, child3]

    def test_add_child_reparents_from_old_parent(self) -> None:
        """Adding a child that has a parent removes it from old parent."""
        old_parent = Bone("old_parent")
        new_parent = Bone("new_parent")
        child = Bone("child")

        old_parent.add_child(child)
        assert child in old_parent.children

        new_parent.add_child(child)
        assert child not in old_parent.children
        assert child in new_parent.children
        assert child.parent is new_parent

    def test_add_child_self_raises_value_error(self) -> None:
        """Cannot add a bone as its own child."""
        bone = Bone("bone")
        with pytest.raises(ValueError, match="Cannot add a bone as its own child"):
            bone.add_child(bone)

    def test_add_child_duplicate_raises_value_error(self) -> None:
        """Cannot add the same bone as a child twice."""
        parent = Bone("parent")
        child = Bone("child")
        parent.add_child(child)
        with pytest.raises(ValueError, match="already a child"):
            parent.add_child(child)

    def test_remove_child_clears_parent(self) -> None:
        """remove_child clears the child's parent reference."""
        parent = Bone("parent")
        child = Bone("child")
        parent.add_child(child)
        result = parent.remove_child(child)

        assert result is True
        assert child.parent is None
        assert child not in parent.children

    def test_remove_child_non_child_returns_false(self) -> None:
        """remove_child returns False for non-child bones."""
        parent = Bone("parent")
        other = Bone("other")
        result = parent.remove_child(other)
        assert result is False

    def test_remove_child_preserves_grandchildren(self) -> None:
        """remove_child keeps grandchildren attached to removed child."""
        parent = Bone("parent")
        child = Bone("child")
        grandchild = Bone("grandchild")

        parent.add_child(child)
        child.add_child(grandchild)

        parent.remove_child(child)

        assert grandchild.parent is child
        assert grandchild in child.children


# =============================================================================
# SECTION 4: Bone Copy and Representation
# =============================================================================


class TestBoneCopyAndRepr:
    """Tests for Bone copy and string representation."""

    def test_bone_copy_creates_deep_copy(self) -> None:
        """copy() creates independent copy of bone and descendants."""
        root = Bone("root")
        child = Bone("child")
        root.add_child(child)

        root_copy = root.copy()

        assert root_copy.name == "root"
        assert root_copy is not root
        assert len(root_copy.children) == 1
        assert root_copy.children[0].name == "child"
        assert root_copy.children[0] is not child

    def test_bone_repr_root(self) -> None:
        """repr shows root status for parentless bones."""
        bone = Bone("root")
        assert "root" in repr(bone)
        assert "Bone(" in repr(bone)

    def test_bone_repr_with_parent(self) -> None:
        """repr shows parent name for child bones."""
        parent = Bone("parent")
        child = Bone("child")
        parent.add_child(child)
        assert "parent" in repr(child)


# =============================================================================
# SECTION 5: Skeleton Class - Creation and Properties
# =============================================================================


class TestSkeletonCreation:
    """Tests for Skeleton construction and initialization."""

    def test_skeleton_creation_default_name(self) -> None:
        """Skeleton can be created with default name."""
        skel = Skeleton()
        assert skel.name == "skeleton"
        assert skel.bone_count == 0

    def test_skeleton_creation_custom_name(self) -> None:
        """Skeleton can be created with custom name."""
        skel = Skeleton("humanoid")
        assert skel.name == "humanoid"

    def test_skeleton_empty_name_raises_value_error(self) -> None:
        """Skeleton creation with empty name raises ValueError."""
        with pytest.raises(ValueError, match="Skeleton name must be a non-empty string"):
            Skeleton("")

    def test_skeleton_properties_read_only(self) -> None:
        """bones and root_bones return copies, not internal state."""
        skel = Skeleton()
        skel.add_bone("root")

        bones_copy = skel.bones
        roots_copy = skel.root_bones

        # Modifying the copies should not affect the skeleton
        bones_copy.clear()
        roots_copy.clear()

        assert skel.bone_count == 1
        assert len(skel.root_bones) == 1


class TestSkeletonBoneManagement:
    """Tests for Skeleton bone management methods."""

    def test_add_bone_root(self) -> None:
        """add_bone with no parent creates a root bone."""
        skel = Skeleton()
        bone = skel.add_bone("root")

        assert bone.name == "root"
        assert bone.parent is None
        assert bone in skel.root_bones
        assert skel.bone_count == 1

    def test_add_bone_with_parent(self) -> None:
        """add_bone with parent_name creates child bone."""
        skel = Skeleton()
        skel.add_bone("root")
        child = skel.add_bone("child", parent_name="root")

        assert child.parent.name == "root"
        assert child not in skel.root_bones
        assert skel.bone_count == 2

    def test_add_bone_with_bind_pose(self) -> None:
        """add_bone with bind_pose stores the transform."""
        skel = Skeleton()
        pose = Transform(translation=Vec3(1, 2, 3))
        bone = skel.add_bone("root", bind_pose=pose)

        assert bone.bind_pose.translation.x == pytest.approx(1.0)
        assert bone.bind_pose.translation.y == pytest.approx(2.0)

    def test_add_bone_duplicate_name_raises_value_error(self) -> None:
        """Cannot add two bones with the same name."""
        skel = Skeleton()
        skel.add_bone("root")
        with pytest.raises(ValueError, match="already exists"):
            skel.add_bone("root")

    def test_add_bone_missing_parent_raises_value_error(self) -> None:
        """add_bone with non-existent parent raises ValueError."""
        skel = Skeleton()
        with pytest.raises(ValueError, match="not found"):
            skel.add_bone("child", parent_name="nonexistent")

    def test_remove_bone_basic(self) -> None:
        """remove_bone removes a bone from the skeleton."""
        skel = Skeleton()
        skel.add_bone("root")
        result = skel.remove_bone("root")

        assert result is True
        assert skel.bone_count == 0
        assert skel.get_bone("root") is None

    def test_remove_bone_adopts_children(self) -> None:
        """remove_bone re-parents children to grandparent."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("middle", parent_name="root")
        skel.add_bone("child", parent_name="middle")

        skel.remove_bone("middle")

        assert skel.bone_count == 2
        child = skel.get_bone("child")
        root = skel.get_bone("root")
        assert child.parent is root
        assert child in root.children

    def test_remove_bone_root_promotes_children(self) -> None:
        """Removing root promotes its children to root status."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child1", parent_name="root")
        skel.add_bone("child2", parent_name="root")

        skel.remove_bone("root")

        assert skel.bone_count == 2
        assert len(skel.root_bones) == 2
        assert skel.get_bone("child1").parent is None
        assert skel.get_bone("child2").parent is None

    def test_remove_bone_nonexistent_returns_false(self) -> None:
        """remove_bone returns False for non-existent bones."""
        skel = Skeleton()
        result = skel.remove_bone("nonexistent")
        assert result is False

    def test_has_bone(self) -> None:
        """has_bone checks bone existence."""
        skel = Skeleton()
        skel.add_bone("root")
        assert skel.has_bone("root") is True
        assert skel.has_bone("nonexistent") is False


# =============================================================================
# SECTION 6: Skeleton get_bone O(1) Lookup
# =============================================================================


class TestSkeletonGetBone:
    """Tests for O(1) bone lookup."""

    def test_get_bone_found(self) -> None:
        """get_bone returns bone when found."""
        skel = Skeleton()
        skel.add_bone("root")
        bone = skel.get_bone("root")
        assert bone is not None
        assert bone.name == "root"

    def test_get_bone_not_found_returns_none(self) -> None:
        """get_bone returns None when not found."""
        skel = Skeleton()
        assert skel.get_bone("nonexistent") is None

    def test_get_bone_o1_complexity(self) -> None:
        """get_bone uses dictionary lookup (O(1))."""
        skel = Skeleton()
        for i in range(1000):
            skel.add_bone(f"bone_{i}", parent_name=f"bone_{i-1}" if i > 0 else None)

        # Should be constant time
        bone = skel.get_bone("bone_999")
        assert bone is not None
        assert bone.name == "bone_999"

    def test_get_child_names(self) -> None:
        """get_child_names returns names of direct children."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child1", parent_name="root")
        skel.add_bone("child2", parent_name="root")

        names = skel.get_child_names("root")
        assert set(names) == {"child1", "child2"}

    def test_get_child_names_nonexistent(self) -> None:
        """get_child_names returns empty list for nonexistent bone."""
        skel = Skeleton()
        assert skel.get_child_names("nonexistent") == []

    def test_get_parent_name(self) -> None:
        """get_parent_name returns parent's name."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")

        assert skel.get_parent_name("child") == "root"
        assert skel.get_parent_name("root") is None
        assert skel.get_parent_name("nonexistent") is None

    def test_get_ancestor_names(self) -> None:
        """get_ancestor_names returns chain from bone to root."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("spine", parent_name="root")
        skel.add_bone("chest", parent_name="spine")
        skel.add_bone("head", parent_name="chest")

        ancestors = skel.get_ancestor_names("head")
        assert ancestors == ["head", "chest", "spine", "root"]

    def test_get_depth(self) -> None:
        """get_depth returns bone depth."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")
        skel.add_bone("grandchild", parent_name="child")

        assert skel.get_depth("root") == 0
        assert skel.get_depth("child") == 1
        assert skel.get_depth("grandchild") == 2
        assert skel.get_depth("nonexistent") == -1

    def test_get_max_depth(self) -> None:
        """get_max_depth returns maximum bone depth."""
        skel = Skeleton()
        skel.add_bone("root1")
        skel.add_bone("child1", parent_name="root1")
        skel.add_bone("root2")
        skel.add_bone("child2", parent_name="root2")
        skel.add_bone("grandchild2", parent_name="child2")

        assert skel.get_max_depth() == 2

    def test_get_max_depth_empty_skeleton(self) -> None:
        """get_max_depth returns -1 for empty skeleton."""
        skel = Skeleton()
        assert skel.get_max_depth() == -1


# =============================================================================
# SECTION 7: Skeleton get_chain for IK Integration
# =============================================================================


class TestSkeletonGetChain:
    """Tests for get_chain algorithm (IK integration)."""

    def test_get_chain_same_bone(self) -> None:
        """Chain from bone to itself returns single-element list."""
        skel = Skeleton()
        skel.add_bone("root")
        chain = skel.get_chain("root", "root")
        assert len(chain) == 1
        assert chain[0].name == "root"

    def test_get_chain_parent_to_child(self) -> None:
        """Chain from parent to child."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")
        skel.add_bone("grandchild", parent_name="child")

        chain = skel.get_chain("root", "grandchild")
        names = [b.name for b in chain]
        assert names == ["root", "child", "grandchild"]

    def test_get_chain_child_to_parent(self) -> None:
        """Chain from child to parent (reversed direction)."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")
        skel.add_bone("grandchild", parent_name="child")

        chain = skel.get_chain("grandchild", "root")
        names = [b.name for b in chain]
        assert names == ["grandchild", "child", "root"]

    def test_get_chain_sibling_to_sibling(self) -> None:
        """Chain between siblings goes through common ancestor."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("left_arm", parent_name="root")
        skel.add_bone("left_hand", parent_name="left_arm")
        skel.add_bone("right_arm", parent_name="root")
        skel.add_bone("right_hand", parent_name="right_arm")

        chain = skel.get_chain("left_hand", "right_hand")
        names = [b.name for b in chain]
        assert names == ["left_hand", "left_arm", "root", "right_arm", "right_hand"]

    def test_get_chain_common_ancestor_in_middle(self) -> None:
        """Chain correctly identifies lowest common ancestor."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("spine", parent_name="root")
        skel.add_bone("chest", parent_name="spine")
        skel.add_bone("left_shoulder", parent_name="chest")
        skel.add_bone("left_elbow", parent_name="left_shoulder")
        skel.add_bone("right_shoulder", parent_name="chest")
        skel.add_bone("right_elbow", parent_name="right_shoulder")

        chain = skel.get_chain("left_elbow", "right_elbow")
        names = [b.name for b in chain]
        # LCA is chest
        assert names == [
            "left_elbow",
            "left_shoulder",
            "chest",
            "right_shoulder",
            "right_elbow",
        ]

    def test_get_chain_disconnected_trees(self) -> None:
        """Chain between disconnected trees returns empty list."""
        skel = Skeleton()
        skel.add_bone("root1")
        skel.add_bone("child1", parent_name="root1")
        skel.add_bone("root2")
        skel.add_bone("child2", parent_name="root2")

        chain = skel.get_chain("child1", "child2")
        assert chain == []

    def test_get_chain_nonexistent_start(self) -> None:
        """Chain with nonexistent start returns empty list."""
        skel = Skeleton()
        skel.add_bone("root")
        chain = skel.get_chain("nonexistent", "root")
        assert chain == []

    def test_get_chain_nonexistent_end(self) -> None:
        """Chain with nonexistent end returns empty list."""
        skel = Skeleton()
        skel.add_bone("root")
        chain = skel.get_chain("root", "nonexistent")
        assert chain == []

    def test_get_chain_ik_chain_arm(self) -> None:
        """Realistic IK chain for arm."""
        skel = Skeleton()
        skel.add_bone("shoulder")
        skel.add_bone("upper_arm", parent_name="shoulder")
        skel.add_bone("elbow", parent_name="upper_arm")
        skel.add_bone("forearm", parent_name="elbow")
        skel.add_bone("wrist", parent_name="forearm")
        skel.add_bone("hand", parent_name="wrist")

        # IK from shoulder to hand
        chain = skel.get_chain("shoulder", "hand")
        names = [b.name for b in chain]
        assert names == [
            "shoulder",
            "upper_arm",
            "elbow",
            "forearm",
            "wrist",
            "hand",
        ]

    def test_get_chain_preserves_bone_references(self) -> None:
        """Chain returns actual bone references, not copies."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")

        chain = skel.get_chain("root", "child")
        assert chain[0] is skel.get_bone("root")
        assert chain[1] is skel.get_bone("child")


# =============================================================================
# SECTION 8: Skeleton Validation
# =============================================================================


class TestSkeletonValidation:
    """Tests for skeleton validation (validate and is_valid)."""

    def test_validate_empty_skeleton(self) -> None:
        """Empty skeleton fails validation."""
        skel = Skeleton()
        errors = skel.validate()
        assert len(errors) == 1
        assert "empty" in errors[0].lower()

    def test_validate_valid_single_root(self) -> None:
        """Valid skeleton with single root passes validation."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child1", parent_name="root")
        skel.add_bone("child2", parent_name="root")

        errors = skel.validate()
        assert errors == []
        assert skel.is_valid() is True

    def test_validate_valid_multi_root(self) -> None:
        """Valid skeleton with multiple roots passes validation."""
        skel = Skeleton()
        skel.add_bone("root1")
        skel.add_bone("child1", parent_name="root1")
        skel.add_bone("root2")
        skel.add_bone("child2", parent_name="root2")

        errors = skel.validate()
        assert errors == []
        assert skel.is_valid() is True

    def test_validate_orphan_bone_detection(self) -> None:
        """Validation detects orphan bones not reachable from roots."""
        skel = Skeleton()
        skel.add_bone("root")

        # Manually inject an orphan bone (bypasses normal add_bone)
        orphan = Bone("orphan")
        skel._bones["orphan"] = orphan
        # Don't add to root_bones or any parent's children

        errors = skel.validate()
        assert any("orphan" in e.lower() for e in errors)
        assert skel.is_valid() is False

    def test_validate_parent_reference_consistency(self) -> None:
        """Validation detects invalid parent references."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")

        # Manually corrupt parent reference
        unregistered = Bone("unregistered")
        skel.get_bone("child").parent = unregistered

        errors = skel.validate()
        assert any("not registered" in e for e in errors)

    def test_validate_child_reference_consistency(self) -> None:
        """Validation detects invalid child references."""
        skel = Skeleton()
        skel.add_bone("root")

        # Manually add unregistered child
        unregistered = Bone("unregistered")
        skel.get_bone("root").children.append(unregistered)

        errors = skel.validate()
        assert any("not registered" in e for e in errors)

    def test_validate_cycle_detection(self) -> None:
        """Validation detects parent cycles."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")

        # Manually create a cycle (root -> child -> root)
        root_bone = skel.get_bone("root")
        child_bone = skel.get_bone("child")
        root_bone.parent = child_bone

        errors = skel.validate()
        assert any("cycle" in e.lower() for e in errors)
        assert skel.is_valid() is False

    def test_is_valid_convenience_wrapper(self) -> None:
        """is_valid returns bool based on validate() results."""
        skel = Skeleton()
        assert skel.is_valid() is False  # Empty

        skel.add_bone("root")
        assert skel.is_valid() is True


# =============================================================================
# SECTION 9: Multi-Root Skeleton Support
# =============================================================================


class TestMultiRootSkeleton:
    """Tests for skeletons with multiple root bones."""

    def test_multi_root_creation(self) -> None:
        """Multiple roots can be created."""
        skel = Skeleton("multi_root")
        skel.add_bone("root1")
        skel.add_bone("root2")
        skel.add_bone("root3")

        assert skel.bone_count == 3
        assert len(skel.root_bones) == 3

    def test_multi_root_independent_trees(self) -> None:
        """Each root forms an independent tree."""
        skel = Skeleton()
        skel.add_bone("root1")
        skel.add_bone("child1", parent_name="root1")
        skel.add_bone("root2")
        skel.add_bone("child2", parent_name="root2")

        root1 = skel.get_bone("root1")
        root2 = skel.get_bone("root2")
        child1 = skel.get_bone("child1")
        child2 = skel.get_bone("child2")

        # Trees are independent
        assert not root1.is_ancestor_of(child2)
        assert not root2.is_ancestor_of(child1)

    def test_multi_root_get_chain_across_trees(self) -> None:
        """get_chain between different trees returns empty list."""
        skel = Skeleton()
        skel.add_bone("left_arm_root")
        skel.add_bone("left_hand", parent_name="left_arm_root")
        skel.add_bone("right_arm_root")
        skel.add_bone("right_hand", parent_name="right_arm_root")

        chain = skel.get_chain("left_hand", "right_hand")
        assert chain == []

    def test_multi_root_validation_passes(self) -> None:
        """Multi-root skeleton passes validation."""
        skel = Skeleton()
        skel.add_bone("tree1_root")
        skel.add_bone("tree1_child", parent_name="tree1_root")
        skel.add_bone("tree2_root")
        skel.add_bone("tree2_child", parent_name="tree2_root")

        assert skel.is_valid() is True


# =============================================================================
# SECTION 10: Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_deep_hierarchy(self) -> None:
        """Deep hierarchies work correctly."""
        skel = Skeleton()
        depth = 100
        skel.add_bone("bone_0")
        for i in range(1, depth):
            skel.add_bone(f"bone_{i}", parent_name=f"bone_{i-1}")

        assert skel.bone_count == depth
        assert skel.get_depth(f"bone_{depth-1}") == depth - 1

        # Chain should work
        chain = skel.get_chain("bone_0", f"bone_{depth-1}")
        assert len(chain) == depth

    def test_wide_hierarchy(self) -> None:
        """Wide hierarchies (many children per bone) work correctly."""
        skel = Skeleton()
        skel.add_bone("root")
        for i in range(100):
            skel.add_bone(f"child_{i}", parent_name="root")

        assert skel.bone_count == 101
        root = skel.get_bone("root")
        assert len(root.children) == 100

    def test_single_bone_skeleton(self) -> None:
        """Single bone skeleton is valid."""
        skel = Skeleton()
        skel.add_bone("only_bone")

        assert skel.is_valid() is True
        assert skel.bone_count == 1
        assert len(skel.root_bones) == 1

    def test_bind_pose_storage_and_retrieval(self) -> None:
        """Bind poses are stored and retrieved correctly."""
        skel = Skeleton()
        pose1 = Transform(translation=Vec3(1, 0, 0))
        pose2 = Transform(translation=Vec3(0, 1, 0))

        skel.add_bone("root", bind_pose=pose1)
        skel.add_bone("child", parent_name="root", bind_pose=pose2)

        assert skel.get_bone("root").bind_pose.translation.x == pytest.approx(1.0)
        assert skel.get_bone("child").bind_pose.translation.y == pytest.approx(1.0)

    def test_bones_from_flat_definition(self) -> None:
        """add_bones_from_flat builds hierarchy from flat list."""
        skel = Skeleton()
        definitions = [
            ("root", None),
            ("spine", "root"),
            ("chest", "spine"),
            ("left_arm", "chest"),
            ("right_arm", "chest"),
        ]
        skel.add_bones_from_flat(definitions)

        assert skel.bone_count == 5
        assert skel.get_parent_name("chest") == "spine"
        assert set(skel.get_child_names("chest")) == {"left_arm", "right_arm"}

    def test_bones_from_flat_with_bind_poses(self) -> None:
        """add_bones_from_flat applies bind poses."""
        skel = Skeleton()
        definitions = [("root", None), ("child", "root")]
        poses = {
            "root": Transform(translation=Vec3(0, 0, 0)),
            "child": Transform(translation=Vec3(1, 2, 3)),
        }
        skel.add_bones_from_flat(definitions, bind_poses=poses)

        child_pose = skel.get_bone("child").bind_pose
        assert child_pose.translation.x == pytest.approx(1.0)
        assert child_pose.translation.y == pytest.approx(2.0)
        assert child_pose.translation.z == pytest.approx(3.0)


# =============================================================================
# SECTION 11: Skeleton Copy and Container Operations
# =============================================================================


class TestSkeletonCopyAndContainerOps:
    """Tests for skeleton copy and container protocol."""

    def test_skeleton_copy(self) -> None:
        """copy() creates independent deep copy."""
        skel = Skeleton("original")
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")

        copy = skel.copy()

        assert copy.name == "original"
        assert copy.bone_count == 2
        assert copy is not skel
        assert copy.get_bone("root") is not skel.get_bone("root")

    def test_skeleton_copy_with_new_name(self) -> None:
        """copy() can rename the skeleton."""
        skel = Skeleton("original")
        skel.add_bone("root")

        copy = skel.copy(name="renamed")
        assert copy.name == "renamed"

    def test_skeleton_contains(self) -> None:
        """__contains__ checks bone existence."""
        skel = Skeleton()
        skel.add_bone("root")

        assert "root" in skel
        assert "nonexistent" not in skel

    def test_skeleton_len(self) -> None:
        """__len__ returns bone count."""
        skel = Skeleton()
        assert len(skel) == 0

        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")
        assert len(skel) == 2

    def test_skeleton_iter(self) -> None:
        """__iter__ iterates over bones."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")

        bone_names = {b.name for b in skel}
        assert bone_names == {"root", "child"}

    def test_skeleton_repr(self) -> None:
        """__repr__ shows name and bone count."""
        skel = Skeleton("test_skeleton")
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")

        repr_str = repr(skel)
        assert "test_skeleton" in repr_str
        assert "2" in repr_str


# =============================================================================
# SECTION 12: Realistic Animation Skeleton Tests
# =============================================================================


class TestRealisticSkeletons:
    """Tests with realistic skeleton configurations."""

    def test_humanoid_skeleton(self) -> None:
        """Build and query a humanoid skeleton."""
        skel = Skeleton("humanoid")

        # Build spine
        skel.add_bone("hips")
        skel.add_bone("spine_01", parent_name="hips")
        skel.add_bone("spine_02", parent_name="spine_01")
        skel.add_bone("spine_03", parent_name="spine_02")
        skel.add_bone("neck", parent_name="spine_03")
        skel.add_bone("head", parent_name="neck")

        # Build arms
        for side in ["left", "right"]:
            skel.add_bone(f"{side}_shoulder", parent_name="spine_03")
            skel.add_bone(f"{side}_arm", parent_name=f"{side}_shoulder")
            skel.add_bone(f"{side}_forearm", parent_name=f"{side}_arm")
            skel.add_bone(f"{side}_hand", parent_name=f"{side}_forearm")

        # Build legs
        for side in ["left", "right"]:
            skel.add_bone(f"{side}_thigh", parent_name="hips")
            skel.add_bone(f"{side}_shin", parent_name=f"{side}_thigh")
            skel.add_bone(f"{side}_foot", parent_name=f"{side}_shin")

        assert skel.bone_count == 20
        assert skel.is_valid() is True

        # Test IK chain for left arm
        arm_chain = skel.get_chain("left_shoulder", "left_hand")
        assert len(arm_chain) == 4
        assert [b.name for b in arm_chain] == [
            "left_shoulder",
            "left_arm",
            "left_forearm",
            "left_hand",
        ]

        # Test cross-body chain (left hand to right hand)
        cross_chain = skel.get_chain("left_hand", "right_hand")
        assert len(cross_chain) == 9  # left_hand -> ... -> spine_03 -> ... -> right_hand

    def test_spider_skeleton(self) -> None:
        """Build a spider skeleton with multiple roots (each leg is a root)."""
        skel = Skeleton("spider_legs")

        # Each leg is an independent kinematic chain
        for leg in range(8):
            skel.add_bone(f"leg_{leg}_hip")
            skel.add_bone(f"leg_{leg}_femur", parent_name=f"leg_{leg}_hip")
            skel.add_bone(f"leg_{leg}_tibia", parent_name=f"leg_{leg}_femur")
            skel.add_bone(f"leg_{leg}_foot", parent_name=f"leg_{leg}_tibia")

        assert skel.bone_count == 32
        assert len(skel.root_bones) == 8
        assert skel.is_valid() is True

        # IK chain within one leg
        chain = skel.get_chain("leg_0_hip", "leg_0_foot")
        assert len(chain) == 4

        # No chain between legs (different roots)
        cross_chain = skel.get_chain("leg_0_foot", "leg_1_foot")
        assert cross_chain == []

    def test_finger_skeleton(self) -> None:
        """Build a detailed finger skeleton for hand animation."""
        skel = Skeleton("hand")
        skel.add_bone("wrist")

        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            skel.add_bone(f"{finger}_meta", parent_name="wrist")
            skel.add_bone(f"{finger}_proximal", parent_name=f"{finger}_meta")
            skel.add_bone(f"{finger}_middle", parent_name=f"{finger}_proximal")
            skel.add_bone(f"{finger}_distal", parent_name=f"{finger}_middle")

        assert skel.bone_count == 21  # wrist + 5 fingers * 4 bones
        assert skel.is_valid() is True

        # Index finger IK chain
        chain = skel.get_chain("index_meta", "index_distal")
        assert len(chain) == 4

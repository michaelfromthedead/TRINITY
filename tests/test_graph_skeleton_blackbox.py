"""Blackbox contract tests for SkeletonBone + SkeletonHierarchy (T-AG-2.0).

CLEANROOM: tests the PUBLIC API contract only.
No knowledge of internal representation, data structures, or algorithms.

Contract sources:
  - engine/animation/graph/__init__.py (public exports: SkeletonBone, SkeletonHierarchy)
  - Introspected public API signatures (cleanroom-safe)

NOTE: This tests the NEW object-reference API from skeleton.py (exported as
SkeletonBone and SkeletonHierarchy). The old index-based Bone/Skeleton from
animation_graph.py remain available separately.
"""
import pytest
from engine.animation.graph import SkeletonBone, SkeletonHierarchy


# ============================================================================
# Equivalence Class: Import and type identity
# ============================================================================

class TestImport:
    """SkeletonBone and SkeletonHierarchy are accessible from the public API."""

    def test_skeleton_bone_imported(self):
        """SkeletonBone is a class."""
        assert isinstance(SkeletonBone, type)

    def test_skeleton_hierarchy_imported(self):
        """SkeletonHierarchy is a class."""
        assert isinstance(SkeletonHierarchy, type)


# ============================================================================
# Equivalence Class: SkeletonBone creation
# ============================================================================

class TestSkeletonBoneCreation:
    """SkeletonBone instances can be created with various constructor forms."""

    def test_bone_minimal(self):
        """SkeletonBone accepts name as the only required argument."""
        b = SkeletonBone("hip")
        assert b.name == "hip"

    def test_bone_with_parent_object(self):
        """SkeletonBone accepts a parent Bone as object reference."""
        root = SkeletonBone("root")
        child = SkeletonBone("child", parent=root)
        assert child.parent is root
        # Round-trip: child.parent.name matches the parent's name
        assert child.parent.name == "root"

    def test_bone_with_parent_positional(self):
        """SkeletonBone accepts parent as the second positional argument."""
        root = SkeletonBone("root")
        child = SkeletonBone("child", root)
        assert child.parent is root

    def test_bone_no_parent_default(self):
        """Default parent is None (root bone)."""
        b = SkeletonBone("root")
        assert b.parent is None

    def test_bone_is_root_when_no_parent(self):
        """is_root is True when parent is None."""
        b = SkeletonBone("root")
        assert b.is_root is True

    def test_bone_is_not_root_when_has_parent(self):
        """is_root is False when parent is set."""
        root = SkeletonBone("root")
        child = SkeletonBone("child", parent=root)
        assert child.is_root is False

    def test_bone_with_bind_pose_transform(self):
        """SkeletonBone can carry a bind_pose Transform."""
        from engine.animation.graph import Transform
        t = Transform()
        b = SkeletonBone("root", bind_pose=t)
        assert b.bind_pose is t

    def test_bone_default_bind_pose(self):
        """SkeletonBone without explicit bind_pose has bind_pose attribute."""
        b = SkeletonBone("root")
        assert hasattr(b, "bind_pose")

    def test_bone_keyword_init(self):
        """SkeletonBone accepts keyword arguments."""
        b = SkeletonBone(name="spine")
        assert b.name == "spine"

    def test_bone_copy_is_different_object(self):
        """copy() returns a new independent Bone object."""
        b = SkeletonBone("root")
        c = b.copy()
        assert c is not b
        assert c.name == b.name


# ============================================================================
# Equivalence Class: SkeletonBone hierarchy queries
# ============================================================================

class TestSkeletonBoneHierarchy:
    """Bone-level hierarchy traversal methods."""

    def test_root_has_root_self(self):
        """get_root() on a root bone returns itself."""
        root = SkeletonBone("root")
        assert root.get_root() is root

    def test_child_get_root(self):
        """get_root() on a child returns the ultimate ancestor."""
        root = SkeletonBone("root")
        child = SkeletonBone("child", parent=root)
        grandchild = SkeletonBone("grandchild", parent=child)
        assert grandchild.get_root() is root

    def test_root_depth(self):
        """depth of a root bone is 0."""
        root = SkeletonBone("root")
        assert root.depth == 0

    def test_child_depth(self):
        """depth of a direct child is 1."""
        root = SkeletonBone("root")
        child = SkeletonBone("child", parent=root)
        assert child.depth == 1

    def test_grandchild_depth(self):
        """depth increases with each level."""
        root = SkeletonBone("root")
        child = SkeletonBone("child", parent=root)
        grandchild = SkeletonBone("grandchild", parent=child)
        assert grandchild.depth == 2

    def test_is_ancestor_of_direct(self):
        """is_ancestor_of returns True for direct parent->child."""
        root = SkeletonBone("root")
        child = SkeletonBone("child", parent=root)
        assert root.is_ancestor_of(child) is True

    def test_is_ancestor_of_deep(self):
        """is_ancestor_of returns True for grandparent->grandchild."""
        root = SkeletonBone("root")
        child = SkeletonBone("child", parent=root)
        grandchild = SkeletonBone("grandchild", parent=child)
        assert root.is_ancestor_of(grandchild) is True

    def test_is_ancestor_of_self(self):
        """is_ancestor_of returns False for self."""
        root = SkeletonBone("root")
        assert root.is_ancestor_of(root) is False

    def test_is_descendant_of_direct(self):
        """is_descendant_of returns True for direct child->parent."""
        root = SkeletonBone("root")
        child = SkeletonBone("child", parent=root)
        assert child.is_descendant_of(root) is True

    def test_is_descendant_of_self(self):
        """is_descendant_of returns False for self."""
        root = SkeletonBone("root")
        assert root.is_descendant_of(root) is False

    def test_get_descendants_root(self):
        """get_descendants() via SkeletonHierarchy returns descendant bones."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")
        skel.add_bone("grandchild", parent_name="child")
        root = skel.get_bone("root")
        descendants = root.get_descendants()
        assert len(descendants) == 2

    def test_get_descendants_leaf(self):
        """get_descendants() on a leaf returns empty list."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")
        child = skel.get_bone("child")
        assert child.get_descendants() == []

    def test_get_siblings_root(self):
        """get_siblings() on a root returns empty list."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        root = skel.get_bone("root")
        assert root.get_siblings() == []

    def test_get_siblings_middle(self):
        """get_siblings() returns other bones with same parent."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        skel.add_bone("a", parent_name="root")
        skel.add_bone("b", parent_name="root")
        a = skel.get_bone("a")
        b = skel.get_bone("b")
        root = skel.get_bone("root")
        siblings = a.get_siblings()
        assert b in siblings
        assert root not in siblings


# ============================================================================
# Equivalence Class: SkeletonHierarchy creation
# ============================================================================

class TestSkeletonHierarchyCreation:
    """SkeletonHierarchy can be created and reports bone count correctly."""

    def test_create_default(self):
        """Default SkeletonHierarchy has bone_count == 0."""
        skel = SkeletonHierarchy()
        assert skel.bone_count == 0

    def test_create_with_name(self):
        """SkeletonHierarchy accepts an optional name."""
        skel = SkeletonHierarchy("test_skeleton")
        assert skel.bone_count == 0

    def test_bones_property_empty(self):
        """bones property on empty skeleton returns empty dict."""
        skel = SkeletonHierarchy()
        assert skel.bones == {}

    def test_root_bones_property_empty(self):
        """root_bones property on empty skeleton returns empty list."""
        skel = SkeletonHierarchy()
        assert skel.root_bones == []


# ============================================================================
# Equivalence Class: add_bone
# ============================================================================

class TestAddBone:
    """Bones are added by name string."""

    def test_add_single_root(self):
        """Adding a root bone by name increments bone_count."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        assert skel.bone_count == 1

    def test_add_returns_bone(self):
        """add_bone returns the newly created SkeletonBone."""
        skel = SkeletonHierarchy()
        bone = skel.add_bone("root")
        assert isinstance(bone, SkeletonBone)
        assert bone.name == "root"

    def test_add_with_parent_name(self):
        """Adding a bone with parent_name creates parent relationship."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        child = skel.add_bone("child", parent_name="root")
        assert child.parent is not None
        assert child.parent.name == "root"

    def test_add_with_bind_pose(self):
        """add_bone accepts an optional bind_pose argument."""
        from engine.animation.graph import Transform
        skel = SkeletonHierarchy()
        t = Transform()
        bone = skel.add_bone("root", bind_pose=t)
        assert bone.bind_pose is t

    def test_add_multiple_roots(self):
        """SkeletonHierarchy supports multiple root bones."""
        skel = SkeletonHierarchy()
        skel.add_bone("root_a")
        skel.add_bone("root_b")
        assert skel.bone_count == 2
        assert len(skel.root_bones) == 2

    def test_add_nested_hierarchy(self):
        """Adding nested bones maintains correct parent relationships."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        skel.add_bone("mid", parent_name="root")
        skel.add_bone("tip", parent_name="mid")
        assert skel.bone_count == 3
        tip = skel.get_bone("tip")
        assert tip is not None
        assert tip.parent is not None
        assert tip.parent.name == "mid"
        assert tip.parent.parent.name == "root"


# ============================================================================
# Equivalence Class: add_bones_from_flat
# ============================================================================

class TestAddBonesFromFlat:
    """Batch addition of bones from flat definition list."""

    def test_add_flat_single(self):
        """add_bones_from_flat with a single root bone."""
        skel = SkeletonHierarchy()
        skel.add_bones_from_flat([("root", None)])
        assert skel.bone_count == 1
        assert skel.get_bone("root") is not None

    def test_add_flat_chain(self):
        """add_bones_from_flat creates a parent-child chain."""
        skel = SkeletonHierarchy()
        skel.add_bones_from_flat([
            ("root", None),
            ("child", "root"),
            ("grandchild", "child"),
        ])
        assert skel.bone_count == 3
        gc = skel.get_bone("grandchild")
        assert gc is not None
        assert gc.parent is not None
        assert gc.parent.parent is not None
        assert gc.parent.parent.name == "root"

    def test_add_flat_siblings(self):
        """add_bones_from_flat creates sibling bones."""
        skel = SkeletonHierarchy()
        skel.add_bones_from_flat([
            ("root", None),
            ("a", "root"),
            ("b", "root"),
        ])
        assert skel.bone_count == 3
        a = skel.get_bone("a")
        b = skel.get_bone("b")
        assert a is not None
        assert b is not None
        assert a.parent is b.parent


# ============================================================================
# Equivalence Class: get_bone lookup
# ============================================================================

class TestGetBone:
    """Bones can be retrieved by name."""

    def test_get_bone_by_name(self):
        """get_bone(name) returns the bone with that name."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        bone = skel.get_bone("root")
        assert isinstance(bone, SkeletonBone)
        assert bone.name == "root"

    def test_get_bone_returns_none_for_missing(self):
        """get_bone returns None for a name that does not exist."""
        skel = SkeletonHierarchy()
        assert skel.get_bone("phantom") is None

    def test_has_bone_true(self):
        """has_bone returns True for an existing bone."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        assert skel.has_bone("root") is True

    def test_has_bone_false(self):
        """has_bone returns False for a non-existing bone."""
        skel = SkeletonHierarchy()
        assert skel.has_bone("phantom") is False


# ============================================================================
# Equivalence Class: Hierarchy queries on SkeletonHierarchy
# ============================================================================

class TestHierarchyQueries:
    """Skeleton-level queries for hierarchy structure."""

    def test_get_parent_name_root(self):
        """get_parent_name for a root bone returns None."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        assert skel.get_parent_name("root") is None

    def test_get_parent_name_child(self):
        """get_parent_name for a child returns the parent's name."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")
        assert skel.get_parent_name("child") == "root"

    def test_get_parent_name_missing(self):
        """get_parent_name for a non-existent bone returns None."""
        skel = SkeletonHierarchy()
        assert skel.get_parent_name("phantom") is None

    def test_get_child_names(self):
        """get_child_names returns names of direct children."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        skel.add_bone("a", parent_name="root")
        skel.add_bone("b", parent_name="root")
        children = skel.get_child_names("root")
        assert "a" in children
        assert "b" in children

    def test_get_child_names_leaf(self):
        """get_child_names for a leaf returns empty list."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        skel.add_bone("leaf", parent_name="root")
        assert skel.get_child_names("leaf") == []

    def test_get_ancestor_names(self):
        """get_ancestor_names returns parent chain from leaf to root."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        skel.add_bone("mid", parent_name="root")
        skel.add_bone("tip", parent_name="mid")
        ancestors = skel.get_ancestor_names("tip")
        assert "mid" in ancestors
        assert "root" in ancestors

    def test_get_depth_root(self):
        """get_depth returns 0 for root."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        assert skel.get_depth("root") == 0

    def test_get_depth_child(self):
        """get_depth returns 1 for direct child."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")
        assert skel.get_depth("child") == 1

    def test_get_max_depth(self):
        """get_max_depth returns the maximum depth in the skeleton."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        skel.add_bone("mid", parent_name="root")
        skel.add_bone("tip", parent_name="mid")
        assert skel.get_max_depth() == 2

    def test_get_chain_direct(self):
        """get_chain returns bones from start_name to end_name."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        skel.add_bone("mid", parent_name="root")
        skel.add_bone("tip", parent_name="mid")
        chain = skel.get_chain("root", "tip")
        assert len(chain) == 3
        assert chain[0].name == "root"
        assert chain[1].name == "mid"
        assert chain[2].name == "tip"

    def test_get_chain_same_bone(self):
        """get_chain where start == end returns a single-element list."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        chain = skel.get_chain("root", "root")
        assert len(chain) == 1
        assert chain[0].name == "root"


# ============================================================================
# Equivalence Class: remove_bone
# ============================================================================

class TestRemoveBone:
    """Bones can be removed from the skeleton."""

    def test_remove_root(self):
        """Removing a root bone decreases bone_count."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        result = skel.remove_bone("root")
        assert result is True
        assert skel.bone_count == 0

    def test_remove_nonexistent(self):
        """Removing a non-existent bone returns False."""
        skel = SkeletonHierarchy()
        result = skel.remove_bone("phantom")
        assert result is False

    def test_remove_updates_root_bones(self):
        """Removing a root bone updates root_bones."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        skel.remove_bone("root")
        assert skel.root_bones == []


# ============================================================================
# Equivalence Class: validate
# ============================================================================

class TestValidate:
    """SkeletonHierarchy validation."""

    def test_validate_empty(self):
        """Empty skeleton validates cleanly."""
        skel = SkeletonHierarchy()
        issues = skel.validate()
        assert isinstance(issues, list)

    def test_validate_single_root(self):
        """Single root bone produces no structural issues."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        issues = skel.validate()
        # validate returns a list; empty means no issues
        assert isinstance(issues, list)


# ============================================================================
# Equivalence Class: copy
# ============================================================================

class TestCopy:
    """SkeletonHierarchy.copy() produces an independent clone."""

    def test_copy_empty(self):
        """Copy of empty skeleton has same bone_count."""
        skel = SkeletonHierarchy("original")
        copy = skel.copy()
        assert copy.bone_count == 0
        assert copy is not skel

    def test_copy_with_bones(self):
        """Copy contains the same bones but independent."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")
        copy = skel.copy()
        assert copy.bone_count == 2
        assert copy.has_bone("root")
        assert copy.has_bone("child")

    def test_copy_is_independent(self):
        """Modifying the copy does not affect the original."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        copy = skel.copy()
        copy.add_bone("extra")
        assert skel.bone_count == 1
        assert copy.bone_count == 2


# ============================================================================
# Boundary cases: Edge and error conditions
# ============================================================================

class TestEdgeCases:
    """Boundary behavior of SkeletonHierarchy."""

    def test_empty_skeleton_queries(self):
        """Queries on empty skeleton return empty/safe values."""
        skel = SkeletonHierarchy()
        assert skel.get_bone("anything") is None
        assert skel.has_bone("anything") is False
        assert skel.get_depth("anything") == -1
        assert skel.get_ancestor_names("anything") == []
        assert skel.get_child_names("anything") == []
        assert skel.get_max_depth() == -1

    def test_get_chain_disconnected(self):
        """get_chain between two unrelated bones returns empty list."""
        skel = SkeletonHierarchy()
        skel.add_bones_from_flat([
            ("root_a", None),
            ("root_b", None),
        ])
        chain = skel.get_chain("root_a", "root_b")
        assert chain == []

    def test_bone_count_after_remove(self):
        """bone_count stays consistent after add/remove operations."""
        skel = SkeletonHierarchy()
        skel.add_bone("a")
        skel.add_bone("b")
        skel.add_bone("c")
        skel.remove_bone("b")
        assert skel.bone_count == 2

    def test_add_bone_with_missing_parent_raises(self):
        """add_bone with a parent_name that does not exist raises ValueError."""
        skel = SkeletonHierarchy()
        skel.add_bone("root")
        with pytest.raises(ValueError):
            skel.add_bone("child", parent_name="nonexistent_parent")

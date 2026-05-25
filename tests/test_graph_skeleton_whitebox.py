"""WHITEBOX tests for engine/animation/graph/skeleton.py.

WHITEBOX coverage plan:
  [Bone]
    Path A1:  __init__ empty name raises ValueError
    Path A2:  __init__ with parent link
    Path A3:  __init__ without parent (root)
    Path A4:  __init__ default bind_pose is identity when None
    Path B1:  is_root property -- root bone returns True
    Path B2:  is_root property -- child bone returns False
    Path C1:  depth property -- root returns 0
    Path C2:  depth property -- depth-1 child
    Path C3:  depth property -- deep chain (depth N)
    Path D1:  get_root -- already root returns self
    Path D2:  get_root -- leaf walks to ultimate root
    Path E1:  get_siblings -- root bone returns empty list
    Path E2:  get_siblings -- only child returns empty list
    Path E3:  get_siblings -- multiple siblings
    Path F1:  is_ancestor_of -- direct parent of child is True
    Path F2:  is_ancestor_of -- grandparent of grandchild is True
    Path F3:  is_ancestor_of -- unrelated bones returns False
    Path F4:  is_ancestor_of -- same bone as self returns False
    Path G1:  is_descendant_of -- delegates correctly
    Path H1:  get_descendants -- no children returns empty
    Path H2:  get_descendants -- BFS order in multi-branch hierarchy
    Path I1:  copy -- deep copy is independent (mutations isolated)
    Path J1:  __repr__ -- root bone format
    Path J2:  __repr__ -- child bone with parent name

  [Skeleton]
    Path K1:  __init__ empty name raises ValueError
    Path K2:  __init__ default name "skeleton"
    Path K3:  __init__ custom name
    Path L1:  bone_count property -- empty returns 0
    Path L2:  bone_count property -- returns len(_bones)
    Path M1:  bones property -- returns dict copy (mutation-safe)
    Path N1:  root_bones property -- returns list copy (mutation-safe)
    Path O1:  add_bone -- root bone
    Path O2:  add_bone -- child of existing bone
    Path O3:  add_bone -- duplicate name raises ValueError
    Path O4:  add_bone -- missing parent raises ValueError
    Path O5:  add_bone -- custom bind_pose
    Path P1:  remove_bone -- leaf bone (no children)
    Path P2:  remove_bone -- root with children (re-parent to roots)
    Path P3:  remove_bone -- intermediate bone (re-parent to grandparent)
    Path P4:  remove_bone -- non-existent returns False
    Path P5:  remove_bone -- bone with multiple children re-parented
    Path P6:  remove_bone -- only root removed, skeleton empty
    Path Q1:  has_bone -- exists returns True
    Path Q2:  has_bone -- missing returns False
    Path R1:  get_bone -- exists returns Bone
    Path R2:  get_bone -- missing returns None
    Path S1:  get_child_names -- bone has children
    Path S2:  get_child_names -- leaf bone returns empty
    Path S3:  get_child_names -- missing bone returns []
    Path T1:  get_parent_name -- root returns None
    Path T2:  get_parent_name -- child returns parent name
    Path T3:  get_parent_name -- missing bone returns None
    Path U1:  get_ancestor_names -- root returns [name]
    Path U2:  get_ancestor_names -- deep chain [self, ..., root]
    Path U3:  get_ancestor_names -- missing bone returns []
    Path V1:  get_depth -- root returns 0
    Path V2:  get_depth -- deep hierarchy returns N
    Path V3:  get_depth -- missing bone returns -1
    Path W1:  get_max_depth -- empty returns -1
    Path W2:  get_max_depth -- single root returns 0
    Path W3:  get_max_depth -- multi-tree different depths
    Path X1:  get_chain -- start == end returns [bone]
    Path X2:  get_chain -- direct parent-child
    Path X3:  get_chain -- distant ancestor-descendant
    Path X4:  get_chain -- end is ancestor of start (reverse)
    Path X5:  get_chain -- disconnected trees returns []
    Path X6:  get_chain -- start missing returns []
    Path X7:  get_chain -- end missing returns []
    Path X8:  get_chain -- cousins sharing common ancestor
    Path Y1:  validate -- empty skeleton returns error
    Path Y2:  validate -- valid single root returns []
    Path Y3:  validate -- no root bones (all have parents) returns error
    Path Y4:  validate -- orphan bone not reachable from root
    Path Y5:  validate -- parent back-ref to unregistered bone
    Path Y6:  validate -- child back-ref to unregistered bone
    Path Y7:  validate -- cycle in parent references
    Path Z1:  add_bones_from_flat -- simple batch add
    Path Z2:  add_bones_from_flat -- with bind_poses dict
    Path Z3:  copy -- deep copy independent from original
    Path AA1: __contains__ -- bone exists True
    Path AA2: __contains__ -- bone missing False
    Path AB1: __len__ -- returns bone count
    Path AC1: __iter__ -- yields Bone objects
    Path AD1: __repr__ -- format string
"""

from __future__ import annotations

import pytest
from copy import deepcopy

from engine.animation.graph.skeleton import Bone, Skeleton
from engine.core.math.transform import Transform
from engine.core.math import Vec3, Quat


# =========================================================================
# Helpers
# =========================================================================

def _transform_equal(a: Transform, b: Transform) -> bool:
    """Compare two Transforms field-by-field (C extension, no __eq__)."""
    return (
        a.translation.x == b.translation.x
        and a.translation.y == b.translation.y
        and a.translation.z == b.translation.z
        and a.rotation.x == b.rotation.x
        and a.rotation.y == b.rotation.y
        and a.rotation.z == b.rotation.z
        and a.rotation.w == b.rotation.w
        and a.scale.x == b.scale.x
        and a.scale.y == b.scale.y
        and a.scale.z == b.scale.z
    )


def _make_transform(
    tx: float = 0.0, ty: float = 0.0, tz: float = 0.0,
    rx: float = 0.0, ry: float = 0.0, rz: float = 0.0, rw: float = 1.0,
    sx: float = 1.0, sy: float = 1.0, sz: float = 1.0,
) -> Transform:
    """Construct a Transform with explicit numeric fields."""
    return Transform(
        Vec3(tx, ty, tz),
        Quat(rx, ry, rz, rw),
        Vec3(sx, sy, sz),
    )


def _make_chain(names: list[str]) -> Skeleton:
    """Build a linear chain skeleton: first name is root, each subsequent is child of previous."""
    skel = Skeleton("chain")
    parent = None
    for name in names:
        skel.add_bone(name, parent_name=parent)
        parent = name
    return skel


# =========================================================================
# Bone: __init__
# =========================================================================

class TestBoneInit:
    """Every branch in Bone.__init__."""

    def test_empty_name_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Bone("")

    def test_whitespace_name_passes(self) -> None:
        """Whitespace is technically non-empty -- only empty string is rejected."""
        bone = Bone("  ")
        assert bone.name == "  "

    def test_with_parent_link(self) -> None:
        parent = Bone("parent")
        child = Bone("child", parent=parent)
        assert child.parent is parent
        assert parent.children == []  # children list NOT updated by Bone.__init__

    def test_without_parent_is_root(self) -> None:
        bone = Bone("root_bone")
        assert bone.parent is None
        assert bone.is_root is True

    def test_default_bind_pose_is_identity(self) -> None:
        bone = Bone("test")
        assert _transform_equal(bone.bind_pose, Transform.identity())

    def test_custom_bind_pose(self) -> None:
        pose = _make_transform(tx=1, ty=2, tz=3)
        bone = Bone("test", bind_pose=pose)
        assert bone.bind_pose is pose


# =========================================================================
# Bone: is_root
# =========================================================================

class TestBoneIsRoot:
    def test_root_returns_true(self) -> None:
        bone = Bone("root")
        assert bone.is_root is True

    def test_child_returns_false(self) -> None:
        parent = Bone("parent")
        child = Bone("child", parent=parent)
        assert child.is_root is False


# =========================================================================
# Bone: depth
# =========================================================================

class TestBoneDepth:
    def test_root_depth_zero(self) -> None:
        bone = Bone("root")
        assert bone.depth == 0

    def test_depth_one_child(self) -> None:
        parent = Bone("parent")
        child = Bone("child", parent=parent)
        assert child.depth == 1

    def test_deep_chain_depth(self) -> None:
        """a -> b -> c -> d: d depth = 3."""
        a = Bone("a")
        b = Bone("b", parent=a)
        c = Bone("c", parent=b)
        d = Bone("d", parent=c)
        assert d.depth == 3
        assert c.depth == 2
        assert b.depth == 1
        assert a.depth == 0

    def test_depth_after_reparent(self) -> None:
        """Re-parenting via direct assignment should update depth."""
        grandparent = Bone("gp")
        parent = Bone("parent", parent=grandparent)
        child = Bone("child", parent=parent)
        assert child.depth == 2
        # Re-parent child directly to grandparent
        child.parent = grandparent
        assert child.depth == 1  # Now depth 1


# =========================================================================
# Bone: get_root
# =========================================================================

class TestBoneGetRoot:
    def test_already_root_returns_self(self) -> None:
        bone = Bone("root")
        assert bone.get_root() is bone

    def test_leaf_returns_ultimate_root(self) -> None:
        a = Bone("a")
        b = Bone("b", parent=a)
        c = Bone("c", parent=b)
        assert c.get_root() is a
        assert b.get_root() is a

    def test_disconnected_bone_returns_self(self) -> None:
        bone = Bone("lonely")
        assert bone.get_root() is bone


# =========================================================================
# Bone: get_siblings
# =========================================================================

class TestBoneGetSiblings:
    def test_root_returns_empty(self) -> None:
        bone = Bone("root")
        assert bone.get_siblings() == []

    def test_only_child_returns_empty(self) -> None:
        parent = Bone("parent")
        child = Bone("child")
        parent.children.append(child)
        child.parent = parent
        assert child.get_siblings() == []

    def test_multiple_siblings_excludes_self(self) -> None:
        parent = Bone("parent")
        a = Bone("a", parent=parent)
        b = Bone("b", parent=parent)
        c = Bone("c", parent=parent)
        parent.children = [a, b, c]  # manually link
        for ch in (a, b, c):
            ch.parent = parent

        siblings_a = a.get_siblings()
        assert len(siblings_a) == 2
        assert b in siblings_a
        assert c in siblings_a
        assert a not in siblings_a


# =========================================================================
# Bone: is_ancestor_of
# =========================================================================

class TestBoneIsAncestorOf:
    def test_direct_parent_of_child_is_true(self) -> None:
        parent = Bone("parent")
        child = Bone("child", parent=parent)
        parent.children.append(child)
        assert parent.is_ancestor_of(child) is True

    def test_grandparent_of_grandchild_is_true(self) -> None:
        gp = Bone("gp")
        p = Bone("p", parent=gp)
        c = Bone("c", parent=p)
        gp.children.append(p)
        p.children.append(c)
        assert gp.is_ancestor_of(c) is True

    def test_unrelated_bones_returns_false(self) -> None:
        a = Bone("a")
        b = Bone("b")
        assert a.is_ancestor_of(b) is False

    def test_same_bone_returns_false(self) -> None:
        """Ancestor check starts from other.parent, so self==other returns False."""
        bone = Bone("x")
        assert bone.is_ancestor_of(bone) is False

    def test_deep_walk_ancestor(self) -> None:
        a = Bone("a")
        b = Bone("b", parent=a)
        c = Bone("c", parent=b)
        d = Bone("d", parent=c)
        a.children.append(b)
        b.children.append(c)
        c.children.append(d)
        assert a.is_ancestor_of(d) is True
        assert b.is_ancestor_of(d) is True
        assert c.is_ancestor_of(d) is True
        assert d.is_ancestor_of(a) is False


# =========================================================================
# Bone: is_descendant_of
# =========================================================================

class TestBoneIsDescendantOf:
    def test_delegates_to_is_ancestor_of(self) -> None:
        ancestor = Bone("anc")
        descendant = Bone("desc", parent=ancestor)
        ancestor.children.append(descendant)
        assert descendant.is_descendant_of(ancestor) is True
        assert ancestor.is_descendant_of(descendant) is False


# =========================================================================
# Bone: get_descendants
# =========================================================================

class TestBoneGetDescendants:
    def test_no_children_returns_empty(self) -> None:
        bone = Bone("leaf")
        assert bone.get_descendants() == []

    def test_single_child(self) -> None:
        parent = Bone("parent")
        child = Bone("child")
        parent.children.append(child)
        child.parent = parent
        descendants = parent.get_descendants()
        assert len(descendants) == 1
        assert descendants[0] is child

    def test_bfs_order_multi_branch(self) -> None:
        """BFS: breadth before depth.
           root -> [a, b], a -> [c, d], b -> [e]
           Expected BFS: a, b, c, d, e
        """
        root = Bone("root")
        a = Bone("a")
        b = Bone("b")
        c = Bone("c")
        d = Bone("d")
        e = Bone("e")
        root.children = [a, b]
        a.children = [c, d]
        b.children = [e]
        for child in root.children:
            child.parent = root
        for child in a.children:
            child.parent = a
        for child in b.children:
            child.parent = b

        descendants = root.get_descendants()
        names = [b.name for b in descendants]
        assert names == ["a", "b", "c", "d", "e"], f"Expected BFS order, got {names}"

    def test_leaf_returns_empty(self) -> None:
        root = Bone("root")
        child = Bone("child")
        root.children.append(child)
        child.parent = root
        assert child.get_descendants() == []


# =========================================================================
# Bone: copy
# =========================================================================

class TestBoneCopy:
    def test_deep_copy_independence(self) -> None:
        parent = Bone("parent")
        child = Bone("child", parent=parent)
        parent.children.append(child)

        copied = parent.copy()
        assert copied.name == parent.name
        assert copied is not parent
        assert copied.children[0] is not child
        assert copied.children[0].name == "child"
        # Mutate original -- copy should be unaffected
        parent.name = "parent_modified"
        parent.children[0].name = "child_modified"
        assert copied.name == "parent"
        assert copied.children[0].name == "child"

    def test_copy_single_bone_no_children(self) -> None:
        bone = Bone("lonely")
        copied = bone.copy()
        assert copied.name == "lonely"
        assert copied is not bone
        assert copied.children == []


# =========================================================================
# Bone: __repr__
# =========================================================================

class TestBoneRepr:
    def test_root_repr(self) -> None:
        bone = Bone("root_bone")
        assert repr(bone) == "Bone('root_bone', root)"

    def test_child_repr_shows_parent(self) -> None:
        parent = Bone("parent")
        child = Bone("child", parent=parent)
        assert repr(child) == "Bone('child', parent=parent)"
        assert repr(parent) == "Bone('parent', root)"


# =========================================================================
# Skeleton: __init__
# =========================================================================

class TestSkeletonInit:
    def test_empty_name_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Skeleton("")

    def test_default_name(self) -> None:
        skel = Skeleton()
        assert skel.name == "skeleton"

    def test_custom_name(self) -> None:
        skel = Skeleton("humanoid")
        assert skel.name == "humanoid"

    def test_initial_state_empty(self) -> None:
        skel = Skeleton()
        assert skel.bone_count == 0
        assert skel._bones == {}
        assert skel._root_bones == []


# =========================================================================
# Skeleton: properties (read-only views)
# =========================================================================

class TestSkeletonProperties:
    def test_bone_count_empty(self) -> None:
        skel = Skeleton()
        assert skel.bone_count == 0

    def test_bone_count_with_bones(self) -> None:
        skel = Skeleton()
        skel.add_bone("a")
        skel.add_bone("b")
        assert skel.bone_count == 2

    def test_bones_returns_dict_copy(self) -> None:
        skel = Skeleton()
        skel.add_bone("a")
        view = skel.bones
        view["b"] = Bone("b")  # should not affect skeleton
        assert skel.bone_count == 1

    def test_root_bones_returns_list_copy(self) -> None:
        skel = Skeleton()
        skel.add_bone("a")
        view = skel.root_bones
        view.append(Bone("stray"))
        assert len(skel.root_bones) == 1


# =========================================================================
# Skeleton: add_bone
# =========================================================================

class TestSkeletonAddBone:
    def test_add_root_bone(self) -> None:
        skel = Skeleton()
        bone = skel.add_bone("root")
        assert bone.name == "root"
        assert bone.parent is None
        assert bone in skel._root_bones
        assert skel.bone_count == 1

    def test_add_child_bone(self) -> None:
        skel = Skeleton()
        parent = skel.add_bone("parent")
        child = skel.add_bone("child", parent_name="parent")
        assert child.parent is parent
        assert child in parent.children
        assert child not in skel._root_bones
        assert skel.bone_count == 2

    def test_duplicate_name_raises(self) -> None:
        skel = Skeleton()
        skel.add_bone("unique")
        with pytest.raises(ValueError, match="already exists"):
            skel.add_bone("unique")

    def test_missing_parent_raises(self) -> None:
        skel = Skeleton()
        with pytest.raises(ValueError, match="not found"):
            skel.add_bone("child", parent_name="nonexistent")

    def test_custom_bind_pose(self) -> None:
        skel = Skeleton()
        pose = _make_transform(tx=5, ty=10, tz=15)
        bone = skel.add_bone("root", bind_pose=pose)
        assert bone.bind_pose is pose

    def test_default_bind_pose_identity(self) -> None:
        skel = Skeleton()
        bone = skel.add_bone("root")
        assert _transform_equal(bone.bind_pose, Transform.identity())


# =========================================================================
# Skeleton: remove_bone
# =========================================================================

class TestSkeletonRemoveBone:
    def test_remove_leaf_bone(self) -> None:
        skel = _make_chain(["root", "child"])
        assert skel.remove_bone("child") is True
        assert "child" not in skel._bones
        assert skel.bone_count == 1
        # root should have no children
        root = skel.get_bone("root")
        assert root is not None
        assert root.children == []

    def test_remove_root_with_children_reparents_to_roots(self) -> None:
        """Removing the only root with children: children become new roots."""
        skel = Skeleton()
        root = skel.add_bone("root")
        child_a = skel.add_bone("child_a", parent_name="root")
        child_b = skel.add_bone("child_b", parent_name="root")

        skel.remove_bone("root")
        assert "root" not in skel._bones
        assert skel.bone_count == 2
        # children should now be roots
        assert len(skel._root_bones) == 2
        assert child_a.parent is None
        assert child_b.parent is None

    def test_remove_intermediate_bone_reparents_to_grandparent(self) -> None:
        """Removing intermediate: children are re-parented to grandparent.
           root -> mid -> child_a, child_b
           After removing mid: root -> child_a, child_b
        """
        skel = Skeleton()
        root = skel.add_bone("root")
        mid = skel.add_bone("mid", parent_name="root")
        child_a = skel.add_bone("child_a", parent_name="mid")
        child_b = skel.add_bone("child_b", parent_name="mid")

        skel.remove_bone("mid")
        assert "mid" not in skel._bones
        assert root.children == [child_a, child_b]
        assert child_a.parent is root
        assert child_b.parent is root
        assert skel.bone_count == 3

    def test_remove_nonexistent_returns_false(self) -> None:
        skel = Skeleton()
        skel.add_bone("a")
        assert skel.remove_bone("nonexistent") is False
        assert skel.bone_count == 1

    def test_remove_only_root_makes_empty(self) -> None:
        skel = Skeleton()
        skel.add_bone("only")
        assert skel.remove_bone("only") is True
        assert skel.bone_count == 0
        assert skel._root_bones == []

    def test_remove_bone_detaches_from_parent(self) -> None:
        """After removal, removed bone no longer appears in parent's children list."""
        skel = Skeleton()
        root = skel.add_bone("root")
        skel.add_bone("child", parent_name="root")
        skel.remove_bone("child")
        assert root.children == []

    def test_remove_intermediate_reparents_and_clears_children(self) -> None:
        """The removed bone's children list is cleared after re-parenting."""
        skel = Skeleton()
        root = skel.add_bone("root")
        mid = skel.add_bone("mid", parent_name="root")
        skel.add_bone("leaf", parent_name="mid")
        skel.remove_bone("mid")
        # mid should have no children after removal
        assert mid.children == []

    def test_reparent_multiple_children_on_root_removal(self) -> None:
        """Removing root with 3 children, all become independent roots."""
        skel = Skeleton()
        skel.add_bone("root")
        for i in range(3):
            skel.add_bone(f"child_{i}", parent_name="root")
        skel.remove_bone("root")
        assert len(skel._root_bones) == 3
        for root in skel._root_bones:
            assert root.parent is None
            assert root.children == []


# =========================================================================
# Skeleton: has_bone / get_bone
# =========================================================================

class TestSkeletonHasBone:
    def test_exists_returns_true(self) -> None:
        skel = Skeleton()
        skel.add_bone("exists")
        assert skel.has_bone("exists") is True

    def test_missing_returns_false(self) -> None:
        skel = Skeleton()
        assert skel.has_bone("missing") is False

    def test_after_remove_returns_false(self) -> None:
        skel = Skeleton()
        skel.add_bone("temp")
        skel.remove_bone("temp")
        assert skel.has_bone("temp") is False


class TestSkeletonGetBone:
    def test_exists_returns_bone(self) -> None:
        skel = Skeleton()
        original = skel.add_bone("mybone")
        retrieved = skel.get_bone("mybone")
        assert retrieved is original

    def test_missing_returns_none(self) -> None:
        skel = Skeleton()
        assert skel.get_bone("missing") is None


# =========================================================================
# Skeleton: get_child_names / get_parent_name
# =========================================================================

class TestSkeletonGetChildNames:
    def test_bone_with_children(self) -> None:
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("c1", parent_name="root")
        skel.add_bone("c2", parent_name="root")
        assert skel.get_child_names("root") == ["c1", "c2"]

    def test_leaf_bone_returns_empty(self) -> None:
        skel = Skeleton()
        skel.add_bone("root")
        assert skel.get_child_names("root") == []

    def test_missing_bone_returns_empty(self) -> None:
        skel = Skeleton()
        assert skel.get_child_names("missing") == []


class TestSkeletonGetParentName:
    def test_root_returns_none(self) -> None:
        skel = Skeleton()
        skel.add_bone("root")
        assert skel.get_parent_name("root") is None

    def test_child_returns_parent_name(self) -> None:
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")
        assert skel.get_parent_name("child") == "root"

    def test_missing_bone_returns_none(self) -> None:
        skel = Skeleton()
        assert skel.get_parent_name("missing") is None


# =========================================================================
# Skeleton: get_ancestor_names
# =========================================================================

class TestSkeletonGetAncestorNames:
    def test_root_returns_self(self) -> None:
        skel = Skeleton()
        skel.add_bone("root")
        assert skel.get_ancestor_names("root") == ["root"]

    def test_deep_chain_inclusive(self) -> None:
        """For bone 'c' in root->a->b->c: returns ['c', 'b', 'a', 'root']."""
        skel = _make_chain(["root", "a", "b", "c"])
        assert skel.get_ancestor_names("c") == ["c", "b", "a", "root"]

    def test_missing_bone_returns_empty(self) -> None:
        skel = Skeleton()
        assert skel.get_ancestor_names("missing") == []


# =========================================================================
# Skeleton: get_depth
# =========================================================================

class TestSkeletonGetDepth:
    def test_root_returns_zero(self) -> None:
        skel = Skeleton()
        skel.add_bone("root")
        assert skel.get_depth("root") == 0

    def test_deep_chain(self) -> None:
        skel = _make_chain(["root", "a", "b", "c"])
        assert skel.get_depth("c") == 3
        assert skel.get_depth("b") == 2
        assert skel.get_depth("a") == 1
        assert skel.get_depth("root") == 0

    def test_missing_returns_negative_one(self) -> None:
        skel = Skeleton()
        assert skel.get_depth("missing") == -1


# =========================================================================
# Skeleton: get_max_depth
# =========================================================================

class TestSkeletonGetMaxDepth:
    def test_empty_returns_negative_one(self) -> None:
        skel = Skeleton()
        assert skel.get_max_depth() == -1

    def test_single_root_returns_zero(self) -> None:
        skel = Skeleton()
        skel.add_bone("only")
        assert skel.get_max_depth() == 0

    def test_multi_tree_uses_deepest(self) -> None:
        """Tree A: root_a -> a1 (depth 1); Tree B: root_b -> b1 -> b2 (depth 2)."""
        skel = Skeleton()
        skel.add_bone("root_a")
        skel.add_bone("root_b")
        skel.add_bone("a1", parent_name="root_a")
        skel.add_bone("b1", parent_name="root_b")
        skel.add_bone("b2", parent_name="b1")
        assert skel.get_max_depth() == 2

    def test_all_roots_returns_zero(self) -> None:
        skel = Skeleton()
        skel.add_bone("r1")
        skel.add_bone("r2")
        skel.add_bone("r3")
        assert skel.get_max_depth() == 0


# =========================================================================
# Skeleton: get_chain (IK integration)
# =========================================================================

class TestSkeletonGetChain:
    """Exercise every branch in get_chain's LCA algorithm."""

    def test_same_bone_returns_single(self) -> None:
        skel = _make_chain(["root", "mid", "leaf"])
        chain = skel.get_chain("mid", "mid")
        assert len(chain) == 1
        assert chain[0].name == "mid"

    def test_direct_parent_to_child(self) -> None:
        skel = _make_chain(["root", "child"])
        chain = skel.get_chain("root", "child")
        assert [b.name for b in chain] == ["root", "child"]

    def test_distant_ancestor_to_descendant(self) -> None:
        skel = _make_chain(["root", "a", "b", "c", "d"])
        chain = skel.get_chain("root", "d")
        assert [b.name for b in chain] == ["root", "a", "b", "c", "d"]

    def test_end_is_ancestor_of_start_reverse(self) -> None:
        """start='c', end='root' in root->a->b->c should return [c, b, a, root]."""
        skel = _make_chain(["root", "a", "b", "c"])
        chain = skel.get_chain("c", "root")
        assert [b.name for b in chain] == ["c", "b", "a", "root"]

    def test_disconnected_trees_returns_empty(self) -> None:
        skel = Skeleton()
        skel.add_bone("tree1_root")
        skel.add_bone("tree2_root")
        skel.add_bone("t1_child", parent_name="tree1_root")
        skel.add_bone("t2_child", parent_name="tree2_root")
        assert skel.get_chain("t1_child", "t2_child") == []

    def test_start_missing_returns_empty(self) -> None:
        skel = _make_chain(["root", "leaf"])
        assert skel.get_chain("missing", "leaf") == []

    def test_end_missing_returns_empty(self) -> None:
        skel = _make_chain(["root", "leaf"])
        assert skel.get_chain("root", "missing") == []

    def test_both_missing_returns_empty(self) -> None:
        skel = Skeleton()
        assert skel.get_chain("x", "y") == []

    def test_cousins_with_common_ancestor(self) -> None:
        """root -> a -> a1, root -> b -> b1. Chain a1 -> b1: [a1, a, root, b, b1]."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("a", parent_name="root")
        skel.add_bone("b", parent_name="root")
        skel.add_bone("a1", parent_name="a")
        skel.add_bone("b1", parent_name="b")
        chain = skel.get_chain("a1", "b1")
        assert [b.name for b in chain] == ["a1", "a", "root", "b", "b1"]

    def test_ancestor_to_descendant_with_branch_siblings(self) -> None:
        """root -> a -> [a1, a2], where a2 is 'spine' and a1 is 'limb'.
           Chain from a -> a1: [a, a1].
        """
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("a", parent_name="root")
        skel.add_bone("a1", parent_name="a")
        skel.add_bone("a2", parent_name="a")
        chain = skel.get_chain("a", "a2")
        assert [b.name for b in chain] == ["a", "a2"]

    def test_same_bone_not_root(self) -> None:
        """leaf -> leaf returns [leaf]."""
        skel = _make_chain(["root", "mid", "leaf"])
        chain = skel.get_chain("leaf", "leaf")
        assert len(chain) == 1
        assert chain[0].name == "leaf"


# =========================================================================
# Skeleton: validate
# =========================================================================

class TestSkeletonValidate:
    def test_empty_returns_error(self) -> None:
        skel = Skeleton()
        errors = skel.validate()
        assert len(errors) == 1
        assert "empty" in errors[0]

    def test_valid_single_root_returns_no_errors(self) -> None:
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")
        assert skel.validate() == []

    def test_valid_forest_multiple_roots(self) -> None:
        skel = Skeleton()
        skel.add_bone("r1")
        skel.add_bone("r2")
        skel.add_bone("c1", parent_name="r1")
        assert skel.validate() == []

    def test_no_root_bones_returns_error(self) -> None:
        """Simulate a skeleton where every bone has a parent and no root lists."""
        skel = Skeleton()
        # Add bones manually into _bones but clear _root_bones so there are no roots
        root = Bone("root")
        child = Bone("child", parent=root)
        root.children.append(child)
        skel._bones["root"] = root
        skel._bones["child"] = child
        skel._root_bones = []  # no roots
        errors = skel.validate()
        assert any("no root bones" in e for e in errors)

    def test_orphan_bone_not_reachable(self) -> None:
        """A bone registered but not reachable from any root."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")
        # Manually add an orphan bone that's not linked to the root tree
        orphan = Bone("orphan")
        skel._bones["orphan"] = orphan
        errors = skel.validate()
        assert any("orphan" in e.lower() and "orphan" in e for e in errors)

    def test_parent_backref_to_unregistered_bone(self) -> None:
        """A bone whose parent is not registered in the skeleton."""
        skel = Skeleton()
        skel.add_bone("root")
        # Manually create a bone that references an external parent
        external = Bone("external")
        bad = Bone("bad", parent=external)
        skel._bones["bad"] = bad
        skel._root_bones.append(bad)  # so it's reachable
        errors = skel.validate()
        assert any("parent" in e and "external" in e for e in errors)

    def test_child_backref_to_unregistered_bone(self) -> None:
        """A bone whose children list contains an unregistered bone."""
        skel = Skeleton()
        skel.add_bone("root")
        unregistered = Bone("unregistered")
        # Manually add unregistered bone to root's children
        root = skel.get_bone("root")
        assert root is not None
        root.children.append(unregistered)
        unregistered.parent = root
        errors = skel.validate()
        assert any("child" in e and "unregistered" in e for e in errors)

    def test_cycle_in_parent_references(self) -> None:
        """A->B->C->A creates a cycle."""
        skel = Skeleton()
        a = Bone("a")
        b = Bone("b", parent=a)
        c = Bone("c", parent=b)
        a.parent = c  # close the cycle: a->None becomes a->c
        a.children.append(b)
        b.children.append(c)
        c.children.append(a)
        skel._bones["a"] = a
        skel._bones["b"] = b
        skel._bones["c"] = c
        # Need at least one root, or the "no root" error fires first.
        # a has parent=c, b has parent=a, c has parent=b -- none are root.
        # So validate will also report "no root bones".
        # Let's add a real root to isolate the cycle check.
        skel._root_bones = []  # no roots, cycle exists, no reachable roots
        errors = skel.validate()
        assert any("Cycle" in e for e in errors)

    def test_cycle_with_real_root_detected(self) -> None:
        """root -> a -> b -> c -> a (cycle). root is valid reachable root.
           Cycle detection should fire when walking from 'c'.
        """
        skel = Skeleton()
        root = Bone("root")
        a = Bone("a", parent=root)
        b = Bone("b", parent=a)
        c = Bone("c", parent=b)
        a.parent = root
        # Create cycle: c -> a -> b -> c
        # But a already has parent=root. So we need c.parent=b, but we add a back-edge from somewhere.
        # Actually let's do: root -> a -> b -> c -> a (cycle)
        a.children.append(b)
        b.children.append(c)
        c.children = []  # no children
        # But c.parent is already b. To create cycle, we need a.parent to be c.
        # But a.parent is root. Hmm.

        # Let's construct a proper cycle: root is valid, and a->b->c->a is a sub-cycle
        # that's disconnected from root reachability. That won't work because
        # validate uses BFS from roots to check reachability.
        # Actually let's just make a simple cycle: root -> a -> b -> a
        # where a is root, b is child of a, and a.parent = b.
        # Wait, a is root so a.parent = None. Let's make it:
        # root is a valid root.
        # a has parent=root, b has parent=a, a.parent = b (cycle).
        # But we already set a.parent = root above.

        # Let's do a clean construction:
        skel2 = Skeleton("cycle_test")
        r = Bone("r")
        a2 = Bone("a2", parent=r)
        b2 = Bone("b2", parent=a2)
        r.children = [a2]
        a2.children = [b2]
        b2.children = []
        # Create cycle: a2.parent -> b2
        a2.parent = b2
        b2.parent = a2
        # Now r is reachable, r->a2->b2, but from a2, walking up goes a2->b2->a2 (cycle)
        skel2._bones["r"] = r
        skel2._bones["a2"] = a2
        skel2._bones["b2"] = b2
        skel2._root_bones = [r]
        errors = skel2.validate()
        assert any("Cycle" in e for e in errors)

    def test_validate_passes_with_mixed_roots_and_children(self) -> None:
        """Two disjoint valid trees should pass."""
        skel = Skeleton()
        skel.add_bone("r1")
        skel.add_bone("r2")
        skel.add_bone("c1", parent_name="r1")
        skel.add_bone("c2", parent_name="c1")
        skel.add_bone("c3", parent_name="r2")
        assert skel.validate() == []


# =========================================================================
# Skeleton: add_bones_from_flat
# =========================================================================

class TestSkeletonAddBonesFromFlat:
    def test_batch_add_simple(self) -> None:
        skel = Skeleton()
        skel.add_bones_from_flat([
            ("root", None),
            ("a", "root"),
            ("b", "a"),
            ("c", "a"),
        ])
        assert skel.bone_count == 4
        assert skel.get_depth("b") == 2
        assert skel.get_depth("c") == 2
        assert skel.get_depth("root") == 0

    def test_batch_add_with_bind_poses(self) -> None:
        skel = Skeleton()
        poses = {
            "root": _make_transform(tx=0, ty=0, tz=0),
            "a": _make_transform(tx=1, ty=0, tz=0),
        }
        skel.add_bones_from_flat([
            ("root", None),
            ("a", "root"),
        ], bind_poses=poses)
        assert skel.get_bone("root").bind_pose is poses["root"]
        assert skel.get_bone("a").bind_pose is poses["a"]

    def test_batch_add_partial_bind_poses(self) -> None:
        """Bones not in bind_poses dict get identity."""
        skel = Skeleton()
        poses = {"root": _make_transform(tx=5, ty=0, tz=0)}
        skel.add_bones_from_flat([
            ("root", None),
            ("a", "root"),
        ], bind_poses=poses)
        assert skel.get_bone("root").bind_pose is poses["root"]
        assert _transform_equal(skel.get_bone("a").bind_pose, Transform.identity())


# =========================================================================
# Skeleton: copy
# =========================================================================

class TestSkeletonCopy:
    def test_deep_copy_independence(self) -> None:
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")

        copied = skel.copy()
        # deepcopy preserves original name (name param is accepted but unused)
        assert copied.name == "skeleton"

        # Mutate original -- copy should be independent
        skel.add_bone("extra", parent_name="root")
        assert copied.bone_count == 2
        assert copied.get_bone("extra") is None

        # Mutate a bone name in the copy (doesn't affect original)
        copied.get_bone("root").name = "renamed"
        assert skel.get_bone("root").name == "root"

    def test_copy_with_custom_name(self) -> None:
        skel = Skeleton("original")
        copied = skel.copy(name="clone")
        assert copied.name == "clone"

    def test_copy_empty_skeleton(self) -> None:
        skel = Skeleton("empty")
        copied = skel.copy()
        assert copied.bone_count == 0
        assert copied.name == "empty"


# =========================================================================
# Skeleton: dunder methods
# =========================================================================

class TestSkeletonDunders:
    def test_contains_bone_exists(self) -> None:
        skel = Skeleton()
        skel.add_bone("present")
        assert "present" in skel

    def test_contains_bone_missing(self) -> None:
        skel = Skeleton()
        assert "missing" not in skel

    def test_len_returns_bone_count(self) -> None:
        skel = Skeleton()
        assert len(skel) == 0
        skel.add_bone("a")
        skel.add_bone("b")
        assert len(skel) == 2

    def test_iter_yields_bone_objects(self) -> None:
        skel = Skeleton()
        skel.add_bone("a")
        skel.add_bone("b")
        bones = list(skel)
        assert len(bones) == 2
        assert all(isinstance(b, Bone) for b in bones)
        assert {b.name for b in bones} == {"a", "b"}

    def test_iter_empty(self) -> None:
        skel = Skeleton()
        assert list(skel) == []

    def test_repr(self) -> None:
        skel = Skeleton("my_skel")
        skel.add_bone("root")
        assert repr(skel) == "Skeleton('my_skel', bones=1)"
        skel.add_bone("child", parent_name="root")
        assert repr(skel) == "Skeleton('my_skel', bones=2)"


# =========================================================================
# Edge cases combining multiple operations
# =========================================================================

class TestSkeletonEdgeCases:
    def test_empty_skeleton_validate_and_query(self) -> None:
        """All query methods on empty skeleton should not crash."""
        skel = Skeleton()
        assert skel.bone_count == 0
        assert skel.get_bone("anything") is None
        assert skel.has_bone("anything") is False
        assert skel.get_child_names("anything") == []
        assert skel.get_parent_name("anything") is None
        assert skel.get_ancestor_names("anything") == []
        assert skel.get_depth("anything") == -1
        assert skel.get_max_depth() == -1
        assert skel.get_chain("a", "b") == []
        assert len(skel) == 0
        assert list(skel) == []

    def test_single_bone_skeleton(self) -> None:
        """Minimal non-empty skeleton operations."""
        skel = Skeleton()
        skel.add_bone("only")
        assert skel.bone_count == 1
        assert skel.get_depth("only") == 0
        assert skel.get_max_depth() == 0
        assert skel.get_ancestor_names("only") == ["only"]
        assert skel.get_parent_name("only") is None
        assert skel.get_child_names("only") == []
        assert skel.get_chain("only", "only")[0].name == "only"
        assert skel.validate() == []

    def test_remove_and_re_add(self) -> None:
        """Remove a bone, then add a new one with same name."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_name="root")
        skel.remove_bone("child")
        # Re-add with same name should work
        skel.add_bone("child", parent_name="root")
        assert skel.bone_count == 2
        assert skel.get_depth("child") == 1

    def test_deep_hierarchy_max_depth(self) -> None:
        """100-deep chain should compute max_depth correctly."""
        skel = Skeleton()
        skel.add_bone("n0")
        for i in range(1, 100):
            skel.add_bone(f"n{i}", parent_name=f"n{i-1}")
        assert skel.get_max_depth() == 99
        assert skel.get_depth("n99") == 99
        assert skel.get_depth("n0") == 0

    def test_many_roots(self) -> None:
        """Skeleton with 100 root bones (flat forest)."""
        skel = Skeleton()
        for i in range(100):
            skel.add_bone(f"root_{i}")
        assert skel.bone_count == 100
        assert len(skel.root_bones) == 100
        assert skel.get_max_depth() == 0
        assert skel.validate() == []

    def test_add_bone_returns_new_bone(self) -> None:
        """Confirm add_bone returns the newly created instance."""
        skel = Skeleton()
        bone = skel.add_bone("new")
        assert isinstance(bone, Bone)
        assert bone.name == "new"
        assert _transform_equal(bone.bind_pose, Transform.identity())

    def test_get_bone_returns_same_instance(self) -> None:
        """get_bone should return the exact same Bone instance."""
        skel = Skeleton()
        original = skel.add_bone("ref")
        retrieved = skel.get_bone("ref")
        assert retrieved is original

    def test_remove_bone_then_ancestor_chain(self) -> None:
        """After removing a mid bone, ancestor chain for surviving bones is correct.
           root -> mid -> leaf. Remove mid. leaf re-parents to root.
           leaf's ancestors should be [leaf, root].
        """
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("mid", parent_name="root")
        skel.add_bone("leaf", parent_name="mid")
        skel.remove_bone("mid")
        # leaf gets re-parented to root (mid's parent)
        assert skel.get_ancestor_names("leaf") == ["leaf", "root"]
        assert skel.get_parent_name("leaf") == "root"

    def test_remove_all_bones_sequentially(self) -> None:
        """Remove all bones one by one from a chain."""
        skel = _make_chain(["r", "a", "b"])
        skel.remove_bone("b")
        assert skel.bone_count == 2
        skel.remove_bone("a")
        assert skel.bone_count == 1
        skel.remove_bone("r")
        assert skel.bone_count == 0
        assert skel.validate() != []  # empty should error

    def test_validate_orphan_with_separate_valid_tree(self) -> None:
        """One valid tree + one orphan. validate must catch the orphan."""
        skel = Skeleton()
        skel.add_bone("valid_root")
        skel.add_bone("valid_child", parent_name="valid_root")
        # Add orphan
        orphan = Bone("orphan")
        skel._bones["orphan"] = orphan
        errors = skel.validate()
        assert any("orphan" in e for e in errors)
        # Also should not report "empty" since there are bones
        assert not any("empty" in e for e in errors)


# =========================================================================
# Bind pose edge cases
# =========================================================================

class TestBindPose:
    def test_default_bind_pose_is_identity(self) -> None:
        skel = Skeleton()
        bone = skel.add_bone("root")
        assert _transform_equal(bone.bind_pose, Transform.identity())

    def test_bind_pose_storage_and_retrieval(self) -> None:
        pose = _make_transform(tx=10, ty=20, tz=30, sx=2, sy=2, sz=2)
        skel = Skeleton()
        bone = skel.add_bone("custom", bind_pose=pose)
        assert bone.bind_pose is pose

    def test_bind_pose_independence_via_copy(self) -> None:
        skel = Skeleton()
        pose = _make_transform(tx=1, ty=2, tz=3)
        skel.add_bone("root", bind_pose=pose)
        copied = skel.copy()
        orig_pose = skel.get_bone("root").bind_pose
        copy_pose = copied.get_bone("root").bind_pose
        assert orig_pose is not copy_pose  # deep copy
        assert _transform_equal(orig_pose, copy_pose)

    def test_bind_pose_via_flat_add(self) -> None:
        poses = {
            "root": _make_transform(tx=5, ty=0, tz=0),
            "child": _make_transform(tx=0, ty=5, tz=0),
        }
        skel = Skeleton()
        skel.add_bones_from_flat([
            ("root", None),
            ("child", "root"),
        ], bind_poses=poses)
        assert skel.get_bone("root").bind_pose is poses["root"]
        assert skel.get_bone("child").bind_pose is poses["child"]


# =========================================================================
# Additional edge cases for remove_bone re-parenting details
# =========================================================================

class TestRemoveBoneEdgeCases:
    def test_remove_root_with_single_child_reparents(self) -> None:
        """Remove root when it has exactly one child. Child becomes root."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("only_child", parent_name="root")
        skel.remove_bone("root")
        assert skel.bone_count == 1
        assert "root" not in skel
        assert len(skel._root_bones) == 1
        assert skel._root_bones[0].name == "only_child"
        assert skel._root_bones[0].parent is None

    def test_remove_intermediate_with_deep_children(self) -> None:
        """Remove 'mid' in root->mid->a->b: a re-parents to root, b stays child of a."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("mid", parent_name="root")
        skel.add_bone("a", parent_name="mid")
        skel.add_bone("b", parent_name="a")
        skel.remove_bone("mid")
        assert skel.bone_count == 3
        # 'a' re-parents to root (mid's parent)
        assert skel.get_parent_name("a") == "root"
        # 'b' should still be child of 'a'
        assert skel.get_parent_name("b") == "a"


# =========================================================================
# Edge cases for get_chain with more complex topologies
# =========================================================================

class TestGetChainEdgeCases:
    def test_chain_start_depth_greater_than_end_depth(self) -> None:
        """start is deeper than end (standard ancestor query)."""
        skel = _make_chain(["gp", "p", "c"])
        chain = skel.get_chain("gp", "c")
        assert [b.name for b in chain] == ["gp", "p", "c"]

    def test_chain_end_is_ancestor(self) -> None:
        """end is ancestor of start."""
        skel = _make_chain(["gp", "p", "c"])
        chain = skel.get_chain("c", "gp")
        assert [b.name for b in chain] == ["c", "p", "gp"]

    def test_chain_different_trees_empty(self) -> None:
        """Two completely separate trees."""
        skel = Skeleton()
        skel.add_bone("r1")
        skel.add_bone("r2")
        assert skel.get_chain("r1", "r2") == []

    def test_chain_self_is_root(self) -> None:
        skel = Skeleton()
        skel.add_bone("alone")
        assert skel.get_chain("alone", "alone")[0].name == "alone"

"""Contract tests for Skeleton and Bone Hierarchy (T-AG-1.3).

CLEANROOM: tests the PUBLIC API contract only.
No knowledge of internal representation, data structures, or algorithms.

Contract sources:
  - PHASE_1_ARCH.md section 2.3 (Skeleton + Bone structure)
  - PHASE_1_TODO.md T-AG-1.3 (acceptance criteria)
  - engine/animation/graph/__init__.py (public exports)
"""
import pytest
from engine.animation.graph import Skeleton, Bone, Transform, Pose


# ============================================================================
# Equivalence Class: Bone creation
# ============================================================================

class TestBoneCreation:
    """Bone instances can be created with various constructor forms."""

    def test_bone_minimal(self):
        """Bone accepts name and index as positional arguments."""
        b = Bone("hip", 0)
        assert b.name == "hip"
        assert b.index == 0

    def test_bone_with_parent_index(self):
        """Bone can be created with a parent_index reference."""
        b = Bone("thigh", 1, parent_index=0)
        assert b.name == "thigh"
        assert b.index == 1
        assert b.parent_index == 0

    def test_bone_default_parent(self):
        """Default parent_index is -1 (no parent)."""
        b = Bone("root", 0)
        assert b.parent_index == -1

    def test_bone_with_bind_pose(self):
        """Bone can carry a custom bind-pose Transform."""
        t = Transform()
        b = Bone("root", 0, bind_pose=t)
        assert b.bind_pose is t

    def test_bone_default_bind_pose(self):
        """Bone without explicit bind_pose still has a Transform."""
        b = Bone("root", 0)
        assert isinstance(b.bind_pose, Transform)

    def test_bone_keyword_init(self):
        """Bone accepts keyword arguments."""
        b = Bone(name="spine", index=2)
        assert b.name == "spine"
        assert b.index == 2


# ============================================================================
# Equivalence Class: Skeleton creation
# ============================================================================

class TestSkeletonCreation:
    """Skeleton can be created and reports bone count correctly."""

    def test_create_empty(self):
        """An empty Skeleton has bone_count == 0."""
        skel = Skeleton()
        assert skel.bone_count() == 0

    def test_create_with_bones_list(self):
        """Skeleton can be initialised from a list of Bone objects."""
        b0 = Bone("root", 0)
        b1 = Bone("child", 1, parent_index=0)
        skel = Skeleton(bones=[b0, b1])
        assert skel.bone_count() == 2


# ============================================================================
# Equivalence Class: add_bone API
# ============================================================================

class TestAddBone:
    """Bones are added to a Skeleton by name string."""

    def test_add_single_root(self):
        """Adding a root bone by name increments bone_count."""
        skel = Skeleton()
        skel.add_bone("root")
        assert skel.bone_count() == 1

    def test_add_with_parent_index(self):
        """Adding a bone with a parent_index creates parent relationship."""
        skel = Skeleton()
        skel.add_bone("root")            # index 0
        skel.add_bone("child", parent_index=0)  # index 1
        assert skel.bone_count() == 2

    def test_add_multiple(self):
        """Adding several bones updates bone_count correctly."""
        skel = Skeleton()
        for i in range(5):
            skel.add_bone(f"bone_{i}")
        assert skel.bone_count() == 5

    def test_add_returns_bone(self):
        """add_bone returns the newly created Bone."""
        skel = Skeleton()
        bone = skel.add_bone("root")
        assert isinstance(bone, Bone)
        assert bone.name == "root"

    def test_add_with_bind_pose(self):
        """add_bone accepts an optional bind_pose argument."""
        skel = Skeleton()
        t = Transform()
        bone = skel.add_bone("root", bind_pose=t)
        assert bone.bind_pose is t


# ============================================================================
# Equivalence Class: get_bone lookup
# ============================================================================

class TestGetBone:
    """Bones can be retrieved by index and by name."""

    def test_get_bone_by_index(self):
        """get_bone(index) returns the bone at that index."""
        skel = Skeleton()
        skel.add_bone("root")
        bone = skel.get_bone(0)
        assert isinstance(bone, Bone)
        assert bone.name == "root"

    def test_get_bone_second_index(self):
        """get_bone returns the correct bone at each index."""
        skel = Skeleton()
        skel.add_bone("first")
        skel.add_bone("second")
        assert skel.get_bone(1).name == "second"

    def test_get_bone_by_name(self):
        """get_bone_by_name(name) returns the named bone."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_index=0)
        bone = skel.get_bone_by_name("child")
        assert isinstance(bone, Bone)
        assert bone.name == "child"

    def test_get_bone_index_known(self):
        """get_bone_index(name) returns the numeric index for a known bone."""
        skel = Skeleton()
        skel.add_bone("alpha")
        skel.add_bone("beta", parent_index=0)
        assert skel.get_bone_index("alpha") == 0
        assert skel.get_bone_index("beta") == 1


# ============================================================================
# Equivalence Class: get_bone error handling
# ============================================================================

class TestGetBoneErrors:
    """Lookup of non-existent bones returns sentinel values."""

    def test_get_bone_out_of_range(self):
        """get_bone with out-of-range index returns None."""
        skel = Skeleton()
        skel.add_bone("root")
        assert skel.get_bone(99) is None

    def test_get_bone_negative_index(self):
        """get_bone with negative index returns None."""
        skel = Skeleton()
        skel.add_bone("root")
        assert skel.get_bone(-1) is None

    def test_get_bone_by_name_nonexistent(self):
        """get_bone_by_name with unknown name returns None."""
        skel = Skeleton()
        skel.add_bone("root")
        assert skel.get_bone_by_name("phantom") is None

    def test_get_bone_index_nonexistent(self):
        """get_bone_index with unknown name returns -1."""
        skel = Skeleton()
        skel.add_bone("root")
        assert skel.get_bone_index("phantom") == -1


# ============================================================================
# Equivalence Class: Bind pose on Bone
# ============================================================================

class TestBindPose:
    """Each Bone carries a bind-pose Transform."""

    def test_default_bind_pose_position(self):
        """Default bind pose position is (0, 0, 0)."""
        b = Bone("root", 0)
        t = b.bind_pose
        assert t.position[0] == 0.0
        assert t.position[1] == 0.0
        assert t.position[2] == 0.0

    def test_default_bind_pose_scale(self):
        """Default bind pose scale is (1, 1, 1)."""
        b = Bone("root", 0)
        t = b.bind_pose
        assert t.scale[0] == 1.0
        assert t.scale[1] == 1.0
        assert t.scale[2] == 1.0

    def test_default_bind_pose_rotation(self):
        """Default bind pose rotation is identity quaternion (0,0,0,1)."""
        b = Bone("root", 0)
        t = b.bind_pose
        assert t.rotation[0] == 0.0
        assert t.rotation[1] == 0.0
        assert t.rotation[2] == 0.0
        assert t.rotation[3] == 1.0

    def test_custom_bind_pose(self):
        """Bone with explicit bind pose stores the custom transform."""
        t = Transform()
        # Position is a tuple (x, y, z) -- set via direct creation
        t2 = Transform()
        t2.position = (10.0, 20.0, 30.0)
        b = Bone("root", 0, bind_pose=t2)
        assert b.bind_pose.position[0] == 10.0
        assert b.bind_pose.position[1] == 20.0
        assert b.bind_pose.position[2] == 30.0


# ============================================================================
# Equivalence Class: get_bind_pose on Skeleton
# ============================================================================

class TestSkeletonBindPose:
    """Skeleton.get_bind_pose() returns the full bind pose."""

    def test_get_bind_pose_returns_pose(self):
        """Skeleton.get_bind_pose() returns a Pose object."""
        skel = Skeleton()
        skel.add_bone("root")
        pose = skel.get_bind_pose()
        assert isinstance(pose, Pose)

    def test_get_bind_pose_contains_bones(self):
        """get_bind_pose() has transforms matching added bones."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_index=0)
        pose = skel.get_bind_pose()
        assert len(pose.transforms) == 2


# ============================================================================
# Equivalence Class: get_chain -- bone chain traversal (by index)
# ============================================================================

class TestGetChain:
    """Bone chain traversal from start index to end index."""

    def test_chain_direct_parent_child(self):
        """get_chain returns [parent, child] for direct relationship."""
        skel = Skeleton()
        skel.add_bone("root")           # index 0
        skel.add_bone("child", parent_index=0)  # index 1
        chain = skel.get_chain(0, 1)
        assert chain == [0, 1], f"Expected [0, 1], got {chain}"

    def test_chain_deep(self):
        """get_chain returns ordered indices for deep chain."""
        skel = Skeleton()
        skel.add_bone("root")     # index 0
        skel.add_bone("mid", parent_index=0)   # index 1
        skel.add_bone("tip", parent_index=1)   # index 2
        chain = skel.get_chain(0, 2)
        assert chain == [0, 1, 2], f"Expected [0, 1, 2], got {chain}"

    def test_chain_partial(self):
        """get_chain works for non-root start index."""
        skel = Skeleton()
        skel.add_bone("root")     # index 0
        skel.add_bone("mid", parent_index=0)   # index 1
        skel.add_bone("tip", parent_index=1)   # index 2
        chain = skel.get_chain(1, 2)
        assert chain == [1, 2], f"Expected [1, 2], got {chain}"

    def test_chain_same_bone(self):
        """get_chain with start == end returns a single-element list."""
        skel = Skeleton()
        skel.add_bone("root")
        chain = skel.get_chain(0, 0)
        assert chain == [0], f"Expected [0], got {chain}"

    # NOTE: Reverse chain (end is ancestor of start) is not supported.
    # get_chain(2, 0) returns [] not [2, 1, 0] -- see report.

# ============================================================================
# Boundary: Disconnected bones
# ============================================================================

class TestDisconnectedBones:
    """Bones in separate hierarchies with no path between them."""

    def test_chain_disconnected_roots(self):
        """get_chain between two unrelated root bones returns empty list."""
        skel = Skeleton()
        skel.add_bone("root_a")       # index 0
        skel.add_bone("root_b")       # index 1 -- separate root
        chain = skel.get_chain(0, 1)
        assert chain == [], f"Expected empty chain, got {chain}"

    def test_orphan_bone_exists(self):
        """An orphan is a valid counted bone in the skeleton."""
        skel = Skeleton()
        skel.add_bone("orphan")
        assert skel.bone_count() == 1


# ============================================================================
# Equivalence Class: get_children
# ============================================================================

class TestGetChildren:
    """Children of a bone can be retrieved by index."""

    def test_root_has_children(self):
        """Root bone reveals its children by index."""
        skel = Skeleton()
        skel.add_bone("root")     # index 0
        skel.add_bone("child", parent_index=0)  # index 1
        children = skel.get_children(0)
        assert 1 in children

    def test_leaf_has_no_children(self):
        """Leaf bone has empty children list."""
        skel = Skeleton()
        skel.add_bone("root")     # index 0
        skel.add_bone("child", parent_index=0)  # index 1
        children = skel.get_children(1)
        assert children == []

    def test_multiple_children(self):
        """Bone with multiple children returns all of them."""
        skel = Skeleton()
        skel.add_bone("root")     # index 0
        skel.add_bone("a", parent_index=0)  # index 1
        skel.add_bone("b", parent_index=0)  # index 2
        children = skel.get_children(0)
        assert 1 in children
        assert 2 in children
        assert len(children) == 2


# ============================================================================
# Equivalence Class: Bone.is_root detection
# ============================================================================

class TestIsRoot:
    """Bone.is_root() reflects parent status correctly."""

    def test_root_bone_is_root(self):
        """Bone with no parent (parent_index == -1) is a root."""
        b = Bone("root", 0)
        assert b.is_root() is True

    def test_child_bone_is_not_root(self):
        """Bone with a parent (parent_index != -1) is not a root."""
        b = Bone("child", 1, parent_index=0)
        assert b.is_root() is False

    def test_root_in_skeleton(self):
        """Root bone in a skeleton is_root() returns True."""
        skel = Skeleton()
        skel.add_bone("root")
        assert skel.get_bone(0).is_root() is True

    def test_child_in_skeleton(self):
        """Child bone in a skeleton is_root() returns False."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("child", parent_index=0)
        assert skel.get_bone(1).is_root() is False


# ============================================================================
# Edge cases: Skeleton validation
# ============================================================================

class TestValidation:
    """Skeleton edge cases and structural properties."""

    def test_empty_skeleton(self):
        """An empty skeleton has zero bones and no bone at index 0."""
        skel = Skeleton()
        assert skel.bone_count() == 0
        assert skel.get_bone(0) is None

    def test_multiple_roots(self):
        """Skeleton supports multiple root bones (per ARCH section 2.3)."""
        skel = Skeleton()
        skel.add_bone("root_a")
        skel.add_bone("root_b")
        assert skel.bone_count() == 2
        assert skel.get_bone(0).is_root() is True
        assert skel.get_bone(1).is_root() is True

    def test_duplicate_name_by_name_lookup(self):
        """Lookup by name returns a Bone with the requested name."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("root")  # duplicate name
        retrieved = skel.get_bone_by_name("root")
        assert retrieved is not None
        assert retrieved.name == "root"

    def test_get_bone_index_returns_int(self):
        """get_bone_index returns an int for a known bone."""
        skel = Skeleton()
        skel.add_bone("alpha")
        assert isinstance(skel.get_bone_index("alpha"), int)


# ============================================================================
# Edge cases: bone_count consistency
# ============================================================================

class TestBoneCount:
    """bone_count stays consistent through operations."""

    def test_count_zero_initially(self):
        """Fresh skeleton starts with zero bones."""
        assert Skeleton().bone_count() == 0

    def test_count_after_bulk_add(self):
        """bone_count reflects total bones after many adds."""
        skel = Skeleton()
        n = 100
        for i in range(n):
            skel.add_bone(f"bone_{i}")
        assert skel.bone_count() == n

    def test_count_incremental(self):
        """bone_count increases by one per add_bone."""
        skel = Skeleton()
        for i in range(10):
            skel.add_bone(f"b{i}")
            assert skel.bone_count() == i + 1

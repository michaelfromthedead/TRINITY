"""
Blackbox tests for Skeleton and Bone structures.

Tests cover observable behavior without knowledge of implementation details.
Focuses on hierarchical bone structures, parent-child relationships,
and skeleton construction.
"""

import pytest


class TestBoneCreation:
    """Tests for Bone object creation and properties."""

    def test_bone_with_name_and_no_parent(self):
        """Root bone should have parent_index of -1."""
        from engine.animation.skeletal.skeleton import Bone

        bone = Bone(index=0, name="root", parent_index=-1)
        assert bone.name == "root"
        assert bone.parent_index == -1
        assert bone.index == 0

    def test_bone_with_parent_index(self):
        """Child bone should store parent index correctly."""
        from engine.animation.skeletal.skeleton import Bone

        bone = Bone(index=1, name="spine", parent_index=0)
        assert bone.name == "spine"
        assert bone.parent_index == 0
        assert bone.index == 1

    def test_bone_name_is_string(self):
        """Bone name should be accessible as string."""
        from engine.animation.skeletal.skeleton import Bone

        bone = Bone(index=2, name="left_arm", parent_index=1)
        assert isinstance(bone.name, str)
        assert "left_arm" in bone.name

    def test_bone_with_deep_parent_index(self):
        """Bone can have parent deep in hierarchy."""
        from engine.animation.skeletal.skeleton import Bone

        bone = Bone(index=16, name="finger_tip", parent_index=15)
        assert bone.parent_index == 15

    def test_bone_name_with_special_characters(self):
        """Bone names may contain underscores and numbers."""
        from engine.animation.skeletal.skeleton import Bone

        bone = Bone(index=0, name="bone_123_left", parent_index=-1)
        assert bone.name == "bone_123_left"

    def test_multiple_bones_are_independent(self):
        """Creating multiple bones should not affect each other."""
        from engine.animation.skeletal.skeleton import Bone

        bone1 = Bone(index=0, name="bone_a", parent_index=-1)
        bone2 = Bone(index=1, name="bone_b", parent_index=0)
        bone3 = Bone(index=2, name="bone_c", parent_index=1)

        assert bone1.name == "bone_a"
        assert bone2.name == "bone_b"
        assert bone3.name == "bone_c"
        assert bone1.parent_index == -1
        assert bone2.parent_index == 0
        assert bone3.parent_index == 1

    def test_bone_is_root_method(self):
        """Bone should have is_root method."""
        from engine.animation.skeletal.skeleton import Bone

        root = Bone(index=0, name="root", parent_index=-1)
        child = Bone(index=1, name="child", parent_index=0)

        assert root.is_root()
        assert not child.is_root()


class TestBoneValidation:
    """Tests for Bone validation."""

    def test_bone_negative_index_raises(self):
        """Bone index must be >= 0."""
        from engine.animation.skeletal.skeleton import Bone

        with pytest.raises(ValueError):
            Bone(index=-1, name="invalid", parent_index=-1)

    def test_bone_empty_name_raises(self):
        """Bone name cannot be empty."""
        from engine.animation.skeletal.skeleton import Bone

        with pytest.raises(ValueError):
            Bone(index=0, name="", parent_index=-1)

    def test_bone_self_parent_raises(self):
        """Bone cannot be its own parent."""
        from engine.animation.skeletal.skeleton import Bone

        with pytest.raises(ValueError):
            Bone(index=5, name="self_parent", parent_index=5)


class TestSkeletonCreation:
    """Tests for Skeleton construction."""

    def test_empty_skeleton(self):
        """Empty skeleton should have no bones."""
        from engine.animation.skeletal.skeleton import Skeleton

        skeleton = Skeleton(name="test")
        assert skeleton.bone_count == 0

    def test_skeleton_name(self):
        """Skeleton should store its name."""
        from engine.animation.skeletal.skeleton import Skeleton

        skeleton = Skeleton(name="humanoid")
        assert skeleton.name == "humanoid"

    def test_add_single_root_bone(self):
        """Adding a root bone should increase bone count."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        assert skeleton.bone_count == 1

    def test_add_multiple_bones(self):
        """Adding multiple bones should be tracked correctly."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="head", parent_index=1))
        assert skeleton.bone_count == 3

    def test_get_bone_by_index(self):
        """Should retrieve bone by its index."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))

        bone = skeleton.get_bone(0)
        assert bone.name == "root"

        bone = skeleton.get_bone(1)
        assert bone.name == "spine"

    def test_get_bone_by_name(self):
        """Should retrieve bone by its name."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))

        bone = skeleton.get_bone_by_name("spine")
        assert bone is not None
        assert bone.name == "spine"


class TestSkeletonHierarchy:
    """Tests for skeleton hierarchical relationships."""

    def test_root_bone_has_no_parent(self):
        """Root bone should report no parent."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        bone = skeleton.get_bone(0)
        assert bone.parent_index == -1

    def test_child_bone_references_parent(self):
        """Child bone should reference its parent correctly."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))

        child = skeleton.get_bone(1)
        assert child.parent_index == 0

    def test_get_children_of_bone(self):
        """Should be able to get all children of a bone."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="left_arm", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="right_arm", parent_index=0))
        skeleton.add_bone(Bone(index=3, name="spine", parent_index=0))

        children = skeleton.get_bone_children(0)
        assert len(children) == 3

    def test_leaf_bone_has_no_children(self):
        """Leaf bones should have no children."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="leaf", parent_index=0))

        children = skeleton.get_bone_children(1)
        assert len(children) == 0

    def test_multi_level_hierarchy(self):
        """Should support multi-level bone hierarchies."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="chest", parent_index=1))
        skeleton.add_bone(Bone(index=3, name="neck", parent_index=2))
        skeleton.add_bone(Bone(index=4, name="head", parent_index=3))

        head = skeleton.get_bone(4)
        assert head.parent_index == 3

        neck = skeleton.get_bone(3)
        assert neck.parent_index == 2

    def test_branching_hierarchy(self):
        """Should support branching hierarchies (multiple children)."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="left_arm", parent_index=1))
        skeleton.add_bone(Bone(index=3, name="right_arm", parent_index=1))
        skeleton.add_bone(Bone(index=4, name="head", parent_index=1))

        spine_children = skeleton.get_bone_children(1)
        assert len(spine_children) == 3

    def test_get_leaf_bones(self):
        """Should identify leaf bones correctly."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="head", parent_index=1))

        leaf_indices = skeleton.get_leaf_bones()
        assert 2 in leaf_indices  # head is leaf
        assert 0 not in leaf_indices  # root not leaf
        assert 1 not in leaf_indices  # spine not leaf

    def test_root_bone_indices(self):
        """Should identify root bones correctly."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))

        assert 0 in skeleton.root_bone_indices
        assert 1 not in skeleton.root_bone_indices


class TestSkeletonTraversal:
    """Tests for skeleton traversal operations."""

    def test_iterate_all_bones(self):
        """Should be able to iterate over all bones."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="head", parent_index=1))

        bone_names = [b.name for b in skeleton]
        assert "root" in bone_names
        assert "spine" in bone_names
        assert "head" in bone_names
        assert len(bone_names) == 3

    def test_get_depth_of_bone(self):
        """Should calculate bone depth in hierarchy."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="chest", parent_index=1))

        assert skeleton.get_depth(0) == 0
        assert skeleton.get_depth(1) == 1
        assert skeleton.get_depth(2) == 2

    def test_get_max_depth(self):
        """Should calculate maximum hierarchy depth."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="chest", parent_index=1))

        assert skeleton.get_max_depth() == 2

    def test_get_bone_descendants(self):
        """Should get all descendants of a bone."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="chest", parent_index=1))
        skeleton.add_bone(Bone(index=3, name="head", parent_index=2))

        descendants = skeleton.get_bone_descendants(0)
        assert 1 in descendants
        assert 2 in descendants
        assert 3 in descendants


class TestSkeletonValidation:
    """Tests for skeleton validation and error handling."""

    def test_get_nonexistent_bone_raises(self):
        """Getting nonexistent bone should raise error."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        with pytest.raises(IndexError):
            skeleton.get_bone(10)

    def test_get_nonexistent_bone_by_name_returns_none(self):
        """Getting bone by nonexistent name should return None."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))

        result = skeleton.get_bone_by_name("nonexistent")
        assert result is None

    def test_duplicate_bone_names_raises(self):
        """Duplicate bone names should raise error."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="bone", parent_index=-1))

        with pytest.raises(ValueError):
            skeleton.add_bone(Bone(index=1, name="bone", parent_index=0))

    def test_validate_skeleton(self):
        """Valid skeleton should have no errors."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))

        errors = skeleton.validate()
        assert len(errors) == 0


class TestSkeletonBuildHumanoid:
    """Tests building a complete humanoid skeleton."""

    def test_build_simple_humanoid(self):
        """Build and verify a simple humanoid skeleton."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="humanoid")

        # Core
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="pelvis", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="spine", parent_index=1))
        skeleton.add_bone(Bone(index=3, name="chest", parent_index=2))
        skeleton.add_bone(Bone(index=4, name="neck", parent_index=3))
        skeleton.add_bone(Bone(index=5, name="head", parent_index=4))

        # Left arm
        skeleton.add_bone(Bone(index=6, name="l_shoulder", parent_index=3))
        skeleton.add_bone(Bone(index=7, name="l_elbow", parent_index=6))
        skeleton.add_bone(Bone(index=8, name="l_wrist", parent_index=7))

        # Right arm
        skeleton.add_bone(Bone(index=9, name="r_shoulder", parent_index=3))
        skeleton.add_bone(Bone(index=10, name="r_elbow", parent_index=9))
        skeleton.add_bone(Bone(index=11, name="r_wrist", parent_index=10))

        # Left leg
        skeleton.add_bone(Bone(index=12, name="l_hip", parent_index=1))
        skeleton.add_bone(Bone(index=13, name="l_knee", parent_index=12))
        skeleton.add_bone(Bone(index=14, name="l_ankle", parent_index=13))

        # Right leg
        skeleton.add_bone(Bone(index=15, name="r_hip", parent_index=1))
        skeleton.add_bone(Bone(index=16, name="r_knee", parent_index=15))
        skeleton.add_bone(Bone(index=17, name="r_ankle", parent_index=16))

        assert skeleton.bone_count == 18

        # Verify hierarchy
        chest_children = skeleton.get_bone_children(3)
        assert len(chest_children) == 3  # neck, l_shoulder, r_shoulder

        pelvis_children = skeleton.get_bone_children(1)
        assert len(pelvis_children) == 3  # spine, l_hip, r_hip

    def test_leaf_bones_in_humanoid(self):
        """Verify leaf bones in humanoid skeleton."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="l_hand", parent_index=1))
        skeleton.add_bone(Bone(index=3, name="r_hand", parent_index=1))
        skeleton.add_bone(Bone(index=4, name="head", parent_index=1))

        leaf_bones = skeleton.get_leaf_bones()
        assert 2 in leaf_bones  # l_hand
        assert 3 in leaf_bones  # r_hand
        assert 4 in leaf_bones  # head
        assert 1 not in leaf_bones  # spine


class TestSkeletonClone:
    """Tests for skeleton cloning/copying."""

    def test_clone_skeleton(self):
        """Cloned skeleton should be independent."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        original = Skeleton(name="original")
        original.add_bone(Bone(index=0, name="root", parent_index=-1))
        original.add_bone(Bone(index=1, name="spine", parent_index=0))

        clone = original.clone()

        assert clone.bone_count == original.bone_count
        assert clone.get_bone(0).name == "root"
        assert clone.name == "original"

    def test_clone_is_independent(self):
        """Modifications to clone should not affect original."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        original = Skeleton(name="original")
        original.add_bone(Bone(index=0, name="root", parent_index=-1))

        clone = original.clone()
        clone.add_bone(Bone(index=1, name="new_bone", parent_index=0))

        assert original.bone_count == 1
        assert clone.bone_count == 2


class TestSkeletonLength:
    """Tests for skeleton length and indexing."""

    def test_skeleton_len(self):
        """Skeleton should support len()."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))

        assert len(skeleton) == 2

    def test_skeleton_getitem(self):
        """Skeleton should support indexing."""
        from engine.animation.skeletal.skeleton import Skeleton, Bone

        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root", parent_index=-1))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))

        assert skeleton[0].name == "root"
        assert skeleton[1].name == "spine"


class TestCreateHumanoidSkeleton:
    """Tests for the humanoid skeleton factory function."""

    def test_create_humanoid_skeleton(self):
        """Factory function should create valid humanoid skeleton."""
        from engine.animation.skeletal.skeleton import create_humanoid_skeleton

        skeleton = create_humanoid_skeleton()

        assert skeleton.name == "humanoid"
        assert skeleton.bone_count > 0
        assert skeleton.get_bone_by_name("root") is not None
        assert skeleton.get_bone_by_name("head") is not None

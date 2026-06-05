"""Whitebox tests for skeleton.py.

Tests skeleton hierarchy, bone management, topological ordering,
root/leaf detection, and traversal algorithms.

Acceptance criteria:
- T-SKEL-1.1: Skeleton Hierarchy
  - Root bone detection (parent_index == -1)
  - Leaf bone detection
  - Topological order
"""

import pytest
from engine.core.math import Transform, Vec3, Quat, Mat4
from engine.animation.skeletal.skeleton import (
    Bone, Skeleton, AnimationMeta, animation_data, create_humanoid_skeleton
)


# =============================================================================
# Bone Tests
# =============================================================================

class TestBone:
    """Tests for Bone dataclass."""

    def test_bone_creation_basic(self):
        """Test basic bone creation with minimal parameters."""
        bone = Bone(index=0, name="root")
        assert bone.index == 0
        assert bone.name == "root"
        assert bone.parent_index == -1
        assert bone.is_root()

    def test_bone_creation_with_parent(self):
        """Test bone creation with parent reference."""
        bone = Bone(index=1, name="child", parent_index=0)
        assert bone.index == 1
        assert bone.parent_index == 0
        assert not bone.is_root()

    def test_bone_creation_with_bind_pose(self):
        """Test bone creation with explicit bind pose."""
        bind_pose = Transform(
            translation=Vec3(1.0, 2.0, 3.0),
            rotation=Quat(0, 0, 0, 1),
            scale=Vec3(1.0, 1.0, 1.0)
        )
        bone = Bone(index=0, name="root", local_bind_pose=bind_pose)
        assert bone.local_bind_pose.translation.x == 1.0
        assert bone.local_bind_pose.translation.y == 2.0
        assert bone.local_bind_pose.translation.z == 3.0

    def test_bone_validation_negative_index(self):
        """Test that negative bone index raises error."""
        with pytest.raises(ValueError, match="Bone index must be >= 0"):
            Bone(index=-1, name="invalid")

    def test_bone_validation_empty_name(self):
        """Test that empty bone name raises error."""
        with pytest.raises(ValueError, match="Bone name cannot be empty"):
            Bone(index=0, name="")

    def test_bone_validation_invalid_parent(self):
        """Test that parent_index < -1 raises error."""
        with pytest.raises(ValueError, match="Parent index must be >= -1"):
            Bone(index=1, name="child", parent_index=-2)

    def test_bone_validation_self_parent(self):
        """Test that bone cannot be its own parent."""
        with pytest.raises(ValueError, match="Bone cannot be its own parent"):
            Bone(index=0, name="self_parent", parent_index=0)

    def test_bone_is_root_detection(self):
        """Test root bone detection with parent_index == -1."""
        root_bone = Bone(index=0, name="root", parent_index=-1)
        child_bone = Bone(index=1, name="child", parent_index=0)

        assert root_bone.is_root() is True
        assert child_bone.is_root() is False

    def test_bone_copy(self):
        """Test deep copy of bone."""
        original = Bone(
            index=0,
            name="root",
            local_bind_pose=Transform(
                translation=Vec3(1, 2, 3),
                rotation=Quat(0.5, 0.5, 0.5, 0.5),
                scale=Vec3(2, 2, 2)
            )
        )
        copied = original.copy()

        assert copied.index == original.index
        assert copied.name == original.name
        assert copied.parent_index == original.parent_index
        # Modify original, ensure copy unchanged
        original.local_bind_pose.translation.x = 999
        assert copied.local_bind_pose.translation.x == 1

    def test_bone_repr(self):
        """Test bone string representation."""
        root = Bone(index=0, name="root")
        child = Bone(index=1, name="child", parent_index=0)

        assert "root" in repr(root)
        assert "parent=0" in repr(child)


# =============================================================================
# Skeleton Tests
# =============================================================================

class TestSkeleton:
    """Tests for Skeleton class."""

    def test_skeleton_creation_empty(self):
        """Test creating empty skeleton."""
        skeleton = Skeleton(name="test")
        assert skeleton.name == "test"
        assert skeleton.bone_count == 0
        assert len(skeleton.bones) == 0

    def test_skeleton_creation_empty_name_fails(self):
        """Test that empty skeleton name raises error."""
        with pytest.raises(ValueError, match="Skeleton name cannot be empty"):
            Skeleton(name="")

    def test_skeleton_creation_with_bones(self):
        """Test creating skeleton with initial bones."""
        bones = [
            Bone(index=0, name="root"),
            Bone(index=1, name="child", parent_index=0)
        ]
        skeleton = Skeleton(name="test", bones=bones)
        assert skeleton.bone_count == 2
        assert skeleton.bones[0].name == "root"
        assert skeleton.bones[1].name == "child"

    def test_skeleton_add_bone(self):
        """Test adding bones to skeleton."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="child", parent_index=0))

        assert skeleton.bone_count == 2

    def test_skeleton_add_bone_wrong_index(self):
        """Test adding bone with incorrect index fails."""
        skeleton = Skeleton(name="test")
        with pytest.raises(ValueError, match="does not match expected"):
            skeleton.add_bone(Bone(index=5, name="wrong_index"))

    def test_skeleton_add_bone_duplicate_name(self):
        """Test adding bone with duplicate name fails."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        with pytest.raises(ValueError, match="already exists"):
            skeleton.add_bone(Bone(index=1, name="root"))

    def test_skeleton_get_bone_by_index(self):
        """Test getting bone by index."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="child", parent_index=0))

        bone = skeleton.get_bone(1)
        assert bone.name == "child"

    def test_skeleton_get_bone_invalid_index(self):
        """Test getting bone with invalid index raises error."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))

        with pytest.raises(IndexError):
            skeleton.get_bone(5)
        with pytest.raises(IndexError):
            skeleton.get_bone(-1)

    def test_skeleton_get_bone_by_name(self):
        """Test getting bone by name."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))

        bone = skeleton.get_bone_by_name("spine")
        assert bone is not None
        assert bone.index == 1

    def test_skeleton_get_bone_by_name_not_found(self):
        """Test getting non-existent bone by name returns None."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))

        assert skeleton.get_bone_by_name("missing") is None

    def test_skeleton_get_bone_index(self):
        """Test getting bone index by name."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))

        assert skeleton.get_bone_index("spine") == 1
        assert skeleton.get_bone_index("missing") == -1


class TestSkeletonRootBoneDetection:
    """Tests for T-SKEL-1.1: Root bone detection."""

    def test_single_root(self):
        """Test skeleton with single root bone."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="child", parent_index=0))
        skeleton._rebuild_caches()

        assert len(skeleton.root_bone_indices) == 1
        assert 0 in skeleton.root_bone_indices

    def test_multiple_roots(self):
        """Test skeleton with multiple root bones."""
        skeleton = Skeleton(name="multi_root")
        skeleton.add_bone(Bone(index=0, name="root_a"))
        skeleton.add_bone(Bone(index=1, name="root_b"))
        skeleton.add_bone(Bone(index=2, name="child_a", parent_index=0))
        skeleton._rebuild_caches()

        assert len(skeleton.root_bone_indices) == 2
        assert 0 in skeleton.root_bone_indices
        assert 1 in skeleton.root_bone_indices

    def test_root_bone_filter(self):
        """Test filtering bones by root status."""
        skeleton = create_humanoid_skeleton()
        root_bones = skeleton.find_bones_by_pattern(lambda b: b.is_root())

        assert len(root_bones) == 1
        assert root_bones[0].name == "root"


class TestSkeletonLeafBoneDetection:
    """Tests for T-SKEL-1.1: Leaf bone detection."""

    def test_get_leaf_bones_simple(self):
        """Test getting leaf bones in simple hierarchy."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="middle", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="leaf", parent_index=1))
        skeleton._rebuild_caches()

        leaves = skeleton.get_leaf_bones()
        assert len(leaves) == 1
        assert 2 in leaves

    def test_get_leaf_bones_multiple(self):
        """Test getting multiple leaf bones."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="left", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="right", parent_index=0))
        skeleton.add_bone(Bone(index=3, name="left_tip", parent_index=1))
        skeleton.add_bone(Bone(index=4, name="right_tip", parent_index=2))
        skeleton._rebuild_caches()

        leaves = skeleton.get_leaf_bones()
        assert len(leaves) == 2
        assert 3 in leaves
        assert 4 in leaves

    def test_leaf_bones_humanoid(self):
        """Test leaf bone detection on humanoid skeleton."""
        skeleton = create_humanoid_skeleton()
        leaves = skeleton.get_leaf_bones()

        # Humanoid should have leaf bones at extremities
        leaf_names = [skeleton.get_bone(i).name for i in leaves]
        assert "hand_l" in leaf_names
        assert "hand_r" in leaf_names
        assert "head" in leaf_names
        assert "foot_l" in leaf_names
        assert "foot_r" in leaf_names


class TestSkeletonTopologicalOrder:
    """Tests for T-SKEL-1.1: Topological ordering."""

    def test_parent_before_child_indices(self):
        """Test that parent bones always have lower indices than children."""
        skeleton = create_humanoid_skeleton()

        for bone in skeleton.bones:
            if bone.parent_index >= 0:
                assert bone.parent_index < bone.index, \
                    f"Parent {bone.parent_index} must be < child {bone.index}"

    def test_validate_topological_order(self):
        """Test skeleton validation catches invalid parent references."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="child", parent_index=0))

        errors = skeleton.validate()
        assert len(errors) == 0

    def test_validate_invalid_parent_reference(self):
        """Test validation catches out-of-range parent index."""
        skeleton = Skeleton(name="test")
        # Manually create bones bypassing validation
        root = Bone(index=0, name="root")
        skeleton._bones.append(root)
        skeleton._bone_name_to_index["root"] = 0

        # Create bone with invalid parent
        bad_bone = Bone(index=1, name="bad", parent_index=99)
        skeleton._bones.append(bad_bone)
        skeleton._bone_name_to_index["bad"] = 1
        skeleton._rebuild_caches()

        errors = skeleton.validate()
        assert any("invalid parent" in e.lower() for e in errors)


class TestSkeletonHierarchy:
    """Tests for skeleton hierarchy operations."""

    def test_get_bone_children(self):
        """Test getting direct children of a bone."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="left", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="right", parent_index=0))
        skeleton.add_bone(Bone(index=3, name="left_child", parent_index=1))
        skeleton._rebuild_caches()

        children = skeleton.get_bone_children(0)
        assert len(children) == 2
        assert 1 in children
        assert 2 in children

    def test_get_bone_descendants(self):
        """Test getting all descendants of a bone."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="mid", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="leaf", parent_index=1))
        skeleton._rebuild_caches()

        descendants = skeleton.get_bone_descendants(0)
        assert 1 in descendants
        assert 2 in descendants
        assert 0 not in descendants

    def test_get_bone_chain(self):
        """Test getting bone chain between two bones."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="chest", parent_index=1))
        skeleton.add_bone(Bone(index=3, name="head", parent_index=2))
        skeleton._rebuild_caches()

        chain = skeleton.get_bone_chain(0, 3)
        assert chain == [0, 1, 2, 3]

    def test_get_bone_chain_same_bone(self):
        """Test bone chain when start == end."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton._rebuild_caches()

        chain = skeleton.get_bone_chain(0, 0)
        assert chain == [0]

    def test_get_bone_depth(self):
        """Test getting bone depth in hierarchy."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="l1", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="l2", parent_index=1))
        skeleton._rebuild_caches()

        assert skeleton.get_depth(0) == 0
        assert skeleton.get_depth(1) == 1
        assert skeleton.get_depth(2) == 2

    def test_get_max_depth(self):
        """Test getting maximum hierarchy depth."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="l1", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="l2", parent_index=1))
        skeleton.add_bone(Bone(index=3, name="l3", parent_index=2))
        skeleton._rebuild_caches()

        assert skeleton.get_max_depth() == 3

    def test_get_bone_path(self):
        """Test getting path string from root to bone."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="spine", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="arm", parent_index=1))
        skeleton._rebuild_caches()

        path = skeleton.get_bone_path(2)
        assert path == "root/spine/arm"


class TestSkeletonTraversal:
    """Tests for skeleton traversal methods."""

    def test_traverse_depth_first(self):
        """Test depth-first traversal order."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="left", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="right", parent_index=0))
        skeleton.add_bone(Bone(index=3, name="left_leaf", parent_index=1))
        skeleton._rebuild_caches()

        visited = []
        skeleton.traverse_depth_first(lambda bone, depth: visited.append(bone.name))

        # DFS should visit left subtree before right
        assert visited.index("left") < visited.index("right")
        assert visited.index("left_leaf") < visited.index("right")

    def test_traverse_breadth_first(self):
        """Test breadth-first traversal order."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="left", parent_index=0))
        skeleton.add_bone(Bone(index=2, name="right", parent_index=0))
        skeleton.add_bone(Bone(index=3, name="left_leaf", parent_index=1))
        skeleton._rebuild_caches()

        visited = []
        skeleton.traverse_breadth_first(lambda bone, depth: visited.append(bone.name))

        # BFS: root first, then all depth-1, then all depth-2
        assert visited[0] == "root"
        assert "left" in visited[1:3]
        assert "right" in visited[1:3]
        assert visited[-1] == "left_leaf"


class TestSkeletonTransforms:
    """Tests for skeleton transform computation."""

    def test_compute_world_transforms_identity(self):
        """Test world transforms with identity local transforms."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="child", parent_index=0))
        skeleton._rebuild_caches()

        world_matrices = skeleton.compute_world_transforms()

        assert len(world_matrices) == 2
        # Root should be identity
        root_mat = world_matrices[0]
        assert abs(root_mat.m[12]) < 1e-6  # translation x
        assert abs(root_mat.m[13]) < 1e-6  # translation y
        assert abs(root_mat.m[14]) < 1e-6  # translation z

    def test_compute_world_transforms_with_translation(self):
        """Test world transforms with translation accumulation."""
        skeleton = Skeleton(name="test")
        root_pose = Transform(translation=Vec3(1, 0, 0))
        child_pose = Transform(translation=Vec3(0, 1, 0))

        skeleton.add_bone(Bone(index=0, name="root", local_bind_pose=root_pose))
        skeleton.add_bone(Bone(index=1, name="child", parent_index=0,
                               local_bind_pose=child_pose))
        skeleton._rebuild_caches()

        world_matrices = skeleton.compute_world_transforms()

        # Child world position should be (1, 1, 0)
        child_mat = world_matrices[1]
        assert abs(child_mat.m[12] - 1.0) < 1e-6
        assert abs(child_mat.m[13] - 1.0) < 1e-6

    def test_compute_skinning_matrices(self):
        """Test skinning matrix computation."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton._rebuild_caches()
        skeleton.compute_inverse_bind_poses()

        world = skeleton.compute_world_transforms()
        skinning = skeleton.compute_skinning_matrices(world)

        assert len(skinning) == 1
        # At bind pose, skinning matrix should be identity
        mat = skinning[0]
        assert abs(mat.m[0] - 1.0) < 1e-6
        assert abs(mat.m[5] - 1.0) < 1e-6
        assert abs(mat.m[10] - 1.0) < 1e-6


class TestSkeletonUtilities:
    """Tests for skeleton utility methods."""

    def test_skeleton_clone(self):
        """Test deep cloning of skeleton."""
        original = create_humanoid_skeleton()
        cloned = original.clone()

        assert cloned.name == original.name
        assert cloned.bone_count == original.bone_count

        # Modify original, ensure clone unchanged
        original._bones[0].name = "modified"
        assert cloned.bones[0].name == "root"

    def test_skeleton_iter(self):
        """Test iterating over skeleton bones."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="child", parent_index=0))
        skeleton._rebuild_caches()

        names = [bone.name for bone in skeleton]
        assert names == ["root", "child"]

    def test_skeleton_len(self):
        """Test skeleton length."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="child", parent_index=0))

        assert len(skeleton) == 2

    def test_skeleton_getitem(self):
        """Test skeleton indexing."""
        skeleton = Skeleton(name="test")
        skeleton.add_bone(Bone(index=0, name="root"))
        skeleton.add_bone(Bone(index=1, name="child", parent_index=0))

        assert skeleton[0].name == "root"
        assert skeleton[1].name == "child"

    def test_find_bones_by_pattern(self):
        """Test finding bones by custom pattern."""
        skeleton = create_humanoid_skeleton()

        # Find all arm bones
        arm_bones = skeleton.find_bones_by_pattern(
            lambda b: "arm" in b.name.lower()
        )

        assert len(arm_bones) > 0
        for bone in arm_bones:
            assert "arm" in bone.name.lower()


class TestHumanoidSkeleton:
    """Tests for humanoid skeleton factory."""

    def test_create_humanoid_skeleton(self):
        """Test humanoid skeleton creation."""
        skeleton = create_humanoid_skeleton()

        assert skeleton.name == "humanoid"
        assert skeleton.bone_count == 21

    def test_humanoid_has_expected_bones(self):
        """Test humanoid has expected bone names."""
        skeleton = create_humanoid_skeleton()

        expected_bones = [
            "root", "pelvis", "spine_01", "neck", "head",
            "hand_l", "hand_r", "foot_l", "foot_r"
        ]

        for name in expected_bones:
            assert skeleton.get_bone_by_name(name) is not None, f"Missing bone: {name}"

    def test_humanoid_symmetric_structure(self):
        """Test humanoid has symmetric left/right bones."""
        skeleton = create_humanoid_skeleton()

        left_bones = [b for b in skeleton if "_l" in b.name]
        right_bones = [b for b in skeleton if "_r" in b.name]

        assert len(left_bones) == len(right_bones)


class TestAnimationDecorator:
    """Tests for animation_data decorator."""

    def test_animation_data_decorator(self):
        """Test that animation_data decorator sets attributes."""
        assert hasattr(Bone, '_animation_data')
        assert Bone._animation_data is True
        assert Bone._animation_type == "Bone"

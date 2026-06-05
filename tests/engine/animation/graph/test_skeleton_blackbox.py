"""
Blackbox Tests for T-AG-1.3: Skeleton and Bone Hierarchy

These tests verify the public API behavior of the Bone and Skeleton classes
from skeleton.py without knowledge of internal implementation details.

Note: There are TWO Bone/Skeleton pairs in the codebase:
  1. animation_graph.py: Bone/Skeleton - dataclass-based, used by animation graph
  2. skeleton.py: Bone/Skeleton - hierarchy-based with add_child/add_bone methods

This test file tests skeleton.py's classes, which are exported as:
  - SkeletonBone (alias for skeleton.py's Bone)
  - SkeletonHierarchy (alias for skeleton.py's Skeleton)

Or can be imported directly from engine.animation.graph.skeleton.

API Summary (from skeleton.py implementation):
    Bone(name, parent=None, bind_pose=None)
        .name: str
        .parent: Bone | None
        .children: list[Bone]
        .bind_pose: Transform
        .add_child(bone) -> None
        .remove_child(bone) -> bool
        .is_root: bool
        .depth: int
        .get_root() -> Bone
        .get_siblings() -> list[Bone]
        .is_ancestor_of(other) -> bool
        .is_descendant_of(ancestor) -> bool
        .get_descendants() -> list[Bone]

    Skeleton(name="skeleton")
        .name: str
        .bone_count: int
        .bones: dict[str, Bone]
        .root_bones: list[Bone]
        .add_bone(name, parent_name=None, bind_pose=None) -> Bone
        .remove_bone(name) -> bool
        .has_bone(name) -> bool
        .get_bone(name) -> Bone | None
        .get_chain(start_name, end_name) -> list[Bone]
        .is_valid() -> bool
        .validate() -> list[str]
"""

import pytest
import time

# Import directly from skeleton.py to avoid confusion with animation_graph.py classes
from engine.animation.graph.skeleton import Bone, Skeleton


# =============================================================================
# SECTION 1: IMPORTS - Verify contract exports exist
# =============================================================================

class TestContractImports:
    """Test that all contract-specified classes are importable."""

    def test_bone_importable(self):
        """Bone class should be importable from engine.animation.graph.skeleton."""
        from engine.animation.graph.skeleton import Bone
        assert Bone is not None

    def test_skeleton_importable(self):
        """Skeleton class should be importable from engine.animation.graph.skeleton."""
        from engine.animation.graph.skeleton import Skeleton
        assert Skeleton is not None

    def test_skeleton_bone_alias_importable(self):
        """SkeletonBone alias should be importable from engine.animation.graph."""
        from engine.animation.graph import SkeletonBone
        assert SkeletonBone is not None
        # Should be the same as skeleton.py's Bone
        from engine.animation.graph.skeleton import Bone
        assert SkeletonBone is Bone

    def test_skeleton_hierarchy_alias_importable(self):
        """SkeletonHierarchy alias should be importable from engine.animation.graph."""
        from engine.animation.graph import SkeletonHierarchy
        assert SkeletonHierarchy is not None
        # Should be the same as skeleton.py's Skeleton
        from engine.animation.graph.skeleton import Skeleton
        assert SkeletonHierarchy is Skeleton


# =============================================================================
# SECTION 2: BONE CREATION AND ATTRIBUTES
# =============================================================================

class TestBoneCreation:
    """Test Bone creation and basic attributes per contract."""

    def test_bone_with_name_only(self):
        """Bone can be created with just a name."""
        bone = Bone("spine")
        assert bone.name == "spine"

    def test_bone_with_name_and_parent_none(self):
        """Bone can be created with name and parent=None."""
        bone = Bone("spine", parent=None)
        assert bone.name == "spine"
        assert bone.parent is None

    def test_bone_with_parent_bone(self):
        """Bone can be created with another Bone as parent."""
        parent = Bone("spine")
        child = Bone("chest", parent=parent)
        assert child.name == "chest"
        assert child.parent is parent

    def test_bone_has_children_attribute(self):
        """Bone should have a children attribute (list)."""
        bone = Bone("spine")
        assert hasattr(bone, 'children')
        # Children should be iterable (list-like)
        assert hasattr(bone.children, '__iter__')

    def test_bone_has_bind_pose_attribute(self):
        """Per architecture: Bone has bind_pose: Transform."""
        bone = Bone("spine")
        assert hasattr(bone, 'bind_pose')


# =============================================================================
# SECTION 3: BONE PARENT/CHILD RELATIONSHIPS
# =============================================================================

class TestBoneRelationships:
    """Test Bone parent/child relationship management."""

    def test_add_child_method_exists(self):
        """Bone should have add_child() method per contract."""
        bone = Bone("spine")
        assert hasattr(bone, 'add_child')
        assert callable(bone.add_child)

    def test_remove_child_method_exists(self):
        """Bone should have remove_child() method per contract."""
        bone = Bone("spine")
        assert hasattr(bone, 'remove_child')
        assert callable(bone.remove_child)

    def test_add_child_adds_to_children(self):
        """add_child() should add bone to children list."""
        parent = Bone("spine")
        child = Bone("chest")
        parent.add_child(child)
        assert child in parent.children

    def test_remove_child_removes_from_children(self):
        """remove_child() should remove bone from children list."""
        parent = Bone("spine")
        child = Bone("chest")
        parent.add_child(child)
        parent.remove_child(child)
        assert child not in parent.children

    def test_add_child_sets_parent_reference(self):
        """Adding a child should set the child's parent reference."""
        parent = Bone("spine")
        child = Bone("chest")
        parent.add_child(child)
        # After adding, child.parent should reference parent
        assert child.parent is parent

    def test_remove_child_clears_parent_reference(self):
        """Removing a child should clear the child's parent reference."""
        parent = Bone("spine")
        child = Bone("chest")
        parent.add_child(child)
        parent.remove_child(child)
        # After removal, parent reference should be None
        assert child.parent is None

    def test_multiple_children(self):
        """A bone can have multiple children."""
        parent = Bone("spine")
        child1 = Bone("left_shoulder")
        child2 = Bone("right_shoulder")
        parent.add_child(child1)
        parent.add_child(child2)
        assert child1 in parent.children
        assert child2 in parent.children
        assert len(parent.children) >= 2

    def test_deep_hierarchy(self):
        """Test a deep bone hierarchy: root -> spine -> chest -> neck -> head."""
        root = Bone("root")
        spine = Bone("spine")
        chest = Bone("chest")
        neck = Bone("neck")
        head = Bone("head")

        root.add_child(spine)
        spine.add_child(chest)
        chest.add_child(neck)
        neck.add_child(head)

        # Verify chain
        assert spine.parent is root
        assert chest.parent is spine
        assert neck.parent is chest
        assert head.parent is neck


# =============================================================================
# SECTION 4: SKELETON CONSTRUCTION
# =============================================================================

class TestSkeletonConstruction:
    """Test Skeleton construction per actual API."""

    def test_skeleton_with_name(self):
        """Skeleton can be created with a name."""
        skeleton = Skeleton("humanoid")
        assert skeleton is not None
        assert skeleton.name == "humanoid"

    def test_skeleton_default_name(self):
        """Skeleton has a default name when none provided."""
        skeleton = Skeleton()
        assert skeleton is not None
        assert skeleton.name == "skeleton"

    def test_skeleton_has_root_bones_attribute(self):
        """Skeleton should have root_bones attribute per architecture."""
        skeleton = Skeleton("test")
        assert hasattr(skeleton, 'root_bones')
        assert isinstance(skeleton.root_bones, list)

    def test_skeleton_has_bones_dict(self):
        """Skeleton should have bones dictionary per architecture."""
        skeleton = Skeleton("test")
        assert hasattr(skeleton, 'bones')
        assert isinstance(skeleton.bones, dict)

    def test_skeleton_has_bone_count(self):
        """Skeleton should have bone_count attribute per architecture."""
        skeleton = Skeleton("test")
        # bone_count may be a property or attribute
        assert hasattr(skeleton, 'bone_count')
        assert skeleton.bone_count == 0


# =============================================================================
# SECTION 5: SKELETON get_bone() - O(1) LOOKUP
# =============================================================================

class TestSkeletonGetBone:
    """Test Skeleton.get_bone() per contract: O(1) lookup."""

    def test_get_bone_method_exists(self):
        """Skeleton should have get_bone() method."""
        skeleton = Skeleton("test")
        assert hasattr(skeleton, 'get_bone')
        assert callable(skeleton.get_bone)

    def test_get_bone_returns_correct_bone(self):
        """get_bone("name") returns the bone with that name."""
        skeleton = Skeleton("test")
        skeleton.add_bone("root")
        skeleton.add_bone("spine", parent_name="root")

        result = skeleton.get_bone("root")
        assert result is not None
        assert result.name == "root"

    def test_get_bone_returns_nested_bone(self):
        """get_bone() can find bones deep in hierarchy."""
        skeleton = Skeleton("test")
        skeleton.add_bone("root")
        skeleton.add_bone("spine", parent_name="root")
        skeleton.add_bone("chest", parent_name="spine")

        result = skeleton.get_bone("chest")
        assert result is not None
        assert result.name == "chest"

    def test_get_bone_nonexistent_returns_none(self):
        """get_bone() returns None for nonexistent bone."""
        skeleton = Skeleton("test")
        skeleton.add_bone("root")

        result = skeleton.get_bone("nonexistent_bone")
        assert result is None

    def test_get_bone_o1_performance(self):
        """get_bone() should be O(1) - constant time regardless of size."""
        # Build a skeleton with 100 bones
        skeleton = Skeleton("test")
        skeleton.add_bone("bone_0")
        for i in range(1, 100):
            skeleton.add_bone(f"bone_{i}", parent_name=f"bone_{i-1}")

        # Time lookups - should be roughly constant
        times = []
        for target in ["bone_0", "bone_50", "bone_99"]:
            start = time.perf_counter()
            for _ in range(1000):
                skeleton.get_bone(target)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # O(1) means times should be roughly similar (within 5x)
        max_time = max(times)
        min_time = min(times)
        assert max_time < min_time * 5, f"get_bone not O(1): times={times}"


# =============================================================================
# SECTION 6: SKELETON get_chain() - IK CHAIN
# =============================================================================

class TestSkeletonGetChain:
    """Test Skeleton.get_chain() for IK integration."""

    def test_get_chain_method_exists(self):
        """Skeleton should have get_chain() method."""
        skeleton = Skeleton("test")
        assert hasattr(skeleton, 'get_chain')
        assert callable(skeleton.get_chain)

    def test_get_chain_returns_list(self):
        """get_chain() returns a list."""
        skeleton = Skeleton("test")
        skeleton.add_bone("root")
        skeleton.add_bone("spine", parent_name="root")

        chain = skeleton.get_chain("spine", "root")
        assert isinstance(chain, list)

    def test_get_chain_ordered_start_to_end(self):
        """get_chain returns ordered list from start to end."""
        # Build: root -> spine -> chest -> shoulder -> arm -> hand
        skeleton = Skeleton("test")
        skeleton.add_bone("root")
        skeleton.add_bone("spine", parent_name="root")
        skeleton.add_bone("chest", parent_name="spine")
        skeleton.add_bone("shoulder", parent_name="chest")
        skeleton.add_bone("arm", parent_name="shoulder")
        skeleton.add_bone("hand", parent_name="arm")

        # Get chain from shoulder to hand (typical IK query)
        chain = skeleton.get_chain("shoulder", "hand")

        # Should be ordered list
        assert isinstance(chain, list)
        assert len(chain) >= 2

        # Extract names
        chain_names = [b.name for b in chain]
        assert "shoulder" in chain_names
        assert "hand" in chain_names

        # shoulder should come before hand in the chain
        shoulder_idx = chain_names.index("shoulder")
        hand_idx = chain_names.index("hand")
        assert shoulder_idx < hand_idx, "Chain should be ordered from start to end"

    def test_get_chain_single_bone(self):
        """get_chain with same start and end returns single-element list."""
        skeleton = Skeleton("test")
        skeleton.add_bone("root")

        chain = skeleton.get_chain("root", "root")
        assert isinstance(chain, list)
        assert len(chain) == 1
        assert chain[0].name == "root"

    def test_get_chain_invalid_bones_returns_empty(self):
        """get_chain with invalid bone names returns empty list."""
        skeleton = Skeleton("test")
        skeleton.add_bone("root")

        chain = skeleton.get_chain("nonexistent1", "nonexistent2")
        # Should return empty list for invalid bones
        assert chain == []

    def test_get_chain_through_common_ancestor(self):
        """get_chain between bones on different branches goes through common ancestor."""
        # Create two separate branches with common ancestor (root)
        skeleton = Skeleton("test")
        skeleton.add_bone("root")
        skeleton.add_bone("left_arm", parent_name="root")
        skeleton.add_bone("right_arm", parent_name="root")
        skeleton.add_bone("left_hand", parent_name="left_arm")
        skeleton.add_bone("right_hand", parent_name="right_arm")

        # Chain between bones on different branches
        chain = skeleton.get_chain("left_hand", "right_hand")

        # Should find path through common ancestor
        assert isinstance(chain, list)
        chain_names = [b.name for b in chain]
        # Path should include root as common ancestor
        assert "root" in chain_names


# =============================================================================
# SECTION 7: SKELETON is_valid() - VALIDATION
# =============================================================================

class TestSkeletonValidation:
    """Test Skeleton.is_valid() per contract."""

    def test_is_valid_method_exists(self):
        """Skeleton should have is_valid() method."""
        skeleton = Skeleton("test")
        assert hasattr(skeleton, 'is_valid')
        assert callable(skeleton.is_valid)

    def test_is_valid_returns_bool(self):
        """is_valid() returns True or False."""
        skeleton = Skeleton("test")
        skeleton.add_bone("root")

        result = skeleton.is_valid()
        assert isinstance(result, bool)

    def test_valid_skeleton_single_root(self):
        """Single root skeleton is valid."""
        skeleton = Skeleton("test")
        skeleton.add_bone("root")
        skeleton.add_bone("spine", parent_name="root")

        assert skeleton.is_valid() is True

    def test_valid_skeleton_complex_hierarchy(self):
        """Complex hierarchy with single root is valid."""
        # Build humanoid skeleton
        skeleton = Skeleton("humanoid")
        skeleton.add_bone("root")
        skeleton.add_bone("spine", parent_name="root")
        skeleton.add_bone("chest", parent_name="spine")
        skeleton.add_bone("neck", parent_name="chest")
        skeleton.add_bone("head", parent_name="neck")
        skeleton.add_bone("left_shoulder", parent_name="chest")
        skeleton.add_bone("right_shoulder", parent_name="chest")

        assert skeleton.is_valid() is True

    def test_valid_skeleton_no_orphans(self):
        """Skeleton with all bones connected is valid (no orphans)."""
        skeleton = Skeleton("test")
        skeleton.add_bone("root")
        skeleton.add_bone("child1", parent_name="root")
        skeleton.add_bone("child2", parent_name="root")

        # All bones are connected to root, so valid
        assert skeleton.is_valid() is True

    def test_empty_skeleton_not_valid(self):
        """Empty skeleton is not valid."""
        skeleton = Skeleton("empty")
        # Empty skeleton should not be valid
        assert skeleton.is_valid() is False


# =============================================================================
# SECTION 8: EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_bone_empty_name_raises(self):
        """Bone with empty name raises ValueError."""
        with pytest.raises(ValueError):
            Bone("")

    def test_bone_special_characters_in_name(self):
        """Bone with special characters in name."""
        bone = Bone("bone_123-test.left")
        assert bone.name == "bone_123-test.left"

    def test_skeleton_large_hierarchy(self):
        """Test skeleton with many bones (stress test)."""
        skeleton = Skeleton("large")
        skeleton.add_bone("bone_0")
        # Build chain of 500 bones
        for i in range(1, 500):
            skeleton.add_bone(f"bone_{i}", parent_name=f"bone_{i-1}")

        assert skeleton is not None
        assert skeleton.is_valid() is True

        # Should still have O(1) lookup
        bone = skeleton.get_bone("bone_250")
        assert bone is not None
        assert bone.name == "bone_250"

    def test_remove_nonexistent_child(self):
        """Removing a child that isn't in the list - returns False."""
        parent = Bone("parent")
        not_a_child = Bone("stranger")

        # Should return False, not raise
        result = parent.remove_child(not_a_child)
        assert result is False

    def test_add_child_twice_raises(self):
        """Adding the same child twice raises ValueError."""
        parent = Bone("parent")
        child = Bone("child")

        parent.add_child(child)
        with pytest.raises(ValueError):
            parent.add_child(child)


# =============================================================================
# SECTION 9: INTEGRATION TESTS
# =============================================================================

class TestSkeletonIntegration:
    """Integration tests combining multiple skeleton features."""

    def test_humanoid_skeleton_workflow(self):
        """Full workflow: create humanoid skeleton, query bones, get chains."""
        # Create humanoid skeleton
        skeleton = Skeleton("humanoid")

        # Add bones
        skeleton.add_bone("hips")
        skeleton.add_bone("spine", parent_name="hips")
        skeleton.add_bone("chest", parent_name="spine")
        skeleton.add_bone("neck", parent_name="chest")
        skeleton.add_bone("head", parent_name="neck")

        # Left arm chain
        skeleton.add_bone("left_shoulder", parent_name="chest")
        skeleton.add_bone("left_upper_arm", parent_name="left_shoulder")
        skeleton.add_bone("left_forearm", parent_name="left_upper_arm")
        skeleton.add_bone("left_hand", parent_name="left_forearm")

        # Right arm chain
        skeleton.add_bone("right_shoulder", parent_name="chest")
        skeleton.add_bone("right_upper_arm", parent_name="right_shoulder")
        skeleton.add_bone("right_forearm", parent_name="right_upper_arm")
        skeleton.add_bone("right_hand", parent_name="right_forearm")

        # Validate
        assert skeleton.is_valid() is True

        # Query bones
        assert skeleton.get_bone("hips").name == "hips"
        assert skeleton.get_bone("left_hand").name == "left_hand"
        assert skeleton.get_bone("right_hand").name == "right_hand"
        assert skeleton.get_bone("head").name == "head"

        # Get IK chain for left arm
        chain = skeleton.get_chain("left_shoulder", "left_hand")
        assert isinstance(chain, list)
        assert len(chain) >= 2

    def test_add_bone_returns_bone(self):
        """add_bone() returns the created Bone object."""
        skeleton = Skeleton("test")
        bone = skeleton.add_bone("root")

        assert bone is not None
        assert bone.name == "root"

    def test_bone_count_tracks_additions(self):
        """bone_count increments with each add_bone()."""
        skeleton = Skeleton("test")
        assert skeleton.bone_count == 0

        skeleton.add_bone("root")
        assert skeleton.bone_count == 1

        skeleton.add_bone("child", parent_name="root")
        assert skeleton.bone_count == 2


# =============================================================================
# SECTION 10: CONTRACT COMPLIANCE VERIFICATION
# =============================================================================

class TestContractCompliance:
    """Verify exact contract compliance."""

    def test_contract_example_bone_creation(self):
        """Verify contract example: bone = Bone("spine")"""
        bone = Bone("spine")
        assert bone is not None
        assert bone.name == "spine"

    def test_contract_example_child_creation(self):
        """Verify contract example: child = Bone("chest", parent=bone)"""
        bone = Bone("spine")
        child = Bone("chest", parent=bone)
        assert child is not None
        assert child.name == "chest"
        assert child.parent is bone

    def test_contract_example_add_child(self):
        """Verify contract example: bone.add_child(child)"""
        bone = Bone("spine")
        child = Bone("chest")
        bone.add_child(child)
        assert child in bone.children

    def test_contract_example_remove_child(self):
        """Verify contract example: bone.remove_child(child)"""
        bone = Bone("spine")
        child = Bone("chest")
        bone.add_child(child)
        result = bone.remove_child(child)
        assert result is True
        assert child not in bone.children

    def test_contract_example_skeleton_creation(self):
        """Verify contract example: skeleton = Skeleton("humanoid")"""
        skeleton = Skeleton("humanoid")
        assert skeleton is not None

    def test_contract_example_get_bone(self):
        """Verify contract example: bone = skeleton.get_bone("spine")"""
        skeleton = Skeleton("test")
        skeleton.add_bone("root")
        skeleton.add_bone("spine", parent_name="root")

        bone = skeleton.get_bone("spine")
        assert bone is not None
        assert bone.name == "spine"

    def test_contract_example_get_chain(self):
        """Verify contract example: chain = skeleton.get_chain("shoulder", "hand")"""
        skeleton = Skeleton("test")
        skeleton.add_bone("root")
        skeleton.add_bone("shoulder", parent_name="root")
        skeleton.add_bone("arm", parent_name="shoulder")
        skeleton.add_bone("hand", parent_name="arm")

        chain = skeleton.get_chain("shoulder", "hand")
        assert isinstance(chain, list)

    def test_contract_example_is_valid(self):
        """Verify contract example: is_valid = skeleton.is_valid()"""
        skeleton = Skeleton("test")
        skeleton.add_bone("root")
        is_valid = skeleton.is_valid()
        assert isinstance(is_valid, bool)

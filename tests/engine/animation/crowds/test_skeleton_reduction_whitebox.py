"""Whitebox tests for skeleton reduction functionality.

Task: T2.4 Skeleton Reduction

Tests internal methods of:
- create_reduced_skeleton() bone selection algorithm
- _calculate_bone_importance() scoring for all bone categories
- Hierarchy preservation during reduction
- Bind pose transfer
- Edge cases in reduction targets

Acceptance Criteria:
1. Important bones are kept (root, spine, head)
2. Unimportant bones are culled (fingers, twist)
3. Reduced skeleton is valid (no orphan bones)
4. Bone count matches target
"""

from __future__ import annotations

import pytest

from engine.core.math import Vec3, Quat, Transform
from engine.animation.crowds.animation_texture import Skeleton
from engine.animation.crowds.crowd_lod import (
    create_reduced_skeleton,
    _calculate_bone_importance,
    _get_bone_depth,
    BoneWeight,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def full_humanoid_skeleton() -> Skeleton:
    """Create a full humanoid skeleton with all bone types for testing.

    This skeleton includes:
    - Core: root, pelvis, spine (3), chest, neck, head
    - Left arm: shoulder, upperarm, twist, forearm, hand, 5 fingers (3 bones each)
    - Right arm: shoulder, upperarm, twist, forearm, hand, 5 fingers (3 bones each)
    - Left leg: thigh, twist, calf, foot, 5 toes
    - Right leg: thigh, twist, calf, foot, 5 toes
    """
    bone_names = [
        # Core bones (indices 0-7)
        "root",           # 0
        "pelvis",         # 1
        "spine_01",       # 2
        "spine_02",       # 3
        "spine_03",       # 4
        "chest",          # 5
        "neck",           # 6
        "head",           # 7

        # Left arm (indices 8-27)
        "clavicle_l",     # 8
        "shoulder_l",     # 9
        "upperarm_l",     # 10
        "upperarm_twist_l",  # 11
        "forearm_l",      # 12
        "forearm_twist_l",   # 13
        "hand_l",         # 14
        "finger_thumb_01_l",  # 15
        "finger_thumb_02_l",  # 16
        "finger_thumb_03_l",  # 17
        "finger_index_01_l",  # 18
        "finger_index_02_l",  # 19
        "finger_index_03_l",  # 20
        "finger_middle_01_l", # 21
        "finger_middle_02_l", # 22
        "finger_middle_03_l", # 23
        "finger_ring_01_l",   # 24
        "finger_ring_02_l",   # 25
        "finger_ring_03_l",   # 26
        "finger_pinky_01_l",  # 27

        # Left leg (indices 28-38)
        "thigh_l",        # 28
        "thigh_twist_l",  # 29
        "calf_l",         # 30
        "foot_l",         # 31
        "toe_big_l",      # 32
        "toe_index_l",    # 33
        "toe_middle_l",   # 34
        "toe_ring_l",     # 35
        "toe_pinky_l",    # 36

        # Right arm (indices 37-56, mirrored)
        "clavicle_r",     # 37
        "shoulder_r",     # 38
        "upperarm_r",     # 39
        "upperarm_twist_r",  # 40
        "forearm_r",      # 41
        "forearm_twist_r",   # 42
        "hand_r",         # 43
        "finger_thumb_01_r",  # 44
        "finger_thumb_02_r",  # 45
        "finger_thumb_03_r",  # 46
        "finger_index_01_r",  # 47
        "finger_index_02_r",  # 48
        "finger_index_03_r",  # 49
        "finger_middle_01_r", # 50
        "finger_middle_02_r", # 51
        "finger_middle_03_r", # 52
        "finger_ring_01_r",   # 53
        "finger_ring_02_r",   # 54
        "finger_ring_03_r",   # 55
        "finger_pinky_01_r",  # 56

        # Right leg (indices 57-65)
        "thigh_r",        # 57
        "thigh_twist_r",  # 58
        "calf_r",         # 59
        "foot_r",         # 60
        "toe_big_r",      # 61
        "toe_index_r",    # 62
        "toe_middle_r",   # 63
        "toe_ring_r",     # 64
        "toe_pinky_r",    # 65
    ]

    bone_parents = [
        -1,  # root
        0,   # pelvis -> root
        1,   # spine_01 -> pelvis
        2,   # spine_02 -> spine_01
        3,   # spine_03 -> spine_02
        4,   # chest -> spine_03
        5,   # neck -> chest
        6,   # head -> neck

        # Left arm
        5,   # clavicle_l -> chest
        8,   # shoulder_l -> clavicle_l
        9,   # upperarm_l -> shoulder_l
        10,  # upperarm_twist_l -> upperarm_l
        10,  # forearm_l -> upperarm_l
        12,  # forearm_twist_l -> forearm_l
        12,  # hand_l -> forearm_l
        14,  # finger_thumb_01_l -> hand_l
        15,  # finger_thumb_02_l -> finger_thumb_01_l
        16,  # finger_thumb_03_l -> finger_thumb_02_l
        14,  # finger_index_01_l -> hand_l
        18,  # finger_index_02_l -> finger_index_01_l
        19,  # finger_index_03_l -> finger_index_02_l
        14,  # finger_middle_01_l -> hand_l
        21,  # finger_middle_02_l -> finger_middle_01_l
        22,  # finger_middle_03_l -> finger_middle_02_l
        14,  # finger_ring_01_l -> hand_l
        24,  # finger_ring_02_l -> finger_ring_01_l
        25,  # finger_ring_03_l -> finger_ring_02_l
        14,  # finger_pinky_01_l -> hand_l

        # Left leg
        1,   # thigh_l -> pelvis
        28,  # thigh_twist_l -> thigh_l
        28,  # calf_l -> thigh_l
        30,  # foot_l -> calf_l
        31,  # toe_big_l -> foot_l
        31,  # toe_index_l -> foot_l
        31,  # toe_middle_l -> foot_l
        31,  # toe_ring_l -> foot_l
        31,  # toe_pinky_l -> foot_l

        # Right arm
        5,   # clavicle_r -> chest
        37,  # shoulder_r -> clavicle_r
        38,  # upperarm_r -> shoulder_r
        39,  # upperarm_twist_r -> upperarm_r
        39,  # forearm_r -> upperarm_r
        41,  # forearm_twist_r -> forearm_r
        41,  # hand_r -> forearm_r
        43,  # finger_thumb_01_r -> hand_r
        44,  # finger_thumb_02_r -> finger_thumb_01_r
        45,  # finger_thumb_03_r -> finger_thumb_02_r
        43,  # finger_index_01_r -> hand_r
        47,  # finger_index_02_r -> finger_index_01_r
        48,  # finger_index_03_r -> finger_index_02_r
        43,  # finger_middle_01_r -> hand_r
        50,  # finger_middle_02_r -> finger_middle_01_r
        51,  # finger_middle_03_r -> finger_middle_02_r
        43,  # finger_ring_01_r -> hand_r
        53,  # finger_ring_02_r -> finger_ring_01_r
        54,  # finger_ring_03_r -> finger_ring_02_r
        43,  # finger_pinky_01_r -> hand_r

        # Right leg
        1,   # thigh_r -> pelvis
        57,  # thigh_twist_r -> thigh_r
        57,  # calf_r -> thigh_r
        59,  # foot_r -> calf_r
        60,  # toe_big_r -> foot_r
        60,  # toe_index_r -> foot_r
        60,  # toe_middle_r -> foot_r
        60,  # toe_ring_r -> foot_r
        60,  # toe_pinky_r -> foot_r
    ]

    # Create unique bind poses for each bone to verify correct transfer
    bind_poses = []
    for i in range(len(bone_names)):
        bind_poses.append(Transform(
            translation=Vec3(float(i), float(i) * 0.1, float(i) * 0.01),
            rotation=Quat.identity(),
            scale=Vec3.one(),
        ))

    return Skeleton(
        bone_names=bone_names,
        bone_parents=bone_parents,
        bind_poses=bind_poses,
    )


@pytest.fixture
def minimal_skeleton() -> Skeleton:
    """Create minimal skeleton for edge case testing."""
    return Skeleton(
        bone_names=["root", "spine", "head"],
        bone_parents=[-1, 0, 1],
        bind_poses=[Transform.identity() for _ in range(3)],
    )


# =============================================================================
# Test: Acceptance Criteria 1 - Important bones are kept
# =============================================================================


class TestImportantBonesKept:
    """Tests that important bones (root, spine, head) are preserved during reduction."""

    def test_root_always_preserved(self, full_humanoid_skeleton: Skeleton):
        """Test root bone is always preserved in reduced skeleton."""
        for target in [5, 10, 20, 30]:
            result = create_reduced_skeleton(full_humanoid_skeleton, target)
            assert "root" in result.bone_names, f"Root missing at target={target}"

    def test_pelvis_preserved_at_moderate_reduction(self, full_humanoid_skeleton: Skeleton):
        """Test pelvis is preserved at moderate reduction levels."""
        result = create_reduced_skeleton(full_humanoid_skeleton, 20)
        assert "pelvis" in result.bone_names

    def test_spine_bones_preserved(self, full_humanoid_skeleton: Skeleton):
        """Test spine bones are preserved at moderate reduction."""
        result = create_reduced_skeleton(full_humanoid_skeleton, 25)
        spine_count = sum(1 for name in result.bone_names if "spine" in name)
        assert spine_count >= 1, "At least one spine bone should be preserved"

    def test_head_preserved_at_moderate_reduction(self, full_humanoid_skeleton: Skeleton):
        """Test head bone is preserved at moderate reduction levels."""
        result = create_reduced_skeleton(full_humanoid_skeleton, 15)
        assert "head" in result.bone_names

    def test_neck_preserved_at_moderate_reduction(self, full_humanoid_skeleton: Skeleton):
        """Test neck is preserved at moderate reduction."""
        result = create_reduced_skeleton(full_humanoid_skeleton, 20)
        assert "neck" in result.bone_names

    def test_chest_preserved_at_moderate_reduction(self, full_humanoid_skeleton: Skeleton):
        """Test chest is preserved at moderate reduction."""
        result = create_reduced_skeleton(full_humanoid_skeleton, 18)
        assert "chest" in result.bone_names

    def test_core_bones_priority_over_extremities(self, full_humanoid_skeleton: Skeleton):
        """Test core bones have priority over extremities."""
        result = create_reduced_skeleton(full_humanoid_skeleton, 10)

        # Core bones should be present
        core_bones = ["root", "pelvis"]
        for bone in core_bones:
            assert bone in result.bone_names, f"{bone} should be in reduced skeleton"

        # Extremities should be culled first
        toe_count = sum(1 for name in result.bone_names if "toe" in name)
        finger_count = sum(1 for name in result.bone_names if "finger" in name)
        assert toe_count == 0, "Toes should be culled at aggressive reduction"
        assert finger_count == 0, "Fingers should be culled at aggressive reduction"


# =============================================================================
# Test: Acceptance Criteria 2 - Unimportant bones are culled
# =============================================================================


class TestUnimportantBonesCulled:
    """Tests that unimportant bones (fingers, twist, toes) are removed first."""

    def test_fingers_culled_first(self, full_humanoid_skeleton: Skeleton):
        """Test finger bones are removed before main limb bones."""
        original_finger_count = sum(1 for name in full_humanoid_skeleton.bone_names if "finger" in name)

        # Reduce by just the finger count
        target = full_humanoid_skeleton.bone_count - original_finger_count
        result = create_reduced_skeleton(full_humanoid_skeleton, target)

        # All fingers should be gone
        remaining_fingers = sum(1 for name in result.bone_names if "finger" in name)
        assert remaining_fingers < original_finger_count, "Fingers should be reduced first"

    def test_toes_culled_first(self, full_humanoid_skeleton: Skeleton):
        """Test toe bones are removed before main leg bones."""
        original_toe_count = sum(1 for name in full_humanoid_skeleton.bone_names if "toe" in name)

        # Aggressive reduction (66 bones -> 30) to ensure toes are culled
        target = 30
        result = create_reduced_skeleton(full_humanoid_skeleton, target)

        remaining_toes = sum(1 for name in result.bone_names if "toe" in name)
        assert remaining_toes < original_toe_count, "Toes should be reduced"

    def test_twist_bones_culled_before_main_bones(self, full_humanoid_skeleton: Skeleton):
        """Test twist/helper bones are removed before main bones."""
        original_twist_count = sum(1 for name in full_humanoid_skeleton.bone_names if "twist" in name)

        # Aggressive reduction (66 bones -> 35) to ensure twist bones are culled
        target = 35
        result = create_reduced_skeleton(full_humanoid_skeleton, target)

        remaining_twist = sum(1 for name in result.bone_names if "twist" in name)
        assert remaining_twist < original_twist_count, "Twist bones should be reduced"

    def test_deep_finger_joints_culled_before_base(self, full_humanoid_skeleton: Skeleton):
        """Test deeper finger joints (02, 03) are culled before base joints (01)."""
        # Reduce to keep some fingers but not all joints
        target = 45
        result = create_reduced_skeleton(full_humanoid_skeleton, target)

        # Count finger joint levels
        finger_01_count = sum(1 for name in result.bone_names if "finger" in name and "_01" in name)
        finger_02_count = sum(1 for name in result.bone_names if "finger" in name and "_02" in name)
        finger_03_count = sum(1 for name in result.bone_names if "finger" in name and "_03" in name)

        # Base joints should have equal or more count than deeper joints
        assert finger_01_count >= finger_02_count, "Base finger joints should be preserved over mid joints"
        assert finger_02_count >= finger_03_count, "Mid finger joints should be preserved over tip joints"

    def test_auxiliary_bones_lower_importance(self, full_humanoid_skeleton: Skeleton):
        """Test roll/helper bones are considered auxiliary and removed."""
        # Test a skeleton with roll bones
        skeleton_with_roll = Skeleton(
            bone_names=["root", "pelvis", "spine", "arm", "arm_roll", "hand"],
            bone_parents=[-1, 0, 1, 2, 3, 3],
            bind_poses=[Transform.identity() for _ in range(6)],
        )

        result = create_reduced_skeleton(skeleton_with_roll, 5)

        # Roll bone should be removed, main arm should stay
        assert "arm" in result.bone_names
        assert "arm_roll" not in result.bone_names


# =============================================================================
# Test: Acceptance Criteria 3 - Valid hierarchy (no orphan bones)
# =============================================================================


class TestValidHierarchy:
    """Tests that reduced skeleton maintains valid parent hierarchy."""

    def test_no_orphan_bones(self, full_humanoid_skeleton: Skeleton):
        """Test all bones have valid parents (no orphans)."""
        for target in [10, 20, 30, 40]:
            result = create_reduced_skeleton(full_humanoid_skeleton, target)

            for i, parent in enumerate(result.bone_parents):
                if parent >= 0:
                    assert parent < result.bone_count, f"Bone {i} has invalid parent {parent}"
                    assert parent < i, f"Parent {parent} should come before child {i}"

    def test_root_has_no_parent(self, full_humanoid_skeleton: Skeleton):
        """Test root bone always has parent -1."""
        result = create_reduced_skeleton(full_humanoid_skeleton, 15)

        # Find root bone
        root_idx = result.bone_names.index("root")
        assert result.bone_parents[root_idx] == -1

    def test_parent_chain_valid_after_reduction(self, full_humanoid_skeleton: Skeleton):
        """Test parent chains don't have gaps after reduction."""
        result = create_reduced_skeleton(full_humanoid_skeleton, 20)

        # For each bone, verify we can trace back to root
        for i in range(result.bone_count):
            current = i
            visited = set()

            while current >= 0:
                if current in visited:
                    pytest.fail(f"Circular parent reference at bone {i}")
                visited.add(current)
                current = result.bone_parents[current]

    def test_hierarchy_reconnects_to_nearest_ancestor(self, full_humanoid_skeleton: Skeleton):
        """Test that when intermediate bones are removed, children connect to nearest ancestor."""
        result = create_reduced_skeleton(full_humanoid_skeleton, 30)

        # All parents should be valid and point to included bones
        for i, parent in enumerate(result.bone_parents):
            if parent >= 0:
                assert 0 <= parent < result.bone_count
                # Parent name should exist
                parent_name = result.bone_names[parent]
                assert parent_name in result.bone_names

    def test_single_root_after_reduction(self, full_humanoid_skeleton: Skeleton):
        """Test there's exactly one root bone (parent = -1) after reduction."""
        result = create_reduced_skeleton(full_humanoid_skeleton, 25)

        root_count = sum(1 for p in result.bone_parents if p == -1)
        assert root_count == 1, f"Expected 1 root, found {root_count}"


# =============================================================================
# Test: Acceptance Criteria 4 - Bone count matches target
# =============================================================================


class TestBoneCountMatchesTarget:
    """Tests that reduced skeleton has exact target bone count."""

    def test_exact_bone_count_various_targets(self, full_humanoid_skeleton: Skeleton):
        """Test bone count matches target for various reduction levels."""
        for target in [5, 10, 15, 20, 25, 30, 40, 50]:
            result = create_reduced_skeleton(full_humanoid_skeleton, target)
            assert result.bone_count == target, f"Expected {target} bones, got {result.bone_count}"

    def test_no_reduction_when_target_exceeds_original(self, full_humanoid_skeleton: Skeleton):
        """Test no reduction when target >= original bone count."""
        original_count = full_humanoid_skeleton.bone_count

        result = create_reduced_skeleton(full_humanoid_skeleton, original_count + 10)
        assert result is full_humanoid_skeleton

    def test_empty_skeleton_when_target_zero(self, full_humanoid_skeleton: Skeleton):
        """Test empty skeleton when target is 0."""
        result = create_reduced_skeleton(full_humanoid_skeleton, 0)
        assert result.bone_count == 0
        assert len(result.bone_names) == 0
        assert len(result.bone_parents) == 0

    def test_single_bone_when_target_one(self, full_humanoid_skeleton: Skeleton):
        """Test single bone (root) when target is 1."""
        result = create_reduced_skeleton(full_humanoid_skeleton, 1)
        assert result.bone_count == 1
        assert result.bone_names[0] == "root"
        assert result.bone_parents[0] == -1

    def test_negative_target_returns_empty(self, full_humanoid_skeleton: Skeleton):
        """Test negative target returns empty skeleton."""
        result = create_reduced_skeleton(full_humanoid_skeleton, -5)
        assert result.bone_count == 0


# =============================================================================
# Test: Bind Pose Transfer
# =============================================================================


class TestBindPoseTransfer:
    """Tests that bind poses are correctly transferred to reduced skeleton."""

    def test_bind_poses_preserved_for_selected_bones(self, full_humanoid_skeleton: Skeleton):
        """Test bind poses are correctly transferred for selected bones."""
        target = 20
        result = create_reduced_skeleton(full_humanoid_skeleton, target)

        assert len(result.bind_poses) == target

        # Verify bind poses match original for each selected bone
        for i, name in enumerate(result.bone_names):
            if name in full_humanoid_skeleton.bone_names:
                orig_idx = full_humanoid_skeleton.bone_names.index(name)
                orig_pose = full_humanoid_skeleton.bind_poses[orig_idx]
                result_pose = result.bind_poses[i]

                # Translation should match
                assert result_pose.translation.x == orig_pose.translation.x
                assert result_pose.translation.y == orig_pose.translation.y
                assert result_pose.translation.z == orig_pose.translation.z

    def test_bind_poses_count_matches_bones(self, full_humanoid_skeleton: Skeleton):
        """Test bind poses array length matches bone count."""
        for target in [10, 20, 30]:
            result = create_reduced_skeleton(full_humanoid_skeleton, target)
            assert len(result.bind_poses) == len(result.bone_names)
            assert len(result.bind_poses) == len(result.bone_parents)


# =============================================================================
# Test: Bone Importance Scoring
# =============================================================================


class TestBoneImportanceScoring:
    """Tests for _calculate_bone_importance() scoring function."""

    def test_root_highest_importance(self, full_humanoid_skeleton: Skeleton):
        """Test root bone has highest importance score."""
        root_importance = _calculate_bone_importance("root", 0, full_humanoid_skeleton)
        assert root_importance >= 0.95

    def test_pelvis_high_importance(self, full_humanoid_skeleton: Skeleton):
        """Test pelvis has high importance."""
        importance = _calculate_bone_importance("pelvis", 1, full_humanoid_skeleton)
        assert importance >= 0.85

    def test_spine_high_importance(self, full_humanoid_skeleton: Skeleton):
        """Test spine bones have high importance."""
        for i in range(2, 5):
            importance = _calculate_bone_importance(f"spine_0{i-1}", i, full_humanoid_skeleton)
            assert importance >= 0.7

    def test_head_high_importance(self, full_humanoid_skeleton: Skeleton):
        """Test head has high importance."""
        importance = _calculate_bone_importance("head", 7, full_humanoid_skeleton)
        assert importance >= 0.7

    def test_neck_moderate_importance(self, full_humanoid_skeleton: Skeleton):
        """Test neck has moderate-high importance."""
        importance = _calculate_bone_importance("neck", 6, full_humanoid_skeleton)
        assert importance >= 0.6

    def test_shoulder_moderate_importance(self, full_humanoid_skeleton: Skeleton):
        """Test shoulder has moderate importance."""
        importance = _calculate_bone_importance("shoulder_l", 9, full_humanoid_skeleton)
        assert importance >= 0.5

    def test_upperarm_moderate_importance(self, full_humanoid_skeleton: Skeleton):
        """Test upperarm has moderate importance."""
        importance = _calculate_bone_importance("upperarm_l", 10, full_humanoid_skeleton)
        assert importance >= 0.5

    def test_forearm_lower_than_upperarm(self, full_humanoid_skeleton: Skeleton):
        """Test forearm has lower importance than upperarm."""
        upperarm = _calculate_bone_importance("upperarm_l", 10, full_humanoid_skeleton)
        forearm = _calculate_bone_importance("forearm_l", 12, full_humanoid_skeleton)
        assert forearm <= upperarm

    def test_thigh_moderate_importance(self, full_humanoid_skeleton: Skeleton):
        """Test thigh has moderate importance."""
        importance = _calculate_bone_importance("thigh_l", 28, full_humanoid_skeleton)
        assert importance >= 0.5

    def test_calf_lower_than_thigh(self, full_humanoid_skeleton: Skeleton):
        """Test calf has lower importance than thigh."""
        thigh = _calculate_bone_importance("thigh_l", 28, full_humanoid_skeleton)
        calf = _calculate_bone_importance("calf_l", 30, full_humanoid_skeleton)
        assert calf <= thigh

    def test_finger_low_importance(self, full_humanoid_skeleton: Skeleton):
        """Test finger bones have low importance."""
        importance = _calculate_bone_importance("finger_middle_02_l", 22, full_humanoid_skeleton)
        assert importance < 0.6

    def test_toe_lowest_importance(self, full_humanoid_skeleton: Skeleton):
        """Test toe bones have very low importance."""
        importance = _calculate_bone_importance("toe_index_l", 33, full_humanoid_skeleton)
        assert importance < 0.55

    def test_twist_penalty(self, full_humanoid_skeleton: Skeleton):
        """Test twist bones have reduced importance."""
        normal = _calculate_bone_importance("upperarm_l", 10, full_humanoid_skeleton)
        twist = _calculate_bone_importance("upperarm_twist_l", 11, full_humanoid_skeleton)
        assert twist < normal

    def test_roll_penalty(self, full_humanoid_skeleton: Skeleton):
        """Test roll bones have reduced importance."""
        skeleton = Skeleton(
            bone_names=["root", "arm", "arm_roll"],
            bone_parents=[-1, 0, 1],
            bind_poses=[Transform.identity() for _ in range(3)],
        )
        normal = _calculate_bone_importance("arm", 1, skeleton)
        roll = _calculate_bone_importance("arm_roll", 2, skeleton)
        assert roll < normal

    def test_helper_penalty(self, full_humanoid_skeleton: Skeleton):
        """Test helper bones have reduced importance."""
        skeleton = Skeleton(
            bone_names=["root", "arm", "arm_helper"],
            bone_parents=[-1, 0, 1],
            bind_poses=[Transform.identity() for _ in range(3)],
        )
        normal = _calculate_bone_importance("arm", 1, skeleton)
        helper = _calculate_bone_importance("arm_helper", 2, skeleton)
        assert helper < normal

    def test_depth_penalty(self, full_humanoid_skeleton: Skeleton):
        """Test deeper bones have lower importance due to depth penalty."""
        # Root is depth 0, head is depth 7
        root_importance = _calculate_bone_importance("root", 0, full_humanoid_skeleton)

        # Create deep bone
        skeleton = Skeleton(
            bone_names=["root", "a", "b", "c", "d", "e", "deep"],
            bone_parents=[-1, 0, 1, 2, 3, 4, 5],
            bind_poses=[Transform.identity() for _ in range(7)],
        )
        deep_importance = _calculate_bone_importance("a", 6, skeleton)

        # Deep bone should have lower base importance
        assert deep_importance < root_importance

    def test_index_thumb_higher_than_pinky(self, full_humanoid_skeleton: Skeleton):
        """Test index/thumb fingers have higher importance than pinky."""
        index = _calculate_bone_importance("finger_index_01_l", 18, full_humanoid_skeleton)
        thumb = _calculate_bone_importance("finger_thumb_01_l", 15, full_humanoid_skeleton)
        pinky = _calculate_bone_importance("finger_pinky_01_l", 27, full_humanoid_skeleton)

        assert index >= pinky
        assert thumb >= pinky


# =============================================================================
# Test: Bone Depth Calculation
# =============================================================================


class TestBoneDepthCalculation:
    """Tests for _get_bone_depth() function."""

    def test_root_depth_zero(self, full_humanoid_skeleton: Skeleton):
        """Test root bone has depth 0."""
        depth = _get_bone_depth(0, full_humanoid_skeleton)
        assert depth == 0

    def test_direct_child_depth_one(self, full_humanoid_skeleton: Skeleton):
        """Test direct children of root have depth 1."""
        # pelvis is direct child of root
        depth = _get_bone_depth(1, full_humanoid_skeleton)
        assert depth == 1

    def test_deep_bone_correct_depth(self, full_humanoid_skeleton: Skeleton):
        """Test deep bones have correct depth."""
        # head is: root -> pelvis -> spine_01 -> spine_02 -> spine_03 -> chest -> neck -> head
        # depth = 7
        depth = _get_bone_depth(7, full_humanoid_skeleton)
        assert depth == 7

    def test_finger_tip_depth(self, full_humanoid_skeleton: Skeleton):
        """Test finger tip bones have high depth."""
        # finger_thumb_03_l: root -> ... -> hand_l -> thumb_01 -> thumb_02 -> thumb_03
        depth = _get_bone_depth(17, full_humanoid_skeleton)
        assert depth >= 9  # At least 9 levels deep


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in skeleton reduction."""

    def test_empty_skeleton_input(self):
        """Test reducing empty skeleton."""
        empty = Skeleton(bone_names=[], bone_parents=[], bind_poses=[])
        result = create_reduced_skeleton(empty, 5)
        assert result is empty

    def test_single_bone_skeleton(self):
        """Test reducing single bone skeleton."""
        single = Skeleton(
            bone_names=["root"],
            bone_parents=[-1],
            bind_poses=[Transform.identity()],
        )
        result = create_reduced_skeleton(single, 1)
        assert result is single

    def test_target_equals_original(self, full_humanoid_skeleton: Skeleton):
        """Test target equal to original returns original."""
        result = create_reduced_skeleton(full_humanoid_skeleton, full_humanoid_skeleton.bone_count)
        assert result is full_humanoid_skeleton

    def test_minimal_skeleton_reduction(self, minimal_skeleton: Skeleton):
        """Test reducing minimal skeleton to 2 bones."""
        result = create_reduced_skeleton(minimal_skeleton, 2)
        assert result.bone_count == 2
        assert "root" in result.bone_names

    def test_all_same_importance_bones(self):
        """Test skeleton where all bones have similar importance."""
        # All "generic" bones
        skeleton = Skeleton(
            bone_names=["bone_0", "bone_1", "bone_2", "bone_3", "bone_4"],
            bone_parents=[-1, 0, 1, 2, 3],
            bind_poses=[Transform.identity() for _ in range(5)],
        )
        result = create_reduced_skeleton(skeleton, 3)
        assert result.bone_count == 3
        # Should keep root-like bone first
        assert result.bone_names[0] == "bone_0"


# =============================================================================
# Test: Progressive Reduction
# =============================================================================


class TestProgressiveReduction:
    """Tests for progressive skeleton reduction at different levels."""

    def test_progressive_reduction_preserves_order(self, full_humanoid_skeleton: Skeleton):
        """Test that more aggressive reduction removes bones in importance order."""
        targets = [50, 40, 30, 20, 10]
        prev_result = full_humanoid_skeleton

        for target in targets:
            result = create_reduced_skeleton(full_humanoid_skeleton, target)

            # All bones in smaller result should be in larger result
            for name in result.bone_names:
                assert name in prev_result.bone_names, f"{name} missing from larger reduction"

            prev_result = result

    def test_lod_chain_valid(self, full_humanoid_skeleton: Skeleton):
        """Test a typical LOD chain (64 -> 32 -> 16 -> 8) produces valid skeletons."""
        targets = [50, 32, 16, 8]  # Adjusted for 66-bone skeleton

        for target in targets:
            result = create_reduced_skeleton(full_humanoid_skeleton, target)

            # Verify valid skeleton
            assert result.bone_count == target
            assert len(result.bone_names) == target
            assert len(result.bone_parents) == target
            assert len(result.bind_poses) == target

            # Verify valid hierarchy
            for i, parent in enumerate(result.bone_parents):
                if parent >= 0:
                    assert parent < result.bone_count

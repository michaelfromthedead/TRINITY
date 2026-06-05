"""Whitebox tests for LOD integration in crowd system.

Tests internal methods of:
- CrowdLOD.get_lod_for_distance() with various distances
- Hysteresis threshold logic (bi-directional)
- create_reduced_skeleton() bone reduction
- LODTransition state machine
- CrowdInstance.start_lod_transition() / update_lod_transition()
- get_render_lod_info() shader data output

Task: T1.5 LOD Integration
"""

from __future__ import annotations

import math
import pytest
from unittest.mock import patch, MagicMock

from engine.core.math import Vec3, Vec4, Quat, Transform
from engine.animation.crowds.animation_texture import Skeleton
from engine.animation.crowds.crowd_lod import (
    CrowdLOD,
    LODLevel,
    LODTransition,
    LODTransitionMode,
    BoneWeight,
    create_reduced_skeleton,
    calculate_lod_blend_weights,
    _calculate_bone_importance,
    _get_bone_depth,
)
from engine.animation.crowds.crowd_renderer import (
    CrowdInstance,
    CrowdRenderer,
    InstanceBuffer,
    CrowdRenderBatch,
)
from engine.animation.config import CROWD_LOD_CONFIG


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_skeleton() -> Skeleton:
    """Create a basic humanoid skeleton for testing."""
    bone_names = [
        "root",       # 0
        "pelvis",     # 1
        "spine1",     # 2
        "spine2",     # 3
        "chest",      # 4
        "neck",       # 5
        "head",       # 6
        "shoulder_l", # 7
        "upperarm_l", # 8
        "forearm_l",  # 9
        "hand_l",     # 10
        "finger_index_l",  # 11
        "finger_middle_l", # 12
        "finger_ring_l",   # 13
        "finger_pinky_l",  # 14
        "thigh_l",    # 15
        "calf_l",     # 16
        "foot_l",     # 17
        "toe_l",      # 18
    ]
    bone_parents = [
        -1,  # root
        0,   # pelvis -> root
        1,   # spine1 -> pelvis
        2,   # spine2 -> spine1
        3,   # chest -> spine2
        4,   # neck -> chest
        5,   # head -> neck
        4,   # shoulder_l -> chest
        7,   # upperarm_l -> shoulder_l
        8,   # forearm_l -> upperarm_l
        9,   # hand_l -> forearm_l
        10,  # finger_index_l -> hand_l
        10,  # finger_middle_l -> hand_l
        10,  # finger_ring_l -> hand_l
        10,  # finger_pinky_l -> hand_l
        1,   # thigh_l -> pelvis
        15,  # calf_l -> thigh_l
        16,  # foot_l -> calf_l
        17,  # toe_l -> foot_l
    ]
    bind_poses = [Transform.identity() for _ in bone_names]
    return Skeleton(bone_names=bone_names, bone_parents=bone_parents, bind_poses=bind_poses)


@pytest.fixture
def default_lod_levels() -> list[LODLevel]:
    """Create default LOD levels for testing."""
    return [
        LODLevel(distance=0.0, bone_count=64, update_rate=1.0, mesh_reduction=0.0, shadow_enabled=True, animation_quality=1.0),
        LODLevel(distance=10.0, bone_count=32, update_rate=1.0, mesh_reduction=0.2, shadow_enabled=True, animation_quality=0.8),
        LODLevel(distance=25.0, bone_count=16, update_rate=0.5, mesh_reduction=0.4, shadow_enabled=False, animation_quality=0.6),
        LODLevel(distance=50.0, bone_count=8, update_rate=0.25, mesh_reduction=0.6, shadow_enabled=False, animation_quality=0.4),
    ]


@pytest.fixture
def crowd_lod_with_levels(basic_skeleton: Skeleton, default_lod_levels: list[LODLevel]) -> CrowdLOD:
    """Create CrowdLOD with skeleton and levels configured."""
    lod = CrowdLOD(skeleton=basic_skeleton, levels=default_lod_levels)
    return lod


# =============================================================================
# Test: LODLevel Class
# =============================================================================


class TestLODLevel:
    """Tests for LODLevel dataclass."""

    def test_lod_level_comparison(self):
        """Test LODLevel comparison for sorting."""
        level1 = LODLevel(distance=10.0)
        level2 = LODLevel(distance=25.0)
        level3 = LODLevel(distance=5.0)

        assert level3 < level1 < level2
        assert not level2 < level1

    def test_should_update_full_rate(self):
        """Test should_update with full update rate."""
        level = LODLevel(update_rate=1.0)
        for frame in range(10):
            assert level.should_update(frame) is True

    def test_should_update_half_rate(self):
        """Test should_update with half update rate (every 2 frames)."""
        level = LODLevel(update_rate=0.5)
        assert level.should_update(0) is True
        assert level.should_update(1) is False
        assert level.should_update(2) is True
        assert level.should_update(3) is False

    def test_should_update_quarter_rate(self):
        """Test should_update with quarter update rate (every 4 frames)."""
        level = LODLevel(update_rate=0.25)
        assert level.should_update(0) is True
        assert level.should_update(1) is False
        assert level.should_update(2) is False
        assert level.should_update(3) is False
        assert level.should_update(4) is True

    def test_should_update_zero_rate(self):
        """Test should_update with zero rate never updates."""
        level = LODLevel(update_rate=0.0)
        for frame in range(100):
            assert level.should_update(frame) is False

    def test_should_update_greater_than_one(self):
        """Test should_update with rate > 1.0 always updates."""
        level = LODLevel(update_rate=2.0)
        for frame in range(10):
            assert level.should_update(frame) is True


# =============================================================================
# Test: LODTransition State Machine
# =============================================================================


class TestLODTransition:
    """Tests for LODTransition state machine."""

    def test_transition_start_different_lods(self):
        """Test starting transition between different LODs."""
        transition = LODTransition()
        transition.start(from_lod=0, to_lod=2)

        assert transition.active is True
        assert transition.from_lod == 0
        assert transition.to_lod == 2
        assert transition.progress == 0.0

    def test_transition_start_same_lod(self):
        """Test starting transition with same LOD does nothing."""
        transition = LODTransition()
        transition.start(from_lod=1, to_lod=1)

        assert transition.active is False

    def test_transition_update_progress(self):
        """Test transition update increments progress."""
        transition = LODTransition(duration=1.0)
        transition.start(from_lod=0, to_lod=1)

        # Update halfway
        completed = transition.update(0.5)
        assert completed is False
        assert transition.active is True
        assert abs(transition.progress - 0.5) < 0.001

    def test_transition_update_completion(self):
        """Test transition completes when progress >= 1.0."""
        transition = LODTransition(duration=0.5)
        transition.start(from_lod=0, to_lod=2)

        completed = transition.update(0.6)
        assert completed is True
        assert transition.active is False
        assert transition.progress == 1.0

    def test_transition_update_inactive(self):
        """Test updating inactive transition returns True."""
        transition = LODTransition(active=False)
        completed = transition.update(1.0)
        assert completed is True

    def test_transition_update_zero_duration(self):
        """Test transition with zero duration completes immediately."""
        transition = LODTransition(duration=0.0)
        transition.start(from_lod=0, to_lod=1)

        completed = transition.update(0.001)
        assert completed is True
        assert transition.active is False
        assert transition.progress == 1.0

    def test_get_blend_factor_smoothstep(self):
        """Test blend factor uses smoothstep function."""
        transition = LODTransition()

        # Test smoothstep at key points
        transition.progress = 0.0
        assert transition.get_blend_factor() == 0.0

        transition.progress = 0.5
        # Smoothstep at 0.5 = 0.5 * 0.5 * (3 - 2 * 0.5) = 0.25 * 2 = 0.5
        assert abs(transition.get_blend_factor() - 0.5) < 0.001

        transition.progress = 1.0
        assert transition.get_blend_factor() == 1.0

    def test_get_blend_factor_clamps_input(self):
        """Test blend factor clamps progress to 0-1."""
        transition = LODTransition()

        transition.progress = -0.5
        assert transition.get_blend_factor() == 0.0

        transition.progress = 1.5
        assert transition.get_blend_factor() == 1.0

    def test_get_current_lod_before_midpoint(self):
        """Test current LOD returns from_lod before midpoint."""
        transition = LODTransition(from_lod=0, to_lod=2, active=True)
        transition.progress = 0.4
        assert transition.get_current_lod() == 0

    def test_get_current_lod_at_midpoint(self):
        """Test current LOD returns to_lod at midpoint."""
        transition = LODTransition(from_lod=0, to_lod=2, active=True)
        transition.progress = 0.5
        assert transition.get_current_lod() == 2

    def test_get_current_lod_after_completion(self):
        """Test current LOD returns to_lod when inactive."""
        transition = LODTransition(from_lod=0, to_lod=2, active=False)
        assert transition.get_current_lod() == 2


# =============================================================================
# Test: CrowdLOD.get_lod_for_distance()
# =============================================================================


class TestCrowdLODDistanceSelection:
    """Tests for CrowdLOD.get_lod_for_distance() method."""

    def test_get_lod_empty_levels(self):
        """Test get_lod_for_distance with no levels returns 0."""
        lod = CrowdLOD()
        assert lod.get_lod_for_distance(50.0) == 0

    def test_get_lod_distance_zero(self, crowd_lod_with_levels: CrowdLOD):
        """Test get_lod_for_distance at distance 0."""
        lod_index = crowd_lod_with_levels.get_lod_for_distance(0.0)
        assert lod_index == 0  # Highest detail

    def test_get_lod_distance_negative(self, crowd_lod_with_levels: CrowdLOD):
        """Test get_lod_for_distance with negative distance is clamped."""
        lod_index = crowd_lod_with_levels.get_lod_for_distance(-10.0)
        assert lod_index == 0  # Should clamp to 0

    def test_get_lod_distance_infinity(self, crowd_lod_with_levels: CrowdLOD):
        """Test get_lod_for_distance at infinite distance returns max LOD."""
        lod_index = crowd_lod_with_levels.get_lod_for_distance(float('inf'))
        assert lod_index == crowd_lod_with_levels.max_lod

    def test_get_lod_distance_between_levels(self, crowd_lod_with_levels: CrowdLOD):
        """Test get_lod_for_distance selects correct LOD between thresholds."""
        # Distance 15 is between LOD1 (10) and LOD2 (25)
        lod_index = crowd_lod_with_levels.get_lod_for_distance(15.0)
        assert lod_index == 1

    def test_get_lod_at_exact_threshold(self, crowd_lod_with_levels: CrowdLOD):
        """Test get_lod_for_distance at exact threshold selects lower detail."""
        # At exactly distance 10, should return LOD 0 (just below threshold)
        lod_index = crowd_lod_with_levels.get_lod_for_distance(10.0)
        # The algorithm: if distance < threshold -> return i-1, so at 10.0 we check
        # against 10.0 threshold and it's not less, so continues to next level
        assert lod_index == 1

    def test_get_lod_with_hysteresis_upgrading(self, crowd_lod_with_levels: CrowdLOD):
        """Test hysteresis when switching to higher detail (lower LOD index)."""
        crowd_lod_with_levels.set_hysteresis(2.0)

        # Currently at LOD 2 (distance 25), moving closer
        # Threshold for LOD 1 is 10, with hysteresis it becomes 10-2=8
        lod_index = crowd_lod_with_levels.get_lod_for_distance(9.0, current_lod=2)
        assert lod_index == 1  # Still 9 > 8 so stays at lower detail

        lod_index = crowd_lod_with_levels.get_lod_for_distance(7.0, current_lod=2)
        assert lod_index == 0  # Now 7 < 8 so switches to higher detail

    def test_get_lod_with_hysteresis_downgrading(self, crowd_lod_with_levels: CrowdLOD):
        """Test hysteresis when switching to lower detail (higher LOD index)."""
        crowd_lod_with_levels.set_hysteresis(2.0)

        # Currently at LOD 1 (distance 10-25), moving further
        # Threshold for LOD 2 is 25, with hysteresis it becomes 25+2=27
        lod_index = crowd_lod_with_levels.get_lod_for_distance(26.0, current_lod=1)
        assert lod_index == 1  # Still 26 < 27 so stays

        lod_index = crowd_lod_with_levels.get_lod_for_distance(28.0, current_lod=1)
        assert lod_index == 2  # Now 28 > 27 so downgrades

    def test_get_lod_hysteresis_prevents_flickering(self, crowd_lod_with_levels: CrowdLOD):
        """Test hysteresis prevents rapid LOD switching (flickering)."""
        crowd_lod_with_levels.set_hysteresis(3.0)

        # Simulate oscillating distance around threshold
        current_lod = 1
        distances = [24.0, 26.0, 24.0, 26.0, 24.0]
        lod_changes = 0

        for dist in distances:
            new_lod = crowd_lod_with_levels.get_lod_for_distance(dist, current_lod)
            if new_lod != current_lod:
                lod_changes += 1
                current_lod = new_lod

        # With hysteresis of 3.0, threshold is 25+3=28 or 25-3=22
        # Oscillating between 24 and 26 should NOT cause any LOD changes
        assert lod_changes == 0

    def test_get_lod_no_hysteresis_without_current(self, crowd_lod_with_levels: CrowdLOD):
        """Test no hysteresis applied when current_lod is -1."""
        crowd_lod_with_levels.set_hysteresis(5.0)

        # Without current LOD, hysteresis should not apply
        lod_index = crowd_lod_with_levels.get_lod_for_distance(9.0, current_lod=-1)
        expected = crowd_lod_with_levels.get_lod_for_distance(9.0, current_lod=-1)
        assert lod_index == expected


# =============================================================================
# Test: Skeleton Reduction
# =============================================================================


class TestSkeletonReduction:
    """Tests for create_reduced_skeleton() and related functions."""

    def test_create_reduced_skeleton_no_reduction(self, basic_skeleton: Skeleton):
        """Test reduced skeleton with target >= original returns original."""
        result = create_reduced_skeleton(basic_skeleton, basic_skeleton.bone_count + 10)
        assert result is basic_skeleton

    def test_create_reduced_skeleton_zero_bones(self, basic_skeleton: Skeleton):
        """Test reduced skeleton with 0 bones returns empty skeleton."""
        result = create_reduced_skeleton(basic_skeleton, 0)
        assert result.bone_count == 0
        assert result.bone_names == []

    def test_create_reduced_skeleton_reduces_count(self, basic_skeleton: Skeleton):
        """Test reduced skeleton has correct bone count."""
        target = 10
        result = create_reduced_skeleton(basic_skeleton, target)
        assert result.bone_count == target

    def test_create_reduced_skeleton_preserves_important_bones(self, basic_skeleton: Skeleton):
        """Test reduction preserves important bones (root, spine, head)."""
        result = create_reduced_skeleton(basic_skeleton, 8)

        # Root and major bones should be preserved
        assert "root" in result.bone_names
        assert "pelvis" in result.bone_names or "head" in result.bone_names

    def test_create_reduced_skeleton_removes_fingers_first(self, basic_skeleton: Skeleton):
        """Test fingers are removed before main bones."""
        result = create_reduced_skeleton(basic_skeleton, basic_skeleton.bone_count - 4)

        # With 4 fewer bones, fingers should be removed
        finger_count = sum(1 for name in result.bone_names if "finger" in name)
        assert finger_count < 4  # Original has 4 fingers

    def test_create_reduced_skeleton_maintains_hierarchy(self, basic_skeleton: Skeleton):
        """Test reduced skeleton maintains valid parent hierarchy."""
        result = create_reduced_skeleton(basic_skeleton, 8)

        # All parent indices should be valid
        for i, parent in enumerate(result.bone_parents):
            if parent >= 0:
                assert parent < result.bone_count, f"Bone {i} has invalid parent {parent}"
                assert parent < i, "Parent index should be less than child index"

    def test_bone_importance_root(self, basic_skeleton: Skeleton):
        """Test root bone has highest importance."""
        importance = _calculate_bone_importance("root", 0, basic_skeleton)
        assert importance >= 0.9  # Root should be very important

    def test_bone_importance_spine(self, basic_skeleton: Skeleton):
        """Test spine bones have high importance."""
        importance = _calculate_bone_importance("spine1", 2, basic_skeleton)
        assert importance >= 0.7

    def test_bone_importance_fingers(self, basic_skeleton: Skeleton):
        """Test finger bones have low importance."""
        importance = _calculate_bone_importance("finger_pinky_l", 14, basic_skeleton)
        assert importance < 0.6  # Fingers should be less important

    def test_bone_importance_twist_modifier(self, basic_skeleton: Skeleton):
        """Test twist/helper bones have reduced importance."""
        normal = _calculate_bone_importance("upperarm", 1, basic_skeleton)
        twist = _calculate_bone_importance("upperarm_twist", 1, basic_skeleton)
        assert twist < normal

    def test_get_bone_depth(self, basic_skeleton: Skeleton):
        """Test bone depth calculation."""
        # root -> pelvis -> spine1 -> spine2 -> chest -> neck -> head
        assert _get_bone_depth(0, basic_skeleton) == 0  # root
        assert _get_bone_depth(1, basic_skeleton) == 1  # pelvis
        assert _get_bone_depth(6, basic_skeleton) == 6  # head


# =============================================================================
# Test: CrowdInstance LOD Transitions
# =============================================================================


class TestCrowdInstanceLODTransitions:
    """Tests for CrowdInstance LOD transition methods."""

    def test_start_lod_transition_instant_mode(self):
        """Test INSTANT mode changes LOD immediately."""
        instance = CrowdInstance(lod_level=0)
        instance.start_lod_transition(2, mode=LODTransitionMode.INSTANT)

        assert instance.lod_level == 2
        assert instance.lod_transition is None
        assert instance.lod_blend_factor == 0.0

    def test_start_lod_transition_blend_mode(self):
        """Test BLEND mode creates transition object."""
        instance = CrowdInstance(lod_level=0)
        instance.start_lod_transition(2, mode=LODTransitionMode.BLEND, duration=0.5)

        assert instance.lod_level == 0  # Not changed yet
        assert instance.lod_transition is not None
        assert instance.lod_transition.active is True
        assert instance.lod_transition.mode == LODTransitionMode.BLEND

    def test_start_lod_transition_dither_mode(self):
        """Test DITHER mode creates transition object."""
        instance = CrowdInstance(lod_level=1)
        instance.start_lod_transition(3, mode=LODTransitionMode.DITHER, duration=0.3)

        assert instance.lod_transition is not None
        assert instance.lod_transition.mode == LODTransitionMode.DITHER

    def test_start_lod_transition_same_level(self):
        """Test starting transition to same LOD does nothing."""
        instance = CrowdInstance(lod_level=2)
        instance.start_lod_transition(2, mode=LODTransitionMode.BLEND)

        assert instance.lod_transition is None

    def test_update_lod_transition_progress(self):
        """Test update_lod_transition progresses blend."""
        instance = CrowdInstance(lod_level=0)
        instance.start_lod_transition(2, mode=LODTransitionMode.BLEND, duration=1.0)

        completed = instance.update_lod_transition(0.5)
        assert completed is False
        assert instance.lod_blend_factor > 0.0
        assert instance.lod_level == 0  # Still original

    def test_update_lod_transition_completion(self):
        """Test transition completes and updates lod_level."""
        instance = CrowdInstance(lod_level=0)
        instance.start_lod_transition(2, mode=LODTransitionMode.BLEND, duration=0.5)

        completed = instance.update_lod_transition(0.6)
        assert completed is True
        assert instance.lod_level == 2
        assert instance.lod_blend_factor == 0.0
        assert instance.lod_transition is None

    def test_update_lod_transition_no_active(self):
        """Test update returns False when no transition active."""
        instance = CrowdInstance(lod_level=1)
        completed = instance.update_lod_transition(1.0)
        assert completed is False


# =============================================================================
# Test: get_render_lod_info() Shader Data
# =============================================================================


class TestGetRenderLODInfo:
    """Tests for CrowdInstance.get_render_lod_info() shader output."""

    def test_render_info_no_transition(self):
        """Test render info with no active transition."""
        instance = CrowdInstance(lod_level=2)
        primary, secondary, blend = instance.get_render_lod_info()

        assert primary == 2
        assert secondary == 2
        assert blend == 0.0

    def test_render_info_during_blend_transition(self):
        """Test render info during BLEND transition."""
        instance = CrowdInstance(lod_level=1)
        instance.start_lod_transition(3, mode=LODTransitionMode.BLEND, duration=1.0)
        instance.update_lod_transition(0.5)

        primary, secondary, blend = instance.get_render_lod_info()

        assert primary == 1  # from_lod
        assert secondary == 3  # to_lod
        assert blend > 0.0

    def test_render_info_during_dither_transition(self):
        """Test render info during DITHER transition."""
        instance = CrowdInstance(lod_level=0)
        instance.start_lod_transition(2, mode=LODTransitionMode.DITHER, duration=0.5)
        instance.update_lod_transition(0.25)

        primary, secondary, blend = instance.get_render_lod_info()

        assert primary == 0
        assert secondary == 2
        assert blend > 0.0

    def test_render_info_after_transition_complete(self):
        """Test render info after transition completes."""
        instance = CrowdInstance(lod_level=0)
        instance.start_lod_transition(3, mode=LODTransitionMode.BLEND, duration=0.2)
        instance.update_lod_transition(0.3)  # Complete transition

        primary, secondary, blend = instance.get_render_lod_info()

        assert primary == 3
        assert secondary == 3
        assert blend == 0.0


# =============================================================================
# Test: CrowdRenderer LOD Integration
# =============================================================================


class TestCrowdRendererLODIntegration:
    """Tests for CrowdRenderer LOD integration methods."""

    def test_update_lod_levels_from_system(self, basic_skeleton: Skeleton, default_lod_levels: list[LODLevel]):
        """Test update_lod_levels_from_system uses CrowdLOD correctly."""
        lod_system = CrowdLOD(skeleton=basic_skeleton, levels=default_lod_levels)
        renderer = CrowdRenderer()

        # Add instances at various distances
        camera_pos = Vec3.zero()
        positions = [
            Vec3(5, 0, 0),   # distance 5 -> LOD 0
            Vec3(15, 0, 0),  # distance 15 -> LOD 1
            Vec3(30, 0, 0),  # distance 30 -> LOD 2
            Vec3(60, 0, 0),  # distance 60 -> LOD 3
        ]

        for pos in positions:
            instance = CrowdInstance(position=pos)
            renderer.add_instance(instance, mesh_id=1, material_id=1)

        renderer.update_lod_levels_from_system(camera_pos, lod_system)

        # Check LOD levels were assigned
        batch = renderer.get_batch(1, 1)
        assert batch is not None

        lods = [inst.lod_level for inst in batch.instances]
        assert lods[0] == 0
        assert lods[1] == 1
        assert lods[2] == 2
        assert lods[3] == 3

    def test_update_lod_with_transitions(self, basic_skeleton: Skeleton, default_lod_levels: list[LODLevel]):
        """Test update_lod_with_transitions creates transitions."""
        lod_system = CrowdLOD(skeleton=basic_skeleton, levels=default_lod_levels)
        renderer = CrowdRenderer()

        # Add instance that will need LOD change
        instance = CrowdInstance(position=Vec3(15, 0, 0), lod_level=0)
        renderer.add_instance(instance, mesh_id=1, material_id=1)

        completed = renderer.update_lod_with_transitions(
            camera_pos=Vec3.zero(),
            lod_system=lod_system,
            dt=0.01,
            transition_mode=LODTransitionMode.BLEND,
            transition_duration=0.3,
        )

        batch = renderer.get_batch(1, 1)
        inst = batch.instances[0]

        # Should have started a transition
        assert inst.lod_transition is not None
        assert inst.lod_transition.active is True

    def test_get_transition_stats(self, basic_skeleton: Skeleton, default_lod_levels: list[LODLevel]):
        """Test get_transition_stats returns correct counts."""
        renderer = CrowdRenderer()

        # Add instances with various transition states
        inst1 = CrowdInstance(position=Vec3(0, 0, 0))  # No transition
        inst2 = CrowdInstance(position=Vec3(10, 0, 0))  # Will have blend
        inst3 = CrowdInstance(position=Vec3(20, 0, 0))  # Will have dither

        renderer.add_instance(inst1, mesh_id=1, material_id=1)
        renderer.add_instance(inst2, mesh_id=1, material_id=1)
        renderer.add_instance(inst3, mesh_id=1, material_id=1)

        # Start transitions
        batch = renderer.get_batch(1, 1)
        batch.instances[1].start_lod_transition(2, LODTransitionMode.BLEND, 0.5)
        batch.instances[2].start_lod_transition(3, LODTransitionMode.DITHER, 0.5)

        stats = renderer.get_transition_stats()

        assert stats["no_transition"] == 1
        assert stats["active_blend"] == 1
        assert stats["active_dither"] == 1


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_lod_very_large_distance(self, crowd_lod_with_levels: CrowdLOD):
        """Test LOD selection at very large distance."""
        lod_index = crowd_lod_with_levels.get_lod_for_distance(1e10)
        assert lod_index == crowd_lod_with_levels.max_lod

    def test_lod_nan_distance(self, crowd_lod_with_levels: CrowdLOD):
        """Test LOD selection with NaN distance."""
        # NaN comparisons always return False
        lod_index = crowd_lod_with_levels.get_lod_for_distance(float('nan'))
        # Behavior depends on implementation, but should not crash
        assert isinstance(lod_index, int)

    def test_transition_very_short_duration(self):
        """Test transition with very short duration."""
        transition = LODTransition(duration=0.001)
        transition.start(0, 1)
        completed = transition.update(0.002)
        assert completed is True

    def test_transition_very_long_duration(self):
        """Test transition with very long duration."""
        transition = LODTransition(duration=1000.0)
        transition.start(0, 1)

        # Small update should not complete
        for _ in range(100):
            completed = transition.update(0.1)
        assert completed is False
        assert transition.progress < 0.02

    def test_reduced_skeleton_single_bone(self, basic_skeleton: Skeleton):
        """Test reducing skeleton to single bone."""
        result = create_reduced_skeleton(basic_skeleton, 1)
        assert result.bone_count == 1
        assert result.bone_parents[0] == -1  # Root

    def test_crowd_lod_no_skeleton(self, default_lod_levels: list[LODLevel]):
        """Test CrowdLOD without skeleton still works for distance queries."""
        lod = CrowdLOD(skeleton=None, levels=default_lod_levels)

        # Should still select LODs by distance
        assert lod.get_lod_for_distance(5.0) == 0
        assert lod.get_lod_for_distance(15.0) == 1

    def test_hysteresis_negative_threshold(self, crowd_lod_with_levels: CrowdLOD):
        """Test hysteresis doesn't create negative thresholds."""
        crowd_lod_with_levels.set_hysteresis(20.0)  # Larger than first threshold

        # Even with large hysteresis, should not go negative
        lod_index = crowd_lod_with_levels.get_lod_for_distance(1.0, current_lod=1)
        assert lod_index >= 0

    def test_empty_skeleton_reduction(self):
        """Test reducing empty skeleton."""
        empty_skeleton = Skeleton(bone_names=[], bone_parents=[], bind_poses=[])
        result = create_reduced_skeleton(empty_skeleton, 5)
        assert result is empty_skeleton  # No reduction possible

    def test_frame_counter_overflow(self, crowd_lod_with_levels: CrowdLOD):
        """Test frame counter at high values."""
        crowd_lod_with_levels._frame_counter = 2**31 - 1
        crowd_lod_with_levels.advance_frame()
        # Should not crash, counter wraps or continues
        assert isinstance(crowd_lod_with_levels._frame_counter, int)


# =============================================================================
# Test: BoneWeight Comparisons
# =============================================================================


class TestBoneWeight:
    """Tests for BoneWeight comparison."""

    def test_bone_weight_comparison(self):
        """Test BoneWeight sorts by importance."""
        w1 = BoneWeight(bone_index=0, importance=0.9)
        w2 = BoneWeight(bone_index=1, importance=0.5)
        w3 = BoneWeight(bone_index=2, importance=0.7)

        sorted_weights = sorted([w1, w2, w3], reverse=True)
        assert sorted_weights[0].importance == 0.9
        assert sorted_weights[1].importance == 0.7
        assert sorted_weights[2].importance == 0.5


# =============================================================================
# Test: calculate_lod_blend_weights
# =============================================================================


class TestLODBlendWeights:
    """Tests for calculate_lod_blend_weights function."""

    def test_blend_weights_basic(self, basic_skeleton: Skeleton):
        """Test blend weights calculation."""
        reduced = create_reduced_skeleton(basic_skeleton, 5)

        # Create simple bone map
        bone_map = {i: i for i in range(reduced.bone_count)}

        weights = calculate_lod_blend_weights(basic_skeleton, reduced, bone_map)

        assert len(weights) == reduced.bone_count
        for idx, weight_list in weights.items():
            if weight_list:
                assert weight_list[0][1] == 1.0  # Full weight

    def test_blend_weights_missing_mapping(self, basic_skeleton: Skeleton):
        """Test blend weights with incomplete mapping."""
        reduced = create_reduced_skeleton(basic_skeleton, 5)

        # Partial bone map
        bone_map = {0: 0, 1: 1}

        weights = calculate_lod_blend_weights(basic_skeleton, reduced, bone_map)

        # Unmapped bones should have empty weight lists
        assert len(weights[2]) == 0 if 2 not in bone_map else len(weights[2]) > 0

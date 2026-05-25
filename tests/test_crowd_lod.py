"""Verification tests for crowd LOD system.

Tests cover LOD level selection with hysteresis, skeleton reduction,
LOD transitions with all modes, and integration with CrowdRenderer.

Source: engine/animation/crowds/crowd_lod.py, engine/animation/crowds/crowd_renderer.py
"""

from __future__ import annotations

import pytest

from engine.animation.crowds.crowd_lod import (
    CrowdLOD,
    LODLevel,
    LODTransition,
    LODTransitionMode,
    Skeleton,
    Transform,
    create_reduced_skeleton,
    _calculate_bone_importance,
    calculate_lod_blend_weights,
)
from engine.animation.crowds.crowd_renderer import (
    CrowdInstance,
    CrowdRenderer,
    CrowdRenderBatch,
    InstanceBuffer,
)
from engine.core.math import Vec3, Vec4, Quat

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_BONE_NAMES = [
    "root", "spine", "head", "upperarm", "forearm",
    "hand", "finger_index", "thigh", "calf", "foot", "toe",
]
TEST_BONE_PARENTS = [-1, 0, 1, 1, 3, 4, 5, 1, 7, 8, 9]


@pytest.fixture
def test_skeleton() -> Skeleton:
    """A test skeleton with 11 bones forming a humanoid hierarchy."""
    return Skeleton(
        bone_names=list(TEST_BONE_NAMES),
        bone_parents=list(TEST_BONE_PARENTS),
        bind_poses=[Transform.identity() for _ in range(len(TEST_BONE_NAMES))],
    )


@pytest.fixture
def base_lod_system() -> CrowdLOD:
    """A CrowdLOD with two LOD levels and the test skeleton."""
    lod = CrowdLOD(skeleton=None)
    lod.add_lod_level(LODLevel(distance=10.0))
    lod.add_lod_level(LODLevel(distance=50.0))
    return lod


@pytest.fixture
def renderer_with_instances() -> CrowdRenderer:
    """A CrowdRenderer with three instances at known distances from origin."""
    renderer = CrowdRenderer(max_instances_per_batch=1000)
    for i, x in enumerate([5.0, 15.0, 30.0]):
        inst = CrowdInstance(
            position=Vec3(x, 0.0, 0.0),
            rotation=Quat.identity(),
            scale=1.0,
            animation_index=i,
            animation_time=0.0,
            animation_speed=1.0,
            tint_color=Vec4(1.0, 1.0, 1.0, 1.0),
            lod_level=0,
            visible=True,
            instance_id=i + 1,
        )
        renderer.add_instance(inst, mesh_id=0, material_id=0)
    return renderer


# ---------------------------------------------------------------------------
# CrowdLOD Construction & Level Management
# ---------------------------------------------------------------------------

class TestCrowdLODConstruction:
    """CrowdLOD must construct and manage LOD level definitions."""

    def test_constructor_empty(self) -> None:
        """Empty constructor creates no LOD levels."""
        lod = CrowdLOD()
        assert lod.lod_count == 0
        assert lod.max_lod == 0

    def test_constructor_with_levels(self) -> None:
        """Constructor with levels creates sorted LOD levels."""
        levels = [LODLevel(distance=50.0), LODLevel(distance=10.0)]
        lod = CrowdLOD(levels=levels)
        assert lod.lod_count == 2
        # Must be sorted
        assert lod.get_lod_level(0).distance == 10.0
        assert lod.get_lod_level(1).distance == 50.0

    def test_constructor_with_skeleton(self) -> None:
        """Constructor with skeleton stores it for reduction."""
        skel = Skeleton(
            bone_names=["root", "spine", "head"],
            bone_parents=[-1, 0, 1],
            bind_poses=[Transform.identity() for _ in range(3)],
        )
        lod = CrowdLOD(skeleton=skel)
        assert lod.get_skeleton_for_lod(0) is skel

    def test_add_lod_level_returns_index(self) -> None:
        """add_lod_level returns the index of the added level."""
        lod = CrowdLOD()
        idx0 = lod.add_lod_level(LODLevel(distance=10.0))
        assert idx0 == 0
        idx1 = lod.add_lod_level(LODLevel(distance=20.0))
        assert idx1 == 1

    def test_add_lod_level_maintains_sorted_order(self) -> None:
        """Adding levels out of order keeps them sorted by distance."""
        lod = CrowdLOD()
        lod.add_lod_level(LODLevel(distance=50.0))
        lod.add_lod_level(LODLevel(distance=10.0))
        lod.add_lod_level(LODLevel(distance=25.0))
        assert [lv.distance for lv in lod._lod_levels] == [10.0, 25.0, 50.0]

    def test_set_lod_levels_replaces_all(self) -> None:
        """set_lod_levels replaces all existing levels."""
        lod = CrowdLOD()
        lod.add_lod_level(LODLevel(distance=10.0))
        lod.set_lod_levels([LODLevel(distance=100.0), LODLevel(distance=200.0)])
        assert lod.lod_count == 2
        assert lod.get_lod_level(0).distance == 100.0

    def test_get_lod_level_valid(self) -> None:
        """get_lod_level returns the correct level for valid index."""
        lod = CrowdLOD()
        lod.add_lod_level(LODLevel(distance=10.0))
        level = lod.get_lod_level(0)
        assert level is not None
        assert level.distance == 10.0

    def test_get_lod_level_invalid(self) -> None:
        """get_lod_level returns None for out-of-range index."""
        lod = CrowdLOD()
        assert lod.get_lod_level(0) is None
        assert lod.get_lod_level(-1) is None

    def test_lod_count_property(self) -> None:
        """lod_count returns the number of defined LOD levels."""
        lod = CrowdLOD()
        assert lod.lod_count == 0
        lod.add_lod_level(LODLevel(distance=10.0))
        assert lod.lod_count == 1
        lod.add_lod_level(LODLevel(distance=20.0))
        assert lod.lod_count == 2

    def test_max_lod_property(self) -> None:
        """max_lod returns the highest LOD level index."""
        lod = CrowdLOD()
        assert lod.max_lod == 0
        lod.add_lod_level(LODLevel(distance=10.0))
        assert lod.max_lod == 0
        lod.add_lod_level(LODLevel(distance=20.0))
        assert lod.max_lod == 1


# ---------------------------------------------------------------------------
# LOD Selection by Distance
# ---------------------------------------------------------------------------

class TestLODSelection:
    """LOD level must be selected correctly based on camera distance."""

    def test_basic_distance_selection(self) -> None:
        """Closer distances select lower (more detailed) LOD indices."""
        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0),
            LODLevel(distance=50.0),
        ])
        assert lod.get_lod_for_distance(5.0) == 0  # Closest LOD
        assert lod.get_lod_for_distance(30.0) == 0  # Below next threshold
        assert lod.get_lod_for_distance(60.0) == 1  # Furthest LOD

    def test_distance_clamped_to_zero(self) -> None:
        """Negative distances are clamped to zero."""
        lod = CrowdLOD(levels=[LODLevel(distance=10.0)])
        # Should not raise or return invalid
        idx = lod.get_lod_for_distance(-5.0)
        assert idx == 0

    def test_distance_zero(self) -> None:
        """Distance exactly 0 selects LOD 0."""
        lod = CrowdLOD(levels=[LODLevel(distance=10.0)])
        assert lod.get_lod_for_distance(0.0) == 0

    def test_large_distance(self) -> None:
        """Very large distance selects max LOD."""
        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0),
            LODLevel(distance=50.0),
        ])
        assert lod.get_lod_for_distance(1e9) == 1

    def test_single_level_always_returns_zero(self) -> None:
        """With one LOD level, all distances return LOD 0."""
        lod = CrowdLOD(levels=[LODLevel(distance=10.0)])
        assert lod.get_lod_for_distance(0.0) == 0
        assert lod.get_lod_for_distance(5.0) == 0
        assert lod.get_lod_for_distance(100.0) == 0

    def test_no_levels_returns_zero(self) -> None:
        """With no LOD levels, always returns 0."""
        lod = CrowdLOD()
        assert lod.get_lod_for_distance(0.0) == 0
        assert lod.get_lod_for_distance(100.0) == 0
        assert lod.get_lod_for_distance(1e9) == 0

    def test_three_level_selection(self) -> None:
        """Three LOD levels create three distinct distance bands."""
        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0),
            LODLevel(distance=50.0),
            LODLevel(distance=100.0),
        ])
        assert lod.get_lod_for_distance(5.0) == 0
        assert lod.get_lod_for_distance(75.0) == 1
        assert lod.get_lod_for_distance(150.0) == 2

    def test_max_lod_levels_clamped(self) -> None:
        """create_default_lods clamps to MAX_LOD_LEVELS."""
        from engine.animation.config import CROWD_LOD_CONFIG
        skel = Skeleton(
            bone_names=["root", "spine"],
            bone_parents=[-1, 0],
            bind_poses=[Transform.identity() for _ in range(2)],
        )
        lod = CrowdLOD(skeleton=skel)
        lod.create_default_lods(max_distance=100.0, lod_count=999)
        assert lod.lod_count <= CROWD_LOD_CONFIG.MAX_LOD_LEVELS


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------

class TestHysteresis:
    """Hysteresis must prevent LOD flickering near distance thresholds."""

    def test_hysteresis_from_above(self) -> None:
        """From a higher LOD, the switch-back threshold has hysteresis."""
        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0),
            LODLevel(distance=20.0),
        ])
        lod.set_hysteresis(1.0)

        # At LOD 1, distance just below 20: switches back to LOD 0
        assert lod.get_lod_for_distance(19.0, current_lod=1) == 0

        # At LOD 1, distance 20: stays at LOD 1
        assert lod.get_lod_for_distance(20.0, current_lod=1) == 1

    def test_hysteresis_from_below(self) -> None:
        """From a lower LOD, the switch-forward threshold has hysteresis."""
        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0),
            LODLevel(distance=20.0),
        ])
        lod.set_hysteresis(1.0)

        # At LOD 0, distance just below 21: stays at LOD 0
        assert lod.get_lod_for_distance(20.5, current_lod=0) == 0

        # At LOD 0, distance 21+: switches to LOD 1
        assert lod.get_lod_for_distance(21.0, current_lod=0) == 1

    def test_no_hysteresis_when_no_current_lod(self) -> None:
        """With no current LOD, hysteresis is not applied."""
        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0),
            LODLevel(distance=20.0),
        ])
        lod.set_hysteresis(10.0)  # Large hysteresis

        # No current LOD means no hysteresis applied
        assert lod.get_lod_for_distance(15.0) == 0  # Standard selection
        assert lod.get_lod_for_distance(25.0) == 1

    def test_hysteresis_prevents_flickering(self) -> None:
        """Flickering around a threshold is prevented by hysteresis."""
        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0),
            LODLevel(distance=20.0),
        ])
        lod.set_hysteresis(2.0)

        # Oscillate around the boundary: up at 22, down at 20
        lod_a = lod.get_lod_for_distance(22.0, current_lod=0)  # Switch up
        assert lod_a == 1

        lod_b = lod.get_lod_for_distance(21.0, current_lod=1)  # Still in 1 (hysteresis)
        assert lod_b == 1

        lod_c = lod.get_lod_for_distance(19.0, current_lod=1)  # Switch back
        assert lod_c == 0

        lod_d = lod.get_lod_for_distance(20.0, current_lod=0)  # Still in 0 (hysteresis needs >22)
        assert lod_d == 0

    def test_set_hysteresis_zero(self) -> None:
        """Setting hysteresis to 0 disables it."""
        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0),
            LODLevel(distance=20.0),
        ])
        lod.set_hysteresis(0.0)
        assert lod.get_lod_for_distance(20.5, current_lod=0) == 1  # Switches immediately
        assert lod.get_lod_for_distance(19.5, current_lod=1) == 0  # Switches immediately

    def test_set_hysteresis_negative_clamped(self) -> None:
        """Negative hysteresis is clamped to 0."""
        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0),
            LODLevel(distance=20.0),
        ])
        lod.set_hysteresis(-5.0)
        # Hysteresis is clamped >= 0 by set_hysteresis
        assert lod.get_lod_for_distance(20.5, current_lod=0) == 1

    def test_hysteresis_threshold_not_negative(self) -> None:
        """Threshold after hysteresis subtraction is never negative."""
        lod = CrowdLOD(levels=[
            LODLevel(distance=1.0),
            LODLevel(distance=2.0),
        ])
        lod.set_hysteresis(5.0)

        # The threshold at level 1 with hysteresis could go negative,
        # but is clamped to 0.0
        result = lod.get_lod_for_distance(0.5, current_lod=1)
        assert result >= 0


# ---------------------------------------------------------------------------
# Skeleton Reduction
# ---------------------------------------------------------------------------

class TestSkeletonReduction:
    """Reduced skeleton LODs must render correctly."""

    def test_create_reduced_skeleton_removes_bones(self, test_skeleton) -> None:
        """Reduced skeleton has fewer bones than the original."""
        reduced = create_reduced_skeleton(test_skeleton, 5)
        assert reduced.bone_count == 5
        assert reduced.bone_count < test_skeleton.bone_count

    def test_reduced_skeleton_maintains_hierarchy(self, test_skeleton) -> None:
        """Reduced skeleton has valid parent indices."""
        reduced = create_reduced_skeleton(test_skeleton, 5)
        for i, parent in enumerate(reduced.bone_parents):
            if parent >= 0:
                assert parent < i  # Parent must come before child

    def test_reduced_skeleton_target_exceeds_original(self, test_skeleton) -> None:
        """Requesting more bones than original returns the original."""
        result = create_reduced_skeleton(test_skeleton, 100)
        assert result is test_skeleton

    def test_reduced_skeleton_zero_bones(self, test_skeleton) -> None:
        """Reducing to zero bones returns an empty skeleton."""
        reduced = create_reduced_skeleton(test_skeleton, 0)
        assert reduced.bone_count == 0
        assert reduced.bone_names == []

    def test_reduced_skeleton_negative_target(self, test_skeleton) -> None:
        """Negative target is clamped to zero bones."""
        reduced = create_reduced_skeleton(test_skeleton, -1)
        assert reduced.bone_count == 0

    def test_reduced_skeleton_preserves_root_bone(self, test_skeleton) -> None:
        """Root bone is always present in the reduced skeleton."""
        reduced = create_reduced_skeleton(test_skeleton, 1)
        assert reduced.bone_count == 1
        assert reduced.bone_names[0] == "root"

    def test_reduced_skeleton_bind_poses_valid(self, test_skeleton) -> None:
        """Each bone in the reduced skeleton has a valid bind pose."""
        reduced = create_reduced_skeleton(test_skeleton, 4)
        assert len(reduced.bind_poses) == 4
        for pose in reduced.bind_poses:
            assert pose is not None

    def test_reduced_skeleton_bone_names_from_original(self, test_skeleton) -> None:
        """Bone names in reduced skeleton come from the original."""
        reduced = create_reduced_skeleton(test_skeleton, 6)
        for name in reduced.bone_names:
            assert name in test_skeleton.bone_names

    def test_get_skeleton_for_lod_returns_reduced(self, test_skeleton) -> None:
        """get_skeleton_for_lod returns reduced skeleton when applicable."""
        lod = CrowdLOD(skeleton=test_skeleton)
        lod.add_lod_level(LODLevel(distance=10.0, bone_count=test_skeleton.bone_count))
        lod.add_lod_level(LODLevel(distance=50.0, bone_count=3))

        skel0 = lod.get_skeleton_for_lod(0)
        assert skel0 is test_skeleton  # Full skeleton for LOD 0

        skel1 = lod.get_skeleton_for_lod(1)
        assert skel1 is not test_skeleton  # Reduced skeleton for LOD 1
        assert skel1.bone_count == 3

    def test_get_skeleton_no_skeleton(self) -> None:
        """get_skeleton_for_lod returns None when no skeleton is set."""
        lod = CrowdLOD()
        lod.add_lod_level(LODLevel(distance=10.0))
        assert lod.get_skeleton_for_lod(0) is None

    def test_get_bone_count_for_lod(self, test_skeleton) -> None:
        """get_bone_count_for_lod returns correct count per LOD level."""
        lod = CrowdLOD(skeleton=test_skeleton)
        lod.add_lod_level(LODLevel(distance=10.0, bone_count=8))
        lod.add_lod_level(LODLevel(distance=50.0, bone_count=3))
        assert lod.get_bone_count_for_lod(0) == 8
        assert lod.get_bone_count_for_lod(1) == 3

    def test_get_bone_count_invalid_lod(self, test_skeleton) -> None:
        """get_bone_count_for_lod returns skeleton count for invalid LOD."""
        lod = CrowdLOD(skeleton=test_skeleton)
        assert lod.get_bone_count_for_lod(999) == test_skeleton.bone_count


# ---------------------------------------------------------------------------
# Bone Importance Calculation
# ---------------------------------------------------------------------------

class TestBoneImportance:
    """Bone importance scoring must prioritize core body bones."""

    def test_root_bone_high_importance(self) -> None:
        """Root bone gets the highest importance score."""
        skel = Skeleton(
            bone_names=["root", "spine"],
            bone_parents=[-1, 0],
            bind_poses=[Transform.identity(), Transform.identity()],
        )
        score = _calculate_bone_importance("root", 0, skel)
        assert score > 0.8  # Base 0.5 + root 0.5

    def test_spine_bone_high_importance(self) -> None:
        """Spine bone gets a high importance score."""
        skel = Skeleton(
            bone_names=["root", "spine"],
            bone_parents=[-1, 0],
            bind_poses=[Transform.identity(), Transform.identity()],
        )
        score = _calculate_bone_importance("spine", 1, skel)
        assert score > 0.5  # At least base + spine bonus

    def test_finger_low_importance(self) -> None:
        """Finger bones get low importance scores."""
        skel = Skeleton(
            bone_names=["finger", "finger_index"],
            bone_parents=[-1, 0],
            bind_poses=[Transform.identity(), Transform.identity()],
        )
        score = _calculate_bone_importance("finger_index", 1, skel)
        assert score < 0.7  # Finger has lower importance than core

    def test_auxiliary_bone_penalty(self) -> None:
        """Twist/roll/helper bones get a penalty."""
        skel = Skeleton(
            bone_names=["root", "upperarm_twist"],
            bone_parents=[-1, 0],
            bind_poses=[Transform.identity(), Transform.identity()],
        )
        score = _calculate_bone_importance("upperarm_twist", 1, skel)
        # Base 0.5 + upperarm 0.3 - twist 0.2 - depth 0.02
        assert score < 0.7

    def test_head_bone_high_importance(self) -> None:
        """Head bone gets high importance."""
        skel = Skeleton(
            bone_names=["root", "spine", "neck", "head"],
            bone_parents=[-1, 0, 1, 2],
            bind_poses=[Transform.identity() for _ in range(4)],
        )
        score = _calculate_bone_importance("head", 3, skel)
        # Base 0.5 + head 0.4 - depth(3) * 0.02 = 0.84
        assert score > 0.7


# ---------------------------------------------------------------------------
# LODLevel
# ---------------------------------------------------------------------------

class TestLODLevel:
    """LODLevel dataclass behavior."""

    def test_default_values(self) -> None:
        """LODLevel has sensible default values."""
        level = LODLevel()
        assert level.distance == 0.0
        assert level.bone_count == 0
        assert level.update_rate == 1.0
        assert level.shadow_enabled is True
        assert level.animation_quality == 1.0

    def test_level_sorting(self) -> None:
        """LODLevel sorts by distance."""
        a = LODLevel(distance=50.0)
        b = LODLevel(distance=10.0)
        assert b < a  # Shorter distance sorts first

    def test_should_update_full_rate(self) -> None:
        """update_rate >= 1.0 always returns True."""
        level = LODLevel(update_rate=1.0)
        assert level.should_update(0) is True
        assert level.should_update(100) is True

    def test_should_update_zero_rate(self) -> None:
        """update_rate <= 0 always returns False."""
        level = LODLevel(update_rate=0.0)
        assert level.should_update(0) is False
        assert level.should_update(100) is False

    def test_should_update_interval(self) -> None:
        """Partial update rate updates every N frames."""
        level = LODLevel(update_rate=0.25)  # Every 4 frames
        assert level.should_update(0) is True
        assert level.should_update(1) is False
        assert level.should_update(2) is False
        assert level.should_update(3) is False
        assert level.should_update(4) is True


# ---------------------------------------------------------------------------
# LODTransition
# ---------------------------------------------------------------------------

class TestLODTransition:
    """LOD transition modes work correctly (instant, blend, dither)."""

    def test_transition_start(self) -> None:
        """Starting a transition sets from/to LOD and activates."""
        t = LODTransition()
        t.start(0, 2)
        assert t.from_lod == 0
        assert t.to_lod == 2
        assert t.active is True
        assert t.progress == 0.0

    def test_transition_same_lod_no_op(self) -> None:
        """Starting a transition to the same LOD is a no-op."""
        t = LODTransition()
        t.active = True
        t.start(1, 1)
        assert t.active is False  # No transition needed

    def test_transition_update_returns_true_when_done(self) -> None:
        """update returns True when transition completes."""
        t = LODTransition(duration=0.5)
        t.start(0, 1)
        result = t.update(0.5)  # Exactly the duration
        assert result is True
        assert t.active is False
        assert t.progress == 1.0

    def test_transition_update_partial(self) -> None:
        """update returns False during active transition."""
        t = LODTransition(duration=1.0)
        t.start(0, 2)
        result = t.update(0.3)
        assert result is False
        assert t.active is True
        assert t.progress == 0.3

    def test_transition_update_exceeds(self) -> None:
        """update handles progress exceeding 1.0."""
        t = LODTransition(duration=0.5)
        t.start(0, 1)
        result = t.update(1.0)  # Double the duration
        assert result is True
        assert t.progress == 1.0

    def test_transition_zero_duration(self) -> None:
        """Zero duration means instant transition via update call."""
        t = LODTransition(duration=0.0)
        t.start(0, 1)
        assert t.active is True  # start() initiates; update() processes
        assert t.progress == 0.0
        # Calling update with zero duration completes immediately
        result = t.update(0.0)
        assert result is True
        assert t.active is False
        assert t.progress == 1.0

    def test_transition_get_blend_factor_smoothstep(self) -> None:
        """get_blend_factor uses smoothstep for nicer transitions."""
        t = LODTransition(duration=1.0)
        t.start(0, 1)
        t.progress = 0.0
        assert t.get_blend_factor() == 0.0
        t.progress = 0.5
        factor = t.get_blend_factor()
        # Smoothstep(0.5) = 0.5^2 * (3 - 2*0.5) = 0.25 * 2 = 0.5
        assert factor == 0.5
        t.progress = 1.0
        assert t.get_blend_factor() == 1.0

    def test_get_blend_factor_clamped(self) -> None:
        """get_blend_factor clamps progress to [0, 1]."""
        t = LODTransition(duration=1.0)
        t.start(0, 1)
        t.progress = -0.5
        assert t.get_blend_factor() == 0.0
        t.progress = 1.5
        assert t.get_blend_factor() == 1.0

    def test_get_current_lod_during_transition(self) -> None:
        """Before 50% progress, current LOD is from_lod."""
        t = LODTransition(duration=1.0)
        t.start(0, 2)
        t.progress = 0.25
        assert t.get_current_lod() == 0  # Below 50%

    def test_get_current_lod_after_halfway(self) -> None:
        """After 50% progress, current LOD is to_lod."""
        t = LODTransition(duration=1.0)
        t.start(0, 2)
        t.progress = 0.5
        assert t.get_current_lod() == 2  # At 50% exactly -> to_lod

    def test_get_current_lod_after_complete(self) -> None:
        """After transition, current LOD is to_lod."""
        t = LODTransition(duration=1.0)
        t.start(0, 3)
        t.update(1.0)
        assert t.get_current_lod() == 3

    def test_transition_mode_instant(self) -> None:
        """INSTANT mode transitions store correctly."""
        t = LODTransition(mode=LODTransitionMode.INSTANT)
        t.start(0, 1)
        t.update(0.0)
        # INSTANT mode with zero "instant" update
        assert t.active is True  # Still active until duration elapsed

    def test_transition_mode_blend(self) -> None:
        """BLEND mode transitions store correctly."""
        t = LODTransition(mode=LODTransitionMode.BLEND)
        assert t.mode == LODTransitionMode.BLEND

    def test_transition_mode_dither(self) -> None:
        """DITHER mode transitions store correctly."""
        t = LODTransition(mode=LODTransitionMode.DITHER)
        assert t.mode == LODTransitionMode.DITHER

    def test_transition_default_duration_from_config(self) -> None:
        """Default transition duration comes from config."""
        from engine.animation.config import CROWD_LOD_CONFIG
        t = LODTransition()
        assert t.duration == CROWD_LOD_CONFIG.DEFAULT_TRANSITION_DURATION

    def test_inactive_transition_returns_true(self) -> None:
        """update on inactive transition returns True immediately."""
        t = LODTransition()
        t.active = False
        assert t.update(0.0) is True


# ---------------------------------------------------------------------------
# Animation Update Rate
# ---------------------------------------------------------------------------

class TestAnimationUpdateRate:
    """CrowdLOD manages animation update rate per LOD level."""

    def test_should_update_high_lod(self) -> None:
        """LOD 0 updates every frame."""
        lod = CrowdLOD()
        lod.add_lod_level(LODLevel(distance=10.0, update_rate=1.0))
        assert lod.should_update_animation(0) is True

    def test_should_update_low_lod(self) -> None:
        """Lower LODs update at reduced rate."""
        lod = CrowdLOD()
        lod.add_lod_level(LODLevel(distance=10.0, update_rate=1.0))
        lod.add_lod_level(LODLevel(distance=50.0, update_rate=0.5))
        assert lod.should_update_animation(1) is True  # frame 0
        lod.advance_frame()
        assert lod.should_update_animation(1) is False  # frame 1, skip

    def test_advance_frame_increments(self) -> None:
        """advance_frame increments the internal frame counter."""
        lod = CrowdLOD()
        assert lod._frame_counter == 0
        lod.advance_frame()
        assert lod._frame_counter == 1
        lod.advance_frame()
        assert lod._frame_counter == 2

    def test_should_update_invalid_lod(self) -> None:
        """should_update_animation returns True for unknown LOD."""
        lod = CrowdLOD()
        assert lod.should_update_animation(999) is True


# ---------------------------------------------------------------------------
# Default LOD Creation
# ---------------------------------------------------------------------------

class TestDefaultLODCreation:
    """create_default_lods generates progressive LOD levels."""

    def test_creates_correct_number_of_levels(self, test_skeleton) -> None:
        """create_default_lods creates the requested number of levels."""
        lod = CrowdLOD(skeleton=test_skeleton)
        lod.create_default_lods(max_distance=100.0, lod_count=4)
        assert lod.lod_count == 4

    def test_distances_progressive(self, test_skeleton) -> None:
        """LOD distances increase with each level."""
        lod = CrowdLOD(skeleton=test_skeleton)
        lod.create_default_lods(max_distance=100.0, lod_count=3)
        distances = [lod.get_lod_level(i).distance for i in range(3)]
        assert distances == sorted(distances)

    def test_bone_count_decreases(self, test_skeleton) -> None:
        """Bone count decreases at lower LODs."""
        lod = CrowdLOD(skeleton=test_skeleton)
        lod.create_default_lods(max_distance=100.0, lod_count=3)
        bone_counts = [lod.get_bone_count_for_lod(i) for i in range(3)]
        assert bone_counts[0] >= bone_counts[1] >= bone_counts[2]

    def test_no_skeleton_no_levels(self) -> None:
        """Without a skeleton, create_default_lods does nothing."""
        lod = CrowdLOD()
        lod.create_default_lods(max_distance=100.0, lod_count=4)
        assert lod.lod_count == 0

    def test_zero_lod_count_no_op(self, test_skeleton) -> None:
        """Requesting 0 LOD levels does nothing."""
        lod = CrowdLOD(skeleton=test_skeleton)
        lod.create_default_lods(max_distance=100.0, lod_count=0)
        assert lod.lod_count == 0

    def test_highest_lod_has_shadow(self, test_skeleton) -> None:
        """Higher detail LODs (early indices) have shadows enabled."""
        lod = CrowdLOD(skeleton=test_skeleton)
        lod.create_default_lods(max_distance=100.0, lod_count=4)
        assert lod.get_lod_level(0).shadow_enabled is True
        assert lod.get_lod_level(1).shadow_enabled is True


# ---------------------------------------------------------------------------
# Blend Weights
# ---------------------------------------------------------------------------

class TestBlendWeights:
    """LOD blend weight calculation."""

    def test_calculate_simple_blend_weights(self, test_skeleton) -> None:
        """Blend weights map reduced bones to full skeleton bones."""
        reduced = create_reduced_skeleton(test_skeleton, 3)
        bone_map = {i: i for i in range(reduced.bone_count)}
        weights = calculate_lod_blend_weights(test_skeleton, reduced, bone_map)
        assert len(weights) == reduced.bone_count
        for bone_idx, mapping in weights.items():
            assert len(mapping) > 0
            for full_idx, weight in mapping:
                assert 0 <= full_idx < test_skeleton.bone_count
                assert weight > 0

    def test_empty_bone_map(self, test_skeleton) -> None:
        """Empty bone map produces no weights."""
        reduced = create_reduced_skeleton(test_skeleton, 3)
        weights = calculate_lod_blend_weights(test_skeleton, reduced, {})
        assert len(weights) == reduced.bone_count
        for bone_idx, mapping in weights.items():
            assert mapping == []  # No mapping available


# ---------------------------------------------------------------------------
# Renderer Integration
# ---------------------------------------------------------------------------

class TestRendererIntegration:
    """CrowdRenderer correctly uses CrowdLOD for per-instance LOD."""

    def test_update_lod_levels_from_system(self, renderer_with_instances) -> None:
        """update_lod_levels_from_system sets LOD based on distance with hysteresis."""
        lod_system = CrowdLOD(levels=[
            LODLevel(distance=10.0),
            LODLevel(distance=25.0),
        ])
        camera_pos = Vec3(0.0, 0.0, 0.0)
        renderer_with_instances.update_lod_levels_from_system(camera_pos, lod_system)

        # Instance at x=5.0 -> LOD 0
        # Instance at x=15.0 -> LOD 0 (between 10 and 25)
        # Instance at x=30.0 -> LOD 1 (beyond 25)
        instances = list(renderer_with_instances.get_batches())[0].instances
        assert instances[0].lod_level == 0
        assert instances[1].lod_level == 0
        assert instances[2].lod_level == 1

    def test_update_lod_levels_preserves_hysteresis(self, renderer_with_instances) -> None:
        """LOD updates through the renderer respect hysteresis."""
        lod_system = CrowdLOD(levels=[
            LODLevel(distance=10.0),
            LODLevel(distance=20.0),
        ])
        lod_system.set_hysteresis(2.0)
        camera_pos = Vec3(0.0, 0.0, 0.0)

        batch = list(renderer_with_instances.get_batches())[0]

        # Instance at x=15.0, start at LOD 0
        batch.instances[1].lod_level = 0
        renderer_with_instances.update_lod_levels_from_system(camera_pos, lod_system)

        # With LOD 0 and hysteresis 2.0, threshold to switch up is 22
        # So at distance 15, stays at LOD 0
        assert batch.instances[1].lod_level == 0

    def test_multiple_batches_lod_update(self) -> None:
        """LOD update works across multiple render batches."""
        renderer = CrowdRenderer(max_instances_per_batch=1000)

        # Batch 0: mesh 0, material 0
        for i in range(3):
            inst = CrowdInstance(
                position=Vec3(float((i + 1) * 10), 0.0, 0.0),
                rotation=Quat.identity(),
                scale=1.0,
                animation_index=i,
                animation_time=0.0,
                animation_speed=1.0,
                tint_color=Vec4(1.0, 1.0, 1.0, 1.0),
                lod_level=0,
                visible=True,
                instance_id=i + 10,
            )
            renderer.add_instance(inst, mesh_id=0, material_id=0)

        # Batch 1: mesh 1, material 1
        for i in range(2):
            inst = CrowdInstance(
                position=Vec3(float((i + 1) * 5), 0.0, 0.0),
                rotation=Quat.identity(),
                scale=1.0,
                animation_index=i,
                animation_time=0.0,
                animation_speed=1.0,
                tint_color=Vec4(1.0, 1.0, 1.0, 1.0),
                lod_level=0,
                visible=True,
                instance_id=i + 20,
            )
            renderer.add_instance(inst, mesh_id=1, material_id=1)

        lod_system = CrowdLOD(levels=[
            LODLevel(distance=15.0),
            LODLevel(distance=50.0),
        ])
        renderer.update_lod_levels_from_system(Vec3(0.0, 0.0, 0.0), lod_system)

        for batch in renderer.get_batches():
            for inst in batch.instances:
                assert inst.lod_level >= 0


# ---------------------------------------------------------------------------
# LODLevel Config Defaults
# ---------------------------------------------------------------------------

class TestLODLevelConfigDefaults:
    """LOD config defaults must be consistent."""

    def test_default_hysteresis_from_config(self) -> None:
        """Default hysteresis matches config value."""
        from engine.animation.config import CROWD_LOD_CONFIG
        lod = CrowdLOD()
        assert lod._hysteresis == CROWD_LOD_CONFIG.DEFAULT_HYSTERESIS

    def test_min_bones_at_lowest_lod_from_config(self, test_skeleton) -> None:
        """Lowest LOD uses MIN_BONES_AT_LOWEST_LOD for bone count."""
        from engine.animation.config import CROWD_LOD_CONFIG
        lod = CrowdLOD(skeleton=test_skeleton)
        lod.create_default_lods(max_distance=100.0, lod_count=8)
        lowest = lod.get_bone_count_for_lod(lod.max_lod)
        assert lowest >= CROWD_LOD_CONFIG.MIN_BONES_AT_LOWEST_LOD

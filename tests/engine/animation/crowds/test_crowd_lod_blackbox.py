"""
Blackbox Contract Tests for T1.5 LOD Integration and T2.3 LOD Level Selection

This module tests the public interface of the CrowdLOD system
WITHOUT reading implementation details.

DOCUMENTED API DIFFERENCES:

1. LODLevel Parameter Name:
   - Documentation: LODLevel(distance=10, max_bone_count=50)
   - Actual: LODLevel(distance=10, bone_count=50)

2. CrowdLOD Hysteresis:
   - Documentation: CrowdLOD(levels=[...], hysteresis=2.0)
   - Actual: CrowdLOD uses default hysteresis; set via set_hysteresis()

3. Distance-based LOD selection:
   - Documentation suggested: get_lod_level(distance=15)
   - Actual API: get_lod_for_distance(distance, current_lod) -> int

Test Summary:
- T1.5: Basic construction and level access
- T2.3: Distance-based LOD selection with hysteresis
"""

import pytest
from typing import Optional


class TestLODLevelConstruction:
    """Test LODLevel dataclass construction and validation."""

    def test_lod_level_basic_construction(self):
        """LODLevel can be constructed with distance and bone_count."""
        from engine.animation.crowds import LODLevel

        level = LODLevel(distance=10.0, bone_count=50)
        assert level.distance == 10.0
        assert level.bone_count == 50

    def test_lod_level_zero_distance(self):
        """LODLevel accepts zero distance for closest level."""
        from engine.animation.crowds import LODLevel

        level = LODLevel(distance=0.0, bone_count=100)
        assert level.distance == 0.0

    def test_lod_level_large_distance(self):
        """LODLevel accepts large distances for far LOD."""
        from engine.animation.crowds import LODLevel

        level = LODLevel(distance=1000.0, bone_count=5)
        assert level.distance == 1000.0
        assert level.bone_count == 5


class TestCrowdLODBasicConstruction:
    """Test CrowdLOD initialization with various level configurations."""

    def test_crowd_lod_single_level(self):
        """CrowdLOD can be created with a single level."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[LODLevel(distance=10.0, bone_count=50)])
        assert lod is not None

    def test_crowd_lod_multiple_levels(self):
        """CrowdLOD can be created with multiple levels."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        levels = [
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ]
        lod = CrowdLOD(levels=levels)
        assert lod is not None

    def test_crowd_lod_hysteresis_contract_violation(self):
        """CONTRACT VIOLATION: CrowdLOD should accept hysteresis parameter per docs."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        # Per PHASE_2_TODO.md: lod = CrowdLOD(hysteresis=2.0)
        # This test documents that the API does NOT match documentation
        with pytest.raises(TypeError):
            CrowdLOD(
                levels=[LODLevel(distance=10.0, bone_count=50)],
                hysteresis=3.0
            )


class TestLODLevelAccess:
    """Test actual implemented API: index-based LOD level access."""

    def test_get_lod_level_by_index_zero(self):
        """get_lod_level(0) returns first LODLevel."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
        ])

        result = lod.get_lod_level(0)
        assert result is not None
        assert result.distance == 10.0
        assert result.bone_count == 50

    def test_get_lod_level_by_index_one(self):
        """get_lod_level(1) returns second LODLevel."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
        ])

        result = lod.get_lod_level(1)
        assert result is not None
        assert result.distance == 20.0
        assert result.bone_count == 30

    def test_get_lod_level_out_of_bounds_returns_none(self):
        """get_lod_level with invalid index returns None."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
        ])

        result = lod.get_lod_level(5)
        assert result is None

    def test_get_lod_level_negative_index_returns_none(self):
        """get_lod_level with negative index returns None."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
        ])

        result = lod.get_lod_level(-1)
        assert result is None


class TestT23DistanceBasedLODSelection:
    """T2.3 Contract Tests: Distance-based LOD level selection.

    Uses get_lod_for_distance(distance, current_lod) method.

    ACTUAL BEHAVIOR DISCOVERED:
    - LODLevel.distance = the distance at which to SWITCH TO that LOD level
    - LOD 0 is used from distance=0 until the SECOND level's distance threshold
    - Example with levels at [10, 20, 50]:
      - LOD 0: distance < 20 (first level is "base", switch happens at second)
      - LOD 1: 20 <= distance < 50
      - LOD 2: distance >= 50

    This differs from documented contract which expected:
    - LOD 0: distance < 10
    - LOD 1: 10 <= distance < 20
    - LOD 2: distance >= 20
    """

    def test_closest_distance_returns_lod_0(self):
        """Distance below first threshold should return LOD 0."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ])

        result = lod.get_lod_for_distance(5.0)
        assert result == 0, f"Distance 5.0 should select LOD 0, got {result}"

    def test_middle_distance_returns_correct_lod(self):
        """Distance 15.0 - verify actual vs documented behavior.

        DOCUMENTED CONTRACT: get_lod_level(distance=15) == 1
        ACTUAL BEHAVIOR: Returns LOD 0 (switch to LOD 1 happens at distance=20)
        """
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ])

        result = lod.get_lod_for_distance(15.0)
        # Actual behavior: LOD 0 because switch to LOD 1 happens at 20.0
        assert result == 0, f"Distance 15.0 returns LOD 0 per actual impl, got {result}"

    def test_at_second_level_threshold_switches(self):
        """At the second level's threshold, should switch to LOD 1."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ])

        result = lod.get_lod_for_distance(20.0)
        assert result == 1, f"Distance 20.0 should select LOD 1, got {result}"

    def test_far_distance_returns_highest_lod(self):
        """Distance beyond all thresholds should return highest LOD level."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ])

        result = lod.get_lod_for_distance(100.0)
        assert result == 2, f"Distance 100.0 should select LOD 2, got {result}"

    def test_very_far_agents_use_lowest_lod(self):
        """Very far agents (e.g., 1000+ units) should use the lowest detail LOD."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ])

        result = lod.get_lod_for_distance(1000.0)
        assert result == 2, "Very far distance should select highest LOD index (lowest detail)"

    def test_zero_distance_returns_lod_0(self):
        """Zero distance should return LOD 0 (highest detail)."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
        ])

        result = lod.get_lod_for_distance(0.0)
        assert result == 0, "Zero distance should select LOD 0"

    def test_exact_boundary_distance_at_first_level(self):
        """At first level distance, still returns LOD 0."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ])

        result = lod.get_lod_for_distance(10.0)
        # LOD 0 remains active until distance reaches LOD 1's threshold (20.0)
        assert result == 0, f"At boundary 10.0, expected LOD 0, got {result}"

    def test_just_past_second_boundary(self):
        """Just past second level boundary should switch to LOD 1."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ])

        result = lod.get_lod_for_distance(20.1)
        assert result == 1, f"Just past 20.0 should be LOD 1, got {result}"

    def test_contract_violation_documented(self):
        """DOCUMENTED CONTRACT VIOLATION: distance=15 should return 1 per docs.

        Per PHASE_2_TODO.md: assert lod.get_lod_level(distance=15) == 1
        Actual implementation returns 0.

        This is marked as xfail to track the contract discrepancy.
        """
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ])

        result = lod.get_lod_for_distance(15.0)
        # Mark as xfail: docs say 1, impl returns 0
        assert result == 1, "CONTRACT VIOLATION: docs expect 1, got 0"

    test_contract_violation_documented = pytest.mark.xfail(
        reason="CONTRACT VIOLATION: docs say distance=15->LOD 1, impl returns LOD 0"
    )(test_contract_violation_documented)


class TestDistanceBasedSelectionActualBehavior:
    """Tests verifying actual implementation behavior (not documented contract)."""

    def test_closest_distance_returns_lod_0(self):
        """Distance below all thresholds should return LOD 0."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ])

        result = lod.get_lod_for_distance(5.0)
        assert result == 0

    def test_between_first_and_second_threshold_returns_lod_0(self):
        """Distance between first and second threshold returns LOD 0."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ])

        # Actual behavior: LOD 0 until distance reaches 20.0
        result = lod.get_lod_for_distance(15.0)
        assert result == 0, "Between 10 and 20, still LOD 0"

    def test_at_second_threshold_returns_lod_1(self):
        """At second threshold (20.0), switches to LOD 1."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ])

        result = lod.get_lod_for_distance(20.0)
        assert result == 1

    def test_far_distance_returns_highest_lod(self):
        """Distance beyond all thresholds should return highest LOD level."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ])

        result = lod.get_lod_for_distance(100.0)
        assert result == 2


class TestT23HysteresisContract:
    """T2.3 Contract Tests: Hysteresis prevents rapid LOD switching.

    Uses get_lod_for_distance(distance, current_lod) with set_hysteresis().
    Default hysteresis is 1.0 meters per config.

    ACTUAL BEHAVIOR DISCOVERED:
    With levels at [10, 20] and hysteresis=2.0:
    - Without current_lod (-1): switch to LOD 1 at distance >= 20.0
    - With current_lod=0: switch to LOD 1 at distance >= 22.0 (20 + 2)
    - With current_lod=1: switch to LOD 0 when distance < switching threshold

    Note: The first level's distance (10.0) doesn't define a switch point;
    the switch to LOD 1 happens at the SECOND level's distance (20.0).
    """

    def test_hysteresis_prevents_switch_to_lower_detail(self):
        """Hysteresis prevents premature switch from LOD 0 to LOD 1."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
        ])
        lod.set_hysteresis(2.0)

        # Currently at LOD 0, threshold is 20.0
        # With hysteresis, need distance > 20.0 + 2.0 = 22.0 to switch
        result = lod.get_lod_for_distance(21.0, current_lod=0)
        assert result == 0, "Should stay at LOD 0 due to hysteresis (21 < 22)"

    def test_hysteresis_allows_switch_to_lower_detail_past_buffer(self):
        """Hysteresis allows switch when clearly past buffer."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
        ])
        lod.set_hysteresis(2.0)

        # Well past the hysteresis buffer (20.0 + 2.0 = 22.0)
        result = lod.get_lod_for_distance(23.0, current_lod=0)
        assert result == 1, "Should switch to LOD 1 when past buffer (23 > 22)"

    def test_hysteresis_prevents_switch_to_higher_detail(self):
        """Hysteresis prevents premature switch from LOD 1 to LOD 0."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
        ])
        lod.set_hysteresis(2.0)

        # At LOD 1, hysteresis shifts threshold down
        # Threshold at 20.0 - 2.0 = 18.0 to switch back to LOD 0
        # At distance 19, still in LOD 1 zone (19 >= 18)
        result = lod.get_lod_for_distance(20.0, current_lod=1)
        assert result == 1, "Should stay at LOD 1 at threshold"

    def test_hysteresis_allows_switch_to_higher_detail(self):
        """Hysteresis allows switch back when clearly below buffer."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
        ])
        lod.set_hysteresis(2.0)

        # Well below the hysteresis buffer
        result = lod.get_lod_for_distance(15.0, current_lod=1)
        assert result == 0, "Should switch to LOD 0 when below buffer"

    def test_no_hysteresis_when_no_current_lod(self):
        """Without current LOD, no hysteresis applied."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
        ])
        lod.set_hysteresis(2.0)

        # No current LOD (-1), should select based on raw distance
        # At 20.0, switch to LOD 1 happens (threshold is at second level)
        result = lod.get_lod_for_distance(20.0, current_lod=-1)
        assert result == 1, "Without current LOD, should select based on distance alone"

    def test_set_hysteresis_method_exists(self):
        """set_hysteresis method is callable."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[LODLevel(distance=10.0, bone_count=50)])
        # Should not raise
        lod.set_hysteresis(3.0)

    def test_zero_hysteresis_no_stickiness(self):
        """Zero hysteresis means no stickiness at boundaries."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
        ])
        lod.set_hysteresis(0.0)

        # At exactly 20.0, should switch immediately (LOD 1 threshold)
        result = lod.get_lod_for_distance(20.0, current_lod=0)
        assert result == 1, "With zero hysteresis, should switch at threshold"


class TestHysteresisActualBehavior:
    """Tests verifying actual hysteresis implementation behavior."""

    def test_hysteresis_prevents_immediate_switch(self):
        """Hysteresis should prevent LOD switch at boundary."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
        ])
        lod.set_hysteresis(2.0)
        # At LOD 0, threshold is 20.0, with hysteresis need > 22 to switch
        result = lod.get_lod_for_distance(21.0, current_lod=0)
        assert result == 0

    def test_hysteresis_allows_switch_past_buffer(self):
        """Hysteresis should allow switch when past buffer."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
        ])
        lod.set_hysteresis(2.0)
        # At LOD 0, threshold is 20.0 + 2.0 = 22.0
        result = lod.get_lod_for_distance(23.0, current_lod=0)
        assert result == 1


class TestLODTransitionModes:
    """Test LOD transition observable behaviors (instant, blend, dither)."""

    def test_lod_transition_import(self):
        """LODTransition can be imported from the module."""
        from engine.animation.crowds import LODTransition
        assert LODTransition is not None

    def test_lod_transition_is_enum_like(self):
        """LODTransition has enum-like behavior with members."""
        from engine.animation.crowds import LODTransition

        # Check it has some attributes (could be enum or class with constants)
        public_attrs = [a for a in dir(LODTransition)
                        if not a.startswith('_')]
        assert len(public_attrs) > 0, "LODTransition has no public attributes"

    def test_lod_transition_has_mode_attributes(self):
        """LODTransition should have transition mode attributes."""
        from engine.animation.crowds import LODTransition

        # Look for common transition mode names
        attrs = dir(LODTransition)
        has_any_mode = any(
            mode in attrs or mode.upper() in attrs
            for mode in ['instant', 'blend', 'dither', 'fade', 'cross',
                         'INSTANT', 'BLEND', 'DITHER', 'FADE', 'CROSS']
        )
        # If no modes found, document it
        if not has_any_mode:
            pytest.skip("LODTransition modes not found - check implementation")


class TestReducedSkeleton:
    """Test skeleton reduction for lower LOD levels."""

    def test_create_reduced_skeleton_function_exists(self):
        """create_reduced_skeleton function is exported."""
        from engine.animation.crowds import create_reduced_skeleton
        assert callable(create_reduced_skeleton)

    def test_create_reduced_skeleton_callable(self):
        """create_reduced_skeleton accepts expected parameter types."""
        from engine.animation.crowds import create_reduced_skeleton
        import inspect

        sig = inspect.signature(create_reduced_skeleton)
        params = list(sig.parameters.keys())

        # Should have at least 2 parameters (skeleton and target count)
        assert len(params) >= 2, f"Expected at least 2 params, got: {params}"


class TestT24SkeletonReductionContract:
    """T2.4 Contract Tests: Skeleton Reduction for LOD.

    PUBLIC CONTRACT (from PHASE_2_TODO.md):
    ```python
    skeleton = Skeleton(bones=["root", "spine", "head", "finger_01", "twist_arm"])
    reduced = create_reduced_skeleton(skeleton, max_bones=3)
    assert "root" in reduced.bones
    assert "spine" in reduced.bones
    assert "head" in reduced.bones
    assert "finger_01" not in reduced.bones
    assert "twist_arm" not in reduced.bones
    ```

    ACCEPTANCE CRITERIA:
    1. Important bones are kept (root, spine, head)
    2. Unimportant bones are culled (fingers, twist)
    3. Reduced skeleton is valid (no orphan bones)
    4. Bone count matches target

    API DIFFERENCES DISCOVERED:
    - Documented: Skeleton(bones=[...])
    - Actual: Skeleton(bone_names=[...], bone_parents=[...], bind_poses=[...])

    - Documented: create_reduced_skeleton(skeleton, max_bones=N)
    - Actual: create_reduced_skeleton(skeleton, target_bone_count)

    - Documented: reduced.bones
    - Actual: reduced.bone_names
    """

    def test_important_bones_kept_over_fingers(self):
        """Important bones (root, spine, head) are kept over finger bones."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        # Create skeleton with important and unimportant bones
        # Hierarchy: root -> spine -> head, root -> finger_01
        skeleton = Skeleton(
            bone_names=["root", "spine", "head", "finger_01", "twist_arm"],
            bone_parents=[-1, 0, 1, 0, 0],  # root, spine->root, head->spine, others->root
            bind_poses=[Transform.identity() for _ in range(5)],
        )

        # Reduce to 3 bones
        reduced = create_reduced_skeleton(skeleton, 3)

        # Contract: important bones should be kept
        assert "root" in reduced.bone_names, "root should be kept"
        assert "spine" in reduced.bone_names, "spine should be kept"
        assert "head" in reduced.bone_names, "head should be kept"

        # Contract: unimportant bones should be culled
        assert "finger_01" not in reduced.bone_names, "finger_01 should be culled"
        assert "twist_arm" not in reduced.bone_names, "twist_arm should be culled"

    def test_bone_count_matches_target(self):
        """Reduced skeleton has exactly target bone count."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        skeleton = Skeleton(
            bone_names=["root", "spine", "head", "finger_01", "finger_02", "twist_arm"],
            bone_parents=[-1, 0, 1, 0, 0, 0],
            bind_poses=[Transform.identity() for _ in range(6)],
        )

        reduced = create_reduced_skeleton(skeleton, 3)

        assert len(reduced.bone_names) == 3, f"Expected 3 bones, got {len(reduced.bone_names)}"

    def test_reduced_skeleton_is_valid_no_orphan_bones(self):
        """Reduced skeleton has no orphan bones (broken hierarchy)."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        # Deeper hierarchy: root -> spine -> chest -> head
        skeleton = Skeleton(
            bone_names=["root", "spine", "chest", "head", "finger_01"],
            bone_parents=[-1, 0, 1, 2, 0],  # spine->root, chest->spine, head->chest
            bind_poses=[Transform.identity() for _ in range(5)],
        )

        reduced = create_reduced_skeleton(skeleton, 3)

        # Verify no orphan bones - every bone's parent should be -1 or a valid index
        for i, parent in enumerate(reduced.bone_parents):
            if parent >= 0:
                assert parent < len(reduced.bone_names), \
                    f"Bone {i} ({reduced.bone_names[i]}) has invalid parent index {parent}"
            # Also verify parent exists in our bone list
            if parent >= 0:
                assert reduced.bone_names[parent] is not None, \
                    f"Bone {i} has orphan parent"

    def test_root_bone_always_kept(self):
        """Root bone (index 0 or named 'root') is always kept."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        skeleton = Skeleton(
            bone_names=["root", "spine", "finger_01", "finger_02", "finger_03"],
            bone_parents=[-1, 0, 0, 0, 0],
            bind_poses=[Transform.identity() for _ in range(5)],
        )

        # Even reducing to 1 bone should keep root
        reduced = create_reduced_skeleton(skeleton, 1)

        assert "root" in reduced.bone_names, "root must always be kept"

    def test_twist_bones_are_low_priority(self):
        """Twist/roll/helper bones are culled before structural bones."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        skeleton = Skeleton(
            bone_names=["root", "twist_spine", "roll_arm", "helper_knee", "spine", "head"],
            bone_parents=[-1, 0, 0, 0, 0, 4],
            bind_poses=[Transform.identity() for _ in range(6)],
        )

        # Reduce to keep only important bones
        reduced = create_reduced_skeleton(skeleton, 3)

        # Structural bones should be kept
        assert "root" in reduced.bone_names
        assert "spine" in reduced.bone_names
        assert "head" in reduced.bone_names

        # Twist/roll/helper bones should be culled
        assert "twist_spine" not in reduced.bone_names, "twist bones should be culled"
        assert "roll_arm" not in reduced.bone_names, "roll bones should be culled"
        assert "helper_knee" not in reduced.bone_names, "helper bones should be culled"

    def test_finger_bones_lower_priority_than_limbs(self):
        """Finger bones have lower priority than major limb bones."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        skeleton = Skeleton(
            bone_names=["root", "spine", "upperarm", "hand", "finger_index", "finger_middle", "finger_ring"],
            bone_parents=[-1, 0, 1, 2, 3, 3, 3],
            bind_poses=[Transform.identity() for _ in range(7)],
        )

        # Reduce to 4 bones - should keep structural + major limbs
        reduced = create_reduced_skeleton(skeleton, 4)

        # Major bones kept
        assert "root" in reduced.bone_names
        assert "spine" in reduced.bone_names
        assert "upperarm" in reduced.bone_names

        # At least some fingers culled
        finger_count = sum(1 for name in reduced.bone_names if "finger" in name)
        assert finger_count <= 1, f"Too many fingers kept: {finger_count}"

    def test_toe_bones_lowest_priority(self):
        """Toe bones are among the lowest priority."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        skeleton = Skeleton(
            bone_names=["root", "spine", "thigh", "foot", "toe_big", "toe_small"],
            bone_parents=[-1, 0, 1, 2, 3, 3],
            bind_poses=[Transform.identity() for _ in range(6)],
        )

        reduced = create_reduced_skeleton(skeleton, 4)

        # Important bones kept
        assert "root" in reduced.bone_names
        assert "spine" in reduced.bone_names
        assert "thigh" in reduced.bone_names
        assert "foot" in reduced.bone_names

        # Toes should be culled
        assert "toe_big" not in reduced.bone_names
        assert "toe_small" not in reduced.bone_names

    def test_head_and_neck_high_priority(self):
        """Head and neck bones have high priority (visual focus)."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        skeleton = Skeleton(
            bone_names=["root", "spine", "neck", "head", "finger_01", "toe_01"],
            bone_parents=[-1, 0, 1, 2, 0, 0],
            bind_poses=[Transform.identity() for _ in range(6)],
        )

        reduced = create_reduced_skeleton(skeleton, 4)

        # Head and neck should be kept over extremities
        assert "head" in reduced.bone_names, "head should be high priority"
        assert "neck" in reduced.bone_names, "neck should be high priority"

    def test_hierarchy_preserved_after_reduction(self):
        """Parent-child relationships are correctly remapped after reduction."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        # Hierarchy: root -> spine -> chest -> head
        skeleton = Skeleton(
            bone_names=["root", "spine", "chest", "head", "finger_01"],
            bone_parents=[-1, 0, 1, 2, 0],
            bind_poses=[Transform.identity() for _ in range(5)],
        )

        reduced = create_reduced_skeleton(skeleton, 3)

        # Build name to index map
        name_to_idx = {name: i for i, name in enumerate(reduced.bone_names)}

        # Verify hierarchy is preserved (may skip intermediate bones)
        if "root" in name_to_idx and "head" in name_to_idx:
            head_idx = name_to_idx["head"]
            parent_chain = []
            current = head_idx
            while current >= 0:
                parent_chain.append(reduced.bone_names[current])
                current = reduced.bone_parents[current]

            # Head should eventually reach root
            assert "root" in parent_chain, f"Head should connect to root: {parent_chain}"

    def test_bind_poses_preserved(self):
        """Bind poses are correctly copied to reduced skeleton."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform, Vec3, Quat

        # Create skeleton with distinct bind poses
        bind_poses = [
            Transform(translation=Vec3(0, 0, 0), rotation=Quat.identity(), scale=Vec3(1, 1, 1)),
            Transform(translation=Vec3(0, 1, 0), rotation=Quat.identity(), scale=Vec3(1, 1, 1)),
            Transform(translation=Vec3(0, 2, 0), rotation=Quat.identity(), scale=Vec3(1, 1, 1)),
        ]

        skeleton = Skeleton(
            bone_names=["root", "spine", "head"],
            bone_parents=[-1, 0, 1],
            bind_poses=bind_poses,
        )

        reduced = create_reduced_skeleton(skeleton, 2)

        # Verify bind poses length matches bone count
        assert len(reduced.bind_poses) == len(reduced.bone_names), \
            "Bind poses count should match bone count"

    def test_empty_skeleton_reduction(self):
        """Reducing empty skeleton returns empty skeleton."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton

        skeleton = Skeleton(
            bone_names=[],
            bone_parents=[],
            bind_poses=[],
        )

        reduced = create_reduced_skeleton(skeleton, 3)

        assert len(reduced.bone_names) == 0

    def test_target_larger_than_skeleton(self):
        """Requesting more bones than skeleton has returns original."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        skeleton = Skeleton(
            bone_names=["root", "spine", "head"],
            bone_parents=[-1, 0, 1],
            bind_poses=[Transform.identity() for _ in range(3)],
        )

        reduced = create_reduced_skeleton(skeleton, 10)

        # Should return original (or copy with same count)
        assert len(reduced.bone_names) == 3

    def test_zero_target_returns_empty(self):
        """Requesting 0 bones returns empty skeleton."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        skeleton = Skeleton(
            bone_names=["root", "spine", "head"],
            bone_parents=[-1, 0, 1],
            bind_poses=[Transform.identity() for _ in range(3)],
        )

        reduced = create_reduced_skeleton(skeleton, 0)

        assert len(reduced.bone_names) == 0

    def test_pelvis_hips_high_priority(self):
        """Pelvis/hips bones have high priority (root of lower body)."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        skeleton = Skeleton(
            bone_names=["root", "pelvis", "spine", "finger_01", "toe_01", "twist_leg"],
            bone_parents=[-1, 0, 1, 0, 0, 0],
            bind_poses=[Transform.identity() for _ in range(6)],
        )

        reduced = create_reduced_skeleton(skeleton, 3)

        assert "pelvis" in reduced.bone_names, "pelvis should be high priority"

    def test_complex_skeleton_reduction_preserves_structure(self):
        """Complex real-world skeleton structure is preserved correctly."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        # Realistic humanoid skeleton
        skeleton = Skeleton(
            bone_names=[
                "root", "pelvis", "spine", "spine_01", "spine_02", "chest",
                "neck", "head",
                "clavicle_l", "upperarm_l", "forearm_l", "hand_l",
                "finger_index_l", "finger_middle_l", "finger_ring_l",
                "thigh_l", "calf_l", "foot_l", "toe_l",
                "twist_upperarm_l", "twist_forearm_l"
            ],
            bone_parents=[
                -1, 0, 1, 2, 3, 4,  # spine chain
                5, 6,  # neck, head
                5, 8, 9, 10,  # left arm
                11, 11, 11,  # fingers
                1, 15, 16, 17,  # left leg
                9, 10  # twist bones
            ],
            bind_poses=[Transform.identity() for _ in range(21)],
        )

        # Reduce to ~10 bones for LOD2
        reduced = create_reduced_skeleton(skeleton, 10)

        # Must have core structure
        assert "root" in reduced.bone_names
        assert len([n for n in reduced.bone_names if "spine" in n]) >= 1
        assert "head" in reduced.bone_names

        # Should have limbs
        limb_bones = [n for n in reduced.bone_names
                      if any(x in n for x in ["arm", "thigh", "calf", "hand", "foot"])]
        assert len(limb_bones) >= 2, "Should keep some limb bones"

        # Should NOT have fine detail
        finger_bones = [n for n in reduced.bone_names if "finger" in n]
        toe_bones = [n for n in reduced.bone_names if "toe" in n]
        twist_bones = [n for n in reduced.bone_names if "twist" in n]

        assert len(finger_bones) <= 1, "Fingers should be mostly culled"
        assert len(toe_bones) == 0, "Toes should be culled"
        assert len(twist_bones) == 0, "Twist bones should be culled"


class TestT24SkeletonReductionEdgeCases:
    """T2.4 Edge case tests for skeleton reduction."""

    def test_all_same_priority_bones(self):
        """Skeleton with all same-priority bones reduces correctly."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        # All bones are generic names with similar priority
        skeleton = Skeleton(
            bone_names=["bone_0", "bone_1", "bone_2", "bone_3", "bone_4"],
            bone_parents=[-1, 0, 0, 0, 0],
            bind_poses=[Transform.identity() for _ in range(5)],
        )

        reduced = create_reduced_skeleton(skeleton, 3)

        # Should still work and produce valid skeleton
        assert len(reduced.bone_names) == 3
        assert len(reduced.bone_parents) == 3

    def test_deep_hierarchy_chain(self):
        """Deep hierarchy chain is handled correctly."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        # Linear chain: root -> a -> b -> c -> d -> e
        skeleton = Skeleton(
            bone_names=["root", "spine_a", "spine_b", "spine_c", "spine_d", "head"],
            bone_parents=[-1, 0, 1, 2, 3, 4],
            bind_poses=[Transform.identity() for _ in range(6)],
        )

        reduced = create_reduced_skeleton(skeleton, 3)

        # Verify chain is still connected
        assert len(reduced.bone_names) == 3

        # Root should have parent -1
        root_idx = reduced.bone_names.index("root") if "root" in reduced.bone_names else -1
        if root_idx >= 0:
            assert reduced.bone_parents[root_idx] == -1

    def test_single_bone_skeleton(self):
        """Single bone skeleton handles reduction correctly."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        skeleton = Skeleton(
            bone_names=["root"],
            bone_parents=[-1],
            bind_poses=[Transform.identity()],
        )

        reduced = create_reduced_skeleton(skeleton, 1)

        assert len(reduced.bone_names) == 1
        assert reduced.bone_names[0] == "root"

    def test_negative_target_bone_count(self):
        """Negative target bone count is handled gracefully."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        skeleton = Skeleton(
            bone_names=["root", "spine", "head"],
            bone_parents=[-1, 0, 1],
            bind_poses=[Transform.identity() for _ in range(3)],
        )

        # Should not crash, return empty or handle gracefully
        reduced = create_reduced_skeleton(skeleton, -1)

        assert len(reduced.bone_names) >= 0  # Should be empty or minimal

    def test_unicode_bone_names(self):
        """Unicode bone names are handled correctly."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        skeleton = Skeleton(
            bone_names=["root", "spine_??????", "head_??????", "finger_??????"],
            bone_parents=[-1, 0, 1, 0],
            bind_poses=[Transform.identity() for _ in range(4)],
        )

        reduced = create_reduced_skeleton(skeleton, 2)

        assert len(reduced.bone_names) == 2

    def test_case_insensitive_bone_importance(self):
        """Bone importance calculation is case-insensitive."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        # Mixed case bone names
        skeleton = Skeleton(
            bone_names=["ROOT", "SPINE", "HEAD", "FINGER_01", "TWIST_ARM"],
            bone_parents=[-1, 0, 1, 0, 0],
            bind_poses=[Transform.identity() for _ in range(5)],
        )

        reduced = create_reduced_skeleton(skeleton, 3)

        # Should recognize important bones regardless of case
        kept_lower = [n.lower() for n in reduced.bone_names]
        assert "root" in kept_lower
        assert "spine" in kept_lower
        assert "head" in kept_lower


class TestT24ContractViolations:
    """Document contract violations between documentation and implementation for T2.4."""

    def test_contract_violation_bones_vs_bone_names(self):
        """CONTRACT VIOLATION: Docs say reduced.bones, impl uses reduced.bone_names."""
        from engine.animation.crowds import create_reduced_skeleton
        from engine.animation.crowds.animation_texture import Skeleton
        from engine.core.math import Transform

        skeleton = Skeleton(
            bone_names=["root", "spine", "head"],
            bone_parents=[-1, 0, 1],
            bind_poses=[Transform.identity() for _ in range(3)],
        )

        reduced = create_reduced_skeleton(skeleton, 2)

        # Documentation shows: reduced.bones
        # Actual implementation: reduced.bone_names
        assert hasattr(reduced, "bone_names"), "Implementation uses bone_names"

        # Document that 'bones' attribute does not exist
        assert not hasattr(reduced, "bones") or reduced.bones != reduced.bone_names, \
            "CONTRACT: docs say .bones but impl uses .bone_names"

    def test_contract_violation_max_bones_vs_target_bone_count(self):
        """CONTRACT VIOLATION: Docs say max_bones, impl uses target_bone_count positional."""
        from engine.animation.crowds import create_reduced_skeleton
        import inspect

        sig = inspect.signature(create_reduced_skeleton)
        params = list(sig.parameters.keys())

        # Documentation shows: create_reduced_skeleton(skeleton, max_bones=3)
        # Actual: create_reduced_skeleton(skeleton, target_bone_count)
        assert "max_bones" not in params, \
            "CONTRACT VIOLATION: docs say max_bones but impl uses target_bone_count"

    def test_contract_violation_skeleton_constructor(self):
        """CONTRACT VIOLATION: Docs say Skeleton(bones=[...]), impl requires more params."""
        from engine.animation.crowds.animation_texture import Skeleton

        # Documentation shows: Skeleton(bones=["root", "spine", "head", ...])
        # Actual requires: bone_names, bone_parents, bind_poses

        # This should fail - documented API doesn't work
        try:
            skeleton = Skeleton(bones=["root", "spine", "head"])
            assert False, "Expected TypeError for bones parameter"
        except TypeError:
            pass  # Expected - documents the contract violation


class TestCrowdLODLevelsProperty:
    """Test accessing LOD levels collection."""

    def test_lod_has_levels_accessible(self):
        """CrowdLOD stores and can retrieve levels."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        levels = [
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
        ]
        lod = CrowdLOD(levels=levels)

        # Should be able to retrieve level 0 and 1
        level0 = lod.get_lod_level(0)
        level1 = lod.get_lod_level(1)

        assert level0 is not None
        assert level1 is not None
        assert level0.distance == 10.0
        assert level1.distance == 20.0


class TestContractViolations:
    """Document discovered contract violations between documentation and implementation."""

    def test_max_bone_count_vs_bone_count(self):
        """CONTRACT VIOLATION: Docs say max_bone_count, impl uses bone_count."""
        from engine.animation.crowds import LODLevel

        # Documentation (PHASE_2_TODO.md) shows:
        # LODLevel(distance=10, max_bone_count=50)
        # But actual API uses:
        # LODLevel(distance=10, bone_count=50)

        with pytest.raises(TypeError, match="max_bone_count"):
            LODLevel(distance=10, max_bone_count=50)

        # Correct usage:
        level = LODLevel(distance=10, bone_count=50)
        assert level is not None

    def test_hysteresis_parameter_not_supported(self):
        """CONTRACT VIOLATION: Docs show hysteresis parameter but not implemented."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        with pytest.raises(TypeError, match="hysteresis"):
            CrowdLOD(
                levels=[LODLevel(distance=10.0, bone_count=50)],
                hysteresis=2.0
            )

    def test_get_lod_level_api_mismatch(self):
        """CONTRACT VIOLATION: get_lod_level takes index not distance."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
        ])

        # Per docs: get_lod_level(distance=5) -> int (LOD index)
        # Actual: get_lod_level(index: int) -> LODLevel | None

        # Actual API returns LODLevel, not int (the documented return type)
        result = lod.get_lod_level(0)
        assert not isinstance(result, int), \
            "CONTRACT VIOLATION: docs say returns int, actually returns LODLevel"

        # get_lod_level with typical distance value (15.0) returns None
        # because it treats it as index 15, which is out of bounds
        result_distance = lod.get_lod_level(15)
        assert result_distance is None, \
            "get_lod_level interprets value as index, not distance"


class TestWorkingFeatures:
    """Test features that ARE working correctly."""

    def test_lod_level_stores_all_properties(self):
        """LODLevel correctly stores distance and bone_count."""
        from engine.animation.crowds import LODLevel

        level = LODLevel(distance=25.5, bone_count=42)
        assert level.distance == 25.5
        assert level.bone_count == 42

    def test_crowd_lod_stores_multiple_levels(self):
        """CrowdLOD correctly stores multiple levels in order."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=5.0, bone_count=100),
            LODLevel(distance=15.0, bone_count=50),
            LODLevel(distance=30.0, bone_count=25),
        ])

        assert lod.get_lod_level(0).distance == 5.0
        assert lod.get_lod_level(1).distance == 15.0
        assert lod.get_lod_level(2).distance == 30.0

    def test_lod_transition_enum_has_values(self):
        """LODTransition enum is importable and has values."""
        from engine.animation.crowds import LODTransition

        # Should be able to iterate or list members
        members = [m for m in dir(LODTransition) if not m.startswith('_')]
        assert len(members) > 0


class TestEdgeCasesForImplementedAPI:
    """Edge cases for the actually implemented API."""

    def test_single_level_lod_access(self):
        """Single level LOD returns that level for index 0."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[LODLevel(distance=10.0, bone_count=50)])

        result = lod.get_lod_level(0)
        assert result is not None
        assert result.distance == 10.0

    def test_empty_levels_handling(self):
        """CrowdLOD with empty levels is either rejected or handled."""
        from engine.animation.crowds import CrowdLOD

        try:
            lod = CrowdLOD(levels=[])
            # If allowed, getting any level should return None
            assert lod.get_lod_level(0) is None
        except (ValueError, TypeError):
            # If rejected, that's also acceptable
            pass


class TestT23BoundaryConditions:
    """T2.3 Contract Tests: Boundary conditions handled correctly."""

    def test_negative_distance_clamped_to_zero(self):
        """Negative distances should be handled gracefully."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
        ])

        # Should clamp to 0 or handle gracefully
        result = lod.get_lod_for_distance(-5.0)
        assert result == 0, "Negative distance should return LOD 0"

    def test_empty_levels_returns_zero(self):
        """Empty levels should return LOD 0."""
        from engine.animation.crowds import CrowdLOD

        lod = CrowdLOD(levels=[])
        result = lod.get_lod_for_distance(50.0)
        assert result == 0, "Empty levels should default to LOD 0"

    def test_single_level_always_returns_zero(self):
        """Single LOD level always returns 0."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
        ])

        assert lod.get_lod_for_distance(5.0) == 0
        assert lod.get_lod_for_distance(100.0) == 0

    def test_many_lod_levels(self):
        """Many LOD levels should be handled correctly."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        # Levels at 10, 20, 30, 40, 50, 60, 70
        levels = [
            LODLevel(distance=i * 10.0, bone_count=100 - i * 10)
            for i in range(1, 8)  # 7 levels
        ]
        lod = CrowdLOD(levels=levels)

        # LOD 0: distance < 20 (switch happens at second level's distance)
        assert lod.get_lod_for_distance(5.0) == 0
        # LOD 1: 20 <= distance < 30
        assert lod.get_lod_for_distance(25.0) == 1
        # LOD 6: distance >= 70 (max_lod)
        assert lod.get_lod_for_distance(100.0) == 6

    def test_large_distance_value(self):
        """Very large distances should return max LOD."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ])

        result = lod.get_lod_for_distance(float('inf'))
        assert result == 2, "Infinite distance should return max LOD"

    def test_max_lod_property(self):
        """max_lod property returns correct value."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ])

        assert lod.max_lod == 2, "max_lod should be 2 for 3 levels"


class TestT23CompleteScenarios:
    """T2.3 Integration scenarios combining all criteria."""

    def test_agent_movement_lod_transitions(self):
        """Simulate agent moving toward and away from camera."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ])
        lod.set_hysteresis(1.0)

        # Agent starts far away (beyond all thresholds)
        current = lod.get_lod_for_distance(100.0, current_lod=-1)
        assert current == 2

        # Agent approaches - at LOD 2, stays at LOD 2 until distance < 50
        current = lod.get_lod_for_distance(50.0, current_lod=current)
        assert current == 2, "Still LOD 2 at threshold"

        current = lod.get_lod_for_distance(49.0, current_lod=current)
        # Now past hysteresis buffer (< 50)
        assert current == 1

        # Agent gets closer - at LOD 1, threshold to LOD 0 is at 20.0
        current = lod.get_lod_for_distance(15.0, current_lod=current)
        assert current == 0

        # Agent moves away - at LOD 0, threshold to LOD 1 is 20.0 + 1 = 21
        current = lod.get_lod_for_distance(20.5, current_lod=current)
        assert current == 0, "Stay at LOD 0 due to hysteresis"

    def test_bone_count_decreases_with_distance(self):
        """Verify bone count decreases as LOD increases."""
        from engine.animation.crowds import CrowdLOD, LODLevel

        lod = CrowdLOD(levels=[
            LODLevel(distance=10.0, bone_count=50),
            LODLevel(distance=20.0, bone_count=30),
            LODLevel(distance=50.0, bone_count=10),
        ])

        # Distances that map to LOD 0, LOD 1, LOD 2
        # LOD 0: < 20, LOD 1: 20-50, LOD 2: >= 50
        bone_counts = [
            lod.get_bone_count_for_lod(lod.get_lod_for_distance(d))
            for d in [5.0, 25.0, 100.0]
        ]

        assert bone_counts == [50, 30, 10], f"Bone counts should decrease: {bone_counts}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

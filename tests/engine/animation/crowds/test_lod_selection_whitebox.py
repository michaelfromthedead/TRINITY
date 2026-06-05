"""Whitebox tests for T2.3 LOD Level Selection.

Tests internal implementation of LOD level selection with focus on:
- Correct LOD level selected for distance ranges
- Hysteresis prevents rapid switching
- Boundary conditions handled correctly
- Very far agents use lowest LOD

Task: T2.3 LOD Level Selection

Note: Many of these criteria are also covered in test_crowd_lod_whitebox.py (T1.5).
This file adds specific targeted tests for T2.3 acceptance criteria.
"""

from __future__ import annotations

import pytest
from engine.core.math import Vec3, Transform
from engine.animation.crowds.animation_texture import Skeleton
from engine.animation.crowds.crowd_lod import (
    CrowdLOD,
    LODLevel,
    LODTransitionMode,
)
from engine.animation.crowds.crowd_renderer import CrowdInstance, CrowdRenderer
from engine.animation.config import CROWD_LOD_CONFIG


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def humanoid_skeleton() -> Skeleton:
    """Create a basic humanoid skeleton for testing."""
    bone_names = ["root", "pelvis", "spine", "chest", "neck", "head",
                  "shoulder_l", "upperarm_l", "forearm_l", "hand_l",
                  "shoulder_r", "upperarm_r", "forearm_r", "hand_r",
                  "thigh_l", "calf_l", "foot_l",
                  "thigh_r", "calf_r", "foot_r"]
    bone_parents = [-1, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 1, 14, 15, 1, 17, 18]
    bind_poses = [Transform.identity() for _ in bone_names]
    return Skeleton(bone_names=bone_names, bone_parents=bone_parents, bind_poses=bind_poses)


@pytest.fixture
def four_lod_levels() -> list[LODLevel]:
    """Create 4 LOD levels with standard distance thresholds."""
    return [
        LODLevel(distance=0.0, bone_count=64, update_rate=1.0, shadow_enabled=True),
        LODLevel(distance=10.0, bone_count=32, update_rate=1.0, shadow_enabled=True),
        LODLevel(distance=25.0, bone_count=16, update_rate=0.5, shadow_enabled=False),
        LODLevel(distance=50.0, bone_count=8, update_rate=0.25, shadow_enabled=False),
    ]


@pytest.fixture
def lod_system(humanoid_skeleton: Skeleton, four_lod_levels: list[LODLevel]) -> CrowdLOD:
    """Create configured LOD system."""
    return CrowdLOD(skeleton=humanoid_skeleton, levels=four_lod_levels)


# =============================================================================
# Test: Correct LOD Level Selected for Distance Ranges (T2.3 Criterion 1)
# =============================================================================


class TestLODDistanceRanges:
    """Tests for correct LOD selection across all distance ranges."""

    def test_lod0_close_range(self, lod_system: CrowdLOD):
        """LOD 0 selected for close distances (0-10m)."""
        close_distances = [0.0, 1.0, 5.0, 9.9]
        for dist in close_distances:
            lod = lod_system.get_lod_for_distance(dist)
            assert lod == 0, f"Distance {dist} should select LOD 0, got LOD {lod}"

    def test_lod1_mid_close_range(self, lod_system: CrowdLOD):
        """LOD 1 selected for mid-close distances (10-25m)."""
        mid_distances = [10.0, 15.0, 20.0, 24.9]
        for dist in mid_distances:
            lod = lod_system.get_lod_for_distance(dist)
            assert lod == 1, f"Distance {dist} should select LOD 1, got LOD {lod}"

    def test_lod2_mid_far_range(self, lod_system: CrowdLOD):
        """LOD 2 selected for mid-far distances (25-50m)."""
        mid_far_distances = [25.0, 30.0, 40.0, 49.9]
        for dist in mid_far_distances:
            lod = lod_system.get_lod_for_distance(dist)
            assert lod == 2, f"Distance {dist} should select LOD 2, got LOD {lod}"

    def test_lod3_far_range(self, lod_system: CrowdLOD):
        """LOD 3 selected for far distances (50m+)."""
        far_distances = [50.0, 75.0, 100.0, 200.0]
        for dist in far_distances:
            lod = lod_system.get_lod_for_distance(dist)
            assert lod == 3, f"Distance {dist} should select LOD 3, got LOD {lod}"

    def test_lod_selection_at_exact_boundaries(self, lod_system: CrowdLOD):
        """Test LOD selection exactly at each boundary threshold."""
        # At threshold, should select the next LOD (boundary belongs to farther LOD)
        boundary_tests = [
            (10.0, 1),  # At LOD1 boundary -> LOD1
            (25.0, 2),  # At LOD2 boundary -> LOD2
            (50.0, 3),  # At LOD3 boundary -> LOD3
        ]
        for dist, expected_lod in boundary_tests:
            lod = lod_system.get_lod_for_distance(dist)
            assert lod == expected_lod, f"Distance {dist} should select LOD {expected_lod}, got {lod}"

    def test_lod_selection_just_below_boundaries(self, lod_system: CrowdLOD):
        """Test LOD selection just below each boundary threshold."""
        epsilon = 0.001
        boundary_tests = [
            (10.0 - epsilon, 0),  # Just below LOD1 boundary -> LOD0
            (25.0 - epsilon, 1),  # Just below LOD2 boundary -> LOD1
            (50.0 - epsilon, 2),  # Just below LOD3 boundary -> LOD2
        ]
        for dist, expected_lod in boundary_tests:
            lod = lod_system.get_lod_for_distance(dist)
            assert lod == expected_lod, f"Distance {dist} should select LOD {expected_lod}, got {lod}"

    def test_lod_midpoint_selection(self, lod_system: CrowdLOD):
        """Test LOD at midpoint of each range."""
        # Midpoints of each LOD range
        midpoint_tests = [
            (5.0, 0),    # Midpoint of 0-10
            (17.5, 1),   # Midpoint of 10-25
            (37.5, 2),   # Midpoint of 25-50
            (75.0, 3),   # Beyond 50
        ]
        for dist, expected_lod in midpoint_tests:
            lod = lod_system.get_lod_for_distance(dist)
            assert lod == expected_lod, f"Distance {dist} should select LOD {expected_lod}, got {lod}"


# =============================================================================
# Test: Hysteresis Prevents Rapid Switching (T2.3 Criterion 2)
# =============================================================================


class TestHysteresisPreventsSwitching:
    """Tests that hysteresis prevents rapid LOD switching (flickering)."""

    def test_hysteresis_no_switch_within_band(self, lod_system: CrowdLOD):
        """Hysteresis prevents switch when distance oscillates within hysteresis band."""
        lod_system.set_hysteresis(3.0)
        current_lod = 1  # Currently at LOD 1 (10-25m range)

        # Oscillate around the LOD2 boundary (25m) within hysteresis band
        # With hysteresis=3, threshold becomes 22-28
        oscillating_distances = [23.0, 27.0, 24.0, 26.0, 25.0]
        for dist in oscillating_distances:
            new_lod = lod_system.get_lod_for_distance(dist, current_lod)
            assert new_lod == current_lod, f"LOD should stay at {current_lod} for dist {dist}"

    def test_hysteresis_allows_switch_outside_band(self, lod_system: CrowdLOD):
        """Hysteresis allows switch when distance exceeds hysteresis band."""
        lod_system.set_hysteresis(3.0)
        current_lod = 1

        # Beyond hysteresis band (25+3=28), should switch
        lod = lod_system.get_lod_for_distance(29.0, current_lod)
        assert lod == 2, "Should switch to LOD 2 when beyond hysteresis band"

        # Within band but at new LOD, should stay
        lod = lod_system.get_lod_for_distance(26.0, current_lod=2)
        assert lod == 2, "Should stay at LOD 2 within hysteresis band"

    def test_hysteresis_bidirectional(self, lod_system: CrowdLOD):
        """Hysteresis works in both directions (upgrading and downgrading)."""
        lod_system.set_hysteresis(2.0)

        # Upgrading (farther to closer): threshold shifts inward
        # At LOD 2, threshold for LOD1 is 10, with hysteresis becomes 10-2=8
        lod = lod_system.get_lod_for_distance(9.0, current_lod=2)
        assert lod >= 1, "Should stay at LOD 1+ when within hysteresis (upgrading)"

        lod = lod_system.get_lod_for_distance(7.0, current_lod=2)
        assert lod == 0, "Should upgrade to LOD 0 when below shifted threshold"

    def test_hysteresis_zero_disables(self, lod_system: CrowdLOD):
        """Zero hysteresis disables the feature."""
        lod_system.set_hysteresis(0.0)
        current_lod = 1

        # Should switch immediately at boundary
        lod = lod_system.get_lod_for_distance(25.0, current_lod)
        assert lod == 2, "Should switch immediately with zero hysteresis"

    def test_hysteresis_config_default(self, lod_system: CrowdLOD):
        """Default hysteresis matches config value."""
        assert lod_system._hysteresis == CROWD_LOD_CONFIG.DEFAULT_HYSTERESIS

    def test_hysteresis_flickering_simulation(self, lod_system: CrowdLOD):
        """Simulate camera jitter and verify no flickering."""
        lod_system.set_hysteresis(2.0)
        current_lod = 1

        # Simulate 20 frames of camera jitter around boundary
        jitter_distances = [
            24.5, 25.5, 24.8, 25.2, 24.6, 25.4, 24.9, 25.1,
            24.7, 25.3, 24.5, 25.5, 24.8, 25.2, 24.6, 25.4,
            24.9, 25.1, 24.7, 25.3
        ]

        lod_switches = 0
        for dist in jitter_distances:
            new_lod = lod_system.get_lod_for_distance(dist, current_lod)
            if new_lod != current_lod:
                lod_switches += 1
                current_lod = new_lod

        # With hysteresis=2 and jitter range 24.5-25.5, no switches should occur
        # (threshold is 25+/-2 = 23-27)
        assert lod_switches == 0, f"Expected 0 LOD switches, got {lod_switches}"


# =============================================================================
# Test: Boundary Conditions Handled Correctly (T2.3 Criterion 3)
# =============================================================================


class TestBoundaryConditions:
    """Tests for correct handling of boundary and edge conditions."""

    def test_zero_distance(self, lod_system: CrowdLOD):
        """Zero distance selects highest detail LOD."""
        lod = lod_system.get_lod_for_distance(0.0)
        assert lod == 0

    def test_negative_distance_clamped(self, lod_system: CrowdLOD):
        """Negative distance is clamped to zero."""
        for neg_dist in [-1.0, -10.0, -100.0]:
            lod = lod_system.get_lod_for_distance(neg_dist)
            assert lod == 0, f"Negative distance {neg_dist} should clamp to LOD 0"

    def test_very_small_positive_distance(self, lod_system: CrowdLOD):
        """Very small positive distance selects LOD 0."""
        small_distances = [0.001, 0.0001, 1e-10]
        for dist in small_distances:
            lod = lod_system.get_lod_for_distance(dist)
            assert lod == 0, f"Small distance {dist} should select LOD 0"

    def test_epsilon_below_threshold(self, lod_system: CrowdLOD):
        """Distance epsilon below threshold stays at lower LOD."""
        epsilon = 1e-6
        lod = lod_system.get_lod_for_distance(10.0 - epsilon)
        assert lod == 0, "Epsilon below 10.0 should select LOD 0"

    def test_empty_lod_levels_returns_zero(self):
        """Empty LOD levels list returns LOD 0."""
        lod_system = CrowdLOD()
        lod = lod_system.get_lod_for_distance(100.0)
        assert lod == 0

    def test_single_lod_level(self, humanoid_skeleton: Skeleton):
        """Single LOD level always returns 0."""
        lod_system = CrowdLOD(
            skeleton=humanoid_skeleton,
            levels=[LODLevel(distance=0.0, bone_count=64)]
        )
        for dist in [0.0, 10.0, 100.0, 1000.0]:
            lod = lod_system.get_lod_for_distance(dist)
            assert lod == 0, f"Single LOD should always return 0, got {lod} for dist {dist}"

    def test_max_lod_property(self, lod_system: CrowdLOD):
        """max_lod property returns correct maximum index."""
        assert lod_system.max_lod == 3  # 4 levels, indices 0-3


# =============================================================================
# Test: Very Far Agents Use Lowest LOD (T2.3 Criterion 4)
# =============================================================================


class TestVeryFarAgentsLowestLOD:
    """Tests that very far agents always use the lowest LOD (highest index)."""

    def test_far_distance_max_lod(self, lod_system: CrowdLOD):
        """Far distances return maximum LOD."""
        far_distances = [100.0, 500.0, 1000.0, 10000.0]
        for dist in far_distances:
            lod = lod_system.get_lod_for_distance(dist)
            assert lod == lod_system.max_lod, f"Distance {dist} should use max LOD"

    def test_infinity_distance(self, lod_system: CrowdLOD):
        """Infinite distance returns maximum LOD."""
        lod = lod_system.get_lod_for_distance(float('inf'))
        assert lod == lod_system.max_lod

    def test_very_large_distance(self, lod_system: CrowdLOD):
        """Very large distances (1e10) return maximum LOD."""
        lod = lod_system.get_lod_for_distance(1e10)
        assert lod == lod_system.max_lod

    def test_far_agents_in_renderer(
        self, humanoid_skeleton: Skeleton, four_lod_levels: list[LODLevel]
    ):
        """Far agents in renderer get lowest LOD assigned."""
        lod_system = CrowdLOD(skeleton=humanoid_skeleton, levels=four_lod_levels)
        renderer = CrowdRenderer()

        # Add agents at very far distances
        far_positions = [
            Vec3(200, 0, 0),
            Vec3(0, 500, 0),
            Vec3(0, 0, 1000),
            Vec3(5000, 5000, 5000),
        ]

        for pos in far_positions:
            instance = CrowdInstance(position=pos)
            renderer.add_instance(instance, mesh_id=1, material_id=1)

        # Update LOD levels
        renderer.update_lod_levels_from_system(Vec3.zero(), lod_system)

        # All should be at max LOD
        batch = renderer.get_batch(1, 1)
        for inst in batch.instances:
            assert inst.lod_level == lod_system.max_lod, (
                f"Far agent at {inst.position} should be at max LOD"
            )

    def test_cull_distance_beyond_max_lod(self, lod_system: CrowdLOD):
        """Agents beyond cull distance still get max LOD (not culled here)."""
        # LOD system doesn't cull, just assigns LOD
        cull_dist = CROWD_LOD_CONFIG.DEFAULT_CULL_DISTANCE
        lod = lod_system.get_lod_for_distance(cull_dist + 100)
        assert lod == lod_system.max_lod


# =============================================================================
# Test: Integration with CrowdRenderer
# =============================================================================


class TestRendererLODIntegration:
    """Integration tests for LOD selection with CrowdRenderer."""

    def test_multiple_agents_different_distances(
        self, humanoid_skeleton: Skeleton, four_lod_levels: list[LODLevel]
    ):
        """Multiple agents at different distances get correct LODs."""
        lod_system = CrowdLOD(skeleton=humanoid_skeleton, levels=four_lod_levels)
        renderer = CrowdRenderer()

        # Add agents at different distances representing each LOD range
        agent_distances = [
            (Vec3(5, 0, 0), 0),    # 5m -> LOD 0
            (Vec3(15, 0, 0), 1),   # 15m -> LOD 1
            (Vec3(35, 0, 0), 2),   # 35m -> LOD 2
            (Vec3(100, 0, 0), 3),  # 100m -> LOD 3
        ]

        for pos, _ in agent_distances:
            instance = CrowdInstance(position=pos)
            renderer.add_instance(instance, mesh_id=1, material_id=1)

        renderer.update_lod_levels_from_system(Vec3.zero(), lod_system)

        batch = renderer.get_batch(1, 1)
        for i, (_, expected_lod) in enumerate(agent_distances):
            actual_lod = batch.instances[i].lod_level
            assert actual_lod == expected_lod, (
                f"Agent {i} expected LOD {expected_lod}, got {actual_lod}"
            )

    def test_lod_update_with_camera_movement(
        self, humanoid_skeleton: Skeleton, four_lod_levels: list[LODLevel]
    ):
        """LOD updates correctly when camera moves."""
        lod_system = CrowdLOD(skeleton=humanoid_skeleton, levels=four_lod_levels)
        renderer = CrowdRenderer()

        # Agent at fixed position
        agent_pos = Vec3(30, 0, 0)  # 30m from origin
        instance = CrowdInstance(position=agent_pos)
        renderer.add_instance(instance, mesh_id=1, material_id=1)

        # Camera at origin -> 30m distance -> LOD 2
        renderer.update_lod_levels_from_system(Vec3.zero(), lod_system)
        batch = renderer.get_batch(1, 1)
        assert batch.instances[0].lod_level == 2

        # Camera moves closer -> 10m distance -> LOD 1
        renderer.update_lod_levels_from_system(Vec3(20, 0, 0), lod_system)
        assert batch.instances[0].lod_level == 1

        # Camera moves even closer -> 5m distance -> LOD 0
        renderer.update_lod_levels_from_system(Vec3(25, 0, 0), lod_system)
        assert batch.instances[0].lod_level == 0

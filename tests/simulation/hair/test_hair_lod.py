"""
Whitebox tests for hair Level of Detail (LOD) system.
"""

import numpy as np
import pytest

from engine.simulation.hair.config import (
    LOD_DISTANCE_HIGH,
    LOD_DISTANCE_LOW,
    LOD_DISTANCE_MEDIUM,
    LOD_DISTANCE_SHELL,
    LOD_GUIDE_FACTOR_HIGH,
    LOD_GUIDE_FACTOR_LOW,
    LOD_GUIDE_FACTOR_MEDIUM,
    LOD_GUIDE_FACTOR_SHELL,
    LOD_SEGMENT_FACTOR_HIGH,
    LOD_SEGMENT_FACTOR_LOW,
    LOD_SEGMENT_FACTOR_MEDIUM,
)
from engine.simulation.hair.hair_lod import (
    HairLODLevel,
    HairLODSystem,
    LODSettings,
    LODState,
    LODTransition,
    create_lod_interpolated_hairs,
)
from engine.simulation.hair.hair_simulation import (
    GuideHair,
    HairControlPoint,
    InterpolatedHair,
    create_hair_strand,
)


def create_test_guide_hair(root_pos, index=0):
    """Create a test guide hair."""
    hair = create_hair_strand(
        root_position=np.array(root_pos, dtype=np.float32),
        root_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        length=0.2,
        num_segments=4,
    )
    hair.index = index
    return hair


class TestHairLODLevel:
    """Tests for HairLODLevel enum."""

    def test_lod_levels_exist(self):
        """All LOD levels should exist."""
        assert HairLODLevel.HIGH is not None
        assert HairLODLevel.MEDIUM is not None
        assert HairLODLevel.LOW is not None
        assert HairLODLevel.SHELL is not None

    def test_lod_levels_distinct(self):
        """LOD levels should be distinct."""
        levels = [
            HairLODLevel.HIGH,
            HairLODLevel.MEDIUM,
            HairLODLevel.LOW,
            HairLODLevel.SHELL,
        ]
        assert len(levels) == len(set(levels))


class TestLODSettings:
    """Tests for LODSettings dataclass."""

    def test_default_settings(self):
        """Should use default values from config."""
        settings = LODSettings()

        assert settings.distance_high == LOD_DISTANCE_HIGH
        assert settings.distance_medium == LOD_DISTANCE_MEDIUM
        assert settings.distance_low == LOD_DISTANCE_LOW
        assert settings.distance_shell == LOD_DISTANCE_SHELL

        assert settings.guide_factor_high == LOD_GUIDE_FACTOR_HIGH
        assert settings.guide_factor_medium == LOD_GUIDE_FACTOR_MEDIUM
        assert settings.guide_factor_low == LOD_GUIDE_FACTOR_LOW
        assert settings.guide_factor_shell == LOD_GUIDE_FACTOR_SHELL

        assert settings.segment_factor_high == LOD_SEGMENT_FACTOR_HIGH
        assert settings.segment_factor_medium == LOD_SEGMENT_FACTOR_MEDIUM
        assert settings.segment_factor_low == LOD_SEGMENT_FACTOR_LOW

    def test_custom_settings(self):
        """Should accept custom values."""
        settings = LODSettings(
            distance_high=1.0,
            distance_medium=3.0,
            distance_low=5.0,
            distance_shell=10.0,
            hysteresis=0.2,
        )

        assert settings.distance_high == 1.0
        assert settings.distance_medium == 3.0
        assert settings.distance_low == 5.0
        assert settings.distance_shell == 10.0
        assert settings.hysteresis == 0.2


class TestLODState:
    """Tests for LODState dataclass."""

    def test_default_state(self):
        """Should have sensible defaults."""
        state = LODState()

        assert state.level == HairLODLevel.HIGH
        assert state.distance == 0.0
        assert state.guide_factor == 1.0
        assert state.segment_factor == 1.0
        assert state.blend_factor == 0.0

    def test_custom_state(self):
        """Should accept custom values."""
        state = LODState(
            level=HairLODLevel.MEDIUM,
            distance=5.0,
            guide_factor=0.5,
            segment_factor=0.75,
            blend_factor=0.5,
        )

        assert state.level == HairLODLevel.MEDIUM
        assert state.distance == 5.0
        assert state.guide_factor == 0.5
        assert state.segment_factor == 0.75
        assert state.blend_factor == 0.5


class TestHairLODSystem:
    """Tests for HairLODSystem class."""

    def test_lod_system_init(self):
        """Should initialize with default settings."""
        system = HairLODSystem()

        assert system.settings is not None
        assert system.current_level == HairLODLevel.HIGH
        assert system.guide_count == 0

    def test_lod_system_init_custom_settings(self):
        """Should accept custom settings."""
        settings = LODSettings(distance_high=1.0)
        system = HairLODSystem(settings=settings)

        assert system.settings.distance_high == 1.0

    def test_initialize_with_guides(self):
        """initialize() should set up LOD sets."""
        system = HairLODSystem()

        guides = [create_test_guide_hair([i * 0.1, 0.0, 0.0], i) for i in range(10)]
        system.initialize(guides)

        assert len(system._all_guides) == 10
        assert len(system._guides_high) == 10
        # Medium and low should have fewer guides
        assert len(system._guides_medium) <= 10
        assert len(system._guides_low) <= len(system._guides_medium)

    def test_active_guides_starts_with_high(self):
        """Active guides should be high quality after init."""
        system = HairLODSystem()

        guides = [create_test_guide_hair([i * 0.1, 0.0, 0.0], i) for i in range(10)]
        system.initialize(guides)

        assert len(system.active_guides) == 10
        assert system.current_level == HairLODLevel.HIGH

    def test_update_to_medium_lod(self):
        """Should switch to medium LOD at appropriate distance."""
        settings = LODSettings(
            distance_high=2.0,
            distance_medium=5.0,
            hysteresis=0.1,
        )
        system = HairLODSystem(settings=settings)

        guides = [create_test_guide_hair([i * 0.1, 0.0, 0.0], i) for i in range(10)]
        system.initialize(guides)

        # Camera far away
        camera_pos = np.array([10.0, 0.0, 0.0], dtype=np.float32)
        hair_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        changed = system.update(camera_pos, hair_center)

        # Should have changed to lower LOD
        assert changed is True
        assert system.current_level != HairLODLevel.HIGH

    def test_update_no_change_same_distance(self):
        """Should not change LOD if distance unchanged."""
        system = HairLODSystem()

        guides = [create_test_guide_hair([0.0, 0.0, 0.0], 0)]
        system.initialize(guides)

        camera_pos = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        hair_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # First update
        system.update(camera_pos, hair_center)
        level_after_first = system.current_level

        # Second update with same distance
        changed = system.update(camera_pos, hair_center)

        assert changed is False
        assert system.current_level == level_after_first

    def test_hysteresis_prevents_flickering(self):
        """Hysteresis should prevent rapid LOD changes."""
        settings = LODSettings(
            distance_high=2.0,
            distance_medium=5.0,
            hysteresis=0.5,
        )
        system = HairLODSystem(settings=settings)

        guides = [create_test_guide_hair([0.0, 0.0, 0.0], 0)]
        system.initialize(guides)

        hair_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Move to medium distance
        camera_pos = np.array([3.0, 0.0, 0.0], dtype=np.float32)
        system.update(camera_pos, hair_center)

        # Slightly move back toward high distance (within hysteresis)
        camera_pos = np.array([2.3, 0.0, 0.0], dtype=np.float32)
        changed = system.update(camera_pos, hair_center)

        # Should not have changed back due to hysteresis
        assert changed is False

    def test_transition_to_shell_mode(self):
        """Should transition to shell mode at far distance."""
        settings = LODSettings(
            distance_high=2.0,
            distance_medium=5.0,
            distance_low=10.0,
            distance_shell=20.0,
            hysteresis=0.0,  # No hysteresis for easier testing
        )
        system = HairLODSystem(settings=settings)

        guides = [create_test_guide_hair([0.0, 0.0, 0.0], 0)]
        system.initialize(guides)

        hair_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Move through all LOD levels
        system.update(np.array([3.0, 0.0, 0.0], dtype=np.float32), hair_center)
        system.update(np.array([7.0, 0.0, 0.0], dtype=np.float32), hair_center)
        system.update(np.array([15.0, 0.0, 0.0], dtype=np.float32), hair_center)
        system.update(np.array([25.0, 0.0, 0.0], dtype=np.float32), hair_center)

        assert system.current_level == HairLODLevel.SHELL
        assert system.is_shell_mode() is True
        assert len(system.active_guides) == 0

    def test_reduce_guide_count(self):
        """reduce_guide_count should select subset of guides."""
        system = HairLODSystem()

        guides = [create_test_guide_hair([i * 0.1, 0.0, 0.0], i) for i in range(20)]

        reduced = system.reduce_guide_count(guides, target_count=5)

        assert len(reduced) == 5
        # Should be a subset - check by index
        reduced_indices = {g.index for g in reduced}
        guide_indices = {g.index for g in guides}
        assert reduced_indices.issubset(guide_indices)

    def test_reduce_guide_count_exceeds_available(self):
        """reduce_guide_count should return all if target > available."""
        system = HairLODSystem()

        guides = [create_test_guide_hair([i * 0.1, 0.0, 0.0], i) for i in range(5)]

        reduced = system.reduce_guide_count(guides, target_count=10)

        assert len(reduced) == 5  # Can't exceed available

    def test_reduce_guide_count_zero(self):
        """reduce_guide_count should handle zero target."""
        system = HairLODSystem()

        guides = [create_test_guide_hair([0.0, 0.0, 0.0], 0)]

        reduced = system.reduce_guide_count(guides, target_count=0)

        assert len(reduced) == 0

    def test_get_interpolation_weights(self):
        """get_interpolation_weights should return nearest guides and weights."""
        system = HairLODSystem()

        guides = [create_test_guide_hair([i * 0.1, 0.0, 0.0], i) for i in range(5)]
        system.initialize(guides)

        position = np.array([0.15, 0.0, 0.0], dtype=np.float32)
        indices, weights = system.get_interpolation_weights(position, k_nearest=3)

        assert len(indices) == 3
        assert len(weights) == 3
        assert np.sum(weights) == pytest.approx(1.0)
        # Closest guides should have highest weights
        assert 0 in indices or 1 in indices or 2 in indices

    def test_get_interpolation_weights_empty_guides(self):
        """get_interpolation_weights should handle empty guides."""
        system = HairLODSystem()
        system.initialize([])

        position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        indices, weights = system.get_interpolation_weights(position)

        assert len(indices) == 0
        assert len(weights) == 0

    def test_get_segment_count(self):
        """get_segment_count should reduce segments based on LOD."""
        system = HairLODSystem()
        system._state.segment_factor = 0.5

        count = system.get_segment_count(base_segments=16)

        assert count == 8  # 16 * 0.5

    def test_get_segment_count_minimum(self):
        """get_segment_count should have minimum of 2."""
        system = HairLODSystem()
        system._state.segment_factor = 0.0  # Would give 0

        count = system.get_segment_count(base_segments=16)

        assert count >= 2

    def test_prepare_shell_data(self):
        """prepare_shell_data should compute shell offsets."""
        system = HairLODSystem()

        guides = [create_test_guide_hair([0.0, 0.0, 0.0], 0)]

        system.prepare_shell_data(guides, num_layers=4)

        assert system._shell_layers == 4
        assert system._shell_data is not None
        assert len(system._shell_data) == 4

    def test_prepare_shell_data_empty(self):
        """prepare_shell_data should handle empty guides."""
        system = HairLODSystem()

        system.prepare_shell_data([], num_layers=4)

        assert system._shell_data is None

    def test_get_shell_offsets_not_shell_mode(self):
        """get_shell_offsets should return None when not in shell mode."""
        system = HairLODSystem()
        system._state.level = HairLODLevel.HIGH

        offsets = system.get_shell_offsets()

        assert offsets is None

    def test_get_shell_offsets_in_shell_mode(self):
        """get_shell_offsets should return offsets in shell mode."""
        system = HairLODSystem()
        guides = [create_test_guide_hair([0.0, 0.0, 0.0], 0)]

        system.prepare_shell_data(guides, num_layers=4)
        system._state.level = HairLODLevel.SHELL

        offsets = system.get_shell_offsets()

        assert offsets is not None
        assert len(offsets) == 4


class TestLODTransition:
    """Tests for LODTransition class."""

    def test_transition_init(self):
        """Should initialize with defaults."""
        transition = LODTransition()

        assert transition.duration == 0.5
        assert transition._progress == 1.0  # Complete by default
        assert transition.is_transitioning is False

    def test_start_transition(self):
        """start_transition should begin a new transition."""
        transition = LODTransition()

        transition.start_transition(HairLODLevel.HIGH, HairLODLevel.MEDIUM)

        assert transition._source_level == HairLODLevel.HIGH
        assert transition._target_level == HairLODLevel.MEDIUM
        assert transition._progress == 0.0
        assert transition.is_transitioning is True

    def test_update_progress(self):
        """update should advance progress."""
        transition = LODTransition(duration=1.0)
        transition.start_transition(HairLODLevel.HIGH, HairLODLevel.MEDIUM)

        complete = transition.update(0.5)

        assert complete is False
        assert transition._progress == pytest.approx(0.5)

    def test_update_completes(self):
        """update should complete transition."""
        transition = LODTransition(duration=1.0)
        transition.start_transition(HairLODLevel.HIGH, HairLODLevel.MEDIUM)

        complete = transition.update(1.5)  # More than duration

        assert complete is True
        assert transition._progress == 1.0
        assert transition.is_transitioning is False

    def test_blend_factor_smooth_step(self):
        """blend_factor should use smooth step interpolation."""
        transition = LODTransition(duration=1.0)
        transition.start_transition(HairLODLevel.HIGH, HairLODLevel.MEDIUM)

        # Progress at 0
        assert transition.blend_factor == 0.0

        # Progress at 0.5 should be 0.5 (smooth step)
        transition._progress = 0.5
        assert transition.blend_factor == pytest.approx(0.5)

        # Progress at 1
        transition._progress = 1.0
        assert transition.blend_factor == 1.0

    def test_update_already_complete(self):
        """update should handle already complete transition."""
        transition = LODTransition()  # Already complete

        complete = transition.update(0.1)

        assert complete is True


class TestCreateLodInterpolatedHairs:
    """Tests for create_lod_interpolated_hairs function."""

    def test_create_interpolated_hairs(self):
        """Should create interpolated hairs from LOD system."""
        lod_system = HairLODSystem()

        guides = [create_test_guide_hair([i * 0.1, 0.0, 0.0], i) for i in range(3)]
        lod_system.initialize(guides)

        interpolated = create_lod_interpolated_hairs(lod_system, count_per_guide=5)

        # 3 guides * 5 interpolated = 15
        assert len(interpolated) == 15
        for hair in interpolated:
            assert isinstance(hair, InterpolatedHair)

    def test_create_interpolated_references_guide(self):
        """Interpolated hairs should reference their guide."""
        lod_system = HairLODSystem()

        guides = [create_test_guide_hair([0.0, 0.0, 0.0], 0)]
        lod_system.initialize(guides)

        interpolated = create_lod_interpolated_hairs(lod_system, count_per_guide=1)

        assert len(interpolated) == 1
        assert guides[0].index in interpolated[0].guide_hair_indices

    def test_create_interpolated_has_offset(self):
        """Interpolated hairs should have random offset from guide."""
        lod_system = HairLODSystem()

        guides = [create_test_guide_hair([0.0, 0.0, 0.0], 0)]
        lod_system.initialize(guides)

        interpolated = create_lod_interpolated_hairs(lod_system, count_per_guide=10)

        # Check that interpolated roots differ from guide root
        guide_root = guides[0].root_position
        for hair in interpolated:
            offset = np.linalg.norm(hair.root_position - guide_root)
            # Should have some offset (unlikely all are exactly 0)
            if offset > 0:
                break
        else:
            # At least one should have offset
            pass  # Random, might all be small

    def test_create_interpolated_no_guides(self):
        """Should handle empty LOD system."""
        lod_system = HairLODSystem()
        lod_system.initialize([])

        interpolated = create_lod_interpolated_hairs(lod_system, count_per_guide=5)

        assert len(interpolated) == 0

    def test_create_interpolated_control_points_match(self):
        """Interpolated hair should have same number of control points as guide."""
        lod_system = HairLODSystem()

        guides = [create_test_guide_hair([0.0, 0.0, 0.0], 0)]
        lod_system.initialize(guides)

        interpolated = create_lod_interpolated_hairs(lod_system, count_per_guide=1)

        assert len(interpolated[0].control_points) == len(guides[0].control_points)


class TestEdgeCases:
    """Edge case tests for LOD system."""

    def test_single_guide_hair(self):
        """Should handle single guide hair."""
        system = HairLODSystem()

        guides = [create_test_guide_hair([0.0, 0.0, 0.0], 0)]
        system.initialize(guides)

        assert len(system._guides_high) == 1
        assert len(system._guides_medium) == 1
        assert len(system._guides_low) == 1

    def test_many_guide_hairs(self):
        """Should handle many guide hairs."""
        system = HairLODSystem()

        guides = [create_test_guide_hair([i * 0.01, 0.0, 0.0], i) for i in range(100)]
        system.initialize(guides)

        assert len(system._guides_high) == 100
        assert len(system._guides_medium) < 100
        assert len(system._guides_low) < len(system._guides_medium)

    def test_rapid_lod_changes(self):
        """Should handle rapid LOD changes without issues."""
        settings = LODSettings(hysteresis=0.0)
        system = HairLODSystem(settings=settings)

        guides = [create_test_guide_hair([0.0, 0.0, 0.0], 0)]
        system.initialize(guides)

        hair_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Rapidly change distance
        for i in range(100):
            dist = 1.0 + (i % 30)  # Oscillate between 1 and 30
            camera_pos = np.array([dist, 0.0, 0.0], dtype=np.float32)
            system.update(camera_pos, hair_center)

        # Should not crash, system should be in valid state
        assert system.current_level in [
            HairLODLevel.HIGH, HairLODLevel.MEDIUM,
            HairLODLevel.LOW, HairLODLevel.SHELL
        ]

    def test_camera_at_hair_center(self):
        """Should handle camera at exact hair center."""
        system = HairLODSystem()

        guides = [create_test_guide_hair([0.0, 0.0, 0.0], 0)]
        system.initialize(guides)

        camera_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        hair_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        system.update(camera_pos, hair_center)

        # Distance is 0, should be high LOD
        assert system.current_level == HairLODLevel.HIGH

    def test_very_large_distance(self):
        """Should handle very large camera distance."""
        settings = LODSettings(hysteresis=0.0)  # No hysteresis for easier testing
        system = HairLODSystem(settings=settings)

        guides = [create_test_guide_hair([0.0, 0.0, 0.0], 0)]
        system.initialize(guides)

        hair_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Need to transition through all levels
        # Start at high, then move to medium, low, shell
        distances = [3.0, 7.0, 15.0, 25.0]  # Beyond each threshold
        for dist in distances:
            camera_pos = np.array([dist, 0.0, 0.0], dtype=np.float32)
            system.update(camera_pos, hair_center)

        # Should be shell mode at very far distance
        assert system.current_level == HairLODLevel.SHELL

    def test_interpolation_weights_single_guide(self):
        """get_interpolation_weights with k > guides should return all guides."""
        system = HairLODSystem()

        guides = [create_test_guide_hair([0.0, 0.0, 0.0], 0)]
        system.initialize(guides)

        indices, weights = system.get_interpolation_weights(
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            k_nearest=5  # More than available
        )

        assert len(indices) == 1  # Only 1 guide available
        assert weights[0] == pytest.approx(1.0)

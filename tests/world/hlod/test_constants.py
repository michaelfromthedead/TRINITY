"""
Tests for HLOD system constants.

Validates that constants are properly defined and have valid values.
"""

import pytest

from engine.world.hlod.constants import (
    FloatingPointConstants,
    SimplificationConstants,
    MergeConstants,
    ImpostorConstants,
    MethodSelectionConstants,
    LayerConstants,
    TransitionConstantsConfig,
    ValidationConstants,
)


class TestFloatingPointConstants:
    """Tests for FloatingPointConstants."""

    def test_epsilon_is_small_positive(self) -> None:
        """Test epsilon is a small positive value."""
        assert FloatingPointConstants.EPSILON > 0
        assert FloatingPointConstants.EPSILON < 1e-6

    def test_transition_epsilon_is_small_positive(self) -> None:
        """Test transition epsilon is a small positive value."""
        assert FloatingPointConstants.TRANSITION_EPSILON > 0
        assert FloatingPointConstants.TRANSITION_EPSILON < 1e-4

    def test_hash_rounding_precision_is_valid(self) -> None:
        """Test hash rounding precision is reasonable."""
        assert FloatingPointConstants.HASH_ROUNDING_PRECISION >= 3
        assert FloatingPointConstants.HASH_ROUNDING_PRECISION <= 10


class TestSimplificationConstants:
    """Tests for SimplificationConstants."""

    def test_default_target_ratio_is_valid(self) -> None:
        """Test default target ratio is in valid range."""
        assert 0.0 < SimplificationConstants.DEFAULT_TARGET_RATIO <= 1.0

    def test_default_max_error_is_non_negative(self) -> None:
        """Test default max error is non-negative."""
        assert SimplificationConstants.DEFAULT_MAX_ERROR >= 0.0

    def test_minimum_values_are_positive(self) -> None:
        """Test minimum triangle/vertex counts are positive."""
        assert SimplificationConstants.MIN_TRIANGLES >= 1
        assert SimplificationConstants.MIN_VERTICES >= 3

    def test_edge_collapse_weights_are_positive(self) -> None:
        """Test edge collapse weights are positive."""
        assert SimplificationConstants.EDGE_COLLAPSE_WEIGHT_POSITION > 0
        assert SimplificationConstants.EDGE_COLLAPSE_WEIGHT_NORMAL >= 0
        assert SimplificationConstants.EDGE_COLLAPSE_WEIGHT_UV >= 0


class TestMergeConstants:
    """Tests for MergeConstants."""

    def test_default_merge_distance_is_small_positive(self) -> None:
        """Test default merge distance is a small positive value."""
        assert MergeConstants.DEFAULT_MERGE_DISTANCE > 0
        assert MergeConstants.DEFAULT_MERGE_DISTANCE < 1.0

    def test_opposing_normal_threshold_is_negative(self) -> None:
        """Test opposing normal threshold is negative (for detecting opposing faces)."""
        assert MergeConstants.OPPOSING_NORMAL_THRESHOLD < 0
        assert MergeConstants.OPPOSING_NORMAL_THRESHOLD >= -1.0

    def test_interior_face_distance_multiplier_is_positive(self) -> None:
        """Test interior face distance multiplier is positive."""
        assert MergeConstants.INTERIOR_FACE_DISTANCE_MULTIPLIER > 0


class TestImpostorConstants:
    """Tests for ImpostorConstants."""

    def test_default_resolution_is_power_of_two(self) -> None:
        """Test default resolution is a power of two (common for textures)."""
        res = ImpostorConstants.DEFAULT_RESOLUTION
        # Check if it's a power of 2
        assert res > 0 and (res & (res - 1)) == 0

    def test_default_view_count_is_reasonable(self) -> None:
        """Test default view count is reasonable."""
        assert ImpostorConstants.DEFAULT_VIEW_COUNT >= 4
        assert ImpostorConstants.DEFAULT_VIEW_COUNT <= 32

    def test_resolution_bounds_are_valid(self) -> None:
        """Test resolution bounds are valid."""
        assert ImpostorConstants.MIN_RESOLUTION > 0
        assert ImpostorConstants.MAX_RESOLUTION > ImpostorConstants.MIN_RESOLUTION
        assert ImpostorConstants.DEFAULT_RESOLUTION >= ImpostorConstants.MIN_RESOLUTION
        assert ImpostorConstants.DEFAULT_RESOLUTION <= ImpostorConstants.MAX_RESOLUTION


class TestMethodSelectionConstants:
    """Tests for MethodSelectionConstants."""

    def test_triangle_thresholds_are_increasing(self) -> None:
        """Test triangle thresholds are in increasing order."""
        assert MethodSelectionConstants.SMALL_MESH_TRIANGLE_THRESHOLD > 0
        assert (MethodSelectionConstants.MEDIUM_MESH_TRIANGLE_THRESHOLD >
                MethodSelectionConstants.SMALL_MESH_TRIANGLE_THRESHOLD)
        assert (MethodSelectionConstants.LARGE_MESH_TRIANGLE_THRESHOLD >
                MethodSelectionConstants.MEDIUM_MESH_TRIANGLE_THRESHOLD)

    def test_many_meshes_threshold_is_positive(self) -> None:
        """Test many meshes threshold is positive."""
        assert MethodSelectionConstants.MANY_MESHES_THRESHOLD > 0


class TestLayerConstants:
    """Tests for LayerConstants."""

    def test_default_distances_are_increasing(self) -> None:
        """Test default LOD distances are in increasing order."""
        distances = [
            LayerConstants.DEFAULT_LOD0_DISTANCE,
            LayerConstants.DEFAULT_LOD1_DISTANCE,
            LayerConstants.DEFAULT_LOD2_DISTANCE,
            LayerConstants.DEFAULT_LOD3_DISTANCE,
        ]
        for i in range(1, len(distances)):
            assert distances[i] > distances[i-1], (
                f"LOD{i} distance not greater than LOD{i-1}"
            )

    def test_default_ratios_are_decreasing(self) -> None:
        """Test default simplification ratios are in decreasing order."""
        ratios = [
            LayerConstants.DEFAULT_LOD0_RATIO,
            LayerConstants.DEFAULT_LOD1_RATIO,
            LayerConstants.DEFAULT_LOD2_RATIO,
            LayerConstants.DEFAULT_LOD3_RATIO,
        ]
        for i in range(1, len(ratios)):
            assert ratios[i] < ratios[i-1], (
                f"LOD{i} ratio not less than LOD{i-1}"
            )

    def test_ratios_are_valid(self) -> None:
        """Test all ratios are in valid range (0, 1]."""
        ratios = [
            LayerConstants.DEFAULT_LOD0_RATIO,
            LayerConstants.DEFAULT_LOD1_RATIO,
            LayerConstants.DEFAULT_LOD2_RATIO,
            LayerConstants.DEFAULT_LOD3_RATIO,
        ]
        for ratio in ratios:
            assert 0.0 < ratio <= 1.0

    def test_max_layers_is_reasonable(self) -> None:
        """Test max layers is a reasonable value."""
        assert LayerConstants.MAX_LAYERS >= 4
        assert LayerConstants.MAX_LAYERS <= 16


class TestTransitionConstantsConfig:
    """Tests for TransitionConstantsConfig."""

    def test_default_transition_range_is_positive(self) -> None:
        """Test default transition range is positive."""
        assert TransitionConstantsConfig.DEFAULT_TRANSITION_RANGE > 0

    def test_default_dither_scale_is_positive(self) -> None:
        """Test default dither scale is positive."""
        assert TransitionConstantsConfig.DEFAULT_DITHER_SCALE > 0

    def test_default_morph_speed_is_positive(self) -> None:
        """Test default morph speed is positive."""
        assert TransitionConstantsConfig.DEFAULT_MORPH_SPEED > 0

    def test_dither_pattern_size_is_power_of_two(self) -> None:
        """Test dither pattern size is a power of two."""
        size = TransitionConstantsConfig.DITHER_PATTERN_SIZE
        assert size > 0 and (size & (size - 1)) == 0

    def test_hysteresis_bounds_are_valid(self) -> None:
        """Test hysteresis factor bounds are valid."""
        assert TransitionConstantsConfig.MIN_HYSTERESIS_FACTOR >= 0.0
        assert TransitionConstantsConfig.MAX_HYSTERESIS_FACTOR <= 1.0
        assert (TransitionConstantsConfig.MIN_HYSTERESIS_FACTOR <=
                TransitionConstantsConfig.DEFAULT_HYSTERESIS_FACTOR <=
                TransitionConstantsConfig.MAX_HYSTERESIS_FACTOR)


class TestValidationConstants:
    """Tests for ValidationConstants."""

    def test_ratio_bounds_are_valid(self) -> None:
        """Test ratio bounds are valid."""
        assert ValidationConstants.MIN_RATIO == 0.0
        assert ValidationConstants.MAX_RATIO == 1.0

    def test_distance_bounds_are_valid(self) -> None:
        """Test distance bounds are valid."""
        assert ValidationConstants.MIN_DISTANCE == 0.0

    def test_screen_bounds_are_valid(self) -> None:
        """Test screen bounds are valid."""
        assert ValidationConstants.MIN_SCREEN_HEIGHT >= 1
        assert ValidationConstants.MIN_FOV_RADIANS > 0
        assert ValidationConstants.MAX_FOV_RADIANS > ValidationConstants.MIN_FOV_RADIANS

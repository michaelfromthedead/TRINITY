"""
Whitebox tests for hair simulation configuration constants.
"""

import pytest

from engine.simulation.hair.config import (
    COLLISION_STIFFNESS,
    DEFAULT_HAIR_LENGTH,
    DEFAULT_HAIR_THICKNESS,
    DEFAULT_STRAND_SEGMENTS,
    GRAVITY_DROOP_FACTOR,
    HAIR_AIR_RESISTANCE,
    HAIR_COLLISION_MARGIN,
    HAIR_CURL_FREQUENCY_MAX,
    HAIR_CURL_FREQUENCY_MIN,
    HAIR_DAMPING,
    HAIR_SOLVER_ITERATIONS,
    HAIR_TIMESTEP,
    HEAD_INERTIA_COEFFICIENT,
    INTERPOLATION_RATIO,
    LENGTH_STIFFNESS,
    LOCAL_SHAPE_CORRECTION_FACTOR,
    LOCAL_SHAPE_STIFFNESS,
    LOD_DISTANCE_HIGH,
    LOD_DISTANCE_LOW,
    LOD_DISTANCE_MEDIUM,
    LOD_DISTANCE_SHELL,
    LOD_GUIDE_FACTOR_HIGH,
    LOD_GUIDE_FACTOR_LOW,
    LOD_GUIDE_FACTOR_MEDIUM,
    LOD_GUIDE_FACTOR_SHELL,
    LOD_INTERPOLATION_OFFSET,
    LOD_SEGMENT_FACTOR_HIGH,
    LOD_SEGMENT_FACTOR_LOW,
    LOD_SEGMENT_FACTOR_MEDIUM,
    MAX_COLLISION_ITERATIONS,
    MAX_GUIDE_HAIRS,
    MAX_INTERPOLATED_HAIRS,
    MAX_STRAND_SEGMENTS,
    MIN_GUIDE_HAIRS,
    MIN_STRAND_SEGMENTS,
    MIN_VELOCITY_TIMESTEP,
    NUMERICAL_EPSILON,
    ROOT_STIFFNESS,
    SELF_COLLISION_DENSITY_THRESHOLD,
    SELF_COLLISION_PUSH_STRENGTH,
    SELF_COLLISION_RADIUS,
    SHAPE_STIFFNESS,
    WIND_INFLUENCE_MULTIPLIER,
    HairQualityPreset,
)


class TestStrandParameters:
    """Tests for strand parameter constants."""

    def test_default_strand_segments_valid(self):
        """Default strand segments should be within valid range."""
        assert MIN_STRAND_SEGMENTS <= DEFAULT_STRAND_SEGMENTS <= MAX_STRAND_SEGMENTS
        assert DEFAULT_STRAND_SEGMENTS == 16

    def test_min_strand_segments_positive(self):
        """Minimum strand segments should be positive."""
        assert MIN_STRAND_SEGMENTS > 0
        assert MIN_STRAND_SEGMENTS == 4

    def test_max_strand_segments_larger_than_min(self):
        """Maximum strand segments should exceed minimum."""
        assert MAX_STRAND_SEGMENTS > MIN_STRAND_SEGMENTS
        assert MAX_STRAND_SEGMENTS == 32

    def test_default_hair_thickness_realistic(self):
        """Default hair thickness should be realistic (0.05-0.1mm)."""
        # Human hair is typically 0.05-0.1mm diameter
        assert 0.00005 <= DEFAULT_HAIR_THICKNESS <= 0.001
        assert DEFAULT_HAIR_THICKNESS == 0.0003  # 0.3mm

    def test_default_hair_length_realistic(self):
        """Default hair length should be reasonable."""
        # 30cm is reasonable for medium-length hair
        assert 0.1 <= DEFAULT_HAIR_LENGTH <= 1.0
        assert DEFAULT_HAIR_LENGTH == 0.3


class TestSimulationParameters:
    """Tests for simulation parameter constants."""

    def test_timestep_reasonable(self):
        """Timestep should allow for stable simulation."""
        # 120Hz is common for physics simulation
        assert 0.001 <= HAIR_TIMESTEP <= 0.02
        assert HAIR_TIMESTEP == pytest.approx(1.0 / 120.0)

    def test_solver_iterations_positive(self):
        """Solver iterations should be positive and reasonable."""
        assert 1 <= HAIR_SOLVER_ITERATIONS <= 10
        assert HAIR_SOLVER_ITERATIONS == 4

    def test_damping_in_valid_range(self):
        """Damping should be between 0 and 1."""
        assert 0.0 <= HAIR_DAMPING <= 1.0
        assert HAIR_DAMPING == 0.95

    def test_air_resistance_non_negative(self):
        """Air resistance should be non-negative."""
        assert HAIR_AIR_RESISTANCE >= 0.0
        assert HAIR_AIR_RESISTANCE == 0.1


class TestStiffnessParameters:
    """Tests for stiffness parameter constants."""

    def test_length_stiffness_valid(self):
        """Length stiffness should be in valid range."""
        assert 0.0 <= LENGTH_STIFFNESS <= 1.0
        assert LENGTH_STIFFNESS == 1.0

    def test_shape_stiffness_valid(self):
        """Shape stiffness should be in valid range."""
        assert 0.0 <= SHAPE_STIFFNESS <= 1.0
        assert SHAPE_STIFFNESS == 0.5

    def test_local_shape_stiffness_valid(self):
        """Local shape stiffness should be in valid range."""
        assert 0.0 <= LOCAL_SHAPE_STIFFNESS <= 1.0
        assert LOCAL_SHAPE_STIFFNESS == 0.3

    def test_root_stiffness_valid(self):
        """Root stiffness should be in valid range."""
        assert 0.0 <= ROOT_STIFFNESS <= 1.0
        assert ROOT_STIFFNESS == 1.0

    def test_collision_stiffness_valid(self):
        """Collision stiffness should be in valid range."""
        assert 0.0 <= COLLISION_STIFFNESS <= 1.0
        assert COLLISION_STIFFNESS == 0.8


class TestGuideHairParameters:
    """Tests for guide hair parameter constants."""

    def test_max_guide_hairs_positive(self):
        """Maximum guide hairs should be positive."""
        assert MAX_GUIDE_HAIRS > 0
        assert MAX_GUIDE_HAIRS == 1000

    def test_min_guide_hairs_positive(self):
        """Minimum guide hairs should be positive."""
        assert MIN_GUIDE_HAIRS > 0
        assert MIN_GUIDE_HAIRS == 100

    def test_max_exceeds_min_guide_hairs(self):
        """Maximum guide hairs should exceed minimum."""
        assert MAX_GUIDE_HAIRS > MIN_GUIDE_HAIRS

    def test_interpolation_ratio_positive(self):
        """Interpolation ratio should be positive."""
        assert INTERPOLATION_RATIO > 0
        assert INTERPOLATION_RATIO == 10

    def test_max_interpolated_hairs_consistent(self):
        """Max interpolated should be compatible with max guides."""
        assert MAX_INTERPOLATED_HAIRS >= MAX_GUIDE_HAIRS
        assert MAX_INTERPOLATED_HAIRS == 10000


class TestCollisionParameters:
    """Tests for collision parameter constants."""

    def test_collision_margin_positive(self):
        """Collision margin should be positive."""
        assert HAIR_COLLISION_MARGIN > 0
        assert HAIR_COLLISION_MARGIN == 0.002

    def test_self_collision_radius_positive(self):
        """Self-collision radius should be positive."""
        assert SELF_COLLISION_RADIUS > 0
        assert SELF_COLLISION_RADIUS == 0.005

    def test_max_collision_iterations_positive(self):
        """Max collision iterations should be positive."""
        assert MAX_COLLISION_ITERATIONS > 0
        assert MAX_COLLISION_ITERATIONS == 4

    def test_self_collision_density_threshold_positive(self):
        """Self-collision density threshold should be positive."""
        assert SELF_COLLISION_DENSITY_THRESHOLD > 0
        assert SELF_COLLISION_DENSITY_THRESHOLD == 2.0

    def test_self_collision_push_strength_positive(self):
        """Self-collision push strength should be positive."""
        assert SELF_COLLISION_PUSH_STRENGTH > 0
        assert SELF_COLLISION_PUSH_STRENGTH == 0.01


class TestPhysicsParameters:
    """Tests for physics-related constants."""

    def test_head_inertia_coefficient_valid(self):
        """Head inertia coefficient should be in valid range."""
        assert 0.0 <= HEAD_INERTIA_COEFFICIENT <= 1.0
        assert HEAD_INERTIA_COEFFICIENT == 0.5

    def test_wind_influence_multiplier_positive(self):
        """Wind influence multiplier should be positive."""
        assert WIND_INFLUENCE_MULTIPLIER > 0
        assert WIND_INFLUENCE_MULTIPLIER == 0.1


class TestNumericalParameters:
    """Tests for numerical stability constants."""

    def test_numerical_epsilon_small_positive(self):
        """Numerical epsilon should be small and positive."""
        assert NUMERICAL_EPSILON > 0
        assert NUMERICAL_EPSILON < 1e-6
        assert NUMERICAL_EPSILON == 1e-8

    def test_min_velocity_timestep_positive(self):
        """Minimum velocity timestep should be positive."""
        assert MIN_VELOCITY_TIMESTEP > 0
        assert MIN_VELOCITY_TIMESTEP == 1e-6

    def test_local_shape_correction_factor_positive(self):
        """Local shape correction factor should be positive."""
        assert LOCAL_SHAPE_CORRECTION_FACTOR > 0
        assert LOCAL_SHAPE_CORRECTION_FACTOR == 0.1

    def test_gravity_droop_factor_valid(self):
        """Gravity droop factor should be in valid range."""
        assert 0.0 <= GRAVITY_DROOP_FACTOR <= 1.0
        assert GRAVITY_DROOP_FACTOR == 0.3

    def test_lod_interpolation_offset_positive(self):
        """LOD interpolation offset should be positive."""
        assert LOD_INTERPOLATION_OFFSET > 0
        assert LOD_INTERPOLATION_OFFSET == 0.005


class TestLODParameters:
    """Tests for LOD parameter constants."""

    def test_lod_distances_increasing(self):
        """LOD distances should be strictly increasing."""
        assert LOD_DISTANCE_HIGH < LOD_DISTANCE_MEDIUM
        assert LOD_DISTANCE_MEDIUM < LOD_DISTANCE_LOW
        assert LOD_DISTANCE_LOW < LOD_DISTANCE_SHELL

    def test_lod_distances_values(self):
        """LOD distances should have expected values."""
        assert LOD_DISTANCE_HIGH == 2.0
        assert LOD_DISTANCE_MEDIUM == 5.0
        assert LOD_DISTANCE_LOW == 10.0
        assert LOD_DISTANCE_SHELL == 20.0

    def test_lod_guide_factors_decreasing(self):
        """LOD guide factors should decrease with distance."""
        assert LOD_GUIDE_FACTOR_HIGH >= LOD_GUIDE_FACTOR_MEDIUM
        assert LOD_GUIDE_FACTOR_MEDIUM >= LOD_GUIDE_FACTOR_LOW
        assert LOD_GUIDE_FACTOR_LOW >= LOD_GUIDE_FACTOR_SHELL

    def test_lod_guide_factors_values(self):
        """LOD guide factors should have expected values."""
        assert LOD_GUIDE_FACTOR_HIGH == 1.0
        assert LOD_GUIDE_FACTOR_MEDIUM == 0.5
        assert LOD_GUIDE_FACTOR_LOW == 0.25
        assert LOD_GUIDE_FACTOR_SHELL == 0.0

    def test_lod_segment_factors_decreasing(self):
        """LOD segment factors should decrease with distance."""
        assert LOD_SEGMENT_FACTOR_HIGH >= LOD_SEGMENT_FACTOR_MEDIUM
        assert LOD_SEGMENT_FACTOR_MEDIUM >= LOD_SEGMENT_FACTOR_LOW

    def test_lod_segment_factors_values(self):
        """LOD segment factors should have expected values."""
        assert LOD_SEGMENT_FACTOR_HIGH == 1.0
        assert LOD_SEGMENT_FACTOR_MEDIUM == 0.75
        assert LOD_SEGMENT_FACTOR_LOW == 0.5


class TestHairQualityPresets:
    """Tests for HairQualityPreset class."""

    def test_ultra_preset_structure(self):
        """Ultra preset should have all required keys."""
        preset = HairQualityPreset.ULTRA
        assert "guide_hairs" in preset
        assert "segments" in preset
        assert "interpolation_ratio" in preset
        assert "solver_iterations" in preset
        assert "self_collision" in preset
        assert "wind_enabled" in preset

    def test_ultra_preset_values(self):
        """Ultra preset should have highest quality values."""
        preset = HairQualityPreset.ULTRA
        assert preset["guide_hairs"] == 1000
        assert preset["segments"] == 32
        assert preset["interpolation_ratio"] == 20
        assert preset["solver_iterations"] == 8
        assert preset["self_collision"] is True
        assert preset["wind_enabled"] is True

    def test_high_preset_values(self):
        """High preset should have high quality values."""
        preset = HairQualityPreset.HIGH
        assert preset["guide_hairs"] == 500
        assert preset["segments"] == 16
        assert preset["interpolation_ratio"] == 10
        assert preset["solver_iterations"] == 4
        assert preset["self_collision"] is True
        assert preset["wind_enabled"] is True

    def test_medium_preset_values(self):
        """Medium preset should have medium quality values."""
        preset = HairQualityPreset.MEDIUM
        assert preset["guide_hairs"] == 250
        assert preset["segments"] == 12
        assert preset["interpolation_ratio"] == 8
        assert preset["solver_iterations"] == 3
        assert preset["self_collision"] is False
        assert preset["wind_enabled"] is True

    def test_low_preset_values(self):
        """Low preset should have low quality values."""
        preset = HairQualityPreset.LOW
        assert preset["guide_hairs"] == 100
        assert preset["segments"] == 8
        assert preset["interpolation_ratio"] == 4
        assert preset["solver_iterations"] == 2
        assert preset["self_collision"] is False
        assert preset["wind_enabled"] is False

    def test_mobile_preset_values(self):
        """Mobile preset should have minimal quality values."""
        preset = HairQualityPreset.MOBILE
        assert preset["guide_hairs"] == 50
        assert preset["segments"] == 4
        assert preset["interpolation_ratio"] == 2
        assert preset["solver_iterations"] == 1
        assert preset["self_collision"] is False
        assert preset["wind_enabled"] is False

    def test_presets_quality_ordering(self):
        """Presets should have decreasing quality order."""
        ultra = HairQualityPreset.ULTRA
        high = HairQualityPreset.HIGH
        medium = HairQualityPreset.MEDIUM
        low = HairQualityPreset.LOW
        mobile = HairQualityPreset.MOBILE

        assert ultra["guide_hairs"] > high["guide_hairs"]
        assert high["guide_hairs"] > medium["guide_hairs"]
        assert medium["guide_hairs"] > low["guide_hairs"]
        assert low["guide_hairs"] > mobile["guide_hairs"]


class TestMaterialParameters:
    """Tests for material property constants."""

    def test_hair_curl_frequency_range_valid(self):
        """Hair curl frequency range should be valid."""
        assert HAIR_CURL_FREQUENCY_MIN >= 0.0
        assert HAIR_CURL_FREQUENCY_MAX > HAIR_CURL_FREQUENCY_MIN
        assert HAIR_CURL_FREQUENCY_MIN == 0.0
        assert HAIR_CURL_FREQUENCY_MAX == 10.0

"""
Whitebox tests for cloth simulation configuration.

Tests:
- Configuration constants validity
- Quality presets structure
- Numerical stability constants
"""

import pytest

from engine.simulation.cloth.config import (
    BENDING_CORRECTION_FACTOR,
    CLOTH_DAMPING,
    CLOTH_SOLVER_ITERATIONS,
    CLOTH_SUBSTEPS,
    CLOTH_TIMESTEP,
    COLLISION_FRICTION,
    COLLISION_MARGIN,
    COLLISION_RESTITUTION,
    DEFAULT_BEND_STIFFNESS,
    DEFAULT_SHEAR_STIFFNESS,
    DEFAULT_STRETCH_STIFFNESS,
    LONG_RANGE_MAX_RATIO,
    LONG_RANGE_STIFFNESS,
    MAX_CLOTH_EDGES,
    MAX_CLOTH_OBJECTS,
    MAX_CLOTH_PARTICLES,
    MAX_CLOTH_TRIANGLES,
    MAX_COLLISION_NEIGHBORS,
    MIN_VELOCITY_TIMESTEP,
    NUMERICAL_EPSILON,
    SELF_COLLISION_CORRECTION_FACTOR,
    SELF_COLLISION_THICKNESS,
    SPATIAL_HASH_CELL_SIZE,
    SPATIAL_HASH_TABLE_SIZE,
    WIND_DRAG_COEFFICIENT,
    WIND_LIFT_COEFFICIENT,
    WIND_TURBULENCE_FREQUENCY,
    WIND_TURBULENCE_OCTAVES,
    WIND_TURBULENCE_STRENGTH,
    ClothQualityPreset,
)


class TestStiffnessConstants:
    """Test stiffness coefficient validity."""

    def test_stretch_stiffness_valid_range(self):
        """Stretch stiffness should be in [0, 1]."""
        assert 0.0 <= DEFAULT_STRETCH_STIFFNESS <= 1.0

    def test_bend_stiffness_valid_range(self):
        """Bend stiffness should be in [0, 1]."""
        assert 0.0 <= DEFAULT_BEND_STIFFNESS <= 1.0

    def test_shear_stiffness_valid_range(self):
        """Shear stiffness should be in [0, 1]."""
        assert 0.0 <= DEFAULT_SHEAR_STIFFNESS <= 1.0

    def test_stiffness_ordering(self):
        """Stretch should typically be highest, bend lowest."""
        # Standard cloth: stretch > shear > bend
        assert DEFAULT_STRETCH_STIFFNESS >= DEFAULT_SHEAR_STIFFNESS
        assert DEFAULT_SHEAR_STIFFNESS >= DEFAULT_BEND_STIFFNESS


class TestSimulationParameters:
    """Test simulation parameter validity."""

    def test_timestep_positive(self):
        """Timestep must be positive."""
        assert CLOTH_TIMESTEP > 0

    def test_timestep_reasonable(self):
        """Timestep should be reasonable for real-time (60-240 Hz)."""
        assert 1.0 / 240 <= CLOTH_TIMESTEP <= 1.0 / 30

    def test_substeps_positive(self):
        """Substeps must be at least 1."""
        assert CLOTH_SUBSTEPS >= 1

    def test_solver_iterations_positive(self):
        """Solver iterations must be at least 1."""
        assert CLOTH_SOLVER_ITERATIONS >= 1

    def test_damping_valid_range(self):
        """Damping should be in (0, 1]."""
        assert 0.0 < CLOTH_DAMPING <= 1.0

    def test_damping_near_one(self):
        """Damping should be near 1 for realistic behavior."""
        assert CLOTH_DAMPING >= 0.9


class TestCollisionParameters:
    """Test collision parameter validity."""

    def test_self_collision_thickness_positive(self):
        """Self-collision thickness must be positive."""
        assert SELF_COLLISION_THICKNESS > 0

    def test_collision_margin_positive(self):
        """Collision margin must be positive."""
        assert COLLISION_MARGIN > 0

    def test_friction_valid_range(self):
        """Friction coefficient should be in [0, 1]."""
        assert 0.0 <= COLLISION_FRICTION <= 1.0

    def test_restitution_valid_range(self):
        """Restitution should be in [0, 1]."""
        assert 0.0 <= COLLISION_RESTITUTION <= 1.0

    def test_max_collision_neighbors_positive(self):
        """Max collision neighbors must be positive."""
        assert MAX_COLLISION_NEIGHBORS > 0


class TestParticleLimits:
    """Test particle and mesh limits."""

    def test_max_particles_positive(self):
        """Max particles must be positive."""
        assert MAX_CLOTH_PARTICLES > 0

    def test_max_particles_reasonable(self):
        """Max particles should be reasonable for GPU."""
        assert MAX_CLOTH_PARTICLES <= 1_000_000

    def test_max_edges_greater_than_particles(self):
        """A mesh typically has more edges than particles."""
        assert MAX_CLOTH_EDGES >= MAX_CLOTH_PARTICLES

    def test_max_triangles_reasonable(self):
        """Max triangles should be reasonable."""
        assert MAX_CLOTH_TRIANGLES > 0
        assert MAX_CLOTH_TRIANGLES <= 2 * MAX_CLOTH_PARTICLES

    def test_max_objects_positive(self):
        """Max cloth objects must be positive."""
        assert MAX_CLOTH_OBJECTS > 0


class TestSpatialHashingParameters:
    """Test spatial hashing configuration."""

    def test_cell_size_positive(self):
        """Cell size must be positive."""
        assert SPATIAL_HASH_CELL_SIZE > 0

    def test_cell_size_relates_to_thickness(self):
        """Cell size should be similar to or larger than collision thickness."""
        assert SPATIAL_HASH_CELL_SIZE >= SELF_COLLISION_THICKNESS * 0.5

    def test_table_size_power_of_two(self):
        """Table size should be power of 2 for efficient hashing."""
        assert SPATIAL_HASH_TABLE_SIZE > 0
        assert (SPATIAL_HASH_TABLE_SIZE & (SPATIAL_HASH_TABLE_SIZE - 1)) == 0


class TestLongRangeParameters:
    """Test long-range attachment parameters."""

    def test_max_ratio_greater_than_one(self):
        """Max ratio must be > 1 to allow any stretch."""
        assert LONG_RANGE_MAX_RATIO > 1.0

    def test_stiffness_valid_range(self):
        """Long-range stiffness should be in [0, 1]."""
        assert 0.0 <= LONG_RANGE_STIFFNESS <= 1.0


class TestNumericalStabilityParameters:
    """Test numerical stability constants."""

    def test_epsilon_small_positive(self):
        """Epsilon must be small but positive."""
        assert NUMERICAL_EPSILON > 0
        assert NUMERICAL_EPSILON < 1e-6

    def test_min_velocity_timestep_positive(self):
        """Min velocity timestep must be positive."""
        assert MIN_VELOCITY_TIMESTEP > 0
        assert MIN_VELOCITY_TIMESTEP < CLOTH_TIMESTEP

    def test_correction_factors_valid_range(self):
        """Correction factors should be in (0, 1]."""
        assert 0.0 < SELF_COLLISION_CORRECTION_FACTOR <= 1.0
        assert 0.0 < BENDING_CORRECTION_FACTOR <= 1.0


class TestWindParameters:
    """Test wind force parameters."""

    def test_drag_coefficient_non_negative(self):
        """Drag coefficient must be non-negative."""
        assert WIND_DRAG_COEFFICIENT >= 0

    def test_lift_coefficient_non_negative(self):
        """Lift coefficient must be non-negative."""
        assert WIND_LIFT_COEFFICIENT >= 0

    def test_turbulence_strength_non_negative(self):
        """Turbulence strength must be non-negative."""
        assert WIND_TURBULENCE_STRENGTH >= 0

    def test_turbulence_frequency_positive(self):
        """Turbulence frequency must be positive."""
        assert WIND_TURBULENCE_FREQUENCY > 0

    def test_turbulence_octaves_positive(self):
        """Turbulence octaves must be at least 1."""
        assert WIND_TURBULENCE_OCTAVES >= 1


class TestClothQualityPresets:
    """Test quality preset structure and values."""

    @pytest.mark.parametrize("preset_name", ["HIGH", "MEDIUM", "LOW", "MOBILE"])
    def test_preset_has_required_keys(self, preset_name):
        """Each preset should have all required keys."""
        preset = getattr(ClothQualityPreset, preset_name)
        required_keys = ["substeps", "solver_iterations", "self_collision", "max_particles"]
        for key in required_keys:
            assert key in preset, f"Preset {preset_name} missing key: {key}"

    @pytest.mark.parametrize("preset_name", ["HIGH", "MEDIUM", "LOW", "MOBILE"])
    def test_preset_substeps_positive(self, preset_name):
        """Preset substeps must be positive."""
        preset = getattr(ClothQualityPreset, preset_name)
        assert preset["substeps"] >= 1

    @pytest.mark.parametrize("preset_name", ["HIGH", "MEDIUM", "LOW", "MOBILE"])
    def test_preset_solver_iterations_positive(self, preset_name):
        """Preset solver iterations must be positive."""
        preset = getattr(ClothQualityPreset, preset_name)
        assert preset["solver_iterations"] >= 1

    @pytest.mark.parametrize("preset_name", ["HIGH", "MEDIUM", "LOW", "MOBILE"])
    def test_preset_max_particles_positive(self, preset_name):
        """Preset max particles must be positive."""
        preset = getattr(ClothQualityPreset, preset_name)
        assert preset["max_particles"] > 0

    def test_quality_ordering(self):
        """Higher quality presets should have more substeps/iterations."""
        high = ClothQualityPreset.HIGH
        medium = ClothQualityPreset.MEDIUM
        low = ClothQualityPreset.LOW
        mobile = ClothQualityPreset.MOBILE

        # Substeps decrease with quality
        assert high["substeps"] >= medium["substeps"]
        assert medium["substeps"] >= low["substeps"]
        assert low["substeps"] >= mobile["substeps"]

        # Max particles decrease with quality
        assert high["max_particles"] >= medium["max_particles"]
        assert medium["max_particles"] >= low["max_particles"]
        assert low["max_particles"] >= mobile["max_particles"]

    def test_high_quality_has_self_collision(self):
        """HIGH preset should have self-collision enabled."""
        assert ClothQualityPreset.HIGH["self_collision"] is True

    def test_mobile_has_no_self_collision(self):
        """MOBILE preset should have self-collision disabled for performance."""
        assert ClothQualityPreset.MOBILE["self_collision"] is False

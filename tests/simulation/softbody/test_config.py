"""Tests for soft body configuration module."""

import pytest
import numpy as np

from engine.simulation.softbody.config import (
    DEFAULT_YOUNG_MODULUS,
    DEFAULT_POISSON_RATIO,
    VOLUME_STIFFNESS,
    SHAPE_MATCHING_STIFFNESS,
    MAX_DEFORMATION,
    SOFTBODY_SUBSTEPS,
    PBD_ITERATIONS,
    FEM_TOLERANCE,
    FEM_MAX_ITERATIONS,
    MIN_TET_VOLUME,
    DEFAULT_DAMPING,
    COLLISION_MARGIN,
    FEM_MIN_JACOBIAN,
    FEM_INVERSION_HANDLING,
    SVD_MIN_SINGULAR_VALUE,
    SVD_REGULARIZATION,
    BOUNDARY_VELOCITY_DAMPING,
    MUSCLE_FORCE_LENGTH_WIDTH,
    MUSCLE_ECCENTRIC_FORCE_MAX,
    MUSCLE_CONCENTRIC_THRESHOLD,
    MUSCLE_VOLUME_STIFFNESS,
    MaterialPreset,
    SoftBodyMaterial,
    SoftBodySolverType,
    SolverConfig,
    ConstraintConfig,
    CollisionConfig,
)


class TestDefaultConstants:
    """Test default constant values."""

    def test_material_properties_positive(self):
        """All material properties should be positive."""
        assert DEFAULT_YOUNG_MODULUS > 0
        assert DEFAULT_POISSON_RATIO > 0
        assert DEFAULT_POISSON_RATIO < 0.5  # Must be less than 0.5 for physical validity

    def test_stiffness_values_in_range(self):
        """Stiffness values should be in valid range [0, 1]."""
        assert 0 <= VOLUME_STIFFNESS <= 1
        assert 0 <= SHAPE_MATCHING_STIFFNESS <= 1

    def test_deformation_limits_positive(self):
        """Deformation limits should be positive."""
        assert MAX_DEFORMATION > 0

    def test_solver_parameters_valid(self):
        """Solver parameters should have valid values."""
        assert SOFTBODY_SUBSTEPS >= 1
        assert PBD_ITERATIONS >= 1
        assert FEM_TOLERANCE > 0
        assert FEM_MAX_ITERATIONS >= 1

    def test_tolerance_values_small(self):
        """Tolerance values should be small positive numbers."""
        assert MIN_TET_VOLUME > 0
        assert MIN_TET_VOLUME < 1e-4
        assert FEM_MIN_JACOBIAN > 0
        assert FEM_MIN_JACOBIAN < 1

    def test_damping_in_range(self):
        """Damping should be between 0 and 1."""
        assert 0 < DEFAULT_DAMPING <= 1.0
        assert 0 < BOUNDARY_VELOCITY_DAMPING <= 1.0

    def test_collision_margin_positive(self):
        """Collision margin should be small positive value."""
        assert COLLISION_MARGIN > 0
        assert COLLISION_MARGIN < 1.0

    def test_singularity_handling_values(self):
        """SVD singularity handling values should be valid."""
        assert SVD_MIN_SINGULAR_VALUE > 0
        assert SVD_REGULARIZATION > 0
        assert FEM_INVERSION_HANDLING in ("clamp", "reflect", "penalty")

    def test_muscle_constants_valid(self):
        """Muscle simulation constants should be valid."""
        assert MUSCLE_FORCE_LENGTH_WIDTH > 0
        assert MUSCLE_ECCENTRIC_FORCE_MAX > 1.0  # Eccentric force exceeds isometric
        assert 0 < MUSCLE_CONCENTRIC_THRESHOLD < 1.0
        assert MUSCLE_VOLUME_STIFFNESS > 0


class TestMaterialPreset:
    """Test material preset enumeration."""

    def test_all_presets_exist(self):
        """All expected presets should exist."""
        presets = [
            MaterialPreset.RUBBER,
            MaterialPreset.MUSCLE,
            MaterialPreset.FAT,
            MaterialPreset.JELLY,
            MaterialPreset.SKIN,
            MaterialPreset.FOAM,
            MaterialPreset.CLAY,
        ]
        assert len(presets) == 7

    def test_presets_are_unique(self):
        """All preset values should be unique."""
        values = [p.value for p in MaterialPreset]
        assert len(values) == len(set(values))


class TestSoftBodyMaterial:
    """Test SoftBodyMaterial dataclass."""

    def test_default_construction(self):
        """Default construction should use default constants."""
        mat = SoftBodyMaterial()
        assert mat.young_modulus == DEFAULT_YOUNG_MODULUS
        assert mat.poisson_ratio == DEFAULT_POISSON_RATIO
        assert mat.density == 1000.0
        assert mat.damping == DEFAULT_DAMPING
        assert mat.plasticity == 0.0

    def test_custom_values(self):
        """Custom values should override defaults."""
        mat = SoftBodyMaterial(
            young_modulus=50000.0,
            poisson_ratio=0.45,
            density=1200.0,
            damping=0.95,
            plasticity=0.05,
        )
        assert mat.young_modulus == 50000.0
        assert mat.poisson_ratio == 0.45
        assert mat.density == 1200.0
        assert mat.damping == 0.95
        assert mat.plasticity == 0.05

    def test_stretch_compress_limits(self):
        """Stretch and compress limits should be computed correctly."""
        mat = SoftBodyMaterial()
        assert mat.max_stretch == 1.0 + MAX_DEFORMATION
        assert mat.max_compress == 1.0 - MAX_DEFORMATION * 0.5

    def test_from_preset_rubber(self):
        """Rubber preset should have high stiffness and incompressibility."""
        mat = SoftBodyMaterial.from_preset(MaterialPreset.RUBBER)
        assert mat.young_modulus == 1e6
        assert mat.poisson_ratio == 0.49
        assert mat.density == 1100.0
        assert mat.max_stretch == 2.0

    def test_from_preset_muscle(self):
        """Muscle preset should have moderate stiffness."""
        mat = SoftBodyMaterial.from_preset(MaterialPreset.MUSCLE)
        assert mat.young_modulus == 1e4
        assert mat.poisson_ratio == 0.4
        assert mat.density == 1060.0

    def test_from_preset_fat(self):
        """Fat preset should be soft with high damping."""
        mat = SoftBodyMaterial.from_preset(MaterialPreset.FAT)
        assert mat.young_modulus == 1e3
        assert mat.poisson_ratio == 0.45
        assert mat.density == 920.0

    def test_from_preset_jelly(self):
        """Jelly preset should be very soft."""
        mat = SoftBodyMaterial.from_preset(MaterialPreset.JELLY)
        assert mat.young_modulus == 500.0
        assert mat.max_stretch == 1.5

    def test_from_preset_skin(self):
        """Skin preset should have moderate stiffness."""
        mat = SoftBodyMaterial.from_preset(MaterialPreset.SKIN)
        assert mat.young_modulus == 5e5

    def test_from_preset_foam(self):
        """Foam preset should be very compressible with low density."""
        mat = SoftBodyMaterial.from_preset(MaterialPreset.FOAM)
        assert mat.young_modulus == 100.0
        assert mat.density == 50.0
        assert mat.max_compress == 0.3

    def test_from_preset_clay(self):
        """Clay preset should have plasticity."""
        mat = SoftBodyMaterial.from_preset(MaterialPreset.CLAY)
        assert mat.plasticity == 0.1
        assert mat.density == 1800.0

    def test_compute_lame_parameters(self):
        """Lame parameters should be computed correctly."""
        mat = SoftBodyMaterial(young_modulus=10000.0, poisson_ratio=0.3)
        lam, mu = mat.compute_lame_parameters()

        # Manual calculation
        E = 10000.0
        nu = 0.3
        expected_mu = E / (2.0 * (1.0 + nu))
        expected_lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        assert np.isclose(mu, expected_mu)
        assert np.isclose(lam, expected_lam)

    def test_compute_lame_parameters_clamps_poisson(self):
        """Poisson ratio near 0.5 should be clamped to prevent singularity."""
        mat = SoftBodyMaterial(young_modulus=10000.0, poisson_ratio=0.5)
        lam, mu = mat.compute_lame_parameters()
        # Should not raise error and should produce finite values
        assert np.isfinite(lam)
        assert np.isfinite(mu)

    def test_lame_parameters_incompressible_limit(self):
        """Near-incompressible material should have large lambda."""
        mat = SoftBodyMaterial(young_modulus=10000.0, poisson_ratio=0.49)
        lam, mu = mat.compute_lame_parameters()
        assert lam > mu  # Lambda dominates for incompressible materials


class TestSoftBodySolverType:
    """Test solver type enumeration."""

    def test_all_solver_types_exist(self):
        """All expected solver types should exist."""
        solvers = [
            SoftBodySolverType.FEM,
            SoftBodySolverType.COROTATIONAL,
            SoftBodySolverType.SHAPE_MATCHING,
            SoftBodySolverType.PBD,
            SoftBodySolverType.XPBD,
        ]
        assert len(solvers) == 5


class TestSolverConfig:
    """Test SolverConfig dataclass."""

    def test_default_construction(self):
        """Default construction should use expected defaults."""
        config = SolverConfig()
        assert config.solver_type == SoftBodySolverType.PBD
        assert config.substeps == SOFTBODY_SUBSTEPS
        assert config.iterations == PBD_ITERATIONS
        assert config.tolerance == FEM_TOLERANCE
        assert config.damping == DEFAULT_DAMPING
        assert config.gravity is True

    def test_custom_values(self):
        """Custom values should override defaults."""
        config = SolverConfig(
            solver_type=SoftBodySolverType.FEM,
            substeps=8,
            iterations=10,
            tolerance=1e-8,
            damping=0.95,
            gravity=False,
        )
        assert config.solver_type == SoftBodySolverType.FEM
        assert config.substeps == 8
        assert config.iterations == 10
        assert config.tolerance == 1e-8
        assert config.damping == 0.95
        assert config.gravity is False


class TestConstraintConfig:
    """Test ConstraintConfig dataclass."""

    def test_default_construction(self):
        """Default construction should use expected defaults."""
        config = ConstraintConfig()
        assert config.volume_stiffness == VOLUME_STIFFNESS
        assert config.shape_stiffness == SHAPE_MATCHING_STIFFNESS
        assert config.edge_stiffness == 1.0
        assert config.collision_stiffness == 1.0
        assert config.friction == 0.5

    def test_custom_values(self):
        """Custom values should override defaults."""
        config = ConstraintConfig(
            volume_stiffness=0.8,
            shape_stiffness=0.6,
            edge_stiffness=0.9,
            collision_stiffness=0.7,
            friction=0.3,
        )
        assert config.volume_stiffness == 0.8
        assert config.shape_stiffness == 0.6
        assert config.edge_stiffness == 0.9
        assert config.collision_stiffness == 0.7
        assert config.friction == 0.3


class TestCollisionConfig:
    """Test CollisionConfig dataclass."""

    def test_default_construction(self):
        """Default construction should use expected defaults."""
        config = CollisionConfig()
        assert config.enabled is True
        assert config.self_collision is False
        assert config.margin == COLLISION_MARGIN
        assert config.friction == 0.5
        assert config.restitution == 0.0
        assert config.iterations == 2

    def test_custom_values(self):
        """Custom values should override defaults."""
        config = CollisionConfig(
            enabled=False,
            self_collision=True,
            margin=0.02,
            friction=0.8,
            restitution=0.3,
            iterations=4,
        )
        assert config.enabled is False
        assert config.self_collision is True
        assert config.margin == 0.02
        assert config.friction == 0.8
        assert config.restitution == 0.3
        assert config.iterations == 4

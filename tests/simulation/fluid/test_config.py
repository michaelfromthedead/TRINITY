"""Tests for fluid simulation configuration module.

Tests configuration constants, materials, and solver configs.
"""

import math
import pytest
import numpy as np

from engine.simulation.fluid.config import (
    # Constants
    PARTICLE_RADIUS,
    SMOOTHING_LENGTH,
    REST_DENSITY,
    VISCOSITY,
    SURFACE_TENSION,
    GAS_CONSTANT,
    MAX_PARTICLES,
    MAX_NEIGHBORS,
    GRID_CELL_SIZE,
    FLUID_SUBSTEPS,
    PBF_ITERATIONS,
    PRESSURE_ITERATIONS,
    CFL_NUMBER,
    PBF_LAMBDA_EPSILON,
    PBF_TENSILE_K,
    PBF_TENSILE_N,
    PBF_DELTA_Q_RATIO,
    BOUNDARY_VELOCITY_DAMPING,
    SPH_POLY6_COEFF,
    SPH_SPIKY_COEFF,
    SPH_SPIKY_GRAD_COEFF,
    SPH_VISC_LAP_COEFF,
    MC_MIN_EDGE_LENGTH,
    MC_ISO_EPSILON,
    # Enums
    FluidSolverType,
    BoundaryCondition,
    KernelType,
    # Data classes
    FluidMaterial,
    FluidConfig,
    SPHConfig,
    PBFConfig,
    FLIPConfig,
    EulerianConfig,
    ShallowWaterConfig,
    BoundaryConfig,
    EmitterConfig,
)


class TestConstants:
    """Tests for core simulation constants."""

    def test_particle_radius_positive(self):
        """Particle radius must be positive."""
        assert PARTICLE_RADIUS > 0

    def test_smoothing_length_positive(self):
        """Smoothing length must be positive."""
        assert SMOOTHING_LENGTH > 0

    def test_smoothing_length_greater_than_particle_radius(self):
        """Smoothing length should be larger than particle radius."""
        assert SMOOTHING_LENGTH >= PARTICLE_RADIUS

    def test_rest_density_physical(self):
        """Rest density should be physically reasonable (water ~ 1000 kg/m^3)."""
        assert 100 < REST_DENSITY < 10000

    def test_viscosity_non_negative(self):
        """Viscosity must be non-negative."""
        assert VISCOSITY >= 0

    def test_surface_tension_non_negative(self):
        """Surface tension must be non-negative."""
        assert SURFACE_TENSION >= 0

    def test_gas_constant_positive(self):
        """Gas constant must be positive for pressure computation."""
        assert GAS_CONSTANT > 0

    def test_max_particles_reasonable(self):
        """Max particles should be reasonable for simulation."""
        assert 1000 <= MAX_PARTICLES <= 10_000_000

    def test_max_neighbors_reasonable(self):
        """Max neighbors should be reasonable (not too many, not too few)."""
        assert 8 <= MAX_NEIGHBORS <= 256

    def test_grid_cell_size_positive(self):
        """Grid cell size must be positive."""
        assert GRID_CELL_SIZE > 0

    def test_fluid_substeps_positive(self):
        """Substeps must be at least 1."""
        assert FLUID_SUBSTEPS >= 1

    def test_pbf_iterations_positive(self):
        """PBF iterations must be at least 1."""
        assert PBF_ITERATIONS >= 1

    def test_pressure_iterations_positive(self):
        """Pressure iterations must be at least 1."""
        assert PRESSURE_ITERATIONS >= 1

    def test_cfl_number_valid(self):
        """CFL number should be in valid range (0, 1]."""
        assert 0 < CFL_NUMBER <= 1.0

    def test_pbf_lambda_epsilon_small(self):
        """PBF lambda epsilon should be small but positive."""
        assert 0 < PBF_LAMBDA_EPSILON < 1.0

    def test_boundary_velocity_damping_valid(self):
        """Boundary damping should be in [0, 1]."""
        assert 0 <= BOUNDARY_VELOCITY_DAMPING <= 1.0

    def test_sph_kernel_coefficients_positive(self):
        """SPH kernel coefficients should be positive."""
        assert SPH_POLY6_COEFF > 0
        assert SPH_SPIKY_COEFF > 0
        assert SPH_VISC_LAP_COEFF > 0

    def test_sph_spiky_gradient_coefficient_negative(self):
        """Spiky gradient coefficient should be negative."""
        assert SPH_SPIKY_GRAD_COEFF < 0

    def test_marching_cubes_epsilon_small(self):
        """Marching cubes epsilons should be very small."""
        assert MC_MIN_EDGE_LENGTH < 1e-6
        assert MC_ISO_EPSILON < 1e-6


class TestEnums:
    """Tests for configuration enumerations."""

    def test_fluid_solver_types(self):
        """All expected solver types should exist."""
        assert FluidSolverType.SPH is not None
        assert FluidSolverType.PBF is not None
        assert FluidSolverType.FLIP is not None
        assert FluidSolverType.PIC is not None
        assert FluidSolverType.APIC is not None
        assert FluidSolverType.EULERIAN is not None
        assert FluidSolverType.SHALLOW_WATER is not None

    def test_boundary_conditions(self):
        """All expected boundary conditions should exist."""
        assert BoundaryCondition.SOLID is not None
        assert BoundaryCondition.FREE_SLIP is not None
        assert BoundaryCondition.OPEN is not None
        assert BoundaryCondition.PERIODIC is not None
        assert BoundaryCondition.INFLOW is not None

    def test_kernel_types(self):
        """All expected kernel types should exist."""
        assert KernelType.POLY6 is not None
        assert KernelType.SPIKY is not None
        assert KernelType.VISCOSITY is not None
        assert KernelType.CUBIC_SPLINE is not None


class TestFluidMaterial:
    """Tests for FluidMaterial dataclass."""

    def test_default_material_values(self):
        """Default material should have reasonable values."""
        mat = FluidMaterial()
        assert mat.rest_density == REST_DENSITY
        assert mat.viscosity == VISCOSITY
        assert mat.surface_tension == SURFACE_TENSION
        assert mat.gas_constant == GAS_CONSTANT

    def test_water_material(self):
        """Water material should have realistic properties."""
        water = FluidMaterial.water()
        assert water.rest_density == 1000.0
        assert water.viscosity > 0
        assert water.surface_tension > 0
        assert len(water.color) == 4  # RGBA

    def test_oil_material(self):
        """Oil material should have lower density than water."""
        oil = FluidMaterial.oil()
        water = FluidMaterial.water()
        assert oil.rest_density < water.rest_density
        assert oil.viscosity > water.viscosity  # Oil is more viscous

    def test_honey_material(self):
        """Honey should be very viscous."""
        honey = FluidMaterial.honey()
        water = FluidMaterial.water()
        assert honey.viscosity > water.viscosity * 100  # Much more viscous

    def test_blood_material(self):
        """Blood should have similar properties to water."""
        blood = FluidMaterial.blood()
        assert 1000 < blood.rest_density < 1100  # Slightly denser than water

    def test_lava_material(self):
        """Lava should be very dense and viscous."""
        lava = FluidMaterial.lava()
        assert lava.rest_density > 2000
        assert lava.viscosity > 10  # Very viscous

    def test_material_color_rgba(self):
        """All materials should have RGBA colors."""
        for mat_factory in [
            FluidMaterial.water,
            FluidMaterial.oil,
            FluidMaterial.honey,
            FluidMaterial.blood,
            FluidMaterial.lava,
        ]:
            mat = mat_factory()
            assert len(mat.color) == 4
            assert all(0 <= c <= 1 for c in mat.color)


class TestFluidConfig:
    """Tests for FluidConfig dataclass."""

    def test_default_config(self):
        """Default config should have reasonable values."""
        config = FluidConfig()
        assert config.solver_type == FluidSolverType.PBF
        assert config.particle_radius == PARTICLE_RADIUS
        assert config.smoothing_length == SMOOTHING_LENGTH
        assert config.max_particles == MAX_PARTICLES
        assert config.substeps >= 1
        assert config.iterations >= 1
        assert len(config.gravity) == 3
        assert config.boundary == BoundaryCondition.SOLID

    def test_gravity_default_downward(self):
        """Default gravity should point downward (-Y)."""
        config = FluidConfig()
        assert config.gravity[1] < 0  # Y is negative


class TestSPHConfig:
    """Tests for SPH-specific configuration."""

    def test_default_sph_config(self):
        """Default SPH config should have reasonable values."""
        config = SPHConfig()
        assert config.kernel == KernelType.POLY6
        assert isinstance(config.adaptive_timestep, bool)
        assert config.cfl_number == CFL_NUMBER
        assert config.xsph_factor >= 0
        assert isinstance(config.tensile_correction, bool)


class TestPBFConfig:
    """Tests for PBF-specific configuration."""

    def test_default_pbf_config(self):
        """Default PBF config should have reasonable values."""
        config = PBFConfig()
        assert config.iterations == PBF_ITERATIONS
        assert 0 <= config.relaxation <= 1
        assert config.vorticity_strength >= 0
        assert config.xsph_viscosity >= 0
        assert isinstance(config.use_poly6, bool)


class TestFLIPConfig:
    """Tests for FLIP-specific configuration."""

    def test_default_flip_config(self):
        """Default FLIP config should have reasonable values."""
        config = FLIPConfig()
        assert 0 <= config.flip_ratio <= 1
        assert config.grid_resolution > 0
        assert config.pressure_iterations == PRESSURE_ITERATIONS
        assert isinstance(config.use_apic, bool)

    def test_flip_ratio_range(self):
        """FLIP ratio 0 = PIC, 1 = FLIP."""
        pic_config = FLIPConfig(flip_ratio=0.0)
        flip_config = FLIPConfig(flip_ratio=1.0)
        assert pic_config.flip_ratio == 0.0
        assert flip_config.flip_ratio == 1.0


class TestEulerianConfig:
    """Tests for Eulerian solver configuration."""

    def test_default_eulerian_config(self):
        """Default Eulerian config should have reasonable values."""
        config = EulerianConfig()
        assert len(config.grid_size) == 3
        assert all(s > 0 for s in config.grid_size)
        assert config.dx > 0
        assert config.advection_method in ["semi_lagrangian", "maccormack"]
        assert config.pressure_solver in ["jacobi", "gauss_seidel", "multigrid"]


class TestShallowWaterConfig:
    """Tests for shallow water configuration."""

    def test_default_shallow_water_config(self):
        """Default shallow water config should have reasonable values."""
        config = ShallowWaterConfig()
        assert len(config.grid_size) == 2
        assert all(s > 0 for s in config.grid_size)
        assert config.dx > 0
        assert config.min_depth > 0
        assert config.friction >= 0
        assert 0 < config.wave_damping <= 1


class TestBoundaryConfig:
    """Tests for boundary configuration."""

    def test_default_boundary_config(self):
        """Default boundary config should have valid values."""
        config = BoundaryConfig()
        assert config.friction >= 0
        assert 0 <= config.restitution <= 1
        assert config.collision_margin >= 0
        assert isinstance(config.enable_particle_collision, bool)


class TestEmitterConfig:
    """Tests for emitter configuration."""

    def test_default_emitter_config(self):
        """Default emitter config should have valid values."""
        config = EmitterConfig()
        assert config.rate > 0
        assert len(config.velocity) == 3
        assert config.spread >= 0
        assert config.lifetime >= 0
        assert config.jitter >= 0

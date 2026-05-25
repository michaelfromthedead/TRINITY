"""Fluid simulation configuration constants.

This module defines default parameters for fluid physics simulation
including SPH, PBF, FLIP/PIC, and Eulerian solvers.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


# =============================================================================
# Core Constants
# =============================================================================

# Particle properties
PARTICLE_RADIUS: float = 0.05  # Particle visual radius in meters
SMOOTHING_LENGTH: float = 0.1  # SPH kernel smoothing length (h)
REST_DENSITY: float = 1000.0  # Water rest density (kg/m^3)

# Fluid properties
VISCOSITY: float = 0.01  # Dynamic viscosity coefficient
SURFACE_TENSION: float = 0.1  # Surface tension coefficient
GAS_CONSTANT: float = 50.0  # Pressure constant (k)
COMPRESSIBILITY: float = 0.0001  # Fluid compressibility

# Simulation limits
MAX_PARTICLES: int = 100000  # Maximum particle count
MAX_NEIGHBORS: int = 64  # Max neighbors per particle
GRID_CELL_SIZE: float = 0.1  # Spatial hash grid cell size

# Solver settings
FLUID_SUBSTEPS: int = 2  # Substeps per physics step
PBF_ITERATIONS: int = 4  # PBF constraint iterations
PRESSURE_ITERATIONS: int = 50  # Pressure solve iterations
CFL_NUMBER: float = 0.4  # CFL condition number

# PBF Numerical Stability
PBF_LAMBDA_EPSILON: float = 1e-6  # Epsilon for lambda computation to prevent division by zero
PBF_TENSILE_K: float = 0.1  # Tensile instability correction strength
PBF_TENSILE_N: int = 4  # Tensile instability correction exponent
PBF_DELTA_Q_RATIO: float = 0.1  # Delta_q as ratio of smoothing length

# Boundary Handling
BOUNDARY_VELOCITY_DAMPING: float = 0.3  # Velocity damping on boundary collision

# SPH Kernel Normalization Constants (computed for 3D)
# Poly6: 315 / (64 * pi * h^9)
# Spiky: 15 / (pi * h^6)
# Spiky gradient: -45 / (pi * h^6)
# Viscosity laplacian: 45 / (pi * h^6)
SPH_POLY6_COEFF: float = 315.0 / (64.0 * 3.14159265359)  # Multiply by h^-9
SPH_SPIKY_COEFF: float = 15.0 / 3.14159265359  # Multiply by h^-6
SPH_SPIKY_GRAD_COEFF: float = -45.0 / 3.14159265359  # Multiply by h^-6
SPH_VISC_LAP_COEFF: float = 45.0 / 3.14159265359  # Multiply by h^-6

# Marching Cubes Edge Cases
MC_MIN_EDGE_LENGTH: float = 1e-10  # Minimum edge length for vertex interpolation
MC_ISO_EPSILON: float = 1e-10  # Epsilon for iso-level comparison


# =============================================================================
# Enumerations
# =============================================================================

class FluidSolverType(Enum):
    """Available fluid solver types."""
    SPH = auto()  # Smoothed Particle Hydrodynamics
    PBF = auto()  # Position Based Fluids
    FLIP = auto()  # FLIP/PIC hybrid
    PIC = auto()  # Particle-in-Cell
    APIC = auto()  # Affine Particle-in-Cell
    EULERIAN = auto()  # Grid-based Eulerian
    SHALLOW_WATER = auto()  # Height field


class BoundaryCondition(Enum):
    """Boundary condition types."""
    SOLID = auto()  # Solid wall (no-slip)
    FREE_SLIP = auto()  # Free slip wall
    OPEN = auto()  # Open boundary (outflow)
    PERIODIC = auto()  # Periodic boundary
    INFLOW = auto()  # Inflow boundary


class KernelType(Enum):
    """SPH kernel function types."""
    POLY6 = auto()  # Poly6 kernel (density)
    SPIKY = auto()  # Spiky kernel (pressure)
    VISCOSITY = auto()  # Viscosity kernel
    CUBIC_SPLINE = auto()  # Cubic spline kernel


# =============================================================================
# Material Data Classes
# =============================================================================

@dataclass
class FluidMaterial:
    """Material properties for fluid simulation.

    Attributes:
        rest_density: Density at rest (kg/m^3)
        viscosity: Dynamic viscosity
        surface_tension: Surface tension coefficient
        gas_constant: Pressure stiffness constant
        compressibility: Compressibility factor
        color: RGBA color for rendering
    """
    rest_density: float = REST_DENSITY
    viscosity: float = VISCOSITY
    surface_tension: float = SURFACE_TENSION
    gas_constant: float = GAS_CONSTANT
    compressibility: float = COMPRESSIBILITY
    color: tuple = (0.2, 0.5, 0.8, 0.8)  # RGBA

    @classmethod
    def water(cls) -> "FluidMaterial":
        """Create water material."""
        return cls(
            rest_density=1000.0,
            viscosity=0.001,
            surface_tension=0.072,
            gas_constant=100.0,
            color=(0.2, 0.5, 0.9, 0.7)
        )

    @classmethod
    def oil(cls) -> "FluidMaterial":
        """Create oil material."""
        return cls(
            rest_density=850.0,
            viscosity=0.03,
            surface_tension=0.03,
            gas_constant=80.0,
            color=(0.3, 0.2, 0.1, 0.8)
        )

    @classmethod
    def honey(cls) -> "FluidMaterial":
        """Create honey material."""
        return cls(
            rest_density=1400.0,
            viscosity=5.0,
            surface_tension=0.05,
            gas_constant=50.0,
            color=(0.9, 0.7, 0.2, 0.9)
        )

    @classmethod
    def blood(cls) -> "FluidMaterial":
        """Create blood material."""
        return cls(
            rest_density=1060.0,
            viscosity=0.004,
            surface_tension=0.06,
            gas_constant=100.0,
            color=(0.6, 0.1, 0.1, 0.85)
        )

    @classmethod
    def lava(cls) -> "FluidMaterial":
        """Create lava material."""
        return cls(
            rest_density=2500.0,
            viscosity=100.0,
            surface_tension=0.4,
            gas_constant=30.0,
            color=(1.0, 0.3, 0.0, 0.95)
        )


# =============================================================================
# Solver Configuration
# =============================================================================

@dataclass
class FluidConfig:
    """Configuration for fluid simulation.

    Attributes:
        solver_type: Type of solver to use
        particle_radius: Visual radius of particles
        smoothing_length: SPH kernel smoothing length
        max_particles: Maximum particle count
        substeps: Simulation substeps per frame
        iterations: Solver iterations per substep
        gravity: Gravity vector components
        boundary: Default boundary condition
    """
    solver_type: FluidSolverType = FluidSolverType.PBF
    particle_radius: float = PARTICLE_RADIUS
    smoothing_length: float = SMOOTHING_LENGTH
    max_particles: int = MAX_PARTICLES
    substeps: int = FLUID_SUBSTEPS
    iterations: int = PBF_ITERATIONS
    gravity: tuple = (0.0, -9.81, 0.0)
    boundary: BoundaryCondition = BoundaryCondition.SOLID


@dataclass
class SPHConfig:
    """SPH-specific configuration.

    Attributes:
        kernel: Kernel function type
        adaptive_timestep: Use adaptive timestep based on CFL
        cfl_number: CFL condition number
        xsph_factor: XSPH viscosity factor
        tensile_correction: Enable tensile instability correction
    """
    kernel: KernelType = KernelType.POLY6
    adaptive_timestep: bool = True
    cfl_number: float = CFL_NUMBER
    xsph_factor: float = 0.01
    tensile_correction: bool = True


@dataclass
class PBFConfig:
    """PBF-specific configuration.

    Attributes:
        iterations: Constraint solver iterations
        relaxation: Relaxation parameter for Jacobi iteration
        vorticity_strength: Vorticity confinement strength
        xsph_viscosity: XSPH viscosity coefficient
        use_poly6: Use poly6 for density (more stable)
    """
    iterations: int = PBF_ITERATIONS
    relaxation: float = 0.5
    vorticity_strength: float = 0.001
    xsph_viscosity: float = 0.01
    use_poly6: bool = True


@dataclass
class FLIPConfig:
    """FLIP/PIC-specific configuration.

    Attributes:
        flip_ratio: Blend ratio (0=PIC, 1=FLIP)
        grid_resolution: Grid cells per unit length
        pressure_iterations: Pressure projection iterations
        use_apic: Use Affine PIC transfer
    """
    flip_ratio: float = 0.95
    grid_resolution: int = 64
    pressure_iterations: int = PRESSURE_ITERATIONS
    use_apic: bool = False


@dataclass
class EulerianConfig:
    """Eulerian solver configuration.

    Attributes:
        grid_size: Grid dimensions (nx, ny, nz)
        dx: Grid cell size
        advection_method: Advection scheme ("semi_lagrangian", "maccormack")
        pressure_solver: Pressure solver ("jacobi", "gauss_seidel", "multigrid")
    """
    grid_size: tuple = (64, 64, 64)
    dx: float = GRID_CELL_SIZE
    advection_method: str = "semi_lagrangian"
    pressure_solver: str = "jacobi"


@dataclass
class ShallowWaterConfig:
    """Shallow water solver configuration.

    Attributes:
        grid_size: Grid dimensions (nx, ny)
        dx: Grid cell size
        min_depth: Minimum water depth
        friction: Bottom friction coefficient
        wave_damping: Wave amplitude damping
    """
    grid_size: tuple = (128, 128)
    dx: float = 0.5
    min_depth: float = 0.01
    friction: float = 0.01
    wave_damping: float = 0.999


# =============================================================================
# Boundary Configuration
# =============================================================================

@dataclass
class BoundaryConfig:
    """Boundary and collision configuration.

    Attributes:
        friction: Boundary friction coefficient
        restitution: Bounce coefficient
        collision_margin: Collision detection margin
        enable_particle_collision: Particle-particle collision
    """
    friction: float = 0.0
    restitution: float = 0.0
    collision_margin: float = PARTICLE_RADIUS
    enable_particle_collision: bool = False


# =============================================================================
# Emitter Configuration
# =============================================================================

@dataclass
class EmitterConfig:
    """Fluid emitter configuration.

    Attributes:
        rate: Particles per second
        velocity: Initial particle velocity
        spread: Velocity spread angle (radians)
        lifetime: Particle lifetime (0 = infinite)
        jitter: Random position jitter
    """
    rate: float = 1000.0
    velocity: tuple = (0.0, 0.0, 0.0)
    spread: float = 0.0
    lifetime: float = 0.0
    jitter: float = 0.0

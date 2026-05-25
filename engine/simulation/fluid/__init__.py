"""Fluid simulation module.

This module provides multiple fluid simulation techniques:
- SPH (Smoothed Particle Hydrodynamics)
- PBF (Position Based Fluids)
- FLIP/PIC hybrid methods
- Eulerian grid-based solvers
- Shallow water simulation
- Surface reconstruction
- GPU acceleration stubs
"""

from .config import (
    PARTICLE_RADIUS,
    SMOOTHING_LENGTH,
    REST_DENSITY,
    VISCOSITY,
    SURFACE_TENSION,
    MAX_PARTICLES,
    GRID_CELL_SIZE,
    FluidMaterial,
    FluidConfig,
)
from .sph import (
    SPHSolver,
    SPHParticle,
    SPHKernels,
)
from .pbf import (
    PBFSolver,
    PBFParticle,
    PBFConfig,
)
from .flip_pic import (
    FLIPSolver,
    MACGrid,
    FLIPConfig,
)
from .eulerian import (
    EulerianSolver,
    StaggeredGrid,
    VelocityField,
)
from .shallow_water import (
    ShallowWaterSolver,
    HeightField,
    TerrainBoundary,
)
from .surface_reconstruction import (
    MarchingCubes,
    DensityField,
    FluidSurface,
)
from .gpu_fluid import (
    GPUFluidSolver,
    GPUSpatialHash,
    GPUFluidConfig,
)

__all__ = [
    # Config
    "PARTICLE_RADIUS",
    "SMOOTHING_LENGTH",
    "REST_DENSITY",
    "VISCOSITY",
    "SURFACE_TENSION",
    "MAX_PARTICLES",
    "GRID_CELL_SIZE",
    "FluidMaterial",
    "FluidConfig",
    # SPH
    "SPHSolver",
    "SPHParticle",
    "SPHKernels",
    # PBF
    "PBFSolver",
    "PBFParticle",
    "PBFConfig",
    # FLIP/PIC
    "FLIPSolver",
    "MACGrid",
    "FLIPConfig",
    # Eulerian
    "EulerianSolver",
    "StaggeredGrid",
    "VelocityField",
    # Shallow Water
    "ShallowWaterSolver",
    "HeightField",
    "TerrainBoundary",
    # Surface Reconstruction
    "MarchingCubes",
    "DensityField",
    "FluidSurface",
    # GPU
    "GPUFluidSolver",
    "GPUSpatialHash",
    "GPUFluidConfig",
]

# Phase 4: Fluids -- Architecture

## Status: 6 [x] 0 [~] 0 [-]

## Module: `engine/simulation/fluid/`

### Overview
Phase 4 provides comprehensive fluid simulation: SPH, PBF, FLIP/PIC/APIC, Eulerian grid, shallow water, and surface reconstruction. All 6 tasks are fully implemented as Python reference code. The TODO correctly lists all as [x].

---

### T-PHY-4.1: SPH Fluid Simulation

**Status**: [x] Complete.
**Location**: `engine/simulation/fluid/sph.py` (729 lines), `gpu_fluid.py` (560 lines)

**Current Implementation**:
- `SPHSolver`: Full SPH pipeline: neighbor build -> density -> pressure -> viscosity -> integrate
- Spatial hash grid for neighbor search
- Density computation with poly6 and spiky kernels
- Pressure force from state equation (ideal gas or Tait)
- Viscosity force (viscosity Laplacian)
- Surface tension (optional, cohesion-based)
- `GPUFluidSimulation` (gpu_fluid.py): Data preparation for WGSL compute path
- Configurable particle count

**Gap**: WGSL compute shader for GPU acceleration does not exist. Python GPU code prepares buffer data but no actual shader.

### T-PHY-4.2: PBF Fluid Simulation

**Status**: [x] Complete.
**Location**: `engine/simulation/fluid/pbf.py` (572 lines)

**Current Implementation**:
- `PBFSolver`: PBF pipeline: predict -> neighbor -> density constraint solve -> velocity update
- Density constraint solver with fixed iterations
- Vorticity confinement for detail preservation
- XSPH viscosity for stable particle interaction
- Better volume preservation than SPH

### T-PHY-4.3: FLIP/PIC/APIC

**Status**: [x] Complete.
**Location**: `engine/simulation/fluid/flip_pic.py` (678 lines)

**Current Implementation**:
- PIC: particle-to-grid transfer -> grid solve -> grid-to-particle transfer
- FLIP: transfer only delta velocity (less damping than PIC)
- APIC: affine velocity per particle (best of both FLIP + PIC)
- MAC grid representation with staggered velocities
- Pressure projection with CG solver (fixed iterations)

**Gap**: WGSL compute shaders for particle transfers not implemented.

### T-PHY-4.4: Eulerian Grid Fluids

**Status**: [x] Complete.
**Location**: `engine/simulation/fluid/eulerian.py` (488 lines)

**Current Implementation**:
- `EulerianFluidSolver`: MAC grid with staggered velocities
- Semi-Lagrangian advection
- Level set free surface tracking
- Pressure projection with CG solver

**Gap**: WGSL compute shaders not implemented.

### T-PHY-4.5: Shallow Water Simulation

**Status**: [x] Complete.
**Location**: `engine/simulation/fluid/shallow_water.py` (477 lines)

**Current Implementation**:
- `ShallowWaterSolver`: Height field representation
- Shallow water equations (2D)
- Wave propagation, reflection, refraction
- Fast enough for ocean-scale bodies

### T-PHY-4.6: Surface Reconstruction

**Status**: [x] Complete.
**Location**: `engine/simulation/fluid/surface_reconstruction.py` (472 lines)

**Current Implementation**:
- Screen-space surface rendering for SPH/PBF particles
- Marching cubes for grid-based fluids
- Anisotropic kernel for SPH surface enhancement
- WGSL compute path planned (data staged in Python)

---

## Key Design Decisions

- **Multiple fluid paradigms**: The Python implementation provides 4 distinct fluid simulation approaches (SPH, PBF, FLIP/PIC/APIC, Eulerian) plus shallow water. This covers the full spectrum from particle-based to grid-based fluids.
- **GPU staging in Python**: `gpu_fluid.py` (560 lines) mirrors `gpu_cloth.py` -- Python handles complex data structure preparation for planned WGSL compute shaders.
- **SPH as primary**: SPH is the highest-priority fluid algorithm (T-PHY-4.1), matching the TODO's designation. The implementation includes spatial hash neighbor search, multiple kernel options, and surface tension support.
- **FLIP/PIC/APIC continuum**: The implementation supports all three transfer schemes (PIC, FLIP, APIC) with a single API, allowing the best algorithm for each use case. APIC is the most modern and generally preferred.
- **Surface reconstruction duality**: Both screen-space (for SPH/PBF particles) and marching cubes (for grid-based fluids) are implemented, matching the dual nature of the fluid simulation approaches.

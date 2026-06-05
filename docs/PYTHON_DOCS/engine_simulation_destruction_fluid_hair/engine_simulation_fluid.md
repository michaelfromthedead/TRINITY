# Investigation: engine/simulation/fluid

## Summary
The fluid simulation module contains substantial, production-ready implementations of multiple fluid simulation techniques. SPH, PBF, FLIP/PIC, Eulerian, and shallow water solvers all include complete physics algorithms with proper kernel functions, spatial hashing, pressure projection, and surface reconstruction via marching cubes. GPU support is stub-only (abstract interface with CPU fallback).

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| __init__.py | 100 | Real | Comprehensive exports for all solvers |
| config.py | 327 | Real | Full constants, enums, material presets, solver configs |
| sph.py | 730 | Real | Complete SPH with poly6/spiky/viscosity kernels, spatial hash |
| pbf.py | 573 | Real | Full PBF with lambda computation, tensile correction, XSPH |
| eulerian.py | 489 | Real | Semi-Lagrangian advection, Jacobi pressure solve, MAC grid |
| flip_pic.py | 679 | Real | FLIP/PIC hybrid with MAC grid, RK2 advection |
| shallow_water.py | 478 | Real | Height field, Strang splitting, terrain boundaries |
| surface_reconstruction.py | 473 | Real | Marching cubes with proper edge tables, density splatting |
| gpu_fluid.py | 560 | Stub+Fallback | Abstract GPU interface with complete CPU reference impl |

**Total**: ~4,409 lines (estimated)

## Fluid Components

### SPH (Smoothed Particle Hydrodynamics)
- Poly6 kernel (density estimation)
- Spiky kernel (pressure gradient)
- Viscosity kernel (Laplacian)
- Cubic spline kernel (alternative)
- Spatial hash grid for O(N) neighbor search
- Tait equation of state for pressure
- Color field method for surface tension
- Boundary handling with velocity damping

### PBF (Position Based Fluids)
- Macklin & Muller 2013 algorithm
- Density constraint solving with Lagrange multipliers
- Tensile instability correction (s_corr)
- Vorticity confinement for energy preservation
- XSPH viscosity for coherent motion
- Jacobi-style iterative solver

### FLIP/PIC Hybrid
- MAC (Marker-And-Cell) staggered grid
- Particle-to-grid velocity transfer with trilinear weights
- Grid-to-particle transfer with PIC/FLIP blending
- Grid-based pressure projection (Jacobi iteration)
- RK2 (midpoint) particle advection

### Eulerian Grid Solver
- Staggered velocity field (u at x-faces, v at y-faces, w at z-faces)
- Semi-Lagrangian advection (backtracing)
- Jacobi pressure solver for divergence-free projection
- Cell type markers (fluid/solid/air)
- Trilinear interpolation for velocity sampling

### Shallow Water
- 2D height field representation
- Strang splitting (heights-velocities-heights)
- Terrain boundary with configurable elevation
- Bottom friction model
- CFL-based adaptive timestep
- Terrain presets (flat, slope, bowl)

### Surface Reconstruction
- Marching cubes isosurface extraction
- Complete edge table (256 configurations)
- Trilinear vertex interpolation
- Gradient-based normal computation
- Density field splatting from particles

### GPU (Stubs)
- Abstract interface for compute shaders
- Buffer layouts for GPU (SoA with vec4 alignment)
- Counting sort spatial hash algorithm
- Complete CPU fallback implementation

## Implementation

- Real SPH particles? **YES** - Full kernel implementations, proper physics
- Real Eulerian grid? **YES** - MAC grid, pressure projection, semi-Lagrangian
- Real surface reconstruction? **YES** - Marching cubes with complete edge table

## Verdict
**REAL IMPLEMENTATION**

This is a comprehensive, production-quality fluid simulation module. All core algorithms are fully implemented with proper physics, numerical stability handling, and performance considerations (spatial hashing, configurable substeps, CFL conditions).

## Evidence

### SPH Kernel (poly6 with proper normalization)
```python
@staticmethod
def poly6(r_sq: float, h: float) -> float:
    h_sq = h * h
    if r_sq > h_sq:
        return 0.0
    diff = h_sq - r_sq
    coeff = 315.0 / (64.0 * math.pi * h ** 9)
    return coeff * diff * diff * diff
```

### PBF Lambda Computation (proper constraint formulation)
```python
def compute_lambda(self, particle_index: int) -> float:
    C = self.compute_density_constraint(particle_index)
    if abs(C) < 1e-10:
        return 0.0
    # Gradient sum computation...
    lambda_val = -C / (grad_sum + self.epsilon)
    return np.clip(lambda_val, -max_lambda, max_lambda)
```

### Eulerian Pressure Projection (Jacobi iteration)
```python
def project_pressure(self, dt: float) -> None:
    for _ in range(PRESSURE_ITERATIONS):
        for i, j, k in fluid_cells:
            p_new[i, j, k] = (p_sum + dx * dx * rhs[i, j, k]) / n_fluid
    # Apply pressure gradient
    self.grid.velocity.u[i, j, k] -= scale * (p[i, j, k] - p[i-1, j, k])
```

### Marching Cubes Edge Table (complete 256-entry)
```python
EDGE_TABLE = [
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    # ... all 256 entries
]
```

### Material Presets (physically-based values)
```python
@classmethod
def water(cls) -> "FluidMaterial":
    return cls(
        rest_density=1000.0,  # kg/m^3
        viscosity=0.001,
        surface_tension=0.072,
        gas_constant=100.0,
    )
```

## Missing/Incomplete

1. **GPU Implementation**: Only abstract interface exists; actual compute shaders not implemented
2. **Anisotropic Kernels**: Mentioned in surface_reconstruction.py docstring but not implemented
3. **APIC**: Flag exists in config but APIC transfer not implemented
4. **Multigrid Pressure Solver**: Mentioned as option but not implemented
5. **MacCormack Advection**: Mentioned as option but not implemented

## Dependencies
- numpy (heavy use of NDArray, vectorized operations)
- Standard library (math, dataclasses, typing)

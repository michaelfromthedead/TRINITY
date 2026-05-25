# PHASE 2 ARCHITECTURE: Fluid System

**Scope**: ~3,504 lines across 6 files  
**Classification**: REAL (Production-Ready, GPU interface partial)

---

## System Overview

The fluid system provides multiple solver implementations for different simulation scenarios. Each solver optimizes for specific trade-offs between accuracy, stability, and performance.

---

## Solver Types

### 1. SPH - Smoothed Particle Hydrodynamics (sph.py, 729 lines)

**Purpose**: Lagrangian particle-based fluid with splashing detail.

**Kernel Functions**:
| Kernel | Formula | Use Case |
|--------|---------|----------|
| Poly6 | W = 315/(64*pi*h^9) * (h^2 - r^2)^3 | Density estimation |
| Spiky | W = 15/(pi*h^6) * (h - r)^3 | Pressure gradient |
| Viscosity | nabla^2 W = 45/(pi*h^6) * (h - r) | Viscosity force |
| Cubic Spline | Piecewise cubic | General purpose |

**Pressure Computation**:
- Tait equation of state: P = B * ((rho/rho_0)^gamma - 1)
- Stiffness B controls compressibility
- gamma typically 7 for water

**Spatial Hash Grid**:
- Cell size = kernel support radius h
- O(1) expected neighbor lookup
- Grid rebuilt each timestep

**Surface Tension**:
- Color field method: compute curvature from particle density gradient
- Force directed toward surface normals

### 2. FLIP/PIC Hybrid (flip_pic.py, 678 lines)

**Purpose**: Large-scale fluid volumes with stability.

**MAC Grid (Marker-And-Cell)**:
```
+--u--+--u--+
|     |     |
v  p  v  p  v
|     |     |
+--u--+--u--+
```
- Pressure (p): Cell centers
- Velocity u: Face centers (horizontal)
- Velocity v: Face centers (vertical)

**Algorithm Flow**:
1. Particle-to-grid transfer (P2G): Accumulate particle velocities to grid faces
2. Grid pressure solve: Jacobi iteration for incompressibility
3. Grid-to-particle transfer (G2P): Sample velocities back to particles
4. FLIP/PIC blend: velocity_new = alpha * velocity_PIC + (1-alpha) * velocity_FLIP

**FLIP/PIC Ratio**:
- PIC (alpha=1): Stable but loses detail (numerical diffusion)
- FLIP (alpha=0): Preserves detail but can be noisy
- Typical blend: alpha = 0.03 to 0.05

**Trilinear Interpolation**:
Used for both P2G (weighted scatter) and G2P (sample).

### 3. Position Based Fluids (pbf.py, 572 lines)

**Purpose**: Fast, GPU-friendly incompressible fluids.

**Constraint**:
```
C_i = rho_i / rho_0 - 1 = 0
```
Each particle's density should equal rest density.

**Lagrange Multiplier (lambda)**:
```
lambda_i = -C_i / (sum_j |grad_j C_i|^2 + epsilon)
```

**Position Correction**:
```
delta_p_i = (1/rho_0) * sum_j (lambda_i + lambda_j + s_corr) * grad W(p_i - p_j)
```

**Tensile Instability Correction (s_corr)**:
Artificial pressure to prevent particle clustering:
```
s_corr = -k * (W(p_i - p_j) / W(delta_q))^n
```

**Vorticity Confinement**:
Restores energy lost to numerical damping:
1. Compute vorticity: omega = curl(v)
2. Compute location vector: N = grad(|omega|) / |grad(|omega|)|
3. Apply force: f_vort = epsilon * (N cross omega)

**XSPH Viscosity**:
Smooths velocity field for coherent motion:
```
v_i = v_i + c * sum_j (v_j - v_i) * W(p_i - p_j)
```

### 4. Eulerian Solver (eulerian.py, 488 lines)

**Purpose**: Grid-based Navier-Stokes for stable large-scale simulation.

**Cell Types**:
- SOLID: Fixed boundary, no flow
- FLUID: Simulated region
- AIR: Free surface

**Semi-Lagrangian Advection**:
1. For each grid cell, trace backward along velocity field
2. Sample quantity (velocity, density) at origin point
3. Assign to current cell

**Benefits**: Unconditionally stable for advection (no CFL restriction on advection term).

**Pressure Projection**:
- Solve Poisson equation: nabla^2 p = -nabla dot v
- Subtract pressure gradient from velocity: v = v - grad(p)

**CFL Condition**:
```
dt < h / max(|v|)
```
Enforced by adaptive timestep or sub-stepping.

### 5. Shallow Water (shallow_water.py, 477 lines)

**Purpose**: 2D height-field simulation for oceans, rivers, puddles.

**State Variables**:
- h: Water height
- hu, hv: Momentum (height * velocity)

**Governing Equations** (Saint-Venant):
```
dh/dt + d(hu)/dx + d(hv)/dy = 0
d(hu)/dt + d(hu^2 + gh^2/2)/dx + d(huv)/dy = -gh * dB/dx
d(hv)/dt + d(huv)/dx + d(hv^2 + gh^2/2)/dy = -gh * dB/dy
```
Where B is terrain height.

**Strang Splitting**:
For 2D stability, alternate:
1. Solve x-direction (dt/2)
2. Solve y-direction (dt)
3. Solve x-direction (dt/2)

**Terrain Handling**:
- Bowl: Concave terrain, water pools at center
- Slope: Inclined plane, water flows downhill

---

## GPU Abstraction (gpu_fluid.py, 560 lines)

**Classification**: PARTIAL - Interface complete, GPU dispatch abstract

**Design Pattern**: Abstract interface with CPU fallback.

```python
class GPUFluidSolver(ABC):
    @abstractmethod
    def dispatch_build_grid(self) -> None: pass
    
    @abstractmethod
    def dispatch_compute_density(self) -> None: pass
    
    @abstractmethod
    def dispatch_compute_forces(self) -> None: pass
    
    @abstractmethod
    def dispatch_integrate(self) -> None: pass

class GPUFluidSolverStub(GPUFluidSolver):
    """Fully functional CPU reference implementation."""
    pass
```

**GPU Backend Requirements**:
- Compute shader dispatch for each abstract method
- SSBO for particle data
- Grid texture for spatial hash
- Atomic operations for P2G accumulation

---

## Component Architecture

```
fluid/
  |
  +-- sph.py (Lagrangian, detail-preserving)
  |     |
  |     +-- Poly6, Spiky, Viscosity, Cubic kernels
  |     +-- Spatial hash grid
  |     +-- Tait equation of state
  |
  +-- flip_pic.py (Hybrid, large-scale)
  |     |
  |     +-- MAC staggered grid
  |     +-- Jacobi pressure solver
  |     +-- Trilinear P2G/G2P
  |
  +-- pbf.py (PBD, GPU-friendly)
  |     |
  |     +-- Uses sph.py kernels
  |     +-- Lagrange multiplier constraints
  |     +-- Vorticity confinement
  |
  +-- gpu_fluid.py (GPU abstraction)
  |     |
  |     +-- Abstract dispatch interface
  |     +-- CPU fallback implementation
  |
  +-- eulerian.py (Grid, stable)
  |     |
  |     +-- Semi-Lagrangian advection
  |     +-- Poisson pressure solve
  |
  +-- shallow_water.py (Height-field, 2D)
        |
        +-- Saint-Venant equations
        +-- Strang splitting
        +-- Terrain boundaries
```

---

## Data Flow

### Lagrangian Solvers (SPH, PBF)

```
Particle Array
    |
    v
Build Spatial Hash
    |
    v
For each particle:
    +-- Find neighbors
    +-- Compute density
    +-- Compute pressure (SPH) or lambda (PBF)
    +-- Compute forces/corrections
    |
    v
Integrate positions
    |
    v
Handle boundaries
    |
    v
(Optional) Surface reconstruction
```

### FLIP/PIC

```
Particle Array
    |
    v
P2G: Accumulate to MAC grid
    |
    v
Pressure solve (Jacobi iteration)
    |
    v
Apply pressure gradient to grid velocity
    |
    v
G2P: Sample grid velocity to particles
    |
    v
Blend FLIP/PIC
    |
    v
Advect particles
```

### Eulerian

```
Velocity/Pressure Grid
    |
    v
Semi-Lagrangian advection
    |
    v
Add external forces
    |
    v
Pressure projection
    |
    v
(Optional) Level set advection for surface
```

---

## Configuration Points

| Parameter | Solver | Purpose |
|-----------|--------|---------|
| `kernel_radius` | SPH, PBF | Neighbor influence range |
| `rest_density` | All | Target incompressibility |
| `stiffness` | SPH | Tait equation B coefficient |
| `viscosity` | SPH, Eulerian | Velocity diffusion |
| `flip_ratio` | FLIP/PIC | Detail vs stability |
| `iteration_count` | PBF, Jacobi | Solver accuracy |
| `vorticity_epsilon` | PBF | Energy restoration strength |
| `gravity` | All | External acceleration |
| `timestep` | All | Integration step |
| `grid_resolution` | FLIP/PIC, Eulerian | Spatial discretization |

---

## Integration Points

### Input Interfaces
- `ParticleSource`: Emitter for spawning particles
- `BoundaryCollider`: Obstacle collision geometry

### Output Interfaces
- `SurfaceMeshCallback`: For marching cubes surface extraction
- `ParticleRenderCallback`: For point sprite rendering

### Shared Components
- SPH kernels reused by PBF
- Spatial hash shared between solvers
- Configuration via `config.py`

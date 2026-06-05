# Phase 3: Advanced Simulation -- Architecture

## Status: 7 [x] 0 [~] 2 [-]

## Module: `engine/simulation/`

### Overview
Phase 3 provides advanced simulation capabilities: destruction, cloth, hair, soft bodies (FEM, shape matching), and muscle simulation. All 9 tasks are implemented as Python reference code. The TODO lists 2 tasks as [-] (cloth and hair simulation) but these are in fact [x] -- fully implemented in the `cloth/` and `hair/` submodules.

---

### T-PHY-3.1: Destruction Pipeline

**Status**: [x] Complete.
**Location**: `engine/simulation/destruction/destruction_system.py` (839 lines)

**Current Implementation**:
- `DestructionSystem`: Damage accumulator per-destructible entity
- Configurable health pool
- Damage type system with type-specific multipliers
- Damage resistance per type
- Support graph evaluation (connectivity check)
- Fragment separation when support drops below threshold
- Debris spawning with velocity from fracture impulse
- LOD-aware debris management

**Gap**: No `@destructible` Foundation decorator. Uses float for damage values.

### T-PHY-3.2: Voronoi Fracture

**Status**: [x] Complete.
**Location**: `engine/simulation/destruction/fracture_voronoi.py` (942 lines)

**Current Implementation**:
- `FractureVoronoi`: Generate Voronoi cells from Poisson disk seed points
- Fracture mesh along cell boundaries
- Configurable cell count and distribution
- 3D cell generation

**Gap**: No `@fracture(pattern="voronoi")` decorator.

### T-PHY-3.3: Radial Fracture

**Status**: [x] Complete.
**Location**: `engine/simulation/destruction/fracture_radial.py` (725 lines)

**Current Implementation**:
- `FractureRadial`: Radial fracture pattern from impact point
- Configurable rings and radial spokes
- Velocity-dependent crack propagation

### T-PHY-3.4: Slice Fracture

**Status**: [x] Complete.
**Location**: `engine/simulation/destruction/fracture_slice.py` (827 lines)

**Current Implementation**:
- `FractureSlice`: Planar slice through mesh
- Configurable plane position and normal
- Cap generation for open meshes

### T-PHY-3.5: Cloth Simulation

**Status**: [x] Complete.
**Location**: `engine/simulation/cloth/` (7 files, 3,508 lines)

**Current Implementation**:
- `cloth_simulation.py` (663 lines): Triangle/quad mesh, particle/edge/triangle representation, PBD solver
- `cloth_constraints.py` (578 lines): Distance, bending, volume constraints
- `cloth_collision.py` (816 lines): Self-collision (face-based), body collider collision
- `cloth_wind.py` (546 lines): Wind force model, configurable drag coefficient
- `gpu_cloth.py` (572 lines): GPU data preparation for compute shader acceleration
- `config.py` (170 lines): Configuration parameters
- Supports PBD and XPBD solver options
- Attachment constraints (pin to rigid body)

**Gap**: No `@wind_affected` decorator. WGSL compute shaders not implemented (gpu_cloth.py prepares data but no shader exists).

### T-PHY-3.6: Hair Simulation

**Status**: [x] Complete.
**Location**: `engine/simulation/hair/` (6 files, 2,600 lines)

**Current Implementation**:
- `hair_simulation.py` (662 lines): Guide strand + interpolated render strand model
- `hair_constraints.py` (542 lines): FTL (Follow-The-Leader), DER (Discrete Elastic Rods), PBD strand options
- `hair_collision.py` (564 lines): Collision with body colliders
- `hair_lod.py` (470 lines): LOD system -- reduce guides with distance
- `config.py` (219 lines): Configuration parameters

**Gap**: WGSL compute shaders not implemented. No Foundation decorator integration.

### T-PHY-3.7: Soft Body FEM

**Status**: [x] Complete.
**Location**: `engine/simulation/softbody/fem_solver.py` (724 lines), `deformable_mesh.py` (582 lines)

**Current Implementation**:
- `FEMSolver`: Linear FEM with strain energy computation
- Corotational FEM: Rotation extraction for large deformations (polar decomposition)
- Force assembly from element stiffness matrices
- `DeformableMesh`: Tetrahedral mesh representation, element/node management
- Integration with rigid body collision

**Gap**: No `@simulation_domain("soft_body")` decorator. Uses float math.

### T-PHY-3.8: Shape Matching Soft Bodies

**Status**: [x] Complete.
**Location**: `engine/simulation/softbody/shape_matching.py` (621 lines)

**Current Implementation**:
- `ShapeMatching`: Geometric deformation via shape matching
- Cluster-based deformation regions
- Fast, suitable for jelly/goo/characters
- Deterministic output within floating-point tolerance

### T-PHY-3.9: Muscle Simulation

**Status**: [x] Complete.
**Location**: `engine/simulation/softbody/muscle.py` (590 lines)

**Current Implementation**:
- `MuscleSimulation`: Active contraction along fiber directions
- Volume preservation constraint
- Integration with FEM soft bodies
- Activation signal input

**TODO Correction**: The TODO marks T-PHY-3.9 as [-] but it is [x]. 590-line implementation exists at `engine/simulation/softbody/muscle.py`.

---

## Key Design Decisions

- **PBD/XPBD duality**: Cloth and soft bodies support both PBD (position-based for speed) and XPBD (compliance-based for accuracy) solver options, enabling quality/performance trade-offs per use case.
- **GPU staging in Python**: Both `gpu_cloth.py` and the fluid GPU code pre-compute data structures in Python for WGSL compute shader consumption. This is the correct separation -- Python handles complex data structure logic, shaders handle parallel computation.
- **FEM corotational**: The corotational FEM formulation corrects the linear FEM's artifact of non-physical deformation under large rotations, critical for production-quality soft bodies.
- **Hair LOD**: The 3-level LOD system (full -> reduced -> no simulation) mirrors game industry standards for hair rendering.

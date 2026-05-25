# GAPSET_19_PHYSICS -- Project Overview

## Physics System Architecture

The Trinity physics system is a comprehensive simulation layer implemented primarily in Python (~60,747 lines across 104 files in 12 submodules) with planned Rust/WGSL acceleration and Foundation determinism integration. It provides rigid body dynamics, collision detection, constraint solving, destruction, cloth, hair, soft bodies, fluids, vehicles, and character physics.

---

## Layer Architecture

### Layer 0: Math Foundation (omega/src/)
- Vec3, Quat, Mat4, Transform, RigidTransform
- Omega math library used by all physics modules
- **Planned**: Fixed16 (Q8.8), Fixed32 (Q16.16), FVec2/3/4, FQuat, FAngle, FTransform for deterministic simulation

### Layer 1: Core Physics (engine/simulation/physics/)
- **RigidBody**: BodyType (STATIC/KINEMATIC/DYNAMIC), BodyState, force accumulation, semi-implicit Euler integration
- **PhysicsWorld**: Body management, step(), collision dispatch, sleeping, queries
- **CollisionShapes**: AABB, Sphere, Box, Capsule, Mesh, Compound shapes
- **PhysicsMaterial**: Friction, restitution, density attributes
- **Queries**: Raycast, overlap, sweep (sphere/box/capsule)
- **Sleeping**: Island-based sleep management with thresholds
- **BodyFlags**: Per-body configuration flags

### Layer 2: Collision Detection (engine/simulation/collision/)
- **Broadphase**: 5 algorithms -- SAP (3-axis), Dynamic BVH (binary tree), Uniform Grid (spatial hash), Octree (sparse), Spatial Hash
- **Narrowphase**: 4 algorithms -- GJK (with simplex tracking), EPA (expanding polytope), SAT (15 test axes for OBB), MPR (Minkowski Portal Refinement)
- **ContactManifold**: Up to 4 contact points, persistent manifold with warm starting
- **CCD**: Speculative (expanded AABB), Sweep (TOI computation)
- **CollisionFilter**: Layer-based and group-based filtering
- **CollisionEvents**: Enter/stay/exit event dispatch

### Layer 3: Constraint Solver (engine/simulation/solver/)
- **PGS/Sequential Impulse**: Velocity-level solving, warm starting, Coulomb friction, restitution
- **TGS**: Temporal subdivision for improved stacking stability
- **XPBD**: Compliance-based position-level solving for cloth/soft bodies
- **IslandManager**: Disjoint set graph for parallel island solving
- **Jacobian**: Constraint Jacobian construction for all joint types

### Layer 4: Joints & Constraints (engine/simulation/constraints/)
- **8 Joint Types**: Fixed, Hinge (motor, limits), Slider (limits), Ball (cone limits), Spring (damped), Distance (min/max), D6 (6-DOF configurable)
- **JointMotors**: Velocity and position motor control
- **JointLimits**: Linear and angular limit specification
- **ContactConstraint**: Normal, friction, and rolling friction constraints

### Layer 5: Destruction (engine/simulation/destruction/)
- **Damage System**: Damage accumulator, 8 damage types, resistance multipliers
- **Voronoi Fracture**: Poisson disk seed distribution, cell decomposition
- **Radial Fracture**: Rings/spokes from impact point, velocity-dependent crack propagation
- **Slice Fracture**: Planar slice with cap generation
- **SupportGraph**: Connectivity evaluation, fragment separation
- **Debris**: Fragment spawning, lifetime management

### Layer 6: Cloth Simulation (engine/simulation/cloth/)
- **ClothMesh**: Triangle/quad mesh, particle/edge/triangle representation
- **PBD Solver**: Distance, bending, and volume constraints
- **XPBD Option**: Compliance-based formulation
- **Self-Collision**: Face-based collision detection
- **Wind**: Configurable wind force via `@wind_affected`
- **GPU Path**: Data preparation for compute shader acceleration (planned WGSL)

### Layer 7: Hair Simulation (engine/simulation/hair/)
- **Guide Strands**: FTL (Follow-The-Leader), DER (Discrete Elastic Rods), PBD strands
- **Interpolated Strands**: Render strands from guide interpolation
- **Collision**: Body collider interaction
- **LOD System**: Distance-based guide reduction

### Layer 8: Soft Bodies & Deformables (engine/simulation/softbody/)
- **FEM**: Linear FEM, corotational FEM (rotation extraction for large deformations)
- **Shape Matching**: Geometric deformation via matched clusters
- **PBD/XPBD**: Position-based soft body option
- **Muscle**: Active contraction along fiber directions, volume preservation
- **DeformableMesh**: Tetrahedral mesh representation

### Layer 9: Fluid Simulation (engine/simulation/fluid/)
- **SPH**: Spatial hash neighbor search, poly6/spiky kernels, pressure/viscosity/surface tension
- **PBF**: Density constraint solver, vorticity confinement, XSPH viscosity
- **FLIP/PIC/APIC**: MAC grid, CG pressure projection, affine particle velocities
- **Eulerian Grid**: Semi-Lagrangian advection, level set, CG projection
- **Shallow Water**: Height field, SWE equations, wave propagation
- **Surface Reconstruction**: Screen-space, marching cubes, anisotropic kernels
- **GPU Path**: Data preparation for WGSL compute shaders (planned)

### Layer 10: Vehicles (engine/simulation/vehicles/)
- **Wheeled**: Raycast wheel model, suspension, Pacejka tire model, 4 drive layouts (FWD/RWD/AWD/4WD), Ackermann steering, brakes
- **Aircraft**: Four forces (lift/drag/thrust/weight), control surfaces, stall model, landing gear
- **Watercraft**: Buoyancy sampling, hydrodynamic drag, propeller/sail forces
- **Tracked**: Continuous track model
- **Hover**: Air-cushion vehicle model
- **Drivetrain**: Engine torque curve, clutch, gearbox ratios, differential

### Layer 11: Character Physics (engine/simulation/character/)
- **CharacterController**: Kinematic, dynamic, and hybrid controller modes
- **GroundDetection**: Swept checks, ground state management
- **SlopeHandling**: Slope limits, stair navigation
- **PlatformHandling**: Moving platform attachment
- **MovementModes**: Ground, air, swim, climb modes
- **Ragdoll**: Passive (joint limits + motors) and active (PD controllers, balance)
- **PhysicsAnimationBlend**: Transition blending between physics and animation
- **CharacterInteraction**: Interaction constraints and forces

### Layer 12: ECS Components (engine/simulation/components/)
- Component wrappers for all physics entity types: RigidBody, Collider, Joint, Character, Cloth, Destruction, Fluid, Vehicle

---

## Evaluation Pipeline

```
1. PRE_PHYSICS: Input processing, force accumulation, CCD setup
2. Broadphase: Pair generation (SAP/BVH/grid/octree)
3. Narrowphase: GJK/EPA/SAT/MPR contact generation
4. Contact Manifold: Persistence, reduction, warm starting
5. Island Construction: Build solver islands
6. Constraint Setup: Jacobian construction
7. Solve: PGS/TGS/XPBD (fixed iterations)
8. Integrate: Semi-implicit Euler (position, rotation)
9. Sleeping: Threshold-based body deactivation
10. POST_PHYSICS: Contact callbacks, destruction, events
```

---

## Planned Missing Layers

- **Rust Backend**: `crates/physics/` -- SIMD-accelerated math, Fixed16/Fixed32 types, parallel solver, FFI bridge
- **WGSL Shaders**: `shaders/physics/` -- GPU compute for SPH, cloth, fluid, surface reconstruction
- **Foundation Integration**: `@simulation_domain`, `@substep`, `@solver_hint`, `@sleep_threshold`, `@continuous_collision`, `@physics_material`, `@joint`, `@destructible`, `@fracture`, `@wind_affected`, `@buoyancy` decorators
- **Determinism Infrastructure**: Fixed-point types, command queue, 13-phase tick, hierarchical checksums, accordion snapshots, replay, time-travel debugging
- **Network Physics**: Lockstep, rollback, server-authoritative + client prediction
- **Tests**: Unit and integration tests for all 12 submodules

## Directory Structure (Actual)

```
engine/simulation/
  __init__.py
  SIMULATION_CONTEXT.md        # 65,833-byte architecture reference
  physics/                     # 6,763 lines / 8 files
  collision/                   # 5,208 lines / 7 files
  solver/                      # 3,689 lines / 6 files
  constraints/                 # 4,932 lines / 10 files
  destruction/                 # 5,755 lines / 8 files
  cloth/                       # 3,508 lines / 7 files
  hair/                        # 2,600 lines / 6 files
  softbody/                    # 3,629 lines / 7 files
  fluid/                       # 4,401 lines / 9 files
  vehicles/                    # 6,604 lines / 10 files
  character/                   # 7,169 lines / 11 files
  components/                  # 4,376 lines / 8 files
```

## Dependency Chain

```
Platform (RHI/Window/Threading)
  -> Core (Memory/Math/ECS/Task)
    -> SIMULATION (THIS LAYER)
      -> Animation (skeletal, ragdoll blend)
        -> Rendering (transform data, debug draw)
```

## Determinism Requirement

All simulation state is planned to use Fixed16 (Q8.8) or Fixed32 (Q16.16) for reproducibility. Currently all code uses standard Python float math. See `engine/determinism/DETERMINISM_CONTEXT.md` for fixed-point math, snapshot system, checksums, and replay architecture.

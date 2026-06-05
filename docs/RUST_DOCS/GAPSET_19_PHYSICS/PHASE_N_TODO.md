# GAPSET_19_PHYSICS -- Task List

> **TASK_ID Format**: T-PHY-{PHASE}.{N}
> **Priority Scale**: CRITICAL > HIGH > MEDIUM > LOW
> **Effort**: S (small, <1d), M (medium, 1-3d), L (large, 3-10d), XL (extra large, 10-30d)

---

## Phase 1: Foundation (CRITICAL -- blocks all other phases)

### T-PHY-1.1: Implement Fixed16 Q8.8 Type (Rust)
**Priority**: CRITICAL
**Effort**: M
**Dependencies**: None
**Acceptance Criteria**:
- [ ] `Fixed16` struct wraps i16 with Q8.8 format
- [ ] add, sub, mul, div implemented with overflow detection in debug
- [ ] Saturation clamping on overflow (configurable per-instance)
- [ ] Conversion from/to f32, i32
- [ ] Comparison operators (PartialOrd, Ord)
- [ ] `const` construction from float/integer literals
- [ ] Unit tests covering edge cases (overflow, underflow, zero, negative)

### T-PHY-1.2: Implement Fixed32 Q16.16 Type (Rust)
**Priority**: CRITICAL
**Effort**: M
**Dependencies**: T-PHY-1.1
**Acceptance Criteria**:
- [ ] `Fixed32` struct wraps i32 with Q16.16 format
- [ ] All arithmetic operations with overflow detection
- [ ] Saturation clamping
- [ ] Conversion from/to f32, f64, i32, Fixed16
- [ ] Comparison operators
- [ ] Unit tests

### T-PHY-1.3: Implement Fixed-Point Vector Types (Rust)
**Priority**: CRITICAL
**Effort**: M
**Dependencies**: T-PHY-1.2
**Acceptance Criteria**:
- [ ] `FVec2<T>`, `FVec3<T>`, `FVec4<T>` generic over Fixed16/Fixed32
- [ ] Component-wise add, sub, mul, div
- [ ] dot(), cross() (3D only), length_squared(), length()
- [ ] normalize() with fallback for zero-length vectors
- [ ] lerp() for interpolation
- [ ] Unit tests for all operations

### T-PHY-1.4: Implement Fixed-Point Rotation Types (Rust)
**Priority**: CRITICAL
**Effort**: M
**Dependencies**: T-PHY-1.3
**Acceptance Criteria**:
- [ ] `FQuat` with identity, mul, conjugate, inverse, rotate_vector
- [ ] `FAngle` with sin, cos, tan, asin, acos, atan2
- [ ] 4096-entry sin/cos lookup table covering [0, pi/2]
- [ ] CORDIC atan2 (16 iterations)
- [ ] Conversion from/to axis-angle, euler angles
- [ ] Unit tests

### T-PHY-1.5: Implement Fixed-Point Transform (Rust)
**Priority**: CRITICAL
**Effort**: S
**Dependencies**: T-PHY-1.3, T-PHY-1.4
**Acceptance Criteria**:
- [ ] `FTransform` with position (FVec3), rotation (FQuat), scale (FVec3)
- [ ] compose, decompose, transform_point, transform_vector
- [ ] Inverse transform
- [ ] Interpolation (lerp position, slerp rotation, lerp scale)

### T-PHY-1.6: Implement Deterministic PCG RNG (Rust)
**Priority**: CRITICAL
**Effort**: M
**Dependencies**: None
**Acceptance Criteria**:
- [ ] PCG-XSH-RR variant with u64 state + u64 increment
- [ ] next_u32(), next_u64(), next_f32(), next_f64() methods
- [ ] `fork()` -- create child RNG deterministically from parent
- [ ] `seed_from_u64()` -- deterministic seeding
- [ ] `seed_from_entity()` -- fork from world_rng using EntityID
- [ ] Cross-platform equivalence test (same seed -> same sequence on all platforms)
- [ ] Unit tests for statistical properties (chi-squared, birthday spacings)

### T-PHY-1.7: Implement Entity ID System (Rust)
**Priority**: HIGH
**Effort**: S
**Dependencies**: None
**Acceptance Criteria**:
- [ ] `EntityID` wraps u64 (48-bit index + 16-bit generation)
- [ ] `NULL_ENTITY` constant
- [ ] index(), generation() accessors
- [ ] Deterministic allocation via command queue (not free-list)
- [ ] Serialization/deserialization to/from [u8; 8]

### T-PHY-1.8: Implement Command Queue (Rust)
**Priority**: CRITICAL
**Effort**: M
**Dependencies**: T-PHY-1.7
**Acceptance Criteria**:
- [ ] `Command` struct with tick, entity_id, command_type, sequence, payload
- [ ] Command sort by (tick, entity_id, command_type, sequence)
- [ ] Binary encode/decode for channel transfer
- [ ] Validation (tick monotonicity, entity_id existence for mutate/despawn)
- [ ] Deterministic execution order guaranteed

### T-PHY-1.9: Implement Simulation Decorators (Python -- Trinity Pattern)
**Priority**: HIGH
**Effort**: M
**Dependencies**: Trinity pattern libraries
**Acceptance Criteria**:
- [ ] `@simulation_domain(domain="physics")` implemented
- [ ] `@deterministic` implemented
- [ ] `@substep(count=4)` implemented
- [ ] `@solver_hint(preference="pgs"|"tgs"|"xpbd")` implemented
- [ ] `@sleep_threshold` implemented
- [ ] `@continuous_collision` implemented
- [ ] `@buoyancy`, `@wind_affected` implemented
- [ ] `@fixed`, `@time_scale`, `@pausable`, `@rewindable` implemented
- [ ] `@spatial`, `@partitioned` implemented
- [ ] All decorators register with simulation domain registry

---

## Phase 2: Core Physics (HIGH)

### T-PHY-2.1: Implement Rigid Body Components (Python + Rust)
**Priority**: HIGH
**Effort**: M
**Dependencies**: T-PHY-1.3, T-PHY-1.5, T-PHY-1.9
**Acceptance Criteria**:
- [ ] `PhysicsBody` component with mass, inertia, damping
- [ ] `MotionState` component with position, rotation, velocities
- [ ] `ForceAccumulator` component
- [ ] Three body types: static, kinematic, dynamic
- [ ] Gravity application
- [ ] Semi-implicit Euler integration
- [ ] Register with ECS archetype storage

### T-PHY-2.2: Implement Sweep and Prune Broadphase (Rust)
**Priority**: HIGH
**Effort**: M
**Dependencies**: T-PHY-1.3, T-PHY-2.1
**Acceptance Criteria**:
- [ ] SAP on AABB min/max in 3 axes
- [ ] O(n log n) sort, O(n) sweep
- [ ] Deterministic pair ordering (stable sort by entity pair)
- [ ] Handles overlapping pairs correctly
- [ ] Unit tests with known configurations

### T-PHY-2.3: Implement Dynamic BVH Broadphase (Rust)
**Priority**: MEDIUM
**Effort**: L
**Dependencies**: T-PHY-1.3, T-PHY-2.1
**Acceptance Criteria**:
- [ ] Binary tree of AABB nodes
- [ ] Insert, remove, update operations
- [ ] Raycast and frustum query support
- [ ] Deterministic tree structure (insertion order fixed)
- [ ] Refit on motion

### T-PHY-2.4: Implement Uniform Grid Broadphase (Rust)
**Priority**: MEDIUM
**Effort**: M
**Dependencies**: T-PHY-1.3, T-PHY-2.1
**Acceptance Criteria**:
- [ ] Sparse grid with cell size configurable
- [ ] O(1) cell lookup via spatial hash
- [ ] Deterministic cell/entity sort order
- [ ] Dynamic cell allocation

### T-PHY-2.5: Implement Octree Broadphase (Rust)
**Priority**: LOW
**Effort**: L
**Dependencies**: T-PHY-1.3, T-PHY-2.1
**Acceptance Criteria**:
- [ ] Sparse octree with configurable depth
- [ ] Insert, remove, update
- [ ] O(log n) query
- [ ] Deterministic traversal

### T-PHY-2.6: Implement GJK Narrowphase (Rust)
**Priority**: HIGH
**Effort**: M
**Dependencies**: T-PHY-1.3, T-PHY-2.1
**Acceptance Criteria**:
- [ ] Support function API for convex shapes
- [ ] GJK distance calculation, fixed max iterations (32)
- [ ] Simplex tracking (point, line, triangle, tetrahedron)
- [ ] Returns closest points + distance
- [ ] Deterministic: same input -> same output regardless of platform

### T-PHY-2.7: Implement EPA Narrowphase (Rust)
**Priority**: HIGH
**Effort**: M
**Dependencies**: T-PHY-2.6
**Acceptance Criteria**:
- [ ] Expanding Polytope algorithm from GJK simplex
- [ ] Penetration depth + contact normal
- [ ] Fixed max iterations (64)
- [ ] Deterministic output

### T-PHY-2.8: Implement SAT Narrowphase (Rust)
**Priority**: MEDIUM
**Effort**: M
**Dependencies**: T-PHY-1.3
**Acceptance Criteria**:
- [ ] Separating Axis Theorem for oriented boxes
- [ ] 15 test axes (3 face + 6 edge for OBB-OBB)
- [ ] Contact point generation
- [ ] Optimization: cache axes for common shapes

### T-PHY-2.9: Implement MPR Narrowphase (Rust)
**Priority**: LOW
**Effort**: M
**Dependencies**: T-PHY-2.6
**Acceptance Criteria**:
- [ ] Minkowski Portal Refinement
- [ ] Penetration depth + normal
- [ ] Faster than EPA for deep penetration
- [ ] Fixed max iterations

### T-PHY-2.10: Implement Contact Manifold (Rust)
**Priority**: HIGH
**Effort**: M
**Dependencies**: T-PHY-2.6, T-PHY-2.7
**Acceptance Criteria**:
- [ ] Up to 4 contact points per pair
- [ ] Persistent manifold with warm starting
- [ ] Contact point reduction (keep deepest, most stable)
- [ ] Deterministic contact ordering by entity pair ID

### T-PHY-2.11: Implement CCD Modes (Rust)
**Priority**: MEDIUM
**Effort**: M
**Dependencies**: T-PHY-2.1, T-PHY-2.6
**Acceptance Criteria**:
- [ ] Speculative CCD: expand AABB by velocity * dt
- [ ] Sweep CCD: swept shape TOI computation
- [ ] Configurable per-body via `@continuous_collision`
- [ ] Deterministic TOI computation

### T-PHY-2.12: Implement PGS/Sequential Impulse Solver (Rust)
**Priority**: HIGH
**Effort**: L
**Dependencies**: T-PHY-2.10
**Acceptance Criteria**:
- [ ] Velocity-level constraint solving (Sequential Impulse)
- [ ] Fixed iteration count (default 8, configurable)
- [ ] Warm starting from previous tick's impulses
- [ ] Friction: Coulomb model with static/dynamic coefficients
- [ ] Restitution: coefficient of restitution
- [ ] Deterministic: fixed iterations, sorted constraint order

### T-PHY-2.13: Implement TGS Solver (Rust)
**Priority**: MEDIUM
**Effort**: L
**Dependencies**: T-PHY-2.12
**Acceptance Criteria**:
- [ ] Temporal subdivision of solver step
- [ ] Better stacking stability than PGS
- [ ] Fixed substep count

### T-PHY-2.14: Implement XPBD Solver (Rust)
**Priority**: MEDIUM
**Effort**: M
**Dependencies**: T-PHY-2.1
**Acceptance Criteria**:
- [ ] Compliance-based constraint formulation
- [ ] Position-level solving with compliance/stiffness parameters
- [ ] Fixed iteration count
- [ ] Used for cloth, soft body constraints

### T-PHY-2.15: Implement Joint Types (Rust)
**Priority**: HIGH
**Effort**: L
**Dependencies**: T-PHY-2.12
**Acceptance Criteria**:
- [ ] Fixed joint
- [ ] Hinge joint (single-axis rotation, limits, motor)
- [ ] Slider joint (single-axis translation, limits)
- [ ] Ball joint (3-axis rotation, cone limits)
- [ ] Spring joint (damped spring, rest length)
- [ ] Distance joint (min/max distance)
- [ ] D6 joint (6-DOF configurable)
- [ ] All joints deterministic: fixed iterations, sorted order

---

## Phase 3: Advanced Simulation (MEDIUM)

### T-PHY-3.1: Implement Destruction Pipeline (Rust)
**Priority**: MEDIUM
**Effort**: L
**Dependencies**: T-PHY-2.1, T-PHY-2.12
**Acceptance Criteria**:
- [ ] Damage accumulator per-destructible entity
- [ ] Damage type system with type-specific multipliers
- [ ] Damage resistance per type
- [ ] Support graph evaluation (connectivity check)
- [ ] Fragment separation when support drops below threshold
- [ ] Debris spawning with velocity from fracture impulse

### T-PHY-3.2: Implement Voronoi Fracture (Rust)
**Priority**: MEDIUM
**Effort**: M
**Dependencies**: T-PHY-3.1
**Acceptance Criteria**:
- [ ] Generate Voronoi cells from seed points (Poisson disk distribution)
- [ ] Fracture mesh along cell boundaries
- [ ] Configurable cell count and distribution
- [ ] Deterministic seed point generation

### T-PHY-3.3: Implement Radial Fracture (Rust)
**Priority**: LOW
**Effort**: M
**Dependencies**: T-PHY-3.1
**Acceptance Criteria**:
- [ ] Radial fracture pattern from impact point
- [ ] Configurable rings and radial spokes
- [ ] Velocity-dependent crack propagation

### T-PHY-3.4: Implement Slice Fracture (Rust)
**Priority**: LOW
**Effort**: S
**Dependencies**: T-PHY-3.1
**Acceptance Criteria**:
- [ ] Planar slice through mesh
- [ ] Configurable plane position and normal
- [ ] Cap generation for open meshes

### T-PHY-3.5: Implement Cloth Simulation (Rust)
**Priority**: MEDIUM
**Effort**: L
**Dependencies**: T-PHY-1.3, T-PHY-2.12, T-PHY-2.14
**Acceptance Criteria**:
- [ ] Triangle/quad mesh representation
- [ ] PBD solver for cloth (distance + bending constraints)
- [ ] XPBD solver option (compliance-based)
- [ ] Self-collision (face-based)
- [ ] Attachment constraints (pin to rigid body)
- [ ] Wind force via `@wind_affected`
- [ ] Deterministic: fixed iterations, sorted constraint order

### T-PHY-3.6: Implement Hair Simulation (Rust)
**Priority**: MEDIUM
**Effort**: L
**Dependencies**: T-PHY-1.3
**Acceptance Criteria**:
- [ ] Guide strand + interpolated render strand model
- [ ] FTL (Follow-The-Leader) strand simulation
- [ ] DER (Discrete Elastic Rods) for bending/twisting
- [ ] PBD strand option
- [ ] LOD system: reduce guides with distance
- [ ] Collision with body colliders
- [ ] Deterministic: fixed iterations, sorted strand order

### T-PHY-3.7: Implement Soft Body FEM (Rust)
**Priority**: MEDIUM
**Effort**: XL
**Dependencies**: T-PHY-1.3, T-PHY-2.1
**Acceptance Criteria**:
- [ ] Tetrahedral mesh representation
- [ ] Linear FEM with strain energy computation
- [ ] Corotational FEM (rotation extraction for large deformations)
- [ ] Force assembly from element stiffness matrices
- [ ] Integration with rigid body collision

### T-PHY-3.8: Implement Shape Matching Soft Bodies (Rust)
**Priority**: LOW
**Effort**: M
**Dependencies**: T-PHY-1.3
**Acceptance Criteria**:
- [ ] Geometric deformation via shape matching
- [ ] Cluster-based deformation regions
- [ ] Fast, suitable for jelly/goo/characters
- [ ] Deterministic output

### T-PHY-3.9: Implement Muscle Simulation (Rust)
**Priority**: LOW
**Effort**: L
**Dependencies**: T-PHY-3.7
**Acceptance Criteria**:
- [ ] Active contraction along fiber directions
- [ ] Volume preservation constraint
- [ ] Integration with FEM soft bodies
- [ ] Activation signal input

---

## Phase 4: Fluids (MEDIUM -- SPH highest priority)

### T-PHY-4.1: Implement SPH Fluid Simulation (Rust + WGSL)
**Priority**: MEDIUM
**Effort**: XL
**Dependencies**: T-PHY-1.3
**Acceptance Criteria**:
- [ ] SPH pipeline: neighbor build -> density -> pressure -> viscosity -> integrate
- [ ] Spatial hash grid for neighbor search (deterministic ordering)
- [ ] Density computation with kernel (poly6 or spiky)
- [ ] Pressure force from state equation
- [ ] Viscosity force (viscosity Laplacian)
- [ ] Surface tension (optional)
- [ ] WGSL compute shader for GPU acceleration (Phase 1: CPU reference, Phase 2: GPU)
- [ ] Fixed particle count, deterministic iteration
- [ ] 10K+ particles at 60Hz
- [ ] Python defines for particle configuration

### T-PHY-4.2: Implement PBF Fluid Simulation (Rust + WGSL)
**Priority**: LOW
**Effort**: L
**Dependencies**: T-PHY-4.1
**Acceptance Criteria**:
- [ ] PBF pipeline: predict -> neighbor -> density constraint solve -> velocity update
- [ ] Density constraint solver (fixed iterations)
- [ ] Vorticity confinement and XSPH viscosity
- [ ] Better volume preservation than SPH

### T-PHY-4.3: Implement FLIP/PIC/APIC (Rust + WGSL)
**Priority**: LOW
**Effort**: XL
**Dependencies**: T-PHY-1.3
**Acceptance Criteria**:
- [ ] PIC: particle-to-grid -> grid solve -> grid-to-particle
- [ ] FLIP: transfer only delta, less damping
- [ ] APIC: affine velocity per particle
- [ ] MAC grid representation
- [ ] Pressure projection (CG solver, fixed iterations)
- [ ] GPU compute for particle transfers

### T-PHY-4.4: Implement Eulerian Grid Fluids (Rust + WGSL)
**Priority**: LOW
**Effort**: XL
**Dependencies**: T-PHY-1.3
**Acceptance Criteria**:
- [ ] MAC grid with staggered velocities
- [ ] Semi-Lagrangian advection (BFECC optional)
- [ ] Level set free surface tracking
- [ ] Pressure projection with CG
- [ ] WGSL compute shaders

### T-PHY-4.5: Implement Shallow Water Simulation (Rust + WGSL)
**Priority**: LOW
**Effort**: M
**Dependencies**: None
**Acceptance Criteria**:
- [ ] Height field representation
- [ ] Shallow water equations (2D)
- [ ] Wave propagation, reflection, refraction
- [ ] Fast enough for ocean-scale bodies

### T-PHY-4.6: Implement Surface Reconstruction (Rust + WGSL)
**Priority**: MEDIUM
**Effort**: L
**Dependencies**: T-PHY-4.1
**Acceptance Criteria**:
- [ ] Screen-space surface rendering for SPH/PBF particles
- [ ] Marching cubes for grid-based fluids
- [ ] Anisotropic kernel for SPH surface
- [ ] WGSL compute shader for reconstruction

---

## Phase 5: Specialized (MEDIUM)

### T-PHY-5.1: Implement Wheeled Vehicle Physics (Rust)
**Priority**: MEDIUM
**Effort**: L
**Dependencies**: T-PHY-2.1, T-PHY-2.12
**Acceptance Criteria**:
- [ ] Wheel model: raycast, wheel collider (optional), suspension spring/damper
- [ ] Pacejka tire model (lookup table for deterministic eval)
- [ ] Drivetrain: engine torque curve, clutch, gearbox ratios, differential
- [ ] 4 drive layouts: FWD, RWD, AWD, 4WD
- [ ] Steering geometry: Ackermann
- [ ] Brakes: per-wheel brake torque
- [ ] Deterministic across platforms

### T-PHY-5.2: Implement Aircraft Physics (Rust)
**Priority**: LOW
**Effort**: L
**Dependencies**: T-PHY-2.1
**Acceptance Criteria**:
- [ ] Four forces: lift, drag, thrust, weight
- [ ] Lift/drag coefficient curves (angle of attack lookup)
- [ ] Control surfaces: aileron, elevator, rudder, flaps
- [ ] Stall model
- [ ] Landing gear
- [ ] Deterministic across platforms

### T-PHY-5.3: Implement Watercraft Physics (Rust)
**Priority**: LOW
**Effort**: M
**Dependencies**: T-PHY-2.1
**Acceptance Criteria**:
- [ ] Buoyancy evaluation at sample points (Archimedes principle)
- [ ] Hydrodynamic drag
- [ ] Propeller thrust, water jet propulsion
- [ ] Sail force model
- [ ] Deterministic across platforms

### T-PHY-5.4: Implement Ragdoll Physics (Rust)
**Priority**: MEDIUM
**Effort**: M
**Dependencies**: T-PHY-2.1, T-PHY-2.15
**Acceptance Criteria**:
- [ ] Passive ragdoll: joint limit + motor constraints
- [ ] Active ragdoll: PD controller per joint tracking target
- [ ] Balance controller (center-of-mass tracking)
- [ ] Transition from animation to ragdoll (blend)
- [ ] Transition from ragdoll to animation (get-up)
- [ ] Deterministic output for replay

### T-PHY-5.5: Implement Character Controllers (Rust)
**Priority**: MEDIUM
**Effort**: M
**Dependencies**: T-PHY-2.1
**Acceptance Criteria**:
- [ ] Kinematic controller: push rigid bodies from contacts
- [ ] Capsule controller: simplified collision + movement
- [ ] Step-up, step-down handling
- [ ] Slope limit, stair navigation
- [ ] Deterministic movement

---

## Phase 6: Determinism Integration (HIGH)

### T-PHY-6.1: Implement SimulationWorld (Rust)
**Priority**: CRITICAL
**Effort**: L
**Dependencies**: T-PHY-1.6, T-PHY-1.8
**Acceptance Criteria**:
- [ ] SimulationWorld struct with ArchetypeStorage, SimRNG, Tick, CommandQueue
- [ ] tick() method executing 13 phases in order
- [ ] snapshot() generating accordion snapshot
- [ ] restore(snapshot) for rollback
- [ ] checksum() computing hierarchical checksum
- [ ] fork() creating branch simulation

### T-PHY-6.2: Implement 13-Phase Tick Executor (Rust)
**Priority**: CRITICAL
**Effort**: L
**Dependencies**: T-PHY-6.1, subsystem implementations (Phase 2-5)
**Acceptance Criteria**:
- [ ] Phase 1: Input receive/validate/convert
- [ ] Phase 2: Command sort and execute
- [ ] Phases 3-13: System execution in fixed order
- [ ] No cross-phase read/write violations
- [ ] Snapshot after Phase 13
- [ ] Deterministic ordering of all operations

### T-PHY-6.3: Implement Hierarchical Checksums (Rust)
**Priority**: CRITICAL
**Effort**: M
**Dependencies**: T-PHY-6.1
**Acceptance Criteria**:
- [ ] Entity-level checksum (hash of component data)
- [ ] Archetype-level checksum (XOR of entity checksums)
- [ ] Chunk-level checksum (XOR of archetype checksums)
- [ ] World-level checksum (XOR of chunk checksums)
- [ ] O(log N) divergence finding via XOR subtraction
- [ ] Bounded computation time per tick
- [ ] Per-tick storage in snapshot

### T-PHY-6.4: Implement Accordion Snapshot Strategy (Rust)
**Priority**: HIGH
**Effort**: M
**Dependencies**: T-PHY-6.1
**Acceptance Criteria**:
- [ ] Dense region: every tick, configurable N (default 10-30)
- [ ] Sparse region: f(n) = n^1.5 spacing
- [ ] Keyframe anchors: every K ticks (default 300-600), never pruned
- [ ] Activity-based density: increase density around active entities
- [ ] Memory-bounded: configurable maximum total snapshots

### T-PHY-6.5: Implement Replay System (Rust)
**Priority**: HIGH
**Effort**: L
**Dependencies**: T-PHY-6.1, T-PHY-6.3, T-PHY-6.4
**Acceptance Criteria**:
- [ ] Input recording: store inputs per tick to buffer
- [ ] Replay from buffer: recreate simulation tick-by-tick from stored inputs
- [ ] File format: header + per-tick input stream (~2-5 MB/10min)
- [ ] 5 playback modes: normal, fast-forward, slow-motion, reverse, pause
- [ ] Checksum verification: compare computed checksum to stored checksum
- [ ] Mismatch detection and reporting

### T-PHY-6.6: Implement Time-Travel Debugging (Python + Rust)
**Priority**: LOW
**Effort**: L
**Dependencies**: T-PHY-6.4, T-PHY-6.5
**Acceptance Criteria**:
- [ ] Python API: step_back(n_ticks)
- [ ] Python API: goto_tick(tick_number)
- [ ] Python API: watch(entity_id, field_name, condition)
- [ ] Breakpoints on checksum mismatch
- [ ] State diff visualization

### T-PHY-6.7: Implement Lockstep Networking (Rust)
**Priority**: LOW
**Effort**: L
**Dependencies**: T-PHY-6.1
**Acceptance Criteria**:
- [ ] Deterministic lockstep: all peers simulate from same inputs
- [ ] Input synchronization protocol
- [ ] Mismatch detection via checksum exchange
- [ ] Peer timeout and recovery

### T-PHY-6.8: Implement Rollback Networking (Rust)
**Priority**: LOW
**Effort**: L
**Dependencies**: T-PHY-6.4
**Acceptance Criteria**:
- [ ] Predict-ahead simulation
- [ ] Rollback on input confirmation mismatch
- [ ] Re-simulate from confirmed state
- [ ] Smooth visual interpolation of rollback

### T-PHY-6.9: Implement Server Authoritative + Prediction (Rust)
**Priority**: LOW
**Effort**: L
**Dependencies**: T-PHY-6.4
**Acceptance Criteria**:
- [ ] Server state as authoritative truth
- [ ] Client-side prediction with local simulation
- [ ] Reconciliation on server state receipt
- [ ] Smooth interpolation

### T-PHY-6.10: Implement Simulation-Presentation Boundary (Python -- Trinity Pattern)
**Priority**: CRITICAL
**Effort**: M
**Dependencies**: T-PHY-1.9, T-PHY-6.1
**Acceptance Criteria**:
- [ ] Simulation components marked `@simulation` -- fixed-point only, no direct writes from presentation
- [ ] Presentation components marked `@presentation` -- floats allowed, no reads by simulation
- [ ] `@command` decorator marks methods that can mutate simulation state
- [ ] `@deterministic_rng` decorator marks methods that access deterministic RNG
- [ ] Descriptor-enforced immutability: presentation cannot write to simulation components
- [ ] Bridge channels enforce boundary at runtime

---

## Task Summary

| Phase | Tasks | CRITICAL | HIGH | MEDIUM | LOW | Total Effort |
|-------|-------|----------|------|--------|-----|-------------|
| 1. Foundation | 9 | 6 | 3 | 0 | 0 | ~18 days |
| 2. Core Physics | 15 | 0 | 8 | 5 | 2 | ~52 days |
| 3. Advanced Simulation | 9 | 0 | 0 | 6 | 3 | ~48 days |
| 4. Fluids | 6 | 0 | 0 | 3 | 3 | ~45 days |
| 5. Specialized | 5 | 0 | 0 | 4 | 1 | ~22 days |
| 6. Determinism Integration | 10 | 3 | 3 | 0 | 4 | ~42 days |
| **Total** | **54** | **9** | **14** | **18** | **13** | **~227 days** |

**Priority Execution Order**: Phase 1 -> Phase 6 (determinism scaffolding) -> Phase 2 -> Phase 3 + Phase 5 -> Phase 4

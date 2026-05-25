# Archaeological Investigation: engine/simulation/{character,cloth,collision}

**Date**: 2026-05-22  
**Scope**: 17 files, approximately 13,308 lines  
**Classification**: 16 REAL implementations, 1 PARTIAL STUB

---

## Executive Summary

| Module | Files | Lines | Classification |
|--------|-------|-------|----------------|
| character | 6 | ~4,614 | ALL REAL |
| cloth | 5 | ~3,175 | 4 REAL, 1 PARTIAL STUB |
| collision | 6 | ~5,349 | ALL REAL |

The engine/simulation subsystem contains production-grade physics implementations with industry-standard algorithms (GJK, EPA, SAT, Position-Based Dynamics, PD controllers). The only incomplete component is GPU cloth simulation (`gpu_cloth.py`), which provides buffer definitions and shader templates but explicitly warns it does not simulate.

---

## Module: engine/simulation/character (6 files, ALL REAL)

### character_controller.py (879 lines) - REAL

**Evidence**:
- Complete `Vector3`, `Quaternion`, `Transform` math primitives with full operator overloading
- `PhysicsWorldInterface` abstract base class defining sweep/overlap contracts
- `CharacterController` with `move_and_slide()` implementing iterative collision resolution
- Ground detection with slope handling and step climbing
- Platform attachment for moving platforms

**Key Algorithms**:
- Move-and-slide with up to 4 collision iterations
- Ground snap with configurable snap distance
- Slope limiting based on surface normal dot product

### movement_modes.py (692 lines) - REAL

**Evidence**:
- `MovementMode` enum with 14 states: WALKING, RUNNING, SPRINTING, CROUCHING, PRONE, SWIMMING, CLIMBING, FLYING, FALLING, LADDERING, SLIDING, VAULTING, HANGING, CUSTOM
- `MovementTransition` rules defining valid state changes with requirements
- `MovementConfig` with complete physics parameters per mode
- `MovementModeManager` state machine with stamina system

**Key Algorithms**:
- Transition validation with requirement checking
- Stamina-gated mode transitions (e.g., sprinting requires > 0 stamina)
- Mode-specific physics parameter application

### ragdoll.py (776 lines) - REAL

**Evidence**:
- `BodyPartType` enum with 20 parts: HEAD, NECK, UPPER_SPINE, MIDDLE_SPINE, LOWER_SPINE, LEFT/RIGHT_CLAVICLE, LEFT/RIGHT_UPPER_ARM, LEFT/RIGHT_LOWER_ARM, LEFT/RIGHT_HAND, PELVIS, LEFT/RIGHT_UPPER_LEG, LEFT/RIGHT_LOWER_LEG, LEFT/RIGHT_FOOT
- `BodyPartConfig` with mass, friction, collision shape parameters
- `JointConfig` with swing/twist limits
- `create_default_humanoid_setup()` returning complete skeleton definition

**Key Algorithms**:
- Physics body creation with compound shapes
- Joint constraint setup with angular limits
- Activation/deactivation with motor disabling

### active_ragdoll.py (678 lines) - REAL

**Evidence**:
- `PDController` class computing torque from quaternion error with `kp`/`kd` gains
- `JointController` wrapping physics joint with PD control
- `BalanceConfig` for center-of-mass tracking
- `ActiveRagdollController` with ankle/hip strategy balance control

**Key Algorithms**:
- Quaternion error calculation: `q_error = q_target * q_current.conjugate()`
- PD torque: `torque = kp * angular_error + kd * (target_vel - current_vel)`
- Recovery behaviors: STEP, STUMBLE, FALL, BRACE
- COM projection for balance detection

### physics_animation_blend.py (693 lines) - REAL

**Evidence**:
- `BlendLayer` with weight and mask per bone
- `PhysicsAnimationBlender` managing multiple blend layers
- SLERP interpolation between animation and physics poses
- Hit reaction system with impulse-to-blend mapping

**Key Algorithms**:
- Per-bone blend weight computation from layer stack
- Quaternion SLERP for rotation blending
- Linear interpolation for position blending
- Impulse magnitude to blend time mapping

### character_interaction.py (896 lines) - REAL

**Evidence**:
- `InteractionType` enum: PUSH, GRAB, CARRY, THROW, CLIMB, VAULT, LEDGE_GRAB, ROPE_SWING
- `GrabInfo` with grab point, hand positions, joint reference
- `ClimbInfo` with hold points, movement direction, stamina cost
- `VaultInfo` with entry/exit points, vault type
- `CharacterInteractionManager` coordinating all interaction types

**Key Algorithms**:
- Grab joint creation with spring/damper parameters
- Two-handed carry with IK target computation
- Throw velocity calculation from charge time
- Vault trajectory planning with entry/exit detection

---

## Module: engine/simulation/cloth (5 files, 4 REAL + 1 PARTIAL STUB)

### cloth_simulation.py (663 lines) - REAL

**Evidence**:
- `ClothParticle` with position, velocity, inverse mass, pinned flag
- `ClothEdge` connecting two particles with rest length
- `ClothTriangle` for area computation and normal calculation
- `ClothMesh` managing particle/edge/triangle collections
- Position-Based Dynamics solver loop

**Key Algorithms**:
- PBD integration: predict positions, project constraints, update velocities
- Damping via velocity scaling
- Gravity and external force accumulation
- `create_cloth_grid()` and `create_cloth_from_mesh()` factories

### cloth_constraints.py (578 lines) - REAL

**Evidence**:
- `DistanceConstraint` maintaining edge rest length
- `BendingConstraint` using dihedral angle between adjacent triangles
- `ShearConstraint` for diagonal edges in quad meshes
- `LongRangeAttachment` connecting distant particles
- `AnchorConstraint` pinning particles to world positions
- `TetherConstraint` limiting maximum distance from anchor

**Key Algorithms**:
- Distance projection: `delta = (length - rest_length) / length * edge_vector`
- Dihedral angle computation via cross/dot products
- Stiffness-weighted position corrections
- Mass-ratio-based correction distribution

### cloth_collision.py (816 lines) - REAL

**Evidence**:
- `SphereCollider`, `CapsuleCollider`, `BoxCollider`, `MeshCollider`, `SDFCollider` classes
- `SpatialHash` for self-collision acceleration
- `ClothCollisionHandler` aggregating all collision types
- Continuous collision detection via raycasting

**Key Algorithms**:
- Sphere: `penetration = radius - distance(particle, center)`
- Capsule: closest point on line segment, then sphere test
- Box: GJK-based or separating axis test
- SDF: trilinear interpolation of distance field
- Spatial hash with cell size = 2 * particle radius

### cloth_wind.py (546 lines) - REAL

**Evidence**:
- `WindForce` base class with `apply_to_triangle()` method
- `DirectionalWind` with constant direction/magnitude
- `PointWind` with radial falloff
- `VortexWind` implementing Rankine vortex model
- `WindSystem` managing multiple wind sources

**Key Algorithms**:
- Aerodynamic drag: `F_drag = 0.5 * rho * Cd * A * v^2`
- Aerodynamic lift: `F_lift = 0.5 * rho * Cl * A * v^2`
- Turbulence via fractal Brownian motion (fBm) noise
- Rankine vortex: `v_theta = Gamma / (2 * pi * r)` for r > r_core

### gpu_cloth.py (572 lines) - PARTIAL STUB

**Evidence**:
- `GPUBuffer` and `GPUClothBuffers` defining GPU memory layout
- `GPUComputePipeline` abstract class for compute shader dispatch
- `GPUClothSolverStub` with explicit warning: `"WARNING: This stub does NOT simulate cloth. It only defines the interface for GPU-based cloth simulation."`
- GLSL shader templates: `INTEGRATION_SHADER_TEMPLATE`, `DISTANCE_CONSTRAINT_SHADER_TEMPLATE`, `VELOCITY_UPDATE_SHADER_TEMPLATE`

**Why PARTIAL STUB**:
- Buffer definitions and shader code are real/usable
- `GPUClothSolverStub.step()` is a no-op
- No actual GPU backend binding (wgpu, OpenGL, Vulkan)
- Shader templates exist but are not compiled/dispatched

---

## Module: engine/simulation/collision (6 files, ALL REAL)

### broadphase.py (1472 lines) - REAL

**Evidence**:
- `Vec3`, `AABB`, `Ray` dataclasses with full math operations
- `Broadphase` abstract base class with `insert`, `remove`, `update`, `query_aabb`, `query_ray`, `find_pairs`
- Four complete implementations:
  1. `SweepAndPrune` with sorted axis lists and incremental updates
  2. `DynamicBVH` with Surface Area Heuristic (SAH) for insertion
  3. `SpatialHashGrid` with configurable cell size
  4. `Octree` with configurable max depth and objects per node

**Key Algorithms**:
- SAP: axis sorting with O(n) incremental update
- BVH: SAH cost = traversal_cost + area_ratio * intersection_cost
- Spatial hash: `cell_key = (floor(x/size), floor(y/size), floor(z/size))`
- Octree: recursive subdivision with configurable thresholds

### narrowphase.py (1091 lines) - REAL

**Evidence**:
- `GJKSimplex` managing 1-4 support points
- `gjk_distance()` implementing Gilbert-Johnson-Keerthi algorithm
- `epa_penetration()` implementing Expanding Polytope Algorithm
- `sat_test()` implementing Separating Axis Theorem for convex shapes
- Specialized functions: `sphere_sphere`, `sphere_capsule`, `capsule_capsule`, `box_box`, `sphere_box`, `capsule_box`
- `collide_shapes()` dispatcher selecting optimal algorithm

**Key Algorithms**:
- GJK: Minkowski difference support mapping, simplex evolution
- EPA: polytope expansion toward origin, penetration depth extraction
- SAT: axis projection, overlap interval testing
- Specialized sphere/capsule for analytical solutions

### ccd.py (858 lines) - REAL

**Evidence**:
- `MotionState` with start/end transform and velocity
- `CCDResult` with time of impact, contact point, normal
- `CCDManager` coordinating continuous collision queries
- Linear sweep functions: `linear_sweep_sphere`, `linear_sweep_capsule`, `linear_sweep_box`

**Key Algorithms**:
- Time of impact via bisection with tolerance
- Sphere-sphere analytical TOI: quadratic formula on distance function
- Conservative advancement: step forward by `d / (v_max + omega_max * r_max)`
- Speculative contacts: predict contact within time horizon

### contact_manifold.py (669 lines) - REAL

**Evidence**:
- `ContactPoint` with position, normal, depth, accumulated impulses (normal + 2 tangent)
- `ContactManifold` managing up to 4 contact points per pair
- `ManifoldCache` for persistent manifold tracking across frames
- Warm starting via cached impulse ratios

**Key Algorithms**:
- Contact point reduction: keep 4 points maximizing convex hull area
- Warm starting: apply previous frame impulses scaled by persistence
- Contact persistence via distance threshold matching
- Touch state tracking: began/persist/ended transitions

### collision_events.py (679 lines) - REAL

**Evidence**:
- `CollisionEventType` enum: BEGIN, PERSIST, END
- `CollisionEvent` dataclass with bodies, contacts, impulse, normal, position, relative velocity
- `CollisionEventDispatcher` with priority-sorted handler lists
- `CollisionEventProcessor` tracking manifold state changes

**Key Algorithms**:
- Priority-based handler invocation (higher priority first)
- Event filtering via callback predicates
- Deferred event queuing for batch processing
- Body-specific handler registration

### collision_filter.py (580 lines) - REAL

**Evidence**:
- `CollisionLayer` IntFlag with 32 layers (bits 0-31)
- Predefined layers: DEFAULT, STATIC, DYNAMIC, KINEMATIC, TRIGGER, PROJECTILE, DEBRIS, SENSOR, PLAYER, NPC, ENEMY, VEHICLE, TERRAIN, WATER, CLIMBABLE, DESTRUCTIBLE, CUSTOM_1 through CUSTOM_16
- `CollisionMask` with bitwise operations
- `CollisionFilter` combining category + mask + group
- `CollisionFilterManager` with 32x32 layer matrix
- `FilterPresets` for FPS, platformer, racing games

**Key Algorithms**:
- Bitwise category/mask filtering: `a_accepts_b = (a.mask & b.category) != 0`
- Group-based filtering: same non-zero group prevents collision
- Layer matrix lookup for global layer rules

---

## Classification Criteria

| Classification | Criteria |
|----------------|----------|
| REAL | Complete implementation with working algorithms, no placeholder returns, handles edge cases |
| PARTIAL STUB | Interface definitions present, some logic exists, but key functionality is stubbed/no-op |
| STUB | Only type signatures, raises NotImplementedError or returns dummy values |

---

## Key Findings

1. **Production Quality**: All REAL implementations follow industry-standard algorithms documented in physics engine literature (Bullet, PhysX, Box2D).

2. **GPU Gap**: The only incomplete component (`gpu_cloth.py`) explicitly marks itself as a stub. The CPU cloth simulation is fully functional.

3. **Algorithm Coverage**:
   - Broadphase: SAP, BVH, Spatial Hash, Octree
   - Narrowphase: GJK, EPA, SAT, analytical sphere/capsule
   - CCD: Conservative advancement, speculative contacts
   - Cloth: PBD with distance, bending, shear, anchor constraints
   - Character: PD control, balance strategies, move-and-slide

4. **Integration Points**: Modules share `Vec3` definition (duplicated in broadphase.py and character_controller.py). Consider extracting to shared math module.

5. **No Rust FFI**: All implementations are pure Python with no pyo3/maturin bindings observed in these files.

---

## Recommendations

1. **Complete GPU Cloth**: Implement actual wgpu/Vulkan backend for `gpu_cloth.py` to enable GPU-accelerated cloth simulation.

2. **Unify Vector Math**: Extract `Vec3`, `Quaternion`, `Transform` to a shared module to avoid duplication.

3. **Add Tests**: Verify coverage for edge cases (degenerate simplexes in GJK, zero-area triangles in cloth, etc.).

4. **Profile Broadphase Selection**: Different games may benefit from different broadphase algorithms. Add runtime selection or benchmarking.

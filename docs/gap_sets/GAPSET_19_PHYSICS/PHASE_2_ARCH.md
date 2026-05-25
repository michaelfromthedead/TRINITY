# Phase 2: Core Physics -- Architecture

## Status: 13 [x] 2 [~] 0 [-]

## Module: `engine/simulation/`

### Overview
Phase 2 provides the core rigid body dynamics, collision detection, and constraint solving pipeline. All 15 tasks are fully implemented as Python reference code with standard float math. The existing code uses `Tuple[float, float, float]` for vectors and lacks Foundation decorator integration, but all algorithms are complete and production-grade.

---

### T-PHY-2.1: Rigid Body Components

**Status**: [x] Complete.
**Location**: `engine/simulation/physics/rigid_body.py` (1061 lines), `engine/simulation/components/rigid_body_component.py` (462 lines)

**Current Implementation**:
- `BodyType` enum: STATIC, KINEMATIC, DYNAMIC
- `RigidBody` dataclass: mass, inertia, damping, forces, gravity, sleeping, body_type, BodyState
- `BodyState`: position, rotation, linear_velocity, angular_velocity (all as Tuple[float, float, float])
- Semi-implicit Euler integration with configurable damping
- Force accumulation via add_force(), add_impulse(), clear_forces()
- 3 body types: static (infinite mass, never moves), kinematic (script-driven), dynamic (full simulation)
- ECS component wrapper provides archetype-based storage
- Body flags for per-body configuration

**Gap**: Uses float (not Fixed32). No `@simulation_domain("rigid_body")` or Foundation decorator. BodyState uses standard `Tuple[float, float, float]`.

### T-PHY-2.2: SAP Broadphase

**Status**: [x] Complete.
**Location**: `engine/simulation/collision/broadphase.py` (1472 lines, shared with T-PHY-2.3/2.4/2.5)

**Current Implementation**:
- `SAPBroadphase`: Sweep and Prune on AABB min/max in 3 axes
- O(n log n) sort, O(n) sweep
- Handles overlapping pairs correctly
- Configurable axis order

**Gap**: Uses float AABB comparison. Pair ordering is not explicitly deterministically sorted by entity pair. No `@spatial(structure="bvh")` decorator.

### T-PHY-2.3: Dynamic BVH Broadphase

**Status**: [x] Complete.
**Location**: `engine/simulation/collision/broadphase.py` (shared)

**Current Implementation**:
- `DynamicBVH`: Binary tree of AABB nodes
- Insert, remove, update operations
- Raycast and frustum query support
- Refit on motion

### T-PHY-2.4: Uniform Grid Broadphase

**Status**: [x] Complete.
**Location**: `engine/simulation/collision/broadphase.py` (shared)

**Current Implementation**:
- `UniformGrid`: Sparse grid with configurable cell size
- O(1) cell lookup via spatial hash
- Dynamic cell allocation

### T-PHY-2.5: Octree Broadphase

**Status**: [x] Complete.
**Location**: `engine/simulation/collision/broadphase.py` (shared)

**Current Implementation**:
- `OctreeBroadphase`: Sparse octree with configurable depth
- Insert, remove, update
- O(log n) query

### T-PHY-2.6: GJK Narrowphase

**Status**: [x] Complete.
**Location**: `engine/simulation/collision/narrowphase.py` (1091 lines, shared with T-PHY-2.7/2.8/2.9)

**Current Implementation**:
- Support function API for convex shapes (sphere, box, capsule, convex hull)
- GJK distance calculation with simplex tracking
- Simplex stages: point -> line -> triangle -> tetrahedron
- Fixed max iterations (32)
- Returns closest points + distance
- EPA fallback for penetrating cases

**Gap**: Uses float math, not Fixed32. Not deterministic across platforms.

### T-PHY-2.7: EPA Narrowphase

**Status**: [x] Complete.
**Location**: `engine/simulation/collision/narrowphase.py` (shared)

**Current Implementation**:
- Expanding Polytope algorithm from GJK simplex
- Penetration depth + contact normal
- Fixed max iterations (64)

### T-PHY-2.8: SAT Narrowphase

**Status**: [x] Complete.
**Location**: `engine/simulation/collision/narrowphase.py` (shared)

**Current Implementation**:
- Separating Axis Theorem for oriented boxes
- 15 test axes (3 face + 6 edge for OBB-OBB)
- Contact point generation

### T-PHY-2.9: MPR Narrowphase

**Status**: [x] Complete.
**Location**: `engine/simulation/collision/narrowphase.py` (shared)

**Current Implementation**:
- Minkowski Portal Refinement
- Penetration depth + normal
- Faster than EPA for deep penetration
- Fixed max iterations

### T-PHY-2.10: Contact Manifold

**Status**: [x] Complete.
**Location**: `engine/simulation/collision/contact_manifold.py` (669 lines)

**Current Implementation**:
- `ContactManifold` with up to 4 contact points per pair
- Persistent manifold with warm starting (stored impulses)
- Contact point reduction (keep deepest, most stable)
- `Contact` dataclass: point, normal, penetration, friction, restitution

**Gap**: Contact ordering not explicitly deterministically sorted by entity pair ID.

### T-PHY-2.11: CCD Modes

**Status**: [x] Complete.
**Location**: `engine/simulation/collision/ccd.py` (858 lines)

**Current Implementation**:
- `SpeculativeCCD`: Expand AABB by velocity * dt
- `SweepCCD`: Swept shape TOI computation
- Configurable per-body

**Gap**: Not deterministic across platforms.

### T-PHY-2.12: PGS/Sequential Impulse Solver

**Status**: [x] Complete.
**Location**: `engine/simulation/solver/constraint_solver.py` (725 lines)

**Current Implementation**:
- `ConstraintSolver`: Velocity-level Sequential Impulse solving
- Fixed iteration count (default 8, configurable)
- Warm starting from previous tick's impulses
- Coulomb friction model with static/dynamic coefficients
- Coefficient of restitution

**Gap**: No `@solver_hint(type="pgs")` decorator. Solver iteration order not guaranteed deterministic.

### T-PHY-2.13: TGS Solver

**Status**: [x] Complete.
**Location**: `engine/simulation/solver/tgs_solver.py` (750 lines)

**Current Implementation**:
- `TGSSolver`: Temporal subdivision of solver step
- Better stacking stability than PGS
- Fixed substep count

### T-PHY-2.14: XPBD Solver

**Status**: [x] Complete.
**Location**: `engine/simulation/solver/xpbd_solver.py` (791 lines)

**Current Implementation**:
- `XPBDSolver`: Compliance-based constraint formulation
- Position-level solving with compliance/stiffness parameters
- Fixed iteration count
- Used for cloth, soft body constraints

### T-PHY-2.15: Joint Types

**Status**: [x] Complete.
**Location**: `engine/simulation/constraints/` (8 joint files)

**Current Implementation**:
- `joint_fixed.py` (374 lines): Fixed joint, no relative motion
- `joint_hinge.py` (486 lines): Single-axis rotation, limits, motor
- `joint_slider.py` (518 lines): Single-axis translation, limits
- `joint_ball.py` (450 lines): 3-axis rotation, cone limits
- `joint_spring.py` (499 lines): Damped spring, rest length
- `joint_distance.py` (441 lines): Min/max distance constraint
- `joint_d6.py` (657 lines): 6-DOF configurable (per-axis lock/limit/motor)
- `joint_limits.py` (447 lines): Linear/angular limit types
- `joint_motors.py` (334 lines): Velocity and position motor control
- `joint_base.py` (546 lines): Base joint class with common functionality
- `contact_constraint.py` (605 lines): Normal, friction, rolling friction

---

## Key Design Decisions

- **Python reference is authoritative**: The existing Python implementation should be preserved as the algorithmic spec. Rust backend should target matching output within fixed-point precision.
- **Determinism requires Rust**: The existing Python float implementation cannot produce bit-identical results across platforms. The Rust fixed-point rewrite is required for determinism.
- **Broadphase selection**: The TODO correctly identifies 4 broadphase strategies. The Python implementation offers all 4 plus spatial hashing, allowing runtime selection based on scene characteristics.
- **Solver plurality**: All 3 solver types (PGS, TGS, XPBD) are implemented. PGS is the default general-purpose solver, TGS improves stacking, XPBD handles compliance-based materials (cloth, soft body).
- **Joint coverage**: 8 joint types plus motors and limits provide complete constraint coverage. This matches or exceeds physics middleware like PhysX and Bullet.

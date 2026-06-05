# Phase 1: Foundation -- Architecture

## Status: 0 [x] 0 [~] 9 [-]

## Module: `crates/physics/` (planned -- does not exist)

### Overview
Phase 1 provides the deterministic math foundation for all physics simulation. These types are planned as Rust implementations that replace the current Python `Tuple[float, float, float]` vector math with fixed-point arithmetic. All 9 tasks are entirely missing.

---

### T-PHY-1.1: Fixed16 Q8.8 Type (Rust)

**Status**: [-] Does not exist.
**Location**: Planned: `crates/physics/src/fixed16.rs`
**Python Reality**: No fixed-point types exist. Physics uses standard Python float.

**Acceptance Criteria**:
- `Fixed16` struct wraps i16 with Q8.8 format (range: [-128, +127.996], precision: 0.0039)
- add, sub, mul, div with overflow detection in debug builds
- Saturation clamping on overflow (configurable per-instance)
- Conversion from/to f32, i32
- Comparison operators (PartialOrd, Ord)
- `const` construction from float/integer literals
- Unit tests covering edge cases

### T-PHY-1.2: Fixed32 Q16.16 Type (Rust)

**Status**: [-] Does not exist.
**Location**: Planned: `crates/physics/src/fixed32.rs`
**Python Reality**: No fixed-point types exist.

**Acceptance Criteria**:
- `Fixed32` struct wraps i32 with Q16.16 format (range: [-32768, +32767.9999], precision: 0.000015)
- All arithmetic operations with overflow detection
- Saturation clamping
- Conversion from/to f32, f64, i32, Fixed16
- Comparison operators

### T-PHY-1.3: Fixed-Point Vector Types (Rust)

**Status**: [-] Does not exist.
**Location**: Planned: `crates/physics/src/fvec.rs`
**Python Reality**: Uses `Tuple[float, float, float]` for all 3D vectors. Helper functions (_vector_add, _vector_sub, _vector_scale, _vector_dot, _vector_cross, _vector_normalize) defined locally in each module.

**Acceptance Criteria**:
- `FVec2<T>`, `FVec3<T>`, `FVec4<T>` generic over Fixed16/Fixed32
- Component-wise add, sub, mul, div
- dot(), cross(), length_squared(), length()
- normalize() with zero-length fallback
- lerp() for interpolation

### T-PHY-1.4: Fixed-Point Rotation Types (Rust)

**Status**: [-] Does not exist.
**Location**: Planned: `crates/physics/src/fquat.rs`, `crates/physics/src/fangle.rs`
**Python Reality**: Uses `Tuple[float, float, float, float]` for quaternions. Trig functions use `math.sin`, `math.cos`, `math.atan2` etc.

**Acceptance Criteria**:
- `FQuat` with identity, mul, conjugate, inverse, rotate_vector
- `FAngle` with sin, cos, tan, asin, acos, atan2
- 4096-entry sin/cos lookup table covering [0, pi/2]
- CORDIC atan2 (16 iterations)
- Conversion from/to axis-angle, euler angles

### T-PHY-1.5: Fixed-Point Transform (Rust)

**Status**: [-] Does not exist.
**Location**: Planned: `crates/physics/src/ftransform.rs`
**Python Reality**: Transforms are constructed inline as tuples and passed through math operations.

**Acceptance Criteria**:
- `FTransform` with position (FVec3), rotation (FQuat), scale (FVec3)
- compose, decompose, transform_point, transform_vector
- Inverse transform
- Interpolation (lerp position, slerp rotation, lerp scale)

### T-PHY-1.6: Deterministic PCG RNG (Rust)

**Status**: [-] Does not exist.
**Location**: Planned: `crates/physics/src/rng.rs`
**Python Reality**: Physics uses `random` module or `uuid.uuid4()` for IDs. Not deterministic.

**Acceptance Criteria**:
- PCG-XSH-RR variant with u64 state + u64 increment
- next_u32(), next_u64(), next_f32(), next_f64()
- `fork()` -- create child RNG deterministically from parent
- `seed_from_u64()`, `seed_from_entity()` -- fork from world_rng using EntityID
- Cross-platform equivalence test

### T-PHY-1.7: Entity ID System (Rust)

**Status**: [-] Does not exist.
**Location**: Planned: `crates/physics/src/entity_id.rs`
**Python Reality**: Physics uses `uuid.uuid4()` string IDs for body identification. Non-deterministic.

**Acceptance Criteria**:
- `EntityID` wraps u64 (48-bit index + 16-bit generation)
- `NULL_ENTITY` constant
- index(), generation() accessors
- Deterministic allocation via command queue
- Serialization/deserialization to/from [u8; 8]

### T-PHY-1.8: Command Queue (Rust)

**Status**: [-] Does not exist.
**Location**: Planned: `crates/physics/src/command_queue.rs`
**Python Reality**: PhysicsWorld allows direct mutation of state (add_body, remove_body, set_transform). No command-based mutation.

**Acceptance Criteria**:
- `Command` struct with tick, entity_id, command_type, sequence, payload
- Command sort by (tick, entity_id, command_type, sequence)
- Binary encode/decode for channel transfer
- Validation (tick monotonicity, entity_id existence)
- Deterministic execution order guaranteed

### T-PHY-1.9: Simulation Decorators (Python -- Trinity Pattern)

**Status**: [-] Does not exist.
**Location**: Planned: `engine/simulation/decorators/` or Foundation decorator modules.
**Python Reality**: No Trinity pattern decorators applied to physics code.

**Acceptance Criteria**:
- `@simulation_domain(domain="physics")` -- marks a class as belonging to a simulation domain
- `@deterministic` -- marks a system as deterministic (no floats, no side effects)
- `@substep(count=4)` -- configurable substep rate
- `@solver_hint(preference="pgs"|"tgs"|"xpbd")` -- solver type hint
- `@sleep_threshold`, `@continuous_collision` -- per-body physics configuration
- `@buoyancy`, `@wind_affected` -- fluid interaction configuration
- `@fixed`, `@time_scale`, `@pausable`, `@rewindable` -- determinism control
- `@spatial`, `@partitioned` -- spatial partitioning configuration
- All decorators register with simulation domain registry

---

## Key Design Decisions
- All fixed-point types are Rust-only -- the Python reference implementation will continue using float
- The Foundation decorator system should mirror the spec in SIMULATION_CONTEXT.md exactly
- PCG RNG is the only deterministic RNG variant that satisfies all requirements (statistical quality, speed, determinism)
- EntityID generation strategy must match the command queue allocation strategy (not free-list)
- Command queue is the single entry point for all physics mutations in the deterministic path

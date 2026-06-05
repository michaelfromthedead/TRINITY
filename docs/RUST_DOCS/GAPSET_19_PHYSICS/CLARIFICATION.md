# GAPSET_19_PHYSICS -- Clarification Document

## Key Discoveries from RDC Investigation

### 1. TODO Was Written as Greenfield Plan, Not Reality

The PHASE_N_TODO.md header claims "Implementation Status: 0% complete, 54 tasks pending (227 days effort)" but the codebase already contains a mature Python implementation of ~70% of the described functionality (38 of 54 tasks fully complete). This suggests the TODO was written before the main implementation effort, or is tracking a *different deliverable* (Rust/WGSL/Foundation integration) while the Python algorithmic layer was built in parallel.

**Action**: The TODO should explicitly distinguish between Python algorithmic implementation (done) and Rust/WGSL/Foundation integration (remaining).

### 2. Location Mismatch: engine/physics/ vs. engine/simulation/

The TODO references `engine/physics/` as the primary location for physics code. This directory does not exist. The actual physics implementation lives at `engine/simulation/` with 12 subdirectories (physics/, collision/, solver/, constraints/, destruction/, cloth/, hair/, softbody/, fluid/, vehicles/, character/, components/). All 104 files totaling ~60,747 lines of Python reside here.

**Action**: The TODO's location references should be corrected to `engine/simulation/`.

### 3. Phase 2-5 Tasks Are 100% Complete in Python

All 35 tasks in Phases 2 (Core Physics), 3 (Advanced Simulation), 4 (Fluids), and 5 (Specialized) are fully implemented as Python reference implementations. The algorithms are production-grade: GJK/EPA/SAT/MPR narrowphase, PGS/TGS/XPBD solvers, 8 joint types, full destruction pipeline with 3 fracture patterns, SPH/PBF/FLIP/PIC/APIC fluids, wheeled/aircraft/watercraft/tracked/hover vehicles, and character controllers. The code uses standard Python float math (not Fixed32), but all algorithms are complete and working.

### 4. Foundation Decorator Integration Is Zero

Despite SIMULATION_CONTEXT.md documenting an extensive decorator system (`@simulation_domain`, `@substep`, `@solver_hint`, `@sleep_threshold`, `@continuous_collision`, `@physics_material`, `@joint`, `@destructible`, `@fracture`, `@wind_affected`, `@buoyancy`, `@spatial`, `@partitioned`, plus determinism decorators), **none of these decorators exist in the Python codebase**. The physics code is entirely free-standing Python with no Trinity Pattern integration. This is unlike GAPSET_14_ANIMATION where custom decorators like `@animation_data` and `@state_machine` were already applied.

**Action**: The Foundation decorator layer must be implemented from scratch as part of the integration effort.

### 5. Determinism Layer Is Entirely Missing

The TODO's Phase 6 (Determinism Integration) describes: fixed-point types (Fixed16/Fixed32), command queue, 13-phase tick executor, hierarchical XOR checksums, accordion snapshots, replay system, time-travel debugging, lockstep/rollback networking. DETERMINISM_CONTEXT.md (2,669 lines) provides a complete spec for all of these systems. However:

- Fixed16/Fixed32 exist only as planned types in DETERMINISM_CONTEXT.md -- no Python or Rust implementation exists
- PhysicsWorld has a single monolithic `step(dt)` method -- no 13-phase tick
- No checksum system, no snapshot/rollback, no command queue
- Physics uses `uuid.uuid4()` for body IDs (non-deterministic)
- Networking prediction exists in `engine/networking/prediction/` but is not physics-integrated

**Action**: This is the largest remaining implementation effort -- approximately 4-6 weeks for fixed-point types plus 4-6 weeks for determinism infrastructure.

### 6. No Rust Backend Exists

The planned Rust backend (`crates/physics/`) does not exist. This would include:
- Fixed16/Fixed32 types with overflow detection
- SIMD-accelerated FVec3/FQuat operations
- Deterministic PCG RNG
- FFI bridge for Python integration
- Parallel solver execution across islands

### 7. No WGSL Compute Shaders Exist

The planned GPU compute shaders (`shaders/physics/`) do not exist. Python staging code exists in `gpu_cloth.py` (572 lines) and `gpu_fluid.py` (560 lines) that prepares data for GPU consumption, but no actual WGSL shaders consume this data.

### 8. No Test Infrastructure

Zero test files exist for any simulation submodule. This is the largest quality gap -- 60,747 lines of complex mathematical code with no automated verification.

### 9. PhysicsAsset Resource Exists

`engine/resource/types/physics_asset.py` provides a PhysicsAsset class with ColliderType enum (BOX/SPHERE/CAPSULE/MESH/CONVEX), which serves as the bridge between the resource system and the physics simulation.

### 10. SIMULATION_CONTEXT.md Is Comprehensive

The existing `engine/simulation/SIMULATION_CONTEXT.md` (65,833 bytes) provides comprehensive architecture documentation covering all 12 submodules, decorators, metaclasses, descriptors, Foundation integration points, evaluation pipelines, spatial partitioning, and canonical usage examples. This document is accurate and can serve as the primary reference for new developers.

### 11. Revised Effort Estimate

The original estimate of 227 days assumes greenfield implementation. For the actual remaining work:

| Remaining Work | Effort | Depends On |
|---------------|--------|------------|
| Rust backend: Fixed16/Fixed32 types, FVec, FQuat, FTransform | 4-6 weeks | None (from GAPSET_1 math layer) |
| Rust backend: physics crate with FFI bridge | 6-8 weeks | Fixed-point types |
| Foundation decorator integration | 2-3 weeks | Foundation decorator system |
| WGSL compute shaders (SPH, cloth, fluid) | 3-4 weeks | GPU programming |
| Determinism integration (checksums, snapshots, replay, tick) | 4-6 weeks | Fixed-point types, command queue |
| Tests (unit + integration for all submodules) | 3-4 weeks | All above |
| **Total remaining** | **~22-31 weeks** | |

### 12. Key Architectural Insight

The physics code represents a complete, working reference implementation that should be preserved as the authoritative algorithmic specification. The Rust/WGSL/Foundation integration layer should be implemented as an optimization and determinism layer on top of (or alongside) this Python reference, not as a replacement. This two-tier approach (Python reference + Rust/WGSL acceleration) mirrors the animation system's architecture where the Python layer is the primary implementation with planned SIMD/GPU acceleration.

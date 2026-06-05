# PROJECT: engine/simulation/{character,cloth,collision}

## Overview

Production-grade physics simulation subsystem comprising character physics, cloth simulation, and collision detection systems. Contains 17 files totaling approximately 13,308 lines with 16 REAL implementations and 1 PARTIAL STUB.

## Scope

### In Scope

- **Character Physics** (6 files, ~4,614 lines)
  - Character controller with move-and-slide collision resolution
  - 14 movement modes with state machine transitions
  - Ragdoll system with 20 body parts
  - Active ragdoll with PD controllers and balance strategies
  - Physics-animation blending with per-bone weights
  - Character interactions (push, grab, carry, throw, climb, vault)

- **Cloth Simulation** (5 files, ~3,175 lines)
  - Position-Based Dynamics (PBD) solver
  - Constraint types: distance, bending, shear, anchor, tether, long-range attachment
  - Collision with sphere, capsule, box, mesh, and SDF colliders
  - Wind system with directional, point, and vortex sources
  - GPU cloth interface (PARTIAL STUB - buffer definitions only)

- **Collision Detection** (6 files, ~5,349 lines)
  - Four broadphase algorithms: SAP, BVH, Spatial Hash, Octree
  - Narrowphase: GJK, EPA, SAT, analytical sphere/capsule
  - Continuous collision detection (CCD) with conservative advancement
  - Contact manifold management with warm starting
  - Collision events with priority-based dispatching
  - 32-layer collision filtering system

### Out of Scope

- Actual GPU backend implementation (wgpu, Vulkan, OpenGL bindings)
- Rust FFI bindings (pure Python implementations)
- Shared math module extraction (duplicated Vec3 definitions remain)

## Goals

1. Maintain production-quality physics implementations
2. Complete GPU cloth simulation backend
3. Unify vector math primitives across modules
4. Ensure test coverage for edge cases

## Constraints

- Python 3.13 target (per TRINITY requirements)
- No external physics engine dependencies (pure Python)
- GPU cloth requires renderer-backend wgpu integration
- Maintain industry-standard algorithm compatibility (Bullet, PhysX, Box2D patterns)

## Module Summary

| Module | Files | Lines | Status |
|--------|-------|-------|--------|
| character | 6 | ~4,614 | ALL REAL |
| cloth | 5 | ~3,175 | 4 REAL, 1 PARTIAL STUB |
| collision | 6 | ~5,349 | ALL REAL |

## Acceptance Criteria

### Phase 1: GPU Cloth Completion
- [ ] GPU cloth solver dispatches actual compute shaders
- [ ] Integration with renderer-backend wgpu pipeline
- [ ] Performance parity or better than CPU PBD solver

### Phase 2: Math Unification
- [ ] Shared Vec3, Quaternion, Transform module extracted
- [ ] All physics modules import from shared location
- [ ] No duplicate math primitive definitions

### Phase 3: Test Coverage
- [ ] GJK degenerate simplex cases covered
- [ ] EPA edge cases (coplanar faces) tested
- [ ] Cloth zero-area triangle handling verified
- [ ] Broadphase benchmark suite for algorithm selection

### Phase 4: Integration Verification
- [ ] Character controller integrates with collision system
- [ ] Cloth collides with character ragdoll bodies
- [ ] All collision layers correctly filter interactions

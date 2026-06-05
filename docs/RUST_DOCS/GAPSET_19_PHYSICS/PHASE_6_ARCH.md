# Phase 6: Determinism Integration -- Architecture

## Status: 2 [x] 0 [~] 8 [-]

## Module: `engine/simulation/` + `engine/determinism/` + `engine/networking/`

### Overview
Phase 6 provides the determinism infrastructure required for multiplayer networking, replay, and time-travel debugging. This is the largest remaining implementation gap -- only 2 of 10 tasks have partial implementations. The core spec is documented in `engine/determinism/DETERMINISM_CONTEXT.md` (2,669 lines).

---

### T-PHY-6.1: SimulationWorld

**Status**: [~] Partial.
**Current Location**: `engine/simulation/physics/physics_world.py` (1010 lines)

**Current Implementation**:
- `PhysicsWorld` class with body management (add/remove/get)
- `step(dt)` method: single monolithic tick
- Collision detection dispatch
- Sleeping manager integration
- Query support (raycast, overlap, sweep)
- Callbacks for collision enter/stay/exit, trigger enter/stay/exit
- Statistics tracking

**Missing from TODO**:
- No `ArchetypeStorage` integration (uses dict-based body storage)
- No `SimRNG` (no deterministic random number generation)
- No `Tick` tracking for determinism
- No `CommandQueue` (direct mutation of state)
- No `snapshot()` for accordion snapshots
- No `restore(snapshot)` for rollback
- No `checksum()` for hierarchical checksums
- No `fork()` for branch simulation

**Gap**: The existing PhysicsWorld is a functionally complete monolithic simulation container but lacks all determinism infrastructure.

### T-PHY-6.2: 13-Phase Tick Executor

**Status**: [-] Does not exist.
**Location**: Planned: Rust backend.

**Current Implementation**: None. PhysicsWorld has a single `step(dt)` method:
```
step(dt):
  -> clear_forces()
  -> apply_gravity()
  -> broadphase()
  -> narrowphase()
  -> setup_constraints()
  -> solve()
  -> integrate()
  -> sleep()
```

**TODO Specification** (13 phases):
1. Input receive/validate/convert
2. Command sort and execute
3. Force accumulation (gravity, external)
4. CCD setup (speculative/sweep)
5. Broadphase pair generation
6. Narrowphase contact generation
7. Contact manifold persistence/reduction
8. Island construction
9. Constraint setup (Jacobian)
10. Solver (PGS/TGS/XPBD, fixed iterations)
11. Integration (semi-implicit Euler)
12. Sleeping
13. Snapshot/post-tick checksum

### T-PHY-6.3: Hierarchical Checksums

**Status**: [-] Does not exist.
**Location**: Planned: `crates/physics/src/checksum.rs`

**Current Implementation**: None. No checksum system exists anywhere in the physics code.

**Specification** (from DETERMINISM_CONTEXT.md):
- Entity-level checksum: hash of component data
- Archetype-level checksum: XOR of entity checksums
- Chunk-level checksum: XOR of archetype checksums
- World-level checksum: XOR of chunk checksums
- O(log N) divergence finding via XOR subtraction
- Per-tick storage in snapshot

### T-PHY-6.4: Accordion Snapshot Strategy

**Status**: [-] Does not exist.
**Location**: Planned: `crates/physics/src/snapshot.rs`

**Current Implementation**: None. No snapshot system. `BodyState` is used for interpolation purposes (storing previous frame state) but not for rollback.

**Specification**:
- Dense region: every tick, configurable N (default 10-30)
- Sparse region: f(n) = n^1.5 spacing
- Keyframe anchors: every K ticks (default 300-600), never pruned
- Activity-based density: increase density around active entities
- Memory-bounded: configurable maximum total snapshots

### T-PHY-6.5: Replay System

**Status**: [-] Does not exist.
**Location**: Planned: `crates/physics/src/replay.rs`

**Current Implementation**: None. No input recording or replay.

**Specification**:
- Input recording: store inputs per tick to buffer
- Replay from buffer: recreate simulation tick-by-tick
- File format: header + per-tick input stream (~2-5 MB/10min)
- 5 playback modes: normal, fast-forward, slow-motion, reverse, pause
- Checksum verification: compare computed to stored checksum

### T-PHY-6.6: Time-Travel Debugging

**Status**: [-] Does not exist.
**Location**: Planned: Python API + Rust backend.

**Current Implementation**: None.

**Specification**:
- Python API: `step_back(n_ticks)`, `goto_tick(tick_number)`
- Python API: `watch(entity_id, field_name, condition)`
- Breakpoints on checksum mismatch
- State diff visualization

### T-PHY-6.7: Lockstep Networking

**Status**: [-] Does not exist.
**Location**: Planned: Rust backend + network layer.

**Current Implementation**: None. Physics is local-only.

**Specification**:
- Deterministic lockstep: all peers simulate from same inputs
- Input synchronization protocol
- Mismatch detection via checksum exchange
- Peer timeout and recovery

### T-PHY-6.8: Rollback Networking

**Status**: [-] Does not exist.
**Location**: Planned: Rust backend + network layer.

**Current Implementation**: None.

**Specification**:
- Predict-ahead simulation
- Rollback on input confirmation mismatch
- Re-simulate from confirmed state
- Smooth visual interpolation of rollback

### T-PHY-6.9: Server Authoritative + Prediction

**Status**: [~] Partial.
**Current Location**: `engine/networking/prediction/client_prediction.py`, `server_reconciliation.py`

**Current Implementation**:
- Client-side prediction: exists in `engine/networking/prediction/` but NOT physics-integrated
- Server reconciliation: exists but NOT physics-integrated
- PhysicsWorld does not expose snapshot/restore for rollback needed by prediction

**Gap**: The networking prediction layer has components but they are not wired into the physics system. PhysicsWorld lacks the snapshot/restore interface that prediction requires.

### T-PHY-6.10: Simulation-Presentation Boundary

**Status**: [-] Does not exist.
**Location**: Planned: Python Foundation decorator system.

**Current Implementation**: None. No separation between simulation and presentation concerns.

**Specification**:
- Simulation components marked `@simulation` -- fixed-point only, no direct writes from presentation
- Presentation components marked `@presentation` -- floats allowed, no reads by simulation
- `@command` decorator marks methods that can mutate simulation state
- `@deterministic_rng` decorator marks methods that access deterministic RNG
- Descriptor-enforced immutability: presentation cannot write to simulation components
- Bridge channels enforce boundary at runtime

---

## Key Design Decisions

- **All determinism infrastructure is Rust-native**: Fixed-point types, checksums, snapshots, and replay must be implemented in Rust to guarantee cross-platform determinism. Python float arithmetic is inherently non-deterministic across platforms.
- **Accordion strategy balances memory vs. determinism**: Dense snapshots near the present, sparse snapshots in the past, and permanent keyframe anchors provide efficient time-travel without unbounded memory growth.
- **XOR hierarchical checksums enable O(log N) divergence search**: The XOR hierarchy means comparing two world checksums immediately identifies which chunk diverged, then binary search within the chunk.
- **Command queue is the single mutation entry point**: All physics mutations (spawn, despawn, set_transform, apply_force) go through the command queue, sorted by (tick, entity_id, command_type, sequence) for deterministic execution order.
- **Simulation-Presentation boundary is critical for correctness**: The Foundation descriptor system enforces at runtime that presentation code cannot write to simulation components, preventing non-deterministic corruption. This is a compile-time guarantee in the planned Rust backend.

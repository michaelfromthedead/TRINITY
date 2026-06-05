# RECOMMENDATIONS: engine/simulation/physics + engine/simulation/solver

## Rust Bridge Requirements

### High Priority

| Requirement | Justification | Effort |
|-------------|---------------|--------|
| **PyO3 bindings in bridge.rs** | Physics data must flow to/from ComponentStore for GPU-accelerated collision. Current bridge.rs is 9-line placeholder. | 2-3 days |
| **type_register() implementation** | Physics shapes need type registration in TypeRegistry for Rust-side storage. | 1 day |
| **component_read/write/delete** | RigidBody state (position, velocity, forces) must be accessible from Rust for frame graph integration. | 2 days |

### Medium Priority

| Requirement | Justification | Effort |
|-------------|---------------|--------|
| **GPU collision broadphase** | O(n^2) Python broadphase won't scale. WGPU compute shader for AABB tests. | 1 week |
| **Shared physics state buffer** | Eliminate Python-Rust copy overhead with shared memory (mmap or WGPU buffer). | 3-4 days |
| **Rust-side constraint solver** | For performance-critical simulations, port Sequential Impulse to Rust. | 2 weeks |

### Low Priority

| Requirement | Justification | Effort |
|-------------|---------------|--------|
| **Physics debug visualization** | Render collision shapes, contact points, forces via frame graph. | 3-4 days |
| **Deterministic simulation** | Fixed-point math or identical float handling across Python/Rust boundary. | 1 week |

---

## Integration Strategy

### Phase 1: Bridge Foundation (Week 1)
1. Implement `type_register()` in bridge.rs with PyO3
2. Add `#[pyfunction]` exports for component_read/write
3. Test with single RigidBody round-trip

### Phase 2: State Synchronization (Week 2)
1. Create `PhysicsStateBuffer` struct in Rust
2. Implement bulk upload of RigidBody transforms
3. Integrate with frame graph node for visibility

### Phase 3: Collision Offload (Weeks 3-4)
1. Port broadphase AABB tests to WGPU compute shader
2. Return collision pairs to Python for narrowphase
3. Benchmark against O(n^2) Python baseline

### Phase 4: Full GPU Physics (Future)
1. Port narrowphase to GPU
2. Implement GPU constraint solver
3. Retain Python for game logic, Rust/GPU for simulation

---

## Testing Strategy

### Unit Tests (Already Present: 317)
- Continue using pytest for physics_core and solver tests
- Add property-based tests for numerical edge cases

### Integration Tests (Needed)
| Test | Purpose |
|------|---------|
| Python-Rust round-trip | Verify data integrity across bridge |
| 100-body simulation | Benchmark broadphase performance |
| Island sleeping stress | Verify Union-Find with 1000+ bodies |
| Constraint chain | 50+ bodies connected by joints |

### Benchmark Suite (Needed)
```bash
# Proposed benchmarks
bench_broadphase_100.py   # 100 bodies, measure collision pairs/ms
bench_broadphase_1000.py  # 1000 bodies, identify bottleneck
bench_solver_stack.py     # 10-body stack, measure stability
bench_xpbd_cloth.py       # 50x50 cloth grid, measure FPS
```

### Regression Tests
- Save simulation state snapshots
- Replay and compare for determinism
- Critical for multiplayer physics sync

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GIL contention in parallel solver | High | Medium | Use Rust threads via PyO3 release_gil |
| Float precision mismatch Python/Rust | Medium | High | Use f64 everywhere, add epsilon tests |
| Broadphase scaling issues | High | High | Prioritize BVH/spatial hash implementation |
| Memory allocation overhead | Medium | Medium | Pool allocators for constraints, contacts |

### Schedule Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| PyO3 binding complexity | Medium | Medium | Start with minimal API, expand incrementally |
| WGPU compute shader debugging | Medium | High | Unit test shaders with known inputs |
| Integration with existing frame graph | Low | Medium | Isolate physics node, minimize dependencies |

### Quality Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical stability regression | Low | High | Run full test suite on every bridge change |
| Performance regression | Medium | Medium | Maintain benchmark baseline, CI alerts |
| API breaking changes | Low | Medium | Version physics interface, deprecation period |

---

## Recommended Action Plan

### Immediate (This Sprint)
1. [ ] Implement PyO3 `type_register()` in bridge.rs
2. [ ] Add `component_read()` for RigidBody position
3. [ ] Create integration test: Python RigidBody -> Rust ComponentStore -> Python

### Near-Term (Next 2 Sprints)
4. [ ] Replace broadphase O(n^2) with spatial hash (Python first)
5. [ ] Implement `ParallelIslandSolver` with ThreadPoolExecutor
6. [ ] Add WGPU compute shader for AABB broadphase

### Medium-Term (This Quarter)
7. [ ] Implement Quickhull for ConvexHullShape
8. [ ] Build BVH tree for MeshShape
9. [ ] Port Sequential Impulse solver to Rust (optional perf path)

### Deferred (Backlog)
10. [ ] True OBB-OBB SAT collision
11. [ ] GPU narrowphase (GJK/EPA in compute shader)
12. [ ] Deterministic simulation mode for netcode

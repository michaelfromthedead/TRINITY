# RECOMMENDATIONS: engine/simulation/{character,cloth,collision}

---

## Rust Bridge Requirements

### High Priority

| Component | Current State | Rust Bridge Action | Rationale |
|-----------|--------------|-------------------|-----------|
| `narrowphase.py` (GJK/EPA) | 1,091 lines Python | Implement in Rust with pyo3 | GJK/EPA are CPU-intensive, SIMD-friendly; expect 10-100x speedup |
| `broadphase.py` (BVH) | 1,472 lines Python | Implement BVH in Rust | Tree operations benefit from cache-friendly Rust structs |
| Shared Math (`Vec3`, `Quaternion`) | Duplicated in 2+ files | Export from `omega` crate | Already have `omega` math crate; expose via pyo3 |

### Medium Priority

| Component | Current State | Rust Bridge Action | Rationale |
|-----------|--------------|-------------------|-----------|
| `ccd.py` | 858 lines Python | Rust implementation | CCD is called per moving object; tight loop performance matters |
| `cloth_constraints.py` | 578 lines Python | Rust for constraint solver | PBD iterations are compute-bound |
| `contact_manifold.py` | 669 lines Python | Rust implementation | Cache warm-starting benefits from low-level memory control |

### Low Priority

| Component | Current State | Rust Bridge Action | Rationale |
|-----------|--------------|-------------------|-----------|
| `collision_filter.py` | 580 lines Python | Keep Python | Bitwise ops are fast enough in Python |
| `collision_events.py` | 679 lines Python | Keep Python | Event dispatch is not performance-critical |
| `movement_modes.py` | 692 lines Python | Keep Python | State machine logic is not hot path |

---

## Integration Strategy

### Phase 1: Math Unification (1 sprint)

1. Extract `Vec3`, `Quaternion`, `Transform` from `broadphase.py` and `character_controller.py`
2. Create `engine/simulation/math/` module
3. Or: Expose `omega` crate math types via pyo3 directly

### Phase 2: GPU Cloth (2-3 sprints)

1. Implement `GPUClothSolver` backend using wgpu (Rust) or WebGPU bindings
2. Shader templates already exist in `gpu_cloth.py` (GLSL format)
3. Wire up to renderer-backend via existing frame graph

### Phase 3: Rust Narrowphase (2 sprints)

1. Implement `gjk_distance()` and `epa_penetration()` in Rust
2. Expose via pyo3 with NumPy array support
3. Keep Python fallback for compatibility

### Phase 4: Rust Broadphase (2 sprints)

1. Implement `DynamicBVH` in Rust with incremental updates
2. Use arena allocator for cache-friendly node storage
3. Expose via pyo3 with batch query interface

---

## Testing Strategy

### Unit Tests Required

| Module | Test Focus | Priority |
|--------|-----------|----------|
| `narrowphase.py` | Degenerate simplex handling, zero-area faces | HIGH |
| `broadphase.py` | Empty tree, single object, overlapping AABBs | HIGH |
| `cloth_simulation.py` | Zero-mass particles, pinned edge cases | HIGH |
| `cloth_constraints.py` | Zero-length edges, inverted masses | MEDIUM |
| `ccd.py` | Tunneling at high velocities, parallel motion | MEDIUM |
| `active_ragdoll.py` | Singularity in quaternion error | LOW |

### Integration Tests Required

| Test | Modules | Priority |
|------|---------|----------|
| Broadphase + Narrowphase pipeline | `broadphase.py`, `narrowphase.py` | HIGH |
| Cloth + Collision interaction | `cloth_simulation.py`, `cloth_collision.py` | HIGH |
| Character + Ragdoll transition | `character_controller.py`, `ragdoll.py`, `active_ragdoll.py` | MEDIUM |

### Performance Benchmarks

| Benchmark | Target | Current |
|-----------|--------|---------|
| GJK 1000 pairs | < 1ms | TBD |
| BVH 10000 objects query | < 0.5ms | TBD |
| Cloth 1000 particles step | < 2ms | TBD |

---

## Risk Assessment

### High Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU cloth shader errors | Medium | High | Test shaders in isolation before integration |
| Narrowphase numerical instability | Low | High | Add epsilon guards, test degenerate cases |

### Medium Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Math module extraction breaks imports | Medium | Medium | Use import aliases during transition |
| Rust pyo3 memory leaks | Low | Medium | Use pytest-memray for leak detection |
| Performance regression during refactor | Medium | Medium | Establish benchmarks before changes |

### Low Risk

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Broadphase algorithm mismatch for scene | Low | Low | Add runtime algorithm selection |
| Cloth self-collision performance | Low | Low | Spatial hash is already implemented |

---

## Recommended Action Order

1. **Immediate**: Extract shared math module (removes duplication debt)
2. **Sprint 1-2**: Implement GPU cloth backend via wgpu
3. **Sprint 3-4**: Rust narrowphase (GJK/EPA) with pyo3
4. **Sprint 5-6**: Rust broadphase (BVH) with pyo3
5. **Ongoing**: Expand test coverage, performance benchmarks

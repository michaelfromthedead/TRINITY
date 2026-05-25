# RECOMMENDATIONS: engine/animation/skeletal + engine/animation/systems

---

## Rust Bridge Requirements

### High Priority

| Requirement | Rationale | Estimated Effort |
|-------------|-----------|------------------|
| **Skinning matrix buffer PyO3 binding** | GPU skinning requires passing matrices from Python to Rust renderer. Without this, all skinning stays CPU-bound. | 2-3 days |
| **AnimationComponent in ComponentStore** | Required for ECS integration across Python/Rust boundary. Animation state must be accessible from both sides. | 1-2 days |
| **SkeletonComponent in TypeRegistry** | Bone hierarchies need to be queryable from Rust for GPU buffer layout. | 1 day |

### Medium Priority

| Requirement | Rationale | Estimated Effort |
|-------------|-----------|------------------|
| **Animation texture atlas renderer support** | Crowd animation uses texture-based skinning. Renderer must support animation texture sampling. | 3-4 days |
| **Streaming decompression in Rust** | Python decompression works but is slow. Rust SIMD decompression needed for large clip libraries. | 2-3 days |
| **Blend state component binding** | Animation blend weights need GPU upload path for blend shape rendering. | 1-2 days |

### Low Priority

| Requirement | Rationale | Estimated Effort |
|-------------|-----------|------------------|
| **Motion matching SIMD search** | Python search is O(n) per frame. Rust SIMD would help with 10K+ frame databases. | 4-5 days |
| **IK solver Rust port** | IK is per-character, not a bottleneck. Only needed for 100+ simultaneous IK solves. | 3-4 days |

---

## Integration Strategy

### Phase 1: Basic Bridge (Week 1)

1. Add `AnimationComponent` and `SkeletonComponent` to `crates/renderer-backend/src/type_registry.rs`
2. Create `SkinningBuffer` struct in `crates/renderer-backend/src/bridge.rs`
3. Add PyO3 bindings for matrix buffer upload

**Deliverable**: Python can prepare skinning matrices and pass to Rust renderer.

### Phase 2: GPU Skinning (Week 2)

1. Implement compute shader skinning in Rust
2. Create double-buffered skinning matrix storage
3. Wire to existing mesh rendering pipeline

**Deliverable**: Characters render with GPU-computed skinning.

### Phase 3: Crowd Support (Week 3)

1. Add animation texture atlas format support
2. Implement texture sampling in vertex shader
3. Create Python->Rust atlas upload path

**Deliverable**: Crowd instances animate via texture lookup.

### Dependency Order

```
TypeRegistry registration
    |
    v
ComponentStore integration
    |
    v
SkinningBuffer PyO3 binding
    |
    v
GPU skinning shader
    |
    v
Animation texture support (parallel track)
```

---

## Testing Strategy

### Unit Tests (Python)

| Test Target | Coverage Goals |
|-------------|----------------|
| `clip.py` interpolation | Edge cases: t=0, t=1, single keyframe, zero duration |
| `skinning.py` DQS | Antipodality flips, identity transforms, extreme weights |
| `compression.py` round-trip | Decompress(Compress(clip)) == clip within tolerance |
| `skeleton.py` path-finding | Disjoint trees, root-to-leaf, leaf-to-leaf |

### Integration Tests (Python)

| Test Target | Coverage Goals |
|-------------|----------------|
| IK convergence | Two-Bone reaches target, FABRIK converges, CCD terminates |
| Animation graph | State transitions fire, blend weights interpolate |
| Motion matching | Best match found, continuation cost prevents jitter |

### Bridge Tests (Rust)

| Test Target | Coverage Goals |
|-------------|----------------|
| Matrix buffer upload | Python matrices arrive in GPU buffer |
| Component registration | AnimationComponent queryable from Rust |
| Type round-trip | Python skeleton -> Rust -> Python unchanged |

### Performance Benchmarks

| Benchmark | Target |
|-----------|--------|
| Skinning 100 characters | < 2ms Python prep, < 0.5ms GPU |
| Decompression 10MB clip | < 50ms |
| Motion matching 10K frames | < 1ms |

---

## Risk Assessment

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Skinning buffer format mismatch | Characters render incorrectly | Define strict matrix layout in shared header |
| Animation texture precision loss | Crowd animation artifacts | Use 16-bit float textures minimum |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| ComponentStore version mismatch | Python/Rust see different data | Version component schemas |
| Memory ownership ambiguity | Use-after-free in skinning buffers | Clear ownership protocol: Python allocates, Rust consumes |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| IK solver divergence | Character limbs misbehave | Keep Python fallback, add convergence timeout |
| Motion matching performance | Frame rate drop with large databases | Implement database LOD or spatial partitioning |

---

## Summary

The animation subsystem is production-ready Python code. GRANDPHASE2 integration requires:

1. **Must-have**: Skinning buffer bridge, component registration (1-2 weeks)
2. **Should-have**: GPU skinning, animation textures (2-3 weeks)
3. **Nice-to-have**: Rust decompression, SIMD search (2-4 weeks)

Total estimated effort: 5-9 weeks for full integration.

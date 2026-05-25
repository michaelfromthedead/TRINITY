# RECOMMENDATIONS: engine/animation/graph + engine/animation/ik

## Rust Bridge Requirements

### High Priority

| Component | Rationale | Bridge Type |
|-----------|-----------|-------------|
| **Jacobian Matrix Operations** | Matrix multiply, inversion are compute-bound; benefits from SIMD | Data + Compute |
| **FABRIK Iteration Loop** | Many iterations over bone chains; cache-friendly in Rust | Compute |
| **Pose Blending** | Per-bone lerp/slerp for large skeletons; GPU-adjacent | Data |
| **Quaternion Operations** | SLERP, multiply called millions of times per frame | Compute |

### Medium Priority

| Component | Rationale | Bridge Type |
|-----------|-----------|-------------|
| **TwoBone IK Solver** | Fast already but frequently called; minor gains | Compute |
| **CCD Iteration** | Similar to FABRIK; iterative joint adjustment | Compute |
| **Skeleton Bone Chain Lookup** | String lookups could be hashed integers | Data |
| **State Machine Logic** | Low compute; branching-heavy; less benefit | Logic |

### Low Priority

| Component | Rationale | Bridge Type |
|-----------|-----------|-------------|
| **Blend Tree Structure** | Rarely changes; mostly configuration | Metadata |
| **Graph Parameter Management** | Python dict access is fine | N/A |
| **Event Callbacks** | Python-side logic for flexibility | N/A |

---

## Integration Strategy

### Phase 1: Data Structures

1. **Define Rust equivalents**:
   ```rust
   #[repr(C)]
   struct Transform {
       position: Vec3,
       rotation: Quat,
       scale: Vec3,
   }
   
   #[repr(C)]
   struct Pose {
       transforms: Vec<Transform>,
       root_motion: Option<Transform>,
   }
   ```

2. **Create PyO3 bindings** for Pose struct
3. **Implement zero-copy view** for Python-side access

### Phase 2: IK Solvers

1. **Port TwoBoneIK to Rust** (simplest, most-called)
2. **Add FABRIK Rust implementation** with same API
3. **Keep Python versions** as fallback/reference
4. **Benchmark both paths** to validate gains

### Phase 3: Graph Evaluation

1. **Identify hot paths** with profiling
2. **Consider Rust-side pose cache** for frequently-evaluated nodes
3. **Keep graph topology in Python** (rarely the bottleneck)

---

## Testing Strategy

### Unit Tests

| Category | Test Coverage |
|----------|---------------|
| Quaternion SLERP | Edge cases: parallel, anti-parallel, threshold fallback |
| Blend Trees | 1D: boundary thresholds, single entry. 2D: Delaunay degenerates |
| State Machine | Transition priority, any-state, interruption |
| Two-Bone IK | Unreachable targets, soft IK, pole vector |
| FABRIK | Convergence, constraint application, multi-chain |
| Jacobian | Singular matrices, damping effect, multi-effector |
| Full Body IK | Balance maintenance, pelvis adjustment |

### Property-Based Tests

```python
# Example: Quaternion SLERP properties
@given(q1=quaternions(), q2=quaternions(), t=floats(0, 1))
def test_slerp_bounds(q1, q2, t):
    result = slerp(q1, q2, t)
    assert abs(result.length() - 1.0) < 1e-6  # Unit quaternion
    
@given(q1=quaternions(), q2=quaternions())
def test_slerp_endpoints(q1, q2):
    assert slerp(q1, q2, 0.0) == q1
    assert slerp(q1, q2, 1.0) == q2
```

### Integration Tests

1. **Graph + IK Pipeline**: Evaluate graph, apply IK, verify pose validity
2. **State Machine + Blend Tree**: Transition between states with blended poses
3. **Full Body + Foot Placement**: Apply foot IK with terrain callback

### Performance Benchmarks

| Test | Target | Metric |
|------|--------|--------|
| 100-bone pose blend | <1ms | Wall time |
| 10-bone FABRIK solve | <100us | Wall time |
| State machine update | <10us | Wall time |
| Two-bone IK | <5us | Wall time |

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Rust bridge overhead exceeds Python savings | Medium | High | Benchmark early; batch operations |
| Animation graph API changes break bridge | Low | Medium | Version bridge API; keep Python fallback |
| IK constraint handling differs in Rust | Low | High | Port exact algorithms; property tests |
| GPU skinning data layout incompatible | Medium | High | Define layout early; prototype integration |

### Architectural Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Over-engineering bridge for low-traffic paths | Medium | Low | Profile first; bridge hot paths only |
| Python-Rust boundary crossing too frequent | Medium | Medium | Batch pose operations; minimize crossings |
| Memory lifetime issues with shared Pose data | Low | High | Use Rust ownership; avoid shared mutability |

### Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Rust implementation diverges from Python | Medium | Medium | Single source of truth for algorithms; test parity |
| Debug complexity increases | Medium | Low | Keep Python reference; conditional compilation |

---

## Next Steps

1. **Profile current system** to identify actual bottlenecks
2. **Define Pose/Transform bridge types** in Rust
3. **Port TwoBoneIK** as proof-of-concept
4. **Benchmark Python vs Rust** for IK workloads
5. **Iterate based on real performance data**

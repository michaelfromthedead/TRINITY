# RECOMMENDATIONS: engine/simulation Subsystems

## Rust Bridge Requirements

### High Priority

| Component | Bridge Type | Rationale |
|-----------|-------------|-----------|
| PBDSoftBody | Full port | Performance-critical; 60Hz+ requires native code |
| FEMSolver | Full port | Matrix operations benefit from Rust + BLAS |
| ConstraintSolver | Full port | Sequential Impulse solver needs cache locality |
| PacejkaTire | Inline bridge | Called per wheel per frame; low overhead required |

**Recommended Approach**:
```rust
// crates/simulation-bridge/src/lib.rs
#[pyclass]
struct RustPBDSolver {
    positions: Vec<[f32; 3]>,
    constraints: Vec<ConstraintData>,
}

#[pymethods]
impl RustPBDSolver {
    fn step(&mut self, dt: f32, iterations: u32);
    fn get_positions(&self) -> Vec<[f32; 3]>;
}
```

### Medium Priority

| Component | Bridge Type | Rationale |
|-----------|-------------|-----------|
| Drivetrain | Partial port | Keep Python config, Rust simulation loop |
| Differential | Partial port | Torque split math is simple, but called frequently |
| ShapeMatching | GPU compute | SVD per cluster is embarrassingly parallel |
| VolumeConstraint | GPU compute | Per-tetrahedron gradients are independent |

### Low Priority

| Component | Bridge Type | Rationale |
|-----------|-------------|-----------|
| Aircraft | Keep Python | Called infrequently, not latency-sensitive |
| Watercraft | Keep Python | Distributed buoyancy is O(sample_points), manageable |
| ClothComponent | Wrapper only | State holder; solver does heavy lifting |
| VehicleComponent | Wrapper only | Thin ECS integration layer |

---

## Integration Strategy

### Phase 1: Bridge Infrastructure (Week 1-2)

1. Create `crates/simulation-bridge/` with PyO3 setup
2. Define shared types: `Vec3`, `Quaternion`, `Matrix3x3`
3. Implement `RustConstraintSolver` with Python bindings
4. Wire to existing `engine/simulation/solver/` as drop-in replacement

### Phase 2: Soft Body Migration (Week 3-4)

1. Port `VolumeConstraint` projection to Rust
2. Port `EdgeLengthConstraint` projection to Rust
3. Port `StrainLimitConstraint` (includes SVD) to Rust
4. Benchmark against Python reference; target 10x speedup

### Phase 3: Vehicle Subsystem (Week 5-6)

1. Port `PacejkaTire._magic_formula()` to Rust inline function
2. Port `Differential.torque_split()` variants to Rust
3. Keep Python configuration/state; Rust computes forces
4. Benchmark Pacejka: target <1us per call

### Phase 4: GPU Compute (Week 7-8)

1. Write WGSL compute shader for PBD constraint projection
2. Use `wgpu` (via `renderer-backend`) for buffer management
3. Implement double-buffered position swapping
4. Target 100K particles at 60Hz

---

## Testing Strategy

### Unit Tests (Rust)

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_pacejka_peak_slip() {
        let tire = PacejkaTire::default();
        let slip_at_peak = tire.get_peak_slip_ratio();
        assert!((slip_at_peak - 0.1).abs() < 0.02);
    }

    #[test]
    fn test_volume_constraint_preserves_volume() {
        let mut solver = RustPBDSolver::new(cube_mesh());
        let initial_vol = solver.compute_volume();
        solver.step(0.016, 10);
        let final_vol = solver.compute_volume();
        assert!((initial_vol - final_vol).abs() / initial_vol < 0.01);
    }
}
```

### Integration Tests (Python)

```python
def test_rust_solver_matches_python():
    """Ensure Rust solver produces same results as Python reference."""
    py_solver = PBDSoftBody(mesh, material)
    rust_solver = RustPBDSolver.from_mesh(mesh, material)
    
    for _ in range(100):
        py_solver.step(dt=0.016)
        rust_solver.step(dt=0.016)
    
    py_pos = py_solver.positions
    rust_pos = rust_solver.get_positions()
    np.testing.assert_allclose(py_pos, rust_pos, rtol=1e-4)
```

### Benchmark Tests

| Scenario | Metric | Target |
|----------|--------|--------|
| PBD 10K particles | ms/frame | <5ms |
| FEM 5K tetrahedra | ms/frame | <10ms |
| Pacejka 4 tires | us/call | <1us |
| Drivetrain update | us/call | <10us |

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical divergence in Rust port | Medium | High | Bit-exact comparison tests vs Python |
| SVD stability issues | Low | Medium | Use proven nalgebra implementation |
| PyO3 GIL contention | Medium | Medium | Release GIL during computation |
| GPU compute complexity | High | Medium | Start with CPU fallback, add GPU later |

### Schedule Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Underestimated FEM complexity | Medium | High | Port simpler PBD first; learn patterns |
| Vehicle edge cases | Low | Low | Extensive test suite from Python |
| Integration bugs | Medium | Medium | Incremental rollout; feature flags |

### Performance Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Rust overhead > Python gains | Low | High | Profile early; ensure 5x+ speedup |
| GPU transfer latency | Medium | Medium | Batch transfers; double buffer |
| Memory fragmentation | Low | Low | Use arena allocators |

---

## Acceptance Criteria

### Rust Bridge Complete When:

- [ ] All unit tests pass in Rust
- [ ] Integration tests match Python reference (rtol=1e-4)
- [ ] Benchmarks meet targets
- [ ] No GIL held during heavy computation
- [ ] Memory safe (no unsafe blocks without justification)

### GPU Compute Complete When:

- [ ] PBD runs on GPU with CPU fallback
- [ ] 100K particles at 60Hz achieved
- [ ] Graceful degradation on older GPUs
- [ ] Hot-reload of compute shaders works

### Integration Complete When:

- [ ] Existing Python components work unchanged
- [ ] New Rust path is opt-in via config
- [ ] No regression in existing tests
- [ ] Documentation updated with performance notes

# RECOMMENDATIONS.md - Simulation Subsystems (Destruction, Fluid, Hair)

---

## Rust Bridge Requirements

### High Priority

| Component | Rationale | Effort |
|-----------|-----------|--------|
| SPH Kernel Functions | Hot path - called millions of times per frame | Low |
| Spatial Hash Grid | Memory-intensive, benefits from Rust allocation | Medium |
| Particle Position Updates | Embarrassingly parallel, compute-bound | Low |
| Hair Constraint Solving | Per-strand iteration, benefits from SIMD | Medium |

### Medium Priority

| Component | Rationale | Effort |
|-----------|-----------|--------|
| Voronoi Cell Generation | Complex geometry, moderate frequency | High |
| Support Graph Dijkstra | Graph traversal benefits from Rust safety | Medium |
| Debris LOD Evaluation | Distance calculations, batch-friendly | Low |
| FLIP/PIC Grid Transfers | Memory bandwidth sensitive | Medium |

### Low Priority

| Component | Rationale | Effort |
|-----------|-----------|--------|
| Config Parameter Loading | Called once at init | Low |
| Fracture Pattern Generation | Infrequent, amortized cost | High |
| Surface Reconstruction | Post-process, not real-time critical | High |

---

## Integration Strategy

### Phase 1: Data Structures (Week 1-2)

1. Define Rust equivalents for:
   - `ParticleBuffer` (already GPU-optimized layout)
   - `GridBuffer` (spatial hash)
   - `HairStrand` / `HairControlPoint`

2. Create PyO3 bindings for data transfer:
   ```rust
   #[pyclass]
   struct RustParticleBuffer {
       positions: Vec<[f32; 4]>,
       velocities: Vec<[f32; 4]>,
   }
   ```

### Phase 2: Hot Path Functions (Week 3-4)

1. Port SPH kernels to Rust:
   - `poly6()`, `spiky()`, `viscosity_laplacian()`
   - Use packed_simd or std::simd for vector ops

2. Port spatial hash construction:
   - Counting sort with parallel prefix sum
   - Return indices to Python for neighbor queries

### Phase 3: Full Solver Ports (Week 5-8)

1. Complete SPH solver in Rust
2. Hair constraint solver with FTL
3. Keep Python versions as reference/fallback

### Phase 4: GPU Compute Integration (Week 9-12)

1. Implement gpu_fluid.py abstract methods via wgpu
2. WGSL compute shaders for:
   - `dispatch_build_grid()`
   - `dispatch_compute_density()`
   - `dispatch_compute_forces()`
   - `dispatch_integrate()`

---

## Testing Strategy

### Unit Tests (Immediate)

| Test Category | Priority | Coverage Target |
|---------------|----------|-----------------|
| Kernel normalization | High | 100% |
| Edge case geometry | High | Degenerate triangles, collinear points |
| Numerical stability | High | Extreme inputs, near-zero values |
| LOD transitions | Medium | Hysteresis behavior |

### Integration Tests (Short-Term)

| Test Category | Priority | Coverage Target |
|---------------|----------|-----------------|
| Fracture → Debris pipeline | High | End-to-end destruction |
| Fluid → Rendering pipeline | High | Particle → mesh |
| Hair → Collision resolution | Medium | Constraint satisfaction |

### Performance Tests (Medium-Term)

| Test Category | Metric | Baseline |
|---------------|--------|----------|
| SPH 10K particles | ms/frame | TBD |
| Fracture 1K tris | ms/fracture | TBD |
| Hair 100 strands | ms/frame | TBD |

### Rust Parity Tests (Bridge Phase)

- Verify Rust output matches Python reference within epsilon
- Benchmark speedup factor
- Memory usage comparison

---

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU driver compatibility | Medium | High | Keep CPU fallback paths |
| Numerical divergence Rust vs Python | Low | Medium | Comprehensive parity tests |
| PyO3 overhead negates gains | Low | High | Profile before committing |
| WGSL limitations | Low | Medium | Prototype shaders early |

### Schedule Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU integration complexity | Medium | High | Phase 4 can slip without blocking |
| Testing coverage gaps | Medium | Medium | Prioritize hot path tests |
| Parallel algorithm debugging | High | Medium | Start with single-threaded Rust |

### Operational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking Python API | Low | High | Keep signatures stable |
| Platform-specific failures | Medium | Medium | CI on Windows/Linux/macOS |

---

## Success Criteria

### Phase 1 (Data Structures)
- Rust structs compile and bind to Python
- Round-trip data transfer verified

### Phase 2 (Hot Paths)
- 5x speedup on kernel evaluation
- Parity with Python reference

### Phase 3 (Full Solvers)
- 10x speedup on full frame
- No visual regression

### Phase 4 (GPU Compute)
- 100x particle count increase
- 60 FPS with 100K+ particles

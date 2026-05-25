# RECOMMENDATIONS: engine/animation/crowds + engine/animation/facial

---

## Rust Bridge Requirements

### High Priority

| Component | Current State | Rust Bridge Benefit | Estimated Speedup |
|-----------|---------------|---------------------|-------------------|
| **Crowd Avoidance Loop** | O(n^2) Python iteration | SIMD-vectorized distance checks, spatial partitioning | 50-100x |
| **Animation Texture Encoding** | Python float-to-RGBA8 | Rust with SIMD packing | 20-50x |
| **Cubic Hermite Interpolation** | Python math | Rust with inline SIMD | 10-30x |

**Rationale**: These are per-frame hot paths executed for every agent/bone. Python loop overhead dominates; Rust eliminates it.

#### Suggested Rust API (High Priority)

```rust
// Crowd avoidance batch processing
#[pyfunction]
fn compute_avoidance_batch(
    positions: Vec<[f32; 3]>,
    velocities: Vec<[f32; 3]>,
    priorities: Vec<f32>,
    avoidance_radius: f32,
    avoidance_strength: f32,
) -> Vec<[f32; 3]>; // Returns avoidance vectors

// Animation texture encoding
#[pyfunction]
fn encode_animation_frame(
    bone_positions: Vec<[f32; 3]>,
    bone_scales: Vec<[f32; 3]>,
    bone_rotations: Vec<[f32; 4]>,
) -> Vec<u8>; // Returns RGBA8 packed data
```

### Medium Priority

| Component | Current State | Rust Bridge Benefit | Estimated Speedup |
|-----------|---------------|---------------------|-------------------|
| **FACS Weight Combination** | Python dict operations | Vectorized AU blending | 5-15x |
| **Blend Shape Evaluation** | NumPy sparse ops | Rust sparse matrix | 3-10x |
| **LOD Bone Importance** | Python list sorting | Rust parallel sort | 5-10x |

**Rationale**: These run per-character per-frame but are less frequent than crowd avoidance.

#### Suggested Rust API (Medium Priority)

```rust
// FACS expression blending
#[pyfunction]
fn blend_action_units(
    au_weights: HashMap<u32, f32>,
    au_left_weights: HashMap<u32, f32>,
    au_right_weights: HashMap<u32, f32>,
    blend_factor: f32,
) -> HashMap<u32, (f32, f32)>; // Returns blended (left, right) weights

// Sparse blend shape evaluation
#[pyfunction]
fn evaluate_blend_shapes(
    base_vertices: &[f32],
    shape_indices: &[Vec<u32>],
    shape_deltas: &[Vec<[f32; 3]>],
    weights: &[f32],
) -> Vec<f32>; // Returns deformed vertices
```

### Low Priority

| Component | Current State | Rust Bridge Benefit | Estimated Speedup |
|-----------|---------------|---------------------|-------------------|
| **Eye Vergence** | Python trig | Rust trig (minimal) | 2-5x |
| **Blink Controller** | Python random | Rust random (minimal) | 2-3x |
| **Lip Sync Timing** | Python time calc | Low benefit | 1-2x |

**Rationale**: These run once per character per frame; Python overhead is acceptable.

---

## Integration Strategy

### Phase 1: Crowd Acceleration (2 weeks)

1. **Implement Spatial Partitioning in Rust**
   - Create uniform grid or octree for O(n log n) neighbor queries
   - Expose `build_spatial_index()` and `query_neighbors()` via PyO3

2. **Port Avoidance Loop to Rust**
   - Match existing Python semantics exactly
   - Batch process all agents in single call
   - Return Vec3 avoidance vectors

3. **Validate Output Parity**
   - Compare Rust vs Python outputs for test scenarios
   - Ensure edge cases (coincident agents, zero distance) match

### Phase 2: Animation Texture Acceleration (1 week)

1. **Port RGBA8 Encoding to Rust**
   - Use SIMD for float-to-byte conversion
   - Process entire frame in single call

2. **Port Cubic Hermite to Rust**
   - Inline interpolation with SIMD
   - Batch process all bones per frame

### Phase 3: Facial Acceleration (1 week)

1. **Port FACS Blending to Rust**
   - Vectorize AU weight combination
   - Handle bilateral asymmetry

2. **Optional: Sparse Blend Shape Evaluation**
   - Only if profiling shows NumPy is bottleneck

### Integration Pattern

```python
# Python wrapper with Rust fallback
try:
    from omega import compute_avoidance_batch_rs
    def compute_avoidance(agents):
        return compute_avoidance_batch_rs(
            [a.position for a in agents],
            [a.velocity for a in agents],
            [a.priority for a in agents],
            self._avoidance_radius,
            self._avoidance_strength,
        )
except ImportError:
    def compute_avoidance(agents):
        # Original Python implementation
        ...
```

---

## Testing Strategy

### Unit Tests

| Test Category | Description | Priority |
|---------------|-------------|----------|
| Avoidance Edge Cases | Coincident agents, zero distance, max priority | HIGH |
| RGBA8 Encoding Precision | Float round-trip accuracy | HIGH |
| Cubic Hermite Continuity | C1 continuity at keyframes | MEDIUM |
| FACS Bilateral Symmetry | Left/right weight independence | MEDIUM |
| Eye Vergence Limits | Max vergence angle clamping | LOW |

### Integration Tests

| Test Category | Description | Priority |
|---------------|-------------|----------|
| Crowd 1000+ Agents | Performance at scale | HIGH |
| Rust/Python Parity | Output equivalence for all paths | HIGH |
| Animation Texture Atlas | Multi-clip UV correctness | MEDIUM |
| Lip Sync Rapid Phonemes | <50ms phoneme sequences | MEDIUM |
| Face Rig Layer Priority | Override vs additive blending | LOW |

### Performance Tests

| Test Category | Baseline | Target | Tool |
|---------------|----------|--------|------|
| Crowd Avoidance 1000 agents | TBD ms | <5ms | pytest-benchmark |
| Animation Texture 100 bones | TBD ms | <1ms | pytest-benchmark |
| FACS Blend 18 AUs | TBD us | <100us | pytest-benchmark |

### Test Data

- Crowd: Generate random agent positions in grid pattern
- Animation: Use captured mocap data or procedural sine waves
- Facial: Use phoneme sequences from real speech samples

---

## Risk Assessment

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Rust/Python Semantic Drift** | Behavioral differences between implementations | Comprehensive parity tests, property-based testing |
| **Spatial Index Rebuild Cost** | Per-frame index rebuild may dominate | Incremental update or frame-to-frame coherence |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **NumPy vs Rust Data Copy** | Memory copies negate speedup | Use buffer protocol, minimize copies |
| **SIMD Portability** | Not all CPUs support AVX2/AVX-512 | Runtime feature detection, scalar fallback |
| **ARKit Blend Shape Changes** | Apple may update blend shape names | Abstract behind enum, version-check at runtime |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Coarticulation Accuracy** | Lip sync may not match live capture | Tunable anticipation/carryover parameters |
| **LOD Pop-in** | Visible transitions despite hysteresis | Increase hysteresis threshold if noticeable |

---

## Dependencies

### External Packages (Rust)

| Package | Purpose | Version |
|---------|---------|---------|
| `pyo3` | Python bindings | >=0.20 |
| `numpy` (pyo3-numpy) | NumPy array access | >=0.20 |
| `rayon` | Parallel iteration | >=1.7 |
| `glam` | Vector math | >=0.24 |
| `bytemuck` | Safe byte casting | >=1.14 |

### External Packages (Python)

| Package | Purpose | Current |
|---------|---------|---------|
| `numpy` | Array operations | existing |
| `dataclasses` | Data structures | stdlib |

---

## Success Criteria

1. **Performance**: Crowd avoidance for 1000 agents <5ms per frame
2. **Parity**: Rust and Python implementations produce identical output for all test cases
3. **Stability**: No crashes or memory leaks under extended runtime (1+ hour)
4. **Integration**: Python code seamlessly uses Rust when available, falls back otherwise
5. **Test Coverage**: >90% line coverage for Rust code, >80% for Python wrappers

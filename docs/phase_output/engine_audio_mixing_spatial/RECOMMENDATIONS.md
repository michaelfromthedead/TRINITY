# RECOMMENDATIONS: engine/audio/mixing + engine/audio/spatial

---

## Rust Bridge Requirements

### High Priority

| Requirement | Rationale | Complexity |
|-------------|-----------|------------|
| **HRTF Convolution** | Per-sample FIR filtering is CPU-intensive; Rust SIMD can provide 4-8x speedup | High |
| **FFT-based Reverb** | Convolution reverb requires FFT/IFFT; existing Rust FFT crates (rustfft) are mature | Medium |
| **Real-time Mixing** | The 8-stage pipeline touches every active source every frame; Rust removes GIL contention | High |
| **Occlusion Raycasting** | Multi-ray occlusion with geometry queries benefits from Rust spatial data structures | Medium |

### Medium Priority

| Requirement | Rationale | Complexity |
|-------------|-----------|------------|
| **Propagation Cache** | Cache invalidation and path recalculation could be lock-free in Rust | Medium |
| **VBAP Triangulation** | Full 3D VBAP requires Delaunay triangulation; Rust has proven geometry crates | Medium |
| **Material Lookups** | 6-band absorption lookups per surface per ray benefit from cache-friendly Rust layout | Low |
| **Ambisonics Encoding** | Spherical harmonic calculations are math-heavy; Rust SIMD helps | Low |

### Low Priority

| Requirement | Rationale | Complexity |
|-------------|-----------|------------|
| **Snapshot Interpolation** | Currently fast enough in Python; bridge only if profiling shows bottleneck | Low |
| **Doppler Pitch Shift** | Simple formula; low call frequency | Low |
| **Speaker Config** | Static data; no runtime benefit from Rust | Minimal |

---

## Integration Strategy

### Phase 1: Core DSP Bridge

1. Add audio functions to `crates/renderer-backend/src/bridge.rs`:
   ```rust
   #[pyfunction]
   fn hrtf_convolve(input: &[f32], left_filter: &[f32], right_filter: &[f32]) -> (Vec<f32>, Vec<f32>);
   
   #[pyfunction]
   fn fft_convolve(input: &[f32], ir: &[f32]) -> Vec<f32>;
   
   #[pyfunction]
   fn mix_buffers(sources: Vec<&[f32]>, gains: Vec<f32>) -> Vec<f32>;
   ```

2. Create `engine/audio/bridge.py` to wrap Rust functions:
   ```python
   try:
       from _omega import hrtf_convolve, fft_convolve, mix_buffers
       USE_RUST = True
   except ImportError:
       USE_RUST = False
   ```

3. Modify `hrtf.py` to use Rust when available:
   ```python
   if bridge.USE_RUST:
       left, right = bridge.hrtf_convolve(input, self._left_filter, self._right_filter)
   else:
       # Python fallback
   ```

### Phase 2: Geometry Integration

1. Share spatial acceleration structure with renderer:
   - BVH from `renderer-backend` can serve audio occlusion
   - Single raycaster for both visual and audio

2. Add audio-specific geometry queries:
   ```rust
   #[pyfunction]
   fn cast_occlusion_rays(
       source: [f32; 3],
       listener: [f32; 3],
       num_rays: u32,
       spread: f32
   ) -> OcclusionResult;
   ```

### Phase 3: GPU Compute

1. Move HRTF batch processing to compute shader:
   - Upload filter banks to GPU
   - Process multiple sources in parallel
   - Download mixed output

2. Integrate with frame graph:
   - Audio compute pass runs before/after render passes
   - Sync points prevent stalls

---

## Testing Strategy

### Unit Tests (Immediate)

| Test Area | Test Cases |
|-----------|------------|
| Ducking FSM | State transitions, timing accuracy |
| HDR Window | Mapping correctness, edge cases (below floor, above ceiling) |
| Sidechain | Gain reduction calculation, soft-knee behavior |
| VBAP | Gain normalization, fallback to stereo |
| Woodworth ITD | Sample count accuracy vs published data |
| RT60 | Sabine and Eyring match hand calculations |
| Attenuation | Linear/log/inverse curves match formulas |

### Integration Tests (Short-term)

| Test Area | Test Cases |
|-----------|------------|
| Mixer Pipeline | All 8 stages execute in order |
| Bus Routing | Pre/post fader sends, direct outputs |
| Snapshot Transitions | Blend between states |
| Spatial -> Mixer | Spatialized sources route to correct buses |
| Thread Safety | Concurrent updates don't crash or deadlock |

### Performance Tests (Medium-term)

| Test Area | Target |
|-----------|--------|
| Mixer tick latency | <1ms for 64 sources |
| HRTF processing | <5ms for 16 binaural sources |
| Propagation update | <2ms for 32 sources |
| Occlusion raycasting | <1ms for 8 rays/source |

### Benchmark Suite (Long-term)

- Compare Python vs Rust implementations
- Profile cache hit rates for propagation
- Measure memory usage under load
- Stress test with 200+ simultaneous sources

---

## Risk Assessment

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| GIL contention in mixer tick | Frame drops, audio glitches | Move to Rust; use threading.RLock minimally |
| HRTF latency | Perceptible delay | GPU batch processing; limit active binaural sources |
| No test coverage | Regressions go unnoticed | Add tests before GRANDPHASE2 changes |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Propagation cache stale | Incorrect spatial audio | Add cache invalidation on geometry change |
| Material database incomplete | Unrealistic reverb | Allow runtime material registration |
| Ambisonics decoder mismatch | Phase issues | Validate against reference decoder |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Doppler pitch artifacts | Zipper noise | Add smoothing |
| Snapshot blend artifacts | Pop/click | Already uses interpolation |
| Speaker config errors | Channels swapped | Add validation on load |

---

## Prioritized Action Plan

### Week 1-2
1. Add test suite for mixing (ducking, HDR, sidechain)
2. Add test suite for spatial (HRTF, VBAP, occlusion)
3. Profile current Python performance baseline

### Week 3-4
4. Implement Rust HRTF convolution in bridge.rs
5. Create engine/audio/bridge.py wrapper
6. Benchmark Rust vs Python HRTF

### Week 5-6
7. Implement Rust mixer tick core
8. Add FFT-based reverb to bridge
9. Integrate with frame graph for sync

### Week 7-8
10. GPU compute for batch HRTF
11. Share BVH with renderer for occlusion
12. Final performance validation

---

## Dependencies on Other GAPSET_3_BRIDGE Phases

| Phase | Dependency |
|-------|------------|
| PHASE_0 (Scaffolding) | PyO3 setup complete |
| PHASE_1 (Type Channel) | TypeRegistry for audio component types |
| PHASE_2 (Component Store) | ComponentStore for audio source data |
| PHASE_7 (Frame Graph) | Frame graph for audio compute passes |
| PHASE_10 (GPU Memory) | GPU buffer allocation for audio data |

Audio bridge should be added as a **new phase** after PHASE_10, or integrated into PHASE_9 (Full Features).

# RECOMMENDATIONS: engine/audio/dialogue + engine/audio/dsp

---

## Rust Bridge Requirements

### High Priority

| Component | Rationale | Effort |
|-----------|-----------|--------|
| Biquad Filter Core | Tight sample-by-sample loop, SIMD opportunity | Medium |
| Compressor Envelope | Per-sample exponential, potential for vectorization | Medium |
| Comb/Allpass Filters | Reverb hot path, buffer operations | Medium |
| Convolution FFT | NumPy FFT acceptable but Rust SIMD FFT faster | High |

**Recommendation**: Create `crates/audio-dsp` with PyO3 bindings. Start with biquad and envelope follower as proof of concept.

### Medium Priority

| Component | Rationale | Effort |
|-----------|-----------|--------|
| Granular Pitch Shift | Window + resample in inner loop | High |
| Delay Line | Interpolation can be vectorized | Low |
| DSP Graph Execution | Graph traversal in Rust, processing in Rust | High |

**Recommendation**: Evaluate after profiling. Python orchestration with Rust primitives may be sufficient.

### Low Priority

| Component | Rationale | Effort |
|-----------|-----------|--------|
| Dialogue State Machine | Logic bound, not compute bound | N/A |
| Localization | I/O bound (file loading) | N/A |
| VO Queue | Python heapq is fast enough | N/A |

**Recommendation**: Keep in Python. No performance benefit from Rust.

---

## Integration Strategy

### Phase 1: Foundation
1. Create `crates/audio-dsp/Cargo.toml` with PyO3
2. Define Rust traits: `AudioProcessor`, `DspNode`
3. Implement biquad filter in Rust
4. Create Python wrapper in `engine/audio/dsp/rust_biquad.py`
5. Benchmark against pure Python biquad

### Phase 2: Core DSP
1. Port envelope follower to Rust
2. Port compressor gain calculation to Rust
3. Port comb/allpass filters to Rust
4. Create `RustReverb` wrapper class
5. Benchmark reverb processing

### Phase 3: Graph Integration
1. Keep DSP graph in Python (orchestration)
2. Process blocks via Rust nodes
3. Use shared memory for inter-node buffers
4. Minimize Python/Rust boundary crossings

### Phase 4: Full Migration (Optional)
1. Move entire DSP graph to Rust
2. Expose high-level API to Python
3. Keep dialogue system in Python
4. Python becomes thin orchestration layer

---

## Testing Strategy

### Unit Tests (per algorithm)
```
tests/audio/dsp/test_biquad.py
tests/audio/dsp/test_compressor.py
tests/audio/dsp/test_reverb.py
tests/audio/dialogue/test_vo_queue.py
tests/audio/dialogue/test_conversation.py
```

### Integration Tests
- Test complete DSP chains
- Test dialogue manager with mocked VO
- Test localization switching mid-conversation

### Performance Benchmarks
```
benchmarks/dsp/bench_biquad.py      # Python vs Rust biquad
benchmarks/dsp/bench_reverb.py      # Full reverb processing
benchmarks/dsp/bench_chain.py       # Effect chain latency
```

### Audio Quality Tests
- Reference signal comparison (known input/output pairs)
- Frequency response verification for filters
- Impulse response verification for reverb

### Fuzz Tests
- Random parameter changes during processing
- Edge cases: SR change mid-stream, extreme parameters
- Memory stress: many simultaneous VO streams

---

## Risk Assessment

### Low Risk
| Risk | Mitigation |
|------|------------|
| Python performance adequate | NumPy handles most vectorization |
| Thread safety | RLock already in place |
| API stability | Well-defined DSPNode interface |

### Medium Risk
| Risk | Mitigation |
|------|------------|
| Rust/Python latency | Batch processing reduces crossings |
| Memory alignment | NumPy handles alignment; Rust must match |
| Sample rate mismatch | Coefficient recalc on SR change |

### High Risk
| Risk | Mitigation |
|------|------------|
| FFI complexity | Start with simple primitives |
| SIMD portability | Use Rust std::simd or portable_simd |
| Debug tooling | Keep Python fallback implementations |

---

## Recommended Next Steps

1. **Profile Current Implementation**
   - Measure CPU usage of DSP processing
   - Identify actual hot paths
   - Determine if Rust is needed at all

2. **Create Minimal Rust Bridge**
   - `crates/audio-dsp` with biquad only
   - Benchmark against Python
   - Validate correctness

3. **Document Audio API**
   - DSPNode interface specification
   - Effect parameter ranges
   - Threading model

4. **Add Test Coverage**
   - Current code likely has implicit testing via integration
   - Add explicit unit tests before any refactoring

---

## GRANDPHASE2 Alignment

The audio subsystem is **production-ready** in its current form. GRANDPHASE2 should:

1. **Defer Rust migration** until profiling proves necessity
2. **Maintain Python API** stability for game code
3. **Follow GAPSET_3_BRIDGE patterns** if bridge is needed
4. **Prioritize renderer bridge** over audio bridge

Audio DSP is well-isolated and can be optimized incrementally without blocking other engine work.

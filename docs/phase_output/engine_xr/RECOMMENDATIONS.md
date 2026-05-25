# RECOMMENDATIONS.md - engine_xr

## Rust Bridge Requirements

### High Priority

| Component | Rationale | Estimated Effort |
|-----------|-----------|------------------|
| **IK Solvers** | FABRIK/CCD/TwoBone run every frame for avatar at 90Hz; SIMD acceleration critical | 2-3 weeks |
| **VRS Rate Image Generation** | Per-frame GPU memory writes; compute shader preferred over CPU | 1 week |
| **Hand Joint Batch Updates** | 26 joints * 2 hands * 90Hz = 4,680 updates/sec; batch FFI | 1 week |
| **Pose Prediction** | Latency-sensitive; Rust async or dedicated thread | 1-2 weeks |

### Medium Priority

| Component | Rationale | Estimated Effort |
|-----------|-----------|------------------|
| **Spatial Mesh Decimation** | MeshMapping outputs high-poly; need LOD pipeline | 2 weeks |
| **Plane Detection Raycast** | Many planes + high query rate = hot path | 1 week |
| **Gesture Smoothing Buffer** | Fixed-size ring buffer for velocity; cache-friendly | 3 days |
| **Throw Velocity Calculation** | Deque operations; consider lock-free Rust queue | 3 days |

### Low Priority

| Component | Rationale | Estimated Effort |
|-----------|-----------|------------------|
| **Teleport Arc Physics** | Already O(n) projectile sim; not frame-critical | 1 week |
| **Config Validation** | Frozen dataclasses sufficient; Rust validation optional | 3 days |
| **Platform Capability Detection** | One-time startup cost; Python adequate | N/A |

## Integration Strategy

### Phase 1: Core Bridge (Weeks 1-4)
1. Expose Rust `IKSolver` trait with Python binding via PyO3
2. Implement `FABRIKSolver`, `CCDSolver`, `TwoBoneSolver` in Rust
3. Keep Python fallback for non-accelerated platforms
4. Benchmark: target <1ms for full avatar IK (currently ~3-5ms estimated)

### Phase 2: Rendering Bridge (Weeks 5-6)
1. Add Rust function to generate VRS rate image into GPU buffer
2. Integrate with frame_graph for automatic scheduling
3. Remove Python foveation image generation path

### Phase 3: Input Bridge (Weeks 7-8)
1. Rust-side hand joint buffer with atomic updates
2. Python reads via memory view (zero-copy)
3. Gesture recognition remains Python (logic-heavy, not compute-heavy)

### Phase 4: Spatial Bridge (Weeks 9-10)
1. Rust mesh processing for MeshMapping
2. Plane detection acceleration
3. Anchor serialization/deserialization

## Testing Strategy

### Unit Tests
```
tests/
  test_ik_solver.py         # FABRIK/CCD/TwoBone convergence tests
  test_foveated.py          # VRS image generation correctness
  test_hand_tracking.py     # Gesture recognition accuracy
  test_plane_detection.py   # Raycast, contains_point, area
  test_teleport.py          # Arc physics, landing calculation
  test_grabbable.py         # Grab state, throw velocity
```

### Integration Tests
- XR runtime lifecycle (init -> session -> frame loop -> shutdown)
- Input pipeline (tracking data -> gesture events)
- Rendering pipeline (foveation -> VRS -> compositor)
- Interaction pipeline (hover -> select -> grab -> throw)

### Performance Tests
- IK solver: 1000 iterations, measure p50/p95/p99 latency
- Foveation: 4K resolution rate image generation time
- Hand tracking: 26-joint batch update throughput
- Plane raycast: 100 planes, 1000 rays, hit rate + latency

### Hardware Tests (Requires HMD)
- OpenXR loader integration with Quest/Index/Reverb
- WebXR in browser with Meta Quest Browser
- End-to-end latency measurement (photon-to-motion)

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **OpenXR API changes** | Low | Medium | Pin to OpenXR 1.0, follow 1.1 roadmap |
| **Platform-specific bugs** | Medium | High | Extensive device matrix testing |
| **Performance regression** | Medium | High | Automated benchmarks in CI |
| **Python/Rust FFI overhead** | Low | Medium | Batch operations, memory views |
| **Gesture false positives** | Medium | Low | Hysteresis tuning, user calibration |
| **Motion sickness reports** | High | High | Default to conservative comfort settings |
| **Hand tracking accuracy** | Medium | Medium | Fallback to controller input |
| **Spatial anchor drift** | Medium | Medium | Cloud anchor re-localization |

### Critical Path Risks
1. **Native OpenXR binding**: Without this, no production HMD support. Must be GRANDPHASE2 Week 1 priority.
2. **IK performance**: If Rust IK not ready, avatar quality degrades at 90Hz. Fallback: reduce IK complexity.
3. **VRS driver support**: Not all GPUs support VRS. Fallback: resolution scaling.

## Recommended GRANDPHASE2 Sprint Plan

| Sprint | Focus | Deliverables |
|--------|-------|--------------|
| Sprint 1 | OpenXR Native | ctypes OpenXR binding, device enumeration |
| Sprint 2 | OpenXR Native | Session management, frame loop, swapchain |
| Sprint 3 | Rust IK | FABRIK/TwoBone Rust impl, PyO3 binding |
| Sprint 4 | Rust IK | CCD Rust impl, benchmark suite, CI integration |
| Sprint 5 | Rendering | VRS compute shader, frame_graph integration |
| Sprint 6 | Input | Rust hand joint buffer, zero-copy Python access |
| Sprint 7 | Testing | Device matrix testing, performance benchmarks |
| Sprint 8 | Polish | Documentation, examples, comfort tuning |

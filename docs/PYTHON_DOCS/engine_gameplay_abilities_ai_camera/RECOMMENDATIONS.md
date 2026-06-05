# RECOMMENDATIONS: engine/gameplay/abilities, ai, camera

---

## Rust Bridge Requirements

### High Priority
**None.** These modules are feature-complete in Python and do not block GAPSET_3_BRIDGE.

### Medium Priority

| Component | Rationale | Bridge Type |
|-----------|-----------|-------------|
| Camera View/Projection Matrices | GPU uniform upload requires camera state | Data Channel (read-only) |
| AttributeSet Current Values | ECS queries may need attribute data | Data Channel (read-only) |

### Low Priority

| Component | Rationale | Bridge Type |
|-----------|-----------|-------------|
| GOAP Planner | Performance optimization for large action sets | Compute Channel |
| Utility AI Scoring | Performance optimization for many considerations | Compute Channel |
| Behavior Tree Tick | Unlikely bottleneck, Python sufficient | Not recommended |
| Camera Spline Evaluation | GPU compute shader alternative | Compute Channel |

---

## Integration Strategy

### Phase 1: No Bridge (Current)
- Keep all gameplay logic in Python
- Use existing Python→Rust type channel for component schemas
- Camera state accessible via Python API only

### Phase 2: Camera Data Channel (If Needed)
```
Python                          Rust
camera/controller.py  ──────►  frame_graph/
  - get_view_matrix()            - CameraUniforms buffer
  - get_projection_matrix()      - Upload per frame
  - get_position()
```

Implementation path:
1. Add `camera_state_to_buffer()` in Python that serializes camera state
2. Add PyO3 function `upload_camera_uniforms(buffer: bytes)` in Rust
3. Call from frame graph's pre-render phase

### Phase 3: ECS Attribute Integration (If Needed)
```
Python                          Rust
abilities/attributes.py ──────► component_store.rs
  - AttributeSet                  - Column<f32> for current values
  - On change: notify Rust        - Direct GPU upload path
```

Implementation path:
1. Store attribute values in ComponentStore alongside transforms
2. Dirty-flag system to batch updates
3. Rust-side query for AI/physics systems that need attribute data

---

## Testing Strategy

### Unit Tests (Python)
All modules have implicit test coverage through their extensive implementations. Recommend adding:

| Test Suite | Coverage Target |
|------------|-----------------|
| test_effects.py | Effect lifecycle, modifier stacking, tag filtering |
| test_attributes.py | Modifier order of operations, bounds clamping |
| test_tags.py | Hierarchy matching, wildcard patterns, LRU cache |
| test_targeting.py | Area shape calculations, target filtering |
| test_behavior_tree.py | Node execution, abort handling, parallel policies |
| test_goap.py | A* search, plan caching, goal prioritization |
| test_utility_ai.py | Response curves, scoring, action selection |
| test_blackboard.py | TTL expiration, observers, typed keys |
| test_camera_rails.py | Spline interpolation, arc-length parameterization |
| test_camera_effects.py | Shake trauma, DOF focus, blend curves |
| test_camera_collision.py | Collision response modes, occlusion fading |

### Integration Tests
| Test | Description |
|------|-------------|
| Ability→Attribute | Effect applies modifier, attribute value changes |
| AI→Blackboard | GOAP/BT/Utility all read/write shared blackboard |
| Camera→Controller | Input → controller update → state change |
| Camera→Collision | Controller position → collision check → adjusted position |

### Performance Benchmarks
| Benchmark | Threshold | Action if Exceeded |
|-----------|-----------|-------------------|
| GOAP planning (100 actions) | <1ms | Consider Rust migration |
| Utility scoring (50 considerations) | <0.5ms | Consider Rust migration |
| Spline evaluation (1000 samples) | <2ms | Consider GPU compute |
| Attribute recalculation (100 modifiers) | <0.1ms | Consider Rust storage |

---

## Risk Assessment

### Low Risk
| Risk | Mitigation |
|------|------------|
| Performance bottleneck in AI | Profile before optimizing; Python likely sufficient |
| Camera jitter from collision | Existing hysteresis and interpolation should handle |
| Modifier stack overflow | Max bounds already implemented |

### Medium Risk
| Risk | Mitigation |
|------|------------|
| GOAP infinite loop | max_iterations limit exists (GOAP_MAX_ITERATIONS) |
| Blackboard memory leak | TTL expiration should clean up; add periodic sweep if needed |
| Spline discontinuity | Arc-length parameterization handles non-uniform control points |

### No Significant Risks Identified
- Code quality is high
- Abstractions are well-designed
- Edge cases appear handled
- No blocking dependencies on unfinished systems

---

## Summary

These subsystems are **production-ready** and require **no immediate Rust bridge work**. They operate above the rendering boundary and interface with the engine through well-defined Python APIs.

Future bridge integration should be driven by:
1. **Performance profiling data** showing actual bottlenecks
2. **GPU data path requirements** (camera uniforms, attribute visualization)
3. **ECS query patterns** that benefit from Rust-side data locality

Until then, recommend keeping all gameplay logic in Python for maximum iteration speed.

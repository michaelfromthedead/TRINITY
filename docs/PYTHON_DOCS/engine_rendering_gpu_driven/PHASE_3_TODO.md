# PHASE 3 TODO: Visibility Buffer Rendering

## Summary

Validate and prepare the visibility buffer system (visibility_buffer.py, 836 lines) for GPU fragment/compute shader port.

---

## T-VIS-3.1: Validate Visibility Data Bit-Packing

**Description**: Verify 32-bit format correctly encodes 12-bit triangle + 20-bit instance.

**Acceptance Criteria**:
- [ ] Pack/unpack round-trip is lossless
- [ ] Test maximum triangle ID (4095)
- [ ] Test maximum instance ID (1048575)
- [ ] Test zero values
- [ ] Test edge cases (boundary values)
- [ ] Verify bit masks are correct

**Files**: `engine/rendering/gpu_driven/visibility_buffer.py` lines 50-100

**Estimate**: 1 hour

---

## T-VIS-3.2: Validate Edge Function Rasterization

**Description**: Verify software rasterizer handles all triangle orientations.

**Acceptance Criteria**:
- [ ] Test CW and CCW triangle winding
- [ ] Test degenerate triangles (zero area)
- [ ] Test triangles at screen edges
- [ ] Test sub-pixel triangles
- [ ] Verify coverage matches expected pixels
- [ ] Test barycentric coordinate computation

**Files**: `engine/rendering/gpu_driven/visibility_buffer.py` lines 439-475

**Estimate**: 3 hours

---

## T-VIS-3.3: Validate Depth Test Integration

**Description**: Verify depth buffer integration with visibility writes.

**Acceptance Criteria**:
- [ ] Closer fragments overwrite farther fragments
- [ ] Equal depth handled correctly (first-wins or last-wins)
- [ ] Test overlapping triangles
- [ ] Test reverse-Z depth range (1.0 = near, 0.0 = far)
- [ ] Verify atomic operation semantics
- [ ] Test depth precision at far distances

**Files**: `engine/rendering/gpu_driven/visibility_buffer.py` lines 475-530

**Estimate**: 2 hours

---

## T-VIS-3.4: Validate Material Tile Classification

**Description**: Verify 8x8 tile material grouping.

**Acceptance Criteria**:
- [ ] Test single-material tile (common case)
- [ ] Test multi-material tile (mixed materials)
- [ ] Test empty tile (no geometry)
- [ ] Verify tile boundary alignment
- [ ] Test screen edges (partial tiles)
- [ ] Measure material coherence ratio

**Files**: `engine/rendering/gpu_driven/visibility_buffer.py` lines 550-620

**Estimate**: 2 hours

---

## T-VIS-3.5: Validate Deferred Texturing Position Reconstruction

**Description**: Verify world position reconstruction from depth.

**Acceptance Criteria**:
- [ ] Test center pixel position
- [ ] Test corner pixel positions
- [ ] Test near plane depth
- [ ] Test far plane depth
- [ ] Verify precision across depth range
- [ ] Test with various projection matrices

**Files**: `engine/rendering/gpu_driven/visibility_buffer.py` lines 700-750

**Estimate**: 2 hours

---

## T-VIS-3.6: Validate Material Sorting Pass

**Description**: Verify tiles are correctly sorted by material for dispatch.

**Acceptance Criteria**:
- [ ] Test single material (all tiles same material)
- [ ] Test many materials (random distribution)
- [ ] Verify tile list per material is complete
- [ ] Test indirect dispatch buffer generation
- [ ] Verify no tiles lost in sorting
- [ ] Benchmark sorting performance

**Files**: `engine/rendering/gpu_driven/visibility_buffer.py` lines 620-700

**Estimate**: 2 hours

---

## T-VIS-3.7: Validate Full Pipeline Integration

**Description**: Test complete visibility buffer pipeline end-to-end.

**Acceptance Criteria**:
- [ ] Test simple scene (single object, single material)
- [ ] Test complex scene (multiple objects, materials)
- [ ] Verify final color output matches reference
- [ ] Test with multiple overlapping objects
- [ ] Profile per-stage timing
- [ ] Document integration points

**Files**: `engine/rendering/gpu_driven/visibility_buffer.py` lines 750-836

**Estimate**: 3 hours

---

## T-VIS-3.8: Create GPU Buffer Layout Specification

**Description**: Document buffer layouts for GPU shader port.

**Acceptance Criteria**:
- [ ] Define visibility buffer texture format (R32_UINT)
- [ ] Define depth buffer texture format
- [ ] Document material tile buffer format
- [ ] Document indirect dispatch buffer format
- [ ] Create WGSL struct definitions
- [ ] Document atomic operation requirements

**Output**: `docs/gpu_driven/visibility_buffer_spec.md`

**Estimate**: 2 hours

---

## T-VIS-3.9: Write WGSL Visibility Buffer Shaders

**Description**: Create WGSL shaders for visibility buffer pipeline.

**Acceptance Criteria**:
- [ ] Visibility fill fragment shader (atomic min)
- [ ] Material tile classifier compute shader
- [ ] Material sorting compute shader (prefix sum)
- [ ] Deferred texturing compute shader
- [ ] Proper binding groups and buffer layouts
- [ ] Position reconstruction utilities

**Output**: `engine/rendering/shaders/visibility_buffer/`

**Estimate**: 5 hours

---

## Phase 3 Totals

| Category | Count |
|----------|-------|
| Tasks | 9 |
| Estimated Hours | 22 |
| Test Coverage | 7 validation tasks |
| Documentation | 1 spec task |
| Implementation | 1 shader task |

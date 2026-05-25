# PHASE 1 TODO: Culling Pipeline

## Summary

Validate and prepare the culling pipeline (culling.py, 1,109 lines) for GPU port.

---

## T-CULL-1.1: Validate Frustum Plane Extraction

**Description**: Verify Gribb-Hartmann plane extraction produces correct 6 planes from view-projection matrix.

**Acceptance Criteria**:
- [ ] Unit tests for perspective projection matrix
- [ ] Unit tests for orthographic projection matrix
- [ ] Verify planes are normalized (length = 1.0)
- [ ] Test point-inside-frustum returns correct results
- [ ] Test sphere-inside-frustum handles edge cases (partial intersection)
- [ ] Test AABB-inside-frustum handles axis-aligned boxes

**Files**: `engine/rendering/gpu_driven/culling.py` lines 202-272

**Estimate**: 2 hours

---

## T-CULL-1.2: Validate HZB Pyramid Construction

**Description**: Verify HZB mip pyramid builds correct mip chain with max-depth reduction.

**Acceptance Criteria**:
- [ ] Unit test for 2x2 depth buffer (single reduction)
- [ ] Unit test for power-of-two dimensions (4x4, 8x8, 16x16)
- [ ] Unit test for non-power-of-two dimensions
- [ ] Verify max-reduction preserves closest depth per tile
- [ ] Test mip level count calculation
- [ ] Verify boundary handling for odd dimensions

**Files**: `engine/rendering/gpu_driven/culling.py` lines 539-577

**Estimate**: 2 hours

---

## T-CULL-1.3: Validate HZB Occlusion Query

**Description**: Verify screen-space AABB projection and HZB depth test.

**Acceptance Criteria**:
- [ ] Test AABB projection to screen coordinates
- [ ] Test mip level selection based on screen rect size
- [ ] Test depth comparison (object_depth vs pyramid_depth)
- [ ] Verify conservative test never culls visible objects
- [ ] Test objects fully behind occluder
- [ ] Test objects partially visible around occluder

**Files**: `engine/rendering/gpu_driven/culling.py` lines 578-650

**Estimate**: 3 hours

---

## T-CULL-1.4: Validate Distance Culler with LOD

**Description**: Verify distance culling integrates correctly with LOD selection.

**Acceptance Criteria**:
- [ ] Test distance calculation from camera position
- [ ] Test LOD index selection based on distance thresholds
- [ ] Test cull distance (max LOD distance)
- [ ] Test LOD bias application
- [ ] Test objects at LOD boundary distances
- [ ] Verify continuous LOD transition (no popping)

**Files**: `engine/rendering/gpu_driven/culling.py` lines 700-800

**Estimate**: 2 hours

---

## T-CULL-1.5: Validate Small Feature Culler

**Description**: Verify screen-space size culling for sub-pixel geometry.

**Acceptance Criteria**:
- [ ] Test screen-space size calculation from sphere + distance
- [ ] Test minimum pixel threshold (configurable)
- [ ] Test objects at screen edge (partial visibility)
- [ ] Verify perspective-correct size estimation
- [ ] Test interaction with LOD system

**Files**: `engine/rendering/gpu_driven/culling.py` lines 850-950

**Estimate**: 1.5 hours

---

## T-CULL-1.6: Validate CullingPipeline Composition

**Description**: Verify pipeline sequences cullers correctly with early-out behavior.

**Acceptance Criteria**:
- [ ] Test pipeline with all stages enabled
- [ ] Test pipeline with stages disabled
- [ ] Verify early-out (culled objects skip later stages)
- [ ] Test statistics tracking per stage
- [ ] Test custom pipeline order
- [ ] Benchmark pipeline throughput (objects/second)

**Files**: `engine/rendering/gpu_driven/culling.py` lines 950-1100

**Estimate**: 2 hours

---

## T-CULL-1.7: Create GPU Buffer Layout Specification

**Description**: Document buffer layouts for GPU compute shader port.

**Acceptance Criteria**:
- [ ] Define CullInput struct layout (vec3 alignment)
- [ ] Define CullOutput struct layout
- [ ] Document frustum uniform buffer format
- [ ] Document HZB texture format (R32_FLOAT or R16_FLOAT)
- [ ] Create WGSL struct definitions
- [ ] Document binding group layout

**Output**: `docs/gpu_driven/culling_buffer_spec.md`

**Estimate**: 2 hours

---

## T-CULL-1.8: Write WGSL Culling Shader Skeleton

**Description**: Create skeleton WGSL compute shaders for culling stages.

**Acceptance Criteria**:
- [ ] Frustum cull compute shader (1 thread per instance)
- [ ] Occlusion cull compute shader with HZB texture sampling
- [ ] Combined culling shader (all stages in one dispatch)
- [ ] Proper workgroup size selection (64 or 256)
- [ ] Input/output buffer bindings
- [ ] Atomic counters for visible count

**Output**: `engine/rendering/shaders/culling/`

**Estimate**: 4 hours

---

## Phase 1 Totals

| Category | Count |
|----------|-------|
| Tasks | 8 |
| Estimated Hours | 18.5 |
| Test Coverage | 6 validation tasks |
| Documentation | 1 spec task |
| Implementation | 1 shader task |

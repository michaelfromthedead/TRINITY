# PHASE 2 TODO: Meshlet/Cluster System

## Summary

Validate and prepare the meshlet/cluster system (meshlet.py, 731 lines) for GPU mesh shader port.

---

## T-MESH-2.1: Validate Meshlet Size Limits

**Description**: Verify meshlet builder respects 64 vertex / 124 triangle limits.

**Acceptance Criteria**:
- [ ] No meshlet exceeds 64 vertices
- [ ] No meshlet exceeds 124 triangles
- [ ] Edge case: mesh with exactly 64 vertices produces single meshlet
- [ ] Edge case: triangle sharing > 64 vertices splits correctly
- [ ] Verify local index range (0-63 for each meshlet)

**Files**: `engine/rendering/gpu_driven/meshlet.py` lines 100-200

**Estimate**: 1.5 hours

---

## T-MESH-2.2: Validate Greedy Clustering Quality

**Description**: Verify greedy clustering maximizes vertex reuse within meshlets.

**Acceptance Criteria**:
- [ ] Test on grid mesh (regular topology)
- [ ] Test on sphere mesh (curved topology)
- [ ] Test on irregular mesh (organic shape)
- [ ] Measure vertex reuse ratio (shared vertices / total vertices)
- [ ] Verify adjacency-aware triangle selection
- [ ] Compare with random partitioning baseline

**Files**: `engine/rendering/gpu_driven/meshlet.py` lines 280-380

**Estimate**: 3 hours

---

## T-MESH-2.3: Validate Ritter Bounding Sphere

**Description**: Verify Ritter refinement produces tight bounding spheres.

**Acceptance Criteria**:
- [ ] Test on axis-aligned box (known optimal sphere)
- [ ] Test on sphere mesh (radius should match)
- [ ] Test on elongated mesh (verify refinement tightens)
- [ ] Verify no vertex lies outside sphere
- [ ] Measure sphere tightness (radius / optimal radius)
- [ ] Test convergence (max refinement passes)

**Files**: `engine/rendering/gpu_driven/meshlet.py` lines 380-415

**Estimate**: 2 hours

---

## T-MESH-2.4: Validate Normal Cone Computation

**Description**: Verify normal cone backface culling is geometrically correct.

**Acceptance Criteria**:
- [ ] Test on flat plane (cone should have cutoff = 1.0)
- [ ] Test on hemisphere (cone should have cutoff ~= 0.0)
- [ ] Test on sphere (cone should have cutoff = -1.0, never cullable)
- [ ] Verify cone axis is normalized
- [ ] Test cone culling from multiple view directions
- [ ] Verify no false positives (visible meshlet culled)

**Files**: `engine/rendering/gpu_driven/meshlet.py` lines 416-467

**Estimate**: 2.5 hours

---

## T-MESH-2.5: Validate MeshletLODChain

**Description**: Verify hierarchical LOD chain construction.

**Acceptance Criteria**:
- [ ] LOD 0 contains full detail meshlets
- [ ] Each LOD level has fewer meshlets than previous
- [ ] LOD distance thresholds are increasing
- [ ] Test LOD selection at various distances
- [ ] Verify smooth transition (no missing geometry)
- [ ] Test edge case: very small mesh (single meshlet all LODs)

**Files**: `engine/rendering/gpu_driven/meshlet.py` lines 550-650

**Estimate**: 2 hours

---

## T-MESH-2.6: Create GPU Buffer Layout Specification

**Description**: Document buffer layouts for mesh shader port.

**Acceptance Criteria**:
- [ ] Define Meshlet struct layout (16-byte aligned)
- [ ] Define MeshletBounds struct layout (32-byte aligned)
- [ ] Document vertex buffer layout (position, normal, UV)
- [ ] Document local index buffer format (u8 or u16)
- [ ] Create WGSL struct definitions
- [ ] Document task/mesh shader payload format

**Output**: `docs/gpu_driven/meshlet_buffer_spec.md`

**Estimate**: 2 hours

---

## T-MESH-2.7: Write WGSL Meshlet Shader Skeleton

**Description**: Create skeleton WGSL task/mesh shaders for meshlet rendering.

**Acceptance Criteria**:
- [ ] Task shader: per-meshlet culling, output payload
- [ ] Mesh shader: vertex transformation, primitive output
- [ ] Proper workgroup sizes (32 for task, 64 for mesh)
- [ ] Meshlet descriptor buffer binding
- [ ] Vertex buffer binding
- [ ] Local index buffer binding

**Output**: `engine/rendering/shaders/meshlet/`

**Estimate**: 4 hours

---

## T-MESH-2.8: Benchmark Meshlet Build Performance

**Description**: Measure meshlet building performance for optimization.

**Acceptance Criteria**:
- [ ] Benchmark on 10K triangle mesh
- [ ] Benchmark on 100K triangle mesh
- [ ] Benchmark on 1M triangle mesh
- [ ] Measure time per triangle
- [ ] Identify performance bottlenecks
- [ ] Document optimization opportunities

**Output**: Performance report with timings

**Estimate**: 2 hours

---

## Phase 2 Totals

| Category | Count |
|----------|-------|
| Tasks | 8 |
| Estimated Hours | 19 |
| Test Coverage | 5 validation tasks |
| Documentation | 1 spec task |
| Implementation | 1 shader task |
| Performance | 1 benchmark task |

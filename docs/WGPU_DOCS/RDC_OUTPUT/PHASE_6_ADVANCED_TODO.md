# PHASE 6: ADVANCED - Task List

**Phase:** 6 - ADVANCED
**Estimated Duration:** 3-4 weeks
**Task ID Prefix:** T-WGPU-P6

---

## Task Summary

| ID | Task | Est. Hours | Status |
|----|------|------------|--------|
| T-WGPU-P6.1.1 | DrawIndirectArgs struct | 3 | - |
| T-WGPU-P6.1.2 | DrawIndexedIndirectArgs struct | 3 | - |
| T-WGPU-P6.1.3 | DispatchIndirectArgs struct | 2 | - |
| T-WGPU-P6.1.4 | Indirect draw buffer | 4 | - |
| T-WGPU-P6.1.5 | Count buffer | 3 | - |
| T-WGPU-P6.2.1 | ObjectData struct | 3 | - |
| T-WGPU-P6.2.2 | Scene data buffers | 4 | - |
| T-WGPU-P6.2.3 | Visibility flags buffer | 3 | - |
| T-WGPU-P6.3.1 | Frustum plane extraction | 4 | - |
| T-WGPU-P6.3.2 | AABB-frustum test WGSL | 4 | - |
| T-WGPU-P6.3.3 | Frustum cull pipeline | 4 | - |
| T-WGPU-P6.4.1 | HiZ pyramid creation | 4 | - |
| T-WGPU-P6.4.2 | HiZ downsample shader | 4 | - |
| T-WGPU-P6.4.3 | HiZ occlusion test | 6 | - |
| T-WGPU-P6.4.4 | HiZ cull pipeline | 4 | - |
| T-WGPU-P6.5.1 | LOD distance calculation | 3 | - |
| T-WGPU-P6.5.2 | LOD selection shader | 4 | - |
| T-WGPU-P6.5.3 | LOD buffer management | 3 | - |
| T-WGPU-P6.6.1 | Stream compaction shader | 4 | - |
| T-WGPU-P6.6.2 | Indirect buffer generation | 6 | - |
| T-WGPU-P6.6.3 | GPUCullingPipeline struct | 6 | - |
| T-WGPU-P6.7.1 | Multi-draw indirect wrapper | 4 | - |
| T-WGPU-P6.7.2 | Multi-draw indexed indirect | 4 | - |
| T-WGPU-P6.7.3 | Multi-draw indirect count | 4 | - |
| T-WGPU-P6.7.4 | Feature fallback | 3 | - |
| T-WGPU-P6.8.1 | TextureRegistry | 6 | - |
| T-WGPU-P6.8.2 | BufferRegistry | 4 | - |
| T-WGPU-P6.8.3 | IndexAllocator | 4 | - |
| T-WGPU-P6.8.4 | MaterialTable | 6 | - |
| T-WGPU-P6.8.5 | Bindless bind group | 4 | - |
| T-WGPU-P6.9.1 | Meshlet struct | 3 | - |
| T-WGPU-P6.9.2 | Meshlet generator (basic) | 6 | - |
| T-WGPU-P6.9.3 | GeometryPath abstraction | 4 | - |
| T-WGPU-P6.10.1 | Unit tests | 6 | - |
| T-WGPU-P6.10.2 | Integration tests | 6 | - |
| T-WGPU-P6.10.3 | Visual tests | 6 | - |

**Total Estimated Hours:** 146 hours

---

## Detailed Tasks

### T-WGPU-P6.1.1 - DrawIndirectArgs Struct

**Description:** Implement DrawIndirectArgs buffer struct.

**Prerequisites:** Phase 4 complete

**Deliverable:** DrawIndirectArgs in gpu_driven/indirect/

**Acceptance Criteria:**
- [ ] 4 u32 fields (16 bytes total)
- [ ] bytemuck derives
- [ ] SIZE constant
- [ ] Documentation

**Estimate:** 3 hours

---

### T-WGPU-P6.1.2 - DrawIndexedIndirectArgs Struct

**Description:** Implement DrawIndexedIndirectArgs buffer struct.

**Prerequisites:** T-WGPU-P6.1.1

**Deliverable:** DrawIndexedIndirectArgs struct

**Acceptance Criteria:**
- [ ] 5 fields (20 bytes total)
- [ ] base_vertex is i32 (signed!)
- [ ] bytemuck derives
- [ ] SIZE constant

**Estimate:** 3 hours

---

### T-WGPU-P6.1.3 - DispatchIndirectArgs Struct

**Description:** Implement DispatchIndirectArgs for compute indirect.

**Prerequisites:** T-WGPU-P6.1.1

**Deliverable:** DispatchIndirectArgs struct

**Acceptance Criteria:**
- [ ] 3 u32 fields (12 bytes)
- [ ] bytemuck derives
- [ ] Usage with dispatch_workgroups_indirect

**Estimate:** 2 hours

---

### T-WGPU-P6.1.4 - Indirect Draw Buffer

**Description:** Implement indirect draw buffer management.

**Prerequisites:** T-WGPU-P6.1.2

**Deliverable:** IndirectDrawBuffer struct

**Acceptance Criteria:**
- [ ] Buffer creation with capacity
- [ ] Clear method
- [ ] Resize method
- [ ] INDIRECT usage flag

**Estimate:** 4 hours

---

### T-WGPU-P6.1.5 - Count Buffer

**Description:** Implement count buffer for multi_draw_indirect_count.

**Prerequisites:** T-WGPU-P6.1.4

**Deliverable:** Count buffer management

**Acceptance Criteria:**
- [ ] Single u32 buffer
- [ ] Atomic write from compute
- [ ] Read from multi_draw
- [ ] Frame reset

**Estimate:** 3 hours

---

### T-WGPU-P6.2.1 - ObjectData Struct

**Description:** Implement per-object GPU data struct.

**Prerequisites:** T-WGPU-P6.1.1

**Deliverable:** ObjectData struct

**Acceptance Criteria:**
- [ ] Transform matrix (64 bytes)
- [ ] AABB min/max
- [ ] Mesh index
- [ ] Material index
- [ ] LOD distances
- [ ] bytemuck derives

**Estimate:** 3 hours

---

### T-WGPU-P6.2.2 - Scene Data Buffers

**Description:** Implement scene data buffer system.

**Prerequisites:** T-WGPU-P6.2.1

**Deliverable:** SceneDataBuffers struct

**Acceptance Criteria:**
- [ ] object_buffer: Storage buffer
- [ ] CPU-side Vec for staging
- [ ] Upload method
- [ ] Resize on demand

**Estimate:** 4 hours

---

### T-WGPU-P6.2.3 - Visibility Flags Buffer

**Description:** Implement visibility flags buffer (bitfield).

**Prerequisites:** T-WGPU-P6.2.2

**Deliverable:** Visibility buffer management

**Acceptance Criteria:**
- [ ] 1 bit per object (u32 array)
- [ ] Clear to 0 each frame
- [ ] Atomic OR in compute
- [ ] Read in compaction

**Estimate:** 3 hours

---

### T-WGPU-P6.3.1 - Frustum Plane Extraction

**Description:** Extract frustum planes from view-projection matrix.

**Prerequisites:** Phase 4 complete

**Deliverable:** Frustum plane extraction helper

**Acceptance Criteria:**
- [ ] 6 planes from VP matrix
- [ ] Plane normalization
- [ ] WGSL and Rust implementations
- [ ] Uniform buffer format

**Estimate:** 4 hours

---

### T-WGPU-P6.3.2 - AABB-Frustum Test WGSL

**Description:** Implement AABB-frustum intersection in WGSL.

**Prerequisites:** T-WGPU-P6.3.1

**Deliverable:** test_aabb_frustum() WGSL function

**Acceptance Criteria:**
- [ ] 6 plane tests
- [ ] Early out on first cull
- [ ] Correct for transformed AABB
- [ ] Performance optimized

**Estimate:** 4 hours

---

### T-WGPU-P6.3.3 - Frustum Cull Pipeline

**Description:** Implement frustum culling compute pipeline.

**Prerequisites:** T-WGPU-P6.3.2

**Deliverable:** FrustumCullPipeline struct

**Acceptance Criteria:**
- [ ] Compute shader
- [ ] Object buffer binding
- [ ] Visibility buffer output
- [ ] Workgroup size (64, 1, 1)

**Estimate:** 4 hours

---

### T-WGPU-P6.4.1 - HiZ Pyramid Creation

**Description:** Create HiZ mip chain texture.

**Prerequisites:** Phase 2 textures

**Deliverable:** HiZ pyramid texture

**Acceptance Criteria:**
- [ ] Depth32Float format
- [ ] Full mip chain
- [ ] TEXTURE_BINDING + STORAGE_BINDING
- [ ] Size calculation

**Estimate:** 4 hours

---

### T-WGPU-P6.4.2 - HiZ Downsample Shader

**Description:** Implement HiZ downsample compute shader.

**Prerequisites:** T-WGPU-P6.4.1

**Deliverable:** hiz_downsample.wgsl

**Acceptance Criteria:**
- [ ] 2x2 max reduction (reverse-Z)
- [ ] Per-mip dispatch
- [ ] Correct UV sampling
- [ ] Workgroup size (8, 8, 1)

**Estimate:** 4 hours

---

### T-WGPU-P6.4.3 - HiZ Occlusion Test

**Description:** Implement HiZ occlusion test in WGSL.

**Prerequisites:** T-WGPU-P6.4.2

**Deliverable:** test_occlusion() WGSL function

**Acceptance Criteria:**
- [ ] AABB projection to screen
- [ ] Mip level selection
- [ ] Depth comparison (reverse-Z)
- [ ] Conservative (max depth)

**Estimate:** 6 hours

---

### T-WGPU-P6.4.4 - HiZ Cull Pipeline

**Description:** Implement HiZ culling compute pipeline.

**Prerequisites:** T-WGPU-P6.4.3

**Deliverable:** HiZCullPipeline struct

**Acceptance Criteria:**
- [ ] Frustum + occlusion test
- [ ] HiZ pyramid binding
- [ ] Visibility buffer update
- [ ] Previous frame depth

**Estimate:** 4 hours

---

### T-WGPU-P6.5.1 - LOD Distance Calculation

**Description:** Implement LOD distance calculation.

**Prerequisites:** T-WGPU-P6.2.1

**Deliverable:** LOD distance helpers

**Acceptance Criteria:**
- [ ] Distance to camera
- [ ] Optional screen-size based
- [ ] Per-object LOD distances
- [ ] Default distances

**Estimate:** 3 hours

---

### T-WGPU-P6.5.2 - LOD Selection Shader

**Description:** Implement LOD selection compute shader.

**Prerequisites:** T-WGPU-P6.5.1

**Deliverable:** lod_select.wgsl

**Acceptance Criteria:**
- [ ] Distance comparison
- [ ] 4 LOD levels (0-3)
- [ ] Output to LOD buffer
- [ ] Blend factor (optional)

**Estimate:** 4 hours

---

### T-WGPU-P6.5.3 - LOD Buffer Management

**Description:** Implement LOD selection buffer.

**Prerequisites:** T-WGPU-P6.5.2

**Deliverable:** LOD buffer management

**Acceptance Criteria:**
- [ ] Per-object LOD index
- [ ] Used in indirect generation
- [ ] Frame reset

**Estimate:** 3 hours

---

### T-WGPU-P6.6.1 - Stream Compaction Shader

**Description:** Implement stream compaction for visible objects.

**Prerequisites:** Phase 3 prefix scan

**Deliverable:** compact.wgsl

**Acceptance Criteria:**
- [ ] Uses prefix scan result
- [ ] Scatter visible object indices
- [ ] Maintain order (stable)

**Estimate:** 4 hours

---

### T-WGPU-P6.6.2 - Indirect Buffer Generation

**Description:** Generate indirect draw commands from compacted list.

**Prerequisites:** T-WGPU-P6.6.1

**Deliverable:** build_indirect.wgsl

**Acceptance Criteria:**
- [ ] Lookup mesh data by object index
- [ ] Write DrawIndexedIndirectArgs
- [ ] Atomic count increment
- [ ] LOD-aware mesh selection

**Estimate:** 6 hours

---

### T-WGPU-P6.6.3 - GPUCullingPipeline Struct

**Description:** Integrate all culling stages.

**Prerequisites:** T-WGPU-P6.3.3, T-WGPU-P6.4.4, T-WGPU-P6.5.2, T-WGPU-P6.6.2

**Deliverable:** GPUCullingPipeline struct

**Acceptance Criteria:**
- [ ] 5 compute stages
- [ ] execute() method
- [ ] Barrier placement
- [ ] Debug visualization

**Estimate:** 6 hours

---

### T-WGPU-P6.7.1 - Multi-Draw Indirect Wrapper

**Description:** Implement multi_draw_indirect wrapper.

**Prerequisites:** T-WGPU-P6.1.4

**Deliverable:** multi_draw_indirect() wrapper

**Acceptance Criteria:**
- [ ] Feature check (MULTI_DRAW_INDIRECT)
- [ ] Buffer and offset parameters
- [ ] Count parameter
- [ ] Fallback to loop

**Estimate:** 4 hours

---

### T-WGPU-P6.7.2 - Multi-Draw Indexed Indirect

**Description:** Implement multi_draw_indexed_indirect wrapper.

**Prerequisites:** T-WGPU-P6.7.1

**Deliverable:** multi_draw_indexed_indirect() wrapper

**Acceptance Criteria:**
- [ ] Same as P6.7.1 but indexed
- [ ] Correct stride (20 bytes)

**Estimate:** 4 hours

---

### T-WGPU-P6.7.3 - Multi-Draw Indirect Count

**Description:** Implement multi_draw_indirect_count wrapper.

**Prerequisites:** T-WGPU-P6.7.2, T-WGPU-P6.1.5

**Deliverable:** multi_draw_indirect_count() wrapper

**Acceptance Criteria:**
- [ ] Feature check (MULTI_DRAW_INDIRECT_COUNT)
- [ ] Count buffer parameter
- [ ] Max count parameter
- [ ] Fallback implementation

**Estimate:** 4 hours

---

### T-WGPU-P6.7.4 - Feature Fallback

**Description:** Implement fallback for missing multi-draw features.

**Prerequisites:** T-WGPU-P6.7.3

**Deliverable:** Fallback implementation

**Acceptance Criteria:**
- [ ] CPU readback of count
- [ ] Loop of individual draws
- [ ] Performance warning
- [ ] Tier detection

**Estimate:** 3 hours

---

### T-WGPU-P6.8.1 - TextureRegistry

**Description:** Implement bindless texture registry.

**Prerequisites:** Phase 2 textures

**Deliverable:** TextureRegistry struct

**Acceptance Criteria:**
- [ ] Feature check (TEXTURE_BINDING_ARRAY)
- [ ] Slot allocation (allocate_slot)
- [ ] Slot release (free_slot)
- [ ] Bind group creation with count
- [ ] Rebuild on change

**Estimate:** 6 hours

---

### T-WGPU-P6.8.2 - BufferRegistry

**Description:** Implement bindless buffer registry.

**Prerequisites:** Phase 2 buffers

**Deliverable:** BufferRegistry struct

**Acceptance Criteria:**
- [ ] Storage buffer array binding
- [ ] Slot allocation
- [ ] Dirty range tracking
- [ ] Update method

**Estimate:** 4 hours

---

### T-WGPU-P6.8.3 - IndexAllocator

**Description:** Implement generic index allocator with recycling.

**Prerequisites:** None (utility)

**Deliverable:** IndexAllocator struct

**Acceptance Criteria:**
- [ ] allocate() -> u32
- [ ] free(index)
- [ ] Free list recycling
- [ ] Capacity growth

**Estimate:** 4 hours

---

### T-WGPU-P6.8.4 - MaterialTable

**Description:** Implement bindless material table.

**Prerequisites:** T-WGPU-P6.8.1, T-WGPU-P6.8.2

**Deliverable:** MaterialTable struct

**Acceptance Criteria:**
- [ ] MaterialDescriptor struct
- [ ] Texture index references
- [ ] Material index lookup
- [ ] GPU buffer upload

**Estimate:** 6 hours

---

### T-WGPU-P6.8.5 - Bindless Bind Group

**Description:** Create bindless bind group layout and group.

**Prerequisites:** T-WGPU-P6.8.4

**Deliverable:** Bindless bind group creation

**Acceptance Criteria:**
- [ ] Layout with texture array
- [ ] Layout with material buffer
- [ ] Group creation
- [ ] Non-uniform indexing

**Estimate:** 4 hours

---

### T-WGPU-P6.9.1 - Meshlet Struct

**Description:** Define meshlet data structure.

**Prerequisites:** None

**Deliverable:** Meshlet struct

**Acceptance Criteria:**
- [ ] Vertex/triangle offsets
- [ ] Vertex/triangle counts
- [ ] Bounding sphere/cone
- [ ] Backface culling cone
- [ ] bytemuck derives

**Estimate:** 3 hours

---

### T-WGPU-P6.9.2 - Meshlet Generator (Basic)

**Description:** Implement basic meshlet generation.

**Prerequisites:** T-WGPU-P6.9.1

**Deliverable:** MeshletGenerator struct

**Acceptance Criteria:**
- [ ] Split mesh into meshlets (64 verts, 124 tris)
- [ ] Local index generation
- [ ] Bounding volume calculation
- [ ] Backface cone calculation

**Estimate:** 6 hours

---

### T-WGPU-P6.9.3 - GeometryPath Abstraction

**Description:** Implement geometry path abstraction.

**Prerequisites:** T-WGPU-P6.9.2

**Deliverable:** GeometryPath enum, trait

**Acceptance Criteria:**
- [ ] Traditional path
- [ ] Meshlet path (stubbed)
- [ ] Path selection based on capability
- [ ] Unified render interface

**Estimate:** 4 hours

---

### T-WGPU-P6.10.1 - Unit Tests

**Description:** Write unit tests for Phase 6.

**Prerequisites:** All T-WGPU-P6.1-9 tasks

**Deliverable:** Unit tests

**Acceptance Criteria:**
- [ ] AABB-frustum test
- [ ] Index allocator tests
- [ ] Indirect struct sizes
- [ ] LOD selection tests
- [ ] 80%+ coverage

**Estimate:** 6 hours

---

### T-WGPU-P6.10.2 - Integration Tests

**Description:** Write integration tests for GPU-driven rendering.

**Prerequisites:** T-WGPU-P6.10.1

**Deliverable:** Integration tests

**Acceptance Criteria:**
- [ ] Full culling pipeline test
- [ ] Bindless material test
- [ ] Multi-draw test
- [ ] Performance regression test

**Estimate:** 6 hours

---

### T-WGPU-P6.10.3 - Visual Tests

**Description:** Create visual verification tests.

**Prerequisites:** T-WGPU-P6.10.2

**Deliverable:** Visual tests

**Acceptance Criteria:**
- [ ] Culling debug visualization
- [ ] LOD transition smoothness
- [ ] Massive scene (100K objects)
- [ ] Reference images

**Estimate:** 6 hours

---

## Task Dependencies

```
Phase 4 Complete
    |
    +---> T-WGPU-P6.1.1 (DrawIndirect)
              |
              +---> T-WGPU-P6.1.2 through P6.1.5
    |
    +---> T-WGPU-P6.2.1 (ObjectData)
              |
              +---> T-WGPU-P6.2.2, P6.2.3
    |
    +---> T-WGPU-P6.3.1 (Frustum planes)
              |
              +---> T-WGPU-P6.3.2 (AABB test)
                        |
                        +---> T-WGPU-P6.3.3 (Frustum pipeline)
    |
    +---> T-WGPU-P6.4.1 (HiZ texture)
              |
              +---> T-WGPU-P6.4.2 (Downsample)
                        |
                        +---> T-WGPU-P6.4.3 (Occlusion test)
                                  |
                                  +---> T-WGPU-P6.4.4 (HiZ pipeline)
    |
    +---> T-WGPU-P6.5.1 through P6.5.3 (LOD)
    |
    +---> T-WGPU-P6.6.1 (Compaction) --> P6.6.2 (Build indirect)
              |
              +---> T-WGPU-P6.6.3 (Full pipeline, needs P6.3.3, P6.4.4, P6.5.2)
    |
    +---> T-WGPU-P6.7.1 through P6.7.4 (Multi-draw)
    |
    +---> T-WGPU-P6.8.1 through P6.8.5 (Bindless)
    |
    +---> T-WGPU-P6.9.1 through P6.9.3 (Meshlet prep)

All --> T-WGPU-P6.10.1 --> T-WGPU-P6.10.2 --> T-WGPU-P6.10.3
```

---

*End of PHASE_6_ADVANCED_TODO.md*

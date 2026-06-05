# PHASE 3: PIPELINES - Task List

**Phase:** 3 - PIPELINES
**Estimated Duration:** 3-4 weeks
**Task ID Prefix:** T-WGPU-P3

---

## Task Summary

| ID | Task | Est. Hours | Status |
|----|------|------------|--------|
| T-WGPU-P3.1.1 | Render pipeline descriptor | 6 | - |
| T-WGPU-P3.1.2 | Vertex state configuration | 4 | - |
| T-WGPU-P3.1.3 | Primitive state | 3 | - |
| T-WGPU-P3.1.4 | Depth/stencil state | 4 | - |
| T-WGPU-P3.1.5 | Multisample state | 3 | - |
| T-WGPU-P3.1.6 | Fragment state and MRT | 4 | - |
| T-WGPU-P3.1.7 | Pipeline cache | 6 | - |
| T-WGPU-P3.1.8 | Cache warming | 4 | - |
| T-WGPU-P3.2.1 | Vertex format registry | 4 | - |
| T-WGPU-P3.2.2 | Vertex attribute formats | 3 | - |
| T-WGPU-P3.2.3 | Instance step mode | 3 | - |
| T-WGPU-P3.3.1 | Primitive topologies | 3 | - |
| T-WGPU-P3.3.2 | Culling and front face | 2 | - |
| T-WGPU-P3.3.3 | Polygon modes | 2 | - |
| T-WGPU-P3.4.1 | Viewport and scissor | 3 | - |
| T-WGPU-P3.4.2 | Depth bias | 2 | - |
| T-WGPU-P3.4.3 | Conservative rasterization | 2 | - |
| T-WGPU-P3.5.1 | Color target state | 3 | - |
| T-WGPU-P3.5.2 | Blend modes | 4 | - |
| T-WGPU-P3.5.3 | Write masks | 2 | - |
| T-WGPU-P3.6.1 | Depth test config | 3 | - |
| T-WGPU-P3.6.2 | Stencil state | 4 | - |
| T-WGPU-P3.7.1 | MSAA configuration | 3 | - |
| T-WGPU-P3.7.2 | MSAA resolve | 3 | - |
| T-WGPU-P3.8.1 | Render pass creation | 4 | - |
| T-WGPU-P3.8.2 | Load/store operations | 3 | - |
| T-WGPU-P3.8.3 | Render pass commands | 4 | - |
| T-WGPU-P3.8.4 | Draw commands (7 variants) | 6 | - |
| T-WGPU-P3.8.5 | Render bundles | 6 | - |
| T-WGPU-P3.9.1 | Compute pipeline descriptor | 4 | - |
| T-WGPU-P3.9.2 | Compute pipeline cache | 4 | - |
| T-WGPU-P3.9.3 | Compute pass creation | 3 | - |
| T-WGPU-P3.9.4 | Dispatch commands | 4 | - |
| T-WGPU-P3.10.1 | Reduction patterns | 6 | - |
| T-WGPU-P3.10.2 | Prefix scan | 6 | - |
| T-WGPU-P3.10.3 | Stream compaction | 4 | - |
| T-WGPU-P3.10.4 | Radix sort | 8 | - |
| T-WGPU-P3.10.5 | Image processing | 6 | - |
| T-WGPU-P3.10.6 | ComputeLibrary integration | 4 | - |
| T-WGPU-P3.11.1 | Unit tests | 8 | - |
| T-WGPU-P3.11.2 | Integration tests | 8 | - |

**Total Estimated Hours:** 160 hours

---

## Detailed Tasks

### T-WGPU-P3.1.1 - Render Pipeline Descriptor

**Description:** Implement full RenderPipelineDescriptor construction.

**Prerequisites:** Phase 2 complete

**Deliverable:** `create_render_pipeline()` in pipeline/render_pipeline.rs

**Acceptance Criteria:**
- [ ] All 9 descriptor fields configurable
- [ ] Layout association required
- [ ] Label for debugging
- [ ] Returns TrinityRenderPipeline wrapper

**Estimate:** 6 hours

---

### T-WGPU-P3.1.2 - Vertex State Configuration

**Description:** Implement VertexState with buffer layouts.

**Prerequisites:** T-WGPU-P3.1.1

**Deliverable:** VertexState helpers

**Acceptance Criteria:**
- [ ] Shader module and entry point
- [ ] Compilation options support
- [ ] Multiple buffer layouts (vertex + instance)
- [ ] Integration with VertexFormatRegistry

**Estimate:** 4 hours

---

### T-WGPU-P3.1.3 - Primitive State

**Description:** Implement PrimitiveState configuration.

**Prerequisites:** T-WGPU-P3.1.1

**Deliverable:** PrimitiveState helpers

**Acceptance Criteria:**
- [ ] All 5 topologies
- [ ] Strip index format
- [ ] Front face (CCW default)
- [ ] Cull mode (Back default)
- [ ] Unclipped depth
- [ ] Polygon mode
- [ ] Conservative flag

**Estimate:** 3 hours

---

### T-WGPU-P3.1.4 - Depth/Stencil State

**Description:** Implement DepthStencilState configuration.

**Prerequisites:** T-WGPU-P3.1.1

**Deliverable:** DepthStencilState helpers

**Acceptance Criteria:**
- [ ] Format selection
- [ ] Depth write enable
- [ ] 8 compare functions
- [ ] Stencil front/back faces
- [ ] Stencil read/write masks
- [ ] Depth bias state

**Estimate:** 4 hours

---

### T-WGPU-P3.1.5 - Multisample State

**Description:** Implement MultisampleState configuration.

**Prerequisites:** T-WGPU-P3.1.1

**Deliverable:** MultisampleState helpers

**Acceptance Criteria:**
- [ ] Sample count query from adapter
- [ ] Sample mask configuration
- [ ] Alpha to coverage option

**Estimate:** 3 hours

---

### T-WGPU-P3.1.6 - Fragment State and MRT

**Description:** Implement FragmentState with multiple render targets.

**Prerequisites:** T-WGPU-P3.1.1

**Deliverable:** FragmentState helpers

**Acceptance Criteria:**
- [ ] Shader module and entry point
- [ ] Multiple color targets (MRT)
- [ ] Per-target format
- [ ] Per-target blend state
- [ ] Per-target write mask

**Estimate:** 4 hours

---

### T-WGPU-P3.1.7 - Pipeline Cache

**Description:** Implement PipelineCache with hash-based lookup.

**Prerequisites:** T-WGPU-P3.1.1 through P3.1.6

**Deliverable:** PipelineCache struct

**Acceptance Criteria:**
- [ ] PipelineKey struct with all state
- [ ] Hash implementation for PipelineKey
- [ ] pipelines: HashMap<PipelineKey, Arc<RenderPipeline>>
- [ ] get_or_create() API
- [ ] Cache hit/miss metrics
- [ ] invalidate(shader_id) for hot-reload

**Estimate:** 6 hours

---

### T-WGPU-P3.1.8 - Cache Warming

**Description:** Implement cache pre-warming for common pipelines.

**Prerequisites:** T-WGPU-P3.1.7

**Deliverable:** warm_cache() method

**Acceptance Criteria:**
- [ ] Accept array of PipelineKey
- [ ] Background compilation (optional)
- [ ] Progress callback
- [ ] Common TRINITY pipelines defined

**Estimate:** 4 hours

---

### T-WGPU-P3.2.1 - Vertex Format Registry

**Description:** Implement VertexFormatRegistry with standard formats.

**Prerequisites:** T-WGPU-P3.1.2

**Deliverable:** VertexFormatRegistry struct

**Acceptance Criteria:**
- [ ] STATIC_MESH (48 bytes)
- [ ] SKINNED_MESH (72 bytes)
- [ ] TERRAIN (32 bytes)
- [ ] PARTICLE (32 bytes)
- [ ] UI (20 bytes)
- [ ] Register by ID
- [ ] Lookup by ID

**Estimate:** 4 hours

---

### T-WGPU-P3.2.2 - Vertex Attribute Formats

**Description:** Document and implement all 32 vertex formats.

**Prerequisites:** T-WGPU-P3.2.1

**Deliverable:** Format helpers and documentation

**Acceptance Criteria:**
- [ ] All VertexFormat variants documented
- [ ] Size calculation helpers
- [ ] Common format combinations
- [ ] vertex_attr_array! macro usage

**Estimate:** 3 hours

---

### T-WGPU-P3.2.3 - Instance Step Mode

**Description:** Implement instance data vertex layouts.

**Prerequisites:** T-WGPU-P3.2.1

**Deliverable:** Instance layout helpers

**Acceptance Criteria:**
- [ ] VertexStepMode::Instance
- [ ] Transform matrix layout (4x vec4)
- [ ] Custom instance data support
- [ ] Per-instance color, etc.

**Estimate:** 3 hours

---

### T-WGPU-P3.3.1 - Primitive Topologies

**Description:** Implement all primitive topologies.

**Prerequisites:** T-WGPU-P3.1.3

**Deliverable:** Topology selection helpers

**Acceptance Criteria:**
- [ ] PointList
- [ ] LineList
- [ ] LineStrip
- [ ] TriangleList
- [ ] TriangleStrip
- [ ] Usage documentation

**Estimate:** 3 hours

---

### T-WGPU-P3.3.2 - Culling and Front Face

**Description:** Implement face culling configuration.

**Prerequisites:** T-WGPU-P3.1.3

**Deliverable:** Culling helpers

**Acceptance Criteria:**
- [ ] FrontFace::Ccw (default), Cw
- [ ] CullMode::None, Front, Back
- [ ] Documentation on winding order

**Estimate:** 2 hours

---

### T-WGPU-P3.3.3 - Polygon Modes

**Description:** Implement polygon fill modes.

**Prerequisites:** T-WGPU-P3.1.3

**Deliverable:** Polygon mode helpers

**Acceptance Criteria:**
- [ ] Fill (default)
- [ ] Line (wireframe)
- [ ] Point
- [ ] Feature flag check (NON_FILL_POLYGON_MODE)

**Estimate:** 2 hours

---

### T-WGPU-P3.4.1 - Viewport and Scissor

**Description:** Implement viewport and scissor configuration.

**Prerequisites:** T-WGPU-P3.8.1

**Deliverable:** Viewport/scissor helpers

**Acceptance Criteria:**
- [ ] set_viewport(x, y, width, height, min_depth, max_depth)
- [ ] set_scissor_rect(x, y, width, height)
- [ ] Viewport struct
- [ ] Default to full render target

**Estimate:** 3 hours

---

### T-WGPU-P3.4.2 - Depth Bias

**Description:** Implement depth bias configuration.

**Prerequisites:** T-WGPU-P3.1.4

**Deliverable:** DepthBiasState helpers

**Acceptance Criteria:**
- [ ] constant: i32
- [ ] slope_scale: f32
- [ ] clamp: f32
- [ ] Shadow map preset

**Estimate:** 2 hours

---

### T-WGPU-P3.4.3 - Conservative Rasterization

**Description:** Implement conservative rasterization support.

**Prerequisites:** T-WGPU-P3.1.3

**Deliverable:** Conservative rasterization helpers

**Acceptance Criteria:**
- [ ] Feature check (CONSERVATIVE_RASTERIZATION)
- [ ] PrimitiveState.conservative flag
- [ ] Use case documentation

**Estimate:** 2 hours

---

### T-WGPU-P3.5.1 - Color Target State

**Description:** Implement ColorTargetState configuration.

**Prerequisites:** T-WGPU-P3.1.6

**Deliverable:** ColorTargetState helpers

**Acceptance Criteria:**
- [ ] Format selection
- [ ] Blend state (optional)
- [ ] Write mask
- [ ] Per-target configuration

**Estimate:** 3 hours

---

### T-WGPU-P3.5.2 - Blend Modes

**Description:** Implement blend mode presets.

**Prerequisites:** T-WGPU-P3.5.1

**Deliverable:** BlendMode enum and presets

**Acceptance Criteria:**
- [ ] Alpha blending (src*srcA + dst*(1-srcA))
- [ ] Premultiplied alpha
- [ ] Additive (src + dst)
- [ ] Multiply (src * dst)
- [ ] BlendFactor enum (13 values)
- [ ] BlendOperation enum (5 values)
- [ ] Color and alpha separate

**Estimate:** 4 hours

---

### T-WGPU-P3.5.3 - Write Masks

**Description:** Implement color write mask configuration.

**Prerequisites:** T-WGPU-P3.5.1

**Deliverable:** ColorWrites helpers

**Acceptance Criteria:**
- [ ] RED, GREEN, BLUE, ALPHA flags
- [ ] ALL preset
- [ ] Common combinations (RGB, NONE)

**Estimate:** 2 hours

---

### T-WGPU-P3.6.1 - Depth Test Config

**Description:** Implement depth test configuration.

**Prerequisites:** T-WGPU-P3.1.4

**Deliverable:** Depth test helpers

**Acceptance Criteria:**
- [ ] depth_write_enabled
- [ ] 8 compare functions
- [ ] Common presets (less, less-equal, always)
- [ ] Depth format selection

**Estimate:** 3 hours

---

### T-WGPU-P3.6.2 - Stencil State

**Description:** Implement stencil test configuration.

**Prerequisites:** T-WGPU-P3.1.4

**Deliverable:** StencilState helpers

**Acceptance Criteria:**
- [ ] Front and back face states
- [ ] Compare function per face
- [ ] 8 stencil operations (fail, depth_fail, pass)
- [ ] Read/write masks
- [ ] Reference value (dynamic)

**Estimate:** 4 hours

---

### T-WGPU-P3.7.1 - MSAA Configuration

**Description:** Implement MSAA sample count selection.

**Prerequisites:** T-WGPU-P3.1.5

**Deliverable:** MSAA helpers

**Acceptance Criteria:**
- [ ] Query supported sample counts
- [ ] Select max supported (1, 4, 8, 16)
- [ ] MultisampleState configuration
- [ ] MSAA render target creation

**Estimate:** 3 hours

---

### T-WGPU-P3.7.2 - MSAA Resolve

**Description:** Implement MSAA resolve attachment.

**Prerequisites:** T-WGPU-P3.7.1

**Deliverable:** Resolve attachment helpers

**Acceptance Criteria:**
- [ ] resolve_target in RenderPassColorAttachment
- [ ] Resolve to non-MSAA texture
- [ ] Store operation (typically Discard on MSAA)

**Estimate:** 3 hours

---

### T-WGPU-P3.8.1 - Render Pass Creation

**Description:** Implement render pass creation.

**Prerequisites:** Phase 2 complete

**Deliverable:** `begin_render_pass()` wrapper

**Acceptance Criteria:**
- [ ] RenderPassDescriptor construction
- [ ] Color attachments array
- [ ] Depth/stencil attachment (optional)
- [ ] Timestamp writes (optional)
- [ ] Occlusion query set (optional)

**Estimate:** 4 hours

---

### T-WGPU-P3.8.2 - Load/Store Operations

**Description:** Implement attachment load/store operations.

**Prerequisites:** T-WGPU-P3.8.1

**Deliverable:** LoadOp/StoreOp helpers

**Acceptance Criteria:**
- [ ] LoadOp::Clear(value)
- [ ] LoadOp::Load
- [ ] StoreOp::Store
- [ ] StoreOp::Discard
- [ ] Common combinations documented

**Estimate:** 3 hours

---

### T-WGPU-P3.8.3 - Render Pass Commands

**Description:** Implement render pass state commands.

**Prerequisites:** T-WGPU-P3.8.1

**Deliverable:** State command wrappers

**Acceptance Criteria:**
- [ ] set_pipeline()
- [ ] set_bind_group() with dynamic offsets
- [ ] set_vertex_buffer()
- [ ] set_index_buffer()
- [ ] set_viewport()
- [ ] set_scissor_rect()
- [ ] set_blend_constant()
- [ ] set_stencil_reference()
- [ ] set_push_constants()

**Estimate:** 4 hours

---

### T-WGPU-P3.8.4 - Draw Commands (7 variants)

**Description:** Implement all draw command variants.

**Prerequisites:** T-WGPU-P3.8.3

**Deliverable:** Draw command wrappers

**Acceptance Criteria:**
- [ ] draw(vertex_count, instance_count, first_vertex, first_instance)
- [ ] draw_indexed(index_count, instance_count, first_index, base_vertex, first_instance)
- [ ] draw_indirect(buffer, offset)
- [ ] draw_indexed_indirect(buffer, offset)
- [ ] multi_draw_indirect(buffer, offset, count)
- [ ] multi_draw_indexed_indirect(buffer, offset, count)
- [ ] multi_draw_indirect_count(indirect, indirect_offset, count_buffer, count_offset, max_count)
- [ ] Feature checks for multi-draw variants

**Estimate:** 6 hours

---

### T-WGPU-P3.8.5 - Render Bundles

**Description:** Implement render bundle recording and cache.

**Prerequisites:** T-WGPU-P3.8.3

**Deliverable:** RenderBundleCache struct

**Acceptance Criteria:**
- [ ] RenderBundleEncoderDescriptor
- [ ] Bundle recording API
- [ ] Bundle finish()
- [ ] execute_bundles() in render pass
- [ ] Cache by BundleKey
- [ ] Invalidation API

**Estimate:** 6 hours

---

### T-WGPU-P3.9.1 - Compute Pipeline Descriptor

**Description:** Implement ComputePipelineDescriptor construction.

**Prerequisites:** Phase 2 complete

**Deliverable:** `create_compute_pipeline()` in pipeline/compute_pipeline.rs

**Acceptance Criteria:**
- [ ] Module and entry point
- [ ] Layout association
- [ ] Compilation options (constants)
- [ ] Label for debugging

**Estimate:** 4 hours

---

### T-WGPU-P3.9.2 - Compute Pipeline Cache

**Description:** Implement compute pipeline caching.

**Prerequisites:** T-WGPU-P3.9.1

**Deliverable:** ComputePipelineCache struct

**Acceptance Criteria:**
- [ ] ComputePipelineKey
- [ ] Hash by shader + entry + specialization
- [ ] get_or_create()
- [ ] invalidate(shader_id)

**Estimate:** 4 hours

---

### T-WGPU-P3.9.3 - Compute Pass Creation

**Description:** Implement compute pass creation.

**Prerequisites:** T-WGPU-P3.9.1

**Deliverable:** `begin_compute_pass()` wrapper

**Acceptance Criteria:**
- [ ] ComputePassDescriptor
- [ ] Timestamp writes (optional)
- [ ] set_pipeline()
- [ ] set_bind_group()
- [ ] set_push_constants()

**Estimate:** 3 hours

---

### T-WGPU-P3.9.4 - Dispatch Commands

**Description:** Implement dispatch commands.

**Prerequisites:** T-WGPU-P3.9.3

**Deliverable:** Dispatch wrappers

**Acceptance Criteria:**
- [ ] dispatch_workgroups(x, y, z)
- [ ] dispatch_workgroups_indirect(buffer, offset)
- [ ] Workgroup count calculation helper
- [ ] Limit validation

**Estimate:** 4 hours

---

### T-WGPU-P3.10.1 - Reduction Patterns

**Description:** Implement parallel reduction compute shaders.

**Prerequisites:** T-WGPU-P3.9.4

**Deliverable:** Reduction shaders and pipelines

**Acceptance Criteria:**
- [ ] reduce_sum.wgsl
- [ ] reduce_min.wgsl
- [ ] reduce_max.wgsl
- [ ] Tree reduction pattern
- [ ] Workgroup memory usage
- [ ] Multi-pass for large arrays

**Estimate:** 6 hours

---

### T-WGPU-P3.10.2 - Prefix Scan

**Description:** Implement Blelloch prefix scan.

**Prerequisites:** T-WGPU-P3.9.4

**Deliverable:** Prefix scan shader and pipeline

**Acceptance Criteria:**
- [ ] prefix_scan.wgsl
- [ ] Up-sweep phase
- [ ] Down-sweep phase
- [ ] Block-level sums
- [ ] Multi-block scan

**Estimate:** 6 hours

---

### T-WGPU-P3.10.3 - Stream Compaction

**Description:** Implement stream compaction via scan.

**Prerequisites:** T-WGPU-P3.10.2

**Deliverable:** Stream compaction shader and pipeline

**Acceptance Criteria:**
- [ ] stream_compact.wgsl
- [ ] Uses prefix scan result
- [ ] Scatter to output indices
- [ ] Count output

**Estimate:** 4 hours

---

### T-WGPU-P3.10.4 - Radix Sort

**Description:** Implement GPU radix sort.

**Prerequisites:** T-WGPU-P3.10.2

**Deliverable:** Radix sort shader and pipeline

**Acceptance Criteria:**
- [ ] radix_sort.wgsl
- [ ] 4-bit digit per pass
- [ ] 8 passes for 32-bit keys
- [ ] Key-value pair sorting
- [ ] Histogram + scatter

**Estimate:** 8 hours

---

### T-WGPU-P3.10.5 - Image Processing

**Description:** Implement image processing compute shaders.

**Prerequisites:** T-WGPU-P3.9.4

**Deliverable:** Image processing shaders

**Acceptance Criteria:**
- [ ] blur_horizontal.wgsl
- [ ] blur_vertical.wgsl
- [ ] downsample.wgsl
- [ ] histogram.wgsl
- [ ] tonemapping.wgsl
- [ ] Shared memory tile optimization

**Estimate:** 6 hours

---

### T-WGPU-P3.10.6 - ComputeLibrary Integration

**Description:** Integrate all compute patterns into ComputeLibrary.

**Prerequisites:** T-WGPU-P3.10.1 through P3.10.5

**Deliverable:** ComputeLibrary struct

**Acceptance Criteria:**
- [ ] All 17 pipelines created
- [ ] Initialization at startup
- [ ] Easy dispatch helpers
- [ ] DispatchHelper utility

**Estimate:** 4 hours

---

### T-WGPU-P3.11.1 - Unit Tests

**Description:** Write unit tests for Phase 3 components.

**Prerequisites:** All T-WGPU-P3.1-10 tasks

**Deliverable:** Tests in pipeline/tests/

**Acceptance Criteria:**
- [ ] Pipeline key hash tests
- [ ] Vertex format tests
- [ ] Blend mode tests
- [ ] Depth/stencil tests
- [ ] Workgroup size tests
- [ ] 80%+ coverage

**Estimate:** 8 hours

---

### T-WGPU-P3.11.2 - Integration Tests

**Description:** Write integration tests for pipeline operations.

**Prerequisites:** T-WGPU-P3.11.1

**Deliverable:** Integration tests

**Acceptance Criteria:**
- [ ] Render pipeline creation test
- [ ] Compute pipeline creation test
- [ ] Render pass execution test
- [ ] Render bundle test
- [ ] Reduction correctness test
- [ ] Prefix scan correctness test

**Estimate:** 8 hours

---

## Task Dependencies

```
Phase 2 Complete
    |
    +---> T-WGPU-P3.1.1 (Render pipeline)
              |
              +---> T-WGPU-P3.1.2 through P3.1.6
              +---> T-WGPU-P3.1.7 (Cache) --> T-WGPU-P3.1.8 (Warm)
    |
    +---> T-WGPU-P3.2.1 (Vertex registry) --> P3.2.2, P3.2.3
    |
    +---> T-WGPU-P3.3.1 through P3.3.3 (Primitive)
    |
    +---> T-WGPU-P3.4.1 through P3.4.3 (Rasterization)
    |
    +---> T-WGPU-P3.5.1 through P3.5.3 (Fragment)
    |
    +---> T-WGPU-P3.6.1, P3.6.2 (Depth/stencil)
    |
    +---> T-WGPU-P3.7.1 --> P3.7.2 (MSAA)
    |
    +---> T-WGPU-P3.8.1 (Render pass)
              |
              +---> P3.8.2 through P3.8.5
    |
    +---> T-WGPU-P3.9.1 (Compute pipeline)
              |
              +---> P3.9.2 through P3.9.4
              +---> P3.10.1 (Reduction)
              +---> P3.10.2 (Scan) --> P3.10.3, P3.10.4
              +---> P3.10.5 (Image)
              +---> P3.10.6 (Library)

All --> T-WGPU-P3.11.1 --> T-WGPU-P3.11.2
```

---

*End of PHASE_3_PIPELINES_TODO.md*

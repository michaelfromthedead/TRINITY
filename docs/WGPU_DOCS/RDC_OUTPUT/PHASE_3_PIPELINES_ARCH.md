# PHASE 3: PIPELINES - Architecture

**Scope:** Render pipelines, compute pipelines, pipeline caching
**Duration:** 3-4 weeks
**Dependencies:** Phase 2 (RESOURCES)
**Produces:** Complete pipeline creation and management

---

## Overview

Phase 3 implements the graphics and compute pipeline creation, including all fixed-function configuration (vertex input, primitive assembly, rasterization, depth/stencil, blending) and compute dispatch patterns.

### Covered Content (from MASTER.md Parts IV-V)

- Chapter 6: Graphics Pipeline
  - 6.1 Pipeline creation (descriptor, layout, vertex/primitive/depth/multisample/fragment state)
  - 6.2 Vertex input (buffer layouts, attribute formats, step modes)
  - 6.3 Primitive assembly (topologies, index formats, culling)
  - 6.4 Rasterization (viewport, scissor, depth bias, conservative)
  - 6.5 Fragment processing (color targets, write masks, blending)
  - 6.6 Depth/stencil (depth test, compare functions, stencil operations)
  - 6.7 Multisampling (sample count, MSAA resolve)

- Chapter 7: Render Passes
  - 7.1-7.2 Fundamentals and attachment operations
  - 7.3 Render pass commands
  - 7.4 Draw commands (all 7 variants)
  - 7.5 Render bundles

- Chapter 8: Compute Fundamentals
  - 8.1 Compute pipeline (descriptor, layout, entry point, caching)
  - 8.2 Compute shaders (@compute, @workgroup_size, built-ins, barriers)
  - 8.3 Compute pass (pipeline binding, dispatch)
  - 8.4 Dispatch commands (direct, indirect, limits)
  - 8.5 Compute patterns (reduction, scan, compaction, sort, etc.)

---

## Architectural Decisions

### ADR-009: Pipeline Caching Strategy

**Context:** Pipeline creation is expensive (shader compilation, state validation).

**Decision:** Implement multi-level pipeline cache:
1. In-memory HashMap by PipelineKey hash
2. warm_cache() for common pipelines at startup
3. Optional disk cache via wgpu's pipeline cache feature

**Rationale:** Eliminates redundant compilation; reduces hitching.

**Consequences:**
- PipelineKey must include all state affecting PSO
- Cache invalidation on shader hot-reload
- Memory usage for cached pipelines

---

### ADR-010: Vertex Format Registry

**Context:** Vertex layouts are used repeatedly across meshes.

**Decision:** Pre-register standard formats with unique IDs:
- STATIC_MESH, SKINNED_MESH, TERRAIN, PARTICLE, UI

**Rationale:** Avoids repeated VertexBufferLayout construction.

**Consequences:**
- Registry lookup by ID
- Custom formats still supported
- Layout descriptor created once

---

### ADR-011: Render Bundle Usage

**Context:** Static geometry draw commands can be pre-recorded.

**Decision:** Implement RenderBundleCache for static geometry:
- Key: mesh ID + material ID + pipeline compatibility
- Record once, replay many frames

**Rationale:** Reduces CPU overhead for static scenes.

**Consequences:**
- Not suitable for dynamic objects
- Bundle invalidation on pipeline change
- Memory for recorded commands

---

### ADR-012: Compute Library Pattern

**Context:** Common compute operations (reduction, scan, sort) are needed repeatedly.

**Decision:** Implement ComputeLibrary with pre-built pipelines:
- Reduction (sum, min, max)
- Prefix scan
- Stream compaction
- Radix sort
- Image processing
- GPU culling

**Rationale:** Reusable, tested implementations.

**Consequences:**
- Standardized workgroup sizes
- Shader permutations for data types
- Initialization cost at startup

---

## Component Breakdown

### 1. Render Pipeline

```
TrinityRenderPipeline
├── pipeline: wgpu::RenderPipeline
├── layout: Arc<wgpu::PipelineLayout>
├── vertex_format: VertexFormatId
├── blend_mode: BlendMode
├── depth_config: DepthConfig
└── key: PipelineKey
```

**PipelineKey:**
- vertex_shader: ShaderId
- fragment_shader: ShaderId
- vertex_layout: VertexLayoutId
- render_target_formats: Vec<TextureFormat>
- depth_format: Option<TextureFormat>
- sample_count: u32
- blend_mode: BlendMode
- cull_mode: CullMode
- depth_write: bool
- depth_compare: CompareFunction

**PipelineCache:**
- pipelines: HashMap<PipelineKey, Arc<RenderPipeline>>
- layouts: HashMap<LayoutKey, Arc<PipelineLayout>>
- get_or_create()
- warm_cache(common_keys)
- invalidate(shader_id)

### 2. Vertex Input

```
VertexFormatRegistry
├── formats: HashMap<VertexFormatId, RegisteredFormat>
└── register_standard_formats()
```

**RegisteredFormat:**
- name: String
- attributes: Vec<VertexAttribute>
- stride: u32

**Standard Formats:**
| ID | Name | Stride | Attributes |
|----|------|--------|------------|
| STATIC_MESH | StaticMesh | 48 | pos, normal, uv, tangent |
| SKINNED_MESH | SkinnedMesh | 72 | + joints, weights |
| TERRAIN | Terrain | 32 | pos, normal, uv |
| PARTICLE | Particle | 32 | pos, velocity, life, size |
| UI | UI | 20 | pos2d, uv, color |

### 3. Render Pass

```
TrinityRenderPass
├── encoder: &mut RenderPass
├── current_pipeline: Option<PipelineId>
├── current_bind_groups: [Option<BindGroupId>; 4]
├── viewport: Viewport
└── scissor: Option<ScissorRect>
```

**Draw Commands:**
1. draw()
2. draw_indexed()
3. draw_indirect()
4. draw_indexed_indirect()
5. multi_draw_indirect()
6. multi_draw_indexed_indirect()
7. multi_draw_indirect_count()

**State Commands:**
- set_pipeline()
- set_bind_group()
- set_vertex_buffer()
- set_index_buffer()
- set_viewport()
- set_scissor_rect()
- set_blend_constant()
- set_stencil_reference()
- set_push_constants()

### 4. Render Bundles

```
RenderBundleCache
├── bundles: HashMap<BundleKey, RenderBundle>
├── device: Arc<Device>
└── create_bundle()
```

**BundleKey:**
- mesh_id: MeshId
- material_id: MaterialId
- color_formats: Vec<TextureFormat>
- depth_format: Option<TextureFormat>

### 5. Compute Pipeline

```
TrinityComputePipeline
├── pipeline: wgpu::ComputePipeline
├── layout: Arc<wgpu::PipelineLayout>
├── workgroup_size: [u32; 3]
└── key: ComputePipelineKey
```

**ComputePipelineKey:**
- shader: ShaderId
- entry_point: String
- specialization: HashMap<String, f64>

**ComputePipelineCache:**
- Same pattern as render pipeline

### 6. Compute Library

```
ComputeLibrary
├── reduce_sum: ComputePipeline
├── reduce_min: ComputePipeline
├── reduce_max: ComputePipeline
├── prefix_sum: ComputePipeline
├── radix_sort: ComputePipeline
├── stream_compact: ComputePipeline
├── blur_horizontal: ComputePipeline
├── blur_vertical: ComputePipeline
├── downsample: ComputePipeline
├── histogram: ComputePipeline
├── tonemapping: ComputePipeline
├── frustum_cull: ComputePipeline
├── build_indirect: ComputePipeline
├── depth_reduce: ComputePipeline
├── particle_emit: ComputePipeline
├── particle_simulate: ComputePipeline
└── particle_sort: ComputePipeline
```

**Workgroup Size Guide:**
| Use Case | Size | Total | Rationale |
|----------|------|-------|-----------|
| Linear data | (256, 1, 1) | 256 | Buffer operations |
| Images | (8, 8, 1) | 64 | 2D locality |
| Volumetric | (4, 4, 4) | 64 | 3D locality |
| Particles | (64, 1, 1) | 64 | Per-particle |
| Culling | (64, 1, 1) | 64 | Per-object |

---

## Module Structure

```
crates/renderer-backend/src/pipeline/
├── mod.rs              # Module exports
├── render_pipeline.rs  # RenderPipeline, PipelineCache
├── compute_pipeline.rs # ComputePipeline, ComputePipelineCache
├── vertex_format.rs    # VertexFormatRegistry
├── blend.rs            # BlendMode presets
├── depth_stencil.rs    # DepthConfig, StencilConfig
├── render_pass.rs      # TrinityRenderPass
├── render_bundle.rs    # RenderBundleCache
└── compute/
    ├── mod.rs          # ComputeLibrary
    ├── reduction.rs    # Reduction shaders
    ├── scan.rs         # Prefix scan
    ├── sort.rs         # Radix sort
    ├── compact.rs      # Stream compaction
    ├── image.rs        # Image processing
    ├── culling.rs      # GPU culling
    ├── particles.rs    # Particle simulation
    └── shaders/
        ├── reduce_sum.wgsl
        ├── prefix_scan.wgsl
        ├── radix_sort.wgsl
        ├── stream_compact.wgsl
        └── ...
```

---

## Testing Strategy

### Unit Tests

1. **Pipeline key hashing** - Verify deterministic hashing
2. **Vertex format registry** - Standard format retrieval
3. **Blend mode presets** - Alpha, premultiplied, additive
4. **Depth/stencil config** - Compare function combinations
5. **Compute dispatch sizing** - Workgroup count calculation

### Integration Tests

1. **Render pipeline creation** - Full descriptor
2. **Compute pipeline creation** - With specialization constants
3. **Render pass execution** - Draw commands
4. **Render bundle recording** - Record and replay
5. **Compute patterns** - Reduction, scan verification

### Blackbox Tests

1. **Pipeline cache stats** - Hit rate, size
2. **Draw call execution** - Visual verification
3. **Compute result verification** - Reduction correctness

---

## Performance Considerations

1. **Pipeline Caching** - Avoid runtime compilation
2. **warm_cache()** - Precompile common pipelines
3. **Render Bundles** - Pre-record static geometry
4. **State Tracking** - Skip redundant state changes
5. **Workgroup Sizing** - Match hardware preferences

---

## Dependencies

### External Crates

- `wgpu` - Core GPU abstraction
- `bytemuck` - Vertex data casting

### Internal Dependencies

- Phase 1: TrinityDevice
- Phase 2: ShaderCache, LayoutCache, BindGroupCache

---

## Deliverables Checklist

- [ ] RenderPipeline with full state configuration
- [ ] PipelineCache with warm_cache()
- [ ] VertexFormatRegistry with standard formats
- [ ] BlendMode presets
- [ ] DepthStencilConfig helpers
- [ ] TrinityRenderPass with all commands
- [ ] RenderBundleCache
- [ ] ComputePipeline with specialization
- [ ] ComputeLibrary (17 pipelines)
- [ ] Compute pattern shaders
- [ ] Unit tests (80%+ coverage)
- [ ] Integration tests
- [ ] Documentation

---

*End of PHASE_3_PIPELINES_ARCH.md*

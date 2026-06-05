# PHASE 6: ADVANCED - Architecture

**Scope:** GPU culling, indirect rendering, bindless resources, mesh shaders (future)
**Duration:** 3-4 weeks
**Dependencies:** Phase 4 (SYNCHRONIZATION)
**Produces:** GPU-driven rendering pipeline

---

## Overview

Phase 6 implements advanced rendering techniques that enable massive scene scaling: GPU-driven culling, indirect draw calls, bindless resource management, and mesh shader preparation. These techniques reduce CPU overhead and maximize GPU throughput.

### Covered Content (from MASTER.md Part VIII)

- Chapter 15: Indirect Rendering
  - 15.1 Indirect draw (DrawIndirect, DrawIndexedIndirect, GPU-driven generation)
  - 15.2 GPU culling (frustum, HiZ occlusion, LOD selection, compaction)
  - 15.3 Multi-draw indirect (batching, features)

- Chapter 16: Mesh Shaders (Future)
  - 16.1 Fundamentals (task/mesh stages, meshlets)
  - 16.2 Pipeline (meshlet generation, culling, rendering)
  - 16.3 TRINITY readiness (preprocessor, fallback, abstraction)

- Chapter 17: Bindless Resources
  - 17.1 Fundamentals (texture/buffer arrays, descriptor indexing)
  - 17.2 Patterns (atlas, array, storage indirection, hybrid)
  - 17.3 TRINITY bindless system (registries, material table, allocation)

---

## Architectural Decisions

### ADR-021: GPU-Driven Rendering Pipeline

**Context:** CPU-side draw call submission is the bottleneck for large scenes.

**Decision:** Implement full GPU-driven pipeline:
1. GPU frustum culling
2. HiZ occlusion culling
3. LOD selection
4. Indirect draw buffer generation
5. multi_draw_indirect_count dispatch

**Rationale:** CPU only touches scene once; GPU handles per-frame visibility.

**Consequences:**
- Requires compute shaders per frame
- Scene data in GPU buffers
- CPU workload constant regardless of visible object count

---

### ADR-022: Hierarchical Z-Buffer Strategy

**Context:** Occlusion culling requires depth information from previous frame.

**Decision:** Implement HiZ pyramid:
- Downsample depth to mip chain
- Sample lowest-resolution mip that covers projected bounds
- Conservative depth test (use max in mip)

**Rationale:** Efficient GPU-based occlusion test.

**Consequences:**
- One frame latency (uses previous frame depth)
- False positives possible (conservative)
- Pyramid generation cost per frame

---

### ADR-023: Bindless Material System

**Context:** Material diversity requires dynamic texture binding.

**Decision:** Implement bindless via:
- Texture arrays with dynamic indexing
- Material descriptor buffer
- Per-draw material index

**Rationale:** Unlimited material counts without rebinding.

**Consequences:**
- Requires Advanced capability tier
- Index management overhead
- Fallback for lower tiers

---

### ADR-024: Mesh Shader Abstraction

**Context:** Mesh shaders not yet in wgpu, but TRINITY should be ready.

**Decision:** Implement GeometryPath abstraction:
- Traditional: Vertex -> Rasterizer
- Meshlet: Task -> Mesh -> Rasterizer
- Auto-select based on capability

**Rationale:** Future-proof architecture.

**Consequences:**
- Meshlet preprocessing required
- Dual path maintenance (initially)
- Transition when wgpu supports

---

## Component Breakdown

### 1. Indirect Draw System

```
IndirectDrawSystem
├── draw_buffer: wgpu::Buffer
├── count_buffer: wgpu::Buffer
├── capacity: u32
└── dispatch_helper: DispatchHelper
```

**DrawIndirectArgs (16 bytes):**
- vertex_count: u32
- instance_count: u32
- first_vertex: u32
- first_instance: u32

**DrawIndexedIndirectArgs (20 bytes):**
- index_count: u32
- instance_count: u32
- first_index: u32
- base_vertex: i32 (signed!)
- first_instance: u32

**DispatchIndirectArgs (12 bytes):**
- workgroup_count_x: u32
- workgroup_count_y: u32
- workgroup_count_z: u32

### 2. GPU Culling Pipeline

```
GPUCullingPipeline
├── frustum_cull: ComputePipeline
├── hiz_generate: ComputePipeline
├── hiz_cull: ComputePipeline
├── lod_select: ComputePipeline
├── compact: ComputePipeline
├── build_indirect: ComputePipeline
└── hiz_pyramid: Texture
```

**Scene Data Buffers:**
- object_data: Vec<ObjectData> (transforms, bounds, flags)
- visibility_flags: Vec<u32> (1 bit per object)
- draw_commands: Vec<DrawIndexedIndirectArgs>
- draw_count: u32 (atomic)

**ObjectData struct:**
```rust
#[repr(C)]
pub struct ObjectData {
    pub transform: [[f32; 4]; 4],
    pub aabb_min: [f32; 3],
    pub mesh_index: u32,
    pub aabb_max: [f32; 3],
    pub material_index: u32,
    pub lod_distances: [f32; 4], // LOD 0-3 switch distances
}
```

### 3. Frustum Culling

**WGSL Shader:**
```wgsl
@compute @workgroup_size(64, 1, 1)
fn frustum_cull(@builtin(global_invocation_id) id: vec3<u32>) {
    let object_idx = id.x;
    if (object_idx >= object_count) { return; }
    
    let object = objects[object_idx];
    let visible = test_aabb_frustum(object.aabb_min, object.aabb_max, frustum_planes);
    
    if (visible) {
        atomicOr(&visibility[object_idx / 32u], 1u << (object_idx % 32u));
    }
}
```

### 4. HiZ Occlusion Culling

**Pyramid Generation:**
```wgsl
@compute @workgroup_size(8, 8, 1)
fn hiz_downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    let src_uv = vec2<f32>(id.xy * 2u) / src_size;
    let d0 = textureSampleLevel(depth_src, sampler, src_uv + vec2(0.0, 0.0), 0.0);
    let d1 = textureSampleLevel(depth_src, sampler, src_uv + vec2(1.0, 0.0), 0.0);
    let d2 = textureSampleLevel(depth_src, sampler, src_uv + vec2(0.0, 1.0), 0.0);
    let d3 = textureSampleLevel(depth_src, sampler, src_uv + vec2(1.0, 1.0), 0.0);
    
    // Conservative max for reverse-Z
    let max_depth = max(max(d0, d1), max(d2, d3));
    textureStore(hiz_dst, id.xy, vec4(max_depth, 0.0, 0.0, 0.0));
}
```

**Occlusion Test:**
```wgsl
fn test_occlusion(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    let screen_rect = project_aabb(aabb_min, aabb_max);
    let mip_level = select_mip_level(screen_rect);
    let hiz_depth = textureSampleLevel(hiz_pyramid, sampler, screen_rect.center, mip_level);
    let closest_depth = min_projected_depth(aabb_min, aabb_max);
    
    return closest_depth <= hiz_depth; // visible if closer than HiZ
}
```

### 5. LOD Selection

```wgsl
@compute @workgroup_size(64, 1, 1)
fn select_lod(@builtin(global_invocation_id) id: vec3<u32>) {
    let object_idx = id.x;
    let object = objects[object_idx];
    
    let center = (object.aabb_min + object.aabb_max) * 0.5;
    let distance = length(camera_position - center);
    
    var lod = 0u;
    if (distance > object.lod_distances[0]) { lod = 1u; }
    if (distance > object.lod_distances[1]) { lod = 2u; }
    if (distance > object.lod_distances[2]) { lod = 3u; }
    
    selected_lod[object_idx] = lod;
}
```

### 6. Stream Compaction

**Uses prefix scan from Phase 3 ComputeLibrary:**
```wgsl
@compute @workgroup_size(256, 1, 1)
fn compact(@builtin(global_invocation_id) id: vec3<u32>) {
    let object_idx = id.x;
    let visible = (visibility[object_idx / 32u] >> (object_idx % 32u)) & 1u;
    
    if (visible == 1u) {
        let output_idx = scan_result[object_idx];
        compacted_objects[output_idx] = object_idx;
    }
}
```

### 7. Multi-Draw Indirect

```rust
// Feature check
if device.features().contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT) {
    render_pass.multi_draw_indexed_indirect_count(
        &indirect_buffer,
        0,
        &count_buffer,
        0,
        max_draw_count,
    );
} else {
    // Fallback: CPU readback of count, individual draws
    for i in 0..visible_count {
        render_pass.draw_indexed_indirect(&indirect_buffer, i * 20);
    }
}
```

### 8. Bindless Resources

```
BindlessResourceManager
├── texture_registry: TextureRegistry
├── buffer_registry: BufferRegistry
├── material_table: MaterialTable
└── index_allocator: IndexAllocator
```

**TextureRegistry:**
- textures: Vec<Option<Arc<wgpu::TextureView>>>
- bind_group: wgpu::BindGroup
- dirty: bool

**MaterialDescriptor:**
```rust
#[repr(C)]
pub struct MaterialDescriptor {
    pub albedo_index: u32,
    pub normal_index: u32,
    pub metallic_roughness_index: u32,
    pub emissive_index: u32,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive_strength: f32,
    pub _padding: f32,
}
```

### 9. Mesh Shader Preparation (Future)

```
MeshletPreprocessor
├── meshlet_generator: MeshletGenerator
├── meshlet_buffer: wgpu::Buffer
└── index_buffer: wgpu::Buffer
```

**Meshlet struct:**
```rust
pub struct Meshlet {
    pub vertex_offset: u32,
    pub triangle_offset: u32,
    pub vertex_count: u32,
    pub triangle_count: u32,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub cone_apex: [f32; 3],
    pub cone_axis: [f32; 3],
    pub cone_cutoff: f32,
}
```

**GeometryPath enum:**
```rust
pub enum GeometryPath {
    Traditional,  // Vertex input -> rasterizer
    Meshlet,      // Task -> Mesh -> rasterizer (future)
}

pub trait GeometryRenderer {
    fn render(&self, objects: &[ObjectData], encoder: &mut CommandEncoder);
}
```

---

## Module Structure

```
crates/renderer-backend/src/gpu_driven/
├── mod.rs              # Module exports
├── indirect/
│   ├── mod.rs          # Indirect exports
│   ├── draw_buffer.rs  # IndirectDrawBuffer
│   ├── dispatch.rs     # DispatchHelper
│   └── multi_draw.rs   # Multi-draw wrapper
│
├── culling/
│   ├── mod.rs          # Culling exports
│   ├── frustum.rs      # Frustum culling shader
│   ├── hiz.rs          # HiZ pyramid + occlusion
│   ├── lod.rs          # LOD selection
│   └── compact.rs      # Stream compaction
│
├── bindless/
│   ├── mod.rs          # Bindless exports
│   ├── texture_registry.rs
│   ├── buffer_registry.rs
│   ├── material_table.rs
│   └── index_allocator.rs
│
├── meshlet/ (future)
│   ├── mod.rs
│   ├── generator.rs
│   ├── preprocessor.rs
│   └── renderer.rs
│
└── shaders/
    ├── frustum_cull.wgsl
    ├── hiz_downsample.wgsl
    ├── hiz_cull.wgsl
    ├── lod_select.wgsl
    ├── compact.wgsl
    └── build_indirect.wgsl
```

---

## Testing Strategy

### Unit Tests

1. **Frustum culling** - AABB intersection
2. **HiZ sampling** - Mip selection
3. **LOD selection** - Distance thresholds
4. **Index allocator** - Allocation/free cycle

### Integration Tests

1. **Full culling pipeline** - End-to-end
2. **Multi-draw performance** - Batch vs individual
3. **Bindless material** - Material switching

### Visual Tests

1. **Culling visualization** - Debug visibility
2. **LOD transitions** - Smooth switching
3. **Massive scene** - 100K+ objects

---

## Performance Considerations

1. **GPU Culling** - Massively parallel
2. **HiZ Pyramid** - Low-resolution test
3. **Stream Compaction** - Prefix scan O(n)
4. **Multi-Draw** - Single API call
5. **Bindless** - Zero rebinding

---

## Dependencies

### External Crates

- `wgpu` - Core GPU abstraction
- `bytemuck` - Buffer data casting

### Internal Dependencies

- Phase 2: Buffers, Textures
- Phase 3: ComputeLibrary (prefix scan)
- Phase 4: Command encoding

---

## Deliverables Checklist

- [ ] IndirectDrawBuffer struct
- [ ] DrawIndirectArgs/DrawIndexedIndirectArgs
- [ ] Frustum culling compute shader
- [ ] HiZ pyramid generation
- [ ] HiZ occlusion culling
- [ ] LOD selection shader
- [ ] Stream compaction
- [ ] Indirect buffer generation
- [ ] Multi-draw indirect wrapper
- [ ] TextureRegistry (bindless)
- [ ] BufferRegistry (bindless)
- [ ] MaterialTable
- [ ] IndexAllocator
- [ ] Meshlet struct (future prep)
- [ ] GeometryPath abstraction
- [ ] Unit tests
- [ ] Integration tests
- [ ] Visual tests
- [ ] Documentation

---

*End of PHASE_6_ADVANCED_ARCH.md*

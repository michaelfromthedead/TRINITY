# MASTER — WGPU Documentation Consolidated

**Status:** Active Consolidation
**Last Updated:** 2026-05-27
**SCRIBE Pass:** 10 of 13 (Parts I-VI, X-XII consolidated)
**wgpu Version Target:** 25.x+

---

## Meta: Architectural Philosophy

**Core Principle:** TRINITY implements the **complete wgpu API surface**. Phase gates (P1/P2/P3) are scheduling constraints, not architectural boundaries. The architecture is designed for the complete picture from day one.

**Document Structure:** 12 Parts, 24 Chapters, covering Device/Instance → Resources → Shaders → Pipelines → Synchronization → Ray Tracing → Advanced Rendering → Presentation → Platform → Debugging → Integration.

---

# PART I: DEVICE & INSTANCE MODEL

## Chapter 1: The wgpu Object Model

### 1.1 Instance

- **1.1.1 Instance Creation**: Entry point to wgpu; selects backend(s)
- **1.1.2 Backend Selection**: Vulkan, Metal, DX12, OpenGL, WebGPU
- **1.1.3 Instance Flags**: Debugging layers, validation
- **1.1.4 TRINITY Strategy**: Multi-backend support with automatic selection

### 1.2 Adapter

- **1.2.1 Enumeration**: List available GPUs
- **1.2.2 Properties**: Vendor, device type, driver info
- **1.2.3 Limits**: Max texture dimensions, buffer size, bind groups
- **1.2.4 Feature Detection**: Capability matrices
- **1.2.5 Power Preference**: High-performance vs low-power
- **1.2.6 TRINITY Selection**: Algorithm for optimal adapter choice

### 1.3 Device

- **1.3.1 Creation**: From adapter with feature/limit requests
- **1.3.2 Features**: Required vs optional
- **1.3.3 Limits**: Negotiation with adapter
- **1.3.4 Device Lost**: Handling and recovery
- **1.3.5 Error Scopes**: Fine-grained error handling
- **1.3.6 TRINITY Lifecycle**: Device management patterns

### 1.4 Queue

- **1.4.1 Submission Model**: Command buffer submission
- **1.4.2 Write Operations**: Direct buffer/texture writes
- **1.4.3 Synchronization**: Queue-level semantics
- **1.4.4 Multi-Queue**: Platform-dependent support
- **1.4.5 TRINITY Batching**: Submission optimization strategy

---

# PART II: RESOURCE MODEL

## Chapter 2: Buffers

### 2.1 Buffer Fundamentals

- **Usage Flags**: VERTEX, INDEX, UNIFORM, STORAGE, INDIRECT, COPY_SRC, COPY_DST, MAP_READ, MAP_WRITE, QUERY_RESOLVE
- **Mapping**: Synchronous vs asynchronous
- **Destruction**: Resource cleanup patterns

### 2.2 Buffer Types

- Vertex buffers, Index buffers (u16/u32)
- Uniform buffers (dynamic offsets)
- Storage buffers (read-only/read-write)
- Indirect buffers (draw/dispatch)
- Staging buffers (upload/readback)

### 2.3 Memory Management

- Suballocation, Pooling, Ring buffers
- Persistent mapping patterns
- **TRINITY Allocator**: Custom buffer allocator architecture

## Chapter 3: Textures

### 3.1 Fundamentals

- **Dimensions**: 1D, 2D, 3D
- **Formats**: Color, depth, stencil, compressed
- **Usage**: TEXTURE_BINDING, STORAGE_BINDING, RENDER_ATTACHMENT, COPY_SRC, COPY_DST
- **Mip Levels**, Array Layers, Cube Maps, Multisampling

### 3.2 Texture Formats

- **Uncompressed**: R8, RG8, RGBA8, R16, RG16, RGBA16, R32, RG32, RGBA32
- **Float**: R16Float, RG16Float, RGBA16Float, R32Float, RG32Float, RGBA32Float
- **sRGB**: Gamma-correct formats
- **Depth**: Depth16Unorm, Depth24Plus, Depth24PlusStencil8, Depth32Float, Depth32FloatStencil8
- **Compressed**: BC (DXT), ETC2, ASTC
- **TRINITY Strategy**: Format selection based on platform/usage

### 3.3 Texture Views

- View creation, dimension conversion
- Format reinterpretation
- Mip/layer subranges, aspect selection

### 3.4 Samplers

- **Address Modes**: ClampToEdge, Repeat, MirrorRepeat, ClampToBorder
- **Filters**: Nearest, Linear, Anisotropic
- **Comparison**: For shadow mapping
- **TRINITY Cache**: Sampler deduplication

### 3.5 Texture Operations

- Queue uploads, texture copies
- Mip generation, streaming foundations

## Chapter 4: Bind Groups & Layouts

### 4.1 Binding Model

- Bind group concept (descriptor set equivalent)
- Layout as contract
- Binding types: buffer, sampler, texture, storage texture, external texture
- Visibility: vertex, fragment, compute

### 4.2 Buffer Bindings

- Uniform, storage (read-only/read-write)
- Dynamic uniform/storage buffers
- Minimum binding size constraints

### 4.3 Texture & Sampler Bindings

- Sampled, storage, multisampled, depth textures
- External textures (video)

### 4.4 Advanced Binding Patterns

- Bindless via storage buffers
- Texture arrays for bindless
- Descriptor indexing, non-uniform indexing
- **TRINITY Bindless**: Architecture for material system

### 4.5 Pipeline Layouts

- Creation, bind group grouping
- Push constants (platform-dependent)
- Layout compatibility rules
- **TRINITY Caching**: Layout deduplication

---

# PART III: SHADER COMPILATION

## Chapter 5: WGSL & Naga

### 5.1 WGSL Language

- **Types**: Scalar, vector, matrix, array, struct
- **Address Spaces**: function, private, workgroup, uniform, storage, handle
- **Attributes**: @vertex, @fragment, @compute, @binding, @group, @location, @builtin
- **Extensions**: Feature gates for experimental

### 5.2 Naga Compiler

- **Pipeline**: WGSL → IR → Validation → Backend
- **Targets**: SPIR-V, MSL, HLSL, GLSL, WGSL
- **Caching**: Compilation result caching
- **TRINITY Hot-Reload**: Shader recompilation system

### 5.3 Shader Modules

- Creation, error handling
- Reflection (where available)
- Runtime vs ahead-of-time compilation

### 5.4 Shader Specialization

- Override constants (pipeline-overridable)
- Permutation management
- **TRINITY Variants**: Shader variant system

---

# PART IV: RENDER PIPELINE — DETAILED IMPLEMENTATION

## Chapter 6: Graphics Pipeline

### 6.1 Pipeline Creation

#### 6.1.1 Render Pipeline Descriptor

```rust
let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
    label: Some("PBR Pipeline"),
    layout: Some(&pipeline_layout),
    vertex: VertexState {
        module: &vertex_shader,
        entry_point: "vs_main",
        compilation_options: Default::default(),
        buffers: &[Vertex::layout()],
    },
    primitive: PrimitiveState {
        topology: PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: FrontFace::Ccw,
        cull_mode: Some(Face::Back),
        unclipped_depth: false,
        polygon_mode: PolygonMode::Fill,
        conservative: false,
    },
    depth_stencil: Some(DepthStencilState {
        format: TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: CompareFunction::Less,
        stencil: StencilState::default(),
        bias: DepthBiasState::default(),
    }),
    multisample: MultisampleState {
        count: 4,
        mask: !0,
        alpha_to_coverage_enabled: false,
    },
    fragment: Some(FragmentState {
        module: &fragment_shader,
        entry_point: "fs_main",
        compilation_options: Default::default(),
        targets: &[Some(ColorTargetState {
            format: TextureFormat::Rgba8UnormSrgb,
            blend: Some(BlendState::ALPHA_BLENDING),
            write_mask: ColorWrites::ALL,
        })],
    }),
    multiview: None,
    cache: None,
});
```

#### 6.1.2 Pipeline Layout Association

```rust
let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
    label: Some("PBR Layout"),
    bind_group_layouts: &[
        &global_layout,    // Group 0: Camera, lights, environment
        &material_layout,  // Group 1: Material parameters
        &object_layout,    // Group 2: Per-object transforms
    ],
    push_constant_ranges: &[
        PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..64,  // Model matrix
        },
    ],
});
```

#### 6.1.3 Vertex State Configuration

```rust
VertexState {
    module: &shader_module,
    entry_point: "vs_main",
    compilation_options: PipelineCompilationOptions {
        constants: &HashMap::new(),
        zero_initialize_workgroup_memory: true,
    },
    buffers: &[
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                VertexAttribute { format: VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                VertexAttribute { format: VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                VertexAttribute { format: VertexFormat::Float32x2, offset: 24, shader_location: 2 },
                VertexAttribute { format: VertexFormat::Float32x4, offset: 32, shader_location: 3 },
            ],
        },
        VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceData>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &[
                VertexAttribute { format: VertexFormat::Float32x4, offset: 0, shader_location: 4 },
                VertexAttribute { format: VertexFormat::Float32x4, offset: 16, shader_location: 5 },
                VertexAttribute { format: VertexFormat::Float32x4, offset: 32, shader_location: 6 },
                VertexAttribute { format: VertexFormat::Float32x4, offset: 48, shader_location: 7 },
            ],
        },
    ],
}
```

#### 6.1.4 Primitive State

```rust
PrimitiveState {
    topology: PrimitiveTopology::TriangleList,
    strip_index_format: None,
    front_face: FrontFace::Ccw,
    cull_mode: Some(Face::Back),
    unclipped_depth: false,
    polygon_mode: PolygonMode::Fill,
    conservative: false,
}
```

#### 6.1.5 Depth/Stencil State

```rust
DepthStencilState {
    format: TextureFormat::Depth32Float,
    depth_write_enabled: true,
    depth_compare: CompareFunction::Less,
    stencil: StencilState {
        front: StencilFaceState {
            compare: CompareFunction::Always,
            fail_op: StencilOperation::Keep,
            depth_fail_op: StencilOperation::Keep,
            pass_op: StencilOperation::Keep,
        },
        back: StencilFaceState::default(),
        read_mask: 0xFF,
        write_mask: 0xFF,
    },
    bias: DepthBiasState { constant: 0, slope_scale: 0.0, clamp: 0.0 },
}
```

#### 6.1.6 Multisample State

```rust
MultisampleState {
    count: 4,
    mask: !0,
    alpha_to_coverage_enabled: false,
}
```

#### 6.1.7 Fragment State and Color Targets (MRT)

```rust
FragmentState {
    module: &fragment_shader,
    entry_point: "fs_main",
    compilation_options: Default::default(),
    targets: &[
        Some(ColorTargetState { format: TextureFormat::Rgba8UnormSrgb, blend: None, write_mask: ColorWrites::ALL }),
        Some(ColorTargetState { format: TextureFormat::Rgba16Float, blend: None, write_mask: ColorWrites::ALL }),
        Some(ColorTargetState { format: TextureFormat::Rgba8Unorm, blend: None, write_mask: ColorWrites::ALL }),
    ],
}
```

#### 6.1.8 TRINITY PSO Pipeline Caching

```rust
pub struct PipelineCache {
    device: Arc<Device>,
    pipelines: HashMap<PipelineKey, Arc<RenderPipeline>>,
    layouts: HashMap<LayoutKey, Arc<PipelineLayout>>,
}

#[derive(Hash, Eq, PartialEq)]
struct PipelineKey {
    vertex_shader: ShaderId,
    fragment_shader: ShaderId,
    vertex_layout: VertexLayoutId,
    render_target_formats: Vec<TextureFormat>,
    depth_format: Option<TextureFormat>,
    sample_count: u32,
    blend_mode: BlendMode,
    cull_mode: CullMode,
    depth_write: bool,
    depth_compare: CompareFunction,
}

impl PipelineCache {
    pub fn get_or_create(&mut self, key: &PipelineKey) -> Arc<RenderPipeline> {
        if let Some(pipeline) = self.pipelines.get(key) { return pipeline.clone(); }
        let pipeline = Arc::new(self.create_pipeline(key));
        self.pipelines.insert(key.clone(), pipeline.clone());
        pipeline
    }
    
    pub fn warm_cache(&mut self, common_keys: &[PipelineKey]) {
        for key in common_keys { self.get_or_create(key); }
    }
}
```

### 6.2 Vertex Input

#### 6.2.1 Vertex Buffer Layouts

```rust
pub struct VertexBufferLayout<'a> {
    pub array_stride: BufferAddress,
    pub step_mode: VertexStepMode,
    pub attributes: &'a [VertexAttribute],
}

pub struct VertexAttribute {
    pub format: VertexFormat,
    pub offset: BufferAddress,
    pub shader_location: ShaderLocation,
}

pub enum VertexStepMode { Vertex, Instance }
```

#### 6.2.2 Vertex Attribute Formats

| Format | Size | Components | Type |
|--------|------|------------|------|
| `Float32x3` | 12 | 3 | f32 |
| `Float32x4` | 16 | 4 | f32 |
| `Uint32` | 4 | 1 | u32 |
| `Unorm8x4` | 4 | 4 | f32 [0, 1] |
| `Uint16x4` | 8 | 4 | u32 |

#### 6.2.3 Step Modes

```rust
VertexBufferLayout { array_stride: 48, step_mode: VertexStepMode::Vertex, attributes: &[/*...*/] }
VertexBufferLayout { array_stride: 80, step_mode: VertexStepMode::Instance, attributes: &[/*...*/] }
```

#### 6.2.4 Interleaved vs Separate Buffers

| Aspect | Interleaved | Separate |
|--------|-------------|----------|
| Cache coherence | Better for full vertex access | Better for partial access |
| Memory | Single allocation | Multiple allocations |
| Flexibility | Must use all attributes | Can mix and match |

#### 6.2.5 TRINITY Vertex Format Registry

```rust
pub struct VertexFormatRegistry {
    formats: HashMap<VertexFormatId, RegisteredFormat>,
}

impl VertexFormatRegistry {
    pub fn register_standard_formats(&mut self) {
        self.register(VertexFormatId::STATIC_MESH, "StaticMesh", &[
            VertexAttribute { format: Float32x3, offset: 0, shader_location: 0 },
            VertexAttribute { format: Float32x3, offset: 12, shader_location: 1 },
            VertexAttribute { format: Float32x2, offset: 24, shader_location: 2 },
            VertexAttribute { format: Float32x4, offset: 32, shader_location: 3 },
        ], 48);
    }
}
```

### 6.3 Primitive Assembly

#### 6.3.1 Primitive Topologies

```rust
pub enum PrimitiveTopology { PointList, LineList, LineStrip, TriangleList, TriangleStrip }
```

#### 6.3.2 Index Formats

```rust
pub enum IndexFormat { Uint16, Uint32 }
```

#### 6.3.3 Front Face and Culling

```rust
pub enum FrontFace { Ccw, Cw }
pub enum Face { Front, Back }
```

#### 6.3.4 Polygon Modes

```rust
pub enum PolygonMode { Fill, Line, Point }
```

### 6.4 Rasterization

#### 6.4.1 Viewport and Scissor

```rust
render_pass.set_viewport(x, y, width, height, min_depth, max_depth);
render_pass.set_scissor_rect(x, y, width, height);
```

#### 6.4.2 Depth Bias

```rust
DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 }
```

#### 6.4.3 Conservative Rasterization

```rust
PrimitiveState { conservative: true, .. }  // Requires CONSERVATIVE_RASTERIZATION
```

#### 6.4.4 Sample Mask

```rust
MultisampleState { mask: 0b1010, .. }
```

#### 6.4.5 Alpha to Coverage

```rust
MultisampleState { alpha_to_coverage_enabled: true, .. }
```

### 6.5 Fragment Processing

#### 6.5.1 Fragment Shader Outputs

```wgsl
struct FragmentOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) material: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    out.albedo = textureSample(albedo_map, sampler, in.uv);
    out.normal = vec4(encode_normal(in.world_normal), 1.0);
    out.material = vec4(metallic, roughness, ao, 1.0);
    return out;
}
```

#### 6.5.2 Color Target State

```rust
ColorTargetState {
    format: TextureFormat::Rgba8UnormSrgb,
    blend: Some(BlendState {
        color: BlendComponent { src_factor: BlendFactor::SrcAlpha, dst_factor: BlendFactor::OneMinusSrcAlpha, operation: BlendOperation::Add },
        alpha: BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::OneMinusSrcAlpha, operation: BlendOperation::Add },
    }),
    write_mask: ColorWrites::ALL,
}
```

#### 6.5.3 Write Mask

```rust
pub struct ColorWrites: u32 { const RED = 0x1; const GREEN = 0x2; const BLUE = 0x4; const ALPHA = 0x8; const ALL = 0xF; }
```

#### 6.5.4 Blending

```rust
pub enum BlendFactor { Zero, One, Src, OneMinusSrc, SrcAlpha, OneMinusSrcAlpha, Dst, OneMinusDst, DstAlpha, OneMinusDstAlpha, SrcAlphaSaturated, Constant, OneMinusConstant }
pub enum BlendOperation { Add, Subtract, ReverseSubtract, Min, Max }
```

### 6.6 Depth/Stencil

#### 6.6.1 Depth Test

```rust
DepthStencilState { format: TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: CompareFunction::Less, .. }
```

#### 6.6.2 Compare Functions

```rust
pub enum CompareFunction { Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always }
```

#### 6.6.3-6.6.4 Stencil State and Operations

```rust
StencilState {
    front: StencilFaceState { compare: CompareFunction::Always, fail_op: StencilOperation::Keep, depth_fail_op: StencilOperation::Keep, pass_op: StencilOperation::Replace },
    back: StencilFaceState::default(),
    read_mask: 0xFF, write_mask: 0xFF,
}

pub enum StencilOperation { Keep, Zero, Replace, Invert, IncrementClamp, DecrementClamp, IncrementWrap, DecrementWrap }
```

### 6.7 Multisampling

#### 6.7.1 Sample Count Selection

```rust
let max_samples = if supported.contains(TextureFormatFeatureFlags::MULTISAMPLE_X16) { 16 }
    else if supported.contains(TextureFormatFeatureFlags::MULTISAMPLE_X8) { 8 }
    else if supported.contains(TextureFormatFeatureFlags::MULTISAMPLE_X4) { 4 }
    else { 1 };
```

#### 6.7.2 MSAA Resolve

```rust
RenderPassColorAttachment {
    view: &msaa_view,
    resolve_target: Some(&resolve_view),
    ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Discard },
}
```

## Chapter 7: Render Passes

### 7.1 Render Pass Fundamentals

```rust
let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
    label: Some("Main Pass"),
    color_attachments: &[Some(RenderPassColorAttachment {
        view: &color_view, resolve_target: None,
        ops: Operations { load: LoadOp::Clear(Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0 }), store: StoreOp::Store },
    })],
    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
        view: &depth_view,
        depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Discard }),
        stencil_ops: None,
    }),
    timestamp_writes: None,
    occlusion_query_set: None,
});
```

### 7.2 Attachment Operations

```rust
pub enum LoadOp<V> { Clear(V), Load }
pub enum StoreOp { Store, Discard }
```

### 7.3 Render Pass Commands

```rust
render_pass.set_pipeline(&pipeline);
render_pass.set_bind_group(0, &global_bind_group, &[]);
render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
render_pass.set_index_buffer(index_buffer.slice(..), IndexFormat::Uint32);
render_pass.set_viewport(0.0, 0.0, width, height, 0.0, 1.0);
render_pass.set_scissor_rect(0, 0, width as u32, height as u32);
render_pass.set_blend_constant(Color::WHITE);
render_pass.set_stencil_reference(0xFF);
render_pass.set_push_constants(ShaderStages::VERTEX, 0, bytemuck::bytes_of(&transform));
```

### 7.4 Draw Commands

```rust
render_pass.draw(vertex_count, instance_count, first_vertex, first_instance);
render_pass.draw_indexed(index_count, instance_count, first_index, base_vertex, first_instance);
render_pass.draw_indirect(&indirect_buffer, offset);
render_pass.draw_indexed_indirect(&indirect_buffer, offset);
render_pass.multi_draw_indirect(&indirect_buffer, offset, count);
render_pass.multi_draw_indexed_indirect(&indirect_buffer, offset, count);
render_pass.multi_draw_indirect_count(&indirect_buffer, indirect_offset, &count_buffer, count_offset, max_count);
```

### 7.5 Render Bundles

```rust
let mut bundle_encoder = device.create_render_bundle_encoder(&RenderBundleEncoderDescriptor {
    label: Some("Static Geometry Bundle"),
    color_formats: &[Some(TextureFormat::Rgba8UnormSrgb)],
    depth_stencil: Some(RenderBundleDepthStencil { format: TextureFormat::Depth32Float, depth_read_only: false, stencil_read_only: true }),
    sample_count: 1, multiview: None,
});

bundle_encoder.set_pipeline(&pipeline);
bundle_encoder.set_bind_group(0, &bind_group, &[]);
for mesh in &static_meshes {
    bundle_encoder.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    bundle_encoder.set_index_buffer(mesh.index_buffer.slice(..), IndexFormat::Uint32);
    bundle_encoder.draw_indexed(mesh.index_count, 1, 0, 0, 0);
}
let bundle = bundle_encoder.finish(&RenderBundleDescriptor { label: Some("Static Geometry") });
render_pass.execute_bundles([&bundle]);
```

**TRINITY Bundle Cache:**
```rust
pub struct RenderBundleCache {
    bundles: HashMap<BundleKey, RenderBundle>,
    device: Arc<Device>,
}
```

---

# PART V: COMPUTE PIPELINE — DETAILED IMPLEMENTATION

## Chapter 8: Compute Fundamentals

### 8.1 Compute Pipeline

```rust
let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
    label: Some("Particle Update"),
    layout: Some(&pipeline_layout),
    module: &compute_shader,
    entry_point: "main",
    compilation_options: PipelineCompilationOptions {
        constants: &[("WORKGROUP_SIZE", 256.0)].into_iter().collect(),
        zero_initialize_workgroup_memory: true,
    },
    cache: None,
});
```

### 8.2 Compute Shaders

#### 8.2.1 @compute Entry Points

```wgsl
@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) { /* ... */ }
```

#### 8.2.2 @workgroup_size Attribute

| Use Case | Size | Total | Rationale |
|----------|------|-------|-----------|
| Linear data | (256, 1, 1) | 256 | Good for buffers |
| Images | (8, 8, 1) | 64 | 2D spatial locality |
| Volumetric | (4, 4, 4) | 64 | 3D spatial locality |

#### 8.2.3 Built-in Variables

```wgsl
@builtin(global_invocation_id) global_id: vec3<u32>   // [0, dispatch_size * workgroup_size)
@builtin(local_invocation_id) local_id: vec3<u32>     // [0, workgroup_size)
@builtin(local_invocation_index) local_index: u32     // Flattened local ID
@builtin(workgroup_id) workgroup_id: vec3<u32>        // [0, dispatch_size)
@builtin(num_workgroups) num_workgroups: vec3<u32>    // Dispatch dimensions
```

#### 8.2.4 Workgroup Memory

```wgsl
var<workgroup> shared_data: array<f32, 256>;
var<workgroup> histogram: array<atomic<u32>, 256>;

@compute @workgroup_size(256, 1, 1)
fn reduction() {
    let idx = local_invocation_index;
    shared_data[idx] = input_data[global_invocation_id.x];
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (idx < stride) { shared_data[idx] += shared_data[idx + stride]; }
        workgroupBarrier();
    }
    if (idx == 0u) { output[workgroup_id.x] = shared_data[0]; }
}
```

#### 8.2.5 Synchronization

```wgsl
workgroupBarrier();  // Full barrier: execution + all memory
storageBarrier();    // Storage buffer memory only
textureBarrier();    // Texture memory (where supported)
```

### 8.3 Compute Pass

```rust
let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
    label: Some("Particle Update Pass"),
    timestamp_writes: Some(ComputePassTimestampWrites { query_set: &timestamp_query_set, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1) }),
});

compute_pass.set_pipeline(&compute_pipeline);
compute_pass.set_bind_group(0, &input_bind_group, &[]);
compute_pass.set_push_constants(0, bytemuck::bytes_of(&push_constants));
```

### 8.4 Dispatch Commands

```rust
compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, workgroup_count_z);

// Indirect dispatch
compute_pass.dispatch_workgroups_indirect(&indirect_buffer, 0);
```

### 8.5 Compute Patterns

#### 8.5.1 Parallel Reduction

```wgsl
@compute @workgroup_size(256, 1, 1)
fn reduce_sum(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_index) local_idx: u32, @builtin(workgroup_id) wg_id: vec3<u32>) {
    if (global_id.x < array_size) { shared_data[local_idx] = input[global_id.x]; }
    else { shared_data[local_idx] = 0.0; }
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (local_idx < stride) { shared_data[local_idx] += shared_data[local_idx + stride]; }
        workgroupBarrier();
    }
    if (local_idx == 0u) { partial_sums[wg_id.x] = shared_data[0]; }
}
```

#### 8.5.2 Prefix Scan

```wgsl
@compute @workgroup_size(256, 1, 1)
fn prefix_sum_block(@builtin(local_invocation_index) idx: u32, @builtin(workgroup_id) wg_id: vec3<u32>) {
    let n = 256u;
    temp[idx] = input[wg_id.x * n + idx];
    workgroupBarrier();
    // Up-sweep
    for (var d = n >> 1u; d > 0u; d >>= 1u) {
        if (idx < d) { let ai = (idx * 2u + 1u) * (n / d) - 1u; let bi = (idx * 2u + 2u) * (n / d) - 1u; temp[bi] += temp[ai]; }
        workgroupBarrier();
    }
    if (idx == 0u) { block_sums[wg_id.x] = temp[n - 1u]; temp[n - 1u] = 0.0; }
    workgroupBarrier();
    // Down-sweep
    for (var d = 1u; d < n; d <<= 1u) {
        if (idx < d) { let ai = (idx * 2u + 1u) * (n / d) - 1u; let bi = (idx * 2u + 2u) * (n / d) - 1u; let t = temp[ai]; temp[ai] = temp[bi]; temp[bi] += t; }
        workgroupBarrier();
    }
    output[wg_id.x * n + idx] = temp[idx];
}
```

#### 8.5.3 Stream Compaction

```wgsl
@compute @workgroup_size(256, 1, 1)
fn compact(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= particle_count) { return; }
    let particle = input[idx];
    if (particle.lifetime > 0.0) {
        let out_idx = scan_result[idx];
        output[out_idx] = particle;
    }
}
```

#### 8.5.4-8.5.7 Additional Patterns

- Radix Sort, Histogram, Image Processing (Gaussian Blur), Physics Simulation

#### 8.5.8 TRINITY Compute Library

```rust
pub struct ComputeLibrary {
    pub reduce_sum: ComputePipeline, pub reduce_min: ComputePipeline, pub reduce_max: ComputePipeline,
    pub prefix_sum: ComputePipeline, pub radix_sort: ComputePipeline, pub stream_compact: ComputePipeline,
    pub blur_horizontal: ComputePipeline, pub blur_vertical: ComputePipeline,
    pub downsample: ComputePipeline, pub histogram: ComputePipeline, pub tonemapping: ComputePipeline,
    pub frustum_cull: ComputePipeline, pub build_indirect: ComputePipeline, pub depth_reduce: ComputePipeline,
    pub particle_emit: ComputePipeline, pub particle_simulate: ComputePipeline, pub particle_sort: ComputePipeline,
}
```

---

# PART VI: SYNCHRONIZATION & COMMANDS — DETAILED IMPLEMENTATION

## Chapter 9: Command Encoding

### 9.1 Command Encoder

```rust
let encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: Some("MainEncoder") });
// Record commands
encoder.copy_buffer_to_buffer(&staging, 0, &gpu_buffer, 0, size);
{ let mut render_pass = encoder.begin_render_pass(&desc); render_pass.draw(...); }
{ let mut compute_pass = encoder.begin_compute_pass(&desc); compute_pass.dispatch_workgroups(...); }
let command_buffer = encoder.finish();
queue.submit(std::iter::once(command_buffer));
```

### 9.2 Copy Commands

```rust
encoder.copy_buffer_to_buffer(source, source_offset, destination, destination_offset, copy_size);
encoder.copy_buffer_to_texture(source, destination, copy_size);
encoder.copy_texture_to_buffer(source, destination, copy_size);
encoder.copy_texture_to_texture(source, destination, copy_size);
```

**Copy Alignment:**
| Constraint | Value |
|------------|-------|
| Buffer offset | 4 bytes |
| Bytes per row | 256 bytes |
| Copy size | 4 bytes |

### 9.3 Clear Commands

```rust
encoder.clear_buffer(buffer, offset, size);  // Clears to zero
```

### 9.4 Query Commands

```rust
let query_set = device.create_query_set(&QuerySetDescriptor { label: Some("Timestamps"), ty: QueryType::Timestamp, count: 64 });
encoder.resolve_query_set(&query_set, 0..64, &resolve_buffer, 0);
```

### 9.5 Debug Commands

```rust
encoder.push_debug_group("Shadow Map Generation");
encoder.insert_debug_marker("Begin Culling");
encoder.pop_debug_group();
```

## Chapter 10: Synchronization

### 10.1 Implicit Synchronization

wgpu automatically handles transitions between passes:
```rust
encoder.copy_buffer_to_buffer(&staging, 0, &gpu_buffer, 0, size);  // COPY_DST
{ let mut pass = encoder.begin_compute_pass(&desc); pass.dispatch_workgroups(64, 1, 1); }  // Barrier auto-inserted
```

### 10.2 Explicit Synchronization

```wgsl
workgroupBarrier();  // Control + memory barrier within workgroup
storageBarrier();    // Storage buffer memory barrier
```

### 10.3 CPU-GPU Synchronization

```rust
buffer.slice(..).map_async(wgpu::MapMode::Read, |result| { /* ... */ });
device.poll(wgpu::Maintain::Wait);
```

**Frame Pacing (Double Buffering):**
```rust
impl DoubleBufferedRenderer {
    pub fn render_frame(&mut self) {
        self.frame_fences[self.current_frame].wait(&self.device);
        let output = self.surface.get_current_texture().unwrap();
        let cmd = self.record_frame(&output);
        let submission = self.queue.submit(std::iter::once(cmd));
        self.frame_fences[self.current_frame].submission_index = Some(submission);
        output.present();
        self.current_frame = (self.current_frame + 1) % 2;
    }
}
```

### 10.4 Resource State Tracking

**TRINITY Barrier Resolution:**
```rust
pub struct BarrierResolver {
    resource_states: HashMap<ResourceHandle, ResourceState>,
    pending_barriers: Vec<Barrier>,
}

impl BarrierResolver {
    pub fn transition(&mut self, resource: ResourceHandle, new_state: ResourceState) {
        let old_state = self.resource_states.get(&resource).copied().unwrap_or(ResourceState::UNDEFINED);
        if self.needs_barrier(old_state, new_state) {
            self.pending_barriers.push(Barrier { resource, old_state, new_state });
        }
        self.resource_states.insert(resource, new_state);
    }
}
```

---

# PART VII: RAY TRACING

## Chapter 11: Acceleration Structures

### 11.1 Fundamentals

- **BVH Concepts**: BLAS (geometry) and TLAS (instances)
- Two-level hierarchy rationale
- `acceleration_structure` feature

### 11.2 BLAS Construction

- **Geometry Types**: Triangles, AABBs
- **Geometry Flags**: OPAQUE, NO_DUPLICATE_ANYHIT_INVOCATION
- **Build Flags**: PREFER_FAST_TRACE, PREFER_FAST_BUILD, ALLOW_UPDATE, ALLOW_COMPACTION, LOW_MEMORY
- Scratch buffer management
- **TRINITY BLAS Builder**: Custom construction

### 11.3 BLAS Compaction

- Memory savings via compaction
- Size query, compaction copy
- When to compact vs skip

### 11.4 BLAS Update (Refit)

- Update vs rebuild trade-offs
- Refit quality degradation
- **TRINITY Dynamic Policy**: When to refit vs rebuild

### 11.5 TLAS Construction

- **Instance Descriptor**: BLAS ref, transform, instance ID, mask, SBT offset, flags
- Per-frame rebuild
- Instance culling integration
- **TRINITY TLAS Builder**: Custom construction

### 11.6 AS Memory Management

- Requirements query
- Scratch memory, allocation
- **TRINITY AS Manager**: Memory budget tracking

## Chapter 12: Ray Queries (Inline)

### 12.1 Fundamentals

- `ray_query` feature
- Ray query vs RT pipeline
- **Use Cases**: Shadows, AO, simple reflections

### 12.2 WGSL API

- RayQuery type
- rayQueryInitialize(), rayQueryProceed()
- rayQueryGetCommittedIntersection*(), rayQueryGetCandidateIntersection*()
- rayQueryConfirmIntersection(), rayQueryTerminate()

### 12.3 Ray Flags

- RAY_FLAG_FORCE_OPAQUE, RAY_FLAG_TERMINATE_ON_FIRST_HIT
- Culling flags (front/back facing, opaque/non-opaque)
- Skip flags (triangles, AABBs)

### 12.4 Patterns

- Shadow ray (early termination)
- Closest hit, any-hit with alpha testing
- **TRINITY Ray Query Library**: Standard patterns

## Chapter 13: Ray Tracing Pipelines

### 13.1 Fundamentals

- `ray_tracing_pipeline` feature (experimental)
- **Stages**: Ray generation, intersection, any-hit, closest-hit, miss, callable
- Recursion depth

### 13.2 Shader Stages

- @raygeneration, @intersection, @anyhit, @closesthit, @miss, @callable

### 13.3 Hit Groups

- Triangle hit groups (closest-hit + optional any-hit)
- Procedural hit groups (intersection + closest-hit + optional any-hit)

### 13.4 Shader Binding Table (SBT)

- Layout, records, indexing
- Stride and alignment
- **TRINITY SBT Builder**: Custom SBT construction

### 13.5 Pipeline Creation

- Shader modules, descriptor
- Max recursion/payload/attribute sizes
- Pipeline libraries

### 13.6 Dispatch

- TraceRay intrinsic
- Dimensions, payload/attribute passing
- Recursive tracing

### 13.7 Patterns

- Primary rays, shadow rays, reflection/refraction
- AO, GI (single bounce), path tracing
- **TRINITY RT Effects**: Effect library

## Chapter 14: RT Advanced Features

### 14.1 Opacity Micromaps (OMM)

- Alpha testing acceleration
- wgpu status: not yet

### 14.2 Displacement Micromaps (DMM)

- Micro-geometry detail
- wgpu status: not yet

### 14.3 Shader Execution Reordering (SER)

- Coherent ray sorting (NVIDIA-specific)
- wgpu status: not yet

### 14.4 Motion Blur

- Motion BLAS/TLAS
- wgpu status: not yet

---

# PART VIII: ADVANCED RENDERING

## Chapter 15: Indirect Rendering

### 15.1 Indirect Draw

- DrawIndirect, DrawIndexedIndirect buffer layouts
- GPU-driven draw call generation
- Indirect count

### 15.2 GPU Culling

- Frustum culling in compute
- Occlusion culling (Hi-Z)
- GPU-driven LOD selection
- Buffer compaction
- **TRINITY GPU Culling**: Full pipeline

### 15.3 Multi-Draw Indirect

- `multi_draw_indirect`, `multi_draw_indexed_indirect` features
- Batching, performance implications

## Chapter 16: Mesh Shaders (Future)

### 16.1 Fundamentals

- `mesh_shaders` feature (not yet in wgpu)
- Task shader, mesh shader stages
- Meshlet concept

### 16.2 Pipeline

- Meshlet generation, culling, rendering
- Vertex deduplication

### 16.3 TRINITY Readiness

- Meshlet preprocessing
- Fallback to traditional pipeline
- Abstraction layer

## Chapter 17: Bindless Resources

### 17.1 Fundamentals

- Bindless texture/buffer arrays
- Descriptor indexing, non-uniform indexing

### 17.2 Patterns

- Texture atlas, texture array
- Storage buffer indirection
- Hybrid approaches

### 17.3 TRINITY Bindless System

- Texture registry, buffer registry
- Material table
- Index allocation/recycling

---

# PART IX: PRESENTATION

## Chapter 18: Surface & Swapchain

### 18.1 Surface

- Creation from window handle
- Capabilities query
- Formats, present modes, alpha modes

### 18.2 Configuration

- Format selection
- **Present Modes**: Immediate, Mailbox, Fifo, FifoRelaxed
- **Alpha Modes**: Auto, Opaque, PreMultiplied, PostMultiplied
- View formats for sRGB reinterpretation

### 18.3 Frame Acquisition

- get_current_texture()
- Suboptimal/outdated handling
- Resize reconfiguration

### 18.4 Presentation

- present() call
- VSync, frame pacing
- Triple buffering
- **TRINITY Presentation Engine**: Frame timing system

---

# PART X: PLATFORM & BACKENDS

## Chapter 19: Platform Considerations

wgpu abstracts multiple graphics APIs behind a unified interface. Understanding backend-specific behaviors is essential for optimal cross-platform performance.

### 19.1 Vulkan Backend

#### 19.1.1 Vulkan Instance/Device Mapping

| wgpu Concept | Vulkan Equivalent |
|--------------|-------------------|
| `Instance` | `VkInstance` |
| `Adapter` | `VkPhysicalDevice` |
| `Device` | `VkDevice` |
| `Queue` | `VkQueue` |
| `Buffer` | `VkBuffer` + `VkDeviceMemory` |
| `Texture` | `VkImage` + `VkDeviceMemory` |
| `BindGroup` | `VkDescriptorSet` |
| `RenderPipeline` | `VkPipeline` (graphics) |
| `ComputePipeline` | `VkPipeline` (compute) |

**Accessing Raw Vulkan Handles** (unsafe):
```rust
#[cfg(feature = "vulkan")]
pub mod vulkan_interop {
    use wgpu::hal::api::Vulkan;
    
    pub unsafe fn get_vulkan_device(device: &wgpu::Device) -> Option<ash::Device> {
        device.as_hal::<Vulkan, _, _>(|hal_device| {
            hal_device.map(|d| d.raw_device().clone())
        })
    }
    
    pub unsafe fn get_vulkan_physical_device(
        adapter: &wgpu::Adapter
    ) -> Option<ash::vk::PhysicalDevice> {
        adapter.as_hal::<Vulkan, _, _>(|hal_adapter| {
            hal_adapter.map(|a| a.raw_physical_device())
        })
    }
}
```

#### 19.1.2 Vulkan Extension Requirements

**Instance Extensions** (required):
- `VK_KHR_surface`
- Platform surface extension (`VK_KHR_win32_surface`, `VK_KHR_xcb_surface`, etc.)

**Device Extensions** (required):
- `VK_KHR_swapchain`
- `VK_KHR_maintenance1`, `VK_KHR_maintenance2`, `VK_KHR_maintenance3`

**Device Extensions** (optional, feature-dependent):
- `VK_KHR_ray_tracing_pipeline` -> `ray_tracing_pipeline`
- `VK_KHR_acceleration_structure` -> `acceleration_structure`
- `VK_KHR_ray_query` -> `ray_query`
- `VK_EXT_mesh_shader` -> `mesh_shader` (future)
- `VK_KHR_16bit_storage` -> `shader_f16`
- `VK_KHR_spirv_1_4` -> Required for ray tracing

```rust
pub fn check_vulkan_ray_tracing_support(adapter: &wgpu::Adapter) -> bool {
    let features = adapter.features();
    features.contains(wgpu::Features::RAY_QUERY)
        || features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE)
}
```

#### 19.1.3 Vulkan-Specific Features

```rust
pub struct VulkanFeatures {
    pub ray_tracing: bool,
    pub mesh_shaders: bool,
    pub descriptor_indexing: bool,
    pub timeline_semaphores: bool,
    pub buffer_device_address: bool,
}

impl VulkanFeatures {
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        let features = adapter.features();
        Self {
            ray_tracing: features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE),
            mesh_shaders: false, // Not yet in wgpu
            descriptor_indexing: features.contains(
                wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            ),
            timeline_semaphores: true, // Required by wgpu on Vulkan 1.2+
            buffer_device_address: features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE),
        }
    }
}
```

#### 19.1.4 Debugging with Validation Layers

```rust
pub fn create_vulkan_instance_with_validation() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DEBUG,
        dx12_shader_compiler: wgpu::Dx12Compiler::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::default(),
    })
}
```

**Validation Layer Environment Variables**:
```bash
# Enable all validation
export VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT

# Enable GPU-assisted validation
export VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT

# Log validation messages to file
export VK_LAYER_LOG_FILENAME=/tmp/vulkan_validation.log
```

### 19.2 Metal Backend

#### 19.2.1 Metal Device Selection

On macOS and iOS, Metal device selection is straightforward:

```rust
pub fn select_metal_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    // Metal typically has only one adapter (the GPU)
    // On Mac Pro with multiple GPUs, all are exposed
    instance.enumerate_adapters(wgpu::Backends::METAL)
        .into_iter()
        .find(|adapter| {
            let info = adapter.get_info();
            // Prefer discrete GPU over integrated
            info.device_type == wgpu::DeviceType::DiscreteGpu
        })
        .or_else(|| {
            instance.enumerate_adapters(wgpu::Backends::METAL)
                .into_iter()
                .next()
        })
}
```

#### 19.2.2 Metal Feature Sets

Metal features map to GPU families:

| GPU Family | Features |
|------------|----------|
| Apple 1-3 | Basic compute, limited textures |
| Apple 4-5 | Tile shaders, imageblocks |
| Apple 6+ | Ray tracing |
| Apple 7+ | Mesh shaders |
| Mac 1 | Discrete GPUs (AMD) |
| Mac 2 | Apple Silicon |

```rust
pub struct MetalCapabilities {
    pub supports_ray_tracing: bool,
    pub supports_mesh_shaders: bool,
    pub max_buffer_length: u64,
    pub apple_gpu_family: u32,
}

impl MetalCapabilities {
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        let features = adapter.features();
        let limits = adapter.limits();
        let info = adapter.get_info();
        
        // Detect Apple GPU family from device name
        let apple_family = if info.name.contains("Apple") {
            if info.name.contains("M3") || info.name.contains("A17") { 7 }
            else if info.name.contains("M2") || info.name.contains("A16") { 6 }
            else if info.name.contains("M1") || info.name.contains("A14") || info.name.contains("A15") { 5 }
            else { 4 }
        } else { 0 }; // AMD/Intel GPU
        
        Self {
            supports_ray_tracing: apple_family >= 6 && features.contains(wgpu::Features::RAY_QUERY),
            supports_mesh_shaders: apple_family >= 7,
            max_buffer_length: limits.max_buffer_size,
            apple_gpu_family: apple_family,
        }
    }
}
```

#### 19.2.3 Metal-Specific Considerations

**Memory Model**:
- Metal uses unified memory on Apple Silicon
- No explicit staging buffers needed for Apple Silicon
- AMD discrete GPUs still need staging

```rust
pub fn should_use_staging_buffer(adapter: &wgpu::Adapter) -> bool {
    let info = adapter.get_info();
    // Apple Silicon has unified memory
    if info.name.contains("Apple") { return false; }
    // AMD/Intel discrete needs staging
    true
}
```

**Texture Compression**: ASTC is preferred on iOS/Apple Silicon; BC formats available on macOS with discrete GPU.

#### 19.2.4 Argument Buffers for Bindless

Metal's argument buffers enable bindless:

```rust
pub fn create_metal_bindless_layout(
    device: &wgpu::Device,
    max_textures: u32,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MetalBindless"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: Some(std::num::NonZeroU32::new(max_textures).unwrap()),
            },
        ],
    })
}
```

### 19.3 DX12 Backend

#### 19.3.1 DX12 Device Selection

```rust
pub fn select_dx12_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::DX12).collect();
    
    // Filter out software adapters (WARP)
    let hardware_adapters: Vec<_> = adapters.iter()
        .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
        .collect();
    
    // Prefer discrete GPU
    hardware_adapters.iter()
        .find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .or_else(|| hardware_adapters.first())
        .cloned()
        .cloned()
}
```

#### 19.3.2 DX12 Feature Levels

| Feature Level | Shader Model | Key Features |
|---------------|--------------|--------------|
| 11_0 | 5.0 | Basic compute |
| 11_1 | 5.1 | UAV at every stage |
| 12_0 | 6.0 | Wave intrinsics |
| 12_1 | 6.3 | Ray tracing |
| 12_2 | 6.5 | Mesh shaders |

```rust
pub struct DX12Capabilities {
    pub feature_level: u32, // 110, 111, 120, 121, 122
    pub shader_model: f32,
    pub ray_tracing_tier: u32,
    pub mesh_shader_tier: u32,
}

impl DX12Capabilities {
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        let features = adapter.features();
        
        let (feature_level, shader_model) = if features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE) {
            (121, 6.5)
        } else if features.contains(wgpu::Features::SHADER_F16) {
            (120, 6.0)
        } else {
            (111, 5.1)
        };
        
        Self {
            feature_level,
            shader_model,
            ray_tracing_tier: if features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE) { 1 } else { 0 },
            mesh_shader_tier: 0, // Not yet in wgpu
        }
    }
}
```

#### 19.3.3 Root Signature Mapping

wgpu's bind group layouts map to DX12 root signatures:

| wgpu Concept | DX12 Equivalent |
|--------------|-----------------|
| `BindGroupLayout` | Root signature |
| `BindGroup` | Descriptor table |
| Uniform buffer | CBV |
| Storage buffer | UAV |
| Texture | SRV |
| Sampler | Sampler |

```rust
pub fn optimize_bind_group_for_dx12(entries: &mut [wgpu::BindGroupLayoutEntry]) {
    // DX12 tip: Put frequently changed bindings in lower indices
    entries.sort_by_key(|e| {
        match e.ty {
            wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, .. } => 0,
            wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, .. } => 1,
            wgpu::BindingType::Texture { .. } => 2,
            wgpu::BindingType::Sampler { .. } => 3,
            _ => 4, // Storage last (UAVs)
        }
    });
}
```

#### 19.3.4 DX12-Specific Features

**Shader Compiler Selection**:
```rust
pub fn create_dx12_instance_with_dxc() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12,
        dx12_shader_compiler: wgpu::Dx12Compiler::Dxc {
            dxil_path: None, // Use bundled or system DXC
            dxc_path: None,
        },
        ..Default::default()
    })
}

pub fn create_dx12_instance_with_fxc() -> wgpu::Instance {
    // FXC is older but more compatible
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12,
        dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
        ..Default::default()
    })
}
```

### 19.4 WebGPU Backend

#### 19.4.1 Browser Compatibility

| Browser | WebGPU Status |
|---------|---------------|
| Chrome 113+ | Stable |
| Edge 113+ | Stable |
| Firefox | Behind flag |
| Safari 17+ | Stable |

```rust
#[cfg(target_arch = "wasm32")]
pub async fn request_webgpu_adapter() -> Option<wgpu::Adapter> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });
    
    instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }).await
}
```

#### 19.4.2 WebGPU Spec Conformance

WebGPU has stricter validation than native backends:

```rust
pub fn validate_for_webgpu(buffer_size: u64) -> Result<(), &'static str> {
    // WebGPU requires buffer sizes to be multiples of 4
    if buffer_size % 4 != 0 {
        return Err("Buffer size must be multiple of 4 for WebGPU");
    }
    
    const WEBGPU_MAX_BUFFER_SIZE: u64 = 256 * 1024 * 1024; // 256MB typical
    if buffer_size > WEBGPU_MAX_BUFFER_SIZE {
        return Err("Buffer too large for WebGPU");
    }
    Ok(())
}
```

#### 19.4.3 Web-Specific Limitations

```rust
pub struct WebGPULimitations {
    pub max_texture_dimension_2d: u32,        // Often 8192 vs 16384 native
    pub max_buffer_size: u64,                 // 256MB typical
    pub max_storage_buffer_binding_size: u64, // 128MB typical
    pub max_compute_workgroup_size_x: u32,    // 256
    pub max_compute_workgroups_per_dimension: u32, // 65535
}

impl WebGPULimitations {
    pub fn default_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_webgl2_defaults()
            .using_resolution(wgpu::Limits::default())
    }
}
```

#### 19.4.4 WASM Integration

```rust
#[cfg(target_arch = "wasm32")]
pub mod wasm {
    use wasm_bindgen::prelude::*;
    use web_sys::HtmlCanvasElement;
    
    pub fn get_canvas(id: &str) -> Result<HtmlCanvasElement, JsValue> {
        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();
        let canvas = document.get_element_by_id(id).unwrap();
        canvas.dyn_into::<HtmlCanvasElement>()
    }
    
    pub fn create_surface_from_canvas(
        instance: &wgpu::Instance,
        canvas: HtmlCanvasElement,
    ) -> wgpu::Surface<'static> {
        instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .expect("Failed to create surface from canvas")
    }
    
    pub fn get_device_pixel_ratio() -> f64 {
        web_sys::window().map(|w| w.device_pixel_ratio()).unwrap_or(1.0)
    }
}
```

### 19.5 OpenGL Backend (Fallback)

#### 19.5.1 OpenGL ES / WebGL Fallback

```rust
pub fn create_gles_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::GL,
        gles_minor_version: wgpu::Gles3MinorVersion::Version2, // OpenGL ES 3.2
        ..Default::default()
    })
}
```

#### 19.5.2 Feature Limitations

| Feature | OpenGL ES 3.2 | WebGL 2 | Native |
|---------|---------------|---------|--------|
| Compute | Yes | Limited | Yes |
| Storage buffers | Yes | No | Yes |
| Storage textures | Yes | Limited | Yes |
| Ray tracing | No | No | Yes |
| Bindless | No | No | Yes |

```rust
pub fn check_gles_compute_support(adapter: &wgpu::Adapter) -> bool {
    let limits = adapter.limits();
    limits.max_compute_workgroup_size_x > 0
}
```

#### 19.5.3 Performance Considerations

```rust
pub struct GLESOptimizations {
    pub prefer_instanced_rendering: bool,
    pub use_uniform_buffers: bool,       // vs uniform locations
    pub batch_state_changes: bool,
    pub minimize_texture_switches: bool,
}

impl GLESOptimizations {
    pub fn for_mobile() -> Self {
        Self {
            prefer_instanced_rendering: true,
            use_uniform_buffers: true,
            batch_state_changes: true,
            minimize_texture_switches: true,
        }
    }
}
```

## Chapter 20: Feature Detection & Capability Abstraction

### 20.1 Feature Flags

#### 20.1.1 Core Features (Always Available)

These features are guaranteed on all wgpu backends:
- Vertex buffers
- Index buffers (u16, u32)
- Uniform buffers
- 2D textures
- Samplers
- Render passes
- Basic compute shaders
- MSAA (up to 4x)

#### 20.1.2 Optional Features (Adapter-Dependent)

```rust
pub struct OptionalFeatures {
    // Textures
    pub texture_compression_bc: bool,
    pub texture_compression_etc2: bool,
    pub texture_compression_astc: bool,
    
    // Buffers
    pub multi_draw_indirect: bool,
    pub multi_draw_indirect_count: bool,
    pub push_constants: bool,
    
    // Shaders
    pub shader_f16: bool,
    pub shader_f64: bool,
    
    // Bindless
    pub texture_binding_array: bool,
    pub sampled_texture_and_storage_buffer_array_non_uniform_indexing: bool,
    
    // Ray tracing
    pub ray_query: bool,
    pub ray_tracing_acceleration_structure: bool,
    
    // Advanced
    pub conservative_rasterization: bool,
    pub depth_clip_control: bool,
    
    // Queries
    pub timestamp_query: bool,
    pub pipeline_statistics_query: bool,
}

pub fn check_optional_features(adapter: &wgpu::Adapter) -> OptionalFeatures {
    let features = adapter.features();
    OptionalFeatures {
        texture_compression_bc: features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC),
        texture_compression_etc2: features.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2),
        texture_compression_astc: features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC),
        multi_draw_indirect: features.contains(wgpu::Features::MULTI_DRAW_INDIRECT),
        multi_draw_indirect_count: features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT),
        push_constants: features.contains(wgpu::Features::PUSH_CONSTANTS),
        shader_f16: features.contains(wgpu::Features::SHADER_F16),
        shader_f64: features.contains(wgpu::Features::SHADER_F64),
        texture_binding_array: features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY),
        sampled_texture_and_storage_buffer_array_non_uniform_indexing: 
            features.contains(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING),
        ray_query: features.contains(wgpu::Features::RAY_QUERY),
        ray_tracing_acceleration_structure: 
            features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE),
        conservative_rasterization: features.contains(wgpu::Features::CONSERVATIVE_RASTERIZATION),
        depth_clip_control: features.contains(wgpu::Features::DEPTH_CLIP_CONTROL),
        timestamp_query: features.contains(wgpu::Features::TIMESTAMP_QUERY),
        pipeline_statistics_query: features.contains(wgpu::Features::PIPELINE_STATISTICS_QUERY),
    }
}
```

#### 20.1.3 Experimental Features (Unstable API)

```rust
pub fn experimental_features() -> &'static [wgpu::Features] {
    &[wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE]
}

pub fn request_experimental_device(
    adapter: &wgpu::Adapter,
    required: wgpu::Features,
) -> Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError> {
    let available = adapter.features();
    let requested = required & available;
    
    if requested != required {
        let missing = required - available;
        eprintln!("Warning: Missing experimental features: {:?}", missing);
    }
    
    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("ExperimentalDevice"),
            required_features: requested,
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
}
```

#### 20.1.4 Feature Dependency Chains

```rust
pub struct FeatureDependencies;

impl FeatureDependencies {
    pub fn dependencies_for(feature: wgpu::Features) -> wgpu::Features {
        match feature {
            f if f.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE) => {
                wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE
            }
            f if f.contains(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING) => {
                wgpu::Features::TEXTURE_BINDING_ARRAY
                    | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            }
            _ => feature,
        }
    }
    
    pub fn expand_features(requested: wgpu::Features) -> wgpu::Features {
        let mut expanded = requested;
        for i in 0..64 {
            let bit = wgpu::Features::from_bits_truncate(1 << i);
            if requested.contains(bit) {
                expanded |= Self::dependencies_for(bit);
            }
        }
        expanded
    }
}
```

### 20.2 Limits

#### 20.2.1 Key Limits

```rust
pub fn inspect_limits(adapter: &wgpu::Adapter) {
    let limits = adapter.limits();
    
    println!("=== Texture Limits ===");
    println!("  max_texture_dimension_1d: {}", limits.max_texture_dimension_1d);
    println!("  max_texture_dimension_2d: {}", limits.max_texture_dimension_2d);
    println!("  max_texture_dimension_3d: {}", limits.max_texture_dimension_3d);
    println!("  max_texture_array_layers: {}", limits.max_texture_array_layers);
    
    println!("\n=== Buffer Limits ===");
    println!("  max_buffer_size: {} bytes", limits.max_buffer_size);
    println!("  max_uniform_buffer_binding_size: {} bytes", limits.max_uniform_buffer_binding_size);
    println!("  max_storage_buffer_binding_size: {} bytes", limits.max_storage_buffer_binding_size);
    
    println!("\n=== Bind Group Limits ===");
    println!("  max_bind_groups: {}", limits.max_bind_groups);
    println!("  max_bindings_per_bind_group: {}", limits.max_bindings_per_bind_group);
    println!("  max_sampled_textures_per_shader_stage: {}", limits.max_sampled_textures_per_shader_stage);
    println!("  max_samplers_per_shader_stage: {}", limits.max_samplers_per_shader_stage);
    println!("  max_storage_buffers_per_shader_stage: {}", limits.max_storage_buffers_per_shader_stage);
    
    println!("\n=== Compute Limits ===");
    println!("  max_compute_workgroup_storage_size: {} bytes", limits.max_compute_workgroup_storage_size);
    println!("  max_compute_invocations_per_workgroup: {}", limits.max_compute_invocations_per_workgroup);
    println!("  max_compute_workgroup_size_x: {}", limits.max_compute_workgroup_size_x);
    println!("  max_compute_workgroups_per_dimension: {}", limits.max_compute_workgroups_per_dimension);
    
    println!("\n=== Other Limits ===");
    println!("  max_push_constant_size: {} bytes", limits.max_push_constant_size);
    println!("  max_color_attachments: {}", limits.max_color_attachments);
}
```

#### 20.2.2 Limit Negotiation

```rust
pub struct LimitRequirements {
    pub min_uniform_buffer_size: u32,
    pub min_storage_buffer_size: u64,
    pub min_texture_size: u32,
    pub min_bind_groups: u32,
    pub requires_compute: bool,
}

impl LimitRequirements {
    pub fn for_trinity() -> Self {
        Self {
            min_uniform_buffer_size: 64 * 1024,        // 64KB
            min_storage_buffer_size: 128 * 1024 * 1024, // 128MB
            min_texture_size: 8192,
            min_bind_groups: 4,
            requires_compute: true,
        }
    }
    
    pub fn can_satisfy(&self, limits: &wgpu::Limits) -> bool {
        limits.max_uniform_buffer_binding_size >= self.min_uniform_buffer_size
            && limits.max_storage_buffer_binding_size >= self.min_storage_buffer_size
            && limits.max_texture_dimension_2d >= self.min_texture_size
            && limits.max_bind_groups >= self.min_bind_groups
            && (!self.requires_compute || limits.max_compute_workgroup_size_x > 0)
    }
}
```

### 20.3 TRINITY Capability System

#### 20.3.1 Capability Tiers

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CapabilityTier {
    Minimal,   // WebGL 2 / OpenGL ES 3.0
    Standard,  // Desktop OpenGL 4.5 / DX11 / Metal 2
    Advanced,  // DX12 / Vulkan 1.2 / Metal 3
    Full,      // All features including RT
}

impl CapabilityTier {
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        let features = adapter.features();
        let limits = adapter.limits();
        
        // Check for Full tier (ray tracing)
        if features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE)
            && features.contains(wgpu::Features::RAY_QUERY) {
            return CapabilityTier::Full;
        }
        
        // Check for Advanced tier (bindless, compute)
        if features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY)
            && features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
            && limits.max_compute_workgroup_storage_size >= 32768 {
            return CapabilityTier::Advanced;
        }
        
        // Check for Standard tier
        if limits.max_texture_dimension_2d >= 8192
            && limits.max_compute_workgroup_size_x >= 256
            && limits.max_storage_buffer_binding_size >= 128 * 1024 * 1024 {
            return CapabilityTier::Standard;
        }
        
        CapabilityTier::Minimal
    }
    
    pub fn required_features(&self) -> wgpu::Features {
        match self {
            CapabilityTier::Minimal | CapabilityTier::Standard => wgpu::Features::empty(),
            CapabilityTier::Advanced => {
                wgpu::Features::TEXTURE_BINDING_ARRAY
                    | wgpu::Features::MULTI_DRAW_INDIRECT
                    | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
            }
            CapabilityTier::Full => {
                wgpu::Features::TEXTURE_BINDING_ARRAY
                    | wgpu::Features::MULTI_DRAW_INDIRECT
                    | wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
                    | wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE
                    | wgpu::Features::RAY_QUERY
            }
        }
    }
}
```

#### 20.3.2 Feature Requirements Per Render Path

```rust
pub enum RenderPath {
    Forward,
    Deferred,
    DeferredBindless,
    DeferredRayTraced,
}

impl RenderPath {
    pub fn required_tier(&self) -> CapabilityTier {
        match self {
            RenderPath::Forward => CapabilityTier::Minimal,
            RenderPath::Deferred => CapabilityTier::Standard,
            RenderPath::DeferredBindless => CapabilityTier::Advanced,
            RenderPath::DeferredRayTraced => CapabilityTier::Full,
        }
    }
    
    pub fn select_for_tier(tier: CapabilityTier) -> Self {
        match tier {
            CapabilityTier::Minimal => RenderPath::Forward,
            CapabilityTier::Standard => RenderPath::Deferred,
            CapabilityTier::Advanced => RenderPath::DeferredBindless,
            CapabilityTier::Full => RenderPath::DeferredRayTraced,
        }
    }
}
```

#### 20.3.3 Automatic Fallback Selection

```rust
pub struct CapabilityManager {
    tier: CapabilityTier,
    render_path: RenderPath,
    features: wgpu::Features,
    limits: wgpu::Limits,
}

impl CapabilityManager {
    pub fn new(adapter: &wgpu::Adapter) -> Self {
        let tier = CapabilityTier::from_adapter(adapter);
        let render_path = RenderPath::select_for_tier(tier);
        Self {
            tier,
            render_path,
            features: adapter.features(),
            limits: adapter.limits(),
        }
    }
    
    pub fn can_use_feature(&self, feature: wgpu::Features) -> bool {
        self.features.contains(feature)
    }
    
    pub fn select_texture_compression(&self) -> TextureCompression {
        if self.features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
            TextureCompression::BC
        } else if self.features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC) {
            TextureCompression::ASTC
        } else if self.features.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2) {
            TextureCompression::ETC2
        } else {
            TextureCompression::None
        }
    }
    
    pub fn max_bindless_textures(&self) -> u32 {
        if self.can_use_feature(wgpu::Features::TEXTURE_BINDING_ARRAY) {
            self.limits.max_sampled_textures_per_shader_stage.min(4096)
        } else {
            16 // Standard texture slots
        }
    }
    
    pub fn create_device_descriptor(&self) -> wgpu::DeviceDescriptor<'static> {
        wgpu::DeviceDescriptor {
            label: Some("TrinityDevice"),
            required_features: self.render_path.required_features() & self.features,
            required_limits: self.limits.clone(),
            memory_hints: wgpu::MemoryHints::Performance,
        }
    }
}

pub enum TextureCompression { None, BC, ASTC, ETC2 }
```

#### 20.3.4 Runtime Capability Queries

```rust
impl CapabilityManager {
    pub fn supports_ray_tracing(&self) -> bool { self.tier >= CapabilityTier::Full }
    pub fn supports_bindless(&self) -> bool { self.tier >= CapabilityTier::Advanced }
    pub fn supports_gpu_culling(&self) -> bool {
        self.can_use_feature(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
    }
    pub fn supports_timestamp_queries(&self) -> bool {
        self.can_use_feature(wgpu::Features::TIMESTAMP_QUERY)
    }
    pub fn supports_mesh_shaders(&self) -> bool { false } // Not yet in wgpu
    pub fn max_msaa_samples(&self) -> u32 { 4 } // Conservative default
    
    pub fn report(&self) -> CapabilityReport {
        CapabilityReport {
            tier: self.tier,
            render_path: self.render_path,
            ray_tracing: self.supports_ray_tracing(),
            bindless: self.supports_bindless(),
            gpu_culling: self.supports_gpu_culling(),
            mesh_shaders: self.supports_mesh_shaders(),
            max_texture_size: self.limits.max_texture_dimension_2d,
            max_buffer_size: self.limits.max_buffer_size,
            texture_compression: self.select_texture_compression(),
        }
    }
}

pub struct CapabilityReport {
    pub tier: CapabilityTier,
    pub render_path: RenderPath,
    pub ray_tracing: bool,
    pub bindless: bool,
    pub gpu_culling: bool,
    pub mesh_shaders: bool,
    pub max_texture_size: u32,
    pub max_buffer_size: u64,
    pub texture_compression: TextureCompression,
}
```

### TRINITY Platform Support Matrix

| Feature | Vulkan | Metal | DX12 | WebGPU | OpenGL |
|---------|--------|-------|------|--------|--------|
| Core rendering | Yes | Yes | Yes | Yes | Yes |
| Compute shaders | Yes | Yes | Yes | Yes | Limited |
| Bindless textures | Yes | Yes | Yes | No | No |
| Multi-draw indirect | Yes | Yes | Yes | Yes | Limited |
| Ray query | Yes | Limited | Yes | No | No |
| RT pipeline | Yes | No | Yes | No | No |
| Mesh shaders | Future | Future | Future | No | No |
| Timestamp queries | Yes | Yes | Yes | Yes | Limited |

---

# PART XI: DEBUGGING & PROFILING

GPU debugging is fundamentally different from CPU debugging. Issues may be non-deterministic, difficult to reproduce, and visible only through corrupted output. wgpu provides built-in validation and integrates with external GPU debuggers.

## Chapter 21: Debugging

### 21.1 wgpu Debugging Features

#### 21.1.1 Validation (WGPU_VALIDATION)

wgpu includes extensive validation that catches errors at record time rather than GPU execution:

```rust
pub fn create_validated_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DEBUG,
        ..Default::default()
    })
}
```

**Validation Catches**: Invalid resource usage combinations, mismatched bind group layouts, shader binding errors, buffer/texture out-of-bounds access, missing required state before draw/dispatch, invalid pipeline state combinations.

**Environment Variables**:
```bash
export WGPU_VALIDATION=1  # Enable validation
export WGPU_DEBUG=1       # Enable debug output
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation  # Vulkan layers
```

#### 21.1.2 Debug Markers and Groups

Debug groups create hierarchical regions visible in GPU capture tools:

```rust
pub struct DebugGroup<'a> {
    encoder: &'a mut wgpu::CommandEncoder,
}

impl<'a> DebugGroup<'a> {
    pub fn new(encoder: &'a mut wgpu::CommandEncoder, label: &str) -> Self {
        encoder.push_debug_group(label);
        Self { encoder }
    }
}

impl Drop for DebugGroup<'_> {
    fn drop(&mut self) { self.encoder.pop_debug_group(); }
}
```

#### 21.1.3 Object Labels

Every wgpu resource can be labeled for easier debugging via the `label` field in descriptors.

#### 21.1.4 Error Scopes

Fine-grained error handling with `push_error_scope()` and `pop_error_scope()` for `Validation` and `OutOfMemory` filters.

### 21.2 External Debuggers

#### 21.2.1 RenderDoc Integration

Primary GPU debugger for Vulkan and DX12. Keyboard: F12 (capture), Ctrl+F12 (multi-capture), F11 (overlay).

```rust
#[cfg(feature = "renderdoc")]
pub struct RenderDocCapture {
    api: Option<RenderDoc<V141>>,
    capture_pending: bool,
}
```

#### 21.2.2 PIX for Windows

Deep DX12 debugging. Features: GPU capture, timing analysis, memory debugging, shader debugging with source correlation.

#### 21.2.3 Xcode GPU Frame Capture

macOS/iOS Metal debugging. Enable via Product > Scheme > Edit Scheme > Run > Diagnostics.

#### 21.2.4 NVIDIA Nsight Graphics

Deep NVIDIA analysis. Features: Ray tracing debugging, shader profiling, memory analysis, warp state inspection.

#### 21.2.5 AMD Radeon GPU Profiler

AMD-specific profiling. Features: Wavefront occupancy, cache analysis, barrier analysis, instruction timing.

### 21.3 TRINITY Debug System

#### 21.3.1 Debug Visualization Modes

```rust
pub enum DebugVisualization {
    None, Wireframe, Normals, Tangents, UVs, Albedo, Metallic, Roughness,
    AO, Depth, Stencil, MotionVectors, MipLevels, Overdraw, LightHeatmap,
    ShadowCascades, RTAccelerationStructure, Meshlets,
}

impl DebugVisualization {
    pub fn shader_define(&self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Normals => Some("DEBUG_NORMALS"),
            Self::Depth => Some("DEBUG_DEPTH"),
            Self::Overdraw => Some("DEBUG_OVERDRAW"),
            // ... other modes
            _ => None,
        }
    }
}
```

#### 21.3.2 Resource Inspection

```rust
pub struct ResourceInspector {
    inspection_buffer: wgpu::Buffer,
    inspection_data: Vec<u8>,
}

impl ResourceInspector {
    pub fn inspect_texture(&mut self, encoder: &mut wgpu::CommandEncoder, 
                           texture: &wgpu::Texture, mip: u32);
    pub fn get_pixel(&self, x: u32, y: u32, width: u32) -> [f32; 4];
}
```

#### 21.3.3 Pipeline State Dump

Capture and log current pipeline state including vertex buffers, index buffer, bind groups, viewport, scissor, blend constants, stencil reference.

#### 21.3.4 Frame Capture Triggers

```rust
pub struct FrameCaptureSystem {
    capture_next_frame: bool,
    capture_on_error: bool,
    capture_on_slow_frame: bool,
    slow_frame_threshold_ms: f32, // 33.3ms = 30 FPS
}
```

## Chapter 22: Profiling

GPU profiling measures execution time, resource usage, and identifies performance bottlenecks.

### 22.1 GPU Timing

#### 22.1.1 Timestamp Queries

```rust
pub struct GPUProfiler {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    timestamp_period: f32,
    regions: Vec<ProfileRegion>,
    results: Vec<ProfileResult>,
}

pub struct ProfileResult { pub name: String, pub duration_ns: f64, pub duration_ms: f64 }

impl GPUProfiler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, max_regions: u32) -> Self;
    pub fn begin_region(&mut self, pass: &mut impl TimestampWriter, name: &str) -> ProfileRegion;
    pub fn end_region(&mut self, pass: &mut impl TimestampWriter, region: ProfileRegion);
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder);
    pub async fn read_results(&mut self, device: &wgpu::Device);
}

pub trait TimestampWriter {
    fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, index: u32);
}
```

#### 22.1.2 Timer Resolution

```rust
impl GPUProfiler {
    pub fn timer_resolution_ns(&self) -> f32 { self.timestamp_period }
    pub fn is_high_resolution(&self) -> bool { self.timestamp_period < 1000.0 }
}
```

### 22.2 Pipeline Statistics

#### 22.2.1 Statistics Queries (Feature: PIPELINE_STATISTICS_QUERY)

```rust
#[repr(C)]
pub struct PipelineStatistics {
    pub vertex_shader_invocations: u64,
    pub clipper_invocations: u64,
    pub clipper_primitives_out: u64,
    pub fragment_shader_invocations: u64,
    pub compute_shader_invocations: u64,
}

impl PipelineStatistics {
    pub fn overdraw_estimate(&self, screen_pixels: u64) -> f64;
    pub fn culling_efficiency(&self) -> f64;
}
```

### 22.3 Memory Profiling

#### 22.3.1 Resource Memory Tracking

```rust
pub struct MemoryTracker {
    allocations: HashMap<ResourceId, AllocationInfo>,
    total_buffer_memory: u64,
    total_texture_memory: u64,
    peak_buffer_memory: u64,
    peak_texture_memory: u64,
}

impl MemoryTracker {
    pub fn track_buffer(&mut self, id: ResourceId, name: &str, size: u64);
    pub fn track_texture(&mut self, id: ResourceId, name: &str, desc: &wgpu::TextureDescriptor);
    pub fn untrack(&mut self, id: ResourceId);
    pub fn report(&self) -> MemoryReport;
}
```

#### 22.3.2 Memory Budget Monitoring

```rust
pub enum BudgetStatus {
    Ok { current_mb: f64, budget_mb: f64 },
    Warning { current_mb: f64, budget_mb: f64, percent_used: f64 },
    Exceeded { current_mb: f64, budget_mb: f64, overage_mb: f64 },
}
```

#### 22.3.3 Memory Leak Detection

```rust
pub struct LeakDetector {
    frame_allocations: HashMap<u64, Vec<ResourceId>>,
    leak_threshold_frames: u64,
}

pub struct LeakWarning { pub resource_id: ResourceId, pub allocated_frame: u64, pub current_frame: u64 }
```

### 22.4 TRINITY Profiling System

#### 22.4.1 Per-Pass Timing

```rust
pub struct FrameProfiler {
    gpu_profiler: GPUProfiler,
    cpu_timers: HashMap<String, std::time::Instant>,
    history: VecDeque<FrameProfileResults>,
}

impl FrameProfiler {
    pub fn begin_cpu_region(&mut self, name: &str);
    pub fn end_cpu_region(&mut self, name: &str);
    pub fn average_gpu_time(&self, region_name: &str) -> Option<f64>;
}
```

#### 22.4.2 Draw Call Statistics

```rust
pub struct DrawCallStats {
    pub draw_calls: u32, pub triangles: u64, pub vertices: u64, pub instances: u64,
    pub state_changes: u32, pub bind_group_changes: u32, pub pipeline_changes: u32,
}

impl DrawCallStats {
    pub fn efficiency_score(&self) -> f64; // Higher = better batching
}
```

#### 22.4.3 Bottleneck Analysis

```rust
pub enum Bottleneck {
    CPUBound { cpu_ms: f64, gpu_ms: f64 },
    GPUBound { cpu_ms: f64, gpu_ms: f64 },
    FragmentBound { vertex_time_ms: f64, fragment_time_ms: f64 },
    DrawCallBound { draw_calls: u32, avg_tris: f64 },
}

impl BottleneckAnalyzer {
    pub fn analyze(&self) -> Vec<Bottleneck>;
    pub fn suggest_optimizations(&self) -> Vec<&'static str>;
    // Suggestions: "GPU-driven rendering", "Batch with multi-draw", "Use instancing", etc.
}
```

### TRINITY Debug & Profiling Tool Summary

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `DebugVisualization` | Visual debugging | Inspecting normals, UVs, overdraw |
| `ResourceInspector` | Data inspection | Verifying buffer/texture contents |
| `RenderDocCapture` | Frame capture | Deep GPU debugging |
| `GPUProfiler` | GPU timing | Performance analysis |
| `MemoryTracker` | Memory monitoring | Tracking allocations |
| `LeakDetector` | Leak detection | Finding resource leaks |
| `BottleneckAnalyzer` | Performance diagnosis | Identifying bottlenecks |

---

# PART XII: TRINITY INTEGRATION

The frame graph is TRINITY's central abstraction for organizing GPU work. It provides automatic resource lifetime management, barrier placement, and execution scheduling.

## Chapter 23: Frame Graph Integration

### 23.1 Resource Declaration

#### 23.1.1 Virtual Resources

Virtual resources are declared at frame graph construction time but allocated lazily during execution.

```rust
pub struct FrameGraph {
    passes: Vec<PassNode>,
    resources: HashMap<ResourceId, ResourceNode>,
    edges: Vec<ResourceEdge>,
    execution_order: Vec<usize>,
    compiled: bool,
}

pub struct ResourceNode {
    id: ResourceId,
    name: String,
    descriptor: ResourceDescriptor,
    lifetime: ResourceLifetime,
    physical: Option<PhysicalResource>,
}

pub enum ResourceDescriptor {
    Buffer(BufferResourceDesc),
    Texture(TextureResourceDesc),
}

pub struct TextureResourceDesc {
    pub size: wgpu::Extent3d,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
    pub mip_count: u32,
    pub sample_count: u32,
}

impl FrameGraph {
    pub fn create_virtual_texture(&mut self, name: &str, desc: TextureResourceDesc) -> ResourceId;
    pub fn create_virtual_buffer(&mut self, name: &str, desc: BufferResourceDesc) -> ResourceId;
}
```

#### 23.1.2 Transient Resources

Transient resources exist only for the duration of a frame and can be aliased:

```rust
pub enum ResourceLifetime {
    Transient,   // Lives only within frame, can be aliased
    Persistent,  // Lives across frames
    External,    // Owned outside frame graph
    Imported,    // Imported from previous frame
}

pub struct TransientResourcePool {
    texture_pool: Vec<PooledTexture>,
    buffer_pool: Vec<PooledBuffer>,
}

impl TransientResourcePool {
    pub fn acquire_texture(&mut self, device: &wgpu::Device, desc: &TextureResourceDesc, frame: u64) -> (wgpu::Texture, wgpu::TextureView);
    pub fn release_all(&mut self);
    pub fn gc(&mut self, current_frame: u64, max_unused_frames: u64);
}
```

#### 23.1.3 External Resources

External resources are provided from outside the frame graph:

```rust
impl FrameGraph {
    pub fn import_texture(&mut self, name: &str, texture: &wgpu::Texture, view: &wgpu::TextureView, desc: TextureResourceDesc) -> ResourceId;
    pub fn import_swapchain(&mut self, surface_texture: &wgpu::SurfaceTexture, view: &wgpu::TextureView) -> ResourceId;
}
```

#### 23.1.4 Resource Aliasing

Aliasing allows multiple transient resources to share the same physical memory when their lifetimes don't overlap:

```rust
pub struct AliasingInfo { pub resource: ResourceId, pub first_use: usize, pub last_use: usize }

pub struct AliasingAnalyzer;
impl AliasingAnalyzer {
    pub fn compute_aliasing(resources: &[ResourceNode], passes: &[PassNode]) -> Vec<AliasingGroup>;
}
```

### 23.2 Pass Declaration

#### 23.2.1 Render Passes

```rust
pub struct PassNode {
    id: PassId,
    name: String,
    pass_type: PassType,
    reads: Vec<ResourceId>,
    writes: Vec<ResourceId>,
    execute: Box<dyn PassExecutor>,
}

pub enum PassType {
    Render(RenderPassConfig),
    Compute(ComputePassConfig),
    RayTracing(RTPassConfig),
    Copy,
}

pub struct RenderPassConfig {
    pub color_attachments: Vec<ColorAttachmentConfig>,
    pub depth_attachment: Option<DepthAttachmentConfig>,
    pub resolve_attachments: Vec<ResolveConfig>,
}

pub enum LoadOp { Load, Clear, DontCare }
pub enum StoreOp { Store, Discard }

impl FrameGraph {
    pub fn add_render_pass<F>(&mut self, name: &str, config: RenderPassConfig, execute: F) -> PassId
    where F: FnMut(&mut RenderPassContext) + 'static;
}
```

#### 23.2.2 Compute Passes

```rust
impl FrameGraph {
    pub fn add_compute_pass<F>(&mut self, name: &str, config: ComputePassConfig, execute: F) -> PassId
    where F: FnMut(&mut ComputePassContext) + 'static;
}
```

#### 23.2.3 Ray Tracing Passes

```rust
pub struct RTPassConfig {
    pub acceleration_structure: ResourceId,
    pub output_image: ResourceId,
    pub shader_binding_table: ResourceId,
}

impl FrameGraph {
    pub fn add_rt_pass<F>(&mut self, name: &str, config: RTPassConfig, execute: F) -> PassId;
}
```

#### 23.2.4 Copy Passes

```rust
impl FrameGraph {
    pub fn add_copy_pass(&mut self, name: &str, src: ResourceId, dst: ResourceId) -> PassId;
}
```

### 23.3 Barrier Resolution

#### 23.3.1 Automatic Barrier Placement

```rust
pub struct BarrierResolver {
    resource_states: HashMap<ResourceId, ResourceState>,
}

pub struct ResourceState {
    pub stage: PipelineStage,
    pub access: AccessFlags,
    pub layout: TextureLayout,
}

impl BarrierResolver {
    pub fn compute_barriers(&mut self, passes: &[PassNode]) -> Vec<PassBarriers>;
}
```

#### 23.3.2 Resource State Tracking

```rust
impl ResourceState {
    pub fn undefined() -> Self;
    pub fn needs_barrier_to(&self, other: &Self) -> bool;
    // Returns true for: write-after-read, read-after-write, write-after-write, layout transitions
}

pub enum TextureLayout {
    Undefined, General, ColorAttachment, DepthAttachment,
    ShaderReadOnly, CopySrc, CopyDst, Present,
}
```

#### 23.3.3 Barrier Batching

Group barriers by pipeline stage for efficiency.

#### 23.3.4 Aliasing Barriers

```rust
pub struct AliasingBarrier {
    pub before: ResourceId,
    pub after: ResourceId,
    pub physical_memory: PhysicalMemoryId,
}
```

### 23.4 Execution

#### 23.4.1 Pass Scheduling

```rust
impl FrameGraph {
    pub fn compile(&mut self) {
        // 1. Build dependency graph
        // 2. Topological sort
        // 3. Compute aliasing
        // 4. Resolve barriers
    }
    
    fn topological_sort(&self, deps: &HashMap<usize, Vec<usize>>) -> Vec<usize>;
}
```

#### 23.4.2 Async Compute Overlap

```rust
pub struct AsyncComputeScheduler {
    graphics_timeline: Vec<PassId>,
    compute_timeline: Vec<PassId>,
    sync_points: Vec<SyncPoint>,
}

pub enum SyncDirection { GraphicsToCompute, ComputeToGraphics }
```

#### 23.4.3 Resource Lifetime Management

```rust
pub struct ResourceLifetimeManager {
    pool: TransientResourcePool,
    allocations: HashMap<ResourceId, PhysicalResource>,
    frame_number: u64,
}

impl ResourceLifetimeManager {
    pub fn begin_frame(&mut self);
    pub fn allocate(&mut self, device: &wgpu::Device, resource: &ResourceNode) -> PhysicalResource;
    pub fn end_frame(&mut self); // GC unused resources
}
```

#### 23.4.4 Frame-to-Frame Resource Recycling

```rust
pub struct ResourceRecycler {
    recycled_textures: HashMap<TextureKey, Vec<wgpu::Texture>>,
    recycled_buffers: HashMap<BufferKey, Vec<wgpu::Buffer>>,
}

impl ResourceRecycler {
    pub fn recycle_texture(&mut self, texture: wgpu::Texture);
    pub fn acquire_texture(&mut self, device: &wgpu::Device, key: &TextureKey) -> wgpu::Texture;
}
```

## Chapter 24: Python Bridge

TRINITY's Python bridge enables scripting and rapid prototyping through PyO3 bindings to the Rust renderer.

### 24.1 PyO3 Binding Layer

#### 24.1.1 Type Marshalling

```rust
#[pyclass]
#[derive(Clone)]
pub struct PyTextureDescriptor {
    #[pyo3(get, set)] pub width: u32,
    #[pyo3(get, set)] pub height: u32,
    #[pyo3(get, set)] pub format: String,
    #[pyo3(get, set)] pub usage: Vec<String>,
}

#[pymethods]
impl PyTextureDescriptor {
    #[new]
    fn new(width: u32, height: u32, format: &str) -> Self;
    fn to_wgpu(&self) -> TextureResourceDesc;
}

fn parse_format(s: &str) -> wgpu::TextureFormat {
    match s {
        "RGBA8" => wgpu::TextureFormat::Rgba8Unorm,
        "RGBA16F" => wgpu::TextureFormat::Rgba16Float,
        "DEPTH32F" => wgpu::TextureFormat::Depth32Float,
        // ... other formats
    }
}
```

#### 24.1.2 Handle Management

```rust
#[pyclass]
pub struct PyRenderer { inner: Arc<Mutex<TrinityRenderer>> }

#[pymethods]
impl PyRenderer {
    #[new] fn new() -> PyResult<Self>;
    fn create_texture(&self, desc: &PyTextureDescriptor) -> PyResult<PyResourceHandle>;
    fn destroy_resource(&self, handle: &PyResourceHandle) -> PyResult<()>;
}
```

#### 24.1.3 Callback Patterns

```rust
#[pyclass]
pub struct PyRenderCallback { callback: PyObject }

impl PyRenderer {
    fn render_frame_with_callback(&self, py: Python, callback: &PyRenderCallback) -> PyResult<()>;
}
```

#### 24.1.4 Error Propagation

```rust
pub fn wgpu_error_to_py(error: wgpu::Error) -> PyErr {
    match error {
        wgpu::Error::OutOfMemory { .. } => PyErr::new::<PyRuntimeError, _>("GPU out of memory"),
        wgpu::Error::Validation { description, .. } => PyErr::new::<PyValueError, _>(format!("Validation: {}", description)),
        wgpu::Error::Internal { description, .. } => PyErr::new::<PyRuntimeError, _>(format!("Internal: {}", description)),
    }
}
```

### 24.2 Resource Descriptors

#### 24.2.1 Python-Side Descriptors

```python
# trinity/descriptors.py
from dataclasses import dataclass
from enum import Enum

class TextureFormat(Enum):
    RGBA8 = "RGBA8"
    RGBA16F = "RGBA16F"
    DEPTH32F = "DEPTH32F"

class TextureUsage(Enum):
    RENDER_ATTACHMENT = "RENDER_ATTACHMENT"
    TEXTURE_BINDING = "TEXTURE_BINDING"
    STORAGE_BINDING = "STORAGE_BINDING"

@dataclass
class TextureDesc:
    width: int
    height: int
    format: TextureFormat = TextureFormat.RGBA8
    usage: list = None
    
    def to_native(self):
        from trinity._native import PyTextureDescriptor
        desc = PyTextureDescriptor(self.width, self.height, self.format.value)
        desc.usage = [u.value for u in (self.usage or [TextureUsage.RENDER_ATTACHMENT])]
        return desc
```

#### 24.2.2 Descriptor Validation

```rust
impl PyTextureDescriptor {
    fn validate(&self) -> PyResult<()> {
        if self.width == 0 || self.height == 0 { return Err(PyValueError::new_err("Dimensions must be non-zero")); }
        if self.width > 16384 || self.height > 16384 { return Err(PyValueError::new_err("Exceeds maximum")); }
        Ok(())
    }
}
```

#### 24.2.3 Descriptor Caching

```rust
pub struct DescriptorCache {
    texture_cache: HashMap<u64, wgpu::Texture>,
    pipeline_cache: HashMap<u64, wgpu::RenderPipeline>,
}
```

### 24.3 Command Recording

#### 24.3.1 Python Command Builder

```python
class RenderPassBuilder:
    def __init__(self, name: str):
        self.name = name
        self.color_attachments = []
        self.commands = []
    
    def add_color_attachment(self, texture, load_op="clear", clear_color=(0,0,0,1)): return self
    def set_pipeline(self, pipeline): self.commands.append(("set_pipeline", pipeline)); return self
    def set_bind_group(self, index, bind_group): return self
    def draw(self, vertex_count, instance_count=1): return self
    def draw_indexed(self, index_count, instance_count=1): return self

class ComputePassBuilder:
    def set_pipeline(self, pipeline): return self
    def dispatch(self, x, y=1, z=1): return self
```

#### 24.3.2 Deferred Execution

```rust
pub enum RecordedCommand {
    BeginRenderPass(RenderPassConfig),
    EndRenderPass,
    SetPipeline(PipelineId),
    SetBindGroup(u32, BindGroupId, Vec<u32>),
    Draw(u32, u32, u32, u32),
    DrawIndexed(u32, u32, u32, i32, u32),
    Dispatch(u32, u32, u32),
}

impl PyCommandList {
    pub fn execute(&self, encoder: &mut wgpu::CommandEncoder, resources: &ResourceManager);
}
```

#### 24.3.3 Command Batching

Remove redundant state changes and batch compatible draw calls.

#### 24.3.4 Error Handling

```python
# trinity/errors.py
class TrinityError(Exception): pass
class ValidationError(TrinityError): pass
class OutOfMemoryError(TrinityError): pass
class DeviceLostError(TrinityError): pass
```

### Complete Python API Example

```python
from trinity import Renderer, TextureDesc, TextureFormat, TextureUsage
from trinity import RenderPassBuilder, ComputePassBuilder

def main():
    renderer = Renderer()
    
    color_target = renderer.create_texture(TextureDesc(
        width=1920, height=1080, format=TextureFormat.RGBA8_SRGB,
        usage=[TextureUsage.RENDER_ATTACHMENT, TextureUsage.TEXTURE_BINDING]
    ))
    
    while renderer.window_open():
        frame = renderer.begin_frame()
        
        main_pass = RenderPassBuilder("Main")
        main_pass.add_color_attachment(color_target, clear_color=(0.1, 0.1, 0.1, 1))
        main_pass.set_pipeline(pipeline)
        for obj in scene.objects:
            main_pass.draw_mesh(obj.mesh, obj.transform)
        frame.add_pass(main_pass)
        
        renderer.execute_frame(frame)
        renderer.present()
```

### TRINITY Integration Component Summary

| Component | Rust Module | Python Module | Purpose |
|-----------|-------------|---------------|---------|
| Frame Graph | `frame_graph` | `trinity.graph` | Pass scheduling & resources |
| Resource Pool | `resource_pool` | Internal | Transient allocation |
| Barrier System | `barriers` | Internal | Automatic sync |
| Command Builder | `commands` | `trinity.commands` | GPU command recording |
| Type Bindings | `pyo3_bindings` | `trinity._native` | Rust-Python bridge |
| Error Handling | `errors` | `trinity.errors` | Cross-language errors |

---

# APPENDICES

## Appendix A: wgpu Feature Matrix

| Feature | Vulkan | Metal | DX12 | WebGPU | Status |
|---------|--------|-------|------|--------|--------|
| Core | ✓ | ✓ | ✓ | ✓ | Stable |
| Compute | ✓ | ✓ | ✓ | ✓ | Stable |
| Ray Query | ✓ | ⚠️ | ✓ | ✗ | Stable |
| RT Pipeline | ✓ | ✗ | ✓ | ✗ | Experimental |
| Mesh Shaders | ✗ | ✗ | ✗ | ✗ | Future |

## Appendix B: Glossary

- **BLAS**: Bottom-Level Acceleration Structure
- **TLAS**: Top-Level Acceleration Structure
- **SBT**: Shader Binding Table
- **PSO**: Pipeline State Object
- **BVH**: Bounding Volume Hierarchy
- **MSAA**: Multisample Anti-Aliasing
- **HiZ**: Hierarchical Z-buffer
- **LOD**: Level of Detail

## Appendix C: GAPSET Cross-References

| GAPSET | Coverage |
|--------|----------|
| GAPSET_1 (Frame Graph) | Part XII, Chapter 23 |
| GAPSET_6 (GI/Reflections) | Part VII (RT), Part VIII (Compute) |
| GAPSET_9 (Ray Tracing) | Part VII (Chapters 11-14) |
| GAPSET_12 (RHI) | Parts I-VI |

---

*End of MASTER.md — Pass 1/13*

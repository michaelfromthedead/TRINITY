# WGPU_PART_IV_RENDER_PIPELINE.md — Render Pipeline

> **TOC Reference**: Part IV, Chapters 6-7
> **Purpose**: Complete specification of wgpu graphics pipeline and render passes for TRINITY
> **Generated**: 2026-05-27

---

# Chapter 6: Graphics Pipeline

## 6.1 Pipeline Creation

### 6.1.1 Render Pipeline Descriptor

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

### 6.1.2 Pipeline Layout Association

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

### 6.1.3 Vertex State Configuration

```rust
VertexState {
    module: &shader_module,
    entry_point: "vs_main",
    compilation_options: PipelineCompilationOptions {
        constants: &HashMap::new(),
        zero_initialize_workgroup_memory: true,
    },
    buffers: &[
        // Buffer 0: Vertices
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
        // Buffer 1: Instance data
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

### 6.1.4 Primitive State

```rust
PrimitiveState {
    topology: PrimitiveTopology::TriangleList,  // How vertices form primitives
    strip_index_format: None,                    // For strip topologies with index buffer
    front_face: FrontFace::Ccw,                  // Counter-clockwise = front
    cull_mode: Some(Face::Back),                 // Cull back faces
    unclipped_depth: false,                      // Clip depth to [0, 1]
    polygon_mode: PolygonMode::Fill,             // Fill triangles (or Line/Point)
    conservative: false,                         // Conservative rasterization
}
```

### 6.1.5 Depth/Stencil State

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
    bias: DepthBiasState {
        constant: 0,
        slope_scale: 0.0,
        clamp: 0.0,
    },
}
```

### 6.1.6 Multisample State

```rust
MultisampleState {
    count: 4,                        // Sample count (1, 2, 4, 8, 16)
    mask: !0,                        // Sample mask (all samples enabled)
    alpha_to_coverage_enabled: false, // Use alpha for coverage
}
```

### 6.1.7 Fragment State and Color Targets

```rust
FragmentState {
    module: &fragment_shader,
    entry_point: "fs_main",
    compilation_options: Default::default(),
    targets: &[
        // MRT: Multiple render targets
        Some(ColorTargetState {
            format: TextureFormat::Rgba8UnormSrgb,  // Albedo
            blend: None,
            write_mask: ColorWrites::ALL,
        }),
        Some(ColorTargetState {
            format: TextureFormat::Rgba16Float,     // Normal + roughness
            blend: None,
            write_mask: ColorWrites::ALL,
        }),
        Some(ColorTargetState {
            format: TextureFormat::Rgba8Unorm,      // Metallic + AO
            blend: None,
            write_mask: ColorWrites::ALL,
        }),
    ],
}
```

### 6.1.8 Pipeline Caching

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
        if let Some(pipeline) = self.pipelines.get(key) {
            return pipeline.clone();
        }
        
        let pipeline = self.create_pipeline(key);
        let pipeline = Arc::new(pipeline);
        self.pipelines.insert(key.clone(), pipeline.clone());
        pipeline
    }
    
    pub fn warm_cache(&mut self, common_keys: &[PipelineKey]) {
        for key in common_keys {
            self.get_or_create(key);
        }
    }
}
```

---

## 6.2 Vertex Input

### 6.2.1 Vertex Buffer Layouts

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

pub enum VertexStepMode {
    Vertex,    // Advance per vertex
    Instance,  // Advance per instance
}
```

### 6.2.2 Vertex Attribute Formats

| Format | Size | Components | Type |
|--------|------|------------|------|
| `Uint8x2` | 2 | 2 | u32 |
| `Uint8x4` | 4 | 4 | u32 |
| `Sint8x2` | 2 | 2 | i32 |
| `Sint8x4` | 4 | 4 | i32 |
| `Unorm8x2` | 2 | 2 | f32 [0, 1] |
| `Unorm8x4` | 4 | 4 | f32 [0, 1] |
| `Snorm8x2` | 2 | 2 | f32 [-1, 1] |
| `Snorm8x4` | 4 | 4 | f32 [-1, 1] |
| `Uint16x2` | 4 | 2 | u32 |
| `Uint16x4` | 8 | 4 | u32 |
| `Sint16x2` | 4 | 2 | i32 |
| `Sint16x4` | 8 | 4 | i32 |
| `Unorm16x2` | 4 | 2 | f32 [0, 1] |
| `Unorm16x4` | 8 | 4 | f32 [0, 1] |
| `Snorm16x2` | 4 | 2 | f32 [-1, 1] |
| `Snorm16x4` | 8 | 4 | f32 [-1, 1] |
| `Float16x2` | 4 | 2 | f32 |
| `Float16x4` | 8 | 4 | f32 |
| `Float32` | 4 | 1 | f32 |
| `Float32x2` | 8 | 2 | f32 |
| `Float32x3` | 12 | 3 | f32 |
| `Float32x4` | 16 | 4 | f32 |
| `Uint32` | 4 | 1 | u32 |
| `Uint32x2` | 8 | 2 | u32 |
| `Uint32x3` | 12 | 3 | u32 |
| `Uint32x4` | 16 | 4 | u32 |
| `Sint32` | 4 | 1 | i32 |
| `Sint32x2` | 8 | 2 | i32 |
| `Sint32x3` | 12 | 3 | i32 |
| `Sint32x4` | 16 | 4 | i32 |
| `Float64` | 8 | 1 | f64 |
| `Float64x2` | 16 | 2 | f64 |
| `Float64x3` | 24 | 3 | f64 |
| `Float64x4` | 32 | 4 | f64 |

### 6.2.3 Step Modes

```rust
// Per-vertex data (geometry)
VertexBufferLayout {
    array_stride: 48,
    step_mode: VertexStepMode::Vertex,
    attributes: &[/* position, normal, uv, tangent */],
}

// Per-instance data (transforms, material IDs)
VertexBufferLayout {
    array_stride: 80,
    step_mode: VertexStepMode::Instance,
    attributes: &[/* model matrix (4x vec4), instance_id */],
}
```

### 6.2.4 Interleaved vs Separate Buffers

**Interleaved (single buffer, multiple attributes):**
```rust
// Memory layout: [P0 N0 UV0 T0] [P1 N1 UV1 T1] [P2 N2 UV2 T2] ...
struct InterleavedVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    tangent: [f32; 4],
}

// Single buffer binding
buffers: &[InterleavedVertex::layout()]
```

**Separate (multiple buffers, one attribute each):**
```rust
// Memory layout:
// Buffer 0: [P0 P1 P2 ...]
// Buffer 1: [N0 N1 N2 ...]
// Buffer 2: [UV0 UV1 UV2 ...]
// Buffer 3: [T0 T1 T2 ...]

buffers: &[
    VertexBufferLayout {
        array_stride: 12,
        step_mode: VertexStepMode::Vertex,
        attributes: &[VertexAttribute { format: Float32x3, offset: 0, shader_location: 0 }],
    },
    VertexBufferLayout {
        array_stride: 12,
        step_mode: VertexStepMode::Vertex,
        attributes: &[VertexAttribute { format: Float32x3, offset: 0, shader_location: 1 }],
    },
    // ...
]
```

**Trade-offs:**
| Aspect | Interleaved | Separate |
|--------|-------------|----------|
| Cache coherence | Better for full vertex access | Better for partial access |
| Memory | Single allocation | Multiple allocations |
| Flexibility | Must use all attributes | Can mix and match |
| Skinning | Positions separate = efficient | Natural fit |

### 6.2.5 TRINITY's Vertex Format Registry

```rust
pub struct VertexFormatRegistry {
    formats: HashMap<VertexFormatId, RegisteredFormat>,
}

pub struct RegisteredFormat {
    pub id: VertexFormatId,
    pub name: &'static str,
    pub layout: VertexBufferLayout<'static>,
    pub size: u64,
}

impl VertexFormatRegistry {
    pub fn register_standard_formats(&mut self) {
        self.register(VertexFormatId::STATIC_MESH, "StaticMesh", &[
            VertexAttribute { format: Float32x3, offset: 0, shader_location: 0 },  // Position
            VertexAttribute { format: Float32x3, offset: 12, shader_location: 1 }, // Normal
            VertexAttribute { format: Float32x2, offset: 24, shader_location: 2 }, // UV0
            VertexAttribute { format: Float32x4, offset: 32, shader_location: 3 }, // Tangent
        ], 48);
        
        self.register(VertexFormatId::SKINNED_MESH, "SkinnedMesh", &[
            VertexAttribute { format: Float32x3, offset: 0, shader_location: 0 },  // Position
            VertexAttribute { format: Float32x3, offset: 12, shader_location: 1 }, // Normal
            VertexAttribute { format: Float32x2, offset: 24, shader_location: 2 }, // UV0
            VertexAttribute { format: Float32x4, offset: 32, shader_location: 3 }, // Tangent
            VertexAttribute { format: Uint16x4, offset: 48, shader_location: 4 },  // Joints
            VertexAttribute { format: Unorm16x4, offset: 56, shader_location: 5 }, // Weights
        ], 64);
        
        self.register(VertexFormatId::TERRAIN, "Terrain", &[
            VertexAttribute { format: Float32x3, offset: 0, shader_location: 0 },  // Position
            VertexAttribute { format: Unorm8x4, offset: 12, shader_location: 1 },  // Normal (packed)
        ], 16);
        
        self.register(VertexFormatId::PARTICLE, "Particle", &[
            VertexAttribute { format: Float32x3, offset: 0, shader_location: 0 },  // Position
            VertexAttribute { format: Float32, offset: 12, shader_location: 1 },   // Size
            VertexAttribute { format: Unorm8x4, offset: 16, shader_location: 2 },  // Color
            VertexAttribute { format: Float32x2, offset: 20, shader_location: 3 }, // UV
        ], 28);
        
        self.register(VertexFormatId::UI, "UI", &[
            VertexAttribute { format: Float32x2, offset: 0, shader_location: 0 },  // Position
            VertexAttribute { format: Float32x2, offset: 8, shader_location: 1 },  // UV
            VertexAttribute { format: Unorm8x4, offset: 16, shader_location: 2 },  // Color
        ], 20);
    }
}
```

---

## 6.3 Primitive Assembly

### 6.3.1 Primitive Topologies

```rust
pub enum PrimitiveTopology {
    PointList,      // Each vertex is a point
    LineList,       // Every 2 vertices form a line
    LineStrip,      // Connected lines (v0-v1, v1-v2, v2-v3, ...)
    TriangleList,   // Every 3 vertices form a triangle
    TriangleStrip,  // Connected triangles
}
```

```
PointList:     v0   v1   v2   v3
                •    •    •    •

LineList:      v0───v1   v2───v3

LineStrip:     v0───v1───v2───v3

TriangleList:  v0─v1    v3─v4
               │╲ │     │╲ │
               v2─┘     v5─┘

TriangleStrip: v0───v2───v4
               │╲  │╲  │╲
               │ ╲ │ ╲ │ ╲
               v1──v3──v5
```

### 6.3.2 Index Formats

```rust
pub enum IndexFormat {
    Uint16,  // 2 bytes, up to 65,535 vertices
    Uint32,  // 4 bytes, up to 4,294,967,295 vertices
}
```

### 6.3.3 Front Face and Culling

```rust
pub enum FrontFace {
    Ccw,  // Counter-clockwise winding = front (default)
    Cw,   // Clockwise winding = front
}

pub enum Face {
    Front,
    Back,
}

// Common configurations
PrimitiveState {
    front_face: FrontFace::Ccw,
    cull_mode: Some(Face::Back),  // Standard: cull back faces
    ..
}

PrimitiveState {
    front_face: FrontFace::Ccw,
    cull_mode: None,  // Double-sided rendering
    ..
}

PrimitiveState {
    front_face: FrontFace::Ccw,
    cull_mode: Some(Face::Front),  // Inside-out (skybox)
    ..
}
```

### 6.3.4 Polygon Modes

```rust
pub enum PolygonMode {
    Fill,   // Filled triangles (default)
    Line,   // Wireframe (requires POLYGON_MODE_LINE feature)
    Point,  // Points at vertices (requires POLYGON_MODE_POINT feature)
}
```

### 6.3.5 Unclipped Depth

```rust
// Enable depth values outside [0, 1] (requires DEPTH_CLIP_CONTROL feature)
PrimitiveState {
    unclipped_depth: true,  // Allow depth < 0 or > 1
    ..
}
```

---

## 6.4 Rasterization

### 6.4.1 Viewport and Scissor

Set during render pass, not in pipeline:

```rust
// Viewport: transform NDC to window coordinates
render_pass.set_viewport(
    x,      // Left edge
    y,      // Top edge
    width,  // Width
    height, // Height
    min_depth, // Depth range min (usually 0.0)
    max_depth, // Depth range max (usually 1.0)
);

// Scissor: clip to rectangle
render_pass.set_scissor_rect(x, y, width, height);
```

### 6.4.2 Depth Bias (Polygon Offset)

```rust
DepthBiasState {
    constant: 2,       // Constant offset (in depth units)
    slope_scale: 2.0,  // Slope-dependent offset
    clamp: 0.0,        // Maximum bias (0 = no clamp)
}

// Use cases:
// - Shadow mapping: offset shadow casters to prevent shadow acne
// - Decals: offset decals above surfaces
// - Coplanar geometry: separate overlapping surfaces
```

### 6.4.3 Conservative Rasterization

```rust
// Rasterize all pixels that touch the triangle (not just centers)
// Requires CONSERVATIVE_RASTERIZATION feature
PrimitiveState {
    conservative: true,
    ..
}

// Use cases:
// - Voxelization
// - Visibility buffer
// - Guaranteed coverage
```

### 6.4.4 Sample Mask

```rust
MultisampleState {
    mask: 0b1010,  // Only samples 1 and 3 enabled
    ..
}
```

### 6.4.5 Alpha to Coverage

```rust
MultisampleState {
    alpha_to_coverage_enabled: true,  // Use alpha for sample coverage
    ..
}

// Effect: alpha 0.5 = ~50% of samples covered
// Use case: Order-independent transparency approximation
```

---

## 6.5 Fragment Processing

### 6.5.1 Fragment Shader Outputs

```wgsl
struct FragmentOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) material: vec4<f32>,
    @builtin(frag_depth) depth: f32,  // Optional depth override
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    out.albedo = textureSample(albedo_map, sampler, in.uv);
    out.normal = vec4(encode_normal(in.world_normal), 1.0);
    out.material = vec4(metallic, roughness, ao, 1.0);
    // out.depth = custom_depth;  // If needed
    return out;
}
```

### 6.5.2 Color Target State

```rust
ColorTargetState {
    format: TextureFormat::Rgba8UnormSrgb,
    blend: Some(BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        },
    }),
    write_mask: ColorWrites::ALL,
}
```

### 6.5.3 Write Mask

```rust
pub struct ColorWrites: u32 {
    const RED = 0x1;
    const GREEN = 0x2;
    const BLUE = 0x4;
    const ALPHA = 0x8;
    const COLOR = Self::RED.bits() | Self::GREEN.bits() | Self::BLUE.bits();
    const ALL = Self::COLOR.bits() | Self::ALPHA.bits();
}

// Examples
ColorWrites::ALL          // Write RGBA
ColorWrites::COLOR        // Write RGB only
ColorWrites::RED          // Write R only (for single-channel effects)
ColorWrites::empty()      // Depth-only pass
```

### 6.5.4 Blending

**Blend factors:**
```rust
pub enum BlendFactor {
    Zero,
    One,
    Src,
    OneMinusSrc,
    SrcAlpha,
    OneMinusSrcAlpha,
    Dst,
    OneMinusDst,
    DstAlpha,
    OneMinusDstAlpha,
    SrcAlphaSaturated,
    Constant,
    OneMinusConstant,
}
```

**Blend operations:**
```rust
pub enum BlendOperation {
    Add,              // src * src_factor + dst * dst_factor
    Subtract,         // src * src_factor - dst * dst_factor
    ReverseSubtract,  // dst * dst_factor - src * src_factor
    Min,              // min(src, dst)
    Max,              // max(src, dst)
}
```

**Common blend modes:**
```rust
// Alpha blending (standard transparency)
BlendState {
    color: BlendComponent {
        src_factor: BlendFactor::SrcAlpha,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        operation: BlendOperation::Add,
    },
    alpha: BlendComponent::OVER,
}

// Premultiplied alpha
BlendState {
    color: BlendComponent {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::OneMinusSrcAlpha,
        operation: BlendOperation::Add,
    },
    alpha: BlendComponent::OVER,
}

// Additive
BlendState {
    color: BlendComponent {
        src_factor: BlendFactor::One,
        dst_factor: BlendFactor::One,
        operation: BlendOperation::Add,
    },
    alpha: BlendComponent::OVER,
}

// Multiply
BlendState {
    color: BlendComponent {
        src_factor: BlendFactor::Dst,
        dst_factor: BlendFactor::Zero,
        operation: BlendOperation::Add,
    },
    alpha: BlendComponent::OVER,
}

// Replace (no blending)
BlendState::REPLACE
```

### 6.5.5 Blend Constants

```rust
// Set in render pass
render_pass.set_blend_constant(Color {
    r: 0.5,
    g: 0.5,
    b: 0.5,
    a: 1.0,
});

// Use in blend state
BlendFactor::Constant      // Use blend constant
BlendFactor::OneMinusConstant
```

---

## 6.6 Depth/Stencil

### 6.6.1 Depth Test

```rust
DepthStencilState {
    format: TextureFormat::Depth32Float,
    depth_write_enabled: true,
    depth_compare: CompareFunction::Less,
    ..
}
```

### 6.6.2 Compare Functions

```rust
pub enum CompareFunction {
    Never,         // Always fail
    Less,          // Pass if new < stored
    Equal,         // Pass if new == stored
    LessEqual,     // Pass if new <= stored
    Greater,       // Pass if new > stored
    NotEqual,      // Pass if new != stored
    GreaterEqual,  // Pass if new >= stored
    Always,        // Always pass
}
```

### 6.6.3 Stencil State

```rust
StencilState {
    front: StencilFaceState {
        compare: CompareFunction::Always,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Replace,
    },
    back: StencilFaceState {
        compare: CompareFunction::Always,
        fail_op: StencilOperation::Keep,
        depth_fail_op: StencilOperation::Keep,
        pass_op: StencilOperation::Replace,
    },
    read_mask: 0xFF,
    write_mask: 0xFF,
}
```

### 6.6.4 Stencil Operations

```rust
pub enum StencilOperation {
    Keep,           // Keep current value
    Zero,           // Set to 0
    Replace,        // Set to reference value
    Invert,         // Bitwise invert
    IncrementClamp, // Increment, clamp to max
    DecrementClamp, // Decrement, clamp to 0
    IncrementWrap,  // Increment, wrap to 0
    DecrementWrap,  // Decrement, wrap to max
}
```

### 6.6.5 Stencil Reference

```rust
// Set in render pass
render_pass.set_stencil_reference(0x01);
```

---

## 6.7 Multisampling

### 6.7.1 Sample Count Selection

```rust
// Query supported sample counts
let supported = adapter.get_texture_format_features(format).flags;
let max_samples = if supported.contains(TextureFormatFeatureFlags::MULTISAMPLE_X16) { 16 }
    else if supported.contains(TextureFormatFeatureFlags::MULTISAMPLE_X8) { 8 }
    else if supported.contains(TextureFormatFeatureFlags::MULTISAMPLE_X4) { 4 }
    else if supported.contains(TextureFormatFeatureFlags::MULTISAMPLE_X2) { 2 }
    else { 1 };
```

### 6.7.2 MSAA Resolve

```rust
// In render pass, resolve to non-MSAA target
let render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
    color_attachments: &[Some(RenderPassColorAttachment {
        view: &msaa_view,           // MSAA target
        resolve_target: Some(&resolve_view),  // Final non-MSAA texture
        ops: Operations {
            load: LoadOp::Clear(Color::BLACK),
            store: StoreOp::Discard,  // Don't need MSAA data after resolve
        },
    })],
    ..
});
```

---

# Chapter 7: Render Passes

## 7.1 Render Pass Fundamentals

### 7.1.1 Render Pass Creation

```rust
let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
    label: Some("Main Pass"),
    color_attachments: &[
        Some(RenderPassColorAttachment {
            view: &color_view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color { r: 0.1, g: 0.1, b: 0.1, a: 1.0 }),
                store: StoreOp::Store,
            },
        }),
    ],
    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
        view: &depth_view,
        depth_ops: Some(Operations {
            load: LoadOp::Clear(1.0),
            store: StoreOp::Discard,
        }),
        stencil_ops: None,
    }),
    timestamp_writes: None,
    occlusion_query_set: None,
});
```

### 7.1.2-7.1.5 Attachments and Queries

```rust
// Color attachment
RenderPassColorAttachment {
    view: &texture_view,           // Target view
    resolve_target: Some(&resolve), // For MSAA resolve
    ops: Operations {
        load: LoadOp::Clear(color),  // Or Load
        store: StoreOp::Store,       // Or Discard
    },
}

// Depth/stencil attachment
RenderPassDepthStencilAttachment {
    view: &depth_view,
    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
    stencil_ops: Some(Operations { load: LoadOp::Clear(0), store: StoreOp::Discard }),
}

// Timestamp queries
timestamp_writes: Some(RenderPassTimestampWrites {
    query_set: &timestamp_query_set,
    beginning_of_pass_write_index: Some(0),
    end_of_pass_write_index: Some(1),
}),

// Occlusion queries
occlusion_query_set: Some(&occlusion_query_set),
```

## 7.2 Attachment Operations

```rust
pub enum LoadOp<V> {
    Clear(V),  // Clear to value
    Load,      // Preserve existing content
}

pub enum StoreOp {
    Store,   // Keep rendered content
    Discard, // Content may be discarded (optimization)
}
```

## 7.3 Render Pass Commands

```rust
// Pipeline and bindings
render_pass.set_pipeline(&pipeline);
render_pass.set_bind_group(0, &global_bind_group, &[]);
render_pass.set_bind_group(1, &material_bind_group, &[dynamic_offset]);

// Vertex/index buffers
render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
render_pass.set_index_buffer(index_buffer.slice(..), IndexFormat::Uint32);

// Dynamic state
render_pass.set_viewport(0.0, 0.0, width, height, 0.0, 1.0);
render_pass.set_scissor_rect(0, 0, width as u32, height as u32);
render_pass.set_blend_constant(Color::WHITE);
render_pass.set_stencil_reference(0xFF);

// Push constants
render_pass.set_push_constants(ShaderStages::VERTEX, 0, bytemuck::bytes_of(&transform));
```

## 7.4 Draw Commands

```rust
// Basic draw
render_pass.draw(vertex_count, instance_count, first_vertex, first_instance);

// Indexed draw
render_pass.draw_indexed(index_count, instance_count, first_index, base_vertex, first_instance);

// Indirect draw
render_pass.draw_indirect(&indirect_buffer, offset);
render_pass.draw_indexed_indirect(&indirect_buffer, offset);

// Multi-draw indirect (requires feature)
render_pass.multi_draw_indirect(&indirect_buffer, offset, count);
render_pass.multi_draw_indexed_indirect(&indirect_buffer, offset, count);

// Multi-draw indirect with count buffer
render_pass.multi_draw_indirect_count(
    &indirect_buffer,
    indirect_offset,
    &count_buffer,
    count_offset,
    max_count,
);
```

## 7.5 Render Bundles

```rust
// Create bundle encoder
let mut bundle_encoder = device.create_render_bundle_encoder(&RenderBundleEncoderDescriptor {
    label: Some("Static Geometry Bundle"),
    color_formats: &[Some(TextureFormat::Rgba8UnormSrgb)],
    depth_stencil: Some(RenderBundleDepthStencil {
        format: TextureFormat::Depth32Float,
        depth_read_only: false,
        stencil_read_only: true,
    }),
    sample_count: 1,
    multiview: None,
});

// Record commands
bundle_encoder.set_pipeline(&pipeline);
bundle_encoder.set_bind_group(0, &bind_group, &[]);
for mesh in &static_meshes {
    bundle_encoder.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
    bundle_encoder.set_index_buffer(mesh.index_buffer.slice(..), IndexFormat::Uint32);
    bundle_encoder.draw_indexed(mesh.index_count, 1, 0, 0, 0);
}

// Finish bundle
let bundle = bundle_encoder.finish(&RenderBundleDescriptor {
    label: Some("Static Geometry"),
});

// Execute bundle in render pass
render_pass.execute_bundles([&bundle]);
```

**TRINITY's bundle system:**
```rust
pub struct RenderBundleCache {
    bundles: HashMap<BundleKey, RenderBundle>,
    device: Arc<Device>,
}

impl RenderBundleCache {
    pub fn get_or_create(
        &mut self,
        key: &BundleKey,
        meshes: &[StaticMesh],
        pipeline: &RenderPipeline,
        bind_groups: &[&BindGroup],
    ) -> &RenderBundle {
        self.bundles.entry(key.clone()).or_insert_with(|| {
            self.create_bundle(meshes, pipeline, bind_groups)
        })
    }
    
    pub fn invalidate(&mut self, key: &BundleKey) {
        self.bundles.remove(key);
    }
}
```

---

# TRINITY Render Pipeline Module Architecture

```
crates/renderer-backend/src/pipeline/
├── mod.rs              # Module root
├── render_pipeline.rs  # Render pipeline creation
├── compute_pipeline.rs # Compute pipeline
├── cache.rs            # Pipeline caching
├── layout.rs           # Pipeline layout management
├── vertex.rs           # Vertex format registry
├── state.rs            # Pipeline state builders
└── bundle.rs           # Render bundle system
```

---

*End of WGPU_PART_IV_RENDER_PIPELINE.md*

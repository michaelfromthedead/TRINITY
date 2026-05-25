# wgpu Investigation

**Date:** 2026-05-24  
**Status:** COMPREHENSIVE REVIEW  
**Version:** wgpu 22

---

## Executive Summary

TRINITY has a substantial wgpu infrastructure that's **architecturally complete but not wired together**. The components exist in isolation:

| Component | Status | Lines |
|-----------|--------|-------|
| RHI Abstraction | Implemented | 6,424 |
| WGSL Shaders | Production-ready | 1,386 |
| Frame Graph Compiler | Working | 10,000+ |
| Python↔Rust Bridge | Working | 350 |
| **Actual Rendering** | **Triangle only** | — |

The gap is integration: PBR shaders exist, material tables exist, the frame graph compiles passes — but the renderer only draws a hardcoded colored triangle.

---

## 1. Current Architecture

### 1.1 Crate Structure

```
crates/renderer-backend/
├── src/
│   ├── lib.rs              # Module exports
│   ├── renderer.rs         # wgpu init + triangle demo (883 lines)
│   ├── pipeline.rs         # ShaderCache + PipelineTable (706 lines)
│   ├── rhi_device.rs       # Device/adapter wrapper (765 lines)
│   ├── rhi_resources.rs    # Buffer/texture creation (869 lines)
│   ├── rhi_pipeline.rs     # Pipeline creation (1,143 lines)
│   ├── rhi_commands.rs     # Command recording (691 lines)
│   ├── rhi_swapchain.rs    # Swapchain management (551 lines)
│   ├── rhi_bind_group.rs   # Bind group layouts (816 lines)
│   ├── frame_graph/        # IR types + compiler (10,000+ lines)
│   └── gpu_driven/         # Mesh/material/texture tables
└── shaders/
    ├── pbr.vert.wgsl       # PBR vertex shader (63 lines)
    ├── pbr.frag.wgsl       # PBR fragment shader (376 lines)
    ├── shadow.vert.wgsl    # Shadow mapping vertex (34 lines)
    ├── shadow.frag.wgsl    # Shadow mapping fragment (12 lines)
    ├── shadow_csm.wgsl     # Cascaded shadow maps (161 lines)
    ├── light_culling.wgsl  # Clustered light culling (229 lines)
    ├── ddgi.wgsl           # Dynamic GI probes (240 lines)
    └── particles.wgsl      # GPU particles (271 lines)
```

### 1.2 Dependencies

```toml
# Cargo.toml
wgpu = "22"
bytemuck = { version = "1", features = ["derive"] }
pollster = "0.3"  # Async runtime for wgpu
parking_lot = "0.12"
sha2 = "0.10"  # Shader hashing
```

---

## 2. What's Implemented

### 2.1 Renderer (renderer.rs)

**Current state:** Renders a single colored triangle with transform uniform.

```rust
pub struct Renderer {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub render_pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub size: (u32, u32),
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl Renderer {
    pub fn new(window: &impl HasWindowHandle, width: u32, height: u32) -> Self;
    pub fn resize(&mut self, width: u32, height: u32);
    pub fn render(&mut self);  // Draws colored triangle
}
```

**Capabilities:**
- wgpu instance creation with all backends
- Surface creation from raw window handles
- Adapter selection (high performance)
- Device + queue creation
- Surface configuration
- Single render pipeline
- Vertex + uniform buffers
- Bind groups

**NOT wired:**
- PBR pipeline
- Material system
- Frame graph execution
- Shadow mapping
- Light culling
- GPU-driven rendering

### 2.2 RHI Device (rhi_device.rs)

```rust
pub struct RhiDevice {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    features: wgpu::Features,
    limits: wgpu::Limits,
}

impl RhiDevice {
    pub async fn new(backend: wgpu::Backends) -> Self;
    pub fn create_buffer(&self, desc: &BufferDesc) -> wgpu::Buffer;
    pub fn create_texture(&self, desc: &TextureDesc) -> wgpu::Texture;
    pub fn create_sampler(&self, desc: &SamplerDesc) -> wgpu::Sampler;
}
```

### 2.3 ShaderCache (pipeline.rs)

```rust
pub struct ShaderCache {
    cache: HashMap<[u8; 32], Arc<wgpu::ShaderModule>>,
}

impl ShaderCache {
    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        source: &str
    ) -> Arc<wgpu::ShaderModule>;
}
```

**Features:**
- SHA-256 content hashing
- Deduplication of identical shaders
- Arc-wrapped modules for sharing (wgpu 22 compatible)

### 2.4 PipelineTable (pipeline.rs)

```rust
pub struct PipelineTable {
    pipelines: HashMap<u32, CachedPipeline>,
    shader_cache: ShaderCache,
}

pub struct CachedPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub generation: u32,
}
```

### 2.5 Command Recording (rhi_commands.rs)

```rust
pub struct CommandBuffer {
    encoder: wgpu::CommandEncoder,
    label: String,
}

impl CommandBuffer {
    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc) -> RenderPass;
    pub fn begin_compute_pass(&mut self, desc: &ComputePassDesc) -> ComputePass;
    pub fn copy_buffer_to_buffer(&mut self, src, dst, size);
    pub fn copy_texture_to_texture(&mut self, src, dst);
    pub fn finish(self) -> wgpu::CommandBuffer;
}
```

---

## 3. WGSL Shaders

### 3.1 PBR Pipeline (pbr.frag.wgsl)

**376 lines** — Production-ready Cook-Torrance BRDF:

```wgsl
struct MaterialTableEntry {
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
    normal_scale: f32,
    albedo_texture_id: u32,
    normal_texture_id: u32,
    metallic_roughness_tex_id: u32,
    emissive_texture_id: u32,
    flags: u32,
    alpha_cutoff: f32,
}
```

**Features:**
- GGX normal distribution
- Smith-GGX geometry term
- Schlick Fresnel
- Bindless material table
- Point/directional/spot lights
- CSM shadow sampling

### 3.2 Shadow Mapping

- `shadow.vert.wgsl` (34 lines) — Depth-only vertex
- `shadow.frag.wgsl` (12 lines) — Depth output
- `shadow_csm.wgsl` (161 lines) — Cascaded shadow maps

### 3.3 Advanced Features

- `light_culling.wgsl` (229 lines) — Clustered forward+
- `ddgi.wgsl` (240 lines) — Dynamic diffuse GI probes
- `particles.wgsl` (271 lines) — GPU particle simulation

### 3.4 Demoscene/Procedural

```
src/demoscene/
├── sdf_domain.wgsl      # SDF ray marching
├── noise_hash.wgsl      # Hash-based noise
├── noise_perlin.wgsl    # Classic Perlin
├── noise_ridged.wgsl    # Ridged multifractal
├── noise_fbm.wgsl       # Fractal Brownian motion
└── noise_domain_warp.wgsl
```

---

## 4. Python↔Rust Bridge

### 4.1 Bridge Functions (omega/src/bridge.rs)

```rust
// Frame graph (WORKING)
#[pyfunction]
pub fn frame_graph_execute(json: String) -> PyResult<String>;

// Renderer control (STUBS)
#[pyfunction]
pub fn renderer_init() -> PyResult<()>;

#[pyfunction]
pub fn renderer_resize(width: u32, height: u32) -> PyResult<()>;

#[pyfunction]
pub fn renderer_screenshot(_path: String) -> PyResult<()>;

#[pyfunction]
pub fn renderer_shutdown() -> PyResult<()>;
```

### 4.2 Global Renderer State

```rust
static RENDERER: OnceLock<Mutex<Option<Renderer>>> = OnceLock::new();

fn get_renderer_lock() -> &'static Mutex<Option<Renderer>> {
    RENDERER.get_or_init(|| Mutex::new(None))
}
```

**Problem:** `renderer_init()` doesn't actually create a Renderer — it just acquires the lock. The Renderer needs a window handle, which Python can't easily provide through PyO3.

---

## 5. Gap Analysis

### 5.1 What's Missing

| Gap | Description | Effort |
|-----|-------------|--------|
| **Window integration** | Renderer::new needs window handle | 2 days |
| **Headless mode** | Offscreen rendering without window | 1 day |
| **Frame graph → GPU** | Execute compiled passes | 3 days |
| **Material binding** | Wire MaterialTable to PBR shader | 2 days |
| **Mesh rendering** | Wire MeshTable to draw calls | 2 days |
| **Light system** | Wire LightTable to shaders | 1 day |
| **Shadow passes** | CSM render pass implementation | 2 days |

### 5.2 Integration Path

```
┌─────────────────────────────────────────────────────────────┐
│                     Python Frame Graph                       │
│  fg.add_pass("shadow", PassType.GRAPHICS)                   │
│  fg.add_pass("main", PassType.GRAPHICS)                     │
│         │                                                    │
│         ▼ JSON IR                                            │
└─────────┬───────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────┐
│                     Rust Frame Graph                         │
│  CompiledFrameGraph::compile() ─────────────────────────────┤
│         │                                                    │
│         │ Barriers, execution order, resource aliasing       │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Frame Graph Executor (MISSING)          │   │
│  │  for pass in compiled.order:                         │   │
│  │      encoder.begin_render_pass(pass)                 │   │
│  │      bind materials, meshes, lights                  │   │
│  │      draw()                                          │   │
│  │  queue.submit(encoder.finish())                      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Recommendations

### 6.1 Immediate (Unblock Development)

1. **Add headless rendering** — Create RhiDevice without surface for testing
2. **Implement FrameGraphExecutor** — Consume CompiledFrameGraph, emit GPU commands

### 6.2 Short-term (First Real Frame)

3. **Wire PBR pipeline** — Connect pbr.wgsl to material/mesh tables
4. **Add simple scene** — Hardcoded cube with one light
5. **Python trigger** — `renderer_render_frame()` bridge function

### 6.3 Medium-term (Production Path)

6. **Shadow passes** — CSM implementation
7. **Light culling** — Clustered forward+
8. **GPU particles** — Wire particles.wgsl

---

## 7. Test Coverage

### 7.1 Working Tests

| Category | Count | Notes |
|----------|-------|-------|
| Frame graph compiler | 715 | Full coverage |
| Mesh/material tables | 120+ | GPU-driven data structures |
| Barriers | 66 | State transitions |

### 7.2 Ignored Tests (Need GPU)

| Test | Reason |
|------|--------|
| `test_create_render_pipeline_invalid_wgsl` | Needs device |
| `test_create_compute_pipeline_invalid_wgsl` | Needs device |

### 7.3 Missing Tests

- Actual frame rendering
- Swapchain present
- Multi-pass execution

---

## 8. wgpu Version Notes

**Current:** wgpu 22

**Key APIs used:**
- `Instance::new(InstanceDescriptor)` — current syntax
- `create_surface_unsafe(SurfaceTargetUnsafe::RawHandle)` — for 'static lifetime
- `Surface<'static>` — avoids lifetime issues with window
- `Arc<ShaderModule>` — for sharing (Clone removed in wgpu 23)

**Upgrade path:** wgpu 23+ changes:
- `ShaderModule::clone()` removed → already using Arc
- Entry point types changed → already fixed
- Surface creation API changed → would need update

---

## 9. Quick Start: First Real Frame

```rust
// Minimal integration to render something beyond triangle

impl Renderer {
    pub fn execute_frame_graph(&mut self, compiled: &CompiledFrameGraph) {
        let frame = self.surface.get_current_texture().unwrap();
        let view = frame.texture.create_view(&Default::default());
        
        let mut encoder = self.device.create_command_encoder(&Default::default());
        
        for pass_idx in &compiled.order {
            let pass = compiled.passes.iter()
                .find(|p| p.index == *pass_idx)
                .unwrap();
            
            match pass.pass_type {
                PassType::Graphics => {
                    let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                        color_attachments: &[Some(RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: Operations {
                                load: LoadOp::Clear(Color::BLACK),
                                store: StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    
                    // TODO: Bind materials, meshes, draw
                    rpass.set_pipeline(&self.render_pipeline);
                    rpass.draw(0..3, 0..1);
                }
                PassType::Compute => {
                    let mut cpass = encoder.begin_compute_pass(&Default::default());
                    // TODO: Dispatch compute
                }
                _ => {}
            }
        }
        
        self.queue.submit([encoder.finish()]);
        frame.present();
    }
}
```

---

*Last updated: 2026-05-24*

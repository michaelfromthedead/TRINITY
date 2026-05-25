# RHI/wgpu Module Evaluation

**Modules:** renderer-backend::rhi_*, renderer, pipeline
**Location:** `/crates/renderer-backend/src/`
**Lines:** ~6,500
**Quality Grade:** A-

---

## Purpose

Rendering Hardware Interface (RHI) abstraction layer using wgpu. Provides GPU resource creation, pipeline management, and command recording.

---

## File Inventory

| File | Lines | Purpose | Quality |
|------|-------|---------|---------|
| renderer.rs | 883 | wgpu renderer (triangle demo) | A |
| pipeline.rs | 705 | PipelineTable + ShaderCache | A |
| rhi_device.rs | 765 | Device/adapter/queue wrapper | A |
| rhi_resources.rs | 869 | Buffer/texture creation | A |
| rhi_pipeline.rs | 1,146 | Pipeline creation | A |
| rhi_commands.rs | 691 | Command buffer recording | A- |
| rhi_swapchain.rs | 555 | Swapchain management | A |
| rhi_bind_group.rs | 1,008 | Bind group layouts | A |

---

## Renderer (renderer.rs)

### wgpu Initialization

```rust
pub struct Renderer {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: Option<wgpu::Surface>,
    config: wgpu::SurfaceConfiguration,
    // ...
}

impl Renderer {
    pub async fn new(window: &winit::window::Window) -> Self;
    pub fn resize(&mut self, width: u32, height: u32);
    pub fn render(&mut self);
}
```

**Current state:** Renders a colored triangle. PBR pipeline not wired.

---

## Pipeline Management (pipeline.rs)

### ShaderCache

```rust
pub struct ShaderCache {
    cache: HashMap<[u8; 32], wgpu::ShaderModule>,
}

impl ShaderCache {
    pub fn get_or_create(&mut self, device: &wgpu::Device, source: &str) -> &wgpu::ShaderModule;
}
```

**Features:**
- SHA-256 content hashing
- Deduplication of identical shaders
- Lazy compilation

### PipelineTable

```rust
pub struct PipelineTable {
    pipelines: HashMap<u32, CachedPipeline>,
    shader_cache: ShaderCache,
}

pub struct CachedPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub layout: wgpu::PipelineLayout,
    pub generation: u32,
}
```

---

## RHI Device (rhi_device.rs)

### Device Wrapper

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

---

## RHI Resources (rhi_resources.rs)

### Buffer Creation

```rust
pub struct BufferDesc {
    pub size: u64,
    pub usage: wgpu::BufferUsages,
    pub mapped_at_creation: bool,
}

pub fn create_buffer(device: &wgpu::Device, desc: &BufferDesc) -> wgpu::Buffer;
```

### Texture Creation

```rust
pub struct TextureDesc {
    pub size: wgpu::Extent3d,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
    pub mip_level_count: u32,
    pub sample_count: u32,
}

pub fn create_texture(device: &wgpu::Device, desc: &TextureDesc) -> wgpu::Texture;
```

---

## RHI Pipeline (rhi_pipeline.rs)

### Pipeline Creation

```rust
pub struct RenderPipelineDesc {
    pub vertex_shader: String,
    pub fragment_shader: String,
    pub vertex_layouts: Vec<wgpu::VertexBufferLayout>,
    pub color_targets: Vec<wgpu::ColorTargetState>,
    pub depth_stencil: Option<wgpu::DepthStencilState>,
    pub primitive: wgpu::PrimitiveState,
    pub multisample: wgpu::MultisampleState,
}

pub fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    desc: &RenderPipelineDesc,
) -> wgpu::RenderPipeline;
```

### Compute Pipeline

```rust
pub struct ComputePipelineDesc {
    pub shader: String,
    pub entry_point: String,
}

pub fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    desc: &ComputePipelineDesc,
) -> wgpu::ComputePipeline;
```

---

## RHI Commands (rhi_commands.rs)

### Command Recording

```rust
pub struct CommandBuffer {
    encoder: wgpu::CommandEncoder,
    label: String,
}

impl CommandBuffer {
    pub fn begin_render_pass(&mut self, desc: &RenderPassDesc) -> RenderPass;
    pub fn begin_compute_pass(&mut self, desc: &ComputePassDesc) -> ComputePass;
    pub fn copy_buffer_to_buffer(&mut self, src: &wgpu::Buffer, dst: &wgpu::Buffer, size: u64);
    pub fn finish(self) -> wgpu::CommandBuffer;
}
```

---

## RHI Swapchain (rhi_swapchain.rs)

### Swapchain Management

```rust
pub struct Swapchain {
    surface: wgpu::Surface,
    config: wgpu::SurfaceConfiguration,
    current_texture: Option<wgpu::SurfaceTexture>,
}

impl Swapchain {
    pub fn new(device: &wgpu::Device, surface: wgpu::Surface, width: u32, height: u32) -> Self;
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32);
    pub fn acquire(&mut self) -> Option<&wgpu::SurfaceTexture>;
    pub fn present(&mut self);
}
```

---

## RHI Bind Groups (rhi_bind_group.rs)

### Bind Group Creation

```rust
pub struct BindGroupLayoutDesc {
    pub entries: Vec<wgpu::BindGroupLayoutEntry>,
}

pub struct BindGroupDesc {
    pub layout: &wgpu::BindGroupLayout,
    pub entries: Vec<wgpu::BindGroupEntry>,
}

pub fn create_bind_group_layout(device: &wgpu::Device, desc: &BindGroupLayoutDesc) -> wgpu::BindGroupLayout;
pub fn create_bind_group(device: &wgpu::Device, desc: &BindGroupDesc) -> wgpu::BindGroup;
```

---

## Test Coverage

| Test Category | Files | Coverage |
|---------------|-------|----------|
| Renderer | 1 | Triangle rendering |
| Pipeline | 3 | ShaderCache, PipelineTable |
| Resources | 2 | Buffer, texture creation |

**Status:** Tests exist but **cannot compile** due to missing exports. Some tests require GPU.

---

## Blocking Issues

### 1. Not exported from lib.rs

```rust
// Need:
pub mod renderer;
pub mod pipeline;
pub mod rhi_device;
pub mod rhi_resources;
pub mod rhi_pipeline;
pub mod rhi_commands;
pub mod rhi_swapchain;
pub mod rhi_bind_group;
```

### 2. Triangle demo only

The renderer draws a colored triangle. PBR pipeline exists in WGSL but isn't wired.

### 3. No Python integration

Python `engine/platform/rhi/` has abstract interfaces. Rust has wgpu impl. No bridge.

---

## Recommendations

1. **Export from lib.rs** - Immediate
2. **Wire PBR pipeline** - Connect pbr.wgsl to renderer
3. **Add Python bridge** - Either PyO3 or command channel
4. **Run GPU tests** - Requires actual GPU

---

## Python Counterpart

| Rust | Python | Status |
|------|--------|--------|
| Renderer | engine/platform/rhi/device.py | ABC only |
| PipelineTable | engine/rendering/materials/shader_compiler.py | Python impl |
| RhiDevice | engine/platform/rhi/device.py | ABC only |
| RhiResources | engine/platform/rhi/resources.py | ABC only |

Python has abstract base classes. Rust has wgpu implementation. They're not connected.

---

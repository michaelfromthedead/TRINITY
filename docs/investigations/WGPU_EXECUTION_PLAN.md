# wgpu Execution Plan

**Goal:** Wire the existing wgpu infrastructure to render real frames  
**Timeline:** 5 phases, ~2 weeks of focused work  
**Starting Point:** Triangle demo + compiled frame graph + PBR shaders

**Progress:** Phase 1 and Phase 2 complete (2026-05-24)

---

## Context: Integration, Not Architecture

**Key Finding (2026-05-24 Investigation):** The wgpu backend is NOT a few hundred lines needing architecture — it's **140,000 lines** of existing code that needs **wiring**.

| Component | Lines | Status |
|-----------|-------|--------|
| Frame Graph IR + Compiler | 27,026 | ✅ Complete |
| RHI Layer (device/pipeline/resources) | 4,901 | ✅ Exists |
| GPU Tables (material/mesh/texture) | 6,529 | ✅ Exists |
| Executor + Headless | ~1,200 | ✅ Complete |
| Test Suite | 66,683 | ✅ 74 test files |

**The gap is wiring, not architecture.** All the pieces exist — they need to be connected.

### Gap Set Alignment

| Phase | Goal | Gap Set | Tasks |
|-------|------|---------|-------|
| 1 | Headless rendering | GAPSET_2 Phase 5 | ✅ Complete |
| 2 | Frame graph executor | GAPSET_2 Phase 7 | ✅ Complete |
| 3 | Material pipeline | GAPSET_4 Phases 1-3 | ⬅ NEXT |
| 4 | Mesh rendering | GAPSET_4 Phase 8 | Pending |
| 5 | Python integration | GAPSET_3 | ✅ Already complete |

**Note:** Phase 5 (Python integration) is actually ALREADY DONE via GAPSET_3_BRIDGE. The bridge exists and works — what's missing is the render passes that it would call.

### Blocker: GAPSET_1_CORE

Before rendering subsystems can be wired, GAPSET_1_CORE needs completion:

| Task | Status | Blocks |
|------|--------|--------|
| ThreadPool with work-stealing | ABSENT | Parallel dispatch |
| JobGraph and dependencies | ABSENT | Frame scheduling |
| Scheduler Bridge | ABSENT | Python↔Rust coordination |

See `docs/gap_sets/GAPSET_1_CORE/PHASE_N_TODO.md` for full task list.

---

## Phase 1: Headless Rendering (2 days)

**Objective:** Render to texture without a window — enables testing and CI.

### Tasks

#### 1.1 HeadlessDevice
```rust
// rhi_device.rs
impl RhiDevice {
    pub async fn new_headless() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,  // No surface needed
            force_fallback_adapter: false,
        }).await.expect("adapter");
        
        let (device, queue) = adapter.request_device(&Default::default(), None)
            .await.expect("device");
        
        Self { instance, adapter, device, queue, ... }
    }
}
```

#### 1.2 RenderTarget Abstraction
```rust
pub enum RenderTarget {
    Surface(wgpu::Surface<'static>),
    Texture {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        width: u32,
        height: u32,
    },
}

impl RenderTarget {
    pub fn view(&self) -> &wgpu::TextureView;
    pub fn readback(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u8>;
}
```

#### 1.3 Test: Render Triangle to PNG
```rust
#[test]
fn test_headless_triangle_render() {
    let device = pollster::block_on(RhiDevice::new_headless());
    let target = RenderTarget::new_texture(&device, 800, 600);
    
    let mut renderer = HeadlessRenderer::new(&device, target);
    renderer.render_triangle();
    
    let pixels = renderer.readback();
    assert!(pixels.iter().any(|&p| p != 0), "should have non-black pixels");
}
```

### Deliverables
- [x] `RhiDevice::new_headless()` — `src/rhi_device.rs`
- [x] `RenderTarget` enum — `src/headless.rs`
- [x] `HeadlessRenderer` struct — `src/headless.rs`
- [x] Passing test that renders to texture — 4 headless tests passing

---

## Phase 2: Frame Graph Executor (3 days)

**Objective:** Execute CompiledFrameGraph as actual GPU commands.

### Tasks

#### 2.1 FrameGraphExecutor
```rust
pub struct FrameGraphExecutor<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    compiled: &'a CompiledFrameGraph,
    
    // Resource allocations for this frame
    textures: HashMap<ResourceHandle, wgpu::TextureView>,
    buffers: HashMap<ResourceHandle, wgpu::Buffer>,
}

impl<'a> FrameGraphExecutor<'a> {
    pub fn new(
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        compiled: &'a CompiledFrameGraph,
    ) -> Self;
    
    pub fn allocate_resources(&mut self);
    pub fn execute(&self, output: &wgpu::TextureView);
}
```

#### 2.2 Pass Execution
```rust
impl FrameGraphExecutor<'_> {
    fn execute_pass(&self, encoder: &mut wgpu::CommandEncoder, pass: &IrPass) {
        match pass.pass_type {
            PassType::Graphics => self.execute_graphics_pass(encoder, pass),
            PassType::Compute => self.execute_compute_pass(encoder, pass),
            PassType::Copy => self.execute_copy_pass(encoder, pass),
            _ => {}
        }
    }
    
    fn execute_graphics_pass(&self, encoder: &mut wgpu::CommandEncoder, pass: &IrPass) {
        let color_attachments: Vec<_> = pass.color_attachments.iter()
            .map(|ca| {
                let view = self.textures.get(&ca.resource).unwrap();
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: match ca.load_op {
                            AttachmentLoadOp::Clear => wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            AttachmentLoadOp::Load => wgpu::LoadOp::Load,
                            AttachmentLoadOp::DontCare => wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })
            })
            .collect();
        
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&pass.name),
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
            ..Default::default()
        });
        
        // TODO: Bind pipeline, draw
    }
}
```

#### 2.3 Resource Allocation
```rust
impl FrameGraphExecutor<'_> {
    fn allocate_resources(&mut self) {
        for resource in &self.compiled.resources {
            match &resource.desc {
                ResourceDesc::Texture2D(desc) => {
                    let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some(&resource.name),
                        size: wgpu::Extent3d {
                            width: desc.width,
                            height: desc.height,
                            depth_or_array_layers: 1,
                        },
                        format: parse_format(&desc.format),
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT 
                             | wgpu::TextureUsages::TEXTURE_BINDING,
                        mip_level_count: desc.mip_levels,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        view_formats: &[],
                    });
                    let view = texture.create_view(&Default::default());
                    self.textures.insert(resource.handle, view);
                }
                ResourceDesc::Buffer(desc) => {
                    let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(&resource.name),
                        size: desc.size,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    self.buffers.insert(resource.handle, buffer);
                }
                _ => {}
            }
        }
    }
}
```

### Deliverables
- [x] `FrameGraphExecutor` struct — `src/executor.rs`
- [x] Resource allocation from IR — Texture2D, Texture3D, TextureCube, Buffer
- [x] Graphics pass execution — with color and depth-stencil attachments
- [x] Compute pass execution — placeholder for dispatch
- [x] Test: Multi-pass frame graph execution — 5 executor tests passing

---

## Phase 3: Material Pipeline (3 days)

**Objective:** Wire MaterialTable to PBR shader.

### Tasks

#### 3.1 Material Bind Group
```rust
pub struct MaterialSystem {
    material_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl MaterialSystem {
    pub fn new(device: &wgpu::Device, materials: &[MaterialTableEntry]) -> Self {
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Material Table"),
            contents: bytemuck::cast_slice(materials),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Material Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        
        // ...
    }
}
```

#### 3.2 PBR Pipeline Creation
```rust
pub fn create_pbr_pipeline(
    device: &wgpu::Device,
    shader_cache: &mut ShaderCache,
    material_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let vert_shader = shader_cache.get_or_create(device, include_str!("../shaders/pbr.vert.wgsl"));
    let frag_shader = shader_cache.get_or_create(device, include_str!("../shaders/pbr.frag.wgsl"));
    
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("PBR Pipeline Layout"),
        bind_group_layouts: &[material_layout],
        push_constant_ranges: &[],
    });
    
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("PBR Pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &vert_shader,
            entry_point: "vs_main",
            buffers: &[Vertex::layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &frag_shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}
```

### Deliverables
- [ ] `MaterialSystem` struct
- [ ] Material buffer upload
- [ ] PBR pipeline creation
- [ ] Material bind group
- [ ] Test: Render colored cube with material

---

## Phase 4: Mesh Rendering (2 days)

**Objective:** Wire MeshTable to draw calls.

### Tasks

#### 4.1 MeshSystem
```rust
pub struct MeshSystem {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    mesh_table: Vec<MeshTableEntry>,
}

impl MeshSystem {
    pub fn upload_mesh(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mesh: &Mesh) -> MeshId;
    
    pub fn draw(&self, rpass: &mut wgpu::RenderPass, mesh_id: MeshId) {
        let entry = &self.mesh_table[mesh_id.0 as usize];
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(entry.vertex_offset..));
        rpass.set_index_buffer(self.index_buffer.slice(entry.index_offset..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..entry.index_count, 0, 0..1);
    }
}
```

#### 4.2 Scene Integration
```rust
pub struct Scene {
    meshes: MeshSystem,
    materials: MaterialSystem,
    objects: Vec<RenderObject>,
}

pub struct RenderObject {
    mesh_id: MeshId,
    material_id: MaterialId,
    transform: Mat4,
}

impl Scene {
    pub fn render(&self, rpass: &mut wgpu::RenderPass) {
        for obj in &self.objects {
            // Push transform, bind material, draw mesh
            self.meshes.draw(rpass, obj.mesh_id);
        }
    }
}
```

### Deliverables
- [ ] `MeshSystem` struct
- [ ] Vertex/index buffer management
- [ ] `Scene` struct
- [ ] Test: Render multiple objects

---

## Phase 5: Python Integration (2 days)

**Objective:** Python can trigger real frame rendering.

### Tasks

#### 5.1 Bridge Functions
```rust
// omega/src/bridge.rs

static SCENE: OnceLock<Mutex<Option<Scene>>> = OnceLock::new();

#[pyfunction]
pub fn renderer_init_headless(width: u32, height: u32) -> PyResult<()> {
    let device = pollster::block_on(RhiDevice::new_headless());
    let renderer = HeadlessRenderer::new(&device, width, height);
    // Store in global
    Ok(())
}

#[pyfunction]
pub fn scene_add_mesh(vertices: Vec<f32>, indices: Vec<u32>) -> PyResult<u32> {
    // Add mesh to scene, return mesh_id
}

#[pyfunction]
pub fn scene_add_object(mesh_id: u32, material_id: u32, transform: Vec<f32>) -> PyResult<u32> {
    // Add render object
}

#[pyfunction]
pub fn render_frame() -> PyResult<Vec<u8>> {
    // Execute frame graph, return pixels
}
```

#### 5.2 Python API
```python
# engine/rendering/backend/rust_renderer.py

import _omega

class RustRenderer:
    def __init__(self, width: int, height: int):
        _omega.renderer_init_headless(width, height)
    
    def add_mesh(self, vertices: list[float], indices: list[int]) -> int:
        return _omega.scene_add_mesh(vertices, indices)
    
    def add_object(self, mesh_id: int, material_id: int, transform: list[float]) -> int:
        return _omega.scene_add_object(mesh_id, material_id, transform)
    
    def render(self) -> bytes:
        return bytes(_omega.render_frame())
```

### Deliverables
- [ ] `renderer_init_headless()` bridge function
- [ ] `scene_add_mesh()` bridge function  
- [ ] `render_frame()` bridge function
- [ ] Python `RustRenderer` wrapper
- [ ] Test: Python renders scene to PNG

---

## Validation Criteria

### Phase 1 Complete When:
```bash
cargo test test_headless_triangle_render -- --nocapture
# Outputs: "Rendered 800x600 frame, 1440000 bytes"
```

### Phase 2 Complete When:
```bash
cargo test test_frame_graph_execution -- --nocapture
# Outputs: "Executed 3 passes: Shadow, GBuffer, Lighting"
```

### Phase 3 Complete When:
```bash
cargo test test_pbr_material -- --nocapture
# Outputs: "Rendered cube with metallic=0.8, roughness=0.2"
```

### Phase 4 Complete When:
```bash
cargo test test_scene_render -- --nocapture
# Outputs: "Rendered scene with 5 objects"
```

### Phase 5 Complete When:
```python
>>> from engine.rendering.backend.rust_renderer import RustRenderer
>>> r = RustRenderer(800, 600)
>>> cube = r.add_mesh([...], [...])
>>> r.add_object(cube, 0, [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])
>>> pixels = r.render()
>>> len(pixels)
1920000
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| wgpu version mismatch | Pin to wgpu 22, test on CI |
| Shader compilation errors | Validate WGSL syntax first |
| Memory leaks | Use RAII, drop resources explicitly |
| Python GIL blocking | Release GIL during GPU work |

---

## Dependencies

```
Phase 1 (Headless) ─────┐
                        ├──► Phase 3 (Materials)
Phase 2 (Executor) ─────┤
                        ├──► Phase 4 (Meshes) ──► Phase 5 (Python)
                        │
                        └──► Can start immediately
```

Phases 1 and 2 can run in parallel. Phase 3 and 4 depend on both. Phase 5 depends on all.

---

*Created: 2026-05-24*

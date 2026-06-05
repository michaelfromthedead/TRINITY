# PHASE 7: INTEGRATION - Architecture

**Scope:** Presentation, platform backends, debugging, frame graph, Python bridge
**Duration:** 4-6 weeks
**Dependencies:** Phases 5 (RAY_TRACING), 6 (ADVANCED)
**Produces:** Complete integrated rendering system

---

## Overview

Phase 7 integrates all previous phases into a cohesive rendering system: surface/swapchain management, platform-specific backends, debugging/profiling tools, the frame graph abstraction, and the Python bridge for scripting.

### Covered Content (from MASTER.md Parts IX-XII)

- Chapter 18: Surface & Swapchain (Part IX)
  - 18.1 Surface (creation, capabilities, formats)
  - 18.2 Configuration (format, present mode, alpha mode)
  - 18.3 Frame acquisition (get_current_texture, resize)
  - 18.4 Presentation (present, frame pacing, triple buffering)

- Chapter 19-20: Platform & Capabilities (Part X)
  - 19.1-19.5 Platform backends (Vulkan, Metal, DX12, WebGPU, OpenGL)
  - 20.1-20.3 Feature detection, limits, TRINITY capability system

- Chapter 21-22: Debugging & Profiling (Part XI)
  - 21.1-21.3 Debug features (validation, markers, error scopes)
  - 22.1-22.4 Profiling (timestamp, statistics, memory, bottleneck)

- Chapter 23: Frame Graph (Part XII)
  - 23.1 Resource declaration (virtual, transient, external)
  - 23.2 Pass declaration (render, compute, RT, copy)
  - 23.3 Barrier resolution
  - 23.4 Execution (scheduling, async compute, lifetime)

- Chapter 24: Python Bridge (Part XII)
  - 24.1 PyO3 binding layer (type marshalling, handles, callbacks)
  - 24.2 Resource descriptors (Python-side, validation, caching)
  - 24.3 Command recording (builder pattern, deferred execution)

---

## Architectural Decisions

### ADR-025: Frame Graph as Central Abstraction

**Context:** Complex rendering requires coordinated passes and resources.

**Decision:** Frame graph is the primary rendering interface:
- All passes declared before execution
- Resources declared as virtual, allocated lazily
- Barriers computed automatically
- Single compile + execute per frame

**Rationale:** Enables automatic optimization and resource aliasing.

**Consequences:**
- Slight overhead for simple renders
- Maximum optimization for complex renders
- Clear pass dependency visualization

---

### ADR-026: Transient Resource Pooling

**Context:** Many render targets are only used within a single frame.

**Decision:** Implement TransientResourcePool:
- Textures/buffers acquired from pool
- Released at frame end
- Reused across frames
- Aliasing for non-overlapping lifetimes

**Rationale:** Minimizes memory allocation and GPU memory usage.

**Consequences:**
- Pool management overhead
- Memory savings 30-50% typical
- GC for unused pooled resources

---

### ADR-027: Python Bridge Design

**Context:** TRINITY needs scripting for rapid prototyping.

**Decision:** PyO3-based bridge with:
- Descriptor classes (not raw handles)
- Command builder pattern
- Deferred execution
- Error propagation to Python exceptions

**Rationale:** Ergonomic Python API; performance via batching.

**Consequences:**
- PyO3 dependency
- GIL considerations
- Memory management across languages

---

### ADR-028: Platform Abstraction Strategy

**Context:** Different backends have different capabilities.

**Decision:** Platform-specific code isolated to:
- Backend selection at instance creation
- Capability detection at adapter selection
- Format selection at surface configuration
- No per-frame backend checks

**Rationale:** Clean abstraction; performance via upfront decisions.

**Consequences:**
- Platform modules for backend-specific code
- Testing matrix for all backends
- Unified API for renderer code

---

## Component Breakdown

### 1. Surface & Presentation

```
TrinitySurface
├── surface: wgpu::Surface
├── config: wgpu::SurfaceConfiguration
├── capabilities: wgpu::SurfaceCapabilities
├── current_size: (u32, u32)
└── present_mode: wgpu::PresentMode
```

**PresentationEngine:**
- frame_pacer: FramePacer
- synchronizer: FrameSynchronizer
- triple_buffer_config: TripleBufferConfig

**Present Modes:**
| Mode | VSync | Tearing | Latency |
|------|-------|---------|---------|
| Fifo | Yes | No | Higher |
| FifoRelaxed | Yes (usually) | Occasional | Medium |
| Mailbox | No | No | Low |
| Immediate | No | Yes | Lowest |

**TripleBufferConfig Presets:**
- low_latency: Mailbox/Immediate, 2 frames
- smooth: Fifo, 3 frames
- uncapped: Immediate, 1 frame

### 2. Platform Backends

```
PlatformBackend
├── vulkan: Option<VulkanFeatures>
├── metal: Option<MetalCapabilities>
├── dx12: Option<DX12Capabilities>
├── webgpu: Option<WebGPULimitations>
└── opengl: Option<GLESOptimizations>
```

**VulkanFeatures:**
- ray_tracing: bool
- mesh_shaders: bool
- descriptor_indexing: bool
- timeline_semaphores: bool

**MetalCapabilities:**
- supports_ray_tracing: bool
- apple_gpu_family: u32
- unified_memory: bool

**DX12Capabilities:**
- feature_level: u32 (110-122)
- shader_model: f32
- ray_tracing_tier: u32

### 3. Debugging System

```
TrinityDebugSystem
├── validation_enabled: bool
├── debug_visualization: DebugVisualization
├── resource_inspector: ResourceInspector
├── frame_capture: FrameCaptureSystem
└── profiler: Option<GPUProfiler>
```

**DebugVisualization (18 modes):**
None, Wireframe, Normals, Tangents, UVs, Albedo, Metallic, Roughness, AO, Depth, Stencil, MotionVectors, MipLevels, Overdraw, LightHeatmap, ShadowCascades, RTAccelerationStructure, Meshlets

**ResourceInspector:**
- inspect_texture() - Copy to CPU, analyze
- get_pixel() - Read single pixel
- dump_buffer() - Buffer contents to log

**FrameCaptureSystem:**
- capture_next_frame: bool
- capture_on_error: bool
- capture_on_slow_frame: bool
- renderdoc_integration: Option<RenderDocCapture>

### 4. Profiling System

```
TrinityProfiler
├── gpu_profiler: GPUProfiler
├── memory_tracker: MemoryTracker
├── draw_stats: DrawCallStats
├── bottleneck_analyzer: BottleneckAnalyzer
└── frame_history: VecDeque<FrameProfile>
```

**GPUProfiler:**
- begin_region() / end_region()
- resolve() / read_results()
- TimestampResult with duration_ns/ms

**MemoryTracker:**
- track_buffer() / track_texture()
- report() -> MemoryReport
- leak_detector: LeakDetector

**BottleneckAnalyzer:**
- analyze() -> Vec<Bottleneck>
- suggest_optimizations() -> Vec<&str>

### 5. Frame Graph

```
FrameGraph
├── passes: Vec<PassNode>
├── resources: HashMap<ResourceId, ResourceNode>
├── edges: Vec<ResourceEdge>
├── execution_order: Vec<usize>
├── barrier_resolver: BarrierResolver
├── transient_pool: TransientResourcePool
└── compiled: bool
```

**ResourceNode:**
- id: ResourceId
- name: String
- descriptor: ResourceDescriptor (Buffer | Texture)
- lifetime: ResourceLifetime (Transient | Persistent | External | Imported)
- physical: Option<PhysicalResource>

**PassNode:**
- id: PassId
- name: String
- pass_type: PassType (Render | Compute | RayTracing | Copy)
- reads: Vec<ResourceId>
- writes: Vec<ResourceId>
- execute: Box<dyn PassExecutor>

**Frame Graph API:**
```rust
let mut graph = FrameGraph::new();

// Declare resources
let gbuffer_albedo = graph.create_virtual_texture("GBuffer/Albedo", desc);
let gbuffer_normal = graph.create_virtual_texture("GBuffer/Normal", desc);
let depth = graph.create_virtual_texture("Depth", depth_desc);
let swapchain = graph.import_swapchain(&surface_texture, &view);

// Declare passes
graph.add_render_pass("GBuffer", config, |ctx| {
    ctx.set_pipeline(&gbuffer_pipeline);
    ctx.draw_scene(&scene);
});

graph.add_compute_pass("Lighting", config, |ctx| {
    ctx.set_pipeline(&lighting_pipeline);
    ctx.dispatch(width, height, 1);
});

graph.add_render_pass("Composite", config, |ctx| {
    ctx.blit(lighting_result, swapchain);
});

// Compile and execute
graph.compile();
graph.execute(&device, &queue);
```

### 6. Python Bridge

```
PyTrinity
├── renderer: PyRenderer
├── descriptors: [PyTextureDescriptor, PyBufferDescriptor, ...]
├── commands: [RenderPassBuilder, ComputePassBuilder, ...]
└── errors: [TrinityError, ValidationError, ...]
```

**PyRenderer:**
- create_texture() -> PyResourceHandle
- create_buffer() -> PyResourceHandle
- begin_frame() -> PyFrame
- execute_frame()
- present()

**PyTextureDescriptor:**
- width, height: u32
- format: String
- usage: Vec<String>
- to_wgpu() -> TextureResourceDesc

**RenderPassBuilder (Python):**
- add_color_attachment()
- set_pipeline()
- set_bind_group()
- draw() / draw_indexed()
- build() -> PyRenderPass

**RecordedCommand (Rust):**
- BeginRenderPass, EndRenderPass
- SetPipeline, SetBindGroup
- Draw, DrawIndexed, Dispatch

---

## Module Structure

```
crates/renderer-backend/src/
├── presentation/
│   ├── mod.rs
│   ├── surface.rs         # TrinitySurface
│   ├── present_mode.rs    # Mode selection
│   ├── frame_pacing.rs    # FramePacer
│   └── headless.rs        # HeadlessRenderer
│
├── platform/
│   ├── mod.rs
│   ├── vulkan.rs          # VulkanFeatures
│   ├── metal.rs           # MetalCapabilities
│   ├── dx12.rs            # DX12Capabilities
│   ├── webgpu.rs          # WebGPULimitations
│   └── opengl.rs          # GLESOptimizations
│
├── debug/
│   ├── mod.rs
│   ├── visualization.rs   # DebugVisualization
│   ├── inspector.rs       # ResourceInspector
│   ├── capture.rs         # FrameCaptureSystem
│   └── profiler.rs        # GPUProfiler, etc.
│
├── frame_graph/
│   ├── mod.rs
│   ├── graph.rs           # FrameGraph
│   ├── resource.rs        # ResourceNode
│   ├── pass.rs            # PassNode
│   ├── barrier.rs         # BarrierResolver
│   ├── pool.rs            # TransientResourcePool
│   ├── aliasing.rs        # AliasingAnalyzer
│   └── scheduler.rs       # Pass scheduling
│
└── bridge/
    ├── mod.rs
    ├── renderer.rs        # PyRenderer
    ├── descriptors.rs     # PyTextureDescriptor, etc.
    ├── commands.rs        # Command builders
    ├── handles.rs         # PyResourceHandle
    └── errors.rs          # Error conversion
```

---

## Testing Strategy

### Unit Tests

1. **Surface configuration** - Format selection
2. **Present mode** - Mode availability
3. **Debug visualization** - Shader defines
4. **Frame graph** - Topological sort
5. **Barrier resolution** - State transitions
6. **Resource aliasing** - Lifetime analysis
7. **Python bindings** - Type conversion

### Integration Tests

1. **Full render loop** - Surface acquire -> render -> present
2. **Frame graph execution** - Multi-pass render
3. **Platform compatibility** - Backend-specific
4. **Profiler output** - Timing data
5. **Python API** - Full Python render script

### System Tests

1. **Multi-window** - Shared device
2. **Headless** - Offline rendering
3. **Hot-reload** - Surface resize
4. **Memory tracking** - Leak detection
5. **Performance** - Frame time targets

---

## Performance Considerations

1. **Surface Configuration** - Match display refresh
2. **Frame Pacing** - Consistent frame times
3. **Transient Pool** - Minimize allocations
4. **Barrier Batching** - Group by stage
5. **Python Bridge** - Batch commands

---

## Dependencies

### External Crates

- `wgpu` - Core GPU abstraction
- `pyo3` - Python bindings
- `raw-window-handle` - Window handle abstraction
- `renderdoc-sys` - RenderDoc integration (optional)

### Internal Dependencies

- All previous phases (1-6)

---

## Deliverables Checklist

**Presentation:**
- [ ] TrinitySurface with configuration
- [ ] Present mode selection
- [ ] Frame pacing
- [ ] Triple buffering
- [ ] HeadlessRenderer
- [ ] Multi-window support

**Platform:**
- [ ] VulkanFeatures detection
- [ ] MetalCapabilities detection
- [ ] DX12Capabilities detection
- [ ] WebGPULimitations handling
- [ ] Platform support matrix

**Debugging:**
- [ ] DebugVisualization (18 modes)
- [ ] ResourceInspector
- [ ] FrameCaptureSystem
- [ ] RenderDoc integration

**Profiling:**
- [ ] GPUProfiler
- [ ] MemoryTracker
- [ ] LeakDetector
- [ ] BottleneckAnalyzer
- [ ] DrawCallStats

**Frame Graph:**
- [ ] FrameGraph struct
- [ ] Virtual resource declaration
- [ ] Transient resource pool
- [ ] Pass declaration (4 types)
- [ ] Automatic barrier resolution
- [ ] Pass scheduling
- [ ] Resource aliasing
- [ ] Async compute overlap

**Python Bridge:**
- [ ] PyRenderer
- [ ] Descriptor classes
- [ ] Command builders
- [ ] Error propagation
- [ ] Complete Python API example

**Tests:**
- [ ] Unit tests
- [ ] Integration tests
- [ ] System tests
- [ ] Documentation

---

*End of PHASE_7_INTEGRATION_ARCH.md*

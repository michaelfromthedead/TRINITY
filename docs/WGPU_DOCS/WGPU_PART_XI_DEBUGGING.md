# WGPU_PART_XI_DEBUGGING.md — Debugging & Profiling

> **Scope**: Complete coverage of wgpu debugging features, external debugger integration, GPU profiling, and performance analysis
> **TRINITY Integration**: Debug visualization, profiling system, bottleneck analysis
> **wgpu Version**: 25.x+

---

# Chapter 21: Debugging

GPU debugging is fundamentally different from CPU debugging. Issues may be non-deterministic, difficult to reproduce, and visible only through corrupted output. wgpu provides built-in validation and integrates with external GPU debuggers.

---

## 21.1 wgpu Debugging Features

### 21.1.1 Validation (WGPU_VALIDATION)

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

**Validation Catches**:
- Invalid resource usage combinations
- Mismatched bind group layouts
- Shader binding errors
- Buffer/texture out-of-bounds access (where possible)
- Missing required state before draw/dispatch
- Invalid pipeline state combinations

**Environment Variable Control**:
```bash
# Enable validation
export WGPU_VALIDATION=1

# Enable debug output
export WGPU_DEBUG=1

# Vulkan validation layers
export VK_LAYER_PATH=/usr/share/vulkan/explicit_layers.d
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
```

### 21.1.2 Debug Markers and Groups

Debug groups create hierarchical regions visible in GPU capture tools:

```rust
impl TrinityCommandEncoder {
    pub fn begin_debug_group(&mut self, label: &str) {
        self.inner.push_debug_group(label);
        self.debug_stack_depth += 1;
    }
    
    pub fn end_debug_group(&mut self) {
        assert!(self.debug_stack_depth > 0, "Unbalanced debug groups");
        self.inner.pop_debug_group();
        self.debug_stack_depth -= 1;
    }
    
    pub fn insert_marker(&mut self, label: &str) {
        self.inner.insert_debug_marker(label);
    }
}

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
    fn drop(&mut self) {
        self.encoder.pop_debug_group();
    }
}

// Usage in render passes
impl<'a> TrinityRenderPass<'a> {
    pub fn debug_group(&mut self, label: &str, f: impl FnOnce(&mut Self)) {
        self.inner.push_debug_group(label);
        f(self);
        self.inner.pop_debug_group();
    }
}
```

**Frame Structure Example**:
```
Frame 42
├── Shadow Maps
│   ├── Cascade 0
│   ├── Cascade 1
│   └── Cascade 2
├── G-Buffer Pass
│   ├── Opaque Objects
│   └── Alpha Tested
├── Lighting
│   ├── Directional Light
│   └── Point Lights
├── Post Processing
│   ├── Bloom
│   ├── Tone Mapping
│   └── FXAA
└── UI
```

### 21.1.3 Object Labels

Every wgpu resource can be labeled for easier debugging:

```rust
pub struct LabeledResourceFactory<'a> {
    device: &'a wgpu::Device,
    prefix: String,
}

impl<'a> LabeledResourceFactory<'a> {
    pub fn new(device: &'a wgpu::Device, prefix: &str) -> Self {
        Self {
            device,
            prefix: prefix.to_string(),
        }
    }
    
    pub fn create_buffer(&self, name: &str, desc: &wgpu::BufferDescriptor) -> wgpu::Buffer {
        let mut labeled_desc = desc.clone();
        let label = format!("{}_{}", self.prefix, name);
        labeled_desc.label = Some(&label);
        self.device.create_buffer(&labeled_desc)
    }
    
    pub fn create_texture(&self, name: &str, desc: &wgpu::TextureDescriptor) -> wgpu::Texture {
        let label = format!("{}_{}", self.prefix, name);
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&label),
            ..*desc
        })
    }
}

// Automatic labeling convention
pub fn auto_label<T>(value: &T, type_hint: &str) -> String
where
    T: std::fmt::Debug,
{
    format!("{}_{:?}", type_hint, std::any::type_name::<T>())
}
```

### 21.1.4 Error Scopes

Error scopes provide fine-grained error handling:

```rust
impl TrinityDevice {
    pub async fn with_error_scope<T>(
        &self,
        filter: wgpu::ErrorFilter,
        operation: impl FnOnce() -> T,
    ) -> Result<T, wgpu::Error> {
        self.inner.push_error_scope(filter);
        let result = operation();
        
        match self.inner.pop_error_scope().await {
            Some(error) => Err(error),
            None => Ok(result),
        }
    }
    
    pub fn begin_error_scope(&self, filter: wgpu::ErrorFilter) {
        self.inner.push_error_scope(filter);
    }
    
    pub async fn end_error_scope(&self) -> Option<wgpu::Error> {
        self.inner.pop_error_scope().await
    }
}

// Error filters
pub enum ErrorFilter {
    Validation,    // Catch validation errors
    OutOfMemory,   // Catch OOM
}

// Usage example
async fn safe_texture_create(device: &TrinityDevice) -> Result<wgpu::Texture, wgpu::Error> {
    device.with_error_scope(wgpu::ErrorFilter::OutOfMemory, || {
        device.inner().create_texture(&wgpu::TextureDescriptor {
            label: Some("LargeTexture"),
            size: wgpu::Extent3d {
                width: 16384,
                height: 16384,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }).await
}
```

---

## 21.2 External Debuggers

### 21.2.1 RenderDoc Integration

RenderDoc is the primary GPU debugger for Vulkan and DX12:

```rust
#[cfg(feature = "renderdoc")]
pub mod renderdoc_integration {
    use renderdoc::{RenderDoc, V141};
    
    pub struct RenderDocCapture {
        api: Option<RenderDoc<V141>>,
        capture_pending: bool,
    }
    
    impl RenderDocCapture {
        pub fn new() -> Self {
            let api = RenderDoc::new().ok();
            
            if api.is_some() {
                println!("RenderDoc attached");
            }
            
            Self {
                api,
                capture_pending: false,
            }
        }
        
        pub fn is_attached(&self) -> bool {
            self.api.is_some()
        }
        
        pub fn trigger_capture(&mut self) {
            if let Some(ref mut api) = self.api {
                api.trigger_capture();
            }
        }
        
        pub fn start_frame_capture(&mut self) {
            if let Some(ref mut api) = self.api {
                api.start_frame_capture(std::ptr::null(), std::ptr::null());
                self.capture_pending = true;
            }
        }
        
        pub fn end_frame_capture(&mut self) {
            if let Some(ref mut api) = self.api {
                if self.capture_pending {
                    api.end_frame_capture(std::ptr::null(), std::ptr::null());
                    self.capture_pending = false;
                }
            }
        }
        
        pub fn get_capture_path(&self) -> Option<String> {
            self.api.as_ref().and_then(|api| {
                let path = api.get_capture_file_path_template();
                Some(path.to_string_lossy().into_owned())
            })
        }
        
        pub fn launch_replay_ui(&self) {
            if let Some(ref api) = self.api {
                api.launch_replay_ui(true, None).ok();
            }
        }
    }
    
    impl Drop for RenderDocCapture {
        fn drop(&mut self) {
            if self.capture_pending {
                self.end_frame_capture();
            }
        }
    }
}
```

**RenderDoc Keyboard Shortcuts**:
- F12: Capture frame
- Ctrl+F12: Capture multiple frames
- F11: Cycle overlay

### 21.2.2 PIX for Windows

PIX provides deep DX12 debugging on Windows:

```rust
#[cfg(all(windows, feature = "pix"))]
pub mod pix_integration {
    pub fn begin_event(context: &wgpu::CommandEncoder, color: u64, name: &str) {
        // PIX markers are automatically inserted when using wgpu debug groups
        // on DX12 backend
    }
    
    pub fn set_marker(name: &str) {
        // Marker inserted
    }
    
    pub fn end_event() {
        // Event ended
    }
}
```

**PIX Features**:
- GPU capture with full pipeline inspection
- Timing analysis
- Memory debugging
- Shader debugging with source correlation

### 21.2.3 Xcode GPU Frame Capture

On macOS/iOS, Xcode provides Metal debugging:

```rust
#[cfg(target_os = "macos")]
pub mod metal_debug {
    pub fn enable_metal_validation() {
        // Set via scheme in Xcode:
        // Product > Scheme > Edit Scheme > Run > Diagnostics
        // Enable "Metal API Validation"
        // Enable "GPU Frame Capture"
    }
    
    pub fn capture_scope(label: &str) {
        // Metal capture scopes map to wgpu debug groups
    }
}
```

**Xcode GPU Debug Features**:
- Metal frame capture
- Shader debugger
- GPU timeline
- Memory viewer
- Dependency viewer

### 21.2.4 NVIDIA Nsight Graphics

Nsight provides deep NVIDIA GPU analysis:

```rust
pub struct NsightMarkers;

impl NsightMarkers {
    pub fn push_range(name: &str) {
        // Nsight picks up wgpu debug groups automatically on Vulkan
    }
    
    pub fn pop_range() {
        // Range ended
    }
}
```

**Nsight Features**:
- Ray tracing debugging
- Shader profiling
- Memory analysis
- Warp state inspection

### 21.2.5 AMD Radeon GPU Profiler

RGP provides AMD-specific profiling:

```rust
pub struct RGPMarkers;

impl RGPMarkers {
    pub fn push_marker(name: &str) {
        // RGP uses Vulkan debug markers
    }
    
    pub fn pop_marker() {
        // Marker ended
    }
}
```

**RGP Features**:
- Wavefront occupancy
- Cache analysis
- Barrier analysis
- Instruction timing

---

## 21.3 TRINITY's Debug System

### 21.3.1 Debug Visualization Modes

```rust
pub enum DebugVisualization {
    None,
    Wireframe,
    Normals,
    Tangents,
    UVs,
    Albedo,
    Metallic,
    Roughness,
    AO,
    Depth,
    Stencil,
    MotionVectors,
    MipLevels,
    Overdraw,
    LightHeatmap,
    ShadowCascades,
    RTAccelerationStructure,
    Meshlets,
}

impl DebugVisualization {
    pub fn shader_define(&self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Wireframe => Some("DEBUG_WIREFRAME"),
            Self::Normals => Some("DEBUG_NORMALS"),
            Self::Tangents => Some("DEBUG_TANGENTS"),
            Self::UVs => Some("DEBUG_UVS"),
            Self::Albedo => Some("DEBUG_ALBEDO"),
            Self::Metallic => Some("DEBUG_METALLIC"),
            Self::Roughness => Some("DEBUG_ROUGHNESS"),
            Self::AO => Some("DEBUG_AO"),
            Self::Depth => Some("DEBUG_DEPTH"),
            Self::Stencil => Some("DEBUG_STENCIL"),
            Self::MotionVectors => Some("DEBUG_MOTION"),
            Self::MipLevels => Some("DEBUG_MIPS"),
            Self::Overdraw => Some("DEBUG_OVERDRAW"),
            Self::LightHeatmap => Some("DEBUG_LIGHT_HEAT"),
            Self::ShadowCascades => Some("DEBUG_CASCADES"),
            Self::RTAccelerationStructure => Some("DEBUG_RT_AS"),
            Self::Meshlets => Some("DEBUG_MESHLETS"),
        }
    }
}

pub struct DebugRenderer {
    current_mode: DebugVisualization,
    debug_pipelines: HashMap<DebugVisualization, wgpu::RenderPipeline>,
    overdraw_buffer: Option<wgpu::Buffer>,
    heatmap_texture: Option<wgpu::Texture>,
}

impl DebugRenderer {
    pub fn set_mode(&mut self, mode: DebugVisualization) {
        self.current_mode = mode;
    }
    
    pub fn render_debug_overlay<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        scene: &'a SceneBindings,
    ) {
        if self.current_mode == DebugVisualization::None {
            return;
        }
        
        if let Some(pipeline) = self.debug_pipelines.get(&self.current_mode) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &scene.bind_group, &[]);
            pass.draw(0..3, 0..1); // Fullscreen triangle
        }
    }
}
```

**Debug Shader Example (Normals)**:
```wgsl
@group(0) @binding(0) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(1) var point_sampler: sampler;

@fragment
fn debug_normals(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let normal = textureSample(gbuffer_normal, point_sampler, uv).xyz;
    // Map [-1,1] to [0,1] for visualization
    let color = normal * 0.5 + 0.5;
    return vec4<f32>(color, 1.0);
}
```

### 21.3.2 Resource Inspection

```rust
pub struct ResourceInspector {
    selected_resource: Option<ResourceHandle>,
    inspection_buffer: wgpu::Buffer,
    inspection_data: Vec<u8>,
}

impl ResourceInspector {
    pub fn inspect_buffer(&mut self, buffer: &TrinityBuffer, offset: u64, size: u64) {
        // Copy to staging for readback
    }
    
    pub fn inspect_texture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &TrinityTexture,
        mip: u32,
        layer: u32,
    ) {
        let extent = texture.mip_extent(mip);
        
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: texture.inner(),
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.inspection_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(extent.width * 4),
                    rows_per_image: None,
                },
            },
            extent,
        );
    }
    
    pub fn get_pixel(&self, x: u32, y: u32, width: u32) -> [f32; 4] {
        let offset = ((y * width + x) * 4) as usize;
        let bytes = &self.inspection_data[offset..offset + 16];
        [
            f32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            f32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            f32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        ]
    }
}
```

### 21.3.3 Pipeline State Dump

```rust
pub struct PipelineStateDump {
    pub pipeline_name: String,
    pub vertex_buffers: Vec<VertexBufferState>,
    pub index_buffer: Option<IndexBufferState>,
    pub bind_groups: Vec<BindGroupState>,
    pub viewport: (f32, f32, f32, f32),
    pub scissor: Option<(u32, u32, u32, u32)>,
    pub blend_constant: [f32; 4],
    pub stencil_reference: u32,
}

impl PipelineStateDump {
    pub fn capture(pass_state: &RenderPassState) -> Self {
        Self {
            pipeline_name: pass_state.pipeline_name.clone(),
            vertex_buffers: pass_state.vertex_buffers.clone(),
            index_buffer: pass_state.index_buffer.clone(),
            bind_groups: pass_state.bind_groups.clone(),
            viewport: pass_state.viewport,
            scissor: pass_state.scissor,
            blend_constant: pass_state.blend_constant,
            stencil_reference: pass_state.stencil_reference,
        }
    }
    
    pub fn log(&self) {
        println!("Pipeline: {}", self.pipeline_name);
        println!("Viewport: {:?}", self.viewport);
        if let Some(scissor) = self.scissor {
            println!("Scissor: {:?}", scissor);
        }
        println!("Vertex Buffers: {}", self.vertex_buffers.len());
        for (i, vb) in self.vertex_buffers.iter().enumerate() {
            println!("  [{}] offset={}, size={}", i, vb.offset, vb.size);
        }
        if let Some(ref ib) = self.index_buffer {
            println!("Index Buffer: format={:?}, offset={}", ib.format, ib.offset);
        }
        println!("Bind Groups: {}", self.bind_groups.len());
    }
}
```

### 21.3.4 Frame Capture Triggers

```rust
pub struct FrameCaptureSystem {
    renderdoc: Option<RenderDocCapture>,
    capture_next_frame: bool,
    capture_on_error: bool,
    capture_on_slow_frame: bool,
    slow_frame_threshold_ms: f32,
}

impl FrameCaptureSystem {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "renderdoc")]
            renderdoc: Some(RenderDocCapture::new()),
            #[cfg(not(feature = "renderdoc"))]
            renderdoc: None,
            capture_next_frame: false,
            capture_on_error: true,
            capture_on_slow_frame: false,
            slow_frame_threshold_ms: 33.3, // 30 FPS threshold
        }
    }
    
    pub fn request_capture(&mut self) {
        self.capture_next_frame = true;
    }
    
    pub fn begin_frame(&mut self) {
        if self.capture_next_frame {
            if let Some(ref mut rd) = self.renderdoc {
                rd.start_frame_capture();
            }
        }
    }
    
    pub fn end_frame(&mut self, frame_time_ms: f32, had_error: bool) {
        let should_capture = self.capture_next_frame
            || (self.capture_on_error && had_error)
            || (self.capture_on_slow_frame && frame_time_ms > self.slow_frame_threshold_ms);
        
        if self.capture_next_frame {
            if let Some(ref mut rd) = self.renderdoc {
                rd.end_frame_capture();
            }
            self.capture_next_frame = false;
        }
        
        if should_capture && !self.capture_next_frame {
            // Capture was triggered by error/slow frame
            if let Some(ref mut rd) = self.renderdoc {
                rd.trigger_capture();
            }
        }
    }
    
    pub fn on_key_press(&mut self, key: KeyCode) {
        match key {
            KeyCode::F12 => self.request_capture(),
            KeyCode::F11 => {
                // Toggle overlay if RenderDoc attached
            }
            _ => {}
        }
    }
}
```

---

# Chapter 22: Profiling

GPU profiling measures execution time, resource usage, and identifies performance bottlenecks.

---

## 22.1 GPU Timing

### 22.1.1 Timestamp Queries

```rust
pub struct GPUProfiler {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    timestamp_period: f32,
    
    regions: Vec<ProfileRegion>,
    next_query_index: u32,
    max_queries: u32,
    
    results: Vec<ProfileResult>,
}

pub struct ProfileRegion {
    name: String,
    start_query: u32,
    end_query: u32,
}

pub struct ProfileResult {
    pub name: String,
    pub duration_ns: f64,
    pub duration_ms: f64,
}

impl GPUProfiler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, max_regions: u32) -> Self {
        let max_queries = max_regions * 2;
        
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("GPUProfiler"),
            ty: wgpu::QueryType::Timestamp,
            count: max_queries,
        });
        
        let buffer_size = max_queries as u64 * 8;
        
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ProfilerResolve"),
            size: buffer_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ProfilerReadback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        let timestamp_period = queue.get_timestamp_period();
        
        Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            timestamp_period,
            regions: Vec::new(),
            next_query_index: 0,
            max_queries,
            results: Vec::new(),
        }
    }
    
    pub fn begin_region(&mut self, pass: &mut impl TimestampWriter, name: &str) -> ProfileRegion {
        let start_query = self.next_query_index;
        pass.write_timestamp(&self.query_set, start_query);
        self.next_query_index += 1;
        
        ProfileRegion {
            name: name.to_string(),
            start_query,
            end_query: 0,
        }
    }
    
    pub fn end_region(&mut self, pass: &mut impl TimestampWriter, mut region: ProfileRegion) {
        region.end_query = self.next_query_index;
        pass.write_timestamp(&self.query_set, region.end_query);
        self.next_query_index += 1;
        self.regions.push(region);
    }
    
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.next_query_index == 0 {
            return;
        }
        
        encoder.resolve_query_set(
            &self.query_set,
            0..self.next_query_index,
            &self.resolve_buffer,
            0,
        );
        
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            self.next_query_index as u64 * 8,
        );
    }
    
    pub async fn read_results(&mut self, device: &wgpu::Device) {
        let slice = self.readback_buffer.slice(..);
        
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);
        
        self.results.clear();
        for region in &self.regions {
            let start = timestamps[region.start_query as usize];
            let end = timestamps[region.end_query as usize];
            let duration_ticks = end.saturating_sub(start);
            let duration_ns = duration_ticks as f64 * self.timestamp_period as f64;
            
            self.results.push(ProfileResult {
                name: region.name.clone(),
                duration_ns,
                duration_ms: duration_ns / 1_000_000.0,
            });
        }
        
        drop(data);
        self.readback_buffer.unmap();
        
        // Reset for next frame
        self.regions.clear();
        self.next_query_index = 0;
    }
    
    pub fn print_results(&self) {
        println!("=== GPU Profile Results ===");
        let mut total_ms = 0.0;
        for result in &self.results {
            println!("  {}: {:.3} ms", result.name, result.duration_ms);
            total_ms += result.duration_ms;
        }
        println!("  TOTAL: {:.3} ms", total_ms);
    }
}

pub trait TimestampWriter {
    fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, index: u32);
}

impl TimestampWriter for wgpu::RenderPass<'_> {
    fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, index: u32) {
        self.write_timestamp(query_set, index);
    }
}

impl TimestampWriter for wgpu::ComputePass<'_> {
    fn write_timestamp(&mut self, query_set: &wgpu::QuerySet, index: u32) {
        self.write_timestamp(query_set, index);
    }
}
```

### 22.1.2 Pass Timing

```rust
pub struct PassTimer<'a> {
    profiler: &'a mut GPUProfiler,
    region: ProfileRegion,
}

impl<'a> PassTimer<'a> {
    pub fn new(profiler: &'a mut GPUProfiler, pass: &mut impl TimestampWriter, name: &str) -> Self {
        let region = profiler.begin_region(pass, name);
        Self { profiler, region }
    }
}

// Usage pattern with render pass timestamps
pub struct TimedRenderPass<'a> {
    pass: wgpu::RenderPass<'a>,
    profiler: &'a mut GPUProfiler,
    pass_name: String,
}

impl<'a> TimedRenderPass<'a> {
    pub fn new(
        encoder: &'a mut wgpu::CommandEncoder,
        desc: &wgpu::RenderPassDescriptor<'a, '_>,
        profiler: &'a mut GPUProfiler,
    ) -> Self {
        let pass = encoder.begin_render_pass(desc);
        let pass_name = desc.label.unwrap_or("Unnamed").to_string();
        
        // Note: timestamp is written at pass start via timestamp_writes
        
        Self {
            pass,
            profiler,
            pass_name,
        }
    }
}
```

### 22.1.3 Compute Kernel Timing

```rust
pub struct ComputeTimer {
    start_time: std::time::Instant,
    gpu_start_query: Option<u32>,
}

impl ComputeTimer {
    pub fn start(profiler: &mut GPUProfiler, pass: &mut wgpu::ComputePass) -> Self {
        let gpu_start = profiler.next_query_index;
        pass.write_timestamp(&profiler.query_set, gpu_start);
        profiler.next_query_index += 1;
        
        Self {
            start_time: std::time::Instant::now(),
            gpu_start_query: Some(gpu_start),
        }
    }
    
    pub fn stop(self, profiler: &mut GPUProfiler, pass: &mut wgpu::ComputePass, name: &str) {
        if let Some(start) = self.gpu_start_query {
            let end = profiler.next_query_index;
            pass.write_timestamp(&profiler.query_set, end);
            profiler.next_query_index += 1;
            
            profiler.regions.push(ProfileRegion {
                name: name.to_string(),
                start_query: start,
                end_query: end,
            });
        }
    }
}
```

### 22.1.4 Timer Resolution

```rust
impl GPUProfiler {
    pub fn timer_resolution_ns(&self) -> f32 {
        self.timestamp_period
    }
    
    pub fn is_high_resolution(&self) -> bool {
        // Better than 1 microsecond resolution
        self.timestamp_period < 1000.0
    }
    
    pub fn estimate_minimum_measurable_time_ms(&self) -> f32 {
        // Approximately 3x timer resolution for reliable measurement
        (self.timestamp_period * 3.0) / 1_000_000.0
    }
}
```

---

## 22.2 Pipeline Statistics

### 22.2.1 Statistics Queries

```rust
// Feature: PIPELINE_STATISTICS_QUERY
pub struct PipelineStats {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    max_queries: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PipelineStatistics {
    pub vertex_shader_invocations: u64,
    pub clipper_invocations: u64,
    pub clipper_primitives_out: u64,
    pub fragment_shader_invocations: u64,
    pub compute_shader_invocations: u64,
}

impl PipelineStats {
    pub fn new(device: &wgpu::Device, max_queries: u32) -> Option<Self> {
        if !device.features().contains(wgpu::Features::PIPELINE_STATISTICS_QUERY) {
            return None;
        }
        
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("PipelineStats"),
            ty: wgpu::QueryType::PipelineStatistics(
                wgpu::PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS
                    | wgpu::PipelineStatisticsTypes::CLIPPER_INVOCATIONS
                    | wgpu::PipelineStatisticsTypes::CLIPPER_PRIMITIVES_OUT
                    | wgpu::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS
                    | wgpu::PipelineStatisticsTypes::COMPUTE_SHADER_INVOCATIONS,
            ),
            count: max_queries,
        });
        
        let stat_size = std::mem::size_of::<PipelineStatistics>() as u64;
        let buffer_size = stat_size * max_queries as u64;
        
        Some(Self {
            query_set,
            resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("PipelineStatsResolve"),
                size: buffer_size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("PipelineStatsReadback"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            max_queries,
        })
    }
}
```

### 22.2.2 Invocation Counts

```rust
impl PipelineStatistics {
    pub fn vertex_to_fragment_ratio(&self) -> f64 {
        if self.vertex_shader_invocations == 0 {
            return 0.0;
        }
        self.fragment_shader_invocations as f64 / self.vertex_shader_invocations as f64
    }
    
    pub fn overdraw_estimate(&self, screen_pixels: u64) -> f64 {
        if screen_pixels == 0 {
            return 0.0;
        }
        self.fragment_shader_invocations as f64 / screen_pixels as f64
    }
    
    pub fn culling_efficiency(&self) -> f64 {
        if self.clipper_invocations == 0 {
            return 1.0;
        }
        self.clipper_primitives_out as f64 / self.clipper_invocations as f64
    }
}
```

---

## 22.3 Memory Profiling

### 22.3.1 Resource Memory Tracking

```rust
pub struct MemoryTracker {
    allocations: HashMap<ResourceId, AllocationInfo>,
    total_buffer_memory: u64,
    total_texture_memory: u64,
    peak_buffer_memory: u64,
    peak_texture_memory: u64,
}

pub struct AllocationInfo {
    pub name: String,
    pub size: u64,
    pub resource_type: ResourceType,
    pub timestamp: std::time::Instant,
}

pub enum ResourceType {
    Buffer,
    Texture,
    AccelerationStructure,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            total_buffer_memory: 0,
            total_texture_memory: 0,
            peak_buffer_memory: 0,
            peak_texture_memory: 0,
        }
    }
    
    pub fn track_buffer(&mut self, id: ResourceId, name: &str, size: u64) {
        self.allocations.insert(id, AllocationInfo {
            name: name.to_string(),
            size,
            resource_type: ResourceType::Buffer,
            timestamp: std::time::Instant::now(),
        });
        
        self.total_buffer_memory += size;
        self.peak_buffer_memory = self.peak_buffer_memory.max(self.total_buffer_memory);
    }
    
    pub fn track_texture(&mut self, id: ResourceId, name: &str, desc: &wgpu::TextureDescriptor) {
        let size = Self::estimate_texture_size(desc);
        
        self.allocations.insert(id, AllocationInfo {
            name: name.to_string(),
            size,
            resource_type: ResourceType::Texture,
            timestamp: std::time::Instant::now(),
        });
        
        self.total_texture_memory += size;
        self.peak_texture_memory = self.peak_texture_memory.max(self.total_texture_memory);
    }
    
    pub fn untrack(&mut self, id: ResourceId) {
        if let Some(info) = self.allocations.remove(&id) {
            match info.resource_type {
                ResourceType::Buffer => self.total_buffer_memory -= info.size,
                ResourceType::Texture => self.total_texture_memory -= info.size,
                ResourceType::AccelerationStructure => {} // Track separately
            }
        }
    }
    
    fn estimate_texture_size(desc: &wgpu::TextureDescriptor) -> u64 {
        let block_size = desc.format.block_copy_size(None).unwrap_or(4) as u64;
        let block_dims = desc.format.block_dimensions();
        
        let mut total = 0u64;
        let mut width = desc.size.width;
        let mut height = desc.size.height;
        let depth = desc.size.depth_or_array_layers;
        
        for _ in 0..desc.mip_level_count {
            let blocks_wide = (width + block_dims.0 - 1) / block_dims.0;
            let blocks_high = (height + block_dims.1 - 1) / block_dims.1;
            total += blocks_wide as u64 * blocks_high as u64 * depth as u64 * block_size;
            
            width = (width / 2).max(1);
            height = (height / 2).max(1);
        }
        
        total * desc.sample_count as u64
    }
    
    pub fn report(&self) -> MemoryReport {
        MemoryReport {
            total_buffer_mb: self.total_buffer_memory as f64 / (1024.0 * 1024.0),
            total_texture_mb: self.total_texture_memory as f64 / (1024.0 * 1024.0),
            peak_buffer_mb: self.peak_buffer_memory as f64 / (1024.0 * 1024.0),
            peak_texture_mb: self.peak_texture_memory as f64 / (1024.0 * 1024.0),
            allocation_count: self.allocations.len(),
            largest_allocations: self.top_allocations(10),
        }
    }
    
    fn top_allocations(&self, n: usize) -> Vec<(String, u64)> {
        let mut sorted: Vec<_> = self.allocations.values()
            .map(|a| (a.name.clone(), a.size))
            .collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(n);
        sorted
    }
}

pub struct MemoryReport {
    pub total_buffer_mb: f64,
    pub total_texture_mb: f64,
    pub peak_buffer_mb: f64,
    pub peak_texture_mb: f64,
    pub allocation_count: usize,
    pub largest_allocations: Vec<(String, u64)>,
}
```

### 22.3.2 Memory Budget Monitoring

```rust
pub struct MemoryBudget {
    buffer_budget_mb: f64,
    texture_budget_mb: f64,
    tracker: MemoryTracker,
}

impl MemoryBudget {
    pub fn new(buffer_budget_mb: f64, texture_budget_mb: f64) -> Self {
        Self {
            buffer_budget_mb,
            texture_budget_mb,
            tracker: MemoryTracker::new(),
        }
    }
    
    pub fn check_buffer_budget(&self, additional_size: u64) -> BudgetStatus {
        let current_mb = self.tracker.total_buffer_memory as f64 / (1024.0 * 1024.0);
        let additional_mb = additional_size as f64 / (1024.0 * 1024.0);
        let total = current_mb + additional_mb;
        
        if total > self.buffer_budget_mb {
            BudgetStatus::Exceeded {
                current_mb,
                budget_mb: self.buffer_budget_mb,
                overage_mb: total - self.buffer_budget_mb,
            }
        } else if total > self.buffer_budget_mb * 0.9 {
            BudgetStatus::Warning {
                current_mb,
                budget_mb: self.buffer_budget_mb,
                percent_used: (total / self.buffer_budget_mb) * 100.0,
            }
        } else {
            BudgetStatus::Ok {
                current_mb,
                budget_mb: self.buffer_budget_mb,
            }
        }
    }
}

pub enum BudgetStatus {
    Ok { current_mb: f64, budget_mb: f64 },
    Warning { current_mb: f64, budget_mb: f64, percent_used: f64 },
    Exceeded { current_mb: f64, budget_mb: f64, overage_mb: f64 },
}
```

### 22.3.3 Memory Leak Detection

```rust
pub struct LeakDetector {
    frame_allocations: HashMap<u64, Vec<ResourceId>>,
    persistent_allocations: HashSet<ResourceId>,
    frame_count: u64,
    leak_threshold_frames: u64,
}

impl LeakDetector {
    pub fn new(leak_threshold_frames: u64) -> Self {
        Self {
            frame_allocations: HashMap::new(),
            persistent_allocations: HashSet::new(),
            frame_count: 0,
            leak_threshold_frames,
        }
    }
    
    pub fn on_allocate(&mut self, id: ResourceId) {
        self.frame_allocations
            .entry(self.frame_count)
            .or_default()
            .push(id);
    }
    
    pub fn on_deallocate(&mut self, id: ResourceId) {
        // Remove from all frames
        for allocations in self.frame_allocations.values_mut() {
            allocations.retain(|&x| x != id);
        }
        self.persistent_allocations.remove(&id);
    }
    
    pub fn end_frame(&mut self) -> Vec<LeakWarning> {
        self.frame_count += 1;
        
        let mut warnings = Vec::new();
        
        // Check for resources that have lived too long
        let threshold_frame = self.frame_count.saturating_sub(self.leak_threshold_frames);
        
        for (&frame, allocations) in &self.frame_allocations {
            if frame < threshold_frame {
                for &id in allocations {
                    if !self.persistent_allocations.contains(&id) {
                        warnings.push(LeakWarning {
                            resource_id: id,
                            allocated_frame: frame,
                            current_frame: self.frame_count,
                        });
                        self.persistent_allocations.insert(id);
                    }
                }
            }
        }
        
        // Cleanup old frame data
        self.frame_allocations.retain(|&frame, _| frame >= threshold_frame);
        
        warnings
    }
}

pub struct LeakWarning {
    pub resource_id: ResourceId,
    pub allocated_frame: u64,
    pub current_frame: u64,
}
```

---

## 22.4 TRINITY's Profiling System

### 22.4.1 Per-Pass Timing

```rust
pub struct FrameProfiler {
    gpu_profiler: GPUProfiler,
    cpu_timers: HashMap<String, std::time::Instant>,
    frame_results: FrameProfileResults,
    history: VecDeque<FrameProfileResults>,
    max_history: usize,
}

pub struct FrameProfileResults {
    pub frame_number: u64,
    pub gpu_times: Vec<ProfileResult>,
    pub cpu_times: HashMap<String, f64>,
    pub total_gpu_ms: f64,
    pub total_cpu_ms: f64,
}

impl FrameProfiler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            gpu_profiler: GPUProfiler::new(device, queue, 64),
            cpu_timers: HashMap::new(),
            frame_results: FrameProfileResults::default(),
            history: VecDeque::with_capacity(120),
            max_history: 120,
        }
    }
    
    pub fn begin_cpu_region(&mut self, name: &str) {
        self.cpu_timers.insert(name.to_string(), std::time::Instant::now());
    }
    
    pub fn end_cpu_region(&mut self, name: &str) {
        if let Some(start) = self.cpu_timers.remove(name) {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            self.frame_results.cpu_times.insert(name.to_string(), elapsed);
        }
    }
    
    pub fn begin_gpu_region(&mut self, pass: &mut impl TimestampWriter, name: &str) -> ProfileRegion {
        self.gpu_profiler.begin_region(pass, name)
    }
    
    pub fn end_gpu_region(&mut self, pass: &mut impl TimestampWriter, region: ProfileRegion) {
        self.gpu_profiler.end_region(pass, region);
    }
    
    pub async fn end_frame(&mut self, device: &wgpu::Device, frame_number: u64) {
        self.gpu_profiler.resolve(&mut self.encoder);
        self.gpu_profiler.read_results(device).await;
        
        self.frame_results.frame_number = frame_number;
        self.frame_results.gpu_times = self.gpu_profiler.results.clone();
        self.frame_results.total_gpu_ms = self.frame_results.gpu_times.iter()
            .map(|r| r.duration_ms)
            .sum();
        self.frame_results.total_cpu_ms = self.frame_results.cpu_times.values().sum();
        
        // Store in history
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(std::mem::take(&mut self.frame_results));
    }
    
    pub fn average_gpu_time(&self, region_name: &str) -> Option<f64> {
        let times: Vec<f64> = self.history.iter()
            .filter_map(|frame| {
                frame.gpu_times.iter()
                    .find(|r| r.name == region_name)
                    .map(|r| r.duration_ms)
            })
            .collect();
        
        if times.is_empty() {
            None
        } else {
            Some(times.iter().sum::<f64>() / times.len() as f64)
        }
    }
}
```

### 22.4.2 Memory Dashboard

```rust
pub struct MemoryDashboard {
    tracker: MemoryTracker,
    budget: MemoryBudget,
    history: VecDeque<MemorySnapshot>,
}

pub struct MemorySnapshot {
    pub timestamp: std::time::Instant,
    pub buffer_mb: f64,
    pub texture_mb: f64,
    pub total_mb: f64,
}

impl MemoryDashboard {
    pub fn render_imgui(&self, ui: &imgui::Ui) {
        // Render memory usage graphs
        // Show largest allocations
        // Display warnings
    }
    
    pub fn export_csv(&self, path: &std::path::Path) {
        // Export memory history for analysis
    }
}
```

### 22.4.3 Draw Call Statistics

```rust
pub struct DrawCallStats {
    pub draw_calls: u32,
    pub triangles: u64,
    pub vertices: u64,
    pub instances: u64,
    pub state_changes: u32,
    pub bind_group_changes: u32,
    pub pipeline_changes: u32,
}

impl DrawCallStats {
    pub fn efficiency_score(&self) -> f64 {
        if self.draw_calls == 0 {
            return 0.0;
        }
        
        let avg_tris_per_draw = self.triangles as f64 / self.draw_calls as f64;
        let state_change_ratio = self.state_changes as f64 / self.draw_calls as f64;
        
        // Higher is better
        (avg_tris_per_draw / 100.0) * (1.0 - state_change_ratio.min(1.0))
    }
}
```

### 22.4.4 Bottleneck Analysis

```rust
pub enum Bottleneck {
    CPUBound { cpu_ms: f64, gpu_ms: f64 },
    GPUBound { cpu_ms: f64, gpu_ms: f64 },
    VertexBound { vertex_time_ms: f64, fragment_time_ms: f64 },
    FragmentBound { vertex_time_ms: f64, fragment_time_ms: f64 },
    BandwidthBound { memory_throughput_gb_s: f64 },
    DrawCallBound { draw_calls: u32, avg_tris: f64 },
}

pub struct BottleneckAnalyzer {
    profiler: FrameProfiler,
    stats: DrawCallStats,
}

impl BottleneckAnalyzer {
    pub fn analyze(&self) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();
        
        let last_frame = self.profiler.history.back();
        if let Some(frame) = last_frame {
            // CPU vs GPU bound
            if frame.total_cpu_ms > frame.total_gpu_ms * 1.2 {
                bottlenecks.push(Bottleneck::CPUBound {
                    cpu_ms: frame.total_cpu_ms,
                    gpu_ms: frame.total_gpu_ms,
                });
            } else if frame.total_gpu_ms > frame.total_cpu_ms * 1.2 {
                bottlenecks.push(Bottleneck::GPUBound {
                    cpu_ms: frame.total_cpu_ms,
                    gpu_ms: frame.total_gpu_ms,
                });
            }
            
            // Draw call bound
            if self.stats.draw_calls > 1000 {
                let avg_tris = self.stats.triangles as f64 / self.stats.draw_calls as f64;
                if avg_tris < 100.0 {
                    bottlenecks.push(Bottleneck::DrawCallBound {
                        draw_calls: self.stats.draw_calls,
                        avg_tris,
                    });
                }
            }
        }
        
        bottlenecks
    }
    
    pub fn suggest_optimizations(&self) -> Vec<&'static str> {
        let bottlenecks = self.analyze();
        let mut suggestions = Vec::new();
        
        for bottleneck in bottlenecks {
            match bottleneck {
                Bottleneck::CPUBound { .. } => {
                    suggestions.push("Consider GPU-driven rendering");
                    suggestions.push("Batch draw calls with multi-draw");
                    suggestions.push("Use instancing for repeated geometry");
                }
                Bottleneck::DrawCallBound { .. } => {
                    suggestions.push("Merge small meshes");
                    suggestions.push("Use instancing");
                    suggestions.push("Implement GPU culling");
                }
                Bottleneck::FragmentBound { .. } => {
                    suggestions.push("Reduce overdraw with depth prepass");
                    suggestions.push("Use early-Z rejection");
                    suggestions.push("Simplify fragment shaders");
                }
                _ => {}
            }
        }
        
        suggestions
    }
}
```

---

# TRINITY Debugging & Profiling Summary

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

*End of WGPU_PART_XI_DEBUGGING.md*

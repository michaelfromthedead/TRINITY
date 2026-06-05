# WGPU_PART_IX_PRESENTATION.md — Surface & Swapchain

> **Scope**: Complete coverage of wgpu presentation model: surface creation, configuration, frame acquisition, and presentation
> **TRINITY Integration**: Presentation engine, window management, frame pacing
> **wgpu Version**: 25.x+

---

# Chapter 18: Surface & Swapchain

The presentation system connects GPU rendering to display output. In wgpu, this is abstracted through `Surface` — a platform-agnostic handle to a window's drawable area.

---

## 18.1 Surface

### 18.1.1 Surface Creation from Window Handle

Surfaces are created from raw window handles using the `raw-window-handle` crate:

```rust
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

impl Instance {
    pub unsafe fn create_surface<
        W: HasRawWindowHandle + HasRawDisplayHandle,
    >(
        &self,
        window: W,
    ) -> Result<Surface<'static>, CreateSurfaceError>;
    
    // Safe alternative when window lifetime is known
    pub fn create_surface<'window>(
        &self,
        target: impl Into<SurfaceTarget<'window>>,
    ) -> Result<Surface<'window>, CreateSurfaceError>;
}
```

**TRINITY Surface Creation**:

```rust
use winit::window::Window;

pub struct TrinitySurface<'window> {
    inner: wgpu::Surface<'window>,
    config: wgpu::SurfaceConfiguration,
    capabilities: wgpu::SurfaceCapabilities,
    size: (u32, u32),
}

impl<'window> TrinitySurface<'window> {
    pub fn new(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        window: &'window Window,
    ) -> Result<Self, CreateSurfaceError> {
        let inner = instance.create_surface(window)?;
        let capabilities = inner.get_capabilities(adapter);
        
        let size = window.inner_size();
        let size = (size.width.max(1), size.height.max(1));
        
        // Select optimal configuration
        let config = Self::create_config(&capabilities, size);
        
        Ok(Self {
            inner,
            config,
            capabilities,
            size,
        })
    }
    
    fn create_config(
        capabilities: &wgpu::SurfaceCapabilities,
        size: (u32, u32),
    ) -> wgpu::SurfaceConfiguration {
        // Prefer sRGB format
        let format = capabilities.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(capabilities.formats[0]);
        
        // Prefer Mailbox (low latency), fallback to Fifo (vsync)
        let present_mode = if capabilities.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        };
        
        // Prefer Opaque alpha
        let alpha_mode = if capabilities.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            capabilities.alpha_modes[0]
        };
        
        wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.0,
            height: size.1,
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        }
    }
}
```

### 18.1.2 Surface Capabilities Query

```rust
impl Surface<'_> {
    pub fn get_capabilities(&self, adapter: &Adapter) -> SurfaceCapabilities;
}

pub struct SurfaceCapabilities {
    pub formats: Vec<TextureFormat>,
    pub present_modes: Vec<PresentMode>,
    pub alpha_modes: Vec<CompositeAlphaMode>,
    pub usages: TextureUsages,
}
```

**Capability Inspection**:

```rust
impl TrinitySurface<'_> {
    pub fn log_capabilities(&self) {
        println!("Surface Capabilities:");
        println!("  Formats: {:?}", self.capabilities.formats);
        println!("  Present Modes: {:?}", self.capabilities.present_modes);
        println!("  Alpha Modes: {:?}", self.capabilities.alpha_modes);
        println!("  Usages: {:?}", self.capabilities.usages);
    }
    
    pub fn supports_hdr(&self) -> bool {
        self.capabilities.formats.iter().any(|f| {
            matches!(f, 
                wgpu::TextureFormat::Rgba16Float |
                wgpu::TextureFormat::Rgb10a2Unorm
            )
        })
    }
    
    pub fn supports_mailbox(&self) -> bool {
        self.capabilities.present_modes.contains(&wgpu::PresentMode::Mailbox)
    }
}
```

### 18.1.3 Supported Formats

Common surface formats by platform:

| Platform | Primary Format | HDR Format |
|----------|----------------|------------|
| Windows (DX12) | Bgra8UnormSrgb | Rgba16Float |
| Windows (Vulkan) | Bgra8UnormSrgb | Rgba16Float |
| macOS (Metal) | Bgra8UnormSrgb | Rgba16Float |
| Linux (Vulkan) | Bgra8UnormSrgb | Rgba16Float |
| Web (WebGPU) | Bgra8Unorm | - |

```rust
pub fn select_surface_format(
    capabilities: &wgpu::SurfaceCapabilities,
    prefer_hdr: bool,
) -> wgpu::TextureFormat {
    if prefer_hdr {
        // Try HDR formats first
        for format in &capabilities.formats {
            if matches!(format, 
                wgpu::TextureFormat::Rgba16Float |
                wgpu::TextureFormat::Rgb10a2Unorm
            ) {
                return *format;
            }
        }
    }
    
    // Prefer sRGB
    for format in &capabilities.formats {
        if format.is_srgb() {
            return *format;
        }
    }
    
    // Fallback to first available
    capabilities.formats[0]
}
```

### 18.1.4 Supported Present Modes

```rust
pub enum PresentMode {
    AutoVsync,    // Platform chooses vsync mode
    AutoNoVsync,  // Platform chooses non-vsync mode
    Fifo,         // VSync, no tearing, may have latency
    FifoRelaxed,  // VSync with tearing if late
    Immediate,    // No VSync, may tear
    Mailbox,      // No VSync, no tearing, discards old frames
}
```

| Mode | VSync | Tearing | Latency | Use Case |
|------|-------|---------|---------|----------|
| Fifo | Yes | No | High | Standard display |
| FifoRelaxed | Partial | Possible | Medium | Smooth playback |
| Immediate | No | Yes | Lowest | Competitive gaming |
| Mailbox | No | No | Low | Low-latency gaming |

### 18.1.5 Supported Alpha Modes

```rust
pub enum CompositeAlphaMode {
    Auto,           // Platform chooses
    Opaque,         // Alpha ignored, fully opaque
    PreMultiplied,  // Alpha pre-multiplied
    PostMultiplied, // Alpha post-multiplied
    Inherit,        // Inherit from window
}
```

```rust
pub fn select_alpha_mode(
    capabilities: &wgpu::SurfaceCapabilities,
    window_transparent: bool,
) -> wgpu::CompositeAlphaMode {
    if window_transparent {
        // Need alpha blending for transparent windows
        if capabilities.alpha_modes.contains(&wgpu::CompositeAlphaMode::PreMultiplied) {
            return wgpu::CompositeAlphaMode::PreMultiplied;
        }
        if capabilities.alpha_modes.contains(&wgpu::CompositeAlphaMode::PostMultiplied) {
            return wgpu::CompositeAlphaMode::PostMultiplied;
        }
    }
    
    // Default to opaque
    if capabilities.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
        wgpu::CompositeAlphaMode::Opaque
    } else {
        capabilities.alpha_modes[0]
    }
}
```

---

## 18.2 Surface Configuration

### 18.2.1 Format Selection

```rust
impl TrinitySurface<'_> {
    pub fn configure_format(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        assert!(
            self.capabilities.formats.contains(&format),
            "Format {:?} not supported by surface",
            format
        );
        
        self.config.format = format;
        
        // Update view formats for sRGB toggle
        if format.is_srgb() {
            self.config.view_formats = vec![format.remove_srgb_suffix()];
        } else {
            self.config.view_formats = vec![format.add_srgb_suffix()];
        }
        
        self.inner.configure(device, &self.config);
    }
}
```

### 18.2.2 Present Mode Selection

```rust
impl TrinitySurface<'_> {
    pub fn set_vsync(&mut self, device: &wgpu::Device, enabled: bool) {
        self.config.present_mode = if enabled {
            wgpu::PresentMode::Fifo
        } else if self.supports_mailbox() {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Immediate
        };
        
        self.inner.configure(device, &self.config);
    }
    
    pub fn set_present_mode(&mut self, device: &wgpu::Device, mode: wgpu::PresentMode) {
        assert!(
            self.capabilities.present_modes.contains(&mode),
            "Present mode {:?} not supported",
            mode
        );
        
        self.config.present_mode = mode;
        self.inner.configure(device, &self.config);
    }
}
```

### 18.2.3 Alpha Mode Configuration

```rust
impl TrinitySurface<'_> {
    pub fn set_transparent(&mut self, device: &wgpu::Device, transparent: bool) {
        self.config.alpha_mode = select_alpha_mode(&self.capabilities, transparent);
        self.inner.configure(device, &self.config);
    }
}
```

### 18.2.4 Width and Height

```rust
impl TrinitySurface<'_> {
    pub fn resize(&mut self, device: &wgpu::Device, new_size: (u32, u32)) {
        // Clamp to valid range
        let width = new_size.0.max(1);
        let height = new_size.1.max(1);
        
        if self.size == (width, height) {
            return; // No change
        }
        
        self.size = (width, height);
        self.config.width = width;
        self.config.height = height;
        
        self.inner.configure(device, &self.config);
    }
    
    pub fn size(&self) -> (u32, u32) {
        self.size
    }
    
    pub fn aspect_ratio(&self) -> f32 {
        self.size.0 as f32 / self.size.1 as f32
    }
}
```

### 18.2.5 View Formats for sRGB Reinterpretation

```rust
impl TrinitySurface<'_> {
    pub fn configure_with_srgb_toggle(&mut self, device: &wgpu::Device) {
        // Allow both sRGB and linear views of the same texture
        let base_format = self.config.format;
        
        if base_format.is_srgb() {
            self.config.view_formats = vec![
                base_format,
                base_format.remove_srgb_suffix(),
            ];
        } else {
            self.config.view_formats = vec![
                base_format,
                base_format.add_srgb_suffix(),
            ];
        }
        
        self.inner.configure(device, &self.config);
    }
    
    pub fn create_linear_view(&self, texture: &wgpu::Texture) -> wgpu::TextureView {
        let linear_format = self.config.format.remove_srgb_suffix();
        texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(linear_format),
            ..Default::default()
        })
    }
}
```

---

## 18.3 Frame Acquisition

### 18.3.1 get_current_texture()

```rust
impl Surface<'_> {
    pub fn get_current_texture(&self) -> Result<SurfaceTexture, SurfaceError>;
}

pub struct SurfaceTexture {
    pub texture: Texture,
    pub suboptimal: bool,
    presented: bool,
}

pub enum SurfaceError {
    Timeout,     // Timed out waiting for frame
    Outdated,    // Surface config outdated (resize needed)
    Lost,        // Surface lost (need to recreate)
    OutOfMemory, // GPU OOM
}
```

**TRINITY Frame Acquisition**:

```rust
pub enum FrameAcquisitionResult<'a> {
    Success(Frame<'a>),
    Reconfigure,
    Lost,
    Error(SurfaceError),
}

pub struct Frame<'a> {
    texture: wgpu::SurfaceTexture,
    view: wgpu::TextureView,
    size: (u32, u32),
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'window> TrinitySurface<'window> {
    pub fn acquire_frame(&self) -> FrameAcquisitionResult<'_> {
        match self.inner.get_current_texture() {
            Ok(texture) => {
                if texture.suboptimal {
                    // Frame is usable but surface should be reconfigured
                    // Still proceed with rendering
                }
                
                let view = texture.texture.create_view(
                    &wgpu::TextureViewDescriptor::default()
                );
                
                FrameAcquisitionResult::Success(Frame {
                    texture,
                    view,
                    size: self.size,
                    _marker: std::marker::PhantomData,
                })
            }
            Err(wgpu::SurfaceError::Timeout) => {
                // Try again next frame
                FrameAcquisitionResult::Error(wgpu::SurfaceError::Timeout)
            }
            Err(wgpu::SurfaceError::Outdated) => {
                FrameAcquisitionResult::Reconfigure
            }
            Err(wgpu::SurfaceError::Lost) => {
                FrameAcquisitionResult::Lost
            }
            Err(e) => {
                FrameAcquisitionResult::Error(e)
            }
        }
    }
}
```

### 18.3.2 SurfaceTexture Handling

```rust
impl<'a> Frame<'a> {
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
    
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture.texture
    }
    
    pub fn size(&self) -> (u32, u32) {
        self.size
    }
    
    pub fn is_suboptimal(&self) -> bool {
        self.texture.suboptimal
    }
    
    pub fn present(self) {
        self.texture.present();
    }
}
```

### 18.3.3 Suboptimal and Outdated Surfaces

```rust
impl<'window> TrinitySurface<'window> {
    pub fn handle_resize(
        &mut self,
        device: &wgpu::Device,
        new_size: (u32, u32),
    ) -> Result<(), SurfaceError> {
        self.resize(device, new_size);
        Ok(())
    }
    
    pub fn try_acquire_or_reconfigure(
        &mut self,
        device: &wgpu::Device,
    ) -> Option<Frame<'_>> {
        loop {
            match self.acquire_frame() {
                FrameAcquisitionResult::Success(frame) => {
                    return Some(frame);
                }
                FrameAcquisitionResult::Reconfigure => {
                    // Reconfigure with current size
                    self.inner.configure(device, &self.config);
                    continue;
                }
                FrameAcquisitionResult::Lost => {
                    // Surface lost, needs recreation (rare)
                    return None;
                }
                FrameAcquisitionResult::Error(_) => {
                    return None;
                }
            }
        }
    }
}
```

### 18.3.4 Surface Reconfiguration on Resize

```rust
impl<'window> TrinitySurface<'window> {
    pub fn on_window_resize(
        &mut self,
        device: &wgpu::Device,
        physical_size: winit::dpi::PhysicalSize<u32>,
    ) {
        let width = physical_size.width.max(1);
        let height = physical_size.height.max(1);
        
        if self.size.0 == width && self.size.1 == height {
            return;
        }
        
        self.size = (width, height);
        self.config.width = width;
        self.config.height = height;
        
        self.inner.configure(device, &self.config);
        
        // Notify dependent systems
        // self.on_resize_callbacks.iter().for_each(|cb| cb(width, height));
    }
}
```

---

## 18.4 Presentation

### 18.4.1 present() Call

```rust
impl SurfaceTexture {
    pub fn present(self);
}
```

The `present()` call:
1. Schedules the texture for display
2. Consumes the `SurfaceTexture`
3. Returns control immediately (doesn't wait for vsync)

**Timing Implications**:
- Fifo: present blocks until vsync (in practice, queued)
- Mailbox: present returns immediately, old frame discarded
- Immediate: present returns immediately, may tear

### 18.4.2 Vsync and Frame Pacing

```rust
pub struct FramePacer {
    target_frame_time: Duration,
    last_frame_time: Instant,
    frame_times: VecDeque<Duration>,
    max_samples: usize,
}

impl FramePacer {
    pub fn new(target_fps: u32) -> Self {
        Self {
            target_frame_time: Duration::from_secs_f64(1.0 / target_fps as f64),
            last_frame_time: Instant::now(),
            frame_times: VecDeque::with_capacity(120),
            max_samples: 120,
        }
    }
    
    pub fn begin_frame(&mut self) -> FrameTime {
        let now = Instant::now();
        let delta = now - self.last_frame_time;
        self.last_frame_time = now;
        
        // Track frame time
        if self.frame_times.len() >= self.max_samples {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(delta);
        
        FrameTime {
            delta,
            delta_secs: delta.as_secs_f32(),
        }
    }
    
    pub fn average_frame_time(&self) -> Duration {
        if self.frame_times.is_empty() {
            return self.target_frame_time;
        }
        
        let sum: Duration = self.frame_times.iter().sum();
        sum / self.frame_times.len() as u32
    }
    
    pub fn fps(&self) -> f32 {
        let avg = self.average_frame_time();
        1.0 / avg.as_secs_f32()
    }
    
    pub fn frame_time_variance(&self) -> Duration {
        if self.frame_times.len() < 2 {
            return Duration::ZERO;
        }
        
        let avg = self.average_frame_time();
        let variance: f64 = self.frame_times.iter()
            .map(|&t| {
                let diff = t.as_secs_f64() - avg.as_secs_f64();
                diff * diff
            })
            .sum::<f64>() / self.frame_times.len() as f64;
        
        Duration::from_secs_f64(variance.sqrt())
    }
}

pub struct FrameTime {
    pub delta: Duration,
    pub delta_secs: f32,
}
```

### 18.4.3 Triple Buffering Strategies

```rust
pub struct TripleBufferConfig {
    pub max_frames_in_flight: u32,
    pub present_mode: wgpu::PresentMode,
}

impl TripleBufferConfig {
    pub fn low_latency() -> Self {
        Self {
            max_frames_in_flight: 2,
            present_mode: wgpu::PresentMode::Mailbox,
        }
    }
    
    pub fn smooth() -> Self {
        Self {
            max_frames_in_flight: 3,
            present_mode: wgpu::PresentMode::Fifo,
        }
    }
    
    pub fn uncapped() -> Self {
        Self {
            max_frames_in_flight: 2,
            present_mode: wgpu::PresentMode::Immediate,
        }
    }
}

impl<'window> TrinitySurface<'window> {
    pub fn apply_triple_buffer_config(
        &mut self,
        device: &wgpu::Device,
        config: TripleBufferConfig,
    ) {
        self.config.desired_maximum_frame_latency = config.max_frames_in_flight;
        
        if self.capabilities.present_modes.contains(&config.present_mode) {
            self.config.present_mode = config.present_mode;
        }
        
        self.inner.configure(device, &self.config);
    }
}
```

### 18.4.4 TRINITY's Presentation Engine

```rust
pub struct PresentationEngine<'window> {
    surface: TrinitySurface<'window>,
    frame_pacer: FramePacer,
    synchronizer: FrameSynchronizer,
    
    // Statistics
    frames_presented: u64,
    frames_dropped: u64,
    suboptimal_count: u32,
}

impl<'window> PresentationEngine<'window> {
    pub fn new(
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        window: &'window Window,
        config: PresentationConfig,
    ) -> Result<Self, CreateSurfaceError> {
        let mut surface = TrinitySurface::new(instance, adapter, window)?;
        surface.inner.configure(device, &surface.config);
        
        let frame_pacer = FramePacer::new(config.target_fps);
        let synchronizer = FrameSynchronizer::new(config.max_frames_in_flight);
        
        Ok(Self {
            surface,
            frame_pacer,
            synchronizer,
            frames_presented: 0,
            frames_dropped: 0,
            suboptimal_count: 0,
        })
    }
    
    pub fn begin_frame(&mut self, device: &wgpu::Device) -> Option<PresentFrame<'_>> {
        let frame_time = self.frame_pacer.begin_frame();
        
        // Wait for old frame to complete
        self.synchronizer.wait_for_frame_slot(device);
        
        // Acquire new frame
        let frame = self.surface.try_acquire_or_reconfigure(device)?;
        
        if frame.is_suboptimal() {
            self.suboptimal_count += 1;
        }
        
        Some(PresentFrame {
            frame,
            frame_time,
            frame_number: self.frames_presented,
        })
    }
    
    pub fn end_frame(
        &mut self,
        queue: &wgpu::Queue,
        frame: PresentFrame<'_>,
        command_buffers: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) {
        // Submit work
        let submission = queue.submit(command_buffers);
        self.synchronizer.record_submission(submission);
        
        // Present
        frame.frame.present();
        self.frames_presented += 1;
    }
    
    pub fn on_resize(&mut self, device: &wgpu::Device, size: winit::dpi::PhysicalSize<u32>) {
        self.surface.on_window_resize(device, size);
        self.suboptimal_count = 0;
    }
    
    pub fn stats(&self) -> PresentationStats {
        PresentationStats {
            frames_presented: self.frames_presented,
            frames_dropped: self.frames_dropped,
            average_fps: self.frame_pacer.fps(),
            frame_time_variance: self.frame_pacer.frame_time_variance(),
            suboptimal_frames: self.suboptimal_count,
        }
    }
}

pub struct PresentFrame<'a> {
    frame: Frame<'a>,
    frame_time: FrameTime,
    frame_number: u64,
}

impl<'a> PresentFrame<'a> {
    pub fn view(&self) -> &wgpu::TextureView {
        self.frame.view()
    }
    
    pub fn size(&self) -> (u32, u32) {
        self.frame.size()
    }
    
    pub fn delta_time(&self) -> f32 {
        self.frame_time.delta_secs
    }
    
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }
}

pub struct PresentationStats {
    pub frames_presented: u64,
    pub frames_dropped: u64,
    pub average_fps: f32,
    pub frame_time_variance: Duration,
    pub suboptimal_frames: u32,
}

pub struct PresentationConfig {
    pub target_fps: u32,
    pub max_frames_in_flight: u32,
    pub vsync: bool,
    pub prefer_hdr: bool,
}

impl Default for PresentationConfig {
    fn default() -> Self {
        Self {
            target_fps: 60,
            max_frames_in_flight: 2,
            vsync: true,
            prefer_hdr: false,
        }
    }
}

struct FrameSynchronizer {
    frame_fences: Vec<Option<wgpu::SubmissionIndex>>,
    current_slot: usize,
    max_slots: usize,
}

impl FrameSynchronizer {
    fn new(max_frames: u32) -> Self {
        Self {
            frame_fences: vec![None; max_frames as usize],
            current_slot: 0,
            max_slots: max_frames as usize,
        }
    }
    
    fn wait_for_frame_slot(&mut self, device: &wgpu::Device) {
        if let Some(submission) = self.frame_fences[self.current_slot].take() {
            device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
        }
    }
    
    fn record_submission(&mut self, submission: wgpu::SubmissionIndex) {
        self.frame_fences[self.current_slot] = Some(submission);
        self.current_slot = (self.current_slot + 1) % self.max_slots;
    }
}
```

---

# Headless Rendering (No Surface)

For offline rendering without a window:

```rust
pub struct HeadlessRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    render_texture: wgpu::Texture,
    render_view: wgpu::TextureView,
    staging_buffer: wgpu::Buffer,
    size: (u32, u32),
}

impl HeadlessRenderer {
    pub fn new(
        adapter: &wgpu::Adapter,
        width: u32,
        height: u32,
    ) -> Self {
        let (device, queue) = pollster::block_on(
            adapter.request_device(&wgpu::DeviceDescriptor::default(), None)
        ).unwrap();
        
        let render_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HeadlessRenderTarget"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        
        let render_view = render_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let row_pitch = align_to(width * 4, 256);
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HeadlessStaging"),
            size: (row_pitch * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        Self {
            device,
            queue,
            render_texture,
            render_view,
            staging_buffer,
            size: (width, height),
        }
    }
    
    pub fn render_to_image(
        &self,
        record_commands: impl FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView),
    ) -> Vec<u8> {
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor::default()
        );
        
        // Record user commands
        record_commands(&mut encoder, &self.render_view);
        
        // Copy to staging
        let row_pitch = align_to(self.size.0 * 4, 256);
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.render_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(row_pitch),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.size.0,
                height: self.size.1,
                depth_or_array_layers: 1,
            },
        );
        
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Read back
        let slice = self.staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        
        // Remove row padding
        let mut pixels = Vec::with_capacity((self.size.0 * self.size.1 * 4) as usize);
        for row in 0..self.size.1 {
            let start = (row * row_pitch) as usize;
            let end = start + (self.size.0 * 4) as usize;
            pixels.extend_from_slice(&data[start..end]);
        }
        
        drop(data);
        self.staging_buffer.unmap();
        
        pixels
    }
}

fn align_to(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}
```

---

# Multi-Window Support

```rust
pub struct MultiWindowRenderer<'windows> {
    surfaces: HashMap<WindowId, TrinitySurface<'windows>>,
    shared_device: Arc<wgpu::Device>,
    shared_queue: Arc<wgpu::Queue>,
}

impl<'windows> MultiWindowRenderer<'windows> {
    pub fn add_window(
        &mut self,
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        window: &'windows Window,
    ) -> Result<WindowId, CreateSurfaceError> {
        let id = window.id();
        let surface = TrinitySurface::new(instance, adapter, window)?;
        surface.inner.configure(&self.shared_device, &surface.config);
        self.surfaces.insert(id, surface);
        Ok(id)
    }
    
    pub fn remove_window(&mut self, id: WindowId) {
        self.surfaces.remove(&id);
    }
    
    pub fn render_all(
        &mut self,
        render_fn: impl Fn(WindowId, &wgpu::TextureView, (u32, u32)) -> wgpu::CommandBuffer,
    ) {
        let mut command_buffers = Vec::new();
        
        for (&id, surface) in &mut self.surfaces {
            if let Some(frame) = surface.try_acquire_or_reconfigure(&self.shared_device) {
                let cmd = render_fn(id, frame.view(), frame.size());
                command_buffers.push((id, cmd, frame));
            }
        }
        
        // Submit all at once
        let cmds: Vec<_> = command_buffers.iter().map(|(_, cmd, _)| cmd.clone()).collect();
        self.shared_queue.submit(cmds);
        
        // Present all
        for (_, _, frame) in command_buffers {
            frame.present();
        }
    }
}
```

---

# TRINITY Presentation Summary

| Component | Purpose | Key Methods |
|-----------|---------|-------------|
| `TrinitySurface` | Window surface abstraction | `acquire_frame()`, `resize()` |
| `Frame` | Single frame's render target | `view()`, `present()` |
| `FramePacer` | Frame timing statistics | `begin_frame()`, `fps()` |
| `PresentationEngine` | Complete presentation system | `begin_frame()`, `end_frame()` |
| `HeadlessRenderer` | Offline rendering | `render_to_image()` |

---

*End of WGPU_PART_IX_PRESENTATION.md*

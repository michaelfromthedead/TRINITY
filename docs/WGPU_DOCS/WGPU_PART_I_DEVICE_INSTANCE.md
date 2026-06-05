# WGPU_PART_I_DEVICE_INSTANCE.md — Device & Instance Model

> **TOC Reference**: Part I, Chapter 1
> **Purpose**: Complete specification of wgpu's object model for TRINITY
> **Generated**: 2026-05-27

---

# Chapter 1: The wgpu Object Model

## 1.1 Instance

### 1.1.1 Instance Creation and Backend Selection

The `wgpu::Instance` is the entry point to the entire graphics API. It represents the wgpu library itself and is used to create adapters and surfaces.

```rust
use wgpu::{Instance, InstanceDescriptor, InstanceFlags, Backends, Dx12Compiler, Gles3MinorVersion};

// Full configuration
let instance = Instance::new(InstanceDescriptor {
    backends: Backends::all(),  // Or specific: Backends::VULKAN | Backends::METAL
    flags: InstanceFlags::default(),
    dx12_shader_compiler: Dx12Compiler::Fxc,  // Or Dxc for better optimization
    gles_minor_version: Gles3MinorVersion::Automatic,
});

// Simple default
let instance = Instance::default();
```

**Backend selection strategy:**

| Platform | Primary Backend | Fallback |
|----------|-----------------|----------|
| Windows | Vulkan | DX12 → DX11 → OpenGL |
| macOS | Metal | (none) |
| Linux | Vulkan | OpenGL |
| Web | WebGPU | WebGL2 |
| iOS | Metal | (none) |
| Android | Vulkan | OpenGL ES |

### 1.1.2 Backend Enumeration

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Backends: u32 {
    const VULKAN = 1 << 0;
    const GL = 1 << 1;
    const METAL = 1 << 2;
    const DX12 = 1 << 3;
    const BROWSER_WEBGPU = 1 << 4;
    
    // Convenience aliases
    const PRIMARY = Self::VULKAN.bits() | Self::METAL.bits() | Self::DX12.bits() | Self::BROWSER_WEBGPU.bits();
    const SECONDARY = Self::GL.bits();
}
```

**TRINITY backend selection:**
```rust
impl TrinityInstance {
    pub fn new(config: &EngineConfig) -> Self {
        let backends = match config.graphics.backend {
            BackendPreference::Auto => {
                #[cfg(target_os = "windows")]
                { Backends::VULKAN | Backends::DX12 }
                #[cfg(target_os = "macos")]
                { Backends::METAL }
                #[cfg(target_os = "linux")]
                { Backends::VULKAN }
                #[cfg(target_arch = "wasm32")]
                { Backends::BROWSER_WEBGPU | Backends::GL }
            }
            BackendPreference::Vulkan => Backends::VULKAN,
            BackendPreference::Metal => Backends::METAL,
            BackendPreference::Dx12 => Backends::DX12,
            BackendPreference::OpenGL => Backends::GL,
        };
        
        let instance = Instance::new(InstanceDescriptor {
            backends,
            flags: if config.debug.validation {
                InstanceFlags::VALIDATION | InstanceFlags::DEBUG
            } else {
                InstanceFlags::empty()
            },
            dx12_shader_compiler: Dx12Compiler::Dxc,
            gles_minor_version: Gles3MinorVersion::Version2,
        });
        
        Self { instance, backends }
    }
}
```

### 1.1.3 Instance Flags and Debugging Layers

```rust
pub struct InstanceFlags: u32 {
    const DEBUG = 1 << 0;
    const VALIDATION = 1 << 1;
    const DISCARD_HAL_LABELS = 1 << 2;
    const ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER = 1 << 3;
    const GPU_BASED_VALIDATION = 1 << 4;
}
```

| Flag | Effect | Performance Impact |
|------|--------|-------------------|
| `DEBUG` | Enable debug markers/groups | Minimal |
| `VALIDATION` | Enable API validation | Moderate (10-30%) |
| `GPU_BASED_VALIDATION` | GPU-side validation | Heavy (50-90%) |
| `DISCARD_HAL_LABELS` | Skip label propagation | Slight improvement |
| `ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER` | Allow non-conformant GPUs | None |

**TRINITY debug configuration:**
```rust
pub struct DebugConfig {
    pub validation: bool,
    pub gpu_validation: bool,
    pub api_trace: bool,
    pub shader_debug: bool,
}

impl DebugConfig {
    pub fn to_instance_flags(&self) -> InstanceFlags {
        let mut flags = InstanceFlags::empty();
        if self.validation {
            flags |= InstanceFlags::DEBUG | InstanceFlags::VALIDATION;
        }
        if self.gpu_validation {
            flags |= InstanceFlags::GPU_BASED_VALIDATION;
        }
        flags
    }
}
```

### 1.1.4 TRINITY's Multi-Backend Strategy

TRINITY supports runtime backend selection with graceful fallback:

```rust
pub struct TrinityGraphics {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    backend: Backend,
    capabilities: DeviceCapabilities,
}

impl TrinityGraphics {
    pub async fn initialize(config: &GraphicsConfig) -> Result<Self, GraphicsError> {
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        
        // Try backends in preference order
        let adapter = Self::select_adapter(&instance, config).await?;
        let backend = adapter.get_info().backend;
        
        log::info!("Selected backend: {:?}", backend);
        log::info!("Adapter: {}", adapter.get_info().name);
        
        let (device, queue) = Self::create_device(&adapter, config).await?;
        let capabilities = DeviceCapabilities::from_adapter(&adapter, &device);
        
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            backend,
            capabilities,
        })
    }
    
    async fn select_adapter(
        instance: &Instance,
        config: &GraphicsConfig,
    ) -> Result<Adapter, GraphicsError> {
        let preference_order = config.backend_preference_order();
        
        for backend in preference_order {
            if let Some(adapter) = instance.request_adapter(&RequestAdapterOptions {
                power_preference: config.power_preference.into(),
                compatible_surface: None,
                force_fallback_adapter: false,
            }).await {
                if adapter.get_info().backend == backend {
                    return Ok(adapter);
                }
            }
        }
        
        // Fallback: any adapter
        instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .ok_or(GraphicsError::NoSuitableAdapter)
    }
}
```

---

## 1.2 Adapter

### 1.2.1 Adapter Enumeration and Selection Criteria

The adapter represents a physical GPU or software renderer:

```rust
// Request adapter with preferences
let adapter = instance.request_adapter(&RequestAdapterOptions {
    power_preference: PowerPreference::HighPerformance,
    compatible_surface: Some(&surface),
    force_fallback_adapter: false,
}).await.expect("No suitable adapter found");

// Enumerate all adapters
let adapters: Vec<Adapter> = instance.enumerate_adapters(Backends::all());
for adapter in &adapters {
    let info = adapter.get_info();
    println!("{}: {:?} ({:?})", info.name, info.device_type, info.backend);
}
```

**Selection criteria hierarchy:**

1. **Surface compatibility** — Must support target surface format
2. **Device type** — Prefer DiscreteGpu > IntegratedGpu > VirtualGpu > Cpu
3. **Feature support** — Must have required features
4. **Power preference** — HighPerformance vs LowPower
5. **Memory** — Larger VRAM preferred

### 1.2.2 Adapter Properties

```rust
pub struct AdapterInfo {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: DeviceType,
    pub driver: String,
    pub driver_info: String,
    pub backend: Backend,
}

pub enum DeviceType {
    Other,
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    Cpu,
}
```

**Vendor IDs:**

| Vendor | ID | Common Products |
|--------|-----|-----------------|
| NVIDIA | 0x10DE | GeForce, Quadro, RTX |
| AMD | 0x1002 | Radeon, RDNA, CDNA |
| Intel | 0x8086 | Iris, UHD, Arc |
| Apple | 0x106B | M1, M2, M3 |
| Qualcomm | 0x5143 | Adreno |
| ARM | 0x13B5 | Mali |

**TRINITY adapter scoring:**
```rust
pub struct AdapterScore {
    pub device_type_score: u32,
    pub feature_score: u32,
    pub memory_score: u32,
    pub driver_score: u32,
    pub total: u32,
}

impl AdapterScore {
    pub fn calculate(adapter: &Adapter, required_features: Features) -> Option<Self> {
        let info = adapter.get_info();
        let features = adapter.features();
        let limits = adapter.limits();
        
        // Must have required features
        if !features.contains(required_features) {
            return None;
        }
        
        let device_type_score = match info.device_type {
            DeviceType::DiscreteGpu => 1000,
            DeviceType::IntegratedGpu => 500,
            DeviceType::VirtualGpu => 200,
            DeviceType::Cpu => 100,
            DeviceType::Other => 50,
        };
        
        // Bonus for RT support
        let feature_score = if features.contains(Features::RAY_TRACING_ACCELERATION_STRUCTURE) {
            500
        } else {
            0
        } + if features.contains(Features::MULTI_DRAW_INDIRECT) {
            100
        } else {
            0
        };
        
        // Memory scoring (rough estimate from limits)
        let memory_score = (limits.max_buffer_size / (1024 * 1024 * 1024)) as u32 * 10;
        
        let total = device_type_score + feature_score + memory_score;
        
        Some(Self {
            device_type_score,
            feature_score,
            memory_score,
            driver_score: 0, // Could add known-good driver versions
            total,
        })
    }
}
```

### 1.2.3 Adapter Limits

Limits define the maximum capabilities of the adapter:

```rust
pub struct Limits {
    // Texture limits
    pub max_texture_dimension_1d: u32,          // 8192
    pub max_texture_dimension_2d: u32,          // 8192
    pub max_texture_dimension_3d: u32,          // 2048
    pub max_texture_array_layers: u32,          // 256
    
    // Bind group limits
    pub max_bind_groups: u32,                   // 4
    pub max_bindings_per_bind_group: u32,       // 1000
    pub max_dynamic_uniform_buffers_per_pipeline_layout: u32,  // 8
    pub max_dynamic_storage_buffers_per_pipeline_layout: u32,  // 4
    
    // Sampler limits
    pub max_sampled_textures_per_shader_stage: u32,   // 16
    pub max_samplers_per_shader_stage: u32,           // 16
    pub max_storage_buffers_per_shader_stage: u32,    // 8
    pub max_storage_textures_per_shader_stage: u32,   // 4
    pub max_uniform_buffers_per_shader_stage: u32,    // 12
    
    // Buffer limits
    pub max_uniform_buffer_binding_size: u32,   // 65536 (64KB)
    pub max_storage_buffer_binding_size: u32,   // 134217728 (128MB)
    pub max_buffer_size: u64,                   // 268435456 (256MB minimum)
    
    // Vertex limits
    pub max_vertex_buffers: u32,                // 8
    pub max_vertex_attributes: u32,             // 16
    pub max_vertex_buffer_array_stride: u32,    // 2048
    
    // Compute limits
    pub max_compute_workgroup_storage_size: u32,       // 16384
    pub max_compute_invocations_per_workgroup: u32,    // 256
    pub max_compute_workgroup_size_x: u32,             // 256
    pub max_compute_workgroup_size_y: u32,             // 256
    pub max_compute_workgroup_size_z: u32,             // 64
    pub max_compute_workgroups_per_dimension: u32,     // 65535
    
    // Push constant limits (where supported)
    pub max_push_constant_size: u32,            // 0 (not in WebGPU, 128+ in native)
    
    // Inter-stage limits
    pub max_inter_stage_shader_components: u32, // 60
    
    // Color attachment limits
    pub max_color_attachments: u32,             // 8
    pub max_color_attachment_bytes_per_sample: u32, // 32
    
    // Subgroup limits (where supported)
    pub min_subgroup_size: u32,                 // 4
    pub max_subgroup_size: u32,                 // 128
}
```

**TRINITY limit requirements:**

```rust
pub struct TrinityLimitRequirements {
    // Minimum limits TRINITY requires
    pub const MIN_TEXTURE_2D: u32 = 4096;
    pub const MIN_TEXTURE_3D: u32 = 256;
    pub const MIN_BIND_GROUPS: u32 = 4;
    pub const MIN_STORAGE_BUFFERS: u32 = 8;
    pub const MIN_SAMPLED_TEXTURES: u32 = 16;
    pub const MIN_BUFFER_SIZE: u64 = 128 * 1024 * 1024; // 128MB
    pub const MIN_UNIFORM_BUFFER_SIZE: u32 = 65536;
    pub const MIN_COMPUTE_WORKGROUP_SIZE: u32 = 256;
}

impl TrinityLimitRequirements {
    pub fn check(limits: &Limits) -> Result<(), LimitError> {
        if limits.max_texture_dimension_2d < Self::MIN_TEXTURE_2D {
            return Err(LimitError::TextureTooSmall);
        }
        if limits.max_storage_buffers_per_shader_stage < Self::MIN_STORAGE_BUFFERS {
            return Err(LimitError::InsufficientStorageBuffers);
        }
        // ... check all
        Ok(())
    }
}
```

### 1.2.4 Feature Detection and Capability Matrices

Features are optional GPU capabilities:

```rust
pub struct Features: u64 {
    // Core WebGPU features (guaranteed on WebGPU)
    const DEPTH_CLIP_CONTROL = 1 << 0;
    const DEPTH32FLOAT_STENCIL8 = 1 << 1;
    const TEXTURE_COMPRESSION_BC = 1 << 2;
    const TEXTURE_COMPRESSION_ETC2 = 1 << 3;
    const TEXTURE_COMPRESSION_ASTC = 1 << 4;
    const TIMESTAMP_QUERY = 1 << 5;
    const INDIRECT_FIRST_INSTANCE = 1 << 6;
    const SHADER_F16 = 1 << 7;
    const RG11B10UFLOAT_RENDERABLE = 1 << 8;
    const BGRA8UNORM_STORAGE = 1 << 9;
    const FLOAT32_FILTERABLE = 1 << 10;
    
    // Native-only features
    const PUSH_CONSTANTS = 1 << 16;
    const TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES = 1 << 17;
    const MULTI_DRAW_INDIRECT = 1 << 18;
    const MULTI_DRAW_INDIRECT_COUNT = 1 << 19;
    const VERTEX_WRITABLE_STORAGE = 1 << 20;
    const TEXTURE_BINDING_ARRAY = 1 << 21;
    const SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING = 1 << 22;
    const PIPELINE_STATISTICS_QUERY = 1 << 23;
    const STORAGE_RESOURCE_BINDING_ARRAY = 1 << 24;
    const PARTIALLY_BOUND_BINDING_ARRAY = 1 << 25;
    const TEXTURE_FORMAT_16BIT_NORM = 1 << 26;
    const TEXTURE_COMPRESSION_ASTC_HDR = 1 << 27;
    const MAPPABLE_PRIMARY_BUFFERS = 1 << 28;
    const BUFFER_BINDING_ARRAY = 1 << 29;
    const UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING = 1 << 30;
    
    // Ray tracing features
    const RAY_TRACING_ACCELERATION_STRUCTURE = 1 << 40;
    const RAY_QUERY = 1 << 41;
    const RAY_TRACING_PIPELINE = 1 << 42;  // Experimental
    
    // Mesh shader features
    const MESH_SHADER = 1 << 48;  // Not yet in wgpu
    const TASK_SHADER = 1 << 49;  // Not yet in wgpu
    
    // Subgroup features
    const SUBGROUP = 1 << 52;
    const SUBGROUP_VERTEX = 1 << 53;
    const SUBGROUP_BARRIER = 1 << 54;
}
```

**TRINITY feature tiers:**

```rust
pub enum FeatureTier {
    Minimal,    // WebGPU baseline
    Standard,   // Desktop baseline
    Advanced,   // Modern desktop
    RayTracing, // RT-capable
    Full,       // All features
}

impl FeatureTier {
    pub fn required_features(&self) -> Features {
        match self {
            Self::Minimal => Features::empty(),
            Self::Standard => {
                Features::DEPTH_CLIP_CONTROL
                | Features::TIMESTAMP_QUERY
                | Features::INDIRECT_FIRST_INSTANCE
                | Features::TEXTURE_COMPRESSION_BC
            }
            Self::Advanced => {
                Self::Standard.required_features()
                | Features::MULTI_DRAW_INDIRECT
                | Features::TEXTURE_BINDING_ARRAY
                | Features::STORAGE_RESOURCE_BINDING_ARRAY
                | Features::PARTIALLY_BOUND_BINDING_ARRAY
            }
            Self::RayTracing => {
                Self::Advanced.required_features()
                | Features::RAY_TRACING_ACCELERATION_STRUCTURE
                | Features::RAY_QUERY
            }
            Self::Full => {
                Self::RayTracing.required_features()
                | Features::RAY_TRACING_PIPELINE
                | Features::MESH_SHADER
            }
        }
    }
    
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let features = adapter.features();
        if features.contains(Self::Full.required_features()) {
            Self::Full
        } else if features.contains(Self::RayTracing.required_features()) {
            Self::RayTracing
        } else if features.contains(Self::Advanced.required_features()) {
            Self::Advanced
        } else if features.contains(Self::Standard.required_features()) {
            Self::Standard
        } else {
            Self::Minimal
        }
    }
}
```

### 1.2.5 Power Preference

```rust
pub enum PowerPreference {
    None,           // No preference
    LowPower,       // Prefer integrated GPU (laptop battery)
    HighPerformance // Prefer discrete GPU
}
```

**TRINITY power management:**
```rust
pub struct PowerManager {
    preference: PowerPreference,
    current_mode: PowerMode,
}

pub enum PowerMode {
    Performance,  // Full GPU clock
    Balanced,     // Adaptive
    PowerSaver,   // Reduced clocks
}

impl PowerManager {
    pub fn select_preference(&self, context: &RenderContext) -> PowerPreference {
        match self.current_mode {
            PowerMode::Performance => PowerPreference::HighPerformance,
            PowerMode::PowerSaver => PowerPreference::LowPower,
            PowerMode::Balanced => {
                // Use high performance if heavy workload detected
                if context.scene_complexity > 10000 || context.rt_enabled {
                    PowerPreference::HighPerformance
                } else {
                    PowerPreference::LowPower
                }
            }
        }
    }
}
```

### 1.2.6 TRINITY's Adapter Selection Algorithm

```rust
pub struct AdapterSelector {
    required_features: Features,
    preferred_features: Features,
    min_limits: Limits,
    power_preference: PowerPreference,
    preferred_vendors: Vec<u32>,
    blacklisted_drivers: Vec<DriverVersion>,
}

impl AdapterSelector {
    pub fn select(&self, adapters: &[Adapter]) -> Option<&Adapter> {
        let mut candidates: Vec<(&Adapter, AdapterScore)> = adapters
            .iter()
            .filter_map(|adapter| {
                // Hard requirements
                let features = adapter.features();
                if !features.contains(self.required_features) {
                    return None;
                }
                
                let limits = adapter.limits();
                if !self.meets_min_limits(&limits) {
                    return None;
                }
                
                // Check driver blacklist
                let info = adapter.get_info();
                if self.is_driver_blacklisted(&info) {
                    return None;
                }
                
                // Score the adapter
                let score = self.score_adapter(adapter);
                Some((adapter, score))
            })
            .collect();
        
        // Sort by score descending
        candidates.sort_by(|a, b| b.1.total.cmp(&a.1.total));
        
        candidates.first().map(|(adapter, _)| *adapter)
    }
    
    fn score_adapter(&self, adapter: &Adapter) -> AdapterScore {
        let info = adapter.get_info();
        let features = adapter.features();
        
        let mut score = AdapterScore::default();
        
        // Device type
        score.device_type_score = match info.device_type {
            DeviceType::DiscreteGpu => 1000,
            DeviceType::IntegratedGpu => {
                if self.power_preference == PowerPreference::LowPower { 900 } else { 500 }
            }
            DeviceType::VirtualGpu => 200,
            DeviceType::Cpu => 100,
            _ => 50,
        };
        
        // Preferred vendor bonus
        if self.preferred_vendors.contains(&info.vendor) {
            score.vendor_score = 100;
        }
        
        // Preferred features bonus
        let preferred_count = (features & self.preferred_features).bits().count_ones();
        score.feature_score = preferred_count * 50;
        
        score.calculate_total();
        score
    }
}
```

---

## 1.3 Device

### 1.3.1 Device Creation from Adapter

```rust
let (device, queue) = adapter.request_device(
    &DeviceDescriptor {
        label: Some("TRINITY Device"),
        required_features: Features::MULTI_DRAW_INDIRECT | Features::TEXTURE_BINDING_ARRAY,
        required_limits: Limits {
            max_bind_groups: 6,
            max_storage_buffers_per_shader_stage: 16,
            ..Limits::default()
        },
        memory_hints: MemoryHints::Performance,
    },
    None, // Trace path for debugging
).await?;
```

### 1.3.2 Required vs Optional Features

**Strategy**: Request minimum required, check for optional at runtime.

```rust
pub struct DeviceRequirements {
    // Must have — device creation fails without
    pub required: Features,
    // Nice to have — affects render path selection
    pub optional: Features,
}

impl DeviceRequirements {
    pub fn for_tier(tier: FeatureTier) -> Self {
        match tier {
            FeatureTier::Minimal => Self {
                required: Features::empty(),
                optional: Features::TIMESTAMP_QUERY | Features::INDIRECT_FIRST_INSTANCE,
            },
            FeatureTier::Standard => Self {
                required: Features::TIMESTAMP_QUERY | Features::INDIRECT_FIRST_INSTANCE,
                optional: Features::MULTI_DRAW_INDIRECT | Features::TEXTURE_BINDING_ARRAY,
            },
            FeatureTier::Advanced => Self {
                required: Features::MULTI_DRAW_INDIRECT 
                    | Features::TEXTURE_BINDING_ARRAY
                    | Features::STORAGE_RESOURCE_BINDING_ARRAY,
                optional: Features::RAY_QUERY | Features::PARTIALLY_BOUND_BINDING_ARRAY,
            },
            FeatureTier::RayTracing => Self {
                required: Features::RAY_TRACING_ACCELERATION_STRUCTURE | Features::RAY_QUERY,
                optional: Features::RAY_TRACING_PIPELINE,
            },
            FeatureTier::Full => Self {
                required: Features::RAY_TRACING_ACCELERATION_STRUCTURE 
                    | Features::RAY_QUERY
                    | Features::RAY_TRACING_PIPELINE,
                optional: Features::empty(),
            },
        }
    }
}
```

### 1.3.3 Device Limits and Limit Negotiation

Request only what you need — some implementations may fail with overly aggressive limits:

```rust
pub fn negotiate_limits(adapter: &Adapter, requirements: &LimitRequirements) -> Limits {
    let adapter_limits = adapter.limits();
    
    Limits {
        // Use required minimum or adapter max
        max_bind_groups: requirements.min_bind_groups.max(adapter_limits.max_bind_groups.min(8)),
        max_storage_buffers_per_shader_stage: requirements.min_storage_buffers
            .max(adapter_limits.max_storage_buffers_per_shader_stage.min(16)),
        max_sampled_textures_per_shader_stage: requirements.min_sampled_textures
            .max(adapter_limits.max_sampled_textures_per_shader_stage.min(128)),
        max_buffer_size: requirements.min_buffer_size
            .max(adapter_limits.max_buffer_size.min(1024 * 1024 * 1024)), // 1GB cap
        
        // Use adapter defaults for everything else
        ..adapter_limits
    }
}
```

### 1.3.4 Device Lost Handling and Recovery

Devices can be lost due to driver crash, GPU reset, or timeout:

```rust
pub struct DeviceManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    lost_receiver: flume::Receiver<DeviceLostReason>,
}

impl DeviceManager {
    pub fn new(adapter: &Adapter, descriptor: &DeviceDescriptor) -> Result<Self, DeviceError> {
        let (lost_sender, lost_receiver) = flume::bounded(1);
        
        let (device, queue) = pollster::block_on(adapter.request_device(
            descriptor,
            None,
        ))?;
        
        device.on_uncaptured_error(Box::new(|error| {
            log::error!("Uncaptured GPU error: {:?}", error);
        }));
        
        // Set up device lost callback
        let sender = lost_sender.clone();
        device.set_device_lost_callback(move |reason, message| {
            log::error!("Device lost: {:?} - {}", reason, message);
            let _ = sender.try_send(reason);
        });
        
        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            lost_receiver,
        })
    }
    
    pub fn check_device_lost(&self) -> Option<DeviceLostReason> {
        self.lost_receiver.try_recv().ok()
    }
    
    pub fn poll(&self, maintain: Maintain) -> MaintainResult {
        self.device.poll(maintain)
    }
}

pub enum DeviceLostReason {
    Unknown,
    Destroyed,
    ReplacedCallback,
    DeviceLost,
}

// Recovery strategy
impl TrinityGraphics {
    pub fn handle_device_lost(&mut self) -> Result<(), GraphicsError> {
        log::warn!("Attempting device recovery...");
        
        // Drop old resources
        self.resource_cache.clear();
        self.pipeline_cache.clear();
        
        // Recreate device
        let (device, queue) = pollster::block_on(
            self.adapter.request_device(&self.device_descriptor, None)
        )?;
        
        self.device = Arc::new(device);
        self.queue = Arc::new(queue);
        
        // Reload essential resources
        self.reload_essential_resources()?;
        
        log::info!("Device recovery successful");
        Ok(())
    }
}
```

### 1.3.5 Error Scopes

Error scopes capture GPU errors for specific operations:

```rust
// Push an error scope
device.push_error_scope(ErrorFilter::Validation);

// Do operations that might fail
let buffer = device.create_buffer(&BufferDescriptor {
    label: Some("Test Buffer"),
    size: 1024,
    usage: BufferUsages::COPY_DST, // Missing COPY_SRC if we try to read
    mapped_at_creation: false,
});

// Pop and check for errors
let error = device.pop_error_scope().await;
if let Some(error) = error {
    match error {
        Error::Validation { description, .. } => {
            log::error!("Validation error: {}", description);
        }
        Error::OutOfMemory { .. } => {
            log::error!("Out of GPU memory");
        }
        Error::Internal { description, .. } => {
            log::error!("Internal error: {}", description);
        }
    }
}
```

**TRINITY error scope wrapper:**
```rust
pub struct ErrorScope<'a> {
    device: &'a Device,
    filter: ErrorFilter,
}

impl<'a> ErrorScope<'a> {
    pub fn new(device: &'a Device, filter: ErrorFilter) -> Self {
        device.push_error_scope(filter);
        Self { device, filter }
    }
    
    pub async fn finish(self) -> Option<Error> {
        self.device.pop_error_scope().await
    }
}

// Usage
let scope = ErrorScope::new(&device, ErrorFilter::Validation);
// ... operations
if let Some(error) = scope.finish().await {
    return Err(error.into());
}
```

### 1.3.6 TRINITY's Device Lifecycle Management

```rust
pub struct TrinityDevice {
    inner: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    features: Features,
    limits: Limits,
    
    // Resource tracking
    buffer_count: AtomicU64,
    texture_count: AtomicU64,
    pipeline_count: AtomicU64,
    
    // Memory tracking
    allocated_buffer_bytes: AtomicU64,
    allocated_texture_bytes: AtomicU64,
    
    // Frame management
    frame_number: AtomicU64,
    pending_work: Mutex<Vec<PendingWork>>,
}

impl TrinityDevice {
    pub fn create_buffer(&self, desc: &BufferDescriptor) -> TrackedBuffer {
        let buffer = self.inner.create_buffer(desc);
        self.buffer_count.fetch_add(1, Ordering::Relaxed);
        self.allocated_buffer_bytes.fetch_add(desc.size, Ordering::Relaxed);
        
        TrackedBuffer {
            buffer,
            size: desc.size,
            device: self.clone(),
        }
    }
    
    pub fn begin_frame(&self) -> FrameContext {
        let frame = self.frame_number.fetch_add(1, Ordering::SeqCst);
        FrameContext {
            frame_number: frame,
            device: self,
        }
    }
    
    pub fn end_frame(&self, ctx: FrameContext) {
        // Clean up resources scheduled for deletion
        let mut pending = self.pending_work.lock().unwrap();
        pending.retain(|work| {
            if work.ready_frame <= ctx.frame_number {
                work.execute();
                false
            } else {
                true
            }
        });
    }
    
    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            buffer_count: self.buffer_count.load(Ordering::Relaxed),
            texture_count: self.texture_count.load(Ordering::Relaxed),
            allocated_buffer_bytes: self.allocated_buffer_bytes.load(Ordering::Relaxed),
            allocated_texture_bytes: self.allocated_texture_bytes.load(Ordering::Relaxed),
        }
    }
}
```

---

## 1.4 Queue

### 1.4.1 Queue Submission Model

The queue is the single point of communication with the GPU:

```rust
// Submit command buffers
queue.submit([command_buffer1, command_buffer2]);

// Or submit iterator
queue.submit(command_buffers.into_iter());
```

### 1.4.2 Command Buffer Submission

```rust
pub struct FrameSubmission {
    command_buffers: Vec<CommandBuffer>,
    signal_on_complete: Option<Arc<AtomicBool>>,
}

impl TrinityQueue {
    pub fn submit_frame(&self, submission: FrameSubmission) {
        // Submit all command buffers
        self.queue.submit(submission.command_buffers);
        
        // If caller wants to know when complete, use on_submitted_work_done
        if let Some(signal) = submission.signal_on_complete {
            self.queue.on_submitted_work_done(move || {
                signal.store(true, Ordering::SeqCst);
            });
        }
    }
}
```

### 1.4.3 Queue Write Operations

Direct CPU→GPU data transfer without explicit command buffers:

```rust
// Write to buffer
queue.write_buffer(&buffer, offset, &data);

// Write to texture
queue.write_texture(
    ImageCopyTexture {
        texture: &texture,
        mip_level: 0,
        origin: Origin3d::ZERO,
        aspect: TextureAspect::All,
    },
    &pixel_data,
    ImageDataLayout {
        offset: 0,
        bytes_per_row: Some(4 * width),
        rows_per_image: Some(height),
    },
    Extent3d { width, height, depth_or_array_layers: 1 },
);
```

**TRINITY immediate upload:**
```rust
impl TrinityQueue {
    pub fn upload_buffer_immediate(&self, buffer: &Buffer, offset: u64, data: &[u8]) {
        self.queue.write_buffer(buffer, offset, data);
    }
    
    pub fn upload_texture_immediate(
        &self,
        texture: &Texture,
        mip_level: u32,
        origin: [u32; 3],
        data: &[u8],
        bytes_per_row: u32,
        size: [u32; 3],
    ) {
        self.queue.write_texture(
            ImageCopyTexture {
                texture,
                mip_level,
                origin: Origin3d { x: origin[0], y: origin[1], z: origin[2] },
                aspect: TextureAspect::All,
            },
            data,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(size[1]),
            },
            Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: size[2],
            },
        );
    }
}
```

### 1.4.4 Queue Synchronization Semantics

wgpu handles most synchronization automatically:

- **Implicit barriers**: Resources are tracked, barriers inserted automatically
- **Submission order**: Commands execute in submission order
- **Queue completion**: `on_submitted_work_done` callback when GPU finishes

```rust
pub struct QueueSync {
    queue: Arc<wgpu::Queue>,
    pending_frames: VecDeque<FrameSync>,
}

struct FrameSync {
    frame_number: u64,
    complete: Arc<AtomicBool>,
}

impl QueueSync {
    pub fn submit_and_track(&mut self, frame: u64, commands: Vec<CommandBuffer>) {
        let complete = Arc::new(AtomicBool::new(false));
        
        self.queue.submit(commands);
        
        let complete_clone = complete.clone();
        self.queue.on_submitted_work_done(move || {
            complete_clone.store(true, Ordering::SeqCst);
        });
        
        self.pending_frames.push_back(FrameSync {
            frame_number: frame,
            complete,
        });
    }
    
    pub fn wait_for_frame(&self, frame: u64) {
        while let Some(pending) = self.pending_frames.front() {
            if pending.frame_number > frame {
                break;
            }
            // Spin wait (or could use device.poll)
            while !pending.complete.load(Ordering::SeqCst) {
                std::hint::spin_loop();
            }
        }
    }
}
```

### 1.4.5 Multi-Queue Considerations

wgpu currently exposes a single queue, but the abstraction supports future multi-queue:

```rust
pub enum QueueFamily {
    Graphics,
    Compute,
    Transfer,
    Present,
}

pub struct MultiQueueDevice {
    graphics_queue: Queue,
    // Future: compute_queue, transfer_queue
}

impl MultiQueueDevice {
    // For now, all queues are the same
    pub fn queue(&self, _family: QueueFamily) -> &Queue {
        &self.graphics_queue
    }
}
```

### 1.4.6 TRINITY's Submission Batching Strategy

```rust
pub struct SubmissionBatcher {
    queue: Arc<wgpu::Queue>,
    pending_commands: Vec<CommandBuffer>,
    batch_size_threshold: usize,
    time_threshold: Duration,
    last_submit: Instant,
}

impl SubmissionBatcher {
    pub fn new(queue: Arc<wgpu::Queue>) -> Self {
        Self {
            queue,
            pending_commands: Vec::with_capacity(16),
            batch_size_threshold: 8,
            time_threshold: Duration::from_micros(500),
            last_submit: Instant::now(),
        }
    }
    
    pub fn enqueue(&mut self, command_buffer: CommandBuffer) {
        self.pending_commands.push(command_buffer);
        
        let should_flush = self.pending_commands.len() >= self.batch_size_threshold
            || self.last_submit.elapsed() >= self.time_threshold;
        
        if should_flush {
            self.flush();
        }
    }
    
    pub fn flush(&mut self) {
        if !self.pending_commands.is_empty() {
            self.queue.submit(self.pending_commands.drain(..));
            self.last_submit = Instant::now();
        }
    }
    
    pub fn submit_immediate(&self, command_buffer: CommandBuffer) {
        self.queue.submit([command_buffer]);
    }
}
```

---

# TRINITY Device Module Architecture

```
crates/renderer-backend/src/device/
├── mod.rs              # Module root, TrinityDevice struct
├── instance.rs         # Instance creation, backend selection
├── adapter.rs          # Adapter enumeration, scoring, selection
├── device.rs           # Device creation, lifecycle, error handling
├── queue.rs            # Queue management, submission batching
├── features.rs         # Feature tiers, capability detection
├── limits.rs           # Limit negotiation, requirements
└── memory.rs           # Memory tracking, stats
```

**Public API:**
```rust
// Re-exports
pub use instance::TrinityInstance;
pub use adapter::{AdapterSelector, AdapterScore};
pub use device::{TrinityDevice, DeviceManager};
pub use queue::{TrinityQueue, SubmissionBatcher};
pub use features::{FeatureTier, DeviceCapabilities};
pub use limits::{TrinityLimitRequirements, negotiate_limits};
pub use memory::MemoryStats;

// Main entry point
pub async fn initialize_graphics(config: &GraphicsConfig) -> Result<TrinityGraphics, GraphicsError> {
    let instance = TrinityInstance::new(config)?;
    let adapter = instance.select_adapter(config).await?;
    let tier = FeatureTier::from_adapter(&adapter);
    let (device, queue) = TrinityDevice::create(&adapter, tier, config).await?;
    
    Ok(TrinityGraphics {
        instance,
        adapter,
        device,
        queue,
        tier,
        capabilities: DeviceCapabilities::from_device(&device),
    })
}
```

---

*End of WGPU_PART_I_DEVICE_INSTANCE.md*

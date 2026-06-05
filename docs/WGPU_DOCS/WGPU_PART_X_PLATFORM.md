# WGPU_PART_X_PLATFORM.md — Platform & Backends

> **Scope**: Complete coverage of wgpu backends (Vulkan, Metal, DX12, WebGPU, OpenGL), platform considerations, feature detection, and capability abstraction
> **TRINITY Integration**: Multi-backend support, feature tier system, automatic fallbacks
> **wgpu Version**: 25.x+

---

# Chapter 19: Platform Considerations

wgpu abstracts multiple graphics APIs behind a unified interface. Understanding backend-specific behaviors is essential for optimal cross-platform performance.

---

## 19.1 Vulkan Backend

### 19.1.1 Vulkan Instance/Device Mapping

wgpu's Vulkan backend maps directly to Vulkan concepts:

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

### 19.1.2 Vulkan Extension Requirements

wgpu requires specific Vulkan extensions:

**Instance Extensions** (required):
- `VK_KHR_surface`
- Platform surface extension (`VK_KHR_win32_surface`, `VK_KHR_xcb_surface`, etc.)

**Device Extensions** (required):
- `VK_KHR_swapchain`
- `VK_KHR_maintenance1`
- `VK_KHR_maintenance2`
- `VK_KHR_maintenance3`

**Device Extensions** (optional, feature-dependent):
- `VK_KHR_ray_tracing_pipeline` → `ray_tracing_pipeline`
- `VK_KHR_acceleration_structure` → `acceleration_structure`
- `VK_KHR_ray_query` → `ray_query`
- `VK_EXT_mesh_shader` → `mesh_shader` (future)
- `VK_KHR_16bit_storage` → `shader_f16`
- `VK_KHR_spirv_1_4` → Required for ray tracing

```rust
pub fn check_vulkan_ray_tracing_support(adapter: &wgpu::Adapter) -> bool {
    let features = adapter.features();
    
    features.contains(wgpu::Features::RAY_QUERY)
        || features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE)
}
```

### 19.1.3 Vulkan-Specific Features

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

### 19.1.4 Debugging with Validation Layers

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

---

## 19.2 Metal Backend

### 19.2.1 Metal Device Selection

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

### 19.2.2 Metal Feature Sets

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
            // Parse "Apple M1", "Apple M2", etc.
            if info.name.contains("M3") || info.name.contains("A17") {
                7
            } else if info.name.contains("M2") || info.name.contains("A16") {
                6
            } else if info.name.contains("M1") || info.name.contains("A14") || info.name.contains("A15") {
                5
            } else {
                4
            }
        } else {
            0 // AMD/Intel GPU
        };
        
        Self {
            supports_ray_tracing: apple_family >= 6 && 
                features.contains(wgpu::Features::RAY_QUERY),
            supports_mesh_shaders: apple_family >= 7,
            max_buffer_length: limits.max_buffer_size,
            apple_gpu_family: apple_family,
        }
    }
}
```

### 19.2.3 Metal-Specific Considerations

**Memory Model**:
- Metal uses unified memory on Apple Silicon
- No explicit staging buffers needed for Apple Silicon
- AMD discrete GPUs still need staging

```rust
pub fn should_use_staging_buffer(adapter: &wgpu::Adapter) -> bool {
    let info = adapter.get_info();
    
    // Apple Silicon has unified memory
    if info.name.contains("Apple") {
        return false;
    }
    
    // AMD/Intel discrete needs staging
    true
}
```

**Texture Compression**:
- ASTC is preferred on iOS/Apple Silicon
- BC formats available on macOS with discrete GPU

### 19.2.4 Argument Buffers for Bindless

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

---

## 19.3 DX12 Backend

### 19.3.1 DX12 Device Selection

```rust
pub fn select_dx12_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    let adapters: Vec<_> = instance
        .enumerate_adapters(wgpu::Backends::DX12)
        .collect();
    
    // Filter out software adapters (WARP)
    let hardware_adapters: Vec<_> = adapters.iter()
        .filter(|a| {
            let info = a.get_info();
            info.device_type != wgpu::DeviceType::Cpu
        })
        .collect();
    
    // Prefer discrete GPU
    hardware_adapters.iter()
        .find(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
        .or_else(|| hardware_adapters.first())
        .cloned()
        .cloned()
}
```

### 19.3.2 DX12 Feature Levels

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
        
        // Infer feature level from capabilities
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

### 19.3.3 Root Signature Mapping

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
pub fn optimize_bind_group_for_dx12(
    entries: &mut [wgpu::BindGroupLayoutEntry],
) {
    // DX12 tip: Put frequently changed bindings in lower indices
    // Root constants (push constants) should come first
    
    entries.sort_by_key(|e| {
        match e.ty {
            // Push constants first (if supported)
            wgpu::BindingType::Buffer { 
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                ..
            } => 0,
            // Then regular uniforms
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                ..
            } => 1,
            // Then textures
            wgpu::BindingType::Texture { .. } => 2,
            // Then samplers
            wgpu::BindingType::Sampler { .. } => 3,
            // Storage last (UAVs)
            _ => 4,
        }
    });
}
```

### 19.3.4 DX12-Specific Features

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

---

## 19.4 WebGPU Backend

### 19.4.1 Browser Compatibility

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

### 19.4.2 WebGPU Spec Conformance

WebGPU has stricter validation than native backends:

```rust
pub fn validate_for_webgpu(buffer_size: u64) -> Result<(), &'static str> {
    // WebGPU requires buffer sizes to be multiples of 4
    if buffer_size % 4 != 0 {
        return Err("Buffer size must be multiple of 4 for WebGPU");
    }
    
    // Max buffer size is lower on WebGPU
    const WEBGPU_MAX_BUFFER_SIZE: u64 = 256 * 1024 * 1024; // 256MB typical
    if buffer_size > WEBGPU_MAX_BUFFER_SIZE {
        return Err("Buffer too large for WebGPU");
    }
    
    Ok(())
}
```

### 19.4.3 Web-Specific Limitations

```rust
pub struct WebGPULimitations {
    pub max_texture_dimension_2d: u32,      // Often 8192 vs 16384 native
    pub max_buffer_size: u64,               // 256MB typical
    pub max_storage_buffer_binding_size: u64, // 128MB typical
    pub max_compute_workgroup_size_x: u32,   // 256
    pub max_compute_workgroups_per_dimension: u32, // 65535
}

impl WebGPULimitations {
    pub fn default_limits() -> wgpu::Limits {
        // Use downlevel defaults suitable for WebGPU
        wgpu::Limits::downlevel_webgl2_defaults()
            .using_resolution(wgpu::Limits::default())
    }
}
```

### 19.4.4 WASM Integration

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
        web_sys::window()
            .map(|w| w.device_pixel_ratio())
            .unwrap_or(1.0)
    }
}
```

---

## 19.5 OpenGL Backend (Fallback)

### 19.5.1 OpenGL ES / WebGL Fallback

```rust
pub fn create_gles_instance() -> wgpu::Instance {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::GL,
        gles_minor_version: wgpu::Gles3MinorVersion::Version2, // OpenGL ES 3.2
        ..Default::default()
    })
}
```

### 19.5.2 Feature Limitations

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

### 19.5.3 Performance Considerations

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

---

# Chapter 20: Feature Detection & Capability Abstraction

---

## 20.1 Feature Flags

### 20.1.1 Core Features (Always Available)

These features are guaranteed on all wgpu backends:

```rust
pub fn core_features() -> Vec<&'static str> {
    vec![
        "Vertex buffers",
        "Index buffers (u16, u32)",
        "Uniform buffers",
        "2D textures",
        "Samplers",
        "Render passes",
        "Basic compute shaders",
        "MSAA (up to 4x)",
    ]
}
```

### 20.1.2 Optional Features (Adapter-Dependent)

```rust
pub fn check_optional_features(adapter: &wgpu::Adapter) -> OptionalFeatures {
    let features = adapter.features();
    
    OptionalFeatures {
        // Texture features
        texture_compression_bc: features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC),
        texture_compression_etc2: features.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2),
        texture_compression_astc: features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC),
        
        // Buffer features
        multi_draw_indirect: features.contains(wgpu::Features::MULTI_DRAW_INDIRECT),
        multi_draw_indirect_count: features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT),
        push_constants: features.contains(wgpu::Features::PUSH_CONSTANTS),
        
        // Shader features
        shader_f16: features.contains(wgpu::Features::SHADER_F16),
        shader_f64: features.contains(wgpu::Features::SHADER_F64),
        
        // Bindless
        texture_binding_array: features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY),
        sampled_texture_and_storage_buffer_array_non_uniform_indexing: 
            features.contains(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING),
        
        // Ray tracing
        ray_query: features.contains(wgpu::Features::RAY_QUERY),
        ray_tracing_acceleration_structure: 
            features.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE),
        
        // Advanced rendering
        conservative_rasterization: features.contains(wgpu::Features::CONSERVATIVE_RASTERIZATION),
        depth_clip_control: features.contains(wgpu::Features::DEPTH_CLIP_CONTROL),
        
        // Timestamps
        timestamp_query: features.contains(wgpu::Features::TIMESTAMP_QUERY),
        pipeline_statistics_query: features.contains(wgpu::Features::PIPELINE_STATISTICS_QUERY),
    }
}

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
```

### 20.1.3 Experimental Features (Unstable API)

```rust
pub fn experimental_features() -> &'static [wgpu::Features] {
    &[
        wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE,
        // Future features will be added here
    ]
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

### 20.1.4 Feature Dependency Chains

```rust
pub struct FeatureDependencies;

impl FeatureDependencies {
    pub fn dependencies_for(feature: wgpu::Features) -> wgpu::Features {
        match feature {
            f if f.contains(wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE) => {
                // RT requires buffer device address (implied)
                wgpu::Features::RAY_TRACING_ACCELERATION_STRUCTURE
            }
            f if f.contains(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING) => {
                // Non-uniform indexing requires binding arrays
                wgpu::Features::TEXTURE_BINDING_ARRAY
                    | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            }
            _ => feature,
        }
    }
    
    pub fn expand_features(requested: wgpu::Features) -> wgpu::Features {
        let mut expanded = requested;
        
        // Add all dependencies
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

---

## 20.2 Limits

### 20.2.1 Key Limits

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
    println!("  max_dynamic_uniform_buffers_per_pipeline_layout: {}", 
             limits.max_dynamic_uniform_buffers_per_pipeline_layout);
    println!("  max_dynamic_storage_buffers_per_pipeline_layout: {}", 
             limits.max_dynamic_storage_buffers_per_pipeline_layout);
    println!("  max_sampled_textures_per_shader_stage: {}", 
             limits.max_sampled_textures_per_shader_stage);
    println!("  max_samplers_per_shader_stage: {}", 
             limits.max_samplers_per_shader_stage);
    println!("  max_storage_buffers_per_shader_stage: {}", 
             limits.max_storage_buffers_per_shader_stage);
    println!("  max_storage_textures_per_shader_stage: {}", 
             limits.max_storage_textures_per_shader_stage);
    println!("  max_uniform_buffers_per_shader_stage: {}", 
             limits.max_uniform_buffers_per_shader_stage);
    
    println!("\n=== Compute Limits ===");
    println!("  max_compute_workgroup_storage_size: {} bytes", 
             limits.max_compute_workgroup_storage_size);
    println!("  max_compute_invocations_per_workgroup: {}", 
             limits.max_compute_invocations_per_workgroup);
    println!("  max_compute_workgroup_size_x: {}", limits.max_compute_workgroup_size_x);
    println!("  max_compute_workgroup_size_y: {}", limits.max_compute_workgroup_size_y);
    println!("  max_compute_workgroup_size_z: {}", limits.max_compute_workgroup_size_z);
    println!("  max_compute_workgroups_per_dimension: {}", 
             limits.max_compute_workgroups_per_dimension);
    
    println!("\n=== Vertex Limits ===");
    println!("  max_vertex_buffers: {}", limits.max_vertex_buffers);
    println!("  max_vertex_attributes: {}", limits.max_vertex_attributes);
    println!("  max_vertex_buffer_array_stride: {}", limits.max_vertex_buffer_array_stride);
    
    println!("\n=== Other Limits ===");
    println!("  max_push_constant_size: {} bytes", limits.max_push_constant_size);
    println!("  max_color_attachments: {}", limits.max_color_attachments);
    println!("  max_color_attachment_bytes_per_sample: {}", 
             limits.max_color_attachment_bytes_per_sample);
}
```

### 20.2.2 Limit Negotiation

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
            min_uniform_buffer_size: 64 * 1024,    // 64KB
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
    
    pub fn build_limits(&self, adapter_limits: &wgpu::Limits) -> wgpu::Limits {
        wgpu::Limits {
            max_uniform_buffer_binding_size: self.min_uniform_buffer_size.max(
                adapter_limits.max_uniform_buffer_binding_size.min(65536)
            ),
            max_storage_buffer_binding_size: self.min_storage_buffer_size.max(
                adapter_limits.max_storage_buffer_binding_size
            ) as u32,
            max_texture_dimension_2d: self.min_texture_size.max(
                adapter_limits.max_texture_dimension_2d
            ),
            max_bind_groups: self.min_bind_groups.max(adapter_limits.max_bind_groups),
            // ... other limits with sensible defaults
            ..adapter_limits.clone()
        }
    }
}
```

---

## 20.3 TRINITY's Capability System

### 20.3.1 Capability Tiers

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
            && features.contains(wgpu::Features::RAY_QUERY)
        {
            return CapabilityTier::Full;
        }
        
        // Check for Advanced tier (bindless, compute)
        if features.contains(wgpu::Features::TEXTURE_BINDING_ARRAY)
            && features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
            && limits.max_compute_workgroup_storage_size >= 32768
        {
            return CapabilityTier::Advanced;
        }
        
        // Check for Standard tier
        if limits.max_texture_dimension_2d >= 8192
            && limits.max_compute_workgroup_size_x >= 256
            && limits.max_storage_buffer_binding_size >= 128 * 1024 * 1024
        {
            return CapabilityTier::Standard;
        }
        
        CapabilityTier::Minimal
    }
    
    pub fn required_features(&self) -> wgpu::Features {
        match self {
            CapabilityTier::Minimal => wgpu::Features::empty(),
            CapabilityTier::Standard => wgpu::Features::empty(),
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

### 20.3.2 Feature Requirements Per Render Path

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
    
    pub fn required_features(&self) -> wgpu::Features {
        self.required_tier().required_features()
    }
}
```

### 20.3.3 Automatic Fallback Selection

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

pub enum TextureCompression {
    None,
    BC,
    ASTC,
    ETC2,
}
```

### 20.3.4 Runtime Capability Queries

```rust
impl CapabilityManager {
    pub fn supports_ray_tracing(&self) -> bool {
        self.tier >= CapabilityTier::Full
    }
    
    pub fn supports_bindless(&self) -> bool {
        self.tier >= CapabilityTier::Advanced
    }
    
    pub fn supports_gpu_culling(&self) -> bool {
        self.can_use_feature(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT)
    }
    
    pub fn supports_timestamp_queries(&self) -> bool {
        self.can_use_feature(wgpu::Features::TIMESTAMP_QUERY)
    }
    
    pub fn supports_mesh_shaders(&self) -> bool {
        false // Not yet in wgpu
    }
    
    pub fn max_msaa_samples(&self) -> u32 {
        // Query sample count support
        4 // Conservative default
    }
    
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

---

# TRINITY Platform Support Matrix

| Feature | Vulkan | Metal | DX12 | WebGPU | OpenGL |
|---------|--------|-------|------|--------|--------|
| Core rendering | ✅ | ✅ | ✅ | ✅ | ✅ |
| Compute shaders | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| Bindless textures | ✅ | ✅ | ✅ | ❌ | ❌ |
| Multi-draw indirect | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| Ray query | ✅ | ⚠️ | ✅ | ❌ | ❌ |
| RT pipeline | ✅ | ❌ | ✅ | ❌ | ❌ |
| Mesh shaders | 🔜 | 🔜 | 🔜 | ❌ | ❌ |
| Timestamp queries | ✅ | ✅ | ✅ | ✅ | ⚠️ |

Legend: ✅ Supported | ⚠️ Limited | ❌ Not supported | 🔜 Future

---

*End of WGPU_PART_X_PLATFORM.md*

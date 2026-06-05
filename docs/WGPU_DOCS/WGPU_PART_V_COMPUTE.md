# WGPU_PART_V_COMPUTE.md — Compute Pipeline

> **TOC Reference**: Part V, Chapter 8
> **Purpose**: Complete specification of wgpu compute shaders and dispatch for TRINITY
> **Generated**: 2026-05-27

---

# Chapter 8: Compute Fundamentals

## 8.1 Compute Pipeline

### 8.1.1 Compute Pipeline Descriptor

```rust
let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
    label: Some("Particle Update"),
    layout: Some(&pipeline_layout),
    module: &compute_shader,
    entry_point: "main",
    compilation_options: PipelineCompilationOptions {
        constants: &[
            ("WORKGROUP_SIZE", 256.0),
            ("MAX_PARTICLES", 100000.0),
        ].into_iter().collect(),
        zero_initialize_workgroup_memory: true,
    },
    cache: None,
});
```

### 8.1.2 Pipeline Layout Association

```rust
let compute_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
    label: Some("Compute Layout"),
    bind_group_layouts: &[
        &input_layout,   // Group 0: Read-only inputs
        &output_layout,  // Group 1: Read-write outputs
        &params_layout,  // Group 2: Uniform parameters
    ],
    push_constant_ranges: &[
        PushConstantRange {
            stages: ShaderStages::COMPUTE,
            range: 0..16,  // Time, delta_time, frame, etc.
        },
    ],
});
```

### 8.1.3 Entry Point Specification

```wgsl
// Multiple entry points in one module
@compute @workgroup_size(256, 1, 1)
fn particle_simulate() { /* ... */ }

@compute @workgroup_size(8, 8, 1)
fn particle_emit() { /* ... */ }

@compute @workgroup_size(1, 1, 1)
fn particle_clear() { /* ... */ }
```

```rust
// Select entry point at pipeline creation
let simulate_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
    entry_point: "particle_simulate",
    ..
});

let emit_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
    entry_point: "particle_emit",
    ..
});
```

### 8.1.4 Compute Pipeline Caching

```rust
pub struct ComputePipelineCache {
    pipelines: HashMap<ComputePipelineKey, Arc<ComputePipeline>>,
}

#[derive(Hash, Eq, PartialEq, Clone)]
struct ComputePipelineKey {
    shader_id: ShaderId,
    entry_point: String,
    specialization: Vec<(String, u64)>,  // Constants as bits
}

impl ComputePipelineCache {
    pub fn get_or_create(
        &mut self,
        device: &Device,
        key: &ComputePipelineKey,
        shader: &ShaderModule,
        layout: &PipelineLayout,
    ) -> Arc<ComputePipeline> {
        self.pipelines.entry(key.clone()).or_insert_with(|| {
            Arc::new(device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(&format!("{:?}", key)),
                layout: Some(layout),
                module: shader,
                entry_point: &key.entry_point,
                compilation_options: PipelineCompilationOptions {
                    constants: &key.specialization.iter()
                        .map(|(k, v)| (k.as_str(), f64::from_bits(*v)))
                        .collect(),
                    zero_initialize_workgroup_memory: true,
                },
                cache: None,
            }))
        }).clone()
    }
}
```

---

## 8.2 Compute Shaders

### 8.2.1 @compute Entry Points

```wgsl
@compute @workgroup_size(8, 8, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    // Compute work here
}
```

### 8.2.2 @workgroup_size Attribute

```wgsl
// Fixed workgroup size
@compute @workgroup_size(256, 1, 1)
fn linear_compute() { }

// 2D workgroup (image processing)
@compute @workgroup_size(8, 8, 1)
fn image_compute() { }

// 3D workgroup (volumetric)
@compute @workgroup_size(4, 4, 4)
fn volume_compute() { }

// Using override constants
@id(0) override WG_X: u32 = 8;
@id(1) override WG_Y: u32 = 8;

@compute @workgroup_size(WG_X, WG_Y, 1)
fn configurable_compute() { }
```

**Workgroup size considerations:**

| Factor | Guidance |
|--------|----------|
| **Occupancy** | Larger workgroups = fewer scheduling units |
| **Shared memory** | More threads = more shared memory needed |
| **Register pressure** | More threads = fewer registers per thread |
| **Memory access** | 32/64 threads for coalesced access |
| **Portability** | 64 threads common minimum |

**Common configurations:**

| Use Case | Size | Total | Rationale |
|----------|------|-------|-----------|
| Linear data | (256, 1, 1) | 256 | Good for buffers |
| Images | (8, 8, 1) | 64 | 2D spatial locality |
| Images (large) | (16, 16, 1) | 256 | More threads, larger tiles |
| Volumetric | (4, 4, 4) | 64 | 3D spatial locality |
| Single | (1, 1, 1) | 1 | Serial work |

### 8.2.3 Built-in Variables

```wgsl
// Global invocation ID: unique thread identifier
// Range: [0, dispatch_size * workgroup_size)
@builtin(global_invocation_id) global_id: vec3<u32>

// Local invocation ID: position within workgroup
// Range: [0, workgroup_size)
@builtin(local_invocation_id) local_id: vec3<u32>

// Local invocation index: flattened local ID
// = local_id.x + local_id.y * wg_size.x + local_id.z * wg_size.x * wg_size.y
@builtin(local_invocation_index) local_index: u32

// Workgroup ID: which workgroup this thread is in
// Range: [0, dispatch_size)
@builtin(workgroup_id) workgroup_id: vec3<u32>

// Number of workgroups dispatched
@builtin(num_workgroups) num_workgroups: vec3<u32>
```

**Computing global index:**
```wgsl
fn get_global_index() -> u32 {
    let wg_size = vec3<u32>(8u, 8u, 1u);
    let num_wgs = num_workgroups;
    
    // Global thread ID
    let global_id = workgroup_id * wg_size + local_id;
    
    // Flatten to 1D index
    return global_id.x 
         + global_id.y * (num_wgs.x * wg_size.x)
         + global_id.z * (num_wgs.x * wg_size.x * num_wgs.y * wg_size.y);
}
```

### 8.2.4 Workgroup Memory

```wgsl
// Shared memory within workgroup
var<workgroup> shared_data: array<f32, 256>;
var<workgroup> histogram: array<atomic<u32>, 256>;
var<workgroup> tile: array<array<vec4<f32>, 16>, 16>;

@compute @workgroup_size(256, 1, 1)
fn reduction() {
    let idx = local_invocation_index;
    
    // Load to shared memory
    shared_data[idx] = input_data[global_invocation_id.x];
    
    // Synchronize
    workgroupBarrier();
    
    // Parallel reduction
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (idx < stride) {
            shared_data[idx] += shared_data[idx + stride];
        }
        workgroupBarrier();
    }
    
    // First thread writes result
    if (idx == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}
```

**Workgroup memory limits:**
```rust
// Check device limits
let max_workgroup_storage = limits.max_compute_workgroup_storage_size; // 16KB typical

// Calculate usage
let shared_bytes = 256 * std::mem::size_of::<f32>(); // 1KB
assert!(shared_bytes <= max_workgroup_storage);
```

### 8.2.5 Synchronization

```wgsl
// Barrier functions

// Full barrier: execution + all memory
workgroupBarrier();

// Memory barriers only
storageBarrier();   // Storage buffer memory
textureBarrier();   // Texture memory (where supported)

// Usage pattern
@compute @workgroup_size(64, 1, 1)
fn multi_pass() {
    let idx = local_invocation_index;
    
    // Phase 1: Load
    shared_data[idx] = load_data(idx);
    workgroupBarrier();  // Wait for all loads
    
    // Phase 2: Process (read neighbors)
    let result = process(shared_data, idx);
    workgroupBarrier();  // Wait for all processing
    
    // Phase 3: Store
    shared_data[idx] = result;
    workgroupBarrier();
    
    // Phase 4: Reduction
    if (idx == 0u) {
        var sum = 0.0;
        for (var i = 0u; i < 64u; i++) {
            sum += shared_data[i];
        }
        output[workgroup_id.x] = sum;
    }
}
```

---

## 8.3 Compute Pass

### 8.3.1 Compute Pass Encoder Creation

```rust
let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
    label: Some("Particle Update Pass"),
    timestamp_writes: Some(ComputePassTimestampWrites {
        query_set: &timestamp_query_set,
        beginning_of_pass_write_index: Some(0),
        end_of_pass_write_index: Some(1),
    }),
});
```

### 8.3.2 Pipeline Binding

```rust
compute_pass.set_pipeline(&compute_pipeline);
```

### 8.3.3 Bind Group Binding

```rust
// Static bindings
compute_pass.set_bind_group(0, &input_bind_group, &[]);
compute_pass.set_bind_group(1, &output_bind_group, &[]);

// Dynamic offsets
compute_pass.set_bind_group(2, &params_bind_group, &[
    frame_uniform_offset,
    material_uniform_offset,
]);
```

### 8.3.4 Push Constants

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ComputePushConstants {
    time: f32,
    delta_time: f32,
    frame: u32,
    _pad: u32,
}

compute_pass.set_push_constants(
    0,  // Offset
    bytemuck::bytes_of(&ComputePushConstants {
        time: 1.5,
        delta_time: 0.016,
        frame: 90,
        _pad: 0,
    }),
);
```

---

## 8.4 Dispatch Commands

### 8.4.1 dispatch_workgroups

```rust
// Direct dispatch
compute_pass.dispatch_workgroups(
    workgroup_count_x,  // Number of workgroups in X
    workgroup_count_y,  // Number of workgroups in Y
    workgroup_count_z,  // Number of workgroups in Z
);

// Common patterns
// Linear data: n elements, 256 threads per workgroup
let workgroups = (element_count + 255) / 256;
compute_pass.dispatch_workgroups(workgroups, 1, 1);

// 2D image: width x height, 8x8 workgroups
let wg_x = (width + 7) / 8;
let wg_y = (height + 7) / 8;
compute_pass.dispatch_workgroups(wg_x, wg_y, 1);

// 3D volume: width x height x depth, 4x4x4 workgroups
let wg_x = (width + 3) / 4;
let wg_y = (height + 3) / 4;
let wg_z = (depth + 3) / 4;
compute_pass.dispatch_workgroups(wg_x, wg_y, wg_z);
```

### 8.4.2 dispatch_workgroups_indirect

```rust
// Indirect dispatch: GPU determines workgroup counts
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DispatchIndirectArgs {
    workgroup_count_x: u32,
    workgroup_count_y: u32,
    workgroup_count_z: u32,
}

let indirect_buffer = device.create_buffer_init(&BufferInitDescriptor {
    label: Some("Dispatch Indirect"),
    contents: bytemuck::bytes_of(&DispatchIndirectArgs {
        workgroup_count_x: 100,
        workgroup_count_y: 1,
        workgroup_count_z: 1,
    }),
    usage: BufferUsages::INDIRECT | BufferUsages::STORAGE,
});

compute_pass.dispatch_workgroups_indirect(&indirect_buffer, 0);
```

**Use case: GPU-driven dispatch counts**
```wgsl
// First pass: compute how many workgroups needed
@compute @workgroup_size(1, 1, 1)
fn compute_dispatch_size() {
    let active_particles = atomicLoad(&particle_count);
    let workgroups = (active_particles + 255u) / 256u;
    
    dispatch_args.workgroup_count_x = workgroups;
    dispatch_args.workgroup_count_y = 1u;
    dispatch_args.workgroup_count_z = 1u;
}

// Second pass: dispatch indirectly
@compute @workgroup_size(256, 1, 1)
fn update_particles() {
    // Process particles
}
```

### 8.4.3 Workgroup Count Limits

```rust
// Check limits
let max_wg_per_dim = limits.max_compute_workgroups_per_dimension; // 65535 typical

// For very large dispatches, may need multiple passes
fn dispatch_large(
    pass: &mut ComputePass,
    total_workgroups: u32,
    max_per_dispatch: u32,
) {
    let mut remaining = total_workgroups;
    let mut offset = 0u32;
    
    while remaining > 0 {
        let batch = remaining.min(max_per_dispatch);
        // Would need to pass offset via push constant or uniform
        pass.set_push_constants(0, bytemuck::bytes_of(&offset));
        pass.dispatch_workgroups(batch, 1, 1);
        remaining -= batch;
        offset += batch;
    }
}
```

### 8.4.4 Dispatch Sizing Strategies

```rust
pub struct DispatchHelper;

impl DispatchHelper {
    pub fn for_elements(count: u32, workgroup_size: u32) -> [u32; 3] {
        [(count + workgroup_size - 1) / workgroup_size, 1, 1]
    }
    
    pub fn for_image(width: u32, height: u32, tile_size: [u32; 2]) -> [u32; 3] {
        [
            (width + tile_size[0] - 1) / tile_size[0],
            (height + tile_size[1] - 1) / tile_size[1],
            1,
        ]
    }
    
    pub fn for_volume(dims: [u32; 3], tile_size: [u32; 3]) -> [u32; 3] {
        [
            (dims[0] + tile_size[0] - 1) / tile_size[0],
            (dims[1] + tile_size[1] - 1) / tile_size[1],
            (dims[2] + tile_size[2] - 1) / tile_size[2],
        ]
    }
}
```

---

## 8.5 Compute Patterns

### 8.5.1 Parallel Reduction

```wgsl
var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn reduce_sum(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    // Load to shared memory
    if (global_id.x < array_size) {
        shared_data[local_idx] = input[global_id.x];
    } else {
        shared_data[local_idx] = 0.0;
    }
    workgroupBarrier();
    
    // Tree reduction
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (local_idx < stride) {
            shared_data[local_idx] += shared_data[local_idx + stride];
        }
        workgroupBarrier();
    }
    
    // Write partial sum
    if (local_idx == 0u) {
        partial_sums[wg_id.x] = shared_data[0];
    }
}
```

### 8.5.2 Prefix Scan (Parallel Prefix Sum)

```wgsl
var<workgroup> temp: array<f32, 512>;

@compute @workgroup_size(256, 1, 1)
fn prefix_sum_block(
    @builtin(local_invocation_index) idx: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let n = 256u;
    let offset = wg_id.x * n;
    
    // Load input
    temp[idx] = input[offset + idx];
    workgroupBarrier();
    
    // Up-sweep (reduce)
    for (var d = n >> 1u; d > 0u; d >>= 1u) {
        if (idx < d) {
            let ai = (idx * 2u + 1u) * (n / d) - 1u;
            let bi = (idx * 2u + 2u) * (n / d) - 1u;
            temp[bi] += temp[ai];
        }
        workgroupBarrier();
    }
    
    // Clear last
    if (idx == 0u) {
        block_sums[wg_id.x] = temp[n - 1u];
        temp[n - 1u] = 0.0;
    }
    workgroupBarrier();
    
    // Down-sweep
    for (var d = 1u; d < n; d <<= 1u) {
        if (idx < d) {
            let ai = (idx * 2u + 1u) * (n / d) - 1u;
            let bi = (idx * 2u + 2u) * (n / d) - 1u;
            let t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        workgroupBarrier();
    }
    
    // Write output (exclusive scan)
    output[offset + idx] = temp[idx];
}
```

### 8.5.3 Stream Compaction

```wgsl
@group(0) @binding(0) var<storage, read> input: array<Particle>;
@group(0) @binding(1) var<storage, read_write> output: array<Particle>;
@group(0) @binding(2) var<storage, read> scan_result: array<u32>;  // Prefix sum of predicates
@group(0) @binding(3) var<storage, read_write> count: atomic<u32>;

@compute @workgroup_size(256, 1, 1)
fn compact(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    if (idx >= particle_count) { return; }
    
    let particle = input[idx];
    
    // Predicate: is particle alive?
    if (particle.lifetime > 0.0) {
        // Get output index from prefix sum
        let out_idx = scan_result[idx];
        output[out_idx] = particle;
        
        // Track count
        if (idx == particle_count - 1u || input[idx + 1u].lifetime <= 0.0) {
            atomicStore(&count, out_idx + 1u);
        }
    }
}
```

### 8.5.4 Radix Sort

```wgsl
// Single digit radix sort pass
var<workgroup> local_histogram: array<atomic<u32>, 16>;  // 4-bit digit = 16 bins
var<workgroup> local_offsets: array<u32, 16>;

@compute @workgroup_size(256, 1, 1)
fn radix_sort_pass(
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let shift = pass_number * 4u;  // Which 4-bit digit
    let idx = wg_id.x * 256u + local_idx;
    
    // Clear histogram
    if (local_idx < 16u) {
        atomicStore(&local_histogram[local_idx], 0u);
    }
    workgroupBarrier();
    
    // Count digits
    if (idx < element_count) {
        let key = keys[idx];
        let digit = (key >> shift) & 0xFu;
        atomicAdd(&local_histogram[digit], 1u);
    }
    workgroupBarrier();
    
    // Compute local offsets (prefix sum of histogram)
    // ... (omitted for brevity)
    
    // Scatter to output
    if (idx < element_count) {
        let key = keys[idx];
        let value = values[idx];
        let digit = (key >> shift) & 0xFu;
        let out_idx = global_offsets[digit] + local_rank;
        output_keys[out_idx] = key;
        output_values[out_idx] = value;
    }
}
```

### 8.5.5 Histogram

```wgsl
var<workgroup> local_histogram: array<atomic<u32>, 256>;

@compute @workgroup_size(256, 1, 1)
fn compute_histogram(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    // Clear local histogram
    atomicStore(&local_histogram[local_idx], 0u);
    workgroupBarrier();
    
    // Count locally
    let pixel_idx = global_id.x;
    if (pixel_idx < image_size) {
        let luminance = compute_luminance(image[pixel_idx]);
        let bin = u32(clamp(luminance * 255.0, 0.0, 255.0));
        atomicAdd(&local_histogram[bin], 1u);
    }
    workgroupBarrier();
    
    // Merge to global
    atomicAdd(&global_histogram[local_idx], atomicLoad(&local_histogram[local_idx]));
}
```

### 8.5.6 Image Processing

```wgsl
// Gaussian blur (separable)
const KERNEL_SIZE: u32 = 5u;
const KERNEL: array<f32, 5> = array(0.0625, 0.25, 0.375, 0.25, 0.0625);

var<workgroup> tile: array<array<vec4<f32>, 72>, 8>;  // 64 + 2*radius

@compute @workgroup_size(64, 8, 1)
fn blur_horizontal(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let radius = 2i;
    let base_x = i32(global_id.x) - i32(local_id.x) - radius;
    
    // Load tile with apron
    let load_x = base_x + i32(local_id.x);
    let y = i32(global_id.y);
    
    // Load center
    tile[local_id.y][local_id.x + 2u] = textureLoad(input_texture, vec2(load_x, y), 0);
    
    // Load left apron
    if (local_id.x < 2u) {
        tile[local_id.y][local_id.x] = textureLoad(input_texture, vec2(load_x - 2, y), 0);
    }
    
    // Load right apron
    if (local_id.x >= 62u) {
        tile[local_id.y][local_id.x + 4u] = textureLoad(input_texture, vec2(load_x + 2, y), 0);
    }
    
    workgroupBarrier();
    
    // Convolve
    var result = vec4(0.0);
    for (var i = 0u; i < KERNEL_SIZE; i++) {
        result += KERNEL[i] * tile[local_id.y][local_id.x + i];
    }
    
    textureStore(output_texture, vec2(global_id.xy), result);
}
```

### 8.5.7 Physics Simulation

```wgsl
struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    lifetime: f32,
    size: f32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

@compute @workgroup_size(256, 1, 1)
fn simulate_particles(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;
    if (idx >= params.particle_count) { return; }
    
    var p = particles[idx];
    
    // Skip dead particles
    if (p.lifetime <= 0.0) { return; }
    
    // Apply forces
    let gravity = vec3(0.0, -9.81, 0.0);
    let drag = -params.drag * p.velocity;
    let acceleration = gravity + drag;
    
    // Integrate
    p.velocity += acceleration * params.delta_time;
    p.position += p.velocity * params.delta_time;
    
    // Update lifetime
    p.lifetime -= params.delta_time;
    
    // Collision with ground
    if (p.position.y < 0.0) {
        p.position.y = 0.0;
        p.velocity.y = -p.velocity.y * params.bounce;
        p.velocity *= params.friction;
    }
    
    particles[idx] = p;
}
```

### 8.5.8 TRINITY's Compute Library

```rust
pub struct ComputeLibrary {
    // Core algorithms
    pub reduce_sum: ComputePipeline,
    pub reduce_min: ComputePipeline,
    pub reduce_max: ComputePipeline,
    pub prefix_sum: ComputePipeline,
    pub radix_sort: ComputePipeline,
    pub stream_compact: ComputePipeline,
    
    // Image processing
    pub blur_horizontal: ComputePipeline,
    pub blur_vertical: ComputePipeline,
    pub downsample: ComputePipeline,
    pub upsample: ComputePipeline,
    pub histogram: ComputePipeline,
    pub tonemapping: ComputePipeline,
    
    // Rendering support
    pub frustum_cull: ComputePipeline,
    pub build_indirect: ComputePipeline,
    pub depth_reduce: ComputePipeline,
    pub light_clustering: ComputePipeline,
    
    // Particle systems
    pub particle_emit: ComputePipeline,
    pub particle_simulate: ComputePipeline,
    pub particle_sort: ComputePipeline,
}

impl ComputeLibrary {
    pub fn new(device: &Device) -> Self {
        // Load and compile all compute shaders
        // ...
    }
    
    pub fn dispatch_reduce_sum(
        &self,
        encoder: &mut CommandEncoder,
        input: &Buffer,
        output: &Buffer,
        count: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass.set_pipeline(&self.reduce_sum);
        
        // Multi-pass reduction for large inputs
        let mut remaining = count;
        let mut current_input = input;
        let mut current_output = output;
        
        while remaining > 1 {
            let workgroups = (remaining + 255) / 256;
            // ... bind groups and dispatch
            remaining = workgroups;
            std::mem::swap(&mut current_input, &mut current_output);
        }
    }
}
```

---

# TRINITY Compute Module Architecture

```
crates/renderer-backend/src/compute/
├── mod.rs              # Module root
├── pipeline.rs         # Compute pipeline management
├── dispatch.rs         # Dispatch helpers
├── algorithms/
│   ├── mod.rs
│   ├── reduce.rs       # Parallel reduction
│   ├── scan.rs         # Prefix scan
│   ├── sort.rs         # Radix sort
│   └── compact.rs      # Stream compaction
├── image/
│   ├── mod.rs
│   ├── blur.rs         # Gaussian blur
│   ├── downsample.rs   # Mip generation
│   └── histogram.rs    # Histogram compute
└── culling/
    ├── mod.rs
    ├── frustum.rs      # Frustum culling
    └── occlusion.rs    # HiZ occlusion
```

---

*End of WGPU_PART_V_COMPUTE.md*

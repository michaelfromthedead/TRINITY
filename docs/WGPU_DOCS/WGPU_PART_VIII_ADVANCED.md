# WGPU_PART_VIII_ADVANCED.md — Advanced Rendering Techniques

> **Scope**: GPU-driven rendering, indirect draw, mesh shaders (future), bindless resources
> **TRINITY Integration**: GPU culling pipeline, bindless material system, meshlet architecture
> **wgpu Version**: 25.x+ with experimental features

---

# Chapter 15: Indirect Rendering

Indirect rendering moves draw call generation from CPU to GPU, enabling massive scalability for complex scenes. Instead of the CPU specifying draw parameters, the GPU reads them from buffers populated by compute shaders.

---

## 15.1 Indirect Draw

### 15.1.1 DrawIndirect Buffer Layout

The `DrawIndirect` structure matches the GPU's expected layout:

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndirect {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

impl DrawIndirect {
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64; // 16 bytes
    
    pub fn new(vertex_count: u32, instance_count: u32) -> Self {
        Self {
            vertex_count,
            instance_count,
            first_vertex: 0,
            first_instance: 0,
        }
    }
}
```

**Usage in Render Pass**:

```rust
impl<'a> RenderPass<'a> {
    pub fn draw_indirect(&mut self, indirect_buffer: &Buffer, indirect_offset: BufferAddress);
}

// Example: Draw using GPU-generated parameters
render_pass.set_pipeline(&pipeline);
render_pass.set_bind_group(0, &bind_group, &[]);
render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
render_pass.draw_indirect(&indirect_buffer, 0);
```

### 15.1.2 DrawIndexedIndirect Buffer Layout

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirect {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,  // Signed!
    pub first_instance: u32,
}

impl DrawIndexedIndirect {
    pub const SIZE: u64 = std::mem::size_of::<Self>() as u64; // 20 bytes
    
    pub fn new(index_count: u32, instance_count: u32) -> Self {
        Self {
            index_count,
            instance_count,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        }
    }
}
```

**Usage**:

```rust
render_pass.set_pipeline(&pipeline);
render_pass.set_bind_group(0, &bind_group, &[]);
render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
render_pass.draw_indexed_indirect(&indirect_buffer, 0);
```

### 15.1.3 GPU-Driven Draw Call Generation

**Compute Shader for Draw Generation**:

```wgsl
struct DrawIndexedIndirect {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct MeshInfo {
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    _pad: u32,
}

struct InstanceData {
    mesh_index: u32,
    lod_level: u32,
    transform_index: u32,
    material_index: u32,
}

@group(0) @binding(0) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(1) var<storage, read> mesh_infos: array<MeshInfo>;
@group(0) @binding(2) var<storage, read> visibility: array<u32>;
@group(0) @binding(3) var<storage, read_write> draw_commands: array<DrawIndexedIndirect>;
@group(0) @binding(4) var<storage, read_write> draw_count: atomic<u32>;

@compute @workgroup_size(256)
fn generate_draws(@builtin(global_invocation_id) gid: vec3<u32>) {
    let instance_index = gid.x;
    if (instance_index >= arrayLength(&instances)) {
        return;
    }
    
    // Check visibility (from culling pass)
    let word_index = instance_index / 32u;
    let bit_index = instance_index % 32u;
    let visible = (visibility[word_index] & (1u << bit_index)) != 0u;
    
    if (!visible) {
        return;
    }
    
    // Allocate draw command slot
    let draw_index = atomicAdd(&draw_count, 1u);
    
    let instance = instances[instance_index];
    let mesh = mesh_infos[instance.mesh_index];
    
    draw_commands[draw_index] = DrawIndexedIndirect(
        mesh.index_count,
        1u,  // Single instance per draw
        mesh.first_index,
        mesh.base_vertex,
        instance_index  // first_instance encodes instance ID
    );
}
```

### 15.1.4 Indirect Count (Where Supported)

The `MULTI_DRAW_INDIRECT_COUNT` feature enables dynamic draw count:

```rust
// Feature: MULTI_DRAW_INDIRECT_COUNT
impl<'a> RenderPass<'a> {
    pub fn multi_draw_indirect_count(
        &mut self,
        indirect_buffer: &Buffer,
        indirect_offset: BufferAddress,
        count_buffer: &Buffer,
        count_offset: BufferAddress,
        max_count: u32,
    );
    
    pub fn multi_draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &Buffer,
        indirect_offset: BufferAddress,
        count_buffer: &Buffer,
        count_offset: BufferAddress,
        max_count: u32,
    );
}
```

**TRINITY Indirect Draw System**:

```rust
pub struct IndirectDrawBuffer {
    draw_buffer: wgpu::Buffer,
    count_buffer: wgpu::Buffer,
    max_draws: u32,
    draw_type: IndirectDrawType,
}

pub enum IndirectDrawType {
    Draw,
    DrawIndexed,
}

impl IndirectDrawBuffer {
    pub fn new(device: &wgpu::Device, max_draws: u32, draw_type: IndirectDrawType) -> Self {
        let draw_size = match draw_type {
            IndirectDrawType::Draw => DrawIndirect::SIZE,
            IndirectDrawType::DrawIndexed => DrawIndexedIndirect::SIZE,
        };
        
        let draw_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IndirectDrawBuffer"),
            size: draw_size * max_draws as u64,
            usage: wgpu::BufferUsages::STORAGE 
                 | wgpu::BufferUsages::INDIRECT 
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IndirectCountBuffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE 
                 | wgpu::BufferUsages::INDIRECT 
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        Self {
            draw_buffer,
            count_buffer,
            max_draws,
            draw_type,
        }
    }
    
    pub fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.count_buffer, 0, None);
    }
    
    pub fn execute<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        match self.draw_type {
            IndirectDrawType::Draw => {
                pass.multi_draw_indirect_count(
                    &self.draw_buffer,
                    0,
                    &self.count_buffer,
                    0,
                    self.max_draws,
                );
            }
            IndirectDrawType::DrawIndexed => {
                pass.multi_draw_indexed_indirect_count(
                    &self.draw_buffer,
                    0,
                    &self.count_buffer,
                    0,
                    self.max_draws,
                );
            }
        }
    }
}
```

---

## 15.2 GPU Culling

### 15.2.1 Frustum Culling in Compute

**Frustum Planes Extraction**:

```wgsl
struct FrustumPlanes {
    planes: array<vec4<f32>, 6>,  // left, right, bottom, top, near, far
}

fn extract_frustum_planes(view_proj: mat4x4<f32>) -> FrustumPlanes {
    var planes: FrustumPlanes;
    
    // Left plane
    planes.planes[0] = vec4<f32>(
        view_proj[0][3] + view_proj[0][0],
        view_proj[1][3] + view_proj[1][0],
        view_proj[2][3] + view_proj[2][0],
        view_proj[3][3] + view_proj[3][0]
    );
    
    // Right plane
    planes.planes[1] = vec4<f32>(
        view_proj[0][3] - view_proj[0][0],
        view_proj[1][3] - view_proj[1][0],
        view_proj[2][3] - view_proj[2][0],
        view_proj[3][3] - view_proj[3][0]
    );
    
    // Bottom plane
    planes.planes[2] = vec4<f32>(
        view_proj[0][3] + view_proj[0][1],
        view_proj[1][3] + view_proj[1][1],
        view_proj[2][3] + view_proj[2][1],
        view_proj[3][3] + view_proj[3][1]
    );
    
    // Top plane
    planes.planes[3] = vec4<f32>(
        view_proj[0][3] - view_proj[0][1],
        view_proj[1][3] - view_proj[1][1],
        view_proj[2][3] - view_proj[2][1],
        view_proj[3][3] - view_proj[3][1]
    );
    
    // Near plane
    planes.planes[4] = vec4<f32>(
        view_proj[0][3] + view_proj[0][2],
        view_proj[1][3] + view_proj[1][2],
        view_proj[2][3] + view_proj[2][2],
        view_proj[3][3] + view_proj[3][2]
    );
    
    // Far plane
    planes.planes[5] = vec4<f32>(
        view_proj[0][3] - view_proj[0][2],
        view_proj[1][3] - view_proj[1][2],
        view_proj[2][3] - view_proj[2][2],
        view_proj[3][3] - view_proj[3][2]
    );
    
    // Normalize planes
    for (var i = 0u; i < 6u; i = i + 1u) {
        let length = length(planes.planes[i].xyz);
        planes.planes[i] = planes.planes[i] / length;
    }
    
    return planes;
}
```

**AABB Frustum Test**:

```wgsl
struct AABB {
    min: vec3<f32>,
    max: vec3<f32>,
}

fn frustum_cull_aabb(aabb: AABB, frustum: FrustumPlanes) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = frustum.planes[i];
        
        // Find the positive vertex (furthest along plane normal)
        var p = aabb.min;
        if (plane.x >= 0.0) { p.x = aabb.max.x; }
        if (plane.y >= 0.0) { p.y = aabb.max.y; }
        if (plane.z >= 0.0) { p.z = aabb.max.z; }
        
        // If positive vertex is outside, AABB is culled
        if (dot(plane.xyz, p) + plane.w < 0.0) {
            return false;  // Culled
        }
    }
    return true;  // Visible
}
```

**Complete Frustum Culling Shader**:

```wgsl
struct CullUniforms {
    view_proj: mat4x4<f32>,
    frustum: FrustumPlanes,
    instance_count: u32,
}

struct BoundingSphere {
    center: vec3<f32>,
    radius: f32,
}

@group(0) @binding(0) var<uniform> uniforms: CullUniforms;
@group(0) @binding(1) var<storage, read> bounds: array<BoundingSphere>;
@group(0) @binding(2) var<storage, read> transforms: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read_write> visibility: array<atomic<u32>>;

@compute @workgroup_size(256)
fn frustum_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let instance_id = gid.x;
    if (instance_id >= uniforms.instance_count) {
        return;
    }
    
    let bound = bounds[instance_id];
    let transform = transforms[instance_id];
    
    // Transform bounding sphere center to world space
    let world_center = (transform * vec4<f32>(bound.center, 1.0)).xyz;
    
    // Approximate radius scaling (max scale component)
    let scale = max(
        max(length(transform[0].xyz), length(transform[1].xyz)),
        length(transform[2].xyz)
    );
    let world_radius = bound.radius * scale;
    
    // Test against frustum planes
    var visible = true;
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = uniforms.frustum.planes[i];
        let distance = dot(plane.xyz, world_center) + plane.w;
        if (distance < -world_radius) {
            visible = false;
            break;
        }
    }
    
    // Set visibility bit
    if (visible) {
        let word_index = instance_id / 32u;
        let bit_index = instance_id % 32u;
        atomicOr(&visibility[word_index], 1u << bit_index);
    }
}
```

### 15.2.2 Occlusion Culling with Hierarchical-Z

**HiZ Pyramid Generation**:

```wgsl
@group(0) @binding(0) var input_depth: texture_2d<f32>;
@group(0) @binding(1) var output_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> mip_size: vec2<u32>;

@compute @workgroup_size(8, 8)
fn generate_hiz_mip(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= mip_size.x || gid.y >= mip_size.y) {
        return;
    }
    
    let src_coord = vec2<i32>(gid.xy) * 2;
    
    // Sample 4 depth values from previous mip
    let d0 = textureLoad(input_depth, src_coord + vec2<i32>(0, 0), 0).r;
    let d1 = textureLoad(input_depth, src_coord + vec2<i32>(1, 0), 0).r;
    let d2 = textureLoad(input_depth, src_coord + vec2<i32>(0, 1), 0).r;
    let d3 = textureLoad(input_depth, src_coord + vec2<i32>(1, 1), 0).r;
    
    // For reverse-Z: take minimum (furthest)
    // For standard Z: take maximum (furthest)
    let max_depth = max(max(d0, d1), max(d2, d3));
    
    textureStore(output_depth, vec2<i32>(gid.xy), vec4<f32>(max_depth, 0.0, 0.0, 1.0));
}
```

**Occlusion Test Against HiZ**:

```wgsl
@group(0) @binding(0) var hiz_pyramid: texture_2d<f32>;
@group(0) @binding(1) var hiz_sampler: sampler;

fn is_occluded(aabb_screen: vec4<f32>, aabb_depth: f32) -> bool {
    // aabb_screen: (min_x, min_y, max_x, max_y) in [0,1]
    
    let size = aabb_screen.zw - aabb_screen.xy;
    let screen_size = vec2<f32>(textureDimensions(hiz_pyramid, 0));
    let pixel_size = size * screen_size;
    
    // Select mip level based on projected size
    let mip = log2(max(pixel_size.x, pixel_size.y));
    let mip_level = u32(ceil(mip));
    
    // Sample HiZ at AABB center
    let center = (aabb_screen.xy + aabb_screen.zw) * 0.5;
    let hiz_depth = textureSampleLevel(hiz_pyramid, hiz_sampler, center, f32(mip_level)).r;
    
    // For reverse-Z: if object depth > hiz depth, it's occluded
    return aabb_depth > hiz_depth;
}
```

### 15.2.3 GPU-Driven LOD Selection

```wgsl
struct LODInfo {
    lod_distances: array<f32, 8>,  // Distance thresholds for each LOD
    lod_count: u32,
}

struct MeshLOD {
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    lod_level: u32,
}

@group(0) @binding(0) var<uniform> camera_pos: vec3<f32>;
@group(0) @binding(1) var<storage, read> lod_info: LODInfo;
@group(0) @binding(2) var<storage, read> instance_positions: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read> mesh_lods: array<MeshLOD>;  // All LODs for all meshes
@group(0) @binding(4) var<storage, read_write> selected_draws: array<DrawIndexedIndirect>;
@group(0) @binding(5) var<storage, read_write> draw_count: atomic<u32>;

struct InstanceInfo {
    mesh_id: u32,
    lods_offset: u32,  // Offset into mesh_lods array
}

@group(0) @binding(6) var<storage, read> instance_info: array<InstanceInfo>;

@compute @workgroup_size(256)
fn select_lod(@builtin(global_invocation_id) gid: vec3<u32>) {
    let instance_id = gid.x;
    if (instance_id >= arrayLength(&instance_positions)) {
        return;
    }
    
    let pos = instance_positions[instance_id];
    let distance = length(pos - camera_pos);
    
    // Find appropriate LOD level
    var lod_level = lod_info.lod_count - 1u;
    for (var i = 0u; i < lod_info.lod_count; i = i + 1u) {
        if (distance < lod_info.lod_distances[i]) {
            lod_level = i;
            break;
        }
    }
    
    let info = instance_info[instance_id];
    let mesh_lod = mesh_lods[info.lods_offset + lod_level];
    
    // Allocate and write draw command
    let draw_index = atomicAdd(&draw_count, 1u);
    selected_draws[draw_index] = DrawIndexedIndirect(
        mesh_lod.index_count,
        1u,
        mesh_lod.first_index,
        mesh_lod.base_vertex,
        instance_id
    );
}
```

### 15.2.4 Indirect Buffer Compaction

Stream compaction to remove gaps in indirect draw buffer:

```wgsl
struct CompactionUniforms {
    input_count: u32,
}

@group(0) @binding(0) var<uniform> uniforms: CompactionUniforms;
@group(0) @binding(1) var<storage, read> visibility: array<u32>;
@group(0) @binding(2) var<storage, read> input_draws: array<DrawIndexedIndirect>;
@group(0) @binding(3) var<storage, read_write> output_draws: array<DrawIndexedIndirect>;
@group(0) @binding(4) var<storage, read_write> output_count: atomic<u32>;

var<workgroup> prefix_sum: array<u32, 256>;
var<workgroup> workgroup_total: u32;

@compute @workgroup_size(256)
fn compact_draws(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let global_id = gid.x;
    let local_id = lid.x;
    
    // Load visibility bit
    var visible = 0u;
    if (global_id < uniforms.input_count) {
        let word = visibility[global_id / 32u];
        let bit = global_id % 32u;
        visible = (word >> bit) & 1u;
    }
    
    // Workgroup-level prefix sum
    prefix_sum[local_id] = visible;
    workgroupBarrier();
    
    // Parallel prefix sum (up-sweep)
    for (var stride = 1u; stride < 256u; stride = stride * 2u) {
        let idx = (local_id + 1u) * stride * 2u - 1u;
        if (idx < 256u) {
            prefix_sum[idx] += prefix_sum[idx - stride];
        }
        workgroupBarrier();
    }
    
    // Down-sweep
    if (local_id == 255u) {
        workgroup_total = prefix_sum[255u];
        prefix_sum[255u] = 0u;
    }
    workgroupBarrier();
    
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        let idx = (local_id + 1u) * stride * 2u - 1u;
        if (idx < 256u) {
            let temp = prefix_sum[idx];
            prefix_sum[idx] += prefix_sum[idx - stride];
            prefix_sum[idx - stride] = temp;
        }
        workgroupBarrier();
    }
    
    // Allocate global slots for this workgroup
    var global_offset: u32;
    if (local_id == 0u) {
        global_offset = atomicAdd(&output_count, workgroup_total);
    }
    workgroupBarrier();
    
    // Write visible draws
    if (visible == 1u && global_id < uniforms.input_count) {
        let output_index = global_offset + prefix_sum[local_id];
        output_draws[output_index] = input_draws[global_id];
    }
}
```

### 15.2.5 TRINITY's GPU Culling Pipeline

```rust
pub struct GPUCullingPipeline {
    frustum_cull_pipeline: wgpu::ComputePipeline,
    hiz_generate_pipeline: wgpu::ComputePipeline,
    occlusion_cull_pipeline: wgpu::ComputePipeline,
    lod_select_pipeline: wgpu::ComputePipeline,
    compact_pipeline: wgpu::ComputePipeline,
    
    visibility_buffer: wgpu::Buffer,
    indirect_buffer: IndirectDrawBuffer,
    hiz_pyramid: HiZPyramid,
}

pub struct HiZPyramid {
    texture: wgpu::Texture,
    views: Vec<wgpu::TextureView>,
    mip_count: u32,
    size: (u32, u32),
}

impl HiZPyramid {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let mip_count = (width.max(height) as f32).log2().ceil() as u32;
        
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HiZ Pyramid"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING 
                 | wgpu::TextureUsages::STORAGE_BINDING
                 | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let views: Vec<_> = (0..mip_count)
            .map(|mip| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        
        Self {
            texture,
            views,
            mip_count,
            size: (width, height),
        }
    }
}

impl GPUCullingPipeline {
    pub fn cull_frame(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &SceneData,
        camera: &CameraData,
    ) {
        // Phase 1: Frustum culling
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Frustum Cull"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.frustum_cull_pipeline);
            pass.set_bind_group(0, &self.frustum_cull_bind_group, &[]);
            pass.dispatch_workgroups((scene.instance_count + 255) / 256, 1, 1);
        }
        
        // Phase 2: Generate HiZ pyramid (from previous frame's depth)
        self.generate_hiz_pyramid(encoder);
        
        // Phase 3: Occlusion culling against HiZ
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Occlusion Cull"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.occlusion_cull_pipeline);
            pass.set_bind_group(0, &self.occlusion_cull_bind_group, &[]);
            pass.dispatch_workgroups((scene.instance_count + 255) / 256, 1, 1);
        }
        
        // Phase 4: LOD selection and draw generation
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LOD Select"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.lod_select_pipeline);
            pass.set_bind_group(0, &self.lod_select_bind_group, &[]);
            pass.dispatch_workgroups((scene.instance_count + 255) / 256, 1, 1);
        }
    }
    
    fn generate_hiz_pyramid(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut width = self.hiz_pyramid.size.0;
        let mut height = self.hiz_pyramid.size.1;
        
        for mip in 1..self.hiz_pyramid.mip_count {
            width = (width + 1) / 2;
            height = (height + 1) / 2;
            
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("HiZ Mip {}", mip)),
                timestamp_writes: None,
            });
            
            pass.set_pipeline(&self.hiz_generate_pipeline);
            // Bind previous mip as input, current mip as output
            pass.set_bind_group(0, &self.hiz_bind_groups[mip as usize - 1], &[]);
            pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
        }
    }
}
```

---

## 15.3 Multi-Draw Indirect

### 15.3.1 multi_draw_indirect Feature

```rust
// Feature: MULTI_DRAW_INDIRECT
impl<'a> RenderPass<'a> {
    pub fn multi_draw_indirect(
        &mut self,
        indirect_buffer: &Buffer,
        indirect_offset: BufferAddress,
        count: u32,
    );
    
    pub fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &Buffer,
        indirect_offset: BufferAddress,
        count: u32,
    );
}
```

### 15.3.2 Batching Multiple Draws

```rust
pub struct MultiDrawBatch {
    draws: Vec<DrawIndexedIndirect>,
    buffer: Option<wgpu::Buffer>,
}

impl MultiDrawBatch {
    pub fn new() -> Self {
        Self {
            draws: Vec::new(),
            buffer: None,
        }
    }
    
    pub fn add_draw(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) {
        self.draws.push(DrawIndexedIndirect {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        });
    }
    
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.draws.is_empty() {
            return;
        }
        
        let size = self.draws.len() as u64 * DrawIndexedIndirect::SIZE;
        
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MultiDrawBatch"),
            size,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&self.draws));
        self.buffer = Some(buffer);
    }
    
    pub fn execute<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if let Some(ref buffer) = self.buffer {
            pass.multi_draw_indexed_indirect(buffer, 0, self.draws.len() as u32);
        }
    }
}
```

### 15.3.3 Performance Implications

| Approach | CPU Cost | GPU Cost | Best For |
|----------|----------|----------|----------|
| Individual draw calls | High | Low | Few objects |
| Multi-draw (CPU count) | Medium | Low | Medium scenes |
| Multi-draw count (GPU count) | Low | Low | Large scenes |
| Indirect with compaction | Lowest | Medium | Massive scenes |

---

# Chapter 16: Mesh Shaders (Future)

Mesh shaders represent the next evolution in geometry processing, replacing the traditional vertex-input/tessellation/geometry pipeline with a more flexible task/mesh model.

---

## 16.1 Mesh Shader Fundamentals

### 16.1.1 mesh_shaders Feature (Not Yet in wgpu)

```rust
// Future feature (not currently available)
wgpu::Features::MESH_SHADER_TIER_1
wgpu::Features::MESH_SHADER_TIER_2
```

**Estimated Timeline**: wgpu mesh shader support is dependent on:
- Vulkan: VK_EXT_mesh_shader (widely supported on modern GPUs)
- Metal: Mesh shaders (Metal 3+, Apple Silicon)
- DX12: D3D12 mesh shaders (available)

### 16.1.2 Task Shader Stage

Task shaders (also called amplification shaders) run before mesh shaders:
- One invocation per task
- Can spawn 0 or more mesh shader workgroups
- Ideal for coarse culling (meshlet-level)

```wgsl
// Future WGSL syntax (speculative)
@task @workgroup_size(32)
fn task_main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let meshlet_index = wgid.x * 32u + lid.x;
    
    // Cull meshlet
    let visible = frustum_test(meshlet_bounds[meshlet_index]);
    
    // Vote on visibility
    let ballot = subgroupBallot(visible);
    let visible_count = countOneBits(ballot);
    
    // Dispatch mesh shader workgroups for visible meshlets
    if (lid.x == 0u) {
        dispatchMesh(visible_count, 1u, 1u);
    }
}
```

### 16.1.3 Mesh Shader Stage

Mesh shaders output primitives and vertices directly:

```wgsl
// Future WGSL syntax (speculative)
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

@mesh @workgroup_size(128)
fn mesh_main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let meshlet_index = task_payload.meshlet_indices[wgid.x];
    let meshlet = meshlets[meshlet_index];
    
    // Output vertices
    if (lid.x < meshlet.vertex_count) {
        let vertex = load_vertex(meshlet, lid.x);
        setMeshOutputVertex(lid.x, transform_vertex(vertex));
    }
    
    // Output primitives (triangles)
    if (lid.x < meshlet.triangle_count) {
        let tri = load_triangle_indices(meshlet, lid.x);
        setMeshOutputPrimitive(lid.x, tri);
    }
    
    // Set output counts
    if (lid.x == 0u) {
        setMeshOutputCounts(meshlet.vertex_count, meshlet.triangle_count);
    }
}
```

### 16.1.4 Meshlet Concept

A meshlet is a small cluster of triangles that fits in a mesh shader workgroup:

```rust
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Meshlet {
    pub vertex_offset: u32,
    pub triangle_offset: u32,
    pub vertex_count: u8,
    pub triangle_count: u8,
    pub _padding: [u8; 2],
}

impl Meshlet {
    pub const MAX_VERTICES: usize = 64;
    pub const MAX_TRIANGLES: usize = 124;  // 126 for some GPUs
}
```

---

## 16.2 Meshlet Pipeline

### 16.2.1 Meshlet Generation

```rust
pub struct MeshletBuilder {
    max_vertices: usize,
    max_triangles: usize,
}

impl MeshletBuilder {
    pub fn new(max_vertices: usize, max_triangles: usize) -> Self {
        Self { max_vertices, max_triangles }
    }
    
    pub fn build(&self, indices: &[u32], vertices: &[Vertex]) -> MeshletMesh {
        let mut meshlets = Vec::new();
        let mut meshlet_vertices = Vec::new();
        let mut meshlet_triangles = Vec::new();
        
        let mut current_vertices: HashMap<u32, u8> = HashMap::new();
        let mut current_triangles: Vec<[u8; 3]> = Vec::new();
        
        for triangle in indices.chunks(3) {
            let can_add = self.can_add_triangle(
                &current_vertices,
                triangle,
            );
            
            if !can_add {
                // Flush current meshlet
                self.flush_meshlet(
                    &mut meshlets,
                    &mut meshlet_vertices,
                    &mut meshlet_triangles,
                    &current_vertices,
                    &current_triangles,
                );
                current_vertices.clear();
                current_triangles.clear();
            }
            
            // Add triangle to current meshlet
            let local_indices = self.add_triangle(
                &mut current_vertices,
                triangle,
            );
            current_triangles.push(local_indices);
        }
        
        // Flush final meshlet
        if !current_triangles.is_empty() {
            self.flush_meshlet(
                &mut meshlets,
                &mut meshlet_vertices,
                &mut meshlet_triangles,
                &current_vertices,
                &current_triangles,
            );
        }
        
        // Compute bounds for each meshlet
        let meshlet_bounds = self.compute_bounds(&meshlets, &meshlet_vertices, vertices);
        
        MeshletMesh {
            meshlets,
            meshlet_vertices,
            meshlet_triangles,
            meshlet_bounds,
        }
    }
    
    fn can_add_triangle(
        &self,
        current_vertices: &HashMap<u32, u8>,
        triangle: &[u32],
    ) -> bool {
        let mut new_vertex_count = 0;
        for &idx in triangle {
            if !current_vertices.contains_key(&idx) {
                new_vertex_count += 1;
            }
        }
        
        current_vertices.len() + new_vertex_count <= self.max_vertices
    }
    
    fn add_triangle(
        &self,
        current_vertices: &mut HashMap<u32, u8>,
        triangle: &[u32],
    ) -> [u8; 3] {
        let mut local = [0u8; 3];
        for (i, &idx) in triangle.iter().enumerate() {
            let local_idx = *current_vertices.entry(idx).or_insert_with(|| {
                current_vertices.len() as u8
            });
            local[i] = local_idx;
        }
        local
    }
}

pub struct MeshletMesh {
    pub meshlets: Vec<Meshlet>,
    pub meshlet_vertices: Vec<u32>,      // Global vertex indices
    pub meshlet_triangles: Vec<u8>,      // Packed triangle indices (3 bytes per tri)
    pub meshlet_bounds: Vec<BoundingSphere>,
}
```

### 16.2.2 Meshlet Culling (Task Shader)

```rust
pub struct MeshletCuller {
    cull_pipeline: wgpu::ComputePipeline,  // Fallback for non-mesh-shader path
}

impl MeshletCuller {
    pub fn cull(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        meshlet_count: u32,
        frustum: &FrustumPlanes,
    ) {
        // For fallback path without mesh shaders
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Meshlet Cull"),
            timestamp_writes: None,
        });
        
        pass.set_pipeline(&self.cull_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups((meshlet_count + 255) / 256, 1, 1);
    }
}
```

### 16.2.3 Meshlet Rendering (Mesh Shader)

For the fallback path without mesh shaders:

```rust
pub struct MeshletRenderer {
    fallback_pipeline: wgpu::RenderPipeline,
    meshlet_index_buffer: wgpu::Buffer,
}

impl MeshletRenderer {
    pub fn render_fallback<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        meshlet_mesh: &'a MeshletMesh,
        visible_meshlet_count: u32,
    ) {
        // Fallback: expand meshlets to regular index buffer
        // Each meshlet's triangles are expanded at draw time
        pass.set_pipeline(&self.fallback_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        
        // Use indirect draw with visibility results
        pass.draw_indexed_indirect(&self.indirect_buffer, 0);
    }
}
```

### 16.2.4 Vertex Deduplication

```rust
impl MeshletBuilder {
    pub fn optimize_meshlet_vertices(
        &self,
        meshlet: &mut Meshlet,
        vertices: &mut Vec<u32>,
        triangles: &[u8],
    ) {
        // Remove unused vertices from meshlet
        let mut used = vec![false; meshlet.vertex_count as usize];
        
        for tri in triangles.chunks(3) {
            used[tri[0] as usize] = true;
            used[tri[1] as usize] = true;
            used[tri[2] as usize] = true;
        }
        
        // Build compaction map
        let mut remap = vec![0u8; meshlet.vertex_count as usize];
        let mut new_count = 0u8;
        
        for (i, &is_used) in used.iter().enumerate() {
            if is_used {
                remap[i] = new_count;
                new_count += 1;
            }
        }
        
        // Compact vertices
        vertices.retain_mut(|_| {
            // Logic to compact vertex buffer
            true
        });
        
        meshlet.vertex_count = new_count;
    }
}
```

---

## 16.3 TRINITY's Mesh Shader Readiness

### 16.3.1 Meshlet Preprocessing

```rust
pub struct MeshletPreprocessor {
    builder: MeshletBuilder,
}

impl MeshletPreprocessor {
    pub fn new() -> Self {
        Self {
            builder: MeshletBuilder::new(
                Meshlet::MAX_VERTICES,
                Meshlet::MAX_TRIANGLES,
            ),
        }
    }
    
    pub fn preprocess_mesh(
        &self,
        mesh: &ImportedMesh,
    ) -> MeshletMesh {
        // Optimize vertex cache
        let optimized_indices = meshopt::optimize_vertex_cache(
            &mesh.indices,
            mesh.vertices.len(),
        );
        
        // Build meshlets
        let meshlet_mesh = self.builder.build(
            &optimized_indices,
            &mesh.vertices,
        );
        
        meshlet_mesh
    }
}
```

### 16.3.2 Fallback to Traditional Pipeline

```rust
pub enum GeometryPath {
    MeshShaders,    // Native mesh shader support
    ComputeCull,    // Compute-based culling + traditional draw
    CPUCull,        // CPU culling + traditional draw
}

impl TrinityRenderer {
    pub fn select_geometry_path(&self, features: &wgpu::Features) -> GeometryPath {
        if features.contains(wgpu::Features::MESH_SHADER_TIER_1) {
            GeometryPath::MeshShaders
        } else if features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT) {
            GeometryPath::ComputeCull
        } else {
            GeometryPath::CPUCull
        }
    }
}
```

### 16.3.3 Abstraction Layer

```rust
pub trait GeometryRenderer {
    fn prepare(&mut self, scene: &Scene, camera: &Camera);
    fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>);
}

pub struct MeshShaderRenderer {
    // Future: mesh shader pipeline
}

pub struct IndirectRenderer {
    cull_pipeline: GPUCullingPipeline,
    indirect_buffer: IndirectDrawBuffer,
}

pub struct TraditionalRenderer {
    draw_calls: Vec<DrawCall>,
}

impl GeometryRenderer for IndirectRenderer {
    fn prepare(&mut self, scene: &Scene, camera: &Camera) {
        // GPU culling in compute shader
    }
    
    fn render<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        self.indirect_buffer.execute(pass);
    }
}
```

---

# Chapter 17: Bindless Resources

Bindless rendering eliminates the overhead of binding descriptors by indexing into large resource arrays directly from shaders.

---

## 17.1 Bindless Fundamentals

### 17.1.1 Bindless Texture Arrays

wgpu supports texture arrays that can be indexed dynamically:

```wgsl
@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 1024>;
@group(0) @binding(1) var samplers: binding_array<sampler, 16>;

@fragment
fn bindless_fragment(
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) texture_index: u32,
    @location(2) @interpolate(flat) sampler_index: u32,
) -> @location(0) vec4<f32> {
    return textureSample(
        textures[texture_index],
        samplers[sampler_index],
        uv
    );
}
```

**Required Features**:
```rust
wgpu::Features::TEXTURE_BINDING_ARRAY
wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
```

### 17.1.2 Bindless Buffer Arrays

```wgsl
struct MaterialData {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    emissive: vec3<f32>,
}

@group(0) @binding(0) var<storage, read> materials: array<MaterialData>;

@fragment
fn material_fragment(
    @location(0) @interpolate(flat) material_index: u32,
) -> @location(0) vec4<f32> {
    let mat = materials[material_index];
    return mat.base_color;
}
```

### 17.1.3 Descriptor Indexing

```wgsl
// Non-uniform indexing requires explicit derivative operations
@fragment
fn non_uniform_sample(
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) tex_idx: u32,
) -> @location(0) vec4<f32> {
    // With non-uniform indexing feature
    return textureSample(textures[tex_idx], default_sampler, uv);
    
    // Without feature (undefined behavior if tex_idx varies within quad)
}
```

### 17.1.4 Non-Uniform Indexing

```rust
// Required for non-uniform resource access
wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING

// Check support
if adapter.features().contains(
    wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
) {
    // Safe to use non-uniform indexing
}
```

---

## 17.2 Implementation Patterns

### 17.2.1 Texture Atlas Approach

Pack multiple textures into a single large texture:

```rust
pub struct TextureAtlas {
    texture: wgpu::Texture,
    allocator: AtlasAllocator,
    entries: HashMap<TextureId, AtlasEntry>,
}

pub struct AtlasEntry {
    pub uv_offset: (f32, f32),
    pub uv_scale: (f32, f32),
    pub layer: u32,  // For array textures
}

impl TextureAtlas {
    pub fn new(device: &wgpu::Device, size: u32, layers: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("TextureAtlas"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: layers,
            },
            mip_level_count: (size as f32).log2() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING 
                 | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        Self {
            texture,
            allocator: AtlasAllocator::new(size, layers),
            entries: HashMap::new(),
        }
    }
    
    pub fn allocate(
        &mut self,
        id: TextureId,
        width: u32,
        height: u32,
    ) -> Option<AtlasEntry> {
        let allocation = self.allocator.allocate(width, height)?;
        
        let entry = AtlasEntry {
            uv_offset: (
                allocation.x as f32 / self.allocator.size as f32,
                allocation.y as f32 / self.allocator.size as f32,
            ),
            uv_scale: (
                width as f32 / self.allocator.size as f32,
                height as f32 / self.allocator.size as f32,
            ),
            layer: allocation.layer,
        };
        
        self.entries.insert(id, entry.clone());
        Some(entry)
    }
}
```

### 17.2.2 Texture Array Approach

Use texture arrays with dynamic indexing:

```rust
pub struct BindlessTextureArray {
    texture_array: wgpu::Texture,
    view: wgpu::TextureView,
    free_slots: Vec<u32>,
    texture_map: HashMap<TextureId, u32>,
    max_textures: u32,
}

impl BindlessTextureArray {
    pub fn new(device: &wgpu::Device, size: u32, max_textures: u32) -> Self {
        let texture_array = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("BindlessTextureArray"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: max_textures,
            },
            mip_level_count: (size as f32).log2() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING 
                 | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let view = texture_array.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        
        Self {
            texture_array,
            view,
            free_slots: (0..max_textures).rev().collect(),
            texture_map: HashMap::new(),
            max_textures,
        }
    }
    
    pub fn allocate(&mut self, id: TextureId) -> Option<u32> {
        if self.texture_map.contains_key(&id) {
            return self.texture_map.get(&id).copied();
        }
        
        let slot = self.free_slots.pop()?;
        self.texture_map.insert(id, slot);
        Some(slot)
    }
    
    pub fn free(&mut self, id: TextureId) {
        if let Some(slot) = self.texture_map.remove(&id) {
            self.free_slots.push(slot);
        }
    }
}
```

### 17.2.3 Storage Buffer Indirection

For complex material data, use a storage buffer with indices:

```wgsl
struct MaterialDescriptor {
    albedo_texture: u32,
    normal_texture: u32,
    roughness_texture: u32,
    metallic_texture: u32,
    base_color: vec4<f32>,
    parameters: vec4<f32>,  // metallic, roughness, emissive_strength, ao
}

@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 1024>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var<storage, read> materials: array<MaterialDescriptor>;

fn sample_material(material_index: u32, uv: vec2<f32>) -> PBRInput {
    let mat = materials[material_index];
    
    var pbr: PBRInput;
    pbr.albedo = textureSample(textures[mat.albedo_texture], linear_sampler, uv).rgb
               * mat.base_color.rgb;
    pbr.normal = textureSample(textures[mat.normal_texture], linear_sampler, uv).rgb;
    pbr.roughness = textureSample(textures[mat.roughness_texture], linear_sampler, uv).r
                  * mat.parameters.y;
    pbr.metallic = textureSample(textures[mat.metallic_texture], linear_sampler, uv).r
                 * mat.parameters.x;
    
    return pbr;
}
```

### 17.2.4 Hybrid Approaches

Combine techniques for optimal performance:

```rust
pub struct HybridBindlessSystem {
    // Frequently accessed textures in binding array
    primary_textures: BindlessTextureArray,
    
    // Rarely accessed textures streamed from disk
    streaming_textures: StreamingTextureCache,
    
    // Material data in storage buffer
    material_buffer: wgpu::Buffer,
    
    // Sampler cache (limited to 16 for most hardware)
    sampler_cache: SamplerCache,
}

impl HybridBindlessSystem {
    pub fn resolve_material(&self, material_id: MaterialId) -> MaterialGPU {
        let material = self.materials.get(&material_id).unwrap();
        
        MaterialGPU {
            albedo_index: self.resolve_texture(material.albedo),
            normal_index: self.resolve_texture(material.normal),
            // ... etc
        }
    }
    
    fn resolve_texture(&self, texture_ref: TextureRef) -> u32 {
        match texture_ref {
            TextureRef::Primary(id) => {
                self.primary_textures.get_index(id)
            }
            TextureRef::Streaming(id) => {
                self.streaming_textures.get_or_load(id)
            }
        }
    }
}
```

---

## 17.3 TRINITY's Bindless System

### 17.3.1 Texture Registry

```rust
pub struct TextureRegistry {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    
    // Main texture array (up to 4096 textures)
    texture_array: wgpu::Texture,
    texture_views: Vec<wgpu::TextureView>,
    
    // Allocation tracking
    free_slots: Vec<u32>,
    slot_to_texture: HashMap<u32, TextureHandle>,
    texture_to_slot: HashMap<TextureHandle, u32>,
    
    // Bind group (recreated when textures change)
    bind_group: Option<wgpu::BindGroup>,
    bind_group_dirty: bool,
    
    max_textures: u32,
    default_texture_index: u32,
}

impl TextureRegistry {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let max_textures = 4096;
        
        // Create individual textures for binding_array
        let texture_views = Vec::new();
        
        // Create default white texture
        let default_texture = Self::create_default_texture(&device, &queue);
        
        Self {
            device,
            queue,
            texture_array: default_texture,
            texture_views,
            free_slots: (1..max_textures).rev().collect(),  // 0 is default
            slot_to_texture: HashMap::new(),
            texture_to_slot: HashMap::new(),
            bind_group: None,
            bind_group_dirty: true,
            max_textures,
            default_texture_index: 0,
        }
    }
    
    pub fn register(&mut self, handle: TextureHandle, texture: &wgpu::Texture) -> u32 {
        if let Some(&index) = self.texture_to_slot.get(&handle) {
            return index;
        }
        
        let slot = self.free_slots.pop().expect("Texture registry full");
        
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        if slot as usize >= self.texture_views.len() {
            self.texture_views.resize_with(slot as usize + 1, || {
                self.texture_array.create_view(&wgpu::TextureViewDescriptor::default())
            });
        }
        self.texture_views[slot as usize] = view;
        
        self.slot_to_texture.insert(slot, handle);
        self.texture_to_slot.insert(handle, slot);
        self.bind_group_dirty = true;
        
        slot
    }
    
    pub fn unregister(&mut self, handle: TextureHandle) {
        if let Some(slot) = self.texture_to_slot.remove(&handle) {
            self.slot_to_texture.remove(&slot);
            self.free_slots.push(slot);
            self.bind_group_dirty = true;
        }
    }
    
    pub fn get_bind_group(&mut self, layout: &wgpu::BindGroupLayout) -> &wgpu::BindGroup {
        if self.bind_group_dirty || self.bind_group.is_none() {
            self.rebuild_bind_group(layout);
            self.bind_group_dirty = false;
        }
        self.bind_group.as_ref().unwrap()
    }
    
    fn rebuild_bind_group(&mut self, layout: &wgpu::BindGroupLayout) {
        let view_refs: Vec<_> = self.texture_views.iter().collect();
        
        self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TextureRegistry"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&view_refs),
                },
            ],
        }));
    }
}
```

### 17.3.2 Buffer Registry

```rust
pub struct BufferRegistry {
    device: Arc<wgpu::Device>,
    
    // Large storage buffer for all materials
    material_buffer: wgpu::Buffer,
    material_data: Vec<MaterialGPU>,
    
    // Instance data buffer
    instance_buffer: wgpu::Buffer,
    instance_data: Vec<InstanceGPU>,
    
    // Transform buffer
    transform_buffer: wgpu::Buffer,
    transform_data: Vec<Mat4>,
    
    dirty_ranges: Vec<Range<u64>>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialGPU {
    pub albedo_texture: u32,
    pub normal_texture: u32,
    pub metallic_roughness_texture: u32,
    pub emissive_texture: u32,
    pub base_color: [f32; 4],
    pub emissive_factor: [f32; 3],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub alpha_cutoff: f32,
    pub flags: u32,
    pub _padding: u32,
}

impl BufferRegistry {
    pub fn update_material(&mut self, index: usize, material: MaterialGPU) {
        if index >= self.material_data.len() {
            self.material_data.resize(index + 1, MaterialGPU::default());
        }
        self.material_data[index] = material;
        
        let offset = (index * std::mem::size_of::<MaterialGPU>()) as u64;
        let size = std::mem::size_of::<MaterialGPU>() as u64;
        self.mark_dirty(offset, offset + size);
    }
    
    fn mark_dirty(&mut self, start: u64, end: u64) {
        // Merge overlapping ranges
        self.dirty_ranges.push(start..end);
    }
    
    pub fn flush(&mut self, queue: &wgpu::Queue) {
        if self.dirty_ranges.is_empty() {
            return;
        }
        
        // Coalesce and upload dirty ranges
        self.dirty_ranges.sort_by_key(|r| r.start);
        
        let mut merged = Vec::new();
        for range in self.dirty_ranges.drain(..) {
            if let Some(last) = merged.last_mut() {
                if last.end >= range.start {
                    last.end = last.end.max(range.end);
                    continue;
                }
            }
            merged.push(range);
        }
        
        for range in merged {
            let data = bytemuck::cast_slice(&self.material_data);
            queue.write_buffer(&self.material_buffer, range.start, 
                &data[range.start as usize..range.end as usize]);
        }
    }
}
```

### 17.3.3 Material Table

```rust
pub struct MaterialTable {
    texture_registry: TextureRegistry,
    buffer_registry: BufferRegistry,
    
    materials: HashMap<MaterialId, MaterialEntry>,
    next_gpu_index: u32,
}

pub struct MaterialEntry {
    pub gpu_index: u32,
    pub material: MaterialData,
}

impl MaterialTable {
    pub fn register_material(&mut self, id: MaterialId, material: MaterialData) -> u32 {
        if let Some(entry) = self.materials.get(&id) {
            return entry.gpu_index;
        }
        
        let gpu_index = self.next_gpu_index;
        self.next_gpu_index += 1;
        
        // Register textures
        let albedo_idx = material.albedo_texture
            .map(|t| self.texture_registry.register(t, &self.get_texture(t)))
            .unwrap_or(0);  // Default white
        
        let normal_idx = material.normal_texture
            .map(|t| self.texture_registry.register(t, &self.get_texture(t)))
            .unwrap_or(1);  // Default normal
        
        let metallic_roughness_idx = material.metallic_roughness_texture
            .map(|t| self.texture_registry.register(t, &self.get_texture(t)))
            .unwrap_or(2);  // Default MR
        
        // Create GPU material
        let gpu_material = MaterialGPU {
            albedo_texture: albedo_idx,
            normal_texture: normal_idx,
            metallic_roughness_texture: metallic_roughness_idx,
            emissive_texture: 0,
            base_color: material.base_color.into(),
            emissive_factor: material.emissive.into(),
            metallic_factor: material.metallic,
            roughness_factor: material.roughness,
            alpha_cutoff: material.alpha_cutoff,
            flags: material.flags.bits(),
            _padding: 0,
        };
        
        self.buffer_registry.update_material(gpu_index as usize, gpu_material);
        
        self.materials.insert(id, MaterialEntry {
            gpu_index,
            material,
        });
        
        gpu_index
    }
}
```

### 17.3.4 Index Allocation and Recycling

```rust
pub struct IndexAllocator {
    free_list: Vec<u32>,
    next_index: u32,
    max_index: u32,
}

impl IndexAllocator {
    pub fn new(max: u32) -> Self {
        Self {
            free_list: Vec::new(),
            next_index: 0,
            max_index: max,
        }
    }
    
    pub fn allocate(&mut self) -> Option<u32> {
        if let Some(recycled) = self.free_list.pop() {
            return Some(recycled);
        }
        
        if self.next_index < self.max_index {
            let index = self.next_index;
            self.next_index += 1;
            Some(index)
        } else {
            None
        }
    }
    
    pub fn free(&mut self, index: u32) {
        self.free_list.push(index);
    }
    
    pub fn allocated_count(&self) -> u32 {
        self.next_index - self.free_list.len() as u32
    }
}
```

---

# TRINITY Advanced Rendering Summary

| Feature | Status | TRINITY API |
|---------|--------|-------------|
| Indirect Draw | Stable | `IndirectDrawBuffer` |
| Multi-Draw Indirect | Feature-gated | `MultiDrawBatch` |
| GPU Frustum Culling | Implemented | `GPUCullingPipeline` |
| HiZ Occlusion | Implemented | `HiZPyramid` |
| GPU LOD Selection | Implemented | Integrated in cull pipeline |
| Mesh Shaders | Future | `MeshletPreprocessor` ready |
| Bindless Textures | Feature-gated | `TextureRegistry` |
| Bindless Buffers | Stable | `BufferRegistry` |
| Material Table | Implemented | `MaterialTable` |

---

*End of WGPU_PART_VIII_ADVANCED.md*

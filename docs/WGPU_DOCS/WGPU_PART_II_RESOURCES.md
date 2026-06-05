# WGPU_PART_II_RESOURCES.md — Resource Model

> **TOC Reference**: Part II, Chapters 2-4
> **Purpose**: Complete specification of wgpu's resource types for TRINITY
> **Generated**: 2026-05-27

---

# Chapter 2: Buffers

## 2.1 Buffer Fundamentals

### 2.1.1 Buffer Creation and Descriptor

```rust
let buffer = device.create_buffer(&BufferDescriptor {
    label: Some("My Buffer"),
    size: 1024,
    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
    mapped_at_creation: false,
});
```

**BufferDescriptor fields:**

| Field | Type | Description |
|-------|------|-------------|
| `label` | `Option<&str>` | Debug name (visible in debuggers) |
| `size` | `u64` | Buffer size in bytes |
| `usage` | `BufferUsages` | How buffer will be used |
| `mapped_at_creation` | `bool` | Create with CPU-accessible mapping |

### 2.1.2 Buffer Usage Flags

```rust
pub struct BufferUsages: u32 {
    const MAP_READ = 1 << 0;      // Can be mapped for CPU read
    const MAP_WRITE = 1 << 1;     // Can be mapped for CPU write
    const COPY_SRC = 1 << 2;      // Source for copy operations
    const COPY_DST = 1 << 3;      // Destination for copy operations
    const INDEX = 1 << 4;         // Index buffer
    const VERTEX = 1 << 5;        // Vertex buffer
    const UNIFORM = 1 << 6;       // Uniform buffer (constant data)
    const STORAGE = 1 << 7;       // Storage buffer (read/write from shaders)
    const INDIRECT = 1 << 8;      // Indirect draw/dispatch arguments
    const QUERY_RESOLVE = 1 << 9; // Destination for query results
}
```

**Usage combinations:**

| Use Case | Usage Flags |
|----------|-------------|
| Static vertex data | `VERTEX | COPY_DST` |
| Dynamic uniform | `UNIFORM | COPY_DST` |
| Staging upload | `MAP_WRITE | COPY_SRC` |
| Staging readback | `MAP_READ | COPY_DST` |
| GPU-writable storage | `STORAGE | COPY_SRC` |
| Indirect draw | `STORAGE | INDIRECT | COPY_DST` |
| Query results | `QUERY_RESOLVE | COPY_SRC | MAP_READ` |

### 2.1.3 Buffer Mapping

**Synchronous mapping (at creation):**
```rust
let buffer = device.create_buffer(&BufferDescriptor {
    label: Some("Mapped Buffer"),
    size: data.len() as u64,
    usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
    mapped_at_creation: true,
});

// Write data while mapped
{
    let mut view = buffer.slice(..).get_mapped_range_mut();
    view.copy_from_slice(data);
}
buffer.unmap(); // Must unmap before GPU use
```

**Asynchronous mapping:**
```rust
let buffer_slice = buffer.slice(..);

// Request mapping
buffer_slice.map_async(MapMode::Read, |result| {
    match result {
        Ok(()) => println!("Mapping succeeded"),
        Err(e) => println!("Mapping failed: {:?}", e),
    }
});

// Poll until ready
device.poll(Maintain::Wait);

// Access data
{
    let data = buffer_slice.get_mapped_range();
    // Read data...
}
buffer.unmap();
```

**TRINITY async mapping wrapper:**
```rust
pub async fn map_buffer_read(device: &Device, buffer: &Buffer) -> Result<BufferView, MapError> {
    let slice = buffer.slice(..);
    
    let (sender, receiver) = futures::channel::oneshot::channel();
    slice.map_async(MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    
    // Poll device until mapping complete
    loop {
        device.poll(Maintain::Poll);
        if let Ok(result) = receiver.try_recv() {
            result?;
            break;
        }
        tokio::task::yield_now().await;
    }
    
    Ok(slice.get_mapped_range())
}
```

### 2.1.4 Mapped Ranges and Mapping Modes

```rust
pub enum MapMode {
    Read,  // MAP_READ usage required
    Write, // MAP_WRITE usage required
}

// Full buffer
let view = buffer.slice(..).get_mapped_range();

// Partial range
let view = buffer.slice(64..128).get_mapped_range();

// Mutable write
let mut view = buffer.slice(..).get_mapped_range_mut();
view.copy_from_slice(&data);
```

### 2.1.5 Buffer Destruction and Resource Cleanup

wgpu uses reference counting — buffers are destroyed when all references are dropped:

```rust
// Buffer is reference counted
let buffer = device.create_buffer(&desc);

// Drop releases reference
drop(buffer); // Buffer destroyed if no other references

// Explicit destruction (optional, marks as invalid)
buffer.destroy();
```

**TRINITY deferred destruction:**
```rust
pub struct DeferredDestroyer {
    pending: Vec<(u64, ResourceHandle)>,
    current_frame: u64,
    frames_to_keep: u64,
}

impl DeferredDestroyer {
    pub fn destroy_buffer(&mut self, buffer: Buffer) {
        let destroy_frame = self.current_frame + self.frames_to_keep;
        self.pending.push((destroy_frame, ResourceHandle::Buffer(buffer)));
    }
    
    pub fn process_frame(&mut self, frame: u64) {
        self.current_frame = frame;
        self.pending.retain(|(destroy_frame, handle)| {
            if *destroy_frame <= frame {
                match handle {
                    ResourceHandle::Buffer(b) => b.destroy(),
                    ResourceHandle::Texture(t) => t.destroy(),
                    // ...
                }
                false
            } else {
                true
            }
        });
    }
}
```

---

## 2.2 Buffer Types by Role

### 2.2.1 Vertex Buffers and Vertex Layouts

```rust
// Define vertex layout
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    tangent: [f32; 4],
}

impl Vertex {
    const ATTRIBS: [VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,  // position
        1 => Float32x3,  // normal
        2 => Float32x2,  // uv
        3 => Float32x4,  // tangent
    ];
    
    fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

// Create vertex buffer
let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
    label: Some("Vertex Buffer"),
    contents: bytemuck::cast_slice(&vertices),
    usage: BufferUsages::VERTEX,
});
```

**TRINITY vertex format registry:**
```rust
pub struct VertexFormatRegistry {
    formats: HashMap<VertexFormatId, VertexFormat>,
}

pub struct VertexFormat {
    pub attributes: Vec<VertexAttributeDesc>,
    pub stride: u32,
    pub step_mode: VertexStepMode,
}

pub struct VertexAttributeDesc {
    pub semantic: VertexSemantic,
    pub format: VertexAttributeFormat,
    pub offset: u32,
}

pub enum VertexSemantic {
    Position,
    Normal,
    Tangent,
    TexCoord0,
    TexCoord1,
    Color0,
    Joints0,
    Weights0,
    Custom(u32),
}

impl VertexFormatRegistry {
    pub fn standard_pbr() -> VertexFormat {
        VertexFormat {
            attributes: vec![
                VertexAttributeDesc { semantic: VertexSemantic::Position, format: Float32x3, offset: 0 },
                VertexAttributeDesc { semantic: VertexSemantic::Normal, format: Float32x3, offset: 12 },
                VertexAttributeDesc { semantic: VertexSemantic::TexCoord0, format: Float32x2, offset: 24 },
                VertexAttributeDesc { semantic: VertexSemantic::Tangent, format: Float32x4, offset: 32 },
            ],
            stride: 48,
            step_mode: VertexStepMode::Vertex,
        }
    }
    
    pub fn skinned() -> VertexFormat {
        VertexFormat {
            attributes: vec![
                VertexAttributeDesc { semantic: VertexSemantic::Position, format: Float32x3, offset: 0 },
                VertexAttributeDesc { semantic: VertexSemantic::Normal, format: Float32x3, offset: 12 },
                VertexAttributeDesc { semantic: VertexSemantic::TexCoord0, format: Float32x2, offset: 24 },
                VertexAttributeDesc { semantic: VertexSemantic::Tangent, format: Float32x4, offset: 32 },
                VertexAttributeDesc { semantic: VertexSemantic::Joints0, format: Uint16x4, offset: 48 },
                VertexAttributeDesc { semantic: VertexSemantic::Weights0, format: Float32x4, offset: 56 },
            ],
            stride: 72,
            step_mode: VertexStepMode::Vertex,
        }
    }
}
```

### 2.2.2 Index Buffers

```rust
// 16-bit indices (up to 65535 vertices)
let index_buffer_u16 = device.create_buffer_init(&BufferInitDescriptor {
    label: Some("Index Buffer U16"),
    contents: bytemuck::cast_slice(&indices_u16),
    usage: BufferUsages::INDEX,
});

// 32-bit indices (up to 4B vertices)
let index_buffer_u32 = device.create_buffer_init(&BufferInitDescriptor {
    label: Some("Index Buffer U32"),
    contents: bytemuck::cast_slice(&indices_u32),
    usage: BufferUsages::INDEX,
});

// Bind in render pass
render_pass.set_index_buffer(index_buffer.slice(..), IndexFormat::Uint16);
// or
render_pass.set_index_buffer(index_buffer.slice(..), IndexFormat::Uint32);
```

**Index format selection:**
```rust
pub fn select_index_format(vertex_count: u32) -> IndexFormat {
    if vertex_count <= 65535 {
        IndexFormat::Uint16  // 2 bytes per index
    } else {
        IndexFormat::Uint32  // 4 bytes per index
    }
}
```

### 2.2.3 Uniform Buffers and Dynamic Offsets

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    view_inv: [[f32; 4]; 4],
    proj_inv: [[f32; 4]; 4],
    position: [f32; 4],
    near_far: [f32; 4],
}

let camera_buffer = device.create_buffer(&BufferDescriptor {
    label: Some("Camera Uniform"),
    size: std::mem::size_of::<CameraUniform>() as u64,
    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    mapped_at_creation: false,
});

// Update via queue
queue.write_buffer(&camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));
```

**Dynamic uniform buffers:**
```rust
// Single buffer for many objects
const MAX_OBJECTS: u64 = 1024;
const UNIFORM_ALIGNMENT: u64 = 256; // Check device limits

let dynamic_buffer = device.create_buffer(&BufferDescriptor {
    label: Some("Dynamic Uniforms"),
    size: MAX_OBJECTS * UNIFORM_ALIGNMENT,
    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    mapped_at_creation: false,
});

// Bind group layout with dynamic offset
let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
    label: Some("Dynamic Layout"),
    entries: &[
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true,  // Key!
                min_binding_size: Some(NonZeroU64::new(std::mem::size_of::<ObjectUniform>() as u64).unwrap()),
            },
            count: None,
        },
    ],
});

// In render loop, set different offsets
for (i, object) in objects.iter().enumerate() {
    let offset = (i as u32) * (UNIFORM_ALIGNMENT as u32);
    render_pass.set_bind_group(1, &bind_group, &[offset]);
    render_pass.draw_indexed(0..object.index_count, 0, 0..1);
}
```

### 2.2.4 Storage Buffers

```rust
// Read-only storage buffer
let readonly_storage = device.create_buffer(&BufferDescriptor {
    label: Some("Read-Only Storage"),
    size: data_size,
    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    mapped_at_creation: false,
});

// Read-write storage buffer (for compute)
let readwrite_storage = device.create_buffer(&BufferDescriptor {
    label: Some("Read-Write Storage"),
    size: data_size,
    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
});
```

**Bind group layout:**
```rust
BindGroupLayoutEntry {
    binding: 0,
    visibility: ShaderStages::COMPUTE,
    ty: BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
    },
    count: None,
}
```

### 2.2.5 Indirect Buffers

```rust
// Draw indirect structure
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
}

// Indexed draw indirect structure
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

// Dispatch indirect structure
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
}

let indirect_buffer = device.create_buffer(&BufferDescriptor {
    label: Some("Indirect Buffer"),
    size: MAX_DRAWS * std::mem::size_of::<DrawIndexedIndirectArgs>() as u64,
    usage: BufferUsages::INDIRECT | BufferUsages::STORAGE | BufferUsages::COPY_DST,
    mapped_at_creation: false,
});
```

### 2.2.6 Query Resolve Buffers

```rust
let query_buffer = device.create_buffer(&BufferDescriptor {
    label: Some("Query Results"),
    size: query_count * 8, // 8 bytes per timestamp
    usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
});

// Resolve queries into buffer
encoder.resolve_query_set(&query_set, 0..query_count, &query_buffer, 0);

// Read back
let staging = device.create_buffer(&BufferDescriptor {
    label: Some("Query Staging"),
    size: query_count * 8,
    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
    mapped_at_creation: false,
});

encoder.copy_buffer_to_buffer(&query_buffer, 0, &staging, 0, query_count * 8);
```

### 2.2.7 Staging Buffers

```rust
pub struct StagingBelt {
    chunk_size: u64,
    active_chunks: Vec<StagingChunk>,
    free_chunks: Vec<StagingChunk>,
}

struct StagingChunk {
    buffer: Buffer,
    offset: u64,
    size: u64,
}

impl StagingBelt {
    pub fn new(chunk_size: u64) -> Self {
        Self {
            chunk_size,
            active_chunks: Vec::new(),
            free_chunks: Vec::new(),
        }
    }
    
    pub fn write_buffer(
        &mut self,
        device: &Device,
        encoder: &mut CommandEncoder,
        target: &Buffer,
        offset: u64,
        size: u64,
    ) -> BufferViewMut {
        let chunk = self.get_or_create_chunk(device, size);
        
        encoder.copy_buffer_to_buffer(
            &chunk.buffer,
            chunk.offset,
            target,
            offset,
            size,
        );
        
        chunk.buffer.slice(chunk.offset..chunk.offset + size).get_mapped_range_mut()
    }
    
    pub fn finish(&mut self) {
        for chunk in self.active_chunks.drain(..) {
            chunk.buffer.unmap();
            self.free_chunks.push(chunk);
        }
    }
    
    pub fn recall(&mut self) {
        // Reclaim chunks after GPU is done
        for chunk in &self.free_chunks {
            chunk.buffer.slice(..).map_async(MapMode::Write, |_| {});
        }
    }
}
```

---

## 2.3 Buffer Memory Management

### 2.3.1 Memory Allocation Strategies

**TRINITY buffer allocator:**
```rust
pub struct BufferAllocator {
    device: Arc<Device>,
    
    // Pool by usage pattern
    vertex_pool: BufferPool,
    index_pool: BufferPool,
    uniform_pool: BufferPool,
    storage_pool: BufferPool,
    staging_pool: BufferPool,
    
    // Stats
    total_allocated: AtomicU64,
    total_in_use: AtomicU64,
}

pub struct BufferPool {
    usage: BufferUsages,
    chunks: Vec<BufferChunk>,
    free_list: FreeList,
    chunk_size: u64,
}

struct BufferChunk {
    buffer: Buffer,
    size: u64,
    allocations: Vec<Allocation>,
}

impl BufferAllocator {
    pub fn allocate(&mut self, desc: &BufferAllocationDesc) -> BufferAllocation {
        let pool = match desc.usage_hint {
            UsageHint::Vertex => &mut self.vertex_pool,
            UsageHint::Index => &mut self.index_pool,
            UsageHint::Uniform => &mut self.uniform_pool,
            UsageHint::Storage => &mut self.storage_pool,
            UsageHint::Staging => &mut self.staging_pool,
        };
        
        pool.allocate(desc.size, desc.alignment)
    }
    
    pub fn free(&mut self, allocation: BufferAllocation) {
        // Return to pool
        let pool = self.get_pool_mut(allocation.usage_hint);
        pool.free(allocation);
    }
}
```

### 2.3.2 Buffer Suballocation and Pooling

```rust
pub struct SuballocatedBuffer {
    buffer: Arc<Buffer>,
    offset: u64,
    size: u64,
    pool_id: PoolId,
}

impl SuballocatedBuffer {
    pub fn slice(&self) -> BufferSlice {
        self.buffer.slice(self.offset..self.offset + self.size)
    }
    
    pub fn binding(&self) -> BufferBinding {
        BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: Some(NonZeroU64::new(self.size).unwrap()),
        }
    }
}

pub struct FreeList {
    entries: BTreeMap<u64, Vec<FreeEntry>>,  // size -> entries
}

struct FreeEntry {
    offset: u64,
    size: u64,
}

impl FreeList {
    pub fn allocate(&mut self, size: u64, alignment: u64) -> Option<(u64, u64)> {
        // Find best-fit block
        for (&block_size, entries) in self.entries.range_mut(size..) {
            if let Some(entry) = entries.pop() {
                let aligned_offset = (entry.offset + alignment - 1) & !(alignment - 1);
                let aligned_size = entry.size - (aligned_offset - entry.offset);
                
                if aligned_size >= size {
                    // Return excess to free list
                    let excess_offset = aligned_offset + size;
                    let excess_size = aligned_size - size;
                    if excess_size > 0 {
                        self.free(excess_offset, excess_size);
                    }
                    
                    return Some((aligned_offset, size));
                }
            }
        }
        None
    }
    
    pub fn free(&mut self, offset: u64, size: u64) {
        self.entries.entry(size).or_default().push(FreeEntry { offset, size });
        self.coalesce();
    }
}
```

### 2.3.3 Ring Buffers for Per-Frame Data

```rust
pub struct RingBuffer {
    buffer: Buffer,
    capacity: u64,
    write_offset: u64,
    frame_offsets: VecDeque<(u64, u64)>,  // (frame_id, start_offset)
    frames_in_flight: usize,
}

impl RingBuffer {
    pub fn new(device: &Device, capacity: u64, usage: BufferUsages, frames: usize) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Ring Buffer"),
            size: capacity,
            usage: usage | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        Self {
            buffer,
            capacity,
            write_offset: 0,
            frame_offsets: VecDeque::with_capacity(frames),
            frames_in_flight: frames,
        }
    }
    
    pub fn begin_frame(&mut self, frame_id: u64) {
        self.frame_offsets.push_back((frame_id, self.write_offset));
        
        // Reclaim space from old frames
        while self.frame_offsets.len() > self.frames_in_flight {
            self.frame_offsets.pop_front();
        }
    }
    
    pub fn allocate(&mut self, size: u64, alignment: u64) -> Option<RingAllocation> {
        let aligned_offset = (self.write_offset + alignment - 1) & !(alignment - 1);
        let end_offset = aligned_offset + size;
        
        // Check wrap-around
        if end_offset > self.capacity {
            // Wrap to beginning
            let aligned_offset = 0;
            let end_offset = size;
            
            // Check if we'd overwrite in-flight data
            if let Some(&(_, oldest_offset)) = self.frame_offsets.front() {
                if end_offset > oldest_offset && oldest_offset > 0 {
                    return None; // Would overwrite in-flight data
                }
            }
            
            self.write_offset = end_offset;
            return Some(RingAllocation {
                buffer: &self.buffer,
                offset: aligned_offset,
                size,
            });
        }
        
        // Check if we'd overwrite in-flight data
        if let Some(&(_, oldest_offset)) = self.frame_offsets.front() {
            if self.write_offset < oldest_offset && end_offset > oldest_offset {
                return None;
            }
        }
        
        self.write_offset = end_offset;
        Some(RingAllocation {
            buffer: &self.buffer,
            offset: aligned_offset,
            size,
        })
    }
}
```

### 2.3.4 Persistent Mapping Patterns

```rust
pub struct PersistentMappedBuffer {
    buffer: Buffer,
    mapped_range: *mut u8,
    size: u64,
    write_offset: AtomicU64,
}

unsafe impl Send for PersistentMappedBuffer {}
unsafe impl Sync for PersistentMappedBuffer {}

impl PersistentMappedBuffer {
    pub fn new(device: &Device, size: u64) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Persistent Mapped"),
            size,
            usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        
        let mapped_range = buffer.slice(..).get_mapped_range_mut().as_mut_ptr();
        
        Self {
            buffer,
            mapped_range,
            size,
            write_offset: AtomicU64::new(0),
        }
    }
    
    pub fn write(&self, data: &[u8]) -> Option<u64> {
        let size = data.len() as u64;
        let offset = self.write_offset.fetch_add(size, Ordering::SeqCst);
        
        if offset + size > self.size {
            return None;
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.mapped_range.add(offset as usize),
                data.len(),
            );
        }
        
        Some(offset)
    }
    
    pub fn reset(&self) {
        self.write_offset.store(0, Ordering::SeqCst);
    }
}
```

### 2.3.5 TRINITY's Buffer Allocator Architecture

```rust
pub struct TrinityBufferSystem {
    // Per-frame allocators (triple-buffered)
    frame_allocators: [FrameAllocator; 3],
    current_frame: usize,
    
    // Persistent allocators
    static_allocator: BufferAllocator,
    
    // Staging belt for uploads
    staging_belt: StagingBelt,
    
    // Global stats
    stats: BufferStats,
}

impl TrinityBufferSystem {
    pub fn begin_frame(&mut self, frame_index: usize) {
        self.current_frame = frame_index % 3;
        self.frame_allocators[self.current_frame].reset();
    }
    
    pub fn allocate_frame_uniform<T: bytemuck::Pod>(&mut self, data: &T) -> BufferBinding {
        let bytes = bytemuck::bytes_of(data);
        let alloc = self.frame_allocators[self.current_frame].allocate(
            bytes.len() as u64,
            256, // Uniform alignment
        );
        alloc.write(bytes);
        alloc.binding()
    }
    
    pub fn allocate_static(&mut self, desc: &BufferDesc) -> BufferHandle {
        self.static_allocator.allocate(desc)
    }
    
    pub fn upload_static(&mut self, encoder: &mut CommandEncoder, handle: BufferHandle, data: &[u8]) {
        let view = self.staging_belt.write_buffer(
            &self.device,
            encoder,
            handle.buffer(),
            handle.offset(),
            data.len() as u64,
        );
        view.copy_from_slice(data);
    }
}
```

---

# Chapter 3: Textures

## 3.1 Texture Fundamentals

### 3.1.1 Texture Creation and Descriptor

```rust
let texture = device.create_texture(&TextureDescriptor {
    label: Some("My Texture"),
    size: Extent3d {
        width: 1024,
        height: 1024,
        depth_or_array_layers: 1,
    },
    mip_level_count: 10,  // log2(1024) + 1
    sample_count: 1,
    dimension: TextureDimension::D2,
    format: TextureFormat::Rgba8UnormSrgb,
    usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
    view_formats: &[TextureFormat::Rgba8Unorm],  // Allow view format reinterpretation
});
```

### 3.1.2 Texture Dimensions

```rust
pub enum TextureDimension {
    D1,  // 1D texture (width only)
    D2,  // 2D texture (width x height), also cube maps and 2D arrays
    D3,  // 3D texture (width x height x depth)
}
```

**Extent3d interpretation by dimension:**

| Dimension | width | height | depth_or_array_layers |
|-----------|-------|--------|----------------------|
| D1 | Width | 1 | Array layers |
| D2 | Width | Height | Array layers (or 6 for cube) |
| D3 | Width | Height | Depth |

### 3.1.3 Texture Formats

**Color formats (partial list):**

| Format | Bytes | Description |
|--------|-------|-------------|
| `R8Unorm` | 1 | Single channel, normalized [0, 1] |
| `R8Snorm` | 1 | Single channel, normalized [-1, 1] |
| `R8Uint` | 1 | Single channel, unsigned integer |
| `R8Sint` | 1 | Single channel, signed integer |
| `R16Uint` | 2 | 16-bit unsigned integer |
| `R16Sint` | 2 | 16-bit signed integer |
| `R16Float` | 2 | 16-bit float |
| `Rg8Unorm` | 2 | Two channels |
| `Rgba8Unorm` | 4 | Four channels, linear |
| `Rgba8UnormSrgb` | 4 | Four channels, sRGB |
| `Bgra8Unorm` | 4 | BGRA order (common for swapchain) |
| `Rgb10a2Unorm` | 4 | 10-bit RGB, 2-bit alpha |
| `Rg11b10Float` | 4 | HDR color (no alpha) |
| `Rgba16Float` | 8 | Half-precision HDR |
| `Rgba32Float` | 16 | Full-precision HDR |

**Depth/stencil formats:**

| Format | Bytes | Description |
|--------|-------|-------------|
| `Depth16Unorm` | 2 | 16-bit depth |
| `Depth24Plus` | 4 | 24+ bit depth |
| `Depth24PlusStencil8` | 4 | Depth + stencil |
| `Depth32Float` | 4 | 32-bit float depth |
| `Depth32FloatStencil8` | 8 | Float depth + stencil |

**Compressed formats:**

| Format | Block | Ratio | Description |
|--------|-------|-------|-------------|
| `Bc1RgbaUnorm` | 4x4 | 8:1 | DXT1, RGB + 1-bit alpha |
| `Bc2RgbaUnorm` | 4x4 | 4:1 | DXT3, explicit alpha |
| `Bc3RgbaUnorm` | 4x4 | 4:1 | DXT5, interpolated alpha |
| `Bc4RUnorm` | 4x4 | 2:1 | Single channel |
| `Bc5RgUnorm` | 4x4 | 2:1 | Two channels (normal maps) |
| `Bc6hRgbUfloat` | 4x4 | 6:1 | HDR RGB |
| `Bc7RgbaUnorm` | 4x4 | 4:1 | High quality RGBA |
| `Etc2Rgb8Unorm` | 4x4 | 6:1 | Mobile RGB |
| `Etc2Rgba8Unorm` | 4x4 | 4:1 | Mobile RGBA |
| `Astc4x4RgbaUnorm` | 4x4 | 4:1 | Adaptive, high quality |
| `Astc8x8RgbaUnorm` | 8x8 | 16:1 | Adaptive, high compression |

### 3.1.4 Texture Usage Flags

```rust
pub struct TextureUsages: u32 {
    const COPY_SRC = 1 << 0;           // Source for copy operations
    const COPY_DST = 1 << 1;           // Destination for copy/write
    const TEXTURE_BINDING = 1 << 2;    // Sampled texture in shaders
    const STORAGE_BINDING = 1 << 3;    // Storage texture (read/write)
    const RENDER_ATTACHMENT = 1 << 4;  // Color/depth attachment
}
```

**Common combinations:**

| Use Case | Usage Flags |
|----------|-------------|
| Sampled texture | `TEXTURE_BINDING | COPY_DST` |
| Render target | `RENDER_ATTACHMENT | TEXTURE_BINDING` |
| Shadow map | `RENDER_ATTACHMENT | TEXTURE_BINDING` |
| Compute output | `STORAGE_BINDING | TEXTURE_BINDING` |
| Screenshot | `RENDER_ATTACHMENT | COPY_SRC` |

### 3.1.5 Mip Levels and Mip Generation

```rust
fn calculate_mip_count(width: u32, height: u32) -> u32 {
    (width.max(height) as f32).log2().floor() as u32 + 1
}

// Mip dimensions
fn mip_size(base: u32, level: u32) -> u32 {
    (base >> level).max(1)
}
```

**Mip generation via compute shader:**
```rust
pub struct MipGenerator {
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    sampler: Sampler,
}

impl MipGenerator {
    pub fn generate(
        &self,
        encoder: &mut CommandEncoder,
        texture: &Texture,
        mip_count: u32,
    ) {
        for level in 1..mip_count {
            let src_view = texture.create_view(&TextureViewDescriptor {
                base_mip_level: level - 1,
                mip_level_count: Some(1),
                ..Default::default()
            });
            
            let dst_view = texture.create_view(&TextureViewDescriptor {
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            });
            
            let bind_group = self.create_bind_group(&src_view, &dst_view);
            
            let width = mip_size(texture.width(), level);
            let height = mip_size(texture.height(), level);
            
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                (width + 7) / 8,
                (height + 7) / 8,
                1,
            );
        }
    }
}
```

### 3.1.6 Array Layers and Cube Maps

```rust
// 2D array texture
let array_texture = device.create_texture(&TextureDescriptor {
    size: Extent3d {
        width: 512,
        height: 512,
        depth_or_array_layers: 16,  // 16 layers
    },
    dimension: TextureDimension::D2,
    // ...
});

// Cube map (6 layers)
let cube_texture = device.create_texture(&TextureDescriptor {
    size: Extent3d {
        width: 512,
        height: 512,
        depth_or_array_layers: 6,  // 6 faces
    },
    dimension: TextureDimension::D2,
    // ...
});

// Cube map view
let cube_view = cube_texture.create_view(&TextureViewDescriptor {
    dimension: Some(TextureViewDimension::Cube),
    ..Default::default()
});

// Cube map array
let cube_array = device.create_texture(&TextureDescriptor {
    size: Extent3d {
        width: 256,
        height: 256,
        depth_or_array_layers: 6 * 4,  // 4 cube maps
    },
    // ...
});
```

### 3.1.7 Multisampled Textures

```rust
let msaa_texture = device.create_texture(&TextureDescriptor {
    label: Some("MSAA Target"),
    size: Extent3d { width, height, depth_or_array_layers: 1 },
    mip_level_count: 1,  // MSAA textures can't have mips
    sample_count: 4,     // 1, 2, 4, 8, 16 (device-dependent)
    dimension: TextureDimension::D2,
    format: TextureFormat::Rgba8UnormSrgb,
    usage: TextureUsages::RENDER_ATTACHMENT,
    view_formats: &[],
});

// Resolve target (non-MSAA)
let resolve_target = device.create_texture(&TextureDescriptor {
    sample_count: 1,
    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
    // ... same size/format
});
```

---

## 3.2 Texture Formats Deep Dive

### 3.2.1-3.2.7 [Format tables above]

### 3.2.8 TRINITY's Format Selection Strategy

```rust
pub struct TextureFormatSelector;

impl TextureFormatSelector {
    pub fn color_attachment(hdr: bool, alpha: bool) -> TextureFormat {
        match (hdr, alpha) {
            (false, false) => TextureFormat::Rgb10a2Unorm,
            (false, true) => TextureFormat::Rgba8Unorm,
            (true, false) => TextureFormat::Rg11b10Float,
            (true, true) => TextureFormat::Rgba16Float,
        }
    }
    
    pub fn depth(needs_stencil: bool, high_precision: bool) -> TextureFormat {
        match (needs_stencil, high_precision) {
            (false, false) => TextureFormat::Depth24Plus,
            (false, true) => TextureFormat::Depth32Float,
            (true, false) => TextureFormat::Depth24PlusStencil8,
            (true, true) => TextureFormat::Depth32FloatStencil8,
        }
    }
    
    pub fn normal_map(compressed: bool) -> TextureFormat {
        if compressed {
            TextureFormat::Bc5RgUnorm
        } else {
            TextureFormat::Rg16Snorm
        }
    }
    
    pub fn albedo(srgb: bool, compressed: bool) -> TextureFormat {
        match (srgb, compressed) {
            (true, true) => TextureFormat::Bc7RgbaUnormSrgb,
            (true, false) => TextureFormat::Rgba8UnormSrgb,
            (false, true) => TextureFormat::Bc7RgbaUnorm,
            (false, false) => TextureFormat::Rgba8Unorm,
        }
    }
}
```

---

## 3.3 Texture Views

### 3.3.1 View Creation

```rust
let view = texture.create_view(&TextureViewDescriptor {
    label: Some("Texture View"),
    format: None,  // Use texture format
    dimension: None,  // Use texture dimension
    aspect: TextureAspect::All,
    base_mip_level: 0,
    mip_level_count: None,  // All mips
    base_array_layer: 0,
    array_layer_count: None,  // All layers
});
```

### 3.3.2-3.3.6 View configuration

```rust
// Single mip view (for mip generation)
let mip_view = texture.create_view(&TextureViewDescriptor {
    base_mip_level: 3,
    mip_level_count: Some(1),
    ..Default::default()
});

// Single array layer view
let layer_view = texture.create_view(&TextureViewDescriptor {
    base_array_layer: 5,
    array_layer_count: Some(1),
    ..Default::default()
});

// Depth-only view of depth-stencil
let depth_view = texture.create_view(&TextureViewDescriptor {
    aspect: TextureAspect::DepthOnly,
    ..Default::default()
});

// Stencil-only view
let stencil_view = texture.create_view(&TextureViewDescriptor {
    aspect: TextureAspect::StencilOnly,
    ..Default::default()
});

// Format reinterpretation
let linear_view = srgb_texture.create_view(&TextureViewDescriptor {
    format: Some(TextureFormat::Rgba8Unorm),  // Must be in view_formats
    ..Default::default()
});
```

---

## 3.4 Samplers

### 3.4.1-3.4.8 Sampler Configuration

```rust
let sampler = device.create_sampler(&SamplerDescriptor {
    label: Some("Linear Sampler"),
    address_mode_u: AddressMode::Repeat,
    address_mode_v: AddressMode::Repeat,
    address_mode_w: AddressMode::Repeat,
    mag_filter: FilterMode::Linear,
    min_filter: FilterMode::Linear,
    mipmap_filter: FilterMode::Linear,
    lod_min_clamp: 0.0,
    lod_max_clamp: 100.0,
    compare: None,
    anisotropy_clamp: 16,
    border_color: None,
});

// Point sampler (no filtering)
let point_sampler = device.create_sampler(&SamplerDescriptor {
    mag_filter: FilterMode::Nearest,
    min_filter: FilterMode::Nearest,
    mipmap_filter: FilterMode::Nearest,
    ..Default::default()
});

// Shadow map sampler
let shadow_sampler = device.create_sampler(&SamplerDescriptor {
    address_mode_u: AddressMode::ClampToEdge,
    address_mode_v: AddressMode::ClampToEdge,
    mag_filter: FilterMode::Linear,
    min_filter: FilterMode::Linear,
    compare: Some(CompareFunction::LessEqual),
    ..Default::default()
});
```

### 3.4.9 TRINITY's Sampler Cache

```rust
pub struct SamplerCache {
    samplers: HashMap<SamplerKey, Sampler>,
}

#[derive(Hash, Eq, PartialEq)]
struct SamplerKey {
    address_mode: AddressMode,
    filter: FilterMode,
    anisotropy: u8,
    compare: Option<CompareFunction>,
}

impl SamplerCache {
    pub fn get(&mut self, device: &Device, desc: &SamplerDescriptor) -> &Sampler {
        let key = SamplerKey::from(desc);
        self.samplers.entry(key).or_insert_with(|| {
            device.create_sampler(desc)
        })
    }
    
    // Predefined samplers
    pub fn linear_repeat(&mut self, device: &Device) -> &Sampler {
        self.get(device, &SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            anisotropy_clamp: 16,
            ..Default::default()
        })
    }
    
    pub fn linear_clamp(&mut self, device: &Device) -> &Sampler {
        self.get(device, &SamplerDescriptor {
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            ..Default::default()
        })
    }
    
    pub fn point(&mut self, device: &Device) -> &Sampler {
        self.get(device, &SamplerDescriptor {
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        })
    }
    
    pub fn shadow(&mut self, device: &Device) -> &Sampler {
        self.get(device, &SamplerDescriptor {
            compare: Some(CompareFunction::LessEqual),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            address_mode_u: AddressMode::ClampToBorder,
            address_mode_v: AddressMode::ClampToBorder,
            border_color: Some(SamplerBorderColor::OpaqueWhite),
            ..Default::default()
        })
    }
}
```

---

## 3.5 Texture Operations

### 3.5.1 Texture Uploads

```rust
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

### 3.5.2 Texture Copies

```rust
// Buffer to texture
encoder.copy_buffer_to_texture(
    ImageCopyBuffer {
        buffer: &staging_buffer,
        layout: ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
    },
    ImageCopyTexture {
        texture: &texture,
        mip_level: 0,
        origin: Origin3d::ZERO,
        aspect: TextureAspect::All,
    },
    Extent3d { width, height, depth_or_array_layers: 1 },
);

// Texture to buffer (readback)
encoder.copy_texture_to_buffer(
    ImageCopyTexture { /* ... */ },
    ImageCopyBuffer { /* ... */ },
    extent,
);

// Texture to texture
encoder.copy_texture_to_texture(
    ImageCopyTexture { texture: &src, /* ... */ },
    ImageCopyTexture { texture: &dst, /* ... */ },
    extent,
);
```

---

# Chapter 4: Bind Groups & Layouts

## 4.1 Binding Model

### 4.1.1-4.1.5 Bind Group Fundamentals

```rust
// Create layout
let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
    label: Some("Scene Layout"),
    entries: &[
        // Uniform buffer
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(std::mem::size_of::<CameraUniform>() as u64).unwrap()),
            },
            count: None,
        },
        // Sampled texture
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        // Sampler
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: None,
        },
        // Storage buffer
        BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ],
});

// Create bind group
let bind_group = device.create_bind_group(&BindGroupDescriptor {
    label: Some("Scene Bind Group"),
    layout: &layout,
    entries: &[
        BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 1,
            resource: BindingResource::TextureView(&texture_view),
        },
        BindGroupEntry {
            binding: 2,
            resource: BindingResource::Sampler(&sampler),
        },
        BindGroupEntry {
            binding: 3,
            resource: storage_buffer.as_entire_binding(),
        },
    ],
});
```

## 4.4 Bindless Resources

### 4.4.1-4.4.5 Bindless Patterns

```rust
// Texture array for bindless texturing
let texture_array_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
    label: Some("Bindless Textures"),
    entries: &[
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: Some(NonZeroU32::new(MAX_TEXTURES).unwrap()),  // Array!
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: Some(NonZeroU32::new(MAX_SAMPLERS).unwrap()),
        },
    ],
});
```

**TRINITY's bindless system:**
```rust
pub struct BindlessManager {
    texture_views: Vec<TextureView>,
    free_indices: Vec<u32>,
    bind_group: BindGroup,
    dirty: bool,
}

impl BindlessManager {
    pub fn allocate_texture(&mut self, view: TextureView) -> TextureIndex {
        let index = if let Some(free) = self.free_indices.pop() {
            self.texture_views[free as usize] = view;
            free
        } else {
            let index = self.texture_views.len() as u32;
            self.texture_views.push(view);
            index
        };
        self.dirty = true;
        TextureIndex(index)
    }
    
    pub fn free_texture(&mut self, index: TextureIndex) {
        self.free_indices.push(index.0);
        // Replace with placeholder
        self.texture_views[index.0 as usize] = self.placeholder_view.clone();
        self.dirty = true;
    }
    
    pub fn rebuild_bind_group(&mut self, device: &Device) {
        if !self.dirty {
            return;
        }
        
        let entries: Vec<_> = self.texture_views
            .iter()
            .enumerate()
            .map(|(i, view)| BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureViewArray(&[view]),
            })
            .collect();
        
        self.bind_group = device.create_bind_group(&BindGroupDescriptor {
            layout: &self.layout,
            entries: &entries,
            label: Some("Bindless Textures"),
        });
        
        self.dirty = false;
    }
}
```

## 4.5 Pipeline Layouts

```rust
let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
    label: Some("Main Pipeline Layout"),
    bind_group_layouts: &[
        &global_layout,    // Group 0: Camera, lights
        &material_layout,  // Group 1: Material params
        &bindless_layout,  // Group 2: Bindless textures
        &object_layout,    // Group 3: Per-object data
    ],
    push_constant_ranges: &[
        PushConstantRange {
            stages: ShaderStages::VERTEX,
            range: 0..64,  // Transform matrix
        },
        PushConstantRange {
            stages: ShaderStages::FRAGMENT,
            range: 64..80,  // Material index + flags
        },
    ],
});
```

---

# TRINITY Resource Module Architecture

```
crates/renderer-backend/src/resources/
├── mod.rs              # Module root
├── buffer.rs           # Buffer types and allocator
├── texture.rs          # Texture types and cache
├── sampler.rs          # Sampler cache
├── bind_group.rs       # Bind group management
├── bindless.rs         # Bindless resource system
├── staging.rs          # Staging belt
├── pool.rs             # Memory pools
└── format.rs           # Format utilities
```

---

*End of WGPU_PART_II_RESOURCES.md*

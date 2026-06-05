# WGPU_PART_VI_SYNCHRONIZATION.md — Synchronization & Command Encoding

> **Scope**: Complete coverage of wgpu's command encoding model, copy operations, query system, debug instrumentation, synchronization primitives, and CPU-GPU coordination
> **TRINITY Integration**: Frame graph barrier resolution, submission batching, frame pacing
> **wgpu Version**: 25.x+

---

# Chapter 9: Command Encoding

Command encoding is the mechanism by which TRINITY records GPU work. Unlike immediate-mode APIs, wgpu uses a deferred command model: commands are recorded into command buffers, then submitted to the GPU queue for execution. This model enables validation, optimization, and cross-platform abstraction.

---

## 9.1 Command Encoder

### 9.1.1 Command Encoder Creation

The `CommandEncoder` is the primary interface for recording GPU commands outside of render/compute passes.

```rust
pub struct CommandEncoderDescriptor<'a> {
    pub label: Option<&'a str>,
}

impl Device {
    pub fn create_command_encoder(
        &self,
        desc: &CommandEncoderDescriptor,
    ) -> CommandEncoder;
}
```

**TRINITY Implementation**:

```rust
pub struct TrinityCommandEncoder {
    inner: wgpu::CommandEncoder,
    frame_index: u64,
    pass_count: u32,
    debug_stack_depth: u32,
    recorded_barriers: Vec<BarrierRecord>,
}

impl TrinityCommandEncoder {
    pub fn new(device: &TrinityDevice, frame_index: u64) -> Self {
        let inner = device.inner().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some(&format!("TrinityEncoder_Frame{}", frame_index)),
            }
        );
        
        Self {
            inner,
            frame_index,
            pass_count: 0,
            debug_stack_depth: 0,
            recorded_barriers: Vec::new(),
        }
    }
}
```

### 9.1.2 Encoder Scope and Lifetime

Command encoders follow strict lifetime rules:

1. **Single-use**: Each encoder produces exactly one `CommandBuffer`
2. **Non-reentrant**: Only one pass encoder can be active at a time
3. **Move semantics**: `finish()` consumes the encoder

```rust
// Lifetime diagram
let encoder = device.create_command_encoder(&desc);  // Encoder created

// Can record copy/clear commands here
encoder.copy_buffer_to_buffer(...);

{
    let mut render_pass = encoder.begin_render_pass(&desc);  // Encoder borrowed
    // Record render commands
    render_pass.draw(...);
}  // render_pass dropped, encoder available again

{
    let mut compute_pass = encoder.begin_compute_pass(&desc);  // Encoder borrowed
    // Record compute commands
    compute_pass.dispatch_workgroups(...);
}  // compute_pass dropped, encoder available again

let command_buffer = encoder.finish();  // Encoder consumed
queue.submit(std::iter::once(command_buffer));
```

**Error Conditions**:
- Creating a second pass while one is active → panic
- Using encoder after `finish()` → compile error (moved)
- Creating pass after `finish()` → compile error (moved)

### 9.1.3 Pass Encoder Creation

```rust
impl CommandEncoder {
    pub fn begin_render_pass<'a>(
        &'a mut self,
        desc: &RenderPassDescriptor<'a>,
    ) -> RenderPass<'a>;
    
    pub fn begin_compute_pass(
        &mut self,
        desc: &ComputePassDescriptor,
    ) -> ComputePass<'_>;
}
```

**TRINITY's Pass Recording**:

```rust
impl TrinityCommandEncoder {
    pub fn begin_render_pass<'a>(
        &'a mut self,
        pass_def: &FrameGraphRenderPass,
        resources: &'a ResolvedResources,
    ) -> TrinityRenderPass<'a> {
        self.pass_count += 1;
        
        let color_attachments: Vec<_> = pass_def.color_outputs
            .iter()
            .map(|output| {
                let view = resources.get_texture_view(output.resource);
                let resolve_target = output.resolve_target
                    .map(|rt| resources.get_texture_view(rt));
                
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target,
                    ops: wgpu::Operations {
                        load: output.load_op.into(),
                        store: output.store_op.into(),
                    },
                })
            })
            .collect();
        
        let depth_stencil_attachment = pass_def.depth_output.as_ref().map(|depth| {
            wgpu::RenderPassDepthStencilAttachment {
                view: resources.get_texture_view(depth.resource),
                depth_ops: Some(wgpu::Operations {
                    load: depth.depth_load_op.into(),
                    store: depth.depth_store_op.into(),
                }),
                stencil_ops: depth.stencil_ops.map(|ops| wgpu::Operations {
                    load: ops.load.into(),
                    store: ops.store.into(),
                }),
            }
        });
        
        let desc = wgpu::RenderPassDescriptor {
            label: Some(&pass_def.name),
            color_attachments: &color_attachments,
            depth_stencil_attachment,
            timestamp_writes: pass_def.timestamp_writes.as_ref(),
            occlusion_query_set: pass_def.occlusion_query_set.as_ref(),
        };
        
        let inner = self.inner.begin_render_pass(&desc);
        
        TrinityRenderPass {
            inner,
            pass_name: pass_def.name.clone(),
            draw_count: 0,
        }
    }
}
```

### 9.1.4 Command Buffer Finalization

```rust
impl CommandEncoder {
    pub fn finish(self) -> CommandBuffer;
}
```

The `finish()` method:
1. Validates all recorded commands
2. Optimizes barrier placement
3. Produces an immutable `CommandBuffer`

**TRINITY's Finalization**:

```rust
impl TrinityCommandEncoder {
    pub fn finish(self) -> TrinityCommandBuffer {
        // Validate debug group stack is empty
        assert_eq!(
            self.debug_stack_depth, 0,
            "Unbalanced debug groups: {} still open",
            self.debug_stack_depth
        );
        
        let inner = self.inner.finish();
        
        TrinityCommandBuffer {
            inner,
            frame_index: self.frame_index,
            pass_count: self.pass_count,
            barrier_count: self.recorded_barriers.len() as u32,
        }
    }
}

pub struct TrinityCommandBuffer {
    inner: wgpu::CommandBuffer,
    frame_index: u64,
    pass_count: u32,
    barrier_count: u32,
}
```

---

## 9.2 Copy Commands

Copy commands transfer data between buffers and textures. They execute outside of render/compute passes and are subject to automatic synchronization.

### 9.2.1 copy_buffer_to_buffer

```rust
impl CommandEncoder {
    pub fn copy_buffer_to_buffer(
        &mut self,
        source: &Buffer,
        source_offset: BufferAddress,
        destination: &Buffer,
        destination_offset: BufferAddress,
        copy_size: BufferAddress,
    );
}
```

**Constraints**:
- `source` must have `COPY_SRC` usage
- `destination` must have `COPY_DST` usage
- Offsets must be aligned to `COPY_BUFFER_ALIGNMENT` (4 bytes)
- `copy_size` must be aligned to `COPY_BUFFER_ALIGNMENT`
- Source and destination ranges must not overlap (even for same buffer)

**TRINITY Implementation**:

```rust
impl TrinityCommandEncoder {
    pub fn copy_buffer_region(
        &mut self,
        src: &TrinityBuffer,
        src_offset: u64,
        dst: &TrinityBuffer,
        dst_offset: u64,
        size: u64,
    ) {
        // Validate alignment
        const ALIGN: u64 = wgpu::COPY_BUFFER_ALIGNMENT as u64;
        assert!(src_offset % ALIGN == 0, "Source offset not aligned");
        assert!(dst_offset % ALIGN == 0, "Destination offset not aligned");
        assert!(size % ALIGN == 0, "Size not aligned");
        
        // Validate ranges
        assert!(src_offset + size <= src.size());
        assert!(dst_offset + size <= dst.size());
        
        // Record barrier
        self.recorded_barriers.push(BarrierRecord::BufferToBuffer {
            src: src.handle(),
            dst: dst.handle(),
        });
        
        self.inner.copy_buffer_to_buffer(
            src.inner(),
            src_offset,
            dst.inner(),
            dst_offset,
            size,
        );
    }
}
```

### 9.2.2 copy_buffer_to_texture

```rust
pub struct ImageCopyBuffer<'a> {
    pub buffer: &'a Buffer,
    pub layout: ImageDataLayout,
}

pub struct ImageDataLayout {
    pub offset: BufferAddress,
    pub bytes_per_row: Option<u32>,
    pub rows_per_image: Option<u32>,
}

pub struct ImageCopyTexture<'a> {
    pub texture: &'a Texture,
    pub mip_level: u32,
    pub origin: Origin3d,
    pub aspect: TextureAspect,
}

impl CommandEncoder {
    pub fn copy_buffer_to_texture(
        &mut self,
        source: ImageCopyBuffer<'_>,
        destination: ImageCopyTexture<'_>,
        copy_size: Extent3d,
    );
}
```

**Layout Calculation**:

```rust
impl TrinityCommandEncoder {
    pub fn copy_buffer_to_texture_2d(
        &mut self,
        buffer: &TrinityBuffer,
        buffer_offset: u64,
        texture: &TrinityTexture,
        mip_level: u32,
        origin: (u32, u32),
        extent: (u32, u32),
    ) {
        let format = texture.format();
        let block_size = format.block_dimensions();
        let bytes_per_block = format.block_copy_size(None).unwrap();
        
        // Calculate bytes_per_row (must be aligned to 256)
        let blocks_wide = (extent.0 + block_size.0 - 1) / block_size.0;
        let unaligned_bytes_per_row = blocks_wide * bytes_per_block;
        let bytes_per_row = align_to(unaligned_bytes_per_row, 256);
        
        // Calculate total buffer size needed
        let blocks_high = (extent.1 + block_size.1 - 1) / block_size.1;
        let required_size = (blocks_high - 1) * bytes_per_row + unaligned_bytes_per_row;
        
        assert!(buffer_offset + required_size as u64 <= buffer.size());
        
        self.inner.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: buffer.inner(),
                layout: wgpu::ImageDataLayout {
                    offset: buffer_offset,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::ImageCopyTexture {
                texture: texture.inner(),
                mip_level,
                origin: wgpu::Origin3d {
                    x: origin.0,
                    y: origin.1,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: extent.0,
                height: extent.1,
                depth_or_array_layers: 1,
            },
        );
    }
}

fn align_to(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}
```

### 9.2.3 copy_texture_to_buffer

```rust
impl CommandEncoder {
    pub fn copy_texture_to_buffer(
        &mut self,
        source: ImageCopyTexture<'_>,
        destination: ImageCopyBuffer<'_>,
        copy_size: Extent3d,
    );
}
```

**GPU Readback Pattern**:

```rust
impl TrinityCommandEncoder {
    pub fn schedule_texture_readback(
        &mut self,
        texture: &TrinityTexture,
        mip_level: u32,
        staging_buffer: &TrinityBuffer,
    ) -> ReadbackTicket {
        let mip_extent = texture.mip_extent(mip_level);
        let format = texture.format();
        let bytes_per_row = align_to(
            mip_extent.width * format.block_copy_size(None).unwrap(),
            256,
        );
        
        self.inner.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: texture.inner(),
                mip_level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: staging_buffer.inner(),
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            mip_extent,
        );
        
        ReadbackTicket {
            buffer: staging_buffer.clone(),
            extent: mip_extent,
            bytes_per_row,
            format,
        }
    }
}
```

### 9.2.4 copy_texture_to_texture

```rust
impl CommandEncoder {
    pub fn copy_texture_to_texture(
        &mut self,
        source: ImageCopyTexture<'_>,
        destination: ImageCopyTexture<'_>,
        copy_size: Extent3d,
    );
}
```

**Mipmap Generation via Copy**:

```rust
impl TrinityCommandEncoder {
    pub fn copy_mip_to_mip(
        &mut self,
        texture: &TrinityTexture,
        src_mip: u32,
        dst_mip: u32,
        src_origin: Origin3d,
        dst_origin: Origin3d,
        extent: Extent3d,
    ) {
        // Same texture, different mip levels
        self.inner.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: texture.inner(),
                mip_level: src_mip,
                origin: src_origin,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: texture.inner(),
                mip_level: dst_mip,
                origin: dst_origin,
                aspect: wgpu::TextureAspect::All,
            },
            extent,
        );
    }
}
```

### 9.2.5 Copy Alignment Requirements

| Constraint | Value | Notes |
|------------|-------|-------|
| Buffer offset alignment | 4 bytes | `COPY_BUFFER_ALIGNMENT` |
| Bytes per row alignment | 256 bytes | For texture copies |
| Texture offset alignment | Block size | Format-dependent |
| Copy size alignment | 4 bytes | For buffer copies |

**Alignment Helper**:

```rust
pub struct CopyAlignmentCalculator;

impl CopyAlignmentCalculator {
    pub fn buffer_copy_size(desired: u64) -> u64 {
        align_to_u64(desired, wgpu::COPY_BUFFER_ALIGNMENT as u64)
    }
    
    pub fn texture_row_pitch(width: u32, format: wgpu::TextureFormat) -> u32 {
        let block_size = format.block_dimensions();
        let bytes_per_block = format.block_copy_size(None).unwrap();
        let blocks_wide = (width + block_size.0 - 1) / block_size.0;
        align_to(blocks_wide * bytes_per_block, 256)
    }
    
    pub fn staging_buffer_size_for_texture(
        extent: wgpu::Extent3d,
        format: wgpu::TextureFormat,
    ) -> u64 {
        let row_pitch = Self::texture_row_pitch(extent.width, format) as u64;
        let block_size = format.block_dimensions();
        let blocks_high = (extent.height + block_size.1 - 1) / block_size.1;
        let layers = extent.depth_or_array_layers.max(1);
        
        row_pitch * blocks_high as u64 * layers as u64
    }
}
```

### 9.2.6 Copy Region Specification

**3D Texture Copy Example**:

```rust
impl TrinityCommandEncoder {
    pub fn copy_3d_region(
        &mut self,
        src_texture: &TrinityTexture,
        src_origin: (u32, u32, u32),
        dst_texture: &TrinityTexture,
        dst_origin: (u32, u32, u32),
        extent: (u32, u32, u32),
    ) {
        self.inner.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: src_texture.inner(),
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: src_origin.0,
                    y: src_origin.1,
                    z: src_origin.2,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: dst_texture.inner(),
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: dst_origin.0,
                    y: dst_origin.1,
                    z: dst_origin.2,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: extent.0,
                height: extent.1,
                depth_or_array_layers: extent.2,
            },
        );
    }
}
```

---

## 9.3 Clear Commands

### 9.3.1 clear_buffer

```rust
impl CommandEncoder {
    pub fn clear_buffer(
        &mut self,
        buffer: &Buffer,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    );
}
```

**Constraints**:
- Buffer must have `COPY_DST` usage
- Offset must be aligned to 4 bytes
- Size must be aligned to 4 bytes (or None for "rest of buffer")
- Clears to zero

**TRINITY Implementation**:

```rust
impl TrinityCommandEncoder {
    pub fn clear_buffer(&mut self, buffer: &TrinityBuffer) {
        self.inner.clear_buffer(buffer.inner(), 0, None);
    }
    
    pub fn clear_buffer_range(
        &mut self,
        buffer: &TrinityBuffer,
        offset: u64,
        size: u64,
    ) {
        assert!(offset % 4 == 0, "Offset must be 4-byte aligned");
        assert!(size % 4 == 0, "Size must be 4-byte aligned");
        assert!(offset + size <= buffer.size());
        
        self.inner.clear_buffer(buffer.inner(), offset, Some(size));
    }
}
```

### 9.3.2 Clear Texture (via Render Pass)

wgpu doesn't have a direct `clear_texture` command. Textures are cleared via render pass load operations.

```rust
impl TrinityCommandEncoder {
    pub fn clear_texture(
        &mut self,
        texture: &TrinityTexture,
        clear_value: ClearValue,
    ) {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        match texture.format().sample_type(None, None) {
            Some(wgpu::TextureSampleType::Depth) => {
                let depth_value = match clear_value {
                    ClearValue::Depth(d) => d,
                    _ => 1.0,
                };
                
                let _pass = self.inner.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ClearDepth"),
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(depth_value),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                // Pass immediately dropped, performing the clear
            }
            _ => {
                let color = match clear_value {
                    ClearValue::Color(c) => c,
                    _ => wgpu::Color::BLACK,
                };
                
                let _pass = self.inner.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("ClearColor"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(color),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
            }
        }
    }
}

pub enum ClearValue {
    Color(wgpu::Color),
    Depth(f32),
    DepthStencil(f32, u32),
}
```

### 9.3.3 Fill Patterns

For non-zero fill patterns, use a compute shader:

```wgsl
@group(0) @binding(0) var<storage, read_write> buffer: array<u32>;

struct FillParams {
    value: u32,
    count: u32,
}
@group(0) @binding(1) var<uniform> params: FillParams;

@compute @workgroup_size(256)
fn fill_buffer(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < params.count) {
        buffer[id.x] = params.value;
    }
}
```

**TRINITY Fill Implementation**:

```rust
impl TrinityCommandEncoder {
    pub fn fill_buffer_u32(
        &mut self,
        buffer: &TrinityBuffer,
        value: u32,
        pipelines: &ComputePipelines,
    ) {
        let count = (buffer.size() / 4) as u32;
        
        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FillParams"),
            contents: bytemuck::bytes_of(&[value, count]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FillBindGroup"),
            layout: pipelines.fill_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.inner().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        
        {
            let mut pass = self.inner.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FillBuffer"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipelines.fill_pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((count + 255) / 256, 1, 1);
        }
    }
}
```

---

## 9.4 Query Commands

### 9.4.1 Timestamp Queries

Timestamp queries measure GPU execution time with nanosecond precision.

```rust
pub struct QuerySetDescriptor<'a> {
    pub label: Option<&'a str>,
    pub ty: QueryType,
    pub count: u32,
}

pub enum QueryType {
    Occlusion,
    Timestamp,
    PipelineStatistics(PipelineStatisticsTypes),
}

impl Device {
    pub fn create_query_set(&self, desc: &QuerySetDescriptor) -> QuerySet;
}
```

**TRINITY Timestamp System**:

```rust
pub struct TimestampQueryPool {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    capacity: u32,
    next_index: u32,
    pending_readbacks: Vec<TimestampReadback>,
}

impl TimestampQueryPool {
    pub fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("TimestampQueries"),
            ty: wgpu::QueryType::Timestamp,
            count: capacity,
        });
        
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TimestampResolve"),
            size: capacity as u64 * 8, // u64 per timestamp
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TimestampReadback"),
            size: capacity as u64 * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            capacity,
            next_index: 0,
            pending_readbacks: Vec::new(),
        }
    }
    
    pub fn allocate_pair(&mut self) -> Option<(u32, u32)> {
        if self.next_index + 2 > self.capacity {
            return None;
        }
        let start = self.next_index;
        self.next_index += 2;
        Some((start, start + 1))
    }
    
    pub fn write_timestamp(&self, pass: &mut impl TimestampWriter, index: u32) {
        pass.write_timestamp(&self.query_set, index);
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

### 9.4.2 Occlusion Queries

Occlusion queries count how many samples pass depth/stencil tests.

```rust
pub struct OcclusionQuerySystem {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    query_count: u32,
    results: Vec<Option<u64>>,
}

impl OcclusionQuerySystem {
    pub fn new(device: &wgpu::Device, max_queries: u32) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("OcclusionQueries"),
            ty: wgpu::QueryType::Occlusion,
            count: max_queries,
        });
        
        let buffer_size = max_queries as u64 * 8;
        
        Self {
            query_set,
            resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("OcclusionResolve"),
                size: buffer_size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("OcclusionReadback"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            query_count: max_queries,
            results: vec![None; max_queries as usize],
        }
    }
    
    pub fn begin_query(&self, pass: &mut wgpu::RenderPass, index: u32) {
        pass.begin_occlusion_query(index);
    }
    
    pub fn end_query(&self, pass: &mut wgpu::RenderPass) {
        pass.end_occlusion_query();
    }
    
    pub fn is_visible(&self, index: u32) -> Option<bool> {
        self.results.get(index as usize)
            .and_then(|r| r.map(|count| count > 0))
    }
    
    pub fn sample_count(&self, index: u32) -> Option<u64> {
        self.results.get(index as usize).copied().flatten()
    }
}
```

### 9.4.3 Pipeline Statistics Queries (Where Supported)

```rust
// Feature: PIPELINE_STATISTICS_QUERY
bitflags::bitflags! {
    pub struct PipelineStatisticsTypes: u8 {
        const VERTEX_SHADER_INVOCATIONS = 1 << 0;
        const CLIPPER_INVOCATIONS = 1 << 1;
        const CLIPPER_PRIMITIVES_OUT = 1 << 2;
        const FRAGMENT_SHADER_INVOCATIONS = 1 << 3;
        const COMPUTE_SHADER_INVOCATIONS = 1 << 4;
    }
}
```

### 9.4.4 Query Set Creation

```rust
impl TrinityDevice {
    pub fn create_timestamp_query_pool(&self, count: u32) -> TimestampQueryPool {
        TimestampQueryPool::new(self.inner(), count)
    }
    
    pub fn create_occlusion_query_system(&self, count: u32) -> OcclusionQuerySystem {
        OcclusionQuerySystem::new(self.inner(), count)
    }
}
```

### 9.4.5 resolve_query_set

```rust
impl CommandEncoder {
    pub fn resolve_query_set(
        &mut self,
        query_set: &QuerySet,
        query_range: Range<u32>,
        destination: &Buffer,
        destination_offset: BufferAddress,
    );
}
```

**TRINITY Query Resolution**:

```rust
impl TrinityCommandEncoder {
    pub fn resolve_timestamps(&mut self, pool: &TimestampQueryPool) {
        if pool.next_index == 0 {
            return;
        }
        
        self.inner.resolve_query_set(
            &pool.query_set,
            0..pool.next_index,
            &pool.resolve_buffer,
            0,
        );
        
        self.inner.copy_buffer_to_buffer(
            &pool.resolve_buffer,
            0,
            &pool.readback_buffer,
            0,
            pool.next_index as u64 * 8,
        );
    }
    
    pub fn resolve_occlusion_queries(&mut self, system: &OcclusionQuerySystem) {
        self.inner.resolve_query_set(
            &system.query_set,
            0..system.query_count,
            &system.resolve_buffer,
            0,
        );
        
        self.inner.copy_buffer_to_buffer(
            &system.resolve_buffer,
            0,
            &system.readback_buffer,
            0,
            system.query_count as u64 * 8,
        );
    }
}
```

### 9.4.6 Query Result Readback

```rust
impl TimestampQueryPool {
    pub async fn read_results(&mut self, device: &wgpu::Device) -> Vec<TimestampResult> {
        let slice = self.readback_buffer.slice(..);
        
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        
        device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);
        
        let mut results = Vec::new();
        for i in (0..self.next_index).step_by(2) {
            let start = timestamps[i as usize];
            let end = timestamps[i as usize + 1];
            results.push(TimestampResult {
                query_index: i / 2,
                start_ticks: start,
                end_ticks: end,
                duration_ticks: end.saturating_sub(start),
            });
        }
        
        drop(data);
        self.readback_buffer.unmap();
        self.next_index = 0;
        
        results
    }
}

pub struct TimestampResult {
    pub query_index: u32,
    pub start_ticks: u64,
    pub end_ticks: u64,
    pub duration_ticks: u64,
}

impl TimestampResult {
    pub fn duration_ns(&self, timestamp_period: f32) -> f64 {
        self.duration_ticks as f64 * timestamp_period as f64
    }
    
    pub fn duration_ms(&self, timestamp_period: f32) -> f64 {
        self.duration_ns(timestamp_period) / 1_000_000.0
    }
}
```

---

## 9.5 Debug Commands

### 9.5.1 push_debug_group / pop_debug_group

Debug groups create hierarchical regions visible in GPU debuggers.

```rust
impl CommandEncoder {
    pub fn push_debug_group(&mut self, label: &str);
    pub fn pop_debug_group(&mut self);
}

impl RenderPass<'_> {
    pub fn push_debug_group(&mut self, label: &str);
    pub fn pop_debug_group(&mut self);
}

impl ComputePass<'_> {
    pub fn push_debug_group(&mut self, label: &str);
    pub fn pop_debug_group(&mut self);
}
```

**TRINITY Debug Group RAII**:

```rust
pub struct DebugScope<'a, T> {
    pass: &'a mut T,
}

impl<'a, T: DebugGroupSupport> DebugScope<'a, T> {
    pub fn new(pass: &'a mut T, label: &str) -> Self {
        pass.push_debug_group(label);
        Self { pass }
    }
}

impl<T: DebugGroupSupport> Drop for DebugScope<'_, T> {
    fn drop(&mut self) {
        self.pass.pop_debug_group();
    }
}

pub trait DebugGroupSupport {
    fn push_debug_group(&mut self, label: &str);
    fn pop_debug_group(&mut self);
}

impl DebugGroupSupport for wgpu::RenderPass<'_> {
    fn push_debug_group(&mut self, label: &str) {
        self.push_debug_group(label);
    }
    fn pop_debug_group(&mut self) {
        self.pop_debug_group();
    }
}

// Usage:
// {
//     let _scope = DebugScope::new(&mut render_pass, "Shadow Map Generation");
//     // ... render commands ...
// } // Automatically pops on drop
```

### 9.5.2 insert_debug_marker

```rust
impl CommandEncoder {
    pub fn insert_debug_marker(&mut self, label: &str);
}

impl RenderPass<'_> {
    pub fn insert_debug_marker(&mut self, label: &str);
}

impl ComputePass<'_> {
    pub fn insert_debug_marker(&mut self, label: &str);
}
```

**TRINITY Marker System**:

```rust
impl TrinityRenderPass<'_> {
    pub fn mark_draw_call(&mut self, object_name: &str) {
        #[cfg(debug_assertions)]
        self.inner.insert_debug_marker(&format!("Draw: {}", object_name));
        self.draw_count += 1;
    }
    
    pub fn mark_state_change(&mut self, change: &str) {
        #[cfg(debug_assertions)]
        self.inner.insert_debug_marker(&format!("State: {}", change));
    }
}
```

### 9.5.3 Debug Labels on Resources

All wgpu resources support labels via their descriptors:

```rust
pub struct BufferDescriptor<'a> {
    pub label: Option<&'a str>,
    // ...
}

pub struct TextureDescriptor<'a> {
    pub label: Option<&'a str>,
    // ...
}

pub struct ShaderModuleDescriptor<'a> {
    pub label: Option<&'a str>,
    // ...
}

pub struct RenderPipelineDescriptor<'a> {
    pub label: Option<&'a str>,
    // ...
}
```

**TRINITY Label Convention**:

```rust
pub trait Labeled {
    fn trinity_label(&self) -> String;
}

impl Labeled for TrinityBuffer {
    fn trinity_label(&self) -> String {
        format!("Buffer_{}_{:?}", self.name, self.usage)
    }
}

impl Labeled for TrinityTexture {
    fn trinity_label(&self) -> String {
        format!("Texture_{}_{}x{}_{:?}", 
            self.name, 
            self.extent.width, 
            self.extent.height,
            self.format
        )
    }
}

impl Labeled for TrinityPipeline {
    fn trinity_label(&self) -> String {
        format!("Pipeline_{}", self.shader_name)
    }
}
```

### 9.5.4 Integration with GPU Debuggers

**RenderDoc Integration**:

```rust
#[cfg(feature = "renderdoc")]
pub struct RenderDocCapture {
    api: renderdoc::RenderDoc<renderdoc::V141>,
}

#[cfg(feature = "renderdoc")]
impl RenderDocCapture {
    pub fn new() -> Option<Self> {
        renderdoc::RenderDoc::new().ok().map(|api| Self { api })
    }
    
    pub fn start_capture(&mut self) {
        self.api.start_frame_capture(std::ptr::null(), std::ptr::null());
    }
    
    pub fn end_capture(&mut self) {
        self.api.end_frame_capture(std::ptr::null(), std::ptr::null());
    }
    
    pub fn trigger_capture(&mut self) {
        self.api.trigger_capture();
    }
}

impl TrinityDevice {
    pub fn capture_next_frame(&self) {
        #[cfg(feature = "renderdoc")]
        if let Some(ref mut rd) = *self.renderdoc.lock() {
            rd.trigger_capture();
        }
    }
}
```

---

# Chapter 10: Synchronization

wgpu provides both implicit and explicit synchronization mechanisms. Understanding when each is needed is critical for correctness and performance.

---

## 10.1 Implicit Synchronization

### 10.1.1 wgpu's Automatic Barrier Insertion

wgpu tracks resource states internally and inserts appropriate barriers automatically. This is the default behavior and covers most use cases.

```rust
// wgpu automatically handles these transitions:
encoder.copy_buffer_to_buffer(&staging, 0, &gpu_buffer, 0, size);
// ↑ barrier inserted: gpu_buffer COPY_DST

{
    let mut pass = encoder.begin_compute_pass(&desc);
    pass.set_bind_group(0, &bind_group, &[]); // bind_group contains gpu_buffer
    pass.dispatch_workgroups(64, 1, 1);
    // ↑ barrier inserted: gpu_buffer STORAGE_READ or STORAGE_WRITE
}

{
    let mut pass = encoder.begin_render_pass(&desc);
    pass.set_vertex_buffer(0, gpu_buffer.slice(..));
    pass.draw(100, 1, 0, 0);
    // ↑ barrier inserted: gpu_buffer VERTEX
}
```

### 10.1.2 Resource Usage Tracking

wgpu internally tracks:
- Current resource state (UNDEFINED, COPY_SRC, COPY_DST, SHADER_READ, etc.)
- Required transitions between passes
- Read vs write hazards

**Tracked States Per Resource Type**:

| Resource | Possible States |
|----------|-----------------|
| Buffer | COPY_SRC, COPY_DST, INDEX, VERTEX, UNIFORM, STORAGE_READ, STORAGE_WRITE, INDIRECT, QUERY_RESOLVE |
| Texture | COPY_SRC, COPY_DST, SAMPLE, STORAGE_READ, STORAGE_WRITE, RENDER_ATTACHMENT |

### 10.1.3 Pass Ordering Semantics

Passes within a command buffer execute **sequentially** in recorded order:

```rust
// Pass A always completes before Pass B starts
{
    let mut pass_a = encoder.begin_render_pass(&desc_a);
    // ... A's commands ...
}
{
    let mut pass_b = encoder.begin_render_pass(&desc_b);
    // ... B's commands ...
}
```

Command buffers submitted together are **not reordered**:

```rust
let cmd_a = encoder_a.finish();
let cmd_b = encoder_b.finish();

// A completes before B starts
queue.submit([cmd_a, cmd_b]);
```

### 10.1.4 When Implicit Sync is Sufficient

Implicit synchronization is sufficient when:
1. Resources are used in different passes
2. Resource transitions are between passes (not within)
3. No workgroup-level coordination is needed

```rust
// ✓ Implicit sync handles this
encoder.copy_buffer_to_texture(&staging, &texture, size);
{
    let mut pass = encoder.begin_render_pass(&desc);
    // texture automatically transitioned to SAMPLE state
    pass.set_bind_group(0, &sampler_bind_group, &[]);
    pass.draw(6, 1, 0, 0);
}
```

---

## 10.2 Explicit Synchronization

### 10.2.1 Memory Barriers in Compute Shaders

WGSL provides explicit barrier functions for synchronization within a compute shader:

```wgsl
// Synchronize all invocations in a workgroup
workgroupBarrier();

// Ensure memory writes to storage buffers are visible
storageBarrier();

// Ensure memory writes to textures are visible (where supported)
textureBarrier();
```

### 10.2.2 workgroupBarrier()

`workgroupBarrier()` is a **control barrier** that:
1. Synchronizes execution: all invocations reach the barrier before any proceed
2. Includes memory semantics for workgroup memory

```wgsl
var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn prefix_sum(@builtin(local_invocation_id) lid: vec3<u32>) {
    // Each thread loads one element
    shared_data[lid.x] = input_buffer[lid.x];
    
    // MUST synchronize before reading neighbor's data
    workgroupBarrier();
    
    // Now safe to read from shared_data written by other invocations
    if (lid.x > 0) {
        shared_data[lid.x] += shared_data[lid.x - 1];
    }
    
    workgroupBarrier();
    
    output_buffer[lid.x] = shared_data[lid.x];
}
```

### 10.2.3 storageBarrier()

`storageBarrier()` ensures writes to storage buffers are visible to other invocations:

```wgsl
@group(0) @binding(0) var<storage, read_write> buffer: array<u32>;

@compute @workgroup_size(64)
fn producer_consumer(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Producer phase
    buffer[gid.x] = compute_value(gid.x);
    
    // Ensure write is visible
    storageBarrier();
    
    // Consumer phase (reads from different location)
    let neighbor = buffer[(gid.x + 1) % arrayLength(&buffer)];
    // ...
}
```

**Key Difference**:
- `workgroupBarrier()`: Control + memory, workgroup scope, includes workgroup memory
- `storageBarrier()`: Memory only, workgroup scope, storage buffers only

### 10.2.4 textureBarrier() (Where Supported)

```wgsl
// Feature: texture_barrier (experimental)
@group(0) @binding(0) var tex: texture_storage_2d<rgba8unorm, read_write>;

@compute @workgroup_size(8, 8)
fn blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Write phase
    textureStorageBarrier();
    
    // Read phase
}
```

### 10.2.5 Full Memory Barrier

Combining barriers for comprehensive synchronization:

```wgsl
@compute @workgroup_size(64)
fn comprehensive_sync() {
    // Write to all memory types
    shared_data[lid.x] = value1;
    storage_buffer[gid.x] = value2;
    
    // Full synchronization
    workgroupBarrier();  // Includes workgroup memory
    storageBarrier();    // Includes storage buffers
    
    // All writes visible to all invocations in workgroup
}
```

---

## 10.3 CPU-GPU Synchronization

### 10.3.1 Buffer Mapping and Callbacks

Buffer mapping is the primary mechanism for CPU-GPU data transfer and synchronization.

```rust
// Async mapping with callback
buffer.slice(..).map_async(wgpu::MapMode::Read, |result| {
    match result {
        Ok(()) => println!("Buffer mapped successfully"),
        Err(e) => eprintln!("Mapping failed: {:?}", e),
    }
});

// Must poll device for callback to fire
device.poll(wgpu::Maintain::Wait);
```

**TRINITY Mapping Abstraction**:

```rust
pub struct BufferReadback<T> {
    buffer: wgpu::Buffer,
    size: u64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: bytemuck::Pod> BufferReadback<T> {
    pub async fn read(&self, device: &wgpu::Device) -> Vec<T> {
        let slice = self.buffer.slice(..);
        
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        
        // Poll until mapping completes
        device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.buffer.unmap();
        
        result
    }
}
```

### 10.3.2 device.poll() Semantics

```rust
pub enum Maintain {
    Wait,      // Block until all submitted work completes
    WaitForSubmissionIndex(SubmissionIndex), // Wait for specific submission
    Poll,      // Non-blocking: process completed work
}

impl Device {
    pub fn poll(&self, maintain: Maintain) -> MaintainResult;
}

pub enum MaintainResult {
    SubmissionQueueEmpty,  // All work complete
    Ok,                    // Some work may remain
}
```

**TRINITY Frame Synchronization**:

```rust
impl TrinityDevice {
    pub fn wait_idle(&self) {
        self.inner.poll(wgpu::Maintain::Wait);
    }
    
    pub fn process_completed(&self) -> bool {
        matches!(
            self.inner.poll(wgpu::Maintain::Poll),
            wgpu::MaintainResult::SubmissionQueueEmpty
        )
    }
    
    pub fn wait_for_submission(&self, index: wgpu::SubmissionIndex) {
        self.inner.poll(wgpu::Maintain::WaitForSubmissionIndex(index));
    }
}
```

### 10.3.3 Async Mapping with Futures

```rust
use std::future::Future;

pub struct AsyncBufferMap {
    buffer: Arc<wgpu::Buffer>,
    device: Arc<wgpu::Device>,
}

impl AsyncBufferMap {
    pub fn map_read(&self) -> impl Future<Output = Result<BufferView, wgpu::BufferAsyncError>> {
        let buffer = self.buffer.clone();
        let device = self.device.clone();
        
        async move {
            let slice = buffer.slice(..);
            
            let (tx, rx) = tokio::sync::oneshot::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            
            // Spawn polling task
            let device_clone = device.clone();
            tokio::task::spawn_blocking(move || {
                device_clone.poll(wgpu::Maintain::Wait);
            }).await.unwrap();
            
            rx.await.unwrap()?;
            
            Ok(BufferView {
                buffer,
                range: slice.get_mapped_range(),
            })
        }
    }
}
```

### 10.3.4 Fence-Like Patterns

wgpu doesn't expose explicit fences, but submission indices provide similar functionality:

```rust
pub struct FrameFence {
    submission_index: Option<wgpu::SubmissionIndex>,
    frame_number: u64,
}

impl FrameFence {
    pub fn signal(&mut self, queue: &wgpu::Queue, frame: u64) {
        // Submit empty work to get a submission index
        self.submission_index = Some(queue.submit(std::iter::empty()));
        self.frame_number = frame;
    }
    
    pub fn wait(&self, device: &wgpu::Device) {
        if let Some(index) = self.submission_index {
            device.poll(wgpu::Maintain::WaitForSubmissionIndex(index));
        }
    }
    
    pub fn is_complete(&self, device: &wgpu::Device) -> bool {
        if self.submission_index.is_none() {
            return true;
        }
        
        matches!(
            device.poll(wgpu::Maintain::Poll),
            wgpu::MaintainResult::SubmissionQueueEmpty
        )
    }
}
```

### 10.3.5 Frame Pacing Strategies

**Strategy 1: Single Buffering (Simplest)**
```rust
impl SingleBufferedRenderer {
    pub fn render_frame(&mut self) {
        let output = self.surface.get_current_texture().unwrap();
        
        // Record and submit
        let cmd = self.record_frame(&output);
        self.queue.submit(std::iter::once(cmd));
        
        // Wait for completion before presenting
        self.device.poll(wgpu::Maintain::Wait);
        
        output.present();
    }
}
```

**Strategy 2: Double Buffering (Standard)**
```rust
impl DoubleBufferedRenderer {
    frame_fences: [FrameFence; 2],
    current_frame: usize,
    
    pub fn render_frame(&mut self) {
        let frame_index = self.current_frame;
        let next_frame = (frame_index + 1) % 2;
        
        // Wait for previous frame's GPU work (2 frames ago for this slot)
        self.frame_fences[frame_index].wait(&self.device);
        
        let output = self.surface.get_current_texture().unwrap();
        let cmd = self.record_frame(&output);
        
        // Submit and record fence
        let submission = self.queue.submit(std::iter::once(cmd));
        self.frame_fences[frame_index].submission_index = Some(submission);
        
        output.present();
        self.current_frame = next_frame;
    }
}
```

**Strategy 3: Triple Buffering (Low Latency)**
```rust
impl TripleBufferedRenderer {
    frame_fences: [FrameFence; 3],
    current_frame: usize,
    
    pub fn render_frame(&mut self) {
        let frame_index = self.current_frame;
        
        // Only wait if 3 frames in flight already
        if self.frame_fences[frame_index].submission_index.is_some() {
            self.frame_fences[frame_index].wait(&self.device);
        }
        
        let output = self.surface.get_current_texture().unwrap();
        let cmd = self.record_frame(&output);
        
        let submission = self.queue.submit(std::iter::once(cmd));
        self.frame_fences[frame_index].submission_index = Some(submission);
        
        output.present();
        self.current_frame = (frame_index + 1) % 3;
    }
}
```

### 10.3.6 TRINITY's Frame Synchronization Model

```rust
pub struct TrinityFrameSynchronizer {
    max_frames_in_flight: u32,
    frame_data: Vec<FrameInFlightData>,
    current_frame_index: u32,
    global_frame_counter: u64,
}

struct FrameInFlightData {
    submission_index: Option<wgpu::SubmissionIndex>,
    staging_buffers: Vec<TrinityBuffer>,
    query_pools: Vec<TimestampQueryPool>,
    timestamp_period: f32,
}

impl TrinityFrameSynchronizer {
    pub fn new(device: &TrinityDevice, max_frames: u32) -> Self {
        let timestamp_period = device.queue().get_timestamp_period();
        
        let frame_data = (0..max_frames)
            .map(|_| FrameInFlightData {
                submission_index: None,
                staging_buffers: Vec::new(),
                query_pools: Vec::new(),
                timestamp_period,
            })
            .collect();
        
        Self {
            max_frames_in_flight: max_frames,
            frame_data,
            current_frame_index: 0,
            global_frame_counter: 0,
        }
    }
    
    pub fn begin_frame(&mut self, device: &TrinityDevice) -> FrameContext {
        let frame_index = self.current_frame_index as usize;
        let frame_data = &mut self.frame_data[frame_index];
        
        // Wait for this slot's previous submission
        if let Some(submission) = frame_data.submission_index.take() {
            device.inner().poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
        }
        
        // Recycle staging buffers
        frame_data.staging_buffers.clear();
        
        // Read back query results from 2 frames ago
        if self.global_frame_counter >= 2 {
            let readback_index = ((frame_index + self.max_frames_in_flight as usize - 2) 
                                  % self.max_frames_in_flight as usize);
            // Process timestamps from frame_data[readback_index]
        }
        
        FrameContext {
            frame_number: self.global_frame_counter,
            frame_index: self.current_frame_index,
        }
    }
    
    pub fn end_frame(
        &mut self,
        queue: &wgpu::Queue,
        command_buffers: impl IntoIterator<Item = wgpu::CommandBuffer>,
    ) {
        let frame_index = self.current_frame_index as usize;
        
        let submission = queue.submit(command_buffers);
        self.frame_data[frame_index].submission_index = Some(submission);
        
        self.current_frame_index = (self.current_frame_index + 1) % self.max_frames_in_flight;
        self.global_frame_counter += 1;
    }
    
    pub fn wait_idle(&self, device: &TrinityDevice) {
        device.inner().poll(wgpu::Maintain::Wait);
    }
}

pub struct FrameContext {
    pub frame_number: u64,
    pub frame_index: u32,
}
```

---

## 10.4 Resource State Tracking

### 10.4.1 Resource States in wgpu

wgpu internally tracks resource states but abstracts this from the user. The conceptual states are:

| State | Description | Valid For |
|-------|-------------|-----------|
| Undefined | Initial state | All |
| CopySrc | Being read by copy | All |
| CopyDst | Being written by copy | All |
| ShaderRead | Sampled/uniform read | Texture, Buffer |
| ShaderWrite | Storage write | Texture, Buffer |
| RenderTarget | Color/depth attachment | Texture |
| Present | Ready for presentation | Surface texture |

### 10.4.2 Transition Barriers

wgpu inserts barriers automatically, but understanding them helps with optimization:

```rust
// Conceptual barrier types (internal to wgpu)
enum BarrierType {
    // Read-after-write: must wait for write to complete
    ReadAfterWrite {
        src_stage: PipelineStage,
        dst_stage: PipelineStage,
    },
    // Write-after-read: must wait for read to complete
    WriteAfterRead {
        src_stage: PipelineStage,
        dst_stage: PipelineStage,
    },
    // Write-after-write: must wait for previous write
    WriteAfterWrite {
        src_stage: PipelineStage,
        dst_stage: PipelineStage,
    },
}
```

### 10.4.3 Split Barriers (Where Supported)

Some backends support split barriers for better pipelining:

```rust
// Conceptual split barrier (not directly exposed in wgpu)
// Begin transition early
encoder.begin_transition(&texture, TextureState::ShaderRead);

// ... other work that doesn't use the texture ...

// Complete transition when needed
encoder.end_transition(&texture);
```

### 10.4.4 TRINITY's Frame Graph Barrier Resolution

```rust
pub struct BarrierResolver {
    resource_states: HashMap<ResourceHandle, ResourceState>,
    pending_barriers: Vec<Barrier>,
}

#[derive(Clone, Copy)]
pub struct ResourceState {
    pub stage: PipelineStage,
    pub access: AccessFlags,
    pub layout: Option<TextureLayout>,
}

impl BarrierResolver {
    pub fn new() -> Self {
        Self {
            resource_states: HashMap::new(),
            pending_barriers: Vec::new(),
        }
    }
    
    pub fn transition(
        &mut self,
        resource: ResourceHandle,
        new_state: ResourceState,
    ) {
        let old_state = self.resource_states
            .get(&resource)
            .copied()
            .unwrap_or(ResourceState::UNDEFINED);
        
        if self.needs_barrier(old_state, new_state) {
            self.pending_barriers.push(Barrier {
                resource,
                old_state,
                new_state,
            });
        }
        
        self.resource_states.insert(resource, new_state);
    }
    
    fn needs_barrier(&self, old: ResourceState, new: ResourceState) -> bool {
        // Write-after-read
        if old.access.contains(AccessFlags::READ) && new.access.contains(AccessFlags::WRITE) {
            return true;
        }
        // Read-after-write
        if old.access.contains(AccessFlags::WRITE) && new.access.contains(AccessFlags::READ) {
            return true;
        }
        // Write-after-write
        if old.access.contains(AccessFlags::WRITE) && new.access.contains(AccessFlags::WRITE) {
            return true;
        }
        // Layout transition
        if old.layout != new.layout {
            return true;
        }
        false
    }
    
    pub fn flush_barriers(&mut self) -> Vec<Barrier> {
        std::mem::take(&mut self.pending_barriers)
    }
}

bitflags::bitflags! {
    pub struct AccessFlags: u32 {
        const READ = 0x1;
        const WRITE = 0x2;
    }
    
    pub struct PipelineStage: u32 {
        const TOP = 0x1;
        const VERTEX = 0x2;
        const FRAGMENT = 0x4;
        const COMPUTE = 0x8;
        const COPY = 0x10;
        const BOTTOM = 0x20;
    }
}
```

**Barrier Batching**:

```rust
impl BarrierResolver {
    pub fn batch_barriers(&self, barriers: &[Barrier]) -> Vec<BarrierBatch> {
        let mut batches: Vec<BarrierBatch> = Vec::new();
        
        for barrier in barriers {
            // Find or create batch with compatible stages
            let batch = batches
                .iter_mut()
                .find(|b| b.can_merge(&barrier));
            
            match batch {
                Some(b) => b.add(barrier.clone()),
                None => batches.push(BarrierBatch::new(barrier.clone())),
            }
        }
        
        batches
    }
}

pub struct BarrierBatch {
    src_stages: PipelineStage,
    dst_stages: PipelineStage,
    buffer_barriers: Vec<BufferBarrier>,
    texture_barriers: Vec<TextureBarrier>,
}

impl BarrierBatch {
    fn can_merge(&self, barrier: &Barrier) -> bool {
        // Can merge if stages are compatible (subset or same)
        self.src_stages.contains(barrier.old_state.stage)
            && self.dst_stages.contains(barrier.new_state.stage)
    }
}
```

---

# TRINITY Synchronization Summary

| Scenario | Mechanism | TRINITY API |
|----------|-----------|-------------|
| Between passes | Implicit (wgpu handles) | Automatic |
| Within compute workgroup | `workgroupBarrier()` | WGSL |
| Storage buffer coherence | `storageBarrier()` | WGSL |
| CPU readback | Buffer mapping + poll | `BufferReadback::read()` |
| Frame pacing | Submission indices | `TrinityFrameSynchronizer` |
| Debug regions | push/pop_debug_group | `DebugScope` RAII |
| Performance queries | Timestamp queries | `TimestampQueryPool` |
| Occlusion testing | Occlusion queries | `OcclusionQuerySystem` |

---

*End of WGPU_PART_VI_SYNCHRONIZATION.md*

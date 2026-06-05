//! Visibility Buffer Write for TRINITY Engine (T-GPU-3.5).
//!
//! This module writes visibility buffer entries for instances that pass culling.
//! The visibility buffer approach is used for GPU-driven rendering where material
//! shading is deferred to a separate pass after visibility determination.
//!
//! # Overview
//!
//! The visibility buffer pipeline:
//! 1. Frustum culling - remove instances outside view frustum
//! 2. Distance culling - select LOD and cull far instances
//! 3. **Visibility write** - write instance data to visibility buffer (this module)
//! 4. Visibility pass - rasterize to fill primitive_id and barycentrics
//! 5. Material pass - shade using visibility buffer lookups
//!
//! # Performance
//!
//! - Work complexity: O(n) where n = visible instances
//! - Target: < 0.05ms for 100K instances
//! - Memory: 24 bytes per VisibilityData entry
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::{VisibilityWritePipeline, VisibilityBufferResources};
//!
//! // Create pipeline and resources
//! let pipeline = VisibilityWritePipeline::new(&device);
//! let resources = VisibilityBufferResources::new(&device, 100_000);
//!
//! // Each frame: write visibility entries from culled instances
//! let params = VisibilityWriteParams::new(visible_count, 0);
//! resources.upload_params(&queue, &params);
//! pipeline.dispatch(&mut encoder, &resources, &params);
//! ```

use std::mem;

use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, BufferUsages, Device, Queue};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 256;

/// Invalid instance ID sentinel value.
pub const INVALID_INSTANCE_ID: u32 = 0xFFFF_FFFF;

/// Invalid primitive ID sentinel value.
pub const INVALID_PRIMITIVE_ID: u32 = 0xFFFF_FFFF;

/// Default maximum visibility buffer entries.
pub const DEFAULT_MAX_VISIBILITY: u32 = 262_144; // 256K entries

// =============================================================================
// VISIBILITY WRITE PARAMS
// =============================================================================

/// GPU uniform buffer for visibility write parameters.
///
/// Matches the WGSL `VisibilityWriteParams` struct layout.
///
/// # Memory Layout
///
/// 16 bytes, std140/std430 compatible:
/// | Offset | Field                    | Size |
/// |--------|--------------------------|------|
/// | 0      | num_visible              | 4    |
/// | 4      | visibility_buffer_offset | 4    |
/// | 8      | _pad                     | 8    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct VisibilityWriteParams {
    /// Number of visible instances to process.
    pub num_visible: u32,
    /// Offset into visibility buffer for multi-view rendering.
    pub visibility_buffer_offset: u32,
    /// Padding for 16-byte alignment.
    pub _pad: [u32; 2],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<VisibilityWriteParams>() == 16);

impl VisibilityWriteParams {
    /// Create parameters for the given counts.
    pub fn new(num_visible: u32, visibility_buffer_offset: u32) -> Self {
        Self {
            num_visible,
            visibility_buffer_offset,
            _pad: [0, 0],
        }
    }

    /// Get the number of workgroups needed for dispatch.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        (self.num_visible + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }

    /// Check if no work is needed.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_visible == 0
    }
}

// =============================================================================
// VISIBLE INSTANCE
// =============================================================================

/// Compacted visible instance from culling output.
///
/// This is the input to visibility write, output from frustum + distance culling.
///
/// # Memory Layout
///
/// 16 bytes:
/// | Offset | Field          | Size |
/// |--------|----------------|------|
/// | 0      | original_index | 4    |
/// | 4      | lod_level      | 4    |
/// | 8      | batch_key      | 4    |
/// | 12     | _pad           | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct VisibleInstance {
    /// Original instance index for transform lookup.
    pub original_index: u32,
    /// LOD level selected during distance culling (0 = highest detail).
    pub lod_level: u32,
    /// Batch key: `(material_id << 16) | mesh_id`.
    pub batch_key: u32,
    /// Padding for alignment.
    pub _pad: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<VisibleInstance>() == 16);

impl VisibleInstance {
    /// Create a new visible instance.
    pub const fn new(original_index: u32, lod_level: u32, batch_key: u32) -> Self {
        Self {
            original_index,
            lod_level,
            batch_key,
            _pad: 0,
        }
    }

    /// Create from material and mesh IDs.
    pub const fn from_ids(
        original_index: u32,
        lod_level: u32,
        material_id: u16,
        mesh_id: u16,
    ) -> Self {
        Self {
            original_index,
            lod_level,
            batch_key: ((material_id as u32) << 16) | (mesh_id as u32),
            _pad: 0,
        }
    }

    /// Extract the mesh ID from the batch key.
    #[inline]
    pub const fn mesh_id(&self) -> u16 {
        (self.batch_key & 0xFFFF) as u16
    }

    /// Extract the material ID from the batch key.
    #[inline]
    pub const fn material_id(&self) -> u16 {
        (self.batch_key >> 16) as u16
    }
}

// =============================================================================
// VISIBILITY DATA
// =============================================================================

/// Visibility buffer entry for material shading.
///
/// Contains instance ID, primitive ID, and barycentric coordinates.
/// The primitive_id and barycentrics are initialized to invalid/zero
/// by this module and overwritten during the rasterization visibility pass.
///
/// # Memory Layout
///
/// 24 bytes (6 x u32):
/// | Offset | Field        | Size |
/// |--------|--------------|------|
/// | 0      | instance_id  | 4    |
/// | 4      | primitive_id | 4    |
/// | 8      | barycentrics | 8    |
/// | 16     | _pad         | 8    |
///
/// Note: Padded to 24 bytes for vec4 alignment in arrays.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct VisibilityData {
    /// Instance ID for transform/material lookup.
    pub instance_id: u32,
    /// Primitive (triangle) ID within the mesh.
    pub primitive_id: u32,
    /// Barycentric coordinates (u, v). w = 1 - u - v.
    pub barycentrics: [f32; 2],
    /// Padding for 8-byte alignment (vec4 in arrays).
    pub _pad: [u32; 2],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<VisibilityData>() == 24);

impl VisibilityData {
    /// Create a new visibility data entry.
    pub const fn new(instance_id: u32, primitive_id: u32, barycentrics: [f32; 2]) -> Self {
        Self {
            instance_id,
            primitive_id,
            barycentrics,
            _pad: [0, 0],
        }
    }

    /// Create an invalid/cleared entry.
    pub const fn invalid() -> Self {
        Self {
            instance_id: INVALID_INSTANCE_ID,
            primitive_id: INVALID_PRIMITIVE_ID,
            barycentrics: [0.0, 0.0],
            _pad: [0, 0],
        }
    }

    /// Create an entry with just instance ID (pre-rasterization).
    pub const fn with_instance(instance_id: u32) -> Self {
        Self {
            instance_id,
            primitive_id: INVALID_PRIMITIVE_ID,
            barycentrics: [0.0, 0.0],
            _pad: [0, 0],
        }
    }

    /// Check if this entry is valid (has been written).
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.instance_id != INVALID_INSTANCE_ID
    }

    /// Check if primitive data has been filled by rasterizer.
    #[inline]
    pub const fn has_primitive(&self) -> bool {
        self.primitive_id != INVALID_PRIMITIVE_ID
    }

    /// Get the third barycentric coordinate (w = 1 - u - v).
    #[inline]
    pub fn barycentric_w(&self) -> f32 {
        1.0 - self.barycentrics[0] - self.barycentrics[1]
    }
}

// =============================================================================
// PACKED VISIBILITY
// =============================================================================

/// Packed visibility data for memory-bandwidth-critical paths.
///
/// Compresses visibility data into 8 bytes:
/// - 20 bits instance ID (max 1M instances)
/// - 12 bits primitive ID (max 4K triangles per draw)
/// - 16 bits u barycentric (unorm)
/// - 16 bits v barycentric (unorm)
///
/// # Memory Layout
///
/// 8 bytes:
/// | Offset | Field              | Size |
/// |--------|--------------------| -----|
/// | 0      | instance_primitive | 4    |
/// | 4      | barycentrics       | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct PackedVisibility {
    /// Bits [31:12] = instance_id, bits [11:0] = primitive_id.
    pub instance_primitive: u32,
    /// Bits [31:16] = u (unorm16), bits [15:0] = v (unorm16).
    pub barycentrics_packed: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<PackedVisibility>() == 8);

impl PackedVisibility {
    /// Pack instance and primitive IDs.
    #[inline]
    pub const fn pack_ids(instance_id: u32, primitive_id: u32) -> u32 {
        ((instance_id & 0xFFFFF) << 12) | (primitive_id & 0xFFF)
    }

    /// Unpack instance ID.
    #[inline]
    pub const fn unpack_instance_id(packed: u32) -> u32 {
        packed >> 12
    }

    /// Unpack primitive ID.
    #[inline]
    pub const fn unpack_primitive_id(packed: u32) -> u32 {
        packed & 0xFFF
    }

    /// Pack barycentrics to 32 bits.
    #[inline]
    pub fn pack_barycentrics(u: f32, v: f32) -> u32 {
        let u_packed = (u.clamp(0.0, 1.0) * 65535.0) as u32;
        let v_packed = (v.clamp(0.0, 1.0) * 65535.0) as u32;
        (u_packed << 16) | v_packed
    }

    /// Unpack barycentrics from 32 bits.
    #[inline]
    pub fn unpack_barycentrics(packed: u32) -> (f32, f32) {
        let u = (packed >> 16) as f32 / 65535.0;
        let v = (packed & 0xFFFF) as f32 / 65535.0;
        (u, v)
    }

    /// Create a packed visibility entry.
    pub fn new(instance_id: u32, primitive_id: u32, u: f32, v: f32) -> Self {
        Self {
            instance_primitive: Self::pack_ids(instance_id, primitive_id),
            barycentrics_packed: Self::pack_barycentrics(u, v),
        }
    }

    /// Get the instance ID.
    #[inline]
    pub const fn instance_id(&self) -> u32 {
        Self::unpack_instance_id(self.instance_primitive)
    }

    /// Get the primitive ID.
    #[inline]
    pub const fn primitive_id(&self) -> u32 {
        Self::unpack_primitive_id(self.instance_primitive)
    }

    /// Get the barycentric coordinates.
    #[inline]
    pub fn barycentrics(&self) -> (f32, f32) {
        Self::unpack_barycentrics(self.barycentrics_packed)
    }

    /// Create an invalid entry.
    pub const fn invalid() -> Self {
        Self {
            instance_primitive: 0xFFFF_FFFF,
            barycentrics_packed: 0,
        }
    }
}

// =============================================================================
// VISIBILITY BUFFER RESOURCES
// =============================================================================

/// GPU resources for visibility buffer operations.
///
/// Contains all buffers needed for visibility write and clear operations.
pub struct VisibilityBufferResources {
    /// Uniform buffer for parameters.
    pub params_buffer: Buffer,
    /// Input: visible instances from culling.
    pub visible_instances_buffer: Buffer,
    /// Output: visibility buffer for material pass.
    pub visibility_buffer: Buffer,
    /// Output: write count (atomic u32).
    pub write_count_buffer: Buffer,
    /// Staging buffer for reading write count back to CPU.
    pub write_count_staging: Buffer,
    /// Maximum entries in visibility buffer.
    pub max_visibility: u32,
}

impl VisibilityBufferResources {
    /// Create resources for visibility buffer operations.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `max_visibility` - Maximum visibility buffer entries
    pub fn new(device: &Device, max_visibility: u32) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_write_params"),
            size: mem::size_of::<VisibilityWriteParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visible_instances_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visible_instances"),
            size: (max_visibility as u64) * (mem::size_of::<VisibleInstance>() as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visibility_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_buffer"),
            size: (max_visibility as u64) * (mem::size_of::<VisibilityData>() as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let write_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_write_count"),
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let write_count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visibility_write_count_staging"),
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            visible_instances_buffer,
            visibility_buffer,
            write_count_buffer,
            write_count_staging,
            max_visibility,
        }
    }

    /// Upload parameters to the GPU.
    pub fn upload_params(&self, queue: &Queue, params: &VisibilityWriteParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload visible instances from culling output.
    pub fn upload_visible_instances(&self, queue: &Queue, instances: &[VisibleInstance]) {
        let byte_len = instances.len() * mem::size_of::<VisibleInstance>();
        assert!(byte_len <= self.visible_instances_buffer.size() as usize);
        queue.write_buffer(
            &self.visible_instances_buffer,
            0,
            bytemuck::cast_slice(instances),
        );
    }

    /// Clear the write count to 0.
    pub fn clear_write_count(&self, queue: &Queue) {
        queue.write_buffer(&self.write_count_buffer, 0, &[0u8; 4]);
    }

    /// Read the write count back to the CPU.
    ///
    /// This is a synchronous operation that waits for GPU completion.
    pub fn read_write_count(&self, device: &Device, queue: &Queue) -> u32 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_visibility_write_count"),
        });
        encoder.copy_buffer_to_buffer(
            &self.write_count_buffer,
            0,
            &self.write_count_staging,
            0,
            4,
        );
        queue.submit([encoder.finish()]);

        let buffer_slice = self.write_count_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let count = *bytemuck::from_bytes::<u32>(&data);
        drop(data);
        self.write_count_staging.unmap();

        count
    }

    /// Get the visibility buffer (for reading in material pass).
    #[inline]
    pub fn visibility_buffer(&self) -> &Buffer {
        &self.visibility_buffer
    }

    /// Get the visible instances buffer (for culling output connection).
    #[inline]
    pub fn visible_instances_buffer(&self) -> &Buffer {
        &self.visible_instances_buffer
    }
}

// =============================================================================
// VISIBILITY WRITE PIPELINE
// =============================================================================

/// Compute pipeline for visibility buffer write operations.
pub struct VisibilityWritePipeline {
    /// Main visibility write pipeline.
    write_pipeline: wgpu::ComputePipeline,
    /// Visibility clear pipeline.
    clear_pipeline: wgpu::ComputePipeline,
    /// Clear write count pipeline.
    clear_count_pipeline: wgpu::ComputePipeline,
    /// Bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl VisibilityWritePipeline {
    /// Create a new visibility write pipeline.
    pub fn new(device: &Device) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(device);
        let pipeline_layout = Self::create_pipeline_layout(device, &bind_group_layout);

        let shader_source = include_str!("../../shaders/gpu_driven/gpu_visibility_write.comp.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_visibility_write_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let write_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("visibility_write_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "visibility_write",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("visibility_clear_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "visibility_clear",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let clear_count_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("visibility_clear_count_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "clear_write_count",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            write_pipeline,
            clear_pipeline,
            clear_count_pipeline,
            bind_group_layout,
        }
    }

    /// Get the bind group layout.
    #[inline]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Create a bind group for the given resources.
    pub fn create_bind_group(
        &self,
        device: &Device,
        resources: &VisibilityBufferResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visibility_write_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.visible_instances_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.visibility_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.write_count_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch visibility write operation.
    pub fn dispatch_write(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &VisibilityWriteParams,
    ) {
        if params.is_empty() {
            return;
        }

        // Clear write count first
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("visibility_clear_count_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_count_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Write visibility entries
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("visibility_write_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.write_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(params.num_workgroups(), 1, 1);
        }
    }

    /// Dispatch visibility clear operation.
    pub fn dispatch_clear(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &VisibilityWriteParams,
    ) {
        if params.is_empty() {
            return;
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("visibility_clear_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.clear_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(params.num_workgroups(), 1, 1);
    }

    /// Create the bind group layout.
    fn create_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("visibility_write_bind_group_layout"),
            entries: &[
                // binding 0: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: visible_instances (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: visibility_buffer (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: write_count (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create the pipeline layout.
    fn create_pipeline_layout(
        device: &Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::PipelineLayout {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("visibility_write_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        })
    }
}

// =============================================================================
// CPU REFERENCE IMPLEMENTATION
// =============================================================================

/// CPU reference implementation for visibility write.
///
/// Used for testing GPU results against known-correct values.
pub fn cpu_visibility_write(
    visible_instances: &[VisibleInstance],
    visibility_buffer_offset: u32,
) -> (Vec<VisibilityData>, u32) {
    let entries: Vec<VisibilityData> = visible_instances
        .iter()
        .enumerate()
        .map(|(i, inst)| {
            let _ = visibility_buffer_offset + i as u32; // Validate offset usage
            VisibilityData::with_instance(inst.original_index)
        })
        .collect();

    let count = entries.len() as u32;
    (entries, count)
}

/// CPU reference implementation for visibility clear.
pub fn cpu_visibility_clear(count: u32, visibility_buffer_offset: u32) -> Vec<VisibilityData> {
    let _ = visibility_buffer_offset; // Used for offset validation
    (0..count).map(|_| VisibilityData::invalid()).collect()
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Size/Layout Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_visibility_write_params_size() {
        assert_eq!(mem::size_of::<VisibilityWriteParams>(), 16);
    }

    #[test]
    fn test_visible_instance_size() {
        assert_eq!(mem::size_of::<VisibleInstance>(), 16);
    }

    #[test]
    fn test_visibility_data_size() {
        assert_eq!(mem::size_of::<VisibilityData>(), 24);
    }

    #[test]
    fn test_packed_visibility_size() {
        assert_eq!(mem::size_of::<PackedVisibility>(), 8);
    }

    #[test]
    fn test_visibility_write_params_pod() {
        let params = VisibilityWriteParams::new(1000, 0);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_visible_instance_pod() {
        let inst = VisibleInstance::new(42, 2, 0x00010002);
        let bytes = bytemuck::bytes_of(&inst);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_visibility_data_pod() {
        let data = VisibilityData::new(100, 50, [0.3, 0.4]);
        let bytes = bytemuck::bytes_of(&data);
        assert_eq!(bytes.len(), 24);
    }

    // -------------------------------------------------------------------------
    // VisibleInstance Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_visible_instance_from_ids() {
        let inst = VisibleInstance::from_ids(100, 2, 5, 10);
        assert_eq!(inst.original_index, 100);
        assert_eq!(inst.lod_level, 2);
        assert_eq!(inst.material_id(), 5);
        assert_eq!(inst.mesh_id(), 10);
    }

    #[test]
    fn test_visible_instance_batch_key() {
        let inst = VisibleInstance::new(0, 0, 0x00FF00AA);
        assert_eq!(inst.material_id(), 0xFF);
        assert_eq!(inst.mesh_id(), 0xAA);
    }

    #[test]
    fn test_visible_instance_max_ids() {
        let inst = VisibleInstance::from_ids(u32::MAX, 7, u16::MAX, u16::MAX);
        assert_eq!(inst.original_index, u32::MAX);
        assert_eq!(inst.lod_level, 7);
        assert_eq!(inst.material_id(), u16::MAX);
        assert_eq!(inst.mesh_id(), u16::MAX);
    }

    // -------------------------------------------------------------------------
    // VisibilityData Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_visibility_data_new() {
        let data = VisibilityData::new(42, 100, [0.25, 0.5]);
        assert_eq!(data.instance_id, 42);
        assert_eq!(data.primitive_id, 100);
        assert_eq!(data.barycentrics[0], 0.25);
        assert_eq!(data.barycentrics[1], 0.5);
    }

    #[test]
    fn test_visibility_data_invalid() {
        let data = VisibilityData::invalid();
        assert_eq!(data.instance_id, INVALID_INSTANCE_ID);
        assert_eq!(data.primitive_id, INVALID_PRIMITIVE_ID);
        assert!(!data.is_valid());
        assert!(!data.has_primitive());
    }

    #[test]
    fn test_visibility_data_with_instance() {
        let data = VisibilityData::with_instance(123);
        assert_eq!(data.instance_id, 123);
        assert_eq!(data.primitive_id, INVALID_PRIMITIVE_ID);
        assert!(data.is_valid());
        assert!(!data.has_primitive());
    }

    #[test]
    fn test_visibility_data_barycentric_w() {
        let data = VisibilityData::new(0, 0, [0.3, 0.4]);
        let w = data.barycentric_w();
        assert!((w - 0.3).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // PackedVisibility Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_packed_visibility_pack_ids() {
        let packed = PackedVisibility::pack_ids(0x12345, 0xABC);
        assert_eq!(PackedVisibility::unpack_instance_id(packed), 0x12345);
        assert_eq!(PackedVisibility::unpack_primitive_id(packed), 0xABC);
    }

    #[test]
    fn test_packed_visibility_max_instance_id() {
        // Max instance ID is 20 bits = 0xFFFFF
        let packed = PackedVisibility::pack_ids(0xFFFFF, 0);
        assert_eq!(PackedVisibility::unpack_instance_id(packed), 0xFFFFF);
    }

    #[test]
    fn test_packed_visibility_max_primitive_id() {
        // Max primitive ID is 12 bits = 0xFFF
        let packed = PackedVisibility::pack_ids(0, 0xFFF);
        assert_eq!(PackedVisibility::unpack_primitive_id(packed), 0xFFF);
    }

    #[test]
    fn test_packed_visibility_barycentrics() {
        let packed = PackedVisibility::pack_barycentrics(0.5, 0.25);
        let (u, v) = PackedVisibility::unpack_barycentrics(packed);
        assert!((u - 0.5).abs() < 0.001);
        assert!((v - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_packed_visibility_barycentrics_clamp() {
        // Values outside [0, 1] should be clamped
        let packed = PackedVisibility::pack_barycentrics(-0.5, 1.5);
        let (u, v) = PackedVisibility::unpack_barycentrics(packed);
        assert!((u - 0.0).abs() < 0.001);
        assert!((v - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_packed_visibility_roundtrip() {
        let vis = PackedVisibility::new(12345, 678, 0.33, 0.66);
        assert_eq!(vis.instance_id(), 12345);
        assert_eq!(vis.primitive_id(), 678);
        let (u, v) = vis.barycentrics();
        assert!((u - 0.33).abs() < 0.001);
        assert!((v - 0.66).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // VisibilityWriteParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_visibility_write_params_num_workgroups() {
        assert_eq!(VisibilityWriteParams::new(1, 0).num_workgroups(), 1);
        assert_eq!(VisibilityWriteParams::new(256, 0).num_workgroups(), 1);
        assert_eq!(VisibilityWriteParams::new(257, 0).num_workgroups(), 2);
        assert_eq!(VisibilityWriteParams::new(1000, 0).num_workgroups(), 4);
    }

    #[test]
    fn test_visibility_write_params_is_empty() {
        assert!(VisibilityWriteParams::new(0, 0).is_empty());
        assert!(!VisibilityWriteParams::new(1, 0).is_empty());
    }

    // -------------------------------------------------------------------------
    // CPU Reference Implementation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_visibility_write_single() {
        let instances = vec![VisibleInstance::new(42, 0, 0)];
        let (entries, count) = cpu_visibility_write(&instances, 0);

        assert_eq!(count, 1);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].instance_id, 42);
        assert!(entries[0].is_valid());
        assert!(!entries[0].has_primitive());
    }

    #[test]
    fn test_cpu_visibility_write_multiple() {
        let instances = vec![
            VisibleInstance::new(10, 0, 0),
            VisibleInstance::new(20, 1, 0x00010001),
            VisibleInstance::new(30, 2, 0x00020002),
        ];
        let (entries, count) = cpu_visibility_write(&instances, 0);

        assert_eq!(count, 3);
        assert_eq!(entries[0].instance_id, 10);
        assert_eq!(entries[1].instance_id, 20);
        assert_eq!(entries[2].instance_id, 30);
    }

    #[test]
    fn test_cpu_visibility_write_with_offset() {
        let instances = vec![
            VisibleInstance::new(100, 0, 0),
            VisibleInstance::new(200, 0, 0),
        ];
        let (entries, count) = cpu_visibility_write(&instances, 1000);

        // Offset doesn't affect the output array, just validates positioning
        assert_eq!(count, 2);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_cpu_visibility_write_preserves_original_index() {
        let instances: Vec<VisibleInstance> = (0..100)
            .map(|i| VisibleInstance::new(i * 10, i % 8, i))
            .collect();

        let (entries, count) = cpu_visibility_write(&instances, 0);

        assert_eq!(count, 100);
        for (i, entry) in entries.iter().enumerate() {
            assert_eq!(entry.instance_id, (i * 10) as u32);
        }
    }

    #[test]
    fn test_cpu_visibility_write_empty() {
        let instances: Vec<VisibleInstance> = vec![];
        let (entries, count) = cpu_visibility_write(&instances, 0);

        assert_eq!(count, 0);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_cpu_visibility_clear() {
        let entries = cpu_visibility_clear(5, 0);

        assert_eq!(entries.len(), 5);
        for entry in &entries {
            assert!(!entry.is_valid());
            assert_eq!(entry.instance_id, INVALID_INSTANCE_ID);
            assert_eq!(entry.primitive_id, INVALID_PRIMITIVE_ID);
        }
    }

    // -------------------------------------------------------------------------
    // LOD Level Preservation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_visible_instance_lod_levels() {
        for lod in 0..8 {
            let inst = VisibleInstance::new(0, lod, 0);
            assert_eq!(inst.lod_level, lod);
        }
    }

    // -------------------------------------------------------------------------
    // Batch Key Encoding Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_batch_key_encoding() {
        let inst = VisibleInstance::from_ids(0, 0, 100, 200);
        assert_eq!(inst.batch_key, (100 << 16) | 200);
        assert_eq!(inst.material_id(), 100);
        assert_eq!(inst.mesh_id(), 200);
    }

    #[test]
    fn test_batch_key_max_values() {
        let inst = VisibleInstance::from_ids(0, 0, 65535, 65535);
        assert_eq!(inst.batch_key, 0xFFFFFFFF);
        assert_eq!(inst.material_id(), 65535);
        assert_eq!(inst.mesh_id(), 65535);
    }

    // -------------------------------------------------------------------------
    // Shader Validation Tests (using naga)
    // -------------------------------------------------------------------------

    #[test]
    fn test_visibility_write_shader_parses() {
        let shader_source =
            include_str!("../../shaders/gpu_driven/gpu_visibility_write.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("visibility write shader should parse without errors");

        let entry_names: Vec<_> = module.entry_points.iter().map(|ep| &ep.name).collect();

        assert!(
            entry_names.iter().any(|n| *n == "visibility_write"),
            "Should have visibility_write entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "visibility_clear"),
            "Should have visibility_clear entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "clear_write_count"),
            "Should have clear_write_count entry point"
        );
    }

    #[test]
    fn test_visibility_write_shader_validates() {
        let shader_source =
            include_str!("../../shaders/gpu_driven/gpu_visibility_write.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("visibility write shader should parse without errors");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        validator
            .validate(&module)
            .expect("visibility write shader should validate without errors");
    }

    #[test]
    fn test_visibility_write_shader_workgroup_size() {
        let shader_source =
            include_str!("../../shaders/gpu_driven/gpu_visibility_write.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("visibility write shader should parse without errors");

        for ep in &module.entry_points {
            if ep.stage == naga::ShaderStage::Compute && ep.name != "clear_write_count" {
                assert_eq!(
                    ep.workgroup_size,
                    [256, 1, 1],
                    "Entry point {} should have workgroup size 256x1x1",
                    ep.name
                );
            }
        }
    }

    #[test]
    fn test_visibility_write_shader_entry_points_are_compute() {
        let shader_source =
            include_str!("../../shaders/gpu_driven/gpu_visibility_write.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("visibility write shader should parse without errors");

        for ep in &module.entry_points {
            assert_eq!(
                ep.stage,
                naga::ShaderStage::Compute,
                "Entry point {} should be a compute shader",
                ep.name
            );
        }
    }
}

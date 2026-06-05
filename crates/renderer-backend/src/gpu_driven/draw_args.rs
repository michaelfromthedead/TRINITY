//! GPU Draw Argument Generation for TRINITY Engine (T-GPU-3.3).
//!
//! This module generates indirect draw commands from a sorted visibility buffer.
//! After frustum culling and sorting by material/mesh ID, this shader detects
//! batch boundaries and emits `IndirectDrawIndexedArgs` for efficient rendering.
//!
//! # Overview
//!
//! The draw argument generation process:
//! 1. Input: Sorted visibility buffer (sorted by `batch_key = material_id << 16 | mesh_id`)
//! 2. Detection: Find batch boundaries where batch_key changes
//! 3. Output: One `IndirectDrawIndexedArgs` per unique batch
//!
//! # Performance
//!
//! - Work complexity: O(n) where n = visible instances
//! - Target: < 0.1ms for 100K instances
//! - Memory: 16 bytes per visibility entry, 20 bytes per draw command
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::{DrawArgsPipeline, DrawArgsResources};
//!
//! // Create pipeline and resources
//! let pipeline = DrawArgsPipeline::new(&device);
//! let resources = DrawArgsResources::new(&device, 100_000, 4096);
//!
//! // Each frame: generate draw commands from sorted visibility
//! let params = DrawArgsParams::new(visible_count, max_draws);
//! resources.upload_params(&queue, &params);
//! pipeline.dispatch(&mut encoder, &resources, &params);
//!
//! // Read back draw count
//! let draw_count = resources.read_draw_count(&device, &queue);
//! ```

use std::mem;

use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, BufferUsages, Device, Queue};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 256;

/// Maximum LOD levels supported.
pub const MAX_LOD_LEVELS: usize = 8;

/// Default maximum draws (should cover most scenes).
pub const DEFAULT_MAX_DRAWS: u32 = 4096;

// =============================================================================
// DRAW ARGS PARAMS
// =============================================================================

/// GPU uniform buffer for draw argument generation parameters.
///
/// Matches the WGSL `DrawArgsParams` struct layout.
///
/// # Memory Layout
///
/// 16 bytes, std140/std430 compatible:
/// | Offset | Field       | Size |
/// |--------|-------------|------|
/// | 0      | num_visible | 4    |
/// | 4      | max_draws   | 4    |
/// | 8      | _pad        | 8    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct DrawArgsParams {
    /// Number of visible instances in the visibility buffer.
    pub num_visible: u32,
    /// Maximum number of draw commands to generate.
    pub max_draws: u32,
    /// Padding for 16-byte alignment.
    pub _pad: [u32; 2],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<DrawArgsParams>() == 16);

impl DrawArgsParams {
    /// Create parameters for the given counts.
    pub fn new(num_visible: u32, max_draws: u32) -> Self {
        Self {
            num_visible,
            max_draws,
            _pad: [0, 0],
        }
    }

    /// Get the number of workgroups needed for dispatch.
    #[inline]
    pub fn num_workgroups(&self) -> u32 {
        (self.num_visible + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }

    /// Check if this can use single-workgroup optimization.
    #[inline]
    pub fn is_single_workgroup(&self) -> bool {
        self.num_visible <= WORKGROUP_SIZE
    }
}

// =============================================================================
// VISIBILITY ENTRY
// =============================================================================

/// Visibility buffer entry - output from culling, input to draw arg generation.
///
/// Must be sorted by `batch_key` before draw argument generation.
///
/// # Memory Layout
///
/// 16 bytes:
/// | Offset | Field       | Size |
/// |--------|-------------|------|
/// | 0      | instance_id | 4    |
/// | 4      | batch_key   | 4    |
/// | 8      | lod_level   | 4    |
/// | 12     | _pad        | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct VisibilityEntry {
    /// Original instance ID for transform lookup.
    pub instance_id: u32,
    /// Batch key: `(material_id << 16) | mesh_id`.
    /// Instances with the same batch_key are drawn together.
    pub batch_key: u32,
    /// LOD level selected during culling (0 = highest detail).
    pub lod_level: u32,
    /// Padding for alignment.
    pub _pad: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<VisibilityEntry>() == 16);

impl VisibilityEntry {
    /// Create a new visibility entry.
    pub const fn new(instance_id: u32, batch_key: u32, lod_level: u32) -> Self {
        Self {
            instance_id,
            batch_key,
            lod_level,
            _pad: 0,
        }
    }

    /// Create a visibility entry from material and mesh IDs.
    pub const fn from_ids(instance_id: u32, material_id: u16, mesh_id: u16, lod_level: u32) -> Self {
        Self {
            instance_id,
            batch_key: ((material_id as u32) << 16) | (mesh_id as u32),
            lod_level,
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

    /// Create a batch key from material and mesh IDs.
    #[inline]
    pub const fn make_batch_key(material_id: u16, mesh_id: u16) -> u32 {
        ((material_id as u32) << 16) | (mesh_id as u32)
    }
}

// =============================================================================
// MESH METADATA
// =============================================================================

/// Mesh metadata for looking up index counts per mesh/LOD.
///
/// # Memory Layout
///
/// 80 bytes (with 8 LOD levels):
/// | Offset | Field             | Size |
/// |--------|-------------------|------|
/// | 0      | index_count       | 4    |
/// | 4      | first_index       | 4    |
/// | 8      | base_vertex       | 4    |
/// | 12     | _pad              | 4    |
/// | 16     | lod_index_offsets | 32   |
/// | 48     | lod_index_counts  | 32   |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct MeshMetadata {
    /// Number of indices in the mesh (LOD 0).
    pub index_count: u32,
    /// Offset into the global index buffer.
    pub first_index: u32,
    /// Base vertex offset for this mesh.
    pub base_vertex: i32,
    /// Padding for vec4 alignment.
    pub _pad: u32,
    /// Index offset per LOD level (relative to first_index).
    pub lod_index_offsets: [u32; MAX_LOD_LEVELS],
    /// Index count per LOD level.
    pub lod_index_counts: [u32; MAX_LOD_LEVELS],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<MeshMetadata>() == 80);

impl MeshMetadata {
    /// Create mesh metadata without LOD support.
    pub const fn new(index_count: u32, first_index: u32, base_vertex: i32) -> Self {
        Self {
            index_count,
            first_index,
            base_vertex,
            _pad: 0,
            lod_index_offsets: [0; MAX_LOD_LEVELS],
            lod_index_counts: [0; MAX_LOD_LEVELS],
        }
    }

    /// Create mesh metadata with LOD information.
    pub fn with_lods(
        index_count: u32,
        first_index: u32,
        base_vertex: i32,
        lod_offsets: &[u32],
        lod_counts: &[u32],
    ) -> Self {
        let mut meta = Self::new(index_count, first_index, base_vertex);

        for (i, &offset) in lod_offsets.iter().take(MAX_LOD_LEVELS).enumerate() {
            meta.lod_index_offsets[i] = offset;
        }
        for (i, &count) in lod_counts.iter().take(MAX_LOD_LEVELS).enumerate() {
            meta.lod_index_counts[i] = count;
        }

        meta
    }

    /// Get the index count for a specific LOD level.
    pub fn index_count_for_lod(&self, lod: usize) -> u32 {
        if lod < MAX_LOD_LEVELS && self.lod_index_counts[lod] > 0 {
            self.lod_index_counts[lod]
        } else {
            self.index_count
        }
    }

    /// Get the first index for a specific LOD level.
    pub fn first_index_for_lod(&self, lod: usize) -> u32 {
        if lod < MAX_LOD_LEVELS && self.lod_index_offsets[lod] > 0 {
            self.first_index + self.lod_index_offsets[lod]
        } else {
            self.first_index
        }
    }
}

// =============================================================================
// DRAW ARGS RESOURCES
// =============================================================================

/// GPU resources for draw argument generation.
///
/// Contains all buffers needed for the draw argument generation algorithm.
pub struct DrawArgsResources {
    /// Uniform buffer for parameters.
    pub params_buffer: Buffer,
    /// Visibility buffer (sorted by batch_key).
    pub visibility_buffer: Buffer,
    /// Mesh metadata table.
    pub mesh_metadata_buffer: Buffer,
    /// Output draw commands buffer.
    pub draw_args_buffer: Buffer,
    /// Output draw count (atomic u32).
    pub draw_count_buffer: Buffer,
    /// Staging buffer for reading draw count back to CPU.
    pub draw_count_staging: Buffer,
    /// Maximum visibility entries supported.
    pub max_visibility: u32,
    /// Maximum draw commands supported.
    pub max_draws: u32,
    /// Maximum meshes in metadata table.
    pub max_meshes: u32,
}

impl DrawArgsResources {
    /// Create resources for draw argument generation.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device
    /// * `max_visibility` - Maximum visible instances
    /// * `max_draws` - Maximum draw commands to generate
    /// * `max_meshes` - Maximum meshes in the metadata table
    pub fn new(device: &Device, max_visibility: u32, max_draws: u32, max_meshes: u32) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw_args_params"),
            size: mem::size_of::<DrawArgsParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let visibility_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw_args_visibility"),
            size: (max_visibility as u64) * (mem::size_of::<VisibilityEntry>() as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mesh_metadata_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw_args_mesh_metadata"),
            size: (max_meshes as u64) * (mem::size_of::<MeshMetadata>() as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // IndirectDrawIndexedArgs is 20 bytes, but we pad to 32 for alignment
        let draw_args_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw_args_output"),
            size: (max_draws as u64) * 20, // 20 bytes per IndirectDrawIndexedArgs
            usage: BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let draw_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw_args_count"),
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::INDIRECT | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let draw_count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("draw_args_count_staging"),
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            visibility_buffer,
            mesh_metadata_buffer,
            draw_args_buffer,
            draw_count_buffer,
            draw_count_staging,
            max_visibility,
            max_draws,
            max_meshes,
        }
    }

    /// Upload parameters to the GPU.
    pub fn upload_params(&self, queue: &Queue, params: &DrawArgsParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Upload sorted visibility entries to the GPU.
    pub fn upload_visibility(&self, queue: &Queue, entries: &[VisibilityEntry]) {
        let byte_len = entries.len() * mem::size_of::<VisibilityEntry>();
        assert!(byte_len <= self.visibility_buffer.size() as usize);
        queue.write_buffer(&self.visibility_buffer, 0, bytemuck::cast_slice(entries));
    }

    /// Upload mesh metadata to the GPU.
    pub fn upload_mesh_metadata(&self, queue: &Queue, metadata: &[MeshMetadata]) {
        let byte_len = metadata.len() * mem::size_of::<MeshMetadata>();
        assert!(byte_len <= self.mesh_metadata_buffer.size() as usize);
        queue.write_buffer(&self.mesh_metadata_buffer, 0, bytemuck::cast_slice(metadata));
    }

    /// Clear the draw count to 0.
    pub fn clear_draw_count(&self, queue: &Queue) {
        queue.write_buffer(&self.draw_count_buffer, 0, &[0u8; 4]);
    }

    /// Read the draw count back to the CPU.
    ///
    /// This is a synchronous operation that waits for GPU completion.
    pub fn read_draw_count(&self, device: &Device, queue: &Queue) -> u32 {
        // Copy from GPU buffer to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_draw_count"),
        });
        encoder.copy_buffer_to_buffer(&self.draw_count_buffer, 0, &self.draw_count_staging, 0, 4);
        queue.submit([encoder.finish()]);

        // Map staging buffer and read
        let buffer_slice = self.draw_count_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let count = *bytemuck::from_bytes::<u32>(&data);
        drop(data);
        self.draw_count_staging.unmap();

        count
    }

    /// Get the visibility buffer.
    #[inline]
    pub fn visibility_buffer(&self) -> &Buffer {
        &self.visibility_buffer
    }

    /// Get the draw args buffer (for indirect draw calls).
    #[inline]
    pub fn draw_args_buffer(&self) -> &Buffer {
        &self.draw_args_buffer
    }

    /// Get the draw count buffer (for indirect count).
    #[inline]
    pub fn draw_count_buffer(&self) -> &Buffer {
        &self.draw_count_buffer
    }
}

// =============================================================================
// DRAW ARGS PIPELINE
// =============================================================================

/// Compute pipeline for draw argument generation.
pub struct DrawArgsPipeline {
    /// Main draw argument generation pipeline.
    gen_pipeline: wgpu::ComputePipeline,
    /// Single-workgroup optimized pipeline.
    gen_single_pipeline: wgpu::ComputePipeline,
    /// Clear draw count pipeline.
    clear_pipeline: wgpu::ComputePipeline,
    /// Bind group layout.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl DrawArgsPipeline {
    /// Create a new draw argument generation pipeline.
    pub fn new(device: &Device) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(device);
        let pipeline_layout = Self::create_pipeline_layout(device, &bind_group_layout);

        let shader_source = include_str!("../../shaders/gpu_driven/gpu_gen_draw_args.comp.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_gen_draw_args_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let gen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("draw_args_gen_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "gen_draw_args",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let gen_single_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("draw_args_gen_single_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "gen_draw_args_single_workgroup",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("draw_args_clear_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "clear_draw_count",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            gen_pipeline,
            gen_single_pipeline,
            clear_pipeline,
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
        resources: &DrawArgsResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw_args_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.visibility_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resources.mesh_metadata_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: resources.draw_args_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: resources.draw_count_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch draw argument generation.
    ///
    /// Automatically selects single-workgroup or multi-workgroup variant.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        params: &DrawArgsParams,
    ) {
        // First clear the draw count
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("draw_args_clear_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.clear_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Then generate draw args
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("draw_args_gen_pass"),
                timestamp_writes: None,
            });

            if params.is_single_workgroup() {
                pass.set_pipeline(&self.gen_single_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            } else {
                pass.set_pipeline(&self.gen_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(params.num_workgroups(), 1, 1);
            }
        }
    }

    /// Create the bind group layout.
    fn create_bind_group_layout(device: &Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("draw_args_bind_group_layout"),
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
                // binding 1: visibility (storage, read)
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
                // binding 2: mesh_metadata (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: draw_args (storage, read_write)
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
                // binding 4: draw_count (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
            label: Some("draw_args_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        })
    }
}

// =============================================================================
// CPU REFERENCE IMPLEMENTATION
// =============================================================================

use super::IndirectDrawIndexedArgs;

/// CPU reference implementation for draw argument generation.
///
/// Used for testing GPU results against known-correct values.
pub fn cpu_gen_draw_args(
    visibility: &[VisibilityEntry],
    mesh_metadata: &[MeshMetadata],
    max_draws: u32,
) -> (Vec<IndirectDrawIndexedArgs>, u32) {
    if visibility.is_empty() {
        return (Vec::new(), 0);
    }

    let mut draw_args = Vec::new();
    let mut i = 0;

    while i < visibility.len() && (draw_args.len() as u32) < max_draws {
        let start_idx = i;
        let batch_key = visibility[i].batch_key;
        let lod = visibility[i].lod_level as usize;

        // Count instances in this batch
        let mut instance_count = 1u32;
        i += 1;
        while i < visibility.len() && visibility[i].batch_key == batch_key {
            instance_count += 1;
            i += 1;
        }

        // Get mesh info
        let mesh_id = (batch_key & 0xFFFF) as usize;
        if mesh_id < mesh_metadata.len() {
            let mesh = &mesh_metadata[mesh_id];
            let index_count = mesh.index_count_for_lod(lod);
            let first_index = mesh.first_index_for_lod(lod);

            draw_args.push(IndirectDrawIndexedArgs::new(
                index_count,
                instance_count,
                first_index,
                mesh.base_vertex,
                start_idx as u32,
            ));
        }
    }

    let count = draw_args.len() as u32;
    (draw_args, count)
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
    fn test_draw_args_params_size() {
        assert_eq!(mem::size_of::<DrawArgsParams>(), 16);
    }

    #[test]
    fn test_visibility_entry_size() {
        assert_eq!(mem::size_of::<VisibilityEntry>(), 16);
    }

    #[test]
    fn test_mesh_metadata_size() {
        assert_eq!(mem::size_of::<MeshMetadata>(), 80);
    }

    #[test]
    fn test_draw_args_params_pod() {
        let params = DrawArgsParams::new(1000, 100);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_visibility_entry_pod() {
        let entry = VisibilityEntry::new(42, 0x00010002, 1);
        let bytes = bytemuck::bytes_of(&entry);
        assert_eq!(bytes.len(), 16);
    }

    // -------------------------------------------------------------------------
    // VisibilityEntry Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_visibility_entry_from_ids() {
        let entry = VisibilityEntry::from_ids(100, 5, 10, 2);
        assert_eq!(entry.instance_id, 100);
        assert_eq!(entry.material_id(), 5);
        assert_eq!(entry.mesh_id(), 10);
        assert_eq!(entry.lod_level, 2);
    }

    #[test]
    fn test_visibility_entry_batch_key() {
        let key = VisibilityEntry::make_batch_key(255, 1000);
        assert_eq!(key, (255 << 16) | 1000);

        let entry = VisibilityEntry::new(0, key, 0);
        assert_eq!(entry.material_id(), 255);
        assert_eq!(entry.mesh_id(), 1000);
    }

    #[test]
    fn test_visibility_entry_max_ids() {
        let entry = VisibilityEntry::from_ids(u32::MAX, u16::MAX, u16::MAX, 7);
        assert_eq!(entry.instance_id, u32::MAX);
        assert_eq!(entry.material_id(), u16::MAX);
        assert_eq!(entry.mesh_id(), u16::MAX);
    }

    // -------------------------------------------------------------------------
    // MeshMetadata Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mesh_metadata_new() {
        let meta = MeshMetadata::new(36, 100, -50);
        assert_eq!(meta.index_count, 36);
        assert_eq!(meta.first_index, 100);
        assert_eq!(meta.base_vertex, -50);
    }

    #[test]
    fn test_mesh_metadata_with_lods() {
        let meta = MeshMetadata::with_lods(
            1000,  // LOD0 index count
            0,     // first index
            0,     // base vertex
            &[0, 1000, 1500, 1800],  // offsets
            &[1000, 500, 300, 100],  // counts
        );

        assert_eq!(meta.index_count_for_lod(0), 1000);
        assert_eq!(meta.index_count_for_lod(1), 500);
        assert_eq!(meta.index_count_for_lod(2), 300);
        assert_eq!(meta.index_count_for_lod(3), 100);

        assert_eq!(meta.first_index_for_lod(0), 0);
        assert_eq!(meta.first_index_for_lod(1), 1000);
        assert_eq!(meta.first_index_for_lod(2), 1500);
        assert_eq!(meta.first_index_for_lod(3), 1800);
    }

    #[test]
    fn test_mesh_metadata_fallback() {
        let meta = MeshMetadata::new(36, 100, 0);
        // No LOD data - should fall back to base values
        assert_eq!(meta.index_count_for_lod(0), 36);
        assert_eq!(meta.index_count_for_lod(5), 36);
        assert_eq!(meta.first_index_for_lod(0), 100);
        assert_eq!(meta.first_index_for_lod(5), 100);
    }

    // -------------------------------------------------------------------------
    // DrawArgsParams Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_args_params_num_workgroups() {
        assert_eq!(DrawArgsParams::new(1, 100).num_workgroups(), 1);
        assert_eq!(DrawArgsParams::new(256, 100).num_workgroups(), 1);
        assert_eq!(DrawArgsParams::new(257, 100).num_workgroups(), 2);
        assert_eq!(DrawArgsParams::new(1000, 100).num_workgroups(), 4);
    }

    #[test]
    fn test_draw_args_params_is_single_workgroup() {
        assert!(DrawArgsParams::new(1, 100).is_single_workgroup());
        assert!(DrawArgsParams::new(256, 100).is_single_workgroup());
        assert!(!DrawArgsParams::new(257, 100).is_single_workgroup());
    }

    // -------------------------------------------------------------------------
    // CPU Reference Implementation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_single_batch_one_draw() {
        let visibility = vec![
            VisibilityEntry::from_ids(0, 1, 0, 0),
            VisibilityEntry::from_ids(1, 1, 0, 0),
            VisibilityEntry::from_ids(2, 1, 0, 0),
        ];
        let mesh_metadata = vec![MeshMetadata::new(36, 0, 0)];

        let (draws, count) = cpu_gen_draw_args(&visibility, &mesh_metadata, 100);

        assert_eq!(count, 1);
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].index_count, 36);
        assert_eq!(draws[0].instance_count, 3);
        assert_eq!(draws[0].first_instance, 0);
    }

    #[test]
    fn test_multiple_batches_multiple_draws() {
        let visibility = vec![
            // Batch 1: material 0, mesh 0
            VisibilityEntry::from_ids(0, 0, 0, 0),
            VisibilityEntry::from_ids(1, 0, 0, 0),
            // Batch 2: material 1, mesh 0
            VisibilityEntry::from_ids(2, 1, 0, 0),
            // Batch 3: material 1, mesh 1
            VisibilityEntry::from_ids(3, 1, 1, 0),
            VisibilityEntry::from_ids(4, 1, 1, 0),
            VisibilityEntry::from_ids(5, 1, 1, 0),
        ];
        let mesh_metadata = vec![
            MeshMetadata::new(36, 0, 0),   // mesh 0
            MeshMetadata::new(24, 100, 0), // mesh 1
        ];

        let (draws, count) = cpu_gen_draw_args(&visibility, &mesh_metadata, 100);

        assert_eq!(count, 3);
        assert_eq!(draws.len(), 3);

        // Batch 1
        assert_eq!(draws[0].instance_count, 2);
        assert_eq!(draws[0].index_count, 36);
        assert_eq!(draws[0].first_instance, 0);

        // Batch 2
        assert_eq!(draws[1].instance_count, 1);
        assert_eq!(draws[1].index_count, 36);
        assert_eq!(draws[1].first_instance, 2);

        // Batch 3
        assert_eq!(draws[2].instance_count, 3);
        assert_eq!(draws[2].index_count, 24);
        assert_eq!(draws[2].first_index, 100);
        assert_eq!(draws[2].first_instance, 3);
    }

    #[test]
    fn test_sorted_input_correct_boundaries() {
        // Already sorted by batch_key
        let visibility = vec![
            VisibilityEntry::new(0, 0x00000000, 0), // mat=0, mesh=0
            VisibilityEntry::new(1, 0x00000001, 0), // mat=0, mesh=1
            VisibilityEntry::new(2, 0x00010000, 0), // mat=1, mesh=0
            VisibilityEntry::new(3, 0x00010001, 0), // mat=1, mesh=1
        ];
        let mesh_metadata = vec![
            MeshMetadata::new(100, 0, 0),
            MeshMetadata::new(200, 100, 0),
        ];

        let (draws, count) = cpu_gen_draw_args(&visibility, &mesh_metadata, 100);

        assert_eq!(count, 4);
        for draw in &draws {
            assert_eq!(draw.instance_count, 1);
        }
    }

    #[test]
    fn test_instance_count_per_batch() {
        let visibility = vec![
            // 5 instances of same batch
            VisibilityEntry::from_ids(0, 0, 0, 0),
            VisibilityEntry::from_ids(1, 0, 0, 0),
            VisibilityEntry::from_ids(2, 0, 0, 0),
            VisibilityEntry::from_ids(3, 0, 0, 0),
            VisibilityEntry::from_ids(4, 0, 0, 0),
            // 3 instances of different batch
            VisibilityEntry::from_ids(5, 1, 0, 0),
            VisibilityEntry::from_ids(6, 1, 0, 0),
            VisibilityEntry::from_ids(7, 1, 0, 0),
        ];
        let mesh_metadata = vec![MeshMetadata::new(36, 0, 0)];

        let (draws, count) = cpu_gen_draw_args(&visibility, &mesh_metadata, 100);

        assert_eq!(count, 2);
        assert_eq!(draws[0].instance_count, 5);
        assert_eq!(draws[1].instance_count, 3);
    }

    #[test]
    fn test_first_instance_offset() {
        let visibility = vec![
            VisibilityEntry::from_ids(0, 0, 0, 0),
            VisibilityEntry::from_ids(1, 0, 0, 0),
            VisibilityEntry::from_ids(2, 1, 0, 0),
            VisibilityEntry::from_ids(3, 1, 0, 0),
            VisibilityEntry::from_ids(4, 1, 0, 0),
            VisibilityEntry::from_ids(5, 2, 0, 0),
        ];
        let mesh_metadata = vec![MeshMetadata::new(36, 0, 0)];

        let (draws, count) = cpu_gen_draw_args(&visibility, &mesh_metadata, 100);

        assert_eq!(count, 3);
        assert_eq!(draws[0].first_instance, 0);
        assert_eq!(draws[1].first_instance, 2);
        assert_eq!(draws[2].first_instance, 5);
    }

    #[test]
    fn test_max_draws_limit() {
        // 10 different batches
        let visibility: Vec<_> = (0..10)
            .map(|i| VisibilityEntry::from_ids(i, i as u16, 0, 0))
            .collect();
        let mesh_metadata = vec![MeshMetadata::new(36, 0, 0)];

        // Limit to 5 draws
        let (draws, count) = cpu_gen_draw_args(&visibility, &mesh_metadata, 5);

        assert_eq!(count, 5);
        assert_eq!(draws.len(), 5);
    }

    #[test]
    fn test_empty_visibility_no_draws() {
        let visibility: Vec<VisibilityEntry> = vec![];
        let mesh_metadata = vec![MeshMetadata::new(36, 0, 0)];

        let (draws, count) = cpu_gen_draw_args(&visibility, &mesh_metadata, 100);

        assert_eq!(count, 0);
        assert!(draws.is_empty());
    }

    #[test]
    fn test_lod_levels() {
        let visibility = vec![
            VisibilityEntry::from_ids(0, 0, 0, 0), // LOD 0
            VisibilityEntry::from_ids(1, 0, 1, 0), // LOD 0, different mesh
            VisibilityEntry::from_ids(2, 1, 0, 2), // LOD 2
        ];
        let mesh_metadata = vec![
            MeshMetadata::with_lods(1000, 0, 0, &[0, 1000, 1500], &[1000, 500, 200]),
            MeshMetadata::with_lods(500, 2000, 0, &[0, 500], &[500, 250]),
        ];

        let (draws, count) = cpu_gen_draw_args(&visibility, &mesh_metadata, 100);

        assert_eq!(count, 3);
        // LOD 0 mesh 0
        assert_eq!(draws[0].index_count, 1000);
        assert_eq!(draws[0].first_index, 0);
        // LOD 0 mesh 1
        assert_eq!(draws[1].index_count, 500);
        assert_eq!(draws[1].first_index, 2000);
        // LOD 2 mesh 0
        assert_eq!(draws[2].index_count, 200);
        assert_eq!(draws[2].first_index, 1500);
    }

    #[test]
    fn test_large_batch() {
        // 1000 instances in single batch
        let visibility: Vec<_> = (0..1000)
            .map(|i| VisibilityEntry::from_ids(i, 0, 0, 0))
            .collect();
        let mesh_metadata = vec![MeshMetadata::new(36, 0, 0)];

        let (draws, count) = cpu_gen_draw_args(&visibility, &mesh_metadata, 100);

        assert_eq!(count, 1);
        assert_eq!(draws[0].instance_count, 1000);
    }

    // -------------------------------------------------------------------------
    // Shader Validation Tests (using naga)
    // -------------------------------------------------------------------------

    #[test]
    fn test_draw_args_shader_parses() {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_gen_draw_args.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("draw args shader should parse without errors");

        // Verify expected entry points exist
        let entry_names: Vec<_> = module.entry_points.iter().map(|ep| &ep.name).collect();

        assert!(
            entry_names.iter().any(|n| *n == "gen_draw_args"),
            "Should have gen_draw_args entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "gen_draw_args_single_workgroup"),
            "Should have gen_draw_args_single_workgroup entry point"
        );
        assert!(
            entry_names.iter().any(|n| *n == "clear_draw_count"),
            "Should have clear_draw_count entry point"
        );
    }

    #[test]
    fn test_draw_args_shader_validates() {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_gen_draw_args.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("draw args shader should parse without errors");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        validator
            .validate(&module)
            .expect("draw args shader should validate without errors");
    }

    #[test]
    fn test_draw_args_shader_workgroup_size() {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_gen_draw_args.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("draw args shader should parse without errors");

        // Verify main compute entry points have 256x1x1 workgroup size
        for ep in &module.entry_points {
            if ep.stage == naga::ShaderStage::Compute && ep.name != "clear_draw_count" {
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
    fn test_draw_args_shader_entry_points_are_compute() {
        let shader_source = include_str!("../../shaders/gpu_driven/gpu_gen_draw_args.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("draw args shader should parse without errors");

        // Verify all entry points are compute shaders
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

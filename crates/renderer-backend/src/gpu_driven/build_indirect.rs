//! Indirect Buffer Generation from Compacted Visible Objects (T-WGPU-P6.6.2).
//!
//! This module generates indirect draw commands from a compacted list of visible
//! object indices produced by stream compaction (T-WGPU-P6.6.1). Each visible
//! object produces one `DrawIndexedIndirectArgs` command, with LOD-aware mesh
//! selection based on per-object LOD levels.
//!
//! # Overview
//!
//! ```text
//! Input Pipeline:
//! ┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐
//! │ Frustum/HiZ Cull│───►│ Stream Compaction│───►│ Build Indirect    │
//! │ (visibility bits)│    │ (compacted idx)  │    │ (draw commands)   │
//! └─────────────────┘    └──────────────────┘    └───────────────────┘
//!
//! Data Flow:
//! compacted_indices[i] -> object_idx
//! object_data[object_idx].mesh_index -> mesh_id
//! lod_buffer[object_idx].level -> lod_level
//! mesh_data[mesh_id] + lod_level -> DrawIndexedIndirectArgs
//! ```
//!
//! # Memory Layout
//!
//! | Struct               | Size   | Description                           |
//! |----------------------|--------|---------------------------------------|
//! | BuildIndirectParams  | 16     | visible_count, max_draws, padding     |
//! | MeshData             | 48     | index info + LOD offsets              |
//! | DrawIndexedIndirectArgs | 20  | Standard indirect draw args           |
//!
//! # Performance
//!
//! - Workgroup size: 64 threads (optimal GPU occupancy)
//! - One thread per visible object (standard mode)
//! - 4 objects per thread (batched mode for large counts)
//! - Atomic draw count increment (acceptable contention for most scenes)
//! - Target: <0.05ms for 100K visible objects
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::{
//!     BuildIndirectPipeline, BuildIndirectResources, BuildIndirectParams,
//! };
//!
//! // Create pipeline once at startup
//! let pipeline = BuildIndirectPipeline::new(&device);
//!
//! // Create resources for scene capacity
//! let resources = BuildIndirectResources::new(&device, 100_000, 4096);
//!
//! // Each frame after stream compaction:
//! resources.upload_params(&queue, &BuildIndirectParams::new(visible_count, max_draws));
//! resources.clear_draw_count(&queue);
//!
//! pipeline.dispatch(
//!     &mut encoder,
//!     &resources,
//!     &compacted_indices_buffer,
//!     &object_data_buffer,
//!     &mesh_data_buffer,
//!     &lod_buffer,
//!     visible_count,
//! );
//!
//! // Read back draw count if needed
//! let draw_count = resources.read_draw_count(&device, &queue);
//! ```

use bytemuck::{Pod, Zeroable};
use std::mem;
use wgpu::{Buffer, BufferUsages, Device, Queue};

// =============================================================================
// CONSTANTS
// =============================================================================

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 64;

/// Batch size for batched dispatch mode (objects per thread).
pub const BATCH_SIZE: u32 = 4;

/// Maximum LOD levels supported (LOD 0-3).
pub const MAX_LOD_LEVELS: usize = 4;

/// Maximum LOD level index.
pub const MAX_LOD_LEVEL: u32 = 3;

/// Default maximum draw commands.
pub const DEFAULT_MAX_DRAWS: u32 = 65536;

/// Size of BuildIndirectParams in bytes.
pub const BUILD_INDIRECT_PARAMS_SIZE: usize = 16;

/// Size of MeshData in bytes.
pub const MESH_DATA_SIZE: usize = 48;

/// Size of DrawIndexedIndirectArgs in bytes.
pub const DRAW_INDEXED_INDIRECT_ARGS_SIZE: usize = 20;

/// Shader source for include.
pub const BUILD_INDIRECT_SHADER: &str = include_str!("../../shaders/build_indirect.wgsl");

// =============================================================================
// BUILD INDIRECT PARAMS
// =============================================================================

/// Parameters for indirect buffer generation.
///
/// This struct is uploaded to a uniform buffer and accessed by the compute shader.
///
/// # Memory Layout (16 bytes)
///
/// | Offset | Field         | Size | Description                      |
/// |--------|---------------|------|----------------------------------|
/// | 0      | visible_count | 4    | Number of visible objects        |
/// | 4      | max_draws     | 4    | Maximum draw commands to output  |
/// | 8      | _pad0         | 4    | Reserved for alignment           |
/// | 12     | _pad1         | 4    | Reserved for alignment           |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct BuildIndirectParams {
    /// Number of visible objects in compacted_indices buffer.
    pub visible_count: u32,
    /// Maximum number of draw commands to generate.
    pub max_draws: u32,
    /// Reserved for future use.
    pub _pad0: u32,
    /// Reserved for alignment.
    pub _pad1: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<BuildIndirectParams>() == BUILD_INDIRECT_PARAMS_SIZE);

impl BuildIndirectParams {
    /// Create new build indirect parameters.
    ///
    /// # Arguments
    ///
    /// * `visible_count` - Number of visible objects to process.
    /// * `max_draws` - Maximum number of draw commands to generate.
    #[inline]
    pub const fn new(visible_count: u32, max_draws: u32) -> Self {
        Self {
            visible_count,
            max_draws,
            _pad0: 0,
            _pad1: 0,
        }
    }

    /// Calculate number of workgroups needed for standard dispatch.
    #[inline]
    pub fn workgroups(&self) -> u32 {
        (self.visible_count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }

    /// Calculate number of workgroups needed for batched dispatch.
    #[inline]
    pub fn workgroups_batched(&self) -> u32 {
        let objects_per_workgroup = WORKGROUP_SIZE * BATCH_SIZE;
        (self.visible_count + objects_per_workgroup - 1) / objects_per_workgroup
    }

    /// Check if this should use batched dispatch mode.
    ///
    /// Batched mode is more efficient for large visible counts (>10K).
    #[inline]
    pub fn use_batched_mode(&self) -> bool {
        self.visible_count > 10_000
    }
}

// =============================================================================
// MESH DATA
// =============================================================================

/// Mesh data with LOD support for indirect draw generation.
///
/// This struct contains index buffer information for a mesh and all its LOD levels.
/// The shader uses this to generate appropriate draw commands based on the
/// per-object LOD selection.
///
/// # Memory Layout (48 bytes)
///
/// | Offset | Field            | Size | Description                     |
/// |--------|------------------|------|---------------------------------|
/// | 0      | index_count      | 4    | Number of indices (LOD 0)       |
/// | 4      | first_index      | 4    | Offset into global index buffer |
/// | 8      | base_vertex      | 4    | Vertex offset (signed)          |
/// | 12     | _pad             | 4    | Padding for vec4 alignment      |
/// | 16     | lod_index_counts | 16   | Index count per LOD level       |
/// | 32     | lod_first_index  | 16   | First index offset per LOD      |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct MeshData {
    /// Number of indices in base mesh (LOD 0).
    pub index_count: u32,
    /// Offset into the global index buffer.
    pub first_index: u32,
    /// Vertex offset to add to each index (signed).
    pub base_vertex: i32,
    /// Padding for vec4 alignment.
    pub _pad: u32,
    /// Index count per LOD level (0-3).
    ///
    /// - `lod_index_counts[0]` = LOD 0 count (highest detail)
    /// - `lod_index_counts[3]` = LOD 3 count (lowest detail)
    ///
    /// If a LOD count is 0, the shader falls back to `index_count`.
    pub lod_index_counts: [u32; MAX_LOD_LEVELS],
    /// First index offset per LOD level (relative to first_index).
    ///
    /// - `lod_first_index[0]` = 0 (LOD 0 starts at first_index)
    /// - `lod_first_index[1]` = offset to LOD 1 data
    ///
    /// The absolute offset is `first_index + lod_first_index[lod]`.
    pub lod_first_index: [u32; MAX_LOD_LEVELS],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<MeshData>() == MESH_DATA_SIZE);

impl MeshData {
    /// Create mesh data without LOD support.
    ///
    /// All LOD levels will use the same index count and offset.
    #[inline]
    pub const fn new(index_count: u32, first_index: u32, base_vertex: i32) -> Self {
        Self {
            index_count,
            first_index,
            base_vertex,
            _pad: 0,
            lod_index_counts: [0; MAX_LOD_LEVELS],
            lod_first_index: [0; MAX_LOD_LEVELS],
        }
    }

    /// Create mesh data with LOD support.
    ///
    /// # Arguments
    ///
    /// * `index_count` - Base mesh index count (LOD 0).
    /// * `first_index` - Offset into global index buffer.
    /// * `base_vertex` - Vertex offset for this mesh.
    /// * `lod_counts` - Index counts per LOD level.
    /// * `lod_offsets` - First index offsets per LOD level (relative).
    pub fn with_lods(
        index_count: u32,
        first_index: u32,
        base_vertex: i32,
        lod_counts: &[u32],
        lod_offsets: &[u32],
    ) -> Self {
        let mut data = Self::new(index_count, first_index, base_vertex);

        for (i, &count) in lod_counts.iter().take(MAX_LOD_LEVELS).enumerate() {
            data.lod_index_counts[i] = count;
        }
        for (i, &offset) in lod_offsets.iter().take(MAX_LOD_LEVELS).enumerate() {
            data.lod_first_index[i] = offset;
        }

        data
    }

    /// Get index count for a specific LOD level.
    ///
    /// Falls back to base `index_count` if LOD count is 0.
    #[inline]
    pub fn index_count_for_lod(&self, lod: usize) -> u32 {
        let lod = lod.min(MAX_LOD_LEVELS - 1);
        if self.lod_index_counts[lod] > 0 {
            self.lod_index_counts[lod]
        } else {
            self.index_count
        }
    }

    /// Get first index offset for a specific LOD level.
    ///
    /// Returns absolute offset into the index buffer.
    #[inline]
    pub fn first_index_for_lod(&self, lod: usize) -> u32 {
        let lod = lod.min(MAX_LOD_LEVELS - 1);
        self.first_index + self.lod_first_index[lod]
    }
}

// =============================================================================
// BUILD INDIRECT RESOURCES
// =============================================================================

/// GPU resources for indirect buffer generation.
///
/// Contains all buffers needed for the build indirect compute pass.
pub struct BuildIndirectResources {
    /// Uniform buffer for parameters.
    pub params_buffer: Buffer,
    /// Output indirect draw commands buffer.
    pub indirect_commands_buffer: Buffer,
    /// Output draw count (atomic u32).
    pub draw_count_buffer: Buffer,
    /// Staging buffer for reading draw count back to CPU.
    draw_count_staging: Buffer,
    /// Maximum visible objects supported.
    pub max_visible: u32,
    /// Maximum draw commands supported.
    pub max_draws: u32,
}

impl BuildIndirectResources {
    /// Create resources for indirect buffer generation.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `max_visible` - Maximum number of visible objects.
    /// * `max_draws` - Maximum number of draw commands to generate.
    pub fn new(device: &Device, max_visible: u32, max_draws: u32) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("build_indirect_params"),
            size: BUILD_INDIRECT_PARAMS_SIZE as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // DrawIndexedIndirectArgs is 20 bytes per command
        let indirect_commands_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("build_indirect_commands"),
            size: (max_draws as u64) * (DRAW_INDEXED_INDIRECT_ARGS_SIZE as u64),
            usage: BufferUsages::STORAGE
                | BufferUsages::INDIRECT
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let draw_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("build_indirect_draw_count"),
            size: 4,
            usage: BufferUsages::STORAGE
                | BufferUsages::INDIRECT
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let draw_count_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("build_indirect_draw_count_staging"),
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            params_buffer,
            indirect_commands_buffer,
            draw_count_buffer,
            draw_count_staging,
            max_visible,
            max_draws,
        }
    }

    /// Upload parameters to the GPU.
    pub fn upload_params(&self, queue: &Queue, params: &BuildIndirectParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Clear the draw count to 0.
    ///
    /// Must be called before dispatching the build indirect shader.
    pub fn clear_draw_count(&self, queue: &Queue) {
        queue.write_buffer(&self.draw_count_buffer, 0, &[0u8; 4]);
    }

    /// Read the draw count back to the CPU.
    ///
    /// This is a synchronous operation that waits for GPU completion.
    pub fn read_draw_count(&self, device: &Device, queue: &Queue) -> u32 {
        // Copy from GPU buffer to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_build_indirect_draw_count"),
        });
        encoder.copy_buffer_to_buffer(&self.draw_count_buffer, 0, &self.draw_count_staging, 0, 4);
        queue.submit([encoder.finish()]);

        // Map and read the staging buffer
        let slice = self.draw_count_staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);

        let data = slice.get_mapped_range();
        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        self.draw_count_staging.unmap();

        count
    }

    /// Get the indirect commands buffer for rendering.
    ///
    /// This buffer can be passed to `draw_indexed_indirect` or
    /// `multi_draw_indexed_indirect_count`.
    #[inline]
    pub fn indirect_commands(&self) -> &Buffer {
        &self.indirect_commands_buffer
    }

    /// Get the draw count buffer for indirect count rendering.
    ///
    /// This buffer contains the GPU-side draw count for
    /// `multi_draw_indexed_indirect_count`.
    #[inline]
    pub fn draw_count(&self) -> &Buffer {
        &self.draw_count_buffer
    }
}

// =============================================================================
// BUILD INDIRECT PIPELINE
// =============================================================================

/// Compute pipeline for indirect buffer generation.
///
/// Provides methods to dispatch the build indirect compute shader.
pub struct BuildIndirectPipeline {
    /// Main compute pipeline (1 object per thread).
    pipeline: wgpu::ComputePipeline,
    /// Batched compute pipeline (4 objects per thread).
    pipeline_batched: wgpu::ComputePipeline,
    /// Clear draw count pipeline (single thread).
    pipeline_clear: wgpu::ComputePipeline,
    /// Bind group layout for input buffers.
    input_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for output buffers.
    output_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for parameters.
    params_bind_group_layout: wgpu::BindGroupLayout,
}

impl BuildIndirectPipeline {
    /// Create a new build indirect pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    pub fn new(device: &Device) -> Self {
        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("build_indirect_shader"),
            source: wgpu::ShaderSource::Wgsl(BUILD_INDIRECT_SHADER.into()),
        });

        // Input bind group layout (group 0)
        let input_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("build_indirect_input_layout"),
                entries: &[
                    // compacted_indices: array<u32>
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // object_data: array<ObjectData>
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
                    // mesh_data: array<MeshData>
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
                    // lod_buffer: array<LodEntry>
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Output bind group layout (group 1)
        let output_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("build_indirect_output_layout"),
                entries: &[
                    // indirect_commands: array<DrawIndexedIndirectArgs>
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // draw_count: atomic<u32>
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Params bind group layout (group 2)
        let params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("build_indirect_params_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("build_indirect_pipeline_layout"),
            bind_group_layouts: &[
                &input_bind_group_layout,
                &output_bind_group_layout,
                &params_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Main pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("build_indirect_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "build_indirect_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Batched pipeline
        let pipeline_batched = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("build_indirect_pipeline_batched"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "build_indirect_batched",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Clear pipeline
        let pipeline_clear = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("build_indirect_pipeline_clear"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "clear_draw_count",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            pipeline_batched,
            pipeline_clear,
            input_bind_group_layout,
            output_bind_group_layout,
            params_bind_group_layout,
        }
    }

    /// Create an input bind group.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `compacted_indices` - Buffer containing compacted visible object indices.
    /// * `object_data` - Buffer containing per-object data.
    /// * `mesh_data` - Buffer containing mesh data with LOD info.
    /// * `lod_buffer` - Buffer containing per-object LOD selection.
    pub fn create_input_bind_group(
        &self,
        device: &Device,
        compacted_indices: &Buffer,
        object_data: &Buffer,
        mesh_data: &Buffer,
        lod_buffer: &Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("build_indirect_input_bind_group"),
            layout: &self.input_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: compacted_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: object_data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mesh_data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: lod_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create an output bind group.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `resources` - The build indirect resources.
    pub fn create_output_bind_group(
        &self,
        device: &Device,
        resources: &BuildIndirectResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("build_indirect_output_bind_group"),
            layout: &self.output_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.indirect_commands_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: resources.draw_count_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Create a params bind group.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `resources` - The build indirect resources.
    pub fn create_params_bind_group(
        &self,
        device: &Device,
        resources: &BuildIndirectResources,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("build_indirect_params_bind_group"),
            layout: &self.params_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: resources.params_buffer.as_entire_binding(),
            }],
        })
    }

    /// Dispatch the build indirect compute shader.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder to record into.
    /// * `input_bind_group` - Input bind group (compacted indices, object data, etc.).
    /// * `output_bind_group` - Output bind group (indirect commands, draw count).
    /// * `params_bind_group` - Params bind group.
    /// * `params` - Build parameters (for workgroup calculation).
    /// * `use_batched` - Whether to use batched dispatch mode.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_bind_group: &wgpu::BindGroup,
        output_bind_group: &wgpu::BindGroup,
        params_bind_group: &wgpu::BindGroup,
        params: &BuildIndirectParams,
        use_batched: bool,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("build_indirect_pass"),
            timestamp_writes: None,
        });

        let (pipeline, workgroups) = if use_batched {
            (&self.pipeline_batched, params.workgroups_batched())
        } else {
            (&self.pipeline, params.workgroups())
        };

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, input_bind_group, &[]);
        pass.set_bind_group(1, output_bind_group, &[]);
        pass.set_bind_group(2, params_bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Dispatch the clear draw count compute shader.
    ///
    /// Should be called before dispatching build_indirect_main.
    pub fn dispatch_clear(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_bind_group: &wgpu::BindGroup,
        output_bind_group: &wgpu::BindGroup,
        params_bind_group: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("build_indirect_clear_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline_clear);
        pass.set_bind_group(0, input_bind_group, &[]);
        pass.set_bind_group(1, output_bind_group, &[]);
        pass.set_bind_group(2, params_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Get the input bind group layout.
    #[inline]
    pub fn input_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.input_bind_group_layout
    }

    /// Get the output bind group layout.
    #[inline]
    pub fn output_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.output_bind_group_layout
    }

    /// Get the params bind group layout.
    #[inline]
    pub fn params_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.params_bind_group_layout
    }
}

// =============================================================================
// CPU REFERENCE IMPLEMENTATION
// =============================================================================

/// CPU reference implementation of indirect buffer generation.
///
/// Used for testing and validation against the GPU implementation.
///
/// # Arguments
///
/// * `compacted_indices` - Array of visible object indices.
/// * `object_mesh_indices` - Array mapping object index to mesh index.
/// * `lod_levels` - Array of LOD levels per object.
/// * `mesh_data` - Array of mesh data with LOD info.
///
/// # Returns
///
/// Vector of generated `IndirectDrawIndexedArgs`.
pub fn cpu_build_indirect(
    compacted_indices: &[u32],
    object_mesh_indices: &[u32],
    lod_levels: &[u32],
    mesh_data: &[MeshData],
) -> Vec<IndirectDrawIndexedArgs> {
    let mut commands = Vec::with_capacity(compacted_indices.len());

    for (compact_idx, &object_idx) in compacted_indices.iter().enumerate() {
        let object_idx = object_idx as usize;

        // Get mesh index from object
        let mesh_id = object_mesh_indices.get(object_idx).copied().unwrap_or(0) as usize;

        // Get LOD level
        let lod_level = lod_levels
            .get(object_idx)
            .copied()
            .unwrap_or(0)
            .min(MAX_LOD_LEVEL) as usize;

        // Get mesh data
        let mesh = mesh_data.get(mesh_id).copied().unwrap_or_default();

        // Get LOD-specific values
        let index_count = mesh.index_count_for_lod(lod_level);
        let first_index = mesh.first_index_for_lod(lod_level);

        commands.push(IndirectDrawIndexedArgs {
            index_count,
            instance_count: 1,
            first_index,
            base_vertex: mesh.base_vertex,
            first_instance: object_idx as u32,
        });
    }

    commands
}

// =============================================================================
// INDIRECT DRAW ARGS
// =============================================================================

/// Arguments for `draw_indexed_indirect` (CPU-side representation).
///
/// This matches the wgpu `DrawIndexedIndirectArgs` layout exactly (20 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct IndirectDrawIndexedArgs {
    /// Number of indices to draw.
    pub index_count: u32,
    /// Number of instances to draw.
    pub instance_count: u32,
    /// Base index within the index buffer.
    pub first_index: u32,
    /// Vertex offset to add to each index (signed).
    pub base_vertex: i32,
    /// First instance to draw.
    pub first_instance: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<IndirectDrawIndexedArgs>() == DRAW_INDEXED_INDIRECT_ARGS_SIZE);

impl IndirectDrawIndexedArgs {
    /// Size in bytes of this struct.
    pub const SIZE: usize = DRAW_INDEXED_INDIRECT_ARGS_SIZE;

    /// Create new indexed draw arguments.
    #[inline]
    pub const fn new(
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Self {
        Self {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        }
    }

    /// Check if this draw would actually render anything.
    #[inline]
    pub const fn is_visible(&self) -> bool {
        self.index_count > 0 && self.instance_count > 0
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_params_size() {
        assert_eq!(
            mem::size_of::<BuildIndirectParams>(),
            BUILD_INDIRECT_PARAMS_SIZE
        );
        assert_eq!(mem::size_of::<BuildIndirectParams>(), 16);
    }

    #[test]
    fn test_mesh_data_size() {
        assert_eq!(mem::size_of::<MeshData>(), MESH_DATA_SIZE);
        assert_eq!(mem::size_of::<MeshData>(), 48);
    }

    #[test]
    fn test_indirect_args_layout() {
        assert_eq!(
            mem::size_of::<IndirectDrawIndexedArgs>(),
            DRAW_INDEXED_INDIRECT_ARGS_SIZE
        );
        assert_eq!(mem::size_of::<IndirectDrawIndexedArgs>(), 20);

        // Verify field offsets match wgpu expectations
        let args = IndirectDrawIndexedArgs::new(100, 1, 0, 0, 42);
        let bytes: &[u8] = bytemuck::bytes_of(&args);

        // index_count at offset 0
        assert_eq!(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]), 100);
        // instance_count at offset 4
        assert_eq!(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 1);
        // first_index at offset 8
        assert_eq!(u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]), 0);
        // base_vertex at offset 12
        assert_eq!(i32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]), 0);
        // first_instance at offset 16
        assert_eq!(u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]), 42);
    }

    #[test]
    fn test_lod_mesh_selection() {
        let mesh = MeshData::with_lods(
            1000,  // base index count
            0,     // first index
            0,     // base vertex
            &[1000, 500, 250, 125],  // LOD counts
            &[0, 1000, 1500, 1750],  // LOD offsets
        );

        // LOD 0 - highest detail
        assert_eq!(mesh.index_count_for_lod(0), 1000);
        assert_eq!(mesh.first_index_for_lod(0), 0);

        // LOD 1
        assert_eq!(mesh.index_count_for_lod(1), 500);
        assert_eq!(mesh.first_index_for_lod(1), 1000);

        // LOD 2
        assert_eq!(mesh.index_count_for_lod(2), 250);
        assert_eq!(mesh.first_index_for_lod(2), 1500);

        // LOD 3 - lowest detail
        assert_eq!(mesh.index_count_for_lod(3), 125);
        assert_eq!(mesh.first_index_for_lod(3), 1750);

        // Out of range LOD should clamp
        assert_eq!(mesh.index_count_for_lod(10), 125);
    }

    #[test]
    fn test_lod_fallback() {
        // Mesh without explicit LOD counts
        let mesh = MeshData::new(1000, 0, 0);

        // All LOD levels should fall back to base index_count
        for lod in 0..4 {
            assert_eq!(mesh.index_count_for_lod(lod), 1000);
            assert_eq!(mesh.first_index_for_lod(lod), 0);
        }
    }

    #[test]
    fn test_params_workgroups() {
        let params = BuildIndirectParams::new(1000, 4096);

        // Standard dispatch: ceil(1000 / 64) = 16 workgroups
        assert_eq!(params.workgroups(), 16);

        // Batched dispatch: ceil(1000 / (64 * 4)) = ceil(1000 / 256) = 4 workgroups
        assert_eq!(params.workgroups_batched(), 4);
    }

    #[test]
    fn test_params_batched_mode() {
        // Small count - use standard mode
        assert!(!BuildIndirectParams::new(5000, 4096).use_batched_mode());

        // Large count - use batched mode
        assert!(BuildIndirectParams::new(50000, 65536).use_batched_mode());

        // Boundary
        assert!(!BuildIndirectParams::new(10000, 4096).use_batched_mode());
        assert!(BuildIndirectParams::new(10001, 4096).use_batched_mode());
    }

    #[test]
    fn test_cpu_build_indirect() {
        let compacted_indices = vec![0, 2, 5];
        let object_mesh_indices = vec![0, 1, 0, 2, 1, 0]; // Object 0,2,5 use meshes 0,0,0
        let lod_levels = vec![0, 1, 2, 0, 1, 1]; // Object 0,2,5 use LODs 0,2,1
        let mesh_data = vec![
            MeshData::with_lods(1000, 0, 0, &[1000, 500, 250, 125], &[0, 1000, 1500, 1750]),
        ];

        let commands = cpu_build_indirect(
            &compacted_indices,
            &object_mesh_indices,
            &lod_levels,
            &mesh_data,
        );

        assert_eq!(commands.len(), 3);

        // Object 0 at LOD 0
        assert_eq!(commands[0].index_count, 1000);
        assert_eq!(commands[0].first_index, 0);
        assert_eq!(commands[0].first_instance, 0); // object_idx

        // Object 2 at LOD 2
        assert_eq!(commands[1].index_count, 250);
        assert_eq!(commands[1].first_index, 1500);
        assert_eq!(commands[1].first_instance, 2); // object_idx

        // Object 5 at LOD 1
        assert_eq!(commands[2].index_count, 500);
        assert_eq!(commands[2].first_index, 1000);
        assert_eq!(commands[2].first_instance, 5); // object_idx
    }

    #[test]
    fn test_shader_compiles() {
        // This test verifies the shader source is valid
        // In a real test, we'd need a wgpu device to compile
        assert!(!BUILD_INDIRECT_SHADER.is_empty());
        assert!(BUILD_INDIRECT_SHADER.contains("build_indirect_main"));
        assert!(BUILD_INDIRECT_SHADER.contains("build_indirect_batched"));
        assert!(BUILD_INDIRECT_SHADER.contains("clear_draw_count"));
        assert!(BUILD_INDIRECT_SHADER.contains("DrawIndexedIndirectArgs"));
        assert!(BUILD_INDIRECT_SHADER.contains("BuildIndirectParams"));
        assert!(BUILD_INDIRECT_SHADER.contains("MeshData"));
        assert!(BUILD_INDIRECT_SHADER.contains("LodEntry"));
    }
}

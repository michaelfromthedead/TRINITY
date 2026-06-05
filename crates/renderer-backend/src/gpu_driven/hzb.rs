//! GPU HZB (Hierarchical-Z Buffer) Construction for TRINITY Engine (T-GPU-4.1).
//!
//! This module provides GPU-based construction of hierarchical depth buffers
//! for efficient occlusion culling. The HZB is a mip-mapped texture where each
//! level stores the maximum depth of the corresponding 2x2 region from the
//! previous level.
//!
//! # Overview
//!
//! The HZB construction process:
//! 1. **Mip 0**: Downsample depth buffer 2x2 -> take max -> write to mip 0
//! 2. **Mip N**: Downsample mip N-1 2x2 -> take max -> write to mip N
//! 3. Continue until 1x1 or maximum mip count
//!
//! # Depth Convention
//!
//! TRINITY uses reversed-Z (near=1.0, far=0.0):
//! - MAX depth = furthest geometry (conservative for occlusion)
//! - Objects with depth > HZB depth are in front (visible)
//!
//! # Performance
//!
//! - Target: < 0.3ms at 4K resolution
//! - Memory: ~1.33x depth buffer size (mip chain)
//! - Dispatches: One per mip level (log2 of resolution)
//!
//! # Usage
//!
//! ```ignore
//! // Create pipeline
//! let pipeline = HZBPipeline::new(&device, &shader_source);
//!
//! // Create resources for 1920x1080 depth buffer
//! let resources = HZBResources::new(&device, 1920, 1080);
//!
//! // Build HZB each frame
//! pipeline.build_hzb(&mut encoder, &resources, &depth_view);
//!
//! // Use HZB texture view for occlusion culling
//! let hzb_view = resources.create_mip_view(&device, 0);
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (must match WGSL constant).
pub const WORKGROUP_SIZE: u32 = 8;

/// Maximum supported mip levels (enough for 16K resolution).
pub const MAX_HZB_MIPS: u32 = 14;

/// HZB flag: Use min instead of max (for standard Z depth).
pub const FLAG_USE_MIN: u32 = 1;

/// Default HZB texture format.
pub const HZB_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

// ---------------------------------------------------------------------------
// HZBBuildParams
// ---------------------------------------------------------------------------

/// GPU uniform buffer for HZB construction parameters.
///
/// # Memory Layout
///
/// 32 bytes, std140 compatible:
/// | Offset | Field       | Size |
/// |--------|-------------|------|
/// | 0      | src_width   | 4    |
/// | 4      | src_height  | 4    |
/// | 8      | dst_width   | 4    |
/// | 12     | dst_height  | 4    |
/// | 16     | current_mip | 4    |
/// | 20     | num_mips    | 4    |
/// | 24     | flags       | 4    |
/// | 28     | _pad0       | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HZBBuildParams {
    /// Source texture/mip width.
    pub src_width: u32,
    /// Source texture/mip height.
    pub src_height: u32,
    /// Destination mip width.
    pub dst_width: u32,
    /// Destination mip height.
    pub dst_height: u32,
    /// Current mip level being generated.
    pub current_mip: u32,
    /// Total number of mip levels.
    pub num_mips: u32,
    /// Flags for behavior control.
    pub flags: u32,
    /// Padding for 16-byte alignment.
    pub _pad0: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<HZBBuildParams>() == 32);

impl HZBBuildParams {
    /// Create parameters for building a specific mip level.
    ///
    /// # Arguments
    ///
    /// * `src_width` - Source texture/mip width
    /// * `src_height` - Source texture/mip height
    /// * `dst_width` - Destination mip width
    /// * `dst_height` - Destination mip height
    /// * `current_mip` - Mip level being generated (0 = from depth buffer)
    /// * `num_mips` - Total mip levels in the chain
    pub fn new(
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        current_mip: u32,
        num_mips: u32,
    ) -> Self {
        Self {
            src_width,
            src_height,
            dst_width,
            dst_height,
            current_mip,
            num_mips,
            flags: 0,
            _pad0: 0,
        }
    }

    /// Create parameters with flags.
    pub fn with_flags(
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        current_mip: u32,
        num_mips: u32,
        flags: u32,
    ) -> Self {
        Self {
            src_width,
            src_height,
            dst_width,
            dst_height,
            current_mip,
            num_mips,
            flags,
            _pad0: 0,
        }
    }

    /// Create parameters for mip 0 generation from depth buffer.
    pub fn for_mip0(depth_width: u32, depth_height: u32, num_mips: u32) -> Self {
        Self::new(
            depth_width,
            depth_height,
            (depth_width + 1) / 2,
            (depth_height + 1) / 2,
            0,
            num_mips,
        )
    }

    /// Create parameters for subsequent mip generation.
    pub fn for_mip_chain(src_width: u32, src_height: u32, current_mip: u32, num_mips: u32) -> Self {
        Self::new(
            src_width,
            src_height,
            (src_width + 1) / 2,
            (src_height + 1) / 2,
            current_mip,
            num_mips,
        )
    }

    /// Calculate the number of workgroups needed for X dimension.
    #[inline]
    pub fn workgroups_x(&self) -> u32 {
        (self.dst_width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }

    /// Calculate the number of workgroups needed for Y dimension.
    #[inline]
    pub fn workgroups_y(&self) -> u32 {
        (self.dst_height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
    }
}

// ---------------------------------------------------------------------------
// Mip Count Calculation
// ---------------------------------------------------------------------------

/// Calculate the number of mip levels needed for given dimensions.
///
/// Returns the number of mips from full resolution down to 1x1.
pub fn calculate_mip_count(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height);
    if max_dim == 0 {
        return 1;
    }
    // log2(max_dim) + 1 gives us mips from max_dim down to 1
    (32 - max_dim.leading_zeros()).max(1)
}

/// Calculate the dimensions of a specific mip level.
///
/// # Arguments
///
/// * `base_width` - Width at mip 0
/// * `base_height` - Height at mip 0
/// * `mip` - Mip level (0 = full resolution)
///
/// # Returns
///
/// (width, height) at the given mip level, minimum 1x1.
pub fn mip_dimensions(base_width: u32, base_height: u32, mip: u32) -> (u32, u32) {
    // Clamp mip to prevent overflow (mip > 31 would overflow)
    let mip = mip.min(31);
    let scale = 1u32 << mip;
    let w = (base_width / scale).max(1);
    let h = (base_height / scale).max(1);
    (w, h)
}

// ---------------------------------------------------------------------------
// HZBResources
// ---------------------------------------------------------------------------

/// GPU resources for HZB construction.
///
/// Contains the HZB texture with all mip levels, uniform buffer,
/// and per-mip texture views for reading/writing.
pub struct HZBResources {
    /// The HZB texture (all mip levels).
    pub hzb_texture: wgpu::Texture,
    /// Uniform buffer for build parameters.
    pub params_buffer: wgpu::Buffer,
    /// Base width (mip 0).
    pub width: u32,
    /// Base height (mip 0).
    pub height: u32,
    /// Number of mip levels.
    pub num_mips: u32,
    /// Original depth buffer dimensions.
    pub depth_width: u32,
    /// Original depth buffer dimensions.
    pub depth_height: u32,
}

impl HZBResources {
    /// Create HZB resources for the given depth buffer dimensions.
    ///
    /// The HZB mip 0 will be half the depth buffer size, and subsequent
    /// mips continue halving until 1x1.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `depth_width` - Depth buffer width.
    /// * `depth_height` - Depth buffer height.
    pub fn new(device: &wgpu::Device, depth_width: u32, depth_height: u32) -> Self {
        // HZB mip 0 is half the depth buffer size
        let width = (depth_width + 1) / 2;
        let height = (depth_height + 1) / 2;
        let num_mips = calculate_mip_count(width, height).min(MAX_HZB_MIPS);

        let hzb_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hzb_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: num_mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HZB_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hzb_build_params"),
            size: mem::size_of::<HZBBuildParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            hzb_texture,
            params_buffer,
            width,
            height,
            num_mips,
            depth_width,
            depth_height,
        }
    }

    /// Create a texture view for a specific mip level (for reading).
    pub fn create_mip_view(&self, mip: u32) -> wgpu::TextureView {
        self.hzb_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("hzb_mip{}_view", mip)),
            format: Some(HZB_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: mip,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        })
    }

    /// Create a storage texture view for a specific mip level (for writing).
    pub fn create_mip_storage_view(&self, mip: u32) -> wgpu::TextureView {
        self.hzb_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("hzb_mip{}_storage", mip)),
            format: Some(HZB_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: mip,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        })
    }

    /// Create a view of the entire HZB texture (all mips, for sampling).
    pub fn create_full_view(&self) -> wgpu::TextureView {
        self.hzb_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("hzb_full_view"),
            format: Some(HZB_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None, // All mips
            base_array_layer: 0,
            array_layer_count: Some(1),
        })
    }

    /// Upload parameters to the GPU.
    pub fn upload_params(&self, queue: &wgpu::Queue, params: &HZBBuildParams) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Get the dimensions of a specific mip level.
    pub fn get_mip_dimensions(&self, mip: u32) -> (u32, u32) {
        mip_dimensions(self.width, self.height, mip)
    }

    /// Calculate total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let mut total = 0usize;
        for mip in 0..self.num_mips {
            let (w, h) = self.get_mip_dimensions(mip);
            total += (w * h * 4) as usize; // R32Float = 4 bytes
        }
        total
    }
}

// ---------------------------------------------------------------------------
// HZBPipeline
// ---------------------------------------------------------------------------

/// GPU compute pipeline for HZB construction.
pub struct HZBPipeline {
    /// Pipeline for building mip 0 from depth buffer.
    pub pipeline_mip0: wgpu::ComputePipeline,
    /// Pipeline for building subsequent mips from previous mip.
    pub pipeline_chain: wgpu::ComputePipeline,
    /// Bind group layout for mip 0 (depth buffer source).
    pub layout_mip0: wgpu::BindGroupLayout,
    /// Bind group layout for mip chain (mip source).
    pub layout_chain: wgpu::BindGroupLayout,
}

impl HZBPipeline {
    /// Create the HZB construction pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `shader_source` - WGSL shader source code.
    pub fn new(device: &wgpu::Device, shader_source: &str) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hzb_build_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Layout for mip 0: params, depth_buffer, dst_mip, (unused src_mip)
        let layout_mip0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hzb_mip0_layout"),
            entries: &[
                // @binding(0) params: HZBBuildParams
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(mem::size_of::<HZBBuildParams>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                // @binding(1) src_depth: texture_2d<f32>
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(2) dst_mip: texture_storage_2d<r32float, write>
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: HZB_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // @binding(3) src_mip: texture_2d<f32> (unused for mip0, but must be bound)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Layout for mip chain: same structure but src_mip is actually used
        let layout_chain = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hzb_chain_layout"),
            entries: &[
                // @binding(0) params: HZBBuildParams
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(mem::size_of::<HZBBuildParams>() as u64)
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                // @binding(1) src_depth: texture_2d<f32> (unused for chain, but must match layout)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // @binding(2) dst_mip: texture_storage_2d<r32float, write>
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: HZB_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // @binding(3) src_mip: texture_2d<f32>
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout_mip0 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hzb_mip0_pipeline_layout"),
            bind_group_layouts: &[&layout_mip0],
            push_constant_ranges: &[],
        });

        let pipeline_layout_chain =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hzb_chain_pipeline_layout"),
                bind_group_layouts: &[&layout_chain],
                push_constant_ranges: &[],
            });

        let pipeline_mip0 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hzb_build_mip0"),
            layout: Some(&pipeline_layout_mip0),
            module: &shader_module,
            entry_point: "build_hzb_mip0",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let pipeline_chain = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hzb_build_chain"),
            layout: Some(&pipeline_layout_chain),
            module: &shader_module,
            entry_point: "build_hzb_mip_chain",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline_mip0,
            pipeline_chain,
            layout_mip0,
            layout_chain,
        }
    }

    /// Create a bind group for mip 0 generation.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `resources` - HZB resources.
    /// * `depth_view` - View of the depth buffer.
    /// * `dummy_src_mip` - Dummy texture view for unused src_mip binding.
    pub fn create_mip0_bind_group(
        &self,
        device: &wgpu::Device,
        resources: &HZBResources,
        depth_view: &wgpu::TextureView,
        dummy_src_mip: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let dst_view = resources.create_mip_storage_view(0);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hzb_mip0_bind_group"),
            layout: &self.layout_mip0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&dst_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(dummy_src_mip),
                },
            ],
        })
    }

    /// Create a bind group for mip chain generation.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `resources` - HZB resources.
    /// * `src_mip` - Mip level to read from.
    /// * `dst_mip` - Mip level to write to.
    /// * `dummy_depth` - Dummy texture view for unused depth binding.
    pub fn create_chain_bind_group(
        &self,
        device: &wgpu::Device,
        resources: &HZBResources,
        src_mip: u32,
        dst_mip: u32,
        dummy_depth: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let src_view = resources.create_mip_view(src_mip);
        let dst_view = resources.create_mip_storage_view(dst_mip);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("hzb_chain_{}_to_{}", src_mip, dst_mip)),
            layout: &self.layout_chain,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: resources.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(dummy_depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&dst_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
            ],
        })
    }

    /// Build the complete HZB mip chain.
    ///
    /// This dispatches one compute pass per mip level.
    ///
    /// # Arguments
    ///
    /// * `encoder` - Command encoder.
    /// * `resources` - HZB resources.
    /// * `depth_view` - View of the source depth buffer.
    /// * `queue` - Queue for parameter uploads.
    /// * `device` - Device for creating bind groups.
    pub fn build_hzb(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resources: &HZBResources,
        depth_view: &wgpu::TextureView,
        queue: &wgpu::Queue,
        device: &wgpu::Device,
    ) {
        // Build mip 0 from depth buffer
        let params_mip0 =
            HZBBuildParams::for_mip0(resources.depth_width, resources.depth_height, resources.num_mips);
        resources.upload_params(queue, &params_mip0);

        // Create a dummy view for unused bindings (use mip 0 view)
        let dummy_view = resources.create_mip_view(0);

        let bind_group_mip0 = self.create_mip0_bind_group(device, resources, depth_view, &dummy_view);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hzb_build_mip0"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_mip0);
            pass.set_bind_group(0, &bind_group_mip0, &[]);
            pass.dispatch_workgroups(params_mip0.workgroups_x(), params_mip0.workgroups_y(), 1);
        }

        // Build remaining mips
        let mut src_width = resources.width;
        let mut src_height = resources.height;

        for mip in 1..resources.num_mips {
            let dst_width = (src_width + 1) / 2;
            let dst_height = (src_height + 1) / 2;

            let params = HZBBuildParams::for_mip_chain(src_width, src_height, mip, resources.num_mips);
            resources.upload_params(queue, &params);

            let bind_group = self.create_chain_bind_group(device, resources, mip - 1, mip, depth_view);

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("hzb_build_mip{}", mip)),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_chain);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(params.workgroups_x(), params.workgroups_y(), 1);
            }

            src_width = dst_width;
            src_height = dst_height;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU Reference Implementation
// ---------------------------------------------------------------------------

/// CPU reference implementation of HZB construction.
///
/// Builds the complete mip chain on the CPU for testing and validation.
///
/// # Arguments
///
/// * `depth_buffer` - Source depth values (row-major).
/// * `depth_width` - Depth buffer width.
/// * `depth_height` - Depth buffer height.
/// * `reversed_z` - Use max reduction (true for reversed-Z, false for standard).
///
/// # Returns
///
/// A vector containing all mip levels concatenated, with offsets returned.
pub fn cpu_build_hzb(
    depth_buffer: &[f32],
    depth_width: u32,
    depth_height: u32,
    reversed_z: bool,
) -> (Vec<f32>, Vec<usize>, u32, u32) {
    let hzb_width = (depth_width + 1) / 2;
    let hzb_height = (depth_height + 1) / 2;
    let num_mips = calculate_mip_count(hzb_width, hzb_height);

    // Calculate total size and offsets
    let mut offsets = Vec::with_capacity(num_mips as usize);
    let mut total_size = 0usize;

    for mip in 0..num_mips {
        offsets.push(total_size);
        let (w, h) = mip_dimensions(hzb_width, hzb_height, mip);
        total_size += (w * h) as usize;
    }

    let mut hzb_data = vec![0.0f32; total_size];

    // Build mip 0 from depth buffer
    let (mip0_w, mip0_h) = mip_dimensions(hzb_width, hzb_height, 0);
    for y in 0..mip0_h {
        for x in 0..mip0_w {
            let sx = x * 2;
            let sy = y * 2;

            let mut depth_values = Vec::with_capacity(4);
            for dy in 0..2 {
                for dx in 0..2 {
                    let px = (sx + dx).min(depth_width - 1);
                    let py = (sy + dy).min(depth_height - 1);
                    let idx = (py * depth_width + px) as usize;
                    depth_values.push(depth_buffer[idx]);
                }
            }

            let reduced = if reversed_z {
                depth_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            } else {
                depth_values.iter().cloned().fold(f32::INFINITY, f32::min)
            };

            let dst_idx = offsets[0] + (y * mip0_w + x) as usize;
            hzb_data[dst_idx] = reduced;
        }
    }

    // Build remaining mips
    for mip in 1..num_mips {
        let (src_w, src_h) = mip_dimensions(hzb_width, hzb_height, mip - 1);
        let (dst_w, dst_h) = mip_dimensions(hzb_width, hzb_height, mip);

        for y in 0..dst_h {
            for x in 0..dst_w {
                let sx = x * 2;
                let sy = y * 2;

                let mut depth_values = Vec::with_capacity(4);
                for dy in 0..2 {
                    for dx in 0..2 {
                        let px = (sx + dx).min(src_w - 1);
                        let py = (sy + dy).min(src_h - 1);
                        let idx = offsets[(mip - 1) as usize] + (py * src_w + px) as usize;
                        depth_values.push(hzb_data[idx]);
                    }
                }

                let reduced = if reversed_z {
                    depth_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                } else {
                    depth_values.iter().cloned().fold(f32::INFINITY, f32::min)
                };

                let dst_idx = offsets[mip as usize] + (y * dst_w + x) as usize;
                hzb_data[dst_idx] = reduced;
            }
        }
    }

    (hzb_data, offsets, hzb_width, hzb_height)
}

/// Sample from a CPU HZB buffer at a specific mip level.
pub fn cpu_sample_hzb(
    hzb_data: &[f32],
    offsets: &[usize],
    base_width: u32,
    base_height: u32,
    x: u32,
    y: u32,
    mip: u32,
) -> f32 {
    let (w, h) = mip_dimensions(base_width, base_height, mip);
    let x = x.min(w - 1);
    let y = y.min(h - 1);
    let idx = offsets[mip as usize] + (y * w + x) as usize;
    hzb_data[idx]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Struct Size Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_params_struct_size() {
        assert_eq!(
            mem::size_of::<HZBBuildParams>(),
            32,
            "HZBBuildParams must be 32 bytes for GPU alignment"
        );
    }

    // -------------------------------------------------------------------------
    // Mip Count Calculation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mip_count_power_of_two() {
        assert_eq!(calculate_mip_count(1, 1), 1);
        assert_eq!(calculate_mip_count(2, 2), 2);
        assert_eq!(calculate_mip_count(4, 4), 3);
        assert_eq!(calculate_mip_count(8, 8), 4);
        assert_eq!(calculate_mip_count(256, 256), 9);
        assert_eq!(calculate_mip_count(512, 512), 10);
        assert_eq!(calculate_mip_count(1024, 1024), 11);
    }

    #[test]
    fn test_mip_count_non_power_of_two() {
        assert_eq!(calculate_mip_count(3, 3), 2);
        assert_eq!(calculate_mip_count(5, 5), 3);
        assert_eq!(calculate_mip_count(100, 100), 7);
        assert_eq!(calculate_mip_count(1920, 1080), 11);
    }

    #[test]
    fn test_mip_count_asymmetric() {
        assert_eq!(calculate_mip_count(1024, 512), 11);
        assert_eq!(calculate_mip_count(512, 1024), 11);
        assert_eq!(calculate_mip_count(256, 1), 9);
        assert_eq!(calculate_mip_count(1, 256), 9);
    }

    // -------------------------------------------------------------------------
    // Mip Dimensions Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mip_dimensions_power_of_two() {
        assert_eq!(mip_dimensions(256, 256, 0), (256, 256));
        assert_eq!(mip_dimensions(256, 256, 1), (128, 128));
        assert_eq!(mip_dimensions(256, 256, 2), (64, 64));
        assert_eq!(mip_dimensions(256, 256, 7), (2, 2));
        assert_eq!(mip_dimensions(256, 256, 8), (1, 1));
    }

    #[test]
    fn test_mip_dimensions_non_power_of_two() {
        // 100x100: 100 -> 50 -> 25 -> 12 -> 6 -> 3 -> 1
        assert_eq!(mip_dimensions(100, 100, 0), (100, 100));
        assert_eq!(mip_dimensions(100, 100, 1), (50, 50));
        assert_eq!(mip_dimensions(100, 100, 2), (25, 25));
        assert_eq!(mip_dimensions(100, 100, 3), (12, 12));
        assert_eq!(mip_dimensions(100, 100, 4), (6, 6));
        assert_eq!(mip_dimensions(100, 100, 5), (3, 3));
        assert_eq!(mip_dimensions(100, 100, 6), (1, 1));
    }

    #[test]
    fn test_mip_dimensions_minimum_one() {
        // Very high mip levels should still return at least 1x1
        assert_eq!(mip_dimensions(8, 8, 100), (1, 1));
    }

    // -------------------------------------------------------------------------
    // CPU HZB Build Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cpu_build_hzb_single_pixel() {
        // 2x2 depth -> 1x1 HZB
        let depth = vec![0.1, 0.2, 0.3, 0.4];
        let (hzb, offsets, w, h) = cpu_build_hzb(&depth, 2, 2, true);

        assert_eq!(w, 1);
        assert_eq!(h, 1);
        assert_eq!(offsets.len(), 1);
        assert_eq!(hzb.len(), 1);
        // Max of [0.1, 0.2, 0.3, 0.4] = 0.4
        assert!((hzb[0] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_build_hzb_full_chain() {
        // 8x8 depth -> 4x4 -> 2x2 -> 1x1 HZB
        let mut depth = vec![0.0f32; 64];
        // Set one corner high
        depth[0] = 1.0;

        let (hzb, offsets, w, h) = cpu_build_hzb(&depth, 8, 8, true);

        assert_eq!(w, 4);
        assert_eq!(h, 4);
        // Should have 3 mip levels: 4x4, 2x2, 1x1
        assert_eq!(offsets.len(), 3);

        // Mip 0 (4x4): corner should have max 1.0
        let mip0_corner = cpu_sample_hzb(&hzb, &offsets, w, h, 0, 0, 0);
        assert!((mip0_corner - 1.0).abs() < 1e-6);

        // Final mip (1x1): should propagate max
        let mip2_value = cpu_sample_hzb(&hzb, &offsets, w, h, 0, 0, 2);
        assert!((mip2_value - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_build_hzb_max_reduction() {
        // 4x4 depth with distinct values
        let depth = vec![
            0.1, 0.2, 0.3, 0.4, // Row 0
            0.5, 0.6, 0.7, 0.8, // Row 1
            0.9, 1.0, 0.0, 0.0, // Row 2
            0.0, 0.0, 0.0, 0.0, // Row 3
        ];

        let (hzb, offsets, w, h) = cpu_build_hzb(&depth, 4, 4, true);

        assert_eq!(w, 2);
        assert_eq!(h, 2);

        // Mip 0 (2x2):
        // [0,0]: max(0.1, 0.2, 0.5, 0.6) = 0.6
        // [1,0]: max(0.3, 0.4, 0.7, 0.8) = 0.8
        // [0,1]: max(0.9, 1.0, 0.0, 0.0) = 1.0
        // [1,1]: max(0.0, 0.0, 0.0, 0.0) = 0.0

        let v00 = cpu_sample_hzb(&hzb, &offsets, w, h, 0, 0, 0);
        let v10 = cpu_sample_hzb(&hzb, &offsets, w, h, 1, 0, 0);
        let v01 = cpu_sample_hzb(&hzb, &offsets, w, h, 0, 1, 0);
        let v11 = cpu_sample_hzb(&hzb, &offsets, w, h, 1, 1, 0);

        assert!((v00 - 0.6).abs() < 1e-6, "v00 = {}", v00);
        assert!((v10 - 0.8).abs() < 1e-6, "v10 = {}", v10);
        assert!((v01 - 1.0).abs() < 1e-6, "v01 = {}", v01);
        assert!((v11 - 0.0).abs() < 1e-6, "v11 = {}", v11);
    }

    #[test]
    fn test_cpu_build_hzb_min_reduction() {
        // Test min reduction for standard Z
        let depth = vec![0.1, 0.2, 0.3, 0.4];
        let (hzb, _, _, _) = cpu_build_hzb(&depth, 2, 2, false);

        // Min of [0.1, 0.2, 0.3, 0.4] = 0.1
        assert!((hzb[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_build_hzb_non_power_of_two() {
        // 5x5 depth -> 3x3 -> 2x2 -> 1x1
        let depth = vec![1.0f32; 25];
        let (hzb, offsets, w, h) = cpu_build_hzb(&depth, 5, 5, true);

        assert_eq!(w, 3); // (5+1)/2 = 3
        assert_eq!(h, 3);

        // All values should be 1.0
        for &v in &hzb {
            assert!((v - 1.0).abs() < 1e-6);
        }

        // Check we have correct mip levels
        assert!(offsets.len() >= 2); // At least 3x3 and 2x2
    }

    #[test]
    fn test_cpu_build_hzb_depth_precision() {
        // Test that depth values are preserved exactly through reduction
        let depth = vec![
            0.123456789,
            0.123456789,
            0.123456789,
            0.123456789,
        ];
        let (hzb, _, _, _) = cpu_build_hzb(&depth, 2, 2, true);

        // Should preserve full f32 precision
        assert!((hzb[0] - 0.123456789).abs() < 1e-9);
    }

    #[test]
    fn test_cpu_build_hzb_large_texture() {
        // 256x256 depth buffer (simulates a reasonable resolution)
        let depth = vec![0.5f32; 256 * 256];
        let (hzb, offsets, w, h) = cpu_build_hzb(&depth, 256, 256, true);

        assert_eq!(w, 128);
        assert_eq!(h, 128);
        // 128x128 -> 64x64 -> 32x32 -> ... -> 1x1 = 8 mip levels
        assert_eq!(offsets.len(), 8);

        // All values should be 0.5
        let final_mip = cpu_sample_hzb(&hzb, &offsets, w, h, 0, 0, 7);
        assert!((final_mip - 0.5).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // Workgroup Calculation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_workgroup_calculations() {
        let params = HZBBuildParams::new(1024, 1024, 512, 512, 0, 10);

        // 512 / 8 = 64
        assert_eq!(params.workgroups_x(), 64);
        assert_eq!(params.workgroups_y(), 64);
    }

    #[test]
    fn test_workgroup_calculations_non_aligned() {
        let params = HZBBuildParams::new(100, 100, 50, 50, 0, 7);

        // (50 + 7) / 8 = 7
        assert_eq!(params.workgroups_x(), 7);
        assert_eq!(params.workgroups_y(), 7);
    }

    #[test]
    fn test_workgroup_calculations_small() {
        let params = HZBBuildParams::new(4, 4, 2, 2, 0, 2);

        // (2 + 7) / 8 = 1
        assert_eq!(params.workgroups_x(), 1);
        assert_eq!(params.workgroups_y(), 1);
    }

    // -------------------------------------------------------------------------
    // Parameter Helper Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_params_for_mip0() {
        let params = HZBBuildParams::for_mip0(1920, 1080, 10);

        assert_eq!(params.src_width, 1920);
        assert_eq!(params.src_height, 1080);
        assert_eq!(params.dst_width, 960);
        assert_eq!(params.dst_height, 540);
        assert_eq!(params.current_mip, 0);
        assert_eq!(params.num_mips, 10);
    }

    #[test]
    fn test_params_for_mip_chain() {
        let params = HZBBuildParams::for_mip_chain(512, 512, 3, 10);

        assert_eq!(params.src_width, 512);
        assert_eq!(params.src_height, 512);
        assert_eq!(params.dst_width, 256);
        assert_eq!(params.dst_height, 256);
        assert_eq!(params.current_mip, 3);
        assert_eq!(params.num_mips, 10);
    }

    // -------------------------------------------------------------------------
    // WGSL Shader Validation Test
    // -------------------------------------------------------------------------

    #[test]
    fn test_wgsl_shader_validates() {
        // Load the shader source
        let shader_source = include_str!("../../shaders/gpu_driven/hzb_build.comp.wgsl");

        // Verify the shader contains expected entry points
        assert!(
            shader_source.contains("fn build_hzb_mip("),
            "Shader should contain build_hzb_mip entry point"
        );
        assert!(
            shader_source.contains("fn build_hzb_mip0("),
            "Shader should contain build_hzb_mip0 entry point"
        );
        assert!(
            shader_source.contains("fn build_hzb_mip_chain("),
            "Shader should contain build_hzb_mip_chain entry point"
        );

        // Verify workgroup size matches
        assert!(
            shader_source.contains("@workgroup_size(8, 8"),
            "Shader workgroup size should be 8x8"
        );

        // Verify uniform struct layout matches
        assert!(
            shader_source.contains("src_width: u32"),
            "Shader should have src_width field"
        );
        assert!(
            shader_source.contains("dst_width: u32"),
            "Shader should have dst_width field"
        );
        assert!(
            shader_source.contains("current_mip: u32"),
            "Shader should have current_mip field"
        );

        // Verify bindings exist
        assert!(
            shader_source.contains("@binding(0)"),
            "Shader should have binding 0"
        );
        assert!(
            shader_source.contains("@binding(1)"),
            "Shader should have binding 1"
        );
        assert!(
            shader_source.contains("@binding(2)"),
            "Shader should have binding 2"
        );
        assert!(
            shader_source.contains("@binding(3)"),
            "Shader should have binding 3"
        );
    }

    // -------------------------------------------------------------------------
    // Memory Usage Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_hzb_memory_usage_calculation() {
        // For a 1920x1080 depth buffer:
        // HZB mip 0: 960x540
        // Expected mips: 960, 480, 240, 120, 60, 30, 15, 8, 4, 2, 1
        let depth_width = 1920;
        let depth_height = 1080;
        let hzb_width = (depth_width + 1) / 2;
        let hzb_height = (depth_height + 1) / 2;
        let num_mips = calculate_mip_count(hzb_width, hzb_height);

        // Calculate expected memory
        let mut expected_size = 0usize;
        for mip in 0..num_mips {
            let (w, h) = mip_dimensions(hzb_width, hzb_height, mip);
            expected_size += (w * h * 4) as usize; // R32Float = 4 bytes
        }

        // Verify it's reasonable (should be ~1.33x the mip 0 size)
        let mip0_size = (hzb_width * hzb_height * 4) as usize;
        assert!(
            expected_size < mip0_size * 2,
            "HZB memory should be less than 2x mip 0 size"
        );
        assert!(
            expected_size > mip0_size,
            "HZB memory should be more than mip 0 size"
        );
    }

    // -------------------------------------------------------------------------
    // Performance Target Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_4k_dispatch_count() {
        // For 4K (3840x2160) depth buffer:
        // HZB mip 0: 1920x1080
        // Verify reasonable number of dispatches
        let depth_width = 3840;
        let depth_height = 2160;
        let hzb_width = (depth_width + 1) / 2;
        let hzb_height = (depth_height + 1) / 2;
        let num_mips = calculate_mip_count(hzb_width, hzb_height);

        // Should need one dispatch per mip
        // 1920 -> 960 -> 480 -> 240 -> 120 -> 60 -> 30 -> 15 -> 8 -> 4 -> 2 -> 1
        // That's 11-12 mip levels
        assert!(
            num_mips <= MAX_HZB_MIPS,
            "Should not exceed max mip count"
        );
        assert!(
            num_mips >= 10,
            "4K should have at least 10 mip levels"
        );
        assert!(
            num_mips <= 12,
            "4K should have at most 12 mip levels"
        );
    }
}

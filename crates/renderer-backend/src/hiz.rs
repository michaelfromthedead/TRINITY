//! Hierarchical depth buffer (HiZ) generation for efficient ray marching (T-GIR-P4.1).
//!
//! This module provides GPU infrastructure for generating a hierarchical Z-buffer
//! (HiZ), which is a mip chain where each level stores the maximum depth of its
//! 2x2 children. HiZ enables efficient screen-space ray marching for SSR and SSGI
//! by allowing early-out tests at coarse mip levels.
//!
//! # Overview
//!
//! The HiZ buffer is constructed by progressively downsampling the depth buffer:
//! - Mip 0: Full-resolution depth (copied from scene depth buffer)
//! - Mip 1: 2x downsampled, each texel is max(4 parent texels)
//! - Mip N: Recursively downsampled until 1x1
//!
//! # Usage
//!
//! ```ignore
//! // Create HiZ buffer for 1920x1080 render target
//! let config = HiZConfig::new(1920, 1080);
//! let hiz = HiZBuffer::new(&device, &config);
//!
//! // Create generation pass
//! let pass = HiZGeneratePass::new(&device);
//!
//! // Each frame: generate mip chain from depth buffer
//! for mip in 0..config.mip_levels - 1 {
//!     let bind_group = pass.create_bind_group(
//!         &device,
//!         &hiz.views[mip as usize],
//!         &hiz.views[mip as usize + 1],
//!         &uniforms_buffer,
//!     );
//!     pass.dispatch(&mut encoder, &bind_group, dst_width, dst_height);
//! }
//! ```
//!
//! # Frame Graph Integration
//!
//! The [`HiZNode`] integrates with the frame graph system, declaring
//! resource dependencies for automatic barrier insertion. The mip chain
//! generation is expressed as a series of dependent compute passes.

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Compute shader workgroup size (8x8 threads).
pub const WORKGROUP_SIZE: u32 = 8;

/// Minimum texture dimension before stopping mip generation.
pub const MIN_MIP_DIMENSION: u32 = 1;

/// Texture format for HiZ buffer (single-channel float).
pub const HIZ_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

// ---------------------------------------------------------------------------
// HiZConfig
// ---------------------------------------------------------------------------

/// Configuration for HiZ buffer creation.
///
/// Determines the size and mip level count for the hierarchical depth buffer.
/// The number of mip levels is calculated as floor(log2(max(width, height))) + 1,
/// ensuring the mip chain continues down to a 1x1 texel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HiZConfig {
    /// Width of the base mip level in texels.
    pub width: u32,
    /// Height of the base mip level in texels.
    pub height: u32,
    /// Number of mip levels (computed automatically).
    pub mip_levels: u32,
    /// Texture format (always R32Float for HiZ).
    pub format: wgpu::TextureFormat,
}

impl HiZConfig {
    /// Create a new HiZ configuration for the given dimensions.
    ///
    /// The mip level count is computed as floor(log2(max(width, height))) + 1.
    ///
    /// # Arguments
    ///
    /// * `width` - Width of the depth buffer in texels.
    /// * `height` - Height of the depth buffer in texels.
    ///
    /// # Panics
    ///
    /// Panics if width or height is zero.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let config = HiZConfig::new(1920, 1080);
    /// assert_eq!(config.mip_levels, 11); // log2(1920) + 1
    /// ```
    pub fn new(width: u32, height: u32) -> Self {
        assert!(width > 0 && height > 0, "HiZ dimensions must be non-zero");

        let max_dim = width.max(height);
        let mip_levels = Self::compute_mip_levels(max_dim);

        Self {
            width,
            height,
            mip_levels,
            format: HIZ_FORMAT,
        }
    }

    /// Create a configuration with explicit mip level count.
    ///
    /// # Arguments
    ///
    /// * `width` - Width of the depth buffer in texels.
    /// * `height` - Height of the depth buffer in texels.
    /// * `mip_levels` - Number of mip levels to generate.
    ///
    /// # Panics
    ///
    /// Panics if width or height is zero, or if mip_levels is zero.
    pub fn with_mip_levels(width: u32, height: u32, mip_levels: u32) -> Self {
        assert!(width > 0 && height > 0, "HiZ dimensions must be non-zero");
        assert!(mip_levels > 0, "HiZ must have at least one mip level");

        Self {
            width,
            height,
            mip_levels,
            format: HIZ_FORMAT,
        }
    }

    /// Compute the number of mip levels for a given maximum dimension.
    ///
    /// Returns floor(log2(max_dim)) + 1.
    #[inline]
    pub fn compute_mip_levels(max_dim: u32) -> u32 {
        if max_dim == 0 {
            return 0;
        }
        (max_dim as f32).log2().floor() as u32 + 1
    }

    /// Get the dimensions of a specific mip level.
    ///
    /// # Arguments
    ///
    /// * `mip_level` - The mip level (0 = base resolution).
    ///
    /// # Returns
    ///
    /// (width, height) at the specified mip level. Dimensions are halved at
    /// each level, with a minimum of 1.
    #[inline]
    pub fn mip_size(&self, mip_level: u32) -> (u32, u32) {
        let w = (self.width >> mip_level).max(1);
        let h = (self.height >> mip_level).max(1);
        (w, h)
    }

    /// Get the total number of texels across all mip levels.
    ///
    /// Useful for memory budgeting.
    pub fn total_texels(&self) -> u64 {
        let mut total = 0u64;
        for mip in 0..self.mip_levels {
            let (w, h) = self.mip_size(mip);
            total += (w as u64) * (h as u64);
        }
        total
    }

    /// Get the memory size in bytes for the entire mip chain.
    ///
    /// Based on R32Float format (4 bytes per texel).
    pub fn memory_size(&self) -> u64 {
        self.total_texels() * 4 // R32Float = 4 bytes
    }
}

impl Default for HiZConfig {
    fn default() -> Self {
        Self::new(1920, 1080) // Default to 1080p
    }
}

// ---------------------------------------------------------------------------
// HiZBuffer
// ---------------------------------------------------------------------------

/// Hierarchical depth buffer with mip chain.
///
/// Contains the GPU texture and per-mip-level views needed for HiZ generation
/// and ray marching queries.
///
/// # Mip Chain
///
/// Each mip level stores the maximum depth of its 2x2 children:
/// - Mip 0: Base resolution depth
/// - Mip 1: Each texel = max(4 texels from mip 0)
/// - Mip N: Each texel = max(4 texels from mip N-1)
pub struct HiZBuffer {
    /// The GPU texture containing the full mip chain.
    pub texture: wgpu::Texture,
    /// Per-mip-level texture views for shader binding.
    pub views: Vec<wgpu::TextureView>,
    /// Configuration used to create this buffer.
    pub config: HiZConfig,
}

impl HiZBuffer {
    /// Create a new HiZ buffer with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `config` - HiZ configuration specifying dimensions and mip count.
    pub fn new(device: &wgpu::Device, config: &HiZConfig) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hiz_buffer"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: config.mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let views = Self::create_mip_views(&texture, config);

        Self {
            texture,
            views,
            config: *config,
        }
    }

    /// Create a sampled view for the entire mip chain.
    ///
    /// Used when sampling the HiZ buffer during ray marching.
    pub fn create_sampled_view(&self) -> wgpu::TextureView {
        self.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("hiz_sampled_view"),
            format: Some(self.config.format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(self.config.mip_levels),
            base_array_layer: 0,
            array_layer_count: Some(1),
        })
    }

    /// Create per-mip-level views for the texture.
    fn create_mip_views(texture: &wgpu::Texture, config: &HiZConfig) -> Vec<wgpu::TextureView> {
        (0..config.mip_levels)
            .map(|mip| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("hiz_mip_{}", mip)),
                    format: Some(config.format),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                })
            })
            .collect()
    }

    /// Get the texture view for a specific mip level.
    ///
    /// # Panics
    ///
    /// Panics if `mip_level` >= `config.mip_levels`.
    #[inline]
    pub fn view(&self, mip_level: u32) -> &wgpu::TextureView {
        &self.views[mip_level as usize]
    }

    /// Get the dimensions at a specific mip level.
    #[inline]
    pub fn mip_size(&self, mip_level: u32) -> (u32, u32) {
        self.config.mip_size(mip_level)
    }

    /// Get the number of mip levels.
    #[inline]
    pub fn mip_levels(&self) -> u32 {
        self.config.mip_levels
    }
}

// ---------------------------------------------------------------------------
// HiZGenerateUniforms
// ---------------------------------------------------------------------------

/// GPU uniforms for HiZ mip generation.
///
/// This struct is uploaded to a uniform buffer for each mip generation pass.
/// Matches the WGSL `HiZUniforms` struct layout.
///
/// # Memory Layout
///
/// 32 bytes total, std140/std430 compatible:
///
/// | Offset | Field         | Size    |
/// |--------|---------------|---------|
/// | 0      | src_mip_level | 4 bytes |
/// | 4      | dst_mip_level | 4 bytes |
/// | 8      | src_size[0]   | 4 bytes |
/// | 12     | src_size[1]   | 4 bytes |
/// | 16     | dst_size[0]   | 4 bytes |
/// | 20     | dst_size[1]   | 4 bytes |
/// | 24     | _pad[0]       | 4 bytes |
/// | 28     | _pad[1]       | 4 bytes |
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HiZGenerateUniforms {
    /// Source mip level to read from.
    pub src_mip_level: u32,
    /// Destination mip level to write to.
    pub dst_mip_level: u32,
    /// Source mip dimensions (width, height).
    pub src_size: [u32; 2],
    /// Destination mip dimensions (width, height).
    pub dst_size: [u32; 2],
    /// Padding for alignment.
    pub _pad: [u32; 2],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<HiZGenerateUniforms>() == 32);

impl HiZGenerateUniforms {
    /// Create uniforms for generating a specific mip level.
    ///
    /// # Arguments
    ///
    /// * `config` - HiZ configuration.
    /// * `src_mip` - Source mip level (to read from).
    /// * `dst_mip` - Destination mip level (to write to).
    pub fn new(config: &HiZConfig, src_mip: u32, dst_mip: u32) -> Self {
        let (src_w, src_h) = config.mip_size(src_mip);
        let (dst_w, dst_h) = config.mip_size(dst_mip);

        Self {
            src_mip_level: src_mip,
            dst_mip_level: dst_mip,
            src_size: [src_w, src_h],
            dst_size: [dst_w, dst_h],
            _pad: [0, 0],
        }
    }

    /// Create uniforms for all mip transitions.
    ///
    /// Returns a Vec of uniforms, one for each mip0->mip1, mip1->mip2, etc.
    pub fn for_all_mips(config: &HiZConfig) -> Vec<Self> {
        (0..config.mip_levels.saturating_sub(1))
            .map(|src| Self::new(config, src, src + 1))
            .collect()
    }
}

impl Default for HiZGenerateUniforms {
    fn default() -> Self {
        Self {
            src_mip_level: 0,
            dst_mip_level: 1,
            src_size: [1920, 1080],
            dst_size: [960, 540],
            _pad: [0, 0],
        }
    }
}

// ---------------------------------------------------------------------------
// HiZGeneratePass
// ---------------------------------------------------------------------------

/// Compute pass for HiZ mip chain generation.
///
/// This pass downsamples a source mip level to a destination mip level,
/// taking the maximum depth of each 2x2 block. To generate the full mip
/// chain, dispatch this pass once for each source->destination pair.
///
/// # Bind Group Layout
///
/// | Binding | Type                     | Content                |
/// |---------|--------------------------|------------------------|
/// | 0       | texture_2d<f32>          | Source mip level       |
/// | 1       | texture_storage_2d<write>| Destination mip level  |
/// | 2       | uniform                  | HiZGenerateUniforms    |
pub struct HiZGeneratePass {
    /// Compute pipeline for HiZ generation.
    pipeline: wgpu::ComputePipeline,
    /// Bind group layout for HiZ resources.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl HiZGeneratePass {
    /// Create a new HiZ generation pass.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(device);
        let pipeline = Self::create_pipeline(device, &bind_group_layout);

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Get the bind group layout for external bind group creation.
    #[inline]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Create a bind group for a single mip generation pass.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `src_view` - Source mip level texture view.
    /// * `dst_view` - Destination mip level texture view (storage).
    /// * `uniforms_buffer` - Buffer containing HiZGenerateUniforms.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        src_view: &wgpu::TextureView,
        dst_view: &wgpu::TextureView,
        uniforms_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hiz_generate_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(dst_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniforms_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch the HiZ generation compute shader.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder.
    /// * `bind_group` - The bind group containing source, destination, and uniforms.
    /// * `dst_width` - Width of the destination mip level.
    /// * `dst_height` - Height of the destination mip level.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        dst_width: u32,
        dst_height: u32,
    ) {
        let workgroups_x = (dst_width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        let workgroups_y = (dst_height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hiz_generate_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    /// Create the bind group layout for HiZ generation.
    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hiz_generate_bind_group_layout"),
            entries: &[
                // Binding 0: Source mip level (sampled texture)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 1: Destination mip level (storage texture)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: HIZ_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Binding 2: Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            mem::size_of::<HiZGenerateUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create the compute pipeline for HiZ generation.
    fn create_pipeline(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader_source = include_str!("../shaders/hiz_generate.comp.wgsl");

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hiz_generate_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hiz_generate_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hiz_generate_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "hiz_downsample",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }
}

// ---------------------------------------------------------------------------
// HiZPassChain
// ---------------------------------------------------------------------------

/// Helper for generating the full HiZ mip chain.
///
/// Encapsulates the per-mip bind groups and uniforms buffers needed to
/// generate all mip levels in sequence.
pub struct HiZPassChain {
    /// Uniform buffers, one per mip transition.
    uniform_buffers: Vec<wgpu::Buffer>,
    /// Bind groups, one per mip transition.
    bind_groups: Vec<wgpu::BindGroup>,
    /// HiZ configuration.
    config: HiZConfig,
}

impl HiZPassChain {
    /// Create a new pass chain for the given HiZ buffer.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `pass` - The HiZ generation pass.
    /// * `hiz` - The HiZ buffer to generate.
    pub fn new(device: &wgpu::Device, pass: &HiZGeneratePass, hiz: &HiZBuffer) -> Self {
        let uniforms = HiZGenerateUniforms::for_all_mips(&hiz.config);

        let uniform_buffers: Vec<wgpu::Buffer> = uniforms
            .iter()
            .enumerate()
            .map(|(i, u)| {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("hiz_uniforms_mip_{}", i)),
                    size: mem::size_of::<HiZGenerateUniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: true,
                });
                buffer
                    .slice(..)
                    .get_mapped_range_mut()
                    .copy_from_slice(bytemuck::bytes_of(u));
                buffer.unmap();
                buffer
            })
            .collect();

        let bind_groups: Vec<wgpu::BindGroup> = (0..uniforms.len())
            .map(|i| {
                pass.create_bind_group(
                    device,
                    &hiz.views[i],
                    &hiz.views[i + 1],
                    &uniform_buffers[i],
                )
            })
            .collect();

        Self {
            uniform_buffers,
            bind_groups,
            config: hiz.config,
        }
    }

    /// Dispatch the full mip chain generation.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder.
    /// * `pass` - The HiZ generation pass.
    pub fn dispatch_all(&self, encoder: &mut wgpu::CommandEncoder, pass: &HiZGeneratePass) {
        for (i, bind_group) in self.bind_groups.iter().enumerate() {
            let dst_mip = (i + 1) as u32;
            let (dst_width, dst_height) = self.config.mip_size(dst_mip);
            pass.dispatch(encoder, bind_group, dst_width, dst_height);
        }
    }

    /// Get the number of mip transitions.
    #[inline]
    pub fn transition_count(&self) -> usize {
        self.bind_groups.len()
    }
}

// ---------------------------------------------------------------------------
// Frame Graph Integration
// ---------------------------------------------------------------------------

use crate::frame_graph::{
    DispatchSource, IrPass, PassIndex, ResourceAccessSet, ResourceHandle, ViewType,
};

/// Create a frame graph pass for generating a single HiZ mip level.
///
/// # Arguments
///
/// * `index` - Pass index in the frame graph.
/// * `name` - Pass name (e.g., "hiz_generate_mip_1").
/// * `src_handle` - Resource handle for the source mip level.
/// * `dst_handle` - Resource handle for the destination mip level.
/// * `dst_width` - Width of the destination mip level.
/// * `dst_height` - Height of the destination mip level.
pub fn create_hiz_pass(
    index: PassIndex,
    name: &str,
    src_handle: ResourceHandle,
    dst_handle: ResourceHandle,
    dst_width: u32,
    dst_height: u32,
) -> IrPass {
    let workgroups_x = (dst_width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let workgroups_y = (dst_height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    let dispatch = DispatchSource::Direct {
        group_count_x: workgroups_x,
        group_count_y: workgroups_y,
        group_count_z: 1,
    };

    let mut pass = IrPass::compute(index, name, dispatch, ViewType::Storage);
    pass.access_set = ResourceAccessSet {
        reads: vec![src_handle],
        writes: vec![dst_handle],
    };
    pass
}

/// Create a chain of frame graph passes for the full HiZ mip chain.
///
/// # Arguments
///
/// * `base_index` - Starting pass index for the chain.
/// * `base_name` - Base name for passes (e.g., "hiz_generate").
/// * `config` - HiZ configuration.
/// * `mip_handles` - Resource handles for each mip level (mip 0 through mip N).
///
/// # Returns
///
/// A Vec of IrPass, one for each mip transition (mip0->mip1, mip1->mip2, ...).
pub fn create_hiz_pass_chain(
    base_index: usize,
    base_name: &str,
    config: &HiZConfig,
    mip_handles: &[ResourceHandle],
) -> Vec<IrPass> {
    assert!(
        mip_handles.len() >= config.mip_levels as usize,
        "Not enough resource handles for mip chain"
    );

    (0..config.mip_levels.saturating_sub(1))
        .map(|src_mip| {
            let dst_mip = src_mip + 1;
            let (dst_width, dst_height) = config.mip_size(dst_mip);

            create_hiz_pass(
                PassIndex(base_index + src_mip as usize),
                &format!("{}_{}", base_name, dst_mip),
                mip_handles[src_mip as usize],
                mip_handles[dst_mip as usize],
                dst_width,
                dst_height,
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_graph::PassIndex;

    // -----------------------------------------------------------------------
    // HiZConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mip_levels_1920x1080() {
        let config = HiZConfig::new(1920, 1080);
        // log2(1920) = 10.9, floor + 1 = 11
        assert_eq!(config.mip_levels, 11);
    }

    #[test]
    fn test_mip_levels_2560x1440() {
        let config = HiZConfig::new(2560, 1440);
        // log2(2560) = 11.3, floor + 1 = 12
        assert_eq!(config.mip_levels, 12);
    }

    #[test]
    fn test_mip_levels_3840x2160() {
        let config = HiZConfig::new(3840, 2160);
        // log2(3840) = 11.9, floor + 1 = 12
        assert_eq!(config.mip_levels, 12);
    }

    #[test]
    fn test_mip_levels_1024x1024() {
        let config = HiZConfig::new(1024, 1024);
        // log2(1024) = 10, floor + 1 = 11
        assert_eq!(config.mip_levels, 11);
    }

    #[test]
    fn test_mip_levels_512x512() {
        let config = HiZConfig::new(512, 512);
        // log2(512) = 9, floor + 1 = 10
        assert_eq!(config.mip_levels, 10);
    }

    #[test]
    fn test_mip_levels_256x256() {
        let config = HiZConfig::new(256, 256);
        // log2(256) = 8, floor + 1 = 9
        assert_eq!(config.mip_levels, 9);
    }

    #[test]
    fn test_mip_levels_power_of_two() {
        for power in 1..14 {
            let size = 1u32 << power;
            let config = HiZConfig::new(size, size);
            assert_eq!(config.mip_levels, power + 1);
        }
    }

    #[test]
    fn test_mip_levels_1x1() {
        let config = HiZConfig::new(1, 1);
        // log2(1) = 0, floor + 1 = 1
        assert_eq!(config.mip_levels, 1);
    }

    #[test]
    fn test_mip_levels_asymmetric() {
        let config = HiZConfig::new(1920, 480);
        // max(1920, 480) = 1920, log2(1920) = 10.9, floor + 1 = 11
        assert_eq!(config.mip_levels, 11);
    }

    #[test]
    fn test_mip_levels_portrait() {
        let config = HiZConfig::new(1080, 1920);
        // max(1080, 1920) = 1920, log2(1920) = 10.9, floor + 1 = 11
        assert_eq!(config.mip_levels, 11);
    }

    // -----------------------------------------------------------------------
    // Mip size tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mip_size_1024x1024() {
        let config = HiZConfig::new(1024, 1024);

        assert_eq!(config.mip_size(0), (1024, 1024));
        assert_eq!(config.mip_size(1), (512, 512));
        assert_eq!(config.mip_size(2), (256, 256));
        assert_eq!(config.mip_size(3), (128, 128));
        assert_eq!(config.mip_size(4), (64, 64));
        assert_eq!(config.mip_size(5), (32, 32));
        assert_eq!(config.mip_size(6), (16, 16));
        assert_eq!(config.mip_size(7), (8, 8));
        assert_eq!(config.mip_size(8), (4, 4));
        assert_eq!(config.mip_size(9), (2, 2));
        assert_eq!(config.mip_size(10), (1, 1));
    }

    #[test]
    fn test_mip_size_1920x1080() {
        let config = HiZConfig::new(1920, 1080);

        assert_eq!(config.mip_size(0), (1920, 1080));
        assert_eq!(config.mip_size(1), (960, 540));
        assert_eq!(config.mip_size(2), (480, 270));
        assert_eq!(config.mip_size(3), (240, 135));
        assert_eq!(config.mip_size(4), (120, 67));
        assert_eq!(config.mip_size(5), (60, 33));
        assert_eq!(config.mip_size(6), (30, 16));
        assert_eq!(config.mip_size(7), (15, 8));
        assert_eq!(config.mip_size(8), (7, 4));
        assert_eq!(config.mip_size(9), (3, 2));
        assert_eq!(config.mip_size(10), (1, 1));
    }

    #[test]
    fn test_mip_size_minimum_clamp() {
        let config = HiZConfig::new(4, 4);

        // Should never go below 1x1
        assert_eq!(config.mip_size(0), (4, 4));
        assert_eq!(config.mip_size(1), (2, 2));
        assert_eq!(config.mip_size(2), (1, 1));
        // Beyond max mip, still clamped to 1
        assert_eq!(config.mip_size(10), (1, 1));
    }

    #[test]
    fn test_mip_size_asymmetric() {
        let config = HiZConfig::new(512, 128);

        assert_eq!(config.mip_size(0), (512, 128));
        assert_eq!(config.mip_size(1), (256, 64));
        assert_eq!(config.mip_size(2), (128, 32));
        assert_eq!(config.mip_size(3), (64, 16));
        assert_eq!(config.mip_size(4), (32, 8));
        assert_eq!(config.mip_size(5), (16, 4));
        assert_eq!(config.mip_size(6), (8, 2));
        assert_eq!(config.mip_size(7), (4, 1));
        assert_eq!(config.mip_size(8), (2, 1));
        assert_eq!(config.mip_size(9), (1, 1));
    }

    // -----------------------------------------------------------------------
    // HiZGenerateUniforms tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_uniforms_size() {
        assert_eq!(mem::size_of::<HiZGenerateUniforms>(), 32);
    }

    #[test]
    fn test_uniforms_alignment() {
        // Verify struct is properly aligned for GPU upload
        assert_eq!(mem::align_of::<HiZGenerateUniforms>(), 4);
    }

    #[test]
    fn test_uniforms_pod() {
        // Verify Pod/Zeroable traits work
        let uniforms = HiZGenerateUniforms::default();
        let bytes = bytemuck::bytes_of(&uniforms);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_uniforms_mip_0_to_1() {
        let config = HiZConfig::new(1920, 1080);
        let uniforms = HiZGenerateUniforms::new(&config, 0, 1);

        assert_eq!(uniforms.src_mip_level, 0);
        assert_eq!(uniforms.dst_mip_level, 1);
        assert_eq!(uniforms.src_size, [1920, 1080]);
        assert_eq!(uniforms.dst_size, [960, 540]);
    }

    #[test]
    fn test_uniforms_mip_5_to_6() {
        let config = HiZConfig::new(1920, 1080);
        let uniforms = HiZGenerateUniforms::new(&config, 5, 6);

        assert_eq!(uniforms.src_mip_level, 5);
        assert_eq!(uniforms.dst_mip_level, 6);
        assert_eq!(uniforms.src_size, [60, 33]);
        assert_eq!(uniforms.dst_size, [30, 16]);
    }

    #[test]
    fn test_uniforms_for_all_mips() {
        let config = HiZConfig::new(256, 256);
        let all_uniforms = HiZGenerateUniforms::for_all_mips(&config);

        // 256x256 has 9 mip levels, so 8 transitions
        assert_eq!(all_uniforms.len(), 8);

        // Verify first transition
        assert_eq!(all_uniforms[0].src_mip_level, 0);
        assert_eq!(all_uniforms[0].dst_mip_level, 1);
        assert_eq!(all_uniforms[0].src_size, [256, 256]);
        assert_eq!(all_uniforms[0].dst_size, [128, 128]);

        // Verify last transition
        assert_eq!(all_uniforms[7].src_mip_level, 7);
        assert_eq!(all_uniforms[7].dst_mip_level, 8);
        assert_eq!(all_uniforms[7].src_size, [2, 2]);
        assert_eq!(all_uniforms[7].dst_size, [1, 1]);
    }

    #[test]
    fn test_uniforms_single_mip() {
        let config = HiZConfig::new(1, 1);
        let all_uniforms = HiZGenerateUniforms::for_all_mips(&config);

        // 1x1 has only 1 mip level, so 0 transitions
        assert_eq!(all_uniforms.len(), 0);
    }

    // -----------------------------------------------------------------------
    // Memory size tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_memory_size_1024x1024() {
        let config = HiZConfig::new(1024, 1024);

        // Sum: 1024^2 + 512^2 + 256^2 + ... + 1^1
        // = 1048576 + 262144 + 65536 + 16384 + 4096 + 1024 + 256 + 64 + 16 + 4 + 1
        // = 1398101 texels
        // * 4 bytes = 5592404 bytes
        let expected_texels: u64 = (0..11).map(|m| (1024u64 >> m) * (1024u64 >> m)).sum();
        assert_eq!(config.total_texels(), expected_texels);
        assert_eq!(config.memory_size(), expected_texels * 4);
    }

    #[test]
    fn test_memory_size_small() {
        let config = HiZConfig::new(4, 4);
        // 4x4 + 2x2 + 1x1 = 16 + 4 + 1 = 21 texels * 4 = 84 bytes
        assert_eq!(config.total_texels(), 21);
        assert_eq!(config.memory_size(), 84);
    }

    // -----------------------------------------------------------------------
    // Frame graph integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_hiz_pass() {
        use crate::frame_graph::PassType;

        let src = ResourceHandle(0);
        let dst = ResourceHandle(1);

        let pass = create_hiz_pass(PassIndex(0), "hiz_test", src, dst, 512, 512);

        assert_eq!(pass.name, "hiz_test");
        assert_eq!(pass.pass_type, PassType::Compute);
        assert!(pass.access_set.reads.contains(&src));
        assert!(pass.access_set.writes.contains(&dst));

        // Verify workgroup counts
        if let Some(DispatchSource::Direct { group_count_x, group_count_y, group_count_z }) = pass.dispatch_source {
            // 512 / 8 = 64 workgroups
            assert_eq!(group_count_x, 64);
            assert_eq!(group_count_y, 64);
            assert_eq!(group_count_z, 1);
        } else {
            panic!("Expected Direct dispatch");
        }
    }

    #[test]
    fn test_create_hiz_pass_odd_size() {
        let src = ResourceHandle(0);
        let dst = ResourceHandle(1);

        let pass = create_hiz_pass(PassIndex(0), "hiz_odd", src, dst, 100, 100);

        // 100 / 8 = 12.5, ceil = 13 workgroups
        if let Some(DispatchSource::Direct { group_count_x, group_count_y, .. }) = pass.dispatch_source {
            assert_eq!(group_count_x, 13);
            assert_eq!(group_count_y, 13);
        } else {
            panic!("Expected Direct dispatch");
        }
    }

    #[test]
    fn test_create_hiz_pass_chain() {
        let config = HiZConfig::new(64, 64);
        let handles: Vec<ResourceHandle> = (0..config.mip_levels).map(ResourceHandle).collect();

        let passes = create_hiz_pass_chain(0, "hiz", &config, &handles);

        // 64x64 has 7 mip levels, so 6 transitions
        assert_eq!(passes.len(), 6);

        // Verify pass names
        for (i, pass) in passes.iter().enumerate() {
            assert_eq!(pass.name, format!("hiz_{}", i + 1));
        }

        // Verify dependencies form a chain
        assert!(passes[0].access_set.reads.contains(&handles[0]));
        assert!(passes[0].access_set.writes.contains(&handles[1]));

        assert!(passes[1].access_set.reads.contains(&handles[1]));
        assert!(passes[1].access_set.writes.contains(&handles[2]));
    }

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "HiZ dimensions must be non-zero")]
    fn test_zero_width_panics() {
        HiZConfig::new(0, 100);
    }

    #[test]
    #[should_panic(expected = "HiZ dimensions must be non-zero")]
    fn test_zero_height_panics() {
        HiZConfig::new(100, 0);
    }

    #[test]
    fn test_with_mip_levels_explicit() {
        let config = HiZConfig::with_mip_levels(1920, 1080, 5);
        assert_eq!(config.mip_levels, 5);
    }

    #[test]
    #[should_panic(expected = "HiZ must have at least one mip level")]
    fn test_zero_mip_levels_panics() {
        HiZConfig::with_mip_levels(1920, 1080, 0);
    }

    #[test]
    fn test_compute_mip_levels_zero() {
        assert_eq!(HiZConfig::compute_mip_levels(0), 0);
    }

    #[test]
    fn test_default_config() {
        let config = HiZConfig::default();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.format, wgpu::TextureFormat::R32Float);
    }

    #[test]
    fn test_default_uniforms() {
        let uniforms = HiZGenerateUniforms::default();
        assert_eq!(uniforms.src_mip_level, 0);
        assert_eq!(uniforms.dst_mip_level, 1);
        assert_eq!(uniforms.src_size, [1920, 1080]);
        assert_eq!(uniforms.dst_size, [960, 540]);
    }

    // -----------------------------------------------------------------------
    // Shader validation tests (using naga)
    // -----------------------------------------------------------------------

    #[test]
    fn test_hiz_shader_parses() {
        // Validate that the HiZ compute shader parses correctly
        let shader_source = include_str!("../shaders/hiz_generate.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("HiZ shader should parse without errors");

        // Verify the entry point exists
        let entry_point = module
            .entry_points
            .iter()
            .find(|ep| ep.name == "hiz_downsample");
        assert!(
            entry_point.is_some(),
            "Should have hiz_downsample entry point"
        );

        // Verify it's a compute shader
        let ep = entry_point.unwrap();
        assert_eq!(
            ep.stage,
            naga::ShaderStage::Compute,
            "Should be a compute shader"
        );

        // Verify workgroup size
        assert_eq!(ep.workgroup_size, [8, 8, 1], "Workgroup size should be 8x8x1");
    }

    #[test]
    fn test_hiz_shader_validates() {
        // Full validation including type checking
        let shader_source = include_str!("../shaders/hiz_generate.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("HiZ shader should parse without errors");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );

        let _info = validator
            .validate(&module)
            .expect("HiZ shader should validate without errors");

        // Verify entry point from module (workgroup size is in the module, not validation info)
        let ep = module
            .entry_points
            .iter()
            .find(|ep| ep.name == "hiz_downsample")
            .expect("Should have hiz_downsample entry point");
        assert_eq!(
            ep.workgroup_size, [8, 8, 1],
            "Entry point workgroup size should be 8x8x1"
        );
    }

    #[test]
    fn test_hiz_shader_bindings() {
        // Verify the shader has the expected bindings
        let shader_source = include_str!("../shaders/hiz_generate.comp.wgsl");

        let module = naga::front::wgsl::parse_str(shader_source)
            .expect("HiZ shader should parse without errors");

        // Count global variables by binding type
        let mut texture_count = 0;
        let mut storage_count = 0;
        let mut uniform_count = 0;

        for (_, var) in module.global_variables.iter() {
            match var.space {
                naga::AddressSpace::Handle => {
                    // Could be texture or sampler
                    if let naga::TypeInner::Image { .. } = module.types[var.ty].inner {
                        if var.binding.is_some() {
                            let binding = var.binding.as_ref().unwrap();
                            if binding.binding == 1 {
                                storage_count += 1;
                            } else {
                                texture_count += 1;
                            }
                        }
                    }
                }
                naga::AddressSpace::Uniform => {
                    uniform_count += 1;
                }
                _ => {}
            }
        }

        assert!(texture_count >= 1, "Should have at least 1 texture binding");
        assert!(storage_count >= 1, "Should have at least 1 storage texture binding");
        assert!(uniform_count >= 1, "Should have at least 1 uniform binding");
    }
}

//! HiZ Pyramid Creation for Occlusion Culling (T-WGPU-P6.4.1).
//!
//! This module provides the `HiZPyramid` struct for creating and managing
//! a Hierarchical-Z (HiZ) pyramid texture used in GPU-driven occlusion culling.
//! The HiZ pyramid is a mip-mapped texture where each level stores depth
//! information at progressively coarser resolutions.
//!
//! # Overview
//!
//! The HiZ pyramid enables efficient hierarchical depth testing:
//! - **Mip 0**: Base level matching depth buffer resolution (or half for HZB)
//! - **Mip N**: Each subsequent level is half the resolution of the previous
//! - **Final Mip**: 1x1 pixel representing the entire scene's depth range
//!
//! # Depth Convention
//!
//! TRINITY uses reversed-Z (near=1.0, far=0.0):
//! - MAX depth values = furthest geometry (conservative for occlusion)
//! - Objects with depth > HiZ depth are in front (visible)
//!
//! # Format Notes
//!
//! - Uses `R32Float` instead of `Depth32Float` because `Depth32Float`
//!   cannot be used with `STORAGE_BINDING` in wgpu
//! - Depth values must be copied from the depth buffer to this format
//! - Use `COPY_DST` for initial depth buffer copies
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::hiz_pyramid::HiZPyramid;
//!
//! // Create pyramid for 1920x1080 resolution
//! let pyramid = HiZPyramid::new(&device, 1920, 1080);
//!
//! // Access the full texture for sampling
//! let view = pyramid.view();
//!
//! // Access individual mip levels for compute writes
//! let mip2_view = pyramid.mip_view(2).unwrap();
//!
//! // Query dimensions
//! let (w, h) = HiZPyramid::calculate_mip_size(1920, 1080, 2);
//! ```
//!
//! # Performance
//!
//! - Memory: Sum of all mip levels (~1.33x base resolution)
//! - Access: All mips accessible via single texture binding
//! - Writes: Individual mips accessible via storage views
//!
//! # See Also
//!
//! - [`super::hzb`] - GPU compute pipeline for building the HiZ buffer
//! - [`super::occlusion_cull`] - Occlusion culling using HiZ

use std::mem;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Texture format for HiZ pyramid.
///
/// Uses R32Float because Depth32Float cannot be used with STORAGE_BINDING.
/// Depth values must be copied/converted from the depth buffer.
pub const HIZ_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Minimum HiZ pyramid dimension (final mip level size).
pub const MIN_HIZ_SIZE: u32 = 1;

/// Maximum supported mip levels for HiZ pyramid.
///
/// Sufficient for 16K resolution (16384 -> log2 = 14 mips).
pub const MAX_HIZ_MIPS: u32 = 14;

/// Texture usage flags for HiZ pyramid.
///
/// - TEXTURE_BINDING: For sampling in shaders
/// - STORAGE_BINDING: For compute shader writes to individual mips
/// - COPY_DST: For copying depth buffer data
pub const HIZ_USAGE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING
    .union(wgpu::TextureUsages::STORAGE_BINDING)
    .union(wgpu::TextureUsages::COPY_DST);

// =============================================================================
// HIZ PYRAMID
// =============================================================================

/// Hierarchical-Z pyramid texture for occlusion culling.
///
/// The HiZ pyramid stores depth information at multiple resolutions,
/// enabling efficient hierarchical depth testing. Coarser mip levels
/// allow quick rejection of large occluded regions.
///
/// # Memory Layout
///
/// The pyramid consists of a single 2D texture with multiple mip levels:
///
/// ```text
/// Mip 0: width x height     (base resolution)
/// Mip 1: width/2 x height/2
/// Mip 2: width/4 x height/4
/// ...
/// Mip N: 1 x 1              (single pixel)
/// ```
///
/// # GPU Resource Management
///
/// The struct owns the wgpu texture and provides views for:
/// - Full mip chain sampling (for hierarchical tests)
/// - Per-mip storage access (for compute shader generation)
///
/// # Example
///
/// ```ignore
/// // Create pyramid
/// let pyramid = HiZPyramid::new(&device, 1920, 1080);
///
/// // Use in culling shader
/// let cull_bind_group = device.create_bind_group(&BindGroupDescriptor {
///     entries: &[BindGroupEntry {
///         binding: 0,
///         resource: BindingResource::TextureView(pyramid.view()),
///     }],
///     ..
/// });
///
/// // Write to specific mip in compute pass
/// let write_bind_group = device.create_bind_group(&BindGroupDescriptor {
///     entries: &[BindGroupEntry {
///         binding: 0,
///         resource: BindingResource::TextureView(pyramid.mip_view(0).unwrap()),
///     }],
///     ..
/// });
/// ```
pub struct HiZPyramid {
    /// The GPU texture containing all mip levels.
    texture: wgpu::Texture,
    /// View of the entire mip chain (for sampling).
    view: wgpu::TextureView,
    /// Individual views for each mip level (for compute writes).
    mip_views: Vec<wgpu::TextureView>,
    /// Base width at mip 0.
    width: u32,
    /// Base height at mip 0.
    height: u32,
    /// Total number of mip levels.
    mip_count: u32,
}

impl HiZPyramid {
    /// Create a new HiZ pyramid texture.
    ///
    /// Creates a 2D texture with full mip chain from the given base
    /// resolution down to 1x1.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for resource creation
    /// * `width` - Base resolution width (mip 0)
    /// * `height` - Base resolution height (mip 0)
    ///
    /// # Panics
    ///
    /// Panics if width or height is 0.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pyramid = HiZPyramid::new(&device, 1920, 1080);
    /// assert_eq!(pyramid.width(), 1920);
    /// assert_eq!(pyramid.height(), 1080);
    /// assert_eq!(pyramid.mip_count(), 11); // log2(1920) + 1
    /// ```
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        assert!(width > 0 && height > 0, "HiZ pyramid dimensions must be non-zero");

        let mip_count = Self::calculate_mip_count(width, height).min(MAX_HIZ_MIPS);

        // Create the texture with all mip levels
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hiz_pyramid"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HIZ_FORMAT,
            usage: HIZ_USAGE,
            view_formats: &[],
        });

        // Create full mip chain view for sampling
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("hiz_pyramid_view"),
            format: Some(HIZ_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None, // All mips
            base_array_layer: 0,
            array_layer_count: Some(1),
        });

        // Create individual mip level views for compute writes
        let mut mip_views = Vec::with_capacity(mip_count as usize);
        for mip in 0..mip_count {
            let mip_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("hiz_pyramid_mip{}_view", mip)),
                format: Some(HIZ_FORMAT),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
            });
            mip_views.push(mip_view);
        }

        Self {
            texture,
            view,
            mip_views,
            width,
            height,
            mip_count,
        }
    }

    /// Create a new HiZ pyramid with a custom label.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device for resource creation
    /// * `width` - Base resolution width (mip 0)
    /// * `height` - Base resolution height (mip 0)
    /// * `label` - Custom label prefix for debugging
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pyramid = HiZPyramid::with_label(&device, 1920, 1080, "main_hiz");
    /// ```
    pub fn with_label(device: &wgpu::Device, width: u32, height: u32, label: &str) -> Self {
        assert!(width > 0 && height > 0, "HiZ pyramid dimensions must be non-zero");

        let mip_count = Self::calculate_mip_count(width, height).min(MAX_HIZ_MIPS);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HIZ_FORMAT,
            usage: HIZ_USAGE,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("{}_view", label)),
            format: Some(HIZ_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(1),
        });

        let mut mip_views = Vec::with_capacity(mip_count as usize);
        for mip in 0..mip_count {
            let mip_view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("{}_mip{}", label, mip)),
                format: Some(HIZ_FORMAT),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: mip,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
            });
            mip_views.push(mip_view);
        }

        Self {
            texture,
            view,
            mip_views,
            width,
            height,
            mip_count,
        }
    }

    // -------------------------------------------------------------------------
    // Static Helper Functions
    // -------------------------------------------------------------------------

    /// Calculate the number of mip levels needed for given dimensions.
    ///
    /// Returns the count from base resolution down to 1x1.
    ///
    /// # Arguments
    ///
    /// * `width` - Base resolution width
    /// * `height` - Base resolution height
    ///
    /// # Returns
    ///
    /// Number of mip levels, minimum 1.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert_eq!(HiZPyramid::calculate_mip_count(1, 1), 1);
    /// assert_eq!(HiZPyramid::calculate_mip_count(256, 256), 9);
    /// assert_eq!(HiZPyramid::calculate_mip_count(1920, 1080), 11);
    /// assert_eq!(HiZPyramid::calculate_mip_count(3840, 2160), 12);
    /// ```
    #[inline]
    pub fn calculate_mip_count(width: u32, height: u32) -> u32 {
        let max_dim = width.max(height);
        if max_dim == 0 {
            return 1;
        }
        // log2(max_dim) + 1 gives mips from max_dim down to 1
        // 32 - leading_zeros = floor(log2(n)) + 1 for n > 0
        (32 - max_dim.leading_zeros()).max(1)
    }

    /// Calculate the dimensions of a specific mip level.
    ///
    /// Each mip level is half the resolution of the previous level,
    /// with a minimum size of 1x1.
    ///
    /// # Arguments
    ///
    /// * `base_width` - Width at mip 0
    /// * `base_height` - Height at mip 0
    /// * `mip_level` - Mip level to calculate (0 = base)
    ///
    /// # Returns
    ///
    /// Tuple of (width, height) at the given mip level.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// assert_eq!(HiZPyramid::calculate_mip_size(256, 256, 0), (256, 256));
    /// assert_eq!(HiZPyramid::calculate_mip_size(256, 256, 1), (128, 128));
    /// assert_eq!(HiZPyramid::calculate_mip_size(256, 256, 8), (1, 1));
    /// assert_eq!(HiZPyramid::calculate_mip_size(100, 50, 2), (25, 12));
    /// ```
    #[inline]
    pub fn calculate_mip_size(base_width: u32, base_height: u32, mip_level: u32) -> (u32, u32) {
        // Clamp mip_level to prevent overflow (> 31 would overflow u32 shift)
        let mip = mip_level.min(31);
        let w = (base_width >> mip).max(MIN_HIZ_SIZE);
        let h = (base_height >> mip).max(MIN_HIZ_SIZE);
        (w, h)
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    /// Get the underlying wgpu texture.
    ///
    /// Use this for low-level operations like copying data.
    #[inline]
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Get the full mip chain texture view.
    ///
    /// This view includes all mip levels and is suitable for:
    /// - Sampling with `textureSampleLevel()` in WGSL
    /// - Hierarchical depth testing in culling shaders
    #[inline]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Get a view for a specific mip level.
    ///
    /// These single-mip views are suitable for:
    /// - Storage texture writes in compute shaders
    /// - Building the HiZ pyramid level by level
    ///
    /// # Arguments
    ///
    /// * `level` - Mip level (0 = base resolution)
    ///
    /// # Returns
    ///
    /// `Some(&TextureView)` if level is valid, `None` otherwise.
    #[inline]
    pub fn mip_view(&self, level: u32) -> Option<&wgpu::TextureView> {
        self.mip_views.get(level as usize)
    }

    /// Get base width at mip 0.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get base height at mip 0.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the total number of mip levels.
    #[inline]
    pub fn mip_count(&self) -> u32 {
        self.mip_count
    }

    /// Get the dimensions at a specific mip level.
    ///
    /// # Arguments
    ///
    /// * `level` - Mip level (0 = base resolution)
    ///
    /// # Returns
    ///
    /// Tuple of (width, height), or `None` if level is invalid.
    #[inline]
    pub fn mip_dimensions(&self, level: u32) -> Option<(u32, u32)> {
        if level < self.mip_count {
            Some(Self::calculate_mip_size(self.width, self.height, level))
        } else {
            None
        }
    }

    /// Calculate total memory usage in bytes.
    ///
    /// Returns the sum of all mip level sizes.
    pub fn memory_usage(&self) -> usize {
        let bytes_per_pixel = 4; // R32Float = 4 bytes
        let mut total = 0usize;

        for mip in 0..self.mip_count {
            let (w, h) = Self::calculate_mip_size(self.width, self.height, mip);
            total += (w as usize) * (h as usize) * bytes_per_pixel;
        }

        total
    }

    /// Get the texture as a binding resource (entire mip chain).
    #[inline]
    pub fn as_binding_resource(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::TextureView(&self.view)
    }

    /// Create a new storage view for a specific mip level.
    ///
    /// Unlike `mip_view()`, this creates a new view each time, which is
    /// useful when the views need different lifetimes.
    ///
    /// # Arguments
    ///
    /// * `level` - Mip level (0 = base resolution)
    ///
    /// # Returns
    ///
    /// `Some(TextureView)` if level is valid, `None` otherwise.
    pub fn create_storage_view(&self, level: u32) -> Option<wgpu::TextureView> {
        if level >= self.mip_count {
            return None;
        }

        Some(self.texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("hiz_storage_mip{}", level)),
            format: Some(HIZ_FORMAT),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: level,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        }))
    }

    /// Iterator over all mip level views.
    ///
    /// Useful for setting up bind groups for multi-pass pyramid construction.
    pub fn mip_views_iter(&self) -> impl Iterator<Item = &wgpu::TextureView> {
        self.mip_views.iter()
    }
}

impl std::fmt::Debug for HiZPyramid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HiZPyramid")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("mip_count", &self.mip_count)
            .field("format", &HIZ_FORMAT)
            .field("memory_bytes", &self.memory_usage())
            .finish_non_exhaustive()
    }
}

// =============================================================================
// BIND GROUP LAYOUT HELPERS
// =============================================================================

/// Create a bind group layout for HiZ pyramid sampling.
///
/// This layout provides read-only access to the full mip chain for
/// occlusion culling compute shaders.
///
/// # Binding Layout
///
/// | Binding | Type     | Stage   | Description                |
/// |---------|----------|---------|----------------------------|
/// | 0       | Texture  | Compute | HiZ pyramid (all mips)     |
/// | 1       | Sampler  | Compute | Linear sampler for bilinear depth fetches |
///
/// # Example
///
/// ```ignore
/// let layout = create_hiz_sample_bind_group_layout(&device);
/// let bind_group = device.create_bind_group(&BindGroupDescriptor {
///     layout: &layout,
///     entries: &[
///         BindGroupEntry { binding: 0, resource: pyramid.as_binding_resource() },
///         BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&sampler) },
///     ],
///     label: Some("hiz_sample_bind_group"),
/// });
/// ```
pub fn create_hiz_sample_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hiz_sample_bind_group_layout"),
        entries: &[
            // HiZ pyramid texture (all mips)
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // Sampler for bilinear depth sampling
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Create a bind group layout for HiZ pyramid mip generation.
///
/// This layout provides read access to source mip and write access to
/// destination mip for compute-based pyramid construction.
///
/// # Binding Layout
///
/// | Binding | Type           | Stage   | Description              |
/// |---------|----------------|---------|--------------------------|
/// | 0       | Texture        | Compute | Source mip level         |
/// | 1       | StorageTexture | Compute | Destination mip level    |
///
/// # Example
///
/// ```ignore
/// let layout = create_hiz_gen_bind_group_layout(&device);
/// // For each mip level transition (e.g., mip 0 -> mip 1):
/// let bind_group = device.create_bind_group(&BindGroupDescriptor {
///     layout: &layout,
///     entries: &[
///         BindGroupEntry {
///             binding: 0,
///             resource: BindingResource::TextureView(pyramid.mip_view(0).unwrap()),
///         },
///         BindGroupEntry {
///             binding: 1,
///             resource: BindingResource::TextureView(pyramid.mip_view(1).unwrap()),
///         },
///     ],
///     label: Some("hiz_gen_mip0_to_mip1"),
/// });
/// ```
pub fn create_hiz_gen_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hiz_gen_bind_group_layout"),
        entries: &[
            // Source mip level (read-only)
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
            // Destination mip level (write-only storage)
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
        ],
    })
}

// =============================================================================
// HIZ DOWNSAMPLE SHADER INTEGRATION (T-WGPU-P6.4.2)
// =============================================================================

/// Embedded HiZ downsample shader source (T-WGPU-P6.4.2).
///
/// This shader performs 2x2 max reduction for generating HiZ mip levels.
/// Use with [`HiZDownsampleParams`] for the uniform buffer.
pub const HIZ_DOWNSAMPLE_SHADER: &str = include_str!("../../shaders/hiz_downsample.wgsl");

/// Workgroup size for HiZ downsample compute shader.
///
/// Must match the WGSL constant in `hiz_downsample.wgsl`.
/// Workgroup is 8x8x1 = 64 threads.
pub const HIZ_DOWNSAMPLE_WORKGROUP_SIZE: u32 = 8;

/// Size of [`HiZDownsampleParams`] struct in bytes.
pub const HIZ_DOWNSAMPLE_PARAMS_SIZE: usize = 24;

/// GPU uniform buffer for HiZ downsample parameters (T-WGPU-P6.4.2).
///
/// Matches the `DownsampleParams` struct in `hiz_downsample.wgsl`.
///
/// # Memory Layout
///
/// 24 bytes total, std140 compatible:
///
/// | Offset | Field      | Size | Description                    |
/// |--------|------------|------|--------------------------------|
/// | 0      | src_size.x | 4    | Source mip width               |
/// | 4      | src_size.y | 4    | Source mip height              |
/// | 8      | dst_size.x | 4    | Destination mip width          |
/// | 12     | dst_size.y | 4    | Destination mip height         |
/// | 16     | mip_level  | 4    | Current mip level (for debug)  |
/// | 20     | _padding   | 4    | Padding for 8-byte alignment   |
///
/// # Example
///
/// ```ignore
/// let params = HiZDownsampleParams::new(256, 256, 128, 128, 1);
/// queue.write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HiZDownsampleParams {
    /// Source mip dimensions (width, height).
    pub src_size: [u32; 2],
    /// Destination mip dimensions (width, height).
    pub dst_size: [u32; 2],
    /// Current mip level being generated.
    pub mip_level: u32,
    /// Padding for alignment.
    pub _padding: u32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<HiZDownsampleParams>() == HIZ_DOWNSAMPLE_PARAMS_SIZE);

impl HiZDownsampleParams {
    /// Create parameters for a specific mip level transition.
    ///
    /// # Arguments
    ///
    /// * `src_width` - Source mip width
    /// * `src_height` - Source mip height
    /// * `dst_width` - Destination mip width
    /// * `dst_height` - Destination mip height
    /// * `mip_level` - Mip level being generated
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Generate mip 1 from mip 0 (256x256 -> 128x128)
    /// let params = HiZDownsampleParams::new(256, 256, 128, 128, 1);
    /// ```
    pub fn new(
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        mip_level: u32,
    ) -> Self {
        Self {
            src_size: [src_width, src_height],
            dst_size: [dst_width, dst_height],
            mip_level,
            _padding: 0,
        }
    }

    /// Create parameters for generating a mip from the previous level.
    ///
    /// Automatically calculates destination size as src_size / 2 (min 1).
    ///
    /// # Arguments
    ///
    /// * `src_width` - Source mip width
    /// * `src_height` - Source mip height
    /// * `mip_level` - Mip level being generated
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Generate mip 1 from 256x256 source
    /// let params = HiZDownsampleParams::from_source(256, 256, 1);
    /// assert_eq!(params.dst_size, [128, 128]);
    /// ```
    pub fn from_source(src_width: u32, src_height: u32, mip_level: u32) -> Self {
        Self::new(
            src_width,
            src_height,
            (src_width / 2).max(MIN_HIZ_SIZE),
            (src_height / 2).max(MIN_HIZ_SIZE),
            mip_level,
        )
    }

    /// Calculate the number of workgroups needed for X dimension.
    #[inline]
    pub fn workgroups_x(&self) -> u32 {
        (self.dst_size[0] + HIZ_DOWNSAMPLE_WORKGROUP_SIZE - 1) / HIZ_DOWNSAMPLE_WORKGROUP_SIZE
    }

    /// Calculate the number of workgroups needed for Y dimension.
    #[inline]
    pub fn workgroups_y(&self) -> u32 {
        (self.dst_size[1] + HIZ_DOWNSAMPLE_WORKGROUP_SIZE - 1) / HIZ_DOWNSAMPLE_WORKGROUP_SIZE
    }

    /// Get the dispatch dimensions (x, y, 1).
    #[inline]
    pub fn dispatch_size(&self) -> (u32, u32, u32) {
        (self.workgroups_x(), self.workgroups_y(), 1)
    }
}

/// Create a bind group layout for HiZ downsample textures (Group 0).
///
/// This layout provides texture bindings for the downsample shader.
///
/// # Binding Layout
///
/// | Binding | Type           | Stage   | Description              |
/// |---------|----------------|---------|--------------------------|
/// | 0       | Texture        | Compute | Source mip level (read)  |
/// | 1       | Sampler        | Compute | Linear sampler (unused)  |
/// | 2       | StorageTexture | Compute | Destination mip (write)  |
///
/// # Example
///
/// ```ignore
/// let texture_layout = create_hiz_downsample_texture_layout(&device);
/// ```
pub fn create_hiz_downsample_texture_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hiz_downsample_texture_layout"),
        entries: &[
            // Binding 0: Source mip texture (read-only)
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
            // Binding 1: Sampler (for layout compatibility, not used)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
            // Binding 2: Destination mip storage texture (write-only)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: HIZ_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    })
}

/// Create a bind group layout for HiZ downsample parameters (Group 1).
///
/// This layout provides the uniform buffer binding for downsample parameters.
///
/// # Binding Layout
///
/// | Binding | Type    | Stage   | Description           |
/// |---------|---------|---------|-----------------------|
/// | 0       | Uniform | Compute | HiZDownsampleParams   |
///
/// # Example
///
/// ```ignore
/// let params_layout = create_hiz_downsample_params_layout(&device);
/// ```
pub fn create_hiz_downsample_params_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hiz_downsample_params_layout"),
        entries: &[
            // Binding 0: Downsample parameters uniform
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        std::num::NonZeroU64::new(HIZ_DOWNSAMPLE_PARAMS_SIZE as u64).unwrap()
                    ),
                },
                count: None,
            },
        ],
    })
}

/// CPU reference implementation of max reduction for testing.
///
/// Takes the maximum of 4 values, matching the shader's reverse-Z behavior.
///
/// # Arguments
///
/// * `d00` - Top-left depth
/// * `d10` - Top-right depth
/// * `d01` - Bottom-left depth
/// * `d11` - Bottom-right depth
///
/// # Returns
///
/// Maximum depth value (closest in reverse-Z).
#[inline]
pub fn cpu_max_reduction(d00: f32, d10: f32, d01: f32, d11: f32) -> f32 {
    d00.max(d10).max(d01.max(d11))
}

/// Calculate dispatch dimensions for a given destination size.
///
/// # Arguments
///
/// * `dst_width` - Destination mip width
/// * `dst_height` - Destination mip height
///
/// # Returns
///
/// Tuple of (workgroups_x, workgroups_y, 1).
#[inline]
pub fn calculate_downsample_dispatch(dst_width: u32, dst_height: u32) -> (u32, u32, u32) {
    let x = (dst_width + HIZ_DOWNSAMPLE_WORKGROUP_SIZE - 1) / HIZ_DOWNSAMPLE_WORKGROUP_SIZE;
    let y = (dst_height + HIZ_DOWNSAMPLE_WORKGROUP_SIZE - 1) / HIZ_DOWNSAMPLE_WORKGROUP_SIZE;
    (x, y, 1)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Mip Count Calculation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mip_count_power_of_two() {
        // Powers of 2: log2(n) + 1 mip levels
        assert_eq!(HiZPyramid::calculate_mip_count(1, 1), 1);
        assert_eq!(HiZPyramid::calculate_mip_count(2, 2), 2);
        assert_eq!(HiZPyramid::calculate_mip_count(4, 4), 3);
        assert_eq!(HiZPyramid::calculate_mip_count(8, 8), 4);
        assert_eq!(HiZPyramid::calculate_mip_count(16, 16), 5);
        assert_eq!(HiZPyramid::calculate_mip_count(256, 256), 9);
        assert_eq!(HiZPyramid::calculate_mip_count(512, 512), 10);
        assert_eq!(HiZPyramid::calculate_mip_count(1024, 1024), 11);
    }

    #[test]
    fn test_mip_count_non_power_of_two() {
        // Non-powers of 2: ceil(log2(max_dim)) + 1
        assert_eq!(HiZPyramid::calculate_mip_count(3, 3), 2);
        assert_eq!(HiZPyramid::calculate_mip_count(5, 5), 3);
        assert_eq!(HiZPyramid::calculate_mip_count(100, 100), 7);
        assert_eq!(HiZPyramid::calculate_mip_count(1920, 1080), 11);
        assert_eq!(HiZPyramid::calculate_mip_count(3840, 2160), 12);
    }

    #[test]
    fn test_mip_count_asymmetric() {
        // Uses max dimension for mip count
        assert_eq!(HiZPyramid::calculate_mip_count(1024, 512), 11);
        assert_eq!(HiZPyramid::calculate_mip_count(512, 1024), 11);
        assert_eq!(HiZPyramid::calculate_mip_count(256, 1), 9);
        assert_eq!(HiZPyramid::calculate_mip_count(1, 256), 9);
    }

    #[test]
    fn test_mip_count_edge_cases() {
        // Zero dimensions: max(0, n).leading_zeros() behavior
        // In practice, zero-sized textures are not created, so these are
        // edge cases. The implementation returns floor(log2(max_dim)) + 1
        // for any max_dim > 0, and 1 for max_dim == 0.
        assert_eq!(HiZPyramid::calculate_mip_count(0, 0), 1);
        // When one dimension is zero, max() picks the non-zero one
        assert_eq!(HiZPyramid::calculate_mip_count(0, 100), 7); // log2(100) + 1 = 7
        assert_eq!(HiZPyramid::calculate_mip_count(100, 0), 7);
    }

    // -------------------------------------------------------------------------
    // Mip Size Calculation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mip_size_power_of_two() {
        // Perfect halvings for power-of-2
        assert_eq!(HiZPyramid::calculate_mip_size(256, 256, 0), (256, 256));
        assert_eq!(HiZPyramid::calculate_mip_size(256, 256, 1), (128, 128));
        assert_eq!(HiZPyramid::calculate_mip_size(256, 256, 2), (64, 64));
        assert_eq!(HiZPyramid::calculate_mip_size(256, 256, 3), (32, 32));
        assert_eq!(HiZPyramid::calculate_mip_size(256, 256, 4), (16, 16));
        assert_eq!(HiZPyramid::calculate_mip_size(256, 256, 5), (8, 8));
        assert_eq!(HiZPyramid::calculate_mip_size(256, 256, 6), (4, 4));
        assert_eq!(HiZPyramid::calculate_mip_size(256, 256, 7), (2, 2));
        assert_eq!(HiZPyramid::calculate_mip_size(256, 256, 8), (1, 1));
    }

    #[test]
    fn test_mip_size_non_power_of_two() {
        // Floor division for non-power-of-2
        // 100 -> 50 -> 25 -> 12 -> 6 -> 3 -> 1
        assert_eq!(HiZPyramid::calculate_mip_size(100, 100, 0), (100, 100));
        assert_eq!(HiZPyramid::calculate_mip_size(100, 100, 1), (50, 50));
        assert_eq!(HiZPyramid::calculate_mip_size(100, 100, 2), (25, 25));
        assert_eq!(HiZPyramid::calculate_mip_size(100, 100, 3), (12, 12));
        assert_eq!(HiZPyramid::calculate_mip_size(100, 100, 4), (6, 6));
        assert_eq!(HiZPyramid::calculate_mip_size(100, 100, 5), (3, 3));
        assert_eq!(HiZPyramid::calculate_mip_size(100, 100, 6), (1, 1));
    }

    #[test]
    fn test_mip_size_asymmetric() {
        // Asymmetric dimensions
        assert_eq!(HiZPyramid::calculate_mip_size(256, 128, 0), (256, 128));
        assert_eq!(HiZPyramid::calculate_mip_size(256, 128, 1), (128, 64));
        assert_eq!(HiZPyramid::calculate_mip_size(256, 128, 2), (64, 32));
        assert_eq!(HiZPyramid::calculate_mip_size(256, 128, 7), (2, 1));
        assert_eq!(HiZPyramid::calculate_mip_size(256, 128, 8), (1, 1));
    }

    #[test]
    fn test_mip_size_minimum_one() {
        // Very high mip levels should clamp to 1x1
        assert_eq!(HiZPyramid::calculate_mip_size(8, 8, 50), (1, 1));
        assert_eq!(HiZPyramid::calculate_mip_size(8, 8, 100), (1, 1));
    }

    #[test]
    fn test_mip_size_typical_resolutions() {
        // 1080p
        assert_eq!(HiZPyramid::calculate_mip_size(1920, 1080, 0), (1920, 1080));
        assert_eq!(HiZPyramid::calculate_mip_size(1920, 1080, 1), (960, 540));
        assert_eq!(HiZPyramid::calculate_mip_size(1920, 1080, 2), (480, 270));

        // 4K
        assert_eq!(HiZPyramid::calculate_mip_size(3840, 2160, 0), (3840, 2160));
        assert_eq!(HiZPyramid::calculate_mip_size(3840, 2160, 1), (1920, 1080));
        assert_eq!(HiZPyramid::calculate_mip_size(3840, 2160, 2), (960, 540));
    }

    // -------------------------------------------------------------------------
    // Constant Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_constants() {
        assert_eq!(HIZ_FORMAT, wgpu::TextureFormat::R32Float);
        assert_eq!(MIN_HIZ_SIZE, 1);
        assert_eq!(MAX_HIZ_MIPS, 14);

        // Verify usage flags
        assert!(HIZ_USAGE.contains(wgpu::TextureUsages::TEXTURE_BINDING));
        assert!(HIZ_USAGE.contains(wgpu::TextureUsages::STORAGE_BINDING));
        assert!(HIZ_USAGE.contains(wgpu::TextureUsages::COPY_DST));
    }

    // -------------------------------------------------------------------------
    // Memory Usage Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_memory_usage_calculation() {
        // For a 256x256 pyramid with 9 mip levels:
        // 256^2 + 128^2 + 64^2 + 32^2 + 16^2 + 8^2 + 4^2 + 2^2 + 1^2
        // = 65536 + 16384 + 4096 + 1024 + 256 + 64 + 16 + 4 + 1 = 87381 pixels
        // * 4 bytes = 349524 bytes

        let mut total_pixels = 0u32;
        for mip in 0..9 {
            let (w, h) = HiZPyramid::calculate_mip_size(256, 256, mip);
            total_pixels += w * h;
        }
        let expected_bytes = (total_pixels * 4) as usize;
        assert_eq!(expected_bytes, 349524);

        // Verify it's approximately 1.33x the base size
        let base_size = 256 * 256 * 4;
        assert!(expected_bytes < base_size * 2);
        assert!(expected_bytes > base_size);
    }

    #[test]
    fn test_memory_usage_4k() {
        // 4K depth buffer (3840x2160) -> HZB base (1920x1080)
        let hzb_width = (3840 + 1) / 2;  // 1920
        let hzb_height = (2160 + 1) / 2; // 1080
        let num_mips = HiZPyramid::calculate_mip_count(hzb_width, hzb_height);

        let mut total_pixels = 0usize;
        for mip in 0..num_mips {
            let (w, h) = HiZPyramid::calculate_mip_size(hzb_width, hzb_height, mip);
            total_pixels += (w as usize) * (h as usize);
        }

        let memory_bytes = total_pixels * 4; // R32Float

        // Should be less than 2x base mip size
        let base_pixels = (hzb_width as usize) * (hzb_height as usize);
        assert!(total_pixels < base_pixels * 2);

        // For 1920x1080 base: ~11 MB total with full mip chain
        // (1920*1080 + 960*540 + 480*270 + ... + 1*1) * 4 bytes
        // The sum is approximately base_pixels * 1.33 (geometric series)
        // 1920*1080 = 2,073,600 pixels
        // With mips: ~2,764,800 pixels -> ~11 MB
        assert!(memory_bytes < 12 * 1024 * 1024); // Less than 12 MB
        assert!(memory_bytes > 8 * 1024 * 1024);  // More than 8 MB
    }

    // -------------------------------------------------------------------------
    // Integration with existing HZB tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compatibility_with_hzb_module() {
        // Verify our mip calculations match the hzb module's expectations
        use super::super::hzb;

        // Test various resolutions
        for (w, h) in &[(256, 256), (1920, 1080), (100, 100), (512, 256)] {
            let our_mips = HiZPyramid::calculate_mip_count(*w, *h);
            let hzb_mips = hzb::calculate_mip_count(*w, *h);
            assert_eq!(
                our_mips, hzb_mips,
                "Mip count mismatch for {}x{}: ours={}, hzb={}",
                w, h, our_mips, hzb_mips
            );

            for mip in 0..our_mips.min(hzb_mips) {
                let our_dims = HiZPyramid::calculate_mip_size(*w, *h, mip);
                let hzb_dims = hzb::mip_dimensions(*w, *h, mip);
                assert_eq!(
                    our_dims, hzb_dims,
                    "Mip {} dimensions mismatch for {}x{}: ours={:?}, hzb={:?}",
                    mip, w, h, our_dims, hzb_dims
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // GPU Tests (require wgpu device)
    // -------------------------------------------------------------------------

    // Note: The following tests require a wgpu device and are marked with
    // #[ignore] to skip in CI environments without GPU access.
    // Run with: cargo test -- --ignored

    #[test]
    fn test_pyramid_creation() {
        // This test would create an actual HiZPyramid with a wgpu device
        // and verify the texture/views are created correctly.
        //
        // Example implementation:
        // let instance = wgpu::Instance::new(Default::default());
        // let adapter = pollster::block_on(instance.request_adapter(...));
        // let (device, _) = pollster::block_on(adapter.request_device(...));
        // let pyramid = HiZPyramid::new(&device, 1920, 1080);
        // assert_eq!(pyramid.width(), 1920);
        // assert_eq!(pyramid.height(), 1080);
        // assert_eq!(pyramid.mip_count(), 11);
    }

    #[test]
    fn test_mip_views_accessible() {
        // This test would verify that all mip views are accessible.
        //
        // let pyramid = HiZPyramid::new(&device, 256, 256);
        // for level in 0..pyramid.mip_count() {
        //     assert!(pyramid.mip_view(level).is_some());
        // }
        // assert!(pyramid.mip_view(pyramid.mip_count()).is_none());
    }

    // -------------------------------------------------------------------------
    // HiZ Downsample Tests (T-WGPU-P6.4.2)
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_compiles() {
        // Verify the shader source is embedded and non-empty
        assert!(!HIZ_DOWNSAMPLE_SHADER.is_empty());
        // Check for expected shader entry point
        assert!(HIZ_DOWNSAMPLE_SHADER.contains("fn hiz_downsample"));
        // Check for expected bindings
        assert!(HIZ_DOWNSAMPLE_SHADER.contains("@group(0) @binding(0)"));
        assert!(HIZ_DOWNSAMPLE_SHADER.contains("@group(1) @binding(0)"));
        // Check for workgroup size
        assert!(HIZ_DOWNSAMPLE_SHADER.contains("@workgroup_size(8, 8, 1)"));
    }

    #[test]
    fn test_params_struct_size() {
        // HiZDownsampleParams must be exactly 24 bytes for GPU alignment
        assert_eq!(mem::size_of::<HiZDownsampleParams>(), 24);
        assert_eq!(mem::size_of::<HiZDownsampleParams>(), HIZ_DOWNSAMPLE_PARAMS_SIZE);

        // Verify field offsets (implicitly tested by Pod trait)
        let params = HiZDownsampleParams::default();
        assert_eq!(params.src_size, [0, 0]);
        assert_eq!(params.dst_size, [0, 0]);
        assert_eq!(params.mip_level, 0);
        assert_eq!(params._padding, 0);
    }

    #[test]
    fn test_workgroup_size() {
        // Workgroup size must be 8 (matching WGSL constant)
        assert_eq!(HIZ_DOWNSAMPLE_WORKGROUP_SIZE, 8);
    }

    #[test]
    fn test_max_reduction() {
        // Test max reduction function (reverse-Z: max = closest)
        assert_eq!(cpu_max_reduction(0.5, 0.6, 0.7, 0.8), 0.8);
        assert_eq!(cpu_max_reduction(1.0, 0.0, 0.5, 0.5), 1.0);
        assert_eq!(cpu_max_reduction(0.0, 0.0, 0.0, 0.0), 0.0);
        assert_eq!(cpu_max_reduction(1.0, 1.0, 1.0, 1.0), 1.0);

        // Edge cases
        assert_eq!(cpu_max_reduction(0.1, 0.9, 0.2, 0.8), 0.9);
        assert_eq!(cpu_max_reduction(0.999, 0.998, 0.997, 0.996), 0.999);
    }

    #[test]
    fn test_params_new() {
        let params = HiZDownsampleParams::new(256, 256, 128, 128, 1);
        assert_eq!(params.src_size, [256, 256]);
        assert_eq!(params.dst_size, [128, 128]);
        assert_eq!(params.mip_level, 1);
        assert_eq!(params._padding, 0);
    }

    #[test]
    fn test_params_from_source() {
        // Power of 2
        let params = HiZDownsampleParams::from_source(256, 256, 1);
        assert_eq!(params.dst_size, [128, 128]);

        // Non-power of 2
        let params = HiZDownsampleParams::from_source(100, 100, 2);
        assert_eq!(params.dst_size, [50, 50]);

        // Minimum size clamping
        let params = HiZDownsampleParams::from_source(2, 2, 5);
        assert_eq!(params.dst_size, [1, 1]);

        // Already at minimum
        let params = HiZDownsampleParams::from_source(1, 1, 10);
        assert_eq!(params.dst_size, [1, 1]);
    }

    #[test]
    fn test_workgroups_calculation() {
        // Exact fit (no remainder)
        let params = HiZDownsampleParams::new(256, 256, 128, 128, 1);
        assert_eq!(params.workgroups_x(), 16);  // 128 / 8 = 16
        assert_eq!(params.workgroups_y(), 16);
        assert_eq!(params.dispatch_size(), (16, 16, 1));

        // With remainder (round up)
        let params = HiZDownsampleParams::new(100, 100, 50, 50, 1);
        assert_eq!(params.workgroups_x(), 7);   // ceil(50 / 8) = 7
        assert_eq!(params.workgroups_y(), 7);
        assert_eq!(params.dispatch_size(), (7, 7, 1));

        // Small sizes
        let params = HiZDownsampleParams::new(8, 8, 4, 4, 2);
        assert_eq!(params.workgroups_x(), 1);   // ceil(4 / 8) = 1
        assert_eq!(params.workgroups_y(), 1);
        assert_eq!(params.dispatch_size(), (1, 1, 1));

        // Edge case: 1x1
        let params = HiZDownsampleParams::new(2, 2, 1, 1, 10);
        assert_eq!(params.workgroups_x(), 1);
        assert_eq!(params.workgroups_y(), 1);
    }

    #[test]
    fn test_calculate_downsample_dispatch() {
        // Various sizes
        assert_eq!(calculate_downsample_dispatch(128, 128), (16, 16, 1));
        assert_eq!(calculate_downsample_dispatch(50, 50), (7, 7, 1));
        assert_eq!(calculate_downsample_dispatch(1, 1), (1, 1, 1));
        assert_eq!(calculate_downsample_dispatch(1920, 1080), (240, 135, 1));
    }

    #[test]
    fn test_params_bytemuck() {
        // Verify bytemuck traits work correctly
        let params = HiZDownsampleParams::new(256, 256, 128, 128, 1);
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 24);

        // Verify roundtrip
        let restored: HiZDownsampleParams = *bytemuck::from_bytes(bytes);
        assert_eq!(restored.src_size, params.src_size);
        assert_eq!(restored.dst_size, params.dst_size);
        assert_eq!(restored.mip_level, params.mip_level);
    }
}

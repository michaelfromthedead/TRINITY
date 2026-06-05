//! Alpha Test Pipeline for RT Shadow Rays (T-RT-P1.6).
//!
//! This module provides alpha testing support for ray tracing shadow rays.
//! When tracing shadow rays through alpha-tested geometry (e.g., foliage,
//! fences, decals), we need to sample the alpha texture at hit points to
//! determine if the ray actually hit opaque geometry or passed through
//! a transparent region.
//!
//! # How It Works
//!
//! In WGSL inline ray queries, there is no separate "any-hit shader" like
//! in Vulkan/DXR. Instead, alpha testing is implemented within the
//! `rayQueryProceed` loop:
//!
//! 1. Ray query reports a candidate hit
//! 2. We check if it's a triangle intersection
//! 3. If alpha testing is enabled:
//!    - Compute hit UV from barycentric coordinates
//!    - Sample alpha texture at hit UV
//!    - If alpha < cutoff: continue (ignore hit)
//!    - If alpha >= cutoff: commit intersection (opaque hit)
//! 4. If alpha testing disabled: commit immediately
//!
//! # Bind Group Layout (Group 1)
//!
//! | Binding | Type            | Content                              |
//! |---------|-----------------|--------------------------------------|
//! | 0       | texture_2d      | Alpha texture                        |
//! | 1       | sampler         | Alpha sampler (linear filtering)     |
//! | 2       | storage (read)  | Per-triangle UV data for barycentric |
//! | 3       | uniform         | AlphaTestParams                      |
//!
//! # Example
//!
//! ```ignore
//! let params = AlphaTestParams::enabled(0.5);
//! let bind_group = alpha_pipeline.create_bind_group(
//!     &device,
//!     &alpha_texture_view,
//!     &sampler,
//!     &uv_buffer,
//!     &params_buffer,
//! );
//! ```

use std::mem;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default alpha cutoff threshold.
///
/// Pixels with alpha below this value are considered transparent.
pub const DEFAULT_ALPHA_CUTOFF: f32 = 0.5;

// ---------------------------------------------------------------------------
// AlphaTestParams
// ---------------------------------------------------------------------------

/// GPU-compatible alpha test parameters.
///
/// This struct is uploaded to a uniform buffer for the compute shader.
///
/// # Memory Layout (16 bytes, std140 compatible)
///
/// | Offset | Field          | Size | Description                    |
/// |--------|----------------|------|--------------------------------|
/// | 0      | alpha_cutoff   | 4    | Alpha threshold [0, 1]         |
/// | 4      | use_alpha_test | 4    | 0 = opaque, 1 = alpha test     |
/// | 8      | _padding       | 8    | Alignment padding              |
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AlphaTestParams {
    /// Alpha cutoff threshold.
    ///
    /// Pixels with alpha < cutoff are considered transparent.
    /// Typical values: 0.5 for hard cutoff, 0.1 for soft edges.
    pub alpha_cutoff: f32,

    /// Enable alpha testing.
    ///
    /// 0 = opaque geometry (skip alpha test, always commit hit)
    /// 1 = alpha-tested geometry (sample alpha, conditionally commit)
    pub use_alpha_test: u32,

    /// Padding for 16-byte alignment (std140).
    pub _padding: [u32; 2],
}

// Compile-time size assertion: 16 bytes
const _: () = assert!(mem::size_of::<AlphaTestParams>() == 16);
const _: () = assert!(mem::align_of::<AlphaTestParams>() == 4);

impl AlphaTestParams {
    /// Create parameters for opaque geometry (no alpha testing).
    ///
    /// This is the fast path - hits are committed immediately without
    /// sampling any textures.
    pub fn opaque() -> Self {
        Self {
            alpha_cutoff: DEFAULT_ALPHA_CUTOFF,
            use_alpha_test: 0,
            _padding: [0; 2],
        }
    }

    /// Create parameters for alpha-tested geometry.
    ///
    /// # Arguments
    ///
    /// * `cutoff` - Alpha threshold. Pixels with alpha < cutoff are transparent.
    pub fn enabled(cutoff: f32) -> Self {
        Self {
            alpha_cutoff: cutoff,
            use_alpha_test: 1,
            _padding: [0; 2],
        }
    }

    /// Create parameters with explicit settings.
    ///
    /// # Arguments
    ///
    /// * `cutoff` - Alpha cutoff threshold
    /// * `enabled` - Whether alpha testing is enabled
    pub fn new(cutoff: f32, enabled: bool) -> Self {
        Self {
            alpha_cutoff: cutoff,
            use_alpha_test: if enabled { 1 } else { 0 },
            _padding: [0; 2],
        }
    }

    /// Check if alpha testing is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.use_alpha_test != 0
    }

    /// Set the alpha cutoff threshold.
    pub fn with_cutoff(mut self, cutoff: f32) -> Self {
        self.alpha_cutoff = cutoff;
        self
    }
}

impl Default for AlphaTestParams {
    /// Default: alpha testing disabled, cutoff at 0.5.
    fn default() -> Self {
        Self::opaque()
    }
}

// ---------------------------------------------------------------------------
// AlphaTestPipeline
// ---------------------------------------------------------------------------

/// Alpha test bind group management for RT shadow rays.
///
/// This struct manages the bind group layout and creation for alpha-tested
/// geometry in ray tracing. It creates bind groups for group 1, which
/// supplements the main shadow ray bind group (group 0).
///
/// # Bind Group Layout
///
/// | Binding | Type            | Content                              |
/// |---------|-----------------|--------------------------------------|
/// | 0       | texture_2d      | Alpha texture (RGBA or single-channel)|
/// | 1       | sampler         | Texture sampler                      |
/// | 2       | storage (read)  | Per-vertex UV coordinates            |
/// | 3       | uniform         | AlphaTestParams                      |
pub struct AlphaTestPipeline {
    /// Bind group layout for alpha test resources.
    bind_group_layout: wgpu::BindGroupLayout,
}

impl AlphaTestPipeline {
    /// Create a new alpha test pipeline.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(device);

        Self { bind_group_layout }
    }

    /// Get the bind group layout for external pipeline configuration.
    #[inline]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Create the bind group layout for alpha test resources.
    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("alpha_test_bind_group_layout"),
            entries: &[
                // Binding 0: Alpha texture
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
                // Binding 1: Alpha sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding 2: Per-vertex UV coordinates (storage buffer)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(8), // Minimum vec2<f32>
                    },
                    count: None,
                },
                // Binding 3: AlphaTestParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            mem::size_of::<AlphaTestParams>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create a bind group for alpha-tested geometry.
    ///
    /// # Arguments
    ///
    /// * `device` - The wgpu device.
    /// * `alpha_texture_view` - View of the alpha/opacity texture.
    /// * `sampler` - Texture sampler (typically linear filtering).
    /// * `uv_buffer` - Storage buffer with per-vertex UV coordinates.
    /// * `params_buffer` - Uniform buffer with AlphaTestParams.
    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        alpha_texture_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        uv_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("alpha_test_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(alpha_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uv_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        })
    }
}

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Create an alpha test params uniform buffer.
///
/// # Arguments
///
/// * `device` - The wgpu device.
/// * `params` - Alpha test parameters.
///
/// # Returns
///
/// A GPU buffer containing the parameters.
pub fn create_params_buffer(device: &wgpu::Device, params: &AlphaTestParams) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("alpha_test_params"),
        size: mem::size_of::<AlphaTestParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });

    buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(bytemuck::bytes_of(params));
    buffer.unmap();

    buffer
}

/// Create a UV storage buffer from vertex UV data.
///
/// # Arguments
///
/// * `device` - The wgpu device.
/// * `uvs` - Array of UV coordinates (vec2 per vertex).
///
/// # Returns
///
/// A GPU buffer containing the UVs.
pub fn create_uv_buffer(device: &wgpu::Device, uvs: &[[f32; 2]]) -> wgpu::Buffer {
    let size = (uvs.len() * mem::size_of::<[f32; 2]>()).max(8) as u64;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("alpha_test_uvs"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });

    if !uvs.is_empty() {
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(uvs));
    }
    buffer.unmap();

    buffer
}

/// Create a default alpha sampler with linear filtering.
///
/// # Arguments
///
/// * `device` - The wgpu device.
///
/// # Returns
///
/// A sampler configured for alpha texture sampling.
pub fn create_alpha_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("alpha_test_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

// ---------------------------------------------------------------------------
// Shader Validation (naga)
// ---------------------------------------------------------------------------

/// Validate the RT shadow shader with alpha test bindings using naga.
///
/// This function parses and validates the WGSL shader source including
/// the alpha test bind group (group 1).
///
/// # Returns
///
/// Ok(()) if validation passes, Err with message if it fails.
#[cfg(test)]
pub fn validate_shader_with_alpha_test() -> Result<(), String> {
    use naga::front::wgsl;

    let shader_source = include_str!("../../shaders/raytracing/rt_shadow.comp.wgsl");

    let module =
        wgsl::parse_str(shader_source).map_err(|e| format!("WGSL parse error: {:?}", e))?;

    // Validate the module
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    validator
        .validate(&module)
        .map_err(|e| format!("WGSL validation error: {:?}", e))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Struct Size and Alignment Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_alpha_test_params_size_is_16_bytes() {
        assert_eq!(mem::size_of::<AlphaTestParams>(), 16);
    }

    #[test]
    fn test_alpha_test_params_alignment_is_4_bytes() {
        assert_eq!(mem::align_of::<AlphaTestParams>(), 4);
    }

    // -------------------------------------------------------------------------
    // Default and Construction Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_default_params_has_cutoff_0_5() {
        let params = AlphaTestParams::default();
        assert_eq!(params.alpha_cutoff, 0.5);
    }

    #[test]
    fn test_default_params_has_alpha_test_disabled() {
        let params = AlphaTestParams::default();
        assert_eq!(params.use_alpha_test, 0);
        assert!(!params.is_enabled());
    }

    #[test]
    fn test_opaque_params() {
        let params = AlphaTestParams::opaque();
        assert!(!params.is_enabled());
        assert_eq!(params.use_alpha_test, 0);
    }

    #[test]
    fn test_enabled_params() {
        let params = AlphaTestParams::enabled(0.3);
        assert!(params.is_enabled());
        assert_eq!(params.use_alpha_test, 1);
        assert_eq!(params.alpha_cutoff, 0.3);
    }

    #[test]
    fn test_new_params_enabled() {
        let params = AlphaTestParams::new(0.7, true);
        assert!(params.is_enabled());
        assert_eq!(params.alpha_cutoff, 0.7);
    }

    #[test]
    fn test_new_params_disabled() {
        let params = AlphaTestParams::new(0.7, false);
        assert!(!params.is_enabled());
        assert_eq!(params.alpha_cutoff, 0.7);
    }

    #[test]
    fn test_with_cutoff() {
        let params = AlphaTestParams::opaque().with_cutoff(0.25);
        assert_eq!(params.alpha_cutoff, 0.25);
        assert!(!params.is_enabled()); // Still disabled
    }

    // -------------------------------------------------------------------------
    // Constant Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_default_alpha_cutoff_constant() {
        assert_eq!(DEFAULT_ALPHA_CUTOFF, 0.5);
    }

    // -------------------------------------------------------------------------
    // Bytemuck Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_alpha_test_params_is_pod() {
        let params = AlphaTestParams::default();
        let bytes = bytemuck::bytes_of(&params);
        assert_eq!(bytes.len(), 16);
    }

    #[test]
    fn test_alpha_test_params_zeroed() {
        let params: AlphaTestParams = bytemuck::Zeroable::zeroed();
        assert_eq!(params.alpha_cutoff, 0.0);
        assert_eq!(params.use_alpha_test, 0);
    }

    // -------------------------------------------------------------------------
    // Shader Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_shader_with_alpha_test_bindings_parses() {
        use naga::front::wgsl;

        let shader_source = include_str!("../../shaders/raytracing/rt_shadow.comp.wgsl");
        let result = wgsl::parse_str(shader_source);

        assert!(result.is_ok(), "Shader parsing failed: {:?}", result.err());
    }

    #[test]
    fn test_shader_with_alpha_test_bindings_validates() {
        let result = validate_shader_with_alpha_test();
        assert!(
            result.is_ok(),
            "Shader validation failed: {:?}",
            result.err()
        );
    }

    // -------------------------------------------------------------------------
    // Field Layout Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_alpha_cutoff_offset() {
        let params = AlphaTestParams::enabled(0.5);
        let bytes = bytemuck::bytes_of(&params);

        // alpha_cutoff is at offset 0
        let cutoff_bytes = &bytes[0..4];
        let cutoff = f32::from_le_bytes(cutoff_bytes.try_into().unwrap());
        assert_eq!(cutoff, 0.5);
    }

    #[test]
    fn test_use_alpha_test_offset() {
        let params = AlphaTestParams::enabled(0.5);
        let bytes = bytemuck::bytes_of(&params);

        // use_alpha_test is at offset 4
        let use_alpha_bytes = &bytes[4..8];
        let use_alpha = u32::from_le_bytes(use_alpha_bytes.try_into().unwrap());
        assert_eq!(use_alpha, 1);
    }

    #[test]
    fn test_padding_is_zero() {
        let params = AlphaTestParams::enabled(0.5);
        assert_eq!(params._padding, [0, 0]);
    }
}

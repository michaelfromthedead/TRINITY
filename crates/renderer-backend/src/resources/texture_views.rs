//! Texture view creation and management for TRINITY.
//!
//! This module provides texture view creation with convenient methods for common
//! view patterns including mip level selection, array layer selection, cubemap
//! face access, and format/aspect reinterpretation.
//!
//! # Overview
//!
//! Texture views allow accessing a texture with different parameters:
//!
//! - **Format reinterpretation**: Access sRGB texture as linear, or vice versa
//! - **Mip level range**: Access specific mip levels for rendering or sampling
//! - **Array layer range**: Access specific layers in texture arrays
//! - **Dimension reinterpretation**: View 2D array as cubemap
//! - **Aspect selection**: Access depth or stencil aspect separately
//!
//! # View Dimension Compatibility
//!
//! | Texture Dimension | Valid View Dimensions |
//! |-------------------|----------------------|
//! | D1 | D1 |
//! | D2 (1 layer) | D2 |
//! | D2 (>1 layers) | D2, D2Array, Cube*, CubeArray* |
//! | D3 | D3 |
//!
//! *Cube requires 6 layers, CubeArray requires multiple of 6 layers
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::{TrinityTexture, TrinityTextureDescriptor, create_texture};
//! use renderer_backend::resources::texture_views::{TrinityTextureViewDescriptor, CubeFace};
//! use wgpu::{TextureFormat, TextureAspect, TextureViewDimension};
//!
//! # fn example(device: &wgpu::Device, texture: &TrinityTexture) {
//! // Create a view for mip level 2 only
//! let mip_view = texture.create_mip_view(2);
//!
//! // Create a view for cubemap face +X
//! let face_view = texture.create_cube_face_view(CubeFace::PosX);
//!
//! // Create a depth-only view for shadow sampling
//! let depth_view = texture.create_depth_only_view();
//!
//! // Create a custom view with full control
//! let custom_view = texture.create_trinity_view(&TrinityTextureViewDescriptor {
//!     label: Some("custom_view"),
//!     format: Some(TextureFormat::Rgba8Unorm), // Reinterpret format
//!     dimension: Some(TextureViewDimension::D2),
//!     aspect: TextureAspect::All,
//!     base_mip_level: 1,
//!     mip_level_count: Some(3),
//!     base_array_layer: 0,
//!     array_layer_count: None,
//! });
//! # }
//! ```

use wgpu::{TextureAspect, TextureDimension, TextureFormat, TextureView, TextureViewDimension};

use super::TrinityTexture;

// ============================================================================
// CubeFace Enum
// ============================================================================

/// Represents a face of a cubemap texture.
///
/// Cubemaps are stored as 2D texture arrays with 6 layers, one for each face.
/// The faces are ordered according to the wgpu/WebGPU specification.
///
/// # Layer Mapping
///
/// | Face | Layer Index | Direction |
/// |------|-------------|-----------|
/// | PosX | 0 | +X (right) |
/// | NegX | 1 | -X (left) |
/// | PosY | 2 | +Y (up) |
/// | NegY | 3 | -Y (down) |
/// | PosZ | 4 | +Z (front) |
/// | NegZ | 5 | -Z (back) |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum CubeFace {
    /// Positive X face (+X, right)
    PosX = 0,
    /// Negative X face (-X, left)
    NegX = 1,
    /// Positive Y face (+Y, up)
    PosY = 2,
    /// Negative Y face (-Y, down)
    NegY = 3,
    /// Positive Z face (+Z, front)
    PosZ = 4,
    /// Negative Z face (-Z, back)
    NegZ = 5,
}

impl CubeFace {
    /// Returns the array layer index for this cube face.
    #[inline]
    pub const fn layer_index(self) -> u32 {
        self as u32
    }

    /// Returns all cube faces in order.
    pub const fn all() -> [CubeFace; 6] {
        [
            CubeFace::PosX,
            CubeFace::NegX,
            CubeFace::PosY,
            CubeFace::NegY,
            CubeFace::PosZ,
            CubeFace::NegZ,
        ]
    }

    /// Creates a CubeFace from a layer index.
    ///
    /// Returns `None` if the index is >= 6.
    pub const fn from_layer_index(index: u32) -> Option<CubeFace> {
        match index {
            0 => Some(CubeFace::PosX),
            1 => Some(CubeFace::NegX),
            2 => Some(CubeFace::PosY),
            3 => Some(CubeFace::NegY),
            4 => Some(CubeFace::PosZ),
            5 => Some(CubeFace::NegZ),
            _ => None,
        }
    }

    /// Returns a human-readable name for this face.
    pub const fn name(self) -> &'static str {
        match self {
            CubeFace::PosX => "+X",
            CubeFace::NegX => "-X",
            CubeFace::PosY => "+Y",
            CubeFace::NegY => "-Y",
            CubeFace::PosZ => "+Z",
            CubeFace::NegZ => "-Z",
        }
    }
}

// ============================================================================
// TrinityTextureViewDescriptor
// ============================================================================

/// Descriptor for creating texture views with TRINITY extensions.
///
/// This descriptor provides a more ergonomic interface than the raw wgpu
/// [`TextureViewDescriptor`](wgpu::TextureViewDescriptor), with validation
/// helpers and sensible defaults.
///
/// # Defaults
///
/// - `label`: None
/// - `format`: None (inherit from texture)
/// - `dimension`: None (inherit from texture)
/// - `aspect`: All
/// - `base_mip_level`: 0
/// - `mip_level_count`: None (all remaining mips)
/// - `base_array_layer`: 0
/// - `array_layer_count`: None (all remaining layers)
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::texture_views::TrinityTextureViewDescriptor;
/// use wgpu::{TextureAspect, TextureViewDimension, TextureFormat};
///
/// // View for rendering to mip level 1
/// let render_view = TrinityTextureViewDescriptor {
///     label: Some("render_target_mip1"),
///     base_mip_level: 1,
///     mip_level_count: Some(1),
///     ..Default::default()
/// };
///
/// // View for sRGB reinterpretation
/// let srgb_view = TrinityTextureViewDescriptor {
///     label: Some("srgb_view"),
///     format: Some(TextureFormat::Rgba8UnormSrgb),
///     ..Default::default()
/// };
///
/// // Depth-only view for shadow sampling
/// let depth_view = TrinityTextureViewDescriptor {
///     label: Some("depth_sample"),
///     aspect: TextureAspect::DepthOnly,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct TrinityTextureViewDescriptor<'a> {
    /// Debug label for the view.
    ///
    /// Appears in GPU debugging tools and error messages.
    pub label: Option<&'a str>,

    /// Override format for the view.
    ///
    /// Must be compatible with the texture's format (same block size).
    /// Common use: `Rgba8UnormSrgb` texture viewed as `Rgba8Unorm` for
    /// linear color access, or vice versa.
    ///
    /// If `None`, inherits the texture's format.
    pub format: Option<TextureFormat>,

    /// Override dimension for the view.
    ///
    /// Allows dimension reinterpretation (e.g., viewing a 2D array as a cubemap).
    /// Must be compatible with the texture's dimension and layer count.
    ///
    /// If `None`, uses the texture's native view dimension.
    pub dimension: Option<TextureViewDimension>,

    /// Which aspect(s) of the texture to include in the view.
    ///
    /// - `All`: Both depth and stencil (or color for color textures)
    /// - `DepthOnly`: Only the depth aspect (for depth-stencil textures)
    /// - `StencilOnly`: Only the stencil aspect (for depth-stencil textures)
    pub aspect: TextureAspect,

    /// First mip level accessible from this view.
    ///
    /// Must be less than the texture's mip level count.
    pub base_mip_level: u32,

    /// Number of mip levels accessible from this view.
    ///
    /// If `None`, includes all mip levels from `base_mip_level` to the end.
    /// If `Some(n)`, includes exactly `n` mip levels starting at `base_mip_level`.
    pub mip_level_count: Option<u32>,

    /// First array layer accessible from this view.
    ///
    /// Must be less than the texture's array layer count.
    pub base_array_layer: u32,

    /// Number of array layers accessible from this view.
    ///
    /// If `None`, includes all layers from `base_array_layer` to the end.
    /// If `Some(n)`, includes exactly `n` layers starting at `base_array_layer`.
    ///
    /// For cubemaps, must be 6 (or None to include all 6).
    /// For cubemap arrays, must be a multiple of 6.
    pub array_layer_count: Option<u32>,
}

impl Default for TrinityTextureViewDescriptor<'_> {
    fn default() -> Self {
        Self {
            label: None,
            format: None,
            dimension: None,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        }
    }
}

impl<'a> TrinityTextureViewDescriptor<'a> {
    /// Creates a descriptor for viewing a single mip level.
    ///
    /// # Arguments
    ///
    /// * `mip_level` - The mip level to view
    /// * `label` - Optional debug label
    pub fn single_mip(mip_level: u32, label: Option<&'a str>) -> Self {
        Self {
            label,
            base_mip_level: mip_level,
            mip_level_count: Some(1),
            ..Default::default()
        }
    }

    /// Creates a descriptor for viewing a single array layer.
    ///
    /// # Arguments
    ///
    /// * `layer` - The array layer to view
    /// * `label` - Optional debug label
    pub fn single_layer(layer: u32, label: Option<&'a str>) -> Self {
        Self {
            label,
            base_array_layer: layer,
            array_layer_count: Some(1),
            dimension: Some(TextureViewDimension::D2),
            ..Default::default()
        }
    }

    /// Creates a descriptor for depth-only access.
    ///
    /// # Arguments
    ///
    /// * `label` - Optional debug label
    pub fn depth_only(label: Option<&'a str>) -> Self {
        Self {
            label,
            aspect: TextureAspect::DepthOnly,
            ..Default::default()
        }
    }

    /// Creates a descriptor for stencil-only access.
    ///
    /// # Arguments
    ///
    /// * `label` - Optional debug label
    pub fn stencil_only(label: Option<&'a str>) -> Self {
        Self {
            label,
            aspect: TextureAspect::StencilOnly,
            ..Default::default()
        }
    }

    /// Creates a descriptor for cubemap view of a 2D array texture.
    ///
    /// The texture must have at least 6 array layers.
    ///
    /// # Arguments
    ///
    /// * `label` - Optional debug label
    pub fn as_cube(label: Option<&'a str>) -> Self {
        Self {
            label,
            dimension: Some(TextureViewDimension::Cube),
            array_layer_count: Some(6),
            ..Default::default()
        }
    }

    /// Creates a descriptor for a single cubemap face.
    ///
    /// # Arguments
    ///
    /// * `face` - The cube face to view
    /// * `label` - Optional debug label
    pub fn cube_face(face: CubeFace, label: Option<&'a str>) -> Self {
        Self {
            label,
            dimension: Some(TextureViewDimension::D2),
            base_array_layer: face.layer_index(),
            array_layer_count: Some(1),
            ..Default::default()
        }
    }

    /// Creates a descriptor with format reinterpretation.
    ///
    /// # Arguments
    ///
    /// * `format` - The format to reinterpret as
    /// * `label` - Optional debug label
    pub fn with_format(format: TextureFormat, label: Option<&'a str>) -> Self {
        Self {
            label,
            format: Some(format),
            ..Default::default()
        }
    }

    /// Converts this descriptor to a wgpu TextureViewDescriptor.
    pub fn to_wgpu(&self) -> wgpu::TextureViewDescriptor<'a> {
        wgpu::TextureViewDescriptor {
            label: self.label,
            format: self.format,
            dimension: self.dimension,
            aspect: self.aspect,
            base_mip_level: self.base_mip_level,
            mip_level_count: self.mip_level_count,
            base_array_layer: self.base_array_layer,
            array_layer_count: self.array_layer_count,
        }
    }
}

// ============================================================================
// Validation Helpers
// ============================================================================

/// Validates that a view dimension is compatible with a texture dimension.
///
/// # Arguments
///
/// * `texture_dim` - The texture's dimension
/// * `view_dim` - The requested view dimension
/// * `array_layers` - Number of array layers in the texture
///
/// # Returns
///
/// `true` if the view dimension is valid for the texture.
///
/// # Valid Combinations
///
/// | Texture | View | Requirements |
/// |---------|------|--------------|
/// | D1 | D1 | - |
/// | D2 | D2 | array_layers == 1 |
/// | D2 | D2Array | array_layers >= 1 |
/// | D2 | Cube | array_layers == 6 |
/// | D2 | CubeArray | array_layers % 6 == 0, array_layers >= 6 |
/// | D3 | D3 | - |
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture_views::validate_view_dimensions;
/// use wgpu::{TextureDimension, TextureViewDimension};
///
/// // 2D texture can be viewed as 2D
/// assert!(validate_view_dimensions(TextureDimension::D2, TextureViewDimension::D2, 1));
///
/// // 2D array with 6 layers can be viewed as Cube
/// assert!(validate_view_dimensions(TextureDimension::D2, TextureViewDimension::Cube, 6));
///
/// // 2D texture cannot be viewed as Cube (needs 6 layers)
/// assert!(!validate_view_dimensions(TextureDimension::D2, TextureViewDimension::Cube, 1));
/// ```
pub fn validate_view_dimensions(
    texture_dim: TextureDimension,
    view_dim: TextureViewDimension,
    array_layers: u32,
) -> bool {
    match texture_dim {
        TextureDimension::D1 => matches!(view_dim, TextureViewDimension::D1),
        TextureDimension::D2 => match view_dim {
            TextureViewDimension::D1 => false,
            TextureViewDimension::D2 => true, // Single layer or array, both valid as D2
            TextureViewDimension::D2Array => array_layers >= 1,
            TextureViewDimension::Cube => array_layers >= 6,
            TextureViewDimension::CubeArray => array_layers >= 6 && array_layers % 6 == 0,
            TextureViewDimension::D3 => false,
        },
        TextureDimension::D3 => matches!(view_dim, TextureViewDimension::D3),
    }
}

/// Validates that a mip level range is within bounds.
///
/// # Arguments
///
/// * `texture_mips` - Total number of mip levels in the texture
/// * `base_mip` - First mip level in the range
/// * `count` - Number of mip levels (None = all remaining)
///
/// # Returns
///
/// `true` if the range is valid.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture_views::validate_mip_range;
///
/// // Valid: mips 0-4 of a 5-mip texture
/// assert!(validate_mip_range(5, 0, Some(5)));
///
/// // Valid: all remaining mips from level 2
/// assert!(validate_mip_range(5, 2, None));
///
/// // Invalid: mip 5 doesn't exist in a 5-mip texture
/// assert!(!validate_mip_range(5, 5, Some(1)));
///
/// // Invalid: requesting more mips than available
/// assert!(!validate_mip_range(5, 3, Some(5)));
/// ```
pub fn validate_mip_range(texture_mips: u32, base_mip: u32, count: Option<u32>) -> bool {
    if base_mip >= texture_mips {
        return false;
    }

    match count {
        None => true, // All remaining mips is always valid if base is valid
        Some(n) => {
            if n == 0 {
                return false; // Zero-count range is invalid
            }
            base_mip.saturating_add(n) <= texture_mips
        }
    }
}

/// Validates that an array layer range is within bounds.
///
/// # Arguments
///
/// * `texture_layers` - Total number of array layers in the texture
/// * `base_layer` - First array layer in the range
/// * `count` - Number of array layers (None = all remaining)
///
/// # Returns
///
/// `true` if the range is valid.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture_views::validate_array_range;
///
/// // Valid: layers 0-5 of a 6-layer texture (cubemap)
/// assert!(validate_array_range(6, 0, Some(6)));
///
/// // Valid: all remaining layers from layer 2
/// assert!(validate_array_range(6, 2, None));
///
/// // Invalid: layer 6 doesn't exist in a 6-layer texture
/// assert!(!validate_array_range(6, 6, Some(1)));
///
/// // Invalid: requesting more layers than available
/// assert!(!validate_array_range(6, 4, Some(5)));
/// ```
pub fn validate_array_range(texture_layers: u32, base_layer: u32, count: Option<u32>) -> bool {
    if base_layer >= texture_layers {
        return false;
    }

    match count {
        None => true, // All remaining layers is always valid if base is valid
        Some(n) => {
            if n == 0 {
                return false; // Zero-count range is invalid
            }
            base_layer.saturating_add(n) <= texture_layers
        }
    }
}

/// Checks if a format is a depth-stencil format.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture_views::is_depth_stencil_format;
/// use wgpu::TextureFormat;
///
/// assert!(is_depth_stencil_format(TextureFormat::Depth32Float));
/// assert!(is_depth_stencil_format(TextureFormat::Depth24PlusStencil8));
/// assert!(!is_depth_stencil_format(TextureFormat::Rgba8Unorm));
/// ```
pub const fn is_depth_stencil_format(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Depth16Unorm
            | TextureFormat::Depth24Plus
            | TextureFormat::Depth24PlusStencil8
            | TextureFormat::Depth32Float
            | TextureFormat::Depth32FloatStencil8
            | TextureFormat::Stencil8
    )
}

/// Checks if a format has a stencil component.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture_views::has_stencil_component;
/// use wgpu::TextureFormat;
///
/// assert!(has_stencil_component(TextureFormat::Depth24PlusStencil8));
/// assert!(has_stencil_component(TextureFormat::Stencil8));
/// assert!(!has_stencil_component(TextureFormat::Depth32Float));
/// ```
pub const fn has_stencil_component(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Depth24PlusStencil8
            | TextureFormat::Depth32FloatStencil8
            | TextureFormat::Stencil8
    )
}

/// Checks if a format has a depth component.
///
/// # Example
///
/// ```
/// use renderer_backend::resources::texture_views::has_depth_component;
/// use wgpu::TextureFormat;
///
/// assert!(has_depth_component(TextureFormat::Depth32Float));
/// assert!(has_depth_component(TextureFormat::Depth24PlusStencil8));
/// assert!(!has_depth_component(TextureFormat::Stencil8));
/// ```
pub const fn has_depth_component(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Depth16Unorm
            | TextureFormat::Depth24Plus
            | TextureFormat::Depth24PlusStencil8
            | TextureFormat::Depth32Float
            | TextureFormat::Depth32FloatStencil8
    )
}

/// Returns the native view dimension for a texture dimension and layer count.
///
/// # Arguments
///
/// * `texture_dim` - The texture's dimension
/// * `array_layers` - Number of array layers
///
/// # Returns
///
/// The default view dimension for sampling the texture.
pub const fn native_view_dimension(
    texture_dim: TextureDimension,
    array_layers: u32,
) -> TextureViewDimension {
    match texture_dim {
        TextureDimension::D1 => TextureViewDimension::D1,
        TextureDimension::D2 => {
            if array_layers > 1 {
                TextureViewDimension::D2Array
            } else {
                TextureViewDimension::D2
            }
        }
        TextureDimension::D3 => TextureViewDimension::D3,
    }
}

// ============================================================================
// TrinityTexture View Extensions
// ============================================================================

impl TrinityTexture {
    /// Creates a texture view using a TRINITY descriptor.
    ///
    /// This method provides validation and a more ergonomic interface
    /// than the raw wgpu `create_view` method.
    ///
    /// # Arguments
    ///
    /// * `desc` - The view descriptor
    ///
    /// # Returns
    ///
    /// A new texture view.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::resources::{TrinityTexture, TrinityTextureDescriptor, create_texture};
    /// # use renderer_backend::resources::texture_views::TrinityTextureViewDescriptor;
    /// # use wgpu::{TextureAspect, TextureViewDimension};
    /// # fn example(device: &wgpu::Device) {
    /// # let texture = create_texture(device, &TrinityTextureDescriptor::default());
    /// let view = texture.create_trinity_view(&TrinityTextureViewDescriptor {
    ///     label: Some("my_view"),
    ///     base_mip_level: 1,
    ///     mip_level_count: Some(2),
    ///     ..Default::default()
    /// });
    /// # }
    /// ```
    pub fn create_trinity_view(&self, desc: &TrinityTextureViewDescriptor) -> TextureView {
        self.texture().create_view(&desc.to_wgpu())
    }

    /// Creates a view for a single mip level.
    ///
    /// Useful for rendering to specific mip levels during mipmap generation
    /// or for multi-resolution rendering techniques.
    ///
    /// # Arguments
    ///
    /// * `mip_level` - The mip level to view (0 = base level)
    ///
    /// # Panics
    ///
    /// Panics if `mip_level >= self.mip_level_count()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::resources::{TrinityTexture, TrinityTextureDescriptor, create_texture};
    /// # fn example(device: &wgpu::Device) {
    /// # let texture = create_texture(device, &TrinityTextureDescriptor::default());
    /// // Create view for mip level 0 (full resolution)
    /// let base_view = texture.create_mip_view(0);
    ///
    /// // Create view for mip level 3 (1/8 resolution)
    /// let small_view = texture.create_mip_view(3);
    /// # }
    /// ```
    pub fn create_mip_view(&self, mip_level: u32) -> TextureView {
        assert!(
            mip_level < self.mip_level_count(),
            "Mip level {} out of range (texture has {} mip levels)",
            mip_level,
            self.mip_level_count()
        );

        self.texture().create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: None,
            aspect: TextureAspect::All,
            base_mip_level: mip_level,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: None,
        })
    }

    /// Creates a view for a range of mip levels.
    ///
    /// # Arguments
    ///
    /// * `base_level` - First mip level
    /// * `count` - Number of mip levels
    ///
    /// # Panics
    ///
    /// Panics if the range exceeds the texture's mip levels.
    pub fn create_mip_range_view(&self, base_level: u32, count: u32) -> TextureView {
        assert!(
            validate_mip_range(self.mip_level_count(), base_level, Some(count)),
            "Mip range {}..{} out of range (texture has {} mip levels)",
            base_level,
            base_level + count,
            self.mip_level_count()
        );

        self.texture().create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: None,
            aspect: TextureAspect::All,
            base_mip_level: base_level,
            mip_level_count: Some(count),
            base_array_layer: 0,
            array_layer_count: None,
        })
    }

    /// Creates a view for a single array layer as a 2D texture.
    ///
    /// Useful for accessing individual layers of texture arrays or
    /// individual faces of cubemaps as 2D textures for rendering.
    ///
    /// # Arguments
    ///
    /// * `layer` - The array layer to view
    ///
    /// # Panics
    ///
    /// Panics if `layer >= self.depth_or_array_layers()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::resources::{TrinityTexture, TrinityTextureDescriptor, create_texture};
    /// # use wgpu::Extent3d;
    /// # fn example(device: &wgpu::Device) {
    /// // Create a 4-layer texture array
    /// # let texture = create_texture(device, &TrinityTextureDescriptor {
    /// #     size: Extent3d { width: 256, height: 256, depth_or_array_layers: 4 },
    /// #     ..Default::default()
    /// # });
    /// // View layer 2 as a 2D texture
    /// let layer_view = texture.create_layer_view(2);
    /// # }
    /// ```
    pub fn create_layer_view(&self, layer: u32) -> TextureView {
        assert!(
            layer < self.depth_or_array_layers(),
            "Array layer {} out of range (texture has {} layers)",
            layer,
            self.depth_or_array_layers()
        );

        self.texture().create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: layer,
            array_layer_count: Some(1),
        })
    }

    /// Creates a view for a range of array layers.
    ///
    /// # Arguments
    ///
    /// * `base_layer` - First array layer
    /// * `count` - Number of array layers
    ///
    /// # Panics
    ///
    /// Panics if the range exceeds the texture's layer count.
    pub fn create_layer_range_view(&self, base_layer: u32, count: u32) -> TextureView {
        assert!(
            validate_array_range(self.depth_or_array_layers(), base_layer, Some(count)),
            "Layer range {}..{} out of range (texture has {} layers)",
            base_layer,
            base_layer + count,
            self.depth_or_array_layers()
        );

        self.texture().create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: Some(TextureViewDimension::D2Array),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: base_layer,
            array_layer_count: Some(count),
        })
    }

    /// Creates a view for a specific cubemap face as a 2D texture.
    ///
    /// This allows rendering to or sampling from individual cubemap faces.
    /// The texture must have at least 6 array layers.
    ///
    /// # Arguments
    ///
    /// * `face` - The cube face to view
    ///
    /// # Panics
    ///
    /// Panics if the texture has fewer than 6 array layers.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::resources::{TrinityTexture, TrinityTextureDescriptor, create_texture};
    /// # use renderer_backend::resources::texture_views::CubeFace;
    /// # use wgpu::Extent3d;
    /// # fn example(device: &wgpu::Device) {
    /// // Create a cubemap (6-layer 2D texture)
    /// # let cubemap = create_texture(device, &TrinityTextureDescriptor {
    /// #     size: Extent3d { width: 256, height: 256, depth_or_array_layers: 6 },
    /// #     ..Default::default()
    /// # });
    /// // View the +X face for rendering
    /// let pos_x_view = cubemap.create_cube_face_view(CubeFace::PosX);
    ///
    /// // View the -Z face for rendering
    /// let neg_z_view = cubemap.create_cube_face_view(CubeFace::NegZ);
    /// # }
    /// ```
    pub fn create_cube_face_view(&self, face: CubeFace) -> TextureView {
        let layer = face.layer_index();
        assert!(
            self.depth_or_array_layers() >= 6,
            "Cannot create cube face view: texture has {} layers, need at least 6",
            self.depth_or_array_layers()
        );
        assert!(
            layer < self.depth_or_array_layers(),
            "Cube face layer {} out of range (texture has {} layers)",
            layer,
            self.depth_or_array_layers()
        );

        self.texture().create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: layer,
            array_layer_count: Some(1),
        })
    }

    /// Creates a cubemap view from a 2D array texture.
    ///
    /// The texture must have exactly 6 array layers (or a multiple of 6
    /// for cubemap arrays, though this method creates a single cubemap
    /// from the first 6 layers).
    ///
    /// # Panics
    ///
    /// Panics if the texture has fewer than 6 array layers.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::resources::{TrinityTexture, TrinityTextureDescriptor, create_texture};
    /// # use wgpu::Extent3d;
    /// # fn example(device: &wgpu::Device) {
    /// // Create a 6-layer texture to use as cubemap
    /// # let texture = create_texture(device, &TrinityTextureDescriptor {
    /// #     size: Extent3d { width: 256, height: 256, depth_or_array_layers: 6 },
    /// #     ..Default::default()
    /// # });
    /// // View as cubemap
    /// let cube_view = texture.create_cube_view();
    /// # }
    /// ```
    pub fn create_cube_view(&self) -> TextureView {
        assert!(
            self.depth_or_array_layers() >= 6,
            "Cannot create cube view: texture has {} layers, need at least 6",
            self.depth_or_array_layers()
        );

        self.texture().create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: Some(TextureViewDimension::Cube),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: Some(6),
        })
    }

    /// Creates a cubemap array view from a 2D array texture.
    ///
    /// The texture must have a multiple of 6 array layers.
    ///
    /// # Panics
    ///
    /// Panics if the texture's layer count is not a multiple of 6.
    pub fn create_cube_array_view(&self) -> TextureView {
        let layers = self.depth_or_array_layers();
        assert!(
            layers >= 6 && layers % 6 == 0,
            "Cannot create cube array view: texture has {} layers, need a multiple of 6",
            layers
        );

        self.texture().create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: Some(TextureViewDimension::CubeArray),
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        })
    }

    /// Creates a depth-only view of a depth-stencil texture.
    ///
    /// This allows sampling only the depth component, which is required
    /// for some shader operations like shadow mapping.
    ///
    /// # Returns
    ///
    /// A texture view with `TextureAspect::DepthOnly`.
    ///
    /// # Note
    ///
    /// For textures without a depth component, this will create a view
    /// with `DepthOnly` aspect, which may cause validation errors when used.
    /// Use `is_depth_stencil_format()` to check format compatibility.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::resources::{TrinityTexture, TrinityTextureDescriptor, create_texture, texture_usages};
    /// # use wgpu::{Extent3d, TextureFormat};
    /// # fn example(device: &wgpu::Device) {
    /// // Create a depth-stencil texture
    /// # let depth_texture = create_texture(device, &TrinityTextureDescriptor {
    /// #     size: Extent3d { width: 1920, height: 1080, depth_or_array_layers: 1 },
    /// #     format: TextureFormat::Depth32FloatStencil8,
    /// #     usage: texture_usages::DEPTH_TARGET,
    /// #     ..Default::default()
    /// # });
    /// // Create depth-only view for shadow sampling
    /// let shadow_view = depth_texture.create_depth_only_view();
    /// # }
    /// ```
    pub fn create_depth_only_view(&self) -> TextureView {
        self.texture().create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: None,
            aspect: TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        })
    }

    /// Creates a stencil-only view of a depth-stencil texture.
    ///
    /// This allows sampling only the stencil component.
    ///
    /// # Returns
    ///
    /// A texture view with `TextureAspect::StencilOnly`.
    ///
    /// # Note
    ///
    /// For textures without a stencil component, this will create a view
    /// with `StencilOnly` aspect, which may cause validation errors when used.
    /// Use `has_stencil_component()` to check format compatibility.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::resources::{TrinityTexture, TrinityTextureDescriptor, create_texture, texture_usages};
    /// # use wgpu::{Extent3d, TextureFormat};
    /// # fn example(device: &wgpu::Device) {
    /// // Create a depth-stencil texture
    /// # let depth_stencil = create_texture(device, &TrinityTextureDescriptor {
    /// #     size: Extent3d { width: 1920, height: 1080, depth_or_array_layers: 1 },
    /// #     format: TextureFormat::Depth24PlusStencil8,
    /// #     usage: texture_usages::DEPTH_TARGET,
    /// #     ..Default::default()
    /// # });
    /// // Create stencil-only view
    /// let stencil_view = depth_stencil.create_stencil_only_view();
    /// # }
    /// ```
    pub fn create_stencil_only_view(&self) -> TextureView {
        self.texture().create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: None,
            aspect: TextureAspect::StencilOnly,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        })
    }

    /// Creates a view with format reinterpretation.
    ///
    /// The new format must be in the texture's `view_formats` list
    /// (specified at texture creation time).
    ///
    /// # Arguments
    ///
    /// * `format` - The format to reinterpret as
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use renderer_backend::resources::{TrinityTexture, TrinityTextureDescriptor, create_texture};
    /// # use wgpu::{Extent3d, TextureFormat};
    /// # fn example(device: &wgpu::Device) {
    /// // Create sRGB texture with linear view format allowed
    /// # let texture = create_texture(device, &TrinityTextureDescriptor {
    /// #     size: Extent3d { width: 256, height: 256, depth_or_array_layers: 1 },
    /// #     format: TextureFormat::Rgba8UnormSrgb,
    /// #     view_formats: &[TextureFormat::Rgba8Unorm],
    /// #     ..Default::default()
    /// # });
    /// // Create linear view for compute shader access
    /// let linear_view = texture.create_format_view(TextureFormat::Rgba8Unorm);
    /// # }
    /// ```
    pub fn create_format_view(&self, format: TextureFormat) -> TextureView {
        self.texture().create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: Some(format),
            dimension: None,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        })
    }

    /// Creates a view for a specific mip level of a specific array layer.
    ///
    /// Useful for cascaded shadow maps, mipmap generation on texture arrays,
    /// or other techniques requiring precise control.
    ///
    /// # Arguments
    ///
    /// * `mip_level` - The mip level
    /// * `layer` - The array layer
    ///
    /// # Panics
    ///
    /// Panics if either index is out of range.
    pub fn create_mip_layer_view(&self, mip_level: u32, layer: u32) -> TextureView {
        assert!(
            mip_level < self.mip_level_count(),
            "Mip level {} out of range (texture has {} mip levels)",
            mip_level,
            self.mip_level_count()
        );
        assert!(
            layer < self.depth_or_array_layers(),
            "Array layer {} out of range (texture has {} layers)",
            layer,
            self.depth_or_array_layers()
        );

        self.texture().create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::All,
            base_mip_level: mip_level,
            mip_level_count: Some(1),
            base_array_layer: layer,
            array_layer_count: Some(1),
        })
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // CubeFace tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cube_face_layer_indices() {
        assert_eq!(CubeFace::PosX.layer_index(), 0);
        assert_eq!(CubeFace::NegX.layer_index(), 1);
        assert_eq!(CubeFace::PosY.layer_index(), 2);
        assert_eq!(CubeFace::NegY.layer_index(), 3);
        assert_eq!(CubeFace::PosZ.layer_index(), 4);
        assert_eq!(CubeFace::NegZ.layer_index(), 5);
    }

    #[test]
    fn test_cube_face_all() {
        let all = CubeFace::all();
        assert_eq!(all.len(), 6);
        for (i, face) in all.iter().enumerate() {
            assert_eq!(face.layer_index(), i as u32);
        }
    }

    #[test]
    fn test_cube_face_from_layer_index() {
        assert_eq!(CubeFace::from_layer_index(0), Some(CubeFace::PosX));
        assert_eq!(CubeFace::from_layer_index(1), Some(CubeFace::NegX));
        assert_eq!(CubeFace::from_layer_index(2), Some(CubeFace::PosY));
        assert_eq!(CubeFace::from_layer_index(3), Some(CubeFace::NegY));
        assert_eq!(CubeFace::from_layer_index(4), Some(CubeFace::PosZ));
        assert_eq!(CubeFace::from_layer_index(5), Some(CubeFace::NegZ));
        assert_eq!(CubeFace::from_layer_index(6), None);
        assert_eq!(CubeFace::from_layer_index(100), None);
    }

    #[test]
    fn test_cube_face_names() {
        assert_eq!(CubeFace::PosX.name(), "+X");
        assert_eq!(CubeFace::NegX.name(), "-X");
        assert_eq!(CubeFace::PosY.name(), "+Y");
        assert_eq!(CubeFace::NegY.name(), "-Y");
        assert_eq!(CubeFace::PosZ.name(), "+Z");
        assert_eq!(CubeFace::NegZ.name(), "-Z");
    }

    // -------------------------------------------------------------------------
    // TrinityTextureViewDescriptor tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_descriptor_defaults() {
        let desc = TrinityTextureViewDescriptor::default();
        assert!(desc.label.is_none());
        assert!(desc.format.is_none());
        assert!(desc.dimension.is_none());
        assert_eq!(desc.aspect, TextureAspect::All);
        assert_eq!(desc.base_mip_level, 0);
        assert!(desc.mip_level_count.is_none());
        assert_eq!(desc.base_array_layer, 0);
        assert!(desc.array_layer_count.is_none());
    }

    #[test]
    fn test_descriptor_single_mip() {
        let desc = TrinityTextureViewDescriptor::single_mip(3, Some("mip3"));
        assert_eq!(desc.label, Some("mip3"));
        assert_eq!(desc.base_mip_level, 3);
        assert_eq!(desc.mip_level_count, Some(1));
    }

    #[test]
    fn test_descriptor_single_layer() {
        let desc = TrinityTextureViewDescriptor::single_layer(2, Some("layer2"));
        assert_eq!(desc.label, Some("layer2"));
        assert_eq!(desc.base_array_layer, 2);
        assert_eq!(desc.array_layer_count, Some(1));
        assert_eq!(desc.dimension, Some(TextureViewDimension::D2));
    }

    #[test]
    fn test_descriptor_depth_only() {
        let desc = TrinityTextureViewDescriptor::depth_only(Some("depth"));
        assert_eq!(desc.label, Some("depth"));
        assert_eq!(desc.aspect, TextureAspect::DepthOnly);
    }

    #[test]
    fn test_descriptor_stencil_only() {
        let desc = TrinityTextureViewDescriptor::stencil_only(Some("stencil"));
        assert_eq!(desc.label, Some("stencil"));
        assert_eq!(desc.aspect, TextureAspect::StencilOnly);
    }

    #[test]
    fn test_descriptor_as_cube() {
        let desc = TrinityTextureViewDescriptor::as_cube(Some("cube"));
        assert_eq!(desc.label, Some("cube"));
        assert_eq!(desc.dimension, Some(TextureViewDimension::Cube));
        assert_eq!(desc.array_layer_count, Some(6));
    }

    #[test]
    fn test_descriptor_cube_face() {
        let desc = TrinityTextureViewDescriptor::cube_face(CubeFace::PosZ, Some("front"));
        assert_eq!(desc.label, Some("front"));
        assert_eq!(desc.dimension, Some(TextureViewDimension::D2));
        assert_eq!(desc.base_array_layer, 4); // PosZ = layer 4
        assert_eq!(desc.array_layer_count, Some(1));
    }

    #[test]
    fn test_descriptor_with_format() {
        let desc = TrinityTextureViewDescriptor::with_format(
            TextureFormat::Rgba8UnormSrgb,
            Some("srgb"),
        );
        assert_eq!(desc.label, Some("srgb"));
        assert_eq!(desc.format, Some(TextureFormat::Rgba8UnormSrgb));
    }

    #[test]
    fn test_descriptor_to_wgpu() {
        let desc = TrinityTextureViewDescriptor {
            label: Some("test"),
            format: Some(TextureFormat::Rgba8Unorm),
            dimension: Some(TextureViewDimension::D2),
            aspect: TextureAspect::DepthOnly,
            base_mip_level: 1,
            mip_level_count: Some(3),
            base_array_layer: 2,
            array_layer_count: Some(4),
        };

        let wgpu_desc = desc.to_wgpu();
        assert_eq!(wgpu_desc.label, Some("test"));
        assert_eq!(wgpu_desc.format, Some(TextureFormat::Rgba8Unorm));
        assert_eq!(wgpu_desc.dimension, Some(TextureViewDimension::D2));
        assert_eq!(wgpu_desc.aspect, TextureAspect::DepthOnly);
        assert_eq!(wgpu_desc.base_mip_level, 1);
        assert_eq!(wgpu_desc.mip_level_count, Some(3));
        assert_eq!(wgpu_desc.base_array_layer, 2);
        assert_eq!(wgpu_desc.array_layer_count, Some(4));
    }

    // -------------------------------------------------------------------------
    // validate_view_dimensions tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_view_dimensions_d1() {
        assert!(validate_view_dimensions(
            TextureDimension::D1,
            TextureViewDimension::D1,
            1
        ));
        assert!(!validate_view_dimensions(
            TextureDimension::D1,
            TextureViewDimension::D2,
            1
        ));
    }

    #[test]
    fn test_validate_view_dimensions_d2_single_layer() {
        assert!(validate_view_dimensions(
            TextureDimension::D2,
            TextureViewDimension::D2,
            1
        ));
        assert!(validate_view_dimensions(
            TextureDimension::D2,
            TextureViewDimension::D2Array,
            1
        ));
        assert!(!validate_view_dimensions(
            TextureDimension::D2,
            TextureViewDimension::Cube,
            1
        ));
    }

    #[test]
    fn test_validate_view_dimensions_d2_cube() {
        assert!(validate_view_dimensions(
            TextureDimension::D2,
            TextureViewDimension::Cube,
            6
        ));
        assert!(!validate_view_dimensions(
            TextureDimension::D2,
            TextureViewDimension::Cube,
            5
        ));
    }

    #[test]
    fn test_validate_view_dimensions_d2_cube_array() {
        assert!(validate_view_dimensions(
            TextureDimension::D2,
            TextureViewDimension::CubeArray,
            6
        ));
        assert!(validate_view_dimensions(
            TextureDimension::D2,
            TextureViewDimension::CubeArray,
            12
        ));
        assert!(!validate_view_dimensions(
            TextureDimension::D2,
            TextureViewDimension::CubeArray,
            7
        )); // Not multiple of 6
    }

    #[test]
    fn test_validate_view_dimensions_d3() {
        assert!(validate_view_dimensions(
            TextureDimension::D3,
            TextureViewDimension::D3,
            1
        ));
        assert!(!validate_view_dimensions(
            TextureDimension::D3,
            TextureViewDimension::D2,
            1
        ));
    }

    // -------------------------------------------------------------------------
    // validate_mip_range tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_mip_range_valid() {
        // All mips from base 0
        assert!(validate_mip_range(5, 0, None));
        // All mips from base 2
        assert!(validate_mip_range(5, 2, None));
        // Specific count
        assert!(validate_mip_range(5, 0, Some(5)));
        assert!(validate_mip_range(5, 2, Some(3)));
        // Single mip
        assert!(validate_mip_range(5, 4, Some(1)));
    }

    #[test]
    fn test_validate_mip_range_invalid() {
        // Base out of range
        assert!(!validate_mip_range(5, 5, None));
        assert!(!validate_mip_range(5, 10, Some(1)));
        // Count exceeds available
        assert!(!validate_mip_range(5, 3, Some(5)));
        // Zero count
        assert!(!validate_mip_range(5, 0, Some(0)));
    }

    // -------------------------------------------------------------------------
    // validate_array_range tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_validate_array_range_valid() {
        // All layers from base 0
        assert!(validate_array_range(6, 0, None));
        // All layers from base 2
        assert!(validate_array_range(6, 2, None));
        // Specific count (cubemap)
        assert!(validate_array_range(6, 0, Some(6)));
        // Single layer
        assert!(validate_array_range(6, 5, Some(1)));
    }

    #[test]
    fn test_validate_array_range_invalid() {
        // Base out of range
        assert!(!validate_array_range(6, 6, None));
        assert!(!validate_array_range(6, 10, Some(1)));
        // Count exceeds available
        assert!(!validate_array_range(6, 4, Some(5)));
        // Zero count
        assert!(!validate_array_range(6, 0, Some(0)));
    }

    // -------------------------------------------------------------------------
    // Format helper tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_depth_stencil_format() {
        assert!(is_depth_stencil_format(TextureFormat::Depth16Unorm));
        assert!(is_depth_stencil_format(TextureFormat::Depth24Plus));
        assert!(is_depth_stencil_format(TextureFormat::Depth24PlusStencil8));
        assert!(is_depth_stencil_format(TextureFormat::Depth32Float));
        assert!(is_depth_stencil_format(TextureFormat::Depth32FloatStencil8));
        assert!(is_depth_stencil_format(TextureFormat::Stencil8));

        assert!(!is_depth_stencil_format(TextureFormat::Rgba8Unorm));
        assert!(!is_depth_stencil_format(TextureFormat::R32Float));
    }

    #[test]
    fn test_has_stencil_component() {
        assert!(has_stencil_component(TextureFormat::Depth24PlusStencil8));
        assert!(has_stencil_component(TextureFormat::Depth32FloatStencil8));
        assert!(has_stencil_component(TextureFormat::Stencil8));

        assert!(!has_stencil_component(TextureFormat::Depth16Unorm));
        assert!(!has_stencil_component(TextureFormat::Depth32Float));
        assert!(!has_stencil_component(TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn test_has_depth_component() {
        assert!(has_depth_component(TextureFormat::Depth16Unorm));
        assert!(has_depth_component(TextureFormat::Depth24Plus));
        assert!(has_depth_component(TextureFormat::Depth24PlusStencil8));
        assert!(has_depth_component(TextureFormat::Depth32Float));
        assert!(has_depth_component(TextureFormat::Depth32FloatStencil8));

        assert!(!has_depth_component(TextureFormat::Stencil8));
        assert!(!has_depth_component(TextureFormat::Rgba8Unorm));
    }

    #[test]
    fn test_native_view_dimension() {
        assert_eq!(
            native_view_dimension(TextureDimension::D1, 1),
            TextureViewDimension::D1
        );
        assert_eq!(
            native_view_dimension(TextureDimension::D2, 1),
            TextureViewDimension::D2
        );
        assert_eq!(
            native_view_dimension(TextureDimension::D2, 4),
            TextureViewDimension::D2Array
        );
        assert_eq!(
            native_view_dimension(TextureDimension::D3, 1),
            TextureViewDimension::D3
        );
    }
}

//! Vertex attribute format utilities for wgpu 22.x
//!
//! This module provides comprehensive documentation and helpers for all wgpu
//! [`VertexFormat`] variants, including size calculations and common format
//! combinations for typical rendering scenarios.
//!
//! # Overview
//!
//! wgpu 22.x provides 32 vertex format variants organized into categories:
//!
//! - **8-bit formats** (1 byte per component): `Uint8x2`, `Uint8x4`, `Sint8x2`, `Sint8x4`,
//!   `Unorm8x2`, `Unorm8x4`, `Snorm8x2`, `Snorm8x4`
//! - **16-bit formats** (2 bytes per component): `Uint16x2`, `Uint16x4`, `Sint16x2`, `Sint16x4`,
//!   `Unorm16x2`, `Unorm16x4`, `Snorm16x2`, `Snorm16x4`, `Float16x2`, `Float16x4`
//! - **32-bit formats** (4 bytes per component): `Uint32`, `Uint32x2`, `Uint32x3`, `Uint32x4`,
//!   `Sint32`, `Sint32x2`, `Sint32x3`, `Sint32x4`, `Float32`, `Float32x2`, `Float32x3`, `Float32x4`
//! - **64-bit formats** (8 bytes per component): `Float64`, `Float64x2`, `Float64x3`, `Float64x4`
//!
//! # Usage
//!
//! ```no_run
//! use renderer_backend::render_pipeline::vertex_attribute::{
//!     vertex_format_size, common, VertexFormatInfo,
//! };
//! use wgpu::VertexFormat;
//!
//! // Get size of a format
//! let size = vertex_format_size(VertexFormat::Float32x3);
//! assert_eq!(size, 12);
//!
//! // Use common format presets
//! let position_format = common::POSITION;
//! let color_format = common::COLOR;
//! ```
//!
//! # Format Reference
//!
//! | Format | Size | Components | Shader Type | Use Case |
//! |--------|------|------------|-------------|----------|
//! | `Float32x3` | 12 | 3 | `vec3<f32>` | Position, Normal |
//! | `Float32x2` | 8 | 2 | `vec2<f32>` | UV coordinates |
//! | `Float32x4` | 16 | 4 | `vec4<f32>` | Tangent, Color (HDR) |
//! | `Unorm8x4` | 4 | 4 | `vec4<f32>` | Color (LDR, packed) |
//! | `Uint8x4` | 4 | 4 | `vec4<u32>` | Bone indices |
//! | `Uint16x4` | 8 | 4 | `vec4<u32>` | Bone indices (large) |

use wgpu::VertexFormat;

// Re-export the vertex_attr_array! macro for convenience
pub use wgpu::vertex_attr_array;

// ---------------------------------------------------------------------------
// Size Calculation
// ---------------------------------------------------------------------------

/// Returns the size in bytes for a wgpu vertex format.
///
/// This is a const function that can be used at compile time for calculating
/// vertex strides and buffer sizes.
///
/// # Example
///
/// ```no_run
/// use wgpu::VertexFormat;
/// use renderer_backend::render_pipeline::vertex_attribute::vertex_format_size;
///
/// // Calculate stride for a position + normal + uv layout
/// const STRIDE: u64 = vertex_format_size(VertexFormat::Float32x3)  // position: 12
///                   + vertex_format_size(VertexFormat::Float32x3)  // normal: 12
///                   + vertex_format_size(VertexFormat::Float32x2); // uv: 8
/// assert_eq!(STRIDE, 32);
/// ```
pub const fn vertex_format_size(format: VertexFormat) -> u64 {
    match format {
        // =====================================================================
        // 8-bit unsigned integer formats (1 byte per component)
        // =====================================================================
        /// Two unsigned 8-bit integers. Shader type: `vec2<u32>`
        VertexFormat::Uint8x2 => 2,
        /// Four unsigned 8-bit integers. Shader type: `vec4<u32>`
        VertexFormat::Uint8x4 => 4,

        // =====================================================================
        // 8-bit signed integer formats (1 byte per component)
        // =====================================================================
        /// Two signed 8-bit integers. Shader type: `vec2<i32>`
        VertexFormat::Sint8x2 => 2,
        /// Four signed 8-bit integers. Shader type: `vec4<i32>`
        VertexFormat::Sint8x4 => 4,

        // =====================================================================
        // 8-bit unsigned normalized formats (1 byte per component, 0.0-1.0)
        // =====================================================================
        /// Two unsigned 8-bit normalized. Shader type: `vec2<f32>`
        VertexFormat::Unorm8x2 => 2,
        /// Four unsigned 8-bit normalized. Shader type: `vec4<f32>`
        /// Common use: LDR vertex colors (RGBA)
        VertexFormat::Unorm8x4 => 4,

        // =====================================================================
        // 8-bit signed normalized formats (1 byte per component, -1.0-1.0)
        // =====================================================================
        /// Two signed 8-bit normalized. Shader type: `vec2<f32>`
        VertexFormat::Snorm8x2 => 2,
        /// Four signed 8-bit normalized. Shader type: `vec4<f32>`
        /// Common use: Compressed normals/tangents
        VertexFormat::Snorm8x4 => 4,

        // =====================================================================
        // 16-bit unsigned integer formats (2 bytes per component)
        // =====================================================================
        /// Two unsigned 16-bit integers. Shader type: `vec2<u32>`
        VertexFormat::Uint16x2 => 4,
        /// Four unsigned 16-bit integers. Shader type: `vec4<u32>`
        /// Common use: Bone indices (up to 65535 bones)
        VertexFormat::Uint16x4 => 8,

        // =====================================================================
        // 16-bit signed integer formats (2 bytes per component)
        // =====================================================================
        /// Two signed 16-bit integers. Shader type: `vec2<i32>`
        VertexFormat::Sint16x2 => 4,
        /// Four signed 16-bit integers. Shader type: `vec4<i32>`
        VertexFormat::Sint16x4 => 8,

        // =====================================================================
        // 16-bit unsigned normalized formats (2 bytes per component, 0.0-1.0)
        // =====================================================================
        /// Two unsigned 16-bit normalized. Shader type: `vec2<f32>`
        VertexFormat::Unorm16x2 => 4,
        /// Four unsigned 16-bit normalized. Shader type: `vec4<f32>`
        VertexFormat::Unorm16x4 => 8,

        // =====================================================================
        // 16-bit signed normalized formats (2 bytes per component, -1.0-1.0)
        // =====================================================================
        /// Two signed 16-bit normalized. Shader type: `vec2<f32>`
        VertexFormat::Snorm16x2 => 4,
        /// Four signed 16-bit normalized. Shader type: `vec4<f32>`
        VertexFormat::Snorm16x4 => 8,

        // =====================================================================
        // 16-bit floating point formats (2 bytes per component)
        // =====================================================================
        /// Two half-precision floats. Shader type: `vec2<f32>`
        VertexFormat::Float16x2 => 4,
        /// Four half-precision floats. Shader type: `vec4<f32>`
        /// Common use: HDR colors, compressed positions
        VertexFormat::Float16x4 => 8,

        // =====================================================================
        // 32-bit unsigned integer formats (4 bytes per component)
        // =====================================================================
        /// Single unsigned 32-bit integer. Shader type: `u32`
        VertexFormat::Uint32 => 4,
        /// Two unsigned 32-bit integers. Shader type: `vec2<u32>`
        VertexFormat::Uint32x2 => 8,
        /// Three unsigned 32-bit integers. Shader type: `vec3<u32>`
        VertexFormat::Uint32x3 => 12,
        /// Four unsigned 32-bit integers. Shader type: `vec4<u32>`
        VertexFormat::Uint32x4 => 16,

        // =====================================================================
        // 32-bit signed integer formats (4 bytes per component)
        // =====================================================================
        /// Single signed 32-bit integer. Shader type: `i32`
        VertexFormat::Sint32 => 4,
        /// Two signed 32-bit integers. Shader type: `vec2<i32>`
        VertexFormat::Sint32x2 => 8,
        /// Three signed 32-bit integers. Shader type: `vec3<i32>`
        VertexFormat::Sint32x3 => 12,
        /// Four signed 32-bit integers. Shader type: `vec4<i32>`
        VertexFormat::Sint32x4 => 16,

        // =====================================================================
        // 32-bit floating point formats (4 bytes per component)
        // =====================================================================
        /// Single-precision float. Shader type: `f32`
        VertexFormat::Float32 => 4,
        /// Two single-precision floats. Shader type: `vec2<f32>`
        /// Common use: UV coordinates
        VertexFormat::Float32x2 => 8,
        /// Three single-precision floats. Shader type: `vec3<f32>`
        /// Common use: Position, Normal
        VertexFormat::Float32x3 => 12,
        /// Four single-precision floats. Shader type: `vec4<f32>`
        /// Common use: Tangent (with handedness), Color (HDR)
        VertexFormat::Float32x4 => 16,

        // =====================================================================
        // 64-bit floating point formats (8 bytes per component)
        // =====================================================================
        /// Double-precision float. Shader type: `f32` (converted)
        /// Note: Requires VERTEX_ATTRIBUTE_64BIT feature
        VertexFormat::Float64 => 8,
        /// Two double-precision floats. Shader type: `vec2<f32>` (converted)
        VertexFormat::Float64x2 => 16,
        /// Three double-precision floats. Shader type: `vec3<f32>` (converted)
        VertexFormat::Float64x3 => 24,
        /// Four double-precision floats. Shader type: `vec4<f32>` (converted)
        VertexFormat::Float64x4 => 32,

        // =====================================================================
        // Packed formats
        // =====================================================================
        /// Packed 10-10-10-2 unsigned normalized format (4 bytes total).
        /// Components: R10 G10 B10 A2 -> vec4<f32>
        /// Common use: Compressed normals with high precision
        VertexFormat::Unorm10_10_10_2 => 4,
    }
}

/// Returns the number of components for a vertex format.
pub const fn vertex_format_components(format: VertexFormat) -> u32 {
    match format {
        // Single component (32-bit and 64-bit only)
        VertexFormat::Uint32 | VertexFormat::Sint32 | VertexFormat::Float32 |
        VertexFormat::Float64 => 1,

        // Two components
        VertexFormat::Uint8x2 | VertexFormat::Sint8x2 | VertexFormat::Unorm8x2 | VertexFormat::Snorm8x2 |
        VertexFormat::Uint16x2 | VertexFormat::Sint16x2 | VertexFormat::Unorm16x2 | VertexFormat::Snorm16x2 |
        VertexFormat::Float16x2 | VertexFormat::Uint32x2 | VertexFormat::Sint32x2 | VertexFormat::Float32x2 |
        VertexFormat::Float64x2 => 2,

        // Three components (32-bit and 64-bit only)
        VertexFormat::Uint32x3 | VertexFormat::Sint32x3 | VertexFormat::Float32x3 |
        VertexFormat::Float64x3 => 3,

        // Four components
        VertexFormat::Uint8x4 | VertexFormat::Sint8x4 | VertexFormat::Unorm8x4 | VertexFormat::Snorm8x4 |
        VertexFormat::Uint16x4 | VertexFormat::Sint16x4 | VertexFormat::Unorm16x4 | VertexFormat::Snorm16x4 |
        VertexFormat::Float16x4 | VertexFormat::Uint32x4 | VertexFormat::Sint32x4 | VertexFormat::Float32x4 |
        VertexFormat::Float64x4 | VertexFormat::Unorm10_10_10_2 => 4,
    }
}

/// Returns whether the format is normalized (values mapped to 0.0-1.0 or -1.0-1.0).
pub const fn vertex_format_is_normalized(format: VertexFormat) -> bool {
    matches!(
        format,
        VertexFormat::Unorm8x2 | VertexFormat::Unorm8x4 |
        VertexFormat::Snorm8x2 | VertexFormat::Snorm8x4 |
        VertexFormat::Unorm16x2 | VertexFormat::Unorm16x4 |
        VertexFormat::Snorm16x2 | VertexFormat::Snorm16x4 |
        VertexFormat::Unorm10_10_10_2
    )
}

/// Returns whether the format is a floating point type.
pub const fn vertex_format_is_float(format: VertexFormat) -> bool {
    matches!(
        format,
        VertexFormat::Float16x2 | VertexFormat::Float16x4 |
        VertexFormat::Float32 | VertexFormat::Float32x2 | VertexFormat::Float32x3 | VertexFormat::Float32x4 |
        VertexFormat::Float64 | VertexFormat::Float64x2 | VertexFormat::Float64x3 | VertexFormat::Float64x4
    )
}

/// Returns whether the format is a signed integer type.
pub const fn vertex_format_is_signed_int(format: VertexFormat) -> bool {
    matches!(
        format,
        VertexFormat::Sint8x2 | VertexFormat::Sint8x4 |
        VertexFormat::Sint16x2 | VertexFormat::Sint16x4 |
        VertexFormat::Sint32 | VertexFormat::Sint32x2 | VertexFormat::Sint32x3 | VertexFormat::Sint32x4
    )
}

/// Returns whether the format is an unsigned integer type.
pub const fn vertex_format_is_unsigned_int(format: VertexFormat) -> bool {
    matches!(
        format,
        VertexFormat::Uint8x2 | VertexFormat::Uint8x4 |
        VertexFormat::Uint16x2 | VertexFormat::Uint16x4 |
        VertexFormat::Uint32 | VertexFormat::Uint32x2 | VertexFormat::Uint32x3 | VertexFormat::Uint32x4
    )
}

// ---------------------------------------------------------------------------
// VertexFormatInfo
// ---------------------------------------------------------------------------

/// Detailed information about a vertex format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexFormatInfo {
    /// The wgpu format enum value.
    pub format: VertexFormat,
    /// Size in bytes.
    pub size: u64,
    /// Number of components (1-4).
    pub components: u32,
    /// Whether values are normalized to [0, 1] or [-1, 1].
    pub normalized: bool,
    /// Whether this is a floating point format.
    pub is_float: bool,
    /// Bytes per component.
    pub bytes_per_component: u32,
}

impl VertexFormatInfo {
    /// Get detailed information about a vertex format.
    pub const fn new(format: VertexFormat) -> Self {
        let size = vertex_format_size(format);
        let components = vertex_format_components(format);
        Self {
            format,
            size,
            components,
            normalized: vertex_format_is_normalized(format),
            is_float: vertex_format_is_float(format),
            bytes_per_component: (size / components as u64) as u32,
        }
    }
}

impl From<VertexFormat> for VertexFormatInfo {
    fn from(format: VertexFormat) -> Self {
        Self::new(format)
    }
}

// ---------------------------------------------------------------------------
// Common Format Combinations
// ---------------------------------------------------------------------------

/// Common vertex attribute format presets for typical rendering scenarios.
///
/// These constants provide recommended formats for standard vertex attributes,
/// balancing precision, memory efficiency, and GPU compatibility.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::render_pipeline::vertex_attribute::common;
/// use wgpu::VertexFormat;
///
/// // Standard vertex layout
/// let position = common::POSITION;  // Float32x3
/// let normal = common::NORMAL;      // Float32x3
/// let uv = common::UV;              // Float32x2
/// let tangent = common::TANGENT;    // Float32x4
/// let color = common::COLOR;        // Unorm8x4
/// ```
pub mod common {
    use wgpu::VertexFormat;

    // =========================================================================
    // Geometry Attributes
    // =========================================================================

    /// Position: `Float32x3` (12 bytes)
    ///
    /// Standard 3D position format. Use for world-space or object-space positions.
    /// Shader type: `vec3<f32>`
    pub const POSITION: VertexFormat = VertexFormat::Float32x3;

    /// Position 2D: `Float32x2` (8 bytes)
    ///
    /// 2D position for UI/2D rendering.
    /// Shader type: `vec2<f32>`
    pub const POSITION_2D: VertexFormat = VertexFormat::Float32x2;

    /// Normal: `Float32x3` (12 bytes)
    ///
    /// Standard surface normal format. Values should be normalized.
    /// Shader type: `vec3<f32>`
    pub const NORMAL: VertexFormat = VertexFormat::Float32x3;

    /// Compressed Normal: `Snorm8x4` (4 bytes)
    ///
    /// Memory-efficient normal storage with slight precision loss.
    /// The w component can store additional data (e.g., face index).
    /// Shader type: `vec4<f32>` with values in [-1, 1]
    pub const NORMAL_COMPRESSED: VertexFormat = VertexFormat::Snorm8x4;

    /// Tangent: `Float32x4` (16 bytes)
    ///
    /// Tangent vector with handedness in the w component.
    /// Bitangent = cross(normal, tangent.xyz) * tangent.w
    /// Shader type: `vec4<f32>`
    pub const TANGENT: VertexFormat = VertexFormat::Float32x4;

    /// Compressed Tangent: `Snorm8x4` (4 bytes)
    ///
    /// Memory-efficient tangent storage. w component stores handedness.
    /// Shader type: `vec4<f32>` with values in [-1, 1]
    pub const TANGENT_COMPRESSED: VertexFormat = VertexFormat::Snorm8x4;

    /// High-Precision Normal (10-10-10-2): `Unorm10_10_10_2` (4 bytes)
    ///
    /// Packed normal with 10 bits per axis, better precision than Snorm8.
    /// The 2-bit alpha can store handedness or other flags.
    /// Shader type: `vec4<f32>` with values in [0, 1]
    pub const NORMAL_PACKED_1010102: VertexFormat = VertexFormat::Unorm10_10_10_2;

    // =========================================================================
    // Texture Coordinates
    // =========================================================================

    /// UV Coordinates: `Float32x2` (8 bytes)
    ///
    /// Standard texture coordinates. Supports values outside [0, 1] for tiling.
    /// Shader type: `vec2<f32>`
    pub const UV: VertexFormat = VertexFormat::Float32x2;

    /// UV Coordinates (Half): `Float16x2` (4 bytes)
    ///
    /// Memory-efficient UV coordinates for static meshes where precision
    /// loss is acceptable. Useful for large meshes with simple texturing.
    /// Shader type: `vec2<f32>`
    pub const UV_HALF: VertexFormat = VertexFormat::Float16x2;

    /// UV Coordinates (Normalized): `Unorm16x2` (4 bytes)
    ///
    /// UV coordinates in [0, 1] range. Good for atlas-based rendering.
    /// Shader type: `vec2<f32>`
    pub const UV_NORMALIZED: VertexFormat = VertexFormat::Unorm16x2;

    // =========================================================================
    // Color Attributes
    // =========================================================================

    /// Vertex Color (LDR): `Unorm8x4` (4 bytes)
    ///
    /// Standard RGBA vertex color in [0, 1] range.
    /// Memory-efficient, suitable for most use cases.
    /// Shader type: `vec4<f32>`
    pub const COLOR: VertexFormat = VertexFormat::Unorm8x4;

    /// Vertex Color (HDR): `Float32x4` (16 bytes)
    ///
    /// High dynamic range color with full float precision.
    /// Use for emissive surfaces or when color values exceed [0, 1].
    /// Shader type: `vec4<f32>`
    pub const COLOR_HDR: VertexFormat = VertexFormat::Float32x4;

    /// Vertex Color (HDR Half): `Float16x4` (8 bytes)
    ///
    /// HDR color with half-precision floats. Good balance of range and memory.
    /// Shader type: `vec4<f32>`
    pub const COLOR_HDR_HALF: VertexFormat = VertexFormat::Float16x4;

    // =========================================================================
    // Skeletal Animation Attributes
    // =========================================================================

    /// Bone Indices (Small): `Uint8x4` (4 bytes)
    ///
    /// Bone indices for skeletal animation with up to 256 bones.
    /// Shader type: `vec4<u32>`
    pub const BONE_INDICES: VertexFormat = VertexFormat::Uint8x4;

    /// Bone Indices (Large): `Uint16x4` (8 bytes)
    ///
    /// Bone indices for skeletons with up to 65535 bones.
    /// Shader type: `vec4<u32>`
    pub const BONE_INDICES_LARGE: VertexFormat = VertexFormat::Uint16x4;

    /// Bone Weights (Normalized): `Unorm8x4` (4 bytes)
    ///
    /// Bone weights normalized to [0, 1]. Memory efficient.
    /// Shader type: `vec4<f32>`
    pub const BONE_WEIGHTS: VertexFormat = VertexFormat::Unorm8x4;

    /// Bone Weights (High Precision): `Float32x4` (16 bytes)
    ///
    /// Bone weights with full float precision for high-quality skinning.
    /// Shader type: `vec4<f32>`
    pub const BONE_WEIGHTS_FLOAT: VertexFormat = VertexFormat::Float32x4;

    // =========================================================================
    // Instance Attributes
    // =========================================================================

    /// Instance Matrix Row: `Float32x4` (16 bytes)
    ///
    /// One row of a 4x4 transformation matrix.
    /// Use 3-4 attributes for a full transform matrix.
    /// Shader type: `vec4<f32>`
    pub const INSTANCE_MATRIX_ROW: VertexFormat = VertexFormat::Float32x4;

    /// Instance ID: `Uint32` (4 bytes)
    ///
    /// Per-instance identifier for indirect rendering or lookup tables.
    /// Shader type: `u32`
    pub const INSTANCE_ID: VertexFormat = VertexFormat::Uint32;

    // =========================================================================
    // Particle Attributes
    // =========================================================================

    /// Particle Size: `Float32` (4 bytes)
    ///
    /// Uniform particle size (for point sprites or billboards).
    /// Shader type: `f32`
    pub const PARTICLE_SIZE: VertexFormat = VertexFormat::Float32;

    /// Particle Size 2D: `Float32x2` (8 bytes)
    ///
    /// Non-uniform particle size (width, height).
    /// Shader type: `vec2<f32>`
    pub const PARTICLE_SIZE_2D: VertexFormat = VertexFormat::Float32x2;

    /// Particle Rotation: `Float32` (4 bytes)
    ///
    /// Particle rotation angle in radians.
    /// Shader type: `f32`
    pub const PARTICLE_ROTATION: VertexFormat = VertexFormat::Float32;

    /// Particle Life: `Float32` (4 bytes)
    ///
    /// Particle lifetime (normalized 0-1 or absolute seconds).
    /// Shader type: `f32`
    pub const PARTICLE_LIFE: VertexFormat = VertexFormat::Float32;

    /// Particle Velocity: `Float32x3` (12 bytes)
    ///
    /// Particle velocity vector.
    /// Shader type: `vec3<f32>`
    pub const PARTICLE_VELOCITY: VertexFormat = VertexFormat::Float32x3;
}

// ---------------------------------------------------------------------------
// Stride Calculation Helpers
// ---------------------------------------------------------------------------

/// Calculate the total stride for a list of vertex formats.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::render_pipeline::vertex_attribute::calculate_stride;
/// use wgpu::VertexFormat;
///
/// let stride = calculate_stride(&[
///     VertexFormat::Float32x3, // position: 12
///     VertexFormat::Float32x3, // normal: 12
///     VertexFormat::Float32x2, // uv: 8
/// ]);
/// assert_eq!(stride, 32);
/// ```
pub const fn calculate_stride(formats: &[VertexFormat]) -> u64 {
    let mut total = 0u64;
    let mut i = 0;
    while i < formats.len() {
        total += vertex_format_size(formats[i]);
        i += 1;
    }
    total
}

/// Calculate offsets for a list of vertex formats.
///
/// Returns an array of offsets where each format begins in the vertex buffer.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::render_pipeline::vertex_attribute::calculate_offsets;
/// use wgpu::VertexFormat;
///
/// let offsets = calculate_offsets(&[
///     VertexFormat::Float32x3, // position: offset 0
///     VertexFormat::Float32x3, // normal: offset 12
///     VertexFormat::Float32x2, // uv: offset 24
/// ]);
/// assert_eq!(offsets, [0, 12, 24]);
/// ```
pub fn calculate_offsets<const N: usize>(formats: &[VertexFormat; N]) -> [u64; N] {
    let mut offsets = [0u64; N];
    let mut current_offset = 0u64;
    for i in 0..N {
        offsets[i] = current_offset;
        current_offset += vertex_format_size(formats[i]);
    }
    offsets
}

// ---------------------------------------------------------------------------
// Standard Vertex Layout Strides
// ---------------------------------------------------------------------------

/// Pre-calculated strides for common vertex layouts.
pub mod strides {
    /// PBR vertex: position (12) + normal (12) + uv (8) + tangent (16) = 48 bytes
    pub const PBR: u64 = 48;

    /// Skinned vertex: PBR (48) + bone_indices (8) + bone_weights (16) = 72 bytes
    pub const SKINNED: u64 = 72;

    /// Terrain vertex: position (12) + normal (12) + uv (8) = 32 bytes
    pub const TERRAIN: u64 = 32;

    /// Particle vertex: position (12) + color (16) + size_rotation (8) = 36 bytes
    pub const PARTICLE: u64 = 36;

    /// UI vertex: position_2d (8) + uv (8) + color (4) = 20 bytes
    pub const UI: u64 = 20;

    /// Position-only vertex: position (12) = 12 bytes
    pub const POSITION_ONLY: u64 = 12;

    /// Shadow vertex: position (12) = 12 bytes (same as position-only)
    pub const SHADOW: u64 = 12;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Size Tests - 8-bit Formats
    // =========================================================================

    #[test]
    fn test_uint8_sizes() {
        assert_eq!(vertex_format_size(VertexFormat::Uint8x2), 2);
        assert_eq!(vertex_format_size(VertexFormat::Uint8x4), 4);
    }

    #[test]
    fn test_sint8_sizes() {
        assert_eq!(vertex_format_size(VertexFormat::Sint8x2), 2);
        assert_eq!(vertex_format_size(VertexFormat::Sint8x4), 4);
    }

    #[test]
    fn test_unorm8_sizes() {
        assert_eq!(vertex_format_size(VertexFormat::Unorm8x2), 2);
        assert_eq!(vertex_format_size(VertexFormat::Unorm8x4), 4);
    }

    #[test]
    fn test_snorm8_sizes() {
        assert_eq!(vertex_format_size(VertexFormat::Snorm8x2), 2);
        assert_eq!(vertex_format_size(VertexFormat::Snorm8x4), 4);
    }

    // =========================================================================
    // Size Tests - 16-bit Formats
    // =========================================================================

    #[test]
    fn test_uint16_sizes() {
        assert_eq!(vertex_format_size(VertexFormat::Uint16x2), 4);
        assert_eq!(vertex_format_size(VertexFormat::Uint16x4), 8);
    }

    #[test]
    fn test_sint16_sizes() {
        assert_eq!(vertex_format_size(VertexFormat::Sint16x2), 4);
        assert_eq!(vertex_format_size(VertexFormat::Sint16x4), 8);
    }

    #[test]
    fn test_unorm16_sizes() {
        assert_eq!(vertex_format_size(VertexFormat::Unorm16x2), 4);
        assert_eq!(vertex_format_size(VertexFormat::Unorm16x4), 8);
    }

    #[test]
    fn test_snorm16_sizes() {
        assert_eq!(vertex_format_size(VertexFormat::Snorm16x2), 4);
        assert_eq!(vertex_format_size(VertexFormat::Snorm16x4), 8);
    }

    #[test]
    fn test_float16_sizes() {
        assert_eq!(vertex_format_size(VertexFormat::Float16x2), 4);
        assert_eq!(vertex_format_size(VertexFormat::Float16x4), 8);
    }

    // =========================================================================
    // Size Tests - 32-bit Formats
    // =========================================================================

    #[test]
    fn test_uint32_sizes() {
        assert_eq!(vertex_format_size(VertexFormat::Uint32), 4);
        assert_eq!(vertex_format_size(VertexFormat::Uint32x2), 8);
        assert_eq!(vertex_format_size(VertexFormat::Uint32x3), 12);
        assert_eq!(vertex_format_size(VertexFormat::Uint32x4), 16);
    }

    #[test]
    fn test_sint32_sizes() {
        assert_eq!(vertex_format_size(VertexFormat::Sint32), 4);
        assert_eq!(vertex_format_size(VertexFormat::Sint32x2), 8);
        assert_eq!(vertex_format_size(VertexFormat::Sint32x3), 12);
        assert_eq!(vertex_format_size(VertexFormat::Sint32x4), 16);
    }

    #[test]
    fn test_float32_sizes() {
        assert_eq!(vertex_format_size(VertexFormat::Float32), 4);
        assert_eq!(vertex_format_size(VertexFormat::Float32x2), 8);
        assert_eq!(vertex_format_size(VertexFormat::Float32x3), 12);
        assert_eq!(vertex_format_size(VertexFormat::Float32x4), 16);
    }

    // =========================================================================
    // Size Tests - 64-bit Formats
    // =========================================================================

    #[test]
    fn test_float64_sizes() {
        assert_eq!(vertex_format_size(VertexFormat::Float64), 8);
        assert_eq!(vertex_format_size(VertexFormat::Float64x2), 16);
        assert_eq!(vertex_format_size(VertexFormat::Float64x3), 24);
        assert_eq!(vertex_format_size(VertexFormat::Float64x4), 32);
    }

    // =========================================================================
    // Size Tests - Packed Formats
    // =========================================================================

    #[test]
    fn test_packed_sizes() {
        assert_eq!(vertex_format_size(VertexFormat::Unorm10_10_10_2), 4);
    }

    // =========================================================================
    // Component Count Tests
    // =========================================================================

    #[test]
    fn test_single_component() {
        assert_eq!(vertex_format_components(VertexFormat::Uint32), 1);
        assert_eq!(vertex_format_components(VertexFormat::Sint32), 1);
        assert_eq!(vertex_format_components(VertexFormat::Float32), 1);
        assert_eq!(vertex_format_components(VertexFormat::Float64), 1);
    }

    #[test]
    fn test_two_components() {
        assert_eq!(vertex_format_components(VertexFormat::Uint8x2), 2);
        assert_eq!(vertex_format_components(VertexFormat::Sint16x2), 2);
        assert_eq!(vertex_format_components(VertexFormat::Float32x2), 2);
        assert_eq!(vertex_format_components(VertexFormat::Float64x2), 2);
    }

    #[test]
    fn test_three_components() {
        assert_eq!(vertex_format_components(VertexFormat::Uint32x3), 3);
        assert_eq!(vertex_format_components(VertexFormat::Sint32x3), 3);
        assert_eq!(vertex_format_components(VertexFormat::Float32x3), 3);
        assert_eq!(vertex_format_components(VertexFormat::Float64x3), 3);
    }

    #[test]
    fn test_four_components() {
        assert_eq!(vertex_format_components(VertexFormat::Uint8x4), 4);
        assert_eq!(vertex_format_components(VertexFormat::Float32x4), 4);
        assert_eq!(vertex_format_components(VertexFormat::Float16x4), 4);
        assert_eq!(vertex_format_components(VertexFormat::Float64x4), 4);
        assert_eq!(vertex_format_components(VertexFormat::Unorm10_10_10_2), 4);
    }

    // =========================================================================
    // Type Classification Tests
    // =========================================================================

    #[test]
    fn test_normalized_formats() {
        assert!(vertex_format_is_normalized(VertexFormat::Unorm8x2));
        assert!(vertex_format_is_normalized(VertexFormat::Snorm8x4));
        assert!(vertex_format_is_normalized(VertexFormat::Unorm16x2));
        assert!(vertex_format_is_normalized(VertexFormat::Snorm16x4));
        assert!(vertex_format_is_normalized(VertexFormat::Unorm10_10_10_2));

        assert!(!vertex_format_is_normalized(VertexFormat::Float32x3));
        assert!(!vertex_format_is_normalized(VertexFormat::Uint8x4));
    }

    #[test]
    fn test_float_formats() {
        assert!(vertex_format_is_float(VertexFormat::Float16x2));
        assert!(vertex_format_is_float(VertexFormat::Float32x3));
        assert!(vertex_format_is_float(VertexFormat::Float64x4));

        assert!(!vertex_format_is_float(VertexFormat::Uint32x4));
        assert!(!vertex_format_is_float(VertexFormat::Unorm8x4));
    }

    #[test]
    fn test_signed_int_formats() {
        assert!(vertex_format_is_signed_int(VertexFormat::Sint8x2));
        assert!(vertex_format_is_signed_int(VertexFormat::Sint16x4));
        assert!(vertex_format_is_signed_int(VertexFormat::Sint32x3));

        assert!(!vertex_format_is_signed_int(VertexFormat::Uint32));
        assert!(!vertex_format_is_signed_int(VertexFormat::Float32));
    }

    #[test]
    fn test_unsigned_int_formats() {
        assert!(vertex_format_is_unsigned_int(VertexFormat::Uint8x2));
        assert!(vertex_format_is_unsigned_int(VertexFormat::Uint16x4));
        assert!(vertex_format_is_unsigned_int(VertexFormat::Uint32x3));

        assert!(!vertex_format_is_unsigned_int(VertexFormat::Sint32));
        assert!(!vertex_format_is_unsigned_int(VertexFormat::Float32));
    }

    // =========================================================================
    // VertexFormatInfo Tests
    // =========================================================================

    #[test]
    fn test_vertex_format_info_float32x3() {
        let info = VertexFormatInfo::new(VertexFormat::Float32x3);
        assert_eq!(info.format, VertexFormat::Float32x3);
        assert_eq!(info.size, 12);
        assert_eq!(info.components, 3);
        assert!(!info.normalized);
        assert!(info.is_float);
        assert_eq!(info.bytes_per_component, 4);
    }

    #[test]
    fn test_vertex_format_info_unorm8x4() {
        let info = VertexFormatInfo::new(VertexFormat::Unorm8x4);
        assert_eq!(info.format, VertexFormat::Unorm8x4);
        assert_eq!(info.size, 4);
        assert_eq!(info.components, 4);
        assert!(info.normalized);
        assert!(!info.is_float);
        assert_eq!(info.bytes_per_component, 1);
    }

    #[test]
    fn test_vertex_format_info_from() {
        let info: VertexFormatInfo = VertexFormat::Float16x4.into();
        assert_eq!(info.size, 8);
        assert_eq!(info.components, 4);
    }

    // =========================================================================
    // Common Format Tests
    // =========================================================================

    #[test]
    fn test_common_geometry_formats() {
        assert_eq!(common::POSITION, VertexFormat::Float32x3);
        assert_eq!(common::POSITION_2D, VertexFormat::Float32x2);
        assert_eq!(common::NORMAL, VertexFormat::Float32x3);
        assert_eq!(common::TANGENT, VertexFormat::Float32x4);
        assert_eq!(common::NORMAL_COMPRESSED, VertexFormat::Snorm8x4);
    }

    #[test]
    fn test_common_uv_formats() {
        assert_eq!(common::UV, VertexFormat::Float32x2);
        assert_eq!(common::UV_HALF, VertexFormat::Float16x2);
        assert_eq!(common::UV_NORMALIZED, VertexFormat::Unorm16x2);
    }

    #[test]
    fn test_common_color_formats() {
        assert_eq!(common::COLOR, VertexFormat::Unorm8x4);
        assert_eq!(common::COLOR_HDR, VertexFormat::Float32x4);
        assert_eq!(common::COLOR_HDR_HALF, VertexFormat::Float16x4);
    }

    #[test]
    fn test_common_bone_formats() {
        assert_eq!(common::BONE_INDICES, VertexFormat::Uint8x4);
        assert_eq!(common::BONE_INDICES_LARGE, VertexFormat::Uint16x4);
        assert_eq!(common::BONE_WEIGHTS, VertexFormat::Unorm8x4);
        assert_eq!(common::BONE_WEIGHTS_FLOAT, VertexFormat::Float32x4);
    }

    // =========================================================================
    // Stride Calculation Tests
    // =========================================================================

    #[test]
    fn test_calculate_stride_pbr() {
        let stride = calculate_stride(&[
            VertexFormat::Float32x3, // position: 12
            VertexFormat::Float32x3, // normal: 12
            VertexFormat::Float32x2, // uv: 8
            VertexFormat::Float32x4, // tangent: 16
        ]);
        assert_eq!(stride, 48);
        assert_eq!(stride, strides::PBR);
    }

    #[test]
    fn test_calculate_stride_terrain() {
        let stride = calculate_stride(&[
            VertexFormat::Float32x3, // position: 12
            VertexFormat::Float32x3, // normal: 12
            VertexFormat::Float32x2, // uv: 8
        ]);
        assert_eq!(stride, 32);
        assert_eq!(stride, strides::TERRAIN);
    }

    #[test]
    fn test_calculate_stride_ui() {
        let stride = calculate_stride(&[
            VertexFormat::Float32x2, // position: 8
            VertexFormat::Float32x2, // uv: 8
            VertexFormat::Unorm8x4,  // color: 4
        ]);
        assert_eq!(stride, 20);
        assert_eq!(stride, strides::UI);
    }

    #[test]
    fn test_calculate_stride_empty() {
        let stride = calculate_stride(&[]);
        assert_eq!(stride, 0);
    }

    #[test]
    fn test_calculate_offsets() {
        let formats = [
            VertexFormat::Float32x3, // position: offset 0
            VertexFormat::Float32x3, // normal: offset 12
            VertexFormat::Float32x2, // uv: offset 24
        ];
        let offsets = calculate_offsets(&formats);
        assert_eq!(offsets, [0, 12, 24]);
    }

    #[test]
    fn test_calculate_offsets_pbr() {
        let formats = [
            VertexFormat::Float32x3, // position: 0
            VertexFormat::Float32x3, // normal: 12
            VertexFormat::Float32x2, // uv: 24
            VertexFormat::Float32x4, // tangent: 32
        ];
        let offsets = calculate_offsets(&formats);
        assert_eq!(offsets, [0, 12, 24, 32]);
    }

    // =========================================================================
    // Standard Stride Tests
    // =========================================================================

    #[test]
    fn test_standard_strides() {
        assert_eq!(strides::PBR, 48);
        assert_eq!(strides::SKINNED, 72);
        assert_eq!(strides::TERRAIN, 32);
        assert_eq!(strides::PARTICLE, 36);
        assert_eq!(strides::UI, 20);
        assert_eq!(strides::POSITION_ONLY, 12);
        assert_eq!(strides::SHADOW, 12);
    }

    // =========================================================================
    // All 33 Formats Exhaustive Test
    // =========================================================================

    #[test]
    fn test_all_formats_have_nonzero_size() {
        let all_formats = [
            // 8-bit (2 and 4 components only in wgpu 22.x)
            VertexFormat::Uint8x2, VertexFormat::Uint8x4,
            VertexFormat::Sint8x2, VertexFormat::Sint8x4,
            VertexFormat::Unorm8x2, VertexFormat::Unorm8x4,
            VertexFormat::Snorm8x2, VertexFormat::Snorm8x4,
            // 16-bit (2 and 4 components)
            VertexFormat::Uint16x2, VertexFormat::Uint16x4,
            VertexFormat::Sint16x2, VertexFormat::Sint16x4,
            VertexFormat::Unorm16x2, VertexFormat::Unorm16x4,
            VertexFormat::Snorm16x2, VertexFormat::Snorm16x4,
            VertexFormat::Float16x2, VertexFormat::Float16x4,
            // 32-bit (1, 2, 3, 4 components)
            VertexFormat::Uint32, VertexFormat::Uint32x2, VertexFormat::Uint32x3, VertexFormat::Uint32x4,
            VertexFormat::Sint32, VertexFormat::Sint32x2, VertexFormat::Sint32x3, VertexFormat::Sint32x4,
            VertexFormat::Float32, VertexFormat::Float32x2, VertexFormat::Float32x3, VertexFormat::Float32x4,
            // 64-bit (1, 2, 3, 4 components)
            VertexFormat::Float64, VertexFormat::Float64x2, VertexFormat::Float64x3, VertexFormat::Float64x4,
            // Packed formats
            VertexFormat::Unorm10_10_10_2,
        ];

        for format in all_formats {
            assert!(vertex_format_size(format) > 0, "Format {:?} should have non-zero size", format);
        }

        // Verify we have tested a reasonable number of formats
        // wgpu 22.x has 33 formats (32 base + 1 packed: Unorm10_10_10_2)
        assert!(all_formats.len() >= 33, "Should test at least 33 formats");
    }

    #[test]
    fn test_all_formats_component_count_valid() {
        let all_formats = [
            VertexFormat::Uint8x2, VertexFormat::Uint8x4,
            VertexFormat::Sint8x2, VertexFormat::Sint8x4,
            VertexFormat::Unorm8x2, VertexFormat::Unorm8x4,
            VertexFormat::Snorm8x2, VertexFormat::Snorm8x4,
            VertexFormat::Uint16x2, VertexFormat::Uint16x4,
            VertexFormat::Sint16x2, VertexFormat::Sint16x4,
            VertexFormat::Unorm16x2, VertexFormat::Unorm16x4,
            VertexFormat::Snorm16x2, VertexFormat::Snorm16x4,
            VertexFormat::Float16x2, VertexFormat::Float16x4,
            VertexFormat::Uint32, VertexFormat::Uint32x2, VertexFormat::Uint32x3, VertexFormat::Uint32x4,
            VertexFormat::Sint32, VertexFormat::Sint32x2, VertexFormat::Sint32x3, VertexFormat::Sint32x4,
            VertexFormat::Float32, VertexFormat::Float32x2, VertexFormat::Float32x3, VertexFormat::Float32x4,
            VertexFormat::Float64, VertexFormat::Float64x2, VertexFormat::Float64x3, VertexFormat::Float64x4,
            VertexFormat::Unorm10_10_10_2,
        ];

        for format in all_formats {
            let components = vertex_format_components(format);
            assert!(components >= 1 && components <= 4, "Format {:?} has invalid component count: {}", format, components);
        }
    }

    // =========================================================================
    // Const Function Tests (compile-time usage)
    // =========================================================================

    #[test]
    fn test_const_size_calculation() {
        const PBR_STRIDE: u64 = vertex_format_size(VertexFormat::Float32x3)
            + vertex_format_size(VertexFormat::Float32x3)
            + vertex_format_size(VertexFormat::Float32x2)
            + vertex_format_size(VertexFormat::Float32x4);
        assert_eq!(PBR_STRIDE, 48);
    }

    #[test]
    fn test_const_component_count() {
        const COMPONENTS: u32 = vertex_format_components(VertexFormat::Float32x3);
        assert_eq!(COMPONENTS, 3);
    }

    // =========================================================================
    // WHITEBOX T-WGPU-P3.2.2: Comprehensive Format Size Tests
    // =========================================================================

    #[test]
    fn test_all_8bit_formats_exact_sizes() {
        // 8-bit unsigned integer
        assert_eq!(vertex_format_size(VertexFormat::Uint8x2), 2, "Uint8x2 should be 2 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Uint8x4), 4, "Uint8x4 should be 4 bytes");
        // 8-bit signed integer
        assert_eq!(vertex_format_size(VertexFormat::Sint8x2), 2, "Sint8x2 should be 2 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Sint8x4), 4, "Sint8x4 should be 4 bytes");
        // 8-bit unsigned normalized
        assert_eq!(vertex_format_size(VertexFormat::Unorm8x2), 2, "Unorm8x2 should be 2 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Unorm8x4), 4, "Unorm8x4 should be 4 bytes");
        // 8-bit signed normalized
        assert_eq!(vertex_format_size(VertexFormat::Snorm8x2), 2, "Snorm8x2 should be 2 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Snorm8x4), 4, "Snorm8x4 should be 4 bytes");
    }

    #[test]
    fn test_all_16bit_formats_exact_sizes() {
        // 16-bit unsigned integer
        assert_eq!(vertex_format_size(VertexFormat::Uint16x2), 4, "Uint16x2 should be 4 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Uint16x4), 8, "Uint16x4 should be 8 bytes");
        // 16-bit signed integer
        assert_eq!(vertex_format_size(VertexFormat::Sint16x2), 4, "Sint16x2 should be 4 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Sint16x4), 8, "Sint16x4 should be 8 bytes");
        // 16-bit unsigned normalized
        assert_eq!(vertex_format_size(VertexFormat::Unorm16x2), 4, "Unorm16x2 should be 4 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Unorm16x4), 8, "Unorm16x4 should be 8 bytes");
        // 16-bit signed normalized
        assert_eq!(vertex_format_size(VertexFormat::Snorm16x2), 4, "Snorm16x2 should be 4 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Snorm16x4), 8, "Snorm16x4 should be 8 bytes");
        // 16-bit float
        assert_eq!(vertex_format_size(VertexFormat::Float16x2), 4, "Float16x2 should be 4 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Float16x4), 8, "Float16x4 should be 8 bytes");
    }

    #[test]
    fn test_all_32bit_formats_exact_sizes() {
        // 32-bit unsigned integer
        assert_eq!(vertex_format_size(VertexFormat::Uint32), 4, "Uint32 should be 4 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Uint32x2), 8, "Uint32x2 should be 8 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Uint32x3), 12, "Uint32x3 should be 12 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Uint32x4), 16, "Uint32x4 should be 16 bytes");
        // 32-bit signed integer
        assert_eq!(vertex_format_size(VertexFormat::Sint32), 4, "Sint32 should be 4 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Sint32x2), 8, "Sint32x2 should be 8 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Sint32x3), 12, "Sint32x3 should be 12 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Sint32x4), 16, "Sint32x4 should be 16 bytes");
        // 32-bit float
        assert_eq!(vertex_format_size(VertexFormat::Float32), 4, "Float32 should be 4 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Float32x2), 8, "Float32x2 should be 8 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Float32x3), 12, "Float32x3 should be 12 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Float32x4), 16, "Float32x4 should be 16 bytes");
    }

    #[test]
    fn test_all_64bit_formats_exact_sizes() {
        assert_eq!(vertex_format_size(VertexFormat::Float64), 8, "Float64 should be 8 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Float64x2), 16, "Float64x2 should be 16 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Float64x3), 24, "Float64x3 should be 24 bytes");
        assert_eq!(vertex_format_size(VertexFormat::Float64x4), 32, "Float64x4 should be 32 bytes");
    }

    #[test]
    fn test_packed_format_exact_size() {
        assert_eq!(vertex_format_size(VertexFormat::Unorm10_10_10_2), 4, "Unorm10_10_10_2 should be 4 bytes");
    }

    // =========================================================================
    // WHITEBOX T-WGPU-P3.2.2: Component Count Exhaustive Tests
    // =========================================================================

    #[test]
    fn test_all_x2_formats_have_2_components() {
        let x2_formats = [
            VertexFormat::Uint8x2,
            VertexFormat::Sint8x2,
            VertexFormat::Unorm8x2,
            VertexFormat::Snorm8x2,
            VertexFormat::Uint16x2,
            VertexFormat::Sint16x2,
            VertexFormat::Unorm16x2,
            VertexFormat::Snorm16x2,
            VertexFormat::Float16x2,
            VertexFormat::Uint32x2,
            VertexFormat::Sint32x2,
            VertexFormat::Float32x2,
            VertexFormat::Float64x2,
        ];
        for format in x2_formats {
            assert_eq!(
                vertex_format_components(format), 2,
                "Format {:?} should have 2 components", format
            );
        }
    }

    #[test]
    fn test_all_x3_formats_have_3_components() {
        let x3_formats = [
            VertexFormat::Uint32x3,
            VertexFormat::Sint32x3,
            VertexFormat::Float32x3,
            VertexFormat::Float64x3,
        ];
        for format in x3_formats {
            assert_eq!(
                vertex_format_components(format), 3,
                "Format {:?} should have 3 components", format
            );
        }
    }

    #[test]
    fn test_all_x4_formats_have_4_components() {
        let x4_formats = [
            VertexFormat::Uint8x4,
            VertexFormat::Sint8x4,
            VertexFormat::Unorm8x4,
            VertexFormat::Snorm8x4,
            VertexFormat::Uint16x4,
            VertexFormat::Sint16x4,
            VertexFormat::Unorm16x4,
            VertexFormat::Snorm16x4,
            VertexFormat::Float16x4,
            VertexFormat::Uint32x4,
            VertexFormat::Sint32x4,
            VertexFormat::Float32x4,
            VertexFormat::Float64x4,
            VertexFormat::Unorm10_10_10_2,
        ];
        for format in x4_formats {
            assert_eq!(
                vertex_format_components(format), 4,
                "Format {:?} should have 4 components", format
            );
        }
    }

    #[test]
    fn test_single_component_formats() {
        let single_formats = [
            VertexFormat::Uint32,
            VertexFormat::Sint32,
            VertexFormat::Float32,
            VertexFormat::Float64,
        ];
        for format in single_formats {
            assert_eq!(
                vertex_format_components(format), 1,
                "Format {:?} should have 1 component", format
            );
        }
    }

    // =========================================================================
    // WHITEBOX T-WGPU-P3.2.2: Type Classification Exhaustive Tests
    // =========================================================================

    #[test]
    fn test_all_normalized_formats_exhaustive() {
        let normalized_formats = [
            VertexFormat::Unorm8x2,
            VertexFormat::Unorm8x4,
            VertexFormat::Snorm8x2,
            VertexFormat::Snorm8x4,
            VertexFormat::Unorm16x2,
            VertexFormat::Unorm16x4,
            VertexFormat::Snorm16x2,
            VertexFormat::Snorm16x4,
            VertexFormat::Unorm10_10_10_2,
        ];
        for format in normalized_formats {
            assert!(
                vertex_format_is_normalized(format),
                "Format {:?} should be normalized", format
            );
        }
    }

    #[test]
    fn test_non_normalized_formats_exhaustive() {
        let non_normalized_formats = [
            VertexFormat::Uint8x2, VertexFormat::Uint8x4,
            VertexFormat::Sint8x2, VertexFormat::Sint8x4,
            VertexFormat::Uint16x2, VertexFormat::Uint16x4,
            VertexFormat::Sint16x2, VertexFormat::Sint16x4,
            VertexFormat::Float16x2, VertexFormat::Float16x4,
            VertexFormat::Uint32, VertexFormat::Uint32x2, VertexFormat::Uint32x3, VertexFormat::Uint32x4,
            VertexFormat::Sint32, VertexFormat::Sint32x2, VertexFormat::Sint32x3, VertexFormat::Sint32x4,
            VertexFormat::Float32, VertexFormat::Float32x2, VertexFormat::Float32x3, VertexFormat::Float32x4,
            VertexFormat::Float64, VertexFormat::Float64x2, VertexFormat::Float64x3, VertexFormat::Float64x4,
        ];
        for format in non_normalized_formats {
            assert!(
                !vertex_format_is_normalized(format),
                "Format {:?} should NOT be normalized", format
            );
        }
    }

    #[test]
    fn test_all_float_formats_exhaustive() {
        let float_formats = [
            VertexFormat::Float16x2,
            VertexFormat::Float16x4,
            VertexFormat::Float32,
            VertexFormat::Float32x2,
            VertexFormat::Float32x3,
            VertexFormat::Float32x4,
            VertexFormat::Float64,
            VertexFormat::Float64x2,
            VertexFormat::Float64x3,
            VertexFormat::Float64x4,
        ];
        for format in float_formats {
            assert!(
                vertex_format_is_float(format),
                "Format {:?} should be float", format
            );
        }
    }

    #[test]
    fn test_non_float_formats_exhaustive() {
        let non_float_formats = [
            VertexFormat::Uint8x2, VertexFormat::Uint8x4,
            VertexFormat::Sint8x2, VertexFormat::Sint8x4,
            VertexFormat::Unorm8x2, VertexFormat::Unorm8x4,
            VertexFormat::Snorm8x2, VertexFormat::Snorm8x4,
            VertexFormat::Uint16x2, VertexFormat::Uint16x4,
            VertexFormat::Sint16x2, VertexFormat::Sint16x4,
            VertexFormat::Unorm16x2, VertexFormat::Unorm16x4,
            VertexFormat::Snorm16x2, VertexFormat::Snorm16x4,
            VertexFormat::Uint32, VertexFormat::Uint32x2, VertexFormat::Uint32x3, VertexFormat::Uint32x4,
            VertexFormat::Sint32, VertexFormat::Sint32x2, VertexFormat::Sint32x3, VertexFormat::Sint32x4,
            VertexFormat::Unorm10_10_10_2,
        ];
        for format in non_float_formats {
            assert!(
                !vertex_format_is_float(format),
                "Format {:?} should NOT be float", format
            );
        }
    }

    #[test]
    fn test_all_signed_int_formats_exhaustive() {
        let signed_int_formats = [
            VertexFormat::Sint8x2,
            VertexFormat::Sint8x4,
            VertexFormat::Sint16x2,
            VertexFormat::Sint16x4,
            VertexFormat::Sint32,
            VertexFormat::Sint32x2,
            VertexFormat::Sint32x3,
            VertexFormat::Sint32x4,
        ];
        for format in signed_int_formats {
            assert!(
                vertex_format_is_signed_int(format),
                "Format {:?} should be signed int", format
            );
        }
    }

    #[test]
    fn test_all_unsigned_int_formats_exhaustive() {
        let unsigned_int_formats = [
            VertexFormat::Uint8x2,
            VertexFormat::Uint8x4,
            VertexFormat::Uint16x2,
            VertexFormat::Uint16x4,
            VertexFormat::Uint32,
            VertexFormat::Uint32x2,
            VertexFormat::Uint32x3,
            VertexFormat::Uint32x4,
        ];
        for format in unsigned_int_formats {
            assert!(
                vertex_format_is_unsigned_int(format),
                "Format {:?} should be unsigned int", format
            );
        }
    }

    #[test]
    fn test_type_classification_mutual_exclusion() {
        // Float formats should not be int
        assert!(!vertex_format_is_signed_int(VertexFormat::Float32));
        assert!(!vertex_format_is_unsigned_int(VertexFormat::Float32));
        // Signed int should not be unsigned int
        assert!(!vertex_format_is_unsigned_int(VertexFormat::Sint32));
        // Unsigned int should not be signed int
        assert!(!vertex_format_is_signed_int(VertexFormat::Uint32));
        // Normalized formats should not be int
        assert!(!vertex_format_is_signed_int(VertexFormat::Unorm8x4));
        assert!(!vertex_format_is_unsigned_int(VertexFormat::Unorm8x4));
    }

    // =========================================================================
    // WHITEBOX T-WGPU-P3.2.2: Calculate Stride Additional Tests
    // =========================================================================

    #[test]
    fn test_calculate_stride_skinned_layout() {
        let stride = calculate_stride(&[
            VertexFormat::Float32x3,  // position: 12
            VertexFormat::Float32x3,  // normal: 12
            VertexFormat::Float32x2,  // uv: 8
            VertexFormat::Float32x4,  // tangent: 16
            VertexFormat::Uint16x4,   // bone_indices: 8
            VertexFormat::Float32x4,  // bone_weights: 16
        ]);
        assert_eq!(stride, 72);
        assert_eq!(stride, strides::SKINNED);
    }

    #[test]
    fn test_calculate_stride_particle_layout() {
        let stride = calculate_stride(&[
            VertexFormat::Float32x3,  // position: 12
            VertexFormat::Float32x4,  // color: 16
            VertexFormat::Float32x2,  // size_rotation: 8
        ]);
        assert_eq!(stride, 36);
        assert_eq!(stride, strides::PARTICLE);
    }

    #[test]
    fn test_calculate_stride_position_only() {
        let stride = calculate_stride(&[
            VertexFormat::Float32x3,  // position: 12
        ]);
        assert_eq!(stride, 12);
        assert_eq!(stride, strides::POSITION_ONLY);
        assert_eq!(stride, strides::SHADOW);
    }

    #[test]
    fn test_calculate_stride_mixed_bit_widths() {
        let stride = calculate_stride(&[
            VertexFormat::Float32x3,  // 12 bytes (32-bit)
            VertexFormat::Unorm8x4,   // 4 bytes (8-bit)
            VertexFormat::Float16x2,  // 4 bytes (16-bit)
            VertexFormat::Float64x2,  // 16 bytes (64-bit)
        ]);
        assert_eq!(stride, 36);
    }

    #[test]
    fn test_calculate_stride_single_format() {
        assert_eq!(calculate_stride(&[VertexFormat::Float32]), 4);
        assert_eq!(calculate_stride(&[VertexFormat::Float64x4]), 32);
        assert_eq!(calculate_stride(&[VertexFormat::Uint8x2]), 2);
    }

    #[test]
    fn test_calculate_stride_all_64bit() {
        let stride = calculate_stride(&[
            VertexFormat::Float64,    // 8
            VertexFormat::Float64x2,  // 16
            VertexFormat::Float64x3,  // 24
            VertexFormat::Float64x4,  // 32
        ]);
        assert_eq!(stride, 80);
    }

    // =========================================================================
    // WHITEBOX T-WGPU-P3.2.2: Calculate Offsets Additional Tests
    // =========================================================================

    #[test]
    fn test_calculate_offsets_skinned() {
        let formats = [
            VertexFormat::Float32x3,  // position: offset 0
            VertexFormat::Float32x3,  // normal: offset 12
            VertexFormat::Float32x2,  // uv: offset 24
            VertexFormat::Float32x4,  // tangent: offset 32
            VertexFormat::Uint16x4,   // bone_indices: offset 48
            VertexFormat::Float32x4,  // bone_weights: offset 56
        ];
        let offsets = calculate_offsets(&formats);
        assert_eq!(offsets, [0, 12, 24, 32, 48, 56]);
    }

    #[test]
    fn test_calculate_offsets_ui() {
        let formats = [
            VertexFormat::Float32x2,  // position: offset 0
            VertexFormat::Float32x2,  // uv: offset 8
            VertexFormat::Unorm8x4,   // color: offset 16
        ];
        let offsets = calculate_offsets(&formats);
        assert_eq!(offsets, [0, 8, 16]);
    }

    #[test]
    fn test_calculate_offsets_single_element() {
        let formats = [VertexFormat::Float32x3];
        let offsets = calculate_offsets(&formats);
        assert_eq!(offsets, [0]);
    }

    #[test]
    fn test_calculate_offsets_two_elements() {
        let formats = [VertexFormat::Float32x3, VertexFormat::Float32x2];
        let offsets = calculate_offsets(&formats);
        assert_eq!(offsets, [0, 12]);
    }

    #[test]
    fn test_calculate_offsets_mixed_sizes() {
        let formats = [
            VertexFormat::Uint8x2,    // offset 0, size 2
            VertexFormat::Float32,    // offset 2, size 4
            VertexFormat::Float16x4,  // offset 6, size 8
            VertexFormat::Float64,    // offset 14, size 8
        ];
        let offsets = calculate_offsets(&formats);
        assert_eq!(offsets, [0, 2, 6, 14]);
    }

    // =========================================================================
    // WHITEBOX T-WGPU-P3.2.2: Common Presets Format and Size Tests
    // =========================================================================

    #[test]
    fn test_common_position_format_and_size() {
        assert_eq!(common::POSITION, VertexFormat::Float32x3);
        assert_eq!(vertex_format_size(common::POSITION), 12);
        assert_eq!(vertex_format_components(common::POSITION), 3);
    }

    #[test]
    fn test_common_position_2d_format_and_size() {
        assert_eq!(common::POSITION_2D, VertexFormat::Float32x2);
        assert_eq!(vertex_format_size(common::POSITION_2D), 8);
        assert_eq!(vertex_format_components(common::POSITION_2D), 2);
    }

    #[test]
    fn test_common_normal_format_and_size() {
        assert_eq!(common::NORMAL, VertexFormat::Float32x3);
        assert_eq!(vertex_format_size(common::NORMAL), 12);
        assert_eq!(vertex_format_components(common::NORMAL), 3);
    }

    #[test]
    fn test_common_normal_compressed_format_and_size() {
        assert_eq!(common::NORMAL_COMPRESSED, VertexFormat::Snorm8x4);
        assert_eq!(vertex_format_size(common::NORMAL_COMPRESSED), 4);
        assert_eq!(vertex_format_components(common::NORMAL_COMPRESSED), 4);
        assert!(vertex_format_is_normalized(common::NORMAL_COMPRESSED));
    }

    #[test]
    fn test_common_tangent_format_and_size() {
        assert_eq!(common::TANGENT, VertexFormat::Float32x4);
        assert_eq!(vertex_format_size(common::TANGENT), 16);
        assert_eq!(vertex_format_components(common::TANGENT), 4);
    }

    #[test]
    fn test_common_tangent_compressed_format_and_size() {
        assert_eq!(common::TANGENT_COMPRESSED, VertexFormat::Snorm8x4);
        assert_eq!(vertex_format_size(common::TANGENT_COMPRESSED), 4);
        assert!(vertex_format_is_normalized(common::TANGENT_COMPRESSED));
    }

    #[test]
    fn test_common_normal_packed_1010102_format_and_size() {
        assert_eq!(common::NORMAL_PACKED_1010102, VertexFormat::Unorm10_10_10_2);
        assert_eq!(vertex_format_size(common::NORMAL_PACKED_1010102), 4);
        assert_eq!(vertex_format_components(common::NORMAL_PACKED_1010102), 4);
        assert!(vertex_format_is_normalized(common::NORMAL_PACKED_1010102));
    }

    #[test]
    fn test_common_uv_format_and_size() {
        assert_eq!(common::UV, VertexFormat::Float32x2);
        assert_eq!(vertex_format_size(common::UV), 8);
        assert_eq!(vertex_format_components(common::UV), 2);
    }

    #[test]
    fn test_common_uv_half_format_and_size() {
        assert_eq!(common::UV_HALF, VertexFormat::Float16x2);
        assert_eq!(vertex_format_size(common::UV_HALF), 4);
        assert_eq!(vertex_format_components(common::UV_HALF), 2);
    }

    #[test]
    fn test_common_uv_normalized_format_and_size() {
        assert_eq!(common::UV_NORMALIZED, VertexFormat::Unorm16x2);
        assert_eq!(vertex_format_size(common::UV_NORMALIZED), 4);
        assert!(vertex_format_is_normalized(common::UV_NORMALIZED));
    }

    #[test]
    fn test_common_color_format_and_size() {
        assert_eq!(common::COLOR, VertexFormat::Unorm8x4);
        assert_eq!(vertex_format_size(common::COLOR), 4);
        assert_eq!(vertex_format_components(common::COLOR), 4);
        assert!(vertex_format_is_normalized(common::COLOR));
    }

    #[test]
    fn test_common_color_hdr_format_and_size() {
        assert_eq!(common::COLOR_HDR, VertexFormat::Float32x4);
        assert_eq!(vertex_format_size(common::COLOR_HDR), 16);
        assert!(vertex_format_is_float(common::COLOR_HDR));
    }

    #[test]
    fn test_common_color_hdr_half_format_and_size() {
        assert_eq!(common::COLOR_HDR_HALF, VertexFormat::Float16x4);
        assert_eq!(vertex_format_size(common::COLOR_HDR_HALF), 8);
        assert!(vertex_format_is_float(common::COLOR_HDR_HALF));
    }

    #[test]
    fn test_common_bone_indices_format_and_size() {
        assert_eq!(common::BONE_INDICES, VertexFormat::Uint8x4);
        assert_eq!(vertex_format_size(common::BONE_INDICES), 4);
        assert!(vertex_format_is_unsigned_int(common::BONE_INDICES));
    }

    #[test]
    fn test_common_bone_indices_large_format_and_size() {
        assert_eq!(common::BONE_INDICES_LARGE, VertexFormat::Uint16x4);
        assert_eq!(vertex_format_size(common::BONE_INDICES_LARGE), 8);
        assert!(vertex_format_is_unsigned_int(common::BONE_INDICES_LARGE));
    }

    #[test]
    fn test_common_bone_weights_format_and_size() {
        assert_eq!(common::BONE_WEIGHTS, VertexFormat::Unorm8x4);
        assert_eq!(vertex_format_size(common::BONE_WEIGHTS), 4);
        assert!(vertex_format_is_normalized(common::BONE_WEIGHTS));
    }

    #[test]
    fn test_common_bone_weights_float_format_and_size() {
        assert_eq!(common::BONE_WEIGHTS_FLOAT, VertexFormat::Float32x4);
        assert_eq!(vertex_format_size(common::BONE_WEIGHTS_FLOAT), 16);
        assert!(vertex_format_is_float(common::BONE_WEIGHTS_FLOAT));
    }

    #[test]
    fn test_common_instance_matrix_row_format_and_size() {
        assert_eq!(common::INSTANCE_MATRIX_ROW, VertexFormat::Float32x4);
        assert_eq!(vertex_format_size(common::INSTANCE_MATRIX_ROW), 16);
        // Full 4x4 matrix = 4 rows * 16 bytes = 64 bytes
        assert_eq!(vertex_format_size(common::INSTANCE_MATRIX_ROW) * 4, 64);
    }

    #[test]
    fn test_common_instance_id_format_and_size() {
        assert_eq!(common::INSTANCE_ID, VertexFormat::Uint32);
        assert_eq!(vertex_format_size(common::INSTANCE_ID), 4);
        assert!(vertex_format_is_unsigned_int(common::INSTANCE_ID));
    }

    #[test]
    fn test_common_particle_size_format_and_size() {
        assert_eq!(common::PARTICLE_SIZE, VertexFormat::Float32);
        assert_eq!(vertex_format_size(common::PARTICLE_SIZE), 4);
        assert_eq!(vertex_format_components(common::PARTICLE_SIZE), 1);
    }

    #[test]
    fn test_common_particle_size_2d_format_and_size() {
        assert_eq!(common::PARTICLE_SIZE_2D, VertexFormat::Float32x2);
        assert_eq!(vertex_format_size(common::PARTICLE_SIZE_2D), 8);
        assert_eq!(vertex_format_components(common::PARTICLE_SIZE_2D), 2);
    }

    #[test]
    fn test_common_particle_rotation_format_and_size() {
        assert_eq!(common::PARTICLE_ROTATION, VertexFormat::Float32);
        assert_eq!(vertex_format_size(common::PARTICLE_ROTATION), 4);
    }

    #[test]
    fn test_common_particle_life_format_and_size() {
        assert_eq!(common::PARTICLE_LIFE, VertexFormat::Float32);
        assert_eq!(vertex_format_size(common::PARTICLE_LIFE), 4);
    }

    #[test]
    fn test_common_particle_velocity_format_and_size() {
        assert_eq!(common::PARTICLE_VELOCITY, VertexFormat::Float32x3);
        assert_eq!(vertex_format_size(common::PARTICLE_VELOCITY), 12);
    }

    // =========================================================================
    // WHITEBOX T-WGPU-P3.2.2: Strides Presets Validation
    // =========================================================================

    #[test]
    fn test_strides_pbr_calculation() {
        // PBR = position(12) + normal(12) + uv(8) + tangent(16) = 48
        let calculated = calculate_stride(&[
            common::POSITION,
            common::NORMAL,
            common::UV,
            common::TANGENT,
        ]);
        assert_eq!(calculated, strides::PBR);
        assert_eq!(strides::PBR, 48);
    }

    #[test]
    fn test_strides_skinned_calculation() {
        // SKINNED = PBR(48) + bone_indices(8) + bone_weights(16) = 72
        let calculated = calculate_stride(&[
            common::POSITION,
            common::NORMAL,
            common::UV,
            common::TANGENT,
            common::BONE_INDICES_LARGE,
            common::BONE_WEIGHTS_FLOAT,
        ]);
        assert_eq!(calculated, strides::SKINNED);
        assert_eq!(strides::SKINNED, 72);
    }

    #[test]
    fn test_strides_terrain_calculation() {
        // TERRAIN = position(12) + normal(12) + uv(8) = 32
        let calculated = calculate_stride(&[
            common::POSITION,
            common::NORMAL,
            common::UV,
        ]);
        assert_eq!(calculated, strides::TERRAIN);
        assert_eq!(strides::TERRAIN, 32);
    }

    #[test]
    fn test_strides_particle_calculation() {
        // PARTICLE = position(12) + color_hdr(16) + size_rotation(8) = 36
        let calculated = calculate_stride(&[
            common::POSITION,
            common::COLOR_HDR,
            common::PARTICLE_SIZE_2D,
        ]);
        assert_eq!(calculated, strides::PARTICLE);
        assert_eq!(strides::PARTICLE, 36);
    }

    #[test]
    fn test_strides_ui_calculation() {
        // UI = position_2d(8) + uv(8) + color(4) = 20
        let calculated = calculate_stride(&[
            common::POSITION_2D,
            common::UV,
            common::COLOR,
        ]);
        assert_eq!(calculated, strides::UI);
        assert_eq!(strides::UI, 20);
    }

    #[test]
    fn test_strides_position_only_calculation() {
        let calculated = calculate_stride(&[common::POSITION]);
        assert_eq!(calculated, strides::POSITION_ONLY);
        assert_eq!(strides::POSITION_ONLY, 12);
    }

    #[test]
    fn test_strides_shadow_equals_position_only() {
        assert_eq!(strides::SHADOW, strides::POSITION_ONLY);
        assert_eq!(strides::SHADOW, 12);
    }

    // =========================================================================
    // WHITEBOX T-WGPU-P3.2.2: VertexFormatInfo Additional Tests
    // =========================================================================

    #[test]
    fn test_vertex_format_info_bytes_per_component_8bit() {
        let info = VertexFormatInfo::new(VertexFormat::Uint8x4);
        assert_eq!(info.bytes_per_component, 1);
    }

    #[test]
    fn test_vertex_format_info_bytes_per_component_16bit() {
        let info = VertexFormatInfo::new(VertexFormat::Float16x4);
        assert_eq!(info.bytes_per_component, 2);
    }

    #[test]
    fn test_vertex_format_info_bytes_per_component_32bit() {
        let info = VertexFormatInfo::new(VertexFormat::Float32x4);
        assert_eq!(info.bytes_per_component, 4);
    }

    #[test]
    fn test_vertex_format_info_bytes_per_component_64bit() {
        let info = VertexFormatInfo::new(VertexFormat::Float64x4);
        assert_eq!(info.bytes_per_component, 8);
    }

    #[test]
    fn test_vertex_format_info_packed_format() {
        let info = VertexFormatInfo::new(VertexFormat::Unorm10_10_10_2);
        assert_eq!(info.size, 4);
        assert_eq!(info.components, 4);
        assert!(info.normalized);
        assert!(!info.is_float);
        // Packed format has 1 byte per component average
        assert_eq!(info.bytes_per_component, 1);
    }

    #[test]
    fn test_vertex_format_info_sint_format() {
        let info = VertexFormatInfo::new(VertexFormat::Sint32x3);
        assert_eq!(info.size, 12);
        assert_eq!(info.components, 3);
        assert!(!info.normalized);
        assert!(!info.is_float);
        assert_eq!(info.bytes_per_component, 4);
    }

    #[test]
    fn test_vertex_format_info_snorm_format() {
        let info = VertexFormatInfo::new(VertexFormat::Snorm16x4);
        assert_eq!(info.size, 8);
        assert_eq!(info.components, 4);
        assert!(info.normalized);
        assert!(!info.is_float);
        assert_eq!(info.bytes_per_component, 2);
    }

    // =========================================================================
    // WHITEBOX T-WGPU-P3.2.2: Edge Case Tests
    // =========================================================================

    #[test]
    fn test_format_size_matches_component_calculation() {
        // For all formats: size = components * bytes_per_component
        let all_formats = [
            VertexFormat::Uint8x2, VertexFormat::Uint8x4,
            VertexFormat::Sint8x2, VertexFormat::Sint8x4,
            VertexFormat::Unorm8x2, VertexFormat::Unorm8x4,
            VertexFormat::Snorm8x2, VertexFormat::Snorm8x4,
            VertexFormat::Uint16x2, VertexFormat::Uint16x4,
            VertexFormat::Sint16x2, VertexFormat::Sint16x4,
            VertexFormat::Unorm16x2, VertexFormat::Unorm16x4,
            VertexFormat::Snorm16x2, VertexFormat::Snorm16x4,
            VertexFormat::Float16x2, VertexFormat::Float16x4,
            VertexFormat::Uint32, VertexFormat::Uint32x2, VertexFormat::Uint32x3, VertexFormat::Uint32x4,
            VertexFormat::Sint32, VertexFormat::Sint32x2, VertexFormat::Sint32x3, VertexFormat::Sint32x4,
            VertexFormat::Float32, VertexFormat::Float32x2, VertexFormat::Float32x3, VertexFormat::Float32x4,
            VertexFormat::Float64, VertexFormat::Float64x2, VertexFormat::Float64x3, VertexFormat::Float64x4,
        ];

        for format in all_formats {
            let info = VertexFormatInfo::new(format);
            let expected_size = info.components as u64 * info.bytes_per_component as u64;
            assert_eq!(
                info.size, expected_size,
                "Format {:?}: size {} != components {} * bytes_per_component {}",
                format, info.size, info.components, info.bytes_per_component
            );
        }
    }

    #[test]
    fn test_format_count_matches_implementation() {
        // Count all formats implemented in vertex_format_size
        // 8-bit: 8 formats (Uint8x2, Uint8x4, Sint8x2, Sint8x4, Unorm8x2, Unorm8x4, Snorm8x2, Snorm8x4)
        // 16-bit: 10 formats (Uint16x2, Uint16x4, Sint16x2, Sint16x4, Unorm16x2, Unorm16x4, Snorm16x2, Snorm16x4, Float16x2, Float16x4)
        // 32-bit: 12 formats (Uint32, Uint32x2, Uint32x3, Uint32x4, Sint32, Sint32x2, Sint32x3, Sint32x4, Float32, Float32x2, Float32x3, Float32x4)
        // 64-bit: 4 formats (Float64, Float64x2, Float64x3, Float64x4)
        // Packed: 1 format (Unorm10_10_10_2)
        // Total = 8 + 10 + 12 + 4 + 1 = 35 formats in wgpu 22.x
        let all_formats = [
            // 8-bit: 8 formats
            VertexFormat::Uint8x2, VertexFormat::Uint8x4,
            VertexFormat::Sint8x2, VertexFormat::Sint8x4,
            VertexFormat::Unorm8x2, VertexFormat::Unorm8x4,
            VertexFormat::Snorm8x2, VertexFormat::Snorm8x4,
            // 16-bit: 10 formats
            VertexFormat::Uint16x2, VertexFormat::Uint16x4,
            VertexFormat::Sint16x2, VertexFormat::Sint16x4,
            VertexFormat::Unorm16x2, VertexFormat::Unorm16x4,
            VertexFormat::Snorm16x2, VertexFormat::Snorm16x4,
            VertexFormat::Float16x2, VertexFormat::Float16x4,
            // 32-bit: 12 formats
            VertexFormat::Uint32, VertexFormat::Uint32x2, VertexFormat::Uint32x3, VertexFormat::Uint32x4,
            VertexFormat::Sint32, VertexFormat::Sint32x2, VertexFormat::Sint32x3, VertexFormat::Sint32x4,
            VertexFormat::Float32, VertexFormat::Float32x2, VertexFormat::Float32x3, VertexFormat::Float32x4,
            // 64-bit: 4 formats
            VertexFormat::Float64, VertexFormat::Float64x2, VertexFormat::Float64x3, VertexFormat::Float64x4,
            // Packed: 1 format
            VertexFormat::Unorm10_10_10_2,
        ];
        // wgpu 22.x has 35 vertex formats total
        assert_eq!(all_formats.len(), 35, "wgpu 22.x should have 35 vertex formats");

        // Verify all formats work with our functions
        for format in all_formats {
            assert!(vertex_format_size(format) > 0);
            assert!(vertex_format_components(format) >= 1 && vertex_format_components(format) <= 4);
        }
    }
}

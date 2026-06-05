//! Vertex format definitions and registry for TRINITY.
//!
//! This module provides standard vertex layouts for common rendering scenarios:
//!
//! - [`PbrVertex`] - Standard PBR rendering (48 bytes)
//! - [`SkinnedVertex`] - Skeletal animation (72 bytes)
//! - [`TerrainVertex`] - Terrain rendering (32 bytes)
//! - [`ParticleVertex`] - GPU particle systems (32 bytes)
//! - [`UiVertex`] - UI/2D rendering (20 bytes)
//!
//! The [`VertexFormatRegistry`] provides runtime lookup of vertex layouts by ID,
//! supporting both standard layouts and custom user-defined formats.
//!
//! # Example
//!
//! ```no_run
//! use renderer_backend::resources::vertex::{VertexFormatRegistry, VertexLayoutId, PbrVertex};
//!
//! let registry = VertexFormatRegistry::new();
//!
//! // Get a standard layout
//! let pbr_layout = registry.get(VertexLayoutId::Pbr).unwrap();
//! assert_eq!(pbr_layout.array_stride, std::mem::size_of::<PbrVertex>() as u64);
//!
//! // Check vertex size at compile time
//! const _: () = assert!(std::mem::size_of::<PbrVertex>() == 48);
//! ```
//!
//! # Layout Sizes
//!
//! | Layout | Size (bytes) | Components |
//! |--------|--------------|------------|
//! | PBR | 48 | position, normal, uv, tangent |
//! | Skinned | 72 | PBR + joints (u16x4), weights |
//! | Terrain | 32 | position, normal, uv |
//! | Particle | 32 | position, color, size, life, rotation |
//! | UI | 20 | position (2D), uv, color (packed) |

use std::collections::HashMap;
use wgpu::{vertex_attr_array, VertexAttribute, VertexBufferLayout, VertexStepMode};

// ============================================================================
// Vertex Layout IDs
// ============================================================================

/// Standard vertex layout identifiers.
///
/// These IDs are used to look up vertex buffer layouts in the [`VertexFormatRegistry`].
/// Custom layouts can be registered using [`VertexLayoutId::Custom`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VertexLayoutId {
    /// Standard PBR layout: position, normal, uv, tangent (48 bytes).
    ///
    /// Used for most static mesh rendering with normal mapping support.
    Pbr,

    /// Skinned mesh layout: PBR + joints, weights (72 bytes).
    ///
    /// Extended PBR layout for skeletal animation with up to 4 bone influences.
    Skinned,

    /// Terrain layout: position, normal, uv (32 bytes).
    ///
    /// Simplified layout for terrain rendering without tangent space.
    Terrain,

    /// Particle layout: position, color, size, life, rotation (32 bytes).
    ///
    /// Per-instance particle data for GPU particle systems.
    Particle,

    /// UI layout: position (2D), uv, color (20 bytes).
    ///
    /// Minimal layout for 2D UI rendering with packed color.
    Ui,

    /// Custom user-defined layout.
    ///
    /// Register custom layouts via [`VertexFormatRegistry::register`].
    Custom(u32),
}

// ============================================================================
// PBR Vertex (48 bytes)
// ============================================================================

/// Standard PBR vertex format with tangent space support.
///
/// This is the primary vertex format for PBR mesh rendering:
/// - `position`: World-space vertex position (vec3<f32>, 12 bytes)
/// - `normal`: Surface normal for lighting (vec3<f32>, 12 bytes)
/// - `uv`: Texture coordinates (vec2<f32>, 8 bytes)
/// - `tangent`: Tangent with handedness in w (vec4<f32>, 16 bytes)
///
/// # Memory Layout
///
/// ```text
/// Offset  Size  Field
///   0      12   position [f32; 3]
///  12      12   normal   [f32; 3]
///  24       8   uv       [f32; 2]
///  32      16   tangent  [f32; 4]
/// ────────────────────────────────
/// Total:  48 bytes
/// ```
///
/// # Shader Locations
///
/// | Location | Attribute | Format |
/// |----------|-----------|--------|
/// | 0 | position | Float32x3 |
/// | 1 | normal | Float32x3 |
/// | 2 | uv | Float32x2 |
/// | 3 | tangent | Float32x4 |
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PbrVertex {
    /// World-space vertex position.
    pub position: [f32; 3],
    /// Surface normal (should be normalized).
    pub normal: [f32; 3],
    /// Texture coordinates (UV).
    pub uv: [f32; 2],
    /// Tangent vector with handedness in w component.
    /// The bitangent is computed as: bitangent = cross(normal, tangent.xyz) * tangent.w
    pub tangent: [f32; 4],
}

impl PbrVertex {
    /// Vertex attributes for PBR layout using `vertex_attr_array!` macro.
    pub const ATTRIBS: [VertexAttribute; 4] = vertex_attr_array![
        0 => Float32x3,  // position
        1 => Float32x3,  // normal
        2 => Float32x2,  // uv
        3 => Float32x4,  // tangent
    ];

    /// Vertex buffer layout for PBR vertices.
    pub const LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as u64,
        step_mode: VertexStepMode::Vertex,
        attributes: &Self::ATTRIBS,
    };

    /// Create a new PBR vertex.
    #[inline]
    pub const fn new(
        position: [f32; 3],
        normal: [f32; 3],
        uv: [f32; 2],
        tangent: [f32; 4],
    ) -> Self {
        Self { position, normal, uv, tangent }
    }

    /// Create a PBR vertex with default tangent (aligned to +X).
    #[inline]
    pub const fn with_default_tangent(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self {
            position,
            normal,
            uv,
            tangent: [1.0, 0.0, 0.0, 1.0],
        }
    }
}

// Compile-time size verification
const _: () = assert!(std::mem::size_of::<PbrVertex>() == 48);

// ============================================================================
// Skinned Vertex (72 bytes)
// ============================================================================

/// Skinned mesh vertex format for skeletal animation.
///
/// Extends the PBR format with bone indices and weights:
/// - `position`: World-space vertex position (vec3<f32>, 12 bytes)
/// - `normal`: Surface normal (vec3<f32>, 12 bytes)
/// - `uv`: Texture coordinates (vec2<f32>, 8 bytes)
/// - `tangent`: Tangent with handedness (vec4<f32>, 16 bytes)
/// - `joints`: Bone indices (vec4<u16>, 8 bytes)
/// - `weights`: Bone weights (vec4<f32>, 16 bytes)
///
/// # Memory Layout
///
/// ```text
/// Offset  Size  Field
///   0      12   position [f32; 3]
///  12      12   normal   [f32; 3]
///  24       8   uv       [f32; 2]
///  32      16   tangent  [f32; 4]
///  48       8   joints   [u16; 4]
///  56      16   weights  [f32; 4]
/// ────────────────────────────────
/// Total:  72 bytes
/// ```
///
/// # Shader Locations
///
/// | Location | Attribute | Format |
/// |----------|-----------|--------|
/// | 0 | position | Float32x3 |
/// | 1 | normal | Float32x3 |
/// | 2 | uv | Float32x2 |
/// | 3 | tangent | Float32x4 |
/// | 4 | joints | Uint16x4 |
/// | 5 | weights | Float32x4 |
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinnedVertex {
    /// World-space vertex position.
    pub position: [f32; 3],
    /// Surface normal (should be normalized).
    pub normal: [f32; 3],
    /// Texture coordinates (UV).
    pub uv: [f32; 2],
    /// Tangent vector with handedness in w component.
    pub tangent: [f32; 4],
    /// Bone indices (up to 4 bones per vertex).
    /// Values are indices into the skeleton's bone array.
    pub joints: [u16; 4],
    /// Bone weights (must sum to 1.0).
    pub weights: [f32; 4],
}

impl SkinnedVertex {
    /// Vertex attributes for skinned mesh layout.
    pub const ATTRIBS: [VertexAttribute; 6] = vertex_attr_array![
        0 => Float32x3,  // position
        1 => Float32x3,  // normal
        2 => Float32x2,  // uv
        3 => Float32x4,  // tangent
        4 => Uint16x4,   // joints
        5 => Float32x4,  // weights
    ];

    /// Vertex buffer layout for skinned vertices.
    pub const LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as u64,
        step_mode: VertexStepMode::Vertex,
        attributes: &Self::ATTRIBS,
    };

    /// Create a new skinned vertex.
    #[inline]
    pub const fn new(
        position: [f32; 3],
        normal: [f32; 3],
        uv: [f32; 2],
        tangent: [f32; 4],
        joints: [u16; 4],
        weights: [f32; 4],
    ) -> Self {
        Self { position, normal, uv, tangent, joints, weights }
    }

    /// Create a skinned vertex from a PBR vertex with default skinning.
    ///
    /// Sets joint 0 with weight 1.0, effectively binding to a single bone.
    #[inline]
    pub const fn from_pbr(pbr: PbrVertex, bone_index: u16) -> Self {
        Self {
            position: pbr.position,
            normal: pbr.normal,
            uv: pbr.uv,
            tangent: pbr.tangent,
            joints: [bone_index, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

// Compile-time size verification
const _: () = assert!(std::mem::size_of::<SkinnedVertex>() == 72);

// ============================================================================
// Terrain Vertex (32 bytes)
// ============================================================================

/// Terrain vertex format without tangent space.
///
/// Simplified layout for terrain rendering:
/// - `position`: World-space vertex position (vec3<f32>, 12 bytes)
/// - `normal`: Surface normal (vec3<f32>, 12 bytes)
/// - `uv`: Texture coordinates (vec2<f32>, 8 bytes)
///
/// # Memory Layout
///
/// ```text
/// Offset  Size  Field
///   0      12   position [f32; 3]
///  12      12   normal   [f32; 3]
///  24       8   uv       [f32; 2]
/// ────────────────────────────────
/// Total:  32 bytes
/// ```
///
/// # Shader Locations
///
/// | Location | Attribute | Format |
/// |----------|-----------|--------|
/// | 0 | position | Float32x3 |
/// | 1 | normal | Float32x3 |
/// | 2 | uv | Float32x2 |
///
/// # Notes
///
/// Terrain tangent space can be computed from heightmap gradients in the
/// vertex shader, so tangent storage is omitted for bandwidth savings.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainVertex {
    /// World-space vertex position.
    pub position: [f32; 3],
    /// Surface normal (should be normalized).
    pub normal: [f32; 3],
    /// Texture coordinates (UV).
    pub uv: [f32; 2],
}

impl TerrainVertex {
    /// Vertex attributes for terrain layout.
    pub const ATTRIBS: [VertexAttribute; 3] = vertex_attr_array![
        0 => Float32x3,  // position
        1 => Float32x3,  // normal
        2 => Float32x2,  // uv
    ];

    /// Vertex buffer layout for terrain vertices.
    pub const LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as u64,
        step_mode: VertexStepMode::Vertex,
        attributes: &Self::ATTRIBS,
    };

    /// Create a new terrain vertex.
    #[inline]
    pub const fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self { position, normal, uv }
    }

    /// Create a terrain vertex with up-facing normal.
    #[inline]
    pub const fn flat(position: [f32; 3], uv: [f32; 2]) -> Self {
        Self {
            position,
            normal: [0.0, 1.0, 0.0],
            uv,
        }
    }
}

// Compile-time size verification
const _: () = assert!(std::mem::size_of::<TerrainVertex>() == 32);

// ============================================================================
// Particle Vertex (32 bytes)
// ============================================================================

/// Particle vertex format for GPU particle systems.
///
/// Per-particle instance data:
/// - `position`: World-space particle position (vec3<f32>, 12 bytes)
/// - `color`: Packed RGBA color (u32, 4 bytes)
/// - `size`: Particle size/scale (f32, 4 bytes)
/// - `life`: Normalized lifetime 0-1 (f32, 4 bytes)
/// - `rotation`: Rotation angle in radians (f32, 4 bytes)
/// - `_padding`: Alignment padding (u32, 4 bytes)
///
/// # Memory Layout
///
/// ```text
/// Offset  Size  Field
///   0      12   position [f32; 3]
///  12       4   color    u32 (packed RGBA)
///  16       4   size     f32
///  20       4   life     f32
///  24       4   rotation f32
///  28       4   _padding u32
/// ────────────────────────────────
/// Total:  32 bytes
/// ```
///
/// # Shader Locations
///
/// | Location | Attribute | Format |
/// |----------|-----------|--------|
/// | 0 | position | Float32x3 |
/// | 1 | color | Uint32 |
/// | 2 | size | Float32 |
/// | 3 | life | Float32 |
/// | 4 | rotation | Float32 |
///
/// # Color Packing
///
/// Color is packed as RGBA in little-endian byte order:
/// `color = r | (g << 8) | (b << 16) | (a << 24)`
///
/// Unpack in shader: `vec4<f32>(unpack4x8unorm(color))`
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleVertex {
    /// World-space particle position.
    pub position: [f32; 3],
    /// Packed RGBA color (use `pack_color` to create).
    pub color: u32,
    /// Particle size/scale.
    pub size: f32,
    /// Normalized lifetime (0.0 = just spawned, 1.0 = about to die).
    pub life: f32,
    /// Rotation angle in radians.
    pub rotation: f32,
    /// Padding for alignment (unused).
    pub _padding: u32,
}

impl ParticleVertex {
    /// Vertex attributes for particle layout.
    pub const ATTRIBS: [VertexAttribute; 5] = vertex_attr_array![
        0 => Float32x3,  // position
        1 => Uint32,     // color (packed RGBA)
        2 => Float32,    // size
        3 => Float32,    // life
        4 => Float32,    // rotation
    ];

    /// Vertex buffer layout for particle vertices (per-instance).
    pub const LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as u64,
        step_mode: VertexStepMode::Instance,
        attributes: &Self::ATTRIBS,
    };

    /// Vertex buffer layout for particle vertices (per-vertex).
    pub const LAYOUT_VERTEX: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as u64,
        step_mode: VertexStepMode::Vertex,
        attributes: &Self::ATTRIBS,
    };

    /// Create a new particle vertex.
    #[inline]
    pub const fn new(position: [f32; 3], color: u32, size: f32, life: f32, rotation: f32) -> Self {
        Self { position, color, size, life, rotation, _padding: 0 }
    }

    /// Pack RGBA color components (0-255) into a u32.
    #[inline]
    pub const fn pack_color(r: u8, g: u8, b: u8, a: u8) -> u32 {
        (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | ((a as u32) << 24)
    }

    /// Pack RGBA color from normalized floats (0.0-1.0).
    #[inline]
    pub fn pack_color_f32(r: f32, g: f32, b: f32, a: f32) -> u32 {
        let r = (r.clamp(0.0, 1.0) * 255.0) as u8;
        let g = (g.clamp(0.0, 1.0) * 255.0) as u8;
        let b = (b.clamp(0.0, 1.0) * 255.0) as u8;
        let a = (a.clamp(0.0, 1.0) * 255.0) as u8;
        Self::pack_color(r, g, b, a)
    }

    /// Create a white particle at origin.
    #[inline]
    pub const fn default_white() -> Self {
        Self::new([0.0, 0.0, 0.0], 0xFFFFFFFF, 1.0, 0.0, 0.0)
    }
}

// Compile-time size verification
const _: () = assert!(std::mem::size_of::<ParticleVertex>() == 32);

// ============================================================================
// UI Vertex (20 bytes)
// ============================================================================

/// UI vertex format for 2D rendering.
///
/// Minimal layout for UI elements:
/// - `position`: Screen-space position (vec2<f32>, 8 bytes)
/// - `uv`: Texture coordinates (vec2<f32>, 8 bytes)
/// - `color`: Packed RGBA color (u32, 4 bytes)
///
/// # Memory Layout
///
/// ```text
/// Offset  Size  Field
///   0       8   position [f32; 2]
///   8       8   uv       [f32; 2]
///  16       4   color    u32 (packed RGBA)
/// ────────────────────────────────
/// Total:  20 bytes
/// ```
///
/// # Shader Locations
///
/// | Location | Attribute | Format |
/// |----------|-----------|--------|
/// | 0 | position | Float32x2 |
/// | 1 | uv | Float32x2 |
/// | 2 | color | Uint32 |
///
/// # Color Packing
///
/// Same as [`ParticleVertex`]: RGBA in little-endian byte order.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiVertex {
    /// Screen-space position (in pixels or normalized coordinates).
    pub position: [f32; 2],
    /// Texture coordinates.
    pub uv: [f32; 2],
    /// Packed RGBA color.
    pub color: u32,
}

impl UiVertex {
    /// Vertex attributes for UI layout.
    pub const ATTRIBS: [VertexAttribute; 3] = vertex_attr_array![
        0 => Float32x2,  // position
        1 => Float32x2,  // uv
        2 => Uint32,     // color (packed RGBA)
    ];

    /// Vertex buffer layout for UI vertices.
    pub const LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: std::mem::size_of::<Self>() as u64,
        step_mode: VertexStepMode::Vertex,
        attributes: &Self::ATTRIBS,
    };

    /// Create a new UI vertex.
    #[inline]
    pub const fn new(position: [f32; 2], uv: [f32; 2], color: u32) -> Self {
        Self { position, uv, color }
    }

    /// Pack RGBA color components (0-255) into a u32.
    #[inline]
    pub const fn pack_color(r: u8, g: u8, b: u8, a: u8) -> u32 {
        ParticleVertex::pack_color(r, g, b, a)
    }

    /// Create a white UI vertex.
    #[inline]
    pub const fn white(position: [f32; 2], uv: [f32; 2]) -> Self {
        Self::new(position, uv, 0xFFFFFFFF)
    }

    /// Create vertices for a textured quad.
    ///
    /// Returns 4 vertices in counter-clockwise order:
    /// bottom-left, bottom-right, top-right, top-left
    #[inline]
    pub const fn quad(x: f32, y: f32, w: f32, h: f32, color: u32) -> [Self; 4] {
        [
            Self::new([x, y + h], [0.0, 1.0], color),      // bottom-left
            Self::new([x + w, y + h], [1.0, 1.0], color),  // bottom-right
            Self::new([x + w, y], [1.0, 0.0], color),      // top-right
            Self::new([x, y], [0.0, 0.0], color),          // top-left
        ]
    }
}

// Compile-time size verification
const _: () = assert!(std::mem::size_of::<UiVertex>() == 20);

// ============================================================================
// Vertex Format Registry
// ============================================================================

/// Registry for vertex buffer layouts.
///
/// Provides runtime lookup of vertex layouts by ID. Pre-populated with
/// standard layouts on creation; supports custom layout registration.
///
/// # Example
///
/// ```no_run
/// use renderer_backend::resources::vertex::{VertexFormatRegistry, VertexLayoutId};
/// use wgpu::{VertexBufferLayout, VertexStepMode, vertex_attr_array};
///
/// let mut registry = VertexFormatRegistry::new();
///
/// // Standard layouts are pre-registered
/// assert!(registry.get(VertexLayoutId::Pbr).is_some());
/// assert!(registry.get(VertexLayoutId::Skinned).is_some());
///
/// // Register a custom layout
/// let custom_layout = VertexBufferLayout {
///     array_stride: 24,
///     step_mode: VertexStepMode::Vertex,
///     attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32x3],
/// };
/// registry.register(VertexLayoutId::Custom(1), custom_layout);
/// ```
pub struct VertexFormatRegistry {
    layouts: HashMap<VertexLayoutId, VertexBufferLayout<'static>>,
}

impl VertexFormatRegistry {
    /// Create a new registry with standard layouts pre-registered.
    pub fn new() -> Self {
        let mut layouts = HashMap::with_capacity(8);

        // Register all standard layouts
        layouts.insert(VertexLayoutId::Pbr, PbrVertex::LAYOUT);
        layouts.insert(VertexLayoutId::Skinned, SkinnedVertex::LAYOUT);
        layouts.insert(VertexLayoutId::Terrain, TerrainVertex::LAYOUT);
        layouts.insert(VertexLayoutId::Particle, ParticleVertex::LAYOUT);
        layouts.insert(VertexLayoutId::Ui, UiVertex::LAYOUT);

        Self { layouts }
    }

    /// Get a vertex layout by ID.
    ///
    /// Returns `None` if the layout is not registered.
    #[inline]
    pub fn get(&self, id: VertexLayoutId) -> Option<&VertexBufferLayout<'static>> {
        self.layouts.get(&id)
    }

    /// Register a custom vertex layout.
    ///
    /// Overwrites any existing layout with the same ID.
    #[inline]
    pub fn register(&mut self, id: VertexLayoutId, layout: VertexBufferLayout<'static>) {
        self.layouts.insert(id, layout);
    }

    /// Remove a custom layout.
    ///
    /// Returns `true` if a layout was removed.
    #[inline]
    pub fn unregister(&mut self, id: VertexLayoutId) -> bool {
        self.layouts.remove(&id).is_some()
    }

    /// Check if a layout ID is registered.
    #[inline]
    pub fn contains(&self, id: VertexLayoutId) -> bool {
        self.layouts.contains_key(&id)
    }

    /// Get the number of registered layouts.
    #[inline]
    pub fn len(&self) -> usize {
        self.layouts.len()
    }

    /// Check if the registry is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layouts.is_empty()
    }

    /// Iterate over all registered layout IDs.
    pub fn layout_ids(&self) -> impl Iterator<Item = &VertexLayoutId> {
        self.layouts.keys()
    }

    /// Get the stride (size in bytes) for a layout.
    #[inline]
    pub fn stride(&self, id: VertexLayoutId) -> Option<u64> {
        self.layouts.get(&id).map(|l| l.array_stride)
    }
}

impl Default for VertexFormatRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────────────
    // Size verification tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_pbr_vertex_size() {
        assert_eq!(std::mem::size_of::<PbrVertex>(), 48);
    }

    #[test]
    fn test_skinned_vertex_size() {
        assert_eq!(std::mem::size_of::<SkinnedVertex>(), 72);
    }

    #[test]
    fn test_terrain_vertex_size() {
        assert_eq!(std::mem::size_of::<TerrainVertex>(), 32);
    }

    #[test]
    fn test_particle_vertex_size() {
        assert_eq!(std::mem::size_of::<ParticleVertex>(), 32);
    }

    #[test]
    fn test_ui_vertex_size() {
        assert_eq!(std::mem::size_of::<UiVertex>(), 20);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Registry tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_registry_lookup() {
        let registry = VertexFormatRegistry::new();

        // All standard layouts should be present
        assert!(registry.get(VertexLayoutId::Pbr).is_some());
        assert!(registry.get(VertexLayoutId::Skinned).is_some());
        assert!(registry.get(VertexLayoutId::Terrain).is_some());
        assert!(registry.get(VertexLayoutId::Particle).is_some());
        assert!(registry.get(VertexLayoutId::Ui).is_some());

        // Custom layout should not exist yet
        assert!(registry.get(VertexLayoutId::Custom(0)).is_none());
    }

    #[test]
    fn test_registry_stride() {
        let registry = VertexFormatRegistry::new();

        assert_eq!(registry.stride(VertexLayoutId::Pbr), Some(48));
        assert_eq!(registry.stride(VertexLayoutId::Skinned), Some(72));
        assert_eq!(registry.stride(VertexLayoutId::Terrain), Some(32));
        assert_eq!(registry.stride(VertexLayoutId::Particle), Some(32));
        assert_eq!(registry.stride(VertexLayoutId::Ui), Some(20));
    }

    #[test]
    fn test_registry_custom_layout() {
        // Define static attributes for the custom layout
        static CUSTOM_ATTRS: [VertexAttribute; 2] = vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3
        ];

        let mut registry = VertexFormatRegistry::new();

        let custom_layout = VertexBufferLayout {
            array_stride: 24,
            step_mode: VertexStepMode::Vertex,
            attributes: &CUSTOM_ATTRS,
        };

        registry.register(VertexLayoutId::Custom(42), custom_layout);

        assert!(registry.contains(VertexLayoutId::Custom(42)));
        assert_eq!(registry.stride(VertexLayoutId::Custom(42)), Some(24));

        // Unregister
        assert!(registry.unregister(VertexLayoutId::Custom(42)));
        assert!(!registry.contains(VertexLayoutId::Custom(42)));
    }

    #[test]
    fn test_registry_len() {
        let registry = VertexFormatRegistry::new();
        assert_eq!(registry.len(), 5);
        assert!(!registry.is_empty());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // vertex_attr_array! macro verification tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_vertex_attr_array_macro() {
        // Verify PBR attributes are correctly generated
        let attrs = PbrVertex::ATTRIBS;
        assert_eq!(attrs.len(), 4);

        // Check shader locations
        assert_eq!(attrs[0].shader_location, 0);
        assert_eq!(attrs[1].shader_location, 1);
        assert_eq!(attrs[2].shader_location, 2);
        assert_eq!(attrs[3].shader_location, 3);

        // Check formats
        assert_eq!(attrs[0].format, wgpu::VertexFormat::Float32x3);
        assert_eq!(attrs[1].format, wgpu::VertexFormat::Float32x3);
        assert_eq!(attrs[2].format, wgpu::VertexFormat::Float32x2);
        assert_eq!(attrs[3].format, wgpu::VertexFormat::Float32x4);
    }

    #[test]
    fn test_skinned_vertex_attr_array() {
        let attrs = SkinnedVertex::ATTRIBS;
        assert_eq!(attrs.len(), 6);

        // Check the additional skinning attributes
        assert_eq!(attrs[4].shader_location, 4);
        assert_eq!(attrs[4].format, wgpu::VertexFormat::Uint16x4);
        assert_eq!(attrs[5].shader_location, 5);
        assert_eq!(attrs[5].format, wgpu::VertexFormat::Float32x4);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Vertex construction tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_pbr_vertex_construction() {
        let v = PbrVertex::new(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0, 0.0, 1.0],
        );
        assert_eq!(v.position, [1.0, 2.0, 3.0]);
        assert_eq!(v.tangent[3], 1.0);
    }

    #[test]
    fn test_skinned_vertex_from_pbr() {
        let pbr = PbrVertex::new(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0, 0.0, 1.0],
        );
        let skinned = SkinnedVertex::from_pbr(pbr, 5);

        assert_eq!(skinned.position, pbr.position);
        assert_eq!(skinned.joints, [5, 0, 0, 0]);
        assert_eq!(skinned.weights, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_particle_color_packing() {
        let color = ParticleVertex::pack_color(255, 128, 64, 255);
        assert_eq!(color & 0xFF, 255);        // R
        assert_eq!((color >> 8) & 0xFF, 128); // G
        assert_eq!((color >> 16) & 0xFF, 64); // B
        assert_eq!((color >> 24) & 0xFF, 255); // A
    }

    #[test]
    fn test_ui_vertex_quad() {
        let quad = UiVertex::quad(10.0, 20.0, 100.0, 50.0, 0xFFFFFFFF);
        assert_eq!(quad.len(), 4);

        // Verify positions
        assert_eq!(quad[0].position, [10.0, 70.0]);   // bottom-left
        assert_eq!(quad[1].position, [110.0, 70.0]);  // bottom-right
        assert_eq!(quad[2].position, [110.0, 20.0]);  // top-right
        assert_eq!(quad[3].position, [10.0, 20.0]);   // top-left
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Layout compatibility tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_layout_step_modes() {
        // Particle uses instance step mode by default
        assert_eq!(ParticleVertex::LAYOUT.step_mode, VertexStepMode::Instance);

        // All others use vertex step mode
        assert_eq!(PbrVertex::LAYOUT.step_mode, VertexStepMode::Vertex);
        assert_eq!(SkinnedVertex::LAYOUT.step_mode, VertexStepMode::Vertex);
        assert_eq!(TerrainVertex::LAYOUT.step_mode, VertexStepMode::Vertex);
        assert_eq!(UiVertex::LAYOUT.step_mode, VertexStepMode::Vertex);
    }

    #[test]
    fn test_layout_array_strides() {
        assert_eq!(PbrVertex::LAYOUT.array_stride, 48);
        assert_eq!(SkinnedVertex::LAYOUT.array_stride, 72);
        assert_eq!(TerrainVertex::LAYOUT.array_stride, 32);
        assert_eq!(ParticleVertex::LAYOUT.array_stride, 32);
        assert_eq!(UiVertex::LAYOUT.array_stride, 20);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Bytemuck compatibility tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_bytemuck_cast() {
        let vertex = PbrVertex::new(
            [1.0, 2.0, 3.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.0, 0.0, 1.0],
        );

        // Verify we can cast to bytes
        let bytes: &[u8] = bytemuck::bytes_of(&vertex);
        assert_eq!(bytes.len(), 48);

        // Verify we can cast a slice
        let vertices = [vertex; 3];
        let bytes: &[u8] = bytemuck::cast_slice(&vertices);
        assert_eq!(bytes.len(), 48 * 3);
    }
}

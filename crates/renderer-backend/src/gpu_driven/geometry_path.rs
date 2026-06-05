//! Geometry Path Abstraction for GPU-driven rendering (T-WGPU-P6.9.3).
//!
//! This module provides an abstraction for choosing between traditional indexed
//! geometry rendering and meshlet-based rendering. The path selection is based
//! on runtime feature detection.
//!
//! # Overview
//!
//! Two rendering paths are supported:
//!
//! - **Traditional**: Standard indexed geometry rendering using vertex/index buffers.
//!   Always available on all hardware.
//! - **Meshlet**: GPU-driven rendering using mesh shaders and meshlet culling.
//!   Requires mesh shader support (not yet stable in wgpu).
//!
//! # Path Selection
//!
//! The optimal path is selected based on device capabilities:
//!
//! ```ignore
//! let features = device.features();
//! let path = GeometryPath::select(features);
//!
//! match path {
//!     GeometryPath::Traditional => render_traditional(mesh),
//!     GeometryPath::Meshlet => render_meshlets(meshlets),
//! }
//! ```
//!
//! # Configuration
//!
//! Use `GeometryPathConfig` to override automatic path selection:
//!
//! ```ignore
//! let config = GeometryPathConfig {
//!     preferred: GeometryPath::Meshlet,
//!     force_traditional: false,
//! };
//!
//! let path = config.resolve(features);
//! ```
//!
//! # Future Extensions
//!
//! When mesh shaders become stable in wgpu, the `meshlet_available` function
//! will check for `wgpu::Features::MESH_SHADER` (or equivalent).

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Human-readable name for traditional path.
pub const TRADITIONAL_PATH_NAME: &str = "Traditional";

/// Human-readable name for meshlet path.
pub const MESHLET_PATH_NAME: &str = "Meshlet";

// ---------------------------------------------------------------------------
// GeometryPath enum
// ---------------------------------------------------------------------------

/// Geometry rendering path selection.
///
/// Determines which rendering pipeline to use for geometry submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum GeometryPath {
    /// Traditional indexed geometry rendering.
    ///
    /// Uses standard vertex/index buffers with draw_indexed calls.
    /// This path is always available on all hardware.
    #[default]
    Traditional,

    /// Meshlet-based rendering using mesh shaders.
    ///
    /// Uses meshlet data structures with mesh shader dispatch.
    /// Requires mesh shader support (future wgpu feature).
    ///
    /// NOTE: Currently stubbed. Actual implementation requires:
    /// - wgpu mesh shader feature (when stable)
    /// - Mesh shader pipeline creation
    /// - Meshlet data upload and dispatch
    Meshlet,
}

impl GeometryPath {
    /// Select optimal path based on device features.
    ///
    /// Currently always returns `Traditional` since mesh shaders are not
    /// yet stable in wgpu. When mesh shader support is added, this will
    /// prefer `Meshlet` on capable hardware.
    ///
    /// # Arguments
    ///
    /// * `features` - Device features from `wgpu::Device::features()`
    ///
    /// # Returns
    ///
    /// The optimal geometry path for the device.
    #[inline]
    pub fn select(features: wgpu::Features) -> Self {
        if Self::meshlet_available(features) {
            Self::Meshlet
        } else {
            Self::Traditional
        }
    }

    /// Check if meshlet path is available on this device.
    ///
    /// Currently returns `false` since mesh shaders are not yet stable
    /// in wgpu. This function will be updated when the feature becomes
    /// available.
    ///
    /// # Arguments
    ///
    /// * `_features` - Device features (currently unused)
    ///
    /// # Returns
    ///
    /// `true` if meshlet rendering is supported, `false` otherwise.
    #[inline]
    pub fn meshlet_available(_features: wgpu::Features) -> bool {
        // Placeholder: mesh shader feature check
        // When wgpu adds mesh shader support, check:
        // features.contains(wgpu::Features::MESH_SHADER)
        false
    }

    /// Get human-readable name for this path.
    ///
    /// Useful for logging and debugging.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Traditional => TRADITIONAL_PATH_NAME,
            Self::Meshlet => MESHLET_PATH_NAME,
        }
    }

    /// Check if this is the traditional path.
    #[inline]
    pub const fn is_traditional(&self) -> bool {
        matches!(self, Self::Traditional)
    }

    /// Check if this is the meshlet path.
    #[inline]
    pub const fn is_meshlet(&self) -> bool {
        matches!(self, Self::Meshlet)
    }
}

// ---------------------------------------------------------------------------
// GeometryRenderable trait
// ---------------------------------------------------------------------------

/// Trait for geometry that can render via either path.
///
/// Implement this trait for mesh types that support both traditional
/// and meshlet rendering paths.
///
/// # Example
///
/// ```ignore
/// struct Mesh {
///     vertex_buffer: wgpu::Buffer,
///     index_buffer: wgpu::Buffer,
///     index_count: u32,
///     meshlets: Option<MeshletData>,
/// }
///
/// impl GeometryRenderable for Mesh {
///     fn render(&self, path: GeometryPath, render_pass: &mut wgpu::RenderPass) {
///         match path {
///             GeometryPath::Traditional => {
///                 render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
///                 render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
///                 render_pass.draw_indexed(0..self.index_count, 0, 0..1);
///             }
///             GeometryPath::Meshlet => {
///                 // Dispatch mesh shader with meshlet data
///                 unimplemented!("Mesh shader path not yet implemented");
///             }
///         }
///     }
///
///     fn supports_path(&self, path: GeometryPath) -> bool {
///         match path {
///             GeometryPath::Traditional => true,
///             GeometryPath::Meshlet => self.meshlets.is_some(),
///         }
///     }
/// }
/// ```
pub trait GeometryRenderable {
    /// Render using the specified path.
    ///
    /// # Arguments
    ///
    /// * `path` - The geometry rendering path to use
    /// * `render_pass` - The render pass to issue draw commands to
    ///
    /// # Panics
    ///
    /// May panic if `supports_path(path)` returns `false`.
    fn render(&self, path: GeometryPath, render_pass: &mut wgpu::RenderPass);

    /// Check if this geometry supports the given path.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to check support for
    ///
    /// # Returns
    ///
    /// `true` if this geometry can be rendered via the given path.
    fn supports_path(&self, path: GeometryPath) -> bool;

    /// Get the best available path for this geometry.
    ///
    /// Returns `Meshlet` if supported, otherwise `Traditional`.
    fn best_path(&self) -> GeometryPath {
        if self.supports_path(GeometryPath::Meshlet) {
            GeometryPath::Meshlet
        } else {
            GeometryPath::Traditional
        }
    }
}

// ---------------------------------------------------------------------------
// GeometryPathConfig
// ---------------------------------------------------------------------------

/// Configuration for geometry path selection.
///
/// Allows overriding automatic path selection based on application
/// requirements or user preferences.
#[derive(Clone, Debug)]
pub struct GeometryPathConfig {
    /// Preferred path (may fall back if unsupported).
    pub preferred: GeometryPath,

    /// Force traditional path even if meshlet is available.
    ///
    /// Useful for debugging, compatibility testing, or when traditional
    /// rendering performs better for specific content.
    pub force_traditional: bool,
}

impl Default for GeometryPathConfig {
    fn default() -> Self {
        Self {
            preferred: GeometryPath::Traditional,
            force_traditional: false,
        }
    }
}

impl GeometryPathConfig {
    /// Create config that prefers meshlet when available.
    pub fn prefer_meshlet() -> Self {
        Self {
            preferred: GeometryPath::Meshlet,
            force_traditional: false,
        }
    }

    /// Create config that forces traditional path.
    pub fn force_traditional() -> Self {
        Self {
            preferred: GeometryPath::Traditional,
            force_traditional: true,
        }
    }

    /// Resolve the actual path to use based on device features.
    ///
    /// Takes into account the preferred path, forced settings,
    /// and actual device capabilities.
    ///
    /// # Arguments
    ///
    /// * `features` - Device features from `wgpu::Device::features()`
    ///
    /// # Returns
    ///
    /// The resolved geometry path to use.
    pub fn resolve(&self, features: wgpu::Features) -> GeometryPath {
        // Force traditional if requested
        if self.force_traditional {
            return GeometryPath::Traditional;
        }

        // Check if preferred path is available
        match self.preferred {
            GeometryPath::Meshlet => {
                if GeometryPath::meshlet_available(features) {
                    GeometryPath::Meshlet
                } else {
                    GeometryPath::Traditional
                }
            }
            GeometryPath::Traditional => GeometryPath::Traditional,
        }
    }

    /// Check if the preferred path is available.
    ///
    /// # Arguments
    ///
    /// * `features` - Device features from `wgpu::Device::features()`
    ///
    /// # Returns
    ///
    /// `true` if the preferred path can be used without fallback.
    pub fn preferred_available(&self, features: wgpu::Features) -> bool {
        match self.preferred {
            GeometryPath::Traditional => true,
            GeometryPath::Meshlet => GeometryPath::meshlet_available(features),
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geometry_path_default() {
        let path = GeometryPath::default();
        assert_eq!(path, GeometryPath::Traditional);
        assert!(path.is_traditional());
        assert!(!path.is_meshlet());
    }

    #[test]
    fn test_geometry_path_names() {
        assert_eq!(GeometryPath::Traditional.name(), "Traditional");
        assert_eq!(GeometryPath::Meshlet.name(), "Meshlet");
    }

    #[test]
    fn test_geometry_path_select() {
        // With empty features, should select Traditional
        let features = wgpu::Features::empty();
        let path = GeometryPath::select(features);
        assert_eq!(path, GeometryPath::Traditional);
    }

    #[test]
    fn test_meshlet_not_available() {
        // Currently meshlet is never available
        let features = wgpu::Features::empty();
        assert!(!GeometryPath::meshlet_available(features));

        let features = wgpu::Features::all();
        assert!(!GeometryPath::meshlet_available(features));
    }

    #[test]
    fn test_geometry_path_config_default() {
        let config = GeometryPathConfig::default();
        assert_eq!(config.preferred, GeometryPath::Traditional);
        assert!(!config.force_traditional);
    }

    #[test]
    fn test_geometry_path_config_prefer_meshlet() {
        let config = GeometryPathConfig::prefer_meshlet();
        assert_eq!(config.preferred, GeometryPath::Meshlet);
        assert!(!config.force_traditional);
    }

    #[test]
    fn test_geometry_path_config_force_traditional() {
        let config = GeometryPathConfig::force_traditional();
        assert_eq!(config.preferred, GeometryPath::Traditional);
        assert!(config.force_traditional);
    }

    #[test]
    fn test_geometry_path_config_resolve_traditional() {
        let config = GeometryPathConfig::default();
        let features = wgpu::Features::empty();
        let path = config.resolve(features);
        assert_eq!(path, GeometryPath::Traditional);
    }

    #[test]
    fn test_geometry_path_config_resolve_forced() {
        let config = GeometryPathConfig {
            preferred: GeometryPath::Meshlet,
            force_traditional: true,
        };
        let features = wgpu::Features::all();
        let path = config.resolve(features);
        assert_eq!(path, GeometryPath::Traditional);
    }

    #[test]
    fn test_geometry_path_config_resolve_meshlet_fallback() {
        // Even if we prefer meshlet, it should fall back to traditional
        // since meshlet is not yet available
        let config = GeometryPathConfig::prefer_meshlet();
        let features = wgpu::Features::all();
        let path = config.resolve(features);
        assert_eq!(path, GeometryPath::Traditional);
    }

    #[test]
    fn test_geometry_path_config_preferred_available() {
        let config = GeometryPathConfig::default();
        let features = wgpu::Features::empty();

        // Traditional is always available
        assert!(config.preferred_available(features));

        // Meshlet is not available
        let meshlet_config = GeometryPathConfig::prefer_meshlet();
        assert!(!meshlet_config.preferred_available(features));
    }

    #[test]
    fn test_geometry_path_hash_eq() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(GeometryPath::Traditional);
        set.insert(GeometryPath::Meshlet);
        set.insert(GeometryPath::Traditional); // Duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&GeometryPath::Traditional));
        assert!(set.contains(&GeometryPath::Meshlet));
    }

    #[test]
    fn test_geometry_path_clone_copy() {
        let path = GeometryPath::Traditional;
        let cloned = path.clone();
        let copied = path;

        assert_eq!(path, cloned);
        assert_eq!(path, copied);
    }

    #[test]
    fn test_geometry_path_debug() {
        let path = GeometryPath::Traditional;
        let debug_str = format!("{:?}", path);
        assert!(debug_str.contains("Traditional"));

        let path = GeometryPath::Meshlet;
        let debug_str = format!("{:?}", path);
        assert!(debug_str.contains("Meshlet"));
    }
}

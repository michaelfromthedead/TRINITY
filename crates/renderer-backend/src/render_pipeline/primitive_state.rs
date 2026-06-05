//! Primitive state descriptor for render pipelines.
//!
//! # Primitive Topologies
//!
//! wgpu supports 5 primitive topologies:
//!
//! | Topology | Description | Use Cases |
//! |----------|-------------|-----------|
//! | **PointList** | Each vertex is a separate point | Particle systems, point clouds, debug visualization |
//! | **LineList** | Every 2 vertices form an independent line | Wireframe, debug lines, grid rendering |
//! | **LineStrip** | Connected line segments | Path visualization, curves, trail effects |
//! | **TriangleList** | Every 3 vertices form an independent triangle | General mesh rendering, indexed geometry (most common) |
//! | **TriangleStrip** | Each vertex after first two forms a new triangle | Terrain, optimized quad rendering, reduced vertex count |
//!
//! # Vertex Count Formulas
//!
//! - **PointList**: `primitives = vertices` (1 vertex per point)
//! - **LineList**: `primitives = vertices / 2` (2 vertices per line)
//! - **LineStrip**: `primitives = vertices - 1` (shared vertices)
//! - **TriangleList**: `primitives = vertices / 3` (3 vertices per triangle)
//! - **TriangleStrip**: `primitives = vertices - 2` (shared vertices)
//!
//! # Strip Topologies
//!
//! Strip topologies (`LineStrip`, `TriangleStrip`) require `strip_index_format` when using
//! indexed drawing with primitive restart. The index format determines the restart index:
//! - `Uint16`: restart at `0xFFFF`
//! - `Uint32`: restart at `0xFFFFFFFF`
//!
//! # Front Face Winding Order
//!
//! The front face winding order determines which triangle faces are considered "front-facing"
//! based on the order vertices appear on screen after projection:
//!
//! | Winding | Description | Convention |
//! |---------|-------------|------------|
//! | **Ccw** (Counter-Clockwise) | Vertices appearing CCW on screen are front-facing | OpenGL, Vulkan, wgpu default |
//! | **Cw** (Clockwise) | Vertices appearing CW on screen are front-facing | DirectX default |
//!
//! When importing assets, ensure the winding order matches:
//! - **OpenGL/Blender exports**: Usually CCW (use `FrontFace::Ccw`)
//! - **DirectX/3ds Max exports**: Usually CW (use `FrontFace::Cw`)
//! - **glTF**: Specifies CCW front faces
//!
//! # Face Culling
//!
//! Face culling skips rendering of faces based on their orientation:
//!
//! | Cull Mode | Description | Use Cases |
//! |-----------|-------------|-----------|
//! | **None** | Render all faces | Two-sided materials, foliage, transparency, debug |
//! | **Front** | Cull front faces | Interior rendering, shadow volumes, inside-out objects |
//! | **Back** | Cull back faces | Standard opaque rendering, performance optimization |
//!
//! Back-face culling is the most common optimization - it skips roughly 50% of triangles
//! for closed meshes since back faces are never visible from outside the object.

// ---------------------------------------------------------------------------
// PrimitiveStateDescriptor
// ---------------------------------------------------------------------------

/// Describes the primitive assembly and rasterization state.
///
/// # Defaults
///
/// - `topology`: `TriangleList`
/// - `strip_index_format`: `None`
/// - `front_face`: `Ccw` (counter-clockwise)
/// - `cull_mode`: `Some(Back)` (backface culling enabled)
/// - `unclipped_depth`: `false`
/// - `polygon_mode`: `Fill`
/// - `conservative`: `false`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrimitiveStateDescriptor {
    /// The primitive topology.
    pub topology: wgpu::PrimitiveTopology,
    /// Index format for strip topologies.
    pub strip_index_format: Option<wgpu::IndexFormat>,
    /// Which triangle winding is considered front-facing.
    pub front_face: wgpu::FrontFace,
    /// Face culling mode.
    pub cull_mode: Option<wgpu::Face>,
    /// Whether depth clipping is disabled.
    pub unclipped_depth: bool,
    /// Polygon fill mode.
    pub polygon_mode: wgpu::PolygonMode,
    /// Whether conservative rasterization is enabled.
    pub conservative: bool,
}

impl Default for PrimitiveStateDescriptor {
    fn default() -> Self {
        Self {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        }
    }
}

impl PrimitiveStateDescriptor {
    /// Create with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the primitive topology.
    pub fn topology(mut self, topology: wgpu::PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Set the strip index format (for strip topologies).
    pub fn strip_index_format(mut self, format: Option<wgpu::IndexFormat>) -> Self {
        self.strip_index_format = format;
        self
    }

    /// Set the front face winding order.
    pub fn front_face(mut self, front_face: wgpu::FrontFace) -> Self {
        self.front_face = front_face;
        self
    }

    /// Set the cull mode.
    pub fn cull_mode(mut self, cull_mode: Option<wgpu::Face>) -> Self {
        self.cull_mode = cull_mode;
        self
    }

    /// Disable face culling.
    pub fn no_culling(mut self) -> Self {
        self.cull_mode = None;
        self
    }

    /// Enable front-face culling.
    pub fn cull_front(mut self) -> Self {
        self.cull_mode = Some(wgpu::Face::Front);
        self
    }

    /// Enable back-face culling (default).
    pub fn cull_back(mut self) -> Self {
        self.cull_mode = Some(wgpu::Face::Back);
        self
    }

    /// Disable culling (alias for `no_culling`).
    pub fn cull_none(mut self) -> Self {
        self.cull_mode = None;
        self
    }

    /// Set front face to counter-clockwise (OpenGL/Vulkan/wgpu default).
    pub fn ccw(self) -> Self {
        self.front_face(wgpu::FrontFace::Ccw)
    }

    /// Set front face to clockwise (DirectX default).
    pub fn cw(self) -> Self {
        self.front_face(wgpu::FrontFace::Cw)
    }

    /// Set the front face winding order (alias for `front_face`).
    pub fn with_front_face(self, front_face: wgpu::FrontFace) -> Self {
        self.front_face(front_face)
    }

    /// Set the cull mode (alias for `cull_mode`).
    pub fn with_cull_mode(self, cull_mode: Option<wgpu::Face>) -> Self {
        self.cull_mode(cull_mode)
    }

    /// Set unclipped depth mode.
    pub fn unclipped_depth(mut self, enabled: bool) -> Self {
        self.unclipped_depth = enabled;
        self
    }

    /// Set the polygon fill mode.
    pub fn polygon_mode(mut self, mode: wgpu::PolygonMode) -> Self {
        self.polygon_mode = mode;
        self
    }

    /// Enable wireframe rendering.
    pub fn wireframe(mut self) -> Self {
        self.polygon_mode = wgpu::PolygonMode::Line;
        self
    }

    /// Enable point rendering.
    pub fn point(mut self) -> Self {
        self.polygon_mode = wgpu::PolygonMode::Point;
        self
    }

    /// Set polygon mode to Fill (default, always available).
    ///
    /// This is the standard rendering mode where polygons are filled.
    pub fn polygon_fill(self) -> Self {
        self.polygon_mode(wgpu::PolygonMode::Fill)
    }

    /// Set polygon mode to Line (wireframe).
    ///
    /// **Note:** Requires `wgpu::Features::POLYGON_MODE_LINE` feature.
    /// Check device capabilities before use.
    pub fn polygon_line(self) -> Self {
        self.polygon_mode(wgpu::PolygonMode::Line)
    }

    /// Set polygon mode to Point (vertex points only).
    ///
    /// **Note:** Requires `wgpu::Features::POLYGON_MODE_POINT` feature.
    /// Check device capabilities before use.
    pub fn polygon_point(self) -> Self {
        self.polygon_mode(wgpu::PolygonMode::Point)
    }

    /// Set conservative rasterization.
    pub fn conservative(mut self, enabled: bool) -> Self {
        self.conservative = enabled;
        self
    }

    /// Create a preset for triangle lists (most common).
    pub fn triangle_list() -> Self {
        Self::default()
    }

    /// Create a preset for triangle strips.
    pub fn triangle_strip(index_format: wgpu::IndexFormat) -> Self {
        Self {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: Some(index_format),
            ..Default::default()
        }
    }

    /// Create a preset for line lists.
    pub fn line_list() -> Self {
        Self {
            topology: wgpu::PrimitiveTopology::LineList,
            cull_mode: None,
            ..Default::default()
        }
    }

    /// Create a preset for point lists.
    ///
    /// Use cases: particle systems, point clouds, debug visualization.
    pub fn point_list() -> Self {
        Self {
            topology: wgpu::PrimitiveTopology::PointList,
            cull_mode: None,
            ..Default::default()
        }
    }

    /// Create a preset for line strips.
    ///
    /// Use cases: path visualization, curves, trail effects.
    pub fn line_strip(index_format: wgpu::IndexFormat) -> Self {
        Self {
            topology: wgpu::PrimitiveTopology::LineStrip,
            strip_index_format: Some(index_format),
            cull_mode: None,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Topology Helpers
// ---------------------------------------------------------------------------

/// Information about a primitive topology with usage documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopologyInfo {
    /// The wgpu primitive topology.
    pub topology: wgpu::PrimitiveTopology,
    /// Human-readable name.
    pub name: &'static str,
    /// Description of how vertices are assembled.
    pub description: &'static str,
    /// Recommended use cases.
    pub use_cases: &'static [&'static str],
}

/// All 5 primitive topologies with documentation.
pub const TOPOLOGIES: [TopologyInfo; 5] = [
    TopologyInfo {
        topology: wgpu::PrimitiveTopology::PointList,
        name: "PointList",
        description: "Individual points, each vertex is a separate point",
        use_cases: &["particle systems", "point clouds", "debug visualization"],
    },
    TopologyInfo {
        topology: wgpu::PrimitiveTopology::LineList,
        name: "LineList",
        description: "Independent line segments, every 2 vertices form a line",
        use_cases: &["wireframe", "debug lines", "grid rendering"],
    },
    TopologyInfo {
        topology: wgpu::PrimitiveTopology::LineStrip,
        name: "LineStrip",
        description: "Connected line segments, each vertex connects to the previous",
        use_cases: &["path visualization", "curves", "trail effects"],
    },
    TopologyInfo {
        topology: wgpu::PrimitiveTopology::TriangleList,
        name: "TriangleList",
        description: "Independent triangles, every 3 vertices form a triangle",
        use_cases: &["general mesh rendering", "indexed geometry", "most common topology"],
    },
    TopologyInfo {
        topology: wgpu::PrimitiveTopology::TriangleStrip,
        name: "TriangleStrip",
        description: "Connected triangles, each vertex after first two forms a new triangle",
        use_cases: &["terrain", "optimized quad rendering", "reduced vertex count"],
    },
];

/// Calculate the number of vertices needed for a given primitive count.
///
/// # Arguments
///
/// * `topology` - The primitive topology
/// * `primitive_count` - Number of primitives to render
///
/// # Returns
///
/// The minimum number of vertices required, or `None` if `primitive_count` is 0
/// for strip topologies (which require at least some primitives to be meaningful).
pub fn topology_vertex_count(topology: wgpu::PrimitiveTopology, primitive_count: u32) -> u32 {
    match topology {
        wgpu::PrimitiveTopology::PointList => primitive_count,
        wgpu::PrimitiveTopology::LineList => primitive_count * 2,
        wgpu::PrimitiveTopology::LineStrip => {
            if primitive_count == 0 { 0 } else { primitive_count + 1 }
        }
        wgpu::PrimitiveTopology::TriangleList => primitive_count * 3,
        wgpu::PrimitiveTopology::TriangleStrip => {
            if primitive_count == 0 { 0 } else { primitive_count + 2 }
        }
    }
}

/// Calculate the number of primitives from a given vertex count.
///
/// # Arguments
///
/// * `topology` - The primitive topology
/// * `vertex_count` - Number of vertices
///
/// # Returns
///
/// The number of complete primitives that can be formed.
/// Returns 0 if there aren't enough vertices for any complete primitive.
pub fn topology_primitive_count(topology: wgpu::PrimitiveTopology, vertex_count: u32) -> u32 {
    match topology {
        wgpu::PrimitiveTopology::PointList => vertex_count,
        wgpu::PrimitiveTopology::LineList => vertex_count / 2,
        wgpu::PrimitiveTopology::LineStrip => vertex_count.saturating_sub(1),
        wgpu::PrimitiveTopology::TriangleList => vertex_count / 3,
        wgpu::PrimitiveTopology::TriangleStrip => vertex_count.saturating_sub(2),
    }
}

/// Check if a topology is a strip topology (shares vertices between primitives).
///
/// Strip topologies (`LineStrip`, `TriangleStrip`) may require `strip_index_format`
/// when using indexed drawing with primitive restart.
pub fn is_strip_topology(topology: wgpu::PrimitiveTopology) -> bool {
    matches!(
        topology,
        wgpu::PrimitiveTopology::LineStrip | wgpu::PrimitiveTopology::TriangleStrip
    )
}

/// Check if a topology is a list topology (independent primitives).
pub fn is_list_topology(topology: wgpu::PrimitiveTopology) -> bool {
    matches!(
        topology,
        wgpu::PrimitiveTopology::PointList
            | wgpu::PrimitiveTopology::LineList
            | wgpu::PrimitiveTopology::TriangleList
    )
}

/// Get the minimum vertex count required for at least one primitive.
pub fn minimum_vertex_count(topology: wgpu::PrimitiveTopology) -> u32 {
    match topology {
        wgpu::PrimitiveTopology::PointList => 1,
        wgpu::PrimitiveTopology::LineList | wgpu::PrimitiveTopology::LineStrip => 2,
        wgpu::PrimitiveTopology::TriangleList | wgpu::PrimitiveTopology::TriangleStrip => 3,
    }
}

/// Get topology info by topology type.
pub fn get_topology_info(topology: wgpu::PrimitiveTopology) -> &'static TopologyInfo {
    match topology {
        wgpu::PrimitiveTopology::PointList => &TOPOLOGIES[0],
        wgpu::PrimitiveTopology::LineList => &TOPOLOGIES[1],
        wgpu::PrimitiveTopology::LineStrip => &TOPOLOGIES[2],
        wgpu::PrimitiveTopology::TriangleList => &TOPOLOGIES[3],
        wgpu::PrimitiveTopology::TriangleStrip => &TOPOLOGIES[4],
    }
}

// ---------------------------------------------------------------------------
// Front Face Winding Order Helpers
// ---------------------------------------------------------------------------

/// Information about a front face winding order with documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrontFaceInfo {
    /// The wgpu front face winding order.
    pub front_face: wgpu::FrontFace,
    /// Human-readable name.
    pub name: &'static str,
    /// Description of the winding convention.
    pub description: &'static str,
}

/// All front face winding orders with documentation.
///
/// # Winding Order Conventions
///
/// - **Counter-Clockwise (CCW)**: OpenGL, Vulkan, wgpu, glTF default. Most common in modern APIs.
/// - **Clockwise (CW)**: DirectX default. Often used in legacy content or DirectX ports.
pub const FRONT_FACES: [FrontFaceInfo; 2] = [
    FrontFaceInfo {
        front_face: wgpu::FrontFace::Ccw,
        name: "Counter-Clockwise",
        description: "Vertices wound counter-clockwise are front-facing (OpenGL/Vulkan/wgpu default)",
    },
    FrontFaceInfo {
        front_face: wgpu::FrontFace::Cw,
        name: "Clockwise",
        description: "Vertices wound clockwise are front-facing (DirectX default)",
    },
];

/// Get front face info by front face type.
pub fn get_front_face_info(front_face: wgpu::FrontFace) -> &'static FrontFaceInfo {
    match front_face {
        wgpu::FrontFace::Ccw => &FRONT_FACES[0],
        wgpu::FrontFace::Cw => &FRONT_FACES[1],
    }
}

// ---------------------------------------------------------------------------
// Cull Mode Helpers
// ---------------------------------------------------------------------------

/// Information about a face culling mode with documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CullModeInfo {
    /// The wgpu cull mode (None for disabled culling).
    pub cull_mode: Option<wgpu::Face>,
    /// Human-readable name.
    pub name: &'static str,
    /// Description of what is culled.
    pub description: &'static str,
    /// Recommended use cases for this cull mode.
    pub use_cases: &'static [&'static str],
}

/// All cull modes with documentation.
///
/// # Performance Note
///
/// Back-face culling (`CullMode::Back`) is the most common setting and provides
/// significant performance benefits by skipping approximately 50% of triangles
/// for closed meshes.
pub const CULL_MODES: [CullModeInfo; 3] = [
    CullModeInfo {
        cull_mode: None,
        name: "None",
        description: "No culling, all faces rendered regardless of orientation",
        use_cases: &["two-sided materials", "foliage", "transparency", "debug visualization"],
    },
    CullModeInfo {
        cull_mode: Some(wgpu::Face::Front),
        name: "Front",
        description: "Front faces culled, only back faces rendered",
        use_cases: &["interior rendering", "shadow volumes", "inside-out objects"],
    },
    CullModeInfo {
        cull_mode: Some(wgpu::Face::Back),
        name: "Back",
        description: "Back faces culled, only front faces rendered (most common)",
        use_cases: &["standard mesh rendering", "performance optimization", "opaque geometry"],
    },
];

/// Get cull mode info by cull mode type.
pub fn get_cull_mode_info(cull_mode: Option<wgpu::Face>) -> &'static CullModeInfo {
    match cull_mode {
        None => &CULL_MODES[0],
        Some(wgpu::Face::Front) => &CULL_MODES[1],
        Some(wgpu::Face::Back) => &CULL_MODES[2],
    }
}

// ---------------------------------------------------------------------------
// Polygon Mode Helpers
// ---------------------------------------------------------------------------

/// Information about a polygon fill mode with documentation.
///
/// # Feature Requirements
///
/// The `Fill` mode is always available, but `Line` (wireframe) and `Point` modes
/// require the `wgpu::Features::NON_FILL_POLYGON_MODE` (also known as
/// `wgpu::Features::POLYGON_MODE_LINE` and `wgpu::Features::POLYGON_MODE_POINT`)
/// feature to be enabled on the device.
///
/// Not all hardware supports non-fill polygon modes. Always check device
/// capabilities before using wireframe or point rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolygonModeInfo {
    /// The wgpu polygon mode.
    pub mode: wgpu::PolygonMode,
    /// Human-readable name.
    pub name: &'static str,
    /// Description of the polygon mode.
    pub description: &'static str,
    /// Recommended use cases for this mode.
    pub use_cases: &'static [&'static str],
    /// Whether this mode requires the NON_FILL_POLYGON_MODE feature.
    pub requires_feature: bool,
}

/// All polygon modes with documentation.
///
/// # Feature Requirements
///
/// - **Fill**: Always available on all hardware.
/// - **Line** (wireframe): Requires `wgpu::Features::POLYGON_MODE_LINE`.
/// - **Point**: Requires `wgpu::Features::POLYGON_MODE_POINT`.
///
/// Before using non-fill modes, verify the feature is available:
///
/// ```ignore
/// // Check if wireframe is supported
/// if adapter.features().contains(wgpu::Features::POLYGON_MODE_LINE) {
///     // Safe to use PolygonMode::Line
/// }
/// ```
pub const POLYGON_MODES: [PolygonModeInfo; 3] = [
    PolygonModeInfo {
        mode: wgpu::PolygonMode::Fill,
        name: "Fill",
        description: "Filled polygons (default mode)",
        use_cases: &["standard rendering", "textured meshes", "most common mode"],
        requires_feature: false,
    },
    PolygonModeInfo {
        mode: wgpu::PolygonMode::Line,
        name: "Line",
        description: "Wireframe rendering (edges only)",
        use_cases: &["debug visualization", "mesh inspection", "technical rendering", "CAD-style display"],
        requires_feature: true, // Requires POLYGON_MODE_LINE feature
    },
    PolygonModeInfo {
        mode: wgpu::PolygonMode::Point,
        name: "Point",
        description: "Point rendering at vertices only",
        use_cases: &["vertex visualization", "point clouds", "debug markers", "sparse geometry"],
        requires_feature: true, // Requires POLYGON_MODE_POINT feature
    },
];

/// Get polygon mode info by polygon mode type.
pub fn get_polygon_mode_info(mode: wgpu::PolygonMode) -> &'static PolygonModeInfo {
    match mode {
        wgpu::PolygonMode::Fill => &POLYGON_MODES[0],
        wgpu::PolygonMode::Line => &POLYGON_MODES[1],
        wgpu::PolygonMode::Point => &POLYGON_MODES[2],
    }
}

/// Check if a polygon mode requires device features beyond the default.
///
/// Returns `true` for `Line` (wireframe) and `Point` modes, which require
/// `wgpu::Features::POLYGON_MODE_LINE` and `wgpu::Features::POLYGON_MODE_POINT`
/// respectively.
///
/// # Example
///
/// ```ignore
/// use wgpu::PolygonMode;
///
/// assert!(!requires_non_fill_feature(PolygonMode::Fill));
/// assert!(requires_non_fill_feature(PolygonMode::Line));
/// assert!(requires_non_fill_feature(PolygonMode::Point));
/// ```
pub fn requires_non_fill_feature(mode: wgpu::PolygonMode) -> bool {
    matches!(mode, wgpu::PolygonMode::Line | wgpu::PolygonMode::Point)
}

/// Get the wgpu feature flag required for a given polygon mode.
///
/// Returns `None` for `Fill` mode (always available), or the specific
/// feature flag needed for `Line` or `Point` modes.
///
/// # Example
///
/// ```ignore
/// let mode = wgpu::PolygonMode::Line;
/// if let Some(feature) = required_feature_for_polygon_mode(mode) {
///     if !device.features().contains(feature) {
///         eprintln!("Warning: wireframe mode not supported");
///     }
/// }
/// ```
pub fn required_feature_for_polygon_mode(mode: wgpu::PolygonMode) -> Option<wgpu::Features> {
    match mode {
        wgpu::PolygonMode::Fill => None,
        wgpu::PolygonMode::Line => Some(wgpu::Features::POLYGON_MODE_LINE),
        wgpu::PolygonMode::Point => Some(wgpu::Features::POLYGON_MODE_POINT),
    }
}

impl From<PrimitiveStateDescriptor> for wgpu::PrimitiveState {
    fn from(desc: PrimitiveStateDescriptor) -> Self {
        wgpu::PrimitiveState {
            topology: desc.topology,
            strip_index_format: desc.strip_index_format,
            front_face: desc.front_face,
            cull_mode: desc.cull_mode,
            unclipped_depth: desc.unclipped_depth,
            polygon_mode: desc.polygon_mode,
            conservative: desc.conservative,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_state_defaults() {
        let state = PrimitiveStateDescriptor::default();
        assert_eq!(state.topology, wgpu::PrimitiveTopology::TriangleList);
        assert_eq!(state.front_face, wgpu::FrontFace::Ccw);
        assert_eq!(state.cull_mode, Some(wgpu::Face::Back));
        assert!(!state.unclipped_depth);
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Fill);
        assert!(!state.conservative);
    }

    #[test]
    fn test_primitive_state_wireframe() {
        let state = PrimitiveStateDescriptor::new().wireframe();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Line);
    }

    #[test]
    fn test_primitive_state_no_culling() {
        let state = PrimitiveStateDescriptor::new().no_culling();
        assert_eq!(state.cull_mode, None);
    }

    #[test]
    fn test_primitive_state_presets() {
        let tri_strip = PrimitiveStateDescriptor::triangle_strip(wgpu::IndexFormat::Uint16);
        assert_eq!(tri_strip.topology, wgpu::PrimitiveTopology::TriangleStrip);
        assert_eq!(tri_strip.strip_index_format, Some(wgpu::IndexFormat::Uint16));

        let line_list = PrimitiveStateDescriptor::line_list();
        assert_eq!(line_list.topology, wgpu::PrimitiveTopology::LineList);
        assert_eq!(line_list.cull_mode, None);

        let point_list = PrimitiveStateDescriptor::point_list();
        assert_eq!(point_list.topology, wgpu::PrimitiveTopology::PointList);
    }

    #[test]
    fn test_primitive_state_into_wgpu() {
        let state = PrimitiveStateDescriptor::new().wireframe().no_culling();
        let wgpu_state: wgpu::PrimitiveState = state.into();
        assert_eq!(wgpu_state.polygon_mode, wgpu::PolygonMode::Line);
        assert_eq!(wgpu_state.cull_mode, None);
    }

    // -------------------------------------------------------------------------
    // Additional Whitebox Tests - All 5 Topologies
    // -------------------------------------------------------------------------

    #[test]
    fn test_primitive_all_topologies() {
        // Test all 5 primitive topologies
        let point = PrimitiveStateDescriptor::new().topology(wgpu::PrimitiveTopology::PointList);
        assert_eq!(point.topology, wgpu::PrimitiveTopology::PointList);

        let line_list = PrimitiveStateDescriptor::new().topology(wgpu::PrimitiveTopology::LineList);
        assert_eq!(line_list.topology, wgpu::PrimitiveTopology::LineList);

        let line_strip = PrimitiveStateDescriptor::new().topology(wgpu::PrimitiveTopology::LineStrip);
        assert_eq!(line_strip.topology, wgpu::PrimitiveTopology::LineStrip);

        let tri_list = PrimitiveStateDescriptor::new().topology(wgpu::PrimitiveTopology::TriangleList);
        assert_eq!(tri_list.topology, wgpu::PrimitiveTopology::TriangleList);

        let tri_strip = PrimitiveStateDescriptor::new().topology(wgpu::PrimitiveTopology::TriangleStrip);
        assert_eq!(tri_strip.topology, wgpu::PrimitiveTopology::TriangleStrip);
    }

    #[test]
    fn test_strip_index_format_uint16() {
        let state = PrimitiveStateDescriptor::triangle_strip(wgpu::IndexFormat::Uint16);
        assert_eq!(state.strip_index_format, Some(wgpu::IndexFormat::Uint16));

        let wgpu_state: wgpu::PrimitiveState = state.into();
        assert_eq!(wgpu_state.strip_index_format, Some(wgpu::IndexFormat::Uint16));
    }

    #[test]
    fn test_strip_index_format_uint32() {
        let state = PrimitiveStateDescriptor::triangle_strip(wgpu::IndexFormat::Uint32);
        assert_eq!(state.strip_index_format, Some(wgpu::IndexFormat::Uint32));

        let wgpu_state: wgpu::PrimitiveState = state.into();
        assert_eq!(wgpu_state.strip_index_format, Some(wgpu::IndexFormat::Uint32));
    }

    #[test]
    fn test_line_strip_with_index_format() {
        // Line strips also need strip_index_format
        let state = PrimitiveStateDescriptor::new()
            .topology(wgpu::PrimitiveTopology::LineStrip)
            .strip_index_format(Some(wgpu::IndexFormat::Uint16));
        assert_eq!(state.topology, wgpu::PrimitiveTopology::LineStrip);
        assert_eq!(state.strip_index_format, Some(wgpu::IndexFormat::Uint16));
    }

    #[test]
    fn test_all_polygon_modes() {
        // Test all polygon modes: Fill, Line, Point
        let fill = PrimitiveStateDescriptor::new().polygon_mode(wgpu::PolygonMode::Fill);
        assert_eq!(fill.polygon_mode, wgpu::PolygonMode::Fill);

        let line = PrimitiveStateDescriptor::new().polygon_mode(wgpu::PolygonMode::Line);
        assert_eq!(line.polygon_mode, wgpu::PolygonMode::Line);

        let point = PrimitiveStateDescriptor::new().polygon_mode(wgpu::PolygonMode::Point);
        assert_eq!(point.polygon_mode, wgpu::PolygonMode::Point);
    }

    #[test]
    fn test_point_method() {
        let state = PrimitiveStateDescriptor::new().point();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Point);
    }

    // -------------------------------------------------------------------------
    // Polygon Mode Helper Tests (T-WGPU-P3.3.3)
    // -------------------------------------------------------------------------

    #[test]
    fn test_polygon_fill_method() {
        let state = PrimitiveStateDescriptor::new().polygon_fill();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Fill);
    }

    #[test]
    fn test_polygon_line_method() {
        let state = PrimitiveStateDescriptor::new().polygon_line();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Line);
    }

    #[test]
    fn test_polygon_point_method() {
        let state = PrimitiveStateDescriptor::new().polygon_point();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Point);
    }

    #[test]
    fn test_wireframe_equals_polygon_line() {
        let wireframe_state = PrimitiveStateDescriptor::new().wireframe();
        let polygon_line_state = PrimitiveStateDescriptor::new().polygon_line();
        assert_eq!(wireframe_state.polygon_mode, polygon_line_state.polygon_mode);
        assert_eq!(wireframe_state.polygon_mode, wgpu::PolygonMode::Line);
    }

    #[test]
    fn test_point_equals_polygon_point() {
        let point_state = PrimitiveStateDescriptor::new().point();
        let polygon_point_state = PrimitiveStateDescriptor::new().polygon_point();
        assert_eq!(point_state.polygon_mode, polygon_point_state.polygon_mode);
        assert_eq!(point_state.polygon_mode, wgpu::PolygonMode::Point);
    }

    #[test]
    fn test_polygon_modes_constant() {
        assert_eq!(super::POLYGON_MODES.len(), 3);
        assert_eq!(super::POLYGON_MODES[0].mode, wgpu::PolygonMode::Fill);
        assert_eq!(super::POLYGON_MODES[1].mode, wgpu::PolygonMode::Line);
        assert_eq!(super::POLYGON_MODES[2].mode, wgpu::PolygonMode::Point);
    }

    #[test]
    fn test_polygon_modes_names() {
        assert_eq!(super::POLYGON_MODES[0].name, "Fill");
        assert_eq!(super::POLYGON_MODES[1].name, "Line");
        assert_eq!(super::POLYGON_MODES[2].name, "Point");
    }

    #[test]
    fn test_polygon_modes_feature_requirements() {
        // Fill does not require any feature
        assert!(!super::POLYGON_MODES[0].requires_feature);
        // Line requires POLYGON_MODE_LINE feature
        assert!(super::POLYGON_MODES[1].requires_feature);
        // Point requires POLYGON_MODE_POINT feature
        assert!(super::POLYGON_MODES[2].requires_feature);
    }

    #[test]
    fn test_requires_non_fill_feature() {
        assert!(!super::requires_non_fill_feature(wgpu::PolygonMode::Fill));
        assert!(super::requires_non_fill_feature(wgpu::PolygonMode::Line));
        assert!(super::requires_non_fill_feature(wgpu::PolygonMode::Point));
    }

    #[test]
    fn test_required_feature_for_polygon_mode() {
        // Fill requires no feature
        assert_eq!(super::required_feature_for_polygon_mode(wgpu::PolygonMode::Fill), None);
        // Line requires POLYGON_MODE_LINE
        assert_eq!(
            super::required_feature_for_polygon_mode(wgpu::PolygonMode::Line),
            Some(wgpu::Features::POLYGON_MODE_LINE)
        );
        // Point requires POLYGON_MODE_POINT
        assert_eq!(
            super::required_feature_for_polygon_mode(wgpu::PolygonMode::Point),
            Some(wgpu::Features::POLYGON_MODE_POINT)
        );
    }

    #[test]
    fn test_get_polygon_mode_info_fill() {
        let info = super::get_polygon_mode_info(wgpu::PolygonMode::Fill);
        assert_eq!(info.mode, wgpu::PolygonMode::Fill);
        assert_eq!(info.name, "Fill");
        assert!(!info.requires_feature);
        assert!(!info.use_cases.is_empty());
        assert!(info.description.contains("Fill") || info.description.contains("default"));
    }

    #[test]
    fn test_get_polygon_mode_info_line() {
        let info = super::get_polygon_mode_info(wgpu::PolygonMode::Line);
        assert_eq!(info.mode, wgpu::PolygonMode::Line);
        assert_eq!(info.name, "Line");
        assert!(info.requires_feature);
        assert!(!info.use_cases.is_empty());
        assert!(info.description.contains("Wireframe") || info.description.contains("edge"));
    }

    #[test]
    fn test_get_polygon_mode_info_point() {
        let info = super::get_polygon_mode_info(wgpu::PolygonMode::Point);
        assert_eq!(info.mode, wgpu::PolygonMode::Point);
        assert_eq!(info.name, "Point");
        assert!(info.requires_feature);
        assert!(!info.use_cases.is_empty());
        assert!(info.description.contains("Point") || info.description.contains("vert"));
    }

    #[test]
    fn test_polygon_mode_info_use_cases_non_empty() {
        for info in &super::POLYGON_MODES {
            assert!(
                !info.use_cases.is_empty(),
                "PolygonMode {} should have use cases",
                info.name
            );
            assert!(
                info.use_cases.len() >= 2,
                "PolygonMode {} should have at least 2 use cases",
                info.name
            );
        }
    }

    #[test]
    fn test_polygon_mode_info_descriptions_non_empty() {
        for info in &super::POLYGON_MODES {
            assert!(
                !info.description.is_empty(),
                "PolygonMode {} should have a description",
                info.name
            );
            assert!(
                info.description.len() >= 10,
                "PolygonMode {} description too short",
                info.name
            );
        }
    }

    #[test]
    fn test_polygon_mode_info_copy_clone() {
        let info = super::POLYGON_MODES[0];
        let info_copy = info;
        let info_clone = info.clone();
        assert_eq!(info, info_copy);
        assert_eq!(info, info_clone);
    }

    #[test]
    fn test_polygon_mode_info_partial_eq() {
        assert_eq!(super::POLYGON_MODES[0], super::POLYGON_MODES[0]);
        assert_ne!(super::POLYGON_MODES[0], super::POLYGON_MODES[1]);
        assert_ne!(super::POLYGON_MODES[1], super::POLYGON_MODES[2]);
    }

    #[test]
    fn test_polygon_mode_info_debug() {
        let info = super::POLYGON_MODES[0];
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("Fill"));
        assert!(debug_str.contains("PolygonModeInfo"));
    }

    #[test]
    fn test_polygon_mode_chaining() {
        // Test that polygon mode can be chained with other settings
        let state = PrimitiveStateDescriptor::new()
            .polygon_line()
            .no_culling()
            .front_face(wgpu::FrontFace::Cw);

        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Line);
        assert_eq!(state.cull_mode, None);
        assert_eq!(state.front_face, wgpu::FrontFace::Cw);
    }

    #[test]
    fn test_polygon_mode_override() {
        // Test that polygon mode can be overridden
        let state = PrimitiveStateDescriptor::new()
            .wireframe()        // Line
            .polygon_fill()     // Override to Fill
            .polygon_point();   // Override to Point

        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Point);
    }

    #[test]
    fn test_all_polygon_modes_into_wgpu() {
        // Verify all polygon modes convert correctly to wgpu::PrimitiveState
        for mode in [
            wgpu::PolygonMode::Fill,
            wgpu::PolygonMode::Line,
            wgpu::PolygonMode::Point,
        ] {
            let state = PrimitiveStateDescriptor::new().polygon_mode(mode);
            let wgpu_state: wgpu::PrimitiveState = state.into();
            assert_eq!(wgpu_state.polygon_mode, mode);
        }
    }

    #[test]
    fn test_conservative_rasterization() {
        let state = PrimitiveStateDescriptor::new().conservative(true);
        assert!(state.conservative);

        let state_disabled = PrimitiveStateDescriptor::new().conservative(false);
        assert!(!state_disabled.conservative);
    }

    #[test]
    fn test_unclipped_depth_enabled() {
        let state = PrimitiveStateDescriptor::new().unclipped_depth(true);
        assert!(state.unclipped_depth);

        let wgpu_state: wgpu::PrimitiveState = state.into();
        assert!(wgpu_state.unclipped_depth);
    }

    #[test]
    fn test_front_face_cw() {
        let state = PrimitiveStateDescriptor::new().front_face(wgpu::FrontFace::Cw);
        assert_eq!(state.front_face, wgpu::FrontFace::Cw);

        let wgpu_state: wgpu::PrimitiveState = state.into();
        assert_eq!(wgpu_state.front_face, wgpu::FrontFace::Cw);
    }

    #[test]
    fn test_front_face_ccw() {
        let state = PrimitiveStateDescriptor::new().front_face(wgpu::FrontFace::Ccw);
        assert_eq!(state.front_face, wgpu::FrontFace::Ccw);
    }

    #[test]
    fn test_cull_front() {
        let state = PrimitiveStateDescriptor::new().cull_front();
        assert_eq!(state.cull_mode, Some(wgpu::Face::Front));
    }

    #[test]
    fn test_cull_back() {
        let state = PrimitiveStateDescriptor::new().cull_back();
        assert_eq!(state.cull_mode, Some(wgpu::Face::Back));
    }

    #[test]
    fn test_cull_mode_none_explicit() {
        let state = PrimitiveStateDescriptor::new().cull_mode(None);
        assert_eq!(state.cull_mode, None);
    }

    #[test]
    fn test_primitive_state_copy() {
        // Test Copy trait
        let state = PrimitiveStateDescriptor::default();
        let state_copy = state;
        assert_eq!(state, state_copy);
    }

    #[test]
    fn test_primitive_state_clone() {
        // Test Clone trait
        let state = PrimitiveStateDescriptor::default();
        let state_clone = state.clone();
        assert_eq!(state, state_clone);
    }

    #[test]
    fn test_primitive_state_equality() {
        // Test PartialEq implementation
        let state1 = PrimitiveStateDescriptor::default();
        let state2 = PrimitiveStateDescriptor::default();
        let state3 = PrimitiveStateDescriptor::default().wireframe();

        assert_eq!(state1, state2);
        assert_ne!(state1, state3);
    }

    #[test]
    fn test_triangle_list_preset() {
        let state = PrimitiveStateDescriptor::triangle_list();
        assert_eq!(state.topology, wgpu::PrimitiveTopology::TriangleList);
        assert_eq!(state.cull_mode, Some(wgpu::Face::Back));
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Fill);
    }

    #[test]
    fn test_combined_settings() {
        // Test combining multiple settings
        let state = PrimitiveStateDescriptor::new()
            .topology(wgpu::PrimitiveTopology::TriangleList)
            .front_face(wgpu::FrontFace::Cw)
            .cull_mode(Some(wgpu::Face::Back))
            .polygon_mode(wgpu::PolygonMode::Fill)
            .conservative(true)
            .unclipped_depth(true);

        assert_eq!(state.topology, wgpu::PrimitiveTopology::TriangleList);
        assert_eq!(state.front_face, wgpu::FrontFace::Cw);
        assert_eq!(state.cull_mode, Some(wgpu::Face::Back));
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Fill);
        assert!(state.conservative);
        assert!(state.unclipped_depth);
    }

    #[test]
    fn test_into_wgpu_all_fields() {
        let state = PrimitiveStateDescriptor::new()
            .topology(wgpu::PrimitiveTopology::TriangleStrip)
            .strip_index_format(Some(wgpu::IndexFormat::Uint32))
            .front_face(wgpu::FrontFace::Cw)
            .cull_mode(None)
            .unclipped_depth(true)
            .polygon_mode(wgpu::PolygonMode::Line)
            .conservative(true);

        let wgpu_state: wgpu::PrimitiveState = state.into();

        assert_eq!(wgpu_state.topology, wgpu::PrimitiveTopology::TriangleStrip);
        assert_eq!(wgpu_state.strip_index_format, Some(wgpu::IndexFormat::Uint32));
        assert_eq!(wgpu_state.front_face, wgpu::FrontFace::Cw);
        assert_eq!(wgpu_state.cull_mode, None);
        assert!(wgpu_state.unclipped_depth);
        assert_eq!(wgpu_state.polygon_mode, wgpu::PolygonMode::Line);
        assert!(wgpu_state.conservative);
    }

    // -------------------------------------------------------------------------
    // Line Strip Preset Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_line_strip_preset() {
        let state = PrimitiveStateDescriptor::line_strip(wgpu::IndexFormat::Uint16);
        assert_eq!(state.topology, wgpu::PrimitiveTopology::LineStrip);
        assert_eq!(state.strip_index_format, Some(wgpu::IndexFormat::Uint16));
        assert_eq!(state.cull_mode, None);
    }

    #[test]
    fn test_line_strip_preset_uint32() {
        let state = PrimitiveStateDescriptor::line_strip(wgpu::IndexFormat::Uint32);
        assert_eq!(state.topology, wgpu::PrimitiveTopology::LineStrip);
        assert_eq!(state.strip_index_format, Some(wgpu::IndexFormat::Uint32));
    }

    // -------------------------------------------------------------------------
    // Topology Helper Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_topology_vertex_count_point_list() {
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::PointList, 0), 0);
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::PointList, 1), 1);
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::PointList, 100), 100);
    }

    #[test]
    fn test_topology_vertex_count_line_list() {
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::LineList, 0), 0);
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::LineList, 1), 2);
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::LineList, 5), 10);
    }

    #[test]
    fn test_topology_vertex_count_line_strip() {
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::LineStrip, 0), 0);
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::LineStrip, 1), 2);
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::LineStrip, 5), 6);
    }

    #[test]
    fn test_topology_vertex_count_triangle_list() {
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::TriangleList, 0), 0);
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::TriangleList, 1), 3);
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::TriangleList, 4), 12);
    }

    #[test]
    fn test_topology_vertex_count_triangle_strip() {
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::TriangleStrip, 0), 0);
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::TriangleStrip, 1), 3);
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::TriangleStrip, 4), 6);
    }

    #[test]
    fn test_topology_primitive_count_point_list() {
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::PointList, 0), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::PointList, 5), 5);
    }

    #[test]
    fn test_topology_primitive_count_line_list() {
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineList, 0), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineList, 1), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineList, 2), 1);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineList, 5), 2);
    }

    #[test]
    fn test_topology_primitive_count_line_strip() {
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineStrip, 0), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineStrip, 1), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineStrip, 2), 1);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineStrip, 5), 4);
    }

    #[test]
    fn test_topology_primitive_count_triangle_list() {
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleList, 0), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleList, 2), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleList, 3), 1);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleList, 7), 2);
    }

    #[test]
    fn test_topology_primitive_count_triangle_strip() {
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleStrip, 0), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleStrip, 2), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleStrip, 3), 1);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleStrip, 6), 4);
    }

    #[test]
    fn test_is_strip_topology() {
        assert!(!super::is_strip_topology(wgpu::PrimitiveTopology::PointList));
        assert!(!super::is_strip_topology(wgpu::PrimitiveTopology::LineList));
        assert!(super::is_strip_topology(wgpu::PrimitiveTopology::LineStrip));
        assert!(!super::is_strip_topology(wgpu::PrimitiveTopology::TriangleList));
        assert!(super::is_strip_topology(wgpu::PrimitiveTopology::TriangleStrip));
    }

    #[test]
    fn test_is_list_topology() {
        assert!(super::is_list_topology(wgpu::PrimitiveTopology::PointList));
        assert!(super::is_list_topology(wgpu::PrimitiveTopology::LineList));
        assert!(!super::is_list_topology(wgpu::PrimitiveTopology::LineStrip));
        assert!(super::is_list_topology(wgpu::PrimitiveTopology::TriangleList));
        assert!(!super::is_list_topology(wgpu::PrimitiveTopology::TriangleStrip));
    }

    #[test]
    fn test_minimum_vertex_count() {
        assert_eq!(super::minimum_vertex_count(wgpu::PrimitiveTopology::PointList), 1);
        assert_eq!(super::minimum_vertex_count(wgpu::PrimitiveTopology::LineList), 2);
        assert_eq!(super::minimum_vertex_count(wgpu::PrimitiveTopology::LineStrip), 2);
        assert_eq!(super::minimum_vertex_count(wgpu::PrimitiveTopology::TriangleList), 3);
        assert_eq!(super::minimum_vertex_count(wgpu::PrimitiveTopology::TriangleStrip), 3);
    }

    #[test]
    fn test_topologies_constant() {
        assert_eq!(super::TOPOLOGIES.len(), 5);
        assert_eq!(super::TOPOLOGIES[0].topology, wgpu::PrimitiveTopology::PointList);
        assert_eq!(super::TOPOLOGIES[1].topology, wgpu::PrimitiveTopology::LineList);
        assert_eq!(super::TOPOLOGIES[2].topology, wgpu::PrimitiveTopology::LineStrip);
        assert_eq!(super::TOPOLOGIES[3].topology, wgpu::PrimitiveTopology::TriangleList);
        assert_eq!(super::TOPOLOGIES[4].topology, wgpu::PrimitiveTopology::TriangleStrip);
    }

    #[test]
    fn test_get_topology_info() {
        let info = super::get_topology_info(wgpu::PrimitiveTopology::TriangleList);
        assert_eq!(info.name, "TriangleList");
        assert!(!info.use_cases.is_empty());
        assert!(info.description.contains("triangle"));
    }

    #[test]
    fn test_topology_info_all_have_use_cases() {
        for info in &super::TOPOLOGIES {
            assert!(!info.use_cases.is_empty(), "{} should have use cases", info.name);
            assert!(!info.description.is_empty(), "{} should have description", info.name);
        }
    }

    #[test]
    fn test_vertex_primitive_roundtrip() {
        // For list topologies, vertex -> primitive -> vertex should give same or less
        for topology in [
            wgpu::PrimitiveTopology::PointList,
            wgpu::PrimitiveTopology::LineList,
            wgpu::PrimitiveTopology::TriangleList,
        ] {
            let primitives = 10u32;
            let vertices = super::topology_vertex_count(topology, primitives);
            let back_primitives = super::topology_primitive_count(topology, vertices);
            assert_eq!(back_primitives, primitives, "Roundtrip failed for {:?}", topology);
        }
    }

    // =========================================================================
    // WHITEBOX - T-WGPU-P3.3.1 Primitive Topologies Additional Tests
    // =========================================================================

    // -------------------------------------------------------------------------
    // topology_vertex_count - Large counts and edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_topology_vertex_count_large_point_list() {
        // PointList: vertices = primitives
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::PointList, 10_000), 10_000);
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::PointList, 1_000_000), 1_000_000);
    }

    #[test]
    fn test_topology_vertex_count_large_line_list() {
        // LineList: vertices = primitives * 2
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::LineList, 10_000), 20_000);
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::LineList, 500_000), 1_000_000);
    }

    #[test]
    fn test_topology_vertex_count_large_line_strip() {
        // LineStrip: vertices = primitives + 1 (for non-zero)
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::LineStrip, 10_000), 10_001);
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::LineStrip, 999_999), 1_000_000);
    }

    #[test]
    fn test_topology_vertex_count_large_triangle_list() {
        // TriangleList: vertices = primitives * 3
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::TriangleList, 10_000), 30_000);
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::TriangleList, 333_333), 999_999);
    }

    #[test]
    fn test_topology_vertex_count_large_triangle_strip() {
        // TriangleStrip: vertices = primitives + 2 (for non-zero)
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::TriangleStrip, 10_000), 10_002);
        assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::TriangleStrip, 999_998), 1_000_000);
    }

    // -------------------------------------------------------------------------
    // topology_primitive_count - Large counts and edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_topology_primitive_count_large_point_list() {
        // PointList: primitives = vertices
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::PointList, 10_000), 10_000);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::PointList, 1_000_000), 1_000_000);
    }

    #[test]
    fn test_topology_primitive_count_large_line_list() {
        // LineList: primitives = vertices / 2
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineList, 20_000), 10_000);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineList, 1_000_000), 500_000);
        // Odd vertex count - truncates
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineList, 1_000_001), 500_000);
    }

    #[test]
    fn test_topology_primitive_count_large_line_strip() {
        // LineStrip: primitives = vertices - 1
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineStrip, 10_001), 10_000);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineStrip, 1_000_000), 999_999);
    }

    #[test]
    fn test_topology_primitive_count_large_triangle_list() {
        // TriangleList: primitives = vertices / 3
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleList, 30_000), 10_000);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleList, 999_999), 333_333);
        // Remainder cases
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleList, 1_000_000), 333_333);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleList, 1_000_001), 333_333);
    }

    #[test]
    fn test_topology_primitive_count_large_triangle_strip() {
        // TriangleStrip: primitives = vertices - 2
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleStrip, 10_002), 10_000);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleStrip, 1_000_000), 999_998);
    }

    // -------------------------------------------------------------------------
    // Minimum vertex count edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_minimum_vertex_count_insufficient_vertices() {
        // Test that primitive_count returns 0 when below minimum
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineList, 1), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineStrip, 1), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleList, 1), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleList, 2), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleStrip, 1), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleStrip, 2), 0);
    }

    #[test]
    fn test_minimum_vertex_count_exactly_minimum() {
        // Test exactly at minimum vertex count
        let min_point = super::minimum_vertex_count(wgpu::PrimitiveTopology::PointList);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::PointList, min_point), 1);

        let min_line_list = super::minimum_vertex_count(wgpu::PrimitiveTopology::LineList);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineList, min_line_list), 1);

        let min_line_strip = super::minimum_vertex_count(wgpu::PrimitiveTopology::LineStrip);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineStrip, min_line_strip), 1);

        let min_tri_list = super::minimum_vertex_count(wgpu::PrimitiveTopology::TriangleList);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleList, min_tri_list), 1);

        let min_tri_strip = super::minimum_vertex_count(wgpu::PrimitiveTopology::TriangleStrip);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleStrip, min_tri_strip), 1);
    }

    // -------------------------------------------------------------------------
    // Strip vs List classification exhaustive
    // -------------------------------------------------------------------------

    #[test]
    fn test_strip_and_list_mutually_exclusive() {
        // Every topology must be either strip OR list, not both
        for info in &super::TOPOLOGIES {
            let is_strip = super::is_strip_topology(info.topology);
            let is_list = super::is_list_topology(info.topology);
            assert!(
                is_strip != is_list,
                "{} must be either strip or list, not both or neither",
                info.name
            );
        }
    }

    #[test]
    fn test_strip_topology_count() {
        // Exactly 2 strip topologies
        let strip_count = super::TOPOLOGIES
            .iter()
            .filter(|t| super::is_strip_topology(t.topology))
            .count();
        assert_eq!(strip_count, 2, "Expected exactly 2 strip topologies");
    }

    #[test]
    fn test_list_topology_count() {
        // Exactly 3 list topologies
        let list_count = super::TOPOLOGIES
            .iter()
            .filter(|t| super::is_list_topology(t.topology))
            .count();
        assert_eq!(list_count, 3, "Expected exactly 3 list topologies");
    }

    // -------------------------------------------------------------------------
    // get_topology_info - All topologies
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_topology_info_point_list() {
        let info = super::get_topology_info(wgpu::PrimitiveTopology::PointList);
        assert_eq!(info.name, "PointList");
        assert_eq!(info.topology, wgpu::PrimitiveTopology::PointList);
        assert!(info.description.contains("point"));
        assert!(info.use_cases.contains(&"particle systems"));
    }

    #[test]
    fn test_get_topology_info_line_list() {
        let info = super::get_topology_info(wgpu::PrimitiveTopology::LineList);
        assert_eq!(info.name, "LineList");
        assert_eq!(info.topology, wgpu::PrimitiveTopology::LineList);
        assert!(info.description.contains("line"));
        assert!(info.use_cases.contains(&"wireframe"));
    }

    #[test]
    fn test_get_topology_info_line_strip() {
        let info = super::get_topology_info(wgpu::PrimitiveTopology::LineStrip);
        assert_eq!(info.name, "LineStrip");
        assert_eq!(info.topology, wgpu::PrimitiveTopology::LineStrip);
        assert!(info.description.contains("line") || info.description.contains("segment"));
        assert!(info.use_cases.contains(&"path visualization"));
    }

    #[test]
    fn test_get_topology_info_triangle_strip() {
        let info = super::get_topology_info(wgpu::PrimitiveTopology::TriangleStrip);
        assert_eq!(info.name, "TriangleStrip");
        assert_eq!(info.topology, wgpu::PrimitiveTopology::TriangleStrip);
        assert!(info.description.contains("triangle"));
        assert!(info.use_cases.contains(&"terrain"));
    }

    // -------------------------------------------------------------------------
    // TOPOLOGIES constant verification
    // -------------------------------------------------------------------------

    #[test]
    fn test_topologies_order() {
        // Verify TOPOLOGIES array order: Point, LineList, LineStrip, TriList, TriStrip
        assert_eq!(super::TOPOLOGIES[0].name, "PointList");
        assert_eq!(super::TOPOLOGIES[1].name, "LineList");
        assert_eq!(super::TOPOLOGIES[2].name, "LineStrip");
        assert_eq!(super::TOPOLOGIES[3].name, "TriangleList");
        assert_eq!(super::TOPOLOGIES[4].name, "TriangleStrip");
    }

    #[test]
    fn test_topologies_names_match_topology() {
        // Each name should match the topology variant name
        for info in &super::TOPOLOGIES {
            let expected_name = format!("{:?}", info.topology);
            assert_eq!(info.name, expected_name);
        }
    }

    #[test]
    fn test_topologies_descriptions_non_empty() {
        for info in &super::TOPOLOGIES {
            assert!(
                info.description.len() >= 10,
                "{} description too short: '{}'",
                info.name,
                info.description
            );
        }
    }

    #[test]
    fn test_topologies_use_cases_minimum() {
        // Each topology should have at least 2 use cases
        for info in &super::TOPOLOGIES {
            assert!(
                info.use_cases.len() >= 2,
                "{} should have at least 2 use cases, found {}",
                info.name,
                info.use_cases.len()
            );
        }
    }

    // -------------------------------------------------------------------------
    // TopologyInfo struct tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_topology_info_copy_clone() {
        let info = super::TOPOLOGIES[0];
        let info_copy = info;
        let info_clone = info.clone();
        assert_eq!(info, info_copy);
        assert_eq!(info, info_clone);
    }

    #[test]
    fn test_topology_info_partial_eq() {
        assert_eq!(super::TOPOLOGIES[0], super::TOPOLOGIES[0]);
        assert_ne!(super::TOPOLOGIES[0], super::TOPOLOGIES[1]);
    }

    #[test]
    fn test_topology_info_debug() {
        let info = super::TOPOLOGIES[0];
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("PointList"));
        assert!(debug_str.contains("TopologyInfo"));
    }

    // -------------------------------------------------------------------------
    // Roundtrip tests for strip topologies
    // -------------------------------------------------------------------------

    #[test]
    fn test_vertex_primitive_roundtrip_strips() {
        // For strip topologies, roundtrip works for non-zero primitive counts
        for topology in [
            wgpu::PrimitiveTopology::LineStrip,
            wgpu::PrimitiveTopology::TriangleStrip,
        ] {
            for primitives in [1, 5, 10, 100, 1000] {
                let vertices = super::topology_vertex_count(topology, primitives);
                let back_primitives = super::topology_primitive_count(topology, vertices);
                assert_eq!(
                    back_primitives, primitives,
                    "Roundtrip failed for {:?} with {} primitives",
                    topology, primitives
                );
            }
        }
    }

    #[test]
    fn test_vertex_primitive_roundtrip_all_topologies() {
        // Test roundtrip for all topologies with various counts
        let all_topologies = [
            wgpu::PrimitiveTopology::PointList,
            wgpu::PrimitiveTopology::LineList,
            wgpu::PrimitiveTopology::LineStrip,
            wgpu::PrimitiveTopology::TriangleList,
            wgpu::PrimitiveTopology::TriangleStrip,
        ];

        for topology in all_topologies {
            for primitives in [1, 2, 3, 10, 100] {
                let vertices = super::topology_vertex_count(topology, primitives);
                let back_primitives = super::topology_primitive_count(topology, vertices);
                assert_eq!(
                    back_primitives, primitives,
                    "Roundtrip failed for {:?} with {} primitives -> {} vertices -> {} primitives",
                    topology, primitives, vertices, back_primitives
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Zero primitive edge cases
    // -------------------------------------------------------------------------

    #[test]
    fn test_zero_primitives_all_topologies() {
        // All topologies should return 0 vertices for 0 primitives
        for topology in [
            wgpu::PrimitiveTopology::PointList,
            wgpu::PrimitiveTopology::LineList,
            wgpu::PrimitiveTopology::LineStrip,
            wgpu::PrimitiveTopology::TriangleList,
            wgpu::PrimitiveTopology::TriangleStrip,
        ] {
            assert_eq!(
                super::topology_vertex_count(topology, 0),
                0,
                "Zero primitives should give zero vertices for {:?}",
                topology
            );
        }
    }

    #[test]
    fn test_zero_vertices_all_topologies() {
        // All topologies should return 0 primitives for 0 vertices
        for topology in [
            wgpu::PrimitiveTopology::PointList,
            wgpu::PrimitiveTopology::LineList,
            wgpu::PrimitiveTopology::LineStrip,
            wgpu::PrimitiveTopology::TriangleList,
            wgpu::PrimitiveTopology::TriangleStrip,
        ] {
            assert_eq!(
                super::topology_primitive_count(topology, 0),
                0,
                "Zero vertices should give zero primitives for {:?}",
                topology
            );
        }
    }

    // -------------------------------------------------------------------------
    // Specific vertex count formulas verification
    // -------------------------------------------------------------------------

    #[test]
    fn test_line_list_formula() {
        // LineList: primitives = vertices / 2, vertices = primitives * 2
        for n in 1..20 {
            let vertices = n * 2;
            assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineList, vertices), n);
            assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::LineList, n), vertices);
        }
    }

    #[test]
    fn test_triangle_list_formula() {
        // TriangleList: primitives = vertices / 3, vertices = primitives * 3
        for n in 1..20 {
            let vertices = n * 3;
            assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleList, vertices), n);
            assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::TriangleList, n), vertices);
        }
    }

    #[test]
    fn test_line_strip_formula() {
        // LineStrip: primitives = vertices - 1, vertices = primitives + 1
        for n in 1..20 {
            let vertices = n + 1;
            assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineStrip, vertices), n);
            assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::LineStrip, n), vertices);
        }
    }

    #[test]
    fn test_triangle_strip_formula() {
        // TriangleStrip: primitives = vertices - 2, vertices = primitives + 2
        for n in 1..20 {
            let vertices = n + 2;
            assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleStrip, vertices), n);
            assert_eq!(super::topology_vertex_count(wgpu::PrimitiveTopology::TriangleStrip, n), vertices);
        }
    }

    // -------------------------------------------------------------------------
    // get_topology_info returns correct TOPOLOGIES index
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_topology_info_returns_correct_info() {
        // Verify get_topology_info returns the correct TopologyInfo for each topology
        for expected in &super::TOPOLOGIES {
            let info = super::get_topology_info(expected.topology);
            assert_eq!(
                info.topology, expected.topology,
                "get_topology_info should return matching topology"
            );
            assert_eq!(
                info.name, expected.name,
                "get_topology_info should return matching name"
            );
            assert_eq!(
                info.description, expected.description,
                "get_topology_info should return matching description"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Saturating subtraction for strip topologies
    // -------------------------------------------------------------------------

    #[test]
    fn test_strip_primitive_count_saturating() {
        // Verify saturating_sub behavior: 0 and 1 vertices should not underflow
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineStrip, 0), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::LineStrip, 1), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleStrip, 0), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleStrip, 1), 0);
        assert_eq!(super::topology_primitive_count(wgpu::PrimitiveTopology::TriangleStrip, 2), 0);
    }

    // =========================================================================
    // WHITEBOX - T-WGPU-P3.3.2 Culling and Front Face Tests
    // =========================================================================

    // -------------------------------------------------------------------------
    // FrontFaceInfo Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_front_faces_constant() {
        assert_eq!(super::FRONT_FACES.len(), 2);
        assert_eq!(super::FRONT_FACES[0].front_face, wgpu::FrontFace::Ccw);
        assert_eq!(super::FRONT_FACES[1].front_face, wgpu::FrontFace::Cw);
    }

    #[test]
    fn test_front_face_info_ccw() {
        let info = &super::FRONT_FACES[0];
        assert_eq!(info.front_face, wgpu::FrontFace::Ccw);
        assert_eq!(info.name, "Counter-Clockwise");
        assert!(info.description.contains("counter-clockwise"));
        assert!(info.description.contains("OpenGL") || info.description.contains("default"));
    }

    #[test]
    fn test_front_face_info_cw() {
        let info = &super::FRONT_FACES[1];
        assert_eq!(info.front_face, wgpu::FrontFace::Cw);
        assert_eq!(info.name, "Clockwise");
        assert!(info.description.contains("clockwise"));
        assert!(info.description.contains("DirectX"));
    }

    #[test]
    fn test_get_front_face_info_ccw() {
        let info = super::get_front_face_info(wgpu::FrontFace::Ccw);
        assert_eq!(info.front_face, wgpu::FrontFace::Ccw);
        assert_eq!(info.name, "Counter-Clockwise");
    }

    #[test]
    fn test_get_front_face_info_cw() {
        let info = super::get_front_face_info(wgpu::FrontFace::Cw);
        assert_eq!(info.front_face, wgpu::FrontFace::Cw);
        assert_eq!(info.name, "Clockwise");
    }

    #[test]
    fn test_front_face_info_copy_clone() {
        let info = super::FRONT_FACES[0];
        let info_copy = info;
        let info_clone = info.clone();
        assert_eq!(info, info_copy);
        assert_eq!(info, info_clone);
    }

    #[test]
    fn test_front_face_info_partial_eq() {
        assert_eq!(super::FRONT_FACES[0], super::FRONT_FACES[0]);
        assert_ne!(super::FRONT_FACES[0], super::FRONT_FACES[1]);
    }

    #[test]
    fn test_front_face_info_debug() {
        let info = super::FRONT_FACES[0];
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("FrontFaceInfo"));
        assert!(debug_str.contains("Ccw"));
    }

    // -------------------------------------------------------------------------
    // CullModeInfo Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cull_modes_constant() {
        assert_eq!(super::CULL_MODES.len(), 3);
        assert_eq!(super::CULL_MODES[0].cull_mode, None);
        assert_eq!(super::CULL_MODES[1].cull_mode, Some(wgpu::Face::Front));
        assert_eq!(super::CULL_MODES[2].cull_mode, Some(wgpu::Face::Back));
    }

    #[test]
    fn test_cull_mode_info_none() {
        let info = &super::CULL_MODES[0];
        assert_eq!(info.cull_mode, None);
        assert_eq!(info.name, "None");
        assert!(info.description.contains("No culling"));
        assert!(!info.use_cases.is_empty());
        assert!(info.use_cases.contains(&"two-sided materials"));
        assert!(info.use_cases.contains(&"foliage"));
    }

    #[test]
    fn test_cull_mode_info_front() {
        let info = &super::CULL_MODES[1];
        assert_eq!(info.cull_mode, Some(wgpu::Face::Front));
        assert_eq!(info.name, "Front");
        assert!(info.description.contains("Front faces culled"));
        assert!(!info.use_cases.is_empty());
        assert!(info.use_cases.contains(&"interior rendering"));
    }

    #[test]
    fn test_cull_mode_info_back() {
        let info = &super::CULL_MODES[2];
        assert_eq!(info.cull_mode, Some(wgpu::Face::Back));
        assert_eq!(info.name, "Back");
        assert!(info.description.contains("Back faces culled"));
        assert!(info.description.contains("most common"));
        assert!(!info.use_cases.is_empty());
        assert!(info.use_cases.contains(&"standard mesh rendering"));
        assert!(info.use_cases.contains(&"performance optimization"));
    }

    #[test]
    fn test_get_cull_mode_info_none() {
        let info = super::get_cull_mode_info(None);
        assert_eq!(info.cull_mode, None);
        assert_eq!(info.name, "None");
    }

    #[test]
    fn test_get_cull_mode_info_front() {
        let info = super::get_cull_mode_info(Some(wgpu::Face::Front));
        assert_eq!(info.cull_mode, Some(wgpu::Face::Front));
        assert_eq!(info.name, "Front");
    }

    #[test]
    fn test_get_cull_mode_info_back() {
        let info = super::get_cull_mode_info(Some(wgpu::Face::Back));
        assert_eq!(info.cull_mode, Some(wgpu::Face::Back));
        assert_eq!(info.name, "Back");
    }

    #[test]
    fn test_cull_mode_info_copy_clone() {
        let info = super::CULL_MODES[0];
        let info_copy = info;
        let info_clone = info.clone();
        assert_eq!(info, info_copy);
        assert_eq!(info, info_clone);
    }

    #[test]
    fn test_cull_mode_info_partial_eq() {
        assert_eq!(super::CULL_MODES[0], super::CULL_MODES[0]);
        assert_ne!(super::CULL_MODES[0], super::CULL_MODES[1]);
        assert_ne!(super::CULL_MODES[1], super::CULL_MODES[2]);
    }

    #[test]
    fn test_cull_mode_info_debug() {
        let info = super::CULL_MODES[0];
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("CullModeInfo"));
        assert!(debug_str.contains("None"));
    }

    #[test]
    fn test_cull_modes_all_have_use_cases() {
        for info in &super::CULL_MODES {
            assert!(
                !info.use_cases.is_empty(),
                "{} cull mode should have use cases",
                info.name
            );
            assert!(
                info.use_cases.len() >= 2,
                "{} cull mode should have at least 2 use cases",
                info.name
            );
        }
    }

    // -------------------------------------------------------------------------
    // Builder Method Tests: ccw(), cw(), cull_none()
    // -------------------------------------------------------------------------

    #[test]
    fn test_ccw_method() {
        let state = PrimitiveStateDescriptor::new().ccw();
        assert_eq!(state.front_face, wgpu::FrontFace::Ccw);
    }

    #[test]
    fn test_cw_method() {
        let state = PrimitiveStateDescriptor::new().cw();
        assert_eq!(state.front_face, wgpu::FrontFace::Cw);

        let wgpu_state: wgpu::PrimitiveState = state.into();
        assert_eq!(wgpu_state.front_face, wgpu::FrontFace::Cw);
    }

    #[test]
    fn test_cull_none_method() {
        let state = PrimitiveStateDescriptor::new().cull_none();
        assert_eq!(state.cull_mode, None);
    }

    #[test]
    fn test_with_front_face_method() {
        let state_ccw = PrimitiveStateDescriptor::new().with_front_face(wgpu::FrontFace::Ccw);
        assert_eq!(state_ccw.front_face, wgpu::FrontFace::Ccw);

        let state_cw = PrimitiveStateDescriptor::new().with_front_face(wgpu::FrontFace::Cw);
        assert_eq!(state_cw.front_face, wgpu::FrontFace::Cw);
    }

    #[test]
    fn test_with_cull_mode_method() {
        let state_none = PrimitiveStateDescriptor::new().with_cull_mode(None);
        assert_eq!(state_none.cull_mode, None);

        let state_front = PrimitiveStateDescriptor::new().with_cull_mode(Some(wgpu::Face::Front));
        assert_eq!(state_front.cull_mode, Some(wgpu::Face::Front));

        let state_back = PrimitiveStateDescriptor::new().with_cull_mode(Some(wgpu::Face::Back));
        assert_eq!(state_back.cull_mode, Some(wgpu::Face::Back));
    }

    // -------------------------------------------------------------------------
    // Combined Culling Configuration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cw_cull_front_combination() {
        // DirectX-style: CW front face, cull front faces
        let state = PrimitiveStateDescriptor::new().cw().cull_front();
        assert_eq!(state.front_face, wgpu::FrontFace::Cw);
        assert_eq!(state.cull_mode, Some(wgpu::Face::Front));

        let wgpu_state: wgpu::PrimitiveState = state.into();
        assert_eq!(wgpu_state.front_face, wgpu::FrontFace::Cw);
        assert_eq!(wgpu_state.cull_mode, Some(wgpu::Face::Front));
    }

    #[test]
    fn test_ccw_cull_back_combination() {
        // OpenGL-style: CCW front face, cull back faces (default)
        let state = PrimitiveStateDescriptor::new().ccw().cull_back();
        assert_eq!(state.front_face, wgpu::FrontFace::Ccw);
        assert_eq!(state.cull_mode, Some(wgpu::Face::Back));
    }

    #[test]
    fn test_two_sided_material_configuration() {
        // Two-sided material: no culling
        let state = PrimitiveStateDescriptor::new().cull_none();
        assert_eq!(state.cull_mode, None);

        // Verify both no_culling and cull_none produce same result
        let state2 = PrimitiveStateDescriptor::new().no_culling();
        assert_eq!(state.cull_mode, state2.cull_mode);
    }

    #[test]
    fn test_interior_rendering_configuration() {
        // Interior rendering: cull front faces to see inside
        let state = PrimitiveStateDescriptor::new().cull_front();
        assert_eq!(state.cull_mode, Some(wgpu::Face::Front));
    }

    #[test]
    fn test_chained_culling_overrides() {
        // Later calls should override earlier ones
        let state = PrimitiveStateDescriptor::new()
            .cull_back()
            .cull_front()
            .cull_none()
            .cull_back();
        assert_eq!(state.cull_mode, Some(wgpu::Face::Back));

        let state2 = PrimitiveStateDescriptor::new().cw().ccw().cw();
        assert_eq!(state2.front_face, wgpu::FrontFace::Cw);
    }

    // -------------------------------------------------------------------------
    // Consistency Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_front_face_consistency() {
        // Verify front_face() and with_front_face() produce identical results
        let state1 = PrimitiveStateDescriptor::new().front_face(wgpu::FrontFace::Cw);
        let state2 = PrimitiveStateDescriptor::new().with_front_face(wgpu::FrontFace::Cw);
        assert_eq!(state1.front_face, state2.front_face);
    }

    #[test]
    fn test_cull_mode_consistency() {
        // Verify cull_mode() and with_cull_mode() produce identical results
        let state1 = PrimitiveStateDescriptor::new().cull_mode(Some(wgpu::Face::Front));
        let state2 = PrimitiveStateDescriptor::new().with_cull_mode(Some(wgpu::Face::Front));
        assert_eq!(state1.cull_mode, state2.cull_mode);
    }

    #[test]
    fn test_convenience_method_consistency() {
        // ccw() should equal front_face(Ccw)
        let state1 = PrimitiveStateDescriptor::new().ccw();
        let state2 = PrimitiveStateDescriptor::new().front_face(wgpu::FrontFace::Ccw);
        assert_eq!(state1.front_face, state2.front_face);

        // cw() should equal front_face(Cw)
        let state3 = PrimitiveStateDescriptor::new().cw();
        let state4 = PrimitiveStateDescriptor::new().front_face(wgpu::FrontFace::Cw);
        assert_eq!(state3.front_face, state4.front_face);

        // cull_none() should equal cull_mode(None)
        let state5 = PrimitiveStateDescriptor::new().cull_none();
        let state6 = PrimitiveStateDescriptor::new().cull_mode(None);
        assert_eq!(state5.cull_mode, state6.cull_mode);
    }

    // -------------------------------------------------------------------------
    // wgpu Conversion Tests with Culling
    // -------------------------------------------------------------------------

    #[test]
    fn test_wgpu_conversion_with_cw_cull_none() {
        let state = PrimitiveStateDescriptor::new().cw().cull_none();
        let wgpu_state: wgpu::PrimitiveState = state.into();

        assert_eq!(wgpu_state.front_face, wgpu::FrontFace::Cw);
        assert_eq!(wgpu_state.cull_mode, None);
    }

    #[test]
    fn test_wgpu_conversion_all_cull_modes() {
        // None
        let state_none = PrimitiveStateDescriptor::new().cull_none();
        let wgpu_none: wgpu::PrimitiveState = state_none.into();
        assert_eq!(wgpu_none.cull_mode, None);

        // Front
        let state_front = PrimitiveStateDescriptor::new().cull_front();
        let wgpu_front: wgpu::PrimitiveState = state_front.into();
        assert_eq!(wgpu_front.cull_mode, Some(wgpu::Face::Front));

        // Back
        let state_back = PrimitiveStateDescriptor::new().cull_back();
        let wgpu_back: wgpu::PrimitiveState = state_back.into();
        assert_eq!(wgpu_back.cull_mode, Some(wgpu::Face::Back));
    }

    #[test]
    fn test_wgpu_conversion_all_front_faces() {
        // CCW
        let state_ccw = PrimitiveStateDescriptor::new().ccw();
        let wgpu_ccw: wgpu::PrimitiveState = state_ccw.into();
        assert_eq!(wgpu_ccw.front_face, wgpu::FrontFace::Ccw);

        // CW
        let state_cw = PrimitiveStateDescriptor::new().cw();
        let wgpu_cw: wgpu::PrimitiveState = state_cw.into();
        assert_eq!(wgpu_cw.front_face, wgpu::FrontFace::Cw);
    }

    // -------------------------------------------------------------------------
    // Additional WHITEBOX T-WGPU-P3.3.2 Tests - Extended Coverage
    // -------------------------------------------------------------------------

    #[test]
    fn test_default_front_face_is_ccw() {
        // Verify default matches wgpu convention (CCW)
        let state = PrimitiveStateDescriptor::default();
        assert_eq!(state.front_face, wgpu::FrontFace::Ccw);
    }

    #[test]
    fn test_default_cull_mode_is_back() {
        // Verify default has back-face culling enabled
        let state = PrimitiveStateDescriptor::default();
        assert_eq!(state.cull_mode, Some(wgpu::Face::Back));
    }

    #[test]
    fn test_front_faces_array_order() {
        // FRONT_FACES[0] should be CCW, FRONT_FACES[1] should be CW
        assert_eq!(super::FRONT_FACES[0].front_face, wgpu::FrontFace::Ccw);
        assert_eq!(super::FRONT_FACES[1].front_face, wgpu::FrontFace::Cw);
    }

    #[test]
    fn test_cull_modes_array_order() {
        // CULL_MODES[0] = None, [1] = Front, [2] = Back
        assert_eq!(super::CULL_MODES[0].cull_mode, None);
        assert_eq!(super::CULL_MODES[1].cull_mode, Some(wgpu::Face::Front));
        assert_eq!(super::CULL_MODES[2].cull_mode, Some(wgpu::Face::Back));
    }

    #[test]
    fn test_get_front_face_info_returns_correct_element() {
        // get_front_face_info should return the correct FRONT_FACES element
        let ccw_info = super::get_front_face_info(wgpu::FrontFace::Ccw);
        assert_eq!(*ccw_info, super::FRONT_FACES[0]);

        let cw_info = super::get_front_face_info(wgpu::FrontFace::Cw);
        assert_eq!(*cw_info, super::FRONT_FACES[1]);
    }

    #[test]
    fn test_get_cull_mode_info_returns_correct_element() {
        // get_cull_mode_info should return the correct CULL_MODES element
        let none_info = super::get_cull_mode_info(None);
        assert_eq!(*none_info, super::CULL_MODES[0]);

        let front_info = super::get_cull_mode_info(Some(wgpu::Face::Front));
        assert_eq!(*front_info, super::CULL_MODES[1]);

        let back_info = super::get_cull_mode_info(Some(wgpu::Face::Back));
        assert_eq!(*back_info, super::CULL_MODES[2]);
    }

    #[test]
    fn test_front_face_info_descriptions_non_empty() {
        for info in &super::FRONT_FACES {
            assert!(
                info.description.len() >= 20,
                "{} description too short: '{}'",
                info.name,
                info.description
            );
        }
    }

    #[test]
    fn test_cull_mode_info_descriptions_non_empty() {
        for info in &super::CULL_MODES {
            assert!(
                info.description.len() >= 20,
                "{} description too short: '{}'",
                info.name,
                info.description
            );
        }
    }

    #[test]
    fn test_all_front_face_cull_mode_combinations() {
        // Test all 6 combinations (2 front faces * 3 cull modes)
        let front_faces = [wgpu::FrontFace::Ccw, wgpu::FrontFace::Cw];
        let cull_modes = [None, Some(wgpu::Face::Front), Some(wgpu::Face::Back)];

        for front_face in front_faces {
            for cull_mode in cull_modes {
                let state = PrimitiveStateDescriptor::new()
                    .front_face(front_face)
                    .cull_mode(cull_mode);
                assert_eq!(state.front_face, front_face);
                assert_eq!(state.cull_mode, cull_mode);

                // Verify wgpu conversion
                let wgpu_state: wgpu::PrimitiveState = state.into();
                assert_eq!(wgpu_state.front_face, front_face);
                assert_eq!(wgpu_state.cull_mode, cull_mode);
            }
        }
    }

    #[test]
    fn test_triangle_list_preset_culling() {
        // triangle_list() should have default culling (back)
        let state = PrimitiveStateDescriptor::triangle_list();
        assert_eq!(state.front_face, wgpu::FrontFace::Ccw);
        assert_eq!(state.cull_mode, Some(wgpu::Face::Back));
    }

    #[test]
    fn test_triangle_strip_preset_culling() {
        // triangle_strip() should have default culling (back)
        let state = PrimitiveStateDescriptor::triangle_strip(wgpu::IndexFormat::Uint16);
        assert_eq!(state.front_face, wgpu::FrontFace::Ccw);
        assert_eq!(state.cull_mode, Some(wgpu::Face::Back));
    }

    #[test]
    fn test_line_list_preset_no_culling() {
        // line_list() should have culling disabled
        let state = PrimitiveStateDescriptor::line_list();
        assert_eq!(state.cull_mode, None);
    }

    #[test]
    fn test_line_strip_preset_no_culling() {
        // line_strip() should have culling disabled
        let state = PrimitiveStateDescriptor::line_strip(wgpu::IndexFormat::Uint16);
        assert_eq!(state.cull_mode, None);
    }

    #[test]
    fn test_point_list_preset_no_culling() {
        // point_list() should have culling disabled
        let state = PrimitiveStateDescriptor::point_list();
        assert_eq!(state.cull_mode, None);
    }

    #[test]
    fn test_culling_with_wireframe() {
        // Wireframe typically uses no culling for visibility
        let state = PrimitiveStateDescriptor::new()
            .wireframe()
            .cull_none();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Line);
        assert_eq!(state.cull_mode, None);
    }

    #[test]
    fn test_culling_with_point_mode() {
        // Point mode typically uses no culling
        let state = PrimitiveStateDescriptor::new()
            .point()
            .cull_none();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Point);
        assert_eq!(state.cull_mode, None);
    }

    #[test]
    fn test_directx_style_configuration() {
        // DirectX convention: CW front face, cull back faces
        let state = PrimitiveStateDescriptor::new()
            .cw()
            .cull_back();
        assert_eq!(state.front_face, wgpu::FrontFace::Cw);
        assert_eq!(state.cull_mode, Some(wgpu::Face::Back));

        let wgpu_state: wgpu::PrimitiveState = state.into();
        assert_eq!(wgpu_state.front_face, wgpu::FrontFace::Cw);
        assert_eq!(wgpu_state.cull_mode, Some(wgpu::Face::Back));
    }

    #[test]
    fn test_opengl_style_configuration() {
        // OpenGL/wgpu convention: CCW front face, cull back faces
        let state = PrimitiveStateDescriptor::new()
            .ccw()
            .cull_back();
        assert_eq!(state.front_face, wgpu::FrontFace::Ccw);
        assert_eq!(state.cull_mode, Some(wgpu::Face::Back));
    }

    #[test]
    fn test_shadow_volume_configuration() {
        // Shadow volumes often render back faces
        let state = PrimitiveStateDescriptor::new().cull_front();
        assert_eq!(state.cull_mode, Some(wgpu::Face::Front));
    }

    #[test]
    fn test_foliage_configuration() {
        // Foliage typically uses no culling for two-sided leaves
        let state = PrimitiveStateDescriptor::triangle_list().cull_none();
        assert_eq!(state.topology, wgpu::PrimitiveTopology::TriangleList);
        assert_eq!(state.cull_mode, None);
    }

    #[test]
    fn test_no_culling_alias_equals_cull_none() {
        // no_culling() and cull_none() should produce identical results
        let state1 = PrimitiveStateDescriptor::new().no_culling();
        let state2 = PrimitiveStateDescriptor::new().cull_none();
        assert_eq!(state1, state2);
    }

    #[test]
    fn test_wgpu_full_state_conversion_with_culling() {
        // Full state conversion test with all fields
        let state = PrimitiveStateDescriptor::new()
            .topology(wgpu::PrimitiveTopology::TriangleList)
            .cw()
            .cull_front()
            .polygon_mode(wgpu::PolygonMode::Fill)
            .conservative(false)
            .unclipped_depth(false);

        let wgpu_state: wgpu::PrimitiveState = state.into();
        assert_eq!(wgpu_state.topology, wgpu::PrimitiveTopology::TriangleList);
        assert_eq!(wgpu_state.front_face, wgpu::FrontFace::Cw);
        assert_eq!(wgpu_state.cull_mode, Some(wgpu::Face::Front));
        assert_eq!(wgpu_state.polygon_mode, wgpu::PolygonMode::Fill);
        assert!(!wgpu_state.conservative);
        assert!(!wgpu_state.unclipped_depth);
    }

    #[test]
    fn test_front_face_info_names_unique() {
        // All front face names should be unique
        let names: Vec<_> = super::FRONT_FACES.iter().map(|i| i.name).collect();
        for (i, name) in names.iter().enumerate() {
            for (j, other) in names.iter().enumerate() {
                if i != j {
                    assert_ne!(name, other, "Duplicate front face name: {}", name);
                }
            }
        }
    }

    #[test]
    fn test_cull_mode_info_names_unique() {
        // All cull mode names should be unique
        let names: Vec<_> = super::CULL_MODES.iter().map(|i| i.name).collect();
        for (i, name) in names.iter().enumerate() {
            for (j, other) in names.iter().enumerate() {
                if i != j {
                    assert_ne!(name, other, "Duplicate cull mode name: {}", name);
                }
            }
        }
    }

    #[test]
    fn test_front_face_info_eq_reflexive() {
        for info in &super::FRONT_FACES {
            assert_eq!(*info, *info, "FrontFaceInfo should be equal to itself");
        }
    }

    #[test]
    fn test_cull_mode_info_eq_reflexive() {
        for info in &super::CULL_MODES {
            assert_eq!(*info, *info, "CullModeInfo should be equal to itself");
        }
    }

    #[test]
    fn test_front_face_toggle() {
        // Test toggling between front faces
        let state_ccw = PrimitiveStateDescriptor::new().ccw();
        let state_cw = state_ccw.cw();
        let state_ccw_again = state_cw.ccw();

        assert_eq!(state_ccw.front_face, wgpu::FrontFace::Ccw);
        assert_eq!(state_cw.front_face, wgpu::FrontFace::Cw);
        assert_eq!(state_ccw_again.front_face, wgpu::FrontFace::Ccw);
    }

    #[test]
    fn test_cull_mode_cycle() {
        // Test cycling through cull modes
        let state_back = PrimitiveStateDescriptor::new().cull_back();
        let state_front = state_back.cull_front();
        let state_none = state_front.cull_none();
        let state_back_again = state_none.cull_back();

        assert_eq!(state_back.cull_mode, Some(wgpu::Face::Back));
        assert_eq!(state_front.cull_mode, Some(wgpu::Face::Front));
        assert_eq!(state_none.cull_mode, None);
        assert_eq!(state_back_again.cull_mode, Some(wgpu::Face::Back));
    }

    // =========================================================================
    // WHITEBOX - T-WGPU-P3.3.3 Polygon Modes Additional Tests
    // =========================================================================

    // -------------------------------------------------------------------------
    // Polygon Mode Default Value Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_polygon_mode_fill_is_default() {
        // Fill is the default polygon mode
        let state = PrimitiveStateDescriptor::new();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Fill);
    }

    #[test]
    fn test_polygon_mode_default_matches_default_impl() {
        let new_state = PrimitiveStateDescriptor::new();
        let default_state = PrimitiveStateDescriptor::default();
        assert_eq!(new_state.polygon_mode, default_state.polygon_mode);
        assert_eq!(new_state.polygon_mode, wgpu::PolygonMode::Fill);
    }

    #[test]
    fn test_triangle_list_preset_polygon_mode() {
        let state = PrimitiveStateDescriptor::triangle_list();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Fill);
    }

    #[test]
    fn test_triangle_strip_preset_polygon_mode() {
        let state = PrimitiveStateDescriptor::triangle_strip(wgpu::IndexFormat::Uint16);
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Fill);
    }

    #[test]
    fn test_point_list_preset_polygon_mode() {
        let state = PrimitiveStateDescriptor::point_list();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Fill);
    }

    // -------------------------------------------------------------------------
    // Polygon Mode + Topology Combination Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_polygon_line_with_triangle_list() {
        // Wireframe with triangle list topology
        let state = PrimitiveStateDescriptor::triangle_list().polygon_line();
        assert_eq!(state.topology, wgpu::PrimitiveTopology::TriangleList);
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Line);
    }

    #[test]
    fn test_polygon_line_with_triangle_strip() {
        // Wireframe with triangle strip topology
        let state = PrimitiveStateDescriptor::triangle_strip(wgpu::IndexFormat::Uint32)
            .polygon_line();
        assert_eq!(state.topology, wgpu::PrimitiveTopology::TriangleStrip);
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Line);
        assert_eq!(state.strip_index_format, Some(wgpu::IndexFormat::Uint32));
    }

    #[test]
    fn test_polygon_point_with_triangle_list() {
        // Point rendering with triangle list topology
        let state = PrimitiveStateDescriptor::triangle_list().polygon_point();
        assert_eq!(state.topology, wgpu::PrimitiveTopology::TriangleList);
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Point);
    }

    #[test]
    fn test_polygon_modes_with_all_topologies() {
        // Test all polygon modes with all topologies
        let topologies = [
            wgpu::PrimitiveTopology::PointList,
            wgpu::PrimitiveTopology::LineList,
            wgpu::PrimitiveTopology::LineStrip,
            wgpu::PrimitiveTopology::TriangleList,
            wgpu::PrimitiveTopology::TriangleStrip,
        ];
        let polygon_modes = [
            wgpu::PolygonMode::Fill,
            wgpu::PolygonMode::Line,
            wgpu::PolygonMode::Point,
        ];

        for topology in topologies {
            for mode in polygon_modes {
                let state = PrimitiveStateDescriptor::new()
                    .topology(topology)
                    .polygon_mode(mode);
                assert_eq!(state.topology, topology);
                assert_eq!(state.polygon_mode, mode);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Polygon Mode + Culling Combination Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_polygon_line_cull_back() {
        // Wireframe with back-face culling
        let state = PrimitiveStateDescriptor::new()
            .polygon_line()
            .cull_back();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Line);
        assert_eq!(state.cull_mode, Some(wgpu::Face::Back));
    }

    #[test]
    fn test_polygon_point_cull_none() {
        // Point rendering typically needs no culling
        let state = PrimitiveStateDescriptor::new()
            .polygon_point()
            .cull_none();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Point);
        assert_eq!(state.cull_mode, None);
    }

    #[test]
    fn test_polygon_fill_all_cull_modes() {
        // Fill mode with all cull modes
        for cull_mode in [None, Some(wgpu::Face::Front), Some(wgpu::Face::Back)] {
            let state = PrimitiveStateDescriptor::new()
                .polygon_fill()
                .cull_mode(cull_mode);
            assert_eq!(state.polygon_mode, wgpu::PolygonMode::Fill);
            assert_eq!(state.cull_mode, cull_mode);
        }
    }

    // -------------------------------------------------------------------------
    // Polygon Mode Feature Flag Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_requires_non_fill_feature_exhaustive() {
        // Test all polygon modes
        assert!(!super::requires_non_fill_feature(wgpu::PolygonMode::Fill));
        assert!(super::requires_non_fill_feature(wgpu::PolygonMode::Line));
        assert!(super::requires_non_fill_feature(wgpu::PolygonMode::Point));
    }

    #[test]
    fn test_required_feature_for_fill_is_none() {
        let feature = super::required_feature_for_polygon_mode(wgpu::PolygonMode::Fill);
        assert!(feature.is_none());
    }

    #[test]
    fn test_required_feature_for_line_is_polygon_mode_line() {
        let feature = super::required_feature_for_polygon_mode(wgpu::PolygonMode::Line);
        assert!(feature.is_some());
        assert_eq!(feature.unwrap(), wgpu::Features::POLYGON_MODE_LINE);
    }

    #[test]
    fn test_required_feature_for_point_is_polygon_mode_point() {
        let feature = super::required_feature_for_polygon_mode(wgpu::PolygonMode::Point);
        assert!(feature.is_some());
        assert_eq!(feature.unwrap(), wgpu::Features::POLYGON_MODE_POINT);
    }

    #[test]
    fn test_requires_feature_matches_polygon_mode_info() {
        // Verify consistency between requires_non_fill_feature and POLYGON_MODES.requires_feature
        for info in &super::POLYGON_MODES {
            let requires = super::requires_non_fill_feature(info.mode);
            assert_eq!(
                requires, info.requires_feature,
                "Mismatch for {:?}: requires_non_fill_feature={}, POLYGON_MODES.requires_feature={}",
                info.mode, requires, info.requires_feature
            );
        }
    }

    #[test]
    fn test_required_feature_consistency() {
        // Verify required_feature_for_polygon_mode aligns with requires_non_fill_feature
        for info in &super::POLYGON_MODES {
            let feature = super::required_feature_for_polygon_mode(info.mode);
            let requires_non_fill = super::requires_non_fill_feature(info.mode);
            assert_eq!(
                feature.is_some(), requires_non_fill,
                "Mismatch for {:?}: feature.is_some()={}, requires_non_fill={}",
                info.mode, feature.is_some(), requires_non_fill
            );
        }
    }

    // -------------------------------------------------------------------------
    // POLYGON_MODES Constant Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_polygon_modes_constant_count() {
        assert_eq!(super::POLYGON_MODES.len(), 3);
    }

    #[test]
    fn test_polygon_modes_constant_order() {
        // Order should be Fill, Line, Point
        assert_eq!(super::POLYGON_MODES[0].mode, wgpu::PolygonMode::Fill);
        assert_eq!(super::POLYGON_MODES[1].mode, wgpu::PolygonMode::Line);
        assert_eq!(super::POLYGON_MODES[2].mode, wgpu::PolygonMode::Point);
    }

    #[test]
    fn test_polygon_modes_names_order() {
        assert_eq!(super::POLYGON_MODES[0].name, "Fill");
        assert_eq!(super::POLYGON_MODES[1].name, "Line");
        assert_eq!(super::POLYGON_MODES[2].name, "Point");
    }

    #[test]
    fn test_polygon_modes_unique_names() {
        let names: Vec<_> = super::POLYGON_MODES.iter().map(|i| i.name).collect();
        for (i, name) in names.iter().enumerate() {
            for (j, other) in names.iter().enumerate() {
                if i != j {
                    assert_ne!(name, other, "Duplicate polygon mode name: {}", name);
                }
            }
        }
    }

    #[test]
    fn test_polygon_modes_use_cases_all_have_content() {
        for info in &super::POLYGON_MODES {
            assert!(
                info.use_cases.len() >= 3,
                "{} should have at least 3 use cases, found {}",
                info.name,
                info.use_cases.len()
            );
        }
    }

    // -------------------------------------------------------------------------
    // get_polygon_mode_info Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_get_polygon_mode_info_returns_correct_element() {
        let fill_info = super::get_polygon_mode_info(wgpu::PolygonMode::Fill);
        assert_eq!(*fill_info, super::POLYGON_MODES[0]);

        let line_info = super::get_polygon_mode_info(wgpu::PolygonMode::Line);
        assert_eq!(*line_info, super::POLYGON_MODES[1]);

        let point_info = super::get_polygon_mode_info(wgpu::PolygonMode::Point);
        assert_eq!(*point_info, super::POLYGON_MODES[2]);
    }

    #[test]
    fn test_get_polygon_mode_info_all_modes() {
        // Test that get_polygon_mode_info works for all modes in POLYGON_MODES
        for expected in &super::POLYGON_MODES {
            let info = super::get_polygon_mode_info(expected.mode);
            assert_eq!(info.mode, expected.mode);
            assert_eq!(info.name, expected.name);
            assert_eq!(info.description, expected.description);
            assert_eq!(info.requires_feature, expected.requires_feature);
        }
    }

    // -------------------------------------------------------------------------
    // wgpu Conversion with Polygon Modes
    // -------------------------------------------------------------------------

    #[test]
    fn test_wgpu_conversion_polygon_fill() {
        let state = PrimitiveStateDescriptor::new().polygon_fill();
        let wgpu_state: wgpu::PrimitiveState = state.into();
        assert_eq!(wgpu_state.polygon_mode, wgpu::PolygonMode::Fill);
    }

    #[test]
    fn test_wgpu_conversion_polygon_line() {
        let state = PrimitiveStateDescriptor::new().polygon_line();
        let wgpu_state: wgpu::PrimitiveState = state.into();
        assert_eq!(wgpu_state.polygon_mode, wgpu::PolygonMode::Line);
    }

    #[test]
    fn test_wgpu_conversion_polygon_point() {
        let state = PrimitiveStateDescriptor::new().polygon_point();
        let wgpu_state: wgpu::PrimitiveState = state.into();
        assert_eq!(wgpu_state.polygon_mode, wgpu::PolygonMode::Point);
    }

    #[test]
    fn test_wgpu_conversion_wireframe_method() {
        let state = PrimitiveStateDescriptor::new().wireframe();
        let wgpu_state: wgpu::PrimitiveState = state.into();
        assert_eq!(wgpu_state.polygon_mode, wgpu::PolygonMode::Line);
    }

    #[test]
    fn test_wgpu_conversion_point_method() {
        let state = PrimitiveStateDescriptor::new().point();
        let wgpu_state: wgpu::PrimitiveState = state.into();
        assert_eq!(wgpu_state.polygon_mode, wgpu::PolygonMode::Point);
    }

    // -------------------------------------------------------------------------
    // Polygon Mode + Conservative Rasterization Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_polygon_fill_conservative() {
        let state = PrimitiveStateDescriptor::new()
            .polygon_fill()
            .conservative(true);
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Fill);
        assert!(state.conservative);
    }

    #[test]
    fn test_polygon_line_conservative() {
        let state = PrimitiveStateDescriptor::new()
            .polygon_line()
            .conservative(true);
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Line);
        assert!(state.conservative);
    }

    #[test]
    fn test_polygon_point_conservative() {
        let state = PrimitiveStateDescriptor::new()
            .polygon_point()
            .conservative(true);
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Point);
        assert!(state.conservative);
    }

    #[test]
    fn test_all_polygon_modes_conservative_conversion() {
        for mode in [
            wgpu::PolygonMode::Fill,
            wgpu::PolygonMode::Line,
            wgpu::PolygonMode::Point,
        ] {
            let state = PrimitiveStateDescriptor::new()
                .polygon_mode(mode)
                .conservative(true);
            let wgpu_state: wgpu::PrimitiveState = state.into();
            assert_eq!(wgpu_state.polygon_mode, mode);
            assert!(wgpu_state.conservative);
        }
    }

    // -------------------------------------------------------------------------
    // Polygon Mode + Unclipped Depth Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_polygon_fill_unclipped_depth() {
        let state = PrimitiveStateDescriptor::new()
            .polygon_fill()
            .unclipped_depth(true);
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Fill);
        assert!(state.unclipped_depth);
    }

    #[test]
    fn test_polygon_line_unclipped_depth() {
        let state = PrimitiveStateDescriptor::new()
            .polygon_line()
            .unclipped_depth(true);
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Line);
        assert!(state.unclipped_depth);
    }

    #[test]
    fn test_polygon_point_unclipped_depth() {
        let state = PrimitiveStateDescriptor::new()
            .polygon_point()
            .unclipped_depth(true);
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Point);
        assert!(state.unclipped_depth);
    }

    // -------------------------------------------------------------------------
    // Full Combination Tests (Polygon Mode + Topology + Culling)
    // -------------------------------------------------------------------------

    #[test]
    fn test_wireframe_debug_configuration() {
        // Common debug configuration: wireframe, no culling, triangle list
        let state = PrimitiveStateDescriptor::triangle_list()
            .wireframe()
            .cull_none();

        assert_eq!(state.topology, wgpu::PrimitiveTopology::TriangleList);
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Line);
        assert_eq!(state.cull_mode, None);

        let wgpu_state: wgpu::PrimitiveState = state.into();
        assert_eq!(wgpu_state.topology, wgpu::PrimitiveTopology::TriangleList);
        assert_eq!(wgpu_state.polygon_mode, wgpu::PolygonMode::Line);
        assert_eq!(wgpu_state.cull_mode, None);
    }

    #[test]
    fn test_point_cloud_configuration() {
        // Point cloud: point mode, point list topology, no culling
        let state = PrimitiveStateDescriptor::point_list()
            .polygon_point()
            .cull_none();

        assert_eq!(state.topology, wgpu::PrimitiveTopology::PointList);
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Point);
        assert_eq!(state.cull_mode, None);
    }

    #[test]
    fn test_filled_mesh_configuration() {
        // Standard filled mesh: fill mode, triangle list, back-face culling
        let state = PrimitiveStateDescriptor::triangle_list()
            .polygon_fill()
            .cull_back();

        assert_eq!(state.topology, wgpu::PrimitiveTopology::TriangleList);
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Fill);
        assert_eq!(state.cull_mode, Some(wgpu::Face::Back));
    }

    #[test]
    fn test_full_combination_all_settings() {
        // Test all settings together
        let state = PrimitiveStateDescriptor::new()
            .topology(wgpu::PrimitiveTopology::TriangleStrip)
            .strip_index_format(Some(wgpu::IndexFormat::Uint16))
            .polygon_line()
            .cw()
            .cull_front()
            .conservative(true)
            .unclipped_depth(true);

        assert_eq!(state.topology, wgpu::PrimitiveTopology::TriangleStrip);
        assert_eq!(state.strip_index_format, Some(wgpu::IndexFormat::Uint16));
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Line);
        assert_eq!(state.front_face, wgpu::FrontFace::Cw);
        assert_eq!(state.cull_mode, Some(wgpu::Face::Front));
        assert!(state.conservative);
        assert!(state.unclipped_depth);

        let wgpu_state: wgpu::PrimitiveState = state.into();
        assert_eq!(wgpu_state.topology, wgpu::PrimitiveTopology::TriangleStrip);
        assert_eq!(wgpu_state.strip_index_format, Some(wgpu::IndexFormat::Uint16));
        assert_eq!(wgpu_state.polygon_mode, wgpu::PolygonMode::Line);
        assert_eq!(wgpu_state.front_face, wgpu::FrontFace::Cw);
        assert_eq!(wgpu_state.cull_mode, Some(wgpu::Face::Front));
        assert!(wgpu_state.conservative);
        assert!(wgpu_state.unclipped_depth);
    }

    // -------------------------------------------------------------------------
    // Polygon Mode Override Chain Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_polygon_mode_chain_fill_to_line() {
        let state = PrimitiveStateDescriptor::new()
            .polygon_fill()
            .polygon_line();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Line);
    }

    #[test]
    fn test_polygon_mode_chain_line_to_point() {
        let state = PrimitiveStateDescriptor::new()
            .polygon_line()
            .polygon_point();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Point);
    }

    #[test]
    fn test_polygon_mode_chain_point_to_fill() {
        let state = PrimitiveStateDescriptor::new()
            .polygon_point()
            .polygon_fill();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Fill);
    }

    #[test]
    fn test_polygon_mode_multiple_overrides() {
        let state = PrimitiveStateDescriptor::new()
            .polygon_fill()
            .polygon_line()
            .polygon_point()
            .polygon_fill()
            .polygon_line();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Line);
    }

    #[test]
    fn test_wireframe_then_polygon_fill() {
        let state = PrimitiveStateDescriptor::new()
            .wireframe()
            .polygon_fill();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Fill);
    }

    #[test]
    fn test_point_then_wireframe() {
        let state = PrimitiveStateDescriptor::new()
            .point()
            .wireframe();
        assert_eq!(state.polygon_mode, wgpu::PolygonMode::Line);
    }

    // -------------------------------------------------------------------------
    // PolygonModeInfo Struct Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_polygon_mode_info_eq_reflexive() {
        for info in &super::POLYGON_MODES {
            assert_eq!(*info, *info, "PolygonModeInfo should be equal to itself");
        }
    }

    #[test]
    fn test_polygon_mode_info_eq_symmetric() {
        let a = super::POLYGON_MODES[0];
        let b = super::POLYGON_MODES[0];
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn test_polygon_mode_info_ne_different_modes() {
        assert_ne!(super::POLYGON_MODES[0], super::POLYGON_MODES[1]);
        assert_ne!(super::POLYGON_MODES[1], super::POLYGON_MODES[2]);
        assert_ne!(super::POLYGON_MODES[0], super::POLYGON_MODES[2]);
    }

    #[test]
    fn test_polygon_mode_info_debug_contains_mode() {
        for info in &super::POLYGON_MODES {
            let debug_str = format!("{:?}", info);
            assert!(debug_str.contains("PolygonModeInfo"));
            assert!(debug_str.contains(info.name));
        }
    }

    // -------------------------------------------------------------------------
    // Cross-Validation Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_polygon_mode_helpers_match_builder_methods() {
        // wireframe() == polygon_line()
        let wireframe = PrimitiveStateDescriptor::new().wireframe();
        let polygon_line = PrimitiveStateDescriptor::new().polygon_line();
        assert_eq!(wireframe.polygon_mode, polygon_line.polygon_mode);

        // point() == polygon_point()
        let point = PrimitiveStateDescriptor::new().point();
        let polygon_point = PrimitiveStateDescriptor::new().polygon_point();
        assert_eq!(point.polygon_mode, polygon_point.polygon_mode);
    }

    #[test]
    fn test_polygon_mode_direct_vs_builder() {
        // Direct polygon_mode() vs builder methods
        let direct_fill = PrimitiveStateDescriptor::new().polygon_mode(wgpu::PolygonMode::Fill);
        let builder_fill = PrimitiveStateDescriptor::new().polygon_fill();
        assert_eq!(direct_fill.polygon_mode, builder_fill.polygon_mode);

        let direct_line = PrimitiveStateDescriptor::new().polygon_mode(wgpu::PolygonMode::Line);
        let builder_line = PrimitiveStateDescriptor::new().polygon_line();
        assert_eq!(direct_line.polygon_mode, builder_line.polygon_mode);

        let direct_point = PrimitiveStateDescriptor::new().polygon_mode(wgpu::PolygonMode::Point);
        let builder_point = PrimitiveStateDescriptor::new().polygon_point();
        assert_eq!(direct_point.polygon_mode, builder_point.polygon_mode);
    }

    #[test]
    fn test_only_fill_mode_needs_no_feature() {
        // Only Fill mode should not require a feature
        let fill_requires = super::requires_non_fill_feature(wgpu::PolygonMode::Fill);
        let line_requires = super::requires_non_fill_feature(wgpu::PolygonMode::Line);
        let point_requires = super::requires_non_fill_feature(wgpu::PolygonMode::Point);

        assert!(!fill_requires, "Fill should not require feature");
        assert!(line_requires, "Line should require feature");
        assert!(point_requires, "Point should require feature");

        // Count: exactly 1 mode should not require feature
        let non_fill_count = [fill_requires, line_requires, point_requires]
            .iter()
            .filter(|&&r| r)
            .count();
        assert_eq!(non_fill_count, 2, "Exactly 2 modes should require features");
    }

    #[test]
    fn test_feature_flags_are_distinct() {
        // Line and Point require different features
        let line_feature = super::required_feature_for_polygon_mode(wgpu::PolygonMode::Line);
        let point_feature = super::required_feature_for_polygon_mode(wgpu::PolygonMode::Point);

        assert!(line_feature.is_some());
        assert!(point_feature.is_some());
        assert_ne!(line_feature.unwrap(), point_feature.unwrap());
    }
}

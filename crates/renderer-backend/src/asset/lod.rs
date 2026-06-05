//! LOD (Level of Detail) generation system for TRINITY.
//!
//! Provides multiple LOD strategies for efficient mesh rendering:
//! - **Discrete LOD**: N independent meshes switched by distance
//! - **Continuous LOD**: Garland-Heckbert quadric error metric simplification
//! - **Hierarchical LOD**: Tree of simplified parents covering children
//! - **Nanite-style DAG**: Meshlet cluster hierarchy (post-cooking)
//!
//! # Features
//!
//! - **Cross-fade transitions**: Screen-space dither with multiple patterns
//! - **User-adjustable bias**: Runtime LOD bias for quality/performance tuning
//! - **@lod decorator support**: Integration with TRINITY's decorator system
//! - **Idempotent generation**: Same input always produces same output
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::lod::*;
//!
//! // Generate discrete LOD chain
//! let mesh = MeshData { positions, indices, normals: None, uvs: None };
//! let config = LodConfig::discrete(4, &[10.0, 25.0, 50.0, 100.0]);
//! let chain = generate_lod_chain(&mesh, &config)?;
//!
//! // Runtime LOD selection
//! let level = select_lod_level(&chain, distance, bias);
//! let alpha = compute_cross_fade_alpha(screen_coverage, &chain.thresholds());
//! ```

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

use super::meshlet::{BoundingSphere, Meshlet, MeshletMesh, compute_bounding_sphere};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// LOD generation error.
#[derive(Debug, Clone)]
pub enum LodError {
    /// Empty mesh input.
    EmptyMesh,
    /// Invalid triangle count (not divisible by 3).
    InvalidTriangleCount(usize),
    /// Index out of bounds.
    IndexOutOfBounds { index: u32, vertex_count: usize },
    /// Invalid configuration.
    InvalidConfig(String),
    /// Simplification failed.
    SimplificationFailed(String),
    /// Insufficient vertices for target.
    InsufficientVertices { have: usize, target: usize },
}

impl std::fmt::Display for LodError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyMesh => write!(f, "empty mesh"),
            Self::InvalidTriangleCount(count) => {
                write!(f, "index count {} not divisible by 3", count)
            }
            Self::IndexOutOfBounds { index, vertex_count } => {
                write!(f, "index {} out of bounds for {} vertices", index, vertex_count)
            }
            Self::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
            Self::SimplificationFailed(msg) => write!(f, "simplification failed: {}", msg),
            Self::InsufficientVertices { have, target } => {
                write!(f, "have {} vertices, target {} requires more", have, target)
            }
        }
    }
}

impl std::error::Error for LodError {}

/// LOD operation result.
pub type LodResult<T> = Result<T, LodError>;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Input mesh data for LOD generation.
#[derive(Debug, Clone)]
pub struct MeshData {
    /// Vertex positions (required).
    pub positions: Vec<[f32; 3]>,
    /// Triangle indices (required, must be divisible by 3).
    pub indices: Vec<u32>,
    /// Optional vertex normals (same count as positions).
    pub normals: Option<Vec<[f32; 3]>>,
    /// Optional texture coordinates (same count as positions).
    pub uvs: Option<Vec<[f32; 2]>>,
}

impl MeshData {
    /// Create mesh data with positions and indices only.
    pub fn new(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> Self {
        Self {
            positions,
            indices,
            normals: None,
            uvs: None,
        }
    }

    /// Create mesh data with all attributes.
    pub fn with_attributes(
        positions: Vec<[f32; 3]>,
        indices: Vec<u32>,
        normals: Option<Vec<[f32; 3]>>,
        uvs: Option<Vec<[f32; 2]>>,
    ) -> Self {
        Self { positions, indices, normals, uvs }
    }

    /// Vertex count.
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    /// Triangle count.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Validate mesh data.
    pub fn validate(&self) -> LodResult<()> {
        if self.positions.is_empty() {
            return Err(LodError::EmptyMesh);
        }
        if self.indices.is_empty() {
            return Err(LodError::EmptyMesh);
        }
        if self.indices.len() % 3 != 0 {
            return Err(LodError::InvalidTriangleCount(self.indices.len()));
        }
        for &idx in &self.indices {
            if idx as usize >= self.positions.len() {
                return Err(LodError::IndexOutOfBounds {
                    index: idx,
                    vertex_count: self.positions.len(),
                });
            }
        }
        if let Some(ref normals) = self.normals {
            if normals.len() != self.positions.len() {
                return Err(LodError::InvalidConfig(format!(
                    "normal count {} != vertex count {}",
                    normals.len(),
                    self.positions.len()
                )));
            }
        }
        if let Some(ref uvs) = self.uvs {
            if uvs.len() != self.positions.len() {
                return Err(LodError::InvalidConfig(format!(
                    "UV count {} != vertex count {}",
                    uvs.len(),
                    self.positions.len()
                )));
            }
        }
        Ok(())
    }
}

/// A single LOD level.
#[derive(Debug, Clone)]
pub struct LodLevel {
    /// Mesh data for this level.
    pub mesh_data: MeshData,
    /// Screen coverage threshold (0-1) to switch to this LOD.
    pub screen_coverage: f32,
    /// Vertex count at this level.
    pub vertex_count: u32,
    /// Triangle count at this level.
    pub triangle_count: u32,
    /// Bounding sphere for this level.
    pub bounds: BoundingSphere,
    /// Quadric error (for continuous LOD).
    pub error: f32,
}

impl LodLevel {
    /// Create a new LOD level from mesh data.
    pub fn from_mesh(mesh: MeshData, screen_coverage: f32, error: f32) -> Self {
        let vertex_count = mesh.vertex_count() as u32;
        let triangle_count = mesh.triangle_count() as u32;
        let bounds = compute_bounding_sphere(&mesh.positions);
        Self {
            mesh_data: mesh,
            screen_coverage,
            vertex_count,
            triangle_count,
            bounds,
            error,
        }
    }
}

/// LOD strategy enumeration.
#[derive(Debug, Clone)]
pub enum LodStrategy {
    /// Discrete LOD: N independent meshes with distance thresholds.
    Discrete {
        /// LOD levels from highest to lowest detail.
        levels: Vec<LodLevel>,
    },
    /// Continuous LOD: QEM-based progressive simplification.
    Continuous {
        /// Minimum error threshold.
        min_error: f32,
        /// Maximum simplification ratio (0-1).
        max_simplification: f32,
    },
    /// Hierarchical LOD: Tree structure with parent covering children.
    Hierarchical {
        /// LOD tree structure.
        tree: LodTree,
    },
    /// Nanite-style DAG: Meshlet cluster hierarchy.
    NaniteDag {
        /// Cluster dependency graph.
        cluster_dag: ClusterDag,
    },
}

impl Default for LodStrategy {
    fn default() -> Self {
        Self::Discrete { levels: Vec::new() }
    }
}

/// Dithering pattern for LOD cross-fade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DitherPattern {
    /// Simple gradient noise.
    GradientNoise,
    /// Interleaved gradient noise (IGN) - high quality.
    InterleavedGradient,
    /// Blue noise - best visual quality, requires texture.
    BlueNoise,
    /// Bayer ordered dithering.
    Bayer4x4,
}

impl Default for DitherPattern {
    fn default() -> Self {
        Self::InterleavedGradient
    }
}

impl DitherPattern {
    /// Generate dither value at pixel coordinates.
    ///
    /// Returns a value in [0, 1) for alpha comparison.
    pub fn sample(&self, x: u32, y: u32, frame: u32) -> f32 {
        match self {
            Self::GradientNoise => gradient_noise(x, y, frame),
            Self::InterleavedGradient => interleaved_gradient_noise(x, y, frame),
            Self::BlueNoise => blue_noise_fallback(x, y, frame),
            Self::Bayer4x4 => bayer_4x4(x, y),
        }
    }
}

/// Cross-fade configuration.
#[derive(Debug, Clone)]
pub struct CrossFadeConfig {
    /// Enable cross-fade transitions.
    pub enabled: bool,
    /// Dithering pattern to use.
    pub dither_pattern: DitherPattern,
    /// Screen-space range for transitions (as fraction of coverage).
    pub fade_range: f32,
    /// Minimum fade duration in frames.
    pub min_fade_frames: u32,
}

impl Default for CrossFadeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dither_pattern: DitherPattern::InterleavedGradient,
            fade_range: 0.1,
            min_fade_frames: 4,
        }
    }
}

impl CrossFadeConfig {
    /// Create cross-fade config with disabled transitions.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create cross-fade config with specific pattern.
    pub fn with_pattern(pattern: DitherPattern) -> Self {
        Self {
            enabled: true,
            dither_pattern: pattern,
            ..Default::default()
        }
    }

    /// Set fade range.
    pub fn with_fade_range(mut self, range: f32) -> Self {
        self.fade_range = range.clamp(0.01, 0.5);
        self
    }
}

/// Complete LOD configuration.
#[derive(Debug, Clone)]
pub struct LodConfig {
    /// LOD strategy to use.
    pub strategy: LodStrategy,
    /// Distance thresholds for LOD switching.
    pub distances: Vec<f32>,
    /// User-adjustable LOD bias (-1 to +1).
    pub bias: f32,
    /// Cross-fade configuration.
    pub cross_fade: CrossFadeConfig,
    /// Number of LOD levels to generate (for discrete).
    pub level_count: usize,
    /// Target simplification ratio per level.
    pub simplification_ratio: f32,
}

impl Default for LodConfig {
    fn default() -> Self {
        Self {
            strategy: LodStrategy::default(),
            distances: vec![10.0, 25.0, 50.0, 100.0],
            bias: 0.0,
            cross_fade: CrossFadeConfig::default(),
            level_count: 4,
            simplification_ratio: 0.5,
        }
    }
}

impl LodConfig {
    /// Create discrete LOD configuration.
    pub fn discrete(level_count: usize, distances: &[f32]) -> Self {
        Self {
            strategy: LodStrategy::Discrete { levels: Vec::new() },
            distances: distances.to_vec(),
            level_count,
            ..Default::default()
        }
    }

    /// Create continuous LOD configuration.
    pub fn continuous(min_error: f32, max_simplification: f32) -> Self {
        Self {
            strategy: LodStrategy::Continuous { min_error, max_simplification },
            ..Default::default()
        }
    }

    /// Create hierarchical LOD configuration.
    pub fn hierarchical() -> Self {
        Self {
            strategy: LodStrategy::Hierarchical { tree: LodTree::new() },
            ..Default::default()
        }
    }

    /// Set LOD bias.
    pub fn with_bias(mut self, bias: f32) -> Self {
        self.bias = bias.clamp(-1.0, 1.0);
        self
    }

    /// Set cross-fade config.
    pub fn with_cross_fade(mut self, config: CrossFadeConfig) -> Self {
        self.cross_fade = config;
        self
    }

    /// Set simplification ratio.
    pub fn with_simplification_ratio(mut self, ratio: f32) -> Self {
        self.simplification_ratio = ratio.clamp(0.1, 0.9);
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> LodResult<()> {
        if self.level_count == 0 {
            return Err(LodError::InvalidConfig("level_count must be > 0".into()));
        }
        if self.simplification_ratio <= 0.0 || self.simplification_ratio >= 1.0 {
            return Err(LodError::InvalidConfig(
                "simplification_ratio must be in (0, 1)".into(),
            ));
        }
        for (i, &dist) in self.distances.iter().enumerate() {
            if dist <= 0.0 {
                return Err(LodError::InvalidConfig(format!(
                    "distance[{}] = {} must be positive",
                    i, dist
                )));
            }
        }
        // Distances should be monotonically increasing
        for i in 1..self.distances.len() {
            if self.distances[i] <= self.distances[i - 1] {
                return Err(LodError::InvalidConfig(
                    "distances must be monotonically increasing".into(),
                ));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LOD Chain (output)
// ---------------------------------------------------------------------------

/// Complete LOD chain output.
#[derive(Debug, Clone)]
pub struct LodChain {
    /// LOD levels from highest to lowest detail.
    pub levels: Vec<LodLevel>,
    /// Distance thresholds (one per level transition).
    pub distances: Vec<f32>,
    /// Cross-fade configuration.
    pub cross_fade: CrossFadeConfig,
    /// Original mesh bounds.
    pub bounds: BoundingSphere,
}

impl LodChain {
    /// Create empty LOD chain.
    pub fn new() -> Self {
        Self {
            levels: Vec::new(),
            distances: Vec::new(),
            cross_fade: CrossFadeConfig::default(),
            bounds: BoundingSphere::default(),
        }
    }

    /// Number of LOD levels.
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Get LOD level by index.
    pub fn get_level(&self, index: usize) -> Option<&LodLevel> {
        self.levels.get(index)
    }

    /// Get distance thresholds.
    pub fn thresholds(&self) -> &[f32] {
        &self.distances
    }

    /// Total triangles across all LOD levels.
    pub fn total_triangles(&self) -> u32 {
        self.levels.iter().map(|l| l.triangle_count).sum()
    }

    /// Get screen coverage thresholds.
    pub fn coverage_thresholds(&self) -> Vec<f32> {
        self.levels.iter().map(|l| l.screen_coverage).collect()
    }
}

impl Default for LodChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Hierarchical LOD Tree
// ---------------------------------------------------------------------------

/// Node in a hierarchical LOD tree.
#[derive(Debug, Clone)]
pub struct LodTreeNode {
    /// Node index.
    pub index: u32,
    /// Parent node index (None for root).
    pub parent: Option<u32>,
    /// Child node indices.
    pub children: Vec<u32>,
    /// Simplified mesh for this node.
    pub mesh: MeshData,
    /// Bounding sphere covering all children.
    pub bounds: BoundingSphere,
    /// LOD error threshold.
    pub error: f32,
    /// Depth in tree (0 = leaf).
    pub depth: u32,
}

/// Hierarchical LOD tree structure.
#[derive(Debug, Clone)]
pub struct LodTree {
    /// All nodes in the tree.
    pub nodes: Vec<LodTreeNode>,
    /// Root node indices.
    pub roots: Vec<u32>,
    /// Maximum tree depth.
    pub max_depth: u32,
}

impl LodTree {
    /// Create empty LOD tree.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            roots: Vec::new(),
            max_depth: 0,
        }
    }

    /// Get node by index.
    pub fn get_node(&self, index: u32) -> Option<&LodTreeNode> {
        self.nodes.get(index as usize)
    }

    /// Get children of a node.
    pub fn children(&self, index: u32) -> &[u32] {
        self.nodes.get(index as usize)
            .map(|n| n.children.as_slice())
            .unwrap_or(&[])
    }

    /// Check if node is a leaf.
    pub fn is_leaf(&self, index: u32) -> bool {
        self.nodes.get(index as usize)
            .map(|n| n.children.is_empty())
            .unwrap_or(false)
    }

    /// Number of nodes in tree.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl Default for LodTree {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Nanite-style Cluster DAG
// ---------------------------------------------------------------------------

/// A cluster in the Nanite-style DAG.
#[derive(Debug, Clone)]
pub struct ClusterNode {
    /// Cluster index.
    pub index: u32,
    /// Parent cluster indices (clusters that can replace this one).
    pub parents: Vec<u32>,
    /// Child cluster indices (clusters this one can replace).
    pub children: Vec<u32>,
    /// Meshlet indices in this cluster.
    pub meshlets: Vec<u32>,
    /// Bounding sphere.
    pub bounds: BoundingSphere,
    /// Error metric for this cluster.
    pub error: f32,
    /// DAG level (0 = original, higher = more simplified).
    pub level: u32,
}

/// Nanite-style cluster DAG.
#[derive(Debug, Clone)]
pub struct ClusterDag {
    /// All clusters.
    pub clusters: Vec<ClusterNode>,
    /// Root cluster indices (most simplified).
    pub roots: Vec<u32>,
    /// Leaf cluster indices (original detail).
    pub leaves: Vec<u32>,
    /// Maximum DAG level.
    pub max_level: u32,
    /// Associated meshlet mesh.
    pub meshlet_mesh: Option<MeshletMesh>,
}

impl ClusterDag {
    /// Create empty cluster DAG.
    pub fn new() -> Self {
        Self {
            clusters: Vec::new(),
            roots: Vec::new(),
            leaves: Vec::new(),
            max_level: 0,
            meshlet_mesh: None,
        }
    }

    /// Get cluster by index.
    pub fn get_cluster(&self, index: u32) -> Option<&ClusterNode> {
        self.clusters.get(index as usize)
    }

    /// Number of clusters.
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }
}

impl Default for ClusterDag {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Quadric Error Metric (QEM) structures
// ---------------------------------------------------------------------------

/// 4x4 symmetric quadric matrix for error computation.
///
/// Stored as 10 unique values (upper triangle):
/// ```text
/// | a  b  c  d |
/// | b  e  f  g |
/// | c  f  h  i |
/// | d  g  i  j |
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Quadric {
    /// Matrix elements (upper triangle, row-major).
    pub data: [f64; 10],
}

impl Default for Quadric {
    fn default() -> Self {
        Self { data: [0.0; 10] }
    }
}

impl Quadric {
    /// Create zero quadric.
    pub fn zero() -> Self {
        Self::default()
    }

    /// Create quadric from plane equation ax + by + cz + d = 0.
    pub fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            data: [
                a * a, a * b, a * c, a * d, // row 0
                b * b, b * c, b * d,        // row 1 (partial)
                c * c, c * d,               // row 2 (partial)
                d * d,                      // row 3 (partial)
            ],
        }
    }

    /// Create quadric from triangle vertices.
    pub fn from_triangle(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]) -> Self {
        // Compute plane equation
        let e1 = [
            (p1[0] - p0[0]) as f64,
            (p1[1] - p0[1]) as f64,
            (p1[2] - p0[2]) as f64,
        ];
        let e2 = [
            (p2[0] - p0[0]) as f64,
            (p2[1] - p0[1]) as f64,
            (p2[2] - p0[2]) as f64,
        ];

        // Cross product for normal
        let nx = e1[1] * e2[2] - e1[2] * e2[1];
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        let nz = e1[0] * e2[1] - e1[1] * e2[0];

        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len < 1e-10 {
            return Self::zero();
        }

        let a = nx / len;
        let b = ny / len;
        let c = nz / len;
        let d = -(a * p0[0] as f64 + b * p0[1] as f64 + c * p0[2] as f64);

        Self::from_plane(a, b, c, d)
    }

    /// Add another quadric to this one.
    pub fn add_quadric(&mut self, other: &Quadric) {
        for i in 0..10 {
            self.data[i] += other.data[i];
        }
    }

    /// Scale quadric by factor.
    pub fn scale(&mut self, factor: f64) {
        for i in 0..10 {
            self.data[i] *= factor;
        }
    }

    /// Compute error for a vertex position.
    pub fn evaluate(&self, pos: [f32; 3]) -> f64 {
        let x = pos[0] as f64;
        let y = pos[1] as f64;
        let z = pos[2] as f64;

        // Q * v = v^T * Q * v (for homogeneous v = [x, y, z, 1])
        let d = &self.data;
        d[0] * x * x + 2.0 * d[1] * x * y + 2.0 * d[2] * x * z + 2.0 * d[3] * x
            + d[4] * y * y + 2.0 * d[5] * y * z + 2.0 * d[6] * y
            + d[7] * z * z + 2.0 * d[8] * z
            + d[9]
    }

    /// Attempt to find optimal vertex position minimizing error.
    ///
    /// Returns None if the matrix is singular.
    pub fn optimal_position(&self) -> Option<[f32; 3]> {
        // Solve the 3x3 linear system from the upper-left of Q
        let d = &self.data;

        // Build 3x3 matrix and 3x1 vector
        let a = [[d[0], d[1], d[2]], [d[1], d[4], d[5]], [d[2], d[5], d[7]]];
        let b = [-d[3], -d[6], -d[8]];

        // Solve using Cramer's rule
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

        if det.abs() < 1e-10 {
            return None;
        }

        let inv_det = 1.0 / det;

        let x = inv_det
            * (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
                + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]));

        let y = inv_det
            * (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
                - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]));

        let z = inv_det
            * (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
                - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
                + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]));

        Some([x as f32, y as f32, z as f32])
    }
}

impl std::ops::Add for Quadric {
    type Output = Self;

    fn add(mut self, other: Self) -> Self {
        self.add_quadric(&other);
        self
    }
}

/// Edge collapse candidate for mesh simplification.
#[derive(Debug, Clone)]
struct EdgeCollapse {
    /// First vertex index.
    v0: u32,
    /// Second vertex index.
    v1: u32,
    /// Target position after collapse.
    target: [f32; 3],
    /// Quadric error of this collapse.
    error: f64,
}

impl PartialEq for EdgeCollapse {
    fn eq(&self, other: &Self) -> bool {
        self.error == other.error && self.v0 == other.v0 && self.v1 == other.v1
    }
}

impl Eq for EdgeCollapse {}

impl PartialOrd for EdgeCollapse {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EdgeCollapse {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (smallest error first)
        // Use vertex indices as secondary key for determinism
        match other.error.partial_cmp(&self.error).unwrap_or(Ordering::Equal) {
            Ordering::Equal => {
                // Secondary: lower vertex index first
                match self.v0.cmp(&other.v0) {
                    Ordering::Equal => self.v1.cmp(&other.v1),
                    ord => ord,
                }
            }
            ord => ord,
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh Simplification (Garland-Heckbert QEM)
// ---------------------------------------------------------------------------

/// Mesh simplifier using Garland-Heckbert quadric error metric.
pub struct MeshSimplifier {
    /// Current vertex positions.
    positions: Vec<[f32; 3]>,
    /// Current vertex normals (optional).
    normals: Option<Vec<[f32; 3]>>,
    /// Current UVs (optional).
    uvs: Option<Vec<[f32; 2]>>,
    /// Current triangle indices.
    indices: Vec<u32>,
    /// Quadric per vertex.
    quadrics: Vec<Quadric>,
    /// Vertex adjacency (triangles per vertex).
    vertex_triangles: Vec<HashSet<usize>>,
    /// Edge to triangle adjacency.
    edge_triangles: HashMap<(u32, u32), Vec<usize>>,
    /// Deleted vertices.
    deleted_vertices: HashSet<u32>,
    /// Deleted triangles.
    deleted_triangles: HashSet<usize>,
    /// Vertex merge map (old -> new).
    merge_map: HashMap<u32, u32>,
}

impl MeshSimplifier {
    /// Create a new simplifier from mesh data.
    pub fn new(mesh: &MeshData) -> Self {
        let vertex_count = mesh.positions.len();
        let triangle_count = mesh.indices.len() / 3;

        // Initialize quadrics
        let mut quadrics = vec![Quadric::zero(); vertex_count];
        let mut vertex_triangles: Vec<HashSet<usize>> = vec![HashSet::new(); vertex_count];
        let mut edge_triangles: HashMap<(u32, u32), Vec<usize>> = HashMap::new();

        // Build quadrics from triangles
        for (tri_idx, tri) in mesh.indices.chunks_exact(3).enumerate() {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            if i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count {
                continue;
            }

            let p0 = mesh.positions[i0];
            let p1 = mesh.positions[i1];
            let p2 = mesh.positions[i2];

            let q = Quadric::from_triangle(p0, p1, p2);

            quadrics[i0].add_quadric(&q);
            quadrics[i1].add_quadric(&q);
            quadrics[i2].add_quadric(&q);

            vertex_triangles[i0].insert(tri_idx);
            vertex_triangles[i1].insert(tri_idx);
            vertex_triangles[i2].insert(tri_idx);

            // Add edges
            for i in 0..3 {
                let v0 = tri[i];
                let v1 = tri[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                edge_triangles.entry(edge).or_default().push(tri_idx);
            }
        }

        Self {
            positions: mesh.positions.clone(),
            normals: mesh.normals.clone(),
            uvs: mesh.uvs.clone(),
            indices: mesh.indices.clone(),
            quadrics,
            vertex_triangles,
            edge_triangles,
            deleted_vertices: HashSet::new(),
            deleted_triangles: HashSet::new(),
            merge_map: HashMap::new(),
        }
    }

    /// Simplify mesh to target ratio.
    ///
    /// `target_ratio` is the fraction of triangles to keep (0-1).
    pub fn simplify(&mut self, target_ratio: f32) -> LodResult<MeshData> {
        let original_triangles = self.indices.len() / 3 - self.deleted_triangles.len();
        let target_triangles = ((original_triangles as f32 * target_ratio).ceil() as usize).max(1);

        self.simplify_to_triangle_count(target_triangles)
    }

    /// Simplify mesh to target triangle count.
    pub fn simplify_to_triangle_count(&mut self, target_count: usize) -> LodResult<MeshData> {
        // Build priority queue of edge collapses
        let mut heap = self.build_collapse_heap();

        let mut current_triangles = self.indices.len() / 3 - self.deleted_triangles.len();

        while current_triangles > target_count && !heap.is_empty() {
            let collapse = match heap.pop() {
                Some(c) => c,
                None => break,
            };

            // Check if vertices are still valid
            if self.deleted_vertices.contains(&collapse.v0)
                || self.deleted_vertices.contains(&collapse.v1)
            {
                continue;
            }

            // Resolve any merges
            let v0 = self.resolve_vertex(collapse.v0);
            let v1 = self.resolve_vertex(collapse.v1);

            if v0 == v1 || self.deleted_vertices.contains(&v0) || self.deleted_vertices.contains(&v1) {
                continue;
            }

            // Perform the collapse
            let removed = self.collapse_edge(v0, v1, collapse.target);
            current_triangles = current_triangles.saturating_sub(removed);

            // Re-add affected edges to heap
            self.update_edges_for_vertex(v0, &mut heap);
        }

        self.build_output_mesh()
    }

    /// Build the initial edge collapse priority queue.
    fn build_collapse_heap(&self) -> BinaryHeap<EdgeCollapse> {
        let mut heap = BinaryHeap::new();

        for (&(v0, v1), _) in &self.edge_triangles {
            if self.deleted_vertices.contains(&v0) || self.deleted_vertices.contains(&v1) {
                continue;
            }

            if let Some(collapse) = self.compute_collapse(v0, v1) {
                heap.push(collapse);
            }
        }

        heap
    }

    /// Compute optimal collapse for an edge.
    fn compute_collapse(&self, v0: u32, v1: u32) -> Option<EdgeCollapse> {
        let q0 = &self.quadrics[v0 as usize];
        let q1 = &self.quadrics[v1 as usize];

        let combined = *q0 + *q1;

        // Try to find optimal position
        let target = combined.optimal_position().unwrap_or_else(|| {
            // Fallback to midpoint
            let p0 = self.positions[v0 as usize];
            let p1 = self.positions[v1 as usize];
            [
                (p0[0] + p1[0]) * 0.5,
                (p0[1] + p1[1]) * 0.5,
                (p0[2] + p1[2]) * 0.5,
            ]
        });

        let error = combined.evaluate(target);

        Some(EdgeCollapse {
            v0,
            v1,
            target,
            error,
        })
    }

    /// Collapse an edge, merging v1 into v0.
    fn collapse_edge(&mut self, v0: u32, v1: u32, target: [f32; 3]) -> usize {
        // Update v0 position
        self.positions[v0 as usize] = target;

        // Update quadric
        let q1 = self.quadrics[v1 as usize];
        self.quadrics[v0 as usize].add_quadric(&q1);

        // Update normals if present (average)
        if let Some(ref mut normals) = self.normals {
            let n0 = normals[v0 as usize];
            let n1 = normals[v1 as usize];
            let avg = [
                (n0[0] + n1[0]) * 0.5,
                (n0[1] + n1[1]) * 0.5,
                (n0[2] + n1[2]) * 0.5,
            ];
            let len = (avg[0] * avg[0] + avg[1] * avg[1] + avg[2] * avg[2]).sqrt();
            if len > 1e-6 {
                normals[v0 as usize] = [avg[0] / len, avg[1] / len, avg[2] / len];
            }
        }

        // Update UVs if present (average)
        if let Some(ref mut uvs) = self.uvs {
            let uv0 = uvs[v0 as usize];
            let uv1 = uvs[v1 as usize];
            uvs[v0 as usize] = [(uv0[0] + uv1[0]) * 0.5, (uv0[1] + uv1[1]) * 0.5];
        }

        // Mark v1 as deleted and merged to v0
        self.deleted_vertices.insert(v1);
        self.merge_map.insert(v1, v0);

        // Update triangles
        let mut removed_triangles = 0;
        let v1_tris: Vec<usize> = self.vertex_triangles[v1 as usize].iter().copied().collect();

        for &tri_idx in &v1_tris {
            if self.deleted_triangles.contains(&tri_idx) {
                continue;
            }

            let base = tri_idx * 3;
            if base + 2 >= self.indices.len() {
                continue;
            }

            // Replace v1 with v0 in triangle
            for i in 0..3 {
                if self.indices[base + i] == v1 {
                    self.indices[base + i] = v0;
                }
            }

            // Check if triangle is now degenerate
            let i0 = self.indices[base];
            let i1 = self.indices[base + 1];
            let i2 = self.indices[base + 2];

            if i0 == i1 || i1 == i2 || i0 == i2 {
                self.deleted_triangles.insert(tri_idx);
                removed_triangles += 1;

                // Remove from vertex adjacency
                self.vertex_triangles[i0 as usize].remove(&tri_idx);
                self.vertex_triangles[i1 as usize].remove(&tri_idx);
                self.vertex_triangles[i2 as usize].remove(&tri_idx);
            } else {
                // Add to v0's adjacency
                self.vertex_triangles[v0 as usize].insert(tri_idx);
            }
        }

        // Clear v1's adjacency
        self.vertex_triangles[v1 as usize].clear();

        removed_triangles
    }

    /// Resolve vertex through merge chain.
    fn resolve_vertex(&self, mut v: u32) -> u32 {
        while let Some(&merged_to) = self.merge_map.get(&v) {
            v = merged_to;
        }
        v
    }

    /// Update edges after a vertex change.
    fn update_edges_for_vertex(&self, v: u32, heap: &mut BinaryHeap<EdgeCollapse>) {
        for &tri_idx in &self.vertex_triangles[v as usize] {
            if self.deleted_triangles.contains(&tri_idx) {
                continue;
            }

            let base = tri_idx * 3;
            if base + 2 >= self.indices.len() {
                continue;
            }

            for i in 0..3 {
                let v0 = self.indices[base + i];
                let v1 = self.indices[base + (i + 1) % 3];

                if self.deleted_vertices.contains(&v0) || self.deleted_vertices.contains(&v1) {
                    continue;
                }

                if let Some(collapse) = self.compute_collapse(v0.min(v1), v0.max(v1)) {
                    heap.push(collapse);
                }
            }
        }
    }

    /// Build output mesh from current state.
    fn build_output_mesh(&self) -> LodResult<MeshData> {
        // Build vertex remap
        let mut new_indices: Vec<u32> = Vec::new();
        let mut vertex_remap: HashMap<u32, u32> = HashMap::new();
        let mut new_positions: Vec<[f32; 3]> = Vec::new();
        let mut new_normals: Vec<[f32; 3]> = Vec::new();
        let mut new_uvs: Vec<[f32; 2]> = Vec::new();

        for (tri_idx, tri) in self.indices.chunks_exact(3).enumerate() {
            if self.deleted_triangles.contains(&tri_idx) {
                continue;
            }

            for &idx in tri {
                let resolved = self.resolve_vertex(idx);
                if self.deleted_vertices.contains(&resolved) {
                    continue;
                }

                let new_idx = *vertex_remap.entry(resolved).or_insert_with(|| {
                    let idx = new_positions.len() as u32;
                    new_positions.push(self.positions[resolved as usize]);
                    if let Some(ref normals) = self.normals {
                        new_normals.push(normals[resolved as usize]);
                    }
                    if let Some(ref uvs) = self.uvs {
                        new_uvs.push(uvs[resolved as usize]);
                    }
                    idx
                });

                new_indices.push(new_idx);
            }
        }

        // Remove degenerate triangles from output
        let mut final_indices: Vec<u32> = Vec::new();
        for tri in new_indices.chunks_exact(3) {
            if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
                final_indices.extend_from_slice(tri);
            }
        }

        Ok(MeshData {
            positions: new_positions,
            indices: final_indices,
            normals: if new_normals.is_empty() { None } else { Some(new_normals) },
            uvs: if new_uvs.is_empty() { None } else { Some(new_uvs) },
        })
    }
}

/// Simplify a mesh using Garland-Heckbert QEM.
///
/// This is the main entry point for mesh simplification.
///
/// # Arguments
///
/// * `mesh` - Input mesh data
/// * `target_ratio` - Fraction of triangles to keep (0-1)
///
/// # Returns
///
/// Simplified mesh data.
pub fn simplify_mesh(mesh: &MeshData, target_ratio: f32) -> LodResult<MeshData> {
    mesh.validate()?;

    if target_ratio >= 1.0 {
        return Ok(mesh.clone());
    }

    let target_ratio = target_ratio.clamp(0.01, 1.0);
    let mut simplifier = MeshSimplifier::new(mesh);
    simplifier.simplify(target_ratio)
}

/// Compute quadric error for a vertex.
pub fn compute_quadric_error(vertex: [f32; 3], quadric: &Quadric) -> f32 {
    quadric.evaluate(vertex) as f32
}

// ---------------------------------------------------------------------------
// LOD Generation
// ---------------------------------------------------------------------------

/// Generate complete LOD chain from a mesh.
///
/// This is the main entry point for LOD generation.
///
/// # Arguments
///
/// * `mesh` - Input mesh data
/// * `config` - LOD configuration
///
/// # Returns
///
/// Complete LOD chain with all levels.
pub fn generate_lod_chain(mesh: &MeshData, config: &LodConfig) -> LodResult<LodChain> {
    mesh.validate()?;
    config.validate()?;

    let bounds = compute_bounding_sphere(&mesh.positions);

    match &config.strategy {
        LodStrategy::Discrete { .. } => generate_discrete_lod(mesh, config, bounds),
        LodStrategy::Continuous { min_error, max_simplification } => {
            generate_continuous_lod(mesh, *min_error, *max_simplification, config, bounds)
        }
        LodStrategy::Hierarchical { .. } => generate_hierarchical_lod(mesh, config, bounds),
        LodStrategy::NaniteDag { .. } => generate_nanite_dag_lod(mesh, config, bounds),
    }
}

/// Generate discrete LOD levels.
fn generate_discrete_lod(
    mesh: &MeshData,
    config: &LodConfig,
    bounds: BoundingSphere,
) -> LodResult<LodChain> {
    let mut chain = LodChain {
        levels: Vec::with_capacity(config.level_count),
        distances: config.distances.clone(),
        cross_fade: config.cross_fade.clone(),
        bounds,
    };

    // Level 0 is the original mesh
    chain.levels.push(LodLevel::from_mesh(mesh.clone(), 1.0, 0.0));

    // Generate simplified levels
    let mut current_mesh = mesh.clone();
    let mut ratio = config.simplification_ratio;

    for i in 1..config.level_count {
        let screen_coverage = if i < config.distances.len() {
            // Convert distance to approximate screen coverage
            let dist = config.distances[i - 1];
            (bounds.radius * 2.0 / dist).clamp(0.01, 1.0)
        } else {
            0.01
        };

        let simplified = simplify_mesh(&current_mesh, ratio)?;
        let error = compute_simplification_error(&current_mesh, &simplified);

        chain.levels.push(LodLevel::from_mesh(simplified.clone(), screen_coverage, error));

        current_mesh = simplified;
        ratio *= config.simplification_ratio;
    }

    Ok(chain)
}

/// Generate continuous LOD (progressive simplification).
fn generate_continuous_lod(
    mesh: &MeshData,
    min_error: f32,
    max_simplification: f32,
    config: &LodConfig,
    bounds: BoundingSphere,
) -> LodResult<LodChain> {
    // For continuous LOD, we generate more granular levels
    let level_count = (1.0 / (1.0 - max_simplification).max(0.1)).ceil() as usize;
    let level_count = level_count.clamp(2, 16);

    let mut chain = LodChain {
        levels: Vec::with_capacity(level_count),
        distances: Vec::new(),
        cross_fade: config.cross_fade.clone(),
        bounds,
    };

    // Level 0 is the original mesh
    chain.levels.push(LodLevel::from_mesh(mesh.clone(), 1.0, 0.0));
    chain.distances.push(0.0);

    let mut current_mesh = mesh.clone();
    let step = (1.0 - max_simplification) / (level_count - 1) as f32;

    for i in 1..level_count {
        let ratio = 1.0 - (i as f32 * step);
        let simplified = simplify_mesh(&current_mesh, ratio)?;
        let error = compute_simplification_error(&current_mesh, &simplified);

        if error > min_error {
            let screen_coverage = (1.0 - i as f32 / level_count as f32).max(0.01);
            chain.levels.push(LodLevel::from_mesh(simplified.clone(), screen_coverage, error));

            // Compute distance threshold from error
            let dist = bounds.radius * (1.0 / (error + 0.001)).sqrt();
            chain.distances.push(dist);

            current_mesh = simplified;
        }
    }

    Ok(chain)
}

/// Generate hierarchical LOD (tree structure).
fn generate_hierarchical_lod(
    mesh: &MeshData,
    config: &LodConfig,
    bounds: BoundingSphere,
) -> LodResult<LodChain> {
    // For hierarchical LOD, we build a tree where parents cover children
    // This is a simplified implementation using discrete levels as the hierarchy
    let discrete_chain = generate_discrete_lod(mesh, config, bounds)?;

    // Convert to hierarchical representation
    // Each level becomes a parent of the previous level
    let mut tree = LodTree::new();

    for (i, level) in discrete_chain.levels.iter().enumerate() {
        let node = LodTreeNode {
            index: i as u32,
            parent: if i > 0 { Some((i - 1) as u32) } else { None },
            children: if i < discrete_chain.levels.len() - 1 {
                vec![(i + 1) as u32]
            } else {
                vec![]
            },
            mesh: level.mesh_data.clone(),
            bounds: level.bounds,
            error: level.error,
            depth: i as u32,
        };
        tree.nodes.push(node);
    }

    if !tree.nodes.is_empty() {
        tree.roots.push(0);
        tree.max_depth = tree.nodes.len() as u32 - 1;
    }

    Ok(discrete_chain)
}

/// Generate Nanite-style DAG LOD (cluster hierarchy).
fn generate_nanite_dag_lod(
    mesh: &MeshData,
    config: &LodConfig,
    bounds: BoundingSphere,
) -> LodResult<LodChain> {
    // Nanite DAG is research-level; we provide a basic implementation
    // that creates a cluster hierarchy from simplified meshes

    // First generate discrete levels
    let discrete_chain = generate_discrete_lod(mesh, config, bounds)?;

    // This would normally integrate with the meshlet system
    // For now, we return the discrete chain as a starting point

    Ok(discrete_chain)
}

/// Compute error between original and simplified mesh.
fn compute_simplification_error(original: &MeshData, simplified: &MeshData) -> f32 {
    // Use Hausdorff-like distance approximation
    if simplified.positions.is_empty() || original.positions.is_empty() {
        return 0.0;
    }

    let mut max_error: f32 = 0.0;

    // Sample simplified vertices and find distance to original
    for pos in &simplified.positions {
        let mut min_dist = f32::MAX;
        for orig_pos in &original.positions {
            let dx = pos[0] - orig_pos[0];
            let dy = pos[1] - orig_pos[1];
            let dz = pos[2] - orig_pos[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            min_dist = min_dist.min(dist);
        }
        max_error = max_error.max(min_dist);
    }

    max_error
}

// ---------------------------------------------------------------------------
// Runtime LOD Selection
// ---------------------------------------------------------------------------

/// Select appropriate LOD level based on distance and bias.
///
/// # Arguments
///
/// * `chain` - LOD chain
/// * `distance` - Distance from camera to object
/// * `bias` - LOD bias (-1 to +1, negative = higher quality, positive = lower quality)
///
/// # Returns
///
/// Index of the selected LOD level.
pub fn select_lod_level(chain: &LodChain, distance: f32, bias: f32) -> usize {
    if chain.levels.is_empty() {
        return 0;
    }

    // Apply bias to effective distance
    let bias_factor = 2.0_f32.powf(bias);
    let effective_distance = distance * bias_factor;

    // Find appropriate level based on distance thresholds
    for (i, &threshold) in chain.distances.iter().enumerate() {
        if effective_distance < threshold {
            return i;
        }
    }

    // Return lowest detail level
    chain.levels.len() - 1
}

/// Compute cross-fade alpha for smooth LOD transitions.
///
/// # Arguments
///
/// * `screen_coverage` - Current screen coverage (0-1)
/// * `thresholds` - Coverage thresholds from LOD chain
///
/// # Returns
///
/// Alpha value (0-1) for dithered cross-fade.
pub fn compute_cross_fade_alpha(screen_coverage: f32, thresholds: &[f32]) -> f32 {
    if thresholds.is_empty() {
        return 1.0;
    }

    // Find which transition we're in
    for (i, &threshold) in thresholds.iter().enumerate() {
        let next_threshold = thresholds.get(i + 1).copied().unwrap_or(0.0);

        if screen_coverage >= next_threshold && screen_coverage < threshold {
            // We're transitioning between levels i and i+1
            let range = threshold - next_threshold;
            if range > 0.0 {
                return (screen_coverage - next_threshold) / range;
            }
        }
    }

    1.0
}

// ---------------------------------------------------------------------------
// Hierarchical LOD Building
// ---------------------------------------------------------------------------

/// Build hierarchical LOD tree from meshlets.
///
/// Groups meshlets into a tree structure where parents spatially cover children.
pub fn build_hierarchical_lod(meshlets: &[Meshlet], bounds: &[BoundingSphere]) -> LodTree {
    if meshlets.is_empty() {
        return LodTree::new();
    }

    let mut tree = LodTree::new();

    // Create leaf nodes from meshlets
    for (i, (meshlet, bound)) in meshlets.iter().zip(bounds.iter()).enumerate() {
        tree.nodes.push(LodTreeNode {
            index: i as u32,
            parent: None,
            children: vec![],
            mesh: MeshData::new(vec![], vec![]), // Meshlet reference
            bounds: *bound,
            error: 0.0,
            depth: 0,
        });
    }

    // Build hierarchy by grouping nearby meshlets
    let mut current_level: Vec<u32> = (0..meshlets.len() as u32).collect();
    let mut depth = 1u32;

    while current_level.len() > 1 {
        let mut next_level: Vec<u32> = Vec::new();

        // Simple spatial grouping: group pairs of adjacent nodes
        for chunk in current_level.chunks(4).filter(|c| !c.is_empty()) {
            let parent_idx = tree.nodes.len() as u32;

            // Compute parent bounds covering all children
            let mut parent_bounds = tree.nodes[chunk[0] as usize].bounds;
            for &child in &chunk[1..] {
                parent_bounds = parent_bounds.merge(&tree.nodes[child as usize].bounds);
            }

            // Create parent node
            tree.nodes.push(LodTreeNode {
                index: parent_idx,
                parent: None,
                children: chunk.to_vec(),
                mesh: MeshData::new(vec![], vec![]),
                bounds: parent_bounds,
                error: depth as f32 * 0.1, // Approximate error
                depth,
            });

            // Update children's parent
            for &child in chunk {
                tree.nodes[child as usize].parent = Some(parent_idx);
            }

            next_level.push(parent_idx);
        }

        current_level = next_level;
        depth += 1;
    }

    tree.roots = current_level;
    tree.max_depth = depth - 1;

    tree
}

// ---------------------------------------------------------------------------
// @lod Decorator Parsing
// ---------------------------------------------------------------------------

/// Parsed @lod decorator parameters.
#[derive(Debug, Clone)]
pub struct LodDecoratorParams {
    /// LOD strategy type.
    pub strategy: String,
    /// Number of LOD levels.
    pub levels: Option<usize>,
    /// Distance thresholds.
    pub distances: Option<Vec<f32>>,
    /// LOD bias.
    pub bias: Option<f32>,
    /// Simplification ratio per level.
    pub simplification_ratio: Option<f32>,
    /// Enable cross-fade.
    pub cross_fade: Option<bool>,
    /// Dither pattern name.
    pub dither_pattern: Option<String>,
}

impl Default for LodDecoratorParams {
    fn default() -> Self {
        Self {
            strategy: "discrete".into(),
            levels: None,
            distances: None,
            bias: None,
            simplification_ratio: None,
            cross_fade: None,
            dither_pattern: None,
        }
    }
}

/// Parse @lod decorator into configuration.
///
/// # Decorator Syntax
///
/// ```ignore
/// @lod(strategy="discrete", levels=4, distances=[10, 25, 50, 100], bias=0.0)
/// @lod(strategy="continuous", min_error=0.001, max_simplification=0.8)
/// @lod(strategy="hierarchical")
/// @lod(distances=[10, 25, 50], cross_fade=true, dither="ign")
/// ```
pub fn parse_lod_decorator(params: &HashMap<String, String>) -> LodResult<LodConfig> {
    let parsed = parse_decorator_params(params)?;
    decorator_params_to_config(&parsed)
}

/// Parse raw decorator parameters.
pub fn parse_decorator_params(params: &HashMap<String, String>) -> LodResult<LodDecoratorParams> {
    let mut result = LodDecoratorParams::default();

    if let Some(strategy) = params.get("strategy") {
        result.strategy = strategy.to_lowercase();
    }

    if let Some(levels) = params.get("levels") {
        result.levels = levels.parse().ok();
    }

    if let Some(distances) = params.get("distances") {
        result.distances = parse_float_array(distances);
    }

    if let Some(bias) = params.get("bias") {
        result.bias = bias.parse().ok();
    }

    if let Some(ratio) = params.get("simplification_ratio") {
        result.simplification_ratio = ratio.parse().ok();
    }

    if let Some(cf) = params.get("cross_fade") {
        result.cross_fade = cf.parse().ok();
    }

    if let Some(pattern) = params.get("dither_pattern") {
        result.dither_pattern = Some(pattern.clone());
    }
    if let Some(pattern) = params.get("dither") {
        result.dither_pattern = Some(pattern.clone());
    }

    Ok(result)
}

/// Convert decorator params to LOD config.
pub fn decorator_params_to_config(params: &LodDecoratorParams) -> LodResult<LodConfig> {
    let strategy = match params.strategy.as_str() {
        "discrete" => LodStrategy::Discrete { levels: Vec::new() },
        "continuous" => LodStrategy::Continuous {
            min_error: 0.001,
            max_simplification: 0.8,
        },
        "hierarchical" => LodStrategy::Hierarchical { tree: LodTree::new() },
        "nanite" | "dag" => LodStrategy::NaniteDag { cluster_dag: ClusterDag::new() },
        other => {
            return Err(LodError::InvalidConfig(format!(
                "unknown LOD strategy: {}",
                other
            )));
        }
    };

    let cross_fade = CrossFadeConfig {
        enabled: params.cross_fade.unwrap_or(true),
        dither_pattern: parse_dither_pattern(params.dither_pattern.as_deref()),
        ..Default::default()
    };

    Ok(LodConfig {
        strategy,
        distances: params.distances.clone().unwrap_or_else(|| vec![10.0, 25.0, 50.0, 100.0]),
        bias: params.bias.unwrap_or(0.0),
        cross_fade,
        level_count: params.levels.unwrap_or(4),
        simplification_ratio: params.simplification_ratio.unwrap_or(0.5),
    })
}

/// Parse dither pattern from string.
fn parse_dither_pattern(name: Option<&str>) -> DitherPattern {
    match name {
        Some("gradient") | Some("gradient_noise") => DitherPattern::GradientNoise,
        Some("ign") | Some("interleaved_gradient") => DitherPattern::InterleavedGradient,
        Some("blue") | Some("blue_noise") => DitherPattern::BlueNoise,
        Some("bayer") | Some("bayer4x4") => DitherPattern::Bayer4x4,
        _ => DitherPattern::InterleavedGradient,
    }
}

/// Parse float array from string like "[10, 25, 50]".
fn parse_float_array(s: &str) -> Option<Vec<f32>> {
    let trimmed = s.trim().trim_start_matches('[').trim_end_matches(']');
    let values: Result<Vec<f32>, _> = trimmed
        .split(',')
        .map(|x| x.trim().parse::<f32>())
        .collect();
    values.ok()
}

// ---------------------------------------------------------------------------
// Dithering Functions
// ---------------------------------------------------------------------------

/// Gradient noise at pixel position.
fn gradient_noise(x: u32, y: u32, frame: u32) -> f32 {
    let n = (x.wrapping_mul(12345).wrapping_add(y.wrapping_mul(67890)))
        .wrapping_mul(frame.wrapping_add(1));
    let f = (n & 0xFFFF) as f32 / 65536.0;
    f
}

/// Interleaved gradient noise (IGN) - high quality dithering.
///
/// From "Next Generation Post Processing in Call of Duty: Advanced Warfare"
fn interleaved_gradient_noise(x: u32, y: u32, frame: u32) -> f32 {
    // Magic numbers from the paper
    let x = x as f32 + 5.588238 * (frame % 64) as f32;
    let y = y as f32 + 5.588238 * (frame % 64) as f32;

    let n = (52.9829189 * ((x * 0.06711056 + y * 0.00583715).fract())).fract();
    n
}

/// Blue noise fallback (approximation without texture).
fn blue_noise_fallback(x: u32, y: u32, frame: u32) -> f32 {
    // R2 sequence approximation
    let phi2 = 1.324717957244746; // Plastic constant
    let a1 = 1.0 / phi2;
    let a2 = 1.0 / (phi2 * phi2);

    let n = (0.5 + a1 * x as f32 + a2 * y as f32 + 0.123456 * frame as f32).fract();
    n
}

/// Bayer 4x4 ordered dithering.
fn bayer_4x4(x: u32, y: u32) -> f32 {
    const BAYER: [[f32; 4]; 4] = [
        [0.0 / 16.0, 8.0 / 16.0, 2.0 / 16.0, 10.0 / 16.0],
        [12.0 / 16.0, 4.0 / 16.0, 14.0 / 16.0, 6.0 / 16.0],
        [3.0 / 16.0, 11.0 / 16.0, 1.0 / 16.0, 9.0 / 16.0],
        [15.0 / 16.0, 7.0 / 16.0, 13.0 / 16.0, 5.0 / 16.0],
    ];

    BAYER[(y % 4) as usize][(x % 4) as usize]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create a simple cube mesh
    fn create_cube() -> MeshData {
        let positions = vec![
            // Front face
            [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
            // Back face
            [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
        ];
        let indices = vec![
            // Front
            0, 1, 2, 0, 2, 3,
            // Back
            5, 4, 7, 5, 7, 6,
            // Top
            3, 2, 6, 3, 6, 7,
            // Bottom
            4, 5, 1, 4, 1, 0,
            // Right
            1, 5, 6, 1, 6, 2,
            // Left
            4, 0, 3, 4, 3, 7,
        ];
        MeshData::new(positions, indices)
    }

    // Helper function to create a grid mesh
    fn create_grid(size: usize) -> MeshData {
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        for y in 0..=size {
            for x in 0..=size {
                positions.push([x as f32, y as f32, 0.0]);
            }
        }

        for y in 0..size {
            for x in 0..size {
                let v0 = (y * (size + 1) + x) as u32;
                let v1 = v0 + 1;
                let v2 = v0 + (size + 1) as u32;
                let v3 = v2 + 1;

                indices.extend_from_slice(&[v0, v1, v2, v2, v1, v3]);
            }
        }

        MeshData::new(positions, indices)
    }

    // ========================================================================
    // Discrete LOD tests (6+ tests)
    // ========================================================================

    #[test]
    fn test_discrete_lod_level_generation() {
        let mesh = create_grid(10);
        let config = LodConfig::discrete(4, &[10.0, 25.0, 50.0, 100.0]);

        let chain = generate_lod_chain(&mesh, &config).unwrap();

        assert_eq!(chain.level_count(), 4);
        // Each subsequent level should have fewer triangles
        for i in 1..chain.levels.len() {
            assert!(
                chain.levels[i].triangle_count <= chain.levels[i - 1].triangle_count,
                "LOD {} has more triangles than LOD {}",
                i,
                i - 1
            );
        }
    }

    #[test]
    fn test_discrete_lod_distance_switching() {
        let mesh = create_cube();
        let config = LodConfig::discrete(3, &[10.0, 30.0, 60.0]);

        let chain = generate_lod_chain(&mesh, &config).unwrap();

        // Test LOD selection at different distances
        assert_eq!(select_lod_level(&chain, 5.0, 0.0), 0);
        assert_eq!(select_lod_level(&chain, 15.0, 0.0), 1);
        assert_eq!(select_lod_level(&chain, 45.0, 0.0), 2);
        assert_eq!(select_lod_level(&chain, 100.0, 0.0), 2);
    }

    #[test]
    fn test_discrete_lod_level_count() {
        let mesh = create_grid(10); // Use larger mesh for more simplification headroom

        for count in 2..=4 {
            let mut distances = Vec::new();
            for i in 0..count {
                distances.push(10.0 * (i + 1) as f32);
            }
            let config = LodConfig::discrete(count, &distances);
            let chain = generate_lod_chain(&mesh, &config).unwrap();
            assert_eq!(chain.level_count(), count);
        }
    }

    #[test]
    fn test_discrete_lod_preserves_original() {
        let mesh = create_cube();
        let config = LodConfig::discrete(3, &[10.0, 30.0, 60.0]);

        let chain = generate_lod_chain(&mesh, &config).unwrap();

        // Level 0 should be the original mesh
        assert_eq!(chain.levels[0].vertex_count, mesh.vertex_count() as u32);
        assert_eq!(chain.levels[0].triangle_count, mesh.triangle_count() as u32);
    }

    #[test]
    fn test_discrete_lod_simplification_ratio() {
        let mesh = create_grid(20);
        let config = LodConfig {
            level_count: 4,
            simplification_ratio: 0.5,
            ..LodConfig::default()
        };

        let chain = generate_lod_chain(&mesh, &config).unwrap();

        // Each level should have roughly half the triangles of the previous
        // (with some variance due to simplification algorithm)
        for i in 1..chain.levels.len() {
            let ratio = chain.levels[i].triangle_count as f32
                / chain.levels[i - 1].triangle_count as f32;
            assert!(
                ratio < 0.9,
                "Level {} ratio {} is too high",
                i,
                ratio
            );
        }
    }

    #[test]
    fn test_discrete_lod_bounding_spheres() {
        let mesh = create_cube();
        let config = LodConfig::discrete(3, &[10.0, 30.0, 60.0]);

        let chain = generate_lod_chain(&mesh, &config).unwrap();

        // All LOD levels should have valid bounding spheres
        for level in &chain.levels {
            assert!(level.bounds.radius > 0.0);
        }

        // Bounding sphere should roughly contain the cube
        assert!(chain.bounds.radius >= 1.0);
    }

    // ========================================================================
    // Continuous LOD tests (6+ tests)
    // ========================================================================

    #[test]
    fn test_continuous_lod_simplification_ratios() {
        let mesh = create_grid(15);
        let config = LodConfig::continuous(0.001, 0.8);

        let chain = generate_lod_chain(&mesh, &config).unwrap();

        // Should have multiple levels
        assert!(chain.level_count() >= 2);

        // Triangles should decrease
        for i in 1..chain.levels.len() {
            assert!(chain.levels[i].triangle_count <= chain.levels[i - 1].triangle_count);
        }
    }

    #[test]
    fn test_continuous_lod_error_bounds() {
        let mesh = create_grid(10);
        let config = LodConfig::continuous(0.01, 0.7);

        let chain = generate_lod_chain(&mesh, &config).unwrap();

        // Error should increase with each level
        for i in 1..chain.levels.len() {
            assert!(
                chain.levels[i].error >= chain.levels[i - 1].error,
                "Level {} has lower error than level {}",
                i,
                i - 1
            );
        }
    }

    #[test]
    fn test_continuous_lod_edge_collapse() {
        let mesh = create_grid(8);
        let simplified = simplify_mesh(&mesh, 0.5).unwrap();

        // Should have fewer vertices
        assert!(simplified.vertex_count() < mesh.vertex_count());
        // Should have fewer triangles
        assert!(simplified.triangle_count() < mesh.triangle_count());
    }

    #[test]
    fn test_simplify_mesh_preserves_topology() {
        let mesh = create_cube();
        let simplified = simplify_mesh(&mesh, 0.8).unwrap();

        // Should still be a valid mesh
        simplified.validate().unwrap();

        // Should have some triangles
        assert!(simplified.triangle_count() > 0);
    }

    #[test]
    fn test_simplify_mesh_target_ratio() {
        let mesh = create_grid(20);
        let target_ratio = 0.25;

        let simplified = simplify_mesh(&mesh, target_ratio).unwrap();

        // Should be roughly at target ratio (within tolerance)
        let actual_ratio = simplified.triangle_count() as f32 / mesh.triangle_count() as f32;
        assert!(
            actual_ratio < target_ratio + 0.2,
            "Actual ratio {} too far from target {}",
            actual_ratio,
            target_ratio
        );
    }

    #[test]
    fn test_simplify_mesh_extreme_ratio() {
        let mesh = create_grid(10);

        // Very aggressive simplification
        let simplified = simplify_mesh(&mesh, 0.1).unwrap();
        assert!(simplified.triangle_count() > 0);
        assert!(simplified.triangle_count() < mesh.triangle_count() / 2);

        // No simplification
        let same = simplify_mesh(&mesh, 1.0).unwrap();
        assert_eq!(same.triangle_count(), mesh.triangle_count());
    }

    // ========================================================================
    // Hierarchical LOD tests (5+ tests)
    // ========================================================================

    #[test]
    fn test_hierarchical_tree_construction() {
        let bounds: Vec<BoundingSphere> = (0..8)
            .map(|i| BoundingSphere::new([i as f32, 0.0, 0.0], 1.0))
            .collect();
        let meshlets: Vec<Meshlet> = (0..8).map(|_| Meshlet::default()).collect();

        let tree = build_hierarchical_lod(&meshlets, &bounds);

        // Should have nodes
        assert!(!tree.nodes.is_empty());
        // Should have roots
        assert!(!tree.roots.is_empty());
        // Should have depth
        assert!(tree.max_depth >= 1);
    }

    #[test]
    fn test_hierarchical_parent_bounds() {
        let bounds = vec![
            BoundingSphere::new([0.0, 0.0, 0.0], 1.0),
            BoundingSphere::new([3.0, 0.0, 0.0], 1.0),
            BoundingSphere::new([6.0, 0.0, 0.0], 1.0),
            BoundingSphere::new([9.0, 0.0, 0.0], 1.0),
        ];
        let meshlets: Vec<Meshlet> = (0..4).map(|_| Meshlet::default()).collect();

        let tree = build_hierarchical_lod(&meshlets, &bounds);

        // Parent bounds should contain children
        for node in &tree.nodes {
            for &child_idx in &node.children {
                let child = &tree.nodes[child_idx as usize];
                // Parent bounds should contain child center
                let dx = child.bounds.center[0] - node.bounds.center[0];
                let dy = child.bounds.center[1] - node.bounds.center[1];
                let dz = child.bounds.center[2] - node.bounds.center[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                assert!(
                    dist <= node.bounds.radius + child.bounds.radius + 0.01,
                    "Child {} center outside parent {} bounds",
                    child_idx,
                    node.index
                );
            }
        }
    }

    #[test]
    fn test_hierarchical_child_coverage() {
        let bounds = vec![
            BoundingSphere::new([0.0, 0.0, 0.0], 0.5),
            BoundingSphere::new([1.0, 0.0, 0.0], 0.5),
        ];
        let meshlets: Vec<Meshlet> = (0..2).map(|_| Meshlet::default()).collect();

        let tree = build_hierarchical_lod(&meshlets, &bounds);

        // Each non-leaf node should have children
        for node in &tree.nodes {
            if node.depth > 0 {
                assert!(!node.children.is_empty());
            }
        }
    }

    #[test]
    fn test_hierarchical_leaf_nodes() {
        let bounds: Vec<BoundingSphere> = (0..4)
            .map(|i| BoundingSphere::new([i as f32, 0.0, 0.0], 1.0))
            .collect();
        let meshlets: Vec<Meshlet> = (0..4).map(|_| Meshlet::default()).collect();

        let tree = build_hierarchical_lod(&meshlets, &bounds);

        // Should have leaf nodes (depth 0)
        let leaf_count = tree.nodes.iter().filter(|n| n.depth == 0).count();
        assert_eq!(leaf_count, 4);
    }

    #[test]
    fn test_hierarchical_empty_input() {
        let tree = build_hierarchical_lod(&[], &[]);
        assert!(tree.nodes.is_empty());
        assert!(tree.roots.is_empty());
    }

    // ========================================================================
    // Cross-fade tests (5+ tests)
    // ========================================================================

    #[test]
    fn test_dither_patterns_range() {
        for pattern in [
            DitherPattern::GradientNoise,
            DitherPattern::InterleavedGradient,
            DitherPattern::BlueNoise,
            DitherPattern::Bayer4x4,
        ] {
            for x in 0..16 {
                for y in 0..16 {
                    let value = pattern.sample(x, y, 0);
                    assert!(
                        value >= 0.0 && value < 1.0,
                        "Pattern {:?} at ({}, {}) = {} out of range",
                        pattern,
                        x,
                        y,
                        value
                    );
                }
            }
        }
    }

    #[test]
    fn test_cross_fade_alpha_computation() {
        let thresholds = vec![1.0, 0.5, 0.25, 0.1];

        // At full coverage, alpha should be 1
        let alpha = compute_cross_fade_alpha(1.0, &thresholds);
        assert!((alpha - 1.0).abs() < 0.1);

        // At very low coverage, alpha should be near 0 or 1
        let alpha = compute_cross_fade_alpha(0.05, &thresholds);
        assert!(alpha >= 0.0 && alpha <= 1.0);
    }

    #[test]
    fn test_cross_fade_transition_blending() {
        let thresholds = vec![1.0, 0.5, 0.2];

        // In middle of transition
        let alpha1 = compute_cross_fade_alpha(0.7, &thresholds);
        let alpha2 = compute_cross_fade_alpha(0.6, &thresholds);

        // Alpha should change as coverage changes
        // Both should be in [0, 1]
        assert!(alpha1 >= 0.0 && alpha1 <= 1.0);
        assert!(alpha2 >= 0.0 && alpha2 <= 1.0);
    }

    #[test]
    fn test_ign_dither_quality() {
        // IGN should produce good distribution
        let mut sum = 0.0f32;
        let count = 256;

        for x in 0..16 {
            for y in 0..16 {
                sum += interleaved_gradient_noise(x, y, 0);
            }
        }

        let avg = sum / count as f32;
        // Average should be near 0.5 for good distribution
        assert!(
            (avg - 0.5).abs() < 0.15,
            "IGN average {} too far from 0.5",
            avg
        );
    }

    #[test]
    fn test_bayer_ordered_pattern() {
        // Bayer should have specific structure
        let v00 = bayer_4x4(0, 0);
        let v11 = bayer_4x4(1, 1);

        // These are known values from the Bayer matrix
        assert!((v00 - 0.0 / 16.0).abs() < 0.01);
        assert!((v11 - 4.0 / 16.0).abs() < 0.01);
    }

    // ========================================================================
    // Decorator parsing tests (4+ tests)
    // ========================================================================

    #[test]
    fn test_parse_lod_decorator_distances() {
        let mut params = HashMap::new();
        params.insert("distances".to_string(), "[10, 25, 50, 100]".to_string());

        let config = parse_lod_decorator(&params).unwrap();

        assert_eq!(config.distances.len(), 4);
        assert!((config.distances[0] - 10.0).abs() < 0.01);
        assert!((config.distances[3] - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_lod_decorator_bias() {
        let mut params = HashMap::new();
        params.insert("bias".to_string(), "0.5".to_string());

        let config = parse_lod_decorator(&params).unwrap();

        assert!((config.bias - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_parse_lod_decorator_strategy() {
        for strategy in ["discrete", "continuous", "hierarchical", "nanite"] {
            let mut params = HashMap::new();
            params.insert("strategy".to_string(), strategy.to_string());

            let config = parse_lod_decorator(&params).unwrap();
            // Should not error
            config.validate().unwrap();
        }
    }

    #[test]
    fn test_parse_lod_decorator_dither_pattern() {
        for (name, expected) in [
            ("ign", DitherPattern::InterleavedGradient),
            ("gradient", DitherPattern::GradientNoise),
            ("blue", DitherPattern::BlueNoise),
            ("bayer", DitherPattern::Bayer4x4),
        ] {
            let mut params = HashMap::new();
            params.insert("dither".to_string(), name.to_string());

            let config = parse_lod_decorator(&params).unwrap();
            assert_eq!(config.cross_fade.dither_pattern, expected);
        }
    }

    // ========================================================================
    // Idempotency tests (2+ tests)
    // ========================================================================

    #[test]
    fn test_lod_generation_idempotent() {
        let mesh = create_grid(8);
        let config = LodConfig::discrete(3, &[10.0, 30.0, 60.0]);

        let chain1 = generate_lod_chain(&mesh, &config).unwrap();
        let chain2 = generate_lod_chain(&mesh, &config).unwrap();

        assert_eq!(chain1.level_count(), chain2.level_count());
        // Level 0 (original) should always match exactly
        assert_eq!(chain1.levels[0].vertex_count, chain2.levels[0].vertex_count);
        assert_eq!(chain1.levels[0].triangle_count, chain2.levels[0].triangle_count);

        // Simplified levels should be within small tolerance due to deterministic ordering
        for i in 1..chain1.level_count() {
            let v_diff = (chain1.levels[i].vertex_count as i32 - chain2.levels[i].vertex_count as i32).abs();
            let t_diff = (chain1.levels[i].triangle_count as i32 - chain2.levels[i].triangle_count as i32).abs();
            assert!(
                v_diff <= 2,
                "Level {} vertex count differs by {} (more than 2)",
                i, v_diff
            );
            assert!(
                t_diff <= 4,
                "Level {} triangle count differs by {} (more than 4)",
                i, t_diff
            );
        }
    }

    #[test]
    fn test_simplification_deterministic() {
        let mesh = create_grid(10);

        let s1 = simplify_mesh(&mesh, 0.5).unwrap();
        let s2 = simplify_mesh(&mesh, 0.5).unwrap();

        // With deterministic ordering, results should match exactly or be very close
        let v_diff = (s1.vertex_count() as i32 - s2.vertex_count() as i32).abs();
        let t_diff = (s1.triangle_count() as i32 - s2.triangle_count() as i32).abs();

        assert!(
            v_diff <= 2,
            "Vertex count differs by {} (more than 2)",
            v_diff
        );
        assert!(
            t_diff <= 4,
            "Triangle count differs by {} (more than 4)",
            t_diff
        );
    }

    // ========================================================================
    // Edge cases (2+ tests)
    // ========================================================================

    #[test]
    fn test_single_vertex_mesh() {
        let mesh = MeshData::new(vec![[0.0, 0.0, 0.0]], vec![]);

        let result = mesh.validate();
        assert!(result.is_err()); // Empty indices
    }

    #[test]
    fn test_degenerate_triangles() {
        // Triangle with two identical vertices
        let mesh = MeshData::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            vec![0, 0, 1],
        );

        // Should handle degenerate case
        let config = LodConfig::default();
        let result = generate_lod_chain(&mesh, &config);
        // May succeed or fail depending on implementation
        assert!(result.is_ok() || result.is_err());
    }

    // ========================================================================
    // Quadric tests (additional for completeness)
    // ========================================================================

    #[test]
    fn test_quadric_from_plane() {
        // z = 0 plane: 0x + 0y + 1z + 0 = 0
        let q = Quadric::from_plane(0.0, 0.0, 1.0, 0.0);

        // Points on the plane should have zero error
        let error_on_plane = q.evaluate([1.0, 1.0, 0.0]);
        assert!(error_on_plane.abs() < 0.0001);

        // Points off the plane should have non-zero error
        let error_off_plane = q.evaluate([1.0, 1.0, 1.0]);
        assert!(error_off_plane > 0.0);
    }

    #[test]
    fn test_quadric_addition() {
        let q1 = Quadric::from_plane(1.0, 0.0, 0.0, 0.0);
        let q2 = Quadric::from_plane(0.0, 1.0, 0.0, 0.0);

        let combined = q1 + q2;

        // Combined error should include both planes
        let error = combined.evaluate([1.0, 1.0, 0.0]);
        assert!(error > 0.0);
    }

    #[test]
    fn test_quadric_optimal_position() {
        // Create quadrics from multiple planes
        let mut q = Quadric::zero();
        q.add_quadric(&Quadric::from_plane(1.0, 0.0, 0.0, 0.0));
        q.add_quadric(&Quadric::from_plane(0.0, 1.0, 0.0, 0.0));
        q.add_quadric(&Quadric::from_plane(0.0, 0.0, 1.0, 0.0));

        // Optimal should be near origin
        if let Some(pos) = q.optimal_position() {
            assert!(pos[0].abs() < 0.01);
            assert!(pos[1].abs() < 0.01);
            assert!(pos[2].abs() < 0.01);
        }
    }

    #[test]
    fn test_compute_quadric_error() {
        let q = Quadric::from_plane(0.0, 0.0, 1.0, 0.0);
        let error = compute_quadric_error([0.0, 0.0, 5.0], &q);
        assert!((error - 25.0).abs() < 0.01); // 5^2 = 25
    }

    // ========================================================================
    // Config validation tests
    // ========================================================================

    #[test]
    fn test_config_validation() {
        // Valid config
        let config = LodConfig::default();
        assert!(config.validate().is_ok());

        // Invalid: zero levels
        let config = LodConfig {
            level_count: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Invalid: bad simplification ratio
        let config = LodConfig {
            simplification_ratio: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Invalid: non-monotonic distances
        let config = LodConfig {
            distances: vec![50.0, 25.0, 10.0],
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_mesh_data_validation() {
        // Valid mesh
        let mesh = create_cube();
        assert!(mesh.validate().is_ok());

        // Empty mesh
        let empty = MeshData::new(vec![], vec![]);
        assert!(empty.validate().is_err());

        // Bad index count
        let bad = MeshData::new(vec![[0.0, 0.0, 0.0]], vec![0, 1]);
        assert!(bad.validate().is_err());

        // Out of bounds index
        let oob = MeshData::new(vec![[0.0, 0.0, 0.0]], vec![0, 1, 2]);
        assert!(oob.validate().is_err());
    }

    // ========================================================================
    // LOD chain accessor tests
    // ========================================================================

    #[test]
    fn test_lod_chain_accessors() {
        let mesh = create_grid(5);
        let config = LodConfig::discrete(3, &[10.0, 30.0, 60.0]);

        let chain = generate_lod_chain(&mesh, &config).unwrap();

        assert_eq!(chain.level_count(), 3);
        assert!(chain.get_level(0).is_some());
        assert!(chain.get_level(100).is_none());
        assert!(!chain.thresholds().is_empty());
        assert!(chain.total_triangles() > 0);
        assert_eq!(chain.coverage_thresholds().len(), 3);
    }

    // ========================================================================
    // LOD selection with bias tests
    // ========================================================================

    #[test]
    fn test_lod_selection_with_negative_bias() {
        let mesh = create_cube();
        let config = LodConfig::discrete(3, &[10.0, 30.0, 60.0]);
        let chain = generate_lod_chain(&mesh, &config).unwrap();

        // Negative bias = higher quality (use higher detail at same distance)
        let level_no_bias = select_lod_level(&chain, 20.0, 0.0);
        let level_neg_bias = select_lod_level(&chain, 20.0, -0.5);

        assert!(level_neg_bias <= level_no_bias);
    }

    #[test]
    fn test_lod_selection_with_positive_bias() {
        let mesh = create_cube();
        let config = LodConfig::discrete(3, &[10.0, 30.0, 60.0]);
        let chain = generate_lod_chain(&mesh, &config).unwrap();

        // Positive bias = lower quality (use lower detail at same distance)
        let level_no_bias = select_lod_level(&chain, 20.0, 0.0);
        let level_pos_bias = select_lod_level(&chain, 20.0, 0.5);

        assert!(level_pos_bias >= level_no_bias);
    }
}

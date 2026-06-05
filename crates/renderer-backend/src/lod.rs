//! LOD (Level of Detail) generation and blending
//!
//! This module implements discrete LOD generation using mesh simplification
//! (Quadric Error Metrics) and LOD blending for smooth transitions.
//!
//! # Features
//!
//! - N discrete LOD levels via QEM mesh simplification
//! - Multiple blending modes: discrete, alpha cross-fade, dither pattern
//! - Per-viewport LOD bias for editor/VR tuning
//! - Screen-space threshold-based LOD selection
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::lod::{LodBuilder, LodBlendMode, LodSelector};
//!
//! // Generate 3 LOD levels with 50% reduction per level
//! let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
//! let indices = vec![0, 1, 2, 1, 3, 2];
//!
//! let chain = LodBuilder::new(positions, indices)
//!     .with_levels(3)
//!     .with_reduction(0.5)
//!     .with_blend_mode(LodBlendMode::AlphaCrossfade { range: 5.0 })
//!     .build()
//!     .unwrap();
//!
//! // Select LOD based on screen-space size
//! let selector = LodSelector::new(&[100.0, 50.0, 25.0]).with_bias(-0.5);
//! let selection = selector.select(75.0, chain.blend_mode);
//! ```

use std::collections::BinaryHeap;
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default number of LOD levels to generate.
pub const DEFAULT_LOD_LEVELS: u32 = 3;

/// Default vertex reduction ratio per LOD level.
pub const DEFAULT_REDUCTION_RATIO: f32 = 0.5;

/// Minimum vertices to keep during simplification.
pub const MIN_VERTICES: usize = 4;

/// Minimum triangles to keep during simplification.
pub const MIN_TRIANGLES: usize = 1;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single LOD level containing simplified mesh data.
#[derive(Debug, Clone)]
pub struct LodLevel {
    /// LOD index (0 = highest detail).
    pub level: u32,
    /// Simplified vertex positions.
    pub positions: Vec<[f32; 3]>,
    /// Triangle indices into the position buffer.
    pub indices: Vec<u32>,
    /// Screen-space threshold for this LOD (pixels).
    /// Objects smaller than this threshold use this LOD.
    pub threshold: f32,
    /// Vertex reduction ratio from the previous level.
    /// 1.0 for LOD 0 (no reduction).
    pub reduction_ratio: f32,
}

impl LodLevel {
    /// Number of triangles in this LOD level.
    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Number of vertices in this LOD level.
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }
}

/// Complete LOD chain for a mesh.
#[derive(Debug, Clone)]
pub struct LodChain {
    /// LOD levels from highest detail (0) to lowest.
    pub levels: Vec<LodLevel>,
    /// Blending mode for transitions between LOD levels.
    pub blend_mode: LodBlendMode,
}

impl LodChain {
    /// Get the number of LOD levels.
    #[inline]
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    /// Get a specific LOD level by index.
    pub fn get_level(&self, index: u32) -> Option<&LodLevel> {
        self.levels.get(index as usize)
    }

    /// Get the highest detail LOD (level 0).
    pub fn highest_detail(&self) -> Option<&LodLevel> {
        self.levels.first()
    }

    /// Get the lowest detail LOD.
    pub fn lowest_detail(&self) -> Option<&LodLevel> {
        self.levels.last()
    }
}

/// Blending mode for LOD transitions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LodBlendMode {
    /// Hard switch between LOD levels (no blending).
    Discrete,
    /// Alpha cross-fade over a screen-space range.
    /// The `range` parameter specifies the pixel range over which to blend.
    AlphaCrossfade { range: f32 },
    /// Dither pattern transition using screen-space dithering.
    /// The `pattern_size` specifies the dither pattern dimension (e.g., 4 for 4x4).
    Dither { pattern_size: u32 },
}

impl Default for LodBlendMode {
    fn default() -> Self {
        Self::Discrete
    }
}

/// LOD selection result.
#[derive(Debug, Clone, Copy)]
pub struct LodSelection {
    /// Primary LOD level to render.
    pub primary_lod: u32,
    /// Secondary LOD level for blending (if applicable).
    pub secondary_lod: Option<u32>,
    /// Blend factor: 0.0 = fully primary, 1.0 = fully secondary.
    pub blend_factor: f32,
}

impl LodSelection {
    /// Create a selection for a single LOD with no blending.
    pub fn single(lod: u32) -> Self {
        Self {
            primary_lod: lod,
            secondary_lod: None,
            blend_factor: 0.0,
        }
    }

    /// Create a selection with blending between two LODs.
    pub fn blended(primary: u32, secondary: u32, factor: f32) -> Self {
        Self {
            primary_lod: primary,
            secondary_lod: Some(secondary),
            blend_factor: factor.clamp(0.0, 1.0),
        }
    }

    /// Check if this selection requires blending.
    #[inline]
    pub fn needs_blending(&self) -> bool {
        self.secondary_lod.is_some() && self.blend_factor > 0.0 && self.blend_factor < 1.0
    }
}

/// LOD generation errors.
#[derive(Debug)]
pub enum LodError {
    /// Invalid index count (not a multiple of 3).
    InvalidIndexCount(usize),
    /// Index out of bounds.
    IndexOutOfBounds { index: u32, vertex_count: usize },
    /// Insufficient geometry for simplification.
    InsufficientGeometry { vertices: usize, triangles: usize },
    /// Invalid reduction ratio.
    InvalidReductionRatio(f32),
    /// Invalid level count.
    InvalidLevelCount(u32),
}

impl std::fmt::Display for LodError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidIndexCount(count) => {
                write!(f, "index count {} is not a multiple of 3", count)
            }
            Self::IndexOutOfBounds { index, vertex_count } => {
                write!(
                    f,
                    "index {} out of bounds for {} vertices",
                    index, vertex_count
                )
            }
            Self::InsufficientGeometry { vertices, triangles } => {
                write!(
                    f,
                    "insufficient geometry: {} vertices, {} triangles",
                    vertices, triangles
                )
            }
            Self::InvalidReductionRatio(ratio) => {
                write!(f, "invalid reduction ratio: {} (must be 0.0 < r < 1.0)", ratio)
            }
            Self::InvalidLevelCount(count) => {
                write!(f, "invalid level count: {} (must be >= 1)", count)
            }
        }
    }
}

impl std::error::Error for LodError {}

// ---------------------------------------------------------------------------
// LodBuilder
// ---------------------------------------------------------------------------

/// Builder for generating LOD chains from mesh data.
///
/// Uses Quadric Error Metrics (QEM) for mesh simplification.
pub struct LodBuilder {
    /// Source vertex positions.
    positions: Vec<[f32; 3]>,
    /// Source triangle indices.
    indices: Vec<u32>,
    /// Number of LOD levels to generate.
    target_levels: u32,
    /// Vertex reduction ratio per level (0.0 to 1.0).
    reduction_per_level: f32,
    /// Screen-space thresholds for LOD selection.
    thresholds: Option<Vec<f32>>,
    /// Blending mode for transitions.
    blend_mode: LodBlendMode,
}

impl LodBuilder {
    /// Create a new LOD builder from vertex positions and indices.
    ///
    /// # Arguments
    ///
    /// * `positions` - Vertex positions (x, y, z).
    /// * `indices` - Triangle indices, must be a multiple of 3.
    pub fn new(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> Self {
        Self {
            positions,
            indices,
            target_levels: DEFAULT_LOD_LEVELS,
            reduction_per_level: DEFAULT_REDUCTION_RATIO,
            thresholds: None,
            blend_mode: LodBlendMode::Discrete,
        }
    }

    /// Set the number of LOD levels to generate.
    ///
    /// Level 0 is always the original mesh. Additional levels are simplified.
    pub fn with_levels(mut self, count: u32) -> Self {
        self.target_levels = count.max(1);
        self
    }

    /// Set the vertex reduction ratio per level.
    ///
    /// A ratio of 0.5 means each level has ~50% of the previous level's vertices.
    pub fn with_reduction(mut self, ratio: f32) -> Self {
        self.reduction_per_level = ratio;
        self
    }

    /// Set explicit screen-space thresholds for each LOD level.
    ///
    /// Thresholds should be in descending order (higher detail LODs have larger thresholds).
    pub fn with_thresholds(mut self, thresholds: Vec<f32>) -> Self {
        self.thresholds = Some(thresholds);
        self
    }

    /// Set the blending mode for LOD transitions.
    pub fn with_blend_mode(mut self, mode: LodBlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Build the LOD chain.
    ///
    /// Returns an error if the input data is invalid or simplification fails.
    pub fn build(&self) -> Result<LodChain, LodError> {
        self.validate()?;

        let mut levels = Vec::with_capacity(self.target_levels as usize);

        // Generate default thresholds if not provided
        let thresholds = self.thresholds.clone().unwrap_or_else(|| {
            (0..self.target_levels)
                .map(|i| 100.0 / (1 << i) as f32) // 100, 50, 25, 12.5, ...
                .collect()
        });

        // LOD 0: original mesh
        levels.push(LodLevel {
            level: 0,
            positions: self.positions.clone(),
            indices: self.indices.clone(),
            threshold: thresholds.first().copied().unwrap_or(100.0),
            reduction_ratio: 1.0,
        });

        // Generate simplified LOD levels
        let mut current_positions = self.positions.clone();
        let mut current_indices = self.indices.clone();

        for level in 1..self.target_levels {
            let target_ratio = self.reduction_per_level;
            let target_vertices = ((current_positions.len() as f32 * target_ratio) as usize)
                .max(MIN_VERTICES);

            // Simplify mesh
            let (simplified_positions, simplified_indices) =
                simplify_mesh(&current_positions, &current_indices, target_vertices)?;

            // Skip if we couldn't reduce vertices significantly
            if simplified_positions.len() >= current_positions.len() && level > 1 {
                break;
            }

            let actual_ratio = if current_positions.is_empty() {
                0.0
            } else {
                simplified_positions.len() as f32 / current_positions.len() as f32
            };

            levels.push(LodLevel {
                level,
                positions: simplified_positions.clone(),
                indices: simplified_indices.clone(),
                threshold: thresholds.get(level as usize).copied().unwrap_or(
                    thresholds.last().copied().unwrap_or(10.0) / 2.0,
                ),
                reduction_ratio: actual_ratio,
            });

            current_positions = simplified_positions;
            current_indices = simplified_indices;

            // Stop if we've reached minimum geometry
            if current_positions.len() <= MIN_VERTICES
                || current_indices.len() / 3 <= MIN_TRIANGLES
            {
                break;
            }
        }

        Ok(LodChain {
            levels,
            blend_mode: self.blend_mode,
        })
    }

    /// Validate input data.
    fn validate(&self) -> Result<(), LodError> {
        if self.indices.len() % 3 != 0 {
            return Err(LodError::InvalidIndexCount(self.indices.len()));
        }

        for &idx in &self.indices {
            if idx as usize >= self.positions.len() {
                return Err(LodError::IndexOutOfBounds {
                    index: idx,
                    vertex_count: self.positions.len(),
                });
            }
        }

        if self.reduction_per_level <= 0.0 || self.reduction_per_level >= 1.0 {
            return Err(LodError::InvalidReductionRatio(self.reduction_per_level));
        }

        if self.target_levels < 1 {
            return Err(LodError::InvalidLevelCount(self.target_levels));
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LodSelector
// ---------------------------------------------------------------------------

/// Selects appropriate LOD levels based on screen-space size and viewport settings.
pub struct LodSelector {
    /// Screen-space thresholds for each LOD level (in pixels).
    /// Higher values = higher detail LODs.
    thresholds: Vec<f32>,
    /// Viewport LOD bias (negative = higher detail, positive = lower detail).
    bias: f32,
}

impl LodSelector {
    /// Create a new LOD selector with the given thresholds.
    ///
    /// Thresholds should be in descending order (LOD 0 has the largest threshold).
    pub fn new(thresholds: &[f32]) -> Self {
        Self {
            thresholds: thresholds.to_vec(),
            bias: 0.0,
        }
    }

    /// Set the viewport LOD bias.
    ///
    /// - Negative values shift toward higher detail (lower LOD indices).
    /// - Positive values shift toward lower detail (higher LOD indices).
    ///
    /// The bias is added to the effective screen size before selection.
    pub fn with_bias(mut self, bias: f32) -> Self {
        self.bias = bias;
        self
    }

    /// Get the current LOD bias.
    pub fn bias(&self) -> f32 {
        self.bias
    }

    /// Set the LOD bias.
    pub fn set_bias(&mut self, bias: f32) {
        self.bias = bias;
    }

    /// Select LOD level and compute blend factor.
    ///
    /// # Arguments
    ///
    /// * `screen_size` - Object's screen-space size in pixels.
    /// * `blend_mode` - The blending mode to use for transitions.
    ///
    /// # Returns
    ///
    /// A `LodSelection` containing the primary LOD, optional secondary LOD,
    /// and blend factor for smooth transitions.
    pub fn select(&self, screen_size: f32, blend_mode: LodBlendMode) -> LodSelection {
        if self.thresholds.is_empty() {
            return LodSelection::single(0);
        }

        // Apply bias to effective screen size
        // Positive bias makes objects appear smaller (use lower detail)
        // Negative bias makes objects appear larger (use higher detail)
        let effective_size = screen_size * (1.0 - self.bias * 0.1);

        // Find appropriate LOD level
        let mut selected_lod = (self.thresholds.len() - 1) as u32;
        for (i, &threshold) in self.thresholds.iter().enumerate() {
            if effective_size >= threshold {
                selected_lod = i as u32;
                break;
            }
        }

        match blend_mode {
            LodBlendMode::Discrete => LodSelection::single(selected_lod),
            LodBlendMode::AlphaCrossfade { range } => {
                self.compute_crossfade_selection(effective_size, selected_lod, range)
            }
            LodBlendMode::Dither { pattern_size: _ } => {
                // Dither uses same blend factor as crossfade, pattern applied in shader
                self.compute_crossfade_selection(effective_size, selected_lod, 10.0)
            }
        }
    }

    /// Compute blended selection for crossfade mode.
    fn compute_crossfade_selection(
        &self,
        effective_size: f32,
        primary_lod: u32,
        blend_range: f32,
    ) -> LodSelection {
        // Get threshold for primary LOD
        let primary_threshold = self.thresholds
            .get(primary_lod as usize)
            .copied()
            .unwrap_or(0.0);

        // If we're at the lowest detail LOD, no blending needed
        let next_lod = primary_lod + 1;
        if next_lod as usize >= self.thresholds.len() {
            return LodSelection::single(primary_lod);
        }

        // Get threshold for next LOD
        let next_threshold = self.thresholds
            .get(next_lod as usize)
            .copied()
            .unwrap_or(0.0);

        // Calculate distance from threshold
        let distance_from_threshold = effective_size - primary_threshold;

        // If we're well above the threshold, no blending
        if distance_from_threshold >= blend_range {
            return LodSelection::single(primary_lod);
        }

        // If we're between thresholds, compute blend factor
        let threshold_range = primary_threshold - next_threshold;
        if threshold_range > 0.0 {
            let blend_start = primary_threshold;
            let blend_end = (primary_threshold - blend_range).max(next_threshold);

            if effective_size < blend_start && effective_size >= blend_end {
                let blend_factor = (blend_start - effective_size) / (blend_start - blend_end);
                return LodSelection::blended(primary_lod, next_lod, blend_factor);
            }
        }

        LodSelection::single(primary_lod)
    }
}

// ---------------------------------------------------------------------------
// Quadric Error Metrics (QEM) Mesh Simplification
// ---------------------------------------------------------------------------

/// Symmetric 4x4 matrix for error quadric (stored as 10 unique values).
#[derive(Debug, Clone, Copy)]
struct Quadric {
    // Upper triangular storage: a11, a12, a13, a14, a22, a23, a24, a33, a34, a44
    data: [f64; 10],
}

impl Quadric {
    fn zero() -> Self {
        Self { data: [0.0; 10] }
    }

    /// Create quadric from plane equation ax + by + cz + d = 0.
    fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            data: [
                a * a,     // a11
                a * b,     // a12
                a * c,     // a13
                a * d,     // a14
                b * b,     // a22
                b * c,     // a23
                b * d,     // a24
                c * c,     // a33
                c * d,     // a34
                d * d,     // a44
            ],
        }
    }

    /// Add another quadric to this one.
    fn add(&mut self, other: &Quadric) {
        for i in 0..10 {
            self.data[i] += other.data[i];
        }
    }

    /// Evaluate quadric error for a point.
    fn evaluate(&self, v: [f64; 3]) -> f64 {
        let x = v[0];
        let y = v[1];
        let z = v[2];

        // Q(v) = v^T * A * v (symmetric matrix)
        self.data[0] * x * x +
        2.0 * self.data[1] * x * y +
        2.0 * self.data[2] * x * z +
        2.0 * self.data[3] * x +
        self.data[4] * y * y +
        2.0 * self.data[5] * y * z +
        2.0 * self.data[6] * y +
        self.data[7] * z * z +
        2.0 * self.data[8] * z +
        self.data[9]
    }
}

/// Edge collapse candidate with error metric.
#[derive(Debug, Clone)]
struct EdgeCollapse {
    /// First vertex index.
    v1: usize,
    /// Second vertex index.
    v2: usize,
    /// Target position after collapse.
    target: [f64; 3],
    /// Error cost of this collapse.
    error: f64,
}

impl PartialEq for EdgeCollapse {
    fn eq(&self, other: &Self) -> bool {
        self.error == other.error && self.v1 == other.v1 && self.v2 == other.v2
    }
}

impl Eq for EdgeCollapse {}

impl Ord for EdgeCollapse {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (smallest error first)
        other.error.partial_cmp(&self.error).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for EdgeCollapse {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Simplify mesh using Quadric Error Metrics.
fn simplify_mesh(
    positions: &[[f32; 3]],
    indices: &[u32],
    target_vertices: usize,
) -> Result<(Vec<[f32; 3]>, Vec<u32>), LodError> {
    if positions.len() <= target_vertices {
        return Ok((positions.to_vec(), indices.to_vec()));
    }

    if indices.len() < 3 {
        return Ok((positions.to_vec(), indices.to_vec()));
    }

    // Convert to f64 for numerical stability
    let mut vertices: Vec<[f64; 3]> = positions
        .iter()
        .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
        .collect();

    // Build triangle list
    let mut triangles: Vec<[usize; 3]> = indices
        .chunks(3)
        .map(|tri| [tri[0] as usize, tri[1] as usize, tri[2] as usize])
        .collect();

    // Track which vertices are still active
    let mut active: Vec<bool> = vec![true; vertices.len()];

    // Compute quadrics for each vertex
    let mut quadrics = compute_vertex_quadrics(&vertices, &triangles);

    // Build edge collapse heap
    let mut heap = build_collapse_heap(&vertices, &triangles, &quadrics);

    // Track vertex remapping (for collapsed vertices)
    let mut remap: Vec<usize> = (0..vertices.len()).collect();

    // Collapse edges until we reach target vertex count
    let mut active_count = vertices.len();

    while active_count > target_vertices && !heap.is_empty() {
        let collapse = match heap.pop() {
            Some(c) => c,
            None => break,
        };

        // Skip if either vertex has been collapsed
        let v1 = find_root(&remap, collapse.v1);
        let v2 = find_root(&remap, collapse.v2);

        if v1 == v2 || !active[v1] || !active[v2] {
            continue;
        }

        // Perform collapse: merge v2 into v1
        vertices[v1] = [
            collapse.target[0],
            collapse.target[1],
            collapse.target[2],
        ];

        // Merge quadrics (copy v2's quadric first to avoid borrow conflict)
        let q2 = quadrics[v2];
        quadrics[v1].add(&q2);

        // Mark v2 as collapsed
        active[v2] = false;
        remap[v2] = v1;
        active_count -= 1;

        // Update triangles (remap v2 to v1, remove degenerate triangles)
        triangles.retain_mut(|tri| {
            for v in tri.iter_mut() {
                *v = find_root(&remap, *v);
            }
            // Remove degenerate triangles
            tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2]
        });

        // Re-add affected edges to heap
        for tri in &triangles {
            for &v in tri {
                if v == v1 {
                    // Add edges from v1 to other vertices in triangle
                    for &other in tri {
                        if other != v1 && active[other] {
                            if let Some(collapse) = compute_edge_collapse(
                                &vertices,
                                &quadrics,
                                v1,
                                other,
                            ) {
                                heap.push(collapse);
                            }
                        }
                    }
                }
            }
        }
    }

    // Build final mesh
    let (final_positions, final_indices) = compact_mesh(&vertices, &triangles, &active, &remap);

    Ok((final_positions, final_indices))
}

/// Find root in union-find structure with path compression.
fn find_root(remap: &[usize], mut idx: usize) -> usize {
    while remap[idx] != idx {
        idx = remap[idx];
    }
    idx
}

/// Compute error quadrics for each vertex.
fn compute_vertex_quadrics(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
) -> Vec<Quadric> {
    let mut quadrics = vec![Quadric::zero(); vertices.len()];

    for tri in triangles {
        // Compute triangle plane
        let v0 = vertices[tri[0]];
        let v1 = vertices[tri[1]];
        let v2 = vertices[tri[2]];

        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        // Cross product for normal
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len < 1e-10 {
            continue;
        }

        // Normalized plane: ax + by + cz + d = 0
        let a = n[0] / len;
        let b = n[1] / len;
        let c = n[2] / len;
        let d = -(a * v0[0] + b * v0[1] + c * v0[2]);

        let plane_quadric = Quadric::from_plane(a, b, c, d);

        // Add to each vertex in the triangle
        quadrics[tri[0]].add(&plane_quadric);
        quadrics[tri[1]].add(&plane_quadric);
        quadrics[tri[2]].add(&plane_quadric);
    }

    quadrics
}

/// Build initial heap of edge collapse candidates.
fn build_collapse_heap(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
    quadrics: &[Quadric],
) -> BinaryHeap<EdgeCollapse> {
    let mut heap = BinaryHeap::new();
    let mut processed = std::collections::HashSet::new();

    for tri in triangles {
        let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];

        for (v1, v2) in edges {
            let key = if v1 < v2 { (v1, v2) } else { (v2, v1) };
            if processed.contains(&key) {
                continue;
            }
            processed.insert(key);

            if let Some(collapse) = compute_edge_collapse(vertices, quadrics, v1, v2) {
                heap.push(collapse);
            }
        }
    }

    heap
}

/// Compute optimal collapse for an edge.
fn compute_edge_collapse(
    vertices: &[[f64; 3]],
    quadrics: &[Quadric],
    v1: usize,
    v2: usize,
) -> Option<EdgeCollapse> {
    // Combined quadric
    let mut combined = quadrics[v1];
    combined.add(&quadrics[v2]);

    // Try to find optimal position analytically (solve Qx = 0)
    // For simplicity, use midpoint (could implement matrix solve for better results)
    let target = [
        (vertices[v1][0] + vertices[v2][0]) * 0.5,
        (vertices[v1][1] + vertices[v2][1]) * 0.5,
        (vertices[v1][2] + vertices[v2][2]) * 0.5,
    ];

    let error = combined.evaluate(target);

    Some(EdgeCollapse {
        v1,
        v2,
        target,
        error,
    })
}

/// Compact mesh by removing collapsed vertices and remapping indices.
fn compact_mesh(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
    active: &[bool],
    remap: &[usize],
) -> (Vec<[f32; 3]>, Vec<u32>) {
    // Build compacted vertex list
    let mut new_positions = Vec::new();
    let mut old_to_new: Vec<Option<u32>> = vec![None; vertices.len()];

    for (old_idx, (&is_active, v)) in active.iter().zip(vertices.iter()).enumerate() {
        if is_active {
            old_to_new[old_idx] = Some(new_positions.len() as u32);
            new_positions.push([v[0] as f32, v[1] as f32, v[2] as f32]);
        }
    }

    // Remap indices
    let mut new_indices = Vec::new();
    for tri in triangles {
        let i0 = find_root(remap, tri[0]);
        let i1 = find_root(remap, tri[1]);
        let i2 = find_root(remap, tri[2]);

        // Skip degenerate triangles
        if i0 == i1 || i1 == i2 || i0 == i2 {
            continue;
        }

        if let (Some(n0), Some(n1), Some(n2)) = (old_to_new[i0], old_to_new[i1], old_to_new[i2]) {
            new_indices.push(n0);
            new_indices.push(n1);
            new_indices.push(n2);
        }
    }

    (new_positions, new_indices)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple quad mesh (2 triangles, 4 vertices).
    fn make_quad() -> (Vec<[f32; 3]>, Vec<u32>) {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 1, 3, 2];
        (positions, indices)
    }

    /// Create a simple grid mesh for testing simplification.
    fn make_grid(size: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        // Create grid of vertices
        for y in 0..=size {
            for x in 0..=size {
                positions.push([x as f32, y as f32, 0.0]);
            }
        }

        // Create triangles
        let stride = size + 1;
        for y in 0..size {
            for x in 0..size {
                let i = y * stride + x;
                // First triangle
                indices.push(i as u32);
                indices.push((i + 1) as u32);
                indices.push((i + stride) as u32);
                // Second triangle
                indices.push((i + 1) as u32);
                indices.push((i + stride + 1) as u32);
                indices.push((i + stride) as u32);
            }
        }

        (positions, indices)
    }

    #[test]
    fn test_lod_builder_single_level() {
        let (positions, indices) = make_quad();
        let chain = LodBuilder::new(positions.clone(), indices.clone())
            .with_levels(1)
            .with_reduction(0.5)
            .build()
            .unwrap();

        assert_eq!(chain.level_count(), 1);
        assert_eq!(chain.levels[0].level, 0);
        assert_eq!(chain.levels[0].positions.len(), positions.len());
        assert_eq!(chain.levels[0].indices.len(), indices.len());
        assert_eq!(chain.levels[0].reduction_ratio, 1.0);
    }

    #[test]
    fn test_lod_builder_three_levels() {
        // Use a larger mesh so simplification is meaningful
        let (positions, indices) = make_grid(4);
        let chain = LodBuilder::new(positions.clone(), indices.clone())
            .with_levels(3)
            .with_reduction(0.5)
            .build()
            .unwrap();

        assert!(chain.level_count() >= 1);
        assert!(chain.level_count() <= 3);

        // LOD 0 should be the original
        assert_eq!(chain.levels[0].positions.len(), positions.len());
        assert_eq!(chain.levels[0].indices.len(), indices.len());

        // Each subsequent level should have fewer vertices
        for i in 1..chain.levels.len() {
            assert!(
                chain.levels[i].positions.len() < chain.levels[i - 1].positions.len(),
                "LOD {} should have fewer vertices than LOD {}",
                i,
                i - 1
            );
        }
    }

    #[test]
    fn test_lod_reduction_ratio() {
        let (positions, indices) = make_grid(6);
        let chain = LodBuilder::new(positions.clone(), indices)
            .with_levels(3)
            .with_reduction(0.5)
            .build()
            .unwrap();

        // Each level should have roughly 50% of previous vertices
        for level in &chain.levels[1..] {
            // Allow some tolerance since simplification isn't exact
            assert!(
                level.reduction_ratio <= 0.8,
                "reduction_ratio {} should be <= 0.8",
                level.reduction_ratio
            );
        }
    }

    #[test]
    fn test_lod_selector_discrete() {
        let selector = LodSelector::new(&[100.0, 50.0, 25.0]);

        // Large object -> LOD 0
        let sel = selector.select(150.0, LodBlendMode::Discrete);
        assert_eq!(sel.primary_lod, 0);
        assert!(sel.secondary_lod.is_none());
        assert_eq!(sel.blend_factor, 0.0);

        // Medium object -> LOD 1
        let sel = selector.select(75.0, LodBlendMode::Discrete);
        assert_eq!(sel.primary_lod, 1);

        // Small object -> LOD 2
        let sel = selector.select(30.0, LodBlendMode::Discrete);
        assert_eq!(sel.primary_lod, 2);

        // Very small object -> LOD 2 (last level)
        let sel = selector.select(10.0, LodBlendMode::Discrete);
        assert_eq!(sel.primary_lod, 2);
    }

    #[test]
    fn test_lod_selector_crossfade() {
        let selector = LodSelector::new(&[100.0, 50.0, 25.0]);

        // Well above threshold -> no blending
        let sel = selector.select(120.0, LodBlendMode::AlphaCrossfade { range: 10.0 });
        assert_eq!(sel.primary_lod, 0);
        assert!(!sel.needs_blending());

        // Near threshold boundary -> may blend to next LOD
        // With screen_size=95, we're below LOD 0 threshold (100), so we get LOD 1
        let sel = selector.select(95.0, LodBlendMode::AlphaCrossfade { range: 10.0 });
        // At 95 pixels, we're between LOD 0 (100) and LOD 1 (50) thresholds
        // So primary_lod should be 1 (since 95 < 100)
        assert!(sel.primary_lod <= 1);
    }

    #[test]
    fn test_lod_selector_bias() {
        let selector_no_bias = LodSelector::new(&[100.0, 50.0, 25.0]);
        let selector_positive = LodSelector::new(&[100.0, 50.0, 25.0]).with_bias(2.0);
        let selector_negative = LodSelector::new(&[100.0, 50.0, 25.0]).with_bias(-2.0);

        let screen_size = 75.0;

        // No bias -> LOD 1
        let sel = selector_no_bias.select(screen_size, LodBlendMode::Discrete);
        assert_eq!(sel.primary_lod, 1);

        // Positive bias (lower detail) -> may select LOD 2
        let sel_pos = selector_positive.select(screen_size, LodBlendMode::Discrete);
        assert!(sel_pos.primary_lod >= sel.primary_lod);

        // Negative bias (higher detail) -> may select LOD 0
        let sel_neg = selector_negative.select(screen_size, LodBlendMode::Discrete);
        assert!(sel_neg.primary_lod <= sel.primary_lod);
    }

    #[test]
    fn test_mesh_simplification() {
        let (positions, indices) = make_grid(8); // 81 vertices, 128 triangles

        let (simplified_pos, simplified_idx) =
            simplify_mesh(&positions, &indices, 20).unwrap();

        // Should have significantly fewer vertices
        assert!(
            simplified_pos.len() < positions.len(),
            "simplified should have fewer vertices: {} < {}",
            simplified_pos.len(),
            positions.len()
        );

        // Should still have valid triangles
        assert!(simplified_idx.len() % 3 == 0);
        assert!(simplified_idx.len() > 0);

        // All indices should be valid
        for &idx in &simplified_idx {
            assert!(
                (idx as usize) < simplified_pos.len(),
                "index {} out of bounds for {} vertices",
                idx,
                simplified_pos.len()
            );
        }
    }

    #[test]
    fn test_lod_chain_accessors() {
        let (positions, indices) = make_grid(4);
        let chain = LodBuilder::new(positions, indices)
            .with_levels(3)
            .with_reduction(0.5)
            .with_blend_mode(LodBlendMode::AlphaCrossfade { range: 5.0 })
            .build()
            .unwrap();

        assert!(chain.highest_detail().is_some());
        assert!(chain.lowest_detail().is_some());

        let highest = chain.highest_detail().unwrap();
        assert_eq!(highest.level, 0);

        let lowest = chain.lowest_detail().unwrap();
        assert_eq!(lowest.level as usize, chain.level_count() - 1);

        assert_eq!(chain.blend_mode, LodBlendMode::AlphaCrossfade { range: 5.0 });
    }

    #[test]
    fn test_lod_selection_needs_blending() {
        let single = LodSelection::single(0);
        assert!(!single.needs_blending());

        let blended = LodSelection::blended(0, 1, 0.5);
        assert!(blended.needs_blending());

        let no_blend = LodSelection::blended(0, 1, 0.0);
        assert!(!no_blend.needs_blending());

        let full_blend = LodSelection::blended(0, 1, 1.0);
        assert!(!full_blend.needs_blending());
    }

    #[test]
    fn test_lod_error_invalid_indices() {
        let positions = vec![[0.0, 0.0, 0.0]];
        let indices = vec![0, 1, 2]; // Invalid: only 1 vertex

        let result = LodBuilder::new(positions, indices)
            .with_levels(2)
            .build();

        assert!(result.is_err());
        match result.unwrap_err() {
            LodError::IndexOutOfBounds { index: 1, .. } => {}
            e => panic!("unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_lod_error_invalid_reduction() {
        let (positions, indices) = make_quad();

        let result = LodBuilder::new(positions.clone(), indices.clone())
            .with_reduction(1.5) // Invalid: > 1.0
            .build();

        assert!(result.is_err());
        match result.unwrap_err() {
            LodError::InvalidReductionRatio(r) if r == 1.5 => {}
            e => panic!("unexpected error: {:?}", e),
        }

        let result = LodBuilder::new(positions, indices)
            .with_reduction(0.0) // Invalid: == 0.0
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_quadric_evaluation() {
        // Test that quadric from plane evaluates to 0 on the plane
        let q = Quadric::from_plane(0.0, 0.0, 1.0, -5.0); // z = 5 plane
        let on_plane = [0.0, 0.0, 5.0];
        let error = q.evaluate(on_plane);
        assert!(error.abs() < 1e-10, "error on plane should be ~0: {}", error);

        let off_plane = [0.0, 0.0, 6.0];
        let error = q.evaluate(off_plane);
        assert!(error > 0.0, "error off plane should be > 0: {}", error);
    }

    #[test]
    fn test_empty_mesh() {
        let positions: Vec<[f32; 3]> = vec![];
        let indices: Vec<u32> = vec![];

        let chain = LodBuilder::new(positions, indices)
            .with_levels(3)
            .build()
            .unwrap();

        // Empty mesh still generates LOD levels (with empty geometry)
        assert!(chain.level_count() >= 1);
        assert_eq!(chain.levels[0].positions.len(), 0);
    }

    #[test]
    fn test_dither_blend_mode() {
        let selector = LodSelector::new(&[100.0, 50.0, 25.0]);
        let sel = selector.select(75.0, LodBlendMode::Dither { pattern_size: 4 });

        // Dither mode should still select proper LOD
        assert!(sel.primary_lod <= 2);
    }

    // =========================================================================
    // EDGE CASE TESTS (T-MAT-8.4)
    // =========================================================================

    /// Edge case: Very small mesh (< 10 triangles) should not crash and should
    /// produce at least one LOD level.
    #[test]
    fn test_very_small_mesh_single_triangle() {
        // Single triangle: 3 vertices, 3 indices
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2];

        let chain = LodBuilder::new(positions.clone(), indices.clone())
            .with_levels(3)
            .with_reduction(0.5)
            .build()
            .unwrap();

        // Should produce at least LOD 0 (original)
        assert!(chain.level_count() >= 1);
        assert_eq!(chain.levels[0].positions.len(), 3);
        assert_eq!(chain.levels[0].indices.len(), 3);
        assert_eq!(chain.levels[0].triangle_count(), 1);
    }

    /// Edge case: Mesh with exactly MIN_VERTICES (4) should not simplify further.
    #[test]
    fn test_mesh_at_min_vertices() {
        let (positions, indices) = make_quad(); // 4 vertices, 2 triangles

        let chain = LodBuilder::new(positions.clone(), indices.clone())
            .with_levels(5)
            .with_reduction(0.5)
            .build()
            .unwrap();

        // LOD 0 is original; further levels may not reduce below MIN_VERTICES
        assert_eq!(chain.levels[0].vertex_count(), 4);

        // All levels should have valid geometry
        for level in &chain.levels {
            assert!(level.vertex_count() >= MIN_VERTICES);
            assert!(level.triangle_count() >= MIN_TRIANGLES);
        }
    }

    /// Edge case: Aggressive reduction (90%+) should not crash or produce
    /// invalid geometry.
    #[test]
    fn test_aggressive_reduction_ratio() {
        let (positions, indices) = make_grid(10); // 121 vertices, 200 triangles

        // 10% retention (90% reduction)
        let chain = LodBuilder::new(positions.clone(), indices.clone())
            .with_levels(5)
            .with_reduction(0.1)
            .build()
            .unwrap();

        // Should produce multiple levels with decreasing vertex counts
        assert!(chain.level_count() >= 2);

        // All levels must have valid geometry
        for level in &chain.levels {
            assert!(level.indices.len() % 3 == 0, "indices must be multiple of 3");
            for &idx in &level.indices {
                assert!(
                    (idx as usize) < level.positions.len(),
                    "LOD {} has invalid index {} >= {}",
                    level.level,
                    idx,
                    level.positions.len()
                );
            }
        }
    }

    /// Edge case: Boundary values for blend range.
    #[test]
    fn test_crossfade_blend_range_boundaries() {
        let selector = LodSelector::new(&[100.0, 50.0, 25.0]);

        // Zero blend range: should behave like discrete
        let sel = selector.select(95.0, LodBlendMode::AlphaCrossfade { range: 0.0 });
        assert!(!sel.needs_blending(), "zero range should not blend");

        // Very large blend range
        let sel = selector.select(75.0, LodBlendMode::AlphaCrossfade { range: 1000.0 });
        // Should select LOD based on threshold, blend factor depends on position
        assert!(sel.primary_lod <= 2);

        // Exact threshold match
        let sel = selector.select(100.0, LodBlendMode::AlphaCrossfade { range: 10.0 });
        assert_eq!(sel.primary_lod, 0, "exact threshold should select that LOD");
    }

    /// Edge case: Negative bias values should shift toward higher detail (lower LOD index).
    #[test]
    fn test_negative_bias_values() {
        let selector = LodSelector::new(&[100.0, 50.0, 25.0]).with_bias(-5.0);

        // With strong negative bias, even small objects should select higher detail
        // effective_size = 40.0 * (1.0 - (-5.0 * 0.1)) = 40.0 * 1.5 = 60.0
        // 60.0 >= 50.0, so LOD 1
        let sel = selector.select(40.0, LodBlendMode::Discrete);
        assert!(sel.primary_lod <= 1, "negative bias should prefer higher detail");
    }

    /// Edge case: Large positive bias should shift toward lower detail.
    #[test]
    fn test_large_positive_bias() {
        let selector = LodSelector::new(&[100.0, 50.0, 25.0]).with_bias(5.0);

        // With strong positive bias, even large objects may select lower detail
        // effective_size = 90.0 * (1.0 - (5.0 * 0.1)) = 90.0 * 0.5 = 45.0
        // 45.0 < 50.0 so LOD 2
        let sel = selector.select(90.0, LodBlendMode::Discrete);
        assert!(sel.primary_lod >= 1, "positive bias should prefer lower detail");
    }

    /// Edge case: Empty thresholds array should return LOD 0.
    #[test]
    fn test_empty_thresholds() {
        let selector = LodSelector::new(&[]);

        let sel = selector.select(100.0, LodBlendMode::Discrete);
        assert_eq!(sel.primary_lod, 0, "empty thresholds should default to LOD 0");
        assert!(!sel.needs_blending());
    }

    /// Edge case: Single threshold should work correctly.
    #[test]
    fn test_single_threshold() {
        let selector = LodSelector::new(&[50.0]);

        // Above threshold -> LOD 0
        let sel = selector.select(100.0, LodBlendMode::Discrete);
        assert_eq!(sel.primary_lod, 0);

        // Below threshold -> still LOD 0 (last available)
        let sel = selector.select(25.0, LodBlendMode::Discrete);
        assert_eq!(sel.primary_lod, 0);
    }

    /// Edge case: Non-monotonic index count (malformed input).
    #[test]
    fn test_index_count_not_multiple_of_three() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = vec![0, 1]; // Not a multiple of 3

        let result = LodBuilder::new(positions, indices)
            .with_levels(2)
            .build();

        assert!(result.is_err());
        match result.unwrap_err() {
            LodError::InvalidIndexCount(2) => {}
            e => panic!("expected InvalidIndexCount(2), got {:?}", e),
        }
    }

    /// Edge case: Reduction ratio at boundary (just above 0, just below 1).
    #[test]
    fn test_reduction_ratio_boundaries() {
        let (positions, indices) = make_grid(4);

        // Just above 0.0 (valid but aggressive)
        let chain = LodBuilder::new(positions.clone(), indices.clone())
            .with_reduction(0.01)
            .with_levels(2)
            .build()
            .unwrap();
        assert!(chain.level_count() >= 1);

        // Just below 1.0 (valid but minimal reduction)
        let chain = LodBuilder::new(positions.clone(), indices.clone())
            .with_reduction(0.99)
            .with_levels(2)
            .build()
            .unwrap();
        assert!(chain.level_count() >= 1);

        // Exactly 1.0 (invalid)
        let result = LodBuilder::new(positions.clone(), indices.clone())
            .with_reduction(1.0)
            .build();
        assert!(result.is_err());

        // Exactly 0.0 (invalid)
        let result = LodBuilder::new(positions, indices)
            .with_reduction(0.0)
            .build();
        assert!(result.is_err());
    }

    /// Edge case: Custom thresholds in non-descending order should still work.
    #[test]
    fn test_custom_thresholds() {
        let (positions, indices) = make_grid(4);

        let chain = LodBuilder::new(positions, indices)
            .with_levels(3)
            .with_reduction(0.5)
            .with_thresholds(vec![200.0, 100.0, 50.0])
            .build()
            .unwrap();

        assert_eq!(chain.levels[0].threshold, 200.0);
        // Threshold assignment depends on how many levels were actually generated
        if chain.level_count() >= 2 {
            assert_eq!(chain.levels[1].threshold, 100.0);
        }
    }

    /// Edge case: Quadric addition should be numerically stable.
    #[test]
    fn test_quadric_addition_stability() {
        let q1 = Quadric::from_plane(1.0, 0.0, 0.0, -1.0);
        let q2 = Quadric::from_plane(0.0, 1.0, 0.0, -2.0);

        let mut combined = q1;
        combined.add(&q2);

        // Point at (1, 2, z) should have low error for both planes
        let point = [1.0, 2.0, 0.0];
        let error = combined.evaluate(point);
        assert!(error < 1e-10, "combined quadric error should be near zero at intersection");
    }

    /// Edge case: Blend factor should be clamped to [0, 1].
    #[test]
    fn test_blend_factor_clamping() {
        let sel_under = LodSelection::blended(0, 1, -0.5);
        assert_eq!(sel_under.blend_factor, 0.0, "negative blend factor should clamp to 0");

        let sel_over = LodSelection::blended(0, 1, 1.5);
        assert_eq!(sel_over.blend_factor, 1.0, "blend factor > 1 should clamp to 1");
    }

    /// Edge case: LodLevel accessors should return correct counts.
    #[test]
    fn test_lod_level_accessors() {
        let (positions, indices) = make_grid(3); // 16 vertices, 18 triangles

        let chain = LodBuilder::new(positions, indices)
            .with_levels(1)
            .with_reduction(0.5)
            .build()
            .unwrap();

        let level = &chain.levels[0];
        assert_eq!(level.triangle_count(), level.indices.len() / 3);
        assert_eq!(level.vertex_count(), level.positions.len());
        assert_eq!(level.triangle_count(), 18);
        assert_eq!(level.vertex_count(), 16);
    }

    /// Edge case: Screen size of zero should select lowest detail LOD.
    #[test]
    fn test_zero_screen_size() {
        let selector = LodSelector::new(&[100.0, 50.0, 25.0]);

        let sel = selector.select(0.0, LodBlendMode::Discrete);
        assert_eq!(sel.primary_lod, 2, "zero screen size should select lowest detail");
    }

    /// Edge case: Very large screen size should select highest detail LOD.
    #[test]
    fn test_very_large_screen_size() {
        let selector = LodSelector::new(&[100.0, 50.0, 25.0]);

        let sel = selector.select(10000.0, LodBlendMode::Discrete);
        assert_eq!(sel.primary_lod, 0, "very large screen size should select highest detail");
    }

    /// Edge case: Crossfade at lowest LOD should not blend (no next LOD).
    #[test]
    fn test_crossfade_at_lowest_lod() {
        let selector = LodSelector::new(&[100.0, 50.0, 25.0]);

        // Screen size below all thresholds -> lowest LOD
        let sel = selector.select(10.0, LodBlendMode::AlphaCrossfade { range: 20.0 });
        assert_eq!(sel.primary_lod, 2);
        assert!(!sel.needs_blending(), "lowest LOD should not blend to non-existent next LOD");
    }

    /// Edge case: Selector bias getter and setter should work.
    #[test]
    fn test_selector_bias_accessors() {
        let mut selector = LodSelector::new(&[100.0, 50.0, 25.0]);
        assert_eq!(selector.bias(), 0.0);

        selector.set_bias(3.5);
        assert_eq!(selector.bias(), 3.5);

        selector.set_bias(-2.0);
        assert_eq!(selector.bias(), -2.0);
    }

    /// Edge case: LodChain get_level with out-of-bounds index returns None.
    #[test]
    fn test_lod_chain_get_level_out_of_bounds() {
        let (positions, indices) = make_quad();
        let chain = LodBuilder::new(positions, indices)
            .with_levels(2)
            .with_reduction(0.5)
            .build()
            .unwrap();

        assert!(chain.get_level(0).is_some());
        assert!(chain.get_level(100).is_none());
        assert!(chain.get_level(u32::MAX).is_none());
    }

    /// Edge case: Error display formatting.
    #[test]
    fn test_error_display_formatting() {
        let e1 = LodError::InvalidIndexCount(5);
        assert!(format!("{}", e1).contains("5"));
        assert!(format!("{}", e1).contains("not a multiple of 3"));

        let e2 = LodError::IndexOutOfBounds { index: 10, vertex_count: 5 };
        assert!(format!("{}", e2).contains("10"));
        assert!(format!("{}", e2).contains("5"));

        let e3 = LodError::InsufficientGeometry { vertices: 2, triangles: 0 };
        assert!(format!("{}", e3).contains("2"));
        assert!(format!("{}", e3).contains("0"));

        let e4 = LodError::InvalidReductionRatio(1.5);
        assert!(format!("{}", e4).contains("1.5"));

        let e5 = LodError::InvalidLevelCount(0);
        assert!(format!("{}", e5).contains("0"));
    }

    /// Edge case: with_levels(0) should be corrected to at least 1.
    #[test]
    fn test_zero_level_count_corrected() {
        let (positions, indices) = make_quad();
        let chain = LodBuilder::new(positions, indices)
            .with_levels(0) // Should be corrected to 1
            .with_reduction(0.5)
            .build()
            .unwrap();

        assert!(chain.level_count() >= 1, "zero levels should be corrected to at least 1");
    }

    /// Edge case: Simplification should preserve mesh manifold properties.
    #[test]
    fn test_simplification_no_duplicate_indices_in_triangle() {
        let (positions, indices) = make_grid(6);

        let chain = LodBuilder::new(positions, indices)
            .with_levels(4)
            .with_reduction(0.5)
            .build()
            .unwrap();

        for level in &chain.levels {
            for tri in level.indices.chunks(3) {
                assert_ne!(tri[0], tri[1], "degenerate triangle in LOD {}", level.level);
                assert_ne!(tri[1], tri[2], "degenerate triangle in LOD {}", level.level);
                assert_ne!(tri[0], tri[2], "degenerate triangle in LOD {}", level.level);
            }
        }
    }
}

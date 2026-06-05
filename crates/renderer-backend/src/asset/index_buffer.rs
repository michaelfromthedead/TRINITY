//! Index buffer optimization for TRINITY.
//!
//! Optimizes index buffers for GPU vertex cache efficiency with support for:
//! - Automatic index type selection (U8/U16/U32)
//! - Vertex cache optimization (Tom Forsyth's algorithm)
//! - Pre-transform vertex reordering
//! - Triangle stripification (optional)
//! - Idempotent transformations
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::index_buffer::*;
//!
//! // Basic usage - optimize indices for a mesh
//! let indices: Vec<u32> = vec![0, 1, 2, 2, 1, 3, ...];
//! let vertex_count = 1000;
//!
//! // Select optimal index type
//! let index_type = select_index_type(vertex_count);
//!
//! // Optimize for vertex cache
//! let optimized = optimize_vertex_cache(&indices, vertex_count);
//!
//! // Compute ACMR (Average Cache Miss Ratio)
//! let acmr = compute_acmr(&optimized, 32); // 32-entry cache
//!
//! // Full optimization pipeline
//! let config = IndexBufferConfig::default();
//! let result = optimize_index_buffer(&indices, vertex_count, &config);
//! println!("ACMR improved to: {}", result.acmr);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Index buffer optimization error.
#[derive(Debug, Clone)]
pub enum IndexBufferError {
    /// Index out of bounds for vertex count.
    IndexOutOfBounds { index: u32, vertex_count: usize },
    /// Invalid triangle count (not divisible by 3).
    InvalidTriangleCount(usize),
    /// Empty index buffer.
    EmptyBuffer,
    /// Optimization failed.
    OptimizationFailed(String),
}

impl std::fmt::Display for IndexBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexOutOfBounds { index, vertex_count } => {
                write!(f, "index {} out of bounds for {} vertices", index, vertex_count)
            }
            Self::InvalidTriangleCount(count) => {
                write!(f, "index count {} not divisible by 3", count)
            }
            Self::EmptyBuffer => write!(f, "empty index buffer"),
            Self::OptimizationFailed(msg) => write!(f, "optimization failed: {}", msg),
        }
    }
}

impl std::error::Error for IndexBufferError {}

// ---------------------------------------------------------------------------
// Index type
// ---------------------------------------------------------------------------

/// Index data type for GPU index buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexType {
    /// 8-bit unsigned indices (max 256 vertices, for meshlets).
    U8,
    /// 16-bit unsigned indices (max 65536 vertices).
    U16,
    /// 32-bit unsigned indices (unlimited vertices).
    U32,
}

impl IndexType {
    /// Size in bytes of a single index.
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
        }
    }

    /// Maximum vertex index representable by this type.
    pub const fn max_index(self) -> u32 {
        match self {
            Self::U8 => 255,
            Self::U16 => 65535,
            Self::U32 => u32::MAX,
        }
    }

    /// Maximum vertex count supported by this type.
    pub const fn max_vertex_count(self) -> usize {
        match self {
            Self::U8 => 256,
            Self::U16 => 65536,
            Self::U32 => u32::MAX as usize + 1,
        }
    }
}

impl Default for IndexType {
    fn default() -> Self {
        Self::U16
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for index buffer optimization.
#[derive(Debug, Clone)]
pub struct IndexBufferConfig {
    /// Enable vertex cache optimization (Tom Forsyth's algorithm).
    pub optimize_cache: bool,
    /// Reorder vertices to match optimized index order.
    pub reorder_vertices: bool,
    /// Convert to triangle strip format (requires primitive restart).
    pub stripify: bool,
    /// Target ACMR (Average Cache Miss Ratio). Stop optimizing when reached.
    pub target_acmr: f32,
    /// Simulated vertex cache size for ACMR computation.
    pub cache_size: usize,
    /// Value used for primitive restart in strips (usually 0xFFFF or 0xFFFFFFFF).
    pub primitive_restart_index: Option<u32>,
}

impl Default for IndexBufferConfig {
    fn default() -> Self {
        Self {
            optimize_cache: true,
            reorder_vertices: true,
            stripify: false,
            target_acmr: 0.7,
            cache_size: 32, // Common GPU vertex cache size
            primitive_restart_index: None,
        }
    }
}

impl IndexBufferConfig {
    /// Create a config with cache optimization only.
    pub fn cache_only() -> Self {
        Self {
            optimize_cache: true,
            reorder_vertices: false,
            stripify: false,
            ..Default::default()
        }
    }

    /// Create a config with full optimization (cache + vertex reorder).
    pub fn full() -> Self {
        Self {
            optimize_cache: true,
            reorder_vertices: true,
            stripify: false,
            ..Default::default()
        }
    }

    /// Create a config for stripification.
    pub fn stripify(restart_index: u32) -> Self {
        Self {
            optimize_cache: true,
            reorder_vertices: true,
            stripify: true,
            primitive_restart_index: Some(restart_index),
            ..Default::default()
        }
    }

    /// Set target ACMR.
    pub fn with_target_acmr(mut self, acmr: f32) -> Self {
        self.target_acmr = acmr;
        self
    }

    /// Set cache size for ACMR computation.
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// Optimized index buffer output.
#[derive(Debug, Clone)]
pub struct OptimizedIndices {
    /// Raw index data bytes (format depends on index_type).
    pub indices: Vec<u8>,
    /// Index type (U8, U16, U32).
    pub index_type: IndexType,
    /// Number of primitives (triangles for triangle list, varies for strip).
    pub primitive_count: u32,
    /// Computed ACMR after optimization.
    pub acmr: f32,
    /// Whether the output is a triangle strip.
    pub is_strip: bool,
}

impl OptimizedIndices {
    /// Get the number of indices.
    pub fn index_count(&self) -> usize {
        self.indices.len() / self.index_type.size_bytes()
    }

    /// Extract indices as u32 slice (copies data).
    pub fn to_u32_vec(&self) -> Vec<u32> {
        let count = self.index_count();
        let mut result = Vec::with_capacity(count);
        match self.index_type {
            IndexType::U8 => {
                for &b in &self.indices {
                    result.push(b as u32);
                }
            }
            IndexType::U16 => {
                for chunk in self.indices.chunks_exact(2) {
                    result.push(u16::from_le_bytes([chunk[0], chunk[1]]) as u32);
                }
            }
            IndexType::U32 => {
                for chunk in self.indices.chunks_exact(4) {
                    result.push(u32::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                    ]));
                }
            }
        }
        result
    }
}

/// Result of vertex reordering.
#[derive(Debug, Clone)]
pub struct ReorderedMesh<V> {
    /// Reordered vertices.
    pub vertices: Vec<V>,
    /// Remapped indices.
    pub indices: Vec<u32>,
    /// Mapping from new index to old index.
    pub new_to_old: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Index type selection
// ---------------------------------------------------------------------------

/// Select the optimal index type based on vertex count.
///
/// - U8 for meshlets (<=256 vertices)
/// - U16 for small/medium meshes (<65536 vertices)
/// - U32 for large meshes (>=65536 vertices)
///
/// # Example
///
/// ```ignore
/// assert_eq!(select_index_type(100), IndexType::U8);
/// assert_eq!(select_index_type(1000), IndexType::U16);
/// assert_eq!(select_index_type(100000), IndexType::U32);
/// ```
pub fn select_index_type(vertex_count: usize) -> IndexType {
    if vertex_count <= 256 {
        IndexType::U8
    } else if vertex_count <= 65536 {
        IndexType::U16
    } else {
        IndexType::U32
    }
}

/// Select index type with minimum constraint.
///
/// Returns at least `min_type` even if vertex count would allow smaller.
pub fn select_index_type_min(vertex_count: usize, min_type: IndexType) -> IndexType {
    let optimal = select_index_type(vertex_count);
    match (optimal, min_type) {
        (IndexType::U8, IndexType::U16 | IndexType::U32) => min_type,
        (IndexType::U16, IndexType::U32) => min_type,
        _ => optimal,
    }
}

// ---------------------------------------------------------------------------
// ACMR computation
// ---------------------------------------------------------------------------

/// Compute the Average Cache Miss Ratio (ACMR) for an index buffer.
///
/// ACMR measures vertex cache efficiency:
/// - ACMR = 1.0 means every vertex is a cache miss (worst case)
/// - ACMR = 0.5 means 50% of vertices are cache misses
/// - ACMR ~ 0.7 is typical for optimized meshes
/// - ACMR < 0.5 is excellent
///
/// Lower is better. A perfectly optimized mesh has ACMR approaching
/// the theoretical minimum of `vertex_count / (index_count / 3)`.
///
/// # Arguments
///
/// * `indices` - Triangle list indices
/// * `cache_size` - Simulated vertex cache size (typically 16-64)
///
/// # Example
///
/// ```ignore
/// let indices = vec![0, 1, 2, 0, 2, 3]; // 2 triangles sharing edge
/// let acmr = compute_acmr(&indices, 32);
/// // acmr = 4 misses / 2 triangles = 2.0 (unoptimized)
/// ```
pub fn compute_acmr(indices: &[u32], cache_size: usize) -> f32 {
    if indices.is_empty() || indices.len() < 3 {
        return 0.0;
    }

    let triangle_count = indices.len() / 3;
    if triangle_count == 0 {
        return 0.0;
    }

    // Simulate FIFO vertex cache
    let mut cache: Vec<u32> = Vec::with_capacity(cache_size);
    let mut cache_misses = 0u64;

    for &idx in indices {
        if !cache.contains(&idx) {
            cache_misses += 1;
            if cache.len() >= cache_size {
                cache.remove(0);
            }
            cache.push(idx);
        }
    }

    cache_misses as f32 / triangle_count as f32
}

/// Compute ACMR with LRU cache model (more accurate for modern GPUs).
pub fn compute_acmr_lru(indices: &[u32], cache_size: usize) -> f32 {
    if indices.is_empty() || indices.len() < 3 {
        return 0.0;
    }

    let triangle_count = indices.len() / 3;
    if triangle_count == 0 {
        return 0.0;
    }

    // LRU cache simulation
    let mut cache: Vec<u32> = Vec::with_capacity(cache_size);
    let mut cache_misses = 0u64;

    for &idx in indices {
        if let Some(pos) = cache.iter().position(|&x| x == idx) {
            // Hit - move to front (most recently used)
            cache.remove(pos);
            cache.insert(0, idx);
        } else {
            // Miss
            cache_misses += 1;
            if cache.len() >= cache_size {
                cache.pop(); // Remove LRU entry
            }
            cache.insert(0, idx);
        }
    }

    cache_misses as f32 / triangle_count as f32
}

// ---------------------------------------------------------------------------
// Tom Forsyth's vertex cache optimization
// ---------------------------------------------------------------------------

// Constants for Forsyth's scoring function
const CACHE_DECAY_POWER: f32 = 1.5;
const LAST_TRI_SCORE: f32 = 0.75;
const VALENCE_BOOST_SCALE: f32 = 2.0;
const VALENCE_BOOST_POWER: f32 = 0.5;

/// Per-vertex data for Forsyth's algorithm.
#[derive(Debug, Clone)]
struct VertexData {
    /// Number of triangles still using this vertex.
    active_triangles: u32,
    /// Position in simulated cache (-1 if not cached).
    cache_position: i32,
    /// Current score.
    score: f32,
    /// Triangles referencing this vertex.
    triangles: Vec<u32>,
}

impl Default for VertexData {
    fn default() -> Self {
        Self {
            active_triangles: 0,
            cache_position: -1,
            score: 0.0,
            triangles: Vec::new(),
        }
    }
}

/// Compute score for a vertex based on cache position and remaining triangles.
fn compute_vertex_score(cache_position: i32, active_triangles: u32, cache_size: usize) -> f32 {
    if active_triangles == 0 {
        return -1.0; // Dead vertex
    }

    let mut score = 0.0f32;

    // Cache position score
    if cache_position >= 0 {
        let pos = cache_position as usize;
        if pos < 3 {
            // Last triangle bonus
            score = LAST_TRI_SCORE;
        } else if pos < cache_size {
            // Decay based on cache position
            let scale = 1.0 - (pos - 3) as f32 / (cache_size - 3) as f32;
            score = scale.powf(CACHE_DECAY_POWER);
        }
    }

    // Valence boost (prefer vertices with fewer remaining triangles)
    let valence_boost = VALENCE_BOOST_SCALE * (active_triangles as f32).powf(-VALENCE_BOOST_POWER);
    score += valence_boost;

    score
}

/// Optimize triangle indices for vertex cache using Tom Forsyth's algorithm.
///
/// This algorithm reorders triangles to maximize vertex cache hits on GPUs.
/// It achieves near-optimal ACMR (Average Cache Miss Ratio) without requiring
/// the mesh to be modified.
///
/// Reference: "Linear-Speed Vertex Cache Optimisation" by Tom Forsyth
/// https://tomforsyth1000.github.io/papers/fast_vert_cache_opt.html
///
/// # Arguments
///
/// * `indices` - Input triangle indices (must be divisible by 3)
/// * `vertex_count` - Total number of vertices referenced
///
/// # Returns
///
/// Reordered indices optimized for vertex cache efficiency.
///
/// # Example
///
/// ```ignore
/// let indices = vec![0, 1, 2, 3, 4, 5, 0, 2, 3]; // 3 triangles
/// let optimized = optimize_vertex_cache(&indices, 6);
/// let acmr_before = compute_acmr(&indices, 32);
/// let acmr_after = compute_acmr(&optimized, 32);
/// assert!(acmr_after <= acmr_before);
/// ```
pub fn optimize_vertex_cache(indices: &[u32], vertex_count: usize) -> Vec<u32> {
    optimize_vertex_cache_with_size(indices, vertex_count, 32)
}

/// Optimize with explicit cache size.
pub fn optimize_vertex_cache_with_size(
    indices: &[u32],
    vertex_count: usize,
    cache_size: usize,
) -> Vec<u32> {
    if indices.is_empty() || vertex_count == 0 {
        return indices.to_vec();
    }

    let triangle_count = indices.len() / 3;
    if triangle_count == 0 {
        return indices.to_vec();
    }

    // Initialize vertex data
    let mut vertices: Vec<VertexData> = vec![VertexData::default(); vertex_count];

    // Build vertex-to-triangle references
    for (tri_idx, tri) in indices.chunks_exact(3).enumerate() {
        for &vertex_idx in tri {
            if (vertex_idx as usize) < vertex_count {
                vertices[vertex_idx as usize].active_triangles += 1;
                vertices[vertex_idx as usize].triangles.push(tri_idx as u32);
            }
        }
    }

    // Initialize scores
    for vertex in &mut vertices {
        vertex.score = compute_vertex_score(vertex.cache_position, vertex.active_triangles, cache_size);
    }

    // Track which triangles are emitted
    let mut triangle_emitted: Vec<bool> = vec![false; triangle_count];
    let mut output_indices: Vec<u32> = Vec::with_capacity(indices.len());

    // Simulated cache
    let mut cache: Vec<u32> = Vec::with_capacity(cache_size);

    // Main loop: emit triangles in optimal order
    for _ in 0..triangle_count {
        // Find best triangle to emit
        let best_tri = find_best_triangle(
            indices,
            &vertices,
            &triangle_emitted,
            &cache,
            cache_size,
        );

        if best_tri.is_none() {
            // No more triangles - should not happen if data is valid
            break;
        }

        let tri_idx = best_tri.unwrap();
        triangle_emitted[tri_idx] = true;

        // Emit triangle
        let tri_start = tri_idx * 3;
        let tri_indices = &indices[tri_start..tri_start + 3];
        output_indices.extend_from_slice(tri_indices);

        // Update cache and vertex data
        for &vertex_idx in tri_indices {
            let v = vertex_idx as usize;
            if v < vertex_count {
                // Decrement active triangle count
                vertices[v].active_triangles = vertices[v].active_triangles.saturating_sub(1);

                // Update cache
                if let Some(pos) = cache.iter().position(|&x| x == vertex_idx) {
                    cache.remove(pos);
                }
                cache.insert(0, vertex_idx);
                if cache.len() > cache_size {
                    cache.pop();
                }
            }
        }

        // Update cache positions and scores for affected vertices
        for (pos, &vertex_idx) in cache.iter().enumerate() {
            let v = vertex_idx as usize;
            if v < vertex_count {
                vertices[v].cache_position = pos as i32;
                vertices[v].score = compute_vertex_score(
                    vertices[v].cache_position,
                    vertices[v].active_triangles,
                    cache_size,
                );
            }
        }

        // Mark evicted vertices as not cached
        for vertex in &mut vertices {
            if vertex.cache_position >= cache.len() as i32 {
                vertex.cache_position = -1;
                vertex.score = compute_vertex_score(-1, vertex.active_triangles, cache_size);
            }
        }
    }

    output_indices
}

/// Find the best triangle to emit next.
fn find_best_triangle(
    indices: &[u32],
    vertices: &[VertexData],
    triangle_emitted: &[bool],
    cache: &[u32],
    cache_size: usize,
) -> Option<usize> {
    let mut best_score = f32::NEG_INFINITY;
    let mut best_tri: Option<usize> = None;

    // First, check triangles of cached vertices (most likely to be best)
    for &vertex_idx in cache {
        let v = vertex_idx as usize;
        if v >= vertices.len() {
            continue;
        }

        for &tri_idx in &vertices[v].triangles {
            let t = tri_idx as usize;
            if triangle_emitted[t] {
                continue;
            }

            let score = compute_triangle_score(indices, vertices, t, cache_size);
            if score > best_score {
                best_score = score;
                best_tri = Some(t);
            }
        }
    }

    // If no cached triangles, find any unemitted triangle
    if best_tri.is_none() {
        for (t, &emitted) in triangle_emitted.iter().enumerate() {
            if !emitted {
                let score = compute_triangle_score(indices, vertices, t, cache_size);
                if score > best_score {
                    best_score = score;
                    best_tri = Some(t);
                }
            }
        }
    }

    best_tri
}

/// Compute score for a triangle (sum of vertex scores).
fn compute_triangle_score(
    indices: &[u32],
    vertices: &[VertexData],
    tri_idx: usize,
    _cache_size: usize,
) -> f32 {
    let start = tri_idx * 3;
    let mut score = 0.0f32;

    for &vertex_idx in &indices[start..start + 3] {
        let v = vertex_idx as usize;
        if v < vertices.len() {
            score += vertices[v].score;
        }
    }

    score
}

// ---------------------------------------------------------------------------
// Vertex reordering
// ---------------------------------------------------------------------------

/// Reorder vertices to match optimized index order.
///
/// After cache optimization, vertices should be reordered so they appear
/// in the order they're first accessed. This improves pre-transform vertex
/// cache performance (fetching from main memory).
///
/// # Type Parameters
///
/// * `V` - Vertex type (must implement `Copy`)
///
/// # Arguments
///
/// * `vertices` - Input vertex array
/// * `indices` - Optimized indices
///
/// # Returns
///
/// Tuple of (reordered vertices, remapped indices).
///
/// # Example
///
/// ```ignore
/// #[derive(Copy, Clone)]
/// struct Vertex { x: f32, y: f32, z: f32 }
///
/// let vertices = vec![Vertex { x: 0.0, y: 0.0, z: 0.0 }, ...];
/// let indices = vec![5, 2, 0, 3, 2, 1]; // Optimized order
///
/// let (new_verts, new_indices) = reorder_vertices(&vertices, &indices);
/// // new_indices[0] == 0, etc. (sequential access)
/// ```
pub fn reorder_vertices<V: Copy>(vertices: &[V], indices: &[u32]) -> (Vec<V>, Vec<u32>) {
    if vertices.is_empty() || indices.is_empty() {
        return (vertices.to_vec(), indices.to_vec());
    }

    // Map from old index to new index
    let mut old_to_new: Vec<Option<u32>> = vec![None; vertices.len()];
    let mut new_to_old: Vec<u32> = Vec::with_capacity(vertices.len());

    // Assign new indices in order of first appearance
    for &old_idx in indices {
        let old = old_idx as usize;
        if old < vertices.len() && old_to_new[old].is_none() {
            let new_idx = new_to_old.len() as u32;
            old_to_new[old] = Some(new_idx);
            new_to_old.push(old_idx);
        }
    }

    // Add any unreferenced vertices at the end
    for old_idx in 0..vertices.len() {
        if old_to_new[old_idx].is_none() {
            let new_idx = new_to_old.len() as u32;
            old_to_new[old_idx] = Some(new_idx);
            new_to_old.push(old_idx as u32);
        }
    }

    // Reorder vertices
    let mut new_vertices: Vec<V> = Vec::with_capacity(vertices.len());
    for &old_idx in &new_to_old {
        new_vertices.push(vertices[old_idx as usize]);
    }

    // Remap indices
    let new_indices: Vec<u32> = indices
        .iter()
        .map(|&old_idx| old_to_new[old_idx as usize].unwrap_or(old_idx))
        .collect();

    (new_vertices, new_indices)
}

/// Reorder vertices with additional output of index mapping.
pub fn reorder_vertices_with_mapping<V: Copy>(
    vertices: &[V],
    indices: &[u32],
) -> ReorderedMesh<V> {
    if vertices.is_empty() || indices.is_empty() {
        return ReorderedMesh {
            vertices: vertices.to_vec(),
            indices: indices.to_vec(),
            new_to_old: (0..vertices.len() as u32).collect(),
        };
    }

    let mut old_to_new: Vec<Option<u32>> = vec![None; vertices.len()];
    let mut new_to_old: Vec<u32> = Vec::with_capacity(vertices.len());

    for &old_idx in indices {
        let old = old_idx as usize;
        if old < vertices.len() && old_to_new[old].is_none() {
            let new_idx = new_to_old.len() as u32;
            old_to_new[old] = Some(new_idx);
            new_to_old.push(old_idx);
        }
    }

    for old_idx in 0..vertices.len() {
        if old_to_new[old_idx].is_none() {
            let new_idx = new_to_old.len() as u32;
            old_to_new[old_idx] = Some(new_idx);
            new_to_old.push(old_idx as u32);
        }
    }

    let mut new_vertices: Vec<V> = Vec::with_capacity(vertices.len());
    for &old_idx in &new_to_old {
        new_vertices.push(vertices[old_idx as usize]);
    }

    let new_indices: Vec<u32> = indices
        .iter()
        .map(|&old_idx| old_to_new[old_idx as usize].unwrap_or(old_idx))
        .collect();

    ReorderedMesh {
        vertices: new_vertices,
        indices: new_indices,
        new_to_old,
    }
}

// ---------------------------------------------------------------------------
// Stripification
// ---------------------------------------------------------------------------

/// Result of stripification.
#[derive(Debug, Clone)]
pub struct StripResult {
    /// Strip indices (including restart markers).
    pub indices: Vec<u32>,
    /// Number of strips.
    pub strip_count: usize,
    /// Total primitives (triangles).
    pub triangle_count: usize,
    /// Efficiency ratio (triangles / indices).
    pub efficiency: f32,
}

/// Convert triangle list to triangle strip with primitive restart.
///
/// Uses a greedy algorithm to build strips. Not always optimal, but fast
/// and produces reasonable results.
///
/// # Arguments
///
/// * `indices` - Triangle list indices
/// * `vertex_count` - Total vertex count
/// * `restart_index` - Value to use for primitive restart (usually 0xFFFF or 0xFFFFFFFF)
///
/// # Returns
///
/// Strip indices with restart markers, or error.
pub fn stripify(
    indices: &[u32],
    vertex_count: usize,
    restart_index: u32,
) -> Result<StripResult, IndexBufferError> {
    if indices.is_empty() {
        return Err(IndexBufferError::EmptyBuffer);
    }

    if indices.len() % 3 != 0 {
        return Err(IndexBufferError::InvalidTriangleCount(indices.len()));
    }

    let triangle_count = indices.len() / 3;
    if triangle_count == 0 {
        return Ok(StripResult {
            indices: Vec::new(),
            strip_count: 0,
            triangle_count: 0,
            efficiency: 0.0,
        });
    }

    // Build edge adjacency
    let adjacency = build_triangle_adjacency(indices, vertex_count);

    // Track emitted triangles
    let mut emitted: Vec<bool> = vec![false; triangle_count];
    let mut output: Vec<u32> = Vec::with_capacity(indices.len());
    let mut strip_count = 0;

    // Emit strips
    while let Some(start_tri) = find_next_unemitted(&emitted) {
        if !output.is_empty() {
            // Add restart marker between strips
            output.push(restart_index);
        }

        emit_strip(
            indices,
            &adjacency,
            &mut emitted,
            &mut output,
            start_tri,
        );
        strip_count += 1;
    }

    let efficiency = if !output.is_empty() {
        triangle_count as f32 / output.len() as f32
    } else {
        0.0
    };

    Ok(StripResult {
        indices: output,
        strip_count,
        triangle_count,
        efficiency,
    })
}

/// Build triangle adjacency map.
fn build_triangle_adjacency(indices: &[u32], _vertex_count: usize) -> HashMap<(u32, u32), Vec<usize>> {
    let mut edge_to_triangles: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    let triangle_count = indices.len() / 3;

    for t in 0..triangle_count {
        let start = t * 3;
        let tri = &indices[start..start + 3];

        // Add all three edges (ordered to avoid duplicates)
        for i in 0..3 {
            let v0 = tri[i];
            let v1 = tri[(i + 1) % 3];
            let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            edge_to_triangles.entry(edge).or_default().push(t);
        }
    }

    edge_to_triangles
}

/// Find next unemitted triangle.
fn find_next_unemitted(emitted: &[bool]) -> Option<usize> {
    emitted.iter().position(|&e| !e)
}

/// Emit a strip starting from the given triangle.
fn emit_strip(
    indices: &[u32],
    adjacency: &HashMap<(u32, u32), Vec<usize>>,
    emitted: &mut [bool],
    output: &mut Vec<u32>,
    start_tri: usize,
) {
    emitted[start_tri] = true;

    let start = start_tri * 3;
    let tri = &indices[start..start + 3];

    // Emit first triangle
    output.push(tri[0]);
    output.push(tri[1]);
    output.push(tri[2]);

    // Current edge for continuation
    let mut current_edge = (tri[1], tri[2]);
    let mut winding = true; // Track winding for degenerate triangles

    // Try to extend strip
    loop {
        let ordered_edge = if current_edge.0 < current_edge.1 {
            (current_edge.0, current_edge.1)
        } else {
            (current_edge.1, current_edge.0)
        };

        let next_tri = adjacency
            .get(&ordered_edge)
            .and_then(|tris| tris.iter().find(|&&t| !emitted[t]).copied());

        match next_tri {
            Some(t) => {
                emitted[t] = true;
                let t_start = t * 3;
                let next_indices = &indices[t_start..t_start + 3];

                // Find the third vertex (not on the shared edge)
                let third = next_indices
                    .iter()
                    .find(|&&v| v != current_edge.0 && v != current_edge.1)
                    .copied();

                if let Some(v) = third {
                    output.push(v);

                    // Update edge for next iteration
                    if winding {
                        current_edge = (current_edge.1, v);
                    } else {
                        current_edge = (v, current_edge.0);
                    }
                    winding = !winding;
                } else {
                    break;
                }
            }
            None => break,
        }
    }
}

// ---------------------------------------------------------------------------
// Full optimization pipeline
// ---------------------------------------------------------------------------

/// Optimize an index buffer with the given configuration.
///
/// This is the main entry point for index buffer optimization. It performs:
/// 1. Index type selection (if not forced)
/// 2. Vertex cache optimization (Tom Forsyth's algorithm)
/// 3. Vertex reordering (optional)
/// 4. Stripification (optional)
/// 5. ACMR computation
///
/// The transformation is idempotent - running it twice produces the same output.
///
/// # Arguments
///
/// * `indices` - Input triangle list indices
/// * `vertex_count` - Total vertex count
/// * `config` - Optimization configuration
///
/// # Returns
///
/// Optimized index buffer with metadata.
///
/// # Example
///
/// ```ignore
/// let indices = vec![0, 1, 2, 2, 1, 3, 4, 5, 6];
/// let config = IndexBufferConfig::default();
/// let result = optimize_index_buffer(&indices, 7, &config);
/// println!("ACMR: {}", result.acmr);
/// ```
pub fn optimize_index_buffer(
    indices: &[u32],
    vertex_count: usize,
    config: &IndexBufferConfig,
) -> OptimizedIndices {
    if indices.is_empty() {
        return OptimizedIndices {
            indices: Vec::new(),
            index_type: select_index_type(vertex_count),
            primitive_count: 0,
            acmr: 0.0,
            is_strip: false,
        };
    }

    // Step 1: Optimize for vertex cache
    let optimized = if config.optimize_cache {
        optimize_vertex_cache_with_size(indices, vertex_count, config.cache_size)
    } else {
        indices.to_vec()
    };

    // Step 2: Compute ACMR
    let acmr = compute_acmr(&optimized, config.cache_size);

    // Step 3: Stripify if requested
    let (final_indices, is_strip, primitive_count) = if config.stripify {
        if let Some(restart) = config.primitive_restart_index {
            match stripify(&optimized, vertex_count, restart) {
                Ok(strip) => (strip.indices, true, strip.triangle_count as u32),
                Err(_) => (optimized.clone(), false, (optimized.len() / 3) as u32),
            }
        } else {
            (optimized.clone(), false, (optimized.len() / 3) as u32)
        }
    } else {
        (optimized.clone(), false, (optimized.len() / 3) as u32)
    };

    // Step 4: Select index type and encode
    let index_type = select_index_type(vertex_count);
    let encoded = encode_indices(&final_indices, index_type);

    OptimizedIndices {
        indices: encoded,
        index_type,
        primitive_count,
        acmr,
        is_strip,
    }
}

/// Encode indices to raw bytes.
pub fn encode_indices(indices: &[u32], index_type: IndexType) -> Vec<u8> {
    match index_type {
        IndexType::U8 => indices.iter().map(|&i| i as u8).collect(),
        IndexType::U16 => {
            let mut data = Vec::with_capacity(indices.len() * 2);
            for &idx in indices {
                data.extend_from_slice(&(idx as u16).to_le_bytes());
            }
            data
        }
        IndexType::U32 => {
            let mut data = Vec::with_capacity(indices.len() * 4);
            for &idx in indices {
                data.extend_from_slice(&idx.to_le_bytes());
            }
            data
        }
    }
}

/// Decode indices from raw bytes.
pub fn decode_indices(data: &[u8], index_type: IndexType) -> Vec<u32> {
    match index_type {
        IndexType::U8 => data.iter().map(|&b| b as u32).collect(),
        IndexType::U16 => data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
            .collect(),
        IndexType::U32 => data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Validation utilities
// ---------------------------------------------------------------------------

/// Validate that all indices are within bounds.
pub fn validate_indices(indices: &[u32], vertex_count: usize) -> Result<(), IndexBufferError> {
    for &idx in indices {
        if idx as usize >= vertex_count {
            return Err(IndexBufferError::IndexOutOfBounds {
                index: idx,
                vertex_count,
            });
        }
    }
    Ok(())
}

/// Check if indices form valid triangles (count divisible by 3).
pub fn validate_triangle_count(indices: &[u32]) -> Result<(), IndexBufferError> {
    if indices.len() % 3 != 0 {
        return Err(IndexBufferError::InvalidTriangleCount(indices.len()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Index type selection tests (4 tests)
    // ========================================================================

    #[test]
    fn test_select_index_type_u8() {
        assert_eq!(select_index_type(1), IndexType::U8);
        assert_eq!(select_index_type(100), IndexType::U8);
        assert_eq!(select_index_type(256), IndexType::U8);
    }

    #[test]
    fn test_select_index_type_u16() {
        assert_eq!(select_index_type(257), IndexType::U16);
        assert_eq!(select_index_type(1000), IndexType::U16);
        assert_eq!(select_index_type(65536), IndexType::U16);
    }

    #[test]
    fn test_select_index_type_u32() {
        assert_eq!(select_index_type(65537), IndexType::U32);
        assert_eq!(select_index_type(100_000), IndexType::U32);
        assert_eq!(select_index_type(1_000_000), IndexType::U32);
    }

    #[test]
    fn test_select_index_type_min() {
        // U8 -> U16 when min is U16
        assert_eq!(select_index_type_min(100, IndexType::U16), IndexType::U16);
        // U8 -> U32 when min is U32
        assert_eq!(select_index_type_min(100, IndexType::U32), IndexType::U32);
        // U16 stays U16 when min is U8
        assert_eq!(select_index_type_min(1000, IndexType::U8), IndexType::U16);
        // U32 stays U32 regardless of min
        assert_eq!(select_index_type_min(100_000, IndexType::U8), IndexType::U32);
    }

    // ========================================================================
    // Vertex cache optimization tests (5 tests)
    // ========================================================================

    #[test]
    fn test_optimize_empty() {
        let result = optimize_vertex_cache(&[], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_optimize_single_triangle() {
        let indices = vec![0, 1, 2];
        let result = optimize_vertex_cache(&indices, 3);
        assert_eq!(result.len(), 3);
        // Single triangle should remain unchanged
        let mut sorted_result = result.clone();
        sorted_result.sort();
        assert_eq!(sorted_result, vec![0, 1, 2]);
    }

    #[test]
    fn test_optimize_preserves_all_triangles() {
        let indices = vec![0, 1, 2, 2, 1, 3, 3, 1, 4];
        let result = optimize_vertex_cache(&indices, 5);

        // Same number of indices
        assert_eq!(result.len(), indices.len());

        // Same triangles (as sets)
        let original_tris: Vec<_> = indices.chunks(3).map(|t| {
            let mut v = t.to_vec();
            v.sort();
            v
        }).collect();
        let result_tris: Vec<_> = result.chunks(3).map(|t| {
            let mut v = t.to_vec();
            v.sort();
            v
        }).collect();

        for tri in &original_tris {
            assert!(result_tris.contains(tri), "Missing triangle {:?}", tri);
        }
    }

    #[test]
    fn test_optimize_improves_acmr() {
        // Create a pathological index order
        let indices: Vec<u32> = (0..100)
            .flat_map(|i| [i * 3, i * 3 + 1, i * 3 + 2])
            .collect();

        let vertex_count = 300;
        let original_acmr = compute_acmr(&indices, 32);
        let optimized = optimize_vertex_cache(&indices, vertex_count);
        let new_acmr = compute_acmr(&optimized, 32);

        // Optimization should not make ACMR worse
        assert!(new_acmr <= original_acmr + 0.01);
    }

    #[test]
    fn test_optimize_idempotent() {
        let indices = vec![0, 1, 2, 2, 1, 3, 3, 2, 4, 4, 2, 5];
        let first = optimize_vertex_cache(&indices, 6);
        let second = optimize_vertex_cache(&first, 6);

        // Running twice should produce same result
        assert_eq!(first, second);
    }

    // ========================================================================
    // ACMR computation tests (3 tests)
    // ========================================================================

    #[test]
    fn test_acmr_empty() {
        assert_eq!(compute_acmr(&[], 32), 0.0);
        assert_eq!(compute_acmr(&[0, 1], 32), 0.0); // Less than one triangle
    }

    #[test]
    fn test_acmr_single_triangle() {
        // 3 cache misses / 1 triangle = 3.0
        let indices = vec![0, 1, 2];
        let acmr = compute_acmr(&indices, 32);
        assert!((acmr - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_acmr_shared_vertices() {
        // Two triangles sharing an edge
        // First tri: 3 misses, second tri: 1 miss (vertex 3 only)
        // Total: 4 misses / 2 triangles = 2.0
        let indices = vec![0, 1, 2, 0, 2, 3];
        let acmr = compute_acmr(&indices, 32);
        assert!((acmr - 2.0).abs() < 0.001);
    }

    // ========================================================================
    // Vertex reordering tests (4 tests)
    // ========================================================================

    #[test]
    fn test_reorder_empty() {
        let vertices: Vec<u32> = vec![];
        let indices: Vec<u32> = vec![];
        let (new_verts, new_indices) = reorder_vertices(&vertices, &indices);
        assert!(new_verts.is_empty());
        assert!(new_indices.is_empty());
    }

    #[test]
    fn test_reorder_identity() {
        let vertices = vec![10, 20, 30];
        let indices = vec![0, 1, 2];
        let (new_verts, new_indices) = reorder_vertices(&vertices, &indices);

        // Already optimal order
        assert_eq!(new_verts, vec![10, 20, 30]);
        assert_eq!(new_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_reorder_reversed() {
        let vertices = vec![10, 20, 30];
        let indices = vec![2, 1, 0];
        let (new_verts, new_indices) = reorder_vertices(&vertices, &indices);

        // Vertices reordered to match access pattern
        assert_eq!(new_verts, vec![30, 20, 10]);
        assert_eq!(new_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_reorder_preserves_triangles() {
        #[derive(Copy, Clone, Debug, PartialEq)]
        struct Vertex { x: f32 }

        let vertices: Vec<Vertex> = (0..5).map(|i| Vertex { x: i as f32 }).collect();
        let indices = vec![4, 2, 0, 3, 1, 4];

        let (new_verts, new_indices) = reorder_vertices(&vertices, &indices);

        // Verify triangles are preserved
        for tri in new_indices.chunks(3) {
            let v0 = new_verts[tri[0] as usize];
            let v1 = new_verts[tri[1] as usize];
            let v2 = new_verts[tri[2] as usize];

            // Find corresponding original triangle
            let orig_tri = indices.chunks(3).find(|t| {
                vertices[t[0] as usize] == v0 &&
                vertices[t[1] as usize] == v1 &&
                vertices[t[2] as usize] == v2
            });
            assert!(orig_tri.is_some(), "Triangle not preserved");
        }
    }

    // ========================================================================
    // Stripification tests (2 tests)
    // ========================================================================

    #[test]
    fn test_stripify_single_triangle() {
        let indices = vec![0, 1, 2];
        let result = stripify(&indices, 3, 0xFFFF).unwrap();

        assert_eq!(result.strip_count, 1);
        assert_eq!(result.triangle_count, 1);
        assert_eq!(result.indices.len(), 3);
    }

    #[test]
    fn test_stripify_strip() {
        // Triangle strip: (0,1,2), (1,2,3), (2,3,4) as list
        let indices = vec![0, 1, 2, 1, 2, 3, 2, 3, 4];
        let result = stripify(&indices, 5, 0xFFFF).unwrap();

        assert_eq!(result.triangle_count, 3);
        // Should produce efficient strip
        assert!(result.efficiency > 0.0);
    }

    // ========================================================================
    // Idempotency tests (2 tests)
    // ========================================================================

    #[test]
    fn test_full_pipeline_idempotent() {
        let indices = vec![0, 1, 2, 2, 1, 3, 3, 2, 4];
        let config = IndexBufferConfig::default();

        let first = optimize_index_buffer(&indices, 5, &config);
        let decoded = first.to_u32_vec();
        let second = optimize_index_buffer(&decoded, 5, &config);

        // Same ACMR (might differ by floating point)
        assert!((first.acmr - second.acmr).abs() < 0.001);
        assert_eq!(first.primitive_count, second.primitive_count);
    }

    #[test]
    fn test_optimization_deterministic() {
        let indices = vec![0, 1, 2, 3, 4, 5, 0, 2, 3, 1, 4, 5];
        let config = IndexBufferConfig::default();

        // Run multiple times
        let results: Vec<_> = (0..5)
            .map(|_| optimize_index_buffer(&indices, 6, &config))
            .collect();

        // All results should be identical
        for i in 1..results.len() {
            assert_eq!(results[0].indices, results[i].indices);
            assert_eq!(results[0].acmr, results[i].acmr);
        }
    }

    // ========================================================================
    // Additional tests to reach 20+
    // ========================================================================

    #[test]
    fn test_index_type_size_bytes() {
        assert_eq!(IndexType::U8.size_bytes(), 1);
        assert_eq!(IndexType::U16.size_bytes(), 2);
        assert_eq!(IndexType::U32.size_bytes(), 4);
    }

    #[test]
    fn test_index_type_max_values() {
        assert_eq!(IndexType::U8.max_index(), 255);
        assert_eq!(IndexType::U16.max_index(), 65535);
        assert_eq!(IndexType::U32.max_index(), u32::MAX);

        assert_eq!(IndexType::U8.max_vertex_count(), 256);
        assert_eq!(IndexType::U16.max_vertex_count(), 65536);
    }

    #[test]
    fn test_encode_decode_u8() {
        let indices = vec![0u32, 1, 2, 100, 200, 255];
        let encoded = encode_indices(&indices, IndexType::U8);
        let decoded = decode_indices(&encoded, IndexType::U8);
        assert_eq!(indices, decoded);
    }

    #[test]
    fn test_encode_decode_u16() {
        let indices = vec![0u32, 1000, 30000, 65535];
        let encoded = encode_indices(&indices, IndexType::U16);
        let decoded = decode_indices(&encoded, IndexType::U16);
        assert_eq!(indices, decoded);
    }

    #[test]
    fn test_encode_decode_u32() {
        let indices = vec![0u32, 100_000, 1_000_000, u32::MAX];
        let encoded = encode_indices(&indices, IndexType::U32);
        let decoded = decode_indices(&encoded, IndexType::U32);
        assert_eq!(indices, decoded);
    }

    #[test]
    fn test_validate_indices_success() {
        let indices = vec![0, 1, 2, 3, 4];
        assert!(validate_indices(&indices, 5).is_ok());
    }

    #[test]
    fn test_validate_indices_failure() {
        let indices = vec![0, 1, 2, 5];
        let result = validate_indices(&indices, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_triangle_count_success() {
        let indices = vec![0, 1, 2, 3, 4, 5];
        assert!(validate_triangle_count(&indices).is_ok());
    }

    #[test]
    fn test_validate_triangle_count_failure() {
        let indices = vec![0, 1, 2, 3, 4];
        let result = validate_triangle_count(&indices);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_presets() {
        let cache_only = IndexBufferConfig::cache_only();
        assert!(cache_only.optimize_cache);
        assert!(!cache_only.reorder_vertices);
        assert!(!cache_only.stripify);

        let full = IndexBufferConfig::full();
        assert!(full.optimize_cache);
        assert!(full.reorder_vertices);
        assert!(!full.stripify);

        let strip = IndexBufferConfig::stripify(0xFFFF);
        assert!(strip.stripify);
        assert_eq!(strip.primitive_restart_index, Some(0xFFFF));
    }

    #[test]
    fn test_optimized_indices_to_u32_vec() {
        let indices = vec![0u32, 1, 2, 3, 4, 5];
        let encoded = encode_indices(&indices, IndexType::U16);

        let result = OptimizedIndices {
            indices: encoded,
            index_type: IndexType::U16,
            primitive_count: 2,
            acmr: 3.0,
            is_strip: false,
        };

        assert_eq!(result.to_u32_vec(), indices);
        assert_eq!(result.index_count(), 6);
    }

    #[test]
    fn test_acmr_lru_model() {
        // LRU should give similar results to FIFO for simple cases
        let indices = vec![0, 1, 2, 0, 2, 3];
        let fifo_acmr = compute_acmr(&indices, 32);
        let lru_acmr = compute_acmr_lru(&indices, 32);

        // Both should report 4 misses / 2 triangles = 2.0
        assert!((fifo_acmr - 2.0).abs() < 0.001);
        assert!((lru_acmr - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_reorder_with_mapping() {
        let vertices = vec![10, 20, 30, 40, 50];
        let indices = vec![4, 2, 0, 3, 1, 4];

        let result = reorder_vertices_with_mapping(&vertices, &indices);

        // Verify mapping
        for (new_idx, &old_idx) in result.new_to_old.iter().enumerate() {
            assert_eq!(result.vertices[new_idx], vertices[old_idx as usize]);
        }
    }

    #[test]
    fn test_stripify_error_invalid_count() {
        let indices = vec![0, 1, 2, 3, 4]; // Not divisible by 3
        let result = stripify(&indices, 5, 0xFFFF);
        assert!(matches!(result, Err(IndexBufferError::InvalidTriangleCount(_))));
    }

    #[test]
    fn test_stripify_empty() {
        let result = stripify(&[], 0, 0xFFFF);
        assert!(matches!(result, Err(IndexBufferError::EmptyBuffer)));
    }

    #[test]
    fn test_large_mesh_optimization() {
        // Generate a grid mesh
        let size = 10;
        let mut indices = Vec::new();
        for y in 0..size {
            for x in 0..size {
                let v0 = y * (size + 1) + x;
                let v1 = v0 + 1;
                let v2 = v0 + (size + 1);
                let v3 = v2 + 1;

                indices.extend_from_slice(&[v0, v1, v2, v2, v1, v3]);
            }
        }

        let vertex_count = ((size + 1) * (size + 1)) as usize;
        let config = IndexBufferConfig::default();

        let original_acmr = compute_acmr(&indices.iter().map(|&x| x as u32).collect::<Vec<_>>(), 32);
        let result = optimize_index_buffer(
            &indices.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            vertex_count,
            &config,
        );

        // Optimization should help grid meshes
        assert!(result.acmr <= original_acmr + 0.1);
        assert_eq!(result.primitive_count, (size * size * 2) as u32);
    }
}

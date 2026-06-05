//! Meshlet Generator for GPU-driven rendering (T-WGPU-P6.9.2).
//!
//! This module provides a configurable meshlet generator that partitions indexed
//! triangle meshes into meshlets suitable for GPU-driven rendering with per-cluster
//! culling.
//!
//! # Overview
//!
//! The generator uses a greedy algorithm to split meshes into meshlets, each
//! containing at most 64 vertices and 124 triangles (default limits). For each
//! meshlet, it computes:
//!
//! - **Local indices**: Remapping of global vertex indices to local meshlet indices
//! - **Bounding sphere**: For frustum culling
//! - **Backface cone**: For backface culling
//!
//! # Algorithm
//!
//! The greedy algorithm processes triangles in order:
//!
//! 1. For each triangle, count how many new vertices it would add
//! 2. If adding the triangle would exceed limits, finalize current meshlet
//! 3. Add triangle to current meshlet, updating vertex tracking
//! 4. After all triangles, compute bounds and cones for each meshlet
//!
//! # Usage
//!
//! ```ignore
//! use renderer_backend::gpu_driven::meshlet_generator::{MeshletGenerator, MeshInput};
//!
//! let generator = MeshletGenerator::new();
//! let input = MeshInput {
//!     positions: &mesh.positions,
//!     indices: &mesh.indices,
//!     normals: Some(&mesh.normals),
//! };
//!
//! let output = generator.generate(&input);
//!
//! // Upload to GPU
//! let meshlet_buffer = create_buffer(&output.meshlets);
//! let bounds_buffer = create_buffer(&output.bounds);
//! let vertex_index_buffer = create_buffer(&output.vertex_indices);
//! let local_index_buffer = create_buffer(&output.triangle_indices);
//! ```
//!
//! # Performance Considerations
//!
//! - Work complexity: O(n) for n triangles
//! - Memory: O(v + t) where v = vertices, t = triangles
//! - The greedy algorithm is cache-friendly and fast
//! - For better spatial locality, consider sorting triangles before generation

use super::meshlet::{Meshlet, MeshletBounds};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum vertices per meshlet (GPU-friendly power of 2).
pub const MAX_MESHLET_VERTICES: usize = 64;

/// Maximum triangles per meshlet.
///
/// This is derived from the vertex limit: with 64 vertices in a triangle
/// strip-like arrangement, we can have at most ~124 triangles.
pub const MAX_MESHLET_TRIANGLES: usize = 124;

/// Epsilon for floating point comparisons.
const EPSILON: f32 = 1e-8;

// ---------------------------------------------------------------------------
// Input/Output types
// ---------------------------------------------------------------------------

/// Input mesh data for meshlet generation.
///
/// Contains references to the source mesh data. Positions are required,
/// normals are optional but improve backface cone accuracy.
#[derive(Debug, Clone, Copy)]
pub struct MeshInput<'a> {
    /// Vertex positions (must match index references).
    pub positions: &'a [[f32; 3]],
    /// Triangle indices (length must be multiple of 3).
    pub indices: &'a [u32],
    /// Optional vertex normals for improved cone computation.
    pub normals: Option<&'a [[f32; 3]]>,
}

impl<'a> MeshInput<'a> {
    /// Create a new mesh input with only positions and indices.
    pub fn new(positions: &'a [[f32; 3]], indices: &'a [u32]) -> Self {
        Self {
            positions,
            indices,
            normals: None,
        }
    }

    /// Create a mesh input with positions, indices, and normals.
    pub fn with_normals(
        positions: &'a [[f32; 3]],
        indices: &'a [u32],
        normals: &'a [[f32; 3]],
    ) -> Self {
        Self {
            positions,
            indices,
            normals: Some(normals),
        }
    }

    /// Returns the number of triangles in the mesh.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Returns true if the input is empty (no triangles).
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() || self.positions.is_empty()
    }

    /// Validate the input data.
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, `Err(reason)` otherwise.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.indices.len() % 3 != 0 {
            return Err("Index count must be multiple of 3");
        }

        for (i, &idx) in self.indices.iter().enumerate() {
            if idx as usize >= self.positions.len() {
                return Err("Index out of bounds");
            }
        }

        if let Some(normals) = self.normals {
            if normals.len() != self.positions.len() {
                return Err("Normals count must match positions count");
            }
        }

        Ok(())
    }
}

/// Output from meshlet generation.
///
/// Contains all the data needed to render the mesh using meshlets.
/// This struct owns the generated data and is ready for GPU upload.
#[derive(Debug, Clone, Default)]
pub struct MeshletOutput {
    /// Array of meshlet descriptors.
    pub meshlets: Vec<Meshlet>,
    /// Array of bounding data for each meshlet.
    pub bounds: Vec<MeshletBounds>,
    /// Global vertex indices (flattened across all meshlets).
    ///
    /// Each meshlet references a contiguous range of these indices,
    /// as specified by `meshlet.vertex_offset` and `meshlet.vertex_count`.
    pub vertex_indices: Vec<u32>,
    /// Local triangle indices (3 bytes per triangle, flattened).
    ///
    /// Each value is a local index (0-63) into the meshlet's vertex list.
    /// Each meshlet references a contiguous range of these indices,
    /// as specified by `meshlet.triangle_offset` and `meshlet.triangle_count * 3`.
    pub triangle_indices: Vec<u8>,
}

impl MeshletOutput {
    /// Create empty meshlet output.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of meshlets.
    pub fn meshlet_count(&self) -> usize {
        self.meshlets.len()
    }

    /// Returns true if there are no meshlets.
    pub fn is_empty(&self) -> bool {
        self.meshlets.is_empty()
    }

    /// Returns total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        use std::mem::size_of;
        self.meshlets.len() * size_of::<Meshlet>()
            + self.bounds.len() * size_of::<MeshletBounds>()
            + self.vertex_indices.len() * size_of::<u32>()
            + self.triangle_indices.len() * size_of::<u8>()
    }

    /// Reconstruct the original mesh indices from meshlet data.
    ///
    /// This is useful for validation and debugging. The reconstructed
    /// indices may be in a different order than the original.
    pub fn reconstruct_indices(&self) -> Vec<u32> {
        let mut result = Vec::new();

        for meshlet in &self.meshlets {
            let v_start = meshlet.vertex_offset as usize;
            let t_start = meshlet.triangle_offset as usize;
            let t_count = meshlet.triangle_count as usize;

            for t in 0..t_count {
                let local_base = t_start + t * 3;
                let i0 = self.triangle_indices[local_base] as usize;
                let i1 = self.triangle_indices[local_base + 1] as usize;
                let i2 = self.triangle_indices[local_base + 2] as usize;

                result.push(self.vertex_indices[v_start + i0]);
                result.push(self.vertex_indices[v_start + i1]);
                result.push(self.vertex_indices[v_start + i2]);
            }
        }

        result
    }

    /// Validate meshlet data integrity.
    ///
    /// # Arguments
    ///
    /// * `vertex_count` - Total number of vertices in the original mesh
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, `Err(reason)` otherwise.
    pub fn validate(&self, vertex_count: usize) -> Result<(), &'static str> {
        if self.meshlets.len() != self.bounds.len() {
            return Err("Meshlet count does not match bounds count");
        }

        for (i, meshlet) in self.meshlets.iter().enumerate() {
            let v_start = meshlet.vertex_offset as usize;
            let v_count = meshlet.vertex_count as usize;
            let t_start = meshlet.triangle_offset as usize;
            let t_count = meshlet.triangle_count as usize;

            // Check vertex indices are in range
            if v_start + v_count > self.vertex_indices.len() {
                return Err("Vertex offset + count exceeds vertex_indices length");
            }

            // Check local indices are in range
            if t_start + t_count * 3 > self.triangle_indices.len() {
                return Err("Triangle offset + count*3 exceeds triangle_indices length");
            }

            // Check vertex count limit
            if v_count > MAX_MESHLET_VERTICES {
                return Err("Vertex count exceeds MAX_MESHLET_VERTICES");
            }

            // Check triangle count limit
            if t_count > MAX_MESHLET_TRIANGLES {
                return Err("Triangle count exceeds MAX_MESHLET_TRIANGLES");
            }

            // Check global indices point to valid vertices
            for j in v_start..(v_start + v_count) {
                if self.vertex_indices[j] as usize >= vertex_count {
                    return Err("Vertex index out of range");
                }
            }

            // Check local indices point to valid local vertices
            for j in t_start..(t_start + t_count * 3) {
                if self.triangle_indices[j] >= v_count as u8 {
                    return Err("Local index out of range");
                }
            }

            // Validate bounds
            let b = &self.bounds[i];
            if b.radius < 0.0 {
                return Err("Negative bounding sphere radius");
            }
        }

        Ok(())
    }

    /// Get statistics about the meshlet output.
    pub fn stats(&self) -> MeshletStats {
        if self.meshlets.is_empty() {
            return MeshletStats::default();
        }

        let mut total_vertices = 0usize;
        let mut total_triangles = 0usize;
        let mut min_vertices = usize::MAX;
        let mut max_vertices = 0usize;
        let mut min_triangles = usize::MAX;
        let mut max_triangles = 0usize;

        for meshlet in &self.meshlets {
            let v = meshlet.vertex_count as usize;
            let t = meshlet.triangle_count as usize;

            total_vertices += v;
            total_triangles += t;
            min_vertices = min_vertices.min(v);
            max_vertices = max_vertices.max(v);
            min_triangles = min_triangles.min(t);
            max_triangles = max_triangles.max(t);
        }

        let count = self.meshlets.len();

        MeshletStats {
            meshlet_count: count,
            total_vertices,
            total_triangles,
            avg_vertices: total_vertices as f32 / count as f32,
            avg_triangles: total_triangles as f32 / count as f32,
            min_vertices,
            max_vertices,
            min_triangles,
            max_triangles,
            memory_bytes: self.memory_usage(),
        }
    }
}

/// Statistics about meshlet generation output.
#[derive(Debug, Clone, Default)]
pub struct MeshletStats {
    /// Total number of meshlets.
    pub meshlet_count: usize,
    /// Total vertices across all meshlets.
    pub total_vertices: usize,
    /// Total triangles across all meshlets.
    pub total_triangles: usize,
    /// Average vertices per meshlet.
    pub avg_vertices: f32,
    /// Average triangles per meshlet.
    pub avg_triangles: f32,
    /// Minimum vertices in any meshlet.
    pub min_vertices: usize,
    /// Maximum vertices in any meshlet.
    pub max_vertices: usize,
    /// Minimum triangles in any meshlet.
    pub min_triangles: usize,
    /// Maximum triangles in any meshlet.
    pub max_triangles: usize,
    /// Total memory usage in bytes.
    pub memory_bytes: usize,
}

// ---------------------------------------------------------------------------
// MeshletGenerator
// ---------------------------------------------------------------------------

/// Configurable meshlet generator using greedy algorithm.
///
/// The generator partitions indexed triangle meshes into meshlets, each
/// containing at most `max_vertices` vertices and `max_triangles` triangles.
///
/// # Example
///
/// ```ignore
/// let generator = MeshletGenerator::new(); // Use default limits
/// let output = generator.generate(&input);
///
/// // Or with custom limits:
/// let generator = MeshletGenerator::with_limits(32, 62);
/// let output = generator.generate(&input);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MeshletGenerator {
    max_vertices: usize,
    max_triangles: usize,
}

impl MeshletGenerator {
    /// Create a new meshlet generator with default limits.
    ///
    /// Uses `MAX_MESHLET_VERTICES` (64) and `MAX_MESHLET_TRIANGLES` (124).
    pub fn new() -> Self {
        Self {
            max_vertices: MAX_MESHLET_VERTICES,
            max_triangles: MAX_MESHLET_TRIANGLES,
        }
    }

    /// Create a meshlet generator with custom limits.
    ///
    /// # Arguments
    ///
    /// * `max_vertices` - Maximum vertices per meshlet (1-255)
    /// * `max_triangles` - Maximum triangles per meshlet (1-255)
    ///
    /// # Panics
    ///
    /// Panics if limits are zero or exceed 255.
    pub fn with_limits(max_vertices: usize, max_triangles: usize) -> Self {
        assert!(
            max_vertices > 0 && max_vertices <= 255,
            "max_vertices must be in range 1..=255"
        );
        assert!(
            max_triangles > 0 && max_triangles <= 255,
            "max_triangles must be in range 1..=255"
        );
        Self {
            max_vertices,
            max_triangles,
        }
    }

    /// Returns the maximum vertices per meshlet.
    pub fn max_vertices(&self) -> usize {
        self.max_vertices
    }

    /// Returns the maximum triangles per meshlet.
    pub fn max_triangles(&self) -> usize {
        self.max_triangles
    }

    /// Generate meshlets from indexed triangle mesh.
    ///
    /// Uses a greedy algorithm to partition the mesh into meshlets:
    ///
    /// 1. Process triangles in order
    /// 2. For each triangle, count new vertices it would add
    /// 3. If exceeds limits, finalize current meshlet and start new
    /// 4. Compute bounds (bounding sphere + normal cone) for each meshlet
    ///
    /// # Arguments
    ///
    /// * `input` - Mesh data (positions, indices, optional normals)
    ///
    /// # Returns
    ///
    /// `MeshletOutput` containing all meshlet structures and index data.
    ///
    /// # Panics
    ///
    /// Panics if `input.indices.len()` is not a multiple of 3.
    pub fn generate(&self, input: &MeshInput) -> MeshletOutput {
        assert!(
            input.indices.len() % 3 == 0,
            "Index count must be multiple of 3"
        );

        if input.is_empty() {
            return MeshletOutput::new();
        }

        // Build raw meshlets using greedy algorithm
        let raw_meshlets = self.build_meshlets_greedy(input.positions, input.indices);

        // Convert to final format with bounds
        self.build_output(input, raw_meshlets)
    }

    /// Build meshlets using greedy algorithm.
    ///
    /// Returns a vector of (vertex_indices, triangles) pairs.
    fn build_meshlets_greedy(
        &self,
        positions: &[[f32; 3]],
        indices: &[u32],
    ) -> Vec<(Vec<u32>, Vec<[u32; 3]>)> {
        let triangle_count = indices.len() / 3;
        if triangle_count == 0 {
            return vec![];
        }

        let mut result = Vec::new();
        let mut current_vertices: Vec<u32> = Vec::with_capacity(self.max_vertices);
        let mut current_vertex_set: HashSet<u32> = HashSet::with_capacity(self.max_vertices);
        let mut current_triangles: Vec<[u32; 3]> = Vec::with_capacity(self.max_triangles);

        for tri_idx in 0..triangle_count {
            let i0 = indices[tri_idx * 3];
            let i1 = indices[tri_idx * 3 + 1];
            let i2 = indices[tri_idx * 3 + 2];

            // Skip degenerate triangles
            if i0 == i1 || i1 == i2 || i0 == i2 {
                continue;
            }

            // Validate indices
            if i0 as usize >= positions.len()
                || i1 as usize >= positions.len()
                || i2 as usize >= positions.len()
            {
                continue;
            }

            // Count new vertices this triangle would add
            let new_verts = [i0, i1, i2]
                .iter()
                .filter(|&&v| !current_vertex_set.contains(&v))
                .count();

            // Check if triangle fits in current meshlet
            let would_exceed_vertices = current_vertices.len() + new_verts > self.max_vertices;
            let would_exceed_triangles = current_triangles.len() >= self.max_triangles;

            if would_exceed_vertices || would_exceed_triangles {
                // Finalize current meshlet and start a new one
                if !current_triangles.is_empty() {
                    result.push((
                        std::mem::take(&mut current_vertices),
                        std::mem::take(&mut current_triangles),
                    ));
                    current_vertex_set.clear();
                }
            }

            // Add triangle to current meshlet
            for &v in &[i0, i1, i2] {
                if !current_vertex_set.contains(&v) {
                    current_vertex_set.insert(v);
                    current_vertices.push(v);
                }
            }
            current_triangles.push([i0, i1, i2]);
        }

        // Don't forget the last meshlet
        if !current_triangles.is_empty() {
            result.push((current_vertices, current_triangles));
        }

        result
    }

    /// Build final output from raw meshlets.
    fn build_output(
        &self,
        input: &MeshInput,
        raw_meshlets: Vec<(Vec<u32>, Vec<[u32; 3]>)>,
    ) -> MeshletOutput {
        let mut meshlets = Vec::with_capacity(raw_meshlets.len());
        let mut bounds = Vec::with_capacity(raw_meshlets.len());
        let mut vertex_indices = Vec::new();
        let mut triangle_indices = Vec::new();

        for (meshlet_vertices, meshlet_triangles) in raw_meshlets {
            let vertex_offset = vertex_indices.len() as u32;
            let triangle_offset = triangle_indices.len() as u32;
            let vertex_count = meshlet_vertices.len() as u8;
            let triangle_count = meshlet_triangles.len() as u8;

            // Build local vertex index map
            let mut local_map: HashMap<u32, u8> = HashMap::new();
            for (local_idx, &global_idx) in meshlet_vertices.iter().enumerate() {
                local_map.insert(global_idx, local_idx as u8);
                vertex_indices.push(global_idx);
            }

            // Emit local triangle indices
            for tri in &meshlet_triangles {
                triangle_indices.push(local_map[&tri[0]]);
                triangle_indices.push(local_map[&tri[1]]);
                triangle_indices.push(local_map[&tri[2]]);
            }

            // Compute bounds
            let (center, radius) = Self::compute_bounds(input.positions, &meshlet_vertices);

            let (cone_axis, cone_cutoff) = if let Some(normals) = input.normals {
                Self::compute_cone_from_normals(normals, &meshlet_triangles)
            } else {
                Self::compute_cone(input.positions, &meshlet_triangles)
            };

            meshlets.push(Meshlet::new(
                vertex_offset,
                triangle_offset,
                vertex_count,
                triangle_count,
            ));

            bounds.push(MeshletBounds::new(center, radius, cone_axis, cone_cutoff));
        }

        MeshletOutput {
            meshlets,
            bounds,
            vertex_indices,
            triangle_indices,
        }
    }

    /// Compute bounding sphere for meshlet vertices (Ritter's algorithm).
    ///
    /// # Algorithm
    ///
    /// 1. Start with sphere containing first two points
    /// 2. Iteratively expand to include all other points
    /// 3. Final pass to ensure all points are contained
    ///
    /// # Returns
    ///
    /// (center, radius) tuple.
    fn compute_bounds(positions: &[[f32; 3]], vertex_indices: &[u32]) -> ([f32; 3], f32) {
        if vertex_indices.is_empty() {
            return ([0.0, 0.0, 0.0], 0.0);
        }

        if vertex_indices.len() == 1 {
            let p = positions[vertex_indices[0] as usize];
            return (p, 0.0);
        }

        // Initial sphere: use first two points
        let p0 = positions[vertex_indices[0] as usize];
        let p1 = positions[vertex_indices[1] as usize];

        let mut center = [
            (p0[0] + p1[0]) * 0.5,
            (p0[1] + p1[1]) * 0.5,
            (p0[2] + p1[2]) * 0.5,
        ];
        let mut radius = distance(p0, p1) * 0.5;

        // Expand to include all other points
        for &idx in &vertex_indices[2..] {
            let p = positions[idx as usize];
            let dist = distance(center, p);

            if dist > radius {
                // Point is outside, expand sphere
                let new_radius = (radius + dist) * 0.5;
                let ratio = (new_radius - radius) / dist;
                center[0] += (p[0] - center[0]) * ratio;
                center[1] += (p[1] - center[1]) * ratio;
                center[2] += (p[2] - center[2]) * ratio;
                radius = new_radius;
            }
        }

        // Final pass: ensure all points are contained
        for &idx in vertex_indices {
            let p = positions[idx as usize];
            let dist = distance(center, p);
            if dist > radius {
                radius = dist;
            }
        }

        (center, radius)
    }

    /// Compute backface culling cone from triangle face normals.
    ///
    /// The normal cone is defined by:
    /// - **axis**: Average face normal direction (normalized)
    /// - **cutoff**: Cosine of half-angle that contains all face normals
    ///
    /// A cutoff of 1.0 means all normals are identical (single ray).
    /// A cutoff of -1.0 means the cone spans a full hemisphere.
    /// A cutoff below -1.0 (e.g., -2.0) disables backface culling.
    ///
    /// # Returns
    ///
    /// (cone_axis, cone_cutoff) tuple.
    fn compute_cone(positions: &[[f32; 3]], triangles: &[[u32; 3]]) -> ([f32; 3], f32) {
        if triangles.is_empty() {
            return ([0.0, 0.0, 1.0], -2.0);
        }

        // Compute face normals
        let mut normals: Vec<[f32; 3]> = Vec::with_capacity(triangles.len());

        for tri in triangles {
            let p0 = positions[tri[0] as usize];
            let p1 = positions[tri[1] as usize];
            let p2 = positions[tri[2] as usize];

            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

            let n = cross(e1, e2);
            let len = length(n);

            if len > EPSILON {
                let inv_len = 1.0 / len;
                normals.push([n[0] * inv_len, n[1] * inv_len, n[2] * inv_len]);
            }
        }

        if normals.is_empty() {
            return ([0.0, 0.0, 1.0], -2.0);
        }

        Self::compute_cone_from_vectors(&normals)
    }

    /// Compute normal cone using pre-computed vertex normals.
    ///
    /// Face normals are approximated by averaging vertex normals.
    fn compute_cone_from_normals(
        normals: &[[f32; 3]],
        triangles: &[[u32; 3]],
    ) -> ([f32; 3], f32) {
        if triangles.is_empty() {
            return ([0.0, 0.0, 1.0], -2.0);
        }

        // Compute face normals by averaging vertex normals
        let mut face_normals: Vec<[f32; 3]> = Vec::with_capacity(triangles.len());

        for tri in triangles {
            if tri[0] as usize >= normals.len()
                || tri[1] as usize >= normals.len()
                || tri[2] as usize >= normals.len()
            {
                continue;
            }

            let n0 = normals[tri[0] as usize];
            let n1 = normals[tri[1] as usize];
            let n2 = normals[tri[2] as usize];

            let avg = [
                (n0[0] + n1[0] + n2[0]) / 3.0,
                (n0[1] + n1[1] + n2[1]) / 3.0,
                (n0[2] + n1[2] + n2[2]) / 3.0,
            ];

            let len = length(avg);
            if len > EPSILON {
                let inv_len = 1.0 / len;
                face_normals.push([avg[0] * inv_len, avg[1] * inv_len, avg[2] * inv_len]);
            }
        }

        if face_normals.is_empty() {
            return ([0.0, 0.0, 1.0], -2.0);
        }

        Self::compute_cone_from_vectors(&face_normals)
    }

    /// Compute cone axis and cutoff from a set of normalized vectors.
    fn compute_cone_from_vectors(normals: &[[f32; 3]]) -> ([f32; 3], f32) {
        if normals.is_empty() {
            return ([0.0, 0.0, 1.0], -2.0);
        }

        if normals.len() == 1 {
            return (normals[0], 1.0);
        }

        // Compute average normal (cone axis)
        let mut axis = [0.0f32; 3];
        for n in normals {
            axis[0] += n[0];
            axis[1] += n[1];
            axis[2] += n[2];
        }

        let len = length(axis);
        if len < EPSILON {
            // Normals cancel out, cone spans full sphere
            return ([0.0, 0.0, 1.0], -2.0);
        }

        let inv_len = 1.0 / len;
        axis = [axis[0] * inv_len, axis[1] * inv_len, axis[2] * inv_len];

        // Find minimum dot product (maximum angle from axis)
        let mut min_dot = 1.0f32;
        for n in normals {
            let d = dot(axis, *n);
            if d < min_dot {
                min_dot = d;
            }
        }

        (axis, min_dot)
    }
}

impl Default for MeshletGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // MeshInput tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mesh_input_new() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = vec![0, 1, 2];
        let input = MeshInput::new(&positions, &indices);

        assert!(input.normals.is_none());
        assert_eq!(input.triangle_count(), 1);
        assert!(!input.is_empty());
    }

    #[test]
    fn test_mesh_input_with_normals() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = vec![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]];
        let indices = vec![0, 1, 2];
        let input = MeshInput::with_normals(&positions, &indices, &normals);

        assert!(input.normals.is_some());
        assert_eq!(input.normals.unwrap().len(), 3);
    }

    #[test]
    fn test_mesh_input_validate_ok() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = vec![0, 1, 2];
        let input = MeshInput::new(&positions, &indices);
        assert!(input.validate().is_ok());
    }

    #[test]
    fn test_mesh_input_validate_bad_index_count() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let indices = vec![0, 1]; // Not multiple of 3
        let input = MeshInput::new(&positions, &indices);
        assert!(input.validate().is_err());
    }

    #[test]
    fn test_mesh_input_validate_index_out_of_bounds() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = vec![0, 1, 10]; // Index 10 is out of bounds
        let input = MeshInput::new(&positions, &indices);
        assert!(input.validate().is_err());
    }

    #[test]
    fn test_mesh_input_validate_normals_mismatch() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = vec![[0.0, 0.0, 1.0]]; // Wrong count
        let indices = vec![0, 1, 2];
        let input = MeshInput::with_normals(&positions, &indices, &normals);
        assert!(input.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // MeshletGenerator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_generator_new() {
        let gen = MeshletGenerator::new();
        assert_eq!(gen.max_vertices(), MAX_MESHLET_VERTICES);
        assert_eq!(gen.max_triangles(), MAX_MESHLET_TRIANGLES);
    }

    #[test]
    fn test_generator_with_limits() {
        let gen = MeshletGenerator::with_limits(32, 62);
        assert_eq!(gen.max_vertices(), 32);
        assert_eq!(gen.max_triangles(), 62);
    }

    #[test]
    fn test_generator_default() {
        let gen = MeshletGenerator::default();
        assert_eq!(gen.max_vertices(), MAX_MESHLET_VERTICES);
        assert_eq!(gen.max_triangles(), MAX_MESHLET_TRIANGLES);
    }

    #[test]
    #[should_panic]
    fn test_generator_with_limits_zero_vertices() {
        MeshletGenerator::with_limits(0, 100);
    }

    #[test]
    #[should_panic]
    fn test_generator_with_limits_zero_triangles() {
        MeshletGenerator::with_limits(64, 0);
    }

    // -----------------------------------------------------------------------
    // Single triangle tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_triangle() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = vec![0, 1, 2];
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert_eq!(output.meshlet_count(), 1);
        assert_eq!(output.meshlets[0].vertex_count, 3);
        assert_eq!(output.meshlets[0].triangle_count, 1);
        assert!(output.validate(positions.len()).is_ok());
    }

    #[test]
    fn test_empty_mesh() {
        let input = MeshInput::new(&[], &[]);
        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert!(output.is_empty());
        assert_eq!(output.meshlet_count(), 0);
    }

    // -----------------------------------------------------------------------
    // Cube mesh tests
    // -----------------------------------------------------------------------

    fn make_cube() -> (Vec<[f32; 3]>, Vec<u32>) {
        let positions = vec![
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ];

        let indices = vec![
            0, 1, 2, 0, 2, 3, // Front
            4, 5, 6, 4, 6, 7, // Back
            3, 2, 6, 3, 6, 5, // Top
            4, 7, 1, 4, 1, 0, // Bottom
            1, 7, 6, 1, 6, 2, // Right
            4, 0, 3, 4, 3, 5, // Left
        ];

        (positions, indices)
    }

    #[test]
    fn test_cube_single_meshlet() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        // Cube has 12 triangles, should fit in one meshlet
        assert_eq!(output.meshlet_count(), 1);
        assert_eq!(output.meshlets[0].triangle_count, 12);
        assert!(output.meshlets[0].vertex_count <= 8);
        assert!(output.validate(positions.len()).is_ok());
    }

    #[test]
    fn test_cube_bounding_sphere() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        let b = &output.bounds[0];

        // All vertices should be within the sphere
        for idx in &output.vertex_indices {
            let p = positions[*idx as usize];
            let dist = distance(b.center, p);
            assert!(
                dist <= b.radius + EPSILON,
                "Vertex outside bounding sphere"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Large mesh splitting tests
    // -----------------------------------------------------------------------

    fn make_large_mesh(triangle_count: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        for i in 0..triangle_count {
            let x = (i % 100) as f32;
            let y = (i / 100) as f32;

            let base = positions.len() as u32;
            positions.push([x, y, 0.0]);
            positions.push([x + 1.0, y, 0.0]);
            positions.push([x + 0.5, y + 1.0, 0.0]);

            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
        }

        (positions, indices)
    }

    #[test]
    fn test_large_mesh_splits() {
        let (positions, indices) = make_large_mesh(200);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert!(output.meshlet_count() > 1);
        assert!(output.validate(positions.len()).is_ok());

        let reconstructed = output.reconstruct_indices();
        assert_eq!(reconstructed.len(), indices.len());
    }

    #[test]
    fn test_vertex_limit_respected() {
        let (positions, indices) = make_large_mesh(500);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        for m in &output.meshlets {
            assert!(
                m.vertex_count as usize <= MAX_MESHLET_VERTICES,
                "Vertex limit exceeded"
            );
        }
    }

    #[test]
    fn test_triangle_limit_respected() {
        // Grid mesh with high vertex sharing
        let grid_size = 20;
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        for y in 0..grid_size {
            for x in 0..grid_size {
                positions.push([x as f32, y as f32, 0.0]);
            }
        }

        for y in 0..(grid_size - 1) {
            for x in 0..(grid_size - 1) {
                let tl = y * grid_size + x;
                let tr = y * grid_size + x + 1;
                let bl = (y + 1) * grid_size + x;
                let br = (y + 1) * grid_size + x + 1;

                indices.push(tl as u32);
                indices.push(bl as u32);
                indices.push(tr as u32);

                indices.push(tr as u32);
                indices.push(bl as u32);
                indices.push(br as u32);
            }
        }

        let input = MeshInput::new(&positions, &indices);
        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        for m in &output.meshlets {
            assert!(
                m.triangle_count as usize <= MAX_MESHLET_TRIANGLES,
                "Triangle limit exceeded"
            );
        }
    }

    #[test]
    fn test_custom_limits() {
        let (positions, indices) = make_large_mesh(100);
        let input = MeshInput::new(&positions, &indices);

        // Use smaller limits
        let gen = MeshletGenerator::with_limits(16, 30);
        let output = gen.generate(&input);

        for m in &output.meshlets {
            assert!(m.vertex_count <= 16);
            assert!(m.triangle_count <= 30);
        }
    }

    // -----------------------------------------------------------------------
    // Degenerate/invalid handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_degenerate_triangle_skipped() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = vec![0, 0, 1, 0, 1, 2]; // First triangle is degenerate
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert_eq!(output.meshlets[0].triangle_count, 1);
    }

    #[test]
    fn test_out_of_range_indices_skipped() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = vec![0, 1, 2, 0, 1, 100]; // Second triangle has invalid index
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert_eq!(output.meshlets[0].triangle_count, 1);
    }

    // -----------------------------------------------------------------------
    // Reconstruction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reconstruct_indices() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);
        let reconstructed = output.reconstruct_indices();

        assert_eq!(reconstructed.len(), indices.len());

        // Same triangles should exist
        use std::collections::HashSet;
        let orig_tris: HashSet<_> = indices
            .chunks(3)
            .map(|c| {
                let mut t = [c[0], c[1], c[2]];
                t.sort();
                t
            })
            .collect();

        let recon_tris: HashSet<_> = reconstructed
            .chunks(3)
            .map(|c| {
                let mut t = [c[0], c[1], c[2]];
                t.sort();
                t
            })
            .collect();

        assert_eq!(orig_tris, recon_tris);
    }

    // -----------------------------------------------------------------------
    // Stats tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats() {
        let (positions, indices) = make_large_mesh(200);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);
        let stats = output.stats();

        assert!(stats.meshlet_count > 0);
        assert!(stats.total_triangles == 200);
        assert!(stats.avg_triangles > 0.0);
        assert!(stats.memory_bytes > 0);
    }

    // -----------------------------------------------------------------------
    // Normal cone tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_triangle_cone() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = vec![0, 1, 2];
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        // Single triangle should have cutoff of 1.0
        assert_eq!(output.bounds[0].cone_cutoff, 1.0);
        // Normal should point in +Z direction
        assert!(
            (output.bounds[0].cone_axis[2] - 1.0).abs() < EPSILON
                || (output.bounds[0].cone_axis[2] + 1.0).abs() < EPSILON
        );
    }

    #[test]
    fn test_with_normals() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ];
        let normals = vec![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];
        let input = MeshInput::with_normals(&positions, &indices, &normals);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert!(output.bounds[0].has_valid_cone());
        // With all normals pointing +Z, cone should be tight
        assert!(output.bounds[0].cone_cutoff > 0.9);
    }

    // -----------------------------------------------------------------------
    // Memory usage tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_memory_usage() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        let usage = output.memory_usage();
        assert!(usage > 0);

        // Should be roughly: 1 meshlet * 12 bytes + 1 bounds * 32 bytes +
        // 8 vertex indices * 4 bytes + 36 local indices * 1 byte
        // = 12 + 32 + 32 + 36 = 112 bytes
        assert!(usage > 100 && usage < 200);
    }

    // -----------------------------------------------------------------------
    // Output validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_output_validate_ok() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert!(output.validate(positions.len()).is_ok());
    }

    #[test]
    fn test_output_validate_corrupted_vertex_index() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let mut output = gen.generate(&input);
        output.vertex_indices[0] = 999;

        assert!(output.validate(positions.len()).is_err());
    }

    #[test]
    fn test_output_validate_corrupted_local_index() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let mut output = gen.generate(&input);
        output.triangle_indices[0] = 200;

        assert!(output.validate(positions.len()).is_err());
    }
}

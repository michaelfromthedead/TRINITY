//! Meshlet Generation for GPU-driven rendering (T-GPU-4.3).
//!
//! This module provides meshlet partitioning for fine-grained GPU culling.
//! Meshlets are small clusters of triangles (typically 64 vertices, ~124
//! triangles) with bounding data that enables efficient per-cluster culling.
//!
//! # Overview
//!
//! Traditional mesh rendering submits entire meshes to the GPU, which can
//! only cull at the object level. Meshlets allow the GPU to cull at a much
//! finer granularity, skipping small clusters of triangles that are occluded
//! or outside the frustum.
//!
//! # Data Structures
//!
//! - [`Meshlet`]: Offsets and counts into vertex/index buffers
//! - [`MeshletBounds`]: Bounding sphere and normal cone for culling
//! - [`MeshletData`]: Complete meshlet data ready for GPU upload
//!
//! # Performance
//!
//! - Work complexity: O(n) for n triangles
//! - Memory: 8 bytes per meshlet + 32 bytes bounds + index data
//! - Target: ~124 triangles per meshlet for optimal GPU cache utilization
//!
//! # Usage
//!
//! ```ignore
//! let positions: &[[f32; 3]] = &mesh.positions;
//! let indices: &[u32] = &mesh.indices;
//! let normals: Option<&[[f32; 3]]> = Some(&mesh.normals);
//!
//! let meshlet_data = MeshletData::generate(positions, indices, normals);
//!
//! // Upload to GPU
//! let meshlet_buffer = create_buffer(&meshlet_data.meshlets);
//! let bounds_buffer = create_buffer(&meshlet_data.bounds);
//! let vertex_index_buffer = create_buffer(&meshlet_data.vertex_indices);
//! let local_index_buffer = create_buffer(&meshlet_data.local_indices);
//! ```
//!
//! # GPU Culling
//!
//! The GPU can use the bounding data for two-phase culling:
//!
//! 1. **Frustum cull**: Test bounding sphere against frustum planes
//! 2. **Backface cull**: Test normal cone against view direction
//!
//! A meshlet passes backface culling if any triangle might face the camera:
//!
//! ```wgsl
//! let visible = dot(cone_axis, view_dir) < cone_cutoff;
//! ```

use std::collections::{HashMap, HashSet};
use std::mem;

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum vertices per meshlet (GPU-friendly power of 2).
pub const MAX_MESHLET_VERTICES: usize = 64;

/// Maximum triangles per meshlet.
///
/// This is derived from the vertex limit: with 64 vertices in a triangle
/// strip-like arrangement, we can have at most ~124 triangles. For
/// irregular meshes, the practical limit is often lower.
pub const MAX_MESHLET_TRIANGLES: usize = 124;

/// Size of a single Meshlet struct in bytes (must match WGSL layout).
pub const MESHLET_SIZE: usize = 12;

/// Size of a single MeshletBounds struct in bytes (must match WGSL layout).
pub const MESHLET_BOUNDS_SIZE: usize = 32;

/// Epsilon for floating point comparisons.
const EPSILON: f32 = 1e-8;

// ---------------------------------------------------------------------------
// Meshlet
// ---------------------------------------------------------------------------

/// Meshlet data structure for GPU consumption.
///
/// Each meshlet references a contiguous range of vertex indices and local
/// triangle indices. The vertex_offset and triangle_offset are byte offsets
/// into the vertex index buffer and local index buffer respectively.
///
/// # Memory Layout
///
/// 12 bytes, 4-byte aligned:
/// | Offset | Field          | Size |
/// |--------|----------------|------|
/// | 0      | vertex_offset  | 4    |
/// | 4      | triangle_offset| 4    |
/// | 8      | vertex_count   | 1    |
/// | 9      | triangle_count | 1    |
/// | 10     | _padding       | 2    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Meshlet {
    /// Offset into vertex index buffer (number of u32s, not bytes).
    pub vertex_offset: u32,
    /// Offset into local triangle index buffer (number of bytes).
    pub triangle_offset: u32,
    /// Number of vertices in this meshlet (max 64).
    pub vertex_count: u8,
    /// Number of triangles in this meshlet (max 124).
    pub triangle_count: u8,
    /// Padding for 4-byte alignment.
    pub _padding: [u8; 2],
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<Meshlet>() == MESHLET_SIZE);

impl Meshlet {
    /// Maximum vertices per meshlet (GPU-friendly power of 2).
    pub const MAX_VERTICES: u8 = 64;

    /// Maximum triangles per meshlet.
    pub const MAX_TRIANGLES: u8 = 124;

    /// Create a new meshlet with the given parameters.
    pub fn new(
        vertex_offset: u32,
        triangle_offset: u32,
        vertex_count: u8,
        triangle_count: u8,
    ) -> Self {
        Self {
            vertex_offset,
            triangle_offset,
            vertex_count,
            triangle_count,
            _padding: [0; 2],
        }
    }

    /// Returns true if this meshlet is empty.
    pub fn is_empty(&self) -> bool {
        self.triangle_count == 0
    }
}

// ---------------------------------------------------------------------------
// MeshletBounds
// ---------------------------------------------------------------------------

/// Meshlet bounding data for GPU culling.
///
/// Contains a bounding sphere for frustum culling and a normal cone for
/// backface culling.
///
/// # Bounding Sphere
///
/// The bounding sphere contains all vertices in the meshlet. A meshlet
/// is potentially visible if the sphere intersects the view frustum.
///
/// # Normal Cone
///
/// The normal cone approximates the range of face normals in the meshlet.
/// The cone axis is the average normal direction, and the cone cutoff is
/// the cosine of the half-angle that contains all face normals.
///
/// A meshlet is backface-culled if the entire normal cone points away
/// from the camera: `dot(cone_axis, view_dir) >= cone_cutoff`
///
/// # Memory Layout
///
/// 32 bytes, vec4 aligned:
/// | Offset | Field       | Size |
/// |--------|-------------|------|
/// | 0      | center.x    | 4    |
/// | 4      | center.y    | 4    |
/// | 8      | center.z    | 4    |
/// | 12     | radius      | 4    |
/// | 16     | cone_axis.x | 4    |
/// | 20     | cone_axis.y | 4    |
/// | 24     | cone_axis.z | 4    |
/// | 28     | cone_cutoff | 4    |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct MeshletBounds {
    /// Bounding sphere center.
    pub center: [f32; 3],
    /// Bounding sphere radius.
    pub radius: f32,
    /// Normal cone axis (normalized average normal).
    pub cone_axis: [f32; 3],
    /// Normal cone cutoff (cos of half-angle).
    ///
    /// A value of 1.0 means the cone is a single ray (all normals identical).
    /// A value of -1.0 means the cone spans a full hemisphere.
    /// Values below -1.0 indicate the cone cannot be used for backface culling.
    pub cone_cutoff: f32,
}

// Compile-time size assertion
const _: () = assert!(mem::size_of::<MeshletBounds>() == MESHLET_BOUNDS_SIZE);

impl MeshletBounds {
    /// Create bounds with a bounding sphere only (no normal cone culling).
    pub fn sphere_only(center: [f32; 3], radius: f32) -> Self {
        Self {
            center,
            radius,
            cone_axis: [0.0, 0.0, 1.0],
            // Cutoff of -2.0 disables normal cone culling
            cone_cutoff: -2.0,
        }
    }

    /// Create bounds with both sphere and normal cone.
    pub fn new(center: [f32; 3], radius: f32, cone_axis: [f32; 3], cone_cutoff: f32) -> Self {
        Self {
            center,
            radius,
            cone_axis,
            cone_cutoff,
        }
    }

    /// Returns true if the normal cone can be used for backface culling.
    ///
    /// A cone with cutoff < -1.0 is invalid or disabled.
    pub fn has_valid_cone(&self) -> bool {
        self.cone_cutoff >= -1.0
    }

    /// Builder-style method to set the bounding sphere.
    ///
    /// # Arguments
    ///
    /// * `center` - Bounding sphere center (xyz)
    /// * `radius` - Bounding sphere radius
    pub fn with_bounds(mut self, center: [f32; 3], radius: f32) -> Self {
        self.center = center;
        self.radius = radius;
        self
    }

    /// Builder-style method to set the normal cone for backface culling.
    ///
    /// # Arguments
    ///
    /// * `normal` - Normalized cone axis (average face normal direction)
    /// * `cutoff` - Cosine of the half-angle that contains all face normals
    ///
    /// # Backface Culling
    ///
    /// If `dot(view_dir, normal) < cutoff`, the entire meshlet is backfacing.
    /// Use cutoff = -1.0 to disable backface culling (cone spans full hemisphere).
    pub fn with_cone(mut self, normal: [f32; 3], cutoff: f32) -> Self {
        self.cone_axis = normal;
        self.cone_cutoff = cutoff;
        self
    }
}

// ---------------------------------------------------------------------------
// MeshletData
// ---------------------------------------------------------------------------

/// Complete meshlet data ready for GPU upload.
///
/// This struct contains all the data needed to render a mesh using meshlets:
///
/// - `meshlets`: Array of Meshlet structs (offsets and counts)
/// - `bounds`: Array of MeshletBounds structs (culling data)
/// - `vertex_indices`: Global vertex indices referenced by each meshlet
/// - `local_indices`: Triangle indices (0-63) into each meshlet's vertex list
///
/// # GPU Buffer Layout
///
/// ```text
/// meshlet_buffer:      [Meshlet, Meshlet, ...]
/// bounds_buffer:       [MeshletBounds, MeshletBounds, ...]
/// vertex_index_buffer: [u32, u32, ...]  // Global vertex indices
/// local_index_buffer:  [u8, u8, ...]    // Triangle indices (triples)
/// ```
///
/// # Rendering
///
/// To render triangle `t` of meshlet `m`:
///
/// 1. Load meshlet: `let ml = meshlets[m];`
/// 2. Load local indices: `let i0 = local_indices[ml.triangle_offset + t*3 + 0];`
/// 3. Map to global: `let v0 = vertex_indices[ml.vertex_offset + i0];`
/// 4. Fetch vertex: `let pos = positions[v0];`
#[derive(Clone, Debug, Default)]
pub struct MeshletData {
    /// Array of meshlet descriptors.
    pub meshlets: Vec<Meshlet>,
    /// Array of bounding data for each meshlet.
    pub bounds: Vec<MeshletBounds>,
    /// Global vertex indices (flattened across all meshlets).
    pub vertex_indices: Vec<u32>,
    /// Local triangle indices (triples of u8, flattened).
    pub local_indices: Vec<u8>,
}

impl MeshletData {
    /// Create empty meshlet data.
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

    /// Generate meshlets from mesh vertex/index data.
    ///
    /// Uses a greedy algorithm to partition the mesh into meshlets. Each
    /// meshlet contains up to MAX_MESHLET_VERTICES vertices and up to
    /// MAX_MESHLET_TRIANGLES triangles.
    ///
    /// # Arguments
    ///
    /// * `positions` - Vertex positions (must match index references)
    /// * `indices` - Triangle indices (length must be multiple of 3)
    /// * `normals` - Optional vertex normals for normal cone computation
    ///
    /// # Returns
    ///
    /// `MeshletData` containing all meshlet structures and index data.
    ///
    /// # Panics
    ///
    /// Panics if `indices.len()` is not a multiple of 3.
    pub fn generate(
        positions: &[[f32; 3]],
        indices: &[u32],
        normals: Option<&[[f32; 3]]>,
    ) -> Self {
        assert!(
            indices.len() % 3 == 0,
            "Index count must be multiple of 3"
        );

        if indices.is_empty() || positions.is_empty() {
            return Self::new();
        }

        // Build meshlets using greedy algorithm
        let raw_meshlets = build_meshlets_greedy(positions, indices);

        // Convert to final format
        let mut meshlets = Vec::with_capacity(raw_meshlets.len());
        let mut bounds = Vec::with_capacity(raw_meshlets.len());
        let mut vertex_indices = Vec::new();
        let mut local_indices = Vec::new();

        for (meshlet_vertices, meshlet_triangles) in raw_meshlets {
            let vertex_offset = vertex_indices.len() as u32;
            let triangle_offset = local_indices.len() as u32;
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
                local_indices.push(local_map[&tri[0]]);
                local_indices.push(local_map[&tri[1]]);
                local_indices.push(local_map[&tri[2]]);
            }

            // Compute bounds
            let (center, radius) =
                compute_bounding_sphere(positions, &meshlet_vertices);

            let (cone_axis, cone_cutoff) = if let Some(norms) = normals {
                compute_normal_cone_from_normals(norms, &meshlet_triangles)
            } else {
                compute_normal_cone(positions, &meshlet_triangles)
            };

            meshlets.push(Meshlet::new(
                vertex_offset,
                triangle_offset,
                vertex_count,
                triangle_count,
            ));

            bounds.push(MeshletBounds::new(center, radius, cone_axis, cone_cutoff));
        }

        Self {
            meshlets,
            bounds,
            vertex_indices,
            local_indices,
        }
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
                let i0 = self.local_indices[local_base] as usize;
                let i1 = self.local_indices[local_base + 1] as usize;
                let i2 = self.local_indices[local_base + 2] as usize;

                result.push(self.vertex_indices[v_start + i0]);
                result.push(self.vertex_indices[v_start + i1]);
                result.push(self.vertex_indices[v_start + i2]);
            }
        }

        result
    }

    /// Validate meshlet data integrity.
    ///
    /// Checks that all indices are valid and within bounds.
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
            if t_start + t_count * 3 > self.local_indices.len() {
                return Err("Triangle offset + count*3 exceeds local_indices length");
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
                if self.local_indices[j] >= v_count as u8 {
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
}

// ---------------------------------------------------------------------------
// Meshlet building algorithms
// ---------------------------------------------------------------------------

/// Build meshlets using a greedy algorithm.
///
/// This algorithm processes triangles in order, adding them to the current
/// meshlet until it's full, then starting a new one.
///
/// # Returns
///
/// A vector of (vertex_indices, triangles) pairs, one per meshlet.
fn build_meshlets_greedy(
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Vec<(Vec<u32>, Vec<[u32; 3]>)> {
    let triangle_count = indices.len() / 3;
    if triangle_count == 0 {
        return vec![];
    }

    let mut result = Vec::new();
    let mut current_vertices: Vec<u32> = Vec::with_capacity(MAX_MESHLET_VERTICES);
    let mut current_vertex_set: HashSet<u32> = HashSet::with_capacity(MAX_MESHLET_VERTICES);
    let mut current_triangles: Vec<[u32; 3]> = Vec::with_capacity(MAX_MESHLET_TRIANGLES);

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
        let would_exceed_vertices = current_vertices.len() + new_verts > MAX_MESHLET_VERTICES;
        let would_exceed_triangles = current_triangles.len() >= MAX_MESHLET_TRIANGLES;

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

// ---------------------------------------------------------------------------
// Bounding computations
// ---------------------------------------------------------------------------

/// Compute the bounding sphere for a set of vertices.
///
/// Uses Ritter's bounding sphere algorithm (iterative expansion).
///
/// # Returns
///
/// (center, radius) tuple.
fn compute_bounding_sphere(
    positions: &[[f32; 3]],
    vertex_indices: &[u32],
) -> ([f32; 3], f32) {
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

/// Compute the normal cone for a set of triangles.
///
/// The normal cone is defined by an axis (average normal) and a cutoff
/// (cos of the half-angle that contains all face normals).
///
/// # Returns
///
/// (cone_axis, cone_cutoff) tuple.
fn compute_normal_cone(
    positions: &[[f32; 3]],
    triangles: &[[u32; 3]],
) -> ([f32; 3], f32) {
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

    compute_cone_from_normals(&normals)
}

/// Compute normal cone using pre-computed vertex normals.
fn compute_normal_cone_from_normals(
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

    compute_cone_from_normals(&face_normals)
}

/// Compute cone axis and cutoff from a set of normalized vectors.
fn compute_cone_from_normals(normals: &[[f32; 3]]) -> ([f32; 3], f32) {
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
    // Meshlet struct tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_meshlet_size_and_alignment() {
        assert_eq!(
            std::mem::size_of::<Meshlet>(),
            MESHLET_SIZE,
            "Meshlet must be exactly {} bytes",
            MESHLET_SIZE,
        );
        assert_eq!(
            std::mem::align_of::<Meshlet>(),
            4,
            "Meshlet alignment must be 4 (u32)"
        );
    }

    #[test]
    fn test_meshlet_new() {
        let m = Meshlet::new(100, 200, 32, 40);
        assert_eq!(m.vertex_offset, 100);
        assert_eq!(m.triangle_offset, 200);
        assert_eq!(m.vertex_count, 32);
        assert_eq!(m.triangle_count, 40);
        assert_eq!(m._padding, [0, 0]);
    }

    #[test]
    fn test_meshlet_default() {
        let m = Meshlet::default();
        assert_eq!(m.vertex_offset, 0);
        assert_eq!(m.triangle_offset, 0);
        assert_eq!(m.vertex_count, 0);
        assert_eq!(m.triangle_count, 0);
    }

    #[test]
    fn test_meshlet_is_empty() {
        assert!(Meshlet::default().is_empty());
        assert!(!Meshlet::new(0, 0, 3, 1).is_empty());
        assert!(Meshlet::new(0, 0, 3, 0).is_empty());
    }

    // -----------------------------------------------------------------------
    // MeshletBounds struct tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_meshlet_bounds_size_and_alignment() {
        assert_eq!(
            std::mem::size_of::<MeshletBounds>(),
            MESHLET_BOUNDS_SIZE,
            "MeshletBounds must be exactly {} bytes",
            MESHLET_BOUNDS_SIZE,
        );
        assert_eq!(
            std::mem::align_of::<MeshletBounds>(),
            4,
            "MeshletBounds alignment must be 4 (f32)"
        );
    }

    #[test]
    fn test_meshlet_bounds_sphere_only() {
        let b = MeshletBounds::sphere_only([1.0, 2.0, 3.0], 5.0);
        assert_eq!(b.center, [1.0, 2.0, 3.0]);
        assert_eq!(b.radius, 5.0);
        assert!(!b.has_valid_cone());
    }

    #[test]
    fn test_meshlet_bounds_new() {
        let b = MeshletBounds::new([0.0, 0.0, 0.0], 1.0, [0.0, 1.0, 0.0], 0.5);
        assert_eq!(b.center, [0.0, 0.0, 0.0]);
        assert_eq!(b.radius, 1.0);
        assert_eq!(b.cone_axis, [0.0, 1.0, 0.0]);
        assert_eq!(b.cone_cutoff, 0.5);
        assert!(b.has_valid_cone());
    }

    #[test]
    fn test_meshlet_bounds_has_valid_cone() {
        assert!(MeshletBounds::new([0.0; 3], 1.0, [0.0, 0.0, 1.0], 0.5).has_valid_cone());
        assert!(MeshletBounds::new([0.0; 3], 1.0, [0.0, 0.0, 1.0], -1.0).has_valid_cone());
        assert!(!MeshletBounds::new([0.0; 3], 1.0, [0.0, 0.0, 1.0], -1.5).has_valid_cone());
        assert!(!MeshletBounds::sphere_only([0.0; 3], 1.0).has_valid_cone());
    }

    // -----------------------------------------------------------------------
    // Simple cube mesh test
    // -----------------------------------------------------------------------

    /// Creates a simple cube mesh for testing.
    fn make_cube() -> (Vec<[f32; 3]>, Vec<u32>) {
        let positions = vec![
            // Front face
            [-1.0, -1.0,  1.0],
            [ 1.0, -1.0,  1.0],
            [ 1.0,  1.0,  1.0],
            [-1.0,  1.0,  1.0],
            // Back face
            [-1.0, -1.0, -1.0],
            [-1.0,  1.0, -1.0],
            [ 1.0,  1.0, -1.0],
            [ 1.0, -1.0, -1.0],
        ];

        let indices = vec![
            // Front
            0, 1, 2, 0, 2, 3,
            // Back
            4, 5, 6, 4, 6, 7,
            // Top
            3, 2, 6, 3, 6, 5,
            // Bottom
            4, 7, 1, 4, 1, 0,
            // Right
            1, 7, 6, 1, 6, 2,
            // Left
            4, 0, 3, 4, 3, 5,
        ];

        (positions, indices)
    }

    #[test]
    fn test_cube_generates_one_meshlet() {
        let (positions, indices) = make_cube();
        let data = MeshletData::generate(&positions, &indices, None);

        // Cube has 12 triangles, should fit in one meshlet
        assert_eq!(data.meshlet_count(), 1);
        assert_eq!(data.meshlets[0].triangle_count, 12);
        assert!(data.meshlets[0].vertex_count <= 8);
        assert!(data.validate(positions.len()).is_ok());
    }

    #[test]
    fn test_cube_bounding_sphere_contains_all_vertices() {
        let (positions, indices) = make_cube();
        let data = MeshletData::generate(&positions, &indices, None);

        let b = &data.bounds[0];
        let center = b.center;
        let radius = b.radius;

        // All vertices should be within or on the sphere
        for idx in &data.vertex_indices {
            let p = positions[*idx as usize];
            let dist = distance(center, p);
            assert!(
                dist <= radius + EPSILON,
                "Vertex at {:?} is outside bounding sphere (dist={}, radius={})",
                p, dist, radius
            );
        }
    }

    // -----------------------------------------------------------------------
    // Large mesh splits into multiple meshlets
    // -----------------------------------------------------------------------

    /// Creates a larger mesh that will require multiple meshlets.
    fn make_large_mesh(triangle_count: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
        // Create a grid of triangles
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        // Each triangle uses 3 unique vertices for simplicity
        // This will force multiple meshlets due to vertex limit
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
    fn test_large_mesh_splits_into_multiple_meshlets() {
        // 200 triangles with 3 verts each = 600 verts total
        // With 64 vert limit, need at least 10 meshlets
        let (positions, indices) = make_large_mesh(200);
        let data = MeshletData::generate(&positions, &indices, None);

        assert!(
            data.meshlet_count() > 1,
            "Large mesh should split into multiple meshlets, got {}",
            data.meshlet_count()
        );
        assert!(data.validate(positions.len()).is_ok());

        // Verify all triangles are covered
        let reconstructed = data.reconstruct_indices();
        assert_eq!(reconstructed.len(), indices.len());
    }

    #[test]
    fn test_vertex_count_respects_limit() {
        let (positions, indices) = make_large_mesh(500);
        let data = MeshletData::generate(&positions, &indices, None);

        for (i, m) in data.meshlets.iter().enumerate() {
            assert!(
                m.vertex_count as usize <= MAX_MESHLET_VERTICES,
                "Meshlet {} has {} vertices, exceeds limit of {}",
                i, m.vertex_count, MAX_MESHLET_VERTICES
            );
        }
    }

    #[test]
    fn test_triangle_count_respects_limit() {
        // Create mesh with shared vertices to pack more triangles per meshlet
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        // Create a grid that shares vertices
        let grid_size = 20;
        for y in 0..grid_size {
            for x in 0..grid_size {
                positions.push([x as f32, y as f32, 0.0]);
            }
        }

        // Create triangles
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

        let data = MeshletData::generate(&positions, &indices, None);

        for (i, m) in data.meshlets.iter().enumerate() {
            assert!(
                m.triangle_count as usize <= MAX_MESHLET_TRIANGLES,
                "Meshlet {} has {} triangles, exceeds limit of {}",
                i, m.triangle_count, MAX_MESHLET_TRIANGLES
            );
        }
    }

    // -----------------------------------------------------------------------
    // Bounding sphere tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bounding_sphere_single_vertex() {
        let positions = vec![[1.0, 2.0, 3.0]];
        let indices = vec![0];
        let (center, radius) = compute_bounding_sphere(&positions, &indices);

        assert_eq!(center, [1.0, 2.0, 3.0]);
        assert_eq!(radius, 0.0);
    }

    #[test]
    fn test_bounding_sphere_two_vertices() {
        let positions = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let indices = vec![0, 1];
        let (center, radius) = compute_bounding_sphere(&positions, &indices);

        assert!((center[0] - 1.0).abs() < EPSILON);
        assert!(center[1].abs() < EPSILON);
        assert!(center[2].abs() < EPSILON);
        assert!((radius - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_bounding_sphere_contains_all_vertices() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ];
        let indices: Vec<u32> = (0..positions.len() as u32).collect();
        let (center, radius) = compute_bounding_sphere(&positions, &indices);

        for (i, p) in positions.iter().enumerate() {
            let dist = distance(center, *p);
            assert!(
                dist <= radius + EPSILON,
                "Vertex {} at {:?} outside sphere (dist={}, radius={})",
                i, p, dist, radius
            );
        }
    }

    // -----------------------------------------------------------------------
    // Normal cone tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_normal_cone_single_triangle() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let triangles = vec![[0, 1, 2]];
        let (axis, cutoff) = compute_normal_cone(&positions, &triangles);

        // Single triangle should have cutoff of 1.0 (all normals identical)
        assert_eq!(cutoff, 1.0);
        // Normal should point in +Z direction
        assert!((axis[2] - 1.0).abs() < EPSILON || (axis[2] + 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_normal_cone_opposing_triangles() {
        // Two triangles facing opposite directions
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 0.1],
            [1.0, 0.0, 0.1],
            [0.5, 1.0, 0.1],
        ];
        // First triangle: CCW from +Z -> normal +Z
        // Second triangle: CW from +Z -> normal -Z
        let triangles = vec![
            [0, 1, 2],  // +Z normal
            [5, 4, 3],  // -Z normal
        ];
        let (_axis, cutoff) = compute_normal_cone(&positions, &triangles);

        // Opposing normals should result in cutoff near -1 or invalid
        assert!(cutoff < 0.0);
    }

    #[test]
    fn test_normal_cone_contains_all_face_normals() {
        let (positions, indices) = make_cube();
        let triangles: Vec<[u32; 3]> = indices
            .chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let (axis, cutoff) = compute_normal_cone(&positions, &triangles);

        // Compute all face normals and verify they're within the cone
        for tri in &triangles {
            let p0 = positions[tri[0] as usize];
            let p1 = positions[tri[1] as usize];
            let p2 = positions[tri[2] as usize];

            let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let n = cross(e1, e2);
            let len = length(n);
            if len < EPSILON {
                continue;
            }

            let face_normal = [n[0] / len, n[1] / len, n[2] / len];
            let d = dot(axis, face_normal);

            assert!(
                d >= cutoff - EPSILON,
                "Face normal {:?} outside cone (dot={}, cutoff={})",
                face_normal, d, cutoff
            );
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_triangle() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2];
        let data = MeshletData::generate(&positions, &indices, None);

        assert_eq!(data.meshlet_count(), 1);
        assert_eq!(data.meshlets[0].vertex_count, 3);
        assert_eq!(data.meshlets[0].triangle_count, 1);
        assert!(data.validate(positions.len()).is_ok());
    }

    #[test]
    fn test_empty_mesh() {
        let data = MeshletData::generate(&[], &[], None);
        assert!(data.is_empty());
        assert_eq!(data.meshlet_count(), 0);
    }

    #[test]
    fn test_degenerate_triangle_skipped() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        // Degenerate triangle (two identical indices)
        let indices = vec![0, 0, 1, 0, 1, 2];
        let data = MeshletData::generate(&positions, &indices, None);

        // Should only have one valid triangle
        assert_eq!(data.meshlet_count(), 1);
        assert_eq!(data.meshlets[0].triangle_count, 1);
    }

    #[test]
    fn test_out_of_range_indices_skipped() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        // One valid triangle, one with out-of-range index
        let indices = vec![0, 1, 2, 0, 1, 100];
        let data = MeshletData::generate(&positions, &indices, None);

        // Should only have one valid triangle
        assert_eq!(data.meshlet_count(), 1);
        assert_eq!(data.meshlets[0].triangle_count, 1);
    }

    // -----------------------------------------------------------------------
    // Round-trip reconstruction
    // -----------------------------------------------------------------------

    #[test]
    fn test_reconstruct_indices_cube() {
        let (positions, indices) = make_cube();
        let data = MeshletData::generate(&positions, &indices, None);
        let reconstructed = data.reconstruct_indices();

        // Should have same number of indices
        assert_eq!(reconstructed.len(), indices.len());

        // Verify all indices are valid
        for idx in &reconstructed {
            assert!((*idx as usize) < positions.len());
        }

        // Verify we have the same triangles (not necessarily in order)
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

    #[test]
    fn test_reconstruct_indices_large_mesh() {
        let (positions, indices) = make_large_mesh(100);
        let data = MeshletData::generate(&positions, &indices, None);
        let reconstructed = data.reconstruct_indices();

        assert_eq!(reconstructed.len(), indices.len());

        // Verify same triangles exist
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
    // Validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_valid_data() {
        let (positions, indices) = make_cube();
        let data = MeshletData::generate(&positions, &indices, None);
        assert!(data.validate(positions.len()).is_ok());
    }

    #[test]
    fn test_validate_catches_vertex_index_out_of_range() {
        let (positions, indices) = make_cube();
        let mut data = MeshletData::generate(&positions, &indices, None);

        // Corrupt a vertex index
        data.vertex_indices[0] = 999;

        assert!(data.validate(positions.len()).is_err());
    }

    #[test]
    fn test_validate_catches_local_index_out_of_range() {
        let (positions, indices) = make_cube();
        let mut data = MeshletData::generate(&positions, &indices, None);

        // Corrupt a local index (must be < vertex_count which is <= 8)
        data.local_indices[0] = 200;

        assert!(data.validate(positions.len()).is_err());
    }

    // -----------------------------------------------------------------------
    // With normals
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_with_normals() {
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
        let data = MeshletData::generate(&positions, &indices, Some(&normals));

        assert_eq!(data.meshlet_count(), 1);
        assert!(data.bounds[0].has_valid_cone());
        // With all normals pointing +Z, cone should be tight
        assert!(data.bounds[0].cone_cutoff > 0.9);
    }

    // -----------------------------------------------------------------------
    // Bytemuck traits
    // -----------------------------------------------------------------------

    #[test]
    fn test_meshlet_pod_zeroable() {
        // Verify bytemuck traits work
        let m = Meshlet::new(1, 2, 3, 4);
        let bytes: &[u8] = bytemuck::bytes_of(&m);
        assert_eq!(bytes.len(), MESHLET_SIZE);

        let zeros = Meshlet::zeroed();
        assert_eq!(zeros.vertex_offset, 0);
        assert_eq!(zeros.triangle_offset, 0);
        assert_eq!(zeros.vertex_count, 0);
        assert_eq!(zeros.triangle_count, 0);
    }

    #[test]
    fn test_meshlet_bounds_pod_zeroable() {
        let b = MeshletBounds::new([1.0, 2.0, 3.0], 4.0, [0.0, 1.0, 0.0], 0.5);
        let bytes: &[u8] = bytemuck::bytes_of(&b);
        assert_eq!(bytes.len(), MESHLET_BOUNDS_SIZE);

        let zeros = MeshletBounds::zeroed();
        assert_eq!(zeros.center, [0.0, 0.0, 0.0]);
        assert_eq!(zeros.radius, 0.0);
    }

    // -----------------------------------------------------------------------
    // Stress tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stress_many_small_triangles() {
        // 1000 triangles, each with unique vertices
        let (positions, indices) = make_large_mesh(1000);
        let data = MeshletData::generate(&positions, &indices, None);

        assert!(data.meshlet_count() > 10);
        assert!(data.validate(positions.len()).is_ok());

        // Verify reconstruction
        let reconstructed = data.reconstruct_indices();
        assert_eq!(reconstructed.len(), indices.len());
    }

    #[test]
    fn test_stress_mesh_with_high_vertex_reuse() {
        // Grid mesh with high vertex sharing
        let grid_size = 50;
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

        let data = MeshletData::generate(&positions, &indices, None);
        assert!(data.validate(positions.len()).is_ok());

        // With high vertex reuse, should pack more efficiently
        let total_triangles: usize = data.meshlets.iter()
            .map(|m| m.triangle_count as usize)
            .sum();
        assert_eq!(total_triangles, indices.len() / 3);
    }

    // -----------------------------------------------------------------------
    // Associated constants tests (T-WGPU-P6.9.1)
    // -----------------------------------------------------------------------

    #[test]
    fn test_meshlet_associated_constants() {
        // Verify associated constants match module-level constants
        assert_eq!(Meshlet::MAX_VERTICES as usize, MAX_MESHLET_VERTICES);
        assert_eq!(Meshlet::MAX_TRIANGLES as usize, MAX_MESHLET_TRIANGLES);

        // Verify typical GPU mesh shader limits
        assert_eq!(Meshlet::MAX_VERTICES, 64);
        assert_eq!(Meshlet::MAX_TRIANGLES, 124);
    }

    // -----------------------------------------------------------------------
    // Builder-style methods tests (T-WGPU-P6.9.1)
    // -----------------------------------------------------------------------

    #[test]
    fn test_meshlet_bounds_with_bounds() {
        let bounds = MeshletBounds::default()
            .with_bounds([1.0, 2.0, 3.0], 5.0);

        assert_eq!(bounds.center, [1.0, 2.0, 3.0]);
        assert_eq!(bounds.radius, 5.0);
        // Default cone values should be preserved
        assert_eq!(bounds.cone_axis, [0.0, 0.0, 0.0]);
        assert_eq!(bounds.cone_cutoff, 0.0);
    }

    #[test]
    fn test_meshlet_bounds_with_cone() {
        let bounds = MeshletBounds::default()
            .with_cone([0.0, 1.0, 0.0], 0.5);

        assert_eq!(bounds.cone_axis, [0.0, 1.0, 0.0]);
        assert_eq!(bounds.cone_cutoff, 0.5);
        // Default center/radius should be preserved
        assert_eq!(bounds.center, [0.0, 0.0, 0.0]);
        assert_eq!(bounds.radius, 0.0);
    }

    #[test]
    fn test_meshlet_bounds_builder_chain() {
        let bounds = MeshletBounds::default()
            .with_bounds([1.0, 2.0, 3.0], 5.0)
            .with_cone([0.0, 0.0, 1.0], 0.707);

        // Verify all values set correctly
        assert_eq!(bounds.center, [1.0, 2.0, 3.0]);
        assert_eq!(bounds.radius, 5.0);
        assert_eq!(bounds.cone_axis, [0.0, 0.0, 1.0]);
        assert!((bounds.cone_cutoff - 0.707).abs() < EPSILON);

        // Should be valid for culling
        assert!(bounds.has_valid_cone());
    }

    #[test]
    fn test_meshlet_bounds_builder_disable_cone() {
        // Use with_cone to disable backface culling
        let bounds = MeshletBounds::default()
            .with_bounds([0.0, 0.0, 0.0], 1.0)
            .with_cone([0.0, 0.0, 1.0], -2.0); // -2.0 disables culling

        assert!(!bounds.has_valid_cone());
    }

    // -----------------------------------------------------------------------
    // GPU layout compatibility tests (T-WGPU-P6.9.1)
    // -----------------------------------------------------------------------

    #[test]
    fn test_meshlet_gpu_layout() {
        // Meshlet should be 12 bytes with 4-byte alignment
        assert_eq!(mem::size_of::<Meshlet>(), 12);
        assert_eq!(mem::align_of::<Meshlet>(), 4);

        // Verify field offsets match expected GPU layout
        // Using bytemuck to verify actual byte layout
        let m = Meshlet::new(0x11223344, 0x55667788, 0xAA, 0xBB);
        let bytes = bytemuck::bytes_of(&m);

        // vertex_offset at offset 0 (4 bytes, little-endian)
        assert_eq!(&bytes[0..4], &[0x44, 0x33, 0x22, 0x11]);
        // triangle_offset at offset 4 (4 bytes, little-endian)
        assert_eq!(&bytes[4..8], &[0x88, 0x77, 0x66, 0x55]);
        // vertex_count at offset 8 (1 byte)
        assert_eq!(bytes[8], 0xAA);
        // triangle_count at offset 9 (1 byte)
        assert_eq!(bytes[9], 0xBB);
        // padding at offset 10-11
        assert_eq!(&bytes[10..12], &[0, 0]);
    }

    #[test]
    fn test_meshlet_bounds_gpu_layout() {
        // MeshletBounds should be 32 bytes with 4-byte alignment (vec4 compatible)
        assert_eq!(mem::size_of::<MeshletBounds>(), 32);
        assert_eq!(mem::align_of::<MeshletBounds>(), 4);

        // Total size should be exactly 2 vec4s (2 * 16 bytes)
        // This ensures proper alignment in GPU storage buffers
        assert_eq!(mem::size_of::<MeshletBounds>(), 2 * 16);
    }

    #[test]
    fn test_meshlet_array_stride() {
        // Arrays of Meshlet should have no padding between elements
        let array: [Meshlet; 4] = [Meshlet::default(); 4];
        let array_size = mem::size_of_val(&array);
        assert_eq!(array_size, 4 * MESHLET_SIZE);

        // Same for MeshletBounds
        let bounds_array: [MeshletBounds; 4] = [MeshletBounds::default(); 4];
        let bounds_size = mem::size_of_val(&bounds_array);
        assert_eq!(bounds_size, 4 * MESHLET_BOUNDS_SIZE);
    }

    #[test]
    fn test_meshlet_cast_to_bytes() {
        // Verify we can safely cast arrays to byte slices for GPU upload
        let meshlets = vec![
            Meshlet::new(0, 0, 10, 5),
            Meshlet::new(10, 15, 20, 10),
        ];
        let bytes: &[u8] = bytemuck::cast_slice(&meshlets);
        assert_eq!(bytes.len(), 2 * MESHLET_SIZE);

        // Same for bounds
        let bounds = vec![
            MeshletBounds::sphere_only([0.0, 0.0, 0.0], 1.0),
            MeshletBounds::new([1.0, 1.0, 1.0], 2.0, [0.0, 1.0, 0.0], 0.5),
        ];
        let bounds_bytes: &[u8] = bytemuck::cast_slice(&bounds);
        assert_eq!(bounds_bytes.len(), 2 * MESHLET_BOUNDS_SIZE);
    }
}

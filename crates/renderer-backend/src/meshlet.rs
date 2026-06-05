//! Meshlet generation for GPU-driven rendering
//!
//! This module implements meshlet partitioning for efficient GPU culling and rendering.
//! Meshlets are small clusters of triangles (max 64 vertices, 124 triangles) that can
//! be processed independently by mesh shaders.
//!
//! # Features
//!
//! - Morton (Z-order) spatial sorting for cache locality
//! - Tight bounding sphere computation (Ritter's algorithm)
//! - Normal cone computation for backface culling
//! - Integration with glTF mesh loader
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::meshlet::{MeshletBuilder, MAX_MESHLET_VERTICES, MAX_MESHLET_TRIANGLES};
//!
//! let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
//! let indices = vec![0, 1, 2];
//!
//! let builder = MeshletBuilder::new(positions, indices);
//! let meshlets = builder.build();
//!
//! for meshlet in &meshlets {
//!     assert!(meshlet.vertices.len() <= MAX_MESHLET_VERTICES);
//!     assert!(meshlet.triangle_count() <= MAX_MESHLET_TRIANGLES);
//! }
//! ```

use std::collections::HashMap;

use crate::gltf::{GltfPrimitive, IndexBuffer, IndexFormat, VertexAttribute, VertexSemantic};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum unique vertices per meshlet (hardware limit for mesh shaders).
pub const MAX_MESHLET_VERTICES: usize = 64;

/// Maximum triangles per meshlet (hardware limit for mesh shaders).
pub const MAX_MESHLET_TRIANGLES: usize = 124;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A small cluster of triangles for GPU-driven rendering.
///
/// Meshlets are designed to fit within hardware limits for mesh shaders:
/// - Up to 64 unique vertices
/// - Up to 124 triangles (372 indices)
///
/// Each meshlet has its own bounding sphere for frustum/occlusion culling
/// and an optional normal cone for backface culling.
#[derive(Debug, Clone)]
pub struct Meshlet {
    /// Local vertex data (up to 64 vertices).
    pub vertices: Vec<[f32; 3]>,
    /// Triangle indices into local vertex buffer.
    /// Each triangle is 3 consecutive u8 indices (up to 124 triangles * 3 = 372 indices).
    pub indices: Vec<u8>,
    /// Bounding sphere center.
    pub center: [f32; 3],
    /// Bounding sphere radius.
    pub radius: f32,
    /// Normal cone axis for backface culling (normalized).
    pub cone_axis: [f32; 3],
    /// Normal cone cutoff (cos of half-angle). If dot(view, axis) < cutoff, entire meshlet is backfacing.
    pub cone_cutoff: f32,
}

impl Meshlet {
    /// Create an empty meshlet.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            center: [0.0, 0.0, 0.0],
            radius: 0.0,
            cone_axis: [0.0, 0.0, 1.0],
            cone_cutoff: -1.0, // Default: no culling (cone covers full sphere)
        }
    }

    /// Number of triangles in this meshlet.
    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Number of vertices in this meshlet.
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Check if adding a triangle would exceed limits.
    pub fn can_add_triangle(&self, new_vertex_count: usize) -> bool {
        let would_have_vertices = self.vertices.len() + new_vertex_count;
        let would_have_triangles = self.triangle_count() + 1;
        would_have_vertices <= MAX_MESHLET_VERTICES && would_have_triangles <= MAX_MESHLET_TRIANGLES
    }
}

impl Default for Meshlet {
    fn default() -> Self {
        Self::new()
    }
}

/// Meshlet generation error.
#[derive(Debug)]
pub enum MeshletError {
    /// Invalid index count (not a multiple of 3).
    InvalidIndexCount(usize),
    /// Index out of bounds.
    IndexOutOfBounds { index: u32, vertex_count: usize },
    /// Missing position attribute.
    MissingPositions,
    /// Invalid position data format.
    InvalidPositionFormat,
}

impl std::fmt::Display for MeshletError {
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
            Self::MissingPositions => write!(f, "missing position attribute"),
            Self::InvalidPositionFormat => write!(f, "position attribute must be VEC3 F32"),
        }
    }
}

impl std::error::Error for MeshletError {}

// ---------------------------------------------------------------------------
// MeshletBuilder
// ---------------------------------------------------------------------------

/// Builder for generating meshlets from mesh data.
///
/// Takes vertex positions and triangle indices, partitions them into meshlets
/// with Morton order spatial sorting for cache locality.
pub struct MeshletBuilder {
    /// Vertex positions from source mesh.
    positions: Vec<[f32; 3]>,
    /// Triangle indices (3 per triangle).
    indices: Vec<u32>,
}

impl MeshletBuilder {
    /// Create a new meshlet builder from vertex positions and indices.
    ///
    /// # Arguments
    ///
    /// * `positions` - Vertex positions (x, y, z).
    /// * `indices` - Triangle indices, must be a multiple of 3.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    /// let indices = vec![0, 1, 2];
    /// let builder = MeshletBuilder::new(positions, indices);
    /// ```
    pub fn new(positions: Vec<[f32; 3]>, indices: Vec<u32>) -> Self {
        Self { positions, indices }
    }

    /// Create a meshlet builder from a glTF primitive.
    ///
    /// Extracts position data and indices from the primitive.
    pub fn from_gltf_primitive(primitive: &GltfPrimitive) -> Result<Self, MeshletError> {
        // Extract positions
        let positions = extract_positions(&primitive.attributes)?;

        // Extract indices (or generate sequential indices if none)
        let indices = if let Some(ref index_buffer) = primitive.indices {
            extract_indices(index_buffer)
        } else {
            // No index buffer: sequential triangles
            (0..positions.len() as u32).collect()
        };

        Ok(Self { positions, indices })
    }

    /// Build meshlets with Morton order sorting for spatial locality.
    ///
    /// Triangles are sorted by the Morton code of their centroid before
    /// partitioning, which improves cache coherence during rendering.
    pub fn build(&self) -> Result<Vec<Meshlet>, MeshletError> {
        self.validate()?;

        if self.indices.is_empty() {
            return Ok(Vec::new());
        }

        // Compute mesh bounds for Morton encoding
        let (min, max) = compute_bounds(&self.positions);

        // Compute Morton codes for each triangle (by centroid)
        let mut triangles: Vec<(usize, u32)> = (0..self.indices.len() / 3)
            .map(|tri_idx| {
                let i0 = self.indices[tri_idx * 3] as usize;
                let i1 = self.indices[tri_idx * 3 + 1] as usize;
                let i2 = self.indices[tri_idx * 3 + 2] as usize;

                // Triangle centroid
                let cx = (self.positions[i0][0] + self.positions[i1][0] + self.positions[i2][0]) / 3.0;
                let cy = (self.positions[i0][1] + self.positions[i1][1] + self.positions[i2][1]) / 3.0;
                let cz = (self.positions[i0][2] + self.positions[i1][2] + self.positions[i2][2]) / 3.0;

                let morton = morton_code(cx, cy, cz, min, max);
                (tri_idx, morton)
            })
            .collect();

        // Sort triangles by Morton code
        triangles.sort_by_key(|(_, morton)| *morton);

        // Build meshlets from sorted triangles
        self.build_from_sorted(&triangles)
    }

    /// Build meshlets without sorting (for testing or when order matters).
    pub fn build_unsorted(&self) -> Result<Vec<Meshlet>, MeshletError> {
        self.validate()?;

        if self.indices.is_empty() {
            return Ok(Vec::new());
        }

        // Sequential triangle order
        let triangles: Vec<(usize, u32)> = (0..self.indices.len() / 3).map(|i| (i, 0)).collect();

        self.build_from_sorted(&triangles)
    }

    /// Validate input data.
    fn validate(&self) -> Result<(), MeshletError> {
        if self.indices.len() % 3 != 0 {
            return Err(MeshletError::InvalidIndexCount(self.indices.len()));
        }

        for &idx in &self.indices {
            if idx as usize >= self.positions.len() {
                return Err(MeshletError::IndexOutOfBounds {
                    index: idx,
                    vertex_count: self.positions.len(),
                });
            }
        }

        Ok(())
    }

    /// Build meshlets from pre-sorted triangle list.
    fn build_from_sorted(&self, triangles: &[(usize, u32)]) -> Result<Vec<Meshlet>, MeshletError> {
        let mut meshlets: Vec<Meshlet> = Vec::new();
        let mut current = Meshlet::new();
        let mut vertex_map: HashMap<u32, u8> = HashMap::new();

        for &(tri_idx, _) in triangles {
            let i0 = self.indices[tri_idx * 3];
            let i1 = self.indices[tri_idx * 3 + 1];
            let i2 = self.indices[tri_idx * 3 + 2];

            // Count how many new vertices this triangle would add
            let mut new_verts = 0;
            if !vertex_map.contains_key(&i0) {
                new_verts += 1;
            }
            if !vertex_map.contains_key(&i1) {
                new_verts += 1;
            }
            if !vertex_map.contains_key(&i2) {
                new_verts += 1;
            }

            // Check if triangle fits in current meshlet
            if !current.can_add_triangle(new_verts) {
                // Finalize current meshlet
                finalize_meshlet(&mut current);
                meshlets.push(current);

                // Start new meshlet
                current = Meshlet::new();
                vertex_map.clear();
            }

            // Add vertices and get local indices
            let local_i0 = get_or_insert_vertex(&mut current, &mut vertex_map, i0, &self.positions);
            let local_i1 = get_or_insert_vertex(&mut current, &mut vertex_map, i1, &self.positions);
            let local_i2 = get_or_insert_vertex(&mut current, &mut vertex_map, i2, &self.positions);

            // Add triangle indices
            current.indices.push(local_i0);
            current.indices.push(local_i1);
            current.indices.push(local_i2);
        }

        // Finalize last meshlet
        if !current.indices.is_empty() {
            finalize_meshlet(&mut current);
            meshlets.push(current);
        }

        Ok(meshlets)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Get or insert a vertex into the meshlet, returning the local index.
fn get_or_insert_vertex(
    meshlet: &mut Meshlet,
    vertex_map: &mut HashMap<u32, u8>,
    global_idx: u32,
    positions: &[[f32; 3]],
) -> u8 {
    if let Some(&local_idx) = vertex_map.get(&global_idx) {
        local_idx
    } else {
        let local_idx = meshlet.vertices.len() as u8;
        meshlet.vertices.push(positions[global_idx as usize]);
        vertex_map.insert(global_idx, local_idx);
        local_idx
    }
}

/// Finalize a meshlet by computing bounding sphere and normal cone.
fn finalize_meshlet(meshlet: &mut Meshlet) {
    if meshlet.vertices.is_empty() {
        return;
    }

    // Compute bounding sphere
    let (center, radius) = compute_bounding_sphere(&meshlet.vertices);
    meshlet.center = center;
    meshlet.radius = radius;

    // Compute normal cone
    let (axis, cutoff) = compute_normal_cone(meshlet);
    meshlet.cone_axis = axis;
    meshlet.cone_cutoff = cutoff;
}

/// Compute tight bounding sphere using Ritter's algorithm.
///
/// 1. Find the two most distant points along any axis
/// 2. Create initial sphere from those points
/// 3. Expand sphere to include any points outside
pub fn compute_bounding_sphere(vertices: &[[f32; 3]]) -> ([f32; 3], f32) {
    if vertices.is_empty() {
        return ([0.0, 0.0, 0.0], 0.0);
    }

    if vertices.len() == 1 {
        return (vertices[0], 0.0);
    }

    // Step 1: Find extremes along each axis
    let mut min_x = 0;
    let mut max_x = 0;
    let mut min_y = 0;
    let mut max_y = 0;
    let mut min_z = 0;
    let mut max_z = 0;

    for (i, v) in vertices.iter().enumerate() {
        if v[0] < vertices[min_x][0] {
            min_x = i;
        }
        if v[0] > vertices[max_x][0] {
            max_x = i;
        }
        if v[1] < vertices[min_y][1] {
            min_y = i;
        }
        if v[1] > vertices[max_y][1] {
            max_y = i;
        }
        if v[2] < vertices[min_z][2] {
            min_z = i;
        }
        if v[2] > vertices[max_z][2] {
            max_z = i;
        }
    }

    // Find the pair with maximum span
    let dx = distance_sq(&vertices[min_x], &vertices[max_x]);
    let dy = distance_sq(&vertices[min_y], &vertices[max_y]);
    let dz = distance_sq(&vertices[min_z], &vertices[max_z]);

    let (p1, p2) = if dx >= dy && dx >= dz {
        (&vertices[min_x], &vertices[max_x])
    } else if dy >= dz {
        (&vertices[min_y], &vertices[max_y])
    } else {
        (&vertices[min_z], &vertices[max_z])
    };

    // Step 2: Initial sphere from diameter
    let mut center = [
        (p1[0] + p2[0]) * 0.5,
        (p1[1] + p2[1]) * 0.5,
        (p1[2] + p2[2]) * 0.5,
    ];
    let mut radius_sq = distance_sq(p1, &center);
    let mut radius = radius_sq.sqrt();

    // Step 3: Expand to include all points
    for v in vertices {
        let dist_sq = distance_sq(v, &center);
        if dist_sq > radius_sq {
            let dist = dist_sq.sqrt();
            let new_radius = (radius + dist) * 0.5;
            let k = (new_radius - radius) / dist;

            center[0] += (v[0] - center[0]) * k;
            center[1] += (v[1] - center[1]) * k;
            center[2] += (v[2] - center[2]) * k;

            radius = new_radius;
            radius_sq = radius * radius;
        }
    }

    (center, radius)
}

/// Compute normal cone for backface culling.
///
/// Returns (axis, cutoff) where:
/// - axis: average normal direction (normalized)
/// - cutoff: minimum cos(angle) between any triangle normal and axis
///
/// If dot(view_dir, axis) < cutoff for a camera direction, entire meshlet is backfacing.
fn compute_normal_cone(meshlet: &Meshlet) -> ([f32; 3], f32) {
    if meshlet.indices.len() < 3 {
        return ([0.0, 0.0, 1.0], -1.0);
    }

    // Compute all triangle normals
    let mut normals: Vec<[f32; 3]> = Vec::new();
    for tri in meshlet.indices.chunks(3) {
        let v0 = &meshlet.vertices[tri[0] as usize];
        let v1 = &meshlet.vertices[tri[1] as usize];
        let v2 = &meshlet.vertices[tri[2] as usize];

        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        let n = cross(&e1, &e2);
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();

        if len > 1e-8 {
            normals.push([n[0] / len, n[1] / len, n[2] / len]);
        }
    }

    if normals.is_empty() {
        return ([0.0, 0.0, 1.0], -1.0);
    }

    // Compute average normal (cone axis)
    let mut axis = [0.0f32, 0.0, 0.0];
    for n in &normals {
        axis[0] += n[0];
        axis[1] += n[1];
        axis[2] += n[2];
    }
    let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if len < 1e-8 {
        return ([0.0, 0.0, 1.0], -1.0);
    }
    axis[0] /= len;
    axis[1] /= len;
    axis[2] /= len;

    // Find minimum dot product (maximum angle)
    let mut min_dot = 1.0f32;
    for n in &normals {
        let d = axis[0] * n[0] + axis[1] * n[1] + axis[2] * n[2];
        min_dot = min_dot.min(d);
    }

    (axis, min_dot)
}

/// Compute Morton code (Z-order curve) for a 3D point.
///
/// The point is normalized to [0, 1] range based on mesh bounds,
/// then quantized to 10 bits per axis (30-bit Morton code).
fn morton_code(x: f32, y: f32, z: f32, min: [f32; 3], max: [f32; 3]) -> u32 {
    // Normalize to [0, 1]
    let range = [
        (max[0] - min[0]).max(1e-8),
        (max[1] - min[1]).max(1e-8),
        (max[2] - min[2]).max(1e-8),
    ];

    let nx = ((x - min[0]) / range[0]).clamp(0.0, 1.0);
    let ny = ((y - min[1]) / range[1]).clamp(0.0, 1.0);
    let nz = ((z - min[2]) / range[2]).clamp(0.0, 1.0);

    // Quantize to 10 bits
    let ix = (nx * 1023.0) as u32;
    let iy = (ny * 1023.0) as u32;
    let iz = (nz * 1023.0) as u32;

    // Interleave bits
    spread_bits(ix) | (spread_bits(iy) << 1) | (spread_bits(iz) << 2)
}

/// Spread bits for Morton encoding (10 bits -> 30 bits with gaps).
#[inline]
fn spread_bits(mut x: u32) -> u32 {
    // Spread 10 bits into 30 bits with 2-bit gaps
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;
    x
}

/// Compute axis-aligned bounding box of vertices.
fn compute_bounds(vertices: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    if vertices.is_empty() {
        return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    }

    let mut min = vertices[0];
    let mut max = vertices[0];

    for v in vertices.iter().skip(1) {
        min[0] = min[0].min(v[0]);
        min[1] = min[1].min(v[1]);
        min[2] = min[2].min(v[2]);
        max[0] = max[0].max(v[0]);
        max[1] = max[1].max(v[1]);
        max[2] = max[2].max(v[2]);
    }

    (min, max)
}

/// Squared distance between two points.
#[inline]
fn distance_sq(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    dx * dx + dy * dy + dz * dz
}

/// Cross product of two vectors.
#[inline]
fn cross(a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Extract position data from vertex attributes.
fn extract_positions(
    attributes: &HashMap<VertexSemantic, VertexAttribute>,
) -> Result<Vec<[f32; 3]>, MeshletError> {
    let pos_attr = attributes
        .get(&VertexSemantic::Position)
        .ok_or(MeshletError::MissingPositions)?;

    // Verify format: must be VEC3 F32
    use crate::gltf::{AttributeType, ComponentType};
    if pos_attr.attribute_type != AttributeType::Vec3
        || pos_attr.component_type != ComponentType::F32
    {
        return Err(MeshletError::InvalidPositionFormat);
    }

    // Extract positions
    let mut positions = Vec::with_capacity(pos_attr.count);
    let stride = if pos_attr.stride > 0 {
        pos_attr.stride
    } else {
        12 // 3 * sizeof(f32)
    };

    for i in 0..pos_attr.count {
        let offset = i * stride;
        if offset + 12 > pos_attr.data.len() {
            break;
        }

        let x = f32::from_le_bytes([
            pos_attr.data[offset],
            pos_attr.data[offset + 1],
            pos_attr.data[offset + 2],
            pos_attr.data[offset + 3],
        ]);
        let y = f32::from_le_bytes([
            pos_attr.data[offset + 4],
            pos_attr.data[offset + 5],
            pos_attr.data[offset + 6],
            pos_attr.data[offset + 7],
        ]);
        let z = f32::from_le_bytes([
            pos_attr.data[offset + 8],
            pos_attr.data[offset + 9],
            pos_attr.data[offset + 10],
            pos_attr.data[offset + 11],
        ]);

        positions.push([x, y, z]);
    }

    Ok(positions)
}

/// Extract indices from index buffer.
fn extract_indices(buffer: &IndexBuffer) -> Vec<u32> {
    let mut indices = Vec::with_capacity(buffer.count);

    match buffer.format {
        IndexFormat::U8 => {
            for &byte in &buffer.data {
                indices.push(byte as u32);
            }
        }
        IndexFormat::U16 => {
            for chunk in buffer.data.chunks_exact(2) {
                indices.push(u16::from_le_bytes([chunk[0], chunk[1]]) as u32);
            }
        }
        IndexFormat::U32 => {
            for chunk in buffer.data.chunks_exact(4) {
                indices.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
    }

    indices
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a simple triangle mesh.
    fn make_triangle() -> (Vec<[f32; 3]>, Vec<u32>) {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2];
        (positions, indices)
    }

    /// Helper: create a quad (2 triangles).
    fn make_quad() -> (Vec<[f32; 3]>, Vec<u32>) {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];
        (positions, indices)
    }

    /// Helper: create a large grid mesh that exceeds meshlet limits.
    fn make_large_grid(size: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        // Create a size x size grid of vertices
        for y in 0..size {
            for x in 0..size {
                positions.push([x as f32, y as f32, 0.0]);
            }
        }

        // Create triangles for each cell
        for y in 0..(size - 1) {
            for x in 0..(size - 1) {
                let i00 = (y * size + x) as u32;
                let i10 = (y * size + x + 1) as u32;
                let i01 = ((y + 1) * size + x) as u32;
                let i11 = ((y + 1) * size + x + 1) as u32;

                // Two triangles per cell
                indices.push(i00);
                indices.push(i10);
                indices.push(i11);

                indices.push(i00);
                indices.push(i11);
                indices.push(i01);
            }
        }

        (positions, indices)
    }

    #[test]
    fn test_meshlet_small_mesh() {
        let (positions, indices) = make_triangle();
        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        assert_eq!(meshlets.len(), 1);
        assert_eq!(meshlets[0].vertex_count(), 3);
        assert_eq!(meshlets[0].triangle_count(), 1);
    }

    #[test]
    fn test_meshlet_quad() {
        let (positions, indices) = make_quad();
        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        assert_eq!(meshlets.len(), 1);
        assert_eq!(meshlets[0].vertex_count(), 4);
        assert_eq!(meshlets[0].triangle_count(), 2);
    }

    #[test]
    fn test_meshlet_large_mesh() {
        // Create a mesh with many triangles
        let (positions, indices) = make_large_grid(20);
        let triangle_count = indices.len() / 3;
        assert!(triangle_count > MAX_MESHLET_TRIANGLES, "Test requires more triangles than meshlet limit");

        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        // Should create multiple meshlets
        assert!(meshlets.len() > 1, "Expected multiple meshlets, got {}", meshlets.len());

        // Verify each meshlet respects limits
        for (i, meshlet) in meshlets.iter().enumerate() {
            assert!(
                meshlet.vertex_count() <= MAX_MESHLET_VERTICES,
                "Meshlet {} has {} vertices (max {})",
                i,
                meshlet.vertex_count(),
                MAX_MESHLET_VERTICES
            );
            assert!(
                meshlet.triangle_count() <= MAX_MESHLET_TRIANGLES,
                "Meshlet {} has {} triangles (max {})",
                i,
                meshlet.triangle_count(),
                MAX_MESHLET_TRIANGLES
            );
        }

        // Verify total triangle count is preserved
        let total_triangles: usize = meshlets.iter().map(|m| m.triangle_count()).sum();
        assert_eq!(total_triangles, triangle_count);
    }

    #[test]
    fn test_meshlet_vertex_limit() {
        // Create mesh that stresses vertex limit
        // Each triangle uses 3 unique vertices to maximize vertex usage
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        // Create 100 triangles, each with 3 unique vertices
        for i in 0..100 {
            let base = (i * 3) as f32;
            positions.push([base, 0.0, 0.0]);
            positions.push([base + 1.0, 0.0, 0.0]);
            positions.push([base + 0.5, 1.0, 0.0]);

            indices.push((i * 3) as u32);
            indices.push((i * 3 + 1) as u32);
            indices.push((i * 3 + 2) as u32);
        }

        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        // Verify vertex limits
        for (i, meshlet) in meshlets.iter().enumerate() {
            assert!(
                meshlet.vertex_count() <= MAX_MESHLET_VERTICES,
                "Meshlet {} exceeds vertex limit: {} > {}",
                i,
                meshlet.vertex_count(),
                MAX_MESHLET_VERTICES
            );
        }

        // With 3 vertices per triangle and 64 max vertices, we get ~21 triangles per meshlet
        // 100 triangles should give us ~5 meshlets
        assert!(meshlets.len() >= 4, "Expected at least 4 meshlets for 100 disjoint triangles");
    }

    #[test]
    fn test_meshlet_triangle_limit() {
        // Create mesh with shared vertices to stress triangle limit
        // Hub-and-spoke pattern: center vertex shared by all triangles
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        // Center vertex
        positions.push([0.0, 0.0, 0.0]);

        // Create 200 triangles sharing the center vertex
        let num_triangles = 200;
        for i in 0..num_triangles {
            let angle = (i as f32) * std::f32::consts::TAU / (num_triangles as f32);
            let next_angle = ((i + 1) as f32) * std::f32::consts::TAU / (num_triangles as f32);

            positions.push([angle.cos(), angle.sin(), 0.0]);
            positions.push([next_angle.cos(), next_angle.sin(), 0.0]);

            indices.push(0); // center
            indices.push((i * 2 + 1) as u32);
            indices.push((i * 2 + 2) as u32);
        }

        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        // Verify triangle limits
        for (i, meshlet) in meshlets.iter().enumerate() {
            assert!(
                meshlet.triangle_count() <= MAX_MESHLET_TRIANGLES,
                "Meshlet {} exceeds triangle limit: {} > {}",
                i,
                meshlet.triangle_count(),
                MAX_MESHLET_TRIANGLES
            );
        }

        // 200 triangles / 124 max = at least 2 meshlets
        assert!(meshlets.len() >= 2, "Expected at least 2 meshlets for 200 triangles");
    }

    #[test]
    fn test_meshlet_morton_sorting() {
        // Create mesh with spatially separated triangles
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        // Create triangles at different spatial locations
        let locations = [
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
            [5.0, 5.0, 5.0],
            [0.0, 10.0, 0.0],
            [10.0, 0.0, 10.0],
        ];

        for (i, loc) in locations.iter().enumerate() {
            let base = i * 3;
            positions.push([loc[0], loc[1], loc[2]]);
            positions.push([loc[0] + 0.1, loc[1], loc[2]]);
            positions.push([loc[0], loc[1] + 0.1, loc[2]]);

            indices.push(base as u32);
            indices.push((base + 1) as u32);
            indices.push((base + 2) as u32);
        }

        let builder = MeshletBuilder::new(positions.clone(), indices.clone());
        let sorted_meshlets = builder.build().unwrap();
        let unsorted_meshlets = builder.build_unsorted().unwrap();

        // Both should produce valid meshlets
        assert!(!sorted_meshlets.is_empty());
        assert!(!unsorted_meshlets.is_empty());

        // Same number of triangles
        let sorted_tris: usize = sorted_meshlets.iter().map(|m| m.triangle_count()).sum();
        let unsorted_tris: usize = unsorted_meshlets.iter().map(|m| m.triangle_count()).sum();
        assert_eq!(sorted_tris, unsorted_tris);
    }

    #[test]
    fn test_meshlet_bounding_sphere() {
        let (positions, indices) = make_quad();
        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        let meshlet = &meshlets[0];

        // Quad spans [0,0,0] to [1,1,0]
        // Center should be approximately [0.5, 0.5, 0.0]
        let center = meshlet.center;
        assert!((center[0] - 0.5).abs() < 0.1, "Center X: {}", center[0]);
        assert!((center[1] - 0.5).abs() < 0.1, "Center Y: {}", center[1]);
        assert!(center[2].abs() < 0.1, "Center Z: {}", center[2]);

        // Radius should be approximately sqrt(0.5^2 + 0.5^2) = ~0.707
        let radius = meshlet.radius;
        assert!(radius > 0.6, "Radius too small: {}", radius);
        assert!(radius < 0.8, "Radius too large: {}", radius);

        // Verify all vertices are inside the bounding sphere
        for v in &meshlet.vertices {
            let dx = v[0] - center[0];
            let dy = v[1] - center[1];
            let dz = v[2] - center[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(
                dist <= radius + 1e-5,
                "Vertex {:?} outside bounding sphere (dist={}, radius={})",
                v,
                dist,
                radius
            );
        }
    }

    #[test]
    fn test_meshlet_bounding_sphere_single_vertex() {
        let vertices = vec![[1.0, 2.0, 3.0]];
        let (center, radius) = compute_bounding_sphere(&vertices);

        assert_eq!(center, [1.0, 2.0, 3.0]);
        assert_eq!(radius, 0.0);
    }

    #[test]
    fn test_meshlet_bounding_sphere_line() {
        let vertices = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
        let (center, radius) = compute_bounding_sphere(&vertices);

        assert!((center[0] - 5.0).abs() < 0.01);
        assert!((radius - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_meshlet_empty() {
        let positions: Vec<[f32; 3]> = Vec::new();
        let indices: Vec<u32> = Vec::new();

        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        assert!(meshlets.is_empty());
    }

    #[test]
    fn test_meshlet_invalid_index_count() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let indices = vec![0, 1]; // Not a multiple of 3

        let builder = MeshletBuilder::new(positions, indices);
        let result = builder.build();

        assert!(matches!(result, Err(MeshletError::InvalidIndexCount(2))));
    }

    #[test]
    fn test_meshlet_index_out_of_bounds() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let indices = vec![0, 1, 5]; // Index 5 is out of bounds

        let builder = MeshletBuilder::new(positions, indices);
        let result = builder.build();

        assert!(matches!(
            result,
            Err(MeshletError::IndexOutOfBounds { index: 5, vertex_count: 3 })
        ));
    }

    #[test]
    fn test_morton_code_corners() {
        let min = [0.0, 0.0, 0.0];
        let max = [1.0, 1.0, 1.0];

        // Origin should have Morton code 0
        let m0 = morton_code(0.0, 0.0, 0.0, min, max);
        assert_eq!(m0, 0);

        // Maximum should have Morton code with all bits set
        let m1 = morton_code(1.0, 1.0, 1.0, min, max);
        assert!(m1 > 0);

        // Points along X axis should have lower Morton codes than points along diagonal
        let mx = morton_code(0.5, 0.0, 0.0, min, max);
        let mxyz = morton_code(0.5, 0.5, 0.5, min, max);
        assert!(mx < mxyz);
    }

    #[test]
    fn test_normal_cone_flat_mesh() {
        // All triangles facing +Z
        let (positions, indices) = make_quad();
        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        let meshlet = &meshlets[0];

        // Normal cone axis should point roughly in +Z direction
        let axis = meshlet.cone_axis;
        assert!(axis[2].abs() > 0.9, "Expected Z-facing normal cone, got {:?}", axis);

        // Cutoff should be 1.0 (all normals identical, cone angle = 0)
        assert!(meshlet.cone_cutoff > 0.9, "Expected tight cone, got {}", meshlet.cone_cutoff);
    }

    #[test]
    fn test_spread_bits() {
        // Test spreading of bits
        assert_eq!(spread_bits(0), 0);
        assert_eq!(spread_bits(1), 1);
        assert_eq!(spread_bits(0b11), 0b1001); // 3 -> bits at positions 0 and 3
        assert_eq!(spread_bits(0b111), 0b1001001); // 7 -> bits at positions 0, 3, 6
    }

    #[test]
    fn test_meshlet_default() {
        let meshlet = Meshlet::default();
        assert!(meshlet.vertices.is_empty());
        assert!(meshlet.indices.is_empty());
        assert_eq!(meshlet.triangle_count(), 0);
        assert_eq!(meshlet.vertex_count(), 0);
    }

    #[test]
    fn test_can_add_triangle() {
        let mut meshlet = Meshlet::new();

        // Empty meshlet can add triangle
        assert!(meshlet.can_add_triangle(3));

        // Fill up to vertex limit
        for i in 0..MAX_MESHLET_VERTICES {
            meshlet.vertices.push([i as f32, 0.0, 0.0]);
        }

        // Can add triangle using existing vertices
        assert!(meshlet.can_add_triangle(0));

        // Cannot add triangle needing new vertex
        assert!(!meshlet.can_add_triangle(1));
    }

    // =========================================================================
    // Edge case tests (added by QA)
    // =========================================================================

    /// Test degenerate triangles with zero area (collinear vertices).
    #[test]
    fn test_degenerate_triangles_zero_area() {
        // Collinear points - triangle has zero area
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0], // All on X axis
        ];
        let indices = vec![0, 1, 2];

        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        assert_eq!(meshlets.len(), 1);
        assert_eq!(meshlets[0].triangle_count(), 1);

        // Normal cone should default to no-culling since normal is undefined
        // (cone_cutoff = -1.0 means no backface culling)
        assert!(
            meshlets[0].cone_cutoff <= -0.99,
            "Degenerate triangle should have no-cull cone, got cutoff={}",
            meshlets[0].cone_cutoff
        );
    }

    /// Test mesh with duplicate vertices at same position.
    #[test]
    fn test_duplicate_vertices() {
        // Three triangles, but vertices at same positions
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            // Duplicate of first triangle
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 3, 4, 5];

        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        // Should create valid meshlets (duplicates are treated as separate vertices)
        assert_eq!(meshlets.len(), 1);
        // 6 vertices because indices reference different positions in array
        assert_eq!(meshlets[0].vertex_count(), 6);
        assert_eq!(meshlets[0].triangle_count(), 2);
    }

    /// Test very large mesh (10K+ triangles) for performance and correctness.
    #[test]
    fn test_very_large_mesh() {
        // Create 100x100 grid = 10,000 vertices, ~20,000 triangles
        let (positions, indices) = make_large_grid(101);
        let triangle_count = indices.len() / 3;

        assert!(triangle_count >= 10000, "Test requires 10K+ triangles, got {}", triangle_count);

        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        // Verify constraints
        let mut total_triangles = 0;
        for (i, meshlet) in meshlets.iter().enumerate() {
            assert!(
                meshlet.vertex_count() <= MAX_MESHLET_VERTICES,
                "Meshlet {} exceeds vertex limit: {}",
                i,
                meshlet.vertex_count()
            );
            assert!(
                meshlet.triangle_count() <= MAX_MESHLET_TRIANGLES,
                "Meshlet {} exceeds triangle limit: {}",
                i,
                meshlet.triangle_count()
            );
            // Bounding sphere should be finite
            assert!(
                meshlet.radius.is_finite(),
                "Meshlet {} has non-finite radius: {}",
                i,
                meshlet.radius
            );
            assert!(
                meshlet.center.iter().all(|c| c.is_finite()),
                "Meshlet {} has non-finite center: {:?}",
                i,
                meshlet.center
            );
            total_triangles += meshlet.triangle_count();
        }

        // All triangles accounted for
        assert_eq!(total_triangles, triangle_count);

        // Should have created many meshlets
        // ~20,000 triangles / 124 max = ~162 meshlets minimum
        assert!(meshlets.len() >= 100, "Expected 100+ meshlets for 10K triangles, got {}", meshlets.len());
    }

    /// Test mesh exactly at meshlet limits (64 vertices, 124 triangles).
    #[test]
    fn test_mesh_exactly_at_limits() {
        // Create a fan mesh: 1 center + 63 outer vertices = 64 total
        // This allows up to 63 triangles sharing the center
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        // Center vertex
        positions.push([0.0, 0.0, 0.0]);

        // 63 outer vertices in a circle
        let num_outer = 63;
        for i in 0..num_outer {
            let angle = (i as f32) * std::f32::consts::TAU / (num_outer as f32);
            positions.push([angle.cos(), angle.sin(), 0.0]);
        }

        // 63 triangles (fan)
        for i in 0..num_outer {
            indices.push(0); // center
            indices.push((i + 1) as u32);
            indices.push(((i + 1) % num_outer + 1) as u32);
        }

        assert_eq!(positions.len(), 64);
        assert_eq!(indices.len() / 3, 63);

        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        // Should fit in exactly one meshlet (63 triangles < 124, 64 vertices = 64)
        assert_eq!(meshlets.len(), 1, "Expected exactly 1 meshlet");
        assert_eq!(meshlets[0].vertex_count(), 64);
        assert_eq!(meshlets[0].triangle_count(), 63);
    }

    /// Test mesh that just exceeds vertex limit (forces split).
    #[test]
    fn test_mesh_exceeds_vertex_limit_by_one() {
        // Create mesh with 65 unique vertices
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        // 22 triangles with 3 unique vertices each = 66 vertices
        // But we need exactly 65, so last triangle shares 1 vertex
        for i in 0..21 {
            let base = (i * 3) as f32;
            positions.push([base, 0.0, 0.0]);
            positions.push([base + 1.0, 0.0, 0.0]);
            positions.push([base + 0.5, 1.0, 0.0]);

            indices.push((i * 3) as u32);
            indices.push((i * 3 + 1) as u32);
            indices.push((i * 3 + 2) as u32);
        }
        // Add 22nd triangle sharing one vertex with 21st
        positions.push([64.0, 0.0, 0.0]);
        positions.push([64.5, 1.0, 0.0]);
        indices.push(62); // Shared vertex from triangle 21
        indices.push(63);
        indices.push(64);

        assert_eq!(positions.len(), 65);

        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        // Should split into 2 meshlets
        assert!(meshlets.len() >= 2, "Expected 2+ meshlets for 65 vertices, got {}", meshlets.len());

        // Verify no meshlet exceeds limits
        for meshlet in &meshlets {
            assert!(meshlet.vertex_count() <= MAX_MESHLET_VERTICES);
            assert!(meshlet.triangle_count() <= MAX_MESHLET_TRIANGLES);
        }
    }

    /// Test mesh that just exceeds triangle limit (forces split).
    #[test]
    fn test_mesh_exceeds_triangle_limit_by_one() {
        // Fan mesh with 125 triangles (just over 124 limit)
        // Using shared center vertex to minimize vertex count
        let mut positions = Vec::new();
        let mut indices = Vec::new();

        // Center vertex
        positions.push([0.0, 0.0, 0.0]);

        // 125 triangles requires 126 outer vertices (or 125 with wrapping)
        // But 126 > 64-1=63 max outer vertices, so we need multiple centers
        // Actually, let's just create 125 disjoint small triangles clustered together
        for i in 0..125 {
            let x = (i % 12) as f32;
            let y = (i / 12) as f32;
            positions.push([x, y, 0.0]);
            positions.push([x + 0.1, y, 0.0]);
            positions.push([x + 0.05, y + 0.1, 0.0]);

            indices.push((i * 3) as u32);
            indices.push((i * 3 + 1) as u32);
            indices.push((i * 3 + 2) as u32);
        }

        let triangle_count = indices.len() / 3;
        assert_eq!(triangle_count, 125);

        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        // Must split because 125 > 124
        // With unique vertices per triangle (375 verts), vertex limit (64) will force splits earlier
        assert!(meshlets.len() >= 2, "Expected 2+ meshlets for 125 triangles");

        // Verify limits
        for meshlet in &meshlets {
            assert!(meshlet.vertex_count() <= MAX_MESHLET_VERTICES);
            assert!(meshlet.triangle_count() <= MAX_MESHLET_TRIANGLES);
        }
    }

    /// Test bounding sphere tightness (all vertices should be inside).
    #[test]
    fn test_bounding_sphere_tightness() {
        // Create a complex 3D mesh (tetrahedron)
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.5, 0.5, 1.0],
        ];
        let indices = vec![
            0, 1, 2, // bottom
            0, 1, 3, // front
            1, 2, 3, // right
            2, 0, 3, // left
        ];

        let builder = MeshletBuilder::new(positions.clone(), indices);
        let meshlets = builder.build().unwrap();

        assert_eq!(meshlets.len(), 1);
        let meshlet = &meshlets[0];

        // All vertices must be inside bounding sphere
        for v in &positions {
            let dx = v[0] - meshlet.center[0];
            let dy = v[1] - meshlet.center[1];
            let dz = v[2] - meshlet.center[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            assert!(
                dist <= meshlet.radius + 1e-5,
                "Vertex {:?} outside sphere (dist={:.4}, radius={:.4})",
                v,
                dist,
                meshlet.radius
            );
        }

        // Radius should be tight (not excessively large)
        // Tetrahedron inscribed sphere radius is small, but bounding sphere should be ~0.8
        assert!(meshlet.radius < 1.5, "Bounding sphere too loose: {}", meshlet.radius);
    }

    /// Test normal cone for opposing faces (should have wide cone).
    #[test]
    fn test_normal_cone_opposing_faces() {
        // Create a "bowtie" with triangles facing opposite directions
        let positions = vec![
            // First triangle facing +Z
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            // Second triangle facing -Z (reversed winding)
            [2.0, 0.0, 0.0],
            [2.5, 1.0, 0.0],
            [3.0, 0.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 3, 4, 5];

        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        assert_eq!(meshlets.len(), 1);

        // With opposing normals, cutoff should be negative (wide cone)
        // Actually, both triangles are coplanar (Z=0), just different windings
        // One faces +Z, one faces -Z, so cutoff should be <= 0
        let cutoff = meshlets[0].cone_cutoff;
        assert!(
            cutoff <= 0.1,
            "Opposing faces should have wide cone (cutoff <= 0), got {}",
            cutoff
        );
    }

    /// Test Morton code ordering preserves spatial locality.
    #[test]
    fn test_morton_spatial_locality() {
        let min = [0.0, 0.0, 0.0];
        let max = [1.0, 1.0, 1.0];

        // Nearby points should have similar Morton codes
        let m_origin = morton_code(0.0, 0.0, 0.0, min, max);
        let m_near_origin = morton_code(0.01, 0.01, 0.01, min, max);
        let m_far = morton_code(0.99, 0.99, 0.99, min, max);

        // Origin and near-origin should be closer in Morton space than origin and far
        let diff_near = (m_origin as i64 - m_near_origin as i64).unsigned_abs();
        let diff_far = (m_origin as i64 - m_far as i64).unsigned_abs();

        assert!(
            diff_near < diff_far,
            "Morton codes should preserve locality: near_diff={} >= far_diff={}",
            diff_near,
            diff_far
        );
    }

    /// Test edge case: mesh with vertices at extreme coordinates.
    #[test]
    fn test_extreme_coordinates() {
        let positions = vec![
            [-1e6, -1e6, -1e6],
            [1e6, -1e6, -1e6],
            [0.0, 1e6, 0.0],
        ];
        let indices = vec![0, 1, 2];

        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        assert_eq!(meshlets.len(), 1);

        // Bounding sphere should encompass all vertices
        let meshlet = &meshlets[0];
        assert!(meshlet.radius.is_finite());
        assert!(meshlet.radius > 1e6, "Radius should be large for extreme coords: {}", meshlet.radius);

        // Note: At 1e6 scale, f32 precision is ~0.06, so use relative epsilon
        let epsilon = meshlet.radius * 1e-5; // Relative tolerance
        for v in &meshlet.vertices {
            let dx = v[0] - meshlet.center[0];
            let dy = v[1] - meshlet.center[1];
            let dz = v[2] - meshlet.center[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(
                dist <= meshlet.radius + epsilon,
                "Vertex {:?} outside sphere (dist={:.2}, radius={:.2}, epsilon={:.2})",
                v, dist, meshlet.radius, epsilon
            );
        }
    }

    /// Test edge case: very small triangles (micro-geometry).
    #[test]
    fn test_micro_triangles() {
        let eps = 1e-7;
        let positions = vec![
            [0.0, 0.0, 0.0],
            [eps, 0.0, 0.0],
            [0.0, eps, 0.0],
        ];
        let indices = vec![0, 1, 2];

        let builder = MeshletBuilder::new(positions, indices);
        let meshlets = builder.build().unwrap();

        assert_eq!(meshlets.len(), 1);
        assert_eq!(meshlets[0].triangle_count(), 1);

        // Bounding sphere should be tiny but valid
        let meshlet = &meshlets[0];
        assert!(meshlet.radius >= 0.0);
        assert!(meshlet.radius < 1e-5);
    }
}

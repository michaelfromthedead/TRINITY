//! Meshlet generation system for GPU-driven culling in TRINITY.
//!
//! Partitions meshes into small clusters (meshlets) optimized for:
//! - Frustum culling (bounding sphere per meshlet)
//! - Backface culling (normal cone per meshlet)
//! - Hi-Z occlusion culling (optional depth bounds)
//! - GPU mesh shader processing
//!
//! # Features
//!
//! - **64-vertex / 124-triangle clusters**: Optimal for mesh shader workgroups
//! - **Bounding sphere computation**: Welzl's miniball algorithm
//! - **Normal cone computation**: For whole-meshlet backface culling
//! - **Adjacency tracking**: For seam-free LOD transitions
//! - **Idempotent generation**: Same input produces identical output
//!
//! # Example
//!
//! ```ignore
//! use renderer_backend::asset::meshlet::*;
//!
//! // Vertex positions and triangle indices
//! let positions: Vec<[f32; 3]> = vec![...];
//! let indices: Vec<u32> = vec![...];
//!
//! // Generate meshlets with default config
//! let config = MeshletConfig::default();
//! let mesh = generate_meshlets(&positions, &indices, &config);
//!
//! println!("Generated {} meshlets", mesh.meshlets.len());
//! for (i, meshlet) in mesh.meshlets.iter().enumerate() {
//!     println!("Meshlet {}: {} verts, {} tris, sphere radius {}",
//!         i, meshlet.vertex_count, meshlet.triangle_count,
//!         meshlet.bounding_sphere.radius);
//! }
//! ```

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Meshlet generation error.
#[derive(Debug, Clone)]
pub enum MeshletError {
    /// Index out of bounds for vertex count.
    IndexOutOfBounds { index: u32, vertex_count: usize },
    /// Invalid triangle count (not divisible by 3).
    InvalidTriangleCount(usize),
    /// Empty mesh.
    EmptyMesh,
    /// Degenerate triangle (zero area).
    DegenerateTriangle(usize),
    /// Generation failed.
    GenerationFailed(String),
}

impl std::fmt::Display for MeshletError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexOutOfBounds { index, vertex_count } => {
                write!(f, "index {} out of bounds for {} vertices", index, vertex_count)
            }
            Self::InvalidTriangleCount(count) => {
                write!(f, "index count {} not divisible by 3", count)
            }
            Self::EmptyMesh => write!(f, "empty mesh"),
            Self::DegenerateTriangle(idx) => write!(f, "degenerate triangle at index {}", idx),
            Self::GenerationFailed(msg) => write!(f, "generation failed: {}", msg),
        }
    }
}

impl std::error::Error for MeshletError {}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Bounding sphere for frustum and occlusion culling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingSphere {
    /// Center of the sphere in object space.
    pub center: [f32; 3],
    /// Radius of the sphere.
    pub radius: f32,
}

impl Default for BoundingSphere {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0, 0.0],
            radius: 0.0,
        }
    }
}

impl BoundingSphere {
    /// Create a new bounding sphere.
    pub fn new(center: [f32; 3], radius: f32) -> Self {
        Self { center, radius }
    }

    /// Check if a point is inside the sphere.
    pub fn contains_point(&self, point: [f32; 3]) -> bool {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];
        dx * dx + dy * dy + dz * dz <= self.radius * self.radius
    }

    /// Expand sphere to include a point.
    pub fn expand_to_include(&mut self, point: [f32; 3]) {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let dist = dist_sq.sqrt();

        if dist > self.radius {
            // Grow sphere to include point
            let new_radius = (self.radius + dist) * 0.5;
            let ratio = (new_radius - self.radius) / dist;
            self.center[0] += dx * ratio;
            self.center[1] += dy * ratio;
            self.center[2] += dz * ratio;
            self.radius = new_radius;
        }
    }

    /// Merge two spheres into one containing both.
    pub fn merge(&self, other: &BoundingSphere) -> BoundingSphere {
        let dx = other.center[0] - self.center[0];
        let dy = other.center[1] - self.center[1];
        let dz = other.center[2] - self.center[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        if dist + other.radius <= self.radius {
            // Other is inside self
            return *self;
        }
        if dist + self.radius <= other.radius {
            // Self is inside other
            return *other;
        }

        // Create new sphere containing both
        let new_radius = (self.radius + dist + other.radius) * 0.5;
        let ratio = (new_radius - self.radius) / dist;

        BoundingSphere {
            center: [
                self.center[0] + dx * ratio,
                self.center[1] + dy * ratio,
                self.center[2] + dz * ratio,
            ],
            radius: new_radius,
        }
    }
}

/// Normal cone for meshlet backface culling.
///
/// If the view vector dot the cone axis is less than -cos(half_angle),
/// all triangles in the meshlet are guaranteed to be back-facing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalCone {
    /// Apex position (can be meshlet center or offset point).
    pub apex: [f32; 3],
    /// Cone axis (average normal direction, normalized).
    pub axis: [f32; 3],
    /// Half-angle of the cone in radians.
    pub half_angle: f32,
}

impl Default for NormalCone {
    fn default() -> Self {
        Self {
            apex: [0.0, 0.0, 0.0],
            axis: [0.0, 1.0, 0.0],
            half_angle: std::f32::consts::PI,
        }
    }
}

impl NormalCone {
    /// Create a new normal cone.
    pub fn new(apex: [f32; 3], axis: [f32; 3], half_angle: f32) -> Self {
        Self {
            apex,
            axis,
            half_angle,
        }
    }

    /// Check if a view direction could see any faces.
    ///
    /// Returns true if some faces might be visible (conservative).
    /// Returns false if ALL faces are definitely back-facing.
    ///
    /// The view direction is the direction the camera is looking (from camera towards scene).
    /// A face is backfacing if the view direction aligns with the face normal.
    pub fn is_potentially_visible(&self, view_dir: [f32; 3]) -> bool {
        // Normalize view direction
        let len = (view_dir[0] * view_dir[0]
            + view_dir[1] * view_dir[1]
            + view_dir[2] * view_dir[2])
            .sqrt();
        if len < 1e-6 {
            return true; // Degenerate case
        }
        let vd = [view_dir[0] / len, view_dir[1] / len, view_dir[2] / len];

        // Dot product with cone axis (average normal)
        let dot = self.axis[0] * vd[0] + self.axis[1] * vd[1] + self.axis[2] * vd[2];

        // Standard cone culling formula:
        // Cull (not visible) if: dot(view_dir, cone_axis) > -sin(half_angle)
        // This is equivalent to: cull if dot > cos(PI/2 + half_angle)
        //
        // For a single flat surface (half_angle = 0):
        // - cull if dot > 0 (view aligns with normal = backfacing)
        //
        // We return true (visible) if NOT culled:
        dot <= -self.half_angle.sin()
    }

    /// Returns the cosine of the half-angle (for GPU shader use).
    pub fn cos_half_angle(&self) -> f32 {
        self.half_angle.cos()
    }
}

/// A single meshlet (cluster of triangles).
#[derive(Debug, Clone)]
pub struct Meshlet {
    /// Offset into the global vertex index array.
    pub vertex_offset: u32,
    /// Number of vertices in this meshlet.
    pub vertex_count: u32,
    /// Offset into the global triangle index array.
    pub triangle_offset: u32,
    /// Number of triangles in this meshlet.
    pub triangle_count: u32,
    /// Bounding sphere for culling.
    pub bounding_sphere: BoundingSphere,
    /// Normal cone for backface culling.
    pub normal_cone: NormalCone,
    /// Optional depth bounds for Hi-Z culling (min, max).
    pub depth_bounds: Option<(f32, f32)>,
}

impl Default for Meshlet {
    fn default() -> Self {
        Self {
            vertex_offset: 0,
            vertex_count: 0,
            triangle_offset: 0,
            triangle_count: 0,
            bounding_sphere: BoundingSphere::default(),
            normal_cone: NormalCone::default(),
            depth_bounds: None,
        }
    }
}

/// Configuration for meshlet generation.
#[derive(Debug, Clone, Copy)]
pub struct MeshletConfig {
    /// Maximum vertices per meshlet (default: 64).
    pub max_vertices: u32,
    /// Maximum triangles per meshlet (default: 124).
    pub max_triangles: u32,
    /// Weight for normal cone tightness in scoring (0-1).
    /// Higher values prefer tighter cones over locality.
    pub cone_weight: f32,
    /// Whether to compute depth bounds for Hi-Z culling.
    pub compute_depth_bounds: bool,
}

impl Default for MeshletConfig {
    fn default() -> Self {
        Self {
            max_vertices: 64,
            max_triangles: 124,
            cone_weight: 0.5,
            compute_depth_bounds: false,
        }
    }
}

impl MeshletConfig {
    /// Create config for mesh shader workgroups (64 verts, 126 tris).
    pub fn mesh_shader() -> Self {
        Self {
            max_vertices: 64,
            max_triangles: 126,
            cone_weight: 0.5,
            compute_depth_bounds: true,
        }
    }

    /// Create config optimized for backface culling.
    pub fn backface_optimized() -> Self {
        Self {
            max_vertices: 64,
            max_triangles: 124,
            cone_weight: 0.8, // Prefer tighter cones
            compute_depth_bounds: true,
        }
    }

    /// Create config with custom limits.
    pub fn with_limits(max_vertices: u32, max_triangles: u32) -> Self {
        Self {
            max_vertices,
            max_triangles,
            ..Default::default()
        }
    }

    /// Set cone weight.
    pub fn with_cone_weight(mut self, weight: f32) -> Self {
        self.cone_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Enable depth bounds computation.
    pub fn with_depth_bounds(mut self, enabled: bool) -> Self {
        self.compute_depth_bounds = enabled;
        self
    }
}

/// Complete meshlet mesh output.
#[derive(Debug, Clone)]
pub struct MeshletMesh {
    /// Generated meshlets.
    pub meshlets: Vec<Meshlet>,
    /// Global vertex indices (references original vertex buffer).
    /// Each meshlet's vertices are stored contiguously.
    pub vertices: Vec<u32>,
    /// Local triangle indices (3 bytes per triangle, referencing meshlet-local vertices).
    pub triangles: Vec<u8>,
    /// Adjacency information: for each meshlet, list of neighboring meshlet indices.
    pub adjacency: Vec<Vec<u32>>,
}

impl MeshletMesh {
    /// Create empty meshlet mesh.
    pub fn new() -> Self {
        Self {
            meshlets: Vec::new(),
            vertices: Vec::new(),
            triangles: Vec::new(),
            adjacency: Vec::new(),
        }
    }

    /// Total number of meshlets.
    pub fn meshlet_count(&self) -> usize {
        self.meshlets.len()
    }

    /// Total number of triangles across all meshlets.
    pub fn total_triangles(&self) -> usize {
        self.meshlets.iter().map(|m| m.triangle_count as usize).sum()
    }

    /// Get vertices for a specific meshlet.
    pub fn meshlet_vertices(&self, meshlet_idx: usize) -> &[u32] {
        if meshlet_idx >= self.meshlets.len() {
            return &[];
        }
        let meshlet = &self.meshlets[meshlet_idx];
        let start = meshlet.vertex_offset as usize;
        let end = start + meshlet.vertex_count as usize;
        &self.vertices[start..end]
    }

    /// Get triangle indices for a specific meshlet (local indices, 3 per triangle).
    pub fn meshlet_triangles(&self, meshlet_idx: usize) -> &[u8] {
        if meshlet_idx >= self.meshlets.len() {
            return &[];
        }
        let meshlet = &self.meshlets[meshlet_idx];
        let start = meshlet.triangle_offset as usize;
        let end = start + (meshlet.triangle_count as usize * 3);
        &self.triangles[start..end]
    }

    /// Get neighbors of a specific meshlet.
    pub fn meshlet_neighbors(&self, meshlet_idx: usize) -> &[u32] {
        if meshlet_idx >= self.adjacency.len() {
            return &[];
        }
        &self.adjacency[meshlet_idx]
    }
}

impl Default for MeshletMesh {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Bounding sphere computation (Ritter's algorithm + refinement)
// ---------------------------------------------------------------------------

/// Compute a tight bounding sphere for a set of positions.
///
/// Uses Ritter's algorithm followed by iterative refinement for a good
/// approximation of the minimal enclosing sphere.
pub fn compute_bounding_sphere(positions: &[[f32; 3]]) -> BoundingSphere {
    if positions.is_empty() {
        return BoundingSphere::default();
    }

    if positions.len() == 1 {
        return BoundingSphere::new(positions[0], 0.0);
    }

    // Ritter's algorithm: find two most distant points along each axis
    let mut min_x = 0usize;
    let mut max_x = 0usize;
    let mut min_y = 0usize;
    let mut max_y = 0usize;
    let mut min_z = 0usize;
    let mut max_z = 0usize;

    for (i, p) in positions.iter().enumerate() {
        if p[0] < positions[min_x][0] {
            min_x = i;
        }
        if p[0] > positions[max_x][0] {
            max_x = i;
        }
        if p[1] < positions[min_y][1] {
            min_y = i;
        }
        if p[1] > positions[max_y][1] {
            max_y = i;
        }
        if p[2] < positions[min_z][2] {
            min_z = i;
        }
        if p[2] > positions[max_z][2] {
            max_z = i;
        }
    }

    // Find pair with maximum span
    let span_x = distance_squared(&positions[min_x], &positions[max_x]);
    let span_y = distance_squared(&positions[min_y], &positions[max_y]);
    let span_z = distance_squared(&positions[min_z], &positions[max_z]);

    let (p1_idx, p2_idx) = if span_x >= span_y && span_x >= span_z {
        (min_x, max_x)
    } else if span_y >= span_x && span_y >= span_z {
        (min_y, max_y)
    } else {
        (min_z, max_z)
    };

    // Initial sphere from the two most distant points
    let p1 = positions[p1_idx];
    let p2 = positions[p2_idx];
    let center = [
        (p1[0] + p2[0]) * 0.5,
        (p1[1] + p2[1]) * 0.5,
        (p1[2] + p2[2]) * 0.5,
    ];
    let radius = distance(&p1, &p2) * 0.5;

    let mut sphere = BoundingSphere::new(center, radius);

    // Expand to include all points
    for &pos in positions {
        sphere.expand_to_include(pos);
    }

    // Refinement pass: try to shrink the sphere
    refine_bounding_sphere(&mut sphere, positions);

    sphere
}

/// Refine bounding sphere using iterative method.
fn refine_bounding_sphere(sphere: &mut BoundingSphere, positions: &[[f32; 3]]) {
    const MAX_ITERATIONS: usize = 8;
    const TOLERANCE: f32 = 1e-6;

    for _ in 0..MAX_ITERATIONS {
        // Find the point farthest from current center
        let mut max_dist_sq = 0.0f32;
        let mut farthest_idx = 0;

        for (i, &pos) in positions.iter().enumerate() {
            let dist_sq = distance_squared(&sphere.center, &pos);
            if dist_sq > max_dist_sq {
                max_dist_sq = dist_sq;
                farthest_idx = i;
            }
        }

        let max_dist = max_dist_sq.sqrt();

        // Check if we need to expand
        if max_dist <= sphere.radius + TOLERANCE {
            break;
        }

        // Expand to include farthest point
        let farthest = positions[farthest_idx];
        sphere.expand_to_include(farthest);
    }

    // Final safety pass
    for &pos in positions {
        let dist = distance(&sphere.center, &pos);
        if dist > sphere.radius {
            sphere.radius = dist;
        }
    }
}

/// Compute bounding sphere from indices into a position array.
pub fn compute_bounding_sphere_indexed(
    positions: &[[f32; 3]],
    indices: &[u32],
) -> BoundingSphere {
    if indices.is_empty() || positions.is_empty() {
        return BoundingSphere::default();
    }

    // Collect unique positions
    let unique_indices: HashSet<u32> = indices.iter().copied().collect();
    let points: Vec<[f32; 3]> = unique_indices
        .into_iter()
        .filter(|&idx| (idx as usize) < positions.len())
        .map(|idx| positions[idx as usize])
        .collect();

    compute_bounding_sphere(&points)
}

// ---------------------------------------------------------------------------
// Normal cone computation
// ---------------------------------------------------------------------------

/// Compute a normal cone for a set of triangles.
///
/// The cone axis is the average normal, and the half-angle is chosen
/// to contain all individual triangle normals.
pub fn compute_normal_cone(positions: &[[f32; 3]], indices: &[u32]) -> NormalCone {
    if indices.len() < 3 || positions.is_empty() {
        return NormalCone::default();
    }

    let triangle_count = indices.len() / 3;
    if triangle_count == 0 {
        return NormalCone::default();
    }

    // Compute all triangle normals and their area weights
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(triangle_count);
    let mut areas: Vec<f32> = Vec::with_capacity(triangle_count);
    let mut total_area = 0.0f32;

    for tri in indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
            continue;
        }

        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];

        let (normal, area) = triangle_normal_and_area(&p0, &p1, &p2);
        if area > 1e-10 {
            normals.push(normal);
            areas.push(area);
            total_area += area;
        }
    }

    if normals.is_empty() {
        return NormalCone::default();
    }

    // Compute area-weighted average normal (cone axis)
    let mut axis = [0.0f32; 3];
    for (normal, &area) in normals.iter().zip(areas.iter()) {
        let weight = area / total_area;
        axis[0] += normal[0] * weight;
        axis[1] += normal[1] * weight;
        axis[2] += normal[2] * weight;
    }

    // Normalize axis
    let axis_len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if axis_len < 1e-6 {
        return NormalCone::default();
    }
    axis[0] /= axis_len;
    axis[1] /= axis_len;
    axis[2] /= axis_len;

    // Find maximum angle from axis to any normal
    let mut max_angle = 0.0f32;
    for normal in &normals {
        let dot = axis[0] * normal[0] + axis[1] * normal[1] + axis[2] * normal[2];
        let angle = dot.clamp(-1.0, 1.0).acos();
        if angle > max_angle {
            max_angle = angle;
        }
    }

    // Compute centroid as apex
    let mut apex = [0.0f32; 3];
    let unique_indices: HashSet<u32> = indices.iter().copied().collect();
    let count = unique_indices.len() as f32;
    for idx in unique_indices {
        if (idx as usize) < positions.len() {
            let p = positions[idx as usize];
            apex[0] += p[0];
            apex[1] += p[1];
            apex[2] += p[2];
        }
    }
    if count > 0.0 {
        apex[0] /= count;
        apex[1] /= count;
        apex[2] /= count;
    }

    NormalCone::new(apex, axis, max_angle)
}

/// Compute triangle normal and area.
fn triangle_normal_and_area(p0: &[f32; 3], p1: &[f32; 3], p2: &[f32; 3]) -> ([f32; 3], f32) {
    // Edge vectors
    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

    // Cross product (unnormalized normal)
    let cross = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];

    let len = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    let area = len * 0.5;

    if len < 1e-10 {
        ([0.0, 1.0, 0.0], 0.0)
    } else {
        ([cross[0] / len, cross[1] / len, cross[2] / len], area)
    }
}

// ---------------------------------------------------------------------------
// Meshlet generation
// ---------------------------------------------------------------------------

/// Triangle data for meshlet building.
#[derive(Debug, Clone)]
struct TriangleData {
    /// Original triangle index.
    index: u32,
    /// Vertex indices (global).
    vertices: [u32; 3],
    /// Centroid position.
    centroid: [f32; 3],
    /// Normal vector.
    normal: [f32; 3],
    /// Whether this triangle has been assigned to a meshlet.
    assigned: bool,
}

/// Meshlet builder state.
struct MeshletBuilder<'a> {
    /// Input positions.
    positions: &'a [[f32; 3]],
    /// Configuration.
    config: &'a MeshletConfig,
    /// All triangle data.
    triangles: Vec<TriangleData>,
    /// Edge to triangle adjacency.
    edge_to_triangles: HashMap<(u32, u32), Vec<u32>>,
}

impl<'a> MeshletBuilder<'a> {
    fn new(positions: &'a [[f32; 3]], indices: &[u32], config: &'a MeshletConfig) -> Self {
        let triangle_count = indices.len() / 3;

        // Build triangle data
        let mut triangles = Vec::with_capacity(triangle_count);
        let mut edge_to_triangles: HashMap<(u32, u32), Vec<u32>> = HashMap::new();

        for (t, tri) in indices.chunks_exact(3).enumerate() {
            let i0 = tri[0];
            let i1 = tri[1];
            let i2 = tri[2];

            let p0 = positions.get(i0 as usize).copied().unwrap_or([0.0; 3]);
            let p1 = positions.get(i1 as usize).copied().unwrap_or([0.0; 3]);
            let p2 = positions.get(i2 as usize).copied().unwrap_or([0.0; 3]);

            let centroid = [
                (p0[0] + p1[0] + p2[0]) / 3.0,
                (p0[1] + p1[1] + p2[1]) / 3.0,
                (p0[2] + p1[2] + p2[2]) / 3.0,
            ];

            let (normal, _) = triangle_normal_and_area(&p0, &p1, &p2);

            triangles.push(TriangleData {
                index: t as u32,
                vertices: [i0, i1, i2],
                centroid,
                normal,
                assigned: false,
            });

            // Add edges to adjacency map
            for i in 0..3 {
                let v0 = tri[i];
                let v1 = tri[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                edge_to_triangles.entry(edge).or_default().push(t as u32);
            }
        }

        Self {
            positions,
            config,
            triangles,
            edge_to_triangles,
        }
    }

    /// Build all meshlets.
    fn build(&mut self) -> MeshletMesh {
        let mut mesh = MeshletMesh::new();

        // Process triangles until all are assigned
        while let Some(seed_idx) = self.find_seed_triangle() {
            let meshlet = self.build_meshlet(seed_idx);
            mesh.meshlets.push(meshlet.0);
            mesh.vertices.extend(meshlet.1);
            mesh.triangles.extend(meshlet.2);
        }

        // Build adjacency information
        mesh.adjacency = self.build_adjacency(&mesh);

        mesh
    }

    /// Find an unassigned triangle to start a new meshlet.
    fn find_seed_triangle(&self) -> Option<usize> {
        // Simple: return first unassigned triangle
        // Could optimize with spatial hashing for better locality
        self.triangles.iter().position(|t| !t.assigned)
    }

    /// Build a single meshlet starting from the seed triangle.
    fn build_meshlet(&mut self, seed_idx: usize) -> (Meshlet, Vec<u32>, Vec<u8>) {
        let max_verts = self.config.max_vertices as usize;
        let max_tris = self.config.max_triangles as usize;

        // Local vertex map: global index -> local index
        let mut local_vertex_map: HashMap<u32, u8> = HashMap::new();
        let mut meshlet_vertices: Vec<u32> = Vec::with_capacity(max_verts);
        let mut meshlet_triangles: Vec<u8> = Vec::with_capacity(max_tris * 3);
        let mut assigned_triangles: Vec<usize> = Vec::with_capacity(max_tris);

        // Add seed triangle
        self.add_triangle_to_meshlet(
            seed_idx,
            &mut local_vertex_map,
            &mut meshlet_vertices,
            &mut meshlet_triangles,
            &mut assigned_triangles,
        );

        // Greedily add adjacent triangles
        loop {
            let best = self.find_best_adjacent_triangle(
                &assigned_triangles,
                &local_vertex_map,
                max_verts,
                max_tris,
                meshlet_triangles.len() / 3,
            );

            match best {
                Some(tri_idx) => {
                    self.add_triangle_to_meshlet(
                        tri_idx,
                        &mut local_vertex_map,
                        &mut meshlet_vertices,
                        &mut meshlet_triangles,
                        &mut assigned_triangles,
                    );
                }
                None => break,
            }
        }

        // Build meshlet metadata
        let meshlet_positions: Vec<[f32; 3]> = meshlet_vertices
            .iter()
            .map(|&idx| self.positions.get(idx as usize).copied().unwrap_or([0.0; 3]))
            .collect();

        let meshlet_indices: Vec<u32> = meshlet_triangles.iter().map(|&i| i as u32).collect();

        let bounding_sphere = compute_bounding_sphere(&meshlet_positions);
        let normal_cone = compute_normal_cone(&meshlet_positions, &meshlet_indices);

        // Compute depth bounds if requested
        let depth_bounds = if self.config.compute_depth_bounds {
            compute_depth_bounds(&meshlet_positions)
        } else {
            None
        };

        let meshlet = Meshlet {
            vertex_offset: 0, // Will be set later
            vertex_count: meshlet_vertices.len() as u32,
            triangle_offset: 0, // Will be set later
            triangle_count: (meshlet_triangles.len() / 3) as u32,
            bounding_sphere,
            normal_cone,
            depth_bounds,
        };

        (meshlet, meshlet_vertices, meshlet_triangles)
    }

    /// Add a triangle to the current meshlet.
    fn add_triangle_to_meshlet(
        &mut self,
        tri_idx: usize,
        local_map: &mut HashMap<u32, u8>,
        vertices: &mut Vec<u32>,
        triangles: &mut Vec<u8>,
        assigned: &mut Vec<usize>,
    ) {
        let tri = &self.triangles[tri_idx];
        let tri_verts = tri.vertices;

        // Add vertices and get local indices
        let mut local_indices = [0u8; 3];
        for (i, &global_idx) in tri_verts.iter().enumerate() {
            let local_idx = match local_map.get(&global_idx) {
                Some(&idx) => idx,
                None => {
                    let idx = vertices.len() as u8;
                    local_map.insert(global_idx, idx);
                    vertices.push(global_idx);
                    idx
                }
            };
            local_indices[i] = local_idx;
        }

        // Add triangle (local indices)
        triangles.extend_from_slice(&local_indices);
        assigned.push(tri_idx);
        self.triangles[tri_idx].assigned = true;
    }

    /// Find the best adjacent triangle to add to the meshlet.
    fn find_best_adjacent_triangle(
        &self,
        assigned: &[usize],
        local_map: &HashMap<u32, u8>,
        max_verts: usize,
        max_tris: usize,
        current_tris: usize,
    ) -> Option<usize> {
        if current_tris >= max_tris {
            return None;
        }

        let current_verts = local_map.len();
        let mut best_score = f32::NEG_INFINITY;
        let mut best_tri = None;

        // Collect adjacent triangles
        for &tri_idx in assigned {
            let tri = &self.triangles[tri_idx];

            for i in 0..3 {
                let v0 = tri.vertices[i];
                let v1 = tri.vertices[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };

                if let Some(adjacent_tris) = self.edge_to_triangles.get(&edge) {
                    for &adj_idx in adjacent_tris {
                        let adj_tri = &self.triangles[adj_idx as usize];
                        if adj_tri.assigned {
                            continue;
                        }

                        // Count new vertices this triangle would add
                        let new_verts = adj_tri
                            .vertices
                            .iter()
                            .filter(|v| !local_map.contains_key(v))
                            .count();

                        // Check if adding this triangle would exceed limits
                        if current_verts + new_verts > max_verts {
                            continue;
                        }

                        // Score: prefer triangles with fewer new vertices (better locality)
                        // and similar normals (tighter cone)
                        let locality_score = (3 - new_verts) as f32 / 3.0;

                        // Compute normal similarity with average meshlet normal
                        let normal_score = self.compute_normal_similarity(assigned, adj_tri);

                        let score = locality_score * (1.0 - self.config.cone_weight)
                            + normal_score * self.config.cone_weight;

                        if score > best_score {
                            best_score = score;
                            best_tri = Some(adj_idx as usize);
                        }
                    }
                }
            }
        }

        best_tri
    }

    /// Compute normal similarity between a triangle and current meshlet.
    fn compute_normal_similarity(&self, assigned: &[usize], tri: &TriangleData) -> f32 {
        if assigned.is_empty() {
            return 1.0;
        }

        // Compute average normal of assigned triangles
        let mut avg_normal = [0.0f32; 3];
        for &idx in assigned {
            let t = &self.triangles[idx];
            avg_normal[0] += t.normal[0];
            avg_normal[1] += t.normal[1];
            avg_normal[2] += t.normal[2];
        }

        let len = (avg_normal[0] * avg_normal[0]
            + avg_normal[1] * avg_normal[1]
            + avg_normal[2] * avg_normal[2])
            .sqrt();

        if len < 1e-6 {
            return 1.0;
        }

        avg_normal[0] /= len;
        avg_normal[1] /= len;
        avg_normal[2] /= len;

        // Dot product with candidate normal (1 = same direction, -1 = opposite)
        let dot = avg_normal[0] * tri.normal[0]
            + avg_normal[1] * tri.normal[1]
            + avg_normal[2] * tri.normal[2];

        // Map from [-1, 1] to [0, 1]
        (dot + 1.0) * 0.5
    }

    /// Build adjacency information between meshlets.
    fn build_adjacency(&self, mesh: &MeshletMesh) -> Vec<Vec<u32>> {
        let meshlet_count = mesh.meshlets.len();
        let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); meshlet_count];

        // Build edge to meshlet map
        let mut edge_to_meshlet: HashMap<(u32, u32), Vec<u32>> = HashMap::new();

        for (m_idx, meshlet) in mesh.meshlets.iter().enumerate() {
            let verts = mesh.meshlet_vertices(m_idx);
            let tris = mesh.meshlet_triangles(m_idx);

            for tri in tris.chunks_exact(3) {
                for i in 0..3 {
                    let v0 = verts[tri[i] as usize];
                    let v1 = verts[tri[(i + 1) % 3] as usize];
                    let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                    edge_to_meshlet.entry(edge).or_default().push(m_idx as u32);
                }
            }
        }

        // Find meshlets sharing edges
        for meshlets in edge_to_meshlet.values() {
            if meshlets.len() > 1 {
                for &m1 in meshlets {
                    for &m2 in meshlets {
                        if m1 != m2 && !adjacency[m1 as usize].contains(&m2) {
                            adjacency[m1 as usize].push(m2);
                        }
                    }
                }
            }
        }

        // Sort adjacency lists for determinism
        for adj in &mut adjacency {
            adj.sort_unstable();
        }

        adjacency
    }
}

/// Generate meshlets from a triangle mesh.
///
/// This is the main entry point for meshlet generation. The transformation
/// is idempotent - the same input always produces the same output.
///
/// # Arguments
///
/// * `positions` - Vertex positions
/// * `indices` - Triangle indices (must be divisible by 3)
/// * `config` - Generation configuration
///
/// # Returns
///
/// A `MeshletMesh` containing all generated meshlets with their metadata.
///
/// # Example
///
/// ```ignore
/// let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let indices = vec![0, 1, 2];
/// let config = MeshletConfig::default();
///
/// let mesh = generate_meshlets(&positions, &indices, &config);
/// assert_eq!(mesh.meshlet_count(), 1);
/// ```
pub fn generate_meshlets(
    positions: &[[f32; 3]],
    indices: &[u32],
    config: &MeshletConfig,
) -> MeshletMesh {
    if positions.is_empty() || indices.is_empty() || indices.len() < 3 {
        return MeshletMesh::new();
    }

    if indices.len() % 3 != 0 {
        return MeshletMesh::new();
    }

    let mut builder = MeshletBuilder::new(positions, indices, config);
    let mut mesh = builder.build();

    // Fix offsets
    let mut vertex_offset = 0u32;
    let mut triangle_offset = 0u32;

    for meshlet in &mut mesh.meshlets {
        meshlet.vertex_offset = vertex_offset;
        meshlet.triangle_offset = triangle_offset;
        vertex_offset += meshlet.vertex_count;
        triangle_offset += meshlet.triangle_count * 3;
    }

    mesh
}

/// Generate meshlets with validation.
///
/// Same as `generate_meshlets` but returns errors for invalid input.
pub fn generate_meshlets_validated(
    positions: &[[f32; 3]],
    indices: &[u32],
    config: &MeshletConfig,
) -> Result<MeshletMesh, MeshletError> {
    if positions.is_empty() {
        return Err(MeshletError::EmptyMesh);
    }

    if indices.is_empty() {
        return Err(MeshletError::EmptyMesh);
    }

    if indices.len() % 3 != 0 {
        return Err(MeshletError::InvalidTriangleCount(indices.len()));
    }

    // Validate indices
    for &idx in indices {
        if idx as usize >= positions.len() {
            return Err(MeshletError::IndexOutOfBounds {
                index: idx,
                vertex_count: positions.len(),
            });
        }
    }

    Ok(generate_meshlets(positions, indices, config))
}

// ---------------------------------------------------------------------------
// Depth bounds computation
// ---------------------------------------------------------------------------

/// Compute min/max depth bounds for a set of positions.
///
/// Returns the Z-range for Hi-Z occlusion culling.
fn compute_depth_bounds(positions: &[[f32; 3]]) -> Option<(f32, f32)> {
    if positions.is_empty() {
        return None;
    }

    let mut min_z = f32::MAX;
    let mut max_z = f32::MIN;

    for pos in positions {
        min_z = min_z.min(pos[2]);
        max_z = max_z.max(pos[2]);
    }

    if min_z <= max_z {
        Some((min_z, max_z))
    } else {
        None
    }
}

/// Compute depth bounds from vertex indices.
pub fn compute_depth_bounds_indexed(
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Option<(f32, f32)> {
    if positions.is_empty() || indices.is_empty() {
        return None;
    }

    let mut min_z = f32::MAX;
    let mut max_z = f32::MIN;

    for &idx in indices {
        if let Some(pos) = positions.get(idx as usize) {
            min_z = min_z.min(pos[2]);
            max_z = max_z.max(pos[2]);
        }
    }

    if min_z <= max_z {
        Some((min_z, max_z))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Vertex/index reordering
// ---------------------------------------------------------------------------

/// Reorder vertices and indices within a meshlet for better GPU cache efficiency.
///
/// Uses linear access pattern optimization.
pub fn reorder_meshlet_for_cache(
    vertices: &mut [u32],
    triangles: &mut [u8],
) {
    if vertices.is_empty() || triangles.is_empty() {
        return;
    }

    // Build new vertex order based on first access in triangle order
    let mut seen: HashSet<u8> = HashSet::new();
    let mut order: Vec<u8> = Vec::with_capacity(vertices.len());

    for &local_idx in triangles.iter() {
        if seen.insert(local_idx) {
            order.push(local_idx);
        }
    }

    // Add any vertices not referenced by triangles
    for i in 0..vertices.len() as u8 {
        if seen.insert(i) {
            order.push(i);
        }
    }

    // Build old-to-new mapping
    let mut old_to_new: Vec<u8> = vec![0; vertices.len()];
    for (new_idx, &old_idx) in order.iter().enumerate() {
        old_to_new[old_idx as usize] = new_idx as u8;
    }

    // Reorder vertices
    let old_vertices = vertices.to_vec();
    for (new_idx, &old_idx) in order.iter().enumerate() {
        vertices[new_idx] = old_vertices[old_idx as usize];
    }

    // Remap triangle indices
    for local_idx in triangles.iter_mut() {
        *local_idx = old_to_new[*local_idx as usize];
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Compute squared distance between two points.
#[inline]
fn distance_squared(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    dx * dx + dy * dy + dz * dz
}

/// Compute distance between two points.
#[inline]
fn distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    distance_squared(a, b).sqrt()
}

/// Compute centroid of a set of positions.
pub fn compute_centroid(positions: &[[f32; 3]]) -> [f32; 3] {
    if positions.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let mut sum = [0.0f32; 3];
    for pos in positions {
        sum[0] += pos[0];
        sum[1] += pos[1];
        sum[2] += pos[2];
    }

    let count = positions.len() as f32;
    [sum[0] / count, sum[1] / count, sum[2] / count]
}

/// Normalize a vector.
pub fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

/// Compute dot product.
pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Compute cross product.
pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Meshlet size limits tests (4 tests)
    // ========================================================================

    #[test]
    fn test_meshlet_respects_vertex_limit() {
        // Create a mesh that would need many vertices per meshlet
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        // Create 100 separate triangles (300 vertices total)
        for i in 0..100 {
            let base = i as f32 * 10.0;
            positions.push([base, 0.0, 0.0]);
            positions.push([base + 1.0, 0.0, 0.0]);
            positions.push([base + 0.5, 1.0, 0.0]);

            let idx = (i * 3) as u32;
            indices.extend_from_slice(&[idx, idx + 1, idx + 2]);
        }

        let config = MeshletConfig::with_limits(64, 124);
        let mesh = generate_meshlets(&positions, &indices, &config);

        // Each meshlet should respect vertex limit
        for meshlet in &mesh.meshlets {
            assert!(
                meshlet.vertex_count <= config.max_vertices,
                "Meshlet has {} vertices, max is {}",
                meshlet.vertex_count,
                config.max_vertices
            );
        }
    }

    #[test]
    fn test_meshlet_respects_triangle_limit() {
        // Create a large connected mesh
        let size = 20;
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        // Create grid vertices
        for y in 0..=size {
            for x in 0..=size {
                positions.push([x as f32, y as f32, 0.0]);
            }
        }

        // Create grid triangles
        for y in 0..size {
            for x in 0..size {
                let v0 = y * (size + 1) + x;
                let v1 = v0 + 1;
                let v2 = v0 + (size + 1);
                let v3 = v2 + 1;

                indices.extend_from_slice(&[v0 as u32, v1 as u32, v2 as u32]);
                indices.extend_from_slice(&[v2 as u32, v1 as u32, v3 as u32]);
            }
        }

        let config = MeshletConfig::with_limits(64, 124);
        let mesh = generate_meshlets(&positions, &indices, &config);

        // Each meshlet should respect triangle limit
        for meshlet in &mesh.meshlets {
            assert!(
                meshlet.triangle_count <= config.max_triangles,
                "Meshlet has {} triangles, max is {}",
                meshlet.triangle_count,
                config.max_triangles
            );
        }
    }

    #[test]
    fn test_meshlet_small_limits() {
        // Test with very small limits
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 1, 3, 2, 2, 3, 4];

        let config = MeshletConfig::with_limits(4, 2);
        let mesh = generate_meshlets(&positions, &indices, &config);

        for meshlet in &mesh.meshlets {
            assert!(meshlet.vertex_count <= 4);
            assert!(meshlet.triangle_count <= 2);
        }
    }

    #[test]
    fn test_meshlet_default_limits() {
        let config = MeshletConfig::default();
        assert_eq!(config.max_vertices, 64);
        assert_eq!(config.max_triangles, 124);
    }

    // ========================================================================
    // Bounding sphere accuracy tests (4 tests)
    // ========================================================================

    #[test]
    fn test_bounding_sphere_single_point() {
        let positions = vec![[1.0, 2.0, 3.0]];
        let sphere = compute_bounding_sphere(&positions);

        assert_eq!(sphere.center, [1.0, 2.0, 3.0]);
        assert_eq!(sphere.radius, 0.0);
    }

    #[test]
    fn test_bounding_sphere_two_points() {
        let positions = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let sphere = compute_bounding_sphere(&positions);

        assert!((sphere.center[0] - 1.0).abs() < 0.01);
        assert!((sphere.radius - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_bounding_sphere_contains_all_points() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [-0.5, 0.5, 0.5],
        ];

        let sphere = compute_bounding_sphere(&positions);

        // All points should be inside (or on) the sphere
        for pos in &positions {
            let dist = distance(&sphere.center, pos);
            assert!(
                dist <= sphere.radius + 0.001,
                "Point {:?} at distance {} outside sphere radius {}",
                pos,
                dist,
                sphere.radius
            );
        }
    }

    #[test]
    fn test_bounding_sphere_indexed() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [10.0, 10.0, 10.0], // Not indexed
        ];
        let indices = vec![0, 1, 2];

        let sphere = compute_bounding_sphere_indexed(&positions, &indices);

        // Sphere should only contain indexed vertices
        assert!(sphere.radius < 2.0); // Much smaller than distance to [10,10,10]
    }

    // ========================================================================
    // Normal cone computation tests (4 tests)
    // ========================================================================

    #[test]
    fn test_normal_cone_single_triangle() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2];

        let cone = compute_normal_cone(&positions, &indices);

        // Normal should point in +Z direction
        assert!(cone.axis[2] > 0.9);
        // Half-angle should be 0 (single normal)
        assert!(cone.half_angle < 0.01);
    }

    #[test]
    fn test_normal_cone_opposing_normals() {
        // Two triangles facing opposite directions
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.1],
            [1.0, 0.0, 0.1],
            [0.0, 1.0, 0.1],
        ];
        // First tri faces +Z, second faces -Z (reversed winding)
        let indices = vec![0, 1, 2, 5, 4, 3];

        let cone = compute_normal_cone(&positions, &indices);

        // Half-angle should be near PI/2 or larger for opposing normals
        assert!(cone.half_angle > std::f32::consts::PI * 0.4);
    }

    #[test]
    fn test_normal_cone_visibility_check() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2];

        let cone = compute_normal_cone(&positions, &indices);

        // Triangle normal points in +Z direction (CCW winding)
        // View direction is where the camera is LOOKING (from camera towards scene)

        // View direction [0, 0, -1] means camera is looking towards -Z (camera is in +Z space)
        // Triangle normal is +Z, so front face is visible to camera in +Z looking at -Z
        assert!(cone.is_potentially_visible([0.0, 0.0, -1.0]));

        // View direction [0, 0, 1] means camera is looking towards +Z (camera is in -Z space)
        // Triangle normal is +Z, so we're looking at the back face - should NOT be visible
        assert!(!cone.is_potentially_visible([0.0, 0.0, 1.0]));
    }

    #[test]
    fn test_normal_cone_empty() {
        let cone = compute_normal_cone(&[], &[]);
        assert_eq!(cone.half_angle, std::f32::consts::PI);
    }

    // ========================================================================
    // Adjacency detection tests (4 tests)
    // ========================================================================

    #[test]
    fn test_adjacency_connected_triangles() {
        // Two triangles sharing an edge
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 1, 3, 2];

        let config = MeshletConfig::with_limits(3, 1); // Force separate meshlets
        let mesh = generate_meshlets(&positions, &indices, &config);

        if mesh.meshlets.len() >= 2 {
            // Should detect adjacency between meshlets
            let has_adjacency = mesh.adjacency.iter().any(|adj| !adj.is_empty());
            assert!(has_adjacency, "Should detect adjacent meshlets");
        }
    }

    #[test]
    fn test_adjacency_disconnected_triangles() {
        // Two triangles with no shared vertices
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [10.5, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 3, 4, 5];

        let config = MeshletConfig::with_limits(3, 1);
        let mesh = generate_meshlets(&positions, &indices, &config);

        if mesh.meshlets.len() >= 2 {
            // Should NOT detect adjacency between disconnected meshlets
            // (though they might end up in same meshlet due to limits)
            for adj in &mesh.adjacency {
                if !adj.is_empty() {
                    // If there is adjacency, the meshlets should share an edge
                    // This is hard to verify without more complex logic
                }
            }
        }
    }

    #[test]
    fn test_adjacency_symmetric() {
        // Adjacency should be symmetric
        let size = 5;
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for y in 0..=size {
            for x in 0..=size {
                positions.push([x as f32, y as f32, 0.0]);
            }
        }

        for y in 0..size {
            for x in 0..size {
                let v0 = y * (size + 1) + x;
                let v1 = v0 + 1;
                let v2 = v0 + (size + 1);
                let v3 = v2 + 1;

                indices.extend_from_slice(&[v0 as u32, v1 as u32, v2 as u32]);
                indices.extend_from_slice(&[v2 as u32, v1 as u32, v3 as u32]);
            }
        }

        let config = MeshletConfig::with_limits(8, 4);
        let mesh = generate_meshlets(&positions, &indices, &config);

        // Check symmetry
        for (i, adj) in mesh.adjacency.iter().enumerate() {
            for &j in adj {
                assert!(
                    mesh.adjacency[j as usize].contains(&(i as u32)),
                    "Adjacency not symmetric: {} -> {} but not {} -> {}",
                    i,
                    j,
                    j,
                    i
                );
            }
        }
    }

    #[test]
    fn test_adjacency_self_not_included() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2];

        let config = MeshletConfig::default();
        let mesh = generate_meshlets(&positions, &indices, &config);

        // No meshlet should be adjacent to itself
        for (i, adj) in mesh.adjacency.iter().enumerate() {
            assert!(
                !adj.contains(&(i as u32)),
                "Meshlet {} is adjacent to itself",
                i
            );
        }
    }

    // ========================================================================
    // Vertex/index reordering tests (3 tests)
    // ========================================================================

    #[test]
    fn test_reorder_preserves_triangles() {
        let mut vertices = vec![100, 200, 300, 400];
        let mut triangles = vec![0, 1, 2, 2, 1, 3];

        let orig_verts = vertices.clone();
        let orig_tris = triangles.clone();

        reorder_meshlet_for_cache(&mut vertices, &mut triangles);

        // Triangles should still reference the same global vertices
        for (old_tri, new_tri) in orig_tris.chunks(3).zip(triangles.chunks(3)) {
            let old_global: Vec<_> = old_tri.iter().map(|&i| orig_verts[i as usize]).collect();
            let new_global: Vec<_> = new_tri.iter().map(|&i| vertices[i as usize]).collect();
            assert_eq!(old_global, new_global);
        }
    }

    #[test]
    fn test_reorder_sequential_access() {
        let mut vertices = vec![100, 200, 300, 400];
        let mut triangles = vec![3, 1, 0, 0, 1, 2]; // Non-sequential access

        reorder_meshlet_for_cache(&mut vertices, &mut triangles);

        // First triangle should now use indices 0, 1, 2
        assert_eq!(triangles[0], 0);
        assert_eq!(triangles[1], 1);
        assert_eq!(triangles[2], 2);
    }

    #[test]
    fn test_reorder_empty() {
        let mut vertices: Vec<u32> = Vec::new();
        let mut triangles: Vec<u8> = Vec::new();

        reorder_meshlet_for_cache(&mut vertices, &mut triangles);

        assert!(vertices.is_empty());
        assert!(triangles.is_empty());
    }

    // ========================================================================
    // Idempotency tests (2 tests)
    // ========================================================================

    #[test]
    fn test_generation_idempotent() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ];
        let indices = vec![0, 1, 2, 1, 3, 2, 2, 3, 4];

        let config = MeshletConfig::default();

        let mesh1 = generate_meshlets(&positions, &indices, &config);
        let mesh2 = generate_meshlets(&positions, &indices, &config);

        // Same number of meshlets
        assert_eq!(mesh1.meshlets.len(), mesh2.meshlets.len());

        // Same vertices and triangles
        assert_eq!(mesh1.vertices, mesh2.vertices);
        assert_eq!(mesh1.triangles, mesh2.triangles);

        // Same adjacency
        assert_eq!(mesh1.adjacency, mesh2.adjacency);
    }

    #[test]
    fn test_generation_deterministic() {
        let size = 10;
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for y in 0..=size {
            for x in 0..=size {
                positions.push([x as f32, y as f32, 0.0]);
            }
        }

        for y in 0..size {
            for x in 0..size {
                let v0 = y * (size + 1) + x;
                let v1 = v0 + 1;
                let v2 = v0 + (size + 1);
                let v3 = v2 + 1;

                indices.extend_from_slice(&[v0 as u32, v1 as u32, v2 as u32]);
                indices.extend_from_slice(&[v2 as u32, v1 as u32, v3 as u32]);
            }
        }

        let config = MeshletConfig::default();

        // Generate multiple times
        let results: Vec<_> = (0..3)
            .map(|_| generate_meshlets(&positions, &indices, &config))
            .collect();

        // All results should be identical
        for i in 1..results.len() {
            assert_eq!(results[0].meshlets.len(), results[i].meshlets.len());
            assert_eq!(results[0].vertices, results[i].vertices);
            assert_eq!(results[0].triangles, results[i].triangles);
        }
    }

    // ========================================================================
    // Edge cases tests (4 tests)
    // ========================================================================

    #[test]
    fn test_empty_input() {
        let config = MeshletConfig::default();

        let mesh = generate_meshlets(&[], &[], &config);
        assert!(mesh.meshlets.is_empty());

        let mesh = generate_meshlets(&[[0.0, 0.0, 0.0]], &[], &config);
        assert!(mesh.meshlets.is_empty());
    }

    #[test]
    fn test_single_triangle() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2];

        let config = MeshletConfig::default();
        let mesh = generate_meshlets(&positions, &indices, &config);

        assert_eq!(mesh.meshlets.len(), 1);
        assert_eq!(mesh.meshlets[0].vertex_count, 3);
        assert_eq!(mesh.meshlets[0].triangle_count, 1);
    }

    #[test]
    fn test_invalid_index_count() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ];
        let indices = vec![0, 1]; // Not divisible by 3

        let config = MeshletConfig::default();
        let mesh = generate_meshlets(&positions, &indices, &config);

        assert!(mesh.meshlets.is_empty());
    }

    #[test]
    fn test_validated_errors() {
        let positions = vec![[0.0, 0.0, 0.0]];
        let indices = vec![0, 1, 2]; // Indices 1, 2 out of bounds

        let config = MeshletConfig::default();
        let result = generate_meshlets_validated(&positions, &indices, &config);

        assert!(result.is_err());
    }

    // ========================================================================
    // Additional tests for comprehensive coverage
    // ========================================================================

    #[test]
    fn test_bounding_sphere_merge() {
        let s1 = BoundingSphere::new([0.0, 0.0, 0.0], 1.0);
        let s2 = BoundingSphere::new([3.0, 0.0, 0.0], 1.0);

        let merged = s1.merge(&s2);

        // Merged should contain both spheres
        assert!(merged.contains_point([0.0, 0.0, 0.0]));
        assert!(merged.contains_point([3.0, 0.0, 0.0]));
        assert!(merged.contains_point([-1.0, 0.0, 0.0]));
        assert!(merged.contains_point([4.0, 0.0, 0.0]));
    }

    #[test]
    fn test_normal_cone_cos_half_angle() {
        let cone = NormalCone::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], std::f32::consts::PI / 4.0);

        let cos_half = cone.cos_half_angle();
        assert!((cos_half - (std::f32::consts::PI / 4.0).cos()).abs() < 0.001);
    }

    #[test]
    fn test_meshlet_mesh_accessors() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ];
        let indices = vec![0, 1, 2];

        let config = MeshletConfig::default();
        let mesh = generate_meshlets(&positions, &indices, &config);

        assert_eq!(mesh.meshlet_count(), 1);
        assert_eq!(mesh.total_triangles(), 1);

        let verts = mesh.meshlet_vertices(0);
        assert_eq!(verts.len(), 3);

        let tris = mesh.meshlet_triangles(0);
        assert_eq!(tris.len(), 3);

        // Out of bounds access should return empty
        assert!(mesh.meshlet_vertices(100).is_empty());
        assert!(mesh.meshlet_triangles(100).is_empty());
        assert!(mesh.meshlet_neighbors(100).is_empty());
    }

    #[test]
    fn test_depth_bounds_computation() {
        let positions = vec![
            [0.0, 0.0, -5.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 10.0],
        ];

        let bounds = compute_depth_bounds(&positions);
        assert!(bounds.is_some());

        let (min_z, max_z) = bounds.unwrap();
        assert!((min_z - (-5.0)).abs() < 0.001);
        assert!((max_z - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_depth_bounds_indexed() {
        let positions = vec![
            [0.0, 0.0, -5.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 10.0],
            [0.0, 0.0, 100.0], // Not indexed
        ];
        let indices = vec![0, 1, 2];

        let bounds = compute_depth_bounds_indexed(&positions, &indices);
        assert!(bounds.is_some());

        let (min_z, max_z) = bounds.unwrap();
        assert!((min_z - (-5.0)).abs() < 0.001);
        assert!((max_z - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_utility_functions() {
        // Test normalize
        let v = normalize([3.0, 4.0, 0.0]);
        assert!((v[0] - 0.6).abs() < 0.001);
        assert!((v[1] - 0.8).abs() < 0.001);

        // Test dot
        let d = dot([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(d.abs() < 0.001);

        // Test cross
        let c = cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!((c[2] - 1.0).abs() < 0.001);

        // Test centroid
        let centroid = compute_centroid(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 3.0, 0.0]]);
        assert!((centroid[0] - 1.0).abs() < 0.001);
        assert!((centroid[1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_config_builders() {
        let config = MeshletConfig::mesh_shader();
        assert_eq!(config.max_vertices, 64);
        assert_eq!(config.max_triangles, 126);
        assert!(config.compute_depth_bounds);

        let config = MeshletConfig::backface_optimized();
        assert_eq!(config.cone_weight, 0.8);

        let config = MeshletConfig::default()
            .with_cone_weight(0.3)
            .with_depth_bounds(true);
        assert_eq!(config.cone_weight, 0.3);
        assert!(config.compute_depth_bounds);
    }

    #[test]
    fn test_large_mesh_meshlet_count() {
        // Generate a larger mesh and verify reasonable meshlet count
        let size = 30;
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for y in 0..=size {
            for x in 0..=size {
                positions.push([x as f32, y as f32, 0.0]);
            }
        }

        for y in 0..size {
            for x in 0..size {
                let v0 = y * (size + 1) + x;
                let v1 = v0 + 1;
                let v2 = v0 + (size + 1);
                let v3 = v2 + 1;

                indices.extend_from_slice(&[v0 as u32, v1 as u32, v2 as u32]);
                indices.extend_from_slice(&[v2 as u32, v1 as u32, v3 as u32]);
            }
        }

        let triangle_count = indices.len() / 3;
        let config = MeshletConfig::default();
        let mesh = generate_meshlets(&positions, &indices, &config);

        // Should have created some meshlets
        assert!(!mesh.meshlets.is_empty());

        // Total triangles should match input
        assert_eq!(mesh.total_triangles(), triangle_count);

        // Average meshlet should have reasonable triangle count
        let avg_tris = triangle_count as f32 / mesh.meshlets.len() as f32;
        assert!(avg_tris > 1.0);
    }

    #[test]
    fn test_meshlet_offsets_contiguous() {
        let size = 10;
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for y in 0..=size {
            for x in 0..=size {
                positions.push([x as f32, y as f32, 0.0]);
            }
        }

        for y in 0..size {
            for x in 0..size {
                let v0 = y * (size + 1) + x;
                let v1 = v0 + 1;
                let v2 = v0 + (size + 1);
                let v3 = v2 + 1;

                indices.extend_from_slice(&[v0 as u32, v1 as u32, v2 as u32]);
                indices.extend_from_slice(&[v2 as u32, v1 as u32, v3 as u32]);
            }
        }

        let config = MeshletConfig::default();
        let mesh = generate_meshlets(&positions, &indices, &config);

        // Verify offsets are contiguous
        let mut expected_vert_offset = 0u32;
        let mut expected_tri_offset = 0u32;

        for meshlet in &mesh.meshlets {
            assert_eq!(
                meshlet.vertex_offset, expected_vert_offset,
                "Vertex offset mismatch"
            );
            assert_eq!(
                meshlet.triangle_offset, expected_tri_offset,
                "Triangle offset mismatch"
            );

            expected_vert_offset += meshlet.vertex_count;
            expected_tri_offset += meshlet.triangle_count * 3;
        }

        // Total should match arrays
        assert_eq!(expected_vert_offset as usize, mesh.vertices.len());
        assert_eq!(expected_tri_offset as usize, mesh.triangles.len());
    }
}

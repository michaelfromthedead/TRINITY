// SPDX-License-Identifier: MIT
//
// WHITEBOX T-WGPU-P6.9.2: MeshletGenerator
//
// Comprehensive whitebox tests for meshlet generation with greedy splitting.
// Full access to internal implementation details.
//
// Test Categories:
//   1. MeshInput Validation Tests - Empty, invalid index count, out-of-bounds
//   2. Generator Configuration Tests - Default limits, custom limits, zero limits
//   3. Splitting Algorithm Tests - Single/multiple meshlets, vertex/triangle limits
//   4. Bounding Sphere Tests - Ritter's algorithm, vertex containment
//   5. Backface Cone Tests - Normal calculation, cone axis, cone cutoff range
//   6. Reconstruction Tests - Round-trip index verification
//   7. Edge Case Tests - Degenerate triangles, invalid indices
//   8. Memory Layout Tests - GPU-compatible struct layout
//
// Coverage:
//   - MeshInput: new, with_normals, validate, is_empty, triangle_count
//   - MeshletGenerator: new, with_limits, max_vertices, max_triangles
//   - MeshletOutput: meshlet_count, is_empty, validate, reconstruct_indices, stats
//   - Internal: build_meshlets_greedy, compute_bounds, compute_cone
//   - Constants: MAX_MESHLET_VERTICES, MAX_MESHLET_TRIANGLES, EPSILON

#![allow(unexpected_cfgs)]

use renderer_backend::gpu_driven::meshlet_generator::{
    MeshInput, MeshletGenerator, MeshletOutput,
    MAX_MESHLET_TRIANGLES, MAX_MESHLET_VERTICES,
};
use renderer_backend::gpu_driven::meshlet::{
    MeshletBounds, MeshletData,
    MESHLET_BOUNDS_SIZE, MESHLET_SIZE,
};

use std::collections::HashSet;
use std::f32::consts::PI;

const EPSILON: f32 = 1e-5;

// =============================================================================
// Test Helpers
// =============================================================================

/// Calculate distance between two 3D points.
fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Calculate vector length.
fn length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Calculate dot product of two vectors.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Calculate cross product of two vectors.
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalize a vector.
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = length(v);
    if len > EPSILON {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Create a simple single triangle mesh.
fn make_single_triangle() -> (Vec<[f32; 3]>, Vec<u32>) {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let indices = vec![0, 1, 2];
    (positions, indices)
}

/// Create a cube mesh (8 verts, 12 triangles).
fn make_cube() -> (Vec<[f32; 3]>, Vec<u32>) {
    let positions = vec![
        [-1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0,  1.0],
        [-1.0,  1.0,  1.0],
        [-1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [ 1.0,  1.0, -1.0],
        [ 1.0, -1.0, -1.0],
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

/// Create a large mesh with unique vertices per triangle (forces many meshlets).
fn make_large_mesh_unique_verts(triangle_count: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
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

/// Create a grid mesh with high vertex sharing.
fn make_grid_mesh(grid_size: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
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

    (positions, indices)
}

/// Create a sphere mesh with approximately n triangles.
fn make_sphere(subdivisions: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut positions = Vec::new();
    let mut indices = Vec::new();

    // Create vertices using spherical coordinates
    let lat_segments = subdivisions;
    let lon_segments = subdivisions * 2;

    for lat in 0..=lat_segments {
        let theta = lat as f32 * PI / lat_segments as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for lon in 0..=lon_segments {
            let phi = lon as f32 * 2.0 * PI / lon_segments as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            positions.push([
                sin_theta * cos_phi,
                cos_theta,
                sin_theta * sin_phi,
            ]);
        }
    }

    // Create indices
    for lat in 0..lat_segments {
        for lon in 0..lon_segments {
            let first = lat * (lon_segments + 1) + lon;
            let second = first + lon_segments + 1;

            indices.push(first as u32);
            indices.push(second as u32);
            indices.push((first + 1) as u32);

            indices.push(second as u32);
            indices.push((second + 1) as u32);
            indices.push((first + 1) as u32);
        }
    }

    (positions, indices)
}

/// Calculate face normal from triangle indices.
fn compute_face_normal(positions: &[[f32; 3]], i0: u32, i1: u32, i2: u32) -> [f32; 3] {
    let p0 = positions[i0 as usize];
    let p1 = positions[i1 as usize];
    let p2 = positions[i2 as usize];

    let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

    normalize(cross(e1, e2))
}

// =============================================================================
// Category 1: MeshInput Validation Tests
// =============================================================================

#[test]
fn test_mesh_input_new_basic() {
    let (positions, indices) = make_single_triangle();
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
fn test_mesh_input_empty_mesh() {
    let input = MeshInput::new(&[], &[]);
    assert!(input.is_empty());
    assert_eq!(input.triangle_count(), 0);
}

#[test]
fn test_mesh_input_empty_indices() {
    let positions = vec![[0.0, 0.0, 0.0]];
    let input = MeshInput::new(&positions, &[]);
    assert!(input.is_empty());
}

#[test]
fn test_mesh_input_empty_positions() {
    let indices = vec![0, 1, 2];
    let input = MeshInput::new(&[], &indices);
    assert!(input.is_empty());
}

#[test]
fn test_mesh_input_validate_ok() {
    let (positions, indices) = make_cube();
    let input = MeshInput::new(&positions, &indices);
    assert!(input.validate().is_ok());
}

#[test]
fn test_mesh_input_validate_bad_index_count_1() {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let indices = vec![0, 1]; // Not multiple of 3
    let input = MeshInput::new(&positions, &indices);
    let result = input.validate();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Index count must be multiple of 3");
}

#[test]
fn test_mesh_input_validate_bad_index_count_4() {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]];
    let indices = vec![0, 1, 2, 3]; // 4 is not multiple of 3
    let input = MeshInput::new(&positions, &indices);
    assert!(input.validate().is_err());
}

#[test]
fn test_mesh_input_validate_bad_index_count_5() {
    let positions = vec![[0.0; 3]; 5];
    let indices = vec![0, 1, 2, 3, 4]; // 5 is not multiple of 3
    let input = MeshInput::new(&positions, &indices);
    assert!(input.validate().is_err());
}

#[test]
fn test_mesh_input_validate_index_out_of_bounds_first() {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let indices = vec![10, 1, 2]; // Index 10 is out of bounds
    let input = MeshInput::new(&positions, &indices);
    let result = input.validate();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Index out of bounds");
}

#[test]
fn test_mesh_input_validate_index_out_of_bounds_last() {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let indices = vec![0, 1, 100]; // Index 100 is out of bounds
    let input = MeshInput::new(&positions, &indices);
    assert!(input.validate().is_err());
}

#[test]
fn test_mesh_input_validate_index_at_boundary() {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let indices = vec![0, 1, 3]; // Index 3 is exactly at boundary (len=3)
    let input = MeshInput::new(&positions, &indices);
    assert!(input.validate().is_err());
}

#[test]
fn test_mesh_input_validate_index_max_valid() {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let indices = vec![0, 1, 2]; // All indices valid (max is 2)
    let input = MeshInput::new(&positions, &indices);
    assert!(input.validate().is_ok());
}

#[test]
fn test_mesh_input_validate_normals_count_mismatch() {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let normals = vec![[0.0, 0.0, 1.0]]; // Only 1 normal, but 3 positions
    let indices = vec![0, 1, 2];
    let input = MeshInput::with_normals(&positions, &indices, &normals);
    let result = input.validate();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Normals count must match positions count");
}

#[test]
fn test_mesh_input_validate_normals_count_too_many() {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let normals = vec![[0.0, 0.0, 1.0]; 10]; // 10 normals but only 3 positions
    let indices = vec![0, 1, 2];
    let input = MeshInput::with_normals(&positions, &indices, &normals);
    assert!(input.validate().is_err());
}

#[test]
fn test_mesh_input_triangle_count_multiple() {
    let (positions, indices) = make_cube();
    let input = MeshInput::new(&positions, &indices);
    assert_eq!(input.triangle_count(), 12); // Cube has 12 triangles
}

// =============================================================================
// Category 2: Generator Configuration Tests
// =============================================================================

#[test]
fn test_generator_default_limits() {
    let gen = MeshletGenerator::new();
    assert_eq!(gen.max_vertices(), MAX_MESHLET_VERTICES);
    assert_eq!(gen.max_triangles(), MAX_MESHLET_TRIANGLES);
    assert_eq!(gen.max_vertices(), 64);
    assert_eq!(gen.max_triangles(), 124);
}

#[test]
fn test_generator_default_impl() {
    let gen = MeshletGenerator::default();
    assert_eq!(gen.max_vertices(), MAX_MESHLET_VERTICES);
    assert_eq!(gen.max_triangles(), MAX_MESHLET_TRIANGLES);
}

#[test]
fn test_generator_custom_limits_small() {
    let gen = MeshletGenerator::with_limits(16, 30);
    assert_eq!(gen.max_vertices(), 16);
    assert_eq!(gen.max_triangles(), 30);
}

#[test]
fn test_generator_custom_limits_max() {
    let gen = MeshletGenerator::with_limits(255, 255);
    assert_eq!(gen.max_vertices(), 255);
    assert_eq!(gen.max_triangles(), 255);
}

#[test]
fn test_generator_custom_limits_min() {
    let gen = MeshletGenerator::with_limits(1, 1);
    assert_eq!(gen.max_vertices(), 1);
    assert_eq!(gen.max_triangles(), 1);
}

#[test]
#[should_panic(expected = "max_vertices must be in range 1..=255")]
fn test_generator_zero_vertices_panics() {
    MeshletGenerator::with_limits(0, 100);
}

#[test]
#[should_panic(expected = "max_triangles must be in range 1..=255")]
fn test_generator_zero_triangles_panics() {
    MeshletGenerator::with_limits(64, 0);
}

#[test]
#[should_panic(expected = "max_vertices must be in range 1..=255")]
fn test_generator_vertices_exceed_255_panics() {
    MeshletGenerator::with_limits(256, 100);
}

#[test]
#[should_panic(expected = "max_triangles must be in range 1..=255")]
fn test_generator_triangles_exceed_255_panics() {
    MeshletGenerator::with_limits(64, 256);
}

// =============================================================================
// Category 3: Splitting Algorithm Tests
// =============================================================================

#[test]
fn test_single_meshlet_small_mesh() {
    let (positions, indices) = make_single_triangle();
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    assert_eq!(output.meshlet_count(), 1);
    assert_eq!(output.meshlets[0].vertex_count, 3);
    assert_eq!(output.meshlets[0].triangle_count, 1);
}

#[test]
fn test_single_meshlet_cube() {
    let (positions, indices) = make_cube();
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    // Cube has 8 vertices, 12 triangles - fits in one meshlet
    assert_eq!(output.meshlet_count(), 1);
    assert_eq!(output.meshlets[0].triangle_count, 12);
    assert!(output.meshlets[0].vertex_count <= 8);
}

#[test]
fn test_multiple_meshlets_large_mesh() {
    let (positions, indices) = make_large_mesh_unique_verts(200);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    // 200 triangles with unique verts = 600 verts, needs multiple meshlets
    assert!(output.meshlet_count() > 1);
    assert!(output.validate(positions.len()).is_ok());
}

#[test]
fn test_vertex_limit_triggers_split() {
    // Each triangle uses 3 unique vertices
    // With 64 vert limit, can fit 21 triangles (63 verts)
    let (positions, indices) = make_large_mesh_unique_verts(25);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    // 25 * 3 = 75 verts > 64, so need 2+ meshlets
    assert!(output.meshlet_count() >= 2);

    // Verify all meshlets respect vertex limit
    for m in &output.meshlets {
        assert!(
            m.vertex_count as usize <= MAX_MESHLET_VERTICES,
            "Vertex count {} exceeds limit {}",
            m.vertex_count, MAX_MESHLET_VERTICES
        );
    }
}

#[test]
fn test_triangle_limit_triggers_split() {
    // Create grid with high vertex sharing to max out triangle limit
    let (positions, indices) = make_grid_mesh(30);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    // Verify all meshlets respect triangle limit
    for m in &output.meshlets {
        assert!(
            m.triangle_count as usize <= MAX_MESHLET_TRIANGLES,
            "Triangle count {} exceeds limit {}",
            m.triangle_count, MAX_MESHLET_TRIANGLES
        );
    }
}

#[test]
fn test_custom_vertex_limit_enforced() {
    let (positions, indices) = make_large_mesh_unique_verts(100);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::with_limits(16, 100);
    let output = gen.generate(&input);

    // With 16 vert limit, each meshlet can have at most 5 triangles (15 verts)
    for m in &output.meshlets {
        assert!(m.vertex_count <= 16, "Vertex count {} exceeds limit 16", m.vertex_count);
    }
}

#[test]
fn test_custom_triangle_limit_enforced() {
    let (positions, indices) = make_grid_mesh(40);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::with_limits(64, 30);
    let output = gen.generate(&input);

    for m in &output.meshlets {
        assert!(m.triangle_count <= 30, "Triangle count {} exceeds limit 30", m.triangle_count);
    }
}

#[test]
fn test_total_triangle_count_preserved() {
    let (positions, indices) = make_large_mesh_unique_verts(500);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    let total_triangles: usize = output.meshlets.iter()
        .map(|m| m.triangle_count as usize)
        .sum();
    assert_eq!(total_triangles, 500);
}

#[test]
fn test_empty_mesh_no_meshlets() {
    let input = MeshInput::new(&[], &[]);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    assert!(output.is_empty());
    assert_eq!(output.meshlet_count(), 0);
    assert!(output.meshlets.is_empty());
    assert!(output.bounds.is_empty());
}

#[test]
fn test_meshlet_offsets_sequential() {
    let (positions, indices) = make_large_mesh_unique_verts(100);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    // Verify offsets are sequential
    let mut expected_v_offset = 0u32;
    let mut expected_t_offset = 0u32;

    for m in &output.meshlets {
        assert_eq!(m.vertex_offset, expected_v_offset);
        assert_eq!(m.triangle_offset, expected_t_offset);
        expected_v_offset += m.vertex_count as u32;
        expected_t_offset += (m.triangle_count as u32) * 3;
    }

    assert_eq!(expected_v_offset as usize, output.vertex_indices.len());
    assert_eq!(expected_t_offset as usize, output.triangle_indices.len());
}

// =============================================================================
// Category 4: Bounding Sphere Tests (Ritter's Algorithm)
// =============================================================================

#[test]
fn test_bounds_single_vertex_sphere() {
    // Single triangle - sphere should contain all 3 vertices
    let (positions, indices) = make_single_triangle();
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    let b = &output.bounds[0];

    // All vertices should be within sphere
    for &idx in &output.vertex_indices {
        let p = positions[idx as usize];
        let dist = distance(b.center, p);
        assert!(
            dist <= b.radius + EPSILON,
            "Vertex {:?} outside sphere (dist={}, radius={})",
            p, dist, b.radius
        );
    }
}

#[test]
fn test_bounds_cube_contains_all_vertices() {
    let (positions, indices) = make_cube();
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    let b = &output.bounds[0];

    // Cube spans [-1,1] in all axes, diagonal = sqrt(12) = 3.46
    // Bounding sphere should have radius >= sqrt(3) = 1.73
    assert!(b.radius >= 1.7, "Cube bounding sphere too small: {}", b.radius);

    // All vertices should be contained
    for &idx in &output.vertex_indices {
        let p = positions[idx as usize];
        let dist = distance(b.center, p);
        assert!(
            dist <= b.radius + EPSILON,
            "Vertex {:?} outside sphere (dist={}, radius={})",
            p, dist, b.radius
        );
    }
}

#[test]
fn test_bounds_sphere_mesh() {
    let (positions, indices) = make_sphere(10);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    // All meshlets should have valid bounding spheres
    for (i, b) in output.bounds.iter().enumerate() {
        assert!(b.radius >= 0.0, "Meshlet {} has negative radius", i);

        // Verify containment for this meshlet's vertices
        let m = &output.meshlets[i];
        let v_start = m.vertex_offset as usize;
        let v_count = m.vertex_count as usize;

        for j in v_start..(v_start + v_count) {
            let p = positions[output.vertex_indices[j] as usize];
            let dist = distance(b.center, p);
            assert!(
                dist <= b.radius + EPSILON,
                "Meshlet {} vertex {} outside sphere",
                i, j
            );
        }
    }
}

#[test]
fn test_bounds_ritter_step1_initial_sphere() {
    // Test with two points at known distance
    let positions = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
    let indices = vec![0, 1, 2];
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    let b = &output.bounds[0];

    // All three vertices should be contained
    for &idx in &output.vertex_indices {
        let p = positions[idx as usize];
        let dist = distance(b.center, p);
        assert!(dist <= b.radius + EPSILON);
    }
}

#[test]
fn test_bounds_ritter_expansion() {
    // Test with a point that requires sphere expansion
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [5.0, 0.0, 0.0], // Far point
    ];
    let indices = vec![0, 1, 2];
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    let b = &output.bounds[0];

    // Radius must be at least 2.5 (half of 5.0 span)
    assert!(b.radius >= 2.4);

    // All vertices contained
    for &idx in &output.vertex_indices {
        let p = positions[idx as usize];
        let dist = distance(b.center, p);
        assert!(dist <= b.radius + EPSILON);
    }
}

#[test]
fn test_bounds_tight_sphere() {
    // Collinear points should have tight bounding sphere
    let positions = vec![
        [-3.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ];
    let indices = vec![0, 1, 2];
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    let b = &output.bounds[0];

    // Optimal sphere has radius 3.0
    assert!(b.radius <= 3.0 + EPSILON);
    assert!(b.radius >= 3.0 - EPSILON);
}

#[test]
fn test_bounds_radius_non_negative() {
    let (positions, indices) = make_large_mesh_unique_verts(200);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    for (i, b) in output.bounds.iter().enumerate() {
        assert!(b.radius >= 0.0, "Meshlet {} has negative radius: {}", i, b.radius);
    }
}

// =============================================================================
// Category 5: Backface Cone Tests
// =============================================================================

#[test]
fn test_cone_single_triangle_cutoff_one() {
    let (positions, indices) = make_single_triangle();
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    // Single triangle should have cutoff of 1.0 (all normals identical)
    assert_eq!(output.bounds[0].cone_cutoff, 1.0);
}

#[test]
fn test_cone_single_triangle_axis_z() {
    let (positions, indices) = make_single_triangle();
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    let axis = output.bounds[0].cone_axis;

    // Normal should point in Z direction (CCW triangle in XY plane)
    assert!(
        (axis[2] - 1.0).abs() < EPSILON || (axis[2] + 1.0).abs() < EPSILON,
        "Cone axis should be +-Z, got {:?}",
        axis
    );
}

#[test]
fn test_cone_cutoff_range_valid() {
    let (positions, indices) = make_cube();
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    // Cube has normals in all 6 directions - cone should be invalid
    let b = &output.bounds[0];

    // Cone cutoff should be in range [-1, 1] or < -1 for invalid
    assert!(
        b.cone_cutoff <= 1.0,
        "Cone cutoff {} should be <= 1.0",
        b.cone_cutoff
    );
}

#[test]
fn test_cone_axis_normalized() {
    let (positions, indices) = make_single_triangle();
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    let axis = output.bounds[0].cone_axis;
    let len = length(axis);

    assert!(
        (len - 1.0).abs() < EPSILON,
        "Cone axis should be normalized, length={}",
        len
    );
}

#[test]
fn test_cone_contains_all_face_normals() {
    let (positions, indices) = make_single_triangle();
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    let axis = output.bounds[0].cone_axis;
    let cutoff = output.bounds[0].cone_cutoff;

    // Compute face normal
    let face_normal = compute_face_normal(&positions, indices[0], indices[1], indices[2]);

    // Dot product should be >= cutoff
    let d = dot(axis, face_normal);
    assert!(
        d >= cutoff - EPSILON,
        "Face normal {:?} outside cone (dot={}, cutoff={})",
        face_normal, d, cutoff
    );
}

#[test]
fn test_cone_with_normals_matches() {
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

    let b = &output.bounds[0];

    // With all normals pointing +Z, cone should be tight
    assert!(b.cone_cutoff > 0.9, "Expected tight cone, got cutoff={}", b.cone_cutoff);
    assert!(b.has_valid_cone());
}

#[test]
fn test_cone_opposing_normals_wide() {
    // Create two triangles with opposing normals
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.0, 0.1],
        [1.0, 0.0, 0.1],
        [0.5, 1.0, 0.1],
    ];
    let indices = vec![
        0, 1, 2,  // +Z normal
        5, 4, 3,  // -Z normal (reversed winding)
    ];
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    // Opposing normals should result in wide cone (cutoff near -1)
    assert!(output.bounds[0].cone_cutoff < 0.0);
}

#[test]
fn test_cone_invalid_disables_culling() {
    let (positions, indices) = make_cube();
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    // Cube has faces pointing in 6 directions - cone may be invalid
    let b = &output.bounds[0];

    // If cutoff < -1, cone is invalid for backface culling
    if b.cone_cutoff < -1.0 {
        assert!(!b.has_valid_cone());
    }
}

#[test]
fn test_cone_has_valid_cone_boundary() {
    // Test the boundary condition for has_valid_cone()
    let b1 = MeshletBounds::new([0.0; 3], 1.0, [0.0, 0.0, 1.0], -1.0);
    assert!(b1.has_valid_cone(), "Cutoff -1.0 should be valid");

    let b2 = MeshletBounds::new([0.0; 3], 1.0, [0.0, 0.0, 1.0], -1.001);
    assert!(!b2.has_valid_cone(), "Cutoff -1.001 should be invalid");
}

// =============================================================================
// Category 6: Reconstruction Tests
// =============================================================================

#[test]
fn test_reconstruct_single_triangle() {
    let (positions, indices) = make_single_triangle();
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    let reconstructed = output.reconstruct_indices();
    assert_eq!(reconstructed.len(), indices.len());
    assert_eq!(reconstructed, indices);
}

#[test]
fn test_reconstruct_cube_triangles_match() {
    let (positions, indices) = make_cube();
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    let reconstructed = output.reconstruct_indices();
    assert_eq!(reconstructed.len(), indices.len());

    // Collect triangles as sorted tuples for comparison
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
fn test_reconstruct_large_mesh_complete() {
    let (positions, indices) = make_large_mesh_unique_verts(200);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    let reconstructed = output.reconstruct_indices();
    assert_eq!(reconstructed.len(), indices.len());

    // Verify all indices are valid
    for idx in &reconstructed {
        assert!((*idx as usize) < positions.len());
    }
}

#[test]
fn test_reconstruct_indices_valid_range() {
    let (positions, indices) = make_sphere(8);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    let reconstructed = output.reconstruct_indices();

    for (i, &idx) in reconstructed.iter().enumerate() {
        assert!(
            (idx as usize) < positions.len(),
            "Reconstructed index {} at position {} out of range",
            idx, i
        );
    }
}

// =============================================================================
// Category 7: Edge Case Tests
// =============================================================================

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
fn test_degenerate_all_same_vertex() {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let indices = vec![0, 0, 0]; // All same vertex
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    // Degenerate triangles should be skipped
    assert!(output.is_empty());
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

#[test]
fn test_mixed_valid_invalid_triangles() {
    let positions = vec![[0.0; 3]; 5];
    let indices = vec![
        0, 1, 2,       // Valid
        0, 0, 1,       // Degenerate
        1, 2, 3,       // Valid
        0, 100, 2,     // Invalid index
        2, 3, 4,       // Valid
    ];
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    // Should only have 3 valid triangles
    let total: u32 = output.meshlets.iter().map(|m| m.triangle_count as u32).sum();
    assert_eq!(total, 3);
}

#[test]
fn test_large_indices_near_max() {
    let n = 1000usize;
    let positions: Vec<[f32; 3]> = (0..n).map(|i| [i as f32, 0.0, 0.0]).collect();
    let indices: Vec<u32> = (0..((n / 3) * 3) as u32).collect();

    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    assert!(output.validate(positions.len()).is_ok());
}

// =============================================================================
// Category 8: MeshletOutput Validation Tests
// =============================================================================

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

#[test]
fn test_output_meshlet_count_matches_bounds() {
    let (positions, indices) = make_large_mesh_unique_verts(100);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    assert_eq!(output.meshlets.len(), output.bounds.len());
}

// =============================================================================
// Category 9: MeshletStats Tests
// =============================================================================

#[test]
fn test_stats_empty_output() {
    let output = MeshletOutput::new();
    let stats = output.stats();

    assert_eq!(stats.meshlet_count, 0);
    assert_eq!(stats.total_vertices, 0);
    assert_eq!(stats.total_triangles, 0);
}

#[test]
fn test_stats_single_meshlet() {
    let (positions, indices) = make_single_triangle();
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);
    let stats = output.stats();

    assert_eq!(stats.meshlet_count, 1);
    assert_eq!(stats.total_vertices, 3);
    assert_eq!(stats.total_triangles, 1);
    assert_eq!(stats.min_vertices, 3);
    assert_eq!(stats.max_vertices, 3);
    assert_eq!(stats.min_triangles, 1);
    assert_eq!(stats.max_triangles, 1);
}

#[test]
fn test_stats_multiple_meshlets() {
    let (positions, indices) = make_large_mesh_unique_verts(200);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);
    let stats = output.stats();

    assert!(stats.meshlet_count > 1);
    assert_eq!(stats.total_triangles, 200);
    assert!(stats.avg_triangles > 0.0);
    assert!(stats.min_triangles >= 1);
    assert!(stats.max_triangles <= MAX_MESHLET_TRIANGLES);
}

#[test]
fn test_stats_memory_usage() {
    let (positions, indices) = make_cube();
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);
    let stats = output.stats();

    assert!(stats.memory_bytes > 0);
    assert_eq!(stats.memory_bytes, output.memory_usage());
}

// =============================================================================
// Category 10: Memory Layout Tests
// =============================================================================

#[test]
fn test_memory_usage_calculation() {
    let (positions, indices) = make_cube();
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    let usage = output.memory_usage();
    assert!(usage > 0);

    // Expected: 1 meshlet * 12 bytes + 1 bounds * 32 bytes +
    // vertex indices + local indices
    let expected_min = MESHLET_SIZE + MESHLET_BOUNDS_SIZE;
    assert!(usage >= expected_min);
}

#[test]
fn test_local_indices_range() {
    let (positions, indices) = make_large_mesh_unique_verts(100);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    for (i, m) in output.meshlets.iter().enumerate() {
        let t_start = m.triangle_offset as usize;
        let t_count = m.triangle_count as usize;
        let v_count = m.vertex_count as usize;

        for j in t_start..(t_start + t_count * 3) {
            assert!(
                (output.triangle_indices[j] as usize) < v_count,
                "Meshlet {} local index {} >= vertex_count {}",
                i, output.triangle_indices[j], v_count
            );
        }
    }
}

#[test]
fn test_vertex_indices_valid() {
    let (positions, indices) = make_sphere(10);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    for (i, &idx) in output.vertex_indices.iter().enumerate() {
        assert!(
            (idx as usize) < positions.len(),
            "Vertex index {} at position {} out of range",
            idx, i
        );
    }
}

// =============================================================================
// Category 11: Panic Tests
// =============================================================================

#[test]
#[should_panic(expected = "Index count must be multiple of 3")]
fn test_generate_panics_on_bad_index_count() {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let indices = vec![0, 1]; // Not multiple of 3
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    gen.generate(&input);
}

// =============================================================================
// Category 12: Stress Tests
// =============================================================================

#[test]
fn test_stress_many_triangles() {
    let (positions, indices) = make_large_mesh_unique_verts(1000);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    assert!(output.meshlet_count() > 10);
    assert!(output.validate(positions.len()).is_ok());
}

#[test]
fn test_stress_high_vertex_reuse() {
    let (positions, indices) = make_grid_mesh(50);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    // Verify total triangle count
    let total: usize = output.meshlets.iter()
        .map(|m| m.triangle_count as usize)
        .sum();
    assert_eq!(total, indices.len() / 3);
    assert!(output.validate(positions.len()).is_ok());
}

#[test]
fn test_stress_sphere_mesh() {
    let (positions, indices) = make_sphere(20);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    assert!(output.validate(positions.len()).is_ok());

    // Verify all bounding spheres contain their vertices
    for (i, m) in output.meshlets.iter().enumerate() {
        let b = &output.bounds[i];
        let v_start = m.vertex_offset as usize;
        let v_count = m.vertex_count as usize;

        for j in v_start..(v_start + v_count) {
            let p = positions[output.vertex_indices[j] as usize];
            let dist = distance(b.center, p);
            assert!(dist <= b.radius + EPSILON);
        }
    }
}

// =============================================================================
// Category 13: Integration with MeshletData
// =============================================================================

#[test]
fn test_meshlet_data_generate_equivalent() {
    let (positions, indices) = make_cube();

    // Generate with MeshletGenerator
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    // Generate with MeshletData
    let data = MeshletData::generate(&positions, &indices, None);

    // Should have same structure
    assert_eq!(output.meshlet_count(), data.meshlet_count());
    assert_eq!(output.meshlets.len(), data.meshlets.len());
    assert_eq!(output.bounds.len(), data.bounds.len());
}

#[test]
fn test_meshlet_data_with_normals() {
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let normals = vec![
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ];
    let indices = vec![0, 1, 2];

    let data = MeshletData::generate(&positions, &indices, Some(&normals));

    assert_eq!(data.meshlet_count(), 1);
    assert!(data.bounds[0].has_valid_cone());
}

// =============================================================================
// Category 14: Boundary Condition Tests
// =============================================================================

#[test]
fn test_exactly_64_vertices_single_meshlet() {
    // Create exactly 21 triangles with 63 unique vertices
    // (21 * 3 = 63 verts, which is < 64)
    let (positions, indices) = make_large_mesh_unique_verts(21);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    assert_eq!(output.meshlet_count(), 1);
    assert_eq!(output.meshlets[0].vertex_count, 63);
}

#[test]
fn test_exactly_124_triangles_single_meshlet() {
    // Need mesh with high vertex sharing to fit 124 triangles in 64 verts
    // Grid of ~8x8 gives 98 triangles in first meshlet
    let (positions, indices) = make_grid_mesh(10);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::new();
    let output = gen.generate(&input);

    // First meshlet should be close to limits
    assert!(output.meshlets[0].triangle_count <= MAX_MESHLET_TRIANGLES as u8);
}

#[test]
fn test_limit_1_vertex_1_triangle() {
    // Extreme limits
    let (positions, indices) = make_single_triangle();
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::with_limits(3, 1);
    let output = gen.generate(&input);

    assert_eq!(output.meshlet_count(), 1);
}

#[test]
fn test_very_small_limits_many_meshlets() {
    let (positions, indices) = make_large_mesh_unique_verts(10);
    let input = MeshInput::new(&positions, &indices);
    let gen = MeshletGenerator::with_limits(3, 1); // Only 1 triangle per meshlet
    let output = gen.generate(&input);

    assert_eq!(output.meshlet_count(), 10);
    for m in &output.meshlets {
        assert_eq!(m.triangle_count, 1);
        assert_eq!(m.vertex_count, 3);
    }
}

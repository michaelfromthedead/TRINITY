//! Blackbox tests for MeshletGenerator (T-WGPU-P6.9.2).
//!
//! This module tests the MeshletGenerator component through its public API,
//! verifying correct behavior across a variety of mesh inputs, edge cases,
//! and integration scenarios.
//!
//! # Test Categories
//!
//! 1. API behavior - default construction, generator reuse, output validation
//! 2. Mesh processing - single triangle, cube, large meshes
//! 3. Output verification - reconstruct_indices(), triangle accounting
//! 4. Integration scenarios - GPU buffer readiness, stats, memory usage
//! 5. Edge cases - degenerate triangles, high vertex reuse, meshlet sizing

use renderer_backend::gpu_driven::meshlet_generator::{
    MeshInput, MeshletGenerator, MeshletOutput, MAX_MESHLET_TRIANGLES, MAX_MESHLET_VERTICES,
};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Test Utilities
// ---------------------------------------------------------------------------

const EPSILON: f32 = 1e-6;

/// Compute Euclidean distance between two 3D points.
fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let dz = b[2] - a[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Create a single triangle mesh.
fn make_single_triangle() -> (Vec<[f32; 3]>, Vec<u32>) {
    let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let indices = vec![0, 1, 2];
    (positions, indices)
}

/// Create a unit cube mesh (8 vertices, 12 triangles).
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

/// Create a cube with per-vertex normals.
fn make_cube_with_normals() -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
    // Expand cube to have separate vertices per face for correct normals
    let positions = vec![
        // Front face (+Z)
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        // Back face (-Z)
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        // Top face (+Y)
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        // Bottom face (-Y)
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [-1.0, -1.0, 1.0],
        // Right face (+X)
        [1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        // Left face (-X)
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
    ];

    let normals = vec![
        // Front face (+Z)
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        // Back face (-Z)
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        // Top face (+Y)
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        // Bottom face (-Y)
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0],
        // Right face (+X)
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        // Left face (-X)
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ];

    let indices = vec![
        // Front
        0, 1, 2, 0, 2, 3, // Back
        4, 5, 6, 4, 6, 7, // Top
        8, 9, 10, 8, 10, 11, // Bottom
        12, 13, 14, 12, 14, 15, // Right
        16, 17, 18, 16, 18, 19, // Left
        20, 21, 22, 20, 22, 23,
    ];

    (positions, normals, indices)
}

/// Create a large mesh with many separate triangles.
/// Each triangle has 3 unique vertices, forcing many meshlets.
fn make_large_mesh(triangle_count: usize) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut positions = Vec::with_capacity(triangle_count * 3);
    let mut indices = Vec::with_capacity(triangle_count * 3);

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
    let mut positions = Vec::with_capacity(grid_size * grid_size);
    let mut indices = Vec::new();

    for y in 0..grid_size {
        for x in 0..grid_size {
            positions.push([x as f32, y as f32, 0.0]);
        }
    }

    for y in 0..(grid_size - 1) {
        for x in 0..(grid_size - 1) {
            let tl = (y * grid_size + x) as u32;
            let tr = (y * grid_size + x + 1) as u32;
            let bl = ((y + 1) * grid_size + x) as u32;
            let br = ((y + 1) * grid_size + x + 1) as u32;

            indices.push(tl);
            indices.push(bl);
            indices.push(tr);

            indices.push(tr);
            indices.push(bl);
            indices.push(br);
        }
    }

    (positions, indices)
}

/// Extract triangles as sorted tuples for comparison (order-independent).
fn extract_sorted_triangles(indices: &[u32]) -> HashSet<[u32; 3]> {
    indices
        .chunks(3)
        .map(|c| {
            let mut t = [c[0], c[1], c[2]];
            t.sort();
            t
        })
        .collect()
}

// ===========================================================================
// 1. API BEHAVIOR
// ===========================================================================

mod api_behavior {
    use super::*;

    #[test]
    fn default_construction_uses_standard_limits() {
        let gen = MeshletGenerator::new();
        assert_eq!(gen.max_vertices(), MAX_MESHLET_VERTICES);
        assert_eq!(gen.max_triangles(), MAX_MESHLET_TRIANGLES);
    }

    #[test]
    fn default_trait_matches_new() {
        let gen1 = MeshletGenerator::new();
        let gen2 = MeshletGenerator::default();
        assert_eq!(gen1.max_vertices(), gen2.max_vertices());
        assert_eq!(gen1.max_triangles(), gen2.max_triangles());
    }

    #[test]
    fn custom_limits_are_respected() {
        let gen = MeshletGenerator::with_limits(32, 62);
        assert_eq!(gen.max_vertices(), 32);
        assert_eq!(gen.max_triangles(), 62);
    }

    #[test]
    #[should_panic(expected = "max_vertices must be in range 1..=255")]
    fn zero_max_vertices_panics() {
        MeshletGenerator::with_limits(0, 100);
    }

    #[test]
    #[should_panic(expected = "max_triangles must be in range 1..=255")]
    fn zero_max_triangles_panics() {
        MeshletGenerator::with_limits(64, 0);
    }

    #[test]
    #[should_panic(expected = "max_vertices must be in range 1..=255")]
    fn max_vertices_over_255_panics() {
        MeshletGenerator::with_limits(256, 100);
    }

    #[test]
    #[should_panic(expected = "max_triangles must be in range 1..=255")]
    fn max_triangles_over_255_panics() {
        MeshletGenerator::with_limits(64, 256);
    }

    #[test]
    fn generator_can_be_reused_multiple_times() {
        let gen = MeshletGenerator::new();

        // Generate from cube
        let (positions1, indices1) = make_cube();
        let input1 = MeshInput::new(&positions1, &indices1);
        let output1 = gen.generate(&input1);

        // Generate from different mesh
        let (positions2, indices2) = make_single_triangle();
        let input2 = MeshInput::new(&positions2, &indices2);
        let output2 = gen.generate(&input2);

        // Both should be valid and independent
        assert_eq!(output1.meshlet_count(), 1);
        assert_eq!(output1.meshlets[0].triangle_count, 12);

        assert_eq!(output2.meshlet_count(), 1);
        assert_eq!(output2.meshlets[0].triangle_count, 1);
    }

    #[test]
    fn generator_is_copy() {
        let gen1 = MeshletGenerator::with_limits(32, 62);
        let gen2 = gen1; // Copy
        assert_eq!(gen1.max_vertices(), gen2.max_vertices());
        assert_eq!(gen1.max_triangles(), gen2.max_triangles());
    }

    #[test]
    fn generator_is_clone() {
        let gen1 = MeshletGenerator::with_limits(32, 62);
        let gen2 = gen1.clone();
        assert_eq!(gen1.max_vertices(), gen2.max_vertices());
        assert_eq!(gen1.max_triangles(), gen2.max_triangles());
    }

    #[test]
    fn mesh_input_new_without_normals() {
        let (positions, indices) = make_single_triangle();
        let input = MeshInput::new(&positions, &indices);
        assert!(input.normals.is_none());
        assert_eq!(input.triangle_count(), 1);
        assert!(!input.is_empty());
    }

    #[test]
    fn mesh_input_with_normals() {
        let (positions, normals, indices) = make_cube_with_normals();
        let input = MeshInput::with_normals(&positions, &indices, &normals);
        assert!(input.normals.is_some());
        assert_eq!(input.normals.unwrap().len(), 24);
    }

    #[test]
    fn mesh_input_validate_correct_data() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);
        assert!(input.validate().is_ok());
    }

    #[test]
    fn mesh_input_validate_index_not_multiple_of_3() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let indices = vec![0, 1]; // Not multiple of 3
        let input = MeshInput::new(&positions, &indices);
        assert!(input.validate().is_err());
    }

    #[test]
    fn mesh_input_validate_index_out_of_bounds() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = vec![0, 1, 10]; // Index 10 is out of bounds
        let input = MeshInput::new(&positions, &indices);
        assert!(input.validate().is_err());
    }

    #[test]
    fn mesh_input_validate_normals_mismatch() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let normals = vec![[0.0, 0.0, 1.0]]; // Wrong count
        let indices = vec![0, 1, 2];
        let input = MeshInput::with_normals(&positions, &indices, &normals);
        assert!(input.validate().is_err());
    }

    #[test]
    fn output_default_is_empty() {
        let output = MeshletOutput::new();
        assert!(output.is_empty());
        assert_eq!(output.meshlet_count(), 0);
    }
}

// ===========================================================================
// 2. MESH PROCESSING
// ===========================================================================

mod mesh_processing {
    use super::*;

    #[test]
    fn single_triangle_produces_one_meshlet() {
        let (positions, indices) = make_single_triangle();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert_eq!(output.meshlet_count(), 1);
        assert_eq!(output.meshlets[0].vertex_count, 3);
        assert_eq!(output.meshlets[0].triangle_count, 1);
    }

    #[test]
    fn single_triangle_passes_validation() {
        let (positions, indices) = make_single_triangle();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert!(output.validate(positions.len()).is_ok());
    }

    #[test]
    fn cube_produces_one_meshlet() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        // 8 vertices, 12 triangles fits easily in one meshlet
        assert_eq!(output.meshlet_count(), 1);
        assert_eq!(output.meshlets[0].triangle_count, 12);
        assert!(output.meshlets[0].vertex_count <= 8);
    }

    #[test]
    fn cube_bounding_sphere_contains_all_vertices() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        let b = &output.bounds[0];
        for idx in &output.vertex_indices {
            let p = positions[*idx as usize];
            let dist = distance(b.center, p);
            assert!(
                dist <= b.radius + EPSILON,
                "Vertex {:?} outside bounding sphere (dist={}, radius={})",
                p,
                dist,
                b.radius
            );
        }
    }

    #[test]
    fn large_mesh_produces_multiple_meshlets() {
        let (positions, indices) = make_large_mesh(200);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        // 200 triangles * 3 unique verts = 600 verts
        // With 64 vert limit, need at least 10 meshlets
        assert!(
            output.meshlet_count() > 1,
            "Expected multiple meshlets, got {}",
            output.meshlet_count()
        );
    }

    #[test]
    fn large_mesh_respects_vertex_limit() {
        let (positions, indices) = make_large_mesh(500);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        for (i, m) in output.meshlets.iter().enumerate() {
            assert!(
                m.vertex_count as usize <= MAX_MESHLET_VERTICES,
                "Meshlet {} has {} vertices, exceeds limit {}",
                i,
                m.vertex_count,
                MAX_MESHLET_VERTICES
            );
        }
    }

    #[test]
    fn grid_mesh_respects_triangle_limit() {
        // Grid mesh has high vertex sharing, will hit triangle limit
        let (positions, indices) = make_grid_mesh(30);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        for (i, m) in output.meshlets.iter().enumerate() {
            assert!(
                m.triangle_count as usize <= MAX_MESHLET_TRIANGLES,
                "Meshlet {} has {} triangles, exceeds limit {}",
                i,
                m.triangle_count,
                MAX_MESHLET_TRIANGLES
            );
        }
    }

    #[test]
    fn custom_small_limits_produce_many_meshlets() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        // Very small limits
        let gen = MeshletGenerator::with_limits(4, 2);
        let output = gen.generate(&input);

        // Should produce many small meshlets
        assert!(
            output.meshlet_count() > 1,
            "Expected multiple meshlets with small limits"
        );

        for m in &output.meshlets {
            assert!(m.vertex_count <= 4);
            assert!(m.triangle_count <= 2);
        }
    }

    #[test]
    fn empty_mesh_produces_empty_output() {
        let input = MeshInput::new(&[], &[]);
        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert!(output.is_empty());
        assert_eq!(output.meshlet_count(), 0);
        assert!(output.vertex_indices.is_empty());
        assert!(output.triangle_indices.is_empty());
    }

    #[test]
    fn mesh_with_only_positions_no_indices_is_empty() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let input = MeshInput::new(&positions, &[]);
        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert!(output.is_empty());
    }

    #[test]
    fn with_normals_generates_valid_cones() {
        let (positions, normals, indices) = make_cube_with_normals();
        let input = MeshInput::with_normals(&positions, &indices, &normals);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        // All bounds should have valid cones since we provided normals
        for b in &output.bounds {
            // Cone cutoff might be < -1 if normals face many directions
            // but should be computed (not default -2.0 from invalid state)
            // The cube faces 6 directions so cutoff will be very negative
            assert!(b.cone_cutoff >= -2.0);
        }
    }
}

// ===========================================================================
// 3. OUTPUT VERIFICATION
// ===========================================================================

mod output_verification {
    use super::*;

    #[test]
    fn reconstruct_indices_matches_triangle_count() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);
        let reconstructed = output.reconstruct_indices();

        assert_eq!(reconstructed.len(), indices.len());
    }

    #[test]
    fn reconstruct_indices_contains_same_triangles() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);
        let reconstructed = output.reconstruct_indices();

        let orig_tris = extract_sorted_triangles(&indices);
        let recon_tris = extract_sorted_triangles(&reconstructed);

        assert_eq!(
            orig_tris, recon_tris,
            "Reconstructed triangles don't match original"
        );
    }

    #[test]
    fn large_mesh_reconstruction_preserves_all_triangles() {
        let (positions, indices) = make_large_mesh(200);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);
        let reconstructed = output.reconstruct_indices();

        assert_eq!(reconstructed.len(), indices.len());

        let orig_tris = extract_sorted_triangles(&indices);
        let recon_tris = extract_sorted_triangles(&reconstructed);

        assert_eq!(orig_tris, recon_tris);
    }

    #[test]
    fn all_reconstructed_indices_are_valid() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);
        let reconstructed = output.reconstruct_indices();

        for idx in &reconstructed {
            assert!(
                (*idx as usize) < positions.len(),
                "Reconstructed index {} out of range",
                idx
            );
        }
    }

    #[test]
    fn no_vertex_duplication_within_meshlet() {
        let (positions, indices) = make_large_mesh(100);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        for (mi, meshlet) in output.meshlets.iter().enumerate() {
            let v_start = meshlet.vertex_offset as usize;
            let v_count = meshlet.vertex_count as usize;

            let vertex_slice = &output.vertex_indices[v_start..v_start + v_count];
            let unique: HashSet<_> = vertex_slice.iter().collect();

            assert_eq!(
                unique.len(),
                v_count,
                "Meshlet {} has duplicate vertex indices",
                mi
            );
        }
    }

    #[test]
    fn local_indices_reference_valid_local_vertices() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        for (mi, meshlet) in output.meshlets.iter().enumerate() {
            let t_start = meshlet.triangle_offset as usize;
            let t_count = meshlet.triangle_count as usize;
            let v_count = meshlet.vertex_count;

            for i in 0..(t_count * 3) {
                let local_idx = output.triangle_indices[t_start + i];
                assert!(
                    local_idx < v_count,
                    "Meshlet {} local index {} exceeds vertex count {}",
                    mi,
                    local_idx,
                    v_count
                );
            }
        }
    }

    #[test]
    fn output_validate_succeeds_for_valid_data() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert!(output.validate(positions.len()).is_ok());
    }

    #[test]
    fn output_validate_catches_corrupted_vertex_index() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let mut output = gen.generate(&input);

        // Corrupt a vertex index
        output.vertex_indices[0] = 999;

        assert!(output.validate(positions.len()).is_err());
    }

    #[test]
    fn output_validate_catches_corrupted_local_index() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let mut output = gen.generate(&input);

        // Corrupt a local index
        output.triangle_indices[0] = 200;

        assert!(output.validate(positions.len()).is_err());
    }

    #[test]
    fn total_triangles_across_meshlets_matches_input() {
        let (positions, indices) = make_large_mesh(150);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        let total: usize = output
            .meshlets
            .iter()
            .map(|m| m.triangle_count as usize)
            .sum();

        assert_eq!(total, 150, "Total triangles mismatch");
    }

    #[test]
    fn meshlet_offsets_are_consistent() {
        let (positions, indices) = make_large_mesh(200);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        let mut expected_v_offset = 0u32;
        let mut expected_t_offset = 0u32;

        for (i, m) in output.meshlets.iter().enumerate() {
            assert_eq!(
                m.vertex_offset, expected_v_offset,
                "Meshlet {} vertex offset mismatch",
                i
            );
            assert_eq!(
                m.triangle_offset, expected_t_offset,
                "Meshlet {} triangle offset mismatch",
                i
            );

            expected_v_offset += m.vertex_count as u32;
            expected_t_offset += m.triangle_count as u32 * 3;
        }

        assert_eq!(expected_v_offset as usize, output.vertex_indices.len());
        assert_eq!(expected_t_offset as usize, output.triangle_indices.len());
    }
}

// ===========================================================================
// 4. INTEGRATION SCENARIOS
// ===========================================================================

mod integration {
    use super::*;

    #[test]
    fn meshlet_output_ready_for_gpu_buffers() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        // Verify data can be cast to bytes (simulating GPU upload)
        let meshlet_bytes: &[u8] = bytemuck::cast_slice(&output.meshlets);
        assert!(!meshlet_bytes.is_empty());
        assert_eq!(meshlet_bytes.len(), output.meshlets.len() * 12); // 12 bytes per Meshlet

        let bounds_bytes: &[u8] = bytemuck::cast_slice(&output.bounds);
        assert!(!bounds_bytes.is_empty());
        assert_eq!(bounds_bytes.len(), output.bounds.len() * 32); // 32 bytes per MeshletBounds

        let vertex_bytes: &[u8] = bytemuck::cast_slice(&output.vertex_indices);
        assert_eq!(vertex_bytes.len(), output.vertex_indices.len() * 4);

        // triangle_indices is already u8
        assert!(!output.triangle_indices.is_empty());
    }

    #[test]
    fn stats_provides_accurate_information() {
        let (positions, indices) = make_large_mesh(200);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);
        let stats = output.stats();

        assert_eq!(stats.meshlet_count, output.meshlet_count());
        assert_eq!(stats.total_triangles, 200);
        assert!(stats.avg_triangles > 0.0);
        assert!(stats.avg_vertices > 0.0);
        assert!(stats.min_vertices <= stats.max_vertices);
        assert!(stats.min_triangles <= stats.max_triangles);
        assert!(stats.memory_bytes > 0);
    }

    #[test]
    fn stats_on_empty_output_has_defaults() {
        let output = MeshletOutput::new();
        let stats = output.stats();

        assert_eq!(stats.meshlet_count, 0);
        assert_eq!(stats.total_triangles, 0);
        assert_eq!(stats.total_vertices, 0);
    }

    #[test]
    fn memory_usage_is_reasonable() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);
        let usage = output.memory_usage();

        // For cube: 1 meshlet (12) + 1 bounds (32) + ~8 vertex indices (32) + 36 local indices
        // ~= 112 bytes
        assert!(usage > 0);
        assert!(usage < 1000, "Memory usage seems too high for cube: {}", usage);
    }

    #[test]
    fn memory_usage_scales_with_mesh_size() {
        let gen = MeshletGenerator::new();

        let (pos1, idx1) = make_large_mesh(100);
        let output1 = gen.generate(&MeshInput::new(&pos1, &idx1));
        let usage1 = output1.memory_usage();

        let (pos2, idx2) = make_large_mesh(1000);
        let output2 = gen.generate(&MeshInput::new(&pos2, &idx2));
        let usage2 = output2.memory_usage();

        // 10x more triangles should use more memory
        assert!(
            usage2 > usage1,
            "Larger mesh should use more memory: {} vs {}",
            usage2,
            usage1
        );
    }

    #[test]
    fn bounds_count_matches_meshlet_count() {
        let (positions, indices) = make_large_mesh(300);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert_eq!(output.meshlets.len(), output.bounds.len());
    }

    #[test]
    fn bounding_spheres_contain_meshlet_vertices() {
        let (positions, indices) = make_large_mesh(100);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        for (mi, meshlet) in output.meshlets.iter().enumerate() {
            let v_start = meshlet.vertex_offset as usize;
            let v_count = meshlet.vertex_count as usize;
            let bounds = &output.bounds[mi];

            for i in v_start..(v_start + v_count) {
                let global_idx = output.vertex_indices[i] as usize;
                let p = positions[global_idx];
                let dist = distance(bounds.center, p);
                assert!(
                    dist <= bounds.radius + EPSILON,
                    "Meshlet {} vertex outside bounds",
                    mi
                );
            }
        }
    }

    #[test]
    fn normal_cones_are_valid_or_disabled() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        for bounds in &output.bounds {
            // Either valid (-1 <= cutoff <= 1) or disabled (cutoff < -1)
            assert!(bounds.cone_cutoff <= 1.0);
            // The has_valid_cone method should work correctly
            if bounds.cone_cutoff >= -1.0 {
                assert!(bounds.has_valid_cone());
            } else {
                assert!(!bounds.has_valid_cone());
            }
        }
    }

    #[test]
    fn round_trip_preserves_geometry() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        // Reconstruct and verify we can compute same vertex positions
        let reconstructed = output.reconstruct_indices();

        for (i, chunk) in reconstructed.chunks(3).enumerate() {
            let v0 = positions[chunk[0] as usize];
            let v1 = positions[chunk[1] as usize];
            let v2 = positions[chunk[2] as usize];

            // Compute triangle normal (basic sanity check)
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();

            // Non-degenerate triangles should have non-zero normal
            assert!(len > EPSILON, "Reconstructed triangle {} is degenerate", i);
        }
    }
}

// ===========================================================================
// 5. EDGE CASES
// ===========================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn degenerate_triangle_indices_equal_skipped() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        // First triangle is degenerate (0, 0, 1)
        let indices = vec![0, 0, 1, 0, 1, 2];
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        // Only the valid triangle should be processed
        assert_eq!(output.meshlets[0].triangle_count, 1);
    }

    #[test]
    fn degenerate_triangle_all_same_skipped() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = vec![0, 0, 0, 0, 1, 2];
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert_eq!(output.meshlets[0].triangle_count, 1);
    }

    #[test]
    fn out_of_range_indices_skipped() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        // Second triangle has invalid index
        let indices = vec![0, 1, 2, 0, 1, 100];
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert_eq!(output.meshlets[0].triangle_count, 1);
    }

    #[test]
    fn all_degenerate_triangles_produces_empty_output() {
        let positions = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let indices = vec![0, 0, 1, 1, 1, 2, 2, 2, 0];
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert!(output.is_empty());
    }

    #[test]
    fn high_vertex_reuse_hits_triangle_limit() {
        // Create mesh where all triangles share many vertices
        let (positions, indices) = make_grid_mesh(20);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        // Should create multiple meshlets due to triangle limit
        // Grid has (20-1)*(20-1)*2 = 722 triangles
        // With 124 triangle limit, need at least 6 meshlets
        assert!(
            output.meshlet_count() >= 6,
            "Expected at least 6 meshlets, got {}",
            output.meshlet_count()
        );
    }

    #[test]
    fn low_vertex_reuse_hits_vertex_limit() {
        // Each triangle has unique vertices
        let (positions, indices) = make_large_mesh(100);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        // 100 tris * 3 verts = 300 unique verts
        // With 64 vert limit, ~5+ meshlets
        assert!(
            output.meshlet_count() >= 5,
            "Expected at least 5 meshlets, got {}",
            output.meshlet_count()
        );
    }

    #[test]
    fn maximum_vertex_limit_255() {
        let gen = MeshletGenerator::with_limits(255, 255);
        assert_eq!(gen.max_vertices(), 255);
        assert_eq!(gen.max_triangles(), 255);

        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);
        let output = gen.generate(&input);

        assert_eq!(output.meshlet_count(), 1);
    }

    #[test]
    fn minimum_limits_1_vertex_1_triangle() {
        let gen = MeshletGenerator::with_limits(3, 1);
        assert_eq!(gen.max_vertices(), 3);
        assert_eq!(gen.max_triangles(), 1);

        let (positions, indices) = make_single_triangle();
        let input = MeshInput::new(&positions, &indices);
        let output = gen.generate(&input);

        assert_eq!(output.meshlet_count(), 1);
        assert_eq!(output.meshlets[0].triangle_count, 1);
    }

    #[test]
    fn single_vertex_mesh_empty_output() {
        let positions = vec![[0.0, 0.0, 0.0]];
        let indices: Vec<u32> = vec![];
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert!(output.is_empty());
    }

    #[test]
    fn stress_many_small_meshlets() {
        // Create mesh that forces many tiny meshlets
        let (positions, indices) = make_large_mesh(1000);
        let input = MeshInput::new(&positions, &indices);

        // Very restrictive limits
        let gen = MeshletGenerator::with_limits(6, 2);
        let output = gen.generate(&input);

        // Should create many meshlets
        assert!(output.meshlet_count() > 100);

        // All should be valid
        assert!(output.validate(positions.len()).is_ok());

        // Reconstruction should work
        let reconstructed = output.reconstruct_indices();
        assert_eq!(reconstructed.len(), indices.len());
    }

    #[test]
    fn stress_1000_triangles_validation() {
        let (positions, indices) = make_large_mesh(1000);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        assert!(output.validate(positions.len()).is_ok());

        let stats = output.stats();
        assert_eq!(stats.total_triangles, 1000);
    }

    #[test]
    fn very_large_grid_mesh() {
        // 100x100 grid = ~20,000 triangles
        let (positions, indices) = make_grid_mesh(100);
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        let expected_tris = 99 * 99 * 2;
        let total_tris: usize = output
            .meshlets
            .iter()
            .map(|m| m.triangle_count as usize)
            .sum();

        assert_eq!(total_tris, expected_tris);
        assert!(output.validate(positions.len()).is_ok());
    }

    #[test]
    fn mixed_valid_and_invalid_triangles() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let indices = vec![
            0, 1, 2, // Valid
            0, 0, 1, // Degenerate
            1, 2, 3, // Valid
            0, 1, 99, // Out of bounds
            0, 2, 3, // Valid
        ];
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        // Should have exactly 3 valid triangles
        let total: usize = output
            .meshlets
            .iter()
            .map(|m| m.triangle_count as usize)
            .sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn bounding_sphere_single_vertex_meshlet() {
        // Force single-vertex situation (degenerate case)
        // Actually minimum is 3 vertices for 1 triangle
        let positions = vec![[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.5, 6.0, 5.0]];
        let indices = vec![0, 1, 2];
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        let bounds = &output.bounds[0];

        // Verify all vertices are within bounds
        for &idx in &output.vertex_indices {
            let p = positions[idx as usize];
            let dist = distance(bounds.center, p);
            assert!(dist <= bounds.radius + EPSILON);
        }
    }

    #[test]
    fn normal_cone_single_triangle_is_tight() {
        let (positions, indices) = make_single_triangle();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        // Single triangle should have cutoff of 1.0 (all normals identical)
        assert!(
            (output.bounds[0].cone_cutoff - 1.0).abs() < EPSILON,
            "Single triangle cone cutoff should be 1.0, got {}",
            output.bounds[0].cone_cutoff
        );
    }

    #[test]
    fn cube_normal_cone_spans_many_directions() {
        let (positions, indices) = make_cube();
        let input = MeshInput::new(&positions, &indices);

        let gen = MeshletGenerator::new();
        let output = gen.generate(&input);

        // Cube faces point in 6 directions, so cone should be very wide
        // or potentially invalid for backface culling
        let cutoff = output.bounds[0].cone_cutoff;
        assert!(
            cutoff < 0.0,
            "Cube cone cutoff should be negative (wide cone), got {}",
            cutoff
        );
    }
}

// ===========================================================================
// 6. DETERMINISM AND CONSISTENCY
// ===========================================================================

mod determinism {
    use super::*;

    #[test]
    fn same_input_produces_same_output() {
        let (positions, indices) = make_large_mesh(100);
        let gen = MeshletGenerator::new();

        let input1 = MeshInput::new(&positions, &indices);
        let output1 = gen.generate(&input1);

        let input2 = MeshInput::new(&positions, &indices);
        let output2 = gen.generate(&input2);

        assert_eq!(output1.meshlet_count(), output2.meshlet_count());
        assert_eq!(output1.vertex_indices, output2.vertex_indices);
        assert_eq!(output1.triangle_indices, output2.triangle_indices);
    }

    #[test]
    fn different_generators_same_limits_produce_same_output() {
        let (positions, indices) = make_cube();

        let gen1 = MeshletGenerator::with_limits(64, 124);
        let gen2 = MeshletGenerator::with_limits(64, 124);

        let input = MeshInput::new(&positions, &indices);
        let output1 = gen1.generate(&input);
        let output2 = gen2.generate(&input);

        assert_eq!(output1.meshlet_count(), output2.meshlet_count());
        assert_eq!(output1.vertex_indices, output2.vertex_indices);
    }
}

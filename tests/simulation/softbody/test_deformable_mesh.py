"""Tests for deformable mesh module."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from engine.simulation.softbody.deformable_mesh import (
    EmbeddedVertex,
    EmbeddedSurface,
    TetSkinning,
    DeformableMesh,
    compute_barycentric_coords,
    point_in_tetrahedron,
    compute_triangle_normal,
    interpolate_position,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_tet_vertices():
    """Simple tetrahedron vertices."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


@pytest.fixture
def simple_tet_indices():
    """Single tetrahedron indices."""
    return np.array([[0, 1, 2, 3]], dtype=np.int32)


@pytest.fixture
def cube_mesh():
    """A cube discretized into 5 tetrahedra."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)

    tetrahedra = np.array([
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 4, 5, 6],
        [3, 4, 6, 7],
        [1, 3, 4, 6],
    ], dtype=np.int32)

    return vertices, tetrahedra


@pytest.fixture
def deformable_mesh(simple_tet_vertices, simple_tet_indices):
    """Create a simple deformable mesh."""
    return DeformableMesh(
        tet_vertices=simple_tet_vertices,
        tet_tetrahedra=simple_tet_indices,
    )


# =============================================================================
# Test Helper Functions
# =============================================================================

class TestComputeBarycentricCoords:
    """Test barycentric coordinate computation."""

    def test_vertex_positions(self, simple_tet_vertices):
        """Vertex positions should have weight 1 at that vertex."""
        v0, v1, v2, v3 = simple_tet_vertices

        # At vertex 0
        bary = compute_barycentric_coords(v0, v0, v1, v2, v3)
        assert_array_almost_equal(bary, [1.0, 0.0, 0.0, 0.0])

        # At vertex 1
        bary = compute_barycentric_coords(v1, v0, v1, v2, v3)
        assert_array_almost_equal(bary, [0.0, 1.0, 0.0, 0.0])

        # At vertex 2
        bary = compute_barycentric_coords(v2, v0, v1, v2, v3)
        assert_array_almost_equal(bary, [0.0, 0.0, 1.0, 0.0])

        # At vertex 3
        bary = compute_barycentric_coords(v3, v0, v1, v2, v3)
        assert_array_almost_equal(bary, [0.0, 0.0, 0.0, 1.0])

    def test_centroid(self, simple_tet_vertices):
        """Centroid should have equal weights."""
        v0, v1, v2, v3 = simple_tet_vertices
        centroid = np.mean(simple_tet_vertices, axis=0)
        bary = compute_barycentric_coords(centroid, v0, v1, v2, v3)
        assert_array_almost_equal(bary, [0.25, 0.25, 0.25, 0.25])

    def test_midpoint_edge(self, simple_tet_vertices):
        """Midpoint of edge should have weights 0.5 on both endpoints."""
        v0, v1, v2, v3 = simple_tet_vertices
        midpoint = (v0 + v1) / 2
        bary = compute_barycentric_coords(midpoint, v0, v1, v2, v3)
        assert_array_almost_equal(bary, [0.5, 0.5, 0.0, 0.0])

    def test_sum_to_one(self, simple_tet_vertices):
        """Barycentric coordinates should sum to 1 for point inside."""
        v0, v1, v2, v3 = simple_tet_vertices
        point = np.array([0.1, 0.1, 0.1])
        bary = compute_barycentric_coords(point, v0, v1, v2, v3)
        assert np.isclose(np.sum(bary), 1.0)

    def test_degenerate_tetrahedron(self):
        """Degenerate tetrahedron should return equal weights."""
        # Collinear vertices
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([2.0, 0.0, 0.0])
        v3 = np.array([3.0, 0.0, 0.0])
        point = np.array([0.5, 0.0, 0.0])
        bary = compute_barycentric_coords(point, v0, v1, v2, v3)
        # Should not crash, returns default weights
        assert_array_almost_equal(bary, [0.25, 0.25, 0.25, 0.25])


class TestPointInTetrahedron:
    """Test point-in-tetrahedron test."""

    def test_centroid_inside(self, simple_tet_vertices):
        """Centroid should be inside tetrahedron."""
        v0, v1, v2, v3 = simple_tet_vertices
        centroid = np.mean(simple_tet_vertices, axis=0)
        assert point_in_tetrahedron(centroid, v0, v1, v2, v3)

    def test_vertices_inside(self, simple_tet_vertices):
        """Vertices should be on boundary (inside with epsilon)."""
        v0, v1, v2, v3 = simple_tet_vertices
        assert point_in_tetrahedron(v0, v0, v1, v2, v3)
        assert point_in_tetrahedron(v1, v0, v1, v2, v3)
        assert point_in_tetrahedron(v2, v0, v1, v2, v3)
        assert point_in_tetrahedron(v3, v0, v1, v2, v3)

    def test_point_outside(self, simple_tet_vertices):
        """Point far outside should not be inside."""
        v0, v1, v2, v3 = simple_tet_vertices
        outside = np.array([10.0, 10.0, 10.0])
        assert not point_in_tetrahedron(outside, v0, v1, v2, v3)

    def test_point_negative_region(self, simple_tet_vertices):
        """Point in negative region should be outside."""
        v0, v1, v2, v3 = simple_tet_vertices
        negative = np.array([-0.5, -0.5, -0.5])
        assert not point_in_tetrahedron(negative, v0, v1, v2, v3)


class TestComputeTriangleNormal:
    """Test triangle normal computation."""

    def test_xy_plane_triangle(self):
        """Triangle in XY plane should have normal along Z."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        normal = compute_triangle_normal(v0, v1, v2)
        assert_array_almost_equal(normal, [0.0, 0.0, 1.0])

    def test_xz_plane_triangle(self):
        """Triangle in XZ plane should have normal along Y (negated due to winding)."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 0.0, 1.0])
        normal = compute_triangle_normal(v0, v1, v2)
        # Normal depends on winding
        assert np.isclose(abs(normal[1]), 1.0)

    def test_normalized_result(self):
        """Normal should be unit length."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([3.0, 0.0, 0.0])
        v2 = np.array([0.0, 4.0, 0.0])
        normal = compute_triangle_normal(v0, v1, v2)
        assert np.isclose(np.linalg.norm(normal), 1.0)

    def test_degenerate_triangle(self):
        """Degenerate triangle should return default normal."""
        v0 = np.array([0.0, 0.0, 0.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([2.0, 0.0, 0.0])  # Collinear
        normal = compute_triangle_normal(v0, v1, v2)
        assert_array_almost_equal(normal, [0.0, 1.0, 0.0])


class TestInterpolatePosition:
    """Test position interpolation."""

    def test_vertex_weights(self):
        """Weight 1 on a vertex should return that vertex."""
        tet_verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        # Weight on vertex 1
        bary = np.array([0.0, 1.0, 0.0, 0.0])
        result = interpolate_position(tet_verts, bary)
        assert_array_almost_equal(result, [1.0, 0.0, 0.0])

    def test_centroid_weights(self):
        """Equal weights should return centroid."""
        tet_verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        bary = np.array([0.25, 0.25, 0.25, 0.25])
        result = interpolate_position(tet_verts, bary)
        expected = np.mean(tet_verts, axis=0)
        assert_array_almost_equal(result, expected)


# =============================================================================
# Test EmbeddedVertex
# =============================================================================

class TestEmbeddedVertex:
    """Test embedded vertex data class."""

    def test_construction(self):
        """Should construct properly."""
        vertex = EmbeddedVertex(
            tet_index=0,
            barycentric=np.array([0.25, 0.25, 0.25, 0.25]),
            surface_index=5,
        )
        assert vertex.tet_index == 0
        assert vertex.surface_index == 5


# =============================================================================
# Test EmbeddedSurface
# =============================================================================

class TestEmbeddedSurface:
    """Test embedded surface data class."""

    def test_construction(self):
        """Should construct properly."""
        surface = EmbeddedSurface(
            surface_vertices=np.zeros((4, 3)),
            surface_triangles=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
            surface_normals=np.zeros((4, 3)),
            embeddings=[],
        )
        assert surface.num_vertices == 4
        assert surface.num_triangles == 2


# =============================================================================
# Test DeformableMesh
# =============================================================================

class TestDeformableMesh:
    """Test deformable mesh."""

    def test_construction(self, simple_tet_vertices, simple_tet_indices):
        """Mesh should initialize properly."""
        mesh = DeformableMesh(
            tet_vertices=simple_tet_vertices,
            tet_tetrahedra=simple_tet_indices,
        )
        assert mesh.tet_vertices.shape == simple_tet_vertices.shape
        assert mesh.surface is not None
        assert mesh.surface.num_vertices > 0

    def test_surface_extraction(self, simple_tet_vertices, simple_tet_indices):
        """Surface should be extracted from tet mesh."""
        mesh = DeformableMesh(
            tet_vertices=simple_tet_vertices,
            tet_tetrahedra=simple_tet_indices,
        )
        # Single tet has 4 triangular faces
        assert mesh.surface.num_triangles == 4

    def test_embedded_surface_vertices(self, deformable_mesh):
        """All surface vertices should have embeddings."""
        n_verts = deformable_mesh.surface.num_vertices
        n_embeddings = len(deformable_mesh.surface.embeddings)
        assert n_embeddings == n_verts

    def test_update_surface_positions(self, deformable_mesh):
        """Surface positions should update with tet positions."""
        # Translate tet vertices
        deformable_mesh.tet_vertices += np.array([1.0, 0.0, 0.0])
        deformable_mesh.update_surface_positions()

        # Surface should also be translated
        surface_center = np.mean(deformable_mesh.surface.surface_vertices, axis=0)
        assert surface_center[0] > 0

    def test_update_normals(self, deformable_mesh):
        """Normals should be recomputed after position update."""
        deformable_mesh.update_normals()
        # Normals should be unit length
        for normal in deformable_mesh.surface.surface_normals:
            length = np.linalg.norm(normal)
            assert np.isclose(length, 1.0)

    def test_normals_dirty_flag(self, deformable_mesh):
        """Normals dirty flag should track updates."""
        deformable_mesh.update_normals()
        assert not deformable_mesh.normals_dirty

        deformable_mesh.update_surface_positions()
        assert deformable_mesh.normals_dirty

    def test_get_surface_vertices(self, deformable_mesh):
        """Should return surface vertices."""
        verts = deformable_mesh.get_surface_vertices()
        assert verts.shape[0] == deformable_mesh.surface.num_vertices

    def test_get_surface_normals_recomputes(self, deformable_mesh):
        """Getting normals should recompute if dirty."""
        deformable_mesh.update_surface_positions()
        assert deformable_mesh.normals_dirty

        normals = deformable_mesh.get_surface_normals()
        assert not deformable_mesh.normals_dirty

    def test_get_surface_triangles(self, deformable_mesh):
        """Should return surface triangles."""
        tris = deformable_mesh.get_surface_triangles()
        assert tris.shape[0] == deformable_mesh.surface.num_triangles

    def test_compute_surface_area(self, deformable_mesh):
        """Surface area should be positive."""
        area = deformable_mesh.compute_surface_area()
        assert area > 0

    def test_compute_bounding_box(self, deformable_mesh):
        """Bounding box should contain all vertices."""
        min_corner, max_corner = deformable_mesh.compute_bounding_box()

        for v in deformable_mesh.surface.surface_vertices:
            assert np.all(v >= min_corner - 1e-6)
            assert np.all(v <= max_corner + 1e-6)

    def test_set_tet_vertices(self, deformable_mesh):
        """Setting tet vertices should update surface."""
        original_surface = deformable_mesh.surface.surface_vertices.copy()

        new_verts = deformable_mesh.tet_vertices * 2.0
        deformable_mesh.set_tet_vertices(new_verts)

        # Surface should have changed
        assert not np.allclose(
            deformable_mesh.surface.surface_vertices, original_surface
        )


class TestDeformableMeshCube:
    """Test deformable mesh with cube."""

    @pytest.fixture
    def cube_deformable(self, cube_mesh):
        vertices, tetrahedra = cube_mesh
        return DeformableMesh(
            tet_vertices=vertices,
            tet_tetrahedra=tetrahedra,
        )

    def test_surface_extraction_cube(self, cube_deformable):
        """Cube surface should have 12 triangles (2 per face)."""
        assert cube_deformable.surface.num_triangles == 12

    def test_surface_area_cube(self, cube_deformable):
        """Cube surface area should be 6 (1 per face)."""
        area = cube_deformable.compute_surface_area()
        assert np.isclose(area, 6.0, atol=0.1)

    def test_bounding_box_cube(self, cube_deformable):
        """Cube bounding box should be unit cube."""
        min_corner, max_corner = cube_deformable.compute_bounding_box()
        assert_array_almost_equal(min_corner, [0.0, 0.0, 0.0], decimal=5)
        assert_array_almost_equal(max_corner, [1.0, 1.0, 1.0], decimal=5)


class TestDeformableMeshSkinning:
    """Test skinning weight generation."""

    def test_skinning_from_tets(self, deformable_mesh):
        """Skinning data should be generated."""
        skinning = deformable_mesh.skinning_from_tets()

        assert skinning.surface_to_tet.shape[0] == deformable_mesh.surface.num_vertices
        assert skinning.weights.shape[0] == deformable_mesh.surface.num_vertices
        assert len(skinning.influence_tets) == deformable_mesh.surface.num_vertices

    def test_skinning_weights_sum_to_one(self, deformable_mesh):
        """Skinning weights should sum to 1."""
        skinning = deformable_mesh.skinning_from_tets()

        for i in range(skinning.weights.shape[0]):
            assert np.isclose(np.sum(skinning.weights[i]), 1.0)


class TestDeformableMeshCustomSurface:
    """Test with custom surface mesh."""

    def test_custom_surface_embedding(self, simple_tet_vertices, simple_tet_indices):
        """Custom surface should be embedded in tet mesh."""
        # Custom surface: just a single triangle
        surface_vertices = np.array([
            [0.1, 0.1, 0.1],
            [0.3, 0.1, 0.1],
            [0.1, 0.3, 0.1],
        ], dtype=np.float64)
        surface_triangles = np.array([[0, 1, 2]], dtype=np.int32)

        mesh = DeformableMesh(
            tet_vertices=simple_tet_vertices,
            tet_tetrahedra=simple_tet_indices,
            surface_vertices=surface_vertices,
            surface_triangles=surface_triangles,
        )

        assert mesh.surface.num_vertices == 3
        assert mesh.surface.num_triangles == 1
        assert len(mesh.surface.embeddings) == 3

    def test_surface_outside_tet_mesh(self, simple_tet_vertices, simple_tet_indices):
        """Surface vertices outside mesh should be handled."""
        # Surface far outside tet
        surface_vertices = np.array([
            [10.0, 10.0, 10.0],
            [11.0, 10.0, 10.0],
            [10.0, 11.0, 10.0],
        ], dtype=np.float64)
        surface_triangles = np.array([[0, 1, 2]], dtype=np.int32)

        # Should not crash - uses nearest tet
        mesh = DeformableMesh(
            tet_vertices=simple_tet_vertices,
            tet_tetrahedra=simple_tet_indices,
            surface_vertices=surface_vertices,
            surface_triangles=surface_triangles,
        )

        assert mesh.surface.num_vertices == 3


class TestDeformableMeshEdgeCases:
    """Test edge cases."""

    def test_single_tet_all_faces_exposed(self, simple_tet_vertices, simple_tet_indices):
        """Single tet should have all 4 faces as surface."""
        mesh = DeformableMesh(
            tet_vertices=simple_tet_vertices,
            tet_tetrahedra=simple_tet_indices,
        )
        assert mesh.surface.num_triangles == 4

    def test_deformation_tracking(self, simple_tet_vertices, simple_tet_indices):
        """Surface should track tet mesh deformation."""
        mesh = DeformableMesh(
            tet_vertices=simple_tet_vertices,
            tet_tetrahedra=simple_tet_indices,
        )

        original_area = mesh.compute_surface_area()

        # Scale tet mesh
        mesh.tet_vertices *= 2.0
        mesh.update_surface_positions()

        new_area = mesh.compute_surface_area()
        # Area should scale by 4 (2^2)
        assert np.isclose(new_area / original_area, 4.0, rtol=0.1)

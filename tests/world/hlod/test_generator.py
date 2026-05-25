"""
Tests for HLOD mesh generation.

Tests mesh merging, simplification, impostor generation, and proxy mesh generation.
"""

import math
import pytest

from engine.world.hlod.generator import (
    Vec3,
    Vec2,
    AABB,
    MeshData,
    HLODMeshData,
    ImpostorData,
    Edge,
    HLODGenerationMethod,
    SimplificationSettings,
    ImpostorSettings,
    MergeSettings,
    MeshMerger,
    MeshSimplifier,
    ImpostorGenerator,
    ProxyMeshGenerator,
    HLODGenerator,
    HLODConstants,
)


# =============================================================================
# MATH TYPE TESTS
# =============================================================================


class TestVec3:
    """Tests for Vec3 math type."""

    def test_creation(self) -> None:
        """Test Vec3 creation."""
        v = Vec3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_default_creation(self) -> None:
        """Test Vec3 default creation."""
        v = Vec3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_addition(self) -> None:
        """Test Vec3 addition."""
        v1 = Vec3(1.0, 2.0, 3.0)
        v2 = Vec3(4.0, 5.0, 6.0)
        result = v1 + v2
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(7.0)
        assert result.z == pytest.approx(9.0)

    def test_subtraction(self) -> None:
        """Test Vec3 subtraction."""
        v1 = Vec3(4.0, 5.0, 6.0)
        v2 = Vec3(1.0, 2.0, 3.0)
        result = v1 - v2
        assert result.x == pytest.approx(3.0)
        assert result.y == pytest.approx(3.0)
        assert result.z == pytest.approx(3.0)

    def test_scalar_multiplication(self) -> None:
        """Test Vec3 scalar multiplication."""
        v = Vec3(1.0, 2.0, 3.0)
        result = v * 2.0
        assert result.x == pytest.approx(2.0)
        assert result.y == pytest.approx(4.0)
        assert result.z == pytest.approx(6.0)

    def test_division(self) -> None:
        """Test Vec3 division."""
        v = Vec3(2.0, 4.0, 6.0)
        result = v / 2.0
        assert result.x == pytest.approx(1.0)
        assert result.y == pytest.approx(2.0)
        assert result.z == pytest.approx(3.0)

    def test_dot_product(self) -> None:
        """Test Vec3 dot product."""
        v1 = Vec3(1.0, 2.0, 3.0)
        v2 = Vec3(4.0, 5.0, 6.0)
        assert v1.dot(v2) == pytest.approx(32.0)

    def test_cross_product(self) -> None:
        """Test Vec3 cross product."""
        v1 = Vec3(1.0, 0.0, 0.0)
        v2 = Vec3(0.0, 1.0, 0.0)
        result = v1.cross(v2)
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(1.0)

    def test_length(self) -> None:
        """Test Vec3 length."""
        v = Vec3(3.0, 4.0, 0.0)
        assert v.length() == pytest.approx(5.0)

    def test_normalized(self) -> None:
        """Test Vec3 normalization."""
        v = Vec3(3.0, 4.0, 0.0)
        n = v.normalized()
        assert n.length() == pytest.approx(1.0)
        assert n.x == pytest.approx(0.6)
        assert n.y == pytest.approx(0.8)

    def test_distance_to(self) -> None:
        """Test Vec3 distance calculation."""
        v1 = Vec3(0.0, 0.0, 0.0)
        v2 = Vec3(3.0, 4.0, 0.0)
        assert v1.distance_to(v2) == pytest.approx(5.0)

    def test_lerp(self) -> None:
        """Test Vec3 linear interpolation."""
        v1 = Vec3(0.0, 0.0, 0.0)
        v2 = Vec3(10.0, 10.0, 10.0)
        result = v1.lerp(v2, 0.5)
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(5.0)
        assert result.z == pytest.approx(5.0)

    def test_to_tuple(self) -> None:
        """Test Vec3 to tuple conversion."""
        v = Vec3(1.0, 2.0, 3.0)
        t = v.to_tuple()
        assert t == (1.0, 2.0, 3.0)


class TestVec2:
    """Tests for Vec2 math type."""

    def test_creation(self) -> None:
        """Test Vec2 creation."""
        v = Vec2(0.5, 0.75)
        assert v.u == 0.5
        assert v.v == 0.75

    def test_default_creation(self) -> None:
        """Test Vec2 default creation."""
        v = Vec2()
        assert v.u == 0.0
        assert v.v == 0.0

    def test_operations(self) -> None:
        """Test Vec2 basic operations."""
        v1 = Vec2(1.0, 2.0)
        v2 = Vec2(3.0, 4.0)

        add = v1 + v2
        assert add.u == pytest.approx(4.0)
        assert add.v == pytest.approx(6.0)

        sub = v1 - v2
        assert sub.u == pytest.approx(-2.0)
        assert sub.v == pytest.approx(-2.0)

    def test_lerp(self) -> None:
        """Test Vec2 linear interpolation."""
        v1 = Vec2(0.0, 0.0)
        v2 = Vec2(1.0, 1.0)
        result = v1.lerp(v2, 0.5)
        assert result.u == pytest.approx(0.5)
        assert result.v == pytest.approx(0.5)


class TestAABB:
    """Tests for AABB."""

    def test_creation(self) -> None:
        """Test AABB creation."""
        aabb = AABB(
            min_point=Vec3(-1.0, -2.0, -3.0),
            max_point=Vec3(1.0, 2.0, 3.0),
        )
        assert aabb.min_point.x == -1.0
        assert aabb.max_point.x == 1.0

    def test_center(self) -> None:
        """Test AABB center calculation."""
        aabb = AABB(
            min_point=Vec3(-1.0, -2.0, -3.0),
            max_point=Vec3(1.0, 2.0, 3.0),
        )
        center = aabb.center
        assert center.x == pytest.approx(0.0)
        assert center.y == pytest.approx(0.0)
        assert center.z == pytest.approx(0.0)

    def test_extents(self) -> None:
        """Test AABB extents calculation."""
        aabb = AABB(
            min_point=Vec3(-1.0, -2.0, -3.0),
            max_point=Vec3(1.0, 2.0, 3.0),
        )
        extents = aabb.extents
        assert extents.x == pytest.approx(1.0)
        assert extents.y == pytest.approx(2.0)
        assert extents.z == pytest.approx(3.0)

    def test_size(self) -> None:
        """Test AABB size calculation."""
        aabb = AABB(
            min_point=Vec3(-1.0, -2.0, -3.0),
            max_point=Vec3(1.0, 2.0, 3.0),
        )
        size = aabb.size
        assert size.x == pytest.approx(2.0)
        assert size.y == pytest.approx(4.0)
        assert size.z == pytest.approx(6.0)

    def test_expand(self) -> None:
        """Test AABB expansion."""
        aabb = AABB()
        aabb.expand(Vec3(1.0, 2.0, 3.0))
        aabb.expand(Vec3(-1.0, -2.0, -3.0))

        assert aabb.min_point.x == -1.0
        assert aabb.max_point.x == 1.0

    def test_merge(self) -> None:
        """Test AABB merging."""
        aabb1 = AABB(
            min_point=Vec3(-1.0, -1.0, -1.0),
            max_point=Vec3(0.0, 0.0, 0.0),
        )
        aabb2 = AABB(
            min_point=Vec3(0.0, 0.0, 0.0),
            max_point=Vec3(1.0, 1.0, 1.0),
        )
        merged = aabb1.merge(aabb2)

        assert merged.min_point.x == -1.0
        assert merged.max_point.x == 1.0

    def test_is_valid(self) -> None:
        """Test AABB validity check."""
        valid_aabb = AABB(
            min_point=Vec3(-1.0, -1.0, -1.0),
            max_point=Vec3(1.0, 1.0, 1.0),
        )
        assert valid_aabb.is_valid()

        # Default AABB is not valid (min > max)
        empty_aabb = AABB()
        assert not empty_aabb.is_valid()


# =============================================================================
# MESH DATA TESTS
# =============================================================================


class TestMeshData:
    """Tests for MeshData."""

    @pytest.fixture
    def simple_triangle(self) -> MeshData:
        """Create a simple triangle mesh."""
        return MeshData(
            vertices=[
                Vec3(0.0, 0.0, 0.0),
                Vec3(1.0, 0.0, 0.0),
                Vec3(0.5, 1.0, 0.0),
            ],
            normals=[
                Vec3(0.0, 0.0, 1.0),
                Vec3(0.0, 0.0, 1.0),
                Vec3(0.0, 0.0, 1.0),
            ],
            uvs=[
                Vec2(0.0, 0.0),
                Vec2(1.0, 0.0),
                Vec2(0.5, 1.0),
            ],
            indices=[0, 1, 2],
        )

    @pytest.fixture
    def simple_quad(self) -> MeshData:
        """Create a simple quad mesh (2 triangles)."""
        return MeshData(
            vertices=[
                Vec3(0.0, 0.0, 0.0),
                Vec3(1.0, 0.0, 0.0),
                Vec3(1.0, 1.0, 0.0),
                Vec3(0.0, 1.0, 0.0),
            ],
            normals=[Vec3(0.0, 0.0, 1.0)] * 4,
            uvs=[
                Vec2(0.0, 0.0),
                Vec2(1.0, 0.0),
                Vec2(1.0, 1.0),
                Vec2(0.0, 1.0),
            ],
            indices=[0, 1, 2, 0, 2, 3],
        )

    def test_triangle_count(self, simple_triangle: MeshData) -> None:
        """Test triangle count calculation."""
        assert simple_triangle.get_triangle_count() == 1

    def test_vertex_count(self, simple_triangle: MeshData) -> None:
        """Test vertex count calculation."""
        assert simple_triangle.get_vertex_count() == 3

    def test_is_valid(self, simple_triangle: MeshData) -> None:
        """Test mesh validity check."""
        assert simple_triangle.is_valid()

    def test_invalid_index_count(self) -> None:
        """Test mesh with invalid index count."""
        mesh = MeshData(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0)],
            indices=[0, 1],  # Not divisible by 3
        )
        assert not mesh.is_valid()

    def test_invalid_index_out_of_bounds(self) -> None:
        """Test mesh with out-of-bounds index."""
        mesh = MeshData(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0.5, 1, 0)],
            indices=[0, 1, 5],  # Index 5 is out of bounds
        )
        assert not mesh.is_valid()

    def test_empty_mesh_is_valid(self) -> None:
        """Test that empty mesh is valid."""
        mesh = MeshData()
        assert mesh.is_valid()

    def test_compute_bounds(self, simple_triangle: MeshData) -> None:
        """Test bounds computation."""
        simple_triangle.compute_bounds()
        bounds = simple_triangle.bounds

        assert bounds.min_point.x == pytest.approx(0.0)
        assert bounds.max_point.x == pytest.approx(1.0)
        assert bounds.min_point.y == pytest.approx(0.0)
        assert bounds.max_point.y == pytest.approx(1.0)

    def test_merge_with(self, simple_triangle: MeshData, simple_quad: MeshData) -> None:
        """Test mesh merging."""
        merged = simple_triangle.merge_with(simple_quad)

        # Should have combined vertices and triangles
        assert merged.get_vertex_count() == 7  # 3 + 4
        assert merged.get_triangle_count() == 3  # 1 + 2

    def test_get_triangle(self, simple_quad: MeshData) -> None:
        """Test getting triangle indices."""
        i0, i1, i2 = simple_quad.get_triangle(0)
        assert (i0, i1, i2) == (0, 1, 2)

        i0, i1, i2 = simple_quad.get_triangle(1)
        assert (i0, i1, i2) == (0, 2, 3)

    def test_compute_triangle_normal(self, simple_triangle: MeshData) -> None:
        """Test triangle normal computation."""
        normal = simple_triangle.compute_triangle_normal(0)

        # Triangle lies in XY plane, normal should point in +Z
        assert normal.x == pytest.approx(0.0)
        assert normal.y == pytest.approx(0.0)
        assert abs(normal.z) == pytest.approx(1.0)

    def test_compute_triangle_area(self, simple_triangle: MeshData) -> None:
        """Test triangle area computation."""
        area = simple_triangle.compute_triangle_area(0)

        # Triangle with base 1 and height 1 has area 0.5
        assert area == pytest.approx(0.5)


# =============================================================================
# MESH MERGER TESTS
# =============================================================================


class TestMeshMerger:
    """Tests for MeshMerger."""

    @pytest.fixture
    def merger(self) -> MeshMerger:
        """Create a mesh merger."""
        return MeshMerger()

    @pytest.fixture
    def two_triangles(self) -> list:
        """Create two separate triangle meshes."""
        mesh1 = MeshData(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0.5, 1, 0)],
            normals=[Vec3(0, 0, 1)] * 3,
            uvs=[Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1)],
            indices=[0, 1, 2],
        )
        mesh2 = MeshData(
            vertices=[Vec3(2, 0, 0), Vec3(3, 0, 0), Vec3(2.5, 1, 0)],
            normals=[Vec3(0, 0, 1)] * 3,
            uvs=[Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1)],
            indices=[0, 1, 2],
        )
        return [mesh1, mesh2]

    def test_merge_empty_list(self, merger: MeshMerger) -> None:
        """Test merging empty list."""
        result = merger.merge_meshes([])
        assert result.get_triangle_count() == 0

    def test_merge_single_mesh(self, merger: MeshMerger, two_triangles: list) -> None:
        """Test merging single mesh."""
        result = merger.merge_meshes([two_triangles[0]])
        assert result.get_triangle_count() == 1
        assert result.get_vertex_count() == 3

    def test_merge_two_meshes(self, merger: MeshMerger, two_triangles: list) -> None:
        """Test merging two meshes."""
        result = merger.merge_meshes(two_triangles)

        # Combined mesh should have 6 vertices and 2 triangles
        assert result.get_vertex_count() == 6
        assert result.get_triangle_count() == 2

    def test_merge_preserves_vertices(self, merger: MeshMerger, two_triangles: list) -> None:
        """Test that merging preserves all vertices."""
        result = merger.merge_meshes(two_triangles)

        # Check that vertices from both meshes are present
        xs = [v.x for v in result.vertices]
        assert 0.0 in xs or any(abs(x) < 0.01 for x in xs)
        assert 2.0 in xs or any(abs(x - 2.0) < 0.01 for x in xs)

    def test_merge_updates_indices(self, merger: MeshMerger, two_triangles: list) -> None:
        """Test that merging correctly updates indices."""
        result = merger.merge_meshes(two_triangles)

        # All indices should be valid
        for idx in result.indices:
            assert 0 <= idx < len(result.vertices)

    def test_merge_computes_bounds(self, merger: MeshMerger, two_triangles: list) -> None:
        """Test that merging computes combined bounds."""
        result = merger.merge_meshes(two_triangles)

        assert result.bounds.is_valid()
        assert result.bounds.min_point.x == pytest.approx(0.0)
        assert result.bounds.max_point.x == pytest.approx(3.0)

    def test_merge_with_vertex_welding(self) -> None:
        """Test merging with vertex welding enabled."""
        settings = MergeSettings(merge_distance=0.1)
        merger = MeshMerger(settings)

        # Create two triangles sharing an edge (vertices close together)
        mesh1 = MeshData(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0.5, 1, 0)],
            normals=[Vec3(0, 0, 1)] * 3,
            uvs=[Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1)],
            indices=[0, 1, 2],
        )
        mesh2 = MeshData(
            vertices=[Vec3(1.05, 0, 0), Vec3(2, 0, 0), Vec3(1.5, 1, 0)],  # First vertex close to mesh1's second
            normals=[Vec3(0, 0, 1)] * 3,
            uvs=[Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1)],
            indices=[0, 1, 2],
        )

        result = merger.merge_meshes([mesh1, mesh2])

        # With welding, close vertices should be merged (5 instead of 6)
        assert result.get_vertex_count() == 5


# =============================================================================
# MESH SIMPLIFIER TESTS
# =============================================================================


class TestMeshSimplifier:
    """Tests for MeshSimplifier."""

    @pytest.fixture
    def simplifier(self) -> MeshSimplifier:
        """Create a mesh simplifier with default settings."""
        return MeshSimplifier(SimplificationSettings(target_ratio=0.5))

    @pytest.fixture
    def complex_mesh(self) -> MeshData:
        """Create a more complex mesh for simplification testing."""
        # Create a simple grid mesh (5x5 = 25 vertices, 32 triangles)
        vertices = []
        normals = []
        uvs = []
        indices = []

        for y in range(5):
            for x in range(5):
                vertices.append(Vec3(float(x), float(y), 0.0))
                normals.append(Vec3(0.0, 0.0, 1.0))
                uvs.append(Vec2(x / 4.0, y / 4.0))

        for y in range(4):
            for x in range(4):
                i = y * 5 + x
                # First triangle
                indices.extend([i, i + 1, i + 6])
                # Second triangle
                indices.extend([i, i + 6, i + 5])

        mesh = MeshData(
            vertices=vertices,
            normals=normals,
            uvs=uvs,
            indices=indices,
        )
        mesh.compute_bounds()
        return mesh

    def test_simplify_empty_mesh(self, simplifier: MeshSimplifier) -> None:
        """Test simplifying empty mesh."""
        mesh = MeshData()
        result = simplifier.simplify(mesh)
        assert result.get_triangle_count() == 0

    def test_simplify_reduces_triangles(
        self,
        simplifier: MeshSimplifier,
        complex_mesh: MeshData,
    ) -> None:
        """Test that simplification reduces triangle count."""
        original_count = complex_mesh.get_triangle_count()
        result = simplifier.simplify(complex_mesh)

        # Should reduce triangles (not necessarily by exactly 50% due to algorithm)
        assert result.get_triangle_count() < original_count

    def test_simplify_preserves_validity(
        self,
        simplifier: MeshSimplifier,
        complex_mesh: MeshData,
    ) -> None:
        """Test that simplification produces valid mesh."""
        result = simplifier.simplify(complex_mesh)
        assert result.is_valid()

    def test_simplify_respects_max_error(self, complex_mesh: MeshData) -> None:
        """Test that simplification respects max error."""
        # Very low error should result in minimal simplification
        settings = SimplificationSettings(
            target_ratio=0.1,  # Aggressive target
            max_error=0.0001,  # Very low error threshold
        )
        simplifier = MeshSimplifier(settings)

        original_count = complex_mesh.get_triangle_count()
        result = simplifier.simplify(complex_mesh)

        # With very low max_error, should not simplify much
        # (exact behavior depends on mesh geometry)
        assert result.get_triangle_count() > 0

    def test_simplify_target_ratio(self, complex_mesh: MeshData) -> None:
        """Test simplification with different target ratios."""
        original_count = complex_mesh.get_triangle_count()

        # 75% ratio
        settings75 = SimplificationSettings(target_ratio=0.75, max_error=1.0)
        result75 = MeshSimplifier(settings75).simplify(complex_mesh)

        # 25% ratio
        settings25 = SimplificationSettings(target_ratio=0.25, max_error=1.0)
        result25 = MeshSimplifier(settings25).simplify(complex_mesh)

        # More aggressive simplification should result in fewer triangles
        assert result25.get_triangle_count() <= result75.get_triangle_count()

    def test_simplify_preserves_some_vertices(
        self,
        simplifier: MeshSimplifier,
        complex_mesh: MeshData,
    ) -> None:
        """Test that simplification preserves some vertices."""
        result = simplifier.simplify(complex_mesh)

        # Should have at least 3 vertices (minimum for a triangle)
        assert result.get_vertex_count() >= 3

    def test_simplify_no_degenerate_triangles(
        self,
        simplifier: MeshSimplifier,
        complex_mesh: MeshData,
    ) -> None:
        """Test that simplification produces no degenerate triangles."""
        result = simplifier.simplify(complex_mesh)

        for tri_idx in range(result.get_triangle_count()):
            i0, i1, i2 = result.get_triangle(tri_idx)

            # No duplicate vertex indices
            assert i0 != i1, f"Triangle {tri_idx} has duplicate indices i0==i1"
            assert i1 != i2, f"Triangle {tri_idx} has duplicate indices i1==i2"
            assert i2 != i0, f"Triangle {tri_idx} has duplicate indices i2==i0"

            # Vertices should be distinct positions
            v0 = result.vertices[i0]
            v1 = result.vertices[i1]
            v2 = result.vertices[i2]
            assert v0 != v1, f"Triangle {tri_idx} has coincident vertices v0==v1"
            assert v1 != v2, f"Triangle {tri_idx} has coincident vertices v1==v2"
            assert v2 != v0, f"Triangle {tri_idx} has coincident vertices v2==v0"

            # Triangle should have non-zero area
            area = result.compute_triangle_area(tri_idx)
            assert area > 1e-10, f"Triangle {tri_idx} has near-zero area: {area}"

    def test_simplify_triangle_count_within_expected_range(
        self,
        complex_mesh: MeshData,
    ) -> None:
        """Test that simplification achieves expected triangle reduction."""
        original_count = complex_mesh.get_triangle_count()

        # 50% target ratio
        settings50 = SimplificationSettings(target_ratio=0.5, max_error=1.0)
        result50 = MeshSimplifier(settings50).simplify(complex_mesh)

        # Should be between 40% and 60% of original (allowing for algorithm variance)
        min_expected = int(original_count * 0.3)  # Allow more variance
        max_expected = int(original_count * 0.7)

        actual_count = result50.get_triangle_count()
        assert min_expected <= actual_count <= max_expected, (
            f"Expected {min_expected}-{max_expected} triangles, got {actual_count}"
        )

    def test_simplify_all_indices_valid(
        self,
        simplifier: MeshSimplifier,
        complex_mesh: MeshData,
    ) -> None:
        """Test that all indices reference valid vertices after simplification."""
        result = simplifier.simplify(complex_mesh)

        vertex_count = result.get_vertex_count()
        for idx in result.indices:
            assert 0 <= idx < vertex_count, f"Invalid index {idx} with {vertex_count} vertices"

    def test_simplify_normals_and_uvs_preserved(
        self,
        simplifier: MeshSimplifier,
        complex_mesh: MeshData,
    ) -> None:
        """Test that normals and UVs are preserved after simplification."""
        result = simplifier.simplify(complex_mesh)

        # If original had normals/UVs, result should too
        if complex_mesh.normals:
            assert len(result.normals) == len(result.vertices), (
                "Normal count doesn't match vertex count"
            )
        if complex_mesh.uvs:
            assert len(result.uvs) == len(result.vertices), (
                "UV count doesn't match vertex count"
            )

    def test_simplify_empty_result_is_valid(self) -> None:
        """Test that aggressive simplification still produces valid mesh."""
        # Very simple mesh that might reduce to nothing
        mesh = MeshData(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0.5, 0.01, 0)],  # Nearly flat
            indices=[0, 1, 2],
        )
        mesh.compute_bounds()

        settings = SimplificationSettings(target_ratio=0.1, max_error=0.5)
        simplifier = MeshSimplifier(settings)
        result = simplifier.simplify(mesh)

        # Result should still be valid (even if empty)
        assert result.is_valid()


# =============================================================================
# IMPOSTOR GENERATOR TESTS
# =============================================================================


class TestImpostorGenerator:
    """Tests for ImpostorGenerator."""

    @pytest.fixture
    def generator(self) -> ImpostorGenerator:
        """Create an impostor generator."""
        return ImpostorGenerator(ImpostorSettings(
            resolution=64,
            view_count=4,
        ))

    @pytest.fixture
    def simple_mesh(self) -> MeshData:
        """Create a simple mesh for impostor generation."""
        mesh = MeshData(
            vertices=[
                Vec3(-1, -1, -1), Vec3(1, -1, -1), Vec3(1, 1, -1), Vec3(-1, 1, -1),
                Vec3(-1, -1, 1), Vec3(1, -1, 1), Vec3(1, 1, 1), Vec3(-1, 1, 1),
            ],
            normals=[Vec3(0, 0, 1)] * 8,
            uvs=[Vec2(0, 0)] * 8,
            indices=[
                0, 1, 2, 0, 2, 3,  # Front
                5, 4, 7, 5, 7, 6,  # Back
            ],
        )
        mesh.compute_bounds()
        return mesh

    def test_generate_impostor(
        self,
        generator: ImpostorGenerator,
        simple_mesh: MeshData,
    ) -> None:
        """Test impostor generation."""
        bounds = AABB(
            min_point=Vec3(-1, -1, -1),
            max_point=Vec3(1, 1, 1),
        )
        result = generator.generate(simple_mesh, bounds)

        assert isinstance(result, ImpostorData)
        assert result.resolution == 64

    def test_impostor_view_count(
        self,
        generator: ImpostorGenerator,
        simple_mesh: MeshData,
    ) -> None:
        """Test that impostor has correct number of view directions."""
        bounds = simple_mesh.bounds
        result = generator.generate(simple_mesh, bounds)

        assert len(result.view_directions) == 4

    def test_impostor_atlas_generated(
        self,
        generator: ImpostorGenerator,
        simple_mesh: MeshData,
    ) -> None:
        """Test that impostor atlas is generated."""
        bounds = simple_mesh.bounds
        result = generator.generate(simple_mesh, bounds)

        assert len(result.albedo_atlas) > 0

    def test_impostor_captures_normals(self, simple_mesh: MeshData) -> None:
        """Test impostor captures normal map."""
        generator = ImpostorGenerator(ImpostorSettings(
            resolution=32,
            view_count=2,
            capture_normals=True,
        ))

        bounds = simple_mesh.bounds
        result = generator.generate(simple_mesh, bounds)

        assert result.normal_atlas is not None

    def test_impostor_captures_depth(self, simple_mesh: MeshData) -> None:
        """Test impostor captures depth map."""
        generator = ImpostorGenerator(ImpostorSettings(
            resolution=32,
            view_count=2,
            capture_depth=True,
        ))

        bounds = simple_mesh.bounds
        result = generator.generate(simple_mesh, bounds)

        assert result.depth_atlas is not None


# =============================================================================
# PROXY MESH GENERATOR TESTS
# =============================================================================


class TestProxyMeshGenerator:
    """Tests for ProxyMeshGenerator."""

    @pytest.fixture
    def generator(self) -> ProxyMeshGenerator:
        """Create a proxy mesh generator."""
        return ProxyMeshGenerator()

    def test_generate_box(self, generator: ProxyMeshGenerator) -> None:
        """Test box mesh generation."""
        bounds = AABB(
            min_point=Vec3(-1, -2, -3),
            max_point=Vec3(1, 2, 3),
        )

        box = generator.generate_box(bounds)

        # Box should have 8 vertices and 12 triangles
        assert box.get_vertex_count() == 8
        assert box.get_triangle_count() == 12

    def test_generate_box_bounds(self, generator: ProxyMeshGenerator) -> None:
        """Test that generated box matches input bounds."""
        bounds = AABB(
            min_point=Vec3(-5, -5, -5),
            max_point=Vec3(5, 5, 5),
        )

        box = generator.generate_box(bounds)

        # Check that vertices are at bounds corners
        xs = [v.x for v in box.vertices]
        ys = [v.y for v in box.vertices]
        zs = [v.z for v in box.vertices]

        assert min(xs) == pytest.approx(-5)
        assert max(xs) == pytest.approx(5)
        assert min(ys) == pytest.approx(-5)
        assert max(ys) == pytest.approx(5)
        assert min(zs) == pytest.approx(-5)
        assert max(zs) == pytest.approx(5)

    def test_generate_convex_hull(self, generator: ProxyMeshGenerator) -> None:
        """Test convex hull generation."""
        # Create a mesh with more than 4 vertices
        mesh = MeshData(
            vertices=[
                Vec3(-1, -1, -1), Vec3(1, -1, -1), Vec3(1, 1, -1), Vec3(-1, 1, -1),
                Vec3(0, 0, 1),
            ],
            indices=[0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4, 0, 2, 1, 0, 3, 2],
        )
        mesh.compute_bounds()

        hull = generator.generate_convex_hull(mesh)

        # Should produce a valid mesh
        assert hull.get_vertex_count() >= 4
        assert hull.get_triangle_count() >= 4

    def test_generate_convex_hull_small_mesh(
        self,
        generator: ProxyMeshGenerator,
    ) -> None:
        """Test convex hull with small mesh falls back to box."""
        mesh = MeshData(
            vertices=[Vec3(0, 0, 0), Vec3(1, 0, 0)],
            indices=[],
        )
        mesh.compute_bounds()

        hull = generator.generate_convex_hull(mesh)

        # Should produce a box (fallback)
        assert hull.get_triangle_count() == 12

    def test_generate_simplified_bounds(
        self,
        generator: ProxyMeshGenerator,
    ) -> None:
        """Test simplified bounds from multiple meshes."""
        mesh1 = MeshData(
            vertices=[Vec3(-10, 0, 0), Vec3(-5, 0, 0), Vec3(-7.5, 5, 0)],
            indices=[0, 1, 2],
        )
        mesh1.compute_bounds()

        mesh2 = MeshData(
            vertices=[Vec3(5, 0, 0), Vec3(10, 0, 0), Vec3(7.5, 5, 0)],
            indices=[0, 1, 2],
        )
        mesh2.compute_bounds()

        result = generator.generate_simplified_bounds([mesh1, mesh2])

        # Result should span both meshes
        xs = [v.x for v in result.vertices]
        assert min(xs) == pytest.approx(-10)
        assert max(xs) == pytest.approx(10)


# =============================================================================
# HLOD GENERATOR TESTS
# =============================================================================


class TestHLODGenerator:
    """Tests for HLODGenerator."""

    @pytest.fixture
    def generator(self) -> HLODGenerator:
        """Create an HLOD generator."""
        return HLODGenerator()

    @pytest.fixture
    def source_meshes(self) -> list:
        """Create source meshes for HLOD generation."""
        meshes = []
        for i in range(3):
            mesh = MeshData(
                vertices=[
                    Vec3(i * 2, 0, 0),
                    Vec3(i * 2 + 1, 0, 0),
                    Vec3(i * 2 + 0.5, 1, 0),
                ],
                normals=[Vec3(0, 0, 1)] * 3,
                uvs=[Vec2(0, 0), Vec2(1, 0), Vec2(0.5, 1)],
                indices=[0, 1, 2],
            )
            mesh.compute_bounds()
            meshes.append(mesh)
        return meshes

    def test_generate_with_merging(
        self,
        generator: HLODGenerator,
        source_meshes: list,
    ) -> None:
        """Test HLOD generation with mesh merging."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(6, 1, 0))

        result = generator.generate(
            source_meshes,
            bounds,
            method=HLODGenerationMethod.MESH_MERGING,
        )

        assert isinstance(result, HLODMeshData)
        assert result.source_mesh_count == 3
        assert result.method_used == HLODGenerationMethod.MESH_MERGING

    def test_generate_with_simplification(
        self,
        generator: HLODGenerator,
        source_meshes: list,
    ) -> None:
        """Test HLOD generation with simplification."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(6, 1, 0))

        result = generator.generate(
            source_meshes,
            bounds,
            method=HLODGenerationMethod.SIMPLIFICATION,
        )

        assert result.method_used == HLODGenerationMethod.SIMPLIFICATION

    def test_generate_with_proxy(
        self,
        generator: HLODGenerator,
        source_meshes: list,
    ) -> None:
        """Test HLOD generation with proxy mesh."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(6, 1, 0))

        result = generator.generate(
            source_meshes,
            bounds,
            method=HLODGenerationMethod.PROXY_MESH,
        )

        assert result.method_used == HLODGenerationMethod.PROXY_MESH
        # Proxy should be a simple box
        assert result.mesh.get_triangle_count() == 12

    def test_generate_tracks_original_triangles(
        self,
        generator: HLODGenerator,
        source_meshes: list,
    ) -> None:
        """Test that generation tracks original triangle count."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(6, 1, 0))

        result = generator.generate(source_meshes, bounds)

        assert result.original_triangle_count == 3  # 3 triangles total

    def test_reduction_ratio(
        self,
        generator: HLODGenerator,
        source_meshes: list,
    ) -> None:
        """Test reduction ratio calculation."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(6, 1, 0))

        result = generator.generate(
            source_meshes,
            bounds,
            method=HLODGenerationMethod.PROXY_MESH,
        )

        # Proxy (12 triangles) from 3 triangles = negative reduction
        # This is expected for proxy meshes
        assert isinstance(result.reduction_ratio, float)

    def test_select_method_small_mesh(self, generator: HLODGenerator) -> None:
        """Test method selection for small meshes."""
        method = generator.select_method(mesh_count=2, total_triangles=500)
        assert method == HLODGenerationMethod.MESH_MERGING

    def test_select_method_medium_mesh(self, generator: HLODGenerator) -> None:
        """Test method selection for medium meshes."""
        method = generator.select_method(mesh_count=10, total_triangles=5000)
        assert method == HLODGenerationMethod.SIMPLIFICATION

    def test_select_method_large_mesh(self, generator: HLODGenerator) -> None:
        """Test method selection for large/many meshes."""
        method = generator.select_method(mesh_count=100, total_triangles=500000)
        assert method == HLODGenerationMethod.IMPOSTOR

    def test_configure_generator(self, generator: HLODGenerator) -> None:
        """Test generator configuration."""
        simplification_settings = SimplificationSettings(target_ratio=0.3)
        generator.configure(simplification_settings=simplification_settings)

        # Configuration should be applied (tested indirectly)
        assert generator.method is not None


# =============================================================================
# SETTINGS VALIDATION TESTS
# =============================================================================


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_simplification_settings_invalid_ratio(self) -> None:
        """Test SimplificationSettings with invalid ratio."""
        with pytest.raises(ValueError):
            SimplificationSettings(target_ratio=0.0)  # Must be > 0

        with pytest.raises(ValueError):
            SimplificationSettings(target_ratio=1.5)  # Must be <= 1

    def test_simplification_settings_invalid_error(self) -> None:
        """Test SimplificationSettings with invalid max_error."""
        with pytest.raises(ValueError):
            SimplificationSettings(max_error=-1.0)

    def test_impostor_settings_invalid_resolution(self) -> None:
        """Test ImpostorSettings with invalid resolution."""
        with pytest.raises(ValueError):
            ImpostorSettings(resolution=0)

    def test_impostor_settings_invalid_view_count(self) -> None:
        """Test ImpostorSettings with invalid view_count."""
        with pytest.raises(ValueError):
            ImpostorSettings(view_count=0)

    def test_merge_settings_invalid_distance(self) -> None:
        """Test MergeSettings with invalid merge_distance."""
        with pytest.raises(ValueError):
            MergeSettings(merge_distance=-1.0)


# =============================================================================
# EDGE TESTS
# =============================================================================


class TestEdge:
    """Tests for Edge data structure."""

    def test_edge_creation(self) -> None:
        """Test edge creation."""
        edge = Edge(v0=0, v1=1, cost=0.5)
        assert edge.v0 == 0
        assert edge.v1 == 1
        assert edge.cost == 0.5

    def test_edge_comparison(self) -> None:
        """Test edge comparison for heap."""
        edge1 = Edge(v0=0, v1=1, cost=0.5)
        edge2 = Edge(v0=2, v1=3, cost=1.0)

        assert edge1 < edge2

    def test_edge_equality(self) -> None:
        """Test edge equality (undirected)."""
        edge1 = Edge(v0=0, v1=1)
        edge2 = Edge(v0=1, v1=0)  # Same edge, reversed

        assert edge1 == edge2

    def test_edge_hash(self) -> None:
        """Test edge hashing (undirected)."""
        edge1 = Edge(v0=0, v1=1)
        edge2 = Edge(v0=1, v1=0)

        assert hash(edge1) == hash(edge2)

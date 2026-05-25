"""
Tests for Meshlet/Cluster system.

Tests meshlet building, bounds computation, and normal cone culling.
"""

import math
import pytest

from engine.rendering.gpu_driven.culling import Vec3, AABB, BoundingSphere
from engine.rendering.gpu_driven.meshlet import (
    MeshletConstants,
    MeshletBounds,
    Meshlet,
    Vertex,
    MeshletBuildConfig,
    MeshletBuilder,
    MeshletCuller,
    MeshletLODLevel,
    MeshletLODChain,
    MeshletMesh,
)


# =============================================================================
# MESHLET DATA STRUCTURE TESTS
# =============================================================================


class TestMeshletBounds:
    """Tests for MeshletBounds."""

    def test_creation(self) -> None:
        """Test MeshletBounds creation."""
        bounds = MeshletBounds(
            bounding_sphere=BoundingSphere(center=Vec3(0, 0, 0), radius=1.0),
            cone_axis=Vec3(0, 1, 0),
            cone_cutoff=0.7,
        )

        assert bounds.bounding_sphere.radius == 1.0
        assert bounds.cone_axis.y == 1.0
        assert bounds.cone_cutoff == 0.7

    def test_degenerate_cone(self) -> None:
        """Test degenerate cone detection."""
        # Degenerate cone (cutoff >= 1.0)
        bounds = MeshletBounds(cone_cutoff=1.0)
        assert bounds.is_cone_degenerate

        # Valid cone
        bounds = MeshletBounds(cone_cutoff=0.5)
        assert not bounds.is_cone_degenerate


class TestMeshlet:
    """Tests for Meshlet."""

    def test_creation(self) -> None:
        """Test Meshlet creation."""
        meshlet = Meshlet(
            meshlet_id=0,
            vertex_indices=[0, 1, 2, 3],
            local_indices=[0, 1, 2, 0, 2, 3],  # 2 triangles
        )

        assert meshlet.meshlet_id == 0
        assert meshlet.vertex_count == 4
        assert meshlet.triangle_count == 2
        assert meshlet.index_count == 6

    def test_get_triangle(self) -> None:
        """Test triangle retrieval."""
        meshlet = Meshlet(
            meshlet_id=0,
            vertex_indices=[10, 11, 12, 13],  # Global indices
            local_indices=[0, 1, 2, 0, 2, 3],  # Local indices
        )

        tri0 = meshlet.get_triangle(0)
        assert tri0 == (10, 11, 12)

        tri1 = meshlet.get_triangle(1)
        assert tri1 == (10, 12, 13)

    def test_iterate_triangles(self) -> None:
        """Test triangle iteration."""
        meshlet = Meshlet(
            meshlet_id=0,
            vertex_indices=[0, 1, 2, 3],
            local_indices=[0, 1, 2, 0, 2, 3],
        )

        triangles = list(meshlet.iterate_triangles())
        assert len(triangles) == 2
        assert triangles[0] == (0, 1, 2)
        assert triangles[1] == (0, 2, 3)


class TestVertex:
    """Tests for Vertex."""

    def test_creation(self) -> None:
        """Test Vertex creation."""
        vertex = Vertex(
            position=Vec3(1.0, 2.0, 3.0),
            normal=Vec3(0.0, 1.0, 0.0),
        )

        assert vertex.position.x == 1.0
        assert vertex.normal.y == 1.0


# =============================================================================
# MESHLET BUILDER TESTS
# =============================================================================


class TestMeshletBuilder:
    """Tests for MeshletBuilder."""

    @pytest.fixture
    def simple_quad_mesh(self) -> tuple[list[Vertex], list[int]]:
        """Create a simple quad mesh (2 triangles)."""
        vertices = [
            Vertex(position=Vec3(0, 0, 0), normal=Vec3(0, 0, 1)),
            Vertex(position=Vec3(1, 0, 0), normal=Vec3(0, 0, 1)),
            Vertex(position=Vec3(1, 1, 0), normal=Vec3(0, 0, 1)),
            Vertex(position=Vec3(0, 1, 0), normal=Vec3(0, 0, 1)),
        ]
        indices = [0, 1, 2, 0, 2, 3]
        return vertices, indices

    @pytest.fixture
    def larger_mesh(self) -> tuple[list[Vertex], list[int]]:
        """Create a larger mesh for testing meshlet splitting."""
        # Create a 10x10 grid of quads (200 triangles)
        vertices: list[Vertex] = []
        indices: list[int] = []

        for y in range(11):
            for x in range(11):
                vertices.append(Vertex(
                    position=Vec3(float(x), float(y), 0),
                    normal=Vec3(0, 0, 1),
                ))

        for y in range(10):
            for x in range(10):
                i = y * 11 + x
                # Triangle 1
                indices.extend([i, i + 1, i + 12])
                # Triangle 2
                indices.extend([i, i + 12, i + 11])

        return vertices, indices

    def test_build_simple_mesh(self, simple_quad_mesh: tuple[list[Vertex], list[int]]) -> None:
        """Test building meshlets from simple mesh."""
        vertices, indices = simple_quad_mesh
        builder = MeshletBuilder()

        meshlets = builder.build(vertices, indices)

        assert len(meshlets) >= 1
        # All triangles should be covered
        total_tris = sum(m.triangle_count for m in meshlets)
        assert total_tris == 2

    def test_build_respects_limits(self, larger_mesh: tuple[list[Vertex], list[int]]) -> None:
        """Test that builder respects vertex/triangle limits."""
        vertices, indices = larger_mesh
        config = MeshletBuildConfig(max_vertices=32, max_triangles=62)
        builder = MeshletBuilder(config)

        meshlets = builder.build(vertices, indices)

        for meshlet in meshlets:
            assert meshlet.vertex_count <= 32
            assert meshlet.triangle_count <= 62

    def test_all_triangles_covered(self, larger_mesh: tuple[list[Vertex], list[int]]) -> None:
        """Test that all triangles are assigned to meshlets."""
        vertices, indices = larger_mesh
        builder = MeshletBuilder()

        meshlets = builder.build(vertices, indices)

        expected_triangles = len(indices) // 3
        total_triangles = sum(m.triangle_count for m in meshlets)
        assert total_triangles == expected_triangles

    def test_meshlet_bounds_computed(self, simple_quad_mesh: tuple[list[Vertex], list[int]]) -> None:
        """Test that meshlet bounds are computed."""
        vertices, indices = simple_quad_mesh
        builder = MeshletBuilder()

        meshlets = builder.build(vertices, indices)

        for meshlet in meshlets:
            assert meshlet.bounds is not None
            assert meshlet.bounds.bounding_sphere.radius > 0

    def test_normal_cone_computed(self, simple_quad_mesh: tuple[list[Vertex], list[int]]) -> None:
        """Test that normal cone is computed."""
        vertices, indices = simple_quad_mesh
        builder = MeshletBuilder()

        meshlets = builder.build(vertices, indices)

        for meshlet in meshlets:
            # For a flat quad, all normals point same direction
            # Cone should be tight (high cutoff value close to 1)
            assert meshlet.bounds.cone_axis.length() > 0.9  # Normalized

    def test_invalid_indices_rejected(self) -> None:
        """Test that invalid index count is rejected."""
        builder = MeshletBuilder()

        with pytest.raises(ValueError):
            builder.build([], [0, 1])  # Not multiple of 3


# =============================================================================
# MESHLET CULLER TESTS
# =============================================================================


class TestMeshletCuller:
    """Tests for MeshletCuller."""

    def test_backface_culling_front(self) -> None:
        """Test that front-facing meshlets are not culled."""
        culler = MeshletCuller()
        culler.update(camera_position=Vec3(0, 0, 5))

        # Meshlet with normals pointing toward camera (+Z)
        meshlet = Meshlet(
            meshlet_id=0,
            bounds=MeshletBounds(
                bounding_sphere=BoundingSphere(center=Vec3(0, 0, 0), radius=1),
                cone_axis=Vec3(0, 0, 1),  # Pointing toward camera
                cone_cutoff=0.9,  # Tight cone
            ),
        )

        # Should NOT be backface culled
        assert not culler.is_backface_culled(meshlet)

    def test_backface_culling_back(self) -> None:
        """Test that back-facing meshlets are culled."""
        culler = MeshletCuller()
        culler.update(camera_position=Vec3(0, 0, 5))

        # Meshlet with normals pointing away from camera (-Z)
        meshlet = Meshlet(
            meshlet_id=0,
            bounds=MeshletBounds(
                bounding_sphere=BoundingSphere(center=Vec3(0, 0, 0), radius=1),
                cone_axis=Vec3(0, 0, -1),  # Pointing away from camera
                cone_cutoff=0.9,  # Tight cone
            ),
        )

        # Should be backface culled
        assert culler.is_backface_culled(meshlet)

    def test_degenerate_cone_not_culled(self) -> None:
        """Test that meshlets with degenerate cones are not culled."""
        culler = MeshletCuller()
        culler.update(camera_position=Vec3(0, 0, 5))

        # Degenerate cone (can't be backface culled)
        meshlet = Meshlet(
            meshlet_id=0,
            bounds=MeshletBounds(
                bounding_sphere=BoundingSphere(center=Vec3(0, 0, 0), radius=1),
                cone_axis=Vec3(0, 0, -1),
                cone_cutoff=1.0,  # Degenerate
            ),
        )

        assert not culler.is_backface_culled(meshlet)

    def test_cull_meshlets(self) -> None:
        """Test batch meshlet culling."""
        culler = MeshletCuller()
        culler.update(camera_position=Vec3(0, 0, 5))

        meshlets = [
            # Front-facing
            Meshlet(
                meshlet_id=0,
                bounds=MeshletBounds(
                    bounding_sphere=BoundingSphere(center=Vec3(0, 0, 0), radius=1),
                    cone_axis=Vec3(0, 0, 1),
                    cone_cutoff=0.9,
                ),
            ),
            # Back-facing
            Meshlet(
                meshlet_id=1,
                bounds=MeshletBounds(
                    bounding_sphere=BoundingSphere(center=Vec3(0, 0, 0), radius=1),
                    cone_axis=Vec3(0, 0, -1),
                    cone_cutoff=0.9,
                ),
            ),
        ]

        visible = culler.cull_meshlets(meshlets)

        assert 0 in visible  # Front-facing visible
        assert 1 not in visible  # Back-facing culled


# =============================================================================
# MESHLET LOD TESTS
# =============================================================================


class TestMeshletLODChain:
    """Tests for MeshletLODChain."""

    def test_lod_chain_creation(self) -> None:
        """Test LOD chain creation."""
        chain = MeshletLODChain()

        assert chain.lod_count == 0

    def test_add_lod_level(self) -> None:
        """Test adding LOD levels."""
        chain = MeshletLODChain()

        meshlets_lod0 = [Meshlet(meshlet_id=0), Meshlet(meshlet_id=1)]
        meshlets_lod1 = [Meshlet(meshlet_id=2)]

        chain.add_lod_level(meshlets_lod0, screen_size_threshold=100.0)
        chain.add_lod_level(meshlets_lod1, screen_size_threshold=50.0)

        assert chain.lod_count == 2

        lod0 = chain.get_lod_level(0)
        assert lod0 is not None
        assert lod0.meshlet_count == 2

        lod1 = chain.get_lod_level(1)
        assert lod1 is not None
        assert lod1.meshlet_count == 1

    def test_lod_selection(self) -> None:
        """Test LOD level selection by screen size."""
        chain = MeshletLODChain()

        chain.add_lod_level([], screen_size_threshold=100.0)
        chain.add_lod_level([], screen_size_threshold=50.0)
        chain.add_lod_level([], screen_size_threshold=10.0)

        assert chain.select_lod(150.0) == 0  # Largest, use LOD 0
        assert chain.select_lod(75.0) == 1  # Medium
        assert chain.select_lod(25.0) == 2  # Small
        assert chain.select_lod(5.0) == 2  # Very small, still LOD 2

    def test_get_all_meshlets(self) -> None:
        """Test getting all meshlets from chain."""
        chain = MeshletLODChain()

        meshlets_lod0 = [Meshlet(meshlet_id=0), Meshlet(meshlet_id=1)]
        meshlets_lod1 = [Meshlet(meshlet_id=2)]

        chain.add_lod_level(meshlets_lod0)
        chain.add_lod_level(meshlets_lod1)

        all_meshlets = chain.get_all_meshlets()
        assert len(all_meshlets) == 3

    def test_lod_level_assignment(self) -> None:
        """Test that LOD level is assigned to meshlets."""
        chain = MeshletLODChain()

        meshlets = [Meshlet(meshlet_id=0)]
        chain.add_lod_level(meshlets)

        assert meshlets[0].lod_level == 0


# =============================================================================
# MESHLET MESH TESTS
# =============================================================================


class TestMeshletMesh:
    """Tests for MeshletMesh."""

    def test_from_mesh_data(self) -> None:
        """Test creating MeshletMesh from raw data."""
        vertices = [
            Vertex(position=Vec3(0, 0, 0), normal=Vec3(0, 0, 1)),
            Vertex(position=Vec3(1, 0, 0), normal=Vec3(0, 0, 1)),
            Vertex(position=Vec3(0, 1, 0), normal=Vec3(0, 0, 1)),
        ]
        indices = [0, 1, 2]

        mesh = MeshletMesh.from_mesh_data(vertices, indices)

        assert mesh.vertex_count == 3
        assert mesh.triangle_count == 1
        assert mesh.meshlet_count >= 1
        assert mesh.bounds.radius > 0

    def test_mesh_properties(self) -> None:
        """Test MeshletMesh properties."""
        mesh = MeshletMesh(
            vertices=[Vertex() for _ in range(10)],
            indices=list(range(9)),  # 3 triangles
            meshlets=[Meshlet(meshlet_id=0), Meshlet(meshlet_id=1)],
        )

        assert mesh.vertex_count == 10
        assert mesh.triangle_count == 3
        assert mesh.meshlet_count == 2


# =============================================================================
# MESHLET CONSTANTS TESTS
# =============================================================================


class TestMeshletConstants:
    """Tests for MeshletConstants."""

    def test_constants(self) -> None:
        """Test meshlet constants."""
        assert MeshletConstants.MAX_VERTICES == 64
        assert MeshletConstants.MAX_TRIANGLES == 124
        assert MeshletConstants.MAX_INDICES == MeshletConstants.MAX_TRIANGLES * 3

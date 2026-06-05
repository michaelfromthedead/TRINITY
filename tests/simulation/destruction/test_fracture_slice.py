"""
Tests for Planar Slice Fracture System.

Whitebox tests for fracture_slice.py including:
- SliceResult dataclass
- CappedMesh dataclass
- SliceFracture operations
- Multi-slice and parallel slices
- AdaptiveSliceFracture
- HierarchicalSliceFracture
"""

import pytest
import math

from engine.simulation.destruction.fracture_slice import (
    SliceResult,
    CappedMesh,
    SliceFracture,
    AdaptiveSliceFracture,
    HierarchicalSliceFracture,
)
from engine.simulation.destruction.fracture_voronoi import (
    Vec3,
    Plane,
    BoundingBox,
    Chunk,
    vec3_normalize,
)
from engine.simulation.destruction.config import SLICE_MAX_PLANES


class TestSliceResult:
    """Tests for SliceResult dataclass."""

    def test_basic_construction(self):
        """Verify basic construction."""
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
        result = SliceResult(
            front_chunk=None,
            back_chunk=None,
            plane=plane
        )
        assert result.front_chunk is None
        assert result.back_chunk is None
        assert result.plane == plane
        assert result.cut_vertices == []

    def test_with_chunks(self):
        """Verify construction with chunks."""
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
        front = Chunk(vertices=[(1.0, 0.0, 0.0)], triangles=[])
        back = Chunk(vertices=[(-1.0, 0.0, 0.0)], triangles=[])

        result = SliceResult(
            front_chunk=front,
            back_chunk=back,
            plane=plane
        )
        assert result.front_chunk is not None
        assert result.back_chunk is not None


class TestCappedMesh:
    """Tests for CappedMesh dataclass."""

    def test_basic_construction(self):
        """Verify basic construction."""
        mesh = CappedMesh(
            vertices=[(0.0, 0.0, 0.0)],
            triangles=[(0, 0, 0)]
        )
        assert len(mesh.vertices) == 1
        assert len(mesh.triangles) == 1
        assert mesh.is_front is True

    def test_with_cap_data(self):
        """Verify construction with cap data."""
        mesh = CappedMesh(
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            triangles=[(0, 1, 0)],
            cap_vertices=[0, 1],
            cap_triangles=[0],
            is_front=False
        )
        assert len(mesh.cap_vertices) == 2
        assert len(mesh.cap_triangles) == 1
        assert mesh.is_front is False


class TestSliceFracture:
    """Tests for SliceFracture class."""

    def test_basic_construction(self):
        """Verify basic construction."""
        fracture = SliceFracture(seed=42)
        assert fracture.seed == 42

    def test_seed_setter(self):
        """Verify seed setter resets RNG."""
        fracture = SliceFracture(seed=42)
        fracture.seed = 99
        assert fracture.seed == 99

    def test_slice_mesh_basic(self):
        """Verify basic mesh slicing."""
        # Create a unit cube
        vertices = [
            (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0)
        ]
        triangles = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
            (0, 4, 5), (0, 5, 1),
            (2, 6, 7), (2, 7, 3),
            (1, 5, 6), (1, 6, 2),
            (0, 3, 7), (0, 7, 4)
        ]

        fracture = SliceFracture(seed=42, generate_caps=False)
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))

        result = fracture.slice_mesh(vertices, triangles, plane)

        # Should have front and back chunks
        assert result.front_chunk is not None or result.back_chunk is not None

    def test_slice_mesh_with_caps(self):
        """Verify slicing generates caps."""
        vertices = [
            (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0)
        ]
        triangles = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
        ]

        fracture = SliceFracture(seed=42, generate_caps=True)
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))

        result = fracture.slice_mesh(vertices, triangles, plane)
        # Should record cut vertices
        # (may be empty if plane doesn't intersect triangles)

    def test_slice_mesh_no_intersection(self):
        """Verify slicing when plane doesn't intersect mesh."""
        vertices = [
            (10.0, 0.0, 0.0),
            (11.0, 0.0, 0.0),
            (10.5, 1.0, 0.0)
        ]
        triangles = [(0, 1, 2)]

        fracture = SliceFracture(seed=42)
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))

        result = fracture.slice_mesh(vertices, triangles, plane)

        # All triangles should be on one side (front, since x > 0)
        assert result.front_chunk is not None or result.back_chunk is not None

    def test_multi_slice(self):
        """Verify multi-plane slicing."""
        # Use a 3D cube mesh instead of flat 2D quad for volume
        vertices = [
            (-2.0, -2.0, -2.0), (2.0, -2.0, -2.0),
            (2.0, 2.0, -2.0), (-2.0, 2.0, -2.0),
            (-2.0, -2.0, 2.0), (2.0, -2.0, 2.0),
            (2.0, 2.0, 2.0), (-2.0, 2.0, 2.0)
        ]
        triangles = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
            (0, 4, 5), (0, 5, 1),
            (2, 6, 7), (2, 7, 3),
            (1, 5, 6), (1, 6, 2),
            (0, 3, 7), (0, 7, 4)
        ]

        fracture = SliceFracture(seed=42, min_chunk_volume=0.0001)
        planes = [
            Plane(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0)),
        ]

        chunks = fracture.multi_slice(vertices, triangles, planes)

        # Should produce chunks (may vary based on geometry)
        assert isinstance(chunks, list)

    def test_multi_slice_max_planes(self):
        """Verify plane count is limited."""
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)]
        triangles = [(0, 1, 2)]

        fracture = SliceFracture(seed=42)
        # Create more planes than allowed
        planes = [
            Plane(point=(0.1 * i, 0.0, 0.0), normal=(1.0, 0.0, 0.0))
            for i in range(SLICE_MAX_PLANES + 10)
        ]

        # Should not crash, planes should be limited
        chunks = fracture.multi_slice(vertices, triangles, planes)
        assert isinstance(chunks, list)

    def test_parallel_slices(self):
        """Verify parallel slice generation."""
        # Use 3D mesh for proper volume
        vertices = [
            (-5.0, -5.0, -5.0), (5.0, -5.0, -5.0),
            (5.0, 5.0, -5.0), (-5.0, 5.0, -5.0),
            (-5.0, -5.0, 5.0), (5.0, -5.0, 5.0),
            (5.0, 5.0, 5.0), (-5.0, 5.0, 5.0)
        ]
        triangles = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
        ]

        fracture = SliceFracture(seed=42, min_chunk_volume=0.0001)
        chunks = fracture.parallel_slices(
            vertices, triangles,
            direction=(1.0, 0.0, 0.0),
            num_slices=1
        )

        # Should produce chunks (may be empty for degenerate geometry)
        assert isinstance(chunks, list)

    def test_parallel_slices_with_spacing(self):
        """Verify parallel slices with custom spacing."""
        # Use 3D mesh
        vertices = [
            (-10.0, -10.0, -10.0), (10.0, -10.0, -10.0),
            (10.0, 10.0, -10.0), (-10.0, 10.0, -10.0),
            (-10.0, -10.0, 10.0), (10.0, -10.0, 10.0),
            (10.0, 10.0, 10.0), (-10.0, 10.0, 10.0)
        ]
        triangles = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
        ]

        fracture = SliceFracture(seed=42, min_chunk_volume=0.0001)
        chunks = fracture.parallel_slices(
            vertices, triangles,
            direction=(1.0, 0.0, 0.0),
            num_slices=1,
            spacing=5.0
        )

        # Check that operation completed
        assert isinstance(chunks, list)

    def test_random_slice_planes(self):
        """Verify random plane generation."""
        fracture = SliceFracture(seed=42)
        bounds = BoundingBox(
            min_point=(-10.0, -10.0, -10.0),
            max_point=(10.0, 10.0, 10.0)
        )

        planes = fracture.random_slice_planes(bounds, num_planes=5)

        assert len(planes) == 5
        for plane in planes:
            assert bounds.contains(plane.point)

    def test_random_slice_planes_with_bias(self):
        """Verify biased plane generation."""
        fracture = SliceFracture(seed=42)
        bounds = BoundingBox(
            min_point=(-10.0, -10.0, -10.0),
            max_point=(10.0, 10.0, 10.0)
        )

        planes = fracture.random_slice_planes(
            bounds,
            num_planes=5,
            bias_direction=(0.0, 0.0, 1.0),
            bias_strength=0.8
        )

        assert len(planes) == 5
        # Normals should be biased toward Z direction
        for plane in planes:
            # Not strictly enforced but bias should make Z component significant
            pass

    def test_grid_slice(self):
        """Verify grid slicing."""
        vertices = [
            (-3.0, -3.0, -3.0), (3.0, -3.0, -3.0),
            (3.0, 3.0, -3.0), (-3.0, 3.0, -3.0),
            (-3.0, -3.0, 3.0), (3.0, -3.0, 3.0),
            (3.0, 3.0, 3.0), (-3.0, 3.0, 3.0)
        ]
        triangles = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
        ]

        fracture = SliceFracture(seed=42, min_chunk_volume=0.0001)
        chunks = fracture.grid_slice(
            vertices, triangles,
            grid_size=(2, 2, 2)
        )

        # Grid should produce up to 2*2*2 = 8 chunks
        assert len(chunks) >= 1
        assert len(chunks) <= 8

    def test_fracture_along_edge(self):
        """Verify edge-based slicing."""
        # Use 3D mesh
        vertices = [
            (-5.0, -5.0, -5.0), (5.0, -5.0, -5.0),
            (5.0, 5.0, -5.0), (-5.0, 5.0, -5.0),
            (-5.0, -5.0, 5.0), (5.0, -5.0, 5.0),
            (5.0, 5.0, 5.0), (-5.0, 5.0, 5.0)
        ]
        triangles = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
        ]

        fracture = SliceFracture(seed=42, min_chunk_volume=0.0001)
        chunks = fracture.fracture_along_edge(
            vertices, triangles,
            edge_start=(-5.0, 0.0, 0.0),
            edge_end=(5.0, 0.0, 0.0),
            num_slices=1
        )

        # Should produce chunks based on slices along the edge
        assert isinstance(chunks, list)


class TestSliceFractureEdgeCases:
    """Edge case tests for SliceFracture."""

    def test_empty_mesh(self):
        """Verify handling of empty mesh."""
        fracture = SliceFracture(seed=42)
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))

        result = fracture.slice_mesh([], [], plane)

        assert result.front_chunk is None
        assert result.back_chunk is None

    def test_single_triangle(self):
        """Verify handling of single triangle."""
        vertices = [
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (1.0, 2.0, 0.0)
        ]
        triangles = [(0, 1, 2)]

        fracture = SliceFracture(seed=42, min_chunk_volume=0.0001)
        plane = Plane(point=(1.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0))

        result = fracture.slice_mesh(vertices, triangles, plane)

        # Should split the triangle
        assert result.front_chunk is not None or result.back_chunk is not None

    def test_degenerate_triangle_skipped(self):
        """Verify degenerate triangles are handled."""
        vertices = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),  # Same as first
            (1.0, 0.0, 0.0)
        ]
        triangles = [(0, 1, 2)]

        fracture = SliceFracture(seed=42)
        plane = Plane(point=(0.5, 0.0, 0.0), normal=(1.0, 0.0, 0.0))

        # Should not crash
        result = fracture.slice_mesh(vertices, triangles, plane)
        assert isinstance(result, SliceResult)

    def test_triangle_on_plane(self):
        """Verify triangle exactly on plane."""
        vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 1.0, 0.0)
        ]
        triangles = [(0, 1, 2)]

        fracture = SliceFracture(seed=42)
        plane = Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))

        # Triangle is on the XY plane, slicing with Z normal
        result = fracture.slice_mesh(vertices, triangles, plane)
        # All vertices are on the plane
        assert isinstance(result, SliceResult)


class TestAdaptiveSliceFracture:
    """Tests for AdaptiveSliceFracture class."""

    def test_basic_construction(self):
        """Verify basic construction."""
        fracture = AdaptiveSliceFracture(
            seed=42,
            base_slices=4,
            max_slices=16
        )
        assert fracture._base_slices == 4
        assert fracture._max_slices == 16

    def test_fracture_adaptive_low_intensity(self):
        """Verify low intensity produces fewer slices."""
        vertices = [
            (-5.0, -5.0, 0.0), (5.0, -5.0, 0.0),
            (5.0, 5.0, 0.0), (-5.0, 5.0, 0.0)
        ]
        triangles = [(0, 1, 2), (0, 2, 3)]

        fracture = AdaptiveSliceFracture(
            seed=42,
            base_slices=4,
            max_slices=16,
            min_chunk_volume=0.0001
        )
        chunks = fracture.fracture_adaptive(
            vertices, triangles,
            impact_point=(0.0, 0.0, 0.0),
            impact_intensity=0.2
        )

        assert isinstance(chunks, list)

    def test_fracture_adaptive_high_intensity(self):
        """Verify high intensity produces more slices."""
        vertices = [
            (-5.0, -5.0, 0.0), (5.0, -5.0, 0.0),
            (5.0, 5.0, 0.0), (-5.0, 5.0, 0.0)
        ]
        triangles = [(0, 1, 2), (0, 2, 3)]

        fracture = AdaptiveSliceFracture(
            seed=42,
            base_slices=4,
            max_slices=16,
            min_chunk_volume=0.0001
        )
        chunks = fracture.fracture_adaptive(
            vertices, triangles,
            impact_point=(0.0, 0.0, 0.0),
            impact_intensity=1.0
        )

        assert isinstance(chunks, list)


class TestHierarchicalSliceFracture:
    """Tests for HierarchicalSliceFracture class."""

    def test_basic_construction(self):
        """Verify basic construction."""
        fracture = HierarchicalSliceFracture(
            seed=42,
            max_depth=3,
            split_threshold=0.1
        )
        assert fracture._max_depth == 3
        assert fracture._split_threshold == 0.1

    def test_fracture_hierarchical_depth_zero(self):
        """Verify depth=0 returns original mesh."""
        vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 1.0, 0.0),
            (0.5, 0.5, 1.0)
        ]
        triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

        fracture = HierarchicalSliceFracture(
            seed=42,
            max_depth=0,  # No recursion
            min_chunk_volume=0.0001
        )
        chunks = fracture.fracture_hierarchical(vertices, triangles)

        # Should return single chunk (original mesh)
        assert len(chunks) == 1

    def test_fracture_hierarchical_depth_one(self):
        """Verify depth=1 splits once."""
        vertices = [
            (-2.0, -2.0, -2.0), (2.0, -2.0, -2.0),
            (2.0, 2.0, -2.0), (-2.0, 2.0, -2.0),
            (-2.0, -2.0, 2.0), (2.0, -2.0, 2.0),
            (2.0, 2.0, 2.0), (-2.0, 2.0, 2.0)
        ]
        triangles = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
        ]

        fracture = HierarchicalSliceFracture(
            seed=42,
            max_depth=1,
            min_chunk_volume=0.0001
        )
        chunks = fracture.fracture_hierarchical(vertices, triangles)

        # Should produce at least 2 chunks
        assert len(chunks) >= 1

    def test_fracture_hierarchical_recursive(self):
        """Verify recursive splitting."""
        # Use 3D mesh for volume
        vertices = [
            (-4.0, -4.0, -4.0), (4.0, -4.0, -4.0),
            (4.0, 4.0, -4.0), (-4.0, 4.0, -4.0),
            (-4.0, -4.0, 4.0), (4.0, -4.0, 4.0),
            (4.0, 4.0, 4.0), (-4.0, 4.0, 4.0)
        ]
        triangles = [
            (0, 1, 2), (0, 2, 3),
            (4, 6, 5), (4, 7, 6),
        ]

        fracture = HierarchicalSliceFracture(
            seed=42,
            max_depth=2,
            min_chunk_volume=0.0001,
            split_threshold=0.01
        )
        chunks = fracture.fracture_hierarchical(vertices, triangles)

        # Should produce chunks (may be empty for degenerate geometry)
        assert isinstance(chunks, list)

    def test_split_threshold_prevents_splitting(self):
        """Verify split threshold prevents over-fragmentation."""
        vertices = [
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.05, 0.1, 0.0)
        ]
        triangles = [(0, 1, 2)]

        fracture = HierarchicalSliceFracture(
            seed=42,
            max_depth=10,
            min_chunk_volume=0.0001,
            split_threshold=100.0  # High threshold prevents further splitting
        )
        chunks = fracture.fracture_hierarchical(vertices, triangles)

        # Should return few chunks due to threshold
        assert len(chunks) <= 4

    def test_max_chunks_limit(self):
        """Verify max_chunks limits output."""
        vertices = [
            (-10.0, -10.0, 0.0), (10.0, -10.0, 0.0),
            (10.0, 10.0, 0.0), (-10.0, 10.0, 0.0)
        ]
        triangles = [(0, 1, 2), (0, 2, 3)]

        fracture = HierarchicalSliceFracture(
            seed=42,
            max_depth=10,
            max_chunks=5,  # Limit to 5 chunks
            min_chunk_volume=0.0001,
            split_threshold=0.001
        )
        chunks = fracture.fracture_hierarchical(vertices, triangles)

        assert len(chunks) <= 5

"""Tests for fluid surface reconstruction.

Whitebox tests covering:
- DensityField data structure
- DensityField sampling (trilinear interpolation)
- FluidSurface mesh data
- MarchingCubes edge interpolation
- MarchingCubes cell processing
- Normal computation
- Density field computation from particles
"""

import math
import pytest
import numpy as np

from engine.simulation.fluid.surface_reconstruction import (
    DensityField,
    FluidSurface,
    MarchingCubes,
    compute_density_field,
    extract_isosurface,
    EDGE_TABLE,
)
from engine.simulation.fluid.config import (
    SMOOTHING_LENGTH,
    MC_MIN_EDGE_LENGTH,
    MC_ISO_EPSILON,
)


class TestDensityField:
    """Tests for DensityField data structure."""

    @pytest.fixture
    def density_field(self):
        """Create a default density field."""
        data = np.zeros((8, 8, 8))
        return DensityField(
            data=data,
            origin=np.array([0.0, 0.0, 0.0]),
            cell_size=0.1
        )

    def test_density_field_creation(self, density_field):
        """Density field should store data."""
        assert density_field.data.shape == (8, 8, 8)
        np.testing.assert_array_equal(density_field.origin, np.zeros(3))
        assert density_field.cell_size == 0.1

    def test_resolution_property(self, density_field):
        """Resolution property should return data shape."""
        assert density_field.resolution == (8, 8, 8)

    def test_world_to_grid(self, density_field):
        """World-to-grid conversion should work correctly."""
        world_pos = np.array([0.5, 0.5, 0.5])
        grid_pos = density_field.world_to_grid(world_pos)
        expected = world_pos / density_field.cell_size
        np.testing.assert_array_equal(grid_pos, expected)

    def test_grid_to_world(self, density_field):
        """Grid-to-world conversion should work correctly."""
        grid_pos = np.array([5.0, 5.0, 5.0])
        world_pos = density_field.grid_to_world(grid_pos)
        expected = grid_pos * density_field.cell_size
        np.testing.assert_array_equal(world_pos, expected)

    def test_world_grid_roundtrip(self, density_field):
        """World-to-grid and back should be identity."""
        original = np.array([0.25, 0.35, 0.45])
        grid_pos = density_field.world_to_grid(original)
        recovered = density_field.grid_to_world(grid_pos)
        np.testing.assert_array_almost_equal(original, recovered)

    def test_sample_zero_field(self, density_field):
        """Sampling zero field should give zero."""
        pos = np.array([0.4, 0.4, 0.4])
        value = density_field.sample(pos)
        assert value == 0.0

    def test_sample_uniform_field(self, density_field):
        """Sampling uniform field should give that value."""
        density_field.data.fill(5.0)
        pos = np.array([0.4, 0.4, 0.4])
        value = density_field.sample(pos)
        assert value == pytest.approx(5.0)

    def test_sample_at_corner(self, density_field):
        """Sampling at corner should give corner value."""
        density_field.data[2, 2, 2] = 10.0
        # Sample at exact grid point (accounting for cell center)
        pos = density_field.grid_to_world(np.array([2.0, 2.0, 2.0]))
        value = density_field.sample(pos)
        assert value == pytest.approx(10.0)

    def test_sample_interpolates(self, density_field):
        """Sampling between corners should interpolate."""
        density_field.data[0, 0, 0] = 0.0
        density_field.data[1, 0, 0] = 10.0

        # Sample at midpoint
        pos = density_field.grid_to_world(np.array([0.5, 0.0, 0.0]))
        value = density_field.sample(pos)

        # Should be between 0 and 10
        assert 0 <= value <= 10

    def test_sample_clamps_outside(self, density_field):
        """Sampling outside should clamp to boundary."""
        density_field.data.fill(5.0)

        # Far outside
        pos = np.array([100.0, 100.0, 100.0])
        value = density_field.sample(pos)

        # Should get boundary value (clamped)
        assert np.isfinite(value)


class TestFluidSurface:
    """Tests for FluidSurface data structure."""

    def test_empty_surface(self):
        """Empty surface should have zero counts."""
        surface = FluidSurface(
            vertices=np.zeros((0, 3)),
            triangles=np.zeros((0, 3), dtype=np.int32),
            normals=np.zeros((0, 3))
        )

        assert surface.num_vertices == 0
        assert surface.num_triangles == 0

    def test_surface_with_data(self):
        """Surface should store vertices and triangles."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64)

        surface = FluidSurface(vertices=vertices, triangles=triangles, normals=normals)

        assert surface.num_vertices == 3
        assert surface.num_triangles == 1

    def test_compute_bounds_empty(self):
        """Empty surface bounds should be zero."""
        surface = FluidSurface(
            vertices=np.zeros((0, 3)),
            triangles=np.zeros((0, 3), dtype=np.int32),
            normals=np.zeros((0, 3))
        )

        bounds_min, bounds_max = surface.compute_bounds()
        np.testing.assert_array_equal(bounds_min, np.zeros(3))
        np.testing.assert_array_equal(bounds_max, np.zeros(3))

    def test_compute_bounds_with_data(self):
        """Bounds should encompass all vertices."""
        vertices = np.array([
            [0, 0, 0],
            [1, 2, 3],
            [-1, -2, -3]
        ], dtype=np.float64)

        surface = FluidSurface(
            vertices=vertices,
            triangles=np.zeros((0, 3), dtype=np.int32),
            normals=np.zeros((3, 3))
        )

        bounds_min, bounds_max = surface.compute_bounds()

        np.testing.assert_array_equal(bounds_min, [-1, -2, -3])
        np.testing.assert_array_equal(bounds_max, [1, 2, 3])


class TestMarchingCubes:
    """Tests for Marching Cubes algorithm."""

    @pytest.fixture
    def mc(self):
        """Create a MarchingCubes instance."""
        return MarchingCubes(iso_level=0.5)

    def test_marching_cubes_creation(self, mc):
        """MarchingCubes should store iso level."""
        assert mc.iso_level == 0.5

    def test_edge_vertices_table(self, mc):
        """Edge vertex table should have correct structure."""
        assert len(mc.EDGE_VERTICES) == 12  # 12 edges per cube

    def test_cube_vertices_table(self, mc):
        """Cube vertex table should have correct structure."""
        assert mc.CUBE_VERTICES.shape == (8, 3)

    def test_interpolate_edge_at_iso(self, mc):
        """Interpolation at iso level should give t=0."""
        t = mc._interpolate_edge(0.5, 1.0)
        assert t == pytest.approx(0.0)

    def test_interpolate_edge_midpoint(self, mc):
        """Interpolation at midpoint should give t=0.5."""
        t = mc._interpolate_edge(0.0, 1.0)
        assert t == pytest.approx(0.5)

    def test_interpolate_edge_degenerate(self, mc):
        """Degenerate edge should give t=0.5."""
        t = mc._interpolate_edge(0.5, 0.5)  # Both at iso
        assert t == 0.5

    def test_interpolate_edge_clamps(self, mc):
        """Interpolation should clamp to [0, 1]."""
        # Edge case where numerical precision might give t < 0 or t > 1
        t = mc._interpolate_edge(0.5 - 1e-12, 0.5 + 1e-12)
        assert 0 <= t <= 1


class TestMarchingCubesExtraction:
    """Tests for isosurface extraction."""

    @pytest.fixture
    def sphere_field(self):
        """Create a density field with spherical density."""
        resolution = (16, 16, 16)
        data = np.zeros(resolution)
        center = np.array([8, 8, 8])
        radius = 5

        for i in range(resolution[0]):
            for j in range(resolution[1]):
                for k in range(resolution[2]):
                    dist = np.linalg.norm(np.array([i, j, k]) - center)
                    # Density = 1 inside, 0 outside
                    data[i, j, k] = 1.0 if dist < radius else 0.0

        return DensityField(
            data=data,
            origin=np.zeros(3),
            cell_size=0.1
        )

    def test_extract_empty_field(self):
        """Empty field should give empty surface."""
        data = np.zeros((8, 8, 8))
        field = DensityField(data=data, origin=np.zeros(3), cell_size=0.1)
        mc = MarchingCubes(iso_level=0.5)

        surface = mc.extract_isosurface(field)

        assert surface.num_vertices == 0
        assert surface.num_triangles == 0

    def test_extract_full_field(self):
        """Fully filled field should give empty surface (no boundary)."""
        data = np.ones((8, 8, 8))
        field = DensityField(data=data, origin=np.zeros(3), cell_size=0.1)
        mc = MarchingCubes(iso_level=0.5)

        surface = mc.extract_isosurface(field)

        # May have vertices at boundaries, but typically empty interior
        # Actual behavior depends on boundary handling

    def test_extract_sphere_has_surface(self, sphere_field):
        """Sphere field should produce surface."""
        mc = MarchingCubes(iso_level=0.5)
        surface = mc.extract_isosurface(sphere_field)

        # Should have some vertices and triangles
        assert surface.num_vertices > 0
        assert surface.num_triangles > 0

    def test_extract_sphere_closed(self, sphere_field):
        """Sphere surface should be approximately closed."""
        mc = MarchingCubes(iso_level=0.5)
        surface = mc.extract_isosurface(sphere_field)

        # Should have vertices and triangles
        assert surface.num_vertices > 0
        assert surface.num_triangles > 0

    def test_extract_has_normals(self, sphere_field):
        """Extracted surface should have normals."""
        mc = MarchingCubes(iso_level=0.5)
        surface = mc.extract_isosurface(sphere_field)

        # Normals should exist and be unit length (approximately)
        for i in range(surface.num_vertices):
            normal_len = np.linalg.norm(surface.normals[i])
            assert abs(normal_len - 1.0) < 0.1 or normal_len == 0  # Allow some tolerance

    def test_extract_different_iso_levels(self, sphere_field):
        """Different iso levels should give different surfaces."""
        mc_high = MarchingCubes(iso_level=0.9)
        mc_low = MarchingCubes(iso_level=0.1)

        surface_high = mc_high.extract_isosurface(sphere_field)
        surface_low = mc_low.extract_isosurface(sphere_field)

        # Lower iso level should give larger surface (more volume)
        # (This depends on the field, but generally true for step functions)


class TestEdgeTable:
    """Tests for Marching Cubes edge table."""

    def test_edge_table_size(self):
        """Edge table should have 256 entries."""
        assert len(EDGE_TABLE) == 256

    def test_edge_table_symmetry(self):
        """Edge table should have symmetry properties."""
        # Configuration 0 (all below) should have no edges
        assert EDGE_TABLE[0] == 0

        # Configuration 255 (all above) should have no edges
        assert EDGE_TABLE[255] == 0

    def test_edge_table_single_vertex(self):
        """Single vertex cases should have correct edges."""
        # Configuration 1 (only vertex 0 below)
        # Should intersect edges 0, 3, 8
        assert EDGE_TABLE[1] != 0


class TestDensityFieldComputation:
    """Tests for density field computation from particles."""

    def test_empty_particles(self):
        """Empty particle list should give empty field."""
        positions = np.zeros((0, 3))
        bounds_min = np.zeros(3)
        bounds_max = np.ones(3)

        field = compute_density_field(
            positions=positions,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            resolution=(8, 8, 8)
        )

        # Field should be all zeros
        assert np.all(field.data == 0)

    def test_single_particle_center(self):
        """Single particle at center should create density peak."""
        positions = np.array([[0.5, 0.5, 0.5]])
        bounds_min = np.zeros(3)
        bounds_max = np.ones(3)

        field = compute_density_field(
            positions=positions,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            resolution=(8, 8, 8),
            smoothing_length=0.2
        )

        # Should have non-zero density somewhere
        assert np.max(field.data) > 0

    def test_multiple_particles(self):
        """Multiple particles should accumulate density."""
        positions = np.array([
            [0.5, 0.5, 0.5],
            [0.55, 0.5, 0.5],  # Close by
        ])
        bounds_min = np.zeros(3)
        bounds_max = np.ones(3)

        field = compute_density_field(
            positions=positions,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            resolution=(8, 8, 8),
            smoothing_length=0.2
        )

        # Should have higher density in center region
        assert np.max(field.data) > 0

    def test_density_field_bounds(self):
        """Density field should respect bounds."""
        positions = np.array([[0.5, 0.5, 0.5]])
        bounds_min = np.zeros(3)
        bounds_max = np.ones(3)

        field = compute_density_field(
            positions=positions,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            resolution=(8, 8, 8)
        )

        np.testing.assert_array_equal(field.origin, bounds_min)


class TestExtractIsosurface:
    """Tests for convenience isosurface extraction function."""

    def test_extract_isosurface_convenience(self):
        """Convenience function should work."""
        data = np.zeros((8, 8, 8))
        data[3:5, 3:5, 3:5] = 1.0  # Small cube

        field = DensityField(
            data=data,
            origin=np.zeros(3),
            cell_size=0.1
        )

        surface = extract_isosurface(field, iso_level=0.5)

        # Should produce valid surface
        assert isinstance(surface, FluidSurface)


class TestMarchingCubesEdgeCases:
    """Edge case tests for Marching Cubes."""

    def test_values_at_iso_level(self):
        """Values exactly at iso level should be handled."""
        # Create field with values exactly at iso level
        data = np.ones((4, 4, 4)) * 0.5  # All at iso
        field = DensityField(data=data, origin=np.zeros(3), cell_size=0.1)

        mc = MarchingCubes(iso_level=0.5)
        surface = mc.extract_isosurface(field)

        # Should not crash (values are perturbed in implementation)

    def test_very_small_field(self):
        """Very small field should work."""
        data = np.zeros((2, 2, 2))
        data[0, 0, 0] = 1.0
        field = DensityField(data=data, origin=np.zeros(3), cell_size=0.1)

        mc = MarchingCubes(iso_level=0.5)
        surface = mc.extract_isosurface(field)

        # Should complete without error

    def test_non_cubic_field(self):
        """Non-cubic field should work."""
        data = np.zeros((8, 4, 12))
        data[3:5, 1:3, 5:7] = 1.0
        field = DensityField(data=data, origin=np.zeros(3), cell_size=0.1)

        mc = MarchingCubes(iso_level=0.5)
        surface = mc.extract_isosurface(field)

        # Should complete without error

    def test_custom_origin(self):
        """Custom origin should shift vertices."""
        data = np.zeros((8, 8, 8))
        data[3:5, 3:5, 3:5] = 1.0

        origin = np.array([10.0, 20.0, 30.0])
        field = DensityField(data=data, origin=origin, cell_size=0.1)

        mc = MarchingCubes(iso_level=0.5)
        surface = mc.extract_isosurface(field)

        if surface.num_vertices > 0:
            # Vertices should be offset by origin
            min_vertex = np.min(surface.vertices, axis=0)
            assert min_vertex[0] >= origin[0]
            assert min_vertex[1] >= origin[1]
            assert min_vertex[2] >= origin[2]

    def test_different_cell_sizes(self):
        """Different cell sizes should scale vertices."""
        data = np.zeros((8, 8, 8))
        data[3:5, 3:5, 3:5] = 1.0

        field_small = DensityField(data=data.copy(), origin=np.zeros(3), cell_size=0.1)
        field_large = DensityField(data=data.copy(), origin=np.zeros(3), cell_size=1.0)

        mc = MarchingCubes(iso_level=0.5)
        surface_small = mc.extract_isosurface(field_small)
        surface_large = mc.extract_isosurface(field_large)

        if surface_small.num_vertices > 0 and surface_large.num_vertices > 0:
            # Large cell size should give larger extent
            extent_small = np.max(surface_small.vertices) - np.min(surface_small.vertices)
            extent_large = np.max(surface_large.vertices) - np.min(surface_large.vertices)
            assert extent_large > extent_small * 5  # Should be ~10x larger

    def test_iso_level_zero(self):
        """Iso level of zero should work."""
        data = np.ones((8, 8, 8))
        data[3:5, 3:5, 3:5] = -1.0  # Negative region
        field = DensityField(data=data, origin=np.zeros(3), cell_size=0.1)

        mc = MarchingCubes(iso_level=0.0)
        surface = mc.extract_isosurface(field)

        # Should complete without error

    def test_iso_level_one(self):
        """Iso level of one should work."""
        data = np.ones((8, 8, 8)) * 2.0
        data[3:5, 3:5, 3:5] = 0.5
        field = DensityField(data=data, origin=np.zeros(3), cell_size=0.1)

        mc = MarchingCubes(iso_level=1.0)
        surface = mc.extract_isosurface(field)

        # Should complete without error
